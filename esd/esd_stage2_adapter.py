#!/usr/bin/env python3
"""Tracked ESD-only WhisperX 3.3.1 alignment and frozen acoustic adapter.

Adapted from repository candidates with provenance locked in PROVENANCE. It has
no Seamless or LIWC dependency. Smoke never fits reliability or Gate E.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,re,tempfile
from pathlib import Path
from typing import Any
from esd_contract import *
from frozen_word_aligner import (FrozenWordAligner,ALIGN_MODEL,
    ALIGN_CHECKPOINT_BASENAME,ALIGN_CHECKPOINT_SHA256,
    HISTORICAL_ALIGNMENT_SHA256,verify_checkpoint)

PROVENANCE={
 "alignment_normalization":{"path":"scripts/align_words_within_segments.py","sha256":"b6b08ff87f81b743e46fc0d54867b375f15acaa3320001f9d45fc5dff9ddf099"},
 "historical_whisperx_alignment":{"package":"whisperx 3.3.1","path":"whisperx/alignment.py","sha256":HISTORICAL_ALIGNMENT_SHA256,"functions":["load_align_model","align","get_trellis","backtrack","merge_repeats"]},
 "word_acoustics":{"path":"scripts/extract_word_local_prosody.py","sha256":"3d84c495bfb2b0cae6f9230183e79e4076280735996a6f3d9f1b59ed1ecd8dc7"},
 "audio_frames":{"path":"scripts/extract_raw_segment_prosody.py","sha256":"b997401dd15c718e06991dba106c603423c866d609bd132874ffa64016c10507"},
 "cmudict_duration":{"path":"scripts/rebuild_phase5ar1_dry_run.py","sha256":"c2cdba3a9f096b82495f2936b7b6f259585b754ec80e4684f11de5356fb98553"}}
CMUDICT_SHA256="81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
CMUDICT_REVISION="74790861f652b15e4ac49015a90074ad62a27690"; CMUDICT_ENTRIES=135166
OUTER_PUNCT=re.compile(r"(^[^\w']+|[^\w']+$)",re.UNICODE)
FIELDS=["sample_id","speaker_id","condition","prompt_id","utterance_id","word_index","lexical_occurrence_index",
 "source_word","normalized_word","alignment_status","alignment_error_code","alignment_confidence","word_start_sec","word_end_sec",
 "word_duration_sec","log_word_duration","f0_mean_hz","f0_p10_hz","f0_p90_hz","f0_robust_range_hz","energy_db",
 "authoritative_syllable_count","syllable_lookup_status","feature_status","feature_error_code","adapter_version"]
ADAPTER_VERSION="esd_stage2_adapter_local_ctc_v2"

def file_sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def normalize_text(text:str):
    output=[]; occurrences={}
    for position,source in enumerate(text.split()):
        normalized=source.replace("’","'").lower(); normalized=OUTER_PUNCT.sub("",normalized); normalized=re.sub(r"[^\w'-]","",normalized)
        if not normalized: continue
        occurrence=occurrences.get(normalized,0); occurrences[normalized]=occurrence+1
        output.append({"source_word":source,"normalized_word":normalized,"word_index":position,"lexical_occurrence_index":occurrence})
    return output

def verify_cmudict(path:Path):
    if not path.is_file() or file_sha(path)!=CMUDICT_SHA256: raise ContractError("frozen CMUdict checksum mismatch")
    dictionary={}; entries=0
    with path.open(encoding="latin-1") as f:
        for line in f:
            if not line.strip() or line.startswith(";;;"): continue
            word,phones=line.split(maxsplit=1); base=re.sub(r"\(\d+\)$","",word).lower()
            count=sum(bool(re.search(r"[012]$",phone)) for phone in phones.split()); dictionary.setdefault(base,[]).append(count); entries+=1
    if entries!=CMUDICT_ENTRIES: raise ContractError("frozen CMUdict pronunciation count mismatch")
    return dictionary

def verify_alignment_checkpoint(model_dir:Path):
    try: return verify_checkpoint(model_dir)
    except Exception as exc: raise ContractError(str(exc)) from exc

def read_manifest(path:Path):
    if file_sha(path)!=APPROVED_STAGE1_MANIFEST_SHA256: raise ContractError("adapter refused non-approved manifest")
    with path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    required={"sample_id","speaker_id","condition","prompt_id","utterance_id","audio_path","transcript_text"}
    if not rows or not required<=set(rows[0]): raise ContractError("invalid ESD manifest schema")
    if len(rows)!=15750 or tuple(sorted({r["speaker_id"] for r in rows}))!=ELIGIBLE_ESD_SPEAKERS: raise ContractError("invalid ESD manifest scope")
    if any(r["speaker_id"]=="0014" for r in rows): raise ContractError("excluded speaker 0014 entered adapter")
    return rows

def select_smoke(rows,count=10):
    if count!=10: raise ContractError("Stage 2A-SMOKE scope is exactly 10 utterances")
    lookup={(r["speaker_id"],r["condition"],int(r["prompt_id"])):r for r in rows}
    conditions=("Neutral","Angry","Happy","Sad","Surprise"); selected=[]
    for i in range(10):
        key=(ELIGIBLE_ESD_SPEAKERS[i%len(ELIGIBLE_ESD_SPEAKERS)],conditions[i%5],1+(i*37)%350)
        if key not in lookup: raise ContractError("deterministic smoke key missing from manifest")
        selected.append(lookup[key])
    if len({r["sample_id"] for r in selected})!=10: raise ContractError("duplicate deterministic smoke selection")
    return selected

def atomic_csv(path:Path,rows):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=path.name+".",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,FIELDS,extrasaction="ignore",lineterminator="\n"); w.writeheader(); w.writerows(rows)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def frames(signal,size,hop):
    import numpy as np
    if not len(signal): return []
    if len(signal)<=size: return [np.pad(signal,(0,size-len(signal)))]
    return [signal[i:i+size] for i in range(0,len(signal)-size+1,hop)]

def analyze_word(signal,sr):
    import numpy as np
    size=round(.040*sr); hop=round(.010*sr); chunks=frames(signal,size,hop); window=np.hanning(size); pitches=[]
    min_lag=max(1,int(sr/500.0)); max_lag=min(size-2,int(sr/60.0))
    for chunk in chunks:
        rms=float(np.sqrt(np.mean(chunk.astype(np.float64)**2)))
        if rms<.0001: continue
        x=(chunk-chunk.mean())*window; nfft=1<<(2*size-1).bit_length(); spectrum=np.fft.rfft(x,nfft); ac=np.fft.irfft(spectrum*np.conj(spectrum),nfft)[:size]
        if ac[0]<=0: continue
        search=ac[min_lag:max_lag+1]/ac[0]; best=int(np.argmax(search))
        if search[best]>=.30: pitches.append(sr/float(best+min_lag))
    p=np.asarray(pitches,float); rms=float(np.sqrt(np.mean(signal.astype(np.float64)**2))) if len(signal) else np.nan
    lo=float(np.percentile(p,10)) if len(p) else None; hi=float(np.percentile(p,90)) if len(p) else None
    return {"f0_mean_hz":float(np.mean(p)) if len(p) else None,"f0_p10_hz":lo,"f0_p90_hz":hi,
            "f0_robust_range_hz":hi-lo if lo is not None else None,"energy_db":max(-100.0,20*np.log10(max(rms,1e-5))) if np.isfinite(rms) else None}

def read_audio_interval(path,start,end):
    import numpy as np, soundfile as sf
    with sf.SoundFile(path) as audio:
        if start<0 or end<=start or end>len(audio)/audio.samplerate+1e-6: raise ContractError("invalid word bounds")
        audio.seek(round(start*audio.samplerate)); signal=audio.read(round((end-start)*audio.samplerate),dtype="float32",always_2d=True); sr=audio.samplerate
    mono=signal.mean(axis=1)
    if sr!=16000 and len(mono):
        n=max(1,round(len(mono)*16000/sr)); mono=np.interp(np.linspace(0,len(mono)-1,n),np.arange(len(mono)),mono).astype("float32"); sr=16000
    return mono,sr

def process_utterance(item,aligner,cmu):
    tokens=normalize_text(item["transcript_text"]); rows=[]
    try: aligned=aligner.align(Path(item["audio_path"]),tokens)
    except Exception as exc:
        return [{**item,**token,"alignment_status":"failed","alignment_error_code":type(exc).__name__,"feature_status":"not_attempted","feature_error_code":"ALIGNMENT_FAILED","adapter_version":ADAPTER_VERSION} for token in tokens]
    for i,token in enumerate(tokens):
        base={k:item[k] for k in ("sample_id","speaker_id","condition","prompt_id","utterance_id")}; base.update(token); candidate=aligned[i] if i<len(aligned) else {}
        start=candidate.get("start"); end=candidate.get("end"); base.update({"alignment_confidence":candidate.get("score","")})
        if start is None or end is None or float(start)<0 or float(end)<=float(start):
            base.update(alignment_status="unaligned",alignment_error_code="INVALID_OR_MISSING_BOUNDS",feature_status="not_attempted",feature_error_code="ALIGNMENT_FAILED",adapter_version=ADAPTER_VERSION); rows.append(base); continue
        base.update(alignment_status="aligned",alignment_error_code="",word_start_sec=float(start),word_end_sec=float(end),word_duration_sec=float(end)-float(start),log_word_duration=math.log(float(end)-float(start)))
        counts=cmu.get(token["normalized_word"]); base["authoritative_syllable_count"]=counts[0] if counts else ""; base["syllable_lookup_status"]="first_canonical" if counts else "missing"
        try:
            signal,sr=read_audio_interval(Path(item["audio_path"]),float(start),float(end)); base.update(analyze_word(signal,sr)); base.update(feature_status="complete",feature_error_code="")
        except Exception as exc: base.update(feature_status="failed",feature_error_code=type(exc).__name__)
        base["adapter_version"]=ADAPTER_VERSION; rows.append(base)
    return rows

def valid_unit(path,sample_id):
    if not path.is_file(): return False
    with path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    return bool(rows) and set(FIELDS)<=set(rows[0]) and all(r["sample_id"]==sample_id for r in rows) and len({(r["sample_id"],r["word_index"]) for r in rows})==len(rows)

def run_scope(manifest,output,cmudict,model_dir,mode,full_approval=None):
    rows=read_manifest(manifest)
    if mode=="smoke": rows=select_smoke(rows,10)
    elif mode=="full":
        if not full_approval or not full_approval.is_file() or not json.loads(full_approval.read_text()).get("stage2b_full_extraction_approved"): raise ContractError("full extraction approval absent")
    else: raise ContractError("unsupported adapter mode")
    cmu=verify_cmudict(cmudict); aligner=FrozenWordAligner(model_dir,"cuda"); by_unit=output/"by_utterance"; statuses=[]
    for item in rows:
        target=by_unit/f"{item['sample_id'].replace(':','__')}.csv"
        if valid_unit(target,item["sample_id"]): statuses.append({"sample_id":item["sample_id"],"status":"skipped_valid"}); continue
        try: result=process_utterance(item,aligner,cmu); atomic_csv(target,result); statuses.append({"sample_id":item["sample_id"],"status":"complete"})
        except Exception as exc: statuses.append({"sample_id":item["sample_id"],"status":"failed","error_code":type(exc).__name__})
    write_json(output/"adapter_run_SAFE.json",{"mode":mode,"utterance_count":len(rows),"gate_e_evaluated":False,"provenance":PROVENANCE,"statuses":statuses})

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("smoke","full"),required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--cmudict",type=Path,required=True); p.add_argument("--model-dir",type=Path,required=True); p.add_argument("--full-approval",type=Path)
    a=p.parse_args(argv); run_scope(a.manifest,a.output_root,a.cmudict,a.model_dir,a.mode,a.full_approval); return 0
if __name__=="__main__": raise SystemExit(main())
