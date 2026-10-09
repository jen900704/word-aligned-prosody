#!/usr/bin/env python3
"""Authorization-gated MSP acoustic extraction runner v3 (shard-local token retention).

Execution/orchestration only. It joins the frozen content-QA token manifest to
the frozen final-scope manifest, reads selected audio only after guard-v4
authorization, and delegates interval acoustics to the frozen MSP adapter using
the frozen ESD analyze_word implementation. It makes no scientific choices.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import tools.msp_acoustic_adapter_v1 as acoustic_adapter
import tools.msp_esd_analyze_word_v1 as frozen_analyzer
from tools.msp_acoustic_adapter_v1 import analyze_interval, relative_position

TOKEN_REQ={"filename","speaker_id","recording_unit_id","word_index","normalized_word","word_start_sec","word_end_sec","word_duration_sec","authoritative_syllable_count","syllable_lookup_status"}
FINAL_REQ={"filename","speaker_id","recording_unit_id","primary_half"}|{f"alt_{i:02d}_half" for i in range(25)}
OUT_FIELDS=["filename","speaker_id","recording_unit_id","primary_half"]+[f"alt_{i:02d}_half" for i in range(25)]+[
 "word_index","normalized_word","relative_position","word_start_sec","word_end_sec","word_duration_sec",
 "authoritative_syllable_count","syllable_lookup_status","f0_mean_hz","f0_robust_range_hz","energy_db","acoustic_status","acoustic_error"]

MSP_ADAPTER_SHA256="06d100fec4f38d0be36dbceaf1b007b65d26b253282e0aa39c97229438bbb3c1"
STANDALONE_ANALYZER_SHA256="3d191cb11bc2ce7a3157ecc5ff26e8cce109b8a905839a645f410d22e13d504d"
SOURCE_ESD_ADAPTER_SHA256="1fc1606ed44784f6bc1c05673f150cad799c0ed030df6043c3260736424a9b01"

class ExtractionRunnerError(ValueError): pass

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def validate_authorization(preflight_path:Path):
    o=json.loads(preflight_path.read_text(encoding='utf-8'))
    if o.get('audit')!='msp_acoustic_preflight_guard_v4' or o.get('msp_audio_processing_authorized') is not True:
        raise ExtractionRunnerError('preflight_not_authorized')
    checks=o.get('checks')
    if not isinstance(checks,dict) or not checks or not all(v is True for v in checks.values()):
        raise ExtractionRunnerError('preflight_checks_incomplete')
    if o.get('scientific_choices_made_by_guard') is not False or o.get('reliability_values_accessed') is not False:
        raise ExtractionRunnerError('preflight_provenance_invalid')
    return o

def load_final(final_csv:Path,final_safe:Path):
    safe=json.loads(final_safe.read_text(encoding='utf-8'))
    if safe.get('audit')!='msp_final_scope_and_split_manifest_v1' or safe.get('private_manifest_sha256')!=sha256(final_csv):
        raise ExtractionRunnerError('final_scope_hash_mismatch')
    out={}
    with final_csv.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f); fields=set(r.fieldnames or [])
        if not FINAL_REQ<=fields: raise ExtractionRunnerError('final_scope_schema')
        for x in r:
            fn=x['filename']
            meta={k:x[k] for k in ['speaker_id','recording_unit_id','primary_half']+[f'alt_{i:02d}_half' for i in range(25)]}
            if fn in out:
                if out[fn]!=meta: raise ExtractionRunnerError('final_scope_duplicate_conflict')
            else: out[fn]=meta
    if not out: raise ExtractionRunnerError('empty_final_scope')
    return out

def load_tokens(token_csv:Path,alignment_safe:Path,selected):
    safe=json.loads(alignment_safe.read_text(encoding='utf-8'))
    if safe.get('audit')!='msp_textgrid_content_qa_v1' or safe.get('private_token_manifest_sha256')!=sha256(token_csv):
        raise ExtractionRunnerError('token_manifest_hash_mismatch')
    by=defaultdict(list); seen=set()
    with token_csv.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f); fields=set(r.fieldnames or [])
        if not TOKEN_REQ<=fields: raise ExtractionRunnerError('token_manifest_schema')
        for x in r:
            fn=x['filename']
            if fn not in selected: continue
            if x['speaker_id']!=selected[fn]['speaker_id'] or x['recording_unit_id']!=selected[fn]['recording_unit_id']:
                raise ExtractionRunnerError('token_final_metadata_mismatch')
            try: wi=int(x['word_index'])
            except Exception as exc: raise ExtractionRunnerError('word_index_invalid') from exc
            key=(fn,wi)
            if key in seen: raise ExtractionRunnerError('duplicate_token_key')
            seen.add(key); y=dict(x); y['word_index_int']=wi; by[fn].append(y)
    if set(by)!=set(selected): raise ExtractionRunnerError('selected_file_missing_tokens')
    for fn,rows in by.items():
        rows.sort(key=lambda x:x['word_index_int']); idx=[x['word_index_int'] for x in rows]
        if idx!=list(range(len(rows))): raise ExtractionRunnerError('noncontiguous_word_index')
        if any(not x['normalized_word'] for x in rows): raise ExtractionRunnerError('empty_normalized_word')
    return by

def shard_for(filename:str,num_shards:int)->int:
    if isinstance(num_shards,bool) or not isinstance(num_shards,int) or num_shards<1: raise ExtractionRunnerError('invalid_num_shards')
    return int(hashlib.sha256(filename.encode('utf-8')).hexdigest(),16)%num_shards

def default_audio_reader(path:Path):
    import soundfile as sf
    data,sr=sf.read(path,dtype='float32',always_2d=True)
    return data,int(sr)

def load_frozen_analyzer():
    if sha256(Path(acoustic_adapter.__file__))!=MSP_ADAPTER_SHA256: raise ExtractionRunnerError('msp_acoustic_adapter_hash_mismatch')
    if sha256(Path(frozen_analyzer.__file__))!=STANDALONE_ANALYZER_SHA256: raise ExtractionRunnerError('standalone_analyzer_hash_mismatch')
    if frozen_analyzer.SOURCE_ESD_ADAPTER_SHA256!=SOURCE_ESD_ADAPTER_SHA256: raise ExtractionRunnerError('source_esd_adapter_provenance_mismatch')
    return frozen_analyzer.analyze_word

def _finite_or_blank(v):
    if v is None: return ''
    try: z=float(v)
    except Exception: return ''
    return format(z,'.12g') if math.isfinite(z) else ''

def process_file(fn,meta,tokens,audio_root,audio_reader,analyze_word):
    path=audio_root/fn
    if not path.is_file(): raise ExtractionRunnerError('audio_file_missing')
    audio,sr=audio_reader(path)
    n=len(tokens); out=[]
    for t in tokens:
        wi=t['word_index_int']; base={"filename":fn,**meta,"word_index":str(wi),"normalized_word":t['normalized_word'],
          "relative_position":format(relative_position(wi,n),'.12g'),"word_start_sec":t['word_start_sec'],"word_end_sec":t['word_end_sec'],
          "word_duration_sec":t['word_duration_sec'],"authoritative_syllable_count":t['authoritative_syllable_count'],"syllable_lookup_status":t['syllable_lookup_status']}
        try:
            vals=analyze_interval(audio,sr,float(t['word_start_sec']),float(t['word_end_sec']),analyze_word)
            base.update(f0_mean_hz=_finite_or_blank(vals.get('f0_mean_hz')),f0_robust_range_hz=_finite_or_blank(vals.get('f0_robust_range_hz')),energy_db=_finite_or_blank(vals.get('energy_db')),acoustic_status='complete',acoustic_error='')
        except Exception as exc:
            base.update(f0_mean_hz='',f0_robust_range_hz='',energy_db='',acoustic_status='failed',acoustic_error=type(exc).__name__)
        out.append(base)
    return out

def atomic_csv(path:Path,rows):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=OUT_FIELDS,extrasaction='raise',lineterminator='\n'); w.writeheader(); w.writerows(rows)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def run(preflight,alignment_safe,final_safe,token_csv,final_csv,audio_root,output_csv,safe_json,shard_id,num_shards,audio_reader=default_audio_reader,analyze_word=None):
    validate_authorization(preflight)
    if not (isinstance(shard_id,int) and not isinstance(shard_id,bool) and isinstance(num_shards,int) and not isinstance(num_shards,bool) and num_shards>=1 and 0<=shard_id<num_shards): raise ExtractionRunnerError('invalid_shard')
    selected_all=load_final(final_csv,final_safe)
    files=sorted(fn for fn in selected_all if shard_for(fn,num_shards)==shard_id)
    selected={fn:selected_all[fn] for fn in files}
    tokens=load_tokens(token_csv,alignment_safe,selected)
    if analyze_word is None: analyze_word=load_frozen_analyzer()
    rows=[]; file_failed=0
    for fn in files:
        try: rows.extend(process_file(fn,selected[fn],tokens[fn],audio_root,audio_reader,analyze_word))
        except Exception:
            file_failed+=1
            raise
    atomic_csv(output_csv,rows)
    failed_tokens=sum(x.get('acoustic_status')=='failed' for x in rows)
    safe={"audit":"msp_acoustic_extraction_shard_v2","status":"PASS","shard_id":shard_id,"num_shards":num_shards,"selected_files_total":len(selected_all),"files_in_shard":len(files),"files_failed":file_failed,"token_rows":len(rows),"acoustic_failed_tokens":failed_tokens,"output_sha256":sha256(output_csv),"preflight_sha256":sha256(preflight),"alignment_safe_sha256":sha256(alignment_safe),"final_scope_safe_sha256":sha256(final_safe),"token_manifest_sha256":sha256(token_csv),"final_private_manifest_sha256":sha256(final_csv),"emotion_fields_used":False,"reliability_values_accessed":False,"clinical_outcomes_accessed":False,"scientific_choices_made_by_runner":False}
    safe_json.parent.mkdir(parents=True,exist_ok=True); safe_json.write_text(json.dumps(safe,indent=2,sort_keys=True)+'\n',encoding='utf-8'); return safe

def main(argv=None):
    p=argparse.ArgumentParser()
    for arg in ('preflight','alignment-safe','final-safe','token-csv','final-csv','audio-root','output-csv','safe-json'): p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--shard-id',type=int,required=True); p.add_argument('--num-shards',type=int,required=True); a=p.parse_args(argv)
    run(a.preflight,a.alignment_safe,a.final_safe,a.token_csv,a.final_csv,a.audio_root,a.output_csv,a.safe_json,a.shard_id,a.num_shards); return 0
if __name__=='__main__': raise SystemExit(main())
