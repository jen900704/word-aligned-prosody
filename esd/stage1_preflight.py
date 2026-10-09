#!/usr/bin/env python3
"""Read-only ESD Stage-1 v3 preflight and deterministic PRIVATE manifest."""
from __future__ import annotations

import argparse, csv, json, re, unicodedata, wave
from collections import Counter
from pathlib import Path
from esd_contract import *

CONDITION_RANGES={
    "Neutral":(1,350,0), "Angry":(351,700,350),
    "Happy":(701,1050,700), "Sad":(1051,1400,1050),
    "Surprise":(1401,1750,1400),
}
AUDIO_RE=re.compile(r"^(\d{4})_(\d{6})\.wav$",re.IGNORECASE)
UTTERANCE_RE=re.compile(r"^(\d{4})_(\d{6})$")

def speaker_dirs(root: Path):
    found={}
    for path in sorted(root.rglob("*"),key=lambda x:x.as_posix()):
        if path.is_dir() and path.name in ENGLISH_SPEAKERS+CHINESE_SPEAKERS:
            found.setdefault(path.name,path)
    return found

def prompt_id_for(condition: str, global_index: int) -> int:
    if condition not in CONDITION_RANGES: raise ContractError("unknown ESD condition")
    lower,upper,offset=CONDITION_RANGES[condition]
    if not lower<=global_index<=upper: raise ContractError("global utterance index outside frozen condition range")
    prompt=global_index-offset
    if not 1<=prompt<=EXPECTED_PROMPTS: raise ContractError("derived prompt ID outside 1-350")
    return prompt

def parse_audio_name(path: Path, speaker: str, condition: str):
    match=AUDIO_RE.fullmatch(path.name)
    if not match: raise ContractError("audio filename does not match <speaker>_<global-index>.wav")
    embedded,index=match.group(1),int(match.group(2))
    if embedded!=speaker: raise ContractError("audio filename speaker mismatch")
    return {"speaker_id":speaker,"condition":condition,"global_utterance_index":index,
            "prompt_id":prompt_id_for(condition,index),"utterance_id":f"{speaker}_{index:06d}"}

def transcript_path(directory: Path, speaker: str) -> Path: return directory/f"{speaker}.txt"

def normalize_parallel_text(text: str) -> str:
    """Equality diagnostic only: NFC plus surrounding format whitespace."""
    return unicodedata.normalize("NFC",text.replace("\r\n","\n").replace("\r","\n")).strip()

def normalize_punctuation_diagnostic(text: str) -> str:
    """Diagnostic-only punctuation removal; never used for joins or alignment."""
    normalized=normalize_parallel_text(text)
    return "".join(ch for ch in normalized if not unicodedata.category(ch).startswith("P"))

def parse_transcript(path: Path, expected_speaker: str):
    """Parse headerless ESD TSV, stripping only surrounding field whitespace."""
    rows=[]; seen=set()
    with path.open("r",encoding="utf-8-sig",newline="") as handle:
        for fields in csv.reader(handle,delimiter="\t"):
            if not fields or all(not field.strip() for field in fields): continue
            if len(fields)!=3: raise ContractError("transcript row does not have exactly three TSV fields")
            utterance=fields[0].strip(); text=normalize_parallel_text(fields[1]); condition=fields[2].strip()
            match=UTTERANCE_RE.fullmatch(utterance)
            if not match: raise ContractError("invalid transcript utterance_id")
            embedded,index=match.group(1),int(match.group(2))
            if embedded!=expected_speaker: raise ContractError("transcript speaker mismatch")
            prompt=prompt_id_for(condition,index)
            if utterance in seen: raise ContractError("duplicate transcript utterance_id")
            seen.add(utterance)
            rows.append({"speaker_id":expected_speaker,"utterance_id":utterance,
                         "global_utterance_index":index,"condition":condition,"prompt_id":prompt,
                         "transcript_text":text,"normalized_parallel_text":text,
                         "punctuation_normalized_text":normalize_punctuation_diagnostic(text)})
    return "utterance_text_condition_tsv_v2",rows

def audio_header(path: Path):
    try:
        with wave.open(str(path),"rb") as handle:
            frames=handle.getnframes(); rate=handle.getframerate(); handle.readframes(frames)
            return {"readable":True,"sample_rate":rate,"channels":handle.getnchannels(),
                    "sample_width":handle.getsampwidth(),"frames":frames,
                    "duration_seconds":round(frames/rate,6) if rate else None}
    except Exception as exc: return {"readable":False,"audio_error":type(exc).__name__}

def expected_prompt_set(count): return set(range(1,count+1))

def comparison_diagnostics(transcripts, expected_prompts):
    counts=Counter({"within_speaker_exact_mismatch":0,
                    "within_speaker_punctuation_normalized_mismatch":0,
                    "cross_speaker_exact_mismatch":0,
                    "cross_speaker_punctuation_normalized_mismatch":0})
    mappings={}
    for speaker,rows in transcripts.items():
        mapping={(row["condition"],row["prompt_id"]):row for row in rows}; mappings[speaker]=mapping
        for prompt in expected_prompt_set(expected_prompts):
            group=[mapping.get((condition,prompt)) for condition in CONDITION_RANGES]
            exact={row["normalized_parallel_text"] for row in group if row}
            punctuation={row["punctuation_normalized_text"] for row in group if row}
            if len(group)!=EXPECTED_CONDITIONS or any(row is None for row in group) or len(exact)!=1:
                counts["within_speaker_exact_mismatch"]+=1
            if len(group)!=EXPECTED_CONDITIONS or any(row is None for row in group) or len(punctuation)!=1:
                counts["within_speaker_punctuation_normalized_mismatch"]+=1
    reference=ELIGIBLE_ESD_SPEAKERS[0]
    reference_map=mappings.get(reference,{})
    for speaker in ELIGIBLE_ESD_SPEAKERS[1:]:
        mapping=mappings.get(speaker,{})
        for key,row in reference_map.items():
            other=mapping.get(key)
            if other is None or row["normalized_parallel_text"]!=other["normalized_parallel_text"]:
                counts["cross_speaker_exact_mismatch"]+=1
            if other is None or row["punctuation_normalized_text"]!=other["punctuation_normalized_text"]:
                counts["cross_speaker_punctuation_normalized_mismatch"]+=1
    return dict(sorted(counts.items()))

def run(root: Path, out: Path, expected_prompts=EXPECTED_PROMPTS):
    reject_forbidden_path(root); ensure_output_outside_source(root,out); out.mkdir(parents=True,exist_ok=True)
    dirs=speaker_dirs(root); errors=[]; before={}
    present_eligible=tuple(sorted(x for x in dirs if x in ELIGIBLE_ESD_SPEAKERS))
    if present_eligible!=ELIGIBLE_ESD_SPEAKERS: errors.append("selected_eligible_speaker_scope_mismatch")

    exclusions=[]
    excluded_dir=dirs.get("0014")
    if excluded_dir is None: errors.append("0014_exclusion_source_missing")
    else:
        try:
            parse_transcript(transcript_path(excluded_dir,"0014"),"0014")
            errors.append("0014_expected_permission_denial_not_observed")
        except PermissionError:
            exclusions.append({"speaker_id":"0014","excluded_reason":"transcript_permission_denied"})
        except (OSError,ContractError) as exc:
            errors.append(f"0014_exclusion_not_permission_denial:{type(exc).__name__}")

    audio=[]
    for speaker in ELIGIBLE_ESD_SPEAKERS:
        directory=dirs.get(speaker)
        if directory is None: continue
        conditions=[path for path in sorted(directory.iterdir(),key=lambda x:x.name) if path.is_dir() and path.name in CONDITION_RANGES]
        if {path.name for path in conditions}!=set(CONDITION_RANGES): errors.append(f"{speaker}:condition_completeness_failed")
        for condition_dir in conditions:
            condition=condition_dir.name; parsed=[]
            files=[path for path in sorted(condition_dir.iterdir(),key=lambda x:x.name)
                   if path.is_file() and path.suffix.casefold()==".wav"]
            for path in files:
                before[str(path.resolve())]=(path.stat().st_size,path.stat().st_mtime_ns)
                try: identity=parse_audio_name(path,speaker,condition)
                except ContractError as exc: errors.append(f"{speaker}/{condition}:audio_identity_error:{exc}"); continue
                header=audio_header(path)
                if not header["readable"]: errors.append(f"{speaker}/{condition}:unreadable_audio")
                audio.append({**identity,"sample_id":f"{speaker}:{condition}:{identity['prompt_id']:03d}",
                              "audio_path":str(path.resolve()),**header}); parsed.append(identity["prompt_id"])
            if len(files)!=expected_prompts or len(parsed)!=expected_prompts or set(parsed)!=expected_prompt_set(expected_prompts):
                errors.append(f"{speaker}/{condition}:prompt_completeness_failed")
    audio_keys=[(row["speaker_id"],row["condition"],row["prompt_id"]) for row in audio]
    if len(audio_keys)!=len(set(audio_keys)): errors.append("duplicate_speaker_condition_prompt_key")

    transcripts={}; schemas={}
    for speaker in ELIGIBLE_ESD_SPEAKERS:
        if speaker not in dirs: continue
        try:
            schema,rows=parse_transcript(transcript_path(dirs[speaker],speaker),speaker)
            schemas[speaker]=schema; transcripts[speaker]=rows
            if len(rows)!=EXPECTED_CONDITIONS*expected_prompts: errors.append(f"{speaker}:transcript_row_count_failed")
        except PermissionError: errors.append(f"{speaker}:eligible_transcript_permission_denied")
        except (OSError,ContractError) as exc: errors.append(f"{speaker}:transcript_parse_failed:{exc}")

    transcript_by_full={}
    expected_keys={(condition,prompt) for condition in CONDITION_RANGES for prompt in expected_prompt_set(expected_prompts)}
    for speaker,rows in transcripts.items():
        by_full={row["utterance_id"]:row for row in rows}; transcript_by_full[speaker]=by_full
        keys=[(row["condition"],row["prompt_id"]) for row in rows]
        if len(keys)!=len(set(keys)): errors.append(f"{speaker}:duplicate_transcript_condition_prompt_key")
        if set(keys)!=expected_keys: errors.append(f"{speaker}:transcript_condition_prompt_completeness_failed")

    diagnostic_counts=comparison_diagnostics(transcripts,expected_prompts)
    manifest=[]; joins=Counter()
    for audio_row in audio:
        speaker=audio_row["speaker_id"]
        transcript=transcript_by_full.get(speaker,{}).get(audio_row["utterance_id"])
        if transcript is None: joins["missing_audio_transcript_join"]+=1; continue
        if transcript["speaker_id"]!=speaker: joins["wrong_speaker_transcript_join"]+=1; continue
        if transcript["condition"]!=audio_row["condition"]: joins["transcript_condition_mismatch"]+=1; continue
        if transcript["prompt_id"]!=audio_row["prompt_id"]: joins["transcript_prompt_mismatch"]+=1; continue
        manifest.append({**audio_row,"transcript_text":transcript["transcript_text"]})
    for key in ("missing_audio_transcript_join","wrong_speaker_transcript_join","transcript_condition_mismatch","transcript_prompt_mismatch"):
        if joins[key]: errors.append(f"{key}:{joins[key]}")

    manifest_keys=[(row["speaker_id"],row["condition"],row["prompt_id"]) for row in manifest]
    expected_rows=len(ELIGIBLE_ESD_SPEAKERS)*EXPECTED_CONDITIONS*expected_prompts
    if len(manifest)!=expected_rows or len(manifest_keys)!=len(set(manifest_keys)): errors.append("final_manifest_completeness_failed")
    if any(row["speaker_id"]=="0014" for row in manifest): errors.append("excluded_speaker_0014_entered_manifest")
    unchanged=all(Path(path).stat().st_size==value[0] and Path(path).stat().st_mtime_ns==value[1] for path,value in before.items())
    if not unchanged: errors.append("source_changed_during_preflight")

    manifest.sort(key=lambda row:(row["speaker_id"],row["prompt_id"],row["condition"],row["global_utterance_index"]))
    fields=["sample_id","speaker_id","condition","global_utterance_index","prompt_id","utterance_id","audio_path","transcript_text","readable","sample_rate","channels","sample_width","frames","duration_seconds"]
    manifest_path=out/"esd_manifest_PRIVATE.csv"; write_csv(manifest_path,manifest,fields)
    manifest_checksum=sha256(manifest_path); atomic_text(out/"esd_manifest_PRIVATE.sha256",f"{manifest_checksum}  {manifest_path.name}\n")
    write_csv(out/"esd_exclusions_SAFE.csv",exclusions,["speaker_id","excluded_reason"])
    distributions=[]
    for field in ("sample_rate","channels","sample_width"):
        for value,count in sorted(Counter(str(row.get(field,"")) for row in audio).items()):
            distributions.append({"metric":field,"value":value,"count":count})
    write_csv(out/"esd_audio_metadata_SAFE.csv",distributions,["metric","value","count"])
    decision={"stage1_pass":not errors,"design_tag":DESIGN_TAG,"parser_version":"esd_final_manifest_v3",
              "prespec_tag":PRESPEC_TAG,"selected_speakers":list(ELIGIBLE_ESD_SPEAKERS),
              "technical_access_exclusions":exclusions,"speaker_count":len({row['speaker_id'] for row in manifest}),
              "condition_count":len({row['condition'] for row in manifest}),"prompt_count":expected_prompts,
              "manifest_rows":len(manifest),"expected_manifest_rows":expected_rows,"manifest_sha256":manifest_checksum,
              "parallel_prompt_text_is_hard_gate":False,"parallel_prompt_diagnostics":diagnostic_counts,
              "own_speaker_transcripts_authoritative":True,"transcript_schemas":sorted(set(schemas.values())),
              "pause_status":PAUSE_STATUS,"acoustic_processing_performed":False,"errors":errors}
    write_json(out/"stage1_decision_SAFE.json",decision)
    write_json(out/"source_readonly_SAFE.json",{"source_unchanged":unchanged,"audio_modified":False,"scope":"files read by Stage 1"})
    return decision

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--esd-root",required=True); parser.add_argument("--output-root",required=True); parser.add_argument("--expected-prompts",type=int,default=EXPECTED_PROMPTS)
    args=parser.parse_args(argv); decision=run(Path(args.esd_root),Path(args.output_root),args.expected_prompts)
    print(json.dumps(decision,sort_keys=True)); return 0 if decision["stage1_pass"] else 2
if __name__=="__main__": raise SystemExit(main())
