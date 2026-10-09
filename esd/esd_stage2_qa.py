#!/usr/bin/env python3
"""SAFE aggregate QA for ESD Stage-2 word outputs; never emits transcript text."""
import argparse,csv,json,math
from collections import Counter
from pathlib import Path
from esd_contract import ContractError,write_json

def audit(input_root:Path,output:Path,expected_units:int):
    files=sorted((input_root/"by_utterance").glob("*.csv")); counts=Counter(); evidence=set(); units=set(); tokens=0
    for path in files:
        try:
            with path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
        except Exception: counts["output_read_failures"]+=1; continue
        for row in rows:
            tokens+=1; units.add(row.get("sample_id","")); key=(row.get("sample_id",""),row.get("word_index",""))
            if key in evidence: counts["duplicate_acoustic_evidence"]+=1
            evidence.add(key)
            status=row.get("alignment_status")
            if status!="aligned": counts["alignment_failures"]+=1
            if status=="unaligned": counts["unmatched_lexical_tokens"]+=1
            try:
                start=float(row["word_start_sec"]); end=float(row["word_end_sec"])
                if start<0 or end<=start: counts["invalid_word_bounds"]+=1
            except (ValueError,KeyError):
                if status=="aligned": counts["invalid_word_bounds"]+=1
            if row.get("feature_status")=="failed": counts["extraction_exceptions"]+=1
            if row.get("feature_error_code") in {"LibsndfileError","SoundFileRuntimeError","audio_read_failure"}: counts["audio_read_failures"]+=1
            if not row.get("f0_mean_hz"): counts["f0_missing"]+=1
            if not row.get("authoritative_syllable_count"): counts["syllable_lookup_failures"]+=1
            for field in ("f0_mean_hz","f0_p10_hz","f0_p90_hz","f0_robust_range_hz","energy_db","word_duration_sec","log_word_duration"):
                value=row.get(field,"")
                if value:
                    try:
                        number=float(value)
                        if not math.isfinite(number) or (field.startswith("f0_") and number<0) or (field=="word_duration_sec" and number<=0): counts["impossible_or_nonfinite_values"]+=1
                    except ValueError: counts["impossible_or_nonfinite_values"]+=1
    counts["missing_utterance_outputs"]=max(0,expected_units-len(units)); counts["utterance_outputs"]=len(units); counts["word_tokens"]=tokens
    artifact={"qa_version":"esd_stage2_qa_v1","expected_utterances":expected_units,"metrics":dict(sorted(counts.items())),
              "technical_defect_threshold_prespecified":False,"technical_defect_human_review_required":True,
              "human_review_status":"pending","confirmed_technical_defect":None,"gate_e_authorized":False}
    write_json(output,artifact); return artifact

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--input-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--expected-utterances",type=int,required=True)
    a=p.parse_args(argv); audit(a.input_root,a.output,a.expected_utterances); return 0
if __name__=="__main__": raise SystemExit(main())
