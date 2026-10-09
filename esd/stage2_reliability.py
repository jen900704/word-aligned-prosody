#!/usr/bin/env python3
"""Checksum-gated Stage 2 orchestration and frozen ESD repeatability analysis.

Alignment/acoustic commands must use the frozen v5 runtime and emit the documented
token CSV. This program never invokes Stage 2 implicitly from Stage 1.
"""
from __future__ import annotations

import argparse, csv, json, math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Optional
from esd_contract import *

REQUIRED_TOKEN_FIELDS={"sample_id","speaker_id","condition","prompt_id","utterance_id","word_index","normalized_word","alignment_status","f0_mean_hz","f0_p10_hz","f0_p90_hz","energy_db","word_duration_sec","authoritative_syllable_count"}
HERE=Path(__file__).resolve().parent

def load_stage1(stage1: Path, expected_checksum: str, approval_file: Optional[Path]=None):
    approval_file=approval_file or HERE/"stage1_realdata_approval.json"
    approval=json.loads(approval_file.read_text())
    decision=json.loads((stage1/"stage1_decision_SAFE.json").read_text())
    manifest=stage1/"esd_manifest_PRIVATE.csv"
    actual=sha256(manifest)
    if not approval.get("approved") or not approval.get("stage1_pass") or approval.get("acoustic_processing_performed") is not False: raise ContractError("Stage 2 refused: explicit Stage-1 approval invalid")
    if expected_checksum!=APPROVED_STAGE1_MANIFEST_SHA256 or approval.get("manifest_sha256")!=APPROVED_STAGE1_MANIFEST_SHA256: raise ContractError("Stage 2 refused: checksum is not the sole approved real manifest")
    if not decision.get("stage1_pass") or decision.get("acoustic_processing_performed") is not False: raise ContractError("Stage 2 refused: Stage 1 decision invalid")
    if actual!=decision.get("manifest_sha256") or actual!=APPROVED_STAGE1_MANIFEST_SHA256: raise ContractError("Stage 2 refused: frozen manifest checksum mismatch")
    if tuple(decision.get("selected_speakers",[]))!=ELIGIBLE_ESD_SPEAKERS: raise ContractError("Stage 2 refused: speaker scope mismatch")
    return manifest,actual

def stage2a_preflight(stage1:Path,expected_checksum:str,output:Path,approval_file:Path,adapter_lock:Path,smoke_limit:int):
    if not 1<=smoke_limit<=20: raise ContractError("Stage 2A smoke limit must be bounded to 1-20 utterances")
    manifest,actual=load_stage1(stage1,expected_checksum,approval_file)
    lock=json.loads(adapter_lock.read_text()); blockers=[]
    for name,item in sorted(lock.items()):
        if not item.get("ready") and item.get("blocks_smoke",True): blockers.append(f"{name}: {item.get('reason','not ready')}")
    report={"phase":"2A","approved_manifest_sha256":actual,"smoke_limit":smoke_limit,
            "full_acoustic_job_invoked":False,"adapter_lock":lock,"ready_for_smoke":not blockers,
            "ready_for_stage2b":False,"blockers":blockers}
    output.mkdir(parents=True,exist_ok=True); write_json(output/"stage2a_readiness_SAFE.json",report)
    if blockers: raise ContractError("Stage 2A blocked: "+"; ".join(blockers))
    return report

def read_tokens(path: Path):
    with path.open(newline="",encoding="utf-8") as f:
        rd=csv.DictReader(f); fields=set(rd.fieldnames or [])
        missing=REQUIRED_TOKEN_FIELDS-fields
        if missing: raise ContractError("token CSV missing fields: "+",".join(sorted(missing)))
        rows=list(rd)
    evidence=[]
    for r in rows:
        evidence.append((r["sample_id"],r["word_index"]))
        try:
            r["word_index"]=int(r["word_index"])
            syllables=r["authoritative_syllable_count"]
            r["authoritative_syllable_count"]=float(syllables) if syllables not in {"","NA","NaN","nan"} else None
            for k in ("f0_mean_hz","f0_p10_hz","f0_p90_hz","energy_db","word_duration_sec"):
                r[k]=float(r[k]) if r[k] not in {"","NA","NaN","nan"} else None
        except ValueError: r["parse_error"]=True
    if len(evidence)!=len(set(evidence)): raise ContractError("duplicate acoustic evidence detected")
    return rows

def prepare_features(rows):
    # Frozen robust range and duration definition. No F0 imputation.
    eligible=[r for r in rows if r.get("alignment_status")=="aligned" and not r.get("parse_error")]
    for r in eligible:
        lo,hi=r["f0_p10_hz"],r["f0_p90_hz"]
        r["f0_robust_range_hz"]=(hi-lo) if lo is not None and hi is not None else None
        r["log_word_duration"]=math.log(r["word_duration_sec"]) if r["word_duration_sec"] and r["word_duration_sec"]>0 else None
    # Frozen v5 OLS, fit once on the complete analysis tokens (never smoke):
    # log(duration) ~ 1 + authoritative syllable count + speaker fixed effects.
    valid=[r for r in eligible if r["log_word_duration"] is not None and r["authoritative_syllable_count"] is not None and r["authoritative_syllable_count"]>0]
    if valid:
        try:
            import numpy as np
            speakers=sorted({x["speaker_id"] for x in valid})
            X=np.array([[1,x["authoritative_syllable_count"]]+[x["speaker_id"]==s for s in speakers[1:]] for x in valid],float); y=np.array([x["log_word_duration"] for x in valid])
            if np.linalg.matrix_rank(X)<X.shape[1]: raise ContractError("duration residualization not identifiable")
            beta=np.linalg.lstsq(X,y,rcond=None)[0]
            for r,v in zip(valid,y-X@beta): r["duration_log_syllable_residual"]=float(v)
        except ImportError as e: raise ContractError("Stage 2 requires numpy in the existing environment") from e
    return eligible

def gate_e_pass(reports):
    passed=[x["feature"] for x in reports if x["passed"]]
    return len(passed)>=3 and "f0_mean_hz" in passed,passed

def analyze(token_csv: Path, output: Path, technical_qa:Path, seed=13, iterations=1000):
    from esd_reml_reliability import analyze_feature,BOOTSTRAP_REPLICATES,BOOTSTRAP_SEED
    if seed!=BOOTSTRAP_SEED or iterations!=BOOTSTRAP_REPLICATES: raise ContractError("frozen bootstrap requires B=1000 and seed=13")
    qa=json.loads(technical_qa.read_text())
    if qa.get("human_review_status")!="approved" or qa.get("confirmed_technical_defect") is None or not qa.get("gate_e_authorized"): raise ContractError("Gate E refused: completed human technical-defect QA approval absent")
    technical_defect=bool(qa["confirmed_technical_defect"])
    rows=prepare_features(read_tokens(token_csv)); reports=[]; errors=[]
    for i,f in enumerate(COMPARABLE_FEATURES):
        try:
            result=analyze_feature(rows,f,bootstrap_replicates=BOOTSTRAP_REPLICATES)
            primary=result["primary"]; repetitions=Counter((r["speaker_id"],r["normalized_word"],r["condition"]) for r in rows if r.get(f) is not None)
            eligible_repetitions=[n for n in repetitions.values() if n>=2]; speakers={r["speaker_id"] for r in rows if r.get(f) is not None}; missing=sum(r.get(f) is None for r in rows)/max(len(rows),1)
            loso=result["leave_one_speaker_out"]; valid_loso=[x["estimate"] for x in loso["results"] if x["status"]=="complete"]
            lower=result["bootstrap"]["ci_95"][0]; point=primary["equal_condition_mean"]
            checks=((len(speakers)>=8,"speaker_count"),(len(eligible_repetitions)>=500,"matched_cells"),(eligible_repetitions and median(eligible_repetitions)>=3,"median_repetitions"),(missing<=.10,"missingness"),(point>=.70,"point_estimate"),(lower is not None and lower>=.50,"bootstrap_lower"),(len(valid_loso)==9 and min(valid_loso)>=.50,"leave_one_speaker_out"),(not technical_defect,"technical_defect"),(not result["bootstrap"]["technical_review_required"],"bootstrap_technical_review"),(not loso["qa_review_required"],"loso_technical_review"))
            failures=[name for ok,name in checks if not ok]
            reports.append({**result,"eligible_speaker_count":len(speakers),"matched_speaker_word_condition_cells":len(eligible_repetitions),"median_repetition_count":median(eligible_repetitions) if eligible_repetitions else None,"missingness":missing,"point_repeatability":point,"bootstrap_95_ci":result["bootstrap"]["ci_95"],"leave_one_speaker_out_range":[min(valid_loso),max(valid_loso)] if valid_loso else None,"technical_defect_confirmed":technical_defect,"passed":not failures,"failure_reasons":failures})
        except Exception as e: errors.append({"feature":f,"error":f"{type(e).__name__}: {e}"})
    if errors: raise ContractError("estimand not identifiable; no Gate E decision: "+json.dumps(errors,sort_keys=True))
    gate,passed=gate_e_pass(reports)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"esd_feature_diagnostics_SAFE.json",{"features":reports,"estimation_method":"REML","primary_condition_fits":5,"condition_aggregation":"unweighted_arithmetic_mean","pause_status":PAUSE_STATUS,"pause_counted_in_gate":False,"cross_emotion_is_secondary":True})
    write_json(output/"gate_e_decision_SAFE.json",{"gate_e_pass":gate,"comparable_features":list(COMPARABLE_FEATURES),"passing_features":passed,"thresholds":GATE_E_THRESHOLDS,"liwc_differentiation_calculated":False})
    return gate

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",choices=("2a","2b","2c"),required=True); ap.add_argument("--stage1-root",required=True); ap.add_argument("--expected-manifest-sha256",required=True); ap.add_argument("--output-root",required=True); ap.add_argument("--approval-file",default=str(HERE/"stage1_realdata_approval.json")); ap.add_argument("--adapter-lock",default=str(HERE/"stage2_adapter_lock.json")); ap.add_argument("--smoke-limit",type=int,default=10); ap.add_argument("--token-features-csv"); ap.add_argument("--technical-qa"); ap.add_argument("--bootstrap-iterations",type=int,default=1000); ap.add_argument("--seed",type=int,default=13)
    a=ap.parse_args(argv); out=Path(a.output_root)
    if a.phase=="2a": stage2a_preflight(Path(a.stage1_root),a.expected_manifest_sha256,out,Path(a.approval_file),Path(a.adapter_lock),a.smoke_limit); return 0
    load_stage1(Path(a.stage1_root),a.expected_manifest_sha256,Path(a.approval_file))
    readiness=out/"stage2a_readiness_SAFE.json"
    if not readiness.is_file() or not json.loads(readiness.read_text()).get("ready_for_stage2b"): raise ContractError("Stage 2B/2C refused: approved Stage 2A readiness absent")
    if a.phase=="2b": raise ContractError("Stage 2B disabled: separate full-extraction approval and shard/aggregation review absent")
    completion=out/"stage2b_completion_SAFE.json"
    if not completion.is_file() or not json.loads(completion.read_text()).get("qa_pass"): raise ContractError("Stage 2C refused: Stage 2B completion and QA pass absent")
    token=Path(a.token_features_csv) if a.token_features_csv else out/"acoustics_raw"/"esd_word_features_PRIVATE.csv"
    if not token.is_file(): raise ContractError("Stage 2 token feature CSV absent")
    if not a.technical_qa: raise ContractError("Stage 2C refused: technical QA artifact absent")
    analyze(token,out/"reliability_SAFE",Path(a.technical_qa),a.seed,a.bootstrap_iterations); return 0
if __name__=="__main__": raise SystemExit(main())
