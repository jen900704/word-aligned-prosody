#!/usr/bin/env python3
"""Frozen Stage-2C ESD word-local crossed-REML reliability analysis."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, warnings
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from esd_contract import (APPROVED_STAGE1_MANIFEST_SHA256, COMPARABLE_FEATURES,
    ELIGIBLE_ESD_SPEAKERS, GATE_E_THRESHOLDS, ContractError,
    reject_forbidden_path, sha256, write_json)

CONDITIONS=("Angry","Happy","Neutral","Sad","Surprise")
ESTIMATION_METHOD="REML"
VC_FORMULA={"S":"0 + C(speaker)","W":"0 + C(word)","SW":"0 + C(speaker_word)"}
BOOTSTRAP_REPLICATES=1000; BOOTSTRAP_SEED=13; BOOTSTRAP_REVIEW_MIN_SUCCESS=950
PRIMARY_CELL_FIELDS=("speaker_id","normalized_word","condition")
LEXICAL_MIN_TOKENS=3; LEXICAL_MIN_SPEAKERS=8; LEXICAL_BALANCED_TOKENS=3
APPROVED_STAGE2B_AGGREGATE_SHA256="a6b03d9f71103e533ada5fe5d8071d2c5248fd01878724117fd025167b91cd9f"
FROZEN_STAGE2B_CLOSURE_SHA256="1ccae939e31f2b0c82210d718af7eaec2ec635bce3c38013b5a2ed7cf522505b"
BOOTSTRAP_PLAN_SHA256="b89fd97bf340b0ea305e4f03791146e9587fa29bbd000e374d145946eba824bd"

class FitFailure(ContractError):
    """Technical fit failure retaining warning and numeric evidence."""
    def __init__(self,message,diagnostic):
        super().__init__(message); self.diagnostic=diagnostic

def failure_record(exc):
    record={"error_type":type(exc).__name__}
    if isinstance(exc,FitFailure): record["fit_diagnostic"]=exc.diagnostic
    return record

def _finite_nonnegative(value):
    try: value=float(value)
    except (TypeError,ValueError): return False
    return math.isfinite(value) and value>=0

def valid_rows(rows,feature,condition,require_repetition=True):
    out=[]
    for row in rows:
        source_speaker=row.get("source_speaker_id",row.get("speaker_id"))
        if row.get("condition")!=condition or row.get("alignment_status","aligned")!="aligned" or source_speaker not in ELIGIBLE_ESD_SPEAKERS: continue
        try: value=float(row.get(feature))
        except (TypeError,ValueError): continue
        if math.isfinite(value): out.append({**row,feature:value})
    if not require_repetition: return out
    counts=Counter((r["speaker_id"],r["normalized_word"],condition) for r in out)
    return [r for r in out if counts[(r["speaker_id"],r["normalized_word"],condition)]>=2]

def build_crossed_model(rows,feature,condition):
    import pandas as pd
    import statsmodels.formula.api as smf
    data=valid_rows(rows,feature,condition)
    if not data: raise ContractError(f"{condition}: no repeated matched cells")
    frame=pd.DataFrame(data); frame["y"]=frame[feature].astype(float); frame["top_group"]="all"
    frame["speaker"]=frame["speaker_id"].astype(str); frame["word"]=frame["normalized_word"].astype(str)
    frame["speaker_word"]=frame["speaker"]+"\x1f"+frame["word"]
    return smf.mixedlm("y ~ 1",frame,groups="top_group",re_formula="0",
        vc_formula=VC_FORMULA,use_sparse=True),frame

def map_variance_components(model,result):
    names=list(model.exog_vc.names); values=list(result.vcomp)
    if len(names)!=len(values) or set(names)!={"S","W","SW"}: raise ContractError(f"unexpected variance-component names: {names!r}")
    mapped=dict(zip(names,map(float,values))); residual=float(result.scale)
    if not all(_finite_nonnegative(x) for x in mapped.values()): raise ContractError("nonfinite or negative REML variance component")
    if not math.isfinite(residual) or residual<=0: raise ContractError("nonfinite or nonpositive REML residual variance")
    if mapped["SW"]+residual<=0: raise ContractError("zero SW-plus-residual denominator")
    return mapped,residual

def fit_condition_reml(rows,feature,condition):
    model,frame=build_crossed_model(rows,feature,condition)
    caught=[]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result=model.fit(reml=True,method="lbfgs",maxiter=2000,disp=False)
    except Exception as exc:
        diagnostic={"converged":False,"llf":None,"warnings":[{"category":w.category.__name__,"message":str(w.message)} for w in caught],"technical_success":False,"fit_exception":{"category":type(exc).__name__,"message":str(exc)}}
        raise FitFailure(f"{condition}: model.fit raised {type(exc).__name__}",diagnostic) from exc
    warning_records=[{"category":w.category.__name__,"message":str(w.message)} for w in caught]
    diagnostic={"converged":None,"llf":None,"warnings":warning_records}
    disallowed=[w for w in warning_records if "boundary of the parameter space" not in w["message"].casefold()]
    try:
        diagnostic["converged"]=bool(result.converged)
        diagnostic["llf"]=float(result.llf)
        if not diagnostic["converged"]: raise ContractError(f"{condition}: REML did not converge")
        if not math.isfinite(diagnostic["llf"]): raise ContractError(f"{condition}: nonfinite REML log likelihood")
        components,residual=map_variance_components(model,result)
        value=components["SW"]/(components["SW"]+residual)
        if not math.isfinite(value) or not 0<=value<=1: raise ContractError(f"{condition}: invalid repeatability")
        diagnostic.update(variance_components={**components,"e":residual},repeatability=value)
        if disallowed: raise ContractError(f"{condition}: disallowed warning from model.fit")
    except Exception as exc:
        diagnostic["technical_success"]=False
        diagnostic["validation_exception"]={"category":type(exc).__name__,"message":str(exc)}
        raise FitFailure(f"{condition}: post-fit validation failed: {exc}",diagnostic) from exc
    diagnostic["technical_success"]=True
    return {"condition":condition,"method":ESTIMATION_METHOD,"repeatability":value,
      "variance_components":{**components,"e":residual},"token_count":len(frame),
      "matched_cell_count":len(set(zip(frame["speaker"],frame["word"]))),"converged":True,"fit_diagnostic":diagnostic}

def five_condition_summary(rows,feature,fit_fn=fit_condition_reml):
    fits=[fit_fn(rows,feature,c) for c in CONDITIONS]; values=[float(x["repeatability"]) for x in fits]
    if len(fits)!=5 or not all(_finite_nonnegative(x) and x<=1 for x in values): raise ContractError("complete finite five-condition REML estimate absent")
    return {"condition_estimates":fits,"equal_condition_mean":mean(values),"aggregation":"unweighted_arithmetic_mean","token_count_weighted":False}

def residualize_duration_once(rows):
    import numpy as np
    prepared=[dict(r) for r in rows]; valid=[]
    for i,row in enumerate(prepared):
        if row.get("alignment_status","aligned")!="aligned" or row.get("condition") not in CONDITIONS or row.get("speaker_id") not in ELIGIBLE_ESD_SPEAKERS: continue
        try: duration=float(row["word_duration_sec"]); syllables=float(row["authoritative_syllable_count"]); speaker=str(row["speaker_id"])
        except (KeyError,TypeError,ValueError): continue
        if duration>0 and syllables>0 and all(math.isfinite(x) for x in (duration,syllables)):
            valid.append((i,duration,syllables,speaker))
    speakers=sorted({x[3] for x in valid})
    design=[[1.,syllables]+[speaker==level for level in speakers[1:]] for _,_,syllables,speaker in valid]
    outcome=[math.log(duration) for _,duration,_,_ in valid]
    matrix=np.asarray(design,dtype=float)
    expected_columns=1+1+max(0,len(speakers)-1)
    rank=int(np.linalg.matrix_rank(matrix)) if matrix.ndim==2 and matrix.size else 0
    if matrix.ndim!=2 or matrix.shape[1:]!=(expected_columns,) or rank!=expected_columns: raise ContractError("duration residualization not identifiable")
    beta,*_=np.linalg.lstsq(matrix,np.asarray(outcome),rcond=None); residuals=np.asarray(outcome)-matrix@beta
    for (i,_,_,_),value in zip(valid,residuals): prepared[i]["duration_log_syllable_residual"]=float(value)
    columns=["intercept","authoritative_syllable_count"]+[f"speaker_id[{s}]" for s in speakers[1:]]
    return prepared,{"formula":"log(word_duration_sec) ~ 1 + authoritative_syllable_count + C(speaker_id)","design_columns":columns,"reference_speaker":speakers[0] if speakers else None,"speaker_levels":speakers,"expected_rank":expected_columns,"rank":rank,"n":len(valid),"beta":[float(x) for x in beta]}

def make_bootstrap_plan():
    import numpy as np
    speakers=tuple(sorted(ELIGIBLE_ESD_SPEAKERS))
    rng=np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices=rng.integers(0,len(speakers),size=(BOOTSTRAP_REPLICATES,len(speakers)))
    return tuple(tuple(speakers[int(i)] for i in row) for row in indices)

def serialize_bootstrap_plan(plan):
    return "".join(",".join(row)+"\n" for row in plan).encode("utf-8")

def validate_bootstrap_plan(plan):
    if len(plan)!=1000 or any(len(row)!=9 for row in plan): raise ContractError("bootstrap plan must be exactly 1000 x 9")
    if any(s not in ELIGIBLE_ESD_SPEAKERS for row in plan for s in row): raise ContractError("bootstrap plan contains ineligible speaker")
    digest=hashlib.sha256(serialize_bootstrap_plan(plan)).hexdigest()
    if digest!=BOOTSTRAP_PLAN_SHA256: raise ContractError("bootstrap plan checksum mismatch")
    return digest

def bootstrap_sample(rows,draws):
    if len(draws)!=9: raise ContractError("bootstrap replicate must contain nine draw slots")
    by=defaultdict(list)
    for row in rows: by[row["speaker_id"]].append(row)
    out=[]
    for i,speaker in enumerate(draws): out.extend({**r,"speaker_id":f"bootstrap_cluster_{i:02d}","source_speaker_id":speaker} for r in by[speaker])
    return out,draws

def speaker_cluster_bootstrap(rows,feature,fit_fn=fit_condition_reml,replicates=BOOTSTRAP_REPLICATES,seed=BOOTSTRAP_SEED,plan=None):
    if replicates!=1000 or seed!=13: raise ContractError("frozen bootstrap requires exactly B=1000 and seed=13")
    import numpy as np
    plan=make_bootstrap_plan() if plan is None else plan; plan_sha256=validate_bootstrap_plan(plan)
    estimates=[]; failures=[]
    for replicate,draws in enumerate(plan):
        try:
            sampled,_=bootstrap_sample(rows,draws); value=float(five_condition_summary(sampled,feature,fit_fn)["equal_condition_mean"])
            if not math.isfinite(value) or not 0<=value<=1: raise ContractError("invalid bootstrap estimate")
            estimates.append(value)
        except Exception as exc:
            item={"replicate":replicate,**failure_record(exc)}
            failures.append(item)
    success=len(estimates)
    ci=[float(x) for x in np.quantile(estimates,[.025,.975],method="linear")] if success>=950 else None
    return {"type":"speaker_cluster_nonparametric","rng":"numpy.random.Generator(numpy.random.PCG64(13))","seed":13,"plan_sha256":plan_sha256,"requested_replicates":1000,"successful_replicates":success,"failed_replicates":len(failures),"failed_details":failures,"ci_95":ci,"technical_review_required":success<950,"minimum_success_for_no_review":950,"replacement_draws_generated":False}

def leave_one_speaker_out(rows,feature,fit_fn=fit_condition_reml):
    if tuple(sorted({r["speaker_id"] for r in rows}))!=ELIGIBLE_ESD_SPEAKERS: raise ContractError("LOSO requires exactly the nine original eligible speakers")
    results=[]
    for speaker in ELIGIBLE_ESD_SPEAKERS:
        subset=[r for r in rows if r["speaker_id"]!=speaker]
        if len({r["speaker_id"] for r in subset})!=8: raise ContractError("LOSO deletion does not contain exactly eight speakers")
        try:
            value=float(five_condition_summary(subset,feature,fit_fn)["equal_condition_mean"])
            if not math.isfinite(value) or not 0<=value<=1: raise ContractError("invalid LOSO estimate")
            results.append({"deleted_speaker":speaker,"remaining_speaker_count":8,"estimate":value,"status":"complete"})
        except Exception as exc: results.append({"deleted_speaker":speaker,"remaining_speaker_count":8,"estimate":None,"status":"failed",**failure_record(exc)})
    return {"deletion_count":9,"results":results,"technical_review_required":any(x["status"]!="complete" for x in results)}

def lexical_balance(rows,feature,condition):
    cells=defaultdict(list)
    for row in valid_rows(rows,feature,condition,False): cells[(row["speaker_id"],row["normalized_word"])].append(row)
    supported=defaultdict(set)
    for (speaker,word),tokens in cells.items():
        if len(tokens)>=3: supported[word].add(speaker)
    words={word for word,speakers in supported.items() if len(speakers)>=8}; out=[]
    for (speaker,word),tokens in sorted(cells.items()):
        if word in words and len(tokens)>=3: out.extend(sorted(tokens,key=lambda r:(str(r["utterance_id"]),int(r["word_index"])))[:3])
    return out,words

def lexical_composition_sensitivity(rows,feature,fit_fn=fit_condition_reml,primary_mean=None):
    fits=[]; support={}
    for condition in CONDITIONS:
        balanced,words=lexical_balance(rows,feature,condition); support[condition]={"retained_words":len(words),"tokens":len(balanced)}
        if not balanced:return {"status":"not_estimable","reason":"structural_support_insufficient","support":support,"part_of_gate_e":False}
        try: fits.append(fit_fn(balanced,feature,condition))
        except Exception as exc:return {"status":"not_estimable","reason":type(exc).__name__,"support":support,"part_of_gate_e":False}
    value=mean(float(x["repeatability"]) for x in fits)
    return {"status":"estimable","condition_estimates":fits,"equal_condition_mean":value,"delta_from_primary":None if primary_mean is None else value-primary_mean,"support":support,"part_of_gate_e":False,"rules":{"minimum_tokens_per_speaker_word_condition":3,"minimum_speakers_per_word_condition":8,"tokens_retained_per_cell":3,"ordering":["utterance_id","word_index"]}}

def feature_support(rows,feature):
    eligible=[r for r in rows if r.get("alignment_status")=="aligned" and r.get("condition") in CONDITIONS and r.get("speaker_id") in ELIGIBLE_ESD_SPEAKERS]; finite=[]
    for row in eligible:
        try:value=float(row.get(feature))
        except (TypeError,ValueError):continue
        if math.isfinite(value):finite.append(row)
    counts=Counter((r["speaker_id"],r["normalized_word"],r["condition"]) for r in finite); reps=[n for n in counts.values() if n>=2]
    speakers={s for (s,_,_),n in counts.items() if n>=2}
    return {"analysis_universe_tokens":len(eligible),"feature_present_tokens":len(finite),"matched_speaker_count":len(speakers),"eligible_speaker_count":len(speakers),"matched_cell_count":len(reps),"median_repetitions":median(reps) if reps else 0,"feature_missingness":1-len(finite)/len(eligible) if eligible else 1.}

def evaluate_feature_gate(feature,support,primary,bootstrap,loso,technical_defect=False):
    reasons=[]
    matched_speakers=support.get("matched_speaker_count",support.get("eligible_speaker_count",0))
    checks=((matched_speakers<8,"matched_speakers_below_8"),(support["matched_cell_count"]<500,"matched_cells_below_500"),(support["median_repetitions"]<3,"median_repetitions_below_3"),(support["feature_missingness"]>.10,"missingness_above_0.10"))
    reasons.extend(name for failed,name in checks if failed)
    if primary is None: reasons.append("primary_non_estimable")
    elif primary["equal_condition_mean"]<.70: reasons.append("primary_below_0.70")
    if bootstrap.get("successful_replicates",0)<950: reasons.append("bootstrap_success_below_950")
    ci=bootstrap.get("ci_95"); lower=ci[0] if isinstance(ci,(list,tuple)) and ci else None
    if lower is None or not math.isfinite(float(lower)) or float(lower)<.50: reasons.append("bootstrap_lower_below_0.50_or_absent")
    results=loso.get("results",[])
    if len(results)!=9 or any(x.get("status")!="complete" for x in results): reasons.append("loso_incomplete")
    elif any(float(x["estimate"])<.50 for x in results): reasons.append("loso_below_0.50")
    if technical_defect: reasons.append("confirmed_technical_defect")
    review=any(x in reasons for x in ("primary_non_estimable","bootstrap_success_below_950","loso_incomplete","confirmed_technical_defect"))
    return {"feature":feature,"status":"technical_review_no_decision" if review else ("pass" if not reasons else "fail"),"passed":not review and not reasons,"gate_e_evaluable":not review,"criteria_failures":reasons,"thresholds":dict(GATE_E_THRESHOLDS)}

def overall_gate_e(feature_gates):
    expected=set(COMPARABLE_FEATURES); records=defaultdict(list); malformed=[]
    try: supplied=list(feature_gates)
    except TypeError: supplied=[]; malformed.append("feature_gates_not_iterable")
    for index,record in enumerate(supplied):
        if not isinstance(record,dict) or not isinstance(record.get("feature"),str):
            malformed.append(f"record_{index}_malformed"); continue
        records[record["feature"]].append(record)
    unknown=sorted(set(records)-expected)
    duplicates=sorted(feature for feature,items in records.items() if len(items)!=1)
    missing=sorted(expected-set(records))
    by={feature:records[feature][0] for feature in expected if len(records.get(feature,()))==1}
    states={}; malformed_features=[]
    for feature,record in by.items():
        status=record.get("status"); passed=record.get("passed")
        if status=="pass" and passed is True: states[feature]="pass"
        elif status=="fail" and passed is False: states[feature]="fail"
        elif status=="technical_review_no_decision" and passed is False: states[feature]="technical_review_no_decision"
        else: malformed_features.append(feature)
    technical=set(missing)|set(duplicates)|set(malformed_features)
    technical.update(feature for feature,state in states.items() if state=="technical_review_no_decision")
    passing={feature for feature,state in states.items() if state=="pass"}
    failed={feature for feature,state in states.items() if state=="fail"}
    current=len(passing); maximum=current+len(technical)
    malformed_input=bool(malformed or unknown or duplicates or missing or malformed_features)
    f0_state=states.get("f0_mean_hz","technical_review_no_decision")
    if f0_state=="fail": status="fail"
    elif f0_state=="technical_review_no_decision" or malformed_input: status="technical_review_no_decision"
    elif current>=3: status="pass"
    elif maximum>=3: status="technical_review_no_decision"
    else: status="fail"
    return {"status":status,"passed":status=="pass","passing_features":sorted(passing),
      "technical_review_features":sorted(technical),"definitive_failed_features":sorted(failed),
      "current_pass_count":current,"maximum_possible_pass_count":maximum,
      "required_passing_features":3,"f0_mean_hz_required":True,
      "malformed_input":malformed,"missing_features":missing,"duplicate_features":duplicates,
      "unknown_features":unknown,"malformed_features":sorted(malformed_features)}

def analyze_feature(rows,feature,fit_fn=fit_condition_reml,bootstrap_plan=None):
    try: primary=five_condition_summary(rows,feature,fit_fn)
    except Exception as exc:return {"feature":feature,"status":"technical_review_no_decision",**failure_record(exc),"primary":None}
    bootstrap=speaker_cluster_bootstrap(rows,feature,fit_fn,plan=bootstrap_plan); loso=leave_one_speaker_out(rows,feature,fit_fn); support=feature_support(rows,feature)
    gate=evaluate_feature_gate(feature,support,primary,bootstrap,loso)
    return {"feature":feature,"status":gate["status"],"support":support,"primary":primary,"bootstrap":bootstrap,"leave_one_speaker_out":loso,"lexical_composition_sensitivity":lexical_composition_sensitivity(rows,feature,fit_fn,primary["equal_condition_mean"]),"cross_condition":{"status":"omitted_secondary","part_of_gate_e":False},"gate_e":gate}

def analyze_comparable_features(rows,fit_fn=fit_condition_reml):
    plan=make_bootstrap_plan(); validate_bootstrap_plan(plan)
    return [analyze_feature(rows,feature,fit_fn,bootstrap_plan=plan) for feature in COMPARABLE_FEATURES]

def validate_real_provenance(stage1_manifest,aggregate,closure,expected_closure_sha256):
    for path in (stage1_manifest,aggregate,closure):
        reject_forbidden_path(path)
        if not path.is_file():raise ContractError(f"required input absent: {path}")
    if expected_closure_sha256!=FROZEN_STAGE2B_CLOSURE_SHA256:raise ContractError("expected Stage-2B closure SHA256 is not the frozen closure SHA256")
    if sha256(stage1_manifest)!=APPROVED_STAGE1_MANIFEST_SHA256:raise ContractError("Stage-1 manifest checksum mismatch")
    if sha256(aggregate)!=APPROVED_STAGE2B_AGGREGATE_SHA256:raise ContractError("Stage-2B aggregate checksum mismatch")
    if sha256(closure)!=expected_closure_sha256:raise ContractError("Stage-2B closure checksum mismatch")
    record=json.loads(closure.read_text())
    exact={"stage":"Stage 2B","run":"full_v1_run3","status":"TECHNICAL_EXTRACTION_PASS",
      "approved_stage1_manifest_sha256":APPROVED_STAGE1_MANIFEST_SHA256,
      "aggregate_csv_sha256":APPROVED_STAGE2B_AGGREGATE_SHA256,"gate_e_evaluated":False,
      "reliability_analysis_performed":False,"rows":98897,"unique_sample_ids":15750,
      "duplicate_token_rows":0}
    if any(record.get(key)!=value for key,value in exact.items()):raise ContractError("Stage-2B closure frozen scalar schema mismatch")
    if record.get("alignment_status_counts",{}).get("aligned")!=98897:raise ContractError("Stage-2B closure aligned count mismatch")
    if record.get("feature_status_counts",{}).get("complete")!=98897:raise ContractError("Stage-2B closure complete-feature count mismatch")
    return record

def load_real_input(stage1_manifest,aggregate,closure,expected_closure_sha256):
    validate_real_provenance(stage1_manifest,aggregate,closure,expected_closure_sha256)
    with aggregate.open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
    return residualize_duration_once(rows)

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--stage1-manifest",required=True,type=Path); parser.add_argument("--stage2b-aggregate",required=True,type=Path); parser.add_argument("--stage2b-closure",required=True,type=Path); parser.add_argument("--expected-closure-sha256",required=True); parser.add_argument("--output",required=True,type=Path); args=parser.parse_args(argv)
    reject_forbidden_path(args.output); rows,duration=load_real_input(args.stage1_manifest,args.stage2b_aggregate,args.stage2b_closure,args.expected_closure_sha256)
    reports=analyze_comparable_features(rows); gates=[r.get("gate_e",{"feature":r["feature"],"gate_e_evaluable":False}) for r in reports]
    write_json(args.output,{"analysis":"frozen_stage2c_esd_wordlocal_reml","duration_model":duration,"features":reports,"gate_e":overall_gate_e(gates),"scientific_claims_authorized":False}); return 0
if __name__=="__main__":raise SystemExit(main())
