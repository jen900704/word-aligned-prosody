#!/usr/bin/env python3
"""Frozen primary MSP Gate-M analysis orchestrator with result-blind SAFE receipt."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from tools.msp_acoustic_extraction_runner_v2 import OUT_FIELDS,sha256
from tools.msp_primary_residualization_v2 import ALL_FEATURES,DURATION_FEATURE,F0_ENERGY_FEATURES,fit_primary_residuals
from tools.msp_reliability_v2 import reliability,bootstrap_from_pairs,loso,alternative_split_metrics,recording_volume_sensitivity,gate_m_feature_conditions
from tools.msp_secondary_distribution_v1 import descriptive_reliability

FEATURE_SOURCE={'f0_mean_hz':'f0_mean_hz','f0_robust_range_hz':'f0_robust_range_hz','energy_db':'energy_db',DURATION_FEATURE:'word_duration_sec'}
F0_FEATURES=frozenset({'f0_mean_hz','f0_robust_range_hz'})
class FinalAnalysisError(ValueError): pass

def load_merged(path:Path,safe_path:Path):
    z=json.loads(safe_path.read_text(encoding='utf-8'))
    if z.get('audit')!='msp_acoustic_merge_v1' or z.get('status')!='PASS' or z.get('output_sha256')!=sha256(path): raise FinalAnalysisError('merge_safe_hash_chain')
    if z.get('reliability_values_accessed') is not False or z.get('clinical_outcomes_accessed') is not False: raise FinalAnalysisError('merge_provenance_invalid')
    with path.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        if list(r.fieldnames or [])!=OUT_FIELDS: raise FinalAnalysisError('merged_schema')
        rows=list(r)
    if len(rows)!=z.get('merged_token_rows') or not rows: raise FinalAnalysisError('merged_row_count')
    return rows,z

def _finite(value):
    try: x=float(value)
    except Exception: return None
    return x if math.isfinite(x) else None

def prepare_feature(rows,feature):
    if feature not in ALL_FEATURES: raise FinalAnalysisError('unknown_feature')
    vals=[]; pos=[]; syll=[]; kept=[]; src=FEATURE_SOURCE[feature]
    for r in rows:
        p=_finite(r.get('relative_position'))
        if p is None or p<0 or p>1: raise FinalAnalysisError('relative_position_invalid')
        v=_finite(r.get(src))
        if feature in F0_ENERGY_FEATURES:
            if r.get('acoustic_status')=='failed' and any(str(r.get(k,'')) for k in ('f0_mean_hz','f0_robust_range_hz','energy_db')): raise FinalAnalysisError('failed_acoustic_row_contains_value')
            if v is None: continue
            vals.append(v); pos.append(p); kept.append(r)
        else:
            sy=_finite(r.get('authoritative_syllable_count'))
            if v is None or v<=0 or sy is None or sy<=0 or sy!=math.floor(sy): continue
            vals.append(v); pos.append(p); syll.append(int(sy)); kept.append(r)
    if not kept: raise FinalAnalysisError('no_valid_feature_rows')
    fit=fit_primary_residuals(feature,vals,pos,syll if feature==DURATION_FEATURE else None)
    out=[]
    for r,res in zip(kept,fit.residuals): out.append({**r,'residual':float(res)})
    return out,fit

def _compact_rel(z): return {k:v for k,v in z.items() if k!='pairs'}
def _compact_alt(z,include_passes=True):
    top={k:v for k,v in z.items() if k!='splits' and (include_passes or k!='passes')}
    return {**top,'splits':[_compact_rel(x) for x in z['splits']]}
def _count_units(rows):
    d=defaultdict(set)
    for r in rows: d[r['speaker_id']].add(r['recording_unit_id'])
    return {s:len(u) for s,u in d.items()}
def _support_from_pairs(primary,total_rows,valid_rows):
    pairs=primary['pairs']; a=[p[3] for p in pairs]; b=[p[4] for p in pairs]
    def sm(x):
        if not x:return None
        q=np.percentile(x,[25,50,75],method='linear'); return {'min':int(min(x)),'q1':float(q[0]),'median':float(q[1]),'q3':float(q[2]),'max':int(max(x))}
    return {'total_selected_tokens':len(total_rows),'valid_feature_tokens':len(valid_rows),'feature_missing_tokens':len(total_rows)-len(valid_rows),
            'feature_missingness':1-len(valid_rows)/len(total_rows),'paired_speakers':len(pairs),'paired_half_A_token_counts':sm(a),'paired_half_B_token_counts':sm(b)}

def _analysis_status(primary,bootstrap):
    if primary.get('paired_speakers',0)<50: return 'complete'
    return 'complete' if bootstrap.get('technical_adequacy') else 'technical_review_no_decision'

def analyze_feature(all_rows,feature):
    residual_rows,fit=prepare_feature(all_rows,feature)
    primary=reliability(residual_rows)
    if not primary['pairs']:
        boot={'requested':1000,'successful':0,'failed':1000,'minimum_success':950,'technical_adequacy':False,'ci_95':None,'replacement_draws_generated':False,'seed':13}
        lo={'base':None,'max_abs_change':None,'changes':[],'passes':False}
    else:
        boot=bootstrap_from_pairs(primary['pairs']); lo=loso(primary['pairs'])
    alt=alternative_split_metrics(residual_rows); vol=recording_volume_sensitivity(residual_rows); gate=gate_m_feature_conditions(primary,boot,alt,lo,vol)
    analysis_status=_analysis_status(primary,boot)
    unit_counts=_count_units(all_rows); speakers3={s for s,n in unit_counts.items() if n>=3}; subgroup=[r for r in residual_rows if r['speaker_id'] in speakers3]
    subgroup_alt=alternative_split_metrics(subgroup) if subgroup else {'valid_splits':0,'median':None,'iqr':None,'min':None,'max':None,'passes':False,'splits':[]}
    return {'feature':feature,'status':analysis_status,'primary':_compact_rel(primary),'bootstrap':boot,'loso':lo,'alternative_splits':_compact_alt(alt),'recording_volume':vol,'gate_m_feature':gate,
            'support':_support_from_pairs(primary,all_rows,residual_rows),'nuisance_beta':[float(x) for x in fit.beta],
            'secondary_distribution':{'iqr':descriptive_reliability(residual_rows,'iqr'),'mad_unscaled':descriptive_reliability(residual_rows,'mad')},
            'at_least_3_units_subgroup':{'selected_speakers_with_at_least_3_units':len(speakers3),'alternative_splits':_compact_alt(subgroup_alt,include_passes=False),'gate_threshold_applied':False},
            'residual_rows_count':len(residual_rows)}

def overall_gate(feature_reports):
    state={}
    for f in ALL_FEATURES:
        r=feature_reports.get(f)
        if not isinstance(r,dict) or r.get('status')!='complete': state[f]='technical_review_no_decision'
        else: state[f]='pass' if r['gate_m_feature']['passes'] else 'fail'
    passes={f for f,s in state.items() if s=='pass'}; technical={f for f,s in state.items() if s=='technical_review_no_decision'}
    if len(passes)>=3 and passes&F0_FEATURES: status='pass'
    elif len(passes)+len(technical)>=3 and (passes&F0_FEATURES or technical&F0_FEATURES): status='technical_review_no_decision'
    else: status='fail'
    return {'status':status,'passes':status=='pass','feature_states':state,'criterion':'at_least_three_features_and_at_least_one_f0'}

def analyze_rows(rows):
    reports={}
    for f in sorted(ALL_FEATURES):
        try: reports[f]=analyze_feature(rows,f)
        except Exception as e: reports[f]={'feature':f,'status':'technical_review_no_decision','error_type':type(e).__name__}
    return {'analysis':'msp_final_gate_m_v1','features':reports,'gate_m':overall_gate(reports),'scientific_claims_authorized':False,
            'emotion_fields_used':False,'clinical_outcomes_accessed':False}

def safe_receipt(result,private_path,merged_path,merge_safe_path):
    feats={}
    for f,r in result['features'].items():
        if r.get('status')!='complete': feats[f]={'status':'technical_review_no_decision','error_type':r.get('error_type')}; continue
        feats[f]={'status':'complete','valid_feature_tokens':r['support']['valid_feature_tokens'],'paired_speakers':r['support']['paired_speakers'],
                  'bootstrap_requested':r['bootstrap']['requested'],'bootstrap_successful':r['bootstrap']['successful'],'bootstrap_failed':r['bootstrap']['failed'],
                  'alternative_splits_valid':r['alternative_splits']['valid_splits'],'loso_deletions':len(r['loso']['changes']),
                  'secondary_distribution_present':set(r['secondary_distribution'])=={'iqr','mad_unscaled'},'subgroup_diagnostic_present':True}
    return {'audit':'msp_final_analysis_technical_safe_v1','status':'PASS' if all(x['status']=='complete' for x in feats.values()) else 'TECHNICAL_REVIEW',
            'features':feats,'private_result_sha256':sha256(private_path),'merged_acoustic_sha256':sha256(merged_path),'merge_safe_sha256':sha256(merge_safe_path),
            'reliability_values_emitted':False,'confidence_interval_values_emitted':False,'gate_m_status_emitted':False,'gate_m_status_inspected_for_safe':False,
            'speaker_ids_emitted':False,'emotion_fields_used':False,'clinical_outcomes_accessed':False,'scientific_claims_authorized':False}

def run(merged_path:Path,merge_safe_path:Path,private_json:Path,safe_json:Path):
    rows,_=load_merged(merged_path,merge_safe_path); result=analyze_rows(rows)
    private_json.parent.mkdir(parents=True,exist_ok=True); private_json.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.chmod(private_json,0o600)
    safe=safe_receipt(result,private_json,merged_path,merge_safe_path); safe_json.parent.mkdir(parents=True,exist_ok=True); safe_json.write_text(json.dumps(safe,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.chmod(safe_json,0o600); return result,safe

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--merged-acoustics',type=Path,required=True); p.add_argument('--merge-safe',type=Path,required=True); p.add_argument('--private-result',type=Path,required=True); p.add_argument('--safe-json',type=Path,required=True); a=p.parse_args(argv)
    run(a.merged_acoustics,a.merge_safe,a.private_result,a.safe_json); return 0
if __name__=='__main__': raise SystemExit(main())
