#!/usr/bin/env python3
"""SAFE-only MSP primary-completion validator; never reads private results."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
class MSPCompletionError(ValueError): pass

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def validate(primary_safe, merge_safe, production_safe, lexical_safe=None):
    p,m,r=load(primary_safe),load(merge_safe),load(production_safe)
    if p.get('audit')!='msp_final_analysis_technical_safe_v1' or p.get('status') not in {'PASS','TECHNICAL_REVIEW'}: raise MSPCompletionError('primary_safe')
    for k in ('reliability_values_emitted','confidence_interval_values_emitted','gate_m_status_emitted','gate_m_status_inspected_for_safe','speaker_ids_emitted'):
        if p.get(k) is not False: raise MSPCompletionError('primary_blinding:'+k)
    if m.get('audit')!='msp_acoustic_merge_v1' or m.get('status')!='PASS': raise MSPCompletionError('merge_safe')
    if m.get('reliability_values_accessed') is not False or m.get('clinical_outcomes_accessed') is not False: raise MSPCompletionError('merge_blinding')
    if r.get('audit')!='msp_production_readiness_v1' or r.get('status')!='PASS' or not all(r.get('checks',{}).values()): raise MSPCompletionError('production_safe')
    lexical_state='PENDING_OR_NOT_RUN'
    lexical_sha=None
    if lexical_safe is not None and Path(lexical_safe).is_file():
        l=load(lexical_safe)
        if l.get('audit')!='msp_crossfitted_lexical_secondary_technical_safe_v1' or l.get('status') not in {'PASS','TECHNICAL_REVIEW'}: raise MSPCompletionError('lexical_safe')
        if l.get('reliability_values_emitted') is not False or l.get('gate_m_status_emitted') is not False or l.get('secondary_only') is not True or l.get('can_modify_gate_m') is not False: raise MSPCompletionError('lexical_blinding')
        lexical_state='COMPLETE' if l['status']=='PASS' else 'TECHNICAL_REVIEW'
        lexical_sha=sha(lexical_safe)
    primary_state='READY_FOR_RESULT_REVIEW' if p['status']=='PASS' else 'TECHNICAL_REVIEW_REQUIRED'
    return {'audit':'msp_experiment_completion_safe_v1','status':'PASS' if p['status']=='PASS' else 'TECHNICAL_REVIEW','primary_result_review_state':primary_state,'lexical_secondary_state':lexical_state,'primary_features':p.get('features',{}),'acoustic_merged_token_rows':m.get('merged_token_rows'),'acoustic_failed_tokens':m.get('acoustic_failed_tokens'),'primary_safe_sha256':sha(primary_safe),'merge_safe_sha256':sha(merge_safe),'production_safe_sha256':sha(production_safe),'lexical_safe_sha256':lexical_sha,'reliability_values_emitted':False,'confidence_interval_values_emitted':False,'gate_m_status_emitted':False,'speaker_ids_emitted':False,'private_results_read':False,'scientific_claims_authorized':False}

def main(argv=None):
    q=argparse.ArgumentParser(); q.add_argument('--primary-safe',type=Path,required=True); q.add_argument('--merge-safe',type=Path,required=True); q.add_argument('--production-safe',type=Path,required=True); q.add_argument('--lexical-safe',type=Path); a=q.parse_args(argv); print(json.dumps(validate(a.primary_safe,a.merge_safe,a.production_safe,a.lexical_safe),indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
