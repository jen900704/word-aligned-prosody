#!/usr/bin/env python3
"""Result-blind final technical QA for frozen ESD Stage-2C parallel rescue."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
EXPECTED=('f0_mean_hz','f0_robust_range_hz','energy_db','duration_log_syllable_residual')
PLAN='b89fd97bf340b0ea305e4f03791146e9587fa29bbd000e374d145946eba824bd'
class ESDQAError(ValueError): pass

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def validate(path):
 z=json.loads(Path(path).read_text(encoding='utf-8'))
 if z.get('analysis')!='frozen_stage2c_esd_wordlocal_reml' or z.get('scientific_claims_authorized') is not False: raise ESDQAError('top_level')
 o=z.get('orchestration_metadata',{})
 if o.get('purpose')!='computational_parallelization_only' or o.get('bootstrap_partitioning')!='deterministic_half_open_replicate_ranges' or o.get('bootstrap_plan_sha256')!=PLAN or o.get('scientific_semantics_changed') is not False: raise ESDQAError('orchestration')
 feats=z.get('features'); names=[x.get('feature') if isinstance(x,dict) else None for x in feats] if isinstance(feats,list) else []
 if len(names)!=4 or set(names)!=set(EXPECTED) or len(set(names))!=4: raise ESDQAError('feature_identity')
 out={}
 for r in feats:
  f=r['feature']; b=r.get('bootstrap'); lo=r.get('leave_one_speaker_out')
  if b is None:
   out[f]={'bootstrap_present':False,'requested':None,'successful':None,'failed':None,'technical_review_required':None,'loso_deletion_count':None}
   continue
  if b.get('type')!='speaker_cluster_nonparametric' or b.get('seed')!=13 or b.get('plan_sha256')!=PLAN or b.get('requested_replicates')!=1000 or b.get('replacement_draws_generated') is not False: raise ESDQAError(f'bootstrap_contract:{f}')
  success=b.get('successful_replicates'); failed=b.get('failed_replicates')
  if not isinstance(success,int) or not isinstance(failed,int) or success+failed!=1000: raise ESDQAError(f'bootstrap_counts:{f}')
  deletion=lo.get('deletion_count') if isinstance(lo,dict) else None
  if deletion!=9: raise ESDQAError(f'loso_count:{f}')
  out[f]={'bootstrap_present':True,'requested':1000,'successful':success,'failed':failed,'technical_review_required':bool(b.get('technical_review_required')),'minimum_success_for_no_review':b.get('minimum_success_for_no_review'),'replacement_draws_generated':False,'loso_deletion_count':9}
 return {'audit':'esd_final_technical_qa_safe_v1','status':'PASS','features':out,'final_artifact_sha256':sha(path),'feature_count':4,'feature_names':list(EXPECTED),'orchestration_purpose':'computational_parallelization_only','scientific_semantics_changed':False,'reliability_values_emitted':False,'confidence_interval_values_emitted':False,'gate_e_status_emitted':False,'gate_e_status_inspected':False,'scientific_claims_authorized':False}

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--final-json',type=Path,required=True); a=p.parse_args(argv); print(json.dumps(validate(a.final_json),indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
