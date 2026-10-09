#!/usr/bin/env python3
"""Verify MSP model-lock v2 executable hashes at authorization time.

Technical provenance only. Does not open MSP data, audio, acoustic values,
reliability values, emotion annotations, or clinical outcomes.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

class IntegrityError(ValueError): pass

REQUIRED_MODEL_FILES={
 'msp_primary_residualization_v2.py','msp_profile_reml_v1.py','msp_lexical_secondary_v1.py','msp_reliability_v2.py','msp_acoustic_adapter_v1.py',
 'test_msp_primary_residualization_v2.py','test_msp_profile_reml_v1.py','test_msp_lexical_secondary_v1.py','test_msp_reliability_v2.py','test_msp_acoustic_adapter_v1.py',
}
REFERENCE_KEYS={'esd_frozen_acoustic_adapter_sha256','seamless_frozen_pipeline_sha256'}

def sha256(path):
    p=Path(path)
    if not p.is_file(): raise IntegrityError(f'missing_file:{p}')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def verify(lock_json,code_root,esd_reference,seamless_reference):
    lock=json.loads(Path(lock_json).read_text(encoding='utf-8'))
    if lock.get('status')!='frozen' or lock.get('human_approved') is not True or lock.get('implementation_status')!='frozen': raise IntegrityError('model_lock_not_frozen')
    impl=lock.get('implementation') or {}; expected=impl.get('files_sha256')
    if not isinstance(expected,dict) or set(expected)!=(REQUIRED_MODEL_FILES|REFERENCE_KEYS): raise IntegrityError('expected_hash_set_mismatch')
    if impl.get('synthetic_test_total')!=43 or impl.get('synthetic_test_passed')!=43: raise IntegrityError('synthetic_test_lock_mismatch')
    root=Path(code_root); actual={}
    for name in sorted(REQUIRED_MODEL_FILES):
        path=root/('tests' if name.startswith('test_') else 'tools')/name
        actual[name]=sha256(path)
    actual['esd_frozen_acoustic_adapter_sha256']=sha256(esd_reference)
    actual['seamless_frozen_pipeline_sha256']=sha256(seamless_reference)
    mismatch=sorted(k for k in expected if actual.get(k)!=expected[k])
    if mismatch: raise IntegrityError('hash_mismatch:'+','.join(mismatch))
    return {
      'audit':'msp_implementation_integrity_v1','status':'PASS','model_lock_status':'frozen','implementation_status':'frozen',
      'synthetic_tests_locked':'43/43','verified_hash_count':len(actual),'all_hashes_match':True,
      'actual_sha256':actual,'mismatched_keys':[],
      'msp_data_accessed':False,'audio_accessed':False,'acoustic_values_accessed':False,'reliability_values_accessed':False,'emotion_fields_used':False,'clinical_outcomes_accessed':False,
    }

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--model-lock',type=Path,required=True); p.add_argument('--code-root',type=Path,required=True); p.add_argument('--esd-reference',type=Path,required=True); p.add_argument('--seamless-reference',type=Path,required=True); p.add_argument('--safe-json',type=Path,required=True); a=p.parse_args(argv)
    out=verify(a.model_lock,a.code_root,a.esd_reference,a.seamless_reference); a.safe_json.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8'); return 0
if __name__=='__main__': raise SystemExit(main())
