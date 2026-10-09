#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
from esd_stage2c_parallel import merge_bootstrap_chunks
ALL=("f0_mean_hz","f0_robust_range_hz","energy_db","duration_log_syllable_residual")

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--run-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--features',nargs='+',choices=ALL,default=list(ALL)); a=p.parse_args(argv)
    out={'audit':'esd_bootstrap_coverage_safe_v1','status':'PASS','features':{},'scientific_values_emitted':False,'ci_emitted':False,'reliability_values_emitted':False}
    for f in a.features:
        root=a.run_root/'chunks'/f; paths=sorted(root.glob('chunk_*.json')); chunks=[json.loads(x.read_text()) for x in paths]
        z=merge_bootstrap_chunks(f,chunks)
        out['features'][f]={
            'chunk_count':len(paths),'requested_replicates':z['requested_replicates'],
            'successful_replicates':z['successful_replicates'],'failed_replicates':z['failed_replicates'],
            'minimum_success_for_no_review':z['minimum_success_for_no_review'],
            'technical_review_required':z['technical_review_required'],
            'replacement_draws_generated':z['replacement_draws_generated'],'seed':z['seed'],'plan_sha256':z['plan_sha256'],
            'chunk_manifest_sha256':hashlib.sha256(('\n'.join(x.name+':'+sha(x) for x in paths)+'\n').encode()).hexdigest()}
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8'); return 0
if __name__=='__main__': raise SystemExit(main())
