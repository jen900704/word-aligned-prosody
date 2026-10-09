#!/usr/bin/env python3
"""Deterministic technical merge for authorized MSP acoustic extraction shards."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,tempfile
from pathlib import Path
from tools.msp_acoustic_extraction_runner_v2 import OUT_FIELDS,load_final,load_tokens,shard_for,sha256

class MergeError(ValueError): pass

def atomic_csv(path:Path,rows):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=OUT_FIELDS,extrasaction='raise',lineterminator='\n'); w.writeheader(); w.writerows(rows)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def merge(shard_dir:Path,num_shards:int,preflight:Path,alignment_safe:Path,final_safe:Path,token_csv:Path,final_csv:Path,output_csv:Path,safe_json:Path):
    if isinstance(num_shards,bool) or not isinstance(num_shards,int) or num_shards<1: raise MergeError('invalid_num_shards')
    selected=load_final(final_csv,final_safe); tokens=load_tokens(token_csv,alignment_safe,selected)
    pf_sha=sha256(preflight); al_sha=sha256(alignment_safe); fs_sha=sha256(final_safe); tok_sha=sha256(token_csv); fin_sha=sha256(final_csv)
    expected={(fn,str(t['word_index_int'])) for fn,items in tokens.items() for t in items}
    rows=[]; seen=set(); failed=0; shard_output_hashes={}
    for i in range(num_shards):
        csvp=shard_dir/f'shard_{i:02d}.csv'; safep=shard_dir/f'shard_{i:02d}_SAFE.json'
        if not csvp.is_file() or not safep.is_file(): raise MergeError('missing_shard_artifact')
        z=json.loads(safep.read_text(encoding='utf-8'))
        checks=(z.get('audit')=='msp_acoustic_extraction_shard_v2',z.get('status')=='PASS',z.get('shard_id')==i,z.get('num_shards')==num_shards,
                z.get('output_sha256')==sha256(csvp),z.get('preflight_sha256')==pf_sha,z.get('alignment_safe_sha256')==al_sha,
                z.get('final_scope_safe_sha256')==fs_sha,z.get('token_manifest_sha256')==tok_sha,z.get('final_private_manifest_sha256')==fin_sha,
                z.get('emotion_fields_used') is False,z.get('reliability_values_accessed') is False,z.get('clinical_outcomes_accessed') is False,
                z.get('scientific_choices_made_by_runner') is False)
        if not all(checks): raise MergeError('shard_safe_chain_mismatch')
        expected_files={fn for fn in selected if shard_for(fn,num_shards)==i}
        if z.get('files_in_shard')!=len(expected_files) or z.get('selected_files_total')!=len(selected): raise MergeError('shard_file_count_mismatch')
        local_rows=0; local_failed=0
        with csvp.open(newline='',encoding='utf-8') as f:
            r=csv.DictReader(f)
            if list(r.fieldnames or [])!=OUT_FIELDS: raise MergeError('shard_output_schema_mismatch')
            for x in r:
                fn=x['filename']; key=(fn,x['word_index'])
                if fn not in expected_files or key not in expected or key in seen: raise MergeError('unexpected_or_duplicate_token_key')
                meta=selected[fn]
                if x['speaker_id']!=meta['speaker_id'] or x['recording_unit_id']!=meta['recording_unit_id'] or x['primary_half']!=meta['primary_half']: raise MergeError('shard_metadata_mismatch')
                if x['acoustic_status'] not in {'complete','failed'}: raise MergeError('invalid_acoustic_status')
                if x['acoustic_status']=='failed': local_failed+=1
                seen.add(key); rows.append(x); local_rows+=1
        if local_rows!=z.get('token_rows') or local_failed!=z.get('acoustic_failed_tokens'): raise MergeError('shard_safe_count_mismatch')
        failed+=local_failed; shard_output_hashes[f'{i:02d}']=sha256(csvp)
    if seen!=expected: raise MergeError('merged_token_coverage_mismatch')
    rows.sort(key=lambda x:(x['filename'],int(x['word_index'])))
    atomic_csv(output_csv,rows)
    safe={"audit":"msp_acoustic_merge_v1","status":"PASS","num_shards":num_shards,"selected_files_total":len(selected),
          "expected_token_rows":len(expected),"merged_token_rows":len(rows),"acoustic_complete_tokens":len(rows)-failed,"acoustic_failed_tokens":failed,
          "output_sha256":sha256(output_csv),"preflight_sha256":pf_sha,"alignment_safe_sha256":al_sha,"final_scope_safe_sha256":fs_sha,
          "token_manifest_sha256":tok_sha,"final_private_manifest_sha256":fin_sha,"shard_output_sha256":shard_output_hashes,
          "emotion_fields_used":False,"acoustic_values_accessed":True,"acoustic_values_emitted_in_safe":False,"reliability_values_accessed":False,
          "clinical_outcomes_accessed":False,"scientific_choices_made_by_merge":False}
    safe_json.parent.mkdir(parents=True,exist_ok=True); safe_json.write_text(json.dumps(safe,indent=2,sort_keys=True)+'\n',encoding='utf-8'); return safe

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--shard-dir',type=Path,required=True); p.add_argument('--num-shards',type=int,required=True)
    for arg in ('preflight','alignment-safe','final-safe','token-csv','final-csv','output-csv','safe-json'): p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args(argv); merge(a.shard_dir,a.num_shards,a.preflight,a.alignment_safe,a.final_safe,a.token_csv,a.final_csv,a.output_csv,a.safe_json); return 0
if __name__=='__main__': raise SystemExit(main())
