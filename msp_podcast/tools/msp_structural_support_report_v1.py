#!/usr/bin/env python3
"""Outcome-blind structural support report after frozen MSP final-scope selection.

This report cannot change selection. It compares selected speakers with all
content-QA-qualified speakers using only eligible utterance counts and recording
unit counts, and quantifies how many distinct unordered partitions the frozen
25 alternative salts generate.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from tools.msp_structural_v1 import MIN_UTTERANCES,MIN_RECORDING_UNITS,MAX_SPEAKERS,ALT_SALTS,speaker_order_key,split_recording_units

class SupportReportError(ValueError): pass
REQ={'filename','speaker_id','recording_unit_id'}; ALLOWED=REQ|{'split_set'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def summary(vals):
    x=np.asarray(list(vals),dtype=float)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all(): raise SupportReportError('invalid_summary_input')
    q=np.percentile(x,[25,50,75],method='linear')
    return {'n':int(len(x)),'min':float(x.min()),'q1':float(q[0]),'median':float(q[1]),'q3':float(q[2]),'max':float(x.max())}

def canonical_partition(a,b):
    aa=tuple(sorted(a)); bb=tuple(sorted(b)); return tuple(sorted((aa,bb)))

def build_report(structural_csv,eligible_txt,final_private_csv):
    eligible=[Path(x.strip()).name for x in Path(eligible_txt).read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(eligible)!=len(set(eligible)): raise SupportReportError('duplicate_eligible_filename')
    eligible=set(eligible); counts=defaultdict(int); units=defaultdict(set); seen=set()
    with Path(structural_csv).open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f); fields=set(r.fieldnames or [])
        if not REQ<=fields or fields-ALLOWED: raise SupportReportError('structural_schema')
        for row in r:
            fn=row['filename']
            if fn in seen: raise SupportReportError('duplicate_structural_filename')
            seen.add(fn)
            if fn in eligible:
                s=row['speaker_id']; u=row['recording_unit_id']
                if not s or not u: raise SupportReportError('empty_structural_id')
                counts[s]+=1; units[s].add(u)
    if not eligible<=seen: raise SupportReportError('eligible_not_in_structural')
    qualified=sorted([s for s in counts if counts[s]>=MIN_UTTERANCES and len(units[s])>=MIN_RECORDING_UNITS],key=speaker_order_key)
    expected_selected=tuple(qualified[:MAX_SPEAKERS])
    with Path(final_private_csv).open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f); rows=list(r); fields=set(r.fieldnames or [])
    if not rows or not {'filename','speaker_id','recording_unit_id','primary_half'}<=fields or any(f'alt_{i:02d}_half' not in fields for i in range(25)): raise SupportReportError('final_scope_schema')
    selected=tuple(sorted({x['speaker_id'] for x in rows},key=speaker_order_key))
    if set(selected)!=set(expected_selected): raise SupportReportError('selected_set_mismatch')
    if any(x['filename'] not in eligible for x in rows): raise SupportReportError('final_scope_contains_ineligible_filename')
    selected_set=set(selected)
    q_utt=[counts[s] for s in qualified]; s_utt=[counts[s] for s in selected]
    q_units=[len(units[s]) for s in qualified]; s_units=[len(units[s]) for s in selected]
    distinct=[]
    for s in selected:
        parts=set()
        for salt in ALT_SALTS:
            a,b=split_recording_units(s,units[s],salt); parts.add(canonical_partition(a,b))
        distinct.append(len(parts))
    return {
      'audit':'msp_structural_support_report_v1','status':'PASS',
      'qualified_speakers':len(qualified),'selected_speakers':len(selected),'speaker_cap':MAX_SPEAKERS,
      'qualified_eligible_utterances':summary(q_utt),'selected_eligible_utterances':summary(s_utt),
      'qualified_recording_units':summary(q_units),'selected_recording_units':summary(s_units),
      'qualified_exactly_two_units':sum(v==2 for v in q_units),'selected_exactly_two_units':sum(v==2 for v in s_units),
      'selected_at_least_three_units':sum(v>=3 for v in s_units),
      'selected_distinct_unordered_alt_partitions':summary(distinct),
      'reselection_performed':False,'selection_change_authorized':False,'significance_tests_run':False,
      'speaker_ids_emitted':False,'emotion_fields_used':False,'audio_accessed':False,'acoustic_values_accessed':False,'reliability_values_accessed':False,'clinical_outcomes_accessed':False,
      'structural_sha256':sha(structural_csv),'eligible_list_sha256':sha(eligible_txt),'final_private_manifest_sha256':sha(final_private_csv),
    }

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--structural-csv',type=Path,required=True); p.add_argument('--eligible-filenames',type=Path,required=True); p.add_argument('--final-private-manifest',type=Path,required=True); p.add_argument('--safe-json',type=Path,required=True); a=p.parse_args(argv)
    out=build_report(a.structural_csv,a.eligible_filenames,a.final_private_manifest); a.safe_json.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8'); return 0
if __name__=='__main__': raise SystemExit(main())
