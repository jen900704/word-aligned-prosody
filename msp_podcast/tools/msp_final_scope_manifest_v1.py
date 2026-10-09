#!/usr/bin/env python3
"""Build the frozen final MSP scope from a future eligible-filename list.

No transcript, TextGrid, audio, emotion, acoustic, reliability, or outcome content is opened.
The eligible list must come from a separately frozen transcript/alignment QA step.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
from tools.msp_structural_v1 import MAX_SPEAKERS,MIN_RECORDING_UNITS,MIN_UTTERANCES,PRIMARY_SALT,ALT_SALTS,speaker_order_key,split_recording_units

REQ={"filename","speaker_id","recording_unit_id"}; ALLOWED=REQ|{"split_set"}
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def load_eligible(p):
 vals=[Path(x.strip()).name for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
 if len(vals)!=len(set(vals)): raise ValueError('eligible filename list contains duplicates')
 return set(vals)
def build(structural_csv,eligible_txt):
 eligible=load_eligible(eligible_txt); rows=[]; seen=set()
 with structural_csv.open(newline='',encoding='utf-8') as f:
  r=csv.DictReader(f); fields=set(r.fieldnames or [])
  if not REQ<=fields or fields-ALLOWED: raise ValueError('structural CSV schema mismatch')
  for x in r:
   fn=x['filename']
   if fn in seen: raise ValueError('duplicate structural filename')
   seen.add(fn)
   if fn in eligible: rows.append({k:x.get(k,'') for k in ('filename','speaker_id','split_set','recording_unit_id')})
 if not eligible<=seen: raise ValueError('eligible list contains filename absent from structural projection')
 counts=defaultdict(int); units=defaultdict(set)
 for x in rows: counts[x['speaker_id']]+=1; units[x['speaker_id']].add(x['recording_unit_id'])
 qualifying=[s for s in counts if counts[s]>=MIN_UTTERANCES and len(units[s])>=MIN_RECORDING_UNITS]
 selected=tuple(sorted(qualifying,key=speaker_order_key)[:MAX_SPEAKERS]); sel=set(selected)
 splitmaps={}
 for s in selected:
  all_units=units[s]; a,b=split_recording_units(s,all_units,PRIMARY_SALT); m={u:'A' for u in a}; m.update({u:'B' for u in b})
  alt=[]
  for salt in ALT_SALTS:
   aa,bb=split_recording_units(s,all_units,salt); mm={u:'A' for u in aa}; mm.update({u:'B' for u in bb}); alt.append(mm)
  splitmaps[s]=(m,alt)
 out=[]
 for x in rows:
  if x['speaker_id'] not in sel: continue
  pm,alts=splitmaps[x['speaker_id']]; y=dict(x); y['primary_half']=pm[x['recording_unit_id']]
  for i,m in enumerate(alts): y[f'alt_{i:02d}_half']=m[x['recording_unit_id']]
  out.append(y)
 return selected,out,{s:{'eligible_utterances':counts[s],'recording_units':len(units[s])} for s in selected},len(qualifying)
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--structural-csv',type=Path,required=True); p.add_argument('--eligible-filenames',type=Path,required=True); p.add_argument('--private-manifest',type=Path,required=True); p.add_argument('--safe-json',type=Path,required=True); a=p.parse_args(argv)
 selected,rows,diag,nq=build(a.structural_csv,a.eligible_filenames)
 fields=['filename','speaker_id','split_set','recording_unit_id','primary_half']+[f'alt_{i:02d}_half' for i in range(25)]
 with a.private_manifest.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 o={'audit':'msp_final_scope_and_split_manifest_v1','eligible_input_rows':sum(1 for _ in a.eligible_filenames.read_text(encoding='utf-8').splitlines() if _.strip()),'qualifying_speakers_before_cap':nq,'selected_speakers':len(selected),'speaker_cap':MAX_SPEAKERS,'private_manifest_rows':len(rows),'primary_split_assigned':True,'alternative_splits_assigned':25,'final_scope_selected':True,'emotion_fields_used':False,'transcript_text_accessed':False,'textgrid_content_accessed':False,'audio_accessed':False,'acoustic_values_accessed':False,'reliability_values_accessed':False,'clinical_outcomes_accessed':False,'structural_sha256':sha(a.structural_csv),'eligible_list_sha256':sha(a.eligible_filenames),'private_manifest_sha256':sha(a.private_manifest)}
 a.safe_json.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n',encoding='utf-8'); return 0
if __name__=='__main__': raise SystemExit(main())