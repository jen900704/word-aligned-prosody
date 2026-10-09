#!/usr/bin/env python3
"""Execute the frozen cross-fitted lexical secondary on MSP merged acoustics."""
from __future__ import annotations
import argparse,csv,json,math,os
from pathlib import Path
import numpy as np
from tools.msp_acoustic_extraction_runner_v2 import OUT_FIELDS,sha256
from tools.msp_lexical_secondary_v1 import crossfit_two_halves
from tools.msp_primary_residualization_v2 import ALL_FEATURES,DURATION_FEATURE,F0_ENERGY_FEATURES
from tools.msp_reliability_v2 import reliability
FEATURE_SOURCE={'f0_mean_hz':'f0_mean_hz','f0_robust_range_hz':'f0_robust_range_hz','energy_db':'energy_db',DURATION_FEATURE:'word_duration_sec'}
SPLITS=['primary_half']+[f'alt_{i:02d}_half' for i in range(25)]
class LexicalAnalysisError(ValueError): pass

def load_merged(path:Path,safe_path:Path):
    z=json.loads(safe_path.read_text(encoding='utf-8'))
    if z.get('audit')!='msp_acoustic_merge_v1' or z.get('status')!='PASS' or z.get('output_sha256')!=sha256(path): raise LexicalAnalysisError('merge_safe_hash_chain')
    if z.get('reliability_values_accessed') is not False or z.get('clinical_outcomes_accessed') is not False: raise LexicalAnalysisError('merge_provenance')
    with path.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        if list(r.fieldnames or [])!=OUT_FIELDS: raise LexicalAnalysisError('merged_schema')
        rows=list(r)
    if not rows or len(rows)!=z.get('merged_token_rows'): raise LexicalAnalysisError('merged_count')
    return rows

def finite(x):
    try: z=float(x)
    except Exception: return None
    return z if math.isfinite(z) else None

def valid_feature_rows(rows,feature):
    src=FEATURE_SOURCE[feature]; out=[]
    for r in rows:
        v=finite(r.get(src)); p=finite(r.get('relative_position')); w=str(r.get('normalized_word',''))
        if p is None or not 0<=p<=1 or not w: raise LexicalAnalysisError('position_or_word_invalid')
        if feature in F0_ENERGY_FEATURES:
            if v is None: continue
            out.append((r,v,p,w,None))
        else:
            sy=finite(r.get('authoritative_syllable_count'))
            if v is None or v<=0 or sy is None or sy<=0 or sy!=math.floor(sy): continue
            out.append((r,v,p,w,int(sy)))
    if not out: raise LexicalAnalysisError('no_valid_rows')
    return out

def analyze_split(valid_rows,feature,half_field):
    halves={'A':[],'B':[]}
    for item in valid_rows:
        h=item[0].get(half_field)
        if h not in halves: raise LexicalAnalysisError('invalid_half')
        halves[h].append(item)
    if not halves['A'] or not halves['B']: raise LexicalAnalysisError('empty_half')
    def unpack(items):
        vals=[x[1] for x in items]; pos=[x[2] for x in items]; words=[x[3] for x in items]; syll=[x[4] for x in items] if feature==DURATION_FEATURE else None
        return vals,pos,words,syll
    av,ap,aw,asy=unpack(halves['A']); bv,bp,bw,bsy=unpack(halves['B'])
    cf=crossfit_two_halves(feature,av,ap,aw,bv,bp,bw,asy,bsy)
    residual_rows=[]
    for h,items in [('A',halves['A']),('B',halves['B'])]:
        rr=cf[h].residuals
        if len(rr)!=len(items): raise LexicalAnalysisError('crossfit_length')
        for x,res in zip(items,rr): residual_rows.append({**x[0],half_field:h,'residual':float(res)})
    rel=reliability(residual_rows,half_field=half_field)
    return {'half_field':half_field,'paired_speakers':rel['paired_speakers'],'support_adequate':rel['support_adequate'],'pearson':rel['pearson'],'corrected_pearson':rel['corrected_pearson'],'spearman':rel['spearman'],
            'A_train_half':'B','B_train_half':'A','A_unseen_words':cf['A'].unseen_word_count,'B_unseen_words':cf['B'].unseen_word_count,
            'A_tau2':cf['A'].tau2,'B_tau2':cf['B'].tau2,'A_sigma2':cf['A'].sigma2,'B_sigma2':cf['B'].sigma2,'gate_m_threshold_applied':False,'can_modify_gate_m':False}

def analyze_feature(rows,feature):
    valid=valid_feature_rows(rows,feature); reports={}
    for split in SPLITS:
        try: reports[split]={'status':'complete',**analyze_split(valid,feature,split)}
        except Exception as e: reports[split]={'status':'technical_failure','error_type':type(e).__name__,'gate_m_threshold_applied':False,'can_modify_gate_m':False}
    return {'feature':feature,'valid_feature_tokens':len(valid),'splits':reports,'gate_m_threshold_applied':False,'can_modify_gate_m':False}

def analyze_rows(rows):
    return {'analysis':'msp_crossfitted_lexical_secondary_v1','features':{f:analyze_feature(rows,f) for f in sorted(ALL_FEATURES)},'secondary_only':True,'can_modify_gate_m':False,'emotion_fields_used':False,'clinical_outcomes_accessed':False}

def safe_receipt(result,private_path,merged_path,merge_safe_path):
    feats={}
    for f,r in result['features'].items():
        states=[x['status'] for x in r['splits'].values()]
        paired=[x.get('paired_speakers') for x in r['splits'].values() if x['status']=='complete']
        feats[f]={'valid_feature_tokens':r['valid_feature_tokens'],'splits_requested':len(SPLITS),'splits_complete':sum(x=='complete' for x in states),'splits_failed':sum(x!='complete' for x in states),'paired_speakers_min':min(paired) if paired else None,'paired_speakers_max':max(paired) if paired else None}
    return {'audit':'msp_crossfitted_lexical_secondary_technical_safe_v1','status':'PASS' if all(x['splits_complete']==len(SPLITS) for x in feats.values()) else 'TECHNICAL_REVIEW','features':feats,'private_result_sha256':sha256(private_path),'merged_acoustic_sha256':sha256(merged_path),'merge_safe_sha256':sha256(merge_safe_path),'reliability_values_emitted':False,'gate_m_status_emitted':False,'speaker_ids_emitted':False,'secondary_only':True,'can_modify_gate_m':False,'emotion_fields_used':False,'clinical_outcomes_accessed':False}

def run(merged_path,merge_safe_path,private_json,safe_json):
    rows=load_merged(merged_path,merge_safe_path); result=analyze_rows(rows); private_json.parent.mkdir(parents=True,exist_ok=True); private_json.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); os.chmod(private_json,0o600); safe=safe_receipt(result,private_json,merged_path,merge_safe_path); safe_json.parent.mkdir(parents=True,exist_ok=True); safe_json.write_text(json.dumps(safe,indent=2,sort_keys=True)+'\n'); os.chmod(safe_json,0o600); return result,safe

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--merged-acoustics',type=Path,required=True); p.add_argument('--merge-safe',type=Path,required=True); p.add_argument('--private-result',type=Path,required=True); p.add_argument('--safe-json',type=Path,required=True); a=p.parse_args(argv); run(a.merged_acoustics,a.merge_safe,a.private_result,a.safe_json); return 0
if __name__=='__main__': raise SystemExit(main())
