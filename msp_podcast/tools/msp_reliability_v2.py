#!/usr/bin/env python3
"""Frozen MSP v2 speaker-location reliability and Gate-M diagnostics.

No CLI or file I/O. Scientific choices are locked by cross_corpus_msp_model_lock_v2.
Real MSP execution remains separately preflight-gated.
"""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np

MIN_HALF_TOKENS=50
MIN_PAIRED_SPEAKERS=50
BOOTSTRAP_REPLICATES=1000
BOOTSTRAP_SEED=13
BOOTSTRAP_MIN_SUCCESS=950
PRIMARY_MIN_CORRECTED=.60
BOOTSTRAP_MIN_LOWER=.40
ALT_MIN_MEDIAN=.60
ALT_MAX_IQR=.15
LOSO_MAX_CHANGE=.15
VOLUME_MAX_CHANGE=.15
N_ALT_SPLITS=25

class MSPReliabilityError(ValueError): pass

def _avg_ranks(x):
    x=np.asarray(x,float); order=np.argsort(x,kind='mergesort'); ranks=np.empty(len(x),float); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        ranks[order[i:j]]=(i+j-1)/2+1; i=j
    return ranks

def _corr(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<2 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all(): return None
    if np.std(x)==0 or np.std(y)==0: return None
    r=float(np.corrcoef(x,y)[0,1]); return r if math.isfinite(r) else None

def corrected_pearson(x,y):
    r=_corr(x,y)
    if r is None or r<=-1: return r,None
    c=2*r/(1+r)
    return r,float(c) if math.isfinite(c) else None

def spearman(x,y): return _corr(_avg_ranks(x),_avg_ranks(y))

def speaker_pairs(rows,half_field='primary_half',recording_balanced=False):
    tok=defaultdict(lambda:defaultdict(list)); rec=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for r in rows:
        if not isinstance(r,dict) or not {'speaker_id','recording_unit_id',half_field,'residual'}<=set(r): raise MSPReliabilityError('row_schema')
        try: v=float(r['residual'])
        except Exception: continue
        if not math.isfinite(v): continue
        s=str(r['speaker_id']); h=str(r[half_field]); u=str(r['recording_unit_id'])
        if not s or not u: raise MSPReliabilityError('empty_structural_id')
        if h not in {'A','B'}: raise MSPReliabilityError('invalid_half')
        tok[s][h].append(v); rec[s][h][u].append(v)
    out=[]
    for s in sorted(tok):
        if len(tok[s]['A'])<MIN_HALF_TOKENS or len(tok[s]['B'])<MIN_HALF_TOKENS: continue
        vals=[]
        for h in ('A','B'):
            if recording_balanced: vals.append(float(np.median([np.median(v) for v in rec[s][h].values()])))
            else: vals.append(float(np.median(tok[s][h])))
        out.append((s,vals[0],vals[1],len(tok[s]['A']),len(tok[s]['B']),len(rec[s]['A']),len(rec[s]['B'])))
    return out

def reliability(rows,half_field='primary_half',recording_balanced=False):
    pairs=speaker_pairs(rows,half_field,recording_balanced); x=[p[1] for p in pairs]; y=[p[2] for p in pairs]; raw,corr=corrected_pearson(x,y)
    return {'paired_speakers':len(pairs),'support_adequate':len(pairs)>=MIN_PAIRED_SPEAKERS,'pearson':raw,'corrected_pearson':corr,'spearman':spearman(x,y) if len(pairs)>=2 else None,'pairs':pairs}

def bootstrap_from_pairs(pairs,replicates=BOOTSTRAP_REPLICATES,seed=BOOTSTRAP_SEED,return_values=False):
    if not pairs: raise MSPReliabilityError('no_pairs')
    x=np.asarray([p[1] for p in pairs]); y=np.asarray([p[2] for p in pairs]); n=len(pairs); rng=np.random.Generator(np.random.PCG64(seed)); vals=[]; failed=0
    for _ in range(replicates):
        idx=rng.integers(0,n,size=n); _,c=corrected_pearson(x[idx],y[idx])
        if c is None: failed+=1
        else: vals.append(c)
    ci=[float(np.percentile(vals,2.5,method='linear')),float(np.percentile(vals,97.5,method='linear'))] if len(vals)>=BOOTSTRAP_MIN_SUCCESS else None
    out={'requested':replicates,'successful':len(vals),'failed':failed,'minimum_success':BOOTSTRAP_MIN_SUCCESS,'technical_adequacy':len(vals)>=BOOTSTRAP_MIN_SUCCESS,'ci_95':ci,'replacement_draws_generated':False,'seed':seed}
    if return_values: out['values']=vals
    return out

def loso(pairs):
    x=np.asarray([p[1] for p in pairs]); y=np.asarray([p[2] for p in pairs]); _,base=corrected_pearson(x,y)
    if base is None: return {'base':None,'max_abs_change':None,'changes':[],'passes':False}
    changes=[]
    for i,p in enumerate(pairs):
        keep=np.arange(len(pairs))!=i; _,c=corrected_pearson(x[keep],y[keep]); changes.append((p[0],None if c is None else abs(c-base)))
    finite=[v for _,v in changes if v is not None]; mx=max(finite) if finite else None
    return {'base':base,'max_abs_change':mx,'changes':changes,'passes':mx is not None and mx<=LOSO_MAX_CHANGE}

def alternative_split_metrics(rows,n=N_ALT_SPLITS):
    if n!=N_ALT_SPLITS: raise MSPReliabilityError('alternative_split_count_must_be_25')
    out=[reliability(rows,f'alt_{i:02d}_half') for i in range(n)]; vals=[r['corrected_pearson'] for r in out if r['corrected_pearson'] is not None]
    med=None if not vals else float(np.median(vals)); iqr=None if not vals else float(np.percentile(vals,75)-np.percentile(vals,25))
    return {'splits':out,'valid_splits':len(vals),'median':med,'iqr':iqr,'min':None if not vals else min(vals),'max':None if not vals else max(vals),'passes':len(vals)==N_ALT_SPLITS and med is not None and med>=ALT_MIN_MEDIAN and iqr is not None and iqr<=ALT_MAX_IQR}

def recording_volume_sensitivity(rows):
    a=reliability(rows,recording_balanced=False); b=reliability(rows,recording_balanced=True); x=a['corrected_pearson']; y=b['corrected_pearson']; delta=None if x is None or y is None else abs(x-y)
    return {'pooled_token_corrected':x,'recording_balanced_corrected':y,'abs_change':delta,'passes':delta is not None and delta<=VOLUME_MAX_CHANGE,'threshold':VOLUME_MAX_CHANGE}

def gate_m_feature_conditions(primary,bootstrap,alternatives,loso_result,volume):
    lower=None if bootstrap.get('ci_95') is None else bootstrap['ci_95'][0]
    conditions={
      'paired_speakers_at_least_50':primary.get('paired_speakers',0)>=MIN_PAIRED_SPEAKERS,
      'speaker_half_support_at_least_50':bool(primary.get('support_adequate')),
      'primary_corrected_at_least_0_60':primary.get('corrected_pearson') is not None and primary['corrected_pearson']>=PRIMARY_MIN_CORRECTED,
      'bootstrap_lower_at_least_0_40':bool(bootstrap.get('technical_adequacy')) and lower is not None and lower>=BOOTSTRAP_MIN_LOWER,
      'alternative_split_median_iqr':bool(alternatives.get('passes')),
      'loso_max_change_at_most_0_15':bool(loso_result.get('passes')),
      'recording_volume_change_at_most_0_15':bool(volume.get('passes')),
    }
    return {'conditions':conditions,'passes':all(conditions.values())}
