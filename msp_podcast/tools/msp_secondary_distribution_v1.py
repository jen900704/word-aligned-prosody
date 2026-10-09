#!/usr/bin/env python3
"""Frozen descriptive IQR/MAD speaker-half diagnostics for MSP v2."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from tools.msp_reliability_v2 import MIN_HALF_TOKENS,corrected_pearson,spearman

class SecondaryDistributionError(ValueError): pass

def iqr_linear(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all(): raise SecondaryDistributionError('invalid_values')
    q=np.percentile(x,[25,75],method='linear'); return float(q[1]-q[0])

def mad_unscaled(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all(): raise SecondaryDistributionError('invalid_values')
    m=float(np.median(x)); return float(np.median(np.abs(x-m)))

def summary_pairs(rows,summary_name,half_field='primary_half'):
    if summary_name not in {'iqr','mad'}: raise SecondaryDistributionError('unknown_summary')
    by=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if not isinstance(r,dict) or not {'speaker_id',half_field,'residual'}<=set(r): raise SecondaryDistributionError('row_schema')
        try: v=float(r['residual'])
        except Exception: continue
        if not math.isfinite(v): continue
        s=str(r['speaker_id']); h=str(r[half_field])
        if not s or h not in {'A','B'}: raise SecondaryDistributionError('structural_id_or_half')
        by[s][h].append(v)
    fn=iqr_linear if summary_name=='iqr' else mad_unscaled; out=[]
    for s in sorted(by):
        if len(by[s]['A'])<MIN_HALF_TOKENS or len(by[s]['B'])<MIN_HALF_TOKENS: continue
        out.append((s,fn(by[s]['A']),fn(by[s]['B']),len(by[s]['A']),len(by[s]['B'])))
    return out

def descriptive_reliability(rows,summary_name,half_field='primary_half'):
    pairs=summary_pairs(rows,summary_name,half_field); x=[p[1] for p in pairs]; y=[p[2] for p in pairs]
    raw,corr=corrected_pearson(x,y)
    return {'summary':summary_name,'paired_speakers':len(pairs),'pearson':raw,'corrected_pearson':corr,'spearman':spearman(x,y) if len(pairs)>=2 else None,
            'gate_m_threshold_applied':False,'can_modify_gate_m':False}
