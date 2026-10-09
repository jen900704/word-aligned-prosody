#!/usr/bin/env python3
"""Frozen MSP v1 array-level acoustic adapter mechanics.

No file I/O or CLI. The acoustic analyzer is injected by the caller so this adapter introduces no new acoustic mathematics. Production use must supply the already frozen ESD/Seamless analyze_word definition and pass the MSP preflight guard.
"""
from __future__ import annotations
import math
import numpy as np

class AdapterPrototypeError(ValueError): pass

def relative_position(word_index:int, n_words:int)->float:
    if isinstance(word_index,bool) or isinstance(n_words,bool) or not isinstance(word_index,int) or not isinstance(n_words,int): raise AdapterPrototypeError('position_type')
    if n_words<1 or word_index<0 or word_index>=n_words: raise AdapterPrototypeError('position_bounds')
    return 0.0 if n_words==1 else float(word_index/(n_words-1))

def slice_resample_mono(audio, sr:int, start:float, end:float, target_sr:int=16000):
    x=np.asarray(audio)
    if x.ndim==1: mono=x.astype(np.float32,copy=False)
    elif x.ndim==2 and x.shape[1]>=1: mono=x.astype(np.float32).mean(axis=1)
    else: raise AdapterPrototypeError('audio_shape')
    if isinstance(sr,bool) or not isinstance(sr,int) or sr<=0 or target_sr<=0: raise AdapterPrototypeError('sample_rate')
    duration=len(mono)/sr
    vals=(start,end,duration)
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(float(v)) for v in vals): raise AdapterPrototypeError('nonfinite_bounds')
    start=float(start); end=float(end)
    if start<0 or end<=start or end>duration+1e-6: raise AdapterPrototypeError('invalid_word_bounds')
    a=round(start*sr); b=a+round((end-start)*sr); seg=mono[a:b]
    if len(seg)==0: raise AdapterPrototypeError('empty_word_segment')
    if sr!=target_sr:
        n=max(1,round(len(seg)*target_sr/sr))
        seg=np.interp(np.linspace(0,len(seg)-1,n),np.arange(len(seg)),seg).astype(np.float32)
        sr=target_sr
    return seg,sr

def analyze_interval(audio,sr,start,end,analyze_word):
    seg,out_sr=slice_resample_mono(audio,sr,start,end,16000)
    out=analyze_word(seg,out_sr)
    required={'f0_mean_hz','f0_robust_range_hz','energy_db'}
    if set(out)<required: raise AdapterPrototypeError('acoustic_output_schema')
    return out
