#!/usr/bin/env python3
"""Standalone exact extraction of the frozen ESD word-acoustic analyzer.

Only frames() and analyze_word() are copied verbatim from the frozen ESD Stage-2C
adapter so MSP execution does not import unrelated alignment dependencies.
"""
SOURCE_ESD_ADAPTER_SHA256 = "1fc1606ed44784f6bc1c05673f150cad799c0ed030df6043c3260736424a9b01"

def frames(signal,size,hop):
    import numpy as np
    if not len(signal): return []
    if len(signal)<=size: return [np.pad(signal,(0,size-len(signal)))]
    return [signal[i:i+size] for i in range(0,len(signal)-size+1,hop)]

def analyze_word(signal,sr):
    import numpy as np
    size=round(.040*sr); hop=round(.010*sr); chunks=frames(signal,size,hop); window=np.hanning(size); pitches=[]
    min_lag=max(1,int(sr/500.0)); max_lag=min(size-2,int(sr/60.0))
    for chunk in chunks:
        rms=float(np.sqrt(np.mean(chunk.astype(np.float64)**2)))
        if rms<.0001: continue
        x=(chunk-chunk.mean())*window; nfft=1<<(2*size-1).bit_length(); spectrum=np.fft.rfft(x,nfft); ac=np.fft.irfft(spectrum*np.conj(spectrum),nfft)[:size]
        if ac[0]<=0: continue
        search=ac[min_lag:max_lag+1]/ac[0]; best=int(np.argmax(search))
        if search[best]>=.30: pitches.append(sr/float(best+min_lag))
    p=np.asarray(pitches,float); rms=float(np.sqrt(np.mean(signal.astype(np.float64)**2))) if len(signal) else np.nan
    lo=float(np.percentile(p,10)) if len(p) else None; hi=float(np.percentile(p,90)) if len(p) else None
    return {"f0_mean_hz":float(np.mean(p)) if len(p) else None,"f0_p10_hz":lo,"f0_p90_hz":hi,
            "f0_robust_range_hz":hi-lo if lo is not None else None,"energy_db":max(-100.0,20*np.log10(max(rms,1e-5))) if np.isfinite(rms) else None}
