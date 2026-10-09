#!/usr/bin/env python3
"""Frozen cross-fitted lexical-adjustment secondary for MSP model-lock v2.

No CLI/file I/O. This module is secondary diagnostic machinery only and cannot
satisfy or modify Gate M primary conditions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tools.msp_primary_residualization_v2 import F0_ENERGY_FEATURES,DURATION_FEATURE,ALL_FEATURES,PrimaryResidualizationError,design_matrix
from tools.msp_profile_reml_v1 import fit_profile_reml_random_intercept,ProfileREMLError

class LexicalSecondaryError(ValueError): pass

@dataclass(frozen=True)
class CrossFitHalf:
    train_half: str
    heldout_half: str
    beta: np.ndarray
    tau2: float
    sigma2: float
    residuals: np.ndarray
    unseen_word_count: int

def _response(feature,values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all(): raise LexicalSecondaryError('invalid_values')
    if feature==DURATION_FEATURE:
        if np.any(x<=0): raise LexicalSecondaryError('nonpositive_duration')
        return np.log(x)
    if feature in F0_ENERGY_FEATURES: return x
    raise LexicalSecondaryError('unknown_feature')

def fit_train_apply_heldout(feature,train_values,train_pos,train_words,heldout_values,heldout_pos,heldout_words,train_syllables=None,heldout_syllables=None,train_half='A',heldout_half='B'):
    if train_half==heldout_half or {train_half,heldout_half}!={'A','B'}: raise LexicalSecondaryError('half_direction_invalid')
    tw=np.asarray(train_words,dtype=object); hw=np.asarray(heldout_words,dtype=object)
    if tw.ndim!=1 or hw.ndim!=1 or len(tw)==0 or len(hw)==0 or any(not str(x) for x in tw) or any(not str(x) for x in hw): raise LexicalSecondaryError('word_group_invalid')
    try:
        Xtr=design_matrix(feature,train_pos,train_syllables)
        Xho=design_matrix(feature,heldout_pos,heldout_syllables)
    except PrimaryResidualizationError as e: raise LexicalSecondaryError(str(e)) from e
    ytr=_response(feature,train_values); yho=_response(feature,heldout_values)
    if len(ytr)!=len(Xtr) or len(ytr)!=len(tw) or len(yho)!=len(Xho) or len(yho)!=len(hw): raise LexicalSecondaryError('shape_mismatch')
    try: fit=fit_profile_reml_random_intercept(ytr,Xtr,tw)
    except ProfileREMLError as e: raise LexicalSecondaryError(str(e)) from e
    lookup={str(k):float(v) for k,v in zip(fit.group_levels,fit.group_blup)}
    random=np.asarray([lookup.get(str(w),0.0) for w in hw],dtype=float)
    unseen=sum(str(w) not in lookup for w in hw)
    residuals=yho-Xho@fit.beta-random
    if not np.isfinite(residuals).all(): raise LexicalSecondaryError('nonfinite_heldout_residual')
    return CrossFitHalf(train_half,heldout_half,np.asarray(fit.beta),float(fit.tau2),float(fit.sigma2),np.asarray(residuals),int(unseen))

def crossfit_two_halves(feature,a_values,a_pos,a_words,b_values,b_pos,b_words,a_syllables=None,b_syllables=None):
    # A residuals are generated from a nuisance model fit only on B, and vice versa.
    a=fit_train_apply_heldout(feature,b_values,b_pos,b_words,a_values,a_pos,a_words,b_syllables,a_syllables,'B','A')
    b=fit_train_apply_heldout(feature,a_values,a_pos,a_words,b_values,b_pos,b_words,a_syllables,b_syllables,'A','B')
    return {'A':a,'B':b}
