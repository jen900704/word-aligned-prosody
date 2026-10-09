#!/usr/bin/env python3
"""Frozen MSP v2 primary nuisance residualization helpers.

No file I/O or CLI. Scientific choices are locked by
notes/cross_corpus_msp_model_lock_v2.{md,json}. This module must be tested on
synthetic data before any MSP acoustic values are supplied.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

F0_ENERGY_FEATURES=frozenset({'f0_mean_hz','f0_robust_range_hz','energy_db'})
DURATION_FEATURE='duration_log_syllable_residual'
ALL_FEATURES=F0_ENERGY_FEATURES|{DURATION_FEATURE}

class PrimaryResidualizationError(ValueError): pass

@dataclass(frozen=True)
class PrimaryOLSFit:
    feature: str
    beta: np.ndarray
    residuals: np.ndarray
    transformed_response: np.ndarray
    design_matrix: np.ndarray

def relative_position(word_index:int,n_nonempty_normalized_words:int)->float:
    if isinstance(word_index,bool) or isinstance(n_nonempty_normalized_words,bool) or not isinstance(word_index,int) or not isinstance(n_nonempty_normalized_words,int):
        raise PrimaryResidualizationError('position_type')
    if n_nonempty_normalized_words<1 or word_index<0 or word_index>=n_nonempty_normalized_words:
        raise PrimaryResidualizationError('position_bounds')
    return 0.0 if n_nonempty_normalized_words==1 else float(word_index/(n_nonempty_normalized_words-1))

def _array(x,name):
    a=np.asarray(x,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all(): raise PrimaryResidualizationError(f'{name}_invalid')
    return a

def design_matrix(feature,relative_positions,syllable_counts=None):
    if feature not in ALL_FEATURES: raise PrimaryResidualizationError('unknown_feature')
    pos=_array(relative_positions,'relative_position')
    if np.any((pos<0)|(pos>1)): raise PrimaryResidualizationError('relative_position_bounds')
    if feature in F0_ENERGY_FEATURES:
        if syllable_counts is not None: raise PrimaryResidualizationError('syllables_forbidden_for_f0_energy_primary')
        X=np.column_stack([np.ones(len(pos)),pos])
    else:
        if syllable_counts is None: raise PrimaryResidualizationError('duration_requires_syllables')
        syll=_array(syllable_counts,'syllable_count')
        if len(syll)!=len(pos) or np.any(syll<=0) or np.any(syll!=np.floor(syll)): raise PrimaryResidualizationError('syllable_count_invalid')
        X=np.column_stack([np.ones(len(pos)),syll,pos])
    return X

def fit_primary_residuals(feature,values,relative_positions,syllable_counts=None):
    if feature not in ALL_FEATURES: raise PrimaryResidualizationError('unknown_feature')
    raw=_array(values,'values'); X=design_matrix(feature,relative_positions,syllable_counts)
    if len(raw)!=len(X) or len(raw)<=X.shape[1]: raise PrimaryResidualizationError('insufficient_rows')
    if feature==DURATION_FEATURE:
        if np.any(raw<=0): raise PrimaryResidualizationError('duration_nonpositive')
        y=np.log(raw)
    else: y=raw.copy()
    if np.linalg.matrix_rank(X)!=X.shape[1]: raise PrimaryResidualizationError('design_rank_deficient')
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    residuals=y-X@beta
    if not np.isfinite(beta).all() or not np.isfinite(residuals).all(): raise PrimaryResidualizationError('nonfinite_fit')
    return PrimaryOLSFit(feature,np.asarray(beta),np.asarray(residuals),np.asarray(y),np.asarray(X))
