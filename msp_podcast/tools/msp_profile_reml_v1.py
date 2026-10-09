#!/usr/bin/env python3
"""Frozen secondary-only deterministic profile-REML word random intercept.

No CLI is provided. Scientific use is restricted to the cross-fitted lexically adjusted secondary defined by cross_corpus_msp_model_lock_v2; it is not Gate M primary evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar

class ProfileREMLError(ValueError):
    pass

@dataclass(frozen=True)
class ProfileREMLFit:
    beta: np.ndarray
    sigma2: float
    tau2: float
    theta: float
    lambda_ratio: float
    group_levels: np.ndarray
    group_blup: np.ndarray
    residuals: np.ndarray
    objective: float


def _as_inputs(y, X, groups):
    y=np.asarray(y,dtype=float); X=np.asarray(X,dtype=float); groups=np.asarray(groups)
    if y.ndim!=1 or X.ndim!=2 or groups.ndim!=1 or len(y)!=X.shape[0] or len(y)!=len(groups):
        raise ProfileREMLError('shape_mismatch')
    n,p=X.shape
    if n<=p or p<1 or not np.isfinite(y).all() or not np.isfinite(X).all():
        raise ProfileREMLError('invalid_or_nonfinite_input')
    if np.linalg.matrix_rank(X)!=p:
        raise ProfileREMLError('fixed_effect_design_rank_deficient')
    levels,inv=np.unique(groups,return_inverse=True)
    counts=np.bincount(inv)
    if len(levels)<2 or counts.max()<2:
        raise ProfileREMLError('random_intercept_not_identifiable')
    return y,X,levels,inv,counts.astype(float)


def fit_profile_reml_random_intercept(y, X, groups, *, xatol=1e-10, maxiter=300):
    y,X,levels,inv,ng=_as_inputs(y,X,groups)
    n,p=X.shape
    sy=np.bincount(inv,weights=y)
    sx=np.vstack([np.bincount(inv,weights=X[:,j]) for j in range(p)]).T
    XtX=X.T@X; Xty=X.T@y; yty=float(y@y)

    def evaluate(theta, full=False):
        if not (0.0<=theta<1.0): return None if full else np.inf
        lam=0.0 if theta==0.0 else theta/(1.0-theta)
        a=np.zeros_like(ng) if lam==0.0 else lam/(1.0+lam*ng)
        M=XtX-sx.T@(a[:,None]*sx)
        b=Xty-sx.T@(a*sy)
        try:
            beta=np.linalg.solve(M,b)
        except np.linalg.LinAlgError:
            return None if full else np.inf
        sse=yty-float(np.dot(a,sy*sy))-float(beta@b)
        df=n-p
        sign,logdetM=np.linalg.slogdet(M)
        if sign<=0 or not np.isfinite(sse) or sse<=0:
            return None if full else np.inf
        logdetW=0.0 if lam==0.0 else float(np.log1p(lam*ng).sum())
        obj=float(df*np.log(sse/df)+logdetW+logdetM)
        if not np.isfinite(obj): return None if full else np.inf
        if not full: return obj
        sigma2=float(sse/df); tau2=float(lam*sigma2)
        raw=y-X@beta; sumraw=np.bincount(inv,weights=raw)
        blup=np.zeros(len(levels)) if lam==0.0 else (lam/(1.0+lam*ng))*sumraw
        residuals=raw-blup[inv]
        if not np.isfinite(residuals).all() or sigma2<=0 or tau2<0:
            raise ProfileREMLError('nonfinite_fit')
        return ProfileREMLFit(np.asarray(beta),sigma2,tau2,float(theta),float(lam),levels,np.asarray(blup),np.asarray(residuals),obj)

    opt=minimize_scalar(lambda t:evaluate(float(t)),bounds=(0.0,1.0-1e-10),method='bounded',options={'xatol':xatol,'maxiter':maxiter})
    candidates=[evaluate(0.0,True)]
    if opt.success and np.isfinite(opt.fun): candidates.append(evaluate(float(opt.x),True))
    candidates=[c for c in candidates if c is not None]
    if not candidates: raise ProfileREMLError('profile_optimization_failed')
    best=min(candidates,key=lambda c:c.objective)
    return best
