#!/usr/bin/env python3
"""POST HOC (post-review, 2026-09-09). Word-level crossed model on MSP-Podcast
(spontaneous speech), to test whether the ESD word-deviation result is confined
to acted read speech.

Estimator mirrors the frozen ESD one (statsmodels MixedLM, vc S/W/SW, REML,
lbfgs, maxiter 2000, cells with >=2 finite tokens) with these differences:
  * no emotion conditions: one fit per speaker subset
  * speaker 'Unknown' (pooled unidentified speakers) excluded
  * speakers subsampled to ESD-sized sets (NSPK speakers, NDRAW draws, seed 13)
    so the fit size and design match the ESD analysis
  * at most CAP tokens per (speaker, word) cell (random, seeded) to bound imbalance
  * duration residual: log dur ~ 1 + syllables + C(speaker) within subset
    (ESD-identical) and a syllables-only variant (_nospk)
Run inside .envs/stage2c_reml.
usage: python msp_postreview_crossed_v1.py NSPK NDRAW CAP [NPROC]
"""
import json, math, sys, time, warnings
import os
from pathlib import Path
import numpy as np, pandas as pd
from multiprocessing import Pool

SRC = os.path.join(os.environ.get('WAP_WORK_ROOT', 'work'), 'prosody_msp_structural_v1/run1/acoustics_v1/merged/msp_acoustics_PRIVATE.csv')
OUTDIR = Path(os.path.join(os.environ.get('WAP_WORK_ROOT', 'work'), 'prosody_esd_x2/postreview_sensitivity_v1/msp_crossed'))
SEED = 13
FEATS = ['f0_mean_hz', 'f0_robust_range_hz', 'energy_db', 'duration_log_syllable_residual', 'duration_log_syllable_residual_nospk']

def load():
    use = ['filename', 'speaker_id', 'recording_unit_id', 'normalized_word', 'word_duration_sec', 'authoritative_syllable_count', 'f0_mean_hz', 'f0_robust_range_hz', 'energy_db', 'acoustic_status']
    df = pd.read_csv(SRC, usecols=use, dtype={'speaker_id': str})
    df = df[(df.acoustic_status == 'complete') & (df.speaker_id != 'Unknown')].copy()
    df['normalized_word'] = df.normalized_word.astype(str)
    return df

def residualize(sub):
    ok = (sub.word_duration_sec > 0) & (sub.authoritative_syllable_count > 0)
    d = sub[ok].copy(); y = np.log(d.word_duration_sec.values)
    spk = sorted(d.speaker_id.unique())
    X = np.column_stack([np.ones(len(d)), d.authoritative_syllable_count.values] + [(d.speaker_id.values == s).astype(float) for s in spk[1:]])
    b = np.linalg.lstsq(X, y, rcond=None)[0]; sub.loc[d.index, 'duration_log_syllable_residual'] = y - X @ b
    X2 = X[:, :2]; b2 = np.linalg.lstsq(X2, y, rcond=None)[0]; sub.loc[d.index, 'duration_log_syllable_residual_nospk'] = y - X2 @ b2
    return sub

def fit_one(frame, feat):
    import statsmodels.formula.api as smf
    f = frame[np.isfinite(frame[feat].astype(float))].copy()
    n = f.groupby(['speaker_id', 'normalized_word'])[feat].transform('size'); f = f[n >= 2].copy()
    f['y'] = f[feat].astype(float); f['top_group'] = 'all'; f['speaker'] = f.speaker_id.astype(str); f['word'] = f.normalized_word.astype(str)
    f['speaker_word'] = f['speaker'] + '\x1f' + f['word']
    model = smf.mixedlm('y ~ 1', f, groups='top_group', re_formula='0', vc_formula={'S': '0 + C(speaker)', 'W': '0 + C(word)', 'SW': '0 + C(speaker_word)'}, use_sparse=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); res = model.fit(reml=True, method='lbfgs', maxiter=2000, disp=False)
    names = list(model.exog_vc.names); vc = dict(zip(names, map(float, res.vcomp))); e = float(res.scale)
    tot = vc['S'] + vc['W'] + vc['SW'] + e
    return {'converged': bool(res.converged), 'llf': float(res.llf), 'warnings': sorted({w.category.__name__ for w in caught}),
            'variance_components': {**vc, 'e': e}, 'percent': {k: 100 * v / tot for k, v in {**vc, 'e': e}.items()},
            'repeatability': vc['SW'] / (vc['SW'] + e), 'token_count': int(len(f)), 'cell_count': int(f.speaker_word.nunique()),
            'speaker_count': int(f.speaker.nunique()), 'word_count': int(f.word.nunique())}

def run_draw(args):
    draw, speakers, cap = args
    df = load(); sub = df[df.speaker_id.isin(speakers)].copy()
    rng = np.random.default_rng([SEED, draw])
    sub['_r'] = rng.random(len(sub)); sub = sub.sort_values('_r')
    sub = sub.groupby(['speaker_id', 'normalized_word'], sort=False).head(cap).copy()
    sub = residualize(sub)
    out = {'draw': draw, 'speakers': list(map(str, speakers)), 'cap': cap, 'tokens_after_cap': int(len(sub)), 'features': {}}
    for feat in FEATS:
        t0 = time.time()
        try:
            r = fit_one(sub, feat); r['seconds'] = time.time() - t0; out['features'][feat] = {'status': 'ok', **r}
            print(f'[draw {draw}] {feat}: R={r["repeatability"]:.4f} S%={r["percent"]["S"]:.1f} W%={r["percent"]["W"]:.1f} SW%={r["percent"]["SW"]:.1f} e%={r["percent"]["e"]:.1f} n={r["token_count"]} cells={r["cell_count"]} conv={r["converged"]} ({r["seconds"]:.0f}s)', flush=True)
        except Exception as exc:
            out['features'][feat] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'seconds': time.time() - t0}
            print(f'[draw {draw}] {feat}: FAILED {type(exc).__name__}: {exc}', flush=True)
    return out

def main():
    nspk, ndraw, cap = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load(); speakers = sorted(df.speaker_id.unique())
    print(f'MSP tokens {len(df)} speakers {len(speakers)} (Unknown excluded); NSPK={nspk} NDRAW={ndraw} CAP={cap}', flush=True)
    rng = np.random.default_rng([SEED, nspk, ndraw])
    draws = [(i, sorted(rng.choice(speakers, nspk, replace=False).tolist()), cap) for i in range(ndraw)]
    t0 = time.time()
    with Pool(nproc) as pool: results = pool.map(run_draw, draws)
    summary = {}
    for feat in FEATS:
        ok = [r['features'][feat] for r in results if r['features'][feat]['status'] == 'ok' and r['features'][feat]['converged']]
        if not ok: summary[feat] = {'n_ok': 0}; continue
        reps = np.array([x['repeatability'] for x in ok])
        summary[feat] = {'n_ok': len(ok), 'n_draws': ndraw, 'repeatability_median': float(np.median(reps)), 'repeatability_min': float(reps.min()), 'repeatability_max': float(reps.max()),
                         'percent_median': {k: float(np.median([x['percent'][k] for x in ok])) for k in ('S', 'W', 'SW', 'e')},
                         'tokens_median': float(np.median([x['token_count'] for x in ok])), 'cells_median': float(np.median([x['cell_count'] for x in ok]))}
        print(f'SUMMARY {feat}: R median={summary[feat]["repeatability_median"]:.4f} [{reps.min():.4f}, {reps.max():.4f}] SW%={summary[feat]["percent_median"]["SW"]:.2f} ({len(ok)}/{ndraw} ok)', flush=True)
    out = {'nspk': nspk, 'ndraw': ndraw, 'cap': cap, 'seed': SEED, 'source': SRC, 'unknown_excluded': True, 'summary': summary, 'draws': results, 'seconds': time.time() - t0}
    (OUTDIR / f'msp_crossed_nspk{nspk}_ndraw{ndraw}_cap{cap}.json').write_text(json.dumps(out, indent=1, default=float))
    print('wrote', OUTDIR / f'msp_crossed_nspk{nspk}_ndraw{ndraw}_cap{cap}.json', flush=True)

if __name__ == '__main__': main()
