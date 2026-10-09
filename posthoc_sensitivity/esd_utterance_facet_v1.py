#!/usr/bin/env python3
"""POST HOC (2026-09-09, review round 3). Add utterance as an explicit facet to the
ESD crossed model, so contextual variation is separated from residual error:
  y = mu_c + S + W + SW + U + e     (U = utterance/sentence occurrence)
R_c is then SW/(SW+e) with U removed from e. Reported alongside the frozen R_c.
usage: python esd_utterance_facet_v1.py <condition_index 0-4>
"""
import json, sys, time, warnings
import os
import numpy as np, pandas as pd
IN = os.path.join(os.environ.get('WAP_WORK_ROOT', 'work'), 'prosody_esd_x2/postreview_sensitivity_v1/esd_postreview_features_PRIVATE.csv')
OUT = os.path.join(os.environ.get('WAP_WORK_ROOT', 'work'), 'prosody_esd_x2/postreview_sensitivity_v1/utt')
FEATS = ['f0_mean_hz', 'f0_robust_range_hz', 'energy_db', 'duration_log_syllable_residual']
def load():
    df = pd.read_csv(IN, dtype={'speaker_id': str, 'utterance_id': str, 'condition': str,
                                'normalized_word': str})
    df = df.rename(columns={'orig__f0_mean_hz': 'f0_mean_hz',
        'orig__f0_robust_range_hz': 'f0_robust_range_hz', 'orig__energy_db': 'energy_db',
        'orig__word_duration_sec': 'word_duration_sec'})
    ok = (df.word_duration_sec > 0) & (df.authoritative_syllable_count > 0)
    d = df[ok]; y = np.log(d.word_duration_sec.values)
    spk = sorted(d.speaker_id.unique())
    X = np.column_stack([np.ones(len(d)), d.authoritative_syllable_count.values]
                        + [(d.speaker_id.values == s).astype(float) for s in spk[1:]])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    df.loc[d.index, 'duration_log_syllable_residual'] = y - X @ b
    return df
def fit(frame, feat, with_u):
    import statsmodels.formula.api as smf
    f = frame[np.isfinite(frame[feat].astype(float))].copy()
    n = f.groupby(['speaker_id', 'normalized_word'])[feat].transform('size')
    f = f[n >= 2].copy()
    f['y'] = f[feat].astype(float); f['top_group'] = 'all'
    f['speaker'] = f.speaker_id; f['word'] = f.normalized_word
    f['speaker_word'] = f['speaker'] + '\x1f' + f['word']
    f['utt'] = f.speaker_id + '\x1f' + f.utterance_id.astype(str)
    vc = {'S': '0 + C(speaker)', 'W': '0 + C(word)', 'SW': '0 + C(speaker_word)'}
    if with_u: vc['U'] = '0 + C(utt)'
    m = smf.mixedlm('y ~ 1', f, groups='top_group', re_formula='0', vc_formula=vc, use_sparse=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = m.fit(reml=True, method='lbfgs', maxiter=2000, disp=False)
    names = list(m.exog_vc.names); comp = dict(zip(names, map(float, res.vcomp)))
    e = float(res.scale); tot = sum(comp.values()) + e
    return {'converged': bool(res.converged), 'variance_components': {**comp, 'e': e},
            'percent': {k: 100*v/tot for k, v in {**comp, 'e': e}.items()},
            'repeatability': comp['SW'] / (comp['SW'] + e), 'token_count': int(len(f)),
            'cell_count': int(f.speaker_word.nunique()), 'utt_count': int(f.utt.nunique()),
            'warnings': sorted({w.category.__name__ for w in caught})}
def main():
    ci = int(sys.argv[1]); df = load()
    conds = sorted(df.condition.unique()); cond = conds[ci]
    sub = df[df.condition == cond].copy()
    out = {'condition': cond, 'n_rows': int(len(sub)), 'features': {}}
    for feat in FEATS:
        out['features'][feat] = {}
        for tag, wu in (('no_u', False), ('with_u', True)):
            t0 = time.time()
            try:
                r = fit(sub, feat, wu); r['seconds'] = time.time() - t0
                out['features'][feat][tag] = {'status': 'ok', **r}
                print('%s %s %s R=%.4f SW%%=%.2f U%%=%s e%%=%.2f n=%d (%.0fs)' % (cond, feat, tag,
                      r['repeatability'], r['percent']['SW'],
                      ('%.2f' % r['percent']['U']) if wu else '-', r['percent']['e'],
                      r['token_count'], r['seconds']), flush=True)
            except Exception as exc:
                out['features'][feat][tag] = {'status': 'failed', 'error': '%s: %s' % (type(exc).__name__, exc)}
                print('%s %s %s FAILED %s' % (cond, feat, tag, exc), flush=True)
    open('%s/utt_%s.json' % (OUT, cond), 'w').write(json.dumps(out, indent=1, default=float))
    print('wrote', '%s/utt_%s.json' % (OUT, cond), flush=True)
if __name__ == '__main__': main()
