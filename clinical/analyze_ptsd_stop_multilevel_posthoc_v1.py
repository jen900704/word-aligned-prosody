#!/usr/bin/env python3
"""POST HOC (2026-09-11). Person-mean-centred within-person association between daily
prosody summaries and same-day PCL in PTSD-STOP. Reuses the frozen session summariser and
daily-PCL loader from the prespecified change analysis; nothing about extraction or
summarisation changes. Aggregates only are printed; participant rows stay on this server."""
import sys, json, math
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
V1 = Path(os.environ.get('WAP_PTSD_ROOT', 'work/ptsd_stop')) / 'ptsd_stop_wordlocal_v1'; sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ptsd_stop_within_person_pcl_v1 import structural_sessions, session_summary, load_day_pcl, MIN_SESSION_TOKENS
from ptsd_stop_aggregate_reliability_blind_v1 import FEATURES as AGG, apply_identity_authority, load_tokens
from prepare_ptsd_stop_gate_licensed_inputs_v1 import FEATURE_MAP
from gated_clinical_correlation import benjamini_hochberg
import statsmodels.formula.api as smf
D = Path(os.environ.get('WAP_PTSD_ROOT', 'work/ptsd_stop')) / 'ptsd_stop_wordlocal_allspeech_v2'
OUT = D/'posthoc_multilevel_v1'
SEED, B = 13, 1000
def cluster_boot(df, fn):
    rng = np.random.default_rng(np.random.PCG64(SEED))
    ids = df['participant_id'].unique(); groups = {k: g for k, g in df.groupby('participant_id')}
    vals = []
    for _ in range(B):
        pick = rng.choice(ids, size=len(ids), replace=True)
        boot = pd.concat([groups[k] for k in pick], ignore_index=True)
        try: vals.append(fn(boot))
        except Exception: pass
    vals = np.array([v for v in vals if np.isfinite(v)])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(len(vals))) if len(vals) else (None, None, 0)
def within_rho(df):
    return float(spearmanr(df['x_wc'], df['y_wc']).correlation)
def main():
    frame = apply_identity_authority(load_tokens(D/'tokens_PRIVATE'), V1/'identity_authority_PRIVATE.csv')
    temporal, flow = structural_sessions(frame, V1/'identity_authority_PRIVATE.csv')
    labels = load_day_pcl('mysql')
    res = {'protocol': 'ptsd_stop_within_person_multilevel_posthoc_v1', 'post_hoc': True, 'seed': SEED,
           'bootstrap_replicates': B, 'min_session_tokens': MIN_SESSION_TOKENS, 'structural_flow': flow,
           'outcome_source': 'outcomes_v8_2_stratified.PCL[is_outcome_present=1,is_prospective_time=0]',
           'model': 'PCL ~ x_wc + x_pm, random intercept by participant (REML); x_wc = day value minus person mean',
           'features': []}
    pvals = []
    for raw in AGG:
        feat = FEATURE_MAP[raw]
        s = session_summary(temporal, raw)
        j = s.merge(labels, on=['participant_id', 'authoritative_user_day_id'], how='inner', validate='one_to_one')
        n_days = j.groupby('participant_id')['authoritative_date'].nunique()
        keep = n_days[n_days >= 2].index
        j = j[j['participant_id'].isin(keep)].copy()
        j['x_pm'] = j.groupby('participant_id')['acoustic'].transform('mean'); j['x_wc'] = j['acoustic'] - j['x_pm']
        j['y_pm'] = j.groupby('participant_id')['PCL_SCORE'].transform('mean'); j['y_wc'] = j['PCL_SCORE'] - j['y_pm']
        rec = {'feature': feat, 'n_participants': int(j['participant_id'].nunique()), 'n_days': int(len(j)),
               'median_days_per_participant': float(n_days[keep].median()),
               'sd_within_x': float(j['x_wc'].std()), 'sd_between_x': float(j.groupby('participant_id')['x_pm'].first().std()),
               'sd_within_y': float(j['y_wc'].std()), 'sd_between_y': float(j.groupby('participant_id')['y_pm'].first().std())}
        rho = within_rho(j); lo, hi, nb = cluster_boot(j, within_rho)
        rec.update({'within_person_spearman': rho, 'within_ci': [lo, hi], 'within_boot_successes': nb})
        z = j.copy(); z['x_wc_z'] = z['x_wc']/z['x_wc'].std(); z['x_pm_z'] = (z['x_pm']-z['x_pm'].mean())/z['x_pm'].std()
        try:
            m = smf.mixedlm('PCL_SCORE ~ x_wc_z + x_pm_z', z, groups=z['participant_id']).fit(reml=True)
            rec['mixedlm'] = {'b_within_per_sd': float(m.params['x_wc_z']), 'se_within': float(m.bse['x_wc_z']), 'p_within': float(m.pvalues['x_wc_z']),
                              'b_between_per_sd': float(m.params['x_pm_z']), 'se_between': float(m.bse['x_pm_z']), 'p_between': float(m.pvalues['x_pm_z']),
                              'converged': bool(m.converged), 'group_var': float(m.cov_re.iloc[0, 0]), 'resid_var': float(m.scale)}
            pvals.append(rec['mixedlm']['p_within'])
        except Exception as e:
            rec['mixedlm'] = {'error': str(e)[:200]}; pvals.append(1.0)
        try:
            ms = smf.mixedlm('PCL_SCORE ~ x_wc_z + x_pm_z', z, groups=z['participant_id'], re_formula='~x_wc_z').fit(reml=True)
            rec['random_slope'] = {'b_within_per_sd': float(ms.params['x_wc_z']), 'p_within': float(ms.pvalues['x_wc_z']),
                                   'slope_var': float(ms.cov_re.loc['x_wc_z', 'x_wc_z']), 'converged': bool(ms.converged)}
        except Exception as e:
            rec['random_slope'] = {'error': str(e)[:200]}
        res['features'].append(rec)
        print(f"{feat:32s} N={rec['n_participants']} days={rec['n_days']} within_rho={rho:+.4f} CI[{lo:+.3f},{hi:+.3f}] "
              f"b_within/SD={rec['mixedlm'].get('b_within_per_sd', float('nan')):+.4f} p={rec['mixedlm'].get('p_within', float('nan')):.4f} "
              f"b_between/SD={rec['mixedlm'].get('b_between_per_sd', float('nan')):+.4f}", flush=True)
    q = benjamini_hochberg(pvals)
    for rec, qq in zip(res['features'], q): rec['bh_q_within'] = float(qq)
    print('BH q (within):', [round(float(x), 4) for x in q], flush=True)
    (OUT/'within_person_multilevel_POSTHOC_PRIVATE.json').write_text(json.dumps(res, indent=1, default=str))
    safe = {k: v for k, v in res.items() if k != 'features'}; safe['features'] = res['features']
    (OUT/'within_person_multilevel_POSTHOC_SAFE.json').write_text(json.dumps(safe, indent=1, default=str))
    print('wrote', OUT/'within_person_multilevel_POSTHOC_SAFE.json', flush=True)
if __name__ == '__main__': main()
