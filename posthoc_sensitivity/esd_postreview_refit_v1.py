#!/usr/bin/env python3
"""POST HOC (post-review, 2026-09-09). Refit the frozen ESD crossed-REML model on
alternative feature columns produced by esd_postreview_extract_v1.py.

Uses the frozen estimator verbatim (esd_reml_reliability.five_condition_summary,
residualize_duration_once) so numbers are comparable with Table I. Point
estimates and per-condition components only; no bootstrap. Gate decisions
are not recomputed. Run inside .envs/stage2c_reml.

usage: python esd_postreview_refit_v1.py <variant> [<variant> ...]
       variants: orig shift_m20 shift_p20 expand20 shrink20 jit20 jit40 core50 praat
"""
import csv, json, math, sys, time
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'esd'))
import esd_reml_reliability as R

IN = os.path.join(os.environ.get('WAP_WORK_ROOT', 'work'), 'prosody_esd_x2/postreview_sensitivity_v1/esd_postreview_features_PRIVATE.csv')
OUTDIR = Path(os.path.join(os.environ.get('WAP_WORK_ROOT', 'work'), 'prosody_esd_x2/postreview_sensitivity_v1/refits'))
FEATURES = ['f0_mean_hz', 'f0_robust_range_hz', 'energy_db', 'duration_log_syllable_residual']

def load(variant):
    rows = []
    with open(IN, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            base = {'sample_id': r['sample_id'], 'speaker_id': r['speaker_id'], 'condition': r['condition'],
                    'prompt_id': r['prompt_id'], 'utterance_id': r['utterance_id'], 'word_index': r['word_index'],
                    'normalized_word': r['normalized_word'], 'alignment_status': 'aligned',
                    'authoritative_syllable_count': r['authoritative_syllable_count']}
            if variant == 'praat':
                base['f0_mean_hz'] = r['praat__f0_mean_hz']; base['f0_robust_range_hz'] = r['praat__f0_robust_range_hz']
                base['energy_db'] = r['orig__energy_db']; base['word_duration_sec'] = r['orig__word_duration_sec']
            else:
                for k in ('f0_mean_hz', 'f0_robust_range_hz', 'energy_db', 'word_duration_sec'):
                    base[k] = r[f'{variant}__{k}']
            rows.append(base)
    return rows

def pct(vc):
    tot = sum(vc.values()); return {k: 100.0 * v / tot for k, v in vc.items()}

def run(variant):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = load(variant)
    rows, dur_info = R.residualize_duration_once(rows)
    out = {'variant': variant, 'n_rows': len(rows), 'duration_residualization': {k: dur_info[k] for k in ('formula', 'n', 'rank')}, 'features': {}}
    for feat in FEATURES:
        t0 = time.time()
        try:
            s = R.five_condition_summary(rows, feat)
            conds = []
            for c in s['condition_estimates']:
                vc = c['variance_components']
                conds.append({'condition': c['condition'], 'repeatability': c['repeatability'], 'variance_components': vc,
                              'percent': pct(vc), 'token_count': c['token_count'], 'matched_cell_count': c['matched_cell_count']})
            mean_pct = {k: sum(x['percent'][k] for x in conds) / 5 for k in ('S', 'W', 'SW', 'e')}
            out['features'][feat] = {'status': 'ok', 'equal_condition_mean': s['equal_condition_mean'],
                                     'condition_range': [min(x['repeatability'] for x in conds), max(x['repeatability'] for x in conds)],
                                     'mean_percent': mean_pct, 'conditions': conds, 'seconds': time.time() - t0}
            print(f'[{variant}] {feat}: Rbar={s["equal_condition_mean"]:.4f} SW%={mean_pct["SW"]:.2f} e%={mean_pct["e"]:.2f} ({time.time()-t0:.0f}s)', flush=True)
        except Exception as exc:
            out['features'][feat] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'seconds': time.time() - t0}
            print(f'[{variant}] {feat}: FAILED {type(exc).__name__}: {exc}', flush=True)
    (OUTDIR / f'refit_{variant}.json').write_text(json.dumps(out, indent=1, default=float))
    print('wrote', OUTDIR / f'refit_{variant}.json', flush=True)

if __name__ == '__main__':
    for v in sys.argv[1:]: run(v)
