#!/usr/bin/env python3
"""POST HOC (2026-09-09, review round 3). Recompute ESD F0 mean / F0 range after
excluding tokens whose word interval yields fewer than T voiced frames, because a
P90-P10 range from 1-2 voiced frames is not a meaningful range. Frozen estimator
used verbatim. Gate decisions are NOT recomputed.
usage: python esd_voiced_sensitivity_v1.py 3 5
"""
import csv, json, sys, time
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'esd'))
import esd_reml_reliability as R
IN = os.path.join(os.environ.get('WAP_WORK_ROOT', 'work'), 'prosody_esd_x2/postreview_sensitivity_v1/esd_postreview_features_PRIVATE.csv')
OUT = Path(os.path.join(os.environ.get('WAP_WORK_ROOT', 'work'), 'prosody_esd_x2/postreview_sensitivity_v1/voiced'))
FEATS = ['f0_mean_hz', 'f0_robust_range_hz']
def load(minv):
    rows, seen_spk, kept, tot = [], set(), 0, 0
    with open(IN, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tot += 1
            try: nv = int(float(r['orig__n_voiced']))
            except Exception: nv = 0
            if nv < minv: continue
            kept += 1; seen_spk.add(r['speaker_id'])
            rows.append({'sample_id': r['sample_id'], 'speaker_id': r['speaker_id'],
                'condition': r['condition'], 'prompt_id': r['prompt_id'],
                'utterance_id': r['utterance_id'], 'word_index': r['word_index'],
                'normalized_word': r['normalized_word'], 'alignment_status': 'aligned',
                'authoritative_syllable_count': r['authoritative_syllable_count'],
                'f0_mean_hz': r['orig__f0_mean_hz'], 'f0_robust_range_hz': r['orig__f0_robust_range_hz'],
                'energy_db': r['orig__energy_db'], 'word_duration_sec': r['orig__word_duration_sec']})
    return rows, kept, tot, len(seen_spk)
def pct(vc):
    t = sum(vc.values()); return {k: 100.0*v/t for k, v in vc.items()}
def run(minv):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, kept, tot, nspk = load(minv)
    rows, dur = R.residualize_duration_once(rows)
    out = {'min_voiced_frames': minv, 'tokens_kept': kept, 'tokens_total': tot,
           'token_retention': kept/tot, 'speakers': nspk, 'features': {}}
    for feat in FEATS:
        t0 = time.time()
        try:
            s = R.five_condition_summary(rows, feat)
            conds = [{'condition': c['condition'], 'repeatability': c['repeatability'],
                      'percent': pct(c['variance_components']), 'token_count': c['token_count'],
                      'matched_cell_count': c['matched_cell_count']} for c in s['condition_estimates']]
            out['features'][feat] = {'status': 'ok', 'equal_condition_mean': s['equal_condition_mean'],
                'condition_range': [min(c['repeatability'] for c in conds), max(c['repeatability'] for c in conds)],
                'mean_percent': {k: sum(c['percent'][k] for c in conds)/5 for k in ('S','W','SW','e')},
                'conditions': conds, 'seconds': time.time()-t0}
            print('[minv=%d] %s Rbar=%.4f range=[%.4f,%.4f] SW%%=%.2f cells=%d' % (minv, feat,
                s['equal_condition_mean'], out['features'][feat]['condition_range'][0],
                out['features'][feat]['condition_range'][1],
                out['features'][feat]['mean_percent']['SW'],
                sum(c['matched_cell_count'] for c in conds)), flush=True)
        except Exception as e:
            out['features'][feat] = {'status': 'failed', 'error': '%s: %s' % (type(e).__name__, e)}
            print('[minv=%d] %s FAILED %s' % (minv, feat, e), flush=True)
    (OUT / ('voiced_min%d.json' % minv)).write_text(json.dumps(out, indent=1, default=float))
    print('wrote', OUT / ('voiced_min%d.json' % minv), 'retention %.4f speakers %d' % (kept/tot, nspk), flush=True)
if __name__ == '__main__':
    for a in sys.argv[1:]: run(int(a))
