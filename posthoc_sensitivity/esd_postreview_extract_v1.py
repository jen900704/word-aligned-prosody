#!/usr/bin/env python3
"""POST HOC (post-review, 2026-09-09). Re-extract ESD word features under perturbed
word boundaries and with an external F0 tracker (Praat, via parselmouth).

Frozen gate decisions are NOT touched. This script only produces alternative
feature columns for a sensitivity refit (esd_postreview_refit_v1.py).

Variants (boundaries in seconds, d = end - start):
  orig       (start, end)                 replicates the frozen adapter (checked)
  shift_m20  (start-.02, end-.02)         translation
  shift_p20  (start+.02, end+.02)
  expand20   (start-.02, end+.02)
  shrink20   (start+.02, end-.02)         None if d < .06
  jit20      start,end + N(0,.02) indep.  seed 13, None if new d < .02
  jit40      start,end + N(0,.04) indep.
  core50     (start+.25d, end-.25d)       central half, boundary-free
  praat      Praat ac pitch at (start,end): mean / P10-P90 of voiced frames
"""
import csv, json, math, os, sys, time
import numpy as np, soundfile as sf
from multiprocessing import Pool

FEAT = os.path.join(os.environ.get('WAP_WORK_ROOT', 'work'), 'prosody_esd_x2/full_v1_run3/acoustics_raw/esd_word_features_PRIVATE.csv')
MANI = os.path.join(os.environ.get('WAP_WORK_ROOT', 'work'), 'prosody_esd_x1/stage1_final_v3_run1/esd_manifest_PRIVATE.csv')
OUT = os.path.join(os.environ.get('WAP_WORK_ROOT', 'work'), 'prosody_esd_x2/postreview_sensitivity_v1/esd_postreview_features_PRIVATE.csv')
SEED = 13
VARIANTS = ['orig', 'shift_m20', 'shift_p20', 'expand20', 'shrink20', 'jit20', 'jit40', 'core50']

# ---- verbatim copy of the frozen adapter's analysis (esd_stage2_adapter.py) ----
def frames(signal, size, hop):
    if not len(signal): return []
    if len(signal) <= size: return [np.pad(signal, (0, size - len(signal)))]
    return [signal[i:i + size] for i in range(0, len(signal) - size + 1, hop)]

def analyze_word(signal, sr):
    size = round(.040 * sr); hop = round(.010 * sr); chunks = frames(signal, size, hop); window = np.hanning(size); pitches = []
    min_lag = max(1, int(sr / 500.0)); max_lag = min(size - 2, int(sr / 60.0))
    for chunk in chunks:
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < .0001: continue
        x = (chunk - chunk.mean()) * window; nfft = 1 << (2 * size - 1).bit_length(); spectrum = np.fft.rfft(x, nfft); ac = np.fft.irfft(spectrum * np.conj(spectrum), nfft)[:size]
        if ac[0] <= 0: continue
        search = ac[min_lag:max_lag + 1] / ac[0]; best = int(np.argmax(search))
        if search[best] >= .30: pitches.append(sr / float(best + min_lag))
    p = np.asarray(pitches, float); rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))) if len(signal) else np.nan
    lo = float(np.percentile(p, 10)) if len(p) else None; hi = float(np.percentile(p, 90)) if len(p) else None
    return {"f0_mean_hz": float(np.mean(p)) if len(p) else None, "f0_p10_hz": lo, "f0_p90_hz": hi,
            "f0_robust_range_hz": hi - lo if lo is not None else None,
            "energy_db": max(-100.0, 20 * np.log10(max(rms, 1e-5))) if np.isfinite(rms) else None, "n_voiced": int(len(p))}

def read_interval(mono, sr, start, end):
    # mirrors read_audio_interval: seek(round(start*sr)), read round((end-start)*sr) samples
    a = round(start * sr); n = round((end - start) * sr)
    return mono[a:a + n]
# -------------------------------------------------------------------------------

def variant_bounds(start, end, rng):
    d = end - start; out = {'orig': (start, end),
        'shift_m20': (start - .02, end - .02), 'shift_p20': (start + .02, end + .02),
        'expand20': (start - .02, end + .02), 'shrink20': (start + .02, end - .02) if d >= .06 else None,
        'core50': (start + .25 * d, end - .25 * d)}
    for name, s in (('jit20', .02), ('jit40', .04)):
        a = start + rng.normal(0, s); b = end + rng.normal(0, s)
        out[name] = (a, b) if b - a >= .02 else None
    return out

def praat_word(pitch_t, pitch_f, start, end):
    m = (pitch_t >= start) & (pitch_t < end) & (pitch_f > 0)
    p = pitch_f[m]
    if len(p) == 0: return {'praat__f0_mean_hz': None, 'praat__f0_robust_range_hz': None, 'praat__n_voiced': 0}
    return {'praat__f0_mean_hz': float(p.mean()), 'praat__f0_robust_range_hz': float(np.percentile(p, 90) - np.percentile(p, 10)), 'praat__n_voiced': int(len(p))}

def process(job):
    sample_id, audio_path, tokens = job
    import parselmouth
    mono, sr = sf.read(audio_path, dtype='float32', always_2d=True); mono = mono.mean(axis=1)
    if sr != 16000:
        n = max(1, round(len(mono) * 16000 / sr)); mono = np.interp(np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono).astype('float32'); sr = 16000
    dur = len(mono) / sr
    snd = parselmouth.Sound(mono.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=60.0, pitch_ceiling=500.0)
    pt = pitch.xs(); pf = pitch.selected_array['frequency']
    # per-utterance seeded RNG so results are reproducible regardless of worker order
    rng = np.random.default_rng([SEED, int(sample_id.split(':')[0]), {'Angry': 1, 'Happy': 2, 'Neutral': 3, 'Sad': 4, 'Surprise': 5}[sample_id.split(':')[1]], int(sample_id.split(':')[2])])
    rows = []
    for t in tokens:
        start, end = float(t['word_start_sec']), float(t['word_end_sec'])
        rec = {k: t[k] for k in ('sample_id', 'speaker_id', 'condition', 'prompt_id', 'utterance_id', 'word_index', 'normalized_word', 'authoritative_syllable_count', 'word_start_sec', 'word_end_sec')}
        rec['csv__f0_mean_hz'] = t['f0_mean_hz']; rec['csv__f0_robust_range_hz'] = t['f0_robust_range_hz']; rec['csv__energy_db'] = t['energy_db']
        for name, b in variant_bounds(start, end, rng).items():
            if b is None: rec.update({f'{name}__f0_mean_hz': None, f'{name}__f0_robust_range_hz': None, f'{name}__energy_db': None, f'{name}__word_duration_sec': None, f'{name}__n_voiced': None}); continue
            s, e = max(0.0, b[0]), min(dur, b[1])
            if e - s < 0.005: rec.update({f'{name}__f0_mean_hz': None, f'{name}__f0_robust_range_hz': None, f'{name}__energy_db': None, f'{name}__word_duration_sec': None, f'{name}__n_voiced': None}); continue
            a = analyze_word(read_interval(mono, sr, s, e), sr)
            rec.update({f'{name}__f0_mean_hz': a['f0_mean_hz'], f'{name}__f0_robust_range_hz': a['f0_robust_range_hz'], f'{name}__energy_db': a['energy_db'], f'{name}__word_duration_sec': e - s, f'{name}__n_voiced': a['n_voiced']})
        rec.update(praat_word(pt, pf, start, end))
        rows.append(rec)
    return rows

def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    with open(MANI, newline='', encoding='utf-8') as f: audio = {r['sample_id']: r['audio_path'] for r in csv.DictReader(f)}
    with open(FEAT, newline='', encoding='utf-8') as f: feat = list(csv.DictReader(f))
    by = {}
    for r in feat:
        if r['alignment_status'] != 'aligned' or r['feature_status'] != 'complete': continue
        by.setdefault(r['sample_id'], []).append(r)
    jobs = [(sid, audio[sid], toks) for sid, toks in sorted(by.items())]
    if limit: jobs = jobs[:limit]
    print(f'utterances {len(jobs)} tokens {sum(len(j[2]) for j in jobs)} nproc {nproc}', flush=True)
    t0 = time.time(); rows = []
    with Pool(nproc) as pool:
        for i, out in enumerate(pool.imap_unordered(process, jobs, chunksize=20)):
            rows.extend(out)
            if i % 1000 == 0: print(f'  {i} utterances {time.time() - t0:.0f}s', flush=True)
    rows.sort(key=lambda r: (r['sample_id'], int(r['word_index'])))
    fields = list(rows[0].keys())
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields, lineterminator='\n'); w.writeheader(); w.writerows(rows)
    # replication check against the frozen CSV
    d = []
    for r in rows:
        for k in ('f0_mean_hz', 'f0_robust_range_hz', 'energy_db'):
            a, b = r[f'orig__{k}'], r[f'csv__{k}']
            if a is None or b in ('', None): continue
            d.append(abs(float(a) - float(b)))
    d = np.asarray(d)
    print(json.dumps({'rows': len(rows), 'orig_vs_csv_max_abs_diff': float(d.max()), 'orig_vs_csv_n': int(len(d)), 'orig_vs_csv_frac_gt_1e-6': float((d > 1e-6).mean()), 'seconds': time.time() - t0}), flush=True)
    print('wrote', OUT, flush=True)

if __name__ == '__main__': main()
