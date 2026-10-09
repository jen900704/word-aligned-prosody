#!/usr/bin/env python3
"""AMENDMENT N (2026-09-09, post-review). Rebuild the MSP analysis input with the
pooled unidentified-speaker label removed BEFORE any model is fitted, then rerun
the frozen Gate M pipeline unchanged on that input.

Reason: 'Unknown' is not a person. It carried 46% of tokens and entered the
nuisance adjustment models and token-level variance of the original run; the
earlier deletion check removed it only from the final paired-speaker set.

Nothing in the estimator changes. Only the input rows change, and the change is
an exclusion of an invalid sampling unit.
"""
import csv, json, shutil, subprocess, sys
import os
from pathlib import Path

ROOT = Path(os.environ.get('WAP_WORK_ROOT', 'work')) / 'prosody_msp_structural_v1'
SRC = ROOT / 'run1/acoustic_merge/msp_acoustics_PRIVATE.csv'
SRC_SAFE = ROOT / 'run1/acoustic_merge/msp_acoustic_merge_SAFE.json'
OUT = ROOT / 'run1/amendment_nounk_v1'
PY = os.environ.get('PYTHON', sys.executable)
CODE = Path(__file__).resolve().parent  # this repository's msp_podcast/ (contains tools/)
sys.path.insert(0, str(CODE))
from tools.msp_acoustic_extraction_runner_v2 import sha256

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / 'msp_acoustics_NOUNK_PRIVATE.csv'
    kept = dropped = 0
    spk_before, spk_after = set(), set()
    with SRC.open(newline='', encoding='utf-8') as f, dst.open('w', newline='', encoding='utf-8') as g:
        rd = csv.DictReader(f); w = csv.DictWriter(g, rd.fieldnames, lineterminator='\n'); w.writeheader()
        for r in rd:
            spk_before.add(r['speaker_id'])
            if r['speaker_id'] == 'Unknown': dropped += 1; continue
            spk_after.add(r['speaker_id']); w.writerow(r); kept += 1
    dst.chmod(0o600)
    safe = json.loads(SRC_SAFE.read_text())
    safe['output_sha256'] = sha256(dst)
    if 'merged_token_rows' in safe: safe['merged_token_rows'] = kept
    safe['amendment'] = {'id': 'N', 'date': '2026-09-09', 'action': 'pooled unidentified-speaker label excluded before model fitting',
                         'source_csv_sha256': sha256(SRC), 'rows_kept': kept, 'rows_dropped': dropped,
                         'speakers_before': len(spk_before), 'speakers_after': len(spk_after)}
    safe_dst = OUT / 'msp_acoustic_merge_NOUNK_SAFE.json'
    safe_dst.write_text(json.dumps(safe, indent=2, sort_keys=True) + '\n', encoding='utf-8'); safe_dst.chmod(0o600)
    print(json.dumps(safe['amendment'], indent=1), flush=True)
    cmd = [PY, '-m', 'tools.msp_final_analysis_v1', '--merged-acoustics', str(dst), '--merge-safe', str(safe_dst),
           '--private-result', str(OUT / 'msp_gate_m_NOUNK_PRIVATE.json'), '--safe-json', str(OUT / 'msp_gate_m_NOUNK_TECHNICAL_SAFE.json')]
    print('running:', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(CODE), capture_output=True, text=True)
    print('rc', r.returncode, flush=True)
    if r.stdout: print(r.stdout[-3000:], flush=True)
    if r.stderr: print('STDERR', r.stderr[-3000:], flush=True)
    if r.returncode: return r.returncode
    # side-by-side comparison with the original frozen run
    old = json.loads((ROOT / 'run1/final_analysis/msp_gate_m_PRIVATE.json').read_text())
    new = json.loads((OUT / 'msp_gate_m_NOUNK_PRIVATE.json').read_text())
    rows = []
    for f in sorted(new['features']):
        o, n = old['features'][f], new['features'][f]
        rows.append({'feature': f,
                     'raw_pearson_old': o['primary']['pearson'], 'raw_pearson_new': n['primary']['pearson'],
                     'corrected_old': o['primary']['corrected_pearson'], 'corrected_new': n['primary']['corrected_pearson'],
                     'delta': n['primary']['corrected_pearson'] - o['primary']['corrected_pearson'],
                     'paired_speakers_old': o['primary']['paired_speakers'], 'paired_speakers_new': n['primary']['paired_speakers'],
                     'gate_old': old['gate_m']['feature_states'][f], 'gate_new': new['gate_m']['feature_states'][f],
                     'weight_check_old': o['recording_volume']['passes'], 'weight_check_new': n['recording_volume']['passes'],
                     'valid_tokens_old': o['support']['valid_feature_tokens'], 'valid_tokens_new': n['support']['valid_feature_tokens']})
        print(f"{f:34s} raw {o['primary']['pearson']:.4f}->{n['primary']['pearson']:.4f}  corr {o['primary']['corrected_pearson']:.4f}->{n['primary']['corrected_pearson']:.4f}  gate {old['gate_m']['feature_states'][f]}->{new['gate_m']['feature_states'][f]}  N {o['primary']['paired_speakers']}->{n['primary']['paired_speakers']}", flush=True)
    print('gate_m overall:', old['gate_m'].get('passes'), '->', new['gate_m'].get('passes'), flush=True)
    (OUT / 'AMENDMENT_N_COMPARISON_SAFE.json').write_text(json.dumps(
        {'amendment': safe['amendment'], 'comparison': rows,
         'gate_m_overall_old': old['gate_m'].get('passes'), 'gate_m_overall_new': new['gate_m'].get('passes')}, indent=1) + '\n')
    print('wrote', OUT / 'AMENDMENT_N_COMPARISON_SAFE.json', flush=True)
    return 0

if __name__ == '__main__': raise SystemExit(main())
