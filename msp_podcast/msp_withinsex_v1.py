#!/usr/bin/env python3
"""POST HOC (2026-09-10): rerun the frozen Gate M pipeline separately within each
corpus-labelled sex, on the Amendment-N (NOUNK) input. Nothing in the estimator
changes; only the rows change."""
import csv, json, subprocess, sys, collections
import os
from pathlib import Path

ROOT = Path(os.environ.get('WAP_WORK_ROOT', 'work')) / 'prosody_msp_structural_v1'
SRC  = ROOT/'run1/amendment_nounk_v1/msp_acoustics_NOUNK_PRIVATE.csv'
SAFE = ROOT/'run1/amendment_nounk_v1/msp_acoustic_merge_NOUNK_SAFE.json'
OUT  = ROOT/'run1/withinsex_v1'
LAB  = Path(os.environ.get('MSP_LABELS_CSV', 'data/MSP-PODCAST-Publish-1.12/labels/labels_consensus.csv'))
PY   = os.environ.get('PYTHON', sys.executable)
CODE = Path(__file__).resolve().parent  # this repository's msp_podcast/ (contains tools/)
sys.path.insert(0, str(CODE))
from tools.msp_acoustic_extraction_runner_v2 import sha256

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sexes = collections.defaultdict(collections.Counter)
    with LAB.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            sexes[str(r['SpkrID']).strip()][str(r['Gender']).strip()] += 1
    spk2sex = {}
    for s, c in sexes.items():
        if len(c) == 1: spk2sex[s] = next(iter(c))
        else: spk2sex[s] = c.most_common(1)[0][0]
    base = json.loads(SAFE.read_text())
    with SRC.open(newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f); fields = rd.fieldnames; rows = list(rd)
    seen = collections.Counter(spk2sex.get(str(r['speaker_id']).strip(), 'UNMAPPED') for r in rows)
    spk = collections.defaultdict(set)
    for r in rows: spk[spk2sex.get(str(r['speaker_id']).strip(), 'UNMAPPED')].add(r['speaker_id'])
    print('token rows by sex:', dict(seen), flush=True)
    print('speakers by sex:', {k: len(v) for k, v in spk.items()}, flush=True)
    summary = {'token_rows_by_sex': dict(seen), 'speakers_by_sex': {k: len(v) for k, v in spk.items()}, 'results': {}}
    for sex in ('Male', 'Female'):
        sub = [r for r in rows if spk2sex.get(str(r['speaker_id']).strip()) == sex]
        tag = sex.lower()
        dst = OUT/f'msp_acoustics_{tag}_PRIVATE.csv'
        with dst.open('w', newline='', encoding='utf-8') as g:
            w = csv.DictWriter(g, fields, lineterminator='\n'); w.writeheader()
            for r in sub: w.writerow(r)
        dst.chmod(0o600)
        z = dict(base); z['output_sha256'] = sha256(dst); z['merged_token_rows'] = len(sub)
        z['post_hoc'] = {'id': 'withinsex_v1', 'date': '2026-09-10', 'sex': sex,
                         'action': 'frozen Gate M pipeline rerun within one corpus-labelled sex',
                         'source_csv_sha256': sha256(SRC), 'rows': len(sub), 'speakers': len(spk[sex])}
        zp = OUT/f'msp_acoustic_merge_{tag}_SAFE.json'
        zp.write_text(json.dumps(z, indent=2, sort_keys=True)+'\n', encoding='utf-8'); zp.chmod(0o600)
        priv = OUT/f'msp_gate_m_{tag}_PRIVATE.json'
        cmd = [PY, '-m', 'tools.msp_final_analysis_v1', '--merged-acoustics', str(dst), '--merge-safe', str(zp),
               '--private-result', str(priv), '--safe-json', str(OUT/f'msp_gate_m_{tag}_TECHNICAL_SAFE.json')]
        print('== running', sex, len(sub), 'rows', flush=True)
        r = subprocess.run(cmd, cwd=str(CODE), capture_output=True, text=True)
        print('rc', r.returncode, flush=True)
        if r.stderr: print('STDERR', r.stderr[-2500:], flush=True)
        if r.returncode: continue
        res = json.loads(priv.read_text())
        summary['results'][sex] = {}
        for feat in sorted(res['features']):
            p = res['features'][feat]['primary']
            summary['results'][sex][feat] = {'raw_pearson': p['pearson'], 'corrected_pearson': p['corrected_pearson'],
                                             'paired_speakers': p['paired_speakers'],
                                             'gate': res['gate_m']['feature_states'][feat]}
            print(f"{sex:7s} {feat:34s} raw {p['pearson']:.4f} corr {p['corrected_pearson']:.4f} N {p['paired_speakers']} gate {res['gate_m']['feature_states'][feat]}", flush=True)
    (OUT/'WITHINSEX_SUMMARY_SAFE.json').write_text(json.dumps(summary, indent=1, sort_keys=True)+'\n')
    print('wrote', OUT/'WITHINSEX_SUMMARY_SAFE.json', flush=True)
    return 0

if __name__ == '__main__': raise SystemExit(main())
