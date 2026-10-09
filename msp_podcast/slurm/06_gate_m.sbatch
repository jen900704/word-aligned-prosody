#!/usr/bin/env bash
#SBATCH --job-name=msp_gate_m
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077
ROOT=${WAP_WORK_ROOT:?set WAP_WORK_ROOT}/prosody_msp_structural_v1
PY=${PYTHON:-python3}
OUT=$ROOT/run1/final_analysis
cd "${WAP_REPO:?set WAP_REPO to this repository}/msp_podcast"
$PY -m tools.msp_final_analysis_v1 \
  --merged-acoustics "$ROOT/run1/acoustic_merge/msp_acoustics_PRIVATE.csv" \
  --merge-safe "$ROOT/run1/acoustic_merge/msp_acoustic_merge_SAFE.json" \
  --private-result "$OUT/msp_gate_m_PRIVATE.json" \
  --safe-json "$OUT/msp_gate_m_TECHNICAL_SAFE.json"
chmod 600 "$OUT/msp_gate_m_PRIVATE.json" "$OUT/msp_gate_m_TECHNICAL_SAFE.json"
echo primary_gate_m_computation_complete=1
sha256sum "$OUT/msp_gate_m_TECHNICAL_SAFE.json"
