#!/usr/bin/env bash
#SBATCH --job-name=msp_support
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077
ROOT=${WAP_WORK_ROOT:?set WAP_WORK_ROOT}/prosody_msp_structural_v1
PY=${PYTHON:-python3}
OUT=$ROOT/run1/structural_support
cd "${WAP_REPO:?set WAP_REPO to this repository}/msp_podcast"
$PY -m tools.msp_structural_support_report_v1 \
  --structural-csv "$ROOT/run1/msp_structural_projection_PRIVATE.csv" \
  --eligible-filenames "$ROOT/run1/content_qa/retry2/msp_eligible_filenames_PRIVATE.txt" \
  --final-private-manifest "$ROOT/run1/final_scope/msp_final_scope_PRIVATE.csv" \
  --safe-json "$OUT/msp_structural_support_report_SAFE.json"
chmod 600 "$OUT/msp_structural_support_report_SAFE.json"
echo support_report_complete=1
sha256sum "$OUT/msp_structural_support_report_SAFE.json"
