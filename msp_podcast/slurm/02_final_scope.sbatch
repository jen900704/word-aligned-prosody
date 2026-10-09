#!/usr/bin/env bash
#SBATCH --job-name=msp_scope
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077
ROOT=${WAP_WORK_ROOT:?set WAP_WORK_ROOT}/prosody_msp_structural_v1
OUT=$ROOT/run1/final_scope
cd "${WAP_REPO:?set WAP_REPO to this repository}/msp_podcast"
python3 -m tools.msp_final_scope_manifest_v1 \
  --structural-csv "$ROOT/run1/msp_structural_projection_PRIVATE.csv" \
  --eligible-filenames "$ROOT/run1/content_qa/retry2/msp_eligible_filenames_PRIVATE.txt" \
  --private-manifest "$OUT/msp_final_scope_PRIVATE.csv" \
  --safe-json "$OUT/msp_final_scope_SAFE.json"
chmod 600 "$OUT/msp_final_scope_PRIVATE.csv" "$OUT/msp_final_scope_SAFE.json"
