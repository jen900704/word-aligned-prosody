#!/usr/bin/env bash
#SBATCH --job-name=msp_tgqa_r2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077
ROOT=${WAP_WORK_ROOT:?set WAP_WORK_ROOT}/prosody_msp_structural_v1
OUT=$ROOT/run1/content_qa/retry2
python3 "${WAP_REPO:?set WAP_REPO to this repository}/msp_podcast/tools/msp_textgrid_content_qa_v1.py" \
  --zip "$ROOT/run1/content_qa/local_inputs/ForceAligned.local.zip" \
  --inventory "$ROOT/run1/package_intake/forcealigned_inventory_PRIVATE.txt" \
  --structural-csv "$ROOT/run1/msp_structural_projection_PRIVATE.csv" \
  --cmudict "${WAP_RESOURCE_DIR:?set WAP_RESOURCE_DIR}/cmudict.dict" \
  --private-token-manifest "$OUT/msp_word_manifest_PRIVATE.csv" \
  --eligible-filenames "$OUT/msp_eligible_filenames_PRIVATE.txt" \
  --safe-json "$OUT/msp_textgrid_content_qa_SAFE.json"
chmod 600 "$OUT/msp_word_manifest_PRIVATE.csv" "$OUT/msp_eligible_filenames_PRIVATE.txt" "$OUT/msp_textgrid_content_qa_SAFE.json"
