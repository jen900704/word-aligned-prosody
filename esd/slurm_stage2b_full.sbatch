#!/usr/bin/env bash
#SBATCH --job-name=esd_x2_full
#SBATCH --array=0-31
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=stage2b_full_%A_%a.log
set -euo pipefail
: "${CODE_DIR:?set CODE_DIR to the shared project bundle directory}"
: "${STAGE1_ROOT:?set STAGE1_ROOT}"
: "${CMUDICT_PATH:?set CMUDICT_PATH}"
: "${ALIGN_MODEL_DIR:?set ALIGN_MODEL_DIR}"
: "${FULL_OUTPUT_ROOT:?set FULL_OUTPUT_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?must run as a Slurm array}"
[[ "$CODE_DIR" == /* && "$STAGE1_ROOT" == /* && "$FULL_OUTPUT_ROOT" == /* ]] || { echo "paths must be absolute" >&2; exit 2; }
PYTHON_BIN=${PYTHON:-python3}
"$PYTHON_BIN" "$CODE_DIR/stage2b_full.py" shard \
  --manifest "$STAGE1_ROOT/esd_manifest_PRIVATE.csv" \
  --output-root "$FULL_OUTPUT_ROOT" \
  --cmudict "$CMUDICT_PATH" \
  --model-dir "$ALIGN_MODEL_DIR" \
  --shard-id "$SLURM_ARRAY_TASK_ID"
