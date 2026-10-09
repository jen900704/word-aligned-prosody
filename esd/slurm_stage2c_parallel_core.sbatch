#!/usr/bin/env bash
#SBATCH --job-name=esd2c_core
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

: "${CODE_DIR:?set CODE_DIR}" "${RESCUE_ROOT:?set RESCUE_ROOT}"
: "${STAGE1_MANIFEST:?set STAGE1_MANIFEST}" "${STAGE2B_AGGREGATE:?set STAGE2B_AGGREGATE}"
: "${STAGE2B_CLOSURE:?set STAGE2B_CLOSURE}" "${EXPECTED_CLOSURE_SHA256:?set EXPECTED_CLOSURE_SHA256}"
PYTHON_BIN=${PYTHON:-python3}
FEATURES=(f0_mean_hz f0_robust_range_hz energy_db duration_log_syllable_residual)
FEATURE=${FEATURES[$SLURM_ARRAY_TASK_ID]}
"$PYTHON_BIN" "$CODE_DIR/esd_stage2c_parallel.py" core --feature "$FEATURE" \
  --stage1-manifest "$STAGE1_MANIFEST" --stage2b-aggregate "$STAGE2B_AGGREGATE" \
  --stage2b-closure "$STAGE2B_CLOSURE" --expected-closure-sha256 "$EXPECTED_CLOSURE_SHA256" \
  --rescue-output-root "$RESCUE_ROOT" \
  --output "$RESCUE_ROOT/core/${FEATURE}.json"
