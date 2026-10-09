#!/usr/bin/env bash
#SBATCH --job-name=esd2c_final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

: "${CODE_DIR:?set CODE_DIR}" "${RESCUE_ROOT:?set RESCUE_ROOT}"
: "${STAGE1_MANIFEST:?set STAGE1_MANIFEST}" "${STAGE2B_AGGREGATE:?set STAGE2B_AGGREGATE}"
: "${STAGE2B_CLOSURE:?set STAGE2B_CLOSURE}" "${EXPECTED_CLOSURE_SHA256:?set EXPECTED_CLOSURE_SHA256}"
PYTHON_BIN=${PYTHON:-python3}
"$PYTHON_BIN" "$CODE_DIR/esd_stage2c_parallel.py" merge-all \
  --features "$RESCUE_ROOT/features/f0_mean_hz.json" "$RESCUE_ROOT/features/f0_robust_range_hz.json" \
    "$RESCUE_ROOT/features/energy_db.json" "$RESCUE_ROOT/features/duration_log_syllable_residual.json" \
  --stage1-manifest "$STAGE1_MANIFEST" --stage2b-aggregate "$STAGE2B_AGGREGATE" \
  --stage2b-closure "$STAGE2B_CLOSURE" --expected-closure-sha256 "$EXPECTED_CLOSURE_SHA256" \
  --rescue-output-root "$RESCUE_ROOT" \
  --output "$RESCUE_ROOT/frozen_stage2c_parallel_final.json"
