#!/usr/bin/env bash
# Prospective template only. Running this file explicitly submits the rescue DAG.
set -euo pipefail
: "${CODE_DIR:?absolute tools/clsp_esd_reliability directory}"
: "${STAGE1_MANIFEST:?}"; : "${STAGE2B_AGGREGATE:?}"; : "${STAGE2B_CLOSURE:?}"
: "${EXPECTED_CLOSURE_SHA256:?}"; : "${RESCUE_ROOT:?new rescue directory, never stage2c_real_v1}"
[[ "$RESCUE_ROOT" == /* && "$(basename "$RESCUE_ROOT")" != stage2c_real_v1 ]] || exit 2

bootstrap_job=$(sbatch --parsable "$CODE_DIR/slurm_stage2c_parallel_bootstrap.sbatch")
core_job=$(sbatch --parsable "$CODE_DIR/slurm_stage2c_parallel_core.sbatch")
merge_job=$(sbatch --parsable --dependency=afterok:"$bootstrap_job":"$core_job" \
  "$CODE_DIR/slurm_stage2c_parallel_merge_feature.sbatch")
sbatch --parsable --dependency=afterok:"$merge_job" \
  "$CODE_DIR/slurm_stage2c_parallel_merge_all.sbatch"
