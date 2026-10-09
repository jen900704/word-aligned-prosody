#!/usr/bin/env bash
#SBATCH --job-name=esd_x2_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=stage2a_smoke_%j.log
set -euo pipefail
: "${CODE_DIR:?set CODE_DIR to the shared project bundle directory}"
: "${STAGE1_ROOT:?set STAGE1_ROOT}"
: "${CMUDICT_PATH:?set CMUDICT_PATH}"
: "${ALIGN_MODEL_DIR:?set ALIGN_MODEL_DIR}"
: "${SMOKE_OUTPUT_ROOT:?set SMOKE_OUTPUT_ROOT}"
[[ "$CODE_DIR" == /* ]] || { echo "CODE_DIR must be absolute" >&2; exit 2; }
[[ -d "$CODE_DIR" ]] || { echo "CODE_DIR directory absent" >&2; exit 2; }
CODE_DIR="$(cd -- "$CODE_DIR" && pwd -P)"
RUNNER="$CODE_DIR/run_stage2_smoke.sh"
[[ -r "$RUNNER" && -x "$RUNNER" ]] || { echo "smoke runner absent, unreadable, or non-executable" >&2; exit 2; }
for required in esd_stage2_adapter.py frozen_word_aligner.py esd_stage2_qa.py verify_stage2_runtime.py stage2_adapter_lock.json; do
  [[ -r "$CODE_DIR/$required" ]] || { echo "required tracked adapter file absent: $required" >&2; exit 2; }
done
"$CODE_DIR/run_stage2_smoke.sh" ${PYTHON:-python3} "$STAGE1_ROOT/esd_manifest_PRIVATE.csv" "$CMUDICT_PATH" "$ALIGN_MODEL_DIR" "$SMOKE_OUTPUT_ROOT"
