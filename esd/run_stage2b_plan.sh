#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON:-python3}
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ $# -ne 4 ]]; then echo "usage: $0 MANIFEST CMUDICT MODEL_DIR OUTPUT_ROOT" >&2; exit 2; fi
MANIFEST="$1"; CMUDICT="$2"; MODEL_DIR="$3"; OUTPUT_ROOT="$4"
mkdir -p "$OUTPUT_ROOT"
"$PYTHON_BIN" "$HERE/verify_stage2_runtime.py" --cmudict "$CMUDICT" --model-dir "$MODEL_DIR" --output "$OUTPUT_ROOT/runtime_verification_SAFE.json"
"$PYTHON_BIN" "$HERE/stage2b_full.py" plan --manifest "$MANIFEST" --output-root "$OUTPUT_ROOT"
