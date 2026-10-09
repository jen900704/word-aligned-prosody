#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ $# -ne 5 ]]; then echo "usage: $0 PYTHON MANIFEST CMUDICT MODEL_DIR OUTPUT_ROOT" >&2; exit 2; fi
PYTHON_BIN="$1"; MANIFEST="$2"; CMUDICT="$3"; MODEL_DIR="$4"; OUTPUT_ROOT="$5"
mkdir -p "$OUTPUT_ROOT"
"$PYTHON_BIN" "$HERE/verify_stage2_runtime.py" --cmudict "$CMUDICT" --model-dir "$MODEL_DIR" --output "$OUTPUT_ROOT/runtime_verification_SAFE.json"
"$PYTHON_BIN" "$HERE/esd_stage2_adapter.py" --mode smoke --manifest "$MANIFEST" --cmudict "$CMUDICT" --model-dir "$MODEL_DIR" --output-root "$OUTPUT_ROOT/extraction_PRIVATE"
"$PYTHON_BIN" "$HERE/esd_stage2_qa.py" --input-root "$OUTPUT_ROOT/extraction_PRIVATE" --expected-utterances 10 --output "$OUTPUT_ROOT/stage2_smoke_QA_SAFE.json"
