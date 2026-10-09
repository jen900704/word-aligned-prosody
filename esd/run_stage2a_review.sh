#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON:-python3}
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ $# -ne 1 ]]; then echo "usage: $0 SMOKE_ROOT" >&2; exit 2; fi
"$PYTHON_BIN" "$HERE/review_stage2a_smoke.py" --smoke-root "$1"
