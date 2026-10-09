#!/usr/bin/env bash
#SBATCH --job-name=msp_preflight
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077
ROOT=${WAP_WORK_ROOT:?set WAP_WORK_ROOT}/prosody_msp_structural_v1
PY=${PYTHON:-python3}
OUT=$ROOT/run1/preflight_v4
TMP=$OUT/preflight_guard_v4_AUTHORIZED_SAFE.json.tmp
FINAL=$OUT/preflight_guard_v4_AUTHORIZED_SAFE.json
rm -f "$TMP" "$FINAL"
$PY "${WAP_REPO:?set WAP_REPO to this repository}/msp_podcast/tools/msp_preflight_guard_v1.py" \
  --structural "$ROOT/run1/msp_structural_projection_SAFE.json" \
  --alignment "$ROOT/run1/content_qa/retry2/msp_textgrid_content_qa_SAFE.json" \
  --final-scope "$ROOT/run1/final_scope/msp_final_scope_SAFE.json" \
  --support-report "$ROOT/run1/structural_support/msp_structural_support_report_SAFE.json" \
  --model-lock "${MSP_MODEL_LOCK:?set MSP_MODEL_LOCK to the frozen MSP model-lock JSON}" > "$TMP"
$PY - "$TMP" <<'PY'
import json,sys
p=sys.argv[1]; o=json.load(open(p,encoding='utf-8'))
assert o.get('audit')=='msp_acoustic_preflight_guard_v4'
assert o.get('msp_audio_processing_authorized') is True
assert all(o.get('checks',{}).values())
assert o.get('scientific_choices_made_by_guard') is False
PY
mv "$TMP" "$FINAL"
chmod 600 "$FINAL"
echo preflight_v4_authorized=1
sha256sum "$FINAL"
