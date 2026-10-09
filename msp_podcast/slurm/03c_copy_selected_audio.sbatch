#!/usr/bin/env bash
#SBATCH --job-name=msp_audio_copy
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077
ROOT=${WAP_WORK_ROOT:?set WAP_WORK_ROOT}/prosody_msp_structural_v1
SRC=${MSP_PODCAST_ROOT:?set MSP_PODCAST_ROOT}/Audios
OUT=$ROOT/run1/selected_audio_transport
DST=$OUT/audio
LIST=$OUT/selected_audio_filenames_PRIVATE.txt
PY=${PYTHON:-python3}
mkdir -p "$DST"
cd "${WAP_REPO:?set WAP_REPO to this repository}/msp_podcast"
expected=$(wc -l < "$LIST")
[ "$expected" -eq 63707 ]
rc=1
for attempt in 1 2 3 4 5; do
  echo "rsync_attempt=$attempt"
  if rsync -a --files-from="$LIST" "$SRC/" "$DST/"; then rc=0; break; fi
  rc=$?
  echo "rsync_rc=$rc"
  sleep 60
done
[ "$rc" -eq 0 ]
$PY -m tools.msp_selected_audio_transport_v1 verify \
  --private-list "$LIST" \
  --manifest-safe "$OUT/selected_audio_manifest_SAFE.json" \
  --dest-root "$DST" \
  --safe-json "$OUT/selected_audio_transport_VERIFY_SAFE.json"
chmod 600 "$OUT/selected_audio_transport_VERIFY_SAFE.json"
echo selected_audio_transport_complete=1
sha256sum "$OUT/selected_audio_transport_VERIFY_SAFE.json"
