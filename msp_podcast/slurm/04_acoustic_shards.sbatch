#!/usr/bin/env bash
#SBATCH --job-name=msp_acoustic
#SBATCH --partition=cpu
#SBATCH --array=0-31%16
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A_%a.out
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=${WAP_WORK_ROOT:?set WAP_WORK_ROOT}/prosody_msp_structural_v1
RUN=$ROOT/run1
SHARDS=$RUN/acoustics_v1/shards
mkdir -p "$SHARDS"
cd "${WAP_REPO:?set WAP_REPO to this repository}/msp_podcast"
# (cluster-specific runtime fingerprint guard omitted from the public release)
# (cluster-specific production-readiness check omitted from the public release)
sid=$SLURM_ARRAY_TASK_ID
${PYTHON:-python3} -m tools.msp_acoustic_extraction_runner_v3 \
 --preflight "$RUN/preflight_v4/preflight_guard_v4_AUTHORIZED_SAFE.json" \
 --alignment-safe "$RUN/content_qa/retry2/msp_textgrid_content_qa_SAFE.json" \
 --final-safe "$RUN/final_scope/msp_final_scope_SAFE.json" \
 --token-csv "$RUN/content_qa/retry2/msp_word_manifest_PRIVATE.csv" \
 --final-csv "$RUN/final_scope/msp_final_scope_PRIVATE.csv" \
 --audio-root "$RUN/selected_audio_transport/audio" \
 --output-csv "$SHARDS/shard_$(printf '%02d' "$sid").csv" \
 --safe-json "$SHARDS/shard_$(printf '%02d' "$sid")_SAFE.json" \
 --shard-id "$sid" --num-shards 32
chmod 600 "$SHARDS/shard_$(printf '%02d' "$sid").csv" "$SHARDS/shard_$(printf '%02d' "$sid")_SAFE.json"
