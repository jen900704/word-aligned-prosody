#!/usr/bin/env bash
#SBATCH --job-name=msp_ac_merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-merge-%j.out
set -euo pipefail
umask 077
ROOT=${WAP_WORK_ROOT:?set WAP_WORK_ROOT}/prosody_msp_structural_v1
RUN=$ROOT/run1
cd "${WAP_REPO:?set WAP_REPO to this repository}/msp_podcast"
${PYTHON:-python3} -m tools.msp_acoustic_merge_v1 \
 --shard-dir "$RUN/acoustics_v1/shards" --num-shards 32 \
 --preflight "$RUN/preflight_v4/preflight_guard_v4_AUTHORIZED_SAFE.json" \
 --alignment-safe "$RUN/content_qa/retry2/msp_textgrid_content_qa_SAFE.json" \
 --final-safe "$RUN/final_scope/msp_final_scope_SAFE.json" \
 --token-csv "$RUN/content_qa/retry2/msp_word_manifest_PRIVATE.csv" \
 --final-csv "$RUN/final_scope/msp_final_scope_PRIVATE.csv" \
 --output-csv "$RUN/acoustics_v1/merged/msp_acoustics_PRIVATE.csv" \
 --safe-json "$RUN/acoustics_v1/merged/msp_acoustic_merge_SAFE.json"
chmod 600 "$RUN/acoustics_v1/merged/msp_acoustics_PRIVATE.csv" "$RUN/acoustics_v1/merged/msp_acoustic_merge_SAFE.json"
