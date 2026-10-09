#!/usr/bin/env bash
#SBATCH --job-name=curve_nounk
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=curve_%j.out
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=${WAP_WORK_ROOT:?set WAP_WORK_ROOT}/prosody_msp_structural_v1
RUN=$ROOT/run1
cd "${WAP_REPO:?set WAP_REPO to this repository}/msp_podcast"
${PYTHON:-python3} -m tools.msp_aggregation_curve_J1_v1 \
 --merged-acoustics "$RUN/amendment_nounk_v1/msp_acoustics_NOUNK_PRIVATE.csv" \
 --merge-safe "$RUN/amendment_nounk_v1/msp_acoustic_merge_NOUNK_SAFE.json" \
 --output "$RUN/amendment_nounk_v1/msp_aggregation_curve_J1_NOUNK_SAFE.json"
echo done
