#!/usr/bin/env bash
#SBATCH --job-name=msp_curve_j1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=${WAP_WORK_ROOT:?set WAP_WORK_ROOT}/prosody_msp_structural_v1
RUN=$ROOT/run1
mkdir -p "$RUN/aggregation_curve_J1"
cd "${WAP_REPO:?set WAP_REPO to this repository}/msp_podcast"
${PYTHON:-python3} -m tools.msp_aggregation_curve_J1_v1 \
 --merged-acoustics "$RUN/acoustics_v1/merged/msp_acoustics_PRIVATE.csv" \
 --merge-safe "$RUN/acoustics_v1/merged/msp_acoustic_merge_SAFE.json" \
 --output "$RUN/aggregation_curve_J1/msp_aggregation_curve_J1_SAFE.json"
chmod 600 "$RUN/aggregation_curve_J1/msp_aggregation_curve_J1_SAFE.json"
echo done
