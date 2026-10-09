#!/bin/bash
#SBATCH -J pr_refit
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 08:00:00
#SBATCH -a 0-8
#SBATCH -o refit_arr_%A_%a.out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${WAP_REPO:?set WAP_REPO to this repository}/posthoc_sensitivity"
V=(orig jit20 jit40 core50 praat shift_m20 shift_p20 expand20 shrink20)
${PYTHON:-python3} esd_postreview_refit_v1.py ${V[$SLURM_ARRAY_TASK_ID]}
