#!/bin/bash
#SBATCH -J pr_msp9
#SBATCH -p cpu
#SBATCH -c 10
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH -o msp9_%j.out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${WAP_REPO:?set WAP_REPO to this repository}/posthoc_sensitivity"
${PYTHON:-python3} msp_postreview_crossed_v1.py 9 20 5 10
