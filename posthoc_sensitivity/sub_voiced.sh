#!/bin/bash
#SBATCH -J esdvoiced
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 20:00:00
#SBATCH -o voiced_%j.out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -n "${WAP_VENV:-}" ] && source "$WAP_VENV/bin/activate"
cd "${WAP_REPO:?set WAP_REPO to this repository}/posthoc_sensitivity"
python esd_voiced_sensitivity_v1.py 3 5
