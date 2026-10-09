#!/bin/bash
#SBATCH -J msp_wsex
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH -o withinsex_%j.out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${WAP_REPO:?set WAP_REPO to this repository}/msp_podcast"
${PYTHON:-python3} msp_withinsex_v1.py
