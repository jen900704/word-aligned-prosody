#!/bin/bash
#SBATCH -J msp_nounk
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH -o amendment_n_%j.out
cd "${WAP_REPO:?set WAP_REPO to this repository}/msp_podcast"
${PYTHON:-python3} msp_amendment_nounk_v1.py
