#!/bin/bash
#SBATCH -J esdutt
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 36:00:00
#SBATCH -a 0-4
#SBATCH -o utt_%A_%a.out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -n "${WAP_VENV:-}" ] && source "$WAP_VENV/bin/activate"
cd "${WAP_REPO:?set WAP_REPO to this repository}/posthoc_sensitivity"
python esd_utterance_facet_v1.py $SLURM_ARRAY_TASK_ID
