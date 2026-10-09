#!/bin/bash
#SBATCH -J pr_extract
#SBATCH -p cpu
#SBATCH -c 12
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o extract_%j.out
cd "${WAP_REPO:?set WAP_REPO to this repository}/posthoc_sensitivity"
python3 esd_postreview_extract_v1.py 12
