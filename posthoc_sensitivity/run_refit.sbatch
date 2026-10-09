#!/bin/bash
#SBATCH -J pr_refit
#SBATCH -p cpu
#SBATCH -c 9
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH -o refit_%j.out
cd "${WAP_REPO:?set WAP_REPO to this repository}/posthoc_sensitivity"
PY=${PYTHON:-python3}
for v in orig jit20 jit40 core50 praat shift_m20 shift_p20 expand20 shrink20; do
  $PY esd_postreview_refit_v1.py $v > "$WAP_WORK_ROOT/refit_$v.log" 2>&1 &
done
wait
echo ALL_REFITS_DONE
