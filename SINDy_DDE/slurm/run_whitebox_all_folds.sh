#!/bin/bash
# Submit the white-box XAI analysis for ALL five Phase 3 folds (eval-only, CPU).
# Run this ON GREAT LAKES from the live project root:
#   cd /nfs/turbo/coe-jaabell/PRI2M && bash slurm/run_whitebox_all_folds.sh
#
# Uses the un-ceiled runner (seeded random stratified sampling) with
# N_WINDOWS=512 for headroom below the 0.99 sample-AUC ceiling
# (docs/phase3_tkan/PROGRESS.md "CURRENT FRONTIER" caveat 1).
#
# fold2 already has a 256-window un-ceiled result (results/phase3/xai/fold2);
# it is re-run here under TAG=fold2_w512 so the 5-fold replication table is
# uniform at 512 windows without overwriting the existing artifact set.
#
# Run IDs / val AUCs = the 5-fold sweep best_auc.pt checkpoints
# (PROGRESS.md, jobs 51670644 + 51711709).

set -euo pipefail

# Overridable per-sweep knobs; the defaults reproduce this script's previous
# behavior exactly (512 windows, and the runner's own TSR defaults of 4 / 8).
# Raise the TSR sample for the whole 5-fold sweep without editing this file:
#   TSR_WINDOWS=256 bash slurm/run_whitebox_all_folds.sh
N_WINDOWS="${N_WINDOWS:-512}"        # dataset-level sample per fold
TSR_WINDOWS="${TSR_WINDOWS:-4}"      # windows attributed with TSR
TSR_STEPS="${TSR_STEPS:-8}"          # time-step chunks per TSR window

submit () {  # fold_tag run_id val_auc
  env RUN_ID="$2" TAG="$1" VAL_AUC="$3" N_WINDOWS="$N_WINDOWS" \
      TSR_WINDOWS="$TSR_WINDOWS" TSR_STEPS="$TSR_STEPS" \
    sbatch --export=ALL slurm/phase3_whitebox.sbatch
}

submit fold0      61c67bddbdbd49ba9727ff92f4814c86 0.9597
submit fold1      c21e58dc0e624e1dbc602ec63a2dc23e 0.9122
submit fold2_w512 a654129d2cdc460093e21373f31ee5a5 0.9752
submit fold3      3a40b71514cd41b98916330c000d700a 0.9605
submit fold4      0937883918354ed899a61afb90aac14f 0.9695

echo ""
echo "Submitted 5 white-box jobs (CPU, standard partition, up to 72h wall; the"
echo "512-window + raised-TSR run overran the old 24h limit, so the sbatch now"
echo "requests 72h -- standard partition MaxTime is 14 days, so this is fine)."
echo "Watch:  squeue -u \$USER"
echo "Then pull artifacts to the Mac (one rsync, run LOCALLY):"
echo "  rsync -av yashkulk@greatlakes.arc-ts.umich.edu:/nfs/turbo/coe-jaabell/PRI2M/results/phase3/xai/ \\"
echo "    ~/Downloads/PRI2M/SINDy_DDE/results/phase3/xai/"
