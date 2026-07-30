#!/bin/bash
# U13-C: submit the regenerative-spline test for ALL five Phase 3 folds.
# Run this ON GREAT LAKES from the live project root:
#   cd /nfs/turbo/coe-jaabell/PRI2M && bash slurm/run_regenerative_all_folds.sh
#
# Eval-only and CPU-light -- no training and no backprop through time, just
# spline-edge evaluation over sampled windows. Independent of every other job,
# so this can go in alongside the TSR control and the counterfactual run.
#
# Run IDs / tags are the SAME five best_auc.pt checkpoints the white-box sweep
# used (slurm/run_whitebox_all_folds.sh), so the symbolic_edges.csv each job
# cross-references is the one written for that exact fold.
#
# WHY ALL FIVE AND NOT ONE: the control is the fold split itself. Folds 0/1/2
# carry 5 delay edges each in the top 25 and folds 3/4 carry 1, matching the
# ablation drops (0.230/0.487/0.266 vs 0.484/0.449). A regenerative signal that
# is real should track that split; one that appears uniformly across folds is
# more likely a property of the architecture than of what was learned.

set -euo pipefail

N_WINDOWS="${N_WINDOWS:-128}"
POOL="${POOL:-4}"
TOP_K="${TOP_K:-6}"

submit () {  # fold_tag run_id
  env RUN_ID="$2" TAG="$1" N_WINDOWS="$N_WINDOWS" POOL="$POOL" TOP_K="$TOP_K" \
    sbatch --export=ALL slurm/phase3_regenerative.sbatch
}

submit fold0      61c67bddbdbd49ba9727ff92f4814c86
submit fold1      c21e58dc0e624e1dbc602ec63a2dc23e
submit fold2_w512 a654129d2cdc460093e21373f31ee5a5
submit fold3      3a40b71514cd41b98916330c000d700a
submit fold4      0937883918354ed899a61afb90aac14f

echo ""
echo "Submitted 5 regenerative jobs (CPU, standard partition, 4h wall each)."
echo "Watch:  squeue -u \$USER"
echo "Each writes results/phase3/xai/regenerative/<tag>/regenerative.json"
echo "Pull them with ./fetch_phase3_results_from_hpc.sh (run LOCALLY)."
