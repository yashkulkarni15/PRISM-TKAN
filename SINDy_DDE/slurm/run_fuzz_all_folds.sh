#!/bin/bash
# Submit the Phase 3 graded-fuzziness suite for all five trained checkpoints.
# Run on Great Lakes from /nfs/turbo/coe-jaabell/PRI2M.
set -euo pipefail

submit () {  # fold_tag run_id val_auc
  env RUN_ID="$2" TAG="$1" VAL_AUC="$3" N_WINDOWS=512 \
    sbatch --export=ALL slurm/phase3_fuzz.sbatch
}

submit fold0 61c67bddbdbd49ba9727ff92f4814c86 0.9597
submit fold1 c21e58dc0e624e1dbc602ec63a2dc23e 0.9122
submit fold2 a654129d2cdc460093e21373f31ee5a5 0.9752
submit fold3 3a40b71514cd41b98916330c000d700a 0.9605
submit fold4 0937883918354ed899a61afb90aac14f 0.9695

echo "Submitted five Phase 3 fuzz jobs."
echo "Watch: squeue -u \$USER"
