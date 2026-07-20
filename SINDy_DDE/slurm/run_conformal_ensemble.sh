#!/bin/bash
# Submit the five-fold cross-checkpoint conformal ensemble.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
RUNS="$PROJECT_ROOT/results/phase3/runs"
CKPTS=""
for run_id in 61c67bddbdbd49ba9727ff92f4814c86 c21e58dc0e624e1dbc602ec63a2dc23e \
              a654129d2cdc460093e21373f31ee5a5 3a40b71514cd41b98916330c000d700a \
              0937883918354ed899a61afb90aac14f; do
  CKPTS+="$RUNS/$run_id/checkpoints/best_auc.pt "
done
env TAG=ensemble CHECKPOINTS="$CKPTS" sbatch --export=ALL slurm/phase3_conformal.sbatch
