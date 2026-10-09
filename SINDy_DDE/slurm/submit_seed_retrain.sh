#!/bin/bash
# Exp 2 (paper close-out, 2026-10-05): retrain folds 0 and 2 with seeds 1, 2, 3.
#
# WHY: every one of the five original folds was trained with seed 0, so they
# share one initialisation, and the "necessary delay cluster" found by the mode
# ablation could be a property of that initialisation rather than of the task.
# Retraining the SAME split from new seeds and asking whether a functionally
# equivalent cluster comes back (scripts/recurrence_phase3.py) separates the two.
# The seed-to-seed AUC spread is also the "seed floor" -- the 0.0092 repeatability
# floor was measured at a fixed seed, so it is only a lower bound.
#
# Six jobs: fold {0,2} x seed {1,2,3}, each through the existing
# slurm/phase3_train_gpu.sbatch with deterministic kernels (DETERMINISTIC=1 also
# exports CUBLAS_WORKSPACE_CONFIG before python starts) and --seed via EXTRA_ARGS.
# Runs land in results/phase3/runs_seeds/<mlflow run id>/; the fold and seed of
# each are in its summary.json ("config": {"fold": .., "seed": ..}).
#
# Every sbatch call carries its own variables through `env VAR=... sbatch`, so
# nothing depends on what happens to be exported in the calling shell, and the
# six submissions cannot leak settings into each other.
#
# Run on Great Lakes from anywhere (it cd's to the project root):
#   bash slurm/submit_seed_retrain.sh
# Preview the six commands without submitting:
#   DRY_RUN=1 bash slurm/submit_seed_retrain.sh
#
# 16 h wall time: a 20-epoch fold is ~8.5 h on an A40 with compile; deterministic
# kernels cost some speed, and 16 h still backfills far better than the 36 h default.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/runs_seeds}"
FOLDS="${FOLDS:-0 2}"
SEEDS="${SEEDS:-1 2 3}"
TIME="${TIME:-16:00:00}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a FOLD_ARR <<< "$FOLDS"
read -r -a SEED_ARR <<< "$SEEDS"

if [[ "$DRY_RUN" != "1" ]]; then
  cd "$PROJECT_ROOT"
  mkdir -p logs "$OUTPUT_DIR"
fi

n=0
for FOLD in "${FOLD_ARR[@]}"; do
  for S in "${SEED_ARR[@]}"; do
    CMD=(env DETERMINISTIC=1 "EXTRA_ARGS=--seed $S" "OUTPUT_DIR=$OUTPUT_DIR"
         sbatch "--array=$FOLD" "--time=$TIME" "--job-name=pri2m-p3-f${FOLD}s${S}"
         --export=ALL slurm/phase3_train_gpu.sbatch)
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '%q ' "${CMD[@]}"; echo
    else
      echo "fold $FOLD seed $S: $("${CMD[@]}")"
    fi
    n=$((n + 1))
  done
done
echo "submitted=$n dry_run=$DRY_RUN output_dir=$OUTPUT_DIR"
