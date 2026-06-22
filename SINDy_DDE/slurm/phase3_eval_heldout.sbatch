#!/bin/bash
#SBATCH --job-name=pri2m-p3-eval
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase3_eval_heldout_%A_%a.log
#SBATCH --error=logs/phase3_eval_heldout_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Held-out (eval-only, NO retrain) sweep — the batch replacement for the salloc
# session that was running scripts/eval_heldout_phase3.py by hand. The eval
# script scores ONE checkpoint per call, so this loops every fold checkpoint
# over every held-out eval set in a single job. Inference is CPU-fine (the
# fold-0 salloc ran it on a standard node); no GPU is needed.
#
# Submit (from the GL project root, after rsync) — ALL folds in one job:
#   sbatch --export=ALL slurm/phase3_eval_heldout.sbatch
# Or fan out one fold per array task (5 in parallel, each ~1 h, failures isolated):
#   sbatch --array=0-4 --export=ALL slurm/phase3_eval_heldout.sbatch
#
# Override anything via env, e.g. a single fold or a different eval set:
#   env RUN_HASHES=61c67bdd sbatch --export=ALL slurm/phase3_eval_heldout.sbatch
#   env EVAL_SETS="122_125=$PWD/data/metadata/features_datasets_122_125.csv" \
#       sbatch --export=ALL slurm/phase3_eval_heldout.sbatch
#
# Outputs land at $OUT_DIR/<run_hash>_<tag>.json (same naming you already have,
# e.g. 61c67bdd_122_125.json). Re-running overwrites, so it is idempotent.

set -uo pipefail   # NB: no `-e` — one failed eval must not abort the whole sweep.

# NOTE: the live Great Lakes project root is /nfs/turbo/coe-jaabell/PRI2M
# (the .../PRI2M/SINDy_DDE subdir is a stale copy; do not use it).
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/results/phase3/runs}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/results/phase3/heldout}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-64}"
# Optional low-speed cutoff (omega < X dropped); leave empty to keep all rows
# (the fold-0 runs used no cutoff -> n_excluded=0).
EXCLUDE_OMEGA_BELOW="${EXCLUDE_OMEGA_BELOW:-}"

# The five validated 5-fold best_auc checkpoints (run-hash prefixes, from
# docs/phase3_tkan/PROGRESS.md). Glob picks up the full run-dir name.
#   fold 0 = 61c67bdd   folds 1-4 = c21e58dc a654129d 3a40b715 09378839
RUN_HASHES="${RUN_HASHES:-61c67bdd c21e58dc a654129d 3a40b715 09378839}"

# Held-out eval sets as "tag=features_csv" pairs. The tag becomes the output
# filename suffix. <<< CONFIRM these two CSV paths match what you ran in salloc.
EVAL_SETS_DEFAULT=(
  "121_125=$PROJECT_ROOT/data/metadata/features_datasets_121_125.csv"
  "122_125=$PROJECT_ROOT/data/metadata/features_datasets_122_125.csv"
)
# Allow a space-separated env override; otherwise use the defaults above.
if [[ -n "${EVAL_SETS:-}" ]]; then
  read -r -a EVAL_SETS_ARR <<< "$EVAL_SETS"
else
  EVAL_SETS_ARR=("${EVAL_SETS_DEFAULT[@]}")
fi

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

OMEGA_FLAG=()
[[ -n "$EXCLUDE_OMEGA_BELOW" ]] && OMEGA_FLAG=(--exclude-omega-below "$EXCLUDE_OMEGA_BELOW")

echo "=== PRI2M Phase 3 held-out eval sweep ==="
echo "Project:   $PROJECT_ROOT"
echo "NPZ root:  $NPZ_ROOT"
echo "Runs dir:  $RUNS_DIR"
echo "Out dir:   $OUT_DIR"
echo "Device:    $DEVICE   Batch: $BATCH_SIZE"
echo "Run hashes:$RUN_HASHES"
echo "Eval sets: ${EVAL_SETS_ARR[*]}"
echo "Started:   $(date)"

# When submitted as a job array, each task handles exactly one fold (indexed into
# RUN_HASHES by SLURM_ARRAY_TASK_ID); a plain submit loops every fold in one job.
read -r -a RUN_ARR <<< "$RUN_HASHES"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_sel="${RUN_ARR[$SLURM_ARRAY_TASK_ID]:-}"
  if [[ -z "$run_sel" ]]; then
    echo "no run hash at array index $SLURM_ARRAY_TASK_ID (have ${#RUN_ARR[@]}); nothing to do"
    exit 0
  fi
  RUNS_TO_DO=("$run_sel")
else
  RUNS_TO_DO=("${RUN_ARR[@]}")
fi

n_ok=0; n_fail=0; n_skip=0
for run in "${RUNS_TO_DO[@]}"; do
  ckpt=$(ls -d "$RUNS_DIR/${run}"*/checkpoints/best_auc.pt 2>/dev/null | head -1)
  if [[ -z "$ckpt" ]]; then
    echo "!!! SKIP run $run — no best_auc.pt under $RUNS_DIR/${run}*/checkpoints/"
    n_skip=$((n_skip+1)); continue
  fi
  for spec in "${EVAL_SETS_ARR[@]}"; do
    tag="${spec%%=*}"; csv="${spec#*=}"
    out="$OUT_DIR/${run}_${tag}.json"
    if [[ ! -f "$csv" ]]; then
      echo "!!! SKIP $run/$tag — features CSV not found: $csv"
      n_skip=$((n_skip+1)); continue
    fi
    echo "--- eval run=$run set=$tag ($(date +%H:%M:%S)) ---"
    if python scripts/eval_heldout_phase3.py \
        --checkpoint "$ckpt" \
        --features-csv "$csv" \
        --npz-root "$NPZ_ROOT" \
        --out "$out" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        "${OMEGA_FLAG[@]}"; then
      n_ok=$((n_ok+1))
    else
      echo "!!! FAILED eval run=$run set=$tag (continuing)"
      n_fail=$((n_fail+1))
    fi
  done
done

echo "=== done: $n_ok ok, $n_fail failed, $n_skip skipped — $(date) ==="
# Non-zero exit if nothing succeeded or anything failed, so SLURM marks it FAILED.
[[ "$n_ok" -gt 0 && "$n_fail" -eq 0 ]]
