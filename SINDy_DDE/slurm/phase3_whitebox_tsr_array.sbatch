#!/bin/bash
#SBATCH --job-name=pri2m-p3-tsr
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/phase3_tsr_%A_%a.log
#SBATCH --error=logs/phase3_tsr_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=FAIL

# One shard of the white-box Temporal Saliency Rescaling stage. TSR is the only
# stage whose cost scales with --tsr-windows (O(tsr_windows * L * tsr_steps *
# channels) serial forward/backward passes), so at a publishable 128-256 windows
# it blew past the 24h monolith wall. This array splits the window set across
# tasks: task k computes global positions p with (p % NSHARDS == k) and writes
# results/phase3/xai/<TAG>/tsr_shards/shard_k_of_NSHARDS.json.
#
# NSHARDS is taken from the array size, so the array range IS the shard count:
#   sbatch --array=0-31  -> 32 shards  (default; 128 windows -> 4 windows/task)
#   sbatch --array=0-63  -> 64 shards  (256 windows -> 4 windows/task)
# Each task costs ~= the TSR portion of the old TSR=4 run, BUT per-window cost is
# heavy-tailed (scales with each window's effective seq length), so under modulo
# sharding an unlucky shard that draws long windows can exceed 6h. The 2026-07-24
# run timed out a ~15% subset of shards per fold at exactly 6h; wall raised to 24h
# to swallow the tail. For a hard imbalance fix instead, raise NSHARDS so each task
# does 1-2 windows (see below). Resubmit only the missing shards with
# slurm/resubmit_missing_tsr_shards.sh (keeps the shards that already finished).
#
# The seeded window sample MUST match the --skip-tsr fixed run: pass the SAME
# RUN_ID / TAG / N_WINDOWS / TSR_WINDOWS / TSR_STEPS / SAMPLE_SEED. Normally you
# do not invoke this directly -- slurm/run_whitebox_sharded.sh wires the whole
# fixed-run + array + merge DAG per fold. Standalone example:
#   env RUN_ID=<run> TAG=fold2_w512 N_WINDOWS=512 TSR_WINDOWS=128 \
#       sbatch --array=0-31 --export=ALL slurm/phase3_whitebox_tsr_array.sbatch
#
# After every task finishes, merge:
#   python scripts/merge_tsr_shards.py --output-dir results/phase3/xai/<TAG>

set -euo pipefail

# NOTE: the live Great Lakes project root is /nfs/turbo/coe-jaabell/PRI2M
# (the .../PRI2M/SINDy_DDE subdir is a stale copy; do not use it).
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"

RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/results/phase3/runs}"
RUN_ID="${RUN_ID:-a654129d2cdc460093e21373f31ee5a5}"   # default: fold 2 (AUC 0.975)
CHECKPOINT="${CHECKPOINT:-$RUNS_DIR/$RUN_ID/checkpoints/best_auc.pt}"
TAG="${TAG:-fold2}"

NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
PREPROCESSING="${PREPROCESSING:-}"   # blank -> runner auto-discovers
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/$TAG}"
N_WINDOWS="${N_WINDOWS:-512}"        # MUST match the fixed --skip-tsr run
SAMPLE_SEED="${SAMPLE_SEED:-0}"      # MUST match the fixed --skip-tsr run
TSR_WINDOWS="${TSR_WINDOWS:-128}"    # total TSR windows to shard across the array
TSR_STEPS="${TSR_STEPS:-8}"

# Shard count = array size. SLURM_ARRAY_TASK_COUNT is the number of tasks;
# SLURM_ARRAY_TASK_MAX+1 equals it for a 0-based contiguous array and is robust
# to --array being given as a subset. Prefer an explicit NSHARDS override if set.
NSHARDS="${NSHARDS:-$((SLURM_ARRAY_TASK_MAX + 1))}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

echo "=== PRI2M Phase 3 white-box TSR shard ==="
echo "Checkpoint:  $CHECKPOINT"
echo "Tag:         $TAG"
echo "Output:      $OUTPUT_DIR"
echo "N windows:   $N_WINDOWS   (sample seed $SAMPLE_SEED)"
echo "TSR windows: $TSR_WINDOWS"
echo "TSR steps:   $TSR_STEPS"
echo "Shard:       $SHARD of $NSHARDS"
echo "Started:     $(date)"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

PRE_FLAG=()
if [[ -n "$PREPROCESSING" ]]; then
  PRE_FLAG=(--preprocessing "$PREPROCESSING")
fi

python scripts/whitebox_phase3.py \
  --checkpoint "$CHECKPOINT" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --output-dir "$OUTPUT_DIR" \
  --n-windows "$N_WINDOWS" \
  --sample-seed "$SAMPLE_SEED" \
  --tsr-windows "$TSR_WINDOWS" \
  --tsr-steps "$TSR_STEPS" \
  --tsr-only \
  --tsr-shard "$SHARD" \
  --tsr-nshards "$NSHARDS" \
  --run-id "$RUN_ID" \
  --tag "$TAG" \
  "${PRE_FLAG[@]}"

echo "Finished shard $SHARD/$NSHARDS: $(date)"
