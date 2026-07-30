#!/bin/bash
#SBATCH --job-name=pri2m-p3-tsr-control
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase3_tsr_control_%j.log
#SBATCH --error=logs/phase3_tsr_control_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=FAIL

# U12-B: is the TSR delay-lock learned, or an artifact of how the input is built?
#
# The 2026-07-23/24 sharded sweep found that when TSR resolves a peak at all, the
# peak lands on that window's own tau far beyond chance (34% exact, p = 0.00005).
# That is only evidence about the MODEL if it is not a by-product of the input
# construction -- the delayed channel is the signal shifted by tau, so a lag-tau
# autocorrelation bump can appear with no learning at all. This runs three
# conditions on the same windows: the trained net, the SAME architecture with
# re-initialised weights (the decisive one), and the trained net on shuffled taus.
#
# WHY THIS IS CHEAP, AND WHY THAT IS NOT A COMPROMISE.
# The sweep needed 32 shards x 5 folds and still lost 27 shards to a 6h wall,
# because TSR with --feature-masking costs ~tsr_windows*(L + 0.5*L*C)*tsr_steps
# serial fwd/bwd passes. This control does NOT need that:
#   * It reads only tr_lock_z, which derives from step 1's `time_relevance`.
#     `feature_masking` selects how step 2 scores the FEATURE dimension, so the
#     lag statistics are bit-identical either way -- pinned by
#     tests/test_tsr_control.py::test_feature_masking_does_not_change_the_lag_statistics.
#     Leaving it off is a free ~3x.
#   * It attributes an effect the sweep already measured, so a few dozen windows
#     on ONE fold settles it. Do not re-run the 128-window five-fold sweep here.
# Defaults below (24 windows, 4 steps, no feature masking) fit the 12h wall with
# room to spare. Raise TSR_WINDOWS only if the verdict comes back UNDECIDED.
#
#   sbatch --export=ALL slurm/phase3_tsr_control.sbatch
#   env RUN_ID=<other-fold-run> TAG=fold3 sbatch --export=ALL slurm/phase3_tsr_control.sbatch

set -euo pipefail

# NOTE: the live Great Lakes project root is /nfs/turbo/coe-jaabell/PRI2M
# (the .../PRI2M/SINDy_DDE subdir is a stale copy; do not use it).
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"

RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/results/phase3/runs}"
RUN_ID="${RUN_ID:-a654129d2cdc460093e21373f31ee5a5}"   # default: fold 2 (AUC 0.975)
CHECKPOINT="${CHECKPOINT:-$RUNS_DIR/$RUN_ID/checkpoints/best_auc.pt}"
TAG="${TAG:-fold2_w512}"

NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
PREPROCESSING="${PREPROCESSING:-}"   # blank -> runner auto-discovers
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/tsr_control/$TAG}"

# MUST match the sweep so the windows are the same ones the effect was measured on.
N_WINDOWS="${N_WINDOWS:-512}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
TSR_WINDOWS="${TSR_WINDOWS:-24}"     # attribution, not measurement -- a few dozen is enough
TSR_STEPS="${TSR_STEPS:-4}"
THRESHOLD="${THRESHOLD:-0.8}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

echo "=== PRI2M Phase 3 TSR construction-artifact control (U12-B) ==="
echo "Checkpoint:  $CHECKPOINT"
echo "Tag:         $TAG"
echo "Output:      $OUTPUT_DIR"
echo "N windows:   $N_WINDOWS  (sample seed $SAMPLE_SEED)"
echo "TSR windows: $TSR_WINDOWS   steps: $TSR_STEPS"
echo "Started:     $(date)"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

PRE_FLAG=()
if [[ -n "$PREPROCESSING" ]]; then
  PRE_FLAG=(--preprocessing "$PREPROCESSING")
fi

python scripts/tsr_control_phase3.py \
  --checkpoint "$CHECKPOINT" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --output-dir "$OUTPUT_DIR" \
  --n-windows "$N_WINDOWS" \
  --sample-seed "$SAMPLE_SEED" \
  --tsr-windows "$TSR_WINDOWS" \
  --tsr-steps "$TSR_STEPS" \
  --threshold "$THRESHOLD" \
  --run-id "$RUN_ID" \
  --tag "$TAG" \
  "${PRE_FLAG[@]}"

echo "Finished: $(date)"
echo "Read the VERDICT line above. UNDECIDED is not a pass -- it means a lock was"
echo "not measurable on one side, so TSR stays unclaimable either way."
