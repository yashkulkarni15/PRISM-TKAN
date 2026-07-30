#!/bin/bash
#SBATCH --job-name=pri2m-p3-regen
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase3_regenerative_%j.log
#SBATCH --error=logs/phase3_regenerative_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=FAIL

# U13-C: did the trained TKAN LEARN the regenerative difference x(t) - x(t-tau)?
#
# The plan's qualitative criterion is "the symbolic expression contains
# x(t) - x(t-tau) without hardcoding". This is the most direct available test of
# it. Every delay edge is in=68 (delay:xtpos) or in=69 (delay:ytpos) on
# encoder.layers.0.kan_{r,z,h}. If the network implements regeneration, then for
# a given (kan, out) unit the DELAYED-channel edge should be approximately the
# NEGATIVE of the instantaneous-channel edge on that same unit -- because that
# composition IS x(t) - x(t-tau). The runner evaluates both phi via
# KANLinear.edge_function over the range the data actually visits and correlates.
#
# Built-in control: folds 0/1/2 carry 5 delay edges each in the top 25, folds 3/4
# carry 1 -- the same split the ablation shows (drops 0.230/0.487/0.266 vs
# 0.484/0.449). A regenerative story should track that split.
#
# Eval-only and CPU-light: no training, no gradients through time -- it samples
# windows once and evaluates spline edges. Independent of every other U13 job.
#
#   sbatch --export=ALL slurm/phase3_regenerative.sbatch
#   env RUN_ID=<run> TAG=fold0 sbatch --export=ALL slurm/phase3_regenerative.sbatch
#
# Run it per fold to get the 5-fold picture (each is ~minutes, not hours):
#   for f in 0 1 2 3 4; do ... done   -- see slurm/run_regenerative_all_folds.sh

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
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/regenerative/$TAG}"

# SYMBOLIC_ONLY=1 selects the runner's CHECKPOINT-FREE FALLBACK, which scores
# matched pairs straight out of a white-box symbolic_edges.csv and RETURNS
# WITHOUT EVER LOADING THE NETWORK (regenerative_phase3.py:220 ->
# _run_symbolic_mode). It is a different, weaker analysis -- NOT a cross-mark
# bolted onto the real one. Passing --symbolic-csv unconditionally is exactly
# what made the 2026-07-29 batch exit 0 in ~90 seconds having written only
# regenerative_symbolic.json. Leave this at 0 to run the real test.
SYMBOLIC_ONLY="${SYMBOLIC_ONLY:-0}"
SYMBOLIC_CSV="${SYMBOLIC_CSV:-$PROJECT_ROOT/results/phase3/xai/$TAG/symbolic_edges.csv}"

N_WINDOWS="${N_WINDOWS:-128}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
POOL="${POOL:-4}"
TOP_K="${TOP_K:-6}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

echo "=== PRI2M Phase 3 regenerative-spline test (U13-C) ==="
echo "Checkpoint:  $CHECKPOINT"
echo "Tag:         $TAG"
echo "Output:      $OUTPUT_DIR"
echo "N windows:   $N_WINDOWS  (sample seed $SAMPLE_SEED, pool $POOL)"
echo "Started:     $(date)"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

EXTRA=()
if [[ -n "$PREPROCESSING" ]]; then EXTRA+=(--preprocessing "$PREPROCESSING"); fi

# --symbolic-csv is MUTUALLY EXCLUSIVE with the real run -- it short-circuits
# the whole analysis -- so it is opt-in and never inferred from a file existing.
if [[ "$SYMBOLIC_ONLY" == "1" ]]; then
  if [[ ! -f "$SYMBOLIC_CSV" ]]; then
    echo "ERROR: SYMBOLIC_ONLY=1 but no CSV at $SYMBOLIC_CSV" >&2
    exit 1
  fi
  echo "Mode:        SYMBOLIC-ONLY fallback (network never loaded) <- $SYMBOLIC_CSV"
  EXTRA+=(--symbolic-csv "$SYMBOLIC_CSV")
else
  echo "Mode:        full regenerative test against the checkpoint"
fi

python scripts/regenerative_phase3.py \
  --checkpoint "$CHECKPOINT" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --output-dir "$OUTPUT_DIR" \
  --n-windows "$N_WINDOWS" \
  --sample-seed "$SAMPLE_SEED" \
  --pool "$POOL" \
  --top-k "$TOP_K" \
  --tag "$TAG" \
  "${EXTRA[@]}"

echo "Finished: $(date)"

# Assert the artifact rather than announcing it. The first version echoed this
# path unconditionally, so five jobs that had silently taken the symbolic
# short-circuit still printed "Manifest: .../regenerative.json" and read as
# successes in the log.
EXPECT="$OUTPUT_DIR/regenerative.json"
[[ "$SYMBOLIC_ONLY" == "1" ]] && EXPECT="$OUTPUT_DIR/regenerative_symbolic.json"
if [[ -s "$EXPECT" ]]; then
  echo "Manifest: $EXPECT ($(wc -c < "$EXPECT") bytes)"
else
  echo "ERROR: expected artifact missing: $EXPECT" >&2
  exit 1
fi
echo
echo "READ IT AS: a strong NEGATIVE correlation between the instantaneous and"
echo "delayed edges of the same unit is the regenerative difference emerging from"
echo "learned functions. The random-edge null in the same file is the control --"
echo "a negative correlation that the null also shows is not evidence."
