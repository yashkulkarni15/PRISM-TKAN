#!/bin/bash
#SBATCH --job-name=pri2m-p3-dynamics
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase3_hidden_dynamics_%j.log
#SBATCH --error=logs/phase3_hidden_dynamics_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Hidden-state dynamical analysis (recurrent-Jacobian eigenvalues) of a TRAINED
# Phase 3 TKAN checkpoint (eval-only, no retraining). Runs
# scripts/hidden_dynamics_phase3.py against one run's best_auc.pt and the Phase 2
# NPZ signal tree, testing whether chatter windows push the encoder's internal
# dynamics past |lambda| = 1 (a delayed-system bifurcation) while stable windows
# stay contractive. Writes hidden_dynamics_manifest.json + a distribution figure
# under results/phase3/xai/dynamics/<TAG> (+ a research-update copy).
#
# CPU-only: small-batch inference + a per-timestep autograd Jacobian; no GPU.
#
# Parameterize with env overrides, e.g. (default = fold-2, the strongest run):
#   env RUN_ID=a654129d2cdc460093e21373f31ee5a5 TAG=fold2 \
#       sbatch --export=ALL slurm/phase3_hidden_dynamics.sbatch
#   # explicit checkpoint instead of RUN_ID:
#   env CHECKPOINT=$PWD/results/phase3/runs/<run>/checkpoints/best_auc.pt TAG=fold0 \
#       sbatch --export=ALL slurm/phase3_hidden_dynamics.sbatch

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
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/dynamics/$TAG}"
N_WINDOWS="${N_WINDOWS:-128}"        # balanced stable+chatter windows to sample
POOL="${POOL:-4}"                    # temporal pool factor (MUST match training)
T_STRIDE="${T_STRIDE:-1}"            # compute J_t every T_STRIDE pooled steps
LAYER="${LAYER:-0}"                  # encoder recurrent layer (0 = delay-aware)
REDUCE="${REDUCE:-mean}"             # per-window reduction of step spectral radii
SLOW_POINTS="${SLOW_POINTS:-1}"      # 1 -> also run the slow/fixed-point probe

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

echo "=== PRI2M Phase 3 hidden-state dynamics ==="
echo "Checkpoint: $CHECKPOINT"
echo "Tag:        $TAG"
echo "NPZ root:   $NPZ_ROOT"
echo "Table:      $TABLE"
echo "Output:     $OUTPUT_DIR"
echo "N windows:  $N_WINDOWS  pool=$POOL  t_stride=$T_STRIDE  layer=$LAYER  reduce=$REDUCE"
echo "Started:    $(date)"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  echo "Set RUN_ID or CHECKPOINT (see header)." >&2
  exit 1
fi

PRE_FLAG=()
if [[ -n "$PREPROCESSING" ]]; then
  PRE_FLAG=(--preprocessing "$PREPROCESSING")
fi
VAL_AUC="${VAL_AUC:-}"
VAL_FLAG=()
if [[ -n "$VAL_AUC" ]]; then
  VAL_FLAG=(--val-auc "$VAL_AUC")
fi
SLOW_FLAG=()
if [[ "$SLOW_POINTS" == "1" ]]; then
  SLOW_FLAG=(--slow-points)
fi

python scripts/hidden_dynamics_phase3.py \
  --checkpoint "$CHECKPOINT" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --output-dir "$OUTPUT_DIR" \
  --n-windows "$N_WINDOWS" \
  --pool "$POOL" \
  --t-stride "$T_STRIDE" \
  --layer "$LAYER" \
  --reduce "$REDUCE" \
  --run-id "$RUN_ID" \
  --tag "$TAG" \
  "${SLOW_FLAG[@]}" \
  "${VAL_FLAG[@]}" \
  "${PRE_FLAG[@]}"

echo "Finished: $(date)"
echo "Manifest: $OUTPUT_DIR/hidden_dynamics_manifest.json"
