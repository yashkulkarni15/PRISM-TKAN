#!/bin/bash
#SBATCH --job-name=pri2m-p3-conformal
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase3_conformal_%j.log
#SBATCH --error=logs/phase3_conformal_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Split-conformal calibration of a TRAINED Phase 3 TKAN checkpoint (eval-only, no
# retraining). Runs scripts/conformal_phase3.py against one run's best_auc.pt (or
# all five folds as a cross-fold ensemble) and the Phase 2 NPZ signal tree,
# writing conformal_manifest.json + a coverage/abstention figure under
# results/phase3/xai/conformal/<TAG> and results/figures/research_update_assets/.
#
# CPU-only: forward inference + numpy calibration; no GPU needed.
#
# Single fold (default = fold-2, the strongest run):
#   env RUN_ID=a654129d2cdc460093e21373f31ee5a5 TAG=fold2 \
#       sbatch --export=ALL slurm/phase3_conformal.sbatch
#
# Cross-fold ENSEMBLE (space-separated absolute checkpoint paths):
#   env TAG=ensemble CHECKPOINTS="$PWD/results/phase3/runs/<r0>/checkpoints/best_auc.pt \
#       $PWD/results/phase3/runs/<r1>/checkpoints/best_auc.pt ..." \
#       sbatch --export=ALL slurm/phase3_conformal.sbatch

set -euo pipefail

# NOTE: the live Great Lakes project root is /nfs/turbo/coe-jaabell/PRI2M
# (the .../PRI2M/SINDy_DDE subdir is a stale copy; do not use it).
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"

RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/results/phase3/runs}"
RUN_ID="${RUN_ID:-a654129d2cdc460093e21373f31ee5a5}"   # default: fold 2 (AUC 0.975)
TAG="${TAG:-fold2}"

# Checkpoint selection: either an explicit space-separated CHECKPOINTS list
# (cross-fold ensemble) or a single RUN_ID-derived best_auc.pt.
CHECKPOINTS="${CHECKPOINTS:-$RUNS_DIR/$RUN_ID/checkpoints/best_auc.pt}"

NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
PREPROCESSING="${PREPROCESSING:-}"   # blank -> runner auto-discovers next to the FIRST checkpoint
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/conformal/$TAG}"
ASSETS_DIR="${ASSETS_DIR:-$PROJECT_ROOT/results/figures/research_update_assets}"
N_WINDOWS="${N_WINDOWS:-512}"
ALPHA="${ALPHA:-0.1}"                # headline target miscoverage (coverage 1-alpha)
ALPHAS="${ALPHAS:-0.2,0.1,0.05,0.02,0.01}"
CAL_FRAC="${CAL_FRAC:-0.5}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR" "$ASSETS_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

echo "=== PRI2M Phase 3 split-conformal calibration ==="
echo "Checkpoints: $CHECKPOINTS"
echo "Tag:         $TAG"
echo "NPZ root:    $NPZ_ROOT"
echo "Table:       $TABLE"
echo "Output:      $OUTPUT_DIR"
echo "Alpha:       $ALPHA  (grid $ALPHAS)"
echo "Started:     $(date)"

# Expand the (space-separated) CHECKPOINTS list into repeated --checkpoint flags.
CKPT_FLAGS=()
for c in $CHECKPOINTS; do
  if [[ ! -f "$c" ]]; then
    echo "ERROR: checkpoint not found: $c" >&2
    exit 1
  fi
  CKPT_FLAGS+=(--checkpoint "$c")
done

PRE_FLAG=()
if [[ -n "$PREPROCESSING" ]]; then
  PRE_FLAG=(--preprocessing "$PREPROCESSING")
fi

python scripts/conformal_phase3.py \
  "${CKPT_FLAGS[@]}" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --output-dir "$OUTPUT_DIR" \
  --research-assets-dir "$ASSETS_DIR" \
  --n-windows "$N_WINDOWS" \
  --alpha "$ALPHA" \
  --alphas "$ALPHAS" \
  --cal-frac "$CAL_FRAC" \
  --tag "$TAG" \
  "${PRE_FLAG[@]}"

echo "Finished: $(date)"
echo "Manifest: $OUTPUT_DIR/conformal_manifest.json"
echo "Figure:   $ASSETS_DIR/conformal_$TAG.png"
