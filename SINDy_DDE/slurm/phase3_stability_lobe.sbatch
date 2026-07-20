#!/bin/bash
#SBATCH --job-name=pri2m-p3-sld
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase3_sld_%j.log
#SBATCH --error=logs/phase3_sld_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Model-implied Stability Lobe Diagram (SLD) for a TRAINED Phase 3 TKAN
# checkpoint (eval-only, no retraining). Runs scripts/stability_lobe_phase3.py
# on one modal group's real windows, binning them by (omega, b) to read the
# model-implied stability boundary and score it against the analytic blim(omega).
#
# CPU-only: small-batch inference over one param-twin group; no GPU needed.
#
# Default = fold-2 checkpoint on param-twin group parent_group 29 (datasets
# 29/89/119). parent_group 29 maps to FOLD 2 in results/phase3/phase3_folds.json,
# so the fold-2 run holds it out of training -> an honest, held-out SLD.
#
# Parameterize with env overrides, e.g.:
#   env RUN_ID=a654129d2cdc460093e21373f31ee5a5 FOLD=2 PARENT_GROUP=29 TAG=fold2_ds29grp \
#       sbatch --export=ALL slurm/phase3_stability_lobe.sbatch
#   # explicit checkpoint + explicit dataset ids instead:
#   env CHECKPOINT=$PWD/results/phase3/runs/<run>/checkpoints/best_auc.pt \
#       DATASET_IDS=29,89,119 TAG=fold2_ds29grp \
#       sbatch --export=ALL slurm/phase3_stability_lobe.sbatch

set -euo pipefail

# NOTE: the live Great Lakes project root is /nfs/turbo/coe-jaabell/PRI2M
# (the .../PRI2M/SINDy_DDE subdir is a stale copy; do not use it).
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"

RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/results/phase3/runs}"
RUN_ID="${RUN_ID:-a654129d2cdc460093e21373f31ee5a5}"   # default: fold 2 (val AUC 0.975)
CHECKPOINT="${CHECKPOINT:-$RUNS_DIR/$RUN_ID/checkpoints/best_auc.pt}"
FOLD="${FOLD:-2}"
TAG="${TAG:-fold2_ds29grp}"

# Modal group: parent_group (all param-twins) OR an explicit dataset id list.
PARENT_GROUP="${PARENT_GROUP:-29}"
DATASET_IDS="${DATASET_IDS:-}"        # blank -> use PARENT_GROUP

NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
PREPROCESSING="${PREPROCESSING:-}"    # blank -> runner auto-discovers next to the ckpt
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/$TAG/sld}"
ASSETS_DIR="${ASSETS_DIR:-$PROJECT_ROOT/results/figures/research_update_assets}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR" "$ASSETS_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

echo "=== PRI2M Phase 3 stability lobe diagram (SLD) ==="
echo "Checkpoint:   $CHECKPOINT"
echo "Fold:         $FOLD"
echo "Tag:          $TAG"
echo "Group:        ${DATASET_IDS:-parent_group $PARENT_GROUP}"
echo "NPZ root:     $NPZ_ROOT"
echo "Table:        $TABLE"
echo "Output:       $OUTPUT_DIR"
echo "Started:      $(date)"

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
# Group flag: DATASET_IDS wins over PARENT_GROUP when set.
if [[ -n "$DATASET_IDS" ]]; then
  GROUP_FLAG=(--dataset-ids "$DATASET_IDS")
else
  GROUP_FLAG=(--parent-group "$PARENT_GROUP")
fi

python scripts/stability_lobe_phase3.py \
  --checkpoint "$CHECKPOINT" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --output-dir "$OUTPUT_DIR" \
  --assets-dir "$ASSETS_DIR" \
  --fold "$FOLD" \
  --tag "$TAG" \
  --run-id "$RUN_ID" \
  "${GROUP_FLAG[@]}" \
  "${VAL_FLAG[@]}" \
  "${PRE_FLAG[@]}"

echo "Finished: $(date)"
echo "SLD manifest: $OUTPUT_DIR/sld_manifest.json"
echo "Figure:       $OUTPUT_DIR/sld_$TAG.png (mirrored to $ASSETS_DIR)"
