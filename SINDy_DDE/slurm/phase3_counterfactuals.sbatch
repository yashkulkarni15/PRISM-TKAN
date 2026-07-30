#!/bin/bash
#SBATCH --job-name=pri2m-p3-cfs
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase3_counterfactuals_%j.log
#SBATCH --error=logs/phase3_counterfactuals_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Physics-constrained counterfactuals for the Phase 3 chatter model (eval-only).
# Runs scripts/counterfactuals_phase3.py: for every chatter operating point it
# searches the actionable (omega, b) plane for the smallest move that flips the
# model to stable, then validates that move against the analytic stability
# boundary blim(omega) read straight from phase3_dataset.csv. Writes a
# counterfactuals_manifest.json + a (omega, b)-plane figure under
# results/phase3/xai/counterfactuals/ (and research_update_assets/).
#
# CPU-only + light (grid search over 2 axes + a small sklearn surrogate).
#
# Real-network counterfactuals (2 steps):
#   1) env RUN_ID=<run> TAG=fold2 sbatch --export=ALL slurm/phase3_predict_windows.sbatch
#   2) env DATASET=all TAG=fold2_net PREDICTIONS=$PROJECT_ROOT/results/phase3/preds_fold2.csv \
#          CHECKPOINT=$RUNS_DIR/<run>/checkpoints/best_auc.pt sbatch --export=ALL slurm/phase3_counterfactuals.sbatch
#
# Parameterize with env overrides, e.g.:
#   env DATASET=1 TAG=fold2 sbatch --export=ALL slurm/phase3_counterfactuals.sbatch
#   env DATASET=all TAG=all \
#       PREDICTIONS=$PWD/results/phase3/preds_fold2.csv \
#       CHECKPOINT=$PWD/results/phase3/runs/<run>/checkpoints/best_auc.pt \
#       TAG=fold2 sbatch --export=ALL slurm/phase3_counterfactuals.sbatch
#
# With PREDICTIONS (the model's per-window chatter probs) the run audits the
# MODEL against physics; without it, it is the physics-consistency sanity check
# (a boundary-faithful classifier should agree with blim almost everywhere).

set -euo pipefail

# NOTE: the live Great Lakes project root is /nfs/turbo/coe-jaabell/PRI2M
# (the .../PRI2M/SINDy_DDE subdir is a stale copy; do not use it).
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"

TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
DATASET="${DATASET:-1}"
TAG="${TAG:-fold2}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/counterfactuals}"
FIG_DIR="${FIG_DIR:-$PROJECT_ROOT/results/figures/research_update_assets}"
CHECKPOINT="${CHECKPOINT:-}"     # optional provenance
PREDICTIONS="${PREDICTIONS:-}"   # optional model per-window probs (file,prob/severity)

# U13-A: Monte Carlo survival (ON by default -- it is the missing half of plan
# criterion 7). U13-B: the real DiCE arm (OFF by default; needs PREDICTIONS,
# because DiCE is wired as a regressor and the label surrogate is a classifier).
MC_DRAWS="${MC_DRAWS:-512}"
MC_SEED="${MC_SEED:-0}"
MC_MARGIN="${MC_MARGIN:-0.0}"
DICE="${DICE:-0}"
DICE_TOTAL_CFS="${DICE_TOTAL_CFS:-2}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR" "$FIG_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

echo "=== PRI2M Phase 3 physics-constrained counterfactuals ==="
echo "Table:       $TABLE"
echo "Dataset(s):  $DATASET"
echo "Tag:         $TAG"
echo "Checkpoint:  ${CHECKPOINT:-<none>}"
echo "Predictions: ${PREDICTIONS:-<none, using table labels>}"
echo "Output:      $OUTPUT_DIR"
echo "Started:     $(date)"

echo "MC draws:    $MC_DRAWS (seed $MC_SEED, margin $MC_MARGIN)"
echo "DiCE:        $DICE"

EXTRA=()
if [[ -n "$CHECKPOINT" ]]; then EXTRA+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "$PREDICTIONS" ]]; then EXTRA+=(--predictions "$PREDICTIONS"); fi
EXTRA+=(--mc-draws "$MC_DRAWS" --mc-seed "$MC_SEED" --mc-margin "$MC_MARGIN")

if [[ "$DICE" == "1" ]]; then
  # Preflight, not a crash: the runner already degrades to an error field if
  # dice_ml is absent, but saying so up front beats reading it out of JSON.
  if python -c "import dice_ml" 2>/dev/null; then
    echo "dice_ml:     present"
  else
    echo "dice_ml:     MISSING -- the DiCE arm will report an error field;" >&2
    echo "             survival + agreement results are unaffected." >&2
  fi
  EXTRA+=(--dice --dice-total-cfs "$DICE_TOTAL_CFS")
fi

python scripts/counterfactuals_phase3.py \
  --table "$TABLE" \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --fig-dir "$FIG_DIR" \
  --tag "$TAG" \
  "${EXTRA[@]}"

echo "Finished: $(date)"
echo "Manifest: $OUTPUT_DIR/counterfactuals_manifest.json"
