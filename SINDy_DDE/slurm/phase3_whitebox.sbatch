#!/bin/bash
#SBATCH --job-name=pri2m-p3-whitebox
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase3_whitebox_%j.log
#SBATCH --error=logs/phase3_whitebox_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# White-box XAI analysis of a TRAINED Phase 3 TKAN checkpoint (eval-only, no
# retraining). Runs scripts/whitebox_phase3.py against one run's best_auc.pt and
# the Phase 2 NPZ signal tree, producing the full white-box artifact set + a
# whitebox_manifest.json under results/phase3/xai/<TAG>.
#
# CPU-only: the analysis is small-batch inference + attribution; no GPU needed.
#
# Parameterize with env overrides, e.g. (default = fold-2, the strongest run):
#   env RUN_ID=a654129d2cdc460093e21373f31ee5a5 TAG=fold2 \
#       sbatch --export=ALL slurm/phase3_whitebox.sbatch
#   # explicit checkpoint instead of RUN_ID:
#   env CHECKPOINT=$PWD/results/phase3/runs/<run>/checkpoints/best_auc.pt TAG=fold0 \
#       sbatch --export=ALL slurm/phase3_whitebox.sbatch
#   # larger temporal-saliency (TSR) sample -- the default 4 windows is too few:
#   env RUN_ID=<run> TAG=fold2_tsr256 TSR_WINDOWS=256 \
#       sbatch --export=ALL slurm/phase3_whitebox.sbatch
#
# To run every fold, submit once per RUN_ID/TAG (the 5 sweep runs are listed in
# docs/phase3_tkan/PROGRESS.md). Running only the strongest fold is the headline;
# running all five is the robustness check.

set -euo pipefail

# NOTE: the live Great Lakes project root is /nfs/turbo/coe-jaabell/PRI2M
# (the .../PRI2M/SINDy_DDE subdir is a stale copy; do not use it).
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"

# Identify the run: either RUN_ID (resolved under $RUNS_DIR) or an explicit CHECKPOINT.
RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/results/phase3/runs}"
RUN_ID="${RUN_ID:-a654129d2cdc460093e21373f31ee5a5}"   # default: fold 2 (AUC 0.975)
CHECKPOINT="${CHECKPOINT:-$RUNS_DIR/$RUN_ID/checkpoints/best_auc.pt}"
TAG="${TAG:-fold2}"

# Inputs (same NPZ tree the trainer used; preprocessing.json is auto-discovered
# next to the checkpoint's run dir by the runner, override only if relocated).
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
PREPROCESSING="${PREPROCESSING:-}"   # blank -> runner auto-discovers
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/$TAG}"
N_WINDOWS="${N_WINDOWS:-256}"        # sample of windows for dataset-level analyses
TOP_K="${TOP_K:-25}"                 # edges to symbolify
# Temporal-saliency-rescaling sample size. Defaults match the runner's own
# defaults, so leaving these unset reproduces previous runs exactly. TSR cost is
# O(TSR_WINDOWS * TSR_STEPS) forward passes -- 4 windows is far too few to trust,
# so raise TSR_WINDOWS (e.g. 256) for a publishable result.
TSR_WINDOWS="${TSR_WINDOWS:-4}"      # windows attributed with TSR
TSR_STEPS="${TSR_STEPS:-8}"          # time-step chunks per TSR window

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1   # stream progress to the log (else it buffers until exit)

echo "=== PRI2M Phase 3 white-box XAI ==="
echo "Checkpoint: $CHECKPOINT"
echo "Tag:        $TAG"
echo "NPZ root:   $NPZ_ROOT"
echo "Table:      $TABLE"
echo "Output:     $OUTPUT_DIR"
echo "N windows:  $N_WINDOWS"
echo "TSR windows:$TSR_WINDOWS"
echo "TSR steps:  $TSR_STEPS"
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
# Optional known validation AUC for this fold (e.g. VAL_AUC=0.975 for fold 2),
# recorded in the report header. Blank -> rendered as "N/A (not run)".
VAL_AUC="${VAL_AUC:-}"
VAL_FLAG=()
if [[ -n "$VAL_AUC" ]]; then
  VAL_FLAG=(--val-auc "$VAL_AUC")
fi

python scripts/whitebox_phase3.py \
  --checkpoint "$CHECKPOINT" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --output-dir "$OUTPUT_DIR" \
  --n-windows "$N_WINDOWS" \
  --top-k "$TOP_K" \
  --tsr-windows "$TSR_WINDOWS" \
  --tsr-steps "$TSR_STEPS" \
  --run-id "$RUN_ID" \
  --tag "$TAG" \
  "${VAL_FLAG[@]}" \
  "${PRE_FLAG[@]}"

echo "Finished: $(date)"
echo "Render the report locally after rsync:"
echo "  python scripts/render_whitebox_report.py --manifest $OUTPUT_DIR/whitebox_manifest.json"
