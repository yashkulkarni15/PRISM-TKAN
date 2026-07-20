#!/bin/bash
#SBATCH --job-name=pri2m-p3-concept-probes
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase3_concept_probes_%j.log
#SBATCH --error=logs/phase3_concept_probes_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# TCAV-style concept probes on a TRAINED Phase 3 TKAN checkpoint (eval-only, no
# retraining). Runs scripts/concept_probes_phase3.py against one run's best_auc.pt
# and the Phase 2 NPZ signal tree: fits a linear probe on the 64-d encoder hidden
# state for each named physics concept (regeneration phase, chatter/tooth-pass
# separation, proximity to blim), writing concept_probes_manifest.json + a probe
# score figure under results/phase3/xai/concept_probes/<TAG> and a copy under
# results/figures/research_update_assets/.
#
# CPU-only: small-batch inference + sklearn ridge/logistic probes; no GPU needed.
#
# Parameterize with env overrides, e.g. (default = fold-2, the strongest run):
#   env RUN_ID=a654129d2cdc460093e21373f31ee5a5 TAG=fold2 \
#       sbatch --export=ALL slurm/phase3_concept_probes.sbatch
#   # explicit checkpoint instead of RUN_ID:
#   env CHECKPOINT=$PWD/results/phase3/runs/<run>/checkpoints/best_auc.pt TAG=fold0 \
#       sbatch --export=ALL slurm/phase3_concept_probes.sbatch

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
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/concept_probes/$TAG}"
ASSETS_DIR="${ASSETS_DIR:-$PROJECT_ROOT/results/figures/research_update_assets}"
N_WINDOWS="${N_WINDOWS:-256}"        # sample of windows for the probe fit
POOL="${POOL:-4}"                    # temporal pool factor (must match training)
TEST_SIZE="${TEST_SIZE:-0.3}"        # probe train/test split fraction
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"    # ridge regularization strength

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR" "$ASSETS_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1   # stream progress to the log (else it buffers until exit)

echo "=== PRI2M Phase 3 concept probes (TCAV-style) ==="
echo "Checkpoint: $CHECKPOINT"
echo "Tag:        $TAG"
echo "NPZ root:   $NPZ_ROOT"
echo "Table:      $TABLE"
echo "Output:     $OUTPUT_DIR"
echo "Assets:     $ASSETS_DIR"
echo "N windows:  $N_WINDOWS"
echo "Pool:       $POOL"
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
# Optional known validation AUC for this fold (recorded in the manifest header).
VAL_AUC="${VAL_AUC:-}"
VAL_FLAG=()
if [[ -n "$VAL_AUC" ]]; then
  VAL_FLAG=(--val-auc "$VAL_AUC")
fi

python scripts/concept_probes_phase3.py \
  --checkpoint "$CHECKPOINT" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --output-dir "$OUTPUT_DIR" \
  --assets-dir "$ASSETS_DIR" \
  --n-windows "$N_WINDOWS" \
  --pool "$POOL" \
  --test-size "$TEST_SIZE" \
  --ridge-alpha "$RIDGE_ALPHA" \
  --run-id "$RUN_ID" \
  --tag "$TAG" \
  "${VAL_FLAG[@]}" \
  "${PRE_FLAG[@]}"

echo "Finished: $(date)"
echo "Manifest: $OUTPUT_DIR/concept_probes_manifest.json"
