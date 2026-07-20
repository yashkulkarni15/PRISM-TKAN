#!/bin/bash
#SBATCH --job-name=pri2m-p3-fuzz
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/phase3_fuzz_%j.log
#SBATCH --error=logs/phase3_fuzz_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Graded input/model degradation of one trained Phase 3 TKAN checkpoint.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/results/phase3/runs}"
RUN_ID="${RUN_ID:-a654129d2cdc460093e21373f31ee5a5}"
CHECKPOINT="${CHECKPOINT:-$RUNS_DIR/$RUN_ID/checkpoints/best_auc.pt}"
TAG="${TAG:-fold2}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
PREPROCESSING="${PREPROCESSING:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/fuzz/$TAG}"
N_WINDOWS="${N_WINDOWS:-512}"
SEEDS="${SEEDS:-3}"
FUZZERS="${FUZZERS:-all}"
VAL_AUC="${VAL_AUC:-}"

cd "$PROJECT_ROOT"
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

[[ -f "$CHECKPOINT" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 1; }

VAL_FLAG=()
if [[ -n "$VAL_AUC" ]]; then VAL_FLAG=(--val-auc "$VAL_AUC"); fi
PRE_FLAG=()
if [[ -n "$PREPROCESSING" ]]; then PRE_FLAG=(--preprocessing "$PREPROCESSING"); fi

echo "=== PRI2M Phase 3 fuzziness suite ==="
echo "Checkpoint: $CHECKPOINT"
echo "Tag:        $TAG"
echo "Output:     $OUTPUT_DIR"
echo "Fuzzers:    $FUZZERS"
echo "Started:    $(date)"

python scripts/fuzz_phase3.py \
  --checkpoint "$CHECKPOINT" --npz-root "$NPZ_ROOT" --table "$TABLE" \
  --output-dir "$OUTPUT_DIR" --n-windows "$N_WINDOWS" --seeds "$SEEDS" \
  --fuzzers "$FUZZERS" --run-id "$RUN_ID" --tag "$TAG" \
  "${VAL_FLAG[@]}" "${PRE_FLAG[@]}"

echo "Finished: $(date)"
echo "Manifest: $OUTPUT_DIR/fuzz_manifest.json"
