#!/bin/bash
#SBATCH --job-name=pri2m-p3-preds
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# NOTE: Phase3SequenceDataset eagerly loads every file's full signal into RAM at
# construction, and this runner builds it over the WHOLE table (no window
# sampling), so memory scales with DATASET scope. 32G covers a single dataset;
# for DATASET=all (the entire signal tree) raise --mem on the command line or
# shard by dataset and concatenate the per-dataset CSVs.
#SBATCH --time=01:00:00
#SBATCH --output=logs/phase3_predict_windows_%j.log
#SBATCH --error=logs/phase3_predict_windows_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Per-file real-network chatter probabilities for the counterfactual runner.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/results/phase3/runs}"
RUN_ID="${RUN_ID:-a654129d2cdc460093e21373f31ee5a5}"
CHECKPOINT="${CHECKPOINT:-$RUNS_DIR/$RUN_ID/checkpoints/best_auc.pt}"
TAG="${TAG:-fold2}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
OUT="${OUT:-$PROJECT_ROOT/results/phase3/preds_$TAG.csv}"
DATASET="${DATASET:-all}"

cd "$PROJECT_ROOT"
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$(dirname "$OUT")"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

[[ -f "$CHECKPOINT" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 1; }

python scripts/predict_windows_phase3.py \
  --checkpoint "$CHECKPOINT" --table "$TABLE" --npz-root "$NPZ_ROOT" \
  --out "$OUT" --dataset "$DATASET"

echo "Predictions: $OUT"
