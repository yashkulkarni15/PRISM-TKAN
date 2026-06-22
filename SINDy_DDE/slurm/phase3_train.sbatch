#!/bin/bash
#SBATCH --job-name=pri2m-p3-train
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase3_train_%j.log
#SBATCH --error=logs/phase3_train_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Run after Phase 2.5 (mode labels + calibrated noise scores) and the Phase 3
# dataset/folds builder have produced the per-file table and folds JSON. Trains
# a single Phase 3 model (TKAN/sigKAN/MLP) on one or more folds via
# scripts/train_phase3.py, writing run artifacts under results/phase3/runs and
# MLflow runs to a quota-safe `file:` store under results/phase3/mlruns.
#
# Parameterize with env overrides, e.g.:
#   env MODEL=tkan FOLD=0 sbatch --export=ALL slurm/phase3_train.sbatch
#   env MODEL=mlp FOLDS='0 1 2' sbatch --export=ALL slurm/phase3_train.sbatch
#   env SMOKE=1 sbatch --export=ALL slurm/phase3_train.sbatch   # fast CI-style run

set -euo pipefail

# NOTE: the live Great Lakes project root is /nfs/turbo/coe-jaabell/PRI2M
# (the .../PRI2M/SINDy_DDE subdir is a stale copy; do not use it).
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
CONFIG="${CONFIG:-configs/phase3_tkan.yaml}"
MODEL="${MODEL:-tkan}"
# FOLD trains a single fold; FOLDS (space-separated) trains several in one job.
FOLD="${FOLD:-0}"
FOLDS="${FOLDS:-$FOLD}"
# NPZ tree lives at $PROJECT_ROOT/phase2_signals (dataset_<id>/*.npz) -- the
# Phase 2 signal output -- NOT under results/phase3. Override NPZ_ROOT if moved.
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
FOLDS_JSON="${FOLDS_JSON:-$PROJECT_ROOT/results/phase3/phase3_folds.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/runs}"
MLFLOW_URI="${MLFLOW_URI:-file:$PROJECT_ROOT/results/phase3/mlruns}"
SMOKE="${SMOKE:-0}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# MLflow 3.x requires this opt-in to use a filesystem (`file:`) tracking store.
export MLFLOW_ALLOW_FILE_STORE=true

echo "=== PRI2M Phase 3 training ==="
echo "Config:    $CONFIG"
echo "Model:     $MODEL"
echo "Folds:     $FOLDS"
echo "NPZ root:  $NPZ_ROOT"
echo "Table:     $TABLE"
echo "Folds JSON:$FOLDS_JSON"
echo "Output:    $OUTPUT_DIR"
echo "MLflow:    $MLFLOW_URI"
echo "Smoke:     $SMOKE"
echo "Started:   $(date)"

SMOKE_FLAG=()
if [[ "$SMOKE" == "1" || "$SMOKE" == "true" || "$SMOKE" == "TRUE" ]]; then
  SMOKE_FLAG=(--smoke)
fi

for fold in $FOLDS; do
  echo "--- Training $MODEL fold $fold ---"
  python scripts/train_phase3.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --fold "$fold" \
    --npz-root "$NPZ_ROOT" \
    --table "$TABLE" \
    --folds-json "$FOLDS_JSON" \
    --output-dir "$OUTPUT_DIR" \
    --mlflow-uri "$MLFLOW_URI" \
    "${SMOKE_FLAG[@]}"
done

echo "Finished: $(date)"
