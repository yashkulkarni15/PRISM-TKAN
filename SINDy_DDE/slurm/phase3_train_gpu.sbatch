#!/bin/bash
#SBATCH --job-name=pri2m-p3-gpu
#SBATCH --account=jaabell0
# MUST be spgpu: those are A40s (sm_86), which the prism torch (cu130) supports.
# The `gpu` partition is V100s (sm_70) -> instant `no kernel image` crash.
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
# spgpu MaxTime is 14 days; the old 6 h was self-imposed and too short. A full
# 20-epoch fold is ~8 h with torch.compile on (config default, ~22 min/epoch) or
# ~26 h eager (--no-compile, ~77 min/epoch). 36 h covers the eager fallback too;
# for faster scheduling pass --time=12:00:00 once you trust compile.
#SBATCH --time=36:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/phase3_train_gpu_%A_%a.log
#SBATCH --error=logs/phase3_train_gpu_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# GPU + per-fold job-array variant of slurm/phase3_train.sbatch.
#
# Each array task trains ONE fold (FOLD = $SLURM_ARRAY_TASK_ID) on a single GPU,
# so all five folds run in parallel and each only needs its own wall-clock budget
# instead of 5x. The recurrent TKAN trunk (a 1024-step KAN-GRU) is the cost.
# A40 benchmark at batch 32 / 1024 steps showed one epoch > 6 h. The config
# (configs/phase3_tkan.yaml) cuts that ~4x: pool 4 shortens the recurrence
# 1024->256 (linear ~4x); batch_size 256 helps little once pooled, because at
# 256 steps the model is compute-bound, not launch-bound (the "free big batch"
# only held at 1024 steps). A40 probe (853k windows): ~77 min/epoch eager, and
# torch.compile on the cell (config `compile: true`) fuses the launch-bound
# recurrence to ~22 min/epoch (~3.4x, identical math). See --time above.
#
# Submit (TKAN, all 5 folds in parallel):
#   sbatch --export=ALL slurm/phase3_train_gpu.sbatch
# Baseline MLP instead:
#   env MODEL=mlp sbatch --export=ALL slurm/phase3_train_gpu.sbatch
# A single fold (smoke first to confirm timing before all 5):
#   sbatch --array=0 --export=ALL slurm/phase3_train_gpu.sbatch
#
# Partition MUST be spgpu (A40 / sm_86). The `gpu` partition is V100 (sm_70),
# which the prism torch build has no kernels for -> instant crash. spgpu access
# under jaabell0 is confirmed.

set -euo pipefail

# NOTE: the live Great Lakes project root is /nfs/turbo/coe-jaabell/PRI2M
# (the .../PRI2M/SINDy_DDE subdir is a stale copy; do not use it).
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
CONFIG="${CONFIG:-configs/phase3_tkan.yaml}"
MODEL="${MODEL:-tkan}"
# One fold per array task; fall back to FOLD for a non-array submission.
FOLD="${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}"
# NPZ tree lives at $PROJECT_ROOT/phase2_signals (dataset_<id>/*.npz) -- the
# Phase 2 signal output -- NOT under results/phase3. Override NPZ_ROOT if moved.
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
FOLDS_JSON="${FOLDS_JSON:-$PROJECT_ROOT/results/phase3/phase3_folds.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/runs}"
MLFLOW_URI="${MLFLOW_URI:-file:$PROJECT_ROOT/results/phase3/mlruns}"
DEVICE="${DEVICE:-cuda}"
# DataLoader workers parallelize per-window normalization on the host CPUs while
# the GPU trains; keep <= cpus-per-task.
NUM_WORKERS="${NUM_WORKERS:-4}"
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

echo "=== PRI2M Phase 3 GPU training ==="
echo "Config:    $CONFIG"
echo "Model:     $MODEL"
echo "Fold:      $FOLD  (array task ${SLURM_ARRAY_TASK_ID:-n/a})"
echo "Device:    $DEVICE"
echo "Workers:   $NUM_WORKERS"
echo "NPZ root:  $NPZ_ROOT"
echo "Table:     $TABLE"
echo "Folds JSON:$FOLDS_JSON"
echo "Output:    $OUTPUT_DIR"
echo "MLflow:    $MLFLOW_URI"
echo "Smoke:     $SMOKE"
echo "GPU(s):    ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started:   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

SMOKE_FLAG=()
if [[ "$SMOKE" == "1" || "$SMOKE" == "true" || "$SMOKE" == "TRUE" ]]; then
  SMOKE_FLAG=(--smoke)
fi

echo "--- Training $MODEL fold $FOLD on $DEVICE ---"
python scripts/train_phase3.py \
  --config "$CONFIG" \
  --model "$MODEL" \
  --fold "$FOLD" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --folds-json "$FOLDS_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --mlflow-uri "$MLFLOW_URI" \
  --device "$DEVICE" \
  --num-workers "$NUM_WORKERS" \
  "${SMOKE_FLAG[@]}"

echo "Finished: $(date)"
