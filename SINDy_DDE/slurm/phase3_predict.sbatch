#!/bin/bash
#SBATCH --job-name=pri2m-p3-predict
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
# NOTE: Phase3SequenceDataset eagerly loads every selected file's full signal
# into RAM at construction, so memory scales with DATASET scope. 32G covers a
# single dataset; for DATASET=all raise --mem on the command line or shard by
# dataset and concatenate the per-dataset CSVs.
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/phase3_predict_%j.log
#SBATCH --error=logs/phase3_predict_%j.err
#SBATCH --mail-type=END,FAIL

# Team-facing chatter scoring with the PUBLISHED Phase 3 TKAN model.
#
# Unlike slurm/phase3_predict_windows.sbatch (which points at a raw MLflow run
# directory for the XAI/counterfactual chain), this uses the frozen model zoo:
# no run ids to look up, sha256-verified weights, and the default fold is the
# one the team is meant to cite.
#
# Submit with:
#   sbatch --export=ALL,DATASET=12,OUT=$HOME/preds_ds12.csv slurm/phase3_predict.sbatch
#
# CPU partition on purpose: inference is short and the GPU partitions have a
# torch/architecture trap (the prism build has no sm_70 kernels, so a V100 node
# crashes instantly). Nothing here needs a GPU.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
DATASET="${DATASET:-all}"
OUT="${OUT:-$PROJECT_ROOT/results/phase3/preds_${DATASET//,/_}.csv}"
FOLD="${FOLD:-}"                 # empty = the zoo's default (blessed) fold
ZOO="${ZOO:-}"                   # empty = $PRI2M_MODEL_ZOO, else the built-in default

cd "$PROJECT_ROOT"
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$(dirname "$OUT")"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

ARGS=(--dataset "$DATASET" --out "$OUT" --device cpu)
[[ -n "$FOLD" ]] && ARGS+=(--fold "$FOLD")
[[ -n "$ZOO" ]] && ARGS+=(--zoo "$ZOO")

python scripts/predict_phase3.py "${ARGS[@]}"

echo "Predictions: $OUT"
echo "Provenance:  $OUT.provenance.json"
