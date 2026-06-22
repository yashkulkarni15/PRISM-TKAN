#!/bin/bash
#SBATCH --job-name=pri2m-p2-extract
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase2_extract_%A_%a.log
#SBATCH --error=logs/phase2_extract_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Submit examples:
#   Pilot clean/dev+anomaly: sbatch --array=1,4,11,12,29,34,35,39,44-48 slurm/phase2_extract_signals.sbatch
#   Pilot noisy mirror:      sbatch --array=91 --export=ALL,SUMMARY_CSV="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/metadata/features_noisy_datasets.csv" slurm/phase2_extract_signals.sbatch
#   Clean full run:    sbatch --array=1-60 slurm/phase2_extract_signals.sbatch
#   Noisy mirrors:     sbatch --array=61-120 --export=ALL,SUMMARY_CSV="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/metadata/features_noisy_datasets.csv" slurm/phase2_extract_signals.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE}"
DATA_VERSION="${DATA_VERSION:-10}"
DATA_ROOT="${DATA_ROOT:-/nfs/turbo/coe-jaabell/datasets/tonylschmitz/digital-machining-database/versions/${DATA_VERSION}}"
SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_ROOT/data/metadata/features_all_datasets.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/phase2_signals}"
DATASETS="${DATASETS:-${SLURM_ARRAY_TASK_ID:-1}}"
MAX_FILES_PER_DATASET="${MAX_FILES_PER_DATASET:-}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "=== PRI2M Phase 2 signal extraction ==="
echo "Datasets: $DATASETS"
echo "Summary:  $SUMMARY_CSV"
echo "Output:   $OUTPUT_DIR"
echo "Started:  $(date)"

ARGS=(
  --summary-csv "$SUMMARY_CSV"
  --data-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --datasets "$DATASETS"
  --sindy-source-dir "$PROJECT_ROOT/sindy_pipeline"
)
if [[ -n "$MAX_FILES_PER_DATASET" ]]; then
  ARGS+=(--max-files-per-dataset "$MAX_FILES_PER_DATASET")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${STRICT:-0}" == "1" ]]; then
  ARGS+=(--strict)
fi

python scripts/extract_phase2_signals.py "${ARGS[@]}"

echo "Finished: $(date)"
