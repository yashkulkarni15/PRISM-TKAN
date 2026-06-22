#!/bin/bash
#SBATCH --job-name=pri2m-p2.5-noise
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase2_5_noise_%j.log
#SBATCH --error=logs/phase2_5_noise_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Run after phase2_extract_signals.sbatch has completed for the selected data.
# Computes per-file noise/chatter metrics on Phase 2 extracted NPZ bundles.
# Use `env DATASETS='1,61,91' sbatch --export=ALL` for comma-containing selectors.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE}"
SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_ROOT/data/metadata/features_all_datasets.csv}"
NOISY_SUMMARY_CSV="${NOISY_SUMMARY_CSV:-$PROJECT_ROOT/data/metadata/features_noisy_datasets.csv}"
SIGNALS_DIR="${SIGNALS_DIR:-$PROJECT_ROOT/data/phase2_signals}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase2_5/noise_metrics}"
DATASETS="${DATASETS:-}"
LIMIT="${LIMIT:-}"
MAX_FILES_PER_DATASET="${MAX_FILES_PER_DATASET:-}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "=== PRI2M Phase 2.5 noise metrics ==="
echo "Summary:       $SUMMARY_CSV"
echo "Noisy summary: $NOISY_SUMMARY_CSV"
echo "Signals:       $SIGNALS_DIR"
echo "Output:        $OUTPUT_DIR"
echo "Datasets:      ${DATASETS:-all}"
echo "Started:       $(date)"

if [[ -z "$DATASETS" ]]; then
  echo "DATASETS environment variable is required, e.g. DATASETS='1,61,91' or DATASETS='1-60,61-120'." >&2
  exit 2
fi

ARGS=(
  --summary-csv "$SUMMARY_CSV"
  --noisy-summary-csv "$NOISY_SUMMARY_CSV"
  --signals-dir "$SIGNALS_DIR"
  --output-dir "$OUTPUT_DIR"
  --datasets "$DATASETS"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ -n "$MAX_FILES_PER_DATASET" ]]; then
  ARGS+=(--max-files-per-dataset "$MAX_FILES_PER_DATASET")
fi

python scripts/compute_noise_metrics.py "${ARGS[@]}"

echo "Finished: $(date)"
