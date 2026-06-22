#!/bin/bash
#SBATCH --job-name=pri2m-p2-psdjs
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/phase2_psdjs_%j.log
#SBATCH --error=logs/phase2_psdjs_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Run after phase2_extract_signals.sbatch has completed for the selected data.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE}"
SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_ROOT/data/metadata/features_all_datasets.csv}"
SIGNALS_DIR="${SIGNALS_DIR:-$PROJECT_ROOT/data/phase2_signals}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase2/psd_js}"
INSPECT_DATASETS="${INSPECT_DATASETS:-11,12,34,35,44-48}"
LIMIT="${LIMIT:-}"
DATASETS="${DATASETS:-}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "=== PRI2M Phase 2 PSD / Jensen-Shannon ==="
echo "Summary:  $SUMMARY_CSV"
echo "Signals:  $SIGNALS_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Datasets: ${DATASETS:-all}"
echo "Started:  $(date)"

ARGS=(
  --summary-csv "$SUMMARY_CSV"
  --signals-dir "$SIGNALS_DIR"
  --output-dir "$OUTPUT_DIR"
  --inspect-datasets "$INSPECT_DATASETS"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ -n "$DATASETS" ]]; then
  ARGS+=(--datasets "$DATASETS")
fi
if [[ -n "${REFERENCE_NPZ:-}" ]]; then
  ARGS+=(--reference-npz "$REFERENCE_NPZ")
fi
if [[ -n "${FS:-}" ]]; then
  ARGS+=(--fs "$FS")
fi

python scripts/run_psd_js.py "${ARGS[@]}"

echo "Finished: $(date)"
