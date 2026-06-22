#!/bin/bash
#SBATCH --job-name=pri2m-p2.5-calib
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/phase2_5_calib_%j.log
#SBATCH --error=logs/phase2_5_calib_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Run after phase2_5_compute_noise_metrics.sbatch has produced noise_metrics.csv.
# Calibrates the clean / 1% noise / 5% noise mirror ladder, then assigns mode
# labels. Mode assignment depends on the calibration result, so the two steps
# run sequentially in this job.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE}"
NOISE_METRICS_CSV="${NOISE_METRICS_CSV:-$PROJECT_ROOT/results/phase2_5/noise_metrics/noise_metrics.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_ROOT/data/metadata/features_all_datasets.csv}"
NOISY_SUMMARY_CSV="${NOISY_SUMMARY_CSV:-$PROJECT_ROOT/data/metadata/features_noisy_datasets.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase2_5/calibration}"
CLEAN_ID="${CLEAN_ID:-1}"
NOISE1_ID="${NOISE1_ID:-61}"
NOISE5_ID="${NOISE5_ID:-91}"
POOLED="${POOLED:-0}"
NOISE1_OFFSET="${NOISE1_OFFSET:-60}"
NOISE5_OFFSET="${NOISE5_OFFSET:-90}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "=== PRI2M Phase 2.5 mirror calibration and mode assignment ==="
echo "Noise metrics: $NOISE_METRICS_CSV"
echo "Summary:       $SUMMARY_CSV"
echo "Noisy summary: $NOISY_SUMMARY_CSV"
echo "Output:        $OUTPUT_DIR"
if [[ "$POOLED" == "1" || "$POOLED" == "true" || "$POOLED" == "TRUE" ]]; then
  echo "Calibration:   pooled full mirror ladders (offsets $NOISE1_OFFSET, $NOISE5_OFFSET)"
else
  echo "Calibration:   single ladder clean=$CLEAN_ID noise1=$NOISE1_ID noise5=$NOISE5_ID"
fi
echo "Started:       $(date)"

CALIB_ARGS=(
  --noise-metrics-csv "$NOISE_METRICS_CSV"
  --summary-csv "$SUMMARY_CSV"
  --noisy-summary-csv "$NOISY_SUMMARY_CSV"
  --output-dir "$OUTPUT_DIR"
)
if [[ "$POOLED" == "1" || "$POOLED" == "true" || "$POOLED" == "TRUE" ]]; then
  CALIB_ARGS+=(
    --pooled
    --noise1-offset "$NOISE1_OFFSET"
    --noise5-offset "$NOISE5_OFFSET"
  )
else
  CALIB_ARGS+=(
    --clean-id "$CLEAN_ID"
    --noise1-id "$NOISE1_ID"
    --noise5-id "$NOISE5_ID"
  )
fi

python scripts/calibrate_noise_mirror_pairs.py "${CALIB_ARGS[@]}"

python scripts/assign_signal_modes.py \
  --noise-metrics-csv "$NOISE_METRICS_CSV" \
  --summary-csv "$SUMMARY_CSV" \
  --noisy-summary-csv "$NOISY_SUMMARY_CSV" \
  --calibration-json "$OUTPUT_DIR/promoted_metrics.json" \
  --output-dir "$OUTPUT_DIR"

echo "Finished: $(date)"
