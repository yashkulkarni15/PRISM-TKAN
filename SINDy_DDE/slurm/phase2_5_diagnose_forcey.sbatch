#!/bin/bash
#SBATCH --job-name=p2.5-diag-forcey
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diag_forcey_%j.log
#SBATCH --error=logs/diag_forcey_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Per-ladder monotone-fraction diagnostic for Forcey-channel metrics that
# narrowly missed promotion in the pooled mirror calibration.
# Hardcoded for the current HPC layout (PRI2M project root, not the
# SINDy_DDE subdir) so no env-var passthrough is required at submission time.

set -euo pipefail

PROJECT_ROOT=/nfs/turbo/coe-jaabell/PRI2M
NOISE_METRICS_CSV=/nfs/turbo/coe-jaabell/PRI2M/phase2_5_noise_metrics_full/noise_metrics.csv
SUMMARY_CSV=/nfs/turbo/coe-jaabell/PRI2M/Dataset/features_all_datasets.csv
NOISY_SUMMARY_CSV=/nfs/turbo/coe-jaabell/PRI2M/Dataset/features_noisy_datasets.csv
OUTPUT_DIR=/nfs/turbo/coe-jaabell/PRI2M/phase2_5_forcey_diagnostic

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "=== PRI2M Phase 2.5 Forcey monotone-fraction diagnostic ==="
echo "Noise metrics: $NOISE_METRICS_CSV"
echo "Summary:       $SUMMARY_CSV"
echo "Noisy summary: $NOISY_SUMMARY_CSV"
echo "Output:        $OUTPUT_DIR"
echo "Started:       $(date)"

python scripts/diagnose_forcey_near_misses.py \
  --noise-metrics-csv "$NOISE_METRICS_CSV" \
  --summary-csv "$SUMMARY_CSV" \
  --noisy-summary-csv "$NOISY_SUMMARY_CSV" \
  --metrics-substring Forcey \
  --output-dir "$OUTPUT_DIR"

echo "Finished: $(date)"
