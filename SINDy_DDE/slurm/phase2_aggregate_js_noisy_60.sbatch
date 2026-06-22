#!/bin/bash
#SBATCH --job-name=pri2m-p2-js-noisy
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/phase2_js_noisy_%j.log
#SBATCH --error=logs/phase2_js_noisy_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Aggregate stable-reference JS divergence for the full noisy set (61-120),
# using the SAME reference rule and parameters as the clean 1-60 run
# (low_b_ratio, quantile 0.10, b/blim cap 0.80) so the noisy curves line up
# against the clean JS story. Self-contained: hardcoded for the real HPC layout
# (PRI2M project root, NOT the SINDy_DDE subdir). Reads NPZs directly, so it is
# independent of the PSD-summary job and can run in parallel with it.

set -euo pipefail

PROJECT_ROOT=/nfs/turbo/coe-jaabell/PRI2M
SUMMARY_CSV=/nfs/turbo/coe-jaabell/PRI2M/Dataset/features_noisy_datasets.csv
SIGNALS_DIR=/nfs/turbo/coe-jaabell/PRI2M/phase2_signals
OUTPUT_DIR=/nfs/turbo/coe-jaabell/PRI2M/phase2_js_curves_noisy_60

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/mplconfig"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="$OUTPUT_DIR/mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "=== PRI2M Phase 2 aggregate JS divergence (noisy 61-120) ==="
echo "Summary:   $SUMMARY_CSV"
echo "Signals:   $SIGNALS_DIR"
echo "Output:    $OUTPUT_DIR"
echo "Datasets:  61-120"
echo "Reference: low_b_ratio q=0.10 max_b=0.80"
echo "Started:   $(date)"

python scripts/aggregate_js_divergence.py \
  --summary-csv "$SUMMARY_CSV" \
  --signals-dir "$SIGNALS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --datasets 61-120 \
  --signals xtpos,ytpos,Forcex,Forcey \
  --bootstrap 200 \
  --seed 20260513 \
  --reference-mode low_b_ratio \
  --reference-quantile 0.10 \
  --reference-max-b-ratio 0.80

echo "Finished: $(date)"
