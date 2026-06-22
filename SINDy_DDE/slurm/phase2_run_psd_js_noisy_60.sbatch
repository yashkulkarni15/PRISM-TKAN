#!/bin/bash
#SBATCH --job-name=pri2m-p2-psdjs-noisy
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/phase2_psdjs_noisy_%j.log
#SBATCH --error=logs/phase2_psdjs_noisy_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Full noisy (datasets 61-120) PSD summary, mirroring the clean 1-60 run so the
# noise-robustness story is directly comparable. Self-contained: all paths are
# hardcoded for the real HPC layout (PRI2M project root, NOT the SINDy_DDE
# subdir), so no env-var passthrough / long --export lines are needed.
# Requires the noisy NPZs in phase2_signals/dataset_61..120 (already present;
# the full noise_metrics.csv was built from them).

set -euo pipefail

PROJECT_ROOT=/nfs/turbo/coe-jaabell/PRI2M
SUMMARY_CSV=/nfs/turbo/coe-jaabell/PRI2M/Dataset/features_noisy_datasets.csv
SIGNALS_DIR=/nfs/turbo/coe-jaabell/PRI2M/phase2_signals
OUTPUT_DIR=/nfs/turbo/coe-jaabell/PRI2M/phase2_psd_js_noisy_60

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "=== PRI2M Phase 2 PSD / Jensen-Shannon (noisy 61-120) ==="
echo "Summary:  $SUMMARY_CSV"
echo "Signals:  $SIGNALS_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Datasets: 61-120"
echo "Started:  $(date)"

python scripts/run_psd_js.py \
  --summary-csv "$SUMMARY_CSV" \
  --signals-dir "$SIGNALS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --datasets 61-120 \
  --inspect-datasets 91

echo "Finished: $(date)"
