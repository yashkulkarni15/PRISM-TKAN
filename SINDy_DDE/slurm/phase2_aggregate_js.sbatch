#!/bin/bash
#SBATCH --job-name=pri2m-p2-js
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/phase2_js_%j.log
#SBATCH --error=logs/phase2_js_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Aggregates stable-vs-unstable PSD Jensen-Shannon divergence by b/blim bin,
# faceted by teeth and cut_dir. Use `env DATASETS='1,4,...' sbatch --export=ALL`
# for comma-containing selectors.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE}"
SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_ROOT/data/metadata/features_all_datasets.csv}"
SIGNALS_DIR="${SIGNALS_DIR:-$PROJECT_ROOT/data/phase2_signals}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase2/js_curves}"
DATASETS="${DATASETS:-}"
SIGNALS="${SIGNALS:-xtpos,ytpos,Forcex,Forcey}"
BOOTSTRAP="${BOOTSTRAP:-200}"
SEED="${SEED:-20260513}"
REFERENCE_MODE="${REFERENCE_MODE:-low_b_ratio}"
REFERENCE_QUANTILE="${REFERENCE_QUANTILE:-0.10}"
REFERENCE_MAX_B_RATIO="${REFERENCE_MAX_B_RATIO:-0.80}"

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

echo "=== PRI2M Phase 2 aggregate JS divergence ==="
echo "Summary:   $SUMMARY_CSV"
echo "Signals:   $SIGNALS_DIR"
echo "Output:    $OUTPUT_DIR"
echo "Datasets:  ${DATASETS:-all}"
echo "Bootstrap: $BOOTSTRAP"
echo "Reference: $REFERENCE_MODE q=$REFERENCE_QUANTILE max_b=$REFERENCE_MAX_B_RATIO"
echo "Started:   $(date)"

ARGS=(
  --summary-csv "$SUMMARY_CSV"
  --signals-dir "$SIGNALS_DIR"
  --output-dir "$OUTPUT_DIR"
  --signals "$SIGNALS"
  --bootstrap "$BOOTSTRAP"
  --seed "$SEED"
  --reference-mode "$REFERENCE_MODE"
  --reference-quantile "$REFERENCE_QUANTILE"
  --reference-max-b-ratio "$REFERENCE_MAX_B_RATIO"
)
if [[ -n "$DATASETS" ]]; then
  ARGS+=(--datasets "$DATASETS")
fi
if [[ -n "${FS:-}" ]]; then
  ARGS+=(--fs "$FS")
fi

python scripts/aggregate_js_divergence.py "${ARGS[@]}"

echo "Finished: $(date)"
