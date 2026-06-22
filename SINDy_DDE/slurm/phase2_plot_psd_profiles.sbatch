#!/bin/bash
#SBATCH --job-name=pri2m-p2-plots
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/phase2_profiles_%j.log
#SBATCH --error=logs/phase2_profiles_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Produces per-dataset stable-vs-unstable PSD overlay plots with TPF and fn/tpf
# markers. Use `env DATASETS='1-60' sbatch --export=ALL ...` for selectors.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE}"
SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_ROOT/data/metadata/features_all_datasets.csv}"
SIGNALS_DIR="${SIGNALS_DIR:-$PROJECT_ROOT/data/phase2_signals}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase2/psd_profiles_clean_60}"
DATASETS="${DATASETS:-1-60}"
SIGNALS="${SIGNALS:-xtpos,Forcex}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR" "$OUTPUT_DIR/mplconfig"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="$OUTPUT_DIR/mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "=== PRI2M Phase 2 per-dataset PSD profiles ==="
echo "Summary:  $SUMMARY_CSV"
echo "Signals:  $SIGNALS_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Datasets: $DATASETS"
echo "Signals:  $SIGNALS"
echo "Started:  $(date)"

ARGS=(
  --summary-csv "$SUMMARY_CSV"
  --signals-dir "$SIGNALS_DIR"
  --output-dir "$OUTPUT_DIR"
  --datasets "$DATASETS"
  --signals "$SIGNALS"
)
if [[ -n "${FS:-}" ]]; then
  ARGS+=(--fs "$FS")
fi

python scripts/plot_psd_profiles.py "${ARGS[@]}"

echo "Finished: $(date)"
