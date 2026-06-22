#!/bin/bash
#SBATCH --job-name=sindy-fit
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fit_%j.log
#SBATCH --error=logs/fit_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ──────────────────────────────────────────────────────────────
# PRI2M Phase 4: SINDy-DDE Fits
# Per-file fit using preprocessed pickles. Picks the sparsity
# threshold from the longest plateau in the lambda sweep.
#
# Pass dataset range via --export. Suggested chunks of 20:
#   sbatch --export=DS_START=1,DS_END=20    run_sindy_fit.sbatch
#   sbatch --export=DS_START=21,DS_END=40   run_sindy_fit.sbatch
#   sbatch --export=DS_START=41,DS_END=60   run_sindy_fit.sbatch
#   sbatch --export=DS_START=61,DS_END=80   run_sindy_fit.sbatch
#   sbatch --export=DS_START=81,DS_END=100  run_sindy_fit.sbatch
#   sbatch --export=DS_START=101,DS_END=120 run_sindy_fit.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail

echo "=== PRI2M Phase 4: SINDy-DDE Fits ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Memory:   ${SLURM_MEM_PER_NODE:-16000} MB"
echo "Started:  $(date)"
echo ""

PROJECT_ROOT=/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline
cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p sindy_fits logs

# pysindy is required for SINDy fits. Install on first run if missing.
python -c "import pysindy; print(f'pysindy {pysindy.__version__}')" || {
    echo ">>> Installing pysindy..."
    pip install pysindy
}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

DS_START="${DS_START:-1}"
DS_END="${DS_END:-20}"

echo ">>> Fitting SINDy on datasets ${DS_START}..${DS_END}..."
python sindy_fit.py \
    --dataset_start "${DS_START}" \
    --dataset_end "${DS_END}"

echo ""
echo "=== Phase 4 fit complete ==="
echo "Finished: $(date)"
echo "Per-dataset CSVs: $(ls sindy_fits/ 2>/dev/null | wc -l) in sindy_fits/"
