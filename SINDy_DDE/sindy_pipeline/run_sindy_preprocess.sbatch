#!/bin/bash
#SBATCH --job-name=sindy-prep
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/prep_%j.log
#SBATCH --error=logs/prep_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ──────────────────────────────────────────────────────────────
# PRI2M Phase 4: SINDy-DDE Preprocessing
# 11-step pipeline per file: load h5, guards, transient trim,
# build state matrix, save pickle to preprocessed/.
#
# Pass dataset range via --export:
#   sbatch --export=DS_START=1,DS_END=60  run_sindy_preprocess.sbatch
#   sbatch --export=DS_START=61,DS_END=120 run_sindy_preprocess.sbatch
#
# Before first submission, create logs dir:
#   mkdir -p /nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline/logs
# ──────────────────────────────────────────────────────────────

set -euo pipefail

echo "=== PRI2M Phase 4: SINDy-DDE Preprocessing ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Memory:   ${SLURM_MEM_PER_NODE:-16000} MB"
echo "Started:  $(date)"
echo ""

# ── Setup ────────────────────────────────────────────────────
PROJECT_ROOT=/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline
cd "$PROJECT_ROOT"

# Load Python and activate virtual env
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

# Create output directories
mkdir -p preprocessed logs

# ── Verify required packages ─────────────────────────────────
python -c "import numpy, pandas, h5py, scipy; print(f'numpy {numpy.__version__}, h5py {h5py.__version__}')" || {
    echo ">>> Installing missing packages..."
    pip install numpy pandas h5py scipy
}

# ── Pin numpy threads to match allocated CPUs ────────────────
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ">>> OMP/MKL threads pinned to $SLURM_CPUS_PER_TASK"
echo ""

# ── Dataset range ────────────────────────────────────────────
DS_START="${DS_START:-1}"
DS_END="${DS_END:-60}"

# ── Run preprocessing ────────────────────────────────────────
echo ">>> Running preprocess on datasets ${DS_START}..${DS_END}..."
python sindy_preprocess.py \
    --dataset_start "${DS_START}" \
    --dataset_end "${DS_END}"

echo ""
echo "=== Phase 4 Complete ==="
echo "Finished: $(date)"
echo "Pickles:  $(ls preprocessed/ 2>/dev/null | wc -l) in preprocessed/"
echo "Log:      $PROJECT_ROOT/rejection_log.csv"
