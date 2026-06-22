#!/bin/bash
#SBATCH --job-name=pri2m-mlflow-clean
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mlflow_cleanup_%j.log
#SBATCH --error=logs/mlflow_cleanup_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch --export=RUN_DIR=/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/mlruns/... slurm/mlflow_checkpoint_cleanup.sbatch
#   sbatch --export=RUN_DIR=/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/mlruns/...,EXECUTE=1 slurm/mlflow_checkpoint_cleanup.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to an MLflow run/artifact directory}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

ARGS=("$RUN_DIR")
if [[ "${EXECUTE:-0}" == "1" ]]; then
  ARGS+=(--execute)
fi

python scripts/cleanup_mlflow_checkpoints.py "${ARGS[@]}"
