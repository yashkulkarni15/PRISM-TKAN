#!/bin/bash
#SBATCH --job-name=pri2m-ab-verify
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/phase1_ab_%j.log
#SBATCH --error=logs/phase1_ab_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE}"
cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "=== PRI2M Phase 1 Altintas-Budak verifier ==="
echo "Started: $(date)"

python scripts/validate_dev_datasets.py
python scripts/verify_altintas_budak.py \
  --mode all \
  --dataset "data/metadata/features_all_datasets.csv" \
  --max-rows "${MAX_ROWS:-160}" \
  --grid-points "${GRID_POINTS:-301}" \
  --max-lobe-order "${MAX_LOBE_ORDER:-16}" \
  ${STRICT:+--strict}

echo "Finished: $(date)"
