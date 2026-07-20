#!/bin/bash
#SBATCH --job-name=pri2m-p3-sindy-dde
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase3_sindy_dde_%j.log
#SBATCH --error=logs/phase3_sindy_dde_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# SINDy-DDE cross-check (eval-only, NO neural network, NO pysindy). Recovers the
# regenerative delay tau directly from raw chatter windows via a hand-rolled
# delay-embedded sparse regression (STLSQ tau-scan), as an independent white-box
# comparison against the kinematic tooth-pass delay round(fs/tpf) and the TKAN's
# delay-ablation result. CPU-only: small-batch finite differences + least squares.
#
# Parameterize with env overrides (defaults = fold 2, xtpos channel):
#   env FOLD=2 CHANNEL=xtpos N_WINDOWS=64 \
#       sbatch --export=ALL slurm/phase3_sindy_dde.sbatch

set -euo pipefail

# NOTE: the live Great Lakes project root is /nfs/turbo/coe-jaabell/PRI2M
# (the .../PRI2M/SINDy_DDE subdir is a stale copy; do not use it).
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"

NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
FOLDS="${FOLDS:-$PROJECT_ROOT/results/phase3/phase3_folds.json}"
FOLD="${FOLD:-2}"
CHANNEL="${CHANNEL:-xtpos}"
N_WINDOWS="${N_WINDOWS:-64}"
DERIV_ORDER="${DERIV_ORDER:-2}"       # 2 = acceleration (physical milling model)
THRESHOLD="${THRESHOLD:-0.05}"
POLY_ORDER="${POLY_ORDER:-3}"
REGEN_POLY_ORDER="${REGEN_POLY_ORDER:-1}"
GRID_LO="${GRID_LO:-0.4}"
GRID_HI="${GRID_HI:-2.5}"
GRID_POINTS="${GRID_POINTS:-44}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/sindy_dde/fold${FOLD}}"
FIGURE="${FIGURE:-$PROJECT_ROOT/results/figures/research_update_assets/sindy_dde_tau_recovery.png}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR" "$(dirname "$FIGURE")"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

echo "=== PRI2M Phase 3 SINDy-DDE cross-check ==="
echo "Fold:      $FOLD"
echo "Channel:   $CHANNEL"
echo "NPZ root:  $NPZ_ROOT"
echo "Table:     $TABLE"
echo "Output:    $OUTPUT_DIR"
echo "N windows: $N_WINDOWS"
echo "Started:   $(date)"

python scripts/sindy_dde_crosscheck.py \
  --table "$TABLE" \
  --npz-root "$NPZ_ROOT" \
  --folds "$FOLDS" \
  --fold "$FOLD" \
  --channel "$CHANNEL" \
  --n-windows "$N_WINDOWS" \
  --deriv-order "$DERIV_ORDER" \
  --threshold "$THRESHOLD" \
  --poly-order "$POLY_ORDER" \
  --regen-poly-order "$REGEN_POLY_ORDER" \
  --grid-lo-frac "$GRID_LO" \
  --grid-hi-frac "$GRID_HI" \
  --grid-points "$GRID_POINTS" \
  --output-dir "$OUTPUT_DIR" \
  --figure "$FIGURE"

echo "Finished: $(date)"
echo "Manifest: $OUTPUT_DIR/sindy_dde_manifest.json"
