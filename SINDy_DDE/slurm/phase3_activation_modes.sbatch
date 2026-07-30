#!/bin/bash
#SBATCH --job-name=pri2m-p3-actmodes
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase3_activation_modes_%j.log
#SBATCH --error=logs/phase3_activation_modes_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=FAIL

# U14-B: do the five folds converge on a shared set of learned activation shapes?
#
# CPU-only on purpose. This is forward passes for data-aware edge importance plus
# k-means over sampled curves -- no training, no backward pass. The GPU queue is
# the scarce resource and this does not need it.
#
# READ THE MARGIN, NOT THE SCORE. A randomly initialised network already scores
# ~0.63 on the agreement measure (synthetic fixtures), because every curve is
# n_coeff spline coefficients over one shared basis and cannot be far from every
# prototype. The run therefore spends most of its time on the untrained null:
# N_SEEDS=32 replicates, which is why the verdict can mean anything at all.
#
# It sweeps k because a single k is not a result: set k above the number of shapes
# actually present and k-means splits a true cluster, collapsing identity agreement
# toward chance while prototype_fit stays near 1.0.
#
#   sbatch --export=ALL slurm/phase3_activation_modes.sbatch
#   env N_WINDOWS=512 N_SEEDS=64 sbatch --export=ALL slurm/phase3_activation_modes.sbatch

set -euo pipefail

# NOTE: the live Great Lakes project root is /nfs/turbo/coe-jaabell/PRI2M
# (the .../PRI2M/SINDy_DDE subdir is a stale copy; do not use it).
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/results/phase3/runs}"

# The five published side-on checkpoints, same run ids as run_fuzz_all_folds.sh.
FOLD0="${FOLD0:-61c67bddbdbd49ba9727ff92f4814c86}"
FOLD1="${FOLD1:-c21e58dc0e624e1dbc602ec63a2dc23e}"
FOLD2="${FOLD2:-a654129d2cdc460093e21373f31ee5a5}"
FOLD3="${FOLD3:-3a40b71514cd41b98916330c000d700a}"
FOLD4="${FOLD4:-0937883918354ed899a61afb90aac14f}"

CKPTS=""
for RID in "$FOLD0" "$FOLD1" "$FOLD2" "$FOLD3" "$FOLD4"; do
  C="$RUNS_DIR/$RID/checkpoints/best_auc.pt"
  if [[ ! -f "$C" ]]; then
    echo "ERROR: checkpoint not found: $C" >&2
    exit 1
  fi
  CKPTS="${CKPTS:+$CKPTS,}$C"
done

NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
TAG="${TAG:-5fold}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/activation_modes/$TAG}"

N_WINDOWS="${N_WINDOWS:-256}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
N_POINTS="${N_POINTS:-64}"
K_SWEEP="${K_SWEEP:-2,3,4,5,6}"
N_SEEDS="${N_SEEDS:-32}"
THRESHOLD="${THRESHOLD:-0.8}"
MIN_SEEDS="${MIN_SEEDS:-20}"
GATE_QUANTILE="${GATE_QUANTILE:-0.9}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

echo "=== PRI2M Phase 3 activation-mode analysis (U14-B) ==="
echo "Output:      $OUTPUT_DIR"
echo "Windows:     $N_WINDOWS  (sample seed $SAMPLE_SEED)"
echo "k sweep:     $K_SWEEP"
echo "Null seeds:  $N_SEEDS   (verdict UNDECIDED below $MIN_SEEDS)"
echo "Started:     $(date)"

python scripts/activation_modes_phase3.py \
  --checkpoints "$CKPTS" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --output-dir "$OUTPUT_DIR" \
  --n-windows "$N_WINDOWS" \
  --sample-seed "$SAMPLE_SEED" \
  --n-points "$N_POINTS" \
  --k-sweep "$K_SWEEP" \
  --n-seeds "$N_SEEDS" \
  --threshold "$THRESHOLD" \
  --min-seeds "$MIN_SEEDS" \
  --gate-quantile "$GATE_QUANTILE" \
  --tag "$TAG"

# ASSERT the artifact, never merely announce it. Two wrappers in this project
# exited 0 while skipping their science and printed a path that was never written.
MANIFEST="$OUTPUT_DIR/activation_modes_manifest.json"
if [[ ! -s "$MANIFEST" ]]; then
  echo "ERROR: manifest missing or empty: $MANIFEST" >&2
  exit 1
fi
python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
if not m.get("provenance", {}).get("complete"):
    raise SystemExit(f"ERROR: manifest is a partial sweep (complete=false): {sys.argv[1]}")
print(f"OK: {len(m.get('by_k', {}))} k values recorded, complete=true")
PY

echo "Finished: $(date)"
echo "UNDECIDED is NOT a pass -- it means the null was under-sampled or a stratum"
echo "was dead in the data. Read the trained-vs-untrained margin, never the"
echo "trained score alone: an untrained network already scores ~0.63."
