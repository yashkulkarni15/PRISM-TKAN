#!/bin/bash
#SBATCH --job-name=pri2m-minirocket
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/phase3_minirocket_%A_%a.log
#SBATCH --error=logs/phase3_minirocket_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Exp 3 (paper close-out, 2026-10-05): MiniROCKET comparator, arms A (4 ch) and
# B (4 + 4 delayed), in-fold val AUC + held-out 121-125 AUC, and the test-time
# delay variants on arm B (zero / tau x0.5 / tau x2). One fold per array task.
#
# ONE-TIME SETUP -- a dedicated CPU env, NOT prism (aeon's numba would downgrade
# prism's numpy 2.4.4 -> 2.3.5). Everything except numpy pinned to prism's versions:
#   module load python3.11-anaconda
#   python3 -m venv /home/$USER/envs/prism_aeon
#   source /home/$USER/envs/prism_aeon/bin/activate
#   pip install --upgrade pip
#   pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
#   pip install aeon==1.6.0 pandas==2.3.3 scikit-learn==1.8.0 scipy==1.17.1 pyyaml pytest
#   PYTHONPATH=src python -m pytest -q tests/test_minirocket_baseline.py   # aeon tests must RUN, not skip
#
# TIME FOLD 0 FIRST. Wall time is an estimate, not a measurement:
#   sbatch --array=0 slurm/phase3_minirocket.sbatch     # read wall_seconds in the manifest
#   sbatch --array=1-4 slurm/phase3_minirocket.sbatch   # then the rest
#
# PER-FILE SCORES + NOISE DOSE-RESPONSE (TKAN-vs-MiniROCKET noise comparison):
#   SCORES=1 sbatch slurm/phase3_minirocket.sbatch
# refits (same seed), writes manifests as minirocket_fold<F>_scores.json (the
# landed fold manifests are untouched), score CSVs to results/phase3/xai/
# noise_compare/ and models to results/phase3/minirocket/models/. Refuses to
# start without results/phase3/xai/noise_compare/mirror_noise_calibration.json.
#
# MEMORY ESTIMATE (peak, one arm at a time -- arms are fit sequentially):
#   train windows: ~26.5k supervised files x 8 = ~212k cases
#   MiniROCKET features 212k x 9 996 x 4 B (float32)      ~ 8.5 GB
#   selected raw windows 212k x 1024 x 4 ch x 4 B          ~ 3.5 GB
#   arm-B inputs 212k x 8 ch x 256 x 4 B (+ aeon's copy)   ~ 1.7-3.4 GB
#   Gram 9 996^2 x 8 B + eigh workspace/eigvecs            ~ 2.5 GB
#   standardisation is chunked (4 096 rows in float64, ~0.3 GB): there is NO
#   full standardised copy of the feature matrix, so the "x2" a naive
#   StandardScaler would need does not apply.
#   => ~17-19 GB peak; 48G is ~2.5x headroom. Eval streams 128 files at a time.
# TIME ESTIMATE (local M-series measurements, scaled): transform ~470 cases/s
#   per core -> ~1.5 M transforms per fold (train 2 arms + val/held-out x 5
#   variants) ~ 15-30 min at 16 jobs; Gram ~90 s per pass and eigh ~95 s at
#   p=9 996, 4 of each per fold ~ 15 min. The unknown is NPZ I/O over ~34k files
#   on turbo; expect 1-4 h total. 24 h is deliberately generous for fold 0.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
FOLD="${FOLD:-${SLURM_ARRAY_TASK_ID:-0}}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
FOLDS_JSON="${FOLDS_JSON:-$PROJECT_ROOT/results/phase3/phase3_folds.json}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
FEATURES="${FEATURES:-$PROJECT_ROOT/data/metadata/features_datasets_121_125.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/minirocket}"
N_KERNELS="${N_KERNELS:-10000}"
SEED="${SEED:-0}"
MAX_PER_FILE="${MAX_PER_FILE:-8}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"
SCORES="${SCORES:-0}"
NOISE_DIR="${NOISE_DIR:-$PROJECT_ROOT/results/phase3/xai/noise_compare}"
NOISE_CAL="${NOISE_CAL:-$NOISE_DIR/mirror_noise_calibration.json}"
NOISE_LEVELS="${NOISE_LEVELS:-0,1,2.5,5,7.5,10}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/results/phase3/minirocket/models}"
if [[ "$SCORES" == "1" ]]; then
  # A distinct default tag so the landed minirocket_fold<F>.json manifests are
  # never overwritten by the scoring rerun.
  TAG="${TAG:-fold${FOLD}_scores}"
else
  TAG="${TAG:-fold${FOLD}}"
fi

cd "$PROJECT_ROOT"
module load python3.11-anaconda
# A SEPARATE env, not prism: aeon pulls numba 0.63, which forces numpy 2.4.4 ->
# 2.3.5 (checked with pip --dry-run on GL, 2026-10-05). Downgrading prism would
# change the environment every validated TKAN run was produced under.
ENV_DIR="${ENV_DIR:-/home/$USER/envs/prism_aeon}"
source "$ENV_DIR/bin/activate"
mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# numba/BLAS threads follow the allocation instead of the node's core count.
export OMP_NUM_THREADS="$N_JOBS" MKL_NUM_THREADS="$N_JOBS" NUMBA_NUM_THREADS="$N_JOBS"

if ! python3 -c "import aeon" 2>/dev/null; then
  echo "ERROR: aeon is not importable from the MiniROCKET env." >&2
  echo "  aeon missing from $ENV_DIR. One-time setup: see the ONE-TIME SETUP block at the top of this file." >&2
  exit 3
fi

echo "=== PRI2M Exp 3: MiniROCKET comparator ==="
echo "Fold:      $FOLD   tag: $TAG"
echo "Kernels:   $N_KERNELS   seed: $SEED   max/file: $MAX_PER_FILE   jobs: $N_JOBS"
echo "aeon:      $(python3 -c 'import aeon;print(aeon.__version__)')"
echo "Started:   $(date)"

for f in "$TABLE" "$FOLDS_JSON" "$FEATURES" "$NPZ_ROOT"; do
  if [[ ! -e "$f" ]]; then echo "ERROR: missing input $f" >&2; exit 1; fi
done

# SCORES=1: per-file scores (val + held-out), the injected-noise dose-response
# on clean val files, and a model dump. Needs the mirror-noise calibration
# (scripts/calibrate_mirror_noise.py) to have landed first.
SCORE_ARGS=()
if [[ "$SCORES" == "1" ]]; then
  if [[ ! -f "$NOISE_CAL" ]]; then
    echo "ERROR: SCORES=1 needs the noise calibration $NOISE_CAL (run calibrate_mirror_noise.py first)" >&2
    exit 1
  fi
  mkdir -p "$NOISE_DIR" "$MODEL_DIR"
  SCORE_ARGS=(--save-scores --scores-dir "$NOISE_DIR"
              --noise-levels "$NOISE_LEVELS" --noise-calibration "$NOISE_CAL"
              --noise-files clean
              --save-model "$MODEL_DIR/minirocket_${TAG}.joblib")
  echo "Scores:    $NOISE_DIR   levels: $NOISE_LEVELS   model: $MODEL_DIR/minirocket_${TAG}.joblib"
fi

python3 scripts/minirocket_phase3.py \
  --fold "$FOLD" \
  --arm both \
  --table "$TABLE" \
  --folds-json "$FOLDS_JSON" \
  --npz-root "$NPZ_ROOT" \
  --features-csv "$FEATURES" \
  --max-train-per-file "$MAX_PER_FILE" \
  --n-kernels "$N_KERNELS" \
  --seed "$SEED" \
  --n-jobs "$N_JOBS" \
  --output-dir "$OUTPUT_DIR" \
  --tag "$TAG" \
  ${SCORE_ARGS[@]+"${SCORE_ARGS[@]}"}

DEST="$OUTPUT_DIR/minirocket_$TAG.json"
python3 - "$DEST" <<'PY'
import json, sys
p = sys.argv[1]
m = json.load(open(p))
assert m.get("complete") is True, f"{p} is not complete"
print(f"Manifest OK: {p}")
PY
if [[ "$SCORES" == "1" ]]; then
  for arm in A B; do
    for split in val heldout val_noise; do
      side="$NOISE_DIR/scores_minirocket_${arm}_fold${FOLD}_${split}.json"
      if [[ ! -f "$side" ]]; then echo "ERROR: missing $side" >&2; exit 1; fi
    done
  done
  echo "Scores OK: $NOISE_DIR/scores_minirocket_{A,B}_fold${FOLD}_*.csv"
fi
echo "Finished:  $(date)"
