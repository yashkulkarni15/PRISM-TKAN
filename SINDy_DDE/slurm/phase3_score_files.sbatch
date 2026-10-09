#!/bin/bash
#SBATCH --job-name=pri2m-score-files
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/phase3_score_files_%A_%a.log
#SBATCH --error=logs/phase3_score_files_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Noise comparison (MiniROCKET vs TKAN), TKAN half: per-file scores of the
# canonical fold checkpoints. One fold per array task.
#
#   ARM=side_on  (default) three passes, in order:
#       1. val real      -- all val files, all windows
#       2. heldout real  -- 121-125, all windows
#       3. val_noise     -- clean val files, 8 windows/file, levels 0,1,2.5,5,7.5,10
#                           (needs the mirror-noise calibration; run
#                           slurm/phase3_calibrate_mirror_noise.sbatch first)
#   ARM=side_off  heldout real only.
#
# Submit (from the GL project root, after rsync):
#   sbatch slurm/phase3_calibrate_mirror_noise.sbatch
#   sbatch --dependency=afterok:<cal_jobid> slurm/phase3_score_files.sbatch
#   sbatch --export=ALL,ARM=side_off slurm/phase3_score_files.sbatch
#
# Outputs: $OUT_DIR/scores_<model>_fold<F>_<split>.csv (+ .json sidecar), with
# model = tkan (side_on) or tkan_side_off.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
ARM="${ARM:-side_on}"
FOLD="${FOLD:-${SLURM_ARRAY_TASK_ID:-0}}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
FOLDS_JSON="${FOLDS_JSON:-$PROJECT_ROOT/results/phase3/phase3_folds.json}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
FEATURES="${FEATURES:-$PROJECT_ROOT/data/metadata/features_datasets_121_125.csv}"
CANONICAL="${CANONICAL:-$PROJECT_ROOT/results/phase3/RUNS_CANONICAL.json}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/results/phase3/xai/noise_compare}"
CALIBRATION="${CALIBRATION:-$OUT_DIR/mirror_noise_calibration.json}"
NOISE_LEVELS="${NOISE_LEVELS:-0,1,2.5,5,7.5,10}"
NOISE_MAX_WINDOWS="${NOISE_MAX_WINDOWS:-8}"
NOISE_SEED="${NOISE_SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PY="${PY:-python}"

case "$ARM" in
  side_on)  MODEL=tkan ;;
  side_off) MODEL=tkan_side_off ;;
  *) echo "ERROR: ARM must be side_on or side_off, got '$ARM'" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
if [[ -z "${SKIP_ENV:-}" ]]; then
  module load python3.11-anaconda
  source /home/$USER/envs/prism/bin/activate
fi
mkdir -p logs "$OUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
NT="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT" OPENBLAS_NUM_THREADS="$NT"

need=("$CANONICAL" "$NPZ_ROOT" "$FEATURES")
[[ "$ARM" == side_on ]] && need+=("$TABLE" "$FOLDS_JSON")
for f in "${need[@]}"; do
  if [[ ! -e "$f" ]]; then echo "ERROR: missing input $f" >&2; exit 1; fi
done
if [[ "$ARM" == side_on && ! -f "$CALIBRATION" ]]; then
  echo "ERROR: noise calibration not found: $CALIBRATION" >&2
  echo "  Run slurm/phase3_calibrate_mirror_noise.sbatch first (submit this job with" >&2
  echo "  --dependency=afterok:<calibration job id>)." >&2
  exit 1
fi

RUN_PATH=$(python3 - "$CANONICAL" "$ARM" "$FOLD" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
print(reg["arms"][sys.argv[2]]["folds"][str(int(sys.argv[3]))]["path"])
PY
)
CKPT="$PROJECT_ROOT/$RUN_PATH/checkpoints/best_auc.pt"
if [[ ! -f "$CKPT" ]]; then echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; fi

echo "=== PRI2M noise comparison: TKAN per-file scores ==="
echo "Arm:     $ARM ($MODEL)   fold: $FOLD"
echo "Ckpt:    $CKPT"
echo "Out:     $OUT_DIR"
echo "Started: $(date)"

COMMON=(--checkpoint "$CKPT" --npz-root "$NPZ_ROOT" --device cpu
        --batch-size "$BATCH_SIZE" --model-name "$MODEL" --fold-label "$FOLD")
OUTS=()

if [[ "$ARM" == side_on ]]; then
  out="$OUT_DIR/scores_${MODEL}_fold${FOLD}_val.csv"
  echo "--- pass 1/3: val real ($(date +%H:%M:%S)) ---"
  "$PY" scripts/score_files_phase3.py "${COMMON[@]}" --split val \
    --table "$TABLE" --folds-json "$FOLDS_JSON" --output-csv "$out"
  OUTS+=("$out")
fi

out="$OUT_DIR/scores_${MODEL}_fold${FOLD}_heldout.csv"
echo "--- pass: heldout real ($(date +%H:%M:%S)) ---"
"$PY" scripts/score_files_phase3.py "${COMMON[@]}" --split heldout \
  --features-csv "$FEATURES" --output-csv "$out"
OUTS+=("$out")

if [[ "$ARM" == side_on ]]; then
  out="$OUT_DIR/scores_${MODEL}_fold${FOLD}_val_noise.csv"
  echo "--- pass 3/3: val_noise levels $NOISE_LEVELS ($(date +%H:%M:%S)) ---"
  "$PY" scripts/score_files_phase3.py "${COMMON[@]}" --split val \
    --table "$TABLE" --folds-json "$FOLDS_JSON" \
    --noise-levels "$NOISE_LEVELS" --noise-calibration "$CALIBRATION" \
    --noise-files clean --noise-max-windows-per-file "$NOISE_MAX_WINDOWS" \
    --noise-seed "$NOISE_SEED" --output-csv "$out"
  OUTS+=("$out")
fi

for csv in "${OUTS[@]}"; do
  python3 - "${csv%.csv}.json" <<'PY'
import json, sys
p = sys.argv[1]
m = json.load(open(p))
assert m.get("complete") is True, f"{p} is not complete"
print(f"Sidecar OK: {p}  auc={json.dumps(m['auc'])}")
PY
done
echo "Finished: $(date)"
