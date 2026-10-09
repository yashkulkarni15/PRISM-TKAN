#!/bin/bash
#SBATCH --job-name=pri2m-tkan-delay
#SBATCH --account=jaabell0
# CPU on purpose, like every XAI/held-out wrapper here: eval-only, and the CPU
# partition avoids the account-wide GPU cap (AssocGrpGRES).
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# 4 held-out evaluations (baseline, zero, x0.5, x2) x ~27 min each ~ 1.8 h.
#SBATCH --time=8:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/phase3_tkan_delay_heldout_%A_%a.log
#SBATCH --error=logs/phase3_tkan_delay_heldout_%A_%a.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Exp 3, matched TKAN delay test: the same three test-time variants as the
# MiniROCKET arm B (delayed input zeroed; tau misstated x0.5 and x2), applied to
# the TKAN ENCODER's tau only (side inputs keep the raw tau), on held-out
# 121-125, all five folds. One fold per array task.
#
# Submit (all five):   sbatch slurm/phase3_tkan_delay_heldout.sbatch
# One fold first:      sbatch --array=2 slurm/phase3_tkan_delay_heldout.sbatch
#
# The checkpoint is read from the fold's regenerative manifest, exactly as
# phase3_mode_ablation.sbatch does, so this scores the same network every other
# held-out XAI number on that fold was measured on.

set -euo pipefail

FOLDS=(fold0 fold1 fold2_w512 fold3 fold4)
IDX="${SLURM_ARRAY_TASK_ID:-2}"
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
FOLD="${FOLD:-${FOLDS[$IDX]}}"
REGEN="${REGEN:-$PROJECT_ROOT/results/phase3/xai/regenerative/$FOLD/regenerative.json}"
FEATURES="${FEATURES:-$PROJECT_ROOT/data/metadata/features_datasets_121_125.csv}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/tkan_delay_heldout}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FLOOR="${FLOOR:-0.0092}"
TAG="${TAG:-$FOLD}"

cd "$PROJECT_ROOT"
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate
mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -e "$REGEN" ]]; then echo "ERROR: missing regenerative manifest $REGEN" >&2; exit 1; fi
CKPT="${CKPT:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['provenance']['checkpoint'])" "$REGEN")}"

echo "=== PRI2M Exp 3: matched TKAN delay test (held-out 121-125) ==="
echo "Fold:       $FOLD (array index $IDX)"
echo "Checkpoint: $CKPT"
echo "Features:   $FEATURES"
echo "Device:     $DEVICE   floor: $FLOOR   tag: $TAG"
echo "Started:    $(date)"

for f in "$CKPT" "$FEATURES" "$NPZ_ROOT"; do
  if [[ ! -e "$f" ]]; then echo "ERROR: missing input $f" >&2; exit 1; fi
done

python3 scripts/tkan_delay_heldout_phase3.py \
  --checkpoint "$CKPT" \
  --features-csv "$FEATURES" \
  --npz-root "$NPZ_ROOT" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --floor "$FLOOR" \
  --output-dir "$OUTPUT_DIR" \
  --tag "$TAG"

DEST="$OUTPUT_DIR/tkan_delay_heldout_$TAG.json"
python3 - "$DEST" <<'PY'
import json, sys
p = sys.argv[1]
m = json.load(open(p))
assert m.get("complete") is True, f"{p} is not complete"
print(f"Manifest OK: {p}")
PY
echo "Finished:   $(date)"
