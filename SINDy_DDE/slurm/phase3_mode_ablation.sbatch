#!/bin/bash
#SBATCH --job-name=pri2m-mode-abl
#SBATCH --account=jaabell0
# GPU only because the held-out evaluation runs windows through the network.
# If the account is at its GPU cap (squeue reason AssocGrpGRES), this will queue
# behind it: pass DEVICE=cpu and a longer --time to run it without a GPU
# instead, which is slower but not blocked.
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/phase3_mode_ablation_%j.log
#SBATCH --error=logs/phase3_mode_ablation_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# U14-B step 8. Zeroes each activation mode's edges and measures held-out AUC
# against two mandatory controls: a random edge set matched on count AND summed
# importance, and an equal-size set drawn from the other modes.
#
# Cost scales as (1 + 3*k) held-out evaluations, so k=4 is 13 passes. Start with
# one fold before committing to five.
#
# Submit:
#   FOLD=fold2_w512 RUN=a654129d2cdc460093e21373f31ee5a5 K=4 \
#     sbatch --export=ALL slurm/phase3_mode_ablation.sbatch
#
# The run id can be read out of the regenerative manifest rather than typed:
#   python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['provenance']['checkpoint'])" \
#     results/phase3/xai/regenerative/fold2_w512/regenerative.json

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
FOLD="${FOLD:-fold2_w512}"
REGEN="${REGEN:-$PROJECT_ROOT/results/phase3/xai/regenerative/$FOLD/regenerative.json}"
FEATURES="${FEATURES:-$PROJECT_ROOT/data/metadata/features_121_155.csv}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/mode_ablation}"
K="${K:-4}"
N_WINDOWS="${N_WINDOWS:-256}"
DEVICE="${DEVICE:-cuda}"
FLOOR="${FLOOR:-0.0092}"

cd "$PROJECT_ROOT"
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate
mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Read the checkpoint from the manifest so the ablation cannot be run against a
# different network than the modes were clustered on.
CKPT="${CKPT:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['provenance']['checkpoint'])" "$REGEN")}"

echo "=== PRI2M mode ablation (U14-B step 8) ==="
echo "Fold:       $FOLD"
echo "Checkpoint: $CKPT"
echo "Features:   $FEATURES"
echo "k:          $K   windows: $N_WINDOWS   device: $DEVICE   floor: $FLOOR"
echo "Started:    $(date)"

for f in "$CKPT" "$FEATURES" "$TABLE"; do
  if [[ ! -e "$f" ]]; then echo "ERROR: missing input $f" >&2; exit 1; fi
done

python3 scripts/mode_ablation_phase3.py \
  --checkpoint "$CKPT" \
  --features-csv "$FEATURES" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --device "$DEVICE" \
  --n-windows "$N_WINDOWS" \
  --k "$K" \
  --floor "$FLOOR" \
  --output-dir "$OUTPUT_DIR" \
  --tag "$FOLD"

DEST="$OUTPUT_DIR/mode_ablation_$FOLD.json"
if [[ ! -s "$DEST" ]]; then
  echo "ERROR: finished but wrote no manifest at $DEST" >&2
  exit 1
fi
echo "Wrote: $DEST"
echo "Finished: $(date)"
