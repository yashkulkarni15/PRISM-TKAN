#!/bin/bash
#SBATCH --job-name=pri2m-prune
#SBATCH --account=jaabell0
# CPU ON PURPOSE, matching every other XAI wrapper in this project
# (phase3_activation_modes.sbatch, phase3_whitebox.sbatch -- neither has a device
# flag at all; the whitebox analysis runs in ~9 min on CPU).
#
# The XAI layer cannot run on CUDA. activations.py:84 captures every hook input
# with .cpu() to bound memory, then feeds it back into the live module; and
# symbolic._eval_edge builds CPU tensors and ends in .numpy(). With a CUDA model
# both raise "Expected all tensors to be on the same device". Jobs 57252876 and
# 57253615 died that way on 2026-08-12. Only the held-out evaluation would
# benefit from a GPU, and mixing the two is what caused those failures.
#
# Running on the CPU partition also sidesteps the account-wide GPU cap
# (squeue reason AssocGrpGRES) entirely, which the GPU version could not:
# DEVICE=cpu alone still left --gres=gpu:1 requested and still queued behind it.
#
# WALL TIME IS UNMEASURED. Time one fold before committing the remaining four.
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase3_heldout_pruning_%j.log
#SBATCH --error=logs/phase3_heldout_pruning_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Item B of the 2026-08-12 spline-substitution spec. Re-measures the pruning
# curve on the HELD-OUT path against a size-matched random-edge control.
#
# The published curve was computed on the in-distribution sample and is
# under-resolved at both ends: no point below 0.02, and its steepest measured
# segment is 0.50 -> 1.00 in three of five folds.
#
# Cost is 1 baseline + 2 evaluations per fraction (ranked arm + random control).
# The 12 default fractions are therefore 25 held-out passes; start with one fold.
#
# Submit:
#   FOLD=fold2_w512 sbatch --export=ALL slurm/phase3_heldout_pruning.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
FOLD="${FOLD:-fold2_w512}"
REGEN="${REGEN:-$PROJECT_ROOT/results/phase3/xai/regenerative/$FOLD/regenerative.json}"
# The 121-125 held-out set, which is what the 0.908 +/- 0.013 baseline was
# measured on (phase3_eval_heldout.sbatch tag "121_125"). NOT
# features_121_155.csv: 126-155 cannot be built from the current Kaggle
# download, which ships those datasets as a single rectified envelope channel,
# so that file has never existed on either side.
FEATURES="${FEATURES:-$PROJECT_ROOT/data/metadata/features_datasets_121_125.csv}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/heldout_pruning}"
N_WINDOWS="${N_WINDOWS:-256}"
DEVICE="${DEVICE:-cpu}"
TOL="${TOL:-0.013}"
FRACTIONS="${FRACTIONS:-}"
# The random control is a SINGLE draw at this seed, so the whole seed-repeat
# experiment is impossible without this passthrough. The manifest path is
# "$OUTPUT_DIR/$TAG/heldout_pruning.json", so TAG must carry the seed or every
# repeat overwrites the last one.
# NOTE: the first landed run (2026-08-13, seed 0) predates this and is stored
# under the older name "$OUTPUT_DIR/fold2_w512/".
SEED="${SEED:-0}"
TAG="${TAG:-${FOLD}_s${SEED}}"

cd "$PROJECT_ROOT"
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate
mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Read the checkpoint from the manifest so this runs against the same network
# every other analysis of this fold used.
CKPT="${CKPT:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['provenance']['checkpoint'])" "$REGEN")}"

echo "=== PRI2M held-out pruning curve (spec item B) ==="
echo "Fold:       $FOLD"
echo "Checkpoint: $CKPT"
echo "Features:   $FEATURES"
echo "windows: $N_WINDOWS   device: $DEVICE   tol: $TOL"
echo "Seed:       $SEED   tag: $TAG"
echo "Started:    $(date)"

for f in "$CKPT" "$FEATURES" "$TABLE"; do
  if [[ ! -e "$f" ]]; then echo "ERROR: missing input $f" >&2; exit 1; fi
done

EXTRA=()
if [[ -n "$FRACTIONS" ]]; then EXTRA+=(--fractions "$FRACTIONS"); fi

python3 scripts/heldout_pruning_phase3.py \
  --checkpoint "$CKPT" \
  --features-csv "$FEATURES" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --device "$DEVICE" \
  --n-windows "$N_WINDOWS" \
  --tol "$TOL" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR" \
  --tag "$TAG" \
  "${EXTRA[@]+"${EXTRA[@]}"}"

DEST="$OUTPUT_DIR/$TAG/heldout_pruning.json"
if [[ ! -s "$DEST" ]]; then
  echo "ERROR: finished but wrote no manifest at $DEST" >&2
  exit 1
fi
# A partial manifest is written after every fraction, so presence proves
# nothing. Assert the completion flag the runner sets last.
python3 -c "
import json,sys
m=json.load(open(sys.argv[1]))
assert m['provenance']['complete'] is True, 'manifest is a partial, not a finished run'
print('rows:', len(m['rows']), ' smallest faithful fraction:', m.get('smallest_faithful_fraction'))
" "$DEST"
echo "Wrote: $DEST"
echo "Finished: $(date)"
