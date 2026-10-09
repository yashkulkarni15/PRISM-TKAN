#!/bin/bash
#SBATCH --job-name=pri2m-comb-abl
#SBATCH --account=jaabell0
#
# Update 15 Section 6, causal arm for the comb reading.
#
# Sections 2 and 3 describe the delay coupling as a delay-and-add comb and
# confirm the description by probe. Section 4 states plainly that this adds a
# mechanism and NOT a line of evidence, because nothing deletes those pairs.
# This job deletes them.
#
# Three arms: the comb-carrying pairs, an importance-matched random control, and
# the sharp one -- an importance-matched control drawn from delay pairs that sit
# BELOW the proportionality gate. Structurally identical, same delayed inputs,
# no comb.
#
# CPU only and deliberately so: this evaluates a frozen checkpoint, there is no
# backward pass, and the GPU allocation is the scarce resource.
#
# KNOWN: fold 1 has 854 comb edges against only 682 pairs below the gate, so its
# NON-PROPORTIONAL control cannot be built. The runner detects that after loading
# the network but before the held-out evaluation, reports the arm absent, and the
# verdict comes back UNDECIDED -- which is not a pass. Its random control is
# unaffected, being drawn from all ~14k edges in the network.
#
# The first run of this script drew that random control from the delay pairs
# alone. Since the comb set and the below-gate set exhaust that population, the
# two controls became one, and four folds ran it without erroring. Only fold 1
# made the draw arithmetically impossible and exposed it. The population is now
# the whole network, and `check_population_is_wider` refuses anything narrower.
#
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase3_comb_ablation_%j.log
#SBATCH --error=logs/phase3_comb_ablation_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL
#
# Submit:
#   FOLD=fold2_w512 sbatch --export=ALL slurm/phase3_comb_ablation.sbatch
#
# All five:
#   for f in fold0 fold1 fold2_w512 fold3 fold4; do
#     FOLD=$f sbatch --export=ALL slurm/phase3_comb_ablation.sbatch
#   done

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
FOLD="${FOLD:-fold2_w512}"
# The regenerative manifest carries BOTH the checkpoint path and the per-pair
# corr/gain the gate reads, so one file pins the network and the selection
# together and they cannot be mismatched.
REGEN="${REGEN:-$PROJECT_ROOT/results/phase3/xai/regenerative/$FOLD/regenerative.json}"
# The 121-125 held-out set. NOT features_121_155.csv, which has never existed.
FEATURES="${FEATURES:-$PROJECT_ROOT/data/metadata/features_datasets_121_125.csv}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
N_WINDOWS="${N_WINDOWS:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/comb_ablation}"
# MUST match the value Update 15 Section 2 reports, or this deletes a different
# set of pairs than the text describes.
CORR_THRESHOLD="${CORR_THRESHOLD:-0.9}"
DEVICE="${DEVICE:-cpu}"
FLOOR="${FLOOR:-0.0092}"
SEED="${SEED:-0}"
N_TRIALS="${N_TRIALS:-200}"
TAG="${TAG:-$FOLD}"

cd "$PROJECT_ROOT"
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate
mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CKPT="${CKPT:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['provenance']['checkpoint'])" "$REGEN")}"

echo "=== PRI2M comb ablation (Update 15 section 6) ==="
echo "Fold:       $FOLD"
echo "Checkpoint: $CKPT"
echo "Regen:      $REGEN"
echo "Features:   $FEATURES"
echo "gate: $CORR_THRESHOLD   seed: $SEED   trials: $N_TRIALS   floor: $FLOOR"
echo "Started:    $(date)"

for f in "$CKPT" "$FEATURES" "$REGEN" "$TABLE"; do
  if [[ ! -e "$f" ]]; then echo "ERROR: missing input $f" >&2; exit 1; fi
done

python3 scripts/comb_ablation_phase3.py \
  --checkpoint "$CKPT" \
  --regenerative "$REGEN" \
  --features-csv "$FEATURES" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --n-windows "$N_WINDOWS" \
  --device "$DEVICE" \
  --corr-threshold "$CORR_THRESHOLD" \
  --seed "$SEED" \
  --n-trials "$N_TRIALS" \
  --floor "$FLOOR" \
  --output-dir "$OUTPUT_DIR" \
  --tag "$TAG"

DEST="$OUTPUT_DIR/comb_ablation_$TAG.json"
if [[ ! -s "$DEST" ]]; then
  echo "ERROR: finished but wrote no manifest at $DEST" >&2
  exit 1
fi
# Provenance is written before the run, so the file exists from the first
# second. Assert the completion flag rather than the path.
python3 -c "
import json,sys
m=json.load(open(sys.argv[1]))
assert m['provenance']['complete'] is True, 'manifest is a partial, not a finished run'
v=m['verdict']
print('comb arm auc      :', m['arms']['comb_pairs']['auc'])
print('drop              :', v['mode_drop'])
print('vs random         :', v['random_matched_drop'])
print('vs nonproportional:', v['other_modes_drop'])
print('CLAIMABLE         :', v['claimable'], '(None means UNDECIDED, not a pass)')
" "$DEST"

echo "Finished: $(date)"
