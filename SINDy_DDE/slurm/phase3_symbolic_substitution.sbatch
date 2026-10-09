#!/bin/bash
#SBATCH --job-name=pri2m-symsub
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
#SBATCH --output=logs/phase3_symbolic_substitution_%j.log
#SBATCH --error=logs/phase3_symbolic_substitution_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Item A of the 2026-08-12 spline-substitution spec. Replaces the top-k edges'
# learned activations with their fitted closed forms at forward time and
# re-scores the network on held-out data.
#
# PRE-COMMITTED, per fold, against THIS FOLD'S OWN baseline (fold 2 = 0.8994,
# never the five-fold mean 0.908):
#   |AUC_substituted(k=25) - AUC_baseline| <= 0.013   AND
#    AUC_substituted(k=25) - AUC_permuted  >  0.013
#
# Arms: baseline + 6 k-values x {correct, permuted} + k=25 clamped x 2
#     = 15 held-out evaluations.
#
# Submit:
#   FOLD=fold2_w512 sbatch --export=ALL slurm/phase3_symbolic_substitution.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
FOLD="${FOLD:-fold2_w512}"
REGEN="${REGEN:-$PROJECT_ROOT/results/phase3/xai/regenerative/$FOLD/regenerative.json}"
# See phase3_heldout_pruning.sbatch for why this is the 121-125 set and not
# features_121_155.csv, which has never existed.
FEATURES="${FEATURES:-$PROJECT_ROOT/data/metadata/features_datasets_121_125.csv}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/symbolic_substitution}"
N_WINDOWS="${N_WINDOWS:-256}"
DEVICE="${DEVICE:-cpu}"
TOL="${TOL:-0.013}"
K_SWEEP="${K_SWEEP:-1,5,10,15,20,25}"
R2_THRESHOLD="${R2_THRESHOLD:-0.9}"
# --seed drives the derangement that forms the control arm, so it must be
# reachable from the wrapper. The fold sweep gives each fold its own TAG
# directory, so the default cannot collide; TAG is here for seed repeats.
SEED="${SEED:-0}"
# Restrict the substitution to one family of edges, e.g. RESTRICT_TO='delay:'.
# The published five-fold run covers the global top 25 by importance, of which
# only three touch a delayed channel, so it shows the equations reproduce the
# NETWORK and not the delay mechanism Sections 2 and 3 describe. Unset, argv is
# byte-identical to the published run.
RESTRICT_TO="${RESTRICT_TO:-}"
# Neither a restricted run nor a re-shuffle may overwrite the manifest beside
# it, so the default TAG carries whatever was varied. Seed 0 adds nothing, which
# keeps the landed five-fold paths exactly as published.
_SUFFIX=""
if [[ -n "$RESTRICT_TO" ]]; then
  _SUFFIX="${_SUFFIX}_$(printf '%s' "$RESTRICT_TO" | tr -c 'A-Za-z0-9' '_')"
fi
if [[ "$SEED" != "0" ]]; then _SUFFIX="${_SUFFIX}_s${SEED}"; fi
TAG="${TAG:-${FOLD}${_SUFFIX}}"

cd "$PROJECT_ROOT"
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate
mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

CKPT="${CKPT:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['provenance']['checkpoint'])" "$REGEN")}"

echo "=== PRI2M symbolic substitution (spec item A) ==="
echo "Fold:       $FOLD"
echo "Checkpoint: $CKPT"
echo "Features:   $FEATURES"
echo "k sweep: $K_SWEEP   windows: $N_WINDOWS   device: $DEVICE   tol: $TOL"
echo "Seed:       $SEED   tag: $TAG"
echo "Started:    $(date)"

for f in "$CKPT" "$FEATURES" "$TABLE"; do
  if [[ ! -e "$f" ]]; then echo "ERROR: missing input $f" >&2; exit 1; fi
done

EXTRA=()
if [[ -n "$RESTRICT_TO" ]]; then EXTRA+=(--restrict-to "$RESTRICT_TO"); fi

python3 scripts/symbolic_substitution_phase3.py \
  "${EXTRA[@]+"${EXTRA[@]}"}" \
  --checkpoint "$CKPT" \
  --features-csv "$FEATURES" \
  --npz-root "$NPZ_ROOT" \
  --table "$TABLE" \
  --device "$DEVICE" \
  --n-windows "$N_WINDOWS" \
  --k-sweep "$K_SWEEP" \
  --r2-threshold "$R2_THRESHOLD" \
  --tol "$TOL" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR" \
  --tag "$TAG"

DEST="$OUTPUT_DIR/$TAG/symbolic_substitution.json"
if [[ ! -s "$DEST" ]]; then
  echo "ERROR: finished but wrote no manifest at $DEST" >&2
  exit 1
fi
# A partial manifest is written after every arm, so presence proves nothing.
python3 -c "
import json,sys
m=json.load(open(sys.argv[1]))
assert m['provenance']['complete'] is True, 'manifest is a partial, not a finished run'
rows=m['rows']
print('arms:', len(rows), ' baseline:', m.get('baseline_auc'))
for r in rows:
    v=r.get('verdict') or {}
    print('  k=%-3s clamped=%-5s correct=%s permuted=%s passed=%s'
          % (r['k'], r['clamped'], r.get('correct_auc'), r.get('permuted_auc'),
             v.get('passed')))
" "$DEST"
echo "Wrote: $DEST"
echo "Finished: $(date)"
