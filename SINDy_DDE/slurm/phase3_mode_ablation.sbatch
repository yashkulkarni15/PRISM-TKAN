#!/bin/bash
#SBATCH --job-name=pri2m-mode-abl
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
# Exp 1b -- pairwise group deletion (PAIRS=1, 2026-10-05). Adds, AFTER the k
# single-mode arms (so those reproduce Exp 1), every pair of modes and all gated
# edges, each with a matched random arm: --pairs --all-gated. Cost at k=3 is
# 1 + 3k singles + 2*C(k,2) pairs + 2 all-gated = 1 + 9 + 6 + 2 = 18 held-out
# evaluations, ~27 min each on CPU, so ~8 h per fold; --time=24:00:00 stays.
# The TAG defaults to ${FOLD}_k${K}_pairs so the landed ${FOLD}_k${K} manifests
# are never overwritten, and REFERENCE defaults to the landed k=3 manifest
# (the _mem one for fold2_w512, the only one there that records membership) so
# the run records whether it reproduced the Exp 1 clusters. All five folds:
#   for F in fold0 fold1 fold2_w512 fold3 fold4; do
#     FOLD=$F K=3 PAIRS=1 sbatch --export=ALL slurm/phase3_mode_ablation.sbatch
#   done
# The checkpoint is read from results/phase3/xai/regenerative/$FOLD/regenerative.json.
#
# The run id can be read out of the regenerative manifest rather than typed:
#   python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['provenance']['checkpoint'])" \
#     results/phase3/xai/regenerative/fold2_w512/regenerative.json

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
FOLD="${FOLD:-fold2_w512}"
REGEN="${REGEN:-$PROJECT_ROOT/results/phase3/xai/regenerative/$FOLD/regenerative.json}"
# The 121-125 held-out set, which is what the 0.908 +/- 0.013 baseline this
# ablation is scored against was measured on (phase3_eval_heldout.sbatch tag
# "121_125"). NOT features_121_155.csv: 126-155 cannot be built from the current
# Kaggle download, which ships those datasets as a single rectified envelope
# channel, so that file has never existed on either side.
FEATURES="${FEATURES:-$PROJECT_ROOT/data/metadata/features_datasets_121_125.csv}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/mode_ablation}"
K="${K:-4}"
N_WINDOWS="${N_WINDOWS:-256}"
DEVICE="${DEVICE:-cpu}"
FLOOR="${FLOOR:-0.0092}"
SEED="${SEED:-0}"
# The manifest name is "mode_ablation_$TAG.json", so a TAG that does not carry k
# makes a k=3 run silently OVERWRITE a k=2 one -- and the whole point of the
# sweep is to report every k side by side. Default it to carry k.
# NOTE: the first landed run (2026-08-13, k=4) predates this and is stored under
# the older name "mode_ablation_fold2_w512.json".
# PAIRS=1 (Exp 1b) gets its own suffix: the landed "${FOLD}_k${K}" manifests
# are the Exp 1 record and must not be overwritten by a superset run.
PAIRS="${PAIRS:-0}"
if [[ "$PAIRS" == "1" ]]; then
  TAG="${TAG:-${FOLD}_k${K}_pairs}"
else
  TAG="${TAG:-${FOLD}_k${K}}"
fi
# Landed manifest to check the single-mode memberships against (never fatal).
# Only defaulted for PAIRS=1; prefer "_mem", which for fold2_w512 is the k=3
# run that recorded membership (the plain one recorded counts only).
REFERENCE="${REFERENCE:-}"
if [[ "$PAIRS" == "1" && -z "$REFERENCE" ]]; then
  for cand in "$OUTPUT_DIR/mode_ablation_${FOLD}_k${K}_mem.json" \
              "$OUTPUT_DIR/mode_ablation_${FOLD}_k${K}.json"; do
    if [[ -s "$cand" ]]; then REFERENCE="$cand"; break; fi
  done
fi

cd "$PROJECT_ROOT"
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate
mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1   # stream progress to the log (else it buffers until exit)

# Read the checkpoint from the manifest so the ablation cannot be run against a
# different network than the modes were clustered on.
CKPT="${CKPT:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['provenance']['checkpoint'])" "$REGEN")}"

echo "=== PRI2M mode ablation (U14-B step 8) ==="
echo "Fold:       $FOLD"
echo "Checkpoint: $CKPT"
echo "Features:   $FEATURES"
echo "k:          $K   windows: $N_WINDOWS   device: $DEVICE   floor: $FLOOR"
echo "Seed:       $SEED   tag: $TAG"
echo "Pairs:      $PAIRS   reference: ${REFERENCE:-none}"
echo "Started:    $(date)"

for f in "$CKPT" "$FEATURES" "$TABLE"; do
  if [[ ! -e "$f" ]]; then echo "ERROR: missing input $f" >&2; exit 1; fi
done

# argv as an array so optional flags append without unquoted expansion. With
# PAIRS and REFERENCE unset this is exactly the Exp 1 command line.
ARGS=(
  --checkpoint "$CKPT"
  --features-csv "$FEATURES"
  --npz-root "$NPZ_ROOT"
  --table "$TABLE"
  --device "$DEVICE"
  --n-windows "$N_WINDOWS"
  --k "$K"
  --seed "$SEED"
  --floor "$FLOOR"
  --output-dir "$OUTPUT_DIR"
  --tag "$TAG"
)
if [[ "$PAIRS" == "1" ]]; then ARGS+=(--pairs --all-gated); fi
if [[ -n "$REFERENCE" ]]; then ARGS+=(--reference-manifest "$REFERENCE"); fi

python3 scripts/mode_ablation_phase3.py "${ARGS[@]}"

DEST="$OUTPUT_DIR/mode_ablation_$TAG.json"
if [[ ! -s "$DEST" ]]; then
  echo "ERROR: finished but wrote no manifest at $DEST" >&2
  exit 1
fi
echo "Wrote: $DEST"
echo "Finished: $(date)"
