#!/bin/bash
#SBATCH --job-name=pri2m-recurrence
#SBATCH --account=jaabell0
# CPU on purpose, like every XAI wrapper here: the XAI layer captures hook inputs
# with .cpu() and cannot run on CUDA (see phase3_mode_ablation.sbatch). This job
# needs no held-out evaluation at all -- one forward pass per network for the
# curves and hidden traces, 20 weight re-reads per network for the untrained
# null, and 10 000 cheap relabellings per pair.
#
# WALL TIME IS UNMEASURED; 12 h is generous for 8 networks.
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase3_recurrence_%j.log
#SBATCH --error=logs/phase3_recurrence_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Exp 2 step 3: does the causally-necessary x_delay cluster recur across seeds,
# compared by FUNCTION (curve shape on the same input channel; Hungarian-aligned
# hidden units), never by raw edge ID. See scripts/recurrence_phase3.py.
#
# Order of operations on GL:
#   1. bash slurm/submit_seed_retrain.sh                 (6 GPU jobs, folds 0,2 x seeds 1-3)
#   2. mode ablation on each new run (k=3, same as the seed-0 manifests)
#   3. this job.
#
# NETWORKS holds space-separated NAME=CKPT=ABLATION_MANIFEST triples. NAME must
# start with f<fold>; networks of one fold are compared pairwise (the per-fold
# "k of n pairs recur"), cross-fold pairs are reported separately.
#
# Submit, folds 0 and 2, once the seed runs and their ablations exist (replace
# <f0s1_run> etc. with the run ids under results/phase3/runs_seeds/):
#   R=/nfs/turbo/coe-jaabell/PRI2M/results/phase3
#   M=$R/xai/mode_ablation
#   NETWORKS="\
#   f0s0=$R/runs/61c67bddbdbd49ba9727ff92f4814c86/checkpoints/best_auc.pt=$M/mode_ablation_fold0_k3.json \
#   f0s1=$R/runs_seeds/<f0s1_run>/checkpoints/best_auc.pt=$M/mode_ablation_fold0_s1_k3.json \
#   f0s2=$R/runs_seeds/<f0s2_run>/checkpoints/best_auc.pt=$M/mode_ablation_fold0_s2_k3.json \
#   f0s3=$R/runs_seeds/<f0s3_run>/checkpoints/best_auc.pt=$M/mode_ablation_fold0_s3_k3.json \
#   f2s0=$R/runs/a654129d2cdc460093e21373f31ee5a5/checkpoints/best_auc.pt=$M/mode_ablation_fold2_w512_k3_mem.json \
#   f2s1=$R/runs_seeds/<f2s1_run>/checkpoints/best_auc.pt=$M/mode_ablation_fold2_s1_k3.json \
#   f2s2=$R/runs_seeds/<f2s2_run>/checkpoints/best_auc.pt=$M/mode_ablation_fold2_s2_k3.json \
#   f2s3=$R/runs_seeds/<f2s3_run>/checkpoints/best_auc.pt=$M/mode_ablation_fold2_s3_k3.json" \
#   TAG=fold0_fold2 sbatch --export=ALL slurm/phase3_recurrence.sbatch
#
# Fold 2 must use the *_mem manifest: mode_ablation_fold2_w512_k3.json predates
# membership logging and records no edges, so its clusters cannot be compared.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
NETWORKS="${NETWORKS:-}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
TABLE="${TABLE:-$PROJECT_ROOT/results/phase3/phase3_dataset.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/recurrence}"
N_WINDOWS="${N_WINDOWS:-256}"
SEED="${SEED:-0}"
N_UNTRAINED="${N_UNTRAINED:-20}"
N_PERM="${N_PERM:-10000}"
TAG="${TAG:-recurrence}"
# Modes dropped from a network's necessary cluster by recorded judgement
# (space-separated NAME:MODE). Default: fold 2 k=3 mode 0, which clears the
# floor by 0.0004 and is excluded from every claim since 2026-09-21. Set
# EXCLUDE="" to count it. Recorded in the manifest's provenance either way.
EXCLUDE="${EXCLUDE-f2s0:0}"

if [[ -z "$NETWORKS" ]]; then
  echo "ERROR: set NETWORKS to space-separated NAME=CKPT=MANIFEST triples" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate
mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1   # stream progress to the log (else it buffers until exit)

# Split on whitespace only (no globbing), one --network per triple.
read -r -a NET_ARR <<< "$NETWORKS"
NET_ARGS=()
for spec in "${NET_ARR[@]}"; do
  IFS='=' read -r name ckpt manifest <<< "$spec"
  for f in "$ckpt" "$manifest"; do
    if [[ ! -e "$f" ]]; then echo "ERROR: $name: missing input $f" >&2; exit 1; fi
  done
  NET_ARGS+=(--network "$spec")
done
EXCL_ARGS=()
if [[ -n "$EXCLUDE" ]]; then
  read -r -a EXCL_ARR <<< "$EXCLUDE"
  for e in "${EXCL_ARR[@]}"; do EXCL_ARGS+=(--exclude-mode "$e"); done
fi
for f in "$TABLE" "$NPZ_ROOT"; do
  if [[ ! -e "$f" ]]; then echo "ERROR: missing input $f" >&2; exit 1; fi
done

echo "=== PRI2M Exp 2 recurrence ==="
echo "Networks:   ${#NET_ARR[@]}"
printf '  %s\n' "${NET_ARR[@]}"
echo "Exclude:    ${EXCLUDE:-(none)}"
echo "Windows: $N_WINDOWS  seed: $SEED  untrained: $N_UNTRAINED  perms: $N_PERM  tag: $TAG"
echo "Started:    $(date)"

python3 scripts/recurrence_phase3.py \
  "${NET_ARGS[@]}" \
  ${EXCL_ARGS[@]+"${EXCL_ARGS[@]}"} \
  --table "$TABLE" \
  --npz-root "$NPZ_ROOT" \
  --n-windows "$N_WINDOWS" \
  --seed "$SEED" \
  --n-untrained "$N_UNTRAINED" \
  --n-perm "$N_PERM" \
  --output-dir "$OUTPUT_DIR" \
  --tag "$TAG"

DEST="$OUTPUT_DIR/recurrence_$TAG.json"
if [[ ! -s "$DEST" ]]; then
  echo "ERROR: finished but wrote no manifest at $DEST" >&2
  exit 1
fi
python3 - "$DEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
if m.get("complete") is not True:
    sys.exit(f"ERROR: {sys.argv[1]} lacks \"complete\": true")
for fold, s in m["per_fold"].items():
    print(f"fold {fold}: {s['summary']}")
PY
echo "Wrote: $DEST"
echo "Finished: $(date)"
