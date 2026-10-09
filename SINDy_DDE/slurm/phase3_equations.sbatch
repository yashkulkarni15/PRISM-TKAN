#!/bin/bash
#SBATCH --job-name=pri2m-equations
#SBATCH --account=jaabell0
# CPU on purpose, like every XAI wrapper here (hooks capture inputs with .cpu()).
# Wall time: 12 h is generous for windows/tkan/minirocket (unmeasured); evalcheck is
# measured at ~3.9 h per CPU validation pass x 3 passes = ~12 h per checkpoint, so it
# is submitted once per checkpoint with --time=24:00:00 on the command line.
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase3_equations_%j.log
#SBATCH --error=logs/phase3_equations_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Equations across seeds. STEP picks the runner; the same windows feed all of them,
# so tkan / minirocket / evalcheck need the windows step to have FINISHED (or chain
# them: sbatch --dependency=afterok:<windows job ids> ...). Outputs land in
# $PROJECT_ROOT/results/phase3/xai/equations.
#
# Run from $PROJECT_ROOT. All four steps (the array size goes on the command line,
# not in this header, because windows and evalcheck are single jobs):
#
#   1. windows (2 jobs, one per fold):
#      W0=$(FOLD=0 STEP=windows sbatch --parsable --export=ALL slurm/phase3_equations.sbatch)
#      W2=$(FOLD=2 STEP=windows sbatch --parsable --export=ALL slurm/phase3_equations.sbatch)
#
#   2. tkan (8-task array over the NETWORKS below; no NETWORKS export needed):
#      STEP=tkan sbatch --array=0-7 --dependency=afterok:$W0:$W2 --export=ALL slurm/phase3_equations.sbatch
#
#   3. minirocket (8-task array over FOLD:SEED = 0:0 0:1 0:2 0:3 2:0 2:1 2:2 2:3; prism_aeon env).
#      16 CPUs like slurm/phase3_minirocket.sbatch: the thread count must not perturb the
#      1e-9 seed-0 reproduction check, so the step refuses any other N_JOBS:
#      STEP=minirocket sbatch --array=0-7 --cpus-per-task=16 --mem=48G --dependency=afterok:$W0:$W2 --export=ALL slurm/phase3_equations.sbatch
#
#   4. evalcheck (TWO jobs, one checkpoint each; needs no windows, the dependency is optional):
#      CKPT_NAME=f0s0 STEP=evalcheck sbatch --time=24:00:00 --export=ALL slurm/phase3_equations.sbatch
#      CKPT_NAME=f0s1 STEP=evalcheck sbatch --time=24:00:00 --export=ALL slurm/phase3_equations.sbatch
#      (TAG defaults to CKPT_NAME; each writes evalcheck_<TAG>.json, the analysis reads both.)
#
# Overrides (single manual run, NOT in an array: with SLURM_ARRAY_TASK_ID set the step
# refuses them, because --export=ALL would otherwise run the same job in every task):
# FOLD / SEED / TAG (minirocket), NETWORK (tkan; one NAME=CKPT=MANIFEST triple).
# NETWORKS (space-separated triples) replaces the default list; NAME must start with f<fold>.
# Nothing is overwritten: a step refuses to start if its output sidecar is already complete.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
STEP="${STEP:?set STEP=windows|tkan|minirocket|evalcheck}"
R="$PROJECT_ROOT/results/phase3"
TABLE="${TABLE:-$R/phase3_dataset.csv}"
FOLDS_JSON="${FOLDS_JSON:-$R/phase3_folds.json}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
FEATURES="${FEATURES:-$PROJECT_ROOT/data/metadata/features_datasets_121_125.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$R/xai/equations}"
MR_DIR="${MR_DIR:-$R/minirocket}"
TASK="${SLURM_ARRAY_TASK_ID:-}"

# Seed-0 runs (preprocessing + manifests) and the seed 1-3 retrains.
S0F0=$R/runs/61c67bddbdbd49ba9727ff92f4814c86
S0F2=$R/runs/a654129d2cdc460093e21373f31ee5a5
RS=$R/runs_seeds
M=$R/xai/mode_ablation
DEFAULT_NETWORKS="\
f0s0=$S0F0/checkpoints/best_auc.pt=$M/mode_ablation_fold0_k3.json \
f0s1=$RS/84bb905ec91c4d2786708b9d24cdf3c4/checkpoints/best_auc.pt=$M/mode_ablation_fold0_s1_k3.json \
f0s2=$RS/e5bc6229519447028351b8e711244d5e/checkpoints/best_auc.pt=$M/mode_ablation_fold0_s2_k3.json \
f0s3=$RS/21c431d7254849a9b3a11267abe2e0bc/checkpoints/best_auc.pt=$M/mode_ablation_fold0_s3_k3.json \
f2s0=$S0F2/checkpoints/best_auc.pt=$M/mode_ablation_fold2_w512_k3_mem.json \
f2s1=$RS/a8c8a52afcbc4984bc1fdd9c1f87985d/checkpoints/best_auc.pt=$M/mode_ablation_fold2_s1_k3.json \
f2s2=$RS/c0e86b7f93a648779b79c87ed5ada202/checkpoints/best_auc.pt=$M/mode_ablation_fold2_s2_k3.json \
f2s3=$RS/c89911bd89104aed95341014d9ad1458/checkpoints/best_auc.pt=$M/mode_ablation_fold2_s3_k3.json"
NETWORKS="${NETWORKS:-$DEFAULT_NETWORKS}"
FOLD_SEEDS=(0:0 0:1 0:2 0:3 2:0 2:1 2:2 2:3)
# Mode dropped from a network's cluster by recorded judgement (as in the recurrence run).
EXCLUDE="${EXCLUDE-f2s0:0}"

cd "$PROJECT_ROOT"
module load python3.11-anaconda
if [[ "$STEP" == "minirocket" ]]; then
  # aeon needs its own env (see phase3_minirocket.sbatch); prism stays untouched.
  source "${ENV_DIR:-/home/$USER/envs/prism_aeon}/bin/activate"
else
  source /home/$USER/envs/prism/bin/activate
fi
mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

need() { for f in "$@"; do [[ -e "$f" ]] || { echo "ERROR: missing input $f" >&2; exit 1; }; done; }
# Fail the job unless the sidecar exists and says "complete": true.
check_complete() {
  python3 - "$1" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
if m.get("complete") is not True:
    sys.exit(f'ERROR: {sys.argv[1]} lacks "complete": true')
print(f"OK: {sys.argv[1]}")
PY
}

# Refuse (never delete) if the step's output sidecar already landed complete.
refuse_if_complete() {
  python3 - "$1" <<'PY'
import json, os, sys
p = sys.argv[1]
if os.path.exists(p):
    try:
        done = json.load(open(p)).get("complete") is True
    except ValueError:
        done = False
    if done:
        sys.exit(f"ERROR: {p} already exists and is complete; refusing to overwrite it. "
                 "Choose a new OUTPUT_DIR / TAG (nothing was deleted).")
PY
}
# In an array task, per-job overrides inherited via --export=ALL would make every task identical.
no_array_overrides() {
  [[ -z "$TASK" ]] && return 0
  for v in "$@"; do
    [[ -z "${!v:-}" ]] || { echo "ERROR: $v is set but this is array task $TASK; every task would run the same job. Unset $v and resubmit." >&2; exit 1; }
  done
}

echo "=== PRI2M equations across seeds: STEP=$STEP ==="
echo "Started: $(date)"

case "$STEP" in
  windows)
    FOLD="${FOLD:?set FOLD=0|2 for STEP=windows}"
    case "$FOLD" in
      0) PRE="$S0F0/preprocessing.json" ;;
      2) PRE="$S0F2/preprocessing.json" ;;
      *) echo "ERROR: no seed-0 preprocessing for fold $FOLD" >&2; exit 1 ;;
    esac
    need "$PRE" "$TABLE" "$FOLDS_JSON" "$FEATURES" "$NPZ_ROOT"
    refuse_if_complete "$OUTPUT_DIR/windows_fold$FOLD.json"
    python3 scripts/equation_windows_phase3.py \
      --preprocessing "$PRE" --table "$TABLE" --folds-json "$FOLDS_JSON" \
      --features-csv "$FEATURES" --npz-root "$NPZ_ROOT" --fold "$FOLD" \
      --output-dir "$OUTPUT_DIR"
    check_complete "$OUTPUT_DIR/windows_fold$FOLD.json"
    ;;
  tkan)
    no_array_overrides NETWORK
    if [[ -z "${NETWORK:-}" ]]; then
      read -r -a NET_ARR <<< "$NETWORKS"
      [[ -n "$TASK" && "$TASK" -lt "${#NET_ARR[@]}" ]] || { echo "ERROR: need SLURM_ARRAY_TASK_ID in 0..$((${#NET_ARR[@]}-1)) (or set NETWORK)" >&2; exit 1; }
      NETWORK="${NET_ARR[$TASK]}"
    fi
    IFS='=' read -r NAME CKPT MANIFEST <<< "$NETWORK"
    FOLD="${NAME:1:1}"
    WIN="$OUTPUT_DIR/windows_fold$FOLD.npz"
    need "$CKPT" "$MANIFEST" "$WIN" "${WIN%.npz}.json"
    refuse_if_complete "$OUTPUT_DIR/tkan_$NAME.json"
    EXCL_ARGS=()
    for e in $EXCLUDE; do EXCL_ARGS+=(--exclude-mode "$e"); done
    echo "Network: $NETWORK"
    python3 scripts/equation_dump_phase3.py \
      --network "$NETWORK" --windows "$WIN" \
      ${EXCL_ARGS[@]+"${EXCL_ARGS[@]}"} \
      --output-dir "$OUTPUT_DIR"
    check_complete "$OUTPUT_DIR/tkan_$NAME.json"
    ;;
  minirocket)
    no_array_overrides FOLD SEED TAG
    if [[ -n "$TASK" ]]; then
      [[ "$TASK" -lt "${#FOLD_SEEDS[@]}" ]] || { echo "ERROR: need SLURM_ARRAY_TASK_ID in 0..$((${#FOLD_SEEDS[@]}-1))" >&2; exit 1; }
      IFS=: read -r FOLD SEED <<< "${FOLD_SEEDS[$TASK]}"
    else
      [[ -n "${FOLD:-}" && -n "${SEED:-}" ]] || { echo "ERROR: set FOLD and SEED (or run as an array task)" >&2; exit 1; }
    fi
    # Same default as phase3_minirocket.sbatch; the 1e-9 seed-0 reproduction needs 16 threads.
    N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"
    [[ "$N_JOBS" == 16 ]] || { echo "ERROR: N_JOBS=$N_JOBS; this step needs 16 (submit with --cpus-per-task=16) so threads cannot perturb the seed-0 reproduction check" >&2; exit 1; }
    export OMP_NUM_THREADS="$N_JOBS" MKL_NUM_THREADS="$N_JOBS" NUMBA_NUM_THREADS="$N_JOBS"
    WIN="$OUTPUT_DIR/windows_fold$FOLD.npz"
    need "$WIN" "${WIN%.npz}.json" "$TABLE" "$FOLDS_JSON" "$FEATURES" "$NPZ_ROOT"
    # Distinct tag: the landed minirocket_fold<F>.json (arms A+B) must not be overwritten.
    TAG="${TAG:-eq_fold${FOLD}_s${SEED}}"
    [[ "$TAG" == eq_* ]] || { echo "ERROR: TAG=$TAG must start with eq_ (never touch the landed minirocket_fold<F>.json)" >&2; exit 1; }
    [[ ! -e "$MR_DIR/minirocket_$TAG.json" ]] || { echo "ERROR: $MR_DIR/minirocket_$TAG.json already exists; choose a new TAG (nothing was deleted)" >&2; exit 1; }
    refuse_if_complete "$OUTPUT_DIR/minirocket_fold${FOLD}_s${SEED}.json"
    echo "Fold: $FOLD  seed: $SEED  tag: $TAG"
    mkdir -p "$MR_DIR"
    python3 scripts/minirocket_phase3.py \
      --fold "$FOLD" --arm B --seed "$SEED" \
      --table "$TABLE" --folds-json "$FOLDS_JSON" --npz-root "$NPZ_ROOT" \
      --features-csv "$FEATURES" --max-train-per-file "${MAX_PER_FILE:-8}" \
      --n-kernels "${N_KERNELS:-10000}" --n-jobs "$N_JOBS" \
      --output-dir "$MR_DIR" --tag "$TAG" \
      --window-file "$WIN" --equations-dir "$OUTPUT_DIR"
    check_complete "$MR_DIR/minirocket_$TAG.json"
    check_complete "$OUTPUT_DIR/minirocket_fold${FOLD}_s${SEED}.json"
    ;;
  evalcheck)
    CKPT_NAME="${CKPT_NAME:?set CKPT_NAME=f0s0|f0s1 for STEP=evalcheck}"
    case "$CKPT_NAME" in
      f0s0) C="$S0F0/checkpoints/best_auc.pt" ;;
      f0s1) C="$RS/84bb905ec91c4d2786708b9d24cdf3c4/checkpoints/best_auc.pt" ;;
      *) echo "ERROR: CKPT_NAME must be f0s0 or f0s1, got '$CKPT_NAME'" >&2; exit 1 ;;
    esac
    TAG="${TAG:-$CKPT_NAME}"
    need "$C" "$TABLE" "$FOLDS_JSON" "$NPZ_ROOT"
    refuse_if_complete "$OUTPUT_DIR/evalcheck_$TAG.json"
    python3 scripts/eval_path_check_phase3.py \
      --checkpoint "$C" \
      --table "$TABLE" --folds-json "$FOLDS_JSON" --npz-root "$NPZ_ROOT" \
      --output-dir "$OUTPUT_DIR" --tag "$TAG"
    check_complete "$OUTPUT_DIR/evalcheck_$TAG.json"
    ;;
  *) echo "ERROR: STEP must be windows|tkan|minirocket|evalcheck, got '$STEP'" >&2; exit 1 ;;
esac
echo "Finished: $(date)"
