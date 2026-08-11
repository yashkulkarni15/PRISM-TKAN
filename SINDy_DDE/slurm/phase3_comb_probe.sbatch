#!/bin/bash
#SBATCH --job-name=pri2m-comb-probe
#SBATCH --account=jaabell0
# CPU partition on purpose. This probe evaluates 1-D learned activation
# functions, not forward passes over data, so it needs no GPU -- which also
# means it does NOT compete for the jaabell0 GPU allocation (squeue reason
# AssocGrpGRES), the cap that is currently holding up the determinism runs.
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/phase3_comb_probe_%j.log
#SBATCH --error=logs/phase3_comb_probe_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Confirms the tooth-pass comb reading ON THE NETWORK: sweeps a sinusoid through
# each top delay pair's two learned edge functions and compares the measured
# response against the comb predicted from the stored corr/gain. Carries an
# untrained-network control, because a comb that appears on random weights is
# architecture rather than learning (the U12-B lesson).
#
# Submit (one fold):
#   FOLD=fold2_w512 RUN=a654129d2cdc460093e21373f31ee5a5 \
#     sbatch --export=ALL slurm/phase3_comb_probe.sbatch
#
# All five folds: submit once per fold with the matching RUN id. The run ids are
# in each results/phase3/xai/regenerative/<fold>/regenerative.json under
# provenance.checkpoint, so they never have to be typed from memory:
#   python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['provenance']['checkpoint'])" \
#     results/phase3/xai/regenerative/fold2_w512/regenerative.json

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
FOLD="${FOLD:-fold2_w512}"
REGEN="${REGEN:-$PROJECT_ROOT/results/phase3/xai/regenerative/$FOLD/regenerative.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/comb}"
FIG_DIR="${FIG_DIR:-$PROJECT_ROOT/results/figures/research_update_assets}"
TAU="${TAU:-87}"
N_PAIRS="${N_PAIRS:-25}"

cd "$PROJECT_ROOT"
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate
mkdir -p logs "$OUTPUT_DIR" "$FIG_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# The checkpoint is read out of the regenerative manifest rather than passed in,
# so the probe can never be run against a different network than the prediction
# it is confirming.
CKPT="${CKPT:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['provenance']['checkpoint'])" "$REGEN")}"

echo "=== PRI2M comb probe (CPU) ==="
echo "Fold:       $FOLD"
echo "Regen:      $REGEN"
echo "Checkpoint: $CKPT"
echo "tau:        $TAU   pairs: $N_PAIRS"
echo "Started:    $(date)"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint from the manifest does not exist: $CKPT" >&2
  exit 1
fi

python3 scripts/comb_probe_phase3.py \
  --checkpoint "$CKPT" \
  --regen-manifest "$REGEN" \
  --tau "$TAU" \
  --n-pairs "$N_PAIRS" \
  --output-dir "$OUTPUT_DIR" \
  --fig-dir "$FIG_DIR" \
  --tag "$FOLD"

# Assert the artifact exists rather than announcing a path that may not have
# been written -- a job exiting 0 is not evidence it did the work.
DEST="$OUTPUT_DIR/comb_probe_$FOLD.json"
if [[ ! -s "$DEST" ]]; then
  echo "ERROR: probe finished but wrote no manifest at $DEST" >&2
  exit 1
fi
echo "Wrote: $DEST"
echo "Finished: $(date)"
