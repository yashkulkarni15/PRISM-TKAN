#!/bin/bash
# Transfer the team-facing model-zoo inference path to Great Lakes.
# Run from /Users/yash/Downloads/PRI2M/SINDy_DDE/.
#
# Same ControlMaster trick as sync_phase3_tkan_to_hpc.sh: one Duo prompt for the
# whole transfer instead of one per file.
#
# After this lands, publish the zoo ON Great Lakes (see the tail of this script).

set -euo pipefail

REMOTE=yashkulk@greatlakes.arc-ts.umich.edu
DEST=/nfs/turbo/coe-jaabell/PRI2M           # live GL project root (NOT .../PRI2M/SINDy_DDE)
CTRL="$HOME/.ssh/cm-pri2m.sock"

cleanup() { ssh -o ControlPath="$CTRL" -O exit "$REMOTE" 2>/dev/null || true; }
trap cleanup EXIT

echo ">>> Opening master SSH connection to Great Lakes."
echo ">>> Approve the Duo 2FA prompt ONCE; all transfers reuse this connection."
ssh -fN -M -o ControlPath="$CTRL" -o ControlPersist=15m "$REMOTE"
echo ">>> Master connection established. No further 2FA needed."

RSH="ssh -o ControlPath=$CTRL"
rs() { rsync -av --exclude=__pycache__ --exclude='*.pyc' -e "$RSH" "$@"; }

ssh -o ControlPath="$CTRL" "$REMOTE" "mkdir -p \
  $DEST/src/pri2m/phase3 $DEST/scripts $DEST/slurm $DEST/tests \
  $DEST/docs/phase3_tkan $DEST/logs"

# --- new package modules (zoo.py + predict.py ride along in the mirror) ---
rs src/pri2m/phase3/    "$REMOTE:$DEST/src/pri2m/phase3/"

# --- entry points ---
rs scripts/publish_phase3_zoo.py      "$REMOTE:$DEST/scripts/"  # NEW: freezes checkpoints into the zoo
rs scripts/predict_phase3.py          "$REMOTE:$DEST/scripts/"  # NEW: the command teammates run
rs scripts/predict_windows_phase3.py  "$REMOTE:$DEST/scripts/"  # MODIFIED: now shares the scoring loop
rs scripts/eval_heldout_phase3.py     "$REMOTE:$DEST/scripts/"  # MODIFIED: window-config moved into src

# --- job wrapper + docs + tests ---
rs slurm/phase3_predict.sbatch        "$REMOTE:$DEST/slurm/"
rs docs/phase3_tkan/USING_THE_MODEL.md "$REMOTE:$DEST/docs/phase3_tkan/"
rs tests/test_phase3_zoo.py           "$REMOTE:$DEST/tests/"
rs tests/test_predict_phase3_cli.py   "$REMOTE:$DEST/tests/"

echo
echo ">>> Transfer complete. Now, ON GREAT LAKES:"
cat <<'NEXT'

  cd /nfs/turbo/coe-jaabell/PRI2M
  module load python3.11-anaconda
  source /home/$USER/envs/prism/bin/activate
  export PYTHONPATH=$PWD/src:$PYTHONPATH

  # 1. Confirm the new code passes there too (fast, CPU-only).
  python -m pytest tests/test_phase3_zoo.py tests/test_predict_phase3_cli.py -q

  # 2. Publish the zoo (one time; creates a read-only v1).
  #    The run ids are pinned explicitly: fold 0 has more than one completed run
  #    on turbo (931af482 = the unstable lr 0.003 attempt), and auto-discovery
  #    refuses to guess. These five are the validated lr-0.0015 sweep.
  python scripts/publish_phase3_zoo.py \
      --out /nfs/turbo/coe-jaabell/PRI2M/models/phase3_tkan/v1 \
      --default-fold 4 \
      --run 0:61c67bddbdbd49ba9727ff92f4814c86 \
      --run 1:c21e58dc0e624e1dbc602ec63a2dc23e \
      --run 2:a654129d2cdc460093e21373f31ee5a5 \
      --run 3:3a40b71514cd41b98916330c000d700a \
      --run 4:0937883918354ed899a61afb90aac14f

  # 3. Make it group-readable so teammates can load it.
  chmod g+rX -R /nfs/turbo/coe-jaabell/PRI2M/models
  ls -l /nfs/turbo/coe-jaabell/PRI2M/models/phase3_tkan/v1

  # 4. Smoke-test the command your team will run.
  python scripts/predict_phase3.py --dataset 12 --out $HOME/preds_ds12.csv

NEXT
