#!/bin/bash
# Sharded white-box XAI for ALL five Phase 3 folds: the expensive TSR stage is
# split across a SLURM array instead of run inline, so a publishable 128-256
# window TSR no longer has to fit one 24h (or 72h) monolith wall.
#
# Run this ON GREAT LAKES from the live project root:
#   cd /nfs/turbo/coe-jaabell/PRI2M && bash slurm/run_whitebox_sharded.sh
#
# Per fold it submits a 3-node dependency DAG:
#   (A) fixed stages   -- phase3_whitebox.sbatch with SKIP_TSR=1 (~4-5h, one job)
#   (B) TSR array      -- phase3_whitebox_tsr_array.sbatch, NSHARDS tasks in
#                         parallel (each ~= the old TSR=4 cost, well under 6h)
#   (C) merge          -- phase3_whitebox_merge.sbatch, runs afterok:A:B and folds
#                         the shards back into A's manifest (reproducing exactly
#                         what a single un-sharded run would have written)
# Wall time per fold is therefore ~max(A, B) + minutes, not sum, and a dead array
# task only loses its handful of windows -- never the whole run.
#
# Overridable knobs (defaults reproduce the 128-window sweep that timed out at
# 24h, now made to fit):
#   TSR_WINDOWS=256 NSHARDS=64 bash slurm/run_whitebox_sharded.sh
# Keep NSHARDS a divisor-ish of TSR_WINDOWS so tasks stay ~4 windows each.
set -euo pipefail

N_WINDOWS="${N_WINDOWS:-512}"        # dataset-level sample per fold (matches fixed run)
SAMPLE_SEED="${SAMPLE_SEED:-0}"      # seeded window draw; A and B MUST agree
TSR_WINDOWS="${TSR_WINDOWS:-128}"    # total TSR windows to shard
TSR_STEPS="${TSR_STEPS:-8}"
NSHARDS="${NSHARDS:-32}"             # array size = shard count (128/32 = 4 win/task)
TOP_K="${TOP_K:-25}"

ARRAY_MAX=$((NSHARDS - 1))

submit_fold () {  # fold_tag run_id val_auc
  local tag="$1" run_id="$2" val_auc="$3"
  local common=(RUN_ID="$run_id" TAG="$tag" N_WINDOWS="$N_WINDOWS" \
                SAMPLE_SEED="$SAMPLE_SEED" TSR_WINDOWS="$TSR_WINDOWS" \
                TSR_STEPS="$TSR_STEPS")

  # (A) fixed stages (everything except TSR).
  local jid_fix
  jid_fix=$(env "${common[@]}" TOP_K="$TOP_K" SKIP_TSR=1 VAL_AUC="$val_auc" \
    sbatch --parsable --export=ALL slurm/phase3_whitebox.sbatch)

  # (B) TSR array -- NSHARDS tasks; the array size defines the shard count.
  local jid_arr
  jid_arr=$(env "${common[@]}" NSHARDS="$NSHARDS" \
    sbatch --parsable --array="0-${ARRAY_MAX}" --export=ALL \
      slurm/phase3_whitebox_tsr_array.sbatch)

  # (C) merge -- only after the fixed run AND every array task succeed.
  local jid_merge
  jid_merge=$(env TAG="$tag" \
    sbatch --parsable --dependency="afterok:${jid_fix}:${jid_arr}" \
      --export=ALL slurm/phase3_whitebox_merge.sbatch)

  printf '  %-11s fixed=%s  tsr_array=%s[0-%d]  merge=%s\n' \
    "$tag" "$jid_fix" "$jid_arr" "$ARRAY_MAX" "$jid_merge"
}

echo "Submitting sharded white-box (TSR_WINDOWS=$TSR_WINDOWS across $NSHARDS shards/fold):"
submit_fold fold0      61c67bddbdbd49ba9727ff92f4814c86 0.9597
submit_fold fold1      c21e58dc0e624e1dbc602ec63a2dc23e 0.9122
submit_fold fold2_w512 a654129d2cdc460093e21373f31ee5a5 0.9752
submit_fold fold3      3a40b71514cd41b98916330c000d700a 0.9605
submit_fold fold4      0937883918354ed899a61afb90aac14f 0.9695

echo ""
echo "Watch:  squeue -u \$USER"
echo "Each fold's 'merge' job fires automatically once its fixed run + all TSR"
echo "shards succeed; if a shard fails, fix it and resubmit just that --array id,"
echo "then run the merge by hand:"
echo "  python scripts/merge_tsr_shards.py --output-dir results/phase3/xai/<TAG>"
echo ""
echo "Then pull artifacts to the Mac (one rsync, run LOCALLY):"
echo "  rsync -av yashkulk@greatlakes.arc-ts.umich.edu:/nfs/turbo/coe-jaabell/PRI2M/results/phase3/xai/ \\"
echo "    ~/Downloads/PRI2M/SINDy_DDE/results/phase3/xai/"
