#!/bin/bash
# Resubmit EVERY missing TSR array shard, across ALL five Phase 3 folds, in one
# shot -- and auto-wire the merge so each fold completes on its own. No picking
# indices by hand, no dependence on sacct (which drops jobs past its retention /
# start-time window).
#
# Coverage guarantee -- disk is the single source of truth. A shard counts as
# DONE iff its file
#     results/phase3/xai/<tag>/tsr_shards/shard_<k>_of_<NSHARDS>.json
# exists AND is non-empty AND ends in '}' (i.e. was fully written, not truncated
# by a mid-write kill). Anything else -- missing, empty, half-written, or a fold
# that never ran at all -- is treated as NOT done and gets resubmitted. So this is
# robust to WHY a task died (TIMEOUT / FAILED / CANCELLED / OUT_OF_MEMORY /
# NODE_FAIL) and fully idempotent: re-run it as often as you like; it only ever
# fills the gaps.
#
# Usage (from the live project root):
#   cd /nfs/turbo/coe-jaabell/PRI2M
#   DRY_RUN=1 bash slurm/resubmit_missing_tsr_shards.sh   # PREVIEW: submits nothing
#   bash slurm/resubmit_missing_tsr_shards.sh             # actually submit
#
# For each fold with gaps it submits: (1) an array over just the missing shards,
# then (2) a merge with --dependency=afterok:<that array> so the fold folds itself
# back together the moment its last gap succeeds. Folds already complete on disk
# are reported with the one-line merge command (their merge already fired if this
# script resubmitted them).
#
# Knobs MUST match the original run_whitebox_sharded.sh submission or the new
# shard files disagree with the finished ones and merge_tsr_shards.py rejects the
# set. Defaults reproduce the 128-window / 32-shard sweep.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
N_WINDOWS="${N_WINDOWS:-512}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
TSR_WINDOWS="${TSR_WINDOWS:-128}"
TSR_STEPS="${TSR_STEPS:-8}"
NSHARDS="${NSHARDS:-32}"
DRY_RUN="${DRY_RUN:-0}"

cd "$PROJECT_ROOT"

# tag:run_id -- the same five folds as slurm/run_whitebox_sharded.sh.
FOLDS=(
  "fold0:61c67bddbdbd49ba9727ff92f4814c86"
  "fold1:c21e58dc0e624e1dbc602ec63a2dc23e"
  "fold2_w512:a654129d2cdc460093e21373f31ee5a5"
  "fold3:3a40b71514cd41b98916330c000d700a"
  "fold4:0937883918354ed899a61afb90aac14f"
)

# A shard file is complete iff it exists, is non-empty, and ends in '}'.
shard_ok () {
  local p="$1"
  [[ -s "$p" ]] || return 1
  [[ "$(tail -c1 "$p" 2>/dev/null)" == "}" ]] || return 1
  return 0
}

run () {  # echo + (unless DRY_RUN) execute, returning the command's stdout
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '      [dry-run] %s\n' "$*" >&2
    echo "DRYRUN"
  else
    "$@"
  fi
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo ">>> DRY RUN -- nothing will be submitted. Unset DRY_RUN to go live. <<<"
fi
printf 'NSHARDS=%s  N_WINDOWS=%s  TSR_WINDOWS=%s  TSR_STEPS=%s  SAMPLE_SEED=%s\n' \
  "$NSHARDS" "$N_WINDOWS" "$TSR_WINDOWS" "$TSR_STEPS" "$SAMPLE_SEED"
printf 'A shard is DONE iff shard_<k>_of_%04d.json exists, is non-empty, ends in }\n\n' "$NSHARDS"

total_present=0
total_missing=0
folds_resubmitted=0

for entry in "${FOLDS[@]}"; do
  tag="${entry%%:*}"; run_id="${entry#*:}"
  xai_dir="$PROJECT_ROOT/results/phase3/xai/$tag"
  shard_dir="$xai_dir/tsr_shards"

  present=(); missing=()
  for ((k=0; k<NSHARDS; k++)); do
    f=$(printf 'shard_%04d_of_%04d.json' "$k" "$NSHARDS")
    if shard_ok "$shard_dir/$f"; then present+=("$k"); else missing+=("$k"); fi
  done
  total_present=$((total_present + ${#present[@]}))
  total_missing=$((total_missing + ${#missing[@]}))

  # Does the fixed-run (run A) manifest exist? Merge cannot succeed without it.
  manifest="$xai_dir/whitebox_manifest.json"
  [[ -f "$manifest" ]] && man_note="" || man_note="  [WARN: no whitebox_manifest.json -- rerun the fixed --skip-tsr stage (A) or the merge will fail]"

  if [[ ${#missing[@]} -eq 0 ]]; then
    printf '  %-11s DONE  %2d/%d shards present%s\n' "$tag" "${#present[@]}" "$NSHARDS" "$man_note"
    printf '      merge (if not already fired): python scripts/merge_tsr_shards.py --output-dir results/phase3/xai/%s\n' "$tag"
    continue
  fi

  csv=$(IFS=,; echo "${missing[*]}")
  printf '  %-11s GAPS  %2d/%d present, resubmit %d: array=%s%s\n' \
    "$tag" "${#present[@]}" "$NSHARDS" "${#missing[@]}" "$csv" "$man_note"
  folds_resubmitted=$((folds_resubmitted + 1))

  jid_arr=$(run env RUN_ID="$run_id" TAG="$tag" N_WINDOWS="$N_WINDOWS" SAMPLE_SEED="$SAMPLE_SEED" \
              TSR_WINDOWS="$TSR_WINDOWS" TSR_STEPS="$TSR_STEPS" NSHARDS="$NSHARDS" \
            sbatch --parsable --array="$csv" --export=ALL \
              slurm/phase3_whitebox_tsr_array.sbatch)
  jid_merge=$(run env TAG="$tag" \
            sbatch --parsable --dependency="afterok:${jid_arr}" --export=ALL \
              slurm/phase3_whitebox_merge.sbatch)
  printf '      array=%s  merge=%s (afterok:%s)\n' "$jid_arr" "$jid_merge" "$jid_arr"
done

# Coverage cross-check: any tsr_shards dir NOT in the known fold list is uncovered.
echo
if compgen -G "$PROJECT_ROOT/results/phase3/xai/*/tsr_shards" > /dev/null; then
  known=" fold0 fold1 fold2_w512 fold3 fold4 "
  for d in "$PROJECT_ROOT"/results/phase3/xai/*/tsr_shards; do
    t=$(basename "$(dirname "$d")")
    [[ "$known" == *" $t "* ]] || printf 'NOTE: results/phase3/xai/%s/tsr_shards exists but %s is NOT in this script -- add its run_id to FOLDS to cover it.\n' "$t" "$t"
  done
fi

printf 'Summary: %d/%d shards present, %d missing across %d fold(s) resubmitted.\n' \
  "$total_present" "$((total_present + total_missing))" "$total_missing" "$folds_resubmitted"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "This was a DRY RUN. Re-run without DRY_RUN=1 to submit the above."
elif [[ $folds_resubmitted -gt 0 ]]; then
  echo "Watch:  squeue -u \$USER    (each fold's merge fires automatically on afterok)"
  echo "Re-run this script anytime to reconfirm coverage; it only fills remaining gaps."
else
  echo "All folds complete -- nothing to resubmit."
fi
