#!/bin/bash
# tidy_pri2m.sh — group the PRI2M Great Lakes root WITHOUT touching the live
# Phase 3 pipeline (or any running job that uses it).
#
# Run from the GL project root:
#   cd /nfs/turbo/coe-jaabell/PRI2M
#   bash scripts/tidy_pri2m.sh            # DRY RUN: prints what it would do
#   bash scripts/tidy_pri2m.sh --apply    # actually move things
#
# DELIBERATELY LEFT UNTOUCHED (pipeline + running-job critical):
#   src/ scripts/ configs/ slurm/ tests/ pyproject.toml   (code; sbatch/sync hardcode these)
#   results/ phase2_signals/ logs/ data/ rapidminer/       (phase3 reads/writes these live)
#   README.md                                              (kept at root)
#
# Everything moved below is Phase 2 / 2.5 *output* or stale material the Phase 3
# job never opens, so moving it is safe even mid-run. Still: prefer to run this
# when `squeue -u $USER` is empty.

set -euo pipefail
ROOT="${ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
cd "$ROOT"

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1
run() { if [[ $APPLY -eq 1 ]]; then eval "$@"; else echo "DRYRUN: $*"; fi; }
mv_if() { [[ -e "$1" ]] && run "mv -- '$1' '$2/'" || true; }

echo "== tidy PRI2M root ($([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)) =="

# 0. delete the 0-byte junk file from a past malformed sbatch line
[[ -e '--export=ALL' ]] && run "rm -f -- '--export=ALL'" || true

# 1. group Phase 2 OUTPUT products under outputs/phase2
run "mkdir -p outputs/phase2"
for d in phase2_psd_js phase2_psd_js_noisy phase2_psd_js_clean_pilot \
         phase2_psd_js_clean_60 phase2_psd_js_noisy_60 \
         phase2_js_curves_noisy phase2_js_curves_clean_pilot \
         phase2_js_curves_clean_pilot_v2 phase2_js_curves_noisy_v2 \
         phase2_js_curves_clean_60 phase2_js_curves_noisy_60 \
         phase2_psd_profiles_clean_60 eda_plots; do
  mv_if "$d" outputs/phase2
done

# 2. group Phase 2.5 products under outputs/phase2_5
run "mkdir -p outputs/phase2_5"
for d in phase2_5_results phase2_5_noise_metrics phase2_5_calibration \
         phase2_5_calibration_1_61_91 phase2_5_noise_metrics_1_61_91 \
         phase2_5_noise_metrics_full phase2_5_calibration_pooled \
         phase2_5_calibration_pooled_v2 phase2_5_forcey_diagnostic; do
  mv_if "$d" outputs/phase2_5
done

# 3. docs + notebooks + stale copies
run "mkdir -p docs notebooks archive"
mv_if research_update.md docs
mv_if handoff.md docs
for f in *.ipynb; do mv_if "$f" notebooks; done
# SINDy_DDE is the stale duplicate tree (live root is PRI2M itself, not this subdir).
mv_if SINDy_DDE archive

echo "== done. Review, then re-run with --apply if this was a dry run. =="
echo "LEFT AT ROOT on purpose (data file + stray sbatch; move by hand only if you're sure):"
echo "   features_noisy_datasets.csv   phase2_extract_signals.sbatch"
