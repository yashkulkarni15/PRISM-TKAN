#!/bin/bash
# One-shot transfer of the 2026-05-29 pre-Phase-3 hardening delta to Great Lakes.
# Run from /Users/yash/Downloads/PRI2M/SINDy_DDE/.
#
# Great Lakes requires Duo 2FA on every new SSH connection. To avoid ~22 Duo
# prompts (one per rsync), this opens ONE master connection (single Duo), then
# multiplexes every rsync/ssh over it via ControlMaster/ControlPath. You approve
# Duo exactly once at the start. Per-file rsyncs are kept because macOS
# openrsync's --relative is buggy.

set -euo pipefail

REMOTE=yashkulk@greatlakes.arc-ts.umich.edu
DEST=/nfs/turbo/coe-jaabell/PRI2M
CTRL="$HOME/.ssh/cm-pri2m.sock"

# Tear the master down on exit (success or failure).
cleanup() { ssh -o ControlPath="$CTRL" -O exit "$REMOTE" 2>/dev/null || true; }
trap cleanup EXIT

echo ">>> Opening master SSH connection to Great Lakes."
echo ">>> Approve the Duo 2FA prompt ONCE; all transfers reuse this connection."
ssh -fN -M -o ControlPath="$CTRL" -o ControlPersist=15m "$REMOTE"
echo ">>> Master connection established. No further 2FA needed."

# All subsequent ssh/rsync reuse the master (ControlPath), so no extra Duo prompts.
RSH="ssh -o ControlPath=$CTRL"

ssh -o ControlPath="$CTRL" "$REMOTE" "mkdir -p $DEST/src/pri2m/noise $DEST/scripts $DEST/slurm $DEST/tests"

# --- source ---
rsync -av -e "$RSH" src/pri2m/noise/metrics.py                  "$REMOTE:$DEST/src/pri2m/noise/"
rsync -av -e "$RSH" src/pri2m/noise/modes.py                    "$REMOTE:$DEST/src/pri2m/noise/"

# --- scripts (modified) ---
rsync -av -e "$RSH" scripts/extract_phase2_signals.py           "$REMOTE:$DEST/scripts/"
rsync -av -e "$RSH" scripts/calibrate_noise_mirror_pairs.py     "$REMOTE:$DEST/scripts/"
rsync -av -e "$RSH" scripts/pareto_front.py                     "$REMOTE:$DEST/scripts/"
rsync -av -e "$RSH" scripts/assign_signal_modes.py              "$REMOTE:$DEST/scripts/"

# --- scripts (new) ---
rsync -av -e "$RSH" scripts/plot_phase2_5_meeting_figures.py    "$REMOTE:$DEST/scripts/"
rsync -av -e "$RSH" scripts/diagnose_clean_false_positives.py   "$REMOTE:$DEST/scripts/"

# --- slurm wrappers (new: noisy 61-120 PSD/JS) ---
rsync -av -e "$RSH" slurm/phase2_run_psd_js_noisy_60.sbatch     "$REMOTE:$DEST/slurm/"
rsync -av -e "$RSH" slurm/phase2_aggregate_js_noisy_60.sbatch   "$REMOTE:$DEST/slurm/"

# --- tests (only needed if running pytest on GL) ---
rsync -av -e "$RSH" tests/test_extract_phase2_signals.py        "$REMOTE:$DEST/tests/"
rsync -av -e "$RSH" tests/test_assign_signal_modes.py           "$REMOTE:$DEST/tests/"
rsync -av -e "$RSH" tests/test_meeting_figures.py               "$REMOTE:$DEST/tests/"
rsync -av -e "$RSH" tests/test_clean_fp_diagnostic.py           "$REMOTE:$DEST/tests/"
rsync -av -e "$RSH" tests/test_noise_metrics.py                 "$REMOTE:$DEST/tests/"
rsync -av -e "$RSH" tests/test_noise_modes.py                   "$REMOTE:$DEST/tests/"
rsync -av -e "$RSH" tests/test_noise_calibration.py             "$REMOTE:$DEST/tests/"
rsync -av -e "$RSH" tests/test_pareto_front.py                  "$REMOTE:$DEST/tests/"

# --- control files ---
rsync -av -e "$RSH" .gitignore                                  "$REMOTE:$DEST/"
rsync -av -e "$RSH" handoff.md                                  "$REMOTE:$DEST/"

echo
echo "Delta transferred over a single authenticated connection. No pip install needed."
echo "Next on GL:  cd $DEST && sbatch slurm/phase2_run_psd_js_noisy_60.sbatch && sbatch slurm/phase2_aggregate_js_noisy_60.sbatch"
