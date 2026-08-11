#!/bin/bash
# One-shot transfer of the Phase 3 T-KAN build to Great Lakes.
# Run from /Users/yash/Downloads/PRI2M/SINDy_DDE/.
#
# Great Lakes requires Duo 2FA on every new SSH connection. To avoid one Duo
# prompt per rsync, this opens ONE master connection (single Duo), then
# multiplexes every rsync/ssh over it via ControlMaster/ControlPath. You approve
# Duo exactly once at the start.
#
# Unlike the older per-file sync, this mirrors whole package directories
# (recursively, excluding __pycache__). Recursive rsync is safe on macOS
# openrsync; only the --relative (-R) flag is buggy, and we don't use it.
#
# Scope: TKAN only. The sigKAN trunk is off by default and its sole extra dep
# (iisignature) is NOT installed below — see the optional note at the end.

set -euo pipefail

# The round-robin name lands on whichever of gl-login1/2/3 answers, and a loaded
# one refuses with "Unable to create login session" *after* the master connects.
# Override to pin a specific node and retry without editing this file:
#   REMOTE=yashkulk@gl-login2.arc-ts.umich.edu ./sync_phase3_tkan_to_hpc.sh
REMOTE="${REMOTE:-yashkulk@greatlakes.arc-ts.umich.edu}"
DEST=/nfs/turbo/coe-jaabell/PRI2M           # live GL project root (NOT .../PRI2M/SINDy_DDE)
# Socket name carries the host, so pinning a different login node cannot collide
# with a master still persisting against the previous one.
CTRL="$HOME/.ssh/cm-pri2m-${REMOTE##*@}.sock"

# --- what gets transferred -------------------------------------------------- #
# Grouped into ONE rsync per destination directory rather than one per file.
# rsync already skips unchanged files, so the old cost was never re-transferred
# data -- it was ~130 separate remote rsync processes, each paying its own
# round trip over the shared SSH master. Grouping cuts that to ~13 while moving
# exactly the same set. `--files-from` would be tidier still, but this is
# openrsync (protocol 29) where the -R path machinery it relies on is unreliable,
# and multiple-sources-to-one-destination is plain, old, safe rsync usage.
#
# Set DRY_RUN=1 to list every path that would be sent, transfer nothing, and
# skip the connection entirely -- the coverage audit this project has needed
# twice, after test files turned out never to have been in the list at all.

# Recursive package mirrors (each is already one call for a whole tree).
SRC_DIRS=(
  src/pri2m/data src/pri2m/models src/pri2m/training
  src/pri2m/xai            # includes comb.py, activation_modes.py, degradation.py
  src/pri2m/phase3         # splits.assert_no_mirror_leakage dep
  src/pri2m/analysis       # sindy_dde.py (Update 9 independent delay recovery)
)

# Top-level package modules pulled in by Phase 3 imports.
PKG_FILES=(src/pri2m/__init__.py src/pri2m/config.py)

SCRIPT_FILES=(
  # core training / eval
  scripts/train_phase3.py scripts/xai_phase3.py
  scripts/build_phase3_dataset.py            # builds dataset.csv + folds.json
  scripts/check_phase3_gate.py
  scripts/_probe_tkan_epoch.py               # bottleneck probe (data-vs-GPU split)
  scripts/extract_phase2_signals.py          # DATA_VERSION-parametrized H5->NPZ
  scripts/build_features_121_155.py          # regen physics metadata CSV for 121-155
  scripts/build_parent_groups.py             # generates data/metadata/parent_groups.json
  scripts/eval_heldout_phase3.py             # score a checkpoint on held-out data
  # white-box + analysis runners (eval-only on the trained checkpoints)
  scripts/whitebox_phase3.py                 # 7-module white box -> manifest + figures
  scripts/merge_tsr_shards.py                # reassemble sharded TSR
  scripts/tsr_control_phase3.py              # U12-B: learned delay-lock or constructed?
  scripts/render_whitebox_report.py          # manifest -> whitebox_analysis.md
  scripts/plot_whitebox_summary.py
  scripts/plot_whitebox_replication.py
  scripts/sindy_dde_crosscheck.py            # independent delay recovery (no net)
  scripts/regenerative_phase3.py             # U13-C: sum or difference in the splines?
  scripts/activation_modes_phase3.py         # U14-B: shared activation shapes?
  scripts/repeatability_audit.py             # run-to-run floor, reads summary.json only
  scripts/canonicalise_runs.py               # which duplicate run dir is canonical
  scripts/repair_counterfactual_manifest.py  # offline U13-B comparison repair
  scripts/comb_phase3.py                     # tooth-pass comb reading (no net needed)
  scripts/comb_probe_phase3.py               # swept comb confirmation (CPU-only)
  scripts/mode_ablation_phase3.py            # U14-B step 8: causal mode ablation
  scripts/stability_lobe_phase3.py           # model-implied stability lobe diagram
  scripts/counterfactuals_phase3.py          # physics-constrained counterfactuals
  scripts/concept_probes_phase3.py           # TCAV-style hidden-state probes
  scripts/conformal_phase3.py                # split-conformal calibrated uncertainty
  scripts/hidden_dynamics_phase3.py          # recurrent-Jacobian eigenvalues
  scripts/fuzz_phase3.py                     # graded degradation suite
  scripts/predict_windows_phase3.py          # per-file network probabilities
  scripts/plot_fuzz_replication.py           # cross-fold fuzz overlay
)

SLURM_FILES=(
  slurm/phase3_train.sbatch
  slurm/phase3_train_gpu.sbatch               # GPU + per-fold job array
  slurm/phase3_eval_heldout.sbatch            # held-out eval sweep, no retrain
  slurm/phase2_extract_signals.sbatch         # DATA_VERSION-aware H5->NPZ array
  slurm/phase3_whitebox.sbatch                # single-fold white box
  slurm/run_whitebox_all_folds.sh
  slurm/phase3_whitebox_tsr_array.sbatch      # sharded TSR array worker
  slurm/phase3_whitebox_merge.sbatch          # fold TSR shards back in
  slurm/phase3_tsr_control.sbatch             # U12-B control
  slurm/run_whitebox_sharded.sh
  slurm/phase3_sindy_dde.sbatch
  slurm/phase3_stability_lobe.sbatch
  slurm/phase3_counterfactuals.sbatch
  slurm/phase3_concept_probes.sbatch
  slurm/phase3_conformal.sbatch
  slurm/phase3_hidden_dynamics.sbatch
  slurm/phase3_fuzz.sbatch
  slurm/run_fuzz_all_folds.sh
  slurm/phase3_predict_windows.sbatch
  slurm/run_conformal_ensemble.sh
  slurm/phase3_regenerative.sbatch            # U13-C single fold
  slurm/run_regenerative_all_folds.sh         # U13-C all 5 checkpoints
  slurm/phase3_activation_modes.sbatch        # U14-B, CPU-only
  slurm/phase3_comb_probe.sbatch              # comb confirmation, `standard` partition
  slurm/phase3_mode_ablation.sbatch           # U14-B step 8, 1+3k held-out evals per stratum
)

# Builder inputs (Phase 2/2.5 tables; ~51 MB, unchanged run to run so rsync
# skips them, but they must stay listed for a first-time or repaired tree).
DATA_FILES=(
  data/metadata/features_all_datasets.csv
  data/metadata/features_noisy_datasets.csv
  data/metadata/parent_groups.json            # leakage-safe group map 1-155
  "data/metadata/Dataset descriptions.xlsx"   # static params for 121-155 builder
)

# Tests: only needed to run pytest on GL. Kept as an explicit list because
# "which tests exist on GL" has silently drifted from "which tests exist" twice.
TEST_FILES=(
  tests/conftest.py tests/_phase3_synth.py
  tests/test_phase3_synth.py tests/test_phase3_sequence.py
  tests/test_phase3_data_prep.py tests/test_phase3_splits.py
  tests/test_phase3_labels.py tests/test_phase3_weights.py
  tests/test_phase3_losses.py tests/test_phase3_metrics.py
  tests/test_kan.py tests/test_tkan.py tests/test_tkan_model.py
  tests/test_phase3_heads.py tests/test_phase3_baseline.py
  tests/test_phase3_trainer.py tests/test_xai_kan_native.py
  tests/test_xai_attribution.py tests/test_xai_counterfactuals.py
  tests/test_xai_phase3_norm.py tests/test_phase3_tkan_end_to_end.py
  tests/test_parent_groups.py tests/test_build_features_121_155.py
  tests/test_eval_heldout.py tests/test_whitebox_phase3.py
  tests/test_whitebox_tsr_shard.py tests/test_render_whitebox_report.py
  tests/test_xai_physics.py tests/test_xai_symbolic.py tests/test_sindy_dde.py
  tests/test_stability_lobe.py tests/test_xai_counterfactuals_physics.py
  tests/test_concept_probes.py tests/test_xai_conformal.py
  tests/test_xai_dynamics.py tests/test_severity_quantile.py
  tests/test_xai_sampling.py tests/test_degradation.py
  tests/test_spline_atlas.py tests/test_predict_windows_phase3.py
  tests/test_concept_probe_circularity.py tests/test_fuzz_phase3_cli.py
  tests/test_plot_fuzz_replication.py tests/test_xai_spectral.py
  tests/test_whitebox_tsr_delay_lock.py tests/test_tsr_control.py
  tests/test_cf_monte_carlo.py tests/test_xai_regenerative.py
  tests/test_counterfactuals_phase3_cli.py tests/test_activation_modes.py
  tests/test_activation_modes_runner.py tests/test_phase3_determinism.py
  tests/test_repair_counterfactual_manifest.py tests/test_canonicalise_runs.py
  tests/test_severity_mode_naming.py tests/test_comb.py
  tests/test_comb_phase3_cli.py tests/test_comb_probe_phase3_cli.py
)

# The renderer's TEMPLATE: render_whitebox_report.py reads it from docs/ by
# default, so shipping the renderer without it leaves a copy on GL that cannot
# run. Also the packaging file that adds the [phase3]/[sigkan] extras.
DOC_FILES=(docs/phase3_tkan/whitebox_analysis.md)
ROOT_FILES=(pyproject.toml)
CONFIG_FILES=(configs/phase3_tkan.yaml)
# Removed 2026-08-11: rapidminer/outputs/phase2/mode_labels.csv has never
# existed in this tree (not on disk, not in git history), and as an rsync source
# it exited 23 -- which under `set -e` aborted the sync at that line and silently
# skipped every transfer after it, including all 59 test files. Nothing in the
# Phase 3 pipeline reads it; the only reference is a Phase 2 diagnostic that
# defaults to results/pri2m_pooled_v2/mode_labels.csv instead. Restore by
# listing it here if a Phase 2 RapidMiner rerun ever regenerates it.
# (no MISC_FILES array: macOS ships bash 3.2, where `set -u` treats an EMPTY
# array expansion as an unbound variable and aborts. Never leave one empty --
# delete the array and its transfer line together.)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo ">>> DRY_RUN: files this sync would send (no connection opened)."
  for d in "${SRC_DIRS[@]}"; do echo "  $d/  (recursive mirror)"; done
  for f in "${PKG_FILES[@]}" "${SCRIPT_FILES[@]}" "${SLURM_FILES[@]}" \
           "${DATA_FILES[@]}" "${TEST_FILES[@]}" "${DOC_FILES[@]}" \
           "${ROOT_FILES[@]}" "${CONFIG_FILES[@]}"; do
    [[ -e "$f" ]] && echo "  $f" || echo "  $f   <-- MISSING LOCALLY"
  done
  exit 0
fi

# Tear the master down on exit (success or failure).
cleanup() { ssh -o ControlPath="$CTRL" -O exit "$REMOTE" 2>/dev/null || true; }
trap cleanup EXIT

echo ">>> Opening master SSH connection to Great Lakes."
echo ">>> Approve the Duo 2FA prompt ONCE; all transfers reuse this connection."
ssh -fN -M -o ControlPath="$CTRL" -o ControlPersist=15m "$REMOTE"

# A master socket is NOT proof the node will run anything: an overloaded login
# node authenticates and then refuses the session ("Unable to create login
# session - glN is under heavy load"), so the old script announced success and
# died on its first real command. Prove the connection can execute something
# before claiming it works.
if ! ssh -o ControlPath="$CTRL" "$REMOTE" true 2>/tmp/pri2m_ssh_probe.err; then
  echo >&2
  echo ">>> The connection authenticated but cannot run commands:" >&2
  sed 's/^/    /' /tmp/pri2m_ssh_probe.err >&2
  echo >&2
  echo ">>> Nothing was transferred. Retry against a specific login node:" >&2
  for n in gl-login1 gl-login2 gl-login3; do
    echo "      REMOTE=${REMOTE%%@*}@$n.arc-ts.umich.edu $0" >&2
  done
  exit 1
fi
echo ">>> Master connection established and verified. No further 2FA needed."

RSH="ssh -o ControlPath=$CTRL"
# Quote -e "$RSH" so the whole "ssh -o ControlPath=..." stays one argument and
# every rsync reuses the master connection (no extra Duo prompts).
rs() { rsync -av --exclude=__pycache__ --exclude='*.pyc' -e "$RSH" "$@"; }

# Send only the paths that exist, and say so about the ones that do not.
# `set -e` plus a stale entry used to abort the whole sync mid-way, silently
# skipping every transfer AFTER the bad line -- which is how 59 test files ended
# up listed but never delivered.
rs_existing() {
  local dest="${!#}" present=() missing=() f
  for f in "${@:1:$#-1}"; do
    if [[ -e "$f" ]]; then present+=("$f"); else missing+=("$f"); fi
  done
  if ((${#missing[@]})); then
    printf ">>> SKIPPED (not present locally): %s\n" "${missing[@]}" >&2
  fi
  ((${#present[@]})) && rs "${present[@]}" "$dest"
}

# Make sure every destination directory exists.
ssh -o ControlPath="$CTRL" "$REMOTE" "mkdir -p \
  $DEST/src/pri2m/data $DEST/src/pri2m/models $DEST/src/pri2m/training \
  $DEST/src/pri2m/xai $DEST/src/pri2m/phase3 $DEST/src/pri2m/analysis \
  $DEST/scripts $DEST/configs $DEST/slurm $DEST/tests $DEST/logs \
  $DEST/docs/phase3_tkan \
  $DEST/data/metadata $DEST/rapidminer/outputs/phase2 \
  $DEST/results/figures/research_update_assets"

# --- transfers: one rsync per destination, sourced from the lists above ------ #
for d in "${SRC_DIRS[@]}"; do rs "$d/" "$REMOTE:$DEST/$d/"; done
rs_existing "${PKG_FILES[@]}"    "$REMOTE:$DEST/src/pri2m/"
rs_existing "${SCRIPT_FILES[@]}" "$REMOTE:$DEST/scripts/"
rs_existing "${SLURM_FILES[@]}"  "$REMOTE:$DEST/slurm/"
rs_existing "${DATA_FILES[@]}"   "$REMOTE:$DEST/data/metadata/"
rs_existing "${TEST_FILES[@]}"   "$REMOTE:$DEST/tests/"
rs_existing "${DOC_FILES[@]}"    "$REMOTE:$DEST/docs/phase3_tkan/"
rs_existing "${ROOT_FILES[@]}"   "$REMOTE:$DEST/"
rs_existing "${CONFIG_FILES[@]}" "$REMOTE:$DEST/configs/"

echo
echo "================================================================"
echo "Phase 3 T-KAN delta transferred over a single authenticated connection."
echo "Next on Great Lakes (copy-paste):"
echo
echo "  cd $DEST"
echo "  module load python3.11-anaconda && source /home/\$USER/envs/prism/bin/activate"
echo "  pip install -e '.[phase3]'                 # torch/mlflow/captum/shap/dice-ml/matplotlib"
echo
echo "  # (optional sanity) run the test suite:"
echo "  PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true python -m pytest -q tests/test_phase3_*.py tests/test_tkan*.py tests/test_kan.py tests/test_xai_*.py"
echo
echo "  # build the per-file table + folds (uses the 3 input CSVs just synced):"
echo "  python scripts/build_phase3_dataset.py     # -> results/phase3/{phase3_dataset.csv,phase3_folds.json}"
echo
echo "  # NPZ signals already live under phase2_signals/dataset_<id>/*.npz -> point NPZ_ROOT there."
echo "  # baseline MLP on CPU (5 folds, one job):"
echo "  env MODEL=mlp  NPZ_ROOT=\$PWD/phase2_signals FOLDS='0 1 2 3 4' sbatch --export=ALL slurm/phase3_train.sbatch"
echo "  # TKAN on GPU, 5 folds in parallel as a job array (recommended -- the 1024-step"
echo "  # recurrence is what blew past the CPU time limit):"
echo "  env MODEL=tkan NPZ_ROOT=\$PWD/phase2_signals sbatch --export=ALL slurm/phase3_train_gpu.sbatch"
echo "  #   (if jaabell0 routes GPUs through a different partition: add --partition=spgpu)"
echo
echo "  # XAI on a finished run:"
echo "  python scripts/xai_phase3.py --checkpoint results/phase3/runs/<run>/checkpoints/best.pt --npz-root \$PWD/phase2_signals"
echo "================================================================"
