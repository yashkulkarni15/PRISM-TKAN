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

REMOTE=yashkulk@greatlakes.arc-ts.umich.edu
DEST=/nfs/turbo/coe-jaabell/PRI2M           # live GL project root (NOT .../PRI2M/SINDy_DDE)
CTRL="$HOME/.ssh/cm-pri2m.sock"

# Tear the master down on exit (success or failure).
cleanup() { ssh -o ControlPath="$CTRL" -O exit "$REMOTE" 2>/dev/null || true; }
trap cleanup EXIT

echo ">>> Opening master SSH connection to Great Lakes."
echo ">>> Approve the Duo 2FA prompt ONCE; all transfers reuse this connection."
ssh -fN -M -o ControlPath="$CTRL" -o ControlPersist=15m "$REMOTE"
echo ">>> Master connection established. No further 2FA needed."

RSH="ssh -o ControlPath=$CTRL"
# Quote -e "$RSH" so the whole "ssh -o ControlPath=..." stays one argument and
# every rsync reuses the master connection (no extra Duo prompts).
rs() { rsync -av --exclude=__pycache__ --exclude='*.pyc' -e "$RSH" "$@"; }

# Make sure every destination directory exists.
ssh -o ControlPath="$CTRL" "$REMOTE" "mkdir -p \
  $DEST/src/pri2m/data $DEST/src/pri2m/models $DEST/src/pri2m/training \
  $DEST/src/pri2m/xai $DEST/src/pri2m/phase3 $DEST/src/pri2m/analysis \
  $DEST/scripts $DEST/configs $DEST/slurm $DEST/tests $DEST/logs \
  $DEST/docs/phase3_tkan \
  $DEST/data/metadata $DEST/rapidminer/outputs/phase2 \
  $DEST/results/figures/research_update_assets"

# --- source packages (recursive mirror) ---
rs src/pri2m/data/      "$REMOTE:$DEST/src/pri2m/data/"
rs src/pri2m/models/    "$REMOTE:$DEST/src/pri2m/models/"
rs src/pri2m/training/  "$REMOTE:$DEST/src/pri2m/training/"
rs src/pri2m/xai/       "$REMOTE:$DEST/src/pri2m/xai/"
rs src/pri2m/phase3/    "$REMOTE:$DEST/src/pri2m/phase3/"   # splits.assert_no_mirror_leakage dep
rs src/pri2m/analysis/  "$REMOTE:$DEST/src/pri2m/analysis/" # sindy_dde.py (Update 9 independent delay recovery)

# --- top-level package modules pulled in by Phase 3 imports ---
rs src/pri2m/__init__.py "$REMOTE:$DEST/src/pri2m/"
rs src/pri2m/config.py   "$REMOTE:$DEST/src/pri2m/"

# --- scripts ---
rs scripts/train_phase3.py        "$REMOTE:$DEST/scripts/"
rs scripts/xai_phase3.py          "$REMOTE:$DEST/scripts/"
rs scripts/build_phase3_dataset.py "$REMOTE:$DEST/scripts/"   # builds dataset.csv + folds.json
rs scripts/check_phase3_gate.py   "$REMOTE:$DEST/scripts/"
rs scripts/_probe_tkan_epoch.py   "$REMOTE:$DEST/scripts/"   # bottleneck probe (data-vs-GPU split)
rs scripts/extract_phase2_signals.py "$REMOTE:$DEST/scripts/"  # DATA_VERSION-parametrized H5->NPZ
rs scripts/build_features_121_155.py  "$REMOTE:$DEST/scripts/"  # regen physics metadata CSV for 121-155
rs scripts/build_parent_groups.py     "$REMOTE:$DEST/scripts/"  # generates data/metadata/parent_groups.json
rs scripts/eval_heldout_phase3.py     "$REMOTE:$DEST/scripts/"  # score a checkpoint on held-out datasets (no retrain)

# --- white-box + Update-9 analysis runners (eval-only on the trained checkpoints) ---
rs scripts/whitebox_phase3.py         "$REMOTE:$DEST/scripts/"  # 7-module white box -> manifest + figures
rs scripts/merge_tsr_shards.py        "$REMOTE:$DEST/scripts/"  # reassemble sharded TSR -> results["tsr"]
rs scripts/tsr_control_phase3.py      "$REMOTE:$DEST/scripts/"  # U12-B: is the delay-lock learned or constructed?
rs scripts/render_whitebox_report.py  "$REMOTE:$DEST/scripts/"  # manifest -> whitebox_analysis.md
# ...and its TEMPLATE, which render_whitebox_report.py reads from docs/ by default.
# The documented flow renders locally after rsync-back, but shipping the renderer
# without its template leaves a copy on GL that cannot run.
rs docs/phase3_tkan/whitebox_analysis.md "$REMOTE:$DEST/docs/phase3_tkan/"
rs scripts/plot_whitebox_summary.py   "$REMOTE:$DEST/scripts/"  # per-fold summary figures
rs scripts/plot_whitebox_replication.py "$REMOTE:$DEST/scripts/" # cross-fold replication figure
rs scripts/sindy_dde_crosscheck.py    "$REMOTE:$DEST/scripts/"  # independent delay recovery (no net, no pysindy)
rs scripts/regenerative_phase3.py     "$REMOTE:$DEST/scripts/"  # U13-C: is x(t)-x(t-tau) in the learned splines?
rs scripts/activation_modes_phase3.py  "$REMOTE:$DEST/scripts/"  # U14-B: do folds share learned activation shapes?
rs scripts/repeatability_audit.py     "$REMOTE:$DEST/scripts/"  # run-to-run floor, reads summary.json only
rs scripts/stability_lobe_phase3.py   "$REMOTE:$DEST/scripts/"  # model-implied stability lobe diagram
rs scripts/counterfactuals_phase3.py  "$REMOTE:$DEST/scripts/"  # physics-constrained counterfactuals
rs scripts/concept_probes_phase3.py   "$REMOTE:$DEST/scripts/"  # TCAV-style hidden-state concept probes
rs scripts/conformal_phase3.py        "$REMOTE:$DEST/scripts/"  # split-conformal calibrated uncertainty
rs scripts/hidden_dynamics_phase3.py  "$REMOTE:$DEST/scripts/"  # recurrent-Jacobian eigenvalue analysis
rs scripts/fuzz_phase3.py             "$REMOTE:$DEST/scripts/"  # graded input/model degradation suite
rs scripts/predict_windows_phase3.py  "$REMOTE:$DEST/scripts/"  # per-file network probabilities
rs scripts/plot_fuzz_replication.py   "$REMOTE:$DEST/scripts/"  # cross-fold fuzz overlay

# --- config + slurm wrapper ---
rs configs/phase3_tkan.yaml       "$REMOTE:$DEST/configs/"
rs slurm/phase3_train.sbatch      "$REMOTE:$DEST/slurm/"
rs slurm/phase3_train_gpu.sbatch  "$REMOTE:$DEST/slurm/"   # GPU + per-fold job array
rs slurm/phase3_eval_heldout.sbatch "$REMOTE:$DEST/slurm/" # held-out eval sweep (5 folds x eval sets, no retrain)
rs slurm/phase2_extract_signals.sbatch "$REMOTE:$DEST/slurm/"   # DATA_VERSION-aware H5->NPZ array
# Update-9 eval-only analysis wrappers (self-contained; default to fold-2 run a654129d):
rs slurm/phase3_whitebox.sbatch        "$REMOTE:$DEST/slurm/"  # single-fold white box (per-window tau sweep + TSR + symbolic gate)
rs slurm/run_whitebox_all_folds.sh     "$REMOTE:$DEST/slurm/"  # re-run the white box on all 5 checkpoints (inline TSR)
rs slurm/phase3_whitebox_tsr_array.sbatch "$REMOTE:$DEST/slurm/"  # sharded TSR array worker (--tsr-only)
rs slurm/phase3_whitebox_merge.sbatch  "$REMOTE:$DEST/slurm/"  # fold TSR shards back into the manifest
rs slurm/phase3_tsr_control.sbatch     "$REMOTE:$DEST/slurm/"  # U12-B control (cheap: 1 fold, ~24 windows)
rs slurm/run_whitebox_sharded.sh       "$REMOTE:$DEST/slurm/"  # sharded workflow DAG: fixed run + TSR array + merge, all 5 folds
rs slurm/phase3_sindy_dde.sbatch       "$REMOTE:$DEST/slurm/"
rs slurm/phase3_stability_lobe.sbatch  "$REMOTE:$DEST/slurm/"
rs slurm/phase3_counterfactuals.sbatch "$REMOTE:$DEST/slurm/"
rs slurm/phase3_concept_probes.sbatch  "$REMOTE:$DEST/slurm/"
rs slurm/phase3_conformal.sbatch       "$REMOTE:$DEST/slurm/"
rs slurm/phase3_hidden_dynamics.sbatch "$REMOTE:$DEST/slurm/"
rs slurm/phase3_fuzz.sbatch            "$REMOTE:$DEST/slurm/"
rs slurm/run_fuzz_all_folds.sh         "$REMOTE:$DEST/slurm/"
rs slurm/phase3_predict_windows.sbatch "$REMOTE:$DEST/slurm/"
rs slurm/run_conformal_ensemble.sh     "$REMOTE:$DEST/slurm/"
rs slurm/phase3_regenerative.sbatch    "$REMOTE:$DEST/slurm/"  # U13-C: regenerative spline test (single fold)
rs slurm/run_regenerative_all_folds.sh "$REMOTE:$DEST/slurm/"  # U13-C across all 5 checkpoints
rs slurm/phase3_activation_modes.sbatch "$REMOTE:$DEST/slurm/"  # U14-B: CPU-only, 5 folds, k-sweep + untrained null

# --- packaging (adds the [phase3] / [sigkan] extras) ---
rs pyproject.toml                 "$REMOTE:$DEST/"

# --- builder inputs (Phase 2/2.5 tables; ~51 MB, not in earlier code syncs) ---
rs data/metadata/features_all_datasets.csv    "$REMOTE:$DEST/data/metadata/"
rs data/metadata/features_noisy_datasets.csv  "$REMOTE:$DEST/data/metadata/"
rs data/metadata/parent_groups.json           "$REMOTE:$DEST/data/metadata/"   # leakage-safe group map 1-155
rs "data/metadata/Dataset descriptions.xlsx"  "$REMOTE:$DEST/data/metadata/"   # static params for build_features_121_155
rs rapidminer/outputs/phase2/mode_labels.csv  "$REMOTE:$DEST/rapidminer/outputs/phase2/"

# --- tests (only needed to run pytest on GL) ---
rs tests/conftest.py              "$REMOTE:$DEST/tests/"
rs tests/_phase3_synth.py         "$REMOTE:$DEST/tests/"
for t in \
  test_phase3_synth test_phase3_sequence test_phase3_data_prep test_phase3_splits \
  test_phase3_labels test_phase3_weights test_phase3_losses test_phase3_metrics \
  test_kan test_tkan test_tkan_model test_phase3_heads test_phase3_baseline \
  test_phase3_trainer test_xai_kan_native test_xai_attribution \
  test_xai_counterfactuals test_xai_phase3_norm test_phase3_tkan_end_to_end \
  test_parent_groups test_build_features_121_155 test_eval_heldout \
  test_whitebox_phase3 test_whitebox_tsr_shard test_render_whitebox_report test_xai_physics \
  test_xai_symbolic test_sindy_dde test_stability_lobe \
  test_xai_counterfactuals_physics test_concept_probes test_xai_conformal \
  test_xai_dynamics test_severity_quantile \
  test_xai_sampling test_degradation test_spline_atlas test_predict_windows_phase3 \
  test_concept_probe_circularity test_fuzz_phase3_cli test_plot_fuzz_replication \
  test_xai_spectral test_whitebox_tsr_delay_lock test_tsr_control \
  test_cf_monte_carlo test_xai_regenerative test_counterfactuals_phase3_cli \
  test_activation_modes ; do
  rs "tests/$t.py" "$REMOTE:$DEST/tests/"
done

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
