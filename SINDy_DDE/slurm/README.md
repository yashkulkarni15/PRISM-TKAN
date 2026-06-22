# PRI2M Great Lakes Run Order

Run from the repository root on Great Lakes, expected at:

```bash
cd /nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE
```

## 0. One-Time Environment

Use the existing `prism` environment from prior SINDy work. If the local package
has not been installed in that environment:

```bash
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate
pip install -e .
```

The SLURM files also set `PYTHONPATH=$PROJECT_ROOT/src`, so editable install is
recommended but not required for the scripts.

## 1. Verify Phase 1 Metadata and Altintas-Budak Scaffold

```bash
sbatch slurm/phase1_verify_altintas.sbatch
```

This is lightweight. It should pass operational checks and may warn that strict
textbook and `blim` validation still need exact reference constants.

Before treating Phase 2 curves as final, verify `b_over_blim` provenance:

```bash
python scripts/verify_b_over_blim_provenance.py \
  --summary-csv "/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/metadata/features_all_datasets.csv" \
  --data-root "/nfs/turbo/coe-jaabell/datasets/tonylschmitz/digital-machining-database/versions/10" \
  --sample-rows 240

python scripts/check_js_reference_rule.py
```

The raw-data version above is the default. Override it without editing code via
`DATA_VERSION` (default `10`) or the full `DATA_ROOT` path, e.g. for the new
datasets: `DATA_ROOT=/nfs/turbo/coe-jaabell/datasets/tonylschmitz/digital-machining-database/versions/new_121_155`.

## 2. Extract Phase 2 Signals to NPZ

Smoke-test I/O first with strict failure handling:

```bash
sbatch --array=1 \
  --export=ALL,MAX_FILES_PER_DATASET=5,STRICT=1 \
  slurm/phase2_extract_signals.sbatch
```

Pilot/dev plus anomaly set first:

```bash
clean_job=$(sbatch --parsable --array=1,4,11,12,29,34,35,39,44-48 \
  --export=ALL \
  slurm/phase2_extract_signals.sbatch)

noisy_job=$(sbatch --parsable --array=91 \
  --export=ALL,SUMMARY_CSV="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/metadata/features_noisy_datasets.csv" \
  slurm/phase2_extract_signals.sbatch)
```

The pilot arrays intentionally omit `STRICT=1`; failed individual files remain
visible in `data/phase2_signals/manifests/`, while successful files can still feed
PSD/Jensen-Shannon summaries.

Run PSD summaries after each extraction parent succeeds:

```bash
sbatch --dependency=afterok:${clean_job} \
  --export=ALL,DATASETS="1,4,11,12,29,34,35,39,44-48",INSPECT_DATASETS="11,12,34,35,44-48" \
  slurm/phase2_run_psd_js.sbatch

sbatch --dependency=afterok:${noisy_job} \
  --export=ALL,DATASETS="91",INSPECT_DATASETS="91",SUMMARY_CSV="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/metadata/features_noisy_datasets.csv",OUTPUT_DIR="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2/psd_js_noisy" \
  slurm/phase2_run_psd_js.sbatch
```

If you submit extraction with `STRICT=1`, prefer `afterany` or inspect the array
status before adding a PSD dependency.

Legacy direct commands:

```bash
sbatch --array=1,4,11,12,29,34,35,39,44-48 slurm/phase2_extract_signals.sbatch
sbatch --array=91 \
  --export=ALL,SUMMARY_CSV="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/metadata/features_noisy_datasets.csv" \
  slurm/phase2_extract_signals.sbatch
```

Full clean run:

```bash
clean60_job=$(sbatch --parsable --array=1-60 \
  --export=ALL \
  slurm/phase2_extract_signals.sbatch)
```

Noisy run, if needed for matched robustness:

```bash
sbatch --array=61-120 \
  --export=ALL,SUMMARY_CSV="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/metadata/features_noisy_datasets.csv" \
  slurm/phase2_extract_signals.sbatch
```

Outputs go to `/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/phase2_signals` by default.

## 3. Run PSD / Jensen-Shannon Summaries

Run after extraction jobs finish:

```bash
sbatch slurm/phase2_run_psd_js.sbatch
```

Full clean Phase 2.1 production run:

```bash
env \
  DATASETS='1-60' \
  INSPECT_DATASETS='11,12,34,35,44-48' \
  OUTPUT_DIR='/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2/psd_js_clean_60' \
  sbatch --dependency=afterok:${clean60_job} --export=ALL slurm/phase2_run_psd_js.sbatch
```

For a smaller pilot:

```bash
sbatch --export=ALL,DATASETS="1,4,11,12,29,34,35,39,44-48",INSPECT_DATASETS="11,12,34,35,44-48" slurm/phase2_run_psd_js.sbatch
sbatch --export=ALL,DATASETS="91",INSPECT_DATASETS="91",SUMMARY_CSV="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/metadata/features_noisy_datasets.csv",OUTPUT_DIR="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2/psd_js_noisy" slurm/phase2_run_psd_js.sbatch
```

Outputs go to `/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2/psd_js` by default.

## 4. Plot Per-Dataset PSD Profiles

Run after full clean extraction exists. This creates the Phase 2.2 artifact:
stable-vs-unstable PSD overlays per dataset, with TPF and `fn/tpf` markers.

```bash
env \
  DATASETS='1-60' \
  SIGNALS='xtpos,Forcex' \
  OUTPUT_DIR='/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2/psd_profiles_clean_60' \
  sbatch --export=ALL slurm/phase2_plot_psd_profiles.sbatch
```

Outputs:

```text
profile_summary.csv
profile_failures.csv
run_metadata.json
plots/dataset_<id>_<signal>_psd_profile.png
```

## 5. Aggregate Stable-vs-Unstable JS Curves

Run after PSD extraction exists. Use `env` for comma-containing selectors:

```bash
env \
  DATASETS='1,4,11,12,29,34,35,39,44-48' \
  OUTPUT_DIR='/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2/js_curves_clean_pilot' \
  BOOTSTRAP=200 \
  REFERENCE_MODE='low_b_ratio' \
  REFERENCE_QUANTILE=0.10 \
  REFERENCE_MAX_B_RATIO=0.80 \
  sbatch --export=ALL slurm/phase2_aggregate_js.sbatch
```

Full clean Phase 2.3/2.4 production aggregation:

```bash
env \
  DATASETS='1-60' \
  OUTPUT_DIR='/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2/js_curves_clean_60' \
  BOOTSTRAP=200 \
  REFERENCE_MODE='low_b_ratio' \
  REFERENCE_QUANTILE=0.10 \
  REFERENCE_MAX_B_RATIO=0.80 \
  sbatch --dependency=afterok:<PSD_JOB_ID> --export=ALL slurm/phase2_aggregate_js.sbatch
```

For noisy dataset 91:

```bash
env \
  DATASETS='91' \
  SUMMARY_CSV='/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/metadata/features_noisy_datasets.csv' \
  OUTPUT_DIR='/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2/js_curves_noisy' \
  BOOTSTRAP=200 \
  REFERENCE_MODE='low_b_ratio' \
  REFERENCE_QUANTILE=0.10 \
  REFERENCE_MAX_B_RATIO=0.80 \
  sbatch --export=ALL slurm/phase2_aggregate_js.sbatch
```

Outputs:

```text
js_divergence_by_b_over_blim.csv
js_divergence_vs_stable_reference_by_b_over_blim.csv
aggregation_failures.csv
run_metadata.json
js_divergence_stable_reference_<signal>.png, if matplotlib is available
js_divergence_within_bin_<signal>.png, if within-bin rows have enough mixed labels
```

Default stable reference rule: `low_b_ratio`, i.e. the lowest 10% of `b/blim`
within each `signal x teeth x cut_dir` facet, with at least `min_per_label`
samples and a hard `b/blim <= 0.80` ceiling. Facets without enough samples under
that ceiling are marked `insufficient_stable_reference`. This avoids selecting
the reference from the label column and avoids using mildly unstable points as a
"stable" reference.

## 5b. Phase 2.5 Noise Characterization and Mode Assignment

Run after Phase 2.1 (signal extraction) is complete on the dataset selector you
care about. The first checkpoint targets datasets 1, 61, and 91 (the clean / 1%
noise / 5% noise mirror ladder).

### Compute per-file noise metrics

Pilot (datasets 1, 61, 91 only):

```bash
env \
  DATASETS='1,61,91' \
  sbatch --export=ALL slurm/phase2_5_compute_noise_metrics.sbatch
```

Full clean + noisy run:

```bash
metrics_job=$(env DATASETS='1-60,61-120' \
  sbatch --parsable --export=ALL slurm/phase2_5_compute_noise_metrics.sbatch)
```

### Calibrate the noise-score gate and assign mode labels

Pilot single-ladder calibration:

```bash
sbatch --dependency=afterok:${metrics_job} --export=ALL \
  slurm/phase2_5_calibrate_mirror_ladder.sbatch
```

Production pooled calibration after `DATASETS='1-60,61-120'` metrics exist:

```bash
env \
  POOLED=1 \
  NOISE_METRICS_CSV='/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2_5/noise_metrics/noise_metrics.csv' \
  OUTPUT_DIR='/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2_5/calibration_pooled' \
  sbatch --dependency=afterok:${metrics_job} --export=ALL \
  slurm/phase2_5_calibrate_mirror_ladder.sbatch
```

Default single-ladder mode compares clean Dataset 1, 1% noisy Dataset 61, and
5% noisy Dataset 91. Override with `CLEAN_ID`, `NOISE1_ID`, `NOISE5_ID`
environment variables. Pooled mode auto-detects every `(D, D+60, D+90)` ladder
in the metrics CSV; override the offsets with `NOISE1_OFFSET` and
`NOISE5_OFFSET` only if the metadata convention changes. The calibration gate
only promotes metrics that move upward from clean to 1% to 5% noise and writes
clean reference percentiles that mode assignment consumes.

Outputs:

- `results/phase2_5/noise_metrics/noise_metrics.csv` — per-file noise/chatter metrics.
- `results/phase2_5/noise_metrics/run_metadata.json` — config and counts.
- `results/phase2_5/calibration/mirror_calibration_<clean>_vs_<n1>_vs_<n5>.csv` — per-metric ladder stats.
- `results/phase2_5/calibration_pooled/mirror_calibration_pooled.csv` — full-ladder production stats when `POOLED=1`.
- `results/phase2_5/calibration/promoted_metrics.json` — promoted metrics plus clean calibration baselines.
- `results/phase2_5/calibration/mode_labels.csv` — per-file mode label, calibrated noise score, and confidence.

Phase 2.5 must complete before the JS spectral loss is unlocked in Phase 3.

## 6. Cleanup MLflow Checkpoints After Training Runs

Dry run first:

```bash
sbatch --export=RUN_DIR=/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/mlruns/<experiment>/<run> \
  slurm/mlflow_checkpoint_cleanup.sbatch
```

Execute deletion after reviewing the log:

```bash
sbatch --export=RUN_DIR=/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/mlruns/<experiment>/<run>,EXECUTE=1 \
  slurm/mlflow_checkpoint_cleanup.sbatch
```

Retention rule: keep best-by-validation-loss checkpoint and final checkpoint,
delete intermediate periodic checkpoints.

## 7. Phase 3 TKAN Training

Run after Phase 2.5 (calibrated noise scores + mode labels) and the Phase 3
dataset/folds builder have produced the per-file table and folds JSON. The
wrapper trains a single Phase 3 model (`tkan`, `sigkan`, or `mlp`) on one or more
folds via `scripts/train_phase3.py`, writing run artifacts under
`results/phase3/runs/` and MLflow runs to a quota-safe `file:` store under
`results/phase3/mlruns`.

Single TKAN fold:

```bash
env MODEL=tkan FOLD=0 \
  sbatch --export=ALL slurm/phase3_train.sbatch
```

Several folds in one job (space-separated), MLP baseline:

```bash
env MODEL=mlp FOLDS='0 1 2' \
  sbatch --export=ALL slurm/phase3_train.sbatch
```

Fast smoke run (clamps epochs/window inside the trainer), useful to validate the
environment before a full sweep:

```bash
env SMOKE=1 MODEL=tkan FOLD=0 \
  sbatch --export=ALL slurm/phase3_train.sbatch
```

Env overrides with sensible defaults: `MODEL` (default `tkan`), `FOLD`
(default `0`), `FOLDS` (default `$FOLD`), `CONFIG`
(default `configs/phase3_tkan.yaml`), `NPZ_ROOT`, `TABLE`, `FOLDS_JSON`,
`OUTPUT_DIR` (default `results/phase3/runs`), `MLFLOW_URI`
(default `file:$PROJECT_ROOT/results/phase3/mlruns`), and `SMOKE` (default `0`).
The wrapper exports `MLFLOW_ALLOW_FILE_STORE=true`, which MLflow 3.x requires for
a `file:` tracking store.

Outputs:

- `results/phase3/runs/<run_id>/summary.json` — fold summary with `val` metrics,
  the trajectory `gate` report, config, and the checkpoint retention plan.
- `results/phase3/runs/<run_id>/checkpoints/{best,final}.pt` — retained
  checkpoints (periodic per-epoch checkpoints are pruned).
- `results/phase3/mlruns/` — MLflow `file:` store with per-epoch and final
  metrics, params, and the logged `summary.json` artifact.

Prune leftover periodic checkpoints from a finished run with the
`mlflow_checkpoint_cleanup.sbatch` wrapper described in Section 6.
