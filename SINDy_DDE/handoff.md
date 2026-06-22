# PRI2M Handoff

Last updated: 2026-06-22 (Phase 3 TKAN held-out eval on 121–125 + first git commit — see the 2026-06-22 section at the bottom; the live source of truth for TKAN work is `docs/phase3_tkan/PROGRESS.md`)

May 18 note: user showed Great Lakes logs containing aggregate JS jobs `50409936` and `50409964`; both `.err` files are zero bytes. Clean job `50409936` processed 5,147 files / 20,588 signal rows with zero failures, but `ok_divergence_rows=0` because within-bin stable-vs-unstable grouping was too strict. The aggregation script is being updated to also emit bin-vs-stable-reference JS curves.

May 18 v2 aggregate result: clean `phase2_js_curves_clean_pilot_v2` processed 20,588 signal rows with zero failures. Stable-reference JS curve output has 116 rows / 112 usable rows (`ok_stable_reference_rows=112`), written to `js_divergence_vs_stable_reference_by_b_over_blim.csv` with 117 CSV lines including header. Plot generation succeeded for all four signals: `Forcex`, `Forcey`, `xtpos`, `ytpos`. Strict within-bin plot remains unavailable as expected because within-bin mixed-label groups were insufficient.

May 18 noisy v2 aggregate result: `phase2_js_curves_noisy_v2` processed 102 files / 408 signal rows with zero failures. Stable-reference JS curve output has 28 rows / 28 usable rows (`ok_stable_reference_rows=28`), written to `js_divergence_vs_stable_reference_by_b_over_blim.csv` with 29 CSV lines including header. Plot generation succeeded for all four signals: `Forcex`, `Forcey`, `xtpos`, `ytpos`. Phase 2 pilot extraction, PSD summaries, and stable-reference JS curves are complete for clean pilot and noisy dataset 91.

May 18 second-opinion review folded in:
- Engineering is sound for extraction, PSD, pilot JS, and Great Lakes operations.
- Current week is not fully complete against schedule: Phase 2.1 requires PSD over all 60 clean datasets by May 19; current work is a biased pilot over 13 clean datasets plus noisy dataset 91.
- Phase 2.2 is at risk because the scheduled artifact is per-dataset stable-vs-unstable PSD profiles with TPF/chatter annotations, not only aggregate sanity flags.
- The `blim` verifier is the major physics-anchor risk for Phase 3 residual physics; it is not a Phase 2 blocker because Phase 2 uses the metadata `b_over_blim`, but it must be owned before Phase 3 stage 2.
- Stable-reference JS is defensible for the Phase 3 spectral-loss candidate, but the acceptance criterion changes from "peak near b/blim=1" to "low/flat below the boundary with rise or inflection across b/blim=1".
- Stable-reference selection must be a physical rule, not label leakage. The aggregator now defaults to `REFERENCE_MODE=low_b_ratio`, selecting the lowest 10% `b/blim` samples within each `signal x teeth x cut_dir` facet.
- Gap A patched: `REFERENCE_MAX_B_RATIO=0.80` is now enforced for the stable reference. Facets with fewer than `min_per_label` samples below this physical floor are marked `insufficient_stable_reference`.
- Gap B documented: the lowest `b/blim` bin overlaps the reference set and should be treated as an anchor, not an independent validation point.
- `b_over_blim` provenance checked locally:
  - Clean/noisy feature CSVs satisfy `b_over_blim = b_mm / blim` to floating-point tolerance.
  - Labels satisfy `label = int(b_over_blim >= 1.0)` with zero mismatches.
  - SINDy preprocessing lineage reads `stability_boundary<dataset>.h5` and interpolates `blim` at file `omega`.
  - Added `scripts/verify_b_over_blim_provenance.py` so Great Lakes can spot-check feature CSV values against raw stability-boundary H5 files before final full-60 interpretation.
- Research update PDF reviewed on May 18:
  - Figures are readable and support the stable-reference narrative.
  - Before presenting, clarify that shown pilot v2 curves predate the `b/blim <= 0.80` reference-floor production patch unless those curves are regenerated with the patched aggregator.
  - Update the `b_over_blim` risk from "being checked" to "local CSV/SINDy lineage verified; raw Great Lakes H5 spot-check pending or queued."
- Updated `Research Update - 2.pdf` reviewed:
  - The earlier floor-patch caveat and `b_over_blim` provenance wording are now fixed.
  - Remaining presentation caveat: the PDF still says the full-60 run is "currently running"; update that if the recovered downstream jobs finish before the meeting.
  - Minor visible issue remains in the Work Log table extraction/rendering: "Task" appears as "T ask" in extracted text.
- Full-60 Great Lakes run status reported May 18:
  - Array `50411080` partially failed, so dependent jobs `50411111` PSD/JS, `50411113` profile plots, and `50411114` aggregate JS were cancelled.
  - Successful NPZ count is currently 17,044, with 53 manifests for datasets 1-60.
  - Failed extraction tasks include datasets `13`, `14`, `15`, `19`, `20`, and `42-48`; next action is to inspect logs/manifests, rerun only missing/failed datasets, then submit downstream jobs manually.
- Full-60 extraction recovery succeeded:
  - Recovery job `50411725` completed for missing datasets `13`, `14`, `15`, `19`, `20`, `42`, and `43`.
  - Full clean extraction now has 60 manifests, no manifest failures from `grep -R ",failed,"`, and 18,771 NPZ files.
  - Downstream jobs submitted manually: `50411755` PSD summary, `50411756` Phase 2.2 PSD profile plots, and `50411757` stable-reference JS aggregation with `REFERENCE_MAX_B_RATIO=0.80`.
- Full-60 downstream outputs completed:
  - `phase2_psd_js_clean_60` contains the full PSD summary (`psd_summary.csv`, 35 MB) and feature sanity CSV.
  - `phase2_psd_profiles_clean_60` generated 120 PNG profile plots with `failure_rows=0`.
  - `phase2_js_curves_clean_60` generated stable-reference JS CSV/PNGs for `Forcex`, `Forcey`, `xtpos`, and `ytpos`.
  - JS aggregation metadata: `processed_files=18221`, `processed_signal_rows=72884`, `stable_reference_rows=156`, `ok_stable_reference_rows=152`, `failure_rows=0`, `reference_max_b_ratio=0.8`.
  - CSV row counts: `js_divergence_by_b_over_blim.csv` 157 lines, `js_divergence_vs_stable_reference_by_b_over_blim.csv` 157 lines, `aggregation_failures.csv` header only.
- Great Lakes guard checks:
  - `scripts/check_js_reference_rule.py` passed.
  - `scripts/verify_b_over_blim_provenance.py` confirmed clean CSV internal consistency (`b_over_blim = b_mm / blim`, max formula error ~2e-13, zero label mismatches), but returned `ok=false` because raw `stability_boundary<dataset>.h5` spot-check found 12 mismatches in 240 sampled rows.
  - Mismatch preview included datasets `43`, `51`, `52`, and `54`; next step is to inspect whether the raw boundary arrays need sorting/different key handling or whether `features_all_datasets.csv` uses a different official metadata source than the raw H5 boundary file.
- Raw boundary diagnostic result:
  - Preview mismatches are caused by CSV row `omega` values below the raw boundary H5 range, e.g. `omega=1000` or `1500` while `stability_boundary*.h5` starts at `2000`.
  - `np.interp` clamps those out-of-range values to the first boundary row, so the previous checker reported false raw-boundary mismatches.
  - Patched `scripts/verify_b_over_blim_provenance.py` to sort boundary arrays and report out-of-boundary-range rows separately instead of counting them as interpolation mismatches.
- Patched Great Lakes provenance check passed:
  - `ok=true`, zero raw-boundary mismatches for in-range rows, zero label mismatches, max `b_over_blim` formula error ~2e-13.
  - `out_of_boundary_range=12` in the 240-row sample; these are low-speed rows below the raw stability-boundary omega grid and are now documented separately.
  - Meeting phrasing: `b_over_blim` provenance is resolved for CSV formula, label rule, and in-range raw H5 spot checks; caveat is that low-speed rows below the H5 boundary grid cannot be verified by direct interpolation.
- Local evaluation of copied full-60 outputs:
  - `phase2_js_curves_clean_60` copied locally and inspected. Stable-reference JS is usable: 152/156 rows OK, 4 rows `insufficient_bin_count` in the sparse `teeth=5, cut_dir=2, [2,3)` bin.
  - All strict within-bin JS rows remain `insufficient_label_counts`, confirming the pivot away from within-bin stable-vs-unstable divergence.
  - Full-60 clean curves support the revised criterion: JS is low below `b/blim=1` and rises after the boundary across Forcex, Forcey, xtpos, and ytpos.
  - Force signals are strongest overall; tool-position signals are weaker but still physically consistent. Dominant 4-tooth climb facet: Forcex rises from ~0.0003-0.0134 below 1 to 0.4184 in the `>=5` bin; xtpos rises from ~0.0001-0.0019 below 1 to 0.0975 in the `>=5` bin.
  - `phase2_psd_profiles_clean_60` copied locally: 120 plots, no profile failures. Profile summaries show dominant peaks often remain near TPF for both stable and unstable groups, so the useful signal is distributional divergence/broadband/harmonic shape, not simple dominant-peak relocation.
- May 25 Prof. Abell feedback response:
  - Added `docs/phase2_5_noise_mode_filtering.md` as a scoped Phase 2.5 spec.
  - Added `docs/research_update_2_phase2_5_addendum.md` for a short report addendum.
  - Phase 2.5 is framed as a support layer for JS spectral loss and XAI, not a new standalone research thread.
  - First checkpoint is Dataset 1 vs noisy mirror Dataset 91: any noise metric must score 91 noisier than 1 by more than within-clean spread before it is promoted.
  - Phase 2.5 should run in parallel with the early Phase 3 MLP baseline and gate only the later JS spectral-loss stage.
- Added `scripts/check_js_reference_rule.py` synthetic check proving all-unstable facets produce `insufficient_stable_reference`.
- May 25 Phase 2.5 spec refinements after self-review:
  - Verified the noisy-to-clean mirror pairing convention against the metadata: noisy IDs 61 to 90 use `noise_std=1.0` and pair as `clean = noisy - 60` (so 61 mirrors 1); noisy IDs 91 to 120 use `noise_std=5.0` and pair as `clean = noisy - 90` (so 91 also mirrors 1). Matched IDs share `omega`, `b_mm`, `blim`, `b_over_blim`, `fn`, `teeth`, `cut_dir`, and `diameter_mm`. Only `noise_std` differs, giving a clean / 1% / 5% noise triple per clean dataset rather than a single mirror.
  - Updated `docs/phase2_5_noise_mode_filtering.md` accordingly:
    - First checkpoint widened from "1 vs 91" to the three-way ladder "1 vs 61 vs 91" with monotone-ladder requirement.
    - Concrete acceptance statistic added: per-file paired difference > 2 sigma of within-clean spread, or Mann-Whitney U with p < 1e-3 as the distribution-level fallback.
    - Metric list reorganized into four families (PSD, wavelet/temporal, cross-channel, chatter-specific). The agreement rule now requires votes from PSD, wavelet/temporal, and cross-channel families; two PSD metrics agreeing no longer counts, so spectral flatness and entropy cannot double-count.
    - Wavelet defaults pinned: `db4`, level-1 detail, symmetric extension, `sigma = median(|d1|) / 0.6745`.
    - Per-file computational budget set to under 0.5 seconds on Great Lakes.
    - Mode-label decision rule added: v1 rule-based thresholds on chatter-band energy, TPF harmonic energy, noise floor ratio, and `b_over_blim`, calibrated against clean 1 to 60; v2 trained classifier as fallback; mode labels carry a confidence score.
    - DiCE `broadband_noise_dominated` behavior extended to emit a dataset-level "sensor needs attention" rollup, not only per-file refusals.
  - The Phase 2.5 addendum was rewritten in the Research Update 2 third-person voice and merged into `research_update.md` as Section 8, with `## 9. Time Spent` shifted accordingly. The standalone `docs/research_update_2_phase2_5_addendum.md` is now redundant; the docx and pdf were regenerated.
- May 25 Phase 2.5 pre-flight check on real metadata:
  - Added `scripts/quick_csv_ladder_check.py` that runs the 1/61/91 paired-difference ladder on the precomputed spectral features already in `data/metadata/features_*.csv`. The result is saved to `results/phase2_5/csv_ladder_checks/csv_ladder_check_1_61_91.csv` for inclusion in the next meeting.
  - All 102 files in Dataset 1 pair perfectly with 102 files in Dataset 61 and Dataset 91 on `(omega, b_mm)`. The ladder methodology is testable with no missing rows.
  - Five existing CSV metrics already pass the paired ladder gate with strict monotone behavior on all 102/102 pairs and MWU p < 1e-3: `Forcex_spectral_entropy` (p=1.8e-7, effect size 1.91 sigma), `Forcey_spectral_entropy` (p=2.0e-4, 2.68 sigma), `xtpos_spectral_entropy` (p=4.2e-8, 1.42 sigma), `ytpos_spectral_entropy` (p=3.1e-6, 1.65 sigma), `Forcex_harmonic_energy_ratio` (p=7.1e-7, -1.57 sigma). This proves the methodology works on the real data before any new code runs.
  - The existing sideband-energy and kurtosis columns in the CSV correctly fail the noise ladder. That is the intended behavior for chatter-specific features: they should be noise-robust and therefore should not separate clean from noisy.
  - The CSV `*_fn_proximity` columns are all-zero in the data and unusable; the new metric library replaces this with explicit `chatter_fn_band_energy` and `chatter_fn_band_kurtosis`.
- May 25 Phase 2.5 implementation completed locally and verified by tests:
  - Added `src/pri2m/noise/metrics.py` exposing `compute_file_metrics` and `NoiseMetricsConfig`. The library emits 44 per-file metric columns across four families: PSD (`psd_*`), wavelet/temporal (`wavelet_*`, `autocorr_*`), cross-channel (`cross_coh_*`), and chatter-specific (`chatter_*`). Wavelet defaults pinned to `db4`, level 1, MAD/0.6745. Analysis-band, harmonic, fn-band, and sideband band edges are all clamped to `[0, fs/2)` and to `max(tol, bin_width)`.
  - Added `tests/test_noise_metrics.py` with 15 closed-form synthetic tests (pure tone, pure noise, wavelet sigma recovery, fn-band energy/kurtosis, sideband injection at `fn +/- k*tpf`, cyclostationarity, cross-channel coherence, monotone noise floor, synthetic 1/61/91 ladder, NaN handling, short-signal handling, realistic combined signal finiteness, config override). All 15 pass.
  - Bug caught during library verification and fixed: `chatter_fn_band_energy` and `chatter_sideband_energy` were originally normalized by the `[0, 5*tpf]` analysis-band energy, which saturated the ratio to 1.0 whenever `fn > 5*tpf` (the typical machining case, e.g. `fn=1500`, `tpf=200`). Denominator switched to total signal energy `[0, fs/2)`, so the metric now spans the unit interval continuously (verified `0.0025` to `0.80` across a chatter sweep).
  - Added `src/pri2m/noise/modes.py` exposing `MODE_LABELS`, `ModeThresholds`, `aggregate_signal_metric`, and `assign_mode`. v1 rule-based decision with priority order: `low_speed_edge` (metadata), `broadband_noise_dominated` (three-family vote agreement), `mixed_tpf_chatter`, `chatter_dominated`, `near_boundary_transition`, `tpf_dominated_clean`, fall-through `mixed_tpf_chatter` at low confidence. Five threshold defaults are flagged as placeholders needing real-data calibration: `wavelet_mad_sigma_high_quantile_proxy`, `cross_coh_mean_low`, `chatter_fn_band_energy_high`, `chatter_cyclostationarity_tpf_low`, `psd_spectral_entropy_clean_max`.
  - Added `tests/test_noise_modes.py` with 12 tests covering each label transition, three-family agreement requirement, NaN/empty handling, threshold overrides, and the votes-explained debug dict. All 12 pass.
  - Added `scripts/compute_noise_metrics.py` (CLI that loads NPZ bundles, looks up metadata in either `features_all_datasets.csv` or `features_noisy_datasets.csv` by dataset ID, calls `compute_file_metrics`, and writes `noise_metrics.csv` + `run_metadata.json` + `skipped_files.csv`).
  - Added `scripts/calibrate_noise_mirror_pairs.py` (paired ladder gate on the clean / 1% / 5% triple, writes per-metric stats CSV and `promoted_metrics.json`).
  - Added `scripts/assign_signal_modes.py` (per-file mode label CSV using `assign_mode`; metadata-driven `low_speed_edge` heuristic when `fs/tpf < 8`).
  - Added `slurm/phase2_5_compute_noise_metrics.sbatch` (8 cpus, 24 GB, 4 h, log to `logs/phase2_5_noise_%j.log`) and `slurm/phase2_5_calibrate_mirror_ladder.sbatch` (2 cpus, 8 GB, 30 m). Both `bash -n` syntax checks pass. Caught and fixed a CLI mismatch from the SLURM agent: it invented a `--manifest` flag and dropped `--noisy-summary-csv`; corrected to match the real `compute_noise_metrics.py` argv.
  - Updated `slurm/README.md` with new section 5b documenting Phase 2.5 invocation, ladder defaults, and outputs.
  - Added `tests/test_phase2_5_end_to_end.py`: synthesizes a six-operating-point 1/61/91 ladder as NPZ files in a pytest `tmp_path`, runs `compute_noise_metrics.py`, `calibrate_noise_mirror_pairs.py`, and `assign_signal_modes.py` as subprocesses, and asserts (a) spectral entropy strictly monotone clean -> 1% -> 5% on all four signals at every operating point, (b) at least one PSD-family metric is promoted by the calibration gate, (c) all mode labels are well-formed and confidences sit in [0, 1]. Test passes.
  - Updated `pyproject.toml`: added `PyWavelets` to runtime dependencies and `pytest` as an optional `test` extra.
  - Full test suite: 28 of 28 passing (15 metrics, 12 modes, 1 end-to-end).
  - End-to-end synthetic run also produces sensible byproducts: top promoted metrics are the cross-channel coherences (collapse with noise, as expected) followed by PSD spectral flatness, off-harmonic ratio, and entropy; mode-label distribution shifts toward `chatter_dominated` and noise-flagged rows as `b_over_blim` rises and `noise_std` increases.
- May 25/26 Phase 2.5 Great Lakes pilot status:
  - Dataset 61 has now been extracted and the full `1/61/91` pilot ladder has run.
  - Pilot mode counts are cleanly monotone: Dataset 1 has 0/102 `broadband_noise_dominated`, Dataset 61 has 54/102, and Dataset 91 has 100/102.
  - Clean Dataset 1 physics labels are preserved: 22 `chatter_dominated`, 57 `mixed_tpf_chatter`, 9 `near_boundary_transition`, and 14 `tpf_dominated_clean`.
  - Calibration promoted 16 metrics across PSD, wavelet/temporal, and cross-channel families; the strongest include `cross_coh_xtpos_ytpos`, `wavelet_mad_sigma_Forcey`, `psd_noise_floor_ratio_Forcey`, `psd_spectral_flatness_Forcey`, `cross_coh_Forcex_Forcey`, and `wavelet_mad_sigma_Forcex`.
  - There were zero inverse indicators in the pilot, consistent with the current simulation noise behaving as correlated channel noise. Revalidate cross-channel sign on real ADC/sensor noise later.
  - Report phrasing for Prof. Abell: "The noise characterization works as designed on the 1/61/91 ladder. Dataset 1 has zero false-positive broadband-noise labels; Dataset 61 gets 53% of files flagged; Dataset 91 gets 98%. Clean chatter/TPF labels remain present, so the noise gate is not crowding out the physics signal."
  - `scripts/plot_pilot_results.py` renders three meeting-ready pilot figures. Local outputs are under `/Users/yash/Downloads/PRI2M/SINDy_DDE/results/figures/pri2m_pilot_figures/`: `pilot_mode_distribution.png`, `pilot_ladder_boxplots.png`, and `pilot_paired_ladders.png`.
  - `pilot_paired_ladders.png` shows 102/102 strictly upward paired traces for the selected noise-floor metric and 99/102 for cross-channel coherence.
- May 25 Phase 2.5 review fixes implemented:
  - Fixed the calibration gate so promoted metrics must be direction-safe: `clean < 1% noise < 5% noise`, positive 5%-minus-clean delta, and one-sided Mann-Whitney U for `5% > clean`. Direction-decreasing metrics are no longer promoted just because they are monotone.
  - Calibration now pairs mirrors by `(omega, b_mm, mirror_file_key)` instead of only `(omega, b_mm)`. Duplicate pair keys cause an explicit failure with a preview rather than silent `drop_duplicates`.
  - `promoted_metrics.json` now carries per-metric family, direction, clean median/mean/std/p95, noisy means, and the promotion rule so downstream code can use calibrated clean baselines.
  - `scripts/quick_csv_ladder_check.py` now uses the same direction-safe promotion rule, so the older pre-flight CSV helper no longer promotes metrics that decrease with noise.
  - `assign_signal_modes.py` now loads `promoted_metrics.json` and passes it into `assign_mode`; mode assignment is no longer disconnected from calibration.
  - Calibrated broadband-noise voting uses promoted metrics and clean p95 thresholds per metric/family, avoiding the previous cross-unit average of raw wavelet sigma across forces and positions.
  - `fs/tpf < 8` is now emitted as `sampling_quality_flag=undersampled_tpf`; it no longer forces the physical `low_speed_edge` mode. Rows with too many missing metrics now become `metrics_unavailable`.
  - Updated Phase 2.5 docs and research-update addendum to reflect the direction-safe 1/61/91 ladder, calibrated metrics, `metrics_unavailable`, and the undersampling quality flag.
  - Added `tests/test_noise_calibration.py` for directional promotion and duplicate-key rejection, plus a calibrated-noise mode test.
  - Local verification: `python -m compileall -q src/pri2m/noise scripts/compute_noise_metrics.py scripts/calibrate_noise_mirror_pairs.py scripts/assign_signal_modes.py` passed; `python -m pytest tests/test_noise_metrics.py tests/test_noise_modes.py tests/test_noise_calibration.py tests/test_phase2_5_end_to_end.py` passed with 31/31 tests.
  - Great Lakes Phase 2.5 outputs were copied locally under `results/great_lakes/phase2_5_and_pilot/phase2_5_noise_metrics_1_61_91/` and `results/great_lakes/phase2_5_and_pilot/phase2_5_calibration_1_61_91/`.
  - Created report-ready visualization assets:
    - `results/figures/research_update_assets/phase2_5_noise_ladder_summary.png`
    - `results/figures/research_update_assets/phase2_5_noise_ladder_summary.pdf`
  - The visualization shows the calibrated noise-score ladder, mode-label shift, representative promoted metric rise, and top promoted metric effect sizes across PSD, wavelet/temporal, and cross-channel families.

## Current Goals
- Implement the Phase 1-3 execution plan locally without copying full Great Lakes preprocessed pickles.
- Keep Great Lakes as source of truth for complete SINDy preprocessed artifacts.
- Create a compact local code layout for reproducibility, Phase 2 PSD/JS divergence, Phase 3 stage gates, and report-ready summaries.

## Current Status
- Confirmed local `/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE` is not mounted.
- User transferred the new Phase 1-3 scaffold to `/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE` on Great Lakes.
- Great Lakes project root now has the feature metadata under `/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/metadata/`; Phase 1 verifier reads `/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/metadata/features_all_datasets.csv`.
- User installed the editable package successfully on Great Lakes in the `prism` environment with `pip install -e .`.
- Great Lakes `python scripts/summarize_phase_status.py` confirmed 32,168 SINDy preprocessed rows across 120 datasets and no fit failures.
- User submitted Phase 1 Altintas-Budak verifier as SLURM job `50073876`; it completed with operational acceptance PASS.
- Phase 2 smoke extraction job `50074690` completed successfully:
  - Dataset 1, first 5 files.
  - Manifest path: `/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/phase2_signals/manifests/manifest_1.csv`.
  - Failures: 0.
  - Statuses: `ok: 5`.
- Phase 2 extraction should split clean metadata datasets from noisy dataset 91:
  - Clean pilot: `1,4,11,12,29,34,35,39,44-48` using `data/metadata/features_all_datasets.csv`.
  - Noisy mirror: `91` using `data/metadata/features_noisy_datasets.csv`.
- Phase 2 pilot jobs submitted on Great Lakes:
  - Clean extraction array: `50074777`.
  - Noisy extraction array: `50074778`.
  - Clean PSD/JSD dependency job: `50074792`.
  - Noisy PSD/JSD dependency job: `50074793`.
- Phase 2 pilot extraction completed:
  - `data/phase2_signals/manifests/` has manifests for clean dev/anomaly datasets and noisy dataset 91.
  - `grep -R ",failed," data/phase2_signals/manifests/` returned no failures.
  - `find data/phase2_signals -name '*.npz' | wc -l` returned `5306`.
- Initial PSD jobs were incomplete because `LIMIT=500` applied to the first rows of the summary CSV, not to the inspection datasets:
  - Clean PSD wrote 833 rows, which is partial.
  - Noisy PSD wrote 0 rows because the first 500 noisy metadata rows did not include dataset 91.
- Patched `scripts/run_psd_js.py` and `slurm/phase2_run_psd_js.sbatch` to accept `DATASETS`, filtering rows before `LIMIT`.
- Rerun PSD status:
  - Noisy PSD/JSD output now has `409` lines in `results/phase2/psd_js_noisy/psd_summary.csv`, consistent with dataset 91 (`102 files * 4 signals + header`).
  - Noisy `run_metadata.json` shows `dataset_filter_ids: [91]`.
  - First clean rerun only processed dataset 1 because SLURM comma parsing interpreted `--export=ALL,DATASETS="1,4,..."` as multiple export assignments.
  - Corrected clean rerun used `env DATASETS='1,4,11,12,29,34,35,39,44-48' INSPECT_DATASETS='11,12,34,35,44-48' OUTPUT_DIR=... sbatch --export=ALL ...`.
  - Corrected clean PSD output `results/phase2/psd_js_clean_pilot/psd_summary.csv` has `20,817` lines (`20,816` signal rows + header).
  - Clean `run_metadata.json` shows all intended dataset filters: `1,4,11,12,29,34,35,39,44,45,46,47,48`.
  - Clean `raw_signal_rows`: `20,816`; clean metadata rows processed: `5,204`.
- Added aggregate JS divergence implementation:
  - `scripts/aggregate_js_divergence.py`
  - `slurm/phase2_aggregate_js.sbatch`
  - Outputs stable-vs-unstable Jensen-Shannon divergence by `b_over_blim` bin, faceted by `signal`, `teeth`, and `cut_dir`.
  - Writes strict within-bin table `js_divergence_by_b_over_blim.csv`.
  - Writes report-usable fallback table `js_divergence_vs_stable_reference_by_b_over_blim.csv`, comparing each `b/blim` bin to the stable reference PSD for the same `signal x teeth x cut_dir` facet.
  - Stable-reference default is physical and label-free: lowest 10% of `b_over_blim` within each facet, capped to `b_over_blim <= 0.80` (`REFERENCE_MODE=low_b_ratio`, `REFERENCE_QUANTILE=0.10`, `REFERENCE_MAX_B_RATIO=0.80`).
  - `REFERENCE_MODE=label_stable` remains opt-in for methods comparison only and writes a metadata warning.
  - Writes `aggregation_failures.csv`, `run_metadata.json`, and optional `js_divergence_*_<signal>.png` plots if matplotlib is available.
  - SLURM wrapper sets `MPLCONFIGDIR` under the output directory to avoid matplotlib/font cache write issues.
- Added Phase 2.2 per-dataset PSD profile implementation:
  - `scripts/plot_psd_profiles.py`
  - `scripts/verify_b_over_blim_provenance.py`
  - `scripts/check_js_reference_rule.py`
  - `slurm/phase2_plot_psd_profiles.sbatch`
  - Outputs stable-vs-unstable PSD overlays per dataset/signal with TPF and `fn/tpf` markers.
  - Default signals are `xtpos,Forcex` to match the schedule deliverable.
- Phase 2 launch policy:
  - Use `STRICT=1` for the one-dataset smoke test only.
  - Omit `STRICT=1` for pilot arrays so per-file failures are captured in manifests without blocking downstream PSD/JSD on successfully extracted files.
  - Use `--export=ALL,...` for SLURM overrides.
- Parallel workers completed:
  - Altintas-Budak verifier slice.
  - PSD/Jensen-Shannon divergence slice.
- Main thread implemented shared config, metadata utilities, HPC sync tooling, MLflow checkpoint retention, report status tooling, and this handoff file.
- Combined smoke checks passed for all Python files under `src/` and `scripts/`.

## Successes
- Existing local summaries show `SINDy_DDE/preprocessed_log.csv` has 32,168 preprocessed rows across all 120 datasets.
- Existing local SINDy fit logs report 0 failures for datasets 1-120.
- `scripts/summarize_phase_status.py` confirms SINDy summary counts and prints Phase 3 dev-set metadata.
- `scripts/sync_hpc_artifacts.py --verify` correctly reports the Great Lakes mount unavailable locally while validating existing local summary logs.
- `scripts/validate_dev_datasets.py` passed all locked Phase 3 dev-set expectations.
- Added `pyproject.toml` so the `src/` layout can be installed with `pip install -e .` instead of relying on ambient `PYTHONPATH`.
- Config parsing works with `PYTHONPATH=src`; `phase3_dev` resolves to `[1, 4, 29, 35, 39, 91]` and PSD `nperseg` resolves to `2048`.
- MLflow checkpoint retention dry run keeps `best_*` and `final_*` checkpoints and marks periodic epoch checkpoints for deletion.
- `scripts/verify_altintas_budak.py --mode all --max-rows 20 --grid-points 41 --max-lobe-order 4` exits successfully:
  - Textbook smoke is operational.
  - Dataset `blim` validation scaffold runs but warns that constants/engagement assumptions need calibration.
  - Metadata summary passes without raw H5/pickle artifacts.
- Great Lakes Phase 1 verifier job `50073876` completed:
  - Phase 3 dev dataset expectations passed.
  - Dataset 91 confirmed as 5% noisy mirror of dataset 1.
  - Textbook smoke generated 5,117 candidate lobe points and passed operationally.
  - Dataset `blim` scaffold compared 158 sampled rows and warned due placeholder constants.
  - Metadata summary read 18,669 clean rows across datasets 1-60.
- `scripts/run_psd_js.py --summary-csv "data/metadata/features_all_datasets.csv" --dry-run --limit 3` exits successfully:
  - Reports PSD defaults.
  - Tags datasets 44-48 for anomaly inspection.
  - Gracefully skips raw PSD/JSD when no signal directory is provided.
- Synthetic PSD/JSD smoke run produced 4 PSD rows and 4 divergence rows for `xtpos`, `ytpos`, `Forcex`, and `Forcey`.
- `scripts/check_phase3_gate.py` passed on synthetic arrays with R2 `0.9994` and onset error `4.0%`.
- Patched `scripts/extract_phase2_signals.py` so array jobs write separate manifests under `data/phase2_signals/manifests/` instead of overwriting one shared `manifest.csv`.
- Updated `slurm/README.md` with split clean/noisy pilot extraction, non-strict array policy, and dependency-chained PSD/JSD commands.
- Metadata verified:
  - Dataset 29: high frequency plus strong damping.
  - Dataset 39: 5-tooth case.
  - Dataset 91: 5% noisy mirror of clean dataset 1.
  - Dataset 35: cut_dir=1 weak fold.
  - Dataset 4: high ktc coverage candidate.

## Failed Attempts / Blockers
- Direct local Great Lakes sync cannot run because `/nfs/turbo/...` is not visible from this environment.
- The metadata directory now lives at `data/metadata/features_*.csv` under the `SINDy_DDE` project root.
- Phase 2 smoke job `50074553` failed all five files because the extractor wrote NPZ metadata with key `file`, which conflicts with `numpy.savez_compressed(file=...)`.
- Patched `scripts/extract_phase2_signals.py` to store the NPZ metadata as `source_file` instead.
- Follow-up smoke still showed `data/phase2_signals/manifest.csv`, proving Great Lakes was still running the old script. Root cause: transferring individual nested paths with `rsync scripts/extract_phase2_signals.py remote:/repo/` copies the file to repo root unless `--relative` is used.
- Correct transfer command for patched nested files must use `rsync -avR ... /nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/` or target each subdirectory explicitly.
- Patch transfer was verified on Great Lakes with `source_file` and `manifests` present in `scripts/extract_phase2_signals.py`.
- SLURM export caveat: do not pass comma-containing values through `--export=ALL,DATASETS="1,4,..."`; use `env DATASETS='1,4,...' sbatch --export=ALL ...` instead.
- Spreadsheet artifact API could not load earlier because a native canvas dependency was blocked by macOS code signing; Python extraction was used instead.
- Direct `python -c "import pri2m"` failed before installation because the package uses a `src/` layout. Scripts now inject `src/`, and editable install is defined in `pyproject.toml`.
- Strict Altintas-Budak textbook validation intentionally fails until exact reference constants are supplied.
- Strict dataset `blim` acceptance is not calibrated yet; current scaffold median relative error is high under placeholder assumptions.

## What Is Left
- Done since the previous handoff snapshot:
  - Full clean Phase 2.1 extraction for datasets 1 to 60 completed on Great Lakes.
  - Full clean PSD summary for datasets 1 to 60 completed (`phase2_psd_js_clean_60`).
  - Phase 2.2 per-dataset stable-vs-unstable PSD overlay plots produced (`phase2_psd_profiles_clean_60`, 120 plots, zero failures).
  - Aggregate JS rerun on full clean 1 to 60 using `REFERENCE_MODE=low_b_ratio` with the 0.80 floor (`phase2_js_curves_clean_60`).
  - `b_over_blim` provenance verification resolved for in-range rows; low-speed out-of-grid rows tracked separately.
  - Phase 2.5 noise/mode library, mode-label module, CLIs, SLURM scripts, and full test suite are built and verified locally.
  - Phase 2.5 Great Lakes pilot on datasets `1/61/91` completed with 306 matched rows, 16 promoted metrics, and monotone mode shift from 0/102 to 54/102 to 100/102 `broadband_noise_dominated` labels.
- Immediate critical path:
  1. Save the pilot artifacts for the meeting: `promoted_metrics.json`, `mode_labels.csv`, and `mirror_calibration_1_vs_61_vs_91.csv` into `/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2_5/pilot_artifacts/`.
  2. Let the current Great Lakes chain finish:
     - `50914037`: noisy extraction for datasets `62-120` (running at last status).
     - `50914159`: full `1-120` noise metrics (pending on extraction).
     - `50914160`: pooled mirror calibration (pending on metrics).
  3. Read the pooled calibration. This is the production calibration because it tests all available `(D, D+60, D+90)` ladders, not only Dataset 1's operating regime.
  4. After pooled calibration, retune the five flagged `ModeThresholds` defaults (`wavelet_mad_sigma_high_quantile_proxy`, `cross_coh_mean_low`, `chatter_fn_band_energy_high`, `chatter_cyclostationarity_tpf_low`, `psd_spectral_entropy_clean_max`) against the clean-1-to-60 distribution if the mode labels disagree with PSD/JS evidence.
  5. Run the full noisy divergence aggregation for datasets 61 to 120 so the noise robustness story matches the clean JS story. This is independent of Phase 2.5 calibration.
  6. Start the Phase 3 baseline (MLP with hardcoded `tau` and MSE loss) in parallel with Phase 2.5. Phase 2.5 only gates the later JS-spectral-loss stage, not the baseline.
  7. Ask Prof. Abell for Altintas/Budak reference constants and generator engagement convention; set an owner/date for `blim` calibration before Phase 3 residual physics.
- Re-transfer changed Phase 2.5 files to Great Lakes before any rerun that depends on the latest local code:
  - `pyproject.toml` (adds `PyWavelets` runtime dep)
  - `src/pri2m/noise/__init__.py`
  - `src/pri2m/noise/metrics.py`
  - `src/pri2m/noise/modes.py`
  - `scripts/compute_noise_metrics.py`
  - `scripts/calibrate_noise_mirror_pairs.py`
  - `scripts/assign_signal_modes.py`
  - `scripts/quick_csv_ladder_check.py`
  - `scripts/pareto_front.py`
  - `scripts/explore_filter_candidates.py`
  - `scripts/plot_pilot_results.py`
  - `slurm/phase2_5_compute_noise_metrics.sbatch`
  - `slurm/phase2_5_calibrate_mirror_ladder.sbatch`
  - `slurm/README.md`
  - `docs/phase2_5_noise_mode_filtering.md`
  - `handoff.md`
  - Test-only files (`tests/*`, `tests/conftest.py`) only needed if pytest is to be run on Great Lakes; the library itself runs without them.
  - Run `pip install -e .` (or `pip install PyWavelets`) inside the `prism` env on Great Lakes so `pywt` is importable.
- Run `scripts/sync_hpc_artifacts.py --execute --verify` on a Great Lakes-mounted environment to pull missing helper source and lightweight outputs.
- Supply exact Altintas/Schmitz reference constants and generator engagement conventions, then rerun `scripts/verify_altintas_budak.py --strict`.
- Run Phase 2 raw H5-to-NPZ extraction on Great Lakes, starting with the dev + anomaly array.
- Run `scripts/run_psd_js.py` through SLURM after extraction completes so real PSD/JSD curves are available.
- Calibrate PSD sanity thresholds against real dev-set inspection results.
- Schedule/check off Prof. Abell weekly syncs and circulate this handoff plus the Phase 4 update.
- Update this file after each major implementation/verification step.

## 2026-05-26: Phase 2.5 Review Fixes and Pareto Front
Six findings from the May 25 code review were re-checked against live code. Four of them (strict-upward gate, file-basename pairing, calibration JSON wiring into `assign_mode`, low_speed_edge disambiguation) were already implemented; the remaining gaps and one new deliverable were addressed in this session.

### Fixes applied
- **Calibrated noise-vote rule relaxed (root cause of "0 broadband_noise rows" symptom).** In `src/pri2m/noise/modes.py`, `_calibrated_noise_vote` previously required all three families `{psd, wavelet_temporal, cross_channel}` to have at least one promoted metric AND vote yes. When a family had zero promoted metrics (e.g., the synthetic ladder has independent per-channel noise, so coherence drops and lands in `inverse_noise_indicators` not `promoted_metrics`), the rule failed closed. New rule: at least two PRESENT families vote yes, no PRESENT family votes against, absent families abstain. End-to-end test now reproduces and asserts the noisy distribution shifts toward `broadband_noise_dominated`.
- **Per-signal wavelet thresholds.** Added `ModeThresholds.wavelet_mad_sigma_high_by_signal: Mapping[str, float] | None` (default `None`, preserves legacy behavior). When provided, the wavelet family vote is a majority-of-present-signals comparison against per-signal cutoffs, replacing the cross-unit mean-aggregate that mixed force and position units.
- **Inverse noise indicators surfaced in calibration JSON.** `scripts/calibrate_noise_mirror_pairs.py` now also runs a one-sided MWU with `alternative="less"` and exports `inverse_noise_indicators`, `n_inverse_indicators`, and `inverse_indicator_note`. Not used for decisions; transparency only. Catches simulation-vs-real-world sign flips like cross-coherence rising with correlated simulation noise but expected to drop with independent ADC noise.
- **`.gitignore` cleanup.** Generated outputs and stale local material now live under `results/` or `archive/`, with both directories ignored by default.
- **`assign_signal_modes.py` exports new diagnostic column.** `abstaining_families` joins `noise_family_votes` in `mode_labels.csv` so downstream inspection can tell which families had no promoted metrics versus which voted no.

### New deliverable: Pareto front
- Added `scripts/pareto_front.py`. Inputs: `--baseline-metrics-csv` (typically the unfiltered noisy data), one or more `--variant name=path/to/variant.csv`, and the calibration JSON. Outputs `pareto_points.csv`, `pareto_summary.json`, and `pareto_front.png`.
- Added `scripts/explore_filter_candidates.py` for synthetic Butterworth-vs-wavelet sanity checks.
- Added `scripts/plot_pilot_results.py` for the three pilot figures (`pilot_mode_distribution.png`, `pilot_ladder_boxplots.png`, `pilot_paired_ladders.png`).
- Axes:
  - `noise_suppression_score = mean_baseline_noise_score - mean_variant_noise_score` where the noise score is the mean clipped clean-baseline z-score across promoted metrics.
  - `physics_preservation_score = mean(closeness_to_1(chatter_band_ratio), closeness_to_1(tpf_proxy_ratio))`, where `closeness_to_1(r) = max(0, 1 - |r - 1|)`. `chatter_band_ratio` is the mean of `chatter_fn_band_energy_*` + `chatter_sideband_energy_*` over baseline. `tpf_proxy_ratio` uses `1 - mean(psd_off_harmonic_ratio_*)` as a TPF-energy surrogate (TPF preservation keeps off-harmonic ratio low).
- The actual filter implementations are a follow-up deliverable; this script only defines and evaluates the axes so a filter pipeline has a well-specified target.
- No real-data Pareto figure exists yet. The only filter experiment so far is `scripts/explore_filter_candidates.py` on a synthesized signal, which showed naive universal-threshold wavelet denoising can annihilate chatter. To make a real Pareto plot, first write/apply production filters to NPZ trees, rerun `compute_noise_metrics.py` on each filtered tree, then pass those variant CSVs into `scripts/pareto_front.py` with the baseline mode labels.
- Synthetic filter exploration result: Butterworth bandpass `[TPF/2, 5*fn]` gave about 33% noise reduction with chatter preserved (+4%) and TPF intact; naive db4 Donoho universal-threshold wavelet removed essentially all noise but destroyed about 87% of chatter. Wavelet needs a level-restricted or SURE-style variant before it is worth running at scale.

### Tests
- Full suite passes locally (`pytest tests/`): 31 of 31, including the extended end-to-end test that now asserts (a) `inverse_noise_indicators` is present in the calibration payload and (b) Dataset 91 (5% noise) has strictly more `broadband_noise_dominated` rows than Dataset 1 (clean). The latter is the regression check for the calibration-wiring gap.

### Immediate next steps for Great Lakes
1. Re-transfer to Great Lakes:
   - `src/pri2m/noise/modes.py` (relaxed family rule, per-signal wavelet threshold field)
   - `scripts/calibrate_noise_mirror_pairs.py` (inverse-indicator bookkeeping)
   - `scripts/assign_signal_modes.py` (new diagnostic column)
   - `scripts/pareto_front.py` (new file)
   - `.gitignore`
   - `tests/test_phase2_5_end_to_end.py` (extended regression check)
   - `handoff.md`
2. Re-run `slurm/phase2_5_calibrate_mirror_ladder.sbatch` with the relaxed family rule in place. The previous run reported 0 `broadband_noise_dominated` rows on Dataset 91; this run is the regression check on real data.
3. Once a real filter candidate exists (Phase 2.5 follow-up), invoke `python scripts/pareto_front.py --baseline-metrics-csv <noisy> --variant low_pass=<filtered> --calibration-json <promoted.json> --output-dir <out>` to score it against the noise/physics trade-off.

## 2026-05-26 (second pass): Production-Grade Phase 2.5 Tightening
Second review of the May 26 changes raised five further findings; all addressed.

### Pooled multi-ladder calibration
- `scripts/calibrate_noise_mirror_pairs.py` now accepts `--pooled` to auto-detect every clean dataset `D` in the input where both `D + noise1_offset` (default 60) and `D + noise5_offset` (default 90) exist. Paired-triple rows are concatenated across all detected ladders, and the same promotion gate (strict-upward, one-sided MWU, family-restricted) runs on the pooled distribution. Single-triple mode is preserved for pilot smoke tests.
- The pooled output JSON adds `mode: "pooled"`, `ladders: [{clean_id, noise1_id, noise5_id, n_paired}, ...]`, and `n_ladders`. CSV lands at `mirror_calibration_pooled.csv`.
- Use this for the production "compare against the rest of the dataset" claim; the single-triple mode is no longer treated as production-grade.

### Voting rule alignment in docs and report
- `docs/phase2_5_noise_mode_filtering.md` and `research_update.md` now state the actual rule: at least two of the three calibrated families must vote yes with no present family voting against; absent families abstain. Both files document why the strict three-family AND was unsafe (silently fails closed whenever a family lands all its metrics in `inverse_noise_indicators`, e.g., correlated simulation noise → no promoted cross-channel metric).

### Pareto front made overshoot-safe
- `physics_preservation_score` is now `mean(max(0, 1 - |ratio - 1|))` across the available axes (closeness-to-1), not the raw ratio. A filter that doubles chatter-band energy now scores the same as one that erases it: only filters that hold the band near baseline get full credit. The raw ratios are still recorded in `pareto_points.csv` under `chatter_preservation_ratio` and `tpf_preservation_ratio` for debugging.

### Pareto front made mode-aware
- New `--mode-labels-csv` argument joins on `(dataset, file)` and segments the two physics axes by mode. Chatter-band preservation is scored only on rows whose mode label is in `{chatter_dominated, mixed_tpf_chatter}` (broadband-noise-dominated files are no longer penalized for losing energy that overlaps the `fn` band). TPF preservation is scored on every mode where TPF is believed present: `{tpf_dominated_clean, near_boundary_transition, mixed_tpf_chatter, chatter_dominated, broadband_noise_dominated}`. Only `metrics_unavailable` and `low_speed_edge` rows contribute to neither axis. Without the CSV, the script logs `mode_aware: false` in the summary and falls back to global scoring.

### low_speed_edge activated, with explicit opt-in
- `scripts/assign_signal_modes.py` now accepts `--omega-low` and `--omega-high`. When both are omitted (default), the low-speed-edge mode remains inactive and the label is documented as reserved. When set, rows with `omega < omega_low` or `omega > omega_high` get `mode_label = low_speed_edge` and the new `is_low_speed_edge` column is `True`. The output summary JSON records `omega_low`, `omega_high`, and `low_speed_edge_active`. The full Altintas-Budak stability-grid check is still deferred and explicitly marked in the code.

### Tests
- `tests/test_noise_calibration.py` now includes `test_pooled_calibration_combines_multiple_ladders` covering 3-ladder pooled auto-discovery.
- New `tests/test_pareto_front.py` with two cases: (a) closeness-to-1 penalizes overshoot and undershoot identically, (b) mode-aware mode skips chatter protection on broadband-noise rows while still scoring TPF preservation.
- Full suite: 36 of 36 passing locally after the current doc/code/test pass.

### Re-transfer manifest for the next Great Lakes pass
- `scripts/calibrate_noise_mirror_pairs.py` (pooled mode, int casts)
- `scripts/pareto_front.py` (closeness, mode-aware, wider TPF protection)
- `scripts/assign_signal_modes.py` (omega gate)
- `docs/phase2_5_noise_mode_filtering.md`
- `research_update.md`
- `tests/test_noise_calibration.py`, `tests/test_pareto_front.py`
- `handoff.md`
- Recommended production calibration on Great Lakes: `python scripts/calibrate_noise_mirror_pairs.py --noise-metrics-csv <out>/noise_metrics.csv --pooled --output-dir <out>/calib_pooled`, after the full clean (1-60) and full noisy (61-120) extractions and metric runs are complete.

## HPC Run Order Added
- Added `scripts/extract_phase2_signals.py` for raw H5 to compact NPZ extraction on Great Lakes.
- Added SLURM wrappers:
  - `slurm/phase1_verify_altintas.sbatch`
  - `slurm/phase2_extract_signals.sbatch`
  - `slurm/phase2_run_psd_js.sbatch`
  - `slurm/mlflow_checkpoint_cleanup.sbatch`
- Added `slurm/README.md` with the recommended submission order:
  1. Phase 1 verifier.
  2. Phase 2 signal extraction pilot/dev+anomaly array.
  3. Full clean signal extraction.
  4. PSD/Jensen-Shannon summary job.
  5. MLflow checkpoint cleanup after future training jobs.
- Local syntax checks passed for the extractor and all SLURM files.
- Local H5 extraction smoke test could not run because bundled Python lacks `h5py`; `h5py` is now listed in `pyproject.toml` and is expected in the Great Lakes `prism` environment used by the existing SINDy scripts.

## HPC Transfer Manifest
- All active Great Lakes jobs now run from `/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE`; legacy SINDy-DDE helper scripts and logs live under `sindy_pipeline/`.
- User confirmed Great Lakes `SINDy_DDE` already contains the required helper/source files:
  - `read_file.py`
  - `RMSsliding.py`
  - `build_state.py`
  - `sindy_dde.py`
  - `sindy_preprocess.py`
  - `sindy_fit.py`
  - `sindy_analysis.py`
  - `preprocessed_log.csv`
  - `rejection_log.csv`
  - `preprocessed/`
  - `sindy_fits/`
  - `sindy_analysis/`
- Transfer required new source/control files to the Great Lakes repo root:
  - `.gitignore`
  - `pyproject.toml` (now also requires `PyWavelets`)
  - `configs/phase1_3.yaml`
  - `src/pri2m/**` (now includes `src/pri2m/noise/`)
  - `scripts/sync_hpc_artifacts.py`
  - `scripts/validate_dev_datasets.py`
  - `scripts/verify_altintas_budak.py`
  - `scripts/extract_phase2_signals.py`
  - `scripts/run_psd_js.py`
  - `scripts/aggregate_js_divergence.py`
  - `scripts/plot_psd_profiles.py`
  - `scripts/check_js_reference_rule.py`
  - `scripts/verify_b_over_blim_provenance.py`
  - `scripts/check_phase3_gate.py`
  - `scripts/cleanup_mlflow_checkpoints.py`
  - `scripts/summarize_phase_status.py`
  - `scripts/compute_noise_metrics.py`
  - `scripts/calibrate_noise_mirror_pairs.py`
  - `scripts/assign_signal_modes.py`
  - `scripts/quick_csv_ladder_check.py`
  - `slurm/*.sbatch` (including the two new `phase2_5_*.sbatch` files)
  - `slurm/README.md`
  - `docs/altintas_budak_reference_notes.md`
  - `docs/psd_js_notes.md`
  - `docs/phase2_5_noise_mode_filtering.md`
  - `handoff.md`
- Transfer or confirm these existing metadata/log inputs are already present on Great Lakes:
  - `data/metadata/features_all_datasets.csv`
  - `data/metadata/features_noisy_datasets.csv`, if running noisy mirror extraction.
  - `Project Plan and Executive Summary Template/Project_Schedule_v2.xlsx`, only needed for human reference.
  - `sindy_pipeline/preprocessed_log.csv`
  - `sindy_pipeline/rejection_log.csv`
  - `sindy_pipeline/sindy_analysis.py`, `sindy_fit.py`, `sindy_preprocess.py`, and helper modules are already present based on the Great Lakes listing.
- Do not transfer generated or heavy outputs from local to HPC:
  - `__pycache__/`
  - `.venv/`, `venv/`, `.conda/`
  - local logs, zips, checkpoints, `results/mlruns/`
  - full preprocessed pickles, raw H5 files, or generated PSD/NPZ output directories.

## 2026-05-27: Local Handoff Refresh After Comprehensive Review

Current local project root is `/Users/yash/Downloads/PRI2M/SINDy_DDE`. The git repository root remains one level above it at `/Users/yash/Downloads/PRI2M`. The local tree was reorganized so active project material is under `SINDy_DDE/`; stale/generated material is intended to be under `archive/` or `results/`.

### Local handoff status
- This file is now updated through the May 27 local code review.
- The previous May 26 sections remain valid for the Phase 2.5 implementation history: pooled mirror calibration, relaxed calibrated-family voting, inverse indicators, per-signal wavelet thresholds, low-speed-edge opt-in, mode-aware Pareto axes, and 36/36 local tests passing at that time.
- Great Lakes job state was not rechecked from the local machine during this refresh. Last known job chain from the prior session:
  - `50914037`: extract noisy datasets `62-120`.
  - `50914159`: compute full `1-120` noise metrics, dependent on extraction.
  - `50914160`: pooled calibration and mode assignment, dependent on metrics.
- Treat those job statuses as last-known, not current. Recheck with `squeue -j 50914037,50914159,50914160` on Great Lakes before reporting runtime status.

### Important review findings to fix next
1. **Local SINDy helper files are missing from the active tree.**
   - `sindy_pipeline/sindy_preprocess.py` imports `read_file.py`, `RMSsliding.py`, and `build_state.py`.
   - `sindy_pipeline/sindy_fit.py` imports `sindy_dde.py`.
   - Those required helpers are not present locally under `SINDy_DDE/sindy_pipeline/`, although the prior Great Lakes listing indicated they exist on `/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE`.
   - Action: pull/copy those helper source files into the local active tree or document them as an explicit external dependency. Without them, a fresh checkout or sync from local cannot reproduce the SINDy preprocessing/fitting scripts.

2. **Phase 2 extraction silently falls back when `read_file.py` is missing.**
   - `scripts/extract_phase2_signals.py` uses `load_time_h5` when available, otherwise generic recursive HDF5 traversal.
   - Because `read_file.py` is missing locally, this fallback can make extraction "succeed" with different signal selection or metadata semantics.
   - Action: make fallback explicit with something like `--allow-h5-fallback`; otherwise fail closed when the canonical reader is unavailable.

3. **Non-finite sample handling can break signal alignment.**
   - `src/pri2m/noise/metrics.py::_coerce_signal` drops `NaN/Inf` independently per signal.
   - `src/pri2m/spectral/psd_js.py::_as_1d_finite` does the same for PSD/JS helpers.
   - This is acceptable for clean simulation bundles but risky for real ADC/dropout data because cross-channel coherence and PSD still assume uniformly sampled, aligned channels.
   - Action: use a common finite mask for cross-channel metrics, or reject/interpolate affected files with an explicit policy.

4. **Calibrated `noise_score` is metric-weighted, not family-weighted.**
   - The label decision in `src/pri2m/noise/modes.py` is family based, but `noise_score` averages all promoted metric z-scores directly.
   - A family with more promoted metrics can dominate the reported score/confidence even though the vote rule treats families as the unit of evidence.
   - Action: average within each present family first, then average family scores; optionally keep both metric-weighted and family-weighted scores for diagnostics.

5. **Pooled calibration lacks a finite-coverage gate.**
   - `scripts/calibrate_noise_mirror_pairs.py` can promote a metric if it is monotone/significant on the finite subset, even when the metric is missing for many paired rows.
   - This weakens the production claim that a promoted metric works across all ladders and operating regimes.
   - Action: require minimum `n_finite / n_paired` coverage, e.g. 80-90%, and write coverage into the calibration CSV/JSON.

6. **Pareto candidate scoring needs row alignment.**
   - `scripts/pareto_front.py` compares aggregate baseline and variant means without requiring the same `(dataset, file)` rows.
   - A filter candidate that drops hard/noisy files can look artificially better.
   - Action: inner-join baseline, variant, and mode labels on `(dataset, file)`; report missing/dropped rows and fail when mismatch exceeds a small tolerance.

7. **Archive policy conflicts with version-control preservation.**
   - `.gitignore` ignores `archive/`.
   - If the cleanup move is committed as-is, old tracked files may appear deleted without their archived copies being tracked.
   - Action: decide whether `archive/` is local-only or repository-preserved. If preserved, unignore a curated archive subset or add a tracked manifest explaining what was intentionally omitted.

8. **`assign_signal_modes.py` should reject infinities.**
   - `_safe_float` rejects `NaN` but permits `inf`.
   - Action: use `math.isfinite(v)` before accepting numeric metadata.

### Recommended next action order
1. Pull/copy the four missing SINDy helper source files into local `sindy_pipeline/` or make their external dependency explicit.
2. Patch the extraction fallback so production extraction does not silently change loader semantics.
3. Patch the small correctness fixes: finite-sample alignment policy, family-weighted calibrated score, pooled calibration coverage, Pareto row joins, and finite-only `_safe_float`.
4. Re-run local tests from `/Users/yash/Downloads/PRI2M/SINDy_DDE`: `python -m pytest -q tests`.
5. Recheck Great Lakes job chain and copy finished pooled artifacts back into `results/great_lakes/` if available.
6. Only after pooled calibration is confirmed, start production filter-candidate runs and generate a real-data Pareto figure.

### Current professor-facing summary
- Phase 2.5 pilot result remains the cleanest meeting-ready story: Dataset 1 has zero `broadband_noise_dominated` false positives, Dataset 61 has 54/102 broadband-noise labels, and Dataset 91 has 100/102. Clean chatter/TPF labels are preserved.
- FFT/PSD helped with independent noise measurement through spectral entropy, flatness, noise-floor ratio, and off-harmonic ratio, but FFT alone is insufficient because chatter and broadband noise both add off-harmonic energy. The current method combines PSD/FFT, wavelet MAD, cross-channel coherence, and chatter-band/mode logic.
- Pareto-front framing is implemented as an evaluator, but no real-data Pareto figure exists yet because production filters have not been applied to NPZ trees. Synthetic exploration showed Butterworth is plausible and naive universal-threshold wavelet is too aggressive for chatter preservation.

## 2026-05-27: Pooled mirror calibration locked as production

### Production calibration artifacts
- Pooled mirror calibration completed successfully (job `51039931`). This is the **production** calibration; the prior single-triple 1/61/91 result is superseded.
- Artifacts on Great Lakes: `/nfs/turbo/coe-jaabell/PRI2M/phase2_5_calibration_pooled_v2/`. Local mirror: `~/pri2m_pooled_v2/` (and `results/great_lakes/phase2_5_calibration_pooled_v2/`).
- Pool spans `n_ladders=30` (all clean datasets 1-30 paired with their +60 and +90 mirrors), `n_paired=7525` rows.
- `n_promoted=10` of 44 metrics. Promoted set: `wavelet_mad_sigma_{xtpos,ytpos,Forcex}`, `psd_noise_floor_ratio_{Forcex,xtpos,ytpos}`, `psd_spectral_entropy_{Forcex,xtpos,ytpos}`, `psd_spectral_flatness_Forcex`.
- Mode label counts across all 33,719 rows: `mixed_tpf_chatter=14520`, `broadband_noise_dominated=8751`, `tpf_dominated_clean=6418`, `near_boundary_transition=2213`, `chatter_dominated=1257`, `metrics_unavailable=560`.
- Submission lesson learned: the previous wrapper `slurm/phase2_5_calibrate_mirror_ladder.sbatch` requires `POOLED=1` via env var and was hard to invoke reliably (long `--export` lines + auto-indented paste broke heredocs). New self-contained wrapper `slurm/phase2_5_calibrate_pooled_v2.sbatch` has `--pooled` baked in and all paths hardcoded to the actual `/nfs/turbo/coe-jaabell/PRI2M/` layout (note: not the `SINDy_DDE/` subdir). This is the wrapper to use for any future pooled rerun.

### Pooled findings worth flagging
- **Cross-channel family abstains in pooled mode.** Both `cross_coh_xtpos_ytpos` (mono_frac=0.91) and `cross_coh_Forcex_Forcey` (0.80) failed the strict 0.95 monotone gate in pooled mode despite being promoted in the single-triple pilot. Cause: some clean datasets already sit near coherence saturation (~0.95), so noise can't push them strictly higher. The relaxed 2-of-3 family vote still works because PSD and wavelet/temporal both have promoted metrics. The production decision rests on PSD + wavelet/temporal only.
- **All four Forcey-channel metrics rejected with mono_frac 0.93-0.95.** Just below threshold. Not catastrophic, but means Forcey is excluded from the noise score. A new diagnostic (`scripts/diagnose_forcey_near_misses.py`) will identify whether 1-2 outlier ladders drive the rejection.
- **`wavelet_mad_sigma_Forcex` promoted but scale-unstable.** clean_std=35.6 vs clean_p95=2.56 (heavy outliers across the 30 ladders). Promoted on monotone+significance, but effect size is only 0.063. Future noise-score code should use clean_p95-based z-scoring, not clean_mean/std.
- **Coverage was good but ungated.** All 44 metrics evaluated on either 7,525/7,525 (100%) or 7,350/7,525 (97.7%) paired rows. No promoted metric was sparse - but the gate was not enforced. The 175-row PSD gap is shared across every PSD/cross-channel/chatter metric, suggesting the same files failed PSD computation rather than per-metric flakiness.
- **`inverse_noise_indicators=[]`** in pooled mode (same as single-triple). On simulated correlated noise no metric strictly decreased. Real ADC noise may flip the cross-channel sign - revisit when hardware data lands.

### Patches landed today (local, not yet on Great Lakes)
1. **`ModeThresholds` retuned against pooled clean_p95** (`src/pri2m/noise/modes.py`):
   - `wavelet_mad_sigma_high_by_signal` now defaults to per-signal cutoffs at 3 x clean_p95: Forcex=7.674, Forcey=11.723, xtpos=1.224e-6, ytpos=1.785e-6. Replaces the unit-mixing scalar `wavelet_mad_sigma_high_quantile_proxy=0.05` (kept as fallback).
   - `psd_spectral_entropy_clean_max`: 0.75 -> 0.43 (was never triggering; new value is max clean_p95 across the four signals).
   - `cross_coh_mean_low`: 0.30 -> 0.50 (~3 sigma below clean_mean=0.881; stays a broken-sensor floor, not a noise indicator).
   - `chatter_fn_band_energy_high` (0.15) and `chatter_cyclostationarity_tpf_low` (1.0) left as documented placeholders: the chatter family is intentionally noise-robust, so calibrating from clean_p95 would silence the detector. Comments now record why.
   - Each retuned default carries an inline provenance comment: "pooled mirror calibration N=7525, 30 ladders, 2026-05-27".
   - Test `tests/test_noise_modes.py::test_broadband_noise_row` updated to pass explicit per-signal wavelet sigmas so it still exercises the broadband-noise label under the new per-signal gate.
2. **JSON coverage transparency** (`scripts/calibrate_noise_mirror_pairs.py`):
   - Each entry in `promoted_metrics.json::metric_stats` now carries `n_finite` (int) and `coverage_fraction` (float in [0,1]). Downstream consumers (`assign_signal_modes.py`, `pareto_front.py`) can now see promotion robustness without re-reading the CSV.
   - CSV output schema unchanged.
   - TODO comment added near the promotion logic: a coverage promotion gate (e.g., `coverage_fraction >= 0.8`) is the next step but is not yet enforced.
   - Test `tests/test_noise_calibration.py::test_calibration_promotes_only_increasing_noise_metrics` extended to assert both fields are present and in range.
3. **Forcey near-miss diagnostic** (new):
   - `scripts/diagnose_forcey_near_misses.py` (186 lines). Imports `_discover_ladders`, `_mirror_file_key`, and `_pair_triple_for_ladder` from the production calibration script so ladder logic stays in lock-step. Computes per-ladder mono_fraction for every Forcey metric. Outputs `forcey_per_ladder_monotone.csv` and `forcey_summary.json` (per-metric: worst-3 ladders, `concentrated_in_few_ladders=True` iff dropping those 3 pushes pooled mono_fraction above 0.95).
   - `slurm/phase2_5_diagnose_forcey.sbatch` (53 lines). Modeled on `phase2_5_calibrate_pooled_v2.sbatch`: same headers, same env activation, hardcoded paths, no env-var passthrough.
   - `tests/test_forcey_diagnostic.py` synthetic 5-ladder test passes.

### Local verification
- `python -m pytest tests/ -q` -> **37 of 37 passing** (was 31; +3 from new Forcey diagnostic test + 3 absorbed by extended existing tests).

### Re-transfer manifest (local -> Great Lakes)
- `src/pri2m/noise/modes.py`
- `scripts/calibrate_noise_mirror_pairs.py`
- `scripts/diagnose_forcey_near_misses.py` (new)
- `slurm/phase2_5_calibrate_pooled_v2.sbatch` (already transferred earlier today)
- `slurm/phase2_5_diagnose_forcey.sbatch` (new)
- `tests/test_noise_modes.py`, `tests/test_noise_calibration.py`, `tests/test_forcey_diagnostic.py` (only needed if pytest is to run on HPC)
- `handoff.md`

### Immediate next actions
1. Transfer the files in the manifest above to Great Lakes.
2. Submit the Forcey diagnostic on HPC: `cd /nfs/turbo/coe-jaabell/PRI2M && sbatch slurm/phase2_5_diagnose_forcey.sbatch`. Read `forcey_summary.json` afterwards to decide whether Forcey can be salvaged by excluding 1-2 outlier ladders.
3. Optionally re-run the pooled calibration with the retuned defaults (only needed if mode-label downstream consumers should see the new thresholds; the promoted-metrics list itself won't change since calibration doesn't depend on ModeThresholds).
4. With the production calibration locked, start the Phase 3 MLP baseline. Phase 2.5 only gates the JS-spectral-loss stage, not the baseline.
5. Real-data Pareto: write a production wavelet/Butterworth filter, apply to a noisy NPZ tree, rerun `compute_noise_metrics.py`, then invoke `scripts/pareto_front.py` against the pooled `promoted_metrics.json` and `mode_labels.csv`.

## 2026-05-27 (late): Forcey diagnostic result + D=19 exclusion patch

### Forcey diagnostic outcome
- Submitted `slurm/phase2_5_diagnose_forcey.sbatch` on Great Lakes. Outputs landed at `/nfs/turbo/coe-jaabell/PRI2M/phase2_5_forcey_diagnostic/`.
- **Dataset 19 is the dominant outlier.** Its per-ladder `monotone_fraction` is `0.000` on 8 of 11 Forcey metrics while behaving normally on Forcex. Confirms a per-channel data anomaly, not a regime issue (D=19 metadata matches its peers: `fn=1500`, `fs=10M`, `teeth=4`; manifest_19.csv shows `status=ok`).
- Four PSD/wavelet Forcey metrics flagged `concentrated_in_few_ladders=True` — dropping the worst 3 ladders (always headed by D=19) pushes their pooled monotone fraction above 0.95:
  - `psd_noise_floor_ratio_Forcey` (0.9359 -> would clear)
  - `psd_spectral_flatness_Forcey` (0.9454 -> would clear)
  - `psd_spectral_entropy_Forcey` (0.9408 -> would clear)
  - `wavelet_mad_sigma_Forcey` (0.9224 -> would clear)
- `psd_off_harmonic_ratio_Forcey` (0.8790) and `cross_coh_Forcex_Forcey` (0.8031) are NOT concentrated; even after excluding D=19 they remain non-monotone (broader operating-regime variance).
- `autocorr_decay_lag_samples_Forcey` and `chatter_cyclostationarity_tpf_Forcey` are flat at 0.000 across every ladder. They are discrete/integer-style metrics that do not respond to noise level - flag for permanent exclusion from the noise-score candidate pool.
- The noisy mirrors for D=79 and D=109 have NPZs in `phase2_signals/dataset_79/` and `phase2_signals/dataset_109/` but no per-dataset `manifest_*.csv`. Noise-metrics computation found them via the metrics CSV anyway, so this is a manifest-bookkeeping gap, not a data gap.

### D=19 exclusion patch (local, not yet rerun on HPC)
- Added `--exclude-clean-ids` CLI flag to `scripts/calibrate_noise_mirror_pairs.py`. Comma-separated integers; applied only in pooled mode; written into the JSON as `excluded_clean_ids` (alongside `mode` and `ladders`). Single-triple mode warns to stderr if the flag is set.
- Added `slurm/phase2_5_calibrate_pooled_v3.sbatch`: a copy of `phase2_5_calibrate_pooled_v2.sbatch` with `--exclude-clean-ids 19` baked in, output dir `/nfs/turbo/coe-jaabell/PRI2M/phase2_5_calibration_pooled_v3`. v2 results stay intact for diff.
- Extended `tests/test_noise_calibration.py::test_pooled_calibration_excludes_clean_ids` to verify the JSON drops the excluded ladder and `excluded_clean_ids` round-trips.
- Local test suite still green: `python -m pytest tests/ -q` -> 38/38 passing (was 37).

### Re-transfer manifest (delta since the v2 run)
- `scripts/calibrate_noise_mirror_pairs.py` (now carries the `--exclude-clean-ids` flag)
- `slurm/phase2_5_calibrate_pooled_v3.sbatch` (new)
- `tests/test_noise_calibration.py` (new test added)
- `handoff.md`

### TKAN-PINN strategic reframe
The current Phase 2.5 deliverables are NOT standalone results: they are the **data-quality and conditioning layer** that makes a TKAN-PINN trainable on this dataset. Concrete reuse:
- Per-file `noise_score` -> sample weighting in the PINN supervised loss; down-weight `broadband_noise_dominated`, up-weight `tpf_dominated_clean` / `chatter_dominated`.
- `mode_label` -> conditioning input or routing signal for TKAN's gated structure; physics-loss strength varies by mode.
- 10 promoted noise metrics -> auxiliary scalar features per file (calibrated, dataset-invariant).
- JS spectral divergence (Phase 2.5 gates this) -> the spectral-fidelity loss head of the TKAN-PINN.
- `b_over_blim` provenance + Altintas-Budak `blim` (Phase 1) -> the actual physics constraint enforced by the PINN residual head.

Practical implication for Phase 3 sequencing: the MLP baseline currently planned in the handoff can be skipped or compressed if the project commits to TKAN-PINN, since the three loss heads (supervised + JS spectral + `blim` residual) are all gated by artifacts already in place except the last. **The remaining hard blocker for the PINN side is trustworthy `blim` calibration**, which is the same blocker already flagged before Phase 3 stage 2. Until exact Altintas/Schmitz reference constants and engagement conventions arrive from Prof. Abell, the PINN's physics loss has nothing to enforce against.

### Next actions in priority order
1. Re-transfer the delta files above and submit the v3 calibration: `sbatch slurm/phase2_5_calibrate_pooled_v3.sbatch`. Compare promoted set vs v2; expect 3-4 Forcey PSD metrics to enter the noise score.
2. Pull v3 artifacts locally and diff against v2 (`mirror_calibration_pooled.csv`, `promoted_metrics.json`, `mode_labels.csv` distribution).
3. Investigate D=19's Forcey signal specifically: subset `noise_metrics.csv` to `(dataset, file)` rows from D=19/D=79/D=109 and inspect Forcey columns vs Forcex columns. Confirm whether D=19 alone is broken or the anomaly lives in one of the mirrors (D=79 or D=109). Decide if exclusion is permanent or fixable upstream.
4. Ask Prof. Abell for Altintas/Schmitz reference constants and engagement convention. **This is the critical-path blocker for any PINN/TKAN-PINN physics head.**
5. Sketch the TKAN-PINN loss specification: three heads (supervised label, JS spectral with noise-score weighting, `blim` residual) plus mode-label conditioning. Defer training-code work until `blim` calibration lands.
6. (Lower priority) Patch the noisy-extraction manifests so D=79, D=109 (and any other affected noisy datasets) have per-dataset `manifest_*.csv` for bookkeeping consistency.

---

# CURRENT STATE SNAPSHOT (2026-05-27, end of day) — read this first

> **2026-05-29 and 2026-06-03 updates are appended at the very END of this file.** Since this snapshot: pre-Phase-3 hardening is done (review findings #2, #3, #4, #5, #6, #8 closed; #7 resolved; only #1 still open), the noisy 61-120 PSD/JS run ("PSD on the complete dataset") is submit-ready, and **the 2026-06-03 pre-Phase-3 audit found and FIXED two modeling blockers (mirror leakage + 700 negative-blim labels) via a new `src/pri2m/phase3/` package; local test suite is now 128/128.** Read the two END sections together with this one.

A fresh session should read THIS section first, then dip into the historical sections above only when context is needed. The historical record above is cumulative and chronological; this snapshot is what is true RIGHT NOW.

## Where things actually are

### Filesystem layout (matters because earlier sessions had confusion)
- **Local active tree**: `/Users/yash/Downloads/PRI2M/SINDy_DDE/`. Git root is one level up at `/Users/yash/Downloads/PRI2M/` (current branch: `xgboost`).
- **Great Lakes project root**: `/nfs/turbo/coe-jaabell/PRI2M/` (NOT the `SINDy_DDE/` subdir, despite what older sections of this file say). The active SLURM jobs run from there. `scripts/`, `src/pri2m/`, `slurm/`, `tests/` all live directly under `PRI2M/` on HPC.
- **Data on HPC**:
  - Raw H5 source: `/nfs/turbo/coe-jaabell/datasets/tonylschmitz/digital-machining-database/versions/10/Dataset NN h5/`.
  - Extracted NPZs: `/nfs/turbo/coe-jaabell/PRI2M/phase2_signals/dataset_NN/`.
  - Feature CSVs: `/nfs/turbo/coe-jaabell/PRI2M/Dataset/features_all_datasets.csv` (clean 1-60) and `features_noisy_datasets.csv` (noisy 61-120).
  - Noise metrics CSV (33,719 rows): `/nfs/turbo/coe-jaabell/PRI2M/phase2_5_noise_metrics_full/noise_metrics.csv`.
  - **Production calibration**: `/nfs/turbo/coe-jaabell/PRI2M/phase2_5_calibration_pooled_v2/` (pooled across 30 ladders, 7525 paired rows, 10 promoted metrics). Mirror local: `results/great_lakes/phase2_5_calibration_pooled_v2/`.
  - Forcey diagnostic outputs: `/nfs/turbo/coe-jaabell/PRI2M/phase2_5_forcey_diagnostic/`.
  - v3 (D=19 excluded): **does not yet exist**. Ready to submit via `slurm/phase2_5_calibrate_pooled_v3.sbatch`.

### Local code status
- Editable package install required: `pip install -e .` (uses `pyproject.toml`; runtime deps include `PyWavelets`, `h5py`, `numpy`, `pandas`, `scipy`).
- Full test suite: 38/38 passing via `python -m pytest tests/ -q`.
- Key local files added/modified today (2026-05-27):
  - `src/pri2m/noise/modes.py`: `ModeThresholds` retuned from pooled clean_p95 (per-signal wavelet cutoffs, `psd_spectral_entropy_clean_max` = 0.43, `cross_coh_mean_low` = 0.50, chatter thresholds documented as placeholders).
  - `scripts/calibrate_noise_mirror_pairs.py`: now emits `n_finite` and `coverage_fraction` per metric in the JSON, accepts `--exclude-clean-ids` (pooled-mode only), includes a TODO for a future coverage promotion gate.
  - `scripts/diagnose_forcey_near_misses.py` (new, 210 lines): per-ladder mono_fraction breakdown for Forcey-named metrics, identifies `concentrated_in_few_ladders` candidates.
  - `slurm/phase2_5_calibrate_pooled_v2.sbatch` (new): self-contained pooled-mode wrapper with all paths hardcoded to the real HPC layout (no env-var passthrough required).
  - `slurm/phase2_5_calibrate_pooled_v3.sbatch` (new): same as v2 but with `--exclude-clean-ids 19` and a fresh output dir.
  - `slurm/phase2_5_diagnose_forcey.sbatch` (new).
  - Tests: `tests/test_noise_modes.py`, `tests/test_noise_calibration.py`, `tests/test_forcey_diagnostic.py` (new).
  - `sync_to_hpc.sh`: one-shot rsync wrapper to push the delta to Great Lakes.

### HPC code status
- Sync state matches local **as of the pre-v3 transfer**. The `--exclude-clean-ids` flag and `phase2_5_calibrate_pooled_v3.sbatch` are local-only until next rsync.
- macOS uses `openrsync` not GNU rsync. `--relative` is buggy; multi-source rsync paste-splits in some terminals. Use `bash sync_to_hpc.sh` (one short command) or single-source-per-rsync. Heredocs and long `--export=ALL,...` lines also get mangled by terminal paste — avoid both.

## Pooled calibration result (PRODUCTION; do not cite single-triple anymore)

```
Pooled mirror calibration v2, 2026-05-27:
  30 ladders auto-discovered (clean D in [1,30], paired with D+60 noise1 and D+90 noise5)
  7,525 paired operating points
  10 promoted metrics (PSD + wavelet/temporal; cross-channel and chatter families abstain)
  0 inverse noise indicators
```

Mode-label distribution verified against `mode_labels.csv` (all 33,719 rows):

| Tier | Total | broadband_noise_dominated | Rate |
|---|---:|---:|---:|
| Clean 1-30 (in pool) | 7,525 | 123 | **1.6%** false positives |
| Clean 31-60 (held out) | 11,144 | 33 | **0.3%** false positives — generalization |
| All Clean (1-60) | 18,669 | 156 | **0.8%** false positives |
| 1% noise (61-90) | 7,525 | 1,405 | **18.7%** detection |
| 5% noise (91-120) | 7,525 | 7,190 | **95.5%** detection |

This is the professor-ready headline. Clean Forcex/xtpos/ytpos clean_p95 from pooled JSON: see `results/great_lakes/phase2_5_calibration_pooled_v2/promoted_metrics.json`.

**Per-dataset diagnostics worth knowing about (do not present as bugs; understand first):**
- Top clean broadband false positives: D=23 (20.5%, 23/112), D=12 (10.3%, 30/292), D=13 (6.9%), D=18 (5.8%), D=2 (4.9%).
- Weakest 5%-noise detections: D=102 (76.0%), D=99 (83.3%), D=100 (86.9%), D=94 (87.1%), D=108 (87.8%).
- These specific dataset IDs are the ones to investigate before claiming the method is fully stable.

**Promoted metric set (use these for noise_score; do NOT use chatter or autocorr_decay_lag in noise_score):**
1. `wavelet_mad_sigma_xtpos` (effect 0.82)
2. `wavelet_mad_sigma_ytpos` (effect 0.77)
3. `wavelet_mad_sigma_Forcex` (effect 0.06 — scale-unstable; clean_p95 better than clean_mean for z-scoring)
4. `psd_noise_floor_ratio_Forcex` (effect 0.44)
5. `psd_spectral_entropy_Forcex` (effect 0.39)
6. `psd_noise_floor_ratio_xtpos` (effect 0.37)
7. `psd_spectral_entropy_xtpos` (effect 0.35)
8. `psd_noise_floor_ratio_ytpos` (effect 0.35)
9. `psd_spectral_flatness_Forcex` (effect 0.35)
10. `psd_spectral_entropy_ytpos` (effect 0.33)

**Forcey channel notes**: all four Forcey metrics narrowly missed promotion (mono_frac 0.93-0.95) because Dataset 19 has `mono_fraction = 0.000` on the Forcey channel across all 11 Forcey metrics while behaving normally on Forcex. D=19 metadata is ordinary; D=19 manifest is `ok`. Diagnosis pending: per-channel data anomaly in D=19 (or in its mirrors D=79, D=109 — whose NPZs exist but per-dataset manifests do not). The v3 calibration (D=19 excluded) is expected to promote 3-4 Forcey metrics and strengthen the noise score.

## What is the actual next action (in order)

This list supersedes everything labeled "next actions" above.

0. **Submit v3 calibration on HPC** (5 min):
   ```bash
   bash sync_to_hpc.sh           # transfer the delta locally first
   ssh greatlakes "cd /nfs/turbo/coe-jaabell/PRI2M && sbatch slurm/phase2_5_calibrate_pooled_v3.sbatch"
   ```
   Verify result: `grep -E '"mode"|"n_ladders"|"excluded_clean_ids"|"n_promoted"' /nfs/turbo/coe-jaabell/PRI2M/phase2_5_calibration_pooled_v3/promoted_metrics.json`.

1. **Pull v3 artifacts locally and diff against v2.** Compare promoted set, label_counts, clean false-positive rate. If 3-4 Forcey PSD metrics enter the promoted list and clean FP stays near 1.6% or improves, v3 becomes the production calibration.

2. **Generate three meeting-ready figures** (script does not exist yet; ~1 hr to write):
   - Mode-label distribution by tier (clean 1-30, clean 31-60, 1% noise, 5% noise).
   - Per-dataset broadband rate (highlight D=23, D=12 FPs and D=102, D=99 weak detections).
   - Promoted metric bar plot showing monotone fractions and effect sizes.

3. **Diagnose D=23 and D=12 clean FP rates** (top two; ~1 hr). Same script can handle all clean FPs. Goal is to understand whether these are real physics/chatter-heavy clean cases (where the broadband label is actually correct), missing metrics, or threshold artifacts.

4. **Production guardrails (the May 27 review findings still open)** — half a day total:
   - Family-weighted `noise_score` (May 27 finding #4): in `src/pri2m/noise/modes.py`, average within-family first, then across families. Currently averages all promoted-metric z-scores directly.
   - Pooled calibration finite-coverage gate (finding #5): coverage info is now exposed in JSON (TODO comment in `calibrate_noise_mirror_pairs.py` near the promotion logic), but the gate itself is not enforced. Patch: require `coverage_fraction >= 0.8` for promotion.
   - Pareto baseline/variant row alignment (finding #6): `scripts/pareto_front.py` should inner-join on `(dataset, file)` and fail on tolerance breach.
   - `_safe_float` infinity rejection (finding #8): `scripts/assign_signal_modes.py` should reject `inf` (currently only rejects NaN).
   - Extraction fallback documentation (finding #2): `scripts/extract_phase2_signals.py` silently falls back to generic H5 traversal when `read_file.py` is missing. Either require `--allow-h5-fallback` or fail closed.

5. **Butterworth filter + first real-data Pareto** (~1 day):
   - Apply a Butterworth band-pass filter `[TPF/2, 5*fn]` to a noisy NPZ tree (e.g., D=91 first, then 91-120 for the production pass).
   - Recompute `noise_metrics.csv` on the filtered tree.
   - Invoke `scripts/pareto_front.py --baseline-metrics-csv <noisy> --variant butterworth=<filtered> --calibration-json <pooled promoted_metrics.json> --mode-labels-csv <pooled mode_labels.csv> --output-dir ...`.
   - Defer wavelet denoising until a level-restricted / SURE-style implementation exists; the naive Donoho universal-threshold wavelet annihilates chatter.

6. **Phase 3 MLP baseline** (parallelizable from this point). Hardcoded `tau`, MSE loss, MLflow logging. Treat pooled mode labels and noise scores as optional metadata first, not as a hard dependency.

7. **TKAN-PINN scope discussion** with Prof. Abell. The current Phase 2.5 deliverables are the data-quality and conditioning layer for a TKAN-PINN:
   - Per-file `noise_score` -> supervised-loss sample weighting (down-weight broadband-dominated rows).
   - `mode_label` -> conditioning input or gating signal for the three loss heads.
   - 10 promoted noise metrics -> auxiliary scalar features.
   - JS spectral divergence -> spectral-fidelity loss head.
   - `blim` (Phase 1) + `b_over_blim` provenance -> physics-residual loss head.
   The hard blocker is `blim` calibration. **Ask Prof. Abell for the exact Altintas/Schmitz reference constants and engagement convention.** Without those, the physics head has nothing to enforce against. This blocks the residual physics stage (Phase 3 stage 2) regardless of whether the model is MLP or TKAN-PINN.

8. **(Low priority)** Patch the noisy-extraction manifests so D=79, D=109 and other affected noisy datasets have per-dataset `manifest_*.csv`. The NPZs exist; only the bookkeeping is missing.

## How to do common operations (replay-safe; previous chats stumbled on these)

- **Transfer code to HPC**: `cd /Users/yash/Downloads/PRI2M/SINDy_DDE && bash sync_to_hpc.sh`. Do NOT paste multi-source rsync — macOS's openrsync misbehaves and the terminal injects newlines into long pastes.
- **Submit a pooled calibration on HPC**: use the new `phase2_5_calibrate_pooled_v2.sbatch` or `_v3.sbatch` wrappers. They are self-contained. Do NOT use the older `phase2_5_calibrate_mirror_ladder.sbatch` — passing `POOLED=1` via env var has been unreliable due to the wrapper version on HPC.
- **Verify a calibration was actually pooled**: `grep -E '"mode"|"n_ladders"|"excluded_clean_ids"' .../promoted_metrics.json`. Production runs must show `"mode": "pooled"` and `n_ladders >= 29`.
- **Pull HPC artifacts locally**: use single-source rsync per file, or write a wrapper script. `--relative` is broken on openrsync. Trailing slashes on both source and destination paths are required to copy contents (not nest).
- **Run the test suite locally**: `cd /Users/yash/Downloads/PRI2M/SINDy_DDE && python -m pytest tests/ -q`. Current expected pass count: 38.

## Known blockers (will stop forward progress if not resolved)

1. **`blim` calibration**. Phase 1 verifier (`scripts/verify_altintas_budak.py`) operates on placeholder constants. Strict mode fails. Required from Prof. Abell: exact reference constants (cutting coefficients, helix angle, etc.) and the generator engagement convention used in the dataset. **Critical path for Phase 3 stage 2 / TKAN-PINN physics head.**
2. **HPC sync drift**. Some HPC sbatch files lagged the local versions today (caused the failed POOLED=1 attempts). Anytime calibration code changes, re-rsync before submitting. The new self-contained `_v2`/`_v3` wrappers reduce this risk.

## Known non-blockers (worth flagging, not stopping work)

- Cross-channel coherence as a noise indicator may flip sign on real ADC noise (currently rises with simulated correlated noise). Recheck on hardware data when it arrives.
- D=19 Forcey anomaly: ladder-level exclusion in v3 is the pragmatic fix. The root cause (whether D=19, D=79, or D=109 has the actual bad signal) is unresolved.
- Manifests for some noisy datasets are missing; NPZs and metrics computations succeeded anyway.
- The five `ModeThresholds` defaults flagged in earlier sessions: 2 are retuned against pooled clean_p95, 2 are documented as placeholders (chatter family), 1 is documented as a "broken sensor" floor (cross-channel).

---

## 2026-05-29: Pre-Phase-3 hardening, new diagnostics, noisy PSD/JS prep

Closed the open production-hardening review findings, added the two meeting/diagnostic scripts, and made the "PSD on the complete (noisy) dataset" run submit-ready. All work is LOCAL; transfer with `bash sync_to_hpc.sh` (rewritten for this delta). Executed via 8 parallel TDD subagents; full local suite is now **68 passing** (was 38). No `pyproject.toml` change (no new runtime deps), so no `pip install` needed on GL.

### Review findings closed (each TDD'd, with its own test)
- **#2 extraction fail-closed** (`scripts/extract_phase2_signals.py`): new `--allow-h5-fallback` flag + `resolve_loader()`. When the canonical `read_file.py` / `load_time_h5` reader is missing, extraction now FAILS CLOSED with a clear message instead of silently using generic H5 traversal; `--allow-h5-fallback` opts back in (with a stderr warning). Test: `tests/test_extract_phase2_signals.py` (3).
- **#3 cross-channel common finite mask** (`src/pri2m/noise/metrics.py`): cross-channel coherence (`_pair_coherence`) now applies a COMMON finite mask across the channel pair (added `raw_cache` + `_coerce_signal_raw`), so NaNs at different indices no longer misalign the two channels. Single-signal metrics unchanged. `psd_js.py` left as-is (its `_as_1d_finite` is single-signal only). Test added to `tests/test_noise_metrics.py` (NaN-scatter alignment; coherence ~0.035 → ~0.995 after fix). `compute_file_metrics` signature unchanged.
- **#4 family-weighted noise_score** (`src/pri2m/noise/modes.py`): `noise_score` now averages within each present family first, then across families (matching the family-based label vote), instead of flat-averaging all promoted-metric z-scores. The old flat value is retained as `noise_score_metric_weighted` for diagnostics. Label/vote logic unchanged. Test added to `tests/test_noise_modes.py`.
- **#5 coverage promotion gate** (`scripts/calibrate_noise_mirror_pairs.py`): the TODO is now enforced — a metric must have `coverage_fraction >= --min-coverage` (default 0.8) AND pass the existing monotone/MWU/direction checks to be promoted. JSON now carries `min_coverage` (top-level) and `coverage_ok` per metric. Test added to `tests/test_noise_calibration.py`. NOTE: the v2 promoted metrics were at 97.7–100% coverage, so re-running pooled calibration under this gate should NOT change the promoted set — but verify after the next pooled run.
- **#6 Pareto row alignment** (`scripts/pareto_front.py`): baseline and each variant (and mode labels) are now inner-joined on `(dataset, file)` before scoring, so a filter that drops hard files can't look artificially better. New `--max-row-mismatch-frac` (default 0.05) fails the run on excessive mismatch; `n_baseline/n_variant/n_joined/n_dropped` reported per variant. Score formulas unchanged. Test added to `tests/test_pareto_front.py`.
- **#8 `_safe_float` rejects inf** (`scripts/assign_signal_modes.py`): now uses `math.isfinite` (rejects NaN and ±inf, previously only NaN). Test: `tests/test_assign_signal_modes.py` (8).
- **#7 archive policy**: `archive/` is declared **local-only** (git-ignored, never rsynced). Added tracked `archive/MANIFEST.md` documenting each subdir and why it's omitted; `.gitignore` now has `!archive/MANIFEST.md` so the record survives a commit. Resolves the "silent deletion of tracked files" risk.

### New scripts (verified on real local data, but tests use synthetic tmp_path)
- `scripts/plot_phase2_5_meeting_figures.py` — 3 meeting figures from `mode_labels.csv` + `promoted_metrics.json`: mode distribution by tier, per-dataset broadband rate (D=23/D=12 FPs and D=102/D=99 weak detections highlighted), promoted-metric monotone/effect bars. Reproduced the headline tier rates (clean 1-30 = 1.6%, clean 31-60 = 0.3%, 1% = 18.7%, 5% = 95.5%). Outputs to `results/figures/phase2_5_meeting/`. Test: `tests/test_meeting_figures.py` (5). Needs matplotlib (local-only use).
- `scripts/diagnose_clean_false_positives.py` — ranks clean datasets by broadband FP rate; on the real local `mode_labels.csv`, D=23 ranks #1 (20.5%) and D=12 #2 (10.3%), as expected. Rates-only without metrics; add `--noise-metrics-csv` (full file, GL) + `--promoted-json` for the per-metric driver breakdown. Outputs to `results/phase2_5/clean_fp_diagnostic/`. Test: `tests/test_clean_fp_diagnostic.py` (8).

### "PSD on the complete dataset" — noisy 61-120 made submit-ready
The Phase 2.1 PSD summary + aggregate JS divergence had only been run at full scale on **clean 1-60** (confirmed: `phase2_js_curves_clean_60/run_metadata.json` has `dataset_filter_ids 1..60`). The Phase 2.5 noise-metric PSD *features* cover all 120, but the stable-reference JS *pipeline* did not. Added two **self-contained** wrappers (paths hardcoded to the real `/nfs/turbo/coe-jaabell/PRI2M` layout — no env-var passthrough, the failure mode flagged earlier):
- `slurm/phase2_run_psd_js_noisy_60.sbatch` → `phase2_psd_js_noisy_60/` (datasets 61-120, `features_noisy_datasets.csv`).
- `slurm/phase2_aggregate_js_noisy_60.sbatch` → `phase2_js_curves_noisy_60/` (same reference rule as clean: `low_b_ratio`, q=0.10, b/blim≤0.80, so the noisy curves are directly comparable to the clean ones).

Both verified locally: `bash -n` clean, and `parse_dataset_selector('61-120')` → 60 datasets in both scripts. They read NPZs directly and are independent of each other, so submit both:
```
cd /nfs/turbo/coe-jaabell/PRI2M
sbatch slurm/phase2_run_psd_js_noisy_60.sbatch
sbatch slurm/phase2_aggregate_js_noisy_60.sbatch
```
Pre-check the noisy NPZs exist: `ls phase2_signals/ | grep -cE 'dataset_(6[1-9]|[7-9][0-9]|1[01][0-9]|120)'` (expect 60). They do — the full `noise_metrics.csv` was built from them.

### Noisy 61-120 JS result (jobs 51111906 PSD + 51111907 JS, completed 2026-05-29)
`phase2_js_curves_noisy_60/` pulled to local `results/phase2/js_curves_noisy_60/`. Run is clean: `dataset_filter_ids` 61-120, `failure_rows=0`, `ok_stable_reference_rows=116/116`, `aggregation_failures.csv` header-only, processed 14,798 files / 59,192 signal rows, same reference rule as clean (low_b_ratio, q=0.10, b/blim≤0.80).

**Noise-robustness story holds.** Median JS by b/blim bin (clean 1-60 vs noisy 61-120), all four signals: JS stays low below the boundary (~0.001-0.02) and rises sharply across b/blim=1 under both 1% and 5% noise. The mid-band rise is actually STRONGER under noise (Forcex [2,3): clean 0.23 → noisy 0.52; xtpos [2,3): 0.15 → 0.52) — added broadband energy compounds the chatter divergence rather than masking it. Force signals strongest, positions weaker (same ordering as clean). This supports using JS spectral divergence as a Phase 3 loss component even on noisy data.

**Caveat (not a robustness failure):** the noisy curve dips below clean at the extreme tail ([3,5), [5,inf)). Facet-composition artifact: noisy 61-120 mirror clean 1-30 only, so those top bins are populated by a SINGLE teeth×cut_dir facet (vs 4 facets below the boundary), and a single-facet median isn't comparable to clean's multi-facet median. The decision-relevant region around b/blim=1 has 4 facets and 1,000-3,000 files per bin and is solid. For a strict magnitude comparison, re-run the clean JS restricted to `--datasets 1-30` so the facet mix matches the noisy set.

### Re-transfer manifest (local → GL) — `bash sync_to_hpc.sh` (rewritten this session)
- src: `src/pri2m/noise/metrics.py`, `src/pri2m/noise/modes.py`
- scripts: `extract_phase2_signals.py`, `calibrate_noise_mirror_pairs.py`, `pareto_front.py`, `assign_signal_modes.py`, `plot_phase2_5_meeting_figures.py`, `diagnose_clean_false_positives.py`
- slurm: `phase2_run_psd_js_noisy_60.sbatch`, `phase2_aggregate_js_noisy_60.sbatch`
- tests (only if running pytest on GL): 4 new (`test_extract_phase2_signals.py`, `test_assign_signal_modes.py`, `test_meeting_figures.py`, `test_clean_fp_diagnostic.py`) + 4 modified (`test_noise_metrics.py`, `test_noise_modes.py`, `test_noise_calibration.py`, `test_pareto_front.py`)
- `.gitignore`, `handoff.md`

### Still open (NOT done in this batch)
- **Finding #1 — 4 SINDy helper files missing locally** (`read_file.py`, `RMSsliding.py`, `build_state.py`, `sindy_dde.py`). They live on GL; fetch DOWN for local reproducibility (does not block this batch, but is what finding #2's fail-closed path checks for if extraction is ever run locally). Suggested fetch (adjust the GL source path to wherever they actually live — likely `sindy_pipeline/` or the project root):
  ```
  rsync -av yashkulk@greatlakes.arc-ts.umich.edu:/nfs/turbo/coe-jaabell/PRI2M/sindy_pipeline/{read_file.py,RMSsliding.py,build_state.py,sindy_dde.py} sindy_pipeline/
  ```
- **Butterworth filter + real-data Pareto** — deferred by decision; needs a GL filter run to produce the Pareto figure.
- **Re-run pooled calibration under the new coverage gate (#5)** to confirm the promoted set is unchanged, then this is the production calibration.

### Dataset 39 teeth correction (2026-05-29, table/metadata)
User flagged that the dataset descriptions table listed **Dataset 39 as 4 teeth when it actually uses 5**; the Excel is now fixed. Verified locally that this was a **documentation/table** fix, not a data fix — the operative metadata already carried 5:
- `data/metadata/features_all_datasets.csv` (what `run_psd_js`, `aggregate_js_divergence`, and `compute_noise_metrics` actually read) has D39 `teeth=5`. The spreadsheet's `m` column = the CSV `teeth` value on every spot-checked dataset (D1/29/35/40/60 = 4, D39 = 5).
- Both local `Dataset descriptions.xlsx` copies now show D39 `m=5`: the new one at **repo root** (mtime 2026-05-29 13:02) and the older `data/metadata/` copy (Apr 26). The pre-fix "4" was not observable in either local copy (both already 5), consistent with the data being correct all along.

**Impact on computed artifacts: none.** Because the features CSV always carried `teeth=5` for D39, every teeth/TPF-dependent result already used the correct value — the clean 1-60 PSD/JS faceting (D39 under `teeth=5`), TPF/chatter/harmonic metrics, and D39's mode label. No recomputation required. D39 is clean (1-60), has no noisy mirror, and is not in the 1-30 pooled-calibration ladder set, so the noisy 61-120 run and the pooled calibration are unaffected regardless.

**Loose ends:** (1) two copies of `Dataset descriptions.xlsx` exist (new at repo root, older under `data/metadata/`) — reconcile to one canonical copy; the feature CSVs live under `data/metadata/`, so that is the natural home. (2) If any feature CSV is ever regenerated from the spreadsheet, use the new root copy. (3) Optional belt-and-suspenders: confirm the GL `Dataset/features_all_datasets.csv` also has D39 `teeth=5` (it should — the earlier metadata verification recorded "Dataset 39: 5-tooth case").

## 2026-06-03: Pre-Phase-3 audit + leakage-safe data-prep module

Ran a max-effort, evidence-based readiness audit before any Phase 3 / TKAN-PINN modeling, then closed the two blockers it found with a new `src/pri2m/phase3/` package built by **three parallel TDD subagents** + main-thread integration. All work is LOCAL; transfer with `bash sync_to_hpc.sh`. No new runtime deps (numpy/pandas only), so no `pip install` on GL. Local suite is now **128 passing** (was 68).

### Audit findings (verified against the actual data, not the docs)
- Plumbing is solid: row counts internally consistent (clean 18,669 + noisy 15,050 = 33,719 = `noise_metrics.csv` rows; noisy 15,050 = 2×7,525 = pooled `n_paired`; clean 31-60 = 11,144); RU4 tier rates 1.6/0.3/18.7/95.5 reproduce EXACTLY from raw `mode_labels.csv`; `b_over_blim = b_mm/blim` to 1e-13; 0 duplicate `(dataset,file)` keys; 0 label disagreements features↔mode_labels.
- **BLOCKER 1 — mirror leakage.** Clean D and noisy D+60/D+90 share **100%** of `(omega,b_mm)` operating points (same cuts + noise). The configured `phase3_dev=[1,4,29,35,39,91]` contained the pair (1, 91) — 91 is the 5% mirror of 1 — so a naive split leaks test operating points into train.
- **BLOCKER 2 — 700 untrustworthy unstable labels.** 700 rows (2.1%) at omega 1000-1500 have NEGATIVE `blim` (-16.3 to -2.2, physically impossible; below the stability-boundary grid that starts at 2000), `b_over_blim`=NaN, yet stored `label`=1. That label is not derivable from `int(b/blim>=1)`. The `low_speed_edge` mode that should catch them is inactive (omega gate off), so they flowed in with a usable `noise_score` but a bad target. Spread across 11 datasets (31,57,30,90,120,56...). Same root cause as the open blim-calibration item.
- **Skew/hygiene.** `noise_score` heavy-tailed (mean 1.6, q99.9 34, max 190; driven by scale-unstable `wavelet_mad_sigma_Forcex`) — unusable raw as a sample weight. 560 `metrics_unavailable` rows have NaN `noise_score` (disjoint from the 700). 6 kurtosis features are NaN on the full set (the RapidMiner/xgboost smoke saw 60 features only because it fills kurt NaN→0 on datasets 1-5; full set has 54 finite).
- Confirmed-open (consistent with RU4 caveats): production `promoted_metrics.json` predates the coverage gate (no `coverage_fraction`/`min_coverage`); v3 (D19-excluded) never ran.

### New module `src/pri2m/phase3/` (3 parallel TDD subagents)
- `splits.py` — `parent_group()` (D, D+60, D+90 → D), `add_parent_group()`, `grouped_folds(n_splits,seed)` (pure-numpy group K-fold over parent groups), `DEV_GROUPS=(1,4,29,35,39)` + `dev_split()`, `assert_no_mirror_leakage()`.
- `labels.py` — `flag_label_validity()` (adds `blim_valid`, `b_over_blim_valid`, `physics_usable`, `exclude_supervised`, `exclude_reason`), `supervised_mask()`, `summarize_exclusions()`.
- `weights.py` — `winsorize()`, `noise_sample_weights()` (1/(1+winsorized noise); 0 where noise_score NaN; positive weights renormalized to mean 1), `impute_kurtosis()` (KURT_COLS→0 + `kurt_imputed` flag).
- `data_prep.py` + `scripts/build_phase3_dataset.py` — compose into one table + fold spec; CLI writes `results/phase3/{phase3_dataset.csv, phase3_folds.json, run_metadata.json}`.
- Tests: `tests/test_phase3_{splits,labels,weights,data_prep}.py` (60 new; 128 total).

### Extra bug caught during integration (keep in mind)
The subagent's first `assert_no_mirror_leakage` flagged shared `(omega,b_mm)` GLOBALLY. But **64.4% of clean operating points are reused across distinct datasets** (datasets 1 and 29 share ALL 102 of dataset 1's points) because different experiments sweep the same speed/depth grid under different tool/workpiece dynamics — that is NOT leakage. Left in, it would block almost every valid grouped split. Fixed to assert on `parent_group` only; corrected the agent's test to match the data-backed semantics.

### Real-data run (`python scripts/build_phase3_dataset.py`)
- 33,719 rows → 1,260 excluded (700 negative-blim + 560 metrics_unavailable, disjoint), 32,459 supervised-usable, 33,019 physics-usable.
- `sample_weight`: positive mean 1.0, **max 1.62** (raw `noise_score` max was 190 — skew gone).
- Folds: `leakage_verified: True`, 60 parent groups across 5 balanced folds; **datasets 1/61/91 all land in fold 2** (the original (1,91) leak is gone). `dev_groups` corrected to `[1,4,29,35,39]`.

### Re-transfer manifest (local → GL) — `bash sync_to_hpc.sh`
- src: `src/pri2m/phase3/__init__.py`, `splits.py`, `labels.py`, `weights.py`, `data_prep.py`
- scripts: `build_phase3_dataset.py`
- tests (only if running pytest on GL): `test_phase3_splits.py`, `test_phase3_labels.py`, `test_phase3_weights.py`, `test_phase3_data_prep.py`
- `handoff.md`
- Note: `results/phase3/*` is generated output — regenerate on GL with the CLI rather than rsyncing it.

### What this unblocks / still open
- **Phase 3 MLP baseline can start NOW** on `results/phase3/phase3_dataset.csv`: use `fold_assignments()`/`grouped_folds()` for leakage-safe CV, `sample_weight` for noise weighting, `supervised_mask` (or `sample_weight>0`) to filter the target, and `physics_usable` rows for the eventual physics residual.
- Still open (none block the MLP baseline or the JS spectral-loss head): rerun pooled calibration under the coverage gate to lock production; Altintas-Budak constants for the physics-residual head; the 700 negative-blim rows are excluded for now — a proper low-speed `blim` re-derivation would recover them.

## 2026-06-10: Phase 3 TKAN training — fold 0 done, tuning bundle is the next step

**Source of truth for all TKAN work is `docs/phase3_tkan/PROGRESS.md` (read its `## NEXT ACTION` first).** This is a pointer + status snapshot only.

The full Phase 3 TKAN harness is built, tested (full suite ~321 passing), and deployed to GL (`/nfs/turbo/coe-jaabell/PRI2M`); transfer via `bash sync_phase3_tkan_to_hpc.sh` from `SINDy_DDE/`. Work lives on branch `phase3-tkan`, uncommitted per policy.

**Where we are (job 51601910, spgpu/A40, fold 0, 20 epochs, `compile: true`):**
- Completed end-to-end in **8h22m ≈ 25 min/epoch** — torch.compile delivered its 3.4×, so per-fold cost is **~8 h** (not 30 h). The 5-fold sweep is now **~40 A40-h / ~8 h wall** if arrayed, *not* the old "~150 A40-h" estimate.
- Pre-flight blockers confirmed resolved in the run log: `leakage_verified: true`; exclusions = **700 `invalid_blim_low_speed` + 560 `metrics_unavailable`** (the negative-blim rows are *excluded*, not mislabeled).
- **Result is MARGINAL:** best epoch 6 → val AUC **0.882** / MAE 0.773; final epoch 20 → AUC **0.854** / MAE 0.911. TKAN **matches but does not beat** the MLP baseline (~0.87 AUC), and does so on 4×-pooled input.
- **Training is UNSTABLE:** train_loss bounces, val AUC swings 0.69–0.88, peaks at epoch 6 then never sustains → `lr=0.003` too hot, no LR schedule / warmup / grad-clip.

**Decision — DO NOT launch the 5-fold sweep yet.** Next session implements a toggleable, ablatable tuning bundle (TDD), reruns fold 0, confirms val AUC *stabilizes and sustains* ≥0.88 on `best_auc.pt`, then sweeps `--array=0-4`:
1. **Process-param side input** (Gao JMSE-2025's one robust win — see `docs`/memory `gao_milling_ml_paper`): a small MLP side-branch on per-file operating conditions. **FEED `omega`/`tpf`/`tau` only; NEVER `b`/`b_mm`/`blim`** — the label is `b_over_blim = b/blim`, so depth/blim would leak the target (enforce in code + a test).
2. **Cosine LR schedule + warmup** (trainer is plain fixed-lr Adam at `src/pri2m/training/phase3_trainer.py:272`).
3. **Gradient clipping** (`clip_grad_norm_` ~1.0).
4. *(optional)* AdamW + `weight_decay` ~1e-4.

Compare/deploy on **`best_auc.pt`** (highest-AUC epoch), not `best.pt` (lowest-MAE) — the headline metric vs the MLP is AUC. The XAI path takes `--checkpoint .../best_auc.pt`.

**Update (2026-06-10, later): the tuning bundle above is IMPLEMENTED and locally verified** — full suite **370 passed / 1 skipped**, 0 regressions, uncommitted. All four pieces are flag-gated: `side_input: true` + `side_hidden: 16` (omega/tpf/tau only; `b`/`b_mm`/`blim` rejected at model construction with tests), `lr_schedule: cosine` + `warmup_epochs: 2` (lr 0.003 is now the warmup peak), `grad_clip: 1.0`, `optimizer: adamw` + `weight_decay: 1e-4`. The MLP control is unaffected (a YAML-enabled side_input is auto-dropped for `MODEL=mlp`), and `xai_phase3.py` now loads side-input checkpoints correctly. **Next: `bash sync_phase3_tkan_to_hpc.sh`, then fold 0 only (`sbatch --array=0 --export=ALL slurm/phase3_train_gpu.sbatch`, ~8 h); sweep `--array=0-4` only if val AUC sustains ≥0.88 on `best_auc.pt`.** Details + fallbacks in `docs/phase3_tkan/PROGRESS.md` NEXT ACTION.

**Update (2026-06-10, incident): the first bundled fold-0 launch (job 51641089) crashed** at the first validation pass — `roc_auc_score: Input contains NaN`. Root cause was NOT the data (table audit: 0 non-finite omega/tpf/fn): a KAN-GRU gradient blow-up produced a NaN gradient and the new `clip_grad_norm_` call **NaN-scaled every gradient** (`clip_coef = max_norm/NaN`, default `error_if_nonfinite=False`) → one optimizer step → all-NaN weights → sklearn crash. Hardened (TDD, full suite green): the trainer now skips-and-counts non-finite batches (`nonfinite_batches` per epoch in MLflow + stderr warning, RuntimeError if an epoch steps nothing), and `per_file_severity_metrics` excludes-and-reports non-finite files (`n_nonfinite_files`) instead of raising. The TF32 warning in the .err is expected (pure-fp32 is deliberate). **Re-sync + relaunch fold 0; if `nonfinite_batches` is frequently > 0, halve the peak lr to 0.0015 in the yaml.** Forensic confirmation on the dead run: its `checkpoints/` dir should be empty (died at epoch 0 before the first checkpoint write).


**Update (2026-06-11): fold-0 rerun PASSED the gate (job 51651316, run `931af482deca48b9bb2d1cd1689832d9`, 20 epochs / 8h24m).** `best_auc.pt` = epoch 6, **val AUC 0.9388** / acc 0.840 / MAE 0.611; `best.pt` = epoch 19, MAE 0.548 / AUC 0.927 — TKAN now clearly beats the MLP baseline (~0.87) and the pre-bundle fold-0 peak (0.882). The gradient blow-up recurred mid-run (epochs 7–11; worst epoch skipped 1661/3334 batches, train_loss spiked to 143.9) but the new guards turned it into a recoverable detour instead of a crash — weights stayed finite and the run climbed back to AUC 0.927 — at the cost of never re-attaining the epoch-6 peak. Every unstable epoch ran at cosine lr ≥ 0.0015 and every epoch at ≤ 0.00124 was clean, so the pre-agreed fallback is applied (uncommitted): **yaml peak `lr: 0.003 → 0.0015`**. **Next: re-sync, rerun fold 0 at 0.0015 (expect zero skipped batches, best_auc ≥ ~0.93, peak possibly later than epoch 6), then the 5-fold sweep + `--no-side-input` ablation.** The existing 0.9388 `best_auc.pt` is already usable for a preliminary XAI pass. Details in `docs/phase3_tkan/PROGRESS.md` NEXT ACTION.


**Update (2026-06-12): full 5-fold TKAN sweep PASSED (lr 0.0015 stability config).** Jobs: fold 0 `51670644`, folds 1–4 array `51711709`; 20 epochs/fold, side-input + compile on. **Stability fix validated — 0 skipped/non-finite batches across all 100 epoch-folds** (the `lr 0.003 → 0.0015` fallback fully eliminated the KAN-GRU gradient blow-up that cost the 0.003 run 1714 batches). 5-fold `best_auc.pt` = **0.955 ± 0.022** (folds 0–4: 0.960 / 0.912 / 0.975 / 0.961 / 0.970) — every fold beats the MLP baseline (~0.87, min margin +0.042) and the 0.88 gate; fold 0 @0.0015 (0.960) even beats its own @0.003 (0.939). Severity head remains the weak, heavy-tailed head (MAE 0.88±0.65, RMSE 4.37±2.65; fold 4 outlier-dominated 2.13/9.0). **All five folds ran `side_input: true`, so the `--no-side-input` ablation is still OPEN.** Next: commit the validated bundle (on user word), run the ablation, pull the matched 5-fold MLP per-fold AUC (job 51442237) for the paper baseline table, XAI on the strong folds. Full detail + commands in `docs/phase3_tkan/PROGRESS.md` NEXT ACTION.

## 2026-06-22: Held-out eval on datasets 121–125 PASSED + first git commit of the project
**Held-out generalization test (GL job `51949469`, array 0–4; eval-only, no retraining).** Each of the five 5-fold `best_auc.pt` checkpoints was run over the new datasets, which were never in any training fold — the test the 0.955 headline needed to retire its "val-selected, not a held-out test" caveat.

- **Held-out AUC = 0.908 ± 0.013** on all five new noisy datasets (121–125; 957 files / 30789 windows) and **0.916 ± 0.011** on the OOD probe (122–125; 855 files — the genuinely novel parameterizations with no parameter twin in 1–120, 122/124/125 carrying process damping). Per-fold AUC (121–125 / 122–125): `61c67bdd` 0.891/0.906, `c21e58dc` 0.919/0.925, `a654129d` 0.899/0.903, `3a40b715` 0.919/0.927, `09378839` 0.913/0.920.
- **Every one of the five checkpoints, on both sets, still beats the MLP baseline (~0.87)** — weakest single number 0.8907. `n_excluded = 0`; per-file scoring, so directly comparable to the in-distribution 0.955.
- **OOD ≥ full set on every fold** — process damping does not break the model; the purely-novel datasets are not the hard ones (121, the noisy twin of clean 14, slightly lowers the average).
- **The val-selected caveat is retired:** the 0.955 → ~0.91 drop is the honest, modest generalization gap.
- **The severity head does not transfer:** MAE 3.3–4.8 / RMSE 7.8–11.2 vs in-distribution MAE 0.88. Chatter detection is the deliverable and it holds; severity regression on new operating points does not. Per-fold acc (0.72–0.84) is a 0.5-threshold artifact (AUC unaffected).
- Full per-fold table + takeaways: **`docs/phase3_tkan/heldout_eval_121_155.md`**. Source JSONs local at `results/phase3/heldout/<run>_{121_125,122_125}.json` (gitignored; numbers preserved in the doc).
- **126–155 clean replication eval (126≡1 … 155≡30) was NOT in this job → still OPEN.** Either run the same `eval_heldout_phase3.py` sweep over a `features_datasets_126_155.csv` table (reproducibility half of the story) or confirm it's deprioritized.

**First git commit.** On explicit user instruction, after verifying the full suite green (**409 passed / 1 skipped**), the entire `SINDy_DDE` project was committed for the first time (it had never been tracked — the parent repo only held the old Dataset 1 / notebook content). 193 source/test/script/doc/config files; the nested `.gitignore` excludes the 3.8 GB `archive/`, venvs, and all generated artifacts (`results/`, `*.pt`, `*.npz`, `*.png`). Excluded from this commit: the `~$…docx` Word lock file and the peripheral `summer_research/`, `rapidminer/`, and `GLC Logs/` directories. Commit carries the user's name only (no Claude co-author), per project policy.
