# PSD/Jensen-Shannon Phase 2 Notes

This slice is intentionally importable without any mounted raw-data filesystem.
The core module accepts in-memory arrays for `xtpos`, `ytpos`, `Forcex`, and
`Forcey`.

Defaults:

- Welch PSD: Hann window, `nperseg=2048`, 50 percent overlap, constant detrend.
- Short signals: cap to the largest valid power-of-two window; reject signals
  shorter than 256 samples.
- Analysis band: 0 to `5*tpf`.
- Divergence: Jensen-Shannon divergence in bits as a bounded Phase 3 spectral
  loss candidate, plus directional KL terms for post-hoc reporting.

Stable-reference JS rule:

- The production stable-reference curve uses `REFERENCE_MODE=low_b_ratio`.
- Reference samples are selected by physics only: the lowest 10 percent of
  `b_over_blim` within each `signal x teeth x cut_dir` facet.
- A hard stability floor is required: reference samples must satisfy
  `b_over_blim <= 0.80`.
- If a facet has fewer than `min_per_label` samples under that floor, the row is
  marked `insufficient_stable_reference` rather than using "least unstable"
  samples.
- `REFERENCE_MODE=label_stable` is retained only as a methods-comparison mode.
  It uses `label=0` and must not feed final reporting or Phase 3 spectral-loss
  artifacts.
- The lowest `b_over_blim` bin overlaps the reference set by design, so its
  divergence is an anchor rather than an independent validation point. Evaluate
  the curve shape above the reference band, especially the rise or inflection
  across `b_over_blim = 1`.

`b_over_blim` provenance:

- In the local feature CSVs, `b_over_blim` is exactly `b_mm / blim` to
  floating-point tolerance, and `label` is exactly `int(b_over_blim >= 1.0)`.
- The SINDy preprocessing lineage loads `stability_boundary<dataset>.h5`, whose
  first dataset is `(omega, blim)`, then interpolates `blim` at the file's
  spindle speed before computing `b / blim`.
- Run `scripts/verify_b_over_blim_provenance.py` on Great Lakes with
  `--data-root` to spot-check the feature CSV against the raw stability-boundary
  H5 files before treating full-60 JS curves as final.

Reference-rule smoke test:

```bash
python scripts/check_js_reference_rule.py
```

CLI dry run example:

```bash
python scripts/run_psd_js.py --summary-csv "data/metadata/features_all_datasets.csv" --dry-run
```

Optional signal NPZ files should contain any subset of the canonical signal keys
plus either `fs`/`sample_rate` or a `time`/`t` vector. Recognized signal aliases
are normalized to `xtpos`, `ytpos`, `Forcex`, and `Forcey`.

Dataset IDs 44-48 are tagged as `inspect_44_48_overcalled_cut_window` in the
sanity helpers and CLI dev-set output so those overcalled cuts remain easy to
filter.
