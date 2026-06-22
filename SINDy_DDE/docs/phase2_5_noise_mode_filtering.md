# Phase 2.5 Noise Characterization and Mode-Aware Filtering

Date: May 25, 2026

## Goal

Phase 2.5 answers Prof. Abell's feedback on noise:

- identify how much noise is present,
- distinguish forced cutting, chatter, and measurement/process noise,
- filter conservatively only when physics is preserved,
- assign a mode/confidence label that can feed the Phase 3 T-KAN-PINN and XAI layer.

This is a support layer for the Phase 3 spectral loss and DiCE/XAI workflow. It is not a new parallel research thread.

## Scope Guardrail

Phase 2.5 must finish before the JS spectral loss is unlocked in Phase 3. It should run in parallel with the early Phase 3 baseline because the baseline only needs hardcoded `tau` and MSE trajectory loss.

In scope:

- per-file noise metrics on existing NPZ signal bundles,
- clean/noisy mirror calibration,
- metadata-supervised mode assignment,
- conservative filtering candidates and acceptance rules,
- confidence weights for spectral loss and XAI.

Out of scope for the first pass:

- full real-time spectrogram/chatter-onset detection,
- final signal-only deployment without metadata,
- full DiCE feasibility optimization,
- label-confidence modeling from Altintas-Budak perturbations,
- model-output uncertainty calibration.

These are tracked in the final section.

## Three-Way Signal Decomposition

Each file should be interpreted as a mixture of:

1. Forced cutting response
   - tooth-pass frequency (`tpf`) and harmonics,
   - expected deterministic cutting periodicity,
   - should be preserved by filtering.

2. Chatter response
   - energy near structural/natural frequency `fn`,
   - regenerative sidebands near `fn +/- k*tpf`,
   - broadband lift or coherent instability features,
   - should be detected and preserved, not filtered away.

3. Measurement/process noise
   - broadband or incoherent energy not aligned with TPF, harmonics, `fn`, or cross-channel coherence,
   - target of filtering.

The filtering objective is to reduce only bucket 3 while preserving buckets 1 and 2.

## Mirror Pairing Convention

The noisy datasets in `features_noisy_datasets.csv` follow a two-level noise ladder against the clean datasets:

- noisy IDs 61 to 90 use `noise_std = 1.0` and pair as `clean_id = noisy_id - 60`. So Dataset 61 mirrors clean Dataset 1.
- noisy IDs 91 to 120 use `noise_std = 5.0` and pair as `clean_id = noisy_id - 90`. So Dataset 91 also mirrors clean Dataset 1.

Identity was verified on the metadata: matched IDs share `omega`, `b_mm`, `blim`, `b_over_blim`, `fn`, `teeth`, `cut_dir`, and `diameter_mm`. Only `noise_std` differs. This gives a clean / 1 percent noise / 5 percent noise triple per clean dataset, which is a stronger calibration signal than a single noisy mirror.

## First Checkpoint: Dataset 1 vs Datasets 61 and 91

Before accepting any metric into the noise-score vector, it must pass a simple mirror test against both noise levels:

- compute metrics on clean Dataset 1, 1 percent noisy Dataset 61, and 5 percent noisy Dataset 91,
- pair by `(omega, b_mm, file index within dataset)`,
- require the score distribution to shift upward monotonically across the three noise levels.

Concrete acceptance statistic (pick one and document the choice):

- per-file paired difference: `mean(noisy_score - clean_score) > 2 sigma` of the within-Dataset-1 file-level noise-score distribution, or
- distribution-level fallback when per-file pairing is incomplete: Mann-Whitney U with `p < 0.01` against the clean baseline.

A metric that does not separate clean from 5 percent noisy, or fails to order 1 percent below 5 percent, is rejected from the noise-score vector. This cheap gate prevents the metric list from becoming bloated and avoids cherry-picking later.

## Noise Metrics

Compute these per file, per signal, and where appropriate per signal pair.

Metrics are grouped by family so the agreement rule below cannot double-count correlated estimators.

PSD family:

- noise floor ratio: high-frequency or off-physics noise floor divided by total signal energy,
- off-harmonic energy ratio: energy outside TPF harmonics and outside the `fn` chatter band,
- spectral flatness: higher for broadband noise,
- spectral entropy: higher for diffuse/noise-like spectra.

Wavelet/temporal family:

- wavelet MAD sigma: independent time-frequency noise estimate from the finest detail coefficients,
- autocorrelation decay rate.

Default wavelet settings (pinned for reproducibility): Daubechies 4 (`db4`), level 1 detail coefficients, symmetric extension, MAD threshold `sigma = median(|d1|) / 0.6745`.

Cross-channel family:

- coherence between `Forcex`, `Forcey`, `xtpos`, and `ytpos`,
- coherent chatter score in the `fn` band,
- incoherent residual score as a noise indicator.

Chatter-specific family (used to separate chatter from generic broadband noise, not to vote on noise level directly):

- chatter-band energy near `fn`,
- spectral kurtosis in the `fn` band,
- sideband energy around `fn +/- k*tpf`,
- cyclostationarity at the TPF period.

Agreement rule:

- no single metric is enough,
- production mode assignment uses the promoted mirror-ladder calibration, not raw cross-unit thresholds,
- a high-noise label requires at least two of the three calibrated families (PSD, wavelet/temporal, cross-channel) to vote yes, with no present family voting against,
- a family is considered "present" only if calibration promoted at least one metric for it; families with no promoted metrics abstain rather than block, because the alternative (strict three-family AND) silently fails closed whenever a dataset's noise model leaves a family with no promoted metrics (for example, simulation noise that is correlated across channels promotes nothing in the cross-channel family, while real ADC noise tends to drop coherence so the same family lands in the inverse-indicator list instead),
- the uncalibrated legacy fallback remains stricter: PSD, wavelet/temporal, and cross-channel all must vote yes before `broadband_noise_dominated` is assigned; this fallback is for smoke tests and missing-calibration diagnostics, not production interpretation,
- two PSD-family metrics agreeing is not enough by itself, because spectral flatness and entropy correlate on the same files for the same reason,
- chatter-family metrics determine whether energy near `fn` is structured chatter or broadband noise, but they do not count as votes on the overall noise level,
- prioritize metrics that separate clean/noisy mirror pairs at both noise levels.

Per-file computational budget: target under 0.5 seconds per file on Great Lakes for the full metric vector, so the metric list cannot grow unbounded.

## Mode Assignment

Mode labels are metadata-supervised in the first pass, with a signal-only branch kept as validation/future deployment.

Physical metadata features:

- `teeth`,
- `cut_dir`,
- `omega`,
- `b_over_blim`,
- `fn/tpf`,
- `ktc`,
- damping,
- diameter.

Spectral features:

- dominant peak location,
- TPF harmonic energy,
- chatter-band energy,
- sideband energy,
- broadband noise floor,
- spectral entropy/flatness,
- force/position coherence.

Initial mode labels:

- `tpf_dominated_clean`,
- `near_boundary_transition`,
- `chatter_dominated`,
- `broadband_noise_dominated`,
- `mixed_tpf_chatter`,
- `low_speed_edge`,
- `metrics_unavailable`.

Decision rule:

- v1 uses rule-based thresholds on (chatter-band energy near `fn`, TPF harmonic energy, noise floor ratio, `b_over_blim`),
- the broadband-noise rule uses only promoted mirror-ladder metrics, direction-checked as `clean < 1 percent noise < 5 percent noise`, with per-metric clean reference percentiles instead of raw cross-unit thresholds,
- chatter and TPF rules remain physics-thresholded until a larger mode-calibration pass is available,
- v2 swaps in a trained classifier (gradient boost or shallow MLP) using the same feature set if rule-based labels disagree visibly with the JS-curve evidence,
- mode labels carry a confidence score so downstream stages can downweight ambiguous cases.

Sampling-quality flag:

- files with fewer than eight samples per tooth-pass period get `sampling_quality_flag = undersampled_tpf`,
- this is not the same as `low_speed_edge`; `low_speed_edge` is reserved for rows outside a trusted spindle-speed or stability-boundary grid,
- rows with too many missing metrics get `metrics_unavailable` instead of being forced into a physical mode.

Signal-only validation branch:

- estimate TPF from autocorrelation or cepstrum,
- estimate dominant periodicity,
- compare signal-inferred TPF/mode against metadata-derived values,
- treat teeth count, cut direction, and immersion inference as future-deployment work unless the validation is strong.

## Filtering Candidates

Filtering is not run until the noise metrics and mirror calibration work.

Candidate filters:

- spectral gating against a mode-specific noise floor,
- conservative band-aware smoothing,
- wavelet denoising using MAD-estimated sigma,
- optional mode-specific thresholds.

Filtering must keep raw and filtered outputs side by side. Never overwrite the original NPZ signals.

## Filtering Acceptance Criteria

A filtered signal is accepted only if all checks pass:

1. Noise score falls into the clean baseline range for the same mode.
   - Suggested criterion: below the 90th or 95th percentile of clean stable cuts in the same mode.

2. TPF structure is preserved.
   - TPF peak location should not move.
   - TPF harmonic energy should remain within tolerance.

3. Chatter-band physics is preserved.
   - Energy near `fn` and sideband structure must not be erased.

4. Stable-reference JS trend survives.
   - Low/flat below `b_over_blim = 1`.
   - Rise or inflection after `b_over_blim = 1`.

5. Noisy mirror moves toward clean twin.
   - Filtered noisy PSD/JS should be closer to the clean mirror than the unfiltered noisy PSD/JS.

If any check fails, filtering is rejected for that mode and the file is tagged as noise-limited rather than force-filtered.

## Phase 3 Integration

Phase 2.5 outputs should feed Phase 3 as confidence and conditioning, not as a hard data deletion rule.

Recommended Phase 3 usage:

- include mode embedding or mode label as a conditioning input,
- include per-file noise score,
- use noise confidence as a sample weight for spectral loss,
- downweight noise-limited files instead of excluding them outright,
- keep MSE trajectory baseline independent of Phase 2.5.

Spectral loss unlock rule:

- do not wire JS spectral loss into Phase 3 until Phase 2.5 has a validated noise score and filtering acceptance rule.

## XAI and DiCE Integration

The final T-KAN-PINN should support DiCE-style counterfactuals over controllable machining parameters only.

Recommended XAI behavior by mode:

- `tpf_dominated_clean`: suggest efficiency-oriented moves only if stability margin allows.
- `near_boundary_transition`: suggest conservative spindle/depth changes and report low margin.
- `chatter_dominated`: suggest spindle speed and depth-of-cut changes toward stable behavior.
- `broadband_noise_dominated`: refuse or defer at the per-file level with explanation "measurement too noisy for a reliable recommendation," and emit a dataset-level "sensor needs attention" rollup when a large fraction of files in a dataset fall in this mode, so the operator sees an aggregate signal instead of only a long string of refusals.
- `mixed_tpf_chatter`: suggest changes with reduced confidence and request validation.
- `low_speed_edge`: report that the row sits outside part of the raw boundary grid where relevant.
- `metrics_unavailable`: refuse file-level recommendations and route the file to data-quality review.

DiCE must report the noise score and model confidence alongside any recommendation.

## Deliverables

Phase 2.5 deliverables:

- `noise_metrics.csv`: per-file and per-signal noise/chatter/mode metrics.
- `noise_score_summary.csv`: dataset/signal/mode summaries.
- `mode_labels.csv`: per-file mode labels and confidence.
- `mirror_calibration_1_vs_61_vs_91.csv`: first checkpoint comparing clean Dataset 1 to its 1 percent and 5 percent noisy mirrors.
- `mirror_calibration_pooled.csv`: full clean/noisy mirror comparison across every available clean / 1 percent / 5 percent ladder after datasets 61-120 are processed.
- noise-score plots by dataset, signal, and `b_over_blim`.
- filtering evaluation report for Dataset 1 vs 91.
- final filtering decision rule and rejected-filter cases.

## Implementation Order

1. Metrics-only pass on existing NPZs.
   - Start with Dataset 1 and noisy Dataset 91.
   - Promote only metrics that separate 91 from 1.

2. Full clean/noisy mirror calibration.
   - Requires noisy datasets 61-120 extraction.
   - Use pooled calibration across every available `(D, D+60, D+90)` mirror ladder; the single `1/61/91` ladder is only the pilot checkpoint.

3. Metadata-supervised mode assignment.
   - Validate mode labels against known metadata and PSD/JS behavior.

4. Conservative filtering experiments.
   - Start with spectral gating.
   - Add wavelet denoising only if needed.

5. Acceptance report.
   - Apply the five acceptance criteria.
   - Decide which modes are safe to filter and which should be downweighted/refused.

6. Phase 3 handoff.
   - Export mode/noise confidence for spectral loss weighting and DiCE/XAI refusal logic.

## Out Of Scope, Tracked

These are important, but should not block Phase 2.5.

1. Within-cut non-stationarity.
   - Current PSD summarizes one file as one spectrum.
   - Future work: spectrogram or short-time JS to detect stable-to-chatter transition within a cut.

2. Label confidence from Altintas-Budak calibration.
   - `b_over_blim` labels may need confidence weights based on margin to lobe and sensitivity to `ktc`/damping perturbations.

3. DiCE feasibility constraints.
   - Counterfactuals must respect spindle limits, depth limits, tool deflection, material removal rate, and machine constraints.

4. Causal vs spurious feature sensitivity.
   - Perturb controllable variables and check the model response matches machining physics.
   - Perturb nuisance variables and verify the model does not overreact.

5. Tool wear and across-cut non-stationarity.
   - Check whether file index or cut order correlates with spectral features at fixed operating settings.

6. Model-side uncertainty.
   - Pair file-side noise confidence with model-output uncertainty using calibration, dropout, or a small ensemble.
