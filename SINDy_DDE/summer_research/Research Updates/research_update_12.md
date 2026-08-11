---
title: "Research Update 12"
date: "July 19 to 25, 2026"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

<!--
Covers July 19 to 25. Closed 2026-07-30. Rewritten shorter 2026-08-02.

Every scientific number traces to a returned GL manifest or a summary.json
fetched from Great Lakes. No number is invented.

NOTHING IN THIS UPDATE IS OPEN.

REWRITE 2026-08-02: cut for length and plain language at Yash's request, then
cut again. No number, negative result, caveat or correction of the record was
removed. Every numeric token here also appears in the version sent before it
(checked mechanically, superset test). What went: definitions of terms Abell
already knows (fold, held-out, saliency map), repeated statements of the
sentinel defect class, the three-submission narrative of the TSR control, the
worked frequency-mislabelling example, the first run's defect description now
that the conclusion is structural, and the old conclusion's restatement of
earlier sections.

Record of corrections carried by this update, all still present in the text:
  - Sec 2 corrects Update 11 Sec 3. The published reason for the TSR negative
    ("underpowered, only four windows") is wrong; the statistic under it was
    reporting its own search floor on 550 of 640 windows. Update 11's units
    were wrong too (peak_lag and tau_raw share an axis).
  - Sec 2 replaces an unreproducible number from the first U12 draft: an exact
    harmonic in 60 of 90 windows (67%) could not be reproduced under any
    natural definition. Integer divisibility gives 44 of 90, a 5% ratio
    tolerance 72. Divisibility is what is reported, with its definition inline.
  - Sec 3: the TSR delay-lock is a construction artifact. Source manifest
    results/phase3/xai/tsr_control/fold2_w512/tsr_control_manifest.json
    (provenance.complete = true, 3 conditions x 24 windows, paired indices).
    Paired statistics come from per_window rows, not from the printed summary,
    whose VERDICT line used the uncorrected rule.
  - Sec 3: the verdict rule as implemented was the ratio test alone. All three
    missing checks (significance gate, minimum windows, shuffled arm) are now
    in tsr_control_verdict() with tests; one test pins the returned fold2_w512
    numbers to the same verdict. Verdict unchanged.
  - Sec 4: both band-notch defects fixed and re-run on GL (jobs 55534785-89,
    all five folds returned and read). Still no usable number, now for a
    diagnosed structural reason. NOTE: the returned manifests predate the
    degenerate-denominator guard, so max_rel_err is unreadable and rel_err_p90
    is the figure quoted. A re-run would make the headline legible without
    changing the conclusion.
  - Sec 5 CORRECTION: the quantile head's classification AUC was called "a
    real improvement" pending held-out evaluation. Withdrawn. Run-to-run drift
    on an identical configuration is up to 0.0092 AUC against a 0.0062 gain.
    The same paragraph called the duplicate sweeps "two independent five-fold
    runs"; they are the same config and seed submitted twice, and 4 of 5
    quantile folds did not reproduce. The 0.0092 observed fold-level difference
    is applied to the side-input ablation in the same section for the same reason.
  - Sec 6: duplicated run directories canonicalised by matching the published
    AUC (scripts/canonicalise_runs.py, registry at
    results/phase3/RUNS_CANONICAL.json). 32 directories: 15 canonical, 11
    superseded, 6 predating the side_input config key. 4 of the 10 duplicated
    folds reproduce the published number identically, so submission order
    breaks the tie and is labelled as a tie-break.
-->

## 1. Summary

This cycle answered three questions.

- **TSR does not provide new evidence that the network learned the delay.** The original metric was broken. A corrected metric found a delay lock, but the same lock was stronger in an untrained network, so it comes from the input construction.
- **The band-notch comparison cannot be run fairly as designed.** The chatter and tooth-pass bands cannot remove equal energy without overlapping, which would invalidate the comparison.
- **The side-input and quantile-head retrains did not produce a measurable improvement.** Their effects are smaller than the run-to-run differences measured in Update 14.

The main delay result is unchanged. It still rests on the three intervention tests and the network-free SINDy-DDE recovery.

## 2. The metric was reporting its own search floor

I ran TSR at full scale: 160 shards across five folds, all returned and merged. I expected the negative either to survive at scale or to dissolve. Instead the metric turned out to be broken.

`tr_peak_lag` searches lags 2 to N/2 for the lag at which the relevance signal most repeats itself. When the signal only decays and never peaks, the search comes back empty and the metric prints 2, the bottom of its own range. A 2 therefore means "no answer", and nothing in the output distinguishes it from a real measurement of 2.

550 of 640 windows, 85.9 percent, come back as exactly 2. The per-fold means of 2.6 to 10.0 that Update 11 reported are averages over a column that is mostly this no-answer value.

![The TSR delay metric. Panel A shows how many of each fold's 128 windows returned the search floor instead of a peak. Fold 3 resolved 32 windows and fold 0 resolved 6, so a per-fold mean of this statistic compares how much evidence each fold produced rather than how much delay it found. Panel B plots, for the 90 windows that did resolve, each window's peak lag against its own delay, both axes logarithmic. The 31 windows whose peak sits exactly on tau are drawn filled on the dashed line, and the rest scatter below it. That scatter is why the corrected statistic tests at tau instead of searching for a peak and asking afterwards where it landed.](../../results/figures/research_update_assets/u12_tsr_floor_and_lock.png)

The 90 windows that do resolve tell a different story. In 31 of them, 34 percent, the peak lands exactly on that window's own tau. Chance alone would put 3.34 windows there, giving p = 0.00005. Counting a peak as harmonic when either the lag divides tau or tau divides the lag, 44 of 90 qualify. The median ratio of `peak_lag` to `tau_raw` is 1.0000.

**This corrects Update 11, Section 3.** The published reason for the negative, "underpowered, only four windows", is wrong. The negative was unsound, because the statistic under it was mostly reporting a floor. I had the units wrong as well: `peak_lag` and `tau_raw` already share an axis, so the comparison against "about 22 steps after four-fold pooling" was on the wrong axis too. I am saying so here rather than quietly superseding the old text.

I rebuilt the metric, test-first. Every summary now carries the count of windows that resolved a peak, so a number derived from 90 of 640 is reported as such. The new delay-lock statistic asks how strongly the signal repeats at exactly lag tau and tests that against a null, so it has no floor to fall back on. And the floor value is excluded from every aggregate instead of pooled in silently.

The defect is worth naming as a class: a search returns a bound when it finds nothing, and that bound feeds an average that cannot tell a bound from a measurement. It is now a standing check in this branch's review brief, and it caught two more cases in Section 4.

## 3. The delay lock is built in, not learned

A randomly initialised network locks onto the delay harder than the trained network does.

I tested 24 windows from fold 2 under three conditions: the trained network at the true tau, the same architecture with re-initialised weights at the true tau, and the trained network at shuffled tau. The untrained model was verified to differ from the trained model, with a mean absolute weight difference of 0.426 across seven re-initialised submodules.

The decision rule was fixed before the run. If the untrained lock reached 80 percent of the trained lock, the effect would be treated as an input-construction artifact. An unmeasurable result returned UNDECIDED rather than passing.

| Condition | Mean lock statistic | Windows locked (z > 2) |
|---|---|---|
| Trained network, true tau | +2.81 | 79.2 percent |
| Untrained network, true tau | **+4.62** | **100 percent** |
| Trained network, shuffled tau | +0.88 | 16.7 percent |

On the same windows at the same delay, the randomly initialised network locks harder in 22 of 24 paired windows, with a mean margin of 1.80 and sign-test p = 3.6e-5. It clears the significance threshold in all 24 windows; the trained network clears it in 19. The untrained lock therefore exceeds the trained lock rather than merely reaching the 80 percent rule.

The shuffled condition does not rescue the result. The trained network does beat itself at shuffled tau, 19 of 24 windows, p = 0.0066, so the effect is specific to each window's own delay rather than to delays in general. That sounds like evidence for learning and is not, because the input construction is equally tau-specific: the delayed channel is the signal shifted by that window's own tau. Only the untrained arm separates learned from built in, and it answers "built in".

Two further details point the same way. The trained lock is stronger on stable windows, 3.52, than on chatter windows, 2.10, which is backwards for anything describing a regenerative mechanism. And the lock falls steadily as tau grows, correlation −0.64 trained and −0.43 untrained. That is a smoothness bias in the statistic: a synthetic curve with no delay structure at all registers as locked at every delay up to 16 steps, and this dataset's delays sit inside that range.

The existing model-randomization check still shows that the full fold-2 saliency map depends on the trained weights: its cosine similarity to the randomized map is 0.0995. The delay summary behaves differently because it also reads structure inherited from the input. This means that a statistic derived from a saliency map needs its own randomization control.

The final verdict rule now includes the significance gate, a minimum-window requirement, and the shuffled arm. The tau shuffle also returned the original tau to 2 of the 24 windows. Excluding those windows lowers the shuffled mean from 0.88 to 0.70 and does not change the decisive trained-versus-untrained comparison.

TSR is therefore a closed negative and will not enter the technical report as delay evidence. The 34 percent exact-lock result is confounded by the input construction. The main delay claim does not depend on attribution.

## 4. The band-notch contrast cannot be run as designed

The stressor suite got its spectral half this cycle: surrogate signals that preserve some spectral properties while destroying others, reliance metrics over them, and the band-notch contrast.

The contrast asks a sharper question than the existing fuzzers can. If the model's delay reliance is really about the regenerative mechanism, deleting the chatter band should cost it more than deleting an equal amount of energy from the tooth-pass harmonics. Matching the energy removed is what makes that fair, and it is harder than it sounds, because a notch either includes a spectral bin or it does not.

The first run was void. Its bands were aimed using a batch-mean sampling rate and chatter frequency across a batch spanning 37 distinct sampling rates from 12.3 to 38.3 kHz and eight chatter frequencies, so both arms deleted energy from physically arbitrary places. And its halfwidth solver, on exhausting its iteration budget, could return a candidate narrower than one FFT bin that removed exactly zero energy, which makes the relative error exactly 1.0 by arithmetic and pins a whole fold once the reported figure is a maximum.

**Both defects were fixed and the contrast was re-run. It still produces no usable comparison because the two perturbations cannot be energy-matched.**

Between 298 and 323 of each fold's 512 windows cannot be solved at all, so 58 to 63 percent of the sample is unusable, consistently across all five folds. Among the windows that do solve, ninety percent miss the energy match by between 97.5 and 97.8 percent, against a requirement of one percent. The cause is a structural asymmetry, which I confirmed on a synthetic window built to match the observed removed fractions. The chatter notch is one band and the least energy it can remove is about 2 percent of the signal. The tooth notch is three bands sitting on the strongest peaks in the spectrum, and the least it can remove is about 13 percent. The two achievable sets never meet anywhere near where the real removed fractions live.

The bind is worse than a tuning problem. Pushing the chatter fraction up to where the tooth side can match it means widening the chatter band until it reaches the tooth harmonics, which is what the safety guard forbids, because a band that swallows both deletes both conditions' evidence. The guard that makes the contrast meaningful is what makes it unmatchable. I tested the two obvious escapes, dropping from three tooth harmonics to one and lengthening the window from 1024 to 4096 samples, and in every combination the tooth floor still sits two to four times above the highest chatter fraction the guard allows.

On four of the five folds, the chatter notch cost more accuracy than the tooth notch, by 0.0042 on average. That direction would cut against the delay interpretation, but it cannot be used because the perturbations were not matched. Their batch averages differ by 3.9 to 7.4 percent in removed energy, while the per-window mismatch is around 97 percent. Across five folds, the correlation between the remaining energy difference and the accuracy gap is −0.74 and is not significant.

Evaluated separately, deleting roughly six percent of the spectrum at either location costs at most 0.006 AUC against a baseline near 0.99. This suggests that the model is not highly sensitive to either perturbation, but it does not rescue the failed paired comparison.

## 5. Two retrains, both negative

The side-input ablation settles a previously unresolved question. Every T-KAN fold had trained with the Gao-style process-parameter side inputs on, so their contribution had never been isolated, and the permutation proxy was too weak to settle it at 0.002 to 0.009. Training once with them on and once without is the real test:

| Fold | side inputs on | side inputs off | delta |
|---|---|---|---|
| 0 | 0.9597 | 0.9626 | −0.0029 |
| 1 | 0.9122 | 0.9080 | +0.0042 |
| 2 | 0.9752 | 0.9747 | +0.0005 |
| 3 | 0.9605 | 0.9546 | +0.0059 |
| 4 | 0.9695 | 0.9632 | +0.0063 |
| **mean** | **0.9554** | **0.9526** | **+0.0028** |

The mean difference is 0.0028 AUC. That is smaller than 0.0092, the largest fold-level difference later observed between two runs of the same configuration (Update 14, Section 3), and every per-fold delta is also below 0.0092. The difference is within the fold-to-fold spread of 0.022, and fold 0 is slightly better without the side inputs. This experiment therefore finds no measurable classification benefit from the side inputs.

The quantile severity head does not fix severity. Severity is the one output never hardened: in distribution it gives MAE 0.88 with RMSE 4.37, and it does not transfer out of distribution at all, at held-out MAE 3.3 to 4.8. My hypothesis was that the heavy tail came from fitting a conditional mean, and that a pinball-loss quantile head would behave better. I measured on the median column so the comparison is like-for-like:

| Severity head | MAE | RMSE | RMSE/MAE |
|---|---|---|---|
| point head (published) | 0.88 | 4.37 | 5.0 |
| quantile, first run | 0.880 | 5.01 | 5.7 |
| quantile, second run | 0.830 | 4.48 | 5.4 |

The median prediction improves MAE by five hundredths, while RMSE is slightly worse and the RMSE/MAE ratio remains high. The quantile outputs still feed the conformal layer, but the median prediction does not remove the heavy-tailed errors. A broader evaluation of the quantile head would also need pinball loss and quantile calibration.

Classification AUC on the quantile runs came in at 0.9616 mean against the published 0.9554, every fold equal or better.

**Correction.** An earlier draft called the classification AUC gain a real improvement. It is not established. The two copies used the same configuration and seed, yet four of the five quantile folds did not reproduce. The largest observed fold-level difference is 0.0092 AUC, while the reported gain is 0.0062. Update 14, Section 3 gives the full repeatability analysis.

## 6. Obstacles

The two retrains were accidentally submitted twice, costing about eighty GPU-hours. The duplicate runs later provided the repeatability data used in Update 14. The run registry now identifies 15 canonical directories, 11 superseded duplicates, and 6 older directories that predate the side-input configuration key. Four of the 10 duplicated folds are bit-identical, so submission order is recorded as the tie-break when the published AUC cannot distinguish the copies.

The TSR control required three submissions. The first reached the 12-hour limit because the estimate counted 24 windows instead of 72 condition-window evaluations. The second exposed a manifest error. The runner now saves partial results after each condition so a timeout still leaves a readable result.

## 7. Project status

- **Established:** the delay evidence still comes from the ablation, per-window tau-sweep, tau-jitter test, and network-free SINDy-DDE recovery. The SINDy-DDE median tau ratio is 0.986.
- **Rejected:** TSR is not an independent line of delay evidence, and the band-notch comparison cannot be interpreted as designed.
- **Next:** republish the model zoo using the canonical run registry.

## 8. Time spent

| Work | Days | Hours |
|---|---|---|
| Physics-floor stressors build: spectral surrogates, reliance metrics, band-notch scaffolding, test-first | Jul 19 to 20 | 4.5 |
| Quantile severity head wiring and cluster argument passthrough, with tests | Jul 20 | 1.5 |
| Submit and land the two five-fold retrains (side-input ablation, quantile severity) | Jul 20 to 21 | 1.0 |
| TSR at scale: 160 shards across five folds, shard-loss diagnosis, merge and verify | Jul 23 to 24 | 3.0 |
| Diagnose the `tr_peak_lag` floor defect; quantify resolved fraction and the delay-lock statistics | Jul 24 | 2.5 |
| Rebuild the metric: delay-lock statistic, resolved-fraction reporting, sentinel exclusion, test-first | Jul 24 to 25 | 3.0 |
| Construction-artifact control: untrained-net comparison, guards, pre-committed verdict rule, with tests | Jul 25 | 2.5 |
| Band-notch contrast: bidirectional halfwidth search and the no-overlap constraint | Jul 25 | 1.5 |
| Docs and drafting this update | Jul 25 | 1.0 |
| **Total hours** | | **20.5** |
