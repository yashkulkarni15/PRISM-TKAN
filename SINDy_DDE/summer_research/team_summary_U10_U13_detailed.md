---
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
header-includes:
  - \usepackage{float}
  - \makeatletter\def\fps@figure{H}\makeatother
  - \setlength{\intextsep}{6pt}
  - \setlength{\textfloatsep}{6pt}
---

# Research Updates 10 to 13: Team Summary

*July 5 to August 1, 2026. Four cycles, about 80 hours.*

The network's explanation held up under every stress test. Four separate attempts to add a
new kind of delay evidence all failed, and I report each as a failure. The delay claim
itself is unchanged.

| Update | Dates | Question | Answer |
|:---|:---------|:-----------------------|:--------------------------|
| 10 | Jul 5-11 | Does the explanation survive noise and blur? | **Survives.** Accuracy and delay-reliance both hold. |
| 11 | Jul 12-18 | Can I finish the remaining explainability tools? | **3 landed, 2 failed.** One claim retired. |
| 12 | Jul 19-25 | Does saliency give new delay evidence? | **No.** Broken metric; the fix finds an input artifact. |
| 13 | Jul 26-Aug 1 | Do the recommendations survive real tolerances? Did the net learn x(t)-x(t-tau)? | **Half survive. And no.** |

## Update 10: the white box holds up under stress

**Graded degradation suite.** Five stressors: added noise, moving-average smoothing and
delay jitter on the input side; weight noise and spline-grid coarsening on the model side.
At each level I record accuracy and *delay-reliance*, the AUC drop when the delayed input
is zeroed. Accuracy holding while delay-reliance collapses would mean the model had
switched from physics to memorisation.

- **Noise to twice signal energy:** AUC never below 0.75.
- **32-sample smoothing:** AUC 0.90 to 0.92.
- **50 percent weight noise:** AUC 0.89 to 0.94 — a wide flat basin, not a knife-edge solution.
- **Delay jitter:** fold 2 goes 0.994 nominal, 0.956 at a 5 percent shift, 0.87 at 50 percent.

Delay jitter is the one stressor that touches the delay alone — same signal, same weights,
only the delay value corrupted — so its smooth proportional decay is causal evidence rather
than correlational. It replicates on all five folds, and each run reproduces its clean
baseline to one part in ten million.

![All five folds under all five stressors. Solid lines are accuracy, dashed are delay-reliance. The accuracy curves stack tightly across five independently trained checkpoints, so the robustness belongs to the method rather than one lucky model. Top right is the delay-targeted stressor: the steepest, cleanest decay.](../results/figures/research_update_assets/fuzz_replication_5fold.png){width=92%}

## Update 11: three tools landed, two failed

All eval-only on existing checkpoints, no retraining.

### Landed

**Spline-edge atlas** — data-aware importance ranking plus symbolic regression on each
learned edge. All 25 top edges carry an accepted formula on every fold, R-squared 0.926 to
1.000. Delayed edges hold 15 to 16 percent of top-edge importance on folds 0 to 2 and 3 to
4 percent on folds 3 and 4, the same split the ablation shows.

![The twenty-five most important learned edge functions for one checkpoint: learned edge in blue, fitted formula dashed orange, the range the real data visits shaded, a rug of real input values beneath. Delayed edges are titled red, side-input edges green.](../results/figures/research_update_assets/spline_atlas_fold2.png){width=66%}

**Network-grounded counterfactuals** — surrogate-based search for the smallest physically
plausible edit that flips the decision. The first version fitted its surrogate to the
dataset's chatter labels, so agreeing with physics was near-automatic at 95 percent.
Refitted to the model's own predictions it scores **81.6 percent** agreement with the
analytic stability boundary: 38 of 42 chatter points flip, all toward stable, 31 landing
truly stable. The drop is the point. The typical flip lowers depth and moves the spindle
toward the nearest stable lobe.

![Each red point is a chatter operating point; arrows show the smallest edit in the speed-depth plane that flips the model's decision to stable. Blue squares land in the truly stable region below the analytic boundary; orange squares flip the model but stay physically unstable.](../results/figures/research_update_assets/counterfactuals_fold2_net.png){width=60%}

**Calibrated confidence** — split-conformal prediction over a five-checkpoint ensemble.
88.3 percent coverage at a 90 percent target, 99.6 percent selective accuracy against a raw
98.4. Every window now carries a confidence and the option to abstain.

### Failed

**Concept probe** — linear probes on the hidden state. Suspecting the regenerative-phase
target leaked from its own input, I rebuilt it circular-safe. The probe still fails
held-out (R-squared -331) while fitting train, so the concept is not linearly decodable
rather than merely circular. Claim retired. Only proximity-to-stability-limit survives
(R-squared 0.82).

**Temporal Saliency Rescaling (TSR)** — meant to restore credit to inputs acting through
long lags, which plain gradient attribution under-weights. Wired in and run, it put no peak
at the delay.

## Update 12: saliency is dead, and the metric was the problem

**A search floor masquerading as a measurement.** `tr_peak_lag` prints 2, the bottom of its
range, when it finds no peak at all. **550 of 640 windows returned exactly 2**,
indistinguishable in the output from a real measurement of 2. The per-fold means were
averages over a mostly-empty column, so the earlier "underpowered, too few windows"
diagnosis was wrong: the negative was unsound, not underpowered.

![Left: how many of each fold's 128 windows returned the search floor instead of a real peak. Right: for the 90 windows that resolved, each window's peak lag against its own delay; the 31 sitting exactly on the delay are filled on the dashed line.](../results/figures/research_update_assets/u12_tsr_floor_and_lock.png){width=92%}

**The rebuilt metric found a lock; a model-randomization control killed it.** The new
statistic tests repetition at exactly the delay, so it has no floor to fall back on. Of the
90 resolving windows, 31 peak exactly on their own delay against 3.34 expected by chance,
p = 0.00005. Then the control — same architecture, randomly initialised weights:

| Condition | Mean lock statistic | Windows locked |
|:---|:---|:---|
| Trained network, true delay | +2.81 | 79.2 percent |
| **Untrained network, true delay** | **+4.62** | **100 percent** |
| Trained network, shuffled delay | +0.88 | 16.7 percent |

The untrained network locks harder, winning 22 of 24 paired windows. The lock comes from
the input construction — the delayed channel is the signal shifted by that window's own
delay — not from training. Two corroborations: the trained lock is stronger on stable
windows (3.52) than chatter windows (2.10), backwards for a regenerative mechanism, and a
synthetic signal with no delay structure registers as locked across this dataset's delay
range. The verdict rule was fixed before the run. TSR is a closed negative and stays out of
the technical report.

**Band-notch contrast** — spectral band-stop ablation with matched removed energy. Deleting
the chatter band should cost more than deleting equal energy from the tooth-pass harmonics,
if the two can be matched. They cannot. The chatter notch is one band with a 2 percent
energy floor; the tooth notch sits on three of the strongest spectral peaks with a 13
percent floor. The ranges never meet, and widening the chatter band to close the gap makes
it swallow the tooth harmonics, deleting both conditions at once. 58 to 63 percent of
windows are unsolvable on all five folds. Both obvious escapes — one tooth harmonic instead
of three, and longer windows — left the floor two to four times too high. Structural, and
confirmed on a synthetic case.

**Two retrains, both null.** Side-input ablation, training with the process parameters on
and then off: 0.0028 AUC on average, with fold 0 better without them. Quantile severity
head, pinball loss replacing the conditional-mean head: MAE 0.88 to 0.83, RMSE unchanged
near 4.5. Both sit below 0.0092, the largest fold-level gap between two runs of an
identical configuration, so neither is claimable.

## Update 13: recommendations under tolerance, and a refuted criterion

**Monte Carlo tolerance propagation.** The 81.6 percent agreement describes the nominal
operating point, which a machinist never gets. 512 perturbed draws per proposal — 2 percent
on spindle speed, 0.05 mm on depth, 10 percent on the boundary, perturbing point and
boundary together. **19 of 42 proposals stay stable in over 85 percent of draws** against a
requirement of ten, so the criterion is met. But median survival is 68.5 percent and the
worst is 8.6, so most are not yet usable machining advice. The 10 percent boundary
uncertainty dominates the result and is my judgement rather than a measurement, so it
belongs with the headline whenever the headline is quoted.

![Survival of the 42 proposed operating points under machining uncertainty, sorted. Each bar is the fraction of 512 perturbed draws that stay below the stability boundary. Blue bars clear the 0.85 target; orange bars do not.](../results/figures/research_update_assets/u13a_counterfactual_survival.png){width=76%}

**DiCE as an independent second opinion.** DiCE never sees the stability boundary. It
answers all 42 cases, 36 physically stable; the physics-constrained search flips 38, of
which 31 are stable. They agree on outcome in **27 of 42 cases**, with 26 joint successes.
They get there differently: the physics search moves spindle speed alone in 35 cases, DiCE
cuts depth in 36. My first metric scored that as disagreement, wrongly — the physics search
is capped at current depth and cannot raise it. Knob choice is now reported as a
description, not a score.

**Did the network learn x(t) - x(t-tau)? No.** Every delayed input arrives on a known edge
with its instantaneous counterpart on another edge of the same unit, so regeneration
predicts the delayed spline is roughly the negative of the instantaneous one. Scoring each
pair from +1 for exact negation to -1 for exact copy: fold means run -0.457 to -0.639, and
-0.54 to -0.71 when weighted by edge importance. At a 0.9 threshold, 23 to 36 percent of
pairs sum and 0.1 to 1.0 percent negate, about thirty to one the wrong way. The top-ranked
fold-0 pair scores -0.989, a near-exact copy. Controls hold: untrained network -0.023,
cross-channel -0.08 to -0.18. **The plan's qualitative delay criterion is refuted.** The
copy-like pattern is consistent with a sum, which reinforces tooth-synchronous content,
though this edge-level test does not establish that for the full recurrent network.

![Left: 3840 delay-edge pairs against their control, on an axis running from exact copy at the left to exact negation at the right. The expected regenerative difference would sit at the right; the mass sits at the left. Right: each fold's mean beside its two controls. The copy-like pattern is 20 to 25 times the untrained null and appears on every fold.](../results/figures/research_update_assets/u13c_regenerative_sum_not_difference.png){width=92%}

## Where this leaves the project

**The delay claim is unchanged.** Four results, none of them an attribution map:

1. Zero the delayed input and detection drops to near chance.
2. Sweep the delay per window and the peak sits at the physical tooth-pass value.
3. Shift the delay and accuracy degrades in smooth proportion.
4. A network-free SINDy-DDE recovery finds the same delay independently, median ratio 0.986.

**Four things died:** TSR (construction artifact), the concept probe (concept absent), the
band-notch contrast (unmatchable by construction), and the edge-level
regenerative-difference criterion (the network sums rather than differences).

**One lesson, four times over.** A routine hits a limit, returns a bound or a sentinel, and
that value flows into an average that cannot tell a bound from a measurement: a search
floor of 2 read as a peak, a notch solver that ran out of iterations and returned a sub-bin
band removing exactly zero energy, and two cluster jobs that exited 0 in 90 seconds without
running. Standing checks now: sentinels excluded from every aggregate, resolved-window
counts reported with any number derived from them, and wrappers must assert their output
exists before exiting clean.

**Techniques used across the four cycles:** graded input and model degradation, symbolic
regression on KAN spline edges, surrogate-based counterfactual search, DiCE, split-conformal
prediction, linear concept probes, Temporal Saliency Rescaling with a model-randomization
control, spectral band-stop ablation, Monte Carlo tolerance propagation, and network-free
SINDy-DDE recovery.
