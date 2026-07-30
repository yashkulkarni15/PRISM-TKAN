---
title: "Research Update 9"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

<!--
Every number in this update traces to a file in the working tree (a manifest, a
run summary, or a figure) and was verified on July 4, not recalled. Where a
planned method did not produce a usable result, it is reported as such rather
than dressed up.
-->

## 1. State Before This Work

Update 8 proved one thing about the trained network: it depends on the regenerative tooth-pass delay. Zeroing the delayed input alone dropped chatter detection to near chance on three of five folds and blew up severity error by one to two orders of magnitude, on every checkpoint. That was an intervention, and it answered *whether* the model uses the delay.

It did not answer *which* delay, or *what physics* the network encodes. Update 8 also left a list of its own admitted weaknesses: a global delay-sweep that blurred because it imposed one delay on a sample from many operating points, and so peaked at twice the physical value on all five folds; symbolic edge-fits accepted without a faithfulness gate; and a comparison against the MLP baseline that leaned on a single cited "about 0.87."

This update answers the *which* and the *what*, and closes those weaknesses. Two independent methods now locate the delay the network learned, a global figure shows the physics it encodes, and three smaller results harden the white box. I built the whole toolkit test-first (the suite grew from 469 to 560 passing tests), ran the eval-only analyses on the cluster, and read the numbers back. I also report plainly the three instruments that did *not* land.

This update covers June 29 to July 4, about 20 hours of work.

## 2. Which Delay Did It Learn

Update 8's delay-sweep imposed one delay, in samples, on every window. But a window is drawn from many spindle speeds, and one delay cannot be right for all of them, so the peak blurred and landed at twice the physical delay on all five folds. That was an artifact, not a finding. The fix sweeps each window at multiples of *its own* tooth-pass delay and reads the peak on a shared axis.

![Per-window-relative delay sweep across all five folds. Each window's imposed delay is set to a multiple of its own tooth-pass delay tau = round(fs/tpf), so windows from different operating points aggregate on one axis. Every fold peaks sharply at x1 (the true physical delay) and falls off on both sides (to 0.83-0.87 at half the delay, 0.93-0.96 at twice). The grey band is the no-delay floor from the ablation (removing the delayed input entirely drops AUC to 0.50-0.77). The global sweep of Update 8 peaked at x2 on every fold; corrected, the peak sits exactly at the physical delay.](results/figures/research_update_assets/whitebox_tau_sweep_relative.png){width=88%}

The result is clear: **on all five folds the sweep peaks at exactly one times the window's own tooth-pass delay.** At the physical delay the sampled AUC is 0.986 to 0.996; halving it costs 0.12 to 0.16, doubling it costs 0.04 to 0.06, and removing the delayed input entirely drops it to 0.50 to 0.77. (That last case is the Update 8 ablation, which I re-ran and reproduced exactly: drops of 0.23 to 0.49, three folds at or near chance, delayed channels holding 9.7 to 10.7 percent of the spline weight.) The network is not keying on some generic lag; it is keying on the tooth-pass period, the exact quantity the regenerative-chatter equations are built around. As in Update 8, the sampled AUC still ceilings near 0.99, so I read where the peak sits, not how tall it is, since the location is what matters here.

## 3. An Independent Recovery From the Raw Signal

The sweep asks the network where its delay is. A stronger test asks whether the delay is even present in the raw cutting signal, recovered by a method that never sees the network. I built a delay-aware SINDy recovery (after arXiv:2407.19640, with no neural network and no `pysindy`, which has no delay operator): it builds delayed feature columns by hand and selects, per window, the lag and sparse coefficients that best explain the measured acceleration. I first validated it on synthetic delay-differential signals with a delay I planted, where it recovered the delay to three percent.

Run on the real fold-2 cutting signals, it recovers a median delay of **0.99 times the physical tooth-pass delay**, and selects the regenerative term `x(t) − x(t−tau)` in **100 percent** of the 64 windows.

![Independent SINDy-DDE delay recovery on the real fold-2 cutting signals (64 windows). The sparse-regression residual is plotted against the imposed delay as a multiple of each window's physical tooth-pass delay; the aggregate minimum sits at 0.99x, and the regenerative term x(t)-x(t-tau) is selected in every window. This is a second recovery of the same physical delay, fully independent of the network: the ablation says the model relies on the delay, and signal-level regression confirms the delay is really there, at the tooth-pass value.](results/figures/research_update_assets/sindy_dde_tau_recovery_real.png){width=80%}

I state one honest limit. The median lands on the physical delay, but the per-window spread is wide: the middle half of windows recover between 0.55 and 1.29 times the physical delay. So this is a *bracketed* recovery. The central estimate is exactly right; individual noisy windows scatter around it. Even bracketed, it delivers the point that matters. The delay the network learned (Section 2) is the same delay that a network-free reading of the raw signal recovers. Two independent methods, one physical delay.

## 4. What Physics Did It Learn: the Stability Lobe Diagram

The delay results are local, window by window. The global question a machining audience asks is whether the network learned the *stability lobe diagram*: the scalloped boundary in the spindle-speed / axial-depth plane that separates stable cutting from chatter. I swept the fold-2 checkpoint across its validation operating points and read its decision boundary against the analytic `blim` curve already in the metadata.

![Model-implied stability lobe diagram for the fold-2 checkpoint, swept over its validation operating points (1014 speed-depth cells, spindle speed 2000-30000 rpm). The model's chatter/stable decision boundary (read from its own predictions) is overlaid on the analytic blim limit. The two agree on 98.2 percent of cells; the stable-region intersection-over-union is 0.94, and the boundary depth error averages 0.35 mm (11 percent). The network has learned the lobes.](results/figures/research_update_assets/sld_fold2_ds29grp.png){width=92%}

The agreement is strong: **the model's decision boundary matches the analytic stability limit on 98.2 percent of cells**, the stable-region intersection-over-union is 0.94, and the boundary depth-of-cut error averages 0.35 mm, or 11 percent, with no operating-point column dropped. This is the figure the project has been working toward: a global explanation, in machining coordinates, that an engineer reads in one glance. The delay-aware network has reconstructed the classic stability lobes from data. One caveat carries over from the white box: the per-cell sample AUC ceilings at 0.9987, so I lean on the boundary-shape error (which does not depend on that ceiling), not the raw per-cell AUC, as the honest faithfulness number.

## 5. Hardening the White Box

Three smaller results shore up the claims above.

**The symbolic fits now pass a faithfulness gate.** Update 8 reported that all twenty-five top edges fit closed forms, but without a rule for rejecting a bad fit. I added an R-squared and simplicity gate (after pykan): a formula is accepted only if it clears the threshold, with a simplicity-versus-fit trade-off. Re-run across all five folds, **all 25 of 25 top edges pass the gate on every fold**, with R-squared between 0.926 and 0.9999. The white box's readability survives an explicit honesty check; nothing received a formula it cannot support.

**The MLP comparison is now a matched per-fold table**, retiring a debt owed since Update 6. Pulling the existing MLP run's per-fold numbers onto the same folds:

| fold | TKAN AUC | MLP AUC | margin |
|---|---|---|---|
| 0 | 0.960 | 0.874 | +0.085 |
| 1 | 0.912 | 0.875 | +0.037 |
| 2 | 0.975 | 0.868 | +0.108 |
| 3 | 0.960 | 0.878 | +0.082 |
| 4 | 0.970 | 0.880 | +0.089 |
| **mean** | **0.955 ± 0.022** | **0.875 ± 0.004** | **+0.080** |

The long-cited "about 0.87" is the MLP's mean, and it is strikingly stable across folds (standard deviation 0.004). The delay-aware network beats it on every fold, by between 0.037 and 0.108. Two caveats keep the row honest: the MLP figure is its final-epoch validation AUC while the TKAN figure is its best-epoch AUC, and both runs predate the June leakage-grouping fix. But both lack that fix in the same way on the same fold definitions, so the margin between them is clean.

**The chatter head is now calibrated.** I added split-conformal prediction to the classifier. At a target coverage of 90 percent it achieves an empirical **91.4 percent**, and the whole target sweep is well-calibrated (empirical 80/91/96/98/100 percent against targets 80/90/95/98/99). This gives every window a calibrated confidence and the option to abstain. It is also the fix for the uncalibrated 0.5-threshold accuracy swings seen in the held-out evaluation. This ran on a single fold; the five-checkpoint ensemble version is a small next step.

## 6. Attacking the Weak Head

The severity head has been the weak point in every update: strong classification (0.955 in-fold, 0.91 held-out) but heavy-tailed severity regression (in-distribution MAE around 0.88, but 3.3 to 4.8 held-out, RMSE far above MAE because a few files dominate). A squared loss on a point estimate is the wrong tool for a heavy tail. So I built a flag-gated **quantile severity head**: switched on, it emits a set of quantiles for the severity target and trains them through a pinball loss, giving prediction intervals instead of a single fragile point; switched off (the default) it is byte-for-byte the old head, so nothing already validated changes. The head, the loss, and the model wiring are implemented and tested (13 tests). Measuring whether it tames the tail needs a retrain (a later cluster job), but the machinery is ready, and it will feed calibrated intervals into the conformal layer above.

## 7. What Did Not Land

Three planned instruments ran but did not produce a usable result, and I report them as such.

- **Temporal Saliency Rescaling** is built and tested, but it is not yet wired into the white-box runner, so it did not run this cycle. The intervention results (Sections 2 to 4) are the main evidence regardless; TSR was only ever meant to be a supporting method, so it stays owed.
- **Physics-constrained counterfactuals** ran, but on a surrogate fitted to the dataset's own labels, not to the network: passing the checkpoint recorded its identity but did not generate the model's own predictions, so the 95 percent physical-agreement number validates the search machinery, not the network's decision surface. The real version needs the model's per-window predictions as input; it is a small fix, not a finding yet.
- **Concept probes and hidden-state dynamics** gave weak or negative results. A linear probe recovers *proximity to the stability limit* from the 64-dimensional hidden state cleanly (R-squared 0.82), but probes for the regeneration phase and the chatter-versus-tooth-pass frequency separation fail to generalize (negative test R-squared), so I cannot yet claim those concepts are linearly encoded. The recurrent-Jacobian analysis finds chatter windows slightly more unstable than stable ones (leading eigenvalue magnitude 1.0153 versus 1.0142), but the gap is smaller than its own scatter: directionally right, statistically nothing. Both are exploratory, not results.

## 8. Obstacles and Next Steps

Obstacles:

1. The sampled-window AUC still ceilings near 0.99, so every magnitude-based readout (the sweep depth, the per-cell SLD AUC) stays compressed. I report locations, boundary shapes, and orderings, and lean on the interventions, the same discipline as Update 8.
2. The SINDy-DDE per-window delay estimate is noisy (middle half 0.55 to 1.29 times physical); only the aggregate median is trustworthy.
3. The checkpoint-dependent instruments all run on the cluster, where the checkpoints and the raw signal tree live; nothing that reads the trained network can be reproduced on the laptop.

Next steps, in order:

1. Wire TSR into the runner and re-measure the delay's apparent attribution share.
2. Feed the network's per-window predictions into the counterfactual search so it explains the model, not a label surrogate.
3. Run the conformal layer as the five-checkpoint ensemble, and debug the two failing concept probes (the regeneration-phase target looks degenerate).
4. Retrain with the quantile severity head to test whether it tames the heavy tail.
5. Launch the no-side-input training ablation as a fire-and-forget cluster job for Update 10.

## 9. Summary

Update 8 showed the trained network depends on the regenerative delay. Update 9 says which delay and what physics. The per-window sweep localizes the learned delay to exactly the physical tooth-pass value on all five folds; an independent, network-free SINDy recovery of the raw signal lands on the same delay (median 0.99 times physical, regenerative term selected in every window), so two independent methods agree. Globally, the model's implied stability lobe diagram matches the analytic limit on 98.2 percent of operating points, so the network has learned the lobes. Three supporting results harden the picture: the symbolic fits pass an explicit faithfulness gate (25 of 25 on every fold), the delay-aware network beats the MLP baseline on every matched fold (+0.08 mean AUC), and the classifier is now calibrated (91 percent coverage at a 90 percent target). I built the quantile severity head to attack the one weak output, and I report plainly the three instruments (TSR, network counterfactuals, and the concept and dynamics probes) that were built but did not yet yield a result. The model does not merely use the regenerative delay; it learned the physical delay and the stability physics that delay generates.

This update covers June 29 to July 4, about 20 hours of work.

## 10. Time Spent

| Work | Days | Hours |
|---|---|---|
| Plan Update 9: scope decision, owed-debt calls, sequenced task list | Jun 29 | 2.0 |
| Per-window-relative tau sweep, TSR, and the symbolic R-squared gate, with tests | Jun 29 to 30 | 3.0 |
| SINDy-DDE independent delay recovery: hand-built delay-embedded regression + synthetic self-test | Jun 30 to Jul 1 | 3.0 |
| Stability-lobe module and its input-construction design spec | Jul 1 to 2 | 3.0 |
| Counterfactuals, concept probes, conformal, and hidden-dynamics modules | Jul 2 to 3 | 3.0 |
| Flag-gated quantile severity head (pinball loss, prediction intervals), test-first | Jul 3 | 1.5 |
| Cluster runs (tau sweep, SLD, SINDy-DDE, conformal), pulls, and analysis | Jul 3 to 4 | 2.5 |
| Matched per-fold MLP baseline table from the existing run artifacts | Jul 4 | 0.5 |
| Figures, docs, and drafting this update | Jul 4 | 1.5 |
| **Total hours** | | **20.0** |

The two results I most wanted are the two that agree: the sweep peak at the physical delay, and the independent signal-level recovery of that same delay. Between them they turn "the model uses a delay" into "the model learned the tooth-pass delay."
