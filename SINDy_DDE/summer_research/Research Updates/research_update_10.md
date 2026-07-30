---
title: "Research Update 10"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

<!--
Every number in this update traces to a fuzz manifest in the working tree
(results/phase3/xai/fuzz/fold{0..4}/fuzz_manifest.json), verified on July 17,
not recalled. The two figures are rendered straight from those manifests.
-->

## 1. State Before This Work

By Update 9 the trained network was, on paper, a white box. Zeroing the delayed input dropped chatter detection to near chance and blew up the severity error (Update 8). A per-window sweep placed the learned delay at exactly the physical tooth-pass value. A network-free SINDy recovery found the same delay. And the model's implied stability lobe diagram matched the analytic limit on 98 percent of operating points (Update 9).

But every one of those results used clean signals and the exact trained weights. That leaves one question open. Is the explanation robust, or does it only hold in perfect conditions? A real cutting signal is noisy, blurred by the sensor, and sampled at a delay the model can only estimate. If the delay-reliance disappears the moment the input is blurred or the weights are nudged, the white box is a lab curiosity. If it survives, it is a property worth trusting on the shop floor.

So this cycle I built a graded stress-test suite (the fuzziness suite, in the code). It blurs either what the model sees or the model itself, by controlled amounts, and re-measures the physics signal at every step. I built it test-first (the suite grew to 585 passing tests), ran it on all five trained checkpoints on the cluster, and read the numbers back. Every job finished, and every one passed its self-check.

This update covers July 5 to 11, about 20 hours of work.

## 2. The Stress-Test Suite

The suite has five stressors in two groups. Three are input-side: they blur what the model sees and leave the weights alone. They are added noise (in units of the signal's own standard deviation), smoothing (a moving average of 1 to 32 samples), and tau-jitter (shifting each window's delay by a random amount). Two are model-side: they blur the model and feed it a clean signal. They are spline-coarsening (refitting every learned KAN edge onto a coarser grid, from 5 knots down to 1) and weight-noise (0 to 50 percent random noise on the weights).

The point is not the accuracy curve. At every level I re-run the Update 8 delay test and record two numbers: the per-window AUC, and the delay-ablation drop, which is how much AUC falls when I zero the delayed input. The AUC says whether the model still works. The drop says whether it still works *by using the delay*. A model that stays accurate while its drop collapses has stopped doing physics and started memorizing. The suite is built to catch that.

![One checkpoint (fold 2) under all five stressors. In each panel the solid blue line is per-window AUC (left axis; the dotted line is the clean baseline, the dashed grey line is chance). The dashed red line is the delay-ablation drop (right axis; the dotted red line marks 0.05, the floor where the delay signal is gone). Reading each panel left to right shows how far the stressor can go before accuracy or the physics signal breaks. Accuracy stays above 0.75 under every stressor; the delay signal (red) only nears its floor under the most extreme noise.](results/figures/research_update_assets/fuzz_dashboard_fold2.png){width=95%}

Everything runs eval-only on the existing `best_auc.pt` checkpoints, with no retraining. Input stressors return a fresh batch and never change the original; model stressors work on a copy. Each run writes a manifest, and both figures come straight from it.

## 3. Accuracy Survives Realistic Input Corruption

The two stressors that mimic real sensor problems, noise and smoothing, barely move the model.

With added noise, accuracy fades slowly. Noise at half the signal's amplitude leaves AUC at 0.90 to 0.93 across the five folds. Noise equal to the signal leaves 0.82 to 0.87. Noise at twice the signal energy, where the signal is now mostly noise, still leaves 0.76 to 0.82. **AUC never falls below 0.75 anywhere in the tested range.** The delay signal is nearly as tough: it holds on four of five folds even at the extreme, and collapses on only one (fold 2, at twice-signal noise). That was the only breakdown the whole noise sweep produced.

Smoothing moves the model even less. A 32-sample moving average, which erases most of the fine detail, still leaves AUC at 0.90 to 0.92. Here the delay drop does shrink as the window widens (on fold 2, from 0.27 to 0.15). That is correct behavior, not a failure. Smoothing destroys the fine regenerative detail the delay term reads, so the model has less delay signal to use, and uses less. The suite is watching the physics fade exactly where it should.

## 4. The Delay Really Drives the Decision

The sharpest result is tau-jitter, the one stressor aimed straight at the physics. It changes nothing about the signal or the weights. It only corrupts the delay value the model is told to use, by a random amount per window. If the delay were incidental, mis-stating it would cost nothing. It is not incidental.

Shifting the delay by five percent (about five samples, which reaches roughly three-quarters of the delays) already costs AUC: fold 2 drops from 0.994 to 0.956. At ten percent, AUC is 0.90 to 0.94. At fifty percent (a shift of about 53 samples) it settles at 0.84 to 0.87. The delay drop falls right along with it, because a mis-stated delay is a partly-wrong delay. The decline is smooth: the model does not shatter, it degrades in proportion to how wrong the delay is.

This is a **third independent line of delay evidence**, and it is causal in a way the first two were not. Update 8 removed the delay and watched the model fail. Update 9 swept the delay and found the peak at the physical value. This one shifts the delay by a set amount and watches accuracy fall in proportion. Removal, location, and now sensitivity all point at the same quantity: the model's decision rides on the tooth-pass delay.

## 5. Robust to Its Own Corruption

The two model-side stressors test a different thing: not whether the model uses the physics, but whether the learned solution is stable.

Weight-noise tests flatness. Adding up to 50 percent random noise to every KAN weight, a heavy hit, still leaves AUC at 0.89 to 0.94 on every fold, and it stays nearly flat until the noise passes twenty percent. A model this insensitive to its own weights sits in a wide, flat basin, the mark of a solution that generalizes rather than one balanced on a knife-edge. The delay signal mostly holds, and even strengthens a little on the middle folds.

Spline-coarsening tests readability. Collapsing every learned edge from five knots down to a single spline segment costs only 0.05 to 0.10 AUC, and the coarse grid still refits the original edge at R-squared above 0.92 (above 0.95 for the severity heads). That the edges survive such heavy simplification confirms the Update 9 finding from the other side: these edges are smooth, simple functions, which is why the symbolic fits accepted 25 of 25 of them. One honest wrinkle: under heavy coarsening the delay drop moves in different directions across folds. It rises on the two folds that leaned least on the delay (folds 0 and 2, since coarsening concentrates the reliance that is left onto the delay channel), holds about flat on fold 1, and collapses on folds 3 and 4 at the coarsest grid. So coarsening is a clean test of accuracy and complexity, but a fold-dependent test of the delay signal, and I do not over-read it.

## 6. It Replicates and It Reproduces

Two things make the suite trustworthy rather than anecdotal.

First, it replicates across all five folds. The figure below overlays every fold on every stressor. The striking part is how tightly the accuracy curves (solid) stack, even though the five checkpoints were trained on different data splits. The same stressors cause the same slow decay everywhere. The delay signals (dashed) sit at different heights, because folds 1, 3, and 4 lean harder on the delay than folds 0 and 2, just as their Update 8 baseline drops predicted. But they all trend the same way.

![All five folds under all five stressors. Solid lines are per-window AUC (left axis); dashed lines are the delay-ablation drop (right axis, auto-scaled per panel, so read the trend within a panel, not the dashed heights across panels). The accuracy curves stack tightly across independently trained folds, which shows the robustness is a property of the method, not of one lucky checkpoint. The tau-jitter panel (top right) is the physics-targeted stressor and shows the steepest, cleanest decay.](results/figures/research_update_assets/fuzz_replication_5fold.png){width=100%}

Second, it reproduces exactly. Every run carries a self-check: it replays the zero-level (unstressed) case and confirms it recovers the clean baseline to within tolerance. All five folds passed, with a largest deviation of about one part in ten million, and no errors on any fold. The harness adds no noise of its own. The curves are the model's, not the measurement's.

## 7. Where This Stands

The white box is now robust, not just detailed. Under realistic input corruption, noise up to twice the signal energy and smoothing that erases the fine detail, the model keeps working and keeps using the delay. Under heavy weight corruption it barely flinches, which marks a flat, generalizing solution. And the one stressor aimed at the physics, tau-jitter, degrades the model in smooth proportion to how wrong the delay is, a third and causal confirmation of the delay-reliance the ablation and the sweep found. All of it replicates across five folds and reproduces to machine precision.

The standing caveat from Updates 8 and 9 still holds: the sampled-window AUC ceilings near 0.99, so I read the shapes of the curves and the order of the breakdown points, not the absolute AUC heights. That is why the suite reports breakdown points and delay drops rather than headline accuracy.

## 8. Obstacles and Next Steps

Obstacles:

1. The AUC ceiling again compresses every size-based reading, so the suite is built around locations and orderings (which stressor breaks first, and at what level) rather than absolute heights.
2. Spline-coarsening's effect on the delay signal is fold-dependent, so it is an honest test of accuracy and complexity but not a clean delay test. Tau-jitter is the clean delay test.
3. All of this reads the trained network, so it runs only on the cluster, where the checkpoints and the raw signals live.

Next steps, in order:

1. Finish the full white-box re-run on the cluster, the run that produces the remaining owed tools: the wired temporal-saliency attribution, network-grounded counterfactuals, and the spline-edge atlas. That run overran its time limit on the first try and was resubmitted with a longer budget. Its results are the subject of **Update 11**.
2. Add the tau-jitter curve to the faithfulness section of the white-box report, next to the ablation and the sweep, as the third delay result.
3. Retrain with the quantile severity head (built in Update 9) to test whether it tames the heavy-tailed severity error, now that the classification side is shown to be robust.

## 9. Summary

Update 9 made the network a detailed white box on clean data. Update 10 shows the white box is robust. A graded stress-test suite degrades the input and the model by controlled amounts and re-measures the physics signal at every step. Accuracy survives noise at twice the signal energy and smoothing that erases the fine detail. It barely moves under fifty-percent weight noise. And it degrades smoothly, in proportion to how wrong the delay is, only when the delay value is corrupted. That last result is a third, causal confirmation that the model's decision rides on the tooth-pass delay. Everything replicates across five folds and reproduces to machine precision, with every job passing its self-check. The network does not just encode the regenerative physics on clean data. It holds onto that physics as the world around it is blurred.

This update covers July 5 to 11, about 20 hours of work.

## 10. Time Spent

| Work | Days | Hours |
|---|---|---|
| Plan Update 10: scope the two stressor families, design the delay-signal probe | Jul 5 | 1.5 |
| Refactor the shared window sampler out of the white-box runner, with tests | Jul 5 to 6 | 1.5 |
| Degradation module: five stressors, physics-reliance probes, breakdown points, test-first | Jul 6 to 8 | 5.0 |
| Per-checkpoint dashboard and cross-fold replication plots, rendered from the manifest | Jul 8 | 2.0 |
| Cluster wrappers (single-fold and all-folds sbatch), env-parameterized | Jul 8 to 9 | 1.0 |
| Cluster runs on all five checkpoints, pulls, and self-check validation | Jul 9 to 10 | 3.0 |
| Cross-fold analysis: breakdown ordering, delay-signal reading, honesty caveats | Jul 10 | 3.0 |
| Figures, docs, and drafting this update | Jul 10 to 11 | 3.0 |
| **Total hours** | | **20.0** |

The result I most wanted is the tau-jitter curve. It is the first stressor that touches the physics and nothing else, and it degrades the model in exact proportion to how wrong the delay is. The ablation said the delay matters. The sweep said which delay. This says the model is sensitive to it, smoothly and causally. Three methods, one delay.
