---
title: "Research Update 8"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

## 1. State Before This Work

The last update measured how well the model performs. The 0.955 chatter AUC had been chosen on the validation fold; I turned it into a held-out number. The five fold checkpoints score a chatter AUC of about 0.91 on datasets they never trained on, and each fold beats the MLP baseline of 0.87. That answered how well the model performs. It said nothing about how the model works.

The reason for building a delay-aware network was physical. Chatter is regenerative: the tool re-cuts the surface it left one tooth pass ago. A model with an explicit delayed input should therefore rely on that input. Until now, that was a design intention, not a measured fact.

This update tests it. I built a tool that reads the trained model's own learned functions, ran it on the strongest fold, found and fixed a measurement error in my own code, repeated the main result across all five folds, and used a checked literature review to plan the next methods. The main finding: removing the delayed input breaks chatter detection. On two folds the model drops to chance, and the severity error grows ten to a hundred times larger. The model depends on the delay.

This update covers June 23 to July 1, about 20 hours of work.

## 2. Why This Model Can Be Opened at All

A KAN can be opened up for two reasons. First, every connection is not a single weight but a learnable one-dimensional function, a spline that training shapes. Each of the 13,984 edges in the trained model can be plotted, ranked, and fit to a simple formula. Second, the inputs have physical meaning by construction. Each recurrent gate sees seventy-two inputs: the four current signal channels (tool positions and cutting forces), the sixty-four hidden-state values, and the same four channels delayed by one tooth pass, the regenerative term of the chatter equations. I wrote this index map down once, in code, and every analysis uses it. So "edge 68" is never just a number; it is "the delayed x-position."

An early design lesson shaped the rest. The standard way to rank KAN edges is by mean absolute weight, the usual pruning signal. It kept none of the 13,984 edges at any reasonable threshold, because the trained splines are each small and a weight-only ranking ignores what the data actually sends through an edge. I replaced it with a data-aware score: push a few hundred real windows through the model and rank each edge by how much its output varies. That single change is what gives the rankings, the symbolic fits, and the pruning curves their meaning.

## 3. The Pipeline

The pipeline has seven analysis modules, each with one job:

- **Semantics** holds the index map.
- **Activations** computes the data-aware edge importance.
- **Symbolic** fits each important edge to a closed-form function, balancing fit quality against complexity.
- **Feature importance** measures per-channel and per-input attribution, including permutation tests.
- **Physics** runs the interventions: removing the delayed input, sweeping the imposed delay, and measuring the delayed input's share of the learned weight.
- **Faithfulness** runs the checks that keep the rest honest: pruning curves, deletion tests, and the model-randomization sanity check from the saliency literature.
- **Case studies** trace single windows through the model.

A runner executes all seven on one checkpoint and writes a machine-readable manifest plus thirteen outputs: figures, symbolic tables, and attribution arrays. A report renderer then fills a fixed template from the manifest. Every number comes from the file, and anything missing is printed as "N/A" rather than invented. A cluster wrapper runs any fold by its run id. I built all of it test-first; the test suite grew from 410 to 469 tests, all passing.

## 4. The First Run, and the Flaw It Hid

The first real run used the fold-2 checkpoint (validation AUC 0.975) and took about nine minutes on CPU. This is the advantage of eval-only analysis: training is expensive, but questioning the trained model afterward is cheap. The results looked clean, but two of them were wrong.

The window sampler took the first files in table order. That gave a single operating point, easy separation, and a sample AUC stuck at exactly 1.0. At that ceiling, permutation importance fails silently: permuting an input cannot lower an AUC that has no room to fall. As a result, the force channels and the spindle-speed side inputs both scored zero importance, and I briefly believed the model ignored them. I replaced the sampler with a seeded random draw that balances the two classes and spreads across operating points, then re-ran. Both findings reversed: permuting the force channels now costs 0.054 to 0.057 AUC, and the side inputs cost a small but real 0.012 to 0.016. I report the correction as plainly as the findings, because it makes the case for the faithfulness checks: the pipeline's own tests caught the bad sample.

## 5. What the White Box Shows

![Delay interventions on the fold-2 checkpoint (256-window corrected run). Panel A is the main test: with the tau-delayed input intact, chatter AUC on the sampled windows is 0.9922 and severity MAE is 0.172; zeroing that input alone drops AUC to 0.7059 and raises severity MAE to 11.57 (log axis). Panel B imposes one global delay at half, one, and two times the physical tooth-pass value; all three land near 0.85 to 0.87, well above the no-delay floor of 0.7059 (dashed), but the sweep is shallow because one global delay cannot match every window in a mixed-operating-point sample. Panel C keeps only the most important edges: the top 5 percent hold an AUC of 0.816, and the curve rises steadily back to 0.9922 once all edges are kept, evidence that the importance ranking finds the load-bearing part of the network.](results/figures/research_update_assets/whitebox_delay_intervention_fold2.png){width=97%}

The strongest result is an intervention, not an attribution. Zeroing the tau-delayed input, and changing nothing else, drops the fold-2 sample AUC from 0.9922 to 0.7059 and raises severity MAE from 0.172 to 11.57. Attribution methods answer "what does the model look at," and each carries its own biases. An ablation answers a cleaner question: "what happens without it?" Without the regenerative term, this model cannot do its job.

The supporting evidence agrees. The delayed channels hold 10.7 percent of the learned spline weight while making up only 5.6 percent of the inputs, nearly twice the weight their count would predict. The permutation importances now make physical sense: the y tool position leads at 0.231, x position at 0.116, and the two forces at 0.054 to 0.057. The network is also readable as formulas: all twenty-five of the most important edges fit closed forms (tanh, sine, and low-order polynomials) with R-squared between 0.96 and 0.9998. These edges feed position, delayed position, and side-branch features into the classifier, including a clean sine wave on the delayed x-position.

![The eight most important learned edge functions of the fold-2 model, plotted from the trained splines with their fitted closed forms. Every panel is a real nonlinearity (saturating tanh, sine, or cubic); none reduce to a straight line. The set includes classifier reads of hidden units, the current x-position, and side-branch features, with fits reaching R-squared up to 0.9998.](results/figures/research_update_assets/whitebox_top_edges_fold2.png){width=97%}

The faithfulness checks pass. Keeping only the top 5 percent of edges holds an AUC of 0.816, and the curve recovers smoothly to full performance, so the importance ranking is finding the part of the network that matters. Randomizing the model's weights drops the attribution similarity to 0.06, near zero, exactly as it should: an explanation that survives scrambling the weights was never reading the model. This one does not survive it.

Three caveats remain. First, the sampled-window AUC sits near its ceiling even after the fix (0.9922), so the permutation magnitudes are compressed; only their ordering is trustworthy, and the validation and held-out numbers stay the model's honest performance. Second, the global tau sweep is shallow, and I now know why: the sample mixes operating points on purpose, so no single imposed delay fits every window. The sweep needs to run per-window, each window swept at multiples of its own tooth-pass delay. Third, the single-window case studies flip verdicts between runs; they are anecdotes until aggregated.

## 6. Replication Across Folds

One fold is only a case study. I re-ran the pipeline on every fold checkpoint at 512 windows, including a fresh fold-2 run at the same size, so all five rows are comparable. All five finished; fold 3 needed one resubmission after its first job died without writing anything.

| fold | validation AUC | sample AUC, delay on | delay zeroed | AUC drop | severity MAE, on → off | randomization cosine (near 0 = pass) |
|---|---|---|---|---|---|---|
| 0 | 0.9597 | 0.9957 | 0.7657 | 0.230 | 0.133 → 7.32 | 0.014 |
| 1 | 0.9122 | 0.9865 | 0.5000 | 0.487 | 0.238 → 14.72 | −0.007 |
| 2 | 0.9752 | 0.9943 | 0.7281 | 0.266 | 0.161 → 11.75 | 0.100 |
| 3 | 0.9605 | 0.9932 | 0.5098 | 0.484 | 0.233 → 3.06 | −0.037 |
| 4 | 0.9695 | 0.9901 | 0.5413 | 0.449 | 0.308 → 16.83 | −0.003 |

![Delay ablation repeated across all five folds at 512 windows. Panel A: chatter AUC per fold with the delayed input intact (0.986 to 0.996) versus zeroed (0.500 to 0.766); fold 1 falls exactly to the chance line at 0.5, fold 3 to 0.510, and fold 4 to 0.541. Panel B: severity MAE per fold on a log scale, rising from 0.13 to 0.31 with the delay intact to 3.1 to 16.8 with it zeroed. The intervention breaks both outputs on every fold.](results/figures/research_update_assets/whitebox_delay_replication.png){width=90%}

The delay dependence holds on every fold, and more strongly than in the first run. The ablation costs between 0.230 and 0.487 AUC, averaging 0.38. Three of the five folds land at or near the coin-flip line: fold 1 at exactly 0.500, fold 3 at 0.510, fold 4 at 0.541. In plain terms: take away the delayed input, and three of the five trained models can no longer tell chatter from stable cutting at all. Severity fails just as badly, with errors growing from a few tenths to between 3 and 17, far too large for the prediction to mean anything. The structural findings repeat too: the delayed channels hold 9.7 to 10.7 percent of the spline weight on every fold, all twenty-five top edges are nonlinear on every fold, and the randomization check passes everywhere.

Two findings became clearer. First, the 512-window runs stayed at the sample-AUC ceiling (0.987 to 0.996), so the caveat about compressed magnitudes still holds. Second, the side-input permutation cost fell from 0.012 to 0.016 on fold 2's first run to 0.002 to 0.009 across the replication runs: the spindle-speed side branch adds little to the trained model's predictions. It was added to stabilize training, and whether it earned its place there is exactly what the still-owed no-side-input training ablation will measure; the white box can only question the model that was actually trained. The global tau sweep, meanwhile, peaked at twice the physical delay on all five runs. I read this as confirmation that imposing one delay on windows from many operating points cannot locate the true delay, and that the per-window sweep is a required fix, not an optional one.

## 7. What the Literature Says

Before choosing the next methods, I ran a careful literature review, checking each citation against the source rather than from memory, re-reading the recent papers by hand, and leaving any number I could not verify uncited. Three results bear directly on this update.

First, the recurrent-saliency literature (Input-Cell Attention, arXiv:1910.12370; Temporal Saliency Rescaling, arXiv:2010.13924) shows that gradient attributions on recurrent models systematically fade for early and lagged timesteps, exactly where the regenerative term lives. This is why I report interventions as the main evidence and treat gradients as supporting at best. Temporal Saliency Rescaling (TSR) is the published correction, and it is next on the attribution path.

Second, the KAN symbolic-extraction literature names my current weakness: fitting each edge on its own depends heavily on initialization. The fixes are an R-squared acceptance gate, so no edge gets a formula it does not support (pykan), followed by greedy end-to-end snapping scored by whole-model loss (arXiv:2603.15250).

Third, delay-aware SINDy (arXiv:2407.19640) can recover a delay and a regenerative term directly from the signals, an independent estimate of tau to compare against the network's.

The review also settled the baseline panel for the eventual paper: vanilla TKAN (arXiv:2405.07344) and HAVOK (arXiv:1608.05306). And it found the closest precedent to this work, a KAN with symbolic activations for joint fault and severity classification (arXiv:2412.01322), which supports the approach while leaving the delay physics to this project.

## 8. Where This Stands

The Phase 3 model now stands on three legs. Performance: 0.955 in-fold, 0.91 held-out, every fold above baseline. Honesty: leakage-proof grouping, held-out evaluation, and a corrected measurement error on the record. And now mechanism: the trained network clearly depends on the regenerative delay, the physics the architecture was built around, shown by intervention and repeated on all five folds. The interpretability layer is not a one-off analysis but a reusable pipeline: any future checkpoint gets the same thirteen outputs and the same report in nine minutes of CPU. What the white box has not yet done is pin down where the delay evidence sits in time and frequency (the per-window sweep and the attribution work), or state the model's stability knowledge in machining terms.

## 9. Obstacles and Next Steps

Obstacles:

1. The sampled-window AUC stays at 0.987 to 0.996 even at 512 windows, so the permutation magnitudes stay compressed. The orderings are reliable; the magnitudes are lower bounds.
2. The global tau sweep cannot locate the delay on a mixed sample. All five replication runs peaked at twice the physical value, a side effect of imposing one delay on windows from different operating points.
3. Gradient attributions under-credit lagged inputs on recurrent models, per the saliency literature, so the attribution side of the story stays weaker than the intervention side until TSR is added.
4. Single-window case studies are unstable across runs; claims about delay engagement need aggregate statistics.

Next steps, in order:

1. Make the tau sweep per-window, sweeping each window at multiples of its own tooth-pass delay, so the peak location becomes meaningful.
2. Add Temporal Saliency Rescaling to the attribution path and re-measure the delay's apparent share.
3. Gate the symbolic fits on R-squared and simplicity, so no edge receives a formula it cannot support, then move to end-to-end snapping.
4. Cross-check the delay with SINDy-DDE: recover tau straight from the raw signals, with no network involved, and compare.
5. Run the two ablations owed since Update 6, no-side-input training and the matched per-fold MLP table, now more pointed given the small side contribution at inference time.
6. Sweep spindle speed and depth through the trained model to draw its implied stability lobe diagram against the analytic one, the kind of global explanation a machining audience will actually read.

## 10. Summary

This update opened the model. A test-driven white-box pipeline turns any trained checkpoint into ranked, plotted, symbolically-fit edge functions, plus physics interventions, faithfulness checks, and an auto-generated report whose every number traces back to a manifest. Its first run also produced its first correction: a sampling ceiling had faked two "unused input" findings, and the fixed sampler withdrew them. The result that survives every check is the one the architecture was designed for. Zeroing the tooth-pass-delayed input collapses chatter detection on all five folds, to near chance on two of them, and breaks severity regression outright, while the delayed channels carry about twice their share of the learned weight and the important edges read position and delayed position through clean closed-form nonlinearities. The model does not merely contain the regenerative delay; it depends on it.

## 11. Time Spent

This update covers June 23 to July 1, about 20 hours of work.

| Work | Days | Hours |
|---|---|---|
| White-box design and spec | Jun 23 to 24 | 2.0 |
| Edge semantics, data-aware importance, attribution modules, with tests | Jun 24 | 3.0 |
| Symbolic extraction, physics interventions, faithfulness, case studies, with tests | Jun 24 to 25 | 3.5 |
| Runner, manifest-driven report renderer, cluster wrapper, end-to-end smoke | Jun 25 | 2.0 |
| First fold-2 run on the cluster and first analysis | Jun 25 | 1.5 |
| Ceiling-flaw diagnosis, sampler fix, corrected re-run | Jun 26 to 27 | 2.5 |
| Corrected fold-2 analysis and summary figures | Jun 27 | 1.0 |
| Checked literature review | Jun 27 | 2.0 |
| Replication runs on all folds and cross-fold analysis | Jul 1 | 1.5 |
| Docs and this update | Jul 1 | 1.0 |
| **Total hours** | | **20.0** |

The two largest blocks were the analysis modules themselves. Together, they are what turned "the architecture has a delay input" into "the trained model measurably depends on it, on every fold."
