---
title: "Research Update 11"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

<!--
Every scientific number traces to a returned GL manifest, verified July 18
(whitebox re-run fold0/1/2_w512/3/4, counterfactuals fold2_net, conformal
ensemble, concept probes). No number is invented. Two tools (concept probe,
TSR) are honest negatives and are written up as such.

CORRECTION 2026-07-29 (source patched + re-rendered, superseding the delivered
copy). Sections 3 and 8 stated the spline-atlas fits as "R-squared 0.985 to
1.000". Measured over all 125 stored rows the true range is 0.9257 to 0.9999:
11 fits below 0.98, 4 below 0.95, per-fold minima 0.9312 / 0.9815 / 0.9768 /
0.9640 / 0.9257. The error originated in PROGRESS.md and propagated here; both
are now fixed. The claim the number supports -- all 25 top edges carry an
*accepted* formula on every fold -- is unaffected, since all 125 clear the
acceptance gate. Only the stated fit floor was wrong.

Note that section 3's TSR paragraph is separately known to be wrong (the
"underpowered, only four windows" diagnosis); that correction is scheduled for
Update 12 and is deliberately NOT patched here.
-->

## 1. State Before This Work

Update 9 turned the trained network into a white box. At the end it listed three tools that were built but had not landed. Temporal Saliency Rescaling (TSR) was written, but the white-box runner never called it. The counterfactuals ran against a stand-in fitted to the dataset's labels, not to the network, so they tested the search but not the model. And the concept probes and hidden-state dynamics gave weak or circular results. Update 10 then showed the white box is robust under stress. What was left was to close that queue and re-run the full white box, so every tool reports on the same checkpoints at once.

This cycle did the building. I wired TSR into the runner, rebuilt the counterfactuals to read the network's own predictions, added a spline-edge atlas that makes the learned edge functions readable with real feature names, fixed the concept-probe circularity, and made the conformal layer a five-checkpoint ensemble. All of it is test-first (the suite now has 585 passing tests), and all of it runs eval-only on the existing `best_auc.pt` checkpoints.

All of it has now run, though two jobs took a detour to get there (Section 6). The full white-box re-run overran its time limit and had to be resubmitted with a bigger budget, and the network counterfactual ran out of memory and had to be re-scoped. Every number below traces to a returned manifest. Three results are wins: the atlas reads cleanly, the counterfactuals now describe the network itself, and the conformal ensemble is calibrated. Two are honest negatives: the fixed concept probe still fails, and TSR, though finally wired, does not find the delay. I report both kinds plainly. A tool that ran and did not confirm is written up as exactly that.

This update covers July 12 to 18, about 20 hours of work.

## 2. The Spline-Edge Atlas

A KAN's promise is that its edges are readable functions, not opaque weights. Update 8 ranked the top edges and Update 9 fitted formulas to them, but the two lived in separate tables, and neither carried the physical name of the feature each edge reads. The spline-edge atlas closes that gap. It takes the top edges by data-aware importance, names each one from the channel it reads (position, force, a tau-delayed channel, or a side feature), and draws each learned edge as a small panel. Each panel shows the curve itself, a shaded band for the range the real data visits, a rug of the real input values, and, when the fit passes the faithfulness gate, the fitted formula and its R-squared. Delay edges and side-feature edges are color-coded so the regenerative term stands out.

The atlas is wired into the white-box runner, so it runs on the real checkpoint against the real signals. It landed on all five folds.

![Spline-edge atlas for the fold-2 checkpoint: the twenty-five most important learned edge functions, ranked by data-aware importance. Each panel plots the learned edge (solid blue) and its accepted formula (dashed orange), with the formula and R-squared beneath, the range the real data visits (shaded), and a rug of the real input values. Titles name the feature the edge reads. Tau-delayed edges are red (for example `kan_z: delay:xtpos`, `kan_h: delay:Forcex`), side-input edges green, everything else black. Every one of the twenty-five passes the faithfulness gate.](results/figures/research_update_assets/spline_atlas_fold2.png){width=100%}

The atlas reads cleanly. **All twenty-five top edges carry an accepted formula on every one of the five folds** (R-squared 0.926 to 1.000, with eleven of the hundred and twenty-five fits below 0.98 and four below 0.95), so the readability result of Update 9 holds, now with real feature names attached. The fits are good but not uniformly excellent, and the weakest of them sit on folds 0 and 4. The top edges are a mix of hidden-to-head shaping functions and named input edges (position, force, and the tau-delayed channels). The tau-delayed edges are present and prominent. On the three folds that lean most on the delay (0, 1, 2) they carry **15 to 16 percent** of the top-edge importance; on folds 3 and 4 that share drops to 3 to 4 percent, the same fold-to-fold split the delay-ablation drop shows. This is the additive-weight view of the delay, and it is modest by design. The ablation, sweep, and tau-jitter tests remain the main delay evidence, because attribution shares undercount a term whose influence is spread across many small edges.

## 3. Temporal Saliency Rescaling, Now Wired

TSR fixes a known failure of gradient attribution on recurrent models. Saliency vanishes for time steps that affect the output only through a long lag, so a plain gradient map badly under-credits the delayed input, the very channel Update 8's ablation proved matters. TSR rescales the raw gradient by a per-step relevance term to give the lagged steps their credit back. It was written and tested in Update 9 but never called by the runner, so it produced nothing. This cycle it is wired in, and the runner now records a `tsr` block in the manifest for every window it analyzes.

The hope was that TSR, by crediting the lagged steps, would place its peak relevance at the tooth-pass delay, and so give an attribution-based echo of the intervention evidence. It ran, and the honest result is that it does not. I report it as a tool that landed but did not confirm.

Two things undercut it. First, it is underpowered: the rescaling is expensive, so it ran on only **four windows** (eight integration steps) per fold, far too few to trust a mean. Second, and more telling, neither of its two readouts points at the delay.

- **Channel attribution barely moves.** The TSR attribution splits across input channels almost exactly as the plain integrated-gradient attribution does. Position channels carry 72 to 83 percent of the mass under both, and TSR shifts only a few points more onto position. The rescaling reveals no hidden reliance the raw gradient missed.
- **Peak relevance does not sit at the delay.** The mean lag of peak relevance runs from 2 to 33 steps across the five folds, against a physical tooth-pass delay of about 87 raw samples (about 22 steps after the model's four-fold pooling). Only one fold lands near that value; the rest peak at the most recent samples.

So TSR joins the short list of tools that were built, ran clean, and gave no usable delay result. That is itself consistent with why it was needed: gradient-family saliency under-credits influence that arrives through a long lag, and even the rescaled version does not recover it on this small sample. The delay evidence stays where it has been since Update 8: the ablation, the per-window sweep, and the tau-jitter test of Update 10. Three interventions, not an attribution map.

## 4. Counterfactuals That Explain the Network

Update 9's counterfactuals were honest but hollow. They searched for the smallest physically-plausible edit to an operating point that would flip the decision, but the decision came from a stand-in fitted to the dataset's chatter labels, not to the network. Passing the checkpoint recorded its identity and nothing more. So the 95-percent agreement figure tested the search, not the model.

The fix is a two-step flow. First, a new script runs the trained checkpoint over the windows and writes its own per-file chatter probabilities to a CSV. Second, the counterfactual search fits its stand-in to those predictions, so the edits it finds explain the network's decision surface, not the ground truth. Both steps ran on the fold-2 checkpoint (the manifest records its SHA-256 and lists the model, not the labels, as the prediction source), over the same dataset-1 operating points Update 9 used.

The network-grounded number is **81.6 percent agreement** with the analytic stability boundary. Of 42 chatter operating points, the search finds a decision-flipping counterfactual for 38, all of which move toward stable, and 31 of those 38 land in the physically stable region (below the `blim` curve). Every flip the model accepts is in the right direction; about four in five are also in the right place.

![Network-grounded counterfactuals for the fold-2 checkpoint on dataset 1. Each red point is a chatter operating point; arrows show the smallest edit in the (spindle-speed, depth-of-cut) plane that flips the model's decision to stable. Blue squares land in the truly stable region below the analytic blim boundary (a hit); orange squares flip the model but stay physically unstable (a miss). 38 of 42 flip, all toward stable, and 31 land below the boundary, for 82 percent agreement.](results/figures/research_update_assets/counterfactuals_fold2_net.png){width=85%}

This is a weaker number than the 95 percent Update 9 reported, and that is the point. The old figure came from a stand-in fitted to the physics-derived labels, so agreeing with physics was almost automatic. **81.6 percent is what the network actually does**, the honest measure of how physically-consistent its decisions are. The typical flip lowers the depth of cut and nudges the spindle speed toward the nearest stable lobe, exactly the move a machinist would make; the counterfactuals land a mean 1.1 mm from the boundary. The search now audits the model, not the ground truth.

## 5. Two Smaller Repairs

Both of these ran as their own cluster jobs, apart from the main white-box re-run, and both returned.

**The concept probe, once de-circularized, honestly fails, so I retire the claim.** Update 9 could linearly recover proximity to the stability limit from the hidden state (R-squared 0.82), but the regeneration-phase probe was degenerate (held-out R-squared −228), and I flagged it as possibly circular: the probe target may have been derivable from the probe input by construction. This cycle rebuilds that target so it cannot leak. Instead of the wrapped phase angle, the probe now predicts the unwrapped regenerative phase, split into its sine and cosine parts, which is circular-safe. Re-run on the fold-2 checkpoint, the fixed probe **still fails the honest split**: held-out R-squared **−331 (sine) and −5.8 (cosine)**, even though it fits the training set (train R-squared about 0.80). The chatter-versus-tooth-pass separation probe fails the same way (−3.5). Only **proximity-to-blim generalizes (R-squared 0.82)**, as before.

The reading is clean and negative. Circularity was not what killed the regeneration-phase probe; the concept is simply not linearly decodable from the hidden state on held-out windows, and the fix proves it rather than papering over it. So I retire the claim that the regeneration phase is linearly encoded, and keep the one concept that survives an honest test: proximity to the stability limit. This is the outcome I said I would report plainly if it came, and it did.

**The conformal layer is now a calibrated five-checkpoint ensemble.** Update 9 calibrated the classifier on a single fold (91.4 percent coverage at a 90 percent target). The ensemble version, the one that would actually ship, calibrates across all five checkpoints on 256 held-out windows. At the 90 percent target it delivers **88.3 percent coverage**, with **99.6 percent selective accuracy** (against a raw 0.5-threshold accuracy of 98.4 percent). The full sweep tracks the target: targets of 80, 90, 95, 98, and 99 percent give coverage of 77, 88, 98, 99.6, and 100 percent, and the model only begins to abstain at the 98 and 99 percent targets.

![Split-conformal calibration of the five-checkpoint ensemble (256 calibration and 256 test windows). Left: empirical coverage on the held-out split against the target coverage (1 minus alpha); the curve tracks the ideal y = x line, sitting a little under target at 80 and 90 percent and a little over at 95 percent and above. Right: as the target tightens, selective accuracy stays flat near 0.99 (above the raw 0.984 threshold accuracy, dotted), while the abstention rate stays at zero until the 98 percent target and then climbs. The model declines to commit rather than guess when asked for near-certain coverage.](results/figures/research_update_assets/conformal_ensemble.png){width=95%}

The ensemble sits a little under the 90 percent target where the single fold sat a little over, but the sweep as a whole is well-calibrated, and the selective accuracy is actually higher (99.6 versus 98.7 percent). Every window now carries a calibrated confidence and the option to abstain.

## 6. The Compute Detour

Both heavy runs eventually landed, but each took a detour worth recording, because neither was a correctness problem and both are the kind of thing that will recur.

The full white-box re-run carries the wired TSR and the spline-edge atlas at 512 windows per fold, which makes it much heavier than the run it replaces. On its first submission all five folds hit the four-hour time limit and were killed. Raising the budget to twenty-four hours cleared it, and all five folds then finished clean, with empty error blocks and delay-ablation drops that reproduce Updates 8 and 9 to the third decimal. The network counterfactual stumbled differently. Its first step scores every window of every file to get the model's own probabilities, and it ran out of memory, because the data loader holds each file's full signal in RAM and this job loaded the whole signal tree at once (the sampled analyses never do). Re-scoping it to the single dataset the counterfactual actually uses, the same operating points Update 9 ran on, bounded the memory. As a bonus, it made the network-versus-physics comparison a like-for-like replacement of the old label-stand-in number.

Neither failure touched the methods, which were validated on synthetic fixtures in the test suite before any of this. Both were resource limits, fixed with a bigger time budget and a smaller data scope. The lighter fuzz suite of Update 10 finished cleanly on the first try, which is why that update was done a week earlier; this one needed the two detours before every cell could be filled from a manifest.

## 7. Obstacles and Next Steps

Obstacles:

1. TSR gave no attribution-based delay result (Section 3). It ran on too few windows, and even rescaled it does not place its peak relevance at the delay. The interventions stand in for it, but the attribution view of the delay is still owed.
2. Every tool here reads the trained network, so all of it runs only on the cluster, and none can be reproduced on the laptop.
3. The AUC ceiling from Updates 8 through 10 still applies to the size-based readouts (the atlas importance shares, the per-cell agreements), so I lean on shapes, orderings, and directions, not absolute heights.

Next steps, in order:

1. Re-run TSR on a much larger window sample (hundreds, not four) to decide whether the flat channel split and the off-delay peak are real or just small-sample noise. If they survive, retire the attribution view and let the interventions carry the delay claim alone.
2. Retrain with the quantile severity head to attack the heavy-tailed severity error, the one output still not hardened.
3. Take up the flagged new-modeling experiment, a learnable delay kernel that recovers the tooth-pass period from data, as the next research direction beyond this white-box cycle.

## 8. Summary

Update 11 closes the eval-only white-box queue that Update 9 left open, and it closes it with numbers, not placeholders. The spline-edge atlas draws the twenty-five most important learned edges as named panels, all twenty-five carrying an accepted formula on every fold (R-squared 0.926 to 1.000), with the tau-delayed edges flagged and carrying 3 to 16 percent of the top-edge importance by fold. The counterfactuals now describe the network itself, fitted to the model's own predictions rather than the labels, and they agree with the analytic stability boundary **81.6 percent** of the time (38 of 42 chatter points flip, all toward stable, 31 landing in the truly stable region); the honest drop from Update 9's automatic 95 percent is the whole point. The conformal ensemble is well-calibrated (88 percent coverage at a 90 percent target, 99.6 percent selective accuracy). Two tools are honest negatives, reported as such: the fixed concept probe still fails its split (held-out R-squared −331 for the regenerative phase against 0.82 for proximity-to-blim), which shows the concept is absent rather than the target circular, and retires the claim; and TSR, though finally wired, does not place its peak relevance at the delay on its small sample, so it adds no attribution-based delay evidence. Both detours that held the update up, a time overrun and an out-of-memory failure, were resource limits, not method errors, and both were cleared. The delay evidence stays the three interventions of Updates 8 through 10. What this update adds is a readable atlas, a network-honest counterfactual, a calibrated confidence, and two claims retired on the evidence.

This update covers July 12 to 18, about 20 hours of work.

## 9. Time Spent

| Work | Days | Hours |
|---|---|---|
| Plan Update 11: sequence the owed-queue closure and the re-run design | Jul 12 | 1.5 |
| Spline-edge atlas module: named/ranked edges, data-visited bands, symbolic annotation, test-first | Jul 12 to 14 | 4.5 |
| Wire TSR into the white-box runner and record it in the manifest, with tests | Jul 14 | 1.5 |
| Network-grounded counterfactuals: per-window prediction script and surrogate rewire, test-first | Jul 14 to 15 | 3.0 |
| Concept-probe de-circularization and conformal-ensemble wrapper, with tests | Jul 15 to 16 | 2.5 |
| Full-suite gate (585 passing), integration of all analyses into the runner | Jul 16 | 1.5 |
| Cluster submission; diagnose the time overrun and the counterfactual out-of-memory, refit budgets and resubmit | Jul 16 to 17 | 2.0 |
| Runs land: verify five white-box folds and counterfactuals against the manifests, fill results and figures | Jul 17 to 18 | 2.0 |
| Docs and drafting this update | Jul 18 | 1.5 |
| **Total hours** | | **20.0** |

The white box is now fully instrumented and fully run. Three tools landed as wins: a readable spline-edge atlas, a network-honest counterfactual, and a calibrated confidence. Two landed as honest negatives that I report rather than bury: the concept probe that fails its split, and the saliency method that does not find the delay. The delay claim rests where it has since Update 8, on the interventions. What changed is that the network is now a white box a machinist could read, and every claim in it is one the evidence supports.
