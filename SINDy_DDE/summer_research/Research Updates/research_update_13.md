---
title: "Research Update 13"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

<!--
Covers July 26 to August 1. Drafted 2026-07-29, so it reports work through
July 29 and the window has three days left.

Every number traces to a returned manifest: the counterfactual manifest from GL
job 55405362 and the five regenerative manifests from GL jobs 55408256 to 60,
all fetched July 29. No number is invented.

NO PLACEHOLDERS. Updated 2026-07-30, after the window's drafting date: Section 6
now reports the TSR control's returned result (it answered against the effect,
see Update 12 Section 4) and the band-notch code fix landing, with only the
cluster re-run still owed. Section 7's hours are unchanged and still cover
July 26 to 29 only; the July 30 work is booked to Update 14.

Language simplified 2026-07-30 for readability. No number, caveat, negative
result or self-correction was changed; verified by comparing the multiset of
every numeric token before and after.
-->

## 1. What this cycle was for

Two of the plan's success criteria still had no answer on disk. Both turned out to be cheap to run.

The first is criterion 7. It asks for DiCE counterfactuals on at least ten unstable operating points, each with a Monte Carlo survival rate above 0.85. A counterfactual here is the smallest change to the cut, in spindle speed or depth, that would have made a chattering cut stable. DiCE is a general-purpose tool for generating such changes. The survival rate says how often a change still works once you allow for normal machining tolerance. The counterfactual code had been reporting a number since Update 9. That number came from the physics-constrained grid search rather than DiCE, and nothing in the repository computed survival at all.

The second is the qualitative delay criterion. It asks whether the network's learned expression contains x(t) − x(t−τ) without anyone hardcoding that difference. I had answered it indirectly through SINDy-DDE, which recovers τ without the network at all. I had never answered it by reading the network's own learned functions.

Three arms, all local, all eval-only. Two came back negative. The third refutes the hypothesis it was built to test.

## 2. How risky is a recommendation that only holds at the nominal point?

Slightly more than half of what this model recommends would not survive ordinary shop-floor variation.

The counterfactual search answers an operator's question. What change in spindle speed or depth of cut flips this cut from chattering to stable? Update 9 reported that 81.6 percent of the flips it found also land in the physically stable region below the analytic boundary. That number describes the nominal recommendation. A machinist never gets the nominal recommendation. Spindle speed drifts. Depth of cut carries a setup tolerance. The boundary itself is only known to within the uncertainty on the cutting coefficient and the modal parameters.

So I sampled that uncertainty. For each recommendation I drew 512 randomly perturbed versions of the operating point and counted the fraction still stable, which is the Monte Carlo survival rate. The perturbations were 2 percent relative jitter on spindle speed, 0.05 mm absolute tolerance on depth, and 10 percent relative uncertainty on the boundary. Each draw perturbs the operating point and the boundary together, since b_lim scales inversely with the cutting coefficient.

![Survival of each of the 42 counterfactual recommendations under machining uncertainty, sorted. Each bar is one recommendation. Its height is the fraction of 512 perturbed draws that stay below the stability boundary. Blue bars clear the 0.85 target the project plan asks for; orange bars do not. The plan asks for at least ten such points and 19 qualify, so the criterion is satisfied. The distribution is the more useful result. The median recommendation survives 68 percent of draws and the worst survives 9 percent, so slightly more than half of what this model recommends would not survive ordinary shop-floor variation.](../../results/figures/research_update_assets/u13a_counterfactual_survival.png)

**19 of 42 recommendations clear 0.85.** The mean survival is 0.702, the median 0.685, and the minimum 0.086. The criterion asks for at least ten points above 0.85 and gets 19, so it is met. I would rather report the distribution than the count, because the count flatters the model and the distribution does not. A recommendation that survives 9 percent of draws is worse than useless on a real machine. It would read as advice while being a coin flip weighted against you.

Two notes on how the code is built. I wrote the survival calculation as a sibling of the existing boundary check instead of an extra argument to it. So the 81.6 percent agreement number it produces is untouched and stays comparable to Update 9. A test asserts exactly that. Running it with zero draws reproduces the old nominal check exactly, which is a consistency check on the new code rather than a separate feature.

The headline figure depends on the tolerances I assumed. Ten percent on b_lim is the dominant term, and it is a judgement rather than a measurement. If the real modal uncertainty is tighter, the survival rates rise. So the number to quote alongside 19 of 42 is the tolerance set, not just the count.

## 3. Two counterfactual generators, and a comparison I got wrong first

DiCE has been in the repository since early June and had never been run. I ran it on the same 42 operating points, against the same surrogate the physics-constrained search uses. The surrogate is the stand-in model both methods query. DiCE knows nothing about the stability boundary, so it gives a second opinion. It found a counterfactual for all 42 cases, and 36 of those, 85.7 percent, land in the physically stable region, against the physics-constrained search's 81.6 percent. Two independent generators, near-identical hit rates, and nothing shared between them but the surrogate.

My first attempt to compare them was wrong, and how it was wrong is worth more than the fixed number. I compared the direction each generator moved the depth of cut and reported 12 of 38 agreement, which reads as two methods contradicting each other. They do not contradict each other. The physics-constrained search cannot raise the depth of cut, because its search range is capped at the current depth by construction. DiCE varies both axes freely. My metric scored "chose the other knob" as "disagrees". The low number was an artifact of comparing a constrained search against an unconstrained one on the axis where the constraint bites.

![The two generators solve the same 42 cases by different means and arrive at comparable places. Panel A shows the recommended move itself, as a change in speed against a change in depth. The physics-constrained points sit along the horizontal line where depth does not change, spread widely in speed, because that search hops the spindle to a neighbouring stability lobe at full depth. The DiCE points sit below that line, because DiCE cuts depth. Panel B counts the same thing. The physics search changes speed alone in 35 of 42 cases; DiCE touches depth in 36, either alone or together with speed. Neither is more correct. They answer the operator's question with different levers.](../../results/figures/research_update_assets/u13b_two_generators.png)

The replacement reports two things separately. The axis breakdown says which knob each generator turned, as a description rather than a score. Agreement is then measured on where the move lands, meaning both generators reach the stable region or both fail to. On that measure they **agree on 27 of 42 cases, and 26 of those are cases where both succeed.**

Reporting 0.643 agreement needs one caveat stated in the same breath. It is agreement on a binary outcome, so chance agreement is not negligible, and the count of joint successes, 26, carries more information than the rate. The manifest is the results file each run writes out. It now says outright that the axis fields are not an agreement measure, so a reader of the artifact cannot repeat my mistake.

## 4. The network learned the sum, not the difference

Regenerative chatter is a delay-differential phenomenon. The cutting force depends on the difference between the tool's current position and its position one tooth-pass earlier, x(t) − x(t−τ). That difference is the chip thickness variation that feeds the instability. The plan asks whether the trained network discovered that difference on its own.

The architecture makes this directly testable. Every delayed input arrives on a known edge of the bottom encoder gates, index 68 for delayed x-position and 69 for delayed y-position. The matching instantaneous channel arrives on its own edge of the same unit. Each edge carries a learned spline, a flexible curve the network fits to that input. If the network implements regeneration, then within a unit the delayed edge's spline should be roughly the negative of the instantaneous edge's, because adding those two gives the difference. So I evaluated both splines over the input range the data actually visits and scored how close one is to the negative of the other. A score of +1 means an exact negation, −1 an exact copy, 0 unrelated.

I ran this on all five folds, each fold being one train and test split of the data. Every fold came back strongly negative.

![The regenerative test on all five folds. Panel A pools every delay-edge pair from all five folds, 3840 pairs, against the cross-channel control of 11520 pairs. The score axis runs from an exact copy on the left to an exact negation on the right, with the plan's regenerative difference at the right-hand end. The delay-edge mass sits hard against the copy end. Panel B gives each fold's mean beside its two controls. The cross-channel control pairs edges that have no delay relationship. The untrained null is the same measurement on a randomly initialised network of identical architecture. The signal is 20 to 25 times the untrained null and replicates on all five folds, so it is a learned structure. It is not the structure the plan predicted.](../../results/figures/research_update_assets/u13c_regenerative_sum_not_difference.png)

Mean scores by fold are −0.533, −0.639, −0.526, −0.574 and −0.457. Weighting each pair by the importance of its two edges moves them further negative, to between −0.54 and −0.71. Under a threshold of 0.9 in either direction, 23 to 36 percent of pairs qualify as summing and 0.1 to 1.0 percent as negating, roughly thirty to one in the wrong direction. The most important pairs are the most extreme. On fold 0 the top-ranked pair scores −0.989, with a correlation of +0.997 between the two edge functions, which is very nearly an exact copy at matched amplitude.

**The network learned x(t) + x(t−τ), a sum, not the regenerative difference.** The plan's qualitative criterion is not met, on the most direct test available for it.

I am reporting this as a result rather than a bug because the controls hold. The untrained network of identical architecture scores −0.023, flat across all five folds. Its negating and summing fractions are nearly symmetric at 10.3 and 12.5 percent, which is what no structure looks like. The cross-channel control, pairing edges with no delay relationship, scores between −0.08 and −0.18. And the fraction of pairs where the plain SiLU base term dominates the learned spline is near zero, so the effect lives in the learned functions rather than in the architecture's fixed part.

What a sum does physically is the interesting part. x(t) + x(t−τ) is a comb filter with peaks at multiples of the tooth-pass frequency, so it reinforces content synchronous with tooth passing. The difference does the opposite. It notches synchronous content and passes what is asynchronous, which is where chatter lives. A classifier that wanted to detect chatter by isolating the asynchronous part would build the difference. This one built the reinforcing filter instead, which is a reasonable way to measure how strong the synchronous component is and therefore how much of the signal is *not* chatter. It is still delay-coupled, and it still needs τ to exist, which is consistent with the three interventions that ground the delay claim. It is not the regenerative term.

One control did not corroborate, and I would rather say so than leave it for a reader to find. Folds 0, 1 and 2 carry five delay edges each among their top 25 by importance, while folds 3 and 4 carry one, and the delay-ablation drops split the same way. If the summing structure were carrying the delay reliance, I would expect it to track that split. It does not. Fold 3, with one top-ranked delay edge, scores −0.574, stronger than folds 0 and 2. So the summing structure is real and learned, but I cannot yet tie it to the fold-level delay reliance.

## 5. Two jobs that exited cleanly without doing the work

Both failures this cycle were in my own wrappers, and both produced logs that read as successes.

Five regenerative jobs finished in about 90 seconds each, exit code 0, and wrote a small output file. The runner has a checkpoint-free fallback mode that scores pairs straight from a saved edge list without loading the network, selected by a command-line flag. My wrapper passed that flag whenever the edge-list file happened to exist, which it did for all five folds, so every job took the fallback and skipped the analysis. The wrapper's final line then printed the path of the real manifest, unconditionally, so the log named a file that was never written. I fixed it twice over. The fallback is now opt-in only and never inferred from a file existing, and the last step asserts its artifact exists rather than announcing it. The re-run took real time and produced the results in Section 4.

The TSR control from Update 12 died in seconds on a missing function argument in its provenance step, in code that had been there for days. Its previous run had spent a full 12-hour wall and hit the same line at the very end, where the failure was indistinguishable from a timeout. Moving that assembly to the start of the run turned a 12-hour loss into an immediate error, which is how it got found.

The lesson generalises past these two. A job exiting 0 is not evidence that it did the work. Two of the three arms this cycle had wrappers that could report success while skipping the analysis, and in both cases the fix was structural. Never infer a mode from a file existing, and make the last line of a wrapper assert its artifact instead of describing it. None of the other cluster wrappers in this project assert anything, which is now on the list.

## 6. Still open in this window

The band-notch spectral contrast from Update 12 needed two defects fixed and a re-run. Both fixes landed on July 30, after this window closed, so only the cluster re-run is still owed. It now aims each notch using that window's own frequencies instead of a batch average, and it counts the windows where the halfwidth solver failed instead of reporting the failure as a real measurement. Fixing it turned up a third defect I had not looked for: the old check for missing frequency values tested the batch average, so a single bad row threw away all 512 windows.

The TSR construction-artifact control returned on July 30, on its third submission. It answers against the result. A network with random weights locks onto the delay harder than the trained one, so the effect is built into the input rather than learned. Update 12, Section 4 carries the numbers. Its verdict rule needed the correction described there, and applying it does not change the answer, it strengthens it.

The counterfactual manifest currently on disk carries the superseded direction metric, because it was produced before the fix. The corrected figures in Section 3 are recomputed from the per-case coordinates in that same manifest, which is exact, but the artifact should be regenerated so it matches the code that describes it.

## 7. Time spent

| Work | Days | Hours |
|---|---|---|
| Monte Carlo survival: design as a sibling of the boundary check, implement, test-first | Jul 26 | 3.0 |
| Wire survival and DiCE into the counterfactual runner; expose the fitted surrogate; guards for a missing DiCE install | Jul 26 | 2.5 |
| Regenerative spline test: edge-pair scoring, cross-channel control, untrained null, test-first | Jul 26 | 4.0 |
| Cluster wrappers for the regenerative test and the U13 counterfactual run; sync and fetch script gaps | Jul 29 | 1.5 |
| Diagnose and fix the two false-success wrapper bugs; re-run the five regenerative folds | Jul 29 | 2.0 |
| Replace the counterfactual comparison metric with the axis breakdown and outcome agreement, test-first | Jul 29 | 1.5 |
| Read the returned results; recompute the corrected comparison; four figures | Jul 29 | 2.5 |
| TSR control survivability work: progress reporting, per-condition checkpointing, provenance crash | Jul 29 | 1.5 |
| Docs and drafting this update | Jul 29 | 1.0 |
| **Total hours** | | **19.5** |
