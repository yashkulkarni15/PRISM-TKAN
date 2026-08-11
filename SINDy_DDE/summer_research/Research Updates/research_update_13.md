---
title: "Research Update 13"
date: "July 26 to August 1, 2026"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

<!--
Covers July 26 to August 1; the work reported runs to July 29. Rewritten
shorter 2026-08-02.

Every number traces to a returned manifest: the counterfactual manifest from GL
job 55405362 and the five regenerative manifests from GL jobs 55408256 to 60,
all fetched July 29. No number is invented.

NOTHING IN THIS UPDATE IS OPEN. July 30 work is booked to Update 14, and
nothing was worked on July 31 or August 1.

REWRITE 2026-08-02: cut for length and plain language at Yash's request, then
cut again. No number, negative result, caveat or self-correction was removed.
Every numeric token here also appears in the version sent before it (checked
mechanically, superset test). What went: definitions of terms Abell already
knows (fold, surrogate, held-out), the two-paragraph wrapper-bug narrative
condensed to its lesson, old section 6's restatement of Update 12's results,
and repeated framing in sections 1 to 3.

Record of corrections carried by this update, all still present in the text:
  - Sec 3 corrects a metric I got wrong first: comparing the direction each
    generator moved depth of cut gave 12 of 38 agreement, which read as two
    methods contradicting each other. The physics-constrained search cannot
    raise depth (capped at current depth by construction), so the metric scored
    "chose the other knob" as "disagrees". Replaced by an axis breakdown plus
    outcome agreement, 27 of 42.
  - Sec 4 REFUTES the plan's qualitative delay criterion. The paired edge
    functions are copy-like rather than the negations expected from the
    regenerative difference x(t)-x(t-tau). Reported as a result, not a bug,
    because all three controls hold.
  - Sec 4 also records the one control that did not corroborate: the summing
    structure does not track the fold-level split in delay-edge importance.
  - Sec 1's "all local" (earlier draft) was wrong; the arms are eval-only but
    ran on Great Lakes. Corrected 2026-08-02.
  - The counterfactual manifest on disk carries the superseded direction
    metric. Nothing published depends on it: the Sec 3 figures are recomputed
    from per-case coordinates and are exact. Regenerating it needs a cluster
    job (it loads the fold 2 checkpoint), so it is booked to Update 14 Sec 6
    rather than left open here.
-->

## 1. Summary

This cycle tested two open criteria.

- **Counterfactual recommendations:** 19 of 42 tested operating points remain physically stable in more than 0.85 of the Monte Carlo draws, so the requirement of at least ten is met. The median survival rate is only 0.685, so many recommendations are not reliable under the assumed machining tolerances.
- **Independent counterfactual generator:** DiCE finds a counterfactual in all 42 cases. The physics-constrained search finds a model flip in 38 of 42 cases. Their physical-validity rates are reported separately in Section 3.
- **Learned delay functions:** the delayed and instantaneous edge functions are mostly copy-like rather than opposites. The expected regenerative difference x(t) − x(t−τ) is not present in this test.

## 2. How risky is a recommendation that only holds at the nominal point?

More than half of the tested candidates would not survive ordinary shop-floor variation under the assumed tolerances.

Update 9 reported that 81.6 percent of the flips the counterfactual search finds also land in the physically stable region below the analytic boundary. That describes the nominal recommendation, and a machinist never gets the nominal recommendation. Spindle speed drifts, depth of cut carries a setup tolerance, and the boundary itself is only known to within the uncertainty on the cutting coefficient and the modal parameters.

So I sampled that uncertainty. For each of the 42 proposed operating points, I drew 512 perturbed versions and counted the fraction that remained physically stable. The physics-constrained search changed the model prediction in 38 of the 42 cases; the other 4 are retained as failed proposals. The perturbations were 2 percent relative jitter on spindle speed, 0.05 mm absolute tolerance on depth, and 10 percent relative uncertainty on the boundary. Each draw perturbed the operating point and the boundary together because b_lim scales inversely with the cutting coefficient.

![Survival of the 42 proposed operating points under machining uncertainty, sorted. Each bar's height is the fraction of 512 perturbed draws that stay below the stability boundary. Blue bars clear the plan's 0.85 target and orange bars do not, with 19 qualifying against a requirement of ten. The median proposal survives 68.5 percent of draws and the worst 8.6 percent.](../../results/figures/research_update_assets/u13a_counterfactual_survival.png)

**19 of 42 proposals clear 0.85.** The mean survival is 0.702, the median is 0.685, and the minimum is 0.086. The criterion asks for at least ten points above 0.85 and gets 19, so it is met. The full distribution is still important because the result is not strong enough to treat every proposal as usable machining advice.

The headline depends on the tolerances I assumed. Ten percent on b_lim is the dominant term and it is a judgement rather than a measurement, so tighter modal uncertainty would raise the survival rates. The number to quote alongside 19 of 42 is the tolerance set.

I wrote the survival calculation as a sibling of the existing boundary check rather than an extra argument to it, so the 81.6 percent number stays untouched and comparable to Update 9. Running it with zero draws reproduces the old nominal check exactly, and a test asserts that.

## 3. Comparing two counterfactual generators

DiCE was already in the repository but had never been run. I ran it on the same 42 operating points against the same surrogate used by the physics-constrained search. DiCE does not use the stability boundary while generating its answer.

The two methods need separate denominators. The physics-constrained search produces a model flip in 38 of 42 cases, and 31 of those 38 flips are physically stable, giving 81.6 percent conditional validity. DiCE produces a counterfactual in all 42 cases, and 36 of the 42 are physically stable, giving 85.7 percent. On the common 42-case denominator, the physically stable totals are 31 of 42, or 73.8 percent, for the physics search and 36 of 42, or 85.7 percent, for DiCE.

My first comparison was wrong. I compared the direction each generator moved the depth of cut and reported 12 of 38 agreement. The physics-constrained search cannot raise the depth of cut because its search range is capped at the current depth, while DiCE varies both axes freely. The metric therefore counted "chose the other knob" as "disagrees".

![The two generators attempt the same 42 cases by different means. Panel A plots each proposed move as a change in speed against a change in depth. The physics-constrained points sit on the line where depth does not change because that search hops the spindle to a neighbouring stability lobe at full depth, while the DiCE points sit below it because DiCE often cuts depth. Panel B counts the same thing: the physics search changes speed alone in 35 of 42 cases, while DiCE touches depth in 36.](../../results/figures/research_update_assets/u13b_two_generators.png)

The replacement reports two things separately. The axis breakdown says which knob each generator turned, as a description rather than a score. Agreement is then measured on where the move lands, meaning both generators reach the stable region or both fail to. On that measure they **agree on 27 of 42 cases, and 26 of those are cases where both succeed.**

The 0.643 agreement rate is based on a binary outcome, so chance agreement is not negligible. The count of 26 joint successes is more informative. The runner now records that the axis fields describe which control was changed and are not an agreement score.

## 4. The learned edge functions copy rather than negate

Regenerative chatter is a delay-differential phenomenon. The cutting force depends on x(t) − x(t−τ), the difference between the tool's current position and its position one tooth-pass earlier, and that difference is the chip thickness variation that feeds the instability. The plan asks whether the trained network discovered it.

The architecture makes this directly testable. Every delayed input arrives on a known edge of the bottom encoder gates, index 68 for delayed x-position and 69 for delayed y-position, and the matching instantaneous channel arrives on its own edge of the same unit. Each edge carries a learned spline. If the network implements regeneration, then within a unit the delayed edge's spline should be roughly the negative of the instantaneous one, because adding those two gives the difference. So I evaluated both splines over the input range the data actually visits and scored how close one is to the negative of the other: +1 is an exact negation, −1 an exact copy, 0 unrelated.

Every fold came back strongly negative.

![The regenerative test on all five folds. Panel A pools all 3840 delay-edge pairs against the cross-channel control of 11520 pairs, on an axis running from an exact copy on the left to an exact negation on the right, where the plan's regenerative difference would sit. The delay-edge mass sits near the copy end. Panel B gives each fold's mean beside its two controls. The copy-like pattern is 20 to 25 times the untrained null and appears on all five folds. It is not the negating structure predicted by the plan.](../../results/figures/research_update_assets/u13c_regenerative_sum_not_difference.png)

Mean scores by fold are −0.533, −0.639, −0.526, −0.574 and −0.457. Weighting each pair by the importance of its two edges moves them further negative, to between −0.54 and −0.71. Under a threshold of 0.9 in either direction, 23 to 36 percent of pairs qualify as summing and 0.1 to 1.0 percent as negating, roughly thirty to one in the wrong direction. The most important pairs are the most extreme: on fold 0 the top-ranked pair scores −0.989, with a correlation of +0.997 between the two edge functions, very nearly an exact copy at matched amplitude.

**The paired edge functions are copy-like rather than the negations expected from x(t) − x(t−τ).** The plan's qualitative criterion is not met.

The controls show that this is a learned edge-level pattern. The untrained network of identical architecture scores −0.023 across the five folds, with negating and copy-like fractions of 10.3 and 12.5 percent. The cross-channel control scores between −0.08 and −0.18. The plain SiLU base term dominates almost none of the tested pairs, so the pattern is in the learned spline functions.

If these paired functions acted linearly and additively, the copy-like pattern would be consistent with x(t) + x(t−τ), which reinforces tooth-synchronous content. This edge-level analysis does not prove that the complete recurrent network implements that global comb filter. What it does show is that the tested functions do not form the expected regenerative difference.

One control does not corroborate a stronger interpretation. Folds 0, 1 and 2 each have five delay edges among their top 25 by importance, while folds 3 and 4 have one, and the delay-ablation drops split the same way. The copy-like score does not follow that split: fold 3 scores −0.574, stronger than folds 0 and 2. The edge-level pattern is learned, but it is not yet tied to the fold-level delay reliance.

## 5. Two jobs that exited cleanly without doing the work

Five regenerative jobs finished in about 90 seconds with exit code 0 but had taken a fallback path and skipped the intended analysis. The wrapper now requires that mode explicitly and verifies that the expected manifest exists. The TSR control from Update 12 had a separate manifest error that was hidden by a previous 12-hour timeout; moving provenance checks to the start turned it into an immediate error.

The practical change is simple: a successful exit code is no longer treated as proof that the analysis ran. Wrappers now need to assert that their expected artifact was produced.

## 6. Project status

- **Established:** the counterfactual criterion is met under the stated tolerance assumptions, with 19 of 42 proposals above 0.85 survival.
- **Rejected:** the tested edge functions do not show the expected regenerative difference.
- **Next:** regenerate the counterfactual manifest with the corrected comparison fields and apply artifact checks to the remaining cluster wrappers.

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
