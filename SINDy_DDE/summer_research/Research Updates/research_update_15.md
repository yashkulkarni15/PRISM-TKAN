---
title: "Research Update 15"
date: "August 9 to 15, 2026"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

<!--
Covers August 9 to 15. DRAFTED 2026-08-11, so Sections 2 to 5 are complete and
Section 6 lists what the window still has running.

PROVENANCE. Every number traces to a returned artifact.
  - Section 2 comes from results/phase3/xai/comb/comb_manifest.json, derived
    entirely from the stored U13-C regenerative manifests. No checkpoint was
    loaded and no window re-scored: the manifest records network_reloaded=false.
    Physical frequencies (tooth-pass 506.8 Hz, fn 1072.3 +/- 492.3 Hz) come from
    the band-notch block of the fuzz manifests, which are the only per-fold
    physical frequencies on local disk.
  - Section 3 comes from results/phase3/xai/comb/comb_probe_*.json, five CPU
    jobs on the `standard` partition, about 22 seconds each.
  - Section 5's determinism work is code plus two SUBMITTED jobs (56977333,
    56977341). They had not returned when this was written and nothing here
    claims a determinism result.
  - Binomial p-values computed exactly, not normal-approximated.

HOURS in Section 7 are an ESTIMATE and need checking before this is sent.
-->

## 1. Summary

This cycle asks what the delayed input is actually used for, and answers it with a filter.

- **The delay coupling is a comb filter tuned to the tooth-passing frequency.** Update 13 found that the network pairs each delayed input with a scaled copy of its instantaneous partner, and recorded that as a failed hypothesis. It is not a failure. That pairing is a delay-and-add filter whose passbands land on the tooth-pass frequency and its harmonics.
- **Training does not build the filter. It chooses which way the filter points.** Randomly initialised networks already form the same comb, but with the sign decided by a coin flip. Trained networks reinforce the tooth-passing family in 110 of 125 measured cases; untrained ones land at 64 of 125, which is chance.
- **The prediction was confirmed by direct measurement** on all five networks, agreeing to a correlation of 0.993 or better.
- **Three naming and logic defects were fixed** before they could reach a written methods section.
- **Next:** the reproducibility test is queued behind a shared cluster limit, and the noise floor it feeds is the gate on every remaining comparison.

## 2. What the delayed input is used for

Update 13 asked whether the network had learned the regenerative difference, the quantity `x(t) - x(t-tau)` that classical chatter theory is written in. It had not. For each delayed edge, the matching instantaneous edge on the same unit was a near-exact scaled copy rather than a negation, which makes the composition a sum. I reported that as a refuted hypothesis and moved on.

That was the wrong place to stop. A sum is not the absence of structure, it is a different structure, and it has a name.

If the two learned curves are proportional, which is what a correlation near one measures, then the unit computes the signal plus a scaled copy of itself delayed by `tau`. That is a delay-and-add filter. Its response has peaks wherever the delay is a whole number of cycles and cancellations halfway between. Because the model fixes `tau` at one tooth-passing period for each cut, those peaks land exactly on the tooth-passing frequency and its harmonics, and they move with the spindle speed automatically.

The difference the plan hoped for would have put cancellations in exactly those places instead. So the choice between a sum and a difference is a choice between reinforcing the forced vibration and erasing it, which is a statement about what the network computes rather than about which hypothesis lost.

Nothing here assumes the learned functions are straight lines. The proportionality is between the two curves, so the filter acts on the shaped signal whatever shape the network gave it.

![The learned delay pair, read as a filter. The blue curve is the response of one instantaneous and delayed edge pair as a function of frequency. Green dashed lines mark the tooth-passing frequency and its harmonics at 507, 1014, 1521 and 2028 Hz. The response peaks on those lines and cancels between them. The red band is where chatter occurs, 1072 Hz plus or minus 492 Hz, which spans both peaks and cancellations rather than sitting neatly in one.](../../results/figures/research_update_assets/comb_response.png){width=85%}

Measured across all five networks, of the pairs proportional enough for this reading to describe them, between 91 and 99 percent reinforce the tooth-passing family. The contrast between peak and cancellation is 15.5 to 17.7 decibels. The same measurement on control pairs, built by pairing a delayed channel with a different channel, gives 63 to 72 percent and 11.3 to 13.3 decibels, with no overlap on any network. That control is a demanding one, because different channels of the same cut are genuinely correlated.

The pattern is uniform rather than local. All three gates of the recurrent cell carry it, and all four input channels do.

**Two things this does not show, stated plainly.** Only 47 percent of pairs are proportional enough for the reading to apply, so this describes about half the delay coupling and not all of it. And the strength does not follow the split seen in the earlier ablation, where two networks lean on the delayed input far less than the other three. Update 13 flagged that same failure to corroborate, and it has not gone away.

## 3. Confirming it on the network

The section above predicts each pair's frequency response from two stored numbers. A prediction from summary statistics is not a measurement, so I made the measurement.

For each of the top 25 pairs in each network I drove the two learned functions with a sweep of pure tones and recorded the response at each frequency. This needs no GPU, because it evaluates learned curves rather than running data through the model. Each of the five jobs finished in about 22 seconds on an ordinary compute node.

| Network | Peaks on a tooth-pass harmonic | Untrained control | Agreement with prediction |
|---|---|---|---|
| 0 | 0.920 | 0.600 | 0.9974 |
| 1 | 0.840 | 0.520 | 0.9962 |
| 2 | 0.880 | 0.520 | 0.9975 |
| 3 | 0.920 | 0.440 | 0.9958 |
| 4 | 0.840 | 0.480 | 0.9928 |

The predicted and measured responses agree to a correlation of 0.993 or better on every network, so the reading in Section 2 describes the real learned functions.

**The control is the more interesting column.** Pooled, trained pairs peak on a tooth-passing harmonic in 110 of 125 cases and untrained pairs in 64 of 125. That untrained figure is a coin toss, at p equal to 0.86.

A coin toss is the right comparison, and this is the part that changes the claim. At random initialisation the two curves of a pair are both close to straight lines, and any two straight lines are almost perfectly proportional. So an untrained network already forms this filter. What it does not have is a preferred direction: half its pairs reinforce the tooth-passing family and half cancel it. Against that baseline the trained networks reinforce 110 times out of 125, at p equal to 2 times ten to the minus 19.

So the honest claim is narrower than "the network built a filter", and stronger for being narrower:

> The delay-and-add structure comes from the architecture. What training decides is the sign, and training reliably chooses to reinforce the tooth-passing family rather than cancel it.

![Swept response of the strongest delay pair in each network. Blue is measured by driving the learned functions with pure tones, red dashed is predicted from the stored correlation and gain, and grey is the same measurement on a randomly initialised copy. Blue and red coincide everywhere. The grey curve is a filter of the same shape, but on four of the five networks it is inverted, peaking exactly where the trained one cancels. That inversion is the whole finding: the shape is architecture, the direction is learned.](../../results/figures/research_update_assets/comb_probe_all_folds.png){width=100%}

**A defect in my own measure, found and corrected before it reached you.** The first version of this table took the highest point of each measured response and asked whether it sat on a tooth-passing harmonic. A shaped edge produces a constant offset as well as a tone, and for 44 of the 125 pairs that constant was the highest point. Because zero counts as a whole number, those were scored as hits. Excluding the constant term lowers the trained figure from 0.912 to 0.880 and the untrained one from 0.576 to 0.512. The conclusion is unchanged and the control is cleaner for it, since 0.512 is a coin toss by any reading. The numbers above are the corrected ones.

**Where the measurement disagrees with the prediction.** The median agreement of 0.996 hides a tail. Four of the 125 pairs measured below 0.90 against their prediction, one of them strongly negative, and all four sit on the same gate and the same input channel, so this is systematic rather than noise. Two of the four fall below the proportionality threshold that Section 2 requires, which means the analysis already declines to describe them and the measurement agrees it should. The other two clear that threshold and still disagree. Disagreement runs at 12.5 percent below the threshold against 1.8 percent above it, so the threshold helps without being sufficient.

I checked whether the probe itself was at fault, by driving the functions harder than the data ever does. It is the reverse: the four disagreeing pairs are the most strongly driven edges in the data, not the least. I have left that unexplained rather than invent a reason.

## 4. What this changes, and what it does not

The delay evidence in this project has never rested on attribution. It rests on three interventions, removing the delayed input, retiming it, and mistuning it, plus an independent recovery of the delay from the raw signals without the network. This section does not add a fourth line of evidence and should not be presented as one.

What it adds is a mechanism. Until now the strongest statement was that the network needs the delayed input. Now there is a description of what it does with it, that description predicts a measurable frequency response, and the measurement matches.

It also puts a bound on interpretation that applies to earlier work, including my own. If a randomly initialised network produces the same filter shape, then reading structure off a single learned curve proves very little without a random-initialisation comparison alongside it. The formula tables in Update 11 were not qualified that way.

## 5. Corrections to method

Three defects, each of which would have put a false or unsupported statement into a written methods section.

**A loss was named after a function it never used.** The severity head's default mode was called "huber" throughout the code and configuration, while the loss actually computed is a weighted mean squared error. There is no Huber loss anywhere in the project. Renamed to "point", with the old name still accepted so that every existing configuration and result file keeps loading. The rename had a trap: one check compared against the old name in order to decide whether to record extra information about the head, so renaming the default alone would have silently changed what every ordinary run writes.

**A planned tool was never used.** The project plan names PySR for symbolic regression. The implementation fits a fixed library of eight closed forms and selects by an information criterion, which is a different procedure. The report should describe what was done.

**A verdict rule could clear a result on no evidence.** The check that decides whether the delay-lock statistic from Update 12 is real accepted a definitive pass when the number of tested windows was unknown, and treated a hairline margin as a meaningful separation. Both could only ever produce a false pass, and the one real result was decided against the model, so no conclusion changes. Both are fixed.

## 6. What is next

- **Read the reproducibility test.** Two runs of one configuration, submitted, waiting behind a shared limit on how many GPUs the group account may hold at once. Nothing here depends on them.
- **Measure a proper noise floor.** The 0.0092 figure from Update 14 is the largest gap between two runs of one configuration, with no repeated baseline. Until it is a real distribution, no third-decimal comparison in this project is safe to claim.
- **Remove edges and measure the damage.** The filter reading is descriptive. Deleting the pairs that carry it and measuring what held-out accuracy does would make it causal.

## 7. Time spent

| Work | Days | Hours |
|---|---|---|
| Reading the delay coupling as a filter: derivation, module, tests | Aug 10 | 4.0 |
| Running it across five networks and checking every claim against its control | Aug 10 | 2.0 |
| Confirmation probe: design, runner, cluster wrapper, tests | Aug 11 | 3.5 |
| Reading the returned probe results, including the disagreement tail | Aug 11 | 1.5 |
| Method corrections: severity mode rename, verdict rule, tool naming | Aug 11 | 3.5 |
| Reproducibility settings and environment recording | Aug 10 | 3.0 |
| Repairing the file transfer scripts after a silent partial sync | Aug 11 | 1.5 |
| Drafting this update | Aug 11 | 1.0 |
| **Total so far** | | **20.0** |
