---
title: "Research Update 12"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

<!--
Covers July 19 to 25. Revised 2026-07-29, closed 2026-07-30.

Every scientific number traces to a returned GL manifest or a summary.json
fetched from Great Lakes. No number is invented.

NO SECTIONS OPEN. Section 4 closed 2026-07-30: the construction-artifact
control returned (GL job 55408255, third submission) and the untrained network
locks HARDER than the trained one, so the TSR delay-lock is an artifact of the
input construction. Branch text replaced with the result. Source manifest:
results/phase3/xai/tsr_control/fold2_w512/tsr_control_manifest.json
(provenance.complete = true, three conditions x 24 windows, paired indices).
The paired statistics quoted in Section 4 (22/24, p = 3.6e-5; 19/24, p = 0.0066;
the by-label and by-tau breakdowns) were computed from that manifest's
per_window rows, not read off the printed summary, whose VERDICT line uses the
uncorrected rule described in the same section.

Language simplified 2026-07-30 for readability. No number, caveat, negative
result or correction of the record was changed; verified by comparing the
multiset of every numeric token before and after (219 each, zero differences).

UPDATED 2026-07-30, Section 5: both band-notch defects were fixed and the
contrast re-run on GL (jobs 55534785-89, ALL FIVE folds returned and read). It
still yields no usable number, now for a diagnosed structural reason rather than
a hidden sentinel: 298-323 of 512 windows per fold are unsolvable (58-63%) and
the 90th percentile of the relative energy error is 0.9751-0.9783 against a 1%
requirement. A third same-family sentinel was found and guarded (degenerate
denominator: a near-empty chatter band gave max_rel_err ~2e4). NOTE: the returned
manifests PREDATE that guard, so their max_rel_err is unreadable and rel_err_p90
is the figure quoted. A re-run would only make the headline legible; it would not
change the conclusion.

CORRECTION 2026-07-30, Section 6: the quantile head's classification AUC was
described as "a real improvement" pending a held-out evaluation. Withdrawn. The
repeatability audit (Update 14 Section 3, scripts/repeatability_audit.py) puts
the same-configuration run-to-run drift at up to 0.0092 AUC against a 0.0062
gain, so the gain is not claimable at all. The same paragraph's description of
the duplicate sweeps as "two independent five-fold runs" was also wrong: same
config, same seed, and 4 of 5 quantile folds did not reproduce.

CHANGED 2026-07-29: the earlier draft claimed the resolved peak lands on an
exact harmonic of tau in 60 of 90 windows (67%). That number could not be
reproduced from the merged manifests under any natural definition of "harmonic"
(integer divisibility gives 44 of 90; a 5% ratio tolerance gives 72). It has
been replaced with the divisibility count and its definition stated inline.
-->

## 1. State before this work

Update 11 closed the white-box queue and reported two honest negatives. One was Temporal Saliency Rescaling, or TSR. TSR scores every time step of the input by how much the network leaned on it. I had finally wired it up. It ran clean, and its top-scoring time step landed nowhere near the tooth-pass delay, the lag between successive cutting teeth that drives chatter. So I reported a tool that ran but confirmed nothing. I gave two reasons for that negative. It was underpowered, at four windows per fold, where a fold is one train/test split of the data. And neither of its readouts pointed at the delay.

The first reason was wrong. The second was measured on a broken statistic. Running TSR at scale did not settle the question the way I expected. It showed that I had posed the question badly.

## 2. The metric was reporting its own search floor

I ran TSR at full scale, 160 shards across five folds on Great Lakes, July 23 to 24, all returned and merged. I expected one of two answers. Either the negative would survive at scale and I would retire the attribution view, or four windows had been the whole problem and a real signal would appear. Neither happened. The metric itself was broken.

`tr_peak_lag` looks for the lag at which the relevance signal most repeats itself, which is the peak of its autocorrelation, searched over lags 2 to N/2. When that autocorrelation only decays and never peaks, the search returns index 0 and the metric prints 2, the bottom of its own search range. That is a "no answer" rather than a measurement.

550 of 640 windows, 85.9 percent, come back as exactly 2. The per-fold means of 2.6 to 10.0 that Update 11 reported are averages over a column that is mostly this no-answer value. An average over sentinels measures nothing.

![The TSR delay metric. Panel A shows how many of each fold's 128 windows returned the search floor instead of a peak. Fold 3 resolved 32 windows and fold 0 resolved 6, so a per-fold mean of this statistic compares how much evidence each fold produced rather than how much delay it found. Panel B plots, for the 90 windows that did resolve, each window's peak lag against that window's own delay, both axes logarithmic. Points on the dashed line are windows whose peak sits exactly on tau. There are 31 of those, drawn filled, and the rest scatter below the line. That scatter is why the corrected statistic tests at tau instead of searching for a peak and asking afterwards where it landed.](../../results/figures/research_update_assets/u12_tsr_floor_and_lock.png)

The 90 windows that do resolve a real peak tell the opposite story. In 31 of those 90, 34 percent, the peak lands exactly on that window's own tau. Chance alone would put 3.34 windows there, p = 0.00005. Counting a peak as harmonic when either the lag divides tau or tau divides the lag, 44 of 90 qualify. The median ratio of `peak_lag` to `tau_raw` is 1.0000. I also had the units wrong. `peak_lag` and `tau_raw` are already in the same units, so Update 11's comparison against "about 22 steps after four-fold pooling" was on the wrong axis as well.

This corrects Update 11, Section 3. The published text gives "underpowered, only four windows" as the reason for the negative. That diagnosis is wrong. The negative was unsound rather than underpowered, because the statistic under it was mostly reporting a floor. I am saying so here instead of quietly superseding the old text, because a reader who saw it deserves the correction.

I cannot yet replace the negative with a positive. Section 4 is where I test that.

## 3. A statistic that cannot report its own floor

I rebuilt the metric. The repair has three parts, written test-first like the rest of this build, meaning the tests came before the code.

Any summary of the lag statistic now carries the count of windows that resolved a peak alongside the value, so a number derived from 90 of 640 windows is reported as such. The new delay-lock statistic asks how strongly the relevance signal repeats itself at exactly lag tau, and tests that against a null. The old one took a global peak and asked afterwards whether it happened to land near tau. The new one has no floor to fall back on. And the floor value is now excluded from every aggregate rather than pooled in silently.

The defect is worth naming as a class, because this metric is not the only place it can happen. A search returns a bound when it finds nothing, and that bound feeds an aggregate that cannot tell the bound from a measurement. I made it a standing check in the review brief for this branch. It caught a second case the same week, in Section 5.

## 4. Is the delay lock learned, or built in?

A network with random, untrained weights locks onto the delay harder than the trained network does. The lock is built into how the input is made, and the network did not learn it. That is what the control returned.

The control matters because the 34 percent exact-match result only says something about the model if the input construction cannot produce it on its own. The delayed channel is the signal shifted by tau, so the relevance vector may carry lag-tau structure by construction, with nothing learned. That would give the same statistic on a network that had never been trained.

The control runs the fixed metric on identical windows under three conditions. The trained network at true tau. The same architecture with re-initialised weights at true tau, which is the decisive one. And the trained network at deliberately shuffled tau. I build the untrained network by deep-copying the loaded model and re-initialising every submodule. Rebuilding it from guessed constructor arguments could drift from the trained architecture and quietly invalidate the comparison. Two guards refuse to run at all, one if no submodule exposes a re-initialiser, one if the "untrained" weights turn out to be indistinguishable from the trained ones. Both guards have now passed on the cluster, at a mean absolute weight difference of 0.426.

I fixed the verdict rule in advance, before any result existed. If the untrained lock reaches 80 percent of the trained lock, the effect is a construction artifact. A lock that is unmeasurable on either side returns UNDECIDED, which is explicitly not a pass. Both outcomes are publishable, which is why the rule could be pre-committed. If the lock collapsed on the untrained network, TSR would become a fourth independent line of delay evidence alongside the ablation, the tau-sweep and the tau-jitter test. If it survived, the attribution view of the delay is genuinely retired and this update is a methods correction, which stands on its own.

It survived, and by a wider margin than the rule was written to catch.

The rule as implemented does not yet match the rule as written, and I found this before reading any result rather than after. The ratio test is there. Three things it depends on are not. Nothing checks that the trained lock is a lock at all, because the significance threshold that defines one gets computed and then never consulted. A trained value indistinguishable from noise, or even pointing the wrong way, still passes the ratio test and prints the fourth-line-of-evidence claim. Nothing requires a minimum number of testable windows, so a single window can carry a definitive verdict. And the third condition, the trained network at shuffled tau, is computed at full cost and then ignored. That is precisely the condition that separates a genuine lock from a lock that only reflects how smooth the attribution curve is.

That last point matters more than the other two, because the lock statistic is measurably biased toward short delays on smooth curves. A synthetic curve with no delay structure of any kind registers as locked at every delay up to 16 steps. The unnormalised autocorrelation decays with lag while the null is drawn mostly from long lags, and that alone produces the reading. The delays in this dataset sit inside that biased range. If training makes the attribution map smoother, which is plausible, the trained network would out-lock the untrained one for a reason that has nothing to do with the delay.

The control took three submissions to return a number. The first burned its full 12-hour wall having written nothing, because my cost estimate counted 24 windows when the three conditions share those windows and the real work is 72 window-evaluations. The second died in seconds on a latent crash in the manifest step. The first run's timeout had been hiding that crash, so that run could never have written its output even if it had finished. The runner then gained a per-window counter and a partial manifest after every condition, with the decisive pair scored first, so a wall-clock kill would leave a readable verdict instead of nothing. The third submission returned all three conditions on July 30.

I checked that it had actually finished before reading any number from it, because two wrappers in the same week exited cleanly while skipping the work they were supposed to do. The manifest carries a completion flag that is written only after all three conditions return. The per-condition checkpoints write it as false, so a run killed at the wall cannot present itself as a finished one. That flag is true. All three conditions are present at 24 of 24 windows, and the delay statistic was testable on every one of them. The window indices are identical across conditions, so every comparison below is paired on the same windows. The untrained arm is a real one, with seven submodules re-initialised and a mean absolute weight difference of 0.426.

| Condition | Mean lock statistic | Windows locked (z > 2) |
|---|---|---|
| Trained network, true tau | +2.81 | 79.2 percent |
| Untrained network, true tau | **+4.62** | **100 percent** |
| Trained network, shuffled tau | +0.88 | 16.7 percent |

The decisive comparison inverts. On the same windows at the same delay, the randomly initialised network locks harder than the trained one. It does so in 22 of 24 paired windows, at a mean margin of 1.80. A sign test, which counts how often one arm beats the other, gives p = 3.6e-5. The untrained network clears the significance threshold in every single window, while the trained network clears it in 19. A network that has learned nothing produces a stronger lag-tau signature than the network that was trained to classify chatter. Under the pre-committed rule the untrained lock does more than reach 80 percent of the trained lock. It exceeds it outright.

The third condition does not rescue the result. The trained network at true tau does beat itself at shuffled tau, 19 of 24 windows, p = 0.0066, so the effect is specific to each window's own delay rather than to delays in general. That sounds like evidence for learning and is not, because the input construction is equally tau-specific. The delayed channel is the signal shifted by that window's own tau, so a construction artifact is required to be tau-specific too. The shuffle separates "this window's delay" from "any lag". It does not separate learned from built in. Only the untrained arm does that, and the untrained arm answers "built in".

Two further details point the same way. The trained lock is stronger on stable windows, 3.52, than on chatter windows, 2.10, which is backwards for anything describing a regenerative mechanism. And the lock statistic falls steadily as tau grows, correlation −0.64 on the trained network and −0.43 on the untrained one. That is the signature of the smoothness bias described above rather than of physics.

There is a methods lesson here that generalises past this project, and I only saw it by putting the two manifests side by side. Since Update 8 the white-box suite has run a model-randomization sanity check on every fold. It is the standard test of whether a saliency map, the picture of which time steps mattered, reflects the trained weights at all. Fold 2 passes it clearly. The map computed on the trained network has cosine similarity 0.0995 with the map computed after randomizing the weights, meaning randomization almost entirely destroys the map, which is the desired outcome. The delay statistic in this section is derived from that same attribution on that same checkpoint, and it goes the other way, scoring higher after randomization. Both numbers are correct. A saliency map can be highly sensitive to the trained weights while a single number summarising that map is not. The summary reads a property the map inherits from the input rather than from the model. I was applying the sanity check one level too high. Any statistic distilled out of an attribution map needs its own randomization control, and a clean result on the map does not transfer to it.

I had expected correcting the verdict rule to be the delicate step. It changes nothing here except to make the answer firmer. The significance gate cannot rescue the trained side, because the untrained side passes it in 24 of 24 windows. The minimum-window requirement is satisfied, at 24 testable windows per condition. The ignored third condition, once read, turns out to test a distinction that does not bear on the verdict. Applying the rule as written gives the same answer as the rule as implemented, with more behind it.

One defect worth recording for anyone repeating this. The tau shuffle is not a derangement, meaning it does not guarantee that every window gets some other window's tau, and 2 of the 24 windows were handed back their own. This biases the shuffled condition upward, which is the safe direction, and excluding those two windows lowers its mean from 0.88 to 0.70. It does not touch the decisive comparison, which never uses the shuffled arm.

TSR is therefore a closed negative, and no TSR claim enters the technical report. The 34 percent exact-match result from Section 2 is a real measurement of the input construction. It is not evidence about the network. This is the outcome the control was built to be able to return, and it cost the delay claim nothing, because that claim has never rested on attribution. It rests on the three interventions and on a network-free recovery of the same delay.

## 5. The physics-floor stressors build

The stressor suite got its spectral half this cycle, and the new band-notch contrast produced nothing usable. It still produces nothing usable after both its defects were fixed, and the end of this section says why. The build adds surrogate signals that preserve some spectral properties while destroying others, reliance metrics over them, and the band-notch contrast itself, which asks a sharper question than the existing fuzzers can.

The contrast is this. If the model's delay reliance is genuinely about the regenerative chatter mechanism, then deleting the chatter band should cost it more than deleting an equal amount of energy from the tooth-pass harmonics. Matching the energy removed is what makes that a fair comparison, and matching it is harder than it sounds. The achievable set of removed-energy values is a staircase, because a notch either includes a spectral bin or it does not. The implementation solves both halfwidths per window rather than fixing one and hoping. It also refuses candidate bands that would widen until they swallow the tooth harmonics, which would "match" the energy by deleting both conditions' evidence at once.

On the returned data the contrast reports `matched = False` on all five folds, with the maximum relative energy error pinned at exactly 1.0. That number sent me to the review, and the review found two independent defects that between them void the run.

The bands were aimed at the wrong frequencies. The runner passes a single batch-mean sampling rate, chatter frequency and tooth-pass frequency to the whole batch. That batch deliberately spans many operating points, 37 distinct sampling rates from 12.3 to 38.3 kHz and eight chatter frequencies. Every window is then notched at the same nominal location. For a window recorded at 12.5 kHz the frequency axis is mislabelled by a factor of about 2.4. Deleting a band nominally at 950 Hz actually deletes content near 400 Hz, while that window's true chatter frequency is 500 Hz and its true tooth-pass fundamental is near 1 kHz. Both arms remove a few percent of the spectrum from physically arbitrary places. This is why the two arms came back indistinguishable, and it holds regardless of whether the energies matched.

Some windows were never notched at all. The halfwidth solver, on exhausting its iteration budget, returns the closest candidate it has seen. That candidate can be narrower than one FFT bin and remove exactly zero energy. That window's tooth arm is then the untouched signal, scored as though it were the tooth condition. Removing zero energy makes the relative error exactly 1.0 by arithmetic, and because the reported figure is a maximum over windows, one such window pins the whole fold. This is the same defect class as Section 2, a "no answer" flowing into a summary as if it were data, and nothing in the output records how many windows are affected.

So the spectral contrast produced no usable number this cycle. I am not writing it up as a null result. A comparison that deleted the wrong frequencies in both arms is not evidence that the model ignores the chatter band. It is not evidence about the model at all. The batch-mean frequency defect needs a per-window fix, and the solver needs to report its unsolved windows instead of notching nothing. Both are scoped for the next cycle.

**Both defects were fixed on July 30 and the contrast re-run. It still produces no usable number, and now I know why.** The fix worked in the sense that mattered: the 1.0 sentinel is gone and the failure is legible. What it made legible is that the experiment cannot be run as designed.

Between 298 and 323 of each fold's 512 windows cannot be solved at all, so 58 to 63 percent of the sample is unusable, and that figure is consistent across all five folds. Among the windows that do solve, ninety percent miss the energy match by between 97.5 and 97.8 percent, against a requirement of one percent. The cause is a structural asymmetry I confirmed by reproducing it on a synthetic window built to match the observed removed fractions. The chatter notch is one band and the smallest amount of energy it can remove is about 2 percent of the signal. The tooth notch is three bands sitting on the strongest peaks in the spectrum, and the smallest amount it can remove is about 13 percent. The two achievable sets never meet anywhere near where the real removed fractions live.

The bind is worse than a tuning problem. To push the chatter fraction up to where the tooth side can match it, the chatter band has to grow wide. A wide chatter band reaches the tooth harmonics, and that is exactly what the safety guard forbids, because a band that swallows both deletes both conditions' evidence. The guard that makes the contrast meaningful is what makes it unmatchable. I tested the two obvious escapes, dropping from three tooth harmonics to one and lengthening the window from 1024 to 4096 samples, and in every combination the tooth floor still sits two to four times above the highest chatter fraction the guard allows.

One direction in the returned numbers looks meaningful and is not. On four of the five folds the chatter notch cost more accuracy than the tooth notch, by 0.0042 on average. That is the direction a spectral chatter-energy detector would show, which would cut against the delay story.

It cannot be read, and the reason is worth stating precisely rather than waving at. Averaged over the batch the two arms look nearly matched: the chatter notch removed between 3.9 and 7.4 percent more energy than the tooth notch. Per window they are not matched at all, missing by around 97 percent. The batch mean is hiding the per-window failure, which is the same pattern as every other defect in this update. A paired contrast whose pairing is broken at the window level cannot support a conclusion no matter how tidy its averages look.

I also checked whether the leftover energy difference could account for the accuracy gap, and it does not. Across folds the correlation between the two runs backwards, at minus 0.74: the fold with the largest energy excess shows the smallest accuracy gap. With five folds that is not significant either way. So I can neither explain the direction as an energy artifact nor claim it as a result, which is the honest position and the one the matching requirement exists to avoid.

What the run does support is narrower and worth keeping. Deleting roughly six percent of the spectrum at either location costs at most 0.006 AUC against a baseline near 0.99. The model is not fragile to either band.

The fix also exposed a third instance of the defect class from Section 2, one level deeper. The relative error divides by the chatter fraction, so a window whose band landed on near-empty spectrum divides by almost nothing and reports an error of twenty thousand. A strictly-positive test admits it, because one near-empty frequency bin is still positive. Those windows are now excluded by an explicit floor, chosen from the measured gap between the degenerate windows at three parts in a million and the smallest legitimate one at eight parts in ten thousand, and counted separately from the windows that simply failed to solve. Three sentinels of the same family in one month is a pattern rather than a coincidence, and the standing check for it has earned its place in the review brief.

## 6. Two retrains that landed five weeks early

Both retrains scoped for Update 14 were unblocked on July 20. I submitted them immediately rather than holding them to their window, on the reasoning that GPU wall-clock is the one resource that cannot be compressed later. Both returned.

The side-input ablation settles a question open since June 14. Every T-KAN fold to date trained with the Gao-style process-parameter side inputs on, so their contribution had never been isolated. The white-box permutation proxy was too weak to settle it, shrinking to 0.002 to 0.009 at 512 windows. Training once with them on and once with them off is the real test:

| Fold | side inputs on | side inputs off | delta |
|---|---|---|---|
| 0 | 0.9597 | 0.9626 | −0.0029 |
| 1 | 0.9122 | 0.9080 | +0.0042 |
| 2 | 0.9752 | 0.9747 | +0.0005 |
| 3 | 0.9605 | 0.9546 | +0.0059 |
| 4 | 0.9695 | 0.9632 | +0.0063 |
| **mean** | **0.9554** | **0.9526** | **+0.0028** |

The side inputs are worth about three thousandths of AUC. That sits well inside the fold-to-fold spread of 0.022, and on fold 0 the network was very slightly better without them. Whatever the side inputs contribute to this model, it is not classification accuracy.

The quantile severity head does not fix severity. Severity is the one output never hardened. In distribution it gives MAE 0.88 with RMSE 4.37. It does not transfer out of distribution at all, at held-out MAE 3.3 to 4.8, where held-out means datasets the model never saw in training. My hypothesis was that the heavy tail came from fitting a conditional mean, and that a pinball-loss quantile head, which fits chosen percentiles instead of the mean, would be better behaved. I measured on the median column, so the comparison is like-for-like:

| Severity head | MAE | RMSE | RMSE/MAE |
|---|---|---|---|
| point head (published) | 0.88 | 4.37 | 5.0 |
| quantile, first run | 0.880 | 5.01 | 5.7 |
| quantile, second run | 0.830 | 4.48 | 5.4 |

MAE moves by five hundredths, RMSE gets marginally worse, and the ratio that characterises the heavy tail is unchanged. The quantile head did not buy what it was wired in to buy. I report it as a negative rather than as a landed feature. The quantile outputs still feed the conformal layer usefully. But the severity problem is not a loss-function problem, and the next attempt should not be another loss function.

Classification AUC on the quantile runs came in at 0.962 mean against the published 0.955, every fold equal or better.

**Correction, added July 30.** The sentence that stood here called that a real improvement, selected on the validation set, and said I would not claim it until the set had its own held-out evaluation. That was too generous, and I am withdrawing it. On July 30 I used the duplicated sweeps to measure how far two runs of the identical configuration drift apart, and the answer is up to 0.0092 AUC. The quantile gain is 0.0062. It is smaller than the gap between two runs of its own settings, so it is not an improvement I can claim, held-out evaluation or not. The same paragraph also described the two runs as replicating the severity result across "two independent five-fold runs". They are not independent. They are the same configuration and the same seed submitted twice, and four of the five quantile folds did not reproduce. Update 14, Section 3 has the measurement.

## 7. Obstacles

About eighty GPU-hours went into re-deriving results I already had. I submitted both five-fold arrays a second time on the night of July 25, and both had already completed, the ablation on July 21 and the quantile head on July 20. Three of the five ablation folds returned AUCs identical to four decimal places. One quantile fold was verified bit-identical to its earlier run, with the same configuration, same best epoch and same training history. That is how the duplication surfaced at all. Nothing was lost but queue time and allocation. The cause was a results directory with no marker distinguishing a completed sweep from an empty one, and the fix is a check on outputs before submission rather than on the job script.

The results tree now holds two complete copies of both sweeps, ten run directories where five belong, with two entries per fold and nothing in the path saying which is canonical. I have to resolve this before anything globs that directory, and before the model zoo is republished.

## 8. Where this leaves the delay evidence

The delay claim still rests on the three interventions it has rested on since Update 8: the ablation, the per-window tau-sweep and the tau-jitter test. On top of those sits the network-free SINDy-DDE recovery, at a median tau ratio of 0.986. TSR is not a fourth line and will not become one. Its control returned while I was finalising this update and answered against it, so the attribution view of the delay is now retired on evidence that decides the question. Update 11 retired it on a reason that turned out to be a misdiagnosis, which is a different thing. The three interventions are untouched by any of this, because none of them is an attribution method. They intervene on the network and measure what breaks.

What this cycle establishes about method is worth separating from what it establishes about the delay. I found a metric that could report its own search floor and rebuilt it. A statistic with no floor replaced it. The effect that new statistic revealed then went to a control with a pre-committed rule, and the control killed it. The claim never reached the report. That sequence is the intended one, and it ran end to end in a week.

Two negatives arrived early and both are worth having. The process-parameter side inputs are worth 0.0028 AUC, so they are not the source of T-KAN's margin over the MLP. The quantile severity head leaves the heavy tail where it was across two independent five-fold runs, so severity is not a loss-function problem.

Next: fix the two band-notch defects and re-run the spectral contrast; canonicalise the duplicated run directories before the zoo is republished.

## 9. Time spent

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
