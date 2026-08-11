---
title: "Research Update 14"
date: "August 2 to 8, 2026"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

<!--
Covers August 2 to 8. STARTED EARLY, on July 30, because both of this window's
retrains had already landed and the analysis work is local.

Every number traces to a returned artifact. The two retrains come from
summary.json files under results/phase3/runs and results/phase3/runs_quantile.
The repeatability numbers are recomputed by scripts/repeatability_audit.py from
those same files, and written to results/phase3/repeatability.json. The node each
job ran on came from sacct on 2026-07-30 and is stored in
results/phase3/slurm_nodes.json, matched to runs by End time (unique to the
second, and agreeing with each summary.json mtime). No number is invented.

CURRENT STATUS.
  - Section 4 (activation modes) RAN 2026-07-30 and is a k-dependent negative.
    Source: results/phase3/xai/activation_modes/5fold/
    activation_modes_manifest.json (provenance.complete = true, 32 untrained
    replicates, 256 windows, k = 2..6, matched N = 77 edges/fold, min untrained
    weight diff 0.42). The verdict flips with k in all three strata and each
    stratum flips at a different k, so it is not claimable: the 0.8 threshold
    was pre-committed but k was not. Margins are positive everywhere
    (0.134-0.259). The earlier job 55552794 died on a runner callback arity bug;
    these numbers come from the resubmission after the fix. NOTE: the runner's
    own summary print and the fetch script both read a top-level "verdict" key
    that does not exist (verdicts are per-stratum) and so printed UNDECIDED for
    every k. Both fixed; the manifest was always correct.
  - Section 6 (added 2026-08-10) CLOSES both items that were pending. The
    determinism flags, the cuBLAS guard, the unconditional environment record
    and the --epochs override are in the tree with tests; the controlled repeat
    is a SUBMISSION, not a result, and is labelled as such throughout. The
    counterfactual manifest is repaired offline, with the pre-repair file kept
    beside it as counterfactuals_manifest.pre_repair_2026-08-10.json. The repair
    recomputes 27/42 = 0.6429, matching what Update 13 published: that equality
    is the check that the offline path gives what a re-run would have given.
  - The band-notch item inherited from Update 12 is closed.
  - DATES: Section 8's rows are dated when the work happened, not when the
    cadence window opens. This update started early (Jul 30) and finished late
    (Aug 10); both fall outside Aug 2-8 and both are recorded as they occurred.
    THE HOURS ARE AN ESTIMATE AND NEED CHECKING before this is sent.
-->

## 1. Summary

This cycle asks whether the two retrains reported in Update 12 produced effects large enough to trust.

- **Repeatability:** ten duplicate run pairs used the same configuration and seed. Four pairs reproduced exactly and six did not. The largest observed difference in a selected fold-level AUC is 0.0092.
- **Retrains:** the side-input effect is 0.0028 AUC and the quantile-head classification effect is 0.0062 AUC. With the current repeatability evidence, neither is an established improvement.
- **Activation functions:** trained networks are more self-consistent across folds than untrained networks at every tested setting, but the pass/fail verdict changes with the number of shapes requested. This result remains exploratory.
- **Acted on it:** the trainer can now request reproducible arithmetic, refuses to start a reproducible run it cannot deliver, and records the machine and library versions of every run. The controlled repeat that tests it is submitted, not yet returned.
- **Next:** read that repeat, then measure a proper noise floor before running any further model comparison.

## 2. The two retrains, and both are negative

A fold is one train and test split of the data. Every number below is a five-fold mean of the validation AUC at the best epoch, which is what every earlier update has reported.

The published model scores 0.9554.

**The side inputs show no measurable classification benefit.** The model feeds spindle speed, depth of cut and feed per tooth in alongside the vibration signal. That idea comes from Gao and was meant to stabilise training. Every earlier run had it switched on, so its value had never been isolated. Training all five folds with it switched off gives 0.9526, a difference of 0.0028 AUC from the published model.

This result is consistent with the cheaper test-time shuffling measurement, which reported a drop of 0.002 to 0.009. Both indicate that any classification benefit is small on this pipeline.

Both sweeps in this section exist in two copies, for the reason in Section 1. Every number I quote comes from the second copy, which is the one Update 12 already reported to you. Section 3 is what the choice of copy costs.

**The quantile head does not fix the heavy tail in the median prediction.** Severity is accurate on average and badly wrong on a few cases. The median output from the quantile head leaves that pattern in place.

Classification AUC on those runs came in at 0.9616. That looks like an improvement over 0.9554. Section 3 is why I am not claiming it.

## 3. What repeated training showed

The ten duplicate pairs share an identical configuration and seed. I verified the fields in every pair before comparing their training histories.

**Four of the ten pairs are bit-identical.** Every epoch's training loss and validation AUC agree to full floating-point precision across all twenty epochs. The other six do not.

The result shows that the pipeline sometimes follows the same numerical path and sometimes does not. The duplicate runs measure the size of that variation, but they do not identify its exact cause.

I checked whether node assignment explained the result. Nine of the ten pairs ran on different machines, and three of those nine still reproduced bit-exactly. A run that moved from gl1517 to gl1512 reproduced exactly. Node identity therefore does not predict which pairs reproduce.

The trainer seeds the random number generators at `phase3_trainer.py:158`, but it does not request deterministic algorithms, and every run uses `compile: true`. Those are plausible sources of different numerical paths. They are not yet a confirmed explanation. A controlled repeat with deterministic settings is needed to isolate the cause.

Future `summary.json` files should also record the node, GPU, PyTorch, CUDA, and library versions because the cluster accounting logs are eventually purged.

Among pairs that ran on different machines, the side-input arm reproduced three times out of five and the quantile arm reproduced zero times out of four. With nine pairs this difference is not significant, at p = 0.17, so it remains an observation rather than a result.

![What two runs of the same configuration do. The left panel plots the gap in validation AUC between the two copies of each duplicated fold, epoch by epoch. Four pairs sit flat on zero for all twenty epochs, meaning they agree to full floating-point precision. Six do not, and the worst reaches 0.0261 at epoch 1. The right panel places the two retrain effects beside the largest observed difference in a selected fold-level AUC, 0.0092.](../../results/figures/research_update_assets/u14_repeatability.png)

The size of the divergence:

| | Side-input ablation | Quantile severity |
|---|---|---|
| Pairs that reproduced bit-identically | 3 of 5 | 1 of 5 |
| Largest gap in the selected best AUC | 0.0024 | 0.0092 |
| Largest gap at any single epoch | 0.0261 | 0.0213 |
| Five-fold mean, first run | 0.9531 | 0.9591 |
| Five-fold mean, second run | 0.9526 | 0.9616 |

On the side-input ablation, fold 0 differs by 0.0261 mid-training and by 0.0024 at the selected best epoch. Comparing only the selected AUC therefore hides much of the difference between the training paths.

The **largest observed fold-level difference in selected AUC is 0.0092**. This is a useful warning, but it is not a formal noise floor. The retrain effects are differences between five-fold means, while 0.0092 is the largest difference in one fold.

The duplicate five-fold means differ by 0.0005 for the side-input arm and 0.0025 for the quantile arm. The reported effects against the published model are 0.0028 and 0.0062. With only one duplicate sweep per arm and no repeated baseline, these comparisons are not enough to establish either effect.

For the side inputs, the practical conclusion remains that no measurable classification benefit has been shown.

For the quantile head, the apparent classification gain is not established. The effect is 0.0037 using the first run and 0.0062 using the second. **I am withdrawing the earlier description of it as a real improvement.**

Ten pairs with one repeat each are not enough to estimate repeatability precisely. They do show that third-decimal AUC effects should not be claimed from a single run on the current pipeline.

## 4. Activation patterns are exploratory

The trained networks are more self-consistent across folds than randomly initialised networks at every tested setting. However, the pass/fail verdict changes with the number of shapes requested, and that number was not fixed in advance. I therefore treat the result as exploratory.

The question is whether the network learns a small set of recurring activation shapes, or whether every fold invents its own. The design carries a mandatory control: the identical analysis run on randomly initialised networks. Update 12 is why. A delay statistic that looked significant there turned out to score higher on an untrained network than a trained one, and the only reason I know that is that the control existed. Any claim in this section has to clear a margin over the untrained networks that is fixed before the numbers are seen.

The original plan for this section asked whether the same edge learns the same function across the five folds. It cannot be answered as written, and I checked that against the stored results before building anything. Across the five folds' top-25 edge lists there are 101 distinct edges. Seventeen appear in two or more folds. **None appears in all five.** There is no five-fold sample to compare. Two smaller problems compound it: the stored results keep each edge's fitted formula but not its curve, so the claim that the data was already on disk was wrong; and hidden-unit numbering is arbitrary across independently trained networks, so most of those edges are not the same edge in the first place.

What is comparable is narrower and still large. Only the inputs that map to physical channels carry stable indices across folds. That leaves 1536 curves per fold, 7680 across five, which is plenty to work with as a population rather than edge by edge.

**A randomly initialised network already scores about 0.63 on the main agreement measure, on a scale where 1.0 is perfect.** The reason is structural. Each curve is eight spline coefficients over one shared basis, so even an untrained network looks moderately consistent. The trained score therefore has to be compared with the untrained distribution; it is not meaningful on its own.

The build also exposed a second issue. A group in which every edge has zero importance still produced a score because normalising a zero curve made it look valid. The code now returns UNDECIDED for that case.

The rule was fixed before any number existed. The trained networks must beat the untrained distribution by a set margin, and a missing measurement returns UNDECIDED, which is not a pass. It is the same rule that killed the delay-lock in Update 12.

### Result across the tested settings

It ran on all five networks: 256 windows each, 32 randomly initialised replicates, every replicate verified to have actually re-initialised, and the trained and untrained populations matched to exactly 77 edges per fold so the two arms are the same size. The sweep is complete.

| Group of edges | 2 shapes | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| delayed channels | survives | survives | artifact | artifact | artifact |
| direct channels | survives | artifact | artifact | artifact | artifact |
| process-parameter inputs | survives | survives | survives | artifact | artifact |

Ask for two shapes and everything passes. Ask for six and nothing does. Each group turns over at its own point.

The mechanism is visible in the numbers. On the delayed channels the trained score climbs from 0.804 to 0.911 across the sweep while the untrained score climbs from 0.567 to 0.777. **The null rises twice as fast as the signal.** More prototypes make it easier for any curve to sit near one of them, so the bar the trained network has to clear rises faster than its own score does, and the two cross between three shapes and four.

I set the threshold in advance but did not set the number of shapes. Choosing two or three after seeing the sweep would make the analysis post hoc, so I do not use a single pass/fail verdict.

**The trained networks are more self-consistent across folds than random ones in every group at every setting**, by margins between 0.13 and 0.26, with no exceptions or sign changes. The size needed for a binary pass remains dependent on the number of shapes.

This exploratory result does not change the main delay evidence. That evidence still comes from the three intervention tests and the network-free SINDy-DDE recovery, not from attribution.

## 5. The ablation matrix

The project plan asks for an ablation matrix with four cells. Three cannot be built from the current implementation.

| Cell | State |
|---|---|
| T-KAN against the MLP baseline | Done. Every fold beats it, by at least 0.042 AUC |
| KL term on and off | No implementation. There is no KL term anywhere in the codebase |
| Residual physics on and off | No implementation. The head is a stub that raises an error, and it is blocked on cutting-force coefficients I do not have |
| Hardcoded against learnable delay | Not reachable by August 14. It needs a new model feature and a five-fold retrain at about 8 hours a fold |

These three cells were design intentions rather than implemented features. I will record them as scope changes in the report.

What replaces it is a table built only from work that ran: T-KAN against the MLP, side inputs on and off, the delay-channel ablation across five folds, the graded delay-jitter test, and spline coarsening. Every row of that exists and most of it is already reported. It answers "what does the network need" better than the original four cells would have.

## 6. Acting on the reproducibility finding

Section 3 identified the trainer, not the hardware, as the likely cause of the divergence. This section is what I did about it. The code is in place; the measurement it enables is a submitted job, and I report it as submitted rather than as a result.

**The trainer can now request reproducible arithmetic.** Seeding fixes the inputs but not which kernels consume them, which is why the old runs were mostly reproducible and intermittently not. A new setting turns on deterministic algorithms, turns on the deterministic convolution backend, and turns off the benchmark mode that picks kernels by timing them at runtime.

It is off by default, and that is deliberate. Every result this project has reported was produced with these flags untouched. Turning them on by default would change the arithmetic of every future run against that history, which is a silent change to the baseline rather than a fix.

**A deterministic run now refuses to start if it cannot deliver determinism.** One of the required settings is read by the linear algebra library when it creates its workspace, which happens before training begins, so the trainer cannot set it for itself. Left unset, the run does not fail at startup. It fails inside the first affected matrix multiply, potentially hours in. The trainer now checks for it up front and stops with a message naming the fix. This is the same lesson as the two wrapper bugs in Update 13: a job that starts is not a job that is doing what you asked.

**Every run now records the machine it ran on.** Section 3 could not identify a cause from stored artifacts, because no stored artifact recorded one. Each run's summary file now carries the hostname, the cluster job and node, the GPU model, the versions of PyTorch, CUDA, cuDNN, NumPy and Python, and the state of the determinism settings. Two details matter. The record is written for every run, not only deterministic ones, so an absent record means the run predates this code rather than that it was run carelessly. And the settings are read back from the library rather than copied from the configuration, so the file records what was in effect rather than what was requested. A configuration echo cannot detect a setting that failed to take.

**The test is two short runs of one fold.** Same fold, same seed, determinism on. Both were submitted on August 10 and neither had returned when I wrote this. The first pair queued behind another user on the same account, which has a shared limit on how many GPUs it may hold at once, so they were moved to a different GPU partition and resubmitted as jobs 56977333 and 56977341. Both runs are on the same partition deliberately: two runs on different hardware would not be a test of reproducibility. If the two histories are identical the flags fixed it; if they diverge, the cause is somewhere else and the environment record now says where to look. I kept it short rather than running two full folds because the existing pairs already diverge at the first epoch, so a long run would spend about sixteen GPU hours to observe something visible in the first few minutes. Making it short needed one small addition, an epoch count override that changes only the epoch count. The existing quick-run switch also shrinks the window length, which would change the very kernels under test.

**The counterfactual manifest is repaired.** The stored file predated Update 13's corrected comparison and still carried the metric that was withdrawn. It needed neither the network nor a boundary evaluation: both generators' stability was computed against the real boundary when the job ran and written down, and both sets of coordinates are stored, so the corrected comparison is a recomputation over recorded facts. I moved the comparison into a function that the live runner and the repair both call, so the repaired file cannot drift from the code that produced the published numbers.

The repair reproduces 27 of 42, or 0.643, which is what Update 13 reported. That agreement is the point of the exercise rather than a new finding. The file now states that it was recomputed and not re-run, and the original is kept beside it.

## 7. What is next

- **Read the determinism repeat.** Two submissions, one fold. Identical histories mean the flags fixed it.
- **Measure a real noise floor.** The 0.0092 in Section 3 is the largest gap in one fold from one duplicate per configuration, with no repeated baseline and no seed variation in it. Repeats of the published configuration would turn a warning into a number that comparisons can be tested against.
- **Ablate the activation modes.** Section 4 shows the trained networks are more self-consistent than random ones but cannot convert that into a pass or fail. Removing each group of edges and measuring what held-out accuracy does is a causal test that does not depend on the grouping being reproducible.
- **Report the activation analysis as a sensitivity result.** Keep the full two-to-six-shape sweep and do not choose a value of k after seeing it.

Two earlier items are closed. The band-notch contrast returned on all five folds on July 30; Update 12, Section 4 explains why it yields no usable paired comparison. The duplicate run directories are now canonicalised in `results/phase3/RUNS_CANONICAL.json`, and submission order is recorded as the tie-break when the published AUC cannot distinguish the copies.

## 8. Time spent

| Work | Days | Hours |
|---|---|---|
| Read the two returned retrains from the run artifacts; per-fold numbers rather than summary figures | Jul 30 | 1.0 |
| Repeatability audit: pair matching, config verification, trajectory comparison, written as a re-runnable script | Jul 30 | 2.5 |
| Activation-mode analysis: design, build and test the module, runner and cluster wrapper | Jul 30 | 5.0 |
| Activation-mode run on five networks, k-sweep, and reading the result | Jul 30 | 1.5 |
| Drafting this update | Jul 30 | 1.0 |
| Determinism settings, the startup guard, the environment record and the epoch override, written test first | Aug 10 | 3.5 |
| Deterministic cluster path, and checking the submission is unchanged when the new settings are off | Aug 10 | 1.0 |
| Counterfactual manifest repair: separating the shared comparison, the repair script and its tests | Aug 10 | 2.5 |
| Running the repair and checking it against the published figure | Aug 10 | 0.5 |
| Scoping the next cycle | Aug 10 | 1.0 |
| Revising this update | Aug 10 | 0.5 |
| **Total** | | **20.0** |
