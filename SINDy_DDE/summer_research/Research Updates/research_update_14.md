---
title: "Research Update 14"
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

TWO SECTIONS ARE OPEN.
  - Section 4 (activation modes) is BUILT AND TESTED but NOT RUN on the trained
    networks (no checkpoints on local disk), so it has no results. Module is
    src/pri2m/xai/activation_modes.py, 26 tests. The edge-overlap numbers in it
    (101 distinct edges, 17 in two or more folds, 0 in all five) reproduce from
    the five stored whitebox manifests keyed on (kan, in, out). The 0.63
    untrained-baseline figure is measured on SYNTHETIC fixtures during the build,
    not on the real networks, and the section says so. Nothing else in the
    section is a measurement.
  - Section 6 carries the band-notch item inherited from Update 12.
Section 7's hours cover only work actually done. The window opens August 2.
-->

## 1. What this cycle is for

Two retrains finished five weeks ahead of schedule, so this window has no cluster work in it. That freed the time for something the project had never done: measure how much a result moves when you train the identical model twice.

I did that by accident. Two five-fold sweeps were submitted a second time on July 25 having already finished. It cost about 80 GPU-hours and I reported it in Update 12 as a mistake. It is still a mistake. It also produced ten pairs of runs with identical settings, which is the only repeatability data this project has.

That number decides whether either retrain is a result. Both report effects in the third decimal place of AUC.

## 2. The two retrains, and both are negative

A fold is one train and test split of the data. Every number below is a five-fold mean of the validation AUC at the best epoch, which is what every earlier update has reported.

The published model scores 0.9554.

**Side inputs are worth about nothing.** The model feeds spindle speed, depth of cut and feed per tooth in alongside the vibration signal. That idea comes from Gao and was meant to stabilise training. Every earlier run had it switched on, so its value had never been isolated. Training all five folds with it switched off gives 0.9526. The side inputs are worth 0.0028 AUC.

This closes a question open since June 14. It also confirms a cheaper measurement I had not trusted. The white-box suite estimates the same quantity by shuffling the side inputs at test time and had reported a drop of 0.002 to 0.009. I had assumed that proxy was too weak to settle the question. It was right.

Both sweeps in this section exist in two copies, for the reason in Section 1. Every number I quote comes from the second copy, which is the one Update 12 already reported to you. Section 3 is what the choice of copy costs.

**The quantile severity head does not fix the heavy tail.** Severity is the one output that never hardened. It predicts how bad the chatter is, and it is accurate on average and badly wrong on a few cases. A quantile head predicts three points of the error distribution rather than one number, which is the standard fix for exactly that shape of failure. It did not work. The heavy tail is where it was.

Classification AUC on those runs came in at 0.9616. That looks like an improvement over 0.9554. Section 3 is why I am not claiming it.

## 3. What a result has to beat

The ten duplicate pairs share an identical configuration and an identical seed. I checked that field by field rather than assuming it, because if the pairs differed anywhere the spread would be measuring that difference instead of repeatability.

**Four of the ten pairs are bit-identical.** Every epoch's training loss and validation AUC agree to full floating-point precision across all twenty epochs. The other six do not.

That combination is the useful part. If training were simply noisy, from GPU rounding or the order floating-point sums happen to be added in, then no pair could ever reproduce exactly. Four did. So the six that diverged are not showing me irreducible noise. Something outside the configuration differed between the two submissions.

The obvious suspect was hardware, so I pulled the node each job ran on out of the cluster accounting logs. It is not hardware. Nine of the ten pairs ran on two different machines, and three of those nine still reproduced bit-exactly. A run that moves from gl1517 to gl1512 can come back identical to the last decimal place. Which machine it landed on does not predict whether it reproduced.

What does explain it is in our own trainer. It seeds every random number generator, at `phase3_trainer.py:158`, and then never asks PyTorch for deterministic algorithms. Nothing sets `use_deterministic_algorithms`, nothing disables cuDNN benchmark mode, and every run has `compile: true`. Benchmark mode picks its convolution kernels by timing candidates at runtime, so the kernel chosen depends on what else the GPU was doing at that moment. Seeding fixes the numbers going in. It does not fix which arithmetic gets used on them. That produces exactly what I see: mostly reproducible, intermittently not, and unrelated to the node.

So the defect is ours and the fix is small. It also means `summary.json` should record the node and library versions, since without the cluster accounting log I could not have tested this at all, and those logs are purged.

One pattern I noticed and am not claiming. Of the pairs that ran on different machines, the side-input arm reproduced three times out of five and the quantile arm never reproduced, zero out of four. That would fit the quantile head pulling in a nondeterministic kernel the point head avoids. With nine pairs it is not significant, at p = 0.17, so it is a hypothesis for one cheap re-run, not a result.

![What two runs of the same configuration do. The left panel plots the gap in validation AUC between the two copies of each duplicated fold, epoch by epoch. Four pairs sit flat on zero for all twenty epochs, meaning they agree to full floating-point precision. Six do not, and the worst reaches 0.0261 at epoch 1. The right panel puts the two retrains against the floor, which is the largest gap between two runs of the same settings. Both effects are inside it.](../../results/figures/research_update_assets/u14_repeatability.png)

The size of the divergence:

| | Side-input ablation | Quantile severity |
|---|---|---|
| Pairs that reproduced bit-identically | 3 of 5 | 1 of 5 |
| Largest gap in the selected best AUC | 0.0024 | 0.0092 |
| Largest gap at any single epoch | 0.0261 | 0.0213 |
| Five-fold mean, first run | 0.9531 | 0.9591 |
| Five-fold mean, second run | 0.9526 | 0.9616 |

Read the third row against the second. On the side-input ablation, fold 0 wandered 0.0261 apart mid-training and landed 0.0024 apart at the epoch that got selected. The headline number understates the instability by a factor of ten. Two runs agreeing on their best AUC is not evidence they trained the same way.

So the floor is **0.0092 AUC**, the largest gap between two runs of the same configuration. It is a lower bound, because these pairs share a seed. Changing the seed would move things further.

Now put the two retrains against it. The side-input effect is 0.0028. The quantile effect is 0.0062. Both are below the floor.

For the side inputs that changes nothing, and if anything it strengthens the conclusion. The claim was already that they are worth about nothing, and an effect too small to measure is what "worth about nothing" looks like.

For the quantile head it changes the answer. Update 12 called its AUC "a real improvement but a val-selected one". It is not a real improvement. It is smaller than the gap between two runs of its own configuration. Worse, the effect depends on which duplicate I keep: 0.0037 using the first run, 0.0062 using the second. The choice of copy moves the headline by more than half its own size. **I am withdrawing that sentence from Update 12.**

One number in this section deserves a caveat I cannot resolve here. Ten pairs is a small sample, one repeat each, and a repeatability floor built from it is rough. It is still the only such measurement the project has, and it points one way: effects in the third decimal of AUC are not claimable from single runs on this pipeline.

## 4. Do the learned functions repeat across folds?

**Built and tested. Not yet run on the trained networks, so there are no results.** The five checkpoints are not on local disk. What follows is the design, the rule I committed to before any number exists, and one thing the build itself taught me about the metric.

*[PLACEHOLDER: fill once the analysis runs on the five checkpoints. Sweep k and report the sweep. Quote the trained-versus-untrained margin, never the trained score alone.]*

The question is whether the network learns a small set of recurring activation shapes, or whether every fold invents its own. The design carries a mandatory control: the identical analysis run on randomly initialised networks. Update 12 is why. A delay statistic that looked significant there turned out to score higher on an untrained network than a trained one, and the only reason I know that is that the control existed. Any claim in this section has to clear a margin over the untrained networks that is fixed before the numbers are seen.

The original plan for this section asked whether the same edge learns the same function across the five folds. It cannot be answered as written, and I checked that against the stored results before building anything. Across the five folds' top-25 edge lists there are 101 distinct edges. Seventeen appear in two or more folds. **None appears in all five.** There is no five-fold sample to compare. Two smaller problems compound it: the stored results keep each edge's fitted formula but not its curve, so the claim that the data was already on disk was wrong; and hidden-unit numbering is arbitrary across independently trained networks, so most of those edges are not the same edge in the first place.

What is comparable is narrower and still large. Only the inputs that map to physical channels carry stable indices across folds. That leaves 1536 curves per fold, 7680 across five, which is plenty to work with as a population rather than edge by edge.

Building it taught me something I would otherwise have learned too late. **On the small synthetic networks the tests run against, a randomly initialised network already scores about 0.63 on the main agreement measure, on a scale where 1.0 is perfect.** Not near zero, as I had assumed. That 0.63 is a property of the test fixtures rather than a measurement on our trained networks, and the real figure will come from the run. The reason is structural. Each curve is eight spline coefficients over one shared basis, so no curve can be far from every prototype, and a network that has learned nothing still looks moderately consistent. If I had run this and read 0.7 as a strong result, I would have reported an artifact. The trained score on its own means nothing here. Only the gap over the untrained networks does, which is why the untrained arm is not an optional check in this design but the denominator of the whole measurement.

Two more things the build turned up, both now guarded in code. A stratum that is dead in the data, where every edge has zero importance, still produced a clean-looking score, because normalising a dead curve makes it indistinguishable from a live one. That now returns UNDECIDED instead of a number. And the number of modes is not a free parameter: set it above the number of shapes actually present and the agreement measure collapses toward chance even while everything else looks healthy. So the run has to sweep it and report the sweep.

The rule, fixed now: the trained networks must beat the untrained distribution by a margin set before the numbers are seen, and a missing measurement returns UNDECIDED, which is not a pass. It is the same rule that killed the delay-lock in Update 12, and I expect it may kill this too.

## 5. The ablation matrix, and what it can honestly contain

The project plan asks for an ablation matrix with four cells. Three of them cannot be built, and I would rather say so than leave the criterion sitting at one of four.

| Cell | State |
|---|---|
| T-KAN against the MLP baseline | Done. Every fold beats it, by at least 0.042 AUC |
| KL term on and off | No implementation. There is no KL term anywhere in the codebase |
| Residual physics on and off | No implementation. The head is a stub that raises an error, and it is blocked on cutting-force coefficients we do not have |
| Hardcoded against learnable delay | Not reachable by August 14. It needs a new model feature and a five-fold retrain at about 8 hours a fold |

You cannot ablate something that was never built. Three of these were design intentions rather than code, and the honest move is to record that as a scope change in the report, the same way I am recording the inverse problem.

What replaces it is a table built only from work that ran: T-KAN against the MLP, side inputs on and off, the delay-channel ablation across five folds, the graded delay-jitter test, and spline coarsening. Every row of that exists and most of it is already reported. It answers "what does the network need" better than the original four cells would have.

## 6. Still open

Determinism flags in the trainer, from Section 3. Setting them costs a little speed and buys exact reproducibility, and I would rather have the reproducibility on a project whose results live in the third decimal place. This should land before any further retrain.

The band-notch spectral contrast from Update 12 is fixed in code, and its cluster re-run was submitted on July 30 as jobs 55534785 to 55534789. It compares deleting the chatter frequency band against deleting the tooth-passing band, at matched removed energy. It had been aiming the bands using a batch-average frequency rather than each window's own, and reporting a failed solve as though it were a real measurement. Both are fixed, with six tests that fail against the old code. A third defect turned up while fixing them: the check for missing frequency values was applied to the batch average, so one bad row discarded all 512 windows.

The duplicate run directories still need one of each pair marked as canonical. Section 3 is the evidence for that choice. I am keeping the later run of each pair, because the numbers already sent to you come from those.

## 7. Time spent

Work done so far. The window opens August 2.

| Work | Days | Hours |
|---|---|---|
| Read the two returned retrains from the run artifacts; per-fold numbers rather than summary figures | Jul 30 | 1.0 |
| Repeatability audit: pair matching, config verification, trajectory comparison, written as a re-runnable script | Jul 30 | 2.5 |
| Activation-mode analysis: design against the stored results, establishing that the original item cannot run, then building and testing the module | Jul 30 | 5.0 |
| Drafting this update | Jul 30 | 1.0 |
| **Hours so far** | | **9.5** |
