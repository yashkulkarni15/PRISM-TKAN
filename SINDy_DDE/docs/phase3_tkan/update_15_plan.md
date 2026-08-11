# Update 15 — work plan

Authored 2026-08-10, at the close of Update 14. Covers the **Aug 9–15** cadence slot.

> **Scheduling note.** The technical report is due **Aug 14** and Phase 9 (Aug 10–15) was
> reserved for writing it. This plan is scoped as research work in parallel with that, at the
> user's direction. Arms A and D produce material the report wants anyway; arm C does not, and
> is the first thing to drop if the report squeezes the window.

## Where this leaves off

Update 14 ends with one measurement and one repair, and neither is a claim about the model:

- the pipeline is **not run-to-run reproducible**, at a scale (0.0092 AUC) that sits **above
  both** 2026-07 retrain effects, so the side-input and quantile results are unclaimable;
- the trainer can now request reproducible arithmetic, refuses to start when it cannot deliver
  it, and records the machine and library versions of every run — but the controlled repeat
  that tests this is a **submission, not a result**;
- the activation-mode analysis shows trained networks beating untrained ones at every setting
  (margins 0.134–0.259) while the pass/fail verdict **flips with k**, so it is exploratory;
- the counterfactual manifest is repaired offline and reproduces the published 0.643.

The delay claim is unchanged and rests where it did: three interventions (ablation 5/5 folds,
per-window τ-sweep peaking at ×1 on all five, τ-jitter degrading in proportion to delay error)
plus a network-free SINDy-DDE recovery at τ*/τ 0.986. Three attribution-side claims are dead
(TSR delay-lock, concept probe, band-notch).

**The single most important consequence: no third-decimal AUC effect can be claimed until the
noise floor is a real number.** That is arm A, and it gates arm B.

---

## Arm A — Determinism, and a noise floor worth quoting

**A1. Read the determinism repeat.** Already handed over: two submissions of the same fold and
seed, 5 epochs, `DETERMINISTIC=1`. Identical histories mean the flags fixed it. Divergence means
the cause is outside the trainer, and the new environment block is what says where to look.
Either outcome is publishable; only the second needs more work.

**A2. Replace the warning with a floor.** Update 14 is explicit that 0.0092 is *not* a formal
noise floor. Three defects in it, all fixable by design rather than cleverness:

| Defect | Why it matters |
|---|---|
| One duplicate per configuration | A max over two samples is not a spread |
| No repeated baseline | The published config itself has never been repeated |
| Seed variation excluded | The floor is therefore a **lower bound**, stated as such |
| Quoted per fold, effects quoted as 5-fold means | The floor and the effects are not the same statistic |

The fourth row is the one that changes the design. Every effect this project reports is a
difference of **five-fold means**, so the floor must be the spread **of that quantity**, not the
largest gap in one fold. That requires repeating whole sweeps, not folds.

*Proposal.* Three repeats of the published configuration at three different seeds, all five
folds: **15 runs**, ~8 h each, arrayed to roughly one day of wall clock and ~120 GPU-h. That
yields the seed spread directly, and combines with the existing duplicate pairs for the
same-seed run-to-run component. Report the floor as the standard deviation of the five-fold
mean, with the same-seed and different-seed components separated so a reader can see which one
dominates.

**This needs the user's approval on GPU budget before submission.** A cheaper variant exists
(3 seeds × 2 folds, ~48 GPU-h) but it estimates the fold-level spread and then extrapolates to
the mean, which is exactly the kind of substitution that produced the 0.0092 confusion. If the
budget is refused, the honest fallback is to report no floor and mark every third-decimal
comparison unclaimable, which is the current state.

*Determinism interacts with this.* If A1 shows the flags give bit-identical runs, then the
same-seed component collapses to zero **for deterministic runs only**, and the floor becomes a
pure seed-spread number. The published history was not deterministic, so its floor does not
change retroactively. Say so explicitly; a floor measured under new settings does not license
re-reading old comparisons.

---

## Arm B — Mode ablation: making Update 14 §4 causal

Update 14 §4 cannot convert its margins into a verdict because the verdict moves with k. A
causal ablation does not depend on the grouping being reproducible, which is precisely why it
survives that problem.

**What already exists.** `xai/activation_modes.py::mode_edge_sets` returns the three sets the
design requires: the mode's edges, a `random_matched` control matched on **count and summed
importance** (best of 200 deterministic draws, with `random_overlap_with_mode` reported rather
than silently accepted), and an `other_modes_matched` control of equal size. It is tested at
`tests/test_activation_modes.py:540`.

**What does not exist.** Nothing consumes it. Verified 2026-08-10: `mode_edge_sets` appears in
its own definition and one test, and nowhere else in `src/`, `scripts/` or `slurm/`. The
selection half of step 8 is built; the **measurement** half was never written.

*Build.* A routine that zeroes a named edge list in a loaded model, and a runner that evaluates
each of the three sets on the held-out 121–155 path (baseline **0.908 ± 0.013**). Warm up at
n = 1 on individual top edges, which is the same machinery and converts the importance
*ranking* into a necessity claim.

*Claimability gate, fixed before any number is seen.* The mode effect must exceed both controls
by more than arm A's floor. Note now, so it cannot be discovered later as a surprise: the
held-out baseline's own **fold spread is 0.013**, already larger than the current 0.0092
estimate, so a small mode effect will not be claimable no matter how clean the ablation is. If
that is where it lands, report it as a bound on what edge-level interpretation can support.

*Cost.* Eval-only on existing checkpoints. Cheap in GPU terms; needs the five `best_auc.pt`
files, which are still **not on local disk**.

---

## Arm C — Learnable τ: design and a single-fold pilot

The strongest science claim available, and the one that cannot finish in this window. Scoped
deliberately as design plus pilot, not as a result.

*Why it is worth starting.* τ is the one physical parameter this project recovers, and it
recovers it three ways, all of which either hold τ fixed at `round(fs/tpf)` or sweep it
externally. A τ that is **learned from data** and converges to the tooth-pass period would be a
fourth recovery and a different kind of claim: physics discovery rather than a classifier that
uses physics.

*The real design question.* τ is currently an integer index into a history buffer, and
`models/heads.py:144` defers per-sample gating to exactly this unlock. An integer index has no
gradient, so the pilot has to choose a differentiable delay: a soft/interpolated read over
neighbouring lags, a learned kernel over a lag window, or a continuous-time formulation. That
choice is the design pass, and it should be made before any code, per the standing note in
PROGRESS.

*Deliverable for U15.* A design doc, an implementation behind a flag defaulting to off, and a
**single-fold pilot** showing whether the learned τ moves toward `round(fs/tpf)` from a
deliberately wrong initialisation. A 5-fold claim is a later window.

*Honest risk.* This arm may produce only the design doc. That is an acceptable outcome and
should be written up as such rather than padded.

---

## Arm D — Methods correctness

Cheap, independent of the queue, and each item would otherwise put a false or weak statement
into the report's methods section.

> **Items 1–3 DONE 2026-08-11** (local, TDD, suite green). Details on the PROGRESS status board.
> Items 4–5 remain.

1. ~~**The `"huber"` mode is not Huber.**~~ **DONE.** Renamed to `"point"`, with `"huber"` kept as
   a deprecated alias normalised at the boundary so every config and `summary.json` on disk still
   loads. The rename had a trap worth recording: the `preprocessing.json` severity block was gated
   on the literal `!= "huber"`, so renaming the default alone would have made every ordinary run
   emit a `severity` block and silently change the default path's schema. Now gated on
   `DEFAULT_SEVERITY_MODE` and pinned by a test. 19 tests, one asserting on behaviour (the point
   path's regression term *equals* `weighted_mse`) rather than on the string.
2. ~~**Never write "PySR".**~~ **DONE — nothing to fix in code.** The word appears in no source
   file and no current doc. It survives in **Research Update 1** as a forward-looking intent that
   was never carried out; left unedited, because it was a true statement of intent when written.
   **Correct wording for the report:** each important edge's activation is sampled over the
   data-visited range and fitted to a *fixed library of eight closed forms* (const, linear,
   quadratic, cubic, saturating-tanh, oscillatory-sin, abs, relu), selected by **BIC** with an
   optional R² threshold and a simplicity tie-break. A library fit, not symbolic regression.
3. ~~**`tsr_control_verdict`.**~~ **DONE — reviewed, and the review found two more defects**, both
   of the family this project keeps hitting (a missing measurement reading as a pass), both able
   to produce only a *false clear*: `min_windows` was skipped entirely when `n_lock_tested` was
   `None` (and the live caller supplies it via `.get()`, which can be `None`), and `tau_specific`
   was a bare `trained > shuffled` that a ten-thousandth of a margin satisfied. Fixed; the real
   fold2_w512 verdict is unchanged.
4. **Aggregate delay-engagement statistics.** The n = 1 `leans_on_delay` case flag was always
   noise and flipped between runs; aggregate it over many windows or drop it.
5. **Threshold calibration**, or cite the conformal ensemble instead. Held-out accuracy swings
   0.72–0.84 purely because the 0.5 cut is uncalibrated, while AUC is unaffected.

---

## Sequencing

1. **A1 first**, because it is already submitted and its outcome changes A2's design.
2. **A2 gates B's claimability**, so B's build can proceed in parallel but B's *verdict* waits.
3. **D fills queue waits.** It needs no GPU and no returned job.
4. **C needs its design pass before any code**, and is the first arm to cut.
5. **No-commit policy holds.** Complete means tests green and review clean, work left in the tree.

## Risks

- **The determinism repeat still diverges.** Then the trainer was not the cause, the environment
  record becomes the diagnostic instrument rather than a formality, and A2's floor has to be
  measured on the pipeline as it is. Still a result.
- **The GPU budget for A2 is refused.** Then no floor exists, and the correct response is to
  mark third-decimal comparisons unclaimable rather than to quote 0.0092 as though it were a
  floor. Write that as the finding.
- **The mode ablation lands under the floor.** Likely, given the held-out fold spread of 0.013.
  This bounds how far any edge-level interpretation from this architecture can be pushed,
  including Update 11's atlas figures and Update 14 §4 itself. That is a real result and should
  be written as one.
- **The report squeezes the window.** Arms A1 and D survive on a compressed schedule; A2 needs
  only a submission and then waits; C is the drop.
