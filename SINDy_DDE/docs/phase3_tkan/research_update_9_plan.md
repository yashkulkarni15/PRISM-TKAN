# Research Update 9 — Planning Document

**Status:** plan only (no code changes). Sequences the ~20 h of work for Update 9.
**Branch:** `phase3-tkan`. Commit policy unchanged: never `git add`/`commit`/`push`; leave work in the tree.
**Covers (target window):** ~Jul 2 – Jul 11, 2026 (about one week, ~20 h).
**Authoritative source of tasks:** `docs/phase3_tkan/PROGRESS.md` → "NEXT ACTION" frontier + "NEW METHODS QUEUE (2026-07-01)". This doc sequences that content; it does not restate it.

---

## 1. Goal / narrative arc

Update 8 answered *whether* the trained TKAN uses the regenerative tooth-pass delay: zeroing the τ-delayed input collapses chatter detection to near-chance on three of five folds and blows up severity error by one to two orders of magnitude, replicated across all five checkpoints. That is an intervention result — the model *depends on* the delay.

Update 9 turns "the model uses the delay" into "**what** delay did it learn, and **what physics** does it encode." I will (a) locate the delay the network actually keys on, by running the per-window-relative τ-sweep that fixes Update 8's blurred global sweep; (b) recover the delay a second, fully independent way — straight from the raw cutting signals with SINDy-DDE, so the ablation and the signal-level recovery become *two independent recoveries of the same physical delay*, a physics-discovery framing rather than a classifier-audit one; and (c) state the model's stability knowledge in machining terms by drawing its implied stability lobe diagram against the analytic `blim` curve, with a quantitative global-faithfulness number. All of the headline work is **eval-only on the existing five `best_auc.pt` checkpoints — no retraining.**

---

## 2. Scope decision

### IN Update 9 (the cheap, eval-only bundle)
1. **Per-window-relative τ-sweep** — closes Update 8's single biggest open caveat (the global sweep peaked at 2τ on all five folds because a single imposed delay cannot fit a multi-operating-point sample).
2. **SINDy-DDE cross-check** (arXiv:2407.19640) — independent, network-free recovery of τ and a regenerative term from the raw signals.
3. **Model-implied stability lobe diagram** — the NEW METHODS QUEUE top pick: a global explanation a machining audience reads instantly, plus a global-faithfulness metric.
4. **Mop-up of the remaining Update-8 caveats** — Temporal Saliency Rescaling (TSR, arXiv:2010.13924) on the attribution path; aggregate delay-engagement statistics to replace the unstable n=1 case studies; the symbolic R²/simplicity gate.
5. **Matched per-fold MLP baseline table** (owed debt) — cheap, local, eval-only; folded in because it costs ~1 h and closes half of a debt named in Updates 6 and 7.

### OUT of Update 9 (deferred, with reasons)
- **No-side-input *training* ablation** (owed debt) — **DEFER from Update 9's deliverables.** It is the only near-term item that needs real cluster compute (a full 5-fold GL retrain, ~8.5 h/fold ≈ 40 A40-h). It belongs to the "how good is the model / how much did the Gao side branch earn" thread, not Update 9's "what physics did it learn" arc, and the white box already bounded its inference-time contribution to a small effect (permutation drop 0.002–0.009 AUC across the replication folds). Recommendation detail in §2.1.
- **Bigger model-development items** — explicitly out of scope, flagged here so the reader sees they were considered and deferred to the paper / a later phase:
  - **Learnable τ** (differentiable delay kernel / neural-DDE variant). The strongest available science claim — with the done ablation and the planned SINDy-DDE recovery it would be a *third* independent recovery of the delay — but it is a new modeling experiment requiring a retrain, so it is next-phase, not Update 9.
  - **MultKAN trunk variant** (KAN 2.0, arXiv:2408.10205) — multiplication nodes to natively express `b·Kt·(x(t)−x(t−τ))`; a model change requiring retrain.
  - **Severity-head quantile redesign** (pinball loss for the heavy tail; MAE 0.88 in-dist vs 3.3–4.8 held-out) — the clearest "model got better" win, but again a retrain.

### 2.1 Owed-debts include/defer call
- **Matched per-fold MLP baseline table → INCLUDE.** One-line reason: it is local/eval-only (pull per-fold AUC from the existing MLP run, job 51442237), costs ~1 h, and finally puts TKAN-vs-MLP on the *same* folds — a debt named in the last two updates.
- **No-side-input training ablation → DEFER (but launch it as a fire-and-forget GL array early).** One-line reason: it needs ~40 A40-h of cluster retrain and serves the model-quality thread rather than Update 9's physics-discovery arc, and its inference-time contribution is already bounded as small. Nuance: kicking off the 5-fold `--no-side-input` array on GL costs ~15 min of my time and zero of the ~20 h eval-only budget (it runs async on the cluster), so I will queue it during Update 9 and report its result in Update 10 / the paper — the debt keeps moving without competing for my analysis hours.

---

## 3. Sequenced task list

Ordering rule: unblock the delay-location story first (τ-sweep), then the independent-recovery story (SINDy-DDE), then the global-explanation figure (lobe diagram, which has an external spec dependency), then the mop-up items that harden the attribution and symbolic sides, then the cheap owed debt, then writing.

Legend — **[GL]** needs the cluster; **[local]** runs on this Mac (eval-only, ~9 min CPU per whitebox fold).

### T1 — Per-window-relative τ-sweep: re-run + analysis + figure
- **What:** the per-window τ-sweep code (sweep each window at multiples of *its own* tooth-pass delay, ×0.5/×1/×2/∞) is being implemented in parallel right now, in `physics.py` + the whitebox runner. This task is the *re-run + analysis + figure*, not the code. Re-run the whitebox physics module on the five `best_auc.pt` checkpoints with the fixed sweep, pull the per-fold sweep curves, and locate the peak in per-window-normalized units.
- **Deliverables:** a per-window-relative τ-sweep figure (peak located near ×1 if the network keys on the true tooth-pass delay), and a one-line per-fold peak-location number to sit beside the ablation in the update.
- **Compute:** **[GL]** for the five-fold re-run (eval-only, minutes/fold), analysis + figure **[local]**.
- **Dependencies:** *blocks on the per-window-sweep code fix landing* (in progress in parallel). Nothing else in Update 9 depends on T1.
- **Hours:** ~3.0.
- **Risk:** low–medium. If the code fix slips, I can run the sweep on the subset of folds whose runner is updated, or fall back to reporting the corrected sweep on the strong folds only. Residual risk: even per-window-normalized, the sample stays near the AUC ceiling (0.987–0.996), so I read the peak *location*, not its depth.

### T2 — SINDy-DDE cross-check (independent delay recovery)
- **What:** recover τ and a regenerative term directly from the raw cutting signals with delay-aware SINDy (arXiv:2407.19640), with no network involved. pysindy has **no native delay operator**, so I build the delayed feature columns by hand (candidate library over `x(t)`, `x(t−τ)` at trial lags) and select the lag/coefficient that best explains the dynamics. Compare the recovered τ to the network's tooth-pass delay and to the ablation result.
- **Deliverables:** a recovered-τ number (point estimate or bounded interval) + a regenerative-term coefficient, and a small figure/table framing this as the *second independent recovery* of the physical delay (ablation = recovery #1, SINDy-DDE = recovery #2).
- **Compute:** **[local]** — small hand-built pipeline over the raw signals, TDD.
- **Dependencies:** none (independent of the network path). Best written up alongside T1 so the two recoveries land in the same section.
- **Hours:** ~5.0.
- **Risk:** medium. Delay recovery from noisy signals can be unstable in the lag estimate. Contingency in §4: if the τ recovery is noisy, report it as a bounded estimate (an interval that brackets the tooth-pass period) rather than a fake point estimate — a bracketed independent recovery still supports the narrative.

### T3 — Model-implied stability lobe diagram + global-faithfulness metric
- **What:** sweep (spindle speed ω, depth b) through a trained checkpoint at fixed modal parameters and plot the model's decision boundary as a stability lobe diagram, overlaid on the analytic/simulated `blim` curve already in the metadata. Compute a global-faithfulness metric (boundary error / agreement vs the `blim` curve).
- **Deliverables:** the SLD-vs-`blim` overlay figure (the figure-rich global explanation the queue calls the top pick) + one global-faithfulness number.
- **Compute:** **[local]** eval-only on the checkpoints.
- **Dependencies:** **blocks on its design spec** — `docs/superpowers/specs/2026-07-03-stability-lobe-diagram-design.md`, which is being scoped in parallel and **does not yet exist**. T3 = build + run + analysis against that spec; it cannot start before the spec lands and it must respect the input-construction approach the spec settles on.
- **Hours:** ~5.0.
- **Risk:** medium–high, concentrated in **input construction**: the model consumes real signal *windows*, not bare (ω, b) pairs, so the sweep needs a principled way to present each (ω, b) grid cell to the checkpoint. This is exactly the open problem the spec is scoping. Contingency in §4: if synthesizing/assigning windows per (ω, b) proves hard, fall back to **binning real windows by their measured (ω, b)** and reading a *measured* decision boundary from data — a slightly weaker but honest SLD that still yields the faithfulness number.

### T4 — Attribution mop-up: Temporal Saliency Rescaling (TSR)
- **What:** add TSR (arXiv:2010.13924) to the attribution path and re-measure the delay's *apparent* attribution share. Motivation from the checked scan: gradient saliency on recurrent models systematically under-credits early/lagged timesteps — exactly where the regenerative term lives — so the un-rescaled attribution understates the delay.
- **Deliverables:** a TSR-adjusted delay-attribution number (expected to rise vs the un-rescaled value), reported as *supporting* evidence behind the interventions.
- **Compute:** **[local]**.
- **Dependencies:** none. Independent of T1–T3.
- **Hours:** ~2.0.
- **Risk:** low. Worst case TSR barely moves the number, which is still a reportable (and honest) result — the intervention remains the headline.

### T5 — Aggregate delay-engagement stats + symbolic R²/simplicity gate
- **What:** two smaller caveat closures. (a) Replace the unstable n=1 case studies (verdict flipped between runs) with an aggregate delay-engagement statistic over many chatter-vs-stable windows. (b) Gate the symbolic fits on R² + a simplicity criterion so no edge gets a closed form it does not support (pykan-style `r2_threshold`), tightening Update 8's "25/25 edges nonlinear, R² 0.96–0.9998" claim.
- **Deliverables:** an aggregate delay-engagement number (chatter vs stable, with spread) that replaces the anecdotal case studies; a symbolic-gate pass-count / R² distribution.
- **Compute:** **[local]**.
- **Dependencies:** none.
- **Hours:** ~2.5.
- **Risk:** low.

### T6 — Matched per-fold MLP baseline table (owed debt, INCLUDE)
- **What:** pull the per-fold AUC from the existing 5-fold MLP run (job 51442237, artifacts under `results/phase3/runs/`) and build the head-to-head TKAN-vs-MLP table on the *same* folds.
- **Deliverables:** a 5-row per-fold TKAN-vs-MLP AUC table (retires the "~0.87 single cited number" from Updates 6/7).
- **Compute:** **[local]** (once the MLP run artifacts are synced local).
- **Dependencies:** MLP run artifacts present locally (rsync per the PROGRESS "pull run artifacts" block, `--exclude='*.pt'`).
- **Hours:** ~1.0.
- **Risk:** low. Only risk is a stale/missing local copy of job 51442237 — resolved by one rsync.

### T7 — Docs, figure assembly, and drafting Update 9
- **What:** assemble the figures, update `PROGRESS.md`, and draft + render Update 9 (md → docx, first-person singular, house style). Inline drafting (not multi-agent), per the noted preference.
- **Deliverables:** `research_update_9.md` + rendered `.docx` with embedded figures.
- **Compute:** **[local]**.
- **Dependencies:** the Definition-of-Done figures/numbers from T1–T6 (see §6).
- **Hours:** ~1.5.
- **Risk:** low.

**Async, off-budget (does not consume the ~20 h):** launch the 5-fold `--no-side-input` training array on GL early in the week so its result is in hand for Update 10 / the paper (see §2.1).

---

## 4. Proposed ~20 h allocation (Days + Hours)

| Work | Days | Hours |
|---|---|---|
| Per-window-relative τ-sweep: GL re-run on 5 checkpoints, pull, peak-location analysis + figure (T1) | Jul 4–5 | 3.0 |
| SINDy-DDE cross-check: hand-built delayed-column pipeline (TDD), τ + regenerative-term recovery, compare-to-network figure (T2) | Jul 5–7 | 5.0 |
| Model-implied stability lobe diagram: build sweep + decision-boundary extraction against the spec, global-faithfulness metric, SLD-vs-`blim` figure (T3) | Jul 7–8 | 5.0 |
| TSR on the attribution path + re-measured delay share (T4) | Jul 8 | 2.0 |
| Aggregate delay-engagement stats + symbolic R²/simplicity gate (T5) | Jul 9 | 2.5 |
| Matched per-fold MLP baseline table (T6, owed debt) | Jul 9 | 1.0 |
| Docs, figure assembly, and drafting Update 9 (T7) | Jul 10 | 1.5 |
| **Total hours** | | **20.0** |

(Launching the async `--no-side-input` GL array is folded into the Jul 4 slot and not counted against the 20 h — it runs on the cluster, not my clock.)

---

## 5. What could slip / contingencies

- **τ-sweep code fix (T1 dependency) slips.** It is being implemented in parallel. If it is not landed when T1's GL slot opens, run the fixed sweep on whichever folds have the updated runner and report the per-window peak on the strong folds only; the ablation headline (already done, all 5 folds) is unaffected and carries the delay story regardless.
- **Lobe-diagram spec (T3 dependency) slips or the input-construction problem proves hard.** If the spec is not finished, T3 waits and I pull T4/T5/T6 forward (they have no dependencies). If input construction at arbitrary (ω, b) is intractable, fall back to **binning real windows by their measured (ω, b)** and reading a measured decision boundary — a weaker but honest SLD that still yields the global-faithfulness number.
- **SINDy-DDE τ recovery is noisy.** Report it as a **bounded estimate** (an interval bracketing the tooth-pass period) rather than a spurious point value. A bracketed independent recovery still delivers the "two independent recoveries" framing; I state the uncertainty plainly, consistent with the series' honesty caveats.
- **Near-ceiling sample AUC persists** (0.987–0.996, unchanged even at 512 windows). This caps what the magnitude-based readouts (τ-sweep depth, permutation magnitudes) can claim. Mitigation: I report *orderings and peak locations*, not magnitudes, and lean on the interventions — the same discipline as Update 8.
- **Update 9 is over-subscribed.** If two of the three headline items (T1/T2/T3) come in clean, that plus the mop-up and the MLP table is already a coherent update; a slipped T3 (its spec is the riskiest dependency) can move to Update 10 without breaking the arc, because T1 (delay located) + T2 (independent recovery) alone deliver "what delay it learned."

---

## 6. Definition of done for Update 9

Update 9 is writable once these figures/numbers exist and trace back to files (manifest-backed, never invented — the Update 8 standard):

1. **Per-window-relative τ-sweep figure** with a located peak in per-window-normalized units, on at least the strong folds (ideally all five). *(from T1)*
2. **SINDy-DDE recovered τ** (point or bounded interval) **+ a regenerative-term coefficient**, with an explicit comparison to the network's tooth-pass delay — framed as the second independent recovery alongside the ablation. *(from T2)*
3. **Stability lobe diagram figure** (model decision boundary overlaid on the analytic/simulated `blim`) **+ one global-faithfulness number** (boundary error / agreement vs `blim`). *(from T3; may move to Update 10 if the spec slips — see §5)*
4. **Matched per-fold MLP-vs-TKAN AUC table** (5 rows, same folds). *(from T6)*
5. **The mop-up numbers that landed:** a TSR-adjusted delay-attribution share *(T4)*, an aggregate delay-engagement statistic replacing the n=1 case studies *(T5)*, and the symbolic R²/simplicity-gate pass count/distribution *(T5)*.

**Minimum viable Update 9** (if T3 slips): items 1, 2, 4, and 5 — "what delay the model learned (τ-sweep) and an independent recovery of it (SINDy-DDE)," plus the owed MLP table and the caveat mop-up. That still fulfills the narrative arc without the lobe-diagram figure.

---

## 7. Explicitly deferred (recorded so it is visibly a choice, not an oversight)

- No-side-input **training** ablation — deferred from deliverables; launched async on GL for Update 10 (§2.1).
- Learnable τ (differentiable delay kernel / neural-DDE) — paper / next phase; would be the third independent delay recovery.
- MultKAN trunk (KAN 2.0, arXiv:2408.10205) — paper / next phase.
- Severity-head quantile redesign (pinball loss) — paper / next phase; the clearest model-quality win.
- Hidden-state dynamical (fixed-point/Jacobian) analysis, physics-constrained counterfactuals, concept probes, calibrated/conformal uncertainty, chatter prognosis, sim-to-real — follow-on-paper scope per the NEW METHODS QUEUE.
