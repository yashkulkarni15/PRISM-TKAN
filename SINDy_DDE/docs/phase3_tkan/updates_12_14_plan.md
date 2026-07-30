# Updates 12–14 — work division to the August 14 report

Authored 2026-07-25. Divides the remaining summer work across three research updates,
each ~20 h, ending in time for Phase 9 (technical report, Aug 10–15).

> **STATUS 2026-07-29.** U13 has been built and run in full, four days into its window, and
> **U14-A landed five weeks early** — both retrains had already completed before this plan was
> written and nobody had read them. Net effect on the plan:
> - **U13-A (survival): DONE.** 19/42 above 0.85 ⇒ criterion 7 met; mean 0.702, min 0.086.
> - **U13-B (DiCE): DONE.** 42/42 found, 36 physically stable, outcome agreement 27/42. The first
>   comparison metric was mislabelled and has been replaced (axis breakdown + outcome agreement).
> - **U13-C (regenerative): DONE and REFUTING.** The network learned **x(t) + x(t−τ)**, a sum, not
>   the regenerative difference. The plan's qualitative criterion in the table below moves from
>   "Partial" to **NOT MET**, on the most direct test available for it.
> - **U14-A (both retrains): DONE, both negative.** Side inputs ≈ +0.002 AUC; the quantile severity
>   head does not fix the heavy tail.
> - **U14 REWRITTEN 2026-07-29** around three arms: U14-A writes up the two negatives, U14-B
>   replaces the unrunnable "cross-fold function agreement" item with a population-level
>   activation-mode analysis carrying a mandatory untrained-network null, and U14-C explicitly
>   descopes the ablation-matrix criterion (3 of its 4 cells have no implementation to ablate).
> - **U12 still cannot ship:** §4 waits on the TSR control (third submission), §5 waits on the
>   band-notch `max_rel_err` sentinel fix.
> Full detail + numbers: `PROGRESS.md` Status board and its **NOW OPEN** list.

**Deadline:** technical report to Prof. Abell, **August 14, 2026**. 20 days out at authoring.

**Cadence** (continues the existing one — U10 covered Jul 5–11, U11 Jul 12–18):

| Update | Window | Theme |
|---|---|---|
| 12 | Jul 19–25 | The delay under the microscope — and the metric that was hiding it |
| 13 | Jul 26 – Aug 1 | Counterfactuals that survive uncertainty |
| 14 | Aug 2–8 | What the network actually needs — do the learned functions reproduce? |

This maps onto the MDP plan's Phase 8 (ablations + DiCE counterfactuals, Aug 3–9) and
leaves Phase 9 clear. Update 13 pulls DiCE *forward* of its planned window because it is
cheap, local, and closes a named quantitative criterion.

---

## Why these three, and not others

Audit of 2026-07-25 against the MDP plan's success criteria. Unmet or partial items only:

| Plan criterion | State | Addressed in |
|---|---|---|
| DiCE counterfactuals on ≥10 unstable points | **MET (2026-07-29).** DiCE ran on the same 42 points: 42/42 found, 36 (85.7%) physically stable vs the physics search's 81.6%. | U13-B — closed |
| …each with **Monte Carlo survival > 0.85** | **MET, with a caveat worth more than the criterion.** 19/42 clear 0.85 (≥10 required), but mean 0.702 / median 0.685 / min 0.086 @ 512 draws ⇒ **fewer than half the recommendations survive** 2% spindle jitter + 0.05 mm depth tolerance + 10% blim uncertainty. | U13-A — closed |
| Full ablation matrix (T-KAN vs MLP, KL on/off, residual physics on/off, hardcoded vs learnable delay) | **1 of 4, and the other 3 are unimplementable (2026-07-29).** No KL term exists anywhere; the physics-residual head is a stub that raises (`heads.py:217`); learnable-τ is deferred (`heads.py:144`) and would need a new feature plus a 5-fold retrain. | **U14-C — descoped, see below** |
| Symbolic expression contains x(t) − x(t−τ) without hardcoding | **NOT MET (2026-07-29).** U13-C ran 5/5 folds: the learned splines implement the **sum** x(t) + x(t−τ), not the difference (mean score −0.46…−0.64, +1 = negation; 23–36% summing vs 0.1–1.0% negating). Real vs a flat untrained null (−0.023), so it is a learned structure — just not this one. SINDy-DDE still recovers τ (0.986) independently. | U13-C — closed |
| Hidden activations encode physical quantities | **Failed** and retired (concept probe held-out R² −331) | not revisited |
| Noise curves at 0/1/5% across three methods | Partial — fuzz suite uses its own units | U12 (band-notch closes the spectral half) |
| ktc/fn/damping inverse problem | No artifact found; project pivoted to classifier + severity | **descoped — see below** |

**Explicit descope.** The inverse-problem criterion (ktc, fn, damping within 10% on ≥80% of
held-out datasets) is not reachable in 20 days and no partial artifact exists to build on.
τ is the one physical parameter this project recovers, and it recovers it three ways
(ablation, τ-sweep, SINDy-DDE at τ*/τ 0.986). The report should state this plainly as a
scope change rather than leave the criterion silently unmet.

**Two naming defects to fix before the report** (both would put false statements in a
methods section):
- The plan names **PySR**; the repo uses a custom BIC library fit over {poly, sin, tanh,
  abs, relu, saturating} in `xai/symbolic.py`. Do not write "PySR".
- `_VALID_SEVERITY_MODES` calls a mode `"huber"` when the loss is `weighted_mse`
  (`models/heads.py:34`; no Huber/smooth-L1 exists in `training/losses.py`).
- **`PROGRESS.md` states the spline atlas fits at "R² 0.985–1.000".** Measured over all 125
  stored rows the range is **0.9257–0.9999**: 11 rows below 0.98, 4 below 0.95 (worst: fold 4
  at 0.9257, fold 0 at 0.9312). All 125 are `accepted`, but the quoted floor is too optimistic
  and would misstate fit quality in a methods section. Fix the line in `PROGRESS.md` too.

---

## Update 12 — Jul 19–25

**Theme:** the TSR run that was supposed to decide a null, and decided something else.

### Already done in this window
- Physics-floor stressors SDD build, Tasks 1–6 (2026-07-20). Suite 634 passed / 1 skipped.
- Quantile severity wiring + SLURM `EXTRA_ARGS` passthroughs (2026-07-20). Suite 603 → green.
- TSR at scale: 160 shards across 5 folds, GL Jul 23–24, all returned, merged, verified.

### The finding to write up

`PROGRESS.md` pending item 3 framed this as a binary — either the null survives at scale and
the attribution view is retired, or the small sample was the whole story. **Neither happened.**

`tr_peak_lag` (`scripts/whitebox_phase3.py:222`) is the argmax of the autocorrelation of the
relevance signal over lags `[2, N/2]`. When the autocorrelation decays monotonically — no
periodicity resolved — argmax returns index 0 and the metric emits **2**, its own search floor.

- **550 / 640 windows (85.9%) return exactly 2.** The fold means of 2.6–10.0 reported in
  Update 11 are arithmetic over a statistic that is mostly a "no answer" sentinel.
- Among the **90 windows that resolve a real peak**, the peak lands on that window's own τ far
  beyond chance: exact match 31/90 (34%) against a null of 3.34, **p = 0.00005**; on an exact
  harmonic 60/90 (67%) against a null of 17.9, **p = 0.00005**; median ratio
  `peak_lag / tau_raw` = **1.0000**.
- Empirically `peak_lag` and `tau_raw` are in the **same units**, so Update 11's comparison
  against "~22 steps after four-fold pooling" was on the wrong axis too.

So the Update 11 negative is not underpowered — it is **unsound**. It must be corrected, and
it cannot yet be replaced with a positive, because of the control below.

### Work items

**U12-A — fix the metric.** Report the resolved fraction explicitly; add a delay-lock
statistic that tests the relevance autocorrelation *at* lag τ against a null instead of taking
a global argmax; never average the floor sentinel into a mean. TDD.

**U12-B — the construction-artifact control.** The delayed channel is built by shifting the
signal by τ, so the relevance vector may carry lag-τ structure *by construction*, independent
of anything learned. Run the fixed metric on (a) a randomly-initialised checkpoint and (b) the
trained net with deliberately wrong τ. If the lock survives both, the result is void and the
attribution view really is retired. If it degrades, TSR becomes a **fourth independent line of
delay evidence** alongside ablation, τ-sweep, and τ-jitter.

**U12-C — close the SDD build.** Task 7 (band-notch contrast) is pre-scoped: brief at
`.superpowers/sdd/pf-task-7-brief.md`, base snapshots taken, and the plan's Step-5 factual
error already corrected (use `batch["fn"]`; do **not** reload the table). Then Task 8 and the
final whole-branch review.

### Honesty constraint

Update 12 must **correct Update 11 §3**, not quietly supersede it. The published text gives
"underpowered, only four windows" as the reason for the negative; that reason is now known to
be the wrong diagnosis. Say so.

**A second Update 11 correction, found 2026-07-29.** Update 11 states the atlas fits at
"R-squared 0.985 to 1.000" in two places — the atlas section and the closing summary. The true
range over all 125 stored rows is **0.9257–0.9999**, with 11 rows below 0.98 and 4 below 0.95
(per-fold minima 0.9312 / 0.9815 / 0.9768 / 0.9640 / 0.9257). The headline claim it supports —
all 25 edges carry an *accepted* formula on every fold — is unaffected; the fit floor is not.
The error originated in `PROGRESS.md` and propagated from there.

**Resolved by patching the source** (user decision, 2026-07-29), not by deferring to U12:
`PROGRESS.md` is fixed, `research_update_11.md` is fixed in both places, and the `.docx` has
been re-rendered from it (three figures re-embedded, verified). The repo copy of Update 11 now
carries a dated correction note in its provenance comment. **The copy already delivered to
Prof. Abell predates this fix** — decide separately whether that warrants a mention; nothing in
the plan depends on it. This does **not** discharge the §3 TSR correction, which is still owed
by Update 12 and was deliberately left unpatched in Update 11.

---

## Update 13 — Jul 26 – Aug 1

**Theme:** counterfactuals that survive uncertainty, and the regenerative term in the splines.

**U13-A — Monte Carlo survival rates.** The missing half of plan criterion 7. For each
counterfactual `(ω_cf, b_cf)`, sample machining uncertainty — spindle-speed jitter,
depth-of-cut tolerance, ktc/fn uncertainty — and compute the fraction of draws still below
`blim(ω)`. `physics/altintas_budak.py` already supplies the analytic boundary. A
recommendation that only holds at the nominal point is not a recommendation a machinist can
use; the survival rate is what makes it risk-rated.

*Design (integration point already exists).* `validate_against_blim(cf_results, blim_fn, *,
margin=0.0)` (`xai/counterfactuals.py:312`) is exactly the abstraction to reuse: it already
takes a list of CF result dicts plus a `blim_fn: omega -> blim` and reduces them to per-case
records and aggregates. Add a **sibling** `monte_carlo_survival(cf_results, blim_fn, *,
uncertainty, n_draws, seed)` with the same shape, where per case

```
survival = mean over draws of [ b_draw < blim_fn(omega_draw) ]
```

and the draws perturb `(ω_cf, b_cf)` by the machining tolerances. Keeping it a sibling rather
than a new argument to `validate_against_blim` preserves that function's tested contract and
the 81.6% agreement number it produces. Aggregate to `n_above_0_85` so the plan's
"survival rates above 0.85" criterion is answered by a single field. The margin knob already
in `validate_against_blim` is the deterministic special case (`n_draws=0`), which is a useful
consistency check on the new code.

**U13-B — run the actual DiCE path.** `dice_counterfactuals()` exists
(`xai/counterfactuals.py:52`, `dice_ml`, `method="random"`, sklearn regressor backend) and has
never been run; the shipped 81.6% came from `physics_constrained_counterfactual`. Run DiCE on
the same 42 operating points against the same model-prediction surrogate. Two independent CF
generators agreeing on identical inputs is a validation the plan never asked for and a
reviewer will.

**U13-C — the regenerative spline test.** The single most direct available test of the plan's
qualitative criterion. Every delay edge is `in=68` (`delay:xtpos`) or `in=69`
(`delay:ytpos`) on `encoder.layers.0.kan_{r,z,h}`. If the network implements regeneration,
then for a given `(kan, out)` unit the delayed-channel edge should be approximately the
**negative** of the instantaneous-channel edge on that same unit — because that composition
*is* x(t) − x(t−τ). Evaluate both φ via `KANLinear.edge_function` over the data-visited range
and correlate. Built-in controls: folds 0/1/2 carry 5 delay edges each in the top 25, folds
3/4 carry 1 — the same split the ablation shows (drops 0.230/0.487/0.266 vs 0.484/0.449).

A strong negative correlation would be the regenerative difference emerging from learned
functions with nothing hardcoded — substantially stronger than "delay edges carry 15% of
importance."

---

## Update 14 — Aug 2–8

**Theme:** what the network actually needs — reproducibility of the learned functions, and
the ablation table we can honestly build.

> **Rewritten 2026-07-29.** U14-A landed five weeks early and both arms came back negative, so
> the window's compute is free. U14-B's "cross-fold function agreement" item was checked
> against the stored manifests and **cannot run as written** (see below). It is replaced by a
> population-level activation-mode analysis. U14-C is new: the plan's headline ablation-matrix
> criterion is mostly unimplementable and must be descoped explicitly rather than left open.

**U14-A — the two completed retrains, written up as results.** No submission needed; both ran
before this window opened. Both are negative, and both close questions that were open for
weeks.
- **`--no-side-input` 5-fold ablation** (open since 2026-06-14): side inputs are worth
  **≈ +0.002 AUC**. Every prior TKAN fold ran `side: True`, so this was the first isolation of
  the Gao side-input at *training* time. It agrees with the white-box permutation proxy
  (drop 0.002–0.009 at 512 windows), which the plan had assumed was too weak to settle it. The
  proxy was right.
- **Quantile severity head 5-fold**: does **not** fix the heavy tail. Severity remains the one
  output that never hardened (in-dist MAE 0.88 ± 0.65; held-out 3.3–4.8).

*Write-up constraint:* pull per-fold numbers from the returned run artifacts. Do not restate
the summary figures above as if they were per-fold results.

**U14-B — population-level activation-mode analysis** (replaces "cross-fold function
agreement"; no retrain needed).

*Why the original item cannot run.* It asks whether the same `(kan, in, out)` edge learns the
same φ across five folds. Measured on the five stored `whitebox_manifest.json` atlases: 101
distinct edges across the top-25 lists, **17 appear in ≥2 folds, and 0 appear in all five**
(pairwise overlap 2–4 of 25). There is no 5-fold sample to compare. Two further defects in the
item as written: the manifest stores `formula`/`r2`/`importance`, **not** the curves, so "the
atlas already holds the data" is false; and hidden-unit indices are permutation-arbitrary
across independently trained folds, so most top edges are not identifiable by index at all.

*What is identifiable.* From `xai/semantics.py`, only three strata carry fold-stable indices:
`encoder.layers.0.kan_{z,r,h}` inputs **0–3** (`x_t`: xtpos, ytpos, Forcex, Forcey) and
**68–71** (`x_delay`: delay:*), comparable on `in` but never on `out`; and the
`severity.{cls,reg}_head` **side** block (`in ≥ hidden_size`), where `out=0` is stable too.
Everything fed by `h_prev`/`r*h_prev`/`hidden`/`L0_h*` is comparable only as a shape
population, never by identity. The comparable strata still give 3 gates × 8 channels × 64
out-units = **1536 curves per fold**, 7680 across five — an ample population for clustering.

*Design.*
1. **Extract all relevant edges, not the top 25.** `activations.collect_activations` already
   returns `importance[o,i]` and `visited_lo/hi` for every `KANLinear`;
   `KANLinear.edge_function(n)` returns the whole `[out, in, n]` curve tensor in one call.
2. **Gate on data-aware importance** to drop near-dead edges *before* any normalisation — a
   shape-normalised dead edge is indistinguishable from a shape-normalised dominant one.
3. **Stratify** by channel × layer × gate, per the table above. Never pool a hidden-fed edge
   with a physical-channel edge.
4. **Cluster spline-only curves.** The base path is exactly `base_weight[o,i] * SiLU(x)` — a
   one-parameter family carrying no shape information, which would otherwise form one large
   trivial cluster. `edge_function` returns base+spline (deliberately — it fixed the
   SiLU-omitting plot), so this needs a `component=` selector.
5. **Record the base fraction and interpret base+spline.** Cluster on the spline deviation,
   but any claim about what the network *computes* must be about the full φ.
6. **Run the identical pipeline on randomly initialised models.** Reuse the untrained-net
   construction from `scripts/tsr_control_phase3.py`: deepcopy of the *loaded* model with
   every submodule's `reset_parameters()` called, plus its two guards (refuse if no submodule
   exposes `reset_parameters()`, refuse if the weights did not change). Untrained controls cost
   one forward pass each, so run **≫5 seeds**, not 5 — the null's spread is the denominator of
   the whole test.
7. **Claim learned modes only on a pre-committed margin.** Fit prototypes on four folds, assign
   the fifth, score cross-fold agreement; the trained score must beat the untrained
   distribution by a margin fixed **before** the numbers are seen, mirroring U12-B's
   `untrained_lock >= 0.8 × |trained_lock| ⇒ artifact`. Compare at **matched N** — if the
   importance gate passes 90% of trained edges and 5% of untrained ones, the two arms are not
   the same population.
8. **Mode ablation, and make it the headline.** Zero every edge assigned to one mode and
   measure the change in **held-out** AUC (the 121–155 path, baseline 0.908 ± 0.013). Two
   controls are mandatory: a random edge set matched on both **count and summed importance**,
   and an equal-size set drawn from the other modes. Without them, "removing 300 edges hurt"
   says nothing. Warm up on the original item's other half — **causal ablation of individual
   top edges** — which is the same machinery at n=1 and converts the importance *ranking* into
   a necessity claim.

*The dimensionality caveat, baked in from the start.* `grid_size: 5` + `spline_order: 3` ⇒
`n_coeff = 8`, and within a `KANLinear` every curve is `B·c` with an **identical** basis. PCA
over sampled curves is therefore PCA over 8-dimensional coefficient vectors in the basis
metric: it cannot exceed 8 components, and a low-dimensional spectrum is partly architectural.
Report the spectrum, but **"k modes explain X% of variance" is never a finding** — only the
trained-vs-untrained margin in step 7 is.

*Free preview (already computed, no new compute).* Tallying the symbolic families across the
125 stored atlas rows: sin 46, tanh 40, cubic 27, quad 12; sin/tanh/quad present in 5/5 folds,
cubic in 4/5. Module split: the encoder is tanh-dominated (30 of 56), `cls_head` is sin+cubic
dominated (49 of 65). Fold 1 is the outlier (0 cubic, 15 tanh) and is worth explaining. This is
the claim in embryo — and it is exactly the kind of result the untrained control exists to
qualify.

*Pilot order.* `xtpos` (`in=0`) first, `ytpos` (`in=1`) as the base-driven comparison, then the
delay channels (`in=68/69`). Expand to forces only if the prototypes are stable.

*Prerequisite.* No checkpoints are on disk locally (`find -name "*.pt"` outside `archive/`
returns nothing). Sync the five `best_auc.pt` down, or run on GL.

*Provenance.* The methodology is external in origin; the numbers must not be. Our importance is
the std of φ over realised inputs, our ranges are 1st/99th percentiles of visited data, and our
grid is 5+3 — no external figure would transfer even if we wanted it. Rerun everything on our
own checkpoints; cite no external numbers.

**U14-C — the ablation matrix: what is reachable, and an explicit descope.** The plan's
criterion names four cells. Checked against the tree:

| Cell | State |
|---|---|
| T-KAN vs MLP | **MET** — every fold beats MLP ~0.87; margin +0.042 min |
| KL on/off | **No implementation.** No KL term exists in `training/losses.py`, `models/`, or `configs/phase3_tkan.yaml` |
| residual physics on/off | **No implementation.** `models/heads.py:217` is a documented stub that raises; blocked on Altintas–Budak cutting-force coefficients |
| hardcoded vs learnable delay | **Not reachable by Aug 14.** τ is `round(fs/tpf)` by approved decision 2; `heads.py:144` defers the learnable-τ unlock. Needs a new model feature *and* a 5-fold retrain at ~8 h/fold |

Three of four cells cannot be ablated because there is nothing to switch off. Say this plainly
in the report as a scope change, exactly as with the inverse problem — do not leave the
criterion silently at "~1 of 4".

*What replaces it.* An ablation table built from work that actually ran: TKAN vs MLP, side-input
on/off (U14-A), delay-channel ablation (5 folds, mean drop 0.38), graded τ-jitter, spline
coarsening, and mode ablation (U14-B step 8). That is a stronger "what the network needs" table
than the original four cells would have produced, and every row of it exists.

---

## Sequencing constraints

1. **U12-B gates any TSR claim.** Nothing about TSR goes in the report until the control
   returns. Both outcomes are publishable; an unverified positive is not.
2. **U14-A needs no submission** — both retrains landed 2026-07-20, five weeks early. The one
   U14 prerequisite is *checkpoint availability*: U14-B needs the five `best_auc.pt` files,
   which are not on local disk. Sync them down (or run U14-B on GL) before the window opens;
   this is a minutes-to-hours transfer, not a queue-bound job.
3. **U13 is fully local.** No GL dependency, so it cannot be blocked by queue time — which is
   why the criterion-closing work lives there.
4. **No-commit policy holds.** "Complete" = tests green + review clean, work left in the tree.
   Review diffs come from the working tree (`git diff`), not commit ranges.

## Risks

- **U12-B comes back positive-for-artifact** (the lag-τ lock survives on an untrained net).
  Then TSR is genuinely dead, Update 12 becomes a shorter methods-correction piece, and U13-C
  carries the interpretability weight for that cycle. Acceptable — the correction is publishable
  on its own.
- **The untrained null comes back indistinguishable** (trained folds cluster no more
  consistently than randomly initialised ones). Then the activation modes are basis geometry —
  8 spline coefficients over a shared grid — and not learned structure. This is a *result*, not
  a failure: it would bound how far anyone may interpret an individual spline curve or symbolic
  formula from this architecture, including our own Update 11 atlas figures. Write it up as
  such. Mode ablation (step 8) still stands on its own in that case, because it is a causal
  measurement that does not depend on the modes being reproducible.
- **GL queue congestion in August** no longer threatens U14-A (landed). It still applies to the
  U12-B TSR control (12 h wall, one fold) and to any held-out eval passes U14-B step 8 needs.
  Mitigation: U14-B's extraction and clustering are CPU-cheap and fully local once the
  checkpoints are synced; only the mode-ablation eval passes want GL.
- **Working tree is large** (26k+ deletions against `phase3-tkan` at authoring). The final
  whole-branch review needs this triaged first or it will not be reviewable in one pass.
