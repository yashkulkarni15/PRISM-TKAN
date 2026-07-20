# Phase 3 TKAN — White-Box Analysis: Design Spec

**Date:** 2026-06-24 · **Branch:** `phase3-tkan` · **Run dir:** `/Users/yash/Downloads/PRI2M/SINDy_DDE`
**Status:** approved design → implementation.
**Commit policy:** never `git add/commit/push` without explicit user instruction; the user commits. Author's name only, no Claude trailer.

## 1. Goal

Turn the trained Phase 3 TKAN severity model into a **complete white box**: explain
*what features* it reads, *how* it computes its decision (the KAN-GRU mechanism, the
delay, the gates, the heads), *why* that maps to milling-chatter physics, and what the
~14k learned **B-spline edge functions** actually are. Deliver (a) a tested analysis
**pipeline** and (b) an auto-rendered **report**.

## 2. Hard integrity constraints (non-negotiable)

- **No fabricated or inflated results.** There is no local checkpoint/data; the heavy
  analysis runs on Great Lakes (GL). Therefore:
  - The **code** is verified now by TDD on synthetic models whose ground truth is known.
  - The **scientific numbers** are verified only when the user runs on GL and real
    outputs return. The report renderer fills *every* quantity from the run manifest.
    The narrative drafted now uses explicit placeholders (`{{...}}`) for any number —
    never an invented value.
- **Data-faithful, not weight-blind.** Importances, plotted curves, and symbolic fits
  are computed over the input range the data **actually visits**, weighted by realized
  output variance — not the grid-uniform / mean-|weight| view the current XAI uses.
- Every artifact carries provenance (checkpoint path, run id, n_windows, normalized flag,
  git sha) in the manifest so a reader can reproduce it.

## 3. The subject model (recap, from code + config)

`TKANModel`: `TKANEncoder` (stacked `TKANCell` KAN-GRU) → `SeverityHead`
(`b_over_blim` regression + `chatter_logit`), optional Gao side branch.
Trained config (`configs/phase3_tkan.yaml`): `hidden_size=64`, `num_layers=1`,
`grid_size=5`, `spline_order=3`, `pool=4` (1024→256), `side_input=true`
(`omega,tpf,tau`, `side_hidden=16`), delay `τ=round(fs/tpf)`.

**Edge inventory (this checkpoint):** encoder gate input
`g = cat([x_t(4), h_prev(64), x_delay(4)]) = 72` → each of `kan_z/kan_r/kan_h` is
`KANLinear(72→64)` = 4,608 edges; ×3 = 13,824. Severity heads `KANLinear(80→1)`
(80 = 64 hidden + 16 side) = 80 each. ≈ **13,984 learned 1-D edge functions**.

**Semantic index map** (a core deliverable — currently the model's inputs are opaque
`in[i]`). For encoder layer 0 (`use_delay=True`):

| index range | block | semantic |
|---|---|---|
| `0..3` | `x_t` | current `(xtpos, ytpos, Forcex, Forcey)` |
| `4..67` | `h_prev` | hidden units `h0..h63` (for `kan_h`: **reset-gated** `r⊙h_prev`) |
| `68..71` | `x_delay` | **τ-delayed** `(xtpos, ytpos, Forcex, Forcey)` |

For severity heads (input = `tanh(cat([context(64), side(16)]))`): `0..63` = encoder
hidden, `64..79` = side-branch features (derived from `omega,tpf,tau`). Upper encoder
layers (k>0, not used by this checkpoint but handled generally): no delay,
`g = cat([x_t(=prev hidden, H), h_prev(H)])`.

**Two correctness fixes the existing XAI needs:**
1. `KANLinear.spline_curves()` returns only the spline residual, omitting the SiLU base
   path `base_weight·SiLU(x)`. The full learned edge function is
   `φ(x) = base_weight[o,i]·SiLU(x) + spline_scale·Σ_c basis_c(x)·W[o,i,c]`. White-box
   reading and symbolic fitting must use **φ (base+spline)**.
2. `prune_edges` at threshold 0.01 pruned **0/13,984** edges on fold 2 — mean-|weight|
   is data-blind and does not discriminate. Replace the single count with a **data-aware
   importance** and a **pruning curve** (AUC vs fraction kept).

## 4. Module architecture (`src/pri2m/xai/`)

All new modules build on a shared foundation. Each is small, single-purpose, importable
without heavy third-party deps at module top (lazy imports for captum/shap/dice_ml/scipy).

### 4.0 Foundation
- **`models/kan.py` (edit):** add `edge_function(n=200)` → `{"x":[n], "curves":[out,in,n]}`
  returning **base+spline** φ over `grid_range`; keep `spline_curves` unchanged
  (back-compat). Optionally `edge_function_at(x_tensor)` evaluating φ at arbitrary points
  (used by the activation collector for visited-range curves).
- **`xai/semantics.py` (new):** `edge_index_map(model)` → for every `KANLinear`, a list of
  `{index, block, name}` describing each input edge (per the table in §3), generalised
  across layer 0 / upper layers / severity heads. Pure, no torch run. Foundation for all
  semantic labelling.
- **`xai/activations.py` (new):** forward-hook collector. `collect_activations(model,
  windows, taus, side, ...)` runs real (normalized) windows, captures per-`KANLinear`
  **actual edge inputs** `x_i` (post-tanh) and **per-edge output contributions**
  `c_{o,i}(x_i)=φ_{o,i}(x_i)`. Returns per edge: visited range `[lo,hi]` (e.g. 1–99 pct),
  histogram, and **data-aware importance** `imp_{o,i}=std_data(c_{o,i})`. This replaces
  mean-|weight| as the importance everywhere downstream.

### 4.1 Analyses (the four the user selected)
- **`xai/symbolic.py` (new):** spline→closed-form. For importance-ranked edges, fit a
  candidate library `{const, linear, quadratic, cubic, saturating tanh(ax+b),
  oscillatory a·sin(ωx+φ)+b, |x|, relu}` to φ (base+spline) **over the visited range**,
  pick best by BIC (penalised R²). Emit per-edge `{name, formula, r2, bic, importance}`.
  Output: `symbolic_edges.csv` + a top-K annotated curve figure.
- **`xai/feature_importance.py` (new):** dataset-level. (a) IG aggregated over many
  windows → mean|attr| per **channel** and per **time-lag**, split chatter vs stable;
  (b) **permutation importance**: permute a channel / side feature across the batch,
  measure AUC and MAE drop. Outputs: `channel_importance.{csv,png}`,
  `side_importance.csv`, `lag_profile.png` (attribution vs time → does it spike near τ?).
- **`xai/physics.py` (new):** delay & DDE linkage. (a) **delay-ablation**: eval with
  `x_delay` forced to zero vs on → ΔAUC, ΔMAE; (b) **τ-sweep**: feed τ∈{0, τ/2, τ, 2τ}
  → does performance peak at the physical τ? (c) **delayed-input weight share** straight
  from the index map (importance mass on `68..71` vs `0..3` vs hidden); (d) **attribution
  spectrum**: FFT of per-time IG → energy at tpf and chatter sidebands. Output:
  `physics_summary.json` + `tau_sweep.png` + `attr_spectrum.png`.
- **`xai/faithfulness.py` (new):** (a) **pruning curve** — zero edges by ascending
  data-aware importance, AUC vs fraction kept ("k% of edges carry the decision");
  (b) IG **deletion curve**; (c) **model-randomization sanity** — attribution on trained
  vs reinit model must differ (cosine ≪ 1). Outputs: `pruning_curve.{csv,png}`,
  `deletion_curve.png`, `sanity.json`.
- **`xai/case_studies.py` (new):** per-decision walkthrough for one clear chatter + one
  clear stable window (high/low predicted `chatter_prob`, correct). Per case: input plot,
  per-channel IG, top firing edges (φ with the input value marked), gate (z/r) + hidden
  trajectories over time, head outputs, and a `why` narrative dict. Output: two
  `case_<label>.png` + `cases.json`.

## 5. Runner, GL hand-off, report

- **`scripts/whitebox_phase3.py` (new):** one entry point. Load checkpoint (dims inferred
  as in `xai_phase3.py`), discover `preprocessing.json`, sample N real windows
  (chatter+stable mix) from the npz tree, normalize, run all six analyses, write all
  artifacts + a single **`whitebox_manifest.json`** (provenance + every scalar result +
  artifact paths). Reuses `_infer_dims`, `_load_model`, `_apply_norm`, `_sample_window`
  patterns from `xai_phase3.py` (refactor shared bits into a small helper rather than
  copy-paste).
- **`slurm/phase3_whitebox.sbatch` (new):** self-contained GL wrapper, live project root
  `/nfs/turbo/coe-jaabell/PRI2M`, parameterized by run-id/fold and npz root; mirrors the
  existing `slurm/phase3_train.sbatch` conventions. Runs on a strong checkpoint
  (default fold 2, `a654129d…`, AUC 0.975) and is trivially re-runnable per fold.
- **`scripts/render_whitebox_report.py` (new):** reads `whitebox_manifest.json`, fills
  `docs/phase3_tkan/whitebox_analysis.md` (template) → final markdown. Sections:
  **Features · How · Why (physics) · B-splines · Faithfulness · Caveats**. The physics
  narrative is drafted now (grounded in code); all numbers/figures auto-fill from the
  manifest. Final docx via pandoc is a later step once GL outputs return.

## 6. Testing strategy (TDD, synthetic ground truth)

Shared fixtures from `tests/_phase3_synth.py`. Each module gets `tests/test_xai_<mod>.py`:
- `semantics`: index map labels match the §3 table for a delay-on layer-0 + a severity
  head; covers an upper-layer (no-delay) case.
- `kan.edge_function`: φ = base+spline numerically equals a manual `base·SiLU(x)+spline`;
  differs from `spline_curves` when `base_weight≠0`.
- `activations`: shapes/finite; a **planted** high-variance edge ranks above a zeroed
  edge; visited range brackets the fed inputs.
- `symbolic`: **recovers a planted linear edge** (r²≈1, picks "linear"); a planted
  saturating edge picks "saturating" over "linear" by BIC.
- `feature_importance`: permuting the channel a synthetic model depends on drops AUC most;
  lag profile peaks at the planted lag.
- `physics`: on a synthetic model trained to need the delayed input, delay-ablation drops
  AUC; τ-sweep peaks at the true τ.
- `faithfulness`: pruning curve is monotone non-increasing in AUC as more edges are
  removed; randomization sanity gives low cosine.
- `case_studies`: returns one chatter + one stable case with correct predicted direction;
  figures written.
- `whitebox_phase3` end-to-end: builds a tiny `TKANModel`, saves a state_dict, runs the
  runner on a synthetic npz tree → manifest + all artifacts exist, manifest scalars finite.

**Gate:** full `pytest -q` green after each wave; no regressions vs the 63-passed/1-skip
baseline (skip is the intentional iisignature-absent branch).

## 7. Execution waves

- **Wave 1 (foundation, built by the driver with TDD):** `kan.edge_function`,
  `xai/semantics.py`, `xai/activations.py`. Highest-risk shared correctness — kept in
  one hand. Gate green.
- **Wave 2 (parallel TDD subagents, disjoint files):** `symbolic`, `feature_importance`,
  `physics`, `faithfulness`, `case_studies`. Each depends only on the Wave-1 foundation.
  Gate green.
- **Wave 3 (driver):** `scripts/whitebox_phase3.py`, `slurm/phase3_whitebox.sbatch`,
  `scripts/render_whitebox_report.py`, `docs/phase3_tkan/whitebox_analysis.md` template +
  physics narrative. Real CPU smoke run on a synthetic checkpoint. Gate green.
- **Hand-off:** user rsyncs + runs `phase3_whitebox.sbatch` on GL; outputs return; driver
  renders the report from the real manifest (no invented numbers).

## 8. Out of scope / YAGNI

Trajectory & physics-residual heads (gated/blocked). The extra-dataset feature (explicitly
deferred). sigKAN. Live retraining — analysis is eval-only on existing checkpoints.
