# Phase 3 TKAN — Fuzziness Degradation Suite + Spline-Edge Atlas: Design Spec

**Date:** 2026-07-16 · **Branch:** `phase3-tkan` · **Run dir:** `/Users/yash/Downloads/PRI2M/SINDy_DDE`
**Status:** approved design → implementation.
**Commit policy:** never `git add/commit/push` without explicit user instruction; the user commits. Author's name only, no Claude trailer.

## 1. Goal (professor's review feedback, 2026-07-16)

Three asks from the professor, one round of work:

1. **"Make the model more fuzzy and see how the results are affected — up to what point
   does it understand the physics?"** → a graded **degradation suite**: apply input-side
   and model-side fuzziness at increasing levels to the existing `best_auc.pt`
   checkpoints (eval-only, no retraining) and, at every level, re-measure not just
   accuracy but the **delay-reliance signature** (delay-ablation AUC drop). The
   deliverable is a set of *physics-understanding-vs-fuzz* curves whose breakdown point
   answers "up to what point".
2. **"More explainability/interpretability on the spline edges with features."** → a
   **spline-edge atlas**: figure-quality panels of the top learned edge functions with
   real feature names, data-density rugs, and symbolic overlays; feature-named
   importance heatmaps; per-feature aggregates.
3. **"Complete anything pending."** → close the **eval-only XAI queue** from
   `docs/phase3_tkan/PROGRESS.md`: TSR wiring, real-network counterfactuals,
   5-checkpoint conformal ensemble, concept-probe debugging.

Everything lands as figures + numbers for **Research Update 10**.

**Out of scope (user decision 2026-07-16):** `--no-side-input` 5-fold retrain,
severity-quantile retrain (both GL training, deferred), 126–155 replication (blocked on
data owner).

## 2. Hard integrity constraints (inherited from the whitebox spec)

- **No fabricated or inflated results.** Code is TDD-verified locally on synthetic
  models; scientific numbers exist only after the user runs GL and results return.
  Every reported quantity comes from a run manifest — never invented.
- Every artifact carries provenance (checkpoint sha/run id, n_windows, seeds, fuzz
  levels, git sha).
- Stochastic fuzzers (noise) run ≥3 seeds and report mean ± sd; deterministic fuzzers
  (smoothing, coarsening, pruning) run once.
- The level-0 ("no fuzz") point of every sweep must reproduce the un-fuzzed baseline
  within float tolerance — a built-in self-check the runner asserts and records.

## 3. Workstream A — degradation suite

### 3.1 Module: `src/pri2m/xai/degradation.py`

Six fuzzers behind one registry (`FUZZERS: dict[str, Fuzzer]`). Input-side fuzzers are
pure functions `batch -> batch` (never mutate in place); model-side fuzzers are
`model -> model` on a `copy.deepcopy` (never touch the original).

**Input-side** (operate on z-scored windows, i.e. post-`NormStats.apply`, matching what
the model actually consumes):

| fuzzer | knob | levels | notes |
|---|---|---|---|
| `input_noise` | σ (SD units, z-scored space) | 0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0 | additive Gaussian on the 4 signal channels only (not side scalars); seeded |
| `input_smooth` | moving-average width (RAW samples, pre-pool — windows are 1024-sample raw; the model pools ×4 internally) | 1, 2, 4, 8, 16, 32 | reflect-padded centered moving average; width 1 = identity; widths ≤4 sit inside the model's own pooling (expected ≈flat — a built-in sanity region); larger widths progressively erase the chatter band |
| `tau_jitter` | jitter fraction of each window's own τ | 0, 0.05, 0.10, 0.25, 0.50 | per-window random sign, seeded; τ' = max(1, round(τ·(1±frac))); reuses the per-window-τ plumbing from `physics.tau_sweep_relative` |

**Model-side** (operate on the checkpointed model):

| fuzzer | knob | levels | notes |
|---|---|---|---|
| `spline_coarsen` | target grid_size | 5 (=orig), 4, 3, 2, 1 | for every `KANLinear`: densely sample each edge's learned spline component on `grid_range`, least-squares refit onto a coarser B-spline basis, swap coefficients into a new `KANLinear(grid_size=k)`; base (SiLU) weights copied unchanged |
| `weight_noise` | σ_rel (multiplicative) | 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 | `w ← w·(1+ε), ε~N(0,σ_rel²)` on spline coefficients + base weights of every `KANLinear`; 3 seeds |
| `prune` | keep-fraction of edges | reuse existing `faithfulness.pruning_curve` grid | not reimplemented — its result is imported into the same dashboard for completeness |

`spline_coarsen` implementation detail: refit **spline component only** (the learned
`edge_function` minus the base/SiLU part) sampled at ≥50 points per edge on
`grid_range`; solve one least-squares per (grid_size, layer) with the shared basis
matrix, batched over edges (`np.linalg.lstsq` on a `[n_points, k+order]` design vs a
`[n_points, n_edges]` target). Record per-layer refit R² in the manifest so coarsening
error is itself visible.

### 3.2 Probes (measured at every fuzz level, every fuzzer)

1. **Chatter AUC** — per-window over the sampled batch, the same convention every
   whitebox number uses (`physics._auc` on window labels); NOT per-file, so fuzz curves
   are directly comparable to the published delay-ablation drops.
2. **Severity MAE** — per-window over the sampled batch (from `delay_ablation`'s
   `mae_on`), same convention.
3. **Delay-ablation drop** — `physics.delay_ablation` re-run **on the fuzzed
   model/batch**: `drop(level) = AUC_fuzzed − AUC_fuzzed+delay-ablated`. This is the
   physics-reliance signature; the level where it collapses toward 0 (while baseline
   AUC is still above chance) is the "stops understanding the physics" point.
4. **x_delay weight share** (model-side fuzzers only; cheap, from `delayed_weight_share`).

**Breakdown-point summary:** for each fuzzer, the manifest records the first level where
(a) AUC < 0.75 (halfway to chance from ~1.0), (b) delay-drop < 0.05, (c) AUC ≤ 0.55
(≈chance). The summary table reports all three; the narrative uses (b) as "physics
breakdown" and (a)/(c) as performance breakdowns. Thresholds are constants in
`degradation.py` — reported, not tuned.

### 3.3 Runner: `scripts/fuzz_phase3.py`

CLI mirrors `whitebox_phase3.py`. `build_labeled_batch` (and its `_npz_exists`/sampling
helpers) move from the whitebox script into a shared `src/pri2m/xai/sampling.py`;
`whitebox_phase3.py` imports from there (existing whitebox tests guard the refactor)
and the fuzz runner uses the same import: `--checkpoint --table --npz-root [--preprocessing auto] --n-windows 512
--sample-seed --out-dir --fuzzers all|csv-list --seeds 3`. Emits:

- `fuzz_manifest.json` — provenance + per-fuzzer per-level probe values (+ per-seed
  values and mean/sd for stochastic fuzzers) + level-0 self-check + breakdown points.
- Figures (all from the manifest, via a small plotting module):
  - one **degradation panel per fuzzer**: x = level, left y = AUC (line + seed band),
    right y (twin) = delay-ablation drop; reference lines at AUC 0.5 and the un-fuzzed
    baseline; training-noise reference marked on the `input_noise` panel.
  - one **combined dashboard** (2×3 grid of the panels).
  - **breakdown-point table** (also as markdown in the report).

`slurm/phase3_fuzz.sbatch` (CPU, whitebox-style env/args) + `slurm/run_fuzz_all_folds.sh`
looping the 5 fold checkpoints (same RUN_ID/TAG pattern as `run_whitebox_all_folds.sh`).
Cross-fold aggregation figure via `scripts/plot_fuzz_replication.py` (whitebox-replication
pattern: 5 folds' curves overlaid per fuzzer).

`sync_phase3_tkan_to_hpc.sh` gains the new module, script, sbatch, wrapper, tests.

### 3.4 Tests (TDD, synthetic — no real data/checkpoint needed)

- Each input fuzzer: level-0 identity; noise is seeded-reproducible; smoothing reduces
  high-frequency energy monotonically; τ-jitter frac=0 identity, τ' ≥ 1, sign mix.
- `spline_coarsen`: refit at grid_size=5 reproduces edge functions (R² ≈ 1, outputs
  close); coarsened model still runs forward; grid_size strictly decreasing basis count.
- `weight_noise`: σ=0 identity; different seeds → different weights; original model
  untouched (deepcopy).
- Probe loop: on the shared synthetic model (`tests/_phase3_synth.py`), manifest schema
  complete, level-0 equals directly-computed baseline, breakdown-point logic on a
  hand-crafted fake curve.
- CLI smoke: synth checkpoint end-to-end → manifest + figures exist, level-0 self-check
  passes.

## 4. Workstream B — spline-edge atlas (inside the whitebox runner)

New module `src/pri2m/xai/spline_atlas.py` + a `_atlas()` analysis in
`whitebox_phase3.py` (manifest key `spline_atlas`, report section):

1. **Edge atlas figure** (`spline_atlas.png`): top-K (default 16) edges by data-aware
   importance (`activations.py`), as small multiples. Each panel: learned φ(x) =
   `KANLinear.edge_function` (base+spline) over `grid_range`; a rug/histogram strip of
   the inputs that edge actually receives (from `collect_activations`); the accepted
   symbolic fit overlaid dashed with its form + R² (from `symbolic.py`, R²-gated);
   title = `gate: feature → unit#` using `semantics.edge_index_map` names, with
   **τ-delayed inputs and side-input features visually flagged** (color/badge).
2. **Feature-named importance heatmaps**: upgrade `_plot_importance_heatmaps` so the
   input axis shows real feature names, with the `x_t | h_prev | x_delay | side` blocks
   delimited and labeled (h_prev collapsed/summarized so 72 rows stay readable).
3. **Per-feature aggregate bars** (`feature_edge_importance.png`): summed data-aware
   edge importance per named input feature, per gate (z/r/h + severity head), grouped
   current vs τ-delayed vs side.

Tests: synthetic model → atlas returns K panels with names matching `edge_index_map`;
delayed/side flags correct; figure files written; manifest lists the top-K edges with
(gate, feature, unit, importance, symbolic form, R²).

## 5. Workstream C — close the eval-only queue

1. **TSR wiring** — `whitebox_phase3.py` calls
   `attribution.temporal_saliency_rescaling` (built + tested 2026-07-04, never invoked);
   manifest gains TSR time/feature shares (incl. the delay channels' share, expected to
   rise vs plain saliency per arXiv:2010.13924); report section + comparison line
   against the plain-gradient share. Re-run on GL with the fold sweep.
2. **Real-network counterfactuals** — new `scripts/predict_windows_phase3.py`: run a
   checkpoint over the sampled windows → predictions CSV (file id, window idx, chatter
   proba). Feed it to
   `counterfactuals_phase3.py --predictions` so the CF search targets the **network's**
   decisions, replacing the RandomForest table-label surrogate (which produced the
   0.952 agreement number with `checkpoint=null`). Re-run fold 2 on GL; report
   flip-rate / physical-validity / agreement against the true SLD.
3. **Conformal ensemble** — `conformal_phase3.py` accepts N checkpoints (mean
   probability ensemble across the 5 folds); report ensemble coverage @0.90, selective
   accuracy, and comparison to the single-fold 0.914 result.
4. **Concept-probe debug** — systematic-debugging on `regeneration_phase` (R² −228:
   degenerate; suspect target definition/scaling or a constant/leaky feature) and
   `tpf_separation` (R² −0.82). Outcome is either a fixed probe or an honest
   "unprobeable as defined" note with the diagnosis — both acceptable; no silent drop.

## 6. Execution plan & deliverable

1. Local TDD build of A + B + C1–C3 code changes (parallel subagents where files are
   disjoint); full suite must stay green (baseline: 560 passed / 1 skip).
2. C4 (probe debug) is local analysis on existing manifests/data first; code fix only
   if the diagnosis demands one.
3. User syncs (`sync_phase3_tkan_to_hpc.sh`) and submits on GL: fuzz suite ×5 folds,
   whitebox re-run ×5 folds (now with atlas + TSR), predictions→counterfactuals (fold
   2), conformal ensemble. All eval-only/CPU.
4. rsync results back → cross-fold figures → **Research Update 10**
   (`summer_research/Research Updates/research_update_10.md` + docx, first-person,
   ~20 h Days+Hours table, figures from `results/figures/research_update_assets/`).
   Placeholders only until real numbers land, per the Update-9 discipline.

## 7. Risks / notes

- **Ceiling compression** (known from whitebox): per-window baseline AUC ≈ 0.99 means
  small fuzz levels may show little AUC movement — the delay-drop probe and the *shape*
  of the curve carry the signal; report flags the ceiling as before.
- `spline_coarsen` at grid_size 1–2 may fail to represent even the base behavior —
  that is itself the finding (physics needs ≥k knots); the refit-R² column separates
  "coarse spline can't fit" from "model breaks".
- τ-jitter uses integer delays after `pool=4`; at small τ the ±5% level can round to
  0 change — the manifest records the *realized* mean |Δτ| per level so the x-axis is
  honest.
- Runtime: naive cost is (Σ levels×seeds) ≈ 40–50 model evals per fold + ablation
  double-evals ≈ 2× that; at whitebox's ~9-min-per-13-analyses scale this stays in the
  tens of minutes per fold on CPU — fine for GL standard nodes; `--fuzzers` subsetting
  exists if it isn't.
