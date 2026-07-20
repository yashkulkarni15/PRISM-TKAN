# Phase 3 TKAN — Model-Implied Stability Lobe Diagram (SLD): Design Spec

**Date:** 2026-07-03 · **Branch:** `phase3-tkan` · **Run dir:** `/Users/yash/Downloads/PRI2M/SINDy_DDE`
**Status:** approved design → implementation (design doc only; no code in this change).
**Commit policy:** never `git add/commit/push` without explicit user instruction; the user commits. Author's name only, no Claude trailer.

Companion to the white-box spec (`docs/superpowers/specs/2026-06-24-phase3-whitebox-design.md`);
this is the **top pick** in the PROGRESS "NEW METHODS QUEUE" (2026-07-01). It is eval-only on the
existing 5-fold `best_auc.pt` checkpoints — no retraining, no new data.

## 1. Goal & the two deliverables

Read a trained TKAN's chatter decision back out in the machining-native
**(spindle speed ω, axial depth of cut b)** plane and show it as a **stability lobe diagram (SLD)**,
overlaid on the analytic/simulated stability limit `blim` already tabulated in the dataset metadata.

Two deliverables:

1. **Global explanation figure** — the model-implied stability boundary in (ω, b), drawn on top of
   the analytic `blim(ω)` lobe curve and the labeled operating points. A machining audience reads
   "the network learned the lobes" in one glance. This is a *global* explanation, complementing the
   *local/mechanistic* white-box results (delay ablation, symbolic edges).
2. **A scalar global-faithfulness metric** — boundary error of the model-implied SLD vs the analytic
   `blim` curve (defined precisely in §7). This turns "looks like the lobes" into a number and is a
   genuinely new axis of evidence: the white-box AUC/ablation numbers say *the model discriminates
   and uses the delay*; the SLD metric says *its decision surface has the right global shape in
   process space*.

## 2. Hard integrity constraints (non-negotiable — inherited from the white-box spec)

- **No fabricated or inflated results.** No local checkpoint/data; the heavy run is on Great Lakes
  (GL). The **code** is verified now by TDD on synthetic models with known ground truth; the
  **scientific numbers** are verified only when the user runs on GL. The renderer fills *every*
  quantity from `sld_manifest.json`; the narrative uses explicit placeholders for any number.
- **The analytic overlay is the dataset's own precomputed `blim`, not our re-derivation.** We do
  **not** recompute an Altintas–Budak SLD (that needs the blocked cutting-force convention — see
  `PhysicsResidualHead` in `models/heads.py`). We overlay the `blim` column the digital-machining
  database already ships. §6 gives the Altintas–Budak context for the reader, but the plotted curve
  is data, not our formula.
- **Ceiling honesty carries over.** Per-window AUC on easy samples ceilings near 0.99 (white-box
  caveat); the honest headline is val 0.955 / held-out 0.908. The SLD metric is a *shape* metric over
  the whole operating envelope, independent of that ceiling, but the manifest must still record the
  per-cell sample AUC so the report can flag it.
- Every artifact carries provenance (checkpoint, run id, dataset(s), fold, grid definition, git sha).

## 3. The subject model & the (ω, b) → signal gap (recap)

`TKANModel` consumes a **batch of signal windows**, not process scalars:
`batch = {"window": [B, L, C], "tau": [B], "omega","tpf","fn","fs": [B], ...}`,
`C=4` channels `(xtpos, ytpos, Forcex, Forcey)`, `L=1024`, `pool=4`. Its chatter output is
`sigmoid(model(batch)["chatter_logit"])`. The optional Gao side branch reads only
`omega`/`tpf`/`tau` (never `b`/`blim` — hard leakage guard, `FORBIDDEN_SIDE_FEATURES`).

So "sweep (ω, b) through the checkpoint" is **not** a direct call — the model never sees `b` and only
sees ω indirectly (through `tpf`/`tau` and through the physics baked into the real vibration signal).
Resolving this is the central design problem (§4).

**Where ω, b, blim, and modal params live** (confirmed by inspection of
`results/phase3/phase3_dataset.csv`, 33,719 rows / 120 datasets):

| quantity | column | units | notes |
|---|---|---|---|
| spindle speed ω | `omega` | rpm | 1000–50000; per-dataset grids |
| axial depth b | `b_mm` | mm | the chosen operating depth |
| **analytic stability limit** | **`blim`** | **mm** | **the overlay curve.** Verified a *pure function of ω within a dataset* (`nunique(blim per (dataset,ω)) == 1`) |
| severity target | `b_over_blim` | – | `== b_mm / blim` (verified, max abs err 2e-13) |
| chatter label | `label` | {0,1} | `== int(b_mm >= blim)` (0 mismatches / 33,719) |
| tooth-pass freq | `tpf` | Hz | `== teeth·ω/60` (verified) |
| delay | `tau` | s | `== 1/tpf`; sample delay = `round(fs/tpf)` |
| modal params (fixed *per dataset*) | `fn`(=fntx), `ktx`, `zetatx`, `fnty`, `ktc`, `teeth`, `cut_dir`(=`ud`), `damping`, `diameter_mm`, `ro` | – | `nunique == 1` within every dataset; 80 distinct modal combos across the 120 datasets |

The full multi-mode modal table + the grid generator (`omega min/max/increment`, `b min/increment`,
`Noise std`) lives in `data/metadata/Dataset descriptions.xlsx` (sheet `Sheet1`, 622 rows). The CSV is
the authoritative per-operating-point source for the overlay; the xlsx is reference only.

**Two structural facts the implementer must design around:**
- `blim` goes **negative at low speed** (700 rows, `exclude_reason=invalid_blim_low_speed`) — the
  no-process-damping artifact. Overlay and metric are defined **only where `blim > 0`**.
- The (ω, b) sampling is **boundary-adaptive, not rectangular.** At each ω the handful of `b_mm`
  values are placed to *bracket* `blim(ω)` (e.g. dataset 1 ω=7500 → b∈1..15, blim=14.1; ω=2000 →
  b∈1..3, blim=1.47). There is no dense fixed b-grid, so boundary extraction is **per-ω-column**
  (§5), not a 2-D image threshold.

## 4. Approach analysis — how to sweep (ω, b) through the checkpoint

### (A) Bin real windows by their (ω, b) — **RECOMMENDED PRIMARY**
For a fixed modal group, every real file already *is* a labeled (ω, b) operating point. Group files
by (ω, b), run the model on their real (z-scored) windows with their true `tau`/`omega`/`tpf`/`fs`,
and read `mean sigmoid(chatter_logit)` per cell → a P(chatter) field over the sampled (ω, b) points.
The model-implied boundary is where that field crosses 0.5; overlay `blim(ω)`.
- **Feasible now, zero new machinery.** Reuses `build_labeled_batch` / `Phase3SequenceDataset` /
  `NormStats` / the `_load_model` loader verbatim — the white-box runner already does exactly this
  sampling; SLD only changes the *grouping* (by (ω, b) instead of by class) and the *readout*.
- **Faithful "sweep":** each real window carries the true physics of its operating point (the
  vibration signature *is* the response at that ω, b), plus the true side inputs. Nothing is
  fabricated or off-distribution.
- **Cost:** coverage limited to (ω, b) points that exist. But the well-sampled datasets are dense
  enough to trace a boundary (see §3.5 grid), and **param-twins pool** to thicken per-cell counts.

### (B) Simulate signals from the milling DDE at each (ω, b) — **FALLBACK / FUTURE**
Integrate the regenerative-chatter DDE at a dense (ω, b) grid, window the synthetic signals, feed the
model → an arbitrarily dense field, no coverage gaps.
- **Not eval-only-cheap and risks distribution shift.** Needs a validated simulator that reproduces
  the digital-machining-database physics (modal superposition, process damping `Cn`, up/down `cut_dir`)
  and emits all four signed channels at the training sample rate. The repo has `sindy_pipeline/` but
  no such forward simulator wired to the training distribution; the training signals came from a
  specific generator we do not control. A sim/train mismatch would make any boundary error
  un-attributable (model error vs sim error). Defer; revisit alongside the SINDy-DDE cross-check
  (PROGRESS TODO 5) and the process-damping physics task.

### (C) Vary only the side branch on a fixed/averaged window — **DIAGNOSTIC PANEL ONLY**
Hold the signal window fixed, sweep only `omega`/`tpf`/`tau` through the side MLP → interrogates what
the side branch alone learned about ω.
- **Low validity as *the* SLD, useful as a control.** It is physically incoherent as a boundary
  ("same vibration, different rpm") and the white-box already found the side branch contributes only
  ≈0.012–0.016 AUC (near-zero, inconclusive at ceiling). It cannot express a b-axis at all (b never
  enters the model). Keep it as an *optional* small side-panel: sweep side-ω on a fixed window and
  show the (expected small) shift — an honest statement of how little the side branch bends the
  boundary, not the boundary itself.

### Recommendation
**Primary = (A).** Eval-only, reuses the entire white-box data path, produces a genuine
model-implied boundary in real process space, and the analytic overlay is already in the CSV.
**Fallback = (B)** for grid density, gated on a validated DDE simulator (out of scope here).
**(C)** ships only as a labeled diagnostic sub-panel, never as the headline boundary.

## 5. Method under approach (A) — concrete pipeline

Given a checkpoint + fold, a chosen **modal group** (one dataset or a set of param-twin datasets that
share all modal params — pool to thicken cells; see §10):

1. **Select rows.** From `phase3_dataset.csv`, take the group's rows with `blim > 0` and an existing
   NPZ (`_npz_exists`). Keep `exclude_supervised` rows out of the metric but optionally show them on
   the plot (greyed) for context.
2. **Build the P(chatter) field.** For each unique (ω, b) cell, gather all its windows across the
   group's files, z-score with the fold's `NormStats` (from `preprocessing.json`, exactly as the
   white-box runner), build the batch with the **true** `tau`/`omega`/`tpf`/`fn`/`fs`, run the model,
   and record `p_cell = mean_windows sigmoid(chatter_logit)`, `n_windows`, and the true `label`.
   No input is synthesized — this is selection + averaging of real windows. (Windows carry the real
   operating physics; the side branch gets the real ω/tpf/tau.)
3. **Extract the model boundary per ω-column.** For each ω, sort cells by `b`, and find the **lowest
   b where `p_cell` crosses 0.5**, by linear interpolation between the bracketing cells →
   `b_model(ω)`. If a column never crosses (all-stable or all-chatter over its sampled b-range),
   record `b_model(ω)=NaN` with a reason (`no_crossing_low` / `no_crossing_high`) so the metric can
   handle censored columns honestly rather than inventing a crossing.
4. **Analytic boundary.** `b_analytic(ω) = blim(ω)` = the unique `blim` for that ω in the group
   (verified constant per (dataset, ω)). Units already mm; no conversion.
5. **Metric + figure.** Compute §7 metrics over the ω-columns / cells; render §8 figure + tables.

Rationale for per-column extraction (not 2-D contouring): the b-sampling is ragged and boundary-
adaptive (§3), so a regular-grid `contour(P==0.5)` would need heavy interpolation and invent values in
un-sampled (ω, b) regions. Per-column crossing uses only sampled cells and degrades gracefully.
A regular-grid contour MAY be added as a *smoothed visual* (nearest/linear griddata) but must NOT feed
the metric.

## 6. Analytic overlay: source, definition, unit alignment

- **Source (exact):** `results/phase3/phase3_dataset.csv`, column **`blim`** (mm), keyed by
  (`dataset`, `omega`); take unique (`omega`, `blim`) pairs for the modal group, filter `blim > 0`,
  sort by `omega`. That *is* the analytic/simulated stability lobe for those modal params. Secondary
  reference: `data/metadata/Dataset descriptions.xlsx` (`Sheet1`) for the modal constants and the
  grid generator; `results/phase3/run_metadata.json` for provenance of the CSV build.
- **Altintas–Budak context (for the reader; we do not recompute):** the analytic milling SLD is the
  zero-order/lobed stability limit
  `b_lim(ω) = −1 / (2·K_t·Re[Λ(ω_c)])`, with the lobe/speed relation
  `n = 60·ω_c / (N·(ε + 2kπ))` (`N`=`teeth`, `K_t`=`ktc`, `Λ`=eigenvalue of the oriented FRF built
  from the modal params `fntx/ktx/zetatx`, …, `k`=lobe number). Negative/absent `blim` at low ω is the
  no-process-damping (`Cn=0`) regime; the process-damping datasets are the intended future fix.
- **Unit alignment:** everything stays CSV-native — ω in **rpm**, b and blim in **mm**. The model's
  own ω dependence is consistent (`tpf = teeth·ω/60` Hz, `tau_s = 1/tpf`, `tau_samples = round(fs/tpf)`
  — all verified). No rpm↔rad/s or m↔mm conversion is introduced; if the xlsx (SI: `b (m)`, `d (m)`)
  is ever read directly, convert `×1000` to mm to match the CSV (the metadata generator already does
  `b_mm = b*1000`, `blim_mm = interp*1000`).

## 7. Global-faithfulness metric (precise)

Let `Ω` = ω-columns with a defined `b_analytic(ω)=blim(ω)>0`. Report all of:

1. **Primary — mean absolute boundary error (mm):**
   `MAE_b = mean_{ω∈Ω, b_model(ω) finite} | b_model(ω) − blim(ω) |`, plus the **relative** form
   `MAPE_b = mean | b_model(ω) − blim(ω) | / blim(ω)`. Report `n_columns_used` and
   `n_columns_censored` (no crossing) so a high censor count can't hide behind a small MAE.
2. **Secondary — stable-region IoU (cell-wise, robust to ragged grid):**
   `stable_model = {cells: p_cell < 0.5}`, `stable_analytic = {cells: b < blim}`;
   `IoU = |model ∩ analytic| / |model ∪ analytic|` over all sampled cells.
3. **Robustness — analytic-label agreement:** fraction of sampled cells where the model's stable/
   unstable call matches the analytic call, `mean[ (p_cell<0.5) == (b<blim) ]` (this is just the
   model's accuracy against the analytic label; interpolation-free, so it never depends on boundary
   fitting). Also report the per-cell sample AUC of `p_cell` vs `label` and flag if `>0.99` (ceiling).

The **headline number** is `MAE_b` (mm) with `MAPE_b` (%); IoU and label-agreement are the supporting
robust checks. All are computed in the module, written to the manifest, and quoted by the renderer —
never hand-typed.

## 8. Outputs & manifest

Output dir (mirrors the white-box convention): `results/phase3/xai/<tag>/sld/` where `<tag>` encodes
fold + modal group, e.g. `fold2_ds29grp`.

Artifacts:
- `sld_<tag>.png` — **the deliverable figure:** P(chatter) field (scatter/pcolor over sampled cells),
  the model boundary `b_model(ω)`, the analytic `blim(ω)` curve, and the labeled operating points;
  annotated with `MAE_b`/`MAPE_b`/IoU. Copied to
  `results/figures/research_update_assets/sld_<tag>.png` for the research update.
- `sld_field.csv` — per cell: `omega, b_mm, blim, label, p_chatter, n_windows`.
- `sld_boundary.csv` — per ω: `omega, blim, b_model, crossing_status`.
- `sld_manifest.json` — provenance + results + artifacts + errors + a flat `report` block, in the
  **same schema** as `whitebox_manifest.json` (top keys `provenance / results / artifacts / errors /
  report`), so `render_whitebox_report.py`-style rendering drops the numbers in with zero invention.

Manifest fields to add (so every reported number is manifest-sourced):
- `provenance`: `checkpoint, preprocessing, run_id, tag, val_auc, fold, dataset_ids, parent_group,
  n_cells, n_omega, n_files, n_windows, omega_min, omega_max, side_input, pool, git_sha,
  rendered_date`.
- `results.sld`: `mae_b, mape_b, iou_stable, label_agreement, per_cell_auc, n_columns_used,
  n_columns_censored, boundary: [{omega, blim, b_model, status}], field_summary`.
- `report`: `sld_mae_b, sld_mape_b, sld_iou, sld_label_agreement, sld_per_cell_auc,
  sld_boundary_table` (markdown), `fig_sld` (basename), plus an auto **caveat** string when
  `per_cell_auc > 0.99` or `n_columns_censored / n_omega > 0.3`.

Composition: SLD ships its **own** `sld_manifest.json` (it may be run per modal group, many per fold).
When run in the same job as the white box, the runner also injects a compact `sld` summary block under
the main `whitebox_manifest.json["results"]` so the combined report can cite it — but the SLD's own
manifest stays the source of truth.

## 9. Module & runner architecture

New, small, single-purpose, lazy heavy imports (matplotlib/sklearn), mirroring `src/pri2m/xai/`:

- **`src/pri2m/xai/lobe_diagram.py` (new)** — pure/core:
  - `build_pchatter_field(model, rows, npz_root, norm, cfg) -> field` (list of cell dicts:
    `omega, b_mm, blim, label, p_chatter, n_windows`). Reuses `Phase3SequenceDataset` + `collate_windows`.
  - `extract_model_boundary(field) -> [{omega, blim, b_model, status}]` (per-ω 0.5 crossing).
  - `analytic_boundary(rows) -> [{omega, blim}]` (unique, `blim>0`, sorted).
  - `boundary_faithfulness(field, boundary) -> {mae_b, mape_b, iou_stable, label_agreement,
    per_cell_auc, n_columns_used, n_columns_censored}`.
  - `plot_sld(field, boundary, analytic, metrics, path) -> path`.
  - Optional `side_only_boundary(model, fixed_window, omega_grid)` for the (C) diagnostic panel.
- **`scripts/stability_lobe_phase3.py` (new)** — one entry point, mirrors `whitebox_phase3.py`: args
  `--checkpoint --npz-root --table --output-dir --dataset-ids/--parent-group --fold --val-auc
  --pool --window-length --sample-seed`; reuse `xai_phase3._find_preprocessing` + `_load_model`;
  select the modal group's rows; build field → boundary → metrics → figure + CSVs +
  `sld_manifest.json`; each analysis wrapped in the white-box `_safe(...)` record-and-continue helper.
- **`slurm/phase3_stability_lobe.sbatch` (new)** — self-contained GL wrapper, live project root
  `/nfs/turbo/coe-jaabell/PRI2M`, mirrors `slurm/phase3_whitebox.sbatch` (PYTHONUNBUFFERED, per-fold
  RUN_ID/TAG/VAL_AUC env). Default: strong fold 2 checkpoint on a dense param-twin group (ds 29/89/119).
- **Report:** extend `scripts/render_whitebox_report.py` (or a thin `render_sld_report.py`) to fill an
  SLD section from `sld_manifest.json["report"]` — narrative drafted now, numbers/figure auto-filled.

## 10. Test plan (TDD, synthetic ground truth — repo house style)

Shared fixtures from `tests/_phase3_synth.py`; new `tests/test_xai_lobe_diagram.py` +
`tests/test_stability_lobe_phase3.py` (mirroring `test_xai_physics.py` / `test_whitebox_phase3.py`):

- **Boundary extraction on a planted field:** a synthetic P field that is a clean logistic in b at
  each ω with a known crossing `b*(ω)` → `extract_model_boundary` recovers `b*(ω)` (within interp
  tol); an all-stable column → `status="no_crossing_low"`, `b_model=NaN` (no invented crossing).
- **Metric correctness:** a field whose model boundary == `blim` everywhere → `MAE_b≈0`, `IoU≈1`,
  `label_agreement≈1`; a field shifted by a known Δb → `MAE_b≈Δb`; a fully-inverted field → `IoU`
  small, `label_agreement≈0`.
- **Analytic boundary:** `analytic_boundary` returns unique sorted (ω, blim), drops `blim≤0`, and is
  invariant to how many b-rows share each ω (uses the per-ω constant).
- **Field builder:** on a tiny synthetic model + a synthetic NPZ tree (fixture), `build_pchatter_field`
  returns one cell per sampled (ω, b) with `n_windows>0`, finite `p_chatter∈[0,1]`, and a model that
  keys chatter off b yields a monotone-in-b P column.
- **Runner end-to-end:** build a tiny `TKANModel`, save a state_dict, synth NPZ tree + a small table
  with a known blim(ω) → `stability_lobe_phase3.main(...)` writes `sld_<tag>.png`, `sld_field.csv`,
  `sld_boundary.csv`, `sld_manifest.json`; manifest scalars finite; `report` has `fig_sld` + the
  metric keys; a ceiling/censoring caveat fires when planted.
- **Unit alignment guard:** a row set in SI (`b` in m) is converted ×1000 and matches the mm field.

**Gate:** full `pytest -q` green after each wave; no regressions vs the current
469-passed/1-skip baseline (skip = the intentional iisignature-absent branch).

## 11. Open questions / risks (implementer must decide)

1. **Grid coverage vs boundary tracing.** The ragged, boundary-adaptive b-sampling means some ω-columns
   have too few b-values to bracket 0.5 (censored). *Decision:* pool param-twin datasets (share all
   modal params) to thicken cells — e.g. ds 29/89/119 (same `parent_group` 29, 1071 cells each, ω
   1000–30000) — and report `n_columns_censored` prominently. Do NOT extrapolate a crossing.
2. **Which "fixed modal params" / group.** There are 80 modal combos; the SLD is inherently
   *per modal group* (params are constant within a dataset). *Decision:* default to the densest
   param-twin group (ds 29/89/119) for the headline figure; make it a CLI arg and optionally emit a
   small multi-group panel. State the modal constants (fn/ktx/ktc/teeth/cut_dir) in the figure caption.
3. **Distribution shift is minimal under (A) but not zero.** Real windows are in-distribution, but if a
   fold *trained on* the chosen group, its per-cell AUC will be optimistic. *Decision:* prefer a
   checkpoint/fold whose **validation** fold contains the group (held-out), or run all 5 folds and
   report the SLD metric cross-fold; record `fold`/`parent_group` so the reader can check.
4. **Ceiling / per-cell confidence.** Easy cells saturate P→{0,1}; the crossing is then set by a few
   near-boundary cells. *Decision:* report `per_cell_auc`, `n_windows` per cell, and auto-caveat at
   AUC>0.99; consider weighting the crossing by cell confidence (documented, off by default).
5. **`b_model` can be multi-valued across lobes.** Per-ω "lowest crossing" captures the first
   stability limit; higher-b re-stabilization (unlikely here but possible) is dropped. *Decision:*
   record all crossings in `sld_boundary.csv`; use the lowest for the scalar metric (matches how
   `blim` is defined as the *limit*).
6. **Approach-(C) validity.** The side-only diagnostic must be clearly labeled "side-branch only,
   fixed signal" and never conflated with the real boundary; gate it behind a flag.

## 12. Effort estimate & dependencies

- `xai/lobe_diagram.py` + tests (TDD): ~4–5 h.
- `scripts/stability_lobe_phase3.py` runner + manifest + tests: ~3 h.
- `slurm/phase3_stability_lobe.sbatch` + report section + narrative: ~2 h.
- Local synthetic end-to-end smoke + gate: ~1 h.
- **Total ≈ 10–11 h local build (eval-only).** Then a short GL run (~minutes CPU per group, like the
  white box) + render from the real manifest.

**Dependencies:** none new — numpy/pandas/torch/sklearn/matplotlib already present. Reuses
`xai_phase3._load_model` / `_find_preprocessing`, `data/phase3_sequence.*`, and the white-box
`_safe` + manifest/report conventions. No retraining, no simulator, no external data. Approach (B)
(DDE simulator) and the process-damping low-speed fix are explicitly out of scope.

## 13. Out of scope / YAGNI

DDE signal simulation (approach B); recomputing the Altintas–Budak SLD (blocked
`PhysicsResidualHead`); the physics-constrained counterfactuals artifact (separate queue item, though
it *consumes* this SLD to validate Δω suggestions); learnable-τ or MultKAN modeling work; the
126–155 clean replication data (blocked on data format).
