# Phase 3 Fuzziness Suite + Spline-Edge Atlas + XAI Queue Closure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build (1) an eval-only graded degradation ("fuzziness") suite that measures how far the trained TKAN's physics understanding survives input- and model-side blurring, (2) a spline-edge atlas that makes the learned edge functions readable with real feature names, and (3) close the pending eval-only XAI queue (TSR wiring, real-net counterfactual predictions, conformal-ensemble wrapper, concept-probe debug).

**Architecture:** New `src/pri2m/xai/degradation.py` (fuzzer registry + probes) driven by a new standalone runner `scripts/fuzz_phase3.py`, mirroring the whitebox-runner pattern (manifest JSON = single source of truth, figures rendered from it). `build_labeled_batch` moves to a shared `src/pri2m/xai/sampling.py`. The spline atlas is a new `src/pri2m/xai/spline_atlas.py` wired into the existing `scripts/whitebox_phase3.py` as additional `_safe(...)` analyses. Everything is eval-only on existing `best_auc.pt` checkpoints — no retraining.

**Tech Stack:** PyTorch (self-built `KANLinear` / `TKANModel`), numpy/pandas/sklearn, matplotlib (Agg), pytest with the shared synthetic fixtures in `tests/_phase3_synth.py`. GL (Great Lakes) sbatch for real runs — always launched by the user, never by workers.

**Spec:** `docs/superpowers/specs/2026-07-16-phase3-fuzz-and-spline-xai-design.md`

## Global Constraints

- **NEVER `git add` / `git commit` / `git push`.** All work stays in the working tree; the user commits manually (their name only, no Claude trailer). Plan steps therefore have NO commit steps — a task ends when its tests pass.
- **Run dir:** `/Users/yash/Downloads/PRI2M/SINDy_DDE`. Python: `/opt/anaconda3/bin/python` (3.13.5). Run tests as `python -m pytest ...` from the run dir (conftest adds repo root; `src/` layout via editable install).
- **Suite baseline: 560 passed / 1 skipped (561 collected).** Full suite must end with 0 failures and no regressions; the 1 skip is the intentional iisignature-absent branch.
- **TDD:** every task writes its failing test first, sees it fail, implements, sees it pass.
- **No fabricated numbers:** runners emit manifests; every reported/plotted value comes from a manifest. Scientific numbers do not exist until the user runs GL.
- **Figures:** matplotlib with `matplotlib.use("Agg")` inside the plot function (repo pattern), `dpi=110`+, `fig.tight_layout()`, written under the runner's `--output-dir` (and `results/figures/research_update_assets/` only for cross-fold aggregates). Follow the visual style of `physics.plot_tau_sweep` / `faithfulness.plot_pruning_curve`.
- **AUC convention:** per-window over the sampled batch (`physics._auc`), matching every published whitebox number.
- **GL jobs:** account `jaabell0`, partition `standard` (CPU), env-var parameterized sbatch, `PYTHONUNBUFFERED=1` — copy the header/env block of `slurm/phase3_conformal.sbatch`.
- **Worktree:** work directly on branch `phase3-tkan` in the run dir (repo policy: uncommitted working tree; do not create worktrees — the tree is intentionally dirty).

## Key existing contracts (read once, trust throughout)

- `build_labeled_batch(table_csv, npz_root, preprocessing, n_windows, cfg, supervised_only=True, seed=0) -> (batch, labels)` currently at `scripts/whitebox_phase3.py:75`; batch keys: `window` f32 `[B,L,C]`, `b_over_blim` f32 `[B]`, `label` f32 `[B]`, `sample_weight` f32 `[B]`, `tau` long `[B]`, `omega/tpf/fn/fs` f32 `[B]`, `dataset/file/parent_group/mode_label` lists. `labels` = int numpy `[B]`.
- `physics.delay_ablation(model, batch, labels, big_tau=10**6) -> {"auc_on","auc_off","auc_drop","mae_on","mae_off"}` (mae keys present when `b_over_blim` in batch).
- `activations.collect_activations(model, batch, max_points=4096) -> {kan_name: {"inputs" [N,in], "visited_lo" [in], "visited_hi" [in], "importance" [out,in], "module"}}`.
- `physics.delayed_weight_share(model, stats) -> {"overall": {"x_t","h_prev","x_delay"}, "per_gate": {...}}`.
- `semantics.edge_index_map(model, channel_names=DEFAULT_CHANNELS) -> {kan_name: [{"index","block","name"}, ...]}`; blocks: `x_t`, `h_prev`/`r*h_prev`, `x_delay` (names `delay:<ch>`), `hidden`, `side` (names `side<m>`); `DEFAULT_CHANNELS=("xtpos","ytpos","Forcex","Forcey")`.
- `symbolic.fit_edge(kan, o, i, lo, hi, n=200, *, r2_threshold=None, simplicity_tol=1e-3) -> {"name","formula","r2","bic","params","out","in","lo","hi"}` (+`accepted`,`best_family` when gated). `symbolic.symbolic_table(model, activations, top_k, *, r2_threshold=0.9) -> list[rows]`.
- `faithfulness.rank_edges(stats) -> [(kan_name, o, i, importance), ...]` desc; `faithfulness.pruning_curve(model, batch, labels, activations, fractions=(0.02,0.05,0.1,0.25,0.5,1.0)) -> {"fraction_kept","auc"}`.
- `kan_native._iter_kanlinear(model)` yields `(qualified_name, KANLinear)`.
- `KANLinear`: `spline_weight [out,in,n_coeff]`, `n_coeff = grid_size + spline_order`, buffer `grid` (knots), `b_splines(x[...,C]) -> [...,C,n_coeff]` (degree-0 basis is right-open ⇒ basis at exactly `grid_range[1]` is all-zero), `edge_function(n) -> {"x","curves" [out,in,n]}` (base+spline), `spline_curves(n)` (spline only, includes `spline_scale`).
- `TKANModel.forward(batch)` needs `window` (+`tau`, + side features `omega/tpf/tau` if `side_input`); returns `{"b_over_blim","chatter_logit"}`. KANLinears at `encoder.layers.{i}.kan_{z,r,h}`, `severity.reg_head`, `severity.cls_head`. `model.pool` is a forward-time attr (avg-pools window ×pool and `scale_tau(tau,pool)=clamp(round(tau/pool),min=1)`); runners set `model.pool = int(args.pool)` after load.
- Model loading for runners: `from scripts.xai_phase3 import _find_preprocessing, _load_model`; `_load_model(checkpoint, args, preprocessing)` reads `args.input_size/hidden_size/grid_size/spline_order` and reconstructs side models from `side_mean` in the state_dict + `preprocessing["side"]["features"]`.
- `attribution.temporal_saliency_rescaling(model, window, target="b_over_blim", tau=1, steps=32, extra_scalars=None, *, alpha=0.5, baseline=None, feature_masking=True) -> {"tsr" [L,C], "ig" [L,C], "time_relevance" [L], "salient" [L] bool, "alpha"}`. `extra_scalars` rejects keys `"window"`/`"tau"`.
- `counterfactuals_phase3.build_surrogate` with `--predictions` CSV: needs column `file` + first-present of `("prob","chatter_prob","severity","p_chatter")`; inner-merges on the table's `file` column, fits `RandomForestRegressor` on `(omega, b_mm) -> prob`.
- `conformal_phase3.py` ALREADY ensembles: `--checkpoint` is `action="append"`; >1 → `conformal.ensemble_probs`. sbatch `phase3_conformal.sbatch` takes space-separated `CHECKPOINTS` env.
- Synthetic fixtures: `tests/_phase3_synth.py` → `make_synth_npz_tree(root, *, clean_ids=(1,2,3), with_mirrors=True, n_files=4, fs=10000.0, length=4096, seed=0)`, `make_synth_table(npz_root) -> DataFrame`, `CHANNELS`. Synthetic checkpoint pattern: `tests/test_whitebox_phase3.py::_make_checkpoint` (side-input `TKANModel(input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3, side_input=True, side_features=("omega","tpf","tau"), side_hidden=4)` → `state_dict()` to `<tmp>/runs/synthrun/checkpoints/best_auc.pt` + sibling `preprocessing.json` `{"norm": {"mean": [...4], "std": [...4]}, "side": {"features": ["omega","tpf","tau"]}}`).
- The 5 fold checkpoints on GL (for slurm wrappers): fold0 `61c67bddbdbd49ba9727ff92f4814c86` (0.9597), fold1 `c21e58dc0e624e1dbc602ec63a2dc23e` (0.9122), fold2 `a654129d2cdc460093e21373f31ee5a5` (0.9752), fold3 `3a40b71514cd41b98916330c000d700a` (0.9605), fold4 `0937883918354ed899a61afb90aac14f` (0.9695).

## Execution waves

```
Wave 1 (parallel, disjoint files): Task 1 sampling | Task 2 degradation | Task 3 atlas | Task 4 predictions | Task 5 probes-debug
Wave 2 (parallel, after wave 1):   Task 6 whitebox integration (needs 1,3) | Task 7 fuzz runner (needs 1,2)
Wave 3 (single):                   Task 8 slurm + replication plot + sync + ensemble wrapper + full-suite gate
```

---

### Task 1: Shared batch sampler — `src/pri2m/xai/sampling.py`

**Files:**
- Create: `src/pri2m/xai/sampling.py`
- Modify: `scripts/whitebox_phase3.py` (remove moved defs; re-export for compat)
- Test: `tests/test_xai_sampling.py` (new); `tests/test_whitebox_phase3.py`, `tests/test_conformal_phase3.py`, `tests/test_concept_probes_phase3.py` (must stay green — they reach `build_labeled_batch` via `whitebox_phase3`)

**Interfaces:**
- Consumes: `Phase3SequenceDataset`, `collate_windows`, `WindowConfig`, `NormStats` from `pri2m.data.phase3_sequence`.
- Produces: `pri2m.xai.sampling.CHANNELS` (tuple, `("xtpos","ytpos","Forcex","Forcey")`), `pri2m.xai.sampling.npz_exists(npz_root: Path, row) -> bool`, `pri2m.xai.sampling.build_labeled_batch(table_csv, npz_root, preprocessing, n_windows, cfg, supervised_only=True, seed=0) -> (batch, labels)` — byte-identical behavior to the current whitebox versions. `scripts/whitebox_phase3.py` keeps module attributes `CHANNELS`, `build_labeled_batch`, `_npz_exists` as aliases so existing imports/tests don't move.

- [x] **Step 1: Write the failing test**

```python
# tests/test_xai_sampling.py
"""build_labeled_batch lives in pri2m.xai.sampling; whitebox re-exports it."""
import numpy as np

from pri2m.data.phase3_sequence import WindowConfig
from tests._phase3_synth import make_synth_npz_tree, make_synth_table, CHANNELS as SYNTH_CHANNELS


def _tree_and_table(tmp_path):
    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2), n_files=4, length=256, seed=0)
    table = make_synth_table(npz_root)
    csv = tmp_path / "table.csv"
    table.to_csv(csv, index=False)
    return npz_root, csv


def test_build_labeled_batch_moved_and_seeded(tmp_path):
    from pri2m.xai import sampling

    npz_root, csv = _tree_and_table(tmp_path)
    cfg = WindowConfig(length=128, stride=64, channels=sampling.CHANNELS)
    b1, l1 = sampling.build_labeled_batch(csv, npz_root, None, 6, cfg, seed=7)
    b2, l2 = sampling.build_labeled_batch(csv, npz_root, None, 6, cfg, seed=7)
    assert b1["window"].shape[0] == len(l1) and set(l1) == {0, 1}
    assert np.array_equal(l1, l2)
    assert (b1["window"] == b2["window"]).all()
    assert sampling.CHANNELS == SYNTH_CHANNELS


def test_whitebox_reexports_sampling():
    import whitebox_phase3 as W
    from pri2m.xai import sampling

    assert W.build_labeled_batch is sampling.build_labeled_batch
    assert W._npz_exists is sampling.npz_exists
    assert W.CHANNELS == sampling.CHANNELS
```

- [x] **Step 2: Run it — expect FAIL** (`ModuleNotFoundError`/`ImportError: sampling`)

Run: `python -m pytest tests/test_xai_sampling.py -v`

- [x] **Step 3: Implement.** Create `src/pri2m/xai/sampling.py`: move `_npz_exists` (rename `npz_exists`) and `build_labeled_batch` VERBATIM from `scripts/whitebox_phase3.py:70-135`, plus `CHANNELS = ("xtpos","ytpos","Forcex","Forcey")` and the imports the moved code needs (`pathlib.Path`, `numpy`, `pandas`, `NormStats/Phase3SequenceDataset/collate_windows` from `pri2m.data.phase3_sequence`). Module docstring: "Seeded, class-balanced window sampling shared by the eval-only XAI runners." In `scripts/whitebox_phase3.py`: delete the moved bodies and add
  ```python
  from pri2m.xai.sampling import CHANNELS, build_labeled_batch, npz_exists as _npz_exists
  ```
  (keep the names in the module namespace — other runners and tests import them from here).

- [x] **Step 4: Run the new test + every consumer suite — expect PASS, no regressions**

Run: `python -m pytest tests/test_xai_sampling.py tests/test_whitebox_phase3.py tests/test_conformal_phase3.py tests/test_concept_probes_phase3.py -q`
(If the conformal/concept-probe test files have different names, find them with `grep -rl "build_labeled_batch" tests/` and run those.)

---

### Task 2: Degradation module — `src/pri2m/xai/degradation.py`

**Files:**
- Create: `src/pri2m/xai/degradation.py`
- Test: `tests/test_degradation.py`

**Interfaces:**
- Consumes: `physics.delay_ablation`, `physics._auc`, `activations.collect_activations`, `physics.delayed_weight_share`, `kan_native._iter_kanlinear`, `KANLinear`, `TKANModel`.
- Produces (used by Task 7's runner):
  - `FUZZERS: dict[str, Fuzzer]` with keys `input_noise, input_smooth, tau_jitter, spline_coarsen, weight_noise`; `Fuzzer` dataclass fields `name: str, kind: str ("input"|"model"), levels: tuple, stochastic: bool, apply: Callable[(model, batch, level, seed)] -> (model, batch, info: dict)`.
  - `measure_probes(model, batch, labels) -> dict` keys `auc, auc_delay_off, delay_drop, mae, mae_delay_off`.
  - `xdelay_share(model, batch) -> float | None` (None when the model has no `x_delay` block).
  - `run_fuzzer(model, batch, labels, fuzzer, seeds=(0,1,2), with_share=None) -> dict` → `{"name","kind","stochastic","levels":[...], "results":[{"level","seeds":[{...probes,"seed"}],"mean":{...},"sd":{...},"info":{...}}], "breakdown":{...}}`.
  - `breakdown_points(levels, mean_auc, mean_drop) -> {"auc_below_075": level|None, "drop_below_005": level|None, "auc_at_chance": level|None}` with module constants `AUC_DEGRADED=0.75, DROP_FLOOR=0.05, AUC_CHANCE=0.55`.
  - `plot_degradation_panel(fuzzer_result: dict, baseline: dict, path) -> str`, `plot_fuzz_dashboard(all_results: list[dict], baseline: dict, path) -> str`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_degradation.py
"""Graded fuzziness suite: input/model fuzzers + physics-reliance probes."""
import copy

import numpy as np
import pytest
import torch

from pri2m.models.kan import KANLinear
from pri2m.models.tkan_model import TKANModel
from pri2m.xai import degradation as dg


def _model(seed=0):
    torch.manual_seed(seed)
    return TKANModel(input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3).eval()


def _batch(B=8, L=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(L, dtype=torch.float32)
    # deterministic multi-tone signal + small noise so smoothing has HF energy to remove
    base = torch.sin(2 * np.pi * 0.45 * t) + 0.5 * torch.sin(2 * np.pi * 0.05 * t)
    window = base.view(1, L, 1).repeat(B, 1, 4) + 0.1 * torch.randn(B, L, 4, generator=g)
    labels = np.array([0, 1] * (B // 2))
    return {
        "window": window,
        "tau": torch.full((B,), 5, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.float32),
        "b_over_blim": torch.rand(B, generator=g),
        "omega": torch.full((B,), 5000.0),
        "tpf": torch.full((B,), 200.0),
        "fs": torch.full((B,), 10_000.0),
    }, labels


# ---- input-side fuzzers -------------------------------------------------- #

def test_input_noise_level0_identity_and_seeded():
    batch, _ = _batch()
    out0 = dg.fuzz_input_noise(batch, 0.0, seed=0)
    assert torch.equal(out0["window"], batch["window"])
    a = dg.fuzz_input_noise(batch, 0.5, seed=1)["window"]
    b = dg.fuzz_input_noise(batch, 0.5, seed=1)["window"]
    c = dg.fuzz_input_noise(batch, 0.5, seed=2)["window"]
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert not torch.equal(a, batch["window"])          # actually fuzzed
    assert torch.equal(batch["window"], _batch()[0]["window"])  # input not mutated


def test_input_smooth_removes_hf_energy_monotonically():
    batch, _ = _batch()

    def hf_energy(w):
        spec = torch.fft.rfft(w - w.mean(dim=1, keepdim=True), dim=1).abs()
        return float(spec[:, spec.shape[1] // 2 :, :].sum())

    e = [hf_energy(dg.fuzz_input_smooth(batch, w)["window"]) for w in (1, 4, 16)]
    assert torch.equal(dg.fuzz_input_smooth(batch, 1)["window"], batch["window"])
    assert e[0] > e[1] > e[2]
    assert dg.fuzz_input_smooth(batch, 4)["window"].shape == batch["window"].shape


def test_tau_jitter_identity_floor_and_realized():
    batch, _ = _batch()
    out0, info0 = dg.fuzz_tau_jitter(batch, 0.0, seed=0)
    assert torch.equal(out0["tau"], batch["tau"]) and info0["realized_abs_dtau_mean"] == 0.0
    out, info = dg.fuzz_tau_jitter(batch, 0.5, seed=0)
    assert (out["tau"] >= 1).all()
    assert info["realized_abs_dtau_mean"] > 0
    assert not torch.equal(out["tau"], batch["tau"])
    signs = (out["tau"] - batch["tau"]).sign()
    assert (signs == 1).any() and (signs == -1).any()   # mixed directions


# ---- model-side fuzzers -------------------------------------------------- #

def test_spline_coarsen_selffit_and_forward():
    model, (batch, _) = _model(), _batch()
    same, _, info5 = dg.FUZZERS["spline_coarsen"].apply(model, batch, 5, 0)
    for r2 in info5["refit_r2"].values():
        assert r2 > 0.999                               # same-grid refit ≈ exact
    with torch.no_grad():
        y0, y5 = model(batch)["chatter_logit"], same(batch)["chatter_logit"]
    assert torch.allclose(y0, y5, atol=1e-3)
    coarse, _, info2 = dg.FUZZERS["spline_coarsen"].apply(model, batch, 2, 0)
    for _, kan in __import__("pri2m.xai.kan_native", fromlist=["_iter_kanlinear"])._iter_kanlinear(coarse):
        assert kan.n_coeff == 2 + kan.spline_order
    with torch.no_grad():
        coarse(batch)                                   # coarse model still runs
    # original untouched
    for (_, a), (_, b) in zip(
        __import__("pri2m.xai.kan_native", fromlist=["_iter_kanlinear"])._iter_kanlinear(model),
        __import__("pri2m.xai.kan_native", fromlist=["_iter_kanlinear"])._iter_kanlinear(_model()),
    ):
        assert torch.equal(a.spline_weight, b.spline_weight)


def test_weight_noise_identity_and_seeded():
    model, (batch, _) = _model(), _batch()
    m0, _, _ = dg.FUZZERS["weight_noise"].apply(model, batch, 0.0, 0)
    with torch.no_grad():
        assert torch.allclose(model(batch)["chatter_logit"], m0(batch)["chatter_logit"])
    m1, _, _ = dg.FUZZERS["weight_noise"].apply(model, batch, 0.2, 1)
    m1b, _, _ = dg.FUZZERS["weight_noise"].apply(model, batch, 0.2, 1)
    with torch.no_grad():
        assert torch.allclose(m1(batch)["chatter_logit"], m1b(batch)["chatter_logit"])
        assert not torch.allclose(model(batch)["chatter_logit"], m1(batch)["chatter_logit"])


# ---- probes / loop / breakdown ------------------------------------------ #

def test_measure_probes_keys_and_level0_selfcheck():
    model, (batch, labels) = _model(), _batch()
    base = dg.measure_probes(model, batch, labels)
    for k in ("auc", "auc_delay_off", "delay_drop", "mae", "mae_delay_off"):
        assert np.isfinite(base[k])
    res = dg.run_fuzzer(model, batch, labels, dg.FUZZERS["input_noise"], seeds=(0,))
    lvl0 = res["results"][0]
    assert lvl0["level"] == 0.0
    assert abs(lvl0["mean"]["auc"] - base["auc"]) < 1e-6
    assert abs(lvl0["mean"]["delay_drop"] - base["delay_drop"]) < 1e-6
    assert set(res["breakdown"]) == {"auc_below_075", "drop_below_005", "auc_at_chance"}


def test_run_fuzzer_stochastic_mean_sd():
    model, (batch, labels) = _model(), _batch()
    res = dg.run_fuzzer(model, batch, labels, dg.FUZZERS["weight_noise"], seeds=(0, 1, 2))
    big = [r for r in res["results"] if r["level"] == 0.5][0]
    assert len(big["seeds"]) == 3
    assert big["sd"]["auc"] >= 0.0


def test_breakdown_points_on_crafted_curve():
    levels = (0.0, 0.1, 0.2, 0.5)
    bp = dg.breakdown_points(levels, mean_auc=(0.95, 0.80, 0.70, 0.50), mean_drop=(0.3, 0.04, 0.02, 0.0))
    assert bp == {"auc_below_075": 0.2, "drop_below_005": 0.1, "auc_at_chance": 0.5}
    none = dg.breakdown_points(levels, (0.9, 0.9, 0.9, 0.9), (0.3, 0.3, 0.3, 0.3))
    assert none == {"auc_below_075": None, "drop_below_005": None, "auc_at_chance": None}


def test_plots_write_files(tmp_path):
    model, (batch, labels) = _model(), _batch()
    res = dg.run_fuzzer(model, batch, labels, dg.FUZZERS["tau_jitter"], seeds=(0,))
    base = dg.measure_probes(model, batch, labels)
    p1 = dg.plot_degradation_panel(res, base, tmp_path / "panel.png")
    p2 = dg.plot_fuzz_dashboard([res], base, tmp_path / "dash.png")
    assert (tmp_path / "panel.png").is_file() and (tmp_path / "dash.png").is_file()
    assert p1.endswith("panel.png") and p2.endswith("dash.png")
```

- [x] **Step 2: Run — expect FAIL** (`ImportError: degradation`)

Run: `python -m pytest tests/test_degradation.py -v`

- [x] **Step 3: Implement `src/pri2m/xai/degradation.py`.** Complete implementation:

```python
"""Graded fuzziness (degradation) suite for the Phase 3 TKAN — eval-only.

Input-side fuzzers blur what the model SEES (noise, smoothing, tau jitter);
model-side fuzzers blur the MODEL itself (spline coarsening, weight noise).
At every fuzz level the probes re-measure not just AUC but the delay-ablation
drop — the physics-reliance signature — so the resulting curves answer
"up to what fuzz level does the model still use the regenerative delay".

All fuzzers are non-mutating: input fuzzers return a new batch dict; model
fuzzers deep-copy. AUC is per-window over the sampled batch (the whitebox
convention), via physics.delay_ablation which measures on+off in one call.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from pri2m.models.kan import KANLinear
from pri2m.xai import physics
from pri2m.xai.kan_native import _iter_kanlinear

AUC_DEGRADED = 0.75   # halfway to chance from ~1.0
DROP_FLOOR = 0.05     # delay-drop below this = physics signature gone
AUC_CHANCE = 0.55     # ≈ chance


# --------------------------------------------------------------------- #
# Input-side fuzzers (batch -> new batch; never mutate the input)
# --------------------------------------------------------------------- #
def fuzz_input_noise(batch: dict, sigma: float, seed: int = 0) -> dict:
    """Additive Gaussian noise (z-scored SD units) on the signal channels."""
    new = dict(batch)
    if sigma <= 0:
        new["window"] = batch["window"].clone()
        return new
    g = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(batch["window"].shape, generator=g) * float(sigma)
    new["window"] = batch["window"] + noise.to(batch["window"].dtype)
    return new


def fuzz_input_smooth(batch: dict, width: int) -> dict:
    """Reflect-padded centered moving average of ``width`` RAW samples."""
    new = dict(batch)
    width = int(width)
    if width <= 1:
        new["window"] = batch["window"].clone()
        return new
    x = batch["window"].permute(0, 2, 1)                      # [B, C, L]
    left = width // 2
    right = width - 1 - left
    padded = F.pad(x, (left, right), mode="reflect")
    smoothed = F.avg_pool1d(padded, kernel_size=width, stride=1)
    new["window"] = smoothed.permute(0, 2, 1).contiguous()
    return new


def fuzz_tau_jitter(batch: dict, frac: float, seed: int = 0) -> tuple[dict, dict]:
    """Perturb each window's tau by ±frac (random sign per window, seeded)."""
    tau = np.asarray(batch["tau"].detach().cpu().numpy(), dtype=float)
    if frac <= 0:
        return dict(batch), {"realized_abs_dtau_mean": 0.0, "frac_pooled_tau_changed": 0.0}
    g = np.random.default_rng(int(seed))
    signs = g.choice([-1.0, 1.0], size=tau.shape)
    new_tau = np.maximum(1, np.round(tau * (1.0 + signs * float(frac)))).astype(np.int64)
    new = dict(batch)
    new["tau"] = torch.as_tensor(new_tau, dtype=torch.long)
    info = {
        "realized_abs_dtau_mean": float(np.mean(np.abs(new_tau - tau))),
        # honesty: after the model's own pooling, small jitters can round away
        "frac_pooled_tau_changed": float(
            np.mean(np.maximum(1, np.round(new_tau / 4)) != np.maximum(1, np.round(tau / 4)))
        ),
    }
    return new, info


# --------------------------------------------------------------------- #
# Model-side fuzzers (model -> deep-copied fuzzed model)
# --------------------------------------------------------------------- #
def _refit_kanlinear_inplace(kan: KANLinear, grid_size: int, n_points: int = 64) -> float:
    """Least-squares refit of kan's spline onto a ``grid_size`` grid, in place.

    Samples the ORIGINAL learned spline component right-open on grid_range
    (the degree-0 basis is zero at exactly the right endpoint), solves one
    batched lstsq against the coarse basis, swaps coefficients + knot buffer.
    Returns the refit R^2 (1.0 = the coarse grid can represent the original).
    """
    order = kan.spline_order
    lo, hi = kan.grid_range
    xs = torch.linspace(lo, hi, n_points + 1, dtype=torch.float32)[:-1]     # right-open
    basis_orig = kan.b_splines(xs.unsqueeze(-1))[:, 0, :]                    # [n, n_coeff_orig]
    target = torch.einsum("xc,oic->oix", basis_orig, kan.spline_weight)      # [out, in, n]
    t = target.reshape(-1, n_points).T.contiguous()                          # [n, out*in]

    tmp = KANLinear(1, 1, grid_size=grid_size, spline_order=order, grid_range=(lo, hi))
    B = tmp.b_splines(xs.unsqueeze(-1))[:, 0, :]                             # [n, k+order]
    coeff = torch.linalg.lstsq(B, t).solution                                # [k+order, out*in]
    fitted = B @ coeff
    ss_res = ((t - fitted) ** 2).sum()
    ss_tot = ((t - t.mean(dim=0)) ** 2).sum().clamp_min(1e-12)
    r2 = float(1.0 - ss_res / ss_tot)

    kan.grid_size = int(grid_size)
    kan.n_coeff = int(grid_size) + order
    h = (hi - lo) / grid_size
    knots = torch.arange(-order, grid_size + order + 1, dtype=torch.float32) * h + lo
    kan._buffers["grid"] = knots.to(kan.grid.device)
    kan.spline_weight = torch.nn.Parameter(
        coeff.T.reshape(kan.out_features, kan.in_features, kan.n_coeff).contiguous()
    )
    return r2


def fuzz_spline_coarsen(model, grid_size: int) -> tuple[torch.nn.Module, dict]:
    fuzzed = copy.deepcopy(model)
    r2s = {}
    for name, kan in _iter_kanlinear(fuzzed):
        r2s[name or "kan"] = _refit_kanlinear_inplace(kan, int(grid_size))
    return fuzzed, {"refit_r2": r2s}


def fuzz_weight_noise(model, sigma_rel: float, seed: int = 0):
    fuzzed = copy.deepcopy(model)
    if sigma_rel <= 0:
        return fuzzed, {}
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for _, kan in _iter_kanlinear(fuzzed):
            for p in (kan.base_weight, kan.spline_weight):
                eps = torch.randn(p.shape, generator=g) * float(sigma_rel)
                p.mul_(1.0 + eps)
    return fuzzed, {}


# --------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------- #
def measure_probes(model, batch: dict, labels) -> dict:
    """AUC / severity MAE / delay-reliance, in one delay_ablation call."""
    abl = physics.delay_ablation(model, batch, labels)
    return {
        "auc": float(abl["auc_on"]),
        "auc_delay_off": float(abl["auc_off"]),
        "delay_drop": float(abl["auc_drop"]),
        "mae": float(abl["mae_on"]) if "mae_on" in abl else None,
        "mae_delay_off": float(abl["mae_off"]) if "mae_off" in abl else None,
    }


def xdelay_share(model, batch: dict):
    """x_delay data-aware weight share (None if the model has no delay block)."""
    from pri2m.xai import activations as act_mod

    stats = act_mod.collect_activations(model, batch)
    share = physics.delayed_weight_share(model, stats)
    overall = share.get("overall", {})
    return float(overall["x_delay"]) if "x_delay" in overall else None


# --------------------------------------------------------------------- #
# Registry + sweep loop
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fuzzer:
    name: str
    kind: str          # "input" | "model"
    levels: tuple
    stochastic: bool
    apply: Callable    # (model, batch, level, seed) -> (model, batch, info)


def _apply_input_noise(model, batch, level, seed):
    return model, fuzz_input_noise(batch, level, seed), {}


def _apply_input_smooth(model, batch, level, seed):
    return model, fuzz_input_smooth(batch, level), {}


def _apply_tau_jitter(model, batch, level, seed):
    new, info = fuzz_tau_jitter(batch, level, seed)
    return model, new, info


def _apply_spline_coarsen(model, batch, level, seed):
    fuzzed, info = fuzz_spline_coarsen(model, level)
    return fuzzed, batch, info


def _apply_weight_noise(model, batch, level, seed):
    fuzzed, info = fuzz_weight_noise(model, level, seed)
    return fuzzed, batch, info


FUZZERS: dict[str, Fuzzer] = {
    "input_noise": Fuzzer("input_noise", "input", (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0), True, _apply_input_noise),
    "input_smooth": Fuzzer("input_smooth", "input", (1, 2, 4, 8, 16, 32), False, _apply_input_smooth),
    "tau_jitter": Fuzzer("tau_jitter", "input", (0.0, 0.05, 0.10, 0.25, 0.50), True, _apply_tau_jitter),
    "spline_coarsen": Fuzzer("spline_coarsen", "model", (5, 4, 3, 2, 1), False, _apply_spline_coarsen),
    "weight_noise": Fuzzer("weight_noise", "model", (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5), True, _apply_weight_noise),
}

_PROBE_KEYS = ("auc", "auc_delay_off", "delay_drop", "mae", "mae_delay_off")


def breakdown_points(levels, mean_auc, mean_drop) -> dict:
    """First level crossing each degradation threshold (None if never)."""
    def first(seq, pred):
        for lvl, v in zip(levels, seq):
            if v is not None and pred(v):
                return lvl
        return None

    return {
        "auc_below_075": first(mean_auc, lambda a: a < AUC_DEGRADED),
        "drop_below_005": first(mean_drop, lambda d: d < DROP_FLOOR),
        "auc_at_chance": first(mean_auc, lambda a: a <= AUC_CHANCE),
    }


def run_fuzzer(model, batch, labels, fuzzer: Fuzzer, seeds=(0, 1, 2), with_share=None) -> dict:
    """Sweep one fuzzer over its levels; stochastic fuzzers get one probe per seed.

    with_share: None -> record xdelay_share only for model-side fuzzers.
    """
    if with_share is None:
        with_share = fuzzer.kind == "model"
    seeds = tuple(seeds) if fuzzer.stochastic else (0,)
    out = []
    for level in fuzzer.levels:
        per_seed, info = [], {}
        for seed in seeds:
            fm, fb, info = fuzzer.apply(model, batch, level, seed)
            probes = measure_probes(fm, fb, labels)
            if with_share:
                probes["xdelay_share"] = xdelay_share(fm, fb)
            probes["seed"] = int(seed)
            per_seed.append(probes)
        keys = [k for k in per_seed[0] if k != "seed"]
        mean = {k: (float(np.mean([p[k] for p in per_seed])) if per_seed[0][k] is not None else None) for k in keys}
        sd = {k: (float(np.std([p[k] for p in per_seed])) if per_seed[0][k] is not None else None) for k in keys}
        out.append({"level": level, "seeds": per_seed, "mean": mean, "sd": sd, "info": info})
    return {
        "name": fuzzer.name,
        "kind": fuzzer.kind,
        "stochastic": fuzzer.stochastic,
        "levels": list(fuzzer.levels),
        "results": out,
        "breakdown": breakdown_points(
            fuzzer.levels,
            [r["mean"]["auc"] for r in out],
            [r["mean"]["delay_drop"] for r in out],
        ),
    }


# --------------------------------------------------------------------- #
# Plots (values only ever come from run_fuzzer results)
# --------------------------------------------------------------------- #
def _panel(ax, result: dict, baseline: dict) -> None:
    levels = result["levels"]
    xs = np.arange(len(levels))
    auc = [r["mean"]["auc"] for r in result["results"]]
    sd = [r["sd"]["auc"] for r in result["results"]]
    drop = [r["mean"]["delay_drop"] for r in result["results"]]
    ax.plot(xs, auc, "o-", color="tab:blue", label="AUC")
    ax.fill_between(xs, np.array(auc) - np.array(sd), np.array(auc) + np.array(sd),
                    color="tab:blue", alpha=0.2)
    ax.axhline(baseline["auc"], color="tab:blue", ls=":", lw=1, label="baseline AUC")
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(xs, [str(l) for l in levels])
    ax.set_xlabel(f"{result['name']} level")
    ax.set_ylabel("per-window AUC")
    twin = ax.twinx()
    twin.plot(xs, drop, "s--", color="tab:red", label="delay-ablation drop")
    twin.axhline(DROP_FLOOR, color="tab:red", ls=":", lw=1)
    twin.set_ylabel("delay-ablation AUC drop", color="tab:red")
    twin.set_ylim(bottom=min(0.0, min(drop)))
    ax.set_title(result["name"])


def plot_degradation_panel(result: dict, baseline: dict, path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(5.5, 4))
    _panel(ax, result, baseline)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_fuzz_dashboard(all_results: list[dict], baseline: dict, path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    n = max(1, len(all_results))
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    for ax, res in zip(flat[:n], all_results):
        _panel(ax, res, baseline)
    fig.suptitle("Physics understanding vs fuzz level (per-window AUC + delay-ablation drop)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
```

- [x] **Step 4: Run — expect PASS**

Run: `python -m pytest tests/test_degradation.py -v`
Note: if `test_measure_probes_keys_and_level0_selfcheck` shows AUC exactly 0.5/1.0 on the untrained synth model, that is fine — the assertions only require finiteness and level-0 equality.

- [x] **Step 5: Quick neighbors check** — `python -m pytest tests/test_xai_physics.py tests/test_xai_activations.py -q` (names via `ls tests/ | grep -i physics`) to prove no shared-state damage.

---

### Task 3: Spline-edge atlas — `src/pri2m/xai/spline_atlas.py`

**Files:**
- Create: `src/pri2m/xai/spline_atlas.py`
- Modify: `src/pri2m/xai/symbolic.py` (add `eval_fit`)
- Test: `tests/test_spline_atlas.py`; existing `tests/test_xai_symbolic.py` must stay green.

**Interfaces:**
- Consumes: `activations.collect_activations` stats, `semantics.edge_index_map`, `faithfulness.rank_edges`, `symbolic.fit_edge`, `kan_native._iter_kanlinear`, `KANLinear.edge_function`.
- Produces:
  - `symbolic.eval_fit(fit: dict, xs: np.ndarray) -> np.ndarray` — evaluate a `fit_curve`/`fit_edge` result (via its `name` + `params`) on `xs`; raises `ValueError` on unknown/None name. Implement by dispatching over the SAME candidate family definitions `fit_curve` already uses (reuse the module's internal candidate registry — do not duplicate formulas).
  - `spline_atlas.top_edge_entries(stats, sem_map, k=16) -> list[dict]` — each `{"kan","out","in","feature","block","importance","is_delay","is_side"}`; edges whose `kan` has no entry in `sem_map` fall back to `feature=f"in{i}", block="input"`.
  - `spline_atlas.plot_spline_atlas(model, stats, sem_map, path, k=16, r2_threshold=0.9, n=200) -> list[dict]` — writes the small-multiples figure AND returns the entries enriched with `{"formula","r2","accepted"}` (manifest rows).
  - `spline_atlas.feature_importance_bars(stats, sem_map) -> dict` — `{kan_name: {feature_label: summed_importance}}` with every `h_prev`/`r*h_prev`/`hidden` input collapsed into one `"h_prev (sum)"` bucket per layer.
  - `spline_atlas.plot_feature_importance_bars(agg: dict, path) -> str`.
  - `spline_atlas.plot_feature_named_heatmap(stats, sem_map, path) -> str` — per-KANLinear heatmaps like the whitebox `_plot_importance_heatmaps`, plus named x-ticks for non-hidden features and dashed vlines at block boundaries.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_spline_atlas.py
"""Spline-edge atlas: named, data-grounded, symbolically-annotated edge panels."""
import numpy as np
import torch

from pri2m.models.tkan_model import TKANModel
from pri2m.xai import activations as act_mod
from pri2m.xai import semantics, spline_atlas, symbolic


def _model_and_batch(B=6, L=32):
    torch.manual_seed(0)
    model = TKANModel(
        input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3,
        side_input=True, side_features=("omega", "tpf", "tau"), side_hidden=4,
    ).eval()
    g = torch.Generator().manual_seed(1)
    batch = {
        "window": torch.randn(B, L, 4, generator=g),
        "tau": torch.full((B,), 3, dtype=torch.long),
        "omega": torch.full((B,), 5000.0),
        "tpf": torch.full((B,), 200.0),
        "b_over_blim": torch.rand(B, generator=g),
        "label": torch.tensor([0.0, 1.0] * (B // 2)),
    }
    stats = act_mod.collect_activations(model, batch)
    sem = semantics.edge_index_map(model)
    return model, batch, stats, sem


def test_eval_fit_linear_roundtrip():
    xs = np.linspace(-1, 1, 50)
    fit = symbolic.fit_curve(xs, 2.0 * xs + 0.5)
    ys = symbolic.eval_fit(fit, xs)
    assert np.allclose(ys, 2.0 * xs + 0.5, atol=1e-6)


def test_top_edge_entries_named_and_flagged():
    model, batch, stats, sem = _model_and_batch()
    entries = spline_atlas.top_edge_entries(stats, sem, k=12)
    assert len(entries) == 12
    assert entries == sorted(entries, key=lambda e: -e["importance"])
    for e in entries:
        named = {d["index"]: d for d in sem[e["kan"]]}
        assert e["feature"] == named[e["in"]]["name"]
        assert e["is_delay"] == (named[e["in"]]["block"] == "x_delay")
        assert e["is_side"] == (named[e["in"]]["block"] == "side")


def test_plot_spline_atlas_writes_and_annotates(tmp_path):
    model, batch, stats, sem = _model_and_batch()
    rows = spline_atlas.plot_spline_atlas(model, stats, sem, tmp_path / "atlas.png", k=6)
    assert (tmp_path / "atlas.png").is_file()
    assert len(rows) == 6
    for r in rows:
        assert {"kan", "out", "in", "feature", "block", "importance", "formula", "r2", "accepted"} <= set(r)


def test_feature_bars_collapse_hidden(tmp_path):
    model, batch, stats, sem = _model_and_batch()
    agg = spline_atlas.feature_importance_bars(stats, sem)
    gate = agg["encoder.layers.0.kan_z"]
    assert "h_prev (sum)" in gate
    assert any(k.startswith("delay:") for k in gate)
    assert not any(k.startswith("h0") for k in gate)          # individual h's collapsed
    p = spline_atlas.plot_feature_importance_bars(agg, tmp_path / "bars.png")
    assert (tmp_path / "bars.png").is_file()


def test_feature_named_heatmap_writes(tmp_path):
    model, batch, stats, sem = _model_and_batch()
    p = spline_atlas.plot_feature_named_heatmap(stats, sem, tmp_path / "heat.png")
    assert (tmp_path / "heat.png").is_file()
```

- [x] **Step 2: Run — expect FAIL** (`ImportError: spline_atlas` / `eval_fit`)

Run: `python -m pytest tests/test_spline_atlas.py -v`

- [x] **Step 3: Implement.**
  - In `symbolic.py`: locate the candidate-family registry `fit_curve` uses (grep `families`/`candidates` in the file) and add a public `eval_fit(fit, xs)` that reconstructs the chosen family's callable with `fit["params"]` and evaluates it; `ValueError` if `fit.get("name")` is None/unknown.
  - `spline_atlas.py` core:

```python
"""Spline-edge atlas: the learned edge functions, named, ranked and annotated."""
from __future__ import annotations

import numpy as np

from pri2m.xai import symbolic
from pri2m.xai.faithfulness import rank_edges
from pri2m.xai.kan_native import _iter_kanlinear

_HIDDEN_BLOCKS = {"h_prev", "r*h_prev", "hidden"}


def top_edge_entries(stats: dict, sem_map: dict, k: int = 16) -> list[dict]:
    entries = []
    for kan_name, o, i, imp in rank_edges(stats)[: int(k)]:
        named = {d["index"]: d for d in sem_map.get(kan_name, [])}
        d = named.get(i, {"block": "input", "name": f"in{i}"})
        entries.append({
            "kan": kan_name, "out": int(o), "in": int(i),
            "feature": d["name"], "block": d["block"],
            "importance": float(imp),
            "is_delay": d["block"] == "x_delay",
            "is_side": d["block"] == "side",
        })
    return entries


def plot_spline_atlas(model, stats, sem_map, path, k=16, r2_threshold=0.9, n=200) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    kans = dict(_iter_kanlinear(model))
    entries = top_edge_entries(stats, sem_map, k)
    ncols = 4
    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(entries):]:
        ax.axis("off")
    for ax, e in zip(flat, entries):
        kan = kans[e["kan"]]
        lo = float(stats[e["kan"]]["visited_lo"][e["in"]])
        hi = float(stats[e["kan"]]["visited_hi"][e["in"]])
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            lo, hi = kan.grid_range
        curves = kan.edge_function(n)
        xs_full, ys_full = curves["x"], curves["curves"][e["out"], e["in"]]
        ax.plot(xs_full, ys_full, color="tab:blue", lw=1.5)
        ax.axvspan(lo, hi, color="tab:blue", alpha=0.08)       # data-visited band
        inputs = np.asarray(stats[e["kan"]]["inputs"])[:, e["in"]]
        rug = inputs[:: max(1, len(inputs) // 256)]
        ax.plot(rug, np.full_like(rug, ax.get_ylim()[0]), "|", color="k", ms=4, alpha=0.35)
        fit = symbolic.fit_edge(kan, e["out"], e["in"], lo, hi, r2_threshold=r2_threshold)
        e.update({"formula": fit.get("formula"), "r2": fit.get("r2"), "accepted": bool(fit.get("accepted", False))})
        if e["accepted"]:
            xs_fit = np.linspace(lo, hi, n)
            ax.plot(xs_fit, symbolic.eval_fit(fit, xs_fit), "--", color="tab:orange", lw=1.2)
            ax.set_xlabel(f"{fit['formula']}  (R²={fit['r2']:.3f})", fontsize=6)
        color = "tab:red" if e["is_delay"] else ("tab:green" if e["is_side"] else "k")
        gate = e["kan"].split(".")[-1]
        ax.set_title(f"{gate}: {e['feature']} → u{e['out']}  (imp {e['importance']:.3g})",
                     fontsize=7, color=color)
        ax.tick_params(labelsize=6)
    fig.suptitle("Top learned edge functions φ(x) — red=τ-delayed, green=side input", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return entries
```

  - `feature_importance_bars`: iterate `stats`; for each kan present in `sem_map`, sum `importance[:, i]` over out-units per input `i`; bucket by `sem_map` name, collapsing `_HIDDEN_BLOCKS` into `"h_prev (sum)"`. `plot_feature_importance_bars`: one horizontal-bar subplot per kan (delay bars red, side green, current blue). `plot_feature_named_heatmap`: copy the whitebox `_plot_importance_heatmaps` layout; add named x-ticks for indices whose block not in `_HIDDEN_BLOCKS` (rotate 90, fontsize 5) and `ax.axvline` at each block boundary (where consecutive entries' `block` differs).

- [x] **Step 4: Run — expect PASS**

Run: `python -m pytest tests/test_spline_atlas.py tests/test_xai_symbolic.py -q`

---

### Task 4: Per-file predictions dump — `scripts/predict_windows_phase3.py`

**Files:**
- Create: `scripts/predict_windows_phase3.py`
- Test: `tests/test_predict_windows_phase3.py`

**Interfaces:**
- Consumes: `scripts.xai_phase3._find_preprocessing/_load_model`, `scripts.eval_heldout_phase3.build_eval_table/_window_config_from_preprocessing`, `Phase3SequenceDataset`, `collate_windows`, `NormStats`.
- Produces: CLI `python scripts/predict_windows_phase3.py --checkpoint X --table Y --npz-root Z --out preds.csv [--preprocessing P] [--dataset all|1|1,2] [--batch-size 64] [--device cpu]` writing a CSV with columns exactly `file,prob,n_windows,dataset,label` — `prob` = mean `sigmoid(chatter_logit)` over the file's windows (the format `counterfactuals_phase3.py --predictions` consumes: merges on `file`, auto-picks `prob`).

- [x] **Step 1: Write the failing test**

```python
# tests/test_predict_windows_phase3.py
"""predict_windows_phase3: per-file network probabilities for counterfactuals."""
import json

import numpy as np
import pandas as pd
import torch

from pri2m.models.tkan_model import TKANModel
from tests._phase3_synth import make_synth_npz_tree, make_synth_table


def _checkpoint(tmp_path):
    torch.manual_seed(0)
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3)
    ckpt_dir = tmp_path / "runs" / "synthrun" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "best_auc.pt"
    torch.save(model.state_dict(), ckpt)
    (ckpt_dir.parent / "preprocessing.json").write_text(json.dumps({
        "norm": {"mean": [0.0] * 4, "std": [1.0] * 4},
        "window": {"length": 128, "stride": 64},
        "pool": 1,
    }))
    return ckpt


def test_cli_writes_per_file_probs(tmp_path):
    import predict_windows_phase3 as P

    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2), n_files=3, length=256, seed=0)
    table = make_synth_table(npz_root)
    csv = tmp_path / "table.csv"
    table.to_csv(csv, index=False)
    ckpt = _checkpoint(tmp_path)
    out = tmp_path / "preds.csv"
    rc = P.main([
        "--checkpoint", str(ckpt), "--table", str(csv), "--npz-root", str(npz_root),
        "--out", str(out), "--batch-size", "8",
    ])
    assert rc == 0 and out.is_file()
    preds = pd.read_csv(out)
    assert list(preds.columns) == ["file", "prob", "n_windows", "dataset", "label"]
    assert preds["prob"].between(0, 1).all()
    assert (preds["n_windows"] >= 1).all()
    assert preds["file"].isin(table["file"]).all()          # merge key intact
    assert len(preds) == preds["file"].nunique()

    rc2 = P.main([
        "--checkpoint", str(ckpt), "--table", str(csv), "--npz-root", str(npz_root),
        "--out", str(tmp_path / "p2.csv"), "--batch-size", "8",
    ])
    assert np.allclose(pd.read_csv(tmp_path / "p2.csv")["prob"], preds["prob"])  # deterministic


def test_predictions_merge_into_counterfactual_surrogate(tmp_path):
    import counterfactuals_phase3 as CF
    import predict_windows_phase3 as P

    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1,), with_mirrors=False, n_files=4, length=256)
    table = make_synth_table(npz_root)
    csv = tmp_path / "table.csv"
    table.to_csv(csv, index=False)
    out = tmp_path / "preds.csv"
    P.main(["--checkpoint", str(_checkpoint(tmp_path)), "--table", str(csv),
            "--npz-root", str(npz_root), "--out", str(out)])
    predict, info = CF.build_surrogate(table, predictions=str(out), seed=0)
    assert info["source"] == "model_predictions" and info["target_col"] == "prob"
    p = predict(np.array([[float(table["omega"].iloc[0]), float(table["b_mm"].iloc[0])]]))
    assert 0.0 <= float(p[0]) <= 1.0
```

(Adapt the `build_surrogate` call signature to the real one at `scripts/counterfactuals_phase3.py:97` — check its exact parameters before writing; if it takes an args-style namespace, construct that instead. The assertion targets — `source == "model_predictions"`, auto-picked `target_col` — are fixed.)

- [x] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: predict_windows_phase3`)

Run: `python -m pytest tests/test_predict_windows_phase3.py -v`

- [x] **Step 3: Implement `scripts/predict_windows_phase3.py`.**

```python
"""Dump per-FILE chatter probabilities from a trained checkpoint (eval-only).

Output CSV (columns: file,prob,n_windows,dataset,label) is the --predictions
input for counterfactuals_phase3.py, replacing its table-label surrogate with
the network's own decisions. Mirrors eval_heldout_phase3's load path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main(argv=None) -> int:
    import json

    import pandas as pd
    import torch

    from pri2m.data.phase3_sequence import NormStats, Phase3SequenceDataset, collate_windows
    from torch.utils.data import DataLoader

    from eval_heldout_phase3 import _window_config_from_preprocessing, build_eval_table
    from xai_phase3 import _find_preprocessing, _load_model

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preprocessing", default=None)
    ap.add_argument("--dataset", default="all", help="'all', an int, or comma list")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--hidden-size", type=int, default=None)
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--spline-order", type=int, default=3)
    args = ap.parse_args(argv)

    pre_path = args.preprocessing or _find_preprocessing(args.checkpoint)
    if not pre_path:
        raise SystemExit("no preprocessing.json found; pass --preprocessing")
    pre = json.loads(Path(pre_path).read_text(encoding="utf-8"))

    model = _load_model(args.checkpoint, args, pre)
    model.pool = int(pre.get("pool", 1))
    model.eval()

    eval_table, _ = build_eval_table(args.table, args.npz_root)
    if args.dataset != "all":
        wanted = {int(x) for x in str(args.dataset).split(",")}
        eval_table = eval_table[eval_table["dataset"].astype(int).isin(wanted)].reset_index(drop=True)

    norm = NormStats.from_dict(pre["norm"])
    cfg = _window_config_from_preprocessing(pre)
    ds = Phase3SequenceDataset(eval_table, args.npz_root, cfg=cfg, norm=norm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_windows)

    sums, counts, meta = {}, {}, {}
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch)["chatter_logit"]).cpu().numpy()
            for f, d, lab, p in zip(batch["file"], batch["dataset"],
                                    batch["label"].cpu().numpy(), probs):
                sums[f] = sums.get(f, 0.0) + float(p)
                counts[f] = counts.get(f, 0) + 1
                meta[f] = (d, int(lab))

    rows = [{"file": f, "prob": sums[f] / counts[f], "n_windows": counts[f],
             "dataset": meta[f][0], "label": meta[f][1]} for f in sums]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["file", "prob", "n_windows", "dataset", "label"]).to_csv(out, index=False)
    print(f"wrote {len(rows)} per-file probabilities -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Before finalizing, check how `Phase3SequenceDataset`/`batch["dataset"]` and `build_eval_table` behave on the synth table (dataset entries may be ints or strings — normalize to `int` in the rows). Match `eval_heldout_phase3.py`'s import style for sibling-script imports (it does `from scripts.xai_phase3 import ...` or path-insertion — copy whichever it actually uses).

- [x] **Step 4: Run — expect PASS**

Run: `python -m pytest tests/test_predict_windows_phase3.py -v`

---

### Task 5: Concept-probe debug (systematic-debugging)

**Files:**
- Modify: `src/pri2m/xai/concept_probes.py` (only if the diagnosis demands it)
- Test: `tests/test_concept_probe_circularity.py` (new; keep existing concept-probe tests green)

**Interfaces:**
- Consumes: `concept_probes.build_concept_targets`, `fit_probe`, `run_concept_probes`, `regeneration_phase`, `_wrap_pi`.
- Produces: either (a) fixed probes — `regeneration_phase` probed as `(sin φ, cos φ)` pair (two concepts `regeneration_phase_sin`/`regeneration_phase_cos`, or one 2-target probe recording an angular R²), `chatter_tpf_separation` on `log1p` scale with `"transform": "log1p"` recorded in the manifest — or (b) a written diagnosis in the task report that the concept is unprobeable as defined, WITH the failing mechanism demonstrated by a test. No silent drop.

**Working hypothesis to test first (from the code contracts):** `regeneration_phase` returns a WRAPPED phase in (−π, π] (`_wrap_pi(2π·k·τ/L)`); a linear Ridge probe on a circular target is degenerate by construction — two windows with nearly identical dynamics can have targets −π and +π, so held-out R² collapses (matches the observed −228). Secondary suspects: `k = _dominant_bin(x)` computed on the RAW window while hidden states come from the ×4-pooled encoder (frequency aliasing above fs/8), and the heavy-tailed `chatter_tpf_separation` in Hz.

- [x] **Step 1: Write the failing reproduction test** — proves the circularity mechanism WITHOUT any model:

```python
# tests/test_concept_probe_circularity.py
"""Why regeneration_phase probes degenerate: circular target vs linear probe."""
import numpy as np

from pri2m.xai.concept_probes import _wrap_pi, fit_probe


def _hidden_encoding_phase(n=400, H=16, seed=0):
    """Hidden states that PERFECTLY encode a phase angle, as (sin, cos) dims."""
    rng = np.random.default_rng(seed)
    phase_unwrapped = rng.uniform(0, 6 * np.pi, size=n)      # crosses the wrap many times
    h = rng.normal(size=(n, H)) * 0.05
    h[:, 0] = np.sin(phase_unwrapped)
    h[:, 1] = np.cos(phase_unwrapped)
    return h, phase_unwrapped


def test_wrapped_phase_probe_is_degenerate_but_sincos_recovers():
    h, phase = _hidden_encoding_phase()
    wrapped = _wrap_pi(phase)
    bad = fit_probe(h, wrapped, kind="regression", seed=0)
    good_sin = fit_probe(h, np.sin(phase), kind="regression", seed=0)
    good_cos = fit_probe(h, np.cos(phase), kind="regression", seed=0)
    # the mechanism: sin/cos targets are near-perfectly linearly decodable,
    # the wrapped angle is not — this is the R^2 = -228 failure mode
    assert good_sin["score"] > 0.9 and good_cos["score"] > 0.9
    assert bad["score"] < 0.5


def test_run_concept_probes_emits_sincos_phase(synth_model_batch=None):
    # AFTER the fix: run_concept_probes must expose circular-safe phase probes.
    import torch
    from pri2m.models.tkan_model import TKANModel
    from pri2m.xai.concept_probes import run_concept_probes

    torch.manual_seed(0)
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3).eval()
    g = torch.Generator().manual_seed(1)
    B, L = 12, 64
    batch = {
        "window": torch.randn(B, L, 4, generator=g),
        "tau": torch.full((B,), 5, dtype=torch.long),
        "tpf": torch.full((B,), 200.0),
        "fs": torch.full((B,), 10_000.0),
        "b_over_blim": torch.rand(B, generator=g),
        "label": torch.tensor([0.0, 1.0] * (B // 2)),
        "omega": torch.full((B,), 5000.0),
    }
    out = run_concept_probes(model, batch, test_size=0.4, seed=0)
    names = set(out["concepts"])
    assert {"regeneration_phase_sin", "regeneration_phase_cos"} <= names
    assert "regeneration_phase" not in names                   # raw wrapped angle retired
    sep = out["concepts"]["chatter_tpf_separation"]
    assert sep.get("transform") == "log1p"
```

- [x] **Step 2: Run — first test PASSES (documents the mechanism), second FAILS** (old names).

Run: `python -m pytest tests/test_concept_probe_circularity.py -v`
If the FIRST test does NOT behave as predicted (wrapped probe scores fine), STOP: the hypothesis is wrong — investigate the real cause (pooled-τ mismatch next: compare `extract_hidden_states`'s pooled window vs the raw-window targets), update this task's tests to the confirmed mechanism, and report the revised diagnosis. Do not force the sin/cos fix onto a disproven hypothesis.

- [x] **Step 3: Implement the fix in `concept_probes.py`** (assuming Step 2 confirms):
  - In `build_concept_targets`: replace the `"regeneration_phase"` entry with two entries `regeneration_phase_sin = np.sin(phi_unwrapped)` and `regeneration_phase_cos = np.cos(phi_unwrapped)` where `phi_unwrapped = 2π·k·τ/L` (NO `_wrap_pi` — sin/cos make wrapping irrelevant). Keep `_wrap_pi` exported (the repro test uses it).
  - `chatter_tpf_separation`: return `np.log1p(|dominant_freq − tpf|)`; record `"transform": "log1p"` — thread it through `run_concept_probes` into each concept's summary dict (concepts without a transform record none).
  - Update `CONCEPT_KINDS` accordingly (both new phase concepts `"regression"`).
  - Update `scripts/concept_probes_phase3.py`'s `_probe_summary`/manifest + its figure labels for the new concept names (grep for `regeneration_phase` across `scripts/` and `docs/` templates).
- [x] **Step 4: Run — expect PASS**, plus the existing suite: `python -m pytest tests/test_concept_probe_circularity.py tests/test_concept_probes*.py -q` (locate exact names via `ls tests/ | grep -i concept`).

---

### Task 6: Whitebox integration — atlas + named heatmap + TSR (needs Tasks 1, 3)

**Files:**
- Modify: `scripts/whitebox_phase3.py`, `docs/phase3_tkan/whitebox_analysis.md` (template), `tests/test_whitebox_phase3.py`
- Test: extended `tests/test_whitebox_phase3.py`

**Interfaces:**
- Consumes: `spline_atlas.*` (Task 3), `attribution.temporal_saliency_rescaling`, existing `_safe`/manifest/report machinery, `_extra_scalars_list(model, batch)` (whitebox helper returning per-window side-scalar dicts).
- Produces: manifest `results["spline_atlas"]` (list of atlas rows), `results["tsr"]` (dict below), artifacts `spline_atlas`, `feature_bars`, `edge_importance_named`, report keys `fig_spline_atlas, fig_feature_bars, fig_edge_importance_named, atlas_table, n_atlas_delay_edges, n_atlas_side_edges, tsr_table, tsr_delay_share_gain, tsr_windows, tsr_steps`; new CLI flags `--tsr-windows` (default 4) and `--tsr-steps` (default 8).

- [x] **Step 1: Extend the CLI smoke test** in `tests/test_whitebox_phase3.py` — in the existing end-to-end test add, after the current manifest assertions:

```python
    # spline atlas
    atlas = manifest["results"]["spline_atlas"]
    assert isinstance(atlas, list) and len(atlas) >= 1
    assert {"kan", "feature", "block", "importance", "formula", "r2", "accepted"} <= set(atlas[0])
    for key in ("spline_atlas", "feature_bars", "edge_importance_named"):
        assert Path(manifest["artifacts"][key]).is_file()
    # TSR
    tsr = manifest["results"]["tsr"]
    assert tsr["n_windows"] >= 1
    for row in tsr["per_window"]:
        assert set(row) >= {"label", "channel_share_tsr", "channel_share_ig", "tr_peak_lag", "tau_raw"}
        assert abs(sum(row["channel_share_tsr"].values()) - 1.0) < 1e-6
    assert manifest["errors"] == {}
```

and pass `--tsr-windows 2 --tsr-steps 2` in the CLI args list (keep the smoke fast).

- [x] **Step 2: Run — expect FAIL** (`KeyError: spline_atlas`).

Run: `python -m pytest tests/test_whitebox_phase3.py -v`

- [x] **Step 3: Implement in `scripts/whitebox_phase3.py`:**
  - After the existing `stats` computation, add semantics map once: `sem_map = semantics.edge_index_map(model)` (already imported for `_index_map_table`; check).
  - New analysis (guard `if stats:`):
    ```python
    def _atlas():
        rows = spline_atlas.plot_spline_atlas(model, stats, sem_map, out_dir / "spline_atlas.png", k=int(args.top_k) if args.top_k else 16)
        artifacts["spline_atlas"] = str(out_dir / "spline_atlas.png")
        agg = spline_atlas.feature_importance_bars(stats, sem_map)
        artifacts["feature_bars"] = spline_atlas.plot_feature_importance_bars(agg, out_dir / "feature_bars.png")
        artifacts["edge_importance_named"] = spline_atlas.plot_feature_named_heatmap(stats, sem_map, out_dir / "edge_importance_named.png")
        return rows

    results["spline_atlas"] = _safe("spline_atlas", _atlas)
    ```
  - TSR analysis: pick up to `args.tsr_windows` window indices, alternating chatter/stable from `labels`; for each index `i`: `res = attribution.temporal_saliency_rescaling(model, batch["window"][i], target="chatter_logit", tau=int(batch["tau"][i]), steps=args.tsr_steps, extra_scalars=extra_scalars_list[i] if side else None)`. FIRST verify `target="chatter_logit"` is a valid target for `integrated_gradients` (read `attribution.py:24-98`; if targets are output-dict keys it is — the model returns `chatter_logit`). Per window compute:
    - `channel_share_tsr = {ch: Σ_t |tsr[t,c]| / Σ|tsr|}` (channels from `sampling.CHANNELS`), same for `ig`;
    - `tr_peak_lag`: lag in `[2, L//2]` maximizing the autocorrelation of `time_relevance` (subtract mean first; `np.correlate(tr, tr, "full")` right half);
    - `tau_raw = int(batch["tau"][i])`, `label = int(labels[i])`.
    Aggregate: `results["tsr"] = {"per_window": [...], "n_windows", "steps", "mean_channel_share_tsr", "mean_channel_share_ig", "tr_peak_lag_mean", "tau_raw_mean"}`.
  - Report keys in `_build_report`: `atlas_table` (md table of top-8 rows: gate | feature | unit | importance | formula | R²), `n_atlas_delay_edges`/`n_atlas_side_edges` (counts of `is_delay`/`is_side` in atlas rows), `fig_*` basenames, `tsr_table` (md: window | label | top TSR channel | top IG channel | tr_peak_lag | τ), `tsr_delay_share_gain` (mean over delayed channels of `share_tsr − share_ig` — clearly label; delayed channels here = the per-channel share difference, channel-level only), `tsr_windows`, `tsr_steps`.
  - Template `docs/phase3_tkan/whitebox_analysis.md`: append under section 5 a `### 5b. Spline-edge atlas` block referencing `{{fig_spline_atlas}}`, `{{fig_feature_bars}}`, `{{fig_edge_importance_named}}`, `{{atlas_table}}`, `{{n_atlas_delay_edges}}`, `{{n_atlas_side_edges}}`; under section 4 a `### 4b. Temporal Saliency Rescaling` block referencing `{{tsr_table}}`, `{{tsr_delay_share_gain}}`, `{{tsr_windows}}`, `{{tsr_steps}}` with one honest sentence: TSR is channel/time-resolved saliency; the delay CHANNELS' share is the observable — the τ-INPUT's causal role is established by the ablation, not TSR.
  - CLI: add `--tsr-windows` (int, 4) and `--tsr-steps` (int, 8).
- [x] **Step 4: Run — expect PASS** (all three whitebox tests, incl. render-report placeholder check).

Run: `python -m pytest tests/test_whitebox_phase3.py -q`

---

### Task 7: Fuzz runner — `scripts/fuzz_phase3.py` (needs Tasks 1, 2)

**Files:**
- Create: `scripts/fuzz_phase3.py`
- Test: `tests/test_fuzz_phase3_cli.py`

**Interfaces:**
- Consumes: `pri2m.xai.sampling.build_labeled_batch/CHANNELS`, `pri2m.xai.degradation` (FUZZERS, measure_probes, run_fuzzer, plot_degradation_panel, plot_fuzz_dashboard), `scripts.xai_phase3._find_preprocessing/_load_model`.
- Produces: `fuzz_manifest.json` with top-level keys `provenance, baseline, fuzzers, self_check, artifacts, errors`; figures `fuzz_<name>.png` per fuzzer + `fuzz_dashboard.png`. CLI: `--checkpoint --npz-root --table [--preprocessing] [--output-dir results/phase3/xai/fuzz] [--n-windows 512] [--sample-seed 0] [--window-length 1024] [--window-stride 512] [--pool 4] [--fuzzers all|csv] [--seeds 3] [--quick] [--input-size/--hidden-size/--grid-size/--spline-order] [--run-id/--tag/--val-auc]`. `--quick`: first 2 levels per fuzzer, 1 seed (test/smoke mode; recorded in provenance).

- [x] **Step 1: Write the failing CLI smoke test**

```python
# tests/test_fuzz_phase3_cli.py
"""End-to-end fuzz runner smoke on a synthetic checkpoint."""
import json

import torch

from pri2m.models.tkan_model import TKANModel
from tests._phase3_synth import make_synth_npz_tree, make_synth_table


def _checkpoint(tmp_path):
    torch.manual_seed(0)
    model = TKANModel(
        input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3,
        side_input=True, side_features=("omega", "tpf", "tau"), side_hidden=4,
    )
    d = tmp_path / "runs" / "synthrun" / "checkpoints"
    d.mkdir(parents=True)
    torch.save(model.state_dict(), d / "best_auc.pt")
    (d.parent / "preprocessing.json").write_text(json.dumps({
        "norm": {"mean": [0.0] * 4, "std": [1.0] * 4},
        "side": {"features": ["omega", "tpf", "tau"]},
    }))
    return d / "best_auc.pt"


def test_fuzz_cli_end_to_end(tmp_path):
    import fuzz_phase3 as FZ

    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2), n_files=4, length=256, seed=0)
    make_synth_table(npz_root).to_csv(tmp_path / "table.csv", index=False)
    out = tmp_path / "fuzz"
    rc = FZ.main([
        "--checkpoint", str(_checkpoint(tmp_path)),
        "--npz-root", str(npz_root), "--table", str(tmp_path / "table.csv"),
        "--output-dir", str(out), "--n-windows", "6",
        "--window-length", "128", "--window-stride", "64", "--pool", "1",
        "--quick", "--tag", "synth",
    ])
    assert rc == 0
    manifest = json.loads((out / "fuzz_manifest.json").read_text())
    assert set(manifest) >= {"provenance", "baseline", "fuzzers", "self_check", "artifacts", "errors"}
    assert manifest["errors"] == {}
    assert manifest["self_check"]["passed"] is True
    assert manifest["provenance"]["quick"] is True
    for name in ("input_noise", "input_smooth", "tau_jitter", "spline_coarsen", "weight_noise"):
        fz = manifest["fuzzers"][name]
        assert fz["results"][0]["level"] == fz["levels"][0]
        assert "breakdown" in fz
        assert (out / f"fuzz_{name}.png").is_file()
    assert (out / "fuzz_dashboard.png").is_file()
    for k in ("auc", "delay_drop", "mae"):
        assert manifest["baseline"][k] is not None
```

- [x] **Step 2: Run — expect FAIL.** `python -m pytest tests/test_fuzz_phase3_cli.py -v`

- [x] **Step 3: Implement `scripts/fuzz_phase3.py`.** Structure (mirror `whitebox_phase3.py`'s conventions — sibling-script import bootstrap, `_git_sha`, `_now_date` helpers can be imported from `whitebox_phase3`):
  - argparse per the interface above.
  - Load: `pre_path = args.preprocessing or xai_phase3._find_preprocessing(args.checkpoint)`; `model = xai_phase3._load_model(args.checkpoint, args, preprocessing)`; `model.pool = int(args.pool)`; `model.eval()`.
  - Batch: `cfg = WindowConfig(length=args.window_length, stride=args.window_stride, channels=sampling.CHANNELS)`; `batch, labels = sampling.build_labeled_batch(args.table, args.npz_root, preprocessing, args.n_windows, cfg, seed=args.sample_seed)`.
  - `baseline = degradation.measure_probes(model, batch, labels)`; `baseline["xdelay_share"] = degradation.xdelay_share(model, batch)`.
  - Select fuzzers (`all` → registry order), apply `--quick` (slice `levels[:2]`, `seeds=(0,)` — build a modified `Fuzzer` via `dataclasses.replace`).
  - Per fuzzer inside a `_safe(name, fn)` copy of the whitebox pattern (errors dict, never crash the run): `res = degradation.run_fuzzer(model, batch, labels, fz, seeds=tuple(range(args.seeds)))`; figure `degradation.plot_degradation_panel(res, baseline, out_dir / f"fuzz_{fz.name}.png")`.
  - Self-check: for every completed fuzzer compare level-0 `mean` vs `baseline` on `auc`/`delay_drop`/`mae`: tolerance `1e-6` except `spline_coarsen` (`5e-3`, refit numerics). `self_check = {"passed": all(...), "max_abs_dev": ..., "per_fuzzer": {...}}`. `passed=False` must NOT fail the run — it must be visible in the manifest (the GL log prints a loud warning).
  - Dashboard over completed fuzzers; manifest provenance mirrors whitebox (`checkpoint, preprocessing, run_id, tag, val_auc, git_sha, rendered_date, n_windows, sample_seed, pool, seeds, quick, fuzzer_levels`) + write `fuzz_manifest.json` with `json.dumps(..., indent=2, default=float)`; print summary lines (baseline auc, per-fuzzer breakdown levels). Return 0.
- [x] **Step 4: Run — expect PASS.** `python -m pytest tests/test_fuzz_phase3_cli.py -v`
- [x] **Step 5: Real-shape sanity (local, synthetic but full levels):** rerun the CLI WITHOUT `--quick` on the synth checkpoint (same args otherwise) via a shell one-liner; confirm it completes < ~10 min and the manifest has all levels. This is the local stand-in for the GL cost estimate.

---

### Task 8: Ops — slurm wrappers, replication plot, sync, ensemble wrapper, full-suite gate

**Files:**
- Create: `slurm/phase3_fuzz.sbatch`, `slurm/run_fuzz_all_folds.sh`, `slurm/phase3_predict_windows.sbatch`, `slurm/run_conformal_ensemble.sh`, `scripts/plot_fuzz_replication.py`
- Modify: `sync_phase3_tkan_to_hpc.sh`, `slurm/phase3_counterfactuals.sbatch` (usage comment only)
- Test: `tests/test_plot_fuzz_replication.py`; `bash -n` on every new/edited shell file; full-suite gate.

**Interfaces:**
- Consumes: fuzz manifest schema (Task 7), the 5 fold run-ids/AUCs (Key contracts), existing sbatch env conventions.
- Produces: user-launchable GL entry points; `plot_fuzz_replication.py --manifests m1.json m2.json ... --labels fold0 fold1 ... --out results/figures/research_update_assets/fuzz_replication.png` overlaying per-fuzzer AUC + delay-drop curves across folds.

- [x] **Step 1: Failing test for the replication plot**

```python
# tests/test_plot_fuzz_replication.py
import json


def _manifest(tmp_path, name, shift):
    m = {"fuzzers": {"input_noise": {
        "levels": [0.0, 0.5, 1.0],
        "results": [{"level": l, "mean": {"auc": 0.95 - shift * i, "delay_drop": 0.3 - 0.1 * i},
                     "sd": {"auc": 0.01, "delay_drop": 0.01}} for i, l in enumerate([0.0, 0.5, 1.0])],
        "breakdown": {"auc_below_075": None, "drop_below_005": None, "auc_at_chance": None},
    }}}
    p = tmp_path / name
    p.write_text(json.dumps(m))
    return p


def test_replication_overlay(tmp_path):
    import plot_fuzz_replication as R

    m1 = _manifest(tmp_path, "a.json", 0.05)
    m2 = _manifest(tmp_path, "b.json", 0.10)
    out = tmp_path / "rep.png"
    rc = R.main(["--manifests", str(m1), str(m2), "--labels", "fold0", "fold1", "--out", str(out)])
    assert rc == 0 and out.is_file()
```

- [x] **Step 2: Run — expect FAIL**, then implement `scripts/plot_fuzz_replication.py`: argparse (`--manifests` nargs=+, `--labels` nargs=+ same length, `--out`); one subplot per fuzzer name present in ALL manifests; per fold plot mean AUC (solid, one color per fold) and delay_drop (dashed, same color, twin axis); legend = labels; style matches `plot_whitebox_replication.py`. Verify: `python -m pytest tests/test_plot_fuzz_replication.py -v` → PASS.

- [x] **Step 3: `slurm/phase3_fuzz.sbatch`** — copy the header + env block of `slurm/phase3_conformal.sbatch` VERBATIM (account/partition/module/venv/PYTHONPATH/threads/PYTHONUNBUFFERED), change: job-name `pri2m-p3-fuzz`, time `08:00:00`, mem `32G`, cpus 8, log names `phase3_fuzz_%j`. Env vars: `PROJECT_ROOT, RUNS_DIR, RUN_ID (default fold2 a654129d...), CHECKPOINT, TAG (fold2), NPZ_ROOT, TABLE, PREPROCESSING, OUTPUT_DIR (default $PROJECT_ROOT/results/phase3/xai/fuzz/$TAG), N_WINDOWS (512), SEEDS (3), FUZZERS (all), VAL_AUC ()`. Command:

```bash
python scripts/fuzz_phase3.py \
  --checkpoint "$CHECKPOINT" --npz-root "$NPZ_ROOT" --table "$TABLE" \
  --output-dir "$OUTPUT_DIR" --n-windows "$N_WINDOWS" --seeds "$SEEDS" \
  --fuzzers "$FUZZERS" --run-id "$RUN_ID" --tag "$TAG" \
  "${VAL_FLAG[@]}" "${PRE_FLAG[@]}"
```

with the same `VAL_FLAG`/`PRE_FLAG` conditional-array pattern as the whitebox sbatch.

- [x] **Step 4: `slurm/run_fuzz_all_folds.sh`** — copy `slurm/run_whitebox_all_folds.sh` structure exactly; `submit()` targets `slurm/phase3_fuzz.sbatch`; the same five `submit foldN <run_id> <val_auc>` lines (run-ids/AUCs from Key contracts; plain `fold2` tag is fine here — output dirs are namespaced by `fuzz/`).

- [x] **Step 5: `slurm/phase3_predict_windows.sbatch`** — conformal header, time `01:00:00`, mem 16G, cpus 4, job-name `pri2m-p3-preds`. Env: `PROJECT_ROOT, RUNS_DIR, RUN_ID (fold2 default), CHECKPOINT, TAG, TABLE, NPZ_ROOT, OUT (default $PROJECT_ROOT/results/phase3/preds_$TAG.csv), DATASET (all)`. Runs `python scripts/predict_windows_phase3.py --checkpoint ... --table ... --npz-root ... --out "$OUT" --dataset "$DATASET"`. Then EDIT the usage comment at the top of `slurm/phase3_counterfactuals.sbatch` to document the two-step flow:

```
# Real-network counterfactuals (2 steps):
#   1) env RUN_ID=<run> TAG=fold2 sbatch --export=ALL slurm/phase3_predict_windows.sbatch
#   2) env DATASET=all TAG=fold2_net PREDICTIONS=$PROJECT_ROOT/results/phase3/preds_fold2.csv \
#          CHECKPOINT=$RUNS_DIR/<run>/checkpoints/best_auc.pt sbatch --export=ALL slurm/phase3_counterfactuals.sbatch
```

- [x] **Step 6: `slurm/run_conformal_ensemble.sh`** — small wrapper (mirror `run_whitebox_all_folds.sh` header):

```bash
#!/bin/bash
# Submit the 5-fold cross-checkpoint conformal ENSEMBLE (uses the existing
# multi-checkpoint support in conformal_phase3.py / phase3_conformal.sbatch).
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
RUNS="$PROJECT_ROOT/results/phase3/runs"
CKPTS=""
for r in 61c67bddbdbd49ba9727ff92f4814c86 c21e58dc0e624e1dbc602ec63a2dc23e \
         a654129d2cdc460093e21373f31ee5a5 3a40b71514cd41b98916330c000d700a \
         0937883918354ed899a61afb90aac14f; do
  CKPTS+="$RUNS/$r/checkpoints/best_auc.pt "
done
env TAG=ensemble CHECKPOINTS="$CKPTS" sbatch --export=ALL slurm/phase3_conformal.sbatch
```

Also: `grep -rn "checkpoint" tests/test_conformal*.py` — if no test drives the CLI with ≥2 checkpoints, add one to the existing conformal test file (2 synthetic checkpoints of the same architecture, assert `manifest["provenance"]["ensemble"] is True` and finite headline coverage).

- [x] **Step 7: `sync_phase3_tkan_to_hpc.sh`** — add in the marked blocks: `rs scripts/fuzz_phase3.py`, `rs scripts/predict_windows_phase3.py`, `rs scripts/plot_fuzz_replication.py` (scripts block ~L78); `rs slurm/phase3_fuzz.sbatch`, `rs slurm/run_fuzz_all_folds.sh`, `rs slurm/phase3_predict_windows.sbatch`, `rs slurm/run_conformal_ensemble.sh` (slurm block ~L94); test stems `test_xai_sampling test_degradation test_spline_atlas test_predict_windows_phase3 test_concept_probe_circularity test_fuzz_phase3_cli test_plot_fuzz_replication` into the `for t in ...` list (~L119). `src/pri2m/xai/` new modules are covered by the recursive mirror. Verify `bash -n sync_phase3_tkan_to_hpc.sh` and `bash -n` each new slurm file.

- [x] **Step 8: FULL-SUITE GATE.** Run: `python -m pytest -q` → expect **0 failures, 1 skipped**, total ≥ 561+new collected, and no test slower than ~120 s (the fuzz smoke must stay `--quick`). Also re-run the two real-CLI smokes end-to-end once more:
`python -m pytest tests/test_whitebox_phase3.py tests/test_fuzz_phase3_cli.py -q`.

---

## After the build (user-run, GL) — NOT part of this plan's tasks

1. `./sync_phase3_tkan_to_hpc.sh`
2. `bash slurm/run_fuzz_all_folds.sh` (5 fuzz manifests) and re-run `bash slurm/run_whitebox_all_folds.sh` (atlas + TSR now included)
3. `env RUN_ID=a654129d... TAG=fold2 sbatch --export=ALL slurm/phase3_predict_windows.sbatch`, then the counterfactuals 2-step (see sbatch comment)
4. `bash slurm/run_conformal_ensemble.sh`; concept probes re-run: `sbatch --export=ALL slurm/phase3_concept_probes.sbatch`
5. rsync results local → `python scripts/plot_fuzz_replication.py --manifests results/phase3/xai/fuzz/fold*/fuzz_manifest.json --labels fold0 fold1 fold2 fold3 fold4 --out results/figures/research_update_assets/fuzz_replication.png`
6. Research Update 10 (per `research_updates_workflow`): fuzz breakdown-point story + atlas figures + closed-queue results; placeholders until GL numbers land.

## Plan self-review (done at write time)

- **Spec coverage:** §3.1 fuzzers → Task 2 (prune reuse = existing `pruning_curve`, surfaced in the whitebox report already — the dashboard covers 5 new fuzzers; pruning stays whitebox-side, spec's "imported for completeness" satisfied by the dashboard note in Update 10 rather than double-running); §3.2 probes/breakdown → Task 2; §3.3 runner/sbatch/replication → Tasks 7, 8; §3.4 tests → Tasks 2, 7; §4 atlas → Tasks 3, 6; §5.1 TSR → Task 6; §5.2 predictions → Task 4 + Task 8 step 5; §5.3 conformal → Task 8 step 6; §5.4 probes → Task 5; §6 execution → waves + "After the build".
- **Type consistency:** `Fuzzer.apply(model, batch, level, seed) -> (model, batch, info)` everywhere; `run_fuzzer` result schema matches Task 7's manifest consumer and Task 8's replication-plot test fixture; `npz_exists` public name with `_npz_exists` alias kept in whitebox.
- **Placeholder scan:** the two "check the real signature first" notes (Task 4 `build_surrogate` call form; Task 6 IG target validity) are deliberate verification steps against live code, each with fixed assertion targets — not TBDs.
