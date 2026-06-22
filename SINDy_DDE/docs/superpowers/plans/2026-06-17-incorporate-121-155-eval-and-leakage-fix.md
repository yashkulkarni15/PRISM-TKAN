# Incorporate Datasets 121–155 (Eval-Only) + Close the Param-Twin Leakage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the new Schmitz datasets 121–155 into PRI2M as an **evaluation-only** extension (no retraining), and in the same pass fix the dataset-grouping so that *parameter-identical* datasets can never straddle train/val — which both enables the new data and closes a latent leakage in the existing 5-fold result.

**Architecture:** Replace the arithmetic `parent_group` (which only knew the D / D+60 / D+90 noise-mirror offsets) with a **connected-component grouping over parameter identity**, materialized once into a checked-in `data/metadata/parent_groups.json` and loaded by `splits.py`. Then quantify the impact of the fix on the published 0.955, fetch + extract the 35 new datasets on Great Lakes, and run the existing 1–120 checkpoints over them as a held-out **replication** eval (126–155) and **OOD probe** (122–125). No model is retrained.

**Tech Stack:** Python 3.13, pandas/numpy, PyTorch 2.9 (existing `pri2m` package), pytest (TDD), Great Lakes SLURM + Kaggle CLI for data ops.

---

## Decisions locked (from the 2026-06-17 EDA + user direction — do not relitigate)

1. **Eval-only.** Do NOT retrain on 121–155. The new batch is mostly a replication of 1–30 (see below), so it goes through the *evaluation* path, not training.
2. **Param-twin co-grouping.** `parent_group` must co-group datasets with identical documented params. This is the single mechanism that (a) maps the new noisy datasets correctly and (b) closes the existing 1≡23 / 16≡17 leakage.
3. **122–125 = OOD probe only.** The four genuinely novel parameterizations exist only at 5% noise; use them as a held-out out-of-distribution stress-test of existing checkpoints, never in training.
4. **Commit policy unchanged.** Leave everything in the working tree; the user commits manually (their name only, no Claude co-author). Branch `phase3-tkan`.

## EDA findings this plan is built on (2026-06-17, from `data/metadata/Dataset descriptions.xlsx`)

- The xlsx documents **155** datasets (contiguous, no gaps). Feature CSVs cover only 1–120.
- **126–155 (30 clean) are exact parameter-twins of 1–30** (126≡1 … 155≡30, including the Cn process-damping pairs 150–155≡25–30). A second realization of operating points already in training.
- **121 (noisy@5%)** is a param-twin of clean **14** (param-group {14, 74, 104, 139}).
- **122, 123, 124, 125 (noisy@5%)** are the **only genuinely novel parameterizations** in the batch — no twin anywhere in 1–120. 122/124/125 carry process damping (Cn 300k/350k/200k), 123 Cn=0; all reach low-speed (omega-min 1000). No clean counterpart exists.
- **Latent leakage in the existing result:** within clean 1–60, **1≡23 and 16≡17** are parameter-identical, but the arithmetic `parent_group` placed them in *separate* groups; `assert_no_mirror_leakage` only checks the D/D+60/D+90 relation, so it misses them. Connected-component over param identity reproduces the arithmetic map on every non-twin dataset and differs only by merging the 6 known deviations (17,23,77,83,107,113 → their twin's id) — it never *splits* a group, so it can only reduce leakage.
- Raw H5 for 121–155 is on a **newer Kaggle version**, to be fetched **directly on Great Lakes** (not local). GL currently has only `versions/10` (= 1–120).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `scripts/build_parent_groups.py` | One-shot generator: xlsx → connected-component group map → `parent_groups.json` | **Create** |
| `data/metadata/parent_groups.json` | Frozen `{dataset: group_id}` for 1–155 (runtime artifact, checked in) | **Create (generated)** |
| `src/pri2m/phase3/splits.py` | `parent_group` loads the JSON map; arithmetic fallback when absent | **Modify** (L53–101) |
| `tests/test_parent_groups.py` | Param-twin grouping + arithmetic-reproduction + new-id mapping | **Create** |
| `src/pri2m/config.py` *(or `configs/phase1_3.yaml`)* | Single `DATA_ROOT` / `DATA_VERSION` source | **Modify** |
| `scripts/extract_phase2_signals.py` (L48), `slurm/phase2_extract_signals.sbatch` (L24), `sindy_pipeline/sindy_preprocess.py` (L44), `slurm/README.md` (L37) | Replace hardcoded `versions/10` | **Modify** |
| `scripts/audit_fold_twin_leakage.py` | Metadata-level quantification of co-located param-twins in the published folds | **Create** |
| `scripts/eval_heldout_phase3.py` | Load a checkpoint + run `evaluate` over an arbitrary dataset table (no folds, no retrain) | **Create** |
| `tests/test_eval_heldout.py` | Synthetic round-trip for the eval harness | **Create** |
| `docs/phase3_tkan/PROGRESS.md` | Update PARKED section + NEXT ACTION to match this plan | **Modify** |

---

## Phase A — Leakage-safe grouping (LOCAL, signal-free, TDD). *Start here once approved.*

### Task 1: Connected-component param-twin group map

**Files:**
- Create: `scripts/build_parent_groups.py`
- Create (generated): `data/metadata/parent_groups.json`
- Modify: `src/pri2m/phase3/splits.py` (replace `parent_group` / `_parent_group_array`, L53–101)
- Test: `tests/test_parent_groups.py`

- [ ] **Step 1: Write the generator script**

```python
# scripts/build_parent_groups.py
"""Generate the leakage-safe parent-group map for datasets 1..155.

Two datasets must never straddle a train/val split when they share operating
points. That happens whenever their documented modal/cutting parameters are
identical (noise is added to the *signal*, not the params, so every noise mirror
shares its clean parent's params; and several clean datasets are exact repeats,
e.g. 1==23, 16==17, 126..155==1..30). We fingerprint each dataset on all params
across all modes (excluding noise std + bookkeeping), union datasets with equal
fingerprints, and label each connected component by its smallest dataset id.

This reproduces the old arithmetic map (D, D+60, D+90 -> D) on every dataset that
has no parameter twin, and only ever MERGES groups the arithmetic map kept apart.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
XLSX = REPO / "data/metadata/Dataset descriptions.xlsx"
OUT = REPO / "data/metadata/parent_groups.json"
EXCLUDE = {"Dataset", "Mode number", "Noise std (%)"}


def fingerprint(group: pd.DataFrame, params: list[str]) -> tuple:
    g = group.sort_values("Mode number")
    vals = np.round(g[params].to_numpy(dtype=float), 6)
    return tuple(np.nan_to_num(vals, nan=-9.99e99).ravel().tolist())


def build_map(xlsx: Path = XLSX) -> dict[int, int]:
    df = pd.read_excel(xlsx)
    df["Dataset"] = pd.to_numeric(df["Dataset"], errors="coerce").ffill()
    df = df[df["Dataset"].notna()].copy()
    df["Dataset"] = df["Dataset"].astype(int)
    params = [c for c in df.columns if c not in EXCLUDE]
    for c in params + ["Mode number"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Mode number"])

    fp = {d: fingerprint(g, params) for d, g in df.groupby("Dataset")}
    by_fp: dict[tuple, list[int]] = defaultdict(list)
    for d, f in fp.items():
        by_fp[f].append(d)
    group_map: dict[int, int] = {}
    for members in by_fp.values():
        gid = min(members)
        for d in members:
            group_map[d] = gid
    return dict(sorted(group_map.items()))


if __name__ == "__main__":
    gm = build_map()
    OUT.write_text(json.dumps({str(k): v for k, v in gm.items()}, indent=2))
    print(f"[build_parent_groups] wrote {len(gm)} datasets -> {OUT}")
    print("  distinct groups:", len(set(gm.values())))
```

- [ ] **Step 2: Generate the artifact and eyeball it**

Run: `python scripts/build_parent_groups.py`
Expected: `wrote 155 datasets`, `distinct groups: 60` (1–155 collapse onto the 60 clean parameterizations 1–60; new clean repeats and all mirrors fold in).

- [ ] **Step 3: Write the failing test**

```python
# tests/test_parent_groups.py
import json
from pathlib import Path
import pytest
from pri2m.phase3.splits import parent_group

REPO = Path(__file__).resolve().parents[1]
MAP = json.loads((REPO / "data/metadata/parent_groups.json").read_text())


def _arith(d):
    if 91 <= d <= 120: return d - 90
    if 61 <= d <= 90: return d - 60
    return d


@pytest.mark.parametrize("d,expected", [
    (23, 1), (83, 1), (113, 1),       # 23-cluster merges into 1
    (17, 16), (77, 16), (107, 16),    # 17-cluster merges into 16
    (139, 14), (121, 14),             # new clean 139 + new noisy 121 -> clean 14
    (126, 1), (148, 1),               # new clean repeats of 1
    (150, 25), (155, 30),             # new Cn clean repeats of 25 / 30
    (122, 122), (123, 123), (124, 124), (125, 125),  # novel noisy -> own group
])
def test_known_mappings(d, expected):
    assert parent_group(d) == expected


def test_reproduces_arithmetic_except_known_twins():
    twins = {17, 23, 77, 83, 107, 113}
    for d in range(1, 121):
        if d in twins:
            continue
        assert parent_group(d) == _arith(d), f"changed non-twin {d}"


def test_only_merges_never_splits():
    # Every arithmetic group id must still be a *subset* of its new group.
    from collections import defaultdict
    new_groups = defaultdict(set)
    for d in range(1, 156):
        new_groups[parent_group(d)].add(d)
    for d in range(1, 121):
        a = _arith(d)
        # d and its arithmetic parent land in the same new group
        assert parent_group(d) == parent_group(a)
```

- [ ] **Step 4: Run it, expect failure** — `pytest tests/test_parent_groups.py -v` → FAIL (`parent_group` still arithmetic; 23→23 not 1).

- [ ] **Step 5: Rewrite `parent_group` to load the map** (replace L53–101 of `src/pri2m/phase3/splits.py`)

```python
import functools, json
from pathlib import Path

_GROUP_MAP_PATH = Path(__file__).resolve().parents[3] / "data/metadata/parent_groups.json"


@functools.lru_cache(maxsize=1)
def _group_map() -> dict[int, int]:
    try:
        raw = json.loads(_GROUP_MAP_PATH.read_text())
    except FileNotFoundError:
        return {}
    return {int(k): int(v) for k, v in raw.items()}


def _arith_scalar(d: int) -> int:
    if 91 <= d <= 120: return d - 90
    if 61 <= d <= 90: return d - 60
    return d


@overload
def parent_group(dataset: int) -> int: ...
@overload
def parent_group(dataset: pd.Series) -> pd.Series: ...
def parent_group(dataset):
    if isinstance(dataset, pd.Series):
        out = _parent_group_array(dataset.to_numpy())
        return pd.Series(out, index=dataset.index, name=dataset.name)
    d = int(dataset)
    return _group_map().get(d, _arith_scalar(d))


def _parent_group_array(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d)
    gm = _group_map()
    if not gm:  # artifact absent -> legacy arithmetic behavior (1..120 only)
        out = d.astype(np.int64, copy=True)
        out = np.where((d >= 61) & (d <= 90), d - 60, out)
        out = np.where((d >= 91) & (d <= 120), d - 90, out)
        return out.astype(np.int64, copy=False)
    flat = [gm.get(int(x), _arith_scalar(int(x))) for x in d.ravel()]
    return np.asarray(flat, dtype=np.int64).reshape(d.shape)
```

- [ ] **Step 6: Run the full splits + new test suite** — `pytest tests/test_parent_groups.py tests/test_phase3_splits.py -v` (use the real existing splits test filename) → PASS. Then `pytest -q` → expect the prior green count with **no regressions** (the only behavioral change is the 6 intended merges + new ids).

- [ ] **Step 7: Commit** *(only on the user's explicit word — see policy)*

### Task 2: Parametrize the data version

**Files:**
- Modify: `scripts/extract_phase2_signals.py:48`, `slurm/phase2_extract_signals.sbatch:24`, `sindy_pipeline/sindy_preprocess.py:44`, `slurm/README.md:37`, `handoff.md:615`

- [ ] **Step 1:** Introduce one source of truth — add `DATA_VERSION` (default `"10"`) + a derived `DATA_ROOT` to `configs/phase1_3.yaml` and read it where a default is needed. The CLI/sbatch already accept `--data-root` / `DATA_ROOT` env; the fix is to stop hardcoding `versions/10` as the literal default.

- [ ] **Step 2:** In `scripts/extract_phase2_signals.py:48` change the default to build from `DATA_VERSION` (env override): `default=os.environ.get("DATA_ROOT", f".../digital-machining-database/versions/{os.environ.get('DATA_VERSION','10')}")`.

- [ ] **Step 3:** In `slurm/phase2_extract_signals.sbatch:24` and `sindy_pipeline/sindy_preprocess.py:44` mirror the same `${DATA_VERSION:-10}` indirection. Update the two doc strings (`slurm/README.md:37`, `handoff.md:615`).

- [ ] **Step 4: Audit count assumptions.** `grep -rn '\b60\b\|\b90\b\|\b120\b\|33719' scripts/build_phase3_dataset.py src/pri2m/phase3/ src/pri2m/noise/calibrate*` — confirm nothing hardcodes "120 datasets" or "33719 rows". `build_phase3_table` reads whatever feature CSVs it's given, so it is count-agnostic; record that finding in the plan log.

- [ ] **Step 5:** No new runtime behavior on the default path → run `pytest -q` to confirm zero regressions; commit on the user's word.

---

## Phase B — Quantify the latent leakage in the published 0.955 (LOCAL metadata + one GL signal check)

### Task 3: Metadata-level fold-twin audit

**Files:** Create `scripts/audit_fold_twin_leakage.py`

- [ ] **Step 1:** Load the published `results/phase3/phase3_folds.json` + `phase3_dataset.csv`. For each fold, list every val `parent_group` (under the OLD arithmetic map) whose param-twin (per `parent_groups.json`) sits in that fold's *train* set. Output a table: `fold, n_val_files_with_twin_in_train, datasets_involved`.

- [ ] **Step 2:** Report the worst case: clusters [1,23] and [16,17]. If, in the published 5-fold assignment, 1 and 23 (or 16 and 17) landed in different folds, those val operating points were also in train via the twin. Print the exact count of affected val files (bounded; these are 4 of 60 clean datasets, so ≤ a few % of val).

### Task 4: GL signal-identity check (decides severity)

> GL only — needs the NPZ that live on Great Lakes.

- [ ] **Step 1:** On GL, compare the actual signals of the twin pairs: are `dataset_1/*.npz` and `dataset_23/*.npz` (matched on `(omega, b)`) **byte-identical** (deterministic sim → true duplication, serious) or merely **similar** (independent realization → mild, comparable to the noise-mirror leakage we already guard)? One-liner: load matched files, report `max|x1-x23|` per signal.

- [ ] **Step 2: Decide.**
  - *Identical* → the 0.955 had real duplicate-operating-point leakage; **re-run the 5-fold with the corrected `parent_groups.json`** (reuse the existing GPU sbatch; only the folds change) and report the corrected AUC. The fix is already in place from Task 1.
  - *Merely similar* → document as a bounded caveat in PROGRESS/the paper; no re-run required.

- [ ] **Step 3:** Record the verdict + (if re-run) the corrected 5-fold AUC in `docs/phase3_tkan/PROGRESS.md`.

---

## Phase C — Acquire + extract 121–155 (GL ops runbook; gated on Kaggle fetch)

> Not unit-testable (Kaggle download + H5 extraction). Run on Great Lakes. **Pull ONLY the 35 new datasets — do not re-pull 1–120.**

- [ ] **Step 1: Find the newer Kaggle version on GL.**
  `kaggle datasets metadata tonylschmitz/digital-machining-database` then
  `ls /nfs/turbo/coe-jaabell/datasets/tonylschmitz/digital-machining-database/versions/` — identify the version `N` that contains `Dataset 121 h5/ … Dataset 155 h5/`. If absent, `kaggle datasets download … -p versions/<N>/ --unzip` and keep only the 35 new `Dataset 12x–155 h5/` folders.

- [ ] **Step 2: Confirm the metadata CSVs for 121–155.** `build_phase3_table` needs per-file feature rows (omega, b_mm, blim, b_over_blim, label, fn, teeth, cut_dir, kurtosis cols). Check whether version `N` ships extended `features_*.csv` covering 121–155. **If not, this is a blocker — flag to the data owner** (the spectral feature columns require the signals; the basic metadata is derivable from the xlsx but the kurtosis/feature columns are not). Do not fabricate feature rows.

- [ ] **Step 3: Extract NPZ for the 35 new datasets** via `slurm/phase2_extract_signals.sbatch` with `DATA_VERSION=<N>` and the dataset filter set to `121-155` (split clean 126–155 vs noisy 121–125 per the existing clean/noisy CSV convention). Verify manifests show 0 failures.

- [ ] **Step 4: Phase 2.5 for the 5 new noisy datasets.** Run `scripts/compute_noise_metrics.py` + `scripts/assign_signal_modes.py` over 121–125 so each new file gets a `mode_label` (the build needs it for the `metrics_unavailable` exclusion). No re-calibration of promoted metrics is required for an eval-only pass; reuse the existing pooled `promoted_metrics.json`.

- [ ] **Step 5: Build the eval tables** (NOT folds) — run `build_phase3_table` over (a) 126–155 and (b) 122–125 to get `phase3_eval_replication.csv` and `phase3_eval_ood.csv` with `parent_group`, `sample_weight`, and exclusion flags. These are scored, never trained.

---

## Phase D — Held-out evaluation (eval-only, no retrain)

### Task 5: The eval harness

**Files:** Create `scripts/eval_heldout_phase3.py`, Test `tests/test_eval_heldout.py`

Reuses (verified signatures): `pri2m.data.phase3_sequence.Phase3SequenceDataset(table, npz_root, cfg, norm=…)` + `collate_windows`; `pri2m.training.phase3_trainer.Trainer.evaluate(model, loader)`; `pri2m.training.metrics.per_file_severity_metrics`; the checkpoint reconstruction logic in `scripts/xai_phase3.py:_load_model` + `_infer_dims` (side-model aware) + `NormStats.from_dict`.

- [ ] **Step 1: Write the failing synthetic test**

```python
# tests/test_eval_heldout.py
def test_eval_heldout_runs_on_synthetic(tmp_path):
    # Reuse tests/_phase3_synth.py to write a tiny NPZ tree + table + a trained-ish
    # checkpoint with preprocessing.json, then assert eval_heldout returns a dict
    # with finite 'auc'/'mae'/'n_files' and never refits normalization (uses the
    # checkpoint's NormStats.from_dict, not the eval data's stats).
    ...
```

- [ ] **Step 2:** Run → FAIL (no `eval_heldout_phase3`).

- [ ] **Step 3: Implement the harness.** Load checkpoint + `preprocessing.json`; reconstruct the (possibly side-input) model via the `_load_model` path; build a `Phase3SequenceDataset` over the eval table with `norm = NormStats.from_dict(preprocessing["norm"])` (**TRAIN-only norm from the checkpoint — do not refit on eval data**); wrap in a plain `DataLoader(shuffle=False, collate_fn=collate_windows)`; call `Trainer.evaluate(model, loader)`; write per-file AUC/MAE/RMSE + `n_files`/`n_nonfinite_files` to JSON. **No fold logic, no `assert_no_mirror_leakage`** (pure inference on a disjoint dataset).

- [ ] **Step 4:** Run → PASS. Then `pytest -q` for no regressions. Commit on the user's word.

### Task 6: Run the two evals on GL + report

- [ ] **Step 1 — Replication (126–155):** for each fold's `best_auc.pt`, `python scripts/eval_heldout_phase3.py --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt --table phase3_eval_replication.csv --npz-root $PWD/phase2_signals`. **Interpretation:** AUC here measures *realization robustness* (same operating points, new draw) — report it as such, NOT as out-of-distribution generalization.

- [ ] **Step 2 — OOD probe (122–125):** same harness on `phase3_eval_ood.csv`. **Interpretation:** novel parameterizations seen through 5% noise — a genuine (hard, small-n) OOD stress-test. Expect lower AUC; that is the headline number for "does TKAN generalize beyond the training envelope".

- [ ] **Step 3:** Tabulate both alongside the in-fold 0.955 in `docs/phase3_tkan/PROGRESS.md`. State plainly that 126–155 is replication and only 122–125 is OOD.

---

## Out of scope here (flagged, separate plans)

- **Richer leakage-safe side inputs** (Cn / fn / ktc / teeth / cut_dir on top of omega/tpf/tau): a *training* change → needs a retrain, deferred by Decision 1. Worth a follow-up once the OOD number is known.
- **Recovering the 700 dropped low-speed rows:** this is a **physics task** (re-derive `blim` with a process-damping model for the Cn=0 low-speed rows), NOT unlocked by the new data — datasets 25–30 already carried Cn≠0 at low speed and were already in the corpus. Corrects the optimistic framing in the old PARKED note.

## Self-review checklist (done)

- **Spec coverage:** leakage fix (Task 1), version parametrization (Task 2), existing-result audit (Tasks 3–4), data acquisition (Phase C), eval-only harness (Task 5), the two evals (Task 6) — all of the user's two decisions are covered; no retraining anywhere.
- **Type consistency:** `parent_group`/`_parent_group_array`/`_group_map`/`_arith_scalar` names are consistent across Task 1; `build_phase3_table`, `Phase3SequenceDataset`, `Trainer.evaluate`, `per_file_severity_metrics`, `NormStats.from_dict` are the real existing symbols.
- **Open blocker to confirm on GL (Phase C Step 2):** whether the newer Kaggle version ships feature CSVs for 121–155; if not, request from the data owner before Phase D.
