"""Shared synthetic data builders for Phase 3 TKAN tests.

Every Phase 3 test (data layer, model, trainer, XAI, end-to-end) builds its
inputs from here so the schema stays consistent across modules and subagents.

Import as::

    from tests._phase3_synth import (
        make_synth_npz_tree, make_synth_table, make_synth_folds, CHANNELS,
    )

The builders mirror the real pipeline:

* NPZ bundles match ``scripts/extract_phase2_signals.py`` output:
  keys ``xtpos, ytpos, Forcex, Forcey`` (float[length]) plus
  ``fs, label, omega, tpf, fn, dataset, source_file``.
* The per-file table matches ``results/phase3/phase3_dataset.csv`` columns used
  downstream: ``dataset, file, label, b_over_blim, blim, noise_score,
  mode_label, mode_confidence, parent_group, sample_weight, exclude_supervised,
  physics_usable, omega, tpf, fn`` (+ kurtosis columns).
* Mirrors ``D+60`` (1% noise) and ``D+90`` (5% noise) reuse clean ``D``'s
  operating points with more noise.
* Two audit pathologies are injected on the first clean dataset: a negative-blim
  row (``fi==0``: ``blim<0``, ``b_over_blim``=NaN, stored ``label``=1) and a
  ``metrics_unavailable`` row (``fi==1``: ``noise_score``=NaN). Both must be
  excluded from the supervised target (``sample_weight==0``).

Unstable files (``b_over_blim>=1``) carry a strong tone near ``fn`` (chatter);
stable files are tooth-pass-frequency dominated with low noise, so severity and
trajectory models have a learnable signal.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import numpy as np
import pandas as pd

CHANNELS: tuple[str, ...] = ("xtpos", "ytpos", "Forcex", "Forcey")

_FS = 10_000.0
_TPF = 200.0
_FN = 900.0
_NOISE_BY_TIER = {"clean": 0.05, "n1": 0.20, "n5": 0.50}
_KURT_COLS = ("xtpos_kurt", "ytpos_kurt", "Forcex_kurt", "Forcey_kurt", "xtvel_kurt", "ytvel_kurt")
_STEM_RE = re.compile(r"^d(\d+)_f(\d+)$")


def _clean_parent(ds: int) -> int:
    if 91 <= ds <= 120:
        return ds - 90
    if 61 <= ds <= 90:
        return ds - 60
    return ds


def _tier(ds: int) -> str:
    if ds <= 60:
        return "clean"
    if ds <= 90:
        return "n1"
    return "n5"


def _file_params(ds: int, fi: int, first_clean_id: int) -> dict[str, Any]:
    """Deterministic per-file parameters shared by the NPZ tree and the table."""
    parent = _clean_parent(ds)
    tier = _tier(ds)
    omega = 2000.0 + 500.0 * ((parent * 7 + fi) % 6)
    b_over_blim = 0.4 + 0.4 * fi
    blim = 2.0
    label = int(b_over_blim >= 1.0)
    unstable = bool(label)

    negative_blim = ds == first_clean_id and fi == 0
    metrics_unavailable = ds == first_clean_id and fi == 1

    if negative_blim:
        # Low-speed integrity defect: stored label=1 but blim is impossible.
        blim = -3.0
        b_over_blim = float("nan")
        label = 1
        unstable = True
        mode_label = "near_boundary_transition"
        noise_score = 0.1
    elif metrics_unavailable:
        mode_label = "metrics_unavailable"
        noise_score = float("nan")
    elif tier == "n5":
        mode_label = "broadband_noise_dominated"
        noise_score = 4.0
    elif tier == "n1":
        mode_label = "mixed_tpf_chatter"
        noise_score = 1.0
    elif unstable:
        mode_label = "chatter_dominated"
        noise_score = 0.1 + 0.05 * fi
    else:
        mode_label = "tpf_dominated_clean"
        noise_score = 0.1 + 0.05 * fi

    b_mm = b_over_blim * blim  # NaN propagates for the negative-blim row
    exclude_supervised = bool(negative_blim or metrics_unavailable)
    physics_usable = bool(np.isfinite(blim) and blim > 0 and np.isfinite(b_over_blim))

    return {
        "ds": ds, "fi": fi, "parent": parent, "tier": tier,
        "noise_std": _NOISE_BY_TIER[tier],
        "omega": omega, "b_over_blim": b_over_blim, "blim": blim, "b_mm": b_mm,
        "label": label, "unstable": unstable, "stem": f"d{ds}_f{fi}",
        "fn": _FN, "tpf": _TPF, "fs": _FS,
        "mode_label": mode_label, "noise_score": noise_score,
        "exclude_supervised": exclude_supervised, "physics_usable": physics_usable,
    }


def _synth_signal(rng: np.random.Generator, length: int, fs: float,
                  *, unstable: bool, noise_std: float) -> np.ndarray:
    t = np.arange(length) / fs
    sig = (
        np.sin(2 * np.pi * _TPF * t)
        + 0.4 * np.sin(2 * np.pi * 2 * _TPF * t)
        + 0.2 * np.sin(2 * np.pi * 3 * _TPF * t)
    )
    if unstable:
        sig = sig + 1.5 * np.sin(2 * np.pi * _FN * t + rng.uniform(0, 2 * np.pi))
    sig = sig + noise_std * rng.standard_normal(length)
    return sig.astype(float)


def make_synth_npz_tree(
    root, *, clean_ids=(1, 2, 3), with_mirrors: bool = True, n_files: int = 4,
    fs: float = _FS, length: int = 4096, seed: int = 0,
) -> pathlib.Path:
    """Write ``dataset_<id>/<stem>.npz`` bundles; return the root path."""
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    first_clean_id = min(clean_ids)

    ids = list(clean_ids)
    if with_mirrors:
        ids += [d + 60 for d in clean_ids] + [d + 90 for d in clean_ids]

    for ds in ids:
        ds_dir = root / f"dataset_{ds}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        for fi in range(n_files):
            p = _file_params(ds, fi, first_clean_id)
            base = _synth_signal(rng, length, fs, unstable=p["unstable"], noise_std=p["noise_std"])
            bundle = {
                "xtpos": (0.001 * base + 1e-4 * p["noise_std"] * rng.standard_normal(length)).astype(float),
                "ytpos": (0.001 * np.roll(base, 5)).astype(float),
                "Forcex": (50.0 * base).astype(float),
                "Forcey": (50.0 * np.roll(base, 3)).astype(float),
            }
            np.savez_compressed(
                ds_dir / f"{p['stem']}.npz",
                fs=np.asarray(fs), label=np.asarray(int(p["label"])),
                omega=np.asarray(p["omega"]), tpf=np.asarray(p["tpf"]), fn=np.asarray(p["fn"]),
                dataset=np.asarray(ds), source_file=np.asarray(f"{p['stem']}.h5"),
                **bundle,
            )
    return root


def _parse_stem(path: pathlib.Path) -> tuple[int, int]:
    m = _STEM_RE.match(path.stem)
    if not m:
        raise ValueError(f"unexpected npz stem: {path.stem!r}")
    return int(m.group(1)), int(m.group(2))


def make_synth_table(npz_root) -> pd.DataFrame:
    """Build the per-file table for a tree written by :func:`make_synth_npz_tree`."""
    npz_root = pathlib.Path(npz_root)
    files = sorted(npz_root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz under {npz_root}")
    clean_present = sorted({_parse_stem(f)[0] for f in files if _parse_stem(f)[0] <= 60})
    first_clean_id = clean_present[0] if clean_present else min(_parse_stem(f)[0] for f in files)

    rows: list[dict[str, Any]] = []
    for f in files:
        ds, fi = _parse_stem(f)
        p = _file_params(ds, fi, first_clean_id)
        if p["exclude_supervised"]:
            weight = 0.0
        else:
            weight = 1.0 / (1.0 + float(p["noise_score"]))
        row = {
            "dataset": ds, "file": f"{p['stem']}.h5", "source_file": f"{p['stem']}.h5",
            "label": int(p["label"]), "b_mm": p["b_mm"], "blim": p["blim"],
            "b_over_blim": p["b_over_blim"], "omega": p["omega"], "tpf": p["tpf"],
            "fn": p["fn"], "fs": p["fs"], "noise_score": p["noise_score"],
            "mode_label": p["mode_label"], "mode_confidence": 0.8,
            "parent_group": p["parent"], "sample_weight": weight,
            "exclude_supervised": p["exclude_supervised"], "physics_usable": p["physics_usable"],
        }
        for k in _KURT_COLS:
            row[k] = 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    pos = df["sample_weight"] > 0
    if pos.any():
        df.loc[pos, "sample_weight"] = df.loc[pos, "sample_weight"] / df.loc[pos, "sample_weight"].mean()
    df["dataset"] = df["dataset"].astype(int)
    return df


def make_synth_folds(table: pd.DataFrame, n_splits: int = 3, seed: int = 0) -> dict:
    """Leakage-safe grouped folds via the production code path."""
    from pri2m.phase3.data_prep import fold_assignments

    return fold_assignments(table, n_splits=n_splits, seed=seed)
