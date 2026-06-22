"""Local tests for ``scripts/build_features_121_155.py`` (NO H5 required).

Two gates:

1. ``test_static_mapping_reproduces_1_120`` -- THE main correctness gate.
   Loads the two existing feature CSVs (datasets 1-60 and 61-120) and the xlsx
   description via the module's own ``static_params_from_xlsx``, and asserts the
   xlsx -> CSV static mapping (fn, ktc, teeth, damping, cut_dir, diameter_mm,
   noise_std) reproduces the existing CSV values for every one of datasets
   1-120 within float tolerance. These columns are constant within a dataset,
   so this needs zero H5 access.

2. ``test_pure_derivations`` -- hand-computed check of derive_row_physics /
   derive_blim_mm (b_mm, blim interp, b_over_blim, label, tpf, tau).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "build_features_121_155.py"
_XLSX = _REPO / "data" / "metadata" / "Dataset descriptions.xlsx"
_CSV_CLEAN = _REPO / "data" / "metadata" / "features_all_datasets.csv"
_CSV_NOISY = _REPO / "data" / "metadata" / "features_noisy_datasets.csv"

# Static columns the mapping must reproduce (constant within a dataset).
_STATIC_COLS = ["fn", "ktc", "teeth", "damping", "cut_dir", "diameter_mm", "noise_std"]


def _load_module():
    """Import the script as a module by path (it must import without h5py)."""
    spec = importlib.util.spec_from_file_location("build_features_121_155", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _csv_static_per_dataset(path):
    """One row per dataset with the static columns (sorted by file index)."""
    df = pd.read_csv(path, usecols=["dataset", "file"] + _STATIC_COLS)
    df = df.sort_values(["dataset", "file"])
    return df.groupby("dataset", as_index=False).first().set_index("dataset")


# ----------------------------------------------------------------------
# Gate 1: static mapping reproduces datasets 1-120 (strict)
# ----------------------------------------------------------------------

@pytest.mark.skipif(not _XLSX.exists(), reason="xlsx description missing")
def test_static_mapping_reproduces_1_120():
    static = MOD.static_params_from_xlsx(str(_XLSX))

    csv = pd.concat([
        _csv_static_per_dataset(_CSV_CLEAN),
        _csv_static_per_dataset(_CSV_NOISY),
    ])

    # Sanity: we expect to cover all of 1-120.
    expected = set(range(1, 121))
    have = set(int(d) for d in csv.index)
    assert expected.issubset(have), f"CSV missing datasets: {sorted(expected - have)}"

    mismatches = []
    for ds in range(1, 121):
        assert ds in static.index, f"xlsx static table missing dataset {ds}"
        for col in _STATIC_COLS:
            a = float(static.loc[ds, col])
            b = float(csv.loc[ds, col])
            if not np.isclose(a, b, rtol=1e-9, atol=1e-9):
                mismatches.append((ds, col, a, b))

    assert not mismatches, (
        "static xlsx->CSV mapping mismatches (dataset, col, xlsx, csv):\n"
        + "\n".join(f"  ds={d} {c}: xlsx={x} csv={y}" for d, c, x, y in mismatches[:30])
    )


@pytest.mark.skipif(not _XLSX.exists(), reason="xlsx description missing")
def test_static_table_covers_121_155():
    """The new datasets must be present in the xlsx static table."""
    static = MOD.static_params_from_xlsx(str(_XLSX))
    for ds in range(121, 156):
        assert ds in static.index, f"xlsx static table missing new dataset {ds}"
        # teeth and cut_dir must be sensible integers.
        assert int(static.loc[ds, "teeth"]) in (2, 3, 4, 5)
        assert int(static.loc[ds, "cut_dir"]) in (1, 2)


# ----------------------------------------------------------------------
# Gate 2: pure derivations against hand-computed values
# ----------------------------------------------------------------------

def test_pure_derivations_stable():
    # Synthetic stability boundary (omega, blim_metres). At omega=3000 the
    # interp of [2000->0.001, 4000->0.003] gives 0.002 m -> 2.0 mm.
    sb_omega = np.array([2000.0, 4000.0])
    sb_blim_m = np.array([0.001, 0.003])
    omega = 3000.0
    blim_mm = MOD.derive_blim_mm(sb_omega, sb_blim_m, omega)
    assert np.isclose(blim_mm, 2.0)

    # b = 0.001 m -> b_mm = 1.0 ; b_over_blim = 1.0 / 2.0 = 0.5 (stable, label 0)
    teeth = 4
    out = MOD.derive_row_physics(omega, b=0.001, teeth=teeth, blim_mm=blim_mm)
    assert np.isclose(out["b_mm"], 1.0)
    assert np.isclose(out["blim"], 2.0)
    assert np.isclose(out["b_over_blim"], 0.5)
    assert out["label"] == 0
    # tpf = 3000/60 * 4 = 200 Hz ; tau = 1/200 = 0.005 s
    assert np.isclose(out["tpf"], 200.0)
    assert np.isclose(out["tau"], 0.005)


def test_pure_derivations_unstable():
    sb_omega = np.array([2000.0, 4000.0])
    sb_blim_m = np.array([0.001, 0.003])
    blim_mm = MOD.derive_blim_mm(sb_omega, sb_blim_m, 3000.0)  # 2.0 mm

    # b = 0.003 m -> b_mm = 3.0 ; b_over_blim = 3.0/2.0 = 1.5 >= 1 -> label 1
    out = MOD.derive_row_physics(3000.0, b=0.003, teeth=4, blim_mm=blim_mm)
    assert np.isclose(out["b_mm"], 3.0)
    assert np.isclose(out["b_over_blim"], 1.5)
    assert out["label"] == 1

    # Exactly at the boundary: b_over_blim == 1.0 -> label 1 (>=).
    out2 = MOD.derive_row_physics(3000.0, b=0.002, teeth=4, blim_mm=blim_mm)
    assert np.isclose(out2["b_over_blim"], 1.0)
    assert out2["label"] == 1


def test_pure_derivations_negative_blim():
    # Non-positive stability limit -> b_over_blim NaN, label forced to 1.
    out = MOD.derive_row_physics(1000.0, b=0.005, teeth=4, blim_mm=-15.0)
    assert np.isnan(out["b_over_blim"])
    assert out["label"] == 1
    assert np.isclose(out["b_mm"], 5.0)
    assert np.isclose(out["blim"], -15.0)
    # tpf/tau still computed: 1000/60*4 = 66.666... ; tau = 1/tpf
    assert np.isclose(out["tpf"], 1000.0 / 60.0 * 4)
    assert np.isclose(out["tau"], 1.0 / (1000.0 / 60.0 * 4))


def test_blim_interp_extrapolation_clamps():
    # np.interp clamps outside the grid to the endpoints.
    sb_omega = np.array([2000.0, 4000.0])
    sb_blim_m = np.array([0.001, 0.003])
    assert np.isclose(MOD.derive_blim_mm(sb_omega, sb_blim_m, 1000.0), 1.0)  # clamp low
    assert np.isclose(MOD.derive_blim_mm(sb_omega, sb_blim_m, 9000.0), 3.0)  # clamp high


def test_parse_datasets():
    assert MOD.parse_datasets("121-155") == list(range(121, 156))
    assert MOD.parse_datasets("1-60") == list(range(1, 61))
    assert MOD.parse_datasets("5") == [5]
    assert MOD.parse_datasets("1,3,5") == [1, 3, 5]


def test_module_imports_without_h5py():
    # The module must import even when h5py / read_file are absent. It imported
    # at collection time (MOD), so just assert the H5-free API is present.
    assert hasattr(MOD, "static_params_from_xlsx")
    assert hasattr(MOD, "derive_row_physics")
    assert hasattr(MOD, "derive_blim_mm")
    assert hasattr(MOD, "enumerate_files")
    assert hasattr(MOD, "build")
