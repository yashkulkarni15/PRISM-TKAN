#!/usr/bin/env python
"""
build_features_121_155.py
=========================

Regenerate the per-file PHYSICS-metadata columns of the feature CSV for new
datasets (default 121-155). The raw Kaggle dataset ships only the H5 signal
matrices; the user generated ``features_all_datasets.csv`` (clean datasets
1-60) and ``features_noisy_datasets.csv`` (noisy 61-120) from those H5 files,
but those CSVs do NOT cover datasets 121-155. The downstream extraction and
evaluation only need the physics subset (NOT the 60+ spectral / kurtosis
columns), so this script reproduces exactly those columns:

    dataset, file, omega, b_mm, blim, b_over_blim, label,
    fn, ktc, teeth, damping, cut_dir, diameter_mm, tpf, tau, noise_std

Where the numbers come from
---------------------------
Per-file dynamic values (require Great Lakes -- real H5 files):
    * ``omega``, ``b`` : ``read_file.load_time_h5(path)`` returns a dict with
      keys ``"omega"`` (rpm) and ``"b"`` (axial depth of cut, in METRES).
    * ``b_mm``         : ``b * 1000`` (metres -> millimetres). Verified locally:
      the success log records ``b=0.001`` -> features CSV ``b_mm=1.0``.
    * ``blim``         : interpolated from ``stability_boundary{dataset}.h5``
      (an (N, 2) array: col0 = omega, col1 = blim_metres) via ``np.interp`` and
      scaled to millimetres (``* 1000``) so it is comparable to ``b_mm``.
                         Verified locally: features CSV ``blim`` == log
                         ``blim_metres * 1000`` (exact ratio 1000.0).

Per-file derived values (pure arithmetic, no H5):
    * ``b_over_blim`` : ``b_mm / blim`` when ``blim > 0``; otherwise ``NaN``.
                        (A non-positive stability limit means the cut is
                        unconditionally unstable, so the ratio is undefined.)
    * ``label``       : ``int(b_over_blim >= 1.0)`` when ``blim > 0``; otherwise
                        ``1`` (non-positive blim => always unstable).
    * ``tpf``         : ``omega / 60.0 * teeth`` (tooth-passing frequency, Hz).
    * ``tau``         : ``1.0 / tpf`` (tooth-passing period, s).

Per-dataset STATIC params (no H5 -- read from the Excel description file):
    These columns are identical for every file within a dataset. The
    xlsx -> CSV mapping (reverse-engineered and validated against the existing
    1-120 CSVs, see ``tests/test_build_features_121_155.py``) is:

        fn           <- column "fntx (Hz)"
        ktc          <- column "ktc (N/m2)"
        teeth        <- column "m"
        damping      <- column "Cn (N/m)"
        cut_dir      <- column "ud"
        diameter_mm  <- column "d (m)" * 1000
        noise_std    <- column "Noise std (%)"

    The xlsx has several mode-rows per dataset (the "Dataset" column is only
    populated on the first row of each block), so we forward-fill "Dataset",
    coerce every column to numeric, and take the FIRST row of each dataset
    block for these scalar params.

Great Lakes vs local
--------------------
Everything except ``omega``, ``b`` and the stability-boundary H5 is computable
locally. The H5-dependent pieces import ``load_time_h5`` LAZILY (inside
``build``) and read ``stability_boundary*.h5`` with ``h5py``, neither of which
is available off Great Lakes. The static-mapping and pure-derivation helpers
import and run fine anywhere (no h5py / read_file needed).

CLI usage (on Great Lakes)
--------------------------
Validate that the derivation reproduces an existing CSV (proves omega/b/blim
logic before trusting 121-155):

    python scripts/build_features_121_155.py \\
        --data-root  /nfs/turbo/coe-jaabell/datasets/.../versions/10 \\
        --datasets   1-60 \\
        --validate-against data/metadata/features_all_datasets.csv

Generate the new 121-155 CSV:

    python scripts/build_features_121_155.py \\
        --data-root  /nfs/turbo/coe-jaabell/datasets/.../versions/10 \\
        --datasets   121-155 \\
        --out        data/metadata/features_datasets_121_155.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Output schema (physics columns only) and the xlsx -> CSV static mapping
# ----------------------------------------------------------------------

PHYSICS_COLUMNS = [
    "dataset", "file", "omega", "b_mm", "blim", "b_over_blim", "label",
    "fn", "ktc", "teeth", "damping", "cut_dir", "diameter_mm",
    "tpf", "tau", "noise_std",
]

# Static (per-dataset) params: CSV column -> xlsx column. ``diameter_mm`` is a
# special case (xlsx "d (m)" scaled by 1000) handled in ``static_params_from_xlsx``.
STATIC_XLSX_MAP = {
    "fn": "fntx (Hz)",
    "ktc": "ktc (N/m2)",
    "teeth": "m",
    "damping": "Cn (N/m)",
    "cut_dir": "ud",
    "noise_std": "Noise std (%)",
}
DIAMETER_XLSX_COL = "d (m)"  # -> diameter_mm = d (m) * 1000

DEFAULT_XLSX = "data/metadata/Dataset descriptions.xlsx"
DEFAULT_SINDY_SOURCE_DIR = "sindy_pipeline"  # where read_file.py lives on GL


# ----------------------------------------------------------------------
# File enumeration (mirrors sindy_preprocess.list_time_files)
# ----------------------------------------------------------------------

def enumerate_files(data_root, dataset):
    """Return the sorted list of file_y indices present for ``dataset``.

    Per-dataset time files live in ``<data_root>/Dataset {dataset} h5/`` and
    are named ``time{dataset}_{k}.h5``. Returns an empty list if the folder is
    missing (matches ``sindy_preprocess.list_time_files``).
    """
    folder = os.path.join(data_root, f"Dataset {dataset} h5")
    if not os.path.isdir(folder):
        return []
    pattern = re.compile(rf"^time{dataset}_(\d+)\.h5$")
    ys = []
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            ys.append(int(m.group(1)))
    return sorted(ys)


# ----------------------------------------------------------------------
# Static per-dataset params from the Excel description file (NO H5 needed)
# ----------------------------------------------------------------------

def static_params_from_xlsx(xlsx_path, sheet_name="Sheet1"):
    """Build the per-dataset static-param table from the xlsx description.

    The xlsx has multiple mode-rows per dataset; the "Dataset" column is only
    populated on the first row of each block. We forward-fill it, coerce every
    column to numeric, and take the first row per dataset.

    Returns a DataFrame indexed by integer ``dataset`` with columns:
        fn, ktc, teeth, damping, cut_dir, diameter_mm, noise_std
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=0)
    raw["Dataset"] = raw["Dataset"].ffill()
    # Coerce everything we touch to numeric (text -> NaN).
    for col in set(STATIC_XLSX_MAP.values()) | {DIAMETER_XLSX_COL, "Dataset"}:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    first = raw.groupby("Dataset", as_index=False).first()
    first = first.dropna(subset=["Dataset"])

    out = pd.DataFrame(index=first["Dataset"].astype(int))
    out.index.name = "dataset"
    for csv_col, xlsx_col in STATIC_XLSX_MAP.items():
        out[csv_col] = first[xlsx_col].to_numpy()
    out["diameter_mm"] = first[DIAMETER_XLSX_COL].to_numpy() * 1000.0

    # teeth is an integer count in the CSV.
    out["teeth"] = out["teeth"].astype(int)
    # cut_dir is an integer code (1 or 2) in the CSV.
    out["cut_dir"] = out["cut_dir"].astype(int)
    # Column order matching the physics schema's static slice.
    return out[["fn", "ktc", "teeth", "damping", "cut_dir", "diameter_mm", "noise_std"]]


# ----------------------------------------------------------------------
# Pure derivations (NO H5 needed)
# ----------------------------------------------------------------------

def _interp_extrap(x, xp, fp):
    """Linear interpolation with linear *extrapolation* beyond the endpoints.

    ``xp`` must be sorted ascending. Inside the grid this is ``np.interp``;
    outside, it continues the slope of the nearest interval (matching
    ``scipy.interp1d(kind="linear", fill_value="extrapolate")``). This is the
    convention the original feature CSVs used, so low-speed rows below the
    boundary grid get the (often negative) extrapolated limit rather than the
    clamped endpoint ``np.interp`` would return.
    """
    if xp.size == 0:
        return float("nan")
    if xp.size == 1:
        return float(fp[0])
    if x <= xp[0]:
        return float(fp[0] + (fp[1] - fp[0]) / (xp[1] - xp[0]) * (x - xp[0]))
    if x >= xp[-1]:
        return float(fp[-1] + (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) * (x - xp[-1]))
    return float(np.interp(x, xp, fp))


def derive_blim_mm(sb_omega, sb_blim_metres, omega):
    """Interpolate the (metre-scale) stability limit at ``omega`` and convert
    to millimetres (``* 1000``) to match the CSV's ``blim`` units.

    The raw H5 boundary columns are not guaranteed monotonic, so they are
    sorted by omega before interpolating (``np.interp`` needs ascending x), and
    omega values outside the grid are linearly extrapolated -- the convention
    the original feature CSVs used (low-speed rows below the grid extrapolate to
    a negative limit). See ``verify_b_over_blim_provenance.py``.
    """
    o = np.asarray(sb_omega, dtype=float)
    b = np.asarray(sb_blim_metres, dtype=float)
    finite = np.isfinite(o) & np.isfinite(b)
    o, b = o[finite], b[finite]
    order = np.argsort(o)
    o, b = o[order], b[order]
    blim_m = _interp_extrap(float(omega), o, b)
    return blim_m * 1000.0


def derive_row_physics(omega, b, teeth, blim_mm):
    """Compute the derived physics quantities for one file from its primitives.

    Parameters
    ----------
    omega    : spindle speed (rpm)
    b        : axial depth of cut (METRES, as returned by load_time_h5)
    teeth    : number of cutter teeth
    blim_mm  : stability limit at this omega, already in millimetres

    Returns a dict with: b_mm, blim, b_over_blim, label, tpf, tau.

    b_over_blim is NaN and label is 1 when blim_mm <= 0 (a non-positive
    stability limit means the cut is unconditionally unstable). This matches
    the existing CSVs and ``sindy_preprocess`` (``b / blim if blim > 0``).
    """
    b_mm = float(b) * 1000.0
    if blim_mm > 0:
        b_over_blim = b_mm / blim_mm
        label = int(b_over_blim >= 1.0)
    else:
        b_over_blim = float("nan")
        label = 1
    tpf = float(omega) / 60.0 * float(teeth)
    tau = 1.0 / tpf if tpf != 0 else float("nan")
    return {
        "b_mm": b_mm,
        "blim": float(blim_mm),
        "b_over_blim": b_over_blim,
        "label": label,
        "tpf": tpf,
        "tau": tau,
    }


def load_blim_interp(sb_path, omega):
    """Read ``stability_boundary{dataset}.h5`` and return ``blim`` (mm) at omega.

    GREAT LAKES ONLY: needs h5py and the real stability-boundary file. The H5
    holds one dataset of shape (N, 2): col0 = omega, col1 = blim (metres).
    """
    import h5py  # lazy: absent off Great Lakes

    with h5py.File(sb_path, "r") as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"No datasets in {sb_path}")
        arr = np.asarray(f[keys[0]])
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) array in {sb_path}, got shape {arr.shape}")
    return derive_blim_mm(arr[:, 0], arr[:, 1], omega)


# ----------------------------------------------------------------------
# Full build (GREAT LAKES ONLY -- needs real H5 + read_file.load_time_h5)
# ----------------------------------------------------------------------

def parse_datasets(selector):
    """Parse a dataset selector like ``"121-155"``, ``"1-60"`` or ``"5"`` /
    ``"1,3,5"`` into a sorted list of ints.
    """
    out = []
    for part in str(selector).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def build(data_root, datasets, xlsx_path, sindy_source_dir=DEFAULT_SINDY_SOURCE_DIR,
          verbose=True):
    """Regenerate the physics-metadata rows for ``datasets``.

    GREAT LAKES ONLY: imports ``read_file.load_time_h5`` lazily and reads real
    H5 files (time + stability boundary). The static-param table and all
    arithmetic are computed locally; only ``omega``/``b``/``blim`` need H5.

    Returns a DataFrame with the PHYSICS_COLUMNS in order.
    """
    # Lazy import so the module loads locally without read_file / h5py.
    if sindy_source_dir and sindy_source_dir not in sys.path:
        sys.path.insert(0, sindy_source_dir)
    from read_file import load_time_h5  # GL-only module

    static = static_params_from_xlsx(xlsx_path)

    rows = []
    for ds in parse_datasets(datasets) if not isinstance(datasets, (list, tuple)) else sorted(datasets):
        folder = os.path.join(data_root, f"Dataset {ds} h5")
        sb_path = os.path.join(folder, f"stability_boundary{ds}.h5")
        ys = enumerate_files(data_root, ds)
        if not ys:
            if verbose:
                print(f"[ds={ds}] no time files found, skipping")
            continue
        if ds not in static.index:
            if verbose:
                print(f"[ds={ds}] WARNING: no static params in xlsx, skipping")
            continue
        if not os.path.exists(sb_path):
            if verbose:
                print(f"[ds={ds}] WARNING: missing {sb_path}, skipping")
            continue

        ds_static = static.loc[ds]
        teeth = int(ds_static["teeth"])

        for fy in ys:
            time_path = os.path.join(folder, f"time{ds}_{fy}.h5")
            try:
                data = load_time_h5(time_path)
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"[ds={ds} fy={fy}] load_failed: {type(e).__name__}: {e}")
                continue
            if not isinstance(data, dict) or "omega" not in data or "b" not in data:
                if verbose:
                    print(f"[ds={ds} fy={fy}] missing omega/b, skipping")
                continue

            omega = float(data["omega"])
            b = float(data["b"])
            try:
                blim_mm = load_blim_interp(sb_path, omega)
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"[ds={ds} fy={fy}] blim_load_failed: {type(e).__name__}: {e}")
                continue

            derived = derive_row_physics(omega, b, teeth, blim_mm)
            rows.append({
                "dataset": ds,
                "file": f"time{ds}_{fy}.h5",
                "omega": omega,
                "b_mm": derived["b_mm"],
                "blim": derived["blim"],
                "b_over_blim": derived["b_over_blim"],
                "label": derived["label"],
                "fn": float(ds_static["fn"]),
                "ktc": float(ds_static["ktc"]),
                "teeth": teeth,
                "damping": float(ds_static["damping"]),
                "cut_dir": int(ds_static["cut_dir"]),
                "diameter_mm": float(ds_static["diameter_mm"]),
                "tpf": derived["tpf"],
                "tau": derived["tau"],
                "noise_std": float(ds_static["noise_std"]),
            })
        if verbose:
            print(f"[ds={ds}] {len(ys)} files processed")

    df = pd.DataFrame(rows, columns=PHYSICS_COLUMNS)
    return df


# ----------------------------------------------------------------------
# Validation against an existing CSV (GL: proves omega/b/blim derivation)
# ----------------------------------------------------------------------

def validate_against(generated, existing_csv, rtol=1e-6, atol=1e-9, verbose=True):
    """Diff the physics columns of ``generated`` against an existing CSV.

    Joins on (dataset, file), compares each physics column with float tolerance
    (NaN positions must match exactly). Prints mismatches and returns True if
    everything matches.
    """
    existing = pd.read_csv(existing_csv)
    keep = [c for c in PHYSICS_COLUMNS if c in existing.columns]
    existing = existing[keep].copy()

    gen = generated[[c for c in PHYSICS_COLUMNS if c in generated.columns]].copy()
    merged = gen.merge(existing, on=["dataset", "file"], suffixes=("_gen", "_exp"),
                       how="outer", indicator=True)

    only_gen = (merged["_merge"] == "left_only").sum()
    only_exp = (merged["_merge"] == "right_only").sum()
    both = merged[merged["_merge"] == "both"]

    ok = True
    if only_gen or only_exp:
        ok = False
        if verbose:
            print(f"  ROW-COUNT MISMATCH: {only_gen} rows only in generated, "
                  f"{only_exp} rows only in existing")

    compare_cols = [c for c in PHYSICS_COLUMNS if c not in ("dataset", "file")]
    for col in compare_cols:
        gcol, ecol = f"{col}_gen", f"{col}_exp"
        if gcol not in both.columns or ecol not in both.columns:
            continue
        g = pd.to_numeric(both[gcol], errors="coerce").to_numpy()
        e = pd.to_numeric(both[ecol], errors="coerce").to_numpy()
        nan_g, nan_e = np.isnan(g), np.isnan(e)
        nan_mismatch = nan_g != nan_e
        finite = ~nan_g & ~nan_e
        close = np.isclose(g[finite], e[finite], rtol=rtol, atol=atol)
        n_bad = int((~close).sum() + nan_mismatch.sum())
        if n_bad:
            ok = False
            if verbose:
                bad_idx = np.where(nan_mismatch)[0][:5]
                print(f"  COLUMN '{col}': {n_bad} mismatches "
                      f"({int(nan_mismatch.sum())} NaN-position, "
                      f"{int((~close).sum())} value)")
                fin_idx = np.where(finite)[0]
                show = fin_idx[~close][:5]
                for i in show:
                    print(f"      {both.iloc[i]['dataset']}/{both.iloc[i]['file']}: "
                          f"gen={g[i]} exp={e[i]}")

    if verbose:
        print("-" * 60)
        print(f"VALIDATION {'PASS' if ok else 'FAIL'}: compared {len(both)} rows "
              f"across {len(compare_cols)} physics columns")
    return ok


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate physics-metadata CSV rows for new datasets "
                    "(GL: needs real H5 + read_file.load_time_h5)."
    )
    parser.add_argument("--data-root", required=True,
                        help="Folder containing 'Dataset N h5/' subfolders.")
    parser.add_argument("--datasets", default="121-155",
                        help="Selector like '121-155' or '1-60' or '5' or '1,3,5'.")
    parser.add_argument("--xlsx", default=DEFAULT_XLSX,
                        help="Path to 'Dataset descriptions.xlsx'.")
    parser.add_argument("--sindy-source-dir", default=DEFAULT_SINDY_SOURCE_DIR,
                        help="Dir containing read_file.py (GL).")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (omit when only validating).")
    parser.add_argument("--validate-against", default=None,
                        help="Existing CSV to diff the regenerated rows against.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    verbose = not args.quiet
    df = build(args.data_root, args.datasets, args.xlsx,
               sindy_source_dir=args.sindy_source_dir, verbose=verbose)

    if verbose:
        print("=" * 60)
        print(f"Generated {len(df)} rows for datasets {args.datasets}")

    if args.validate_against:
        ok = validate_against(df, args.validate_against, verbose=verbose)
        if not ok:
            return 1

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df.to_csv(args.out, index=False)
        if verbose:
            print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
