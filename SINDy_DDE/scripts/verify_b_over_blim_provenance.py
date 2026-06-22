#!/usr/bin/env python3
"""Verify Phase 2 b_over_blim provenance and optional raw boundary lineage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        default=str(REPO_ROOT / "data" / "metadata" / "features_all_datasets.csv"),
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional raw Digital Machining Database root containing Dataset X h5 folders.",
    )
    parser.add_argument("--sample-rows", type=int, default=160)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    df = pd.read_csv(summary_path)
    b_column = "b_mm" if "b_mm" in df.columns else "b"
    required = {b_column, "blim", "b_over_blim", "label", "dataset", "omega"}
    missing = sorted(required.difference(df.columns))
    if missing:
        print(json.dumps({"ok": False, "missing_columns": missing}, indent=2))
        return 2

    calc = df[b_column].astype(float) / df["blim"].astype(float)
    finite = np.isfinite(calc) & np.isfinite(df["b_over_blim"].astype(float))
    diff = (calc[finite] - df.loc[finite, "b_over_blim"].astype(float)).abs()
    label_expected = (df.loc[finite, "b_over_blim"].astype(float) >= 1.0).astype(int)
    label_actual = df.loc[finite, "label"].astype(int)

    payload: dict[str, Any] = {
        "summary_csv": str(summary_path),
        "rows": int(len(df)),
        "valid_b_over_blim_rows": int(finite.sum()),
        "b_column": b_column,
        "b_over_blim_formula": f"{b_column} / blim",
        "max_abs_formula_error": float(diff.max()) if len(diff) else math.nan,
        "p99_abs_formula_error": float(diff.quantile(0.99)) if len(diff) else math.nan,
        "label_rule": "label = int(b_over_blim >= 1.0)",
        "label_mismatches": int((label_expected != label_actual).sum()),
        "raw_boundary_checked": False,
    }

    if args.data_root:
        payload["raw_boundary_checked"] = True
        payload["raw_boundary"] = check_raw_boundaries(
            df=df,
            data_root=Path(args.data_root),
            sample_rows=args.sample_rows,
            rtol=args.rtol,
            atol=args.atol,
        )

    ok = (
        payload["max_abs_formula_error"] <= args.atol
        and payload["label_mismatches"] == 0
        and (
            not payload["raw_boundary_checked"]
            or payload["raw_boundary"]["mismatches"] == 0
        )
    )
    payload["ok"] = bool(ok)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 1


def check_raw_boundaries(
    df: pd.DataFrame,
    data_root: Path,
    sample_rows: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    import h5py

    if sample_rows > 0 and len(df) > sample_rows:
        sample = df.iloc[np.linspace(0, len(df) - 1, sample_rows).round().astype(int)].copy()
    else:
        sample = df.copy()

    checked = 0
    mismatches = []
    out_of_range = []
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _, row in sample.iterrows():
        dataset = int(row["dataset"])
        omega = float(row["omega"])
        expected_blim = float(row["blim"])
        if dataset not in cache:
            sb_path = data_root / f"Dataset {dataset} h5" / f"stability_boundary{dataset}.h5"
            cache[dataset] = load_stability_boundary(sb_path, expected_blim)
        sb_omega, sb_blim = cache[dataset]
        omega_min = float(np.nanmin(sb_omega))
        omega_max = float(np.nanmax(sb_omega))
        actual_blim = float(np.interp(omega, sb_omega, sb_blim))
        checked += 1
        if omega < omega_min or omega > omega_max:
            out_of_range.append(
                {
                    "dataset": dataset,
                    "file": row.get("file"),
                    "omega": omega,
                    "csv_blim": expected_blim,
                    "boundary_omega_min": omega_min,
                    "boundary_omega_max": omega_max,
                    "clamped_boundary_blim": actual_blim,
                }
            )
            continue
        if not math.isclose(actual_blim, expected_blim, rel_tol=rtol, abs_tol=atol):
            mismatches.append(
                {
                    "dataset": dataset,
                    "file": row.get("file"),
                    "omega": omega,
                    "csv_blim": expected_blim,
                    "boundary_blim": actual_blim,
                    "abs_error": abs(actual_blim - expected_blim),
                }
            )

    return {
        "data_root": str(data_root),
        "checked_rows": checked,
        "unique_datasets": int(sample["dataset"].nunique()) if len(sample) else 0,
        "out_of_boundary_range": len(out_of_range),
        "out_of_boundary_range_preview": out_of_range[:8],
        "mismatches": len(mismatches),
        "mismatch_preview": mismatches[:8],
        "lineage": (
            "blim is interpolated from raw stability_boundary<dataset>.h5 at row omega; "
            "b_over_blim is the CSV cutting depth divided by that interpolated blim. "
            "Rows with omega outside the raw boundary range are reported separately because "
            "np.interp would otherwise clamp to the boundary endpoint."
        ),
    }


def load_stability_boundary(path: Path, expected_blim: float) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"no datasets in {path}")
        arr = np.asarray(handle[keys[0]])
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected shape (N,2) in {path}, got {arr.shape}")
    omega = np.asarray(arr[:, 0], dtype=float)
    blim = np.asarray(arr[:, 1], dtype=float)
    finite = np.isfinite(omega) & np.isfinite(blim)
    omega = omega[finite]
    blim = blim[finite]
    order = np.argsort(omega)
    omega = omega[order]
    blim = blim[order]
    if np.nanmedian(np.abs(blim)) < 0.1 and abs(expected_blim) > 0.1:
        blim = blim * 1000.0
    return omega, blim


if __name__ == "__main__":
    raise SystemExit(main())
