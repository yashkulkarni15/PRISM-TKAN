#!/usr/bin/env python3
"""Diagnostic for Forcey-family metrics that nearly pass the mirror-ladder gate.

For each metric column whose name contains ``--metrics-substring`` (default
``"Forcey"``), and for every clean ladder D in [1,60] where both D+60 (1% noise)
and D+90 (5% noise) ladders exist in the noise_metrics CSV, this script
reports the per-ladder monotone fraction (strict ``clean < noise1 < noise5``)
across the paired ``(omega, b_mm, mirror_file_key)`` triples.

A metric is flagged ``concentrated_in_few_ladders`` if dropping the three
worst-monotone ladders pushes the pooled monotone_fraction above 0.95.

Pairing reuses ``_discover_ladders``, ``_mirror_file_key``, and
``_pair_triple_for_ladder`` from ``calibrate_noise_mirror_pairs.py`` so the
diagnostic matches the production calibration exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from calibrate_noise_mirror_pairs import (  # noqa: E402
    _discover_ladders,
    _mirror_file_key,  # noqa: F401  (re-exported for parity with production)
    _pair_triple_for_ladder,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-metrics-csv", required=True)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--noisy-summary-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-substring", default="Forcey")
    return parser.parse_args(argv)


def _ladder_stats_for_metric(
    nm: pd.DataFrame,
    metric: str,
    ladders: list[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clean_id, n1_id, n5_id in ladders:
        c, a, b, common = _pair_triple_for_ladder(nm, clean_id, n1_id, n5_id)
        if c is None or not common:
            rows.append({
                "metric": metric,
                "clean_id": int(clean_id),
                "noise1_id": int(n1_id),
                "noise5_id": int(n5_id),
                "n_paired": 0,
                "n_finite": 0,
                "n_strict_monotone_up": 0,
                "monotone_fraction": math.nan,
            })
            continue
        v1 = pd.to_numeric(c.loc[common, metric], errors="coerce").to_numpy()
        v2 = pd.to_numeric(a.loc[common, metric], errors="coerce").to_numpy()
        v3 = pd.to_numeric(b.loc[common, metric], errors="coerce").to_numpy()
        mask = np.isfinite(v1) & np.isfinite(v2) & np.isfinite(v3)
        n_finite = int(mask.sum())
        v1f, v2f, v3f = v1[mask], v2[mask], v3[mask]
        n_up = int(np.sum((v1f < v2f) & (v2f < v3f)))
        frac = (n_up / n_finite) if n_finite else math.nan
        rows.append({
            "metric": metric,
            "clean_id": int(clean_id),
            "noise1_id": int(n1_id),
            "noise5_id": int(n5_id),
            "n_paired": int(len(common)),
            "n_finite": n_finite,
            "n_strict_monotone_up": n_up,
            "monotone_fraction": frac,
        })
    return rows


def _pooled_fraction(per_ladder: list[dict[str, Any]]) -> float:
    n_finite = sum(int(r["n_finite"]) for r in per_ladder)
    n_up = sum(int(r["n_strict_monotone_up"]) for r in per_ladder)
    if n_finite == 0:
        return math.nan
    return n_up / n_finite


def _worst_three(per_ladder: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [r for r in per_ladder if r["n_finite"] > 0]
    scored.sort(key=lambda r: (r["monotone_fraction"], r["clean_id"]))
    return [
        {
            "clean_id": int(r["clean_id"]),
            "noise1_id": int(r["noise1_id"]),
            "noise5_id": int(r["noise5_id"]),
            "n_finite": int(r["n_finite"]),
            "n_strict_monotone_up": int(r["n_strict_monotone_up"]),
            "monotone_fraction": float(r["monotone_fraction"]),
        }
        for r in scored[:3]
    ]


def _pooled_excluding(per_ladder: list[dict[str, Any]], exclude_ids: set[int]) -> float:
    kept = [r for r in per_ladder if int(r["clean_id"]) not in exclude_ids]
    n_finite = sum(int(r["n_finite"]) for r in kept)
    n_up = sum(int(r["n_strict_monotone_up"]) for r in kept)
    if n_finite == 0:
        return math.nan
    return n_up / n_finite


def _json_float(value: object) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    nm = pd.read_csv(args.noise_metrics_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Restrict to ladders with clean D in [1,60] and both mirrors present.
    all_ladders = _discover_ladders(nm, noise1_offset=60, noise5_offset=90)
    ladders = [(c, n1, n5) for (c, n1, n5) in all_ladders if 1 <= c <= 60]

    metric_columns = [
        col for col in nm.columns
        if args.metrics_substring in col
    ]

    per_ladder_all: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "metrics_substring": args.metrics_substring,
        "n_ladders": len(ladders),
        "ladders": [
            {"clean_id": int(c), "noise1_id": int(n1), "noise5_id": int(n5)}
            for (c, n1, n5) in ladders
        ],
        "per_metric": {},
    }

    for metric in metric_columns:
        per_ladder = _ladder_stats_for_metric(nm, metric, ladders)
        per_ladder_all.extend(per_ladder)
        pooled = _pooled_fraction(per_ladder)
        worst = _worst_three(per_ladder)
        exclude_ids = {int(w["clean_id"]) for w in worst}
        pooled_excl = _pooled_excluding(per_ladder, exclude_ids)
        concentrated = bool(
            math.isfinite(pooled)
            and pooled < 0.95
            and math.isfinite(pooled_excl)
            and pooled_excl > 0.95
        )
        summary["per_metric"][metric] = {
            "n_ladders": len(per_ladder),
            "pooled_monotone_fraction": _json_float(pooled),
            "pooled_monotone_fraction_excluding_worst3": _json_float(pooled_excl),
            "worst_3_ladders": worst,
            "worst_ladders": [int(w["clean_id"]) for w in worst],
            "concentrated_in_few_ladders": concentrated,
        }

    per_ladder_df = pd.DataFrame(per_ladder_all, columns=[
        "metric", "clean_id", "noise1_id", "noise5_id",
        "n_paired", "n_finite", "n_strict_monotone_up", "monotone_fraction",
    ])
    per_ladder_csv = out_dir / "forcey_per_ladder_monotone.csv"
    per_ladder_df.to_csv(per_ladder_csv, index=False)

    summary_path = out_dir / "forcey_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Text summary to stdout.
    print(f"diagnose_forcey_near_misses: {len(metric_columns)} metrics x {len(ladders)} ladders")
    for metric, info in summary["per_metric"].items():
        worst_ids = ", ".join(
            f"D={w['clean_id']} ({w['monotone_fraction']:.3f})"
            for w in info["worst_3_ladders"]
        )
        pooled_v = info["pooled_monotone_fraction"]
        pooled_str = f"{pooled_v:.4f}" if pooled_v is not None else "nan"
        flag = " [concentrated]" if info["concentrated_in_few_ladders"] else ""
        print(f"  {metric}: pooled_mono={pooled_str}{flag}; worst: {worst_ids}")
    print(f"Wrote {per_ladder_csv}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
