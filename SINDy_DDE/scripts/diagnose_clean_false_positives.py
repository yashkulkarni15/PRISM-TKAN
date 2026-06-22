"""Diagnose CLEAN dataset false positives for broadband_noise_dominated label.

Ranks clean datasets (1-60) by their broadband_noise_dominated rate to surface
false positives.  Optionally enriches the analysis with per-file noise_metrics.csv
to identify which promoted metric is the likely driver.

Usage (rates only):
    python scripts/diagnose_clean_false_positives.py

Usage (with metrics):
    python scripts/diagnose_clean_false_positives.py \\
        --noise-metrics-csv /path/to/noise_metrics.csv \\
        --promoted-json results/pri2m_pooled_v2/promoted_metrics.json

Known offenders expected near the top: D=23, D=12.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_mode_labels(path: str | Path) -> pd.DataFrame:
    """Load mode_labels.csv with low_memory=False to avoid dtype warnings."""
    return pd.read_csv(path, low_memory=False)


def load_noise_metrics(path: str | Path) -> pd.DataFrame:
    """Load noise_metrics.csv."""
    return pd.read_csv(path, low_memory=False)


def load_promoted_json(path: str | Path) -> dict:
    """Load promoted_metrics.json and return the parsed dict."""
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_clean_fp_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-dataset broadband_noise_dominated rates for clean rows only.

    Parameters
    ----------
    df : DataFrame with columns dataset, source, mode_label (at minimum).

    Returns
    -------
    DataFrame sorted descending by broadband_rate with columns:
        dataset, n_files, n_broadband, broadband_rate
    """
    clean = df[df["source"] == "clean"].copy()
    agg = (
        clean.groupby("dataset")
        .apply(
            lambda g: pd.Series(
                {
                    "n_files": len(g),
                    "n_broadband": (g["mode_label"] == "broadband_noise_dominated").sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    agg["broadband_rate"] = agg["n_broadband"] / agg["n_files"]
    agg = agg.sort_values("broadband_rate", ascending=False).reset_index(drop=True)
    return agg


def explain_with_metrics(
    rates_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    nm_df: pd.DataFrame,
    promoted_json: dict,
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """For the top-N FP datasets, explain which promoted metric drives the flag.

    Parameters
    ----------
    rates_df     : output of compute_clean_fp_rates — sorted descending.
    ml_df        : full mode_labels DataFrame.
    nm_df        : noise_metrics DataFrame with columns dataset, file, + metric columns.
    promoted_json: parsed promoted_metrics.json dict.
    top_n        : how many top FP datasets to analyse.

    Returns
    -------
    (breakdown_df, per_dataset_dict)

    breakdown_df columns: dataset, file, metric, value, clean_p95, exceeds
    per_dataset_dict: {str(dataset): {broadband_rate, top_driver_metric, n_broadband_files}}
    """
    promoted = promoted_json.get("promoted_metrics", [])
    metric_stats = promoted_json.get("metric_stats", {})

    top_datasets = rates_df["dataset"].head(top_n).tolist()

    # Identify broadband-flagged files for the top datasets
    broadband_mask = (
        (ml_df["source"] == "clean")
        & (ml_df["dataset"].isin(top_datasets))
        & (ml_df["mode_label"] == "broadband_noise_dominated")
    )
    bb_files = ml_df[broadband_mask][["dataset", "file"]].drop_duplicates()

    # Merge with noise_metrics
    nm_cols = ["dataset", "file"] + [m for m in promoted if m in nm_df.columns]
    nm_sub = nm_df[nm_cols].copy()
    merged = bb_files.merge(nm_sub, on=["dataset", "file"], how="left")

    # Melt to long form for breakdown
    id_cols = ["dataset", "file"]
    metric_cols = [c for c in merged.columns if c not in id_cols]

    if not metric_cols:
        # No promoted metrics found in the noise_metrics file — return empty
        breakdown_df = pd.DataFrame(
            columns=["dataset", "file", "metric", "value", "clean_p95", "exceeds"]
        )
        per_dataset: dict = {}
        for ds in top_datasets:
            row = rates_df[rates_df["dataset"] == ds].iloc[0]
            per_dataset[str(ds)] = {
                "broadband_rate": float(row["broadband_rate"]),
                "n_broadband_files": int(row["n_broadband"]),
                "top_driver_metric": None,
            }
        return breakdown_df, per_dataset

    long = merged.melt(id_vars=id_cols, value_vars=metric_cols, var_name="metric", value_name="value")

    # Attach clean_p95 from promoted_json
    p95_map = {
        m: metric_stats[m]["clean_p95"]
        for m in metric_cols
        if m in metric_stats and "clean_p95" in metric_stats[m]
    }
    long["clean_p95"] = long["metric"].map(p95_map)
    long = long.dropna(subset=["clean_p95"])

    long["exceeds"] = long["value"] > long["clean_p95"]
    breakdown_df = long[["dataset", "file", "metric", "value", "clean_p95", "exceeds"]].copy()
    breakdown_df = breakdown_df.sort_values(["dataset", "file", "metric"]).reset_index(drop=True)

    # Summarise per dataset: most frequent exceeding metric
    per_dataset = {}
    for ds in top_datasets:
        row = rates_df[rates_df["dataset"] == ds].iloc[0]
        ds_break = breakdown_df[
            (breakdown_df["dataset"] == ds) & (breakdown_df["exceeds"])
        ]
        if len(ds_break) > 0:
            counts = Counter(ds_break["metric"])
            top_driver = counts.most_common(1)[0][0]
        else:
            top_driver = None
        per_dataset[str(ds)] = {
            "broadband_rate": float(row["broadband_rate"]),
            "n_broadband_files": int(row["n_broadband"]),
            "top_driver_metric": top_driver,
        }

    return breakdown_df, per_dataset


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rank clean datasets by broadband_noise_dominated rate (false positives)."
    )
    parser.add_argument(
        "--mode-labels-csv",
        default="results/pri2m_pooled_v2/mode_labels.csv",
        help="Path to mode_labels.csv",
    )
    parser.add_argument(
        "--noise-metrics-csv",
        default=None,
        help="Optional: path to full noise_metrics.csv for metric-level explanation",
    )
    parser.add_argument(
        "--promoted-json",
        default="results/pri2m_pooled_v2/promoted_metrics.json",
        help="Path to promoted_metrics.json (needed when noise-metrics-csv is supplied)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase2_5/clean_fp_diagnostic/",
        help="Directory to write output files",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top FP datasets to analyse in metric breakdown",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: compute rates ---
    print(f"[INFO] Loading mode labels from {args.mode_labels_csv}")
    ml_df = load_mode_labels(args.mode_labels_csv)
    rates_df = compute_clean_fp_rates(ml_df)

    rates_out = out_dir / "clean_false_positive_rates.csv"
    rates_df.to_csv(rates_out, index=False)
    print(f"[INFO] Wrote rates → {rates_out}")
    print(rates_df.head(args.top_n).to_string(index=False))

    # --- Step 2 (optional): metric breakdown ---
    metrics_available = False
    per_dataset: dict = {}

    if args.noise_metrics_csv is not None and Path(args.noise_metrics_csv).exists():
        promoted_path = Path(args.promoted_json)
        if not promoted_path.exists():
            print(
                f"[WARN] --promoted-json not found at {promoted_path}; "
                "skipping metric breakdown"
            )
        else:
            print(f"[INFO] Loading noise metrics from {args.noise_metrics_csv}")
            nm_df = load_noise_metrics(args.noise_metrics_csv)
            promoted_json = load_promoted_json(promoted_path)

            breakdown_df, per_dataset = explain_with_metrics(
                rates_df, ml_df, nm_df, promoted_json, top_n=args.top_n
            )

            breakdown_out = out_dir / "clean_fp_metric_breakdown.csv"
            breakdown_df.to_csv(breakdown_out, index=False)
            print(f"[INFO] Wrote metric breakdown → {breakdown_out}")

            metrics_available = True
    else:
        # Populate per_dataset from rates only for top-N
        for _, row in rates_df.head(args.top_n).iterrows():
            per_dataset[str(int(row["dataset"]))] = {
                "broadband_rate": float(row["broadband_rate"]),
                "n_broadband_files": int(row["n_broadband"]),
            }

    # --- Step 3: always write summary JSON ---
    summary = {
        "metrics_available": metrics_available,
        "top_n": args.top_n,
        "per_dataset": per_dataset,
    }
    summary_out = out_dir / "clean_fp_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote summary  → {summary_out}")


if __name__ == "__main__":
    main()
