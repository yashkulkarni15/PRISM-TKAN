"""Metadata summaries for Digital Machining Database feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_METADATA_COLUMNS = [
    "dataset",
    "fn",
    "fnty",
    "teeth",
    "damping",
    "ktc",
    "diameter_mm",
    "cut_dir",
    "omega",
    "noise_std",
    "label",
]


@dataclass(frozen=True)
class DatasetCheck:
    dataset: int
    expected: dict[str, float | int]


def load_feature_metadata(path: str | Path, datasets: Iterable[int] | None = None) -> pd.DataFrame:
    """Load one row per file from the feature CSV, restricted to metadata columns."""
    csv_path = Path(path)
    header = pd.read_csv(csv_path, nrows=0)
    cols = [c for c in DEFAULT_METADATA_COLUMNS if c in header.columns]
    df = pd.read_csv(csv_path, usecols=cols)
    if datasets is not None:
        wanted = {int(d) for d in datasets}
        df = df[df["dataset"].astype(int).isin(wanted)].copy()
    return df


def summarize_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """Return one metadata row per dataset with file counts and label balance."""
    aggregations = {
        "fn": "first",
        "fnty": "first",
        "teeth": "first",
        "damping": "first",
        "ktc": "first",
        "diameter_mm": "first",
        "cut_dir": "first",
        "noise_std": "first",
        "omega": ["min", "max", "nunique"],
        "label": "mean",
        "dataset": "size",
    }
    present = {col for col in df.columns}
    safe_aggs = {k: v for k, v in aggregations.items() if k in present}
    out = df.groupby("dataset").agg(safe_aggs)
    out.columns = ["_".join(c).rstrip("_") for c in out.columns.to_flat_index()]
    out = out.rename(columns={"dataset_size": "n_files", "label_mean": "pct_unstable"})
    return out.reset_index()


def verify_dataset_expectations(summary: pd.DataFrame, checks: Iterable[DatasetCheck]) -> list[str]:
    """Return human-readable expectation failures."""
    failures: list[str] = []
    by_dataset = {int(row["dataset"]): row for _, row in summary.iterrows()}
    for check in checks:
        row = by_dataset.get(check.dataset)
        if row is None:
            failures.append(f"dataset {check.dataset}: missing from summary")
            continue
        for key, expected in check.expected.items():
            actual = row.get(key)
            if pd.isna(actual) or float(actual) != float(expected):
                failures.append(
                    f"dataset {check.dataset}: expected {key}={expected}, got {actual}"
                )
    return failures


def phase3_dev_expectations() -> list[DatasetCheck]:
    """Metadata expectations locked by the May 13 planning review."""
    return [
        DatasetCheck(1, {"fn_first": 500.0, "teeth_first": 4, "damping_first": 0.0}),
        DatasetCheck(4, {"ktc_first": 2_000_000_000.0}),
        DatasetCheck(29, {"fn_first": 1500.0, "damping_first": 400_000.0}),
        DatasetCheck(35, {"cut_dir_first": 1}),
        DatasetCheck(39, {"teeth_first": 5}),
        DatasetCheck(91, {"noise_std_first": 5.0}),
    ]
