"""Tests for Phase 2.5 clean / 1% / 5% mirror calibration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_calibration_promotes_only_increasing_noise_metrics(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for idx, b_mm in enumerate((0.25, 0.50, 0.75, 1.00), start=1):
        for dataset, inc, dec in (
            (1, 0.10, 0.90),
            (61, 0.30, 0.70),
            (91, 0.60, 0.40),
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "file": f"time{dataset}_{idx}.h5",
                    "omega": 1000.0 + 100.0 * idx,
                    "b_mm": b_mm,
                    "label": int(b_mm >= 1.0),
                    "psd_noise_floor_ratio_Forcex": inc,
                    "psd_off_harmonic_ratio_Forcex": dec,
                }
            )

    metrics_csv = tmp_path / "noise_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    out_dir = tmp_path / "calib"

    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "calibrate_noise_mirror_pairs.py"),
            "--noise-metrics-csv",
            str(metrics_csv),
            "--clean-id",
            "1",
            "--noise1-id",
            "61",
            "--noise5-id",
            "91",
            "--output-dir",
            str(out_dir),
            "--monotone-fraction",
            "1.0",
            "--p-value-threshold",
            "1.0",
        ],
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, f"calibration failed: {res.stderr}\n{res.stdout}"
    payload = json.loads((out_dir / "promoted_metrics.json").read_text())
    promoted = set(payload["promoted_metrics"])
    assert "psd_noise_floor_ratio_Forcex" in promoted
    assert "psd_off_harmonic_ratio_Forcex" not in promoted
    promoted_stats = payload["metric_stats"]["psd_noise_floor_ratio_Forcex"]
    assert promoted_stats["direction"] == "higher_is_noisier"
    # Coverage transparency fields must be present on every promoted metric.
    assert "n_finite" in promoted_stats
    assert "coverage_fraction" in promoted_stats
    assert isinstance(promoted_stats["n_finite"], int)
    assert 0.0 <= promoted_stats["coverage_fraction"] <= 1.0


def test_calibration_rejects_duplicate_pair_keys(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for dataset in (1, 61, 91):
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "file": f"time{dataset}_1.h5",
                    "omega": 1000.0,
                    "b_mm": 0.25,
                    "psd_noise_floor_ratio_Forcex": 0.1,
                },
                {
                    "dataset": dataset,
                    "file": f"time{dataset}_1.h5",
                    "omega": 1000.0,
                    "b_mm": 0.25,
                    "psd_noise_floor_ratio_Forcex": 0.2,
                },
            ]
        )

    metrics_csv = tmp_path / "noise_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    out_dir = tmp_path / "calib"

    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "calibrate_noise_mirror_pairs.py"),
            "--noise-metrics-csv",
            str(metrics_csv),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert res.returncode == 2
    payload = json.loads((out_dir / "promoted_metrics.json").read_text())
    assert payload["reason"] == "duplicate_pair_keys"


def test_pooled_calibration_combines_multiple_ladders(tmp_path: Path) -> None:
    """Pooled mode should auto-detect every (D, D+60, D+90) triple."""
    rows: list[dict[str, object]] = []
    # Three clean datasets 1/2/3, each with a 1% mirror (61/62/63) and a 5%
    # mirror (91/92/93). Each ladder has two operating points.
    ladder_specs = [(1, 61, 91), (2, 62, 92), (3, 63, 93)]
    for clean_id, n1_id, n5_id in ladder_specs:
        for idx, b_mm in enumerate((0.25, 0.50), start=1):
            for dataset, inc in ((clean_id, 0.10), (n1_id, 0.30), (n5_id, 0.60)):
                rows.append({
                    "dataset": dataset,
                    "file": f"time{dataset}_{idx}.h5",
                    "omega": 1000.0 + 100.0 * idx,
                    "b_mm": b_mm,
                    "label": 0,
                    "psd_noise_floor_ratio_Forcex": inc,
                })

    metrics_csv = tmp_path / "noise_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    out_dir = tmp_path / "calib"

    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "calibrate_noise_mirror_pairs.py"),
            "--noise-metrics-csv", str(metrics_csv),
            "--pooled",
            "--output-dir", str(out_dir),
            "--monotone-fraction", "1.0",
            "--p-value-threshold", "0.5",  # n=6 paired triples; loose for the smoke test
        ],
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, f"pooled calibration failed: {res.stderr}\n{res.stdout}"
    payload = json.loads((out_dir / "promoted_metrics.json").read_text())
    assert payload["mode"] == "pooled"
    assert payload["n_ladders"] == 3
    assert payload["n_paired"] == 6  # 3 ladders * 2 operating points
    detected_ladders = {(entry["clean_id"], entry["noise1_id"], entry["noise5_id"])
                        for entry in payload["ladders"]}
    assert detected_ladders == set(ladder_specs)
    assert "psd_noise_floor_ratio_Forcex" in payload["promoted_metrics"]
    # Pooled output should land at a stable file name.
    assert (out_dir / "mirror_calibration_pooled.csv").exists()


def test_pooled_calibration_excludes_clean_ids(tmp_path: Path) -> None:
    """`--exclude-clean-ids` should drop the matching ladder before pairing."""
    rows: list[dict[str, object]] = []
    ladder_specs = [(1, 61, 91), (2, 62, 92), (3, 63, 93)]
    for clean_id, n1_id, n5_id in ladder_specs:
        for idx, b_mm in enumerate((0.25, 0.50), start=1):
            for dataset, inc in ((clean_id, 0.10), (n1_id, 0.30), (n5_id, 0.60)):
                rows.append({
                    "dataset": dataset,
                    "file": f"time{dataset}_{idx}.h5",
                    "omega": 1000.0 + 100.0 * idx,
                    "b_mm": b_mm,
                    "label": 0,
                    "psd_noise_floor_ratio_Forcex": inc,
                })

    metrics_csv = tmp_path / "noise_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    out_dir = tmp_path / "calib"

    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "calibrate_noise_mirror_pairs.py"),
            "--noise-metrics-csv", str(metrics_csv),
            "--pooled",
            "--exclude-clean-ids", "2",
            "--output-dir", str(out_dir),
            "--monotone-fraction", "1.0",
            "--p-value-threshold", "0.5",
        ],
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, f"pooled calibration failed: {res.stderr}\n{res.stdout}"
    payload = json.loads((out_dir / "promoted_metrics.json").read_text())
    assert payload["mode"] == "pooled"
    assert payload["excluded_clean_ids"] == [2]
    detected_clean_ids = {entry["clean_id"] for entry in payload["ladders"]}
    assert 2 not in detected_clean_ids
    assert detected_clean_ids == {1, 3}
    assert payload["n_ladders"] == 2  # reduced by exactly 1 from baseline 3


def test_coverage_gate_blocks_sparse_metric_and_passes_dense(tmp_path: Path) -> None:
    """A metric that is strictly monotone+significant but finite on <80% of paired rows
    must NOT be promoted.  A metric with the same behaviour but >=80% coverage MUST be.
    The output JSON must contain a top-level ``min_coverage`` field.
    """
    import math
    import numpy as np

    rng = np.random.default_rng(42)
    n_total = 20  # paired triple rows

    # Build a base monotone signal: clean < noise1 < noise5
    base_clean = np.linspace(0.1, 0.5, n_total)
    base_n1 = base_clean + 0.2
    base_n5 = base_n1 + 0.2

    rows: list[dict[str, object]] = []
    for i in range(n_total):
        omega = 1000.0 + 50.0 * i
        b_mm = 0.25
        file_stem = f"_{i + 1}"

        # sparse_metric: inject NaN into ~35% of rows (7 rows → 13/20 = 65% finite < 80%)
        sparse_clean = float("nan") if i < 7 else float(base_clean[i])
        sparse_n1 = float("nan") if i < 7 else float(base_n1[i])
        sparse_n5 = float("nan") if i < 7 else float(base_n5[i])

        # dense_metric: all rows finite (100% coverage)
        dense_clean = float(base_clean[i])
        dense_n1 = float(base_n1[i])
        dense_n5 = float(base_n5[i])

        for dataset, sm, dm in (
            (1, sparse_clean, dense_clean),
            (61, sparse_n1, dense_n1),
            (91, sparse_n5, dense_n5),
        ):
            rows.append({
                "dataset": dataset,
                "file": f"time{dataset}{file_stem}.h5",
                "omega": omega,
                "b_mm": b_mm,
                "label": 0,
                "psd_sparse_metric_Forcex": sm,
                "psd_dense_metric_Forcex": dm,
            })

    metrics_csv = tmp_path / "noise_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    out_dir = tmp_path / "calib"

    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "calibrate_noise_mirror_pairs.py"),
            "--noise-metrics-csv", str(metrics_csv),
            "--clean-id", "1",
            "--noise1-id", "61",
            "--noise5-id", "91",
            "--output-dir", str(out_dir),
            "--monotone-fraction", "1.0",
            "--p-value-threshold", "1.0",  # accept any p-value to isolate coverage
            "--min-coverage", "0.8",
        ],
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, f"calibration failed: {res.stderr}\n{res.stdout}"
    payload = json.loads((out_dir / "promoted_metrics.json").read_text())

    promoted = set(payload["promoted_metrics"])
    # sparse_metric has ~65% finite coverage → must be blocked
    assert "psd_sparse_metric_Forcex" not in promoted, (
        "sparse metric (65% coverage) should not be promoted with min_coverage=0.8"
    )
    # dense_metric has 100% coverage → must be promoted
    assert "psd_dense_metric_Forcex" in promoted, (
        "dense metric (100% coverage) should be promoted"
    )

    # The output JSON must record the min_coverage threshold used
    assert "min_coverage" in payload, "promoted_metrics.json must contain 'min_coverage'"
    assert payload["min_coverage"] == 0.8

    # Per-metric stats for promoted dense metric must include coverage_ok
    dense_stats = payload["metric_stats"].get("psd_dense_metric_Forcex", {})
    assert "coverage_ok" in dense_stats, "metric_stats entry must include 'coverage_ok'"
    assert dense_stats["coverage_ok"] is True
