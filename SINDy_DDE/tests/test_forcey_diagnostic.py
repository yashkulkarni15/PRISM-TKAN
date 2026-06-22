"""Synthetic test for the Forcey per-ladder monotone-fraction diagnostic."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_flags_concentrated_bad_ladder(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    # Five ladders. Ladder D=3 is non-monotone (flat); the other four are
    # strictly increasing. Pooled fraction = 16/20 = 0.80 (< 0.95); excluding
    # the worst 3 ladders (the bad one plus two good ones) leaves 2 good
    # ladders with fraction 1.0 (>= 0.95) so concentrated_in_few_ladders=True.
    bad_clean = 3
    for clean_id in (1, 2, 3, 4, 5):
        for idx, b_mm in enumerate((0.25, 0.50, 0.75, 1.00), start=1):
            for ds, val in (
                (clean_id, 0.1 if clean_id != bad_clean else 0.5),
                (clean_id + 60, 0.3 if clean_id != bad_clean else 0.5),
                (clean_id + 90, 0.6 if clean_id != bad_clean else 0.5),
            ):
                rows.append({
                    "dataset": ds,
                    "file": f"time{ds}_{idx}.h5",
                    "omega": 1000.0 + 100.0 * idx,
                    "b_mm": b_mm,
                    "psd_test_Forcey": val,
                })
    metrics_csv = tmp_path / "noise_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    out_dir = tmp_path / "diag"

    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "diagnose_forcey_near_misses.py"),
         "--noise-metrics-csv", str(metrics_csv),
         "--output-dir", str(out_dir),
         "--metrics-substring", "Forcey"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    summary = json.loads((out_dir / "forcey_summary.json").read_text())
    info = summary["per_metric"]["psd_test_Forcey"]
    assert info["concentrated_in_few_ladders"] is True
    worst = info["worst_3_ladders"]
    assert worst[0]["clean_id"] == bad_clean
    assert worst[0]["monotone_fraction"] == 0.0
