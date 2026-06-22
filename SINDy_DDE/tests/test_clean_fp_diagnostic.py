"""Tests for scripts/diagnose_clean_false_positives.py — TDD first."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the scripts package importable
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Helpers to synthesise tiny fixtures
# ---------------------------------------------------------------------------

def _make_mode_labels(tmp_path: Path) -> Path:
    """Three clean datasets with different broadband rates.
    D=23: 4/5 broadband  (highest FP)
    D=12: 2/5 broadband  (medium)
    D=1:  1/5 broadband  (lowest)
    """
    rows = []
    for dataset, n_bb, n_total in [(23, 4, 5), (12, 2, 5), (1, 1, 5)]:
        for i in range(n_total):
            label = "broadband_noise_dominated" if i < n_bb else "tpf_dominated_clean"
            rows.append(
                {
                    "dataset": dataset,
                    "file": f"time{dataset}_{i}.h5",
                    "source": "clean",
                    "noise_std": 0.0,
                    "mode_label": label,
                }
            )
    # Add one noisy row that should be ignored
    rows.append(
        {
            "dataset": 61,
            "file": "noisy_file.h5",
            "source": "noisy",
            "noise_std": 1.0,
            "mode_label": "broadband_noise_dominated",
        }
    )
    p = tmp_path / "mode_labels.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _make_noise_metrics(tmp_path: Path, datasets=(23, 12)) -> Path:
    """Minimal noise_metrics.csv with two promoted metrics."""
    rng = np.random.default_rng(42)
    rows = []
    for dataset in datasets:
        n = 5
        for i in range(n):
            rows.append(
                {
                    "dataset": dataset,
                    "file": f"time{dataset}_{i}.h5",
                    "metric_A": rng.uniform(0, 2),
                    "metric_B": rng.uniform(0, 2),
                }
            )
    p = tmp_path / "noise_metrics.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _make_promoted_json(tmp_path: Path) -> Path:
    data = {
        "promoted_metrics": ["metric_A", "metric_B"],
        "metric_stats": {
            "metric_A": {
                "family": "wavelet_temporal",
                "clean_median": 0.3,
                "clean_p95": 1.0,
            },
            "metric_B": {
                "family": "psd",
                "clean_median": 0.2,
                "clean_p95": 0.8,
            },
        },
    }
    p = tmp_path / "promoted_metrics.json"
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# Import the module under test (deferred so RED phase shows ImportError or
# AttributeError, not NameError in fixture code)
# ---------------------------------------------------------------------------

def _import_module():
    import importlib
    return importlib.import_module("diagnose_clean_false_positives")


# ---------------------------------------------------------------------------
# CASE 1 – rates only
# ---------------------------------------------------------------------------

class TestComputeCleanFpRates:
    def test_ranks_higher_rate_dataset_first(self, tmp_path):
        mod = _import_module()
        csv_path = _make_mode_labels(tmp_path)
        df = mod.load_mode_labels(csv_path)
        rates_df = mod.compute_clean_fp_rates(df)

        # Should only include clean datasets (1, 12, 23) not noisy (61)
        assert set(rates_df["dataset"]) == {1, 12, 23}

        # Ranked descending by broadband_rate
        assert rates_df.iloc[0]["dataset"] == 23
        assert rates_df.iloc[1]["dataset"] == 12
        assert rates_df.iloc[2]["dataset"] == 1

    def test_rates_values_correct(self, tmp_path):
        mod = _import_module()
        csv_path = _make_mode_labels(tmp_path)
        df = mod.load_mode_labels(csv_path)
        rates_df = mod.compute_clean_fp_rates(df)

        row23 = rates_df[rates_df["dataset"] == 23].iloc[0]
        assert row23["n_files"] == 5
        assert row23["n_broadband"] == 4
        assert abs(row23["broadband_rate"] - 0.8) < 1e-9

    def test_output_csv_written(self, tmp_path):
        mod = _import_module()
        csv_path = _make_mode_labels(tmp_path)
        out_dir = tmp_path / "output"
        mod.main(
            [
                "--mode-labels-csv", str(csv_path),
                "--output-dir", str(out_dir),
            ]
        )
        assert (out_dir / "clean_false_positive_rates.csv").exists()

    def test_summary_json_written_rates_only(self, tmp_path):
        mod = _import_module()
        csv_path = _make_mode_labels(tmp_path)
        out_dir = tmp_path / "output"
        mod.main(
            [
                "--mode-labels-csv", str(csv_path),
                "--output-dir", str(out_dir),
            ]
        )
        summary_path = out_dir / "clean_fp_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary.get("metrics_available") is False


# ---------------------------------------------------------------------------
# CASE 2 – with noise_metrics + promoted_json
# ---------------------------------------------------------------------------

class TestExplainWithMetrics:
    def test_breakdown_csv_produced(self, tmp_path):
        mod = _import_module()
        ml_path = _make_mode_labels(tmp_path)
        nm_path = _make_noise_metrics(tmp_path, datasets=(23, 12))
        pj_path = _make_promoted_json(tmp_path)
        out_dir = tmp_path / "output"

        mod.main(
            [
                "--mode-labels-csv", str(ml_path),
                "--noise-metrics-csv", str(nm_path),
                "--promoted-json", str(pj_path),
                "--output-dir", str(out_dir),
                "--top-n", "2",
            ]
        )

        breakdown = out_dir / "clean_fp_metric_breakdown.csv"
        assert breakdown.exists()
        bdf = pd.read_csv(breakdown)
        assert set(bdf.columns) >= {"dataset", "file", "metric", "value", "clean_p95", "exceeds"}
        # Should contain rows for broadband-flagged files in top-2 datasets
        assert len(bdf) > 0

    def test_breakdown_exceeds_flag_correct(self, tmp_path):
        """Every row where value > clean_p95 must have exceeds==True."""
        mod = _import_module()
        ml_path = _make_mode_labels(tmp_path)
        nm_path = _make_noise_metrics(tmp_path, datasets=(23, 12))
        pj_path = _make_promoted_json(tmp_path)
        out_dir = tmp_path / "output"

        mod.main(
            [
                "--mode-labels-csv", str(ml_path),
                "--noise-metrics-csv", str(nm_path),
                "--promoted-json", str(pj_path),
                "--output-dir", str(out_dir),
                "--top-n", "2",
            ]
        )

        bdf = pd.read_csv(out_dir / "clean_fp_metric_breakdown.csv")
        for _, row in bdf.iterrows():
            expected = row["value"] > row["clean_p95"]
            assert bool(row["exceeds"]) == expected, (
                f"exceeds mismatch at dataset={row['dataset']} file={row['file']} "
                f"metric={row['metric']}: value={row['value']} p95={row['clean_p95']}"
            )

    def test_summary_json_metrics_available_true(self, tmp_path):
        mod = _import_module()
        ml_path = _make_mode_labels(tmp_path)
        nm_path = _make_noise_metrics(tmp_path, datasets=(23, 12))
        pj_path = _make_promoted_json(tmp_path)
        out_dir = tmp_path / "output"

        mod.main(
            [
                "--mode-labels-csv", str(ml_path),
                "--noise-metrics-csv", str(nm_path),
                "--promoted-json", str(pj_path),
                "--output-dir", str(out_dir),
                "--top-n", "2",
            ]
        )

        summary = json.loads((out_dir / "clean_fp_summary.json").read_text())
        assert summary.get("metrics_available") is True
        # Top datasets should appear in per_dataset section
        assert "per_dataset" in summary
        assert len(summary["per_dataset"]) > 0

    def test_summary_json_has_top_driver(self, tmp_path):
        """Each per_dataset entry should carry broadband_rate and top_driver_metric."""
        mod = _import_module()
        ml_path = _make_mode_labels(tmp_path)
        nm_path = _make_noise_metrics(tmp_path, datasets=(23, 12))
        pj_path = _make_promoted_json(tmp_path)
        out_dir = tmp_path / "output"

        mod.main(
            [
                "--mode-labels-csv", str(ml_path),
                "--noise-metrics-csv", str(nm_path),
                "--promoted-json", str(pj_path),
                "--output-dir", str(out_dir),
                "--top-n", "2",
            ]
        )

        summary = json.loads((out_dir / "clean_fp_summary.json").read_text())
        for entry in summary["per_dataset"].values():
            assert "broadband_rate" in entry
            # top_driver_metric may be None when no metric exceeds p95, but key must exist
            assert "top_driver_metric" in entry
