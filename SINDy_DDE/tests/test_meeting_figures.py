"""Tests for scripts/plot_phase2_5_meeting_figures.py.

Uses synthetic tiny data (a handful of rows spanning tiers + a couple mode
labels) so the tests do NOT depend on the real large CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make sure the scripts/ directory is importable as a module source when the
# script is imported directly (it lives outside src/).
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MODE_LABELS_HEADER = (
    "dataset,file,source,noise_std,mode_label\n"
)

# Rows spanning all 4 tiers (1-30, 31-60, 61-90, 91-120) and several modes.
_MODE_LABELS_ROWS = [
    # tier 1: clean 1-30
    "5,t5_a.h5,clean,0.0,tpf_dominated_clean\n",
    "5,t5_b.h5,clean,0.0,mixed_tpf_chatter\n",
    "5,t5_c.h5,clean,0.0,broadband_noise_dominated\n",
    "20,t20_a.h5,clean,0.0,chatter_dominated\n",
    "20,t20_b.h5,clean,0.0,near_boundary_transition\n",
    # tier 2: clean 31-60 (held-out)
    "35,t35_a.h5,clean,0.0,tpf_dominated_clean\n",
    "35,t35_b.h5,clean,0.0,chatter_dominated\n",
    "50,t50_a.h5,clean,0.0,broadband_noise_dominated\n",
    # tier 3: 1% noise 61-90
    "65,t65_a.h5,noisy,1.0,broadband_noise_dominated\n",
    "65,t65_b.h5,noisy,1.0,mixed_tpf_chatter\n",
    "80,t80_a.h5,noisy,1.0,chatter_dominated\n",
    # tier 4: 5% noise 91-120
    "95,t95_a.h5,noisy,5.0,broadband_noise_dominated\n",
    "95,t95_b.h5,noisy,5.0,broadband_noise_dominated\n",
    "110,t110_a.h5,noisy,5.0,metrics_unavailable\n",
]

_PROMOTED_METRICS_DATA = {
    "promoted_metrics": ["wavelet_mad_sigma_xtpos", "psd_noise_floor_ratio_Forcex"],
    "metric_stats": {
        "wavelet_mad_sigma_xtpos": {
            "family": "wavelet_temporal",
            "clean_std": 1.5e-7,
            "clean_p95": 4.0e-7,
            "mean_delta_5pct_minus_clean": 3.9e-6,
            "monotone_fraction": 0.97,
        },
        "psd_noise_floor_ratio_Forcex": {
            "family": "psd",
            "clean_std": 0.05,
            "clean_p95": 0.9,
            "mean_delta_5pct_minus_clean": 0.4,
            "monotone_fraction": 0.88,
        },
    },
}


@pytest.fixture
def synth_dir(tmp_path: Path) -> Path:
    """Write synthetic CSVs/JSON to tmp_path and return it."""
    mode_csv = tmp_path / "mode_labels.csv"
    mode_csv.write_text(_MODE_LABELS_HEADER + "".join(_MODE_LABELS_ROWS))

    promoted_json = tmp_path / "promoted_metrics.json"
    promoted_json.write_text(json.dumps(_PROMOTED_METRICS_DATA))

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tier_for_dataset():
    """Unit test for the tier helper."""
    from plot_phase2_5_meeting_figures import tier_for_dataset

    assert tier_for_dataset(1) == "clean_1_30"
    assert tier_for_dataset(30) == "clean_1_30"
    assert tier_for_dataset(31) == "clean_31_60"
    assert tier_for_dataset(60) == "clean_31_60"
    assert tier_for_dataset(61) == "noise_1pct"
    assert tier_for_dataset(90) == "noise_1pct"
    assert tier_for_dataset(91) == "noise_5pct"
    assert tier_for_dataset(120) == "noise_5pct"


def test_load_mode_labels(synth_dir: Path):
    """load_mode_labels returns a DataFrame with a 'tier' column."""
    from plot_phase2_5_meeting_figures import load_mode_labels

    df = load_mode_labels(synth_dir / "mode_labels.csv")
    assert "tier" in df.columns
    assert set(df["tier"]).issubset(
        {"clean_1_30", "clean_31_60", "noise_1pct", "noise_5pct"}
    )
    assert len(df) == len(_MODE_LABELS_ROWS)


def test_main_creates_outputs(synth_dir: Path, tmp_path: Path):
    """main() produces the 3 PNGs and the JSON summary."""
    from plot_phase2_5_meeting_figures import main

    out_dir = tmp_path / "figs"
    main(
        [
            "--mode-labels-csv", str(synth_dir / "mode_labels.csv"),
            "--promoted-json", str(synth_dir / "promoted_metrics.json"),
            "--output-dir", str(out_dir),
        ]
    )

    expected_pngs = [
        "fig_mode_distribution_by_tier.png",
        "fig_per_dataset_broadband_rate.png",
        "fig_promoted_metrics.png",
    ]
    for name in expected_pngs:
        p = out_dir / name
        assert p.exists(), f"Missing output: {name}"
        assert p.stat().st_size > 0, f"Empty output: {name}"

    summary_path = out_dir / "figures_summary.json"
    assert summary_path.exists(), "Missing figures_summary.json"
    assert summary_path.stat().st_size > 0

    summary = json.loads(summary_path.read_text())
    assert "tier_mode_counts" in summary
    assert "top5_clean_broadband_rates" in summary
    assert "promoted_metrics" in summary


def test_summary_json_structure(synth_dir: Path, tmp_path: Path):
    """figures_summary.json has the expected keys and non-empty values."""
    from plot_phase2_5_meeting_figures import main

    out_dir = tmp_path / "figs2"
    main(
        [
            "--mode-labels-csv", str(synth_dir / "mode_labels.csv"),
            "--promoted-json", str(synth_dir / "promoted_metrics.json"),
            "--output-dir", str(out_dir),
        ]
    )

    summary = json.loads((out_dir / "figures_summary.json").read_text())
    # tier_mode_counts should have entries for the tiers that appear in synth data
    assert len(summary["tier_mode_counts"]) > 0
    # promoted_metrics list should match what we put in the JSON
    assert "wavelet_mad_sigma_xtpos" in summary["promoted_metrics"]
    # top5 should be a list of [dataset_id, rate] pairs
    assert isinstance(summary["top5_clean_broadband_rates"], list)


def test_with_optional_calibration_csv(synth_dir: Path, tmp_path: Path):
    """Passing --calibration-csv (even minimal) doesn't crash main()."""
    import csv

    from plot_phase2_5_meeting_figures import main

    # Write a tiny calibration CSV with the expected columns
    cal_csv = synth_dir / "mirror_calibration_pooled.csv"
    with cal_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "paired_effect_size", "monotone_fraction", "promoted"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "metric": "wavelet_mad_sigma_xtpos",
                "paired_effect_size": "2.5",
                "monotone_fraction": "0.97",
                "promoted": "True",
            }
        )
        writer.writerow(
            {
                "metric": "psd_noise_floor_ratio_Forcex",
                "paired_effect_size": "1.8",
                "monotone_fraction": "0.88",
                "promoted": "True",
            }
        )

    out_dir = tmp_path / "figs3"
    main(
        [
            "--mode-labels-csv", str(synth_dir / "mode_labels.csv"),
            "--promoted-json", str(synth_dir / "promoted_metrics.json"),
            "--calibration-csv", str(cal_csv),
            "--output-dir", str(out_dir),
        ]
    )

    assert (out_dir / "fig_promoted_metrics.png").exists()
