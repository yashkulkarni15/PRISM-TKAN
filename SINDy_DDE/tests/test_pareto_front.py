"""Smoke tests for scripts/pareto_front.py.

The script computes two Pareto axes from the calibrated-noise score and a
band-energy preservation closeness. Tests verify:

1. Closeness-to-1 penalizes overshoot the same as undershoot -- a variant that
   doubles the chatter band scores no higher than one that erases it.
2. Mode-aware filtering protects chatter bands only on chatter rows.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _baseline_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(4):
        rows.append({
            "dataset": 91,
            "file": f"time091_{idx}.h5",
            "omega": 1000.0,
            "b_mm": 0.5 + 0.1 * idx,
            "psd_noise_floor_ratio_Forcex": 0.30,
            "psd_off_harmonic_ratio_Forcex": 0.40,
            "psd_off_harmonic_ratio_Forcey": 0.40,
            "psd_off_harmonic_ratio_xtpos": 0.40,
            "psd_off_harmonic_ratio_ytpos": 0.40,
            "chatter_fn_band_energy_Forcex": 0.10,
            "chatter_sideband_energy_Forcex": 0.05,
        })
    return rows


def _variant_rows(scale_chatter: float, scale_off_harm: float) -> list[dict[str, object]]:
    rows = _baseline_rows()
    for r in rows:
        r["chatter_fn_band_energy_Forcex"] = 0.10 * scale_chatter
        r["chatter_sideband_energy_Forcex"] = 0.05 * scale_chatter
        for col in (
            "psd_off_harmonic_ratio_Forcex", "psd_off_harmonic_ratio_Forcey",
            "psd_off_harmonic_ratio_xtpos", "psd_off_harmonic_ratio_ytpos",
        ):
            r[col] = 0.40 * scale_off_harm
        # Drop the noise-floor metric proportionally so the noise score goes down.
        r["psd_noise_floor_ratio_Forcex"] = 0.10
    return rows


def _calibration() -> dict[str, object]:
    return {
        "mode": "single_triple",
        "promoted_metrics": ["psd_noise_floor_ratio_Forcex"],
        "noise_score_families": ["psd", "wavelet_temporal", "cross_channel"],
        "metric_stats": {
            "psd_noise_floor_ratio_Forcex": {
                "family": "psd",
                "direction": "higher_is_noisier",
                "clean_median": 0.10,
                "clean_mean": 0.10,
                "clean_std": 0.02,
                "clean_p95": 0.14,
                "noise1_mean": 0.20,
                "noise5_mean": 0.30,
                "mean_delta_5pct_minus_clean": 0.20,
                "monotone_fraction": 1.0,
                "mwu_p_value_5pct_vs_clean": 1.0e-9,
            },
        },
    }


def _write(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_closeness_to_one_penalizes_overshoot_and_undershoot_equally(tmp_path: Path) -> None:
    baseline = _write(tmp_path, "baseline.csv", _baseline_rows())
    erased = _write(tmp_path, "erased.csv", _variant_rows(scale_chatter=0.0, scale_off_harm=1.0))
    doubled = _write(tmp_path, "doubled.csv", _variant_rows(scale_chatter=2.0, scale_off_harm=1.0))
    calib = tmp_path / "promoted_metrics.json"
    calib.write_text(json.dumps(_calibration()))
    out_dir = tmp_path / "out"

    res = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "pareto_front.py"),
            "--baseline-metrics-csv", str(baseline),
            "--variant", f"erased={erased}",
            "--variant", f"doubled={doubled}",
            "--calibration-json", str(calib),
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"pareto_front failed: {res.stderr}\n{res.stdout}"

    points = pd.read_csv(out_dir / "pareto_points.csv").set_index("variant")
    erased_chatter = points.loc["erased", "chatter_preservation_closeness"]
    doubled_chatter = points.loc["doubled", "chatter_preservation_closeness"]
    # erased_chatter: ratio=0.0, closeness=max(0, 1 - |0-1|)=0.0
    # doubled_chatter: ratio=2.0, closeness=max(0, 1 - |2-1|)=0.0
    assert abs(erased_chatter - doubled_chatter) < 1.0e-6, (
        f"closeness should penalize overshoot and undershoot equally; "
        f"erased={erased_chatter}, doubled={doubled_chatter}"
    )


def test_mode_aware_skips_chatter_protection_on_broadband_noise_rows(tmp_path: Path) -> None:
    """A variant that erases chatter on broadband-noise files should not be
    penalized when mode-aware filtering is on."""
    baseline = _write(tmp_path, "baseline.csv", _baseline_rows())
    erased = _write(tmp_path, "erased.csv", _variant_rows(scale_chatter=0.0, scale_off_harm=1.0))
    calib = tmp_path / "promoted_metrics.json"
    calib.write_text(json.dumps(_calibration()))

    # Tag all baseline rows as broadband_noise so chatter protection abstains.
    mode_rows = []
    for r in _baseline_rows():
        mode_rows.append({
            "dataset": r["dataset"], "file": r["file"],
            "mode_label": "broadband_noise_dominated",
            "mode_confidence": 0.9,
        })
    mode_csv = _write(tmp_path, "modes.csv", mode_rows)

    out_dir = tmp_path / "out"
    res = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "pareto_front.py"),
            "--baseline-metrics-csv", str(baseline),
            "--variant", f"erased={erased}",
            "--calibration-json", str(calib),
            "--mode-labels-csv", str(mode_csv),
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"pareto_front failed: {res.stderr}\n{res.stdout}"

    points = pd.read_csv(out_dir / "pareto_points.csv").set_index("variant")
    summary = json.loads((out_dir / "pareto_summary.json").read_text())
    assert summary["mode_aware"] is True
    # Chatter rows count should be zero -> closeness is NaN, dropped from the mean.
    assert points.loc["erased", "n_chatter_rows"] == 0
    # The remaining axis (TPF preservation) should drive the score.
    physics = points.loc["erased", "physics_preservation_score"]
    assert physics > 0.5, (
        f"with broadband-noise rows excluded from chatter protection, "
        f"physics_preservation should reflect TPF preservation only; got {physics}"
    )


# ---------------------------------------------------------------------------
# New tests: inner-join enforcement and mismatch tolerance
# ---------------------------------------------------------------------------


def _baseline_rows_abc() -> list[dict[str, object]]:
    """Three rows: (1, a), (1, b), (1, c)."""
    base = []
    for file_id in ("a", "b", "c"):
        base.append({
            "dataset": 1,
            "file": file_id,
            "omega": 1000.0,
            "b_mm": 0.5,
            "psd_noise_floor_ratio_Forcex": 0.30,
            "psd_off_harmonic_ratio_Forcex": 0.40,
            "psd_off_harmonic_ratio_Forcey": 0.40,
            "psd_off_harmonic_ratio_xtpos": 0.40,
            "psd_off_harmonic_ratio_ytpos": 0.40,
            "chatter_fn_band_energy_Forcex": 0.10,
            "chatter_sideband_energy_Forcex": 0.05,
        })
    return base


def _variant_rows_ab() -> list[dict[str, object]]:
    """Variant that only covers (1, a) and (1, b) -- missing (1, c)."""
    rows = []
    for file_id in ("a", "b"):
        rows.append({
            "dataset": 1,
            "file": file_id,
            "omega": 1000.0,
            "b_mm": 0.5,
            "psd_noise_floor_ratio_Forcex": 0.10,  # lower noise
            "psd_off_harmonic_ratio_Forcex": 0.40,
            "psd_off_harmonic_ratio_Forcey": 0.40,
            "psd_off_harmonic_ratio_xtpos": 0.40,
            "psd_off_harmonic_ratio_ytpos": 0.40,
            "chatter_fn_band_energy_Forcex": 0.10,
            "chatter_sideband_energy_Forcex": 0.05,
        })
    return rows


def test_mismatch_beyond_tolerance_exits_nonzero(tmp_path: Path) -> None:
    """Variant missing (1,c) drops 1/3 rows; with default 5 % tolerance this must fail."""
    baseline = _write(tmp_path, "baseline.csv", _baseline_rows_abc())
    variant_ab = _write(tmp_path, "variant_ab.csv", _variant_rows_ab())
    calib = tmp_path / "promoted_metrics.json"
    calib.write_text(json.dumps(_calibration()))
    out_dir = tmp_path / "out"

    res = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "pareto_front.py"),
            "--baseline-metrics-csv", str(baseline),
            "--variant", f"variant_ab={variant_ab}",
            "--calibration-json", str(calib),
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True,
    )
    assert res.returncode != 0, (
        "Expected non-zero exit when row mismatch exceeds default tolerance "
        f"(1/3 ~ 33 %); got returncode={res.returncode}\nstdout={res.stdout}\nstderr={res.stderr}"
    )
    combined = res.stdout + res.stderr
    assert "variant_ab" in combined, (
        f"Error message should name the offending variant; got: {combined!r}"
    )


def test_mismatch_within_permissive_tolerance_passes_and_reports_counts(tmp_path: Path) -> None:
    """With --max-row-mismatch-frac 0.5 a 1/3-row drop is allowed;
    n_baseline/n_variant/n_joined/n_dropped must appear in output."""
    baseline = _write(tmp_path, "baseline.csv", _baseline_rows_abc())
    variant_ab = _write(tmp_path, "variant_ab.csv", _variant_rows_ab())
    calib = tmp_path / "promoted_metrics.json"
    calib.write_text(json.dumps(_calibration()))
    out_dir = tmp_path / "out"

    res = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "pareto_front.py"),
            "--baseline-metrics-csv", str(baseline),
            "--variant", f"variant_ab={variant_ab}",
            "--calibration-json", str(calib),
            "--output-dir", str(out_dir),
            "--max-row-mismatch-frac", "0.5",
        ],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, (
        f"Expected success with permissive tolerance; "
        f"returncode={res.returncode}\nstdout={res.stdout}\nstderr={res.stderr}"
    )

    points = pd.read_csv(out_dir / "pareto_points.csv").set_index("variant")
    row = points.loc["variant_ab"]
    assert int(row["n_baseline"]) == 3, f"n_baseline should be 3; got {row['n_baseline']}"
    assert int(row["n_variant"]) == 2, f"n_variant should be 2; got {row['n_variant']}"
    assert int(row["n_joined"]) == 2, f"n_joined should be 2; got {row['n_joined']}"
    assert int(row["n_dropped"]) == 1, f"n_dropped should be 1; got {row['n_dropped']}"


def test_full_overlap_reports_zero_dropped(tmp_path: Path) -> None:
    """When baseline and variant share all (dataset, file) keys, n_dropped must be 0
    and scores are computed only on the joined set."""
    baseline = _write(tmp_path, "baseline.csv", _baseline_rows())
    variant = _write(tmp_path, "variant.csv", _variant_rows(scale_chatter=0.5, scale_off_harm=1.0))
    calib = tmp_path / "promoted_metrics.json"
    calib.write_text(json.dumps(_calibration()))
    out_dir = tmp_path / "out"

    res = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "pareto_front.py"),
            "--baseline-metrics-csv", str(baseline),
            "--variant", f"half={variant}",
            "--calibration-json", str(calib),
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, (
        f"pareto_front failed: {res.stderr}\n{res.stdout}"
    )

    points = pd.read_csv(out_dir / "pareto_points.csv").set_index("variant")
    row = points.loc["half"]
    assert int(row["n_dropped"]) == 0, f"n_dropped should be 0 on full overlap; got {row['n_dropped']}"
    assert int(row["n_joined"]) == 4, f"n_joined should be 4; got {row['n_joined']}"
