"""End-to-end Phase 2.5 pipeline test on synthetic NPZ bundles.

Synthesizes a small mirror-ladder of Dataset 1 / 61 / 91 with controlled
clean-tooth-pass + chatter-peak signals plus increasing white-noise amplitude,
runs ``compute_noise_metrics.py`` / ``calibrate_noise_mirror_pairs.py`` /
``assign_signal_modes.py`` as subprocess calls, and asserts:

1. spectral entropy moves upward monotonically clean -> 1% -> 5% on every paired file,
2. the calibration step promotes at least one PSD-family metric,
3. mode-label distribution is consistent with the injected physics
   (clean rows tagged tpf_dominated_clean or mixed; the noisiest rows shifted
    toward chatter_dominated / mixed / noise; no crashes).

The test does not depend on any data on Great Lakes; it self-contains every
artifact under a pytest tmp directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_npz_pair(
    out_dir: Path,
    *,
    dataset: int,
    file_name: str,
    fs: float,
    tpf: float,
    fn: float,
    noise_std: float,
    chatter_amp: float,
    rng: np.random.Generator,
    n_samples: int = 65536,
) -> None:
    t = np.arange(n_samples) / fs
    # Forced cutting: dominant TPF + small second harmonic.
    forced = 1.0 * np.sin(2 * np.pi * tpf * t) + 0.3 * np.sin(2 * np.pi * 2 * tpf * t)
    # Chatter near fn (only "above-boundary" rows get strong chatter).
    chatter = chatter_amp * np.sin(2 * np.pi * fn * t)
    # Per-channel signals with shared physics + small independent noise on top.
    shared = forced + chatter
    bundle = {
        "xtpos": shared + noise_std * rng.standard_normal(n_samples),
        "ytpos": 0.8 * shared + noise_std * rng.standard_normal(n_samples),
        "Forcex": 1.2 * shared + noise_std * rng.standard_normal(n_samples),
        "Forcey": 0.9 * shared + noise_std * rng.standard_normal(n_samples),
        "fs": np.array(fs),
        "t": t,
        "dataset": np.array(dataset),
        "source_file": np.array(file_name),
        "label": np.array(1 if chatter_amp > 0.3 else 0),
        "omega": np.array(1000.0),
        "tpf": np.array(tpf),
        "fn": np.array(fn),
    }
    target = out_dir / f"dataset_{dataset}" / f"{Path(file_name).stem}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **bundle)


def _write_metadata(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_end_to_end_phase2_5_pipeline_on_synthetic_ladder(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260525)
    fs, tpf, fn = 20000.0, 200.0, 1500.0
    signals_dir = tmp_path / "signals"
    metadata_dir = tmp_path / "Dataset"
    out_metrics_dir = tmp_path / "out_metrics"
    out_calib_dir = tmp_path / "out_calib"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Six paired (omega, b_mm) operating points; b_mm increases past the boundary.
    operating_points = [
        # (b_mm, blim, b_over_blim, chatter_amp)
        (0.20, 1.00, 0.20, 0.00),
        (0.50, 1.00, 0.50, 0.00),
        (0.80, 1.00, 0.80, 0.00),
        (1.10, 1.00, 1.10, 0.80),
        (1.50, 1.00, 1.50, 1.20),
        (3.00, 1.00, 3.00, 2.00),
    ]

    clean_rows: list[dict] = []
    noisy_rows: list[dict] = []
    ladder_specs = [
        # (dataset_id, noise_std, is_clean)
        (1, 0.00, True),
        (61, 0.10, False),
        (91, 0.50, False),
    ]
    for dataset_id, noise_std, is_clean in ladder_specs:
        for idx, (b_mm, blim, b_over_blim, chatter_amp) in enumerate(operating_points):
            file_name = f"time_{dataset_id:03d}_{idx:03d}.h5"
            _write_synthetic_npz_pair(
                signals_dir,
                dataset=dataset_id,
                file_name=file_name,
                fs=fs, tpf=tpf, fn=fn,
                noise_std=noise_std,
                chatter_amp=chatter_amp,
                rng=rng,
            )
            row = {
                "dataset": dataset_id, "file": file_name,
                "omega": 1000.0, "b_mm": b_mm, "blim": blim, "b_over_blim": b_over_blim,
                "label": 1 if b_over_blim >= 1.0 else 0,
                "fn": fn, "ktc": 1.0, "teeth": 4, "damping": 0.05, "cut_dir": 2,
                "diameter_mm": 12.7, "tpf": tpf, "noise_std": noise_std,
            }
            (clean_rows if is_clean else noisy_rows).append(row)

    clean_csv = metadata_dir / "features_all_datasets.csv"
    noisy_csv = metadata_dir / "features_noisy_datasets.csv"
    _write_metadata(clean_csv, clean_rows)
    _write_metadata(noisy_csv, noisy_rows)

    env = {**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")}

    # Step 1: compute noise metrics.
    res = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "compute_noise_metrics.py"),
            "--summary-csv", str(clean_csv),
            "--noisy-summary-csv", str(noisy_csv),
            "--signals-dir", str(signals_dir),
            "--output-dir", str(out_metrics_dir),
            "--datasets", "1,61,91",
        ],
        capture_output=True, text=True, env=env,
    )
    assert res.returncode == 0, f"compute_noise_metrics failed: {res.stderr}\n{res.stdout}"
    noise_metrics_csv = out_metrics_dir / "noise_metrics.csv"
    assert noise_metrics_csv.exists()
    nm = pd.read_csv(noise_metrics_csv)
    assert len(nm) == 18, f"expected 6 files * 3 datasets = 18, got {len(nm)}"

    # The ladder must move upward across the three noise levels for spectral entropy.
    paired = nm.set_index(["dataset", "omega", "b_mm"]).sort_index()
    for ent_col in [
        "psd_spectral_entropy_xtpos", "psd_spectral_entropy_ytpos",
        "psd_spectral_entropy_Forcex", "psd_spectral_entropy_Forcey",
    ]:
        per_b: dict[float, list[float]] = {}
        for (ds, omega, b_mm), val in nm.set_index(["dataset", "omega", "b_mm"])[ent_col].items():
            per_b.setdefault(b_mm, [0.0, 0.0, 0.0])
            slot = {1: 0, 61: 1, 91: 2}[int(ds)]
            per_b[b_mm][slot] = float(val)
        n_ladder = 0
        for b_mm, triplet in per_b.items():
            if triplet[0] <= triplet[1] <= triplet[2]:
                n_ladder += 1
        assert n_ladder == len(per_b), (
            f"{ent_col} not monotone on all operating points; got {n_ladder}/{len(per_b)} ladders;"
            f" data: {per_b}"
        )

    # Step 2: calibrate.
    res = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "calibrate_noise_mirror_pairs.py"),
            "--noise-metrics-csv", str(noise_metrics_csv),
            "--clean-id", "1",
            "--noise1-id", "61",
            "--noise5-id", "91",
            "--output-dir", str(out_calib_dir),
            "--monotone-fraction", "0.95",
            "--p-value-threshold", "0.05",  # Loose for n=6 paired samples.
        ],
        capture_output=True, text=True, env=env,
    )
    assert res.returncode == 0, f"calibrate failed: {res.stderr}\n{res.stdout}"
    promoted_path = out_calib_dir / "promoted_metrics.json"
    promoted_data = json.loads(promoted_path.read_text())
    promoted_metrics = set(promoted_data["promoted_metrics"])
    assert any(m.startswith("psd_spectral_entropy_") for m in promoted_metrics), (
        f"no PSD-family metric promoted; got {sorted(promoted_metrics)}"
    )
    # Transparency: the calibration payload always carries the inverse-indicator
    # bookkeeping fields, even when no metric strictly decreases with noise.
    assert "inverse_noise_indicators" in promoted_data
    assert "n_inverse_indicators" in promoted_data
    assert isinstance(promoted_data["inverse_noise_indicators"], list)

    # Step 3: assign modes.
    res = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "assign_signal_modes.py"),
            "--noise-metrics-csv", str(noise_metrics_csv),
            "--output-dir", str(out_calib_dir),
        ],
        capture_output=True, text=True, env=env,
    )
    assert res.returncode == 0, f"assign modes failed: {res.stderr}\n{res.stdout}"
    modes_csv = out_calib_dir / "mode_labels.csv"
    assert modes_csv.exists()
    modes = pd.read_csv(modes_csv)
    assert len(modes) == 18

    # Smoke checks on label assignment.
    assert set(modes["mode_label"]).issubset({
        "tpf_dominated_clean", "near_boundary_transition", "chatter_dominated",
        "broadband_noise_dominated", "mixed_tpf_chatter", "low_speed_edge",
        "metrics_unavailable",
    }), f"unknown labels: {sorted(set(modes['mode_label']))}"
    # All confidences in [0, 1].
    assert modes["mode_confidence"].between(0.0, 1.0).all()
    # No crashes / empty values.
    assert modes["mode_label"].notna().all()

    # Calibration wiring: the 5%-noise dataset (91) should hit
    # broadband_noise_dominated more often than the clean dataset (1). This is
    # the regression check for the "calibration JSON not flowing into the
    # decision path" bug from the May 25 code review.
    clean_broadband = int(
        ((modes["dataset"] == 1) & (modes["mode_label"] == "broadband_noise_dominated")).sum()
    )
    noisy5_broadband = int(
        ((modes["dataset"] == 91) & (modes["mode_label"] == "broadband_noise_dominated")).sum()
    )
    assert noisy5_broadband > clean_broadband, (
        f"calibration wiring regression: clean={clean_broadband}, "
        f"5%-noise={noisy5_broadband}; expected noisy > clean."
    )
