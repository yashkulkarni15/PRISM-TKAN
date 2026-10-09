"""Tests for ``scripts/calibrate_mirror_noise.py``.

A synthetic NPZ tree carries clean datasets 1-2 and their mirrors 61-62 (1 %)
and 91-92 (5 %), built as ``clean + white Gaussian noise`` with a KNOWN
per-channel std (``tier * 0.01 * std(clean channel)``). The calibration must
recover ``rel_sigma_per_unit ~= 0.01`` on every channel, count a missing
mirror file instead of crashing, and count omega/label disagreements between
the paired metas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.calibrate_mirror_noise import main, run

CHANNELS = ("xtpos", "ytpos", "Forcex", "Forcey")
UNIT = 0.01  # 1 % of the clean channel std per tier unit


def _write(path: Path, sig: np.ndarray, *, dataset: int, omega: float, label: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{ch: sig[:, j] for j, ch in enumerate(CHANNELS)},
        fs=np.asarray(10_000.0), label=np.asarray(label), omega=np.asarray(omega),
        tpf=np.asarray(200.0), fn=np.asarray(900.0), dataset=np.asarray(dataset),
        source_file=np.asarray(path.with_suffix(".h5").name),
    )


def _make_tree(root: Path, *, clean=(1, 2), n_files=3, T=20_000, seed=0) -> Path:
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 10_000.0
    scales = np.array([1e-4, 2e-4, 50.0, 80.0])
    for d in clean:
        for k in range(1, n_files + 1):
            base = np.sin(2 * np.pi * (150 + 10 * k) * t)[:, None] * scales
            base = base + 0.3 * np.sin(2 * np.pi * 900 * t + k)[:, None] * scales
            omega = 3000.0 + 100 * k
            label = k % 2
            _write(root / f"dataset_{d}" / f"time{d}_{k}.npz", base,
                   dataset=d, omega=omega, label=label)
            std = base.std(axis=0)
            for off, tier in ((60, 1), (90, 5)):
                noisy = base + rng.standard_normal(base.shape) * (tier * UNIT * std)
                _write(root / f"dataset_{d + off}" / f"time{d + off}_{k}.npz", noisy,
                       dataset=d + off, omega=omega, label=label)
    return root


def test_calibration_recovers_known_noise(tmp_path: Path) -> None:
    root = _make_tree(tmp_path / "npz")
    out = tmp_path / "cal.json"
    res = run(npz_root=str(root), datasets=[1, 2], max_files_per_dataset=None,
              output=str(out))
    summ = res["summary"]
    assert summ["undecided_reason"] is None
    np.testing.assert_allclose(summ["rel_sigma_per_unit"], [UNIT] * 4, rtol=0.05)
    assert summ["linear_ok"] is True
    assert summ["n_pairs"] == {"1": 6, "5": 6}
    assert res["n_missing"] == 0
    assert res["n_meta_disagree"] == 0
    assert res["complete"] is True
    assert res["channels"] == list(CHANNELS)
    sp = res["rel_sigma_spread"]
    np.testing.assert_allclose(sp["5"]["p50"], [5 * UNIT] * 4, rtol=0.05)
    assert all(lo <= hi for lo, hi in zip(sp["1"]["p10"], sp["1"]["p90"]))
    assert len(res["pairs"]) == 12
    row = res["pairs"][0]
    assert set(row) >= {"dataset", "mirror", "file", "tier", "stats"}
    assert row["file"].startswith("time1_")
    # Written file equals the returned dict.
    assert json.loads(out.read_text()) == res


def test_missing_mirror_is_counted_and_skipped(tmp_path: Path) -> None:
    root = _make_tree(tmp_path / "npz", clean=(1,), n_files=2)
    (root / "dataset_91" / "time91_2.npz").unlink()
    res = run(npz_root=str(root), datasets=[1], max_files_per_dataset=None,
              output=str(tmp_path / "c.json"))
    assert res["n_missing"] == 1
    assert len(res["pairs"]) == 3
    assert res["summary"]["n_pairs"] == {"1": 2, "5": 1}


def test_meta_disagreement_is_counted(tmp_path: Path) -> None:
    root = _make_tree(tmp_path / "npz", clean=(1,), n_files=2)
    p = root / "dataset_61" / "time61_1.npz"
    d = dict(np.load(p, allow_pickle=True))
    d["omega"] = np.asarray(float(d["omega"]) + 500.0)
    np.savez_compressed(p, **d)
    res = run(npz_root=str(root), datasets=[1], max_files_per_dataset=None,
              output=str(tmp_path / "c.json"))
    assert res["n_meta_disagree"] == 1
    bad = [r for r in res["pairs"] if not r["meta_agree"]]
    assert len(bad) == 1 and bad[0]["mirror"] == 61


def test_max_files_per_dataset_and_cli(tmp_path: Path) -> None:
    root = _make_tree(tmp_path / "npz", clean=(1, 2), n_files=3)
    out = tmp_path / "sub" / "cal.json"
    rc = main(["--npz-root", str(root), "--datasets", "1-2",
               "--max-files-per-dataset", "1", "--output", str(out)])
    assert rc == 0
    res = json.loads(out.read_text())
    assert len(res["pairs"]) == 4  # 2 datasets x 1 file x 2 tiers
    assert res["datasets"] == [1, 2]


def test_rejects_non_clean_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run(npz_root=str(tmp_path), datasets=[31], max_files_per_dataset=None,
            output=str(tmp_path / "c.json"))
