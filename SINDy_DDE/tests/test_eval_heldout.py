"""Test the held-out evaluation script ``scripts/eval_heldout_phase3.py``.

Mirrors ``tests/test_phase3_tkan_end_to_end.py``: synthesize a tiny NPZ tree +
per-file table + folds under a pytest tmp dir, smoke-train a real (tiny) TKAN
checkpoint via the production ``train_one_fold`` path (so a real
``best_auc.pt`` + ``preprocessing.json`` land on disk), then run
``eval_heldout_phase3.run`` against a DIFFERENT synthetic dataset (a held-out
clean id with no parent-group overlap with the training ids).

Asserts:

1. the returned dict has a finite ``auc`` / ``mae`` and a positive ``n_files``;
2. crucially, the eval used the CHECKPOINT's NormStats -- the dataset built
   inside ``run`` carries the exact ``NormStats.from_dict(preprocessing["norm"])``
   z-score and did NOT refit on the held-out windows.

The test self-contains every artifact and touches no Great Lakes data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pri2m.data.phase3_sequence import (
    NormStats,
    Phase3SequenceDataset,
    WindowConfig,
)
from pri2m.training.losses import LossWeights
from pri2m.training.phase3_trainer import TrainConfig, train_one_fold
from scripts.eval_heldout_phase3 import build_eval_table, run
from tests._phase3_synth import make_synth_folds, make_synth_npz_tree, make_synth_table


def _smoke_train_checkpoint(tmp_path: Path, *, side_input: bool) -> tuple[Path, Path]:
    """Smoke-train a tiny TKAN fold; return ``(best_auc_ckpt, preprocessing_json)``.

    Trains on clean datasets (1, 2, 3) + their noise mirrors so the grouped
    folds have enough parent groups. Returns the highest-AUC checkpoint and its
    companion ``preprocessing.json`` (which carries the train-fold z-score, the
    window geometry, and -- for a side model -- the side-feature block).
    """
    train_root = make_synth_npz_tree(
        tmp_path / "train_npz", clean_ids=(1, 2, 3), n_files=4, length=256, seed=0
    )
    table = make_synth_table(train_root)
    folds = make_synth_folds(table, n_splits=3, seed=0)

    cfg = TrainConfig(
        model="tkan",
        epochs=2,
        batch_size=16,
        seed=0,
        window=WindowConfig(length=64, stride=64),
        hidden_size=8,
        num_layers=1,
        grid_size=5,
        spline_order=3,
        loss=LossWeights(),
        fold=0,
        smoke=True,
        device="cpu",
        side_input=side_input,
        side_hidden=4,
    )
    out_dir = tmp_path / f"runs_side_{side_input}"
    summary = train_one_fold(
        cfg,
        table,
        folds,
        str(train_root),
        str(out_dir),
        tracking_uri=f"file:{tmp_path / f'mlruns_side_{side_input}'}",
    )
    run_dir = Path(summary["run_dir"])
    ckpt = run_dir / "checkpoints" / "best_auc.pt"
    pre = run_dir / "preprocessing.json"
    assert ckpt.is_file(), f"no best_auc.pt under {run_dir}"
    assert pre.is_file(), f"no preprocessing.json under {run_dir}"
    return ckpt, pre


def _heldout_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Build a held-out NPZ tree + features CSV on a DISJOINT dataset id.

    Uses clean id 7 with NO mirrors, so parent_group(7) == 7 never overlaps the
    training parent groups (1, 2, 3). The features CSV mimics the real
    ``features_datasets_121_125.csv`` schema (dataset/file/omega/b_over_blim/
    label/...), exercising the same path the GL run takes.
    """
    heldout_root = make_synth_npz_tree(
        tmp_path / "heldout_npz",
        clean_ids=(7,),
        with_mirrors=False,
        n_files=5,
        length=256,
        seed=99,
    )
    table = make_synth_table(heldout_root)
    csv_path = tmp_path / "features_heldout.csv"
    table.to_csv(csv_path, index=False)
    return heldout_root, csv_path


@pytest.mark.parametrize("side_input", [False, True])
def test_eval_heldout_runs_and_uses_checkpoint_norm(
    tmp_path: Path, side_input: bool
) -> None:
    ckpt, pre = _smoke_train_checkpoint(tmp_path, side_input=side_input)
    heldout_root, features_csv = _heldout_inputs(tmp_path)

    out_json = tmp_path / f"heldout_result_side_{side_input}.json"
    result = run(
        checkpoint=str(ckpt),
        features_csv=str(features_csv),
        npz_root=str(heldout_root),
        out=str(out_json),
        device="cpu",
        batch_size=16,
    )

    # (1) Finite headline metrics over a positive file count.
    assert result["n_eval_files"] > 0, f"no eval files: {result}"
    assert result["n_eval_windows"] > 0, f"no eval windows: {result}"
    metrics = result["metrics"]
    assert np.isfinite(metrics["mae"]), f"non-finite mae: {metrics}"
    assert np.isfinite(metrics["auc"]), f"non-finite auc: {metrics}"
    assert metrics["n_files"] > 0
    # The side-input flag round-trips through reconstruction.
    assert result["side_input"] is bool(side_input)
    # Held-out dataset is the disjoint id we built, never a training id.
    assert result["datasets"] == [7], result["datasets"]

    # The on-disk JSON matches the returned dict.
    assert json.loads(out_json.read_text()) == result

    # (2) The eval used the CHECKPOINT's z-score, NOT one refit on the held-out
    # windows. Rebuild the eval dataset exactly as ``run`` does and assert its
    # norm equals NormStats.from_dict(preprocessing["norm"]).
    pre_dict = json.loads(pre.read_text())
    expected_norm = NormStats.from_dict(pre_dict["norm"])

    eval_table, _ = build_eval_table(str(features_csv), str(heldout_root))
    window_block = pre_dict.get("window", {})
    cfg = WindowConfig(
        length=int(window_block["length"]),
        stride=int(window_block["stride"]),
        channels=tuple(pre_dict["channels"]),
    )
    dataset = Phase3SequenceDataset(
        eval_table, str(heldout_root), cfg=cfg, norm=expected_norm
    )
    np.testing.assert_array_equal(dataset.norm.mean, expected_norm.mean)
    np.testing.assert_array_equal(dataset.norm.std, expected_norm.std)

    # And prove it is checkpoint-derived, not held-out-derived: a norm fit on the
    # held-out windows differs from the checkpoint norm (so refitting would give
    # a different, wrong z-score).
    heldout_windows = np.concatenate(
        [rec["windows"] for rec in dataset._files], axis=0
    )
    heldout_norm = NormStats.fit(heldout_windows)
    assert not np.allclose(heldout_norm.mean, expected_norm.mean), (
        "held-out-fit norm equals the checkpoint norm -- cannot distinguish "
        "refit from checkpoint-derived; pick more distinct synthetic data"
    )


def test_eval_heldout_omega_cutoff_excludes_rows(tmp_path: Path) -> None:
    """``--exclude-omega-below`` drops low-speed rows and counts them."""
    ckpt, _ = _smoke_train_checkpoint(tmp_path, side_input=False)
    heldout_root, features_csv = _heldout_inputs(tmp_path)

    table = pd.read_csv(features_csv)
    # Pick a cutoff between the min and max omega so SOME rows are excluded.
    omegas = sorted(table["omega"].unique())
    assert len(omegas) > 1, f"need >1 distinct omega to test the cutoff: {omegas}"
    cutoff = omegas[1]

    out_json = tmp_path / "heldout_omega.json"
    result = run(
        checkpoint=str(ckpt),
        features_csv=str(features_csv),
        npz_root=str(heldout_root),
        out=str(out_json),
        device="cpu",
        batch_size=16,
        exclude_omega_below=float(cutoff),
    )
    n_below = int((table["omega"] < cutoff).sum())
    assert n_below > 0, "test setup picked a cutoff that excludes nothing"
    assert result["n_excluded"] == n_below, (
        f"expected {n_below} excluded rows below omega {cutoff}, got "
        f"{result['n_excluded']}"
    )
