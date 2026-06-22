"""Unit tests for the Phase 3 sequence data layer (Task 1A).

Builds all inputs from the shared synthetic fixtures so the schema stays
consistent with the rest of the Phase 3 suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tests._phase3_synth import (
    CHANNELS,
    make_synth_folds,
    make_synth_npz_tree,
    make_synth_table,
)

from pri2m.data.phase3_sequence import (
    NormStats,
    Phase3SequenceDataset,
    WindowConfig,
    aggregate_file_predictions,
    build_fold_loaders,
    collate_windows,
    iter_windows,
    load_npz_signals,
    tau_samples,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def synth(tmp_path):
    """A synthetic npz tree, its per-file table, and leakage-safe folds."""
    root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2, 3), n_files=4, seed=0)
    table = make_synth_table(root)
    folds = make_synth_folds(table, n_splits=3, seed=0)
    return root, table, folds


# --------------------------------------------------------------------------- #
# iter_windows
# --------------------------------------------------------------------------- #
def test_iter_windows_count_and_exact_slices():
    # [T=10, C=2] ramp; channel 0 is 0..9, channel 1 is 100..109.
    T, C, L, S = 10, 2, 4, 2
    signal = np.stack(
        [np.arange(T, dtype=float), 100.0 + np.arange(T, dtype=float)], axis=1
    )
    cfg = WindowConfig(length=L, stride=S, channels=("a", "b"))
    windows = iter_windows(signal, cfg)

    # starts at 0,2,4,6 -> exactly 4 windows of [L, C].
    assert windows.shape == (4, L, C)
    # first window == signal[0:4]
    np.testing.assert_array_equal(windows[0], signal[0:4])
    # last window starts at 6 -> signal[6:10]
    np.testing.assert_array_equal(windows[-1], signal[6:10])


def test_iter_windows_shorter_than_length_returns_empty():
    signal = np.zeros((3, 2), dtype=float)
    cfg = WindowConfig(length=4, stride=2, channels=("a", "b"))
    windows = iter_windows(signal, cfg)
    assert windows.shape == (0, 4, 2)


def test_iter_windows_exact_fit_single_window():
    signal = np.arange(8, dtype=float).reshape(4, 2)
    cfg = WindowConfig(length=4, stride=2, channels=("a", "b"))
    windows = iter_windows(signal, cfg)
    assert windows.shape == (1, 4, 2)
    np.testing.assert_array_equal(windows[0], signal)


# --------------------------------------------------------------------------- #
# tau_samples
# --------------------------------------------------------------------------- #
def test_tau_samples_basic():
    assert tau_samples(10000.0, 200.0) == 50


def test_tau_samples_floors_at_one_for_huge_tpf():
    assert tau_samples(10000.0, 1e9) == 1


def test_tau_samples_rounds_to_nearest():
    # 10000/300 = 33.33 -> 33
    assert tau_samples(10000.0, 300.0) == 33


def test_tau_samples_rejects_nonpositive():
    with pytest.raises(ValueError):
        tau_samples(0.0, 200.0)
    with pytest.raises(ValueError):
        tau_samples(10000.0, 0.0)
    with pytest.raises(ValueError):
        tau_samples(-1.0, 200.0)


# --------------------------------------------------------------------------- #
# NormStats
# --------------------------------------------------------------------------- #
def test_normstats_fit_apply_train_is_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    # train windows with distinct per-channel mean/scale.
    train = np.stack(
        [
            5.0 + 2.0 * rng.standard_normal((20, 8)),
            -3.0 + 0.5 * rng.standard_normal((20, 8)),
            10.0 + 4.0 * rng.standard_normal((20, 8)),
        ],
        axis=-1,
    )  # [N=20, L=8, C=3]
    stats = NormStats.fit(train)
    assert stats.mean.shape == (3,)
    assert stats.std.shape == (3,)

    normed = stats.apply(train)
    per_channel_mean = normed.reshape(-1, 3).mean(axis=0)
    per_channel_std = normed.reshape(-1, 3).std(axis=0)
    assert np.all(np.abs(per_channel_mean) < 1e-5)
    np.testing.assert_allclose(per_channel_std, 1.0, atol=1e-5)


def test_normstats_val_normalized_with_train_stats_is_not_zero_mean():
    rng = np.random.default_rng(1)
    train = 5.0 + 2.0 * rng.standard_normal((20, 8, 2))
    # validation has a clearly different mean -> normalizing with train stats
    # must leave a non-zero mean.
    val = 12.0 + 2.0 * rng.standard_normal((20, 8, 2))
    stats = NormStats.fit(train)
    val_normed = stats.apply(val)
    val_mean = val_normed.reshape(-1, 2).mean(axis=0)
    assert np.all(np.abs(val_mean) > 0.5)


def test_normstats_apply_preserves_float32_dtype():
    # A float32 window must normalize in float32 (no float64 intermediate), so
    # the data loader's hot path never round-trips through float64.
    stats = NormStats(mean=np.array([1.0, 2.0]), std=np.array([2.0, 4.0]))
    x32 = np.ones((3, 2), dtype=np.float32)
    out32 = stats.apply(x32)
    assert out32.dtype == np.float32
    # float64 input still normalizes in float64 (back-compat for callers/tests).
    out64 = stats.apply(np.ones((3, 2), dtype=np.float64))
    assert out64.dtype == np.float64
    # value parity within float32 precision.
    np.testing.assert_allclose(out32, out64, rtol=1e-6, atol=1e-6)


def test_build_fold_loaders_accepts_worker_and_pin_options(synth):
    # num_workers/pin_memory are plumbed through without changing the contract.
    root, table, folds = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    train_loader, val_loader, _ = build_fold_loaders(
        table, folds, 1, root, cfg, batch_size=8, seed=0, num_workers=0, pin_memory=False
    )
    batch = next(iter(train_loader))
    assert batch["window"].dtype == torch.float32
    assert next(iter(val_loader))["window"].shape[1] == 1024


def test_normstats_std_floor_avoids_div_by_zero():
    constant = np.full((4, 8, 2), 7.0)
    stats = NormStats.fit(constant)
    # std floored at 1e-8 -> apply must be finite (no inf/nan).
    out = stats.apply(constant)
    assert np.all(np.isfinite(out))
    assert np.all(stats.std >= 1e-8)


def test_normstats_to_dict_is_plain_python_floats():
    # to_dict must emit JSON-serializable plain Python floats, not numpy scalars,
    # so a trained checkpoint's preprocessing.json round-trips through json.
    import json

    stats = NormStats(
        mean=np.array([1.5, -2.0, 3.25], dtype=np.float64),
        std=np.array([0.5, 4.0, 2.0], dtype=np.float64),
    )
    d = stats.to_dict()
    assert set(d) == {"mean", "std"}
    assert d["mean"] == [1.5, -2.0, 3.25]
    assert d["std"] == [0.5, 4.0, 2.0]
    for v in (*d["mean"], *d["std"]):
        assert type(v) is float  # not np.float64
    # the whole dict is JSON round-trippable.
    assert json.loads(json.dumps(d)) == d


def test_normstats_from_dict_reconstructs_and_applies():
    d = {"mean": [1.5, -2.0], "std": [0.5, 4.0]}
    stats = NormStats.from_dict(d)
    assert isinstance(stats, NormStats)
    np.testing.assert_array_equal(stats.mean, np.array([1.5, -2.0]))
    np.testing.assert_array_equal(stats.std, np.array([0.5, 4.0]))
    # reconstructed stats normalize exactly like a hand-built NormStats.
    x = np.array([[2.0, 2.0]], dtype=np.float64)
    np.testing.assert_allclose(stats.apply(x), (x - [1.5, -2.0]) / [0.5, 4.0])


def test_normstats_dict_round_trip_is_exact():
    rng = np.random.default_rng(0)
    original = NormStats.fit(rng.standard_normal((10, 8, 4)))
    restored = NormStats.from_dict(original.to_dict())
    # round-trip is exact on the fitted statistics.
    np.testing.assert_array_equal(restored.mean, original.mean)
    np.testing.assert_array_equal(restored.std, original.std)
    # and re-serializing yields the identical dict.
    assert restored.to_dict() == original.to_dict()


# --------------------------------------------------------------------------- #
# load_npz_signals
# --------------------------------------------------------------------------- #
def test_load_npz_signals_shape_and_meta(synth):
    root, table, _ = synth
    row = table.iloc[0]
    npz_path = root / f"dataset_{int(row.dataset)}" / f"{Path(row.file).stem}.npz"
    signal, meta = load_npz_signals(npz_path, CHANNELS)

    assert signal.ndim == 2
    assert signal.shape[1] == len(CHANNELS)
    assert signal.shape[0] == 4096
    # meta scalars are python scalars, not 0-d arrays.
    for key in ("fs", "label", "omega", "tpf", "fn", "dataset"):
        assert key in meta
        assert np.isscalar(meta[key]) or isinstance(meta[key], (int, float))
    assert isinstance(meta["source_file"], str)
    assert meta["fs"] == 10000.0


def test_load_npz_signals_channel_order(synth):
    root, table, _ = synth
    row = table.iloc[0]
    npz_path = root / f"dataset_{int(row.dataset)}" / f"{Path(row.file).stem}.npz"
    # request a custom channel order and confirm columns follow it.
    chans = ("Forcex", "xtpos")
    signal, _ = load_npz_signals(npz_path, chans)
    raw = np.load(npz_path, allow_pickle=True)
    np.testing.assert_array_equal(signal[:, 0], raw["Forcex"])
    np.testing.assert_array_equal(signal[:, 1], raw["xtpos"])


# --------------------------------------------------------------------------- #
# Phase3SequenceDataset
# --------------------------------------------------------------------------- #
def test_dataset_getitem_dtypes_and_shapes(synth):
    root, table, _ = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    ds = Phase3SequenceDataset(table, root, cfg)
    assert len(ds) > 0

    item = ds[0]
    assert isinstance(item["window"], torch.Tensor)
    assert item["window"].dtype == torch.float32
    assert tuple(item["window"].shape) == (1024, len(CHANNELS))

    assert isinstance(item["b_over_blim"], float)
    assert isinstance(item["label"], float)
    assert isinstance(item["sample_weight"], float)
    assert isinstance(item["tau"], int)
    assert isinstance(item["dataset"], int)
    assert isinstance(item["file"], str)
    assert isinstance(item["parent_group"], int)
    assert isinstance(item["mode_label"], str)
    for key in ("omega", "tpf", "fn", "fs"):
        assert isinstance(item[key], float)

    # tau matches tau_samples(fs, tpf).
    assert item["tau"] == tau_samples(item["fs"], item["tpf"])


def test_dataset_nan_b_over_blim_becomes_zero(synth):
    root, table, _ = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    # d1_f0 has NaN b_over_blim; with supervised_only=False it is included.
    ds = Phase3SequenceDataset(table, root, cfg, supervised_only=False)
    seen = {item["b_over_blim"] for item in (ds[i] for i in range(len(ds)))}
    assert all(np.isfinite(v) for v in seen)  # NaN replaced by 0.0
    # the NaN-row file is present and its windows carry b_over_blim == 0.0.
    nan_windows = [
        ds[i]["b_over_blim"] for i in range(len(ds)) if ds[i]["file"] == "d1_f0.h5"
    ]
    assert nan_windows  # file present
    assert all(v == 0.0 for v in nan_windows)


def test_dataset_supervised_only_drops_excluded_files(synth):
    root, table, _ = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    excluded_stems = {
        Path(f).stem for f in table.loc[table.exclude_supervised, "file"]
    }
    assert excluded_stems  # sanity: fixture injects excluded rows

    ds = Phase3SequenceDataset(table, root, cfg, supervised_only=True)
    files_in_index = {ds[i]["file"] for i in range(len(ds))}
    stems_in_index = {Path(f).stem for f in files_in_index}
    assert excluded_stems.isdisjoint(stems_in_index)

    # and the corresponding zero weights never appear.
    weights = {ds[i]["sample_weight"] for i in range(len(ds))}
    assert 0.0 not in weights

    # without supervised_only the excluded files DO appear.
    ds_all = Phase3SequenceDataset(table, root, cfg, supervised_only=False)
    stems_all = {Path(ds_all[i]["file"]).stem for i in range(len(ds_all))}
    assert excluded_stems.issubset(stems_all)


def test_dataset_applies_normalization(synth):
    root, table, _ = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    ds_raw = Phase3SequenceDataset(table, root, cfg, norm=None)
    # fit norm on all windows then build a normalized dataset.
    all_windows = np.stack([ds_raw[i]["window"].numpy() for i in range(len(ds_raw))])
    stats = NormStats.fit(all_windows)
    ds_norm = Phase3SequenceDataset(table, root, cfg, norm=stats)

    raw0 = ds_raw[0]["window"].numpy()
    norm0 = ds_norm[0]["window"].numpy()
    expected = stats.apply(raw0)
    np.testing.assert_allclose(norm0, expected, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# collate_windows
# --------------------------------------------------------------------------- #
def test_collate_windows_batches_tensors_and_keeps_lists(synth):
    root, table, _ = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    ds = Phase3SequenceDataset(table, root, cfg)
    batch = [ds[0], ds[1], ds[2]]
    out = collate_windows(batch)

    assert tuple(out["window"].shape) == (3, 1024, len(CHANNELS))
    assert out["window"].dtype == torch.float32
    for key in ("b_over_blim", "label", "sample_weight", "tau"):
        assert tuple(out[key].shape) == (3,)
    assert out["tau"].dtype == torch.long
    # metadata stays as python lists.
    for key in ("dataset", "file", "parent_group", "mode_label"):
        assert isinstance(out[key], list)
        assert len(out[key]) == 3


# --------------------------------------------------------------------------- #
# build_fold_loaders
# --------------------------------------------------------------------------- #
def test_build_fold_loaders_no_leakage_and_mirrors_together(synth):
    root, table, folds = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    fold_k = 1  # group 1 (datasets 1, 61, 91) maps here per the fixture.

    train_loader, val_loader, norm = build_fold_loaders(
        table, folds, fold_k, root, cfg, batch_size=8, seed=0
    )

    train_table = train_loader.dataset.table
    val_table = val_loader.dataset.table
    # this raises AssertionError on leakage; reaching the next line proves it passed.
    from pri2m.phase3.splits import assert_no_mirror_leakage

    assert_no_mirror_leakage(train_table, val_table)

    # dataset 1 and its mirrors 61, 91 must all be on the SAME side.
    val_ds = set(val_table["dataset"].tolist())
    train_ds = set(train_table["dataset"].tolist())
    mirror_family = {1, 61, 91}
    assert mirror_family.issubset(val_ds) or mirror_family.issubset(train_ds)
    assert mirror_family.isdisjoint(train_ds) or mirror_family.isdisjoint(val_ds)

    # norm fit on TRAIN only -> returned a NormStats with C channels.
    assert isinstance(norm, NormStats)
    assert norm.mean.shape == (len(CHANNELS),)


def test_build_fold_loaders_norm_fit_on_train_only(synth):
    root, table, folds = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    fold_k = 1
    _, _, norm = build_fold_loaders(table, folds, fold_k, root, cfg, batch_size=8, seed=0)

    # Recompute train windows independently and compare stats.
    group_to_fold = {int(k): int(v) for k, v in folds["group_to_fold"].items()}
    train_rows = table[
        table["parent_group"].astype(int).map(group_to_fold) != fold_k
    ]
    train_ds = Phase3SequenceDataset(train_rows, root, cfg)
    train_windows = np.stack(
        [train_ds[i]["window"].numpy() for i in range(len(train_ds))]
    )
    expected = NormStats.fit(train_windows)
    np.testing.assert_allclose(norm.mean, expected.mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(norm.std, expected.std, rtol=1e-6, atol=1e-6)


def test_build_fold_loaders_deterministic_under_seed(synth):
    root, table, folds = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    fold_k = 0

    tl1, _, _ = build_fold_loaders(table, folds, fold_k, root, cfg, batch_size=8, seed=123)
    tl2, _, _ = build_fold_loaders(table, folds, fold_k, root, cfg, batch_size=8, seed=123)

    b1 = next(iter(tl1))
    b2 = next(iter(tl2))
    # same seed -> identical first-batch ordering (file ids) and tensors.
    assert b1["file"] == b2["file"]
    torch.testing.assert_close(b1["window"], b2["window"])


def test_build_fold_loaders_val_loader_not_shuffled(synth):
    root, table, folds = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    fold_k = 1
    _, val_loader, _ = build_fold_loaders(
        table, folds, fold_k, root, cfg, batch_size=8, seed=0
    )
    # iterating twice yields identical ordering (no shuffle).
    files_a = [f for b in val_loader for f in b["file"]]
    files_b = [f for b in val_loader for f in b["file"]]
    assert files_a == files_b


def test_train_loader_batch_shapes(synth):
    root, table, folds = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    train_loader, _, _ = build_fold_loaders(
        table, folds, 1, root, cfg, batch_size=8, seed=0
    )
    batch = next(iter(train_loader))
    B = batch["window"].shape[0]
    assert tuple(batch["window"].shape) == (B, 1024, len(CHANNELS))
    assert tuple(batch["sample_weight"].shape) == (B,)
    assert B <= 8


# --------------------------------------------------------------------------- #
# aggregate_file_predictions
# --------------------------------------------------------------------------- #
def test_aggregate_file_predictions_means_per_file():
    preds = np.array([1.0, 3.0, 10.0, 20.0, 30.0])
    file_ids = ["a", "a", "b", "b", "b"]
    out = aggregate_file_predictions(preds, file_ids)
    assert set(out.keys()) == {"a", "b"}
    assert out["a"] == pytest.approx(2.0)
    assert out["b"] == pytest.approx(20.0)


def test_aggregate_file_predictions_single_window():
    out = aggregate_file_predictions(np.array([5.0]), ["only"])
    assert out == {"only": pytest.approx(5.0)}


# --------------------------------------------------------------------------- #
# Memory-footprint guards (regression for the 32 G OOM on Great Lakes)
# --------------------------------------------------------------------------- #
def test_dataset_stores_windows_as_float32(synth):
    """Cached windows must be float32, not float64.

    The model consumes float32 (``__getitem__`` casts), so float64 storage
    doubled resident memory for no benefit and was a direct OOM contributor.
    """
    root, table, _ = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    ds = Phase3SequenceDataset(table, root, cfg)
    assert ds._files, "fixture should yield at least one file with windows"
    for rec in ds._files:
        assert rec["windows"].dtype == np.float32


def test_normstats_fit_streaming_matches_fit():
    """Incremental fit must equal fitting on the fully concatenated array."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 1024, 4))
    b = rng.normal(size=(3, 1024, 4)) * 2.0 + 1.0
    streamed = NormStats.fit_streaming([a, b])
    direct = NormStats.fit(np.concatenate([a, b], axis=0))
    np.testing.assert_allclose(streamed.mean, direct.mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(streamed.std, direct.std, rtol=1e-10, atol=1e-12)


def test_build_fold_loaders_loads_each_file_once(synth, monkeypatch):
    """Norm fitting must reuse the loaded train windows, not re-read the disk.

    The old path built a throwaway ``train_for_norm`` dataset (a full second
    load of every train file) just to fit normalization, so each train file was
    decoded twice and held resident twice. After the fix every file is read
    exactly once: train once + val once == the whole table.
    """
    import pri2m.data.phase3_sequence as seqmod

    root, table, folds = synth
    cfg = WindowConfig(length=1024, stride=512, channels=CHANNELS)
    fold_k = 1

    calls = {"n": 0}
    original = seqmod.load_npz_signals

    def counting(npz_path, channels):
        calls["n"] += 1
        return original(npz_path, channels)

    monkeypatch.setattr(seqmod, "load_npz_signals", counting)
    build_fold_loaders(table, folds, fold_k, root, cfg, batch_size=8, seed=0)

    assert calls["n"] == len(table)
