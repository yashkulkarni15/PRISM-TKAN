"""Tests for the Exp 3 MiniROCKET comparator (``pri2m.models.minirocket_baseline``).

Most of the comparator is deliberately aeon-free -- input construction, the
delayed channel, window sampling, the ridge path, alpha selection and the
file-level metric -- so it is tested here in the main env. Only the thin aeon
wrapper and the end-to-end runner with the real transform need aeon; those
tests ``importorskip`` it.

The test that carries the most weight is the delayed-channel equality: arm B is
only "the input TKAN sees" if its delayed channels are, element for element,
``pri2m.models.tkan.delayed_input`` stacked over time at the pooled delay.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pri2m.data.phase3_sequence import NormStats
from pri2m.models.minirocket_baseline import (
    DEFAULT_ALPHAS,
    MiniRocketBaseline,
    StreamingNorm,
    build_inputs,
    choose_alpha,
    file_level_metrics,
    fit_ridge_path,
    inner_group_split,
    select_train_windows,
    summarise_delay_tests,
)
from pri2m.models.tkan import delayed_input
from pri2m.models.tkan_model import avg_pool_time, scale_tau
from pri2m.xai.delay_variants import rescale_raw_tau

_ROOT = Path(__file__).resolve().parents[1]


def _windows(N=6, L=64, C=4, seed=0):
    return np.random.default_rng(seed).standard_normal((N, L, C)).astype(np.float32)


# --------------------------------------------------------------------------- #
# build_inputs
# --------------------------------------------------------------------------- #
def test_arm_a_is_pooled_window_channel_first():
    w = _windows()
    x = build_inputs(w, np.full(6, 10), pool=4, arm="A")
    assert x.shape == (6, 4, 16) and x.dtype == np.float32
    ref = avg_pool_time(torch.from_numpy(w), 4).numpy().transpose(0, 2, 1)
    assert np.array_equal(x, ref)


def test_arm_b_delayed_channels_equal_tkan_delayed_input():
    w = _windows()
    tau_raw = np.array([1, 4, 9, 22, 41, 300])  # incl. tau beyond the window
    x = build_inputs(w, tau_raw, pool=4, arm="B")
    assert x.shape == (6, 8, 16)
    pooled = avg_pool_time(torch.from_numpy(w), 4)
    tau_p = scale_tau(torch.as_tensor(tau_raw), 4)
    ref = torch.stack([delayed_input(pooled, t, tau_p) for t in range(pooled.shape[1])],
                      dim=1)  # [B, Lp, C]
    assert np.array_equal(x[:, :4, :], pooled.numpy().transpose(0, 2, 1))
    assert np.array_equal(x[:, 4:, :], ref.numpy().transpose(0, 2, 1))


def test_arm_b_tau_scale_matches_tkan_variant_arithmetic():
    w = _windows()
    tau_raw = np.array([3, 8, 13, 22, 41, 50])
    for s in (0.5, 2.0):
        x = build_inputs(w, tau_raw, pool=4, arm="B", tau_scale=s)
        pooled = avg_pool_time(torch.from_numpy(w), 4)
        tau_p = scale_tau(rescale_raw_tau(torch.as_tensor(tau_raw), s), 4)
        ref = torch.stack([delayed_input(pooled, t, tau_p)
                           for t in range(pooled.shape[1])], dim=1)
        assert np.array_equal(x[:, 4:, :], ref.numpy().transpose(0, 2, 1))


def test_arm_b_zero_delay_zeros_only_delayed_channels():
    w = _windows()
    x = build_inputs(w, np.full(6, 10), pool=4, arm="B", zero_delay=True)
    base = build_inputs(w, np.full(6, 10), pool=4, arm="B")
    assert np.all(x[:, 4:, :] == 0)
    assert np.array_equal(x[:, :4, :], base[:, :4, :])


def test_pool_one_uses_raw_tau():
    w = _windows(L=16)
    x = build_inputs(w, np.full(6, 3), pool=1, arm="B")
    assert np.array_equal(x[:, 4:, 3:], w.transpose(0, 2, 1)[:, :, :-3])
    assert np.all(x[:, 4:, :3] == 0)


def test_arm_a_refuses_delay_arguments():
    w = _windows()
    with pytest.raises(ValueError):
        build_inputs(w, np.full(6, 10), arm="A", zero_delay=True)
    with pytest.raises(ValueError):
        build_inputs(w, np.full(6, 10), arm="A", tau_scale=2.0)
    with pytest.raises(ValueError):
        build_inputs(w, np.full(6, 10), arm="C")
    with pytest.raises(ValueError):
        build_inputs(w, np.full(5, 10), arm="B")


# --------------------------------------------------------------------------- #
# sampling + normalisation
# --------------------------------------------------------------------------- #
def test_select_train_windows_even_and_deterministic():
    assert select_train_windows(3, 8).tolist() == [0, 1, 2]
    assert select_train_windows(8, 8).tolist() == list(range(8))
    idx = select_train_windows(31, 8)
    assert len(idx) == 8 and idx[0] == 0 and idx[-1] == 30
    assert len(set(idx.tolist())) == 8 and np.all(np.diff(idx) > 0)
    assert np.array_equal(idx, select_train_windows(31, 8))
    assert select_train_windows(0, 8).tolist() == []
    assert select_train_windows(5, 1).tolist() == [0]


def test_streaming_norm_matches_normstats_fit():
    rng = np.random.default_rng(3)
    arrays = [rng.normal(5.0, 2.0, (n, 32, 4)).astype(np.float32) for n in (3, 1, 7, 2)]
    sn = StreamingNorm(4)
    for a in arrays:
        sn.update(a)
    sn.update(np.empty((0, 32, 4), dtype=np.float32))
    got = sn.to_normstats()
    ref = NormStats.fit(np.concatenate(arrays))
    np.testing.assert_allclose(got.mean, ref.mean, rtol=1e-12)
    np.testing.assert_allclose(got.std, ref.std, rtol=1e-10)
    with pytest.raises(ValueError):
        StreamingNorm(4).to_normstats()


# --------------------------------------------------------------------------- #
# inner split
# --------------------------------------------------------------------------- #
def test_inner_group_split_never_splits_a_group():
    groups = np.repeat(np.arange(10), 5)
    val = inner_group_split(groups, frac=0.2, seed=0)
    assert val.dtype == bool and val.shape == groups.shape
    vg, tg = set(groups[val]), set(groups[~val])
    assert not (vg & tg)
    assert len(vg) == 2
    assert np.array_equal(val, inner_group_split(groups, frac=0.2, seed=0))


def test_inner_group_split_keeps_one_group_each_side():
    assert inner_group_split(np.array([1, 1, 2, 2]), frac=0.01).sum() == 2
    assert (~inner_group_split(np.array([1, 1, 2, 2]), frac=0.99)).sum() == 2
    with pytest.raises(ValueError):
        inner_group_split(np.array([1, 1, 1]))


# --------------------------------------------------------------------------- #
# ridge
# --------------------------------------------------------------------------- #
def _ridge_data(n=60, p=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32) * rng.uniform(0.5, 5, p)
    X[:, 3] = 7.0  # constant feature must not blow up standardisation
    y = np.where(X[:, 0] + 0.5 * rng.standard_normal(n) > 0, 1.0, -1.0)
    return X, y


def test_ridge_path_matches_sklearn_on_standardised_features():
    from sklearn.linear_model import Ridge

    X, y = _ridge_data()
    alphas = [0.01, 1.0, 100.0]
    path = fit_ridge_path(X, y, alphas, chunk_rows=7)
    mean = X.astype(np.float64).mean(0)
    std = X.astype(np.float64).std(0)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    for i, a in enumerate(alphas):
        sk = Ridge(alpha=a).fit(Z, y)
        np.testing.assert_allclose(path.weights[i], sk.coef_, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(path.intercept, sk.intercept_, rtol=1e-9)
        np.testing.assert_allclose(path.decision(X, i, chunk_rows=5),
                                   sk.predict(Z), rtol=1e-6, atol=1e-9)


def test_ridge_rows_equals_fitting_the_subset():
    X, y = _ridge_data()
    rows = np.arange(0, 60, 3)
    a = fit_ridge_path(X, y, [1.0], rows=rows)
    b = fit_ridge_path(X[rows], y[rows], [1.0])
    np.testing.assert_allclose(a.weights, b.weights, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(a.mean, b.mean)


def test_weighted_ridge_matches_sklearn_sample_weight():
    """TKAN weights its loss by the continuous sample_weight (noisy files count
    less); the ridge must be able to do the same, exactly as sklearn would."""
    from sklearn.linear_model import Ridge

    X, y = _ridge_data()
    w = np.random.default_rng(3).uniform(0.2, 1.6, X.shape[0])
    alphas = [0.01, 1.0, 100.0]
    path = fit_ridge_path(X, y, alphas, chunk_rows=7, sample_weight=w)
    Z = (X - path.mean) / path.std  # standardisation itself stays unweighted
    for i, a in enumerate(alphas):
        sk = Ridge(alpha=a).fit(Z, y, sample_weight=w)
        np.testing.assert_allclose(path.weights[i], sk.coef_, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(path.decision(X, i, chunk_rows=5),
                                   sk.predict(Z), rtol=1e-6, atol=1e-9)


def test_unit_weights_are_bit_identical_to_unweighted():
    X, y = _ridge_data()
    a = fit_ridge_path(X, y, [0.1, 10.0])
    b = fit_ridge_path(X, y, [0.1, 10.0], sample_weight=np.ones(X.shape[0]))
    np.testing.assert_allclose(a.decision(X, 1), b.decision(X, 1), rtol=1e-9, atol=1e-12)


def test_weighted_ridge_rejects_bad_weights():
    X, y = _ridge_data()
    with pytest.raises(ValueError):
        fit_ridge_path(X, y, [1.0], sample_weight=np.full(X.shape[0], -1.0))
    with pytest.raises(ValueError):
        fit_ridge_path(X, y, [1.0], sample_weight=np.ones(3))


def test_ridge_requires_pm1_labels():
    X, y = _ridge_data()
    with pytest.raises(ValueError):
        fit_ridge_path(X, (y > 0).astype(float), [1.0])


# --------------------------------------------------------------------------- #
# file-level metrics + alpha choice
# --------------------------------------------------------------------------- #
def test_file_level_metrics_mean_score_per_file():
    scores = np.array([2.0, 0.0, -1.0, -3.0, 1.0, 1.0])
    labels = np.array([1, 1, 0, 0, 1, 1])
    files = ["a", "a", "b", "b", "c", "c"]
    m = file_level_metrics(scores, labels, files)
    assert m["auc"] == 1.0 and m["acc"] == 1.0
    assert m["n_files"] == 3 and m["n_nonfinite_files"] == 0


def test_file_level_metrics_single_class_is_none_and_nan_excluded():
    m = file_level_metrics(np.array([1.0, 2.0]), np.array([1, 1]), ["a", "b"])
    assert m["auc"] is None
    m = file_level_metrics(np.array([np.nan, 1.0, -1.0]), np.array([1, 1, 0]),
                           ["a", "b", "c"])
    assert m["n_nonfinite_files"] == 1 and m["n_files"] == 3 and m["auc"] == 1.0
    m = file_level_metrics(np.array([]), np.array([]), [])
    assert m["auc"] is None and m["acc"] is None and m["n_files"] == 0


def test_choose_alpha_picks_best_inner_auc_and_reports_grid():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(10), 20)
    files = np.repeat(np.arange(50), 4)
    y = np.repeat(np.where(rng.random(50) > 0.5, 1.0, -1.0), 4)
    X = (rng.standard_normal((200, 8)) + 0.8 * y[:, None] * np.array([1, 1, 0, 0, 0, 0, 0, 0])
         ).astype(np.float32)
    sel = choose_alpha(X, y, files, groups, [0.1, 10.0, 1e7], frac=0.2, seed=0)
    assert sel["alpha"] in (0.1, 10.0, 1e7)
    assert len(sel["inner_auc"]) == 3
    best = max(a for a in sel["inner_auc"] if a is not None)
    assert sel["inner_auc"][sel["alpha_index"]] == best
    assert sel["n_inner_val_groups"] == 2 and sel["n_inner_train_groups"] == 8
    assert isinstance(sel["at_grid_edge"], bool)


def test_choose_alpha_breaks_ties_toward_more_regularisation():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(5), 8)
    files = np.arange(40)
    y = np.where(np.arange(40) % 2 == 0, 1.0, -1.0)
    X = np.c_[y * 5.0, rng.standard_normal(40)].astype(np.float32)  # perfectly separable
    sel = choose_alpha(X, y, files, groups, [0.01, 0.1, 1.0], frac=0.4, seed=0)
    assert sel["alpha"] == 1.0


def test_choose_alpha_refuses_when_inner_val_is_single_class():
    groups = np.array([0, 0, 1, 1])
    y = np.array([1.0, 1.0, -1.0, -1.0])  # each group single-class
    X = np.arange(8, dtype=np.float32).reshape(4, 2)
    with pytest.raises(ValueError, match="single class"):
        choose_alpha(X, y, np.arange(4), groups, [1.0], frac=0.5, seed=0)


def test_default_alpha_grid_spans_large_n_regime():
    assert min(DEFAULT_ALPHAS) <= 1e-3 and max(DEFAULT_ALPHAS) >= 1e6


# --------------------------------------------------------------------------- #
# delay summary
# --------------------------------------------------------------------------- #
def test_summarise_delay_tests_drops_and_floor():
    metrics = {
        "B": {"auc": 0.90}, "B_zero_delay": {"auc": 0.70},
        "B_tau_x0.5": {"auc": 0.895}, "B_tau_x2": {"auc": None},
    }
    out = summarise_delay_tests(metrics, baseline="B", floor=0.0092)
    assert out["B_zero_delay"]["drop"] == pytest.approx(0.2)
    assert out["B_zero_delay"]["exceeds_floor"] is True
    assert out["B_tau_x0.5"]["exceeds_floor"] is False
    assert out["B_tau_x2"] == {"auc": None, "drop": None, "exceeds_floor": None}
    assert "B" not in out


# --------------------------------------------------------------------------- #
# MiniRocketBaseline with an injected (aeon-free) transform
# --------------------------------------------------------------------------- #
class _FakeTransform:
    """Stand-in for aeon's MiniRocket: per-channel mean + std features."""

    def __init__(self):
        self.fitted = False

    def fit(self, X):
        self.fitted = True
        return self

    def transform(self, X):
        assert self.fitted
        return np.concatenate([X.mean(-1), X.std(-1)], axis=1).astype(np.float32)


def _separable_inputs(n_files=40, per_file=3, seed=0):
    rng = np.random.default_rng(seed)
    labels = (np.arange(n_files) % 2).astype(int)
    X, y, g, f = [], [], [], []
    for i, lab in enumerate(labels):
        for _ in range(per_file):
            X.append(rng.standard_normal((4, 32)) + (1.5 * lab))
            y.append(lab)
            g.append(i // 4)
            f.append(f"file{i}")
    return np.asarray(X, np.float32), np.asarray(y), np.asarray(g), np.asarray(f)


def test_baseline_with_fake_transform_fits_and_scores():
    X, y, g, f = _separable_inputs()
    clf = MiniRocketBaseline(transformer_factory=_FakeTransform, chunk_size=7,
                             alphas=[0.1, 10.0])
    clf.fit(X, y, g, f)
    assert clf.alpha_ in (0.1, 10.0)
    s = clf.decision_function(X)
    assert s.shape == (len(X),)
    m = file_level_metrics(s, y, f)
    assert m["auc"] > 0.95
    assert clf.n_features_ == 8
    assert clf.selection_["n_inner_val_groups"] >= 1


# --------------------------------------------------------------------------- #
# aeon-backed
# --------------------------------------------------------------------------- #
def test_aeon_minirocket_wrapper_end_to_end():
    pytest.importorskip("aeon")
    X, y, g, f = _separable_inputs(per_file=2)
    clf = MiniRocketBaseline(n_kernels=168, seed=0, n_jobs=1, chunk_size=13,
                             alphas=[1.0, 100.0])
    clf.fit(X, y, g, f)
    assert clf.n_features_ == 168
    s1 = clf.decision_function(X)
    m = file_level_metrics(s1, y, f)
    assert m["auc"] > 0.9
    # Same seed -> same kernels/biases -> identical scores.
    clf2 = MiniRocketBaseline(n_kernels=168, seed=0, n_jobs=1, chunk_size=1000,
                              alphas=[1.0, 100.0]).fit(X, y, g, f)
    np.testing.assert_allclose(clf2.decision_function(X), s1, rtol=1e-5, atol=1e-6)


def test_minirocket_runner_end_to_end(tmp_path):
    pytest.importorskip("aeon")
    _run_runner(tmp_path, fake=False)


def test_minirocket_runner_end_to_end_fake_transform(tmp_path):
    _run_runner(tmp_path, fake=True)


def _run_runner(tmp_path, fake: bool):
    sys.path.insert(0, str(_ROOT / "scripts"))
    sys.path.insert(0, str(_ROOT))
    import minirocket_phase3 as MR
    from tests._phase3_synth import make_synth_folds, make_synth_npz_tree, make_synth_table

    root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2, 3, 4, 5), n_files=4,
                               length=4096, seed=0)
    table = make_synth_table(root)
    folds = make_synth_folds(table, n_splits=5, seed=0)
    table_csv = tmp_path / "table.csv"
    table.to_csv(table_csv, index=False)
    folds_json = tmp_path / "folds.json"
    folds_json.write_text(json.dumps(folds))
    held = make_synth_npz_tree(tmp_path / "held", clean_ids=(7,), with_mirrors=False,
                               n_files=6, length=4096, seed=9)
    held_csv = tmp_path / "held.csv"
    make_synth_table(held).to_csv(held_csv, index=False)
    out = tmp_path / "out"
    argv = ["--fold", "0", "--table", str(table_csv), "--folds-json", str(folds_json),
            "--npz-root", str(root), "--features-csv", str(held_csv),
            "--heldout-npz-root", str(held),
            "--n-kernels", "168", "--output-dir", str(out), "--tag", "t",
            "--eval-chunk-files", "3", "--alphas", "1,100"]
    factory = _FakeTransform if fake else None
    assert MR.main(argv, transformer_factory=factory) == 0
    man = json.loads((out / "minirocket_t.json").read_text())
    assert man["complete"] is True
    prov = man["provenance"]
    assert prov["fold"] == 0 and prov["n_kernels"] == 168
    assert set(prov["arms"]) == {"A", "B"}
    for arm in ("A", "B"):
        assert prov["arms"][arm]["chosen_alpha"] in (1.0, 100.0)
        assert prov["arms"][arm]["n_train_windows"] > 0
    assert prov["n_val_files"] > 0 and prov["n_heldout_files"] == 6
    if fake:
        assert prov["aeon_version"] is None
    for split in ("val", "heldout"):
        ms = man["metrics"][split]
        assert set(ms) == {"A", "B", "B_zero_delay", "B_tau_x0.5", "B_tau_x2"}
        for v in ms.values():
            assert set(v) >= {"auc", "acc", "n_files", "n_nonfinite_files"}
        dd = man["delay_tests"][split]
        assert set(dd) == {"B_zero_delay", "B_tau_x0.5", "B_tau_x2"}
        assert man["arm_b_minus_a"][split] is None or isinstance(
            man["arm_b_minus_a"][split], float)
    assert man["floor"] == 0.0092
    # Train used only supervised files, at most 8 windows each (7 windows/file here).
    assert prov["arms"]["A"]["n_train_windows"] <= 7 * prov["arms"]["A"]["n_train_files"]
