"""Tests for split-conformal calibration of the chatter head.

The 121-125 held-out eval showed that a raw 0.5 threshold on the chatter
probability produces accuracy that swings run-to-run: the probabilities are
uncalibrated. Split-conformal prediction fixes this with a *distribution-free*
guarantee: given exchangeable calibration and test scores, the conformal
prediction set contains the true label with probability at least ``1 - alpha``.

These tests exercise the guarantee on SYNTHETIC scores (no model, no checkpoint):

* the empirical coverage on a fresh split lands at ``1 - alpha`` (the conformal
  marginal-coverage guarantee), across several ``alpha``;
* the abstention rate (windows whose conformal set is ambiguous -- contains
  BOTH chatter and stable -- so no single class can be committed) rises
  monotonically as ``alpha`` shrinks (a higher coverage demand => larger sets);
* the cross-fold ensemble score is the plain average of the per-checkpoint
  chatter probabilities.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

from pri2m.models.tkan_model import TKANModel
from pri2m.xai.conformal import (
    SplitConformal,
    conformal_quantile,
    ensemble_probs,
    nonconformity,
    prediction_sets,
)
from tests._phase3_synth import make_synth_npz_tree, make_synth_table

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))


def _synthetic_scores(n, seed=0):
    """Exchangeable (prob, label) pairs: ``p ~ U(0.1, 0.9)``, ``y ~ Bernoulli(p)``.

    Drawing the label from the score makes ``p`` a well-calibrated probability
    and, more importantly, makes the calibration and test draws exchangeable --
    the only assumption the conformal coverage guarantee needs.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    return p, y


# --------------------------------------------------------------------------- #
# 1. Nonconformity + quantile primitives
# --------------------------------------------------------------------------- #
def test_nonconformity_is_one_minus_true_class_prob():
    probs = np.array([0.9, 0.2, 0.5])
    labels = np.array([1, 0, 1])
    s = nonconformity(probs, labels)
    # true-class prob is [0.9, 0.8, 0.5] -> score = 1 - that.
    np.testing.assert_allclose(s, [0.1, 0.2, 0.5])


def test_conformal_quantile_is_monotone_in_alpha():
    rng = np.random.default_rng(1)
    scores = rng.uniform(size=500)
    q_tight = conformal_quantile(scores, alpha=0.05)  # 95% coverage
    q_loose = conformal_quantile(scores, alpha=0.30)  # 70% coverage
    # Demanding more coverage (smaller alpha) needs a LARGER threshold.
    assert q_tight >= q_loose


# --------------------------------------------------------------------------- #
# 2. THE conformal guarantee: coverage ~ 1 - alpha on a fresh split
# --------------------------------------------------------------------------- #
def test_coverage_matches_target_on_fresh_split():
    p, y = _synthetic_scores(20_000, seed=7)
    p_cal, y_cal = p[:10_000], y[:10_000]
    p_te, y_te = p[10_000:], y[10_000:]

    for alpha in (0.05, 0.1, 0.2):
        conf = SplitConformal.calibrate(p_cal, y_cal, alpha=alpha)
        res = conf.evaluate(p_te, y_te)
        cov = res["coverage"]
        # Marginal-coverage guarantee: cov >= 1 - alpha (up to O(1/n_cal)); and it
        # concentrates AT 1 - alpha for continuous scores. Both directions checked.
        assert cov >= (1 - alpha) - 0.02
        assert abs(cov - (1 - alpha)) < 0.03


def test_prediction_sets_contain_true_label_at_coverage_rate():
    p, y = _synthetic_scores(8000, seed=3)
    p_cal, y_cal = p[:4000], y[:4000]
    p_te, y_te = p[4000:], y[4000:]
    alpha = 0.1
    q = conformal_quantile(nonconformity(p_cal, y_cal), alpha=alpha)
    sets = prediction_sets(p_te, q)  # [N, 2] bool, columns = [stable(0), chatter(1)]
    assert sets.shape == (len(p_te), 2)
    covered = sets[np.arange(len(y_te)), y_te]
    assert abs(covered.mean() - (1 - alpha)) < 0.03


# --------------------------------------------------------------------------- #
# 3. Abstention rate rises as alpha shrinks
# --------------------------------------------------------------------------- #
def test_abstention_rate_rises_as_alpha_shrinks():
    p, y = _synthetic_scores(20_000, seed=11)
    p_cal, y_cal = p[:10_000], y[:10_000]
    p_te = p[10_000:]

    alphas = [0.30, 0.20, 0.10, 0.05, 0.01]  # shrinking alpha => rising coverage
    rates = []
    for alpha in alphas:
        conf = SplitConformal.calibrate(p_cal, y_cal, alpha=alpha)
        rates.append(conf.evaluate(p_te, y[10_000:])["abstention_rate"])

    # Membership is monotone in the conformal threshold q (a larger q only ADDS
    # labels to a set), so the ambiguous-set abstention rate is non-decreasing as
    # alpha shrinks -- and strictly higher at the tightest alpha.
    for lo, hi in zip(rates, rates[1:]):
        assert hi >= lo - 1e-9
    assert rates[-1] > rates[0]


def test_selective_accuracy_beats_raw_threshold_on_committed_windows():
    # Committing only on singleton conformal sets should be at least as accurate
    # as the raw 0.5-threshold accuracy over ALL windows (it drops the ambiguous
    # ones). This is the "fix the 0.5-threshold swing" claim, in miniature.
    p, y = _synthetic_scores(20_000, seed=5)
    p_cal, y_cal = p[:10_000], y[:10_000]
    p_te, y_te = p[10_000:], y[10_000:]
    conf = SplitConformal.calibrate(p_cal, y_cal, alpha=0.1)
    res = conf.evaluate(p_te, y_te)

    raw_acc = ((p_te >= 0.5).astype(int) == y_te).mean()
    assert 0.0 <= res["commit_rate"] <= 1.0
    if res["commit_rate"] > 0.0:
        assert res["selective_accuracy"] >= raw_acc - 0.02


# --------------------------------------------------------------------------- #
# 4. Per-window calibrated confidence
# --------------------------------------------------------------------------- #
def test_calibrated_confidence_in_unit_range_and_monotone():
    p, y = _synthetic_scores(6000, seed=9)
    p_cal, y_cal = p[:3000], y[:3000]
    conf = SplitConformal.calibrate(p_cal, y_cal, alpha=0.1)

    grid = np.linspace(0.5, 0.99, 25)  # predicted class 1 (p >= 0.5)
    c = conf.calibrated_confidence(grid)
    assert c.shape == grid.shape
    assert np.all(c >= 0.0) and np.all(c <= 1.0)
    # For predicted-chatter windows a higher chatter prob is at least as
    # confident (the conformal p-value is monotone in the model score).
    assert np.all(np.diff(c) >= -1e-9)


# --------------------------------------------------------------------------- #
# 5. Cross-fold ensemble score
# --------------------------------------------------------------------------- #
def test_ensemble_probs_averages_checkpoints():
    a = np.array([0.1, 0.9, 0.5])
    b = np.array([0.3, 0.7, 0.5])
    c = np.array([0.2, 0.8, 0.5])
    out = ensemble_probs([a, b, c])
    np.testing.assert_allclose(out, [0.2, 0.8, 0.5])


def test_ensemble_probs_accepts_2d_stack_and_weights():
    stack = np.array([[0.0, 1.0], [1.0, 0.0]])  # [n_folds=2, n_windows=2]
    out = ensemble_probs(stack, weights=[1.0, 3.0])
    # weighted average down the fold axis: (0*1 + 1*3)/4 = 0.75 ; (1*1+0*3)/4=0.25
    np.testing.assert_allclose(out, [0.75, 0.25])


def test_conformal_cli_records_two_checkpoint_ensemble(tmp_path):
    """The user-facing runner must actually exercise its repeated-checkpoint path."""
    import conformal_phase3 as runner

    npz_root = make_synth_npz_tree(
        tmp_path / "npz", clean_ids=(1, 2), n_files=4, length=256, seed=0
    )
    table = tmp_path / "table.csv"
    make_synth_table(npz_root).to_csv(table, index=False)

    preprocessing = {
        "norm": {"mean": [0.0] * 4, "std": [1.0] * 4},
        "window": {"length": 128, "stride": 64},
        "pool": 1,
    }
    checkpoints = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = TKANModel(input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3)
        run_dir = tmp_path / "runs" / f"run{seed}"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        checkpoint = ckpt_dir / "best_auc.pt"
        torch.save(model.state_dict(), checkpoint)
        (run_dir / "preprocessing.json").write_text(json.dumps(preprocessing))
        checkpoints.append(checkpoint)

    out = tmp_path / "conformal"
    assets = tmp_path / "assets"
    rc = runner.main([
        "--checkpoint", str(checkpoints[0]),
        "--checkpoint", str(checkpoints[1]),
        "--npz-root", str(npz_root),
        "--table", str(table),
        "--output-dir", str(out),
        "--research-assets-dir", str(assets),
        "--n-windows", "8",
        "--window-length", "128",
        "--window-stride", "64",
        "--pool", "1",
        "--alphas", "0.2,0.1",
        "--tag", "synth-ensemble",
    ])
    assert rc == 0
    manifest = json.loads((out / "conformal_manifest.json").read_text())
    assert manifest["provenance"]["ensemble"] is True
    assert manifest["provenance"]["n_folds"] == 2
    assert np.isfinite(manifest["headline"]["coverage"])
