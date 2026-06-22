"""Tests for Phase 3 metrics (Task 1C)."""

from __future__ import annotations

import math

import numpy as np

from sklearn.metrics import roc_auc_score

from pri2m.models.stage_gates import evaluate_trajectory_gate
from pri2m.training.metrics import (
    per_file_severity_metrics,
    trajectory_gate_report,
)


# ---------------------------------------------------------------------------
# trajectory_gate_report (must delegate to stage_gates)
# ---------------------------------------------------------------------------
def _gate_arrays(seed=0):
    rng = np.random.default_rng(seed)
    stable_true = rng.normal(size=32)
    stable_pred = stable_true + rng.normal(scale=0.05, size=32)
    unstable_true = rng.normal(size=32)
    unstable_pred = unstable_true + rng.normal(scale=0.05, size=32)
    return stable_true, stable_pred, unstable_true, unstable_pred


def test_trajectory_gate_report_equals_stage_gate_direct():
    st, sp, ut, up = _gate_arrays()
    true_onset, pred_onset = 100.0, 102.0

    got = trajectory_gate_report(st, sp, ut, up, true_onset, pred_onset)
    expected = evaluate_trajectory_gate(st, sp, ut, up, true_onset, pred_onset)

    assert set(got) == set(expected)
    for key in expected:
        if key == "checks":
            assert got[key] == expected[key]
        else:
            assert got[key] == expected[key]


def test_trajectory_gate_report_passes_previous_metrics():
    st, sp, ut, up = _gate_arrays(seed=1)
    true_onset, pred_onset = 50.0, 51.0
    previous = {"min_r2": 0.99}

    got = trajectory_gate_report(
        st, sp, ut, up, true_onset, pred_onset, previous=previous
    )
    expected = evaluate_trajectory_gate(
        st, sp, ut, up, true_onset, pred_onset, previous_metrics=previous
    )
    assert got["checks"] == expected["checks"]
    assert got["passed"] == expected["passed"]


def test_trajectory_gate_report_high_quality_passes():
    # near-perfect reconstruction + small onset error -> should pass
    rng = np.random.default_rng(7)
    st = rng.normal(size=64)
    ut = rng.normal(size=64)
    got = trajectory_gate_report(st, st.copy(), ut, ut.copy(), 100.0, 101.0)
    assert got["passed"] is True


# ---------------------------------------------------------------------------
# per_file_severity_metrics
# ---------------------------------------------------------------------------
def test_per_file_severity_metrics_perfect_separation_auc_one():
    # Two files, perfectly separable logits and labels.
    file_ids = ["f0", "f0", "f1", "f1"]
    label = np.array([0.0, 0.0, 1.0, 1.0])
    logit = np.array([-5.0, -4.0, 4.0, 5.0])
    b_true = np.array([0.2, 0.2, 1.5, 1.5])
    b_pred = np.array([0.25, 0.15, 1.4, 1.6])

    out = per_file_severity_metrics(b_pred, b_true, label, logit, file_ids)
    assert math.isclose(out["auc"], 1.0, abs_tol=1e-9)
    # Backward-compat pin: clean inputs report zero dropped files.
    assert out["n_files"] == 2
    assert out["n_nonfinite_files"] == 0


def test_per_file_severity_metrics_perfect_pred_mae_zero():
    file_ids = ["a", "a", "b", "b"]
    label = np.array([0.0, 0.0, 1.0, 1.0])
    logit = np.array([-3.0, -3.0, 3.0, 3.0])
    b_true = np.array([0.3, 0.5, 1.2, 1.4])
    b_pred = b_true.copy()  # identical preds

    out = per_file_severity_metrics(b_pred, b_true, label, logit, file_ids)
    assert math.isclose(out["mae"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["rmse"], 0.0, abs_tol=1e-9)


def test_per_file_severity_metrics_returns_python_floats():
    file_ids = ["a", "a", "b", "b"]
    label = np.array([0.0, 0.0, 1.0, 1.0])
    logit = np.array([-1.0, -1.0, 1.0, 1.0])
    b_true = np.array([0.3, 0.5, 1.2, 1.4])
    b_pred = np.array([0.4, 0.4, 1.0, 1.5])

    out = per_file_severity_metrics(b_pred, b_true, label, logit, file_ids)
    for key in ("mae", "rmse", "auc", "acc"):
        assert key in out
        assert isinstance(out[key], float)
    for key in ("n_files", "n_nonfinite_files"):
        assert key in out
        assert isinstance(out[key], int)


def test_per_file_severity_metrics_aggregates_by_file_mean():
    # one file, windows averaged -> mae computed on the file mean
    file_ids = ["only", "only"]
    label = np.array([1.0, 1.0])
    logit = np.array([2.0, 2.0])
    b_true = np.array([1.0, 1.0])
    b_pred = np.array([0.0, 2.0])  # mean = 1.0 == true mean -> mae 0
    out = per_file_severity_metrics(b_pred, b_true, label, logit, file_ids)
    assert math.isclose(out["mae"], 0.0, abs_tol=1e-9)


def test_per_file_severity_metrics_single_class_auc_nan():
    file_ids = ["a", "a", "b", "b"]
    label = np.array([1.0, 1.0, 1.0, 1.0])  # only one class
    logit = np.array([1.0, 1.0, 2.0, 2.0])
    b_true = np.array([1.0, 1.0, 1.2, 1.2])
    b_pred = np.array([1.0, 1.0, 1.2, 1.2])
    out = per_file_severity_metrics(b_pred, b_true, label, logit, file_ids)
    assert math.isnan(out["auc"])


def test_per_file_severity_metrics_ignores_nonfinite_b_true():
    # File "b" has NaN b_true -> excluded from mae/rmse, file "a" perfect.
    file_ids = ["a", "a", "b", "b"]
    label = np.array([0.0, 0.0, 1.0, 1.0])
    logit = np.array([-2.0, -2.0, 2.0, 2.0])
    b_true = np.array([0.5, 0.5, np.nan, np.nan])
    b_pred = np.array([0.5, 0.5, 9.0, 9.0])
    out = per_file_severity_metrics(b_pred, b_true, label, logit, file_ids)
    assert math.isclose(out["mae"], 0.0, abs_tol=1e-9)
    assert np.isfinite(out["mae"])


def test_per_file_severity_metrics_acc_threshold_half():
    # Classification logits put file a below / file b above the 0.5 prob threshold.
    file_ids = ["a", "a", "b", "b"]
    label = np.array([0.0, 0.0, 1.0, 1.0])
    logit = np.array([-2.0, -2.0, 2.0, 2.0])  # sigmoid -> 0.12 / 0.88 -> 0 / 1
    b_true = np.array([0.2, 0.2, 1.0, 1.0])
    b_pred = np.array([0.1, 0.1, 0.9, 0.9])
    out = per_file_severity_metrics(b_pred, b_true, label, logit, file_ids)
    assert math.isclose(out["acc"], 1.0, abs_tol=1e-9)


def test_per_file_severity_metrics_acc_uses_classification_head_not_severity():
    # The classification logits are perfectly separating (acc must be 1.0), while
    # the severity regression b_pred is uninformative (constant 2.0, i.e. always
    # >= 0.5). If acc still thresholded b_pred it would label both files positive
    # and score 0.5 -- so a result of 1.0 proves acc reads the logit head.
    file_ids = ["a", "a", "b", "b"]
    label = np.array([0.0, 0.0, 1.0, 1.0])
    logit = np.array([-6.0, -6.0, 6.0, 6.0])
    b_true = np.array([0.2, 0.2, 1.0, 1.0])
    b_pred = np.array([2.0, 2.0, 2.0, 2.0])  # severity head says "positive" for both
    out = per_file_severity_metrics(b_pred, b_true, label, logit, file_ids)
    assert math.isclose(out["acc"], 1.0, abs_tol=1e-9)


def test_per_file_severity_metrics_nan_logits_excluded_from_classification():
    # GL job 51641089 regression: NaN logits must not crash roc_auc_score.
    # File "f1" has NaN logits; auc/acc must equal the hand-computed values
    # over the three clean files only.
    file_ids = ["f0", "f0", "f1", "f1", "f2", "f2", "f3", "f3"]
    label = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    logit = np.array([-1.0, -1.0, np.nan, np.nan, 0.5, 0.5, 1.0, 1.0])
    b_true = np.array([0.2, 0.2, 1.0, 1.0, 1.1, 1.1, 0.3, 0.3])
    b_pred = np.array([0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 0.4, 0.4])

    out = per_file_severity_metrics(b_pred, b_true, label, logit, file_ids)

    # Hand-computed over the retained (clean) files f0, f2, f3.
    clean_label = np.array([0, 1, 0])
    clean_prob = 1.0 / (1.0 + np.exp(-np.array([-1.0, 0.5, 1.0])))
    assert math.isclose(out["auc"], roc_auc_score(clean_label, clean_prob))
    expected_acc = np.mean((clean_prob >= 0.5).astype(int) == clean_label)
    assert math.isclose(out["acc"], expected_acc, abs_tol=1e-9)
    assert out["n_files"] == 4
    assert out["n_nonfinite_files"] == 1


def test_per_file_severity_metrics_nan_b_pred_excluded_from_regression():
    # NaN b_pred for file "b" (logits finite) -> mae/rmse over remaining
    # files only; classification side unaffected and counts zero drops.
    file_ids = ["a", "a", "b", "b"]
    label = np.array([0.0, 0.0, 1.0, 1.0])
    logit = np.array([-2.0, -2.0, 2.0, 2.0])
    b_true = np.array([0.5, 0.5, 1.0, 1.0])
    b_pred = np.array([0.7, 0.7, np.nan, np.nan])

    out = per_file_severity_metrics(b_pred, b_true, label, logit, file_ids)

    assert math.isclose(out["mae"], 0.2, abs_tol=1e-9)  # file "a" only
    assert math.isclose(out["rmse"], 0.2, abs_tol=1e-9)
    assert math.isclose(out["auc"], 1.0, abs_tol=1e-9)
    assert math.isclose(out["acc"], 1.0, abs_tol=1e-9)
    assert out["n_files"] == 2
    assert out["n_nonfinite_files"] == 0


def test_per_file_severity_metrics_all_files_nonfinite_returns_nan():
    # Every file non-finite on both sides -> every metric NaN, no exception.
    file_ids = ["a", "a", "b", "b"]
    label = np.array([0.0, 0.0, 1.0, 1.0])
    logit = np.array([np.nan, np.nan, np.nan, np.nan])
    b_true = np.array([0.5, 0.5, 1.0, 1.0])
    b_pred = np.array([np.nan, np.nan, np.nan, np.nan])

    out = per_file_severity_metrics(b_pred, b_true, label, logit, file_ids)

    for key in ("mae", "rmse", "auc", "acc"):
        assert math.isnan(out[key])
    assert out["n_files"] == 2
    assert out["n_nonfinite_files"] == 2


def test_per_file_severity_metrics_inf_logit_retained():
    # sigmoid(+-inf) is a finite 0/1, so an infinite logit-mean is RETAINED
    # (only NaN probabilities are dropped) and auc is computed normally.
    file_ids = ["a", "a", "b", "b"]
    label = np.array([0.0, 0.0, 1.0, 1.0])
    logit = np.array([-np.inf, -np.inf, np.inf, np.inf])
    b_true = np.array([0.2, 0.2, 1.0, 1.0])
    b_pred = np.array([0.2, 0.2, 1.0, 1.0])

    out = per_file_severity_metrics(b_pred, b_true, label, logit, file_ids)

    assert math.isclose(out["auc"], 1.0, abs_tol=1e-9)
    assert math.isclose(out["acc"], 1.0, abs_tol=1e-9)
    assert out["n_files"] == 2
    assert out["n_nonfinite_files"] == 0
