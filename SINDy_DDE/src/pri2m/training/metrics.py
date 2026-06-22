"""Phase 3 evaluation metrics (Task 1C).

Per-file severity metrics aggregate window-level predictions to one row per
file (mean over the file's windows), mirroring the design's "per-file score is
the mean over the file's windows". The trajectory gate is delegated to the
existing :func:`pri2m.models.stage_gates.evaluate_trajectory_gate` rather than
reimplemented.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from pri2m.models.stage_gates import evaluate_trajectory_gate


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _file_means(
    values: np.ndarray, file_ids: Sequence
) -> tuple[list, np.ndarray]:
    """Mean of ``values`` grouped by ``file_ids``, preserving first-seen order."""
    order: list = []
    sums: dict = {}
    counts: dict = {}
    for fid, val in zip(file_ids, values):
        if fid not in sums:
            order.append(fid)
            sums[fid] = 0.0
            counts[fid] = 0
        sums[fid] += float(val)
        counts[fid] += 1
    means = np.array([sums[f] / counts[f] for f in order], dtype=float)
    return order, means


def per_file_severity_metrics(
    b_pred: np.ndarray,
    b_true: np.ndarray,
    label: np.ndarray,
    logit: np.ndarray,
    file_ids: Sequence,
) -> dict:
    """Aggregate window-level severity predictions to per-file metrics.

    Windows are averaged per file (by ``file_ids``). Returns python floats:

    - ``mae``/``rmse``: over files whose mean ``b_true`` AND mean ``b_pred``
      are both finite.
    - ``auc``: ``roc_auc_score(file_label, sigmoid(file_mean_logit))``; NaN if
      only one class is present among the retained files.
    - ``acc``: file-level accuracy of the **classification head**
      (``sigmoid(file_mean_logit) >= 0.5``) vs ``label`` -- the same head that
      ``auc`` scores, so the two stay consistent. (Earlier this thresholded the
      severity regression ``file_mean_b_pred`` at 0.5, which decoupled it from
      ``auc`` and made it swing wildly because ``b_over_blim`` is not a 0..1
      probability.)
    - ``n_files``/``n_nonfinite_files``: total file count, and how many files
      were dropped from ``auc``/``acc`` for a non-finite probability.

    Non-finite tolerance: files with a NaN probability (NaN mean logit) are
    excluded from ``auc``/``acc`` and counted in ``n_nonfinite_files`` instead
    of crashing -- GL job 51641089 died hours in when a mid-epoch NaN blow-up
    reached ``roc_auc_score`` (``ValueError: Input contains NaN``). Note
    ``sigmoid(+-inf)`` is a finite 0/1, so infinite logit-means are retained.
    If no files survive a mask, that side's metrics are NaN -- never an
    exception, mirroring the single-class ``auc = nan`` rule.
    """
    b_pred = np.asarray(b_pred, dtype=float)
    b_true = np.asarray(b_true, dtype=float)
    label = np.asarray(label, dtype=float)
    logit = np.asarray(logit, dtype=float)

    order, mean_pred = _file_means(b_pred, file_ids)
    _, mean_true = _file_means(b_true, file_ids)
    _, mean_logit = _file_means(logit, file_ids)
    # Labels are constant within a file; the mean recovers the file's label.
    _, mean_label = _file_means(label, file_ids)

    # Regression metrics over files where both the ground-truth severity and
    # the predicted severity means are finite.
    finite = np.isfinite(mean_true) & np.isfinite(mean_pred)
    if finite.any():
        diff = mean_pred[finite] - mean_true[finite]
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
    else:
        mae = float("nan")
        rmse = float("nan")

    # File-level AUC from sigmoid of the mean classification logit. Files with
    # a non-finite probability are excluded (and counted) rather than handed to
    # roc_auc_score, which raises on NaN. Judged on ``file_prob`` because
    # sigmoid(+-inf) is a finite 0/1, so infinite logits stay scoreable.
    file_label = (mean_label >= 0.5).astype(int)
    file_prob = _sigmoid(mean_logit)
    cls_finite = np.isfinite(file_prob)
    n_nonfinite_files = int(np.count_nonzero(~cls_finite))
    kept_label = file_label[cls_finite]
    kept_prob = file_prob[cls_finite]
    if kept_label.size == 0 or kept_label.min() == kept_label.max():
        auc = float("nan")
    else:
        auc = float(roc_auc_score(kept_label, kept_prob))

    # File-level accuracy from the classification head at a 0.5 probability
    # threshold -- same head as ``auc`` so the two cannot disagree by
    # construction, evaluated over the same retained files.
    if kept_label.size:
        pred_pos = (kept_prob >= 0.5).astype(int)
        acc = float(np.mean(pred_pos == kept_label))
    else:
        acc = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "auc": auc,
        "acc": acc,
        "n_files": len(order),
        "n_nonfinite_files": n_nonfinite_files,
    }


def trajectory_gate_report(
    stable_true: np.ndarray,
    stable_pred: np.ndarray,
    unstable_true: np.ndarray,
    unstable_pred: np.ndarray,
    true_onset: float,
    predicted_onset: float,
    previous: Mapping[str, float] | None = None,
) -> dict:
    """Delegate to :func:`evaluate_trajectory_gate` (no reimplementation).

    ``previous`` is forwarded as ``previous_metrics``.
    """
    return evaluate_trajectory_gate(
        stable_true,
        stable_pred,
        unstable_true,
        unstable_pred,
        true_onset,
        predicted_onset,
        previous_metrics=previous,
    )
