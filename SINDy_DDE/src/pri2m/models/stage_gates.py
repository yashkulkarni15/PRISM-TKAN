"""Phase 3 stage-gate checks for trajectory reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TrajectoryGateConfig:
    min_r2: float = 0.85
    max_onset_error_pct: float = 5.0
    max_regression_r2_drop: float = 0.03


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R squared without requiring scikit-learn."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} != {y_pred.shape}")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float("-inf")
    return 1.0 - ss_res / ss_tot


def onset_error_pct(true_onset: float, predicted_onset: float) -> float:
    """Percent error for chatter onset time/index."""
    if true_onset == 0:
        return 0.0 if predicted_onset == 0 else float("inf")
    return abs(predicted_onset - true_onset) / abs(true_onset) * 100.0


def evaluate_trajectory_gate(
    stable_true: np.ndarray,
    stable_pred: np.ndarray,
    unstable_true: np.ndarray,
    unstable_pred: np.ndarray,
    true_onset: float,
    predicted_onset: float,
    previous_metrics: Mapping[str, float] | None = None,
    config: TrajectoryGateConfig = TrajectoryGateConfig(),
) -> dict:
    """Evaluate the plan's Phase 3 handoff gate."""
    stable_r2 = r2_score_np(stable_true, stable_pred)
    unstable_r2 = r2_score_np(unstable_true, unstable_pred)
    onset_pct = onset_error_pct(true_onset, predicted_onset)
    min_r2_observed = min(stable_r2, unstable_r2)

    checks = {
        "stable_r2_pass": stable_r2 >= config.min_r2,
        "unstable_r2_pass": unstable_r2 >= config.min_r2,
        "onset_pass": onset_pct <= config.max_onset_error_pct,
        "regression_pass": True,
    }
    if previous_metrics and "min_r2" in previous_metrics:
        checks["regression_pass"] = (
            previous_metrics["min_r2"] - min_r2_observed
        ) <= config.max_regression_r2_drop

    return {
        "stable_r2": stable_r2,
        "unstable_r2": unstable_r2,
        "min_r2": min_r2_observed,
        "onset_error_pct": onset_pct,
        "checks": checks,
        "passed": all(checks.values()),
    }
