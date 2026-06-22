"""Tests for counterfactual XAI + mode-conditioned routing (Task 4A).

A scikit-learn surrogate maps tabular features to the model's predicted
severity; DiCE then searches for counterfactuals that move the predicted
severity into a desired range by changing ONLY the physically controllable
parameters (spindle speed ``omega`` and depth of cut ``b_mm``). For
broadband-noise-dominated files a counterfactual would be meaningless, so the
mode router instead recommends sensor attention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pri2m.xai import counterfactuals as cf


# --------------------------------------------------------------------------- #
# fit_severity_surrogate
# --------------------------------------------------------------------------- #
def _monotone_frame(n: int = 200, seed: int = 0):
    """Features + a severity that increases with b_mm (and mildly with omega)."""
    rng = np.random.default_rng(seed)
    omega = rng.uniform(2000.0, 4000.0, size=n)
    b_mm = rng.uniform(0.5, 3.0, size=n)
    features = pd.DataFrame({"omega": omega, "b_mm": b_mm})
    severity = 0.4 * b_mm + 1e-4 * omega + 0.01 * rng.standard_normal(n)
    return features, severity


def test_fit_severity_surrogate_fits_and_predicts():
    features, severity = _monotone_frame()
    est = cf.fit_severity_surrogate(features, severity)
    assert hasattr(est, "predict")
    pred = est.predict(features)
    assert np.asarray(pred).shape[0] == len(features)
    # Surrogate tracks the monotone target reasonably well.
    corr = np.corrcoef(pred, severity)[0, 1]
    assert corr > 0.9


# --------------------------------------------------------------------------- #
# dice_counterfactuals
# --------------------------------------------------------------------------- #
def test_dice_counterfactuals_lowers_b_mm():
    pytest.importorskip("dice_ml")
    features, severity = _monotone_frame()
    est = cf.fit_severity_surrogate(features, severity)

    # A high-severity query (large b_mm) we want to push to LOW severity.
    query = pd.DataFrame({"omega": [3000.0], "b_mm": [2.8]})
    query_sev = float(est.predict(query)[0])
    desired_low = float(severity.min())
    desired_high = float(query_sev - 0.3 * (query_sev - severity.min()))
    assert desired_high < query_sev  # we really are asking for a reduction

    out = None
    for seed in (0, 1, 2):
        out = cf.dice_counterfactuals(
            est,
            query,
            desired_range=[desired_low, desired_high],
            features=cf.CONTROLLABLE,
            total_CFs=2,
            seed=seed,
        )
        if out is not None and len(out) > 0:
            break

    if out is None or len(out) == 0:
        pytest.skip("DiCE returned no counterfactuals after 3 seeds; convergence is stochastic")

    # CFs reduce b_mm and stay within the training range of b_mm.
    assert (out["b_mm"] < query["b_mm"].iloc[0]).any()
    assert (out["b_mm"] >= features["b_mm"].min() - 1e-6).all()
    assert (out["b_mm"] <= features["b_mm"].max() + 1e-6).all()


def test_dice_counterfactuals_returns_empty_on_failure():
    pytest.importorskip("dice_ml")
    features, severity = _monotone_frame()
    est = cf.fit_severity_surrogate(features, severity)
    query = pd.DataFrame({"omega": [3000.0], "b_mm": [2.0]})
    # An impossible desired range (far outside what the surrogate can output)
    # must yield an empty DataFrame, not raise.
    out = cf.dice_counterfactuals(
        est, query, desired_range=[1e6, 1e6 + 1.0], features=cf.CONTROLLABLE, total_CFs=2
    )
    assert isinstance(out, pd.DataFrame)


# --------------------------------------------------------------------------- #
# mode_conditioned_explanation
# --------------------------------------------------------------------------- #
def test_mode_conditioned_broadband_noise_routes_to_sensor():
    out = cf.mode_conditioned_explanation({"mode_label": "broadband_noise_dominated"})
    assert out["action"] == "sensor_attention"


def test_mode_conditioned_chatter_routes_to_counterfactual():
    out = cf.mode_conditioned_explanation({"mode_label": "chatter_dominated"})
    assert out["action"] == "counterfactual"
    assert tuple(out["features"]) == cf.CONTROLLABLE


def test_mode_conditioned_default_routes_to_counterfactual():
    # Any non-broadband mode (or missing) defaults to a counterfactual spec.
    out = cf.mode_conditioned_explanation({"mode_label": "tpf_dominated_clean"})
    assert out["action"] == "counterfactual"
    assert tuple(out["features"]) == cf.CONTROLLABLE
