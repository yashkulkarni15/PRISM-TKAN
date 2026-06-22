"""Tests for post-hoc attribution XAI (Task 4A).

Captum Integrated Gradients over the input window, and a SHAP KernelExplainer
over a small per-file tabular feature frame. The IG test checks the model
attends differently to a stable window (low amplitude, no chatter tone) vs an
unstable window (a strong tone near the natural frequency).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pri2m.models.tkan_model import TKANModel
from pri2m.xai import attribution

_FS = 10_000.0
_TPF = 200.0
_FN = 900.0
_L = 64
_C = 4


def _model(seed: int = 0) -> TKANModel:
    torch.manual_seed(seed)
    return TKANModel(input_size=_C, hidden_size=8)


def _window(*, unstable: bool, seed: int = 1) -> np.ndarray:
    """A tiny [L, C] window: TPF-dominated when stable, +chatter tone when unstable."""
    rng = np.random.default_rng(seed)
    t = np.arange(_L) / _FS
    base = np.sin(2 * np.pi * _TPF * t)
    if unstable:
        base = base + 1.5 * np.sin(2 * np.pi * _FN * t)
    else:
        base = 0.1 * base  # low amplitude, stable
    chans = np.stack([base, np.roll(base, 5), 50.0 * base, 50.0 * np.roll(base, 3)], axis=-1)
    chans = chans + 0.01 * rng.standard_normal(chans.shape)
    return chans.astype(np.float32)


# --------------------------------------------------------------------------- #
# integrated_gradients
# --------------------------------------------------------------------------- #
def test_integrated_gradients_shape_and_finite():
    pytest.importorskip("captum")
    model = _model()
    w = _window(unstable=True)
    attr = attribution.integrated_gradients(model, w, target="b_over_blim", tau=1, steps=16)
    assert isinstance(attr, np.ndarray)
    assert attr.shape == (_L, _C)
    assert np.isfinite(attr).all()


def test_integrated_gradients_accepts_batched_window():
    pytest.importorskip("captum")
    model = _model()
    w = _window(unstable=True)[None]  # [1, L, C]
    attr = attribution.integrated_gradients(model, w, steps=16)
    assert attr.shape == (_L, _C)


def test_integrated_gradients_targets_chatter_logit():
    pytest.importorskip("captum")
    model = _model()
    w = _window(unstable=True)
    attr = attribution.integrated_gradients(model, w, target="chatter_logit", steps=16)
    assert attr.shape == (_L, _C)
    assert np.isfinite(attr).all()


def test_integrated_gradients_differs_stable_vs_unstable():
    pytest.importorskip("captum")
    model = _model()
    stable = attribution.integrated_gradients(model, _window(unstable=False), steps=24)
    unstable = attribution.integrated_gradients(model, _window(unstable=True), steps=24)
    # The model attends differently to the two regimes: total |attribution| differs.
    assert not np.isclose(np.abs(stable).sum(), np.abs(unstable).sum(), rtol=1e-3)


# --------------------------------------------------------------------------- #
# integrated_gradients + extra_scalars (side-input models)
# --------------------------------------------------------------------------- #
def _side_model(seed: int = 0) -> TKANModel:
    torch.manual_seed(seed)
    return TKANModel(input_size=_C, hidden_size=8, side_input=True)


def test_integrated_gradients_extra_scalars_none_matches_default():
    """``extra_scalars=None`` is exactly the old behavior on a plain model."""
    pytest.importorskip("captum")
    model = _model()
    w = _window(unstable=True)
    default = attribution.integrated_gradients(model, w, tau=2, steps=8)
    explicit = attribution.integrated_gradients(model, w, tau=2, steps=8, extra_scalars=None)
    np.testing.assert_allclose(explicit, default)


def test_integrated_gradients_side_model_without_extra_scalars_raises():
    """A side-input model needs its scalar features: the missing-key error propagates."""
    pytest.importorskip("captum")
    model = _side_model()
    w = _window(unstable=True)
    with pytest.raises(ValueError, match="omega"):
        attribution.integrated_gradients(model, w, tau=2, steps=8)


def test_integrated_gradients_side_model_with_extra_scalars():
    pytest.importorskip("captum")
    model = _side_model()
    w = _window(unstable=True)
    attr = attribution.integrated_gradients(
        model, w, tau=2, steps=8, extra_scalars={"omega": 3000.0, "tpf": 200.0}
    )
    assert attr.shape == (_L, _C)
    assert np.isfinite(attr).all()


@pytest.mark.parametrize("key", ["tau", "window"])
def test_integrated_gradients_extra_scalars_rejects_reserved_keys(key):
    """``tau``/``window`` must flow through their dedicated arguments, never extra_scalars."""
    model = _model()
    w = _window(unstable=True)
    with pytest.raises(ValueError, match=key):
        attribution.integrated_gradients(
            model, w, tau=2, steps=8, extra_scalars={key: 1.0, "omega": 3000.0}
        )


# --------------------------------------------------------------------------- #
# shap_feature_summary
# --------------------------------------------------------------------------- #
def test_shap_feature_summary_shape():
    pytest.importorskip("shap")
    import pandas as pd

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "omega": rng.uniform(2000, 4000, size=12),
            "b_mm": rng.uniform(0.5, 2.0, size=12),
            "noise_score": rng.uniform(0.1, 4.0, size=12),
        }
    )

    # A simple deterministic "model": severity rises with b_mm, falls with noise.
    def model_fn(X):
        X = np.asarray(X)
        return X[:, 1] - 0.3 * X[:, 2]

    vals = attribution.shap_feature_summary(model_fn, frame)
    vals = np.asarray(vals)
    # One SHAP value per (sample, feature).
    assert vals.shape[-1] == frame.shape[1]
    assert vals.shape[0] == frame.shape[0]
    assert np.isfinite(vals).all()
