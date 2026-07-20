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
# temporal_saliency_rescaling (TSR, Ismail et al. 2020; arXiv:2010.13924)
# --------------------------------------------------------------------------- #
import torch.nn as nn  # noqa: E402

_TSR_L = 8
_T_LAG = 1  # an early / lagged timestep (the regenerative source lives here)
_T_NOW = 6  # a late timestep


class _PowerCoupling(nn.Module):
    """A stub with an asymmetric cross-timestep coupling ``y = x[t_lag,0]*x[t_now,1]**2``.

    Straight-path Integrated Gradients splits a monomial's credit by exponent, so
    it hands the lagged linear factor only ``1/3`` of the credit and the late
    squared factor ``2/3`` -- i.e. plain IG *under-credits the lagged timestep*.
    Masking-based time-relevance is symmetric (removing either timestep zeros the
    output equally), so TSR restores the lagged timestep to a ``1/2`` share.
    """

    def forward(self, batch: dict) -> dict:
        w = batch["window"]
        y = w[:, _T_LAG, 0] * (w[:, _T_NOW, 1] ** 2)
        return {"chatter_logit": y, "b_over_blim": y}


def _coupling_window() -> np.ndarray:
    w = np.zeros((_TSR_L, _C), dtype=np.float32)
    w[_T_LAG, 0] = 1.3
    w[_T_NOW, 1] = 1.7
    return w


def test_temporal_saliency_rescaling_shape_keys_finite():
    pytest.importorskip("captum")
    model = _PowerCoupling()
    res = attribution.temporal_saliency_rescaling(
        model, _coupling_window(), target="b_over_blim", tau=1, steps=32
    )
    assert set(res) >= {"tsr", "ig", "time_relevance", "salient", "alpha"}
    assert res["tsr"].shape == (_TSR_L, _C)
    assert res["ig"].shape == (_TSR_L, _C)
    assert res["time_relevance"].shape == (_TSR_L,)
    assert np.isfinite(res["tsr"]).all() and np.isfinite(res["ig"]).all()


def test_temporal_saliency_rescaling_raises_lagged_share_vs_plain_ig():
    """TSR un-vanishes the lagged timestep: its attribution share rises above IG."""
    pytest.importorskip("captum")
    model = _PowerCoupling()
    res = attribution.temporal_saliency_rescaling(
        model, _coupling_window(), target="b_over_blim", tau=1, steps=64, alpha=0.5
    )
    ig, tsr = res["ig"], res["tsr"]

    ig_lag, ig_now = abs(ig[_T_LAG, 0]), abs(ig[_T_NOW, 1])
    ig_share = ig_lag / (ig_lag + ig_now)
    tsr_lag, tsr_now = tsr[_T_LAG, 0], tsr[_T_NOW, 1]
    tsr_share = tsr_lag / (tsr_lag + tsr_now)

    # Plain IG hands the lagged term only ~1/3 (exponent split); TSR restores ~1/2.
    assert ig_share == pytest.approx(1.0 / 3.0, abs=0.02)
    assert tsr_share == pytest.approx(0.5, abs=0.02)
    assert tsr_share > ig_share  # the delayed input's apparent attribution rises


def test_temporal_saliency_rescaling_gates_nonsalient_timesteps():
    """A high alpha keeps only the top time-relevance steps; the rest are zeroed."""
    pytest.importorskip("captum")
    model = _PowerCoupling()
    res = attribution.temporal_saliency_rescaling(
        model, _coupling_window(), target="b_over_blim", tau=1, steps=32, alpha=0.99
    )
    salient = np.where(res["salient"])[0].tolist()
    assert salient == [_T_LAG, _T_NOW]
    # Non-salient timesteps carry exactly zero rescaled importance.
    nonsalient = [t for t in range(_TSR_L) if t not in (_T_LAG, _T_NOW)]
    assert np.all(res["tsr"][nonsalient] == 0.0)


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
