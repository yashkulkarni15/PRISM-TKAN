"""Tests for the flag-gated quantile severity head + pinball loss (Variant 1).

The severity head/loss is plain point regression; held-out MAE is heavy-tailed.
``severity_mode="quantile"`` makes ``SeverityHead`` emit a set of quantiles for
``b_over_blim`` (prediction intervals) and routes the regression term through a
pinball (quantile) loss. ``severity_mode="huber"`` (default) is the current
point-regression behavior and MUST be byte-for-byte unchanged.
"""

from __future__ import annotations

import pytest
import torch

from pri2m.models.heads import SeverityHead
from pri2m.models.tkan_model import TKANModel
from pri2m.training.losses import (
    LossWeights,
    pinball_loss,
    severity_loss,
    weighted_bce_with_logits,
    weighted_mse,
)


def _toy_batch(B=6, L=16, C=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    window = torch.randn(B, L, C, generator=g)
    target = window.abs().mean(dim=(1, 2))
    return {
        "window": window,
        "tau": torch.full((B,), 4, dtype=torch.long),
        "b_over_blim": target,
        "label": (target > target.median()).float(),
        "sample_weight": torch.ones(B),
    }


# --------------------------------------------------------------------------- #
# pinball_loss
# --------------------------------------------------------------------------- #
def test_pinball_loss_zero_when_exact():
    target = torch.tensor([1.0, 2.0, 3.0])
    q = (0.1, 0.5, 0.9)
    pred = target.unsqueeze(-1).repeat(1, len(q))  # exact for every quantile
    weight = torch.ones(3)
    got = pinball_loss(pred, target, q, weight)
    assert torch.allclose(got, torch.tensor(0.0), atol=1e-7)


def test_pinball_loss_asymmetric_penalty():
    # q=0.9: under-prediction (target>pred) weighted 0.9; over-prediction 0.1.
    weight = torch.ones(1)
    under = pinball_loss(torch.tensor([[0.0]]), torch.tensor([1.0]), (0.9,), weight)
    over = pinball_loss(torch.tensor([[0.0]]), torch.tensor([-1.0]), (0.9,), weight)
    assert torch.allclose(under, torch.tensor(0.9), atol=1e-6)
    assert torch.allclose(over, torch.tensor(0.1), atol=1e-6)


def test_pinball_loss_nan_safe_zero_weight_and_grad():
    pred = torch.zeros(3, 2, requires_grad=True)
    target = torch.tensor([1.0, 2.0, float("nan")])
    weight = torch.tensor([1.0, 1.0, 0.0])
    loss = pinball_loss(pred, target, (0.25, 0.75), weight)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    # excluded row contributes zero gradient
    assert torch.allclose(pred.grad[2], torch.zeros(2), atol=1e-7)


# --------------------------------------------------------------------------- #
# SeverityHead: huber (default) unchanged, quantile new path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("use_kan", [True, False])
def test_severity_head_huber_default_shapes(use_kan):
    torch.manual_seed(0)
    head = SeverityHead(16, use_kan=use_kan)  # severity_mode defaults to "huber"
    out = head(torch.randn(8, 16))
    assert set(out) == {"b_over_blim", "chatter_logit"}
    assert out["b_over_blim"].shape == (8,)
    assert out["chatter_logit"].shape == (8,)


def test_severity_head_default_mode_is_huber():
    head = SeverityHead(8)
    assert head.severity_mode == "huber"


@pytest.mark.parametrize("use_kan", [True, False])
def test_severity_head_quantile_outputs(use_kan):
    torch.manual_seed(1)
    q = (0.1, 0.5, 0.9)
    head = SeverityHead(16, use_kan=use_kan, severity_mode="quantile", quantiles=q)
    out = head(torch.randn(5, 16))
    assert "b_over_blim_quantiles" in out
    assert out["b_over_blim_quantiles"].shape == (5, 3)
    assert tuple(out["quantile_levels"]) == q
    assert out["b_over_blim"].shape == (5,)
    assert out["chatter_logit"].shape == (5,)
    assert torch.isfinite(out["b_over_blim_quantiles"]).all()


def test_severity_head_quantile_point_is_median():
    torch.manual_seed(2)
    q = (0.1, 0.5, 0.9)
    head = SeverityHead(12, severity_mode="quantile", quantiles=q)
    out = head(torch.randn(7, 12))
    # point estimate == the 0.5 (median) quantile column (index 1 here).
    assert torch.equal(out["b_over_blim"], out["b_over_blim_quantiles"][:, 1])


def test_severity_head_quantile_backward_finite():
    torch.manual_seed(3)
    head = SeverityHead(8, severity_mode="quantile", quantiles=(0.25, 0.5, 0.75))
    ctx = torch.randn(4, 8, requires_grad=True)
    out = head(ctx)
    out["b_over_blim_quantiles"].sum().backward()
    assert ctx.grad is not None
    assert torch.isfinite(ctx.grad).all()


def test_severity_head_rejects_bad_mode_and_quantiles():
    with pytest.raises(ValueError):
        SeverityHead(8, severity_mode="banana")
    with pytest.raises(ValueError):
        SeverityHead(8, severity_mode="quantile", quantiles=(0.0, 1.5))
    with pytest.raises(ValueError):
        SeverityHead(8, severity_mode="quantile", quantiles=())


# --------------------------------------------------------------------------- #
# severity_loss: quantile path routes through pinball; default path unchanged
# --------------------------------------------------------------------------- #
def test_severity_loss_default_path_uses_mse():
    torch.manual_seed(4)
    out = {"b_over_blim": torch.randn(6), "chatter_logit": torch.randn(6)}
    batch = {
        "b_over_blim": torch.rand(6),
        "label": (torch.rand(6) > 0.5).float(),
        "sample_weight": torch.ones(6),
    }
    loss, parts = severity_loss(out, batch, LossWeights())
    mse = weighted_mse(out["b_over_blim"], batch["b_over_blim"], batch["sample_weight"])
    bce = weighted_bce_with_logits(out["chatter_logit"], batch["label"], batch["sample_weight"])
    assert set(parts) == {"sev_mse", "sev_bce"}
    assert torch.allclose(loss, mse + bce, atol=1e-6)


def test_severity_loss_quantile_path_uses_pinball():
    torch.manual_seed(5)
    q = (0.1, 0.5, 0.9)
    out = {
        "b_over_blim": torch.randn(6),
        "b_over_blim_quantiles": torch.randn(6, 3),
        "quantile_levels": q,
        "chatter_logit": torch.randn(6),
    }
    batch = {
        "b_over_blim": torch.rand(6),
        "label": (torch.rand(6) > 0.5).float(),
        "sample_weight": torch.ones(6),
    }
    loss, parts = severity_loss(out, batch, LossWeights())
    pin = pinball_loss(out["b_over_blim_quantiles"], batch["b_over_blim"], q, batch["sample_weight"])
    bce = weighted_bce_with_logits(out["chatter_logit"], batch["label"], batch["sample_weight"])
    assert set(parts) == {"sev_pinball", "sev_bce"}
    assert torch.allclose(loss, pin + bce, atol=1e-6)


# --------------------------------------------------------------------------- #
# TKANModel wiring
# --------------------------------------------------------------------------- #
def test_tkan_model_default_severity_mode_huber():
    model = TKANModel(input_size=4, hidden_size=8)
    out = model(_toy_batch())
    assert set(out) == {"b_over_blim", "chatter_logit"}


def test_tkan_model_quantile_forward_and_grad():
    model = TKANModel(input_size=4, hidden_size=8, severity_mode="quantile",
                      severity_quantiles=(0.1, 0.5, 0.9))
    batch = _toy_batch()
    out = model(batch)
    assert out["b_over_blim_quantiles"].shape == (6, 3)
    assert out["b_over_blim"].shape == (6,)
    loss, parts = severity_loss(out, batch, LossWeights())
    assert "sev_pinball" in parts
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
