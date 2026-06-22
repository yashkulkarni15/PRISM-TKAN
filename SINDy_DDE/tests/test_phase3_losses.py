"""Tests for Phase 3 losses (Task 1C)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from pri2m.training.losses import (
    LossWeights,
    severity_loss,
    total_loss,
    trajectory_loss,
    weighted_bce_with_logits,
    weighted_mse,
)


# ---------------------------------------------------------------------------
# weighted_mse
# ---------------------------------------------------------------------------
def test_weighted_mse_uniform_equals_plain_mse():
    torch.manual_seed(0)
    pred = torch.randn(8)
    target = torch.randn(8)
    weight = torch.ones(8)
    got = weighted_mse(pred, target, weight)
    expected = F.mse_loss(pred, target)
    assert torch.allclose(got, expected, atol=1e-6)


def test_weighted_mse_uniform_equals_plain_mse_2d():
    torch.manual_seed(1)
    pred = torch.randn(4, 3)
    target = torch.randn(4, 3)
    weight = torch.ones(4, 3)
    got = weighted_mse(pred, target, weight)
    expected = F.mse_loss(pred, target)
    assert torch.allclose(got, expected, atol=1e-6)


def test_weighted_mse_zero_weight_nan_row_ignored_and_finite():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, float("nan")])
    weight = torch.tensor([1.0, 1.0, 0.0])

    got = weighted_mse(pred, target, weight)

    # Expected = MSE over the first two (equally weighted) elements.
    expected = F.mse_loss(pred[:2], target[:2])
    assert torch.isfinite(got)
    assert torch.allclose(got, expected, atol=1e-6)


def test_weighted_mse_changing_excluded_target_does_not_change_loss():
    pred = torch.tensor([1.0, 2.0, 3.0])
    weight = torch.tensor([1.0, 1.0, 0.0])

    target_a = torch.tensor([1.5, 2.5, float("nan")])
    target_b = torch.tensor([1.5, 2.5, 1e9])
    target_c = torch.tensor([1.5, 2.5, -42.0])

    la = weighted_mse(pred, target_a, weight)
    lb = weighted_mse(pred, target_b, weight)
    lc = weighted_mse(pred, target_c, weight)

    assert torch.allclose(la, lb, atol=1e-6)
    assert torch.allclose(la, lc, atol=1e-6)


def test_weighted_mse_nonuniform_weights_matches_manual():
    pred = torch.tensor([0.0, 1.0, 2.0])
    target = torch.tensor([1.0, 1.0, 0.0])
    weight = torch.tensor([2.0, 1.0, 3.0])
    # sum(w*(p-t)^2)/sum(w) = (2*1 + 1*0 + 3*4) / 6 = 14/6
    got = weighted_mse(pred, target, weight)
    assert torch.allclose(got, torch.tensor(14.0 / 6.0), atol=1e-6)


def test_weighted_mse_all_zero_weight_is_finite():
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([float("nan"), float("nan")])
    weight = torch.tensor([0.0, 0.0])
    got = weighted_mse(pred, target, weight)
    assert torch.isfinite(got)


def test_weighted_mse_zero_weight_nan_row_gradient_is_finite():
    # NaN-safety must hold in the BACKWARD pass too: torch.where alone leaks a
    # nan gradient through the masked branch. Gradients must stay finite.
    pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = torch.tensor([1.5, 2.5, float("nan")])
    weight = torch.tensor([1.0, 1.0, 0.0])
    loss = weighted_mse(pred, target, weight)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    # excluded row contributes zero gradient
    assert torch.allclose(pred.grad[2], torch.tensor(0.0), atol=1e-6)


def test_weighted_bce_zero_weight_nan_row_gradient_is_finite():
    logit = torch.tensor([0.5, -0.5, 2.0], requires_grad=True)
    target = torch.tensor([1.0, 0.0, float("nan")])
    weight = torch.tensor([1.0, 1.0, 0.0])
    loss = weighted_bce_with_logits(logit, target, weight)
    loss.backward()
    assert torch.isfinite(logit.grad).all()
    assert torch.allclose(logit.grad[2], torch.tensor(0.0), atol=1e-6)


def test_trajectory_loss_zero_weight_nan_sample_gradient_is_finite():
    pred = torch.zeros(2, 4, 2, requires_grad=True)
    true = torch.zeros(2, 4, 2)
    true[1] = float("nan")  # NaN trajectory target on the excluded sample
    weight = torch.tensor([1.0, 0.0])
    loss = trajectory_loss(pred, true, weight)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# weighted_bce_with_logits
# ---------------------------------------------------------------------------
def test_weighted_bce_uniform_equals_plain_bce():
    torch.manual_seed(2)
    logit = torch.randn(8)
    target = (torch.rand(8) > 0.5).float()
    weight = torch.ones(8)
    got = weighted_bce_with_logits(logit, target, weight)
    expected = F.binary_cross_entropy_with_logits(logit, target)
    assert torch.allclose(got, expected, atol=1e-6)


def test_weighted_bce_zero_weight_nan_row_ignored_and_finite():
    logit = torch.tensor([0.5, -0.5, 2.0])
    target = torch.tensor([1.0, 0.0, float("nan")])
    weight = torch.tensor([1.0, 1.0, 0.0])

    got = weighted_bce_with_logits(logit, target, weight)
    expected = F.binary_cross_entropy_with_logits(logit[:2], target[:2])
    assert torch.isfinite(got)
    assert torch.allclose(got, expected, atol=1e-6)


def test_weighted_bce_changing_excluded_target_does_not_change_loss():
    logit = torch.tensor([0.5, -0.5, 2.0])
    weight = torch.tensor([1.0, 1.0, 0.0])
    la = weighted_bce_with_logits(
        logit, torch.tensor([1.0, 0.0, float("nan")]), weight
    )
    lb = weighted_bce_with_logits(logit, torch.tensor([1.0, 0.0, 1.0]), weight)
    assert torch.allclose(la, lb, atol=1e-6)


# ---------------------------------------------------------------------------
# severity_loss
# ---------------------------------------------------------------------------
def _make_sev_batch(seed=3, n=6):
    torch.manual_seed(seed)
    out = {
        "b_over_blim": torch.randn(n),
        "chatter_logit": torch.randn(n),
    }
    batch = {
        "b_over_blim": torch.rand(n),
        "label": (torch.rand(n) > 0.5).float(),
        "sample_weight": torch.ones(n),
    }
    return out, batch


def test_severity_loss_returns_scalar_and_parts():
    out, batch = _make_sev_batch()
    lw = LossWeights()
    loss, parts = severity_loss(out, batch, lw)
    assert torch.is_tensor(loss)
    assert loss.dim() == 0
    assert set(parts) == {"sev_mse", "sev_bce"}
    assert isinstance(parts["sev_mse"], float)
    assert isinstance(parts["sev_bce"], float)


def test_severity_loss_matches_manual_combination():
    out, batch = _make_sev_batch()
    lw = LossWeights(w_sev=2.0, w_reg=3.0, w_cls=0.5)
    loss, parts = severity_loss(out, batch, lw)

    mse = weighted_mse(out["b_over_blim"], batch["b_over_blim"], batch["sample_weight"])
    bce = weighted_bce_with_logits(
        out["chatter_logit"], batch["label"], batch["sample_weight"]
    )
    expected = lw.w_sev * (lw.w_reg * mse + lw.w_cls * bce)
    assert torch.allclose(loss, expected, atol=1e-6)
    assert math.isclose(parts["sev_mse"], float(mse), rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(parts["sev_bce"], float(bce), rel_tol=1e-6, abs_tol=1e-6)


def test_severity_loss_requires_grad_propagates():
    out, batch = _make_sev_batch()
    out["b_over_blim"] = out["b_over_blim"].clone().requires_grad_(True)
    out["chatter_logit"] = out["chatter_logit"].clone().requires_grad_(True)
    loss, _ = severity_loss(out, batch, LossWeights())
    loss.backward()
    assert out["b_over_blim"].grad is not None
    assert torch.isfinite(out["b_over_blim"].grad).all()


# ---------------------------------------------------------------------------
# trajectory_loss
# ---------------------------------------------------------------------------
def test_trajectory_loss_uniform_equals_plain_mse():
    torch.manual_seed(4)
    pred = torch.randn(2, 5, 3)
    true = torch.randn(2, 5, 3)
    weight = torch.ones(2)
    got = trajectory_loss(pred, true, weight)
    # broadcast per-sample weights, all ones -> plain mse
    assert torch.allclose(got, F.mse_loss(pred, true), atol=1e-6)


def test_trajectory_loss_zero_weight_sample_ignored():
    pred = torch.zeros(2, 4, 2)
    true = torch.zeros(2, 4, 2)
    true[1] = 5.0  # huge error on the excluded sample
    weight = torch.tensor([1.0, 0.0])
    got = trajectory_loss(pred, true, weight)
    assert torch.isfinite(got)
    assert torch.allclose(got, torch.tensor(0.0), atol=1e-6)


# ---------------------------------------------------------------------------
# total_loss
# ---------------------------------------------------------------------------
def test_total_loss_no_traj_keys_equals_severity():
    out, batch = _make_sev_batch()
    lw = LossWeights(w_traj=0.0)
    tot, tot_parts = total_loss(out, batch, lw)
    sev, sev_parts = severity_loss(out, batch, lw)
    assert torch.allclose(tot, sev, atol=1e-6)
    assert "traj_mse" not in tot_parts
    assert math.isclose(tot_parts["sev_mse"], sev_parts["sev_mse"], abs_tol=1e-6)


def test_total_loss_wtraj_zero_with_traj_keys_equals_severity():
    out, batch = _make_sev_batch()
    n = out["b_over_blim"].shape[0]
    out["trajectory"] = torch.randn(n, 4, 2)
    batch["trajectory_target"] = torch.randn(n, 4, 2)
    lw = LossWeights(w_traj=0.0)
    tot, _ = total_loss(out, batch, lw)
    sev, _ = severity_loss(out, batch, lw)
    assert torch.allclose(tot, sev, atol=1e-6)


def test_total_loss_with_traj_greater_than_severity_when_error_positive():
    out, batch = _make_sev_batch()
    n = out["b_over_blim"].shape[0]
    out["trajectory"] = torch.zeros(n, 4, 2)
    batch["trajectory_target"] = torch.ones(n, 4, 2)  # nonzero traj error
    lw = LossWeights(w_traj=1.0)
    tot, parts = total_loss(out, batch, lw)
    sev, _ = severity_loss(out, batch, lw)
    assert float(tot) > float(sev)
    assert "traj_mse" in parts
    assert parts["traj_mse"] > 0.0


def test_total_loss_includes_severity_parts():
    out, batch = _make_sev_batch()
    n = out["b_over_blim"].shape[0]
    out["trajectory"] = torch.zeros(n, 3, 2)
    batch["trajectory_target"] = torch.ones(n, 3, 2)
    lw = LossWeights(w_traj=0.5)
    _, parts = total_loss(out, batch, lw)
    assert {"sev_mse", "sev_bce", "traj_mse"} <= set(parts)
