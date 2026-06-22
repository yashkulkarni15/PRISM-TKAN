"""Tests for Phase 3 prediction heads + the model protocol (Task 2B).

Covers ``SeverityHead``, ``TrajectoryHead``, ``PhysicsResidualHead`` and the
``Phase3Model`` runtime-checkable protocol. Tensors are kept small with fixed
seeds.
"""

from __future__ import annotations

import pytest
import torch

from pri2m.models.heads import (
    Phase3Model,
    PhysicsResidualHead,
    SeverityHead,
    TrajectoryHead,
)


# ---------------------------------------------------------------------------
# SeverityHead
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_kan", [True, False])
def test_severity_head_shapes_and_finite(use_kan):
    torch.manual_seed(0)
    B, H = 8, 16
    head = SeverityHead(H, use_kan=use_kan)
    context = torch.randn(B, H)
    out = head(context)

    assert set(out) == {"b_over_blim", "chatter_logit"}
    for key in ("b_over_blim", "chatter_logit"):
        assert out[key].shape == (B,), key
        assert torch.isfinite(out[key]).all(), key


@pytest.mark.parametrize("use_kan", [True, False])
def test_severity_head_backward_finite(use_kan):
    torch.manual_seed(1)
    B, H = 4, 8
    head = SeverityHead(H, use_kan=use_kan)
    context = torch.randn(B, H, requires_grad=True)
    out = head(context)
    loss = out["b_over_blim"].sum() + out["chatter_logit"].sum()
    loss.backward()
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()


def test_severity_head_default_uses_kan():
    # use_kan defaults to True per the contract.
    head = SeverityHead(8)
    out = head(torch.randn(3, 8))
    assert out["b_over_blim"].shape == (3,)


# ---------------------------------------------------------------------------
# TrajectoryHead
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_kan", [True, False])
def test_trajectory_head_rollout_shape(use_kan):
    torch.manual_seed(2)
    B, H, Cout = 6, 16, 4
    n_steps = 5
    head = TrajectoryHead(H, Cout, use_kan=use_kan)
    context = torch.randn(B, H)
    seed_step = torch.randn(B, Cout)
    traj = head(context, seed_step, n_steps=n_steps, tau=4)
    assert traj.shape == (B, n_steps, Cout)
    assert torch.isfinite(traj).all()


def test_trajectory_head_single_step():
    torch.manual_seed(3)
    B, H, Cout = 2, 8, 4
    head = TrajectoryHead(H, Cout)
    traj = head(torch.randn(B, H), torch.randn(B, Cout), n_steps=1, tau=2)
    assert traj.shape == (B, 1, Cout)
    assert torch.isfinite(traj).all()


def test_trajectory_head_backward_finite():
    torch.manual_seed(4)
    B, H, Cout = 3, 8, 2
    head = TrajectoryHead(H, Cout)
    context = torch.randn(B, H, requires_grad=True)
    seed_step = torch.randn(B, Cout)
    traj = head(context, seed_step, n_steps=4, tau=3)
    traj.sum().backward()
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()


def test_trajectory_head_accepts_tensor_tau():
    # tau may be a LongTensor[B] (per-sample) as well as an int.
    torch.manual_seed(5)
    B, H, Cout = 4, 8, 3
    head = TrajectoryHead(H, Cout)
    tau = torch.tensor([1, 2, 3, 4])
    traj = head(torch.randn(B, H), torch.randn(B, Cout), n_steps=3, tau=tau)
    assert traj.shape == (B, 3, Cout)
    assert torch.isfinite(traj).all()


# ---------------------------------------------------------------------------
# PhysicsResidualHead
# ---------------------------------------------------------------------------
def test_physics_residual_head_disabled_by_default():
    head = PhysicsResidualHead()
    assert head.enabled is False


def test_physics_residual_head_raises_altintas():
    head = PhysicsResidualHead()
    with pytest.raises(NotImplementedError) as exc:
        head.forward(torch.randn(2, 4))
    assert "Altintas" in str(exc.value)


def test_physics_residual_head_raises_with_any_args():
    head = PhysicsResidualHead()
    with pytest.raises(NotImplementedError):
        head.forward()
    with pytest.raises(NotImplementedError):
        head.forward(1, 2, foo="bar")


# ---------------------------------------------------------------------------
# Phase3Model protocol
# ---------------------------------------------------------------------------
def test_protocol_is_runtime_checkable():
    # An object with a forward(batch)->dict satisfies the protocol.
    from pri2m.models.baseline import MLPBaseline

    model = MLPBaseline(window_length=32, n_channels=4, hidden_size=8)
    assert isinstance(model, Phase3Model)


def test_protocol_rejects_object_without_forward():
    class NoForward:
        pass

    assert not isinstance(NoForward(), Phase3Model)
