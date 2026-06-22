"""Unit tests for the self-built KANLinear (efficient-kan style) layer.

Task 1B — KAN substrate. Pure-torch B-spline + SiLU-base linear layer with
per-edge interpretable spline curves and an efficient-kan regularization loss.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from pri2m.models.kan import KANLinear


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_construct_parameter_shapes():
    """base_weight is [out,in]; spline_weight is [out,in,grid_size+spline_order]."""
    layer = KANLinear(3, 5, grid_size=5, spline_order=3)
    assert layer.base_weight.shape == (5, 3)
    n_coeff = 5 + 3
    assert layer.spline_weight.shape == (5, 3, n_coeff)
    # parameters are trainable
    assert layer.base_weight.requires_grad
    assert layer.spline_weight.requires_grad


def test_n_coeff_matches_grid_plus_order():
    """Number of B-spline basis coefficients == grid_size + spline_order."""
    layer = KANLinear(2, 2, grid_size=7, spline_order=2)
    assert layer.spline_weight.shape[-1] == 7 + 2


# --------------------------------------------------------------------------- #
# Forward shapes (representative test 1)
# --------------------------------------------------------------------------- #
def test_forward_shape_2d():
    """[B,in] -> [B,out]."""
    torch.manual_seed(0)
    layer = KANLinear(4, 6)
    x = torch.randn(8, 4)
    y = layer(x)
    assert y.shape == (8, 6)
    assert torch.isfinite(y).all()


def test_forward_shape_3d_batched():
    """[B,T,in] -> [B,T,out] (arbitrary leading dims)."""
    torch.manual_seed(0)
    layer = KANLinear(4, 6)
    x = torch.randn(2, 10, 4)
    y = layer(x)
    assert y.shape == (2, 10, 6)
    assert torch.isfinite(y).all()


def test_forward_shape_4d_leading_dims():
    """Arbitrary number of leading dims is preserved."""
    torch.manual_seed(0)
    layer = KANLinear(3, 2)
    x = torch.randn(2, 5, 7, 3)
    y = layer(x)
    assert y.shape == (2, 5, 7, 2)


def test_forward_in_features_mismatch_raises():
    """Last dim must equal in_features."""
    layer = KANLinear(4, 6)
    with pytest.raises((RuntimeError, ValueError)):
        layer(torch.randn(8, 5))


# --------------------------------------------------------------------------- #
# B-spline basis
# --------------------------------------------------------------------------- #
def test_b_splines_shape_and_partition_of_unity():
    """b_splines(x) -> [...,in,n_coeff]; degree-p B-splines sum to 1 inside grid."""
    layer = KANLinear(2, 1, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0))
    x = torch.linspace(-0.9, 0.9, 11).unsqueeze(-1).expand(11, 2)
    bases = layer.b_splines(x)
    assert bases.shape == (11, 2, 5 + 3)
    assert torch.isfinite(bases).all()
    # Partition of unity: B-spline bases sum to 1 within the interior grid span.
    sums = bases.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_b_splines_nonnegative_inside_grid():
    """B-spline basis functions are non-negative on their support."""
    layer = KANLinear(1, 1, grid_size=6, spline_order=3)
    x = torch.linspace(-0.8, 0.8, 25).unsqueeze(-1)
    bases = layer.b_splines(x)
    assert (bases >= -1e-6).all()


# --------------------------------------------------------------------------- #
# Gradients (representative test 4)
# --------------------------------------------------------------------------- #
def test_gradients_finite_and_nonzero():
    """After one backward pass, grads on base & spline weights are finite & not all-zero."""
    torch.manual_seed(0)
    layer = KANLinear(3, 2, grid_size=5, spline_order=3)
    x = torch.randn(16, 3)
    y = layer(x)
    loss = (y**2).mean()
    loss.backward()

    assert layer.base_weight.grad is not None
    assert layer.spline_weight.grad is not None
    assert torch.isfinite(layer.base_weight.grad).all()
    assert torch.isfinite(layer.spline_weight.grad).all()
    assert layer.base_weight.grad.abs().sum() > 0
    assert layer.spline_weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# Function fitting (representative tests 2 & 3)
# --------------------------------------------------------------------------- #
def _train_1d(target_fn, *, grid_size=8, steps=500, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    layer = KANLinear(1, 1, grid_size=grid_size, spline_order=3, grid_range=(-1.0, 1.0))
    x = torch.linspace(-1.0, 1.0, 200).unsqueeze(-1)
    y = target_fn(x)
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = layer(x)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = torch.mean((layer(x) - y) ** 2).item()
    return final


def test_fit_sine():
    """KANLinear(1,1) fits y=sin(pi*x) on [-1,1] to MSE < 1e-2 in ~500 Adam steps."""
    mse = _train_1d(lambda x: torch.sin(np.pi * x), grid_size=8, steps=500)
    assert mse < 1e-2, f"sine fit MSE={mse:.3e} (expected < 1e-2)"


def test_fit_linear():
    """Fits y=2x to MSE < 1e-3."""
    mse = _train_1d(lambda x: 2.0 * x, grid_size=8, steps=500)
    assert mse < 1e-3, f"linear fit MSE={mse:.3e} (expected < 1e-3)"


# --------------------------------------------------------------------------- #
# spline_curves (representative test 5)
# --------------------------------------------------------------------------- #
def test_spline_curves_shape_and_finite():
    """spline_curves(n) -> {'x':[n], 'curves':[out,in,n]}; all finite."""
    layer = KANLinear(3, 4, grid_size=5, spline_order=3)
    out = layer.spline_curves(n=50)
    assert set(out.keys()) >= {"x", "curves"}
    assert isinstance(out["x"], np.ndarray)
    assert isinstance(out["curves"], np.ndarray)
    assert out["x"].shape == (50,)
    assert out["curves"].shape == (4, 3, 50)
    assert np.isfinite(out["x"]).all()
    assert np.isfinite(out["curves"]).all()


def test_spline_curves_x_spans_grid_range():
    """The swept x covers grid_range endpoints."""
    layer = KANLinear(2, 2, grid_range=(-2.0, 3.0))
    out = layer.spline_curves(n=100)
    assert np.isclose(out["x"][0], -2.0)
    assert np.isclose(out["x"][-1], 3.0)


def test_spline_curves_reflect_weights():
    """Zeroing the spline weight of one edge zeros that edge's curve."""
    torch.manual_seed(0)
    layer = KANLinear(2, 2, grid_size=5, spline_order=3, spline_scale=1.0)
    with torch.no_grad():
        # make all spline weights clearly non-zero, then zero a single edge
        layer.spline_weight.add_(1.0)
        layer.spline_weight[1, 0, :] = 0.0
    out = layer.spline_curves(n=30)
    curves = out["curves"]
    assert np.allclose(curves[1, 0, :], 0.0)
    # a non-zeroed edge is not identically zero
    assert not np.allclose(curves[0, 0, :], 0.0)


# --------------------------------------------------------------------------- #
# regularization_loss (representative test 6)
# --------------------------------------------------------------------------- #
def test_regularization_loss_scalar_nonnegative():
    """Returns a non-negative scalar tensor."""
    layer = KANLinear(3, 4)
    reg = layer.regularization_loss()
    assert torch.is_tensor(reg)
    assert reg.dim() == 0
    assert float(reg.detach()) >= 0.0


def test_regularization_loss_increases_with_scaled_weights():
    """Scaling spline weights up increases the L1 regularization term."""
    torch.manual_seed(0)
    layer = KANLinear(3, 4)
    with torch.no_grad():
        layer.spline_weight.add_(0.5)  # ensure non-trivial
    base = layer.regularization_loss(l1=1.0, entropy=0.0)
    with torch.no_grad():
        layer.spline_weight.mul_(10.0)
    scaled = layer.regularization_loss(l1=1.0, entropy=0.0)
    assert float(scaled.detach()) > float(base.detach())


def test_regularization_loss_differentiable():
    """The regularization loss carries gradient to spline weights."""
    torch.manual_seed(0)
    layer = KANLinear(2, 2)
    with torch.no_grad():
        layer.spline_weight.add_(0.3)
    reg = layer.regularization_loss(l1=1.0, entropy=0.1)
    reg.backward()
    assert layer.spline_weight.grad is not None
    assert torch.isfinite(layer.spline_weight.grad).all()


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_determinism_under_seed():
    """Same manual seed -> identical initial parameters and forward output."""
    torch.manual_seed(123)
    a = KANLinear(4, 3, grid_size=6, spline_order=3)
    torch.manual_seed(123)
    b = KANLinear(4, 3, grid_size=6, spline_order=3)
    assert torch.allclose(a.base_weight, b.base_weight)
    assert torch.allclose(a.spline_weight, b.spline_weight)
    x = torch.randn(5, 4)
    assert torch.allclose(a(x), b(x))


# --------------------------------------------------------------------------- #
# Base path contribution
# --------------------------------------------------------------------------- #
def test_base_path_matches_silu_linear_when_spline_zero():
    """With spline weights zeroed, forward == base_activation(x) @ base_weight^T."""
    torch.manual_seed(0)
    layer = KANLinear(3, 2, base_activation=nn.SiLU)
    with torch.no_grad():
        layer.spline_weight.zero_()
    x = torch.randn(7, 3)
    expected = nn.SiLU()(x) @ layer.base_weight.t()
    assert torch.allclose(layer(x), expected, atol=1e-5)
