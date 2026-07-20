"""Tests for the hidden-state *dynamics* XAI module (recurrent-Jacobian spectra).

Regenerative milling chatter IS a delayed-system (DDE) instability, so the KAN-GRU
encoder that classifies it should carry that instability in its *internal*
dynamics. This module analyses the recurrent Jacobian ``J_t = d h_t / d h_{t-1}``
of the encoder's recurrent cell (Sussillo & Barak style): its leading eigenvalue
magnitude ``|λ|`` is the local contraction/expansion rate of the hidden state, and
the hypothesis is that chatter windows push the model's dynamics toward / past the
``|λ| = 1`` bifurcation line while stable windows stay contractive (``|λ| < 1``).

Ground truth is a **linear recurrent map** ``h_t = A h_{t-1} + B x_t`` whose
recurrent Jacobian is exactly ``A`` and whose spectral radius is ``max|eig(A)|`` —
an analytically known value the Jacobian + eigenvalue machinery must reproduce.
The shape / aggregation contracts are then checked on the real synthetic
``TKANModel``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from pri2m.models.tkan_model import TKANModel  # noqa: E402
from pri2m.xai import dynamics  # noqa: E402

# A 3x3 upper-triangular recurrent matrix: its eigenvalues are its diagonal, so
# the spectral radius is exactly max(|0.4|, |0.9|, |-0.3|) = 0.9.
_A = torch.tensor(
    [[0.4, 0.2, 0.1],
     [0.0, 0.9, -0.3],
     [0.0, 0.0, -0.3]],
    dtype=torch.float32,
)
_ANALYTIC_RADIUS = 0.9
_H_TKAN = 6
_C = 4


class LinearRecur(nn.Module):
    """Ground-truth linear recurrent cell ``h_t = A h_{t-1} + B x_t``.

    The cell signature matches :class:`pri2m.models.tkan.TKANCell`
    (``forward(x_t, h_prev, x_delay=None)``), so the same Jacobian routine drives
    both. Because the map is linear, ``d h_t / d h_{t-1} == A`` everywhere,
    independent of ``x_t`` / ``h_prev`` — that is the analytic anchor.
    """

    def __init__(self, A: torch.Tensor, B: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.hidden_size = int(A.shape[0])
        self.input_size = int(B.shape[1])

    def forward(self, x_t, h_prev, x_delay=None):  # noqa: ARG002 - delay unused
        return h_prev @ self.A.T + x_t @ self.B.T


def _linear_cell(seed: int = 0) -> LinearRecur:
    torch.manual_seed(seed)
    return LinearRecur(_A.clone(), torch.randn(3, _C))


def _tkan(seed: int = 0) -> TKANModel:
    torch.manual_seed(seed)
    return TKANModel(input_size=_C, hidden_size=_H_TKAN, num_layers=1, side_input=False)


def _window_batch(n: int, length: int, seed: int = 1):
    """``(windows[N, L, C], taus[list], labels[N])`` with a balanced label split."""
    rng = np.random.default_rng(seed)
    windows = rng.standard_normal((n, length, _C)).astype(np.float32)
    taus = [6] * n
    labels = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=int)
    return windows, taus, labels


# --------------------------------------------------------------------------- #
# 1. recurrent Jacobian: shape + analytic value on the linear map
# --------------------------------------------------------------------------- #
def test_recurrent_jacobian_of_linear_map_equals_weight_matrix():
    cell = _linear_cell()
    J = dynamics.recurrent_jacobian(cell, torch.randn(_C), torch.randn(3))
    assert tuple(J.shape) == (3, 3)
    # d(A h + B x)/dh == A, exactly (up to autograd float error).
    assert torch.allclose(J.float(), _A, atol=1e-5)


def test_recurrent_jacobian_shape_and_eigcount_on_tkan():
    model = _tkan()
    cell = model.encoder.layers[0]
    J = dynamics.recurrent_jacobian(
        cell, torch.randn(_C), torch.randn(_H_TKAN), torch.randn(_C)
    )
    assert tuple(J.shape) == (_H_TKAN, _H_TKAN)
    eig = dynamics.jacobian_eigenvalues(J)
    assert eig.shape == (_H_TKAN,)
    assert np.isfinite(np.abs(eig)).all()


# --------------------------------------------------------------------------- #
# 2. eigenvalues + spectral radius against the analytic value
# --------------------------------------------------------------------------- #
def test_spectral_radius_of_linear_map_matches_analytic():
    cell = _linear_cell()
    J = dynamics.recurrent_jacobian(cell, torch.zeros(_C), torch.randn(3))
    eig = dynamics.jacobian_eigenvalues(J)
    # Eigenvalues of a triangular matrix are its diagonal magnitudes {0.4,0.9,0.3}.
    mags = np.sort(np.abs(eig))
    assert np.allclose(mags, [0.3, 0.4, 0.9], atol=1e-5)
    # Leading eigenvalue is returned first (sorted by descending magnitude).
    assert np.isclose(np.abs(eig[0]), _ANALYTIC_RADIUS, atol=1e-5)
    assert np.isclose(dynamics.spectral_radius(J), _ANALYTIC_RADIUS, atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. per-window spectrum across timesteps on the TKAN encoder
# --------------------------------------------------------------------------- #
def test_hidden_jacobian_spectrum_shapes_on_tkan():
    model = _tkan()
    length = 32
    window = np.random.default_rng(2).standard_normal((length, _C)).astype(np.float32)
    spec = dynamics.hidden_jacobian_spectrum(model, window, tau=6)
    assert spec["spectral_radius"].shape == (length,)
    assert spec["eigenvalues"].shape == (length, _H_TKAN)
    assert np.all(spec["spectral_radius"] >= 0.0)
    assert np.isfinite(spec["spectral_radius"]).all()


def test_hidden_jacobian_spectrum_respects_timestep_stride():
    model = _tkan()
    length = 32
    stride = 4
    window = np.random.default_rng(3).standard_normal((length, _C)).astype(np.float32)
    spec = dynamics.hidden_jacobian_spectrum(model, window, tau=6, t_stride=stride)
    n_t = len(range(0, length, stride))
    assert spec["spectral_radius"].shape == (n_t,)
    assert spec["eigenvalues"].shape == (n_t, _H_TKAN)


def test_window_spectral_radius_is_a_finite_scalar():
    model = _tkan()
    window = np.random.default_rng(4).standard_normal((24, _C)).astype(np.float32)
    sr = dynamics.window_spectral_radius(model, window, tau=6, reduce="mean")
    assert np.isscalar(sr) or np.ndim(sr) == 0
    assert np.isfinite(sr) and sr >= 0.0


# --------------------------------------------------------------------------- #
# 4. stable-vs-chatter aggregation returns per-class stats
# --------------------------------------------------------------------------- #
def test_stable_vs_chatter_aggregation_per_class_stats():
    model = _tkan()
    windows, taus, labels = _window_batch(n=8, length=24)
    res = dynamics.stable_vs_chatter_spectra(model, windows, taus, labels)

    assert res["radii"].shape == (8,)
    assert np.isfinite(res["radii"]).all()
    for cls in ("stable", "chatter"):
        assert res[cls]["n"] == 4
        for key in ("mean", "std", "median", "frac_above_1"):
            assert np.isfinite(res[cls][key])
    assert "delta_mean" in res and np.isfinite(res["delta_mean"])
    assert isinstance(res["hypothesis_supported"], bool)
    # delta_mean is chatter.mean - stable.mean by construction.
    assert np.isclose(res["delta_mean"], res["chatter"]["mean"] - res["stable"]["mean"])


# --------------------------------------------------------------------------- #
# 5. slow / fixed points of the hidden dynamics (optional Sussillo-Barak probe)
# --------------------------------------------------------------------------- #
def test_find_slow_points_locates_linear_fixed_point():
    # For x=0 the linear map h_t = A h_{t-1} has a unique fixed point at h*=0
    # (1 is not an eigenvalue of A), and the Jacobian there is A -> radius 0.9.
    cell = _linear_cell()
    pts = dynamics.find_slow_points(
        cell, torch.zeros(_C), h_dim=3, n_inits=4, steps=400, lr=0.15, seed=0
    )
    assert len(pts) >= 1
    best = pts[0]
    assert np.isfinite(best["speed"]) and best["speed"] < 1e-2
    assert np.allclose(best["h"], np.zeros(3), atol=1e-1)
    assert np.isclose(best["spectral_radius"], _ANALYTIC_RADIUS, atol=1e-3)


# --------------------------------------------------------------------------- #
# 6. figure
# --------------------------------------------------------------------------- #
def test_plot_spectral_distribution_writes_file(tmp_path):
    model = _tkan()
    windows, taus, labels = _window_batch(n=8, length=24)
    res = dynamics.stable_vs_chatter_spectra(model, windows, taus, labels)
    path = tmp_path / "dist.png"
    out = dynamics.plot_spectral_distribution(res, str(path))
    assert out == str(path)
    assert path.exists() and path.stat().st_size > 0
