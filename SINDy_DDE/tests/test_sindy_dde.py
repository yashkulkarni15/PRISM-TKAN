"""TDD spec for the hand-rolled delay-embedded sparse regression (SINDy-DDE).

The cross-check recovers the regenerative delay tau AND a delayed-differential
regenerative term directly from a raw signal, with NO neural network and NO
``pysindy`` -- an independent white-box estimate to compare against the
tooth-pass period and the TKAN's delay-ablation result.

We synthesize a DDE with a KNOWN delay (a bounded, broadband, chaotic first-order
regenerative oscillator -- the cubic saturation keeps it bounded and the chaos
makes the delay identifiable; a purely linear/narrow-band signal makes any
delayed column a linear combination of x and x-dot, so the delay is NOT
identifiable). We then assert the tau-scan:

* recovers the true delay within a small tolerance (a fixed +1 sample offset
  comes from the RK4 / central-difference alignment, well inside the tolerance),
* selects the regenerative term ``x(t) - x(t-tau)`` with a nonzero coefficient,
* and that STLSQ zeros a guaranteed-irrelevant (decoy) library column.
"""

from __future__ import annotations

import numpy as np
import pytest

from pri2m.analysis import sindy_dde as sd


# --------------------------------------------------------------------------- #
# Synthetic ground-truth: bounded chaotic regenerative DDE with a KNOWN delay
# --------------------------------------------------------------------------- #
def synth_regenerative_dde(
    fs: float = 500.0,
    dur_s: float = 16.0,
    gamma: float = 4.0,
    k1: float = 40.0,
    k3: float = 300.0,
    tau_s: float = 0.05,
    seed: int = 0,
    x0: float = 0.5,
):
    """Integrate ``xdot = -gamma x - k1 r - k3 r^3`` with ``r = x(t) - x(t-tau)``.

    First-order regenerative delay oscillator. ``k1`` (past the Hopf boundary)
    destabilizes the origin; the cubic regenerative term ``k3 r^3`` saturates the
    growth into a bounded, broadband (chaotic) attractor -- so the delay is
    sharply identifiable. Returns ``(x, D)`` with ``D`` the true delay in samples.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    n = int(round(dur_s * fs))
    D = int(round(tau_s * fs))
    x = np.zeros(n)
    x[: D + 1] = x0 + 1e-2 * rng.standard_normal(D + 1)

    def rhs(ix, ixd):
        r = ix - ixd
        return -gamma * ix - k1 * r - k3 * r**3

    for i in range(D, n - 1):
        xd = x[i - D]
        a = rhs(x[i], xd)
        b = rhs(x[i] + 0.5 * dt * a, xd)
        c = rhs(x[i] + 0.5 * dt * b, xd)
        d = rhs(x[i] + dt * c, xd)
        x[i + 1] = x[i] + dt / 6.0 * (a + 2 * b + 2 * c + d)
    return x, D


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_delay_embed_shifts_and_zero_pads():
    x = np.arange(10.0)
    xd = sd.delay_embed(x, 3)
    assert xd.shape == x.shape
    assert np.all(xd[:3] == 0.0)
    # result[t] == x[t-3] for t >= 3
    assert np.allclose(xd[3:], x[:-3])
    # tau=0 is the identity
    assert np.allclose(sd.delay_embed(x, 0), x)


def test_finite_difference_is_exact_on_polynomials():
    dt = 0.01
    t = np.arange(0, 5, dt)
    # d/dt of a linear ramp is constant; central diff is exact on the interior.
    v = sd.finite_difference(3.0 * t + 1.0, dt, order=1)
    assert np.allclose(v[2:-2], 3.0, atol=1e-6)
    # 2nd derivative of a parabola is constant (=2a); exact on the interior.
    a = sd.finite_difference(2.0 * t**2, dt, order=2)
    assert np.allclose(a[2:-2], 4.0, atol=1e-4)


def test_stlsq_recovers_sparse_coefficients_and_zeros_irrelevant_column():
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    decoy = rng.standard_normal(n)  # irrelevant: zero true coefficient
    Theta = np.column_stack([a, b, decoy])
    y = 3.0 * a - 2.0 * b + 1e-4 * rng.standard_normal(n)
    coef, keep, resid = sd.stlsq(Theta, y, threshold=0.05, alpha=1e-10)
    assert keep[0] and keep[1]
    assert not keep[2]  # STLSQ eliminates the decoy
    assert coef[2] == 0.0
    assert abs(coef[0] - 3.0) < 1e-2
    assert abs(coef[1] + 2.0) < 1e-2
    assert resid < 1e-3


# --------------------------------------------------------------------------- #
# The headline: tau-scan recovers the delay + the regenerative term
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tau_scan_recovers_known_delay_and_regenerative_term(seed):
    fs = 500.0
    x, D = synth_regenerative_dde(seed=seed)
    dt = 1.0 / fs
    grid = np.arange(D - 20, D + 21, 1)
    decoy = np.random.default_rng(777).standard_normal(len(x))

    res = sd.tau_scan(
        x,
        dt,
        grid,
        deriv_order=1,           # first-order system -> velocity target
        threshold=0.05,
        poly_order=3,            # x^2, x^3 (low-order polynomials)
        regen_poly_order=3,      # regen, regen^2, regen^3
        include_delayed=True,    # x(t-tau) as its own candidate
        extra_columns={"decoy": decoy},
    )

    # (1) delay recovered within tolerance (fixed +1 discretization offset).
    assert abs(res["tau_star"] - D) <= 2, (res["tau_star"], D)

    # (2) the residual-vs-tau curve genuinely dips at the recovered delay: the
    #     interior minimum beats both grid extremes (an informative scan, not a
    #     flat/degenerate one).
    residuals = res["residuals"]
    assert residuals[np.argmin(residuals)] < residuals[0]
    assert residuals[np.argmin(residuals)] < residuals[-1]

    # (3) the regenerative term x(t)-x(t-tau) is selected with a nonzero coeff.
    assert res["regen_selected"]
    assert abs(res["regen_coef"]) > 0.0

    # (4) STLSQ zeros the guaranteed-irrelevant decoy column.
    names, keep = res["names"], res["keep"]
    assert not keep[names.index("decoy")]

    # (5) the model is sparse (not every candidate survives).
    assert int(keep.sum()) < len(names)


def test_narrowband_signal_is_reported_as_weakly_identifiable():
    """A pure sinusoid makes x(t-tau) a linear combo of x and x-dot for ANY tau,
    so the scan cannot localize the delay. The scan should still run and return a
    (near-flat) residual curve -- we only check it does not crash and flags a
    shallow dip."""
    fs = 500.0
    dt = 1.0 / fs
    t = np.arange(0, 8, dt)
    x = np.sin(2 * np.pi * 30.0 * t)
    grid = np.arange(10, 61, 1)
    res = sd.tau_scan(x, dt, grid, deriv_order=2, threshold=0.05)
    residuals = np.asarray(res["residuals"])
    # near-degenerate: the curve is essentially flat (tiny relative spread).
    spread = (residuals.max() - residuals.min()) / (residuals.mean() + 1e-12)
    assert spread < 0.5  # weakly identifiable -> flagged by a small spread
