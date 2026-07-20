"""Tests for *physics-constrained* counterfactuals (Task 4B).

Where :mod:`test_xai_counterfactuals` checks the DiCE/surrogate machinery, this
module checks the physics-constrained search + its validation against the
analytic stability boundary ``blim(omega)``.

The idea: for a chatter window the only *actionable* knobs are spindle speed
``omega`` and depth of cut ``b``. A physics-constrained counterfactual searches
the ``(omega, b)`` plane (restricted to actionable ranges) for the smallest move
that flips the model to *stable*. We then validate that move against the
analytic boundary: a physically-correct recommendation must land in the stable
region ``b < blim(omega)``.

The toy model below has a chatter probability that is a *known* function of
``(omega, b)`` straddling a planted ``blim(omega)``. When the model's decision
boundary coincides with the planted ``blim`` every flipping move is genuinely
stable, so the agreement rate is 100%; shifting the model's boundary away from
the physics makes some flips land in the (still) unstable region, dropping the
agreement rate below 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from pri2m.xai import counterfactuals as cf


# --------------------------------------------------------------------------- #
# Toy "physics": a planted stability boundary + a model whose chatter prob is a
# sigmoid of the signed distance to a (possibly offset) copy of that boundary.
# --------------------------------------------------------------------------- #
def _planted_blim(omega):
    """A gently sloped analytic stability limit b_lim(omega) (mm)."""
    omega = np.asarray(omega, dtype=float)
    return 1.0 + 0.0005 * (omega - 3000.0)


def _lobed_blim(omega):
    """A single-lobe blim: ~1 mm everywhere with a tall stable pocket at 7500."""
    omega = np.asarray(omega, dtype=float)
    return 1.0 + 4.0 * np.exp(-0.5 * ((omega - 7500.0) / 400.0) ** 2)


def _make_predict(blim_fn, boundary_shift=0.0, k=6.0):
    """Toy model: P(chatter) = sigmoid(k * (b - (blim(omega) + shift))).

    With ``boundary_shift == 0`` the model's 0.5 decision boundary is exactly the
    planted ``blim``; a positive shift lets the model call windows *stable* while
    they are still physically above the analytic limit.
    """

    def predict(pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        omega, b = pts[:, 0], pts[:, 1]
        z = k * (b - (blim_fn(omega) + boundary_shift))
        return 1.0 / (1.0 + np.exp(-z))

    return predict


# --------------------------------------------------------------------------- #
# physics_constrained_counterfactual
# --------------------------------------------------------------------------- #
def test_cf_finds_flip_by_reducing_depth():
    predict = _make_predict(_planted_blim)
    omega0 = 6000.0
    b0 = 3.0  # blim(6000) = 2.5 -> chatter
    res = cf.physics_constrained_counterfactual(
        predict,
        {"omega": omega0, "b_mm": b0},
        omega_grid=[omega0],          # hold spindle speed, vary depth only
        b_range=(0.2, b0),
        n_b=60,
    )
    assert res["found"] is True
    assert res["flips"] is True
    assert res["base_chatter"] is True
    assert res["cf_prob"] < 0.5
    assert res["delta_b"] < 0.0              # depth was reduced
    assert res["b_cf"] < _planted_blim(omega0)  # moved below the analytic limit


def test_cf_finds_flip_by_shifting_speed_into_a_lobe():
    predict = _make_predict(_lobed_blim)
    omega0, b0 = 11000.0, 2.0  # blim~1 here -> chatter; a stable pocket sits at 7500
    res = cf.physics_constrained_counterfactual(
        predict,
        {"omega": omega0, "b_mm": b0},
        omega_range=(5000.0, 12000.0),
        b_grid=[b0],                  # hold depth, vary spindle speed only
        n_omega=141,
    )
    assert res["found"] is True
    assert res["flips"] is True
    assert abs(res["delta_omega"]) > 0.0
    assert abs(res["omega_cf"] - 7500.0) < 800.0   # moved toward the lobe


def test_cf_reports_not_found_when_no_stable_point_in_range():
    predict = _make_predict(_planted_blim)
    omega0 = 6000.0
    b0 = 3.0
    # Restrict the depth grid to values that all stay above blim(6000)=2.5.
    res = cf.physics_constrained_counterfactual(
        predict,
        {"omega": omega0, "b_mm": b0},
        omega_grid=[omega0],
        b_grid=[2.7, 2.8, 2.9, 3.0],
        flip_threshold=0.5,
    )
    assert res["found"] is False
    assert res["flips"] is False
    assert res["omega_cf"] is None and res["b_cf"] is None


# --------------------------------------------------------------------------- #
# validate_against_blim
# --------------------------------------------------------------------------- #
def _chatter_cases(blim_fn, n=12):
    """A batch of chatter windows (b just above the analytic limit)."""
    rng = np.random.default_rng(0)
    omegas = np.linspace(4000.0, 11000.0, n)
    cases = []
    for om in omegas:
        lim = float(blim_fn(om))
        b0 = lim + float(rng.uniform(0.4, 1.2))  # above the limit -> chatter
        cases.append({"omega": float(om), "b_mm": b0})
    return cases


def _run_cfs(predict, cases):
    return [
        cf.physics_constrained_counterfactual(
            predict,
            c,
            omega_grid=[c["omega"]],   # depth-reduction search (deterministic)
            b_range=(0.1, c["b_mm"]),
            n_b=80,
        )
        for c in cases
    ]


def test_agreement_is_100pct_when_model_boundary_matches_blim():
    predict = _make_predict(_planted_blim, boundary_shift=0.0)
    cases = _chatter_cases(_planted_blim)
    results = _run_cfs(predict, cases)

    report = cf.validate_against_blim(results, _planted_blim)
    assert report["n_flips"] == len(cases)          # every chatter case flipped
    assert report["agreement_rate"] == pytest.approx(1.0)
    assert report["moves_toward_stable_rate"] == pytest.approx(1.0)
    # Every flipping CF lands strictly inside the stable region.
    for pc in report["per_case"]:
        if pc["flips"]:
            assert pc["distance_to_boundary_cf"] > 0.0
            assert pc["physics_stable_cf"] is True
            assert pc["hit"] is True


def test_agreement_drops_when_model_boundary_disagrees_with_blim():
    # Model thinks it is stable 0.9 mm *above* the true analytic limit.
    predict = _make_predict(_planted_blim, boundary_shift=0.9)
    cases = _chatter_cases(_planted_blim)
    results = _run_cfs(predict, cases)

    report = cf.validate_against_blim(results, _planted_blim)
    assert report["n_flips"] >= 1
    assert report["agreement_rate"] < 1.0   # some flips land in the unstable region

    # And it is strictly worse than the physics-consistent model.
    aligned = cf.validate_against_blim(
        _run_cfs(_make_predict(_planted_blim, boundary_shift=0.0), cases),
        _planted_blim,
    )
    assert report["agreement_rate"] < aligned["agreement_rate"]


def test_validate_single_dict_and_distance_signs():
    predict = _make_predict(_planted_blim)
    omega0, b0 = 6000.0, 3.0
    res = cf.physics_constrained_counterfactual(
        predict, {"omega": omega0, "b_mm": b0}, omega_grid=[omega0], b_range=(0.2, b0), n_b=60
    )
    report = cf.validate_against_blim(res, _planted_blim)  # single dict accepted
    pc = report["per_case"][0]
    # Base sits above the limit -> negative signed distance; CF below -> positive.
    assert pc["distance_to_boundary_base"] < 0.0
    assert pc["distance_to_boundary_cf"] > 0.0
    assert pc["moves_toward_stable"] is True
    assert report["agreement_rate"] == pytest.approx(1.0)
