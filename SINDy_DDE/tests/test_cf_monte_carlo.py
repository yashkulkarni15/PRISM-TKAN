"""Monte Carlo survival rates for physics-constrained counterfactuals (U13-A).

:func:`pri2m.xai.counterfactuals.validate_against_blim` scores a counterfactual
at its *nominal* ``(omega_cf, b_cf)`` only. A machinist cannot use a
recommendation that holds only at the nominal point: real spindles jitter and
real depths of cut carry a tolerance. :func:`monte_carlo_survival` re-asks the
same physics question under those tolerances::

    survival = mean over draws of [ b_draw < blim(omega_draw) ]

The tests below use a *synthetic* ``blim_fn`` (never the real Altintas-Budak
physics) so the expected survival of every planted case is known analytically.
The correctness anchor is ``n_draws=0``: with no draws the estimator must
collapse onto the deterministic ``validate_against_blim(..., margin=0.0)``
verdict, case for case.
"""

from __future__ import annotations

import numpy as np
import pytest

from pri2m.xai import counterfactuals as cf


# --------------------------------------------------------------------------- #
# Synthetic physics: a flat-ish analytic boundary + a toy model whose decision
# boundary is a shifted copy of it (same construction as the Task 4B tests, but
# self-contained so this module owns its fixtures).
# --------------------------------------------------------------------------- #
def _flat_blim(omega):
    """b_lim(omega) = 2 mm, gently sloped so omega jitter actually matters."""
    omega = np.asarray(omega, dtype=float)
    return 2.0 + 0.0002 * (omega - 6000.0)


def _scalar_only_blim(omega):
    """A ``blim_fn`` that refuses arrays -- exercises the scalar fallback path."""
    arr = np.asarray(omega, dtype=float)
    if arr.ndim != 0:
        raise TypeError("this blim_fn only accepts scalars")
    return float(_flat_blim(arr))


def _make_predict(blim_fn, boundary_shift=0.0, k=6.0):
    def predict(pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        omega, b = pts[:, 0], pts[:, 1]
        z = k * (b - (blim_fn(omega) + boundary_shift))
        return 1.0 / (1.0 + np.exp(-z))

    return predict


def _cf_case(omega_cf, b_cf, *, omega_base=None, b_base=None, flips=True):
    """A hand-built counterfactual result dict in the shape the module emits."""
    omega_base = float(omega_cf if omega_base is None else omega_base)
    b_base = float(b_cf + 1.0 if b_base is None else b_base)
    return {
        "omega_base": omega_base,
        "b_base": b_base,
        "base_prob": 0.9,
        "base_chatter": True,
        "found": True,
        "flips": bool(flips),
        "omega_cf": float(omega_cf),
        "b_cf": float(b_cf),
        "delta_omega": float(omega_cf) - omega_base,
        "delta_b": float(b_cf) - b_base,
        "cf_prob": 0.1,
        "cost": 0.2,
    }


def _no_cf_case(omega_base=6000.0, b_base=3.0):
    """A case where the grid search found nothing (all ``*_cf`` fields None)."""
    return {
        "omega_base": float(omega_base),
        "b_base": float(b_base),
        "base_prob": 0.9,
        "base_chatter": True,
        "found": False,
        "flips": False,
        "omega_cf": None,
        "b_cf": None,
        "delta_omega": None,
        "delta_b": None,
        "cf_prob": None,
        "cost": None,
    }


_TOL = {"omega_rel": 0.02, "b_abs": 0.05}


# --------------------------------------------------------------------------- #
# The correctness anchor: n_draws=0 == validate_against_blim(margin=0.0)
# --------------------------------------------------------------------------- #
def test_zero_draws_reduces_to_validate_against_blim():
    predict = _make_predict(_flat_blim)
    cases = [
        {"omega": om, "b_mm": float(_flat_blim(om)) + 0.6}
        for om in (4000.0, 6000.0, 8000.0, 10500.0)
    ]
    results = [
        cf.physics_constrained_counterfactual(
            predict, c, omega_grid=[c["omega"]], b_range=(0.1, c["b_mm"]), n_b=64
        )
        for c in cases
    ]
    results.append(_no_cf_case())

    det = cf.validate_against_blim(results, _flat_blim, margin=0.0)
    mc = cf.monte_carlo_survival(results, _flat_blim, uncertainty=_TOL, n_draws=0, seed=0)

    assert mc["n_cases"] == det["n_cases"]
    assert mc["n_flips"] == det["n_flips"]
    for got, want in zip(mc["per_case"], det["per_case"]):
        if want["omega_cf"] is None:
            assert got["survival"] is None
        else:
            # Deterministic verdict: survival is exactly the indicator that
            # validate_against_blim already reports as physics_stable_cf.
            assert got["survival"] == pytest.approx(float(want["physics_stable_cf"]))
            assert got["physics_stable_cf"] is want["physics_stable_cf"]

    n_stable = sum(1 for pc in det["per_case"] if pc["physics_stable_cf"])
    assert mc["n_above_0_85"] == n_stable


def test_zero_draws_honours_margin_like_validate_against_blim():
    # A CF sitting 0.2 mm below the limit is stable at margin=0 but not at 0.5.
    case = _cf_case(6000.0, float(_flat_blim(6000.0)) - 0.2)
    for margin in (0.0, 0.5):
        det = cf.validate_against_blim(case, _flat_blim, margin=margin)
        mc = cf.monte_carlo_survival(
            case, _flat_blim, uncertainty=_TOL, n_draws=0, seed=0, margin=margin
        )
        assert mc["per_case"][0]["survival"] == pytest.approx(
            float(det["per_case"][0]["physics_stable_cf"])
        )


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_same_survival_and_no_global_rng():
    case = _cf_case(6000.0, float(_flat_blim(6000.0)) - 0.05)  # marginal on purpose
    a = cf.monte_carlo_survival(case, _flat_blim, uncertainty=_TOL, n_draws=400, seed=7)

    # Disturb the *global* numpy RNG between calls: a seeded Generator must not care.
    np.random.seed(12345)
    np.random.standard_normal(1000)

    b = cf.monte_carlo_survival(case, _flat_blim, uncertainty=_TOL, n_draws=400, seed=7)
    assert a["per_case"][0]["survival"] == b["per_case"][0]["survival"]

    c = cf.monte_carlo_survival(case, _flat_blim, uncertainty=_TOL, n_draws=400, seed=8)
    assert c["per_case"][0]["survival"] != a["per_case"][0]["survival"]


# --------------------------------------------------------------------------- #
# The statistics behave like survival probabilities
# --------------------------------------------------------------------------- #
def test_deep_inside_survives_and_on_boundary_is_a_coin_flip():
    lim = float(_flat_blim(6000.0))
    deep = _cf_case(6000.0, lim - 1.0)     # 1 mm below the limit, tolerance is 0.05
    on_edge = _cf_case(6000.0, lim)        # exactly on the limit

    rep = cf.monte_carlo_survival(
        [deep, on_edge], _flat_blim, uncertainty=_TOL, n_draws=4000, seed=0
    )
    assert rep["per_case"][0]["survival"] == pytest.approx(1.0)
    assert rep["per_case"][1]["survival"] == pytest.approx(0.5, abs=0.05)


def test_survival_falls_as_tolerances_widen():
    case = _cf_case(6000.0, float(_flat_blim(6000.0)) - 0.15)
    tight = cf.monte_carlo_survival(
        case, _flat_blim, uncertainty={"omega_rel": 0.0, "b_abs": 0.05}, n_draws=4000, seed=1
    )
    loose = cf.monte_carlo_survival(
        case, _flat_blim, uncertainty={"omega_rel": 0.0, "b_abs": 0.30}, n_draws=4000, seed=1
    )
    assert tight["per_case"][0]["survival"] > loose["per_case"][0]["survival"]
    assert loose["per_case"][0]["survival"] < 1.0


def test_boundary_uncertainty_channel_bites():
    """``blim_rel`` (ktc/fn uncertainty) must move survival on its own."""
    case = _cf_case(6000.0, float(_flat_blim(6000.0)) - 0.2)
    none = cf.monte_carlo_survival(
        case, _flat_blim, uncertainty={"omega_rel": 0.0, "b_abs": 0.0}, n_draws=2000, seed=2
    )
    ktc = cf.monte_carlo_survival(
        case,
        _flat_blim,
        uncertainty={"omega_rel": 0.0, "b_abs": 0.0, "blim_rel": 0.25},
        n_draws=2000,
        seed=2,
    )
    assert none["per_case"][0]["survival"] == pytest.approx(1.0)
    assert ktc["per_case"][0]["survival"] < 1.0


def test_uniform_tolerance_is_supported():
    lim = float(_flat_blim(6000.0))
    case = _cf_case(6000.0, lim - 0.05)  # half a 0.1 mm half-width below the limit
    rep = cf.monte_carlo_survival(
        case,
        _flat_blim,
        uncertainty={"omega_rel": 0.0, "b_abs": 0.1, "dist": "uniform"},
        n_draws=6000,
        seed=3,
    )
    # Uniform(-0.1, +0.1) about a point 0.05 below the limit -> 75% of draws survive.
    assert rep["per_case"][0]["survival"] == pytest.approx(0.75, abs=0.03)


# --------------------------------------------------------------------------- #
# Aggregate contract
# --------------------------------------------------------------------------- #
def test_aggregate_schema_and_n_above_0_85_counts():
    lim = float(_flat_blim(6000.0))
    deep_a = _cf_case(6000.0, lim - 1.0)
    deep_b = _cf_case(6000.0, lim - 0.9)
    marginal = _cf_case(6000.0, lim)          # ~0.5 survival
    doomed = _cf_case(6000.0, lim + 1.0)      # above the limit -> ~0
    rep = cf.monte_carlo_survival(
        [deep_a, deep_b, marginal, doomed, _no_cf_case()],
        _flat_blim,
        uncertainty=_TOL,
        n_draws=2000,
        seed=0,
    )

    expected_keys = {
        "per_case",
        "n_cases",
        "n_flips",
        "n_scored",
        "n_above_0_85",
        "frac_above_0_85",
        "mean_survival",
        "median_survival",
        "min_survival",
        "n_draws",
        "seed",
        "uncertainty",
    }
    assert set(rep) == expected_keys

    assert rep["n_cases"] == 5
    assert rep["n_flips"] == 4          # the no-CF case does not flip
    assert rep["n_scored"] == 4         # ...and is not scored
    assert rep["n_above_0_85"] == 2     # only the two deep-inside cases
    assert rep["frac_above_0_85"] == pytest.approx(0.5)
    assert rep["n_draws"] == 2000
    assert rep["seed"] == 0
    assert rep["min_survival"] == pytest.approx(0.0, abs=0.02)
    assert rep["uncertainty"]["omega_rel"] == pytest.approx(0.02)

    survs = [pc["survival"] for pc in rep["per_case"] if pc["survival"] is not None]
    assert rep["mean_survival"] == pytest.approx(float(np.mean(survs)))
    assert rep["median_survival"] == pytest.approx(float(np.median(survs)))
    assert sum(pc["survival_above_0_85"] for pc in rep["per_case"]) == 2


def test_per_case_record_mirrors_validate_against_blim_fields():
    case = _cf_case(6000.0, float(_flat_blim(6000.0)) - 0.5, omega_base=6000.0, b_base=3.4)
    rep = cf.monte_carlo_survival(case, _flat_blim, uncertainty=_TOL, n_draws=200, seed=0)
    pc = rep["per_case"][0]
    for key in (
        "omega_base",
        "b_base",
        "omega_cf",
        "b_cf",
        "flips",
        "blim_at_cf",
        "distance_to_boundary_cf",
        "physics_stable_cf",
        "survival",
        "survival_above_0_85",
        "n_draws",
    ):
        assert key in pc
    assert pc["blim_at_cf"] == pytest.approx(float(_flat_blim(6000.0)))
    assert pc["distance_to_boundary_cf"] == pytest.approx(0.5)
    assert pc["flips"] is True


def test_single_dict_is_accepted_like_validate_against_blim():
    case = _cf_case(6000.0, float(_flat_blim(6000.0)) - 1.0)
    rep = cf.monte_carlo_survival(case, _flat_blim, uncertainty=_TOL, n_draws=100, seed=0)
    assert rep["n_cases"] == 1
    assert rep["per_case"][0]["survival"] == pytest.approx(1.0)


def test_no_counterfactual_is_unscored_not_zero():
    rep = cf.monte_carlo_survival(
        [_no_cf_case()], _flat_blim, uncertainty=_TOL, n_draws=100, seed=0
    )
    pc = rep["per_case"][0]
    assert pc["survival"] is None
    assert pc["survival_above_0_85"] is False
    assert rep["n_scored"] == 0
    assert rep["n_above_0_85"] == 0
    assert rep["frac_above_0_85"] is None
    assert rep["mean_survival"] is None


def test_empty_input_is_an_empty_report():
    rep = cf.monte_carlo_survival([], _flat_blim, uncertainty=_TOL, n_draws=10, seed=0)
    assert rep["n_cases"] == 0
    assert rep["n_above_0_85"] == 0
    assert rep["mean_survival"] is None


# --------------------------------------------------------------------------- #
# Defensive contract
# --------------------------------------------------------------------------- #
def test_scalar_only_blim_fn_still_works():
    case = _cf_case(6000.0, float(_flat_blim(6000.0)) - 1.0)
    rep = cf.monte_carlo_survival(case, _scalar_only_blim, uncertainty=_TOL, n_draws=50, seed=0)
    assert rep["per_case"][0]["survival"] == pytest.approx(1.0)


def test_default_uncertainty_is_used_when_omitted():
    case = _cf_case(6000.0, float(_flat_blim(6000.0)) - 1.0)
    rep = cf.monte_carlo_survival(case, _flat_blim, n_draws=50, seed=0)
    # The echoed spec is the *resolved* one: every channel present, unspecified
    # channels zeroed, defaults carried through unchanged.
    for key, val in cf.DEFAULT_CF_UNCERTAINTY.items():
        assert rep["uncertainty"][key] == val
    assert rep["uncertainty"]["omega_abs"] == 0.0
    assert rep["per_case"][0]["survival"] == pytest.approx(1.0)


def test_unknown_or_negative_uncertainty_keys_raise():
    case = _cf_case(6000.0, 1.0)
    with pytest.raises(ValueError):
        cf.monte_carlo_survival(case, _flat_blim, uncertainty={"omega_pct": 2.0}, n_draws=10, seed=0)
    with pytest.raises(ValueError):
        cf.monte_carlo_survival(case, _flat_blim, uncertainty={"b_abs": -0.1}, n_draws=10, seed=0)
    with pytest.raises(ValueError):
        cf.monte_carlo_survival(
            case, _flat_blim, uncertainty={"dist": "cauchy"}, n_draws=10, seed=0
        )
    with pytest.raises(ValueError):
        cf.monte_carlo_survival(case, _flat_blim, uncertainty=_TOL, n_draws=-1, seed=0)
