"""Counterfactual explanations + mode-conditioned routing.

Operators cannot change a part's natural frequency or the noise floor; they can
change the *spindle speed* (``omega``) and the *depth of cut* (``b_mm``). So a
useful "what should I do to leave chatter?" answer is a counterfactual that only
varies those two controllable parameters.

Pipeline
--------
1. :func:`fit_severity_surrogate` -- fit a scikit-learn regressor mapping
   tabular features -> the model's predicted severity (a fast, differentiable-
   free stand-in DiCE can query cheaply).
2. :func:`dice_counterfactuals` -- DiCE (``method="random"``, regressor backend)
   searches for feature settings landing in a desired severity range, varying
   ONLY :data:`CONTROLLABLE`. Returns an empty frame if none are found.
3. :func:`mode_conditioned_explanation` -- route by signal mode: a
   broadband-noise-dominated file gets a "look at the sensor" recommendation
   (a counterfactual over cutting parameters would be meaningless there); every
   other mode gets a counterfactual spec over the controllable parameters.

``dice_ml`` is imported lazily so importing this module never hard-fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTROLLABLE: tuple[str, ...] = ("omega", "b_mm")

_OUTCOME = "outcome"


def fit_severity_surrogate(features: pd.DataFrame, severity_pred: np.ndarray):
    """Fit a scikit-learn regressor: features -> model-predicted severity.

    Uses a small :class:`~sklearn.ensemble.RandomForestRegressor` -- it captures
    the monotone-but-nonlinear feature/severity relationship without tuning and
    plays nicely with DiCE's ``backend="sklearn"`` regressor path.

    Returns the fitted estimator.
    """
    from sklearn.ensemble import RandomForestRegressor

    X = pd.DataFrame(features).reset_index(drop=True)
    y = np.asarray(severity_pred, dtype=float).reshape(-1)
    est = RandomForestRegressor(n_estimators=200, random_state=0)
    est.fit(X, y)
    return est


def dice_counterfactuals(
    surrogate,
    query_row,
    desired_range,
    features=CONTROLLABLE,
    total_CFs: int = 2,
    seed: int = 0,
):
    """DiCE counterfactuals varying only the controllable parameters.

    Parameters
    ----------
    surrogate:
        A fitted sklearn regressor (from :func:`fit_severity_surrogate`).
    query_row:
        Single-row ``DataFrame`` (or dict) of the file's current features.
    desired_range:
        ``[lo, hi]`` target range for the surrogate's severity output.
    features:
        Which features DiCE may vary (defaults to :data:`CONTROLLABLE`).
    total_CFs:
        How many counterfactuals to request.
    seed:
        Random seed for DiCE's stochastic ``"random"`` search.

    Returns
    -------
    pandas.DataFrame
        The found counterfactuals (DiCE's ``final_cfs_df``), or an EMPTY
        DataFrame if DiCE finds none / errors out (it never raises here).
    """
    import dice_ml

    query_df = pd.DataFrame(query_row).reset_index(drop=True)
    feat_cols = list(query_df.columns)

    # Build a training frame DiCE can sample from: spread each feature across a
    # plausible range around the query, label every row with the surrogate's
    # prediction. This gives DiCE the feature ranges + the regressor target.
    rng = np.random.default_rng(seed)
    n = 400
    cols: dict[str, np.ndarray] = {}
    for c in feat_cols:
        v = float(query_df[c].iloc[0])
        if c in features:
            span = max(abs(v), 1.0)
            cols[c] = rng.uniform(v - span, v + span, size=n)
        else:
            # Non-varied features are held near their query value (small jitter
            # so DiCE sees a continuous column, not a constant).
            cols[c] = np.full(n, v) + 1e-6 * rng.standard_normal(n)
    train = pd.DataFrame(cols)
    train[_OUTCOME] = surrogate.predict(train[feat_cols])

    try:
        data = dice_ml.Data(
            dataframe=train,
            continuous_features=feat_cols,
            outcome_name=_OUTCOME,
        )
        dmodel = dice_ml.Model(model=surrogate, backend="sklearn", model_type="regressor")
        engine = dice_ml.Dice(data, dmodel, method="random")
        result = engine.generate_counterfactuals(
            query_df[feat_cols],
            total_CFs=int(total_CFs),
            desired_range=list(desired_range),
            features_to_vary=list(features),
            random_seed=int(seed),
        )
        cf_df = result.cf_examples_list[0].final_cfs_df
        if cf_df is None:
            return pd.DataFrame(columns=feat_cols)
        return cf_df.reset_index(drop=True)
    except Exception:
        # DiCE's random search is stochastic and can find nothing; degrade to an
        # empty frame rather than propagating its internal errors.
        return pd.DataFrame(columns=feat_cols)


# --------------------------------------------------------------------------- #
# Physics-constrained counterfactuals (Task 4B)
# --------------------------------------------------------------------------- #
# Where :func:`dice_counterfactuals` leans on DiCE's stochastic search over a
# surrogate, the functions below do an exhaustive, deterministic grid search over
# ONLY the two actionable process axes -- spindle speed ``omega`` and depth of
# cut ``b`` -- and then *validate* the recommended move against the analytic
# stability boundary ``blim(omega)``. This lets us ask, and check, the operator's
# real question: "what change in speed/depth flips this window to stable, and
# does that move actually point toward the physically stable region b < blim?"


def _base_omega_b(base) -> tuple[float, float]:
    """Pull ``(omega, b)`` out of a dict / pandas row / 2-tuple."""
    if isinstance(base, dict):
        return float(base["omega"]), float(base["b_mm"])
    # pandas Series (has .get) or mapping-like
    get = getattr(base, "get", None)
    if callable(get) and get("omega", None) is not None:
        return float(base["omega"]), float(base["b_mm"])
    seq = list(base)  # (omega, b) tuple/array fallback
    return float(seq[0]), float(seq[1])


def physics_constrained_counterfactual(
    predict,
    base,
    *,
    omega_range: tuple[float, float] | None = None,
    b_range: tuple[float, float] | None = None,
    n_omega: int = 41,
    n_b: int = 41,
    omega_grid=None,
    b_grid=None,
    flip_threshold: float = 0.5,
):
    """Smallest actionable ``(omega, b)`` move that flips the model to stable.

    Searches a grid over the two controllable process axes and returns the
    minimum-cost grid point the model scores as *stable* (chatter probability
    ``< flip_threshold``). "Cost" is a range-normalised L1 distance in
    ``(omega, b)`` so a fractional change in speed trades off against a
    fractional change in depth; the closest stable pocket therefore wins whether
    it is reached by lowering the depth or by hopping to a stability lobe.

    Parameters
    ----------
    predict:
        Callable mapping an ``(N, 2)`` array of ``[omega, b]`` rows to an
        ``(N,)`` array of chatter probabilities (the audited model, or a
        surrogate standing in for it).
    base:
        Current operating point -- a dict/Series with ``omega`` & ``b_mm`` keys,
        or an ``(omega, b)`` pair.
    omega_range, b_range:
        ``(lo, hi)`` actionable ranges. ``omega_range`` defaults to +/-15% of the
        base speed; ``b_range`` defaults to ``(0, b_base)`` (depth can only be
        reduced). Ignored when the matching ``*_grid`` is given.
    n_omega, n_b:
        Grid resolutions when ranges (not explicit grids) are used.
    omega_grid, b_grid:
        Explicit axis values (e.g. ``[omega_base]`` to hold speed fixed).
    flip_threshold:
        Probability below which the model is considered to predict *stable*.

    Returns
    -------
    dict
        ``omega_base, b_base, base_prob, base_chatter, found, omega_cf, b_cf,
        delta_omega, delta_b, cf_prob, cost, flips``. When no stable grid point
        exists the ``*_cf`` / ``delta_*`` / ``cost`` fields are ``None`` and
        ``found`` / ``flips`` are ``False``.
    """
    omega0, b0 = _base_omega_b(base)

    if omega_grid is not None:
        og = np.asarray(omega_grid, dtype=float).reshape(-1)
    else:
        lo, hi = omega_range if omega_range is not None else (0.85 * omega0, 1.15 * omega0)
        og = np.linspace(float(lo), float(hi), int(n_omega))
    if b_grid is not None:
        bg = np.asarray(b_grid, dtype=float).reshape(-1)
    else:
        lo, hi = b_range if b_range is not None else (0.0, b0)
        bg = np.linspace(float(lo), float(hi), int(n_b))

    OO, BB = np.meshgrid(og, bg, indexing="xy")
    pts = np.column_stack([OO.ravel(), BB.ravel()])
    probs = np.asarray(predict(pts), dtype=float).reshape(-1)

    base_prob = float(np.asarray(predict(np.array([[omega0, b0]])), dtype=float).reshape(-1)[0])
    base_chatter = base_prob >= flip_threshold

    result: dict = {
        "omega_base": omega0,
        "b_base": b0,
        "base_prob": base_prob,
        "base_chatter": bool(base_chatter),
        "found": False,
        "flips": False,
        "omega_cf": None,
        "b_cf": None,
        "delta_omega": None,
        "delta_b": None,
        "cf_prob": None,
        "cost": None,
    }

    stable = probs < flip_threshold
    if not stable.any():
        return result

    # Range-normalised L1 move cost (guard against degenerate single-value axes).
    o_scale = float(og.max() - og.min()) or max(abs(omega0), 1.0)
    b_scale = float(bg.max() - bg.min()) or max(abs(b0), 1.0)
    d_omega = pts[:, 0] - omega0
    d_b = pts[:, 1] - b0
    cost = np.abs(d_omega) / o_scale + np.abs(d_b) / b_scale

    cand = np.where(stable)[0]
    best = cand[int(np.argmin(cost[cand]))]

    result.update(
        found=True,
        flips=bool(base_chatter),
        omega_cf=float(pts[best, 0]),
        b_cf=float(pts[best, 1]),
        delta_omega=float(d_omega[best]),
        delta_b=float(d_b[best]),
        cf_prob=float(probs[best]),
        cost=float(cost[best]),
    )
    return result


def _validate_one(cf_result: dict, blim_fn, margin: float) -> dict:
    """Score one counterfactual against ``blim`` (see :func:`validate_against_blim`)."""
    o_base = float(cf_result["omega_base"])
    b_base = float(cf_result["b_base"])
    blim_base = float(np.asarray(blim_fn(o_base)).reshape(-1)[0])
    # Signed distance-to-boundary: blim - b. Positive => inside the stable region.
    dist_base = blim_base - b_base

    pc: dict = {
        "omega_base": o_base,
        "b_base": b_base,
        "blim_at_base": blim_base,
        "distance_to_boundary_base": dist_base,
        "physics_stable_base": bool(dist_base > margin),
        "flips": bool(cf_result.get("flips", False)),
        "omega_cf": cf_result.get("omega_cf"),
        "b_cf": cf_result.get("b_cf"),
        "blim_at_cf": None,
        "distance_to_boundary_cf": None,
        "physics_stable_cf": False,
        "moves_toward_stable": False,
        "hit": False,
    }

    if cf_result.get("omega_cf") is None:
        return pc

    o_cf = float(cf_result["omega_cf"])
    b_cf = float(cf_result["b_cf"])
    blim_cf = float(np.asarray(blim_fn(o_cf)).reshape(-1)[0])
    dist_cf = blim_cf - b_cf
    physics_stable_cf = dist_cf > margin

    pc.update(
        blim_at_cf=blim_cf,
        distance_to_boundary_cf=dist_cf,
        physics_stable_cf=bool(physics_stable_cf),
        # The recommended move increased the (signed) margin to the boundary.
        moves_toward_stable=bool(dist_cf > dist_base),
        # A "hit": the model flipped the window to stable AND that CF is genuinely
        # in the physically stable region b < blim(omega).
        hit=bool(cf_result.get("flips", False) and physics_stable_cf),
    )
    return pc


def validate_against_blim(cf_results, blim_fn, *, margin: float = 0.0) -> dict:
    """Validate physics-constrained CFs against the analytic boundary ``blim``.

    For each counterfactual we ask whether the recommended move lands in the
    physically stable region ``b < blim(omega)``. A *hit* is a case where the
    model flipped the window to stable AND the suggested ``(omega_cf, b_cf)`` is
    truly below the analytic limit; the headline **agreement rate** is the
    fraction of model-flipping cases that are hits::

        agreement_rate = (# flips that are physically stable) / (# flips)

    Parameters
    ----------
    cf_results:
        A single :func:`physics_constrained_counterfactual` result dict, or a
        list of them.
    blim_fn:
        Callable ``omega -> blim`` (the analytic stability limit, mm). The runner
        builds this by interpolating ``blim`` vs ``omega`` straight from
        ``phase3_dataset.csv``; the tests pass a planted boundary.
    margin:
        A window is counted stable only when ``blim(omega) - b > margin`` (mm).

    Returns
    -------
    dict
        ``per_case`` (list of per-CF records with signed distance-to-boundary,
        ``physics_stable_cf``, ``moves_toward_stable``, ``hit``), plus aggregates
        ``n_cases, n_flips, n_hits, agreement_rate, moves_toward_stable_rate,
        mean_distance_to_boundary_cf``. ``agreement_rate`` is ``None`` when no
        case flipped.
    """
    if isinstance(cf_results, dict):
        cf_results = [cf_results]

    per_case = [_validate_one(r, blim_fn, margin) for r in cf_results]

    flips = [pc for pc in per_case if pc["flips"]]
    n_flips = len(flips)
    n_hits = sum(1 for pc in flips if pc["hit"])
    toward = [pc["moves_toward_stable"] for pc in flips]
    dists = [pc["distance_to_boundary_cf"] for pc in flips if pc["distance_to_boundary_cf"] is not None]

    return {
        "per_case": per_case,
        "n_cases": len(per_case),
        "n_flips": n_flips,
        "n_hits": n_hits,
        "agreement_rate": (n_hits / n_flips) if n_flips else None,
        "moves_toward_stable_rate": (float(np.mean(toward)) if toward else None),
        "mean_distance_to_boundary_cf": (float(np.mean(dists)) if dists else None),
    }


def mode_conditioned_explanation(row: dict) -> dict:
    """Route a per-file explanation by its signal mode.

    * ``mode_label == "broadband_noise_dominated"`` -> recommend sensor
      attention; a cutting-parameter counterfactual is meaningless when the
      signal is dominated by broadband noise.
    * any other mode (or missing) -> a counterfactual spec over the controllable
      parameters.
    """
    mode = (row or {}).get("mode_label")
    if mode == "broadband_noise_dominated":
        return {
            "action": "sensor_attention",
            "mode_label": mode,
            "reason": "signal dominated by broadband noise; inspect sensor/setup "
            "before trusting a cutting-parameter counterfactual",
        }
    return {
        "action": "counterfactual",
        "mode_label": mode,
        "features": list(CONTROLLABLE),
    }
