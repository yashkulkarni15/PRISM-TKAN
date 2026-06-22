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
