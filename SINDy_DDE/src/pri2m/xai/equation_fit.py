"""Distil a model's per-window scores into a sparse linear formula (Lasso), then measure
held-out fidelity, selection stability across seeds, and ridge-weight concentration."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-4, 1, 40)
FAITHFUL_R2 = 0.5


def _pipe(alpha):
    return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))


def fit_formula(T, y, groups, term_names, n_splits=5, seed=0):
    """1-SE Lasso on standardised terms, alpha picked by grouped CV; coefs in original units.

    `seed` is unused (GroupKFold is deterministic); kept for the interface."""
    folds = list(GroupKFold(n_splits).split(T, y, groups))
    scores = np.array([[r2_score(y[te], _pipe(a).fit(T[tr], y[tr]).predict(T[te]))
                        for tr, te in folds] for a in ALPHAS])  # (alpha, fold)
    mean, se = scores.mean(1), scores.std(1) / np.sqrt(n_splits)
    best = mean.argmax()
    i = np.flatnonzero(mean >= mean[best] - se[best]).max()  # ALPHAS ascending: last = largest
    alpha = ALPHAS[i]
    model = _pipe(alpha).fit(T, y)
    scaler, lasso = model[0], model[1]
    beta = lasso.coef_ / scaler.scale_
    return {
        "alpha": float(alpha),
        "selected": [n for n, b in zip(term_names, beta) if b != 0],
        "coef": {n: float(b) for n, b in zip(term_names, beta) if b != 0},
        "intercept": float(lasso.intercept_ - beta @ scaler.mean_),
        "cv_r2": float(mean[i]),
        "model": model,
    }


def fidelity(fit, T_ho, y_ho):
    r2 = float(r2_score(y_ho, fit["model"].predict(T_ho)))
    return {"r2": r2, "faithful": bool(r2 >= FAITHFUL_R2)}


def bootstrap_selections(T, y, groups, term_names, n_boot=200, seed=0):
    """Selected-term sets of refits on group-resampled data (whole groups, with replacement)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in ids}
    out = []
    for _ in range(n_boot):
        draw = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([rows[g] for g in draw])
        # keep ORIGINAL ids: copies of one group must stay together in CV (fresh ids leak across folds)
        out.append(set(fit_formula(T[idx], y[idx], groups[idx], term_names)["selected"]))
    return out


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def same_equation(seed_fits, boot_sets):
    """Pre-committed rule: median cross-seed Jaccard >= 5th pct of bootstrap pairwise Jaccard,
    and every term selected by all seeds has one sign across seeds."""
    if len(seed_fits) < 2:
        raise ValueError("same_equation needs at least 2 seed fits to compare")
    sets = [set(f["selected"]) for f in seed_fits]
    med = float(np.median([jaccard(a, b) for a, b in combinations(sets, 2)]))
    p05 = float(np.percentile([jaccard(a, b) for a, b in combinations(boot_sets, 2)], 5))
    shared = set.intersection(*sets)
    signs = all(len({np.sign(f["coef"][t]) for f in seed_fits}) == 1 for t in shared)
    return {"median_cross_jaccard": med, "boot_p05": p05,
            "sign_consistent": bool(signs), "same": bool(med >= p05 and signs)}


def ridge_contrast(weights, k=20):
    """k90: fewest weights holding 90% of total |w| mass; top_idx: the k largest |w|."""
    a = np.abs(weights)
    order = np.argsort(-a)
    k90 = int(np.searchsorted(np.cumsum(a[order]), 0.9 * a.sum()) + 1)
    return {"k90": k90, "top_idx": order[:k]}
