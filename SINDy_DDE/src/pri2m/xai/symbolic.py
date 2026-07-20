"""Symbolic distillation of learned ``KANLinear`` edges into closed forms.

The activation/importance view tells us *which* edges a TKAN model leans on; this
module tells us *what each one computes*. For an important edge ``(o, i)`` we
sample its full learned activation

    ``phi_{o,i}(x) = base_weight[o,i]*base_activation(x)
                      + spline_scale * sum_c basis_c(x) * spline_weight[o,i,c]``

over the range the data actually visits, then fit a small library of human-readable
candidates (const / linear / quadratic / cubic / saturating-tanh / oscillatory-sin /
abs / relu) and keep the one the Bayesian Information Criterion prefers. The result
is a row per edge with a string formula (e.g. ``"1.998*x + 0.501"``), so the
white-box report can read the network as equations rather than weights.

Heavy/optional dependencies (``scipy``, ``pandas``, ``matplotlib``) are imported
lazily inside the functions that need them, so importing this module never hard-fails
in a minimal environment.
"""

from __future__ import annotations

import numpy as np
import torch

from pri2m.xai.semantics import edge_index_map


# ---------------------------------------------------------------------------- #
# Formatting helpers
# ---------------------------------------------------------------------------- #
def _fmt(value: float, sig: int = 4) -> str:
    """Round ``value`` to ~3 significant figures for a compact formula string."""
    return f"{float(value):.{sig}g}"


def _goodness(ys: np.ndarray, pred: np.ndarray, k: int) -> tuple[float, float, float]:
    """Return ``(sse, r2, bic)`` for a fit with ``k`` free parameters.

    ``r2 = 1 - SSE/TSS`` (TSS guarded so a constant target still yields a finite
    score) and ``bic = n*ln(SSE/n + 1e-12) + k*ln(n)``.
    """
    n = len(ys)
    resid = ys - pred
    sse = float(np.sum(resid**2))
    tss = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - sse / tss if tss > 1e-30 else (1.0 if sse < 1e-30 else 0.0)
    bic = n * np.log(sse / n + 1e-12) + k * np.log(n)
    return sse, r2, float(bic)


# ---------------------------------------------------------------------------- #
# Candidate fitters: each returns (params, predict_fn, formula) or None on failure
# ---------------------------------------------------------------------------- #
def _fit_poly(xs: np.ndarray, ys: np.ndarray, degree: int):
    """Least-squares polynomial of ``degree`` via ``np.polyfit``.

    ``params`` are returned in ascending-power order (a0 + a1*x + ...), matching
    the linear assertion ``[slope, intercept] -> [a*x + b]`` only for degree 1, so
    we instead store them low-to-high and build the formula accordingly. For the
    linear case the test expects ``params == [a, b]`` with ``a*x + b``; we honour
    that by returning ``[slope, intercept]`` specifically for degree 1.
    """
    coeffs = np.polyfit(xs, ys, degree)  # highest power first
    pred = np.polyval(coeffs, xs)

    if degree == 1:
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        params = [slope, intercept]
        formula = f"{_fmt(slope)}*x + {_fmt(intercept)}"
    else:
        params = [float(c) for c in coeffs]  # highest power first
        terms = []
        p = degree
        for c in coeffs:
            if p == 0:
                terms.append(f"{_fmt(c)}")
            elif p == 1:
                terms.append(f"{_fmt(c)}*x")
            else:
                terms.append(f"{_fmt(c)}*x^{p}")
            p -= 1
        formula = " + ".join(terms)
    return params, pred, formula


def _fit_const(xs: np.ndarray, ys: np.ndarray):
    a = float(np.mean(ys))
    pred = np.full_like(ys, a)
    return [a], pred, f"{_fmt(a)}"


def _fit_abs(xs: np.ndarray, ys: np.ndarray):
    """``a*|x| + b`` by least squares on the design matrix ``[|x|, 1]``."""
    design = np.column_stack([np.abs(xs), np.ones_like(xs)])
    coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = a * np.abs(xs) + b
    return [a, b], pred, f"{_fmt(a)}*|x| + {_fmt(b)}"


def _fit_relu(xs: np.ndarray, ys: np.ndarray):
    """``a*max(x, 0) + b`` by least squares on ``[relu(x), 1]``."""
    design = np.column_stack([np.maximum(xs, 0.0), np.ones_like(xs)])
    coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = a * np.maximum(xs, 0.0) + b
    return [a, b], pred, f"{_fmt(a)}*max(x,0) + {_fmt(b)}"


def _fit_saturating(xs: np.ndarray, ys: np.ndarray):
    """``a*tanh(b*x + c) + d`` via ``scipy.optimize.curve_fit`` (lazy import)."""
    import warnings

    from scipy.optimize import OptimizeWarning, curve_fit

    def model(x, a, b, c, d):
        return a * np.tanh(b * x + c) + d

    span = float(np.ptp(ys)) or 1.0
    # Sensible init: amplitude ~ half-span, moderate slope, no offset, mean level.
    p0 = [0.5 * span, 1.0, 0.0, float(np.mean(ys))]
    # A near-perfect fit gives a singular covariance (OptimizeWarning); the
    # parameters are still valid, so silence that one warning only.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, _ = curve_fit(model, xs, ys, p0=p0, maxfev=20000)
    a, b, c, d = (float(v) for v in popt)
    pred = model(xs, *popt)
    formula = f"{_fmt(a)}*tanh({_fmt(b)}*x + {_fmt(c)}) + {_fmt(d)}"
    return [a, b, c, d], pred, formula


def _fit_oscillatory(xs: np.ndarray, ys: np.ndarray):
    """``a*sin(w*x + p) + d`` via ``scipy.optimize.curve_fit`` (lazy import)."""
    import warnings

    from scipy.optimize import OptimizeWarning, curve_fit

    def model(x, a, w, p, d):
        return a * np.sin(w * x + p) + d

    span = float(np.ptp(xs)) or 1.0
    amp = 0.5 * (float(np.ptp(ys)) or 1.0)
    # Init frequency at one period across the swept range; refine via least squares.
    p0 = [amp, 2.0 * np.pi / span, 0.0, float(np.mean(ys))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, _ = curve_fit(model, xs, ys, p0=p0, maxfev=20000)
    a, w, p, d = (float(v) for v in popt)
    pred = model(xs, *popt)
    formula = f"{_fmt(a)}*sin({_fmt(w)}*x + {_fmt(p)}) + {_fmt(d)}"
    return [a, w, p, d], pred, formula


# (name, k=#params, fitter) -- poly fitters carry their degree via a closure.
_CANDIDATES: list[tuple[str, int, object]] = [
    ("const", 1, _fit_const),
    ("linear", 2, lambda xs, ys: _fit_poly(xs, ys, 1)),
    ("quadratic", 3, lambda xs, ys: _fit_poly(xs, ys, 2)),
    ("cubic", 4, lambda xs, ys: _fit_poly(xs, ys, 3)),
    ("saturating", 4, _fit_saturating),
    ("oscillatory", 4, _fit_oscillatory),
    ("abs", 2, _fit_abs),
    ("relu", 2, _fit_relu),
]


# ---------------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------------- #
def fit_curve(xs, ys, *, r2_threshold: float | None = None,
              simplicity_tol: float = 1e-3) -> dict:
    """Fit a small library of closed forms to 1-D samples and pick the best by BIC.

    Parameters
    ----------
    xs, ys:
        1-D arrays of equal length ``[n]`` -- the sweep inputs and the sampled
        activation values to explain.
    r2_threshold:
        Optional **faithfulness gate** (pykan's ``r2_threshold``: *"if r2 is below
        this threshold, the edge will not be fixed"*). ``None`` (default) leaves
        behaviour identical to the historic BIC-only selection. When a float is
        given, the edge is accepted only if some candidate reaches this coefficient
        of determination; otherwise no closed form is faithful and the result is
        flagged (``accepted=False``, ``name``/``formula`` ``None``) rather than
        snapped to a bad formula.
    simplicity_tol:
        Only used when ``r2_threshold`` is set. Among the candidates that meet the
        gate, the **simplest family** (fewest parameters) whose ``r2`` is within
        ``simplicity_tol`` of the best available ``r2`` is preferred (ties broken
        by BIC). This realises pykan's simplicity-vs-R^2 trade-off, so a linear
        edge is not reported as a cubic that fits a hair better.

    Candidates (free params ``k``): ``const`` (1), ``linear`` (2), ``quadratic``
    (3), ``cubic`` (4), ``saturating`` ``a*tanh(b*x+c)+d`` (4), ``oscillatory``
    ``a*sin(w*x+p)+d`` (4), ``abs`` ``a*|x|+b`` (2), ``relu`` ``a*max(x,0)+b`` (2).
    Polynomials use ``np.polyfit``; ``saturating``/``oscillatory`` use
    ``scipy.optimize.curve_fit`` wrapped in ``try/except`` so a non-converging
    candidate is silently skipped.

    Returns
    -------
    dict
        Default (``r2_threshold is None``): ``{"name", "formula", "r2", "bic",
        "params"}`` for the candidate with the **minimum** BIC (ties broken toward
        fewer parameters) -- unchanged. With a gate, additionally ``{"accepted":
        bool, "best_family": str}``; a rejected edge has ``name``/``formula`` set to
        ``None`` (its ``r2``/``bic``/``params`` still report the closest family).
        ``formula`` is a human-readable string with params rounded to ~3 significant
        figures (e.g. ``"1.998*x + 0.501"``); ``params`` is the parameter list.
    """
    xs = np.asarray(xs, dtype=float).ravel()
    ys = np.asarray(ys, dtype=float).ravel()

    candidates: list[dict] = []
    best: dict | None = None
    for name, k, fitter in _CANDIDATES:
        try:
            params, pred, formula = fitter(xs, ys)
        except Exception:
            # Non-converging / singular candidate -- skip it.
            continue
        if pred is None or not np.all(np.isfinite(pred)):
            continue
        sse, r2, bic = _goodness(ys, pred, k)
        cand = {
            "name": name,
            "formula": formula,
            "r2": r2,
            "bic": bic,
            "params": params,
            "_k": k,
        }
        candidates.append(cand)
        if best is None:
            best = cand
            continue
        # Minimum BIC; tie-break toward fewer parameters.
        if bic < best["bic"] - 1e-9:
            best = cand
        elif abs(bic - best["bic"]) <= 1e-9 and k < best["_k"]:
            best = cand

    if best is None:  # pragma: no cover - const never fails, so unreachable
        a = float(np.mean(ys))
        fallback = {"name": "const", "formula": _fmt(a), "r2": 0.0, "bic": float("inf"),
                    "params": [a]}
        if r2_threshold is not None:
            fallback.update({"accepted": False, "best_family": "const",
                             "name": None, "formula": None})
        return fallback

    # Historic default path: BIC winner, no gate, no extra keys.
    if r2_threshold is None:
        best.pop("_k", None)
        return best

    # --- Faithfulness + simplicity gate (opt-in) ---------------------------- #
    r2_max = max(c["r2"] for c in candidates)
    if r2_max < float(r2_threshold):
        # No family is faithful -> do not snap to a bad formula.
        top = max(candidates, key=lambda c: c["r2"])
        return {
            "name": None,
            "formula": None,
            "r2": float(top["r2"]),
            "bic": float(top["bic"]),
            "params": top["params"],
            "accepted": False,
            "best_family": top["name"],
        }
    # A faithful form exists: prefer the simplest family within simplicity_tol of
    # the best r2 that still clears the gate.
    floor = max(float(r2_threshold), r2_max - float(simplicity_tol))
    pool = [c for c in candidates if c["r2"] >= floor]
    chosen = min(pool, key=lambda c: (c["_k"], c["bic"]))
    return {
        "name": chosen["name"],
        "formula": chosen["formula"],
        "r2": float(chosen["r2"]),
        "bic": float(chosen["bic"]),
        "params": chosen["params"],
        "accepted": True,
        "best_family": chosen["name"],
    }


def eval_fit(fit: dict, xs: np.ndarray) -> np.ndarray:
    """Evaluate a :func:`fit_curve`/:func:`fit_edge` result on ``xs``.

    Dispatches over the SAME candidate-family formulas :func:`fit_curve` fits
    (``_CANDIDATES``' ``name`` -> closed-form callable), reconstructed from
    ``fit["params"]`` rather than re-fit. This lets callers (e.g. the spline
    atlas) draw the accepted symbolic overlay on an arbitrary sweep without
    duplicating the family definitions.

    Parameters
    ----------
    fit:
        A dict as returned by :func:`fit_curve`/:func:`fit_edge` -- must carry
        ``"name"`` (one of the ``_CANDIDATES`` names) and ``"params"``.
    xs:
        1-D array of x-values to evaluate the closed form at.

    Returns
    -------
    np.ndarray
        ``fit``'s closed form evaluated at ``xs``, same shape as ``xs``.

    Raises
    ------
    ValueError
        If ``fit.get("name")`` is ``None`` or not a recognised family (e.g. a
        rejected fit from a faithfulness-gated :func:`fit_curve` call).
    """
    xs = np.asarray(xs, dtype=float)
    name = fit.get("name")
    if name is None:
        raise ValueError("eval_fit: fit has no accepted family (name is None)")
    family_names = {fam_name for fam_name, _k, _fitter in _CANDIDATES}
    if name not in family_names:
        raise ValueError(f"eval_fit: unknown family {name!r}")

    params = fit["params"]
    if name == "const":
        (a,) = params
        return np.full_like(xs, a)
    if name == "linear":
        a, b = params
        return a * xs + b
    if name == "quadratic":
        a, b, c = params  # highest power first: a*x^2 + b*x + c
        return a * xs**2 + b * xs + c
    if name == "cubic":
        a, b, c, d = params  # highest power first: a*x^3 + b*x^2 + c*x + d
        return a * xs**3 + b * xs**2 + c * xs + d
    if name == "saturating":
        a, b, c, d = params
        return a * np.tanh(b * xs + c) + d
    if name == "oscillatory":
        a, w, p, d = params
        return a * np.sin(w * xs + p) + d
    if name == "abs":
        a, b = params
        return a * np.abs(xs) + b
    if name == "relu":
        a, b = params
        return a * np.maximum(xs, 0.0) + b
    raise ValueError(f"eval_fit: unknown family {name!r}")  # pragma: no cover


def _eval_edge(kan, o: int, i: int, xs: np.ndarray) -> np.ndarray:
    """Evaluate edge ``(o, i)``'s full activation ``phi_{o,i}(x)`` over ``xs[n]``.

    Returns ``np.ndarray[n]`` -- ``base + spline`` exactly as the forward pass
    composes them (recipe from the foundation API).
    """
    xs = np.asarray(xs, dtype=float).ravel()
    n = len(xs)
    X = np.zeros((n, kan.in_features), dtype=np.float32)
    X[:, i] = xs
    with torch.no_grad():
        basis = kan.b_splines(torch.as_tensor(X, dtype=torch.float32))  # [n, in, n_coeff]
        spline = (basis[:, i, :] @ kan.spline_weight[o, i]) * kan.spline_scale  # [n]
        base = kan.base_weight[o, i] * kan.base_activation(
            torch.as_tensor(xs, dtype=torch.float32)
        )  # [n]
        phi = (base + spline).detach().numpy()
    return phi


def fit_edge(kan, o: int, i: int, lo, hi, n: int = 200, *,
             r2_threshold: float | None = None, simplicity_tol: float = 1e-3) -> dict:
    """Symbolic-fit edge ``(o, i)`` of ``kan`` over the range ``[lo, hi]``.

    Evaluates ``phi_{o,i}`` over ``linspace(lo, hi, n)`` (``np.ndarray[n]``), calls
    :func:`fit_curve`, and returns its dict augmented with the edge's coordinates.
    ``r2_threshold``/``simplicity_tol`` are forwarded to :func:`fit_curve` (default
    ``None`` leaves the historic BIC-only behaviour).

    Returns
    -------
    dict
        :func:`fit_curve`'s output plus ``{"out": o, "in": i, "lo": lo, "hi": hi}``.
    """
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Degenerate visited range -> tiny symmetric window so the fit is defined.
        center = lo if np.isfinite(lo) else 0.0
        lo, hi = center - 1e-3, center + 1e-3
    xs = np.linspace(lo, hi, n)
    ys = _eval_edge(kan, o, i, xs)
    res = fit_curve(xs, ys, r2_threshold=r2_threshold, simplicity_tol=simplicity_tol)
    res.update({"out": int(o), "in": int(i), "lo": lo, "hi": hi})
    return res


def symbolic_table(model, activations: dict, top_k: int = 20, *,
                   r2_threshold: float | None = 0.9,
                   simplicity_tol: float = 1e-3) -> list[dict]:
    """Rank every edge by data-aware importance and symbolic-fit the top ``top_k``.

    Parameters
    ----------
    model:
        The TKAN model the activations were collected from (used for semantics).
    activations:
        Output of :func:`pri2m.xai.activations.collect_activations` --
        ``{kan_name: {"importance": [out, in], "visited_lo": [in],
        "visited_hi": [in], "module": KANLinear, ...}}``.
    top_k:
        Number of top edges (by importance, descending) to fit.
    r2_threshold:
        Faithfulness gate forwarded to :func:`fit_curve` (default ``0.9``). An edge
        whose best closed form falls below this R^2 is marked ``accepted=False``
        with ``formula="no faithful closed form"`` instead of being snapped to a
        misleading equation; edges that *do* fit well keep their formula unchanged.
        Pass ``None`` to disable the gate (historic behaviour: every edge gets a
        formula).
    simplicity_tol:
        Forwarded to :func:`fit_curve` -- prefer the simpler family when R^2 is
        within this tolerance of the best.

    Returns
    -------
    list[dict]
        One row per fitted edge, sorted by importance descending, each:
        ``{"kan", "out", "in", "block", "name", "importance", "formula",
        "r2", "bic", "accepted"}``. Each edge is fit over the range its input
        actually visits (``visited_lo[i]`` .. ``visited_hi[i]``) and tagged with the
        semantic block/name from :func:`pri2m.xai.semantics.edge_index_map`. A
        rejected edge (``accepted=False``) carries ``formula="no faithful closed
        form"``.
    """
    semantics = edge_index_map(model)

    # Flatten all edges across all KANLinears with their importance.
    ranked: list[tuple[float, str, int, int]] = []
    for kan_name, stat in activations.items():
        imp = np.asarray(stat["importance"])  # [out, in]
        out_n, in_n = imp.shape
        for o in range(out_n):
            for i in range(in_n):
                ranked.append((float(imp[o, i]), kan_name, o, i))

    # Importance descending; deterministic tie-break on (kan, out, in).
    ranked.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))

    rows: list[dict] = []
    for importance, kan_name, o, i in ranked[: int(top_k)]:
        stat = activations[kan_name]
        kan = stat["module"]
        lo = float(stat["visited_lo"][i])
        hi = float(stat["visited_hi"][i])
        fit = fit_edge(kan, o, i, lo, hi,
                       r2_threshold=r2_threshold, simplicity_tol=simplicity_tol)

        # Semantic block/name for this input edge (fallback if absent).
        entries = semantics.get(kan_name, [])
        block, name = "input", f"in{i}"
        if i < len(entries):
            block = entries[i].get("block", block)
            name = entries[i].get("name", name)

        # Faithfulness gate: keep a real formula for well-fit edges; flag the rest
        # (non-empty sentinel string so downstream string handling never breaks).
        accepted = bool(fit.get("accepted", True))
        formula = fit["formula"] if accepted else "no faithful closed form"

        rows.append({
            "kan": kan_name,
            "out": int(o),
            "in": int(i),
            "block": block,
            "name": name,
            "importance": float(importance),
            "formula": formula,
            "r2": float(fit["r2"]),
            "bic": float(fit["bic"]),
            "accepted": accepted,
        })
    return rows


def write_symbolic_csv(rows: list[dict], path: str) -> str:
    """Write the symbolic table ``rows`` to ``path`` as CSV (pandas; returns path)."""
    import pandas as pd

    columns = ["kan", "out", "in", "block", "name", "importance", "formula", "r2", "bic",
               "accepted"]
    df = pd.DataFrame(rows, columns=columns if rows else columns)
    df.to_csv(path, index=False)
    return path


def plot_top_edges(model, activations: dict, rows: list[dict], path: str, k: int = 8) -> str:
    """Plot the top-``k`` edges' activations with their fitted formulas (Agg).

    For each of the first ``k`` ``rows`` (already sorted by importance) the edge's
    full activation ``phi_{o,i}`` is sampled over its visited range and drawn
    against the fitted closed form; the formula appears in each subplot title. The
    figure is saved to ``path`` (parent created if needed) and ``path`` is returned.
    """
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = int(min(k, len(rows)))
    if k <= 0:
        # Nothing to plot -- still emit a (tiny) placeholder figure so callers get
        # a real file back.
        fig = plt.figure(figsize=(3, 2))
        fig.text(0.5, 0.5, "no edges", ha="center", va="center")
    else:
        ncols = min(4, k)
        nrows = int(np.ceil(k / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                 squeeze=False)
        flat_axes = axes.ravel()
        for ax in flat_axes[k:]:
            ax.axis("off")

        for ax, row in zip(flat_axes[:k], rows[:k]):
            stat = activations[row["kan"]]
            kan = stat["module"]
            i = int(row["in"])
            o = int(row["out"])
            lo = float(stat["visited_lo"][i])
            hi = float(stat["visited_hi"][i])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = -1.0, 1.0
            xs = np.linspace(lo, hi, 200)
            ys = _eval_edge(kan, o, i, xs)
            ax.plot(xs, ys, color="C0", lw=2, label="phi(x)")
            ax.set_title(
                f"{row['kan']} ({o},{i})\n{row['name']}: {row['formula']}",
                fontsize=8,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("phi")
            ax.legend(fontsize=7, loc="best")

        fig.suptitle("Top symbolic KAN edges (by data-aware importance)", fontsize=11)

    fig.tight_layout()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
