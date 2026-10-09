"""Do learned edge functions recur across retrained networks, and do their equations?

Curves are the full edge function ``base + spline`` on the range the data reaches
(not the knot grid). Matching is sign/scale/offset-free: ``u = 1 - cos^2`` of
mean-centred curves is the variance left unexplained by the best affine map.
"""

from __future__ import annotations

import numpy as np

from pri2m.xai.recurrence import centred_unit
from pri2m.xai.symbolic import _eval_edge

CHANNELS = ("xtpos", "ytpos", "Forcex", "Forcey")


def channel_ranges(pooled: np.ndarray, q=(1.0, 99.0)) -> dict[str, tuple[float, float]]:
    """Per-channel ``q`` percentiles of ``pooled[N, Lp, 4]`` (columns in ``CHANNELS`` order)."""
    lo, hi = np.percentile(np.asarray(pooled).reshape(-1, len(CHANNELS)), q, axis=0)
    return {c: (float(lo[k]), float(hi[k])) for k, c in enumerate(CHANNELS)}


def range_curves(model, edges: list[dict], ranges: dict[str, tuple[float, float]],
                 n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """``(xs[E, n], ys[E, n])``: each edge's full function over its channel's range."""
    xs = np.stack([np.linspace(*ranges[e["feature"].split(":")[1]], n) for e in edges])
    ys = np.stack([_eval_edge(model.get_submodule(e["kan"]), e["out"], e["in"], x)
                   for e, x in zip(edges, xs)])
    return xs, ys


def unexplained(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``u[nA, nB] = 1 - cos^2`` of mean-centred rows; NaN where either row is flat."""
    ua, ub = centred_unit(a)[0], centred_unit(b)[0]
    with np.errstate(all="ignore"):         # NaN rows propagate by design
        return 1.0 - (ua @ ub.T) ** 2


def best_match(u: np.ndarray, feats_a: list[str], feats_b: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per A edge ``(u_min, j_best)`` among B edges on the same feature; ``(nan, -1)`` if none."""
    if u.shape[1] == 0:                      # B has no edges (empty cluster)
        return np.full(len(u), np.nan), np.full(len(u), -1)
    same = np.asarray(feats_a, dtype=object)[:, None] == np.asarray(feats_b, dtype=object)[None, :]
    masked = np.where(same & np.isfinite(u), u, np.inf)
    j = masked.argmin(axis=1)
    u_min = masked[np.arange(len(u)), j]
    none = ~np.isfinite(u_min)
    return np.where(none, np.nan, u_min), np.where(none, -1, j)


def _frac_below(u: np.ndarray, thr: float) -> tuple[float, int]:
    u = np.asarray(u, dtype=float)
    u = u[np.isfinite(u)]
    return (float((u < thr).mean()) if u.size else float("nan")), int(u.size)


def recur_verdict(u_trained: np.ndarray, u_untrained: list[np.ndarray], thr: float = 0.05) -> dict:
    """Recurs iff the fraction of edges with ``u < thr`` beats the 95th pct of that fraction
    over the untrained re-inits. Unmatched (NaN) edges are excluded from every fraction."""
    frac, n_used = _frac_below(u_trained, thr)
    null = np.array([_frac_below(u, thr)[0] for u in u_untrained])
    p95 = float(np.nanpercentile(null, 95)) if np.isfinite(null).any() else float("nan")
    return {"frac_trained": frac, "null_p95": p95, "recurs": bool(frac > p95), "n_used": n_used}


def _fit(f) -> bool:
    return f is not None and str(f).strip().lower() not in ("", "none")


def family_agreement(fam_a: list[str], fam_b: list[str], j_best: np.ndarray) -> dict:
    """Do matched edges get the same fitted formula family?

    Pairs where either edge has no accepted fit (None / "" / "none") are dropped and counted in
    ``n_no_fit``. Chance = sum p_f^2 over the families of the valid pairs (both sides pooled).
    """
    matched = [(fam_a[i], fam_b[k]) for i, k in enumerate(np.asarray(j_best)) if k >= 0]
    valid = [(a, b) for a, b in matched if _fit(a) and _fit(b)]
    n = len(valid)
    same = sum(a == b for a, b in valid)
    _, counts = np.unique([str(f) for p in valid for f in p], return_counts=True)
    return {"n": n, "same": same, "n_no_fit": len(matched) - n,
            "rate": same / n if n else float("nan"),
            "chance": float(((counts / counts.sum()) ** 2).sum()) if n else float("nan")}
