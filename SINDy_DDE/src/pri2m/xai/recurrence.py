"""Exp 2: does a network's causally-necessary delay cluster recur in another network?

The mode ablation (``scripts/mode_ablation_phase3.py``) names, per network, a set
of ``encoder.L0.x_delay`` edges whose removal costs held-out AUC beyond two
matched controls -- the "necessary cluster". Exp 2 retrains the same split from
new seeds and asks whether that cluster comes back. This module holds the pure
arithmetic of that comparison; ``scripts/recurrence_phase3.py`` does the IO.

Why edge IDs cannot be compared directly
----------------------------------------
An edge is ``(kan, out, in)``. ``in`` names an input channel (``delay:xtpos``
etc.) and is the same physical quantity in every network. ``out`` names a
**hidden unit**, and a recurrent network's hidden units are permutation
symmetric: relabelling them (and every weight that touches them, consistently)
gives the identical function. Two independently trained networks therefore have
no reason to put "the same" unit at the same index, and counting shared
``(kan, out, in)`` triples measures index coincidence, not shared function (the
KAN 2.0 point about comparing KANs by function rather than by position). The
previous cross-fold comparison did exactly that, and it was additionally
confounded because all five original folds were trained with ``seed: 0`` --
the same initialisation -- so any ID agreement could be inherited from init.

Two comparisons that respect the symmetry
-----------------------------------------
1. **Curve shape on the comparable index.** For each edge in A's cluster, the
   best centred-cosine match among B's cluster edges *on the same input
   channel*. Curves are sampled on the layer's shared spline grid, so cosine is
   a genuine similarity in coefficient space. ``S`` = mean best-match cosine.
   Signed cosine is the primary statistic because the sign of a delay edge's
   learned function is part of the function; ``|cos|`` is reported alongside,
   named as such, never substituted.
2. **Aligned ID overlap.** Hidden units are matched between A and B by the
   Hungarian algorithm on ``|corr|`` of their hidden-state traces over a shared
   window batch, B's ``out`` indices are mapped through that alignment, and only
   then are IDs compared -- against a null of random unit relabellings.

Every unmeasurable quantity is ``None``/NaN and is counted, never scored as 0 or
read as False: a flat curve has no shape, and a verdict without its null is not
a verdict.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

# Rows whose centred L2 norm falls at or below this fraction of the largest norm
# in the same matrix (or below an absolute 1e-12) are degenerate: a flat curve.
_DEGENERATE_REL = 1e-9
_DEGENERATE_ABS = 1e-12


def _matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``A @ B`` with finite operands checked and numpy's spurious BLAS flags muted.

    Same reason as :func:`pri2m.xai.activation_modes._dot`: numpy >= 2.1 reports
    divide/overflow/invalid flags left on padding lanes by the vectorised BLAS
    kernel even when every input and output is finite. Checking the operands
    first keeps a genuine NaN loud.
    """
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise ValueError("non-finite values reached a matmul")
    with np.errstate(all="ignore"):
        return A @ B


def _key(edge) -> tuple[str, int, int]:
    """``(kan, out, in)`` of an edge given as a dict or a 3-sequence."""
    if isinstance(edge, dict):
        return (str(edge["kan"]), int(edge["out"]), int(edge["in"]))
    kan, out, i = edge
    return (str(kan), int(out), int(i))


# --------------------------------------------------------------------------- #
# Descriptive
# --------------------------------------------------------------------------- #

def cluster_channel_mix(edges: Iterable[dict]) -> dict[str, int]:
    """``{feature: count}`` -- which delayed channels the cluster lives on.

    Descriptive only, and comparable across networks without any alignment
    because ``feature`` is fixed by ``in``.
    """
    return dict(sorted(Counter(str(e["feature"]) for e in edges).items()))


def curve_lookup(bundle: dict) -> dict[tuple[str, int, int], int]:
    """Map ``(kan, out, in)`` -> row of ``bundle["curves"]``.

    Raises on a duplicate ID: two rows for one edge would make "the curve of this
    edge" ambiguous, and silently taking the last would hide a pooling bug.
    """
    lut: dict[tuple[str, int, int], int] = {}
    for row, rec in enumerate(bundle["records"]):
        k = _key(rec)
        if k in lut:
            raise ValueError(f"duplicate edge id {k} in bundle (rows {lut[k]} and {row})")
        lut[k] = row
    return lut


# --------------------------------------------------------------------------- #
# Curve shape
# --------------------------------------------------------------------------- #

def centred_unit(curves) -> tuple[np.ndarray, int]:
    """Centre each curve and scale it to unit L2 norm.

    Centring removes the constant offset, which other edges and the head bias
    absorb and which therefore carries no shape. A curve that is flat after
    centring has no shape at all: its row becomes NaN and is counted in the
    returned ``n_degenerate`` -- never rescaled to an ordinary-looking unit vector
    and never treated as cosine 0, either of which would be a number nobody
    measured.

    Returns
    -------
    (unit[E, P] float64, n_degenerate)
    """
    c = np.asarray(curves, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"curves must be 2-D [E, P], got shape {c.shape}")
    if c.shape[0] == 0:
        return c.copy(), 0
    c = c - c.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(c, axis=1)
    finite = np.isfinite(norms)
    scale = float(norms[finite].max()) if finite.any() else 0.0
    thresh = max(_DEGENERATE_ABS, _DEGENERATE_REL * scale)
    ok = finite & (norms > thresh)
    out = np.full_like(c, np.nan)
    out[ok] = c[ok] / norms[ok][:, None]
    return out, int((~ok).sum())


def best_match_cosine(curves_a, feats_a: Sequence[str], curves_b, feats_b: Sequence[str],
                      *, same_input: bool = True) -> dict:
    """Per A-edge best centred cosine against B's edges.

    Parameters
    ----------
    same_input:
        Restrict each A edge's candidates to B edges on the same input channel
        (the pre-registered comparison: ``in`` is the comparable index). False
        searches all of B, a looser read reported only as secondary.

    Returns
    -------
    dict
        ``best_cosine[nA]`` -- max **signed** cosine (primary: the sign of a
        learned delay function is part of the function);
        ``best_abs_cosine[nA]`` -- max ``|cos|``, sign-blind, secondary;
        ``best_index[nA]`` -- the B row of the signed best match (-1 if none);
        NaN wherever the A curve is degenerate or B has no non-degenerate
        candidate; plus ``n_unmatched`` (A edges with no candidate),
        ``n_degenerate_a``/``n_degenerate_b``.
    """
    ua, deg_a = centred_unit(curves_a)
    ub, deg_b = centred_unit(curves_b)
    feats_a = [str(f) for f in feats_a]
    feats_b = [str(f) for f in feats_b]
    if len(feats_a) != ua.shape[0] or len(feats_b) != ub.shape[0]:
        raise ValueError("feature lists must match the curve row counts")
    if ua.shape[0] and ub.shape[0] and ua.shape[1] != ub.shape[1]:
        raise ValueError(
            f"curves sampled at different resolutions ({ua.shape[1]} vs {ub.shape[1]}); "
            "cosine is only meaningful on one shared grid"
        )
    n_a = ua.shape[0]
    best = np.full(n_a, np.nan)
    best_abs = np.full(n_a, np.nan)
    best_idx = np.full(n_a, -1, dtype=np.int64)
    valid_b = np.isfinite(ub).all(axis=1) if ub.shape[0] else np.zeros(0, bool)
    fb = np.asarray(feats_b, dtype=object)
    n_unmatched = 0
    for j in range(n_a):
        if not np.isfinite(ua[j]).all():
            continue                       # degenerate A curve: counted in deg_a
        cand = valid_b & (fb == feats_a[j]) if same_input else valid_b.copy()
        idx = np.flatnonzero(cand)
        if idx.size == 0:
            n_unmatched += 1
            continue
        cos = _matmul(ub[idx], ua[j])
        k = int(np.argmax(cos))
        best[j] = float(cos[k])
        best_idx[j] = int(idx[k])
        best_abs[j] = float(np.abs(cos).max())
    return {
        "best_cosine": best,
        "best_abs_cosine": best_abs,
        "best_index": best_idx,
        "n_unmatched": int(n_unmatched),
        "n_degenerate_a": int(deg_a),
        "n_degenerate_b": int(deg_b),
        "same_input": bool(same_input),
    }


def recurrence_statistic(curves_a, feats_a, curves_b, feats_b, *,
                         same_input: bool = True) -> dict:
    """``S`` = mean best-match signed cosine of A's edges into B (NaNs excluded).

    ``S`` is ``None`` when no A edge could be scored at all -- an empty cluster,
    every curve degenerate, or no shared channel -- because a mean over nothing
    is not a low similarity. ``n_used`` says how many edges stand behind it.
    """
    m = best_match_cosine(curves_a, feats_a, curves_b, feats_b, same_input=same_input)
    used = np.isfinite(m["best_cosine"])
    n_used = int(used.sum())
    S = float(np.mean(m["best_cosine"][used])) if n_used else None
    S_abs = float(np.mean(m["best_abs_cosine"][used])) if n_used else None
    return {
        "S": S,
        "S_abs": S_abs,
        "n_used": n_used,
        "n_a": int(len(feats_a)),
        "n_b": int(len(feats_b)),
        "n_unmatched": m["n_unmatched"],
        "n_degenerate_a": m["n_degenerate_a"],
        "n_degenerate_b": m["n_degenerate_b"],
        "same_input": m["same_input"],
        "per_edge_best_cosine": m["best_cosine"],
        "per_edge_best_abs_cosine": m["best_abs_cosine"],
        "per_edge_best_index": m["best_index"],
    }


def recurrence_verdict(S, S_random, S_untrained_list, *, min_null: int = 20) -> dict:
    """The pre-committed rule: recurs iff ``S > p95(untrained)`` AND ``S > S_random``.

    * The **untrained null** asks whether B's cluster edges match A's shapes
      merely because spline curves on a shared grid look alike: B's weights are
      re-initialised and the same edge IDs re-read.
    * The **random-matched** control asks whether *any* importance-matched set of
      A's delay edges matches B's cluster as well as A's necessary one does.

    ``recurs`` is ``None`` (UNDECIDED, with ``undecided_reason``) whenever an
    input is missing or fewer than ``min_null`` finite null values exist -- a p95
    of a handful of draws is not a reference, and a missing number must never
    read as a failure.
    """
    reasons = []
    if S is None or not np.isfinite(S):
        reasons.append("S_unmeasurable")
        S = None
    if S_random is None or not np.isfinite(S_random):
        reasons.append("S_random_unmeasurable")
        S_random = None
    null = np.asarray([v for v in (S_untrained_list or [])
                       if v is not None and np.isfinite(v)], dtype=np.float64)
    if null.size < int(min_null):
        reasons.append(f"null_too_small:{null.size}<{int(min_null)}")
    p95 = float(np.percentile(null, 95)) if null.size else None
    mean = float(null.mean()) if null.size else None
    recurs = None
    if not reasons:
        recurs = bool(S > p95 and (S - S_random) > 0)
    return {
        "S": S,
        "S_random": S_random,
        "null_p95": p95,
        "null_mean": mean,
        "n_null": int(null.size),
        "margin_over_null": (S - p95) if (S is not None and p95 is not None) else None,
        "margin_over_random": (S - S_random) if (S is not None and S_random is not None) else None,
        "recurs": recurs,
        "undecided_reason": ";".join(reasons) if reasons else None,
        "rule": "recurs iff S > p95(untrained null) AND S - S_random > 0",
    }


# --------------------------------------------------------------------------- #
# Hidden-unit alignment
# --------------------------------------------------------------------------- #

def align_hidden_units(H_a, H_b) -> dict:
    """Match A's hidden units to B's by Hungarian assignment on ``|corr|``.

    ``H_a``/``H_b`` are ``[N, H]`` hidden-state traces of the two networks on the
    same inputs. Unit ``j`` of A corresponds to unit ``perm[j]`` of B.

    ``|corr|`` rather than ``corr``: a unit and its negation carry the same
    information, and a downstream sign flip is absorbed by the next weights.
    A constant unit has no defined correlation; it is given corr 0 (it can match
    nothing better than chance) and listed in ``constant_a``/``constant_b`` so a
    reader can see how much of the assignment was forced.
    """
    from scipy.optimize import linear_sum_assignment

    a = np.asarray(H_a, dtype=np.float64)
    b = np.asarray(H_b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError(f"traces must be equal-shape [N, H]; got {a.shape} vs {b.shape}")
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    sa = np.linalg.norm(a, axis=0)
    sb = np.linalg.norm(b, axis=0)
    tol_a = max(_DEGENERATE_ABS, _DEGENERATE_REL * float(sa.max(initial=0.0)))
    tol_b = max(_DEGENERATE_ABS, _DEGENERATE_REL * float(sb.max(initial=0.0)))
    ca, cb = sa <= tol_a, sb <= tol_b
    a_n = np.divide(a, sa, out=np.zeros_like(a), where=~ca)
    b_n = np.divide(b, sb, out=np.zeros_like(b), where=~cb)
    corr = _matmul(a_n.T, b_n)
    corr[ca, :] = 0.0
    corr[:, cb] = 0.0
    corr = np.clip(corr, -1.0, 1.0)
    rows, cols = linear_sum_assignment(-np.abs(corr))
    perm = np.empty(a.shape[1], dtype=np.int64)
    perm[rows] = cols
    matched = corr[np.arange(a.shape[1]), perm]
    return {
        "perm": perm,
        "matched_abs_corr": np.abs(matched),
        "matched_signed_corr": matched,
        "constant_a": [int(i) for i in np.flatnonzero(ca)],
        "constant_b": [int(i) for i in np.flatnonzero(cb)],
    }


def aligned_id_overlap(edges_a, edges_b, perm) -> int:
    """Count A edges ``(kan, out, in)`` whose ``(kan, perm[out], in)`` is in B."""
    perm = np.asarray(perm, dtype=np.int64)
    sb = {_key(e) for e in edges_b}
    n = 0
    for e in edges_a:
        kan, out, i = _key(e)
        if (kan, int(perm[out]), i) in sb:
            n += 1
    return n


def overlap_null(edges_a, edges_b, H: int, n_perm: int, seed: int,
                 *, observed: int | None = None) -> dict:
    """Aligned-overlap distribution under uniformly random relabellings of B's units.

    Returns ``expected`` (null mean), ``null_p95``, and the one-sided
    ``p = (1 + #{null >= observed}) / (1 + n_perm)`` when ``observed`` is given
    (``None`` otherwise -- no observation, no p). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(int(seed))
    ka = [_key(e) for e in edges_a]
    kb = {_key(e) for e in edges_b}
    # Vectorise: index (kan, in) pairs, then a boolean [n_pairs, H] membership.
    pairs = sorted({(k, i) for k, _, i in ka} | {(k, i) for k, _, i in kb})
    pidx = {p: n for n, p in enumerate(pairs)}
    member = np.zeros((max(len(pairs), 1), int(H)), dtype=bool)
    for k, o, i in kb:
        member[pidx[(k, i)], o] = True
    a_pair = np.asarray([pidx[(k, i)] for k, _, i in ka], dtype=np.int64)
    a_out = np.asarray([o for _, o, _ in ka], dtype=np.int64)
    null = np.empty(int(n_perm), dtype=np.int64)
    for t in range(int(n_perm)):
        perm = rng.permutation(int(H))
        null[t] = int(member[a_pair, perm[a_out]].sum()) if a_out.size else 0
    p = None
    if observed is not None and n_perm > 0:
        p = float((1 + int((null >= int(observed)).sum())) / (1 + int(n_perm)))
    return {
        "expected": float(null.mean()) if n_perm else None,
        "null_p95": float(np.percentile(null, 95)) if n_perm else None,
        "n_perm": int(n_perm),
        "observed": None if observed is None else int(observed),
        "p": p,
        "seed": int(seed),
    }


# --------------------------------------------------------------------------- #
# Model read-out (no data needed)
# --------------------------------------------------------------------------- #

def edge_curves_from_model(model, edges, *, n: int, component: str = "spline") -> np.ndarray:
    """Sample the curves of specific edges straight from ``model``'s KAN layers.

    An edge's curve is a function of its weights alone (``KANLinear.edge_function``
    on the layer's fixed grid), so no forward pass is needed. This is what lets
    the untrained null re-read B's cluster IDs from a re-initialised copy at the
    cost of a weight read per seed. Rows follow ``edges`` order.
    """
    cache: dict[str, np.ndarray] = {}
    rows = []
    for e in edges:
        kan, out, i = _key(e)
        if kan not in cache:
            cache[kan] = np.asarray(
                model.get_submodule(kan).edge_function(int(n), component=component)["curves"])
        rows.append(cache[kan][out, i])
    if not rows:
        return np.zeros((0, int(n)), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)
