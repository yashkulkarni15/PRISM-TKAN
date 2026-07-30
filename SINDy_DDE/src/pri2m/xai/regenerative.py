"""Did the network LEARN the regenerative difference ``x(t) - x(t-tau)``?

Milling chatter obeys a delay differential equation whose driving term is the
*regenerative difference* -- the current tool position minus its position one
revolution ago. The Phase 3 TKAN is never told this. Its bottom encoder gates
receive the instantaneous channels and the tau-delayed channels as **separate,
unrelated inputs** (``cat([x_t, h_prev, x_delay])``, see
:meth:`~pri2m.models.tkan.TKANCell.forward`); nothing in the architecture or the
loss subtracts them.

In a KAN every input->output edge carries its own learned 1-D function
``phi_{o,i}``. So the network can only form the regenerative difference one way:
for some out unit ``o`` the delayed-channel edge must learn (approximately) the
NEGATIVE of the instantaneous-channel edge of the same channel, because

    phi_inst(x_t) + phi_delay(x_{t-tau})  ==  phi(x_t) - phi(x_{t-tau})
    exactly when phi_delay == -phi_inst.

This module evaluates both edge functions over the range the data actually
visits and scores that relationship.

The metric
----------
For a pair of mean-centred curves ``a = phi_inst`` and ``b = phi_delay`` sampled
on a common grid,

    score = 1 - ||a + b||^2 / (||a||^2 + ||b||^2)
          = -corr(a, b) * 2g / (1 + g^2),      g = ||b|| / ||a||

``score = +1`` iff ``b == -a`` (an exact regenerative difference), ``-1`` iff
``b == +a`` (a *sum* -- the non-regenerative composition), ``~0`` when the two
edges are unrelated or wildly different in magnitude. It factorises into a shape
term (``-corr``) and a gain-balance term (``2g/(1+g^2) <= 1``, ``= 1`` only for
equal magnitudes), so a pair that has the right shape but the wrong amplitude
cannot masquerade as regeneration.

Reading the numbers honestly
----------------------------
Every KAN edge is ``base_weight * SiLU(x) + spline(x)``. If the spline is small,
BOTH edges of a pair collapse to scalar multiples of the same SiLU, so
``|corr| == 1`` for free and the sign is just ``sign(w_inst * w_delay)`` -- a
construction artifact, not learned structure. Two guards are therefore built in:

* ``spline_share`` / ``base_dominated`` per row -- the fraction of the edge's
  (centred) energy that comes from the learned spline rather than the base path;
* :func:`scan_model` also scores **cross-channel control pairs** (delayed
  channel ``c`` against instantaneous channel ``c' != c`` on the same out unit),
  which share the construction but carry no regenerative meaning. A matched
  score that does not exceed its cross-channel control is not evidence.

Edge functions are sampled through the public ``KANLinear`` API
(:meth:`~pri2m.models.kan.KANLinear.edge_function` /
:meth:`~pri2m.models.kan.KANLinear.spline_curves`), the same objects the spline
atlas plots, so this view and the atlas cannot drift apart.
"""

from __future__ import annotations

import numpy as np

from pri2m.xai.semantics import DEFAULT_CHANNELS, edge_index_map

#: ``score >= NEGATION_THRESHOLD`` == "this pair implements a difference".
NEGATION_THRESHOLD = 0.9
#: ``score <= SUMMATION_THRESHOLD`` == "this pair implements a sum".
SUMMATION_THRESHOLD = -0.9
#: Rows whose spline share falls below this are base-path (SiLU) artifacts.
BASE_DOMINATED_SPLINE_SHARE = 0.1

_DELAY_BLOCK = "x_delay"
_INST_BLOCK = "x_t"
_DELAY_PREFIX = "delay:"


# --------------------------------------------------------------------------- #
# Curve sampling (public KANLinear API only)
# --------------------------------------------------------------------------- #
class _EdgeCurves:
    """Cached ``[out, in, n_grid]`` samples of every edge of one ``KANLinear``.

    ``total`` is the full learned activation ``phi_{o,i}`` (base + spline) and
    ``spline`` its spline component; both come straight from the public
    ``KANLinear`` samplers, so a single call serves every pair of the layer.
    """

    def __init__(self, kan, n_grid: int = 801) -> None:
        full = kan.edge_function(int(n_grid))
        self.x = np.asarray(full["x"], dtype=float)
        self.total = np.asarray(full["curves"], dtype=float)
        self.spline = np.asarray(kan.spline_curves(int(n_grid))["curves"], dtype=float)
        self.grid_range = (float(kan.grid_range[0]), float(kan.grid_range[1]))

    def sample(self, out: int, idx: int, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(phi, spline)`` of edge ``(out, idx)`` interpolated onto ``xs``."""
        total = np.interp(xs, self.x, self.total[out, idx])
        spline = np.interp(xs, self.x, self.spline[out, idx])
        return total, spline


def _resolve_range(lo, hi, grid_range) -> tuple[float, float]:
    """Clip ``[lo, hi]`` into ``grid_range``; fall back to it when degenerate."""
    g_lo, g_hi = float(grid_range[0]), float(grid_range[1])
    try:
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError):
        return g_lo, g_hi
    if not np.isfinite(lo_f) or not np.isfinite(hi_f) or hi_f <= lo_f:
        return g_lo, g_hi
    lo_f = max(lo_f, g_lo)
    hi_f = min(hi_f, g_hi)
    if hi_f <= lo_f:
        return g_lo, g_hi
    return lo_f, hi_f


# --------------------------------------------------------------------------- #
# The metric
# --------------------------------------------------------------------------- #
def negation_metrics(phi_inst, phi_delay, *, center: bool = True) -> dict:
    """Score how close ``phi_delay`` is to ``-phi_inst`` on a common grid.

    Parameters
    ----------
    phi_inst, phi_delay:
        Equal-length samples of the instantaneous- and delayed-channel edge
        functions, evaluated at the SAME input values.
    center:
        Subtract each curve's mean first (default). A constant offset on either
        edge is inert -- it shifts the gate pre-activation by a constant that the
        rest of the layer absorbs -- so it must not change the verdict.

    Returns
    -------
    dict
        ``{"score", "corr", "gain", "balance", "rms_inst", "rms_delay",
        "degenerate"}``. ``score`` is ``+1`` for an exact negation, ``-1`` for an
        exact copy, ``~0`` for unrelated curves; ``corr`` is the Pearson
        correlation of the two curves and ``gain`` the amplitude ratio
        ``||phi_delay|| / ||phi_inst||``. When both curves are flat (two dead
        edges) the pair is ``degenerate``: ``score``/``corr``/``gain``/``balance``
        are ``nan`` and only the ``rms_*`` values are real.
    """
    a = np.asarray(phi_inst, dtype=float).ravel()
    b = np.asarray(phi_delay, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"curve length mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("empty curves")
    if center:
        a = a - a.mean()
        b = b - b.mean()

    ea = float(a @ a)
    eb = float(b @ b)
    n = a.size
    rms_a, rms_b = float(np.sqrt(ea / n)), float(np.sqrt(eb / n))
    nan = float("nan")
    if ea <= 0.0 and eb <= 0.0:
        return {"score": nan, "corr": nan, "gain": nan, "balance": nan,
                "rms_inst": rms_a, "rms_delay": rms_b, "degenerate": True}

    dot = float(a @ b)
    # score = 1 - ||a+b||^2 / (||a||^2 + ||b||^2); robust when one side is dead.
    score = -2.0 * dot / (ea + eb)
    if ea > 0.0 and eb > 0.0:
        corr = dot / float(np.sqrt(ea * eb))
        g = float(np.sqrt(eb / ea))
        balance = 2.0 * g / (1.0 + g * g)
    else:
        corr, balance = nan, 0.0
        g = 0.0 if eb <= 0.0 else float("inf")
    return {"score": float(score), "corr": float(corr), "gain": float(g),
            "balance": float(balance), "rms_inst": rms_a, "rms_delay": rms_b,
            "degenerate": False}


def _spline_share(total: np.ndarray, spline: np.ndarray) -> float:
    """Centred-energy fraction of an edge carried by the spline (vs SiLU base)."""
    spline_c = spline - spline.mean()
    base_c = (total - spline) - (total - spline).mean()
    es, eb = float(spline_c @ spline_c), float(base_c @ base_c)
    if es + eb <= 0.0:
        return float("nan")
    return es / (es + eb)


# --------------------------------------------------------------------------- #
# One pair
# --------------------------------------------------------------------------- #
def edge_pair_metrics(kan, out: int, inst_index: int, delay_index: int, *,
                      lo=None, hi=None, n: int = 200, n_grid: int = 801,
                      curves: _EdgeCurves | None = None) -> dict:
    """Score the (instantaneous, delayed) edge pair ``(out, i)`` / ``(out, j)``.

    Both edges are evaluated on the SAME ``n``-point sweep of
    ``[lo, hi]`` (the data-visited range, clipped into the layer's
    ``grid_range``; the full grid range is used when ``lo``/``hi`` are missing or
    degenerate), then handed to :func:`negation_metrics`.

    Returns
    -------
    dict
        :func:`negation_metrics`' keys plus ``{"out", "in_inst", "in_delay",
        "lo", "hi", "spline_share_inst", "spline_share_delay",
        "base_dominated"}``. ``base_dominated`` marks pairs where EITHER edge is
        essentially the SiLU base path, whose correlation is +-1 by construction
        and therefore carries no evidence.
    """
    cur = curves if curves is not None else _EdgeCurves(kan, n_grid)
    lo_f, hi_f = _resolve_range(lo, hi, cur.grid_range)
    xs = np.linspace(lo_f, hi_f, int(n))
    tot_i, spl_i = cur.sample(int(out), int(inst_index), xs)
    tot_d, spl_d = cur.sample(int(out), int(delay_index), xs)

    row = negation_metrics(tot_i, tot_d)
    share_i = _spline_share(tot_i, spl_i)
    share_d = _spline_share(tot_d, spl_d)
    base_dominated = bool(
        (np.isfinite(share_i) and share_i < BASE_DOMINATED_SPLINE_SHARE)
        or (np.isfinite(share_d) and share_d < BASE_DOMINATED_SPLINE_SHARE)
    )
    row.update({
        "out": int(out), "in_inst": int(inst_index), "in_delay": int(delay_index),
        "lo": lo_f, "hi": hi_f,
        "spline_share_inst": share_i, "spline_share_delay": share_d,
        "base_dominated": base_dominated,
    })
    return row


# --------------------------------------------------------------------------- #
# Which indices pair with which (derived from semantics, never hardcoded)
# --------------------------------------------------------------------------- #
def delay_pairs(model, channel_names=DEFAULT_CHANNELS) -> dict:
    """Map every delay-aware ``KANLinear`` to its (instantaneous, delayed) indices.

    Uses :func:`pri2m.xai.semantics.edge_index_map`, so the pairing follows the
    model's ACTUAL gate layout: a ``x_delay`` entry named ``"delay:<ch>"`` is
    paired with the ``x_t`` entry named ``"<ch>"``.

    Returns
    -------
    dict
        ``{kan_name: [{"channel", "inst_index", "delay_index"}, ...]}``, only for
        layers that actually have a delay block (empty dict for a model built
        with ``use_delay=False``).
    """
    sem = edge_index_map(model, channel_names=channel_names)
    out: dict[str, list[dict]] = {}
    for kan_name, entries in sem.items():
        inst = {e["name"]: int(e["index"]) for e in entries if e["block"] == _INST_BLOCK}
        pairs = []
        for e in entries:
            if e["block"] != _DELAY_BLOCK:
                continue
            channel = str(e["name"])
            if channel.startswith(_DELAY_PREFIX):
                channel = channel[len(_DELAY_PREFIX):]
            if channel in inst:
                pairs.append({"channel": channel, "inst_index": inst[channel],
                              "delay_index": int(e["index"])})
        if pairs:
            out[kan_name] = pairs
    return out


def _iter_named_kans(model) -> dict:
    from pri2m.models.kan import KANLinear

    return {name: mod for name, mod in model.named_modules()
            if isinstance(mod, KANLinear)}


def random_edge_null(model, seed: int | None = 0):
    """A deep copy of ``model`` with every ``KANLinear`` edge re-initialised.

    The untrained twin is the sharpest null available: identical shape, identical
    base-path/spline construction, zero learned structure. Scoring it with the
    same visited ranges says what :func:`scan_model` returns for a model that
    cannot possibly have learned regeneration -- so a trained score is only
    evidence insofar as it separates from this.
    """
    import copy

    import torch

    from pri2m.models.kan import KANLinear

    null = copy.deepcopy(model)
    state = torch.get_rng_state()  # reset_parameters draws from the global RNG
    try:
        if seed is not None:
            torch.manual_seed(int(seed))
        for module in null.modules():
            if isinstance(module, KANLinear):
                module.reset_parameters()
    finally:
        torch.set_rng_state(state)
    return null


def _stat(stats, kan_name, key, index, default=None):
    entry = (stats or {}).get(kan_name)
    if entry is None or key not in entry:
        return default
    arr = np.asarray(entry[key])
    try:
        return arr[index]
    except (IndexError, TypeError):
        return default


# --------------------------------------------------------------------------- #
# Whole-model scan
# --------------------------------------------------------------------------- #
def scan_model(model, stats: dict | None = None, *, channel_names=DEFAULT_CHANNELS,
               n: int = 200, n_grid: int = 801, control: bool = True) -> dict:
    """Score every (instantaneous, delayed) edge pair of a delay-aware model.

    Parameters
    ----------
    model:
        A trained ``TKANModel`` (or any module whose gates carry a delay block).
    stats:
        Optional output of :func:`pri2m.xai.activations.collect_activations`.
        Supplies the data-visited range per input edge (the pair is evaluated on
        the INTERSECTION of the two edges' visited ranges) and the data-aware
        edge importances used to weight the summary. Without it every pair is
        evaluated over the full ``grid_range`` with unit weight.
    control:
        Also score cross-channel pairs (delayed channel ``c`` vs instantaneous
        channel ``c' != c`` on the same out unit) as a null.

    Returns
    -------
    dict
        ``{"rows", "summary", "control_rows", "control_summary"}``. Each row is
        an :func:`edge_pair_metrics` dict plus ``{"kan", "channel",
        "inst_channel", "importance_inst", "importance_delay", "weight"}``;
        control rows also carry ``"control_channel"``.
    """
    pairs = delay_pairs(model, channel_names=channel_names)
    kans = _iter_named_kans(model)
    rows: list[dict] = []
    control_rows: list[dict] = []

    for kan_name, plist in pairs.items():
        kan = kans.get(kan_name)
        if kan is None:
            continue
        cur = _EdgeCurves(kan, n_grid)
        inst_of = {p["channel"]: p["inst_index"] for p in plist}
        for p in plist:
            targets = [(p["channel"], p["inst_index"], False)]
            if control:
                targets += [(ch, idx, True) for ch, idx in inst_of.items()
                            if ch != p["channel"]]
            for inst_channel, inst_index, is_control in targets:
                lo = _stat(stats, kan_name, "visited_lo", inst_index)
                hi = _stat(stats, kan_name, "visited_hi", inst_index)
                lo_d = _stat(stats, kan_name, "visited_lo", p["delay_index"])
                hi_d = _stat(stats, kan_name, "visited_hi", p["delay_index"])
                if lo is not None and lo_d is not None:
                    lo = max(float(lo), float(lo_d))
                if hi is not None and hi_d is not None:
                    hi = min(float(hi), float(hi_d))
                imp = (stats or {}).get(kan_name, {}).get("importance")
                imp = np.asarray(imp) if imp is not None else None
                for o in range(kan.out_features):
                    row = edge_pair_metrics(kan, o, inst_index, p["delay_index"],
                                            lo=lo, hi=hi, n=n, curves=cur)
                    imp_i = float(imp[o, inst_index]) if imp is not None else 1.0
                    imp_d = float(imp[o, p["delay_index"]]) if imp is not None else 1.0
                    row.update({
                        "kan": kan_name, "channel": p["channel"],
                        "inst_channel": inst_channel,
                        "importance_inst": imp_i, "importance_delay": imp_d,
                        "weight": float(min(imp_i, imp_d)),
                    })
                    if is_control:
                        row["control_channel"] = inst_channel
                        control_rows.append(row)
                    else:
                        rows.append(row)

    return {
        "rows": rows,
        "summary": summarize_rows(rows),
        "control_rows": control_rows,
        "control_summary": summarize_rows(control_rows),
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def summarize_rows(rows) -> dict:
    """Aggregate :func:`edge_pair_metrics` rows into one verdict block.

    Degenerate (dead-edge) rows are counted but excluded from every statistic.

    Returns
    -------
    dict
        ``{"n", "n_scored", "mean", "median", "weighted_mean", "max",
        "frac_negating", "frac_summing", "frac_base_dominated",
        "mean_abs_corr"}``. ``weighted_mean`` weights each pair by
        ``min(importance_inst, importance_delay)`` -- a pair only counts if BOTH
        of its edges matter to the data.
    """
    nan = float("nan")
    rows = list(rows or [])
    scored = [r for r in rows if not r.get("degenerate") and np.isfinite(r.get("score", nan))]
    if not scored:
        return {"n": len(rows), "n_scored": 0, "mean": nan, "median": nan,
                "weighted_mean": nan, "max": nan, "frac_negating": nan,
                "frac_summing": nan, "frac_base_dominated": nan,
                "mean_abs_corr": nan}
    scores = np.array([float(r["score"]) for r in scored])
    weights = np.array([float(r.get("weight", 1.0)) for r in scored])
    wsum = float(weights.sum())
    corrs = np.array([float(r.get("corr", nan)) for r in scored])
    corrs = corrs[np.isfinite(corrs)]
    # Rows that cannot know their base/spline split (symbolic surrogates) carry
    # base_dominated=None and must not be counted as "verified not dominated".
    flags = [bool(r["base_dominated"]) for r in scored
             if r.get("base_dominated") is not None]
    return {
        "n": len(rows),
        "n_scored": len(scored),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "weighted_mean": float((scores * weights).sum() / wsum) if wsum > 0 else nan,
        "max": float(scores.max()),
        "frac_negating": float((scores >= NEGATION_THRESHOLD).mean()),
        "frac_summing": float((scores <= SUMMATION_THRESHOLD).mean()),
        "frac_base_dominated": float(np.mean(flags)) if flags else nan,
        "mean_abs_corr": float(np.abs(corrs).mean()) if corrs.size else nan,
    }


def group_summaries(rows, key: str) -> dict:
    """``{value_of_key: summarize_rows(subset)}`` -- e.g. per gate or channel."""
    groups: dict[str, list[dict]] = {}
    for r in rows or []:
        groups.setdefault(str(r.get(key)), []).append(r)
    return {k: summarize_rows(v) for k, v in sorted(groups.items())}


def top_pairs(rows, k: int = 20) -> list[dict]:
    """The ``k`` highest-weight scored pairs (both edges important), score-sorted."""
    scored = [r for r in rows or [] if not r.get("degenerate")]
    scored.sort(key=lambda r: (float(r.get("weight", 0.0)), float(r.get("score", 0.0))),
                reverse=True)
    return scored[: int(k)]
