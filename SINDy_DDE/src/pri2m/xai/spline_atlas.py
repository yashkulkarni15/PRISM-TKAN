"""Spline-edge atlas: the learned edge functions, named, ranked and annotated.

The other XAI views tell us which edges matter (:mod:`activations`,
:mod:`faithfulness`) and what physical quantity each input index means
(:mod:`semantics`); this module renders the actual learned edge curves for the
top-importance edges as a small-multiples figure, with real feature names,
tau-delay/side-input color coding, a data-visited band + rug, and (when a
closed form clears the faithfulness gate) the symbolic overlay from
:mod:`symbolic`. It also aggregates edge importance into per-feature bars and
a named version of the whitebox importance heatmap, so the ~14k learned
B-spline edges of a trained TKAN model become a readable atlas rather than an
opaque weight tensor.
"""
from __future__ import annotations

import numpy as np

from pri2m.xai import symbolic
from pri2m.xai.faithfulness import rank_edges
from pri2m.xai.kan_native import _iter_kanlinear

_HIDDEN_BLOCKS = {"h_prev", "r*h_prev", "hidden"}


def top_edge_entries(stats: dict, sem_map: dict, k: int = 16) -> list[dict]:
    """Top-``k`` edges by data-aware importance, named + delay/side-flagged.

    Parameters
    ----------
    stats:
        Output of :func:`pri2m.xai.activations.collect_activations`.
    sem_map:
        Output of :func:`pri2m.xai.semantics.edge_index_map`.
    k:
        Number of top edges (by :func:`pri2m.xai.faithfulness.rank_edges`) to
        return.

    Returns
    -------
    list[dict]
        One entry per edge, importance-descending:
        ``{"kan","out","in","feature","block","importance","is_delay","is_side"}``.
        Edges whose ``kan`` has no entry in ``sem_map`` fall back to
        ``feature=f"in{i}"``, ``block="input"``.
    """
    entries = []
    for kan_name, o, i, imp in rank_edges(stats)[: int(k)]:
        named = {d["index"]: d for d in sem_map.get(kan_name, [])}
        d = named.get(i, {"block": "input", "name": f"in{i}"})
        entries.append({
            "kan": kan_name, "out": int(o), "in": int(i),
            "feature": d["name"], "block": d["block"],
            "importance": float(imp),
            "is_delay": d["block"] == "x_delay",
            "is_side": d["block"] == "side",
        })
    return entries


def plot_spline_atlas(model, stats, sem_map, path, k=16, r2_threshold=0.9, n=200) -> list[dict]:
    """Small-multiples atlas of the top-``k`` learned edge functions.

    For each of the top-``k`` edges (by :func:`top_edge_entries`), plots the
    full learned activation ``phi(x)`` over the visited range (shaded band),
    a data rug, and -- when a closed form clears ``r2_threshold`` -- the
    symbolic overlay from :func:`pri2m.xai.symbolic.fit_edge`/``eval_fit``.
    Delay-block edges are titled in red, side-input edges in green.

    Returns
    -------
    list[dict]
        ``top_edge_entries`` rows enriched with ``{"formula","r2","accepted"}``
        (manifest rows) -- one per plotted edge.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    kans = dict(_iter_kanlinear(model))
    entries = top_edge_entries(stats, sem_map, k)
    ncols = 4
    nrows = int(np.ceil(len(entries) / ncols)) if entries else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(entries):]:
        ax.axis("off")
    for ax, e in zip(flat, entries):
        kan = kans[e["kan"]]
        lo = float(stats[e["kan"]]["visited_lo"][e["in"]])
        hi = float(stats[e["kan"]]["visited_hi"][e["in"]])
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            lo, hi = kan.grid_range
        curves = kan.edge_function(n)
        xs_full, ys_full = curves["x"], curves["curves"][e["out"], e["in"]]
        ax.plot(xs_full, ys_full, color="tab:blue", lw=1.5)
        ax.axvspan(lo, hi, color="tab:blue", alpha=0.08)       # data-visited band
        inputs = np.asarray(stats[e["kan"]]["inputs"])[:, e["in"]]
        rug = inputs[:: max(1, len(inputs) // 256)]
        ax.plot(rug, np.full_like(rug, ax.get_ylim()[0]), "|", color="k", ms=4, alpha=0.35)
        fit = symbolic.fit_edge(kan, e["out"], e["in"], lo, hi, r2_threshold=r2_threshold)
        e.update({"formula": fit.get("formula"), "r2": fit.get("r2"), "accepted": bool(fit.get("accepted", False))})
        if e["accepted"]:
            xs_fit = np.linspace(lo, hi, n)
            ax.plot(xs_fit, symbolic.eval_fit(fit, xs_fit), "--", color="tab:orange", lw=1.2)
            ax.set_xlabel(f"{fit['formula']}  (R²={fit['r2']:.3f})", fontsize=6)
        color = "tab:red" if e["is_delay"] else ("tab:green" if e["is_side"] else "k")
        gate = e["kan"].split(".")[-1]
        ax.set_title(f"{gate}: {e['feature']} → u{e['out']}  (imp {e['importance']:.3g})",
                     fontsize=7, color=color)
        ax.tick_params(labelsize=6)
    fig.suptitle("Top learned edge functions φ(x) — red=τ-delayed, green=side input", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return entries


def feature_importance_bars(stats: dict, sem_map: dict) -> dict:
    """Per-KANLinear summed edge importance, bucketed by named feature.

    For each ``kan_name`` present in BOTH ``stats`` and ``sem_map`` (kans
    absent from ``sem_map`` are skipped), sums ``importance[:, i]`` over
    out-units for every input ``i``, then buckets the per-input sums by their
    semantic name -- with every ``h_prev``/``r*h_prev``/``hidden`` input
    collapsed into one ``"h_prev (sum)"`` bucket per layer.

    Returns
    -------
    dict
        ``{kan_name: {feature_label: summed_importance}}``.
    """
    out: dict[str, dict[str, float]] = {}
    for kan_name, stat in stats.items():
        if kan_name not in sem_map:
            continue
        imp = np.asarray(stat["importance"])  # [out, in]
        per_input = imp.sum(axis=0)           # [in]
        named = {d["index"]: d for d in sem_map[kan_name]}
        bucket: dict[str, float] = {}
        for i, val in enumerate(per_input):
            d = named.get(i, {"block": "input", "name": f"in{i}"})
            key = "h_prev (sum)" if d["block"] in _HIDDEN_BLOCKS else d["name"]
            bucket[key] = bucket.get(key, 0.0) + float(val)
        out[kan_name] = bucket
    return out


def plot_feature_importance_bars(agg: dict, path) -> str:
    """One horizontal-bar subplot per kan from :func:`feature_importance_bars`.

    Delay-feature bars (``"delay:*"``) are red, side-feature bars
    (``"side*"``) are green, everything else (current channels, the
    collapsed ``"h_prev (sum)"`` bucket) is blue.

    Returns
    -------
    str
        The path the figure was saved to.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    names = list(agg.keys())
    n = max(1, len(names))
    fig, axes = plt.subplots(n, 1, figsize=(6.5, 2.6 * n), squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(names):]:
        ax.axis("off")
    for ax, kan_name in zip(flat, names):
        bucket = agg[kan_name]
        items = sorted(bucket.items(), key=lambda kv: kv[1])
        labels = [key for key, _ in items]
        values = [val for _, val in items]
        colors = [
            "tab:red" if lab.startswith("delay:") else
            ("tab:green" if lab.startswith("side") else "tab:blue")
            for lab in labels
        ]
        ax.barh(labels, values, color=colors)
        ax.set_title(kan_name, fontsize=9)
        ax.set_xlabel("summed importance")
        ax.tick_params(labelsize=7)
    fig.suptitle("Feature importance per KANLinear (red=τ-delayed, green=side, blue=current/hidden)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def plot_feature_named_heatmap(stats: dict, sem_map: dict, path) -> str:
    """Per-KANLinear data-aware edge-importance heatmaps, with named x-ticks.

    Same grid-of-``imshow`` layout as the whitebox report's
    ``_plot_importance_heatmaps`` (viridis, one panel per ``KANLinear``), plus:
    named x-ticks (fontsize 5, rotated 90) for input indices whose semantic
    block is NOT in ``{"h_prev","r*h_prev","hidden"}``, and a dashed vline at
    every index where the semantic block changes from the previous index.

    Returns
    -------
    str
        The path the figure was saved to.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    names = list(stats.keys())
    n = max(1, len(names))
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    for ax, name in zip(flat[:n], names):
        imp = np.asarray(stats[name]["importance"])
        im = ax.imshow(imp, aspect="auto", cmap="viridis")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("input edge")
        ax.set_ylabel("output unit")
        fig.colorbar(im, ax=ax, fraction=0.046)

        entries = sorted(sem_map.get(name, []), key=lambda d: d["index"])
        if entries:
            xticks, xlabels = [], []
            for d in entries:
                if d["block"] not in _HIDDEN_BLOCKS:
                    xticks.append(d["index"])
                    xlabels.append(d["name"])
            if xticks:
                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabels, fontsize=5, rotation=90)
            for prev, cur in zip(entries, entries[1:]):
                if prev["block"] != cur["block"]:
                    ax.axvline(cur["index"] - 0.5, color="white", lw=0.8,
                               linestyle="--", alpha=0.8)
    fig.suptitle("Named data-aware edge importance per KANLinear", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)
