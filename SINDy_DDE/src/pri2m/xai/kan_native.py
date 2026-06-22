"""KAN-native explainability.

Because :class:`~pri2m.models.kan.KANLinear` learns an explicit B-spline
activation on every input->output edge, a KAN model is interpretable *without*
a post-hoc attribution method: each edge's learned activation curve can be read
directly, low-importance edges can be pruned by their spline-weight L1, and a
human-readable symbolic summary lists the dominant edges feeding each output.

Functions
---------
``extract_spline_curves`` -- sample every ``KANLinear``'s per-edge curves.
``prune_edges``           -- per-layer kept/pruned report by mean |spline weight|
                             (non-mutating: it never zeroes the live weights).
``symbolic_summary``      -- text listing each output's top input edges by |weight|.
``plot_edge_activations`` -- render the sampled curves to a PNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pri2m.models.kan import KANLinear


def _iter_kanlinear(model):
    """Yield ``(qualified_name, KANLinear)`` for every KAN layer in ``model``.

    If ``model`` *is* a :class:`KANLinear` it yields itself under the name
    ``""`` so single-layer callers work too.
    """
    if isinstance(model, KANLinear):
        yield "", model
        return
    for name, module in model.named_modules():
        if isinstance(module, KANLinear):
            yield name, module


def extract_spline_curves(model, n: int = 200) -> dict:
    """Sample every ``KANLinear``'s per-edge B-spline activation curves.

    Parameters
    ----------
    model:
        Any module containing :class:`KANLinear` submodules (e.g. a
        ``TKANModel``), or a bare ``KANLinear``.
    n:
        Number of sweep points per curve.

    Returns
    -------
    dict
        ``{module_name: {"x": np.ndarray[n], "curves": np.ndarray[out, in, n]}}``
        with one entry per ``KANLinear``. Curves are finite.
    """
    out: dict[str, dict] = {}
    for name, module in _iter_kanlinear(model):
        sampled = module.spline_curves(n)
        out[name] = {
            "x": np.asarray(sampled["x"]),
            "curves": np.asarray(sampled["curves"]),
        }
    return out


def prune_edges(model, threshold: float) -> dict:
    """Report which edges would be pruned at ``threshold`` (non-mutating).

    Edge importance is the mean absolute spline weight over the coefficient axis
    (the same ``edge_l1`` used by :meth:`KANLinear.regularization_loss`). An edge
    is pruned when its importance is strictly below ``threshold``; ``threshold ==
    0`` therefore keeps every edge (all importances are ``>= 0``).

    The model is never mutated -- this only computes a plan.

    Returns
    -------
    dict
        ``{module_name: {"kept": int, "pruned": int,
                          "edge_l1": np.ndarray[out, in]}}``.
        ``kept + pruned`` equals the layer's edge count.
    """
    report: dict[str, dict] = {}
    for name, module in _iter_kanlinear(model):
        # [out, in] mean |coeff| per edge; detach so nothing touches autograd.
        edge_l1 = module.spline_weight.detach().abs().mean(dim=-1).cpu().numpy()
        pruned_mask = edge_l1 < float(threshold)
        pruned = int(pruned_mask.sum())
        kept = int(edge_l1.size - pruned)
        report[name] = {"kept": kept, "pruned": pruned, "edge_l1": edge_l1}
    return report


def symbolic_summary(module, top_k: int = 3) -> str:
    """Human-readable summary of the dominant input edges per output.

    For each ``KANLinear`` in ``module`` (or ``module`` itself if it is one),
    list each output unit's ``top_k`` input edges ranked by their per-edge
    importance (mean |spline weight|). Useful for a quick read of which inputs
    drive a :class:`~pri2m.models.heads.SeverityHead` output.
    """
    lines: list[str] = []
    for name, kan in _iter_kanlinear(module):
        edge_l1 = kan.spline_weight.detach().abs().mean(dim=-1).cpu().numpy()  # [out, in]
        title = name if name else type(module).__name__
        lines.append(f"{title} (out={kan.out_features}, in={kan.in_features}):")
        for o in range(kan.out_features):
            row = edge_l1[o]
            order = np.argsort(row)[::-1][: min(top_k, row.size)]
            edges = ", ".join(f"in[{int(i)}]:|w|={row[int(i)]:.4f}" for i in order)
            lines.append(f"  out[{o}] <- {edges}")
    if not lines:
        return f"{type(module).__name__}: no KANLinear layers found."
    return "\n".join(lines)


def plot_edge_activations(curves: dict, out_path) -> str:
    """Render sampled spline curves to a PNG and return the written path.

    Parameters
    ----------
    curves:
        Output of :func:`extract_spline_curves`.
    out_path:
        Destination PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless / CI-safe backend.
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_layers = max(len(curves), 1)
    fig, axes = plt.subplots(n_layers, 1, figsize=(7, 2.4 * n_layers), squeeze=False)

    for ax, (name, entry) in zip(axes[:, 0], curves.items()):
        x = np.asarray(entry["x"])
        c = np.asarray(entry["curves"])  # [out, in, n]
        out_dim, in_dim, _ = c.shape
        for o in range(out_dim):
            for i in range(in_dim):
                ax.plot(x, c[o, i], linewidth=0.8, alpha=0.7)
        ax.set_title(f"{name}  ({out_dim}x{in_dim} edges)", fontsize=9)
        ax.set_xlabel("input")
        ax.set_ylabel("spline activation")
        ax.grid(True, alpha=0.3)

    # If there were no curves the single axis stays blank but still renders.
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)
