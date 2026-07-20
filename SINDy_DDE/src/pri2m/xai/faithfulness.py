"""Faithfulness diagnostics: do the explanations actually reflect the model?

The other XAI views (``attribution``, ``activations``, ``kan_native``) tell us
*what* the model attends to. Faithfulness asks the harder question -- if we
believe an edge or an input cell is important, does *removing* it actually
change the model's behavior? This module supplies four complementary probes:

* :func:`prune_context` -- a reversible context manager that zeroes selected
  ``KANLinear`` edges (both the SiLU base path and the spline path) and restores
  them exactly on exit. Every other probe edits the model *through* it, so the
  model is never left mutated.
* :func:`pruning_curve` -- keep only the top-fraction most-important edges (by
  the data-aware importance from :func:`pri2m.xai.activations.collect_activations`)
  and watch how the chatter AUC degrades. A faithful importance ranking should
  let a small fraction of edges carry most of the AUC.
* :func:`deletion_curve` -- progressively zero the highest-|attribution| input
  cells and watch the prediction collapse. A faithful attribution map deletes
  the cells that matter first.
* :func:`randomization_sanity` -- the Adebayo et al. sanity check: an
  attribution that genuinely depends on the learned weights must change when the
  model is re-initialized. We compare attributions on the trained model vs. a
  freshly ``reset_parameters()`` copy by cosine similarity.

Heavy/optional dependencies (``sklearn``, ``matplotlib``, ``captum``) are
imported lazily inside the functions that need them, so importing this module
never hard-fails in a minimal environment.
"""

from __future__ import annotations

import contextlib
import copy

import numpy as np
import torch

from pri2m.xai.attribution import integrated_gradients


# --------------------------------------------------------------------------- #
# Reversible edge pruning
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def prune_context(model, edges_to_zero: dict):
    """Temporarily zero selected ``KANLinear`` edges, restoring them on exit.

    Parameters
    ----------
    model:
        Any ``nn.Module`` whose ``KANLinear`` sub-layers are reachable through
        ``model.named_modules()`` (e.g. :class:`pri2m.models.tkan_model.TKANModel`).
    edges_to_zero:
        ``{kan_name: [(o, i), ...]}`` -- for each named ``KANLinear``, the list of
        ``(out, in)`` edges to silence. Zeroing edge ``(o, i)`` clears BOTH paths:
        ``base_weight[o, i]`` (scalar) and ``spline_weight[o, i, :]`` (all
        ``n_coeff`` coefficients), so the edge contributes nothing.

    Yields
    ------
    ``model`` with the requested edges zeroed. On ``__exit__`` (normal OR
    exceptional) the affected ``base_weight``/``spline_weight`` slices are
    restored byte-for-byte from clones taken on enter.

    Notes
    -----
    Only the touched slices are snapshotted (not the whole state_dict), so the
    cost is proportional to the number of pruned edges, not the model size.
    """
    modules = dict(model.named_modules())
    saved: list[tuple] = []  # (module, o, i, base_clone, spline_clone)
    try:
        with torch.no_grad():
            for name, cells in edges_to_zero.items():
                if name not in modules:
                    raise KeyError(f"no module named {name!r} in model.named_modules()")
                mod = modules[name]
                for o, i in cells:
                    base_clone = mod.base_weight[o, i].detach().clone()
                    spline_clone = mod.spline_weight[o, i, :].detach().clone()
                    saved.append((mod, o, i, base_clone, spline_clone))
                    mod.base_weight[o, i] = 0.0
                    mod.spline_weight[o, i, :] = 0.0
        yield model
    finally:
        with torch.no_grad():
            # Restore in reverse so overlapping edits (if any) unwind correctly.
            for mod, o, i, base_clone, spline_clone in reversed(saved):
                mod.base_weight[o, i] = base_clone
                mod.spline_weight[o, i, :] = spline_clone


# --------------------------------------------------------------------------- #
# Edge ranking
# --------------------------------------------------------------------------- #
def rank_edges(activations: dict) -> list[tuple]:
    """Flatten every edge into ``(kan_name, o, i, importance)`` sorted desc.

    Parameters
    ----------
    activations:
        The dict returned by
        :func:`pri2m.xai.activations.collect_activations`; each value carries an
        ``"importance"`` array of shape ``[out_features, in_features]``.

    Returns
    -------
    list of ``(kan_name: str, o: int, i: int, importance: float)``
        One tuple per edge across all ``KANLinear`` layers, sorted by
        ``importance`` descending (most important first).
    """
    edges: list[tuple] = []
    for name, stats in activations.items():
        imp = np.asarray(stats["importance"])
        out_f, in_f = imp.shape
        for o in range(out_f):
            for i in range(in_f):
                edges.append((name, int(o), int(i), float(imp[o, i])))
    edges.sort(key=lambda e: e[3], reverse=True)
    return edges


# --------------------------------------------------------------------------- #
# Pruning curve
# --------------------------------------------------------------------------- #
def pruning_curve(
    model,
    batch: dict,
    labels,
    activations: dict,
    fractions=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
) -> dict:
    """Chatter AUC as a function of the fraction of most-important edges kept.

    For each fraction ``f`` we KEEP the top-``f`` edges of
    :func:`rank_edges` and zero the rest through :func:`prune_context`, then
    measure ``roc_auc_score(labels, sigmoid(chatter_logit))`` on ``batch``.

    Parameters
    ----------
    model, batch:
        The model and a batch dict ``{"window": [B, L, C], "tau": ..., ...}``.
    labels:
        Binary chatter labels, shape ``[B]`` (tensor or array-like).
    activations:
        Output of :func:`pri2m.xai.activations.collect_activations` for ``model``
        on ``batch`` -- supplies the edge importance ranking.
    fractions:
        Increasing fractions of edges to keep; ``1.0`` keeps everything and so
        reproduces the unpruned AUC exactly.

    Returns
    -------
    dict
        ``{"fraction_kept": [...], "auc": [...]}`` -- one finite AUC in ``[0, 1]``
        per fraction. The model is left unmodified (every edit is reversed).
    """
    from sklearn.metrics import roc_auc_score

    ranked = rank_edges(activations)
    n_edges = len(ranked)
    y = np.asarray(_as_numpy(labels)).reshape(-1)

    was_training = model.training
    model.eval()

    fraction_kept: list[float] = []
    aucs: list[float] = []
    try:
        for f in fractions:
            n_keep = int(round(float(f) * n_edges))
            n_keep = max(0, min(n_edges, n_keep))
            keep = set((name, o, i) for name, o, i, _ in ranked[:n_keep])

            # Everything NOT kept gets zeroed.
            to_zero: dict[str, list[tuple]] = {}
            for name, o, i, _ in ranked[n_keep:]:
                to_zero.setdefault(name, []).append((o, i))

            with prune_context(model, to_zero):
                with torch.no_grad():
                    logit = model(batch)["chatter_logit"]
                proba = torch.sigmoid(logit).reshape(-1).detach().cpu().numpy()
            aucs.append(float(roc_auc_score(y, proba)))
            fraction_kept.append(float(f))
    finally:
        if was_training:
            model.train()

    return {"fraction_kept": fraction_kept, "auc": aucs}


# --------------------------------------------------------------------------- #
# Deletion curve
# --------------------------------------------------------------------------- #
def deletion_curve(
    model,
    window,
    attr,
    tau,
    n_steps: int = 10,
    target: str = "chatter_logit",
    extra_scalars: dict | None = None,
) -> dict:
    """Model output as the highest-|attribution| input cells are deleted.

    The ``[L, C]`` cells are sorted by ``|attr|`` descending and zeroed
    cumulatively in ``n_steps`` equal-size groups; after each group the model is
    re-run and the scalar ``target`` output (mean over the batch dim) is
    recorded. A faithful attribution deletes the cells that matter first, so the
    prediction should move away from its undeleted value.

    Parameters
    ----------
    model:
        Called as ``model({"window": w, "tau": tau, ...})`` returning a dict with
        the ``target`` key (shape ``[B]`` or ``[1]``).
    window:
        Tensor/array ``[L, C]`` or ``[1, L, C]``.
    attr:
        Attribution map ``[L, C]`` aligned with ``window`` (e.g. the output of
        :func:`pri2m.xai.attribution.integrated_gradients`).
    tau:
        Delay (samples) passed through to the model.
    n_steps:
        Number of cumulative deletion steps (cells are split into this many
        groups by descending ``|attr|``).
    target:
        Which head scalar to record.
    extra_scalars:
        Optional side-input scalar features broadcast into the batch (mirrors
        :func:`pri2m.xai.attribution.integrated_gradients`).

    Returns
    -------
    dict
        ``{"fraction_deleted": [...], "pred": [...]}`` -- ``fraction_deleted`` is
        the cumulative fraction of cells zeroed before each measurement (starts at
        0.0 = nothing deleted), ``pred`` the recorded scalar at that point.
    """
    w = torch.as_tensor(np.asarray(window), dtype=torch.float32)
    if w.dim() == 2:  # [L, C] -> [1, L, C]
        w = w.unsqueeze(0)
    if w.dim() != 3:
        raise ValueError(f"window must be [L,C] or [1,L,C], got shape {tuple(w.shape)}")
    w = w.clone()

    a = np.abs(np.asarray(attr, dtype=np.float64))
    L, C = w.shape[1], w.shape[2]
    if a.shape != (L, C):
        raise ValueError(f"attr must be [L,C]=({L},{C}), got {a.shape}")

    # Cells ordered most- to least-important by |attribution|.
    order = np.argsort(a.reshape(-1))[::-1]  # flat indices into [L*C]
    n_cells = order.size

    was_training = model.training
    model.eval()

    def predict(x: torch.Tensor) -> float:
        b = {"window": x, "tau": torch.full((x.shape[0],), int(tau), dtype=torch.long)}
        if extra_scalars:
            for key, value in extra_scalars.items():
                b[key] = torch.full((x.shape[0],), float(value), dtype=torch.float32)
        with torch.no_grad():
            out = model(b)[target]
        return float(out.reshape(-1).mean().item())

    fraction_deleted: list[float] = []
    preds: list[float] = []
    try:
        # Step 0: nothing deleted yet (baseline prediction).
        fraction_deleted.append(0.0)
        preds.append(predict(w))

        # Equal-size cumulative groups of cells, highest |attr| first.
        cuts = np.linspace(0, n_cells, int(n_steps) + 1).astype(np.int64)
        for step in range(1, int(n_steps) + 1):
            for flat in order[cuts[step - 1] : cuts[step]]:
                t, c = divmod(int(flat), C)
                w[:, t, c] = 0.0
            n_del = int(cuts[step])
            fraction_deleted.append(n_del / n_cells if n_cells else 0.0)
            preds.append(predict(w))
    finally:
        if was_training:
            model.train()

    return {"fraction_deleted": fraction_deleted, "pred": preds}


# --------------------------------------------------------------------------- #
# Randomization sanity check (Adebayo et al.)
# --------------------------------------------------------------------------- #
def randomization_sanity(
    model,
    window,
    tau,
    steps: int = 24,
    target: str = "b_over_blim",
    extra_scalars: dict | None = None,
    seed: int = 0,
) -> dict:
    """Compare attributions on ``model`` vs. a re-initialized copy.

    A faithful attribution must depend on the learned weights: if we re-init the
    model (``reset_parameters()`` on every ``KANLinear``) the attribution map
    should change. We run :func:`pri2m.xai.attribution.integrated_gradients` on
    both and report the cosine similarity of the two flattened ``[L, C]`` maps --
    a value near 1.0 would mean the explanation ignores the weights (BAD).

    Parameters
    ----------
    model, window, tau, steps, target, extra_scalars:
        Forwarded to :func:`pri2m.xai.attribution.integrated_gradients`.
    seed:
        Seeds ``torch`` before re-initializing the copy, so the control map is
        reproducible.

    Returns
    -------
    dict
        ``{"cosine": float}`` -- cosine similarity in ``[-1, 1]`` between the
        trained and re-initialized attribution maps. The original ``model`` is
        untouched (the re-init happens on a ``deepcopy``).
    """
    from pri2m.models.kan import KANLinear

    attr_trained = integrated_gradients(
        model, window, target=target, tau=tau, steps=steps, extra_scalars=extra_scalars
    )

    reinit = copy.deepcopy(model)
    torch.manual_seed(int(seed))
    for mod in reinit.modules():
        if isinstance(mod, KANLinear):
            mod.reset_parameters()

    attr_reinit = integrated_gradients(
        reinit, window, target=target, tau=tau, steps=steps, extra_scalars=extra_scalars
    )

    a = attr_trained.reshape(-1).astype(np.float64)
    b = attr_reinit.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = 0.0 if denom == 0.0 else float(np.dot(a, b) / denom)
    return {"cosine": cosine}


# --------------------------------------------------------------------------- #
# Plot helpers (Agg backend; return the written path)
# --------------------------------------------------------------------------- #
def plot_pruning_curve(result: dict, path: str) -> str:
    """Plot AUC vs. fraction of edges kept; save to ``path`` and return it.

    ``result`` is the dict from :func:`pruning_curve`
    (``{"fraction_kept", "auc"}``). Uses the headless ``Agg`` backend.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(result["fraction_kept"], result["auc"], marker="o")
    ax.set_xlabel("fraction of edges kept (top-importance first)")
    ax.set_ylabel("chatter AUC")
    ax.set_title("Pruning curve (edge faithfulness)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_deletion_curve(result: dict, path: str) -> str:
    """Plot prediction vs. fraction of input cells deleted; save and return path.

    ``result`` is the dict from :func:`deletion_curve`
    (``{"fraction_deleted", "pred"}``). Uses the headless ``Agg`` backend.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(result["fraction_deleted"], result["pred"], marker="o")
    ax.set_xlabel("fraction of input cells deleted (high |attr| first)")
    ax.set_ylabel("model output")
    ax.set_title("Deletion curve (attribution faithfulness)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_numpy(x):
    """Detach a tensor / pass through an array-like to a NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
