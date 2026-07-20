"""Per-decision case studies: a white-box walkthrough of one chatter vs one
stable window.

Where :mod:`pri2m.xai.attribution` and :mod:`pri2m.xai.activations` give global,
data-aware views, this module zooms all the way in to *single decisions*. It
picks the most-confident-correct chatter window and the most-confident-correct
stable window from a batch, then explains each one end-to-end:

* :func:`predict` -- the model's chatter probability and ``b_over_blim`` per
  sample (the logit run through a sigmoid).
* :func:`select_cases` -- the two representative indices.
* :func:`gate_trajectories` -- the layer-0 update (``z``) and reset (``r``) gate
  values over time, captured per timestep with forward hooks on the recurrent
  ``KANLinear`` gates.
* :func:`top_firing_edges` -- the highest data-aware-importance ``KANLinear``
  edges for the single window, each tagged with its semantic name/block.
* :func:`explain_case` -- bundles the above plus per-channel integrated
  gradients into one ``why`` explanation.
* :func:`plot_case` -- renders the bundle to a multi-panel figure.

All heavy/optional plotting deps are imported lazily inside the functions, so
importing this module never hard-fails. Channels follow the canonical order
``("xtpos", "ytpos", "Forcex", "Forcey")``.
"""

from __future__ import annotations

import numpy as np
import torch

from pri2m.xai import physics
from pri2m.xai.activations import collect_activations
from pri2m.xai.attribution import integrated_gradients
from pri2m.xai.semantics import DEFAULT_CHANNELS, edge_index_map


def predict(model, batch: dict) -> dict:
    """Model chatter probability and severity for every sample in ``batch``.

    Parameters
    ----------
    model:
        A Phase 3 model called as ``model(batch)`` returning a dict with
        ``"chatter_logit"`` ``[B]`` and ``"b_over_blim"`` ``[B]``.
    batch:
        Input batch dict (``{"window": [B, L, C], "tau": ..., ...}``).

    Returns
    -------
    dict
        ``{"chatter_prob": np.ndarray[B], "b_over_blim": np.ndarray[B]}`` where
        ``chatter_prob = sigmoid(chatter_logit)``.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(batch)
            prob = torch.sigmoid(out["chatter_logit"].reshape(-1))
            bob = out["b_over_blim"].reshape(-1)
    finally:
        if was_training:
            model.train()
    return {
        "chatter_prob": prob.detach().cpu().numpy(),
        "b_over_blim": bob.detach().cpu().numpy(),
    }


def select_cases(model, batch: dict, labels) -> dict:
    """Pick a representative chatter and stable index from ``batch``.

    Among the ``label == 1`` samples, pick the index with the **highest**
    chatter probability (the most-confident-correct chatter); among the
    ``label == 0`` samples, pick the index with the **lowest** chatter
    probability (the most-confident-correct stable). Both indices are into the
    batch's sample axis.

    Parameters
    ----------
    model:
        A Phase 3 model (see :func:`predict`).
    batch:
        Input batch dict with ``B`` samples.
    labels:
        Array-like of length ``B`` with binary chatter labels (1 = chatter,
        0 = stable).

    Returns
    -------
    dict
        ``{"chatter": int, "stable": int}``.

    Raises
    ------
    ValueError
        If either class (label 1 or label 0) is absent from ``labels``.
    """
    labels = np.asarray(labels).reshape(-1)
    prob = predict(model, batch)["chatter_prob"]
    if labels.shape[0] != prob.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} != batch size {prob.shape[0]}"
        )

    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    if pos.size == 0:
        raise ValueError("no chatter (label==1) samples in batch")
    if neg.size == 0:
        raise ValueError("no stable (label==0) samples in batch")

    chatter = int(pos[int(np.argmax(prob[pos]))])
    stable = int(neg[int(np.argmin(prob[neg]))])
    return {"chatter": chatter, "stable": stable}


def _single_batch(window, tau, extra_scalars: dict | None) -> dict:
    """Build a ``B=1`` batch dict from a single ``[L, C]`` window.

    ``window`` is coerced to a float ``[1, L, C]`` tensor; ``tau`` is broadcast
    to a length-1 ``LongTensor``; any ``extra_scalars`` are broadcast as
    length-1 float tensors (for side-input models).
    """
    w = torch.as_tensor(np.asarray(window), dtype=torch.float32)
    if w.dim() == 2:
        w = w.unsqueeze(0)  # [L, C] -> [1, L, C]
    if w.dim() != 3 or w.shape[0] != 1:
        raise ValueError(f"window must be [L,C] or [1,L,C], got {tuple(w.shape)}")
    batch = {"window": w, "tau": torch.full((1,), int(tau), dtype=torch.long)}
    if extra_scalars:
        for key, value in extra_scalars.items():
            batch[key] = torch.full((1,), float(value), dtype=torch.float32)
    return batch


def gate_trajectories(model, window, tau, extra_scalars: dict | None = None) -> dict:
    """Capture the layer-0 update (``z``) and reset (``r``) gate trajectories.

    Runs a single-window forward and hooks the layer-0 ``kan_z`` and ``kan_r``
    ``KANLinear`` gates, which fire once per timestep. The captured per-step
    **outputs** are the pre-sigmoid gate logits, so this returns
    ``z = sigmoid(kan_z_out)`` and ``r = sigmoid(kan_r_out)`` -- exactly the gate
    values used inside :meth:`TKANCell.forward`.

    Parameters
    ----------
    model:
        A :class:`~pri2m.models.tkan_model.TKANModel` with a delay-aware encoder.
    window:
        ``[L, C]`` (or ``[1, L, C]``) input window.
    tau:
        Kinematic delay in samples.
    extra_scalars:
        Optional side-input scalars (see :func:`pri2m.xai.attribution.integrated_gradients`).

    Returns
    -------
    dict
        ``{"z": np.ndarray[L, H], "r": np.ndarray[L, H]}`` in ``[0, 1]``.
    """
    cell = model.encoder.layers[0]
    captured: dict[str, list[torch.Tensor]] = {"z": [], "r": []}

    def make_hook(key: str):
        def hook(module, inp, out):  # noqa: ARG001
            captured[key].append(torch.sigmoid(out.detach()).reshape(-1))
        return hook

    handles = [
        cell.kan_z.register_forward_hook(make_hook("z")),
        cell.kan_r.register_forward_hook(make_hook("r")),
    ]

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(_single_batch(window, tau, extra_scalars))
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    z = torch.stack(captured["z"], dim=0).cpu().numpy()  # [L, H]
    r = torch.stack(captured["r"], dim=0).cpu().numpy()  # [L, H]
    return {"z": z, "r": r}


def top_firing_edges(
    model, window, tau, extra_scalars: dict | None = None, top_k: int = 8
) -> list[dict]:
    """Rank the highest data-aware-importance ``KANLinear`` edges for one window.

    Runs :func:`pri2m.xai.activations.collect_activations` on the single-window
    batch (so the importance reflects *this* decision's realised activations),
    flattens every edge across every ``KANLinear``, sorts by importance
    descending, and returns the top ``top_k`` -- each tagged with its semantic
    ``block``/``name`` from :func:`pri2m.xai.semantics.edge_index_map`.

    Parameters
    ----------
    model, window, tau, extra_scalars:
        As in :func:`gate_trajectories`.
    top_k:
        Number of top edges to return.

    Returns
    -------
    list[dict]
        Up to ``top_k`` dicts, each
        ``{"kan", "out", "in", "block", "name", "importance",
        "visited_lo", "visited_hi"}`` (``out``/``in`` are the output/input edge
        indices into the ``KANLinear``; ``visited_lo``/``visited_hi`` are the
        1st/99th-percentile realised input value for that edge's input).
    """
    batch = _single_batch(window, tau, extra_scalars)
    stats = collect_activations(model, batch)
    names = edge_index_map(model)

    edges: list[dict] = []
    for kan_name, info in stats.items():
        importance = info["importance"]  # [out, in]
        lo = info["visited_lo"]  # [in]
        hi = info["visited_hi"]  # [in]
        name_map = names.get(kan_name, [])
        n_out, n_in = importance.shape
        for o in range(n_out):
            for i in range(n_in):
                meta = name_map[i] if i < len(name_map) else {}
                edges.append(
                    {
                        "kan": kan_name,
                        "out": int(o),
                        "in": int(i),
                        "block": meta.get("block", "input"),
                        "name": meta.get("name", f"in{i}"),
                        "importance": float(importance[o, i]),
                        "visited_lo": float(lo[i]),
                        "visited_hi": float(hi[i]),
                    }
                )

    edges.sort(key=lambda e: e["importance"], reverse=True)
    return edges[: int(top_k)]


def explain_case(
    model,
    window,
    tau,
    label,
    extra_scalars: dict | None = None,
    top_k: int = 8,
) -> dict:
    """Bundle a full white-box explanation of one decision.

    Combines :func:`predict` (on a ``B=1`` batch), per-channel integrated
    gradients (summed ``|IG|`` over time), the layer-0 gate means, the
    top-firing edges, and a ``why`` summary into one dict.

    Parameters
    ----------
    model:
        A :class:`~pri2m.models.tkan_model.TKANModel`.
    window:
        ``[L, C]`` (or ``[1, L, C]``) input window.
    tau:
        Kinematic delay in samples.
    label:
        The ground-truth label for this window (carried through unchanged for
        the report; the prediction itself comes from the model).
    extra_scalars:
        Optional side-input scalars.
    top_k:
        Number of top-firing edges to include.

    Returns
    -------
    dict
        ``{"label", "chatter_prob": float, "b_over_blim": float,
        "per_channel_ig": np.ndarray[C], "gate_means": {"z": np[H], "r": np[H]},
        "top_edges": list[dict], "why": {...}}`` where ``why`` is
        ``{"predicted": "chatter"|"stable", "chatter_prob", "b_over_blim",
        "top_channel": str, "leans_on_delay": bool}``. ``per_channel_ig[c]`` is
        the time-summed absolute IG of channel ``c`` onto ``chatter_logit``.
    """
    pred = predict(model, _single_batch(window, tau, extra_scalars))
    chatter_prob = float(pred["chatter_prob"][0])
    b_over_blim = float(pred["b_over_blim"][0])

    # Per-channel attribution: sum |IG| over time -> [C].
    ig = integrated_gradients(
        model, window, target="chatter_logit", tau=tau, extra_scalars=extra_scalars
    )  # [L, C]
    per_channel_ig = np.abs(ig).sum(axis=0)  # [C]

    gates = gate_trajectories(model, window, tau, extra_scalars)
    gate_means = {
        "z": gates["z"].mean(axis=0),  # [H]
        "r": gates["r"].mean(axis=0),  # [H]
    }

    edges = top_firing_edges(model, window, tau, extra_scalars, top_k=top_k)

    channels = DEFAULT_CHANNELS
    top_idx = int(np.argmax(per_channel_ig))
    top_channel = channels[top_idx] if top_idx < len(channels) else f"ch{top_idx}"
    why = {
        "predicted": "chatter" if chatter_prob >= 0.5 else "stable",
        "chatter_prob": chatter_prob,
        "b_over_blim": b_over_blim,
        "top_channel": top_channel,
        "leans_on_delay": any(e["block"] == "x_delay" for e in edges),
    }

    return {
        "label": label,
        "chatter_prob": chatter_prob,
        "b_over_blim": b_over_blim,
        "per_channel_ig": per_channel_ig,
        "gate_means": gate_means,
        "top_edges": edges,
        "why": why,
    }


def _bootstrap_ci(values: np.ndarray, n_boot: int, ci: float, rng) -> tuple[float, float]:
    """Percentile bootstrap CI for the MEAN of ``values``.

    Resamples ``values`` with replacement ``n_boot`` times, takes each resample's
    mean, and returns the ``[(1-ci)/2, 1-(1-ci)/2]`` percentiles. A percentile
    bootstrap makes no normality assumption -- the per-window prob drop is
    heavy-tailed, so a normal-approx CI would be too tight. ``n <= 1`` (or
    ``n_boot <= 0``) returns a degenerate ``(mean, mean)`` bracket.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    n = values.shape[0]
    mean = float(values.mean()) if n else float("nan")
    if n <= 1 or n_boot <= 0:
        return mean, mean
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_means = values[idx].mean(axis=1)
    lo_pct = 100.0 * (1.0 - ci) / 2.0
    hi_pct = 100.0 * (1.0 - (1.0 - ci) / 2.0)
    lo, hi = np.percentile(boot_means, [lo_pct, hi_pct])
    return float(lo), float(hi)


def delay_engagement_stats(
    model,
    batch: dict,
    labels,
    big_tau: int = 10**6,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Aggregate the delay-ablation effect over MANY chatter vs stable windows.

    The single-window :func:`explain_case` ``why["leans_on_delay"]`` verdict is an
    ``n == 1`` read -- it flips between runs on noise. This probe instead runs the
    :func:`pri2m.xai.physics.delay_ablation` intervention (zeroing the entire
    tau-delayed input path by swapping ``tau`` for ``big_tau``) across the whole
    batch and measures, PER CLASS, how far each window's chatter probability falls
    when the delayed input is removed. If the model is genuinely regenerative, the
    chatter windows -- which need the delay to be flagged -- lose far more
    probability than the stable ones, and the per-class means (with bootstrap CIs)
    make that asymmetry statistically legible instead of anecdotal.

    Parameters
    ----------
    model:
        A model called as ``model(batch) -> {"chatter_logit": [B], ...}``.
    batch:
        Batch dict ``{"window": [B, L, C], "tau": int|LongTensor[B], ...}``.
    labels:
        Binary chatter labels ``[B]`` (1 = chatter, 0 = stable).
    big_tau:
        Delay used to ablate the delayed path (``>= L`` zeros it; default
        ``10**6``), forwarded to :func:`~pri2m.xai.physics.delay_ablation`.
    n_boot:
        Bootstrap resamples for each class's mean CI (default ``1000``).
    ci:
        Confidence level of the interval (default ``0.95``).
    seed:
        Seed for the bootstrap RNG (reproducible CIs).

    Returns
    -------
    dict
        ``{"big_tau", "auc_on", "auc_off", "auc_drop", "per_window_prob_drop":
        np[B], "labels": np[B], "per_class": {"chatter": {...}, "stable":
        {...}}}``. The batch-level ``auc_*`` come straight from
        :func:`~pri2m.xai.physics.delay_ablation`. ``per_window_prob_drop[i] =
        chatter_prob_on[i] - chatter_prob_off[i]`` (positive => the delay was
        propping up that window's chatter probability). Each ``per_class`` entry
        is ``{"n", "mean_prob_drop", "std_prob_drop", "ci_low", "ci_high",
        "mean_prob_on", "mean_prob_off"}`` (CIs are percentile-bootstrap on the
        mean; a class absent from ``labels`` yields ``n == 0`` and ``nan`` stats).
    """
    labels = np.asarray(labels).reshape(-1).astype(int)

    # Batch-level AUC intervention (reuse physics.delay_ablation verbatim).
    abl = physics.delay_ablation(model, batch, labels, big_tau=big_tau)

    # Per-window probabilities on vs. with the delayed path zeroed. Reuse
    # delay_ablation's OWN zeroing mechanism (physics._with_tau -> big_tau) so the
    # ablation is byte-identical to the batch-level one above.
    prob_on = predict(model, batch)["chatter_prob"]
    prob_off = predict(model, physics._with_tau(batch, big_tau))["chatter_prob"]
    prob_on = np.asarray(prob_on, dtype=float).reshape(-1)
    prob_off = np.asarray(prob_off, dtype=float).reshape(-1)
    if prob_on.shape[0] != labels.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} != batch size {prob_on.shape[0]}"
        )
    per_window_drop = prob_on - prob_off

    rng = np.random.default_rng(seed)
    per_class: dict[str, dict] = {}
    for name, value in (("chatter", 1), ("stable", 0)):
        mask = labels == value
        drops = per_window_drop[mask]
        n = int(mask.sum())
        if n == 0:
            per_class[name] = {
                "n": 0,
                "mean_prob_drop": float("nan"),
                "std_prob_drop": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "mean_prob_on": float("nan"),
                "mean_prob_off": float("nan"),
            }
            continue
        lo, hi = _bootstrap_ci(drops, n_boot=n_boot, ci=ci, rng=rng)
        per_class[name] = {
            "n": n,
            "mean_prob_drop": float(drops.mean()),
            "std_prob_drop": float(drops.std(ddof=1)) if n > 1 else 0.0,
            "ci_low": lo,
            "ci_high": hi,
            "mean_prob_on": float(prob_on[mask].mean()),
            "mean_prob_off": float(prob_off[mask].mean()),
        }

    return {
        "big_tau": int(big_tau),
        "auc_on": float(abl["auc_on"]),
        "auc_off": float(abl["auc_off"]),
        "auc_drop": float(abl["auc_drop"]),
        "per_window_prob_drop": per_window_drop,
        "labels": labels,
        "per_class": per_class,
    }


def plot_case(case: dict, window, path: str) -> str:
    """Render a case-study bundle into a multi-panel figure.

    Panels: (1) the raw input channels over time, (2) the per-channel IG bar,
    (3) the layer-0 mean ``z`` and ``r`` gate means per hidden unit, and (4) the
    top-firing-edge importance bar.

    Parameters
    ----------
    case:
        A bundle from :func:`explain_case`.
    window:
        ``[L, C]`` (or ``[1, L, C]``) input window used for the case.
    path:
        Output image path.

    Returns
    -------
    str
        ``path`` (the figure is written there).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = np.asarray(torch.as_tensor(np.asarray(window), dtype=torch.float32))
    if w.ndim == 3:
        w = w[0]  # [1, L, C] -> [L, C]
    L, C = w.shape

    channels = DEFAULT_CHANNELS
    channel_labels = [
        channels[c] if c < len(channels) else f"ch{c}" for c in range(C)
    ]
    per_channel_ig = np.asarray(case["per_channel_ig"]).reshape(-1)
    z_mean = np.asarray(case["gate_means"]["z"]).reshape(-1)
    r_mean = np.asarray(case["gate_means"]["r"]).reshape(-1)
    edges = case["top_edges"]
    why = case.get("why", {})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel 1: input channels over time.
    ax = axes[0, 0]
    for c in range(C):
        ax.plot(np.arange(L), w[:, c], label=channel_labels[c])
    ax.set_title("Input channels")
    ax.set_xlabel("time step")
    ax.legend(fontsize=8)

    # Panel 2: per-channel integrated gradients.
    ax = axes[0, 1]
    ax.bar(np.arange(C), per_channel_ig, color="tab:orange")
    ax.set_xticks(np.arange(C))
    ax.set_xticklabels(channel_labels, rotation=30, ha="right", fontsize=8)
    ax.set_title("Per-channel |IG| (chatter_logit)")

    # Panel 3: mean z and r gates per hidden unit.
    ax = axes[1, 0]
    h = np.arange(len(z_mean))
    width = 0.4
    ax.bar(h - width / 2, z_mean, width, label="mean z (update)")
    ax.bar(h + width / 2, r_mean, width, label="mean r (reset)")
    ax.set_title("Layer-0 gate means over time")
    ax.set_xlabel("hidden unit")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # Panel 4: top-firing edge importance.
    ax = axes[1, 1]
    if edges:
        imps = [e["importance"] for e in edges]
        labels = [f"{e['kan'].split('.')[-1]}:{e['name']}" for e in edges]
        y = np.arange(len(edges))
        ax.barh(y, imps, color="tab:green")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
    ax.set_title("Top firing edges (importance)")

    predicted = why.get("predicted", "?")
    prob = why.get("chatter_prob", float("nan"))
    fig.suptitle(
        f"Case study (label={case.get('label')}) -- "
        f"predicted {predicted} (chatter_prob={prob:.3f})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
