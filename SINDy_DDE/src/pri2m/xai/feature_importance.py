"""Dataset-level feature importance: which inputs drive the decision.

Where :func:`pri2m.xai.attribution.integrated_gradients` explains a *single*
window, this module aggregates over a *set* of windows to answer "across the
dataset, which channels / time-lags / process parameters move the chatter
decision?". Two complementary, model-respecting views:

``channel_attribution`` / ``attribution_by_class`` -- gradient-based. Run
    Integrated Gradients per window, take ``|attr|``, and average it into a
    per-channel profile (``[C]``) and a per-lag profile (``[L]``). The
    by-class variant splits that per-channel profile by the chatter label.
``permutation_importance`` -- model-agnostic. Measure the drop in chatter-AUC
    (``sigmoid(chatter_logit)`` vs labels) when one input channel -- or one
    side-feature scalar -- is shuffled across the batch, breaking only that
    feature's relationship with the target while leaving every other feature
    intact.

Heavy third-party libs (``captum`` via :mod:`pri2m.xai.attribution`,
``sklearn``, ``matplotlib``) are imported lazily inside the functions so that
importing this module never hard-fails in an environment that lacks them.
"""

from __future__ import annotations

import numpy as np
import torch


def _stack_windows(windows) -> np.ndarray:
    """Coerce a windows argument into a contiguous ``[N, L, C]`` float array.

    Accepts an ``[N, L, C]`` array/tensor or a list of ``[L, C]`` windows.
    """
    if isinstance(windows, torch.Tensor):
        arr = windows.detach().cpu().numpy()
    elif isinstance(windows, (list, tuple)):
        arr = np.stack([np.asarray(w, dtype=np.float32) for w in windows], axis=0)
    else:
        arr = np.asarray(windows)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(
            f"windows must be [N, L, C] or a list of [L, C]; got shape {arr.shape}"
        )
    return arr


def _broadcast_taus(taus, n: int) -> list[int]:
    """Normalise ``taus`` (int or length-N sequence) into a length-N int list."""
    if isinstance(taus, torch.Tensor):
        taus = taus.tolist()
    if np.isscalar(taus):
        return [int(taus)] * n
    taus = [int(t) for t in taus]
    if len(taus) != n:
        raise ValueError(f"taus must be a scalar or length-{n}; got length {len(taus)}")
    return taus


def channel_attribution(
    model,
    windows,
    taus,
    target: str = "chatter_logit",
    steps: int = 24,
    extra_scalars_list=None,
) -> dict:
    """Aggregate Integrated-Gradients ``|attribution|`` over a set of windows.

    Runs :func:`pri2m.xai.attribution.integrated_gradients` on every window,
    takes the absolute attribution ``[L, C]``, and averages it two ways.

    Parameters
    ----------
    model:
        A Phase 3 model (``forward(batch) -> dict`` with the ``target`` head).
    windows:
        ``[N, L, C]`` array/tensor, or a list of ``[L, C]`` windows.
    taus:
        Delay (samples): a single int, or a length-``N`` sequence.
    target:
        Head to attribute (``"chatter_logit"`` or ``"b_over_blim"``).
    steps:
        Riemann steps for the IG path integral.
    extra_scalars_list:
        Optional per-window side-feature dicts for a side-input model: either a
        single dict broadcast to all windows, or a length-``N`` list of dicts.
        ``None`` (default) is the plain ``{"window", "tau"}`` path. See
        :func:`pri2m.xai.attribution.integrated_gradients` for the dict format.

    Returns
    -------
    dict
        ``{"per_channel": np[C], "per_lag": np[L]}`` -- mean ``|attr|`` over
        (windows, time) and over (windows, channels) respectively.
    """
    from pri2m.xai.attribution import integrated_gradients

    arr = _stack_windows(windows)  # [N, L, C]
    n = arr.shape[0]
    tau_list = _broadcast_taus(taus, n)

    if extra_scalars_list is None:
        scalars_list = [None] * n
    elif isinstance(extra_scalars_list, dict):
        scalars_list = [extra_scalars_list] * n
    else:
        scalars_list = list(extra_scalars_list)
        if len(scalars_list) != n:
            raise ValueError(
                f"extra_scalars_list must be a dict or length-{n}; "
                f"got length {len(scalars_list)}"
            )

    abs_attrs = []  # each [L, C]
    for i in range(n):
        attr = integrated_gradients(
            model,
            arr[i],
            target=target,
            tau=tau_list[i],
            steps=steps,
            extra_scalars=scalars_list[i],
        )
        abs_attrs.append(np.abs(attr))
    stacked = np.stack(abs_attrs, axis=0)  # [N, L, C]

    per_channel = stacked.mean(axis=(0, 1))  # [C]
    per_lag = stacked.mean(axis=(0, 2))  # [L]
    return {"per_channel": per_channel, "per_lag": per_lag}


def attribution_by_class(model, windows, taus, labels, **kw) -> dict:
    """Per-channel ``|attribution|`` profile split by the chatter label.

    Runs :func:`channel_attribution` separately on the chatter (``label == 1``)
    and stable (``label == 0``) windows.

    Parameters
    ----------
    model, windows, taus:
        As in :func:`channel_attribution`.
    labels:
        Length-``N`` binary array (``1`` = chatter, ``0`` = stable).
    **kw:
        Forwarded to :func:`channel_attribution` (e.g. ``target``, ``steps``,
        ``extra_scalars_list``).

    Returns
    -------
    dict
        ``{"chatter": np[C], "stable": np[C]}`` -- the per-channel profile for
        each class. A class with no windows yields an all-zero ``[C]`` vector.
    """
    arr = _stack_windows(windows)  # [N, L, C]
    n, _, c = arr.shape
    tau_list = _broadcast_taus(taus, n)
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != n:
        raise ValueError(f"labels must be length-{n}; got {labels.shape[0]}")

    extra_scalars_list = kw.pop("extra_scalars_list", None)
    if extra_scalars_list is None or isinstance(extra_scalars_list, dict):
        scalars_per = [extra_scalars_list] * n
    else:
        scalars_per = list(extra_scalars_list)
        if len(scalars_per) != n:
            raise ValueError(
                f"extra_scalars_list must be a dict or length-{n}; "
                f"got length {len(scalars_per)}"
            )

    out = {}
    for name, label_value in (("chatter", 1), ("stable", 0)):
        idx = np.where(labels == label_value)[0]
        if idx.size == 0:
            out[name] = np.zeros(c, dtype=np.float64)
            continue
        sub_scalars = [scalars_per[i] for i in idx]
        # If all entries are None, pass None through (the plain path).
        scalars_arg = None if all(s is None for s in sub_scalars) else sub_scalars
        sub = channel_attribution(
            model,
            arr[idx],
            [tau_list[i] for i in idx],
            extra_scalars_list=scalars_arg,
            **kw,
        )
        out[name] = sub["per_channel"]
    return out


def _clone_batch(batch: dict) -> dict:
    """Shallow clone of a batch dict, cloning tensors so in-place edits are safe."""
    clone = {}
    for k, v in batch.items():
        clone[k] = v.clone() if isinstance(v, torch.Tensor) else v
    return clone


def _chatter_prob(model, batch: dict) -> np.ndarray:
    """``sigmoid(chatter_logit)`` for the batch as a 1-D numpy array."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logit = model(batch)["chatter_logit"].reshape(-1)
        prob = torch.sigmoid(logit).detach().cpu().numpy()
    if was_training:
        model.train()
    return prob


def permutation_importance(
    model,
    batch: dict,
    labels,
    side_features=(),
    n_repeats: int = 3,
    seed: int = 0,
) -> dict:
    """Permutation importance via drop in chatter-classification AUC.

    For each input channel ``c`` (and each named side-feature scalar), shuffle
    that feature's values across the batch -- breaking only its relationship to
    the target -- and measure how much the ROC-AUC of ``sigmoid(chatter_logit)``
    vs ``labels`` falls relative to the unpermuted baseline. A large drop means
    the model relies on that feature.

    Parameters
    ----------
    model:
        A Phase 3 model (``forward(batch) -> dict`` with ``chatter_logit``).
    batch:
        Batch dict ``{"window": [B, L, C], "tau": ..., + side scalars}``.
    labels:
        Length-``B`` binary chatter labels (``1`` = chatter).
    side_features:
        Names of scalar batch keys (each ``[B]``) to additionally permute,
        e.g. ``("omega", "tpf")`` for a side-input model.
    n_repeats:
        Number of shuffles to average per feature (the shuffle is random).
    seed:
        Base seed for the permutations (reproducible).

    Returns
    -------
    dict
        ``{"baseline_auc": float, "channel_drop": {c: float}, "side_drop":
        {name: float}}`` -- ``drop = baseline_auc - permuted_auc``, averaged
        over ``n_repeats``.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels).reshape(-1)
    window = batch["window"]
    if not isinstance(window, torch.Tensor):
        window = torch.as_tensor(np.asarray(window), dtype=torch.float32)
    b, _, c = window.shape

    base_batch = _clone_batch(batch)
    base_batch["window"] = window
    baseline_prob = _chatter_prob(model, base_batch)
    baseline_auc = float(roc_auc_score(labels, baseline_prob))

    rng = np.random.default_rng(seed)

    def _mean_drop(make_permuted_batch) -> float:
        drops = []
        for _ in range(int(n_repeats)):
            perm = rng.permutation(b)
            permuted_batch = make_permuted_batch(perm)
            prob = _chatter_prob(model, permuted_batch)
            auc = float(roc_auc_score(labels, prob))
            drops.append(baseline_auc - auc)
        return float(np.mean(drops))

    channel_drop = {}
    for ci in range(c):
        def make_batch(perm, ci=ci):
            pb = _clone_batch(batch)
            w = window.clone()
            w[:, :, ci] = w[perm, :, ci]  # shuffle only channel ci across samples
            pb["window"] = w
            return pb

        channel_drop[ci] = _mean_drop(make_batch)

    side_drop = {}
    for name in side_features:
        if name not in batch:
            raise ValueError(
                f"side feature {name!r} not in batch keys {sorted(batch)}"
            )

        def make_batch(perm, name=name):
            pb = _clone_batch(batch)
            scalar = batch[name]
            if isinstance(scalar, torch.Tensor):
                pb[name] = scalar.clone()[perm]
            else:
                pb[name] = np.asarray(scalar)[perm]
            return pb

        side_drop[name] = _mean_drop(make_batch)

    return {
        "baseline_auc": baseline_auc,
        "channel_drop": channel_drop,
        "side_drop": side_drop,
    }


def plot_channel_importance(result: dict, path: str) -> str:
    """Bar chart of the per-channel (and per-lag) importance profile.

    Parameters
    ----------
    result:
        Output of :func:`channel_attribution` -- a dict with a ``"per_channel"``
        ``[C]`` array (and optionally a ``"per_lag"`` ``[L]`` array).
    path:
        Output image path; the figure is written here.

    Returns
    -------
    str
        ``path`` (the file written).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_channel = np.asarray(result["per_channel"]).reshape(-1)
    has_lag = "per_lag" in result and result["per_lag"] is not None

    n_axes = 2 if has_lag else 1
    fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 3.2))
    axes = np.atleast_1d(axes)

    axes[0].bar(np.arange(per_channel.size), per_channel, color="C0")
    axes[0].set_xlabel("channel")
    axes[0].set_ylabel("mean |attribution|")
    axes[0].set_title("Per-channel importance")
    axes[0].set_xticks(np.arange(per_channel.size))

    if has_lag:
        per_lag = np.asarray(result["per_lag"]).reshape(-1)
        axes[1].plot(np.arange(per_lag.size), per_lag, marker="o", color="C1")
        axes[1].set_xlabel("time lag (step)")
        axes[1].set_ylabel("mean |attribution|")
        axes[1].set_title("Per-lag profile")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_lag_profile(per_lag, path: str) -> str:
    """Line plot of the per-lag (per-timestep) importance profile.

    Parameters
    ----------
    per_lag:
        ``[L]`` array of mean ``|attribution|`` per timestep
        (the ``"per_lag"`` entry from :func:`channel_attribution`).
    path:
        Output image path; the figure is written here.

    Returns
    -------
    str
        ``path`` (the file written).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_lag = np.asarray(per_lag).reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(np.arange(per_lag.size), per_lag, marker="o", color="C1")
    ax.set_xlabel("time lag (step)")
    ax.set_ylabel("mean |attribution|")
    ax.set_title("Per-lag importance profile")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
