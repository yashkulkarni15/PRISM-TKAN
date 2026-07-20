"""Post-hoc attribution explainability.

Two complementary views:

``integrated_gradients`` -- Captum Integrated Gradients over the raw input
    window, attributing a chosen scalar head output (``b_over_blim`` or
    ``chatter_logit``) back to every (time, channel) cell. We wrap the
    ``forward(batch)->dict`` model in a plain ``window_tensor -> scalar``
    function so Captum can differentiate through it.
``shap_feature_summary`` -- a SHAP ``KernelExplainer`` over a small per-file
    tabular feature frame (omega, depth, noise, ...), explaining a black-box
    ``model_fn(X)->severity``.

Third-party libs (``captum``, ``shap``) are imported lazily so importing this
module never hard-fails in an environment that lacks them.
"""

from __future__ import annotations

import numpy as np
import torch


def integrated_gradients(
    model,
    window,
    target: str = "b_over_blim",
    tau=1,
    steps: int = 32,
    extra_scalars: dict[str, float] | None = None,
) -> np.ndarray:
    """Integrated-Gradients attribution of ``target`` onto the input window.

    Parameters
    ----------
    model:
        A Phase 3 model called as ``model({"window": w, "tau": tau})`` returning
        a dict with the requested ``target`` key (shape ``[B]``).
    window:
        ``np.ndarray``/tensor of shape ``[L, C]`` or ``[1, L, C]``.
    target:
        Which head scalar to attribute (``"b_over_blim"`` or ``"chatter_logit"``).
    tau:
        Delay (samples) passed through to the model.
    steps:
        Number of Riemann steps in the IG path integral.
    extra_scalars:
        Optional extra scalar batch features, e.g. ``{"omega": 3000.0,
        "tpf": 200.0}`` for a side-input :class:`TKANModel`. Each value is
        broadcast as ``torch.full((B,), v)`` into the batch dict so the wrapped
        forward satisfies the model's required side keys. ``None`` (default)
        reproduces the plain ``{"window", "tau"}`` batch exactly. The keys
        ``"window"`` and ``"tau"`` are rejected with ``ValueError`` -- they must
        flow through their dedicated arguments (``tau`` stays the long-dtype
        delay; silently overriding either would corrupt the attribution).

    Returns
    -------
    np.ndarray
        Attribution of shape ``[L, C]`` (finite).
    """
    if extra_scalars:
        reserved = {"window", "tau"} & set(extra_scalars)
        if reserved:
            raise ValueError(
                f"extra_scalars must not contain {sorted(reserved)}: 'window' and "
                f"'tau' flow through their dedicated arguments (tau stays the "
                f"long-dtype delay); overriding them would corrupt the attribution"
            )

    from captum.attr import IntegratedGradients

    w = torch.as_tensor(np.asarray(window), dtype=torch.float32)
    if w.dim() == 2:  # [L, C] -> [1, L, C]
        w = w.unsqueeze(0)
    if w.dim() != 3:
        raise ValueError(f"window must be [L,C] or [1,L,C], got shape {tuple(w.shape)}")
    w = w.clone().requires_grad_(True)

    was_training = model.training
    model.eval()

    def f(x: torch.Tensor) -> torch.Tensor:
        batch = {"window": x, "tau": torch.full((x.shape[0],), int(tau), dtype=torch.long)}
        if extra_scalars:
            for key, value in extra_scalars.items():
                batch[key] = torch.full((x.shape[0],), float(value), dtype=torch.float32)
        out = model(batch)[target]
        return out.reshape(out.shape[0])  # [B] scalar-per-sample for captum

    ig = IntegratedGradients(f)
    baseline = torch.zeros_like(w)
    attr = ig.attribute(w, baselines=baseline, n_steps=int(steps))

    if was_training:
        model.train()

    return attr[0].detach().cpu().numpy()  # [L, C]


def temporal_saliency_rescaling(
    model,
    window,
    target: str = "b_over_blim",
    tau=1,
    steps: int = 32,
    extra_scalars: dict[str, float] | None = None,
    *,
    alpha: float = 0.5,
    baseline=None,
    feature_masking: bool = True,
) -> dict:
    """Temporal Saliency Rescaling (Ismail et al., NeurIPS 2020; arXiv:2010.13924).

    Plain Integrated Gradients on a time series *conflates the time and feature
    domains* and -- for recurrent models -- systematically **under-credits the
    lagged (early) timesteps** where the regenerative/DDE delay term lives
    (Input-Cell Attention, arXiv:1910.12370). TSR removes both biases with a
    two-step, masking-based correction that scores *which lag* separately from
    *which feature*:

    1. **Time-relevance.** For each timestep ``t`` the window is masked at ``t``
       (that whole row replaced by ``baseline``) and IG recomputed; the
       time-relevance ``TR[t] = sum_{i,c} |IG(x) - IG(x masked at t)|`` measures
       how much *the entire saliency map* moves when ``t`` is removed. A lagged
       timestep whose removal reshuffles many downstream attributions scores high
       here even when its own local IG magnitude has vanished.
    2. **Feature-relevance within salient timesteps.** Timesteps whose ``TR`` is
       above the ``alpha``-quantile are "salient"; for each the per-feature
       relevance ``FR[t,c]`` is the same masked-IG delta (``feature_masking=True``,
       paper-faithful) or simply ``|IG[t,c]|`` (cheaper). Non-salient timesteps
       get zero.

    The rescaled map is ``TSR[t,c] = TR[t] * FR[t,c]`` -- time importance drives
    the row, feature importance shapes it, and the two are never conflated.

    Parameters
    ----------
    model, window, target, tau, steps, extra_scalars:
        Exactly as :func:`integrated_gradients` (each masked pass calls it).
    alpha:
        Quantile in ``[0, 1)`` of the time-relevance above which a timestep is
        treated as salient (default ``0.5`` -> the more-relevant half). ``0``
        keeps every timestep; higher values keep fewer. The single most-relevant
        timestep is always salient.
    baseline:
        Value a masked cell/row is set to (broadcast to the window shape). Default
        ``None`` -> zeros, matching the IG baseline.
    feature_masking:
        ``True`` (default) computes ``FR`` by masking each feature of a salient
        timestep and re-running IG (paper-faithful, ``O(salient*C)`` extra
        passes). ``False`` uses ``|IG|`` directly (cheap).

    Returns
    -------
    dict
        ``{"tsr": np[L, C], "ig": np[L, C], "time_relevance": np[L],
        "salient": np[L] bool, "alpha": float}``. ``tsr`` is exactly zero on
        non-salient timesteps.
    """
    w = np.asarray(window, dtype=float)
    if w.ndim == 3:
        if w.shape[0] != 1:
            raise ValueError(f"window batch must be 1, got shape {tuple(w.shape)}")
        w = w[0]
    if w.ndim != 2:
        raise ValueError(f"window must be [L,C] or [1,L,C], got shape {tuple(w.shape)}")
    L, C = w.shape

    if baseline is None:
        base = np.zeros_like(w)
    else:
        base = np.broadcast_to(np.asarray(baseline, dtype=float), w.shape).astype(float)

    def _ig(x: np.ndarray) -> np.ndarray:
        return integrated_gradients(
            model, x, target=target, tau=tau, steps=steps, extra_scalars=extra_scalars
        )

    ig0 = _ig(w)  # [L, C]

    # Step 1 -- time relevance: mask each whole timestep, measure the total move.
    time_rel = np.zeros(L, dtype=float)
    for t in range(L):
        xm = w.copy()
        xm[t, :] = base[t, :]
        time_rel[t] = float(np.sum(np.abs(ig0 - _ig(xm))))

    # Salient timesteps: time relevance above the alpha-quantile (>= keeps the max).
    if alpha <= 0.0:
        thresh = -np.inf
    else:
        thresh = float(np.quantile(time_rel, min(max(alpha, 0.0), 1.0)))
    salient = time_rel >= thresh
    if not salient.any():  # degenerate: keep at least the most-relevant timestep
        salient[int(np.argmax(time_rel))] = True

    # Step 2 -- feature relevance within salient timesteps.
    feat_rel = np.zeros_like(ig0)
    for t in range(L):
        if not salient[t]:
            continue
        if feature_masking:
            for c in range(C):
                xm = w.copy()
                xm[t, c] = base[t, c]
                feat_rel[t, c] = float(np.sum(np.abs(ig0 - _ig(xm))))
        else:
            feat_rel[t, :] = np.abs(ig0[t, :])

    tsr = feat_rel * time_rel[:, None]  # [L, C]; zero on non-salient rows
    return {
        "tsr": tsr,
        "ig": ig0,
        "time_relevance": time_rel,
        "salient": salient,
        "alpha": float(alpha),
    }


def shap_feature_summary(model_fn, feature_frame):
    """SHAP KernelExplainer over a small per-file tabular feature frame.

    Parameters
    ----------
    model_fn:
        Callable ``X(np.ndarray[n, d]) -> np.ndarray[n]`` predicting severity.
    feature_frame:
        ``pandas.DataFrame`` of per-file features (small; KernelExplainer is
        O(n) in samples and exponential in features, so keep it tiny).

    Returns
    -------
    np.ndarray
        SHAP values shaped ``[n_samples, n_features]``.
    """
    import shap

    X = np.asarray(feature_frame, dtype=float)
    # Background = the data itself (small frames) so the explainer is self-contained.
    explainer = shap.KernelExplainer(model_fn, X)
    shap_values = explainer.shap_values(X, silent=True)
    # KernelExplainer returns an array for single-output models; normalise to ndarray.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.asarray(shap_values)
