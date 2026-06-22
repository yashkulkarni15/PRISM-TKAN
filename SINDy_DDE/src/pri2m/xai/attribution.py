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
