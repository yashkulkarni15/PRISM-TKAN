"""Data-aware activation statistics for every ``KANLinear`` in a TKAN model.

The existing KAN-native view ranks edges by mean |spline weight| -- which is
data-blind (it pruned 0/14k edges on the real model) and ignores both the SiLU
base path and the input distribution. :func:`collect_activations` instead runs
real windows through the model with forward hooks, captures each ``KANLinear``'s
**actual inputs**, and computes:

* ``importance[o, i]`` -- the std **over the data** of edge ``(o, i)``'s full
  output contribution ``phi_{o,i}(x_i) = base_weight*SiLU(x_i) + spline(x_i)``.
  A dead edge (zero base + spline) scores exactly 0; an edge whose contribution
  swings across the realised inputs scores high. This is the importance used by
  the symbolic, faithfulness, and physics analyses.
* ``visited_lo`` / ``visited_hi`` -- the 1st/99th percentile of each input edge's
  realised values, so curves and symbolic fits cover the range the data actually
  visits rather than the full ``grid_range``.

The collector is generic: it hooks every ``KANLinear`` and runs ``model(batch)``
once, so it works for any Phase 3 model that forwards a batch dict.
"""

from __future__ import annotations

import numpy as np
import torch

from pri2m.models.kan import KANLinear


def _edge_importance(kan: KANLinear, x: torch.Tensor) -> np.ndarray:
    """Std over rows of ``x`` of each edge's full contribution ``phi_{o,i}(x_i)``.

    ``x`` is ``[N, in_features]`` (the inputs actually fed to ``kan``). Returns
    ``[out_features, in_features]`` (population std; a zero-contribution edge is
    exactly 0).
    """
    with torch.no_grad():
        basis = kan.b_splines(x)  # [N, in, n_coeff]
        spline = torch.einsum("nic,oic->noi", basis, kan.spline_weight)
        spline = spline * kan.spline_scale  # [N, out, in]
        base_x = kan.base_activation(x)  # [N, in]
        base = kan.base_weight.unsqueeze(0) * base_x.unsqueeze(1)  # [N, out, in]
        phi = spline + base
        return phi.std(dim=0, unbiased=False).cpu().numpy()  # [out, in]


def _subsample(x: torch.Tensor, max_points: int) -> torch.Tensor:
    """Evenly down-sample rows of ``x`` to at most ``max_points`` (deterministic)."""
    n = x.shape[0]
    if n <= max_points:
        return x
    idx = np.unique(np.linspace(0, n - 1, max_points).astype(np.int64))
    return x[torch.as_tensor(idx)]


def collect_activations(model, batch: dict, max_points: int = 4096) -> dict:
    """Capture per-edge data-aware statistics for every ``KANLinear`` in ``model``.

    Parameters
    ----------
    model:
        A Phase 3 model called as ``model(batch)``.
    batch:
        The input batch dict (``{"window", "tau", ...}``). For side-input models
        the caller must include the side-feature keys.
    max_points:
        Cap on the number of input rows kept (and used for the importance std)
        per ``KANLinear`` -- recurrent cells fire once per timestep, so the raw
        row count is ``B * L``. Down-sampled evenly and deterministically.

    Returns
    -------
    dict
        ``{kanlinear_name: {"inputs": np.ndarray[N, in], "visited_lo": np.ndarray[in],
        "visited_hi": np.ndarray[in], "importance": np.ndarray[out, in],
        "module": KANLinear}}`` for every ``KANLinear`` invoked by the forward.
    """
    captured: dict[str, list[torch.Tensor]] = {}
    modules: dict[str, KANLinear] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inp, out):  # noqa: ARG001
            x = inp[0].detach().reshape(-1, module.in_features).float().cpu()
            captured.setdefault(name, []).append(x)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, KANLinear):
            modules[name] = module
            handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    out: dict[str, dict] = {}
    for name, chunks in captured.items():
        if not chunks:
            continue
        x = torch.cat(chunks, dim=0)  # [N_raw, in]
        x = _subsample(x, max_points)
        x_np = x.numpy()
        out[name] = {
            "inputs": x_np,
            "visited_lo": np.percentile(x_np, 1, axis=0),
            "visited_hi": np.percentile(x_np, 99, axis=0),
            "importance": _edge_importance(modules[name], x),
            "module": modules[name],
        }
    return out
