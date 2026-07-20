"""Hidden-state *dynamical* analysis of the TKAN encoder (Sussillo & Barak style).

Regenerative milling chatter is a **delay-differential-equation instability**: as
the axial depth crosses ``b_lim`` the milling map's dominant characteristic
multiplier crosses the unit circle and vibration self-amplifies. A recurrent
model that classifies chatter should carry that same bifurcation in its *internal*
dynamics. This module reads it out.

The object of study is the recurrent Jacobian of the encoder's KAN-GRU cell

    J_t = d h_t / d h_{t-1}   evaluated at the state (x_t, h_{t-1}, x_delay_t)

taken along the trajectory a real window drives through the hidden state. ``J_t``
is the local linearization of the hidden-state map; its **leading eigenvalue
magnitude** ``|λ_1|`` (the spectral radius) is the one-step contraction/expansion
rate. ``|λ_1| < 1`` -> perturbations decay (a stable window is a contraction),
``|λ_1| > 1`` -> perturbations grow (the model's own dynamics have gone unstable).
The hypothesis this module tests is that **chatter windows show a larger leading
``|λ|`` than stable windows** — the model has internalized the ``|λ| = 1``
bifurcation line.

How ``h_t`` is extracted
------------------------
``TKANEncoder.forward`` walks its cells step by step (``h_t = cell(x_t, h_{t-1},
x_delay_t)``) but only returns the final/context hidden state, so we re-roll the
recurrence here — matching the encoder exactly, including the temporal pooling
(:func:`pri2m.models.tkan_model.avg_pool_time`) and the kinematic delay
(:func:`pri2m.models.tkan.delayed_input`) — and capture, at every step, the tuple
``(x_t, h_{t-1}, x_delay_t)`` the cell was actually evaluated at. ``J_t`` is then
the autograd Jacobian of ``h -> cell(x_t, h, x_delay_t)`` at ``h = h_{t-1}``, via
:func:`torch.autograd.functional.jacobian`. Because the map is re-linearized at
the *real* state each step, the spectra reflect the trajectory the window drives,
not a generic operating point.

Everything here is **eval-only** (no retraining, no gradient updates to weights);
the only optimisation is the optional fixed/slow-point search
(:func:`find_slow_points`), which optimises a *state* ``h`` while the weights are
frozen.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from pri2m.models.tkan import delayed_input
from pri2m.models.tkan_model import avg_pool_time, scale_tau

__all__ = [
    "recurrent_jacobian",
    "jacobian_eigenvalues",
    "spectral_radius",
    "hidden_jacobian_spectrum",
    "window_spectral_radius",
    "stable_vs_chatter_spectra",
    "find_slow_points",
    "plot_spectral_distribution",
]


# --------------------------------------------------------------------------- #
# Small tensor helpers
# --------------------------------------------------------------------------- #
def _as_1d(t) -> torch.Tensor:
    """Coerce a scalar/1-D/``[1, D]`` input to a 1-D float tensor ``[D]``."""
    t = torch.as_tensor(t, dtype=torch.float32)
    if t.ndim == 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t.reshape(-1) if t.ndim == 0 else t


# --------------------------------------------------------------------------- #
# 1. recurrent Jacobian J_t = d h_t / d h_{t-1}
# --------------------------------------------------------------------------- #
def recurrent_jacobian(cell, x_t, h_prev, x_delay=None) -> torch.Tensor:
    """Recurrent Jacobian ``J = d cell(x_t, h, x_delay) / d h`` at ``h = h_prev``.

    ``cell`` is any recurrent cell with the :class:`pri2m.models.tkan.TKANCell`
    signature ``forward(x_t, h_prev, x_delay=None) -> h_t``. Inputs may be
    unbatched (``x_t`` shape ``[C]``, ``h_prev`` shape ``[H]``) or single-sample
    batched (``[1, C]`` / ``[1, H]``); a batch dim is added internally so the cell
    runs, then squeezed. The Jacobian is computed with autograd
    (:func:`torch.autograd.functional.jacobian`) and returned as an ``[H, H]``
    tensor (``J[i, j] = d h_t[i] / d h_prev[j]``), detached from the graph.
    """
    x_t = _as_1d(x_t)
    h0 = _as_1d(h_prev).detach().clone()
    xd = _as_1d(x_delay) if x_delay is not None else None

    def _f(h: torch.Tensor) -> torch.Tensor:
        xb = x_t.unsqueeze(0)
        hb = h.unsqueeze(0)
        xdb = xd.unsqueeze(0) if xd is not None else None
        return cell(xb, hb, xdb).squeeze(0)

    J = torch.autograd.functional.jacobian(_f, h0, create_graph=False, vectorize=True)
    return J.detach()


def jacobian_eigenvalues(J) -> np.ndarray:
    """Eigenvalues of a Jacobian, sorted by **descending magnitude**.

    Accepts a torch tensor or array-like ``[H, H]`` and returns a length-``H``
    complex ``ndarray`` whose first entry is the leading (largest-``|λ|``)
    eigenvalue. The recurrent Jacobian is generally non-symmetric, so eigenvalues
    are complex; the spectral radius is the magnitude of the first entry.
    """
    if isinstance(J, torch.Tensor):
        J = J.detach().cpu().numpy()
    J = np.asarray(J, dtype=np.float64)
    eig = np.linalg.eigvals(J)
    order = np.argsort(-np.abs(eig))
    return eig[order]


def spectral_radius(J) -> float:
    """Spectral radius ``max_i |λ_i(J)|`` — the leading eigenvalue magnitude."""
    eig = jacobian_eigenvalues(J)
    return float(np.abs(eig[0])) if eig.size else 0.0


# --------------------------------------------------------------------------- #
# 2. per-timestep spectra along a window's hidden trajectory
# --------------------------------------------------------------------------- #
def _layer_step_states(model, window, tau, layer: int):
    """Roll the encoder out and capture ``(x_t, h_{t-1}, x_delay_t)`` per step.

    Reproduces :meth:`pri2m.models.tkan.TKANEncoder.forward` exactly (temporal
    pooling + kinematic delay on layer 0, hidden-sequence input on upper layers)
    so the captured states are the ones the target ``layer``'s cell is actually
    evaluated at along this window's trajectory. Returns ``(cell, states)`` where
    ``states`` is a list of length ``L`` of ``(x_t, h_prev, x_delay)`` tensors
    (each single-sample batched, ``[1, D]`` / ``None``).
    """
    enc = model.encoder
    if layer < 0 or layer >= enc.num_layers:
        raise IndexError(f"layer {layer} out of range for {enc.num_layers}-layer encoder")

    seq = torch.as_tensor(np.asarray(window), dtype=torch.float32)
    if seq.ndim != 2:
        raise ValueError(f"window must be 2-D [L, C], got shape {tuple(seq.shape)}")
    seq = seq.unsqueeze(0)  # [1, L, C]

    pool = int(getattr(model, "pool", 1) or 1)
    if pool > 1:
        seq = avg_pool_time(seq, pool)
        tau = scale_tau(int(tau), pool)
    tau = int(tau)

    B, L, _ = seq.shape
    layer_input = seq
    states: list[tuple] = []
    with torch.no_grad():
        for li, cell in enumerate(enc.layers):
            h = seq.new_zeros(B, enc.hidden_size)
            hidden_steps = []
            captured: list[tuple] = []
            use_delay = (li == 0) and enc.use_delay
            for t in range(L):
                x_t = layer_input[:, t, :]
                x_delay = delayed_input(seq, t, tau) if use_delay else None
                captured.append(
                    (x_t.clone(), h.clone(), None if x_delay is None else x_delay.clone())
                )
                h = cell(x_t, h, x_delay)
                hidden_steps.append(h)
            if li == layer:
                return cell, captured
            layer_input = torch.stack(hidden_steps, dim=1)  # feed up
    raise RuntimeError("unreachable: target layer not captured")  # pragma: no cover


def hidden_jacobian_spectrum(model, window, tau, *, t_stride: int = 1, layer: int = 0) -> dict:
    """Recurrent-Jacobian eigenspectrum at each (strided) step of a window.

    Rolls the encoder over ``window`` (``[L, C]`` array), and at every ``t_stride``-th
    step computes ``J_t = d h_t / d h_{t-1}``, its eigenvalues, and its spectral
    radius. ``layer`` selects the recurrent cell (default 0 = the delay-aware
    bottom cell, where the regenerative term enters).

    Returns a dict with
    ``spectral_radius`` (``[n_t]`` float, leading ``|λ|`` per step),
    ``eigenvalues`` (``[n_t, H]`` complex, descending-magnitude per step),
    ``leading_eig`` (``[n_t]`` complex, the leading eigenvalue itself), and
    ``timesteps`` (``[n_t]`` int, pooled-step indices).
    """
    cell, states = _layer_step_states(model, window, tau, layer)
    L = len(states)
    ts = list(range(0, L, max(1, int(t_stride))))
    H = model.encoder.hidden_size

    radii = np.empty(len(ts), dtype=np.float64)
    eigs = np.empty((len(ts), H), dtype=np.complex128)
    leading = np.empty(len(ts), dtype=np.complex128)
    for i, t in enumerate(ts):
        x_t, h_prev, x_delay = states[t]
        J = recurrent_jacobian(cell, x_t, h_prev, x_delay)
        ev = jacobian_eigenvalues(J)
        eigs[i] = ev
        leading[i] = ev[0]
        radii[i] = float(np.abs(ev[0]))
    return {
        "spectral_radius": radii,
        "eigenvalues": eigs,
        "leading_eig": leading,
        "timesteps": np.asarray(ts, dtype=int),
    }


def window_spectral_radius(model, window, tau, *, reduce: str = "mean",
                           t_stride: int = 1, layer: int = 0) -> float:
    """Reduce a window's per-step spectral radii to one scalar.

    ``reduce`` is one of ``mean`` / ``median`` / ``max`` / ``last`` (``last`` =
    the spectral radius at the final step, i.e. of the hidden state that feeds the
    classifier).
    """
    spec = hidden_jacobian_spectrum(model, window, tau, t_stride=t_stride, layer=layer)
    sr = spec["spectral_radius"]
    if sr.size == 0:
        return 0.0
    reducers = {
        "mean": np.mean, "median": np.median, "max": np.max,
        "last": lambda a: a[-1],
    }
    if reduce not in reducers:
        raise ValueError(f"reduce must be one of {sorted(reducers)}, got {reduce!r}")
    return float(reducers[reduce](sr))


# --------------------------------------------------------------------------- #
# 3. stable-vs-chatter aggregation
# --------------------------------------------------------------------------- #
def _class_stats(radii: np.ndarray) -> dict:
    """Per-class summary of a set of leading-``|λ|`` values."""
    radii = np.asarray(radii, dtype=np.float64)
    if radii.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan,
                "frac_above_1": np.nan, "radii": radii}
    return {
        "n": int(radii.size),
        "mean": float(np.mean(radii)),
        "std": float(np.std(radii)),
        "median": float(np.median(radii)),
        "frac_above_1": float(np.mean(radii > 1.0)),
        "radii": radii,
    }


def stable_vs_chatter_spectra(model, windows, taus, labels, *, reduce: str = "mean",
                              t_stride: int = 1, layer: int = 0) -> dict:
    """Per-window spectral radius, aggregated into stable (0) vs chatter (1) stats.

    ``windows`` is ``[N, L, C]``, ``taus`` a length-``N`` sequence, ``labels`` a
    length-``N`` 0/1 array (0 = stable, 1 = chatter). Each window is reduced to one
    leading-``|λ|`` (via ``reduce``), then split by label. Returns a dict with the
    full ``radii`` (``[N]``), a ``stable`` / ``chatter`` block each carrying
    ``n``/``mean``/``std``/``median``/``frac_above_1``/``radii``, ``delta_mean``
    (``chatter.mean - stable.mean``), and ``hypothesis_supported`` (True iff
    chatter's mean leading ``|λ|`` exceeds stable's — the bifurcation hypothesis).
    """
    windows = np.asarray(windows)
    labels = np.asarray(labels).astype(int)
    taus = list(taus)
    n = len(windows)
    radii = np.empty(n, dtype=np.float64)
    for i in range(n):
        radii[i] = window_spectral_radius(
            model, windows[i], int(taus[i]), reduce=reduce, t_stride=t_stride, layer=layer
        )

    stable = _class_stats(radii[labels == 0])
    chatter = _class_stats(radii[labels == 1])
    delta_mean = float(chatter["mean"] - stable["mean"])
    return {
        "radii": radii,
        "labels": labels,
        "reduce": reduce,
        "stable": stable,
        "chatter": chatter,
        "delta_mean": delta_mean,
        "hypothesis_supported": bool(np.isfinite(delta_mean) and delta_mean > 0.0),
    }


# --------------------------------------------------------------------------- #
# 4. fixed / slow points (optional Sussillo & Barak probe)
# --------------------------------------------------------------------------- #
def find_slow_points(cell, x_t, *, h_dim: int, n_inits: int = 8, steps: int = 500,
                     lr: float = 0.1, seed: int = 0, x_delay=None) -> list[dict]:
    """Locate fixed / slow points of the (input-clamped) hidden-state map.

    Minimises the squared "speed" ``q(h) = ½ ||cell(x_t, h, x_delay) - h||²`` from
    ``n_inits`` random initial states (weights frozen — only ``h`` is optimised),
    a directly Sussillo & Barak fixed-point search. Each returned dict carries
    ``h`` (the located state, ``[h_dim]`` ndarray), ``speed`` (``||F(h) - h||`` at
    that state), and ``spectral_radius`` (leading ``|λ|`` of ``J`` there). The list
    is sorted by ascending ``speed`` (the slowest / most fixed-point-like first).
    """
    x_t = _as_1d(x_t)
    xd = _as_1d(x_delay) if x_delay is not None else None
    gen = torch.Generator().manual_seed(int(seed))

    def _step(h: torch.Tensor) -> torch.Tensor:
        xb = x_t.unsqueeze(0)
        hb = h.unsqueeze(0)
        xdb = xd.unsqueeze(0) if xd is not None else None
        return cell(xb, hb, xdb).squeeze(0)

    results: list[dict] = []
    for _ in range(int(n_inits)):
        h = torch.randn(h_dim, generator=gen).requires_grad_(True)
        opt = torch.optim.Adam([h], lr=lr)
        for _ in range(int(steps)):
            opt.zero_grad()
            speed_vec = _step(h) - h
            q = 0.5 * (speed_vec ** 2).sum()
            q.backward()
            opt.step()
        with torch.no_grad():
            speed = float(torch.linalg.norm(_step(h) - h))
        J = recurrent_jacobian(cell, x_t, h.detach(), x_delay)
        results.append({
            "h": h.detach().cpu().numpy(),
            "speed": speed,
            "spectral_radius": spectral_radius(J),
        })
    results.sort(key=lambda d: d["speed"])
    return results


# --------------------------------------------------------------------------- #
# 5. figure
# --------------------------------------------------------------------------- #
def plot_spectral_distribution(res: dict, path: str, *, title: str | None = None) -> str:
    """Plot the stable-vs-chatter leading-``|λ|`` distributions with the ``|λ|=1`` line.

    ``res`` is the dict from :func:`stable_vs_chatter_spectra`. Draws overlaid
    histograms of the per-window leading ``|λ|`` for each class plus the ``|λ| = 1``
    bifurcation line, annotated with the per-class means and ``delta_mean``.
    Returns ``path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    stable = np.asarray(res["stable"].get("radii", []), dtype=float)
    chatter = np.asarray(res["chatter"].get("radii", []), dtype=float)
    allr = np.concatenate([r for r in (stable, chatter) if r.size]) if (stable.size or chatter.size) else np.array([0.0, 1.0])
    lo, hi = float(np.min(allr)), float(np.max(allr))
    hi = max(hi, 1.05)
    bins = np.linspace(min(lo, 0.0), hi, 24)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    if stable.size:
        ax.hist(stable, bins=bins, alpha=0.55, color="#2c7fb8", label=f"stable (n={stable.size})")
    if chatter.size:
        ax.hist(chatter, bins=bins, alpha=0.55, color="#d95f0e", label=f"chatter (n={chatter.size})")
    ax.axvline(1.0, color="k", ls="--", lw=1.5, label=r"$|\lambda| = 1$ (bifurcation)")
    for r, c in ((stable, "#2c7fb8"), (chatter, "#d95f0e")):
        if r.size:
            ax.axvline(float(np.mean(r)), color=c, ls=":", lw=1.5)
    ax.set_xlabel(r"leading recurrent-Jacobian eigenvalue magnitude  $|\lambda_1|$")
    ax.set_ylabel("window count")
    dm = res.get("delta_mean", float("nan"))
    supported = res.get("hypothesis_supported", False)
    ax.set_title(
        title
        or (r"Hidden-state spectral radius: stable vs chatter"
            + f"\nΔmean = {dm:+.3f}  |λ|  (chatter − stable) — "
            + ("supported" if supported else "not supported"))
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
