"""White-box *physics* probes for a trained Temporal-KAN chatter model.

Regenerative milling chatter is a delay differential equation (DDE): the cutter
re-encounters a surface left ``one tooth-pass ago``, so the governing dynamics
couple the current state to a delayed state. The TKAN encoder bakes this in with
a kinematic delay ``tau = round(fs / tpf)`` (the tooth-pass period in samples)
fed to its bottom, delay-aware layer. These probes ask, white-box, whether the
trained model actually *uses* that delay -- and whether the structure it learns
lines up with the known regenerative frequencies.

Four complementary views (plus two figure helpers):

``delay_ablation`` -- compare discrimination with the real ``tau`` against a
    delay-killed run (``big_tau >= L`` zeros every delayed slice; see
    :func:`pri2m.models.tkan.delayed_input`). A large AUC drop means the delay is
    load-bearing.
``tau_sweep`` -- AUC as a function of the imposed delay; a peak at the physical
    tooth-pass period is the DDE signature.
``delayed_weight_share`` -- from data-aware edge importances, what fraction of an
    encoder gate's input drive comes from the delayed channels vs the current
    input vs the recurrent state.
``attribution_spectrum`` -- the power spectrum of the time-resolved
    Integrated-Gradients attribution, and how much of its energy sits in a band
    around the tooth-pass frequency.

Heavy/optional dependencies (``sklearn``, ``matplotlib``) are imported lazily
inside the functions so importing this module never hard-fails.
"""

from __future__ import annotations

import numpy as np
import torch

from pri2m.xai.attribution import integrated_gradients
from pri2m.xai.semantics import edge_index_map


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic sigmoid, applied elementwise."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _batch_size(batch: dict) -> int:
    """Number of samples ``B`` in a batch (from its ``window`` ``[B, L, C]``)."""
    return int(torch.as_tensor(batch["window"]).shape[0])


def _with_tau(batch: dict, tau_value: int) -> dict:
    """Shallow copy of ``batch`` with ``tau`` replaced by ``full((B,), tau_value)``.

    Every other key (window, side scalars, ...) is shared by reference; only the
    delay is swapped, so the model sees an otherwise-identical batch.
    """
    B = _batch_size(batch)
    new_batch = dict(batch)
    new_batch["tau"] = torch.full((B,), int(tau_value), dtype=torch.long)
    return new_batch


def _with_tau_vector(batch: dict, tau_vec) -> dict:
    """Shallow copy of ``batch`` with ``tau`` replaced by a per-window vector.

    ``tau_vec`` is any length-``B`` sequence of integer delays (one per window);
    every other key is shared by reference, so the model sees an otherwise
    identical batch but a *different* delay per window.
    """
    new_batch = dict(batch)
    new_batch["tau"] = torch.as_tensor(list(tau_vec), dtype=torch.long)
    return new_batch


def _own_tau_per_window(batch: dict) -> list[int]:
    """Each window's OWN kinematic delay ``tau = round(fs / tpf)`` (samples).

    The tooth-pass period differs across operating points (spindle speed / tooth
    count), so the physical delay is per-window. Priority:

    1. If ``batch`` carries per-window ``fs`` and ``tpf``, compute
       ``max(1, round(fs / tpf))`` for each window (scalars broadcast to ``B``).
    2. Any window whose ``fs``/``tpf`` is missing or non-physical (``tpf <= 0`` or
       non-finite) falls back to that window's precomputed ``batch["tau"]``.
    3. If ``batch`` has neither ``fs``/``tpf``, the precomputed ``batch["tau"]``
       (itself ``round(fs / tpf)`` at load time) is used directly.

    Raises ``ValueError`` if neither source is available.
    """
    B = _batch_size(batch)

    tau_field = batch.get("tau")
    if tau_field is not None:
        tau_field = torch.as_tensor(tau_field).reshape(-1).cpu().numpy()

    fs = batch.get("fs")
    tpf = batch.get("tpf")
    if fs is not None and tpf is not None:
        fs_arr = torch.as_tensor(fs, dtype=torch.float64).reshape(-1).cpu().numpy()
        tpf_arr = torch.as_tensor(tpf, dtype=torch.float64).reshape(-1).cpu().numpy()
        if fs_arr.size == 1:
            fs_arr = np.broadcast_to(fs_arr, (B,))
        if tpf_arr.size == 1:
            tpf_arr = np.broadcast_to(tpf_arr, (B,))
        own: list[int] = []
        for i in range(B):
            f, p = float(fs_arr[i]), float(tpf_arr[i])
            if np.isfinite(f) and f > 0 and np.isfinite(p) and p > 0:
                own.append(max(1, int(round(f / p))))
            elif tau_field is not None:
                own.append(max(1, int(tau_field[i])))
            else:
                raise ValueError(
                    f"window {i}: fs/tpf are non-physical (fs={f}, tpf={p}) and no "
                    f"tau fallback is present"
                )
        return own

    if tau_field is not None:
        return [max(1, int(v)) for v in tau_field]

    raise ValueError(
        "tau_sweep_relative needs per-window 'fs'/'tpf' (or a 'tau' field) to "
        "derive each window's own delay"
    )


def _chatter_proba(model, batch: dict) -> np.ndarray:
    """Run ``model`` and return ``sigmoid(chatter_logit)`` as ``np[B]``."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logit = model(batch)["chatter_logit"]
    finally:
        if was_training:
            model.train()
    logit = logit.detach().reshape(-1).cpu().numpy()
    return _sigmoid(logit)


def _b_over_blim(model, batch: dict) -> np.ndarray:
    """Run ``model`` and return the ``b_over_blim`` head as ``np[B]``."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(batch)["b_over_blim"]
    finally:
        if was_training:
            model.train()
    return out.detach().reshape(-1).cpu().numpy()


def _auc(labels, proba: np.ndarray) -> float:
    """ROC-AUC of ``proba`` against binary ``labels`` (lazy sklearn import)."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(labels), np.asarray(proba)))


# --------------------------------------------------------------------------- #
# 1. Delay ablation
# --------------------------------------------------------------------------- #
def delay_ablation(model, batch: dict, labels, big_tau: int = 10**6) -> dict:
    """Discrimination with the real delay vs with the delay removed.

    Runs ``model`` twice on the same window: once with ``batch["tau"]`` (the
    kinematic delay) and once with ``tau`` replaced by ``big_tau``. Because
    :func:`pri2m.models.tkan.delayed_input` returns zeros for every step where
    ``t - tau < 0``, any ``big_tau >= L`` zeros out the *entire* delayed-input
    path -- the delay is ablated without touching a single weight or the
    architecture. A large ``auc_drop`` means the regenerative (delayed) term is
    doing the discriminating.

    Parameters
    ----------
    model:
        A model called as ``model(batch) -> {"chatter_logit": [B], ...}``.
    batch:
        Batch dict ``{"window": [B, L, C], "tau": int|LongTensor[B], ...}``. If
        it also carries a ``"target"`` (b_over_blim ground truth, shape ``[B]``),
        the regression MAE on each run is reported too.
    labels:
        Binary chatter labels ``[B]`` for the AUC.
    big_tau:
        Delay value used to ablate the delayed path (default ``10**6``; any
        value ``>= L`` works).

    Returns
    -------
    dict
        ``{"auc_on", "auc_off", "auc_drop"}`` with ``auc = roc_auc_score(labels,
        sigmoid(chatter_logit))`` and ``auc_drop = auc_on - auc_off``. If
        ``batch`` carried a ``"target"``, also ``{"mae_on", "mae_off"}`` (the
        mean absolute error of ``b_over_blim`` vs that target on each run).
    """
    proba_on = _chatter_proba(model, batch)
    proba_off = _chatter_proba(model, _with_tau(batch, big_tau))

    auc_on = _auc(labels, proba_on)
    auc_off = _auc(labels, proba_off)
    result = {"auc_on": auc_on, "auc_off": auc_off, "auc_drop": auc_on - auc_off}

    target = batch.get("target")
    if target is None:
        target = batch.get("b_over_blim")
    if target is not None:
        tgt = torch.as_tensor(target).detach().reshape(-1).cpu().numpy()
        pred_on = _b_over_blim(model, batch)
        pred_off = _b_over_blim(model, _with_tau(batch, big_tau))
        result["mae_on"] = float(np.mean(np.abs(pred_on - tgt)))
        result["mae_off"] = float(np.mean(np.abs(pred_off - tgt)))
    return result


# --------------------------------------------------------------------------- #
# 2. Tau sweep
# --------------------------------------------------------------------------- #
def tau_sweep(model, batch: dict, labels, taus) -> dict:
    """AUC as a function of the imposed kinematic delay.

    For each ``t`` in ``taus`` the batch's ``tau`` is replaced by
    ``torch.full((B,), int(t))`` and the chatter AUC re-measured. The physical
    signature of regenerative chatter is a peak at the true tooth-pass period.

    Parameters
    ----------
    model:
        A model called as ``model(batch) -> {"chatter_logit": [B], ...}``.
    batch:
        Batch dict ``{"window": [B, L, C], ...}``; its ``tau`` is overwritten per
        sweep point.
    labels:
        Binary chatter labels ``[B]``.
    taus:
        Iterable of integer delays (samples) to evaluate.

    Returns
    -------
    dict
        ``{"taus": [int, ...], "auc": [float, ...]}`` aligned elementwise.
    """
    tau_list = [int(t) for t in taus]
    aucs = []
    for t in tau_list:
        proba = _chatter_proba(model, _with_tau(batch, t))
        aucs.append(_auc(labels, proba))
    return {"taus": tau_list, "auc": aucs}


def tau_sweep_relative(
    model,
    batch: dict,
    labels,
    factors=(0.5, 1.0, 2.0),
    *,
    big_tau: int = 10**6,
    own_tau=None,
) -> dict:
    """AUC vs an imposed delay measured **relative to each window's own tau**.

    The global :func:`tau_sweep` imposes a single delay (in samples) on *every*
    window. When the sample spans heterogeneous operating points -- different
    spindle speeds / tooth-pass frequencies, hence different physical delays --
    no single swept delay is optimal for all of them, so the peak blurs. This
    sweep instead imposes, for each ``factor`` in ``factors``, a delay of
    ``round(factor * tau_w)`` on window ``w``, where ``tau_w`` is *that window's*
    own delay ``round(fs / tpf)`` (see :func:`_own_tau_per_window`). Thus
    ``factor == 1.0`` reproduces every window's physical delay, ``0.5`` a
    half-period, ``2.0`` two periods -- each relative to the window's own tooth
    pass rather than a shared global value.

    The ``big_tau`` no-delay baseline (an infinite delay that zeros the entire
    delayed-input path for every window) is measured separately; it is exactly
    the :func:`delay_ablation` "off" run, so ``baseline_auc`` equals that path.

    Parameters
    ----------
    model:
        A model called as ``model(batch) -> {"chatter_logit": [B], ...}`` that
        honours a per-window ``LongTensor`` ``tau``.
    batch:
        Batch dict ``{"window": [B, L, C], "tau": LongTensor[B], "fs": [B],
        "tpf": [B], ...}``; ``fs``/``tpf`` (falling back to ``tau``) give each
        window its own delay. ``tau`` is overwritten per sweep point.
    labels:
        Binary chatter labels ``[B]``.
    factors:
        Iterable of relative multipliers of each window's own tau (default
        ``(0.5, 1.0, 2.0)``).
    big_tau:
        Delay used for the no-delay baseline (default ``10**6``; any ``>= L``).
    own_tau:
        Optional explicit per-window own-tau ``[B]``; if given it overrides the
        ``fs``/``tpf``/``tau`` derivation.

    Returns
    -------
    dict
        ``{"factors": [float, ...], "auc": [float, ...], "imposed_tau":
        [[int, ...], ...], "own_tau": [int, ...], "baseline_auc": float,
        "big_tau": int}``. ``factors``/``auc``/``imposed_tau`` are aligned
        elementwise; each ``imposed_tau[k]`` is the per-window delay actually
        imposed for ``factors[k]``. ``own_tau`` is the per-window physical delay
        used as the reference.
    """
    factor_list = [float(f) for f in factors]
    if own_tau is None:
        own = _own_tau_per_window(batch)
    else:
        own_seq = own_tau.tolist() if isinstance(own_tau, torch.Tensor) else list(own_tau)
        own = [max(1, int(v)) for v in own_seq]

    B = _batch_size(batch)
    if len(own) != B:
        raise ValueError(f"own_tau length {len(own)} != batch size {B}")

    aucs: list[float] = []
    imposed_all: list[list[int]] = []
    for f in factor_list:
        imposed = [max(1, int(round(f * o))) for o in own]
        imposed_all.append(imposed)
        proba = _chatter_proba(model, _with_tau_vector(batch, imposed))
        aucs.append(_auc(labels, proba))

    baseline = [int(big_tau)] * B
    baseline_auc = _auc(labels, _chatter_proba(model, _with_tau_vector(batch, baseline)))

    return {
        "factors": factor_list,
        "auc": aucs,
        "imposed_tau": imposed_all,
        "own_tau": [int(o) for o in own],
        "baseline_auc": baseline_auc,
        "big_tau": int(big_tau),
    }


# --------------------------------------------------------------------------- #
# 3. Delayed weight share
# --------------------------------------------------------------------------- #
#: Semantic input blocks of an encoder gate collapsed onto the three physical
#: drivers: the current input, the recurrent state, and the delayed input.
_BLOCK_TO_DRIVER = {
    "x_t": "x_t",
    "h_prev": "h_prev",
    "r*h_prev": "h_prev",  # reset-gated h_prev still reads the recurrent state
    "x_delay": "x_delay",
}
_DRIVERS = ("x_t", "h_prev", "x_delay")


def _gate_block_shares(entries: list[dict], importance: np.ndarray) -> dict:
    """Per-driver importance fractions for one gate (or ``None`` if no drive).

    ``entries`` is the gate's :func:`edge_index_map` list (one dict per input
    edge, in index order); ``importance`` is the gate's ``[out, in]`` data-aware
    importance. Each input edge's importance is summed over outputs, then over
    the edges that belong to each driver block; the three driver totals are
    normalized to sum 1. A gate whose three drivers carry zero total importance
    returns ``None`` (it contributes nothing and must not divide by zero).
    """
    per_in = np.asarray(importance).sum(axis=0)  # [in] importance per input edge
    totals = {d: 0.0 for d in _DRIVERS}
    for e in entries:
        driver = _BLOCK_TO_DRIVER.get(e["block"])
        if driver is None:
            continue
        totals[driver] += float(per_in[e["index"]])
    grand = sum(totals.values())
    if grand <= 0.0:
        return None
    return {d: totals[d] / grand for d in _DRIVERS}


def delayed_weight_share(model, activations: dict) -> dict:
    """Fraction of each encoder gate's drive carried by the delayed input.

    Combines :func:`edge_index_map` (which input edge reads ``x_t`` vs the
    recurrent ``h_prev``/``r*h_prev`` vs the tau-delayed ``x_delay``) with the
    data-aware edge ``importance`` from
    :func:`pri2m.xai.activations.collect_activations`. For every encoder gate it
    sums importance by input block, collapsing ``r*h_prev`` onto ``h_prev``, and
    normalizes the three drivers to sum 1. The ``overall`` share pools the raw
    (un-normalized) importance across all encoder gates before normalizing, so a
    high-importance gate counts more than a near-dead one.

    Parameters
    ----------
    model:
        A :class:`~pri2m.models.tkan_model.TKANModel` (or any model whose
        encoder gates :func:`edge_index_map` recognises).
    activations:
        The dict returned by
        :func:`~pri2m.xai.activations.collect_activations` -- each gate name maps
        to a dict with an ``"importance"`` ``[out, in]`` array.

    Returns
    -------
    dict
        ``{"overall": {"x_t": f, "h_prev": f, "x_delay": f},
        "per_gate": {gate_name: {"x_t", "h_prev", "x_delay"}}}`` with each
        triple of fractions summing to 1. Only gates that have an ``x_delay``
        block (the delay-aware encoder gates) are included.
    """
    emap = edge_index_map(model)
    per_gate: dict[str, dict] = {}
    overall_totals = {d: 0.0 for d in _DRIVERS}

    for name, entries in emap.items():
        if not name.startswith("encoder.layers."):
            continue
        if not any(e["block"] == "x_delay" for e in entries):
            continue  # only delay-aware encoder gates carry a delayed driver
        act = activations.get(name)
        if act is None or "importance" not in act:
            continue
        importance = np.asarray(act["importance"])

        # Accumulate raw per-driver importance into the overall pool.
        per_in = importance.sum(axis=0)  # [in]
        for e in entries:
            driver = _BLOCK_TO_DRIVER.get(e["block"])
            if driver is None:
                continue
            overall_totals[driver] += float(per_in[e["index"]])

        share = _gate_block_shares(entries, importance)
        if share is not None:
            per_gate[name] = share

    grand = sum(overall_totals.values())
    if grand <= 0.0:
        overall = {d: float("nan") for d in _DRIVERS}
    else:
        overall = {d: overall_totals[d] / grand for d in _DRIVERS}
    return {"overall": overall, "per_gate": per_gate}


# --------------------------------------------------------------------------- #
# 4. Attribution spectrum
# --------------------------------------------------------------------------- #
def attribution_spectrum(
    model,
    window,
    tau,
    fs: float,
    tpf: float,
    target: str = "chatter_logit",
    steps: int = 24,
    extra_scalars: dict[str, float] | None = None,
) -> dict:
    """Power spectrum of the time-resolved Integrated-Gradients attribution.

    Computes ``a[t] = sum_c |IG[t, c]|`` (the per-timestep attribution
    magnitude), removes its mean, and takes the real-FFT power spectrum. The
    tooth-pass frequency ``tpf`` is the fundamental regenerative tone; the
    fraction of spectral energy in a band around it (here the single nearest FFT
    bin) measures how much the model's attention rides the tooth-pass rhythm.

    Parameters
    ----------
    model:
        A model whose head ``target`` Integrated Gradients can attribute.
    window:
        Input window ``[L, C]`` or ``[1, L, C]``.
    tau:
        Kinematic delay (samples) passed to the attribution.
    fs:
        Sample rate (Hz) -- sets the frequency axis.
    tpf:
        Tooth-pass frequency (Hz); the band centre.
    target:
        Head to attribute (``"chatter_logit"`` or ``"b_over_blim"``).
    steps:
        Integrated-Gradients Riemann steps.
    extra_scalars:
        Optional side-input scalars forwarded to
        :func:`~pri2m.xai.attribution.integrated_gradients`.

    Returns
    -------
    dict
        ``{"freqs": np[F], "power": np[F], "tpf": float, "tpf_band_energy":
        float, "total_energy": float}`` where ``freqs = rfftfreq(L, 1/fs)``,
        ``power = |rfft(a - a.mean())|**2``, ``total_energy = power.sum()``, and
        ``tpf_band_energy`` is the power summed over the bin(s) nearest ``tpf``.
    """
    attr = integrated_gradients(
        model, window, target=target, tau=tau, steps=steps, extra_scalars=extra_scalars
    )  # [L, C]
    a = np.abs(attr).sum(axis=1)  # [L]
    L = a.shape[0]

    spectrum = np.fft.rfft(a - a.mean())
    power = np.abs(spectrum) ** 2  # [F]
    freqs = np.fft.rfftfreq(L, 1.0 / fs)  # [F]

    total_energy = float(power.sum())
    # Band = the single FFT bin nearest the tooth-pass frequency (rfftfreq is
    # uniform, so "nearest bin" is a well-defined +-half-bin band).
    band_bin = int(np.argmin(np.abs(freqs - tpf)))
    tpf_band_energy = float(power[band_bin])

    return {
        "freqs": freqs,
        "power": power,
        "tpf": float(tpf),
        "tpf_band_energy": tpf_band_energy,
        "total_energy": total_energy,
    }


# --------------------------------------------------------------------------- #
# 4b. Attribution chatter-band spectrum (fn + tooth-pass sidebands, over windows)
# --------------------------------------------------------------------------- #
def _band_centers(fn: float, tpf: float, n_sidebands: int, nyquist: float) -> list[float]:
    """Chatter-band centres: the natural frequency ``fn`` and its tooth-pass sidebands.

    Regenerative chatter concentrates near a structural natural frequency ``fn``,
    flanked by sidebands spaced one tooth-pass ``tpf`` apart (amplitude modulation
    of the chatter tone by the tooth-passing rhythm). Returns ``[fn, fn+-tpf,
    fn+-2 tpf, ...]`` up to ``n_sidebands`` on each side, dropping any centre that
    is non-positive or above ``nyquist``.
    """
    centers = [float(fn)]
    for k in range(1, int(n_sidebands) + 1):
        centers.append(fn + k * tpf)
        centers.append(fn - k * tpf)
    return sorted({c for c in centers if 0.0 < c <= nyquist})


def attribution_chatter_band(
    model,
    windows,
    tau,
    fs: float,
    fn: float,
    tpf: float,
    *,
    target: str = "chatter_logit",
    steps: int = 24,
    extra_scalars: dict[str, float] | None = None,
    bandwidth: float | None = None,
    n_sidebands: int = 2,
) -> dict:
    """Fraction of attribution energy in the **chatter band + tooth-pass sidebands**.

    The single-bin-at-``tpf`` measure of :func:`attribution_spectrum` is
    inconclusive: the regenerative signature does not sit at the tooth-pass
    *fundamental* but near the structural natural frequency ``fn``, flanked by
    sidebands at ``fn +- k*tpf``. This probe scores, per window, how much of the
    Integrated-Gradients attribution power falls in a band around ``fn`` and those
    sidebands, then **aggregates over a set of windows** (the per-window measure is
    noisy; the distribution is the evidence).

    For each window ``a[t] = sum_c |IG[t, c]|`` is de-meaned and power-spectrum'd;
    the chatter fraction is ``sum(power in bands) / sum(power)`` where the bands are
    ``|f - centre| <= bandwidth`` for every centre in ``[fn, fn+-tpf, ...,
    fn+-n_sidebands*tpf]``.

    Parameters
    ----------
    model:
        A model whose head ``target`` Integrated Gradients can attribute.
    windows:
        One window ``[L, C]`` / ``[1, L, C]``, a stack ``[N, L, C]``, or a sequence
        of ``[L, C]`` windows (all the same length ``L``).
    tau:
        Kinematic delay(s) for the attribution -- a scalar (shared) or a length-``N``
        sequence (one per window).
    fs:
        Sample rate (Hz); sets the frequency axis and Nyquist.
    fn:
        Structural natural / chatter frequency (Hz) -- the central band.
    tpf:
        Tooth-pass frequency (Hz) -- the sideband spacing.
    target, steps, extra_scalars:
        Forwarded to :func:`~pri2m.xai.attribution.integrated_gradients`.
    bandwidth:
        Half-width (Hz) of each band. ``None`` (default) -> ``1.5 * fs / L`` (the
        nearest FFT bin plus a neighbour); clamped to at least one bin.
    n_sidebands:
        Number of tooth-pass sidebands per side of ``fn`` (default ``2``).

    Returns
    -------
    dict
        ``{"per_window_fraction": np[N], "chatter_fraction": float,
        "chatter_fraction_median": float, "band_centers": [float, ...],
        "bandwidth": float, "freqs": np[F], "fn": float, "tpf": float,
        "n_windows": int}``. ``chatter_fraction`` is the mean over windows;
        ``per_window_fraction`` is the full distribution.
    """
    # Normalise `windows` -> list of [L, C] arrays.
    if isinstance(windows, (list, tuple)):
        win_list = [np.asarray(w, dtype=float) for w in windows]
    else:
        arr = np.asarray(windows, dtype=float)
        if arr.ndim == 2:
            win_list = [arr]
        elif arr.ndim == 3:
            win_list = [arr[i] for i in range(arr.shape[0])]
        else:
            raise ValueError(f"windows must be [L,C], [N,L,C] or a list; got {arr.ndim}D")
    win_list = [w[0] if w.ndim == 3 else w for w in win_list]
    N = len(win_list)
    if N == 0:
        raise ValueError("windows is empty")

    taus = list(tau) if isinstance(tau, (list, tuple, np.ndarray)) else [tau] * N
    if len(taus) != N:
        raise ValueError(f"tau length {len(taus)} != number of windows {N}")

    nyquist = 0.5 * fs
    per_window: list[float] = []
    centers_ref: list[float] = []
    bw_ref = 0.0
    freqs_ref = None
    for w, tau_w in zip(win_list, taus):
        L = w.shape[0]
        attr = integrated_gradients(
            model, w, target=target, tau=tau_w, steps=steps, extra_scalars=extra_scalars
        )  # [L, C]
        a = np.abs(attr).sum(axis=1)  # [L]
        power = np.abs(np.fft.rfft(a - a.mean())) ** 2  # [F]
        freqs = np.fft.rfftfreq(L, 1.0 / fs)  # [F]

        df = fs / L
        bw = float(bandwidth) if bandwidth is not None else 1.5 * df
        bw = max(bw, df)  # never narrower than one bin
        centers = _band_centers(fn, tpf, n_sidebands, nyquist)

        band_mask = np.zeros_like(freqs, dtype=bool)
        for c in centers:
            band_mask |= np.abs(freqs - c) <= bw
        total = float(power.sum())
        frac = float(power[band_mask].sum() / total) if total > 0.0 else 0.0
        per_window.append(frac)

        centers_ref, bw_ref, freqs_ref = centers, bw, freqs

    per_arr = np.asarray(per_window, dtype=float)
    return {
        "per_window_fraction": per_arr,
        "chatter_fraction": float(np.mean(per_arr)),
        "chatter_fraction_median": float(np.median(per_arr)),
        "band_centers": centers_ref,
        "bandwidth": float(bw_ref),
        "freqs": freqs_ref,
        "fn": float(fn),
        "tpf": float(tpf),
        "n_windows": int(N),
    }


# --------------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------------- #
def plot_tau_sweep(result: dict, path: str, true_tau=None) -> str:
    """Plot a delay-sweep result (AUC vs imposed delay) to ``path``.

    Handles both schemas:

    * :func:`tau_sweep_relative` -- ``{"factors", "auc", "baseline_auc", ...}``:
      AUC vs the imposed delay as a multiple of each window's own tau, with a
      vertical marker at ``x1`` (own tau) and a horizontal no-delay baseline.
    * :func:`tau_sweep` -- ``{"taus": [...], "auc": [...]}``: AUC vs the imposed
      delay in samples, optionally with a ``true_tau`` reference line.

    Parameters
    ----------
    result:
        A dict from either :func:`tau_sweep_relative` or :func:`tau_sweep`.
    path:
        Output image path.
    true_tau:
        (Absolute-tau schema only) draws a vertical reference line at the
        physical delay; ignored for the relative schema.

    Returns
    -------
    str
        ``path`` (the figure written).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    if "factors" in result:
        factors = [float(f) for f in result["factors"]]
        aucs = list(result["auc"])
        ax.plot(factors, aucs, marker="o")
        ax.axvline(1.0, color="r", linestyle="--", label="own tau (x1)")
        baseline = result.get("baseline_auc")
        if baseline is not None:
            ax.axhline(float(baseline), color="0.5", linestyle=":", label="no-delay baseline")
        ax.set_xlabel("imposed delay (x window's own tau)")
        ax.set_ylabel("chatter AUC")
        ax.set_title("Per-window relative delay sweep")
        ax.legend()
    else:
        taus = list(result["taus"])
        aucs = list(result["auc"])
        ax.plot(taus, aucs, marker="o")
        if true_tau is not None:
            ax.axvline(float(true_tau), color="r", linestyle="--", label=f"true tau={true_tau}")
            ax.legend()
        ax.set_xlabel("imposed delay tau (samples)")
        ax.set_ylabel("chatter AUC")
        ax.set_title("Delay sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_attr_spectrum(result: dict, path: str) -> str:
    """Plot an :func:`attribution_spectrum` result (power vs frequency) to ``path``.

    Parameters
    ----------
    result:
        A dict from :func:`attribution_spectrum` (``freqs``, ``power``, ``tpf``).
    path:
        Output image path.

    Returns
    -------
    str
        ``path`` (the figure written).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = np.asarray(result["freqs"])
    power = np.asarray(result["power"])
    tpf = result.get("tpf")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(freqs, power)
    if tpf is not None:
        ax.axvline(float(tpf), color="r", linestyle="--", label=f"tpf={tpf:.0f} Hz")
        ax.legend()
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("attribution power")
    ax.set_title("Attribution spectrum")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
