"""Concept probes (TCAV-style) on a trained TKAN model's hidden state.

The TKAN encoder squashes each window into a 64-d hidden state ``h`` (its final
context). These probes ask a *concept-activation-vector* (TCAV, Kim et al. 2018)
question, adapted to regression: **is a named physics quantity linearly decodable
from ``h``, and does the chatter decision move along that concept's direction?**

Three named concepts, each a per-window scalar computed from the batch (no model,
so the target is model-independent ground truth):

``regeneration_phase_sin`` / ``regeneration_phase_cos``
    The regenerative-term phase lag between the surface ``x(t)`` and its
    tooth-pass-delayed copy ``x(t - tau)``, probed as its ``sin``/``cos``
    components. For a window whose dominant channel-0 FFT bin is ``k`` (length
    ``L``) the phase advance across a delay of ``tau`` samples is ``2*pi*k*tau/L``
    -- the regenerative phase ``eps = omega_c * tau`` of DDE chatter theory
    (``omega_c = 2*pi*k*fs/L``, delay ``tau/fs`` s). The phase is **circular**,
    so a linear probe on the raw wrapped angle in ``(-pi, pi]`` is degenerate by
    construction (two near-identical windows can read ``-pi`` and ``+pi``,
    collapsing held-out ``R^2`` -- observed at ``-228``). Probing ``sin`` and
    ``cos`` of the unwrapped phase is circular-safe (both are ``2*pi``-periodic).
``chatter_tpf_separation``
    Spectral distance ``|f_dom - tpf|`` (Hz) between the window's dominant
    frequency (channel-0 FFT peak) and the metadata tooth-pass frequency ``tpf``.
    Large separation == the energy has walked off the forced tooth-pass line
    toward a chatter tone. Heavy-tailed in Hz, so the probe target is fit on a
    ``log1p`` scale (recorded as ``transform = "log1p"``).
``proximity_to_blim``
    The severity label ``b_over_blim = b_mm / blim`` straight from the batch
    metadata (how close the cut sits to the stability limit).

For each concept we fit a linear probe (ridge for regression, logistic for
classification) on a train/test split, report the held-out probe score
(``R^2`` / accuracy) and the unit concept **direction** in ``h``-space (the
probe weights mapped back to raw activation coordinates), and a TCAV directional
sensitivity: the dot of ``d chatter_logit / d h`` with the concept direction,
summarised as a mean sensitivity and the fraction of windows with positive
sensitivity.

sklearn / matplotlib are imported lazily so importing this module never
hard-fails in an environment that lacks them.
"""

from __future__ import annotations

import numpy as np
import torch

from pri2m.models.tkan_model import avg_pool_time, scale_tau

#: Canonical input-channel order; channel 0 (``xtpos``) is the surface position
#: whose regenerative self-delay drives chatter.
CHANNELS: tuple[str, ...] = ("xtpos", "ytpos", "Forcex", "Forcey")

#: Each concept's probe kind (all continuous -> ridge regression). The
#: regenerative phase is circular, so it is probed as its ``sin``/``cos``
#: components rather than the raw wrapped angle (a linear probe on a wrapped
#: angle in ``(-pi, pi]`` is degenerate: two near-identical windows can sit at
#: ``-pi`` and ``+pi``, collapsing held-out ``R^2`` -- observed at ``-228``).
CONCEPT_KINDS: dict[str, str] = {
    "regeneration_phase_sin": "regression",
    "regeneration_phase_cos": "regression",
    "chatter_tpf_separation": "regression",
    "proximity_to_blim": "regression",
}

#: Optional monotone target transform applied when *building* a concept target,
#: recorded in each concept's summary so downstream plots/manifests read on the
#: right scale. ``chatter_tpf_separation`` is heavy-tailed in Hz, so its target
#: is fit on a ``log1p`` scale. Concepts absent here carry ``transform = None``.
CONCEPT_TRANSFORMS: dict[str, str] = {
    "chatter_tpf_separation": "log1p",
}


# --------------------------------------------------------------------------- #
# Hidden-state extraction + chatter-logit gradient (TKANModel plumbing)
# --------------------------------------------------------------------------- #
def _pooled_window_tau(model, batch):
    """The (possibly pooled) encoder input window and rescaled delay ``tau``.

    Mirrors :meth:`TKANModel.forward`'s pre-encoder pooling so ``h`` is extracted
    from exactly the sequence the trained model consumes.
    """
    window = torch.as_tensor(batch["window"], dtype=torch.float32)
    tau = batch.get("tau", 1)
    pool = int(getattr(model, "pool", 1) or 1)
    if pool > 1:
        window = avg_pool_time(window, pool)
        tau = scale_tau(tau, pool)
    return window, tau


def extract_hidden_states(model, batch: dict) -> np.ndarray:
    """Final encoder hidden state ``h`` ``[B, H]`` for each window (no grad).

    Pool-aware: applies the model's temporal pooling and delay rescaling before
    the encoder, exactly as :meth:`TKANModel.forward` does, so the probes see the
    same ``h`` the severity head consumes.
    """
    was_training = model.training
    model.eval()
    window, tau = _pooled_window_tau(model, batch)
    with torch.no_grad():
        context, _ = model.encoder(window, tau)
    if was_training:
        model.train()
    return context.detach().cpu().numpy()


def chatter_logit_grad_wrt_hidden(model, batch: dict) -> np.ndarray:
    """``d chatter_logit / d h`` ``[B, H]`` for each window.

    Only the hidden state ``h`` (the encoder context) is differentiated; the
    optional side-input branch is held fixed, so the gradient is the sensitivity
    of the chatter decision to a move in *activation* space -- the quantity TCAV
    dots against the concept direction.
    """
    was_training = model.training
    model.eval()
    window, tau = _pooled_window_tau(model, batch)
    context, _ = model.encoder(window, tau)
    context = context.detach().clone().requires_grad_(True)

    sev_context = context
    if getattr(model, "side_input", False):
        sev_context = torch.cat([context, model._side_context(batch)], dim=-1)
    out = model.severity(sev_context)
    logit = out["chatter_logit"].reshape(-1).sum()
    (grad,) = torch.autograd.grad(logit, context)

    if was_training:
        model.train()
    return grad.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Concept-target builders (per-window physics scalars from the batch)
# --------------------------------------------------------------------------- #
def _dominant_bin(x: np.ndarray) -> int:
    """Index of the dominant rfft bin of 1-D ``x`` (DC removed)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x))
    if spec.size > 1:
        spec[0] = 0.0  # ignore the DC bin
    return int(np.argmax(spec))


def _wrap_pi(a):
    """Wrap an angle (rad) into ``[-pi, pi)``. Scalar-in -> float, array-in -> array."""
    wrapped = (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi
    return float(wrapped) if np.ndim(a) == 0 else wrapped


def dominant_frequency(x: np.ndarray, fs: float) -> float:
    """Dominant frequency (Hz) of 1-D signal ``x`` sampled at ``fs``."""
    x = np.asarray(x, dtype=float)
    return float(_dominant_bin(x) * float(fs) / len(x))


def regeneration_phase_unwrapped(x: np.ndarray, tau) -> float:
    """Unwrapped regenerative phase lag ``2*pi*k*tau/L`` (rad) -- NOT wrapped.

    At the dominant FFT bin ``k`` (length ``L``) a shift of ``tau`` samples
    advances the phase by ``2*pi*k*tau/L``; this is the regenerative phase
    ``eps = omega_c * tau`` of DDE chatter. Probes consume ``sin``/``cos`` of
    this value (circular-safe), so it is deliberately left un-wrapped -- ``sin``
    and ``cos`` are ``2*pi``-periodic, making the wrap irrelevant.
    """
    x = np.asarray(x, dtype=float)
    L = len(x)
    k = _dominant_bin(x)
    return 2.0 * np.pi * k * float(tau) / L


def regeneration_phase(x: np.ndarray, tau) -> float:
    """Regenerative phase lag (rad, wrapped to ``[-pi, pi)``) -- raw physics value.

    Wraps :func:`regeneration_phase_unwrapped`. Retained for the analytic phase
    definition; the *concept probes* use ``sin``/``cos`` of the unwrapped phase
    (see :data:`CONCEPT_KINDS`) because a linear probe on this wrapped angle is
    degenerate by construction.
    """
    return _wrap_pi(regeneration_phase_unwrapped(x, tau))


def chatter_tpf_separation(x: np.ndarray, tpf: float, fs: float) -> float:
    """Spectral distance ``|f_dom - tpf|`` (Hz) of ``x`` from the tooth-pass line."""
    return float(abs(dominant_frequency(x, fs) - float(tpf)))


def _np1d(batch, key) -> np.ndarray:
    return np.asarray(torch.as_tensor(batch[key]).detach().cpu().numpy(), dtype=float).reshape(-1)


def build_concept_targets(batch: dict, channel: int = 0) -> dict[str, np.ndarray]:
    """Per-window concept targets from a batch dict.

    Uses ``channel`` (default 0 == ``xtpos``, the regenerative surface position)
    of the raw window plus the batch metadata (``tau``/``tpf``/``fs``/
    ``b_over_blim``). Returns ``{concept_name: np.ndarray[B]}``.
    """
    window = np.asarray(torch.as_tensor(batch["window"]).detach().cpu().numpy(), dtype=float)
    tau = _np1d(batch, "tau")
    tpf = _np1d(batch, "tpf")
    fs = _np1d(batch, "fs")
    b_over_blim = _np1d(batch, "b_over_blim")
    B = window.shape[0]

    # Circular-safe regenerative phase: probe sin/cos of the UNWRAPPED phase, not
    # the wrapped angle (a linear Ridge on a wrapped circular target is degenerate).
    phi = np.array(
        [regeneration_phase_unwrapped(window[i, :, channel], int(round(tau[i]))) for i in range(B)]
    )
    # Heavy-tailed spectral separation (Hz) fit on a log1p scale (see CONCEPT_TRANSFORMS).
    sep = np.array(
        [chatter_tpf_separation(window[i, :, channel], float(tpf[i]), float(fs[i])) for i in range(B)]
    )
    return {
        "regeneration_phase_sin": np.sin(phi),
        "regeneration_phase_cos": np.cos(phi),
        "chatter_tpf_separation": np.log1p(sep),
        "proximity_to_blim": b_over_blim.astype(float),
    }


# --------------------------------------------------------------------------- #
# Linear probe + TCAV score
# --------------------------------------------------------------------------- #
def fit_probe(
    h: np.ndarray,
    target: np.ndarray,
    kind: str = "regression",
    *,
    test_size: float = 0.3,
    seed: int = 0,
    alpha: float = 1.0,
) -> dict:
    """Fit a linear probe ``h -> target`` with a train/test split.

    Parameters
    ----------
    h:
        Hidden states ``[N, H]``.
    target:
        Per-sample concept value ``[N]`` (continuous for ``kind="regression"``,
        integer labels for ``kind="classification"``).
    kind:
        ``"regression"`` (ridge, score = held-out ``R^2``) or
        ``"classification"`` (logistic, score = held-out accuracy).
    test_size, seed:
        Train/test split fraction and RNG seed.
    alpha:
        Ridge regularization strength (ignored for classification).

    Returns
    -------
    dict
        ``kind``, held-out ``score`` and ``train_score``, the unit ``direction``
        (probe weights mapped to raw ``h``-space, ``[H]``), the un-normalized
        ``weights`` ``[H]``, ``intercept``, ``chance_score``, and the
        ``n_used``/``n_train``/``n_test`` counts. The direction lives in raw
        activation coordinates so it dots directly against
        :func:`chatter_logit_grad_wrt_hidden`.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    h = np.asarray(h, dtype=float)
    target = np.asarray(target).reshape(-1)
    mask = np.all(np.isfinite(h), axis=1) & np.isfinite(target.astype(float))
    h, target = h[mask], target[mask]
    n_used = int(h.shape[0])
    if n_used < 4:
        raise ValueError(f"need >=4 finite samples to fit a probe, got {n_used}")

    is_cls = kind == "classification"
    strat = target if is_cls else None
    h_tr, h_te, y_tr, y_te = train_test_split(
        h, target, test_size=test_size, random_state=seed, stratify=strat
    )

    scaler = StandardScaler().fit(h_tr)
    z_tr, z_te = scaler.transform(h_tr), scaler.transform(h_te)

    if is_cls:
        y_tr_i, y_te_i = y_tr.astype(int), y_te.astype(int)
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(z_tr, y_tr_i)
        score = float(accuracy_score(y_te_i, clf.predict(z_te)))
        train_score = float(accuracy_score(y_tr_i, clf.predict(z_tr)))
        coef = np.asarray(clf.coef_).reshape(-1)
        intercept = float(np.asarray(clf.intercept_).reshape(-1)[0])
        # Majority-class accuracy on the test split = the chance baseline.
        vals, counts = np.unique(y_te_i, return_counts=True)
        chance = float(counts.max() / counts.sum())
    else:
        reg = Ridge(alpha=alpha, random_state=seed)
        reg.fit(z_tr, y_tr)
        score = float(r2_score(y_te, reg.predict(z_te)))
        train_score = float(r2_score(y_tr, reg.predict(z_tr)))
        coef = np.asarray(reg.coef_).reshape(-1)
        intercept = float(reg.intercept_)
        chance = 0.0  # R^2 of the mean predictor

    # Map standardized coefficients back to raw h-space: target ~ coef . ((h-mu)/sigma)
    # so d target / d h = coef / sigma. This is the concept direction TCAV uses.
    weights = coef / scaler.scale_
    norm = float(np.linalg.norm(weights))
    direction = weights / norm if norm > 0 else weights
    return {
        "kind": kind,
        "score": score,
        "train_score": train_score,
        "chance_score": chance,
        "direction": direction,
        "weights": weights,
        "intercept": intercept,
        "n_used": n_used,
        "n_train": int(h_tr.shape[0]),
        "n_test": int(h_te.shape[0]),
    }


def tcav_score(grad_h: np.ndarray, direction: np.ndarray) -> dict:
    """TCAV directional sensitivity of the chatter decision along a concept.

    Parameters
    ----------
    grad_h:
        ``d chatter_logit / d h`` ``[N, H]`` (see
        :func:`chatter_logit_grad_wrt_hidden`).
    direction:
        Concept direction ``[H]`` (re-normalized to a unit vector here).

    Returns
    -------
    dict
        ``mean_sensitivity`` (mean dot), ``sensitivity_std``, ``tcav_fraction``
        (fraction of windows whose sensitivity is positive -- the classic TCAV
        sign-count score), and the ``per_window`` dot array.
    """
    grad_h = np.asarray(grad_h, dtype=float)
    d = np.asarray(direction, dtype=float).reshape(-1)
    n = np.linalg.norm(d)
    if n > 0:
        d = d / n
    dots = grad_h @ d  # [N]
    return {
        "mean_sensitivity": float(np.mean(dots)),
        "sensitivity_std": float(np.std(dots)),
        "tcav_fraction": float(np.mean(dots > 0)),
        "per_window": dots,
    }


# --------------------------------------------------------------------------- #
# Orchestrator + figure
# --------------------------------------------------------------------------- #
def run_concept_probes(
    model,
    batch: dict,
    *,
    test_size: float = 0.3,
    seed: int = 0,
    alpha: float = 1.0,
    channel: int = 0,
) -> dict:
    """Fit every concept probe on ``model``'s hidden state over ``batch``.

    Returns ``{"concepts": {name: {"kind", "probe", "tcav"}}, "hidden_size",
    "n_windows"}``. ``probe`` is the :func:`fit_probe` dict and ``tcav`` the
    :func:`tcav_score` dict for that concept's direction.
    """
    h = extract_hidden_states(model, batch)
    grad_h = chatter_logit_grad_wrt_hidden(model, batch)
    targets = build_concept_targets(batch, channel=channel)

    concepts: dict[str, dict] = {}
    for name, tgt in targets.items():
        kind = CONCEPT_KINDS[name]
        probe = fit_probe(h, tgt, kind=kind, test_size=test_size, seed=seed, alpha=alpha)
        tcav = tcav_score(grad_h, probe["direction"])
        # Record the target transform (None if the concept was fit on its raw scale).
        concepts[name] = {
            "kind": kind,
            "probe": probe,
            "tcav": tcav,
            "transform": CONCEPT_TRANSFORMS.get(name),
        }
    return {
        "concepts": concepts,
        "hidden_size": int(h.shape[1]),
        "n_windows": int(h.shape[0]),
    }


def plot_probe_scores(results: dict, path: str) -> str:
    """Bar chart of the held-out probe score per concept (annotated with TCAV).

    Returns ``path``. The bar height is the probe ``score`` (``R^2`` / accuracy);
    each bar is annotated with the concept's TCAV mean sensitivity.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    concepts = results["concepts"]
    names = list(concepts.keys())
    scores = [float(concepts[n]["probe"]["score"]) for n in names]
    kinds = [concepts[n]["kind"] for n in names]
    tcav = [float(concepts[n]["tcav"]["mean_sensitivity"]) for n in names]

    fig, ax = plt.subplots(figsize=(1.8 + 1.7 * len(names), 4.2))
    colors = ["#2a7fb8" if k == "regression" else "#d1662a" for k in kinds]
    bars = ax.bar(range(len(names)), scores, color=colors)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("probe score  (R² / accuracy)")
    ax.set_title("Concept decodability from the TKAN hidden state")
    lo = min(0.0, min(scores) if scores else 0.0)
    ax.set_ylim(lo - 0.05, 1.05)
    for i, (b, s, t) in enumerate(zip(bars, scores, tcav)):
        ax.text(
            b.get_x() + b.get_width() / 2, s + 0.02,
            f"{s:.2f}\nTCAV {t:+.2g}", ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
