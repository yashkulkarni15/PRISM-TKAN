"""Test-time delay interventions on a trained TKAN, matched to the MiniROCKET arm.

Why this exists
---------------
Exp 3 asks whether a plain time-series classifier handed the delayed channel
(MiniROCKET arm B) leans on it the way TKAN does. That comparison is only
meaningful if both models are hit with the *same* three interventions on the
*same* evaluation set (held-out 121-125):

``zero``
    the delayed signal is removed entirely.
``scale`` with ``s`` in {0.5, 2}
    the delay is misstated: the network is told ``round(s * tau)`` instead of
    ``tau``.

Both act on the delay the ENCODER sees and nothing else. The Gao-style side
branch also receives ``tau`` (as a scalar process parameter, in raw samples);
it is left untouched so that a drop measures the recurrent delay path, not a
perturbed side feature. Any reliance on ``tau`` through the side branch is a
different question and is not mixed in here.

The tau arithmetic (:func:`rescale_raw_tau`, :func:`encoder_tau`) is shared
with :mod:`pri2m.models.minirocket_baseline`, so "x0.5" means exactly the same
integer delay for both models, by construction rather than by agreement.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from pri2m.models.tkan_model import avg_pool_time, scale_tau

__all__ = [
    "VARIANT_MODES",
    "rescale_raw_tau",
    "encoder_tau",
    "DelayVariant",
    "delay_drop",
    "exceeds_floor",
]

VARIANT_MODES = ("scale", "zero")


def rescale_raw_tau(tau, scale: float) -> torch.Tensor:
    """Misstate a delay in RAW samples: ``max(1, round(tau * scale))``.

    The scale is applied before pooling, on the integer the data pipeline
    actually produces (``tau_samples = round(fs / tpf)``), because that is the
    quantity being misstated -- "the tooth period is twice what it is". Scaling
    the already-pooled integer instead would compound two roundings and make
    x0.5 and x2 not exact inverses at small tau.

    ``torch.round`` is round-half-to-even, the same rule as Python's ``round``
    that :func:`scale_tau` uses for an int, so the two never disagree on a tie.
    The clamp to 1 keeps the invariant every delay in this project has
    (``tau_samples`` and ``scale_tau`` both clamp): a delay never collapses
    onto the current step, which would silently turn the delayed channel into a
    copy of the input rather than a misstated delay.
    """
    t = torch.as_tensor(tau)
    if t.ndim == 0:
        t = t.reshape(1)
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError(f"scale must be a finite positive number, got {scale}")
    if scale == 1.0:
        return t.to(torch.long)
    return torch.clamp(torch.round(t.to(torch.float64) * float(scale)).to(torch.long), min=1)


def encoder_tau(tau, pool: int, pooled_length: int, mode: str, scale: float = 1.0) -> torch.Tensor:
    """The delay (in pooled steps) the encoder receives under a variant.

    ``mode="scale"``: ``scale_tau(rescale_raw_tau(tau, scale), pool)`` -- with
    ``scale=1`` this is exactly what :meth:`TKANModel.forward` computes.

    ``mode="zero"``: ``pooled_length + 1`` for every sample. ``delayed_input``
    returns zeros whenever ``t - tau < 0``; with ``tau > L - 1`` that holds at
    every step, so every delayed slice is zero -- the same input the encoder
    would see if the delayed channel were deleted, without touching weights.
    """
    t = torch.as_tensor(tau)
    if t.ndim == 0:
        t = t.reshape(1)
    if mode == "zero":
        return torch.full_like(t.to(torch.long), int(pooled_length) + 1)
    if mode != "scale":
        raise ValueError(f"mode must be one of {VARIANT_MODES}, got {mode!r}")
    raw = rescale_raw_tau(t, scale)
    return scale_tau(raw, pool) if pool > 1 else raw


class DelayVariant(nn.Module):
    """A trained :class:`TKANModel` whose encoder is fed a modified delay.

    ``forward`` replicates :meth:`TKANModel.forward` line for line, changing
    only the ``tau`` handed to ``self.model.encoder``. The wrapped model is a
    registered submodule, so ``.eval()``, ``.to()`` and ``.parameters()`` reach
    it -- ``Phase3Trainer.evaluate`` reads the device off the first parameter
    and ``eval_heldout_phase3.run`` reads ``side_input``; both are exposed.

    The trajectory head (if present) keeps the TRUE pooled delay: the
    intervention is defined on the encoder's delayed input, and letting it leak
    into a second consumer would make the measured drop a mixture.
    """

    def __init__(self, model: nn.Module, mode: str = "scale", scale: float = 1.0) -> None:
        super().__init__()
        if mode not in VARIANT_MODES:
            raise ValueError(f"mode must be one of {VARIANT_MODES}, got {mode!r}")
        if mode == "scale" and (scale <= 0 or not math.isfinite(scale)):
            raise ValueError(f"scale must be a finite positive number, got {scale}")
        self.model = model
        self.mode = mode
        self.scale = float(scale)
        self.side_input = bool(getattr(model, "side_input", False))
        self.pool = int(getattr(model, "pool", 1))
        self.with_trajectory = bool(getattr(model, "with_trajectory", False))

    def describe(self) -> dict:
        """JSON-able description; ``scale`` is None for ``zero`` (not 0.0)."""
        return {"mode": self.mode, "scale": self.scale if self.mode == "scale" else None}

    def forward(self, batch: dict) -> dict:
        m = self.model
        window = batch["window"]  # [B, L, C]
        tau = batch.get("tau", 1)
        if not isinstance(tau, torch.Tensor):
            # TKANModel accepts a scalar int tau; broadcast it so the variant
            # arithmetic (which is per-sample) has a [B] tensor to work on.
            tau = torch.full((window.shape[0],), int(tau), dtype=torch.long)
        tau = tau.to(window.device)
        if self.pool > 1:
            window = avg_pool_time(window, self.pool)
        true_tau = scale_tau(tau, self.pool) if self.pool > 1 else tau
        if self.mode == "scale" and self.scale == 1.0:
            enc_tau = true_tau  # the exact tensor TKANModel.forward would pass
        else:
            enc_tau = encoder_tau(tau, self.pool, window.shape[1], self.mode, self.scale)
        enc_tau = enc_tau.to(window.device)
        context, _ = m.encoder(window, enc_tau)
        sev_context = context
        if self.side_input:
            # Side features from the RAW batch (tau unmodified, unpooled).
            sev_context = torch.cat([context, m._side_context(batch)], dim=-1)
        out = m.severity(sev_context)
        if self.with_trajectory:
            seed = window[:, -1, list(m.traj_channels)]
            out["trajectory"] = m.trajectory(context, seed, m.traj_steps, true_tau)
        return out


def delay_drop(baseline_auc, variant_auc):
    """``baseline - variant``; None when either side was unmeasurable.

    A NaN AUC (single-class set, failed eval) is unmeasurable, not a number --
    reporting it as a drop of NaN or 0 would read as "no reliance".
    """
    vals = []
    for v in (baseline_auc, variant_auc):
        if v is None:
            return None
        v = float(v)
        if not math.isfinite(v):
            return None
        vals.append(v)
    return vals[0] - vals[1]


def exceeds_floor(drop, floor: float):
    """True iff ``drop > floor``; None when the drop itself is unmeasurable.

    One-sided on purpose: the pre-committed read is "a drop counts only above
    the floor". A variant that IMPROVES AUC is reported as a negative drop and
    is never counted as reliance.
    """
    if drop is None:
        return None
    return bool(float(drop) > float(floor))
