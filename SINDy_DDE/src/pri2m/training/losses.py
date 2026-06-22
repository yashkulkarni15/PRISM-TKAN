"""Phase 3 noise-weighted losses (Task 1C).

All losses are sample-weighted and NaN-safe: rows whose weight is zero are
ignored entirely, even if their target is non-finite (NaN/inf). This matters
because ``exclude_supervised`` rows carry weight 0 and may have undefined
severity targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_EPS = 1e-8


def _mask_target(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace target entries on excluded (weight==0) rows with 0.

    This is the crux of NaN-safety. A NaN/inf target on a weight-0 row must
    never reach the error op, because autograd differentiates through it: even
    if the result is masked out afterwards, ``mse(pred, nan)`` leaks ``nan``
    into ``pred.grad`` (verified empirically). Substituting a finite constant
    *before* the error op keeps both the forward value and the backward
    gradient finite, while the row's zero weight removes it from the average.
    """
    return torch.where(mask, target, torch.zeros_like(target))


def _weighted_reduce(err: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean of a per-element error tensor.

    Returns ``sum(weight * err) / sum(weight).clamp_min(eps)``. Callers pass an
    ``err`` already computed from a NaN-sanitized target, so excluded rows
    contribute exactly ``weight*err == 0`` with a clean (zero) gradient.
    """
    weight = weight.to(err.dtype)
    numer = (weight * err).sum()
    denom = weight.sum().clamp_min(_EPS)
    return numer / denom


def weighted_mse(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Sample-weighted MSE: ``sum(w*(p-t)^2)/sum(w)``, NaN-safe at weight 0.

    With all weights equal to 1 this equals ``F.mse_loss(pred, target)``.
    Rows with weight 0 are ignored even when their target is NaN/inf, in both
    the forward and backward pass.
    """
    weight = weight.to(pred.dtype)
    safe_target = _mask_target(target, weight > 0)
    err = F.mse_loss(pred, safe_target, reduction="none")
    return _weighted_reduce(err, weight)


def weighted_bce_with_logits(
    logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Sample-weighted BCE-with-logits, NaN-safe at weight 0.

    With uniform weight this equals ``F.binary_cross_entropy_with_logits``.
    Rows with weight 0 are ignored even when their target is NaN/inf.
    """
    weight = weight.to(logit.dtype)
    safe_target = _mask_target(target, weight > 0)
    err = F.binary_cross_entropy_with_logits(logit, safe_target, reduction="none")
    return _weighted_reduce(err, weight)


@dataclass
class LossWeights:
    """Relative weights for the multi-task severity + trajectory objective."""

    w_sev: float = 1.0
    w_reg: float = 1.0
    w_cls: float = 1.0
    w_traj: float = 0.0


def severity_loss(
    out: dict, batch: dict, lw: LossWeights
) -> tuple[torch.Tensor, dict]:
    """Weighted MSE on ``b_over_blim`` + weighted BCE on the chatter label.

    Reads ``out["b_over_blim"]``, ``out["chatter_logit"]`` and
    ``batch["b_over_blim"]``, ``batch["label"]``, ``batch["sample_weight"]``.
    Returns ``(lw.w_sev*(lw.w_reg*mse + lw.w_cls*bce), {"sev_mse":..., "sev_bce":...})``.
    """
    weight = batch["sample_weight"]
    mse = weighted_mse(out["b_over_blim"], batch["b_over_blim"], weight)
    bce = weighted_bce_with_logits(out["chatter_logit"], batch["label"], weight)
    loss = lw.w_sev * (lw.w_reg * mse + lw.w_cls * bce)
    parts = {"sev_mse": float(mse.detach()), "sev_bce": float(bce.detach())}
    return loss, parts


def trajectory_loss(
    traj_pred: torch.Tensor, traj_true: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Sample-weighted MSE over an autoregressive trajectory rollout.

    ``traj_pred``/``traj_true`` are ``[B, n_steps, Cout]``; ``weight`` is the
    per-sample weight ``[B]`` (broadcast over steps/channels). NaN-safe at
    weight 0 in both the forward and backward pass.
    """
    weight = weight.to(traj_pred.dtype)
    # Sanitize the target on excluded samples before the error op (see
    # _mask_target): a NaN trajectory target on a weight-0 sample would
    # otherwise leak into gradients despite being averaged away.
    sample_mask = (weight > 0).reshape((-1,) + (1,) * (traj_true.dim() - 1))
    safe_true = torch.where(sample_mask, traj_true, torch.zeros_like(traj_true))
    err = F.mse_loss(traj_pred, safe_true, reduction="none")
    # Reduce non-batch dims to a per-sample mean error, then weight by sample.
    per_sample = err.reshape(err.shape[0], -1).mean(dim=1)
    return _weighted_reduce(per_sample, weight)


def total_loss(
    out: dict, batch: dict, lw: LossWeights
) -> tuple[torch.Tensor, dict]:
    """Severity loss plus an optional trajectory term.

    Adds ``lw.w_traj * trajectory_loss(...)`` only when BOTH ``out["trajectory"]``
    and ``batch["trajectory_target"]`` exist. With ``w_traj == 0`` (or missing
    trajectory keys) this equals :func:`severity_loss`.
    """
    loss, parts = severity_loss(out, batch, lw)
    if "trajectory" in out and "trajectory_target" in batch:
        traj = trajectory_loss(
            out["trajectory"], batch["trajectory_target"], batch["sample_weight"]
        )
        loss = loss + lw.w_traj * traj
        parts = {**parts, "traj_mse": float(traj.detach())}
    return loss, parts
