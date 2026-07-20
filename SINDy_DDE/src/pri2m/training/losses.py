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


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Sample-weighted pinball (quantile) loss, NaN-safe at weight 0.

    ``pred`` is ``[B, Q]`` (one column per quantile level in ``quantiles``),
    ``target`` is ``[B]``, ``weight`` is ``[B]``. For an error ``e = y - q_pred``
    the pinball loss at level ``q`` is ``max(q*e, (q-1)*e)`` (under-prediction of
    a high quantile is penalized more heavily). The per-sample loss is the mean
    over quantile columns; excluded rows (weight 0) are ignored in both the
    forward and backward pass even when their target is NaN/inf.
    """
    weight = weight.to(pred.dtype)
    q = torch.as_tensor(tuple(quantiles), dtype=pred.dtype, device=pred.device)
    safe_target = _mask_target(target, weight > 0)
    err = safe_target.unsqueeze(-1) - pred  # [B, Q], e = y - q_pred
    per_quantile = torch.maximum(q * err, (q - 1.0) * err)  # [B, Q]
    per_sample = per_quantile.mean(dim=-1)  # [B]
    return _weighted_reduce(per_sample, weight)


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
    """Weighted regression on ``b_over_blim`` + weighted BCE on the chatter label.

    Reads ``out["chatter_logit"]``, ``batch["b_over_blim"]``, ``batch["label"]``,
    ``batch["sample_weight"]``. The regression term is chosen by the head's mode:

    * default (point) head -- weighted MSE on ``out["b_over_blim"]`` (part
      ``"sev_mse"``); byte-for-byte the previous behavior.
    * quantile head (``out`` carries ``"b_over_blim_quantiles"`` +
      ``"quantile_levels"``) -- weighted pinball loss over the quantile columns
      (part ``"sev_pinball"``).

    Returns ``(lw.w_sev*(lw.w_reg*reg + lw.w_cls*bce), parts)``.
    """
    weight = batch["sample_weight"]
    bce = weighted_bce_with_logits(out["chatter_logit"], batch["label"], weight)
    if "b_over_blim_quantiles" in out and "quantile_levels" in out:
        reg = pinball_loss(
            out["b_over_blim_quantiles"],
            batch["b_over_blim"],
            out["quantile_levels"],
            weight,
        )
        reg_key = "sev_pinball"
    else:
        reg = weighted_mse(out["b_over_blim"], batch["b_over_blim"], weight)
        reg_key = "sev_mse"
    loss = lw.w_sev * (lw.w_reg * reg + lw.w_cls * bce)
    parts = {reg_key: float(reg.detach()), "sev_bce": float(bce.detach())}
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
