"""Phase 3 prediction heads + the model-agnostic ``Phase3Model`` protocol.

Three heads sit on top of a shared trunk context ``[B, H]``:

* :class:`SeverityHead` -- the trainable severity head. Maps the context to two
  scalars per sample: a real-valued ``b_over_blim`` regression output and a raw
  binary chatter ``chatter_logit``.
* :class:`TrajectoryHead` -- an autoregressive decoder that rolls out the
  downstream channels (e.g. ``xtpos``, ``Forcex``). Built now but gated by
  ``stage_gates.evaluate_trajectory_gate`` at the harness level.
* :class:`PhysicsResidualHead` -- a documented stub. It is blocked on the
  Altintas-Budak cutting-force constants / engagement convention and raises
  loudly if anyone tries to use it.

:class:`Phase3Model` is the runtime-checkable structural protocol every concrete
Phase 3 model (``TKANModel``, :class:`~pri2m.models.baseline.MLPBaseline`)
satisfies: a single ``forward(batch: dict) -> dict``.

KAN note: ``KANLinear``'s B-spline path is only active for inputs in
``grid_range == (-1, 1)``; outside that range only the SiLU base path acts. So
every tensor fed to a ``KANLinear`` here is first squashed with ``torch.tanh``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from pri2m.models.kan import KANLinear


# The point head is named for what it emits, matching its sibling "quantile".
# It was called "huber" until 2026-08-11, which named a loss the code has never
# contained: the point path calls weighted_mse and reports the part as
# "sev_mse", and there is no Huber/smooth-L1 anywhere in pri2m.training.losses.
DEFAULT_SEVERITY_MODE = "point"
_VALID_SEVERITY_MODES = ("point", "quantile")
# Deprecated spellings, still accepted: every config, sweep and summary.json
# written before the rename says "huber", and rejecting them would strand the
# validated runs for a cosmetic gain.
_SEVERITY_MODE_ALIASES = {"huber": "point"}


def normalise_severity_mode(mode: str) -> str:
    """Canonical severity mode, mapping deprecated spellings forward.

    Normalising at the boundary (rather than storing whatever the caller passed)
    means exactly one spelling reaches the model, the loss router, the config
    echo in ``summary.json`` and the ``preprocessing.json`` gate -- so none of
    them has to know the alias exists.
    """
    canonical = _SEVERITY_MODE_ALIASES.get(str(mode), str(mode))
    if canonical not in _VALID_SEVERITY_MODES:
        raise ValueError(
            f"unknown severity_mode {mode!r}; expected one of "
            f"{_VALID_SEVERITY_MODES} (or the deprecated alias "
            f"{tuple(_SEVERITY_MODE_ALIASES)})"
        )
    return canonical


class SeverityHead(nn.Module):
    """Map a trunk context ``[B, H]`` to severity outputs.

    Parameters
    ----------
    hidden_size:
        Width ``H`` of the incoming context vector.
    use_kan:
        If ``True`` (default), each output is produced by a ``KANLinear``
        applied to ``tanh(context)`` (so the B-spline path is active). If
        ``False``, plain ``nn.Linear`` heads are used.
    severity_mode:
        ``"point"`` (default) -- the point-regression head: a single
        ``b_over_blim`` scalar per sample. ``"quantile"`` -- emit one column per
        level in ``quantiles`` (prediction intervals for the heavy-tailed
        severity target), routed through a pinball loss at train time.
    quantiles:
        Quantile levels for ``severity_mode="quantile"`` (each strictly in
        ``(0, 1)``; ignored in ``"point"`` mode). The point estimate
        ``b_over_blim`` is the column nearest the median (0.5).

    Returns (from :meth:`forward`)
    ------------------------------
    dict with
        ``"b_over_blim"`` : Tensor ``[B]`` -- point severity (median column in
        quantile mode).
        ``"chatter_logit"`` : Tensor ``[B]`` -- raw binary-chatter logit.
        (quantile mode only) ``"b_over_blim_quantiles"`` : Tensor ``[B, Q]`` and
        ``"quantile_levels"`` : the ``quantiles`` tuple.
    """

    def __init__(
        self,
        hidden_size: int,
        use_kan: bool = True,
        severity_mode: str = DEFAULT_SEVERITY_MODE,
        quantiles=(0.1, 0.5, 0.9),
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.use_kan = bool(use_kan)
        # Normalise at the boundary so only the canonical spelling is ever
        # stored, compared against, or written into an artifact.
        severity_mode = normalise_severity_mode(severity_mode)
        self.severity_mode = severity_mode
        if severity_mode == "quantile":
            q = tuple(float(x) for x in quantiles)
            if len(q) == 0:
                raise ValueError("quantile mode requires at least one quantile level")
            if any(not (0.0 < x < 1.0) for x in q):
                raise ValueError(
                    f"quantile levels must lie strictly in (0, 1); got {q}"
                )
            self.quantiles = q
            # Point estimate = the column nearest the median (0.5).
            self._point_index = min(range(len(q)), key=lambda i: abs(q[i] - 0.5))
            reg_out = len(q)
        else:
            self.quantiles = None
            self._point_index = 0
            reg_out = 1

        if self.use_kan:
            self.reg_head = KANLinear(self.hidden_size, reg_out)
            self.cls_head = KANLinear(self.hidden_size, 1)
        else:
            self.reg_head = nn.Linear(self.hidden_size, reg_out)
            self.cls_head = nn.Linear(self.hidden_size, 1)

    def forward(self, context: torch.Tensor) -> dict:
        # KANLinear's spline path is only active in (-1, 1); squash first.
        x = torch.tanh(context) if self.use_kan else context
        chatter_logit = self.cls_head(x).squeeze(-1)  # [B, 1] -> [B]
        reg = self.reg_head(x)  # [B, reg_out]
        if self.severity_mode == "quantile":
            point = reg[:, self._point_index]  # [B]
            return {
                "b_over_blim": point,
                "b_over_blim_quantiles": reg,  # [B, Q]
                "quantile_levels": self.quantiles,
                "chatter_logit": chatter_logit,
            }
        b_over_blim = reg.squeeze(-1)  # [B, 1] -> [B]
        return {"b_over_blim": b_over_blim, "chatter_logit": chatter_logit}


class TrajectoryHead(nn.Module):
    """Autoregressive decoder that rolls out downstream channels.

    The context initializes a GRU hidden state; the ``seed_step`` is the first
    decoder input. At each step a small output projection maps the hidden state
    to the next channel vector, which is fed back as the next input. The rollout
    of ``n_steps`` produces ``[B, n_steps, n_channels_out]``.

    Delay (``tau``) handling
    ------------------------
    ``tau = round(fs / tpf)`` is the kinematic tooth-pass delay (it does NOT
    require the blocked Altintas-Budak force constants). When the rollout has
    already produced at least ``tau`` steps, the channel vector from ``tau``
    steps back is mixed into the current decoder input through a learnable
    ``delay_mix`` gate -- a delayed-feedback term that mirrors the DDE structure
    of regenerative chatter. Before ``tau`` steps of history exist there is no
    delayed sample, so the decoder is purely autoregressive there. ``tau`` may
    be an ``int`` or a per-sample ``LongTensor[B]``; the minimum over the batch
    is used to decide when *any* sample has enough history (per-sample gating is
    deferred to the learnable-tau unlock step).

    Parameters
    ----------
    hidden_size:
        Width ``H`` of the context / GRU hidden state.
    n_channels_out:
        Number of channels ``Cout`` produced per step.
    use_kan:
        If ``True`` (default), the output projection is a ``KANLinear`` fed
        ``tanh`` of the hidden state; else a plain ``nn.Linear``.
    """

    def __init__(
        self, hidden_size: int, n_channels_out: int, use_kan: bool = True
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_channels_out = int(n_channels_out)
        self.use_kan = bool(use_kan)

        # Initialize the recurrent hidden state from the trunk context.
        self.init_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        # GRU cell consuming a channel vector (+ delayed feedback already mixed
        # into that vector) and the previous hidden state.
        self.cell = nn.GRUCell(self.n_channels_out, self.hidden_size)
        # Learnable scalar gate weighting the delayed-feedback channel vector.
        self.delay_mix = nn.Parameter(torch.zeros(1))

        if self.use_kan:
            self.out_proj = KANLinear(self.hidden_size, self.n_channels_out)
        else:
            self.out_proj = nn.Linear(self.hidden_size, self.n_channels_out)

    def _tau_int(self, tau) -> int:
        """Reduce ``tau`` (int or LongTensor[B]) to a single int threshold."""
        if torch.is_tensor(tau):
            if tau.numel() == 0:
                return 1
            return int(tau.min().item())
        return int(tau)

    def forward(
        self,
        context: torch.Tensor,
        seed_step: torch.Tensor,
        n_steps: int,
        tau,
    ) -> torch.Tensor:
        n_steps = int(n_steps)
        tau_i = max(1, self._tau_int(tau))

        h = torch.tanh(self.init_hidden(context))  # [B, H]
        inp = seed_step  # [B, Cout]
        history: list[torch.Tensor] = []  # produced channel vectors

        for t in range(n_steps):
            # Delayed-feedback term: once tau steps of output exist, mix the
            # channel vector from tau steps back into the current input.
            if t >= tau_i:
                delayed = history[t - tau_i]
                inp = inp + self.delay_mix * delayed

            h = self.cell(inp, h)  # [B, H]
            x = torch.tanh(h) if self.use_kan else h
            step_out = self.out_proj(x)  # [B, Cout]
            history.append(step_out)
            inp = step_out  # feed back for the next step

        return torch.stack(history, dim=1)  # [B, n_steps, Cout]


class PhysicsResidualHead(nn.Module):
    """Documented stub for the physics-residual head (BLOCKED).

    The Altintas-Budak regenerative-chatter residual needs cutting-force
    constants and an engagement convention that have not yet been provided
    (owed by Prof. Abell). The head therefore stays disabled and any call to
    :meth:`forward` raises loudly rather than silently returning garbage.
    """

    enabled: bool = False

    def forward(self, *args, **kwargs):  # noqa: D401, ARG002 - intentional stub
        raise NotImplementedError(
            "physics-residual head is blocked: Altintas-Budak cutting-force "
            "constants / engagement convention not yet provided"
        )


@runtime_checkable
class Phase3Model(Protocol):
    """Structural protocol for every Phase 3 model.

    A Phase 3 model consumes a batch dict and returns a dict whose keys are a
    subset of ``{"b_over_blim", "chatter_logit", "trajectory"}``. Runtime
    checkable so the harness can assert ``isinstance(model, Phase3Model)``.
    """

    def forward(self, batch: dict) -> dict:  # pragma: no cover - protocol
        ...
