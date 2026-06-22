"""MLP baseline control model for Phase 3 (Task 2B).

:class:`MLPBaseline` is the ``hardcoded_tau_mlp`` control referenced in the
design spec: it flattens the raw window, optionally concatenates the
tau-delayed sample, runs a small MLP trunk to a context vector, then a
:class:`~pri2m.models.heads.SeverityHead`. It satisfies the
:class:`~pri2m.models.heads.Phase3Model` protocol so the harness can train it
interchangeably with ``TKANModel``.

The "hardcoded tau" is the point of the control: rather than learning a delay,
the model is simply *handed* the kinematic tooth-pass delay
``tau = round(fs / tpf)`` and concatenates the sample that many steps back.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pri2m.models.heads import SeverityHead


class MLPBaseline(nn.Module):
    """Flatten-window MLP with an optional tau-delayed sample (control model).

    Parameters
    ----------
    window_length:
        Window length ``L``.
    n_channels:
        Channel count ``C``.
    hidden_size:
        Trunk / context width ``H``.
    use_delay:
        If ``True`` (default), concatenate the tau-delayed sample
        ``window[b, clamp(L-1-tau[b], 0, L-1), :]`` (shape ``[B, C]``) to the
        flattened window before the trunk. This is the "hardcoded tau" feature.
    use_kan:
        Passed through to the :class:`SeverityHead`. Default ``False`` (the
        baseline is a plain-MLP control).

    Forward
    -------
    Consumes ``batch["window"]`` ``[B, L, C]`` (and ``batch["tau"]`` ``[B]`` when
    ``use_delay``); returns the severity dict
    ``{"b_over_blim": [B], "chatter_logit": [B]}``.
    """

    def __init__(
        self,
        window_length: int,
        n_channels: int,
        hidden_size: int,
        use_delay: bool = True,
        use_kan: bool = False,
    ) -> None:
        super().__init__()
        self.window_length = int(window_length)
        self.n_channels = int(n_channels)
        self.hidden_size = int(hidden_size)
        self.use_delay = bool(use_delay)

        in_features = self.window_length * self.n_channels
        if self.use_delay:
            in_features += self.n_channels  # + the delayed [B, C] sample

        self.trunk = nn.Sequential(
            nn.Linear(in_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.severity = SeverityHead(self.hidden_size, use_kan=use_kan)

    def _delayed_sample(
        self, window: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """Gather ``window[b, clamp(L-1-tau[b], 0, L-1), :]`` -> ``[B, C]``."""
        B, L, _ = window.shape
        tau = tau.to(window.device).long()
        idx = torch.clamp(L - 1 - tau, min=0, max=L - 1)  # [B]
        batch_idx = torch.arange(B, device=window.device)
        return window[batch_idx, idx, :]  # [B, C]

    def forward(self, batch: dict) -> dict:
        window = batch["window"]  # [B, L, C]
        B = window.shape[0]
        feats = window.reshape(B, -1)  # [B, L*C]

        if self.use_delay:
            delayed = self._delayed_sample(window, batch["tau"])  # [B, C]
            feats = torch.cat([feats, delayed], dim=-1)  # [B, L*C + C]

        context = self.trunk(feats)  # [B, H]
        return self.severity(context)
