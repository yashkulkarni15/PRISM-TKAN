"""Temporal Kolmogorov-Arnold Network (TKAN) cell, encoder, and optional sigKAN.

This module builds the *temporal* layer of the Phase 3 KAN substrate on top of
:class:`pri2m.models.kan.KANLinear`:

    KANLinear (per-edge B-spline + SiLU base)
        -> TKANCell  (a KAN-GRU recurrent cell, optionally delay-aware)
        -> TKANEncoder (stacked cells over a sequence; feeds a kinematic delay)
        -> (optional) SignatureWeighting (path-signature gate, the "sigKAN" trunk)

Design (KAN-GRU cell)
---------------------
A GRU cell whose three affine gate maps are replaced by :class:`KANLinear`:

    g       = cat([x_t, h_prev], -1)        (+ x_delay if use_delay)
    z       = sigmoid(KAN_z(tanh(g)))       update gate   [B, H]
    r       = sigmoid(KAN_r(tanh(g)))       reset gate    [B, H]
    g2      = cat([x_t, r * h_prev], -1)    (+ x_delay if use_delay)
    h_tilde = tanh(KAN_h(tanh(g2)))         candidate     [B, H]
    h_t     = (1 - z) * h_prev + z * h_tilde

**tanh squashing.** ``KANLinear``'s interpretable spline path is only active for
inputs inside ``grid_range=(-1, 1)``; outside that range only the SiLU base acts.
Hidden states and raw signals routinely exceed that band, so every tensor is
squashed with :func:`torch.tanh` immediately before it enters *any* ``KANLinear``
(both the gate input ``g`` and the candidate input ``g2``). This keeps the spline
path engaged and the model interpretable.

Delay
-----
The encoder feeds each layer-0 step a *delayed* slice of the input sequence,
``delayed_input(seq, t, tau)`` with ``tau = round(fs / tpf)`` (the tooth-pass
period in samples). Steps before the delay (``t - tau < 0``) receive zeros.
Upper layers (>0) consume the previous layer's hidden sequence and receive
``x_delay=None``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pri2m.models.kan import KANLinear

__all__ = [
    "delayed_input",
    "TKANCell",
    "TKANEncoder",
    "path_signature",
    "SignatureWeighting",
]


# --------------------------------------------------------------------------- #
# Delayed-input helper
# --------------------------------------------------------------------------- #
def delayed_input(seq: torch.Tensor, t: int, tau) -> torch.Tensor:
    """Return the delayed slice ``seq[:, t - tau, :]`` at time ``t``.

    Parameters
    ----------
    seq : Tensor ``[B, L, C]``
        The input sequence.
    t : int
        The current time index.
    tau : int | LongTensor ``[B]``
        The delay in samples. Either a single int applied to every sample, or a
        per-sample ``LongTensor`` of shape ``[B]``.

    Returns
    -------
    Tensor ``[B, C]``
        The delayed slice. Rows where ``t - tau < 0`` are all zeros (the signal
        has not yet "happened" for that sample).
    """
    B, L, C = seq.shape

    if isinstance(tau, torch.Tensor):
        if tau.ndim != 1 or tau.shape[0] != B:
            raise ValueError(
                f"per-sample tau must be a 1-D LongTensor of length B={B}, "
                f"got shape {tuple(tau.shape)}"
            )
        # Source index per sample; clamp to [0, L-1] so the gather is always
        # in-bounds, then mask the rows that should be zero.
        src = (t - tau).to(device=seq.device, dtype=torch.long)  # [B]
        valid = src >= 0  # [B] bool
        src_clamped = src.clamp(min=0, max=L - 1)
        # gather along the time axis: index [B, 1, C] -> out [B, 1, C].
        idx = src_clamped.view(B, 1, 1).expand(B, 1, C)
        out = torch.gather(seq, 1, idx).squeeze(1)  # [B, C]
        out = out * valid.view(B, 1).to(out.dtype)
        return out

    # Scalar int tau.
    src = int(t) - int(tau)
    if src < 0:
        return seq.new_zeros(B, C)
    return seq[:, src, :]


# --------------------------------------------------------------------------- #
# KAN-GRU recurrent cell
# --------------------------------------------------------------------------- #
class TKANCell(nn.Module):
    """A KAN-GRU recurrent cell (optionally delay-aware).

    Parameters
    ----------
    input_size : int
        Channel count of ``x_t`` (and of ``x_delay`` when ``use_delay`` is True).
    hidden_size : int
        Hidden-state width ``H``.
    grid_size, spline_order : int
        Forwarded to every internal :class:`KANLinear`.
    use_delay : bool
        If True, a delayed input ``x_delay`` (shape ``[B, input_size]``) is
        concatenated into both gate inputs. A None ``x_delay`` is treated as
        zeros so the cell still runs (e.g. on upper encoder layers or early
        steps).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        grid_size: int = 5,
        spline_order: int = 3,
        use_delay: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.use_delay = bool(use_delay)

        # Gate input width: [x_t, h_prev] (+ x_delay if used).
        delay_extra = self.input_size if self.use_delay else 0
        gate_in = self.input_size + self.hidden_size + delay_extra

        kw = dict(grid_size=grid_size, spline_order=spline_order)
        # Update / reset gates and the candidate map -> each outputs H.
        self.kan_z = KANLinear(gate_in, self.hidden_size, **kw)
        self.kan_r = KANLinear(gate_in, self.hidden_size, **kw)
        self.kan_h = KANLinear(gate_in, self.hidden_size, **kw)

    def _maybe_delay(self, x_delay: torch.Tensor | None, ref: torch.Tensor) -> list[torch.Tensor]:
        """Return ``[x_delay]`` (or zeros of the right shape) when use_delay, else ``[]``."""
        if not self.use_delay:
            return []
        if x_delay is None:
            x_delay = ref.new_zeros(ref.shape[0], self.input_size)
        return [x_delay]

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        x_delay: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One recurrent step.

        Parameters
        ----------
        x_t : Tensor ``[B, input_size]``
        h_prev : Tensor ``[B, hidden_size]``
        x_delay : Tensor ``[B, input_size]`` or None
            Delayed input; only used when ``use_delay`` is True.

        Returns
        -------
        Tensor ``[B, hidden_size]`` -- the new hidden state ``h_t``.
        """
        delay = self._maybe_delay(x_delay, x_t)

        # Gate input g = cat([x_t, h_prev], (+ x_delay)).  tanh-squash before KAN.
        g = torch.cat([x_t, h_prev, *delay], dim=-1)
        g = torch.tanh(g)
        z = torch.sigmoid(self.kan_z(g))  # update gate [B, H]
        r = torch.sigmoid(self.kan_r(g))  # reset gate  [B, H]

        # Candidate input g2 = cat([x_t, r*h_prev], (+ x_delay)). tanh-squash.
        g2 = torch.cat([x_t, r * h_prev, *delay], dim=-1)
        g2 = torch.tanh(g2)
        h_tilde = torch.tanh(self.kan_h(g2))  # candidate [B, H]

        h_t = (1.0 - z) * h_prev + z * h_tilde
        return h_t


# --------------------------------------------------------------------------- #
# Stacked recurrent encoder
# --------------------------------------------------------------------------- #
class TKANEncoder(nn.Module):
    """Stacked :class:`TKANCell` encoder over a sequence, with kinematic delay.

    Layer 0 consumes ``seq`` and is fed ``delayed_input(seq, t, tau)`` as its
    ``x_delay``. Layers >0 consume the previous layer's hidden sequence and are
    fed ``x_delay=None`` (the delay is applied once, on the raw input).

    Parameters
    ----------
    input_size : int
        Channel count ``C`` of the input sequence.
    hidden_size : int
        Hidden width ``H`` of every layer.
    num_layers : int
        Number of stacked cells.
    grid_size, spline_order : int
        Forwarded to every cell's :class:`KANLinear`.
    use_delay : bool
        If True (default), layer 0 is delay-aware.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        use_delay: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.use_delay = bool(use_delay)

        kw = dict(grid_size=grid_size, spline_order=spline_order)
        layers = []
        for layer_idx in range(self.num_layers):
            in_size = self.input_size if layer_idx == 0 else self.hidden_size
            # Only the bottom layer is delay-aware; upper layers see hidden states.
            layer_use_delay = self.use_delay and (layer_idx == 0)
            layers.append(TKANCell(in_size, self.hidden_size, use_delay=layer_use_delay, **kw))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        seq: torch.Tensor,
        tau,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a sequence.

        Parameters
        ----------
        seq : Tensor ``[B, L, C]``
        tau : int | LongTensor ``[B]``
            Kinematic delay in samples, applied on the raw input at layer 0.

        Returns
        -------
        (context, all_hidden) :
            ``context`` ``[B, H]`` is the top layer's final hidden state.
            ``all_hidden`` ``[B, L, H]`` is the top layer's hidden state over time.
        """
        B, L, C = seq.shape
        if C != self.input_size:
            raise ValueError(f"expected input_size={self.input_size}, got C={C}")

        # The input to layer 0 is the raw sequence; upper layers consume the
        # hidden sequence produced by the layer below.
        layer_input = seq  # [B, L, C_in]
        all_hidden: torch.Tensor | None = None

        for layer_idx, cell in enumerate(self.layers):
            in_size = layer_input.shape[-1]
            h = seq.new_zeros(B, self.hidden_size)  # h0 = zeros
            hidden_steps = []
            for t in range(L):
                x_t = layer_input[:, t, :]  # [B, in_size]
                if layer_idx == 0 and self.use_delay:
                    x_delay = delayed_input(seq, t, tau)  # [B, C]
                else:
                    x_delay = None
                h = cell(x_t, h, x_delay)
                hidden_steps.append(h)
            layer_hidden = torch.stack(hidden_steps, dim=1)  # [B, L, H]
            all_hidden = layer_hidden
            layer_input = layer_hidden  # feed up to the next layer

        assert all_hidden is not None  # num_layers >= 1
        context = all_hidden[:, -1, :]  # top layer final hidden state [B, H]
        return context, all_hidden


# --------------------------------------------------------------------------- #
# Optional sigKAN path (guarded iisignature import)
# --------------------------------------------------------------------------- #
def path_signature(window: np.ndarray, depth: int = 2) -> np.ndarray:
    """Truncated path signature of a multi-channel window.

    Parameters
    ----------
    window : np.ndarray ``[L, C]``
        A path sampled at ``L`` time steps over ``C`` channels.
    depth : int
        Truncation depth of the signature (default 2).

    Returns
    -------
    np.ndarray ``[sig_dim]``
        The truncated path signature flattened to 1-D, where
        ``sig_dim == iisignature.siglength(C, depth)``.

    Raises
    ------
    ImportError
        If the optional ``iisignature`` dependency is not installed.
    """
    try:
        import iisignature
    except ImportError as exc:  # pragma: no cover - exercised only when absent
        raise ImportError(
            "path_signature requires the optional 'iisignature' package, which "
            "is not installed. Install the sigKAN extra (e.g. "
            "`pip install iisignature` or `pip install -e .[sigkan]`) to use the "
            "signature-weighted TKAN trunk."
        ) from exc

    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2:
        raise ValueError(f"window must be 2-D [L, C], got shape {window.shape}")
    sig = iisignature.sig(window, int(depth))
    return np.asarray(sig, dtype=np.float64).reshape(-1)


class SignatureWeighting(nn.Module):
    """Gate an encoder context by path-signature-derived features ("sigKAN").

    A small :class:`KANLinear` maps the signature features to a per-dimension
    gate in ``(0, 1)`` (via sigmoid) that re-weights the encoder context.

    Parameters
    ----------
    sig_dim : int
        Dimension of the path-signature feature vector.
    hidden_size : int
        Width ``H`` of the context being gated.
    """

    def __init__(self, sig_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.sig_dim = int(sig_dim)
        self.hidden_size = int(hidden_size)
        self.kan_gate = KANLinear(self.sig_dim, self.hidden_size)

    def forward(self, context: torch.Tensor, sig_feats: torch.Tensor) -> torch.Tensor:
        """Gate ``context`` ``[B, H]`` by ``sig_feats`` ``[B, sig_dim]`` -> ``[B, H]``."""
        # tanh-squash the signature features before the KAN, then sigmoid-gate.
        gate = torch.sigmoid(self.kan_gate(torch.tanh(sig_feats)))  # [B, H]
        return context * gate
