"""Assembled TKAN model: TKAN encoder trunk + Phase 3 heads.

Composes the temporal KAN encoder (:mod:`pri2m.models.tkan`) with the
severity head (and an optional, gated trajectory head) from
:mod:`pri2m.models.heads` into a single ``forward(batch) -> dict`` that
satisfies :class:`pri2m.models.heads.Phase3Model`. This is the concrete TKAN
counterpart to the :class:`pri2m.models.baseline.MLPBaseline` control.

The physics-residual head is intentionally NOT wired in (blocked on the
Altintas-Budak constants); the trainer keeps the severity head as the
currently-trainable target and gates the trajectory head separately.

Optional process-parameter side input (Gao et al., JMSE 2025)
-------------------------------------------------------------
Gao et al. report that concatenating scalar process parameters (spindle
speed, feed, depth) as a side branch both raised chatter-classification
accuracy and stabilized training. ``side_input=True`` reproduces that idea:
scalar batch features (default ``omega``/``tpf``/``tau``) are normalized by
buffered train-fold statistics (:meth:`TKANModel.set_side_norm`), passed
through a small MLP, and concatenated onto the encoder context before the
severity head. The side branch feeds the severity head only; the trajectory
head keeps the raw encoder context.

Leakage guard: the training label is ``b_over_blim = b / blim``, so ``b`` /
``b_mm`` / ``blim`` each leak the answer. :data:`FORBIDDEN_SIDE_FEATURES`
rejects them (and the targets) at construction time — this is a hard guard,
not style.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from pri2m.models.heads import SeverityHead, TrajectoryHead
from pri2m.models.tkan import TKANEncoder

#: Default scalar process parameters fed to the side branch (Gao JMSE-2025).
DEFAULT_SIDE_FEATURES = ("omega", "tpf", "tau")

#: Hard leakage guard: the label is ``b_over_blim = b / blim``, so ``b`` /
#: ``b_mm`` / ``blim`` each leak the answer, and the targets obviously do.
FORBIDDEN_SIDE_FEATURES = frozenset(
    {"b", "b_mm", "blim", "b_over_blim", "label", "sample_weight"}
)


def avg_pool_time(window: torch.Tensor, pool: int) -> torch.Tensor:
    """Average-pool a ``[B, L, C]`` window over time by ``pool`` -> ``[B, L//pool, C]``.

    Acts as an anti-alias low-pass + decimate, shortening the sequence the
    recurrent trunk has to walk (its cost is linear in length). ``pool <= 1``
    returns the window unchanged; a non-divisible tail is trimmed.
    """
    if pool <= 1:
        return window
    B, L, C = window.shape
    out_len = L // pool
    if out_len == 0:
        raise ValueError(f"pool={pool} exceeds sequence length L={L}")
    return window[:, : out_len * pool, :].reshape(B, out_len, pool, C).mean(dim=2)


def scale_tau(tau, pool: int):
    """Rescale the kinematic delay (in samples) into pooled-step units.

    After pooling by ``pool``, a delay of ``tau`` original samples spans
    ``round(tau / pool)`` pooled steps, clamped to ``>= 1`` so the delay never
    collapses to the current step. ``pool <= 1`` is the identity.
    """
    if pool <= 1:
        return tau
    if isinstance(tau, torch.Tensor):
        return torch.clamp((tau.float() / pool).round().long(), min=1)
    return max(1, round(int(tau) / pool))


class TKANModel(nn.Module):
    """TKAN encoder trunk feeding a severity head (+ optional trajectory head).

    Parameters
    ----------
    input_size:
        Channel count ``C`` of the input window.
    hidden_size:
        Trunk hidden width ``H``.
    num_layers, grid_size, spline_order, use_delay:
        Forwarded to :class:`~pri2m.models.tkan.TKANEncoder`.
    use_kan_heads:
        Forwarded to the heads (KAN vs plain-linear output maps).
    with_trajectory:
        If ``True``, also roll out the trajectory head and add a ``"trajectory"``
        key to the output. Off by default -- the trajectory head is gated.
    traj_channels:
        Indices of the downstream channels the trajectory head reproduces
        (default ``(0, 2)`` == ``xtpos, Forcex`` for the canonical channel order
        ``(xtpos, ytpos, Forcex, Forcey)``).
    traj_steps:
        Rollout length when ``with_trajectory`` is True.
    pool:
        Temporal average-pool factor applied to the window before the encoder
        (``1`` = no pooling). Pooling by ``p`` cuts the recurrence length ``p``x
        (its cost is linear in length) and rescales ``tau`` into pooled steps.
    side_input:
        If ``True``, enable the Gao-style process-parameter side branch: the
        scalar batch features named in ``side_features`` are normalized
        (see :meth:`set_side_norm`), passed through a small MLP, and
        concatenated onto the encoder context before the severity head. The
        trajectory head is unaffected — it keeps the raw ``[B, H]`` context.
        Off by default; when off, no side modules or buffers are created and
        the state_dict is byte-identical to the pre-side-input model.
    side_features:
        Names of the scalar batch keys to use (default
        :data:`DEFAULT_SIDE_FEATURES`). Keys in
        :data:`FORBIDDEN_SIDE_FEATURES` are rejected with ``ValueError``
        because they leak the ``b_over_blim`` label.
    side_hidden:
        Width of the side-branch MLP output concatenated onto the context.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        use_delay: bool = True,
        use_kan_heads: bool = True,
        with_trajectory: bool = False,
        traj_channels: Sequence[int] = (0, 2),
        traj_steps: int = 16,
        pool: int = 1,
        side_input: bool = False,
        side_features: Sequence[str] = DEFAULT_SIDE_FEATURES,
        side_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.pool = int(pool)
        self.with_trajectory = bool(with_trajectory)
        self.traj_channels = tuple(int(c) for c in traj_channels)
        self.traj_steps = int(traj_steps)
        self.side_input = bool(side_input)
        self.side_features = tuple(str(f) for f in side_features)
        self.side_hidden = int(side_hidden)

        self.encoder = TKANEncoder(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            use_delay=use_delay,
        )

        severity_width = self.hidden_size
        if self.side_input:
            for key in self.side_features:
                if key in FORBIDDEN_SIDE_FEATURES:
                    raise ValueError(
                        f"side feature {key!r} is forbidden: the label is "
                        f"b_over_blim = b/blim, so b/b_mm/blim (and the targets) "
                        f"leak the answer (forbidden: "
                        f"{sorted(FORBIDDEN_SIDE_FEATURES)})"
                    )
            n_side = len(self.side_features)
            self.side_mlp = nn.Sequential(
                nn.Linear(n_side, self.side_hidden),
                nn.SiLU(),
                nn.Linear(self.side_hidden, self.side_hidden),
                nn.SiLU(),
            )
            # Train-fold normalization stats live in the state_dict so
            # checkpoints are self-contained (see set_side_norm).
            self.register_buffer("side_mean", torch.zeros(n_side))
            self.register_buffer("side_std", torch.ones(n_side))
            severity_width += self.side_hidden

        self.severity = SeverityHead(severity_width, use_kan=use_kan_heads)
        if self.with_trajectory:
            self.trajectory = TrajectoryHead(
                self.hidden_size, len(self.traj_channels), use_kan=use_kan_heads
            )

    def set_side_norm(self, mean, std) -> None:
        """Copy train-fold side-feature statistics into the norm buffers.

        ``mean`` / ``std`` are array-likes or tensors of length
        ``len(side_features)``; ``std`` is floored at ``1e-8`` so a constant
        feature cannot divide by zero. The trainer must call this with
        TRAIN-fold-only statistics (val/test stats would leak).
        """
        if not self.side_input:
            raise ValueError("set_side_norm requires side_input=True")
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std_t = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        n_side = len(self.side_features)
        if mean_t.numel() != n_side or std_t.numel() != n_side:
            raise ValueError(
                f"set_side_norm expects length-{n_side} mean/std for "
                f"side_features={self.side_features}; got mean of length "
                f"{mean_t.numel()} and std of length {std_t.numel()}"
            )
        self.side_mean.copy_(mean_t)
        self.side_std.copy_(std_t.clamp(min=1e-8))

    def _side_context(self, batch: dict) -> torch.Tensor:
        """Build the normalized side-branch output ``[B, side_hidden]``."""
        feats = []
        for key in self.side_features:
            if key not in batch:
                raise ValueError(
                    f"side_input requires batch key {key!r} "
                    f"(side_features={self.side_features}); refusing to "
                    f"default silently"
                )
            feats.append(torch.as_tensor(batch[key]).float())
        x_side = torch.stack(feats, dim=-1)  # [B, S]
        z = (x_side - self.side_mean) / self.side_std
        return self.side_mlp(z)

    def forward(self, batch: dict) -> dict:
        window = batch["window"]  # [B, L, C]
        tau = batch.get("tau", 1)
        if self.pool > 1:
            # Shorten the sequence (and rescale the delay) before the recurrence.
            window = avg_pool_time(window, self.pool)
            tau = scale_tau(tau, self.pool)
        context, _ = self.encoder(window, tau)  # [B, H]
        sev_context = context
        if self.side_input:
            # Side features come from the raw batch (tau in original samples,
            # not pool-rescaled) and never touch the pooled/encoder path.
            sev_context = torch.cat([context, self._side_context(batch)], dim=-1)
        out = self.severity(sev_context)
        if self.with_trajectory:
            # Seed the rollout with the last observed step's downstream channels.
            seed = window[:, -1, list(self.traj_channels)]  # [B, len(traj_channels)]
            # Trajectory head keeps the raw encoder context (no side input).
            out["trajectory"] = self.trajectory(context, seed, self.traj_steps, tau)
        return out
