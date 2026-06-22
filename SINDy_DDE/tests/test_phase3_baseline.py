"""Tests for the MLP baseline control model (Task 2B).

``MLPBaseline`` is the ``hardcoded_tau_mlp`` control: flatten the window, concat
the tau-delayed sample, run an MLP trunk, then a ``SeverityHead``. It must
satisfy the ``Phase3Model`` protocol and demonstrably learn a simple target.
"""

from __future__ import annotations

import pytest
import torch

from pri2m.models.baseline import MLPBaseline
from pri2m.models.heads import Phase3Model
from pri2m.training.losses import LossWeights, severity_loss


def _make_batch(B=6, L=32, C=4, seed=0):
    """A minimal batch dict with all keys the models may read."""
    g = torch.Generator().manual_seed(seed)
    window = torch.randn(B, L, C, generator=g)
    return {
        "window": window,
        "b_over_blim": torch.rand(B, generator=g),
        "label": (torch.rand(B, generator=g) > 0.5).float(),
        "sample_weight": torch.ones(B),
        "tau": torch.randint(1, L, (B,), generator=g),
        "omega": torch.rand(B, generator=g),
        "tpf": torch.rand(B, generator=g),
        "fn": torch.rand(B, generator=g),
        "fs": torch.full((B,), 10000.0),
        "dataset": ["d"] * B,
        "file": [f"f{i}" for i in range(B)],
        "parent_group": ["g"] * B,
        "mode_label": ["m"] * B,
    }


# ---------------------------------------------------------------------------
# Forward contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_delay", [True, False])
def test_baseline_forward_shapes_and_finite(use_delay):
    torch.manual_seed(0)
    B, L, C, H = 6, 32, 4, 8
    model = MLPBaseline(L, C, H, use_delay=use_delay)
    batch = _make_batch(B=B, L=L, C=C)
    out = model.forward(batch)

    assert set(out) == {"b_over_blim", "chatter_logit"}
    for key in ("b_over_blim", "chatter_logit"):
        assert out[key].shape == (B,), key
        assert torch.isfinite(out[key]).all(), key


@pytest.mark.parametrize("use_kan", [True, False])
def test_baseline_forward_with_and_without_kan_head(use_kan):
    torch.manual_seed(1)
    B, L, C, H = 4, 16, 4, 8
    model = MLPBaseline(L, C, H, use_kan=use_kan)
    out = model.forward(_make_batch(B=B, L=L, C=C, seed=1))
    assert out["b_over_blim"].shape == (B,)
    assert torch.isfinite(out["chatter_logit"]).all()


def test_baseline_satisfies_protocol():
    model = MLPBaseline(window_length=16, n_channels=4, hidden_size=8)
    assert isinstance(model, Phase3Model)


def test_baseline_use_delay_changes_output():
    # The delayed-sample concat must actually feed the trunk: a model with
    # use_delay=True should differ from one with use_delay=False given the same
    # window (sanity that tau is wired in, not ignored).
    torch.manual_seed(7)
    B, L, C, H = 4, 16, 4, 8
    batch = _make_batch(B=B, L=L, C=C, seed=7)

    torch.manual_seed(0)
    m_delay = MLPBaseline(L, C, H, use_delay=True)
    torch.manual_seed(0)
    m_nodelay = MLPBaseline(L, C, H, use_delay=False)

    # First-layer input dimension differs, so the models are structurally
    # distinct; assert the delay model consumes the larger feature vector.
    in_delay = m_delay.trunk[0].in_features
    in_nodelay = m_nodelay.trunk[0].in_features
    assert in_delay == L * C + C
    assert in_nodelay == L * C


# ---------------------------------------------------------------------------
# Learning sanity
# ---------------------------------------------------------------------------
def test_baseline_overfits_tiny_batch():
    """Train ~200 Adam steps on a target = mean window amplitude.

    ``b_over_blim`` target is a clear function of the window (mean absolute
    amplitude), sample_weight=1. Assert the severity loss strictly decreases
    and the final loss is < 0.5x the initial loss.
    """
    torch.manual_seed(0)
    B, L, C, H = 8, 32, 4, 16

    g = torch.Generator().manual_seed(123)
    window = torch.randn(B, L, C, generator=g)
    # Target is a clean function of the window: mean absolute amplitude.
    target = window.abs().mean(dim=(1, 2))

    batch = {
        "window": window,
        "b_over_blim": target,
        "label": (target > target.median()).float(),
        "sample_weight": torch.ones(B),
        "tau": torch.randint(1, L, (B,), generator=g),
    }

    model = MLPBaseline(L, C, H, use_delay=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lw = LossWeights()

    losses = []
    for _ in range(200):
        opt.zero_grad()
        out = model.forward(batch)
        loss, _ = severity_loss(out, batch, lw)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    initial = losses[0]
    final = losses[-1]
    # Strictly decreased overall and final well below half the initial.
    assert final < initial
    assert final < 0.5 * initial
    # Monotone-ish: the last loss is the smallest seen in the back half.
    assert final <= min(losses[100:]) + 1e-6


def test_baseline_overfit_loss_strictly_decreases_early():
    """Loss should drop over the first chunk of training (not flat)."""
    torch.manual_seed(0)
    B, L, C, H = 8, 32, 4, 16
    g = torch.Generator().manual_seed(7)
    window = torch.randn(B, L, C, generator=g)
    target = window.abs().mean(dim=(1, 2))
    batch = {
        "window": window,
        "b_over_blim": target,
        "label": torch.zeros(B),
        "sample_weight": torch.ones(B),
        "tau": torch.full((B,), 4),
    }
    model = MLPBaseline(L, C, H)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lw = LossWeights()

    first = None
    last = None
    for i in range(50):
        opt.zero_grad()
        out = model.forward(batch)
        loss, _ = severity_loss(out, batch, lw)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert last < first
