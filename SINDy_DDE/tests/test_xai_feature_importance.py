"""Tests for dataset-level feature-importance XAI (feature_importance.py).

These are the "which features drive the decision" views that sit on top of the
per-window Integrated-Gradients attribution from :mod:`pri2m.xai.attribution`:

* ``channel_attribution`` / ``attribution_by_class`` -- aggregate |IG| over a set
  of windows into a per-channel and per-lag (per-timestep) profile.
* ``permutation_importance`` -- model-agnostic drop in chatter-classification AUC
  when a single input channel (or side-feature scalar) is shuffled across the
  batch.

Ground truth comes from tiny *differentiable* stub models (no training needed):
their forward depends on a known channel / timestep, so the correct importance
ranking is known in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from pri2m.xai import feature_importance


# --------------------------------------------------------------------------- #
# Differentiable stub models with KNOWN feature dependence.
# --------------------------------------------------------------------------- #
class _ChannelStub(nn.Module):
    """forward depends ONLY on channel ``k0`` (summed over time)."""

    def __init__(self, k0: int) -> None:
        super().__init__()
        self.k0 = int(k0)
        # A real parameter so .eval()/.train() and grad machinery behave normally.
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict) -> dict:
        logit = batch["window"][:, :, self.k0].sum(dim=1) + 0.0 * self._dummy
        return {"chatter_logit": logit, "b_over_blim": logit}


class _LagStub(nn.Module):
    """forward depends ONLY on a single timestep ``t0`` of channel ``k0``."""

    def __init__(self, t0: int, k0: int) -> None:
        super().__init__()
        self.t0 = int(t0)
        self.k0 = int(k0)
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict) -> dict:
        logit = batch["window"][:, self.t0, self.k0] + 0.0 * self._dummy
        return {"chatter_logit": logit, "b_over_blim": logit}


# --------------------------------------------------------------------------- #
# 1. permutation_importance recovers the channel a model depends on.
# --------------------------------------------------------------------------- #
def test_permutation_importance_recovers_dependent_channel():
    pytest.importorskip("sklearn")
    torch.manual_seed(0)
    L, C, N, K0 = 16, 4, 64, 2
    model = _ChannelStub(K0)
    window = torch.randn(N, L, C)
    batch = {"window": window, "tau": torch.full((N,), 5, dtype=torch.long)}
    logit = model(batch)["chatter_logit"]
    labels = (logit > logit.median()).long().numpy()

    result = feature_importance.permutation_importance(
        model, batch, labels, n_repeats=3, seed=0
    )

    assert set(result) == {"baseline_auc", "channel_drop", "side_drop"}
    assert 0.0 <= result["baseline_auc"] <= 1.0
    drops = result["channel_drop"]
    assert set(drops) == set(range(C))
    # The model depends only on K0, so shuffling K0 must hurt AUC the most.
    assert max(drops, key=drops.get) == K0
    for c in range(C):
        if c != K0:
            assert drops[K0] > drops[c]


# --------------------------------------------------------------------------- #
# 2. channel_attribution recovers the timestep a model depends on (per_lag).
# --------------------------------------------------------------------------- #
def test_channel_attribution_recovers_dependent_lag():
    pytest.importorskip("captum")
    torch.manual_seed(1)
    L, C, N, T0, K0 = 20, 4, 6, 7, 1
    model = _LagStub(T0, K0)
    windows = torch.randn(N, L, C)
    taus = 5

    result = feature_importance.channel_attribution(
        model, windows, taus, target="chatter_logit", steps=24
    )

    per_channel = result["per_channel"]
    per_lag = result["per_lag"]
    assert per_channel.shape == (C,)
    assert per_lag.shape == (L,)
    assert np.isfinite(per_channel).all()
    assert np.isfinite(per_lag).all()
    # Attribution concentrates at the single timestep the model reads.
    assert int(np.argmax(per_lag)) == T0
    # ... and at the single channel it reads.
    assert int(np.argmax(per_channel)) == K0


# --------------------------------------------------------------------------- #
# 3. attribution_by_class splits the per-channel profile by label.
# --------------------------------------------------------------------------- #
def test_attribution_by_class_splits_by_label():
    pytest.importorskip("captum")
    torch.manual_seed(2)
    L, C, N, K0 = 12, 4, 8, 2
    model = _ChannelStub(K0)
    windows = torch.randn(N, L, C)
    labels = np.array([0, 1] * (N // 2))

    out = feature_importance.attribution_by_class(
        model, windows, 5, labels, target="chatter_logit", steps=16
    )

    assert set(out) == {"chatter", "stable"}
    assert out["chatter"].shape == (C,)
    assert out["stable"].shape == (C,)
    assert np.isfinite(out["chatter"]).all()
    assert np.isfinite(out["stable"]).all()


# --------------------------------------------------------------------------- #
# 4. Smoke test on a real TKANModel.
# --------------------------------------------------------------------------- #
def test_smoke_on_real_tkan_model():
    pytest.importorskip("captum")
    pytest.importorskip("sklearn")
    from pri2m.models.tkan_model import TKANModel

    torch.manual_seed(3)
    L, C, N = 18, 4, 10
    model = TKANModel(C, 8, 1, side_input=False)
    windows = torch.randn(N, L, C)
    taus = [5] * N

    result = feature_importance.channel_attribution(model, windows, taus, steps=16)
    assert result["per_channel"].shape == (C,)
    assert result["per_lag"].shape == (L,)
    assert np.isfinite(result["per_channel"]).all()
    assert np.isfinite(result["per_lag"]).all()

    batch = {"window": windows, "tau": torch.full((N,), 5, dtype=torch.long)}
    labels = np.random.default_rng(0).integers(0, 2, size=N)
    perm = feature_importance.permutation_importance(
        model, batch, labels, n_repeats=2, seed=0
    )
    assert np.isfinite(perm["baseline_auc"])
    assert 0.0 <= perm["baseline_auc"] <= 1.0


# --------------------------------------------------------------------------- #
# 5. Figure functions write a file and return the path.
# --------------------------------------------------------------------------- #
def test_plot_channel_importance_returns_path(tmp_path):
    pytest.importorskip("matplotlib")
    result = {
        "per_channel": np.array([0.1, 0.4, 0.2, 0.3]),
        "per_lag": np.array([0.2, 0.5, 0.1]),
    }
    path = tmp_path / "channel.png"
    out = feature_importance.plot_channel_importance(result, str(path))
    assert out == str(path)
    assert path.exists() and path.stat().st_size > 0


def test_plot_lag_profile_returns_path(tmp_path):
    pytest.importorskip("matplotlib")
    per_lag = np.array([0.2, 0.5, 0.1, 0.3, 0.05])
    path = tmp_path / "lag.png"
    out = feature_importance.plot_lag_profile(per_lag, str(path))
    assert out == str(path)
    assert path.exists() and path.stat().st_size > 0
