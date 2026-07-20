"""xai.case_studies: per-decision walkthrough of a chatter vs a stable window.

A white-box "case study" picks one most-confident chatter window and one most
confident stable window, then explains each decision end-to-end: the predicted
probability/severity, which raw channel the integrated-gradients attribution
leans on, the layer-0 update/reset gate trajectories over time, and the
top-firing ``KANLinear`` edges (named via the semantic edge map). The plotting
helper renders all of that into a single multi-panel figure.

The selection logic is exercised with an analytic ENERGY STUB (chatter logit =
window energy minus a threshold) whose sign split is known exactly, so the test
asserts the right index is chosen without any hard-coded model numbers. The
gate-trajectory, explanation-bundle, and figure tests run a real (untrained)
``TKANModel`` and assert shapes, ranges, and documented keys only.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from pri2m.models.tkan import delayed_input
from pri2m.models.tkan_model import TKANModel
from pri2m.xai.case_studies import (
    delay_engagement_stats,
    explain_case,
    gate_trajectories,
    plot_case,
    predict,
    select_cases,
)

CHANNELS = ("xtpos", "ytpos", "Forcex", "Forcey")


def _real_model():
    torch.manual_seed(0)
    return TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False).eval()


def _real_batch(B=8, L=24, C=4, tau=5):
    torch.manual_seed(0)
    return {
        "window": torch.randn(B, L, C),
        "tau": torch.full((B,), tau, dtype=torch.long),
    }


class _EnergyStub(torch.nn.Module):
    """Analytic stub: chatter_logit = window energy - C0, b_over_blim = energy/scale.

    High-amplitude windows have large energy -> positive logit (chatter); low
    amplitude windows -> negative logit (stable). ``C0`` is chosen between the two
    energy regimes so the sign split is exact and known, with no trained weights.
    """

    def __init__(self, c0: float):
        super().__init__()
        self.c0 = float(c0)

    def forward(self, batch):
        energy = (batch["window"] ** 2).sum(dim=(1, 2))  # [B]
        return {
            "chatter_logit": energy - self.c0,
            "b_over_blim": energy / 100.0,
        }


def _energy_split_batch():
    """Half high-amplitude (label 1) windows, half low-amplitude (label 0)."""
    torch.manual_seed(0)
    L, C = 24, 4
    high = torch.randn(4, L, C) * 3.0  # large energy
    low = torch.randn(4, L, C) * 0.1  # tiny energy
    window = torch.cat([high, low], dim=0)
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # Threshold between the two energy clouds.
    he = (high**2).sum(dim=(1, 2))
    le = (low**2).sum(dim=(1, 2))
    c0 = 0.5 * (float(he.min()) + float(le.max()))
    batch = {"window": window, "tau": torch.full((8,), 5, dtype=torch.long)}
    return batch, labels, c0


# --------------------------------------------------------------------------- #
# 1. select_cases with the energy stub
# --------------------------------------------------------------------------- #
def test_select_cases_picks_confident_chatter_and_stable():
    batch, labels, c0 = _energy_split_batch()
    model = _EnergyStub(c0)

    sel = select_cases(model, batch, labels)
    assert labels[sel["chatter"]] == 1
    assert labels[sel["stable"]] == 0

    probs = predict(model, batch)["chatter_prob"]
    assert probs[sel["chatter"]] > probs[sel["stable"]]
    # The chosen chatter index is the most-confident correct positive.
    pos = np.where(labels == 1)[0]
    assert sel["chatter"] == pos[int(np.argmax(probs[pos]))]
    neg = np.where(labels == 0)[0]
    assert sel["stable"] == neg[int(np.argmin(probs[neg]))]


def test_select_cases_raises_when_a_class_is_absent():
    batch, _, c0 = _energy_split_batch()
    model = _EnergyStub(c0)
    all_pos = np.ones(8, dtype=int)
    try:
        select_cases(model, batch, all_pos)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when label==0 class is absent")


# --------------------------------------------------------------------------- #
# 2. gate_trajectories on a real model
# --------------------------------------------------------------------------- #
def test_gate_trajectories_shapes_and_range():
    model = _real_model()
    L = 24
    window = torch.randn(L, 4)
    traj = gate_trajectories(model, window, tau=5)
    assert traj["z"].shape == (L, model.hidden_size)
    assert traj["r"].shape == (L, model.hidden_size)
    for key in ("z", "r"):
        assert np.isfinite(traj[key]).all()
        assert (traj[key] >= 0.0).all() and (traj[key] <= 1.0).all()


# --------------------------------------------------------------------------- #
# 3. top_firing_edges + explain_case on a real model
# --------------------------------------------------------------------------- #
def test_explain_case_returns_documented_bundle():
    model = _real_model()
    window = torch.randn(24, 4)
    case = explain_case(model, window, tau=5, label=1, top_k=6)

    for key in (
        "chatter_prob",
        "b_over_blim",
        "per_channel_ig",
        "gate_means",
        "top_edges",
        "why",
    ):
        assert key in case

    pc = np.asarray(case["per_channel_ig"])
    assert pc.shape == (4,)
    assert np.isfinite(pc).all()

    assert set(case["gate_means"]) == {"z", "r"}
    assert len(case["top_edges"]) == 6
    edge = case["top_edges"][0]
    for k in ("kan", "out", "in", "block", "name", "importance",
              "visited_lo", "visited_hi"):
        assert k in edge

    why = case["why"]
    assert why["predicted"] in {"chatter", "stable"}
    assert why["top_channel"] in CHANNELS
    assert isinstance(why["leans_on_delay"], bool)
    assert 0.0 <= float(why["chatter_prob"]) <= 1.0


# --------------------------------------------------------------------------- #
# 4. plot_case writes a non-empty PNG
# --------------------------------------------------------------------------- #
def test_plot_case_writes_png(tmp_path):
    model = _real_model()
    window = torch.randn(24, 4)
    case = explain_case(model, window, tau=5, label=0, top_k=5)
    path = tmp_path / "case.png"
    out = plot_case(case, window, str(path))
    assert out == str(path)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


# --------------------------------------------------------------------------- #
# 5. delay_engagement_stats: aggregate the delay-ablation effect over MANY
#    windows (the single-window `leans_on_delay` verdict is n=1 noise).
# --------------------------------------------------------------------------- #
_TRUE_TAU = 8


class _DelayRegen(torch.nn.Module):
    """Regenerative DDE stub: ``logit = sum_t w[:,t,0] * w[:,t-tau,0]``.

    A period-``tau`` tone on channel 0 aligns with its own delayed copy (large
    positive logit -> chatter prob ~ 1); noise does not (logit ~ 0 -> prob ~
    0.5). Zeroing the delayed path (``physics.delay_ablation``'s ``big_tau``)
    sends every logit to 0, so the chatter windows LOSE their probability while
    the stable windows barely move -- exactly the per-class asymmetry
    ``delay_engagement_stats`` must surface.
    """

    def forward(self, batch: dict) -> dict:
        w = batch["window"]
        tau = batch["tau"]
        if isinstance(tau, torch.Tensor):
            tau = int(tau[0])
        _, L, _ = w.shape
        wd = torch.stack([delayed_input(w, t, tau) for t in range(L)], dim=1)
        logit = (w[:, :, 0] * wd[:, :, 0]).sum(dim=1)  # [B]
        return {"chatter_logit": logit, "b_over_blim": logit}


class _NoDelayStub(torch.nn.Module):
    """Control: ``logit = sum_t w[:,t,0]`` -- ignores the delay entirely.

    Zeroing the delayed path changes nothing, so BOTH classes have ~zero
    per-window drop (the delay is not load-bearing).
    """

    def forward(self, batch: dict) -> dict:
        w = batch["window"]
        logit = w[:, :, 0].sum(dim=1)
        return {"chatter_logit": logit, "b_over_blim": logit}


def _delay_batch(n_pos=24, n_neg=24, L=40, seed=0):
    """Periodic (period-``_TRUE_TAU``) positives, channel-0 noise negatives."""
    torch.manual_seed(seed)
    t = torch.arange(L, dtype=torch.float32)
    pos, neg = [], []
    for _ in range(n_pos):
        phase = 2 * np.pi * torch.rand(1)
        ch0 = torch.sin(2 * np.pi * t / _TRUE_TAU + phase)
        pos.append(torch.stack([ch0] + [0.05 * torch.randn(L) for _ in range(3)], dim=-1))
    for _ in range(n_neg):
        ch0 = torch.randn(L)
        neg.append(torch.stack([ch0] + [0.05 * torch.randn(L) for _ in range(3)], dim=-1))
    window = torch.stack(pos + neg, dim=0)  # [B, L, 4]
    B = window.shape[0]
    labels = np.array([1] * n_pos + [0] * n_neg, dtype=int)
    batch = {"window": window, "tau": torch.full((B,), _TRUE_TAU, dtype=torch.long)}
    return batch, labels


def test_delay_engagement_stats_returns_per_class_aggregates_with_cis():
    model = _DelayRegen()
    batch, labels = _delay_batch()
    stats = delay_engagement_stats(model, batch, labels, seed=0)

    # Batch-level ablation numbers (reused from physics.delay_ablation).
    for k in ("auc_on", "auc_off", "auc_drop", "big_tau"):
        assert k in stats
    assert stats["per_window_prob_drop"].shape == (len(labels),)

    assert set(stats["per_class"]) == {"chatter", "stable"}
    for cls in ("chatter", "stable"):
        agg = stats["per_class"][cls]
        for k in ("n", "mean_prob_drop", "ci_low", "ci_high"):
            assert k in agg
        assert agg["n"] > 0
        # A confidence interval is an ordered bracket around the mean.
        assert agg["ci_low"] <= agg["mean_prob_drop"] <= agg["ci_high"]


def test_delay_engagement_chatter_drop_exceeds_stable_when_model_uses_delay():
    model = _DelayRegen()
    batch, labels = _delay_batch()
    stats = delay_engagement_stats(model, batch, labels, seed=0)

    chatter = stats["per_class"]["chatter"]
    stable = stats["per_class"]["stable"]
    # The regenerative stub leans on the delay to flag chatter, so zeroing the
    # delayed input collapses the chatter probability far more than the stable
    # one -- the aggregate makes that asymmetry unambiguous (unlike n=1).
    assert chatter["mean_prob_drop"] > stable["mean_prob_drop"]
    # And the effect is real: the chatter CI clears zero.
    assert chatter["ci_low"] > 0.0


def test_delay_engagement_no_asymmetry_for_delay_free_model():
    model = _NoDelayStub()
    batch, labels = _delay_batch()
    stats = delay_engagement_stats(model, batch, labels, seed=0)
    # The control ignores the delay, so neither class moves when it is zeroed.
    assert abs(stats["per_class"]["chatter"]["mean_prob_drop"]) < 1e-5
    assert abs(stats["per_class"]["stable"]["mean_prob_drop"]) < 1e-5
