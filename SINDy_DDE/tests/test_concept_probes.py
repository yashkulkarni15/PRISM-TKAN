"""Tests for TCAV-style concept probes on the TKAN hidden state.

The probe layer fits a *linear* map from the 64-d encoder hidden state ``h`` to a
named physics quantity (regeneration phase, chatter/tooth-pass separation,
proximity to blim). A concept that is linearly encoded in ``h`` must be
recovered with a high probe score and a direction that aligns with the planted
one; a concept absent from ``h`` must score near chance. The TCAV directional
sensitivity is the dot of the chatter-logit gradient wrt ``h`` with the concept
direction.

Ground truth here is synthetic: we *plant* a known direction in ``h`` so the
probe has an analytically-known answer, and we build pure-tone windows so the
concept-target builders have an analytically-known value.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pri2m.models.tkan_model import TKANModel  # noqa: E402
from pri2m.xai import concept_probes as cp  # noqa: E402


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------------------------------------------------------------- #
# Planted-concept synthesis: h that linearly encodes a known direction
# --------------------------------------------------------------------------- #
def _planted(n=400, hdim=32, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, hdim))
    w = rng.standard_normal(hdim)
    w /= np.linalg.norm(w)
    target = h @ w + noise * rng.standard_normal(n)
    return h, target, w, rng


# --------------------------------------------------------------------------- #
# 1. Probe recovers a planted regression concept
# --------------------------------------------------------------------------- #
def test_probe_recovers_planted_regression_concept():
    h, target, w_true, _ = _planted()
    res = cp.fit_probe(h, target, kind="regression", seed=0)
    assert res["kind"] == "regression"
    assert res["score"] > 0.9  # R^2 on the held-out split
    assert np.asarray(res["direction"]).shape == (h.shape[1],)
    assert _cos(res["direction"], w_true) > 0.95  # planted direction recovered
    assert np.isclose(np.linalg.norm(res["direction"]), 1.0, atol=1e-6)


def test_probe_near_chance_for_absent_regression_concept():
    h, _, _, rng = _planted()
    absent = rng.standard_normal(h.shape[0])  # independent of h
    res = cp.fit_probe(h, absent, kind="regression", seed=0)
    assert res["score"] < 0.25  # R^2 collapses to ~0 for an un-encoded concept


# --------------------------------------------------------------------------- #
# 2. Probe recovers a planted classification concept
# --------------------------------------------------------------------------- #
def test_probe_recovers_planted_classification_concept():
    # Mild noise -> a soft (not perfectly separable) boundary, as a real
    # hidden-state concept would be, so logistic regression converges cleanly.
    h, cont, w_true, _ = _planted(noise=0.05)
    labels = (cont > np.median(cont)).astype(int)
    res = cp.fit_probe(h, labels, kind="classification", seed=0)
    assert res["kind"] == "classification"
    assert res["score"] > 0.85  # accuracy on the held-out split
    assert _cos(res["direction"], w_true) > 0.9


def test_probe_near_chance_for_absent_classification_concept():
    h, _, _, rng = _planted()
    labels = rng.integers(0, 2, size=h.shape[0])  # independent of h
    res = cp.fit_probe(h, labels, kind="classification", seed=0)
    assert res["score"] < 0.7  # ~chance for a balanced un-encoded label


# --------------------------------------------------------------------------- #
# 3. TCAV directional sensitivity
# --------------------------------------------------------------------------- #
def test_tcav_directional_sign():
    rng = np.random.default_rng(1)
    hdim, n = 16, 50
    direction = rng.standard_normal(hdim)
    direction /= np.linalg.norm(direction)
    scales = rng.uniform(0.5, 2.0, size=n)

    aligned = scales[:, None] * direction[None, :]
    r = cp.tcav_score(aligned, direction)
    assert r["mean_sensitivity"] > 0
    assert r["tcav_fraction"] == pytest.approx(1.0)

    r2 = cp.tcav_score(-aligned, direction)
    assert r2["mean_sensitivity"] < 0
    assert r2["tcav_fraction"] == pytest.approx(0.0)

    rand = rng.standard_normal((n, hdim))
    orth = rand - (rand @ direction)[:, None] * direction[None, :]
    r3 = cp.tcav_score(orth, direction)
    assert abs(r3["mean_sensitivity"]) < 1e-6  # orthogonal -> no sensitivity


# --------------------------------------------------------------------------- #
# 4. Concept-target builders (analytically known values)
# --------------------------------------------------------------------------- #
def _tone_window(length, bin_k, n_ch=4, phase=0.0):
    """A ``[L, C]`` window whose channel 0 is a pure cosine at FFT bin ``bin_k``."""
    t = np.arange(length)
    w = np.zeros((length, n_ch), dtype=np.float32)
    w[:, 0] = np.cos(2 * np.pi * bin_k * t / length + phase)
    return w


def test_regeneration_phase_matches_analytic():
    # Bin 8 over L=1024 with tau=32 -> phase advance 2*pi*k*tau/L = 2*pi*(1/4) = pi/2.
    L, k, tau = 1024, 8, 32
    x = _tone_window(L, k)[:, 0]
    phi = cp.regeneration_phase(x, tau)
    assert phi == pytest.approx(np.pi / 2, abs=1e-6)
    assert -np.pi <= phi <= np.pi  # wrapped


def test_chatter_tpf_separation_matches_analytic():
    L, k, fs = 1024, 10, 10_000.0
    f_dom = k * fs / L  # dominant frequency in Hz
    tpf = 200.0
    x = _tone_window(L, k)[:, 0]
    sep = cp.chatter_tpf_separation(x, tpf, fs)
    assert sep == pytest.approx(abs(f_dom - tpf), abs=1e-6)


def test_build_concept_targets_from_batch():
    L, fs = 512, 10_000.0
    tpf = 200.0
    tau = int(round(fs / tpf))  # 50
    windows = np.stack([_tone_window(L, k) for k in (12, 20, 30, 40)], axis=0)
    B = windows.shape[0]
    b_over_blim = np.array([0.3, 0.8, 1.1, 1.6], dtype=np.float32)
    batch = {
        "window": torch.as_tensor(windows),
        "tau": torch.full((B,), tau, dtype=torch.long),
        "tpf": torch.full((B,), tpf, dtype=torch.float32),
        "fs": torch.full((B,), fs, dtype=torch.float32),
        "b_over_blim": torch.as_tensor(b_over_blim),
    }
    targets = cp.build_concept_targets(batch)
    # Circular regenerative phase is probed as sin/cos (the raw wrapped angle is
    # degenerate for a linear probe); separation is on a log1p scale.
    assert set(targets) == {
        "regeneration_phase_sin",
        "regeneration_phase_cos",
        "chatter_tpf_separation",
        "proximity_to_blim",
    }
    for v in targets.values():
        assert np.asarray(v).shape == (B,)
    # proximity_to_blim is exactly the batch's b_over_blim.
    assert np.allclose(targets["proximity_to_blim"], b_over_blim)
    # separation for the k=12 window: log1p(|12*fs/L - tpf|).
    assert targets["chatter_tpf_separation"][0] == pytest.approx(
        np.log1p(abs(12 * fs / L - tpf)), abs=1e-4
    )
    # sin/cos targets stay bounded in [-1, 1] regardless of the phase wrap.
    for comp in ("regeneration_phase_sin", "regeneration_phase_cos"):
        assert np.all(np.abs(targets[comp]) <= 1.0 + 1e-9)


# --------------------------------------------------------------------------- #
# 5. Model plumbing on a real (small) TKAN model
# --------------------------------------------------------------------------- #
def _small_model_and_batch(seed=0, B=6, L=64, C=4, H=8):
    torch.manual_seed(seed)
    model = TKANModel(input_size=C, hidden_size=H, num_layers=1, use_delay=True)
    model.eval()
    window = torch.randn(B, L, C)
    batch = {
        "window": window,
        "tau": torch.full((B,), 8, dtype=torch.long),
        "tpf": torch.full((B,), 200.0, dtype=torch.float32),
        "fs": torch.full((B,), 10_000.0, dtype=torch.float32),
        "b_over_blim": torch.linspace(0.2, 1.5, B),
    }
    return model, batch, H


def test_extract_hidden_states_shape():
    model, batch, H = _small_model_and_batch()
    h = cp.extract_hidden_states(model, batch)
    assert h.shape == (batch["window"].shape[0], H)
    assert np.all(np.isfinite(h))


def test_chatter_logit_grad_wrt_hidden_shape():
    model, batch, H = _small_model_and_batch()
    grad = cp.chatter_logit_grad_wrt_hidden(model, batch)
    assert grad.shape == (batch["window"].shape[0], H)
    assert np.all(np.isfinite(grad))


def test_run_concept_probes_end_to_end():
    model, batch, _ = _small_model_and_batch(B=24, L=64)
    res = cp.run_concept_probes(model, batch, test_size=0.4, seed=0)
    assert set(res["concepts"]) == {
        "regeneration_phase_sin",
        "regeneration_phase_cos",
        "chatter_tpf_separation",
        "proximity_to_blim",
    }
    for name, rec in res["concepts"].items():
        assert "probe" in rec and "tcav" in rec
        assert np.isfinite(rec["probe"]["score"])
        assert np.isfinite(rec["tcav"]["mean_sensitivity"])
        assert 0.0 <= rec["tcav"]["tcav_fraction"] <= 1.0
    # The transform is threaded onto each concept (log1p for separation, else None).
    assert res["concepts"]["chatter_tpf_separation"]["transform"] == "log1p"
    assert res["concepts"]["proximity_to_blim"]["transform"] is None
    assert res["hidden_size"] == model.hidden_size


def test_plot_probe_scores_writes_file(tmp_path):
    model, batch, _ = _small_model_and_batch(B=24, L=64)
    res = cp.run_concept_probes(model, batch, test_size=0.4, seed=0)
    path = tmp_path / "concept_probes.png"
    out = cp.plot_probe_scores(res, str(path))
    assert out == str(path)
    assert path.exists() and path.stat().st_size > 0
