"""Why regeneration_phase probes degenerate: circular target vs linear probe.

Reproduces the mechanism behind the real GL failure (held-out R^2 = -228 for the
``regeneration_phase`` concept). A linear Ridge probe cannot decode a *wrapped*
circular angle even when the underlying phase is fully present in the hidden
state -- probing ``sin``/``cos`` of the (unwrapped) phase recovers it.

Two complementary reproductions:

* ``test_wrapped_phase_probe_is_degenerate_but_sincos_recovers`` -- an idealized,
  well-conditioned encoding (phase present as two clean ``sin``/``cos`` dims):
  ``sin``/``cos`` are decoded at R^2 ~= 1.0 while the wrapped angle lands far
  below. Seed-stable; proves the phase *is* present yet the wrapped target is
  not linearly decodable.
* ``test_wrapped_phase_probe_collapses_negative_in_ill_conditioned_regime`` --
  the ill-conditioned 64-d regime that matches the real pipeline: the ridge
  overfits the training split (train R^2 ~= 0.95) then the wrapped held-out R^2
  collapses *negative* (worse than the mean predictor) -- the -228 failure mode
  itself. Robust across seeds (averaged), while ``sin``/``cos`` stay decodable.

The observed numbers (documented for the report):
    idealized  -> sin=1.0000 cos=1.0000  wrapped=+0.571
    ill-cond.  -> train_wrapped=+0.954   wrapped(test) mean=-0.857 (<0 for 12/12 seeds)
"""
import numpy as np

from pri2m.xai.concept_probes import _wrap_pi, fit_probe


def _hidden_encoding_phase(n=400, H=16, seed=0):
    """Hidden states that PERFECTLY encode a phase angle, as (sin, cos) dims."""
    rng = np.random.default_rng(seed)
    phase_unwrapped = rng.uniform(0, 6 * np.pi, size=n)      # crosses the wrap many times
    h = rng.normal(size=(n, H)) * 0.05
    h[:, 0] = np.sin(phase_unwrapped)
    h[:, 1] = np.cos(phase_unwrapped)
    return h, phase_unwrapped


def _hidden_encoding_phase_illconditioned(n=100, H=64, noise=0.25, spread=6, seed=0):
    """Phase encoded *imperfectly* across many correlated 64-d dims (the real regime).

    Unlike the idealized two-dim encoding, the phase is smeared across ``spread``
    noisy dimensions, so a Ridge probe on the discontinuous wrapped target
    overfits training and collapses on held-out points -- the -228 mechanism.
    """
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 6 * np.pi, size=n)
    h = rng.normal(size=(n, H)) * noise
    a = rng.standard_normal(spread)
    b = rng.standard_normal(spread)
    for j in range(spread):
        h[:, j] += a[j] * np.sin(phase)
        h[:, spread + j] += b[j] * np.cos(phase)
    return h, phase


def test_wrapped_phase_probe_is_degenerate_but_sincos_recovers():
    h, phase = _hidden_encoding_phase()
    wrapped = _wrap_pi(phase)
    bad = fit_probe(h, wrapped, kind="regression", seed=0)
    good_sin = fit_probe(h, np.sin(phase), kind="regression", seed=0)
    good_cos = fit_probe(h, np.cos(phase), kind="regression", seed=0)
    # The mechanism: sin/cos targets are near-perfectly linearly decodable
    # (the phase IS present in h), the wrapped circular angle is not -- this
    # non-decodability is what blows up to R^2 = -228 in the real ill-conditioned
    # 64-d ridge (see the companion test below). In this well-conditioned
    # idealization the collapse shows up as a large decodability gap.
    assert good_sin["score"] > 0.9 and good_cos["score"] > 0.9
    assert bad["score"] < good_sin["score"] - 0.25   # wrapped clearly NOT recoverable
    assert bad["score"] < 0.75


def test_wrapped_phase_probe_collapses_negative_in_ill_conditioned_regime():
    # Faithful reproduction of the -228 held-out collapse: the wrapped-angle
    # probe overfits training then scores WORSE than the mean predictor
    # (R^2 < 0) on the held-out split, while sin/cos stay decodable. Averaged
    # over seeds so the qualitative collapse (not a single lucky seed) is asserted.
    train_wrapped, test_wrapped, sin_scores, cos_scores = [], [], [], []
    for seed in range(12):
        h, phase = _hidden_encoding_phase_illconditioned(seed=seed)
        bad = fit_probe(h, _wrap_pi(phase), kind="regression", seed=0)
        gs = fit_probe(h, np.sin(phase), kind="regression", seed=0)
        gc = fit_probe(h, np.cos(phase), kind="regression", seed=0)
        train_wrapped.append(bad["train_score"])
        test_wrapped.append(bad["score"])
        sin_scores.append(gs["score"])
        cos_scores.append(gc["score"])
    train_wrapped = np.array(train_wrapped)
    test_wrapped = np.array(test_wrapped)
    # Overfit on train (the ridge fits the discontinuous target in-sample)...
    assert train_wrapped.mean() > 0.85
    # ...but the held-out wrapped R^2 collapses below zero for EVERY seed.
    assert np.all(test_wrapped < 0.0)
    assert test_wrapped.mean() < -0.2
    # sin/cos of the same phase remain far more decodable than the wrapped angle.
    assert np.mean(sin_scores) > 0.6 and np.mean(cos_scores) > 0.6
    assert np.mean(sin_scores) - test_wrapped.mean() > 0.5


def test_run_concept_probes_emits_sincos_phase(synth_model_batch=None):
    # AFTER the fix: run_concept_probes must expose circular-safe phase probes.
    import torch
    from pri2m.models.tkan_model import TKANModel
    from pri2m.xai.concept_probes import run_concept_probes

    torch.manual_seed(0)
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3).eval()
    g = torch.Generator().manual_seed(1)
    B, L = 12, 64
    batch = {
        "window": torch.randn(B, L, 4, generator=g),
        "tau": torch.full((B,), 5, dtype=torch.long),
        "tpf": torch.full((B,), 200.0),
        "fs": torch.full((B,), 10_000.0),
        "b_over_blim": torch.rand(B, generator=g),
        "label": torch.tensor([0.0, 1.0] * (B // 2)),
        "omega": torch.full((B,), 5000.0),
    }
    out = run_concept_probes(model, batch, test_size=0.4, seed=0)
    names = set(out["concepts"])
    assert {"regeneration_phase_sin", "regeneration_phase_cos"} <= names
    assert "regeneration_phase" not in names                   # raw wrapped angle retired
    sep = out["concepts"]["chatter_tpf_separation"]
    assert sep.get("transform") == "log1p"
