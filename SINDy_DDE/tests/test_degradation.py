"""Graded fuzziness suite: input/model fuzzers + physics-reliance probes."""
import copy

import numpy as np
import pytest
import torch

from pri2m.models.kan import KANLinear
from pri2m.models.tkan_model import TKANModel
from pri2m.xai import degradation as dg
from pri2m.xai.kan_native import _iter_kanlinear


def _model(seed=0):
    torch.manual_seed(seed)
    return TKANModel(input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3).eval()


def _batch(B=8, L=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(L, dtype=torch.float32)
    # deterministic multi-tone signal + small noise so smoothing has HF energy to remove
    base = torch.sin(2 * np.pi * 0.45 * t) + 0.5 * torch.sin(2 * np.pi * 0.05 * t)
    window = base.view(1, L, 1).repeat(B, 1, 4) + 0.1 * torch.randn(B, L, 4, generator=g)
    labels = np.array([0, 1] * (B // 2))
    return {
        "window": window,
        "tau": torch.full((B,), 5, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.float32),
        "b_over_blim": torch.rand(B, generator=g),
        "omega": torch.full((B,), 5000.0),
        "tpf": torch.full((B,), 200.0),
        "fs": torch.full((B,), 10_000.0),
    }, labels


# ---- input-side fuzzers -------------------------------------------------- #

def test_input_noise_level0_identity_and_seeded():
    batch, _ = _batch()
    out0 = dg.fuzz_input_noise(batch, 0.0, seed=0)
    assert torch.equal(out0["window"], batch["window"])
    a = dg.fuzz_input_noise(batch, 0.5, seed=1)["window"]
    b = dg.fuzz_input_noise(batch, 0.5, seed=1)["window"]
    c = dg.fuzz_input_noise(batch, 0.5, seed=2)["window"]
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert not torch.equal(a, batch["window"])          # actually fuzzed
    assert torch.equal(batch["window"], _batch()[0]["window"])  # input not mutated


def test_input_smooth_removes_hf_energy_monotonically():
    batch, _ = _batch()

    def hf_energy(w):
        spec = torch.fft.rfft(w - w.mean(dim=1, keepdim=True), dim=1).abs()
        return float(spec[:, spec.shape[1] // 2 :, :].sum())

    e = [hf_energy(dg.fuzz_input_smooth(batch, w)["window"]) for w in (1, 4, 16)]
    assert torch.equal(dg.fuzz_input_smooth(batch, 1)["window"], batch["window"])
    assert e[0] > e[1] > e[2]
    assert dg.fuzz_input_smooth(batch, 4)["window"].shape == batch["window"].shape


def test_tau_jitter_identity_floor_and_realized():
    batch, _ = _batch()
    out0, info0 = dg.fuzz_tau_jitter(batch, 0.0, seed=0)
    assert torch.equal(out0["tau"], batch["tau"]) and info0["realized_abs_dtau_mean"] == 0.0
    out, info = dg.fuzz_tau_jitter(batch, 0.5, seed=0)
    assert (out["tau"] >= 1).all()
    assert info["realized_abs_dtau_mean"] > 0
    assert not torch.equal(out["tau"], batch["tau"])
    signs = (out["tau"] - batch["tau"]).sign()
    assert (signs == 1).any() and (signs == -1).any()   # mixed directions


# ---- model-side fuzzers -------------------------------------------------- #

def test_spline_coarsen_selffit_and_forward():
    model, (batch, _) = _model(), _batch()
    same, _, info5 = dg.FUZZERS["spline_coarsen"].apply(model, batch, 5, 0)
    for r2 in info5["refit_r2"].values():
        assert r2 > 0.999                               # same-grid refit ≈ exact
    with torch.no_grad():
        y0, y5 = model(batch)["chatter_logit"], same(batch)["chatter_logit"]
    assert torch.allclose(y0, y5, atol=1e-3)
    coarse, _, info2 = dg.FUZZERS["spline_coarsen"].apply(model, batch, 2, 0)
    for _, kan in _iter_kanlinear(coarse):
        assert kan.n_coeff == 2 + kan.spline_order
    with torch.no_grad():
        coarse(batch)                                   # coarse model still runs
    # original untouched
    for (_, a), (_, b) in zip(
        _iter_kanlinear(model),
        _iter_kanlinear(_model()),
    ):
        assert torch.equal(a.spline_weight, b.spline_weight)


def test_weight_noise_identity_and_seeded():
    model, (batch, _) = _model(), _batch()
    m0, _, _ = dg.FUZZERS["weight_noise"].apply(model, batch, 0.0, 0)
    with torch.no_grad():
        assert torch.allclose(model(batch)["chatter_logit"], m0(batch)["chatter_logit"])
    m1, _, _ = dg.FUZZERS["weight_noise"].apply(model, batch, 0.2, 1)
    m1b, _, _ = dg.FUZZERS["weight_noise"].apply(model, batch, 0.2, 1)
    with torch.no_grad():
        assert torch.allclose(m1(batch)["chatter_logit"], m1b(batch)["chatter_logit"])
        assert not torch.allclose(model(batch)["chatter_logit"], m1(batch)["chatter_logit"])


# ---- probes / loop / breakdown ------------------------------------------ #

def test_measure_probes_keys_and_level0_selfcheck():
    model, (batch, labels) = _model(), _batch()
    base = dg.measure_probes(model, batch, labels)
    for k in ("auc", "auc_delay_off", "delay_drop", "mae", "mae_delay_off"):
        assert np.isfinite(base[k])
    res = dg.run_fuzzer(model, batch, labels, dg.FUZZERS["input_noise"], seeds=(0,))
    lvl0 = res["results"][0]
    assert lvl0["level"] == 0.0
    assert abs(lvl0["mean"]["auc"] - base["auc"]) < 1e-6
    assert abs(lvl0["mean"]["delay_drop"] - base["delay_drop"]) < 1e-6
    assert set(res["breakdown"]) == {"auc_below_075", "drop_below_005", "auc_at_chance"}


def test_run_fuzzer_stochastic_mean_sd():
    model, (batch, labels) = _model(), _batch()
    res = dg.run_fuzzer(model, batch, labels, dg.FUZZERS["weight_noise"], seeds=(0, 1, 2))
    big = [r for r in res["results"] if r["level"] == 0.5][0]
    assert len(big["seeds"]) == 3
    assert big["sd"]["auc"] >= 0.0


def test_breakdown_points_on_crafted_curve():
    levels = (0.0, 0.1, 0.2, 0.5)
    bp = dg.breakdown_points(levels, mean_auc=(0.95, 0.80, 0.70, 0.50), mean_drop=(0.3, 0.04, 0.02, 0.0))
    assert bp == {"auc_below_075": 0.2, "drop_below_005": 0.1, "auc_at_chance": 0.5}
    none = dg.breakdown_points(levels, (0.9, 0.9, 0.9, 0.9), (0.3, 0.3, 0.3, 0.3))
    assert none == {"auc_below_075": None, "drop_below_005": None, "auc_at_chance": None}


def test_plots_write_files(tmp_path):
    model, (batch, labels) = _model(), _batch()
    res = dg.run_fuzzer(model, batch, labels, dg.FUZZERS["tau_jitter"], seeds=(0,))
    base = dg.measure_probes(model, batch, labels)
    p1 = dg.plot_degradation_panel(res, base, tmp_path / "panel.png")
    p2 = dg.plot_fuzz_dashboard([res], base, tmp_path / "dash.png")
    assert (tmp_path / "panel.png").is_file() and (tmp_path / "dash.png").is_file()
    assert p1.endswith("panel.png") and p2.endswith("dash.png")
