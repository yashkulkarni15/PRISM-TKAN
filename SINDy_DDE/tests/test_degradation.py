"""Graded fuzziness suite: input/model fuzzers + physics-reliance probes."""
import copy

import numpy as np
import pytest
import torch

from pri2m.models.kan import KANLinear
from pri2m.models.tkan_model import TKANModel
from pri2m.xai import degradation as dg
from pri2m.xai.kan_native import _iter_kanlinear
from pri2m.xai.degradation import (
    normalized_reliance, reliance_breakdown, spectral_floor_report,
)


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


def test_lag_tau_shuffle_is_a_permutation_of_samples():
    # (B, L, C) -- B=3 windows of length 128 with 4 channels
    batch, _ = _batch(B=3, L=128, seed=0)
    batch["tau"] = torch.full((3,), 16, dtype=torch.long)
    new, info = dg.fuzz_lag_tau_shuffle(batch, frac=1.0, seed=0)
    for b in range(3):
        for c in range(4):
            np.testing.assert_allclose(
                np.sort(new["window"][b, :, c].numpy()),
                np.sort(batch["window"][b, :, c].numpy()),
            )
    assert info["frac_blocks_permuted"] > 0


def test_lag_tau_shuffle_frac_zero_is_identity():
    batch, _ = _batch(B=2, L=64, seed=1)
    new, info = dg.fuzz_lag_tau_shuffle(batch, frac=0.0, seed=0)
    np.testing.assert_allclose(new["window"].numpy(), batch["window"].numpy())
    assert info["frac_blocks_permuted"] == 0.0


def test_lag_tau_shuffle_does_not_mutate_the_original_batch():
    batch, _ = _batch(B=2, L=64, seed=2)
    before = batch["window"].clone()
    dg.fuzz_lag_tau_shuffle(batch, frac=1.0, seed=0)
    np.testing.assert_allclose(batch["window"].numpy(), before.numpy())


def test_lag_tau_shuffle_actually_reorders_at_frac_one():
    batch, _ = _batch(B=2, L=64, seed=3)
    new, _ = dg.fuzz_lag_tau_shuffle(batch, frac=1.0, seed=0)
    assert not np.allclose(new["window"].numpy(), batch["window"].numpy())


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


# ---- reliance metrics ---------------------------------------------------- #

def test_normalized_reliance_divides_by_available_headroom():
    assert normalized_reliance(1.0, 0.25) == pytest.approx(0.5)
    assert normalized_reliance(0.75, 0.125) == pytest.approx(0.5)


def test_normalized_reliance_is_none_at_or_near_chance():
    assert normalized_reliance(0.5, 0.0) is None
    assert normalized_reliance(0.505, 0.01) is None   # inside the 0.51 guard
    assert normalized_reliance(0.4, 0.0) is None


def test_normalized_reliance_reproduces_the_fold2_noise_cliff():
    # The finding that motivated the metric: flat to sigma=0.5, cliff after.
    auc = [0.994, 0.917, 0.867, 0.756]
    drop = [0.266, 0.226, 0.126, 0.008]
    norm = [normalized_reliance(a, d) for a, d in zip(auc, drop)]
    assert norm[0] == pytest.approx(0.539, abs=0.01)
    assert norm[1] == pytest.approx(0.542, abs=0.01)   # still flat
    assert norm[3] == pytest.approx(0.030, abs=0.01)   # collapsed


def test_reliance_breakdown_finds_half_life_and_floor():
    levels = [0.0, 1.0, 2.0, 3.0]
    norm = [0.50, 0.40, 0.20, 0.02]
    out = reliance_breakdown(levels, norm)
    assert out["reliance_half_life"] == 2.0    # first <= 0.25
    assert out["reliance_floor"] == 3.0        # first <= 0.05


def test_reliance_breakdown_returns_none_when_never_crossed():
    out = reliance_breakdown([0.0, 1.0], [0.5, 0.49])
    assert out["reliance_half_life"] is None
    assert out["reliance_floor"] is None


def test_reliance_breakdown_ignores_none_entries():
    out = reliance_breakdown([0.0, 1.0, 2.0], [0.5, None, 0.1])
    assert out["reliance_half_life"] == 2.0


def test_spectral_floor_report_flags_convergence_to_the_delay_off_auc():
    out = spectral_floor_report([0.99, 0.90, 0.86, 0.855], baseline_auc_delay_off=0.85)
    assert out["spectral_floor_auc"] == pytest.approx(0.85)
    assert out["asymptote_auc"] == pytest.approx(0.8575)
    assert out["converged"] is True
    assert out["asymptote_gap"] == pytest.approx(0.0075, abs=1e-6)


def test_spectral_floor_report_reports_not_converged_when_still_falling():
    out = spectral_floor_report([0.99, 0.90, 0.80, 0.70], baseline_auc_delay_off=0.5)
    assert out["converged"] is False


# ---- Task 6: new stressors registered + extended ranges ------------------ #

def test_new_stressors_are_registered_as_input_fuzzers():
    from pri2m.xai.degradation import FUZZERS
    for name in ("phase_randomize", "magnitude_randomize", "lag_tau_shuffle"):
        assert name in FUZZERS, f"{name} not registered"
        assert FUZZERS[name].kind == "input"
        assert FUZZERS[name].levels[0] == 0.0, "level 0 must be the identity"


def test_existing_levels_are_appended_to_never_rewritten():
    # Update 10 numbers must stay reproducible: every previously measured level
    # must still be present, at the same value, in the same order.
    from pri2m.xai.degradation import FUZZERS
    assert FUZZERS["input_noise"].levels[:7] == (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    assert FUZZERS["input_noise"].levels == (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0)
    assert FUZZERS["tau_jitter"].levels[:5] == (0.0, 0.05, 0.10, 0.25, 0.50)
    assert FUZZERS["tau_jitter"].levels == (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.9)
    assert FUZZERS["input_smooth"].levels == (1, 2, 4, 8, 16, 32, 64, 128)
    assert FUZZERS["weight_noise"].levels == (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    assert FUZZERS["spline_coarsen"].levels == (5, 4, 3, 2, 1)


def test_tau_jitter_never_reaches_the_degenerate_clamp():
    # At frac >= 1.0 the implementation clamps ~half the windows to tau=1
    # rather than jittering them, which is a different experiment.
    from pri2m.xai.degradation import FUZZERS
    assert max(FUZZERS["tau_jitter"].levels) < 1.0


def test_phase_randomize_fuzzer_leaves_the_original_batch_untouched():
    import torch
    from pri2m.xai.degradation import FUZZERS
    rng = np.random.default_rng(3)
    w = torch.as_tensor(rng.standard_normal((2, 4, 64)), dtype=torch.float32)
    before = w.clone()
    batch = {"window": w, "tau": torch.as_tensor([8, 8], dtype=torch.long)}
    _, new, _ = FUZZERS["phase_randomize"].apply(None, batch, 1.0, 0)
    np.testing.assert_allclose(w.numpy(), before.numpy())
    assert not np.allclose(new["window"].numpy(), before.numpy())


# ---- Task 7: matched-energy band-notch contrast --------------------------- #

def test_band_notch_contrast_matches_energy_between_conditions():
    """Matched removed energy is what makes the contrast fair -- assert it directly.

    Uses L=1024, the real window length, because the match is only achievable at
    realistic spectral resolution. Energy removed is a *step* function of
    halfwidth (it moves only when a band edge crosses an FFT bin), so at L=64 the
    spectrum has just 33 bins and the achievable fractions are far too coarse for
    any pair of halfwidths to agree to 1%. Measured floors at L=64 were 37-100%
    relative error no matter how the halfwidths were chosen. That coarse case is
    covered by test_band_notch_contrast_flags_an_unmatchable_contrast instead.
    """
    model, (batch, labels) = _model(), _batch(B=8, L=1024)
    out = dg.run_band_notch_contrast(
        model, batch, labels, fs=10_000.0, fn=950.0, tpf=200.0, halfwidth_hz=200.0
    )
    assert set(out) >= {"chatter", "tooth", "energy_match"}
    em = out["energy_match"]
    assert abs(em["chatter_frac"] - em["tooth_frac"]) <= 0.01 * em["chatter_frac"] + 1e-3
    assert em["matched"] is True
    for cond in ("chatter", "tooth"):
        assert "auc" in out[cond] and "delay_drop" in out[cond]


def test_band_notch_contrast_flags_an_unmatchable_contrast():
    """A contrast that could not equalise the energies must SAY so.

    Silence here is the dangerous failure: unequal deletions read as a physics
    result, when they only measure which band held more energy.
    """
    model, (batch, labels) = _model(), _batch(B=8, L=64)
    em = dg.run_band_notch_contrast(
        model, batch, labels, fs=10_000.0, fn=950.0, tpf=200.0, halfwidth_hz=200.0
    )["energy_match"]
    assert em["matched"] is False
    assert em["max_rel_err"] > 0.01


def test_band_notch_chatter_band_never_reaches_a_tooth_harmonic():
    """The search may widen the chatter band, but never onto a tooth harmonic.

    A chatter notch that also deletes the tooth-pass harmonics removes both
    conditions' evidence at once, so the comparison stops meaning anything. Given
    a search span wide enough to cross them, the guard must still hold.
    """
    model, (batch, labels) = _model(), _batch(B=8, L=1024)
    em = dg.run_band_notch_contrast(
        model, batch, labels, fs=10_000.0, fn=950.0, tpf=200.0,
        halfwidth_hz=200.0, search_span=(0.25, 10.0),
    )["energy_match"]
    nearest_harmonic = min(abs(950.0 - c) for c in em["tooth_centers_hz"])
    assert em["chatter_halfwidth_hz"] < nearest_harmonic


def test_band_notch_contrast_does_not_mutate_the_original_batch():
    model, (batch, labels) = _model(), _batch(B=8, L=64)
    before = batch["window"].clone()
    dg.run_band_notch_contrast(
        model, batch, labels, fs=10_000.0, fn=950.0, tpf=200.0, halfwidth_hz=200.0
    )
    np.testing.assert_allclose(batch["window"].numpy(), before.numpy())


# ---- per-window frequencies (defect 1) ------------------------------------ #
# The 2026-07-29 five-fold cluster run notched every window at the BATCH MEAN
# fn/fs/tpf. Every window has its own chatter frequency, sampling rate and
# tooth-pass frequency, so the mean aims the band away from the actual chatter
# peak for nearly every window and the experiment deletes empty spectrum. The
# function's own docstring already argues that averaging HALFWIDTHS across
# windows is wrong (energy removed is a step function of halfwidth); averaging
# the CENTRES is the same mistake one level up.

def _tone_batch(fn_per_window, fs_per_window, L=1024, seed=0, noise=0.05):
    """Batch whose window ``b`` carries a dominant tone at its OWN ``fn``.

    The tone is placed at an exact FFT bin of that window's own ``fs`` so the
    removable energy is unambiguous: a band aimed at the window's own fn deletes
    nearly all of it, a band aimed anywhere else deletes almost none.
    """
    fn_arr = np.atleast_1d(np.asarray(fn_per_window, dtype=float))
    fs_arr = np.array(
        np.broadcast_to(np.atleast_1d(np.asarray(fs_per_window, dtype=float)), fn_arr.shape)
    )
    g = torch.Generator().manual_seed(seed)
    n = np.arange(L, dtype=float)
    tones = np.stack([np.sin(2 * np.pi * f / s * n) for f, s in zip(fn_arr, fs_arr)])
    win = torch.as_tensor(tones, dtype=torch.float32).unsqueeze(-1).repeat(1, 1, 4)
    win = win + noise * torch.randn(win.shape, generator=g)
    n_b = win.shape[0]
    labels = np.array([b % 2 for b in range(n_b)])
    batch = {
        "window": win,
        "tau": torch.full((n_b,), 5, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.float32),
        "b_over_blim": torch.rand(n_b, generator=g),
        "omega": torch.full((n_b,), 5000.0),
        "tpf": torch.full((n_b,), 200.0),
        "fn": torch.as_tensor(fn_arr, dtype=torch.float32),
        "fs": torch.as_tensor(fs_arr, dtype=torch.float32),
    }
    return batch, labels


def test_band_notch_aims_at_each_windows_own_chatter_frequency():
    """Per-window fn: EVERY window's notch must land on that window's own tone.

    Two windows an octave apart; the batch mean of their fn (1400 Hz) is a
    frequency at which neither window has any chatter energy at all. Asserting
    on the per-window minimum -- not the mean -- is the point: a mean removed
    fraction can look healthy while half the batch was never notched.
    """
    model = _model()
    batch, labels = _tone_batch([800.0, 2000.0], 10_240.0)
    em = dg.run_band_notch_contrast(
        model, batch, labels,
        fs=batch["fs"].numpy(), fn=batch["fn"].numpy(), tpf=200.0,
        halfwidth_hz=50.0,
    )["energy_match"]
    assert em["chatter_frac_min"] > 0.9, "a window's band missed its own chatter tone"
    assert em["fn_hz_sd"] > 0, "the per-window spread of fn must be recorded"


def test_band_notch_accepts_per_window_sampling_rate_and_tooth_frequency():
    """fs and tpf are per-window too, and the tooth-harmonic guard follows them.

    fs sets the frequency-to-bin map, so a batch-mean fs mis-aims the band even
    when fn is right; tpf sets the tooth centres ``tpf*(k+1)``, which are
    therefore per-window, and with them the hard constraint that the chatter
    band must never reach a tooth harmonic.
    """
    model = _model()
    batch, labels = _tone_batch([800.0, 800.0], [10_240.0, 20_480.0])
    batch["tpf"] = torch.tensor([200.0, 250.0])
    em = dg.run_band_notch_contrast(
        model, batch, labels,
        fs=batch["fs"].numpy(), fn=batch["fn"].numpy(), tpf=batch["tpf"].numpy(),
        halfwidth_hz=20.0,
    )["energy_match"]
    assert em["chatter_frac_min"] > 0.9, "per-window fs was not honoured"
    # window 1's own nearest harmonic is 750 Hz, only 50 Hz from its fn=800; the
    # batch-mean tpf (225) would have claimed 125 Hz of headroom there.
    assert em["chatter_tooth_margin_hz"] > 0
    np.testing.assert_allclose(em["tooth_centers_hz"], [225.0, 450.0, 675.0])


def test_band_notch_scalar_and_broadcast_frequencies_agree_exactly():
    """Scalars must keep working: a scalar and its length-B broadcast are the same run."""
    model, (batch, labels) = _model(), _batch(B=4, L=1024)
    kw = dict(fs=10_000.0, tpf=200.0, halfwidth_hz=200.0)
    scalar = dg.run_band_notch_contrast(model, batch, labels, fn=950.0, **kw)["energy_match"]
    vector = dg.run_band_notch_contrast(
        model, batch, labels, fn=np.full(4, 950.0), **kw
    )["energy_match"]
    for key in ("chatter_frac", "tooth_frac", "chatter_halfwidth_hz",
                "tooth_halfwidth_hz", "max_rel_err"):
        assert scalar[key] == vector[key], key
    assert scalar["matched"] == vector["matched"]


def test_band_notch_rejects_a_frequency_array_of_the_wrong_length():
    """A length mismatch is a silent mis-pairing of windows to frequencies."""
    model, (batch, labels) = _model(), _batch(B=4, L=256)
    with pytest.raises(ValueError, match="fn"):
        dg.run_band_notch_contrast(
            model, batch, labels, fs=10_000.0, fn=np.array([950.0, 960.0]), tpf=200.0
        )


# ---- unsolved windows are not measurements (defect 2) --------------------- #
# When every candidate halfwidth is narrower than one FFT bin the chatter notch
# removes nothing, the solve loop never records a solution, and the window used
# to fall through to a TOOTH halfwidth of 0.0 -- a notch that filters nothing.
# Its relative error is then |0 - ch_frac|/ch_frac = exactly 1.0, and because
# the reported error is a MAX over windows, one such window pinned the whole
# five-fold run at 1.0. Same failure as the tr_peak_lag search floor: a sentinel
# reported as data.

def test_band_notch_reports_unsolved_windows_instead_of_a_fake_full_mismatch():
    """No candidate halfwidth reaches an FFT bin -> say "unsolved", not "100% off".

    L=64 at fs=10 kHz has 156.25 Hz bins and the nearest bin to fn sits 12.5 Hz
    away, so every candidate in this search span (2-10 Hz) notches nothing.
    Before the fix this returned max_rel_err == 1.0, indistinguishable from a
    genuine 100% energy mismatch.
    """
    model, (batch, labels) = _model(), _batch(B=4, L=64)
    em = dg.run_band_notch_contrast(
        model, batch, labels, fs=10_000.0, fn=950.0, tpf=200.0,
        halfwidth_hz=200.0, search_span=(0.01, 0.05),
    )["energy_match"]
    assert em["n_windows"] == 4
    assert em["n_unsolved"] == 4
    assert em["n_solved"] == 0
    assert em["unsolved_fraction"] == 1.0
    assert em["matched"] is False
    assert em["max_rel_err"] is None, "an unsolved window is not a measured error"
    assert em["rel_err_p90"] is None


def test_band_notch_unsolved_window_does_not_pin_the_reported_error():
    """One unsolved window must not become the reported error of the whole batch.

    Window 0 has 350 Hz of headroom to its nearest tooth harmonic and solves.
    Window 1's third harmonic (960 Hz) sits 10 Hz from its chatter frequency, so
    every safe halfwidth is far narrower than the 156.25 Hz FFT bin and nothing
    can be solved there -- a realistic failure, not a contrived one.
    ``match_tol`` is deliberately loose so the ONLY thing that can set
    ``matched`` False is the unsolved window itself.
    """
    model, (batch, labels) = _model(), _batch(B=2, L=64)
    kw = dict(fs=10_000.0, fn=950.0, tpf=np.array([200.0, 320.0]),
              halfwidth_hz=200.0, match_tol=1.0)
    em = dg.run_band_notch_contrast(model, batch, labels, **kw)["energy_match"]
    assert em["n_unsolved"] == 1 and em["n_solved"] == 1
    assert em["unsolved_fraction"] == 0.5
    assert em["max_rel_err"] is not None and em["max_rel_err"] < 1.0
    assert np.isfinite(em["max_rel_err"])
    assert em["rel_err_p90"] == em["max_rel_err"]      # one solved window
    assert em["matched"] is False, "an unsolved window must veto the match"
    # The old fallback used the NOMINAL halfwidth (200 Hz) for an unsolved
    # window, which on window 1 sits straight on top of its 960 Hz harmonic.
    assert em["chatter_tooth_margin_hz"] > 0

    # ...and the veto threshold is a named parameter, not a magic constant.
    loose = dg.run_band_notch_contrast(
        model, batch, labels, max_unsolved_frac=1.0, **kw
    )["energy_match"]
    assert loose["max_unsolved_frac"] == 1.0
    assert loose["matched"] is True


def test_band_notch_vanishing_chatter_frac_is_degenerate_not_solved():
    """A window that removes almost nothing must not pin `max_rel_err`.

    The GL run of 2026-07-30 returned `max_rel_err` around 20301 on fold 0 while
    `rel_err_p90` was 0.975. That spread is one defect, not two: the relative
    error is |tooth - chatter| / chatter, so a window whose chatter notch removes
    a vanishing fraction divides by ~0 and emits an astronomical "error" that is
    really a failure to notch anything meaningful. The `> 0` test admits it,
    because a single near-empty FFT bin is still strictly positive.

    Same family as the `tr_peak_lag` search floor and the original zero-width
    tooth notch: a degenerate value passing a liveness check and poisoning a max.
    """
    model, (batch, labels) = _model(), _batch(B=2, L=64)
    kw = dict(fs=10_000.0, fn=950.0, tpf=np.array([200.0, 320.0]),
              halfwidth_hz=200.0, match_tol=1.0)

    # A floor above every window's removed fraction must leave nothing solved,
    # and the error fields must then be None rather than a number.
    none_left = dg.run_band_notch_contrast(
        model, batch, labels, min_chatter_frac=1.0, **kw
    )["energy_match"]
    assert none_left["n_solved"] == 0
    assert none_left["max_rel_err"] is None, "no solved window => no error value"
    assert none_left["rel_err_p90"] is None
    assert none_left["matched"] is False
    assert none_left["n_degenerate_chatter"] >= 1

    # The floor is a named parameter with a reported value, not a magic constant,
    # and at 0 it cannot exclude anything, so the default must be recoverable.
    off = dg.run_band_notch_contrast(
        model, batch, labels, min_chatter_frac=0.0, **kw
    )["energy_match"]
    assert off["min_chatter_frac"] == 0.0
    assert off["n_degenerate_chatter"] == 0
    assert off["n_solved"] >= 1
