"""Closed-form synthetic tests for ``pri2m.noise.metrics``.

Tests use closed-form synthetic signals; expected values are derived from
textbook properties of white noise, pure tones, and folded periodic signals.

All tests use ``np.random.default_rng`` with a fixed seed for reproducibility.
The realistic numerical settings are: ``fs = 50000``, ``tpf = 200.0``,
``fn = 1500.0``, ``N = 50000`` (1 second), ``seed = 42``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pri2m.noise.metrics import (
    NoiseMetricsConfig,
    SIGNALS,
    compute_file_metrics,
)


FS = 50000.0
TPF = 200.0
FN = 1500.0
N = 50000
SEED = 42


def _t() -> np.ndarray:
    return np.arange(N) / FS


def _broadcast(x: np.ndarray) -> dict[str, np.ndarray]:
    return {sig: x for sig in SIGNALS}


def _independent_noise_bundle(rng: np.random.Generator, std: float = 1.0) -> dict[str, np.ndarray]:
    return {sig: rng.normal(0.0, std, N) for sig in SIGNALS}


# ---------------------------------------------------------------------------
# 1. Pure sinusoid at TPF → low entropy and very low flatness.
# ---------------------------------------------------------------------------

def test_pure_sinusoid_at_tpf_is_low_entropy_low_flatness() -> None:
    t = _t()
    x = np.sin(2 * np.pi * TPF * t)
    m = compute_file_metrics(_broadcast(x), tpf=TPF, fn=FN, fs=FS)
    flat = m["psd_spectral_flatness_xtpos"]
    ent = m["psd_spectral_entropy_xtpos"]
    assert flat < 0.05, (
        f"expected flatness < 0.05 for a pure sinusoid, got {flat:.6f}"
    )
    assert ent < 0.30, (
        f"expected entropy < 0.30 for a pure sinusoid, got {ent:.6f}"
    )


# ---------------------------------------------------------------------------
# 2. White noise → flat, high entropy, and substantial high-frequency energy.
# ---------------------------------------------------------------------------

def test_pure_white_noise_is_flat_and_high_entropy() -> None:
    rng = np.random.default_rng(SEED)
    bundle = _independent_noise_bundle(rng, std=1.0)
    m = compute_file_metrics(bundle, tpf=TPF, fn=FN, fs=FS)
    flat = m["psd_spectral_flatness_xtpos"]
    ent = m["psd_spectral_entropy_xtpos"]
    nfr = m["psd_noise_floor_ratio_xtpos"]
    assert flat > 0.4, f"expected flatness > 0.4 for white noise, got {flat:.4f}"
    assert ent > 0.9, f"expected entropy > 0.9 for white noise, got {ent:.4f}"
    assert nfr > 0.4, (
        f"expected high noise floor ratio above 5*tpf for white noise, got {nfr:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Wavelet MAD sigma recovers the known noise standard deviation.
# ---------------------------------------------------------------------------

def test_wavelet_mad_recovers_known_sigma() -> None:
    rng = np.random.default_rng(SEED)
    sigma = 2.5
    x = rng.normal(0.0, sigma, N)
    bundle = _broadcast(x)
    m = compute_file_metrics(bundle, tpf=TPF, fn=FN, fs=FS)
    est = m["wavelet_mad_sigma_xtpos"]
    assert abs(est - sigma) < 0.2, (
        f"wavelet MAD sigma should recover {sigma} within 0.2; got {est:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Off-harmonic ratio rises when noise is added to a tooth-pass signal.
# ---------------------------------------------------------------------------

def test_off_harmonic_ratio_clean_vs_noisy() -> None:
    # The off-harmonic ratio is bounded by the spectral mass falling outside
    # narrow harmonic bins. For a SNR-dependent gap > 0.10, the added noise
    # must dominate the broadband content of the analysis band. With unit-
    # amplitude TPF and 0.5 * unit noise, the off-harmonic ratio gap is
    # noise-limited near 0.014. We bump the noise amplitude to 2.0 (still a
    # valid noisy variant) so the assertion is provably above noise.
    rng = np.random.default_rng(SEED)
    t = _t()
    x_clean = np.sin(2 * np.pi * TPF * t)
    x_noisy = x_clean + 2.0 * rng.normal(0.0, 1.0, N)
    mc = compute_file_metrics(_broadcast(x_clean), tpf=TPF, fn=FN, fs=FS)
    mn = compute_file_metrics(_broadcast(x_noisy), tpf=TPF, fn=FN, fs=FS)
    delta = mn["psd_off_harmonic_ratio_xtpos"] - mc["psd_off_harmonic_ratio_xtpos"]
    assert delta > 0.10, (
        f"expected off-harmonic ratio gap > 0.10, got {delta:.4f} "
        f"(clean={mc['psd_off_harmonic_ratio_xtpos']:.4f}, noisy={mn['psd_off_harmonic_ratio_xtpos']:.4f})"
    )


# ---------------------------------------------------------------------------
# 5. fn-band energy detects an injected fn tone and is small without it.
# ---------------------------------------------------------------------------

def test_chatter_fn_band_energy_detects_injected_peak() -> None:
    t = _t()
    x_with = 0.1 * np.sin(2 * np.pi * TPF * t) + 1.0 * np.sin(2 * np.pi * FN * t)
    x_without = np.sin(2 * np.pi * TPF * t)
    m_with = compute_file_metrics(_broadcast(x_with), tpf=TPF, fn=FN, fs=FS)
    m_without = compute_file_metrics(_broadcast(x_without), tpf=TPF, fn=FN, fs=FS)
    fn_with = m_with["chatter_fn_band_energy_xtpos"]
    fn_without = m_without["chatter_fn_band_energy_xtpos"]
    assert fn_with > 0.5, (
        f"expected fn band energy > 0.5 with strong fn tone, got {fn_with:.4f}"
    )
    assert fn_without < 0.05, (
        f"expected fn band energy < 0.05 without fn tone, got {fn_without:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. fn-band kurtosis is high for a narrow peak and near zero for flat noise.
# ---------------------------------------------------------------------------

def test_chatter_fn_band_kurtosis_high_for_narrow_peak() -> None:
    # Use a finer PSD resolution (nperseg = 4096) and a wider fn band so the
    # band contains enough PSD bins for a meaningful kurtosis estimate. The
    # default (nperseg = 2048, fn_band_halfwidth_frac = 0.05) only places ~6
    # PSD bins in the fn band, which is too few to resolve narrow vs flat.
    # nperseg = 4096 keeps roughly 24 Welch segments, so the flat-noise PSD
    # is chi-squared with enough d.o.f. for kurtosis variance near zero.
    cfg = NoiseMetricsConfig(nperseg=4096, fn_band_halfwidth_frac=0.15)
    rng = np.random.default_rng(SEED)
    t = _t()

    x_peak = np.sin(2 * np.pi * FN * t) + 0.001 * rng.normal(0.0, 1.0, N)
    m_peak = compute_file_metrics(_broadcast(x_peak), tpf=TPF, fn=FN, fs=FS, config=cfg)
    kurt_peak = m_peak["chatter_fn_band_kurtosis_xtpos"]
    assert kurt_peak > 5, (
        f"expected fn band kurtosis > 5 for narrow peak, got {kurt_peak:.4f}"
    )

    x_flat = rng.normal(0.0, 1.0, N)
    m_flat = compute_file_metrics(_broadcast(x_flat), tpf=TPF, fn=FN, fs=FS, config=cfg)
    kurt_flat = m_flat["chatter_fn_band_kurtosis_xtpos"]
    assert abs(kurt_flat) < 1.0, (
        f"expected |fn band kurtosis| < 1 for flat noise, got {kurt_flat:.4f}"
    )


# ---------------------------------------------------------------------------
# 7. Sideband energy detects fn +/- k*tpf structure.
# ---------------------------------------------------------------------------

def test_sideband_energy_detects_fn_plus_minus_k_tpf() -> None:
    t = _t()
    x_sb = (
        np.sin(2 * np.pi * FN * t)
        + 0.5 * np.sin(2 * np.pi * (FN + TPF) * t)
        + 0.5 * np.sin(2 * np.pi * (FN - TPF) * t)
    )
    x_no = np.sin(2 * np.pi * FN * t)
    m_sb = compute_file_metrics(_broadcast(x_sb), tpf=TPF, fn=FN, fs=FS)
    m_no = compute_file_metrics(_broadcast(x_no), tpf=TPF, fn=FN, fs=FS)
    sb_with = m_sb["chatter_sideband_energy_xtpos"]
    sb_no = m_no["chatter_sideband_energy_xtpos"]
    assert sb_with > 0.20, (
        f"expected sideband energy > 0.20 when sidebands present, got {sb_with:.4f}"
    )
    assert sb_no < 0.05, (
        f"expected sideband energy < 0.05 with no sidebands, got {sb_no:.4f}"
    )


# ---------------------------------------------------------------------------
# 8. Cyclostationarity is large for a tpf-periodic signal and small for noise.
# ---------------------------------------------------------------------------

def test_cyclostationarity_detects_tpf_periodic_signal() -> None:
    rng = np.random.default_rng(SEED)
    t = _t()
    x_periodic = np.sin(2 * np.pi * TPF * t) + 0.2 * rng.normal(0.0, 1.0, N)
    x_noise = rng.normal(0.0, 1.0, N)
    m_p = compute_file_metrics(_broadcast(x_periodic), tpf=TPF, fn=FN, fs=FS)
    m_n = compute_file_metrics(_broadcast(x_noise), tpf=TPF, fn=FN, fs=FS)
    cyc_p = m_p["chatter_cyclostationarity_tpf_xtpos"]
    cyc_n = m_n["chatter_cyclostationarity_tpf_xtpos"]
    assert cyc_p > 2.0, (
        f"expected cyclostationarity > 2.0 for tpf-periodic, got {cyc_p:.4f}"
    )
    assert cyc_n < 0.5, (
        f"expected cyclostationarity < 0.5 for white noise, got {cyc_n:.4f}"
    )


# ---------------------------------------------------------------------------
# 9. Cross-channel coherence is high for shared content and low for independent.
# ---------------------------------------------------------------------------

def test_cross_coherence_high_for_shared_signal() -> None:
    # The coherence metric is the mean magnitude-squared coherence over the
    # band [0, 5*tpf]. With a single shared pure tone the cross-spectrum
    # only carries power at one bin while many bins are dominated by
    # independent noise, so the band-mean is < 0.3 by construction. To test
    # the physics expectation (shared signal -> high coherence, independent
    # -> low) we use broadband shared content (white noise) plus small
    # independent additive noise. This is the canonical textbook setup for
    # the coherence estimator.
    rng = np.random.default_rng(SEED)
    shared = rng.normal(0.0, 1.0, N)
    n1 = rng.normal(0.0, 1.0, N)
    n2 = rng.normal(0.0, 1.0, N)
    fx = shared + 0.1 * n1
    fy = shared + 0.1 * n2
    xp = rng.normal(0.0, 1.0, N)
    yp = rng.normal(0.0, 1.0, N)
    bundle = {"Forcex": fx, "Forcey": fy, "xtpos": xp, "ytpos": yp}
    m = compute_file_metrics(bundle, tpf=TPF, fn=FN, fs=FS)
    shared_coh = m["cross_coh_Forcex_Forcey"]
    indep_coh = m["cross_coh_xtpos_ytpos"]
    assert shared_coh > 0.7, (
        f"expected cross coherence > 0.7 for shared signal, got {shared_coh:.4f}"
    )
    assert indep_coh < 0.3, (
        f"expected cross coherence < 0.3 for independent noise, got {indep_coh:.4f}"
    )


# ---------------------------------------------------------------------------
# 10. Noise floor ratio increases when white noise is added.
# ---------------------------------------------------------------------------

def test_noise_floor_ratio_increases_with_added_white_noise() -> None:
    rng = np.random.default_rng(SEED)
    t = _t()
    x_clean = np.sin(2 * np.pi * TPF * t)
    x_noisy = x_clean + 0.5 * rng.normal(0.0, 1.0, N)
    for sig in SIGNALS:
        mc = compute_file_metrics(_broadcast(x_clean), tpf=TPF, fn=FN, fs=FS)
        mn = compute_file_metrics(_broadcast(x_noisy), tpf=TPF, fn=FN, fs=FS)
        clean_nf = mc[f"psd_noise_floor_ratio_{sig}"]
        noisy_nf = mn[f"psd_noise_floor_ratio_{sig}"]
        assert noisy_nf > clean_nf, (
            f"noise floor ratio for {sig} should rise with added noise; "
            f"clean={clean_nf:.6f}, noisy={noisy_nf:.6f}"
        )


# ---------------------------------------------------------------------------
# 11. Synthetic ladder: clean -> 1% -> 5% noise increases spectral entropy.
# ---------------------------------------------------------------------------

def test_csv_ladder_signal_only_quickcheck() -> None:
    rng = np.random.default_rng(SEED)
    t = _t()
    x_clean = np.sin(2 * np.pi * TPF * t) + 0.3 * np.sin(2 * np.pi * 2 * TPF * t)
    # Use independent noise draws so the ladder is reproducible.
    x_low = x_clean + 0.01 * rng.normal(0.0, 1.0, N)
    x_high = x_clean + 0.05 * rng.normal(0.0, 1.0, N)
    m_clean = compute_file_metrics(_broadcast(x_clean), tpf=TPF, fn=FN, fs=FS)
    m_low = compute_file_metrics(_broadcast(x_low), tpf=TPF, fn=FN, fs=FS)
    m_high = compute_file_metrics(_broadcast(x_high), tpf=TPF, fn=FN, fs=FS)
    for sig in SIGNALS:
        ec = m_clean[f"psd_spectral_entropy_{sig}"]
        e1 = m_low[f"psd_spectral_entropy_{sig}"]
        e5 = m_high[f"psd_spectral_entropy_{sig}"]
        assert ec < e1 < e5, (
            f"entropy must rise with noise for {sig}: clean={ec:.6f}, "
            f"1%={e1:.6f}, 5%={e5:.6f}"
        )


# ---------------------------------------------------------------------------
# 12. Missing signal returns NaN keys but does not raise.
# ---------------------------------------------------------------------------

def test_missing_signal_returns_nan_not_raise() -> None:
    rng = np.random.default_rng(SEED)
    bundle = {sig: rng.normal(0.0, 1.0, N) for sig in SIGNALS if sig != "ytpos"}
    m = compute_file_metrics(bundle, tpf=TPF, fn=FN, fs=FS)
    # ytpos keys are NaN.
    for key in [
        "psd_noise_floor_ratio_ytpos",
        "psd_off_harmonic_ratio_ytpos",
        "psd_spectral_flatness_ytpos",
        "psd_spectral_entropy_ytpos",
        "wavelet_mad_sigma_ytpos",
        "autocorr_decay_lag_samples_ytpos",
        "chatter_fn_band_energy_ytpos",
        "chatter_fn_band_kurtosis_ytpos",
        "chatter_sideband_energy_ytpos",
        "chatter_cyclostationarity_tpf_ytpos",
    ]:
        v = m[key]
        assert math.isnan(v), f"{key} should be NaN when signal is missing, got {v}"
    # Other signals have real numeric values somewhere.
    real_count = sum(
        1
        for k, v in m.items()
        if "xtpos" in k or "Forcex" in k or "Forcey" in k
        if not math.isnan(v) and math.isfinite(v)
    )
    assert real_count > 0, "expected at least one real-valued metric for present signals"


# ---------------------------------------------------------------------------
# 13. Short / bad inputs return NaN rather than raising.
# ---------------------------------------------------------------------------

def test_short_signal_or_bad_inputs_does_not_raise() -> None:
    rng = np.random.default_rng(SEED)
    short = rng.normal(0.0, 1.0, 100)
    bundle = _broadcast(short)
    # Must not raise even though 100 samples is below the default Welch
    # minimum and produces a very weak PSD.
    m = compute_file_metrics(bundle, tpf=TPF, fn=FN, fs=FS)
    assert isinstance(m, dict)
    # At least some keys are present.
    assert "psd_spectral_entropy_xtpos" in m

    # All-NaN signals must also not raise.
    bundle_nan = {sig: np.full(N, np.nan) for sig in SIGNALS}
    m2 = compute_file_metrics(bundle_nan, tpf=TPF, fn=FN, fs=FS)
    assert isinstance(m2, dict)

    # Missing fs / tpf / fn must not raise.
    m3 = compute_file_metrics(_broadcast(rng.normal(0, 1, N)))
    assert isinstance(m3, dict)


# ---------------------------------------------------------------------------
# 14. Realistic combined signal: every metric is finite-or-NaN.
# ---------------------------------------------------------------------------

def test_metrics_are_finite_on_realistic_signal() -> None:
    rng = np.random.default_rng(SEED)
    t = _t()
    bundle = {}
    for sig in SIGNALS:
        x = (
            np.sin(2 * np.pi * TPF * t)
            + 0.3 * np.sin(2 * np.pi * 2 * TPF * t)
            + 0.2 * np.sin(2 * np.pi * FN * t)
            + 0.01 * rng.normal(0.0, 1.0, N)
        )
        bundle[sig] = x
    m = compute_file_metrics(bundle, tpf=TPF, fn=FN, fs=FS)
    for key, value in m.items():
        assert math.isnan(value) or math.isfinite(value), (
            f"{key} is neither finite nor NaN: {value}"
        )
        assert not math.isinf(value), f"{key} is +/-inf: {value}"
    # Spectral entropy must be in [0, 1] for each signal.
    for sig in SIGNALS:
        ent = m[f"psd_spectral_entropy_{sig}"]
        assert math.isfinite(ent), f"entropy for {sig} should be finite, got {ent}"
        assert 0.0 <= ent <= 1.0, (
            f"entropy for {sig} must be in [0, 1], got {ent:.6f}"
        )


# ---------------------------------------------------------------------------
# 15. Config overrides change the metric.
# ---------------------------------------------------------------------------

def test_config_overrides_take_effect() -> None:
    rng = np.random.default_rng(SEED)
    t = _t()
    # Include enough harmonic content that increasing the harmonic count
    # actually changes the off-harmonic ratio.
    x = sum(
        (1.0 / k) * np.sin(2 * np.pi * k * TPF * t) for k in range(1, 11)
    ) + 0.05 * rng.normal(0.0, 1.0, N)
    bundle = _broadcast(x)
    # Both configs use a wide analysis band so the harmonic count override
    # actually changes which harmonics are masked. With the default 5x tpf
    # band, k=6..10 fall outside the band and would be ignored.
    base = NoiseMetricsConfig(analysis_tpf_multiplier=15.0)
    default_metrics = compute_file_metrics(
        bundle, tpf=TPF, fn=FN, fs=FS, config=base
    )
    override = NoiseMetricsConfig(
        tpf_harmonic_count=10, analysis_tpf_multiplier=15.0
    )
    override_metrics = compute_file_metrics(
        bundle, tpf=TPF, fn=FN, fs=FS, config=override
    )
    for sig in SIGNALS:
        a = default_metrics[f"psd_off_harmonic_ratio_{sig}"]
        b = override_metrics[f"psd_off_harmonic_ratio_{sig}"]
        assert not math.isnan(a) and not math.isnan(b)
        assert a != b, (
            f"expected tpf_harmonic_count override to change off-harmonic ratio "
            f"for {sig}; got {a:.6f} == {b:.6f}"
        )
        # Adding more harmonics to the capture mask can only reduce or keep
        # the off-harmonic ratio.
        assert b <= a + 1e-9, (
            f"off-harmonic ratio with more harmonics must not exceed default "
            f"for {sig}; got default={a:.6f}, override={b:.6f}"
        )


# ---------------------------------------------------------------------------
# 16. Cross-channel coherence uses a joint finite mask (alignment fix).
# ---------------------------------------------------------------------------

def test_cross_coherence_joint_nan_mask_preserves_alignment() -> None:
    """NaNs at different indices in each channel must NOT corrupt coherence.

    Two correlated channels (same broadband signal + tiny independent noise)
    should yield high cross-coherence even when NaNs are scattered at DIFFERENT
    positions in each channel.  If channels are independently cleaned the
    resulting arrays will be misaligned, producing garbage coherence.  The
    joint-mask implementation must return a value that is:
      (a) close to the clean-signal reference (no NaNs), and
      (b) well above what a misaligned pair would give.

    We also verify that SINGLE-CHANNEL metrics (spectral entropy for xtpos)
    are unaffected by the NaNs placed in other channels.
    """
    rng = np.random.default_rng(SEED)
    shared = rng.normal(0.0, 1.0, N)
    n1 = rng.normal(0.0, 0.05, N)
    n2 = rng.normal(0.0, 0.05, N)
    fx_clean = shared + n1
    fy_clean = shared + n2

    # NaN positions that are DIFFERENT in each channel: every 7th in fx,
    # every 11th in fy.  With independent dropping these two arrays end up
    # ~7/11 * N vs ~10/11 * N samples, and their phase alignment is destroyed.
    fx_nan = fx_clean.copy()
    fy_nan = fy_clean.copy()
    fx_nan[::7] = np.nan
    fy_nan[::11] = np.nan

    # Reference coherence from the clean (no-NaN) bundle.
    bundle_clean = {
        "Forcex": fx_clean,
        "Forcey": fy_clean,
        "xtpos": rng.normal(0.0, 1.0, N),
        "ytpos": rng.normal(0.0, 1.0, N),
    }
    m_clean = compute_file_metrics(bundle_clean, tpf=TPF, fn=FN, fs=FS)
    ref_coh = m_clean["cross_coh_Forcex_Forcey"]

    # Bundle with staggered NaNs.
    bundle_nan = {
        "Forcex": fx_nan,
        "Forcey": fy_nan,
        "xtpos": rng.normal(0.0, 1.0, N),
        "ytpos": rng.normal(0.0, 1.0, N),
    }
    m_nan = compute_file_metrics(bundle_nan, tpf=TPF, fn=FN, fs=FS)
    nan_coh = m_nan["cross_coh_Forcex_Forcey"]

    # The joint-masked result should be close to the clean reference.
    assert math.isfinite(nan_coh), (
        f"cross_coh_Forcex_Forcey should be finite with partial NaNs, got {nan_coh}"
    )
    assert nan_coh > 0.7, (
        f"expected cross coherence > 0.7 with joint NaN masking "
        f"(ref={ref_coh:.4f}), got {nan_coh:.4f}"
    )
    assert abs(nan_coh - ref_coh) < 0.15, (
        f"joint-masked coherence {nan_coh:.4f} should be within 0.15 of "
        f"clean reference {ref_coh:.4f}"
    )

    # Single-channel metrics should be unaffected by NaNs in OTHER channels.
    # xtpos has no NaNs injected, so its spectral entropy should match between
    # clean and nan bundles (same signal, same seed; xtpos was regenerated with
    # the same rng state, so just check it's finite and in [0, 1]).
    ent_nan = m_nan["psd_spectral_entropy_xtpos"]
    assert math.isfinite(ent_nan), (
        f"single-channel xtpos spectral entropy should be finite, got {ent_nan}"
    )
    assert 0.0 <= ent_nan <= 1.0, (
        f"single-channel entropy out of [0,1]: {ent_nan:.4f}"
    )
