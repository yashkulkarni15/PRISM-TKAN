import numpy as np
import pytest
from pri2m.xai.spectral import (
    magnitude_randomize,
    match_notch_halfwidth,
    notch,
    phase_randomize,
)


def _sig(C=4, L=256, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    base = np.sin(2 * np.pi * 7 * t / L) + 0.3 * np.sin(2 * np.pi * 23 * t / L)
    return np.stack([base * (c + 1) + 0.1 * rng.standard_normal(L) for c in range(C)])


def test_phase_randomize_preserves_magnitude_spectrum_exactly():
    x = _sig()
    y = phase_randomize(x, alpha=1.0, seed=0, amplitude_adjust=False)
    mx = np.abs(np.fft.rfft(x, axis=-1))
    my = np.abs(np.fft.rfft(y, axis=-1))
    np.testing.assert_allclose(my, mx, rtol=1e-8, atol=1e-8)


def test_phase_randomize_alpha_zero_is_identity():
    x = _sig()
    y = phase_randomize(x, alpha=0.0, seed=0, amplitude_adjust=False)
    np.testing.assert_allclose(y, x, rtol=1e-8, atol=1e-8)


def test_phase_randomize_output_is_real_and_finite():
    x = _sig()
    y = phase_randomize(x, alpha=1.0, seed=3)
    assert y.dtype.kind == "f"
    assert np.all(np.isfinite(y))


def test_phase_randomize_preserves_amplitude_distribution_when_adjusted():
    x = _sig()
    y = phase_randomize(x, alpha=1.0, seed=1, amplitude_adjust=True)
    for c in range(x.shape[0]):
        np.testing.assert_allclose(np.sort(y[c]), np.sort(x[c]), rtol=1e-8, atol=1e-8)


def test_amplitude_adjust_keeps_spectral_deviation_bounded():
    # AAFT trades exact spectrum for exact amplitude distribution. Bound it.
    x = _sig()
    y = phase_randomize(x, alpha=1.0, seed=2, amplitude_adjust=True)
    mx = np.abs(np.fft.rfft(x, axis=-1))
    my = np.abs(np.fft.rfft(y, axis=-1))
    rel = np.linalg.norm(my - mx) / np.linalg.norm(mx)
    assert rel < 0.05, f"AAFT perturbed the spectrum by {rel:.3f}"


def test_phase_randomize_shares_perturbation_across_channels():
    # Inter-channel phase differences must be preserved: that is what keeps
    # x/y and force/position coupling intact.
    x = _sig()
    y = phase_randomize(x, alpha=1.0, seed=4, amplitude_adjust=False)
    px = np.angle(np.fft.rfft(x, axis=-1))
    py = np.angle(np.fft.rfft(y, axis=-1))
    dx = np.unwrap(px[0] - px[1])
    dy = np.unwrap(py[0] - py[1])
    np.testing.assert_allclose(dy, dx, rtol=1e-6, atol=1e-6)


def test_phase_randomize_actually_changes_the_signal():
    x = _sig()
    y = phase_randomize(x, alpha=1.0, seed=5, amplitude_adjust=False)
    assert np.linalg.norm(y - x) / np.linalg.norm(x) > 0.1


def test_phase_randomize_rejects_non_2d():
    with pytest.raises(ValueError, match=r"\(C, L\)"):
        phase_randomize(np.zeros(16), alpha=1.0)


def test_magnitude_randomize_preserves_phase_exactly():
    x = _sig()
    y = magnitude_randomize(x, alpha=1.0, seed=0)
    px = np.angle(np.fft.rfft(x, axis=-1))
    py = np.angle(np.fft.rfft(y, axis=-1))
    # compare on bins with non-negligible magnitude; phase of a ~0 bin is noise
    mx = np.abs(np.fft.rfft(x, axis=-1))
    keep = mx > 1e-6 * mx.max()
    np.testing.assert_allclose(np.cos(py[keep]), np.cos(px[keep]), atol=1e-6)
    np.testing.assert_allclose(np.sin(py[keep]), np.sin(px[keep]), atol=1e-6)


def test_magnitude_randomize_alpha_zero_is_identity():
    x = _sig()
    np.testing.assert_allclose(magnitude_randomize(x, 0.0, seed=0), x, rtol=1e-8, atol=1e-8)


def test_magnitude_randomize_changes_the_spectrum():
    x = _sig()
    y = magnitude_randomize(x, alpha=1.0, seed=1)
    mx = np.abs(np.fft.rfft(x, axis=-1))
    my = np.abs(np.fft.rfft(y, axis=-1))
    assert np.linalg.norm(my - mx) / np.linalg.norm(mx) > 0.1


def test_magnitude_randomize_preserves_total_energy():
    x = _sig()
    y = magnitude_randomize(x, alpha=1.0, seed=2)
    np.testing.assert_allclose(
        np.sum(y**2, axis=-1), np.sum(x**2, axis=-1), rtol=1e-6
    )


def test_notch_removes_energy_at_the_target_frequency():
    fs, L = 1000.0, 1024
    t = np.arange(L) / fs
    x = np.stack([np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 300 * t)])
    y, frac = notch(x, fs=fs, centers=[100.0], halfwidth_hz=10.0)
    # the 100 Hz component should be largely gone, 300 Hz largely intact
    spec = np.abs(np.fft.rfft(y, axis=-1))[0]
    freqs = np.fft.rfftfreq(L, 1 / fs)
    assert spec[np.argmin(np.abs(freqs - 100))] < 1e-6 * spec.max()
    assert spec[np.argmin(np.abs(freqs - 300))] > 0.1 * spec.max()
    assert 0.0 < frac < 1.0


def test_notch_zero_halfwidth_is_identity():
    fs, L = 1000.0, 512
    x = np.random.default_rng(0).standard_normal((2, L))
    y, frac = notch(x, fs=fs, centers=[100.0], halfwidth_hz=0.0)
    np.testing.assert_allclose(y, x, rtol=1e-8, atol=1e-8)
    assert frac == 0.0


def test_match_notch_halfwidth_hits_target_within_tolerance():
    fs, L = 1000.0, 1024
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, L))
    target = 0.05
    hw = match_notch_halfwidth(x, fs=fs, centers=[200.0, 400.0], target_frac=target)
    _, frac = notch(x, fs=fs, centers=[200.0, 400.0], halfwidth_hz=hw)
    assert abs(frac - target) <= 0.01 * target + 1e-3
