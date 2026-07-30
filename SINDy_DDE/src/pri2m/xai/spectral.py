"""Spectral surrogates and notch filters for the physics-floor stressors.

Pure numpy: no torch, no model. Every function takes and returns a real
``(C, L)`` array so the callers in ``degradation.py`` stay thin.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "magnitude_randomize",
    "match_notch_halfwidth",
    "notch",
    "phase_randomize",
    "rank_remap",
]


def rank_remap(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Remap ``y`` onto ``x``'s amplitude distribution, preserving y's rank order.

    Per channel. This is the "amplitude-adjusted" step of an AAFT surrogate: it
    guarantees the surrogate has exactly the original amplitude histogram, so a
    model's reaction cannot be attributed to a distribution shift.
    """
    out = np.empty_like(y)
    for c in range(y.shape[0]):
        ranks = np.argsort(np.argsort(y[c]))
        out[c] = np.sort(x[c])[ranks]
    return out


def phase_randomize(
    x: np.ndarray, alpha: float, seed: int = 0, amplitude_adjust: bool = True
) -> np.ndarray:
    """Randomize Fourier phase by fraction ``alpha``, holding magnitude fixed.

    ONE perturbation array is drawn and applied to EVERY channel, so
    inter-channel relative phase (x/y position coupling, force/position
    coupling) survives untouched. Randomizing per channel would destroy that
    too and confound "lost regeneration" with "lost cross-channel structure".

    ``alpha=0`` is the identity; ``alpha=1`` fully randomizes phase. Magnitude
    is preserved exactly unless ``amplitude_adjust`` is set, which trades a
    small spectral deviation for an exact amplitude distribution.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (C, L) array, got shape {x.shape}")
    length = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    mag, phase = np.abs(spec), np.angle(spec)

    rng = np.random.default_rng(int(seed))
    dphi = rng.uniform(-np.pi, np.pi, size=spec.shape[-1])
    # DC (and Nyquist for even length) must stay real, else irfft silently
    # discards the imaginary part and the magnitude is NOT preserved there.
    dphi[0] = 0.0
    if length % 2 == 0:
        dphi[-1] = 0.0

    out = np.fft.irfft(mag * np.exp(1j * (phase + float(alpha) * dphi)), n=length, axis=-1)
    if amplitude_adjust:
        out = rank_remap(out, x)
    return out


def magnitude_randomize(x: np.ndarray, alpha: float, seed: int = 0) -> np.ndarray:
    """Scramble the magnitude spectrum by fraction ``alpha``, holding phase fixed.

    The exact dual of :func:`phase_randomize`, and the control that makes it
    interpretable. Total per-channel energy is renormalized to the original so
    that a performance change cannot be attributed to loudness.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (C, L) array, got shape {x.shape}")
    length = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    mag, phase = np.abs(spec), np.angle(spec)

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(mag.shape[-1])
    new_mag = (1.0 - float(alpha)) * mag + float(alpha) * mag[:, perm]

    out = np.fft.irfft(new_mag * np.exp(1j * phase), n=length, axis=-1)

    # Renormalize per channel so total TIME-DOMAIN energy matches the
    # original exactly. Renormalizing in the frequency domain via the raw
    # sum(mag**2) is only approximate: Parseval's theorem for rfft weights
    # the DC (and Nyquist, for even length) bins by 1 and every other bin
    # by 2, so an unweighted spectral-sum match does not imply an exact
    # time-domain energy match once the permutation moves magnitude
    # into/out of those two special bins. Rescaling the reconstructed
    # signal directly sidesteps the weighting entirely and is exact by
    # construction; it is just a positive real per-channel scalar, so it
    # does not touch the phase.
    e_x = np.sum(x**2, axis=-1, keepdims=True)
    e_out = np.sum(out**2, axis=-1, keepdims=True)
    scale = np.sqrt(np.divide(e_x, e_out, out=np.ones_like(e_x), where=e_out > 0))

    return out * scale


def notch(
    x: np.ndarray, fs: float, centers, halfwidth_hz: float
) -> tuple[np.ndarray, float]:
    """Zero the spectrum within ``halfwidth_hz`` of each centre frequency.

    Returns the filtered signal and the fraction of total spectral energy
    removed, which the matched-energy contrast needs in order to be fair.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (C, L) array, got shape {x.shape}")
    length = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(length, 1.0 / float(fs))

    mask = np.ones(freqs.shape, dtype=float)
    if float(halfwidth_hz) > 0:
        for f0 in centers:
            mask[np.abs(freqs - float(f0)) <= float(halfwidth_hz)] = 0.0

    before = float(np.sum(np.abs(spec) ** 2))
    filtered = spec * mask[None, :]
    after = float(np.sum(np.abs(filtered) ** 2))
    removed = 0.0 if before <= 0 else (before - after) / before
    return np.fft.irfft(filtered, n=length, axis=-1), removed


def match_notch_halfwidth(
    x: np.ndarray,
    fs: float,
    centers,
    target_frac: float,
    tol: float = 0.01,
    max_iter: int = 60,
) -> float:
    """Bisect the notch halfwidth until it removes ``target_frac`` of the energy.

    Without this the chatter-band vs tooth-pass contrast is confounded: the
    condition that deletes more energy wins regardless of what it deleted.
    """
    lo, hi = 0.0, float(fs) / 2.0
    best, best_err = hi, float("inf")
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        _, frac = notch(x, fs=fs, centers=centers, halfwidth_hz=mid)
        err = abs(frac - target_frac)
        # Energy-removed is a step function of halfwidth (it only changes
        # when the window crosses an FFT bin), so an exact target can be
        # unreachable and bisection can oscillate between the two nearest
        # achievable steps without ever satisfying the tolerance. Track the
        # closest candidate seen so far rather than just the last one tried,
        # so a legitimate max_iter exhaustion still returns the best fit.
        if err < best_err:
            best, best_err = mid, err
        if err <= tol * target_frac:
            return mid
        if frac < target_frac:
            lo = mid
        else:
            hi = mid
    return best
