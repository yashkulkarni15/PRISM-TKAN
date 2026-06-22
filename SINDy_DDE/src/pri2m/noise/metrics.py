"""Phase 2.5 per-file noise, chatter, and cross-channel metrics.

This module implements the metric library defined in
``docs/phase2_5_noise_mode_filtering.md``. It interprets each file as a mixture
of three components and emits per-signal scalar metrics that downstream code
uses for mode labeling and conservative filtering decisions:

1. Forced cutting response: energy at the tooth-pass frequency ``tpf`` and its
   harmonics. Captured by the PSD off-harmonic ratio (low when TPF energy
   dominates) and indirectly by spectral entropy/flatness.

2. Chatter response: energy near the structural natural frequency ``fn`` and
   regenerative sidebands at ``fn +/- k*tpf``. Captured by the chatter family
   (``chatter_fn_band_energy``, ``chatter_fn_band_kurtosis``,
   ``chatter_sideband_energy``, ``chatter_cyclostationarity_tpf``).

3. Measurement/process noise: broadband incoherent energy that is not aligned
   with TPF harmonics, the fn band, or cross-channel coherence. Captured by the
   PSD high-frequency noise floor ratio, spectral flatness, spectral entropy,
   the wavelet MAD sigma, and the cross-channel coherence family.

The library is pure: no file I/O, no CLI. ``compute_file_metrics`` accepts an
NPZ-like mapping (as produced by ``scripts/extract_phase2_signals.py``) and
returns one flat dict per file with keys named ``<family>_<metric>_<signal>``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

try:
    import pywt
except ImportError as exc:  # pragma: no cover - explicit hard failure
    raise ImportError(
        "pri2m.noise.metrics requires PyWavelets (pywt); install it via "
        "'pip install PyWavelets'."
    ) from exc

from scipy.signal import coherence as _scipy_coherence
from scipy.signal import welch as _scipy_welch
from scipy.stats import kurtosis as _scipy_kurtosis

from pri2m.spectral.psd_js import choose_nperseg


SIGNALS: tuple[str, ...] = ("xtpos", "ytpos", "Forcex", "Forcey")


@dataclass(frozen=True)
class NoiseMetricsConfig:
    """Pinned defaults for the Phase 2.5 noise metric library.

    All thresholds, band widths, and harmonic counts live here so a single
    config object completely describes the metric definitions.
    """

    nperseg: int = 2048
    overlap_fraction: float = 0.5
    analysis_tpf_multiplier: float = 5.0
    tpf_harmonic_count: int = 5
    tpf_harmonic_halfwidth_frac: float = 0.05
    fn_band_halfwidth_frac: float = 0.05
    sideband_count: int = 5
    sideband_halfwidth_frac: float = 0.03
    noise_floor_low_multiplier: float = 5.0
    wavelet: str = "db4"
    wavelet_level: int = 1
    coherence_band_tpf_multiplier: float = 5.0
    eps: float = 1e-12


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_file_metrics(
    bundle: Mapping[str, np.ndarray],
    *,
    tpf: float | None = None,
    fn: float | None = None,
    fs: float | None = None,
    config: NoiseMetricsConfig | None = None,
) -> dict[str, float]:
    """Compute the full Phase 2.5 noise/chatter/cross metric vector for one file.

    Parameters
    ----------
    bundle:
        NPZ-like mapping. Must contain a subset of ``SIGNALS``. May contain
        ``fs``, ``tpf``, ``fn`` arrays/scalars used as fallback.
    tpf, fn, fs:
        Optional overrides. When ``None``, fall back to ``bundle[name]``. Any
        missing or non-positive value gracefully yields NaN for metrics that
        require it; the function never raises on bad inputs.
    config:
        Pinned numerical defaults. ``None`` uses ``NoiseMetricsConfig()``.

    Returns
    -------
    dict[str, float]
        Flat dictionary with keys ``<family>_<metric>_<signal>`` for the
        PSD/wavelet/chatter families and ``cross_coh_<pair>`` for the cross
        family. Missing signals produce NaN entries for their keys but never
        raise.
    """

    cfg = config or NoiseMetricsConfig()
    fs_value = _resolve_scalar(fs, bundle.get("fs"))
    tpf_value = _resolve_scalar(tpf, bundle.get("tpf"))
    fn_value = _resolve_scalar(fn, bundle.get("fn"))

    out: dict[str, float] = {}

    # Per-signal PSD, wavelet, and chatter families ------------------------
    psd_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    signal_cache: dict[str, np.ndarray] = {}
    # Raw (un-NaN-dropped) arrays kept for joint finite masking in cross-
    # channel coherence.  Independently dropping NaNs before coherence
    # computation misaligns the two channels and corrupts the estimate.
    raw_cache: dict[str, np.ndarray] = {}
    for signal in SIGNALS:
        raw = _coerce_signal_raw(bundle.get(signal))
        x = _coerce_signal(bundle.get(signal))
        if x is None:
            _fill_signal_nan(out, signal)
            continue
        signal_cache[signal] = x
        if raw is not None:
            raw_cache[signal] = raw

        # PSD family.
        freqs, psd = _safe_welch(x, fs_value, cfg)
        if freqs is not None:
            psd_cache[signal] = (freqs, psd)
        _fill_psd_metrics(
            out, signal, freqs, psd, tpf_value, fn_value, fs_value, cfg
        )

        # Wavelet/temporal family.
        out[f"wavelet_mad_sigma_{signal}"] = _wavelet_mad_sigma(x, cfg)
        out[f"autocorr_decay_lag_samples_{signal}"] = _autocorr_decay_lag(x)

        # Chatter family.
        _fill_chatter_metrics(
            out, signal, freqs, psd, x, tpf_value, fn_value, fs_value, cfg
        )

    # Cross-channel coherence family --------------------------------------
    # Pass raw arrays so _pair_coherence can apply a joint finite mask that
    # keeps both channels index-aligned.
    _fill_cross_coherence(out, raw_cache, fs_value, tpf_value, cfg)

    return out


# ---------------------------------------------------------------------------
# Helpers: resolution and coercion
# ---------------------------------------------------------------------------

def _resolve_scalar(primary: object, fallback: object) -> float | None:
    for candidate in (primary, fallback):
        if candidate is None:
            continue
        try:
            arr = np.asarray(candidate, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if arr.size == 0:
            continue
        value = float(arr[0])
        if math.isfinite(value) and value > 0:
            return value
    return None


def _coerce_signal(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        x = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if x.size == 0:
        return None
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return x


def _coerce_signal_raw(value: object) -> np.ndarray | None:
    """Return the signal as a 1-D float array WITHOUT dropping non-finite values.

    Used to preserve the original sample index alignment for cross-channel
    operations that need to apply a JOINT finite mask across both channels.
    Returns ``None`` only when the input cannot be converted or is empty.
    """
    if value is None:
        return None
    try:
        x = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if x.size == 0:
        return None
    return x


def _fill_signal_nan(out: dict[str, float], signal: str) -> None:
    for key in (
        f"psd_noise_floor_ratio_{signal}",
        f"psd_off_harmonic_ratio_{signal}",
        f"psd_spectral_flatness_{signal}",
        f"psd_spectral_entropy_{signal}",
        f"wavelet_mad_sigma_{signal}",
        f"autocorr_decay_lag_samples_{signal}",
        f"chatter_fn_band_energy_{signal}",
        f"chatter_fn_band_kurtosis_{signal}",
        f"chatter_sideband_energy_{signal}",
        f"chatter_cyclostationarity_tpf_{signal}",
    ):
        out[key] = math.nan


# ---------------------------------------------------------------------------
# PSD helpers
# ---------------------------------------------------------------------------

def _safe_welch(
    x: np.ndarray, fs: float | None, cfg: NoiseMetricsConfig
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if fs is None or x.size < 8:
        return None, None
    try:
        nperseg = choose_nperseg(x.size, cfg.nperseg, min_nperseg=8)
    except ValueError:
        return None, None
    noverlap = int(round(nperseg * cfg.overlap_fraction))
    noverlap = max(0, min(noverlap, nperseg - 1))
    try:
        freqs, psd = _scipy_welch(
            x,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            return_onesided=True,
        )
    except Exception:
        return None, None
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    return freqs, psd


def _bin_widths(freqs: np.ndarray) -> np.ndarray:
    if freqs.size <= 1:
        return np.ones_like(freqs)
    widths = np.empty_like(freqs)
    widths[0] = freqs[1] - freqs[0]
    widths[-1] = freqs[-1] - freqs[-2]
    widths[1:-1] = 0.5 * (freqs[2:] - freqs[:-2])
    return np.maximum(widths, 0.0)


def _band_energy(
    freqs: np.ndarray, psd: np.ndarray, low: float, high: float
) -> float:
    if freqs.size == 0:
        return 0.0
    nyq = float(freqs[-1])
    low = max(0.0, float(low))
    high = max(low, min(float(high), nyq))
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    widths = _bin_widths(freqs)
    return float(np.sum(psd[mask] * widths[mask]))


def _band_mask_energy(
    freqs: np.ndarray, psd: np.ndarray, mask: np.ndarray
) -> float:
    if freqs.size == 0 or not np.any(mask):
        return 0.0
    widths = _bin_widths(freqs)
    return float(np.sum(psd[mask] * widths[mask]))


def _analysis_band_max(
    freqs: np.ndarray, tpf: float | None, cfg: NoiseMetricsConfig
) -> float:
    nyq = float(freqs[-1]) if freqs.size else math.inf
    if tpf is None or tpf <= 0:
        return nyq
    return float(min(cfg.analysis_tpf_multiplier * tpf, nyq))


def _fill_psd_metrics(
    out: dict[str, float],
    signal: str,
    freqs: np.ndarray | None,
    psd: np.ndarray | None,
    tpf: float | None,
    fn: float | None,
    fs: float | None,
    cfg: NoiseMetricsConfig,
) -> None:
    nf_key = f"psd_noise_floor_ratio_{signal}"
    oh_key = f"psd_off_harmonic_ratio_{signal}"
    flat_key = f"psd_spectral_flatness_{signal}"
    ent_key = f"psd_spectral_entropy_{signal}"

    if freqs is None or psd is None or freqs.size == 0:
        out[nf_key] = math.nan
        out[oh_key] = math.nan
        out[flat_key] = math.nan
        out[ent_key] = math.nan
        return

    nyq = float(freqs[-1])

    # Noise floor ratio.
    if tpf is not None and tpf > 0:
        low_edge = cfg.noise_floor_low_multiplier * tpf
        if low_edge < nyq:
            high_band = _band_energy(freqs, psd, low_edge, nyq)
            total_band = _band_energy(freqs, psd, 0.0, nyq)
            if total_band > 0:
                out[nf_key] = float(np.clip(high_band / total_band, 0.0, 1.0))
            else:
                out[nf_key] = math.nan
        else:
            out[nf_key] = math.nan
    else:
        out[nf_key] = math.nan

    # Analysis band for the next three metrics.
    band_max = _analysis_band_max(freqs, tpf, cfg)
    band_mask = (freqs >= 0.0) & (freqs <= band_max)
    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]

    if band_psd.size == 0 or band_max <= 0.0:
        out[oh_key] = math.nan
        out[flat_key] = math.nan
        out[ent_key] = math.nan
        return

    band_widths = _bin_widths(band_freqs)
    analysis_energy = float(np.sum(band_psd * band_widths))

    # Off-harmonic ratio.
    if tpf is not None and tpf > 0 and analysis_energy > 0:
        harmonic_mask = np.zeros_like(band_freqs, dtype=bool)
        half = cfg.tpf_harmonic_halfwidth_frac
        for k in range(1, cfg.tpf_harmonic_count + 1):
            center = k * tpf
            if center >= band_max:
                break
            tol = max(center * half, band_freqs[1] - band_freqs[0] if band_freqs.size > 1 else 0.0)
            harmonic_mask |= np.abs(band_freqs - center) <= tol

        fn_mask_in_band = np.zeros_like(band_freqs, dtype=bool)
        if fn is not None and fn > 0:
            tol_fn = fn * cfg.fn_band_halfwidth_frac
            fn_low = fn - tol_fn
            fn_high = fn + tol_fn
            fn_mask_in_band = (band_freqs >= fn_low) & (band_freqs <= fn_high)

        combined = harmonic_mask | fn_mask_in_band
        captured = float(np.sum(band_psd[combined] * band_widths[combined]))
        ratio = 1.0 - (captured / analysis_energy)
        out[oh_key] = float(np.clip(ratio, 0.0, 1.0))
    else:
        out[oh_key] = math.nan

    # Spectral flatness on the analysis band.
    positive = band_psd > 0
    if np.any(positive):
        log_mean = float(np.mean(np.log(band_psd[positive] + cfg.eps)))
        gmean = math.exp(log_mean)
        amean = float(np.mean(band_psd[positive]))
        out[flat_key] = float(gmean / amean) if amean > 0 else math.nan
    else:
        out[flat_key] = math.nan

    # Spectral entropy on the analysis band.
    psd_sum = float(np.sum(band_psd))
    if psd_sum > 0 and band_psd.size > 1:
        p = band_psd / psd_sum
        p = np.clip(p, cfg.eps, None)
        ent = -float(np.sum(p * np.log2(p)))
        out[ent_key] = float(np.clip(ent / math.log2(band_psd.size), 0.0, 1.0))
    else:
        out[ent_key] = math.nan


# ---------------------------------------------------------------------------
# Wavelet / temporal helpers
# ---------------------------------------------------------------------------

def _wavelet_mad_sigma(x: np.ndarray, cfg: NoiseMetricsConfig) -> float:
    try:
        coeffs = pywt.wavedec(
            x, cfg.wavelet, level=cfg.wavelet_level, mode="symmetric"
        )
    except (ValueError, RuntimeError):
        return math.nan
    if not coeffs:
        return math.nan
    detail = np.asarray(coeffs[-1], dtype=float)
    detail = detail[np.isfinite(detail)]
    if detail.size == 0:
        return math.nan
    return float(np.median(np.abs(detail)) / 0.6745)


def _autocorr_decay_lag(x: np.ndarray) -> float:
    if x.size < 4:
        return math.nan
    centered = x - float(np.mean(x))
    if not np.any(centered):
        return math.nan
    corr = np.correlate(centered, centered, mode="full")
    mid = corr.size // 2
    corr = corr[mid:]
    zero_val = corr[0]
    if not math.isfinite(zero_val) or zero_val <= 0:
        return math.nan
    norm = corr / zero_val
    envelope = np.abs(norm)
    threshold = 1.0 / math.e
    below = np.where(envelope < threshold)[0]
    if below.size == 0:
        return math.nan
    return float(below[0])


# ---------------------------------------------------------------------------
# Chatter family
# ---------------------------------------------------------------------------

def _fill_chatter_metrics(
    out: dict[str, float],
    signal: str,
    freqs: np.ndarray | None,
    psd: np.ndarray | None,
    x: np.ndarray,
    tpf: float | None,
    fn: float | None,
    fs: float | None,
    cfg: NoiseMetricsConfig,
) -> None:
    fn_e_key = f"chatter_fn_band_energy_{signal}"
    fn_k_key = f"chatter_fn_band_kurtosis_{signal}"
    sb_key = f"chatter_sideband_energy_{signal}"
    cy_key = f"chatter_cyclostationarity_tpf_{signal}"

    # PSD-based chatter metrics.
    # Denominator is total signal energy over [0, fs/2), not the [0, 5*tpf]
    # analysis band, because fn (and therefore the fn band and sidebands) is
    # typically above 5*tpf for real machining data. Using the analysis-band
    # energy as the denominator silently clips the ratio to 1 whenever the
    # chatter content lives outside the cutting band, which is the common case.
    if freqs is None or psd is None or freqs.size == 0:
        out[fn_e_key] = math.nan
        out[fn_k_key] = math.nan
        out[sb_key] = math.nan
    else:
        nyq = float(freqs[-1])
        total_energy = _band_energy(freqs, psd, 0.0, nyq)

        # fn band energy and kurtosis.
        if fn is not None and fn > 0 and total_energy > 0:
            bin_dx = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
            tol = max(fn * cfg.fn_band_halfwidth_frac, bin_dx)
            fn_mask = (freqs >= fn - tol) & (freqs <= fn + tol)
            if np.any(fn_mask):
                fn_energy = _band_mask_energy(freqs, psd, fn_mask)
                out[fn_e_key] = float(np.clip(fn_energy / total_energy, 0.0, 1.0))
                fn_psd_vals = psd[fn_mask]
                if fn_psd_vals.size >= 4:
                    try:
                        kurt = _scipy_kurtosis(
                            fn_psd_vals, fisher=True, bias=False, nan_policy="omit"
                        )
                        out[fn_k_key] = float(kurt)
                    except (ValueError, RuntimeError):
                        out[fn_k_key] = math.nan
                else:
                    out[fn_k_key] = math.nan
            else:
                out[fn_e_key] = math.nan
                out[fn_k_key] = math.nan
        else:
            out[fn_e_key] = math.nan
            out[fn_k_key] = math.nan

        # Sideband energy.
        if (
            fn is not None
            and fn > 0
            and tpf is not None
            and tpf > 0
            and total_energy > 0
        ):
            sb_mask = np.zeros_like(freqs, dtype=bool)
            bin_dx = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
            tol = max(tpf * cfg.sideband_halfwidth_frac, bin_dx)
            for k in range(1, cfg.sideband_count + 1):
                for sign in (-1.0, 1.0):
                    center = fn + sign * k * tpf
                    if center <= 0 or center >= nyq:
                        continue
                    sb_mask |= np.abs(freqs - center) <= tol
            if np.any(sb_mask):
                sb_energy = _band_mask_energy(freqs, psd, sb_mask)
                out[sb_key] = float(np.clip(sb_energy / total_energy, 0.0, 1.0))
            else:
                out[sb_key] = math.nan
        else:
            out[sb_key] = math.nan

    # Cyclostationarity (purely time-domain).
    out[cy_key] = _cyclostationarity_tpf(x, fs, tpf)


def _cyclostationarity_tpf(
    x: np.ndarray, fs: float | None, tpf: float | None
) -> float:
    if fs is None or tpf is None:
        return math.nan
    if not (math.isfinite(fs) and math.isfinite(tpf)) or fs <= 0 or tpf <= 0:
        return math.nan
    samples_per_period = int(round(fs / tpf))
    if samples_per_period < 2:
        return math.nan
    n_periods = x.size // samples_per_period
    if n_periods < 4:
        return math.nan
    usable = n_periods * samples_per_period
    folded = x[:usable].reshape(n_periods, samples_per_period)
    mean_over_periods = np.mean(folded, axis=0)
    var_over_periods = np.var(folded, axis=0)
    num = float(np.var(mean_over_periods))
    den = float(np.mean(var_over_periods))
    if den <= 0 or not math.isfinite(den):
        return math.nan
    return num / den


# ---------------------------------------------------------------------------
# Cross-channel coherence family
# ---------------------------------------------------------------------------

def _fill_cross_coherence(
    out: dict[str, float],
    signals: dict[str, np.ndarray],
    fs: float | None,
    tpf: float | None,
    cfg: NoiseMetricsConfig,
) -> None:
    pairs = (
        ("Forcex", "Forcey"),
        ("xtpos", "ytpos"),
        ("Forcex", "xtpos"),
    )
    pair_values: list[float] = []
    for a, b in pairs:
        key = f"cross_coh_{a}_{b}"
        value = _pair_coherence(signals.get(a), signals.get(b), fs, tpf, cfg)
        out[key] = value
        if math.isfinite(value):
            pair_values.append(value)

    if pair_values:
        out["cross_coh_mean_pairwise"] = float(np.mean(pair_values))
    else:
        out["cross_coh_mean_pairwise"] = math.nan


def _pair_coherence(
    a: np.ndarray | None,
    b: np.ndarray | None,
    fs: float | None,
    tpf: float | None,
    cfg: NoiseMetricsConfig,
) -> float:
    if a is None or b is None or fs is None:
        return math.nan
    # Build a JOINT finite mask so both channels are trimmed at the same
    # sample indices.  Independent per-channel dropping of NaN/Inf values
    # would produce arrays of different lengths whose samples no longer
    # correspond to the same time instants, corrupting the coherence estimate.
    n_common = min(a.size, b.size)
    joint_mask = np.isfinite(a[:n_common]) & np.isfinite(b[:n_common])
    a_aligned = a[:n_common][joint_mask]
    b_aligned = b[:n_common][joint_mask]
    n = a_aligned.size
    if n < 16:
        return math.nan
    try:
        nperseg = choose_nperseg(n, cfg.nperseg, min_nperseg=8)
    except ValueError:
        return math.nan
    noverlap = int(round(nperseg * cfg.overlap_fraction))
    noverlap = max(0, min(noverlap, nperseg - 1))
    try:
        freqs, cxy = _scipy_coherence(
            a_aligned,
            b_aligned,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
        )
    except Exception:
        return math.nan
    freqs = np.asarray(freqs, dtype=float)
    cxy = np.asarray(cxy, dtype=float)
    if freqs.size == 0:
        return math.nan
    nyq = float(freqs[-1])
    if tpf is not None and tpf > 0:
        band_max = float(min(cfg.coherence_band_tpf_multiplier * tpf, nyq))
    else:
        band_max = nyq
    if band_max <= 0:
        return math.nan
    mask = (freqs >= 0.0) & (freqs <= band_max)
    if not np.any(mask):
        return math.nan
    value = float(np.mean(cxy[mask]))
    if not math.isfinite(value):
        return math.nan
    return float(np.clip(value, 0.0, 1.0))
