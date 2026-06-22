"""Welch PSD summaries and spectral divergence helpers for PRI2M.

The functions in this module are intentionally data-location agnostic. They
operate on in-memory signals and metadata so importing them does not require a
Great Lakes mount or the original HDF5 files.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

try:  # SciPy is preferred when available, but the module must remain portable.
    from scipy.signal import welch as _scipy_welch
except Exception:  # pragma: no cover - exercised in environments without SciPy.
    _scipy_welch = None

try:
    from scipy.special import rel_entr as _scipy_rel_entr
except Exception:  # pragma: no cover - exercised in environments without SciPy.
    _scipy_rel_entr = None


SIGNAL_NAMES: tuple[str, ...] = ("xtpos", "ytpos", "Forcex", "Forcey")
DEFAULT_NPERSEG = 2048
DEFAULT_MIN_NPERSEG = 256
DEFAULT_OVERLAP_FRACTION = 0.5
DEFAULT_ANALYSIS_TPF_MULTIPLIER = 5.0
DEFAULT_EPS = 1.0e-12


@dataclass(frozen=True)
class WelchConfig:
    """Configuration for Welch PSD estimates.

    Short signals are capped to the largest valid power-of-two window not larger
    than the sample count. Signals shorter than ``min_nperseg`` are rejected so
    the PSD estimate does not silently fall back to very noisy tiny windows.
    """

    nperseg: int = DEFAULT_NPERSEG
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION
    window: str = "hann"
    detrend: str = "constant"
    min_nperseg: int = DEFAULT_MIN_NPERSEG
    analysis_tpf_multiplier: float = DEFAULT_ANALYSIS_TPF_MULTIPLIER
    eps: float = DEFAULT_EPS


@dataclass(frozen=True)
class AnomalyThresholds:
    """Thresholds for lightweight PSD sanity checks.

    These are intentionally conservative review flags, not model labels.
    """

    harmonic_tolerance_fraction: float = 0.05
    fn_tolerance_fraction: float = 0.08
    tpf_fraction_min: float = 0.12
    fn_fraction_min: float = 0.08
    peak_fraction_min: float = 0.05
    low_broadband_max: float = 0.45
    broadband_lift_min: float = 0.65
    low_entropy_max: float = 0.62
    high_entropy_min: float = 0.72


@dataclass(frozen=True)
class PsdEstimate:
    """Welch PSD result for one signal."""

    signal_name: str
    frequencies: np.ndarray
    psd: np.ndarray
    fs: float
    n_samples: int
    nperseg: int
    noverlap: int
    window: str
    detrend: str
    used_scipy: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "signal": self.signal_name,
            "fs": self.fs,
            "n_samples": self.n_samples,
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "window": self.window,
            "detrend": self.detrend,
            "used_scipy": self.used_scipy,
            "frequency_bins": int(self.frequencies.size),
        }


def scipy_available() -> bool:
    """Return whether SciPy-backed Welch/KL routines are importable."""

    return _scipy_welch is not None


def choose_nperseg(
    n_samples: int,
    requested: int = DEFAULT_NPERSEG,
    min_nperseg: int = DEFAULT_MIN_NPERSEG,
) -> int:
    """Choose a valid power-of-two Welch window for ``n_samples``.

    The default is 2048. Shorter signals are capped to the largest power of two
    no larger than both ``requested`` and ``n_samples``. A ``ValueError`` is
    raised when the signal is shorter than the minimum supported window.
    """

    n_samples = int(n_samples)
    requested = int(requested)
    min_nperseg = int(min_nperseg)
    if n_samples < min_nperseg:
        raise ValueError(
            f"need at least {min_nperseg} samples for Welch PSD; got {n_samples}"
        )
    target = min(n_samples, requested)
    nperseg = 1 << (target.bit_length() - 1)
    if nperseg < min_nperseg:
        raise ValueError(
            f"largest power-of-two window {nperseg} is below minimum {min_nperseg}"
        )
    return nperseg


def welch_psd(
    signal: Sequence[float] | np.ndarray,
    fs: float,
    signal_name: str = "",
    config: WelchConfig | None = None,
) -> PsdEstimate:
    """Estimate one-sided Welch PSD with PRI2M defaults.

    Defaults are Hann window, 2048 samples per segment, 50 percent overlap, and
    constant detrending. SciPy is used when installed; otherwise a NumPy Welch
    implementation with density scaling is used.
    """

    cfg = config or WelchConfig()
    x = _as_1d_finite(signal)
    fs = _validate_positive_float(fs, "fs")
    nperseg = choose_nperseg(x.size, cfg.nperseg, cfg.min_nperseg)
    noverlap = int(round(nperseg * cfg.overlap_fraction))
    noverlap = max(0, min(noverlap, nperseg - 1))

    if _scipy_welch is not None:
        frequencies, psd = _scipy_welch(
            x,
            fs=fs,
            window=cfg.window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=cfg.detrend,
            scaling="density",
            return_onesided=True,
        )
        used_scipy = True
    else:
        frequencies, psd = _numpy_welch(
            x=x,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=cfg.window,
            detrend=cfg.detrend,
        )
        used_scipy = False

    return PsdEstimate(
        signal_name=signal_name,
        frequencies=np.asarray(frequencies, dtype=float),
        psd=np.asarray(psd, dtype=float),
        fs=fs,
        n_samples=int(x.size),
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        window=cfg.window,
        detrend=cfg.detrend,
        used_scipy=used_scipy,
    )


def summarize_signal_psd(
    estimate: PsdEstimate,
    tpf: float | None,
    fn: float | None = None,
    thresholds: AnomalyThresholds | None = None,
    dataset: int | None = None,
    label: int | None = None,
    config: WelchConfig | None = None,
) -> dict[str, Any]:
    """Summarize one PSD inside the default analysis band, 0 to 5*tpf."""

    cfg = config or WelchConfig()
    threshold_values = thresholds or AnomalyThresholds()
    band_freq, band_psd = analysis_band(
        estimate.frequencies,
        estimate.psd,
        tpf=tpf,
        multiplier=cfg.analysis_tpf_multiplier,
    )
    summary = estimate.metadata()
    summary.update(
        {
            "status": "ok",
            "tpf": _optional_float(tpf),
            "fn": _optional_float(fn),
            "analysis_band_min_hz": 0.0,
            "analysis_band_max_hz": _band_max_from_tpf(
                estimate.frequencies, tpf, cfg.analysis_tpf_multiplier
            ),
            "analysis_band_bins": int(band_freq.size),
        }
    )
    if band_freq.size == 0 or band_psd.size == 0:
        summary.update(
            {
                "status": "empty_analysis_band",
                "sanity_class": "indeterminate_review",
                "inspection_tags": _inspection_tags(dataset, label, False, False),
            }
        )
        return summary

    masses = spectral_mass(band_freq, band_psd)
    total_power = float(np.sum(masses))
    distribution = normalize_distribution(masses, eps=cfg.eps)
    dominant_idx = int(np.argmax(distribution))
    dominant_freq = float(band_freq[dominant_idx])
    peak_fraction = float(distribution[dominant_idx])
    entropy = normalized_entropy(distribution, eps=cfg.eps)

    tpf_value = _optional_float(tpf)
    fn_value = _optional_float(fn)
    tpf_harmonic_fraction = math.nan
    tpf_fundamental_fraction = math.nan
    broadband_fraction = math.nan
    nearest_harmonic = None
    harmonic_error = math.nan
    dominant_freq_to_tpf = math.nan

    if tpf_value is not None and tpf_value > 0:
        harmonic_mask = _harmonic_mask(
            band_freq,
            tpf=tpf_value,
            max_frequency=float(band_freq[-1]),
            tolerance_fraction=threshold_values.harmonic_tolerance_fraction,
        )
        tpf_harmonic_fraction = _fraction_for_mask(masses, harmonic_mask)
        tpf_fundamental_fraction = _fraction_for_mask(
            masses,
            _center_mask(
                band_freq,
                center=tpf_value,
                tolerance=max(_frequency_resolution(band_freq), tpf_value * 0.05),
            ),
        )
        broadband_fraction = float(max(0.0, 1.0 - tpf_harmonic_fraction))
        dominant_freq_to_tpf = float(dominant_freq / tpf_value)
        nearest_harmonic = int(max(1, round(dominant_freq_to_tpf)))
        harmonic_error = float(abs(dominant_freq_to_tpf - nearest_harmonic))

    fn_fraction = math.nan
    dominant_fn_proximity = math.nan
    if fn_value is not None and fn_value > 0:
        fn_fraction = _fraction_for_mask(
            masses,
            _center_mask(
                band_freq,
                center=fn_value,
                tolerance=max(
                    _frequency_resolution(band_freq),
                    fn_value * threshold_values.fn_tolerance_fraction,
                ),
            ),
        )
        dominant_fn_proximity = float(abs(dominant_freq - fn_value) / fn_value)

    summary.update(
        {
            "total_band_power": total_power,
            "dominant_freq_hz": dominant_freq,
            "dominant_freq_to_tpf": dominant_freq_to_tpf,
            "nearest_tpf_harmonic": nearest_harmonic,
            "tpf_harmonic_error": harmonic_error,
            "peak_power_fraction": peak_fraction,
            "tpf_power_fraction": tpf_fundamental_fraction,
            "tpf_harmonic_power_fraction": tpf_harmonic_fraction,
            "fn_power_fraction": fn_fraction,
            "dominant_fn_proximity": dominant_fn_proximity,
            "broadband_power_fraction": broadband_fraction,
            "normalized_spectral_entropy": entropy,
        }
    )
    summary.update(
        classify_anomaly_sanity(
            summary,
            thresholds=threshold_values,
            dataset=dataset,
            label=label,
        )
    )
    return summary


def summarize_signal_map(
    signals: Mapping[str, Sequence[float] | np.ndarray],
    fs: float,
    tpf: float | None,
    fn: float | None = None,
    signal_names: Sequence[str] = SIGNAL_NAMES,
    config: WelchConfig | None = None,
    thresholds: AnomalyThresholds | None = None,
    dataset: int | None = None,
    label: int | None = None,
) -> list[dict[str, Any]]:
    """Compute PSD summaries for the standard PRI2M signal channels."""

    rows: list[dict[str, Any]] = []
    cfg = config or WelchConfig()
    for signal_name in signal_names:
        if signal_name not in signals:
            rows.append(
                {
                    "signal": signal_name,
                    "status": "missing_signal",
                    "tpf": _optional_float(tpf),
                    "fn": _optional_float(fn),
                    "sanity_class": "indeterminate_review",
                    "inspection_tags": _inspection_tags(dataset, label, False, False),
                }
            )
            continue
        try:
            estimate = welch_psd(
                signals[signal_name],
                fs=fs,
                signal_name=signal_name,
                config=cfg,
            )
            rows.append(
                summarize_signal_psd(
                    estimate,
                    tpf=tpf,
                    fn=fn,
                    thresholds=thresholds,
                    dataset=dataset,
                    label=label,
                    config=cfg,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "signal": signal_name,
                    "status": "error",
                    "error": str(exc),
                    "tpf": _optional_float(tpf),
                    "fn": _optional_float(fn),
                    "sanity_class": "indeterminate_review",
                    "inspection_tags": _inspection_tags(dataset, label, False, False),
                }
            )
    return rows


def spectral_divergence_summary(
    reference_frequencies: Sequence[float] | np.ndarray,
    reference_psd: Sequence[float] | np.ndarray,
    candidate_frequencies: Sequence[float] | np.ndarray,
    candidate_psd: Sequence[float] | np.ndarray,
    band_max: float | None = None,
    eps: float = DEFAULT_EPS,
    max_bins: int = 4096,
) -> dict[str, Any]:
    """Compare two spectra with JSD and directional KL.

    Jensen-Shannon divergence is returned in bits, so it is bounded to [0, 1].
    Directional KL terms are also in bits and intended for post-hoc reporting.
    """

    ref_freq, ref_psd = _finite_xy(reference_frequencies, reference_psd)
    cand_freq, cand_psd = _finite_xy(candidate_frequencies, candidate_psd)
    grid = common_frequency_grid(ref_freq, cand_freq, band_max=band_max, max_bins=max_bins)
    if grid.size == 0:
        return {
            "status": "empty_common_band",
            "js_divergence": math.nan,
            "kl_candidate_to_reference": math.nan,
            "kl_reference_to_candidate": math.nan,
            "n_bins": 0,
            "band_max_hz": _optional_float(band_max),
        }

    ref_interp = np.interp(grid, ref_freq, ref_psd, left=0.0, right=0.0)
    cand_interp = np.interp(grid, cand_freq, cand_psd, left=0.0, right=0.0)
    ref_dist = normalize_distribution(spectral_mass(grid, ref_interp), eps=eps)
    cand_dist = normalize_distribution(spectral_mass(grid, cand_interp), eps=eps)

    return {
        "status": "ok",
        "js_divergence": jensen_shannon_divergence(cand_dist, ref_dist, eps=eps),
        "kl_candidate_to_reference": kl_divergence(cand_dist, ref_dist, eps=eps),
        "kl_reference_to_candidate": kl_divergence(ref_dist, cand_dist, eps=eps),
        "n_bins": int(grid.size),
        "band_max_hz": float(grid[-1]),
    }


def compare_signal_maps(
    reference_signals: Mapping[str, Sequence[float] | np.ndarray],
    candidate_signals: Mapping[str, Sequence[float] | np.ndarray],
    fs: float,
    tpf: float | None,
    signal_names: Sequence[str] = SIGNAL_NAMES,
    config: WelchConfig | None = None,
) -> list[dict[str, Any]]:
    """Compute per-signal PSD divergence between two signal bundles."""

    cfg = config or WelchConfig()
    band_max = _band_max_from_tpf(None, tpf, cfg.analysis_tpf_multiplier)
    rows: list[dict[str, Any]] = []
    for signal_name in signal_names:
        if signal_name not in reference_signals or signal_name not in candidate_signals:
            rows.append({"signal": signal_name, "status": "missing_signal"})
            continue
        try:
            ref = welch_psd(
                reference_signals[signal_name],
                fs=fs,
                signal_name=signal_name,
                config=cfg,
            )
            cand = welch_psd(
                candidate_signals[signal_name],
                fs=fs,
                signal_name=signal_name,
                config=cfg,
            )
            row = {"signal": signal_name}
            row.update(
                spectral_divergence_summary(
                    ref.frequencies,
                    ref.psd,
                    cand.frequencies,
                    cand.psd,
                    band_max=band_max,
                    eps=cfg.eps,
                )
            )
            rows.append(row)
        except Exception as exc:
            rows.append({"signal": signal_name, "status": "error", "error": str(exc)})
    return rows


def kl_divergence(
    p: Sequence[float] | np.ndarray,
    q: Sequence[float] | np.ndarray,
    eps: float = DEFAULT_EPS,
    base: float = 2.0,
) -> float:
    """Directional KL divergence KL(p || q)."""

    p_dist = normalize_distribution(p, eps=eps)
    q_dist = normalize_distribution(q, eps=eps)
    if _scipy_rel_entr is not None:
        value = float(np.sum(_scipy_rel_entr(p_dist, q_dist)))
    else:
        value = float(np.sum(p_dist * np.log(p_dist / q_dist)))
    if base and base != math.e:
        value /= math.log(base)
    return max(0.0, value)


def jensen_shannon_divergence(
    p: Sequence[float] | np.ndarray,
    q: Sequence[float] | np.ndarray,
    eps: float = DEFAULT_EPS,
    base: float = 2.0,
) -> float:
    """Bounded Jensen-Shannon divergence between two distributions.

    With the default base 2, the result is measured in bits and lies in [0, 1].
    """

    p_dist = normalize_distribution(p, eps=eps)
    q_dist = normalize_distribution(q, eps=eps)
    midpoint = 0.5 * (p_dist + q_dist)
    value = 0.5 * kl_divergence(p_dist, midpoint, eps=eps, base=base)
    value += 0.5 * kl_divergence(q_dist, midpoint, eps=eps, base=base)
    if base == 2.0:
        return min(1.0, max(0.0, value))
    return max(0.0, value)


def classify_anomaly_sanity(
    metrics: Mapping[str, Any],
    thresholds: AnomalyThresholds | None = None,
    dataset: int | None = None,
    label: int | None = None,
) -> dict[str, Any]:
    """Classify PSD metrics into plan-level sanity buckets.

    The buckets mirror the Phase 2 plan language:
    - stable clean: tooth-pass-frequency peak and low broadband energy.
    - unstable: natural-frequency/chatter peak or broadband lift.
    """

    th = thresholds or AnomalyThresholds()
    harmonic = _optional_int(metrics.get("nearest_tpf_harmonic"))
    harmonic_error = _optional_float(metrics.get("tpf_harmonic_error"))
    peak_fraction = _finite_or_nan(metrics.get("peak_power_fraction"))
    tpf_fraction = _finite_or_nan(metrics.get("tpf_harmonic_power_fraction"))
    fn_fraction = _finite_or_nan(metrics.get("fn_power_fraction"))
    fn_proximity = _finite_or_nan(metrics.get("dominant_fn_proximity"))
    broadband_fraction = _finite_or_nan(metrics.get("broadband_power_fraction"))
    entropy = _finite_or_nan(metrics.get("normalized_spectral_entropy"))

    has_tpf_peak = (
        harmonic is not None
        and 1 <= harmonic <= 5
        and math.isfinite(harmonic_error)
        and harmonic_error <= th.harmonic_tolerance_fraction
        and math.isfinite(peak_fraction)
        and peak_fraction >= th.peak_fraction_min
    )
    has_tpf_energy = math.isfinite(tpf_fraction) and tpf_fraction >= th.tpf_fraction_min
    low_broadband = (
        math.isfinite(broadband_fraction)
        and broadband_fraction <= th.low_broadband_max
        and (not math.isfinite(entropy) or entropy <= th.low_entropy_max)
    )
    stable_clean_signature = bool((has_tpf_peak or has_tpf_energy) and low_broadband)

    has_fn_peak = (
        (math.isfinite(fn_fraction) and fn_fraction >= th.fn_fraction_min)
        or (math.isfinite(fn_proximity) and fn_proximity <= th.fn_tolerance_fraction)
    )
    broadband_lift = (
        (math.isfinite(broadband_fraction) and broadband_fraction >= th.broadband_lift_min)
        or (math.isfinite(entropy) and entropy >= th.high_entropy_min)
    )
    unstable_signature = bool(has_fn_peak or broadband_lift)

    if stable_clean_signature and not unstable_signature:
        sanity_class = "stable_clean_tpf_peak_low_broadband"
    elif unstable_signature and not stable_clean_signature:
        sanity_class = "unstable_fn_chatter_peak_or_broadband_lift"
    elif stable_clean_signature and unstable_signature:
        sanity_class = "mixed_tpf_with_chatter_or_broadband_review"
    else:
        sanity_class = "indeterminate_review"

    return {
        "sanity_class": sanity_class,
        "stable_clean_signature": stable_clean_signature,
        "unstable_signature": unstable_signature,
        "inspection_tags": _inspection_tags(
            dataset=dataset,
            label=label,
            stable_clean_signature=stable_clean_signature,
            unstable_signature=unstable_signature,
        ),
    }


def classify_summary_feature_row(
    row: Mapping[str, Any],
    signal_name: str,
    thresholds: AnomalyThresholds | None = None,
) -> dict[str, Any]:
    """Apply the same sanity labels to existing summary CSV spectral columns."""

    th = thresholds or AnomalyThresholds()
    freq_ratio = _optional_float(row.get(f"{signal_name}_freq_ratio_tpf"))
    harmonic = None
    harmonic_error = math.nan
    if freq_ratio is not None and freq_ratio > 0:
        harmonic = int(max(1, round(freq_ratio)))
        harmonic_error = float(abs(freq_ratio - harmonic))

    harmonic_fraction = _optional_float(row.get(f"{signal_name}_harmonic_energy_ratio"))
    if harmonic_fraction is not None and harmonic_fraction > 1.0:
        harmonic_fraction = math.nan
    fn_proximity = _optional_float(row.get(f"{signal_name}_fn_proximity"))
    entropy = _optional_float(row.get(f"{signal_name}_spectral_entropy"))
    normalized_entropy = math.nan
    if entropy is not None and math.isfinite(entropy):
        normalized_entropy = min(1.0, entropy / 5.0)
    broadband_fraction = (
        float(max(0.0, 1.0 - harmonic_fraction))
        if harmonic_fraction is not None and math.isfinite(harmonic_fraction)
        else math.nan
    )
    metrics = {
        "nearest_tpf_harmonic": harmonic,
        "tpf_harmonic_error": harmonic_error,
        "peak_power_fraction": (
            harmonic_fraction
            if harmonic_fraction is not None and math.isfinite(harmonic_fraction)
            else math.nan
        ),
        "tpf_harmonic_power_fraction": (
            harmonic_fraction
            if harmonic_fraction is not None and math.isfinite(harmonic_fraction)
            else math.nan
        ),
        "fn_power_fraction": math.nan,
        "dominant_fn_proximity": (
            fn_proximity if fn_proximity is not None else math.nan
        ),
        "broadband_power_fraction": broadband_fraction,
        "normalized_spectral_entropy": normalized_entropy,
    }
    dataset = _optional_int(row.get("dataset"))
    label = _optional_int(row.get("label"))
    result = classify_anomaly_sanity(
        metrics,
        thresholds=th,
        dataset=dataset,
        label=label,
    )
    result.update(
        {
            "signal": signal_name,
            "dominant_freq_to_tpf": freq_ratio,
            "nearest_tpf_harmonic": harmonic,
            "tpf_harmonic_error": harmonic_error,
            "summary_harmonic_energy_ratio": harmonic_fraction,
            "summary_spectral_entropy": entropy,
        }
    )
    return result


def analysis_band(
    frequencies: Sequence[float] | np.ndarray,
    psd: Sequence[float] | np.ndarray,
    tpf: float | None,
    multiplier: float = DEFAULT_ANALYSIS_TPF_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PSD values in the default 0 to 5*tpf analysis band."""

    freq, values = _finite_xy(frequencies, psd)
    if freq.size == 0:
        return freq, values
    band_max = _band_max_from_tpf(freq, tpf, multiplier)
    mask = (freq >= 0.0) & (freq <= band_max)
    return freq[mask], values[mask]


def spectral_mass(
    frequencies: Sequence[float] | np.ndarray,
    psd: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Convert density-like PSD values to nonnegative bin masses."""

    freq = np.asarray(frequencies, dtype=float)
    values = np.asarray(psd, dtype=float)
    values = np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if values.size == 0:
        return values
    if freq.size != values.size:
        raise ValueError("frequencies and psd must have the same length")
    if values.size == 1:
        widths = np.ones_like(values)
    else:
        widths = np.empty_like(values)
        widths[0] = max(freq[1] - freq[0], DEFAULT_EPS)
        widths[-1] = max(freq[-1] - freq[-2], DEFAULT_EPS)
        widths[1:-1] = np.maximum((freq[2:] - freq[:-2]) * 0.5, DEFAULT_EPS)
    return values * widths


def normalize_distribution(
    values: Sequence[float] | np.ndarray,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Return a finite probability vector for divergence calculations."""

    arr = np.asarray(values, dtype=float)
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if arr.size == 0:
        raise ValueError("cannot normalize an empty distribution")
    arr = arr + float(eps)
    total = float(np.sum(arr))
    if not math.isfinite(total) or total <= 0.0:
        return np.full(arr.shape, 1.0 / arr.size, dtype=float)
    return arr / total


def normalized_entropy(
    distribution: Sequence[float] | np.ndarray,
    eps: float = DEFAULT_EPS,
) -> float:
    """Return Shannon entropy normalized to [0, 1]."""

    probs = normalize_distribution(distribution, eps=eps)
    if probs.size <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log2(probs)))
    return min(1.0, max(0.0, entropy / math.log2(probs.size)))


def common_frequency_grid(
    reference_frequencies: Sequence[float] | np.ndarray,
    candidate_frequencies: Sequence[float] | np.ndarray,
    band_max: float | None = None,
    max_bins: int = 4096,
) -> np.ndarray:
    """Build a shared nonnegative frequency grid for two spectra."""

    ref = np.asarray(reference_frequencies, dtype=float)
    cand = np.asarray(candidate_frequencies, dtype=float)
    ref = ref[np.isfinite(ref) & (ref >= 0.0)]
    cand = cand[np.isfinite(cand) & (cand >= 0.0)]
    if ref.size == 0 or cand.size == 0:
        return np.asarray([], dtype=float)
    upper = min(float(np.max(ref)), float(np.max(cand)))
    if band_max is not None and math.isfinite(float(band_max)):
        upper = min(upper, float(band_max))
    if upper < 0.0:
        return np.asarray([], dtype=float)
    grid = np.unique(np.concatenate([ref[ref <= upper], cand[cand <= upper]]))
    if grid.size > max_bins:
        grid = np.linspace(0.0, upper, max_bins)
    return grid


def _numpy_welch(
    x: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    window: str,
    detrend: str,
) -> tuple[np.ndarray, np.ndarray]:
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("noverlap must be smaller than nperseg")
    starts = np.arange(0, x.size - nperseg + 1, step, dtype=int)
    if starts.size == 0:
        starts = np.asarray([0], dtype=int)
    segments = np.stack([x[start : start + nperseg] for start in starts])
    if detrend == "constant":
        segments = segments - np.mean(segments, axis=1, keepdims=True)
    elif detrend not in (None, False, "none"):
        raise ValueError(f"unsupported detrend mode for NumPy fallback: {detrend}")
    win = _window_values(window, nperseg)
    windowed = segments * win[None, :]
    spectrum = np.fft.rfft(windowed, axis=1)
    scale = 1.0 / (fs * float(np.sum(win * win)))
    periodograms = (np.abs(spectrum) ** 2) * scale
    if nperseg % 2 == 0:
        periodograms[:, 1:-1] *= 2.0
    else:
        periodograms[:, 1:] *= 2.0
    psd = np.mean(periodograms, axis=0)
    frequencies = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return frequencies, psd


def _window_values(window: str, nperseg: int) -> np.ndarray:
    normalized = window.lower()
    if normalized in {"hann", "hanning"}:
        return np.hanning(nperseg)
    if normalized in {"boxcar", "rect", "rectangular", "ones"}:
        return np.ones(nperseg, dtype=float)
    raise ValueError(f"unsupported window for NumPy fallback: {window}")


def _as_1d_finite(signal: Sequence[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("signal has no finite samples")
    return x


def _finite_xy(
    frequencies: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    freq = np.asarray(frequencies, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if freq.size != y.size:
        raise ValueError("frequencies and values must have the same length")
    mask = np.isfinite(freq) & np.isfinite(y)
    freq = freq[mask]
    y = y[mask]
    order = np.argsort(freq)
    return freq[order], y[order]


def _validate_positive_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return parsed


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _optional_int(value: Any) -> int | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _finite_or_nan(value: Any) -> float:
    parsed = _optional_float(value)
    return parsed if parsed is not None else math.nan


def _band_max_from_tpf(
    frequencies: Sequence[float] | np.ndarray | None,
    tpf: float | None,
    multiplier: float,
) -> float:
    tpf_value = _optional_float(tpf)
    if tpf_value is not None and tpf_value > 0:
        band_max = float(multiplier) * tpf_value
    elif frequencies is not None:
        freq = np.asarray(frequencies, dtype=float)
        finite = freq[np.isfinite(freq)]
        band_max = float(np.max(finite)) if finite.size else 0.0
    else:
        band_max = math.inf
    if frequencies is not None:
        freq = np.asarray(frequencies, dtype=float)
        finite = freq[np.isfinite(freq)]
        if finite.size:
            band_max = min(band_max, float(np.max(finite)))
    return band_max


def _frequency_resolution(frequencies: np.ndarray) -> float:
    if frequencies.size < 2:
        return DEFAULT_EPS
    diffs = np.diff(frequencies)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return DEFAULT_EPS
    return float(np.median(diffs))


def _center_mask(frequencies: np.ndarray, center: float, tolerance: float) -> np.ndarray:
    return np.abs(frequencies - float(center)) <= float(tolerance)


def _harmonic_mask(
    frequencies: np.ndarray,
    tpf: float,
    max_frequency: float,
    tolerance_fraction: float,
) -> np.ndarray:
    mask = np.zeros(frequencies.shape, dtype=bool)
    resolution = _frequency_resolution(frequencies)
    harmonic = 1
    while harmonic * tpf <= max_frequency + resolution:
        center = harmonic * tpf
        tolerance = max(resolution, center * tolerance_fraction)
        mask |= _center_mask(frequencies, center=center, tolerance=tolerance)
        harmonic += 1
    return mask


def _fraction_for_mask(masses: np.ndarray, mask: np.ndarray) -> float:
    total = float(np.sum(masses))
    if total <= 0.0 or not math.isfinite(total):
        return math.nan
    return float(np.sum(masses[mask]) / total)


def _inspection_tags(
    dataset: int | None,
    label: int | None,
    stable_clean_signature: bool,
    unstable_signature: bool,
) -> list[str]:
    tags: list[str] = []
    if dataset is not None and 44 <= int(dataset) <= 48:
        tags.append("inspect_44_48_overcalled_cut_window")
    if label == 0 and unstable_signature:
        tags.append("possible_overcall_stable_cut")
    if label == 1 and stable_clean_signature:
        tags.append("possible_undercall_unstable_cut")
    return tags
