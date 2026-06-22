#!/usr/bin/env python3
"""Local exploratory comparison of two filter candidates for Phase 2.5.

Synthesizes a milling-style signal with known forced-cutting energy at the
tooth-pass frequency (TPF) + harmonics, structured chatter near the modal
natural frequency `fn`, and added broadband Gaussian noise. Then applies two
candidate denoisers:

1. Butterworth bandpass over [TPF/2, ~5 fn] -- removes broadband energy outside
   the physics band but keeps the chatter region.
2. Wavelet soft-threshold denoising (db4, Donoho universal threshold). The MAD
   sigma is estimated from the level-1 detail coefficients, threshold is
   ``sigma * sqrt(2 log N)``, soft thresholding is applied to every detail
   level, the approximation level is kept untouched.

For each variant the script computes the Phase 2.5 metric vector via
``pri2m.noise.metrics.compute_file_metrics`` and prints a side-by-side table
with the per-channel results for the metrics that matter most:

- wavelet MAD sigma (lower = less noise)
- PSD spectral entropy (lower = more peaked = more structure preserved)
- TPF energy ratio (proxy via 1 - off-harmonic ratio)
- chatter fn-band energy (preservation target)

This is not a benchmark, just a sanity check on whether either candidate is
worth wiring into the production filter pipeline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pri2m.noise.metrics import (  # noqa: E402
    NoiseMetricsConfig,
    SIGNALS,
    compute_file_metrics,
)


# ---------- Signal synthesis ----------

@dataclass
class SyntheticParams:
    fs: float = 20_000.0
    duration_s: float = 2.0
    tpf: float = 200.0
    fn: float = 1_500.0
    chatter_amp: float = 1.0
    noise_std: float = 0.5
    seed: int = 20260526


def synthesize_signals(params: SyntheticParams) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(params.seed)
    n = int(params.fs * params.duration_s)
    t = np.arange(n) / params.fs
    forced = (
        1.0 * np.sin(2 * np.pi * params.tpf * t)
        + 0.3 * np.sin(2 * np.pi * 2 * params.tpf * t)
        + 0.1 * np.sin(2 * np.pi * 3 * params.tpf * t)
    )
    chatter = params.chatter_amp * np.sin(2 * np.pi * params.fn * t)
    # Add sidebands at fn +/- k*tpf to mimic regenerative chatter signature.
    for k in (1, 2):
        chatter += 0.3 * params.chatter_amp * np.sin(2 * np.pi * (params.fn + k * params.tpf) * t)
        chatter += 0.3 * params.chatter_amp * np.sin(2 * np.pi * (params.fn - k * params.tpf) * t)
    shared = forced + chatter
    bundle = {
        "xtpos": shared + params.noise_std * rng.standard_normal(n),
        "ytpos": 0.8 * shared + params.noise_std * rng.standard_normal(n),
        "Forcex": 1.2 * shared + params.noise_std * rng.standard_normal(n),
        "Forcey": 0.9 * shared + params.noise_std * rng.standard_normal(n),
    }
    return bundle, t


# ---------- Filter candidates ----------

def butterworth_bandpass(
    x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass via sos + filtfilt."""
    nyq = fs / 2.0
    low = max(1e-6, low_hz / nyq)
    high = min(1.0 - 1e-6, high_hz / nyq)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def wavelet_soft_denoise(
    x: np.ndarray, wavelet: str = "db4", level: int | None = None
) -> np.ndarray:
    """Donoho universal-threshold soft denoising on detail coefficients."""
    if level is None:
        # Be conservative -- 4 levels is plenty for fs=20 kHz over 2 s.
        level = min(4, pywt.dwt_max_level(len(x), wavelet))
    coeffs = pywt.wavedec(x, wavelet, level=level)
    d1 = coeffs[-1]
    sigma = float(np.median(np.abs(d1))) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(len(x)))
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, threshold, mode="soft"))
    reconstructed = pywt.waverec(new_coeffs, wavelet)
    return reconstructed[: len(x)]


# ---------- Driver ----------

def apply_filter(
    bundle: dict[str, np.ndarray],
    kind: str,
    fs: float,
    tpf: float,
    fn: float,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for signal_name, x in bundle.items():
        if kind == "butterworth":
            out[signal_name] = butterworth_bandpass(
                x, fs=fs, low_hz=tpf / 2.0, high_hz=5.0 * fn, order=4
            )
        elif kind == "wavelet":
            out[signal_name] = wavelet_soft_denoise(x)
        else:
            raise ValueError(f"unknown filter kind: {kind}")
    return out


def metric_summary(
    bundle: dict[str, np.ndarray], *, fs: float, tpf: float, fn: float
) -> dict[str, float]:
    config = NoiseMetricsConfig()
    metrics = compute_file_metrics(bundle, tpf=tpf, fn=fn, fs=fs, config=config)
    # Roll up per-signal metrics into single scalars for the report.
    def _mean(prefix: str) -> float:
        vals = [metrics[f"{prefix}_{s}"] for s in SIGNALS if f"{prefix}_{s}" in metrics]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")
    return {
        "wavelet_mad_sigma": _mean("wavelet_mad_sigma"),
        "psd_spectral_entropy": _mean("psd_spectral_entropy"),
        "psd_off_harmonic_ratio": _mean("psd_off_harmonic_ratio"),
        "chatter_fn_band_energy_max": float(np.nanmax([
            metrics.get(f"chatter_fn_band_energy_{s}", np.nan) for s in SIGNALS
        ])),
        "cross_coh_mean_pairwise": float(metrics.get("cross_coh_mean_pairwise", float("nan"))),
    }


def main() -> int:
    params = SyntheticParams()
    clean_params = SyntheticParams(noise_std=0.0, seed=params.seed + 1)
    clean_bundle, _ = synthesize_signals(clean_params)
    noisy_bundle, _ = synthesize_signals(params)
    butter_bundle = apply_filter(noisy_bundle, "butterworth", fs=params.fs, tpf=params.tpf, fn=params.fn)
    wavelet_bundle = apply_filter(noisy_bundle, "wavelet", fs=params.fs, tpf=params.tpf, fn=params.fn)

    rows = {
        "clean (reference)": metric_summary(clean_bundle, fs=params.fs, tpf=params.tpf, fn=params.fn),
        "noisy (baseline)": metric_summary(noisy_bundle, fs=params.fs, tpf=params.tpf, fn=params.fn),
        "butterworth bandpass": metric_summary(butter_bundle, fs=params.fs, tpf=params.tpf, fn=params.fn),
        "wavelet soft-thresh": metric_summary(wavelet_bundle, fs=params.fs, tpf=params.tpf, fn=params.fn),
    }

    metric_order = [
        "wavelet_mad_sigma",
        "psd_spectral_entropy",
        "psd_off_harmonic_ratio",
        "chatter_fn_band_energy_max",
        "cross_coh_mean_pairwise",
    ]
    col_w = 22
    print(f"{'variant':<{col_w}} " + " ".join(f"{m:>26}" for m in metric_order))
    print("-" * (col_w + 1 + 27 * len(metric_order)))
    for name, row in rows.items():
        cells = " ".join(f"{row[m]:>26.5g}" for m in metric_order)
        print(f"{name:<{col_w}} {cells}")

    # Interpretation hints
    baseline = rows["noisy (baseline)"]
    print()
    for name in ("butterworth bandpass", "wavelet soft-thresh"):
        row = rows[name]
        wavelet_drop = (
            (baseline["wavelet_mad_sigma"] - row["wavelet_mad_sigma"])
            / baseline["wavelet_mad_sigma"]
        )
        chatter_change = (
            (row["chatter_fn_band_energy_max"] - baseline["chatter_fn_band_energy_max"])
            / max(1e-12, baseline["chatter_fn_band_energy_max"])
        )
        tpf_proxy_baseline = 1.0 - baseline["psd_off_harmonic_ratio"]
        tpf_proxy_variant = 1.0 - row["psd_off_harmonic_ratio"]
        tpf_change = (tpf_proxy_variant - tpf_proxy_baseline) / max(1e-12, tpf_proxy_baseline)
        print(
            f"{name}: noise-level drop {wavelet_drop:+.1%}, "
            f"chatter-band change {chatter_change:+.1%}, "
            f"TPF-proxy change {tpf_change:+.1%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
