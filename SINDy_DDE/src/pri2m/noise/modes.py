"""Phase 2.5 mode-label assignment.

This module turns a per-file metric vector (as produced by
``pri2m.noise.metrics.compute_file_metrics``) plus the metadata-derived
``b_over_blim`` into a coarse mode label and a confidence score.

The module is pure: no file I/O, no globals, no CLI. The decision rule is the
v1 rule documented in ``docs/phase2_5_noise_mode_filtering.md`` under
"Mode Assignment". The thresholds in :class:`ModeThresholds` are starting
points only; they are meant to be re-tuned during clean-1-to-60 calibration.
Do not hardcode dataset-specific behavior here.

The six labels in :data:`MODE_LABELS` are the v1 vocabulary. See the docs for
what each label means physically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


MODE_LABELS: tuple[str, ...] = (
    "tpf_dominated_clean",
    "near_boundary_transition",
    "chatter_dominated",
    "broadband_noise_dominated",
    "mixed_tpf_chatter",
    "low_speed_edge",
    "metrics_unavailable",
)


# Canonical per-signal metric family prefixes (the prefix before ``_<signal>``)
# that ``compute_file_metrics`` emits. Used to count NaN metrics for the
# "metrics_unavailable" guard.
_PER_SIGNAL_PREFIXES: tuple[str, ...] = (
    "psd_noise_floor_ratio",
    "psd_off_harmonic_ratio",
    "psd_spectral_flatness",
    "psd_spectral_entropy",
    "wavelet_mad_sigma",
    "autocorr_decay_lag_samples",
    "chatter_fn_band_energy",
    "chatter_fn_band_kurtosis",
    "chatter_sideband_energy",
    "chatter_cyclostationarity_tpf",
)

_CROSS_KEYS: tuple[str, ...] = (
    "cross_coh_Forcex_Forcey",
    "cross_coh_xtpos_ytpos",
    "cross_coh_Forcex_xtpos",
    "cross_coh_mean_pairwise",
)

_DEFAULT_SIGNALS: tuple[str, ...] = ("xtpos", "ytpos", "Forcex", "Forcey")


@dataclass(frozen=True)
class ModeThresholds:
    """Thresholds used by :func:`assign_mode`.

    Defaults are deliberate starting points: they are intended to be re-tuned
    during the clean-1-to-60 calibration pass described in the Phase 2.5 docs.
    The :func:`assign_mode` function never reads any global configuration; pass
    a custom instance to override.
    """

    # --- Legacy uncalibrated noise vote thresholds.
    # Production mode assignment should pass promoted_metrics.json from the
    # mirror-ladder calibration. These scalar thresholds are a conservative
    # fallback only, and require all three raw families to vote noise-dominant.
    psd_spectral_entropy_high: float = 0.92         # PSD family vote
    psd_off_harmonic_ratio_high: float = 0.40       # PSD family vote
    wavelet_mad_sigma_high_quantile_proxy: float = 0.05  # wavelet/temporal family vote (legacy scalar)
    # cross_coh_mean_low: "broken sensor" floor, not a noise indicator.
    # Pooled mirror calibration N=7525, 30 ladders, 2026-05-27 found clean
    # cross_coh_mean_pairwise has clean_mean=0.881, clean_std=0.061,
    # clean_p95=0.968 (clean coherence is high, and the noisy mirrors don't
    # drop it). The cross_channel family was not promoted -- coherence is not
    # a usable noise discriminator on this dataset. We keep this gate well
    # below the clean range so it only fires on pathological (broken-sensor)
    # rows where coherence collapses; ~3 sigma below clean_mean ~= 0.70.
    cross_coh_mean_low: float = 0.50                # cross-channel family vote

    # Per-signal wavelet sigma thresholds. When provided, the wavelet family
    # vote uses signal-specific cutoffs instead of the cross-unit scalar above
    # (force and position channels have different physical units, so a single
    # scalar averaged across them is not meaningful). Family votes True when a
    # majority of signals with a finite value exceed their per-signal cutoff.
    # Pooled mirror calibration N=7525, 30 ladders, 2026-05-27 supplies the
    # per-signal clean_p95 baselines; the defaults below use 3x clean_p95 as
    # an outlier gate so that ordinary clean rows do not trip the wavelet
    # vote while +60/+90 noisy mirrors do (their noise5_mean is ~5x clean_p95).
    wavelet_mad_sigma_high_by_signal: Mapping[str, float] | None = field(
        default_factory=lambda: {
            "Forcex": 7.674,    # 3 x clean_p95 (2.558); noise5_mean=12.14
            "Forcey": 11.723,   # 3 x clean_p95 (3.908); noise5_mean=19.94
            "xtpos": 1.224e-6,  # 3 x clean_p95 (4.08e-7); noise5_mean=4.02e-6
            "ytpos": 1.785e-6,  # 3 x clean_p95 (5.95e-7); noise5_mean=5.79e-6
        }
    )

    # --- Chatter detection thresholds (any-channel).
    # chatter_fn_band_energy_high: not calibratable from pooled mirror
    # calibration N=7525, 30 ladders, 2026-05-27. The chatter family was
    # fully rejected as a noise indicator (correct -- chatter is not noise).
    # Worse, clean_p95 for chatter_fn_band_energy_Forcex/Forcey is 0.98/0.90
    # because clean rows include actual chatter events; setting a default
    # from clean_p95 here would silence the chatter detector entirely.
    # Default 0.15 is retained as a physically motivated chatter-energy cut.
    chatter_fn_band_energy_high: float = 0.15
    chatter_sideband_energy_high: float = 0.05
    # chatter_cyclostationarity_tpf_low: pooled mirror calibration N=7525,
    # 30 ladders, 2026-05-27 shows clean per-signal medians 22.7 / 49.0 /
    # 85.5 / 249.0 (xtpos/ytpos/Forcex/Forcey), but the distribution is
    # extremely heavy-tailed (std >> median) and the chatter family was
    # rejected. We keep the conservative default of 1.0 -- it acts only as
    # a tiny floor that excludes degenerate rows where TPF cyclostationarity
    # collapses to ~0; calibrating to clean medians would over-restrict.
    chatter_cyclostationarity_tpf_low: float = 1.0
    chatter_fn_band_kurtosis_high: float = 5.0

    # --- TPF-dominated clean-cut thresholds.
    psd_off_harmonic_ratio_clean_max: float = 0.20
    # psd_spectral_entropy_clean_max: pooled mirror calibration N=7525,
    # 30 ladders, 2026-05-27. clean_p95 per signal: Forcex=0.430,
    # Forcey=0.418, xtpos=0.371, ytpos=0.419. We take the max of these
    # so a "clean" row only needs its mean entropy below the worst clean
    # per-signal ceiling; the placeholder 0.75 never triggered on real data.
    psd_spectral_entropy_clean_max: float = 0.43

    # --- b_over_blim window for the boundary-transition mode.
    b_over_blim_near_boundary_low: float = 0.80
    b_over_blim_near_boundary_high: float = 1.20


def aggregate_signal_metric(
    metrics_row: Mapping[str, float],
    family_prefix: str,
    signals: Sequence[str] = _DEFAULT_SIGNALS,
    reducer: str = "mean",
) -> float:
    """Reduce a per-signal metric family to one scalar.

    Looks up ``f"{family_prefix}_{signal}"`` for every signal in ``signals``,
    drops NaN/missing entries, and applies ``reducer`` to the rest. Returns NaN
    if every signal entry is missing or NaN.

    Parameters
    ----------
    metrics_row:
        Mapping that includes the per-signal metric keys.
    family_prefix:
        Family prefix without trailing underscore, e.g. ``"psd_spectral_entropy"``.
    signals:
        Signal names to aggregate over.
    reducer:
        One of ``"mean"``, ``"max"``, or ``"min"``.

    Returns
    -------
    float
        The reduced value, or NaN if no finite value is available.
    """

    values: list[float] = []
    for signal in signals:
        key = f"{family_prefix}_{signal}"
        if key not in metrics_row:
            continue
        try:
            v = float(metrics_row[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            values.append(v)

    if not values:
        return math.nan

    if reducer == "mean":
        return float(sum(values) / len(values))
    if reducer == "max":
        return float(max(values))
    if reducer == "min":
        return float(min(values))
    raise ValueError(
        f"Unknown reducer {reducer!r}; expected one of 'mean', 'max', 'min'."
    )


def _safe_float(value: object) -> float:
    """Coerce ``value`` to float; return NaN on any failure."""
    if value is None:
        return math.nan
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(v):
        return math.nan
    return v


def _count_nan_metrics(
    metrics_row: Mapping[str, float],
    signals: Sequence[str] = _DEFAULT_SIGNALS,
) -> tuple[int, int]:
    """Count (nan_or_missing, total_expected) over the canonical 44 metric keys."""
    total = 0
    missing = 0
    for prefix in _PER_SIGNAL_PREFIXES:
        for signal in signals:
            total += 1
            key = f"{prefix}_{signal}"
            if key not in metrics_row:
                missing += 1
                continue
            try:
                v = float(metrics_row[key])
            except (TypeError, ValueError):
                missing += 1
                continue
            if not math.isfinite(v):
                missing += 1
    for key in _CROSS_KEYS:
        total += 1
        if key not in metrics_row:
            missing += 1
            continue
        try:
            v = float(metrics_row[key])
        except (TypeError, ValueError):
            missing += 1
            continue
        if not math.isfinite(v):
            missing += 1
    return missing, total


def assign_mode(
    metrics_row: Mapping[str, float],
    *,
    b_over_blim: float | None = None,
    is_low_speed_edge: bool = False,
    thresholds: ModeThresholds | None = None,
    calibration: Mapping[str, Any] | None = None,
) -> tuple[str, float, dict[str, str]]:
    """Assign a Phase 2.5 mode label to a single file.

    Parameters
    ----------
    metrics_row:
        Mapping of metric name to float, as produced by
        :func:`pri2m.noise.metrics.compute_file_metrics`. Missing keys and NaN
        values are tolerated; the function never raises on input-shape issues.
    b_over_blim:
        Metadata-derived ``b/blim`` ratio. ``None`` disables the near-boundary
        check but does not block other rules.
    is_low_speed_edge:
        External flag set by the metadata stage when the spindle speed is so
        low that cyclostationarity at TPF cannot be computed.
    thresholds:
        Override for the default :class:`ModeThresholds`.
    calibration:
        Optional payload from ``scripts/calibrate_noise_mirror_pairs.py``. When
        present, broadband-noise votes use promoted metrics and their clean
        reference percentiles instead of raw cross-unit thresholds.

    Returns
    -------
    (label, confidence, votes_explained):
        ``label`` is one of :data:`MODE_LABELS`.
        ``confidence`` is in ``[0, 1]``. The numbers are coarse on purpose so
        that Phase 3 sample weighting has something to consume; tightening them
        is v2 work.
        ``votes_explained`` is a short, human-readable diagnostic dict.
    """

    th = thresholds or ModeThresholds()
    votes: dict[str, str] = {}

    # Guard: too many NaNs -> we cannot characterize the file at all.
    try:
        missing, total = _count_nan_metrics(metrics_row)
    except Exception:  # pragma: no cover - defensive
        missing, total = 0, 1
    if total > 0 and missing > 0.5 * total:
        return ("metrics_unavailable", 0.0, {"reason": "metrics_unavailable"})

    # Explicit low-speed-edge flag wins over everything except the NaN guard.
    if is_low_speed_edge:
        votes["reason"] = "low_speed_edge_flag"
        return ("low_speed_edge", 0.9, votes)

    # --- Aggregate per-signal metrics. NaN-safe via aggregate_signal_metric.
    mean_psd_entropy = aggregate_signal_metric(
        metrics_row, "psd_spectral_entropy", reducer="mean"
    )
    mean_psd_off_harm = aggregate_signal_metric(
        metrics_row, "psd_off_harmonic_ratio", reducer="mean"
    )
    mean_wavelet_sigma = aggregate_signal_metric(
        metrics_row, "wavelet_mad_sigma", reducer="mean"
    )
    mean_cyclo = aggregate_signal_metric(
        metrics_row, "chatter_cyclostationarity_tpf", reducer="mean"
    )

    max_fn_energy = aggregate_signal_metric(
        metrics_row, "chatter_fn_band_energy", reducer="max"
    )
    max_sideband = aggregate_signal_metric(
        metrics_row, "chatter_sideband_energy", reducer="max"
    )
    max_fn_kurt = aggregate_signal_metric(
        metrics_row, "chatter_fn_band_kurtosis", reducer="max"
    )

    cross_mean = _safe_float(metrics_row.get("cross_coh_mean_pairwise"))

    # --- Family votes for the noise-dominant rule.
    psd_entropy_high = (
        math.isfinite(mean_psd_entropy)
        and mean_psd_entropy >= th.psd_spectral_entropy_high
    )
    psd_off_harm_high = (
        math.isfinite(mean_psd_off_harm)
        and mean_psd_off_harm >= th.psd_off_harmonic_ratio_high
    )
    psd_family_vote_noise = psd_entropy_high or psd_off_harm_high

    if th.wavelet_mad_sigma_high_by_signal:
        wavelet_family_vote_noise = _per_signal_wavelet_vote(
            metrics_row, th.wavelet_mad_sigma_high_by_signal
        )
    else:
        wavelet_family_vote_noise = (
            math.isfinite(mean_wavelet_sigma)
            and mean_wavelet_sigma >= th.wavelet_mad_sigma_high_quantile_proxy
        )

    cross_family_vote_noise = (
        math.isfinite(cross_mean) and cross_mean <= th.cross_coh_mean_low
    )

    calibrated_noise = _calibrated_noise_vote(metrics_row, calibration)
    if calibrated_noise is not None:
        noise_dominant = calibrated_noise["noise_dominant"] == "yes"
        votes["noise_family_votes"] = calibrated_noise["noise_family_votes"]
        votes["noise_score"] = calibrated_noise["noise_score"]
        votes["noise_score_metric_weighted"] = calibrated_noise["noise_score_metric_weighted"]
        votes["calibrated_noise_metrics_used"] = calibrated_noise["metrics_used"]
        votes["abstaining_families"] = calibrated_noise["abstaining_families"]
    else:
        noise_dominant = (
            psd_family_vote_noise
            and wavelet_family_vote_noise
            and cross_family_vote_noise
        )

    # --- Chatter detection (any-channel).
    chatter_present = (
        (math.isfinite(max_fn_energy) and max_fn_energy >= th.chatter_fn_band_energy_high)
        or (math.isfinite(max_sideband) and max_sideband >= th.chatter_sideband_energy_high)
        or (math.isfinite(max_fn_kurt) and max_fn_kurt >= th.chatter_fn_band_kurtosis_high)
    )

    # --- TPF-dominant clean cut.
    tpf_dominant = (
        math.isfinite(mean_psd_off_harm)
        and mean_psd_off_harm <= th.psd_off_harmonic_ratio_clean_max
        and math.isfinite(mean_psd_entropy)
        and mean_psd_entropy <= th.psd_spectral_entropy_clean_max
        and math.isfinite(mean_cyclo)
        and mean_cyclo >= th.chatter_cyclostationarity_tpf_low
    )

    # --- Near-boundary transition (metadata).
    near_boundary = False
    if b_over_blim is not None:
        try:
            b_val = float(b_over_blim)
        except (TypeError, ValueError):
            b_val = math.nan
        if math.isfinite(b_val) and (
            th.b_over_blim_near_boundary_low
            <= b_val
            <= th.b_over_blim_near_boundary_high
        ):
            near_boundary = True

    # --- Diagnostic strings (human-readable).
    noise_bits = []
    if psd_entropy_high:
        noise_bits.append("psd_entropy=high")
    if psd_off_harm_high:
        noise_bits.append("psd_off_harm=high")
    if wavelet_family_vote_noise:
        noise_bits.append("wavelet_sigma=high")
    if cross_family_vote_noise:
        noise_bits.append("cross_coh=low")
    if calibrated_noise is None:
        votes["noise_family_votes"] = "; ".join(noise_bits) if noise_bits else "none"
    votes["chatter_signal"] = "yes" if chatter_present else "no"
    votes["near_boundary"] = "yes" if near_boundary else "no"
    votes["tpf_dominant"] = "yes" if tpf_dominant else "no"

    # --- Label decision (priority order).
    if noise_dominant:
        # With calibration, confidence reflects clean-baseline exceedance.
        if calibrated_noise is not None:
            conf = _safe_float(calibrated_noise.get("confidence"))
        elif math.isfinite(cross_mean):
            # Legacy fallback reflects how strongly the file failed coherence.
            conf = 1.0 - cross_mean
        else:
            conf = 0.5
        conf = max(0.3, min(0.95, conf))
        return ("broadband_noise_dominated", float(conf), votes)

    if chatter_present and tpf_dominant:
        return ("mixed_tpf_chatter", 0.6, votes)

    if chatter_present:
        return ("chatter_dominated", 0.7, votes)

    if near_boundary:
        return ("near_boundary_transition", 0.6, votes)

    if tpf_dominant:
        return ("tpf_dominated_clean", 0.8, votes)

    return ("mixed_tpf_chatter", 0.4, votes)


def _per_signal_wavelet_vote(
    metrics_row: Mapping[str, float],
    per_signal_thresholds: Mapping[str, float],
) -> bool:
    """Vote True when a majority of signals exceed their per-signal cutoff.

    Only signals with both a finite metric value AND a finite threshold count
    toward the denominator; signals with missing thresholds or NaN values are
    ignored. Ties (exactly half) do not vote True.
    """
    n_exceed = 0
    n_eligible = 0
    for signal, threshold in per_signal_thresholds.items():
        if not (isinstance(threshold, (int, float)) and math.isfinite(float(threshold))):
            continue
        value = _safe_float(metrics_row.get(f"wavelet_mad_sigma_{signal}"))
        if not math.isfinite(value):
            continue
        n_eligible += 1
        if value >= float(threshold):
            n_exceed += 1
    if n_eligible == 0:
        return False
    return n_exceed * 2 > n_eligible


def _calibrated_noise_vote(
    metrics_row: Mapping[str, float],
    calibration: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    if not calibration:
        return None
    metric_stats = calibration.get("metric_stats")
    if not isinstance(metric_stats, Mapping) or not metric_stats:
        return None

    required_families = set(calibration.get("noise_score_families") or ())
    if not required_families:
        required_families = {"psd", "wavelet_temporal", "cross_channel"}

    family_votes: dict[str, bool] = {family: False for family in required_families}
    family_scores: dict[str, list[float]] = {family: [] for family in required_families}
    metrics_used = 0

    for metric, stats in metric_stats.items():
        if not isinstance(stats, Mapping):
            continue
        family = str(stats.get("family") or "")
        if family not in required_families:
            continue
        direction = str(stats.get("direction") or "higher_is_noisier")
        if direction != "higher_is_noisier":
            continue
        value = _safe_float(metrics_row.get(str(metric)))
        clean_p95 = _safe_float(stats.get("clean_p95"))
        clean_median = _safe_float(stats.get("clean_median"))
        clean_std = _safe_float(stats.get("clean_std"))
        if not (math.isfinite(value) and math.isfinite(clean_p95) and math.isfinite(clean_median)):
            continue

        metrics_used += 1
        scale = clean_std if math.isfinite(clean_std) and clean_std > 0 else abs(clean_p95 - clean_median)
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        z = (value - clean_median) / max(scale, 1e-12)
        family_scores[family].append(max(0.0, float(z)))
        if value > clean_p95:
            family_votes[family] = True

    if metrics_used == 0:
        return None

    # A family is "present" if calibration promoted at least one metric for it.
    # A family that is absent from promotion abstains rather than blocking -- the
    # alternative (require all three required families be present AND all vote
    # yes) silently fails closed whenever the dataset's noise model leaves one
    # family with no promoted metrics (e.g., simulation with correlated noise,
    # or real ADC noise that drops coherence -- which then becomes an inverse
    # indicator, not a promoted metric).
    present_families = {family for family, scores in family_scores.items() if scores}
    voting_present_families = [
        f for f in required_families if f in present_families and family_votes[f]
    ]
    # Production decision rule: at least two PRESENT families must vote
    # noise-dominant, AND no PRESENT family votes against (i.e., every present
    # family agrees). This lets unpromoted families abstain without allowing a
    # single family, or two correlated metrics inside one family, to trigger
    # the broadband-noise label.
    all_present_voted = all(family_votes[f] for f in present_families)
    noise_dominant = (
        len(present_families) >= 2
        and len(voting_present_families) >= 2
        and all_present_voted
    )
    # Family-weighted score: compute each present family's mean clipped z-score,
    # then average those family means.  This prevents a family with many promoted
    # metrics from dominating the score even though the vote counts each family as
    # a single unit of evidence.
    all_scores = [score for values in family_scores.values() for score in values]
    per_family_means = [
        sum(family_scores[f]) / len(family_scores[f])
        for f in present_families
        if family_scores[f]
    ]
    noise_score = float(sum(per_family_means) / len(per_family_means)) if per_family_means else 0.0
    # Diagnostic: flat metric-weighted average (old behaviour) kept for reference.
    noise_score_metric_weighted = float(sum(all_scores) / len(all_scores)) if all_scores else 0.0
    confidence = min(0.95, max(0.0, noise_score / 3.0))

    voted = [family for family in sorted(required_families) if family_votes.get(family)]
    abstaining = [
        family for family in sorted(required_families) if family not in present_families
    ]
    return {
        "noise_dominant": "yes" if noise_dominant else "no",
        "noise_family_votes": "; ".join(voted) if voted else "none",
        "noise_score": f"{noise_score:.6g}",
        "noise_score_metric_weighted": f"{noise_score_metric_weighted:.6g}",
        "confidence": f"{confidence:.6g}",
        "metrics_used": str(metrics_used),
        "abstaining_families": "; ".join(abstaining) if abstaining else "none",
    }
