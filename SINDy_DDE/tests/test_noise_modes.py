"""Tests for ``pri2m.noise.modes``.

These tests construct synthetic metric rows (not synthetic signals) and verify
that :func:`pri2m.noise.modes.assign_mode` returns the expected label and that
the helpers behave as documented.
"""

from __future__ import annotations

import math

import pytest

from pri2m.noise.modes import (
    MODE_LABELS,
    ModeThresholds,
    aggregate_signal_metric,
    assign_mode,
)


SIGNALS = ("xtpos", "ytpos", "Forcex", "Forcey")

_PER_SIGNAL_PREFIXES = (
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

_CROSS_KEYS = (
    "cross_coh_Forcex_Forcey",
    "cross_coh_xtpos_ytpos",
    "cross_coh_Forcex_xtpos",
    "cross_coh_mean_pairwise",
)


def _make_full_row(
    *,
    psd_entropy: float = 0.5,
    psd_off_harm: float = 0.1,
    psd_flat: float = 0.1,
    psd_noise_floor: float = 0.1,
    wavelet_sigma: float = 0.001,
    autocorr_lag: float = 10.0,
    chatter_fn_energy: float = 0.01,
    chatter_fn_kurt: float = 0.5,
    chatter_sideband: float = 0.005,
    chatter_cyclo: float = 5.0,
    cross_coh_mean: float = 0.7,
    cross_coh_FxFy: float = 0.7,
    cross_coh_xy: float = 0.7,
    cross_coh_Fxx: float = 0.7,
) -> dict[str, float]:
    """Build a synthetic 44-key metric row uniformly filled across signals."""
    per_signal_values = {
        "psd_noise_floor_ratio": psd_noise_floor,
        "psd_off_harmonic_ratio": psd_off_harm,
        "psd_spectral_flatness": psd_flat,
        "psd_spectral_entropy": psd_entropy,
        "wavelet_mad_sigma": wavelet_sigma,
        "autocorr_decay_lag_samples": autocorr_lag,
        "chatter_fn_band_energy": chatter_fn_energy,
        "chatter_fn_band_kurtosis": chatter_fn_kurt,
        "chatter_sideband_energy": chatter_sideband,
        "chatter_cyclostationarity_tpf": chatter_cyclo,
    }
    row: dict[str, float] = {}
    for prefix, value in per_signal_values.items():
        for signal in SIGNALS:
            row[f"{prefix}_{signal}"] = value
    row["cross_coh_Forcex_Forcey"] = cross_coh_FxFy
    row["cross_coh_xtpos_ytpos"] = cross_coh_xy
    row["cross_coh_Forcex_xtpos"] = cross_coh_Fxx
    row["cross_coh_mean_pairwise"] = cross_coh_mean
    return row


def test_clean_tpf_dominated_row():
    row = _make_full_row(
        psd_entropy=0.4,
        psd_off_harm=0.05,
        chatter_cyclo=5.0,
        chatter_fn_energy=0.001,
        chatter_fn_kurt=0.5,
        chatter_sideband=0.001,
        wavelet_sigma=0.0005,
        cross_coh_mean=0.8,
    )
    label, conf, votes = assign_mode(row, b_over_blim=0.4)
    assert label == "tpf_dominated_clean"
    assert conf > 0.5
    assert votes["tpf_dominant"] == "yes"


def test_chatter_dominated_row():
    # Single-channel chatter on Forcex: high fn-band energy, high sideband,
    # high kurtosis. Other channels and noise indicators stay clean. We also
    # suppress the cyclostationarity index so the row is NOT tpf_dominant —
    # otherwise the file would route to mixed_tpf_chatter.
    row = _make_full_row(
        psd_entropy=0.4,
        psd_off_harm=0.25,         # > clean_max so not tpf_dominant
        chatter_cyclo=0.5,         # below threshold -> not tpf_dominant
        chatter_fn_energy=0.001,
        chatter_fn_kurt=0.5,
        chatter_sideband=0.001,
        wavelet_sigma=0.0005,
        cross_coh_mean=0.8,
    )
    row["chatter_fn_band_energy_Forcex"] = 0.30
    row["chatter_sideband_energy_Forcex"] = 0.10
    row["chatter_fn_band_kurtosis_Forcex"] = 10.0
    # b_over_blim > 1 to keep this out of near-boundary regardless of priority.
    label, conf, votes = assign_mode(row, b_over_blim=2.5)
    assert label == "chatter_dominated"
    assert votes["chatter_signal"] == "yes"


def test_broadband_noise_row():
    row = _make_full_row(
        psd_entropy=0.96,       # PSD vote: high entropy
        psd_off_harm=0.55,      # PSD vote: high off-harmonic
        wavelet_sigma=0.5,      # placeholder; overwritten per-signal below
        cross_coh_mean=0.10,    # cross vote: low coherence
        cross_coh_FxFy=0.1,
        cross_coh_xy=0.1,
        cross_coh_Fxx=0.1,
        chatter_fn_energy=0.001,
        chatter_fn_kurt=0.5,
        chatter_sideband=0.001,
        chatter_cyclo=0.1,
    )
    # Per-signal wavelet vote now uses unit-aware cutoffs (force vs position
    # have different units). Push each signal above its default cutoff
    # (Forcex ~7.67, Forcey ~11.72, xtpos ~1.2e-6, ytpos ~1.78e-6).
    row["wavelet_mad_sigma_Forcex"] = 20.0
    row["wavelet_mad_sigma_Forcey"] = 30.0
    row["wavelet_mad_sigma_xtpos"] = 1.0e-5
    row["wavelet_mad_sigma_ytpos"] = 1.0e-5
    label, conf, votes = assign_mode(row, b_over_blim=1.5)
    assert label == "broadband_noise_dominated"
    assert 0.3 <= conf <= 0.95
    assert "cross_coh=low" in votes["noise_family_votes"]


def test_near_boundary_row():
    row = _make_full_row(
        psd_entropy=0.5,
        psd_off_harm=0.25,        # above clean_max (0.20) -> not tpf_dominant
        wavelet_sigma=0.001,
        cross_coh_mean=0.7,
        chatter_fn_energy=0.001,
        chatter_fn_kurt=0.5,
        chatter_sideband=0.001,
        chatter_cyclo=0.5,        # below clean threshold -> not tpf_dominant
    )
    label, conf, votes = assign_mode(row, b_over_blim=1.0)
    assert label == "near_boundary_transition"
    assert votes["near_boundary"] == "yes"


def test_mixed_tpf_chatter_row():
    row = _make_full_row(
        psd_entropy=0.4,
        psd_off_harm=0.05,
        chatter_cyclo=5.0,
        chatter_fn_energy=0.001,
        chatter_fn_kurt=0.5,
        chatter_sideband=0.001,
        wavelet_sigma=0.0005,
        cross_coh_mean=0.8,
    )
    # Add a chatter peak on Forcey.
    row["chatter_fn_band_energy_Forcey"] = 0.40
    label, conf, votes = assign_mode(row, b_over_blim=0.5)
    assert label == "mixed_tpf_chatter"
    assert votes["tpf_dominant"] == "yes"
    assert votes["chatter_signal"] == "yes"


def test_low_speed_edge_when_flag_set():
    row = _make_full_row()
    label, conf, votes = assign_mode(row, b_over_blim=0.5, is_low_speed_edge=True)
    assert label == "low_speed_edge"
    assert conf == pytest.approx(0.9)


def test_returns_metrics_unavailable_when_most_metrics_nan():
    # 44 total expected metrics. Put 40 NaNs.
    row: dict[str, float] = {}
    # 10 per-signal prefixes * 4 signals = 40 per-signal keys.
    nan_filled = 0
    for prefix in _PER_SIGNAL_PREFIXES:
        for signal in SIGNALS:
            row[f"{prefix}_{signal}"] = math.nan
            nan_filled += 1
            if nan_filled >= 40:
                break
        if nan_filled >= 40:
            break
    # Provide finite values for the four cross-channel keys (so exactly 40 NaN of 44).
    for key in _CROSS_KEYS:
        row[key] = 0.7
    label, conf, votes = assign_mode(row, b_over_blim=0.5)
    assert label == "metrics_unavailable"
    assert conf == 0.0
    assert votes.get("reason") == "metrics_unavailable"


def test_noise_dominant_requires_all_three_families():
    # PSD + wavelet votes "noise", cross_coh is healthy (not low).
    row = _make_full_row(
        psd_entropy=0.96,
        psd_off_harm=0.55,
        wavelet_sigma=0.5,
        cross_coh_mean=0.8,
        cross_coh_FxFy=0.8,
        cross_coh_xy=0.8,
        cross_coh_Fxx=0.8,
        chatter_fn_energy=0.001,
        chatter_fn_kurt=0.5,
        chatter_sideband=0.001,
        chatter_cyclo=0.1,
    )
    label, conf, votes = assign_mode(row, b_over_blim=0.5)
    assert label != "broadband_noise_dominated"


def test_b_over_blim_none_does_not_crash():
    row = _make_full_row()
    label, conf, votes = assign_mode(row, b_over_blim=None)
    assert label in MODE_LABELS
    assert votes["near_boundary"] == "no"


def test_calibrated_noise_vote_uses_clean_reference_families():
    row = _make_full_row(
        psd_entropy=0.4,
        psd_off_harm=0.05,
        wavelet_sigma=0.001,
        cross_coh_mean=0.8,
        chatter_cyclo=5.0,
    )
    row["psd_noise_floor_ratio_Forcey"] = 0.55
    row["wavelet_mad_sigma_Forcey"] = 0.20
    row["cross_coh_xtpos_ytpos"] = 0.90
    calibration = {
        "noise_score_families": ["psd", "wavelet_temporal", "cross_channel"],
        "metric_stats": {
            "psd_noise_floor_ratio_Forcey": {
                "family": "psd",
                "direction": "higher_is_noisier",
                "clean_median": 0.10,
                "clean_std": 0.05,
                "clean_p95": 0.20,
            },
            "wavelet_mad_sigma_Forcey": {
                "family": "wavelet_temporal",
                "direction": "higher_is_noisier",
                "clean_median": 0.01,
                "clean_std": 0.02,
                "clean_p95": 0.05,
            },
            "cross_coh_xtpos_ytpos": {
                "family": "cross_channel",
                "direction": "higher_is_noisier",
                "clean_median": 0.30,
                "clean_std": 0.10,
                "clean_p95": 0.50,
            },
        },
    }

    label, conf, votes = assign_mode(row, b_over_blim=2.0, calibration=calibration)

    assert label == "broadband_noise_dominated"
    assert conf > 0.3
    assert "psd" in votes["noise_family_votes"]
    assert "wavelet_temporal" in votes["noise_family_votes"]
    assert "cross_channel" in votes["noise_family_votes"]
    assert float(votes["noise_score"]) > 0


def test_calibrated_noise_vote_allows_absent_family_to_abstain():
    row = _make_full_row(
        psd_entropy=0.4,
        psd_off_harm=0.05,
        wavelet_sigma=0.001,
        cross_coh_mean=0.8,
        chatter_cyclo=0.5,
    )
    row["psd_noise_floor_ratio_Forcey"] = 0.55
    row["wavelet_mad_sigma_Forcey"] = 0.20
    calibration = {
        "noise_score_families": ["psd", "wavelet_temporal", "cross_channel"],
        "metric_stats": {
            "psd_noise_floor_ratio_Forcey": {
                "family": "psd",
                "direction": "higher_is_noisier",
                "clean_median": 0.10,
                "clean_std": 0.05,
                "clean_p95": 0.20,
            },
            "wavelet_mad_sigma_Forcey": {
                "family": "wavelet_temporal",
                "direction": "higher_is_noisier",
                "clean_median": 0.01,
                "clean_std": 0.02,
                "clean_p95": 0.05,
            },
        },
    }

    label, _, votes = assign_mode(row, b_over_blim=2.0, calibration=calibration)

    assert label == "broadband_noise_dominated"
    assert "psd" in votes["noise_family_votes"]
    assert "wavelet_temporal" in votes["noise_family_votes"]
    assert votes["abstaining_families"] == "cross_channel"


def test_calibrated_noise_vote_rejects_present_family_voting_no():
    row = _make_full_row(
        psd_entropy=0.4,
        psd_off_harm=0.05,
        wavelet_sigma=0.001,
        cross_coh_mean=0.8,
        chatter_cyclo=0.5,
    )
    row["psd_noise_floor_ratio_Forcey"] = 0.55
    row["wavelet_mad_sigma_Forcey"] = 0.20
    row["cross_coh_xtpos_ytpos"] = 0.40
    calibration = {
        "noise_score_families": ["psd", "wavelet_temporal", "cross_channel"],
        "metric_stats": {
            "psd_noise_floor_ratio_Forcey": {
                "family": "psd",
                "direction": "higher_is_noisier",
                "clean_median": 0.10,
                "clean_std": 0.05,
                "clean_p95": 0.20,
            },
            "wavelet_mad_sigma_Forcey": {
                "family": "wavelet_temporal",
                "direction": "higher_is_noisier",
                "clean_median": 0.01,
                "clean_std": 0.02,
                "clean_p95": 0.05,
            },
            "cross_coh_xtpos_ytpos": {
                "family": "cross_channel",
                "direction": "higher_is_noisier",
                "clean_median": 0.30,
                "clean_std": 0.10,
                "clean_p95": 0.50,
            },
        },
    }

    label, _, votes = assign_mode(row, b_over_blim=2.0, calibration=calibration)

    assert label != "broadband_noise_dominated"
    assert "psd" in votes["noise_family_votes"]
    assert "wavelet_temporal" in votes["noise_family_votes"]
    assert "cross_channel" not in votes["noise_family_votes"]


def test_aggregate_signal_metric_reducer_modes():
    row = {
        "psd_spectral_entropy_xtpos": 0.1,
        "psd_spectral_entropy_ytpos": 0.5,
        "psd_spectral_entropy_Forcex": 0.9,
        "psd_spectral_entropy_Forcey": math.nan,
    }
    mean_val = aggregate_signal_metric(row, "psd_spectral_entropy", reducer="mean")
    max_val = aggregate_signal_metric(row, "psd_spectral_entropy", reducer="max")
    min_val = aggregate_signal_metric(row, "psd_spectral_entropy", reducer="min")
    # Finite values only: 0.1, 0.5, 0.9.
    assert mean_val == pytest.approx((0.1 + 0.5 + 0.9) / 3.0)
    assert max_val == pytest.approx(0.9)
    assert min_val == pytest.approx(0.1)

    # All NaN -> NaN.
    all_nan_row = {f"foo_{s}": math.nan for s in SIGNALS}
    assert math.isnan(aggregate_signal_metric(all_nan_row, "foo", reducer="mean"))


def test_votes_explained_is_human_readable():
    row = _make_full_row()
    label, conf, votes = assign_mode(row, b_over_blim=0.5)
    for key in ("noise_family_votes", "chatter_signal", "near_boundary", "tpf_dominant"):
        assert key in votes
        assert isinstance(votes[key], str)


def test_noise_score_is_family_weighted_not_metric_weighted():
    """noise_score must be the mean of per-family mean z-scores, not the flat
    average of all metric z-scores.

    Layout (promoted metrics only; calibration only promotes these keys):
      - psd family: 3 promoted metrics, each with z-score 3.0
        (value=3.0, clean_median=0.0, clean_std=1.0, clean_p95=1.0 => z=3, votes yes)
      - wavelet_temporal family: 1 promoted metric with z-score 0.0
        (value=0.0, clean_median=0.0, clean_std=1.0, clean_p95=1.0 => z=0, votes no)
      - cross_channel family: no promoted metrics => abstains

    Flat metric average: (3+3+3+0) / 4 = 2.25
    Family-weighted average: mean(psd_mean=3.0, wavelet_mean=0.0) = 1.5

    The function must return noise_score == 1.5 (family-weighted).
    """
    # Use real per-signal metric keys so the NaN guard passes.
    # psd family: promote 3 specific per-signal PSD keys, each value=3.0
    # wavelet_temporal family: promote 1 per-signal wavelet key, value=0.0
    # cross_channel family: no promoted metrics (abstains)
    row = _make_full_row(
        psd_entropy=0.5,
        psd_off_harm=0.15,
        wavelet_sigma=0.001,
        cross_coh_mean=0.7,
        chatter_cyclo=0.5,
    )
    # Set the three promoted PSD metrics to 3.0 (well above p95=1.0).
    row["psd_spectral_entropy_xtpos"] = 3.0
    row["psd_spectral_entropy_ytpos"] = 3.0
    row["psd_off_harmonic_ratio_xtpos"] = 3.0
    # Set the promoted wavelet metric to 0.0 (equals clean_median => z=0).
    row["wavelet_mad_sigma_Forcex"] = 0.0

    calibration = {
        "noise_score_families": ["psd", "wavelet_temporal", "cross_channel"],
        "metric_stats": {
            # Three PSD promoted metrics: value=3.0, median=0, std=1, p95=1 => z=3, votes yes.
            "psd_spectral_entropy_xtpos": {
                "family": "psd",
                "direction": "higher_is_noisier",
                "clean_median": 0.0,
                "clean_std": 1.0,
                "clean_p95": 1.0,
            },
            "psd_spectral_entropy_ytpos": {
                "family": "psd",
                "direction": "higher_is_noisier",
                "clean_median": 0.0,
                "clean_std": 1.0,
                "clean_p95": 1.0,
            },
            "psd_off_harmonic_ratio_xtpos": {
                "family": "psd",
                "direction": "higher_is_noisier",
                "clean_median": 0.0,
                "clean_std": 1.0,
                "clean_p95": 1.0,
            },
            # One wavelet promoted metric: value=0.0, median=0, std=1, p95=1 => z=0, votes no.
            "wavelet_mad_sigma_Forcex": {
                "family": "wavelet_temporal",
                "direction": "higher_is_noisier",
                "clean_median": 0.0,
                "clean_std": 1.0,
                "clean_p95": 1.0,
            },
            # No cross_channel metrics => that family abstains.
        },
    }

    _, _, votes = assign_mode(row, b_over_blim=2.0, calibration=calibration)

    assert "noise_score" in votes, f"noise_score missing from votes: {votes}"
    noise_score = float(votes["noise_score"])
    # Family-weighted: mean(psd_mean=3.0, wavelet_mean=0.0) = 1.5
    assert noise_score == pytest.approx(1.5), (
        f"Expected family-weighted noise_score=1.5 but got {noise_score}. "
        f"Flat metric average would be 2.25."
    )
    # Diagnostic field must also be present and equal the old flat average.
    assert "noise_score_metric_weighted" in votes, (
        "Expected diagnostic field 'noise_score_metric_weighted' in votes"
    )
    assert float(votes["noise_score_metric_weighted"]) == pytest.approx(2.25)


def test_thresholds_override_takes_effect():
    # Build a row that would normally be chatter_dominated.
    row = _make_full_row(
        psd_entropy=0.4,
        psd_off_harm=0.05,
        chatter_cyclo=0.5,        # avoid tpf_dominant so default label is chatter
        chatter_fn_energy=0.001,
        chatter_fn_kurt=0.5,
        chatter_sideband=0.001,
        wavelet_sigma=0.0005,
        cross_coh_mean=0.8,
    )
    row["chatter_fn_band_energy_Forcex"] = 0.30
    row["chatter_fn_band_energy_Forcey"] = 0.001  # keep others quiet
    # Sanity: default thresholds classify as chatter_dominated.
    default_label, _, _ = assign_mode(row, b_over_blim=2.0)
    assert default_label == "chatter_dominated"

    # Suppress chatter detection by pushing fn-band threshold above 0.30, and
    # also lift sideband and kurtosis thresholds (their values are tiny but be
    # explicit) so the chatter rule cannot fire on this row.
    custom = ModeThresholds(
        chatter_fn_band_energy_high=0.99,
        chatter_sideband_energy_high=0.99,
        chatter_fn_band_kurtosis_high=1000.0,
    )
    new_label, _, votes = assign_mode(row, b_over_blim=2.0, thresholds=custom)
    assert new_label != "chatter_dominated"
    assert votes["chatter_signal"] == "no"
