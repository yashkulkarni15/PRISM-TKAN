"""Tests for pri2m.xai.noise_probe (T1/T2 model comparison, 2026-10-05)."""

from __future__ import annotations

import numpy as np
import pytest

from pri2m.xai import noise_probe as NP


def _sig(T=4000, C=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    return np.sin(2 * np.pi * t / 50.0) * rng.uniform(1, 100, C) + rng.uniform(-5, 5, C)


# --------------------------------------------------------------------------- #
# Mirror pairing + tier
# --------------------------------------------------------------------------- #
def test_noise_tier_from_dataset_id():
    assert NP.noise_tier(1) == 0 and NP.noise_tier(60) == 0
    assert NP.noise_tier(61) == 1 and NP.noise_tier(90) == 1
    assert NP.noise_tier(91) == 5 and NP.noise_tier(120) == 5
    assert NP.noise_tier(121) is None  # held-out: tier from metadata, not id


def test_mirror_file_name():
    assert NP.mirror_file("time1_10.h5", 1, 91) == "time91_10.h5"
    assert NP.mirror_file("time30_7.h5", 30, 60 + 30) == "time90_7.h5"
    with pytest.raises(ValueError):
        NP.mirror_file("time2_10.h5", 1, 91)  # name does not belong to dataset 1


# --------------------------------------------------------------------------- #
# Calibration: mirror - clean
# --------------------------------------------------------------------------- #
def test_calibration_recovers_additive_white_noise():
    clean = _sig()
    rng = np.random.default_rng(1)
    rel = np.array([0.05, 0.05, 0.02, 0.02])
    noisy = clean + rng.standard_normal(clean.shape) * rel * clean.std(axis=0)
    cal = NP.mirror_noise_stats(clean, noisy)
    np.testing.assert_allclose(cal["rel_sigma"], rel, rtol=0.05)
    assert np.all(np.abs(cal["lag1_autocorr"]) < 0.05)        # white
    assert np.all(np.abs(cal["corr_with_clean"]) < 0.05)      # additive, independent
    assert cal["additive_ok"] is True


def test_calibration_flags_non_additive_difference():
    clean = _sig()
    noisy = clean * 1.3  # a gain change, not added noise
    cal = NP.mirror_noise_stats(clean, noisy)
    assert cal["additive_ok"] is False


def test_calibration_truncates_to_common_length():
    clean = _sig(T=4000)
    noisy = _sig(T=3990)
    cal = NP.mirror_noise_stats(clean, noisy)
    assert cal["n_samples"] == 3990


# --------------------------------------------------------------------------- #
# Injection: deterministic, identical for every model
# --------------------------------------------------------------------------- #
def test_inject_zero_level_is_identity():
    s = _sig()
    out = NP.inject_noise(s, level=0.0, rel_sigma_per_unit=np.ones(4), key="f.h5")
    assert out is not s
    np.testing.assert_array_equal(out, s)


def test_inject_is_deterministic_by_key_and_level():
    s = _sig()
    u = np.array([0.01, 0.01, 0.004, 0.004])
    a = NP.inject_noise(s, 5.0, u, key="time1_3.h5", seed=0)
    b = NP.inject_noise(s, 5.0, u, key="time1_3.h5", seed=0)
    c = NP.inject_noise(s, 5.0, u, key="time1_4.h5", seed=0)
    d = NP.inject_noise(s, 2.5, u, key="time1_3.h5", seed=0)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    assert not np.allclose(a, d)


def test_inject_scales_with_level_and_file_std():
    s = _sig(T=200_000)
    u = np.array([0.01, 0.02, 0.01, 0.01])  # relative sigma per 1 % of tier
    out = NP.inject_noise(s, 5.0, u, key="k", seed=0)
    got = (out - s).std(axis=0) / s.std(axis=0)
    np.testing.assert_allclose(got, 5.0 * u, rtol=0.02)


def test_inject_rejects_bad_shapes():
    with pytest.raises(ValueError):
        NP.inject_noise(np.zeros(10), 1.0, np.ones(1), key="k")
    with pytest.raises(ValueError):
        NP.inject_noise(_sig(), 1.0, np.ones(3), key="k")
    with pytest.raises(ValueError):
        NP.inject_noise(_sig(), -1.0, np.ones(4), key="k")


# --------------------------------------------------------------------------- #
# Summary of a calibration over many pairs
# --------------------------------------------------------------------------- #
def test_summarise_calibration_per_unit_and_linearity():
    rows = []
    for tier, k in ((1, 0.01), (5, 0.05)):
        for _ in range(20):
            rows.append({"tier": tier, "rel_sigma": [k, k, 0.4 * k, 0.4 * k],
                         "additive_ok": True})
    s = NP.summarise_calibration(rows)
    np.testing.assert_allclose(s["rel_sigma_per_unit"], [0.01, 0.01, 0.004, 0.004], rtol=1e-9)
    assert s["linear_ok"] is True
    assert s["n_pairs"] == {"1": 20, "5": 20}


def test_summarise_calibration_refuses_when_not_additive():
    rows = [{"tier": 5, "rel_sigma": [0.05] * 4, "additive_ok": False}] * 10
    s = NP.summarise_calibration(rows)
    assert s["rel_sigma_per_unit"] is None
    assert s["undecided_reason"] == "mirror_difference_not_additive"
