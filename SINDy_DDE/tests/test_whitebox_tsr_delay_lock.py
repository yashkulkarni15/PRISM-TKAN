"""TSR delay statistics: does the relevance curve lock to the physical delay?

Motivation (2026-07-23/24 GL run, 160 shards / 640 windows across 5 folds). The
shipped ``tr_peak_lag`` is a *global argmax of the relevance autocorrelation over
lags [2, N/2]*. When that autocorrelation decays monotonically -- i.e. no
periodicity is resolved at all -- ``argmax`` returns index 0 and the metric emits
**2**, its own search floor. 550 of 640 real windows (85.9%) came back as exactly
2, so ``tr_peak_lag_mean`` was averaging a "no answer" sentinel and reading as a
measurement. These tests pin the two fixes:

1. ``tr_peak_at_floor`` states outright whether a peak was resolved, and the
   aggregate reports the resolved fraction plus a resolved-only mean.
2. ``tr_lock_z`` is the argmax-free statistic: how far the autocorrelation **at
   lag tau** sits above the spread of autocorrelations at every other searched
   lag. It cannot be pinned by a boundary and it answers the question the peak
   lag was only ever a proxy for.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import whitebox_phase3 as W  # noqa: E402


def _regenerative_tr(n: int, tau: int, *, weight: float = 0.9, seed: int = 0) -> np.ndarray:
    """Relevance that genuinely carries structure at lag ``tau``.

    Mirrors the regenerative term: a base process plus a scaled copy of itself
    delayed by tau. Its autocorrelation has a distinct bump at tau and nowhere
    else, which is exactly the signature TSR was built to detect.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n + tau)
    return base[tau:] + weight * base[:-tau]


# --------------------------------------------------------------------------
# tr_delay_statistics -- the pure, model-free core
# --------------------------------------------------------------------------

def test_monotonically_decaying_relevance_is_reported_as_unresolved():
    """A smooth ramp has no periodicity; the argmax lands on the search floor.

    This is the 85.9% case. The old code returned 2 here and said nothing; the
    fix must flag it so it can never be averaged into a mean as if it were a
    measured lag.
    """
    tr = np.linspace(1.0, 0.0, 256)
    stats = W.tr_delay_statistics(tr, tau=32)

    assert stats["tr_peak_lag"] == 2, "ramp autocorrelation should argmax at the floor"
    assert stats["tr_peak_at_floor"] is True
    # And the lock statistic must NOT claim a delay here. z > 2 is the
    # pre-committed decision rule for "locked"; smooth decay must stay under it.
    assert stats["tr_lock_z"] is not None
    assert stats["tr_lock_z"] < 2.0, "smooth decay must not read as a delay lock"


def test_relevance_built_at_tau_resolves_a_peak_at_tau():
    tau = 32
    tr = _regenerative_tr(512, tau)
    stats = W.tr_delay_statistics(tr, tau=tau)

    assert stats["tr_peak_at_floor"] is False
    assert stats["tr_peak_lag"] == tau
    assert stats["tr_lock_z"] > 2.0, "a real lag-tau bump must stand out from other lags"


def test_white_noise_relevance_does_not_lock_to_tau():
    rng = np.random.default_rng(7)
    tr = rng.normal(size=512)
    stats = W.tr_delay_statistics(tr, tau=32)

    assert stats["tr_lock_z"] < 2.0, "unstructured relevance must not read as locked"


def test_lock_is_none_when_tau_falls_outside_the_searched_range():
    """tau longer than half the window cannot be tested -- say so, don't guess."""
    tr = _regenerative_tr(128, 16)
    stats = W.tr_delay_statistics(tr, tau=500)
    assert stats["tr_lock_z"] is None


def test_degenerate_short_window_does_not_raise():
    stats = W.tr_delay_statistics(np.array([1.0, 2.0, 3.0]), tau=1)
    assert np.isfinite(stats["tr_peak_lag"])
    assert stats["tr_lock_z"] is None


def test_constant_relevance_does_not_divide_by_zero():
    stats = W.tr_delay_statistics(np.ones(128), tau=8)
    assert stats["tr_lock_z"] is None


# --------------------------------------------------------------------------
# tsr_aggregate -- the reduction that Update 11 read off
# --------------------------------------------------------------------------

def _row(peak_lag, at_floor, lock_z, tau=100):
    shares = {ch: 0.25 for ch in W.CHANNELS}
    return {
        "index": 0, "label": 1,
        "channel_share_tsr": dict(shares), "channel_share_ig": dict(shares),
        "tr_peak_lag": peak_lag, "tr_peak_at_floor": at_floor,
        "tr_lock_z": lock_z, "tau_raw": tau,
    }


def test_aggregate_reports_the_resolved_fraction():
    per_window = [_row(2, True, -0.5)] * 8 + [_row(100, False, 3.0)] * 2
    agg = W.tsr_aggregate(per_window, steps=8)

    assert agg["n_windows"] == 10
    assert agg["n_resolved"] == 2
    assert abs(agg["resolved_fraction"] - 0.2) < 1e-9


def test_resolved_mean_excludes_the_floor_sentinel():
    """The whole defect in one assertion.

    Eight unresolved windows plus two real peaks at 100. The old mean is 21.6,
    which describes nothing. The resolved-only mean is 100.
    """
    per_window = [_row(2, True, -0.5)] * 8 + [_row(100, False, 3.0)] * 2
    agg = W.tsr_aggregate(per_window, steps=8)

    assert abs(agg["tr_peak_lag_mean"] - 21.6) < 1e-9, "back-compat key must be unchanged"
    assert abs(agg["tr_peak_lag_mean_resolved"] - 100.0) < 1e-9


def test_resolved_mean_is_none_when_nothing_resolved():
    agg = W.tsr_aggregate([_row(2, True, -0.5)] * 5, steps=8)
    assert agg["n_resolved"] == 0
    assert agg["tr_peak_lag_mean_resolved"] is None


def test_aggregate_summarises_the_lock_statistic():
    per_window = [_row(2, True, 0.1)] * 6 + [_row(100, False, 4.0)] * 4
    agg = W.tsr_aggregate(per_window, steps=8)

    assert abs(agg["lock_z_mean"] - (6 * 0.1 + 4 * 4.0) / 10) < 1e-9
    assert abs(agg["lock_z_frac_above_2"] - 0.4) < 1e-9


def test_aggregate_ignores_none_locks_without_crashing():
    per_window = [_row(2, True, None), _row(100, False, 3.0)]
    agg = W.tsr_aggregate(per_window, steps=8)
    assert abs(agg["lock_z_mean"] - 3.0) < 1e-9
    assert agg["n_lock_tested"] == 1


def test_aggregate_keeps_every_backcompat_key():
    """Already-returned manifests must stay readable by the renderer."""
    agg = W.tsr_aggregate([_row(2, True, 0.0)], steps=8)
    for key in (
        "per_window", "n_windows", "steps",
        "mean_channel_share_tsr", "mean_channel_share_ig",
        "tr_peak_lag_mean", "tau_raw_mean",
    ):
        assert key in agg, f"dropped back-compat key {key!r}"


def test_aggregate_tolerates_rows_from_the_old_schema():
    """The 2026-07-23/24 manifests predate the new per-window keys."""
    old = {
        "index": 0, "label": 1,
        "channel_share_tsr": {ch: 0.25 for ch in W.CHANNELS},
        "channel_share_ig": {ch: 0.25 for ch in W.CHANNELS},
        "tr_peak_lag": 2, "tau_raw": 100,
    }
    agg = W.tsr_aggregate([old], steps=8)
    assert agg["n_windows"] == 1
    assert agg["n_lock_tested"] == 0
    assert agg["lock_z_mean"] is None
