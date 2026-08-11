"""The delay-coupled edge pairs read as a tooth-pass comb filter.

U13-C measured, on all five folds, that a delayed edge is a near-exact scaled
*copy* of its instantaneous partner on the same unit (top pair: corr +0.997,
gain 1.03) rather than its negation. It filed that as a refuted hypothesis --
the network did not learn the regenerative difference x(t) - x(t-tau).

What a sum *is*, mechanically, was never extracted. With the two curves
proportional, the unit computes

    y(t) + b*y(t-tau),      y = phi_inst(x),   b = corr * gain

which is a delay-and-add comb: ``|H(f)| = sqrt(1 + b^2 + 2b cos(2 pi f tau))``,
peaking at every integer multiple of ``1/tau``. Since the model sets
``tau = round(fs/tpf)`` per sample, ``1/tau`` *is* the tooth-pass frequency, so
the comb auto-tunes its passbands to the tooth-passing harmonic family of
whatever cut it is looking at. A difference would have put notches there
instead. Sum vs difference is therefore reinforce-vs-cancel of the forced
content, not "wrong structure".

No linearity assumption is needed: the proportionality is between the two
*curves* and is measured per pair, so the comb acts on the nonlinearly warped
signal ``y``. What must be checked, and is, is how many pairs are actually
proportional enough for the reading to hold -- on fold 2 only 41% of 768 clear
|corr| >= 0.9, and that fraction is a headline number, not a footnote.
"""

from __future__ import annotations

import numpy as np
import pytest

from pri2m.xai.comb import (
    aggregate_comb,
    chatter_attenuation,
    comb_coefficient,
    comb_metrics,
    comb_response,
    expected_chatter_attenuation,
)


# --------------------------------------------------------------------------- #
# The coefficient
# --------------------------------------------------------------------------- #
def test_coefficient_is_the_projection_of_delayed_onto_instantaneous():
    """b = <inst, delay>/<inst, inst> = corr * gain, exactly."""
    assert comb_coefficient(corr=0.9966, gain=1.0264) == pytest.approx(1.0229, abs=1e-4)


def test_a_negating_pair_gives_a_negative_coefficient():
    """The regenerative difference the plan hoped for."""
    assert comb_coefficient(corr=-0.95, gain=1.0) < 0


def test_coefficient_is_nan_when_the_pair_is_unmeasurable():
    assert np.isnan(comb_coefficient(corr=float("nan"), gain=1.0))


# --------------------------------------------------------------------------- #
# The response: peaks land on the tooth-pass harmonics for a sum
# --------------------------------------------------------------------------- #
def test_a_summing_pair_peaks_at_every_tooth_pass_harmonic():
    b = 1.0
    at_harmonics = comb_response(b, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.allclose(at_harmonics, 1.0 + b)


def test_a_summing_pair_notches_halfway_between_harmonics():
    b = 1.0
    between = comb_response(b, np.array([0.5, 1.5, 2.5]))
    assert np.allclose(between, abs(1.0 - b), atol=1e-12)


def test_a_negating_pair_inverts_the_pattern():
    """The difference notches exactly where the sum reinforces.

    This is the whole physical content of U13-C's sign: the network chose to
    reinforce the tooth-passing family rather than cancel it.
    """
    b = -1.0
    assert comb_response(b, np.array([1.0, 2.0]))[0] == pytest.approx(0.0, abs=1e-12)
    assert comb_response(b, np.array([0.5, 1.5]))[0] == pytest.approx(2.0)


def test_response_is_periodic_in_normalised_frequency():
    b = 0.7
    rng = np.array([0.13, 1.13, 2.13, 3.13])
    r = comb_response(b, rng)
    assert np.allclose(r, r[0])


# --------------------------------------------------------------------------- #
# Contrast
# --------------------------------------------------------------------------- #
def test_metrics_report_passband_stopband_and_contrast():
    m = comb_metrics(1.0264 * 0.9966)  # the real fold-2 top pair

    assert m["passband"] == pytest.approx(1 + 1.0229, abs=1e-3)
    assert m["stopband"] == pytest.approx(abs(1 - 1.0229), abs=1e-3)
    assert m["contrast"] > 40                      # a deep notch: b is near 1
    assert m["peaks_at_tooth_pass"] is True


def test_a_negating_pair_is_flagged_as_notching_the_tooth_pass():
    m = comb_metrics(-0.9)

    assert m["peaks_at_tooth_pass"] is False


def test_a_dead_delayed_edge_is_a_flat_response_not_a_comb():
    """b = 0 means no delay coupling at all: contrast 1, nothing to interpret."""
    m = comb_metrics(0.0)

    assert m["passband"] == pytest.approx(1.0)
    assert m["contrast"] == pytest.approx(1.0)


def test_contrast_is_finite_even_at_a_perfect_notch():
    """|b| == 1 gives a zero stopband; an inf would poison every aggregate."""
    m = comb_metrics(1.0)

    assert np.isfinite(m["contrast"])
    assert m["perfect_notch"] is True


# --------------------------------------------------------------------------- #
# Where chatter actually lands -- measured, not assumed
# --------------------------------------------------------------------------- #
def test_chatter_on_a_harmonic_is_not_attenuated_at_all():
    """The convenient story would be that chatter always falls in a notch.

    It does not: fold 2's fn/tpf is about 2.12, near the 2nd harmonic. If
    chatter sits ON a harmonic the comb amplifies it exactly as much as the
    forced content, and the selectivity story fails for that operating point.
    """
    assert chatter_attenuation(1.0, fn_over_tpf=2.0) == pytest.approx(1.0)


def test_chatter_between_harmonics_is_strongly_attenuated():
    assert chatter_attenuation(1.0, fn_over_tpf=2.5) < 0.01


def test_the_real_fold2_ratio_is_only_mildly_attenuated():
    """Fold 2: tooth-pass 506.8 Hz, fn 1072.3 Hz -> 2.116. Pinning the honest
    number so nobody quotes a notch that is not there."""
    att = chatter_attenuation(1.0229, fn_over_tpf=1072.265625 / 506.78710804879665)

    assert 0.85 < att < 1.0


def test_expected_attenuation_averages_over_the_fn_distribution():
    """fn has a 492 Hz spread, so one ratio is not the answer -- the expectation
    over the distribution is."""
    rng = np.random.default_rng(0)
    ratios = 2.116 + rng.normal(0, 492.3 / 506.8, size=4000)

    got = expected_chatter_attenuation(1.0, ratios)

    assert 0.0 < got < 1.0
    # A spread-out chatter frequency lands anywhere on the comb, so on average it
    # is far below the passband that the forced content always sits on.
    assert got < 0.9


# --------------------------------------------------------------------------- #
# Aggregation over the stored rows
# --------------------------------------------------------------------------- #
def _rows():
    return [
        {"corr": 0.99, "gain": 1.0, "weight": 2.0, "kan": "encoder.layers.0.kan_z",
         "channel": "xtpos", "degenerate": 0},
        {"corr": 0.30, "gain": 1.0, "weight": 1.0, "kan": "encoder.layers.0.kan_r",
         "channel": "ytpos", "degenerate": 0},
        {"corr": float("nan"), "gain": float("nan"), "weight": 1.0,
         "kan": "encoder.layers.0.kan_h", "channel": "xtpos", "degenerate": 1},
    ]


def test_aggregate_reports_the_proportional_fraction_as_a_headline():
    out = aggregate_comb(_rows(), corr_threshold=0.9)

    assert out["n_rows"] == 3
    assert out["n_degenerate"] == 1
    assert out["n_proportional"] == 1
    assert out["frac_proportional"] == pytest.approx(0.5)  # of the 2 usable rows


def test_aggregate_restricts_comb_statistics_to_proportional_pairs():
    """A pair at corr 0.30 is not a scaled copy, so reading a comb off it would
    be describing a filter the unit does not implement."""
    out = aggregate_comb(_rows(), corr_threshold=0.9)

    assert out["contrast_median"] is not None
    assert out["n_used"] == 1


def test_aggregate_returns_none_rather_than_a_number_when_nothing_qualifies():
    out = aggregate_comb(_rows(), corr_threshold=0.999999)

    assert out["n_used"] == 0
    assert out["contrast_median"] is None
    assert out["frac_peaks_at_tooth_pass"] is None


def test_aggregate_weights_by_importance():
    """An edge the model barely uses must not count like one it leans on."""
    rows = [
        {"corr": 0.99, "gain": 1.0, "weight": 100.0, "kan": "k", "channel": "c",
         "degenerate": 0},
        {"corr": -0.99, "gain": 1.0, "weight": 0.01, "kan": "k", "channel": "c",
         "degenerate": 0},
    ]

    out = aggregate_comb(rows, corr_threshold=0.9)

    assert out["frac_peaks_at_tooth_pass"] == pytest.approx(0.5)
    assert out["weighted_frac_peaks_at_tooth_pass"] > 0.99


def test_aggregate_splits_by_gate_and_channel():
    out = aggregate_comb(_rows(), corr_threshold=0.9)

    assert "encoder.layers.0.kan_z" in out["by_gate"]
    assert "xtpos" in out["by_channel"]


# --------------------------------------------------------------------------- #
# Measuring the response of a real pair, for the confirmation run
# --------------------------------------------------------------------------- #
def test_measured_response_of_a_linear_pair_matches_the_analytic_comb():
    """The prediction and the measurement must agree where theory is exact.

    With phi_inst = identity and phi_delay = b*identity the composed unit *is*
    y(t) + b*y(t-tau), so the swept measurement has to reproduce |H| exactly. If
    this drifts, the confirmation run is measuring its own harness rather than
    the network.
    """
    from pri2m.xai.comb import measure_pair_response

    b, tau = 0.8, 40
    ratios = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.25])

    measured = measure_pair_response(lambda x: x, lambda x: b * x, tau, ratios)

    assert np.allclose(measured, comb_response(b, ratios), atol=1e-6)


def test_measured_response_isolates_the_fundamental_from_harmonics():
    """A nonlinear edge generates harmonics; the comb claim is about the tap.

    Measuring total power would fold the nonlinearity's own harmonics into the
    answer and inflate the apparent response, so the probe projects onto the
    drive frequency instead.
    """
    from pri2m.xai.comb import measure_pair_response

    tau = 40
    ratios = np.array([1.0, 1.5])
    # A cubic distorts heavily but the pair is still a pure delay-and-add.
    measured = measure_pair_response(lambda x: x ** 3, lambda x: x ** 3, tau, ratios)

    assert measured[0] > measured[1]                      # peak at the harmonic
    assert np.isfinite(measured).all()


def test_peak_alignment_ignores_the_dc_lobe():
    """DC is not a tooth-pass harmonic, and it was being counted as one.

    A nonlinear edge rectifies, so the composed response can carry a large
    constant term. Taking a global argmax then returns f = 0, and because 0 is
    an integer a "distance to nearest harmonic" test scored it as a hit. On the
    real probe that happened for 44 of 125 pairs, inflating the headline.
    """
    from pri2m.xai.comb import peak_alignment

    ratios = np.linspace(0.0, 3.0, 241)
    # A comb that notches the harmonics, sitting on a huge DC pedestal.
    response = comb_response(-0.9, ratios)
    response[0] = 100.0

    out = peak_alignment(ratios, response)

    assert out["peak_ratio"] != 0.0, "the DC lobe must not win the peak search"
    assert out["on_harmonic"] is False, "this comb peaks between harmonics"


def test_peak_alignment_finds_a_genuine_harmonic_peak():
    from pri2m.xai.comb import peak_alignment

    ratios = np.linspace(0.0, 3.0, 241)

    out = peak_alignment(ratios, comb_response(0.9, ratios))

    assert out["on_harmonic"] is True
    assert abs(out["peak_ratio"] - round(out["peak_ratio"])) <= 0.1


def test_peak_alignment_reports_the_distance_it_judged_on():
    from pri2m.xai.comb import peak_alignment

    ratios = np.linspace(0.0, 3.0, 241)

    out = peak_alignment(ratios, comb_response(0.9, ratios))

    assert out["distance_to_nearest_harmonic"] <= 0.1
    assert out["min_ratio"] > 0


def test_a_dead_delayed_edge_measures_as_flat():
    from pri2m.xai.comb import measure_pair_response

    ratios = np.array([0.5, 1.0, 1.5, 2.0])
    measured = measure_pair_response(lambda x: x, lambda x: 0.0 * x, 40, ratios)

    assert np.allclose(measured, measured[0], atol=1e-6)
