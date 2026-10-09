"""Read the delay-coupled edge pairs as a tooth-pass comb filter.

U13-C established, on all five folds against a flat untrained null, that a
delayed edge is a near-exact scaled **copy** of its instantaneous partner on the
same unit rather than its negation: the network implements ``x(t) + x(t-tau)``,
a sum, not the regenerative difference ``x(t) - x(t-tau)``. That was recorded as
a refuted hypothesis and left there.

This module extracts what the sum *means*. For a pair whose two curves are
proportional -- which is what ``corr -> 1`` measures -- the unit computes

    y(t) + b * y(t - tau),    y = phi_inst(x),    b = corr * gain

where ``b`` is the least-squares projection of the delayed curve onto the
instantaneous one. That is a delay-and-add comb with

    |H(f)| = sqrt(1 + b^2 + 2 b cos(2 pi f tau))

whose passbands sit at every integer multiple of ``1/tau``. The model fixes
``tau = round(fs/tpf)`` per sample, so ``1/tau`` **is** that cut's tooth-pass
frequency: the comb auto-tunes its passbands to the tooth-passing harmonic
family of whatever operating point it is looking at. A negating pair (``b < 0``)
inverts the pattern and notches the same family instead. Sum-vs-difference is
therefore *reinforce vs cancel the forced content*, which is a statement about
what the network computes rather than about which hypothesis lost.

Two honesty constraints are built in rather than bolted on.

**Proportionality is measured, never assumed.** The comb reading only describes
a pair whose delayed curve really is a scaled copy; at ``corr = 0.3`` the unit
implements something else and quoting a comb for it would be describing a filter
that is not there. Every aggregate reports how many pairs qualified, and returns
``None`` -- never a number -- when none do. On fold 2 only 41% of 768 rows clear
``|corr| >= 0.9``.

**Where chatter lands is measured, not assumed.** The tidy story would be that
chatter always falls in a comb notch. It does not follow: attenuation depends on
``fn/tpf`` mod 1, and fold 2's ratio is about 2.12, close enough to the second
harmonic that a single operating point is barely attenuated at all. What is
structurally true is only that the *forced* content sits on a passband by
construction; the chatter side has to be integrated over the real spread of
``fn/tpf`` (see :func:`expected_chatter_attenuation`).

No linearity assumption is required anywhere. The proportionality holds between
the two learned curves, so the comb acts on the nonlinearly warped signal ``y``;
``phi`` itself may be as nonlinear as it likes.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "comb_coefficient",
    "comb_response",
    "comb_metrics",
    "chatter_attenuation",
    "expected_chatter_attenuation",
    "measure_pair_response",
    "peak_alignment",
    "aggregate_comb",
]

# Below this |corr| the delayed curve is not a scaled copy of the instantaneous
# one and the delay-and-add reading does not describe the unit. Pre-committed
# here rather than chosen after seeing the distributions.
DEFAULT_CORR_THRESHOLD = 0.9
# |b| within this of 1 gives a mathematically zero stopband. Reported as a
# perfect notch and floored, because an infinity would poison every aggregate
# it touched -- the same "no sentinel in a mean" rule as tr_peak_lag.
_PERFECT_NOTCH_EPS = 1e-6


def comb_coefficient(corr: float, gain: float) -> float:
    """The delay tap ``b``, as the projection of the delayed curve onto the
    instantaneous one.

    ``b = <inst, delay> / <inst, inst> = corr * gain`` exactly, given
    ``gain = ||delay|| / ||inst||``. Using the projection rather than the raw
    amplitude ratio keeps the sign (sum vs difference) and the magnitude in one
    number, and makes ``b`` the coefficient that actually multiplies the delayed
    copy in the unit's output.
    """
    return float(corr) * float(gain)


def comb_response(b, f_over_tooth_pass):
    """``|H(f)|`` of ``y(t) + b*y(t-tau)``, in units of the tooth-pass frequency.

    The frequency axis is normalised by ``1/tau``, so ``f_over_tooth_pass = 1``
    is the tooth-pass fundamental, ``2`` its second harmonic, and so on. This is
    the natural axis: it needs no sampling rate, and it is the axis on which the
    comb is the same filter for every operating point even though ``tau`` and
    ``fs`` both vary across the dataset.
    """
    b = float(b)
    f = np.asarray(f_over_tooth_pass, dtype=float)
    return np.sqrt(np.maximum(1.0 + b * b + 2.0 * b * np.cos(2.0 * np.pi * f), 0.0))


def comb_metrics(b: float) -> dict:
    """Passband, stopband and contrast of the comb with tap ``b``.

    ``peaks_at_tooth_pass`` is the physical content of the U13-C sign: ``True``
    when the pair reinforces the tooth-passing harmonic family (a sum), ``False``
    when it notches it (the regenerative difference).
    """
    b = float(b)
    if not math.isfinite(b):
        return {"b": float("nan"), "passband": None, "stopband": None,
                "contrast": None, "contrast_db": None,
                "peaks_at_tooth_pass": None, "perfect_notch": None}
    passband = abs(1.0 + b) if b >= 0 else abs(1.0 - b)
    stopband = abs(1.0 - abs(b))
    perfect = stopband < _PERFECT_NOTCH_EPS
    contrast = passband / max(stopband, _PERFECT_NOTCH_EPS)
    return {
        "b": b,
        "passband": passband,
        "stopband": stopband,
        "contrast": contrast,
        "contrast_db": 20.0 * math.log10(max(contrast, 1e-12)),
        # A sum reinforces the tooth-pass family; a difference notches it.
        "peaks_at_tooth_pass": bool(b > 0),
        "perfect_notch": bool(perfect),
    }


def chatter_attenuation(b: float, fn_over_tpf: float) -> float:
    """``|H(fn)| / |H(passband)|`` -- how the comb treats the chatter frequency.

    1.0 means chatter is amplified exactly as much as the forced content, which
    happens whenever ``fn`` lands on a tooth-pass harmonic. This function exists
    to make that case visible rather than to confirm a notch.
    """
    peak = comb_metrics(b)["passband"]
    if not peak:
        return float("nan")
    at_fn = float(comb_response(b, np.array([float(fn_over_tpf)]))[0])
    return at_fn / peak


def expected_chatter_attenuation(b: float, fn_over_tpf_samples) -> float:
    """Mean attenuation over a *distribution* of ``fn/tpf``.

    One ratio answers nothing: ``fn`` varies widely across the dataset (fold 2
    records 1072 +/- 492 Hz against a 507 Hz tooth-pass), so chatter lands
    somewhere different in every cut. The forced content sits on a passband in
    every cut by construction; this is the honest comparison against it.
    """
    ratios = np.asarray(fn_over_tpf_samples, dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return float("nan")
    peak = comb_metrics(b)["passband"]
    if not peak:
        return float("nan")
    return float(np.mean(comb_response(b, ratios)) / peak)


def measure_pair_response(phi_inst, phi_delay, tau: int, f_over_tpf,
                          n_periods: int = 64, amplitude: float = 1.0):
    """Sweep a sinusoid through a real edge pair and measure its response.

    This is the *empirical* counterpart of :func:`comb_response`. Rather than
    trusting ``b = corr*gain``, it drives the two learned activation functions
    with ``x(t) = A sin(2 pi f t)``, forms the composed contribution

        ``s(t) = phi_inst(x(t)) + phi_delay(x(t - tau))``

    and reports the amplitude of ``s`` **at the drive frequency**. Projecting
    onto the drive rather than taking total power matters: a learned edge is
    nonlinear and generates its own harmonics, which would otherwise be counted
    as response and inflate the answer.

    Where theory is exact -- ``phi_inst`` linear and ``phi_delay`` a scaled copy
    -- this reproduces ``comb_response`` to floating point, which is what pins
    the harness in the tests.

    Parameters
    ----------
    phi_inst, phi_delay:
        Callables mapping an array of inputs to the edge's output. In the
        confirmation run these are ``KANLinear.edge_function`` interpolants for
        the instantaneous and delayed edges of one ``(kan, out)`` unit.
    tau:
        Delay in samples. The frequency axis is expressed in units of ``1/tau``,
        so ``f_over_tpf = 1`` is the tooth-pass fundamental.
    """
    tau = int(tau)
    ratios = np.atleast_1d(np.asarray(f_over_tpf, dtype=float))
    # Enough cycles that the DFT bin is sharp, plus tau samples of run-in that
    # are discarded so the delayed term is always defined.
    n = max(int(n_periods) * tau, 8 * tau)
    t = np.arange(n + tau, dtype=float)
    out = np.empty(ratios.shape, dtype=float)

    for i, r in enumerate(ratios):
        # cycles per sample; r = 1 puts one full cycle in tau samples.
        f = float(r) / tau
        # Cosine, not sine: at f = 0 a sine drive is identically zero, so there
        # would be no signal to respond to and DC would read as a dead pair.
        x = amplitude * np.cos(2.0 * np.pi * f * t)
        raw = np.asarray(phi_inst(x[tau:]), dtype=float) + np.asarray(
            phi_delay(x[:-tau]), dtype=float
        )
        if f == 0.0:
            # DC: the response IS the mean, with no two-sided factor of 2.
            out[i] = abs(float(np.mean(raw))) / max(amplitude, 1e-12)
            continue
        # Drop the DC offset a nonlinear edge adds by rectification, then take
        # the amplitude at the drive frequency: 2/N * |sum s(t) e^{-i2pi f t}|.
        s = raw - raw.mean()
        tt = t[tau:]
        proj = np.sum(s * np.exp(-2j * np.pi * f * tt))
        out[i] = 2.0 * abs(proj) / (s.size * max(amplitude, 1e-12))

    return out if out.size > 1 else out


# Below this multiple of the tooth-pass frequency the response is the DC lobe,
# which is not a passband of the comb. A nonlinear edge rectifies, so the
# composed signal can carry a large constant term; a global argmax then returns
# f = 0, and because 0 is an integer a "distance to nearest harmonic" test scores
# it as a hit. Measured on the real probe: 44 of 125 pairs peaked at DC.
_DC_LOBE_MAX_RATIO = 0.25


def peak_alignment(f_over_tpf, response, min_ratio: float = _DC_LOBE_MAX_RATIO) -> dict:
    """Where a measured response peaks, and whether that is a tooth-pass harmonic.

    The DC lobe is excluded from the search rather than from the reported curve:
    the comb claim is about the *periodic* structure, and a constant offset is
    neither evidence for it nor against it.
    """
    f = np.asarray(f_over_tpf, dtype=float)
    r = np.asarray(response, dtype=float)
    keep = f >= float(min_ratio)
    if not keep.any():
        return {"peak_ratio": None, "distance_to_nearest_harmonic": None,
                "on_harmonic": None, "min_ratio": float(min_ratio)}
    peak = float(f[keep][int(np.argmax(r[keep]))])
    dist = abs(peak - round(peak))
    return {
        "peak_ratio": peak,
        "distance_to_nearest_harmonic": float(dist),
        "on_harmonic": bool(dist <= 0.1),
        "min_ratio": float(min_ratio),
    }


def _usable(row: dict) -> bool:
    if int(row.get("degenerate", 0) or 0):
        return False
    return math.isfinite(float(row.get("corr", float("nan")))) and math.isfinite(
        float(row.get("gain", float("nan")))
    )


#: Public alias. The ablation in ``comb_ablation`` must exclude exactly the rows
#: this reading excludes, and importing the predicate is the only way to
#: guarantee the two cannot drift apart. Re-implementing it there is how the
#: selection would quietly come to describe a different set of pairs than
#: Section 2 reports.
usable = _usable


def _weighted_mean(values, weights) -> float | None:
    v, w = np.asarray(values, float), np.asarray(weights, float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return None
    return float(np.sum(v[keep] * w[keep]) / np.sum(w[keep]))


def _stats(rows: list[dict], corr_threshold: float) -> dict:
    """Comb statistics over the pairs that qualify, and the count that did not."""
    usable = [r for r in rows if _usable(r)]
    used = [r for r in usable if abs(float(r["corr"])) >= corr_threshold]
    out = {
        "n_rows": len(rows),
        "n_degenerate": sum(1 for r in rows if not _usable(r)),
        "n_usable": len(usable),
        "n_proportional": len(used),
        "n_used": len(used),
        # Of the pairs that could be measured at all -- the headline caveat.
        "frac_proportional": (len(used) / len(usable)) if usable else None,
        "corr_threshold": float(corr_threshold),
    }
    if not used:
        out.update({
            "contrast_median": None, "contrast_db_median": None,
            "b_median": None, "frac_peaks_at_tooth_pass": None,
            "weighted_frac_peaks_at_tooth_pass": None,
            "frac_perfect_notch": None, "weighted_contrast_mean": None,
        })
        return out

    bs = [comb_coefficient(r["corr"], r["gain"]) for r in used]
    ms = [comb_metrics(b) for b in bs]
    weights = [float(r.get("weight", 1.0) or 0.0) for r in used]
    peaks = [1.0 if m["peaks_at_tooth_pass"] else 0.0 for m in ms]
    out.update({
        "b_median": float(np.median(bs)),
        "contrast_median": float(np.median([m["contrast"] for m in ms])),
        "contrast_db_median": float(np.median([m["contrast_db"] for m in ms])),
        "weighted_contrast_mean": _weighted_mean([m["contrast"] for m in ms], weights),
        "frac_peaks_at_tooth_pass": float(np.mean(peaks)),
        "weighted_frac_peaks_at_tooth_pass": _weighted_mean(peaks, weights),
        "frac_perfect_notch": float(np.mean([1.0 if m["perfect_notch"] else 0.0
                                             for m in ms])),
    })
    return out


def aggregate_comb(rows, corr_threshold: float = DEFAULT_CORR_THRESHOLD) -> dict:
    """Comb statistics over stored ``regenerative.json`` rows.

    Reports the whole population and the qualifying subset separately, split by
    GRU gate and by input channel -- the delay does not have to enter every gate
    equally, and which gate it enters is a mechanistic claim about the update
    path rather than a summary statistic.
    """
    rows = list(rows)
    out = _stats(rows, corr_threshold)

    by_gate: dict[str, dict] = {}
    for gate in sorted({str(r.get("kan")) for r in rows}):
        by_gate[gate] = _stats([r for r in rows if str(r.get("kan")) == gate],
                               corr_threshold)
    by_channel: dict[str, dict] = {}
    for ch in sorted({str(r.get("channel")) for r in rows}):
        by_channel[ch] = _stats([r for r in rows if str(r.get("channel")) == ch],
                                corr_threshold)
    out["by_gate"] = by_gate
    out["by_channel"] = by_channel
    return out
