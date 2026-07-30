"""Graded fuzziness (degradation) suite for the Phase 3 TKAN — eval-only.

Input-side fuzzers blur what the model SEES (noise, smoothing, tau jitter);
model-side fuzzers blur the MODEL itself (spline coarsening, weight noise).
At every fuzz level the probes re-measure not just AUC but the delay-ablation
drop — the physics-reliance signature — so the resulting curves answer
"up to what fuzz level does the model still use the regenerative delay".

All fuzzers are non-mutating: input fuzzers return a new batch dict; model
fuzzers deep-copy. AUC is per-window over the sampled batch (the whitebox
convention), via physics.delay_ablation which measures on+off in one call.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from pri2m.models.kan import KANLinear
from pri2m.xai import physics
from pri2m.xai.kan_native import _iter_kanlinear
from pri2m.xai.spectral import (
    magnitude_randomize,
    match_notch_halfwidth,
    notch,
    phase_randomize,
)

AUC_DEGRADED = 0.75   # halfway to chance from ~1.0
DROP_FLOOR = 0.05     # delay-drop below this = physics signature gone
AUC_CHANCE = 0.55     # ≈ chance


# --------------------------------------------------------------------- #
# Input-side fuzzers (batch -> new batch; never mutate the input)
# --------------------------------------------------------------------- #
def fuzz_input_noise(batch: dict, sigma: float, seed: int = 0) -> dict:
    """Additive Gaussian noise (z-scored SD units) on the signal channels."""
    new = dict(batch)
    if sigma <= 0:
        new["window"] = batch["window"].clone()
        return new
    g = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(batch["window"].shape, generator=g) * float(sigma)
    new["window"] = batch["window"] + noise.to(batch["window"].dtype)
    return new


def fuzz_input_smooth(batch: dict, width: int) -> dict:
    """Reflect-padded centered moving average of ``width`` RAW samples."""
    new = dict(batch)
    width = int(width)
    if width <= 1:
        new["window"] = batch["window"].clone()
        return new
    x = batch["window"].permute(0, 2, 1)                      # [B, C, L]
    left = width // 2
    right = width - 1 - left
    padded = F.pad(x, (left, right), mode="reflect")
    smoothed = F.avg_pool1d(padded, kernel_size=width, stride=1)
    new["window"] = smoothed.permute(0, 2, 1).contiguous()
    return new


def fuzz_tau_jitter(batch: dict, frac: float, seed: int = 0) -> tuple[dict, dict]:
    """Perturb each window's tau by ±frac (random sign per window, seeded)."""
    tau = np.asarray(batch["tau"].detach().cpu().numpy(), dtype=float)
    if frac <= 0:
        return dict(batch), {"realized_abs_dtau_mean": 0.0, "frac_pooled_tau_changed": 0.0}
    g = np.random.default_rng(int(seed))
    signs = g.choice([-1.0, 1.0], size=tau.shape)
    new_tau = np.maximum(1, np.round(tau * (1.0 + signs * float(frac)))).astype(np.int64)
    new = dict(batch)
    new["tau"] = torch.as_tensor(new_tau, dtype=torch.long)
    info = {
        "realized_abs_dtau_mean": float(np.mean(np.abs(new_tau - tau))),
        # honesty: after the model's own pooling, small jitters can round away
        "frac_pooled_tau_changed": float(
            np.mean(np.maximum(1, np.round(new_tau / 4)) != np.maximum(1, np.round(tau / 4)))
        ),
    }
    return new, info


def fuzz_lag_tau_shuffle(batch: dict, frac: float, seed: int = 0) -> tuple[dict, dict]:
    """Permute blocks of length tau, destroying tau-periodicity.

    A permutation, so the amplitude distribution and every marginal statistic
    survive exactly -- only the temporal arrangement at the delay scale changes.
    At ``frac=1.0`` this is the clean "delay fully destroyed" condition;
    ``fuzz_tau_jitter`` cannot supply one because at frac >= 1 it clamps roughly
    half the windows to tau=1 instead of jittering them.
    """
    if frac <= 0:
        return dict(batch), {"frac_blocks_permuted": 0.0, "mean_blocks": 0.0}

    # Stored layout is (B, L, C). Work in (C, L) per window, then write back.
    win = batch["window"].detach().cpu().numpy().copy()
    tau = np.asarray(batch["tau"].detach().cpu().numpy(), dtype=int)
    rng = np.random.default_rng(int(seed))
    length, n_ch = win.shape[1], win.shape[2]

    permuted, total = 0, 0
    for b in range(win.shape[0]):
        step = max(1, int(tau[b]))
        n_blocks = length // step
        if n_blocks < 2:
            continue
        total += n_blocks
        n_sel = max(2, int(round(float(frac) * n_blocks)))
        sel = rng.choice(n_blocks, size=min(n_sel, n_blocks), replace=False)
        order = rng.permutation(sel)
        chan_first = win[b].T                                  # (C, L)
        src = chan_first[:, : n_blocks * step].reshape(n_ch, n_blocks, step).copy()
        dst = src.copy()
        dst[:, sel, :] = src[:, order, :]
        chan_first[:, : n_blocks * step] = dst.reshape(n_ch, n_blocks * step)
        win[b] = chan_first.T                                  # back to (L, C)
        permuted += len(sel)

    new = dict(batch)
    new["window"] = torch.as_tensor(win, dtype=batch["window"].dtype)
    return new, {
        "frac_blocks_permuted": float(permuted / total) if total else 0.0,
        "mean_blocks": float(total / max(1, win.shape[0])),
    }


def _spectral_fuzz(batch: dict, level: float, seed: int, fn) -> dict:
    """Apply a per-window spectral surrogate across the batch.

    ``batch["window"]`` is (B, L, C); the spectral helpers take (C, L). The
    transposes are load-bearing -- without them the FFT runs along the 4-element
    channel axis instead of the time axis and every result is meaningless.
    """
    win = batch["window"].detach().cpu().numpy()               # (B, L, C)
    out = np.stack([fn(win[b].T, level, seed + b).T for b in range(win.shape[0])])
    new = dict(batch)
    new["window"] = torch.as_tensor(out, dtype=batch["window"].dtype)
    return new


def fuzz_phase_randomize(batch: dict, alpha: float, seed: int = 0) -> dict:
    if alpha <= 0:
        return dict(batch)
    return _spectral_fuzz(
        batch, alpha, seed,
        lambda w, a, s: phase_randomize(w, a, seed=s, amplitude_adjust=True),
    )


def fuzz_magnitude_randomize(batch: dict, alpha: float, seed: int = 0) -> dict:
    if alpha <= 0:
        return dict(batch)
    return _spectral_fuzz(
        batch, alpha, seed, lambda w, a, s: magnitude_randomize(w, a, seed=s)
    )


def _apply_phase_randomize(model, batch, level, seed):
    return model, fuzz_phase_randomize(batch, level, seed), {}


def _apply_magnitude_randomize(model, batch, level, seed):
    return model, fuzz_magnitude_randomize(batch, level, seed), {}


def _apply_lag_tau_shuffle(model, batch, level, seed):
    new, info = fuzz_lag_tau_shuffle(batch, level, seed)
    return model, new, info


# --------------------------------------------------------------------- #
# Model-side fuzzers (model -> deep-copied fuzzed model)
# --------------------------------------------------------------------- #
def _refit_kanlinear_inplace(kan: KANLinear, grid_size: int, n_points: int = 64) -> float:
    """Least-squares refit of kan's spline onto a ``grid_size`` grid, in place.

    Samples the ORIGINAL learned spline component right-open on grid_range
    (the degree-0 basis is zero at exactly the right endpoint), solves one
    batched lstsq against the coarse basis, swaps coefficients + knot buffer.
    Returns the refit R^2 (1.0 = the coarse grid can represent the original).
    """
    order = kan.spline_order
    lo, hi = kan.grid_range
    xs = torch.linspace(lo, hi, n_points + 1, dtype=torch.float32)[:-1]     # right-open
    basis_orig = kan.b_splines(xs.unsqueeze(-1))[:, 0, :]                    # [n, n_coeff_orig]
    target = torch.einsum("xc,oic->oix", basis_orig, kan.spline_weight)      # [out, in, n]
    t = target.reshape(-1, n_points).T.contiguous()                          # [n, out*in]

    tmp = KANLinear(1, 1, grid_size=grid_size, spline_order=order, grid_range=(lo, hi))
    B = tmp.b_splines(xs.unsqueeze(-1))[:, 0, :]                             # [n, k+order]
    coeff = torch.linalg.lstsq(B, t).solution                                # [k+order, out*in]
    fitted = B @ coeff
    ss_res = ((t - fitted) ** 2).sum()
    ss_tot = ((t - t.mean(dim=0)) ** 2).sum().clamp_min(1e-12)
    r2 = float(1.0 - ss_res / ss_tot)

    kan.grid_size = int(grid_size)
    kan.n_coeff = int(grid_size) + order
    h = (hi - lo) / grid_size
    knots = torch.arange(-order, grid_size + order + 1, dtype=torch.float32) * h + lo
    kan._buffers["grid"] = knots.to(kan.grid.device)
    kan.spline_weight = torch.nn.Parameter(
        coeff.T.reshape(kan.out_features, kan.in_features, kan.n_coeff).contiguous()
    )
    return r2


def fuzz_spline_coarsen(model, grid_size: int) -> tuple[torch.nn.Module, dict]:
    fuzzed = copy.deepcopy(model)
    r2s = {}
    for name, kan in _iter_kanlinear(fuzzed):
        r2s[name or "kan"] = _refit_kanlinear_inplace(kan, int(grid_size))
    return fuzzed, {"refit_r2": r2s}


def fuzz_weight_noise(model, sigma_rel: float, seed: int = 0):
    fuzzed = copy.deepcopy(model)
    if sigma_rel <= 0:
        return fuzzed, {}
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for _, kan in _iter_kanlinear(fuzzed):
            for p in (kan.base_weight, kan.spline_weight):
                eps = torch.randn(p.shape, generator=g) * float(sigma_rel)
                p.mul_(1.0 + eps)
    return fuzzed, {}


# --------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------- #
def measure_probes(model, batch: dict, labels) -> dict:
    """AUC / severity MAE / delay-reliance, in one delay_ablation call."""
    abl = physics.delay_ablation(model, batch, labels)
    return {
        "auc": float(abl["auc_on"]),
        "auc_delay_off": float(abl["auc_off"]),
        "delay_drop": float(abl["auc_drop"]),
        "mae": float(abl["mae_on"]) if "mae_on" in abl else None,
        "mae_delay_off": float(abl["mae_off"]) if "mae_off" in abl else None,
    }


def xdelay_share(model, batch: dict):
    """x_delay data-aware weight share (None if the model has no delay block)."""
    from pri2m.xai import activations as act_mod

    stats = act_mod.collect_activations(model, batch)
    share = physics.delayed_weight_share(model, stats)
    overall = share.get("overall", {})
    return float(overall["x_delay"]) if "x_delay" in overall else None


# --------------------------------------------------------------------- #
# Registry + sweep loop
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fuzzer:
    name: str
    kind: str          # "input" | "model"
    levels: tuple
    stochastic: bool
    apply: Callable    # (model, batch, level, seed) -> (model, batch, info)


def _apply_input_noise(model, batch, level, seed):
    return model, fuzz_input_noise(batch, level, seed), {}


def _apply_input_smooth(model, batch, level, seed):
    return model, fuzz_input_smooth(batch, level), {}


def _apply_tau_jitter(model, batch, level, seed):
    new, info = fuzz_tau_jitter(batch, level, seed)
    return model, new, info


def _apply_spline_coarsen(model, batch, level, seed):
    fuzzed, info = fuzz_spline_coarsen(model, level)
    return fuzzed, batch, info


def _apply_weight_noise(model, batch, level, seed):
    fuzzed, info = fuzz_weight_noise(model, level, seed)
    return fuzzed, batch, info


FUZZERS: dict[str, Fuzzer] = {
    # -- existing five; levels are APPEND-ONLY so Update 10 stays reproducible
    "input_noise": Fuzzer("input_noise", "input", (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0), True, _apply_input_noise),
    "input_smooth": Fuzzer("input_smooth", "input", (1, 2, 4, 8, 16, 32, 64, 128), False, _apply_input_smooth),
    # capped below 1.0: at frac >= 1 the tau*(1+sign*frac) form clamps roughly
    # half the windows to tau=1 instead of jittering them. lag_tau_shuffle
    # supplies the fully-destroyed-delay condition instead.
    "tau_jitter": Fuzzer("tau_jitter", "input", (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.9), True, _apply_tau_jitter),
    "spline_coarsen": Fuzzer("spline_coarsen", "model", (5, 4, 3, 2, 1), False, _apply_spline_coarsen),
    "weight_noise": Fuzzer("weight_noise", "model", (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0), True, _apply_weight_noise),
    # -- physics-targeted
    "phase_randomize": Fuzzer("phase_randomize", "input", (0.0, 0.1, 0.25, 0.5, 0.75, 1.0), True, _apply_phase_randomize),
    "magnitude_randomize": Fuzzer("magnitude_randomize", "input", (0.0, 0.1, 0.25, 0.5, 0.75, 1.0), True, _apply_magnitude_randomize),
    "lag_tau_shuffle": Fuzzer("lag_tau_shuffle", "input", (0.0, 0.1, 0.25, 0.5, 1.0), True, _apply_lag_tau_shuffle),
}

_PROBE_KEYS = ("auc", "auc_delay_off", "delay_drop", "mae", "mae_delay_off")


def breakdown_points(levels, mean_auc, mean_drop) -> dict:
    """First level crossing each degradation threshold (None if never)."""
    def first(seq, pred):
        for lvl, v in zip(levels, seq):
            if v is not None and pred(v):
                return lvl
        return None

    return {
        "auc_below_075": first(mean_auc, lambda a: a < AUC_DEGRADED),
        "drop_below_005": first(mean_drop, lambda d: d < DROP_FLOOR),
        "auc_at_chance": first(mean_auc, lambda a: a <= AUC_CHANCE),
    }


MIN_HEADROOM = 0.01
HALF_LIFE_FRAC = 0.5
FLOOR_FRAC = 0.10
ASYMPTOTE_TOL = 0.01


def normalized_reliance(auc, delay_drop, min_headroom: float = MIN_HEADROOM):
    """Delay reliance as a fraction of the AUC headroom actually available.

    Raw ``delay_drop`` is measured against an AUC that is itself falling, so as
    accuracy approaches chance there is mechanically no room left to drop and
    reliance trends to zero regardless of what the model is doing. Dividing by
    ``auc - 0.5`` removes that confound.

    ``None`` below ``0.5 + min_headroom``: there the denominator is dominated by
    sampling noise in ``auc``, so the ratio is an artifact rather than a finding.
    """
    if auc is None or delay_drop is None:
        return None
    if float(auc) <= 0.5 + float(min_headroom):
        return None
    return float(delay_drop) / (float(auc) - 0.5)


def reliance_breakdown(levels, norm_reliance) -> dict:
    """First level where normalized reliance falls to 50% / 10% of its baseline.

    Fractions of each fold's OWN baseline, not an absolute threshold: baseline
    reliance ranges 0.230-0.486 across folds, so a fixed cut would demand a 78%
    loss on one fold and 90% on another and the folds would not be comparable.
    """
    base = next((v for v in norm_reliance if v is not None), None)
    if base is None or base <= 0:
        return {"reliance_half_life": None, "reliance_floor": None}

    def first_at_or_below(frac):
        for lvl, val in zip(levels, norm_reliance):
            if val is not None and val <= frac * base:
                return lvl
        return None

    return {
        "reliance_half_life": first_at_or_below(HALF_LIFE_FRAC),
        "reliance_floor": first_at_or_below(FLOOR_FRAC),
    }


def spectral_floor_report(mean_auc, baseline_auc_delay_off) -> dict:
    """Does the stressor drive AUC down to the no-delay-information floor?

    ``auc_delay_off`` already measures accuracy with the delayed input zeroed,
    so it IS the spectral floor. A stressor that genuinely removes delay usage
    should converge to it; a large positive gap means it did not, however
    severe it looked.
    """
    tail = [a for a in mean_auc[-2:] if a is not None]
    asymptote = float(np.mean(tail)) if tail else None
    converged = len(tail) == 2 and abs(tail[0] - tail[1]) < ASYMPTOTE_TOL
    gap = None
    if asymptote is not None and baseline_auc_delay_off is not None:
        gap = asymptote - float(baseline_auc_delay_off)
    return {
        "spectral_floor_auc": (
            None if baseline_auc_delay_off is None else float(baseline_auc_delay_off)
        ),
        "asymptote_auc": asymptote,
        "converged": bool(converged),
        "asymptote_gap": gap,
    }


def run_fuzzer(model, batch, labels, fuzzer: Fuzzer, seeds=(0, 1, 2), with_share=None,
               baseline=None) -> dict:
    """Sweep one fuzzer over its levels; stochastic fuzzers get one probe per seed.

    with_share: None -> record xdelay_share only for model-side fuzzers.
    baseline: optional fold-baseline probe dict (from measure_probes), used to
        read auc_delay_off for the spectral-floor report; None if unavailable.
    """
    if with_share is None:
        with_share = fuzzer.kind == "model"
    seeds = tuple(seeds) if fuzzer.stochastic else (0,)
    out = []
    for level in fuzzer.levels:
        per_seed, info = [], {}
        for seed in seeds:
            fm, fb, info = fuzzer.apply(model, batch, level, seed)
            probes = measure_probes(fm, fb, labels)
            if with_share:
                probes["xdelay_share"] = xdelay_share(fm, fb)
            probes["seed"] = int(seed)
            per_seed.append(probes)
        keys = [k for k in per_seed[0] if k != "seed"]
        mean = {k: (float(np.mean([p[k] for p in per_seed])) if per_seed[0][k] is not None else None) for k in keys}
        sd = {k: (float(np.std([p[k] for p in per_seed])) if per_seed[0][k] is not None else None) for k in keys}
        out.append({"level": level, "seeds": per_seed, "mean": mean, "sd": sd, "info": info})

    mean_auc = [r["mean"]["auc"] for r in out]
    mean_drop = [r["mean"]["delay_drop"] for r in out]

    result = {
        "name": fuzzer.name,
        "kind": fuzzer.kind,
        "stochastic": fuzzer.stochastic,
        "levels": list(fuzzer.levels),
        "results": out,
        "breakdown": breakdown_points(fuzzer.levels, mean_auc, mean_drop),
    }

    norm = [normalized_reliance(a, d) for a, d in zip(mean_auc, mean_drop)]
    result["normalized_reliance"] = norm
    result["reliance_breakdown"] = reliance_breakdown(list(fuzzer.levels), norm)
    result["spectral_floor"] = spectral_floor_report(
        mean_auc, baseline.get("auc_delay_off") if baseline else None
    )
    return result


# --------------------------------------------------------------------- #
# Matched-energy band-notch contrast (a paired contrast, NOT a graded sweep,
# so deliberately not a FUZZERS entry -- it writes to its own manifest key)
# --------------------------------------------------------------------- #
MAX_UNSOLVED_FRAC = 0.02

# Smallest chatter-notch removed fraction that can carry a relative energy error.
# This is a DENOMINATOR GUARD, not a quality gate: |tooth - chatter| / chatter
# divides by it, so below it the ratio reports how empty the band was rather than
# how badly the pair matched.
#
# Chosen from measured evidence, not taste. The 2026-07-30 GL run returned
# max_rel_err ~20301 beside tooth_frac ~0.064, implying a window that removed
# ~3.2e-6 -- four orders of magnitude under the 0.051-0.068 those folds removed
# on average. The smallest LEGITIMATE fraction on the L=1024 synthetic fixture is
# 8.1e-4. So any floor in [3.2e-6, 8.1e-4] separates the two populations; 1e-4
# sits ~30x above the degenerate case and ~8x below the smallest real one.
# Do not raise it toward 1e-3: that discarded a fixture window which had removed
# 0.0017 and matched to 0.4%, i.e. a good measurement thrown away.
MIN_CHATTER_FRAC = 1e-4


def _per_window_hz(value, n_b: int, name: str) -> np.ndarray:
    """Broadcast a scalar frequency, or validate a per-window one, to length ``n_b``.

    Parameters
    ----------
    value : float or array_like
        One frequency for the whole batch, or one per window.
    n_b : int
        Number of windows in the batch.
    name : str
        Argument name, quoted in the error message.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(n_b,)``.

    Raises
    ------
    ValueError
        If ``value`` is neither a scalar nor exactly ``n_b`` long. A wrong-length
        array would pair each window with some *other* window's frequency, which
        is a quieter version of the very bug this plumbing exists to fix, so it
        is refused rather than broadcast or truncated.
    """
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 1:
        return np.full(int(n_b), float(arr[0]))
    if arr.size != int(n_b):
        raise ValueError(
            f"{name} must be a scalar or a length-{n_b} per-window array, "
            f"got size {arr.size}"
        )
    return arr.astype(float, copy=True)


def run_band_notch_contrast(
    model, batch, labels, fs, fn, tpf,
    halfwidth_hz: float = 50.0, n_harmonics: int = 3,
    match_tol: float = 0.01, n_candidates: int = 41,
    search_span: tuple[float, float] = (0.25, 4.0),
    max_unsolved_frac: float = MAX_UNSOLVED_FRAC,
    min_chatter_frac: float = MIN_CHATTER_FRAC,
) -> dict:
    """Chatter-band vs tooth-pass notch at MATCHED removed energy.

    The comparison is the result: if the model is a spectral chatter-energy
    detector the chatter notch hurts more; if it is doing regeneration the
    tooth notch does. Matching the removed energy is what makes that a fair
    question rather than a measure of how much was deleted -- without the
    match, whichever condition deletes more energy wins regardless of what it
    deleted.

    **Matching is bidirectional**, and it has to be. Energy removed is a *step*
    function of halfwidth (it only moves when a band edge crosses an FFT bin),
    and with ``n_harmonics`` tooth centres each increment sweeps ~2*n_harmonics
    bins at once. Holding the chatter halfwidth fixed and solving only for the
    tooth halfwidth therefore leaves one staircase to hit an arbitrary target,
    and the target usually falls in a gap between steps: measured floors were
    7.65% relative error on the multi-tone fixture at L=1024 and 4.15% on pure
    white noise, against a 1% requirement. Searching *both* halfwidths makes the
    achievable set the product of two staircases, which is dense enough to match.

    ``halfwidth_hz`` is therefore a **nominal** chatter halfwidth; the solver may
    move it within ``search_span`` (a multiplicative range) to find a matched
    pair, and reports what it actually used. ``energy_match["matched"]`` still
    states whether the match succeeded, so a caller can never read an unfair
    contrast as a physics result by accident.

    **Every frequency here is per-window.** ``fs``, ``fn`` and ``tpf`` each take a
    scalar or a length-B array, and each window is notched at its OWN values
    throughout: the halfwidth search, the applied notch, and the tooth centres
    ``tpf*(k+1)``. Collapsing them to batch means (as the 2026-07-29 five-fold
    run did) aims the band away from the actual chatter peak for nearly every
    window and the experiment mostly deletes empty spectrum -- the same
    step-function argument that forbids averaging halfwidths, one level up at
    the band centres. ``chatter_frac_min`` is reported for exactly this reason:
    a healthy *mean* removed fraction can hide windows that were never notched.

    **An unsolved window is not a measurement.** If no candidate halfwidth is
    wide enough to reach an FFT bin, the chatter notch removes nothing and there
    is no energy target to match against; the same is true if the best tooth
    halfwidth found removes no energy either. Such a window used to fall through
    to a tooth halfwidth of 0.0 -- a filter that filters nothing -- whose
    relative error is then exactly 1.0, so a MAX over windows pinned the whole
    run at 1.0 and made a failed solve indistinguishable from a genuine 100%
    energy mismatch. Unsolved windows are now counted (``n_unsolved``) and
    excluded from the match statistics, which are additionally reported at the
    90th percentile (``rel_err_p90``) so a single window can never be the whole
    story. This follows the ``tr_peak_at_floor`` precedent: a sentinel is
    flagged, never averaged in as if it were data.

    Note that the probes still run over the *whole* batch, so an unsolved window
    also enters the tooth condition unnotched. That is a second reason
    ``n_unsolved > 0`` invalidates the contrast rather than merely widening its
    error bar, and why ``matched`` vetoes on it.

    Parameters
    ----------
    model, batch, labels
        As elsewhere in this module: ``batch["window"]`` is ``(B, L, C)``.
    fs, fn, tpf : float or array_like
        Sampling rate, chatter frequency and tooth-pass frequency, each either a
        scalar or one value per window (length B).
    halfwidth_hz : float, optional
        Nominal chatter halfwidth in Hz; the solver searches
        ``search_span * halfwidth_hz`` around it.
    n_harmonics : int, optional
        Number of tooth-pass harmonics notched in the tooth condition.
    match_tol : float, optional
        Relative energy-match tolerance required for ``matched``.
    n_candidates : int, optional
        Chatter halfwidths tried per window.
    search_span : tuple of float, optional
        Multiplicative search range around ``halfwidth_hz``.
    max_unsolved_frac : float, optional
        Largest fraction of unsolved windows still allowed to report
        ``matched=True``; default 2%. Defensible because the probes are
        per-window AUCs over the batch, so a couple of pathological windows
        perturb them by O(1/B) and cannot flip the contrast, whereas the failure
        that actually occurs -- bands aimed off-bin, or halfwidths below the bin
        spacing -- fails a large fraction of windows at once. On the small
        batches used in tests, 2% means "any unsolved window vetoes the match".

    Returns
    -------
    dict
        ``{"chatter": probes, "tooth": probes, "energy_match": {...}}``. The
        ``energy_match`` block carries the applied batch means
        (``chatter_frac``, ``tooth_frac``, ``chatter_halfwidth_hz``,
        ``tooth_halfwidth_hz``, ``tooth_centers_hz``, all including unsolved
        windows, since they describe what the model actually saw), the aim
        diagnostics (``chatter_frac_min``, ``fn_hz_mean``, ``fn_hz_sd``,
        ``chatter_tooth_margin_hz``), and the honesty block
        (``n_windows``, ``n_solved``, ``n_unsolved``, ``unsolved_fraction``,
        ``max_unsolved_frac``, ``max_rel_err``, ``rel_err_p90``, ``matched``).
        ``max_rel_err`` and ``rel_err_p90`` are ``None`` -- never a number --
        when no window solved.
    """
    win = batch["window"].detach().cpu().numpy()               # (B, L, C)
    n_b = win.shape[0]
    fs_w = _per_window_hz(fs, n_b, "fs")
    fn_w = _per_window_hz(fn, n_b, "fn")
    tpf_w = _per_window_hz(tpf, n_b, "tpf")
    n_harm = max(0, int(n_harmonics))
    # Tooth centres are tpf*(k+1), so a per-window tpf makes the centres, and
    # with them the tooth-harmonic guard below, per-window as well. (B, n_harm).
    tooth_w = tpf_w[:, None] * np.arange(1, n_harm + 1, dtype=float)[None, :]

    def _apply(centers, halfwidths):
        hw = np.broadcast_to(np.asarray(halfwidths, dtype=float), (n_b,))
        cen = np.asarray(centers, dtype=float).reshape(n_b, -1)
        out = np.empty_like(win)
        fracs = []
        for b in range(n_b):
            # notch() wants (C, L); the batch carries (L, C) per item. Getting
            # this axis wrong filters along the channel axis and fails silently.
            filt, frac = notch(win[b].T, fs=float(fs_w[b]), centers=cen[b],
                               halfwidth_hz=float(hw[b]))
            out[b] = filt.T
            fracs.append(frac)
        new = dict(batch)                                      # shallow: no mutation
        new["window"] = torch.as_tensor(out, dtype=batch["window"].dtype)
        return new, np.asarray(fracs, dtype=float)

    # Solve each window independently: averaging halfwidths across windows and
    # applying the mean is wrong for the same step-function reason -- the mean
    # halfwidth does not remove the mean energy.
    lo, hi = float(search_span[0]), float(search_span[1])
    base_candidates = np.linspace(lo * float(halfwidth_hz), hi * float(halfwidth_hz),
                                  max(1, int(n_candidates)))

    ch_hws, th_hws = np.empty(n_b), np.empty(n_b)
    margins = np.full(n_b, np.inf)
    solved = np.zeros(n_b, dtype=bool)
    for b in range(n_b):
        x = win[b].T
        fs_b, fn_b, tooth_b = float(fs_w[b]), float(fn_w[b]), tooth_w[b]

        # HARD CONSTRAINT: the chatter band must never reach a tooth-pass
        # harmonic -- now evaluated against THIS window's own harmonics. A wide
        # search span is what makes the match reachable, but a chatter notch
        # that also deletes the tooth harmonics is not a contrast between two
        # bands: it deletes both conditions' evidence and means nothing.
        candidates, max_safe = base_candidates, float("inf")
        if n_harm:
            max_safe = float(np.min(np.abs(fn_b - tooth_b)))
            candidates = base_candidates[base_candidates < max_safe]
            if candidates.size == 0:                # nominal already overlaps
                candidates = np.asarray([min(float(halfwidth_hz), 0.5 * max_safe)])

        best = None                                 # (rel_err, ch_hw, th_hw)
        for ch_hw in candidates:
            _, ch_frac = notch(x, fs=fs_b, centers=[fn_b], halfwidth_hz=float(ch_hw))
            if ch_frac <= 0:
                continue                            # narrower than one FFT bin
            th_hw = match_notch_halfwidth(x, fs=fs_b, centers=tooth_b,
                                          target_frac=float(ch_frac))
            _, th_frac = notch(x, fs=fs_b, centers=tooth_b, halfwidth_hz=float(th_hw))
            if th_frac <= 0:
                # A notch that removes no energy is a failure to solve, not a
                # 100%-mismatched solution: scoring it as err=1.0 is what let an
                # unfilled sentinel be reported as a real measurement.
                continue
            err = abs(th_frac - ch_frac) / ch_frac
            if best is None or err < best[0]:
                best = (err, float(ch_hw), float(th_hw))

        if best is None:
            # Unsolved. Record it as such; the fallback halfwidth exists only so
            # the batch stays a single tensor, and unlike the old nominal-value
            # fallback it is clamped to the largest SAFE candidate so a failed
            # solve can no longer smuggle a band onto a tooth harmonic.
            ch_hws[b] = float(min(float(halfwidth_hz), float(candidates.max())))
            th_hws[b] = 0.0
        else:
            solved[b] = True
            ch_hws[b], th_hws[b] = best[1], best[2]
        margins[b] = max_safe - ch_hws[b]

    chatter_batch, chatter_fracs = _apply(fn_w[:, None], ch_hws)
    tooth_batch, tooth_fracs = _apply(tooth_w, th_hws)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(tooth_fracs - chatter_fracs) / np.where(
            chatter_fracs > 0, chatter_fracs, np.nan
        )
    # A solved window always has ch_frac > 0 at the same halfwidth, so this only
    # tightens the bookkeeping: whatever cannot yield a finite relative error is
    # counted as unsolved rather than dropped silently by nanmax.
    solved &= np.isfinite(rel)

    # DEGENERATE-DENOMINATOR GUARD. `rel` divides by the chatter fraction, so a
    # window that removed a vanishing amount emits an enormous "error" that is
    # really a failure to notch anything. Strictly-positive is not enough of a
    # liveness test -- one near-empty FFT bin passes it. The 2026-07-30 GL run
    # showed the damage: max_rel_err ~2e4 beside rel_err_p90 ~0.98, i.e. a
    # handful of degenerate windows setting the headline for all 512.
    degenerate = solved & (chatter_fracs < float(min_chatter_frac))
    n_degenerate = int(degenerate.sum())
    solved &= ~degenerate

    rel_solved = rel[solved]
    n_solved = int(solved.sum())
    # Kept distinct from the degenerate count: "no matched pair exists" and "the
    # pair exists but removed nothing measurable" are different failures with
    # different fixes, and collapsing them is how the 1.0 sentinel survived.
    n_unsolved = int(n_b - n_solved - n_degenerate)
    unsolved_frac = float(n_unsolved / n_b) if n_b else 1.0
    # The veto counts both, because either one leaves a window unmatched.
    unusable_frac = float((n_unsolved + n_degenerate) / n_b) if n_b else 1.0
    max_rel = float(np.max(rel_solved)) if rel_solved.size else None
    rel_p90 = float(np.percentile(rel_solved, 90)) if rel_solved.size else None

    return {
        "chatter": measure_probes(model, chatter_batch, labels),
        "tooth": measure_probes(model, tooth_batch, labels),
        "energy_match": {
            # Applied batch means: what the model actually saw, unsolved windows
            # included (their tooth halfwidth is 0, i.e. no notch at all).
            "chatter_frac": float(chatter_fracs.mean()),
            "tooth_frac": float(tooth_fracs.mean()),
            "chatter_halfwidth_hz": float(ch_hws.mean()),
            "chatter_halfwidth_nominal_hz": float(halfwidth_hz),
            "tooth_halfwidth_hz": float(th_hws.mean()),
            "tooth_centers_hz": [float(np.mean(tooth_w[:, k])) for k in range(n_harm)],
            # Aim: the mean removed fraction cannot show a window whose band
            # missed its own chatter peak, but the minimum can.
            "chatter_frac_min": float(chatter_fracs.min()),
            "fn_hz_mean": float(fn_w.mean()),
            "fn_hz_sd": float(fn_w.std()),
            "chatter_tooth_margin_hz": (
                float(np.min(margins)) if n_harm else None
            ),
            # Honesty block: a failed solve is never reported as an error value.
            "n_windows": int(n_b),
            "n_solved": n_solved,
            "n_unsolved": n_unsolved,
            "unsolved_fraction": unsolved_frac,
            "max_unsolved_frac": float(max_unsolved_frac),
            # Excluded for a vanishing denominator, not for failing to solve.
            # Reported separately so the two failure modes stay distinguishable.
            "n_degenerate_chatter": n_degenerate,
            "min_chatter_frac": float(min_chatter_frac),
            "unusable_fraction": unusable_frac,
            "max_rel_err": max_rel,
            "rel_err_p90": rel_p90,
            "matched": bool(
                max_rel is not None
                and max_rel <= float(match_tol)
                and unusable_frac <= float(max_unsolved_frac)
            ),
        },
    }


# --------------------------------------------------------------------- #
# Plots (values only ever come from run_fuzzer results)
# --------------------------------------------------------------------- #
def _panel(ax, result: dict, baseline: dict) -> None:
    levels = result["levels"]
    xs = np.arange(len(levels))
    auc = [r["mean"]["auc"] for r in result["results"]]
    sd = [r["sd"]["auc"] for r in result["results"]]
    drop = [r["mean"]["delay_drop"] for r in result["results"]]
    ax.plot(xs, auc, "o-", color="tab:blue", label="AUC")
    ax.fill_between(xs, np.array(auc) - np.array(sd), np.array(auc) + np.array(sd),
                    color="tab:blue", alpha=0.2)
    ax.axhline(baseline["auc"], color="tab:blue", ls=":", lw=1, label="baseline AUC")
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(xs, [str(level) for level in levels])
    ax.set_xlabel(f"{result['name']} level")
    ax.set_ylabel("per-window AUC")
    twin = ax.twinx()
    twin.plot(xs, drop, "s--", color="tab:red", label="delay-ablation drop")
    twin.axhline(DROP_FLOOR, color="tab:red", ls=":", lw=1)
    twin.set_ylabel("delay-ablation AUC drop", color="tab:red")
    twin.set_ylim(bottom=min(0.0, min(drop)))
    ax.set_title(result["name"])


def plot_degradation_panel(result: dict, baseline: dict, path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(5.5, 4))
    _panel(ax, result, baseline)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_fuzz_dashboard(all_results: list[dict], baseline: dict, path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    n = max(1, len(all_results))
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    for ax, res in zip(flat[:n], all_results):
        _panel(ax, res, baseline)
    fig.suptitle("Physics understanding vs fuzz level (per-window AUC + delay-ablation drop)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
