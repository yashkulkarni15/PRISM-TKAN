# Physics-Floor Stressors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the level at which the trained TKAN stops using the regenerative delay, using stressors that destroy the physics while holding the power spectrum fixed, and metrics that stay comparable across folds.

**Architecture:** New spectral primitives live in `src/pri2m/xai/spectral.py`. New stressors register into the existing `FUZZERS` dict in `xai/degradation.py` as `Fuzzer` entries, so the sweep loop (`run_fuzzer`) is untouched. Metric changes are additive fields on existing manifest records. The paired notch contrast is a sibling `contrasts` key in the manifest, never nested inside `fuzzers`, so all Update 10 consumers keep working.

**Tech Stack:** Python 3.13, numpy, torch, pytest. No new dependencies — `numpy.fft` supplies everything the spectral work needs.

**Spec:** `docs/superpowers/specs/2026-07-20-physics-floor-stressors-design.md`

## Global Constraints

- **Do NOT `git add`, `git commit`, or `git push`.** Hard project rule. Work stays in the working tree; the user commits manually under their own name. The commit steps normally required by the writing-plans skill are deliberately omitted from every task below.
- **TDD is mandatory.** Write the failing test, run it, observe the correct failure, then implement. Never write production code first.
- Suite baseline is **603 passed / 1 skipped**. Every task ends green with zero failures. Full run takes ~3.5 min.
- Run from `/Users/yash/Downloads/PRI2M/SINDy_DDE`. Python is `/opt/anaconda3/bin/python`.
- **Eval-only.** Nothing in this plan retrains a model or touches training code.
- Existing stressor levels are **append-only**. Every level measured in Update 10 must be re-measured at exactly the same value.
- **TENSOR LAYOUT — read this before writing any array code.** `batch["window"]`
  is stored as **`(B, L, C)`** — batch, *length*, channels. It is NOT `(B, C, L)`.
  Confirm by reading `fuzz_input_smooth` in `degradation.py`, which permutes
  `(0, 2, 1)` with the comment `# [B, C, L]` before doing any time-axis work and
  permutes back afterwards.
  The spectral helpers in `spectral.py` take **`(C, L)`**, so every call site
  must transpose: `win[b].T` going in, `.T` again coming out. Getting this wrong
  runs the FFT along the 4-element channel axis instead of the 1024-sample time
  axis. It fails *silently* — the Task 1 unit tests feed `(C, L)` directly and
  would still pass — and you would not discover it until a cluster sweep
  returned meaningless numbers.
- Test fixtures: `tests/test_degradation.py` defines its own local `_model()`
  and `_batch()` helpers (top of file). Use those for anything needing a model
  or batch. `tests/_phase3_synth.py` builds NPZ trees and tables, not models —
  it has no `tiny_tkan_model`.

## File Structure

| File | Responsibility |
|---|---|
| `src/pri2m/xai/spectral.py` (new) | FFT surrogate generation and notch design. Pure numpy, no torch, no model. |
| `src/pri2m/xai/degradation.py` (modify) | New `fuzz_*` stressors, `_apply_*` adapters, `FUZZERS` entries, extended levels, new metrics. |
| `scripts/fuzz_phase3.py` (modify) | Write the `contrasts` manifest key. Stressor selection already exists as `--fuzzers`; do not add a second flag. |
| `tests/test_xai_spectral.py` (new) | Property tests for the spectral primitives. |
| `tests/test_degradation.py` (modify) | Stressor registration, metrics, back-compat. |

`spectral.py` is deliberately separate: `degradation.py` is already the largest module in `xai/`, and the spectral helpers are pure array math with no model dependency, which makes them cheap to test in isolation.

---

### Task 1: Phase randomization primitive

**Files:**
- Create: `src/pri2m/xai/spectral.py`
- Test: `tests/test_xai_spectral.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `phase_randomize(x, alpha, seed=0, amplitude_adjust=True) -> np.ndarray` where `x` is `(C, L)` float; `rank_remap(y, x) -> np.ndarray`.

**Design note — the AAFT tradeoff.** Rank-remapping onto the original amplitude
distribution perturbs the magnitude spectrum; you cannot preserve both exactly
without iterating (that is what IAAFT does). We take one AAFT pass and *bound*
the spectral deviation rather than claiming it is zero. Tests assert exact
magnitude preservation with `amplitude_adjust=False`, and bounded deviation
(< 5% relative) with it on. This corrects spec §4.1, which overstates the
guarantee.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_xai_spectral.py
import numpy as np
import pytest
from pri2m.xai.spectral import phase_randomize


def _sig(C=4, L=256, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    base = np.sin(2 * np.pi * 7 * t / L) + 0.3 * np.sin(2 * np.pi * 23 * t / L)
    return np.stack([base * (c + 1) + 0.1 * rng.standard_normal(L) for c in range(C)])


def test_phase_randomize_preserves_magnitude_spectrum_exactly():
    x = _sig()
    y = phase_randomize(x, alpha=1.0, seed=0, amplitude_adjust=False)
    mx = np.abs(np.fft.rfft(x, axis=-1))
    my = np.abs(np.fft.rfft(y, axis=-1))
    np.testing.assert_allclose(my, mx, rtol=1e-8, atol=1e-8)


def test_phase_randomize_alpha_zero_is_identity():
    x = _sig()
    y = phase_randomize(x, alpha=0.0, seed=0, amplitude_adjust=False)
    np.testing.assert_allclose(y, x, rtol=1e-8, atol=1e-8)


def test_phase_randomize_output_is_real_and_finite():
    x = _sig()
    y = phase_randomize(x, alpha=1.0, seed=3)
    assert y.dtype.kind == "f"
    assert np.all(np.isfinite(y))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_xai_spectral.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pri2m.xai.spectral'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pri2m/xai/spectral.py
"""Spectral surrogates and notch filters for the physics-floor stressors.

Pure numpy: no torch, no model. Every function takes and returns a real
``(C, L)`` array so the callers in ``degradation.py`` stay thin.
"""
from __future__ import annotations

import numpy as np

__all__ = ["phase_randomize", "rank_remap"]


def rank_remap(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Remap ``y`` onto ``x``'s amplitude distribution, preserving y's rank order.

    Per channel. This is the "amplitude-adjusted" step of an AAFT surrogate: it
    guarantees the surrogate has exactly the original amplitude histogram, so a
    model's reaction cannot be attributed to a distribution shift.
    """
    out = np.empty_like(y)
    for c in range(y.shape[0]):
        ranks = np.argsort(np.argsort(y[c]))
        out[c] = np.sort(x[c])[ranks]
    return out


def phase_randomize(
    x: np.ndarray, alpha: float, seed: int = 0, amplitude_adjust: bool = True
) -> np.ndarray:
    """Randomize Fourier phase by fraction ``alpha``, holding magnitude fixed.

    ONE perturbation array is drawn and applied to EVERY channel, so
    inter-channel relative phase (x/y position coupling, force/position
    coupling) survives untouched. Randomizing per channel would destroy that
    too and confound "lost regeneration" with "lost cross-channel structure".

    ``alpha=0`` is the identity; ``alpha=1`` fully randomizes phase. Magnitude
    is preserved exactly unless ``amplitude_adjust`` is set, which trades a
    small spectral deviation for an exact amplitude distribution.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (C, L) array, got shape {x.shape}")
    length = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    mag, phase = np.abs(spec), np.angle(spec)

    rng = np.random.default_rng(int(seed))
    dphi = rng.uniform(-np.pi, np.pi, size=spec.shape[-1])
    # DC (and Nyquist for even length) must stay real, else irfft silently
    # discards the imaginary part and the magnitude is NOT preserved there.
    dphi[0] = 0.0
    if length % 2 == 0:
        dphi[-1] = 0.0

    out = np.fft.irfft(mag * np.exp(1j * (phase + float(alpha) * dphi)), n=length, axis=-1)
    if amplitude_adjust:
        out = rank_remap(out, x)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_xai_spectral.py -v`
Expected: 3 passed

- [ ] **Step 5: Add the remaining property tests**

```python
def test_phase_randomize_preserves_amplitude_distribution_when_adjusted():
    x = _sig()
    y = phase_randomize(x, alpha=1.0, seed=1, amplitude_adjust=True)
    for c in range(x.shape[0]):
        np.testing.assert_allclose(np.sort(y[c]), np.sort(x[c]), rtol=1e-8, atol=1e-8)


def test_amplitude_adjust_keeps_spectral_deviation_bounded():
    # AAFT trades exact spectrum for exact amplitude distribution. Bound it.
    x = _sig()
    y = phase_randomize(x, alpha=1.0, seed=2, amplitude_adjust=True)
    mx = np.abs(np.fft.rfft(x, axis=-1))
    my = np.abs(np.fft.rfft(y, axis=-1))
    rel = np.linalg.norm(my - mx) / np.linalg.norm(mx)
    assert rel < 0.05, f"AAFT perturbed the spectrum by {rel:.3f}"


def test_phase_randomize_shares_perturbation_across_channels():
    # Inter-channel phase differences must be preserved: that is what keeps
    # x/y and force/position coupling intact.
    x = _sig()
    y = phase_randomize(x, alpha=1.0, seed=4, amplitude_adjust=False)
    px = np.angle(np.fft.rfft(x, axis=-1))
    py = np.angle(np.fft.rfft(y, axis=-1))
    dx = np.unwrap(px[0] - px[1])
    dy = np.unwrap(py[0] - py[1])
    np.testing.assert_allclose(dy, dx, rtol=1e-6, atol=1e-6)


def test_phase_randomize_actually_changes_the_signal():
    x = _sig()
    y = phase_randomize(x, alpha=1.0, seed=5, amplitude_adjust=False)
    assert np.linalg.norm(y - x) / np.linalg.norm(x) > 0.1


def test_phase_randomize_rejects_non_2d():
    with pytest.raises(ValueError, match=r"\(C, L\)"):
        phase_randomize(np.zeros(16), alpha=1.0)
```

- [ ] **Step 6: Run and confirm all pass**

Run: `python -m pytest tests/test_xai_spectral.py -v`
Expected: 8 passed. If `test_phase_randomize_shares_perturbation_across_channels` fails, the perturbation is being drawn per channel — it must be drawn once and broadcast.

---

### Task 2: Magnitude randomization (the dual control)

**Files:**
- Modify: `src/pri2m/xai/spectral.py`
- Test: `tests/test_xai_spectral.py`

**Interfaces:**
- Consumes: `rank_remap` from Task 1.
- Produces: `magnitude_randomize(x, alpha, seed=0) -> np.ndarray`.

This is the control that makes Task 1 interpretable. Phase kills reliance and magnitude does not → double dissociation. Both kill it → Task 1 proves nothing. Energy is renormalized so the surrogate carries the same total power as the original, removing loudness as a confound.

- [ ] **Step 1: Write the failing test**

```python
def test_magnitude_randomize_preserves_phase_exactly():
    x = _sig()
    y = magnitude_randomize(x, alpha=1.0, seed=0)
    px = np.angle(np.fft.rfft(x, axis=-1))
    py = np.angle(np.fft.rfft(y, axis=-1))
    # compare on bins with non-negligible magnitude; phase of a ~0 bin is noise
    mx = np.abs(np.fft.rfft(x, axis=-1))
    keep = mx > 1e-6 * mx.max()
    np.testing.assert_allclose(np.cos(py[keep]), np.cos(px[keep]), atol=1e-6)
    np.testing.assert_allclose(np.sin(py[keep]), np.sin(px[keep]), atol=1e-6)


def test_magnitude_randomize_alpha_zero_is_identity():
    x = _sig()
    np.testing.assert_allclose(magnitude_randomize(x, 0.0, seed=0), x, rtol=1e-8, atol=1e-8)


def test_magnitude_randomize_changes_the_spectrum():
    x = _sig()
    y = magnitude_randomize(x, alpha=1.0, seed=1)
    mx = np.abs(np.fft.rfft(x, axis=-1))
    my = np.abs(np.fft.rfft(y, axis=-1))
    assert np.linalg.norm(my - mx) / np.linalg.norm(mx) > 0.1


def test_magnitude_randomize_preserves_total_energy():
    x = _sig()
    y = magnitude_randomize(x, alpha=1.0, seed=2)
    np.testing.assert_allclose(
        np.sum(y**2, axis=-1), np.sum(x**2, axis=-1), rtol=1e-6
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_xai_spectral.py -k magnitude -v`
Expected: FAIL — `ImportError: cannot import name 'magnitude_randomize'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/pri2m/xai/spectral.py` and extend `__all__` with `"magnitude_randomize"`:

```python
def magnitude_randomize(x: np.ndarray, alpha: float, seed: int = 0) -> np.ndarray:
    """Scramble the magnitude spectrum by fraction ``alpha``, holding phase fixed.

    The exact dual of :func:`phase_randomize`, and the control that makes it
    interpretable. Total per-channel energy is renormalized to the original so
    that a performance change cannot be attributed to loudness.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (C, L) array, got shape {x.shape}")
    length = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    mag, phase = np.abs(spec), np.angle(spec)

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(mag.shape[-1])
    new_mag = (1.0 - float(alpha)) * mag + float(alpha) * mag[:, perm]

    out = np.fft.irfft(new_mag * np.exp(1j * phase), n=length, axis=-1)

    # Renormalize per channel in the TIME domain so energy matches exactly.
    # Doing it in the frequency domain via raw sum(mag**2) is only approximate
    # (~6e-5): Parseval for rfft weights the DC bin (and Nyquist, for even
    # length) by 1 and every other bin by 2, so an unweighted spectral-sum
    # match does not imply an exact time-domain energy match once the
    # permutation moves magnitude into or out of those two special bins.
    # Rescaling the reconstructed signal sidesteps the weighting entirely and
    # is exact by construction; it is a positive real per-channel scalar, so
    # phase is untouched -- which this control depends on absolutely.
    e_x = np.sum(x**2, axis=-1, keepdims=True)
    e_out = np.sum(out**2, axis=-1, keepdims=True)
    scale = np.sqrt(np.divide(e_x, e_out, out=np.ones_like(e_x), where=e_out > 0))
    return out * scale
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_xai_spectral.py -v`
Expected: 12 passed.

Note: `test_magnitude_randomize_preserves_total_energy` relies on Parseval, so
the time-domain energy check follows from the spectral renormalization. If it
fails by a constant factor, the renormalization is being applied to power
instead of amplitude.

---

### Task 3: Matched-energy notch filter

**Files:**
- Modify: `src/pri2m/xai/spectral.py`
- Test: `tests/test_xai_spectral.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `notch(x, fs, centers, halfwidth_hz) -> tuple[np.ndarray, float]` returning `(filtered, energy_removed_frac)`; `match_notch_halfwidth(x, fs, centers, target_frac, tol=0.01) -> float`.

The matched-energy search is what makes the chatter-vs-tooth contrast mean anything: without it the comparison measures how much was deleted rather than what was deleted.

- [ ] **Step 1: Write the failing test**

```python
def test_notch_removes_energy_at_the_target_frequency():
    fs, L = 1000.0, 1024
    t = np.arange(L) / fs
    x = np.stack([np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 300 * t)])
    y, frac = notch(x, fs=fs, centers=[100.0], halfwidth_hz=10.0)
    # the 100 Hz component should be largely gone, 300 Hz largely intact
    spec = np.abs(np.fft.rfft(y, axis=-1))[0]
    freqs = np.fft.rfftfreq(L, 1 / fs)
    assert spec[np.argmin(np.abs(freqs - 100))] < 1e-6 * spec.max()
    assert spec[np.argmin(np.abs(freqs - 300))] > 0.1 * spec.max()
    assert 0.0 < frac < 1.0


def test_notch_zero_halfwidth_is_identity():
    fs, L = 1000.0, 512
    x = np.random.default_rng(0).standard_normal((2, L))
    y, frac = notch(x, fs=fs, centers=[100.0], halfwidth_hz=0.0)
    np.testing.assert_allclose(y, x, rtol=1e-8, atol=1e-8)
    assert frac == 0.0


def test_match_notch_halfwidth_hits_target_within_tolerance():
    fs, L = 1000.0, 1024
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, L))
    target = 0.05
    hw = match_notch_halfwidth(x, fs=fs, centers=[200.0, 400.0], target_frac=target)
    _, frac = notch(x, fs=fs, centers=[200.0, 400.0], halfwidth_hz=hw)
    assert abs(frac - target) <= 0.01 * target + 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_xai_spectral.py -k notch -v`
Expected: FAIL — `ImportError: cannot import name 'notch'`

- [ ] **Step 3: Write minimal implementation**

Add to `spectral.py`, extending `__all__` with `"notch"` and `"match_notch_halfwidth"`:

```python
def notch(
    x: np.ndarray, fs: float, centers, halfwidth_hz: float
) -> tuple[np.ndarray, float]:
    """Zero the spectrum within ``halfwidth_hz`` of each centre frequency.

    Returns the filtered signal and the fraction of total spectral energy
    removed, which the matched-energy contrast needs in order to be fair.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (C, L) array, got shape {x.shape}")
    length = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(length, 1.0 / float(fs))

    mask = np.ones(freqs.shape, dtype=float)
    if float(halfwidth_hz) > 0:
        for f0 in centers:
            mask[np.abs(freqs - float(f0)) <= float(halfwidth_hz)] = 0.0

    before = float(np.sum(np.abs(spec) ** 2))
    filtered = spec * mask[None, :]
    after = float(np.sum(np.abs(filtered) ** 2))
    removed = 0.0 if before <= 0 else (before - after) / before
    return np.fft.irfft(filtered, n=length, axis=-1), removed


def match_notch_halfwidth(
    x: np.ndarray,
    fs: float,
    centers,
    target_frac: float,
    tol: float = 0.01,
    max_iter: int = 60,
) -> float:
    """Bisect the notch halfwidth until it removes ``target_frac`` of the energy.

    Without this the chatter-band vs tooth-pass contrast is confounded: the
    condition that deletes more energy wins regardless of what it deleted.
    """
    lo, hi = 0.0, float(fs) / 2.0
    best, best_err = hi, float("inf")
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        _, frac = notch(x, fs=fs, centers=centers, halfwidth_hz=mid)
        err = abs(frac - target_frac)
        # Track the CLOSEST halfwidth seen, not the last one tried. Removed
        # energy is a STEP function of halfwidth (it only changes when the
        # band crosses an FFT bin), so an exact target can be unreachable and
        # the loop can exhaust max_iter legitimately -- at which point
        # returning the last midpoint would return an arbitrary value rather
        # than the best fit, and the caller would silently mis-match energy.
        if err < best_err:
            best, best_err = mid, err
        if err <= tol * target_frac:
            return mid
        if frac < target_frac:
            lo = mid
        else:
            hi = mid
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_xai_spectral.py -v`
Expected: 15 passed.

---

### Task 4: Reliance metrics

**Files:**
- Modify: `src/pri2m/xai/degradation.py`
- Test: `tests/test_degradation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `normalized_reliance(auc, delay_drop, min_headroom=0.01) -> float | None`; `reliance_breakdown(levels, norm) -> dict` with keys `reliance_half_life`, `reliance_floor`; `spectral_floor_report(mean_auc, baseline_auc_delay_off) -> dict` with keys `spectral_floor_auc`, `asymptote_auc`, `converged`, `asymptote_gap`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_degradation.py
from pri2m.xai.degradation import (
    normalized_reliance, reliance_breakdown, spectral_floor_report,
)


def test_normalized_reliance_divides_by_available_headroom():
    assert normalized_reliance(1.0, 0.25) == pytest.approx(0.5)
    assert normalized_reliance(0.75, 0.125) == pytest.approx(0.5)


def test_normalized_reliance_is_none_at_or_near_chance():
    assert normalized_reliance(0.5, 0.0) is None
    assert normalized_reliance(0.505, 0.01) is None   # inside the 0.51 guard
    assert normalized_reliance(0.4, 0.0) is None


def test_normalized_reliance_reproduces_the_fold2_noise_cliff():
    # The finding that motivated the metric: flat to sigma=0.5, cliff after.
    auc = [0.994, 0.917, 0.867, 0.756]
    drop = [0.266, 0.226, 0.126, 0.008]
    norm = [normalized_reliance(a, d) for a, d in zip(auc, drop)]
    assert norm[0] == pytest.approx(0.539, abs=0.01)
    assert norm[1] == pytest.approx(0.542, abs=0.01)   # still flat
    assert norm[3] == pytest.approx(0.030, abs=0.01)   # collapsed


def test_reliance_breakdown_finds_half_life_and_floor():
    levels = [0.0, 1.0, 2.0, 3.0]
    norm = [0.50, 0.40, 0.20, 0.02]
    out = reliance_breakdown(levels, norm)
    assert out["reliance_half_life"] == 2.0    # first <= 0.25
    assert out["reliance_floor"] == 3.0        # first <= 0.05


def test_reliance_breakdown_returns_none_when_never_crossed():
    out = reliance_breakdown([0.0, 1.0], [0.5, 0.49])
    assert out["reliance_half_life"] is None
    assert out["reliance_floor"] is None


def test_reliance_breakdown_ignores_none_entries():
    out = reliance_breakdown([0.0, 1.0, 2.0], [0.5, None, 0.1])
    assert out["reliance_half_life"] == 2.0


def test_spectral_floor_report_flags_convergence_to_the_delay_off_auc():
    out = spectral_floor_report([0.99, 0.90, 0.86, 0.855], baseline_auc_delay_off=0.85)
    assert out["spectral_floor_auc"] == pytest.approx(0.85)
    assert out["asymptote_auc"] == pytest.approx(0.8575)
    assert out["converged"] is True
    assert out["asymptote_gap"] == pytest.approx(0.0075, abs=1e-6)


def test_spectral_floor_report_reports_not_converged_when_still_falling():
    out = spectral_floor_report([0.99, 0.90, 0.80, 0.70], baseline_auc_delay_off=0.5)
    assert out["converged"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_degradation.py -k "reliance or spectral_floor" -v`
Expected: FAIL — `ImportError: cannot import name 'normalized_reliance'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/pri2m/xai/degradation.py`, near `breakdown_points`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_degradation.py -v`
Expected: all pass, including the 9 pre-existing degradation tests.

---

### Task 5: `lag_tau_shuffle` stressor

**Files:**
- Modify: `src/pri2m/xai/degradation.py`
- Test: `tests/test_degradation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fuzz_lag_tau_shuffle(batch, frac, seed=0) -> tuple[dict, dict]`, matching the `(new_batch, info)` shape of `fuzz_tau_jitter`.

Destroys τ-periodicity by permuting blocks of length τ. Because it is a permutation, the amplitude distribution is preserved *exactly* — which is what distinguishes it from noise. At `frac=1.0` this is the clean fully-destroyed-delay condition that `tau_jitter` cannot provide (see spec §3.3).

- [ ] **Step 1: Write the failing test**

Note the `(B, L, C)` layout in every shape below — 128 and 64 are *lengths*, 4 is channels.

```python
def test_lag_tau_shuffle_is_a_permutation_of_samples():
    # (B, L, C) -- B=3 windows of length 128 with 4 channels
    batch, _ = _batch(B=3, L=128, seed=0)
    batch["tau"] = torch.full((3,), 16, dtype=torch.long)
    new, info = dg.fuzz_lag_tau_shuffle(batch, frac=1.0, seed=0)
    for b in range(3):
        for c in range(4):
            np.testing.assert_allclose(
                np.sort(new["window"][b, :, c].numpy()),
                np.sort(batch["window"][b, :, c].numpy()),
            )
    assert info["frac_blocks_permuted"] > 0


def test_lag_tau_shuffle_frac_zero_is_identity():
    batch, _ = _batch(B=2, L=64, seed=1)
    new, info = dg.fuzz_lag_tau_shuffle(batch, frac=0.0, seed=0)
    np.testing.assert_allclose(new["window"].numpy(), batch["window"].numpy())
    assert info["frac_blocks_permuted"] == 0.0


def test_lag_tau_shuffle_does_not_mutate_the_original_batch():
    batch, _ = _batch(B=2, L=64, seed=2)
    before = batch["window"].clone()
    dg.fuzz_lag_tau_shuffle(batch, frac=1.0, seed=0)
    np.testing.assert_allclose(batch["window"].numpy(), before.numpy())


def test_lag_tau_shuffle_actually_reorders_at_frac_one():
    batch, _ = _batch(B=2, L=64, seed=3)
    new, _ = dg.fuzz_lag_tau_shuffle(batch, frac=1.0, seed=0)
    assert not np.allclose(new["window"].numpy(), batch["window"].numpy())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_degradation.py -k lag_tau -v`
Expected: FAIL — `ImportError: cannot import name 'fuzz_lag_tau_shuffle'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_degradation.py -v`
Expected: all pass.

---

### Task 6: Register the new stressors and extend existing ranges

**Files:**
- Modify: `src/pri2m/xai/degradation.py`
- Test: `tests/test_degradation.py`

**Interfaces:**
- Consumes: `phase_randomize`, `magnitude_randomize` (Tasks 1-2), `fuzz_lag_tau_shuffle` (Task 5).
- Produces: `FUZZERS` entries `phase_randomize`, `magnitude_randomize`, `lag_tau_shuffle`; extended `levels` on four existing entries.

- [ ] **Step 1: Write the failing test**

```python
def test_new_stressors_are_registered_as_input_fuzzers():
    from pri2m.xai.degradation import FUZZERS
    for name in ("phase_randomize", "magnitude_randomize", "lag_tau_shuffle"):
        assert name in FUZZERS, f"{name} not registered"
        assert FUZZERS[name].kind == "input"
        assert FUZZERS[name].levels[0] == 0.0, "level 0 must be the identity"


def test_existing_levels_are_appended_to_never_rewritten():
    # Update 10 numbers must stay reproducible: every previously measured level
    # must still be present, at the same value, in the same order.
    from pri2m.xai.degradation import FUZZERS
    assert FUZZERS["input_noise"].levels[:7] == (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    assert FUZZERS["input_noise"].levels == (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0)
    assert FUZZERS["tau_jitter"].levels[:5] == (0.0, 0.05, 0.10, 0.25, 0.50)
    assert FUZZERS["tau_jitter"].levels == (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.9)
    assert FUZZERS["input_smooth"].levels == (1, 2, 4, 8, 16, 32, 64, 128)
    assert FUZZERS["weight_noise"].levels == (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    assert FUZZERS["spline_coarsen"].levels == (5, 4, 3, 2, 1)


def test_tau_jitter_never_reaches_the_degenerate_clamp():
    # At frac >= 1.0 the implementation clamps ~half the windows to tau=1
    # rather than jittering them, which is a different experiment.
    from pri2m.xai.degradation import FUZZERS
    assert max(FUZZERS["tau_jitter"].levels) < 1.0


def test_phase_randomize_fuzzer_leaves_the_original_batch_untouched():
    import torch
    from pri2m.xai.degradation import FUZZERS
    rng = np.random.default_rng(3)
    w = torch.as_tensor(rng.standard_normal((2, 4, 64)), dtype=torch.float32)
    before = w.clone()
    batch = {"window": w, "tau": torch.as_tensor([8, 8], dtype=torch.long)}
    _, new, _ = FUZZERS["phase_randomize"].apply(None, batch, 1.0, 0)
    np.testing.assert_allclose(w.numpy(), before.numpy())
    assert not np.allclose(new["window"].numpy(), before.numpy())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_degradation.py -k "registered or appended or degenerate" -v`
Expected: FAIL — `AssertionError: phase_randomize not registered`

- [ ] **Step 3: Write minimal implementation**

Add the import at the top of `degradation.py`:

```python
from pri2m.xai.spectral import magnitude_randomize, phase_randomize
```

Add the batch-level wrappers and adapters:

```python
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
```

Then replace the `FUZZERS` dict:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_degradation.py -v`
Expected: all pass.

- [ ] **Step 5: Wire the new metrics into `run_fuzzer`'s summary**

In `run_fuzzer`, after the existing mean-AUC/mean-drop aggregation, add the
normalized curve and the new breakdown fields to the returned dict, alongside
(never replacing) the existing `breakdown` key:

```python
    norm = [normalized_reliance(a, d) for a, d in zip(mean_auc, mean_drop)]
    result["normalized_reliance"] = norm
    result["reliance_breakdown"] = reliance_breakdown(list(fuzzer.levels), norm)
    result["spectral_floor"] = spectral_floor_report(
        mean_auc, baseline.get("auc_delay_off") if baseline else None
    )
```

`run_fuzzer` must accept the fold `baseline` dict to read `auc_delay_off`; add
it as a keyword argument defaulting to `None` so existing callers keep working.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -q`
Expected: `603 + N passed, 1 skipped`, zero failures.

---

### Task 7: Matched-energy band-notch contrast

**Files:**
- Modify: `src/pri2m/xai/degradation.py`
- Modify: `scripts/fuzz_phase3.py`
- Test: `tests/test_degradation.py`

**Interfaces:**
- Consumes: `notch`, `match_notch_halfwidth` (Task 3); `measure_probes` (existing).
- Produces: `run_band_notch_contrast(model, batch, labels, fs, fn, tpf, halfwidth_hz=...) -> dict` with keys `chatter`, `tooth`, `energy_match`.

This is a paired contrast, not a graded sweep, so it does not go in `FUZZERS`. It is written to a new top-level `contrasts` manifest key.

- [ ] **Step 1: Write the failing test**

Uses the `_model()` / `_batch()` helpers defined at the top of
`tests/test_degradation.py`. `_batch()` supplies `fs=10_000` and `tpf=200` but
**no `fn`** — natural frequency lives in the phase-3 table, not the batch — so
`fn` is an explicit parameter of the contrast.

```python
def test_band_notch_contrast_matches_energy_between_conditions():
    model, (batch, labels) = _model(), _batch(B=8, L=64)
    out = dg.run_band_notch_contrast(
        model, batch, labels, fs=10_000.0, fn=950.0, tpf=200.0, halfwidth_hz=200.0
    )
    assert set(out) >= {"chatter", "tooth", "energy_match"}
    em = out["energy_match"]
    assert abs(em["chatter_frac"] - em["tooth_frac"]) <= 0.01 * em["chatter_frac"] + 1e-3
    for cond in ("chatter", "tooth"):
        assert "auc" in out[cond] and "delay_drop" in out[cond]


def test_band_notch_contrast_does_not_mutate_the_original_batch():
    model, (batch, labels) = _model(), _batch(B=8, L=64)
    before = batch["window"].clone()
    dg.run_band_notch_contrast(
        model, batch, labels, fs=10_000.0, fn=950.0, tpf=200.0, halfwidth_hz=200.0
    )
    np.testing.assert_allclose(batch["window"].numpy(), before.numpy())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_degradation.py -k band_notch -v`
Expected: FAIL — `ImportError: cannot import name 'run_band_notch_contrast'`

- [ ] **Step 3: Write minimal implementation**

```python
def run_band_notch_contrast(
    model, batch, labels, fs: float, fn: float, tpf: float,
    halfwidth_hz: float = 50.0, n_harmonics: int = 3,
) -> dict:
    """Chatter-band vs tooth-pass notch at MATCHED removed energy.

    The comparison is the result: if the model is a spectral chatter-energy
    detector the chatter notch hurts more; if it is doing regeneration the
    tooth notch does. Matching the removed energy is what makes that a fair
    question rather than a measure of how much was deleted.
    """
    win = batch["window"].detach().cpu().numpy()               # (B, L, C)

    def _apply(centers, halfwidth):
        out = np.empty_like(win)
        fracs = []
        for b in range(win.shape[0]):
            filt, frac = notch(win[b].T, fs=fs, centers=centers, halfwidth_hz=halfwidth)
            out[b] = filt.T                                    # back to (L, C)
            fracs.append(frac)
        new = dict(batch)
        new["window"] = torch.as_tensor(out, dtype=batch["window"].dtype)
        return new, float(np.mean(fracs))

    chatter_batch, chatter_frac = _apply([fn], halfwidth_hz)

    tooth_centers = [tpf * (k + 1) for k in range(int(n_harmonics))]
    tooth_hw = float(np.mean([
        match_notch_halfwidth(win[b].T, fs=fs, centers=tooth_centers, target_frac=chatter_frac)
        for b in range(win.shape[0])
    ]))
    tooth_batch, tooth_frac = _apply(tooth_centers, tooth_hw)

    return {
        "chatter": measure_probes(model, chatter_batch, labels),
        "tooth": measure_probes(model, tooth_batch, labels),
        "energy_match": {
            "chatter_frac": chatter_frac,
            "tooth_frac": tooth_frac,
            "chatter_halfwidth_hz": float(halfwidth_hz),
            "tooth_halfwidth_hz": tooth_hw,
            "tooth_centers_hz": tooth_centers,
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_degradation.py -k band_notch -v`
Expected: PASS

- [ ] **Step 5: Write the contrast into the manifest as a sibling key**

In `scripts/fuzz_phase3.py`, after the fuzzer sweep populates
`manifest["fuzzers"]`, add:

```python
    manifest["contrasts"] = {
        "band_notch": _safe("band_notch", lambda: degradation.run_band_notch_contrast(
            model, batch, labels,
            fs=float(np.mean(batch["fs"].numpy())),
            fn=float(fn_hz),
            tpf=float(np.mean(batch["tpf"].numpy())),
        ))
    }
```

`_safe` is the local error-capturing closure already defined inside `main()`
(around line 132) — this code goes inside `main()` alongside the existing
`_safe(...)` calls, not at module level.

**Sourcing `fn`.** `fs` and `tpf` are present on the batch; `fn` (natural
frequency) is **not** — it lives in the phase-3 table. Read the mean `fn` for
the sampled windows from the table the script has already loaded (`--table`),
the same way other per-window physics columns are read elsewhere in the
pipeline. If `fn` is absent or non-finite for the sample, skip the contrast and
record `{"error": "fn unavailable"}` rather than substituting a guess — a
notch centred on the wrong frequency would silently invalidate the comparison.

**`contrasts` must be a sibling of `fuzzers`, never nested inside it.**

- [ ] **Step 6: Write the back-compatibility test**

Every Update 10 consumer (`plot_fuzz_dashboard`, `plot_fuzz_replication.py`)
iterates `manifest["fuzzers"]`. Adding `contrasts` as a sibling must not disturb
them. This test reads a real returned manifest, so it is the genuine check:

```python
def test_update10_manifest_schema_is_still_honoured():
    import json, pathlib
    p = pathlib.Path("results/phase3/xai/fuzz/fold2/fuzz_manifest.json")
    if not p.exists():
        pytest.skip("Update 10 manifest not present locally")
    m = json.loads(p.read_text())
    # the five original stressors, each with the keys the plotters rely on
    assert set(m["fuzzers"]) >= {
        "input_noise", "input_smooth", "tau_jitter", "spline_coarsen", "weight_noise"
    }
    for rec in m["fuzzers"].values():
        assert {"levels", "results", "breakdown"} <= set(rec)
    # contrasts, if present, must be a SIBLING of fuzzers -- never nested inside
    assert "band_notch" not in m["fuzzers"]
```

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q`
Expected: zero failures.

---

### Task 8: Confirm stressor selection covers the new registry entries

**Files:**
- Test: `tests/test_fuzz_phase3_cli.py`

**Interfaces:**
- Consumes: `FUZZERS` (Task 6).
- Produces: nothing new.

**No implementation is required.** `scripts/fuzz_phase3.py` already ships a
`--fuzzers` flag (line ~80, `"'all' or a comma-separated registry subset"`)
backed by `_parse_fuzzers`, which defaults to the whole registry and raises
`ValueError` on unknown names — exactly the selection behaviour this plan
needs. Do NOT add a parallel `--stressors` flag; that would be a second way to
do the same thing.

This task only proves the existing flag picks up the newly registered
stressors, since `_parse_fuzzers` validates against `FUZZERS` at call time.

- [ ] **Step 1: Write the failing test**

```python
def test_fuzzers_flag_accepts_the_new_physics_stressors():
    from scripts.fuzz_phase3 import _parse_fuzzers
    chosen = _parse_fuzzers("phase_randomize,magnitude_randomize,lag_tau_shuffle")
    assert chosen == ["phase_randomize", "magnitude_randomize", "lag_tau_shuffle"]


def test_fuzzers_all_includes_the_new_physics_stressors():
    from pri2m.xai.degradation import FUZZERS
    from scripts.fuzz_phase3 import _parse_fuzzers
    chosen = _parse_fuzzers("all")
    assert set(chosen) == set(FUZZERS)
    assert {"phase_randomize", "magnitude_randomize", "lag_tau_shuffle"} <= set(chosen)
```

Match the import mechanism already used in `tests/test_fuzz_phase3_cli.py` — if
that file loads the script by path rather than importing it as a module, follow
its existing pattern rather than changing it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fuzz_phase3_cli.py -k physics_stressors -v`
Expected: FAIL — `ValueError: unknown fuzzers: ['phase_randomize', ...]` if Task 6
has not landed, which is the correct failure and confirms the test is coupled to
the registry.

- [ ] **Step 3: No implementation**

If Task 6 is complete these pass with no source change. If they fail, the defect
is a missing `FUZZERS` entry in Task 6, not a missing CLI feature — fix it there.

- [ ] **Step 4: Final full-suite gate**

Run: `python -m pytest -q`
Expected: `603 + N passed, 1 skipped`, zero failures. Record the exact line.

---

## After the build (user, on Great Lakes)

1. `./sync_phase3_tkan_to_hpc.sh`
2. New stressors only, all five folds:
   `FUZZERS=phase_randomize,magnitude_randomize,lag_tau_shuffle bash slurm/run_fuzz_all_folds.sh`
   The runner already accepts `--fuzzers`; add a `FUZZERS` env passthrough to
   `run_fuzz_all_folds.sh` and `phase3_fuzz.sbatch` if absent, using the same
   pattern as the `TSR_WINDOWS` passthrough added 2026-07-20.
3. Full re-run including the extended ranges on the existing five.
4. rsync results back; read `reliance_breakdown` and `spectral_floor` per fold.

**Read the dual control before believing anything from `phase_randomize`.** If
`magnitude_randomize` destroys reliance just as thoroughly, the phase result is
uninformative and must be reported as such.
