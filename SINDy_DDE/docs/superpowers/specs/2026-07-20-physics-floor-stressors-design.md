# Physics-Floor Stressors — Design

**Date:** 2026-07-20
**Branch:** `phase3-tkan`
**Status:** approved in brainstorming; ready for implementation plan
**Builds on:** `2026-07-16-phase3-fuzz-and-spline-xai-design.md` (the fuzziness suite this extends)

## 1. Motivation

The fuzziness suite (Update 10) asked whether the white-box explanation survives
stress. It answered "yes" — AUC never fell below 0.75 under any stressor. That
reads as a robustness win, but re-reading the manifests shows it is equally a
statement that **we never found the edge**. Of the fifteen breakdown points the
suite computes (5 stressors × 3 thresholds, fold 2), **fourteen are `null`** —
the stressor never pushed the model past the threshold anywhere in its tested
range. The single exception, `input_noise → drop_below_005 = 2.0`, lands on the
*last tested level*, with nothing beyond it.

So the question "how far can the model be degraded before it stops understanding
the physics?" is currently unanswered, not answered in the affirmative.

Two further problems motivate the metric work in §3:

1. **Every existing stressor is a generic corruption.** Noise, blur, and jitter
   degrade everything at once. When delay-reliance dies under them we cannot say
   whether the physics was *destroyed* or merely *buried*.
2. **The reported reliance metric is confounded.** `delay_drop` is measured
   against an AUC that is itself falling. As AUC approaches chance there is
   mechanically no headroom left to drop, so reliance must trend to zero
   regardless of what the model is doing.

### 1.1 The finding that motivated the metric change

Normalizing reliance by available headroom — `delay_drop / (AUC − 0.5)` —
transforms the fold-2 noise sweep from a gentle fade into a flat line and a
cliff:

| noise σ | AUC | raw `delay_drop` | normalized reliance |
|---|---|---|---|
| 0.0 | 0.994 | 0.266 | 0.539 |
| 0.05 | 0.991 | 0.263 | 0.535 |
| 0.1 | 0.987 | 0.261 | 0.535 |
| 0.2 | 0.965 | 0.247 | 0.530 |
| 0.5 | 0.917 | 0.226 | 0.542 |
| 1.0 | 0.867 | 0.126 | 0.343 |
| 2.0 | 0.756 | 0.008 | 0.030 |

Physics reliance is **invariant to noise up to half the signal amplitude** and
collapses between 1× and 2×. The raw curve's apparent gradual decline from σ=0
to σ=0.5 was almost entirely shrinking headroom, not the model abandoning the
physics. This is a sharper and more defensible claim than Update 10 makes, and
it comes from data already on disk.

## 2. Goals and non-goals

**Goals**

- Find the **physics-reliance floor**: the degradation level at which the model
  stops using the regenerative delay, whether or not it remains accurate.
- Separate "the model does physics" from "the model reads the spectrum" with a
  stressor that holds the power spectrum fixed and destroys only phase.
- Report floors that are **comparable across folds**.
- Extend every existing stressor until its breakdown points stop returning
  `null`.

**Non-goals**

- No retraining. Everything is eval-only on the five existing `best_auc.pt`
  checkpoints, like the rest of the fuzz suite.
- Not the abstraction axis (quantization, downsampling, channel dropping,
  summary-statistic substitution). Deferred by decision; revisit after this
  lands.
- Not a fix for the 0.99 per-window ceiling. This study moves *away* from the
  ceiling, so compression affects the baseline reliance value but not the shape
  of the floor curves. Noted as a caveat, not addressed here.

## 3. Metrics

### 3.1 Normalized reliance (new primary metric)

```
normalized_reliance = delay_drop / (auc - 0.5)   if auc > 0.51 else None
```

Interpretation: the fraction of the model's *available* discriminative headroom
that depends on the delayed input.

The `0.51` guard is deliberate and is not an epsilon for numerical safety. Below
roughly one point of headroom the ratio's denominator is dominated by sampling
noise in `auc`, so the quantity becomes unstable long before it becomes
undefined at exactly 0.5. Reporting `None` there is the honest answer: at chance
accuracy there is no reliance to measure, and a large ratio computed from a
near-zero denominator would be an artifact, not a finding.

Raw `delay_drop` continues to be recorded. The floor is defined on the
normalized curve; the raw curve stays in the manifest so nothing already
published becomes unreproducible.

### 3.2 Reliance half-life (replaces absolute thresholds)

The current `breakdown_points` uses a fixed `DROP_FLOOR = 0.05`. Baseline
reliance varies 0.230–0.486 across folds, so that constant demands a 78% loss on
fold 0 and a 90% loss on fold 1 — the folds are not comparable.

Replace with fractions of **each fold's own baseline** normalized reliance:

- `reliance_half_life`: first level where normalized reliance ≤ 50% of baseline.
- `reliance_floor`: first level where normalized reliance ≤ 10% of baseline.

Both `None` if never crossed — but with the extended ranges of §5, `None` should
become rare, and a remaining `None` is then a real finding ("survives the entire
tested range") rather than an artifact of stopping early.

The existing absolute thresholds are retained in the manifest for continuity
with Update 10.

### 3.3 Spectral floor — and a convergence test

The accuracy the model retains with **no usable delay information** is not a
quantity we need to estimate by extrapolation: it is already measured at every
level as `auc_delay_off`, the AUC with the delayed input zeroed. Fold 2's
baseline is 0.728; folds 1 and 3 are 0.500 and 0.510. That is the spectral
floor — how much of the task is doable without the physics, and therefore a
ceiling on what the physics is worth.

This converts §3.3 from a curve-fit into a **falsifiable prediction**: a
stressor that genuinely destroys the model's delay usage should drive AUC down
toward that fold's `auc_delay_off`, not below it and not to some unrelated
asymptote. Recorded per stressor:

- `spectral_floor_auc` = baseline `auc_delay_off` for that fold.
- `asymptote_auc` = mean AUC over the two most-degraded levels.
- `converged: bool` — those two levels differ by < 0.01.
- `asymptote_gap` = `asymptote_auc − spectral_floor_auc`.

A gap near zero means the stressor fully removed delay usage. A large positive
gap means it did not, however severe it looked. This is a much stronger check
than "the curve flattened."

**Design consequence — `tau_jitter` cannot be pushed past frac ≈ 0.9.**
`fuzz_tau_jitter` computes `tau * (1 + sign*frac)` clamped by `np.maximum(1, …)`.
At `frac = 1.0` every negative-sign window collapses to `tau = 1`; at `frac = 2.0`
the product goes negative and clamps to the same degenerate value. So high-frac
jitter does not measure "randomly wrong delay" — it pins roughly half the
windows to a fixed degenerate delay, which is a different experiment. Jitter is
therefore extended only to 0.75 and 0.9 (§5), and the fully-destroyed-delay
condition comes from `lag_tau_shuffle` at α=1.0, which is a clean permutation
with no degenerate case.

## 4. New stressors

All are `kind="input"`, plug into the existing `FUZZERS` registry, and require no
change to `run_fuzzer`.

### 4.1 `phase_randomize` — the centerpiece

Destroys the regenerative correlation while holding the power spectrum exactly
fixed. Chatter's regenerative term lives in the correlation between `x(t)` and
`x(t−τ)`, which is phase information; tooth-pass and chatter-band *energy* is
magnitude. A model doing physics should collapse; a model reading spectral
energy should not notice.

Algorithm, per window:

1. Real FFT each channel.
2. Draw **one** random phase perturbation array `Δφ ~ U(−π, π)` over the positive
   frequency bins and **apply the same array to every channel**. This preserves
   inter-channel relative phase, so cross-channel coupling (x↔y position,
   force↔position) survives and cannot confound the result.
3. Set `φ_new = φ_orig + α · Δφ`, preserving Hermitian symmetry so the inverse
   transform is real.
4. Inverse FFT.
5. **Amplitude-adjust**: rank-order remap the result onto the original
   amplitude distribution (the AAFT step). Without this, a phase-randomized
   surrogate is Gaussianized, and a performance drop could be the model reacting
   to a distribution shift rather than to destroyed phase.

α = 0 is the identity; α = 1 is a full surrogate. The magnitude spectrum is
preserved exactly at every α by construction.

Levels: `(0.0, 0.1, 0.25, 0.5, 0.75, 1.0)`.

### 4.2 `magnitude_randomize` — the dual control

The exact inverse: scramble the magnitude spectrum, preserve phase. This is what
makes §4.1 interpretable.

- Phase kills reliance, magnitude does not → **double dissociation**; the model
  depends on phase-coherent regeneration. Claim is airtight.
- Both kill reliance → the stressors are simply breaking the signal, and §4.1
  proves nothing. This outcome must be reported as such.

Levels: same as §4.1.

### 4.3 `lag_tau_shuffle`

Cut each window into blocks of length τ (`tau_samples = round(fs / tpf)`, the
same quantity the delay ablation uses) and permute a fraction α of the blocks.
Preserves within-block dynamics and the amplitude distribution exactly — it is a
permutation — while destroying τ-periodicity. Blunter than §4.1 but trivially
interpretable and aimed at exactly the delay.

Levels: `(0.0, 0.1, 0.25, 0.5, 1.0)`.

### 4.4 `band_notch` — a paired contrast, not a sweep

Two conditions at matched energy removal:

- **chatter notch**: attenuate a band around `fn`.
- **tooth notch**: attenuate bands around `tpf` and its harmonics.

The *comparison* is the result. If the model is a spectral chatter-energy
detector, the chatter notch hurts more; if it is doing regeneration, the tooth
notch does.

**Matched energy is essential.** Choose the chatter notch width first, measure
the fraction of total energy it removes, then widen or narrow the tooth notch to
remove the same fraction (bisection on width, tolerance 1%). Record
`energy_removed_frac` for both so the match is auditable. Without this the
contrast measures how much was deleted rather than what was deleted.

`fn` and `tpf` come from the phase-3 table per window, as elsewhere in the
pipeline.

## 5. Extended ranges (existing stressors)

| stressor | current | extended |
|---|---|---|
| `input_noise` | …, 1.0, 2.0 | …, 2.0, **4.0, 8.0** |
| `tau_jitter` | …, 0.25, 0.50 | …, 0.50, **0.75, 0.9** |
| `input_smooth` | …, 16, 32 | …, 32, **64, 128** |
| `weight_noise` | …, 0.2, 0.5 | …, 0.5, **1.0** |
| `spline_coarsen` | 5…1 | unchanged (already bottoms at a single segment) |

`tau_jitter` stops at 0.9 for the degeneracy reason given in §3.3: at frac ≥ 1.0
the existing implementation clamps roughly half the windows to `tau = 1` rather
than jittering them. The fully-destroyed-delay condition is supplied by
`lag_tau_shuffle` at α=1.0 instead.

All extensions **append** to the existing level tuples, so every previously
measured level is re-measured at the same value and the Update 10 numbers stay
reproducible.

## 6. Architecture

The `Fuzzer` dataclass (`name`, `kind`, `levels`, `stochastic`, `apply`) and the
`run_fuzzer` sweep loop are unchanged. Each new stressor is:

- a pure `fuzz_*(batch, level, seed) -> dict` function in `xai/degradation.py`,
  matching the signature style of `fuzz_input_noise`;
- a thin `_apply_*` adapter;
- one `FUZZERS` registry entry.

`band_notch` is the exception: it is a paired contrast rather than a graded
sweep, so it gets its own runner function returning both conditions plus the
matched-energy audit, and is recorded in the manifest under a new top-level
`contrasts` key rather than under `fuzzers`.

**Manifest back-compatibility is a hard requirement.** `plot_fuzz_dashboard`,
`plot_fuzz_replication.py`, and the Update 10 figures all iterate
`manifest["fuzzers"]`. Adding `contrasts` as a *sibling* key rather than nesting
anything new inside `fuzzers` keeps every existing consumer working untouched;
new per-stressor metric fields are additive within each fuzzer record. A test
must load an existing Update 10 manifest from `results/phase3/xai/fuzz/fold2/`
and assert the plotting entry points still run against it.

Metric changes (§3.1–3.3) live in `breakdown_points` and the `run_fuzzer`
summary block. `scripts/fuzz_phase3.py` gains `--stressors a,b,c` (default: all)
so a run can target only the new stressors without re-running the five existing
ones, which take the bulk of the sweep time.

New spectral helpers (FFT surrogate generation, rank remapping, notch design)
go in a new `xai/spectral.py` rather than swelling `degradation.py`, which is
already the largest module in `xai/`.

## 7. Testing

TDD throughout; every behavior gets a failing test first.

**Property tests (these are the load-bearing ones — they verify the stressors
actually do what the science claims):**

- `phase_randomize` preserves the magnitude spectrum to floating-point
  tolerance at every α, including α=1.
- `phase_randomize` at α=0 is the identity.
- `phase_randomize` output is real (Hermitian symmetry maintained).
- `phase_randomize` preserves the amplitude distribution after the AAFT step
  (compare sorted values).
- `phase_randomize` applies the *same* perturbation across channels: inter-channel
  cross-spectral phase is preserved.
- `magnitude_randomize` preserves phase and destroys magnitude — the mirror
  assertions.
- `lag_tau_shuffle` is a permutation: multiset of samples is unchanged.
- `band_notch` matched-energy search converges to within 1%.
- Normalized reliance returns `None` at or below chance rather than dividing by
  a non-positive headroom.
- `reliance_half_life` is `None` when the curve never crosses, and returns the
  first crossing level when it does.

**Non-mutation:** every new input stressor must leave the original batch
untouched, enforced by the suite's existing `self_check` mechanism (which
verified `max_abs_dev ≤ 5.5e-07` across all five folds).

**Regression:** the existing five stressors must produce byte-identical results
at their existing levels. The extended ranges only append.

Suite baseline is 603 passed / 1 skipped; it must end green with zero failures.

## 8. Risks

**The result may be uncomfortable, and that is the point.** If reliance survives
phase randomization, then the three interventions of Updates 8–11 are showing
that the model uses *something* at lag τ that is not phase-coherent
regeneration, and the delay story becomes more complicated than currently told.
This was raised explicitly in brainstorming and the decision was to ask the
question before publication rather than after. If it comes back that way, it is
reported plainly, as the TSR and concept-probe negatives were.

**The dual control may collapse the whole design.** If both phase and magnitude
randomization destroy reliance, neither is informative about phase specifically.
Mitigation: this is precisely why §4.2 exists, and it is checked before any
claim is made from §4.1.

**Ceiling caveat persists.** Baseline per-window AUC is 0.986–0.996, so the
*baseline* reliance value is measured under compression. Floor *locations*
should be robust since they occur far from the ceiling, but absolute reliance
magnitudes remain compressed and must not be over-read.

## 9. Execution

Local build and tests; the sweep runs on Great Lakes via the existing
`slurm/phase3_fuzz.sbatch` and `run_fuzz_all_folds.sh`, which already accept
per-run configuration. Eval-only on the five existing checkpoints — no GPU
training, same cost profile as the Update 10 sweep, which completed cleanly on
all five folds.
