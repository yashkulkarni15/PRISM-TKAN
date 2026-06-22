# Research Update 2 Addendum: Phase 2.5 Plan

This short section can be added after the current risks/next steps section.

## Phase 2.5: Noise Characterization and Mode-Aware Filtering

Prof. Abell's feedback raised a gap in the current Phase 2 pipeline: the clean-data PSD/JS result is strong, but the project still needs a formal way to identify measurement noise, decide when enough noise has been filtered, and avoid confusing noise with chatter. I am adding Phase 2.5 as a support layer before the spectral loss is used in the T-KAN-PINN.

The key idea is to separate each signal into three buckets:

1. forced cutting response, mainly tooth-pass frequency and harmonics,
2. chatter response, including energy near the natural frequency and regenerative sidebands,
3. measurement/process noise, which should be broadband, incoherent, or outside the physics bands.

Filtering should only target the third bucket. It should not remove tooth-pass behavior or chatter signatures.

The first checkpoint will be the Dataset 1 / 61 / 91 mirror ladder: clean, 1 percent noise, and 5 percent noise. A proposed noise metric must move upward across the ladder and pass a one-sided clean-versus-5-percent distribution test. Metrics that fail this direction-safe mirror test will not be used in the noise score.

The Phase 2.5 metrics will include PSD noise floor ratio, off-harmonic energy ratio, spectral entropy, spectral flatness, wavelet-MAD noise sigma, cross-channel coherence across force and position signals, chatter-band energy near `fn`, and sideband energy around `fn +/- k*tpf`. The promoted metrics will be normalized against clean per-signal baselines before they feed mode labels such as TPF-dominated clean, near-boundary transition, chatter-dominated, broadband-noise-dominated, mixed TPF plus chatter, low-speed edge, or metrics unavailable.

Filtering will be conservative. A filtered signal is accepted only if:

1. its noise score moves into the clean baseline range for the same mode,
2. tooth-pass peaks are preserved,
3. chatter-band and sideband energy are not erased,
4. the stable-reference JS trend still rises across `b/blim = 1`,
5. the noisy mirror moves closer to its clean twin in PSD/JS space.

This stage does not block the early Phase 3 baseline. The MLP with hardcoded `tau` and MSE loss can start while Phase 2.5 runs in parallel. Phase 2.5 gates the later JS spectral-loss stage, where the model first needs a noise-aware confidence weight. It also supports the final XAI goal: DiCE counterfactuals should report when a recommendation is noise-limited and should refuse to suggest control changes when the measurement is too noisy to trust.
