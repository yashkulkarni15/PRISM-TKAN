---
title: "Research Update 2"
geometry: margin=1in
fontsize: 11pt
---

## 1. State Before This Work

The last update finished the SINDy-DDE analysis. It found that one term, the regenerative chip term chip_x, is the clearest physics signal for chatter. It shows up 3 to 7 times more often in unstable cuts, and this holds across all 120 datasets. It also found that adding 5 percent noise wipes out that signal. That set a clear limit for the simple regression method and was the reason to move to a physics-informed neural model.

This update is about the next stage. The job here was to set up the full Phase 1 to Phase 3 plan: get everything running on the Great Lakes cluster, build a separate physics check based on Altintas-Budak stability theory, and build the signal analysis pipeline that turns raw machining data into a stability score the neural model can later learn from. None of this existed before. The earlier work was loose scripts run by hand. This stage turns it into a clean code base with cluster jobs that anyone can rerun.

The final goal is a T-KAN-PINN model with an explainability layer. It will use DiCE-style counterfactuals that only change the knobs an operator can actually turn, like spindle speed and depth of cut, and suggest small, physically sensible changes that move a cut from unstable back to stable.

## 2. What Was Built

### 2.1 Code base and Great Lakes setup

A clean package was set up under src/pri2m. It holds the shared config, the data loaders, the Altintas-Budak check, the signal and divergence tools, the Phase 3 stage gates, the model checkpoint cleanup, and the reporting helpers. SLURM wrappers were added so each step runs on the cluster in a fixed order. The big preprocessed data files stay on Great Lakes. Only code, configs, summary tables, logs, and final figures are kept on the local machine. A handoff file tracks the status, blockers, fixes, and next goals so the work can be picked up without guessing.

The SINDy summary was re-checked on Great Lakes and matched: 32,168 preprocessed rows, 120 datasets, and 0 fit failures. The six development datasets were confirmed (clean 1, 4, 29, 35, 39, and the 5 percent noisy copy 91).

### 2.2 Altintas-Budak stability verifier

A standard milling stability lobe model was built as a separate physics check. It works out the machine's vibration response on each axis, applies the cutting factors, and solves for the deepest cut that stays stable and the spindle speeds that go with it. This lets us test whether the data really behaves like the textbook chatter theory, instead of just trusting the stable/unstable labels in the dataset.

The check runs in three modes. The textbook test passed, which shows the math and pipeline work. The dev metadata test passed, which shows the six dev datasets have the properties we expect. The third test, comparing predicted depth limits against the data, ran but is only a warning for now. The exact textbook constants and the way the data generator sets up the cut are not known yet, so these depth numbers are a placeholder, not a final pass. This is the main open physics question for Phase 3.

### 2.3 Phase 2 signal extraction and PSD pipeline

A signal extractor was built to read the raw H5 files on Great Lakes and save small bundles for four signals: tool position x, tool position y, force x, and force y. A small pilot ran first on the dev and anomaly datasets plus noisy dataset 91, with zero failures. The full clean run then covered all 60 clean datasets.

The PSD pipeline turns each signal into a power spectrum, which simply shows how much energy sits at each frequency. It then tags each file with a spectral sanity category, such as clean tooth-pass behavior, a chatter-like peak near the machine's natural frequency, a mixed case, or an indeterminate case. The full clean run processed 18,221 files and 72,884 signal rows with zero failures. The tags flagged strong odd behavior in datasets 44 to 48, which lines up with the long-standing note that these cuts may be over-called as unstable.

## 3. The Divergence Metric Problem and the Fix

The first plan was to measure how different the stable and unstable spectra are inside each b/blim bin, using a standard distance score (Jensen-Shannon divergence). When this ran, it gave zero usable rows. The reason is simple once you see it. The number b/blim is the stability line itself. Cuts below 1.0 are stable and cuts above 1.0 are unstable, almost by definition. So a single bin almost never has both stable and unstable files in it, which means there is nothing to compare inside a bin. This stayed true even on the full dataset, so it is a built-in property of the data, not a small-sample fluke.

The fix was to change the question. Instead of comparing the two labels inside one bin, each bin is now compared against a fixed stable reference spectrum, built from the same signal, tooth count, and cut direction. This measures how far the spectrum has drifted away from a stable cut as the cut gets deeper. This shape is also a better fit for the neural model later, because it gives a smooth penalty that grows as the cut gets more unstable, instead of an on/off jump.

Two methodology guards were added so this metric is trustworthy:

1. The stable reference is now chosen by physics only. It uses the lowest b/blim samples in each facet and never uses the stability label to pick the reference. This avoids label leakage into something that is meant to be a spectral loss.
2. A hard stability floor was added. Reference samples must have b/blim at or below 0.80. If a facet does not have enough samples under that floor, the row is marked as insufficient rather than quietly using the least unstable samples. This protects the thin facets that show up at full scale.

One honest caveat is recorded in the notes. The lowest b/blim bin overlaps the reference set by design, so its divergence is an anchor near zero and not an independent test point. The curve should be read by its shape above the reference band, especially the rise across b/blim equal to 1.0. Both guards and this caveat are now applied and recorded in the production run metadata.

## 4. Results

The full clean run shows the expected physics clearly, and more cleanly than the earlier pilot. For the biggest group of cuts (force x, four teeth, climb cut, about fifteen thousand files), the score stays very low and flat while the cut is stable, then climbs steadily and keeps speeding up once the cut crosses the stability line.

| b/blim bin | JS divergence (force x, 4 teeth, climb) | label |
|---|---|---|
| 0.00 to 0.50 | 0.00034 | stable |
| 0.50 to 0.75 | 0.00825 | stable |
| 0.75 to 1.00 | 0.01342 | stable |
| 1.00 to 1.25 | 0.02124 | unstable |
| 1.25 to 1.50 | 0.04190 | unstable |
| 1.50 to 2.00 | 0.08225 | unstable |
| 2.00 to 3.00 | 0.14274 | unstable |
| 3.00 to 5.00 | 0.27375 | unstable |
| 5.00 and above | 0.41837 | unstable |

What this means in plain terms: while the cut is stable, its spectrum looks almost identical to a known stable cut, so the score is near zero. The moment the cut passes the stability line at b/blim equal to 1.0, the spectrum starts changing fast, and it keeps changing more as the cut goes deeper. By the deepest bin the score is about 31 times higher than the last stable bin, and more than an order of magnitude above the near-boundary stable region. The score does not just flip on and off at the boundary, it rises smoothly with how unstable the cut is. So it behaves less like a stable/unstable classifier and more like a severity measure for instability. That is exactly the behavior we want, because it gives the neural model a graded signal to learn from instead of a hard yes/no.

![Full 60 clean datasets, force x. The score stays low and flat while the cut is stable, then bends upward right at the stability line (b/blim equal to 1.0) and keeps climbing. The bend at 1.0 is the key feature: it shows the spectrum reacts to real instability, not to the label. The colored curves are different physical regimes (tooth count and cut direction). The four-tooth climb-cut curve is the smooth, well-populated one and should be read as the main result. The five-tooth and three-tooth curves rise more sharply but have thinner data support, so they are supportive rather than central. This makes the force channel the main candidate for the Phase 3 spectral loss.](results/figures/research_update_assets/full60_Forcex.png){width=85%}

![Full 60 clean datasets, tool position x. Same boundary-aligned shape as the force signal, but at a lower magnitude in the dominant four-tooth climb-cut regime. Tool position still carries stability information, just less strongly than force, so force channels look more sensitive for spectral loss design.](results/figures/research_update_assets/full60_xtpos.png){width=85%}

The full clean run produced 156 facet and bin rows across 60 datasets. Of these, 152 are in good standing and 4 are flagged as insufficient. All four flagged rows are the same thin cell (five teeth, climb cut, the 2 to 3 b/blim bin) that held only a single file, so the pipeline correctly marked it rather than producing a fake number. The stability floor stranded no facet. There were zero aggregation failures.

The noisy copy, dataset 91, was also run as a small early check. Its curve keeps the same broad direction but is flatter early, bumpier, and less certain. This agrees with the earlier SINDy finding that 5 percent noise weakens chatter signatures. It is shown only as motivation for the full noisy run, not as a robustness conclusion. The full noisy datasets 61 to 120 are needed before any production robustness claim.

![Noisy dataset 91, tool position x. The rise is weaker and the line is bumpier and less certain than the clean case. This is the expected effect of noise weakening chatter signatures and motivates the full noisy run, rather than settling the noise question.](results/figures/research_update_assets/noisy_xtpos.png){width=85%}

## 5. Where This Stands Against the Schedule

The clean side of Phase 2 is now done, not a pilot. PSD ran on all 60 clean datasets, the per-dataset stable versus unstable PSD profile plots were produced for all 60 (120 plots across the two reported signals, zero failures), the divergence aggregation ran on the full clean set, and the rise across b/blim equal to 1.0 is confirmed on production data against the revised criterion. So Phase 2.1 through 2.4 are met on clean data.

The metric change is documented and the success criterion was revised in writing before the curves were read, so the result above is judged against the corrected target and not the old one.

What is still open is the noisy side at full scale and the physics calibration. These are covered next.

## 6. Obstacles and Challenges

This stage hit several real obstacles. They are listed here plainly so the path is clear.

1. Great Lakes path not mounted locally. The shared project path was not visible from the local machine, so the local environment can only hold code, configs, summaries, and final figures. The heavy data stays on the cluster. This shaped the whole local versus cluster split.
2. Transfer and path naming. The feature CSV folder name carried a space, which broke an early copy and split files into the wrong place. The folder convention was changed to a no-space name and every script and SLURM file was updated to match.
3. The within-bin divergence dead end. The first divergence design produced zero usable rows. This was not a code bug. The b/blim ratio is the stability boundary itself, so a bin is almost always all stable or all unstable. The metric had to be redesigned, not patched.
4. Label leakage risk in the fix. The first version of the new metric could have used the stability label to pick the stable reference. That would leak labels into something meant to be a spectral loss. The reference selection was moved to a physics only rule.
5. Thin facet protection. Without a floor, a facet with no truly stable samples would have used the least unstable samples as the reference and produced confident looking nonsense. A hard b/blim floor of 0.80 and an explicit insufficient flag were added.
6. Altintas-Budak calibration still open. The exact textbook constants and the data generator engagement convention are not available yet, so strict blim validation cannot pass. This is the main physics anchor risk for Phase 3 and only the data owner can unblock it.
7. Schedule pressure. The full 60 clean run was behind the planned date for a short period. It is now complete, but the noisy run at scale is still outstanding.

## 7. Risks and Next Steps

One verification item is now resolved. The b_over_blim provenance is verified: the CSV satisfies b_over_blim = b_mm / blim, labels match int(b_over_blim >= 1), and in-range raw H5 boundary checks have zero mismatches. A small set of low-speed rows falls below the raw boundary grid and is reported separately. This is no longer a risk apart from that small low-speed edge case.

The main open risks before the T-KAN-PINN work are:

1. The Altintas-Budak blim calibration is still open and is the physics anchor for the Phase 3 residual term. This needs the exact reference constants and the generator engagement convention from the data source. It is the one item that only the data owner can unblock.
2. The clean side is done, but the noisy datasets 61 to 120 have only been run for the single pilot dataset 91. The noise robustness story is not yet production grade.

Next steps in order:

1. Run the full noisy extraction and divergence aggregation for datasets 61 to 120 so the noise robustness story matches the clean one.
2. Start the Phase 3 baseline: the MLP with hardcoded tau and an MSE loss.
3. Get the Altintas-Budak constants and the generator engagement convention from the data owner, so the Phase 3 physics residual is not blocked.
4. Later, integrate the JS spectral loss once the Phase 3 baseline works.

The Phase 3 model work can still start on schedule, because the planned unlock order puts the spectral loss at a later stage. So the remaining Phase 2 work and the early model work can move in parallel.

## 8. Phase 2.5: Noise Characterization and Mode-Aware Filtering

The feedback on the v2 report raised a real gap in the current pipeline. The clean PSD and JS results are strong, but there is still no formal way to measure how noisy a signal is, decide when enough noise has been filtered, or stop chatter from being confused with noise. A new stage, Phase 2.5, is added before the JS spectral loss is wired into the T-KAN-PINN. Phase 2.5 is a support layer for the spectral loss and the XAI step, not a parallel research thread. The detailed spec is in `docs/phase2_5_noise_mode_filtering.md`.

Each signal is read as a mixture of three things: the forced cutting response (tooth-pass frequency and harmonics), the chatter response (energy near the natural frequency `fn` and regenerative sidebands near `fn +/- k*tpf`), and measurement or process noise (broadband or incoherent energy that does not line up with the physics bands or with the other channels). Filtering should only target the third bucket. It should not remove tooth-pass behavior or chatter signatures.

The noisy datasets in the metadata follow a two-step noise ladder against the clean datasets. Noisy IDs 61 to 90 use 1 percent noise and pair as `clean = noisy - 60`, so 61 mirrors clean 1. Noisy IDs 91 to 120 use 5 percent noise and pair as `clean = noisy - 90`, so 91 also mirrors clean 1. This was verified on the metadata: matched IDs share `omega`, `b_mm`, `blim`, `b_over_blim`, `fn`, `teeth`, `cut_dir`, and `diameter_mm`. Only `noise_std` differs. The clean / 1 percent / 5 percent triple is the calibration anchor for every Phase 2.5 metric.

The first checkpoint compares Dataset 1, Dataset 61, and Dataset 91. A noise metric is only kept if its scores move upward monotonically across the three noise levels, by more than 2 sigma of the within-clean spread, or by a Mann-Whitney U with `p < 0.01` when per-file pairing is incomplete. Metrics that fail this gate are dropped before anything else is built on them. The production version then pools every available clean / 1 percent / 5 percent mirror ladder across datasets 1 to 60, rather than trusting the single 1/61/91 pilot ladder by itself.

The pilot ladder now shows the intended monotone behavior in the assigned modes. Dataset 1, the clean baseline, has 0 broadband-noise labels out of 102 files. Dataset 61, the 1 percent noisy mirror, has 54 broadband-noise labels out of 102 files. Dataset 91, the 5 percent noisy mirror, has 100 broadband-noise labels out of 102 files. The clean dataset still keeps its physics modes: 22 chatter-dominated files, 57 mixed TPF/chatter files, 9 near-boundary files, and 14 TPF-dominated clean files. This is the key pilot result because the noise gate detects increasing injected noise without crowding out chatter and TPF structure on the clean data.

The pilot calibration promoted 16 metrics across PSD, wavelet/temporal, and cross-channel families. The strongest promoted metrics include cross-channel coherence, wavelet MAD sigma, PSD noise-floor ratio, and PSD spectral flatness. The promoted metrics all moved upward on the mirror ladder, with very small Mann-Whitney p-values, and there were zero inverse indicators in this simulation case. This confirms that the simulation noise model is correlated across channels; the sign of cross-channel coherence still needs to be revalidated on real sensor noise.

The Phase 2.5 metric list is grouped into families so correlated estimators cannot double-vote. The PSD family includes noise floor ratio, off-harmonic energy ratio, spectral flatness, and spectral entropy. The wavelet and temporal family includes a wavelet MAD sigma (default Daubechies 4, level 1 detail) and autocorrelation decay rate. The cross-channel family uses coherence across Forcex, Forcey, xtpos, and ytpos. The chatter-specific family separates structured chatter from broadband noise using kurtosis in the `fn` band, sideband energy around `fn +/- k*tpf`, and cyclostationarity at the TPF period. In calibrated production mode, a file is labeled high-noise only when at least two of the three calibrated families agree, with no present family voting against. A family abstains when calibration finds no metrics for it on the current mirror ladder; this prevents the rule from silently failing closed when a dataset's noise model lands a family entirely in the inverse-indicator list (for example, simulation noise correlated across channels yields no promoted cross-channel metric, whereas real independent ADC noise would). The uncalibrated fallback is stricter and requires all three raw families to vote yes, so it is only a diagnostic path.

Each file then receives a mode label: `tpf_dominated_clean`, `near_boundary_transition`, `chatter_dominated`, `broadband_noise_dominated`, `mixed_tpf_chatter`, `low_speed_edge`, or `metrics_unavailable`. The first version uses rule-based thresholds on chatter-band energy, TPF harmonic energy, noise floor ratio, and `b_over_blim`, calibrated against clean datasets 1 to 60. The `low_speed_edge` label is reserved for rows outside a trusted spindle-speed or stability-boundary grid; simple TPF undersampling is tracked separately as a sampling-quality flag. A trained classifier replaces the rules if they disagree visibly with the JS-curve evidence.

Filtering is conservative. A filtered signal is only accepted if its noise score moves into the clean baseline range for the same mode, tooth-pass peak location and harmonic energy are preserved, chatter-band and sideband energy are not erased, the stable-reference JS trend still rises across `b/blim = 1`, and the noisy mirror moves measurably closer to its clean twin in PSD and JS space. If any check fails, the file is tagged noise-limited rather than force-filtered.

Phase 2.5 hands two things to Phase 3: a mode label per file as a conditioning input, and a noise confidence score per file as a sample weight for the spectral loss. Noise-limited files are downweighted, not dropped, so the model still sees the regime. The JS spectral loss is only unlocked after Phase 2.5 produces a validated noise score and filtering acceptance rule. The MLP baseline with hardcoded `tau` and MSE loss does not need Phase 2.5 and can start in parallel.

The same outputs support the XAI step. DiCE counterfactuals will report the noise score and the model confidence alongside any recommendation. For `tpf_dominated_clean` cuts the system can suggest efficiency-oriented moves; for `near_boundary_transition` and `chatter_dominated` it can suggest spindle and depth changes toward stable behavior; for `broadband_noise_dominated` cuts it refuses at the per-file level and emits a dataset-level "sensor needs attention" rollup, so the operator sees an aggregate signal instead of a long string of refusals.

Several items are deliberately out of scope for Phase 2.5 and tracked separately: within-cut non-stationarity using a spectrogram or short-time JS, label confidence from Altintas-Budak parameter perturbations, DiCE feasibility constraints on spindle, depth, deflection, and material removal rate, causal-versus-spurious feature sensitivity audits, tool wear and across-cut drift, and model-output uncertainty calibration. These are listed in the spec so they are not rediscovered later.

## 9. Time Spent

This update covers the week from May 11 to May 18, about 20 hours of work.

| Work | Hours |
|---|---|
| Code base setup: src/pri2m package, configs, and SLURM wrappers | 3.0 |
| Great Lakes bring-up: install and verify the SINDy summary and dev datasets | 2.0 |
| Altintas-Budak verifier: stability lobe model, three modes, Phase 1 check | 3.0 |
| Signal extraction: H5 to NPZ extractor, pilot, full 60 clean run | 2.5 |
| PSD pipeline: Welch spectra, sanity tags, 120 profile plots | 2.5 |
| Divergence metric fix: diagnose the dead end, rebuild, add the two guards | 3.0 |
| Full clean production run and reading the result | 1.5 |
| Verification scripts: provenance check and reference-rule check | 1.0 |
| Documentation: notes, handoff file, this update | 1.5 |
| Total hours | 20.0 |

The largest blocks were the verifier, the metric fix, and the code base setup. These were the parts that needed real design work and debugging rather than just running jobs.
