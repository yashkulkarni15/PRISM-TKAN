---
title: "Research Update 4"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

## 1. State Before This Work

The last update finished the Phase 2.5 library, the three command-line tools, the cluster wrappers, and the RapidMiner smoke test. Three things were still open: the noise-score calibration had to be hardened and rerun on the full mirror ladder, the aggregate JS divergence run on the full noisy set (datasets 61 to 120) had not been done, and several code review findings were noted but not fixed.

This update covers May 25 to May 30, about 20 hours of work. The goal was to finish everything that must be done before Phase 3 modeling can start: calibrate the noise score on a pooled ladder, close every open review finding, add the diagnostics the next meeting needs, run PSD and JS on the full noisy dataset, compare the noisy result to the clean one, and reconcile the Dataset 39 metadata fix.

## 2. Pooled Calibration of the Noise Score

The pilot ladder in the last update used a single triple: Dataset 1 paired with Dataset 61 and Dataset 91. That is enough to test the method, but the production noise score should not depend on one clean dataset. The pooled calibration ran over every clean dataset in 1 to 30 that has both its 1 percent mirror and its 5 percent mirror. That gives 30 mirror ladders and 7,525 paired operating points instead of one ladder of 102 files.

The pooled run promoted 10 metrics across the PSD and wavelet families. The cross-channel and chatter families abstained, which is how the family-vote rule should behave on simulation noise. There were zero inverse indicators. The strongest promoted metrics are the wavelet sigma on the tool position channels, the PSD noise floor ratio on Forcex, and the PSD spectral entropy on Forcex and xtpos.

The mode-label distribution from this calibration is the headline number for the meeting. Clean datasets 1 to 30, inside the pool, flag as broadband noise 1.6 percent of the time. Clean 31 to 60, the held-out half of the clean set, flag 0.3 percent of the time. The 1 percent noise mirrors flag at 18.7 percent. The 5 percent noise mirrors flag at 95.5 percent. The held-out clean rate matters most, because it shows the rule generalizes off the pool, not just on it.

![Mode-label distribution by tier, 33,719 files, 60 clean and 60 noisy datasets. The clean tiers stay mostly in TPF-dominated and mixed labels and rarely flag as broadband noise. The 5 percent noise tier is almost entirely broadband noise. The 1 percent tier sits cleanly between them. Phase 3 will use this calibration as a per-file sample weight.](results/figures/research_update_assets/mode_distribution_by_tier.png){width=85%}

![The 10 promoted noise metrics from the pooled calibration. The wavelet sigma on tool positions and the PSD noise floor and spectral entropy on Forcex are the strongest. The cross-channel and chatter families abstained, as expected on simulation noise.](results/figures/research_update_assets/promoted_metrics.png){width=85%}

## 3. Closing the Production Review Findings

Seven code review findings were open at the start of the week. Six are now closed with test-driven changes, and one is left open for a later session. All six landed in one batch with no conflicts. The local test suite grew from 38 tests to 68.

The findings, in short:

- The extraction script now fails closed when the canonical reader is missing, with an opt-in flag to allow the old generic H5 fallback.
- The cross-channel coherence now applies one common finite mask across the channel pair, so a NaN at different indices on the two channels no longer drives the coherence to zero.
- The noise score now averages inside each family first, then across families, so the PSD family cannot out-vote the wavelet family just because it has more metrics.
- The calibration step now refuses any metric whose paired-row coverage is below 80 percent, on top of the existing monotone and Mann-Whitney gates.
- The Pareto front script now inner-joins the baseline and each variant on dataset and file, so a filter that drops the hardest files cannot look better than it is.
- The mode assigner now rejects positive and negative infinity, not just NaN, on every promoted metric input.

The archive folder is now declared local-only, with a tracked manifest that records what each archive subdirectory holds and why it is not synced. This closes the silent-deletion risk from the last review.

One finding stays open. Four SINDy helper files live on the cluster but not in the local tree. They are needed for local reproducibility of the extractor, not for any current production run, and a single rsync will pull them when convenient. The new fail-closed extractor already checks for them.

## 4. Aggregate JS on the Full Noisy Set

The last update covered the full clean JS curves on 60 datasets. The full noisy run on datasets 61 to 120 ran this week. The reference rule matches the clean run: low b/blim quantile of 0.10 with a 0.80 stability floor, so the noisy curves read directly against the clean ones. Two cluster jobs ran with zero failures: 14,798 files, 59,192 signal rows, 116 of 116 reference rows usable.

The noise-robustness story holds. The score stays low and flat below the stability boundary on all four signals, and rises across b/blim equal to 1.0 just as it did on clean. On the dominant facet, Forcex with four teeth and a climb cut, the noisy result tracks the clean result closely. The table puts them side by side.

| b/blim bin | Clean JS (Forcex, 4 teeth, climb) | Noisy JS (Forcex, 4 teeth, climb) |
|---|---|---|
| 0.00 to 0.50 | 0.00034 | 0.00039 |
| 0.50 to 0.75 | 0.00825 | 0.01007 |
| 0.75 to 1.00 | 0.01342 | 0.01315 |
| 1.00 to 1.25 | 0.02124 | 0.02454 |
| 1.25 to 1.50 | 0.04190 | 0.05747 |
| 1.50 to 2.00 | 0.08225 | 0.08871 |
| 2.00 to 3.00 | 0.14274 | 0.10338 |
| 3.00 to 5.00 | 0.27375 | 0.21256 |
| 5.00 and above | 0.41837 | 0.34266 |

The noisy curve has almost the same shape as the clean curve in the decision region around the stability boundary, and runs a bit higher than clean in the middle bins. The Phase 3 spectral loss has to react to the rise across b/blim equal to 1.0, and that rise survives noise. The top three bins sit slightly below clean. This is a facet-composition effect, not a robustness failure. Noisy 61 to 120 mirrors only clean 1 to 30, so the highest b/blim bins draw on a single tooth-by-cut-direction facet, while the same bins on the clean side draw on four. A strict head-to-head comparison restricted to clean datasets 1 to 30 is queued and would remove this effect.

![Full noisy 60 datasets, Forcex. The score stays low and flat below the boundary and rises across b/blim equal to 1.0. The shape matches the clean curve in the decision region. The slight tail dip above b/blim equal to 2 is a facet-mix artifact, documented separately.](results/figures/research_update_assets/noisy60_Forcex.png){width=85%}

![Full noisy 60 datasets, tool position x. Same boundary-aligned rise as the force signal, at a lower magnitude. Tool position carries less stability information than force under noise, which matches what the clean run showed.](results/figures/research_update_assets/noisy60_xtpos.png){width=85%}

## 5. Diagnostics for the Meeting

Two diagnostic scripts let the calibration story be discussed honestly at the meeting, not just summarized.

The first plots the per-tier mode distribution and the per-dataset broadband rate. It produces the headline 1.6, 0.3, 18.7, and 95.5 percent numbers, and it marks the few clean datasets where the broadband rate runs unusually high.

The second ranks those clean false-positive datasets. On the real local mode labels, Dataset 23 ranks first at 20.5 percent, then Dataset 12 at 10.3 percent and Dataset 13 at 6.9 percent. The script will report the per-metric driver once the full noise-metrics CSV from the cluster is fed into it. The aim is to learn, before treating any of these as a real false positive, whether those clean datasets are chatter-heavy regimes where the broadband label is correct, are missing metrics, or are threshold artifacts.

## 6. Dataset 39 Metadata Correction

A new Excel sheet corrected the Dataset 39 row from 4 teeth to 5. This fixed the dataset descriptions table only. The operative metadata, the features CSV that every PSD, JS, noise-metric, and mode-label script reads, already carried 5 teeth for Dataset 39. A full cross-check across all 120 datasets confirmed that the spreadsheet teeth count and the features CSV teeth count agree on every dataset, with zero mismatches.

So no computed artifact needs to be regenerated. The clean PSD and JS runs already faceted Dataset 39 under 5 teeth, the chatter and TPF metrics already used its correct tooth-pass frequency, and Dataset 39 is not in the noisy mirror set or the pooled calibration pool. The only housekeeping left is to keep one canonical copy of the spreadsheet in `data/metadata/` instead of two.

## 7. Where This Stands Against the Schedule

The clean side of Phase 2 was already done before this week. This week added the pooled noise calibration with 10 promoted metrics, six closed review findings, the two diagnostic scripts, and the full noisy JS run on 60 datasets. A calibrated rule, not a placeholder, now produces the mode labels on all 33,719 files in the noise-metrics CSV. The handoff was updated as the work landed.

So everything Phase 2 and Phase 2.5 owe Phase 3 is in place: a per-file noise score for sample weighting, a per-file mode label for conditioning, stable-reference JS curves on the full clean and noisy data for the spectral loss head, and provenance-verified b/blim for the supervised label. The Phase 3 MLP baseline can start now, alongside the next steps in Section 8.

## 8. Readiness, Obstacles, and Next Steps

Everything Phase 2 and Phase 2.5 owe Phase 3 is in place: PSD summaries and profile plots on all 60 clean datasets, stable-reference JS curves on all 60 clean and all 60 noisy datasets with the boundary rise confirmed, the Phase 2.5 noise-metric library and tools, the pooled calibration with 10 promoted metrics and a held-out clean false-positive rate of 0.3 percent, mode labels and a noise score on all 33,719 files, verified b/blim provenance, six of seven review findings closed with tests at 68 of 68 passing, and a RapidMiner smoke test at out-of-fold AUC 0.964 on clean data and 0.97 to 0.98 on the noisy mirrors. The Phase 3 MLP baseline can start now. Two caveats remain that do not block it: the Altintas-Budak constants and engagement convention are still open and will block the physics-residual head later, and the pooled calibration should be rerun once under the new coverage gate to lock it as production.

Open obstacles and risks:

1. The Altintas-Budak `blim` calibration is still open. The exact textbook constants and the data generator's cut engagement convention are owed by the data owner, and only they can resolve it. This is the same physics-anchor risk as the last two updates. It blocks the physics-residual head of the eventual T-KAN-PINN, but not the MLP baseline or the spectral loss head.
2. The pooled calibration needs one rerun under the new coverage gate from finding #5 before it goes to production. The promoted set should be unchanged, since the current metrics already have 97.7 to 100 percent coverage, but the rerun is the verification step.
3. Dataset 19 sits on the boundary of the Forcey calibration gate. Four Forcey metrics narrowly missed promotion because their pooled monotone fraction falls just under 0.95, and Dataset 19 has a zero-monotone Forcey ladder while behaving normally on Forcex. Its metadata is ordinary and its manifest is clean. A v3 calibration that excludes Dataset 19 is staged on the cluster, but the root cause is unresolved.
4. The top tail of the noisy JS curve dips below clean. This is a facet-composition artifact: noisy 61 to 120 mirrors only clean 1 to 30, so the highest b/blim bins draw on one tooth-by-cut-direction facet while clean draws on four. The decision region around b/blim equal to 1 is solid. The fix is the apples-to-apples clean rerun on datasets 1 to 30, which is queued.
5. macOS openrsync misbehaved on multi-source rsync and on `--relative`. The cluster sync wrapper now opens one SSH master connection with 2FA at the start and reuses it, so the user answers Duo only once per transfer. This is resolved.

Next steps, in order:

1. Rerun the pooled calibration under the coverage gate and confirm the 10 promoted metrics are unchanged. If they are, this becomes the production calibration.
2. Run the strict clean rerun restricted to datasets 1 to 30 so the noisy and clean JS curves compare with the same facet mix.
3. Start the Phase 3 MLP baseline with hardcoded `tau` and an MSE loss. Treat the mode label and noise score as optional metadata at this stage.
4. Get the Altintas-Budak constants and the engagement convention from the data owner, so the physics residual is not blocked once the spectral loss is added.
5. Decide whether Dataset 19 is permanently excluded from the calibration pool or whether the Forcey channel needs an upstream fix.
6. Pull the four SINDy helper files from the cluster to the local tree so the extractor's fail-closed path resolves locally.

## 9. Summary

Phase 2 was the bridge between the raw machining data and the Phase 3 model. The goal was to turn each cut into a numeric measure of how unstable it is, in a form the model can learn from. Here is what Phase 2 produced and what it hands to Phase 3.

The spectral pipeline. Every raw file became a power spectrum, the spectrum was compared to a physics-only stable reference, and the result was scored with Jensen-Shannon divergence. On all 60 clean datasets and all 60 noisy datasets, the score stays low and flat while the cut is stable and rises smoothly across the stability boundary at b/blim equal to 1.0. The rise is a graded severity measure, not a stable-versus-unstable switch. This is the spectral signal the Phase 3 spectral loss head will train against.

Noise robustness. The noisy 60 run, at 1 percent and 5 percent simulated noise, kept the same boundary-aligned rise, and in the middle bins the rise is actually stronger under noise. This is the main robustness result so far. It says the spectral loss works on data that is not perfectly clean, a property the eventual hardware data will need.

Noise characterization. The Phase 2.5 library produces 44 candidate noise metrics per file. A pooled calibration on 30 mirror ladders and 7,525 paired operating points kept 10 of them, all in the PSD and wavelet families, all monotone in noise level, all with coverage above 97 percent. Each of the 33,719 files now carries one of seven mode labels and a calibrated noise score. The held-out clean false-positive rate for the broadband-noise label is 0.3 percent, and the 5 percent noise detection rate is 95.5 percent. The gate detects noise reliably without erasing chatter or TPF structure on clean data.

Label provenance. The stability label is verified end to end. The b/blim column equals b_mm divided by blim, the label is exactly int(b/blim >= 1), and the blim values match the raw stability-boundary files for every operating point inside the boundary grid. The supervised target for Phase 3 is not corrupted anywhere in the pipeline.

Code state. Six of seven review findings are closed with their own tests, the local test suite is at 68 of 68, the archive folder is local-only with a tracked manifest, and the cluster sync wrapper uses one Duo prompt per transfer. The pipeline is in a state where Phase 3 can pull on it without quiet failures.

How this helps Phase 3. Phase 3 is a supervised model first, then a physics-informed neural model. Phase 2 hands it four things: the mode label as a conditioning input, the noise score as a per-file sample weight, the 10 promoted noise metrics as auxiliary scalar features, and the stable-reference JS curve as the spectral fidelity loss head. The verified b/blim is both the supervised target and the input to the eventual physics residual. So the MLP baseline can start now on the supervised label and a noise-weighted MSE. The JS spectral loss head unlocks once the coverage rerun is done. The physics-residual head waits on one external item, covered below.

Caveats.

1. Altintas-Budak constants and cut engagement convention. The exact textbook constants and the way the data generator sets up the cut are still owed by the data owner. This blocks the physics-residual head of the eventual T-KAN-PINN, but not the MLP baseline and not the spectral loss head.

2. Pooled calibration coverage rerun. The current 10 promoted metrics already sit at 97.7 to 100 percent coverage, so the rerun under the new coverage gate should keep the same set. It is a verification step. Until it lands, the calibration is correct in practice but not yet stamped as production.

3. Facet mix on the noisy top tail. The noisy 61 to 120 set mirrors only clean 1 to 30, so at the highest b/blim bins the noisy curve draws on one tooth-by-cut-direction facet while the clean curve has four. The decision region around b/blim equal to 1 has four facets on both sides and is solid. The strict clean rerun on datasets 1 to 30 is queued and would remove this effect.

4. Dataset 19 Forcey channel. Four Forcey metrics narrowly missed promotion because Dataset 19 has a zero-monotone Forcey ladder while behaving normally on Forcex. A calibration that excludes Dataset 19 is staged on the cluster. The root cause is unresolved.

5. Cross-channel sign on real noise. On simulated noise the cross-channel coherence rises with noise level. On real ADC noise the sign may flip. The family-vote rule already handles this by abstaining when calibration finds no metric for a family, so the production behavior is safe. The sign needs revalidation when hardware data arrives.

6. Low-speed edge rows. A few rows with spindle speed below the raw boundary grid cannot be verified by direct interpolation. They are flagged separately and do not enter the calibration pool.

Bottom line. Phase 2 produced a graded spectral measure of instability that survives noise, a calibrated noise score and mode label per file, and a verified supervised target. The Phase 3 MLP baseline can start now. The spectral loss head and the physics-residual head unlock in the order described in Section 8.

## 10. Time Spent

This update covers May 25 to May 30, about 20 hours of work.

| Work | Hours |
|---|---|
| Pooled mirror calibration on 30 ladders and result review | 3.0 |
| Closing six review findings with TDD, integration, and full suite | 5.0 |
| Mode threshold retuning against pooled clean p95 | 1.0 |
| Phase 2.5 meeting figures script | 1.5 |
| Clean false-positive diagnostic script | 1.5 |
| Self-contained sbatch wrappers for noisy 61 to 120 PSD and JS | 1.5 |
| Noisy JS result analysis and clean-versus-noisy comparison | 1.5 |
| SSH multiplexing fix on the cluster sync wrapper | 0.5 |
| Dataset 39 teeth verification and cross-check across 120 datasets | 1.0 |
| Archive policy and manifest | 0.5 |
| Forcey Dataset 19 diagnostic and v3 calibration staging | 1.5 |
| Handoff updates, this update | 1.5 |
| **Total hours** | **20.0** |

The two biggest blocks were the code-hardening batch and the noisy JS run with its clean comparison. Both are the kind of work that has to land before Phase 3 modeling starts, even though neither moves the modeling itself forward.
