---
title: "Research Update 3"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

## 1. State Before This Work

The last update closed the clean side of Phase 2. The spectra ran on all 60 clean datasets and the divergence score rose smoothly across the stability line. Three things were still open. The noisy side at full scale was not done. The Phase 2.5 noise layer was only a written plan. And there was no easy way for outside tools like RapidMiner to call the pipeline without using the full Python stack.

This update covers May 18 to May 24, about 19 hours of work. The job was to turn the Phase 2.5 plan into real code with passing tests, check the method on real data before writing new code, set up the cluster jobs that will calibrate it, and add a small RapidMiner test so the XGBoost side can be reached from outside Python.

## 2. Phase 2.5 Check on Real Data

Before writing any new code, the Phase 2.5 ladder test was run on the spectral features that already sit in the metadata. The goal was to make sure the method itself works on real data, not just on a toy example.

Two things came out. First, all 102 files in Dataset 1 pair cleanly with the 102 files in Dataset 61 and the 102 files in Dataset 91. So the clean, 1 percent, and 5 percent triple is fully paired. Second, five metrics that already exist pass the ladder with strict monotone behavior and very small p-values. The strongest are Forcex spectral entropy, xtpos spectral entropy, and Forcex harmonic energy ratio. So the method is sound. The new metric library is being added because the chatter and cross-channel families are not in the data yet, not because the test is unproven.

A useful negative check also showed up. The sideband and kurtosis features that already exist correctly fail the ladder. That is what they should do. Chatter features should not separate clean from noisy. If they had passed, it would have meant the chatter family was being polluted by noise, which is the very confusion Phase 2.5 is meant to prevent.

## 3. Phase 2.5 Library and Mode Labels

The Phase 2.5 library was added in two parts. One part computes noise metrics per file. The other part gives each file a mode label.

The metric library gives 44 columns per file in four families. The PSD family covers noise floor ratio, off-harmonic energy, spectral flatness, and spectral entropy. The wavelet and temporal family covers a wavelet sigma and an autocorrelation decay rate. The cross-channel family covers coherence across the four signals. The chatter family covers kurtosis in the natural-frequency band, sideband energy around the natural frequency, and a tooth-pass cycle metric.

One real bug was caught while testing. The first version of the chatter metrics divided by the energy in a narrow band that did not contain the natural frequency for a typical machining case. This made the ratio sit at 1.0 for almost every file and useless. The fix was to divide by total signal energy instead. The metric now sits cleanly between 0 and 1. The bug only showed up because the tests used realistic numbers for the natural frequency and the tooth-pass frequency, not a textbook example.

The mode labels are: TPF-dominated clean, near-boundary transition, chatter-dominated, broadband-noise-dominated, mixed TPF and chatter, low-speed edge, and metrics unavailable. The v1 rule fires in a set order. The low-speed edge label is checked first. Then the broadband-noise label, which needs one PSD, one wavelet, and one cross-channel metric to all agree. After that the other labels are picked by chatter-band energy, TPF energy, noise floor, and the stability ratio. Five thresholds inside the rule are marked as placeholders until the calibration job sets them from the real data.

The library has 27 unit tests. 15 for the metrics and 12 for the modes. All 27 pass.

## 4. Cluster Wiring

Three command-line tools were added so the new library can be run on Great Lakes.

The first computes the noise metrics per file. The second runs the paired ladder test on the clean, 1 percent, and 5 percent triple and writes the list of promoted metrics. This is the step that produces the numeric thresholds for the mode rule. The third reads the metrics output and writes a mode label per file.

Two cluster job scripts run these. The compute job uses 8 CPUs, 24 GB, and 4 hours. The calibration job uses 2 CPUs, 8 GB, and 30 minutes.

One wiring mistake was caught here. The first cluster script used a flag the tool did not accept and dropped a flag the tool needed. The scripts were rewritten to match the real tools, and the cluster README was updated.

An end-to-end test builds a small synthetic ladder, runs all three tools as subprocesses, and checks that the metrics rise across the ladder, that at least one PSD metric is promoted, and that every mode label is well-formed. It passes. This is the guard that says the library, the tools, and their flags all work together.

## 5. RapidMiner Smoke Test

A small RapidMiner test was added. The point is not to port the whole pipeline. The point is to show RapidMiner can call a PRI2M Python script and read its outputs.

The wrapper is a stripped-down version of the archived v3 XGBoost pipeline. It keeps the v3 feature rules and the training and evaluation. It drops SHAP, LIME, DiCE, and the plot code from the original. RapidMiner runs the wrapper and reads three CSV files.

The smoke test ran on clean datasets 1 to 5 (768 rows, 60 features). Out-of-fold AUC was 0.964 and F1 was 0.885. The same model was then run on the noisy mirrors 61 to 65 at 1 percent noise and 91 to 95 at 5 percent noise. Mean AUC stayed around 0.98 at 1 percent and 0.97 at 5 percent.

This is a smoke test, not a result claim. The point is the integration path. The numbers line up with what the SINDy work showed before: the classifier holds up at 1 percent noise and starts to drift on the F1 side at 5 percent.

## 6. Obstacles and Challenges

Four obstacles came up this week.

1. The chatter-band metric saturating to 1.0. The fix was to switch the denominator to total signal energy. The bug only showed up because the tests used realistic numbers.

2. The cluster job script using the wrong flags. The scripts were rewritten to match the real tools.

3. Five placeholder thresholds inside the v1 mode rule. They are only correct after the calibration job runs. Until then mode labels carry a confidence score and the broadband-noise label needs three-family agreement, so one bad placeholder cannot swing the label.

4. RapidMiner Python environment. RapidMiner runs Python but only against the interpreter it is pointed at. The README spells this out so the next person does not have to find it again.

## 7. Where This Stands Against the Schedule

Phase 2 on the clean side is still done. This week added the Phase 2.5 library, the three command-line tools, the two cluster job scripts, the end-to-end test, and the RapidMiner smoke test. Phase 3 modeling has not started yet, which is fine. The MLP baseline can start in parallel with Phase 2.5, and the JS spectral loss only unlocks after Phase 2.5 produces calibrated thresholds.

## 8. Risks and Next Steps

Two open risks remain.

1. The five placeholder thresholds in the v1 mode rule are not yet calibrated. The mirror-ladder calibration job on the cluster is the next planned run.

2. The Altintas-Budak constants and the data generator engagement convention are still open. This is the same risk as last week. It does not block Phase 2.5 or the Phase 3 baseline. It only gates the eventual physics residual.

Next steps in order:

1. Run the full noise-metric extraction on datasets 1 to 120.
2. Run the mirror-ladder calibration on every clean dataset that has both mirrors, replace the placeholders, and freeze the v1 mode rule.
3. Start the Phase 3 MLP baseline in parallel with step 2.
4. Once mode labels are stable, regenerate the stable-reference JS curves restricted to clean and near-boundary files, to confirm the rise across the stability boundary survives when broadband-noise files are removed.

## 9. Time Spent

This update covers May 18 to May 24, about 19 hours of work.

| Work | Hours |
|---|---|
| Phase 2.5 ladder check on real metadata | 1.5 |
| Noise metrics library: 44 metrics, four families | 4.0 |
| Mode label module: v1 rule, seven labels | 2.0 |
| Unit tests: 27 tests for metrics and modes | 2.0 |
| Command-line tools: compute, calibrate, assign modes | 2.0 |
| Cluster job scripts for Phase 2.5 and README update | 1.0 |
| Phase 2.5 end-to-end test | 1.5 |
| Bug fix: chatter-band normalization | 0.5 |
| RapidMiner smoke test: wrapper and output schema | 2.5 |
| RapidMiner clean and noisy smoke runs and review | 1.0 |
| Documentation: handoff entries, this update | 1.0 |
| **Total hours** | **19.0** |

The two biggest blocks were the metric library and the RapidMiner smoke test. Both are the kind of work where the design choices matter more than the line count.
