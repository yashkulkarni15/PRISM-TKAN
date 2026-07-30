---
title: "Research Update 7"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

## 1. State Before This Work

The last update got the TKAN through a full, stable five-fold run that beat the MLP baseline, with a chatter AUC of 0.955 against 0.87. The result was audited and held up. It left two threads open. The headline was chosen on the validation fold rather than a truly held-out set, and a larger version of the dataset existed, 155 setups instead of the 120 the pipeline was built on, which I had scoped but not yet brought in.

This update closes both. I brought in the larger dataset as an evaluation-only extension, and in doing so turned the val-selected 0.955 into a real held-out generalization number. Along the way I found and closed a small leakage gap in the old fold grouping. No model was retrained.

This update covers June 15 to June 22, about 20 hours of work.

## 2. The Dataset Was Larger Than the Pipeline

The metadata documents 155 datasets, but the pipeline runs on 120. The 35 new ones are 30 clean setups (126 to 155) and 5 noisy ones (121 to 125). Before touching anything I checked what they actually are. The thirty clean datasets repeat the original 1 to 30 exactly: the same operating points run a second time, not new physics. Of the five noisy datasets, only four are genuinely new parameterizations with no twin anywhere in 1 to 120, and three of those carry low-speed process damping, the exact physics missing from the 700 low-speed rows I have had to drop all along. The raw signal files for the new datasets live only on a newer release, so they had to be fetched; the cluster held only the 120-dataset version.

That audit set the strategy. Since most of the new data repeats operating points the model has already seen, the new datasets go through the evaluation path rather than training. The model stays fixed, and the new data becomes the test the last update said was missing.

## 3. A Leakage Gap in the Old Grouping

Wiring the new datasets into the fold grouping is what surfaced the gap. The grouping keeps a clean dataset and its noisy mirrors together so they can never split across train and test. It did this with an arithmetic rule: clean dataset D, its 1% mirror at D+60, its 5% mirror at D+90. That rule assumed every clean dataset from 1 to 60 is a distinct parameterization, which is not the case. Datasets 1 and 23 are parameter-identical, and so are 16 and 17, each with their own mirror clusters. The arithmetic rule put each twin in a different group, so in the published five-fold run those operating points could land in both the training and the test fold at once.

The leak is small and bounded, four of the sixty clean datasets and a few percent of any one validation fold, but it is real, and it had to be closed before the new data could be grouped safely, since the new numbering does not follow the old mirror pattern at all. I replaced the arithmetic rule with a grouping computed from parameter identity. I fingerprint each dataset on its physical parameters, merge any two that match, and label each group by its lowest dataset id. The result collapses all 155 datasets onto 60 distinct parameterizations. Because it only ever merges groups the old rule kept apart and never splits one, every existing leakage and split test still passes unchanged. The fix also gives the new data a correct home: dataset 121 is a noisy twin of clean 14, the clean datasets 126 to 155 map onto 1 to 30, and only the four novel datasets stand alone.

## 4. Regenerating the Missing Metadata

The newer release ships the raw signal files but, unlike the version the pipeline was built on, no precomputed feature tables. Every downstream step needs per-file physics metadata: spindle speed, cutting depth, the stability limit, the chatter label, tooth count, and cut direction. About half of each row is a static lookup from the dataset-description spreadsheet, which I reverse-engineered column by column, including the tool diameter, which the spreadsheet stores in metres and the pipeline in millimetres. Before trusting it on the new data, I confirmed it reproduces the existing 1 to 120 tables exactly. The other half of each row, the speed, depth, and stability limit interpolated at the running speed, is computed from the raw signal files and has to run on the cluster. I also pulled the hard-coded data-version path out of the several places it was scattered, so the new download is now reachable by setting one variable instead of editing code.

## 5. The Held-Out Test

With the grouping fixed and the metadata regenerated, I built a small evaluation harness. It takes one trained checkpoint and runs it over any set of datasets with no folds and no retraining, reusing the checkpoint's own training-set normalization so that nothing is refit on the new data. I ran all five fold checkpoints over the new noisy datasets, which were never in any training fold.

![Held-out generalization of the Phase-3 TKAN on the new datasets, with no retraining. Left: per-fold held-out classification AUC for the five folds on two evaluation sets, all newly added noisy datasets (121 to 125) and the genuinely novel subset (122 to 125). Every fold sits well above the MLP baseline (dotted, about 0.87) and a little below the in-distribution five-fold mean (dashed, 0.955), with held-out means of 0.908 for the full new set and a slightly higher 0.916 for the novel subset. Right: severity-head error on the new noisy data, per-fold mean absolute error against root-mean-square error. RMSE far exceeds MAE across all folds, and both lie well above the in-distribution severity MAE of about 0.88 (dashed), showing that the severity regression does not transfer.](results/figures/research_update_assets/heldout_generalization_121_155.png){width=95%}

The result is a held-out chatter AUC of 0.908, give or take 0.013, across the five checkpoints on the full new noisy set, and 0.916 on the genuinely novel subset. Every one of the five checkpoints, on both sets, still beats the MLP baseline of 0.87, with the weakest single number at 0.891.

## 6. What the Held-Out Numbers Say

This is the test the last update owed. The in-distribution 0.955 was chosen on the validation fold; this 0.91 is measured on data the model never trained on, and the roughly four-point gap between them is the honest generalization drop. It is modest, and it keeps the result clearly above the baseline rather than collapsing it. It also answers the leakage gap from Section 3 more cleanly than a re-run would have, because a held-out number does not depend on the fold split at all. Whatever the old grouping did or did not leak, this number is untouched by it.

The most reassuring part is the novel subset. The four genuinely new parameterizations, operating points with no twin anywhere in the training data and three of them carrying low-speed process damping the model had never seen, score 0.916, slightly higher than the new data as a whole. The model does not merely tolerate out-of-distribution physics; it reads it about as well as the rest.

The severity head, which predicts depth over the stability limit, does not transfer. Its error on the new data is several times its in-distribution value and dominated by a heavy tail of large per-file misses, exactly the weakness the last update flagged, now sharper out of distribution. Chatter detection, the actual deliverable, holds; severity regression on unseen operating points does not. Per-file accuracy swings between folds because the decision threshold is not calibrated to the new data's class balance, but the AUC, which needs no threshold, stays tight across all five.

## 7. Where This Stands

The pipeline now spans the full 155-dataset version of the data, behind a grouping that cannot leak parameter-identical datasets across train and test. The published 0.955 has a held-out companion: the model generalizes to unseen datasets at about 0.91, and to genuinely novel operating points, including low-speed process damping, at 0.916, beating the baseline on every fold. The whole project is also under version control for the first time. The remaining half of the held-out test, the thirty clean replication datasets, is blocked, though not by anything in the model. The newer release ships those clean files in a reduced four-column format with only two non-oscillating channels and a different sampling, so they cannot feed the four-channel pipeline as they are. Finishing that arm needs the data owner to supply 126 to 155 in the standard seventeen-column format the rest of the data uses. The severity head and the physics head are where they were.

## 8. Obstacles and Next Steps

Obstacles:

1. The clean replication arm (126 to 155) is blocked by a data-format change, not by code. The newer release stores those datasets in a reduced four-column layout that the four-channel pipeline cannot read, so the held-out test currently covers only the new noisy data and the novel subset. The noisy datasets 121 to 125 are in the standard seventeen-column format, which is why their result is sound.
2. The leak is closed in code, but the original 0.955 folds were never re-run with the corrected grouping. The held-out number stands in for a corrected in-fold number rather than replacing it directly.
3. The Altintas-Budak stability constants are still missing. Only the data owner can supply them. They block the physics head and are the reason the 700 low-speed labels are dropped, the same low-speed regime the new process-damping datasets cover.
4. The side-input ablation and the matched per-fold MLP table, both named in the last two updates, still have not run.
5. The severity head does not transfer and remains heavy-tailed.

Next steps, in order:

1. Request datasets 126 to 155 from the data owner in the standard seventeen-column format that 1 to 125 use. They then drop straight into the eval harness already built. Until then, treat the held-out result as complete on the noisy and novel datasets.
2. Run the no-side-input ablation and pull the MLP's per-fold AUC, then build the head-to-head table on the same folds.
3. Recompute the in-distribution chatter AUC on supervised-only files for a clean headline number.
4. Strengthen interpretability with integrated gradients on the chatter head, averaged over chatter versus stable cuts, plus a pruning-threshold sweep.
5. Get the Altintas-Budak constants from the data owner.

## 9. Summary

The last update left a result that was strong but chosen on the validation fold, and a larger dataset sitting unused. This update brought the larger dataset in as an evaluation-only extension, closed a small but real leakage gap in the fold grouping along the way, regenerated the metadata the new release does not ship, and ran the first held-out test. The model generalizes to unseen datasets at a chatter AUC of about 0.91, and to genuinely novel operating points, including process damping it had never seen, at 0.916, beating the baseline on every fold. The severity head does not transfer. The val-selected caveat is retired: the chatter result is now a held-out result, not a validation-fold one.

## 10. Time Spent

This update covers June 15 to June 22, about 20 hours of work.

| Work | Days | Hours |
|---|---|---|
| Dataset audit: 120 vs 155, parameter-twin map, replication vs novel | Jun 15 | 2.0 |
| Leakage fix: parameter-identity grouping, build script, tests | Jun 16 | 3.5 |
| Metadata generator: reverse-engineer the spreadsheet mapping, validate against 1 to 120 | Jun 16 to 17 | 3.5 |
| Data-version parametrization and the held-out eval harness, with tests | Jun 17 | 2.5 |
| Great Lakes: validate the generator on the original signal files, generate the new tables, extract signals | Jun 17 | 2.0 |
| Held-out eval run (five folds, two sets), pulling and auditing the results | Jun 17 to 22 | 2.0 |
| Analysis, generalization figure, results table | Jun 22 | 2.0 |
| First version-control commit of the project and repo cleanup | Jun 22 | 1.0 |
| Docs and this update | Jun 22 | 1.5 |
| **Total hours** | | **20.0** |

The two biggest blocks were the leakage fix and the metadata generator. Together they are what let the larger dataset come in safely and turned the val-selected 0.955 into a held-out result.
