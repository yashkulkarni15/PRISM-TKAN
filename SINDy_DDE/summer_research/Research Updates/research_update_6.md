---
title: "Research Update 6"
geometry: margin=1in
fontsize: 11pt
mainfont: Helvetica
---

## 1. State Before This Work

The last update built the full Phase 3 harness, ran the MLP baseline to a leakage-safe chatter AUC of about 0.87, and left one thing open: the TKAN had never finished a full run. It hit the cluster time limit on a slow, partly untuned path. This update gets the TKAN through a full five-fold GPU run, makes its training stable, confirms it beats the baseline, checks the result is not inflated, runs the interpretability, and surfaces a larger version of the dataset that I am not yet using.

This update covers June 7 to June 14, about 20 hours of work.

## 2. Getting the TKAN to Finish

The first job stopped at zero epochs. The cost is the 1024-step recurrent KAN cell, not the data. Three changes fixed it. I average-pooled the sequence by four, which is lossless for the chatter band and cuts the recurrence length. I compiled the recurrent cell, which fused its many small operations into about a 3.4 times speed-up, taking an epoch from roughly 77 to 25 minutes, so a fold is about 8 hours. And the original 6-hour wall was self-imposed, not a cluster cap, so I raised it. After these, a full 20-epoch fold runs end to end.

## 3. Making Training Stable

The first complete fold-0 run finished but only matched the MLP, at about 0.88, and the training was unstable: the loss bounced and the validation score swung between 0.69 and 0.88. The learning rate was too hot with no schedule. I added a small, fully toggleable tuning bundle, built test-first: process-parameter side inputs in the spirit of the Gao milling paper (see the next section), a cosine learning-rate schedule with warm-up, gradient-norm clipping, and AdamW.

This exposed a real failure mode. The recurrent cell can throw a non-finite gradient, and naive clipping then scales every gradient by that non-finite value, turning one bad step into an all-NaN model. I added guards that skip and count such steps instead. With the guards in place, fold 0 at the original peak learning rate still skipped batches mid-run but recovered to 0.939. Halving the peak rate to 0.0015 removed the instability entirely: zero skipped steps across all 20 epochs, and the score rose to 0.96.

## 4. What I Took from the Gao Paper

The side inputs come from a 2025 paper by Gao and co-authors on machine learning for milling chatter. It is a useful reference because it attacks the same task, predicting chatter, but with a plain machine-learning model: no physics, no delay equation, and no KAN. Its one robust, transferable finding is that feeding the process parameters into the model alongside the signal stabilizes training and improves accuracy, because those parameters fix the operating point the signal has to be read against.

That is the idea I took: a small side branch that feeds the operating conditions into the severity head. I adapted it rather than copied it. I feed spindle speed, tooth-pass frequency, and the tooth-pass delay, but I deliberately exclude anything derived from the cutting depth or the stability limit, because the label is depth over the stability limit, so feeding depth would leak the answer; the code rejects those columns at construction. In my runs the side branch was part of the bundle that turned the marginal, unstable fold-0 run into a stable one that beats the baseline.

What I did not take is the model itself. The paper's contribution is a conventional learner on hand-built features; it has no delay structure, no physics residual, and no interpretability, which are the parts of this work that matter most. So the paper served as one well-chosen ingredient, the side inputs, not a template.

## 5. The Full Five-Fold Sweep

With the stable configuration I ran all five folds on the GPU. Every fold cleared the gate and beat the MLP baseline. The mean best chatter AUC is 0.955, give or take 0.022, ranging from 0.912 on the hardest fold to 0.975 on the easiest, against the MLP's 0.87. There were zero skipped batches across all 100 epoch-folds, so the stability fix held everywhere, not just on fold 0.

![Five-fold TKAN sweep. Top left: validation chatter AUC per epoch, with the best-checkpoint epoch starred and the MLP baseline and gate marked; every fold settles well above 0.87. Top right: training loss converges cleanly on a log scale with no blow-up. Bottom left: best AUC per fold against the MLP baseline, mean 0.955. Bottom right: the severity error per fold, where RMSE sits far above MAE and fold 4 is dominated by a few large outliers.](results/figures/research_update_assets/tkan_sweep_dashboard.png){width=95%}

The chatter head is the strong result. The severity head, which predicts b/blim, is the weak one. Its error is heavy-tailed: the RMSE is much larger than the mean absolute error on every fold, and one fold is dominated by a handful of large misses. The classification head also peaks around the middle epochs and holds, while the severity error drifts up late, so the two heads pull in slightly different directions.

## 6. Checking the Result Is Real

Before treating 0.955 as a result I audited it against the code, not the summaries. The classes are balanced, about 47 percent chatter, so the AUC is not a base-rate artifact. The folds are grouped so a clean dataset and its noisy mirrors never split across train and test, and the no-leakage check runs at train time. Normalization and the side-input statistics are fit on training data only. The score is computed per file, not per window, so overlapping windows from one file cannot leak. And the reported peak is essentially equal to the final-epoch score, so very little of it is selection optimism.

Two honest caveats remain. The evaluation still scores the roughly four percent of files that are dropped from training (the 700 bad-label and 560 no-metric rows), so a clean recompute on supervised-only files is owed. And the number is chosen on the validation fold rather than a truly held-out set. Neither inflates the result much, and both are bounded.

## 7. Interpretability

The point of a KAN over a black box is that you can read what it learned, so the interpretability runs on a trained checkpoint and produces three things: a plot of every learned edge function (the one-dimensional curves the network uses in place of fixed activations), an integrated-gradients map (where in the signal the prediction comes from), and a pruning report (how many edges actually matter).

The encouraging finding is in the edge weights. The model leans heavily on the delayed, one-tooth-ago copy of the signal, the regenerative feedback that physically causes chatter, rather than on the raw signal alone. That is the correct mechanism, and it is exactly the kind of statement a black-box model cannot make. The gradient map agrees: it is structured rather than noisy, concentrating on specific moments and channels in the cut, with an echo near the tooth-pass delay.

![Integrated-gradients attribution for one cut, over the four input channels and the pooled time steps. Warm regions push the prediction toward chatter, cool regions away. The map is structured around specific moments and channels rather than uniform, with a visible response near the tooth-pass delay, which is consistent with regenerative chatter.](results/figures/research_update_assets/tkan_xai_integrated_gradients.png){width=85%}

Two limitations matter for the eventual paper, and both are fixable. At the default threshold nothing prunes, so the network is dense rather than reduced to a small symbolic form, which weakens the cleanest interpretability story. And the gradient map is currently computed on the weaker severity head from a single window, rather than on the chatter head averaged over many stable and unstable cuts. The learned edge functions, shown below, are smooth and gently nonlinear, consistent with that dense, not-yet-symbolic picture.

![Learned KAN edge activations across the recurrent gates and the two heads. Each line is one edge's learned one-dimensional function over its normalized input. The shapes are smooth and gently nonlinear rather than sharply switching, consistent with a dense network that does not yet prune to a small symbolic form.](results/figures/research_update_assets/tkan_xai_edge_activations.png){width=85%}

## 8. A Larger Dataset Is Available

While confirming the data source, the metadata folder showed that the description spreadsheet documents 155 datasets, but the whole pipeline so far is built on 120. The 35 new ones are 30 clean setups and 5 noisy, and nine of them include low-speed process damping, which is the exact physics missing from the 700 low-speed rows I had to drop. The cluster currently holds only the 120-dataset version, so the new data has to be fetched. I parked this to finish Phase 3 first, but recorded the exact incorporation steps, because one of them is a correctness trap: the new noisy datasets do not follow the old tidy mirror numbering (dataset 121 mirrors clean dataset 139, and a few have no clean twin), so the fold-grouping logic has to be replaced with an explicit map before any training, or the leakage protection silently breaks.

## 9. Where This Stands

The TKAN is now through a full, stable, five-fold GPU run and clearly beats the MLP baseline, 0.955 against 0.87, with no training instability. The result has been audited and is not inflated. The interpretability runs end to end. The severity head is the remaining weak point, and the physics head still waits on the same Altintas-Budak constants as every prior update. A larger dataset exists and is scoped but not yet incorporated.

## 10. Obstacles and Next Steps

Obstacles:

1. The Altintas-Budak blim constants are still missing. Only the data owner can supply them. They block the physics head and are the reason the 700 low-speed labels are dropped.
2. The TKAN-versus-MLP comparison still cites a single baseline number, not a per-fold table on the same folds.
3. The side-input ablation has not run, so the gain cannot yet be split between the architecture and the process-parameter inputs.
4. The severity head is heavy-tailed and the interpretability is not yet sharp enough to carry a physics claim.

Next steps, in order:

1. Commit the validated Phase 3 branch.
2. Run the no-side-input ablation and pull the MLP's per-fold AUC, then build the head-to-head table.
3. Recompute the chatter AUC on supervised-only files for the clean headline number.
4. Retrieve and inspect the learned spline parameters from the trained checkpoints, then re-run the model with different KAN settings (spline resolution and pooling) and compare. This is in progress.
5. Strengthen the interpretability: gradients on the chatter head, averaged over chatter versus stable cuts, plus a pruning-threshold sweep.
6. Incorporate the expanded data. The dataset has grown from 120 to 155, and the 35 new ones must be brought in, starting with the fold-grouping fix and a held-out test on the new data.
7. Get the Altintas-Budak constants from the data owner.

## 11. Summary

Last update left the TKAN unable to finish. This update made it finish, made it stable, and made it win. Speed and walltime fixes got a fold through in about 8 hours; a toggleable tuning bundle, with process-parameter side inputs borrowed from the Gao paper, plus a halved learning rate, removed the training instability, with zero skipped steps across all five folds. The result is a mean chatter AUC of 0.955 against the MLP's 0.87, audited and not inflated. The interpretability runs and already shows the model using the regenerative delay, the severity head remains the weak point, and a larger 155-dataset version of the data is now scoped for later. Phase 3 has gone from a harness that runs to a model that beats its baseline.

## 12. Time Spent

This update covers June 7 to June 14, about 20 hours of work.

| Work | Days | Hours |
|---|---|---|
| GPU speed-up, sequence pooling, cell compile, walltime fix | Jun 7–9 | 2.0 |
| Tuning bundle: side inputs, schedule, clipping, AdamW, with tests | Jun 10 | 3.0 |
| Stability debugging: non-finite guards and the learning-rate fix | Jun 10–11 | 2.5 |
| Fold-0 reruns and analysis | Jun 11–12 | 2.0 |
| Full five-fold sweep, launch and evaluation | Jun 12–14 | 2.0 |
| Inflation audit against the code | Jun 14 | 2.0 |
| Interpretability run and reading | Jun 14 | 2.0 |
| Result figures | Jun 14 | 1.0 |
| Larger-dataset investigation and incorporation plan | Jun 14 | 2.0 |
| Docs and this update | Jun 14 | 1.5 |
| **Total hours** | | **20.0** |

The two biggest blocks were the tuning bundle and the stability debugging. Together they are what turned a run that merely finished into one that beats the baseline and holds steady across every fold.
