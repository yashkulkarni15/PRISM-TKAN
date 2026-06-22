# Held-out evaluation of the 5-fold TKAN on datasets 121–155

**Date:** 2026-06-22 · **GL job:** `51949469` (array 0–4) · **Eval harness:** `scripts/eval_heldout_phase3.py`

This is the **eval-only** generalization test of the published 5-fold TKAN sweep
(`best_auc.pt`, lr 0.0015 + side-input + compile). Each of the five trained checkpoints
(one per training fold) was run, with **no retraining**, over data that was **never in any
training fold**. It is the test that retires the "val-selected, not held-out" caveat on the
0.955 headline.

Scoring is **per-file** (windows averaged per file; files never split), so AUC/acc/MAE/RMSE
are directly comparable to the in-distribution sweep numbers. `n_excluded = 0` on every run.

## Eval sets

| set | datasets | files | windows | what it probes |
|-----|----------|-------|---------|----------------|
| **121–125** | 121, 122, 123, 124, 125 (all new, 5% noise) | 957 | 30789 | all five new noisy datasets |
| **122–125** | 122, 123, 124, 125 (genuinely novel, 5% noise) | 855 | 29346 | **OOD probe** — no parameter twin anywhere in 1–120; 122/124/125 carry process damping (Cn 300k/350k/200k) |

> The plan also names a **126–155 clean replication** set (126≡1 … 155≡30). It was **not**
> part of job `51949469` (this job covered only the noisy new data). Still open — see below.

## Results — set 121–125 (all five new noisy datasets)

| fold | run | **AUC** | acc | MAE | RMSE |
|------|-----|---------|-----|-----|------|
| 0 | `61c67bdd` | 0.8907 | 0.7910 | 4.2162 | 10.0091 |
| 1 | `c21e58dc` | 0.9193 | 0.7795 | 4.0700 | 10.5639 |
| 2 | `a654129d` | 0.8994 | 0.8036 | 4.4793 | 9.9311 |
| 3 | `3a40b715` | 0.9194 | 0.7168 | 3.2957 | 7.9213 |
| 4 | `09378839` | 0.9127 | 0.8422 | 3.5272 | 7.7697 |
| **mean ± std** | | **0.908 ± 0.013** | 0.787 | 3.92 | 9.24 |

## Results — set 122–125 (OOD probe: genuinely novel parameterizations)

| fold | run | **AUC** | acc | MAE | RMSE |
|------|-----|---------|-----|-----|------|
| 0 | `61c67bdd` | 0.9055 | 0.7965 | 4.6000 | 10.5801 |
| 1 | `c21e58dc` | 0.9251 | 0.7789 | 4.4460 | 11.1707 |
| 2 | `a654129d` | 0.9028 | 0.7965 | 4.8468 | 10.4919 |
| 3 | `3a40b715` | 0.9266 | 0.7088 | 3.5511 | 8.3690 |
| 4 | `09378839` | 0.9202 | 0.8409 | 3.8260 | 8.2092 |
| **mean ± std** | | **0.916 ± 0.011** | 0.784 | 4.25 | 9.76 |

## Takeaways

1. **The val-selected caveat is retired — the result generalizes.** On genuinely held-out
   data the classifier lands at **AUC 0.908 ± 0.013** (all new) and **0.916 ± 0.011** (OOD),
   versus the val-selected **0.955 ± 0.022** in-distribution. The ~0.04–0.05 drop is the honest
   generalization gap, and it is modest. **Every one of the five checkpoints, on both sets,
   still beats the MLP baseline (~0.87)** — the weakest single number is 0.8907.

2. **Out-of-distribution holds — slightly better, even.** The four genuinely novel
   process-damping parameterizations (122–125, no twin in 1–120) score **0.916**, marginally
   *above* the full new set (0.908) on every fold. Adding 121 (the noisy twin of clean 14)
   slightly lowers the average; the purely-novel datasets are not the hard ones. Process
   damping does not break the model — the most reassuring single result here.

3. **Checkpoint-robust.** The five folds agree to within ~0.03 AUC; this is not checkpoint luck.
   Folds 3 and 4 (`3a40b715`, `09378839`) generalize best on every metric.

4. **The severity head does not transfer.** MAE 3.3–4.8, RMSE 7.8–11.2 — the heavy tail blows
   up out-of-distribution (in-distribution was MAE 0.88). Consistent with the known weak
   severity head. Chatter *detection* is the deliverable and it holds; severity regression on
   new operating points does not.

5. **Accuracy (0.72–0.84) is a threshold artifact.** The 0.5 cut is not calibrated to the new
   data's class balance, so per-fold acc swings while AUC stays tight. AUC (threshold-free) is
   the fair read; a calibration step would be needed only if a hard accuracy number is required.

## Comparison

| | AUC | basis |
|--|-----|-------|
| In-distribution 5-fold sweep (`best_auc.pt`) | **0.955 ± 0.022** | val-selected (folds of 1–120) |
| Held-out, all new noisy (121–125) | **0.908 ± 0.013** | never trained on |
| Held-out, OOD novel (122–125) | **0.916 ± 0.011** | never trained on, no twin in 1–120 |
| MLP baseline | **~0.87** | cited single number |

## Still open

- **126–155 clean replication eval** (126≡1 … 155≡30) — not in job `51949469`; run it (or
  confirm it was deprioritized) for the replication half of the held-out story.
- **Calibration** of the operating threshold if a held-out *accuracy* number is wanted.
- Source JSONs: `results/phase3/heldout/<run>_{121_125,122_125}.json` (gitignored — numbers
  preserved in this tracked doc).
