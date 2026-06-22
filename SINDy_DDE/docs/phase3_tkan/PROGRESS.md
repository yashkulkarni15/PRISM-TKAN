# Phase 3 T-KAN — Live Progress & Resume Doc

**This file is the single source of truth for resuming Phase 3 TKAN work.**
If you are a fresh session: read this top-to-bottom, run the **Resume protocol**
below, then continue from **NEXT ACTION**. Keep this file updated after every
meaningful step (mark checkboxes, update NEXT ACTION, append to the log).

- **Spec:** `docs/superpowers/specs/2026-06-04-phase3-tkan-design.md`
- **Plan (task briefs):** `docs/superpowers/plans/2026-06-04-phase3-tkan.md`
- **Branch:** `phase3-tkan`. **Commit policy: never `git add`/`git commit`/`git push` without explicit user instruction** — leave work in the working tree; the user commits manually. (Tree also has unrelated `Dataset 1/...` deletions.)
- **Run dir for work:** `/Users/yash/Downloads/PRI2M/SINDy_DDE`

## Approved decisions (do not relitigate)
1. Two heads, staged: severity now, trajectory built+gated, physics = stub (Altintas–Budak blocked).
2. Trajectory = autoregressive rollout of xtpos/Forcex, hardcoded `tau=round(fs/tpf)`.
3. Self-built `KANLinear` (no pykan/efficient-kan runtime dep).
4. Severity multi-task: weighted MSE(b/blim) + weighted BCE(label).
5. sigKAN = optional off-by-default trunk variant.
6. PyTorch + mlflow/captum/shap/dice-ml/matplotlib (+iisignature optional).

## Environment facts
- Python 3.13.5 (`/opt/anaconda3/bin/python`); **torch 2.9.1 present**, matplotlib/sklearn/numpy/pandas/scipy/pytest present.
- **Missing** (install per-phase): `mlflow` (Phase 0/3), `captum`/`shap`/`dice_ml` (Phase 4), `iisignature` (optional, sigKAN only).
- Local data present: `results/phase3/{phase3_dataset.csv,phase3_folds.json,run_metadata.json}` (real schema reference; tests use synthetic trees).
- Tests are flat: `tests/test_*.py`; shared synth fixtures will live in `tests/_phase3_synth.py`.
- Reuse: `pri2m.phase3.splits.assert_no_mirror_leakage`, `pri2m.models.stage_gates.evaluate_trajectory_gate`, `pri2m.training.mlflow_retention`.
- GL project root is `/nfs/turbo/coe-jaabell/PRI2M` (NOT the stale `/SINDy_DDE` subdir). Build local + tests; user runs GL.

## Execution model
Phase-gated. Parallel subagents **within** a wave (disjoint files). A wave must be
fully green before the next starts. Each subagent uses TDD.

## Dependency graph
```
Phase 0 (driver) ──► Phase 1: 1A data | 1B kan | 1C losses/metrics   (parallel)
                         └──► Phase 2: 2A tkan | 2B heads+baseline    (parallel, need 1B)
                                 └──► Phase 3: trainer+CLI            (single, needs 1A,1C,2A,2B)
                                         └──► Phase 4: 4A xai | 4B sbatch+E2E (parallel, need P3)
```

## Status board
- [x] Brainstorm + design approved (6 decisions)
- [x] Spec written + plan written
- [x] **Phase 0** — scaffolding: pyproject deps, `xai/__init__.py`, `tests/_phase3_synth.py` (+ `test_phase3_synth.py`, 3 green), `mlflow 3.13.0` installed
- [x] **Phase 1** (parallel) — 1A `data/phase3_sequence.py` (24) · 1B `models/kan.py` (19) · 1C `training/{losses,metrics}.py` (31). All reviewed; contracts met; 1C fixed a NaN-gradient leak.
- [x] **Wave-1 gate** — 77 passed (4 module suites + fixture); full suite **205 passed**, 0 regressions.
- [x] **Phase 2** (parallel) — 2A `models/tkan.py` (14+2skip) · 2B `models/{heads,baseline}.py` (23) · driver-assembled `models/tkan_model.py` (+`test_tkan_model.py`, incl. data→model integration)
- [x] **Wave-2 gate** — 43 passed, 2 skipped; full suite **248 passed**, 0 failures
- [x] **Phase 3** — `training/phase3_trainer.py` + `scripts/train_phase3.py` + `configs/phase3_tkan.yaml` (15 tests). MLflow `file:` store needs `MLFLOW_ALLOW_FILE_STORE=true` (set at module top before mlflow import).
- [x] **Wave-3 gate** — 15 passed; full suite **263 passed**, 2 skipped, 0 failures
- [x] **Phase 4** (parallel) — 4A `xai/{kan_native,attribution,counterfactuals}.py` + `scripts/xai_phase3.py` (20) · 4B `slurm/phase3_train.sbatch` + `slurm/README.md` + keystone `tests/test_phase3_tkan_end_to_end.py` (3). Driver fixed sbatch `PROJECT_ROOT` → live GL root.
- [x] **Final gate** — full suite **287 passed, 1 skipped** (the skip is intentional: iisignature-absent branch). Real CLI smoke: train tkan → checkpoint → XAI artifacts, all produced.
- [x] **Tuning bundle (2026-06-10)** — Gao side-input + cosine/warmup + grad-clip + AdamW, all flag-gated, TDD (2 parallel waves, 4 subagents); XAI made side-checkpoint-compatible. Full suite **370 passed, 1 skipped**, 0 regressions.
- [x] **Fold-0 gate PASSED (2026-06-11, GL job 51651316)** — 20 epochs / 8h24m; `best_auc.pt` = epoch 6, val AUC **0.9388** (gate >=0.88; MLP ~0.87 clearly beaten). Mid-run gradient blow-up epochs 7–11 (1714 batches skipped, worst 1661/3334 at epoch 9) — guards held, run recovered to AUC 0.927 by epoch 19. Applied pre-agreed fallback: yaml peak lr 0.003 → 0.0015 (uncommitted).
- [x] **Full 5-fold sweep PASSED (2026-06-12, GL job 51670644 fold 0 + array 51711709 folds 1–4)** — lr 0.0015, side-input on, compile on. **0 skipped batches across all 100 epoch-folds** (stability fix validated). 5-fold `best_auc.pt` = **0.955 ± 0.022** (0.960/0.912/0.975/0.961/0.970); every fold beats MLP ~0.87 (min margin +0.042) and the 0.88 gate. Severity head heavy-tailed (MAE 0.88±0.65, RMSE 4.37±2.65; fold 4 outliers 2.13/9.0).
- [x] **Held-out eval on 121–125 PASSED (2026-06-17, GL job 51949469 array 0–4)** — all 5 `best_auc.pt` checkpoints run eval-only (no retrain) on never-trained data. **Held-out AUC 0.908 ± 0.013** (all new noisy 121–125) and **0.916 ± 0.011** (OOD novel 122–125); every fold beats MLP ~0.87, 0 excluded. **Retires the "val-selected" caveat** (gap 0.955→0.91 is the honest generalization drop). Severity head does NOT transfer (MAE 3.3–4.8 / RMSE 7.8–11.2). Full table: `docs/phase3_tkan/heldout_eval_121_155.md`. **126–155 clean replication eval still OPEN.**

---

## NEXT ACTION
**FULL 5-FOLD SWEEP PASSED (2026-06-12).** lr 0.0015 + side-input + compile, 20 epochs/fold.
Jobs: fold 0 = `51670644` (run `61c67bdd…`), folds 1–4 = array `51711709`
(runs `c21e58dc`/`a654129d`/`3a40b715`/`09378839`). ~8–8.5 h/fold; folds 1–4 arrayed in parallel.

**Headline — classification (chatter detection) clears the gate on every fold:**

| fold | best_auc.pt (epoch) | final AUC | acc | severity MAE | severity RMSE | skipped batches |
|------|---------------------|-----------|-----|--------------|---------------|-----------------|
| 0 | **0.9597** (e12) | 0.9565 | 0.906 | 0.580 | 4.835 | 0 |
| 1 | **0.9122** (e11) | 0.9092 | 0.851 | 0.855 | 4.353 | 0 |
| 2 | **0.9752** (e13) | 0.9744 | 0.923 | 0.521 | 2.321 | 0 |
| 3 | **0.9605** (e12) | 0.9551 | 0.888 | 0.323 | 1.324 | 0 |
| 4 | **0.9695** (e19) | 0.9695 | 0.921 | 2.128 | 9.005 | 0 |
| **mean** | **0.955 ± 0.022** | 0.953 ± 0.023 | 0.898 | 0.881 ± 0.646 | 4.368 ± 2.653 | **0** |

- **Stability FIXED.** Zero non-finite/skipped batches across all 100 epoch-folds — the
  pre-agreed `lr 0.003 → 0.0015` fallback fully eliminated the KAN-GRU gradient blow-up
  (the 0.003 fold-0 run skipped 1714 batches; worst epoch lost 1661/3334). The lower peak
  also *raised* fold 0: 0.9597 @0.0015 vs 0.9388 @0.003. train_loss converges cleanly
  73–131 → 0.06–0.10 on every fold.
- **Classification PASSES decisively.** 5-fold best_auc 0.955 ± 0.022, every fold > MLP ~0.87
  (weakest = fold 1 at 0.912, still +0.042) and > the 0.88 gate. AUC peaks mid-run (e11–e13)
  for folds 0–3, e19 for fold 4; `best_auc.pt` captures whichever.
- **Severity head is the weak head (known).** RMSE ≫ MAE on every fold; fold 4 is
  outlier-dominated (MAE 2.13 / RMSE 9.0 — a few files dominate). Absolute scale isn't
  comparable across folds (different val groups), but the heavy tail is consistent — the
  open quality lever. See [[gao_milling_ml_paper]] / [[phase3_tkan_perf]].
- err files = the benign TF32 perf hint only. `leakage_verified: true`, 700 + 560 excluded
  on every fold (both pre-flight blockers stay cleared). Total val files = 33719 (= dataset).

**Inflation audit (2026-06-14, against source not docs) — NOT materially inflated:**
classes are balanced (17742 neg / 15977 pos ≈ 47% positive, so AUC isn't a base-rate
artifact); folds are group-disjoint on `parent_group` with `assert_no_mirror_leakage`
called at runtime (`leakage_verified: true`) → no clean↔noisy mirror leakage; norm + side
stats are fit TRAIN-only; scoring is **per-file** (windows averaged per file, files never
split across train/val) so no within-file window leakage; final-epoch AUC 0.953 ≈ peak 0.955
→ `best_auc.pt` (argmax-over-epochs val AUC) carries only ~0.002 selection optimism.
**Two honest caveats:** (1) `build_fold_loaders` builds val with `supervised_only=False`, so
the 1260 weight-0 excluded files (700 untrustworthy `label=1` negative-blim + 560
missing-metric) are still **scored in eval** (≈2–4% of val files; bounded, direction unknown)
— recompute supervised-only for the clean headline; (2) the cited number is val-selected, not
a held-out test (the dev groups `[1,4,29,35,39]` exist but aren't used for selection here).

**Figures (local):** `results/phase3/sweep_figs/{sweep_dashboard.png, two_head_dynamics.png,
sweep_summary.csv}` via `python scripts/plot_phase3_sweep.py` (reads `GLC Logs/*.log`).

**USER next — pick up here (GL unless noted):**
1. **Commit the validated bundle.** lr 0.0015 + the whole tuning bundle (side input, cosine +
   warmup, grad-clip, AdamW) is now proven over 5 folds. Say the word and I'll commit the
   uncommitted yaml/code (your name only, no Claude co-author).
2. **`--no-side-input` ablation (5 folds).** All five sweep folds ran `side_input: true`, so we
   can't yet *isolate* the Gao side-input contribution — only that TKAN+side beats MLP. Launch
   the same array with `--no-side-input` and compare best_auc per fold (the sbatch threads CLI
   flags via the `MODEL`/env path; confirm the flag plumbing before submit).
3. **Matched MLP baseline table.** "~0.87" is a single cited number; for the paper pull the
   per-fold AUC from the 5-fold MLP run (job 51442237, under `results/phase3/runs/`) so the
   TKAN-vs-MLP comparison is on the *same* folds. I'll build the table once it's local.
4. **XAI pass** on the strong checkpoints (e.g. fold 2 `a654129d…`, fold 0 `61c67bdd…`):
   `python scripts/xai_phase3.py --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt
   --npz-root $PWD/phase2_signals` (auto-finds the run's `preprocessing.json`).
5. **Severity head (optional, quality).** Heavy tail + fold-4 outliers → try a log/robust
   (Huber) severity target or per-fold severity scaling — the next modeling lever *if* severity
   matters for the deliverable. Classification is already done.

**Pull run artifacts local (for the table / plots) — one remote path per rsync on this Mac:**
```
rsync -av --exclude='*.pt' \
  yashkulk@greatlakes.arc-ts.umich.edu:/nfs/turbo/coe-jaabell/PRI2M/results/phase3/runs/ \
  ~/Downloads/PRI2M/SINDy_DDE/results/phase3/runs/
```
(`--exclude='*.pt'` skips the heavy checkpoints; drop it to also pull `best_auc.pt` for XAI.)

When you want anything committed, say so (commit will carry only your name; no Claude co-author).

## INCORPORATE datasets 121–155 — IN PROGRESS (2026-06-17)
Full plan: `docs/superpowers/plans/2026-06-17-incorporate-121-155-eval-and-leakage-fix.md`.
User direction: **eval-only** (no retrain); 122–125 = OOD probe; fix grouping by param-twin co-grouping.
EDA verdict: 126–155 are exact param-twins of 1–30 (replication); only 122–125 are novel (noisy-only);
latent **1≡23 / 16≡17** leakage existed in the published 0.955 folds (now fixed). Raw H5 for 121–155 are
on the **current** Kaggle version (06-11-2026), fetched to GL `versions/new_121_155/` (35 folders, 19 GB);
v10 has only 1–120. Kaggle ships NO feature CSVs — the user generated `features_*.csv` from H5; so we
regenerate the physics metadata ourselves.

DONE this session (local, TDD, uncommitted — **full suite 406 passed / 1 skipped**):
- **Leakage fix:** `src/pri2m/phase3/splits.py` `parent_group` now reads `data/metadata/parent_groups.json`
  (generated by new `scripts/build_parent_groups.py`; param-identity connected components, 62 groups,
  arithmetic fallback). Co-groups 1≡23, 16≡17, 121→14, 126→1, 139→14, 150→25, etc.; 122–125 own groups.
  `tests/test_parent_groups.py` (18). Only merges, never splits → existing splits tests unchanged.
- **DATA_VERSION** parametrized (default 10) across extract script / sbatch / sindy_preprocess / README.
- **Metadata generator:** new `scripts/build_features_121_155.py` (+8 tests). Static xlsx→CSV mapping
  reverse-engineered + validated LOCALLY against 1–120 (cut_dir = xlsx `ud`; b_mm=b*1000; blim_mm=interp*1000;
  label=int(b/blim≥1)). H5-derived columns (omega/b/blim) need GL.

NEXT (GL, user runs — sync local changes first via `sync_phase3_tkan_to_hpc.sh`):
1. [DONE] Validate generator reproduces 1–120 using **versions/10**, generate 121–155 CSV, extract NPZ.
2. [DONE 2026-06-17] **Held-out eval on the noisy new data (121–125 + OOD 122–125) — job 51949469, PASSED.**
   All 5 `best_auc.pt` checkpoints, eval-only: **AUC 0.908 ± 0.013** (121–125) / **0.916 ± 0.011** (OOD 122–125);
   every fold > MLP ~0.87; 0 excluded. Retires the val-selected caveat. Per-fold table +
   takeaways in `docs/phase3_tkan/heldout_eval_121_155.md`. JSONs local at `results/phase3/heldout/`.
3. [OPEN] **126–155 clean replication eval** — NOT in job 51949469 (that job ran only the noisy new
   data). Build the `features_datasets_126_155.csv` eval table + extract NPZ if needed, then run the
   same `eval_heldout_phase3.py` sweep over 126–155 for the replication half of the story (or confirm
   it's deprioritized — 126–155 are exact param-twins of 1–30, i.e. a reproducibility check, not OOD).
4. [OPTIONAL] Threshold calibration if a held-out *accuracy* number is wanted (per-fold acc swings
   0.72–0.84 because the 0.5 cut isn't calibrated to the new data's class balance; AUC is unaffected).

Watch (subagent flag): validate against 1–120 with **versions/10** H5, not the new download — if the
06-11-2026 version regenerated the 1–120 stability boundaries they may differ from the originals.

---

## PARKED (original 2026-06-14 notes — superseded by the plan doc above)
Deferred by user 2026-06-14 (do Phase 3 finishing first). Context in memory
[[dataset-expansion-121-155]]. `data/metadata/Dataset descriptions.xlsx` documents **155**
datasets; the pipeline only processed **120** (clean 1–60 + noisy mirrors 61–120). The new
**35 (121–155)** = 30 clean (126–155) + 5 at 5% noise (121–125); 9 carry process damping
(Cn≠0: 122,124,125,150–155). GL has **only `versions/10` (120)** — 121–155 are NOT on GL.

Do in this order:
1. **Get the raw H5 for 121–155.** First confirm whether they're on public Kaggle
   (the `kaggle datasets files … | grep "Dataset 12x"` came back empty → may be owner-supplied,
   not yet on Kaggle — ask the data owner / get the link). Pull ONLY the 35 new folders into a
   new `versions/<N>/` (per-file `kaggle … -f`, or whole-version download keeping only the new
   `Dataset 121–155 h5/`). Do **not** re-pull 1–120.
2. **Fix leakage mapping FIRST (gating correctness).** `src/pri2m/phase3/splits.py`
   `_parent_group_array` (≈L89–90) is arithmetic (61–90→−60, 91–120→−90, else d); it CANNOT
   express the new pairing. Replace with an explicit lookup: **121→139** (confirmed exact
   modal-param mirror); **122,123,124,125 have NO exact clean twin** — resolve by fuzzy param
   match or confirm noisy-only (own group is safe only if truly no clean twin exists);
   **126–155 = own parent** (verify none duplicate an existing clean dataset's params). Keep
   1–120 mapping byte-identical. Add a pytest. Without this, folds silently leak.
3. **Parametrize the data version.** Replace the 5 hardcoded `versions/10` refs
   (`scripts/extract_phase2_signals.py:48`, `slurm/phase2_extract_signals.sbatch:24`,
   `sindy_pipeline/sindy_preprocess.py:44`, `slurm/README.md`, `handoff.md`) with one
   `DATA_VERSION`/`DATA_ROOT`. Audit `build_phase3_dataset.py` + Phase 2.5 mirror-ladder for
   any other hardcoded 60/90/120 / count assumptions.
4. **Extract just the 35 new datasets** (Phase 2 H5→NPZ) → recalibrate Phase 2.5 only for the
   new 5% batch → rebuild the Phase 3 table+folds over 1–155.
5. **Pick a path:** (a) **held-out eval (do first, cheap):** run the existing 1–120
   `best_auc.pt` checkpoints on 121–155 — no retraining — for a real generalization number that
   retires the “val-selected” caveat; new data is interpolative (within the 1–120 param
   envelope; only new vals seen: ktc 650e6, Cn 350k). (b) **pool + retrain** the 5-fold sweep on
   1–155 only if you want the new data in training.
6. **Science wins to attempt:** add **Cn/fn/ktc/teeth/cut_dir as side inputs** (currently only
   omega/tpf/tau); use the **process-damping datasets to recover the 700 dropped low-speed
   rows** (negative-blim at omega 1000–1500 is the no-damping artifact these datasets fix);
   confirm whether the new Cn datasets cover that speed regime before relying on it.

## Resume protocol (cold session)
```bash
cd /Users/yash/Downloads/PRI2M/SINDy_DDE
git rev-parse --abbrev-ref HEAD          # expect phase3-tkan
git log --oneline -12                    # see which phases committed
python -m pytest -q 2>&1 | tail -20      # see what's green / what's missing
```
Map the last green wave to the **Status board**, then continue from the first unchecked item.

## Log (newest last)
- 2026-06-04 — Brainstormed; design approved (6 decisions). Created branch `phase3-tkan`, spec, plan, this progress doc.
- 2026-06-04 — Phase 0 done: deps in pyproject (`phase3`,`sigkan` groups), `src/pri2m/xai/`, shared `tests/_phase3_synth.py` + smoke test (3 green), `mlflow` installed, conftest adds repo root. Next: dispatch Phase 1 (1A/1B/1C) in parallel.
- 2026-06-04 — Phase 1 done (3 parallel agents): 1A data layer (24), 1B KANLinear (19), 1C losses/metrics (31). Wave-1 gate 77 passed; full suite 205 passed, 0 regressions. All 4 source files reviewed — contracts met; 1C caught+fixed a NaN-gradient leak. **Integration note for P2:** feed every KANLinear input through `tanh` (grid_range (-1,1) vs z-scored data). 2B scope reduced (no TKANModel); driver assembles `tkan_model.py`. Next: Phase 2 (2A tkan, 2B heads+baseline).
- 2026-06-04 — Phase 2 done (2 parallel agents + driver): tkan cell/encoder (14+2skip), heads/baseline (23), driver-assembled `models/tkan_model.py` with data→model integration test. Wave-2 gate 43 passed; full suite 248 passed, 0 failures. Reviewed all files — tanh-squash honored, delayed-feedback wired. Next: Phase 3 trainer+CLI (single agent).
- 2026-06-04 — Phase 3 done (single agent): trainer+CLI+config (15 tests), reviewed. Wave-3 gate 15 passed; full suite 263 passed. Installed XAI deps: captum 0.9.0, shap 0.52.0, dice_ml, iisignature 0.24 (built — sigKAN tests will now RUN). Next: Phase 4 (4A XAI, 4B sbatch+E2E) in parallel.
- 2026-06-04 — Phase 4 done (2 parallel agents + driver): XAI module + xai CLI (20), keystone E2E (3). Reviewed counterfactuals + sbatch; **fixed sbatch `PROJECT_ROOT` from stale `/SINDy_DDE` → live `/nfs/turbo/coe-jaabell/PRI2M`**. **Final gate: 287 passed, 1 skipped (intentional iisignature-absent branch).** Verified the real workflow end-to-end: `train_phase3.py` (tkan,--smoke) → checkpoint → `xai_phase3.py` → 6 artifacts. **BUILD COMPLETE, uncommitted per policy.**
- 2026-06-05 — Wrote `sync_phase3_tkan_to_hpc.sh` (one Duo → multiplexed rsyncs, recursive dir mirror, prints copy-paste GL steps). Verified bash syntax + every referenced local path exists. TKAN-only (no sigKAN/iisignature). Nothing committed. Fixed a word-splitting bug (`-e "$RSH"` must stay quoted → wrapped rsync in `rs()` fn); added the 3 builder-input CSVs (~51 MB) to the manifest after they turned up missing on GL.
- 2026-06-05 — **GL deploy + launch.** Transfer + `pip install -e '.[phase3]'` OK; GL sanity subset 123 passed / 0 fail (login-node run, intentionally not finished). NPZ tree already on GL at `phase2_signals/dataset_<id>/*.npz` → train with `NPZ_ROOT=$PWD/phase2_signals`. `build_phase3_dataset.py` → 33719 rows, **700 neg-blim + 560 metrics-unavailable excluded (weight 0)**, 32459 supervised-usable; weights normalized mean 1.0 / max 1.62 (upper_q=0.99 tamed heavy tail). Folds: 5×12 val groups, 5 dev groups `[1,4,29,35,39]`, val rows sum to n_total, **`leakage_verified: true`** (both pre-flight blockers cleared). Launched `slurm/phase3_train.sbatch` for MODEL=mlp (job 51439343) and MODEL=tkan (job 51439345), 5 folds each; both reached "Training fold 0" (runtime leakage assert passed).
- 2026-06-06 — **GPU + speed + metrics pass (uncommitted).** Diagnosed GLC Logs: job 51442237 (mlp, 5 folds) finished 3h40m; job 51442244 (tkan) killed by time limit. Root slowness = `TKANEncoder.forward` 1024-step Python recurrence (3 KANLinear/step); pipeline was CPU-only (no device handling anywhere). Changes: (1) **device plumbing** — `resolve_device`/`_batch_to_device` in `phase3_trainer.py`, model+batch moved to device in fit/evaluate/gate, `device`/`num_workers` added to `TrainConfig`, logged to MLflow. (2) **metrics bug** — `metrics.py` `acc` now from the classification head `sigmoid(mean_logit)>=0.5` (was thresholding the severity regression `mean_pred`, which decoupled acc from auc and caused the wild swings); trainer summary now also reports `best_epoch`/`best_val` (the metrics of the kept best.pt) + `best_val_*` MLflow metrics. (3) **DataLoader speedups** — `build_fold_loaders` gained `num_workers`/`pin_memory` (+persistent_workers); `NormStats.apply` is now dtype-preserving so float32 windows normalize in float32 (no float64 round-trip). (4) **CLI+sbatch** — `train_phase3.py` `--device`/`--num-workers`; new self-contained `slurm/phase3_train_gpu.sbatch` (GPU `--gres=gpu:1`, `--array=0-4` one fold/task, `DEVICE=cuda NUM_WORKERS=4`). Sync script + printed GL steps updated (tkan→GPU array). **Full suite: 295 passed, 1 skipped (intentional iisignature skip), 0 regressions.** pyarrow ruled out — data is NPZ arrays, not columnar; it doesn't touch the hot path. Next: user reruns tkan via `slurm/phase3_train_gpu.sbatch` on GL (confirm `gpu` vs `spgpu` partition for jaabell0).
- 2026-06-08 — **TKAN timeout diagnosed + fixed (uncommitted).** New GPU run (job 51468046, A40) still hit the 6 h limit with **0 epochs done**. Built `scripts/_bench_tkan.py` and benchmarked on a real A40 (gl1504, spgpu): the 1024-step recurrence is ~2880-3275x an MLP/batch and **kernel-launch-bound** — per-batch time is ~flat B=32->256 (~5.4-5.6 s) while ms/sample drops 169->12.7 (B=512), and the A40 is *slower* than CPU at B=32. So bigger batch is a near-free ~8-13x. Also surfaced a hard blocker: the `gpu` partition has **V100s (sm_70)** that the prism torch (`2.12.0+cu130`, archs sm_75/80/86/90/100/120) **cannot run** (`no kernel image`); **A40s are on `spgpu`**. Fix (TDD, 8 new tests, full suite 303 passed/1 skipped): `avg_pool_time`/`scale_tau` + `pool` param in `TKANModel` (avg-pool 1024->256, rescale tau), `pool` in `TrainConfig`/`build_model`/CLI; yaml `batch_size 256` + `lr 0.003` + `pool 4`; sbatch pinned `--partition=spgpu`. End-to-end `train_one_fold(pool=4)` smoke OK. See [[phase3-tkan-perf]]. Next: user re-syncs + relaunches (smoke fold 0 first, check AUC vs MLP ~0.87).
- 2026-06-05 — **OOM fixed (root cause, not mem bump).** Both jobs were `oom_kill`ed at 32 G ~4 min into fold 0, identically across mlp/tkan → shared data layer, not model. Root cause in `data/phase3_sequence.py` `build_fold_loaders`: it (1) stored windows as **float64** (model consumes float32), (2) built a throwaway `train_for_norm` dataset → **loaded every train file twice** (load-counter test proved 60 reads for 36 files), and (3) `_stack_all_windows` made a **full concatenated copy** of all windows just to fit norm. Peak ≈ 2×train(f64)+concat+val. Fix: float32 storage; new exact `NormStats.fit_streaming` (two-pass, one array resident at a time); load train once and fit from its cache; deleted `_stack_all_windows`. Peak now ≈ train(f32)+val(f32), ~3–4× smaller. TDD: 3 new regression tests in `test_phase3_sequence.py` (float32 dtype, fit_streaming==fit, each file read once). **Full Phase 3 suite: 222 passed, 1 skipped, 0 regressions.** Next: re-sync `src/pri2m/data/phase3_sequence.py` + test, re-launch both sbatch jobs.
- 2026-06-09 — **"Timeout again" diagnosed: NOT a bug — self-imposed wall.** User reran a single array task (job 51529121, spgpu/A40, `Device:cuda`, `Smoke:0` → full 20-epoch fold) and it hit TIMEOUT at 6 h. Evidence ruled out every prior failure mode: fix IS live on GL (`batch_size 256`/`pool 4`/`lr 0.003` in synced yaml; `pool` in `tkan_model.py`), ran on a real `NVIDIA A40` on `spgpu`. It completed **3 epochs** (epoch_0/1/2 mtimes 19:25/20:57/22:28 → **~1.5 h/epoch**), so 20 epochs ≈ 30 h can't fit the **6 h `#SBATCH --time`** — which was self-imposed, not a cap (`spgpu MaxTime=14-00:00:00`). Root cause of the under-delivery: the projected ~32x (8x batch × 4x pool) was wrong — pool gave ~4x but the big batch added ~nothing because at 256 steps the recurrence is compute-bound, not launch-bound (the "free batch" only held at 1024 steps). **TKAN is undertrained, not underpowered:** val_auc 0.643→0.756→**0.814** (e0→e2), train_loss 124→50→15.6, killed mid-climb just below MLP's 0.87. Fix: `slurm/phase3_train_gpu.sbatch` `--time 6:00:00 → 36:00:00` + corrected the stale timing comment. Lossless — no extra pooling. Next: user relaunches fold 0 with the longer wall, confirms AUC > 0.87 before all 5 folds. Flagged 2 follow-ups (best.pt selected by MAE not AUC; per-epoch cost probe).
- 2026-06-09 (cont.) — **Perf probe + 2 preprocessing fixes (3 parallel subagents, TDD, uncommitted).** (1) `scripts/_probe_tkan_epoch.py` — drives the REAL loaders+model+train step on one fold for N batches, splits wall-time into data-fetch / host→device / synced-GPU, extrapolates an epoch estimate, `--compile cell` arm (torch.compile the recurrent cell — the metric-safe launch-bound lever). Added to `sync_phase3_tkan_to_hpc.sh`. Verified eager arm end-to-end on CPU synth (GPU 99.9% → GPU-bound); CPU `--compile` too slow to finish locally (CPU inductor — expected; A40 run is the real test). Re-did the perf arithmetic: old run was ~47 h/epoch (not ~6 h floor), so batch+pool delivered the full ~31× as designed — run is **GPU-launch-bound**, not data-bound. **(2) Norm-stat persistence (Agent 1):** `train_one_fold` was discarding the TRAIN-only `NormStats` (`_norm`), so a checkpoint's z-score was unrecoverable. Added `NormStats.to_dict/from_dict`; trainer now writes `<run_dir>/preprocessing.json` (norm mean/std, channels, window, pool, fold, counts, excluded_reasons, leakage_verified, seed) + mlflow artifact + a `preprocessing` block in summary. **(3) XAI off-distribution fix (Agent 2):** `xai_phase3.py` fed RAW windows to a model trained on z-scored windows → tanh saturates → meaningless attributions. Now auto-discovers `preprocessing.json` (or `--preprocessing`), z-scores the window before IG, warns loudly if absent. **Full suite 313 passed / 1 skipped, 0 regressions.** **Chatter-band-vs-Nyquist check is still OPEN:** the table `tau` is the tooth-pass period in **seconds** (tau·tpf=1), not the sample delay, so `fs` must come from NPZ meta — run the band one-liner on GL (fn p50≈950/p95≈1500 Hz, dominant_freq p50≈400-465 Hz; pool 4 safe only if fs/8 > those). **Paper:** read gao JMSE-2025 — medium relevance, key idea = process-param side inputs stabilize training; see [[gao_milling_ml_paper]].
- 2026-06-09 (cont. 2) — **Probe ran on real A40 + compile shipped (TDD, uncommitted).** Probe on fold 0 (853,335 train windows, 3334 batches/epoch): **GPU-bound** (data fetch 0.6%), eager **~77 min/epoch** (1374 ms/batch); `--compile cell` → **399 ms/batch = 3.4× → ~22 min/epoch**, still GPU-bound. Confirms data-pipeline caching would buy ~nothing (Idea 1 correctly dropped). Implemented the win in the real training path: `TrainConfig.compile` + `maybe_compile_cells` (CUDA-only, mutates encoder cells in place, also enables TF32) + `strip_compile_prefix` (drops the `_orig_mod.` keys torch.compile adds, so checkpoints still load into a plain TKANModel — the XAI/inference path); routed all `fit()` checkpoint saves through the stripper; `--compile/--no-compile` CLI + `compile: true` in yaml. TDD: 4 new tests incl. a compiled-save→uncompiled-load round-trip. **Full suite 317 passed / 1 skipped.** sbatch comment updated (compile ~8 h/fold vs ~26 h eager; --time kept 36 h to cover eager fallback). Next: user re-syncs + relaunches fold 0 (compile on by default) — expect ~8 h; confirm val AUC > MLP 0.87.
- 2026-06-09 (cont. 3) — **AUC-best checkpoint shipped (keep-both, TDD).** `best.pt` was selected by lowest val **MAE**, but the headline vs MLP is **AUC**, which keeps climbing after MAE plateaus (fold-0 evidence: MAE best at epoch 1, AUC best at epoch 2) — and retention deletes the per-epoch checkpoints, so the AUC-best epoch was being lost. Fix (additive, non-breaking): `fit()` now also mirrors the highest-finite-val-AUC epoch to **`best_auc.pt`** (NaN-safe + end-of-fit fallback); retention auto-keeps it (name contains "best"); summary/MLflow now report `best_auc_epoch` + `best_auc_val_*` so the cited number is the actual AUC peak, not the MAE-best epoch's AUC. `best.pt` (MAE-best) unchanged. New `_best_auc_epoch` helper. For XAI on the classifier, point `--checkpoint` at `best_auc.pt`. TDD: 4 new tests (helper max/NaN, best_auc.pt written+retained, summary keys). **Full suite 321 passed / 1 skipped.** Resolves follow-up #1.
- 2026-06-09 (cont. 4) — **Compile live ~25 min/epoch (job 51601910, ~8 h/fold projected) BUT training is unstable.** val_auc bounced 0.768/0.820/0.733/0.695/**0.840** (e0–e4), peak ~0.84 near MLP 0.87 with 15 epochs left. Same-seed/same-config divergence vs the earlier no-compile run (which was 0.643/0.756/0.814 at e0–2) — even epoch 0 differs (0.643 vs 0.768) → ill-conditioned optimization (sensitive-dependence on numerics), not the model. Removed the TF32 I'd bundled into `maybe_compile_cells` (`set_float32_matmul_precision("high")`) — it drops matmul precision and perturbs the spline training; compile is now pure fp32 kernel fusion. **Root cause is optimization stability, not TF32:** `lr=0.003` (sqrt-scaled) likely too high, no warmup/schedule/grad-clip. NEXT-RUN recommendation (needs user/prof sign-off, not changed unilaterally): lower lr and/or add warmup+cosine + grad-norm clip; possibly rebalance loss weights (classifier AUC may wander while regression dominates gradients). This run lacks best_auc.pt (pre-fix launch) — the bounce makes best_auc.pt essential; full 5-fold run with synced code fixes it. Current job: let it finish for timing + peak-AUC read.
- 2026-06-10 — **Fold 0 FINISHED (full 20 epochs) + tuning plan decided (NOT implemented this session).** Job 51601910 ran to completion: 8h22m / **~25 min/epoch** (confirms compile's 3.4x; per-fold ~8 h). Final-log read: `leakage_verified: true`, 700 `invalid_blim_low_speed` + 560 `metrics_unavailable` excluded. **Result MARGINAL:** best epoch 6 -> AUC **0.882** / MAE 0.773; final epoch 20 -> AUC **0.854** / MAE 0.911 -> TKAN *matches* but does not *beat* MLP (~0.87), and on 4x-pooled input. **Training unstable:** train_loss bounces, val AUC swings 0.69-0.88, peaks e6 then no sustain -> lr too hot, no schedule/clip. **Decision (see rewritten NEXT ACTION):** before any 5-fold sweep, implement a toggleable bundle — (1) Gao process-param **side input** (FEED `omega`/`tpf`/`tau`; **NEVER** `b`/`b_mm`/`blim` = leakage; see [[gao_milling_ml_paper]]), (2) cosine LR + warmup, (3) grad-norm clip, (opt) AdamW wd — TDD, rerun fold 0, confirm val AUC *stabilizes and sustains* >=0.88 on `best_auc.pt`, THEN `sbatch --array=0-4`. **Cost corrected:** 5 folds ~= 40 A40-h / ~8 h wall arrayed (old "~150 A40-h" assumed 30 h/fold). Updated `handoff.md` (new 2026-06-10 section) + memory [[phase3-tkan-perf]]. Nothing implemented or committed this session — the next session implements the bundle.
- 2026-06-10 (cont.) — **Tuning bundle IMPLEMENTED (TDD, 2 parallel waves, 4 subagents, uncommitted).** Wave 1: (A) `TKANModel` side-input branch — `side_input/side_features/side_hidden` ctor args, `DEFAULT_SIDE_FEATURES=("omega","tpf","tau")`, `FORBIDDEN_SIDE_FEATURES` leakage guard raising at construction, `set_side_norm` buffers in the state_dict, severity-head width `H+side_hidden`, trajectory head untouched; functional tests prove mutating `b_over_blim`/`label` in the batch cannot change outputs while `omega` does (9 new tests). (B) Optimizer bundle — `build_optimizer` (adam/adamw+wd) + `schedule_factor` (pinned warmup/cosine semantics) + LambdaLR stepped per epoch + `clip_grad_norm_` when `grad_clip>0` + per-epoch `lr` MLflow metric + CLI/yaml (17 new tests). Wave-1 gate: 347 passed/1 skip. Wave 2: (C) trainer wiring — `TrainConfig.side_input/side_hidden`, `build_model` passes them (mlp+side_input=ValueError), window-weighted TRAIN-only `side_feature_stats` -> `set_side_norm` before fit + `"side"` block in `preprocessing.json` + mlflow params + CLI `--side-input/--side-hidden` + yaml `side_input: true`; yaml-sourced side_input auto-dropped for mlp so the baseline sbatch keeps working (14 new tests); also fixed `_sample_for_label` to carry omega/tpf so run_gate works on side models. (D) XAI compat — `integrated_gradients(extra_scalars=...)` (guards "window"/"tau" overrides), `_load_model` reconstructs side models (side_mean detection, features from preprocessing `side` block, fallback+warning), `_sample_window` surfaces real omega/tpf, manifest gains side info; **fixed latent `_infer_dims` bug** (hidden_size over-counted by side_hidden on side checkpoints) (9 new tests). **Final gate: full suite 370 passed / 1 skipped (intentional iisignature skip), 0 regressions.** E2E test drives the real CLI with the updated yaml, so tkan smoke now exercises side+adamw+cosine+clip end-to-end; XAI synthetic round-trip verified both with and without side checkpoints. Sync manifest already covers all changed files. Next: user syncs + relaunches fold 0 (see NEXT ACTION).
- 2026-06-10 (cont. 2) — **First bundled fold-0 run (job 51641089) crashed; root-caused + hardened (2 parallel subagents, TDD, uncommitted).** Crash: `ValueError: Input contains NaN` from `roc_auc_score` inside the FIRST `evaluate()`. Systematic debugging: table audit found 0 non-finite omega/tpf/fn (33719 rows); NPZ omega must parse to float (empty-string fallback would have ValueError'd every prior run at dataset build); losses are NaN-safe (masked targets, clamped denominator); `side_feature_stats` uses the stable two-pass form. Surviving root cause: **KAN-GRU gradient blow-up -> NaN grad; the new `clip_grad_norm_` (default `error_if_nonfinite=False`) computed `clip_coef = max_norm/NaN` and NaN-scaled every gradient -> one `optimizer.step()` -> all-NaN model** (without clip a NaN grad only corrupts its own path — naive clipping AMPLIFIES). Side branch also reshuffled the init RNG draw (extra params), so this run sat at a different point of an already ill-conditioned optimization. Fixes: trainer skip-and-count guards (`history["nonfinite_batches"]`, MLflow metric, stderr warning, RuntimeError on zero-step epochs) + NaN/inf-tolerant `per_file_severity_metrics` (`n_files`/`n_nonfinite_files`, never raises; ±inf logits retained via sigmoid). **Full suite 380 passed / 1 skipped (381 collected), 0 failures at any gate.** Forensic check for GL: the dead run's `checkpoints/` dir should be EMPTY (crash precedes the first checkpoint write) — confirms epoch-0 death.
- 2026-06-11 — **Fold-0 rerun COMPLETED + gate PASSED (GL job 51651316, run `931af482deca48b9bb2d1cd1689832d9`, 20 epochs / 8h24m).** `best_auc.pt` = epoch 6, **val AUC 0.9388** / acc 0.840 / MAE 0.611; `best.pt` = epoch 19, MAE 0.548 / AUC 0.927 / acc 0.861 — clearly beats MLP ~0.87 and the pre-bundle peak 0.882; the tuning bundle is validated. **The gradient blow-up recurred but the guards converted a crash into a detour:** epochs 7–11 skipped 15/10/**1661**/19/9 of 3334 batches (train_loss 1.38 -> 17.2/34.9/**143.9**), weights stayed finite (`n_nonfinite_files: 0` at every eval), training fully recovered (epochs 12–19 zero skips, loss down to 0.51, AUC back to 0.927) — but the 5-epoch detour meant the epoch-6 peak was never re-attained. Effective-lr forensics: every skip epoch ran at cosine lr 0.0015–0.0025; every epoch at <= 0.00124 was clean -> the stable ceiling is ~0.0015 once the spline weights have grown. **Applied the pre-agreed fallback (one yaml line, uncommitted): peak `lr: 0.003 -> 0.0015`** so the entire schedule sits at/below the ceiling. Next: user re-syncs, reruns fold 0 at 0.0015 (expect 0 skips, best_auc >= ~0.93, peak possibly later than epoch 6), then the 5-fold sweep + `--no-side-input` ablation. The existing 0.9388 `best_auc.pt` is already XAI-ready meanwhile.
- 2026-06-14 — **Full 5-fold TKAN sweep evaluated (lr 0.0015 stability config) — PASSED.** Pulled GL logs (job 51670644 fold 0 + array 51711709 folds 1–4) to `GLC Logs/`. **Stability fix validated: 0 skipped/non-finite batches across all 100 epoch-folds** — the `lr 0.003→0.0015` fallback fully killed the KAN-GRU gradient blow-up (the 0.003 fold-0 had skipped 1714). 5-fold `best_auc.pt` = **0.955 ± 0.022** (folds 0–4: 0.960/0.912/0.975/0.961/0.970); every fold beats MLP ~0.87 (min margin +0.042) and the 0.88 gate; fold 0 @0.0015 (0.960) > its own @0.003 (0.939). Severity head heavy-tailed (MAE 0.88±0.65, RMSE 4.37±2.65; fold 4 outlier-dominated 2.13/9.0) — the known weak head. err files = benign TF32 hint only; `leakage_verified: true` + 700+560 excluded every fold. **All five ran `side_input: true` → the `--no-side-input` ablation is still OPEN** (side-input value not yet isolated). Updated Status board + NEXT ACTION here, handoff.md, memory [[phase3_tkan_perf]]. Next: commit the validated bundle (on user word), run the ablation, pull matched MLP per-fold AUC (job 51442237) for the baseline table, XAI on strong folds.
- 2026-06-22 — **Held-out eval on 121–125 evaluated (GL job 51949469, array 0–4) — PASSED; first commit of the project.** All 5 `best_auc.pt` checkpoints (`61c67bdd`/`c21e58dc`/`a654129d`/`3a40b715`/`09378839`) run eval-only (no retrain) on never-trained data. **Held-out AUC 0.908 ± 0.013** (all new noisy 121–125, 957 files) and **0.916 ± 0.011** (OOD novel 122–125, 855 files; OOD ≥ full set on every fold — process damping does not break it). Every fold > MLP ~0.87 (min 0.8907); 0 excluded; per-file scoring → directly comparable to the 0.955. **This retires the "val-selected, not held-out test" caveat** — the 0.955→0.91 gap is the honest generalization drop. Severity head does NOT transfer (MAE 3.3–4.8 / RMSE 7.8–11.2 vs in-dist 0.88). acc 0.72–0.84 is a 0.5-threshold artifact (AUC unaffected). JSON values verified against the GL logs; full per-fold table + takeaways written to `docs/phase3_tkan/heldout_eval_121_155.md`. **126–155 clean replication eval was NOT in this job → still OPEN.** Verified full suite **409 passed / 1 skipped** then, on explicit user instruction, made the **first git commit** of the whole SINDy_DDE project (193 files; `.gitignore` excludes the 3.8 GB archive/venv + all artifacts; Word lock file + peripheral summer_research/rapidminer/GLC-Logs dirs excluded). Commit carries the user's name only (no Claude co-author) per [[git_commit_policy]].
