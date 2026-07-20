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
- [x] **White-box XAI pipeline BUILT + tested (2026-06-25, local; branch `phase3-tkan`)** — turns the trained TKAN into a full white box. Spec: `docs/superpowers/specs/2026-06-24-phase3-whitebox-design.md`. New `src/pri2m/xai/`: `semantics.py` (edge index-map: `in[68..71]`=τ-delayed channels), `activations.py` (data-aware per-edge importance, replaces the useless 0/14k mean-|w| prune), `symbolic.py` (each top edge → closed-form φ via BIC), `feature_importance.py` (channel/lag/side + permutation), `physics.py` (delay-ablation, τ-sweep, delayed-weight-share, attribution spectrum), `faithfulness.py` (pruning curve, deletion, randomization sanity), `case_studies.py` (chatter-vs-stable walkthrough). Plus `KANLinear.edge_function` (base+spline, fixes the SiLU-omitting plot). Runner `scripts/whitebox_phase3.py` → `whitebox_manifest.json` + figures; `scripts/render_whitebox_report.py` fills `docs/phase3_tkan/whitebox_analysis.md` (template; **every number from the manifest, never invented**); GL wrapper `slurm/phase3_whitebox.sbatch`. TDD throughout: **full suite 469 passed / 1 skip**, real CPU end-to-end smoke = 13 artifacts, 0 analysis errors.
- [x] **White-box XAI RAN on the real fold-2 checkpoint (2026-06-25, GL job 52339476; run `a654129d…`, val AUC 0.975)** — eval-only, ~9 min CPU, produced all 13 artifacts + `whitebox_manifest.json` + rendered `whitebox_analysis.md` (local: `results/phase3/xai/fold2/`). **Verified findings + honesty caveats + research-driven TODO are in NEXT ACTION below.** Post-run fix: the sampler used `head()` → ceilinged the sample AUC at 1.0; switched to **seeded random stratified** sampling (`--sample-seed`) + an auto ceiling-caveat in the report, so **fold 2 must be re-run and folds 0/1/3/4 run** with the updated runner for un-ceilinged channel/side importances (the ablation/sweep/pruning *deltas* were always valid).
- [x] **Verified external-research scan (2026-06-27)** — 4 parallel web subagents, verified-only (every citation fetched + quoted; 4 key post-cutoff papers re-verified by hand). Full grouped list + "how it maps to our pipeline": `docs/phase3_tkan/related_work_scan_2026.md`.
- [x] **White-box REPLICATION across folds COMPLETE (2026-07-01, GL via `slurm/run_whitebox_all_folds.sh`, N_WINDOWS=512, un-ceiled runner)** — ALL FIVE folds returned (fold 3 after one resubmit; its first job, 52701776, died without writing artifacts). Delay ablation replicates on every fold: AUC drop 0.230/0.487/0.266/0.484/0.449, mean 0.38 (fold 1 → exactly **0.500 = chance**, fold 3 → 0.510, fold 4 → 0.541: **3 of 5 at/near chance**), severity MAE explodes 3.1–16.8 on all folds; x_delay weight share 9.7–10.7% everywhere; randomization sanity passes everywhere; 25/25 top edges nonlinear everywhere. Sharper: 512 windows did NOT clear the ceiling (baseline 0.987–0.996 → magnitudes still compressed); side-input permute drop shrinks to 0.002–0.009 (fold-2's 0.012–0.016 was the high end) → the no-side-input **training** ablation is the real test; global τ-sweep peaked at 2τ on ALL FIVE runs → per-window-relative sweep is mandatory before interpreting peak location. Cross-fold figure: `results/figures/research_update_assets/whitebox_delay_replication.png` (`scripts/plot_whitebox_replication.py`).
- [x] **Fuzziness suite + spline-edge atlas + XAI queue closure BUILT (2026-07-16; gate re-verified fresh 2026-07-17)** — plan `docs/superpowers/plans/2026-07-16-phase3-fuzz-and-spline-xai.md` (8 tasks / 3 waves, TDD). Graded degradation suite (`xai/degradation.py`: input_noise / input_smooth / tau_jitter / spline_coarsen / weight_noise + delay-reliance probes + breakdown points) with manifest-first runner `scripts/fuzz_phase3.py`; spline-edge atlas (`xai/spline_atlas.py` + `symbolic.eval_fit`) wired into the whitebox runner alongside the named heatmap AND TSR (closes the "TSR built but never called" gap — `results["tsr"]` now in the manifest); shared sampler `xai/sampling.py`; real-net per-window predictions dump `scripts/predict_windows_phase3.py` (unblocks real-net counterfactuals via `--predictions`); concept-probe circularity fix; conformal-ensemble wrapper; slurm `phase3_fuzz.sbatch` / `run_fuzz_all_folds.sh` / `phase3_predict_windows.sbatch` / `run_conformal_ensemble.sh` + sync-script updates (all `bash -n` clean). Full suite **585 passed / 1 skipped, 0 failures** (+25 tests; 24 in 7 new files). GL runs = user, per the plan's "After the build" list.
- [x] **Research Update 8 DRAFTED + rendered (2026-07-01)** — `summer_research/Research Updates/research_update_8.md` + `.docx` (3 embedded figures: fold-2 3-panel intervention summary, top symbolic edges, cross-fold replication). Covers Jun 23–Jul 1 (~20 h): white-box build → fold-2 run → ceiling-artifact retractions → replicated delay-ablation headline → verified scan → next steps. **FINAL, no placeholders** — fold 3 returned same day (0.9932→0.5098, drop 0.484); table, cross-fold figure, and docx all updated with the full 5-fold replication.
- [x] **Fuzziness suite RAN on all 5 folds (2026-07-17, GL; local `results/phase3/xai/fuzz/fold{0..4}/fuzz_manifest.json`)** — every fold returned, `self_check.passed: true` with `max_abs_dev` ≤ 5.5e-07 (input stressors provably non-mutating), `errors: {}` everywhere. Baseline per-window AUC 0.986–0.996 and delay_drop **0.230/0.486/0.266/0.483/0.449** — reproduces the Update-8 ablation exactly on the same checkpoints. Headline (Update 10, all manifest-derived): AUC never falls below **0.75** anywhere in the tested range; 32-sample smoothing still leaves 0.90–0.92; **τ-jitter is a third, causal line of delay evidence** (5% jitter costs fold 2 0.994→0.956; 50% settles at 0.84–0.87 — accuracy degrades in proportion to how wrong the delay is); 50% weight-noise still leaves 0.89–0.94 (wide flat basin); spline-coarsen to 1 knot costs only 0.05–0.10 AUC at refit R² > 0.92. Honest wrinkle: under heavy coarsening the delay drop moves in *different directions* by fold → not over-read. Figures: `fuzz_dashboard_fold2.png`, `fuzz_replication_5fold.png`.
- [x] **White-box RE-RUN with atlas + TSR, and the whole owed queue CLOSED (2026-07-17, GL; local `results/phase3/xai/{fold0,fold1,fold2_w512,fold3,fold4}` + `counterfactuals` + `conformal/ensemble` + `concept_probes/fold2`)** — verified in the manifests: **TSR is now actually called** (`results["tsr"]` present on all five re-run folds; `fold2/` is the stale pre-wiring dir, use `fold2_w512/`) and **spline atlas landed 25/25 accepted edges on every fold**. Two detours, both resource limits not method errors: the whitebox re-run overran its wall and was resubmitted bigger; the net counterfactual OOM'd and was re-scoped. Results split three wins / two honest negatives:
  - **WIN — network-grounded counterfactuals.** The 2-step predict→CF flow works; the surrogate now fits the **model's own predictions** (manifest records checkpoint sha256 + model-as-source). **81.6% agreement** with the analytic boundary (42 chatter points → 38 flip, **all** toward stable, 31 land below `blim`). The drop from Update 9's label-fitted 0.952 is the point.
  - **WIN — spline-edge atlas.** 25/25 edges carry an accepted formula on all 5 folds (R² 0.985–1.000); τ-delayed edges carry 15–16% of top-edge importance on folds 0/1/2 but only 3–4% on folds 3/4 — the same fold split the ablation shows.
  - **WIN — conformal ensemble** (5 checkpoints, `conformal/ensemble`): coverage **0.883** @ target 0.90, selective accuracy **0.996**, raw accuracy 0.984, abstention 0 at α=0.1; sweep well-behaved α=0.2→0.01.
  - **NEGATIVE (reported, not buried) — TSR does not find the delay.** Underpowered at **n_windows=4 / steps=8** per fold; channel shares nearly identical to plain IG (position 72–83% under both); `tr_peak_lag_mean` = 2.0/2.0/33.0/20.5/2.0 steps against `tau_raw_mean` 87 raw ≈ 22 pooled steps → only one fold near the delay. **Owed: re-run at hundreds of windows before this null is trustworthy.**
  - **NEGATIVE — concept probe still fails after the circularity fix** (`concept_probes/fold2`): held-out R² regeneration_phase_sin **−331.0**, _cos −5.83, chatter_tpf_separation −3.50, vs proximity_to_blim **+0.823**. De-circularized and still failing = the concept is absent from the 64-d state, not the target leaking. Claim retired.
  - **NULL — hidden-state dynamics** (from Update 9, unchanged): chatter |λ| 1.0153 vs stable 1.0142, Δ0.001 ≪ std; the manifest's `supported: True` oversells it.
- [x] **Research Update 10 DRAFTED + rendered (2026-07-18)** — `research_update_10.{md,docx}`, covers Jul 5–11 (~20 h): the graded stress-test suite and what survives it. 2 figures from the fuzz manifests.
- [x] **Quantile-severity wiring + SLURM passthroughs DONE (2026-07-20, local, TDD, 2 parallel subagents)** — closes the "built but not trainable" gap and unblocks BOTH pending GL retrains. (A) `severity_mode`/`severity_quantiles` now flow `TrainConfig` → `build_model` → `TKANModel`, with `ValueError` guards for `mlp`+quantile and unknown modes (mirrors the `side_input` guard); `--severity-mode` on the CLI with the same CLI>YAML>default precedence as `--side-input`; `severity_mode: huber` + `severity_quantiles: [0.1,0.5,0.9]` documented in `configs/phase3_tkan.yaml`. **Checkpoint round-trip fixed too:** a quantile head is *wider*, so `scripts/xai_phase3.py::_load_model` could not reload one (`size mismatch ... [3,8] vs [1,8]`); it now detects Q>1 from the state_dict and reads levels from a new `severity` block in `preprocessing.json` (block ABSENT in default mode, exactly like the `side` block). No loss-side changes were needed — `severity_loss` already routes off the model's output dict. (B) `slurm/phase3_train_gpu.sbatch` gained a generic `EXTRA_ARGS` passthrough (`read -r -a` split — deliberately NOT unquoted `$EXTRA_ARGS`, which also glob-expands: `"--tag-glob *"` measured 23 argv entries instead of 2); `slurm/phase3_whitebox.sbatch` + `run_whitebox_all_folds.sh` gained `TSR_WINDOWS`/`TSR_STEPS`. Both wrappers verified argv-identical when the new vars are unset. **Full suite 585 → 603 passed / 1 skipped, 0 failures** (+18 tests), re-verified independently by the driver; `bash -n` clean on all 3 shell files. Uncommitted per policy.
- [x] **Research Update 11 DRAFTED + rendered (2026-07-18)** — `research_update_11.{md,docx}`, covers Jul 12–18 (~20 h): atlas + TSR wiring + net-grounded counterfactuals + conformal ensemble + the two retired claims. 0 placeholders; every number traced to a returned manifest.

---

## NEXT ACTION

### ▶ CURRENT FRONTIER (2026-07-20) — the eval-only white-box queue is CLOSED; what is left is training

**Where the project actually stands.** Everything that can be learned from the five existing
`best_auc.pt` checkpoints *without retraining* has now been extracted, run, and written up
(Updates 8→11). The delay claim rests on **three independent interventions** — ablation (5/5 folds,
mean drop 0.38, 3 folds to chance), per-window τ-sweep (peaks at ×1 on **all five** folds), and
τ-jitter (accuracy falls in proportion to delay error) — plus a network-free SINDy-DDE recovery
(median τ*/τ_phys 0.986). Attribution-family methods (IG, TSR) do **not** corroborate it and are
written up as negatives. Two claims were retired on the evidence (concept probe, TSR).

**Therefore the frontier has moved: every remaining high-value item requires a GL *retrain*, not
another eval pass.** Do not add more eval-only XAI — it is saturated.

#### PENDING — in priority order

1. **Quantile severity head — the one remaining "the model got better" result.**
   **It is NOT currently trainable — the plumbing is half-finished.** Verified 2026-07-20 by
   reading source, not docs: `pinball_loss` + `severity_loss` quantile routing exist in
   `training/losses.py`, and `TKANModel(severity_mode=…, severity_quantiles=…)` /
   `SeverityHead(severity_mode="quantile")` exist in the model — but **`severity_mode` appears
   nowhere in `training/phase3_trainer.py` or `scripts/train_phase3.py`**. `TrainConfig` has no
   such field and `build_model` (`phase3_trainer.py:247`) never passes one, so setting a yaml key
   today would be silently ignored. Good news: `severity_loss` routes off the **model's output
   dict** (`"b_over_blim_quantiles"` present ⇒ pinball), so once the model is constructed in
   quantile mode the loss follows automatically — no loss-side work needed.
   - **1a [local, TDD] — DONE 2026-07-20.** Wiring + guards + CLI + yaml + the quantile checkpoint
     round-trip through `xai_phase3.py::_load_model`. Suite 603 passed / 1 skipped. See the Status
     board row. Nothing is blocking the run now.
   - **1b [GL retrain] — READY TO SUBMIT.** From the GL project root:
     ```
     env EXTRA_ARGS="--severity-mode quantile" \
         OUTPUT_DIR=/nfs/turbo/coe-jaabell/PRI2M/results/phase3/runs_quantile \
         sbatch --export=ALL slurm/phase3_train_gpu.sbatch
     ```
     (`--array=0-4` is already declared in the script, so this is all 5 folds; ~8 h/fold, arrayed
     ≈ 8 h wall. `OUTPUT_DIR` is optional but keeps quantile runs out of the validated huber tree.)
     **Sync first** (`./sync_phase3_tkan_to_hpc.sh`) — the wiring is local-only. Compare per-fold
     severity MAE/RMSE against huber **and** confirm chatter AUC does not regress from 0.955±0.022.
   Why it matters: severity is the only output never hardened (in-dist MAE 0.88 ± 0.65 / RMSE 4.37;
   held-out MAE 3.3–4.8 / RMSE 7.8–11.2 — it does not transfer). The quantiles also feed the
   conformal layer directly, giving prediction intervals for free.

2. **[GL retrain] `--no-side-input` 5-fold ablation — still open since 2026-06-14.**
   Verified 2026-07-20 across all run dirs: **every TKAN fold ran `side: True`**; the one
   `side: False` run (`93ea5f598f…`) is the pre-bundle lr-0.003 fold 0, not an ablation. So we can
   say TKAN+side beats MLP but cannot isolate the Gao side-input's contribution. The white-box
   *permutation* proxy is too weak to settle it (drop shrank to 0.002–0.009 at 512 windows) — the
   **training** ablation is the real test.
   **UNBLOCKED 2026-07-20 — READY TO SUBMIT.** The CLI flag `--no-side-input` always existed; what
   was missing was any way for the sbatch to pass it. `EXTRA_ARGS` now exists:
   ```
   env EXTRA_ARGS="--no-side-input" sbatch --export=ALL slurm/phase3_train_gpu.sbatch
   ```
   Sync first. Compare per-fold `best_auc` against the side-on sweep (0.960/0.912/0.975/0.961/0.970).

3. **[GL eval, cheap] TSR at scale — decide the null. READY TO SUBMIT.**
   Every completed run analyzed only `--tsr-windows 4 / --tsr-steps 8`, because the sbatch never
   threaded those flags (the runner has had them all along). Fixed 2026-07-20:
   ```
   TSR_WINDOWS=256 bash slurm/run_whitebox_all_folds.sh
   ```
   Sync first. If the flat channel split and the off-delay peak survive at 256 windows, **retire the
   attribution view of the delay** and let the three interventions carry it alone — say so
   explicitly in the next update. If they do not survive, the small sample was the whole story.
   Either outcome is publishable; the present n=4 one is not.

4. **[Research] Learnable τ — the strongest available science claim.**
   Differentiable delay kernel / neural-DDE variant; show the learned τ converges to the tooth-pass
   period from data. With the ablation (done) and the SINDy-DDE cross-check (done) that is **three
   independent recoveries of the physical delay** — a physics-discovery result, not a classifier.
   This is the next *research* direction, and it needs a design pass before any code.

5. **[Blocked on the data owner] 126–155 clean replication eval.**
   Not a code problem and not fixable here: the current Kaggle download ships 126–155 as `(N,4)`
   with one rectified envelope channel; four signed oscillating channels cannot be reconstructed
   from it. **Only unblock = request 126–155 from Schmitz in the standard 17-column format.**
   Then `build_features_121_155.py` → extract NPZ → `eval_heldout_phase3.py`, unchanged.

6. **[Small, but fix before the paper] `"huber"` is a misnomer.** Surfaced 2026-07-20 while wiring
   the quantile head. `_VALID_SEVERITY_MODES = ("huber", "quantile")` in `models/heads.py:34`, but
   there is **no Huber/smooth-L1 loss anywhere in `training/losses.py`** — the point path calls
   `weighted_mse` and the loss part is literally `"sev_mse"`. So the mode string says Huber while
   the math is MSE. Harmless today, but it would put a false loss name in a paper's methods section.
   Either rename the mode to `"point"`/`"mse"` (touches heads, trainer, yaml, tests) or document the
   discrepancy loudly. Predates this session — it came in with the 2026-07-04 build.

7. **[Optional / small]** Threshold calibration for a held-out *accuracy* number (per-fold acc
   swings 0.72–0.84 purely because the 0.5 cut is uncalibrated; AUC unaffected — the conformal
   ensemble already solves this if you would rather cite it). Aggregate delay-engagement stats over
   many windows (the n=1 `leans_on_delay` case flag was always noise). Chatter-band
   `attribution_spectrum` — but note that the TSR/IG negatives (item 3) make any further
   attribution work low-value until item 3 is settled.

8. **[Housekeeping]** Everything since commit `46d76736` is **uncommitted** per policy — including
   the entire fuzz/atlas/TSR build, all the returned GL manifests, Updates 9/10/11, and the
   2026-07-20 quantile wiring. Say the word and it gets committed under your name only.

**Follow-on-paper scope (not now):** MultKAN trunk, vanilla-TKAN / HAVOK comparators, chatter
prognosis (relabel by time-until-onset), sim-to-real DSAN, process-damping re-derivation to recover
the 700 dropped low-speed rows.

---

### Superseded — 2026-06-27 frontier (kept for the record)

> **Status of its TODO list as of 2026-07-20:** items **1, 2, 3, 5 are DONE** (fold replication +
> per-window sweep; TSR wired and run — negative; symbolic R²-gate, 25/25 accepted every fold;
> SINDy-DDE real-signal cross-check, median τ*/τ 0.986). Item **6 is partly done** (matched MLP
> table landed: 0.955 ± 0.022 vs 0.875 ± 0.004; vanilla-TKAN and HAVOK comparators never started).
> Items **4, 7, 8 were never done** and are carried forward — see PENDING 6 and the
> follow-on-paper list in the current frontier. All four caveats below were resolved or retired by
> Updates 9–11.

**Verified findings — fold-2 RANDOM-sample re-run (2026-06-27; supersedes the ceilinged `head()`
run). Two earlier claims were ceiling artifacts and are CORRECTED below.** The
regenerative-delay story survives the honest sampling:
- **Delay ablation (strongest, an intervention not an attribution):** zeroing the τ-delayed
  input drops chatter AUC **0.992→0.706 (−0.286)** and raises severity MAE **0.17→11.6**. (The
  head run showed −0.435; smaller here because random sampling spans diverse operating points and
  is harder. Load-bearing either way.)
- **Symbolic white box intact:** ~25 important edges, **all nonlinear** (tanh/sin/cubic), R²≈0.99;
  top edges = classifier hidden units + position (xtpos/ytpos) + their **τ-delayed** copies +
  (now) **side-branch** features (`side1`,`side12`).
- **Faithfulness:** pruning monotone (keep 5%→AUC 0.82, 100%→0.99); randomization cosine **0.06**
  (passes; = Adebayo 2018). x_delay weight share **10.7%** (≈1.9× its 5.5% input-count share).

**CORRECTED by the un-ceiling (these were ceiling artifacts in the head run):**
- **Force channels are NOT unused.** Permuting Forcex/Forcey costs ~**0.05–0.06** AUC (was ~0.001
  at the ceiling). Position still dominates (ytpos 0.23, xtpos 0.12). *(Retraction of the earlier
  "forces ≈ unused" claim.)*
- **The Gao side-input IS (modestly) used.** Permuting omega/tpf costs ~**0.012–0.016** AUC (was
  0), and side edges appear in the classifier's top symbolic edges. *(Earlier "0 importance" was a
  ceiling artifact.)*

**Open caveats / methodology fixes still needed:**
1. **Still near-ceiling** (baseline per-window AUC 0.992) → importances are *directionally* right
   but magnitudes are compressed; bump `--n-windows` (512–1024) for headroom. Report auto-flags it.
2. **τ-sweep is now SHALLOW** (0.85/0.87/0.86 at τ/2, τ, 2τ). Cause: random sampling drew
   heterogeneous operating points (this sample fs 37.7 kHz / tpf 1300 Hz) so a single *global*
   swept τ can't be optimal for every window → it blurs the peak. The head run's sharp peak @τ was
   a homogeneity artifact. **FIX: sweep each window relative to its OWN τ (×0.5/×1/×2).** Ablation
   is unaffected.
3. **Gradient attribution UNDER-credits the delay** — RNN saliency vanishes for early/lagged steps
   + conflates time/feature (Input-Cell Attention arXiv:1910.12370; TSR arXiv:2010.13924). Trust
   the ablation/τ-sweep (interventions) over the gradient. (NB: tooth-pass spectral energy was 25%
   on this run's window-0 vs 0.6% before — window/operating-point dependent; aggregate + use the
   chatter band.)
4. **Case `leans_on_delay` flipped between runs** (chatter→False, stable→True this time; opposite
   of the head run) → it's n=1 noise; need aggregate delay-engagement stats.

**TODO (priority order; verified sources in `related_work_scan_2026.md`):**
1. **Run remaining folds + un-blur the sweeps** (acts directly on the new caveats):
   - [DONE 2026-07-01, all 5 folds] 512-window runs REPLICATE on every fold (drops
     0.230/0.487/0.266/0.484/0.449; folds 1/3/4 → 0.500/0.510/0.541, at/near chance). Fold 3
     needed one resubmit (first job died silently). Note: 512 windows did NOT clear the 0.99
     ceiling (0.987–0.996) — magnitudes stay compressed, orderings trustworthy.
   - **Per-window-relative τ-sweep** (`physics.py` + runner): sweep each window by factors of
     *its own* τ (×0.5/×1/×2/∞), not one global τ — the global sweep blurs on a heterogeneous
     (multi-operating-point) sample; the head run's sharp peak was a homogeneity artifact.
2. **Add Temporal Saliency Rescaling (TSR)** to `xai/attribution.py` (arXiv:2010.13924) — expect
   the delay's *apparent* attribution to rise once the time/feature conflation is removed.
3. **Upgrade `xai/symbolic.py`**: adopt pykan's `r2_threshold` + simplicity gate (don't snap an
   edge with no faithful closed form); later, greedy end-to-end-loss snapping (arXiv:2603.15250)
   to fix the per-edge-isolation init-sensitivity.
4. **Improve `attribution_spectrum`**: target the chatter band (near `fn`) + tooth-pass sidebands
   and aggregate over windows (current single-bin-at-tpf measure is inconclusive by construction).
5. **SINDy-DDE cross-check** (arXiv:2407.19640): recover τ + a regenerative term directly from the
   cutting signals and compare to the TKAN's learned τ — an independent white-box validation.
   (pysindy has **no native delay** — build delayed columns by hand.)
6. **Baselines/ablations for the paper**: vanilla TKAN (arXiv:2405.07344, `tkan` lib) vs our
   delay-aware KAN-GRU; HAVOK (arXiv:1608.05306) as an independent delay-coordinate comparator;
   MLP-baseline channel/delay contrast.
7. **Aggregate case studies** (delay-engagement stats over many chatter vs stable windows, not n=1).
8. **Data robustness (longer-term)**: CEEMD-SVD denoiser + WGAN imbalance fix (Gao 2025); DSAN
   discrepancy loss for clean↔noisy shift (Zhao 2025). Closest precedent / related-work anchor:
   Rigas et al. KAN+symbolic+severity (arXiv:2412.01322).

### NEW METHODS QUEUE (added 2026-07-01) — STATUS AS OF 2026-07-20

**All five eval-only artifacts below have now been BUILT AND RUN.** Outcomes: SLD **done/win**
(98.2% label agreement, IoU_stable 0.941, boundary MAE_b 0.354 mm); physics-constrained
counterfactuals **done/win** (net-grounded, 81.6%); conformal **done/win** (ensemble, coverage
0.883 @ 0.90, selective acc 0.996); concept probes **done/NEGATIVE, claim retired** (R² −331 vs
+0.823 for proximity_to_blim); hidden-state dynamics **done/NULL** (Δ|λ| 0.001 ≪ std).
Of the "new research / model development" list, **only the severity-quantile head has been built**
(and not yet trained — see PENDING 1); learnable τ, MultKAN, prognosis, and sim-to-real are all
still untouched. Original text kept below for the rationale.

**New explainability artifacts (all eval-only on existing checkpoints, no retraining):**

- **Model-implied stability lobe diagram — TOP PICK, figure-rich, cheapest.** Sweep (omega, b)
  through a trained checkpoint at fixed modal parameters and plot the decision boundary as an
  SLD, overlaid on the analytic/simulated blim curve already in the metadata. A *global*
  explanation ("the network learned the lobes") a machining audience reads instantly, plus a
  quantitative global-faithfulness metric (boundary error vs the blim curve).
- **Physics-constrained counterfactuals.** Constrain the dice-ml search to actionable process
  axes (omega, b) — "what Δomega flips this window to stable?" — then validate the suggested
  direction against the true SLD (does it point at the nearest stable lobe?). Turns XAI into a
  process recommendation AND gives an objective correctness check on the explanations.
- **Hidden-state dynamical analysis.** Fixed-point / recurrent-Jacobian eigenvalue analysis
  (Sussillo & Barak style) of the KAN-GRU across stable vs chatter windows: does the model's
  internal dynamics cross |λ|≈1 where the physical system bifurcates? Chatter IS a
  delayed-system instability — a mechanistic result no attribution method can give.
- **Concept probes (TCAV-style).** Linear probes on the 64-d hidden state for named physics
  quantities: the regeneration phase between x(t) and x(t−τ), chatter-frequency vs tooth-pass
  separation, proximity to blim. Bridges latents → the Altintas–Budak picture
  (template: Huang et al. arXiv:2511.17894).
- **Calibrated uncertainty.** Split-conformal on the chatter head (or a cross-fold ensemble of
  the 5 checkpoints) → per-window calibrated confidence / abstention; doubles as the fix for
  the uncalibrated 0.5-threshold accuracy swings seen in the 121–125 held-out eval.

**New research / model development:**

- **Learnable τ — the strongest available science claim.** Differentiable delay kernel or
  neural-DDE variant; show the learned τ converges to the tooth-pass period from data. With the
  ablation (done) + the SINDy-DDE cross-check (TODO 5) that is THREE independent recoveries of
  the physical delay — a physics-discovery result, not just a classifier.
- **MultKAN trunk variant (KAN 2.0, arXiv:2408.10205).** Multiplication nodes natively express
  the multiplicative regenerative term b·Kt·(x(t)−x(t−τ)) that additive splines only
  approximate → symbolic extraction that *looks like* the physics, possibly a better model.
- **Severity head redesign — clearest remaining "model got better" win.** Quantile regression
  (pinball loss) rather than plain Huber: handles the heavy tail natively (MAE 0.88 in-dist but
  3.3–4.8 held-out, RMSE ≫ MAE every fold) AND yields prediction intervals that feed the
  conformal-uncertainty artifact above.
- **Chatter prognosis (next-paper scope).** Relabel windows by time-until-onset within each
  file → lead-time vs AUC curve; moves the work from detection (crowded) to prognosis (not).
  HAVOK forcing bursts (arXiv:1608.05306) are the natural comparator.
- **Sim-to-real (long-term arc).** DSAN-style discrepancy loss (Zhao 2025) scaled from
  clean↔noisy to sim→experiment on a public experimental chatter dataset — decides whether the
  paper is "a method on simulation" or "a method".
- **Process-damping physics task.** Re-derive blim with process damping to recover the 700
  dropped low-speed rows — the only labeled-data expansion NOT blocked on the data owner.

**Recommended order (paper-first):** verified TODO 1–4 above (replication hole + sweep/TSR/
symbolic fixes defend what exists) → SLD figure + physics-constrained counterfactuals (high
figure value, zero training) → learnable τ as the one new modeling experiment. Hidden-state
dynamics + prognosis are follow-on-paper scope.

**Commit policy reminder:** all white-box + research work is uncommitted in the working tree;
the user commits (their name only, no Claude trailer).

---

### Earlier — 5-fold training sweep (historical)
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

## INCORPORATE datasets 121–155 — CLOSED except the blocked replication arm (last touched 2026-06-22)
> **Status 2026-07-20:** steps 1–2 DONE (held-out eval PASSED, 0.908 / 0.916 — the headline result).
> Step 3 (126–155 clean replication) is **BLOCKED on the data owner**, not on us, and no code change
> can unblock it; step 4 is optional. Nothing here is actionable without new data from Schmitz.
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
3. [BLOCKED — data format, not code] **126–155 clean replication eval.** Cannot run: the newer Kaggle
   download ships 126–155 in a reduced `(N,4)` H5 layout (2 non-oscillating channels, zero-padded,
   omega/b as leading columns) — incompatible with the standard 17-col / 4-channel pipeline. (121–125
   are standard `(45120,17)`, so their done eval is sound.) **WHAT TO DO LATER:** request 126–155 from
   the data owner (Schmitz) in the standard **17-column** format that 1–125 use; then
   `build_features_121_155.py` (blim fix validated) → extract NPZ → `eval_heldout_phase3.py` run
   unchanged. Until then, the held-out result is complete on the noisy/novel arms (0.908 / 0.916).
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
- 2026-06-22 (cont.) — **Wrote Research Update 7** (`summer_research/Research Updates/research_update_7.md` + rendered `.docx` w/ embedded figure; ~20h, Jun 15–22) via 4 parallel subagents (leakage-fix brief, metadata-gen brief, held-out figure, GL replication-eval prep). New asset `results/figures/research_update_assets/heldout_generalization_121_155.png` + `scripts/plot_heldout_generalization.py`. These + the update are UNCOMMITTED (summer_research is outside the tracked set).
- 2026-06-22 (cont.) — **blim generator bug found + fixed (systematic-debugging, TDD).** Running `build_features_121_155.py --validate-against` 1–60 on GL **FAILED**: 934 blim + 934 b_over_blim + 36 label mismatches (e.g. ds29 gen=+1.99 vs csv=**−15.17**; ds41 in-range 4.34×). Root cause: the original feature CSV built blim by **sorting the boundary by omega + linear extrapolation** (low-speed omega<grid → negative limit = the 700-row artifact), but `derive_blim_mm` did **unsorted `np.interp` (clamp)**. Confirmed against `verify_b_over_blim_provenance.py` (which sorts → matched in-range; flagged low-speed as out-of-range = extrapolated). Fix: `derive_blim_mm` now filters-finite → `argsort` by omega → `_interp_extrap` (linear interp + 2-point linear extrapolation, matches `scipy interp1d(fill_value="extrapolate")`); ×1000 m→mm kept. Updated the test that baked in clamp behavior + added an unsorted-boundary test. **Local full suite 410 passed / 1 skipped; GL re-validation = VALIDATION PASS (18669 rows, 14 cols).** Generator now safe for 126–155. Uncommitted.
- 2026-06-22 (cont.) — **126–155 clean replication eval BLOCKED on a data-format change (NOT code).** After the blim fix + PASS, generating the 126–155 table failed (`read_file.load_time_h5: Expected 17 columns, got shape (N,4)`). GL inspection: **121–125 (noisy) are standard `(45120,17)`** (so their 0.908/0.916 eval is sound), but **126–155 (clean) are `(376004,4)`** — ~8× length, zero-padded, omega/b as leading cols (`omega n_unique=2 {0,2000}`), and only **2 non-oscillating signal channels** (col2 0–15, col3 0–62, both ≥0). The model needs 4 oscillating channels (Forcex/Forcey/xtpos/ytpos) → the new clean files cannot build the model input. Not a reader bug — missing channels. **DECISION (user): ship the held-out story on 121–125 (0.908) + OOD 122–125 (0.916); defer 126–155.** **WHAT TO DO LATER:** request 126–155 from the data owner (Schmitz) in the standard **17-column** format that 1–125 use → then `build_features_121_155.py` (blim fix validated) → extract NPZ → `eval_heldout_phase3.py` run unchanged for the replication arm. Everything built today (blim fix, generator, leakage fix, parent_groups, eval harness) and Research Update 7's numbers are unaffected. Updated Update 7 §7/§8 + heldout doc to mark the deferral.
- 2026-07-01 — **Fold replication + Research Update 8.** Submitted all 5 white-box jobs (`slurm/run_whitebox_all_folds.sh`, 512 windows; fold 2 re-run as `fold2_w512` to keep the 256-window artifacts). 4/5 returned and replicate the delay story (drops 0.23–0.49; fold 1 → chance 0.500); **fold 3 produced an empty dir — check `logs/phase3_whitebox_*.err` on GL and resubmit** (`env RUN_ID=3a40b71514cd41b98916330c000d700a TAG=fold3 VAL_AUC=0.9605 N_WINDOWS=512 sbatch --export=ALL slurm/phase3_whitebox.sbatch`). New `scripts/plot_whitebox_replication.py` + softened the τ-sweep panel title in `plot_whitebox_summary.py` (peak location doesn't replicate). Drafted + rendered Update 8 (fold-3 row = the only placeholder). All uncommitted per policy. Aborted a multi-agent Workflow draft attempt — user prefers inline drafting for updates.
- 2026-07-01 (cont.) — **Fold 3 returned; Update 8 FINAL.** Resubmitted fold-3 job completed: AUC 0.9932→0.5098 (drop 0.484, ≈chance), MAE 0.233→3.06, x_delay 9.8%, rand cosine −0.037, 25/25 nonlinear — replication now 5/5 with 3 folds at/near chance under delay ablation (mean drop 0.38); τ-sweep again peaked at 2τ. Filled the Update-8 table, regenerated `whitebox_delay_replication.png` (5 folds), updated every four-of-five statement, re-rendered the docx (390 KB). Also added `PYTHONUNBUFFERED=1` to `phase3_whitebox.sbatch` (logs now stream). Update 8 is ready to send. Uncommitted per policy.
- 2026-06-22 (cont.) — **126–155 conversion script RULED OUT (data lacks the channels).** Followed `pri2m-eda.ipynb` to confirm the 17-col layout: cols = `[omega, b, time, xtpos, ytpos, xwpos, ywpos, xtvel, ytvel, xwvel, ywvel, xtacc, ytacc, xwacc, ywacc, Forcex, Forcey]`; model uses xtpos(3)/ytpos(4)/Forcex(15)/Forcey(16). The new clean files are `[omega, b, time, ONE non-negative channel]` per operating point — col3 smoothly decays 2.52→0, max 62.4, **never negative** across all 376k samples = an amplitude/envelope, NOT the 4 raw signed channels. 261 files/dataset (vs ~102). **You cannot reconstruct 4 signed oscillating waveforms from 1 rectified envelope — the info is gone, so no converter is possible.** ONLY unblock = obtain 126–155 from the data owner in the original 17-column (or ≥4-channel signed) format; then the existing pipeline runs unchanged. Replication arm stays deferred.
- 2026-07-04 — **Resumed the paused severity-quantile head + drafted Research Update 9.** (1) **Resume:** the Update-9 "new methods" suite had been built test-first (per-window `physics.tau_sweep_relative`, `attribution.temporal_saliency_rescaling` TSR, `symbolic.fit_curve(r2_threshold=…)` gate, `analysis.sindy_dde`, `xai.{stability_lobe,counterfactuals,concept_probes,conformal,dynamics}` + one sbatch wrapper each), but the interruption left `test_severity_quantile.py` with NO implementation → `ImportError: pinball_loss` blocked the **whole suite from collecting**. Implemented it minimally + TDD: `pinball_loss` in `training/losses.py`, `severity_loss` quantile routing (parts `sev_pinball`/`sev_bce`), `SeverityHead(severity_mode="huber"|"quantile", quantiles=…)` (median-column point est, R²-safe, default byte-identical), and `TKANModel(severity_mode=…, severity_quantiles=…)` wiring. **Full suite 469→560 passed / 1 skip** (target file 15/15). Default (huber) path unchanged so nothing already validated moves; quantile retrain to *measure* the heavy-tail win stays deferred (needs GL). (2) **Local first results:** `sindy_dde_crosscheck.py --synthetic` recovers a planted delay to 3% (median τ*/τ_phys **1.035**, regen term selected 100%) → `results/figures/research_update_assets/sindy_dde_tau_recovery.png` (SYNTHETIC validation, not real signals); counterfactuals had already run locally on a **table-label RandomForest surrogate** (checkpoint=null): 42/42 flips toward stable, 40/42 in the physically-stable region, **0.952** agreement (`results/phase3/xai/counterfactuals/`). Both are machinery validations — the real-checkpoint/real-signal numbers are queued on GL. (3) **Update 9 DRAFTED + rendered** (`summer_research/Research Updates/research_update_9.md` + `.docx`, 2 embedded figures, Jun 29–Jul 4 / ~20 h). It is a build-the-toolkit update: **10 numbered `[PLACEHOLDER: …]` tags** mark every number owed by a queued GL run (per-window τ-sweep peak, real-signal SINDy-DDE τ, SLD faithfulness, TSR share, aggregate delay-engagement, symbolic-gate count, real-network counterfactuals, MLP baseline table). Note: user shifted Update 8's time-table to end Jun 28 (docx prose header still reads "July 1" — user's file, untouched). (4) **Sync script updated** (`sync_phase3_tkan_to_hpc.sh`): added `src/pri2m/analysis/` mirror + the 10 new scripts + 8 new slurm wrappers + 10 new test names + the `results/figures/research_update_assets` mkdir (the old cherry-pick list would have MISSED `analysis/sindy_dde.py` and every new runner). `bash -n` clean. All uncommitted per policy. **NEXT (user, GL):** `./sync_phase3_tkan_to_hpc.sh` → re-run `run_whitebox_all_folds.sh` (per-window sweep + TSR + symbolic gate) + submit `phase3_{sindy_dde,stability_lobe,counterfactuals(with CHECKPOINT),concept_probes,conformal,hidden_dynamics}.sbatch` → rsync results back → fill the 10 placeholders for the final Update 9.
- 2026-07-04 (cont.) — **GL runs came back; Update 9 REWRITTEN around real results (0 placeholders).** User ran the eval-only analyses on GL and pulled results local. Verified from the manifests: **(1) per-window τ-sweep (`run_whitebox_all_folds.sh` re-run, 5 folds) PEAKS AT ×1.0 ON EVERY FOLD** (AUC ×1 = 0.986–0.996; ½× = 0.83–0.87; 2× = 0.93–0.96; no-delay floor 0.50–0.77) — un-blurs Update 8's global ×2 artifact; ablation reproduced exactly (drops 0.23–0.49, x_delay 9.7–10.7%). **(2) symbolic R²-gate: 25/25 accepted every fold, R² 0.926–0.9999.** **(3) stability lobe diagram (`fold2_ds29grp/sld`, real checkpoint, 1014 cells, ω 2000–30000): label agreement 98.2%, IoU_stable 0.941, boundary MAE_b 0.354 mm (11.2%), 0 censored** (per-cell AUC ceilinged 0.9987 → use MAE_b). **(4) SINDy-DDE on REAL fold-2 signals: median τ*/τ_phys 0.986, regen term selected 100% of 64 windows, but wide per-window IQR [0.55,1.29] → bracketed recovery** (synthetic self-test earlier = 1.035). **(5) conformal (`conformal/fold2`, single fold): coverage 0.914 @ target 0.90, well-calibrated sweep, selective acc 0.987.** **(6) matched MLP table** (local run artifacts, `--exclude=*.pt` pull): TKAN 0.955±0.022 vs MLP 0.875±0.004, +0.080 every fold. **NOT landed (reported as such in §7):** TSR built+tested but the whitebox runner NEVER calls `temporal_saliency_rescaling` (still owed, needs wiring + re-run); counterfactuals `fold2_net` STILL a table-label RandomForest surrogate (checkpoint recorded via sha256 but `--checkpoint` alone does NOT generate model predictions → identical 0.952; needs a `--predictions` file from running the net over the windows); concept probes 1/3 (proximity_to_blim R² 0.82 ✓; regeneration_phase R² **−228** degenerate, tpf_separation R² −0.82 ✗); hidden dynamics null (chatter |λ| 1.0153 vs stable 1.0142, Δ0.001 ≪ std, "supported:True" oversells it). Generated cross-fold `results/figures/research_update_assets/whitebox_tau_sweep_relative.png` from the manifests. `research_update_9.{md,docx}` now has **3 embedded figures** (τ-sweep, SLD, SINDy real) + MLP/conformal tables, **0 placeholders**, arc = "which delay (×1 all folds) + independent recovery (SINDy 0.99×) + what physics (SLD 98.2%)." All uncommitted. **Remaining to fully close Update 9's queue:** wire TSR + re-run; real-net counterfactuals via predictions file; conformal ensemble; debug concept probes.
- 2026-07-17 — **Resumed the cut-off 2026-07-16 session + VERIFIED the fuzz/atlas build.** The interruption hit after the plan's final gate checkbox but before this doc was updated; nothing was missing on disk (all 10 new source/script/slurm files present, timestamps Jul 16 22:29–23:25). Fresh verification: full suite **585 passed / 1 skipped, 0 failures in 3m12s** (baseline 560/1 → +25; new files: sampling 2, degradation 9, spline_atlas 6, predict_windows 2, circularity 3, fuzz-CLI 1, replication-plot 1); `bash -n` clean on `sync_phase3_tkan_to_hpc.sh` + all 4 new slurm files; sync script carries all 7 new test stems + new script/slurm entries; wiring spot-checked in source (whitebox runner imports `spline_atlas` and calls `attribution.temporal_saliency_rescaling` → `results["tsr"]`; `counterfactuals_phase3.py --predictions` fits the surrogate on real-net outputs; counterfactuals sbatch documents the 2-step predict→CF flow). Benign wart noted, not changed: `degradation.py:107` emits a requires_grad→scalar UserWarning during spline-coarsen refit (cosmetic). All uncommitted per policy. **NEXT (user, GL) — the plan's "After the build" list:** `./sync_phase3_tkan_to_hpc.sh` → `bash slurm/run_fuzz_all_folds.sh` + re-run `bash slurm/run_whitebox_all_folds.sh` (atlas+TSR now included) → predict-windows sbatch then counterfactuals 2-step → `bash slurm/run_conformal_ensemble.sh` + concept-probes re-run → rsync results local → `python scripts/plot_fuzz_replication.py --manifests results/phase3/xai/fuzz/fold*/fuzz_manifest.json --labels fold0 fold1 fold2 fold3 fold4 --out results/figures/research_update_assets/fuzz_replication.png` → Research Update 10.
- 2026-07-20 — **PROGRESS.md brought current (it was 2 updates stale) + pending list re-derived from disk, not docs.** The 2026-07-17 entry above ended with a "NEXT (user, GL)" list; that list was in fact **executed the same day** and this doc never recorded it. Verified from the working tree: `results/phase3/xai/fuzz/fold{0..4}` (Jul 17), re-run `fold{0,1,3,4}` + `fold2_w512` whitebox dirs (Jul 17 — `tsr` and 25/25 `spline_atlas` present in all five manifests; the stale `fold2/` dir predates the wiring and must not be used), `counterfactuals/` net-grounded (Jul 17), `conformal/ensemble` (Jul 17), `concept_probes/fold2` (Jul 17); and **Research Updates 10 and 11 both drafted + rendered Jul 18**. Added six Status-board rows for all of the above with manifest-verified numbers, rewrote NEXT ACTION around the real frontier, demoted the 2026-06-27 frontier to "Superseded", and annotated the NEW METHODS QUEUE with per-item outcomes. **Two long-standing "open" items re-confirmed open by direct check rather than memory:** (1) the severity **quantile head has never been trained** — no `configs/*.yaml` sets `severity_mode` and all 25 run dirs are huber, so the 2026-07-04 build is still unmeasured; (2) the **`--no-side-input` ablation has never run** — every TKAN run dir carries `side: True`, and the only `side: False` run (`93ea5f598f…`) is the pre-bundle lr-0.003 fold 0. Conclusion recorded in NEXT ACTION: eval-only XAI is **saturated**; every remaining high-value item is a GL retrain. Nothing run, nothing committed.
- 2026-07-20 (cont.) — **Quantile-severity wiring + SLURM passthroughs (2 parallel subagents, TDD, uncommitted).** Recon first changed the plan: two of the three "pending" items turned out to need almost no code. `--no-side-input` has existed on `train_phase3.py` since the tuning bundle, and `--tsr-windows`/`--tsr-steps` have existed on `whitebox_phase3.py` since the 07-16 build — in both cases the **sbatch wrapper was the only thing missing**, which is why the side-input ablation sat "blocked" for weeks and why every landed whitebox run analyzed 4 TSR windows. Agent A (python, TDD): `severity_mode`/`severity_quantiles` on `TrainConfig` → `build_model` → `TKANModel`; `ValueError` guards for `mlp`+quantile and unknown modes, mirroring the `side_input` guard and importing `_VALID_SEVERITY_MODES` from `heads.py` rather than redefining it; `--severity-mode` with CLI>YAML>default precedence; documented yaml keys defaulting to the validated `huber`. Also caught and fixed a real blocker the brief only suspected: a quantile head is wider, so `xai_phase3.py::_load_model` failed to reload one (`size mismatch for severity.reg_head.base_weight: [3,8] vs [1,8]`) — it now infers Q>1 from the state_dict and reads levels from a new `severity` block in `preprocessing.json`, ABSENT in default mode exactly like the `side` block. No loss-side change needed: `severity_loss` already routes off the model's output dict. Agent B (shell): generic `EXTRA_ARGS` on `phase3_train_gpu.sbatch` using `read -r -a` — NOT unquoted `$EXTRA_ARGS`, which also glob-expands (measured: `"--tag-glob *"` → 23 argv entries instead of 2, i.e. a silently corrupted 8-hour allocation); `TSR_WINDOWS`/`TSR_STEPS` on `phase3_whitebox.sbatch` + `run_whitebox_all_folds.sh`. Both wrappers verified argv-identical with the new vars unset (agent's first "identical" check was bogus — local bash 3.2 killed both runs at a pre-existing `set -u` empty-array line and it diffed two empty files; it caught this itself via an ARGC check and redid it). **Driver verification, not taken on trust: full suite re-run independently = 603 passed / 1 skipped in 3m14s** (585 + 18), `bash -n` clean on all 3 shell files, trainer/config/sbatch diffs read line by line. **New finding, filed as PENDING 6:** `"huber"` is a misnomer — `_VALID_SEVERITY_MODES` names it, but `losses.py` has no Huber/smooth-L1 at all; the point path is `weighted_mse` with loss part `sev_mse`. Predates this session (came in with the 07-04 build); fix the name or document it before it reaches a paper's methods section. Net effect: **all three actionable pending items are now submit-ready**, commands recorded in NEXT ACTION. Nothing committed.
