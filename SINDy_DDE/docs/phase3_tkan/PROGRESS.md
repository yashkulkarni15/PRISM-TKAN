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
  - **WIN — spline-edge atlas.** 25/25 edges carry an accepted formula on all 5 folds (**R² 0.9257–0.9999** — corrected 2026-07-29; the range previously stated here, "0.985–1.000", was wrong: measured over all 125 stored rows, **11 are below 0.98 and 4 below 0.95**, worst fold 4 at 0.9257 and fold 0 at 0.9312. Per-fold minima: 0.9312 / 0.9815 / 0.9768 / 0.9640 / 0.9257. All 125 still clear the `accepted` gate, but the fit floor is materially lower than the old line implied — do not quote 0.985 in the report). τ-delayed edges carry 15–16% of top-edge importance on folds 0/1/2 but only 3–4% on folds 3/4 — the same fold split the ablation shows. The wrong range also reached `research_update_11.md` (§3 atlas and §8 summary); **both were patched and the `.docx` re-rendered on 2026-07-29** (figures verified re-embedded; the md carries a dated correction note in its provenance comment). The copy delivered to Prof. Abell before that date still shows the old range.
  - **WIN — conformal ensemble** (5 checkpoints, `conformal/ensemble`): coverage **0.883** @ target 0.90, selective accuracy **0.996**, raw accuracy 0.984, abstention 0 at α=0.1; sweep well-behaved α=0.2→0.01.
  - **NEGATIVE (reported, not buried) — TSR does not find the delay.** Underpowered at **n_windows=4 / steps=8** per fold; channel shares nearly identical to plain IG (position 72–83% under both); `tr_peak_lag_mean` = 2.0/2.0/33.0/20.5/2.0 steps against `tau_raw_mean` 87 raw ≈ 22 pooled steps → only one fold near the delay. **Owed: re-run at hundreds of windows before this null is trustworthy.**
  - **NEGATIVE — concept probe still fails after the circularity fix** (`concept_probes/fold2`): held-out R² regeneration_phase_sin **−331.0**, _cos −5.83, chatter_tpf_separation −3.50, vs proximity_to_blim **+0.823**. De-circularized and still failing = the concept is absent from the 64-d state, not the target leaking. Claim retired.
  - **NULL — hidden-state dynamics** (from Update 9, unchanged): chatter |λ| 1.0153 vs stable 1.0142, Δ0.001 ≪ std; the manifest's `supported: True` oversells it.
- [x] **Research Update 10 DRAFTED + rendered (2026-07-18)** — `research_update_10.{md,docx}`, covers Jul 5–11 (~20 h): the graded stress-test suite and what survives it. 2 figures from the fuzz manifests.
- [x] **Quantile-severity wiring + SLURM passthroughs DONE (2026-07-20, local, TDD, 2 parallel subagents)** — closes the "built but not trainable" gap and unblocks BOTH pending GL retrains. (A) `severity_mode`/`severity_quantiles` now flow `TrainConfig` → `build_model` → `TKANModel`, with `ValueError` guards for `mlp`+quantile and unknown modes (mirrors the `side_input` guard); `--severity-mode` on the CLI with the same CLI>YAML>default precedence as `--side-input`; `severity_mode: huber` + `severity_quantiles: [0.1,0.5,0.9]` documented in `configs/phase3_tkan.yaml`. **Checkpoint round-trip fixed too:** a quantile head is *wider*, so `scripts/xai_phase3.py::_load_model` could not reload one (`size mismatch ... [3,8] vs [1,8]`); it now detects Q>1 from the state_dict and reads levels from a new `severity` block in `preprocessing.json` (block ABSENT in default mode, exactly like the `side` block). No loss-side changes were needed — `severity_loss` already routes off the model's output dict. (B) `slurm/phase3_train_gpu.sbatch` gained a generic `EXTRA_ARGS` passthrough (`read -r -a` split — deliberately NOT unquoted `$EXTRA_ARGS`, which also glob-expands: `"--tag-glob *"` measured 23 argv entries instead of 2); `slurm/phase3_whitebox.sbatch` + `run_whitebox_all_folds.sh` gained `TSR_WINDOWS`/`TSR_STEPS`. Both wrappers verified argv-identical when the new vars are unset. **Full suite 585 → 603 passed / 1 skipped, 0 failures** (+18 tests), re-verified independently by the driver; `bash -n` clean on all 3 shell files. Uncommitted per policy.
- [x] **Research Update 11 DRAFTED + rendered (2026-07-18)** — `research_update_11.{md,docx}`, covers Jul 12–18 (~20 h): atlas + TSR wiring + net-grounded counterfactuals + conformal ensemble + the two retired claims. 0 placeholders; every number traced to a returned manifest.
- [x] **TSR AT SCALE RAN — 160 shards, all 5 folds (2026-07-23/24, GL arrays 54522626/29/702 + resubmits 54636665/667/669/671/673 + 5 merges)** — closes pending item 3 below, but **not with either answer it predicted**. Infrastructure was flawless: 160/160 shards, 128 unique windows per fold, exactly 64/64 chatter/stable, and a local re-merge that came out **byte-identical** to GL's. 27 shards timed out at the old 6 h wall and were refilled idempotently by `resubmit_missing_tsr_shards.sh` (disk is the source of truth, not `sacct`); the 24 h wall was the fix — one refilled shard ran 6 h 40 m. **The finding is a metric defect, not a null.** `tr_peak_lag` (`scripts/whitebox_phase3.py:222`) is `argmax` of the relevance autocorrelation over lags `[2, N/2]`; a monotonically-decaying autocorrelation returns index 0, i.e. the search floor. **550/640 windows (85.9%) emit exactly 2**, so Update 11's fold means (2.6–10.0) are arithmetic over a "no answer" sentinel. Among the **90 windows that resolve a real peak**, the peak locks to that window's own τ far beyond chance: exact `peak_lag == tau_raw` 31/90 (34%) vs null 3.34, **p = 0.00005**; on an exact harmonic 60/90 (67%) vs null 17.9, **p = 0.00005**; median ratio **1.0000**. Also: `peak_lag` and `tau_raw` are empirically in the **same units**, so U11's "~22 steps after four-fold pooling" comparison was on the wrong axis. **⚠ Not claimable yet** — the delayed channel is built by shifting the signal by τ, so the lock may be a construction artifact; the untrained-checkpoint / wrong-τ control (U12-B) gates it. **Update 11 §3 must be corrected, not superseded** — its stated reason ("underpowered, only four windows") is the wrong diagnosis.

- [x] **U12-A: TSR delay metric FIXED (2026-07-25, local, TDD).** New `tr_delay_statistics()` in `scripts/whitebox_phase3.py` adds `tr_peak_at_floor` (states outright when the argmax was pinned to its search floor) and **`tr_lock_z`** — the argmax-free statistic: the relevance autocorrelation *at lag τ* scored against the spread over every other searched lag, so there is no boundary to pin against and the physical delay is tested directly. `None`, never a guess, when τ is untestable. `z > 2` is the pre-committed decision rule. `tsr_aggregate` gained `n_resolved` / `resolved_fraction` / `tr_peak_lag_mean_resolved` / `lock_z_mean` / `lock_z_frac_above_2` / `n_lock_tested`, keeps every back-compat key, and tolerates pre-fix rows. **Also fixed a vacuous renderer metric:** `tsr_delay_share_gain` was `mean(share_tsr − share_ig)` over channels, which is **identically zero for any two share vectors** — now the mean *absolute* shift (0.03365 on fold2 instead of 0). Report template gained the resolved fraction, the lock statistic, and the construction-artifact caveat. 13 tests in `tests/test_whitebox_tsr_delay_lock.py`. Recomputed on the real run: resolved fractions 4.7/12.5/22.7/25.0/5.5% with resolved-only means 22.67/60.94/37.52/23.41/12.71 vs the old 2.97/9.37/10.05/7.35/2.59. ⚠️ `tr_lock_z` is NOT recomputable from returned manifests (needs `time_relevance`) — it requires the U12-B rerun.
- [x] **U12-C: SDD Tasks 7 + 8 COMPLETE (2026-07-25) — the tree is GREEN for the first time since the 2026-07-20 pause.** `run_band_notch_contrast` landed with **bidirectional** energy matching (user decision): both halfwidths are searched, so the achievable set is the product of two staircases rather than one. **Plan bug #3 found, both halves:** (a) the brief averaged per-window halfwidths and applied the mean — wrong, since removed energy is a step function of halfwidth, measured ~2× off; (b) the brief's 1% tolerance is **unachievable by the one-sided design on any fixture** (7.65% at L=1024 multi-tone, 4.15% on pure white noise). Added a hazard guard the chosen option would otherwise have introduced: the chatter band may never reach a tooth harmonic, or it deletes both conditions' evidence. Result — matches at the real window length (L=1024: `max_rel_err` 0.0043, `matched=True`) and honestly refuses at coarse resolution. `energy_match` now carries `max_rel_err` + `matched`. Wired as `manifest["contrasts"]["band_notch"]`, a **sibling** of `fuzzers`, sourcing `fn` from `batch["fn"]` (verified at `phase3_sequence.py:355` — the plan's "reload the table" was false). Strengthened `test_fuzz_cli_end_to_end`, which asserted only `errors == {}` and so would not have caught the contrast degrading to its `fn unavailable` fallback (a returned dict, not an exception). **Suite 681 passed / 1 skipped / 0 failed.** Remaining for this build: the final whole-branch review, then the GL run.

- [x] **U12-B: TSR construction-artifact control BUILT (2026-07-25, local, TDD) — U12 is code-complete; only the GL run is outstanding.** `scripts/tsr_control_phase3.py` + `slurm/phase3_tsr_control.sbatch` + 11 tests. Three conditions on the same windows: trained/true-τ, **untrained/true-τ (decisive)**, trained/shuffled-τ. The untrained net is a deepcopy of the *loaded* model with every submodule's `reset_parameters()` called — not a re-construction from guessed ctor args, which could drift from the trained architecture and quietly invalidate the control; two guards refuse to run if no submodule exposes `reset_parameters()` or if the weights did not actually change. `tsr_one_window` gained `tau_override` and `feature_masking`. **Cost:** at sweep settings this would take days; it doesn't need them. `tr_lock_z` derives from step 1's `time_relevance`, and `feature_masking` only selects how step 2 scores the *feature* dimension, so the lag statistics are **bit-identical** with it off — verified empirically, not read off the docstring. That plus "attribute, don't re-measure" brings it to ~24 windows on one fold, 12 h wall. **Verdict rule pre-committed:** `untrained_lock >= 0.8 × |trained_lock|` ⇒ artifact; a missing lock on either side returns `None` = UNDECIDED, which is explicitly **not** a pass. **Suite 684 passed / 1 skipped / 0 failed** before this landed.
- [x] **Sync-script gaps found and fixed (2026-07-25).** Three, one pre-existing: `tests/test_xai_spectral.py` was **never** in the sync list (since the Tasks 1–3 landing on 2026-07-20), so the spectral module's tests could never run on GL; **`docs/` was not synced at all**, yet `render_whitebox_report.py` reads `docs/phase3_tkan/whitebox_analysis.md` as its default template — GL held a renderer it could not run. Added the template + its `mkdir`, plus `test_whitebox_tsr_delay_lock`, `test_tsr_control`, `scripts/tsr_control_phase3.py`, `slurm/phase3_tsr_control.sbatch`. `bash -n` clean.
- [x] **BOTH ready GL retrains RETURNED (2026-07-21 + 2026-07-26; pulled local 2026-07-26).** The two items open since 2026-06-14 are now measured, both against the validated side-on sweep (0.9597/0.9122/0.9752/0.9605/0.9695, mean 0.9554). **(1) `--no-side-input` 5-fold ablation** (`results/phase3/runs`, `side_input: false`): 0.9650/0.9080/0.9747/0.9547/0.9632, **mean 0.9531 → the Gao process-parameter side inputs are worth ≈ +0.002 AUC**, i.e. they are not what puts TKAN ahead of the MLP. **(2) Quantile severity head** (`results/phase3/runs_quantile`, `severity_mode: quantile`): 0.9629/0.9226/0.9796/0.9631/0.9799 — **chatter AUC does not regress** (if anything +0.005), but the severity heavy tail is essentially unmoved (fold-4 MAE 1.44–1.62 vs huber 2.13; fold-1 *worse* at 1.12–1.18 vs 1.04). The one remaining "the model got better" lever did not move the weak head. **DUPLICATE SWEEPS: both were submitted a second time on the night of 2026-07-25 having already completed, so `runs/` and `runs_quantile/` now hold TEN run dirs where five belong** (two per fold, three ablation folds identical to 4 dp, one quantile fold bit-identical). Nothing in the path says which is canonical — **resolve before anything globs those trees or the zoo is republished.**
- [x] **U13-A + U13-B WIRED AND RAN (2026-07-26 local wiring, TDD; GL job 55405362, 2026-07-29).** Both library halves existed and neither had ever been called by a runner, so plan criterion 7 was answerable by nothing on disk. `monte_carlo_survival` wired into `counterfactuals_phase3.py` (`--mc-draws/--mc-seed/--mc-margin`, on by default, pooled across datasets by concatenating per-case survivals so a 3-case dataset cannot outweigh a 40-case one); DiCE wired behind `--dice` (off by default). `build_surrogate` now returns the fitted estimator too — DiCE takes the sklearn object, and it must be the *same fit* the physics search used. **Results on the real 42 cases (fold2_net, dataset 1, `preds_fold2.csv`): survival ≥ 0.85 in 19/42, mean 0.702, median 0.685, min 0.086 @ 512 draws** — criterion ("≥10 unstable points above 0.85") **MET**, but **less than half of the recommendations survive** 2% spindle jitter + 0.05 mm depth tolerance + 10% blim uncertainty. **DiCE found CFs for 42/42, 36 (85.7%) physically stable** vs the physics search's 81.6%. Shipped agreement number **unchanged at 0.8158** in the real run, as the additive-only tests required.
- [x] **U13-B comparison metric REWORKED after its first result was mislabelled (2026-07-29, local, TDD).** The first cut compared the *direction* each generator moved depth and reported 12/38 "agreement" — but `physics_constrained_counterfactual` caps `b` at `b_base` by construction, so it can only hold or lower depth while DiCE varies both axes freely; the metric was scoring "chose the other knob" as "disagrees". Replaced with an **axis breakdown** plus **outcome agreement** (both land stable, or neither). Recomputed on the returned manifest: **outcome agreement 27/42 = 0.643, both-stable 26**; axis — physics moves **omega in 35/42**, DiCE touches **depth in 36/42** (17 depth-only + 19 both). The two generators pick near-mirror-image strategies and corroborate each other on where the move lands. `n_agrees_with_physics_cf` deleted, with a regression test asserting it stays gone; `axis_note` in the manifest states outright that the axis fields are not an agreement measure.
- [x] **U13-C REGENERATIVE SPLINE TEST RAN, 5/5 FOLDS — AND REFUTES ITS OWN HYPOTHESIS (2026-07-29, GL jobs 55408256–60).** `score` is **+1 for an exact negation, −1 for an exact copy**. Every fold is strongly **negative**: mean −0.533/−0.639/−0.526/−0.574/−0.457 (importance-weighted −0.54…−0.71), and the top-importance pairs are near-exact copies (`kan_r out=55`, score −0.989, **corr +0.997**, gain 1.14). Under the ±0.9 thresholds **23–36% of pairs qualify as summing and 0.1–1.0% as negating** — a ~30:1 split the wrong way. **The network learned x(t) + x(t−τ), a SUM, not the regenerative difference x(t) − x(t−τ). The plan's qualitative criterion is NOT met**, on the most direct test available for it. It is a real learned structure, not an artifact: **untrained null mean −0.023** (flat across all five folds), cross-channel control −0.08…−0.18, signal 20–25× the null and 3–6× the control, `frac_base_dominated ≈ 0` so it lives in the learned splines rather than the SiLU base. Physically a comb filter that *reinforces* tooth-pass-synchronous content where the difference would notch it out — still delay-coupled (it needs τ), but not the regenerative term. **Caveat: the effect does NOT track the ablation fold split** (folds 0/1/2 carry 5 delay edges in the top 25 and 3/4 carry 1, yet fold 3 at −0.574 beats folds 0 and 2) — the built-in control does not corroborate, so do not lean on it.
- [x] **TSR-control survivability fixes (2026-07-29, local, TDD).** Job 55098053 burned its full 12 h wall (2026-07-26) having printed nothing since the header and written nothing at all. `run_tsr_control` gained optional `progress(condition, done, total)` and `on_condition(name, result_so_far)` hooks (both default None ⇒ old behaviour byte-identical); the runner prints an up-front `scoring 72 window-evaluations (3 conditions x 24 windows)` — **the number the original 12 h estimate missed, since the three conditions share the same 24 windows** — then a per-window counter, and writes the manifest after **every** condition. The snapshot computes the verdict as soon as the **decisive pair** is in, which makes the condition ordering load-bearing (trained → untrained → shuffled) and is pinned by its own test: a kill inside condition 3 still leaves a decidable verdict. `_manifest()` stamps `provenance.complete` so a partial file can never be read as finished.
- [x] **U12-B: TSR CONSTRUCTION-ARTIFACT CONTROL RETURNED — the delay-lock is NOT attributable to training (GL job 55408255, launched 2026-07-29; fetched local 2026-07-30, `results/phase3/xai/tsr_control/fold2_w512/tsr_control_manifest.json`, 53 KB).** **Completion verified, not assumed:** `provenance.complete: true`, and that flag is written only at `tsr_control_phase3.py:244` **after** `run_tsr_control` returns all three conditions — the per-condition checkpoints write `complete: false`, so a wall-killed partial cannot present as finished. All three conditions present at 24/24 windows, `n_lock_tested` 24 everywhere, window indices **identical across conditions** (so every comparison below is paired on the same windows), untrained arm genuine (`n_submodules_reinitialised: 7`, `mean_weight_diff: 0.426`). Only provenance gap: `git_sha: null` — the `_provenance()` fix survives the missing root instead of crashing, but records nothing. **The result:** trained/true-τ `lock_z_mean` **+2.81** (79.2% of windows > 2, 8/24 resolved) · untrained/true-τ **+4.62** (**100%** > 2, 0/24 resolved) · trained/shuffled-τ **+0.88** (16.7% > 2). **The decisive pair goes the wrong way for the model: the randomly-initialised network locks HARDER than the trained one, in 22/24 windows, paired mean +1.80, sign test p = 3.6e-5** — and it clears the `z > 2` significance gate in 24/24 windows, so this is not a trained result being compared against noise. The shuffled arm does not rescue it: trained-true beats trained-shuffled 19/24 (p = 6.6e-3; shuffled falls to 0.70 / 9.1% > 2 after excluding the **2 windows where the shuffle returned the window's own τ** — the shuffle is not a derangement, a real but harmless defect since it biases *toward* the shuffled arm and it collapsed anyway), but τ-specificity is exactly what the artifact predicts, because the delayed channel *is* the signal shifted by that window's own τ. Two corroborating details: the trained lock is **higher on stable windows (3.52) than chatter (2.10)**, the opposite of a learned regenerative mechanism; and `lock_z` falls monotonically with τ in both arms (corr(z, τ) = −0.64 trained / −0.43 untrained; τ∈[20,250], 511 searched lags), the signature of the documented autocorrelation-decay bias rather than physics. **VERDICT: TSR delay-lock is a construction artifact and is permanently unclaimable as evidence about the network.** The 2026-07-23/24 sweep's 34%-exact / p = 0.00005 finding measures the *input*, not the model. The delay claim is unaffected — it never rested on TSR. **GENERALISABLE METHODS FINDING (verified across two manifests on the SAME checkpoint `a654129d`):** `fold2_w512/whitebox_manifest.json` records `randomization_cosine = 0.0995`, i.e. the saliency **map** passes the standard model-randomization sanity check cleanly (map destroyed by randomizing weights), while the **scalar derived from that map** scores *higher* on the randomized net. A map-level sanity check does **not** transfer to statistics distilled from the map — those need their own randomization control. Applies to every derived number in `results["tsr"]` and to `tr_peak_lag`/`tr_lock_z` wherever they are quoted.
- [x] **REPEATABILITY FLOOR MEASURED — and it retires the quantile head's apparent AUC win (2026-07-30, local; `scripts/repeatability_audit.py` → `results/phase3/repeatability.json`).** The 2026-07-25 duplicate submissions were a mistake that cost ~80 GPU-h; they are also the only run-to-run repeatability data this project has, and the audit recomputes everything from `summary.json` with no re-run. Pairs verified to share an **identical config and identical seed** field-by-field. **4 of 10 pairs are BIT-IDENTICAL over all 20 epochs** (train loss *and* val AUC at full float precision); 6 diverge, worst single-epoch gap **0.0261**. That combination is the finding: an inherently nondeterministic pipeline could not reproduce *any* pair exactly, so the divergence points at something outside the config. **CAUSE IDENTIFIED 2026-07-30 (user ran `sacct`; nodes captured in `results/phase3/slurm_nodes.json`, matched to runs by End time and consumed by the audit).** **It is NOT the hardware: 9 of 10 pairs ran on two different nodes and 3 of those 9 still reproduced bit-exactly** (gl1517→gl1512, gl1502→gl1513, gl1508→gl1515), so which machine a job landed on does not predict reproduction. **It is our trainer.** `phase3_trainer.py:158` seeds every RNG and then nothing ever requests deterministic algorithms — no `torch.use_deterministic_algorithms`, no `cudnn.deterministic`, no `cudnn.benchmark = False`, no `CUBLAS_WORKSPACE_CONFIG` — while every run config carries `compile: true`. Benchmark mode picks kernels by timing them at runtime, so the arithmetic path tracks transient GPU state. Seeding fixes the inputs, not which kernels consume them, which is exactly the observed pattern: mostly reproducible, intermittently not, uncorrelated with the node. **Fix (open, local, small): set the determinism flags, and record node + library versions in `summary.json`** — without the accounting log, which is eventually purged, this was unanswerable from stored artifacts. **Not claimed:** among different-node pairs the ablation arm reproduced 3/5 and the quantile arm 0/4, which would fit the quantile head pulling in a nondeterministic kernel, but Fisher p = 0.167 on n=9. One re-run would test it. Watch ablation fold 0: **0.0261 apart mid-training, 0.0024 apart at the selected best AUC** — same-best-AUC agreement is NOT reproducibility and must never be quoted as it. **Floor = 0.0092 AUC** (max same-config drift in selected best AUC), a **lower bound** since seed spread is excluded. Side-input effect **0.0028** and quantile effect **0.0062** both sit under it. Side-input conclusion unchanged and reinforced ("worth about nothing" is what an unmeasurable effect looks like). **Quantile conclusion CHANGES: U12 §6's "a real improvement but val-selected" is withdrawn** — it is smaller than the gap between two runs of its own config, and which duplicate you keep swings it 41–69% (0.0037 vs 0.0062). U12 §6 carries a dated correction; U14 §3 carries the measurement + figure (`scripts/plot_update_14.py` → `u14_repeatability.png`).
- [x] **Two wrapper bugs that produced convincing false successes (2026-07-29, both found by post-mortem, both fixed).** **(1) `tsr_control_phase3.py:186` called `W._git_sha()` with no argument** while every other runner passes a root — a **latent crash that predates these edits**; the old code only reached that line at the very end, so the 12 h timeout hid it and the run could never have written a manifest anyway. Moving provenance assembly *before* the run surfaced it in seconds (job 55405355, `TypeError`). Fixed by extracting and testing `_provenance()`. **(2) `slurm/phase3_regenerative.sbatch` passed `--symbolic-csv` whenever the file existed** — but that flag dispatches to `_run_symbolic_mode` at `regenerative_phase3.py:220` and **returns without ever loading the network**. All five folds took the checkpoint-free fallback, exited 0 in ~90 s, and wrote only `regenerative_symbolic.json`; the wrapper's unconditional `echo "Manifest: …/regenerative.json"` then named a file that was never written, so the logs read as clean successes. Now `SYMBOLIC_ONLY=1` opt-in only, never inferred from a file existing, and the final step **asserts** its artifact exists instead of announcing it. Re-run (jobs 55408256–60) confirmed `Mode: full regenerative test against the checkpoint` on all five with ~2.1 MB manifests.

- [x] **UPDATE 14 CLOSED — determinism fix landed and the counterfactual manifest repaired (2026-08-10, local, TDD).** The two items U14 §6 left open are done, and neither required a returned job. **(1) Determinism.** `TrainConfig.deterministic` (**default False**, so every validated run stays bit-comparable) routes through `_seed_everything`, which now sets `use_deterministic_algorithms(True)` / `cudnn.deterministic` / `cudnn.benchmark=False` — strict, not `warn_only`, so a kernel with no deterministic implementation stops the run at setup instead of yielding a run that quietly is not deterministic. New `check_deterministic_env()` **refuses to start** unless `CUBLAS_WORKSPACE_CONFIG` is `:4096:8` or `:16:8`: cuBLAS reads it when it creates its handle, so the trainer cannot set it for itself and the failure would otherwise surface inside a matmul hours in. New `environment_block()` writes an **unconditional** `environment` key into `summary.json` (hostname, SLURM job/node, GPU, torch/CUDA/cuDNN/numpy/python, cuBLAS config, and the three determinism flags **read back from torch, not echoed from cfg** — a config echo cannot detect a flag that failed to take). Unconditional so an absent block means "predates this code", never "ran non-deterministically". Plus `--epochs`, because `--smoke` also clamps window geometry (`effective_window`) and would change the very kernels under test. 16 tests in `tests/test_phase3_determinism.py`. **Regression found and fixed:** `test_phase3_trainer.py::_cli_args` builds its Namespace explicitly, so the new args broke 7 `_build_train_config` tests until the helper was extended. **(2) sbatch:** `DETERMINISTIC=1` exports the cuBLAS var **and** appends the flag (they must travel together, and a `--deterministic` arriving via `EXTRA_ARGS` also triggers the export, so the flag cannot be passed without its prerequisite). Verified argv-**identical** when unset. **(3) Counterfactual manifest repaired** — `scripts/repair_counterfactual_manifest.py` + 9 tests. Needs **neither the network nor a blim evaluation**: both generators' stability was computed against the real boundary at run time and stored, as were both sets of coordinates, so the corrected comparison is a recomputation over recorded facts. The comparison was **extracted** from the DiCE loop into `compare_generator_outcomes()` and is now called by the runner *and* the repair, so the artifact cannot drift from the code that produced the published number — pinned by `test_repair_matches_what_the_runner_computes`. Repair reproduces **27/42 = 0.6429**, exactly U13's published figure; `n_agrees_with_physics_cf` gone; provenance states `network_reloaded: false`; pre-repair file kept as `counterfactuals_manifest.pre_repair_2026-08-10.json` (`results/` is gitignored, so it had no other history). **Full suite 840 passed / 1 skipped / 0 failures** (16 m 30 s, verified after every change above, not before). **OWED: the determinism repeat itself is a submission, not a result** — see NOW OPEN item H.

- [x] **U15 ARM D — methods correctness (2026-08-11, local, TDD).** The three items that would each have put a false or unsupported statement in the report. **(1) The `"huber"` severity mode is renamed `"point"`.** It named a loss this code has never contained: the point path calls `weighted_mse` and reports the part as `"sev_mse"`, and there is no Huber/smooth-L1 anywhere in `training/losses.py`. `"point"` rather than `"mse"` so it is the sibling of `"quantile"` and both describe **what the head emits**, leaving the loss to be stated where it can be stated accurately. `"huber"` remains an accepted **deprecated alias**, normalised at the boundary by `normalise_severity_mode()` so exactly one spelling reaches the model, the loss router, the MLflow params and the `preprocessing.json` gate — every config and `summary.json` on disk says `"huber"`, and rejecting them would strand the validated runs for a cosmetic gain. **The trap this set, and it was real:** `phase3_trainer.py` gated the `preprocessing.json` severity block on the literal `!= "huber"`, so renaming the default alone would have made **every ordinary run start emitting a `severity` block**, silently changing the default path's schema — and `xai_phase3.py` reads an absent block as "the default point head". Now compares against `DEFAULT_SEVERITY_MODE`, pinned by a test that greps for the old literal. 19 tests in `tests/test_severity_mode_naming.py`, including one asserting on **behaviour** rather than a string (the point path's regression term equals `weighted_mse` exactly), so the name stays honest if the loss ever changes. **(2) "PySR" — nothing to fix in code, one thing to know.** Grep-verified: the word appears in no source file and no current doc except the plans that say not to write it. It survives in **Research Update 1** (`summer_research/Research Update.md:57`) as a forward-looking intent ("PySR symbolic regression ... will be compared against the SINDy recovery"), which was never carried out and is **left unedited** — it was a true statement of intent when written, and rewriting delivered history is worse than a footnote. **The accurate wording for the report**, from reading `xai/symbolic.py`: each important edge's learned activation is sampled over the data-visited range and fitted to a **fixed library of eight closed forms** (const / linear / quadratic / cubic / saturating-tanh / oscillatory-sin / abs / relu), selected by **BIC** with an optional R² acceptance threshold and a simplicity tie-break. That is a library fit, not genetic-programming symbolic regression. **(3) `tsr_control_verdict` reviewed, and the review found two more defects of the project's recurring family** — a missing measurement reading as a pass. Both could only ever produce a **false clear**, and the one real result is an artifact call, so **no recorded conclusion changes**; both are fixed because the plan says to fix them if the control is re-run. **(3a)** `min_windows` was skipped entirely when `n_lock_tested` was `None`, so a caller who omitted it got a definitive clear with nothing establishing how much evidence stood behind it — the exact principle the lock arms already applied to themselves. Now `window_count_unknown` blocks a clear while still permitting the conservative artifact call. Note the live caller passes it via `.get()` (`whitebox_phase3.py:510`), which **can** be `None`. **(3b)** `tau_specific` was a bare `trained > shuffled`, so a shuffled lock a ten-thousandth below the trained one counted as tau-specificity and permitted a clear; it now must clear the **same ratio** the untrained arm is held to, rather than a second invented constant. Verified the real fold2_w512 values keep their recorded reading (`tau_specific` still True, verdict still artifact). One pre-existing test encoded the 3a weakness (it expected a clear while omitting the count) and was updated to match the real call site.

- [x] **NEW INTERPRETABILITY RESULT — the learned delay coupling is a tooth-pass comb filter (2026-08-11, local, TDD, no GPU and no checkpoint).** U13-C measured that a delayed edge is a near-exact scaled **copy** of its instantaneous partner and filed the result as a refuted hypothesis ("the net learned the SUM, not the regenerative difference"). What a sum *is*, mechanically, was never extracted. With the two curves proportional, the unit computes `y(t) + b*y(t-tau)` with `b = corr*gain` the projection of one curve onto the other — **a delay-and-add comb**, `|H(f)| = sqrt(1 + b^2 + 2b cos(2 pi f tau))`, with passbands on every integer multiple of `1/tau`. Since the model sets `tau = round(fs/tpf)` per sample, `1/tau` **is** that cut's tooth-pass frequency, so the comb auto-tunes to the tooth-passing harmonic family of whatever operating point it sees. **The difference the plan wanted would have notched exactly the same family.** Sum-vs-difference is therefore *reinforce vs cancel the forced content*, not "wrong structure" — a statement about what the network computes rather than about which hypothesis lost. **No linearity assumption is used anywhere:** the proportionality is between the two learned *curves*, so the comb acts on the nonlinearly warped signal `y` and `phi` may be as nonlinear as it likes. New `xai/comb.py` + `scripts/comb_phase3.py`, 30 tests, derived **entirely from stored U13-C manifests** (`network_reloaded: false` in the manifest). **RESULT, all five folds:** of pairs proportional enough for the reading to describe them at all (`|corr| >= 0.9`, pre-committed), **91.2-99.0% reinforce the tooth-pass family** (importance-weighted), median comb contrast **15.5-17.7 dB**; pooled **1810/3840 pairs qualify, 96.6% weighted**. **Against its control on every fold, no overlap:** the cross-channel control reaches only **63.0-71.8%** and 11.3-13.3 dB — and that control is *conservative*, since cross-channel pairs still share real signal correlation. Untrained null (summary-only in the stored file): `frac_base_dominated 0.66`, i.e. two-thirds of an untrained net's pairs carry no learned spline structure at all. **Uniform across the architecture** rather than a quirk of one gate: fold 2 gives kan_r 1.000 / kan_z 0.987 / kan_h 0.981 weighted, and ytpos 1.000 / Forcex 0.993 / Forcey 0.991 / xtpos 0.961, all at 15.8-18.1 dB. Top pair: `kan_z`, xtpos, out 58, corr 0.9966, gain 1.0264 => **b = 1.023, contrast 88x (38.9 dB)**. **THE TIDY STORY DOES NOT HOLD, and is reported as such:** "chatter falls in the notch" depends on `fn/tpf` mod 1, and the dataset's mean ratio is **2.116** — close enough to the second harmonic that at the mean the chatter band is attenuated only to **0.935**. Integrated over the measured spread (fn 1072 +/- 492 Hz against tpf 507 Hz) it is **0.64** against **1.000** for the forced content, which always sits on a passband by construction. Both numbers are in the manifest; quoting either alone misleads. **Caveats to carry:** only **47%** of pairs are proportional enough to describe (per-fold 41-56%), so this covers about half the delay coupling, not all of it; and the effect **does not track the ablation fold split** (fold 3 scores 0.979 while fold 4 scores 0.912), the same non-corroboration U13-C already flagged. Figures: `comb_response.png` (peaks landing on the tooth-pass harmonics, chatter band overlaid), `comb_contrast_by_fold.png`. **Confirmation built and ready to submit:** `scripts/comb_probe_phase3.py` + `slurm/phase3_comb_probe.sbatch` sweep a sinusoid through the *real* learned edge functions and compare measured against predicted, with an untrained-network control — and it runs on the **`standard` CPU partition**, so it does **not** compete for the GPU allocation that is currently capped at `AssocGrpGRES`. The harness is pinned by a test asserting measured == analytic to 1e-6 for a linear pair, so a mismatch on the real net cannot be the probe's own artifact.

- [x] **COMB CONFIRMATION RETURNED, 5/5 FOLDS — and it sharpens the claim rather than just supporting it (2026-08-11, GL jobs on `standard`, ~22 s each; local `results/phase3/xai/comb/comb_probe_*.json`).** Driving a swept cosine through the **real** learned edge functions reproduces the analytically predicted comb: **median corr(predicted, measured) = 0.9928-0.9975** per fold. **Trained pairs peak within 0.1 of a tooth-pass harmonic in 114/125 (0.912)**; the untrained control in **72/125 (0.576)**. **THE CONTROL IS THE RESULT.** 0.576 is **not** distinguishable from a coin flip (two-sided binomial vs 0.5, **p = 0.107**), and chance for "peaks on a harmonic" is *0.5, not 0.2*: at random initialisation the KAN base path dominates (`frac_base_dominated 0.66` in the stored null), any two near-linear curves are almost perfectly correlated, so a random pair **already forms a comb** whose sign is a toss-up. Trained vs 0.5: **p = 4.8e-23**. **So the delay-and-add structure is ARCHITECTURAL, and what training determines is the SIGN — reinforce or cancel the tooth-pass harmonic family. Training picks reinforce 114/125 while random init is at chance.** That is a narrower claim than "the network built a comb" and it is the one the evidence supports; state it that way. It also agrees with the offline route (trained 96.6% weighted vs cross-channel control 63-72%), and two independent derivations landing together is worth more than either. **THE TAIL, which the median hides:** min corr(predicted, measured) is **-0.7631**; 4/125 pairs measure below 0.90 against prediction, and **all four are `kan_r` on `ytpos`** — systematic, not noise. Two of the four sit **below** the 0.9 proportionality gate (corr 0.806, 0.389), i.e. the module already says the comb reading does not describe them and the measurement agrees — the gate earning its place empirically. **The other two are above the gate (0.922, 0.972) and still disagree, so the gate helps but is not sufficient:** disagreement runs **12.5% below the gate vs 1.8% above it**, a 7x enrichment rather than a clean separation. A first hypothesis (the probe driving beyond the data-visited range) was **checked and is wrong in the opposite direction** — the disagreeing pairs span 1.997 of the grid against a typical 1.03, so they are the most strongly driven edges, not the least. Left unexplained rather than given an invented mechanism. **Known probe limitation, deliberately not "fixed":** it probes the top 25 by importance **without** applying the proportionality gate, so 16/125 probed pairs are ones the offline reading excludes; filtering them would raise the headline and would be cherry-picking after the fact, so the split is reported instead.

---

## NEXT ACTION

### ▶ CURRENT FRONTIER (2026-07-29) — three updates to the Aug 14 report

**Full work division: `docs/phase3_tkan/updates_12_14_plan.md`.** Updates 12 (Jul 19–25),
13 (Jul 26 – Aug 1), 14 (Aug 2–8), each ~20 h, leaving Phase 9 (Aug 10–15) clear for the report.

**Where the project actually stands.** The delay claim rests on **three independent
interventions** — ablation (5/5 folds, mean drop 0.38, 3 folds to chance), per-window τ-sweep
(peaks at ×1 on **all five** folds), and τ-jitter (accuracy falls in proportion to delay error) —
plus a network-free SINDy-DDE recovery (median τ*/τ_phys 0.986). The **possible fourth** line
(TSR delay-lock, p = 0.00005) is **dead as of 2026-07-30**: its control returned and the untrained
network locks harder than the trained one, so the lag-τ structure is in the input construction, not
in what the network learned. Two claims are now retired on the evidence (concept probe, TSR) — but
note the TSR retirement rests on a *different and correct* reason than Update 11 gave for it.

**STATE AS OF 2026-07-29 — what moved since this section was written.** Five of the seven items
below are now closed by returned results, and two of the closures are **negative**: the
`--no-side-input` ablation and the quantile severity head both came back (see the Status board) and
neither delivered — side inputs are worth ≈ 0.002 AUC, and the quantile head leaves the severity
heavy tail where it was. U13-A/B/C all ran. **The regenerative spline test refutes the plan's
"contains x(t) − x(t−τ)" criterion**: the network learned the SUM, replicated 5/5 folds against a
flat untrained null. **The U12-B TSR control returned 2026-07-30 on its third submission** (the
first timed out, the second hit a latent `_git_sha()` crash) and is the third negative: the
delay-lock is a construction artifact. Two bugs in this session's own wrappers produced convincing
false successes — both fixed, both worth remembering as a pattern: **a job exiting 0 is not evidence
it did the work**, which is why the U12-B manifest was verified via `provenance.complete` and a
per-condition window count before a single number was read. Suite 741 → **759 passed / 1 skipped**.

**Correction to the 2026-07-20 frontier note.** It said the eval-only queue was saturated and
every remaining high-value item needed a retrain. That was wrong on two counts, both found in the
2026-07-25 audit against the MDP plan PDF: **Monte Carlo survival rates do not exist anywhere**
(grep-verified: zero hits for `survival`/`monte_carlo`), and **the DiCE path has never been run**
— the shipped 81.6% came from `physics_constrained_counterfactual`. Both are named pieces of
plan criterion 7, both are local and cheap, and neither needs a GPU. They are Update 13.

**Two naming defects to fix before the report** (each would put a false statement in a methods
section): the plan names **PySR** but the repo uses a custom BIC library fit in `xai/symbolic.py`;
and `_VALID_SEVERITY_MODES` calls a mode `"huber"` when the loss is `weighted_mse`.

**One explicit descope.** The inverse-problem criterion (ktc/fn/damping within 10% on ≥80% of
held-out) is unreachable in the time left and has no partial artifact to build on. τ is the one
physical parameter this project recovers, and it recovers it three ways. State this as a scope
change in the report rather than leaving the criterion silently unmet.

#### PENDING — in priority order

**⚠️ ITEMS 1 AND 2 ARE DONE (returned 2026-07-21/26, pulled local 2026-07-26).** Both came back
negative; the numbers are on the Status board. The text is kept below because the *reasoning* about
why each mattered is still the right framing for the report — but do not re-run either. What
replaced them as live work is listed under **NOW OPEN** at the end of this list.

1. ~~**Quantile severity head — the one remaining "the model got better" result.**~~ **RAN. AUC
   0.9629/0.9226/0.9796/0.9631/0.9799 (no regression), severity heavy tail essentially unmoved.**
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

2. ~~**[GL retrain] `--no-side-input` 5-fold ablation — still open since 2026-06-14.**~~ **RAN.
   Mean 0.9531 side-off vs 0.9554 side-on ⇒ the side inputs are worth ≈ 0.002 AUC. Open since
   2026-06-14, now closed, and the answer is that this was never the source of TKAN's margin.**
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

3. ~~**[GL eval, cheap] TSR at scale — decide the null.**~~ **RAN 2026-07-23/24 — see the Status
   board row.** It decided neither branch: the metric emits its search floor on 85.9% of windows,
   and within the 14% that resolve, the peak locks to τ at p = 0.00005. Superseded by U12-A/U12-B
   below.

3a. **[local, TDD] U12-A — fix the TSR delay metric.** `tr_peak_lag` is `argmax` of the relevance
   autocorrelation over `[2, N/2]`, so a monotonically-decaying autocorrelation silently returns
   the floor (2) rather than "no peak found". Report the **resolved fraction** explicitly; add a
   delay-lock statistic that tests the autocorrelation *at* lag τ against a null instead of taking
   a global argmax; never average the floor sentinel into a mean. Files: `scripts/whitebox_phase3.py`
   (`tsr_one_window`, `tsr_aggregate`) + new tests. Back-compat: keep `tr_peak_lag` in `per_window`
   so already-returned manifests stay readable.

3b. **[local then GL, cheap] U12-B — the construction-artifact control. GATES ANY TSR CLAIM.**
   The delayed channel is built by shifting the signal by τ, so the relevance vector may carry
   lag-τ structure *by construction*, independent of anything learned. Run the fixed metric on
   (a) a randomly-initialised checkpoint and (b) the trained net with deliberately wrong τ.
   **Survives both ⇒ artifact, TSR is genuinely dead, retire it for real.** Degrades ⇒ TSR is a
   **fourth independent line of delay evidence**. Nothing about TSR enters the report until this
   returns.

3c. ~~**[local, SDD] U12-C — close the physics-floor stressors build.**~~ **DONE 2026-07-25
   (Tasks 7 + 8). But the band-notch contrast it shipped reports `matched: false` with
   `max_rel_err` exactly 1.0 on all five folds — see NOW OPEN item A: that is a sentinel, not a
   measurement.** Original note follows. PAUSED at Task 7 since
   2026-07-20; Tasks 1–6 complete + reviewed, suite 634 passed / 1 skipped, all uncommitted.
   Task 7 is pre-scoped — brief at `.superpowers/sdd/pf-task-7-brief.md`, base snapshots taken,
   and the plan's Step-5 factual error already corrected (use `batch["fn"]`; do **NOT** reload the
   table). Then Task 8 and the final whole-branch review. Resume via `.superpowers/sdd/progress.md`.

3d. ~~**[local, U13] Monte Carlo survival + the real DiCE run — closes plan criterion 7.**~~
   **DONE + RAN 2026-07-29. Survival ≥ 0.85 in 19/42 (criterion met), mean 0.702, min 0.086.
   DiCE 42/42 found, 36 physically stable, outcome agreement 27/42. See the Status board.**
   Grep-verified 2026-07-25: **no `survival`/`monte_carlo` code exists anywhere**, and
   `dice_counterfactuals()` (`xai/counterfactuals.py:52`) has never been run — the shipped 81.6%
   is `physics_constrained_counterfactual`. Sample machining uncertainty around each CF and compute
   the fraction of draws still below `blim(ω)` via `physics/altintas_budak.py`; then run DiCE on the
   same 42 points and verify three ways (DiCE vs physics-constrained vs analytic boundary).

3e. ~~**[local, U13] The regenerative spline test.**~~ **RAN 5/5 FOLDS 2026-07-29 — AND IT
   REFUTES THE HYPOTHESIS. The network learned x(t) + x(t−τ) (a sum), not the regenerative
   difference: mean score −0.46…−0.64 where +1 is negation and −1 is a copy, 23–36% summing vs
   0.1–1.0% negating, top pairs at corr +0.997. Real (untrained null −0.023, 5/5 replication) but
   NOT the plan's criterion, and it does NOT track the ablation fold split.** Original note
   follows. Most direct available test of the plan's
   qualitative criterion "contains x(t) − x(t−τ) without hardcoding". Delay edges are `in=68`
   (`delay:xtpos`) / `in=69` (`delay:ytpos`) on `encoder.layers.0.kan_{r,z,h}`. For each, pull the
   matched **instantaneous** edge on the same `(kan, out)` unit, evaluate both φ via
   `KANLinear.edge_function` over the data-visited range, and correlate — a strong *negative*
   correlation is the regenerative difference emerging from learned functions. Controls are built
   in: folds 0/1/2 carry 5 delay edges each in the top 25, folds 3/4 carry 1.

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

#### NOW OPEN (2026-07-29) — this is the live list; everything above is context

**A. [local, TDD] Fix the band-notch contrast — TWO defects, gates U12 §5.** The first is the more
serious and was found by the Update-12 §5 review, not by this session: **the runner passes a single
batch-mean `fs`/`fn`/`tpf` to a batch that spans 37 distinct sampling rates (12.3–38.3 kHz) and
eight chatter frequencies**, so every window is notched at the same nominal location. A window
recorded at 12.5 kHz has its frequency axis mislabelled by ~2.4×: a band nominally at 950 Hz deletes
content near 400 Hz while that window's true chatter frequency is 500 Hz. Both arms delete a few
percent of spectrum from physically arbitrary places, which is why they came back
indistinguishable — and that holds **regardless of whether the energies matched**. Needs per-window
frequencies. The second defect is the sentinel below, which is what makes the number *look* broken:
all five folds report
`matched: false` with `max_rel_err` **exactly 1.0**, while the batch-mean energies agree to ~0.4%
(fold 0: chatter 0.03921 vs tooth 0.03937). Exactly 1.0 is a sentinel: `degradation.py:520` seeds
`best = (inf, halfwidth_hz, 0.0)`, so any window where every candidate returns `ch_frac <= 0` skips
the loop body and keeps a **zero-width tooth notch** ⇒ rel err = |0−c|/c = 1.0 — and `max_rel_err`
is a max over windows, so one degenerate window pins it. Same class as the `tr_peak_lag` floor that
U12-A fixed. Fix: count and report the unsolved windows instead of silently notching nothing, and
report a quantile beside the max. This likely converts §5 from "refused to match" into a result.

**➤➤ CLOSED 2026-07-30: fix landed, re-run returned 5/5, and the contrast is STRUCTURALLY UNACHIEVABLE.** GL jobs 55534785–89, all five folds read. **The fix did exactly its job — the 1.0 sentinel is gone, `dropped=0` everywhere — and what it exposed is that the experiment cannot be run as designed.** Per fold: **unsolved 298/298/323/298/304 of 512 (58.2–63.1%)**, `rel_err_p90` **0.9751–0.9783** against a 1% requirement, `matched: false` on all five. **My prediction below ("likely converts §5 into a result") was WRONG.** **Mechanism, reproduced locally on a synthetic window tuned to the observed fractions:** the chatter notch is ONE band with a minimum removable energy ≈0.022; the tooth notch is THREE bands on the strongest peaks with a minimum ≈0.130. `match_notch_halfwidth` saturates at its 0.78 Hz floor for every target below ~0.10 and returns a halfwidth that notches nothing ⇒ `th_frac=0` ⇒ candidate rejected ⇒ window unsolved. **The bind is structural: raising `ch_frac` to where the tooth side can match needs a wide chatter band, which reaches the tooth harmonics, which the safety guard forbids — the guard that makes the contrast meaningful is what makes it unmatchable.** Tested both escapes: `n_harmonics` 3→1 and L 1024→4096 leave the tooth floor **2–4× above** the highest guard-legal chatter fraction in every combination. Finer bins do not help (tooth energy is concentrated at the peak bin). **DO NOT read the AUC direction:** chatter notch hurt more in 4/5 folds (mean +0.0042), which would look like a spectral-energy detector and cut against the delay story — but the pairing is broken at the window level (~97% per-window mismatch behind a 3.9–7.4% batch-mean difference), **and the energy excess does not explain it either** (corr(excess, ΔAUC) = **−0.744**, n=5, the fold with the largest excess has the smallest gap). Uninterpretable in both directions. **Supportable statement:** deleting ~6% of spectrum at either location costs ≤0.006 AUC against a ~0.99 baseline. **THIRD same-family sentinel found + guarded:** `rel = |th−ch|/ch` divides by a near-empty chatter band ⇒ `max_rel_err ≈ 2e4`; a `>0` test admits it since one near-empty bin is positive. New `min_chatter_frac` (**1e-4**, chosen from the measured gap: degenerate ≈3.2e-6, smallest legitimate 8.09e-4 — 1e-3 was tried first and wrongly discarded a window that matched to 0.4%), reported as `n_degenerate_chatter` **separately from** `n_unsolved`, with `unusable_fraction` driving the veto. **Returned manifests PREDATE this guard**, so quote `rel_err_p90`, not `max_rel_err`. U12 §5 rewritten from "no usable number" to the diagnosed structural negative. **No further re-run planned** (user decision: accept the negative).

**➤ Code fix, 2026-07-30 (local, TDD) — history:** Both
defects are fixed in `xai/degradation.py` + `scripts/fuzz_phase3.py`, with **6 new targeted tests**
(`test_degradation.py`: `..._aims_at_each_windows_own_chatter_frequency`,
`..._accepts_per_window_sampling_rate_and_tooth_frequency`,
`..._scalar_and_broadcast_frequencies_agree_exactly`, `..._rejects_a_frequency_array_of_the_wrong_length`,
`..._reports_unsolved_windows_instead_of_a_fake_full_mismatch`,
`..._unsolved_window_does_not_pin_the_reported_error`). **43/43 pass on the two touched test files.**
(1) `fs`/`fn`/`tpf` now accept a scalar **or** a per-window array; tooth centres and the
tooth-harmonic safety guard became per-window with them, and scalars stay byte-identical so nothing
already validated moves. (2) Unsolved windows are counted and excluded rather than pinning the max:
`energy_match` gained `n_windows`/`n_solved`/`n_unsolved`/`unsolved_fraction`/`max_unsolved_frac`
(2% default, a named parameter not a magic number) and **`rel_err_p90` beside `max_rel_err`**; both
error fields are **`None`, never a number**, when nothing solved — the same "no answer is not a
measurement" precedent as `tr_lock_z`. `matched` vetoes on the unsolved fraction. **Third defect
found during the fix, not in the brief:** the old caller's `np.isfinite` guard ran on the batch
*mean*, so **one NaN `fn` row discarded all 512 windows**; non-finite rows are now dropped
individually and `n_windows_dropped_nonfinite` is reported unconditionally, so a reader can tell
"none dropped" from "manifest predates the check".

**A2. [applied 2026-07-30; code fix now optional] The TSR control's verdict rule did not match the
rule as written.** Recorded in Update 12 §4, found before any result was read. The ratio test exists;
three things it depends on do not. **(1)** The significance threshold that defines "a lock" is
computed and never consulted, so a trained value indistinguishable from noise — or pointing the wrong
way — still passes the ratio and prints the fourth-line-of-evidence claim. **(2)** No minimum number
of testable windows, so one window can carry a definitive verdict. **(3)** The `trained_shuffled_tau`
condition is computed at full cost and then ignored, and it is exactly the condition that separates a
real lock from one that only reflects how smooth the attribution curve is. That matters because the
lock statistic is **biased toward short delays on smooth curves**: a synthetic curve with no delay
structure registers as locked at every delay up to 16 steps, and this dataset's delays sit inside
that range.

**The corrected rule was applied by hand to the returned manifest on 2026-07-30 and it does not
change the verdict — it strengthens it.** All three gates now pass *against* the model: the untrained
arm clears the `z > 2` significance gate in **24/24** windows (so gate 1 cannot rescue the trained
side), every condition has 24 testable windows (gate 2 satisfied), and the shuffled condition — the
one the code ignored — was read, and shows the effect is τ-specific in a way the artifact hypothesis
already predicts. **`tsr_control_verdict()` in `whitebox_phase3.py:345` is still the unfixed
version.** It is no longer science-blocking, because the one result it will ever be asked to judge
has been judged by hand and recorded on the Status board. Fix it only if the control is ever re-run;
if so, add the significance gate, an `n_lock_tested` minimum, and make the shuffled arm load-bearing.

**A3. [minor, only if the control is re-run] The τ-shuffle is not a derangement.** 2 of 24 windows
were handed back their own τ (`shuffle_seed: 0`). It biases toward the shuffled arm looking better,
which is the safe direction, and excluding those two only sharpens the result (mean 0.88 → 0.70).
Use a derangement if there is ever a second run.

**B. [CLOSED 2026-07-30] The U12-B TSR control returned on its third submission (job 55408255).**
Fetched, verified complete, and read; full numbers on the Status board. The two survivability fixes
did their job — all three conditions finished, not just the decisive pair. **TSR is now a closed
negative: the delay-lock is a construction artifact, and no TSR claim goes in the technical report.**
Update 12 §4 rewritten from branch-text to result on 2026-07-30. Only follow-on is item C below;
nothing about TSR is pending any further run.

**C. [user, decision — now evidence-backed, 2026-07-30] Canonicalise the duplicate run dirs.** Ten
run dirs where five belong, in both `runs/` (side-off) and `runs_quantile/`, with nothing in the path
marking which is which. Blocks the zoo republish and anything that globs those trees.

**⚠️ The old recommendation here rested on a wrong number.** It said "the pairs agree to ≤0.005 AUC
and three ablation folds match to 4 dp" and advised keeping the earlier run per fold. The first
clause is false: **quantile fold 1 differs by 0.0092**, and the choice of which copy to keep moves
the quantile arm's headline effect by 41–69% of the effect itself. New audit: `scripts/repeatability_audit.py`
→ `results/phase3/repeatability.json` (recomputes from `summary.json` only; no re-run needed).

**What the audit found, and it is worth more than the canonicalisation decision.** Both copies of
every pair share an **identical config and an identical seed** (verified field-by-field, not assumed).
**4 of 10 pairs are BIT-IDENTICAL across all 20 epochs** — train loss and val AUC agree at full float
precision at every epoch. The other 6 diverge, with single-epoch gaps up to **0.0261 AUC**. An
inherently nondeterministic pipeline could not reproduce *any* pair bit-exactly, so the divergence is
**not** irreducible training noise; something outside the config differed between submissions (GPU
model, node, library version). **`summary.json` records no hardware or environment, so the cause
cannot be recovered from stored artifacts** — that is the real defect, and the fix is to record it.
Note ablation fold 0: the two runs wander **0.0261** apart mid-training and land **0.0024** apart at
the selected best AUC, so the headline number badly understates the instability. Never quote
same-best-AUC agreement as reproducibility.

**Consequence for the two 2026-07 retrains (this is U14-A's spine).** Repeatability floor = **0.0092
AUC** (max same-config drift in selected best AUC), and it is a **lower bound** since seed spread is
not included. Against it: the side-input effect is **−0.0023 to −0.0028** and the quantile effect is
**+0.0037 to +0.0062**, both *below* the floor. The side-input conclusion is unchanged and in fact
reinforced (the claim was already "worth about nothing"). **The quantile conclusion changes:** U12 §6
calls its 0.962-vs-0.955 a "real improvement but val-selected". It is not claimable at all — it is
smaller than the drift between two runs of its own configuration. U12 §6 needs that correction.

**Recommendation, revised:** keep the **later** run per fold, since the already-published numbers in
U12 §6 and on this board come from it, so keeping the earlier one silently changes text already sent
to Prof. Abell. Record the choice in a marker file in each tree. **[user, GL, cheap]** `sacct -j` on
the two submission batches will show the node and GPU for each run and would confirm or kill the
hardware hypothesis in one command.

**D. [done 2026-07-29] Updates 12 and 13 revised/written and rendered.** Both in
`summer_research/Research Updates/`, both `.md` + `.docx` (pandoc, figures embedded: U12 one, U13
three). Rewritten against the `no-ai-slop` skill (installed at `~/.claude/skills/no-ai-slop`);
banned-word/em-dash/binary-contrast checks clean. **U12 §5 is now closed as "no usable number and
here is why", not left pending** — the review returned, and both defects are written up. **U12 §4 is
the one remaining placeholder**, waiting on item B. **Figures come from
`scripts/plot_updates_12_13.py`** (validated 3-slot palette; every series direct-labelled because
aqua is under 3:1 on the light surface). **One number was removed as unreproducible:** the earlier
U12 draft claimed the resolved TSR peak lands on an exact harmonic of τ in 60/90 windows (67%). No
natural definition reproduces it from the merged manifests — integer divisibility gives 44/90, a 5%
ratio tolerance 72/90, a 10% tolerance 80/90. Replaced with the divisibility count, definition
stated inline. The verified numbers (550/640 at floor, 31/90 exact, median ratio 1.0000) all
reproduce exactly from `per_window`.

**C-CLOSED (2026-08-10).** The canonicalisation shipped: `scripts/canonicalise_runs.py` +
`tests/test_canonicalise_runs.py` write `results/phase3/RUNS_CANONICAL.json` plus a marker in each
run dir, resolving each fold **by matching the published AUC** rather than by timestamp, and
refusing rather than guessing when no run reproduces the published figure. The four bit-identical
pairs cannot be separated by the number, so submission order is recorded there **as a tie-break,
not dressed up as a measurement**. Runs belonging to no published sweep are marked too, so
"unmarked" never means "unclassified".

**H. [SUBMITTED — jobs 56977333 and 56977341 on `gpu_mig40`, `--time=04:00:00`, awaiting return]
The determinism repeat.** **Queue-routing finding, worth keeping (2026-08-11):** the first pair (56886459 /
56886484, `spgpu`) sat a full day at `AssocGrpGRES`, which is **not** a shortage of A40s — it is a
cap on how many GPUs the `jaabell0` account may hold at once, and another user on the account was
holding all of them with ~27 array tasks behind them. `sacctmgr show assoc` displays it as blank
because the limit lives on a **parent** association; `scontrol show assoc_mgr account=jaabell0
flags=assoc` shows the live limit and usage. **The cap is ACCOUNT-WIDE, not partition-scoped — changing partition does not help.**
A first read of this was wrong and is recorded here so it is not repeated: resubmitting to
`gpu_mig40` appeared to clear the block (reason `(None)`), but a second pair on the *same*
partition came back `AssocGrpGRES`. **`(None)` means the scheduler has not evaluated the job yet,
NOT that it is eligible** — a freshly submitted job shows it for one scheduling cycle regardless of
limits. Never read a reason code off a job submitted seconds earlier. Confirming evidence that was
visible the whole time: the other user's *pending* jobs also carry `AssocGrpGRES`, so the account
is at its ceiling and every member is blocked behind the same five running jobs. Partition options and their architectures: `spgpu` A40 sm_86
(known-good) · `gpu` V100 sm_70 (**the documented `no kernel image` crash — never**) · `gpu_mig40`
A100 sm_80 (**in use now, and healthy**) · `gpu-rtx6000` Blackwell sm_120 (most capacity — 7 nodes
against `gpu_mig40`'s 2 — but **whether the prism torch build has sm_120 kernels is still UNKNOWN**;
the 2026-08-11 probe was abandoned before it scheduled. **Settle this before planning U15's
noise-floor arm**, which wants 15 runs and is capacity-bound: `srun -p gpu-rtx6000 -A jaabell0
--gres=gpu:1 --time=00:10:00 --pty` then a one-line `torch.randn(...).cuda()` matmul. A failure
looks exactly like the V100's `no kernel image`) ·
`viz` P40 sm_61 (older than the V100). **Reading `sinfo` state suffixes correctly matters here:
a trailing `-` means "planned by the backfill scheduler for a higher-priority job", NOT draining.
A drained node carries an admin reason and timestamp; `sinfo -p P -o "%n %t %E %H"` showing
`none`/`Unknown` proves the node is fine. Misreading `mix-` as drained nearly cost two correctly
queued jobs.** **Both runs must be on
the same GPU model or it is not a determinism test**; `summary.json["environment"]["gpu_name"]` now
records it, so verify rather than assume. Also pass `--time` on short runs: the sbatch's 36 h
default is sized for 20 epochs and makes a 2 h job ineligible for backfill. The code is in the tree and tested; **the measurement is not done**, and
nothing about determinism may be claimed until both return. Two submissions of the SAME fold and
seed, 5 epochs, `DETERMINISTIC=1`, writing to `results/phase3/runs_determinism`. The command, for
the record and for re-running:
```
env DETERMINISTIC=1 EXTRA_ARGS="--epochs 5" \
    OUTPUT_DIR=/nfs/turbo/coe-jaabell/PRI2M/results/phase3/runs_determinism \
    sbatch --array=0 --export=ALL slurm/phase3_train_gpu.sbatch
```
Run it **twice** (done: the two job IDs above). Then compare the two `summary.json` histories epoch
by epoch: identical at full float precision ⇒ the flags fixed it. **Verify before reading:** the
log must contain `Determinism: on (CUBLAS_WORKSPACE_CONFIG=:4096:8)` and the run's new
`summary.json["environment"]` must show `deterministic_algorithms: true` / `cudnn_benchmark: false`
— read back from torch, so it states what was in effect. A job that exits 0 is not evidence it ran
deterministically (Update 13's lesson, twice). **This tests determinism under the NEW settings and
does not retroactively change the 0.0092 floor**, which was measured on runs that had these flags
off. 5 epochs rather than 20 because the existing duplicate pairs already diverge at
**epoch 1** (0.0261), so a full pair would spend ~16 GPU-h to observe something visible in
minutes. `--epochs` rather than `--smoke` because smoke also clamps the window geometry.

**I. [plan] Update 15 scoped — `docs/phase3_tkan/update_15_plan.md` (2026-08-10).** Four arms:
determinism result + **a noise floor worth quoting** (the 0.0092 is one duplicate per config, no
repeated baseline, no seed spread, and is quoted per-fold while every effect is quoted as a
5-fold mean — the proposed 3 seeds x 5 folds needs a GPU-budget decision); **mode ablation**
(`mode_edge_sets` already returns the mode set plus both mandatory controls and is tested, but
**nothing consumes it** — the selection half of U14-B step 8 exists, the measurement half was
never written); **learnable tau** scoped deliberately as design + single-fold pilot, since an
integer lag index has no gradient and the differentiable-delay choice is the design pass;
**methods correctness** (the `"huber"` misnomer, never write "PySR", the `tsr_control_verdict`
fix now sitting unreviewed in the tree, aggregate delay-engagement stats, threshold calibration).
Pre-committed there, before any number: the mode effect must clear the floor, and the held-out
baseline's own fold spread (**0.013**) already exceeds the current 0.0092 estimate.

**F. [housekeeping] Still nothing committed since `c2392b2e`.** Now ~55 paths under `SINDy_DDE/`.

**G. [process, learned the hard way 2026-07-29] A job exiting 0 is not evidence it did the work.**
Two wrappers this session produced clean-looking successes while skipping the analysis (see the
Status board). Both fixes generalise: **never infer a mode flag from a file existing**, and make the
last line of a wrapper **assert its artifact** rather than echo its path. Worth applying to the
other sbatch files, none of which currently assert anything.

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
- 2026-07-29 — **Both retrains read, U13 built+run, U12-B still chasing its wall, two false-success wrapper bugs found.** Session started from a cold read of disk (this doc was 4 days stale and understated the state badly). **Returned GL work, verified against `summary.json`/`preprocessing.json` rather than docs:** the `--no-side-input` ablation (open since 2026-06-14) and the quantile severity head both completed on 2026-07-21/26 and were sitting unrecorded in `results/phase3/runs{,_quantile}` — **both negative** (side inputs ≈ +0.002 AUC; quantile head no AUC regression but the severity heavy tail unmoved), and **both trees hold duplicate sweeps** (10 dirs where 5 belong) because each was resubmitted on 2026-07-25 having already finished, costing ~80 GPU-hours. **U13 wired + run:** `monte_carlo_survival` and DiCE both existed as library code that no runner had ever called; wired with TDD (`--mc-draws`, `--dice`, `build_surrogate` now returns the estimator), ran on the real 42 cases — **survival ≥ 0.85 in 19/42 (criterion 7 MET) but mean 0.702 / min 0.086, so fewer than half the recommendations survive machining uncertainty**; DiCE 42/42 found, 36 physically stable, and its first comparison metric was **mislabelled by me** (compared depth *direction* when the physics search structurally cannot raise depth) — replaced with an axis breakdown + outcome agreement (27/42), recomputed from the returned manifest with no re-run needed. **U13-C ran 5/5 folds and refutes its own hypothesis: the network learned x(t) + x(t−τ), the SUM** (mean score −0.46…−0.64 where +1 = negation; 23–36% summing vs 0.1–1.0% negating; top pairs corr +0.997), real against a flat untrained null (−0.023) but **not** the plan's regenerative criterion, and it does not track the ablation fold split. **U12-B TSR control, three attempts:** 12 h timeout writing nothing (55098053) → progress + per-condition checkpointing added, decisive pair ordered first → instant `TypeError` from `W._git_sha()` called without its root, a **latent bug the timeout had been hiding** (55405355), fixed via a tested `_provenance()` → third submission running (55408255). **Two wrapper bugs that exited 0 while skipping the science:** `--symbolic-csv` inferred from a file existing sent all five regenerative jobs into the checkpoint-free fallback (90 s, no network loaded) while an unconditional `echo` named a manifest that was never written. Both fixed; wrappers now assert artifacts instead of announcing them. Sync + fetch scripts extended (`regenerative_phase3.py`, 2 sbatch, 3 test stems, U13 result pulls). Suite **741 → 759 passed / 1 skipped**. Band-notch `max_rel_err = 1.0` diagnosed as an unfilled-`best` sentinel at `degradation.py:520` but **NOT yet fixed** — it gates Update 12 §5. All uncommitted per policy.
- 2026-07-30 — **U12-B returned and TSR is closed as a negative: the delay-lock is a construction artifact.** Job 55408255 (third submission) finished; user fetched it. **Verified before reading, per the lesson that exit 0 is not evidence:** `provenance.complete: true` — written only at `tsr_control_phase3.py:244` after all three conditions return, while the per-condition checkpoints write `false`, so a wall-killed partial cannot masquerade as finished — plus three conditions × 24/24 windows, `n_lock_tested` 24 everywhere, identical window indices across conditions, and a genuine untrained arm (7 submodules re-initialised, mean weight diff 0.426). Only gap: `git_sha: null`. **Numbers:** trained/true-τ lock_z **+2.81** (79.2% > 2) · untrained/true-τ **+4.62** (100% > 2) · trained/shuffled-τ **+0.88** (16.7% > 2). **The decisive pair inverts: the untrained net locks harder than the trained one in 22/24 paired windows** (mean +1.80, sign test p = 3.6e-5) while clearing the significance gate in 24/24, so this is not a trained signal measured against noise. Trained beats shuffled 19/24 (p = 6.6e-3), but τ-specificity is what the artifact predicts — the delayed channel *is* the signal shifted by that window's own τ — so it discriminates "this τ vs any lag", not "learned vs constructed". Two corroborations: the trained lock is **stronger on stable windows (3.52) than chatter (2.10)**, backwards for a regenerative mechanism; and lock_z falls with τ in both arms (corr −0.64 / −0.43), the documented autocorrelation-decay bias. **Blocker A2 resolved by application, not by code:** the corrected rule (significance gate, window minimum, shuffled arm consulted) was applied by hand and *strengthens* the printed `ARTIFACT SUSPECTED`; `tsr_control_verdict()` is still the unfixed version and is now optional, since the only result it will ever judge has been judged. New minor item A3: the τ-shuffle is not a derangement (2/24 self-shuffles; excluding them sharpens the result). **Update 12 §4 rewritten from pre-committed branch text to result**, §8 and the header note updated, `.docx` re-rendered. The delay claim is untouched — it never rested on TSR; it rests on ablation, per-window τ-sweep, τ-jitter, and the network-free SINDy-DDE recovery. All uncommitted per policy.
- 2026-07-30 (cont.) — **U12/U13 simplified, band-notch fixed, U14 started, and the duplicate-sweep mistake turned into a measured repeatability floor.** Four parallel subagents; **two were killed mid-flight by a session limit** and that is recorded here because the survivors' output had to be verified rather than trusted. **(1) U12 + U13 language simplified** for a non-ML reader (advisor-facing). Verified independently, not accepted on the agents' word: numeric-token multisets identical before and after (**U12 219 tokens, U13 140, zero drift**), all headings/tables/figure paths preserved, 0 em dashes, every negative and self-correction still reads as one. U12's longest sentence 93 → 35 words. Both `.docx` re-rendered (U12 1 figure, U13 3). **(2) Band-notch code fix LANDED** (details in item A above) — the agent died right before its full-suite check, so I ran it: **769 passed / 1 skipped / 0 failed** (16m57s), up from 759, and `git show HEAD:` confirms **no test was deleted or weakened** (`test_degradation.py` 9→35, `test_fuzz_phase3_cli.py` 1→8 vs the last commit, zero removed). **GL re-run submitted by user: jobs 55534785–89.** **(3) U14 STARTED** — `research_update_14.{md,docx}`, sections 1/2/3/5/6/7 written from verified artifacts, §4 explicitly **design-only with no code**. Its §4 edge-overlap numbers (101 distinct / 17 in ≥2 folds / **0 in all five**, pairwise 2–4) were re-derived from the five stored manifests keyed on `(kan, in, out)` and reproduce the plan exactly. Figure `u14_repeatability.png` via `scripts/plot_update_14.py` (2 panels, palette validated, layout checked by rendering and looking at it — first render had two label collisions). **(4) The repeatability finding** (status board above): floor 0.0092 AUC, both retrains under it, quantile "improvement" withdrawn from U12 §6 with a dated correction. **(5) CAUSE OF THE DIVERGENCE FOUND** — user ran `sacct`, nodes stored in `results/phase3/slurm_nodes.json` and consumed by the audit: **hardware is refuted** (3 of 9 cross-node pairs reproduced bit-exactly), and the real cause is that the trainer seeds RNGs but never requests deterministic algorithms while running `compile: true`. **New open item: set the determinism flags + record node/library versions in `summary.json`** — deliberately NOT done unasked, since it changes training behaviour. **(6) U14-B activation-mode build relaunched** after the limit reset. All uncommitted per policy.
- 2026-07-30 (cont. 2) — **U14-B activation-mode module BUILT (local, TDD) — and the build itself found the metric's floor before any real number was read.** `src/pri2m/xai/activation_modes.py` + `tests/test_activation_modes.py` (**26 tests**), plus a `component="full"|"spline"|"base"` selector on `KANLinear.edge_function`. **Full suite 769 → 795 passed / 1 skipped / 0 failed.** **Byte-identity independently verified by me, not taken on report:** `edge_function()` with no args equals the legacy `spline_curves + base_weight*SiLU(x)` formula with **max abs diff exactly 0.0**, so the spline atlas and every symbolic fit already validated are untouched; `spline + base == full` exactly; and the base path really is a one-parameter family (all base curves the same unit shape), which is the justification for clustering on spline only. **THE FINDING THAT MATTERS: the agreement metric has a high untrained floor.** On the synthetic test fixtures a randomly initialised net already scores `prototype_fit ≈ 0.63` (1.0 = perfect), because 8 spline coefficients over one shared basis cannot produce curves far from every prototype. **Reading a trained 0.7 as a result would have been another U12-B.** `identity_agreement` is the discriminating statistic (untrained 0.33–0.49 ≈ chance vs 1.00 on a structured positive control). Two more defects found and guarded during the build: **(a) a stratum dead in the data scored cleanly** — every `severity.side` edge at importance 0, yet normalised dead curves cluster like live ones and a quantile gate (threshold 0) passed all of them; now a hard `importance > 0` floor in every gate mode + `n_zero_importance` sentinel + UNDECIDED. **(b) `k` is not free** — above the true number of shapes, k-means splits a real cluster and `identity_agreement` collapses to chance while `prototype_fit` stays ~1.0, so the run MUST sweep k. All four caveats ship in `activation_modes.CAVEATS`. Verdict mirrors the TSR precedent: explicit `threshold=0.8`, `min_seeds=20`, `untrained_p95 >= threshold*|trained| ⇒ artifact`, UNDECIDED is `None` and explicitly NOT a pass; default `n_seeds=32`. **NOT RUN — no checkpoints on local disk.** Owed: the runner + sbatch (agent scope excluded them; the call order is in its report), a k-sweep, and step 8 mode-ablation against held-out 0.908 ± 0.013. U14 §4 written as "built and tested, no results", with the 0.63 flagged inline as a synthetic-fixture property. Uncommitted per policy.
- 2026-07-30 (cont. 3) — **Band-notch CLOSED as a diagnosed structural negative; U12 §5 rewritten; a wrong explanation of mine caught and corrected.** All five GL folds (55534785–89) returned and were read; numbers and mechanism on item A above. **Suite 795 → 796 passed / 1 skipped / 0 failed** (+1: the degenerate-denominator test). **Two things I got wrong today, both caught before they reached the report.** (1) I predicted in item A that the fix would "likely convert §5 into a result." It did not — it converted a hidden failure into a legible one, which is a real gain but not the one I forecast. (2) I told the user the AUC direction (chatter notch hurt more in 4/5 folds, mean +0.0042) was confounded because the chatter notch removed more energy in every fold. **I then checked it instead of leaving it as a plausible story and it runs BACKWARDS: corr(energy excess, ΔAUC) = −0.744** (fold 2 has the largest excess 7.38% and the smallest gap −0.0003; fold 3 has a middling 4.89% and the largest +0.0092). The correct reason the direction is unreadable is that **the pairing is broken at the window level** — a 3.9–7.4% batch-mean difference sitting on top of a ~97% per-window mismatch, the mean hiding the failure exactly as in every other defect this month. §5 now says that, and says explicitly that the direction can be neither explained away nor claimed. **Floor-value selection was also gotten wrong once and fixed by test:** `min_chatter_frac` at 1e-3 discarded a fixture window that had removed 0.0017 and matched to 0.4%; measured the two populations (degenerate ≈3.2e-6 vs smallest legitimate 8.09e-4) and set it to 1e-4, ~30× above one and ~8× below the other, with the evidence written into the constant's comment. `n_degenerate_chatter` is reported separately from `n_unsolved`; `unusable_fraction` drives the veto. U12 `.docx` re-rendered. **Open, unchanged:** determinism flags + hardware/env recording in `summary.json` (user's call, changes training behaviour); U14-B runner + sbatch + k-sweep + mode ablation; U14 §4 still a placeholder. All uncommitted per policy.
- 2026-07-30 (cont. 4) — **First commit since `c2392b2e` (user instruction), and the U14-B runner + sbatch wired.** **COMMIT `900b4e2d`, 110 files, 17.5 MB, user's name only per [[git_commit_policy]].** ⚠️ **A plain `git add -A` would have been destructive and was NOT done:** the working tree carries **26,241 unstaged file deletions** — the repo root's old content (`.gitignore`, `.devcontainer/`, `Dataset 1/AutogluonModels/**`) is gone from disk while HEAD still tracks it, so only `SINDy_DDE/` remains. **Those deletions are still unstaged and are a separate decision the user has not made.** Staged `SINDy_DDE` only; verified `--diff-filter=D` staged count was **0** before committing. `SINDy_DDE/.gitignore` is intact and kept the 3.8 GB `archive/` out. One Word lock file (`~$search_update_6.docx`) slipped in and was removed by amend; `~$*` added to `SINDy_DDE/.gitignore`. **U14-B runner:** `scripts/activation_modes_phase3.py` — provenance built BEFORE the run (the `_git_sha` lesson), **per-k checkpointing** so a wall kill leaves a readable partial sweep, `--k-sweep` defaulting to 2,3,4,5,6 because a single k is not a result, `--gate-min-importance` documented as preferable to a quantile when comparing arms (a quantile passes the same fraction of both by construction, hiding the very mismatch matched-N exists to expose), and a final block that **asserts** the manifest and its `complete` flag rather than announcing a path. Verified by direct call on trained-like synthetic models: both gate paths run end-to-end and the UNDECIDED path is reachable and labelled "NOT a pass" (the re-init guard also correctly fired on un-perturbed fixtures). **`slurm/phase3_activation_modes.sbatch`** — CPU-only on purpose (no backward pass; the GPU queue is the scarce resource), 8 cpus / 48 G / 12 h, all five published run ids, checkpoint existence checked up front, artifact asserted at the end. **Sync/fetch gaps closed BEFORE they bit** (the 2026-07-25 lesson): sync now ships the runner, `repeatability_audit.py`, the sbatch and `test_activation_modes`; fetch pulls `activation_modes/` and its verification block **asserts `provenance.complete`**, since a partial manifest is written after every k and presence alone proves nothing. `bash -n` clean on both; embedded python compiles. **User decisions this round:** commit now (done); U14-B = write the runner, user runs on GL (done, awaiting their submit).
- 2026-07-30 (cont. 5) — **U14-B RAN 5/5 and is a k-DEPENDENT NEGATIVE; plus a self-inflicted local data loss, fully recovered.** **(1) Runner bug found by the cluster:** job 55552794 died in k=2 with `TypeError: _progress() takes 1 positional argument but 3 were given` — `activation_modes._tick` calls `progress(stage, done, total)`. **My smoke test never passed a progress callback**, so it exercised everything except the failing path. Fixed (`349f8bdd`), `_progress` moved to module level, and **`tests/test_activation_modes_runner.py` (6 tests) now covers the runner/library seam** — arity, the real analysis driving the real callback, provenance JSON-safety, and arg errors firing *before* the expensive checkpoint loads. Verified by reintroducing the bug: 2 of 6 fail, both pass on restore. Suite **796 → 802 passed / 1 skipped**. **(2) THE RESULT (resubmission, manifest 19:21, `complete: true`, 32 untrained replicates, 256 windows, matched N = 77 edges/fold, min untrained weight diff 0.42, 7 submodules reinit):** the verdict **flips with k, and each stratum flips at a different k** — `encoder.L0.x_delay` survives k=2,3 then ARTIFACT k≥4; `encoder.L0.x_t` survives only k=2; `severity.side` survives k=2,3,4 then ARTIFACT k≥5. At k=2 all three survive, at k=6 none do. **Mechanism is in the numbers: the null climbs ~2× faster than the signal** (x_delay trained 0.804→0.911, untrained p95 0.567→0.777), so the ratio rises monotonically 0.706→0.853 and crosses the pre-committed 0.8 between k=3 and k=4 (k=3 clears by **0.0103** — a coin on its edge). **NOT CLAIMABLE: the threshold was pre-committed, k was NOT**, so picking k=2/3 is choosing the analysis after seeing the answer. **What IS true: margins are positive in every stratum at every k (0.134–0.259), no exceptions, no sign changes** — the trained nets really are more self-consistent than random; there is just no measurement of it that survives its own control. **Fourth attribution-side claim to fail its control** (TSR delay-lock, concept probe, band-notch, this). Delay claim untouched. **(3) MY REPORTING BUG, in two places:** the runner's summary print and the fetch verification both read a **top-level `verdict` key that does not exist** (verdicts are per-stratum) → printed `UNDECIDED / trained=None` for every k, which looked exactly like a scientific null. **The manifest was always correct.** Both fixed to iterate `strata`. **(4) DATA LOSS, mine:** `rm -rf SINDy_DDE` during a branch rebuild deleted **gitignored** files git could not restore — all 12 figures in `results/figures/research_update_assets/`, `repeatability.json`, `slurm_nodes.json`, and the whitebox fold manifests. **Fully recovered:** `slurm_nodes.json` rewritten from the sacct capture in-session, `repeatability.json` regenerated by the audit script (identical: floor 0.0092, 4/10 bit-identical), `u14_repeatability.png` re-plotted, and **all 12 other figures extracted back out of the delivered `.docx` files and verified byte-identical round-trip**. Still gone locally: whitebox fold manifests + conformal + concept_probes + heldout — **the fetch script never covered them**, now extended to do so. **Lesson: `rm -rf` on a tracked directory also destroys the untracked artifacts inside it, and git restores none of them.**
