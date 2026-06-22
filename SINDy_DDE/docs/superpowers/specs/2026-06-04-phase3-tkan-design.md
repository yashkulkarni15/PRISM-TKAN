# Phase 3 T-KAN Design Spec

**Date:** 2026-06-04
**Status:** Approved (all six judgment calls accepted by user)
**Author:** Phase 3 brainstorming session

## Goal

Build a soundproof, model-agnostic Phase 3 training harness in PyTorch with a
Temporal Kolmogorov–Arnold Network (TKAN) as the first concrete model. The model
predicts milling-chatter behaviour on two staged heads — graded stability
severity (`b_over_blim` + binary chatter label) and trajectory reproduction —
with interpretability (XAI) treated as a first-class deliverable.

## Approved decisions

1. **Two heads, staged.** Severity head trainable now; trajectory head built but
   gated by `stage_gates.evaluate_trajectory_gate`; physics-residual head is a
   documented stub (blocked by the open Altintas–Budak constants).
2. **Trajectory = autoregressive rollout** of the downstream signals
   (`xtpos`, `Forcex`) seeded by an initial window + operating params, with a
   **hardcoded delay** `tau = round(fs / tpf)` (tooth-pass period). The kinematic
   delay does NOT require the blocked Altintas–Budak force constants.
3. **Self-built `KANLinear`** (B-spline + SiLU base, efficient-kan style). No
   external `pykan`/`efficient-kan` runtime dependency — full test + install
   control on Great Lakes. Exposes per-edge spline curves for XAI.
4. **Severity is multi-task**: weighted MSE on `b_over_blim` + weighted BCE on the
   binary chatter `label`.
5. **sigKAN included** as an optional, off-by-default trunk variant (path-signature
   weighting on the same KAN substrate).
6. **PyTorch** stack: `torch` (present), `mlflow`, `captum`, `shap`, `dice-ml`,
   `matplotlib`, optional `iisignature` (sigKAN only).

## Architecture

```
NPZ windows + phase3_dataset.csv + phase3_folds.json
   │
[1] Data layer ─ windowing · train-fold-only normalization · leakage-safe folds
   │  (window[L,C] + b_over_blim + label + sample_weight + metadata + tau)
   ▼
[2] KAN substrate ─ KANLinear(B-spline) → TKANCell(temporal, +delay) → TKANEncoder
   │                                          (optional sigKAN signature path)
   ▼
[3] Shared trunk → SeverityHead | TrajectoryHead(gated) | PhysicsResidual(stub)
   │  Phase3Model protocol; MLPBaseline control (matches `hardcoded_tau_mlp_mse`)
   ▼
[4] Harness ─ noise-weighted loss · grouped-fold CV · MLflow · stage_gate
   ▼
[5] XAI ─ KAN edge plots + symbolic · Captum IG · SHAP · DiCE (mode-conditioned)
   ▼
[6] GL packaging ─ sbatch wrappers + smoke config + keystone end-to-end test
```

## Leakage discipline (non-negotiable)

- Folds come from `phase3_folds.json` (parent-group keyed). The data loader calls
  `pri2m.phase3.splits.assert_no_mirror_leakage` on every fold split.
- Per-channel normalization statistics are fit on **training-fold windows only**.
- `exclude_supervised` rows get sample weight 0 on the severity target (the
  ~1,260 negative-blim + metrics-unavailable rows). Trajectory may still use their
  raw signal (no label needed).

## Targets & loss

- Severity per window inherits its file's `b_over_blim` and `label`; per-file score
  is the mean over the file's windows. Sample weight from `phase3_dataset.csv`.
- `total = w_sev·(w_reg·MSE(b/blim) + w_cls·BCE(label)) + w_traj·MSE(trajectory)`,
  all sample-weighted. JS spectral-loss term is a stubbed seam (later unlock).

## XAI (mode-conditioned)

- **KAN-native**: per-edge activation plots, edge pruning, symbolic summary of the
  severity head.
- **Attribution**: Captum Integrated Gradients (channel×time saliency) + SHAP on
  the per-file feature summary (team's prior tool).
- **DiCE counterfactuals** over **controllable params only `{omega, b_mm}`** via a
  tabular surrogate of the model's per-file severity score; mode-conditioned per
  `docs/phase2_5_noise_mode_filtering.md` (e.g. `broadband_noise_dominated` →
  "sensor needs attention" rollup rather than a parameter counterfactual).

## Testing — soundproof contract

TDD throughout; mirrors `tests/test_phase2_5_end_to_end.py`. The keystone is a
synthetic NPZ-tree end-to-end run of `scripts/train_phase3.py --smoke` for both
`mlp` and `tkan` that asserts determinism, no leakage, MLflow artifacts, a gate
report, and XAI artifacts. Each module ships its own unit suite.

## Out of scope (deferred)

- Physics-residual head implementation (Altintas–Budak constants owed by Prof. Abell).
- JS spectral-loss head (later `unlock_order` step; interface seam only).
- Learnable `tau` (final `unlock_order` step).
- Real-data training (runs on Great Lakes; this work is build + local synthetic tests + sbatch wrappers).

## References

- `configs/phase1_3.yaml` → `phase3.unlock_order`, `trajectory_gate`, `mlflow`.
- `src/pri2m/models/stage_gates.py` → trajectory gate (R²≥0.85, onset≤5%).
- `src/pri2m/phase3/` → leakage-safe splits, label hygiene, sample weights.
- Memory: `phase3_preflight_blockers.md` (blockers fixed), `gl_project_root.md`.
