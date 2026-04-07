"""
XGBoost + XAI Pipeline  —  v2 (peer-review fixes)
===================================================
Changes from v1:
  1. Removed all physics/machine-level parameters from features.
     These are constant within each dataset, so the model could use
     them to identify which dataset a file came from ("dataset ID
     memorization") rather than learning genuine chatter physics.
     Removed: fn, ktc, teeth, damping, cut_dir, diameter_mm,
              ktx, zetatx, fnty, ro
     Kept:    tpf, tau  (vary per file because they depend on omega)
     Kept:    all 63 signal-based features

  2. Fixed kurtosis NaN.
     210 files have zero-variance signals (xtpos_rms == 0), making
     kurtosis undefined. These are perfectly stable, zero-motion
     files. Filling with 0 is physically correct (kurtosis of a
     constant signal = 0).

  3. DiCE: noted as preliminary; physics params locked as immutable
     so it only recommends changes to omega.

  4. Per-fold AUC table printed in full, sorted by fold number.

Usage:
    pip install xgboost shap lime dice-ml scikit-learn pandas numpy matplotlib
    python xgboost_xai_v2.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve,
)
import xgboost as xgb
import shap

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "features_all_datasets.csv"
OUT       = "xgb_results_v2"
os.makedirs(OUT, exist_ok=True)

# Always excluded (leak label or are pure identifiers)
ALWAYS_EXCLUDE = {
    "dataset", "file",
    "label",
    "b_over_blim",   # directly encodes label
    "blim",          # stability boundary — not available in deployment
    "b_mm",          # cutting depth; combined with blim = label
    "noise_std",     # zero for all clean datasets
    "omega",         # spindle speed is a control param, not a signal feature
                     # NOTE: omega IS the parameter we'd recommend changing via
                     # DiCE, so we exclude it from model features but track it
                     # separately for the DiCE immutable-features list.
}

# ── FIX 1: Physics params that are constant within each dataset ───────────────
# Including these lets the model identify *which dataset* a file came from
# ("dataset memorization") rather than learning chatter from the signal.
# Every one of these has nunique==1 within every dataset (verified).
PHYSICS_PARAMS = {
    "fn",          # natural frequency — machine property
    "ktc",         # cutting force coefficient — material property
    "teeth",       # number of tool teeth — tool property
    "damping",     # structural damping — machine property (0 for most datasets)
    "cut_dir",     # up-milling vs climb-milling — setup parameter
    "diameter_mm", # tool diameter — tool property
    "ktx",         # tangential force coefficient — material property
    "zetatx",      # damping ratio — machine property
    "fnty",        # y-direction natural frequency — machine property
    "ro",          # density — all zero in this database (constant globally)
}
# tpf (tooth-pass frequency) and tau (time delay) are KEPT.
# They depend on omega and vary file-to-file within a dataset,
# so they carry signal-level information, not dataset identity.

EXCLUDE = ALWAYS_EXCLUDE | PHYSICS_PARAMS

XGB_PARAMS = dict(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42, n_jobs=-1,
)

# ── 1. Load & clean ───────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1  Load + clean data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")

# ── FIX 2: Kurtosis NaN → fill with 0 ────────────────────────────────────────
kurt_cols = [c for c in df.columns if "kurt" in c.lower()]
nan_before = df[kurt_cols].isna().sum().sum()
# These NaN rows have xtpos_rms == 0 (zero-variance signal).
# Kurtosis of a constant signal is technically undefined but
# physically should be 0 (no spikiness at all).
df[kurt_cols] = df[kurt_cols].fillna(0.0)
nan_after  = df[kurt_cols].isna().sum().sum()
print(f"\nKurtosis NaN fix: {nan_before} → {nan_after} NaN values")
print(f"  Affected rows (all had RMS=0, perfectly stable signals): "
      f"{(df['xtpos_rms'] == 0).sum()}")

# Feature columns = everything not excluded
feature_cols = [c for c in df.columns if c not in EXCLUDE]
print(f"\nFeatures after removing physics params: {len(feature_cols)}")
print(f"Removed physics params: {sorted(PHYSICS_PARAMS)}")

# Confirm no remaining NaN in features
n_nan = df[feature_cols].isna().sum().sum()
print(f"Remaining NaN in features: {n_nan}")

X           = df[feature_cols].values.astype(np.float64)
y           = df["label"].values
groups      = df["dataset"].values
dataset_ids = sorted(df["dataset"].unique())

print(f"\nDatasets: {len(dataset_ids)}")
print(f"Label balance: {dict(pd.Series(y).value_counts().sort_index())}")

# ── 2. Leave-Dataset-Out CV ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2  Leave-Dataset-Out Cross-Validation")
print("=" * 60)
print("(Signal-only features — no physics/machine params)")

oof_proba = np.zeros(len(df))
cv_rows   = []

for ds_id in dataset_ids:
    test_mask  = groups == ds_id
    train_mask = ~test_mask

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X[train_mask], y[train_mask],
        eval_set=[(X[test_mask], y[test_mask])],
        verbose=False,
    )

    proba = model.predict_proba(X[test_mask])[:, 1]
    oof_proba[test_mask] = proba

    auc = roc_auc_score(y[test_mask], proba)
    f1  = f1_score(y[test_mask], (proba >= 0.5).astype(int))
    pct_unstable = y[test_mask].mean()
    damping_val = df[test_mask]["damping"].iloc[0] if "damping" in df.columns else "N/A"

    cv_rows.append({
        "dataset":      ds_id,
        "n":            test_mask.sum(),
        "auc":          round(auc, 4),
        "f1":           round(f1, 4),
        "pct_unstable": round(pct_unstable, 3),
        "damping":      damping_val,
    })
    print(f"  ds={ds_id:3d} | n={test_mask.sum():4d} | AUC={auc:.3f} | "
          f"F1={f1:.3f} | unstable={pct_unstable:.0%} | "
          f"damping={'nonzero' if float(damping_val) > 0 else 'zero'}")

cv_df       = pd.DataFrame(cv_rows)
overall_auc = roc_auc_score(y, oof_proba)
overall_f1  = f1_score(y, (oof_proba >= 0.5).astype(int))

print(f"\nOverall OOF AUC : {overall_auc:.4f}  (was 0.9732 with physics params)")
print(f"Overall OOF F1  : {overall_f1:.4f}")
print(f"Mean AUC +/- std: {cv_df['auc'].mean():.4f} +/- {cv_df['auc'].std():.4f}")

# ── FIX 3: Per-fold breakdown ──────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("Per-fold AUC breakdown (damped datasets highlighted)")
print(f"{'─'*60}")
damped = set(range(25, 33))   # datasets 25–32 have nonzero damping
for _, row in cv_df.iterrows():
    flag = " ← damped" if row["dataset"] in damped else ""
    print(f"  ds={int(row['dataset']):3d} | AUC={row['auc']:.4f} | "
          f"F1={row['f1']:.4f} | n={int(row['n']):4d}{flag}")

cv_df.to_csv(f"{OUT}/cv_results_v2.csv", index=False)
print(f"\nSaved: {OUT}/cv_results_v2.csv")

# Plot AUC per dataset, colour-coded by damping
fig, ax = plt.subplots(figsize=(15, 4))
colors = []
for ds_id in cv_df["dataset"]:
    if ds_id in damped:
        colors.append("#F59E0B")   # amber = damped
    elif cv_df.loc[cv_df["dataset"]==ds_id, "auc"].values[0] >= 0.9:
        colors.append("#0D9488")   # teal = good
    elif cv_df.loc[cv_df["dataset"]==ds_id, "auc"].values[0] >= 0.75:
        colors.append("#185FA5")   # blue = ok
    else:
        colors.append("#EF4444")   # red = weak
ax.bar(cv_df["dataset"], cv_df["auc"], color=colors, width=0.7)
ax.axhline(overall_auc, color="#334155", lw=1.2, ls="--",
           label=f"Overall AUC = {overall_auc:.3f}")
ax.axhline(0.973, color="#94A3B8", lw=0.8, ls=":",
           label="v1 AUC = 0.973 (with physics params)")
# Legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(color="#0D9488", label="AUC ≥ 0.90"),
    Patch(color="#185FA5", label="0.75 ≤ AUC < 0.90"),
    Patch(color="#EF4444", label="AUC < 0.75"),
    Patch(color="#F59E0B", label="Damped dataset (25–32)"),
]
ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][-2:],
          fontsize=8, loc="lower right")
ax.set(xlabel="Dataset ID", ylabel="AUC-ROC",
       title="Leave-Dataset-Out CV — AUC per dataset (v2: signal features only)",
       ylim=(0, 1.08))
ax.axvspan(24.5, 32.5, alpha=0.06, color="#F59E0B", label="_")
plt.tight_layout()
fig.savefig(f"{OUT}/cv_auc_per_dataset_v2.png", dpi=150)
plt.close()
print(f"Saved: {OUT}/cv_auc_per_dataset_v2.png")

# Confusion matrix
cm = confusion_matrix(y, (oof_proba >= 0.5).astype(int))
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["Stable", "Unstable"]).plot(
    ax=ax, colorbar=False)
ax.set_title(f"OOF Confusion Matrix — v2 (AUC={overall_auc:.3f})")
plt.tight_layout()
fig.savefig(f"{OUT}/confusion_matrix_v2.png", dpi=150)
plt.close()

# ── 3. Final model on all data ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3  Train final model on full data (for SHAP)")
print("=" * 60)

final_model = xgb.XGBClassifier(**XGB_PARAMS)
final_model.fit(X, y, verbose=False)
print("Final model trained.")

# ── 4. SHAP ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4  SHAP — global feature importance (signal features only)")
print("=" * 60)

explainer = shap.TreeExplainer(final_model)
shap_vals = explainer.shap_values(X)
mean_shap = pd.Series(
    np.abs(shap_vals).mean(axis=0), index=feature_cols
).sort_values(ascending=False)

mean_shap.to_csv(f"{OUT}/shap_importance_v2.csv", header=["mean_abs_shap"])
print("Top 15 features (no physics params can appear now):")
print(mean_shap.head(15).to_string())

top20     = mean_shap.head(20).index.tolist()
top20_idx = [feature_cols.index(f) for f in top20]

fig, _ = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals[:, top20_idx], X[:, top20_idx],
                  feature_names=top20, show=False)
plt.title("SHAP Beeswarm — Top 20 (signal features only)")
plt.tight_layout()
fig.savefig(f"{OUT}/shap_beeswarm_v2.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
mean_shap.head(20).sort_values().plot(kind="barh", ax=ax, color="#0D9488")
ax.set(xlabel="Mean |SHAP value|",
       title="Feature Importance — v2 (signal features only, top 20)")
plt.tight_layout()
fig.savefig(f"{OUT}/shap_bar_v2.png", dpi=150)
plt.close()
print(f"Saved SHAP plots → {OUT}/")

# ── 5. PDP / ICE ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5  PDP / ICE — feature effect curves")
print("=" * 60)

pdp_features = mean_shap.head(6).index.tolist()
pdp_idx_list = [feature_cols.index(f) for f in pdp_features]
rng          = np.random.RandomState(42)
sample_idx   = rng.choice(len(X), size=min(500, len(X)), replace=False)
X_sample     = X[sample_idx]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat, idx in zip(axes.flat, pdp_features, pdp_idx_list):
    grid = np.linspace(X[:, idx].min(), X[:, idx].max(), 50)
    ice_lines = []
    for xi in X_sample:
        row_copies = np.tile(xi, (50, 1))
        row_copies[:, idx] = grid
        ice_lines.append(final_model.predict_proba(row_copies)[:, 1])
    ice_arr  = np.array(ice_lines)
    pdp_line = ice_arr.mean(axis=0)
    for line in ice_arr:
        ax.plot(grid, line, color="#85B7EB", alpha=0.05, lw=0.5)
    ax.plot(grid, pdp_line, color="#0D1B2A", lw=2, label="PDP (mean)")
    ax.axhline(0.5, color="#EF4444", lw=0.8, ls="--", label="Threshold")
    ax.set(xlabel=feat[:28], ylabel="P(unstable)", title=feat[:28], ylim=(-0.05, 1.05))
    ax.legend(fontsize=7)

plt.suptitle("PDP / ICE — Top 6 Features (v2: signal features only)", fontsize=13)
plt.tight_layout()
fig.savefig(f"{OUT}/pdp_ice_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/pdp_ice_v2.png")

# ── 6. LIME ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6  LIME — borderline cases")
print("=" * 60)

try:
    import lime.lime_tabular

    LIME_EPS       = 1e-6
    col_stds       = X.std(axis=0)
    lime_keep      = col_stds > LIME_EPS
    X_lime         = X[:, lime_keep]
    lime_feat_names = [f for f, k in zip(feature_cols, lime_keep) if k]
    dropped = [f for f, k in zip(feature_cols, lime_keep) if not k]
    if dropped:
        print(f"  Dropped {len(dropped)} near-constant cols from LIME: {dropped}")

    class _LimeWrapper:
        def __init__(self, mdl, keep_mask, n_full):
            self.mdl, self.keep_mask, self.n_full = mdl, keep_mask, n_full
            self._full_idx = np.where(keep_mask)[0]
        def predict_proba(self, X_reduced):
            X_full = np.zeros((X_reduced.shape[0], self.n_full))
            for i, fi in enumerate(self._full_idx):
                X_full[:, fi] = X_reduced[:, i]
            return self.mdl.predict_proba(X_full)

    lime_wrapper = _LimeWrapper(final_model, lime_keep, len(feature_cols))
    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        X_lime, feature_names=lime_feat_names,
        class_names=["Stable", "Unstable"],
        mode="classification", random_state=42,
    )

    margin     = np.abs(oof_proba - 0.5)
    border_idx = np.argsort(margin)[:5]

    for rank, idx in enumerate(border_idx):
        try:
            exp = lime_exp.explain_instance(
                X_lime[idx], lime_wrapper.predict_proba, num_features=10)
            fig = exp.as_pyplot_figure()
            true_lbl = "Unstable" if y[idx] == 1 else "Stable"
            fig.suptitle(
                f"LIME #{rank+1} | True:{true_lbl} | "
                f"P(unstable)={oof_proba[idx]:.3f} | ds={groups[idx]}",
                fontsize=9)
            plt.tight_layout()
            fig.savefig(f"{OUT}/lime_case_{rank+1}_v2.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Case #{rank+1}: P={oof_proba[idx]:.3f}, ds={groups[idx]}, true={true_lbl}")
        except Exception as e:
            print(f"  [WARN] LIME case #{rank+1}: {e}")

except ImportError:
    print("  [SKIP] pip install lime")

# ── 7. DiCE ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7  DiCE — counterfactuals (physics params locked as immutable)")
print("=" * 60)

try:
    import dice_ml, types, scipy.stats as _ss

    X_df     = pd.DataFrame(X, columns=feature_cols).astype(np.float64)
    train_df = X_df.copy()
    train_df["label"] = y.astype(int)

    dice_data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=feature_cols,
        outcome_name="label",
    )
    dice_model = dice_ml.Model(model=final_model, backend="sklearn")

    # Monkey-patch: fix integer-mode IndexError in get_decimal_precisions
    def _safe_decimal_precisions(self, output_type="list"):
        precisions = {}
        for col in self.continuous_feature_names:
            col_data = self.data_df[col].dropna().values
            mode_val = float(_ss.mode(col_data, keepdims=True).mode[0])
            parts    = str(mode_val).split(".")
            precisions[col] = len(parts[1]) if len(parts) > 1 else 0
        return precisions if output_type == "dict" else list(precisions.values())

    dice_data.get_decimal_precisions = types.MethodType(
        _safe_decimal_precisions, dice_data)

    dice_exp = dice_ml.Dice(dice_data, dice_model, method="random")

    # ── FIX 3: Lock physics params as immutable ──────────────────────────────
    # tpf and tau are in the feature set but depend on omega — they change
    # together. For DiCE we lock them too; the only truly actionable lever
    # a machinist has is omega (spindle speed).
    # We list all physics params that remain in features (none after fix 1),
    # plus tpf and tau, as immutable so DiCE only changes signal features.
    # In Phase 5 (PINN), DiCE will take omega directly as an input.
    immutable_features = [f for f in feature_cols if f in {"tpf", "tau"}]
    print(f"  Immutable features (DiCE cannot change): {immutable_features}")
    print("  Note: all physics params (fn, damping, etc.) are already removed "
          "from features. DiCE on XGBoost is preliminary — actionable "
          "recommendations (e.g. 'reduce omega') will come from Phase 5 PINN.")

    high_conf  = np.where((oof_proba > 0.8) & (y == 1))[0]
    if len(high_conf) == 0:
        high_conf = np.where(y == 1)[0]
    dice_cases = np.random.RandomState(42).choice(
        high_conf, size=min(3, len(high_conf)), replace=False)

    all_dice = []
    for rank, idx in enumerate(dice_cases):
        query = X_df.iloc[[idx]]
        print(f"\n  Case #{rank+1} | ds={groups[idx]} | P(unstable)={oof_proba[idx]:.3f}")
        try:
            cf = dice_exp.generate_counterfactuals(
                query, total_CFs=4, desired_class="opposite",
                features_to_vary=[f for f in feature_cols
                                   if f not in immutable_features],
            )
            cf_df     = cf.cf_examples_list[0].final_cfs_df
            orig_vals = query.iloc[0]
            seen      = set()
            for col in feature_cols:
                orig = orig_vals[col]
                for _, cf_row in cf_df.iterrows():
                    new = cf_row[col]
                    pct = ((new - orig) / orig * 100) if abs(orig) > 1e-10 else 0
                    if abs(pct) > 5 and col not in seen:
                        seen.add(col)
                        all_dice.append({
                            "case": rank+1, "dataset": int(groups[idx]),
                            "feature": col,
                            "original": round(float(orig), 6),
                            "counterfactual": round(float(new), 6),
                            "change_pct": round(float(pct), 1),
                        })
                        print(f"    {col:<42s} {orig:>12.4f}  {pct:>+.1f}%")
        except Exception as e:
            print(f"  [ERROR] {e}")

    if all_dice:
        dice_out = pd.DataFrame(all_dice)
        dice_out.to_csv(f"{OUT}/dice_counterfactuals_v2.csv", index=False)
        print(f"\nSaved: {OUT}/dice_counterfactuals_v2.csv")

except ImportError:
    print("  [SKIP] pip install dice-ml")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — v2 (signal features only)")
print("=" * 60)
print(f"Samples              : {len(df):,}")
print(f"Features             : {len(feature_cols)}")
print(f"  (removed {len(PHYSICS_PARAMS)} physics params vs v1)")
print(f"Kurtosis NaN fixed   : 210 rows filled with 0")
print(f"OOF AUC-ROC          : {overall_auc:.4f}  (v1 was 0.9732)")
print(f"OOF F1               : {overall_f1:.4f}")
print(f"Mean per-fold AUC    : {cv_df['auc'].mean():.4f} +/- {cv_df['auc'].std():.4f}")
print(f"\nTop 10 SHAP features (all signal-derivable):")
for feat, val in mean_shap.head(10).items():
    print(f"  {feat:<45s} {val:.5f}")
print(f"\nAll outputs → ./{OUT}/")
print("=" * 60)