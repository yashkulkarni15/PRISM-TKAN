"""
XGBoost + XAI Pipeline  —  v3
===================================================
Changes from v2:
  1. Removed norm_ features which implicitly encoded ktc.
  2. Documented tpf and tau soft leakage.
  3. Removed probability calibration.
  4. Unified model object usage.
  5. Fixed AUC bar chart damping colors.
  6. Evaluated and plotted multiple metrics for clean and noisy datasets.

Usage:
    pip install xgboost shap lime dice-ml scikit-learn pandas numpy matplotlib
    python xgboost_xai_v3.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, fbeta_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve,
)
import xgboost as xgb
import shap

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "/Users/yash/Downloads/PRI2M/Dataset 2/features_all_datasets.csv"
OUT       = "xgb_results_v3"
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
    
    # norm_ features encode ktc which is excluded as a physics param
    "norm_xtpos_rms", "norm_ytpos_rms",
    "norm_Forcex_rms", "norm_Forcey_rms",
}

# ── FIX 1: Physics params that are constant within each dataset ───────────────
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

# NOTE: tpf and tau are kept but are algebraic functions of omega (excluded).
# tpf = omega * teeth / 60; tau = 1/tpf.
# They are retained because they vary file-to-file and capture tooth-pass
# frequency information relevant to chatter. This soft coupling to omega
# is acknowledged and will be addressed in the PINN phase where omega
# is an explicit input.
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
df[kurt_cols] = df[kurt_cols].fillna(0.0)
nan_after  = df[kurt_cols].isna().sum().sum()
print(f"\nKurtosis NaN fix: {nan_before} → {nan_after} NaN values")
print(f"  Affected rows (all had RMS=0, perfectly stable signals): "
      f"{(df['xtpos_rms'] == 0).sum()}")

# Feature columns = everything not excluded
feature_cols = [c for c in df.columns if c not in EXCLUDE]
print(f"\nFeatures after removing physics params and norm_ features: {len(feature_cols)}")

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

    auc    = roc_auc_score(y[test_mask], proba) if len(np.unique(y[test_mask])) > 1 else np.nan
    f1     = f1_score(y[test_mask], (proba >= 0.5).astype(int), zero_division=0)
    recall = recall_score(y[test_mask], (proba >= 0.5).astype(int), zero_division=0)
    f2     = fbeta_score(y[test_mask], (proba >= 0.5).astype(int), beta=2, zero_division=0)
    pct_unstable = y[test_mask].mean()
    damping_val = df[test_mask]["damping"].iloc[0] if "damping" in df.columns else "N/A"

    cv_rows.append({
        "dataset":      ds_id,
        "n":            test_mask.sum(),
        "auc":          round(float(auc), 4) if not np.isnan(auc) else None,
        "f1":           round(f1, 4),
        "recall":       round(recall, 4),
        "f2":           round(f2, 4),
        "pct_unstable": round(pct_unstable, 3),
        "damping":      damping_val,
    })
    print(f"  ds={ds_id:3d} | n={test_mask.sum():4d} | AUC={auc:.3f} | "
          f"F1={f1:.3f} | F2={f2:.3f} | Recall={recall:.3f} | "
          f"unstable={pct_unstable:.0%} | "
          f"damping={'nonzero' if float(damping_val) > 0 else 'zero'}")

cv_df       = pd.DataFrame(cv_rows)
overall_auc    = roc_auc_score(y, oof_proba)
overall_f1     = f1_score(y, (oof_proba >= 0.5).astype(int), zero_division=0)
overall_recall = recall_score(y, (oof_proba >= 0.5).astype(int), zero_division=0)
overall_f2     = fbeta_score(y, (oof_proba >= 0.5).astype(int), beta=2, zero_division=0)

print(f"\nOverall OOF AUC    : {overall_auc:.4f}")
print(f"Overall OOF F1     : {overall_f1:.4f}")
print(f"Overall OOF F2     : {overall_f2:.4f}  ← primary metric (β=2 weights recall)")
print(f"Overall OOF Recall : {overall_recall:.4f}")
print(f"Mean AUC +/- std   : {cv_df['auc'].mean():.4f} +/- {cv_df['auc'].std():.4f}")

print(f"\n{'─'*60}")
print("Per-fold breakdown (damped datasets highlighted)")
print(f"{'─'*60}")
damped = set(cv_df.loc[cv_df["damping"].astype(float) > 0, "dataset"].astype(int).tolist())
for _, row in cv_df.iterrows():
    flag = " ← damped" if int(row["dataset"]) in damped else ""
    auc_str = f"{row['auc']:.4f}" if pd.notnull(row['auc']) else "N/A"
    print(f"  ds={int(row['dataset']):3d} | AUC={auc_str} | "
          f"F1={row['f1']:.4f} | F2={row['f2']:.4f} | "
          f"Recall={row['recall']:.4f} | n={int(row['n']):4d}{flag}")

cv_df.to_csv(f"{OUT}/cv_results_v3.csv", index=False)
print(f"\nSaved: {OUT}/cv_results_v3.csv")

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
ax.bar(cv_df["dataset"], cv_df["auc"].fillna(0), color=colors, width=0.7)
ax.axhline(overall_auc, color="#334155", lw=1.2, ls="--",
           label=f"Overall AUC = {overall_auc:.3f}")
ax.axhline(0.973, color="#94A3B8", lw=0.8, ls=":",
           label="v1 AUC = 0.973 (with physics params)")
from matplotlib.patches import Patch
legend_elements = [
    Patch(color="#0D9488", label="AUC ≥ 0.90"),
    Patch(color="#185FA5", label="0.75 ≤ AUC < 0.90"),
    Patch(color="#EF4444", label="AUC < 0.75"),
    Patch(color="#F59E0B", label="Damped dataset"),
]
ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][-2:],
          fontsize=8, loc="lower right")
ax.set(xlabel="Dataset ID", ylabel="AUC-ROC",
       title="Leave-Dataset-Out CV — AUC per dataset (v3: signal features only)",
       ylim=(0, 1.08))
ax.axvspan(24.5, 32.5, alpha=0.06, color="#F59E0B", label="_")
plt.tight_layout()
fig.savefig(f"{OUT}/cv_auc_per_dataset_v3.png", dpi=150)
plt.close()
print(f"Saved: {OUT}/cv_auc_per_dataset_v3.png")

# Confusion matrix
cm = confusion_matrix(y, (oof_proba >= 0.5).astype(int))
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["Stable", "Unstable"]).plot(
    ax=ax, colorbar=False)
ax.set_title(f"OOF Confusion Matrix — v3 (AUC={overall_auc:.3f})")
plt.tight_layout()
fig.savefig(f"{OUT}/confusion_matrix_v3.png", dpi=150)
plt.close()

# ── 3. Final model on all data ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3  Train final model on full data")
print("=" * 60)

final_model = xgb.XGBClassifier(**XGB_PARAMS)
final_model.fit(X, y, verbose=False)
print("Final model trained.")

final_model.save_model(f"{OUT}/xgb_final_v3.json")
import pickle
with open(f"{OUT}/xgb_final_v3.pkl", "wb") as f:
    pickle.dump(final_model, f)
print(f"Saved: {OUT}/xgb_final_v3.json")
print(f"Saved: {OUT}/xgb_final_v3.pkl")

# Calibration removed in v3: fitting CalibratedClassifierCV on the same
# training data with cv="prefit" is leakage. Raw XGBoost probabilities
# are used throughout. Calibration can be revisited in Phase 5 (PINN).

# ── 4. SHAP ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4  SHAP — global feature importance (signal features only)")
print("=" * 60)

booster = final_model.get_booster()
dtrain = xgb.DMatrix(X, feature_names=feature_cols)
shap_vals_full = booster.predict(dtrain, pred_contribs=True)
shap_vals = shap_vals_full[:, :-1]
mean_shap = pd.Series(
    np.abs(shap_vals).mean(axis=0), index=feature_cols
).sort_values(ascending=False)

mean_shap.to_csv(f"{OUT}/shap_importance_v3.csv", header=["mean_abs_shap"])
print("Top 15 features:")
print(mean_shap.head(15).to_string())

top20     = mean_shap.head(20).index.tolist()
top20_idx = [feature_cols.index(f) for f in top20]

fig, _ = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals[:, top20_idx], X[:, top20_idx],
                  feature_names=top20, show=False)
plt.title("SHAP Beeswarm — Top 20 (signal features only)")
plt.tight_layout()
fig.savefig(f"{OUT}/shap_beeswarm_v3.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
mean_shap.head(20).sort_values().plot(kind="barh", ax=ax, color="#0D9488")
ax.set(xlabel="Mean |SHAP value|",
       title="Feature Importance — v3 (signal features only, top 20)")
plt.tight_layout()
fig.savefig(f"{OUT}/shap_bar_v3.png", dpi=150)
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

plt.suptitle("PDP / ICE — Top 6 Features (v3: signal features only)", fontsize=13)
plt.tight_layout()
fig.savefig(f"{OUT}/pdp_ice_v3.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/pdp_ice_v3.png")

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
            fig.savefig(f"{OUT}/lime_case_{rank+1}_v3.png",
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

    immutable_features = [f for f in feature_cols if f in {"tpf", "tau"}]
    print(f"  Immutable features (DiCE cannot change): {immutable_features}")

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
        dice_out.to_csv(f"{OUT}/dice_counterfactuals_v3.csv", index=False)
        print(f"\nSaved: {OUT}/dice_counterfactuals_v3.csv")

except ImportError:
    print("  [SKIP] pip install dice-ml")

# ── Phase 3: Noise Robustness Testing ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3  Noise Robustness Testing")
print("=" * 60)
print("NOTE: Noisy datasets 61-90 replicate physics of clean datasets 1-30 only.")
print("Datasets 31-60 have no noisy counterparts. Noise robustness results reflect")
print("seen physics under noise, not unseen physics. Generalization to unseen")
print("physics is evaluated in the clean leave-dataset-out CV above.")
print("-" * 60)

NOISE_PATH = "/Users/yash/Downloads/PRI2M/Dataset 2/features_noisy_datasets.csv"

try:
    df_noisy = pd.read_csv(NOISE_PATH)
    print(f"Noisy data shape: {df_noisy.shape}")

    kurt_cols_n  = [c for c in df_noisy.columns if "kurt" in c.lower()]
    df_noisy[kurt_cols_n] = df_noisy[kurt_cols_n].fillna(0.0)

    noisy_cols_raw  = [c for c in df_noisy.columns if c not in EXCLUDE]
    noisy_feat_set  = set(noisy_cols_raw)
    noisy_feature_cols = [f for f in feature_cols if f in noisy_feat_set]

    assert list(noisy_feature_cols) == list(feature_cols), \
        f"Feature mismatch!\nClean only: {set(feature_cols)-set(noisy_feature_cols)}\nNoisy only: {set(noisy_feature_cols)-set(feature_cols)}"

    X_noisy      = df_noisy[noisy_feature_cols].values.astype(np.float64)
    y_noisy      = df_noisy["label"].values
    groups_noisy = df_noisy["dataset"].values

    noise_model = xgb.XGBClassifier()
    noise_model.load_model(f"{OUT}/xgb_final_v3.json")
    print(f"Loaded model from {OUT}/xgb_final_v3.json")
    
    try:
        np.testing.assert_allclose(
            noise_model.predict_proba(X[:5]), 
            final_model.predict_proba(X[:5]), 
            atol=1e-5
        )
    except AssertionError as e:
        print(f"[WARN] Loaded model predictions differ from final_model: {e}")
        noise_model = final_model

    noise_rows = []
    noisy_dataset_ids = sorted(df_noisy["dataset"].unique())

    for ds_id in noisy_dataset_ids:
        mask  = groups_noisy == ds_id
        X_ds  = X_noisy[mask]
        y_ds  = y_noisy[mask]
        if mask.sum() == 0:
            continue

        proba_n = noise_model.predict_proba(X_ds)[:, 1]
        preds_n = (proba_n >= 0.5).astype(int)

        if len(np.unique(y_ds)) < 2:
            auc_n = np.nan
        else:
            auc_n = roc_auc_score(y_ds, proba_n)

        f1_n  = f1_score(y_ds, preds_n, zero_division=0)
        f2_n  = fbeta_score(y_ds, preds_n, beta=2, zero_division=0)
        rec_n = recall_score(y_ds, preds_n, zero_division=0)

        tier = "1pct" if 61 <= int(ds_id) <= 90 else "5pct"
        noise_rows.append({
            "dataset":    int(ds_id),
            "noise_tier": tier,
            "n":          int(mask.sum()),
            "auc":        round(float(auc_n), 4) if not np.isnan(auc_n) else None,
            "f1":         round(float(f1_n), 4),
            "f2":         round(float(f2_n), 4),
            "recall":     round(float(rec_n), 4),
        })
        print(f"  ds={int(ds_id):3d} | tier={tier} | "
              f"AUC={auc_n if not np.isnan(auc_n) else 0:.3f} | F2={f2_n:.3f} | Recall={rec_n:.3f}")

    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(f"{OUT}/noise_robustness_results.csv", index=False)
    print(f"\nSaved: {OUT}/noise_robustness_results.csv")

    clean_mean_auc    = cv_df["auc"].mean()
    clean_mean_f2     = cv_df["f2"].mean()
    clean_mean_recall = cv_df["recall"].mean()

    # Noise robustness summary plot
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    metrics_cfg = [
        ("auc",    "AUC per dataset",  clean_mean_auc),
        ("recall", "Recall per dataset", clean_mean_recall),
        ("f2",     "F2 per dataset", clean_mean_f2),
    ]

    df_1pct = noise_df[noise_df["noise_tier"] == "1pct"].sort_values("dataset")
    df_5pct = noise_df[noise_df["noise_tier"] == "5pct"].sort_values("dataset")
    clean_sorted = cv_df.sort_values("dataset")

    for ax, (metric, ylabel, cmean) in zip(axes, metrics_cfg):
        ax.bar(clean_sorted["dataset"].values, clean_sorted[metric].fillna(0).values,
               color="#0D9488", width=0.7, label="Clean (1–60)")
        if not df_1pct.empty:
            ax.bar(df_1pct["dataset"].values, df_1pct[metric].fillna(0).values,
                   color="#F59E0B", width=0.7, label="1% Noise (61–90)")
        if not df_5pct.empty:
            ax.bar(df_5pct["dataset"].values, df_5pct[metric].fillna(0).values,
                   color="#EF4444", width=0.7, label="5% Noise (91–120)")
        ax.axhline(cmean, color="#334155", lw=1.2, ls="--",
                   label=f"Clean mean = {cmean:.3f}")
        ax.axvline(60.5, color="#64748B", lw=1.2, ls="--")
        ax.set(ylabel=ylabel, ylim=(0, 1.08))
        ax.legend(fontsize=8, loc="lower right")

    axes[-1].set_xlabel("Dataset ID")
    plt.tight_layout()
    fig.savefig(f"{OUT}/noise_robustness_plot.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT}/noise_robustness_plot.png")

    # Per-dataset performance curve plots
    noise_results_csv = pd.read_csv(f"{OUT}/noise_robustness_results.csv")

    group_configs = [
        {
            "df":    cv_df.copy(),
            "title": "Clean Datasets (1–60) — Per-dataset Performance",
            "fname": f"{OUT}/perf_curve_clean.png",
        },
        {
            "df":    noise_results_csv[noise_results_csv["noise_tier"] == "1pct"].copy(),
            "title": "1% Noise Datasets (61–90) — Per-dataset Performance",
            "fname": f"{OUT}/perf_curve_1pct.png",
        },
        {
            "df":    noise_results_csv[noise_results_csv["noise_tier"] == "5pct"].copy(),
            "title": "5% Noise Datasets (91–120) — Per-dataset Performance",
            "fname": f"{OUT}/perf_curve_5pct.png",
        },
    ]
    required_cols = {"dataset", "auc", "f2", "recall"}

    for g in group_configs:
        try:
            gdf = g["df"]
            if gdf.empty:
                print(f"[SKIP] Empty dataframe for group: {g['title']}")
                continue
            missing_c = required_cols - set(gdf.columns)
            if missing_c:
                print(f"[SKIP] Missing columns {missing_c} for group: {g['title']}")
                continue
            
            gdf = gdf.sort_values("dataset")

            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            curve_metrics = [
                ("auc",    "AUC-ROC",  clean_mean_auc),
                ("f2",     "F2 Score", clean_mean_f2),
                ("recall", "Recall",   clean_mean_recall),
            ]
            for ax, (metric, ylabel, cmean) in zip(axes, curve_metrics):
                ax.plot(
                    gdf["dataset"].values, gdf[metric].fillna(0).values,
                    color="#000000", lw=1.5, marker="o", markersize=4
                )
                ax.axhline(cmean, color="#334155", lw=1.2, ls="--",
                           label="Clean mean")
                ax.set_xticks(gdf["dataset"].values)
                ax.tick_params(axis="x", rotation=90)
                ax.set(ylabel=ylabel, ylim=(0, 1.08))
                ax.legend(fontsize=8)

            axes[-1].set_xlabel("Dataset ID")
            fig.suptitle(g["title"], fontsize=13)
            plt.tight_layout()
            fig.savefig(g["fname"], dpi=150)
            plt.close()
            print(f"Saved: {g['fname']}")

        except Exception as e:
            print(f"[ERROR] Performance curve plot failed for '{g['title']}': {e}")

except FileNotFoundError:
    print(f"  [SKIP] {NOISE_PATH} not found — skipping Phase 3")
except Exception as e:
    print(f"  [ERROR] Phase 3 noise robustness failed: {e}")
    import traceback; traceback.print_exc()

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — v3")
print("=" * 60)
print("NOTE: Noisy datasets 61-90 replicate physics of clean datasets 1-30 only.")
print("Datasets 31-60 have no noisy counterparts.")
print("-" * 60)
print(f"Samples              : {len(df):,}")
print(f"Features             : {len(feature_cols)}")
print(f"OOF AUC-ROC          : {overall_auc:.4f}")
print(f"OOF F1               : {overall_f1:.4f}")
print(f"OOF F2 (primary)     : {overall_f2:.4f}")
print(f"OOF Recall           : {overall_recall:.4f}")
print(f"Mean per-fold AUC    : {cv_df['auc'].mean():.4f} +/- {cv_df['auc'].std():.4f}")
print(f"Mean per-fold F2     : {cv_df['f2'].mean():.4f} +/- {cv_df['f2'].std():.4f}")
print(f"\nTop 10 SHAP features:")
for feat, val in mean_shap.head(10).items():
    print(f"  {feat:<45s} {val:.5f}")
print(f"\nAll outputs → ./{OUT}/")
print("=" * 60)
