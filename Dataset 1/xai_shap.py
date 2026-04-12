import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# --------------------
# CONFIG
# --------------------
CSV_PATH = "data/dataset1.csv"
MODELS_DIR = "model/trained_models"
OUT_DIR = "outputs/shap"

BACKGROUND_SIZE = 50
EXPLAIN_SIZE = 100

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------
# LOAD DATA
# --------------------
df = pd.read_csv(CSV_PATH)

y = (df["diagnosis"] == "M").astype(int).to_numpy()
X_df = df.drop(columns=["diagnosis", "id"], errors="ignore")
X = X_df.to_numpy()
feature_names = X_df.columns.tolist()

# Fixed background + explanation set (same for all models)
rng = np.random.default_rng(42)
bg_idx = rng.choice(len(X), size=BACKGROUND_SIZE, replace=False)
ex_idx = rng.choice(len(X), size=EXPLAIN_SIZE, replace=False)

X_bg = X[bg_idx]
X_explain = X[ex_idx]

# --------------------
# RUN SHAP FOR ALL MODELS
# --------------------
for fname in os.listdir(MODELS_DIR):
    if not fname.endswith(".joblib"):
        continue

    name = fname.replace(".joblib", "")
    model = joblib.load(os.path.join(MODELS_DIR, fname))

    def predict_fn(x):
        return model.predict_proba(x)[:, 1]  # malignant prob

    explainer = shap.KernelExplainer(
        predict_fn,
        X_bg
    )

    shap_values = explainer.shap_values(
        X_explain,
        nsamples=200
    )

    model_dir = os.path.join(OUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    # Save raw SHAP values
    np.save(os.path.join(model_dir, "shap_values.npy"), shap_values)

    # Global summary plot
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False
    )
    plt.savefig(os.path.join(model_dir, "summary.png"), dpi=200)
    plt.close()

    # Mean absolute SHAP (global importance)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    df_imp.to_csv(os.path.join(model_dir, "global_importance.csv"), index=False)

    print(f"SHAP done: {name}")
