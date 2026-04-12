import os
import joblib
import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer

# --------------------
# CONFIG
# --------------------
CSV_PATH = "data/dataset1.csv"
MODELS_DIR = "model/trained_models"
OUT_DIR = "outputs/lime"
NUM_SAMPLES = 5000
INSTANCES_TO_EXPLAIN = [0, 5, 10]   # indices in dataset

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------
# LOAD DATA
# --------------------
df = pd.read_csv(CSV_PATH)

y = (df["diagnosis"] == "M").astype(int).to_numpy()
X_df = df.drop(columns=["diagnosis", "id"], errors="ignore")
X = X_df.to_numpy()
feature_names = X_df.columns.tolist()

# --------------------
# LIME EXPLAINER
# --------------------
explainer = LimeTabularExplainer(
    training_data=X,
    feature_names=feature_names,
    class_names=["benign", "malignant"],
    discretize_continuous=True,
    mode="classification"
)

# --------------------
# RUN LIME FOR ALL MODELS
# --------------------
for fname in os.listdir(MODELS_DIR):
    if not fname.endswith(".joblib"):
        continue

    name = fname.replace(".joblib", "")
    model = joblib.load(os.path.join(MODELS_DIR, fname))

    def predict_fn(x):
        return model.predict_proba(x)

    model_dir = os.path.join(OUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    for idx in INSTANCES_TO_EXPLAIN:
        exp = explainer.explain_instance(
            X[idx],
            predict_fn,
            num_features=10,
            num_samples=NUM_SAMPLES
        )

        # Save explanation as text
        with open(os.path.join(model_dir, f"instance_{idx}.txt"), "w") as f:
            for feat, weight in exp.as_list(label=1):
                f.write(f"{feat}: {weight:.6f}\n")

        # Optional: save figure
        fig = exp.as_pyplot_figure(label=1)
        fig.set_size_inches(10, 6)  # width, height in inches

        fig.tight_layout()
        fig.savefig(
            os.path.join(model_dir, f"instance_{idx}.png"),
            dpi=200,
            bbox_inches="tight"
        )
        fig.clf()

    print(f"LIME done: {name}")
