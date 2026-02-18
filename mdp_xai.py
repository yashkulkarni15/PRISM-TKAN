"""
mdp_xai.py — Random Forest + 6 XAI visualizations
Methods: Permutation Importance, PDP, ICE, ALE, LIME, SHAP
Outputs: saves figures to ./xai_outputs/

Works best for binary classification (e.g., diagnosis M/B).
If your target is regression, it also supports it via TASK="regression".

Run:
  source .venv/bin/activate
  python mdp_xai.py
"""

from __future__ import annotations

# ---- Force headless plotting (WSL safe)
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


# =========================
# Config (EDIT HERE)
# =========================
@dataclass
class Config:
    DATA_PATH: str = "dataset1.csv"   # <-- your csv filename/path
    TARGET_COL: str = "diagnosis"     # <-- label column (e.g., M/B)
    TASK: str = "auto"                # "auto" | "classification" | "regression"

    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # RF params
    N_ESTIMATORS: int = 400
    MAX_DEPTH: Optional[int] = None
    MIN_SAMPLES_LEAF: int = 1

    # XAI params
    TOP_K_FEATURES: int = 6
    ICE_SUBSAMPLE: int = 200
    SHAP_BACKGROUND_SIZE: int = 200
    LIME_NUM_FEATURES: int = 10

    # PDP/ICE settings
    PDP_GRID_RESOLUTION: int = 25     # lower = faster

    OUTPUT_DIR: str = "xai_outputs"


CFG = Config()


# =========================
# Utilities
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(s))


def detect_task(y: pd.Series, task_setting: str) -> str:
    if task_setting in {"classification", "regression"}:
        return task_setting
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        return "classification" if nunique <= 20 else "regression"
    return "classification"


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = get_feature_types(X)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, task: str) -> dict:
    if task == "classification":
        y_pred = model.predict(X_test)
        out = {"accuracy": float(accuracy_score(y_test, y_pred))}
        if hasattr(model, "predict_proba") and y_test.nunique() == 2:
            proba = model.predict_proba(X_test)[:, 1]
            out["roc_auc"] = float(roc_auc_score(y_test, proba))
        return out

    y_pred = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }


def to_dense(X_any) -> np.ndarray:
    if hasattr(X_any, "toarray"):
        return X_any.toarray()
    return np.asarray(X_any)


# =========================
# Main
# =========================
def main():
    warnings.filterwarnings("ignore")
    ensure_dir(CFG.OUTPUT_DIR)
    plt.ioff()  # no interactive

    # ---- Load
    df = pd.read_csv(CFG.DATA_PATH)
    if CFG.TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL='{CFG.TARGET_COL}' not found. Columns={list(df.columns)}")

    y = df[CFG.TARGET_COL]
    X = df.drop(columns=[CFG.TARGET_COL])

    # ---- Task
    task = detect_task(y, CFG.TASK)
    print(f"[Info] Task detected: {task}")

    # ---- Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CFG.TEST_SIZE,
        random_state=CFG.RANDOM_STATE,
        stratify=y if task == "classification" and y.nunique() > 1 else None,
    )

    # ---- Pipeline
    preprocessor = build_preprocess(X_train)

    if task == "classification":
        rf = RandomForestClassifier(
            n_estimators=CFG.N_ESTIMATORS,
            random_state=CFG.RANDOM_STATE,
            n_jobs=-1,
            max_depth=CFG.MAX_DEPTH,
            min_samples_leaf=CFG.MIN_SAMPLES_LEAF,
        )
    else:
        rf = RandomForestRegressor(
            n_estimators=CFG.N_ESTIMATORS,
            random_state=CFG.RANDOM_STATE,
            n_jobs=-1,
            max_depth=CFG.MAX_DEPTH,
            min_samples_leaf=CFG.MIN_SAMPLES_LEAF,
        )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("rf", rf),
    ])

    # ---- Train
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, task)
    print("[Info] Test metrics:", metrics)

    feature_names = get_feature_names(model.named_steps["preprocess"])
    if not feature_names:
        # fallback (won't match one-hot perfectly)
        feature_names = list(X_train.columns)

    # =========================================================
    # 1) Permutation Importance (Global) + visualization
    # =========================================================
    print("[XAI] Permutation Importance...")
    scoring = "roc_auc" if task == "classification" and y_test.nunique() == 2 else None
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=CFG.RANDOM_STATE,
        n_jobs=-1,
        scoring=scoring,
    )

    importances = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
    importances.to_csv(os.path.join(CFG.OUTPUT_DIR, "permutation_importance.csv"))

    plt.figure(figsize=(8, 6))
    top_imp = importances.head(15)
    plt.barh(top_imp.index[::-1], top_imp.values[::-1])
    plt.xlabel("Decrease in model performance")
    plt.title("Permutation Feature Importance (Global)")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUTPUT_DIR, "perm_importance.png"), dpi=300)
    plt.close()

    top_features = list(importances.head(CFG.TOP_K_FEATURES).index)
    print(f"[Info] Top features (for plots): {top_features}")

    # Prepare processed matrix + rf estimator once (fast + stable)
    X_train_proc = to_dense(model.named_steps["preprocess"].transform(X_train))
    X_test_proc = to_dense(model.named_steps["preprocess"].transform(X_test))
    rf_est = model.named_steps["rf"]

    # Indices for top features in processed space
    top_idx = [feature_names.index(f) for f in top_features]

    # =========================================================
    # 2) PDP (Global effect) — one plot per feature (stable)
    # =========================================================
    print("[XAI] PDP...")
    for f, j in zip(top_features, top_idx):
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(
            rf_est, X_train_proc,
            features=[j],
            kind="average",
            grid_resolution=CFG.PDP_GRID_RESOLUTION,
            ax=ax
        )
        ax.set_title(f"PDP — Effect of {f}")
        plt.tight_layout()
        plt.savefig(os.path.join(CFG.OUTPUT_DIR, f"pdp_{safe_name(f)}.png"), dpi=300)
        plt.close()

    # =========================================================
    # 3) ICE (Individual effect) — one plot per feature (stable)
    # =========================================================
    print("[XAI] ICE...")
    n_ice = min(CFG.ICE_SUBSAMPLE, X_train_proc.shape[0])
    rng = np.random.default_rng(CFG.RANDOM_STATE)
    rows = rng.choice(X_train_proc.shape[0], size=n_ice, replace=False)
    X_ice_proc = X_train_proc[rows]

    for f, j in zip(top_features, top_idx):
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(
            rf_est, X_ice_proc,
            features=[j],
            kind="individual",
            grid_resolution=CFG.PDP_GRID_RESOLUTION,
            ax=ax
        )
        ax.set_title(f"ICE — Individual effect of {f}")
        plt.tight_layout()
        plt.savefig(os.path.join(CFG.OUTPUT_DIR, f"ice_{safe_name(f)}.png"), dpi=300)
        plt.close()

    # =========================================================
    # 4) ALE (Robust effect under correlation) — using alibi
    # =========================================================
    print("[XAI] ALE (alibi)...")
    try:
        from alibi.explainers import ALE

        if task == "classification":
            predictor_proc = lambda Xp: rf_est.predict_proba(Xp)[:, 1]
        else:
            predictor_proc = lambda Xp: rf_est.predict(Xp)

        ale = ALE(predictor=predictor_proc, feature_names=feature_names, target_names=["prediction"])
        exp = ale.explain(X_train_proc, features=top_idx)

        for feat, idx in zip(top_features, top_idx):
            exp.plot(features=[idx])
            plt.title(f"ALE — Accumulated Local Effect of {feat}")
            plt.tight_layout()
            plt.savefig(os.path.join(CFG.OUTPUT_DIR, f"ale_{safe_name(feat)}.png"), dpi=300)
            plt.close()

    except ImportError:
        print("  [Skip] alibi not installed. Install: pip install alibi")
    except Exception as e:
        print(f"  [Skip] ALE failed: {e}")

    # =========================================================
    # 5) LIME (Local surrogate) — HTML + barplot PNG
    # =========================================================
    print("[XAI] LIME...")
    try:
        from lime.lime_tabular import LimeTabularExplainer

        # LIME explains in raw feature space (more interpretable for you)
        X_train_np = X_train.to_numpy()
        X_test_np = X_test.to_numpy()
        feature_names_raw = list(X_train.columns)

        if task == "classification":
            class_names = [str(c) for c in sorted(pd.unique(y_train))]
            lime_predict = model.predict_proba
            mode = "classification"
        else:
            class_names = None
            lime_predict = model.predict
            mode = "regression"

        explainer = LimeTabularExplainer(
            training_data=X_train_np,
            feature_names=feature_names_raw,
            class_names=class_names,
            mode=mode,
            discretize_continuous=True,
            random_state=CFG.RANDOM_STATE
        )

        idx0 = 0
        x0 = X_test_np[idx0]

        if task == "classification":
            exp = explainer.explain_instance(
                x0, lime_predict, num_features=CFG.LIME_NUM_FEATURES, top_labels=1
            )
        else:
            exp = explainer.explain_instance(
                x0, lime_predict, num_features=CFG.LIME_NUM_FEATURES
            )

        lime_html = os.path.join(CFG.OUTPUT_DIR, "lime_explanation_sample0.html")
        exp.save_to_file(lime_html)

        fig = exp.as_pyplot_figure()
        plt.title("LIME Local Explanation (sample 0)")
        plt.tight_layout()
        plt.savefig(os.path.join(CFG.OUTPUT_DIR, "lime_barplot_sample0.png"), dpi=300)
        plt.close()

    except ImportError:
        print("  [Skip] lime not installed. Install: pip install lime")
    except Exception as e:
        print(f"  [Skip] LIME failed: {e}")

    # =========================================================
    # 6) SHAP (Global + Local) — beeswarm + bar + dependence + waterfall
    # =========================================================
    print("[XAI] SHAP...")
    try:
        import shap

        bg = shap.sample(
            X_train_proc,
            min(CFG.SHAP_BACKGROUND_SIZE, X_train_proc.shape[0]),
            random_state=CFG.RANDOM_STATE
        )

        explainer = shap.TreeExplainer(rf_est, data=bg, feature_names=feature_names)
        shap_values = explainer.shap_values(X_test_proc)

        # Pick positive class for binary classification
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        # (a) beeswarm
        plt.figure()
        shap.summary_plot(sv, X_test_proc, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(CFG.OUTPUT_DIR, "shap_beeswarm.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # (b) global bar
        plt.figure()
        shap.summary_plot(sv, X_test_proc, plot_type="bar", feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(CFG.OUTPUT_DIR, "shap_bar.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # (c) dependence plots for top 3
        for f in top_features[:3]:
            plt.figure()
            shap.dependence_plot(
                f, sv, X_test_proc,
                feature_names=feature_names,
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(CFG.OUTPUT_DIR, f"shap_dependence_{safe_name(f)}.png"), dpi=300)
            plt.close()

        # (d) waterfall sample 0 (if supported by your shap version)
        try:
            explanation = explainer(X_test_proc[:1])
            shap.plots.waterfall(explanation[0], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(CFG.OUTPUT_DIR, "shap_waterfall_sample0.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()
        except Exception:
            print("  [Note] Waterfall not available in your shap version; beeswarm/bar saved.")

    except ImportError:
        print("  [Skip] shap not installed. Install: pip install shap")
    except Exception as e:
        print(f"  [Skip] SHAP failed: {e}")

    print("\n✅ Done. All visuals saved in:", CFG.OUTPUT_DIR)
    print("Look for:")
    print("  perm_importance.png + permutation_importance.csv")
    print("  pdp_*.png, ice_*.png, ale_*.png")
    print("  lime_barplot_sample0.png + lime_explanation_sample0.html")
    print("  shap_beeswarm.png, shap_bar.png, shap_dependence_*.png, shap_waterfall_sample0.png (if supported)")


if __name__ == "__main__":
    main()
