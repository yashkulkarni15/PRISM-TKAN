"""
PyCaret Automation Script with GUI and Research-Grade XAI Suite
===============================================================
- Forces Random Forest classifier
- Trains dual RF models (original features + PCA features) for fair comparison
- LIME with stability quantification (N=10 runs, error bars)
- DiCE diverse counterfactual explanations
- Research-grade sample sizes throughout
- SHAP, LIME, PDP, ALE, ICE, Permutation Importance, SHAP Dependence, PCA

Install requirements:
    pip install pycaret gradio shap lime PyALE scikit-learn matplotlib pillow dice-ml
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '2'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gradio as gr

from pycaret.classification import ClassificationExperiment

import shap
shap.initjs()
import lime
import lime.lime_tabular
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import dice_ml

import io
from PIL import Image
import gc
import traceback as tb


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def fig_to_pil(fig=None, dpi=100):
    """Convert matplotlib figure to PIL Image and close it."""
    if fig is None:
        fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    result = img.copy()
    img.close()
    buf.close()
    plt.close(fig)
    return result


def error_image(msg):
    """Placeholder image with error/info message."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=9, wrap=True,
            transform=ax.transAxes)
    ax.axis('off')
    return fig_to_pil(fig)


def get_actual_model(model):
    """Extract the final estimator from a sklearn Pipeline if needed."""
    if hasattr(model, 'named_steps'):
        return model.named_steps.get('actual_estimator', model.steps[-1][1])
    return model


def _get_numeric_features(X):
    """Return only numeric columns from a DataFrame."""
    return X.select_dtypes(include=[np.number])


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLEANING
# ──────────────────────────────────────────────────────────────────────────────
def clean_data(df):
    """Basic data cleaning."""
    original_rows = len(df)
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    cleaned_rows = len(df)
    return df, f"Cleaned: {original_rows - cleaned_rows} rows removed, {cleaned_rows} rows remaining"


# ──────────────────────────────────────────────────────────────────────────────
# DUAL-MODEL TRAINING HELPER
# ──────────────────────────────────────────────────────────────────────────────
def train_pca_rf(X_train, y_train, X_test, n_components=10):
    """Train a separate RF from scratch on PCA-transformed features.

    Returns:
        rf_pca: trained RandomForestClassifier
        X_train_pca_df: PCA-transformed train data (DataFrame with PC names)
        X_test_pca_df: PCA-transformed test data
        pca: fitted PCA object
        scaler: fitted StandardScaler
    """
    X_train_num = _get_numeric_features(X_train)
    X_test_num = _get_numeric_features(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    n_comp = min(n_components, X_train_num.shape[1])
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    pc_names = [f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)"
                for i in range(n_comp)]
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=pc_names,
                                   index=X_train.index)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=pc_names,
                                  index=X_test.index)

    # Encode y if string
    y_enc = y_train.copy()
    le = None
    if y_enc.dtype == object or y_enc.dtype.name == 'category':
        le = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(y_enc), index=y_train.index)

    rf_pca = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    )
    rf_pca.fit(X_train_pca_df, y_enc)

    return rf_pca, X_train_pca_df, X_test_pca_df, pca, scaler, le


# ──────────────────────────────────────────────────────────────────────────────
# XAI GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

# 1. SHAP Summary Plot  ───────────────────────────────────────────────────────
def generate_shap_plot(model, X_train, X_test, title_suffix="Original Features"):
    """SHAP Summary Plot – global feature importance via Shapley values."""
    try:
        plt.close('all')
        actual_model = get_actual_model(model)

        n_test = min(200, len(X_test))
        n_bg = min(200, len(X_train))
        X_test_sample = X_test.iloc[:n_test].copy()
        X_bg = X_train.iloc[:n_bg].copy()

        # TreeExplainer for RF
        try:
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer.shap_values(X_test_sample)
        except Exception:
            explainer = shap.Explainer(actual_model.predict, X_bg)
            shap_values = explainer(X_test_sample).values

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_sample, show=False, max_display=15)
        plt.title(f"SHAP Feature Importance ({title_suffix})", fontsize=12)
        plt.tight_layout()
        return fig_to_pil(plt.gcf())

    except Exception as e:
        print(f"SHAP error ({title_suffix}): {e}")
        traceback_str = tb.format_exc()
        print(traceback_str)
        return error_image(f"SHAP ({title_suffix}) unavailable:\n{str(e)[:120]}")


# 2. LIME with Stability Quantification  ──────────────────────────────────────
def _make_lime_predict_fn(model, col_names):
    """Create a predict_proba wrapper that converts numpy → DataFrame.

    PyCaret pipelines reject raw numpy arrays because they expect
    named columns for internal preprocessing lookups.
    """
    def predict_proba_wrapper(X_np):
        X_df = pd.DataFrame(X_np, columns=col_names)
        return model.predict_proba(X_df)
    return predict_proba_wrapper


def generate_lime_explanation(model, X_train, X_test, n_runs=10,
                               title_suffix="Original Features"):
    """
    LIME explanation with stability quantification.

    Runs LIME n_runs times on the same sample with different seeds.
    Shows mean contributions with std error bars.
    Reports a stability score (mean R² across runs).
    """
    try:
        plt.close('all')

        X_train_np = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
        X_test_np = X_test.values if hasattr(X_test, 'values') else np.array(X_test)
        feature_names = (X_train.columns.tolist() if hasattr(X_train, 'columns')
                         else [f'Feature_{i}' for i in range(X_train_np.shape[1])])

        # CRITICAL: LIME passes numpy to predict_fn.
        # PyCaret needs DataFrames with column names.
        predict_fn = _make_lime_predict_fn(model, feature_names)

        # Pick a sample that gives a non-trivial explanation
        actual_idx = 0
        for idx in range(min(10, len(X_test_np))):
            try:
                test_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train_np, feature_names=feature_names,
                    class_names=['Class 0', 'Class 1'],
                    mode='classification', discretize_continuous=True,
                    random_state=0
                )
                sample = X_test_np[idx]
                test_exp = test_explainer.explain_instance(
                    sample, predict_fn, num_features=10, num_samples=1000
                )
                exp_list = test_exp.as_list()
                if exp_list and max(abs(v) for _, v in exp_list) > 1e-10:
                    actual_idx = idx
                    break
            except Exception:
                continue

        sample = X_test_np[actual_idx]

        # Run LIME n_runs times with different seeds
        all_contributions = {}  # feature_name -> list of values across runs
        for run in range(n_runs):
            try:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train_np, feature_names=feature_names,
                    class_names=['Class 0', 'Class 1'],
                    mode='classification', discretize_continuous=True,
                    random_state=run * 42
                )
                explanation = explainer.explain_instance(
                    sample, predict_fn,
                    num_features=10, num_samples=5000
                )
                for feat_name, value in explanation.as_list():
                    if feat_name not in all_contributions:
                        all_contributions[feat_name] = []
                    all_contributions[feat_name].append(value)
            except Exception as e:
                print(f"LIME run {run} failed: {e}")

        if not all_contributions:
            return error_image("LIME returned empty explanation across all runs")

        # Compute statistics
        features = list(all_contributions.keys())
        means = [np.mean(all_contributions[f]) for f in features]
        stds = [np.std(all_contributions[f]) for f in features]

        # Sort by absolute mean
        sorted_idx = np.argsort([abs(m) for m in means])[::-1]
        features = [features[i] for i in sorted_idx]
        means = [means[i] for i in sorted_idx]
        stds = [stds[i] for i in sorted_idx]

        # Stability score: 1 - mean(CV) where CV = std/|mean|
        cvs = []
        for m, s in zip(means, stds):
            if abs(m) > 1e-10:
                cvs.append(s / abs(m))
        stability = 1 - np.mean(cvs) if cvs else 0.0
        stability = max(0.0, min(1.0, stability))

        colors = ['#2ecc71' if m > 0 else '#e74c3c' for m in means]

        fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.5)))
        y_pos = np.arange(len(features))
        ax.barh(y_pos, means, xerr=stds, color=colors,
                edgecolor='white', linewidth=0.5,
                error_kw={'elinewidth': 1.5, 'capsize': 3, 'capthick': 1})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Feature Contribution (mean ± std)', fontsize=10)
        ax.set_title(f"LIME Explanation ({title_suffix}, Sample #{actual_idx})\n"
                      f"{n_runs} runs · Stability Score: {stability:.2f} "
                      f"(1.0 = perfectly stable)",
                      fontsize=11)
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

        # Stability annotation
        stability_color = '#2ecc71' if stability > 0.7 else '#f39c12' if stability > 0.4 else '#e74c3c'
        ax.text(0.98, 0.98,
                f"Stability: {stability:.2f}",
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                ha='right', va='top', color=stability_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=stability_color, alpha=0.9))

        plt.tight_layout()
        return fig_to_pil(fig)

    except Exception as e:
        print(f"LIME error ({title_suffix}): {e}")
        import traceback; traceback.print_exc()
        return error_image(f"LIME ({title_suffix}) unavailable:\n{str(e)[:100]}")


# 3. PDP – Partial Dependence Plots  ──────────────────────────────────────────
def generate_pdp_plots(model, X_train):
    """PDP – marginal feature effect averaged over the dataset."""
    try:
        plt.close('all')
        actual_model = get_actual_model(model)
        X_sample = X_train.sample(n=min(300, len(X_train)), random_state=42)
        n_features = min(4, len(X_train.columns))

        if hasattr(actual_model, 'feature_importances_'):
            top_idx = np.argsort(actual_model.feature_importances_)[-n_features:][::-1]
            features = [X_train.columns[i] for i in top_idx]
        else:
            features = list(X_train.columns[:n_features])

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for idx, feature in enumerate(features[:4]):
            try:
                PartialDependenceDisplay.from_estimator(
                    model, X_sample, features=[feature], ax=axes[idx],
                    kind='average', n_jobs=1)
                axes[idx].set_title(f"PDP: {feature}", fontsize=10)
            except Exception as e:
                axes[idx].text(0.5, 0.5, f"Error: {str(e)[:40]}",
                               ha='center', va='center', fontsize=8)
                axes[idx].set_title(f"PDP: {feature} (failed)")

        for idx in range(len(features), 4):
            axes[idx].set_visible(False)

        plt.suptitle("Partial Dependence Plots (Top Features)", fontsize=12)
        plt.tight_layout()
        return fig_to_pil(fig)

    except Exception as e:
        print(f"PDP error: {e}")
        return error_image(f"PDP unavailable:\n{str(e)[:100]}")


# 4. ALE – Accumulated Local Effects  ─────────────────────────────────────────
def generate_ale_plots(model, X_train):
    """ALE – unbiased feature-effect plots that handle correlated features."""
    try:
        plt.close('all')

        try:
            from PyALE import ale
        except ImportError:
            return error_image("ALE requires PyALE.\nInstall: pip install PyALE")

        actual_model = get_actual_model(model)
        X_numeric = _get_numeric_features(X_train)
        if X_numeric.shape[1] == 0:
            return error_image("ALE requires numeric features. None found.")

        X_sample = X_numeric.sample(n=min(500, len(X_numeric)), random_state=42).copy()

        # Select top features (numeric only)
        n_features = min(4, X_numeric.shape[1])
        if hasattr(actual_model, 'feature_importances_'):
            all_cols = list(X_train.columns)
            num_cols = list(X_numeric.columns)
            num_imp = []
            for col in num_cols:
                num_imp.append(actual_model.feature_importances_[all_cols.index(col)])
            top_idx = np.argsort(num_imp)[-n_features:][::-1]
            features = [num_cols[i] for i in top_idx]
        else:
            features = list(X_numeric.columns[:n_features])

        # Wrapper that preserves dtypes and returns numeric predictions
        class ALEModelWrapper:
            def __init__(self, model, full_X_template):
                self._model = model
                self._template = full_X_template.iloc[0:1].copy()
            def predict(self, X):
                n = len(X)
                full_df = pd.concat([self._template] * n, ignore_index=True)
                for col in X.columns:
                    if col in full_df.columns:
                        full_df[col] = X[col].values
                preds = self._model.predict_proba(full_df)[:, 1]
                if hasattr(preds, 'dtype') and np.issubdtype(preds.dtype, np.number):
                    return preds.astype(float)
                try:
                    return preds.astype(float)
                except (ValueError, TypeError):
                    le = LabelEncoder()
                    return le.fit_transform(preds).astype(float)

        wrapped = ALEModelWrapper(model, X_train)

        ale_images = []
        for feat in features[:4]:
            try:
                plt.close('all')
                ale_result = ale(
                    X=X_sample, model=wrapped, feature=[feat],
                    grid_size=20, include_CI=True, plot=True
                )
                if isinstance(ale_result, tuple) and len(ale_result) >= 1:
                    ale_fig = ale_result[0]
                    if ale_fig is not None and hasattr(ale_fig, 'savefig'):
                        ale_fig.suptitle(f"ALE: {feat}", fontsize=10)
                        ale_fig.tight_layout()
                        ale_images.append(fig_to_pil(ale_fig))
                    else:
                        f = plt.gcf()
                        f.suptitle(f"ALE: {feat}", fontsize=10)
                        f.tight_layout()
                        ale_images.append(fig_to_pil(f))
                else:
                    f = plt.gcf()
                    f.suptitle(f"ALE: {feat}", fontsize=10)
                    f.tight_layout()
                    ale_images.append(fig_to_pil(f))
            except Exception as e:
                print(f"ALE error for {feat}: {e}")
                ef, ea = plt.subplots(figsize=(5, 4))
                ea.text(0.5, 0.5, f"ALE Error:\n{str(e)[:80]}", ha='center',
                        va='center', fontsize=8, transform=ea.transAxes)
                ea.axis('off')
                ale_images.append(fig_to_pil(ef))

        if ale_images:
            max_w = max(img.width for img in ale_images)
            max_h = max(img.height for img in ale_images)
            composite = Image.new('RGB', (max_w * 2, max_h * 2), 'white')
            for i, img in enumerate(ale_images):
                r, c = divmod(i, 2)
                composite.paste(img, (c * max_w, r * max_h))
            return composite
        else:
            return error_image("No ALE plots generated")

    except Exception as e:
        print(f"ALE error: {e}")
        return error_image(f"ALE unavailable:\n{str(e)[:120]}")


# 5. ICE – Individual Conditional Expectation  ────────────────────────────────
def generate_ice_plots(model, X_train):
    """ICE – individual-level PDP showing prediction heterogeneity."""
    try:
        plt.close('all')
        actual_model = get_actual_model(model)
        X_sample = X_train.sample(n=min(300, len(X_train)), random_state=42)
        n_features = min(4, len(X_train.columns))

        if hasattr(actual_model, 'feature_importances_'):
            top_idx = np.argsort(actual_model.feature_importances_)[-n_features:][::-1]
            features = [X_train.columns[i] for i in top_idx]
        else:
            features = list(X_train.columns[:n_features])

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for idx, feature in enumerate(features[:4]):
            try:
                PartialDependenceDisplay.from_estimator(
                    model, X_sample, features=[feature], ax=axes[idx],
                    kind='both', n_jobs=1,
                    ice_lines_kw={'color': 'steelblue', 'alpha': 0.15, 'linewidth': 0.5},
                    pd_line_kw={'color': 'red', 'linewidth': 2}
                )
                axes[idx].set_title(f"ICE + PDP: {feature}", fontsize=10)
            except Exception as e:
                axes[idx].text(0.5, 0.5, f"Error: {str(e)[:40]}",
                               ha='center', va='center', fontsize=8)

        for idx in range(len(features), 4):
            axes[idx].set_visible(False)

        plt.suptitle("ICE Plots (individual lines) + PDP (red average)", fontsize=12)
        plt.tight_layout()
        return fig_to_pil(fig)

    except Exception as e:
        print(f"ICE error: {e}")
        return error_image(f"ICE unavailable:\n{str(e)[:100]}")


# 6. Permutation Importance  ──────────────────────────────────────────────────
def generate_permutation_importance(model, X_test, y_test):
    """Permutation Importance – model-agnostic, 30 repeats."""
    try:
        plt.close('all')

        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=30, random_state=42, n_jobs=-1, scoring='accuracy'
        )

        sorted_idx = result.importances_mean.argsort()[-15:]
        feature_names = X_test.columns[sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=feature_names)
        ax.set_xlabel("Decrease in Accuracy")
        ax.set_title("Permutation Importance (Top 15, 30 repeats)\n"
                      "Model-agnostic — RF on original features", fontsize=11)
        plt.tight_layout()
        return fig_to_pil(fig)

    except Exception as e:
        print(f"Permutation Importance error: {e}")
        return error_image(f"Permutation Importance unavailable:\n{str(e)[:100]}")


# 7. SHAP Dependence Plot  ────────────────────────────────────────────────────
def generate_shap_dependence(model, X_train, X_test):
    """SHAP Dependence Plot – colored by strongest interacting feature."""
    try:
        plt.close('all')
        actual_model = get_actual_model(model)

        n_bg = min(200, len(X_train))
        n_test = min(200, len(X_test))
        X_bg = X_train.iloc[:n_bg].copy()
        X_eval = X_test.iloc[:n_test].copy()

        try:
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer.shap_values(X_eval)
        except Exception:
            explainer = shap.Explainer(actual_model.predict, X_bg)
            shap_values = explainer(X_eval).values

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        mean_abs = np.abs(shap_values).mean(axis=0)
        top2_idx = np.argsort(mean_abs)[-2:][::-1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for i, feat_idx in enumerate(top2_idx):
            shap.dependence_plot(
                feat_idx, shap_values, X_eval, ax=axes[i], show=False
            )
            axes[i].set_title(f"SHAP Dependence: {X_eval.columns[feat_idx]}", fontsize=10)

        plt.suptitle("SHAP Dependence (color = strongest interaction)", fontsize=12)
        plt.tight_layout()
        return fig_to_pil(fig)

    except Exception as e:
        print(f"SHAP Dependence error: {e}")
        return error_image(f"SHAP Dependence unavailable:\n{str(e)[:100]}")


# 8. DiCE Counterfactual Explanations  ────────────────────────────────────────
def generate_dice_counterfactual(model, X_train, X_test, y_train):
    """DiCE – generates 5 diverse counterfactual explanations."""
    try:
        plt.close('all')

        target_name = y_train.name if y_train.name else 'target'

        # CRITICAL: convert categorical/object to plain python str
        # PyCaret often returns categorical dtype which DiCE can't handle
        y_plain = y_train.astype(str)

        train_df = X_train.copy()
        train_df[target_name] = y_plain.values

        continuous_features = _get_numeric_features(X_train).columns.tolist()

        d = dice_ml.Data(
            dataframe=train_df,
            continuous_features=continuous_features,
            outcome_name=target_name
        )

        # DiCE model wrapper — use 'sklearn' backend
        m = dice_ml.Model(model=model, backend='sklearn', model_type='classifier')

        exp = dice_ml.Dice(d, m, method='random')

        query = X_test.iloc[0:1].copy()

        # Use 'opposite' so DiCE auto-detects the other class
        cf_explanation = exp.generate_counterfactuals(
            query, total_CFs=5, desired_class='opposite',
            features_to_vary='all'
        )

        # Extract results
        cf_df = cf_explanation.cf_examples_list[0].final_cfs_df

        if cf_df is None or len(cf_df) == 0:
            return error_image("DiCE could not generate counterfactuals")

        # Visualise: highlight changed features
        original_row = query.iloc[0]
        # Get original prediction
        if hasattr(model, 'predict_proba'):
            orig_proba = model.predict_proba(query)[0]
            pred_int = int(np.argmax(orig_proba))
        else:
            pred_int = int(model.predict(query)[0])
        desired_class = 1 - pred_int
        n_cfs = len(cf_df)

        # Find features that changed
        changed_features = set()
        for _, cf_row in cf_df.iterrows():
            for col in X_train.columns:
                if col in cf_row.index:
                    try:
                        if abs(float(cf_row[col]) - float(original_row[col])) > 1e-6:
                            changed_features.add(col)
                    except (ValueError, TypeError):
                        if str(cf_row[col]) != str(original_row[col]):
                            changed_features.add(col)

        if not changed_features:
            return error_image("DiCE found no feature changes needed")

        # Plot changed features only
        changed_list = sorted(list(changed_features))[:12]

        fig, ax = plt.subplots(figsize=(12, max(5, len(changed_list) * 0.6)))
        y_pos = np.arange(len(changed_list))
        bar_h = 0.8 / (n_cfs + 1)

        # Original values
        orig_vals = [float(original_row[f]) if f in original_row.index else 0
                     for f in changed_list]
        ax.barh(y_pos - 0.4 + bar_h/2, orig_vals, bar_h,
                label=f'Original (Class {pred_int})',
                color='#e74c3c', alpha=0.8)

        # Counterfactual values
        colors_cf = plt.cm.Greens(np.linspace(0.4, 0.9, n_cfs))
        for cf_i in range(n_cfs):
            cf_row = cf_df.iloc[cf_i]
            cf_vals = []
            for f in changed_list:
                try:
                    cf_vals.append(float(cf_row[f]))
                except (ValueError, TypeError, KeyError):
                    cf_vals.append(orig_vals[changed_list.index(f)])
            ax.barh(y_pos - 0.4 + (cf_i + 1.5) * bar_h, cf_vals, bar_h,
                    label=f'CF #{cf_i+1}', color=colors_cf[cf_i], alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(changed_list, fontsize=9)
        ax.set_xlabel('Feature Value')
        ax.set_title(f"DiCE: {n_cfs} Diverse Counterfactuals\n"
                      f"Flip Class {pred_int} → {desired_class} "
                      f"(showing {len(changed_list)} changed features)", fontsize=11)
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        return fig_to_pil(fig)

    except Exception as e:
        print(f"DiCE error: {e}")
        import traceback; traceback.print_exc()
        return error_image(f"DiCE unavailable:\n{str(e)[:120]}")


# 9. PCA Visualization  ──────────────────────────────────────────────────────
def generate_pca_plot(X_train, y_train):
    """PCA: 2D scatter (colored by class) + explained variance scree plot."""
    try:
        plt.close('all')

        X_num = _get_numeric_features(X_train)
        if X_num.shape[1] < 2:
            return error_image("PCA requires at least 2 numeric features")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)

        n_components = min(X_num.shape[1], 30)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        y_values = y_train.copy()
        if y_values.dtype == object or y_values.dtype.name == 'category':
            le = LabelEncoder()
            y_numeric = le.fit_transform(y_values)
            class_labels = le.classes_
        else:
            y_numeric = y_values.values if hasattr(y_values, 'values') else np.array(y_values)
            class_labels = np.unique(y_numeric)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        unique_classes = np.unique(y_numeric)
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_classes), 2)))
        for i, cls in enumerate(unique_classes):
            mask = y_numeric == cls
            label = str(class_labels[i]) if i < len(class_labels) else str(cls)
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[colors[i]], label=f'Class {label}',
                       alpha=0.6, s=30, edgecolors='white', linewidth=0.3)

        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax1.set_title('PCA – 2D Projection', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
        components = range(1, n_components + 1)
        ax2.bar(components, pca.explained_variance_ratio_ * 100,
                color='steelblue', alpha=0.7, label='Individual')
        ax2.plot(components, cum_var, 'ro-', markersize=4, linewidth=1.5,
                 label='Cumulative')
        ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=95, color='green', linestyle='--', alpha=0.7)
        ax2.text(n_components, 80, ' 80%', va='center', fontsize=8, color='orange')
        ax2.text(n_components, 95, ' 95%', va='center', fontsize=8, color='green')

        n_95 = np.argmax(cum_var >= 95) + 1 if np.any(cum_var >= 95) else n_components
        ax2.axvline(x=n_95, color='green', linestyle=':', alpha=0.5)
        ax2.text(n_95, 5, f'  {n_95} PCs\n  for 95%', fontsize=8, color='green')

        ax2.set_xlabel('Principal Component', fontsize=10)
        ax2.set_ylabel('Explained Variance (%)', fontsize=10)
        ax2.set_title('Explained Variance Ratio', fontsize=11)
        ax2.legend(fontsize=9, loc='center right')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"PCA Analysis ({n_components} components, "
                     f"Top 2 = {cum_var[1]:.1f}%)", fontsize=12)
        plt.tight_layout()
        return fig_to_pil(fig)

    except Exception as e:
        print(f"PCA error: {e}")
        return error_image(f"PCA unavailable:\n{str(e)[:120]}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_analysis(file, target_column):
    """Main analysis: PyCaret comparison (for table) → RF → XAI suite."""

    N_OUTPUTS = 13  # result, table, 11 images

    if file is None:
        return ("Please upload a CSV file.",) + (None,) * (N_OUTPUTS - 1)
    if not target_column:
        return ("Please specify a target column.",) + (None,) * (N_OUTPUTS - 1)

    try:
        df = pd.read_csv(file.name)

        if target_column not in df.columns:
            return (f"Target column '{target_column}' not found. "
                    f"Available: {list(df.columns)}",) + (None,) * (N_OUTPUTS - 1)

        df, clean_msg = clean_data(df)

        if df.columns[0].lower() in ['id', 'index', 'unnamed: 0']:
            df = df.drop(columns=[df.columns[0]])
        if 'diagnosis_numeric' in df.columns and 'diagnosis' in df.columns:
            df = df.drop(columns=['diagnosis_numeric'])

        if len(df) > 500:
            df = df.sample(n=500, random_state=42)
            clean_msg += " (Sampled to 500 rows for faster processing)"

        # ── PyCaret setup & model comparison (for the table) ──
        exp = ClassificationExperiment()
        exp.setup(
            data=df, target=target_column, session_id=42,
            verbose=False, html=False, log_experiment=False,
            system_log=False, n_jobs=1, train_size=0.8, fold=5
        )

        models_to_compare = [
            'lr', 'knn', 'nb', 'dt', 'rf',
            'ada', 'gbc', 'lightgbm', 'et', 'ridge'
        ]
        _ = exp.compare_models(include=models_to_compare, n_select=1)
        comparison_df = exp.pull()

        # ── Force Random Forest ──
        print("Training Random Forest (forced)...")
        rf_model = exp.create_model('rf')
        final_model = exp.finalize_model(rf_model)

        X_train = exp.get_config('X_train')
        X_test = exp.get_config('X_test')
        y_train = exp.get_config('y_train')
        y_test = exp.get_config('y_test')

        gc.collect()

        # ── Train PCA RF model from scratch ──
        print("Training RF on PCA features...")
        rf_pca, X_train_pca, X_test_pca, pca_obj, pca_scaler, pca_le = \
            train_pca_rf(X_train, y_train, X_test)
        gc.collect()

        # ── Generate all XAI explanations ──
        print("Generating SHAP (original features)...")
        shap_orig_img = generate_shap_plot(final_model, X_train, X_test,
                                            "Original Features – RF")
        gc.collect()

        print("Generating SHAP (PCA features)...")
        shap_pca_img = generate_shap_plot(rf_pca, X_train_pca, X_test_pca,
                                           "PCA Features – RF")
        gc.collect()

        print("Generating LIME (original features, 10 stability runs)...")
        lime_orig_img = generate_lime_explanation(final_model, X_train, X_test,
                                                   n_runs=10,
                                                   title_suffix="Original – RF")
        gc.collect()

        print("Generating LIME (PCA features, 10 stability runs)...")
        lime_pca_img = generate_lime_explanation(rf_pca, X_train_pca, X_test_pca,
                                                  n_runs=10,
                                                  title_suffix="PCA – RF")
        gc.collect()

        print("Generating PDP...")
        pdp_img = generate_pdp_plots(final_model, X_train)
        gc.collect()

        print("Generating ALE...")
        ale_img = generate_ale_plots(final_model, X_train)
        gc.collect()

        print("Generating ICE...")
        ice_img = generate_ice_plots(final_model, X_train)
        gc.collect()

        print("Generating Permutation Importance (30 repeats)...")
        perm_img = generate_permutation_importance(final_model, X_test, y_test)
        gc.collect()

        print("Generating SHAP Dependence...")
        shap_dep_img = generate_shap_dependence(final_model, X_train, X_test)
        gc.collect()

        print("Generating DiCE Counterfactuals...")
        cf_img = generate_dice_counterfactual(final_model, X_train, X_test, y_train)
        gc.collect()

        print("Generating PCA Analysis...")
        pca_img = generate_pca_plot(X_train, y_train)
        gc.collect()

        result_msg = f"""
## ✅ Analysis Complete!

{clean_msg}

### 🏆 Model: Random Forest (forced)
- Trained on original features AND PCA-reduced features independently
- All XAI methods use the real RF models (no surrogates)

### 📊 XAI Methods Applied (11 visualizations):
| Method | Type | Details |
|--------|------|---------|
| SHAP (Original) | Global | TreeExplainer, 200 samples |
| SHAP (PCA) | Global | Separate RF on PCA features |
| LIME (Original) | Local | 10 runs, 5000 perturbations, stability score |
| LIME (PCA) | Local | 10 runs, separate RF on PCA features |
| PDP | Global | 300 samples |
| ALE | Global | 500 samples, handles correlations |
| ICE | Local/Global | 300 samples |
| Permutation | Global | 30 repeats |
| SHAP Dependence | Pairwise | 200 samples |
| DiCE Counterfactual | Local | 5 diverse counterfactuals |
| PCA Analysis | Global | 2D projection + scree plot |
        """

        return (result_msg, comparison_df,
                shap_orig_img, shap_pca_img,
                lime_orig_img, lime_pca_img,
                pdp_img, ale_img, ice_img,
                perm_img, shap_dep_img, cf_img, pca_img)

    except Exception as e:
        return (f"Error: {str(e)}\n\nTraceback:\n{tb.format_exc()}",) + (None,) * (N_OUTPUTS - 1)


# ──────────────────────────────────────────────────────────────────────────────
# GRADIO INTERFACE
# ──────────────────────────────────────────────────────────────────────────────
def create_interface():
    with gr.Blocks(title="PyCaret AutoML + Research-Grade XAI",
                   theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🤖 PyCaret AutoML with Research-Grade XAI Suite

        Upload a CSV dataset, specify the target column, and this tool will:
        1. **Clean** the data (missing values, duplicates)
        2. **Compare 10 ML models** (table only) then **force Random Forest**
        3. **Train dual RF models** (original features + PCA features)
        4. **Explain with 11 XAI visualizations** – research-grade sample sizes

        ⚡ LIME runs 10× for stability quantification · DiCE for diverse counterfactuals
        """)

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="📁 Upload CSV Dataset", file_types=[".csv"])
                target_input = gr.Textbox(
                    label="🎯 Target Column Name",
                    placeholder="e.g., diagnosis, target, class",
                    info="Enter the exact column name to predict"
                )
                run_btn = gr.Button("🚀 Run Analysis", variant="primary", size="lg")

        with gr.Row():
            result_output = gr.Markdown(label="Results")
        with gr.Row():
            comparison_output = gr.Dataframe(label="📊 Model Comparison Table (RF forced for XAI)")

        gr.Markdown("## 🔍 Explainable AI (XAI) Visualizations")

        with gr.Tabs():
            with gr.TabItem("🎯 SHAP – Original vs PCA"):
                gr.Markdown("*Two independent RF models — no surrogates*")
                with gr.Row():
                    with gr.Column():
                        shap_orig_out = gr.Image(label="SHAP (Original Features – RF)", type="pil")
                    with gr.Column():
                        shap_pca_out = gr.Image(label="SHAP (PCA Features – RF)", type="pil")

            with gr.TabItem("🔬 LIME – Original vs PCA (with stability)"):
                gr.Markdown("*10 runs per explanation · Error bars = std · Stability score shown*")
                with gr.Row():
                    with gr.Column():
                        lime_orig_out = gr.Image(label="LIME (Original – RF)", type="pil")
                    with gr.Column():
                        lime_pca_out = gr.Image(label="LIME (PCA – RF)", type="pil")

            with gr.TabItem("📈 Feature Effects (PDP + ALE)"):
                with gr.Row():
                    with gr.Column():
                        pdp_out = gr.Image(label="PDP", type="pil")
                    with gr.Column():
                        ale_out = gr.Image(label="ALE (handles correlated features)", type="pil")

            with gr.TabItem("🔗 Interactions & Heterogeneity"):
                with gr.Row():
                    with gr.Column():
                        ice_out = gr.Image(label="ICE + PDP Overlay", type="pil")
                    with gr.Column():
                        shap_dep_out = gr.Image(label="SHAP Dependence", type="pil")

            with gr.TabItem("🎯 Global Importance & Counterfactuals"):
                with gr.Row():
                    with gr.Column():
                        perm_out = gr.Image(label="Permutation Importance (30 repeats)", type="pil")
                    with gr.Column():
                        cf_out = gr.Image(label="DiCE Counterfactuals (5 diverse)", type="pil")

            with gr.TabItem("📊 PCA Analysis"):
                with gr.Row():
                    with gr.Column():
                        pca_out = gr.Image(label="PCA – 2D Projection & Variance", type="pil")

        run_btn.click(
            fn=run_analysis,
            inputs=[file_input, target_input],
            outputs=[result_output, comparison_output,
                     shap_orig_out, shap_pca_out,
                     lime_orig_out, lime_pca_out,
                     pdp_out, ale_out, ice_out,
                     perm_out, shap_dep_out, cf_out, pca_out]
        )

        gr.Markdown("""
        ---
        ### 📝 Methodology Notes

        | Method | Scope | Sample Size | Key Detail |
        |--------|-------|-------------|------------|
        | **SHAP** | Global | 200 test / 200 bg | TreeExplainer (exact for RF) |
        | **LIME** | Local | 5,000 perturbations × 10 runs | Stability quantified |
        | **PDP** | Global | 300 | Assumes uncorrelated features |
        | **ALE** | Global | 500 | Handles correlated features |
        | **ICE** | Local+Global | 300 | Shows heterogeneity |
        | **Permutation** | Global | Full test set × 30 | Most unbiased importance |
        | **SHAP Dependence** | Pairwise | 200 | Strongest interaction |
        | **DiCE** | Local | 5 diverse CFs | Actionable, diverse |
        | **PCA** | Global | Full train | Data-level analysis |

        All SHAP/LIME/PDP/ALE/ICE/Permutation run on the **same RF model** (no surrogates).
        PCA comparison uses a **separate RF trained from scratch** on PCA features.
        """)

    return interface


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 PyCaret AutoML + Research-Grade XAI Suite")
    print("=" * 60)
    print("Starting... browser will open automatically.")
    print()

    interface = create_interface()
    interface.launch(
        share=False, server_name="127.0.0.1",
        server_port=7860, inbrowser=True
    )
