"""
PyCaret Automation Script with GUI and XAI Tools
================================================
A simple script that provides a Gradio GUI for:
1. Uploading CSV datasets
2. Performing data cleaning
3. Comparing 10 ML models using PyCaret
4. Explaining the best model with SHAP, LIME, and PDP

OPTIMIZED VERSION - Reduced memory usage and faster execution
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '2'  # Limit parallel jobs

import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# PyCaret
from pycaret.classification import ClassificationExperiment

# XAI Libraries
import shap
shap.initjs()  # Initialize SHAP JS
import lime
import lime.lime_tabular
from sklearn.inspection import PartialDependenceDisplay

import io
from PIL import Image

# Performance optimization
import gc


def clean_data(df):
    """
    Perform basic data cleaning on the dataframe.
    """
    original_rows = len(df)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Remove any completely empty rows
    df = df.dropna(how='all')
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For numeric columns, fill NaN with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill NaN with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
    
    cleaned_rows = len(df)
    
    return df, f"Cleaned: {original_rows - cleaned_rows} rows removed, {cleaned_rows} rows remaining"


def run_analysis(file, target_column):
    """
    Main analysis function that runs PyCaret model comparison and XAI explanations.
    """
    if file is None:
        return "Please upload a CSV file.", None, None, None, None
    
    if not target_column:
        return "Please specify a target column.", None, None, None, None
    
    try:
        # Load the data
        df = pd.read_csv(file.name)
        
        if target_column not in df.columns:
            return f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}", None, None, None, None
        
        # Clean the data
        df, clean_msg = clean_data(df)
        
        # Remove ID-like columns (usually first column if it's just an index)
        if df.columns[0].lower() in ['id', 'index', 'unnamed: 0']:
            df = df.drop(columns=[df.columns[0]])
        
        # Also remove diagnosis_numeric if diagnosis exists (for breast cancer dataset)
        if 'diagnosis_numeric' in df.columns and 'diagnosis' in df.columns:
            df = df.drop(columns=['diagnosis_numeric'])
        
        # PERFORMANCE: Limit dataset size for faster processing
        if len(df) > 500:
            df = df.sample(n=500, random_state=42)
            clean_msg += f" (Sampled to 500 rows for faster processing)"
        
        # Setup PyCaret Classification Experiment with reduced complexity
        exp = ClassificationExperiment()
        exp.setup(
            data=df,
            target=target_column,
            session_id=42,
            verbose=False,
            html=False,
            log_experiment=False,
            system_log=False,
            n_jobs=1,  # Single thread to reduce CPU usage
            train_size=0.8,  # Use 80% for training
            fold=3  # Reduce cross-validation folds
        )
        
        # Define 10 models to compare (using faster models only)
        models_to_compare = [
            'lr',      # Logistic Regression
            'knn',     # K-Neighbors Classifier
            'nb',      # Naive Bayes
            'dt',      # Decision Tree Classifier
            'rf',      # Random Forest Classifier
            'ada',     # AdaBoost Classifier
            'gbc',     # Gradient Boosting Classifier
            'lightgbm',# LightGBM Classifier
            'et',      # Extra Trees Classifier
            'ridge'    # Ridge Classifier
        ]
        
        # Compare models with cross-validation disabled for speed
        best_model = exp.compare_models(include=models_to_compare, n_select=1, cross_validation=False)
        
        # Get comparison results
        comparison_df = exp.pull()
        
        # Finalize the best model
        final_model = exp.finalize_model(best_model)
        
        # Get training data for XAI
        X_train = exp.get_config('X_train')
        X_test = exp.get_config('X_test')
        
        # Force garbage collection to free memory
        gc.collect()
        
        # Generate SHAP plot
        shap_img = generate_shap_plot(final_model, X_train, X_test, exp)
        gc.collect()
        
        # Generate LIME explanation
        lime_img = generate_lime_explanation(final_model, X_train, X_test)
        gc.collect()
        
        # Generate PDP plots
        pdp_img = generate_pdp_plots(final_model, X_train)
        gc.collect()
        
        result_msg = f"""
## ✅ Analysis Complete!

{clean_msg}

### 🏆 Best Model: {type(final_model).__name__}
        """
        
        return result_msg, comparison_df, shap_img, lime_img, pdp_img
        
    except Exception as e:
        import traceback
        return f"Error during analysis: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", None, None, None, None


def generate_shap_plot(model, X_train, X_test, exp):
    """
    Generate SHAP summary plot for feature importance.
    """
    try:
        plt.close('all')
        
        # Use only a small sample for SHAP (reduces computation significantly)
        n_samples = min(30, len(X_test))
        n_background = min(50, len(X_train))
        
        X_test_sample = X_test.iloc[:n_samples].copy()
        X_train_sample = X_train.iloc[:n_background].copy()
        
        # Get the pipeline's final estimator if it's a pipeline
        if hasattr(model, 'named_steps'):
            # It's a pipeline, get the actual classifier
            actual_model = model.named_steps.get('actual_estimator', model.steps[-1][1])
        else:
            actual_model = model
        
        # Try TreeExplainer first (fast for tree-based models)
        shap_values = None
        explainer = None
        
        model_name = type(actual_model).__name__.lower()
        
        if any(tree_type in model_name for tree_type in ['forest', 'tree', 'boost', 'lgbm', 'xgb', 'catboost']):
            try:
                explainer = shap.TreeExplainer(actual_model)
                shap_values = explainer.shap_values(X_test_sample)
            except Exception as e:
                print(f"TreeExplainer failed: {e}")
        
        # If TreeExplainer didn't work, try LinearExplainer
        if shap_values is None and 'logistic' in model_name or 'linear' in model_name or 'ridge' in model_name:
            try:
                explainer = shap.LinearExplainer(actual_model, X_train_sample)
                shap_values = explainer.shap_values(X_test_sample)
            except Exception as e:
                print(f"LinearExplainer failed: {e}")
        
        # As a last resort, use Permutation explainer (slower but works with any model)
        if shap_values is None:
            try:
                explainer = shap.Explainer(actual_model.predict, X_train_sample)
                shap_values = explainer(X_test_sample).values
            except Exception as e:
                print(f"Permutation explainer failed: {e}")
                # Final fallback: use feature importances if available
                if hasattr(actual_model, 'feature_importances_'):
                    return generate_feature_importance_plot(actual_model, X_train)
                raise e
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_sample, show=False, max_display=10)
        plt.title("SHAP Feature Importance Summary", fontsize=12)
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        plt.close('all')
        
        return Image.open(buf)
        
    except Exception as e:
        print(f"SHAP error: {e}")
        # Try to generate feature importance instead
        try:
            return generate_feature_importance_plot(model, X_train)
        except:
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, f"SHAP unavailable:\n{str(e)[:100]}", 
                    ha='center', va='center', fontsize=9, wrap=True)
            plt.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            buf.seek(0)
            plt.close('all')
            
            return Image.open(buf)


def generate_feature_importance_plot(model, X_train):
    """
    Generate a simple feature importance bar plot as fallback.
    """
    plt.figure(figsize=(10, 6))
    
    # Get the actual model from pipeline if needed
    actual_model = model
    if hasattr(model, 'named_steps'):
        actual_model = model.steps[-1][1]
    
    if hasattr(actual_model, 'feature_importances_'):
        importance = actual_model.feature_importances_
        features = X_train.columns
        
        # Sort by importance
        indices = np.argsort(importance)[-10:]  # Top 10
        
        plt.barh(range(len(indices)), importance[indices], align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance (Top 10)')
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "Feature importance not available for this model", 
                ha='center', va='center')
        plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    plt.close('all')
    
    return Image.open(buf)


def generate_lime_explanation(model, X_train, X_test):
    """
    Generate LIME explanation for a sample prediction.
    """
    try:
        plt.close('all')
        
        # Convert to numpy arrays to avoid column name issues
        X_train_np = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
        X_test_np = X_test.values if hasattr(X_test, 'values') else np.array(X_test)
        
        # Get feature names
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X_train_np.shape[1])]
        
        # Create LIME explainer with numpy arrays
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_np,
            feature_names=feature_names,
            class_names=['Class 0', 'Class 1'],
            mode='classification',
            discretize_continuous=True
        )
        
        # Get sample as 1D numpy array
        sample = X_test_np[0]
        
        # Create predict_proba wrapper that accepts numpy arrays
        def safe_predict_proba(X):
            try:
                # Convert to DataFrame with proper column names for model
                if hasattr(X_train, 'columns'):
                    X_df = pd.DataFrame(X, columns=X_train.columns)
                    return model.predict_proba(X_df)
                else:
                    return model.predict_proba(X)
            except Exception as e:
                print(f"predict_proba error: {e}")
                # If predict_proba fails, create dummy probabilities
                try:
                    if hasattr(X_train, 'columns'):
                        X_df = pd.DataFrame(X, columns=X_train.columns)
                        preds = model.predict(X_df)
                    else:
                        preds = model.predict(X)
                    proba = np.zeros((len(preds), 2))
                    for i, p in enumerate(preds):
                        proba[i, int(p) if isinstance(p, (int, float, np.integer)) else 0] = 1.0
                    return proba
                except:
                    return np.array([[0.5, 0.5]] * len(X))
        
        explanation = explainer.explain_instance(
            sample,
            safe_predict_proba,
            num_features=8,
            num_samples=500
        )
        
        # Get the figure from LIME
        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(10, 5)
        plt.title(f"LIME Explanation for Sample #0", fontsize=12)
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        plt.close('all')
        
        return Image.open(buf)
        
    except Exception as e:
        print(f"LIME error: {e}")
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, f"LIME unavailable:\n{str(e)[:100]}", 
                ha='center', va='center', fontsize=9, wrap=True)
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        plt.close('all')
        
        return Image.open(buf)


def generate_pdp_plots(model, X_train):
    """
    Generate Partial Dependence Plots for top features.
    """
    try:
        plt.close('all')
        
        # Get the actual model
        actual_model = model
        if hasattr(model, 'named_steps'):
            actual_model = model.steps[-1][1]
        
        # Use smaller sample for PDP
        X_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)
        
        # Select top 4 features based on feature importance if available
        n_features = min(4, len(X_train.columns))
        
        if hasattr(actual_model, 'feature_importances_'):
            importance = actual_model.feature_importances_
            top_indices = np.argsort(importance)[-n_features:][::-1]
            features = [X_train.columns[i] for i in top_indices]
        else:
            features = list(X_train.columns[:n_features])
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        # Create PDP for each feature
        for idx, feature in enumerate(features):
            if idx >= 4:
                break
            try:
                PartialDependenceDisplay.from_estimator(
                    model, 
                    X_sample, 
                    features=[feature],
                    ax=axes[idx],
                    kind='average',
                    n_jobs=1
                )
                axes[idx].set_title(f"PDP: {feature}", fontsize=10)
            except Exception as e:
                axes[idx].text(0.5, 0.5, f"Error: {str(e)[:40]}", 
                              ha='center', va='center', fontsize=8)
                axes[idx].set_title(f"PDP: {feature} (failed)")
        
        # Hide unused subplots
        for idx in range(len(features), 4):
            axes[idx].set_visible(False)
        
        plt.suptitle("Partial Dependence Plots (Top Features)", fontsize=12)
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        plt.close('all')
        
        return Image.open(buf)
        
    except Exception as e:
        print(f"PDP error: {e}")
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, f"PDP unavailable:\n{str(e)[:100]}", 
                ha='center', va='center', fontsize=9, wrap=True)
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        plt.close('all')
        
        return Image.open(buf)


# Create Gradio Interface
def create_interface():
    """
    Create the Gradio web interface.
    """
    with gr.Blocks(title="PyCaret AutoML with XAI", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🤖 PyCaret AutoML with XAI Tools
        
        Upload a CSV dataset, specify the target column, and this tool will:
        1. **Clean the data** (handle missing values, duplicates)
        2. **Compare 10 ML models** using PyCaret
        3. **Explain the best model** using SHAP, LIME, and PDP
        
        ⚡ *Optimized for performance - uses sampling for large datasets*
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="📁 Upload CSV Dataset", file_types=[".csv"])
                target_input = gr.Textbox(
                    label="🎯 Target Column Name",
                    placeholder="e.g., diagnosis, target, class",
                    info="Enter the exact name of the column you want to predict"
                )
                run_btn = gr.Button("🚀 Run Analysis", variant="primary", size="lg")
        
        with gr.Row():
            result_output = gr.Markdown(label="Results")
        
        with gr.Row():
            comparison_output = gr.Dataframe(label="📊 Model Comparison Table")
        
        gr.Markdown("## 🔍 Explainable AI (XAI) Visualizations")
        
        with gr.Row():
            with gr.Column():
                shap_output = gr.Image(label="SHAP / Feature Importance", type="pil")
            with gr.Column():
                lime_output = gr.Image(label="LIME Explanation", type="pil")
        
        with gr.Row():
            pdp_output = gr.Image(label="Partial Dependence Plots", type="pil")
        
        # Connect the button to the analysis function
        run_btn.click(
            fn=run_analysis,
            inputs=[file_input, target_input],
            outputs=[result_output, comparison_output, shap_output, lime_output, pdp_output]
        )
        
        gr.Markdown("""
        ---
        ### 📝 Notes:
        - **SHAP/Feature Importance**: Shows which features matter most for predictions
        - **LIME**: Explains individual predictions locally
        - **PDP**: Shows how each feature affects the prediction on average
        
        ### 🧪 Test with the breast cancer dataset:
        - Upload `breast_cancer_cleaned.csv`
        - Set target column to `diagnosis`
        """)
    
    return interface


if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Starting PyCaret AutoML with XAI Tools...")
    print("=" * 50)
    print("Browser will open automatically...")
    print("")
    
    interface = create_interface()
    
    # Launch with auto-open browser
    interface.launch(
        share=False, 
        server_name="127.0.0.1", 
        server_port=7860,
        inbrowser=True  # Auto-open browser
    )
