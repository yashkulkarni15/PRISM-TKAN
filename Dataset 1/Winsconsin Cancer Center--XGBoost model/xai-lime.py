import joblib
import pandas as pd
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Load model and test data
model = joblib.load("xgb_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
id_test = joblib.load("id_test.pkl")

# Convert to numpy for LIME
X_np = X_test.values

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_np,
    feature_names=X_test.columns.tolist(),
    class_names=["Benign", "Malignant"],
    mode="classification",
    discretize_continuous=True
)

# Choose a test case
i = 27
sample = X_np[i]
case_id = id_test.iloc[i]

# Explain prediction
exp = explainer.explain_instance(
    sample,
    model.predict_proba,
    num_features=30
)

print(f"Case ID: {case_id}")
# exp.as_pyplot_figure()      # works in PyCharm
fig = exp.as_pyplot_figure()
fig.set_size_inches(14, 14)
fig.tight_layout()

fig.show()