import joblib
import shap
import matplotlib.pyplot as plt

# ======================
# Load model and data
# ======================
model = joblib.load("xgb_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
id_test = joblib.load("id_test.pkl")

# ======================
# Define scalar prediction function
# (THIS IS THE KEY FIX)
# ======================


def predict_malignant(X):
    """
    Returns P(malignant) as a 1D array.
    """
    return model.predict_proba(X)[:, 1]


# ======================
# User input
# ======================
i = int(input("Enter test case index (0-based): "))

sample = X_test.iloc[[i]]
case_id = id_test.iloc[i]
true_label = y_test.iloc[i]

print("\n" + "=" * 50)
print(f"Case index: {i}")
print(f"Case ID: {case_id}")
print(f"True label (1 = malignant): {true_label}")
print("=" * 50 + "\n")

# ======================
# Background dataset
# ======================
background = shap.sample(X_test, 50, random_state=42)

# ======================
# Kernel SHAP explainer
# ======================
explainer = shap.KernelExplainer(
    predict_malignant,
    background
)

# ======================
# Compute SHAP values
# ======================
shap_values = explainer.shap_values(sample)

# shap_values shape: (1, n_features)
values = shap_values[0]
base_value = explainer.expected_value

# ======================
# Build Explanation object
# ======================
explanation = shap.Explanation(
    values=values,
    base_values=base_value,
    data=sample.iloc[0],
    feature_names=sample.columns
)

# ======================
# Plot
# ======================
shap.plots.waterfall(explanation,show=False)
fig = plt.gcf()          # this is the figure SHAP created
fig.set_size_inches(15, 12)
fig.tight_layout()
# fig.subplots_adjust(left=0.5)

plt.show()


