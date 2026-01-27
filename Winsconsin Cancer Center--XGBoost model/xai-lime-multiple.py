import joblib
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ======================
# Load model and test data
# ======================
model = joblib.load("xgb_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
id_test = joblib.load("id_test.pkl")

X_np = X_test.values

# ======================
# User input (simple interface)
# ======================
i = int(input("Enter test case index (0-based): "))
n_runs = int(input("Enter number of LIME runs: "))

sample = X_np[i]
case_id = id_test.iloc[i]
true_label = y_test.iloc[i]

print("\n" + "=" * 50)
print(f"Case index: {i}")
print(f"Case ID: {case_id}")
print(f"True label (1 = malignant): {true_label}")
print("=" * 50 + "\n")

# ======================
# LIME explainer
# ======================
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_np,
    feature_names=X_test.columns.tolist(),
    class_names=["Benign", "Malignant"],
    mode="classification",
    discretize_continuous=True
)

# ======================
# Run LIME multiple times
# ======================
weights = defaultdict(list)

for _ in range(n_runs):
    exp = explainer.explain_instance(
        sample,
        model.predict_proba,
        num_features=30
    )
    for feature, weight in exp.as_list():
        weights[feature].append(weight)

# ======================
# Aggregate (mean weights)
# ======================
mean_weights = {f: np.mean(ws) for f, ws in weights.items()}

# Sort by absolute importance
sorted_items = sorted(
    mean_weights.items(),
    key=lambda x: abs(x[1]),
    reverse=True
)

# Keep top 30 (or fewer if not available)
top_items = sorted_items[:30]

features = [f for f, _ in top_items]
values = [w for _, w in top_items]

# ======================
# Plot (LIME-style bar chart, but stable)
# ======================
plt.figure(figsize=(14, 14))
plt.barh(features[::-1], values[::-1])
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Mean LIME weight")
plt.title(f"Stable LIME explanation (N = {n_runs})")
plt.tight_layout()
plt.show()
