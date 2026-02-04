import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample


# =========================
# Utility: section header
# =========================
def banner(title: str):
    print("\n" + "=" * 85)
    print(title)
    print("=" * 85)


# =========================
# PART 1 — Data Preparation
# =========================
banner("PART 1 — Load dataset and prepare features")

df = pd.read_csv("dataset1.csv")

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop(columns=["diagnosis"])
if "id" in X.columns:
    X = X.drop(columns=["id"])
y = df["diagnosis"]

print(f"Dataset shape: {df.shape}")
print(f"Feature matrix shape: {X.shape}")
print(f"Label vector shape: {y.shape}")

print("\nClass distribution (1 = Malignant):")
print(y.value_counts(normalize=True).rename("proportion"))

missing_total = X.isna().sum().sum()
print(f"\nTotal missing values in features: {missing_total}")
if missing_total > 0:
    print("Warning: Missing values detected. Imputation would be required.")
else:
    print("No missing values detected.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


# =========================
# PART 2 — Black-box Model A
# =========================
banner("PART 2 — Black-box Model: RandomForest")

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("Training RandomForest classifier...")
rf.fit(X_train, y_train)

rf_prob = rf.predict_proba(X_test)[:, 1]
rf_pred = (rf_prob >= 0.5).astype(int)

print("\nRandomForest performance on diagnosis prediction:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred, target_names=["Benign", "Malignant"]))


# =========================
# PART 3 — Black-box Model B
# =========================
banner("PART 3 — Black-box Model: HistGradientBoosting")

hgb = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=400,
    random_state=42
)

print("Training HistGradientBoosting classifier...")
hgb.fit(X_train, y_train)

hgb_prob = hgb.predict_proba(X_test)[:, 1]
hgb_pred = (hgb_prob >= 0.5).astype(int)

print("\nHistGradientBoosting performance on diagnosis prediction:")
print("Accuracy:", accuracy_score(y_test, hgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, hgb_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, hgb_pred))
print(classification_report(y_test, hgb_pred, target_names=["Benign", "Malignant"]))


# =========================
# PART 4 — Linear Surrogate Model
# =========================
banner("PART 4 — Linear Surrogate to Approximate RandomForest Behavior")

print("Objective:")
print("We approximate the RandomForest prediction probabilities using a linear model.")
print("The goal is not to predict the true label, but to understand the global behavior of the black-box model.\n")

# Behavioral labels
y_bb_train = rf.predict_proba(X_train)[:, 1]
y_bb_test  = rf.predict_proba(X_test)[:, 1]

print("Behavioral labels created from RandomForest predicted probabilities.")
print(f"Probability range (train): {y_bb_train.min():.4f} to {y_bb_train.max():.4f}")

# Standardization
print("\nStandardizing features for interpretability of coefficients...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Fit surrogate
print("Training linear surrogate model...")
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_bb_train)

print("\nLinear model form:")
print("Surrogate ≈ intercept + Σ(coefficient_i × standardized_feature_i)")
print(f"Intercept: {linreg.intercept_:.6f}")

# Fidelity evaluation
print("\nEvaluating surrogate fidelity...")
surr_pred = linreg.predict(X_test_scaled)
surr_pred = np.clip(surr_pred, 0, 1)

mse = mean_squared_error(y_bb_test, surr_pred)
r2  = r2_score(y_bb_test, surr_pred)
corr = np.corrcoef(y_bb_test, surr_pred)[0, 1]

print(f"MSE:  {mse:.6f}")
print(f"R^2:  {r2:.6f}")
print(f"Correlation: {corr:.6f}")

print("\nInterpretation:")
print("Higher R^2 and correlation indicate that the linear model closely approximates the black-box behavior.")
print("Lower values suggest strong nonlinear patterns in the RandomForest decision function.")


# =========================
# PART 5 — Coefficient Interpretation
# =========================
banner("PART 5 — Global Interpretation of Linear Coefficients")

coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": linreg.coef_,
    "absolute_value": np.abs(linreg.coef_)
}).sort_values("absolute_value", ascending=False)

print("Top 15 features ranked by absolute coefficient magnitude:")
print(coef_df.head(15).to_string(index=False))

print("\nInterpretation:")
print("Because features are standardized, each coefficient represents:")
print("'Change in RandomForest predicted probability associated with a 1 standard deviation increase in the feature.'")
print("Positive coefficient → increases predicted malignancy probability.")
print("Negative coefficient → decreases predicted malignancy probability.")
print("This is a global linear approximation, not a causal interpretation.")


# =========================
# PART 6 — Bootstrap Stability
# =========================
banner("PART 6 — Bootstrap Stability Analysis of Coefficients")

print("We assess coefficient stability using bootstrap resampling.")
print("Stable coefficients indicate reliable global explanations.")
print("Unstable coefficients suggest sensitivity to sampling or nonlinear behavior.\n")

B = 200
rng = np.random.default_rng(42)

coef_matrix = np.zeros((B, X_train_scaled.shape[1]))
n = X_train_scaled.shape[0]

for b in range(B):
    idx = rng.integers(0, n, size=n)
    Xb = X_train_scaled[idx]
    yb = y_bb_train[idx]

    model = LinearRegression().fit(Xb, yb)
    coef_matrix[b, :] = model.coef_

coef_mean = coef_matrix.mean(axis=0)
coef_std  = coef_matrix.std(axis=0)
ci_low  = np.percentile(coef_matrix, 2.5, axis=0)
ci_high = np.percentile(coef_matrix, 97.5, axis=0)

stability_df = pd.DataFrame({
    "feature": X_train.columns,
    "mean_coef": coef_mean,
    "std_coef": coef_std,
    "ci_2.5%": ci_low,
    "ci_97.5%": ci_high
}).sort_values("std_coef")

print("Top 15 most stable coefficients (lowest std):")
print(stability_df.head(15).to_string(index=False))

print("\nInterpretation:")
print("- Low standard deviation → coefficient is stable across bootstrap samples.")
print("- Confidence interval not crossing zero → direction is consistently positive or negative.")
print("- Confidence interval crossing zero → direction may change; explanation is unstable.")


# =========================
# PART 7 — Visualization
# =========================
banner("PART 7 — Surrogate Fidelity Visualization")

plt.figure(figsize=(6, 6))
plt.scatter(y_bb_test, surr_pred, alpha=0.5)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("RandomForest predicted probability")
plt.ylabel("Linear surrogate prediction")
plt.title("Fidelity of Linear Surrogate vs RandomForest")
plt.tight_layout()
plt.show()

residuals = y_bb_test - surr_pred
plt.figure(figsize=(6, 4))
plt.scatter(surr_pred, residuals, alpha=0.5)
plt.axhline(0, linestyle="--")
plt.xlabel("Surrogate prediction")
plt.ylabel("Residual (RF - Surrogate)")
plt.title("Residual Analysis")
plt.tight_layout()
plt.show()

print("\nExecution completed.")
