import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import joblib


# 1. Load data
df = pd.read_csv("dataset1.csv")
# Keep ID separately
ids = df["id"]

# 2. Separate features and label
X = df.drop(columns=["id", "diagnosis"])
y = (df["diagnosis"] == "M").astype(int)

# 3. Train/test split
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    ids,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4. Create model
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# 5. Train model
model.fit(X_train, y_train)

# 6. Accuracy Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUROC:   {auc:.4f}")

# 7. Save model
joblib.dump(model, "xgb_model.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(id_test, "id_test.pkl")

print("Training complete. Model and test set saved.")