
import joblib

# Load model
model = joblib.load("xgb_model.pkl")

# Load data
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
id_test = joblib.load("id_test.pkl")

# Pick a single case
i = 45  # change this index to test different cases
sample = X_test.iloc[[i]]
true_label = y_test.iloc[i]
case_id = id_test.iloc[i]

pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0][1]

# Print Result
print("=" * 20)
print(f"Test index: {i}")
print(f"Case ID: {case_id}")
print(f"True label (1 = malignant): {true_label}")
print(f"Predicted label: {pred}")
print(f"Predicted malignant probability: {prob:.3f}")
print("=" * 30)
# print("Prediction (1 = malignant):", pred)
# print("Malignancy probability:", prob)
