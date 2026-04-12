import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# Load model and test data
model = joblib.load("xgb_model.pkl")
X_test = joblib.load("X_test.pkl")

features = list(X_test.columns)
# # Choose features to inspect
# features = [
#     "radius_worst",
#     "concave points_worst",
#     "area_worst"
# ]

# Plot PDPs
fig, ax = plt.subplots(nrows=6,
    ncols=5,
    figsize=(20, 18))
PartialDependenceDisplay.from_estimator(
    model,
    X_test,
    features,
    grid_resolution=50,
    ax=ax
)

plt.tight_layout()
plt.show()
