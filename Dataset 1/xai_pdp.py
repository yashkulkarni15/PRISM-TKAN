from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

from utils.load_data import load_data
from utils.load_models import load_models

X, y, feature_names = load_data("data/dataset1.csv")
models = load_models("model/trained_models")

FEATURES = [0, 1, 2]  # same indices for all models

for name, model in models.items():
    disp = PartialDependenceDisplay.from_estimator(
        model, X, FEATURES, feature_names=feature_names
    )
    plt.suptitle(name)
    plt.savefig(f"outputs/pdp/{name}.png", dpi=200)
    plt.close()
