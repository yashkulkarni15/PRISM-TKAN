import matplotlib.pyplot as plt
import numpy as np

from utils.load_data import load_data
from utils.load_models import load_models

X, y, _ = load_data("data/breast_cancer.csv")
models = load_models("models/trained_models")

for name, model in models.items():
    probs = model.predict_proba(X)[:, 1]

    plt.hist(probs[y == 1], bins=30, alpha=0.6, label="Malignant")
    plt.hist(probs[y == 0], bins=30, alpha=0.6, label="Benign")
    plt.legend()
    plt.title(name)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")

    plt.savefig(f"outputs/diagnostics/{name}_prob_hist.png", dpi=200)
    plt.close()
