import pandas as pd
from sklearn.inspection import permutation_importance

from utils.load_data import load_data
from utils.load_models import load_models

X, y, feature_names = load_data("data/dataset1.csv")
models = load_models("model/trained_models")

for name, model in models.items():
    r = permutation_importance(
        model, X, y,
        n_repeats=20,
        scoring="f1",
        random_state=42
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": r.importances_mean
    }).sort_values("importance", ascending=False)

    df.to_csv(f"outputs/permutation/{name}.csv", index=False)
