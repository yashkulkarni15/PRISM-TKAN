import pandas as pd
from utils.load_data import load_data
from utils.load_models import load_models
from utils.metrics import evaluate

X, y, features = load_data("data/dataset1.csv")
models = load_models("model/trained_models")

rows = []
for name, model in models.items():
    m = evaluate(model, X, y)
    m["model"] = name
    rows.append(m)

df = pd.DataFrame(rows)
df.to_csv("outputs/eval/metrics.csv", index=False)
print(df.sort_values("f1", ascending=False))
