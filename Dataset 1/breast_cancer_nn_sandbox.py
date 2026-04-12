# nn_breast_cancer_sandbox.py
# Run: python nn_breast_cancer_sandbox.py
#
# pip install numpy pandas scikit-learn joblib matplotlib

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Any, List

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# -----------------------------
# Model Spec (explicit description)
# -----------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    group: str                 # A/B/C/D groups
    solver: str                # 'adam', 'sgd', 'lbfgs'
    activation: str            # 'relu', 'tanh', 'logistic' (sigmoid)
    hidden_layers: Tuple[int, ...]  # e.g. (32,16) means two hidden layers
    learning_rate_init: Optional[float]  # None allowed for lbfgs (not comparable)
    alpha: float               # L2 weight decay
    max_iter: int
    early_stopping: bool
    changed: str               # explicit "what changed vs baseline"
    notes: str = ""            # any extra explanation


def make_pipeline(spec: ModelSpec, random_state: int = 42) -> Pipeline:
    """
    Builds a StandardScaler + MLPClassifier pipeline based on spec.
    """
    # Solver-specific defaults
    mlp_kwargs: Dict[str, Any] = dict(
        hidden_layer_sizes=spec.hidden_layers,
        activation=spec.activation,
        solver=spec.solver,
        alpha=spec.alpha,
        max_iter=spec.max_iter,
        random_state=random_state,
    )

    if spec.solver in ("adam", "sgd"):
        # These solvers meaningfully use learning_rate_init
        if spec.learning_rate_init is None:
            raise ValueError(f"{spec.name}: learning_rate_init must be set for solver={spec.solver}")
        mlp_kwargs.update(
            learning_rate_init=spec.learning_rate_init,
            early_stopping=spec.early_stopping,
            validation_fraction=0.2,
            n_iter_no_change=10,
        )

        if spec.solver == "sgd":
            # Reasonable SGD defaults; feel free to tune later.
            mlp_kwargs.update(
                momentum=0.9,
                nesterovs_momentum=True,
                learning_rate="adaptive",  # reduces lr when plateauing
            )

    else:
        # lbfgs: early stopping not used; learning rate not comparable
        mlp_kwargs.update(early_stopping=False)

    mlp = MLPClassifier(**mlp_kwargs)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp),
    ])


def evaluate(pipe: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Compute clinically-relevant metrics (treat class 0=malignant as "positive" if you want).
    Note: sklearn dataset uses 0=malignant, 1=benign.

    We'll compute metrics with "malignant" as the positive class (pos_label=0),
    since recall for malignant is often the key.
    """
    y_prob_benign = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    # Convert to "malignant probability" if you want:
    y_prob_malignant = 1.0 - y_prob_benign

    # Treat malignant (0) as positive class:
    # We need to map labels to {0,1} with malignant=1 for sklearn metric functions
    y_true_m = (y_test == 0).astype(int)
    y_pred_m = (y_pred == 0).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc_malignant": roc_auc_score(y_true_m, y_prob_malignant),
        "precision_malignant": precision_score(y_true_m, y_pred_m, zero_division=0),
        "recall_malignant": recall_score(y_true_m, y_pred_m, zero_division=0),
        "f1_malignant": f1_score(y_true_m, y_pred_m, zero_division=0),
        "confusion_matrix_malignant_[TP_FN;FP_TN]": confusion_matrix(y_true_m, y_pred_m).tolist(),
        "n_iter": getattr(pipe.named_steps["mlp"], "n_iter_", None),
        "final_loss": getattr(pipe.named_steps["mlp"], "loss_", None),
    }


def save_models(models: Dict[str, Pipeline], folder: str = "trained_models") -> None:
    os.makedirs(folder, exist_ok=True)
    for name, pipe in models.items():
        path = os.path.join(folder, f"{name}.joblib")
        joblib.dump(pipe, path)
    print(f"Saved {len(models)} models to: {folder}/")


def decision_boundary_pca(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    out_png: str,
    grid_res: int = 300,
) -> None:
    """
    Plot model behavior in a 2D PCA projection, while evaluating the *actual 30D model*.

    Steps:
      1) Fit StandardScaler + PCA(2) on X_train (for visualization coordinates)
      2) Make a 2D grid in PCA space
      3) Map grid back to 30D (inverse PCA + inverse scaling)
      4) Evaluate model.predict_proba on those 30D points
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(Xs)

    x_min, x_max = X2[:, 0].min() - 1.0, X2[:, 0].max() + 1.0
    y_min, y_max = X2[:, 1].min() - 1.0, X2[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    # Map back to original feature space
    grid_scaled_30d = pca.inverse_transform(grid_2d)
    grid_30d = scaler.inverse_transform(grid_scaled_30d)

    # Probability of malignant = 1 - P(benign)
    Z = 1.0 - model.predict_proba(grid_30d)[:, 1]
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, levels=50, alpha=0.35)

    malignant = (y_train == 0)
    benign = (y_train == 1)

    plt.scatter(X2[malignant, 0], X2[malignant, 1], marker="s", alpha=0.7, label="malignant (0)")
    plt.scatter(X2[benign, 0], X2[benign, 1], marker="o", alpha=0.7, label="benign (1)")

    plt.title("Decision boundary in PCA space (prob malignant)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved decision boundary: {out_png}")


def build_12_specs() -> List[ModelSpec]:
    """
    The 12 models, grouped A/B/C/D, with explicit 'changed' descriptions.
    """
    # Baseline shared defaults
    alpha = 1e-4
    max_iter = 600

    # Group A — Adam: activation × depth/width
    A = [
        ModelSpec(
            name="A1_adam_relu_1x16_lr1e-3",
            group="A",
            solver="adam",
            activation="relu",
            hidden_layers=(16,),
            learning_rate_init=1e-3,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            changed="Baseline",
        ),
        ModelSpec(
            name="A2_adam_relu_1x32_lr1e-3",
            group="A",
            solver="adam",
            activation="relu",
            hidden_layers=(32,),
            learning_rate_init=1e-3,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            changed="Width: (16,) -> (32,)",
        ),
        ModelSpec(
            name="A3_adam_relu_2x32_16_lr1e-3",
            group="A",
            solver="adam",
            activation="relu",
            hidden_layers=(32, 16),
            learning_rate_init=1e-3,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            changed="Depth: 1 hidden layer -> 2 hidden layers (32,16)",
        ),
        ModelSpec(
            name="A4_adam_tanh_1x16_lr1e-3",
            group="A",
            solver="adam",
            activation="tanh",
            hidden_layers=(16,),
            learning_rate_init=1e-3,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            changed="Activation: relu -> tanh (same as baseline otherwise)",
        ),
        ModelSpec(
            name="A5_adam_sigmoid_1x16_lr1e-3",
            group="A",
            solver="adam",
            activation="logistic",  # sigmoid in sklearn
            hidden_layers=(16,),
            learning_rate_init=1e-3,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            changed="Activation: relu -> sigmoid/logistic (same as baseline otherwise)",
        ),
    ]

    # Group B — Adam: learning rate sensitivity
    B = [
        ModelSpec(
            name="B1_adam_relu_1x16_lr3e-4",
            group="B",
            solver="adam",
            activation="relu",
            hidden_layers=(16,),
            learning_rate_init=3e-4,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            changed="Learning rate: 1e-3 -> 3e-4 (baseline architecture)",
        ),
        ModelSpec(
            name="B2_adam_relu_2x32_16_lr3e-4",
            group="B",
            solver="adam",
            activation="relu",
            hidden_layers=(32, 16),
            learning_rate_init=3e-4,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            changed="Learning rate: 1e-3 -> 3e-4 (deeper architecture 32,16)",
        ),
    ]

    # Group C — SGD: solver effects (use a larger LR typical for SGD)
    C = [
        ModelSpec(
            name="C1_sgd_relu_1x16_lr1e-2",
            group="C",
            solver="sgd",
            activation="relu",
            hidden_layers=(16,),
            learning_rate_init=1e-2,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            changed="Solver: adam -> sgd (baseline arch), LR adjusted to 1e-2",
        ),
        ModelSpec(
            name="C2_sgd_relu_2x32_16_lr1e-2",
            group="C",
            solver="sgd",
            activation="relu",
            hidden_layers=(32, 16),
            learning_rate_init=1e-2,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            changed="Solver: adam -> sgd + Depth: (32,16), LR=1e-2",
        ),
        ModelSpec(
            name="C3_sgd_tanh_1x16_lr1e-2",
            group="C",
            solver="sgd",
            activation="tanh",
            hidden_layers=(16,),
            learning_rate_init=1e-2,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            changed="Solver: sgd + Activation: relu -> tanh (baseline arch), LR=1e-2",
        ),
    ]

    # Group D — LBFGS: quasi-Newton (small-data optimizer)
    D = [
        ModelSpec(
            name="D1_lbfgs_relu_1x16",
            group="D",
            solver="lbfgs",
            activation="relu",
            hidden_layers=(16,),
            learning_rate_init=None,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=False,
            changed="Solver: adam -> lbfgs (baseline arch)",
            notes="LBFGS does not use early_stopping/minibatches; LR is not comparable.",
        ),
        ModelSpec(
            name="D2_lbfgs_tanh_2x32_16",
            group="D",
            solver="lbfgs",
            activation="tanh",
            hidden_layers=(32, 16),
            learning_rate_init=None,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=False,
            changed="Solver: lbfgs + Activation: tanh + Depth: (32,16)",
            notes="LBFGS is often strong on small tabular data; can converge differently than Adam/SGD.",
        ),
    ]

    return A + B + C + D


def main() -> None:
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target  # 0=malignant, 1=benign

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build specs
    specs_list = build_12_specs()

    # Train explicit models
    specs: Dict[str, ModelSpec] = {}
    models: Dict[str, Pipeline] = {}
    metrics: Dict[str, Dict[str, Any]] = {}

    for spec in specs_list:
        specs[spec.name] = spec
        pipe = make_pipeline(spec, random_state=42)
        pipe.fit(X_train, y_train)

        models[spec.name] = pipe
        metrics[spec.name] = evaluate(pipe, X_test, y_test)

        # Minimal training confirmation + model description line
        print(f"\nTrained: {spec.name}")
        print(f"  Group: {spec.group}")
        print(f"  Changed: {spec.changed}")
        print(f"  Solver={spec.solver}, Activation={spec.activation}, Hidden={spec.hidden_layers}, "
              f"LR={spec.learning_rate_init}, alpha={spec.alpha}")
        print(f"  Test malignant recall={metrics[spec.name]['recall_malignant']:.3f}, "
              f"F1={metrics[spec.name]['f1_malignant']:.3f}, "
              f"AUC(malignant)={metrics[spec.name]['roc_auc_malignant']:.3f}")

    # Save trained models
    save_models(models, folder="trained_models")

    # Save specs + metrics (for optional report later)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({k: asdict(v) for k, v in specs.items()}, "artifacts/model_specs.joblib")
    joblib.dump(metrics, "artifacts/model_metrics.joblib")
    print("\nSaved specs+metrics to artifacts/ (for optional report later).")

    # Decision boundary plot for one model (pick whichever you like)
    chosen = "A2_adam_relu_1x32_lr1e-3"
    decision_boundary_pca(
        models[chosen],
        X_train,
        y_train,
        out_png=f"artifacts/decision_boundary_{chosen}.png",
    )

    # EXAMPLE: using a trained model explicitly
    # (e.g., run predictions on the first 3 test samples)
    sample = X_test[:3]
    mdl = models[chosen]
    prob_benign = mdl.predict_proba(sample)[:, 1]
    pred = mdl.predict(sample)
    print(f"\nExample predictions using {chosen}:")
    for i in range(len(sample)):
        print(f"  i={i} pred={pred[i]}  P(benign)={prob_benign[i]:.3f}  P(malignant)={1-prob_benign[i]:.3f}")


if __name__ == "__main__":
    main()
