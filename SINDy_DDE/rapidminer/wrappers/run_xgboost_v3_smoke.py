#!/usr/bin/env python3
"""Small RapidMiner-compatible smoke test for the archived XGBoost v3 pipeline.

This intentionally keeps the same v3 feature-exclusion rules but avoids SHAP,
LIME, DiCE, and large plot generation. RapidMiner can call this with Execute
Program and then read the output CSVs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score, recall_score, roc_auc_score
import xgboost as xgb


REPO_ROOT = Path(__file__).resolve().parents[2]

ALWAYS_EXCLUDE = {
    "dataset",
    "file",
    "label",
    "b_over_blim",
    "blim",
    "b_mm",
    "noise_std",
    "omega",
    "norm_xtpos_rms",
    "norm_ytpos_rms",
    "norm_Forcex_rms",
    "norm_Forcey_rms",
}

PHYSICS_PARAMS = {
    "fn",
    "ktc",
    "teeth",
    "damping",
    "cut_dir",
    "diameter_mm",
    "ktx",
    "zetatx",
    "fnty",
    "ro",
}

EXCLUDE = ALWAYS_EXCLUDE | PHYSICS_PARAMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-csv",
        default=str(REPO_ROOT / "data/metadata/features_all_datasets.csv"),
        help="Clean feature CSV.",
    )
    parser.add_argument(
        "--noisy-csv",
        default=str(REPO_ROOT / "data/metadata/features_noisy_datasets.csv"),
        help="Optional noisy feature CSV.",
    )
    parser.add_argument(
        "--datasets",
        default="1-5",
        help="Clean dataset IDs/ranges for the smoke test, e.g. 1-5 or 1,4,11.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "rapidminer/outputs/xgboost_v3_smoke"),
        help="Output directory for RapidMiner-readable CSVs.",
    )
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-noisy",
        action="store_true",
        help="Skip noisy mirror evaluation even if --noisy-csv exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = Path(args.clean_csv)
    if not clean_path.exists():
        raise SystemExit(f"clean CSV not found: {clean_path}")

    selected_datasets = parse_dataset_selector(args.datasets)
    df = pd.read_csv(clean_path)
    df = df[df["dataset"].astype(int).isin(selected_datasets)].copy()
    if df.empty:
        raise SystemExit(f"no rows selected by --datasets={args.datasets!r}")

    df = clean_features(df)
    feature_cols = select_feature_columns(df)
    if not feature_cols:
        raise SystemExit("no feature columns remain after v3 exclusions")

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=int)
    groups = df["dataset"].to_numpy(dtype=int)

    cv_df, oof_proba = leave_dataset_out_cv(
        X=X,
        y=y,
        groups=groups,
        params=xgb_params(args),
    )
    cv_df.to_csv(output_dir / "cv_results_v3_smoke.csv", index=False)

    final_model = xgb.XGBClassifier(**xgb_params(args))
    final_model.fit(X, y, verbose=False)
    final_model.save_model(output_dir / "xgb_final_v3_smoke.json")

    write_feature_outputs(final_model, feature_cols, output_dir)

    noisy_df = pd.DataFrame()
    if not args.skip_noisy:
        noisy_df = evaluate_noisy_mirrors(
            model=final_model,
            feature_cols=feature_cols,
            noisy_csv=Path(args.noisy_csv),
            clean_datasets=selected_datasets,
        )
        noisy_df.to_csv(output_dir / "noise_robustness_v3_smoke.csv", index=False)

    summary = build_summary(
        df=df,
        feature_cols=feature_cols,
        cv_df=cv_df,
        y=y,
        oof_proba=oof_proba,
        noisy_df=noisy_df,
        args=args,
    )
    pd.DataFrame([summary]).to_csv(output_dir / "summary_v3_smoke.csv", index=False)
    (output_dir / "summary_v3_smoke.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"RapidMiner outputs written to: {output_dir}")
    return 0


def xgb_params(args: argparse.Namespace) -> dict:
    return {
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": int(args.random_state),
        "n_jobs": -1,
    }


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    kurt_cols = [c for c in df.columns if "kurt" in c.lower()]
    if kurt_cols:
        df[kurt_cols] = df[kurt_cols].fillna(0.0)
    return df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [
        c for c in df.columns
        if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    usable = []
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().all():
            usable.append(col)
    return usable


def leave_dataset_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    params: dict,
) -> tuple[pd.DataFrame, np.ndarray]:
    oof_proba = np.zeros(len(y), dtype=float)
    rows: list[dict] = []
    for dataset in sorted(set(groups.tolist())):
        test_mask = groups == dataset
        train_mask = ~test_mask
        if len(np.unique(y[train_mask])) < 2:
            raise SystemExit(
                f"training fold for dataset {dataset} has only one class; "
                "select more datasets for the smoke test"
            )

        model = xgb.XGBClassifier(**params)
        model.fit(X[train_mask], y[train_mask], verbose=False)
        proba = model.predict_proba(X[test_mask])[:, 1]
        oof_proba[test_mask] = proba
        rows.append(metric_row(dataset, y[test_mask], proba, n=int(test_mask.sum())))

    return pd.DataFrame(rows), oof_proba


def metric_row(dataset: int, y_true: np.ndarray, proba: np.ndarray, n: int) -> dict:
    pred = (proba >= 0.5).astype(int)
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, proba))
    else:
        auc = math.nan
    return {
        "dataset": int(dataset),
        "n": int(n),
        "auc": round_or_nan(auc),
        "f1": round_or_nan(f1_score(y_true, pred, zero_division=0)),
        "recall": round_or_nan(recall_score(y_true, pred, zero_division=0)),
        "f2": round_or_nan(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "pct_unstable": round_or_nan(float(np.mean(y_true)), ndigits=3),
    }


def evaluate_noisy_mirrors(
    model: xgb.XGBClassifier,
    feature_cols: list[str],
    noisy_csv: Path,
    clean_datasets: set[int],
) -> pd.DataFrame:
    if not noisy_csv.exists():
        return pd.DataFrame(columns=["dataset", "noise_tier", "n", "auc", "f1", "recall", "f2"])

    mirror_ids = {d + 60 for d in clean_datasets} | {d + 90 for d in clean_datasets}
    df = pd.read_csv(noisy_csv)
    df = clean_features(df[df["dataset"].astype(int).isin(mirror_ids)].copy())
    if df.empty:
        return pd.DataFrame(columns=["dataset", "noise_tier", "n", "auc", "f1", "recall", "f2"])

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"noisy CSV missing model feature columns: {missing[:10]}")

    rows = []
    for dataset in sorted(df["dataset"].astype(int).unique().tolist()):
        mask = df["dataset"].astype(int) == dataset
        Xn = df.loc[mask, feature_cols].to_numpy(dtype=np.float64)
        yn = df.loc[mask, "label"].to_numpy(dtype=int)
        proba = model.predict_proba(Xn)[:, 1]
        row = metric_row(dataset, yn, proba, n=int(mask.sum()))
        row["noise_tier"] = "1pct" if 61 <= dataset <= 90 else "5pct"
        rows.append(row)
    return pd.DataFrame(rows)


def write_feature_outputs(
    model: xgb.XGBClassifier,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    pd.DataFrame({"feature": feature_cols}).to_csv(
        output_dir / "feature_columns_v3_smoke.csv",
        index=False,
    )
    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_.astype(float),
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(output_dir / "top_features_v3_smoke.csv", index=False)


def build_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    cv_df: pd.DataFrame,
    y: np.ndarray,
    oof_proba: np.ndarray,
    noisy_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    pred = (oof_proba >= 0.5).astype(int)
    summary = {
        "status": "ok",
        "pipeline": "xgboost_v3_smoke",
        "clean_csv": str(Path(args.clean_csv)),
        "noisy_csv": str(Path(args.noisy_csv)),
        "datasets": args.datasets,
        "n_rows": int(len(df)),
        "n_datasets": int(df["dataset"].nunique()),
        "n_features": int(len(feature_cols)),
        "oof_auc": round_or_nan(roc_auc_score(y, oof_proba)),
        "oof_f1": round_or_nan(f1_score(y, pred, zero_division=0)),
        "oof_recall": round_or_nan(recall_score(y, pred, zero_division=0)),
        "oof_f2": round_or_nan(fbeta_score(y, pred, beta=2, zero_division=0)),
        "mean_fold_auc": round_or_nan(float(pd.to_numeric(cv_df["auc"], errors="coerce").mean())),
        "mean_fold_f2": round_or_nan(float(pd.to_numeric(cv_df["f2"], errors="coerce").mean())),
        "noisy_rows": int(len(noisy_df)),
    }
    if not noisy_df.empty:
        summary["mean_noisy_auc"] = round_or_nan(
            float(pd.to_numeric(noisy_df["auc"], errors="coerce").mean())
        )
        summary["mean_noisy_f2"] = round_or_nan(
            float(pd.to_numeric(noisy_df["f2"], errors="coerce").mean())
        )
    else:
        summary["mean_noisy_auc"] = math.nan
        summary["mean_noisy_f2"] = math.nan
    return summary


def parse_dataset_selector(selector: str) -> set[int]:
    result: set[int] = set()
    for part in selector.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(token))
    return result


def round_or_nan(value: float, ndigits: int = 4) -> float:
    value = float(value)
    if math.isfinite(value):
        return round(value, ndigits)
    return math.nan


if __name__ == "__main__":
    raise SystemExit(main())
