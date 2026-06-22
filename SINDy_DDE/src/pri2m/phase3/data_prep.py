"""Compose the Phase 3 training table from features + mode labels.

Joins the clean/noisy feature CSVs with the per-file mode labels, then applies
the three audit fixes built as separate modules:

  - :mod:`pri2m.phase3.splits`  -> ``parent_group`` + leakage-safe folds
  - :mod:`pri2m.phase3.labels`  -> negative-blim / metrics_unavailable exclusion
  - :mod:`pri2m.phase3.weights` -> winsorized noise sample weights + kurtosis impute

Output is one tidy table (one row per file) carrying ``parent_group``,
``sample_weight``, ``exclude_supervised``, ``physics_usable`` and a fold
assignment keyed on the clean-parent group so (D, D+60, D+90) never split.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .labels import flag_label_validity, summarize_exclusions, supervised_mask
from .splits import (
    DEV_GROUPS,
    add_parent_group,
    assert_no_mirror_leakage,
    grouped_folds,
    parent_group,
)
from .weights import KURT_COLS, impute_kurtosis, noise_sample_weights

_MODE_COLS = ["dataset", "file", "noise_score", "mode_label", "mode_confidence"]


def _read_features(clean_csv: str, noisy_csv: str | None) -> pd.DataFrame:
    frames = [pd.read_csv(clean_csv)]
    if noisy_csv:
        frames.append(pd.read_csv(noisy_csv))
    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = df["dataset"].astype(int)
    return df


def _read_mode_labels(mode_labels_csv: str) -> pd.DataFrame:
    ml = pd.read_csv(mode_labels_csv, low_memory=False)
    ml["dataset"] = ml["dataset"].astype(int)
    keep = [c for c in _MODE_COLS if c in ml.columns]
    return ml[keep].copy()


def build_phase3_table(
    features_clean_csv: str,
    features_noisy_csv: str | None,
    mode_labels_csv: str,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """Build the leakage-safe, hygiene-cleaned Phase 3 table.

    The returned frame is the input feature rows plus:
      ``parent_group``, ``noise_score``, ``mode_label``, the label-validity flags
      (``blim_valid``, ``b_over_blim_valid``, ``physics_usable``,
      ``exclude_supervised``, ``exclude_reason``), imputed kurtosis columns +
      ``kurt_imputed``, and a final ``sample_weight`` (0 for any row excluded
      from the supervised target, positive weights renormalized to mean 1).
    """
    feats = _read_features(features_clean_csv, features_noisy_csv)
    modes = _read_mode_labels(mode_labels_csv)

    table = feats.merge(modes, on=["dataset", "file"], how="left", suffixes=("", "_mode"))
    table = add_parent_group(table)
    table = flag_label_validity(table)
    table = impute_kurtosis(table)

    # Base weights: 1/(1+winsorized noise), 0 where noise_score is NaN.
    weight = noise_sample_weights(table, upper_q=upper_q, normalize=False)
    # Zero out anything excluded from the supervised target (e.g. negative-blim rows
    # that still carry a finite noise_score), then renormalize positive weights.
    weight = weight.where(~table["exclude_supervised"].to_numpy(), 0.0)
    positive = weight > 0
    if positive.any():
        weight = weight.where(~positive, weight / weight[positive].mean())
    table["sample_weight"] = weight.astype(float)
    return table


def fold_assignments(
    table: pd.DataFrame, n_splits: int = 5, seed: int = 0
) -> dict:
    """Assign each clean-parent group to a fold and verify no mirror leakage.

    Returns a JSON-serializable dict: per-group fold map, per-fold summary, and
    the held-out dev groups. Every fold is checked with
    :func:`assert_no_mirror_leakage`; the check raising would surface a bug.
    """
    folds = grouped_folds(table, n_splits=n_splits, seed=seed)
    pg = parent_group(table["dataset"]).to_numpy()
    group_to_fold: dict[int, int] = {}
    fold_summary = []
    for k, (train_idx, val_idx) in enumerate(folds):
        assert_no_mirror_leakage(table.iloc[train_idx], table.iloc[val_idx])
        val_groups = sorted({int(g) for g in pg[val_idx]})
        for g in val_groups:
            group_to_fold[g] = k
        usable = table["sample_weight"].to_numpy() > 0
        fold_summary.append(
            {
                "fold": k,
                "n_val_groups": len(val_groups),
                "n_train_rows": int(len(train_idx)),
                "n_val_rows": int(len(val_idx)),
                "n_train_usable": int(usable[train_idx].sum()),
                "n_val_usable": int(usable[val_idx].sum()),
            }
        )
    return {
        "n_splits": int(n_splits),
        "seed": int(seed),
        "dev_groups": list(DEV_GROUPS),
        "group_to_fold": {int(k): int(v) for k, v in sorted(group_to_fold.items())},
        "fold_summary": fold_summary,
        "leakage_verified": True,
    }


def summarize(table: pd.DataFrame) -> dict:
    """Counts and weight stats for a built Phase 3 table."""
    out = summarize_exclusions(table)
    w = table["sample_weight"].to_numpy()
    pos = w[w > 0]
    out.update(
        {
            "n_rows": int(len(table)),
            "n_weight_zero": int((w == 0).sum()),
            "n_weight_positive": int((w > 0).sum()),
            "weight_positive_mean": float(pos.mean()) if pos.size else 0.0,
            "weight_max": float(w.max()) if w.size else 0.0,
            "n_kurt_imputed": int(table.get("kurt_imputed", pd.Series(dtype=bool)).sum()),
        }
    )
    return out
