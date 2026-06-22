"""Leakage-safe train/validation splits for the milling-chatter dataset.

Why this module exists
----------------------
Datasets are organised in three noise tiers that share identical machining cuts:

* ``1..60``   -- the *clean* datasets.
* ``61..90``  -- 1%-noise mirrors; dataset ``60 + D`` is the 1% mirror of clean ``D``.
* ``91..120`` -- 5%-noise mirrors; dataset ``90 + D`` is the 5% mirror of clean ``D``.

A clean dataset ``D`` and its two mirrors (``D+60`` and ``D+90``) contain the SAME
operating points (the same ``(omega, b_mm)`` pairs) with only added measurement
noise -- they overlap 100%. If a clean dataset lands in training while one of its
mirrors lands in validation, the model effectively sees the validation operating
points during training. That is *mirror leakage* and it inflates every reported
metric.

The cure is to collapse every row onto its *clean-parent* group id and split on
those groups rather than on raw rows or raw dataset ids. All functions here are
pure functions over DataFrames -- no file I/O, only numpy and pandas.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import overload

import numpy as np
import pandas as pd

__all__ = [
    "DEV_GROUPS",
    "parent_group",
    "add_parent_group",
    "grouped_folds",
    "dev_split",
    "assert_no_mirror_leakage",
]


# Curated development groups for fast iteration. These are *clean-parent* group
# ids (each id stands for clean dataset D together with its mirrors D+60, D+90).
#
# CORRECTION / leakage note:
#   The previous selection ``phase3_dev = [1, 4, 29, 35, 39, 91]`` was leaky.
#   Dataset 91 is the 5% mirror of clean dataset 1 -- the SAME machining cuts --
#   so ``parent_group(91) == 1``. Listing both 1 and 91 did not add an
#   independent group; it silently smuggled a mirror of group 1 into the dev set
#   and would have placed the same operating points on both sides of any
#   downstream split. Reduced to the five distinct clean-parent groups below.
DEV_GROUPS: tuple[int, int, int, int, int] = (1, 4, 29, 35, 39)


@overload
def parent_group(dataset: int) -> int: ...
@overload
def parent_group(dataset: pd.Series) -> pd.Series: ...
def parent_group(dataset: int | pd.Series) -> int | pd.Series:
    """Map a dataset id (or Series) to its leakage-safe clean-parent group id.

    The group id is the smallest dataset id among all datasets that share an
    identical documented parameter fingerprint (see ``data/metadata/parent_groups.json``,
    built by ``scripts/build_parent_groups.py``). This collapses every noise mirror
    onto its clean parent AND co-groups parameter-identical clean datasets (e.g.
    1==23, 16==17, and the 126..155 repeats of 1..30) so identical operating points
    can never straddle a train/val split. When the artifact is absent the legacy
    arithmetic rule is used (clean ``d``; 61..90 -> d-60; 91..120 -> d-90), correct
    for datasets 1..120 only.
    """
    if isinstance(dataset, pd.Series):
        out = _parent_group_array(dataset.to_numpy())
        return pd.Series(out, index=dataset.index, name=dataset.name)
    d = int(dataset)
    return _group_map().get(d, _arith_scalar(d))


def _arith_scalar(d: int) -> int:
    """Legacy arithmetic clean-parent rule (fallback for 1..120 when the map is absent)."""
    if 91 <= d <= 120:
        return d - 90
    if 61 <= d <= 90:
        return d - 60
    return d


_GROUP_MAP_PATH = Path(__file__).resolve().parents[3] / "data/metadata/parent_groups.json"


@functools.lru_cache(maxsize=1)
def _group_map() -> dict[int, int]:
    """Frozen ``{dataset_id: group_id}`` map from ``parent_groups.json`` (cached)."""
    try:
        raw = json.loads(_GROUP_MAP_PATH.read_text())
    except FileNotFoundError:
        return {}
    return {int(k): int(v) for k, v in raw.items()}


def _parent_group_array(d: np.ndarray) -> np.ndarray:
    """Vectorized core of :func:`parent_group` for integer arrays."""
    d = np.asarray(d)
    gm = _group_map()
    if not gm:  # artifact absent -> legacy arithmetic behavior (1..120 only)
        out = d.astype(np.int64, copy=True)
        out = np.where((d >= 61) & (d <= 90), d - 60, out)
        out = np.where((d >= 91) & (d <= 120), d - 90, out)
        return out.astype(np.int64, copy=False)
    flat = [gm.get(int(x), _arith_scalar(int(x))) for x in d.ravel()]
    return np.asarray(flat, dtype=np.int64).reshape(d.shape)


def add_parent_group(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with an integer ``parent_group`` column.

    The column is derived from ``df['dataset']`` via :func:`parent_group`. The
    input frame is not mutated.
    """
    out = df.copy()
    out["parent_group"] = _parent_group_array(df["dataset"].to_numpy()).astype(np.int64)
    return out


def _parent_groups_of(df: pd.DataFrame) -> np.ndarray:
    """Positional array of parent-group ids for every row of ``df``."""
    if "parent_group" in df.columns:
        return df["parent_group"].to_numpy()
    return _parent_group_array(df["dataset"].to_numpy())


def grouped_folds(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build group-disjoint cross-validation folds (positional indices).

    The unique parent-group ids are sorted, shuffled deterministically with
    ``np.random.default_rng(seed)``, then partitioned into ``n_splits``
    contiguous, near-equal chunks of *groups* (never rows). For each fold the
    validation set is every row whose ``parent_group`` falls in that chunk, and
    the training set is all remaining rows.

    Returns a list of ``(train_idx, val_idx)`` tuples of numpy arrays holding
    POSITIONAL row indices into ``df`` (0..len(df)-1).

    Invariants:
      * a ``parent_group`` never appears in both train and val of one fold;
      * every group is in exactly one fold's validation set;
      * the validation sets partition all rows.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")

    pg = _parent_groups_of(df)
    n_rows = len(df)
    groups = np.unique(pg)  # sorted unique parent groups

    if n_splits > len(groups):
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of parent groups "
            f"({len(groups)}); cannot place a group in every fold"
        )

    rng = np.random.default_rng(seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)

    # Contiguous, near-equal chunks of GROUPS (numpy spreads the remainder over
    # the leading folds, so sizes differ by at most one).
    group_chunks = np.array_split(shuffled, n_splits)

    positions = np.arange(n_rows)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for chunk in group_chunks:
        val_mask = np.isin(pg, chunk)
        val_idx = positions[val_mask]
        train_idx = positions[~val_mask]
        folds.append((train_idx, val_idx))
    return folds


def dev_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split ``df`` into the curated dev set and the rest (positional indices).

    Dev rows are those whose clean-parent group is in :data:`DEV_GROUPS`. Because
    the split is by clean-parent group, a dev group automatically pulls in its
    1% and 5% mirrors (e.g. group 1 grabs datasets 1, 61 and 91), which is
    exactly what keeps mirror operating points off the other side.

    Returns ``(dev_idx, rest_idx)`` as numpy arrays of positional row indices.
    """
    pg = _parent_groups_of(df)
    positions = np.arange(len(df))
    dev_mask = np.isin(pg, np.asarray(DEV_GROUPS))
    dev_idx = positions[dev_mask]
    rest_idx = positions[~dev_mask]
    return dev_idx, rest_idx


def assert_no_mirror_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Raise ``AssertionError`` if train and val share a clean-parent group.

    The leakage that matters is the **mirror** relationship: a clean cut and its
    noise-added copies (D, D+60, D+90) are the same machining event, so they must
    never straddle a train/val boundary. ``parent_group`` captures exactly this,
    and grouping by it is both necessary and sufficient.

    Note: this deliberately does NOT flag shared ``(omega, b_mm)`` operating
    points across *different* parent groups. In this dataset 64% of operating
    points are reused across distinct datasets (e.g. datasets 1 and 29 share all
    of dataset 1's points) because different experiments sweep the same
    spindle-speed/depth grid under different tool/workpiece dynamics. Those are
    independent cuts, not leakage; asserting on raw operating points would block
    almost every valid grouped split.
    """
    train_pg = set(_parent_groups_of(train_df).tolist())
    val_pg = set(_parent_groups_of(val_df).tolist())
    shared_groups = train_pg & val_pg
    assert not shared_groups, (
        f"mirror leakage: parent_group(s) {sorted(shared_groups)} appear in both "
        f"train and validation (a clean dataset and a noisy mirror share the same "
        f"clean parent and thus the same cuts)"
    )
