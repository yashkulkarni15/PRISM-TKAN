"""Tests for leakage-safe Phase 3 splits (clean-parent grouping).

Datasets 1..60 are clean; 61..90 are 1%-noise mirrors (D+60 mirrors clean D);
91..120 are 5%-noise mirrors (D+90 mirrors clean D). A clean dataset and its two
mirrors share 100% of their (omega, b_mm) operating points, so every split must
keep (D, D+60, D+90) on the same side. These tests pin that invariant down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pri2m.phase3.splits import (
    DEV_GROUPS,
    add_parent_group,
    assert_no_mirror_leakage,
    dev_split,
    grouped_folds,
    parent_group,
)


# ---------------------------------------------------------------------------
# parent_group
# ---------------------------------------------------------------------------


def test_parent_group_scalar_mirror_collapse() -> None:
    # A clean dataset and both of its mirrors map to the same clean parent.
    assert parent_group(1) == 1
    assert parent_group(61) == 1
    assert parent_group(91) == 1


def test_parent_group_scalar_boundaries() -> None:
    assert parent_group(60) == 60   # last clean
    assert parent_group(90) == 30   # last 1% mirror -> clean 30
    assert parent_group(120) == 30  # last 5% mirror -> clean 30
    assert parent_group(61) == 1    # first 1% mirror -> clean 1
    assert parent_group(91) == 1    # first 5% mirror -> clean 1


def test_parent_group_scalar_returns_int() -> None:
    result = parent_group(91)
    assert isinstance(result, (int, np.integer))


def test_parent_group_series_vectorized() -> None:
    s = pd.Series([1, 61, 91, 60, 90, 120, 5, 65, 95])
    out = parent_group(s)
    assert isinstance(out, pd.Series)
    expected = pd.Series([1, 1, 1, 60, 30, 30, 5, 5, 5])
    pd.testing.assert_series_equal(
        out.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_parent_group_series_preserves_index() -> None:
    s = pd.Series([91, 65], index=[10, 20])
    out = parent_group(s)
    assert list(out.index) == [10, 20]
    assert list(out.values) == [1, 5]


# ---------------------------------------------------------------------------
# add_parent_group
# ---------------------------------------------------------------------------


def test_add_parent_group_adds_correct_column() -> None:
    df = pd.DataFrame({"dataset": [1, 61, 91, 60, 90, 120]})
    out = add_parent_group(df)
    assert "parent_group" in out.columns
    assert list(out["parent_group"]) == [1, 1, 1, 60, 30, 30]
    assert pd.api.types.is_integer_dtype(out["parent_group"])


def test_add_parent_group_returns_copy() -> None:
    df = pd.DataFrame({"dataset": [1, 61]})
    out = add_parent_group(df)
    assert out is not df
    assert "parent_group" not in df.columns  # original untouched


# ---------------------------------------------------------------------------
# grouped_folds
# ---------------------------------------------------------------------------


def _synthetic_frame(n_groups: int = 20, rows_per_dataset: int = 3) -> pd.DataFrame:
    """Build clean datasets 1..n_groups, each with rows_per_dataset rows, plus
    their 1% and 5% mirrors. Every clean parent thus appears as three datasets."""
    rows = []
    for d in range(1, n_groups + 1):
        for tier in (0, 60, 90):  # clean, 1% mirror, 5% mirror
            ds = d + tier
            for k in range(rows_per_dataset):
                rows.append({"dataset": ds, "omega": 1000.0 + d, "b_mm": 0.1 * k})
    return pd.DataFrame(rows)


def test_grouped_folds_train_val_disjoint_by_group() -> None:
    df = add_parent_group(_synthetic_frame())
    folds = grouped_folds(df, n_splits=5, seed=0)
    assert len(folds) == 5
    pg = df["parent_group"].to_numpy()
    for train_idx, val_idx in folds:
        train_groups = set(pg[train_idx])
        val_groups = set(pg[val_idx])
        assert train_groups.isdisjoint(val_groups), (
            "a parent_group appeared in both train and val of the same fold"
        )


def test_grouped_folds_each_group_in_exactly_one_val() -> None:
    df = add_parent_group(_synthetic_frame())
    folds = grouped_folds(df, n_splits=5, seed=0)
    pg = df["parent_group"].to_numpy()
    all_groups = set(pg)
    seen: dict[int, int] = {}
    for fold_id, (_, val_idx) in enumerate(folds):
        for g in set(pg[val_idx]):
            assert g not in seen, f"group {g} in val of folds {seen[g]} and {fold_id}"
            seen[g] = fold_id
    assert set(seen) == all_groups, "every group must land in exactly one val fold"


def test_grouped_folds_val_partitions_all_rows() -> None:
    df = add_parent_group(_synthetic_frame())
    folds = grouped_folds(df, n_splits=5, seed=0)
    n = len(df)
    collected = np.concatenate([val_idx for _, val_idx in folds])
    # Every row appears in exactly one fold's validation set.
    assert np.array_equal(np.sort(collected), np.arange(n))


def test_grouped_folds_train_is_complement_of_val() -> None:
    df = add_parent_group(_synthetic_frame())
    folds = grouped_folds(df, n_splits=5, seed=0)
    n = len(df)
    for train_idx, val_idx in folds:
        assert np.array_equal(
            np.sort(np.concatenate([train_idx, val_idx])), np.arange(n)
        )
        assert len(np.intersect1d(train_idx, val_idx)) == 0


def test_grouped_folds_indices_are_positional_numpy_int() -> None:
    # Use a non-default (shuffled, non-range) index to prove positions, not labels.
    df = add_parent_group(_synthetic_frame())
    df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    df.index = df.index * 10 + 3  # weird, non-positional labels
    folds = grouped_folds(df, n_splits=5, seed=0)
    n = len(df)
    for train_idx, val_idx in folds:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(val_idx, np.ndarray)
        assert train_idx.dtype.kind == "i"
        assert val_idx.dtype.kind == "i"
        assert val_idx.max() < n and train_idx.max() < n  # positional, in [0, n)


def test_grouped_folds_deterministic_same_seed() -> None:
    df = add_parent_group(_synthetic_frame())
    a = grouped_folds(df, n_splits=5, seed=0)
    b = grouped_folds(df, n_splits=5, seed=0)
    assert len(a) == len(b)
    for (ta, va), (tb, vb) in zip(a, b):
        assert np.array_equal(ta, tb)
        assert np.array_equal(va, vb)


def test_grouped_folds_different_seed_changes_assignment() -> None:
    df = add_parent_group(_synthetic_frame())
    a = grouped_folds(df, n_splits=5, seed=0)
    b = grouped_folds(df, n_splits=5, seed=999)
    # Validation group membership should differ for at least one fold.
    pg = df["parent_group"].to_numpy()
    a_sets = [frozenset(pg[v]) for _, v in a]
    b_sets = [frozenset(pg[v]) for _, v in b]
    assert a_sets != b_sets


# ---------------------------------------------------------------------------
# Mirror group co-location: clean 5 with mirrors 65 and 95
# ---------------------------------------------------------------------------


def _frame_with_group5_and_mirrors() -> pd.DataFrame:
    rows = []
    # Clean 5 + mirrors 65, 95 (all parent_group 5), plus other groups for context.
    for ds in (5, 65, 95):
        for k in range(3):
            rows.append({"dataset": ds, "omega": 1500.0, "b_mm": 0.2 * k})
    for d in (1, 2, 3, 4, 6, 7, 8, 9, 10):
        for tier in (0, 60, 90):
            for k in range(2):
                rows.append({"dataset": d + tier, "omega": 1000.0 + d, "b_mm": 0.1 * k})
    return add_parent_group(pd.DataFrame(rows))


def test_group5_and_mirrors_share_parent_group() -> None:
    df = _frame_with_group5_and_mirrors()
    mask = df["dataset"].isin([5, 65, 95])
    assert set(df.loc[mask, "parent_group"]) == {5}


def test_group5_and_mirrors_always_same_fold_and_side() -> None:
    df = _frame_with_group5_and_mirrors()
    pg = df["parent_group"].to_numpy()
    group5_pos = np.where(pg == 5)[0]
    for seed in (0, 1, 2, 42):
        for train_idx, val_idx in grouped_folds(df, n_splits=5, seed=seed):
            in_val = np.isin(group5_pos, val_idx)
            in_train = np.isin(group5_pos, train_idx)
            # All of group 5's rows are on exactly one side together.
            assert in_val.all() or in_train.all(), (
                f"group 5 (datasets 5/65/95) was split across train and val "
                f"at seed {seed}"
            )


def test_group5_and_mirrors_together_in_dev_split() -> None:
    # 4 is a DEV group, 5 is not -> group 5 (and mirrors) must be in 'rest'.
    df = _frame_with_group5_and_mirrors()
    pg = df["parent_group"].to_numpy()
    dev_idx, rest_idx = dev_split(df)
    group5_pos = np.where(pg == 5)[0]
    assert np.isin(group5_pos, rest_idx).all()
    assert not np.isin(group5_pos, dev_idx).any()


# ---------------------------------------------------------------------------
# dev_split / DEV_GROUPS
# ---------------------------------------------------------------------------


def test_dev_groups_value_is_clean_parents_not_leaky() -> None:
    # The corrected dev set is clean-parent groups (no 91, which maps to parent 1).
    assert DEV_GROUPS == (1, 4, 29, 35, 39)
    assert 91 not in DEV_GROUPS


def test_dev_split_selects_dev_group_rows_and_mirrors() -> None:
    # Include clean 1 and its 5% mirror 91 (same parent 1): both must be in dev.
    rows = []
    for ds in (1, 91, 4, 29, 35, 39, 2, 7, 50):
        for k in range(2):
            rows.append({"dataset": ds, "omega": 2000.0 + ds, "b_mm": 0.3 * k})
    df = add_parent_group(pd.DataFrame(rows))
    pg = df["parent_group"].to_numpy()
    dev_idx, rest_idx = dev_split(df)

    # Partition the rows with no overlap.
    n = len(df)
    assert np.array_equal(np.sort(np.concatenate([dev_idx, rest_idx])), np.arange(n))
    assert len(np.intersect1d(dev_idx, rest_idx)) == 0

    # dataset 1 and its mirror 91 (parent 1) are both in dev.
    dev_datasets = set(df.iloc[dev_idx]["dataset"])
    assert {1, 91, 4, 29, 35, 39}.issubset(dev_datasets)
    # Non-dev parent groups (2, 7, 50) are NOT in dev.
    assert set(df.iloc[dev_idx]["parent_group"]) == set(DEV_GROUPS)


def test_dev_split_returns_positional_numpy_int() -> None:
    rows = [{"dataset": d, "omega": 1.0, "b_mm": 0.0} for d in (1, 4, 2, 9)]
    df = add_parent_group(pd.DataFrame(rows))
    dev_idx, rest_idx = dev_split(df)
    assert isinstance(dev_idx, np.ndarray) and dev_idx.dtype.kind == "i"
    assert isinstance(rest_idx, np.ndarray) and rest_idx.dtype.kind == "i"


# ---------------------------------------------------------------------------
# assert_no_mirror_leakage
# ---------------------------------------------------------------------------


def _df_for(datasets: list[int], omega_b: list[tuple[float, float]] | None = None) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        if omega_b is None:
            rows.append({"dataset": ds, "omega": 1000.0 + ds, "b_mm": 0.5})
        else:
            for (om, b) in omega_b:
                rows.append({"dataset": ds, "omega": om, "b_mm": b})
    return pd.DataFrame(rows)


def test_leakage_raises_when_mirror_dataset_split_across_sides() -> None:
    # dataset 5 in train, its 1% mirror 65 in val: same parent group AND same
    # operating points -> must raise.
    train = _df_for([5], omega_b=[(1500.0, 0.2), (1500.0, 0.4)])
    val = _df_for([65], omega_b=[(1500.0, 0.2), (1500.0, 0.4)])
    with pytest.raises(AssertionError):
        assert_no_mirror_leakage(train, val)


def test_no_leakage_passes_when_whole_group_on_one_side() -> None:
    # Whole group 5 (5, 65, 95) in train; a different group 8 in val. No raise.
    train = _df_for([5, 65, 95], omega_b=[(1500.0, 0.2), (1500.0, 0.4)])
    val = _df_for([8, 68, 98], omega_b=[(1800.0, 0.6), (1800.0, 0.8)])
    assert_no_mirror_leakage(train, val)  # should not raise


def test_no_leakage_when_distinct_groups_share_operating_point() -> None:
    # Distinct parent groups sharing a (omega, b_mm) pair is NOT leakage: in this
    # dataset 64% of operating points are reused across different datasets (e.g.
    # datasets 1 and 29 share all of dataset 1's points) because different
    # experiments sweep the same speed/depth grid under different dynamics. Only
    # the mirror relationship (same parent_group) is leakage.
    train = _df_for([3], omega_b=[(1234.0, 0.7)])
    val = _df_for([8], omega_b=[(1234.0, 0.7)])  # parent 3 vs 8, same op point
    assert_no_mirror_leakage(train, val)  # must NOT raise


def test_assert_no_mirror_leakage_computes_parent_group_when_absent() -> None:
    # No parent_group column present; function must derive it and still catch leak.
    train = _df_for([5], omega_b=[(1500.0, 0.2)])
    val = _df_for([95], omega_b=[(9999.0, 0.9)])  # parent 5 on both, distinct op pts
    assert "parent_group" not in train.columns
    with pytest.raises(AssertionError):
        assert_no_mirror_leakage(train, val)


def test_assert_no_mirror_leakage_uses_existing_parent_group_column() -> None:
    train = add_parent_group(_df_for([8], omega_b=[(2000.0, 0.1)]))
    val = add_parent_group(_df_for([9], omega_b=[(3000.0, 0.2)]))
    assert "parent_group" in train.columns
    assert_no_mirror_leakage(train, val)  # disjoint groups + op points -> ok


def test_grouped_folds_output_obeys_no_mirror_leakage() -> None:
    # End-to-end: every fold produced by grouped_folds must pass the leakage check.
    df = add_parent_group(_synthetic_frame())
    folds = grouped_folds(df, n_splits=5, seed=3)
    for train_idx, val_idx in folds:
        assert_no_mirror_leakage(df.iloc[train_idx], df.iloc[val_idx])
