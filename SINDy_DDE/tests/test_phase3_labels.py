"""Tests for ``pri2m.phase3.labels``.

These tests build small synthetic DataFrames that mix the three target-integrity
cases discovered in the pre-Phase-3 audit and verify that the label-hygiene
functions flag and exclude rows correctly:

  - negative / non-finite ``blim`` at the lowest spindle speeds (the ~700
    untrustworthy "unstable" labels),
  - rows whose noise metrics could not be computed
    (``mode_label == 'metrics_unavailable'``),
  - ordinary, physically valid rows.

The functions under test are pure: they operate on DataFrames and never touch
the filesystem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pri2m.phase3.labels import (
    flag_label_validity,
    summarize_exclusions,
    supervised_mask,
)


def _negative_blim_row() -> dict:
    """A low-speed row with negative blim and a stored-but-untrustworthy label."""
    return {
        "omega": 1200.0,
        "b_mm": 4.0,
        "blim": -3.0,
        "b_over_blim": np.nan,
        "label": 1,
        "mode_label": "mixed_tpf_chatter",
    }


def _metrics_unavailable_row() -> dict:
    """A physically valid row whose noise metrics could not be computed."""
    return {
        "omega": 8000.0,
        "b_mm": 3.0,
        "blim": 2.0,
        "b_over_blim": 1.5,
        "label": 1,
        "mode_label": "metrics_unavailable",
    }


def _normal_row() -> dict:
    """An ordinary, physically valid, supervised-usable stable row."""
    return {
        "omega": 9000.0,
        "b_mm": 0.5,
        "blim": 1.0,
        "b_over_blim": 0.5,
        "label": 0,
        "mode_label": "tpf_dominated_clean",
    }


# ---------------------------------------------------------------------------
# flag_label_validity
# ---------------------------------------------------------------------------


def test_flag_negative_blim_row():
    df = pd.DataFrame([_negative_blim_row()])
    out = flag_label_validity(df)
    r = out.iloc[0]
    assert bool(r["blim_valid"]) is False
    assert bool(r["b_over_blim_valid"]) is False
    assert bool(r["physics_usable"]) is False
    assert bool(r["exclude_supervised"]) is True
    assert r["exclude_reason"] == "invalid_blim_low_speed"


def test_flag_metrics_unavailable_row():
    df = pd.DataFrame([_metrics_unavailable_row()])
    out = flag_label_validity(df)
    r = out.iloc[0]
    # blim itself is fine (2.0 > 0, finite, b_over_blim finite) -> physics usable.
    assert bool(r["blim_valid"]) is True
    assert bool(r["b_over_blim_valid"]) is True
    assert bool(r["physics_usable"]) is True
    # But the noise metrics are missing, so it is excluded from supervised target.
    assert bool(r["exclude_supervised"]) is True
    assert r["exclude_reason"] == "metrics_unavailable"


def test_flag_normal_row():
    df = pd.DataFrame([_normal_row()])
    out = flag_label_validity(df)
    r = out.iloc[0]
    assert bool(r["blim_valid"]) is True
    assert bool(r["b_over_blim_valid"]) is True
    assert bool(r["physics_usable"]) is True
    assert bool(r["exclude_supervised"]) is False
    assert r["exclude_reason"] == ""


def test_flag_returns_copy_does_not_mutate_input():
    df = pd.DataFrame([_normal_row()])
    original_cols = list(df.columns)
    out = flag_label_validity(df)
    # Input frame untouched.
    assert list(df.columns) == original_cols
    # Output has the new flag columns.
    for col in (
        "blim_valid",
        "b_over_blim_valid",
        "physics_usable",
        "exclude_supervised",
        "exclude_reason",
    ):
        assert col in out.columns


def test_flag_zero_blim_with_finite_ratio():
    # blim == 0 is not strictly positive, so it is NOT physics-usable and the
    # reason is invalid_blim_low_speed. But the spec keys supervised exclusion
    # on b_over_blim finiteness; with a finite b_over_blim the label is still
    # derivable, so exclude_supervised stays False. (The real negative-blim rows
    # always carry NaN b_over_blim and so ARE supervised-excluded.)
    row = {
        "omega": 1000.0,
        "b_mm": 0.0,
        "blim": 0.0,
        "b_over_blim": 0.0,
        "label": 0,
        "mode_label": "mixed_tpf_chatter",
    }
    out = flag_label_validity(pd.DataFrame([row]))
    r = out.iloc[0]
    assert bool(r["blim_valid"]) is False
    assert bool(r["b_over_blim_valid"]) is True
    assert bool(r["physics_usable"]) is False
    assert bool(r["exclude_supervised"]) is False
    assert r["exclude_reason"] == "invalid_blim_low_speed"


def test_flag_no_mode_label_column_does_not_crash():
    # No mode_label column at all: metrics_unavailable contribution is all-False.
    row = {"omega": 9000.0, "b_mm": 0.5, "blim": 1.0, "b_over_blim": 0.5, "label": 0}
    out = flag_label_validity(pd.DataFrame([row]))
    r = out.iloc[0]
    assert bool(r["exclude_supervised"]) is False
    assert r["exclude_reason"] == ""


def test_flag_invalid_blim_priority_over_metrics_unavailable():
    # A row that is BOTH negative-blim AND metrics_unavailable should report the
    # higher-priority invalid_blim_low_speed reason.
    row = {
        "omega": 1100.0,
        "b_mm": 2.0,
        "blim": -5.0,
        "b_over_blim": np.nan,
        "label": 1,
        "mode_label": "metrics_unavailable",
    }
    out = flag_label_validity(pd.DataFrame([row]))
    r = out.iloc[0]
    assert r["exclude_reason"] == "invalid_blim_low_speed"
    assert bool(r["exclude_supervised"]) is True


# ---------------------------------------------------------------------------
# supervised_mask
# ---------------------------------------------------------------------------


def test_supervised_mask_excludes_bad_keeps_normal():
    df = pd.DataFrame(
        [_negative_blim_row(), _metrics_unavailable_row(), _normal_row()]
    )
    mask = supervised_mask(df)
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool
    assert list(mask) == [False, False, True]


def test_supervised_mask_runs_on_raw_frame_without_flag_columns():
    # supervised_mask must internally call flag_label_validity if flags absent.
    df = pd.DataFrame([_normal_row()])
    assert "exclude_supervised" not in df.columns
    mask = supervised_mask(df)
    assert list(mask) == [True]
    # Calling on a raw frame must not have added columns to the caller's frame.
    assert "exclude_supervised" not in df.columns


def test_supervised_mask_uses_existing_flag_columns_when_present():
    df = flag_label_validity(pd.DataFrame([_normal_row(), _negative_blim_row()]))
    mask = supervised_mask(df)
    assert list(mask) == [True, False]


# ---------------------------------------------------------------------------
# summarize_exclusions
# ---------------------------------------------------------------------------


def test_summarize_exclusions_counts_on_mixed_frame():
    df = pd.DataFrame(
        [
            _negative_blim_row(),
            _metrics_unavailable_row(),
            _normal_row(),
            _normal_row(),
        ]
    )
    summary = summarize_exclusions(df)

    assert summary["n_total"] == 4
    assert summary["n_negative_blim"] == 1
    assert summary["n_nan_b_over_blim"] == 1
    assert summary["n_metrics_unavailable"] == 1
    # Negative-blim row + metrics_unavailable row are excluded; two normals stay.
    assert summary["n_excluded"] == 2
    assert summary["n_supervised_usable"] == 2
    # Physics-usable: both normals + the metrics_unavailable row (its blim is fine).
    assert summary["n_physics_usable"] == 3
    # All values are plain Python ints.
    for value in summary.values():
        assert isinstance(value, int)


def test_summarize_exclusions_no_mode_label_column_does_not_crash():
    df = pd.DataFrame(
        [
            {"omega": 9000.0, "b_mm": 0.5, "blim": 1.0, "b_over_blim": 0.5, "label": 0},
            {"omega": 1200.0, "b_mm": 4.0, "blim": -3.0, "b_over_blim": np.nan, "label": 1},
        ]
    )
    summary = summarize_exclusions(df)
    assert summary["n_total"] == 2
    assert summary["n_negative_blim"] == 1
    assert summary["n_nan_b_over_blim"] == 1
    assert summary["n_metrics_unavailable"] == 0
    assert summary["n_excluded"] == 1
    assert summary["n_supervised_usable"] == 1
    assert summary["n_physics_usable"] == 1


def test_summarize_exclusions_empty_frame():
    df = pd.DataFrame(
        {
            "blim": pd.Series([], dtype=float),
            "b_over_blim": pd.Series([], dtype=float),
            "mode_label": pd.Series([], dtype=object),
        }
    )
    summary = summarize_exclusions(df)
    assert summary["n_total"] == 0
    assert summary["n_supervised_usable"] == 0
    assert summary["n_physics_usable"] == 0
    for value in summary.values():
        assert isinstance(value, int)
