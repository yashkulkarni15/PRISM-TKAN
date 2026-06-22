"""Unit tests for ``pri2m.phase3.weights``.

These cover the Phase 3 sample-weight derivation from the heavy-tailed
``noise_score`` column and the kurtosis-NaN imputation step. The weight design
goal is: clean files (noise_score near 0) keep ~full weight, noisy files are
down-weighted, files with missing noise_score get zero weight, and the heavy
upper tail (max ~190) is winsorized so no single outlier dominates training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pri2m.phase3.weights import (
    KURT_COLS,
    impute_kurtosis,
    noise_sample_weights,
    winsorize,
)


# ---------------------------------------------------------------------------
# winsorize
# ---------------------------------------------------------------------------


def test_winsorize_clips_upper_tail_down_to_quantile():
    # A clean-ish bulk plus one extreme value (190). With a realistic-sized
    # sample the q0.99 boundary lands near the bulk -- well below 190 -- and the
    # outlier is clipped DOWN to it.
    s = pd.Series([0.0, 1.0, 2.0, 3.0] * 25 + [190.0])
    out = winsorize(s, upper_q=0.99)

    q = s.quantile(0.99)
    # The q0.99 value is well below 190 (clipping genuinely tames the tail).
    assert q < 10.0
    assert out.max() == pytest.approx(q)
    assert out.iloc[-1] == pytest.approx(q)
    assert out.iloc[-1] < 190.0
    # Small values below the quantile are unchanged.
    assert out.iloc[0] == 0.0


def test_winsorize_leaves_small_values_unchanged_and_preserves_nan():
    s = pd.Series([0.0, 1.0, np.nan, 3.0, 190.0])
    out = winsorize(s, upper_q=0.99)

    # NaN stays NaN.
    assert np.isnan(out.iloc[2])
    # Values at or below the quantile are untouched.
    assert out.iloc[0] == 0.0
    assert out.iloc[1] == 1.0
    assert out.iloc[3] == 3.0


def test_winsorize_quantile_uses_finite_values_only():
    # If NaNs were treated as numbers the quantile would shift; here the result
    # must equal the quantile of the finite values only.
    finite = pd.Series([0.0, 1.0, 2.0, 3.0, 190.0])
    with_nans = pd.Series([0.0, 1.0, 2.0, 3.0, 190.0, np.nan, np.nan])

    q_finite = finite.quantile(0.99)
    out = winsorize(with_nans, upper_q=0.99)
    assert out.max() == pytest.approx(q_finite)


def test_winsorize_returns_named_series_aligned_to_input_index():
    s = pd.Series([0.0, 1.0, 190.0], index=[10, 20, 30])
    out = winsorize(s, upper_q=0.99)
    assert list(out.index) == [10, 20, 30]


# ---------------------------------------------------------------------------
# noise_sample_weights
# ---------------------------------------------------------------------------


def test_nan_noise_score_gets_zero_weight():
    df = pd.DataFrame({"noise_score": [0.0, 1.0, np.nan, 2.0]})
    w = noise_sample_weights(df, normalize=False)
    assert w.iloc[2] == 0.0


def test_weight_series_is_named_sample_weight():
    df = pd.DataFrame({"noise_score": [0.0, 1.0, 2.0]})
    w = noise_sample_weights(df)
    assert w.name == "sample_weight"


def test_smaller_noise_score_yields_larger_weight():
    df = pd.DataFrame({"noise_score": [0.0, 0.5, 1.0, 2.0]})
    w = noise_sample_weights(df, normalize=False)
    # Monotonic decreasing in noise_score.
    assert w.iloc[0] > w.iloc[1] > w.iloc[2] > w.iloc[3]


def test_clean_file_unnormalized_weight_is_one():
    df = pd.DataFrame({"noise_score": [0.0, 1.0, 3.0]})
    w = noise_sample_weights(df, normalize=False)
    # raw = 1/(1+0) = 1.0 for a perfectly clean file.
    assert w.iloc[0] == pytest.approx(1.0)


def test_normalized_positive_weights_have_mean_one():
    df = pd.DataFrame({"noise_score": [0.0, 0.5, 1.0, np.nan, 2.0]})
    w = noise_sample_weights(df, normalize=True)
    positive = w[w > 0]
    assert positive.mean() == pytest.approx(1.0)
    # The NaN row stays at zero even after normalization.
    assert w.iloc[3] == 0.0


def test_heavy_tail_outlier_weight_is_bounded_and_not_dominating():
    # Realistic heavy-tailed noise_score: a clean-ish bulk plus one extreme
    # outlier (190), mirroring the dataset where p99 sits far below the max.
    # Winsorization at q0.99 must clip the outlier down so its weight is on the
    # same scale as the rest, not vanishingly small.
    bulk = [0.0, 0.5, 1.0, 1.5, 2.0] * 40  # 200 rows, p99 well below 190
    df = pd.DataFrame({"noise_score": bulk + [190.0]})

    cap = pd.Series(df["noise_score"]).quantile(0.99)
    # Sanity: the q0.99 boundary is far below the extreme value.
    assert cap < 190.0

    w_raw = noise_sample_weights(df, normalize=False)

    # The outlier row is winsorized to ``cap`` -> weight 1/(1+cap), NOT the
    # naive 1/(1+190). It must be many times larger than the un-winsorized value.
    naive_outlier_weight = 1.0 / (1.0 + 190.0)
    expected_capped_weight = 1.0 / (1.0 + cap)
    assert w_raw.iloc[-1] == pytest.approx(expected_capped_weight)
    assert w_raw.iloc[-1] > naive_outlier_weight * 5

    # All raw weights live in (0, 1]; none explodes and none collapses to ~0.
    assert (w_raw > 0).all()
    assert (w_raw <= 1.0 + 1e-9).all()
    # The outlier weight is the smallest (largest score) but still the same
    # order of magnitude as the bulk -- it does not dominate or vanish.
    assert w_raw.iloc[-1] == pytest.approx(w_raw.min())
    assert w_raw.iloc[-1] > 0.1 * w_raw.max()

    # Normalized: positive-weight mean is exactly 1.0.
    w = noise_sample_weights(df, normalize=True)
    positive = w[w > 0]
    assert positive.mean() == pytest.approx(1.0)


def test_weights_align_to_input_index():
    df = pd.DataFrame(
        {"noise_score": [0.0, np.nan, 2.0]}, index=["a", "b", "c"]
    )
    w = noise_sample_weights(df)
    assert list(w.index) == ["a", "b", "c"]
    assert w.loc["b"] == 0.0


# ---------------------------------------------------------------------------
# impute_kurtosis
# ---------------------------------------------------------------------------


def test_impute_kurtosis_fills_kurt_nan_and_leaves_other_cols_untouched():
    df = pd.DataFrame(
        {
            "Forcex_kurt": [1.0, np.nan, 3.0],
            "xtpos_kurt": [np.nan, 2.0, 3.0],
            "other": [10.0, np.nan, 30.0],
        }
    )
    out = impute_kurtosis(df)

    # Kurt NaNs filled with 0.0.
    assert out["Forcex_kurt"].tolist() == [1.0, 0.0, 3.0]
    assert out["xtpos_kurt"].tolist() == [0.0, 2.0, 3.0]
    # Non-kurt column is untouched (its NaN remains).
    assert np.isnan(out["other"].iloc[1])
    assert out["other"].iloc[0] == 10.0


def test_impute_kurtosis_flags_rows_with_any_kurt_nan():
    df = pd.DataFrame(
        {
            "Forcex_kurt": [1.0, np.nan, 3.0],
            "xtpos_kurt": [np.nan, 2.0, 3.0],
            "other": [10.0, np.nan, 30.0],
        }
    )
    out = impute_kurtosis(df)
    # Row 0 had a NaN in xtpos_kurt; row 1 had a NaN in Forcex_kurt; row 2 clean.
    assert out["kurt_imputed"].tolist() == [True, True, False]


def test_impute_kurtosis_returns_copy_and_does_not_mutate_input():
    df = pd.DataFrame({"Forcex_kurt": [1.0, np.nan, 3.0]})
    original = df.copy()
    out = impute_kurtosis(df)
    # Input frame is unchanged.
    pd.testing.assert_frame_equal(df, original)
    # Output is a distinct object.
    assert out is not df
    assert np.isnan(df["Forcex_kurt"].iloc[1])


def test_impute_kurtosis_handles_missing_kurt_columns_gracefully():
    # Only a subset of KURT_COLS present; absent ones must not raise.
    df = pd.DataFrame({"Forcex_kurt": [np.nan, 2.0], "keep": [1.0, 2.0]})
    out = impute_kurtosis(df)
    assert out["Forcex_kurt"].tolist() == [0.0, 2.0]
    assert out["kurt_imputed"].tolist() == [True, False]
    assert "keep" in out.columns


def test_impute_kurtosis_no_kurt_columns_flags_all_false():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [3.0, 4.0]})
    out = impute_kurtosis(df)
    # No present KURT_COLS -> nothing imputed -> all False.
    assert out["kurt_imputed"].tolist() == [False, False]
    # The non-kurt NaN is preserved.
    assert np.isnan(out["a"].iloc[1])


def test_impute_kurtosis_custom_fill_value():
    df = pd.DataFrame({"ytvel_kurt": [np.nan, 5.0]})
    out = impute_kurtosis(df, fill=-1.0)
    assert out["ytvel_kurt"].tolist() == [-1.0, 5.0]


def test_kurt_cols_constant():
    assert KURT_COLS == (
        "xtpos_kurt",
        "ytpos_kurt",
        "Forcex_kurt",
        "Forcey_kurt",
        "xtvel_kurt",
        "ytvel_kurt",
    )
