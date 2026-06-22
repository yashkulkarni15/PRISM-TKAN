"""Phase 3 sample weights and kurtosis imputation.

The milling-chatter dataset carries a per-file ``noise_score`` from the
Phase 2.5 calibration step. It is heavy-tailed (mean ~1.6, p99 ~15, p99.9 ~34,
max ~190; one scale-unstable metric drives the extreme tail), and ~560 rows
have ``NaN`` because their noise metrics could not be computed.

For Phase 3 the ``noise_score`` becomes a per-file SAMPLE WEIGHT: clean files
(``noise_score`` near 0) keep ~full weight, noisy files are down-weighted, and
no single outlier file is allowed to dominate training. We achieve that by

1. winsorizing the upper tail down to a quantile (default p99) so the extreme
   tail cannot blow the weight scale apart, then
2. mapping ``s`` to ``1 / (1 + s)`` so ``s = 0 -> 1`` and large ``s -> ~0``,
3. zeroing rows whose ``noise_score`` is ``NaN`` (metrics unavailable),
4. optionally rescaling the strictly-positive weights to mean 1.0.

Separately, six kurtosis feature columns contain ``NaN`` on the full dataset;
the established pipeline imputes them with ``0.0``. ``impute_kurtosis`` mirrors
that and records which rows were imputed.

Pure functions on DataFrames/Series: no file I/O, only numpy and pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Kurtosis feature columns that may carry ``NaN`` on the full dataset.
KURT_COLS = (
    "xtpos_kurt",
    "ytpos_kurt",
    "Forcex_kurt",
    "Forcey_kurt",
    "xtvel_kurt",
    "ytvel_kurt",
)


def winsorize(series: pd.Series, upper_q: float = 0.99) -> pd.Series:
    """Clip the upper tail of ``series`` down to its ``upper_q`` quantile.

    Finite values strictly above the ``upper_q`` quantile (computed over finite
    values only) are clipped down to that quantile. Values at or below it are
    unchanged, and ``NaN`` entries stay ``NaN``. The result keeps the input's
    index and name.

    Parameters
    ----------
    series:
        Numeric Series, possibly containing ``NaN``.
    upper_q:
        Upper quantile in ``[0, 1]`` used as the clip threshold.
    """
    s = pd.Series(series, copy=True)
    # ``Series.quantile`` ignores NaN, so this is the finite-only quantile
    # (the standard linear-interpolation winsorization threshold).
    cap = s.quantile(upper_q)
    if pd.isna(cap):
        # All values were NaN -> nothing to clip.
        return s
    # ``clip(upper=...)`` leaves NaN as NaN and only lowers values above ``cap``.
    return s.clip(upper=cap)


def noise_sample_weights(
    df: pd.DataFrame, upper_q: float = 0.99, normalize: bool = True
) -> pd.Series:
    """Derive per-row training weights from the heavy-tailed ``noise_score``.

    Steps:
      * winsorize ``df['noise_score']`` at ``upper_q`` (tail clipped down),
      * ``raw = 1 / (1 + s)`` so clean files (``s ~ 0``) get ~1 and noisy files
        get small weights,
      * force weight ``0.0`` wherever ``noise_score`` is ``NaN``
        (metrics unavailable),
      * if ``normalize``, rescale the strictly-positive weights so their mean is
        exactly 1.0 (zeros stay zero).

    Returns a Series named ``'sample_weight'`` aligned to ``df.index``.
    """
    noise = df["noise_score"]
    s = winsorize(noise, upper_q=upper_q)

    weight = 1.0 / (1.0 + s)
    # Rows with unavailable noise metrics carry no training signal here.
    weight = weight.where(noise.notna(), other=0.0)

    if normalize:
        positive = weight > 0
        mean_positive = weight[positive].mean()
        if pd.notna(mean_positive) and mean_positive > 0:
            weight = weight.where(~positive, weight / mean_positive)

    weight.name = "sample_weight"
    return weight


def impute_kurtosis(df: pd.DataFrame, fill: float = 0.0) -> pd.DataFrame:
    """Impute ``NaN`` in the present kurtosis columns and flag affected rows.

    Returns a COPY of ``df`` in which every column from :data:`KURT_COLS` that
    is present has its ``NaN`` replaced by ``fill``. A new boolean column
    ``kurt_imputed`` is ``True`` for any row where at least one present
    kurtosis column was ``NaN`` before filling. Columns outside
    :data:`KURT_COLS` are left untouched.
    """
    out = df.copy()
    present = [c for c in KURT_COLS if c in out.columns]

    if present:
        was_nan = out[present].isna().any(axis=1)
        out[present] = out[present].fillna(fill)
    else:
        # No kurtosis columns present -> nothing was imputed.
        was_nan = pd.Series(False, index=out.index)

    out["kurt_imputed"] = was_nan.astype(bool)
    return out
