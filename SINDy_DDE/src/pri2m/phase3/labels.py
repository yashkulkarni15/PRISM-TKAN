"""Phase 3 label hygiene for the milling-chatter supervised target.

The supervised target is ``label`` (1 = unstable cut, 0 = stable), defined
physically as ``label = int(b_over_blim >= 1.0)`` where
``b_over_blim = b_mm / blim`` and ``blim`` is the critical stability depth of
cut. The pre-Phase-3 audit found two target-integrity defects that must be kept
out of the supervised training target:

  1. ``invalid_blim_low_speed`` -- ~700 rows (~2.1%) at the lowest spindle
     speeds (omega 1000-1500, below the valid stability-boundary grid) carry a
     NEGATIVE ``blim`` (e.g. -16 to -2). A depth of cut cannot be negative, so
     ``b_over_blim`` is NaN there yet the stored ``label`` is 1. That label is
     not derivable from the definition and is untrustworthy.
  2. ``metrics_unavailable`` -- ~560 rows whose noise metrics could not be
     computed (``mode_label == 'metrics_unavailable'``). Their physics may be
     fine but they cannot participate in the noise-aware supervised target.

The later physics-residual model head additionally requires a finite,
strictly-positive ``blim`` and a finite ``b_over_blim`` (``physics_usable``).

This module is pure: every function takes a DataFrame and returns a new object;
nothing is read from or written to disk, and the caller's frame is never
mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Sentinel used by the noise-mode stage for rows whose metrics failed.
METRICS_UNAVAILABLE_LABEL = "metrics_unavailable"

# Exclusion-reason strings, in priority order (highest first).
REASON_INVALID_BLIM = "invalid_blim_low_speed"
REASON_METRICS_UNAVAILABLE = "metrics_unavailable"

# Columns added by :func:`flag_label_validity`.
_FLAG_COLUMNS = (
    "blim_valid",
    "b_over_blim_valid",
    "physics_usable",
    "exclude_supervised",
    "exclude_reason",
)


def _metrics_unavailable_series(df: pd.DataFrame) -> pd.Series:
    """Boolean Series: True where ``mode_label == 'metrics_unavailable'``.

    If the frame has no ``mode_label`` column, every entry is False (the
    metrics_unavailable contribution is treated as absent, never as an error).
    """
    if "mode_label" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["mode_label"].eq(METRICS_UNAVAILABLE_LABEL)


def flag_label_validity(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with target-integrity flag columns added.

    Added columns:

    ``blim_valid`` (bool)
        ``np.isfinite(blim) & (blim > 0)`` -- finite and strictly positive.
    ``b_over_blim_valid`` (bool)
        ``np.isfinite(b_over_blim)``.
    ``physics_usable`` (bool)
        ``blim_valid & b_over_blim_valid`` -- usable by the physics-residual head.
    ``exclude_supervised`` (bool)
        ``(~b_over_blim_valid) | (mode_label == 'metrics_unavailable')``. When
        the frame has no ``mode_label`` column the metrics_unavailable part is
        all-False.
    ``exclude_reason`` (str)
        Per-row reason, in priority order: ``'invalid_blim_low_speed'`` if
        ``b_over_blim`` is non-finite OR ``blim <= 0``; else
        ``'metrics_unavailable'`` if ``mode_label == 'metrics_unavailable'``;
        else ``''``.

    The input frame is never mutated.
    """
    out = df.copy()

    blim = pd.to_numeric(out["blim"], errors="coerce")
    b_over_blim = pd.to_numeric(out["b_over_blim"], errors="coerce")

    blim_finite = np.isfinite(blim)
    blim_valid = blim_finite & (blim > 0.0)
    b_over_blim_valid = np.isfinite(b_over_blim)
    physics_usable = blim_valid & b_over_blim_valid

    metrics_unavailable = _metrics_unavailable_series(out)

    # A non-positive (or non-finite) blim is the low-speed integrity defect; a
    # non-finite b_over_blim is its visible symptom. Either triggers the
    # invalid-blim reason and supervised exclusion.
    invalid_blim = (~b_over_blim_valid) | (~blim_valid)

    exclude_supervised = (~b_over_blim_valid) | metrics_unavailable

    # Priority order: invalid_blim wins over metrics_unavailable.
    exclude_reason = np.where(
        invalid_blim,
        REASON_INVALID_BLIM,
        np.where(metrics_unavailable, REASON_METRICS_UNAVAILABLE, ""),
    )

    out["blim_valid"] = blim_valid.to_numpy(dtype=bool)
    out["b_over_blim_valid"] = b_over_blim_valid.to_numpy(dtype=bool)
    out["physics_usable"] = physics_usable.to_numpy(dtype=bool)
    out["exclude_supervised"] = exclude_supervised.to_numpy(dtype=bool)
    out["exclude_reason"] = pd.Series(exclude_reason, index=out.index, dtype=object)

    return out


def supervised_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series, True for rows usable as supervised training examples.

    Usable means ``~exclude_supervised``: a finite ``b_over_blim`` (so the label
    is derivable from its physical definition) and noise metrics that were
    actually computed. If the flag columns produced by
    :func:`flag_label_validity` are absent, they are computed on a local copy so
    the caller's frame is left untouched.
    """
    if "exclude_supervised" in df.columns:
        flags = df
    else:
        flags = flag_label_validity(df)
    return (~flags["exclude_supervised"].astype(bool)).rename("supervised_usable")


def summarize_exclusions(df: pd.DataFrame) -> dict[str, int]:
    """Return integer counts describing the label-hygiene outcome.

    Keys:

    ``n_total``
        Number of rows.
    ``n_negative_blim``
        Rows whose ``blim`` is not finite-and-positive (the low-speed defect).
    ``n_nan_b_over_blim``
        Rows whose ``b_over_blim`` is non-finite.
    ``n_metrics_unavailable``
        Rows with ``mode_label == 'metrics_unavailable'`` (0 if no such column).
    ``n_excluded``
        Rows excluded from the supervised target.
    ``n_supervised_usable``
        Rows kept for supervised training.
    ``n_physics_usable``
        Rows with finite, strictly-positive ``blim`` and finite ``b_over_blim``.
    """
    flagged = flag_label_validity(df)

    blim = pd.to_numeric(flagged["blim"], errors="coerce")
    n_negative_blim = int((~(np.isfinite(blim) & (blim > 0.0))).sum())
    n_nan_b_over_blim = int((~flagged["b_over_blim_valid"]).sum())
    n_metrics_unavailable = int(_metrics_unavailable_series(flagged).sum())
    n_excluded = int(flagged["exclude_supervised"].sum())

    return {
        "n_total": int(len(flagged)),
        "n_negative_blim": n_negative_blim,
        "n_nan_b_over_blim": n_nan_b_over_blim,
        "n_metrics_unavailable": n_metrics_unavailable,
        "n_excluded": n_excluded,
        "n_supervised_usable": int(len(flagged)) - n_excluded,
        "n_physics_usable": int(flagged["physics_usable"].sum()),
    }
