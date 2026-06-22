#!/usr/bin/env python3
"""Phase 2.5 mode-label assignment.

Reads the per-file noise metric CSV produced by ``compute_noise_metrics.py`` and
emits a ``mode_labels.csv`` with one row per file, each tagged with one of the
six Phase 2.5 mode labels and a confidence in ``[0, 1]``. The decision rule is
encoded in ``pri2m.noise.modes.assign_mode``.

The ``fs/tpf`` check is tracked as a sampling-quality flag only. It does not
set the physical ``low_speed_edge`` mode, which is reserved for rows outside a
trusted spindle-speed / boundary-grid range.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pri2m.noise.modes import (  # noqa: E402
    MODE_LABELS,
    ModeThresholds,
    assign_mode,
)


LOW_SPEED_SAMPLES_PER_PERIOD = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-metrics-csv", required=True)
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional clean summary CSV; only used to backfill b_over_blim if missing.",
    )
    parser.add_argument(
        "--noisy-summary-csv",
        default=None,
        help="Optional noisy summary CSV; only used to backfill b_over_blim if missing.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--samples-per-period-min",
        type=int,
        default=LOW_SPEED_SAMPLES_PER_PERIOD,
        help="Below this many samples per tooth-pass period the row gets an undersampled_tpf quality flag.",
    )
    parser.add_argument(
        "--calibration-json",
        default=None,
        help="Optional promoted_metrics.json from calibrate_noise_mirror_pairs.py. "
             "If omitted, <output-dir>/promoted_metrics.json is used when present.",
    )
    parser.add_argument(
        "--omega-low",
        type=float,
        default=None,
        help="Spindle speed (RPM) lower bound for the trusted operating grid. "
             "Rows with omega < omega_low are assigned mode 'low_speed_edge'. "
             "When omitted, the low_speed_edge mode is inactive at the CLI; the "
             "label remains reserved for future stability-grid-based gating.",
    )
    parser.add_argument(
        "--omega-high",
        type=float,
        default=None,
        help="Spindle speed (RPM) upper bound for the trusted operating grid. "
             "Rows with omega > omega_high are assigned mode 'low_speed_edge'. "
             "Same activation semantics as --omega-low.",
    )
    return parser.parse_args(argv)


def _is_outside_trusted_speed_grid(
    omega: float | None,
    omega_low: float | None,
    omega_high: float | None,
) -> bool:
    """Return True when the row falls outside the configured spindle-speed window.

    When neither bound is provided, the low_speed_edge gate is inactive and this
    function returns False so the label is never assigned. The full stability-
    boundary grid check (intersecting omega and b/blim with the Altintas-Budak
    lobe family) is deliberately out of scope here and tracked separately; the
    omega window is the v1 trusted-grid proxy.
    """
    if omega_low is None and omega_high is None:
        return False
    if omega is None or not math.isfinite(omega):
        return False
    if omega_low is not None and omega < omega_low:
        return True
    if omega_high is not None and omega > omega_high:
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    nm = pd.read_csv(args.noise_metrics_csv)
    thresholds = ModeThresholds()
    calibration, calibration_path = load_calibration(args.calibration_json, args.output_dir)

    out_rows: list[dict[str, Any]] = []
    for record in nm.to_dict(orient="records"):
        fs = _safe_float(record.get("fs_used_hz"))
        tpf = _safe_float(record.get("tpf"))
        samples_per_period = (fs / tpf) if (fs and tpf and tpf > 0) else None
        undersampled_tpf = (
            tpf is None
            or tpf <= 0
            or samples_per_period is None
            or samples_per_period < args.samples_per_period_min
        )
        sampling_quality_flag = "undersampled_tpf" if undersampled_tpf else "ok"

        b_over_blim = _safe_float(record.get("b_over_blim"))
        omega = _safe_float(record.get("omega"))
        is_low_speed_edge = _is_outside_trusted_speed_grid(
            omega, args.omega_low, args.omega_high
        )
        label, confidence, votes = assign_mode(
            record,
            b_over_blim=b_over_blim,
            is_low_speed_edge=is_low_speed_edge,
            thresholds=thresholds,
            calibration=calibration,
        )

        out_rows.append({
            "dataset": record.get("dataset"),
            "file": record.get("file"),
            "source": record.get("source"),
            "label": record.get("label"),
            "omega": record.get("omega"),
            "b_mm": record.get("b_mm"),
            "b_over_blim": b_over_blim,
            "tpf": tpf,
            "fn": _safe_float(record.get("fn")),
            "noise_std": _safe_float(record.get("noise_std")),
            "samples_per_period": samples_per_period,
            "sampling_quality_flag": sampling_quality_flag,
            "is_low_speed_edge": bool(is_low_speed_edge),
            "mode_label": label,
            "mode_confidence": confidence,
            "noise_score": _safe_float(votes.get("noise_score")),
            "calibrated_noise_metrics_used": _safe_float(votes.get("calibrated_noise_metrics_used")),
            "noise_family_votes": votes.get("noise_family_votes", ""),
            "abstaining_families": votes.get("abstaining_families", ""),
            "chatter_signal": votes.get("chatter_signal", ""),
            "near_boundary": votes.get("near_boundary", ""),
            "tpf_dominant": votes.get("tpf_dominant", ""),
            "reason": votes.get("reason", ""),
        })

    out_df = pd.DataFrame(out_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "mode_labels.csv"
    out_df.to_csv(out_csv, index=False)

    summary = {
        "thresholds": {
            "psd_spectral_entropy_high": thresholds.psd_spectral_entropy_high,
            "psd_off_harmonic_ratio_high": thresholds.psd_off_harmonic_ratio_high,
            "wavelet_mad_sigma_high_quantile_proxy": thresholds.wavelet_mad_sigma_high_quantile_proxy,
            "cross_coh_mean_low": thresholds.cross_coh_mean_low,
            "chatter_fn_band_energy_high": thresholds.chatter_fn_band_energy_high,
            "chatter_sideband_energy_high": thresholds.chatter_sideband_energy_high,
            "chatter_fn_band_kurtosis_high": thresholds.chatter_fn_band_kurtosis_high,
            "chatter_cyclostationarity_tpf_low": thresholds.chatter_cyclostationarity_tpf_low,
            "psd_off_harmonic_ratio_clean_max": thresholds.psd_off_harmonic_ratio_clean_max,
            "psd_spectral_entropy_clean_max": thresholds.psd_spectral_entropy_clean_max,
            "b_over_blim_near_boundary_low": thresholds.b_over_blim_near_boundary_low,
            "b_over_blim_near_boundary_high": thresholds.b_over_blim_near_boundary_high,
            "samples_per_period_min": args.samples_per_period_min,
            "omega_low": args.omega_low,
            "omega_high": args.omega_high,
            "low_speed_edge_active": (
                args.omega_low is not None or args.omega_high is not None
            ),
        },
        "label_counts": out_df["mode_label"].value_counts().to_dict(),
        "sampling_quality_counts": out_df["sampling_quality_flag"].value_counts().to_dict(),
        "n_rows": int(len(out_df)),
        "mode_labels": list(MODE_LABELS),
        "calibration": {
            "path": str(calibration_path) if calibration_path else None,
            "loaded": calibration is not None,
            "n_promoted": int(len((calibration or {}).get("promoted_metrics", []))) if calibration else 0,
        },
        "paths": {
            "noise_metrics_csv": str(args.noise_metrics_csv),
            "mode_labels_csv": str(out_csv),
        },
    }
    (out_dir / "mode_labels_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def load_calibration(
    explicit_path: str | None,
    output_dir: str,
) -> tuple[dict[str, Any] | None, Path | None]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.append(Path(output_dir) / "promoted_metrics.json")

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Could not parse calibration JSON {candidate}: {exc}") from exc
        if not payload.get("promoted_metrics"):
            print(f"warning: calibration JSON {candidate} has no promoted metrics; using legacy thresholds", file=sys.stderr)
            return None, candidate
        return payload, candidate
    print("warning: no calibration JSON found; using legacy uncalibrated thresholds", file=sys.stderr)
    return None, None


if __name__ == "__main__":
    raise SystemExit(main())
