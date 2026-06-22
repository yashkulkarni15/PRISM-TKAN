#!/usr/bin/env python3
"""Phase 2.5 mirror-ladder calibration gate.

Given the per-file noise metrics produced by ``compute_noise_metrics.py``, this
script tests every metric column on the clean / 1% noise / 5% noise mirror
ladder. A metric is promoted only if:

1. Strictly increasing across the three noise levels on at least 95 percent of
   matched ``(omega, b_mm, mirror_file_key)`` triples.
2. Mann-Whitney U comparing the 5%-noise distribution to the clean distribution
   has one-sided ``p < 1e-3`` for ``5% > clean``.

Outputs:

- ``mirror_calibration_<clean>_vs_<n1>_vs_<n5>.csv`` — per-metric statistics.
- ``promoted_metrics.json`` — the list of metric columns that passed the gate.

Pairing assumption (verified on the metadata): noisy_id 61..90 mirrors
clean_id = noisy_id - 60 at 1 percent noise; noisy_id 91..120 mirrors
clean_id = noisy_id - 90 at 5 percent noise. So Dataset 61 and Dataset 91 both
mirror Dataset 1.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]


IDENTITY_COLUMNS = {
    "dataset",
    "file",
    "source",
    "label",
    "omega",
    "b_mm",
    "blim",
    "b_over_blim",
    "tpf",
    "fn",
    "teeth",
    "cut_dir",
    "noise_std",
    "fs_used_hz",
    "n_samples",
    "npz_path",
}

METRIC_FAMILY_PREFIXES = (
    "psd_",
    "wavelet_",
    "autocorr_",
    "chatter_",
    "cross_coh_",
)

NOISE_SCORE_FAMILIES = {"psd", "wavelet_temporal", "cross_channel"}

# Minimum fraction of paired rows that must be finite for a metric to be
# eligible for promotion.  Metrics below this threshold may be artefactually
# monotone on the sparse finite subset; the gate prevents them entering
# production decisions.  Overridable via --min-coverage.
MIN_COVERAGE = 0.8


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
        help="Optional noisy summary CSV; only used to backfill noise_std if missing.",
    )
    parser.add_argument("--clean-id", type=int, default=1)
    parser.add_argument("--noise1-id", type=int, default=61)
    parser.add_argument("--noise5-id", type=int, default=91)
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Pooled multi-ladder calibration mode. Ignores --clean-id/--noise1-id/--noise5-id. "
             "Auto-detects every clean dataset id D in the input where both D+60 and D+90 exist, "
             "concatenates the paired-triple values across all such ladders, and runs the same "
             "promotion gate on the pooled distribution. Use this for production calibration; "
             "the single-triple mode is for pilot smoke tests only.",
    )
    parser.add_argument(
        "--noise1-offset",
        type=int,
        default=60,
        help="Offset from clean_id to the 1%% noise mirror (pooled mode).",
    )
    parser.add_argument(
        "--noise5-offset",
        type=int,
        default=90,
        help="Offset from clean_id to the 5%% noise mirror (pooled mode).",
    )
    parser.add_argument(
        "--exclude-clean-ids",
        type=str,
        default="",
        help="Comma-separated list of clean dataset ids to exclude from pooled "
             "calibration (e.g. '19' or '19,2,16'). Default empty (no exclusions). "
             "Applied only in --pooled mode: any auto-discovered ladder whose "
             "clean_id is in this set is dropped before pairing. Ignored in "
             "single-triple mode (a warning is emitted to stderr if combined with "
             "single-triple args).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--monotone-fraction",
        type=float,
        default=0.95,
        help="Minimum fraction of paired files that must be strictly monotone "
             "across the three noise levels for the metric to be promoted.",
    )
    parser.add_argument(
        "--p-value-threshold",
        type=float,
        default=1.0e-3,
        help="Mann-Whitney U threshold for promoting a metric.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=MIN_COVERAGE,
        help=(
            "Minimum fraction of paired rows that must be finite (not NaN/inf) "
            "for a metric to be eligible for promotion. Default %(default)s. "
            "Metrics below this threshold are blocked even if they are strictly "
            "monotone and statistically significant on the finite subset."
        ),
    )
    return parser.parse_args(argv)


def _pair_triple_for_ladder(
    nm: pd.DataFrame,
    clean_id: int,
    noise1_id: int,
    noise5_id: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, list[Any]]:
    """Return (clean, n1, n5) DataFrames indexed on pair_keys, plus the common index.

    Returns (None, None, None, []) when any of the three datasets is empty after
    keying.
    """
    pair_keys = ["omega", "b_mm", "__mirror_file_key"]
    frames = {}
    for label, dataset_id in (("clean", clean_id), ("noise1", noise1_id), ("noise5", noise5_id)):
        sub = nm[nm["dataset"] == dataset_id].copy()
        sub["__mirror_file_key"] = sub.apply(_mirror_file_key, axis=1)
        sub.dropna(subset=pair_keys, inplace=True)
        frames[label] = sub
    if any(df.empty for df in frames.values()):
        return None, None, None, []
    c = frames["clean"].set_index(pair_keys).sort_index()
    a = frames["noise1"].set_index(pair_keys).sort_index()
    b = frames["noise5"].set_index(pair_keys).sort_index()
    common = c.index.intersection(a.index).intersection(b.index).tolist()
    return c, a, b, common


def _discover_ladders(
    nm: pd.DataFrame,
    noise1_offset: int,
    noise5_offset: int,
) -> list[tuple[int, int, int]]:
    """Return all (clean_id, noise1_id, noise5_id) triples present in the data."""
    available = set(nm["dataset"].astype(int).unique())
    triples: list[tuple[int, int, int]] = []
    for clean_id in sorted(available):
        n1 = clean_id + noise1_offset
        n5 = clean_id + noise5_offset
        if n1 in available and n5 in available and n1 != clean_id and n5 != clean_id:
            triples.append((clean_id, n1, n5))
    return triples


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    nm = pd.read_csv(args.noise_metrics_csv)

    excluded_clean_ids: list[int] = []
    if args.exclude_clean_ids.strip():
        try:
            excluded_clean_ids = sorted(
                {int(tok) for tok in args.exclude_clean_ids.split(",") if tok.strip()}
            )
        except ValueError as exc:
            print(
                f"error: --exclude-clean-ids must be a comma-separated list of "
                f"integers, got {args.exclude_clean_ids!r}: {exc}",
                file=sys.stderr,
            )
            return 2

    if args.pooled:
        ladders = _discover_ladders(nm, args.noise1_offset, args.noise5_offset)
        if excluded_clean_ids:
            excluded_set = set(excluded_clean_ids)
            ladders = [t for t in ladders if t[0] not in excluded_set]
        if not ladders:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "promoted_metrics.json").write_text(json.dumps({
                "mode": "pooled",
                "excluded_clean_ids": excluded_clean_ids,
                "ladders": [],
                "promoted_metrics": [],
                "reason": "no_ladders_detected",
                "noise1_offset": args.noise1_offset,
                "noise5_offset": args.noise5_offset,
                "available_datasets": sorted(set(nm["dataset"].astype(int).unique())),
            }, indent=2))
            print(json.dumps({"status": "no_ladders_detected"}, indent=2))
            return 1
    else:
        if excluded_clean_ids:
            print(
                "warning: --exclude-clean-ids is ignored in single-triple mode; "
                f"got {excluded_clean_ids}",
                file=sys.stderr,
            )
        needed_ids = {args.clean_id, args.noise1_id, args.noise5_id}
        available = set(nm["dataset"].astype(int).unique())
        missing = needed_ids - available
        if missing:
            print(
                f"warning: noise_metrics.csv is missing required ladder datasets: {sorted(missing)}; "
                f"results will be skipped or partial",
                file=sys.stderr,
            )
        ladders = [(args.clean_id, args.noise1_id, args.noise5_id)]

    # Per-ladder pairing + early duplicate guard. The duplicate check is run
    # against each ladder's three datasets independently; a duplicate within one
    # ladder still kills the run so the symptom surfaces immediately.
    per_ladder_indices: list[dict[str, Any]] = []
    pooled_clean_frames: list[pd.DataFrame] = []
    pooled_n1_frames: list[pd.DataFrame] = []
    pooled_n5_frames: list[pd.DataFrame] = []
    for (clean_id, n1_id, n5_id) in ladders:
        clean_sub = nm[nm["dataset"] == clean_id].copy()
        n1_sub = nm[nm["dataset"] == n1_id].copy()
        n5_sub = nm[nm["dataset"] == n5_id].copy()
        for sub in (clean_sub, n1_sub, n5_sub):
            sub["__mirror_file_key"] = sub.apply(_mirror_file_key, axis=1)
            sub.dropna(subset=["omega", "b_mm", "__mirror_file_key"], inplace=True)
        dup = _duplicate_payload(
            {"clean": clean_sub, "noise1": n1_sub, "noise5": n5_sub},
            ["omega", "b_mm", "__mirror_file_key"],
        )
        if dup["duplicate_key_rows"]:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "promoted_metrics.json").write_text(json.dumps({
                "mode": "pooled" if args.pooled else "single_triple",
                "excluded_clean_ids": excluded_clean_ids,
                "clean_id": clean_id, "noise1_id": n1_id, "noise5_id": n5_id,
                "promoted_metrics": [],
                "reason": "duplicate_pair_keys",
                "duplicate_key_rows": dup["duplicate_key_rows"],
                "duplicate_preview": dup["duplicate_preview"],
            }, indent=2))
            print(json.dumps({"status": "duplicate_pair_keys",
                              "ladder": [clean_id, n1_id, n5_id],
                              "duplicate_key_rows": dup["duplicate_key_rows"],
                              "duplicate_preview": dup["duplicate_preview"]}, indent=2))
            return 2

        c, a, b, common = _pair_triple_for_ladder(nm, clean_id, n1_id, n5_id)
        if c is None or not common:
            per_ladder_indices.append({
                "clean_id": int(clean_id), "noise1_id": int(n1_id), "noise5_id": int(n5_id),
                "n_paired": 0,
            })
            continue
        per_ladder_indices.append({
            "clean_id": int(clean_id), "noise1_id": int(n1_id), "noise5_id": int(n5_id),
            "n_paired": int(len(common)),
        })
        # Stash the rows we'll use for pooled stats. Keep them indexed so the
        # eventual per-metric extraction is a simple .loc lookup.
        pooled_clean_frames.append(c.loc[common])
        pooled_n1_frames.append(a.loc[common])
        pooled_n5_frames.append(b.loc[common])

    n_paired = sum(int(entry["n_paired"]) for entry in per_ladder_indices)
    if n_paired == 0:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "promoted_metrics.json").write_text(json.dumps({
            "mode": "pooled" if args.pooled else "single_triple",
            "excluded_clean_ids": excluded_clean_ids,
            "ladders": per_ladder_indices,
            "promoted_metrics": [], "reason": "missing_data",
        }, indent=2))
        print(json.dumps({"status": "missing_data",
                          "ladders": per_ladder_indices}, indent=2))
        return 1

    c = pd.concat(pooled_clean_frames)
    a = pd.concat(pooled_n1_frames)
    b = pd.concat(pooled_n5_frames)
    # In pooled mode the index may repeat across ladders (different clean_ids
    # can share omega/b_mm/mirror_key); align by positional concat instead.
    common = list(range(len(c)))
    c = c.reset_index(drop=True)
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)

    metric_columns = [
        col for col in nm.columns
        if col not in IDENTITY_COLUMNS and any(col.startswith(p) for p in METRIC_FAMILY_PREFIXES)
    ]

    rows: list[dict[str, Any]] = []
    for col in metric_columns:
        v1 = pd.to_numeric(c.loc[common, col], errors="coerce").to_numpy()
        v2 = pd.to_numeric(a.loc[common, col], errors="coerce").to_numpy()
        v3 = pd.to_numeric(b.loc[common, col], errors="coerce").to_numpy()
        mask = np.isfinite(v1) & np.isfinite(v2) & np.isfinite(v3)
        v1, v2, v3 = v1[mask], v2[mask], v3[mask]
        n_finite = int(mask.sum())

        if n_finite == 0:
            rows.append({
                "metric": col, "n_paired": n_paired, "n_finite": 0,
                "mean_delta_5pct_minus_clean": math.nan,
                "sd_paired_delta": math.nan,
                "paired_effect_size": math.nan,
                "n_strict_monotone_up_ladder": 0,
                "n_strict_monotone_down_ladder": 0,
                "mwu_p_value_5pct_vs_clean": math.nan,
                "monotone_fraction": math.nan,
                "metric_family": metric_family(col),
                "promoted": False,
            })
            continue

        delta = v3 - v1
        sd = float(np.std(delta, ddof=1)) if delta.size >= 2 else math.nan
        mean_delta = float(np.mean(delta))
        effect = mean_delta / sd if sd and math.isfinite(sd) and sd > 0 else math.nan
        n_up = int(np.sum((v1 < v2) & (v2 < v3)))
        n_dn = int(np.sum((v1 > v2) & (v2 > v3)))
        monotone_fraction = n_up / n_finite if n_finite else math.nan
        try:
            _, p_mwu = stats.mannwhitneyu(v3, v1, alternative="greater")
            p_mwu = float(p_mwu)
        except ValueError:
            p_mwu = math.nan
        try:
            _, p_mwu_down = stats.mannwhitneyu(v3, v1, alternative="less")
            p_mwu_down = float(p_mwu_down)
        except ValueError:
            p_mwu_down = math.nan
        family = metric_family(col)
        coverage_fraction = float(n_finite) / float(n_paired) if n_paired else 0.0
        coverage_ok = coverage_fraction >= args.min_coverage
        promoted = bool(
            n_finite > 0
            and coverage_ok
            and mean_delta > 0.0
            and monotone_fraction >= args.monotone_fraction
            and math.isfinite(p_mwu)
            and p_mwu < args.p_value_threshold
            and family in NOISE_SCORE_FAMILIES
        )
        monotone_fraction_down = n_dn / n_finite if n_finite else math.nan
        inverse_indicator = bool(
            n_finite > 0
            and mean_delta < 0.0
            and math.isfinite(monotone_fraction_down)
            and monotone_fraction_down >= args.monotone_fraction
            and math.isfinite(p_mwu_down)
            and p_mwu_down < args.p_value_threshold
            and family in NOISE_SCORE_FAMILIES
        )
        clean_stats = _distribution_stats(v1)
        noise1_stats = _distribution_stats(v2)
        noise5_stats = _distribution_stats(v3)
        rows.append({
            "metric": col, "n_paired": n_paired, "n_finite": n_finite,
            "coverage_fraction": coverage_fraction,
            "coverage_ok": coverage_ok,
            "metric_family": family,
            "mean_delta_5pct_minus_clean": mean_delta,
            "sd_paired_delta": sd,
            "paired_effect_size": effect,
            "n_strict_monotone_up_ladder": n_up,
            "n_strict_monotone_down_ladder": n_dn,
            "monotone_fraction": monotone_fraction,
            "monotone_fraction_down": monotone_fraction_down,
            "mwu_p_value_5pct_vs_clean": p_mwu,
            "mwu_p_value_5pct_below_clean": p_mwu_down,
            "clean_median": clean_stats["median"],
            "clean_mean": clean_stats["mean"],
            "clean_std": clean_stats["std"],
            "clean_p95": clean_stats["p95"],
            "noise1_mean": noise1_stats["mean"],
            "noise5_mean": noise5_stats["mean"],
            "promoted": promoted,
            "inverse_indicator": inverse_indicator,
        })

    out_df = pd.DataFrame(rows).sort_values(
        "paired_effect_size", key=lambda s: s.abs(), ascending=False, na_position="last"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.pooled:
        out_csv = out_dir / "mirror_calibration_pooled.csv"
    else:
        out_csv = out_dir / (
            f"mirror_calibration_{args.clean_id}_vs_{args.noise1_id}_vs_{args.noise5_id}.csv"
        )
    out_df.to_csv(out_csv, index=False)

    promoted_df = out_df.loc[out_df["promoted"]].copy()
    promoted = promoted_df["metric"].tolist()
    inverse_df = out_df.loc[out_df["inverse_indicator"]].copy()
    inverse_metrics = [
        {
            "metric": str(row["metric"]),
            "family": str(row["metric_family"]),
            "mean_delta_5pct_minus_clean": _json_float(row["mean_delta_5pct_minus_clean"]),
            "monotone_fraction_down": _json_float(row.get("monotone_fraction_down")),
            "mwu_p_value_5pct_below_clean": _json_float(row.get("mwu_p_value_5pct_below_clean")),
        }
        for row in inverse_df.to_dict(orient="records")
    ]
    metric_stats = {
        str(row["metric"]): {
            "family": str(row["metric_family"]),
            "direction": "higher_is_noisier",
            "clean_median": _json_float(row["clean_median"]),
            "clean_mean": _json_float(row["clean_mean"]),
            "clean_std": _json_float(row["clean_std"]),
            "clean_p95": _json_float(row["clean_p95"]),
            "noise1_mean": _json_float(row["noise1_mean"]),
            "noise5_mean": _json_float(row["noise5_mean"]),
            "mean_delta_5pct_minus_clean": _json_float(row["mean_delta_5pct_minus_clean"]),
            "monotone_fraction": _json_float(row["monotone_fraction"]),
            "mwu_p_value_5pct_vs_clean": _json_float(row["mwu_p_value_5pct_vs_clean"]),
            "n_finite": int(row["n_finite"]),
            "coverage_fraction": float(row["coverage_fraction"]),
            "coverage_ok": bool(row["coverage_ok"]),
        }
        for row in promoted_df.to_dict(orient="records")
    }
    promoted_payload = {
        "mode": "pooled" if args.pooled else "single_triple",
        "excluded_clean_ids": excluded_clean_ids,
        "ladders": per_ladder_indices,
        "clean_id": args.clean_id if not args.pooled else None,
        "noise1_id": args.noise1_id if not args.pooled else None,
        "noise5_id": args.noise5_id if not args.pooled else None,
        "n_ladders": len(per_ladder_indices),
        "noise1_offset": args.noise1_offset if args.pooled else None,
        "noise5_offset": args.noise5_offset if args.pooled else None,
        "n_paired": int(n_paired),
        "pair_keys": ["omega", "b_mm", "__mirror_file_key"],
        "promotion_rule": "higher_is_noisier; strict clean < 1pct < 5pct; one-sided MWU 5pct > clean",
        "monotone_fraction_threshold": args.monotone_fraction,
        "p_value_threshold": args.p_value_threshold,
        "min_coverage": args.min_coverage,
        "promoted_metrics": promoted,
        "n_promoted": len(promoted),
        "n_tested": len(out_df),
        "noise_score_families": sorted(NOISE_SCORE_FAMILIES),
        "metric_stats": metric_stats,
        "inverse_noise_indicators": inverse_metrics,
        "n_inverse_indicators": len(inverse_metrics),
        "inverse_indicator_note": (
            "Metrics that strictly DECREASE with noise on the mirror ladder. "
            "Not used for decisions in this run; surfaced for transparency. "
            "Useful for catching simulation-vs-real-world sign flips "
            "(e.g., cross-channel coherence dropping with independent ADC noise)."
        ),
    }
    (out_dir / "promoted_metrics.json").write_text(json.dumps(promoted_payload, indent=2))

    print(json.dumps(promoted_payload, indent=2))
    print(f"\nWrote {out_csv}")
    return 0


def _mirror_file_key(row: pd.Series) -> str:
    """Return a dataset-independent file key for clean/noisy mirror matching."""

    file_value = str(row.get("file") or "")
    stem = Path(file_value).stem
    match = re.match(r"^time[_-]?(\d+)[_-](.+)$", stem)
    if match:
        return match.group(2)
    return stem


def _duplicate_payload(frames: Mapping[str, pd.DataFrame], pair_keys: list[str]) -> dict[str, Any]:
    duplicate_key_rows = 0
    preview: list[dict[str, Any]] = []
    for name, df in frames.items():
        duplicates = df[df.duplicated(subset=pair_keys, keep=False)]
        duplicate_key_rows += int(len(duplicates))
        if len(preview) < 12 and not duplicates.empty:
            for row in duplicates.head(12 - len(preview)).to_dict(orient="records"):
                preview.append({
                    "source": name,
                    "dataset": row.get("dataset"),
                    "file": row.get("file"),
                    "omega": row.get("omega"),
                    "b_mm": row.get("b_mm"),
                    "mirror_file_key": row.get("__mirror_file_key"),
                })
    return {"duplicate_key_rows": duplicate_key_rows, "duplicate_preview": preview}


def metric_family(metric: str) -> str:
    if metric.startswith("psd_"):
        return "psd"
    if metric.startswith("wavelet_") or metric.startswith("autocorr_"):
        return "wavelet_temporal"
    if metric.startswith("cross_coh_"):
        return "cross_channel"
    if metric.startswith("chatter_"):
        return "chatter"
    return "other"


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"median": math.nan, "mean": math.nan, "std": math.nan, "p95": math.nan}
    std = float(np.std(finite, ddof=1)) if finite.size >= 2 else 0.0
    return {
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "std": std,
        "p95": float(np.quantile(finite, 0.95)),
    }


def _json_float(value: object) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


if __name__ == "__main__":
    raise SystemExit(main())
