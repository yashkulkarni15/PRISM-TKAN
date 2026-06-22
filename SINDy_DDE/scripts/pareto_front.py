#!/usr/bin/env python3
"""Phase 2.5 Pareto front: noise suppression vs. physics preservation.

Given a baseline ``noise_metrics.csv`` (typically the unfiltered noisy data) and
one or more candidate ``noise_metrics.csv`` files (each representing a filter
variant applied to the same operating points), this script computes two scores
per variant and emits a Pareto-front summary:

* ``noise_suppression_score = mean_baseline_noise_score - mean_variant_noise_score``
* ``physics_preservation_score = mean(closeness_to_1(chatter_ratio), closeness_to_1(tpf_ratio))``
  where ``closeness_to_1(r) = max(0, 1 - |r - 1|)`` and each ratio compares
  variant mean to baseline mean.

The noise score uses the calibrated metric stats produced by
``calibrate_noise_mirror_pairs.py``: each promoted metric contributes a clipped
clean-baseline z-score (max(0, (value - clean_median) / clean_std)), averaged
across promoted metrics. The physics band-energy score uses the mean of
``chatter_fn_band_energy_*`` and ``chatter_sideband_energy_*`` across the four
signals (TPF energy is captured indirectly via the off-harmonic ratio, which
goes UP when TPF is removed -- so ``1 - psd_off_harmonic_ratio`` is the
TPF-preservation surrogate). Closeness-to-1 means a filter that amplifies the
chatter band scores no better than one that erases it; only filters that hold
the band near baseline get full credit.

When a ``mode_labels.csv`` is supplied (typically from the baseline's
``assign_signal_modes`` run), the physics axes are mode-aware: chatter
preservation is scored only on rows whose mode label is in
``{chatter_dominated, mixed_tpf_chatter}`` and TPF preservation only on rows
in ``{tpf_dominated_clean, near_boundary_transition, mixed_tpf_chatter}``.
Without a mode-labels CSV the scoring falls back to a global mode that
penalizes filters for removing broadband noise that happens to overlap the
``fn`` band on broadband-noise files.

The script writes:

* ``pareto_points.csv`` -- one row per variant with both axes plus diagnostics.
* ``pareto_summary.json`` -- the Pareto-optimal subset and the inputs.
* ``pareto_front.png`` -- a scatter plot with the Pareto-optimal subset connected.

The actual filter implementations are a separate Phase 2.5 deliverable; this
script just defines and evaluates the two axes so that downstream filter work
has a well-specified target.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pri2m.noise.metrics import SIGNALS  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-metrics-csv",
        required=True,
        help="noise_metrics.csv computed on the un-filtered (noisy) data.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Filter variant entry, formatted name=path/to/variant_noise_metrics.csv. "
             "May be repeated.",
    )
    parser.add_argument(
        "--calibration-json",
        required=True,
        help="promoted_metrics.json from calibrate_noise_mirror_pairs.py. "
             "Used to define the noise score.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset-filter",
        type=str,
        default=None,
        help="Optional comma-separated dataset IDs to restrict the scoring to. "
             "If omitted, every row in baseline and variant is used.",
    )
    parser.add_argument(
        "--mode-labels-csv",
        default=None,
        help="Optional mode_labels.csv (typically from the baseline's assign_signal_modes run). "
             "When provided, chatter-band preservation is scored only on rows whose mode is in "
             "{chatter_dominated, mixed_tpf_chatter} and TPF preservation only on rows in "
             "{tpf_dominated_clean, near_boundary_transition, mixed_tpf_chatter}. Without it the "
             "script falls back to global scoring (degraded mode: a filter that removes broadband "
             "noise overlapping the fn band will be penalized on broadband-noise files).",
    )
    parser.add_argument(
        "--max-row-mismatch-frac",
        type=float,
        default=0.05,
        help="Maximum tolerated fraction of baseline rows absent from a variant after inner-join "
             "on (dataset, file).  If (n_baseline - n_joined) / n_baseline exceeds this value the "
             "script exits with a non-zero status.  Default: 0.05 (5 %%).",
    )
    return parser.parse_args(argv)


def _parse_variant(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise SystemExit(f"--variant must be name=path; got {arg!r}")
    name, raw_path = arg.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name:
        raise SystemExit(f"--variant has empty name in {arg!r}")
    if not path.exists():
        raise SystemExit(f"--variant {name!r} points at missing file: {path}")
    return name, path


def _restrict_datasets(df: pd.DataFrame, dataset_filter: str | None) -> pd.DataFrame:
    if not dataset_filter:
        return df
    ids = [int(x.strip()) for x in dataset_filter.split(",") if x.strip()]
    return df[df["dataset"].astype(int).isin(ids)].copy()


def _inner_join_on_dataset_file(
    baseline_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    variant_name: str,
    max_mismatch_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int, int]:
    """Inner-join baseline and variant on (dataset, file).

    Returns
    -------
    (baseline_joined, variant_joined, n_baseline, n_variant, n_joined, n_dropped)

    Raises SystemExit when the fraction of dropped baseline rows exceeds
    *max_mismatch_frac*.
    """
    n_baseline = len(baseline_df)
    n_variant = len(variant_df)

    # Build merge keys; cast consistently so dtypes do not cause mismatches.
    base_keys = baseline_df[["dataset", "file"]].copy()
    base_keys["dataset"] = base_keys["dataset"].astype(str)
    base_keys["file"] = base_keys["file"].astype(str)

    var_keys = variant_df[["dataset", "file"]].copy()
    var_keys["dataset"] = var_keys["dataset"].astype(str)
    var_keys["file"] = var_keys["file"].astype(str)

    # Identify the shared (dataset, file) pairs.
    shared = pd.merge(
        base_keys.drop_duplicates(),
        var_keys.drop_duplicates(),
        on=["dataset", "file"],
        how="inner",
    )
    shared_set = set(zip(shared["dataset"], shared["file"]))

    def _in_shared(df: pd.DataFrame) -> pd.Series:
        return df.apply(
            lambda r: (str(r["dataset"]), str(r["file"])) in shared_set, axis=1
        )

    baseline_joined = baseline_df[_in_shared(baseline_df)].copy()
    variant_joined = variant_df[_in_shared(variant_df)].copy()

    n_joined = len(baseline_joined)
    n_dropped = n_baseline - n_joined

    if n_baseline > 0:
        mismatch_frac = n_dropped / n_baseline
        if mismatch_frac > max_mismatch_frac:
            raise SystemExit(
                f"Row mismatch for variant {variant_name!r}: "
                f"n_baseline={n_baseline}, n_variant={n_variant}, n_joined={n_joined}, "
                f"n_dropped={n_dropped} ({mismatch_frac:.1%} > tolerance {max_mismatch_frac:.1%}). "
                "Ensure the variant CSV covers the same (dataset, file) operating points as the "
                "baseline, or raise --max-row-mismatch-frac."
            )

    return baseline_joined, variant_joined, n_baseline, n_variant, n_joined, n_dropped


def _compute_noise_score(
    df: pd.DataFrame, calibration: dict[str, Any]
) -> tuple[float, int, list[str]]:
    """Return (mean_noise_score, n_metrics_used, metrics_used)."""
    metric_stats = calibration.get("metric_stats") or {}
    if not isinstance(metric_stats, dict) or not metric_stats:
        return math.nan, 0, []

    per_row_scores: list[float] = []
    metrics_used: list[str] = []
    for metric, stats in metric_stats.items():
        if not isinstance(stats, dict):
            continue
        if metric not in df.columns:
            continue
        clean_median = stats.get("clean_median")
        clean_std = stats.get("clean_std")
        if clean_median is None:
            continue
        scale = clean_std if (clean_std is not None and clean_std > 0) else 1.0
        v = pd.to_numeric(df[metric], errors="coerce").to_numpy()
        finite = np.isfinite(v)
        if not finite.any():
            continue
        z = (v[finite] - float(clean_median)) / float(scale)
        per_row_scores.extend([max(0.0, float(zi)) for zi in z])
        metrics_used.append(str(metric))

    if not per_row_scores:
        return math.nan, 0, metrics_used
    return float(np.mean(per_row_scores)), len(metrics_used), metrics_used


# Chatter band is protected only when the baseline file actually has chatter.
_CHATTER_PROTECTING_MODES = ("chatter_dominated", "mixed_tpf_chatter")

# TPF is the cutting signature -- a filter that removes it on a noisy file is
# also failing, so TPF protection applies on every mode where TPF is believed
# to be present. Only metrics_unavailable and low_speed_edge are excluded.
_TPF_PROTECTING_MODES = (
    "tpf_dominated_clean",
    "near_boundary_transition",
    "mixed_tpf_chatter",
    "chatter_dominated",
    "broadband_noise_dominated",
)


def _restrict_to_modes(
    df: pd.DataFrame,
    mode_labels: pd.DataFrame | None,
    allowed_modes: tuple[str, ...],
) -> pd.DataFrame:
    """Filter df to rows whose (dataset, file) maps to a mode in allowed_modes.

    When mode_labels is None, returns df unchanged (degraded global mode).
    """
    if mode_labels is None:
        return df
    keep_keys = mode_labels.loc[
        mode_labels["mode_label"].isin(allowed_modes), ["dataset", "file"]
    ]
    if keep_keys.empty:
        return df.iloc[0:0]
    keys = set(zip(keep_keys["dataset"].astype(int), keep_keys["file"].astype(str)))
    return df[df.apply(lambda r: (int(r["dataset"]), str(r["file"])) in keys, axis=1)].copy()


def _compute_physics_preservation_axes(
    df: pd.DataFrame,
    mode_labels: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute mean chatter-band energy and TPF proxy.

    When mode_labels is supplied, the chatter-band mean is computed only on
    chatter-protecting rows and the TPF proxy only on TPF-protecting rows.
    """
    chatter_df = _restrict_to_modes(df, mode_labels, _CHATTER_PROTECTING_MODES)
    chatter_cols = [
        col for col in chatter_df.columns
        if col.startswith("chatter_fn_band_energy_")
        or col.startswith("chatter_sideband_energy_")
    ]
    if chatter_cols and not chatter_df.empty:
        chatter_values = pd.to_numeric(
            chatter_df[chatter_cols].stack(), errors="coerce"
        ).dropna()
        chatter_mean = float(chatter_values.mean()) if not chatter_values.empty else math.nan
    else:
        chatter_mean = math.nan

    tpf_df = _restrict_to_modes(df, mode_labels, _TPF_PROTECTING_MODES)
    # TPF preservation surrogate: (1 - off-harmonic ratio) averaged across signals.
    # When TPF is preserved, off-harmonic ratio stays low, so this surrogate stays high.
    off_harm_cols = [
        f"psd_off_harmonic_ratio_{signal}" for signal in SIGNALS
        if f"psd_off_harmonic_ratio_{signal}" in tpf_df.columns
    ]
    if off_harm_cols and not tpf_df.empty:
        off_harm = pd.to_numeric(tpf_df[off_harm_cols].stack(), errors="coerce").dropna()
        tpf_proxy = 1.0 - float(off_harm.mean()) if not off_harm.empty else math.nan
    else:
        tpf_proxy = math.nan

    return {
        "chatter_band_energy_mean": chatter_mean,
        "tpf_proxy_mean": tpf_proxy,
        "n_chatter_rows": int(len(chatter_df)),
        "n_tpf_rows": int(len(tpf_df)),
    }


def _closeness_to_one(ratio: float) -> float:
    """Map a preservation ratio onto a [0, 1] closeness score.

    1.0 -> perfect preservation. Both under-preservation (ratio<1, signal
    removed) and over-preservation (ratio>1, band amplified by the filter)
    decrease the score symmetrically. A filter that doubles chatter energy
    therefore scores no better than one that erases it entirely.
    """
    if not math.isfinite(ratio):
        return math.nan
    return max(0.0, 1.0 - abs(ratio - 1.0))


def _is_pareto_optimal(points: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (maximize both axes)."""
    n = points.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated_by = (
            (points[:, 0] >= points[i, 0])
            & (points[:, 1] >= points[i, 1])
            & ((points[:, 0] > points[i, 0]) | (points[:, 1] > points[i, 1]))
        )
        if dominated_by.any():
            mask[i] = False
    return mask


def _render_plot(points_df: pd.DataFrame, out_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dep
        print(f"warning: matplotlib unavailable, skipping plot ({exc})", file=sys.stderr)
        return False

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        points_df["noise_suppression_score"],
        points_df["physics_preservation_score"],
        c=["tab:blue" if optimal else "lightgray" for optimal in points_df["pareto_optimal"]],
        s=80,
        edgecolor="black",
        linewidth=0.5,
    )
    for _, row in points_df.iterrows():
        ax.annotate(
            row["variant"],
            (row["noise_suppression_score"], row["physics_preservation_score"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )
    front = points_df.loc[points_df["pareto_optimal"]].sort_values(
        "noise_suppression_score"
    )
    if len(front) >= 2:
        ax.plot(
            front["noise_suppression_score"],
            front["physics_preservation_score"],
            "--",
            color="tab:blue",
            alpha=0.4,
            label="Pareto front",
        )
        ax.legend(loc="best")
    ax.set_xlabel("Noise suppression score (baseline - variant)")
    ax.set_ylabel("Physics preservation score (closeness to 1, mode-aware)")
    ax.set_title("Phase 2.5 Pareto front: noise vs. physics")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    calibration_path = Path(args.calibration_json)
    if not calibration_path.exists():
        raise SystemExit(f"calibration JSON not found: {calibration_path}")
    calibration = json.loads(calibration_path.read_text())
    if not calibration.get("metric_stats"):
        raise SystemExit(
            f"calibration JSON {calibration_path} has no metric_stats; "
            "rerun calibrate_noise_mirror_pairs.py first."
        )

    baseline_df = _restrict_datasets(
        pd.read_csv(args.baseline_metrics_csv), args.dataset_filter
    )
    if baseline_df.empty:
        raise SystemExit("baseline metrics CSV is empty after dataset filter")

    mode_labels = None
    if args.mode_labels_csv:
        mode_labels_path = Path(args.mode_labels_csv)
        if not mode_labels_path.exists():
            raise SystemExit(f"mode-labels CSV not found: {mode_labels_path}")
        mode_labels = pd.read_csv(mode_labels_path)
        for required in ("dataset", "file", "mode_label"):
            if required not in mode_labels.columns:
                raise SystemExit(
                    f"mode-labels CSV missing required column {required!r}: {mode_labels_path}"
                )

    # Compute overall baseline stats (on the full baseline_df) for the summary.
    _baseline_noise_global, _baseline_n_metrics_global, _ = _compute_noise_score(
        baseline_df, calibration
    )
    _baseline_phys_global = _compute_physics_preservation_axes(baseline_df, mode_labels)

    rows: list[dict[str, Any]] = []
    for raw in args.variant:
        name, path = _parse_variant(raw)
        variant_df_full = _restrict_datasets(pd.read_csv(path), args.dataset_filter)
        if variant_df_full.empty:
            print(f"warning: variant {name!r} has no rows after dataset filter", file=sys.stderr)
            continue

        # Inner-join baseline and variant on (dataset, file) so both sides are
        # scored on the identical row set.  Raises SystemExit when mismatch
        # fraction exceeds the configured tolerance.
        baseline_joined, variant_joined, n_baseline, n_variant, n_joined, n_dropped = (
            _inner_join_on_dataset_file(
                baseline_df, variant_df_full, name, args.max_row_mismatch_frac
            )
        )

        # Compute baseline scores on the joined subset for this variant.
        baseline_noise, baseline_n_metrics, _ = _compute_noise_score(baseline_joined, calibration)
        baseline_phys = _compute_physics_preservation_axes(baseline_joined, mode_labels)
        baseline_chatter = baseline_phys["chatter_band_energy_mean"]
        baseline_tpf = baseline_phys["tpf_proxy_mean"]

        variant_noise, n_metrics, _ = _compute_noise_score(variant_joined, calibration)
        variant_phys = _compute_physics_preservation_axes(variant_joined, mode_labels)

        noise_suppression = (
            baseline_noise - variant_noise
            if math.isfinite(baseline_noise) and math.isfinite(variant_noise)
            else math.nan
        )

        chatter_ratio = (
            variant_phys["chatter_band_energy_mean"] / baseline_chatter
            if (
                math.isfinite(baseline_chatter)
                and baseline_chatter > 0
                and math.isfinite(variant_phys["chatter_band_energy_mean"])
            )
            else math.nan
        )
        tpf_ratio = (
            variant_phys["tpf_proxy_mean"] / baseline_tpf
            if (
                math.isfinite(baseline_tpf)
                and baseline_tpf > 0
                and math.isfinite(variant_phys["tpf_proxy_mean"])
            )
            else math.nan
        )
        chatter_closeness = _closeness_to_one(chatter_ratio)
        tpf_closeness = _closeness_to_one(tpf_ratio)
        closeness_components = [
            v for v in (chatter_closeness, tpf_closeness) if math.isfinite(v)
        ]
        physics_preservation = (
            float(np.mean(closeness_components)) if closeness_components else math.nan
        )

        rows.append({
            "variant": name,
            "path": str(path),
            "n_rows": int(len(variant_joined)),
            "n_baseline": n_baseline,
            "n_variant": n_variant,
            "n_joined": n_joined,
            "n_dropped": n_dropped,
            "noise_score": variant_noise,
            "noise_suppression_score": noise_suppression,
            "chatter_band_energy_mean": variant_phys["chatter_band_energy_mean"],
            "chatter_preservation_ratio": chatter_ratio,
            "chatter_preservation_closeness": chatter_closeness,
            "tpf_proxy_mean": variant_phys["tpf_proxy_mean"],
            "tpf_preservation_ratio": tpf_ratio,
            "tpf_preservation_closeness": tpf_closeness,
            "physics_preservation_score": physics_preservation,
            "n_chatter_rows": variant_phys["n_chatter_rows"],
            "n_tpf_rows": variant_phys["n_tpf_rows"],
            "n_calibrated_metrics_used": n_metrics,
        })

    if not rows:
        raise SystemExit("no usable variant rows; nothing to plot")

    points_df = pd.DataFrame(rows)
    valid = (
        points_df[["noise_suppression_score", "physics_preservation_score"]]
        .apply(lambda s: np.isfinite(s)).all(axis=1)
    )
    points_df["pareto_optimal"] = False
    if valid.any():
        front_mask = _is_pareto_optimal(
            points_df.loc[valid, ["noise_suppression_score", "physics_preservation_score"]].to_numpy()
        )
        points_df.loc[valid, "pareto_optimal"] = front_mask

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "pareto_points.csv"
    points_df.to_csv(out_csv, index=False)

    plot_path = out_dir / "pareto_front.png"
    plot_written = _render_plot(points_df, plot_path)

    summary = {
        "baseline_metrics_csv": args.baseline_metrics_csv,
        "baseline_n_rows": int(len(baseline_df)),
        "baseline_noise_score": _baseline_noise_global,
        "baseline_chatter_band_energy_mean": _baseline_phys_global["chatter_band_energy_mean"],
        "baseline_tpf_proxy_mean": _baseline_phys_global["tpf_proxy_mean"],
        "baseline_n_chatter_rows": _baseline_phys_global["n_chatter_rows"],
        "baseline_n_tpf_rows": _baseline_phys_global["n_tpf_rows"],
        "calibration_json": str(calibration_path),
        "n_calibrated_metrics_total": _baseline_n_metrics_global,
        "dataset_filter": args.dataset_filter,
        "mode_labels_csv": args.mode_labels_csv,
        "mode_aware": mode_labels is not None,
        "max_row_mismatch_frac": args.max_row_mismatch_frac,
        "physics_score_formula": "closeness_to_1 = max(0, 1 - |variant/baseline - 1|)",
        "n_variants": int(len(points_df)),
        "pareto_optimal_variants": (
            points_df.loc[points_df["pareto_optimal"], "variant"].tolist()
        ),
        "paths": {
            "pareto_points_csv": str(out_csv),
            "pareto_front_plot": str(plot_path) if plot_written else None,
        },
    }
    (out_dir / "pareto_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
