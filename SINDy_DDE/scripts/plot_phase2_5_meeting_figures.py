#!/usr/bin/env python3
"""Produce 3 meeting-ready PNG figures from the pooled Phase 2.5 calibration outputs.

Inputs (all have sensible defaults that match the standard pooled-v2 layout):

  --mode-labels-csv      Per-file mode labels.
  --promoted-json        Promoted metrics + metric_stats from calibration.
  --calibration-csv      Optional per-metric calibration CSV (for effect sizes).
  --output-dir           Where to write PNGs and summary JSON.

Outputs written to --output-dir:

  fig_mode_distribution_by_tier.png
      Stacked-bar of mode-label PROPORTIONS across 4 dataset tiers.

  fig_per_dataset_broadband_rate.png
      Bar per dataset 1-120: fraction of files labelled broadband_noise_dominated.
      Known clean false-positives (D=23, D=12) and weak 5%-noise detections
      (D=102, D=99) are annotated.

  fig_promoted_metrics.png
      Two panels: monotone_fraction (left) and effect proxy (right), sorted by
      effect.  Effect = mean_delta_5pct_minus_clean / clean_std from the JSON,
      or paired_effect_size from the calibration CSV when available.

  figures_summary.json
      Per-tier mode counts, top-5 clean broadband rates, promoted metric list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Tier helpers (pure logic, no matplotlib dependency)
# ---------------------------------------------------------------------------

_TIER_BOUNDS = [
    (1, 30, "clean_1_30"),
    (31, 60, "clean_31_60"),
    (61, 90, "noise_1pct"),
    (91, 120, "noise_5pct"),
]

_TIER_LABELS = {
    "clean_1_30": "Clean D1-30\n(in pool)",
    "clean_31_60": "Clean D31-60\n(held-out)",
    "noise_1pct": "1% noise\nD61-90",
    "noise_5pct": "5% noise\nD91-120",
}


def tier_for_dataset(dataset_id: int) -> str:
    """Return tier name for a dataset id 1-120.

    >>> tier_for_dataset(1)
    'clean_1_30'
    >>> tier_for_dataset(91)
    'noise_5pct'
    """
    for lo, hi, name in _TIER_BOUNDS:
        if lo <= dataset_id <= hi:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_mode_labels(path: os.PathLike) -> "pandas.DataFrame":
    """Load mode_labels CSV and append a ``tier`` column."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required: pip install pandas") from exc

    df = pd.read_csv(path, low_memory=False)
    df["tier"] = df["dataset"].apply(tier_for_dataset)
    return df


def load_promoted_json(path: os.PathLike) -> dict:
    """Load promoted_metrics.json and return the parsed dict."""
    with open(path) as fh:
        return json.load(fh)


def load_calibration_csv(path: Optional[os.PathLike]) -> "Optional[pandas.DataFrame]":
    """Load optional calibration CSV; return None if path is None or missing."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    return pd.read_csv(p, low_memory=False)


# ---------------------------------------------------------------------------
# Mode label ordering / colours (shared across figures)
# ---------------------------------------------------------------------------

MODE_ORDER = [
    "tpf_dominated_clean",
    "near_boundary_transition",
    "chatter_dominated",
    "mixed_tpf_chatter",
    "broadband_noise_dominated",
    "metrics_unavailable",
]

MODE_COLORS = {
    "tpf_dominated_clean":      "#2ca02c",
    "near_boundary_transition": "#ff7f0e",
    "chatter_dominated":        "#d62728",
    "mixed_tpf_chatter":        "#9467bd",
    "broadband_noise_dominated": "#1f77b4",
    "metrics_unavailable":      "#7f7f7f",
}

_TIER_ORDER = ["clean_1_30", "clean_31_60", "noise_1pct", "noise_5pct"]
_TIER_PALETTE = {
    "clean_1_30":   "#2ca02c",
    "clean_31_60":  "#98df8a",
    "noise_1pct":   "#ff7f0e",
    "noise_5pct":   "#d62728",
}


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_mode_distribution_by_tier(
    df: "pandas.DataFrame",
) -> "matplotlib.figure.Figure":
    """Grouped stacked bars of mode-label proportions across the 4 tiers.

    Each bar group corresponds to one tier; within the bar, each segment is the
    proportion of files assigned to that mode label.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    tiers_present = [t for t in _TIER_ORDER if t in df["tier"].unique()]
    modes_present = [m for m in MODE_ORDER if m in df["mode_label"].unique()]

    # Proportions (0-1) per tier
    prop = {}
    for tier in tiers_present:
        sub = df[df["tier"] == tier]
        total = len(sub)
        row = {}
        for mode in modes_present:
            row[mode] = (sub["mode_label"] == mode).sum() / max(total, 1)
        prop[tier] = row

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(tiers_present))
    bottoms = [0.0] * len(tiers_present)

    for mode in modes_present:
        vals = [prop[t][mode] for t in tiers_present]
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            width=0.65,
            color=MODE_COLORS.get(mode, "#aaaaaa"),
            edgecolor="white",
            linewidth=0.6,
            label=mode,
        )
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 0.04:
                ax.text(
                    i, b + v / 2,
                    f"{v * 100:.0f}%",
                    ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold",
                )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(list(x))
    ax.set_xticklabels([_TIER_LABELS.get(t, t) for t in tiers_present], fontsize=10)
    ax.set_ylabel("Proportion of files", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 2.5 pooled: mode-label distribution by tier", fontsize=11)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        frameon=False,
        title="Mode label",
    )
    # Annotate tier n
    for i, tier in enumerate(tiers_present):
        n = (df["tier"] == tier).sum()
        ax.text(i, 1.02, f"n={n}", ha="center", va="bottom", fontsize=8, color="#555")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    return fig


def build_per_dataset_broadband_rate(
    df: "pandas.DataFrame",
) -> "matplotlib.figure.Figure":
    """Bar plot of broadband_noise_dominated fraction for each dataset 1-120.

    Bars are coloured by tier.  Known clean false-positives and weak 5%-noise
    detections are annotated with a dagger symbol and a brief label.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute broadband fraction per dataset
    datasets = sorted(df["dataset"].unique())
    rates = {}
    for d in datasets:
        sub = df[df["dataset"] == d]
        total = len(sub)
        bb = (sub["mode_label"] == "broadband_noise_dominated").sum()
        rates[d] = bb / max(total, 1)

    xs = list(range(len(datasets)))
    ys = [rates[d] for d in datasets]
    colors = [_TIER_PALETTE.get(tier_for_dataset(d), "#888") for d in datasets]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    bars = ax.bar(xs, ys, width=0.8, color=colors, edgecolor="none", alpha=0.85)

    # Vertical tier separators
    tier_boundaries = [30, 60, 90]  # dataset IDs at which tier changes
    for tb in tier_boundaries:
        if tb in datasets:
            idx = datasets.index(tb)
            ax.axvline(idx + 0.5, color="#888", linewidth=1.0, linestyle="--", alpha=0.5)

    # Annotations for notable datasets
    _ANNOTATIONS = {
        23:  ("D23\ncln FP", "bottom"),
        12:  ("D12\ncln FP", "bottom"),
        102: ("D102\nweak", "top"),
        99:  ("D99\nweak", "top"),
    }
    for d_id, (label, va) in _ANNOTATIONS.items():
        if d_id in datasets:
            idx = datasets.index(d_id)
            y_val = rates[d_id]
            offset = 0.03 if va == "top" else -0.03
            ax.annotate(
                label,
                xy=(idx, y_val),
                xytext=(idx, y_val + offset),
                ha="center",
                va="bottom" if va == "top" else "top",
                fontsize=7,
                color="#c00",
                arrowprops=dict(arrowstyle="-", color="#c00", lw=0.8),
            )
            # Outline the bar
            bars[idx].set_edgecolor("#c00")
            bars[idx].set_linewidth(1.5)

    ax.set_xlabel("Dataset ID", fontsize=10)
    ax.set_ylabel("Broadband fraction", fontsize=10)
    ax.set_title(
        "Per-dataset broadband_noise_dominated rate (Phase 2.5 pooled)", fontsize=11
    )
    ax.set_ylim(0, min(1.0, max(ys + [0.05]) * 1.25))

    # x-tick: every 10th dataset
    tick_positions = [i for i, d in enumerate(datasets) if d % 10 == 0 or d == 1]
    tick_labels = [str(datasets[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)

    # Tier legend patches
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_TIER_PALETTE[t], label=_TIER_LABELS[t].replace("\n", " "))
        for t in _TIER_ORDER
        if t in {tier_for_dataset(d) for d in datasets}
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        frameon=False,
        title="Tier",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    return fig


def build_promoted_metrics(
    promoted_json: dict,
    calibration_df: "Optional[pandas.DataFrame]",
) -> "matplotlib.figure.Figure":
    """Two-panel figure: monotone_fraction and effect size for promoted metrics.

    Effect proxy = mean_delta_5pct_minus_clean / clean_std (from JSON), OR
    paired_effect_size from the calibration CSV when available.  Sorted by
    effect (descending).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    promoted_names: list[str] = promoted_json.get("promoted_metrics", [])
    metric_stats: dict = promoted_json.get("metric_stats", {})

    if not promoted_names:
        # Nothing to plot; return a blank figure
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No promoted metrics", ha="center", va="center")
        return fig

    # Build effect values — prefer calibration CSV's paired_effect_size
    effect_from_csv: dict[str, float] = {}
    if calibration_df is not None and "metric" in calibration_df.columns:
        cal = calibration_df[calibration_df["promoted"].astype(str).str.lower() == "true"]
        if "paired_effect_size" in cal.columns:
            for _, row in cal.iterrows():
                try:
                    effect_from_csv[row["metric"]] = float(row["paired_effect_size"])
                except (ValueError, TypeError):
                    pass

    metrics = []
    monotone_vals = []
    effect_vals = []

    for m in promoted_names:
        stats = metric_stats.get(m, {})
        monotone_fraction = float(stats.get("monotone_fraction", 0.0))

        if m in effect_from_csv:
            effect = effect_from_csv[m]
        else:
            delta = float(stats.get("mean_delta_5pct_minus_clean", 0.0))
            std = float(stats.get("clean_std", 0.0))
            effect = delta / std if std > 0 else 0.0

        metrics.append(m)
        monotone_vals.append(monotone_fraction)
        effect_vals.append(effect)

    # Sort by effect descending
    order = sorted(range(len(metrics)), key=lambda i: -effect_vals[i])
    metrics = [metrics[i] for i in order]
    monotone_vals = [monotone_vals[i] for i in order]
    effect_vals = [effect_vals[i] for i in order]

    # Shorten metric names for display
    short_names = [m.replace("psd_", "").replace("wavelet_", "wav_") for m in metrics]

    y = np.arange(len(metrics))

    fig, (ax_mono, ax_eff) = plt.subplots(1, 2, figsize=(12, max(3.5, len(metrics) * 0.55)))

    # Left: monotone_fraction
    bars_m = ax_mono.barh(y, monotone_vals, color="#1f77b4", edgecolor="white", height=0.65)
    ax_mono.axvline(0.85, color="orange", linewidth=1.2, linestyle="--", alpha=0.8,
                    label="0.85 threshold")
    ax_mono.set_xlim(0, 1.05)
    ax_mono.set_yticks(y)
    ax_mono.set_yticklabels(short_names, fontsize=8)
    ax_mono.set_xlabel("Monotone fraction", fontsize=9)
    ax_mono.set_title("Monotone fraction\n(clean→1%→5%)", fontsize=9)
    ax_mono.legend(fontsize=7, frameon=False)
    ax_mono.spines["top"].set_visible(False)
    ax_mono.spines["right"].set_visible(False)
    ax_mono.grid(axis="x", alpha=0.25, linewidth=0.6)

    for bar, val in zip(bars_m, monotone_vals):
        ax_mono.text(
            min(val + 0.01, 1.0), bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=7
        )

    # Right: effect proxy
    eff_label = "Effect size\n(paired_effect_size or Δ/σ)"
    bars_e = ax_eff.barh(y, effect_vals, color="#d62728", edgecolor="white", height=0.65)
    ax_eff.set_yticks(y)
    ax_eff.set_yticklabels(short_names, fontsize=8)
    ax_eff.set_xlabel(eff_label, fontsize=9)
    ax_eff.set_title("Effect proxy\n(5% noise vs clean)", fontsize=9)
    ax_eff.spines["top"].set_visible(False)
    ax_eff.spines["right"].set_visible(False)
    ax_eff.grid(axis="x", alpha=0.25, linewidth=0.6)

    for bar, val in zip(bars_e, effect_vals):
        ax_eff.text(
            val + max(abs(val) * 0.01, 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=7
        )

    fig.suptitle("Promoted metrics: monotone fraction & effect proxy", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def build_summary(
    df: "pandas.DataFrame",
    promoted_json: dict,
) -> dict:
    """Build the figures_summary.json payload."""
    # Per-tier mode counts
    tier_mode_counts: dict[str, dict[str, int]] = {}
    for tier in _TIER_ORDER:
        sub = df[df["tier"] == tier]
        counts = sub["mode_label"].value_counts().to_dict()
        if counts:
            tier_mode_counts[tier] = {k: int(v) for k, v in counts.items()}

    # Top-5 clean (tiers 1+2) broadband rates
    clean_df = df[df["tier"].isin(["clean_1_30", "clean_31_60"])]
    bb_rates: list[tuple[int, float]] = []
    for d in sorted(clean_df["dataset"].unique()):
        sub = clean_df[clean_df["dataset"] == d]
        rate = (sub["mode_label"] == "broadband_noise_dominated").sum() / max(len(sub), 1)
        bb_rates.append((int(d), float(rate)))
    bb_rates.sort(key=lambda x: -x[1])
    top5 = bb_rates[:5]

    return {
        "tier_mode_counts": tier_mode_counts,
        "top5_clean_broadband_rates": top5,
        "promoted_metrics": promoted_json.get("promoted_metrics", []),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode-labels-csv",
        default="results/pri2m_pooled_v2/mode_labels.csv",
        help="Per-file mode-label CSV (output of assign_signal_modes.py)",
    )
    p.add_argument(
        "--promoted-json",
        default="results/pri2m_pooled_v2/promoted_metrics.json",
        help="promoted_metrics.json from calibrate_noise_mirror_pairs.py",
    )
    p.add_argument(
        "--calibration-csv",
        default=None,
        help=(
            "Optional mirror_calibration_pooled.csv.  When present and it "
            "contains a paired_effect_size column, those values replace the "
            "Δ/σ proxy in fig_promoted_metrics.png."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="results/figures/phase2_5_meeting/",
        help="Directory where PNGs and figures_summary.json are written.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Set up matplotlib with a writable font-cache dir before any pyplot import
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mpl_config = out_dir / ".mplconfig"
    mpl_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401 — force Agg before any draw
    except ImportError as exc:
        raise SystemExit("matplotlib is required: pip install matplotlib") from exc

    # Load data
    df = load_mode_labels(args.mode_labels_csv)
    promoted_json = load_promoted_json(args.promoted_json)
    calibration_df = load_calibration_csv(args.calibration_csv)

    # Build and save figures
    figs = {
        "fig_mode_distribution_by_tier.png":  build_mode_distribution_by_tier(df),
        "fig_per_dataset_broadband_rate.png":  build_per_dataset_broadband_rate(df),
        "fig_promoted_metrics.png":            build_promoted_metrics(promoted_json, calibration_df),
    }
    for fname, fig in figs.items():
        dest = out_dir / fname
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  wrote {dest}", file=sys.stderr)

    # Write summary JSON
    summary = build_summary(df, promoted_json)
    summary_path = out_dir / "figures_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
