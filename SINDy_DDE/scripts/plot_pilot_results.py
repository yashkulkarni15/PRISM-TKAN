#!/usr/bin/env python3
"""Render Phase 2.5 pilot calibration figures for the 1/61/91 mirror ladder.

Inputs (all required):
  --noise-metrics-csv      Per-file noise metrics (output of compute_noise_metrics.py).
  --promoted-json          promoted_metrics.json from calibrate_noise_mirror_pairs.py.
  --mode-labels-csv        mode_labels.csv from assign_signal_modes.py.
  --output-dir             Where to write PNGs.

Produces three figures suitable for sharing with the professor:

  pilot_mode_distribution.png
      Stacked-bar mode-label distribution per dataset. The headline figure:
      shows the monotone shift from physics-dominated (Dataset 1) to
      broadband-noise-dominated (Dataset 91).

  pilot_ladder_boxplots.png
      Box-and-whisker plot of every promoted metric across the three noise
      levels. One panel per metric (top N by paired effect size).

  pilot_paired_ladders.png
      Per-file paired-triple trace for two highlight metrics
      (psd_noise_floor_ratio_Forcex and cross_coh_xtpos_ytpos). Each thin
      gray line is one operating point's (clean, 1pct, 5pct) values; the
      thick line is the median per noise level. Demonstrates that the ladder
      gate is satisfied on essentially every paired file, not in distribution
      only.

All three figures save at 150 dpi.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODE_ORDER = (
    "tpf_dominated_clean",
    "near_boundary_transition",
    "chatter_dominated",
    "mixed_tpf_chatter",
    "broadband_noise_dominated",
    "low_speed_edge",
    "metrics_unavailable",
)

MODE_COLORS = {
    "tpf_dominated_clean": "#2ca02c",
    "near_boundary_transition": "#ff7f0e",
    "chatter_dominated": "#d62728",
    "mixed_tpf_chatter": "#9467bd",
    "broadband_noise_dominated": "#1f77b4",
    "low_speed_edge": "#8c564b",
    "metrics_unavailable": "#7f7f7f",
}

HIGHLIGHT_LADDER_METRICS = (
    "psd_noise_floor_ratio_Forcex",
    "cross_coh_xtpos_ytpos",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-metrics-csv", required=True)
    parser.add_argument("--promoted-json", required=True)
    parser.add_argument("--mode-labels-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--top-n-ladder-metrics",
        type=int,
        default=6,
        help="Number of promoted metrics to show in the ladder boxplot grid.",
    )
    parser.add_argument(
        "--clean-id", type=int, default=1,
        help="Clean dataset id (for paired-ladder traces).",
    )
    parser.add_argument("--noise1-id", type=int, default=61)
    parser.add_argument("--noise5-id", type=int, default=91)
    return parser.parse_args(argv)


def figure_mode_distribution(
    modes: pd.DataFrame, out_path: Path
) -> None:
    """Stacked-bar mode-label distribution per dataset."""
    counts = (
        modes.groupby(["dataset", "mode_label"]).size().unstack(fill_value=0)
    )
    counts = counts.reindex(columns=[m for m in MODE_ORDER if m in counts.columns])
    datasets = counts.index.tolist()
    labels = counts.columns.tolist()
    totals = counts.sum(axis=1)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bottoms = np.zeros(len(datasets), dtype=float)
    bar_width = 0.55
    for label in labels:
        values = counts[label].to_numpy().astype(float)
        ax.bar(
            range(len(datasets)),
            values,
            width=bar_width,
            bottom=bottoms,
            color=MODE_COLORS.get(label, "#7f7f7f"),
            edgecolor="white",
            linewidth=0.6,
            label=label,
        )
        for i, v in enumerate(values):
            if v >= max(2.0, 0.04 * totals.iloc[i]):
                ax.text(
                    i,
                    bottoms[i] + v / 2,
                    f"{int(v)}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
        bottoms += values

    ax.set_xticks(range(len(datasets)))
    noise_pct = {1: "clean", 61: "1% noise", 91: "5% noise"}
    ax.set_xticklabels(
        [f"Dataset {d}\n({noise_pct.get(int(d), '?')})" for d in datasets]
    )
    ax.set_ylabel("File count")
    ax.set_title("Phase 2.5 pilot: mode-label distribution by noise level")
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False
    )
    for i, total in enumerate(totals):
        ax.text(
            i,
            total + max(2, total * 0.02),
            f"n={int(total)}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )
    ax.set_ylim(0, max(totals) * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_ladder_boxplots(
    nm: pd.DataFrame,
    promoted: dict,
    out_path: Path,
    *,
    top_n: int,
    clean_id: int,
    noise1_id: int,
    noise5_id: int,
) -> None:
    """Boxplot of each top-N promoted metric at the three noise levels."""
    stats = promoted.get("metric_stats") or {}
    # Rank by |paired effect size| -- recompute from the promoted CSV side
    # would be cleaner, but the JSON has enough.
    ranking = []
    for metric, m_stats in stats.items():
        mean_delta = m_stats.get("mean_delta_5pct_minus_clean")
        clean_std = m_stats.get("clean_std")
        if mean_delta is None or clean_std in (None, 0):
            continue
        ranking.append((metric, abs(float(mean_delta)) / float(clean_std)))
    ranking.sort(key=lambda r: r[1], reverse=True)
    chosen = [m for m, _ in ranking[:top_n]]

    if not chosen:
        print("warning: no promoted metrics; skipping ladder boxplot", file=sys.stderr)
        return

    cols = 3 if len(chosen) >= 5 else 2
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows), squeeze=False)
    ax_iter = iter(axes.flatten())

    for metric in chosen:
        ax = next(ax_iter)
        groups = []
        labels = []
        for ds_id, lbl in (
            (clean_id, "clean"),
            (noise1_id, "1%"),
            (noise5_id, "5%"),
        ):
            vals = pd.to_numeric(nm.loc[nm["dataset"] == ds_id, metric], errors="coerce").dropna().to_numpy()
            groups.append(vals)
            labels.append(f"{lbl}\nn={len(vals)}")
        # matplotlib 3.9 renamed `labels` to `tick_labels`; support both.
        boxplot_kwargs: dict = {"patch_artist": True, "widths": 0.55, "showfliers": False}
        try:
            from packaging.version import Version
            mpl_new = Version(matplotlib.__version__) >= Version("3.9")
        except Exception:
            mpl_new = matplotlib.__version__ >= "3.9"
        boxplot_kwargs["tick_labels" if mpl_new else "labels"] = labels
        bp = ax.boxplot(groups, **boxplot_kwargs)
        for patch, color in zip(bp["boxes"], ("#2ca02c", "#ff7f0e", "#1f77b4")):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_title(metric, fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for unused_ax in ax_iter:
        unused_ax.set_visible(False)

    fig.suptitle(
        f"Phase 2.5 pilot: promoted-metric distribution across noise levels "
        f"(top {len(chosen)} by paired effect size)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_paired_ladder_traces(
    nm: pd.DataFrame,
    out_path: Path,
    *,
    metrics: tuple[str, ...],
    clean_id: int,
    noise1_id: int,
    noise5_id: int,
) -> None:
    """Per-file paired traces across the three noise levels."""

    metrics = tuple(m for m in metrics if m in nm.columns)
    if not metrics:
        print("warning: no highlight metrics found in CSV; skipping ladder traces", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.5), squeeze=False)
    for ax, metric in zip(axes.flatten(), metrics):
        clean = nm[nm["dataset"] == clean_id].set_index(["omega", "b_mm"])
        n1 = nm[nm["dataset"] == noise1_id].set_index(["omega", "b_mm"])
        n5 = nm[nm["dataset"] == noise5_id].set_index(["omega", "b_mm"])
        common = clean.index.intersection(n1.index).intersection(n5.index)
        v1 = pd.to_numeric(clean.loc[common, metric], errors="coerce").to_numpy()
        v2 = pd.to_numeric(n1.loc[common, metric], errors="coerce").to_numpy()
        v3 = pd.to_numeric(n5.loc[common, metric], errors="coerce").to_numpy()
        mask = np.isfinite(v1) & np.isfinite(v2) & np.isfinite(v3)
        v1, v2, v3 = v1[mask], v2[mask], v3[mask]

        x = np.array([0, 1, 2])
        for i in range(len(v1)):
            up = (v1[i] < v2[i] < v3[i])
            color = "#1f77b4" if up else "#d62728"
            ax.plot(x, [v1[i], v2[i], v3[i]], color=color, alpha=0.2, linewidth=0.7)
        # Median per noise level
        ax.plot(
            x,
            [np.median(v1), np.median(v2), np.median(v3)],
            color="black",
            linewidth=2.0,
            marker="o",
            label="median",
        )
        n_up = int(np.sum((v1 < v2) & (v2 < v3)))
        ax.set_xticks(x)
        ax.set_xticklabels(["clean", "1% noise", "5% noise"])
        ax.set_title(f"{metric}\n{n_up}/{len(v1)} strictly upward", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        "Phase 2.5 pilot: per-file paired ladder traces (blue=upward, red=violation)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    nm = pd.read_csv(args.noise_metrics_csv)
    modes = pd.read_csv(args.mode_labels_csv)
    promoted = json.loads(Path(args.promoted_json).read_text())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_mode_distribution(modes, out_dir / "pilot_mode_distribution.png")
    figure_ladder_boxplots(
        nm, promoted, out_dir / "pilot_ladder_boxplots.png",
        top_n=args.top_n_ladder_metrics,
        clean_id=args.clean_id,
        noise1_id=args.noise1_id,
        noise5_id=args.noise5_id,
    )
    figure_paired_ladder_traces(
        nm, out_dir / "pilot_paired_ladders.png",
        metrics=HIGHLIGHT_LADDER_METRICS,
        clean_id=args.clean_id,
        noise1_id=args.noise1_id,
        noise5_id=args.noise5_id,
    )
    print(f"Wrote three figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
