#!/usr/bin/env python3
"""Figure for Research Update 14, Section 3: the repeatability check.

Every value is read from results/phase3/repeatability.json, which
scripts/repeatability_audit.py recomputes from the run summaries. Nothing here
is illustrative.

Panel A is the argument. It plots the per-epoch gap between the two runs of each
duplicated fold, so a pair that reproduced exactly is a flat line on zero and a
pair that did not is visibly not. Plotting the GAP rather than two AUC curves per
fold keeps one measure on one axis: ten overlaid AUC traces across two different
arms would need either two y-scales or a misleading shared one.

Panel B places the two July retrain effects beside the largest observed
same-settings difference in a selected fold-level AUC. These quantities are
shown together as a warning, not as a formal noise-floor comparison.

Palette is the project's validated categorical set; both panels carry direct
labels so identity never rests on colour alone.

    python scripts/plot_update_14.py [--out results/figures/research_update_assets]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})

ARM_LABEL = {"no_side_input_ablation": "side-input ablation",
             "quantile_severity": "quantile severity"}


def _grid(ax, axis="y"):
    ax.grid(True, axis=axis, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def _epoch_gaps(root: Path, tree: str, run_a: str, run_b: str) -> list[float]:
    """Per-epoch |AUC difference| between two runs, straight from their summaries."""
    def aucs(rid):
        p = root / "results/phase3" / tree / rid / "summary.json"
        return [v["auc"] for v in json.loads(p.read_text())["history"]["val"]]
    a, b = aucs(run_a), aucs(run_b)
    return [abs(x - y) for x, y in zip(a, b)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".")
    p.add_argument("--report", default="results/phase3/repeatability.json")
    p.add_argument("--out", default="results/figures/research_update_assets")
    args = p.parse_args(argv)

    root = Path(args.root)
    rep = json.loads((root / args.report).read_text())
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # ---------------- Panel A: did the pair reproduce, epoch by epoch? --------
    tree_of = {"no_side_input_ablation": "runs", "quantile_severity": "runs_quantile"}
    n_identical = n_diverged = 0
    worst = (0.0, None)
    for arm in rep["arms"]:
        tree = tree_of[arm["arm"]]
        for pair in arm["pairs"]:
            gaps = _epoch_gaps(root, tree, pair["earlier"]["run_id"],
                               pair["later"]["run_id"])
            identical = pair.get("bit_identical_run", False)
            if identical:
                n_identical += 1
            else:
                n_diverged += 1
                if max(gaps) > worst[0]:
                    worst = (max(gaps), (gaps, ARM_LABEL[arm["arm"]], pair["fold"]))
            ax1.plot(range(len(gaps)), gaps,
                     color=S1 if identical else S2,
                     lw=2.4 if identical else 1.6,
                     alpha=1.0 if identical else 0.75,
                     zorder=3 if identical else 2)

    # The flat-on-zero lines are the point, so say so on the plot itself rather
    # than leaving the reader to infer that four overlapping lines are four.
    ax1.annotate(f"{n_identical} pairs reproduced exactly\n(flat on zero)",
                 xy=(13, 0), xytext=(9.2, 0.0062), color=S1, fontweight="bold",
                 fontsize=9.5, ha="left",
                 arrowprops=dict(arrowstyle="->", color=S1, lw=1.4))
    if worst[1] is not None:
        gaps, arm_label, fold = worst[1]
        i = int(np.argmax(gaps))
        # Label to the RIGHT of the spike and slightly below its tip: placing it
        # above runs into the axes title at this figure height.
        ax1.annotate(f"largest gap {worst[0]:.4f}\n({arm_label}, fold {fold})",
                     xy=(i, gaps[i]), xytext=(i + 2.0, gaps[i] - 0.0018),
                     color=S2, fontweight="bold", fontsize=9.5, va="top",
                     arrowprops=dict(arrowstyle="->", color=S2, lw=1.4))
    ax1.plot([], [], color=S1, lw=2.4, label=f"reproduced exactly ({n_identical})")
    ax1.plot([], [], color=S2, lw=1.6, label=f"did not ({n_diverged})")
    ax1.legend(frameon=False, loc="upper right", fontsize=9)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("gap in validation AUC between the two runs")
    ax1.set_title("Same settings, same seed, submitted twice", pad=12)
    ax1.set_xlim(-0.4, 19.4)
    ax1.set_xticks(range(0, 20, 2))          # epochs are integers, not 2.5
    ax1.set_ylim(-0.0012, max(worst[0] * 1.18, 0.028))
    _grid(ax1)

    # ----- Panel B: effects beside the largest selected fold-level gap -------
    floor = rep["repeatability_floor_auc"]
    effects, labels = [], []
    for arm in rep["arms"]:
        effects.append(abs(arm["effect_vs_baseline_later"]))
        labels.append(ARM_LABEL[arm["arm"]])
    y = np.arange(len(effects))
    ax2.barh(y, effects, height=0.45, color=S1, zorder=3)
    for yi, v in zip(y, effects):
        ax2.text(v + 0.00025, yi, f"{v:.4f}", va="center", color=INK,
                 fontsize=9.5, fontweight="bold")
    ax2.axvline(floor, color=S2, lw=2.0, ls="--", zorder=4)
    # Inside the axes, to the right of the rule, vertically centred between the
    # two bars: above the top bar it lands outside the plot area.
    ax2.text(floor + floor * 0.05, (len(effects) - 1) / 2,
             f"largest observed\nfold gap {floor:.4f}",
             color=S2, fontsize=9.5, fontweight="bold", va="center", ha="left")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_ylim(-0.6, len(effects) - 0.4)
    ax2.set_xlim(0, floor * 1.62)
    ax2.set_xlabel("effect on 5-fold mean AUC")
    ax2.set_title("Effects beside the largest observed fold gap", pad=12)
    _grid(ax2, axis="x")

    fig.tight_layout()
    path = out_dir / "u14_repeatability.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")
    print(f"  reproduced exactly {n_identical}/{n_identical + n_diverged}, "
          f"largest single-epoch gap {worst[0]:.4f}, "
          f"largest selected fold gap {floor:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
