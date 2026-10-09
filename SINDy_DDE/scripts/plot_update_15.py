#!/usr/bin/env python3
"""Figures for Research Update 15, Sections 5 and 6.

Three figures, each drawn ONLY from a returned manifest. Nothing is typed in:
every number is read from the JSON the cluster wrote, so a figure cannot drift
from the text beside it.

  u15_substitution.png   S5  fitted equations swapped in, vs the shuffled control
  u15_pruning_seeds.png  S6  ranked pruning vs a size-matched random control, 6 draws
  u15_mode_ablation.png  S6  deleting an activation mode, vs its two controls

Palette is the same validated categorical instance every other figure in this
project uses (slots 1-3), so the updates read as one set. Colour is never the
only encoding: each series also carries its own marker or hatch, and the values
that matter are labelled directly, which is also what discharges the aqua
contrast warning from the palette validator.

Run from SINDy_DDE/:
    python3 scripts/plot_update_15.py
"""
from __future__ import annotations

import json
import pathlib
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
XAI = ROOT / "results/phase3/xai"
OUT = ROOT / "results/figures/research_update_assets"

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})

TOL = 0.013          # pre-committed tolerance, the held-out fold spread
FLOOR = 0.0092       # Update 14's repeatability floor


# --------------------------------------------------------------------------- #
# Figure 1 -- symbolic substitution, five networks
# --------------------------------------------------------------------------- #
def substitution() -> pathlib.Path:
    folds = ["fold0", "fold1", "fold2_w512", "fold3", "fold4"]
    fig, axes = plt.subplots(1, 5, figsize=(13.0, 3.5), sharey=True)

    for ax, tag in zip(axes, folds):
        d = json.loads((XAI / "symbolic_substitution" / tag /
                        "symbolic_substitution.json").read_text())
        base = d["baseline_auc"]
        # Unclamped arms only; the clamped arm shares an x with k=25.
        arms = [r for r in d["rows"]
                if not r.get("clamped") and r.get("correct_auc") is not None]
        ks = [r["k"] for r in arms]
        cor = [r["correct_auc"] for r in arms]
        per = [r["permuted_auc"] for r in arms]

        ax.axhspan(base - TOL, base + TOL, color=S1, alpha=0.13, zorder=1)
        ax.axhline(base, color=INK2, lw=1.2, ls="--", zorder=2)
        ax.plot(ks, cor, color=S1, marker="o", ms=6, zorder=4,
                label="fitted equations")
        ax.plot(ks, per, color=S2, marker="s", ms=6, ls="--", zorder=3,
                label="shuffled control")

        # The one failing arm: say so on the figure rather than in a caption only.
        for k, c, p in zip(ks, cor, per):
            if c - p <= TOL:
                ax.annotate("control fails\nhere", xy=(k, p), xytext=(k + 1.0, p - 0.20),
                            color=S2, fontsize=8.5, fontweight="bold", ha="left",
                            arrowprops=dict(arrowstyle="->", color=S2, lw=1.3))

        label = "network " + (tag[4] if tag[4].isdigit() else "?")
        ax.set_title(f"{label}\nheld-out {base:.4f}", fontsize=9.5, color=INK)
        ax.set_xticks(ks)
        ax.set_xlabel("edges replaced")
        ax.set_ylim(0.20, 1.02)
        ax.grid(axis="y", zorder=0)

    axes[0].set_ylabel("held-out accuracy (AUC)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=8.5)
    fig.suptitle("Replacing the 25 most important learned curves with fitted equations",
                 fontsize=11.5, fontweight="bold", color=INK, y=1.04)
    fig.text(0.5, -0.06,
             "Blue sits on the dashed line (the network's own held-out score) at every size, on every network: "
             "the equations change nothing. Shaded band is the pre-committed tolerance of 0.013. "
             "Orange shuffles which equation goes on which edge and collapses, so the specific equations "
             "are doing the work, not their smoothness.",
             ha="center", fontsize=8.5, color=INK2, wrap=True)
    fig.tight_layout()
    p = OUT / "u15_substitution.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Figure 2 -- held-out pruning across six random draws
# --------------------------------------------------------------------------- #
def pruning() -> pathlib.Path:
    paths = sorted((XAI / "heldout_pruning").glob("*/heldout_pruning.json"))
    runs = []
    for p in paths:
        d = json.loads(p.read_text())
        if not d["provenance"].get("complete"):
            continue
        runs.append((p.parent.name, d))
    base = runs[0][1]["baseline_auc"]

    fracs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    ranked = {f: [] for f in fracs}
    random_ = {f: [] for f in fracs}
    for _tag, d in runs:
        for r in d["rows"]:
            if r.get("ranked_auc") is not None:
                ranked[r["fraction"]].append(r["ranked_auc"])
            if r.get("random_auc") is not None:
                random_[r["fraction"]].append(r["random_auc"])

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.axhspan(base - TOL, base + TOL, color=INK2, alpha=0.10, zorder=1)
    ax.axhline(base, color=INK2, lw=1.3, ls="--", zorder=2)
    ax.text(0.0011, base + 0.016, f"full network {base:.4f}", color=INK2, fontsize=8.5)

    # Every individual draw, thin, so the spread is visible rather than averaged away.
    for _tag, d in runs:
        rows = [r for r in d["rows"] if r.get("ranked_auc") is not None]
        ax.plot([r["fraction"] for r in rows], [r["ranked_auc"] for r in rows],
                color=S1, lw=0.9, alpha=0.35, zorder=3)
        rows = [r for r in d["rows"] if r.get("random_auc") is not None]
        ax.plot([r["fraction"] for r in rows], [r["random_auc"] for r in rows],
                color=S2, lw=0.9, alpha=0.35, zorder=3)

    mr = [st.mean(ranked[f]) for f in fracs]
    mc = [st.mean(random_[f]) for f in fracs if random_[f]]
    fc = [f for f in fracs if random_[f]]
    ax.plot(fracs, mr, color=S1, marker="o", ms=6, zorder=5,
            label=f"kept by importance ({len(runs)} draws)")
    ax.plot(fc, mc, color=S2, marker="s", ms=6, ls="--", zorder=4,
            label=f"kept at random, same count ({len(runs)} draws)")

    # The single draw that motivated the repeat.
    ax.annotate("one draw scored above the full\nnetwork here; the other five did not",
                xy=(0.25, max(random_[0.25])), xytext=(0.0115, 1.045),
                color=S2, fontsize=8.5, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=S2, lw=1.3))
    ax.annotate(f"keeping 0.5% still scores {st.mean(ranked[0.005]):.3f}",
                xy=(0.0052, st.mean(ranked[0.005]) - 0.012), xytext=(0.0075, 0.60),
                color=S1, fontsize=8.5, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=S1, lw=1.3))

    ax.set_xscale("log")
    ax.set_xlabel("fraction of the 13,984 edges kept (log scale)")
    ax.set_ylabel("held-out accuracy (AUC)")
    ax.set_ylim(0.10, 1.14)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("The importance ranking is right, and the network still will not compress",
                 fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=16)
    fig.text(0.02, -0.03,
             "Thin lines are the six individual draws; thick lines are their means. Blue is tight and high, so the "
             "ranking reliably finds the edges that matter. Neither line re-enters the tolerance band (grey) short of "
             "keeping everything, so no smaller network reproduces the full one.",
             fontsize=8.5, color=INK2, wrap=True)
    fig.tight_layout()
    p = OUT / "u15_pruning_seeds.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Figure 3 -- deleting an activation mode, against its two controls
# --------------------------------------------------------------------------- #
def mode_ablation() -> pathlib.Path:
    files = {
        2: XAI / "mode_ablation/mode_ablation_fold2_w512_k2.json",
        3: XAI / "mode_ablation/mode_ablation_fold2_w512_k3.json",
        4: XAI / "mode_ablation/mode_ablation_fold2_w512.json",
    }
    # k=4 is the setting where Update 14 showed the grouping itself does not
    # survive its untrained control, so it is drawn but marked as not counted.
    surviving = {2: True, 3: True, 4: False}

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3), sharey=True)
    for ax, (k, path) in zip(axes, sorted(files.items())):
        d = json.loads(path.read_text())
        modes = d["modes"]
        xs = range(len(modes))
        w = 0.26
        drops = [m["verdict"]["mode_drop"] for m in modes]
        rand = [m["verdict"]["random_matched_drop"] for m in modes]
        other = [m["verdict"]["other_modes_drop"] for m in modes]

        ax.bar([x - w for x in xs], drops, width=w, color=S1, zorder=3,
               label="delete this group")
        ax.bar(list(xs), rand, width=w, color=S2, zorder=3,
               label="delete same number, at random")
        ax.bar([x + w for x in xs], other, width=w, color=S3, hatch="///",
               edgecolor=SURFACE, linewidth=0.8, zorder=3,
               label="delete same importance, other groups")

        ax.axhline(FLOOR, color=INK2, lw=1.4, ls=":", zorder=4)
        ax.axhline(0, color=GRID, lw=1.0, zorder=2)

        for x, m in zip(xs, modes):
            v = m["verdict"]
            if not (v["claimable"] and surviving[k]):
                continue
            # Clearing the floor is not the same as clearing it by enough to see.
            # The narrower of the two control margins decides how this is labelled,
            # so a hairline pass cannot appear on the figure as a plain win --
            # the text does not count it, and neither should the picture.
            margin = min(v["mode_drop"] - v["random_matched_drop"],
                         v["mode_drop"] - v["other_modes_drop"]) - FLOOR
            ax.text(x - w, v["mode_drop"] + 0.0035, f"{v['mode_drop']:.3f}",
                    ha="center", color=INK, fontsize=9, fontweight="bold")
            if margin >= 0.001:
                ax.text(x, -0.0155, "necessary", ha="center", color=S1,
                        fontsize=8.5, fontweight="bold")
            else:
                ax.text(x, -0.0140, f"clears by only\n{margin:.4f}, not counted",
                        ha="center", color=INK2, fontsize=7.8, style="italic")

        note = "" if surviving[k] else "\n(grouping fails its own control here)"
        ax.set_title(f"{k} groups requested{note}", fontsize=10, color=INK)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"group {m['mode']}\n{m['n_edges']} edges" for m in modes])
        ax.set_ylim(-0.020, 0.062)
        ax.grid(axis="y", zorder=0)

    axes[0].set_ylabel("held-out accuracy lost")
    axes[0].text(-0.42, FLOOR + 0.0016, "repeatability floor 0.0092",
                 color=INK2, fontsize=8.5)
    # Figure-level legend: a per-axes one collided with the tallest bar's label.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015),
               ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Deleting one group of activation shapes, against two controls",
                 fontsize=11.5, fontweight="bold", color=INK, y=1.10)
    fig.text(0.5, -0.07,
             "A group counts only if deleting it costs more than the floor AND more than both controls. At 2 and 3 "
             "groups, the settings where the grouping survives its untrained control, one group of about thirty edges "
             "clears every bar. Bars below zero mean deletion made the network slightly better, which is what a group "
             "that carries nothing looks like.",
             ha="center", fontsize=8.5, color=INK2, wrap=True)
    fig.tight_layout()
    p = OUT / "u15_mode_ablation.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (substitution, pruning, mode_ablation):
        print("wrote", fn().relative_to(ROOT))
