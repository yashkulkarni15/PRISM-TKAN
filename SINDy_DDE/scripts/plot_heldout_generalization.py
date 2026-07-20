#!/usr/bin/env python
"""Plot held-out generalization of the Phase-3 TKAN on datasets 121-155.

Reads results/phase3/heldout/<run>_<set>.json for 5 runs (folds) x 2 eval sets,
assembles the per-fold table from the JSON metrics, and renders a two-panel
publication-quality figure:

  LEFT  : per-fold held-out classification AUC for the two eval sets, with
          in-distribution (0.955) and MLP-baseline (~0.87) reference lines.
  RIGHT : severity-head transfer failure -- per-fold MAE vs RMSE for set
          121-125, showing RMSE >> MAE (heavy tail).

No metric values are hardcoded; everything is read from the JSON files.
Re-runnable: just execute the script.
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- paths --------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HELDOUT_DIR = os.path.join(ROOT, "results", "phase3", "heldout")
OUT_DIR = os.path.join(ROOT, "results", "figures", "research_update_assets")
OUT_PNG = os.path.join(OUT_DIR, "heldout_generalization_121_155.png")

# Map short run hash -> fold index. (Source of truth for fold ordering.)
RUN_TO_FOLD = {
    "61c67bdd": 0,
    "c21e58dc": 1,
    "a654129d": 2,
    "3a40b715": 3,
    "09378839": 4,
}

# Eval-set keys -> human label.
SET_LABELS = {
    "121_125": "121-125 (all new noisy)",
    "122_125": "122-125 (OOD novel)",
}

# In-distribution / baseline reference values (context, not part of the held-out data).
IN_DIST_AUC = 0.955   # in-distribution 5-fold sweep mean best_auc
MLP_BASELINE_AUC = 0.87  # MLP side-input baseline
IN_DIST_SEV_MAE = 0.88   # in-distribution severity-head MAE, for contrast

# Filenames look like "61c67bdd_121_125.json". We require an exactly-8-char hex
# run id so the longer full-hash duplicate file is ignored.
FNAME_RE = re.compile(r"^([0-9a-f]{8})_(121_125|122_125)\.json$")


def load_table():
    """Return {set_key: {fold: metrics_dict}} read from the heldout JSONs."""
    table = {k: {} for k in SET_LABELS}
    for path in sorted(glob.glob(os.path.join(HELDOUT_DIR, "*.json"))):
        m = FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        run, set_key = m.group(1), m.group(2)
        if run not in RUN_TO_FOLD:
            continue
        with open(path) as fh:
            data = json.load(fh)
        table[set_key][RUN_TO_FOLD[run]] = data["metrics"]
    return table


def main():
    table = load_table()
    folds = sorted(RUN_TO_FOLD.values())

    # Assemble arrays in fold order.
    auc = {
        s: np.array([table[s][f]["auc"] for f in folds]) for s in SET_LABELS
    }
    mae_121 = np.array([table["121_125"][f]["mae"] for f in folds])
    rmse_121 = np.array([table["121_125"][f]["rmse"] for f in folds])

    mean_auc = {s: float(np.mean(auc[s])) for s in SET_LABELS}
    std_auc = {s: float(np.std(auc[s], ddof=1)) for s in SET_LABELS}

    # --- print the assembled per-fold table -----------------------------
    print("Per-fold held-out table (read from JSON):")
    print(f"{'fold':>4} {'AUC 121-125':>12} {'AUC 122-125':>12} "
          f"{'MAE 121-125':>12} {'RMSE 121-125':>12}")
    for i, f in enumerate(folds):
        print(f"{f:>4} {auc['121_125'][i]:>12.4f} {auc['122_125'][i]:>12.4f} "
              f"{mae_121[i]:>12.3f} {rmse_121[i]:>12.3f}")
    for s in SET_LABELS:
        print(f"  mean AUC {s}: {mean_auc[s]:.3f} +/- {std_auc[s]:.3f}")

    # --- style ----------------------------------------------------------
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })

    # Muted palette.
    C_ALL = "#4C72B0"   # in-set noisy (121-125)
    C_OOD = "#DD8452"   # OOD novel (122-125)
    C_MAE = "#55A868"   # MAE
    C_RMSE = "#C44E52"  # RMSE
    C_REF_IND = "#555555"
    C_REF_MLP = "#999999"

    # Letter page text width with 1-in margins is ~6.5 in.
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(6.5, 3.4), constrained_layout=True
    )

    x = np.arange(len(folds))

    # ---------------- LEFT: held-out AUC -------------------------------
    bw = 0.38
    axL.bar(x - bw / 2, auc["121_125"], bw, color=C_ALL,
            label=SET_LABELS["121_125"], edgecolor="white", linewidth=0.5)
    axL.bar(x + bw / 2, auc["122_125"], bw, color=C_OOD,
            label=SET_LABELS["122_125"], edgecolor="white", linewidth=0.5)

    axL.axhline(IN_DIST_AUC, ls="--", lw=1.1, color=C_REF_IND,
                label=f"in-dist 5-fold mean = {IN_DIST_AUC:.3f}")
    axL.axhline(MLP_BASELINE_AUC, ls=":", lw=1.1, color=C_REF_MLP,
                label=f"MLP baseline ~ {MLP_BASELINE_AUC:.2f}")

    # Mean annotations for each held-out set.
    # Held-out means, in a small box in the clear space above the bars.
    axL.text(0.025, 0.965, f"mean {mean_auc['121_125']:.3f}",
             transform=axL.transAxes, ha="left", va="top", fontsize=7.5,
             color=C_ALL, fontweight="bold")
    axL.text(0.025, 0.885, f"mean {mean_auc['122_125']:.3f}",
             transform=axL.transAxes, ha="left", va="top", fontsize=7.5,
             color=C_OOD, fontweight="bold")

    axL.set_ylim(0.84, 0.98)
    axL.set_xticks(x)
    axL.set_xticklabels([f"f{f}" for f in folds])
    axL.set_xlabel("fold")
    axL.set_ylabel("held-out AUC")
    axL.set_title("Classification transfers", fontsize=9.0)
    axL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
               frameon=False, columnspacing=1.0, handlelength=1.4,
               borderpad=0.2)

    # ---------------- RIGHT: severity MAE vs RMSE ----------------------
    axR.bar(x - bw / 2, mae_121, bw, color=C_MAE, label="MAE",
            edgecolor="white", linewidth=0.5)
    axR.bar(x + bw / 2, rmse_121, bw, color=C_RMSE, label="RMSE",
            edgecolor="white", linewidth=0.5)

    for i in range(len(folds)):
        axR.text(x[i] - bw / 2, mae_121[i] + 0.1, f"{mae_121[i]:.2f}",
                 ha="center", va="bottom", fontsize=6.5, color=C_MAE)
        axR.text(x[i] + bw / 2, rmse_121[i] + 0.1, f"{rmse_121[i]:.2f}",
                 ha="center", va="bottom", fontsize=6.5, color=C_RMSE)

    axR.axhline(IN_DIST_SEV_MAE, ls="--", lw=1.1, color=C_REF_IND,
                label=f"in-dist severity MAE ~ {IN_DIST_SEV_MAE:.2f}")

    axR.set_ylim(0, max(rmse_121) * 1.18)
    axR.set_xticks(x)
    axR.set_xticklabels([f"f{f}" for f in folds])
    axR.set_xlabel("fold")
    axR.set_ylabel("severity error (units)")
    axR.set_title("Severity head: RMSE $\\gg$ MAE (heavy tail)", fontsize=9.0)
    axR.legend(loc="upper left", frameon=False, handlelength=1.4,
               borderpad=0.2)

    fig.suptitle(
        "Held-out generalization on datasets 121-155 "
        "(eval-only, no retraining)",
        fontsize=10.5, fontweight="bold",
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    size = os.path.getsize(OUT_PNG)
    print(f"\nWrote {OUT_PNG} ({size} bytes)")
    assert size > 0, "output PNG is empty"


if __name__ == "__main__":
    main()
