#!/usr/bin/env python
"""Plot the cross-fold white-box delay-ablation replication (Phase 3 TKAN XAI).

Reads every fold's results/phase3/xai/<TAG>/whitebox_manifest.json (the
512-window replication runs: fold0, fold1, fold2_w512, fold3, fold4) and
renders ONE two-panel figure:

  A : chatter AUC with the tau-delayed input intact vs zeroed, per fold
      (paired bars, chance = 0.5 reference line).
  B : severity MAE with the delayed input intact vs zeroed, per fold
      (paired bars, log scale).

Folds whose manifest is missing (job still pending) are skipped and named in
the console output. No metric values are hardcoded. Re-runnable: just execute
the script again once the pending fold's artifacts land.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
XAI = os.path.join(ROOT, "results", "phase3", "xai")
OUT_PNG = os.path.join(
    ROOT, "results", "figures", "research_update_assets",
    "whitebox_delay_replication.png",
)

# tag -> display label; fold2_w512 is the uniform 512-window re-run of fold 2.
FOLD_TAGS = [
    ("fold0", "fold 0"),
    ("fold1", "fold 1"),
    ("fold2_w512", "fold 2"),
    ("fold3", "fold 3"),
    ("fold4", "fold 4"),
]


def main():
    rows, missing = [], []
    for tag, label in FOLD_TAGS:
        path = os.path.join(XAI, tag, "whitebox_manifest.json")
        if not os.path.exists(path):
            missing.append(label)
            continue
        with open(path) as fh:
            man = json.load(fh)
        abl = man["results"]["delay_ablation"]
        rows.append({
            "label": label,
            "val_auc": man["provenance"].get("val_auc"),
            "auc_on": abl["auc_on"],
            "auc_off": abl["auc_off"],
            "mae_on": abl["mae_on"],
            "mae_off": abl["mae_off"],
        })
        print(f"[{label}] AUC {abl['auc_on']:.4f} -> {abl['auc_off']:.4f} "
              f"(drop {abl['auc_drop']:.4f}); "
              f"MAE {abl['mae_on']:.3f} -> {abl['mae_off']:.2f}")
    if missing:
        print("pending (no manifest, skipped): " + ", ".join(missing))
    assert rows, "no whitebox manifests found"

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })
    C_ON, C_OFF, C_REF = "#4C72B0", "#C44E52", "#555555"

    x = np.arange(len(rows))
    bw = 0.38
    labels = [r["label"] for r in rows]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 3.4))

    # ---------------- A: chatter AUC, delay on vs zeroed ------------------
    on = [r["auc_on"] for r in rows]
    off = [r["auc_off"] for r in rows]
    axA.bar(x - bw / 2, on, bw, color=C_ON, edgecolor="white",
            linewidth=0.5, label="delayed input intact")
    axA.bar(x + bw / 2, off, bw, color=C_OFF, edgecolor="white",
            linewidth=0.5, label="delayed input zeroed")
    for xi, v in zip(x - bw / 2, on):
        axA.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7)
    for xi, v in zip(x + bw / 2, off):
        axA.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7)
    axA.axhline(0.5, ls="--", lw=1.1, color=C_REF, label="chance (AUC 0.5)")
    axA.set_xticks(x)
    axA.set_xticklabels(labels)
    axA.set_ylim(0.0, 1.14)
    axA.set_ylabel("chatter AUC (sampled windows)")
    axA.set_title("A  Chatter detection")
    axA.legend(loc="lower left", frameon=True, framealpha=0.95,
               edgecolor="none", facecolor="white", handlelength=1.6,
               borderpad=0.3)

    # ---------------- B: severity MAE, delay on vs zeroed -----------------
    mon = [r["mae_on"] for r in rows]
    moff = [r["mae_off"] for r in rows]
    axB.bar(x - bw / 2, mon, bw, color=C_ON, edgecolor="white",
            linewidth=0.5, label="delayed input intact")
    axB.bar(x + bw / 2, moff, bw, color=C_OFF, edgecolor="white",
            linewidth=0.5, label="delayed input zeroed")
    for xi, v in zip(x - bw / 2, mon):
        axB.text(xi, v * 1.2, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=7)
    for xi, v in zip(x + bw / 2, moff):
        axB.text(xi, v * 1.2, f"{v:.3g}", ha="center", va="bottom",
                 fontsize=7)
    axB.set_yscale("log")
    axB.set_ylim(min(mon) / 4, max(moff) * 12)
    axB.set_xticks(x)
    axB.set_xticklabels(labels)
    axB.set_ylabel("severity MAE (log)")
    axB.set_title("B  Severity regression")
    axB.legend(loc="upper left", frameon=True, framealpha=0.95,
               edgecolor="none", facecolor="white", handlelength=1.6,
               borderpad=0.3)

    n = len(rows)
    fig.suptitle(
        f"Delay ablation replicated across {n} folds\n"
        "(512-window samples; zeroing the delayed input breaks both heads)",
        fontsize=10, fontweight="bold",
    )
    fig.subplots_adjust(top=0.82)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    size = os.path.getsize(OUT_PNG)
    print(f"\nWrote {OUT_PNG} ({size} bytes)")
    assert size > 0, "output PNG is empty"


if __name__ == "__main__":
    main()
