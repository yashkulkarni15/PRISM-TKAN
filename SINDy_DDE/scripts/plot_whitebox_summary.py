#!/usr/bin/env python
"""Plot the fold-2 white-box delay-intervention summary (Phase 3 TKAN XAI).

Reads results/phase3/xai/fold2/whitebox_manifest.json (or --manifest) and
renders ONE three-panel figure:

  A : delay ablation -- chatter AUC with the tau-delayed input on vs zeroed
      (paired bars, left axis) and severity MAE on vs off (paired bars,
      twin log axis). This is an *intervention* on the model input.
  B : tau sweep -- chatter AUC vs imposed delay (samples), with the
      no-delay (tau -> inf) run drawn as a dashed horizontal reference and
      the physical tau marked with a vertical line.
  C : pruning faithfulness -- chatter AUC vs fraction of KAN edges kept,
      x log scale.

No metric values are hardcoded; everything is read from the manifest JSON.
Re-runnable: just execute the script.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- paths --------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MANIFEST = os.path.join(
    ROOT, "results", "phase3", "xai", "fold2", "whitebox_manifest.json"
)
OUT_DIR = os.path.join(ROOT, "results", "figures", "research_update_assets")
OUT_PNG = os.path.join(OUT_DIR, "whitebox_delay_intervention_fold2.png")

# Any imposed tau at/above this many samples is the "no delay" control run
# (the harness encodes tau -> inf as a sentinel like 1e6). Structural
# threshold only -- not a metric value.
INF_TAU_SENTINEL = 1e5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="path to whitebox_manifest.json")
    ap.add_argument("--out", default=OUT_PNG, help="output PNG path")
    args = ap.parse_args()

    with open(args.manifest) as fh:
        man = json.load(fh)

    res = man["results"]
    prov = man["provenance"]
    tag = prov.get("tag", "fold?")
    tau_physical = prov["tau_samples"]

    # --- delay ablation ---------------------------------------------------
    abl = res["delay_ablation"]
    auc_on, auc_off = abl["auc_on"], abl["auc_off"]
    mae_on, mae_off = abl["mae_on"], abl["mae_off"]

    # --- tau sweep ----------------------------------------------------------
    sweep = res["tau_sweep"]
    taus = np.asarray(sweep["taus"], dtype=float)
    sweep_auc = np.asarray(sweep["auc"], dtype=float)
    finite = taus < INF_TAU_SENTINEL
    taus_fin, auc_fin = taus[finite], sweep_auc[finite]
    auc_nodelay = float(sweep_auc[~finite][0]) if (~finite).any() else auc_off

    # --- pruning faithfulness ------------------------------------------------
    faith = res["faithfulness"]
    frac_kept = np.asarray(faith["prune_fraction_kept"], dtype=float)
    prune_auc = np.asarray(faith["prune_auc"], dtype=float)

    # --- print the assembled numbers (sanity trace) -----------------------
    print(f"[{tag}] delay ablation: AUC {auc_on:.4f} -> {auc_off:.4f}, "
          f"MAE {mae_on:.4f} -> {mae_off:.4f}")
    print(f"[{tag}] tau sweep: " + ", ".join(
        f"tau={int(t)}: {a:.4f}" for t, a in zip(taus_fin, auc_fin))
        + f"; no-delay: {auc_nodelay:.4f}; physical tau={tau_physical}")
    print(f"[{tag}] pruning: " + ", ".join(
        f"{f:g}: {a:.4f}" for f, a in zip(frac_kept, prune_auc)))

    # --- style ----------------------------------------------------------
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

    # Muted palette.
    C_ON = "#4C72B0"     # delayed input intact
    C_OFF = "#C44E52"    # delayed input zeroed
    C_MAE = "#55A868"    # severity MAE
    C_SWEEP = "#4C72B0"  # tau-sweep curve
    C_PRUNE = "#8172B3"  # pruning curve
    C_REF = "#555555"

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(9.6, 3.3))

    # ---------------- A: delay ablation (intervention) -------------------
    bw = 0.7
    x_auc = np.array([0.0, 1.0])
    axA.bar(x_auc, [auc_on, auc_off], bw,
            color=[C_ON, C_OFF], edgecolor="white", linewidth=0.5)
    for xi, v in zip(x_auc, [auc_on, auc_off]):
        axA.text(xi, v + 0.015, f"{v:.4f}", ha="center", va="bottom",
                 fontsize=7.5, fontweight="bold")
    axA.set_ylim(0.0, 1.12)
    axA.set_ylabel("chatter AUC")

    # Severity MAE on a twin log axis (11.57 vs 0.17 spans ~2 decades).
    axA2 = axA.twinx()
    axA2.grid(False)
    x_mae = np.array([2.4, 3.4])
    axA2.bar(x_mae, [mae_on, mae_off], bw,
             color=C_MAE, edgecolor="white", linewidth=0.5,
             hatch=["", "//"])
    axA2.set_yscale("log")
    for xi, v in zip(x_mae, [mae_on, mae_off]):
        axA2.text(xi, v * 1.25, f"{v:.3g}", ha="center", va="bottom",
                  fontsize=7.5, fontweight="bold", color=C_MAE)
    axA2.set_ylim(mae_on / 4, mae_off * 12)
    axA2.set_ylabel("severity MAE (log)", color=C_MAE)
    axA2.tick_params(axis="y", labelcolor=C_MAE)

    axA.set_xticks(np.concatenate([x_auc, x_mae]))
    axA.set_xticklabels(["delay\non", "delay\nzeroed",
                         "delay\non", "delay\nzeroed"])
    axA.set_title("A  Zeroing the delayed input")

    # ---------------- B: tau sweep ---------------------------------------
    axB.plot(taus_fin, auc_fin, "o-", color=C_SWEEP, lw=1.4, ms=5,
             label="imposed delay")
    for t, a in zip(taus_fin, auc_fin):
        axB.annotate(f"{a:.3f}", (t, a), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7.5)
    axB.axhline(auc_nodelay, ls="--", lw=1.1, color=C_REF,
                label=f"no delay (tau$\\to\\infty$) = {auc_nodelay:.3f}")
    axB.axvline(tau_physical, ls=":", lw=1.2, color=C_OFF,
                label=f"physical tau = {tau_physical} samples")
    axB.set_xscale("log", base=2)
    axB.set_xticks(taus_fin)
    axB.set_xticklabels([f"{int(t)}" for t in taus_fin])
    axB.minorticks_off()
    axB.set_ylim(min(auc_nodelay, auc_fin.min()) - 0.04,
                 auc_fin.max() + 0.05)
    axB.set_xlabel("imposed delay tau (samples)")
    axB.set_ylabel("chatter AUC")
    axB.set_title("B  Global tau sweep (one tau imposed on all windows)")
    axB.legend(loc="lower center", frameon=True, framealpha=0.95,
               edgecolor="none", facecolor="white", handlelength=1.6,
               borderpad=0.3)

    # ---------------- C: pruning / faithfulness --------------------------
    axC.plot(frac_kept, prune_auc, "o-", color=C_PRUNE, lw=1.4, ms=5)
    for f, a in zip(frac_kept, prune_auc):
        axC.annotate(f"{a:.3f}", (f, a), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7.5)
    axC.axhline(prune_auc[-1], ls="--", lw=1.1, color=C_REF,
                label=f"all edges kept = {prune_auc[-1]:.3f}")
    axC.set_xscale("log")
    axC.set_xticks(frac_kept)
    axC.set_xticklabels([f"{f:g}" for f in frac_kept])
    axC.minorticks_off()
    axC.set_ylim(prune_auc.min() - 0.04, prune_auc.max() + 0.05)
    axC.set_xlabel("fraction of KAN edges kept")
    axC.set_ylabel("chatter AUC")
    axC.set_title("C  Pruning by edge importance")
    axC.legend(loc="lower right", frameon=False, handlelength=1.6,
               borderpad=0.2)

    fig.suptitle(
        f"White-box delay interventions ({tag}, n={prov['n_windows']} "
        "windows): the regenerative (delayed) input is load-bearing",
        fontsize=10.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

    size = os.path.getsize(args.out)
    print(f"\nWrote {args.out} ({size} bytes)")
    assert size > 0, "output PNG is empty"


if __name__ == "__main__":
    main()
