#!/usr/bin/env python3
"""Visualize the Phase 3 TKAN 5-fold sweep from the GL training logs.

Reads the per-fold JSON summary embedded in each `phase3_train_gpu_*.log`
(header lines, then a pretty-printed JSON block, then a `Finished:` line) and
renders a one-page dashboard + a few standalone figures under
results/phase3/sweep_figs/.

Run: python scripts/plot_phase3_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "GLC Logs"
OUT = ROOT / "results/phase3/sweep_figs"
OUT.mkdir(parents=True, exist_ok=True)

MLP_GATE = 0.87   # cited MLP baseline AUC
PASS_GATE = 0.88  # the >=0.88 pass gate

FOLD_LOGS = {
    0: "phase3_train_gpu_51670644_0.log",
    1: "phase3_train_gpu_51711709_1.log",
    2: "phase3_train_gpu_51711709_2.log",
    3: "phase3_train_gpu_51711709_3.log",
    4: "phase3_train_gpu_51711709_4.log",
}
COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd"}


def load_summary(path: Path) -> dict:
    txt = path.read_text()
    blob = txt[txt.index("{"): txt.rindex("}") + 1]
    return json.loads(blob)


def main() -> None:
    S = {k: load_summary(LOGDIR / v) for k, v in FOLD_LOGS.items()}
    folds = sorted(S)

    epochs = {k: [v["auc"] for v in S[k]["history"]["val"]] for k in folds}
    rmse_h = {k: [v["rmse"] for v in S[k]["history"]["val"]] for k in folds}
    mae_h = {k: [v["mae"] for v in S[k]["history"]["val"]] for k in folds}
    loss_h = {k: S[k]["history"]["train_loss"] for k in folds}

    best_auc = np.array([S[k]["best_auc_val"]["auc"] for k in folds])
    best_auc_ep = [S[k]["best_auc_epoch"] for k in folds]
    final_mae = np.array([S[k]["val"]["mae"] for k in folds])
    final_rmse = np.array([S[k]["val"]["rmse"] for k in folds])

    mean_auc, std_auc = best_auc.mean(), best_auc.std()

    # ---------------- dashboard ----------------
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Phase 3 TKAN — 5-fold sweep (lr 0.0015, side-input, compile)\n"
        f"mean best-AUC = {mean_auc:.3f} ± {std_auc:.3f}  |  0 skipped batches / 100 epoch-folds",
        fontsize=14, fontweight="bold",
    )

    # (0,0) val AUC vs epoch
    a = ax[0, 0]
    for k in folds:
        x = range(1, len(epochs[k]) + 1)
        a.plot(x, epochs[k], color=COLORS[k], marker="o", ms=3, lw=1.6, label=f"fold {k}")
        a.scatter([best_auc_ep[k] + 1], [best_auc[k]], color=COLORS[k], s=90,
                  marker="*", zorder=5, edgecolor="k", linewidth=0.5)
    a.axhline(MLP_GATE, color="gray", ls="--", lw=1.2, label=f"MLP ~{MLP_GATE}")
    a.axhline(PASS_GATE, color="k", ls=":", lw=1.0, label=f"gate {PASS_GATE}")
    a.set(title="Classification val AUC vs epoch (★ = best_auc.pt)",
          xlabel="epoch", ylabel="val AUC", ylim=(0.80, 1.0))
    a.legend(fontsize=8, ncol=2); a.grid(alpha=0.3)

    # (0,1) train loss vs epoch (log)
    a = ax[0, 1]
    for k in folds:
        a.plot(range(1, len(loss_h[k]) + 1), loss_h[k], color=COLORS[k], marker="o", ms=3, lw=1.6, label=f"fold {k}")
    a.set_yscale("log")
    a.set(title="Train loss vs epoch (clean convergence, no blow-up)",
          xlabel="epoch", ylabel="train loss (log)")
    a.legend(fontsize=8, ncol=2); a.grid(alpha=0.3, which="both")

    # (1,0) per-fold best AUC bars
    a = ax[1, 0]
    bars = a.bar([f"f{k}" for k in folds], best_auc, color=[COLORS[k] for k in folds], alpha=0.85)
    a.axhline(mean_auc, color="k", lw=1.5, label=f"mean {mean_auc:.3f}")
    a.axhspan(mean_auc - std_auc, mean_auc + std_auc, color="k", alpha=0.08)
    a.axhline(MLP_GATE, color="gray", ls="--", lw=1.3, label=f"MLP ~{MLP_GATE}")
    for b, v in zip(bars, best_auc):
        a.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    a.set(title="Best-AUC per fold vs MLP baseline", ylabel="best val AUC", ylim=(0.80, 1.0))
    a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")

    # (1,1) severity MAE / RMSE per fold (the heavy-tailed head)
    a = ax[1, 1]
    x = np.arange(len(folds)); w = 0.38
    a.bar(x - w / 2, final_mae, w, label="MAE", color="#4c72b0")
    a.bar(x + w / 2, final_rmse, w, label="RMSE", color="#c44e52")
    for i, (m, r) in enumerate(zip(final_mae, final_rmse)):
        a.text(i - w / 2, m + 0.1, f"{m:.2f}", ha="center", fontsize=8)
        a.text(i + w / 2, r + 0.1, f"{r:.2f}", ha="center", fontsize=8)
    a.set(title="Severity head: MAE vs RMSE per fold (RMSE≫MAE = heavy tail; fold 4 = outliers)",
          xticks=x, xlabel="fold", ylabel="error (b/blim units)")
    a.set_xticklabels([f"f{k}" for k in folds])
    a.legend(fontsize=9); a.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT / "sweep_dashboard.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)

    # ---------------- standalone: two-head tension ----------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Two-head dynamics: classification peaks & holds while severity RMSE drifts up", fontsize=12)
    for k in folds:
        xa = range(1, len(epochs[k]) + 1)
        ax[0].plot(xa, epochs[k], color=COLORS[k], lw=1.5, label=f"fold {k}")
        ax[1].plot(range(1, len(rmse_h[k]) + 1), rmse_h[k], color=COLORS[k], lw=1.5, label=f"fold {k}")
    ax[0].axhline(MLP_GATE, color="gray", ls="--", lw=1.0)
    ax[0].set(title="val AUC vs epoch", xlabel="epoch", ylabel="AUC"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].set(title="val severity RMSE vs epoch", xlabel="epoch", ylabel="RMSE"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p2 = OUT / "two_head_dynamics.png"
    fig.savefig(p2, dpi=130); plt.close(fig)
    print("wrote", p2)

    # ---------------- text summary ----------------
    lines = ["fold,best_auc,best_auc_epoch,final_auc,final_acc,final_mae,final_rmse,val_files,max_skipped_batches"]
    for k in folds:
        nb = max(S[k]["history"]["nonfinite_batches"])
        lines.append(
            f"{k},{S[k]['best_auc_val']['auc']:.4f},{S[k]['best_auc_epoch']},"
            f"{S[k]['val']['auc']:.4f},{S[k]['val']['acc']:.4f},"
            f"{S[k]['val']['mae']:.4f},{S[k]['val']['rmse']:.4f},"
            f"{S[k]['val']['n_files']},{nb}"
        )
    lines.append(f"# mean best_auc {mean_auc:.4f} +/- {std_auc:.4f}; "
                 f"final_auc {np.mean([S[k]['val']['auc'] for k in folds]):.4f}")
    (OUT / "sweep_summary.csv").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "sweep_summary.csv")


if __name__ == "__main__":
    main()
