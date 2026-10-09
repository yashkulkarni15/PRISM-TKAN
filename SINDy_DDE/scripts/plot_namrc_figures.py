import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
# ONLY pool4/ manifests: the ones one dir up were made with the pool=1 loader bug.
ABL = "results/phase3/xai/mode_ablation/pool4/mode_ablation_{}_k3_pairs.json"
TKD = "results/phase3/xai/tkan_delay_heldout/pool4/tkan_delay_heldout_{}.json"
MR = "results/phase3/minirocket/minirocket_fold{}.json"
folds = [("fold0","Fold 0"),("fold1","Fold 1"),("fold2_w512","Fold 2"),("fold3","Fold 3"),("fold4","Fold 4")]
FLOOR = 0.0092
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # dataviz categorical slots 1-3
PALE, GRAY, INK, MUTED = "#c9dcf3", "#b4b2ab", "#0b0b0b", "#52514e"
OUT = "results/figures/namrc"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK, "font.family": "sans-serif"})
def save(fig, name):
    for ext in ("png", "pdf"): fig.savefig(f"{OUT}/{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}/{name}.png/.pdf")
def tidy(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.axhline(FLOOR, color=MUTED, ls="--", lw=0.8, zorder=1)

# ---------- Figure 1: deletion (Exp 1 singles, Exp 1b pairs, all 77) ----------
fig, axs = plt.subplots(1, 5, figsize=(7.0, 2.5), sharey=True)
W = 0.38
print("FIG 1  drop (random-matched control) [* = claimable]")
for ax, (tag, name) in zip(axs, folds):
    d = json.load(open(ABL.format(tag)))
    singles = sorted(d["modes"], key=lambda m: -m["verdict"]["mode_drop"])
    pairs = sorted(d["pairs"], key=lambda m: -m["verdict"]["pair_drop"])
    a = d["all_gated"][0]
    rows = [(m["n_edges"], m["verdict"]["mode_drop"], m["verdict"]["random_matched_drop"], m["verdict"]["claimable"]) for m in singles] + \
           [(m["n_edges"], m["verdict"]["pair_drop"], m["verdict"]["random_matched_drop"], m["verdict"]["claimable"]) for m in pairs] + \
           [(77, a["verdict"]["all_drop"], a["verdict"]["random_matched_drop"], a["verdict"]["claimable"])]
    assert a["verdict"]["claimable"] and any(m["verdict"]["claimable"] for m in singles), tag
    xs = [0, 1, 2, 3.5, 4.5, 5.5, 7]
    for x, (n, drop, ctl, ok) in zip(xs, rows):
        ax.bar(x - W/2, drop, W, color=BLUE if ok else PALE, edgecolor="white", linewidth=0.5, zorder=2)
        ax.bar(x + W/2, ctl, W, color=GRAY, edgecolor="white", linewidth=0.5, zorder=2)
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=2)
    for x, lab in ((1, "single"), (4.5, "pair"), (7, "all")):
        ax.text(x, -0.024, lab, ha="center", va="top", fontsize=7, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_title(f"{name}  (AUC {d['baseline_auc']:.3f})", fontsize=8, color=INK)
    tidy(ax); ax.set_xlim(-0.6, 7.6)
    print(f"  {name} base {d['baseline_auc']:.4f}: " + "  ".join(
        f"{n}e {dr:.4f}({c:.4f}){'*' if ok else ''}" for n, dr, c, ok in rows))
axs[0].set_ylabel("Held-out AUC drop")
axs[0].set_ylim(-0.01, 0.135)
fig.legend(handles=[Patch(color=BLUE, label="Deleted cluster(s), claimable"),
                    Patch(color=PALE, label="Deleted cluster(s), not claimable"),
                    Patch(color=GRAY, label="Random matched deletion"),
                    Line2D([], [], color=MUTED, ls="--", lw=0.8, label=f"Repeatability floor ({FLOOR})")],
           loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False, fontsize=7)
plt.tight_layout(w_pad=0.6)
fig.text(0.5, -0.04, "Delay edges deleted (singles and pairs ordered by drop within each fold)", ha="center", fontsize=8)
save(fig, "fig1_deletion")

# ---------- Figure 2: TKAN vs MiniROCKET ----------
tk = [json.load(open(TKD.format(t))) for t, _ in folds]
mr = [json.load(open(MR.format(i)))["metrics"]["heldout"] for i in range(5)]
auc = {"TKAN": [t["baseline_auc"] for t in tk], "MiniROCKET-A": [m["A"]["auc"] for m in mr],
       "MiniROCKET-B": [m["B"]["auc"] for m in mr]}
V = [("zero_delay", "zeroed"), ("tau_x0.5", "τ × 0.5"), ("tau_x2", "τ × 2")]
drop = {"TKAN": {v: [t["variants"][v]["drop"] for t in tk] for v, _ in V},
        "MiniROCKET-B": {v: [m["B"]["auc"] - m["B_" + v]["auc"] for m in mr] for v, _ in V}}
EXP_AUC = {"TKAN": 0.960, "MiniROCKET-A": 0.962, "MiniROCKET-B": 0.967}
EXP_DROP = {"TKAN": (0.211, 0.118, 0.015), "MiniROCKET-B": (0.244, 0.071, 0.004)}
print("FIG 2a  held-out file-level AUC (folds 0-4) mean±sd")
for k, v in auc.items():
    print(f"  {k:13s} " + " ".join(f"{x:.4f}" for x in v) + f"  {np.mean(v):.3f}±{np.std(v, ddof=1):.3f}")
    assert round(np.mean(v), 3) == EXP_AUC[k], (k, np.mean(v))
print("FIG 2b  AUC drop under delay corruption (folds 0-4) mean±sd")
for k in drop:
    for (v, lab), e in zip(V, EXP_DROP[k]):
        x = drop[k][v]
        print(f"  {k:13s} {lab:8s} " + " ".join(f"{y:.4f}" for y in x) + f"  {np.mean(x):.3f}±{np.std(x, ddof=1):.3f}")
        assert round(np.mean(x), 3) == e, (k, v, np.mean(x))

COL = {"TKAN": BLUE, "MiniROCKET-A": ORANGE, "MiniROCKET-B": AQUA}
MK = {"TKAN": "o", "MiniROCKET-A": "s", "MiniROCKET-B": "D"}
LAB = {"TKAN": "TKAN", "MiniROCKET-A": "MiniROCKET A (4 ch)", "MiniROCKET-B": "MiniROCKET B (4 + 4 delayed)"}
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
for j, k in enumerate(auc):
    off = (j - 1) * 0.22
    a.scatter(np.arange(5) + off, auc[k], s=22, marker=MK[k], color=COL[k], edgecolor="white", lw=0.6, zorder=3, label=LAB[k])
    a.errorbar(5.4 + off, np.mean(auc[k]), np.std(auc[k], ddof=1), fmt=MK[k], ms=5, color=COL[k], mec="white", mew=0.6, capsize=2, lw=1, zorder=3)
a.axvline(4.7, color=GRAY, lw=0.6)
a.set_xticks(list(range(5)) + [5.4]); a.set_xticklabels([n for _, n in folds] + ["Mean ± sd"], fontsize=7)
a.tick_params(axis="x", length=0)
a.set_ylim(0.94, 0.985); a.set_ylabel("Held-out file-level AUC")
a.grid(axis="y", color="#e6e5e0", lw=0.5); a.set_axisbelow(True)
for s in ("top", "right"): a.spines[s].set_visible(False)
a.legend(loc="upper left", frameon=False, fontsize=6.5, ncol=1, handletextpad=0.2, borderaxespad=0.1)
a.set_title("(a) Held-out datasets 121–125", fontsize=8, loc="left", color=INK)
rng = np.random.default_rng(0)
for j, k in enumerate(drop):
    for i, (v, _) in enumerate(V):
        x = i + (j - 0.5) * 0.36; y = drop[k][v]
        b.bar(x, np.mean(y), 0.32, color=COL[k], alpha=0.35, zorder=2, label=LAB[k] if i == 0 else None)
        b.errorbar(x, np.mean(y), np.std(y, ddof=1), color=COL[k], capsize=2, lw=1, zorder=3)
        b.scatter(x + rng.uniform(-0.07, 0.07, 5), y, s=10, color=COL[k], edgecolor="white", lw=0.4, zorder=4)
        b.text(x, max(y) + 0.012, f"{np.mean(y):.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
b.set_xticks(range(3)); b.set_xticklabels([f"{l}" for _, l in V])
b.set_xlabel("Test-time corruption of the delayed channel", fontsize=7.5)
b.tick_params(axis="x", length=0); b.axhline(0, color=MUTED, lw=0.6)
b.set_ylabel("Held-out AUC drop"); tidy(b)
h, l = b.get_legend_handles_labels()
b.legend(h + [Line2D([], [], color=MUTED, ls="--", lw=0.8)], l + [f"Floor ({FLOOR})"], loc="upper right", frameon=False, fontsize=6.5)
b.set_ylim(-0.01, 0.46)
b.set_title("(b) Delay-damage test", fontsize=8, loc="left", color=INK)
plt.tight_layout(w_pad=1.5)
save(fig, "fig2_tkan_vs_minirocket")
