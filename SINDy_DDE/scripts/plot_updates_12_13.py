#!/usr/bin/env python3
"""Figures for Research Updates 12 and 13.

Every value is read from a returned manifest -- nothing here is illustrative.
Palette is the validated 3-slot categorical set (blue/orange/aqua); aqua sits
below 3:1 on the light surface, so every series carries a direct label rather
than relying on the legend swatch alone.

    python scripts/plot_updates_12_13.py [--out results/figures/research_update_assets]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FOLDS = ["fold0", "fold1", "fold2_w512", "fold3", "fold4"]
LABEL = {"fold0": "fold 0", "fold1": "fold 1", "fold2_w512": "fold 2",
         "fold3": "fold 3", "fold4": "fold 4"}

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


def _grid(ax, axis="y"):
    ax.grid(True, axis=axis, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Figure 1 -- U12: the TSR metric was reporting its own search floor
# --------------------------------------------------------------------------- #
def fig_tsr(root: Path, out: Path) -> dict:
    per_fold, rows = {}, []
    for f in FOLDS:
        tsr = json.loads((root / f"results/phase3/xai/{f}/whitebox_manifest.json")
                         .read_text())["results"]["tsr"]
        pw = tsr["per_window"]
        rows += pw
        floor = sum(1 for r in pw if r["tr_peak_lag"] == 2)
        per_fold[f] = (floor, len(pw) - floor)

    res = [r for r in rows if r["tr_peak_lag"] != 2]
    exact = sum(1 for r in res if r["tr_peak_lag"] == r["tau_raw"])
    n, nf = len(rows), sum(v[0] for v in per_fold.values())

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ys = np.arange(len(FOLDS))
    fl = np.array([per_fold[f][0] for f in FOLDS], float)
    rs = np.array([per_fold[f][1] for f in FOLDS], float)
    axa.barh(ys, fl, color=S2, height=0.62, label="returned the floor (no answer)", zorder=3)
    axa.barh(ys, rs, left=fl + 2, color=S1, height=0.62, label="resolved a real peak", zorder=3)
    for y, a, b in zip(ys, fl, rs):
        axa.text(a / 2, y, f"{a:.0f}", ha="center", va="center", color="white",
                 fontsize=9, fontweight="bold", zorder=4)
        axa.text(a + b + 6, y, f"{b:.0f} resolved", va="center", color=INK2, fontsize=9)
    axa.set_yticks(ys, [LABEL[f] for f in FOLDS])
    axa.set_xlim(0, 152)
    axa.set_xlabel("windows (128 per fold)")
    axa.set_title(f"A. {nf} of {n} windows ({nf/n:.1%}) never resolved a peak")
    axa.legend(frameon=False, fontsize=9, loc="upper center",
               bbox_to_anchor=(0.5, -0.18), ncol=2)
    _grid(axa, "x")

    tau = np.array([r["tau_raw"] for r in res], float)
    lag = np.array([r["tr_peak_lag"] for r in res], float)
    lim = [1.5, max(tau.max(), lag.max()) * 1.35]
    axb.plot(lim, lim, color=INK2, lw=1.2, ls="--", zorder=2)
    # Axes fraction, not data coords: the axes are log, so linear placement
    # here lands the label on top of the title.
    axb.text(0.30, 0.36, "peak lag = $\\tau$", color=INK2, fontsize=9,
             rotation=34, transform=axb.transAxes)
    on = lag == tau
    axb.scatter(tau[~on], lag[~on], s=42, facecolor="none", edgecolor=S1,
                linewidth=1.6, zorder=3, label="resolved window")
    axb.scatter(tau[on], lag[on], s=64, color=S1, edgecolor=SURFACE, linewidth=1.4,
                zorder=4, label=f"exactly on $\\tau$ ({exact} of {len(res)})")
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlim(lim); axb.set_ylim(lim)
    axb.set_xlabel("that window's own delay $\\tau$ (samples)")
    axb.set_ylabel("TSR peak relevance lag")
    axb.set_title(f"B. Of the 90 that resolve, {exact} land exactly on $\\tau$")
    axb.legend(frameon=False, loc="upper left", fontsize=9)
    _grid(axb, "both")

    fig.tight_layout()
    p = out / "u12_tsr_floor_and_lock.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return {"path": p, "n": n, "floor": nf, "resolved": len(res), "exact": exact}


# --------------------------------------------------------------------------- #
# Figure 2 -- U13-A: how many recommendations survive machining uncertainty
# --------------------------------------------------------------------------- #
def fig_survival(root: Path, out: Path) -> dict:
    m = json.loads((root / "results/phase3/xai/counterfactuals/counterfactuals_manifest.json")
                   .read_text())
    blk = m["per_dataset"]["1"]
    sv = sorted(r["survival"] for r in blk["survival"]["per_case"]
                if r["survival"] is not None)
    sv = np.array(sv, float)
    thr = 0.85
    above = int((sv >= thr).sum())

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(1, len(sv) + 1)
    colors = [S1 if v >= thr else S2 for v in sv]
    ax.vlines(x, 0, sv, color=colors, lw=3.2, zorder=3)
    ax.scatter(x, sv, s=26, color=colors, edgecolor=SURFACE, linewidth=1.0, zorder=4)

    ax.axhline(thr, color=INK, lw=1.3, ls="--", zorder=5)
    mean = float(sv.mean())
    ax.axhline(mean, color=INK2, lw=1.0, ls=":", zorder=5)

    # Every annotation goes in the empty upper-left; anything at y~0 disappears
    # behind the bars themselves.
    ax.text(1, 1.005, f"{above} of {len(sv)} clear the target",
            color=S1, fontsize=10, fontweight="bold", va="top")
    ax.text(1, 0.955, f"{len(sv)-above} fall short", color=S2, fontsize=10,
            fontweight="bold", va="top")
    ax.text(1, thr + 0.018, "0.85 survival target", color=INK, fontsize=9,
            fontweight="bold")
    ax.text(1, mean + 0.018, f"mean {mean:.3f}", color=INK2, fontsize=9)

    ax.set_xlim(0, len(sv) + 1); ax.set_ylim(0, 1.06)
    ax.set_xlabel("counterfactual recommendation (42 chatter cases, sorted)")
    ax.set_ylabel("fraction of perturbed draws still stable")
    ax.set_title("Survival under 2% spindle jitter, 0.05 mm depth tolerance, 10% $b_{lim}$ uncertainty\n"
                 "512 Monte Carlo draws per case", loc="left")
    _grid(ax, "y")
    fig.tight_layout()
    p = out / "u13a_counterfactual_survival.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return {"path": p, "above": above, "n": len(sv), "mean": mean,
            "median": float(np.median(sv)), "min": float(sv.min())}


# --------------------------------------------------------------------------- #
# Figure 3 -- U13-B: two generators, two different levers, same destination
# --------------------------------------------------------------------------- #
def _axis_breakdown(blk: dict, cases: dict, dice: dict) -> dict:
    """Which lever each generator pulled, per case.

    Prefers the manifest's own block, which only exists for runs made after the
    2026-07-29 metric rework. The manifest fetched before that carries the
    superseded direction metric instead, so recompute from the per-case
    coordinates -- identical arithmetic to run_dice's ``_axis``.
    """
    have = blk.get("dice", {}).get("axis_breakdown")
    if have:
        return have

    def axis(wb, bb, wc, bc):
        mw = not np.isclose(wc, wb, rtol=1e-6, atol=1e-9)
        mb = not np.isclose(bc, bb, rtol=1e-6, atol=1e-9)
        return "both" if (mw and mb) else "omega" if mw else "b_mm" if mb else "none"

    empty = {"omega": 0, "b_mm": 0, "both": 0, "none": 0}
    axes = {"dice": dict(empty), "physics": dict(empty)}
    for f, d in dice.items():
        c = cases.get(f)
        if not d.get("found") or c is None or c.get("omega_cf") is None:
            continue
        axes["dice"][axis(d["omega_base"], d["b_base"], d["omega_cf"], d["b_cf"])] += 1
        axes["physics"][axis(c["omega_base"], c["b_base"], c["omega_cf"], c["b_cf"])] += 1
    return axes


def _outcome_agreement(blk: dict, cases: dict, dice: dict) -> dict:
    """Both land stable, or neither. Same fallback logic as _axis_breakdown."""
    d = blk.get("dice", {})
    if d.get("n_outcome_agrees") is not None:
        return {"agree": d["n_outcome_agrees"], "n": d["n_comparable"],
                "both_stable": d["n_both_stable"], "stable": d["n_physically_stable"]}
    agree = both = n = stable = 0
    for f, r in dice.items():
        c = cases.get(f)
        if not r.get("found") or c is None or c.get("omega_cf") is None:
            continue
        ds, ps = bool(r["physically_stable"]), bool(c["physics_stable_cf"])
        n += 1
        stable += int(ds)
        agree += int(ds == ps)
        both += int(ds and ps)
    return {"agree": agree, "n": n, "both_stable": both, "stable": stable}


def fig_strategies(root: Path, out: Path) -> dict:
    m = json.loads((root / "results/phase3/xai/counterfactuals/counterfactuals_manifest.json")
                   .read_text())
    blk = m["per_dataset"]["1"]
    cases = {c["file"]: c for c in blk["cases"]}
    dice = {r["file"]: r for r in blk["dice"]["per_case"]}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.6, 4.6),
                                   gridspec_kw={"width_ratios": [1.55, 1.0]})

    # The manifest does not persist the boundary, so rebuild it exactly as
    # counterfactuals_phase3.build_blim_fn does: dedupe on omega, sort, interpolate.
    import pandas as pd

    tab = pd.read_csv(root / "results/phase3/phase3_dataset.csv",
                      usecols=["dataset", "omega", "blim"])
    pts = (tab[tab.dataset == 1][["omega", "blim"]]
           .dropna().drop_duplicates("omega").sort_values("omega"))
    # Plotting all 84 moves as arrows on the (omega, b) plane is unreadable
    # spaghetti. The move ITSELF is the finding, so plot the deltas: the
    # physics search lands on the db = 0 line, DiCE below it.
    ph_x, ph_y, di_x, di_y = [], [], [], []
    for f, c in cases.items():
        if c.get("omega_cf") is None:
            continue
        ph_x.append(c["omega_cf"] - c["omega_base"])
        ph_y.append(c["b_cf"] - c["b_base"])
        d = dice.get(f)
        if d and d.get("found"):
            di_x.append(d["omega_cf"] - d["omega_base"])
            di_y.append(d["b_cf"] - d["b_base"])

    axa.axhline(0, color=INK2, lw=1.0, zorder=2)
    axa.axvline(0, color=INK2, lw=1.0, zorder=2)
    axa.scatter(ph_x, ph_y, s=70, color=S1, edgecolor=SURFACE, linewidth=1.4,
                zorder=5, label=f"physics-constrained ({len(ph_x)})")
    axa.scatter(di_x, di_y, s=70, color=S2, edgecolor=SURFACE, linewidth=1.4,
                alpha=0.9, zorder=4, label=f"DiCE ({len(di_x)})")
    axa.text(min(ph_x) * 0.95, 0.30, "changes speed only\n(depth untouched)",
             color=S1, fontsize=9, fontweight="bold", va="bottom")
    axa.text(min(di_x) * 0.9, min(di_y) * 0.72, "cuts depth", color=S2,
             fontsize=9, fontweight="bold")
    axa.set_xlabel("change in spindle speed  $\\Delta\\omega$ (rpm)")
    axa.set_ylabel("change in depth  $\\Delta b$ (mm)")
    axa.set_title("A. The move each generator recommends")
    axa.legend(frameon=False, fontsize=9, loc="lower left")
    _grid(axa, "both")

    order = ["omega", "b_mm", "both", "none"]
    names = ["speed\nonly", "depth\nonly", "both", "neither"]
    ax_ = _axis_breakdown(blk, cases, dice)
    ph = [ax_["physics"][k] for k in order]
    di = [ax_["dice"][k] for k in order]
    x = np.arange(len(order)); w = 0.38
    axb.bar(x - w/2 - 0.01, ph, w, color=S1, zorder=3, label="physics-constrained")
    axb.bar(x + w/2 + 0.01, di, w, color=S2, zorder=3, label="DiCE")
    for xi, v in zip(x - w/2 - 0.01, ph):
        if v: axb.text(xi, v + 0.7, str(v), ha="center", color=S1, fontsize=9, fontweight="bold")
    for xi, v in zip(x + w/2 + 0.01, di):
        if v: axb.text(xi, v + 0.7, str(v), ha="center", color=S2, fontsize=9, fontweight="bold")
    axb.set_xticks(x, names)
    axb.set_ylabel("cases (of 42)")
    axb.set_ylim(0, max(ph + di) * 1.22)
    axb.set_title("B. Which knob it turned")
    axb.legend(frameon=False, fontsize=9, loc="upper center")
    _grid(axb, "y")

    fig.tight_layout()
    p = out / "u13b_two_generators.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return {"path": p, **_outcome_agreement(blk, cases, dice)}


# --------------------------------------------------------------------------- #
# Figure 4 -- U13-C: the network learned the SUM, not the difference
# --------------------------------------------------------------------------- #
def fig_regenerative(root: Path, out: Path) -> dict:
    data = {f: json.loads((root / f"results/phase3/xai/regenerative/{f}/regenerative.json")
                          .read_text()) for f in FOLDS}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.6, 4.6),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    sig = np.array([r["score"] for d in data.values() for r in d["rows"]
                    if r.get("score") is not None and np.isfinite(r["score"])])
    ctl = np.array([r["score"] for d in data.values() for r in d["control_rows"]
                    if r.get("score") is not None and np.isfinite(r["score"])])
    bins = np.linspace(-1, 1, 41)
    axa.hist(sig, bins=bins, density=True, color=S1, alpha=0.85, zorder=3)
    axa.hist(ctl, bins=bins, density=True, histtype="step", color=S2, lw=2.0, zorder=4)

    axa.axvline(0, color=INK2, lw=1.0, zorder=2)
    for t, lab in ((-0.9, "copy"), (0.9, "negation")):
        axa.axvline(t, color=INK, lw=1.1, ls="--", zorder=5)
    top = axa.get_ylim()[1]
    axa.text(-0.86, top * 0.99, "$\\phi_{delay}\\approx+\\phi_{inst}$\nSUM",
             color=INK, fontsize=9, fontweight="bold", va="top", ha="left")
    axa.text(0.86, top * 0.99,
             "$\\phi_{delay}\\approx-\\phi_{inst}$\nREGENERATIVE\nDIFFERENCE",
             color=INK, fontsize=9, fontweight="bold", va="top", ha="right")
    axa.text(-0.55, axa.get_ylim()[1] * 0.45, f"delay edges\n(n={len(sig)})",
             color=S1, fontsize=10, fontweight="bold")
    axa.text(0.10, axa.get_ylim()[1] * 0.62, f"cross-channel\ncontrol (n={len(ctl)})",
             color=S2, fontsize=10, fontweight="bold")
    axa.set_xlim(-1.02, 1.02)
    axa.set_xlabel("negation score  (-1 = exact copy,  +1 = exact negation)")
    axa.set_ylabel("density")
    axa.set_title("A. All five folds, pooled: the mass sits on the SUM side")
    _grid(axa, "y")

    ys = np.arange(len(FOLDS))
    sm = [data[f]["summary"]["mean"] for f in FOLDS]
    cm = [data[f]["control_summary"]["mean"] for f in FOLDS]
    nm = [data[f]["untrained_null_summary"]["mean"] for f in FOLDS]
    axb.hlines(ys, [-1] * len(ys), [1] * len(ys), color=GRID, lw=0.8, zorder=1)
    axb.axvline(0, color=INK2, lw=1.0, zorder=2)
    axb.scatter(nm, ys, s=70, color=S3, edgecolor=SURFACE, linewidth=1.2, zorder=5)
    axb.scatter(cm, ys, s=70, color=S2, edgecolor=SURFACE, linewidth=1.2, zorder=5)
    axb.scatter(sm, ys, s=90, color=S1, edgecolor=SURFACE, linewidth=1.2, zorder=6)
    for y, v in zip(ys, sm):
        axb.text(v - 0.05, y, f"{v:+.2f}", color=S1, fontsize=9, fontweight="bold",
                 ha="right", va="center")
    axb.text(np.mean(sm), len(ys) - 0.42, "trained,\ndelay edges", color=S1,
             fontsize=9, fontweight="bold", ha="center")
    axb.text(np.mean(cm) - 0.02, -0.62, "cross-channel", color=S2, fontsize=9,
             fontweight="bold", ha="center")
    axb.text(np.mean(nm) + 0.06, -0.62, "untrained null", color=S3, fontsize=9,
             fontweight="bold", ha="left")
    axb.set_yticks(ys, [LABEL[f] for f in FOLDS])
    axb.set_xlim(-1.0, 0.45)
    axb.set_ylim(-0.95, len(ys) - 0.05)
    axb.set_xlabel("mean negation score")
    axb.set_title("B. Replicates 5/5; absent before training")
    fig.tight_layout()
    p = out / "u13c_regenerative_sum_not_difference.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return {"path": p, "sig_mean": float(np.mean(sm)), "ctl_mean": float(np.mean(cm)),
            "null_mean": float(np.mean(nm))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "results/figures/research_update_assets"
    out.mkdir(parents=True, exist_ok=True)

    for fn in (fig_tsr, fig_survival, fig_strategies, fig_regenerative):
        info = fn(root, out)
        p = info.pop("path")
        print(f"wrote {p}  {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
