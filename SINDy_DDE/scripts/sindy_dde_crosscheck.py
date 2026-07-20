#!/usr/bin/env python3
"""SINDy-DDE cross-check runner: independent white-box recovery of the delay tau.

Eval-only, NO neural network and NO ``pysindy``. For a chosen parent-group fold,
this pulls REAL chatter windows from the Phase 2 NPZ signal tree, runs the
hand-rolled delay-embedded sparse regression (:mod:`pri2m.analysis.sindy_dde`) on
one channel of each window, and recovers the regenerative delay ``tau`` directly
from the raw cutting signal -- an independent estimate to compare against the
kinematic tooth-pass delay ``round(fs / tpf)`` and the TKAN's delay-ablation
result.

Each window is scanned on a *relative* grid (multiples of that window's OWN
physical tooth-pass delay), so windows spanning different operating points
aggregate on a common ``tau / tau_physical`` axis (the same per-window-relative
convention as ``physics.tau_sweep_relative`` in the TKAN white-box). The runner
selects ``tau*`` per window by minimum sparse-regression residual, reports the
residual-vs-ratio curve, aggregates the recovered ratio + the fraction of
windows in which the regenerative term ``x(t)-x(t-tau)`` is selected, and writes
a ``sindy_dde_manifest.json`` plus a residual-vs-tau figure.

Real-data example (Great Lakes)::

    python scripts/sindy_dde_crosscheck.py \
        --table results/phase3/phase3_dataset.csv \
        --npz-root $PWD/phase2_signals --folds results/phase3/phase3_folds.json \
        --fold 2 --channel xtpos --n-windows 64 \
        --output-dir results/phase3/xai/sindy_dde

Local smoke test (no NPZ tree needed; synthesizes DDE windows with a known tau)::

    python scripts/sindy_dde_crosscheck.py --synthetic 8 --deriv-order 1 \
        --output-dir /tmp/sindy_dde_selftest
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from pri2m.analysis import sindy_dde as sd  # noqa: E402

CHANNELS = ("xtpos", "ytpos", "Forcex", "Forcey")
DEFAULT_FIGURE = "results/figures/research_update_assets/sindy_dde_tau_recovery.png"


# --------------------------------------------------------------------------- #
# Window sources
# --------------------------------------------------------------------------- #
def _synthetic_windows(n_windows: int, seed: int) -> list[dict]:
    """Bounded chaotic regenerative DDE windows with a KNOWN delay (self-test).

    First-order oscillator ``xdot = -gamma x - k1 r - k3 r^3`` with
    ``r = x - x(t-tau)``; the cubic regenerative saturation keeps it bounded and
    broadband so the delay is identifiable. Returns per-window dicts matching the
    real-data path (``signal``/``fs``/``tpf``/``physical_delay``/...).
    """
    fs, tau_s, gamma, k1, k3 = 500.0, 0.05, 4.0, 40.0, 300.0
    dt = 1.0 / fs
    n = int(round(16.0 * fs))
    D = int(round(tau_s * fs))
    out = []
    for w in range(n_windows):
        rng = np.random.default_rng(1000 + seed + w)
        x = np.zeros(n)
        x[: D + 1] = 0.5 + 1e-2 * rng.standard_normal(D + 1)
        for i in range(D, n - 1):
            xd = x[i - D]
            def rhs(ix):
                r = ix - xd
                return -gamma * ix - k1 * r - k3 * r**3
            a = rhs(x[i]); b = rhs(x[i] + 0.5 * dt * a)
            c = rhs(x[i] + 0.5 * dt * b); e = rhs(x[i] + dt * c)
            x[i + 1] = x[i] + dt / 6.0 * (a + 2 * b + 2 * c + e)
        out.append({
            "signal": x, "fs": fs, "tpf": fs / D, "physical_delay": D,
            "dataset": -1, "file": f"synthetic_{w}", "label": 1,
        })
    return out


def _fold_of_row(table, folds_json):
    """Map each table row to its fold via ``group_to_fold`` on ``parent_group``."""
    g2f = {int(k): int(v) for k, v in folds_json["group_to_fold"].items()}
    pg = table["parent_group"].astype(int)
    return pg.map(g2f)


def _real_windows(args) -> list[dict]:
    """Pull raw (un-normalized) chatter windows for one fold from the NPZ tree."""
    import pandas as pd

    from pri2m.data.phase3_sequence import (
        Phase3SequenceDataset,
        WindowConfig,
        collate_windows,
    )

    npz_root = Path(args.npz_root)
    table = pd.read_csv(args.table)
    folds_json = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    fold_of = _fold_of_row(table, folds_json)

    sel = table[(fold_of == args.fold) & (table["label"] == 1)].copy()
    if "exclude_supervised" in sel.columns:
        sel = sel[~sel["exclude_supervised"].astype(bool)]

    def _npz_exists(row):
        stem = Path(str(row["file"])).stem
        return (npz_root / f"dataset_{int(row['dataset'])}" / f"{stem}.npz").is_file()

    sel = sel[sel.apply(_npz_exists, axis=1)]
    if sel.empty:
        raise RuntimeError(
            f"no chatter windows with NPZ files found for fold {args.fold} "
            f"under {npz_root}"
        )

    rng = np.random.default_rng(args.seed)
    if len(sel) > args.n_windows:
        sel = sel.iloc[rng.choice(len(sel), size=args.n_windows, replace=False)]

    cfg = WindowConfig(length=args.window_length, stride=args.window_stride, channels=CHANNELS)
    ds = Phase3SequenceDataset(sel, npz_root, cfg, norm=None, supervised_only=False)
    if len(ds) == 0:
        raise RuntimeError("no windows fit the requested length on the sampled files")

    ch_idx = CHANNELS.index(args.channel)
    # One window per file (the first), to keep operating points diverse.
    seen: set = set()
    out = []
    for i in range(len(ds)):
        item = ds[i]
        key = item["file"]
        if key in seen:
            continue
        seen.add(key)
        win = item["window"].detach().cpu().numpy()  # [L, C]
        out.append({
            "signal": win[:, ch_idx].astype(np.float64),
            "fs": float(item["fs"]),
            "tpf": float(item["tpf"]),
            "physical_delay": int(item["tau"]),
            "dataset": int(item["dataset"]),
            "file": str(item["file"]),
            "label": int(item["label"]),
        })
        if len(out) >= args.n_windows:
            break
    return out


# --------------------------------------------------------------------------- #
# Scan + aggregate
# --------------------------------------------------------------------------- #
def scan_windows(windows, fracs, args) -> dict:
    """Run a per-window relative tau-scan; return per-window + aggregate results."""
    per_window = []
    residual_stack = []  # [n_ok, len(fracs)]
    for w in windows:
        x = np.asarray(w["signal"], dtype=np.float64)
        fs = float(w["fs"])
        dt = 1.0 / fs
        Dw = int(w["physical_delay"])
        # Integer trial delays = multiples of THIS window's own physical delay.
        grid = np.array([max(1, int(round(f * Dw))) for f in fracs], dtype=int)
        # Need the signal long enough for the largest trial delay + support.
        if len(x) - int(grid.max()) < 64:
            continue
        try:
            res = sd.tau_scan(
                x, dt, grid,
                deriv_order=args.deriv_order,
                threshold=args.threshold,
                poly_order=args.poly_order,
                regen_poly_order=args.regen_poly_order,
                include_delayed=True,
            )
        except (ValueError, FloatingPointError):
            continue
        residuals = np.asarray(res["residuals"], dtype=float)
        best = int(np.argmin(residuals))
        per_window.append({
            "dataset": w["dataset"],
            "file": w["file"],
            "fs": fs,
            "tpf": float(w["tpf"]),
            "physical_delay_samples": Dw,
            "tau_star_samples": int(res["tau_star"]),
            "recovered_over_physical": float(fracs[best]),
            "regen_selected": bool(res["regen_selected"]),
            "regen_coef": float(res["regen_coef"]),
            "residual_at_star": float(res["residual_at_star"]),
        })
        residual_stack.append(residuals)

    if not per_window:
        raise RuntimeError("no windows produced a usable tau-scan")

    residual_stack = np.vstack(residual_stack)
    ratios = np.array([p["recovered_over_physical"] for p in per_window])
    phys = np.array([p["physical_delay_samples"] for p in per_window])
    taus = np.array([p["tau_star_samples"] for p in per_window])
    regen_sel = np.array([p["regen_selected"] for p in per_window])
    regen_coef = np.array([p["regen_coef"] for p in per_window])

    aggregate = {
        "n_windows_used": len(per_window),
        "physical_delay_median_samples": float(np.median(phys)),
        "tau_star_median_samples": float(np.median(taus)),
        "recovered_over_physical_median": float(np.median(ratios)),
        "recovered_over_physical_iqr": [float(np.percentile(ratios, 25)),
                                        float(np.percentile(ratios, 75))],
        "frac_regen_selected": float(regen_sel.mean()),
        "regen_coef_median": float(np.median(regen_coef)),
        "tpf_median_hz": float(np.median([p["tpf"] for p in per_window])),
        "fs_median_hz": float(np.median([p["fs"] for p in per_window])),
    }
    curve = {
        "ratio_grid": [float(f) for f in fracs],
        "mean_residual": [float(v) for v in residual_stack.mean(axis=0)],
        "std_residual": [float(v) for v in residual_stack.std(axis=0)],
    }
    return {"per_window": per_window, "aggregate": aggregate, "residual_curve": curve}


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot_residual_curve(curve, aggregate, paths) -> list[str]:
    """Residual-vs-(tau / tau_physical), mean +/- std across windows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = np.asarray(curve["ratio_grid"])
    mean = np.asarray(curve["mean_residual"])
    std = np.asarray(curve["std_residual"])
    rec = aggregate["recovered_over_physical_median"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(fracs, mean, color="#1f77b4", lw=2, label="mean residual")
    ax.fill_between(fracs, mean - std, mean + std, color="#1f77b4", alpha=0.18,
                    label="+/- 1 sd across windows")
    ax.axvline(1.0, color="#444444", ls="--", lw=1.5,
               label="physical tooth-pass delay (x1)")
    ax.axvline(rec, color="#d62728", ls="-", lw=1.5,
               label=f"recovered tau* (median {rec:.2f}x)")
    ax.set_xlabel(r"trial delay $\tau\,/\,\tau_{\mathrm{physical}}$")
    ax.set_ylabel("normalized STLSQ residual")
    ax.set_title(
        f"SINDy-DDE delay recovery  "
        f"(n={aggregate['n_windows_used']} windows, "
        f"regen selected in {100*aggregate['frac_regen_selected']:.0f}%)"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    written = []
    for p in paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=130)
        written.append(str(p))
    plt.close(fig)
    return written


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _git_sha(root):
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="results/phase3/phase3_dataset.csv")
    ap.add_argument("--npz-root", default=None)
    ap.add_argument("--folds", default="results/phase3/phase3_folds.json")
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--channel", default="xtpos", choices=CHANNELS)
    ap.add_argument("--n-windows", type=int, default=64)
    ap.add_argument("--window-length", type=int, default=1024)
    ap.add_argument("--window-stride", type=int, default=512)
    ap.add_argument("--deriv-order", type=int, default=2, choices=(1, 2),
                    help="Target derivative: 2 = acceleration (milling model), 1 = velocity.")
    ap.add_argument("--threshold", type=float, default=0.05, help="STLSQ sparsity threshold.")
    ap.add_argument("--poly-order", type=int, default=3, help="Max polynomial degree of x.")
    ap.add_argument("--regen-poly-order", type=int, default=1,
                    help="Max power of the regenerative term (1 = linear regen only).")
    ap.add_argument("--grid-lo-frac", type=float, default=0.4,
                    help="Lowest trial delay as a fraction of the physical tooth-pass delay.")
    ap.add_argument("--grid-hi-frac", type=float, default=2.5,
                    help="Highest trial delay as a fraction of the physical tooth-pass delay.")
    ap.add_argument("--grid-points", type=int, default=44, help="Number of trial delays.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results/phase3/xai/sindy_dde")
    ap.add_argument("--figure", default=DEFAULT_FIGURE,
                    help="Extra path for the residual-vs-tau figure (research update asset).")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="If >0, skip real data and synthesize N DDE windows (self-test).")
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic > 0:
        windows = _synthetic_windows(args.synthetic, args.seed)
        source = "synthetic"
    else:
        if not args.npz_root:
            ap.error("--npz-root is required unless --synthetic is set")
        windows = _real_windows(args)
        source = "real"

    fracs = np.linspace(args.grid_lo_frac, args.grid_hi_frac, args.grid_points)
    scanned = scan_windows(windows, fracs, args)

    fig_out = out_dir / "residual_vs_tau.png"
    fig_paths = plot_residual_curve(
        scanned["residual_curve"], scanned["aggregate"], [fig_out, args.figure]
    )

    provenance = {
        "method": "SINDy-DDE (hand-rolled delay-embedded STLSQ, no neural net)",
        "reference": "delay-aware SINDy, arXiv:2407.19640",
        "uses_pysindy": False,
        "source": source,
        "table": str(args.table),
        "npz_root": str(args.npz_root) if args.npz_root else None,
        "folds": str(args.folds),
        "fold": args.fold,
        "channel": args.channel,
        "n_windows_requested": args.n_windows if source == "real" else args.synthetic,
        "window_length": args.window_length,
        "window_stride": args.window_stride,
        "deriv_order": args.deriv_order,
        "target": "acceleration (xddot)" if args.deriv_order == 2 else "velocity (xdot)",
        "threshold": args.threshold,
        "poly_order": args.poly_order,
        "regen_poly_order": args.regen_poly_order,
        "grid": {
            "lo_frac": args.grid_lo_frac, "hi_frac": args.grid_hi_frac,
            "points": args.grid_points, "axis": "tau / tau_physical (per-window relative)",
        },
        "seed": args.seed,
        "git_sha": _git_sha(_ROOT),
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "python": sys.version.split()[0],
    }

    manifest = {
        "provenance": provenance,
        "aggregate": scanned["aggregate"],
        "per_window": scanned["per_window"],
        "residual_curve": scanned["residual_curve"],
        "artifacts": {"figure": str(fig_out), "figures_written": fig_paths},
    }
    manifest_path = out_dir / "sindy_dde_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    agg = scanned["aggregate"]
    print(f"SINDy-DDE cross-check ({source}): {agg['n_windows_used']} windows")
    print(f"  recovered tau* / physical (median): {agg['recovered_over_physical_median']:.3f} "
          f"(IQR {agg['recovered_over_physical_iqr'][0]:.3f}-{agg['recovered_over_physical_iqr'][1]:.3f})")
    print(f"  regenerative term selected in {100*agg['frac_regen_selected']:.0f}% of windows")
    print(f"  manifest -> {manifest_path}")
    print(f"  figure   -> {fig_paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
