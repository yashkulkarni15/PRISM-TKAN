#!/usr/bin/env python3
"""Did the trained TKAN LEARN the regenerative difference x(t) - x(t-tau)?

Eval-only runner for :mod:`pri2m.xai.regenerative`. Loads a trained checkpoint,
samples real windows so every edge is scored over the range the data actually
visits, and scores each (instantaneous, tau-delayed) edge pair of the bottom
encoder gates: a pair whose delayed edge is the NEGATIVE of its instantaneous
edge implements ``phi(x_t) - phi(x_{t-tau})``, i.e. the DDE's regenerative term,
learned from data with nothing hardcoded.

Writes ``regenerative.json`` (all rows + summaries + the cross-channel control)
and ``regenerative.png`` (score histogram vs control, plus the top pairs' curves).

Example::

    python scripts/regenerative_phase3.py \
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \
        --npz-root $PWD/phase2_signals --table results/phase3/phase3_dataset.csv \
        --output-dir results/phase3/xai/fold0 --n-windows 128
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import xai_phase3  # noqa: E402  -- reuse the battle-tested checkpoint loaders
from pri2m.data.phase3_sequence import WindowConfig  # noqa: E402
from pri2m.xai import activations as act_mod  # noqa: E402
from pri2m.xai import regenerative as regen  # noqa: E402
from pri2m.xai.sampling import CHANNELS, build_labeled_batch  # noqa: E402


def _plot(res, model, out_path, top_k=6, n=200):
    """Score histogram (matched vs control) + the top pairs' phi curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = [r for r in regen.top_pairs(res["rows"], top_k)]
    kans = {name: m for name, m in model.named_modules()}
    ncols = 3
    nrows = 1 + int(np.ceil(len(top) / ncols))
    fig = plt.figure(figsize=(3.6 * ncols, 2.8 * nrows))

    ax = fig.add_subplot(nrows, 1, 1)
    matched = [r["score"] for r in res["rows"] if np.isfinite(r["score"])]
    ctrl = [r["score"] for r in res["control_rows"] if np.isfinite(r["score"])]
    bins = np.linspace(-1, 1, 41)
    ax.hist(matched, bins=bins, alpha=0.65, label=f"matched channel (n={len(matched)})",
            color="tab:red", density=True)
    ax.hist(ctrl, bins=bins, alpha=0.45, label=f"cross-channel control (n={len(ctrl)})",
            color="tab:gray", density=True)
    ax.axvline(regen.NEGATION_THRESHOLD, ls="--", c="k", lw=1)
    ax.axvline(regen.SUMMATION_THRESHOLD, ls="--", c="k", lw=1)
    ax.set_xlabel("negation score   (+1 = x(t) - x(t-tau),  -1 = x(t) + x(t-tau))")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    for k, r in enumerate(top):
        axk = fig.add_subplot(nrows, ncols, ncols + k + 1)
        kan = kans[r["kan"]]
        cur = kan.edge_function(n)
        xs = np.asarray(cur["x"])
        m = (xs >= r["lo"]) & (xs <= r["hi"])
        inst = np.asarray(cur["curves"][r["out"], r["in_inst"]])
        dly = np.asarray(cur["curves"][r["out"], r["in_delay"]])
        axk.plot(xs[m], inst[m] - inst[m].mean(), c="tab:blue", lw=1.4, label="phi inst")
        axk.plot(xs[m], dly[m] - dly[m].mean(), c="tab:red", lw=1.4, label="phi delay")
        axk.plot(xs[m], -(inst[m] - inst[m].mean()), c="tab:blue", ls=":", lw=1.1,
                 label="-phi inst")
        axk.set_title(f"{r['kan'].split('.')[-1]} u{r['out']} {r['channel']}\n"
                      f"score {r['score']:.2f}  corr {r['corr']:.2f}  gain {r['gain']:.2f}",
                      fontsize=7)
        axk.tick_params(labelsize=6)
        if k == 0:
            axk.legend(fontsize=6)
    fig.suptitle("Learned regenerative difference: is the delay edge the negative "
                 "of the instantaneous edge?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


# --------------------------------------------------------------------------- #
# Checkpoint-free fallback: score the pairs a white-box run already fitted
# --------------------------------------------------------------------------- #
_FORMULA_OK = re.compile(r"^[0-9eE.+\-*/()^,x |a-z]+$")
_FORMULA_NS = {"tanh": np.tanh, "sin": np.sin, "cos": np.cos, "exp": np.exp,
               "abs": np.abs, "maximum": np.maximum, "sqrt": np.sqrt}


def evaluate_formula(formula: str, xs: np.ndarray) -> np.ndarray:
    """Evaluate one ``symbolic_edges.csv`` closed form on ``xs``.

    Only the shapes ``pri2m.xai.symbolic`` emits are accepted (polynomials,
    ``a*tanh(b*x+c)+d``, ``a*sin(w*x+p)+d``, ``a*|x|+b``, ``a*max(x, 0)+b``);
    anything outside the character whitelist is refused rather than executed.
    A constant formula broadcasts to ``xs``' shape.
    """
    text = str(formula).strip()
    if not text or not _FORMULA_OK.match(text):
        raise ValueError(f"refusing to evaluate formula {formula!r}")
    text = (text.replace("^", "**").replace("|x|", "abs(x)")
                .replace("max(x, 0)", "maximum(x, 0)"))
    value = eval(text, {"__builtins__": {}}, {**_FORMULA_NS, "x": xs})  # noqa: S307
    return np.asarray(value, dtype=float) * np.ones_like(xs)


def score_symbolic_csv(path, lo=-0.95, hi=0.95, n=400) -> list[dict]:
    """Score matched pairs from a white-box run's ``symbolic_edges.csv``.

    A row pairs up when the SAME gate and out unit carry both a fitted
    ``x_delay`` edge ``delay:<ch>`` and its ``x_t`` partner ``<ch>``. This is a
    fallback for machines without checkpoints: the closed forms are surrogates
    for the real edge functions (their ``r2`` is reported per row) and only the
    run's top-k edges were ever fitted, so absence of a pair means "not in the
    top-k", never "no relationship".
    """
    xs = np.linspace(float(lo), float(hi), int(n))
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if str(r.get("kan", "")).startswith("encoder")]
    inst = {(r["kan"], r["out"], r["name"]): r for r in rows if r["block"] == "x_t"}
    out = []
    for r in rows:
        if r["block"] != "x_delay":
            continue
        channel = r["name"].removeprefix("delay:")
        partner = inst.get((r["kan"], r["out"], channel))
        if partner is None or not r["formula"] or not partner["formula"]:
            continue
        row = regen.negation_metrics(evaluate_formula(partner["formula"], xs),
                                     evaluate_formula(r["formula"], xs))
        row.update({
            "kan": r["kan"], "out": int(r["out"]), "channel": channel,
            "in_inst": int(partner["in"]), "in_delay": int(r["in"]),
            "lo": float(lo), "hi": float(hi),
            "importance_inst": float(partner["importance"]),
            "importance_delay": float(r["importance"]),
            "weight": min(float(partner["importance"]), float(r["importance"])),
            "r2_inst": float(partner["r2"]), "r2_delay": float(r["r2"]),
            # the closed form hides the base/spline split -> unknowable here
            "spline_share_inst": float("nan"), "spline_share_delay": float("nan"),
            "base_dominated": None, "surrogate": True,
        })
        out.append(row)
    return out


def _run_symbolic_mode(args) -> int:
    payload = {"mode": "symbolic-surrogate", "sweep": [args.sweep_lo, args.sweep_hi],
               "per_source": {}}
    for path in args.symbolic_csv:
        rows = score_symbolic_csv(path, args.sweep_lo, args.sweep_hi)
        payload["per_source"][str(path)] = {
            "n_pairs": len(rows),
            "summary": regen.summarize_rows(rows),
            "rows": rows,
        }
        s = payload["per_source"][str(path)]["summary"]
        if rows:
            print(f"{path}: n_pairs={len(rows)} mean={s['mean']:+.3f} "
                  f"frac_negating={s['frac_negating']:.2f} "
                  f"frac_summing={s['frac_summing']:.2f}")
        else:
            print(f"{path}: n_pairs=0 (no matched pair in this run's top-k edges)")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "regenerative_symbolic.json").write_text(
        json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(f"wrote {out_dir / 'regenerative_symbolic.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--npz-root")
    p.add_argument("--table")
    p.add_argument("--symbolic-csv", nargs="+", default=None,
                   help="Checkpoint-free fallback: score matched pairs straight "
                        "from one or more white-box symbolic_edges.csv files.")
    p.add_argument("--sweep-lo", type=float, default=-0.95)
    p.add_argument("--sweep-hi", type=float, default=0.95)
    p.add_argument("--preprocessing", default=None)
    p.add_argument("--output-dir", default="results/phase3/xai")
    p.add_argument("--n-windows", type=int, default=128)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--window-length", type=int, default=1024)
    p.add_argument("--window-stride", type=int, default=512)
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--sweep-points", type=int, default=200)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--tag", default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-null", action="store_true",
                   help="Skip the untrained-twin null scan.")
    # xai_phase3._load_model reads these off the args namespace.
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    args = p.parse_args(argv)

    if args.symbolic_csv:
        return _run_symbolic_mode(args)
    missing = [f"--{k}" for k in ("checkpoint", "npz_root", "table")
               if not getattr(args, k)]
    if missing:
        p.error(f"{', '.join(missing)} are required unless --symbolic-csv is given")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_path = args.preprocessing or xai_phase3._find_preprocessing(args.checkpoint)
    preprocessing = json.loads(Path(pre_path).read_text(encoding="utf-8")) if pre_path else None
    model = xai_phase3._load_model(args.checkpoint, args, preprocessing)
    model.pool = int(args.pool)
    model.eval()

    cfg = WindowConfig(length=args.window_length, stride=args.window_stride, channels=CHANNELS)
    batch, _labels = build_labeled_batch(
        args.table, args.npz_root, preprocessing, args.n_windows, cfg, seed=args.sample_seed
    )
    stats = act_mod.collect_activations(model, batch)

    res = regen.scan_model(model, stats, channel_names=CHANNELS, n=args.sweep_points)

    # Untrained twin, same shape and same visited ranges: what the metric returns
    # for a model that cannot have learned anything.
    null_summary = None
    if not args.no_null:
        null_model = regen.random_edge_null(model, seed=args.sample_seed)
        null_summary = regen.scan_model(null_model, stats, channel_names=CHANNELS,
                                        n=args.sweep_points, control=False)["summary"]

    payload = {
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "tag": args.tag or out_dir.name,
            "n_windows": int(batch["window"].shape[0]),
            "sweep_points": int(args.sweep_points),
            "negation_threshold": regen.NEGATION_THRESHOLD,
            "summation_threshold": regen.SUMMATION_THRESHOLD,
        },
        "summary": res["summary"],
        "control_summary": res["control_summary"],
        "untrained_null_summary": null_summary,
        "by_gate": regen.group_summaries(res["rows"], "kan"),
        "by_channel": regen.group_summaries(res["rows"], "channel"),
        "top_pairs": regen.top_pairs(res["rows"], 25),
        "rows": res["rows"],
        "control_rows": res["control_rows"],
    }
    if not args.no_plot:
        payload["figure"] = _plot(res, model, out_dir / "regenerative.png",
                                  top_k=args.top_k, n=args.sweep_points)
    (out_dir / "regenerative.json").write_text(json.dumps(payload, indent=1, default=float),
                                               encoding="utf-8")

    s, c = res["summary"], res["control_summary"]
    print(f"matched pairs: n={s['n_scored']} mean={s['mean']:.3f} "
          f"weighted={s['weighted_mean']:.3f} frac_negating={s['frac_negating']:.3f} "
          f"frac_summing={s['frac_summing']:.3f} base_dominated={s['frac_base_dominated']:.3f}")
    print(f"control pairs: n={c['n_scored']} mean={c['mean']:.3f} "
          f"frac_negating={c['frac_negating']:.3f}")
    if null_summary:
        print(f"untrained null: n={null_summary['n_scored']} "
              f"mean={null_summary['mean']:.3f} "
              f"frac_negating={null_summary['frac_negating']:.3f}")
    print(f"wrote {out_dir / 'regenerative.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
