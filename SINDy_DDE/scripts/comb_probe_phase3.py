#!/usr/bin/env python3
"""Confirm the comb reading by sweeping a sinusoid through the real edge pairs.

``scripts/comb_phase3.py`` *predicts* each delay-coupled pair's frequency
response from two stored scalars (``b = corr*gain``). This script **measures**
it: it loads the trained checkpoint, pulls the two learned activation functions
for a pair, drives them with ``x(t) = A cos(2 pi f t)``, forms

    ``s(t) = phi_inst(x(t)) + phi_delay(x(t - tau))``

and reports the amplitude of ``s`` at the drive frequency across a sweep. If the
measured curve tracks the predicted comb, the reading is confirmed on the
network itself rather than on a summary statistic.

**This needs no GPU.** It evaluates one-dimensional learned functions, not
forward passes over data, so it runs on a CPU partition in minutes. That matters
when the GPU allocation is capped: the confirmation is not competing for it.

Two controls, both mandatory:

* an **untrained** reinitialised copy of the same architecture, probed
  identically -- a comb that shows up on random weights is architecture, not
  learning (the lesson U12-B paid for);
* the **predicted-vs-measured residual** per pair, so a pair whose measurement
  does not match its prediction is reported rather than averaged away.

Usage::

    python3 scripts/comb_probe_phase3.py \
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \
        --regen-manifest results/phase3/xai/regenerative/fold2_w512/regenerative.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xai_phase3  # noqa: E402
from pri2m.xai.activation_modes import assert_reinitialised, reinitialised_copy  # noqa: E402
from pri2m.xai.comb import (  # noqa: E402
    comb_coefficient,
    comb_metrics,
    comb_response,
    measure_pair_response,
    peak_alignment,
)

DEFAULT_OUT = pathlib.Path("results/phase3/xai/comb")
DEFAULT_FIG = pathlib.Path("results/figures/research_update_assets")
_DC_NOTE = 0.25


def _load_model_args(args) -> argparse.Namespace:
    """The namespace ``xai_phase3._load_model`` reads off its ``args``.

    It takes an ``argparse.Namespace`` rather than keywords, so a missing
    attribute is an ``AttributeError`` raised *inside* the loader -- which on a
    cluster means a job that queues, starts and dies. Building it in one tested
    place keeps that failure in the test suite instead of in the queue.

    ``input_size``/``hidden_size`` stay ``None`` so the loader infers them from
    the state_dict; ``grid_size``/``spline_order`` must match training and carry
    the same defaults as ``xai_phase3``'s own parser (5 and 3).
    """
    return argparse.Namespace(
        input_size=None,
        hidden_size=None,
        grid_size=int(getattr(args, "grid_size", 5)),
        spline_order=int(getattr(args, "spline_order", 3)),
    )


def _module_by_name(model, name: str):
    """Resolve ``encoder.layers.0.kan_z`` to the module itself."""
    obj = model
    for part in name.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def pair_interpolants(model, kan_name: str, out_unit: int, in_inst: int,
                      in_delay: int, n: int = 400):
    """Callable ``(phi_inst, phi_delay)`` for one pair, over the learned grid.

    ``edge_function`` returns base+spline -- the function the forward pass
    actually computes -- sampled on ``grid_range``. Linear interpolation between
    those samples is exact enough for a frequency sweep and keeps the probe
    independent of the spline basis internals.
    """
    kan = _module_by_name(model, kan_name)
    sampled = kan.edge_function(n=n)
    x, curves = np.asarray(sampled["x"], float), np.asarray(sampled["curves"], float)
    lo, hi = float(x[0]), float(x[-1])

    def _make(in_idx: int):
        y = curves[int(out_unit), int(in_idx), :]

        def phi(u):
            # Clip to the sampled range: the drive is unit-amplitude and the KAN
            # grid is (-1, 1), so this only bites at the endpoints.
            return np.interp(np.clip(np.asarray(u, float), lo, hi), x, y)

        return phi

    return _make(in_inst), _make(in_delay)


def probe_model(model, pairs, tau: int, ratios, label: str) -> list[dict]:
    """Measure every pair's swept response on one model."""
    rows = []
    for p in pairs:
        try:
            phi_i, phi_d = pair_interpolants(
                model, p["kan"], p["out"], p["in_inst"], p["in_delay"]
            )
        except (AttributeError, IndexError, KeyError) as exc:
            rows.append({"arm": label, "kan": p.get("kan"), "out": p.get("out"),
                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        measured = measure_pair_response(phi_i, phi_d, tau, ratios)
        peak = float(np.max(measured)) if measured.size else float("nan")
        norm = measured / peak if peak > 0 else measured
        rows.append({
            "arm": label,
            "kan": p["kan"], "out": int(p["out"]),
            "in_inst": int(p["in_inst"]), "in_delay": int(p["in_delay"]),
            "channel": p.get("channel"),
            "measured": [float(v) for v in measured],
            "measured_normalised": [float(v) for v in norm],
            # Where the measured response peaks, in tooth-pass units. The comb
            # claim is about the PERIODIC structure, so the DC lobe is excluded
            # from the search: a nonlinear edge rectifies, and a global argmax
            # then returns f = 0, which a distance-to-nearest-harmonic test
            # scores as a hit because 0 is an integer.
            **peak_alignment(ratios, measured),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--regen-manifest", required=True,
                    help="U13-C regenerative.json; supplies the pairs and their b.")
    ap.add_argument("--preprocessing", default=None,
                    help="preprocessing.json (defaults to the checkpoint's run dir).")
    ap.add_argument("--tau", type=int, default=87,
                    help="Delay in samples for the sweep axis (dataset tau_raw mean).")
    ap.add_argument("--n-pairs", type=int, default=25,
                    help="Top pairs by importance to probe.")
    ap.add_argument("--n-freqs", type=int, default=241)
    ap.add_argument("--max-ratio", type=float, default=3.0)
    ap.add_argument("--grid-size", type=int, default=5,
                    help="KAN grid size (must match training).")
    ap.add_argument("--spline-order", type=int, default=3,
                    help="KAN spline order (must match training).")
    ap.add_argument("--untrained-seed", type=int, default=0)
    ap.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--fig-dir", type=pathlib.Path, default=DEFAULT_FIG)
    ap.add_argument("--tag", default="fold2")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    manifest_path = pathlib.Path(args.regen_manifest)
    regen = json.loads(manifest_path.read_text())
    pairs = (regen.get("top_pairs") or [])[: int(args.n_pairs)]
    if not pairs:
        print(f"No top_pairs in {manifest_path}", file=sys.stderr)
        return 2

    # Provenance BEFORE the run, so a missing input fails in seconds rather than
    # after the sweep (the tsr_control_phase3 lesson).
    ckpt = pathlib.Path(args.checkpoint)
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}", file=sys.stderr)
        return 2
    pre_path = pathlib.Path(args.preprocessing) if args.preprocessing else (
        ckpt.parent.parent / "preprocessing.json"
    )
    preprocessing = json.loads(pre_path.read_text()) if pre_path.exists() else None

    load_args = _load_model_args(args)
    model = xai_phase3._load_model(str(ckpt), load_args, preprocessing)
    model.eval()

    # reinitialised_copy returns (model, n_submodules_reset) -- the count is part
    # of the control's evidence, so it goes into provenance rather than being
    # dropped on the floor.
    untrained, n_reset = reinitialised_copy(model, seed=int(args.untrained_seed))
    weight_diff = assert_reinitialised(model, untrained)

    ratios = np.linspace(0.0, float(args.max_ratio), int(args.n_freqs))
    trained_rows = probe_model(model, pairs, args.tau, ratios, "trained")
    untrained_rows = probe_model(untrained, pairs, args.tau, ratios, "untrained")

    # Predicted vs measured, per pair.
    comparisons = []
    for p, row in zip(pairs, trained_rows):
        if "error" in row:
            continue
        b = comb_coefficient(p["corr"], p["gain"])
        predicted = comb_response(b, ratios)
        predicted = predicted / np.max(predicted)
        measured = np.asarray(row["measured_normalised"], float)
        comparisons.append({
            "kan": row["kan"], "out": row["out"], "channel": row.get("channel"),
            "b": b,
            "predicted_contrast_db": comb_metrics(b)["contrast_db"],
            "rms_residual": float(np.sqrt(np.mean((measured - predicted) ** 2))),
            "corr_predicted_measured": float(
                np.corrcoef(measured, predicted)[0, 1]
            ) if measured.std() > 0 and predicted.std() > 0 else None,
            "measured_peak_ratio": row["peak_ratio"],
        })

    def _peak_alignment(rows):
        d = [r["distance_to_nearest_harmonic"] for r in rows
             if r.get("distance_to_nearest_harmonic") is not None]
        return {"n": len(d), "median_distance_to_harmonic": float(np.median(d)) if d else None,
                "frac_within_0_1": float(np.mean([x <= 0.1 for x in d])) if d else None,
                "dc_lobe_excluded_below": _DC_NOTE}

    out = {
        "analysis": "swept confirmation of the tooth-pass comb reading",
        "provenance": {
            "checkpoint": str(ckpt),
            "regen_manifest": str(manifest_path),
            "tau": int(args.tau),
            "n_pairs": len(pairs),
            "untrained_seed": int(args.untrained_seed),
            "untrained_mean_weight_diff": float(weight_diff),
            "untrained_n_submodules_reinitialised": int(n_reset),
            "device": "cpu",
        },
        "ratios": [float(r) for r in ratios],
        "trained": trained_rows,
        "untrained": untrained_rows,
        "predicted_vs_measured": comparisons,
        "summary": {
            "trained_peak_alignment": _peak_alignment(trained_rows),
            "untrained_peak_alignment": _peak_alignment(untrained_rows),
            "median_rms_residual": float(np.median(
                [c["rms_residual"] for c in comparisons])) if comparisons else None,
            "median_corr_predicted_measured": float(np.median(
                [c["corr_predicted_measured"] for c in comparisons
                 if c["corr_predicted_measured"] is not None])) if comparisons else None,
        },
        "complete": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest = args.output_dir / f"comb_probe_{args.tag}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"Wrote {dest}")
    s = out["summary"]
    print(f"  trained peaks within 0.1 of a harmonic: "
          f"{s['trained_peak_alignment']['frac_within_0_1']}")
    print(f"  untrained (control):                    "
          f"{s['untrained_peak_alignment']['frac_within_0_1']}")
    print(f"  median corr(predicted, measured):       "
          f"{s['median_corr_predicted_measured']}")

    if not args.no_figures and comparisons:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        args.fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        row = trained_rows[0]
        b = comparisons[0]["b"]
        pred = comb_response(b, ratios); pred = pred / pred.max()
        ax.plot(ratios, row["measured_normalised"], lw=2.0, color="#1f4e79",
                label="measured (swept through the learned edges)")
        ax.plot(ratios, pred, lw=1.4, ls="--", color="#c62828",
                label=f"predicted comb, b = {b:.3f}")
        u = untrained_rows[0]
        if "measured_normalised" in u:
            ax.plot(ratios, u["measured_normalised"], lw=1.2, color="#9e9e9e",
                    alpha=0.9, label="untrained control")
        for k in range(1, int(args.max_ratio) + 1):
            ax.axvline(k, color="#2e7d32", ls=":", lw=1.0)
        ax.set_xlabel("frequency / tooth-pass frequency")
        ax.set_ylabel("normalised response")
        ax.set_title("Predicted vs measured response of the top delay pair")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = args.fig_dir / f"comb_probe_{args.tag}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        print(f"  figure: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
