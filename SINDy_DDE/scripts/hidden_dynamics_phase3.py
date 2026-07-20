#!/usr/bin/env python3
"""Hidden-state dynamical analysis of a trained Phase 3 TKAN checkpoint.

Regenerative chatter is a delay-differential-equation instability: at ``b_lim``
the milling map's dominant characteristic multiplier crosses the unit circle. A
recurrent chatter classifier should carry that bifurcation in its *internal*
dynamics. This runner reads it out (Sussillo & Barak style), eval-only.

It loads a trained ``TKANModel`` (no retraining), samples a class-balanced set of
real stable + chatter windows from the Phase 2 NPZ tree (reusing
``whitebox_phase3.build_labeled_batch``), and for every window rolls the encoder's
recurrent cell out and computes the recurrent Jacobian ``J_t = d h_t / d h_{t-1}``
at each timestep, its eigenvalues, and its leading ``|λ|`` (spectral radius). The
per-window leading ``|λ|`` are aggregated into stable-vs-chatter statistics and
the ``chatter > stable`` bifurcation hypothesis is scored. An optional slow/fixed
-point probe reports the spectral radius near a fixed point of the input-clamped
map.

Outputs (under ``--output-dir``, default ``results/phase3/xai/dynamics``):
* ``hidden_dynamics_manifest.json`` -- provenance + results + artifact paths.
* ``hidden_dynamics_<tag>.png`` -- stable-vs-chatter leading-``|λ|`` distribution
  with the ``|λ| = 1`` line; also copied to
  ``results/figures/research_update_assets/hidden_dynamics_<tag>.png``.

Example::

    python scripts/hidden_dynamics_phase3.py \
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \
        --npz-root $PWD/phase2_signals --table results/phase3/phase3_dataset.csv \
        --output-dir results/phase3/xai/dynamics --tag fold2 --n-windows 128
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

# Reuse the battle-tested checkpoint loader + labeled-batch builder.
import xai_phase3  # noqa: E402
from whitebox_phase3 import build_labeled_batch  # noqa: E402
from pri2m.data.phase3_sequence import WindowConfig  # noqa: E402
from pri2m.xai import dynamics  # noqa: E402

CHANNELS = ("xtpos", "ytpos", "Forcex", "Forcey")


def _git_sha(root) -> str | None:
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _now_date() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz-root", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--preprocessing", default=None)
    parser.add_argument("--output-dir", default="results/phase3/xai/dynamics")
    parser.add_argument("--n-windows", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=512)
    parser.add_argument("--pool", type=int, default=4, help="Temporal pool factor (match training).")
    parser.add_argument("--t-stride", type=int, default=1,
                        help="Compute J_t every t-stride pooled steps (>1 subsamples the trajectory).")
    parser.add_argument("--layer", type=int, default=0,
                        help="Encoder recurrent layer to analyse (0 = delay-aware bottom cell).")
    parser.add_argument("--reduce", default="mean", choices=("mean", "median", "max", "last"),
                        help="How to reduce a window's per-step spectral radii to one scalar.")
    parser.add_argument("--slow-points", action="store_true",
                        help="Also run the optional Sussillo-Barak slow/fixed-point probe.")
    parser.add_argument("--slow-inits", type=int, default=8)
    parser.add_argument("--slow-steps", type=int, default=500)
    # xai_phase3._load_model reads these off the args namespace.
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--val-auc", type=float, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--assets-dir", default="results/figures/research_update_assets",
                        help="Where to also copy the headline figure for research updates.")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model + preprocessing (reuse xai_phase3 loaders) -------------- #
    pre_path = args.preprocessing or xai_phase3._find_preprocessing(args.checkpoint)
    preprocessing = json.loads(Path(pre_path).read_text(encoding="utf-8")) if pre_path else None
    model = xai_phase3._load_model(args.checkpoint, args, preprocessing)
    model.pool = int(args.pool)  # forward-time op; not in the state_dict
    model.eval()
    tag = args.tag or out_dir.name

    # ---- Sample a class-balanced batch of real stable + chatter windows ----- #
    cfg = WindowConfig(length=args.window_length, stride=args.window_stride, channels=CHANNELS)
    batch, labels = build_labeled_batch(
        args.table, args.npz_root, preprocessing, args.n_windows, cfg, seed=args.sample_seed
    )
    windows = batch["window"].detach().cpu().numpy()  # [N, L, C]
    taus = batch["tau"].detach().cpu().numpy().tolist()
    n_windows = int(windows.shape[0])
    n_stable = int(np.sum(labels == 0))
    n_chatter = int(np.sum(labels == 1))

    errors: dict = {}

    def _safe(name, fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            errors[name] = traceback.format_exc()
            print(f"WARNING: {name!r} failed:\n{errors[name]}", file=sys.stderr)
            return None

    # ---- Core: recurrent-Jacobian spectra, stable vs chatter ---------------- #
    res = dynamics.stable_vs_chatter_spectra(
        model, windows, taus, labels,
        reduce=args.reduce, t_stride=args.t_stride, layer=args.layer,
    )

    # ---- Figure (headline + research-update copy) --------------------------- #
    fig_path = out_dir / f"hidden_dynamics_{tag}.png"
    artifacts: dict = {}
    artifacts["distribution"] = _safe(
        "plot_distribution", lambda: dynamics.plot_spectral_distribution(res, str(fig_path))
    )
    if artifacts["distribution"]:
        assets = Path(args.assets_dir)
        assets.mkdir(parents=True, exist_ok=True)
        asset_path = assets / f"hidden_dynamics_{tag}.png"
        _safe("copy_asset", lambda: shutil.copyfile(fig_path, asset_path))
        artifacts["research_asset"] = str(asset_path)

    # ---- Optional slow / fixed-point probe ---------------------------------- #
    slow_summary = None
    if args.slow_points:
        def _slow():
            cell = model.encoder.layers[args.layer]
            # Clamp the input at the mean input of the chatter windows (a
            # representative operating point); zero the delayed input.
            x_bar = windows[labels == 1].reshape(-1, windows.shape[-1]).mean(axis=0)
            use_delay = (args.layer == 0) and model.encoder.use_delay
            x_delay = np.zeros_like(x_bar) if use_delay else None
            pts = dynamics.find_slow_points(
                cell, x_bar, h_dim=int(model.encoder.hidden_size),
                n_inits=int(args.slow_inits), steps=int(args.slow_steps),
                lr=0.05, seed=0, x_delay=x_delay,
            )
            best = pts[0]
            return {
                "n_points": len(pts),
                "best_speed": float(best["speed"]),
                "best_spectral_radius": float(best["spectral_radius"]),
                "speeds": [float(p["speed"]) for p in pts],
                "spectral_radii": [float(p["spectral_radius"]) for p in pts],
            }
        slow_summary = _safe("slow_points", _slow)

    # ---- Provenance + manifest ---------------------------------------------- #
    run_id = args.run_id or Path(args.checkpoint).resolve().parent.parent.name
    provenance = {
        "checkpoint": str(args.checkpoint),
        "preprocessing": str(pre_path) if pre_path else None,
        "run_id": run_id,
        "tag": tag,
        "val_auc": args.val_auc,
        "rendered_date": _now_date(),
        "git_sha": _git_sha(_ROOT),
        "n_windows": n_windows,
        "n_stable": n_stable,
        "n_chatter": n_chatter,
        "n_channels": int(model.input_size),
        "hidden_size": int(model.hidden_size),
        "num_layers": int(model.encoder.num_layers),
        "analysed_layer": int(args.layer),
        "use_delay": bool(model.encoder.use_delay),
        "pool": int(args.pool),
        "t_stride": int(args.t_stride),
        "reduce": args.reduce,
        "grid_size": int(args.grid_size),
        "spline_order": int(args.spline_order),
        "side_input": bool(getattr(model, "side_input", False)),
    }

    def _class_block(b: dict) -> dict:
        return {k: (float(v) if k != "n" else int(v)) for k, v in b.items() if k != "radii"}

    results = {
        "hypothesis": "chatter windows show larger leading |lambda| than stable "
                      "(internal dynamics cross the |lambda|=1 bifurcation)",
        "hypothesis_supported": bool(res["hypothesis_supported"]),
        "delta_mean": float(res["delta_mean"]),
        "stable": _class_block(res["stable"]),
        "chatter": _class_block(res["chatter"]),
        "radii_per_window": [float(x) for x in res["radii"]],
        "labels_per_window": [int(x) for x in res["labels"]],
    }
    if slow_summary is not None:
        results["slow_points"] = slow_summary

    manifest = {
        "provenance": provenance,
        "results": results,
        "artifacts": {k: v for k, v in artifacts.items() if v},
        "errors": errors,
    }
    manifest_path = out_dir / "hidden_dynamics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")

    print(f"Hidden-dynamics manifest -> {manifest_path}")
    print(f"  stable  mean |lambda| = {res['stable']['mean']:.4f} "
          f"(frac>1 = {res['stable']['frac_above_1']:.3f}, n={res['stable']['n']})")
    print(f"  chatter mean |lambda| = {res['chatter']['mean']:.4f} "
          f"(frac>1 = {res['chatter']['frac_above_1']:.3f}, n={res['chatter']['n']})")
    print(f"  delta_mean = {res['delta_mean']:+.4f}  "
          f"hypothesis_supported = {res['hypothesis_supported']}")
    if errors:
        print(f"WARNING: {len(errors)} step(s) failed: {sorted(errors)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
