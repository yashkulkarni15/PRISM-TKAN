#!/usr/bin/env python3
"""U14-B: does the network learn recurring activation shapes, or does every fold invent its own?

Each KAN edge carries a learned 1-D activation. If the five folds independently
converge on a small set of shapes, that is a statement about what the architecture
extracts from milling data. If they do not, the "modes" are a clustering artefact.

WHAT THIS RUNNER WILL NOT LET YOU CONCLUDE.
The absolute agreement score is NOT interpretable on its own. On synthetic
fixtures a randomly initialised network already scores ~0.63 where 1.0 is perfect,
because every curve is n_coeff spline coefficients over one shared basis and so
cannot be far from every prototype. Only the margin over the untrained null is a
finding. That is why `--n-seeds` defaults to 32 rather than 5: the null's spread
is the denominator of the whole test, and untrained replicates are cheap.

The verdict rule mirrors `tsr_control_phase3.py`, which is the precedent that
killed the TSR delay-lock: a threshold fixed before the numbers exist, and an
UNDECIDED state that is explicitly NOT a pass.

SWEEP k. Set k above the number of shapes actually present and k-means splits a
true cluster, collapsing identity agreement toward chance while prototype_fit
stays near 1.0. A single k is not a result; `--k-sweep 2,3,4,5,6` runs the lot and
writes every k to the manifest.

    python scripts/activation_modes_phase3.py \
        --checkpoints results/phase3/runs/<run0>/checkpoints/best_auc.pt,... \
        --npz-root phase2_signals --table results/phase3/phase3_dataset.csv \
        --output-dir results/phase3/xai/activation_modes \
        --n-windows 256 --k-sweep 2,3,4,5,6 --n-seeds 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import whitebox_phase3 as W  # noqa: E402
import xai_phase3  # noqa: E402
from pri2m.data.phase3_sequence import WindowConfig  # noqa: E402
from pri2m.xai import activation_modes as am  # noqa: E402
from pri2m.xai.sampling import CHANNELS, build_labeled_batch  # noqa: E402


def _provenance(*, checkpoints, tag, n_windows, n_seeds, k_sweep, gate_quantile,
                threshold, min_seeds, component, n_points, sample_seed):
    """Assembled BEFORE the run, not after.

    `tsr_control_phase3.py` learned this the expensive way: provenance built at
    the end hid a crash behind a 12-hour timeout, and the run could never have
    written its manifest even if it had finished. Build it first so a broken
    field fails in seconds.
    """
    return {
        "checkpoints": [str(c) for c in checkpoints],
        "tag": tag,
        "n_windows": int(n_windows),
        "n_untrained_seeds": int(n_seeds),
        "k_sweep": list(k_sweep),
        "gate_quantile": gate_quantile,
        "threshold": float(threshold),
        "min_seeds": int(min_seeds),
        "component": component,
        "n_points": int(n_points),
        "sample_seed": int(sample_seed),
        "rendered_date": W._now_date(),
        "git_sha": W._git_sha(W._ROOT),
    }


def _progress(stage: str, done: int, total: int) -> None:
    """Progress line for `run_activation_mode_analysis`.

    The signature is FIXED by `activation_modes._tick`, which calls
    `progress(stage, done, total)`. This lived inside `main()` as a one-argument
    callback and blew up on the first real submission, after loading all five
    checkpoints -- an expensive way to discover a three-line mismatch. It is
    module-level now so `tests/test_activation_modes_runner.py` can bind the real
    function to the real analysis and fail here instead of on the cluster.
    """
    print(f"    [{done}/{total}] {stage}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", required=True,
                   help="Comma-separated best_auc.pt paths, one per fold")
    p.add_argument("--npz-root", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--preprocessing", default=None,
                   help="Comma-separated, one per checkpoint; blank -> auto-discover each")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-windows", type=int, default=256)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--window-length", type=int, default=1024)
    p.add_argument("--window-stride", type=int, default=512)
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--n-points", type=int, default=64,
                   help="Sweep points per edge curve")
    p.add_argument("--component", default="spline", choices=("spline", "full", "base"),
                   help="Cluster on the spline residual; the base path is a "
                        "one-parameter family and carries no shape information")
    p.add_argument("--gate-quantile", type=float, default=0.9)
    p.add_argument("--gate-min-importance", type=float, default=None,
                   help="Absolute importance floor INSTEAD of a quantile. A quantile "
                        "passes the same fraction of trained and untrained edges by "
                        "construction, which hides exactly the arm mismatch that "
                        "matched-N exists to expose. Prefer this when comparing arms.")
    p.add_argument("--k-sweep", default="2,3,4,5,6",
                   help="Comma-separated k values. A single k is not a result.")
    p.add_argument("--n-seeds", type=int, default=32, help="Untrained replicates")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--min-seeds", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    # xai_phase3._load_model reads these off the namespace.
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--tag", default=None)
    args = p.parse_args(argv)

    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    if len(ckpts) < 2:
        p.error("need at least two folds to measure cross-fold agreement")
    pres = [x.strip() or None for x in args.preprocessing.split(",")] if args.preprocessing else [None] * len(ckpts)
    if len(pres) != len(ckpts):
        p.error(f"--preprocessing has {len(pres)} entries for {len(ckpts)} checkpoints")
    k_sweep = [int(k) for k in args.k_sweep.split(",") if k.strip()]
    if not k_sweep:
        p.error("--k-sweep parsed to nothing")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or out_dir.name
    path = out_dir / "activation_modes_manifest.json"

    provenance = _provenance(
        checkpoints=ckpts, tag=tag, n_windows=args.n_windows, n_seeds=args.n_seeds,
        k_sweep=k_sweep, gate_quantile=args.gate_quantile, threshold=args.threshold,
        min_seeds=args.min_seeds, component=args.component, n_points=args.n_points,
        sample_seed=args.sample_seed,
    )

    cfg = WindowConfig(length=int(args.window_length),
                       stride=int(args.window_stride), channels=CHANNELS)

    models_by_fold, batches_by_fold = {}, {}
    for i, (ckpt, pre) in enumerate(zip(ckpts, pres)):
        pre_path = pre or xai_phase3._find_preprocessing(ckpt)
        if not pre_path:
            raise SystemExit(f"no preprocessing.json for {ckpt}; pass --preprocessing")
        preprocessing = json.loads(Path(pre_path).read_text(encoding="utf-8"))
        model = xai_phase3._load_model(ckpt, args, preprocessing)
        model.pool = int(args.pool)
        model.eval()
        batch, _labels = build_labeled_batch(
            args.table, args.npz_root, preprocessing, int(args.n_windows), cfg,
            seed=int(args.sample_seed),
        )
        models_by_fold[f"fold{i}"] = model
        batches_by_fold[f"fold{i}"] = batch
        print(f"  loaded fold{i}: {Path(ckpt).parent.parent.name}", flush=True)

    print(f"\n{len(models_by_fold)} folds x {args.n_windows} windows, "
          f"{args.n_seeds} untrained seeds, k in {k_sweep}", flush=True)

    # One entry per k, written after EACH k so a wall-clock kill leaves a
    # readable partial sweep rather than nothing. Same lesson as U12-B.
    by_k = {}
    for k in k_sweep:
        print(f"\n=== k = {k} ===", flush=True)
        res = am.run_activation_mode_analysis(
            models_by_fold, batches_by_fold,
            n_seeds=int(args.n_seeds), k=int(k), n_points=int(args.n_points),
            component=args.component,
            gate_quantile=(None if args.gate_min_importance is not None
                           else float(args.gate_quantile)),
            gate_min_importance=args.gate_min_importance,
            threshold=float(args.threshold), min_seeds=int(args.min_seeds),
            seed=int(args.seed), progress=_progress,
        )
        by_k[str(k)] = res
        path.write_text(json.dumps(
            {"provenance": {**provenance, "complete": False},
             "caveats": list(am.CAVEATS), "by_k": by_k},
            indent=2, default=float), encoding="utf-8")
        head = res.get("headline", "(no headline)")
        print(f"  -> k={k} done; partial manifest written. {head}", flush=True)

    path.write_text(json.dumps(
        {"provenance": {**provenance, "complete": True},
         "caveats": list(am.CAVEATS), "by_k": by_k},
        indent=2, default=float), encoding="utf-8")

    # Assert the artifact rather than announcing it. Two wrappers in this project
    # exited 0 while skipping their science and printed a path that was never written.
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"ERROR: manifest not written: {path}")

    print(f"\nmanifest -> {path}  ({path.stat().st_size} bytes)")
    print("\nk    verdict                       trained   untrained p95")
    for k in k_sweep:
        v = by_k[str(k)].get("verdict") or {}
        art = v.get("artifact_suspected")
        state = ("UNDECIDED -- NOT a pass" if art is None
                 else "ARTIFACT SUSPECTED" if art else "MODES SURVIVE THE NULL")
        print(f"{k:<4} {state:<29} {v.get('trained_score')}   {v.get('untrained_reference')}")
    print("\nRead the margin, never the trained score alone: an untrained network "
          "already scores ~0.63 on this measure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
