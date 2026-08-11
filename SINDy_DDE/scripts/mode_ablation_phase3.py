#!/usr/bin/env python3
"""U14-B step 8: remove each activation mode's edges and measure the damage.

Update 14 section 4 showed that trained networks are more self-consistent in the
shapes of their activation functions than randomly initialised ones, at every
tested setting, and could not turn that into a verdict because the pass/fail
flips with the number of prototypes requested. This runner answers the question
causally instead: zero the edges assigned to a mode and see what held-out
accuracy does. That measurement does not care whether the grouping reproduces.

Three arms per mode, all from ``mode_edge_sets`` and all mandatory:

``mode``
    every edge assigned to the mode.
``random_matched``
    the same number of edges, matched on summed importance, drawn from the
    wider population. Answers "what does removing this much arbitrary capacity
    cost?", without which the mode arm means nothing.
``other_modes``
    an equal-size set drawn from the other modes. Answers "is this mode special
    among modes?"

A mode is only reported as claimable when it beats BOTH controls by more than
the repeatability floor. Update 14 measured that floor at 0.0092 AUC, a lower
bound, and the held-out baseline's own fold spread is 0.013 -- so a small effect
here is expected to be unclaimable, and that is a result rather than a failure.

Usage::

    python3 scripts/mode_ablation_phase3.py \
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \
        --features-csv data/metadata/features_121_155.csv \
        --npz-root phase2_signals --k 4
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

import eval_heldout_phase3 as EH  # noqa: E402
import xai_phase3  # noqa: E402
from pri2m.xai import activation_modes as AM  # noqa: E402
from pri2m.xai.mode_ablation import (  # noqa: E402
    ablation_verdict,
    edges_are_zeroed,
    zero_edges,
)
from pri2m.xai.sampling import build_labeled_batch  # noqa: E402

# Update 14's measured lower bound on run-to-run drift in selected AUC.
DEFAULT_FLOOR = 0.0092
DEFAULT_OUT = pathlib.Path("results/phase3/xai/mode_ablation")


def _score(checkpoint, args, preprocessing, model, tag: str, out_dir: pathlib.Path):
    """Held-out AUC for ``model``, via the existing held-out evaluator."""
    try:
        res = EH.run(
            checkpoint=str(checkpoint),
            features_csv=args.features_csv,
            npz_root=args.npz_root,
            out=str(out_dir / f"_eval_{tag}.json"),
            preprocessing=args.preprocessing,
            device=args.device,
            batch_size=args.batch_size,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 - an arm that dies must not be a pass
        return None, f"{type(exc).__name__}: {exc}"
    auc = (res or {}).get("auc")
    return (float(auc) if auc is not None else None), None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features-csv", required=True,
                    help="Held-out features CSV (the 121-155 path).")
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--table", default="results/phase3/phase3_dataset.csv",
                    help="Training table, used only to build the activation bundle.")
    ap.add_argument("--preprocessing", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=256,
                    help="Windows used to measure edge importance for clustering.")
    ap.add_argument("--k", type=int, default=4, help="Number of activation modes.")
    ap.add_argument("--strata", default="encoder.L0.x_delay",
                    help="Comma-separated strata to ablate, or 'all'. Cost is "
                         "1 + 3*k evaluations PER stratum, so the default is the "
                         "delayed-channel stratum alone: it is the one the delay "
                         "story turns on, and 'all' triples the bill.")
    ap.add_argument("--gate-quantile", type=float, default=0.9,
                    help="Importance quantile above which an edge is kept before "
                         "clustering. Matches the U14-B runner's default, so the "
                         "modes ablated here are the modes that update reports.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                    help="Smallest AUC difference this pipeline can resolve.")
    ap.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--tag", default="fold2")
    args = ap.parse_args(argv)

    ckpt = pathlib.Path(args.checkpoint)
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}", file=sys.stderr)
        return 2
    pre_path = pathlib.Path(args.preprocessing) if args.preprocessing else (
        ckpt.parent.parent / "preprocessing.json")
    preprocessing = json.loads(pre_path.read_text()) if pre_path.exists() else None

    load_args = argparse.Namespace(input_size=None, hidden_size=None,
                                   grid_size=5, spline_order=3)
    model = xai_phase3._load_model(str(ckpt), load_args, preprocessing)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. baseline, before anything is removed --------------------------- #
    baseline, err = _score(ckpt, args, preprocessing, model, "baseline", args.output_dir)
    if baseline is None:
        print(f"Baseline evaluation failed: {err}", file=sys.stderr)
        return 1
    print(f"baseline held-out AUC = {baseline:.4f}")

    # -- 2. cluster the activation shapes into modes, per stratum ---------- #
    # fold_bundles returns one bundle PER STRATUM, not one bundle: hidden-fed
    # edges are only comparable as a shape population, never by identity, so
    # pooling them with the physical-channel edges would be meaningless.
    batch, _labels = build_labeled_batch(
        args.table, args.npz_root, preprocessing, args.n_windows,
        cfg=None, seed=args.seed,
    )
    bundles = AM.fold_bundles(model, batch, n=args.n_windows)
    wanted = (sorted(bundles) if args.strata.strip() == "all"
              else [s.strip() for s in args.strata.split(",") if s.strip()])
    missing = [s for s in wanted if s not in bundles]
    if missing:
        print(f"Strata not present in this model: {missing}; "
              f"available: {sorted(bundles)}", file=sys.stderr)
        return 2

    # -- 3. one ablation per mode per stratum, each against its two controls - #
    results = []
    for stratum in wanted:
        gated = AM.gate_by_importance(bundles[stratum],
                                      quantile=float(args.gate_quantile))
        norm = AM.normalise_curves(gated)
        protos = AM.fit_prototypes(norm["curves"], int(args.k), seed=int(args.seed))
        # assign_to_prototypes returns (labels, distances).
        labels, _dist = AM.assign_to_prototypes(norm["curves"], protos)

        for mode in sorted({int(v) for v in np.asarray(labels).tolist()}):
            sets = AM.mode_edge_sets(norm, labels, mode=mode, seed=int(args.seed),
                                     population=bundles[stratum])
            arms = {}
            for arm in ("mode_edges", "random_matched", "other_modes_matched"):
                edges = sets.get(arm) or []
                if not edges:
                    arms[arm] = (None, "empty edge set")
                    continue
                with zero_edges(model, edges):
                    # An ablation that silently failed to apply would report the
                    # mode as unnecessary, the most damaging false result here.
                    if not edges_are_zeroed(model, edges):
                        arms[arm] = (None, "edges did not zero")
                        continue
                    arms[arm] = _score(ckpt, args, preprocessing, model,
                                       f"{stratum}_m{mode}_{arm}", args.output_dir)

            verdict = ablation_verdict(
                baseline_auc=baseline,
                mode_auc=arms["mode_edges"][0],
                random_matched_auc=arms["random_matched"][0],
                other_modes_auc=arms["other_modes_matched"][0],
                floor=float(args.floor),
            )
            results.append({
                "stratum": stratum,
                "mode": mode,
                "n_edges": len(sets.get("mode_edges") or []),
                "summed_importance": sets.get("summed_importance"),
                "random_overlap_with_mode": sets.get("random_overlap_with_mode"),
                "arms": {k: {"auc": v[0], "error": v[1]} for k, v in arms.items()},
                "verdict": verdict,
            })
            print(f"  {stratum} mode {mode}: n={results[-1]['n_edges']:4d}  "
                  f"drop={verdict['mode_drop']}  "
                  f"random={verdict['random_matched_drop']}  "
                  f"other={verdict['other_modes_drop']}  "
                  f"claimable={verdict['claimable']}")

    manifest = {
        "analysis": "U14-B step 8: causal ablation of activation modes",
        "provenance": {
            "checkpoint": str(ckpt), "k": int(args.k), "seed": int(args.seed),
            "strata": wanted, "gate_quantile": float(args.gate_quantile),
            "n_windows": int(args.n_windows), "floor": float(args.floor),
            "features_csv": args.features_csv,
        },
        "baseline_auc": baseline,
        "modes": results,
        # Stated so a reader does not have to infer it: the held-out baseline's
        # own fold spread is 0.013, larger than the floor, so a small effect is
        # expected to be unclaimable and that is a result rather than a failure.
        "note": ("A mode is claimable only when it beats BOTH controls by more "
                 "than the floor. Update 14 measured the floor at 0.0092 AUC as "
                 "a LOWER bound, and the held-out baseline's fold spread is "
                 "0.013."),
        "complete": True,
    }
    dest = args.output_dir / f"mode_ablation_{args.tag}.json"
    dest.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
