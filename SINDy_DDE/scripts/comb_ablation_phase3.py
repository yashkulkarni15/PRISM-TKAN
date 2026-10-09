#!/usr/bin/env python3
"""Delete the comb-carrying delay pairs and measure the damage.

Update 15 Section 2 reads the delay coupling as a delay-and-add comb tuned to the
tooth-passing frequency, and Section 3 confirms that reading by driving the real
learned curves with a tone sweep. Both are descriptions, and Section 4 says so:
the comb adds a mechanism, not a line of evidence, because nothing removes those
pairs and shows the network misses them. This runner is that removal.

Three arms, all mandatory:

``comb_pairs``
    both edges of every pair clearing the proportionality gate -- exactly the
    population Section 2's numbers describe, selected by the same predicate.
``random_matched``
    the same number of edges carrying the same total importance, drawn from the
    wider population. Answers "is this about these edges, or about deleting that
    much?"
``nonproportional_matched``
    the sharp control. The same count and the same total importance, drawn only
    from delay pairs that fall BELOW the gate: structurally identical, on the
    same delayed inputs, simply not carrying the comb. If the comb is what
    matters rather than delay coupling in general, this is the arm that shows it.

A result is claimable only when the comb arm beats BOTH controls by more than
the repeatability floor, the same rule the mode ablation is judged by.

Usage::

    python3 scripts/comb_ablation_phase3.py \
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \
        --regenerative results/phase3/xai/regenerative/fold2_w512/regenerative.json \
        --features-csv data/metadata/features_datasets_121_125.csv \
        --npz-root phase2_signals
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_heldout_phase3 as EH  # noqa: E402
import xai_phase3  # noqa: E402
from pri2m.xai.comb import DEFAULT_CORR_THRESHOLD  # noqa: E402
from pri2m.xai.activations import collect_activations  # noqa: E402
from pri2m.xai.comb_ablation import (  # noqa: E402
    build_controls,
    check_population_is_wider,
    population_from_ranked,
    select_comb_edges,
    summed_importance,
)
from pri2m.xai.faithfulness import rank_edges  # noqa: E402
from pri2m.xai.mode_ablation import (  # noqa: E402
    ablation_verdict,
    edges_are_zeroed,
    zero_edges,
)
from pri2m.xai.sampling import (  # noqa: E402
    batch_to_model_device,
    build_labeled_batch,
    window_config_from_preprocessing,
)

DEFAULT_FLOOR = 0.0092                       # Update 14's repeatability floor
DEFAULT_OUT = pathlib.Path("results/phase3/xai/comb_ablation")


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
    # The AUC is nested at result["metrics"]["auc"]; the top level has no "auc".
    auc = ((res or {}).get("metrics") or {}).get("auc")
    if auc is None:
        return None, f"evaluator returned no metrics.auc (keys: {sorted((res or {}))})"
    return float(auc), None


def _network_population(model, args, preprocessing) -> list[dict]:
    """EVERY edge in the network, with importance, as the random arm's pool.

    Deliberately not the delay-pair edges. The comb set and the below-gate set
    exhaust that smaller population, so drawing from it leaves exactly the
    below-gate edges and collapses the two controls into one -- which is what the
    first run of this script did on four folds without erroring.
    """
    cfg = window_config_from_preprocessing(preprocessing)
    batch, _labels = build_labeled_batch(args.table, args.npz_root, preprocessing,
                                         args.n_windows, cfg, seed=args.seed)
    batch = batch_to_model_device(batch, model)
    return population_from_ranked(rank_edges(collect_activations(model, batch)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--regenerative", required=True,
                    help="The fold's regenerative.json; its rows carry the "
                         "per-pair corr/gain the comb reading gates on.")
    ap.add_argument("--features-csv", required=True,
                    help="Held-out features CSV (the 121-125 path).")
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--table", default="results/phase3/phase3_dataset.csv",
                    help="In-distribution table the activation batch is drawn "
                         "from; scoring stays on the held-out features CSV.")
    ap.add_argument("--preprocessing", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=256,
                    help="Windows used to rank edge importance for the controls.")
    ap.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD,
                    help="Proportionality gate. MUST match the value Section 2 "
                         "reports (0.9) or the arm describes a different set.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trials", type=int, default=200,
                    help="Draws per matched control; best match on summed importance.")
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    ap.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--tag", default="fold2_w512")
    args = ap.parse_args(argv)

    regen = json.loads(pathlib.Path(args.regenerative).read_text())
    rows = regen.get("rows") or []
    if not rows:
        print("regenerative manifest carries no rows", file=sys.stderr)
        return 1

    comb_edges = select_comb_edges(rows, corr_threshold=args.corr_threshold)
    below = select_comb_edges(rows, corr_threshold=args.corr_threshold,
                              below_threshold=True)
    if not comb_edges:
        print("no pair clears the gate; nothing to ablate", file=sys.stderr)
        return 1

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"comb_ablation_{args.tag}.json"

    # Provenance written BEFORE the run, so a crash leaves a record of what was
    # attempted rather than nothing.
    manifest = {
        "analysis": "Update 15 section 6: causal test of the comb reading",
        "provenance": {
            "complete": False,
            "checkpoint": args.checkpoint,
            "regenerative": args.regenerative,
            "corr_threshold": float(args.corr_threshold),
            "seed": int(args.seed), "n_trials": int(args.n_trials),
            "floor": float(args.floor), "tag": args.tag,
            "n_rows": len(rows),
            "n_comb_edges": len(comb_edges),
            "n_below_gate_edges": len(below),
        },
        "note": ("A pair clears the gate at abs(corr) >= threshold on a usable "
                 "row, the same predicate comb._stats applies, so this deletes "
                 "exactly the population Section 2 describes."),
    }
    # Resolved exactly as heldout_pruning_phase3 does: _load_model and
    # window_config_from_preprocessing both want the LOADED dict, not a path.
    # The first version of this called a function that does not exist
    # (xai_phase3._resolve_preprocessing); the real name is _find_preprocessing,
    # and four folds died on it.
    ckpt = pathlib.Path(args.checkpoint)
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}", file=sys.stderr)
        return 2
    pre_path = (pathlib.Path(args.preprocessing) if args.preprocessing
                else ckpt.parent.parent / "preprocessing.json")
    preprocessing = json.loads(pre_path.read_text()) if pre_path.exists() else None

    load_args = argparse.Namespace(input_size=None, hidden_size=None,
                                   grid_size=5, spline_order=3)
    model = xai_phase3._load_model(str(ckpt), load_args, preprocessing)
    model.eval()

    # Controls are built BEFORE the baseline evaluation. Collecting activations
    # over a few hundred windows is minutes; the held-out evaluation is ~30k
    # windows, so a control that cannot be drawn must be discovered here rather
    # than an hour later. Same lesson as the determinism startup guard.
    population = _network_population(model, args, preprocessing)
    check_population_is_wider(population, target=comb_edges, below_gate=below)
    manifest["provenance"]["n_population_edges"] = len(population)

    controls, control_errors = build_controls(
        population=population, below_gate=below, target=comb_edges,
        seed=args.seed, n_trials=args.n_trials)
    manifest["provenance"]["control_errors"] = control_errors
    for name, why in control_errors.items():
        print(f"WARNING: control {name!r} could not be built: {why}", file=sys.stderr)
        print("         that arm will be reported absent and the verdict will be "
              "UNDECIDED, which is NOT a pass.", file=sys.stderr)
    if not controls:
        manifest["provenance"]["abandoned"] = "no control could be built"
        dest.write_text(json.dumps(manifest, indent=2))
        print("no control could be built; refusing to score an uncontrolled arm",
              file=sys.stderr)
        return 1
    dest.write_text(json.dumps(manifest, indent=2))

    baseline, err = _score(args.checkpoint, args, preprocessing, model,
                           "baseline", out_dir)
    if baseline is None:
        print(f"Baseline evaluation failed: {err}", file=sys.stderr)
        return 1
    manifest["baseline_auc"] = baseline
    print(f"baseline held-out AUC = {baseline:.4f}  (this fold's own)")

    arms: dict = {}
    for name, edges in (("comb_pairs", comb_edges), *controls.items()):
        with zero_edges(model, edges):
            # An ablation that silently failed to apply would report the pairs
            # as unnecessary, the most damaging false result available here.
            if not edges_are_zeroed(model, edges):
                arms[name] = (None, "edges did not zero")
                continue
            arms[name] = _score(args.checkpoint, args, preprocessing, model,
                                name, out_dir)
        auc = arms[name][0]
        print(f"  {name:<26} n={len(edges):4d} "
              f"summed_importance={summed_importance(edges):8.3f} "
              f"auc={'None' if auc is None else format(auc, '.4f')}")

    # An arm that could not be built passes through as None, which
    # ablation_verdict turns into claimable=None. UNDECIDED, never a pass.
    verdict = ablation_verdict(
        baseline_auc=baseline,
        mode_auc=arms["comb_pairs"][0],
        random_matched_auc=arms.get("random_matched", (None, None))[0],
        other_modes_auc=arms.get("nonproportional_matched", (None, None))[0],
        floor=float(args.floor),
    )
    manifest["arms"] = {
        k: {"auc": v[0], "error": v[1],
            "n_edges": len(comb_edges if k == "comb_pairs" else controls[k]),
            "summed_importance": summed_importance(
                comb_edges if k == "comb_pairs" else controls[k])}
        for k, v in arms.items()
    }
    # Edge membership is recorded, not just counted: without it nobody can check
    # afterwards which pairs this deleted.
    manifest["edges"] = {"comb_pairs": comb_edges, **controls}
    manifest["verdict"] = verdict
    manifest["provenance"]["complete"] = True
    dest.write_text(json.dumps(manifest, indent=2))

    print(f"\n  drop={verdict['mode_drop']}  vs random={verdict['random_matched_drop']}"
          f"  vs non-proportional={verdict['other_modes_drop']}"
          f"  CLAIMABLE={verdict['claimable']}")

    written = json.loads(dest.read_text())
    assert written["provenance"]["complete"] is True, "manifest not marked complete"
    assert written["arms"], "manifest has no arms"
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
