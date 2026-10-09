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

Exp 1b (``--pairs`` / ``--all-gated``, 2026-10-05)
    Fold 4 has no single necessary cluster. Either none matters, or they cover
    for each other -- the singles cannot tell those apart, deleting two at once
    can. ``--pairs`` adds, per stratum and AFTER the single-mode arms (which
    therefore reproduce Exp 1 unchanged), every unordered pair of modes with a
    random arm matched on count and summed importance, and reads the
    interaction = pair drop - (sum of the two single drops measured in this
    same run). ``--all-gated`` deletes every gated edge against its matched
    random arm, as a reference. Neither has an other-modes control: at k=3 a
    pair leaves one cluster and all-gated leaves none. With neither flag the
    manifest is exactly the Exp 1 manifest.

``--reference-manifest``
    Checks that this run's single-mode memberships equal a landed manifest's,
    so a pair result can be tied to the Exp 1 clusters by evidence rather than
    by assumption. Recorded, never fatal.

Usage::

    python3 scripts/mode_ablation_phase3.py \
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \
        --features-csv data/metadata/features_datasets_121_125.csv \
        --npz-root phase2_signals --k 4
"""

from __future__ import annotations

import argparse
import itertools
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
    all_gated_verdict,
    combined_edge_sets,
    edges_are_zeroed,
    pair_verdict,
    zero_edges,
)
from pri2m.xai.sampling import (  # noqa: E402
    batch_to_model_device,
    build_labeled_batch,
    window_config_from_preprocessing,
)

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
    # eval_heldout_phase3.run nests its metrics: the AUC is at
    # result["metrics"]["auc"], NOT at the top level. Reading the top level
    # returns None for a run that in fact succeeded.
    auc = ((res or {}).get("metrics") or {}).get("auc")
    if auc is None:
        return None, f"evaluator returned no metrics.auc (keys: {sorted((res or {}))})"
    return float(auc), None


def mode_record(stratum: str, mode: int, sets: dict, arms: dict,
                verdict: dict) -> dict:
    """One mode's manifest entry, INCLUDING which edges each arm deleted.

    The k sweep found a group of about thirty edges at k=2 and again at k=3, and
    whether those are the same group could not be checked from the stored
    manifests: they recorded the count and the summed importance but never the
    membership. That is a gap in the record rather than in the analysis, and
    leaving it open means every future cross-k comparison is an inference from
    two summary numbers.
    """
    return {
        "stratum": stratum,
        "mode": int(mode),
        "n_edges": len(sets.get("mode_edges") or []),
        "summed_importance": sets.get("summed_importance"),
        "random_overlap_with_mode": sets.get("random_overlap_with_mode"),
        "arms": {k: {"auc": v[0], "error": v[1]} for k, v in arms.items()},
        "verdict": verdict,
        "edges": {k: sets.get(k) or []
                  for k in ("mode_edges", "random_matched", "other_modes_matched")},
    }


def combined_record(stratum: str, sets: dict, arms: dict, verdict: dict,
                    edge_key: str) -> dict:
    """One multi-mode deletion's manifest entry (a pair, or all gated edges).

    Same shape as :func:`mode_record`, membership included, with the deleted set
    stored under ``edge_key`` ("pair_edges" / "all_edges") so a reader cannot
    mistake it for a single mode. The other-modes set is not stored: it was
    never scored.
    """
    return {
        "stratum": stratum,
        "modes": [int(m) for m in sets.get("modes") or []],
        "n_edges": len(sets.get("mode_edges") or []),
        "summed_importance": sets.get("summed_importance"),
        "random_overlap_with_mode": sets.get("random_overlap_with_mode"),
        "arms": {k: {"auc": v[0], "error": v[1]} for k, v in arms.items()},
        "verdict": verdict,
        "edges": {edge_key: sets.get("mode_edges") or [],
                  "random_matched": sets.get("random_matched") or []},
    }


def _run_arms(model, ckpt, args, preprocessing, sets: dict, arm_map: dict,
              tag_prefix: str) -> dict:
    """Score each ``{arm_name: sets_key}`` with its edges zeroed.

    The same zero -> guard -> score sequence as the single-mode loop: an empty
    set or an ablation that did not apply is an arm with no AUC (UNDECIDED
    downstream), never a measured "no effect".
    """
    arms = {}
    for arm, key in arm_map.items():
        edges = sets.get(key) or []
        if not edges:
            arms[arm] = (None, "empty edge set")
            continue
        with zero_edges(model, edges):
            if not edges_are_zeroed(model, edges):
                arms[arm] = (None, "edges did not zero")
                continue
            arms[arm] = _score(ckpt, args, preprocessing, model,
                               f"{tag_prefix}_{arm}", args.output_dir)
    return arms


def _edge_keys(edges) -> set:
    return {(str(e["kan"]), int(e["out"]), int(e["in"])) for e in edges or []}


def reference_check(path, results: list[dict]) -> dict:
    """Compare this run's single-mode memberships with a landed manifest's.

    The pair arms are only interpretable against Exp 1 if the clusters they
    merge ARE the Exp 1 clusters. The eval path is deterministic, so they should
    be; this records whether they were. Membership is compared as sets of
    ``(kan, out, in)`` for the same stratum and mode label.

    Per mode ``identical`` is ``None`` when the reference cannot answer (no
    such mode, or a manifest from before membership was recorded -- the first
    fold2_w512 k=3 run is one). ``modes_identical`` is False if any mode
    differs, else None if any is undecided, else True. Nothing here raises: a
    bad reference is a recorded ``error``, not a lost eight-hour run.
    """
    out = {"manifest": str(path), "modes_identical": None, "per_mode": {},
           "error": None}
    try:
        ref = json.loads(pathlib.Path(path).read_text())
        ref_modes = {(str(r["stratum"]), int(r["mode"])): r
                     for r in ref.get("modes") or []}
        ref_ckpt = (ref.get("provenance") or {}).get("checkpoint")
        out["reference_checkpoint"] = ref_ckpt
        ref_keys = set(ref_modes)
        this_keys = {(str(r["stratum"]), int(r["mode"])) for r in results}
        for key in sorted(this_keys | ref_keys):
            this = next((r for r in results
                         if (str(r["stratum"]), int(r["mode"])) == key), None)
            entry = {"identical": None, "n_this": None, "n_reference": None,
                     "n_shared": None, "reason": None}
            r = ref_modes.get(key)
            mine = _edge_keys(((this or {}).get("edges") or {}).get("mode_edges"))
            if this is not None:
                entry["n_this"] = len(mine)
            if r is None:
                entry["reason"] = "mode not in reference"
            elif this is None:
                entry["reason"] = "mode not in this run"
                entry["identical"] = False
            else:
                theirs_raw = (r.get("edges") or {}).get("mode_edges")
                if not theirs_raw:
                    entry["reason"] = "reference records no membership"
                else:
                    theirs = _edge_keys(theirs_raw)
                    entry["n_reference"] = len(theirs)
                    entry["n_shared"] = len(mine & theirs)
                    entry["identical"] = bool(mine == theirs)
            out["per_mode"][f"{key[0]}:{key[1]}"] = entry
        verdicts = [v["identical"] for v in out["per_mode"].values()]
        if any(v is False for v in verdicts):
            out["modes_identical"] = False
        elif verdicts and all(v is True for v in verdicts):
            out["modes_identical"] = True
    except Exception as exc:  # noqa: BLE001 - a sanity check must not kill the run
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["modes_identical"] = None
    return out


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
    ap.add_argument("--pairs", action="store_true",
                    help="Exp 1b: after the single-mode arms, delete every "
                         "unordered pair of modes (+ matched random arm) and read "
                         "the interaction. Adds 2*C(k,2) evaluations per stratum.")
    ap.add_argument("--all-gated", action="store_true",
                    help="Exp 1b: also delete every gated edge (+ matched random "
                         "arm). Reference only. Adds 2 evaluations per stratum.")
    ap.add_argument("--reference-manifest", default=None,
                    help="A landed mode_ablation manifest; record whether this "
                         "run's single-mode memberships equal its. Never fatal.")
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
    # cfg is REQUIRED: passing None reaches Phase3SequenceDataset and dies on
    # cfg.channels. Take the geometry from the checkpoint's own preprocessing so
    # the windows analysed match the ones the network was trained on.
    cfg = window_config_from_preprocessing(preprocessing)
    batch, _labels = build_labeled_batch(
        args.table, args.npz_root, preprocessing, args.n_windows,
        cfg, seed=args.seed,
    )
    # The baseline eval above already ran, and eval_heldout_phase3.run moves the
    # model to its device as a side effect. build_labeled_batch always returns
    # CPU tensors, so without this the next model(batch) dies on a device
    # mismatch. Read the device from the model, not from --device.
    batch = batch_to_model_device(batch, model)
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
    pair_results = []
    all_gated_results = []
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
            results.append(mode_record(stratum, mode, sets, arms, verdict))
            print(f"  {stratum} mode {mode}: n={results[-1]['n_edges']:4d}  "
                  f"drop={verdict['mode_drop']}  "
                  f"random={verdict['random_matched_drop']}  "
                  f"other={verdict['other_modes_drop']}  "
                  f"claimable={verdict['claimable']}")

        if not (args.pairs or args.all_gated):
            continue
        # -- 3b. Exp 1b: several modes at once, after the singles ------------ #
        # Run strictly after the single-mode arms, so those reproduce Exp 1 and
        # the interaction reads drops measured on this run, not copied from an
        # older manifest. A single drop needs only the baseline and the mode arm,
        # so it is taken from the arm directly: ablation_verdict blanks mode_drop
        # whenever ANY of its three arms failed, and an other-modes failure
        # should not cost the pair its interaction.
        single_drops = {}
        for rec in results:
            if rec["stratum"] != stratum:
                continue
            auc = rec["arms"].get("mode_edges", {}).get("auc")
            single_drops[int(rec["mode"])] = (None if auc is None
                                              else float(baseline) - float(auc))
        present = sorted({int(v) for v in np.asarray(labels).tolist()})
        population = bundles[stratum]

        if args.pairs:
            for a, b in itertools.combinations(present, 2):
                sets = combined_edge_sets(norm, labels, [a, b], seed=int(args.seed),
                                          population=population)
                arms = _run_arms(model, ckpt, args, preprocessing, sets,
                                 {"pair_edges": "mode_edges",
                                  "random_matched": "random_matched"},
                                 f"{stratum}_p{a}{b}")
                verdict = pair_verdict(
                    baseline_auc=baseline,
                    pair_auc=arms["pair_edges"][0],
                    random_matched_auc=arms["random_matched"][0],
                    single_drops={a: single_drops.get(a), b: single_drops.get(b)},
                    floor=float(args.floor),
                )
                pair_results.append(
                    combined_record(stratum, sets, arms, verdict, "pair_edges"))
                print(f"  {stratum} pair {a}+{b}: n={pair_results[-1]['n_edges']:4d}  "
                      f"drop={verdict['pair_drop']}  "
                      f"random={verdict['random_matched_drop']}  "
                      f"interaction={verdict['interaction']} "
                      f"({verdict['interaction_class']})  "
                      f"claimable={verdict['claimable']}")

        if args.all_gated:
            sets = combined_edge_sets(norm, labels, present, seed=int(args.seed),
                                      population=population)
            arms = _run_arms(model, ckpt, args, preprocessing, sets,
                             {"all_edges": "mode_edges",
                              "random_matched": "random_matched"},
                             f"{stratum}_all")
            verdict = all_gated_verdict(
                baseline_auc=baseline,
                all_auc=arms["all_edges"][0],
                random_matched_auc=arms["random_matched"][0],
                floor=float(args.floor),
            )
            all_gated_results.append(
                combined_record(stratum, sets, arms, verdict, "all_edges"))
            print(f"  {stratum} all gated: n={all_gated_results[-1]['n_edges']:4d}  "
                  f"drop={verdict['all_drop']}  "
                  f"random={verdict['random_matched_drop']}  "
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
    # Exp 1b keys exist ONLY when asked for, so a run without the flags writes
    # the Exp 1 manifest key for key and the landed files stay reproducible.
    if args.pairs or args.all_gated:
        manifest["provenance"]["pairs"] = bool(args.pairs)
        manifest["provenance"]["all_gated"] = bool(args.all_gated)
        manifest["pairs"] = pair_results
        manifest["all_gated"] = all_gated_results
        manifest["exp1b_note"] = (
            "Pair claimable = drop > floor AND drop - random_drop > floor. "
            "interaction = pair drop - (sum of this run's two single drops): "
            "super_additive if > floor, sub_additive if < -floor, else additive. "
            "No other-modes control: at k=3 a pair leaves one cluster.")
    if args.reference_manifest:
        manifest["reference_check"] = reference_check(args.reference_manifest,
                                                      results)
        rc = manifest["reference_check"]
        print(f"reference check vs {rc['manifest']}: "
              f"modes_identical={rc['modes_identical']}"
              + (f"  error={rc['error']}" if rc["error"] else ""))
    dest = args.output_dir / f"mode_ablation_{args.tag}.json"
    dest.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
