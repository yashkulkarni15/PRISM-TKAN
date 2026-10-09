#!/usr/bin/env python3
"""Exp 2 step 3: does the necessary delay cluster recur across seeds, by FUNCTION?

WHAT IT MEASURES
    Each network comes with a mode-ablation manifest naming its causally
    necessary cluster: the union of ``mode_edges`` over the modes whose
    ``verdict.claimable`` is True (stratum ``encoder.L0.x_delay``). For every
    pair of networks this runner asks whether A's cluster has a functional
    counterpart in B's cluster, three ways:

    (a) channel mix -- which delayed channels each cluster lives on (descriptive);
    (b) curve shape -- per A edge, the best centred signed cosine against B's
        cluster edges on the SAME input channel; S = mean best match. Controls:
        an untrained null (B re-initialised ``--n-untrained`` times, the same
        edge IDs re-read) and a random-matched arm (A's count-and-importance-
        matched random set in place of A's cluster). Also, secondarily, S of A's
        cluster against B's whole gated x_delay pool (top-decile importance):
        "the function exists in B but was not found necessary there";
    (c) aligned ID overlap -- B's hidden units are matched to A's by Hungarian
        assignment on |corr| of layer-0 hidden-state traces over the same
        windows; B's ``out`` indices are mapped through it and only then are
        edge IDs compared, against ``--n-perm`` random relabellings.

WHY EDGE-ID OVERLAP WAS INVALID
    An edge ID is (kan, out, in). ``in`` is a physical channel and is comparable
    across networks; ``out`` is a hidden unit, and hidden units are permutation
    symmetric -- relabelling them consistently leaves the function unchanged --
    so two independently trained networks need not put the same unit at the same
    index (the KAN 2.0 point: compare KANs by function, not position). The old
    cross-fold ID overlap counted index coincidences. Worse, all five original
    folds were trained with seed 0 and therefore from the IDENTICAL
    initialisation, so whatever IDs they shared may simply have been inherited
    from init. Exp 2 retrains from seeds 1-3 to break that.

PRE-COMMITTED RULE (fixed 2026-10-05, before any number existed)
    Direction A->B recurs iff S > p95(untrained null) AND S - S_random > 0.
    UNDECIDED (None, never False) when either network has no claimable cluster
    (reason ``no_necessary_cluster:<name>`` -- itself a reported result: "the
    necessary cluster did not come back"), when S or S_random is unmeasurable,
    or when fewer than ``--min-null`` null values exist. A PAIR recurs iff both
    directions recur (conservative); the per-fold summary reads "k of n
    within-fold pairs recur". Cross-fold pairs are reported separately
    (``cross_fold: true``) and never enter the per-fold count.

SUBMIT (Great Lakes, after the seed retrains AND their mode ablations land)
    NETWORKS="f0s0=<ckpt>=<manifest> f0s1=... f2s0=..." \\
      sbatch --export=ALL slurm/phase3_recurrence.sbatch
    The sbatch header carries the full fold 0 / fold 2 command.

Local::

    python scripts/recurrence_phase3.py \\
        --network f0s0=results/phase3/runs/<run>/checkpoints/best_auc.pt=results/phase3/xai/mode_ablation/mode_ablation_fold0_k3.json \\
        --network f0s1=... --table results/phase3/phase3_dataset.csv \\
        --npz-root phase2_signals --tag fold0_fold2
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import re
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xai_phase3  # noqa: E402
from pri2m.xai import activation_modes as AM  # noqa: E402
from pri2m.xai import recurrence as R  # noqa: E402
from pri2m.xai.sampling import (  # noqa: E402
    batch_to_model_device,
    build_labeled_batch,
    window_config_from_preprocessing,
)

STRATUM = "encoder.L0.x_delay"
DEFAULT_OUT = pathlib.Path("results/phase3/xai/recurrence")
_FOLD_RE = re.compile(r"^f(\d+)")
_EDGE_FIELDS = ("kan", "out", "in", "feature", "importance")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def parse_network(spec: str) -> dict:
    """``NAME=CKPT=MANIFEST`` -> dict. NAME must start with ``f<fold>``.

    The fold comes from the name rather than from guessing at paths, so a pair
    is only ever called "within-fold" because the caller said so.
    """
    parts = spec.split("=")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"--network must be NAME=CKPT=MANIFEST, got {spec!r}")
    name, ckpt, manifest = parts
    m = _FOLD_RE.match(name)
    if m is None:
        raise ValueError(f"network name {name!r} must start with f<fold>, e.g. f0s1")
    return {"name": name, "fold": int(m.group(1)),
            "checkpoint": pathlib.Path(ckpt), "manifest": pathlib.Path(manifest)}


def _union(edge_lists) -> list[dict]:
    """Union of edge dicts by (kan, out, in), first occurrence kept, order stable."""
    seen, out = set(), []
    for edges in edge_lists:
        for e in edges or []:
            k = (str(e["kan"]), int(e["out"]), int(e["in"]))
            if k not in seen:
                seen.add(k)
                out.append({f: e.get(f) for f in _EDGE_FIELDS})
    return out


def necessary_cluster(manifest: dict, exclude=()) -> dict:
    """Union of ``mode_edges`` (and of their ``random_matched`` controls) over the
    claimable x_delay modes of one mode-ablation manifest.

    A claimable mode recorded WITHOUT its membership (manifests predating
    membership logging) cannot be compared; it is reported as
    ``missing_membership`` rather than read as an empty cluster.

    ``exclude`` lists modes dropped by a recorded judgement on top of the flag
    (fold 2 k=3 mode 0 clears the floor by 0.0004, 4% of the floor; PROGRESS
    2026-09-21 excludes it from every claim). They are reported as ``excluded``
    so the judgement is visible in the manifest instead of silently applied.
    """
    claimable, missing, excluded = [], [], []
    exclude = {int(m) for m in exclude}
    cluster, random = [], []
    for md in manifest.get("modes") or []:
        if md.get("stratum", STRATUM) != STRATUM:
            continue
        if (md.get("verdict") or {}).get("claimable") is not True:
            continue
        if int(md.get("mode")) in exclude:
            excluded.append(md.get("mode"))
            continue
        edges = md.get("edges") or {}
        if not edges.get("mode_edges"):
            missing.append(md.get("mode"))
            continue
        claimable.append(md.get("mode"))
        cluster.append(edges.get("mode_edges"))
        random.append(edges.get("random_matched"))
    return {"claimable_modes": claimable, "missing_membership": missing,
            "excluded_modes": excluded,
            "cluster": _union(cluster), "random": _union(random)}


def _load_network(net: dict, args) -> dict:
    ckpt = net["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt} ({net['name']})")
    pre_path = ckpt.parent.parent / "preprocessing.json"
    preprocessing = json.loads(pre_path.read_text()) if pre_path.exists() else None
    load_args = argparse.Namespace(input_size=None, hidden_size=None,
                                   grid_size=5, spline_order=3)
    model = xai_phase3._load_model(str(ckpt), load_args, preprocessing)
    model.eval()
    summary_path = ckpt.parent.parent / "summary.json"
    train_seed = None
    if summary_path.exists():
        try:
            train_seed = (json.loads(summary_path.read_text()).get("config") or {}).get("seed")
        except (OSError, ValueError):
            train_seed = None
    manifest = json.loads(net["manifest"].read_text())
    return {**net, "model": model, "preprocessing": preprocessing,
            "train_seed": train_seed, "ablation": manifest,
            "ablation_checkpoint": (manifest.get("provenance") or {}).get("checkpoint")}


# --------------------------------------------------------------------------- #
# Model reads
# --------------------------------------------------------------------------- #

def layer0_hidden_traces(model, batch: dict) -> np.ndarray:
    """``[N*T, H]`` layer-0 hidden states over ``batch``, via the real forward.

    A forward hook on ``encoder.layers[0]`` (the TKAN cell, called once per step)
    collects every ``h_t``, so the pooling, tau rescaling and delayed input are
    exactly those of ``TKANModel.forward`` -- nothing re-implemented. Layer 0 is
    the one that matters: the x_delay edges live on ``encoder.layers.0.kan_*``
    and their ``out`` indexes layer-0 hidden units.
    """
    steps: list[torch.Tensor] = []
    handle = model.encoder.layers[0].register_forward_hook(
        lambda _m, _i, out: steps.append(out.detach().cpu()))
    try:
        with torch.no_grad():
            model(batch)
    finally:
        handle.remove()
    if not steps:
        raise RuntimeError("layer-0 hook never fired; the encoder did not run")
    h = torch.stack(steps, dim=1)                   # [B, T, H]
    return h.reshape(-1, h.shape[-1]).numpy().astype(np.float64)


def _rows(bundle: dict, lut: dict, edges: list[dict], name: str, what: str) -> np.ndarray:
    try:
        idx = [lut[(str(e["kan"]), int(e["out"]), int(e["in"]))] for e in edges]
    except KeyError as exc:
        raise KeyError(f"{name}: {what} edge {exc} is not in this network's "
                       f"{STRATUM} stratum; manifest and checkpoint disagree") from exc
    return np.asarray(bundle["curves"], dtype=np.float64)[idx] if idx else \
        np.zeros((0, bundle["curves"].shape[1]))


def _batch_key(pre: dict | None) -> str:
    pre = pre or {}
    return json.dumps({k: pre.get(k) for k in ("norm", "window", "channels")},
                      sort_keys=True, default=str)


def _same_windows(b1: dict, b2: dict) -> bool:
    """Same raw windows? (Same files/labels/tau; z-scores may differ by fold.)"""
    for k in ("label", "tau", "omega"):
        if k in b1 and k in b2:
            if not torch.equal(torch.as_tensor(b1[k]).cpu(), torch.as_tensor(b2[k]).cpu()):
                return False
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _direction(src: dict, dst: dict, args) -> dict:
    """A->B: A's necessary cluster (and controls) against B."""
    feats = lambda es: [str(e["feature"]) for e in es]  # noqa: E731
    main = R.recurrence_statistic(src["cluster_curves"], feats(src["cluster"]),
                                  dst["cluster_curves"], feats(dst["cluster"]))
    rnd = R.recurrence_statistic(src["random_curves"], feats(src["random"]),
                                 dst["cluster_curves"], feats(dst["cluster"]))
    null = [R.recurrence_statistic(src["cluster_curves"], feats(src["cluster"]),
                                   u, feats(dst["cluster"]))["S"]
            for u in dst["untrained_cluster_curves"]]
    verdict = R.recurrence_verdict(main["S"], rnd["S"], null, min_null=int(args.min_null))
    pool = R.recurrence_statistic(src["cluster_curves"], feats(src["cluster"]),
                                  dst["pool_curves"], feats(dst["pool"]))
    pool_null = [R.recurrence_statistic(src["cluster_curves"], feats(src["cluster"]),
                                        u, feats(dst["pool"]))["S"]
                 for u in dst["untrained_pool_curves"]]
    pool_null_f = [v for v in pool_null if v is not None]
    return {
        "S": main["S"], "S_abs": main["S_abs"], "n_used": main["n_used"],
        "n_unmatched": main["n_unmatched"],
        "n_degenerate_a": main["n_degenerate_a"], "n_degenerate_b": main["n_degenerate_b"],
        "per_edge_best_cosine": main["per_edge_best_cosine"],
        "per_edge_best_abs_cosine": main["per_edge_best_abs_cosine"],
        "S_random": rnd["S"], "S_random_n_used": rnd["n_used"],
        "S_untrained": null,
        "verdict": verdict,
        "secondary_gated_pool": {
            "note": ("A's necessary cluster against B's whole gated x_delay pool: "
                     "'the function exists in B but was not found necessary there'. "
                     "Descriptive; not part of the pre-committed rule."),
            "S": pool["S"], "S_abs": pool["S_abs"], "n_used": pool["n_used"],
            "n_pool": len(dst["pool"]),
            "null_p95": float(np.percentile(pool_null_f, 95)) if pool_null_f else None,
            "n_null": len(pool_null_f),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", action="append", required=True,
                    help="NAME=CKPT=ABLATION_MANIFEST; NAME starts with f<fold>. Repeat.")
    ap.add_argument("--table", default="results/phase3/phase3_dataset.csv")
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--n-windows", type=int, default=256)
    ap.add_argument("--curve-points", type=int, default=256,
                    help="Sweep points per edge curve. 256 matches what the mode "
                         "ablation clustered on (it samples n=--n-windows points).")
    ap.add_argument("--gate-quantile", type=float, default=0.9,
                    help="Gate for the secondary whole-pool read (matches the ablation).")
    ap.add_argument("--seed", type=int, default=0, help="Window-sample seed.")
    ap.add_argument("--n-untrained", type=int, default=20)
    ap.add_argument("--min-null", type=int, default=20,
                    help="Fewer finite null values than this -> UNDECIDED.")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--tag", default="recurrence")
    ap.add_argument("--exclude-mode", action="append", default=[],
                    metavar="NAME:MODE",
                    help="Drop a claimable mode from NAME's necessary cluster by "
                         "recorded judgement (repeatable), e.g. f2s0:0 for the "
                         "fold-2 mode that clears the floor by 0.0004.")
    args = ap.parse_args(argv)
    exclusions: dict[str, list[int]] = {}
    for spec in args.exclude_mode:
        nm, _, md = spec.rpartition(":")
        if not nm or not md.lstrip("-").isdigit():
            print(f"bad --exclude-mode {spec!r}; expected NAME:MODE", file=sys.stderr)
            return 2
        exclusions.setdefault(nm, []).append(int(md))

    nets = [parse_network(s) for s in args.network]
    names = [n["name"] for n in nets]
    if len(set(names)) != len(names):
        print(f"duplicate network names: {names}", file=sys.stderr)
        return 2

    # -- a. per network: model, batch, curves, necessary cluster ----------- #
    loaded: dict[str, dict] = {}
    batches: dict[str, dict] = {}
    x_grid = None
    for net in nets:
        n = _load_network(net, args)
        name = n["name"]
        # Networks of one fold share the split and so the z-score: same batch.
        key = _batch_key(n["preprocessing"])
        if key not in batches:
            cfg = window_config_from_preprocessing(n["preprocessing"])
            batches[key], _ = build_labeled_batch(
                args.table, args.npz_root, n["preprocessing"], args.n_windows,
                cfg, seed=args.seed)
        batch = batch_to_model_device(batches[key], n["model"])
        bundles = AM.fold_bundles(n["model"], batch, n=int(args.curve_points),
                                  strata=[STRATUM])
        if STRATUM not in bundles:
            print(f"{name}: no {STRATUM} stratum in this model", file=sys.stderr)
            return 2
        bundle = bundles[STRATUM]
        if x_grid is None:
            x_grid = np.asarray(bundle["x"])
        elif not np.allclose(x_grid, bundle["x"]):
            # Different sweep grids put curves in different function spaces.
            raise ValueError(f"{name}: curve grid differs from the first network's")
        lut = R.curve_lookup(bundle)
        nc = necessary_cluster(n["ablation"], exclusions.get(name, ()))
        cluster, random = nc["cluster"], nc["random"]
        gated = AM.gate_by_importance(bundle, quantile=float(args.gate_quantile))
        pool = [{f: r[f] for f in _EDGE_FIELDS} for r in gated["records"]]

        cluster_curves = _rows(bundle, lut, cluster, name, "cluster")
        # Self-check: the direct weight read-out the untrained null relies on must
        # reproduce the curves the bundle stored, or the null is a different object.
        direct = R.edge_curves_from_model(n["model"], cluster, n=int(args.curve_points))
        if cluster and not np.allclose(direct, cluster_curves, rtol=1e-4, atol=1e-6):
            raise RuntimeError(f"{name}: direct edge read-out disagrees with the bundle")

        untrained_c, untrained_p, untrained_diag = [], [], []
        for s in range(int(args.n_untrained)):
            fresh, n_reset = AM.reinitialised_copy(n["model"], seed=s)
            diff = AM.assert_reinitialised(n["model"], fresh)
            untrained_diag.append({"seed": s, "n_reset": n_reset, "mean_weight_diff": diff})
            untrained_c.append(R.edge_curves_from_model(fresh, cluster, n=int(args.curve_points)))
            untrained_p.append(R.edge_curves_from_model(fresh, pool, n=int(args.curve_points)))
            del fresh

        traces = layer0_hidden_traces(n["model"], batch)
        loaded[name] = {
            **n, "batch_key": key, "batch": batches[key], "bundle_lut": lut,
            "cluster": cluster, "random": random, "pool": pool,
            "claimable_modes": nc["claimable_modes"],
            "missing_membership": nc["missing_membership"],
            "excluded_modes": nc["excluded_modes"],
            "cluster_curves": cluster_curves,
            "random_curves": _rows(bundle, lut, random, name, "random_matched"),
            "pool_curves": np.asarray(gated["curves"], dtype=np.float64),
            "untrained_cluster_curves": untrained_c,
            "untrained_pool_curves": untrained_p,
            "untrained_diag": untrained_diag,
            "traces": traces,
            "n_x_delay_edges": int(bundle["n_edges"]),
        }
        print(f"{name}: fold {n['fold']}  claimable modes {nc['claimable_modes']}  "
              f"cluster {len(cluster)} edges  mix {R.cluster_channel_mix(cluster)}")

    # -- c. pairs ------------------------------------------------------------ #
    pairs = []
    for a_name, b_name in itertools.combinations(names, 2):
        A, B = loaded[a_name], loaded[b_name]
        rec = {"a": a_name, "b": b_name, "fold_a": A["fold"], "fold_b": B["fold"],
               "cross_fold": A["fold"] != B["fold"],
               "channel_mix_a": R.cluster_channel_mix(A["cluster"]),
               "channel_mix_b": R.cluster_channel_mix(B["cluster"]),
               "recurs": None, "undecided_reason": None}
        empty = [x["name"] for x in (A, B) if not x["cluster"]]
        if empty:
            rec["undecided_reason"] = f"no_necessary_cluster:{empty[0]}"
            rec["a_to_b"] = rec["b_to_a"] = rec["alignment"] = None
            pairs.append(rec)
            print(f"{a_name} vs {b_name}: UNDECIDED ({rec['undecided_reason']})")
            continue
        rec["a_to_b"] = _direction(A, B, args)
        rec["b_to_a"] = _direction(B, A, args)
        dirs = [rec["a_to_b"]["verdict"]["recurs"], rec["b_to_a"]["verdict"]["recurs"]]
        if None in dirs:
            rec["undecided_reason"] = "direction_undecided:" + ",".join(
                str(d["verdict"]["undecided_reason"])
                for d in (rec["a_to_b"], rec["b_to_a"]) if d["verdict"]["recurs"] is None)
        else:
            rec["recurs"] = bool(all(dirs))

        same = A["batch_key"] == B["batch_key"] or _same_windows(A["batch"], B["batch"])
        if not same or A["traces"].shape != B["traces"].shape:
            rec["alignment"] = {"aligned_overlap": None,
                                "undecided_reason": "different_windows"}
        else:
            al = R.align_hidden_units(A["traces"], B["traces"])
            obs = R.aligned_id_overlap(A["cluster"], B["cluster"], al["perm"])
            H = int(A["traces"].shape[1])
            rec["alignment"] = {
                "same_z_score": A["batch_key"] == B["batch_key"],
                "n_a": len(A["cluster"]), "n_b": len(B["cluster"]),
                "aligned_overlap": int(obs),
                "raw_id_overlap_INVALID": int(R.aligned_id_overlap(
                    A["cluster"], B["cluster"], np.arange(H))),
                "null": R.overlap_null(A["cluster"], B["cluster"], H,
                                       n_perm=int(args.n_perm), seed=int(args.seed),
                                       observed=obs),
                "perm": al["perm"],
                "matched_abs_corr_median": float(np.median(al["matched_abs_corr"])),
                "matched_abs_corr": al["matched_abs_corr"],
                "constant_a": al["constant_a"], "constant_b": al["constant_b"],
            }
        pairs.append(rec)
        print(f"{a_name} vs {b_name}: S {rec['a_to_b']['S']} / {rec['b_to_a']['S']}  "
              f"recurs={rec['recurs']}  aligned overlap "
              f"{(rec['alignment'] or {}).get('aligned_overlap')}")

    # -- per-fold summary (within-fold pairs only) -------------------------- #
    per_fold = {}
    for fold in sorted({n["fold"] for n in nets}):
        within = [p for p in pairs if not p["cross_fold"] and p["fold_a"] == fold]
        n_rec = sum(p["recurs"] is True for p in within)
        n_no = sum(p["recurs"] is False for p in within)
        n_und = sum(p["recurs"] is None for p in within)
        per_fold[str(fold)] = {
            "networks": [n["name"] for n in nets if n["fold"] == fold],
            "n_pairs": len(within), "n_recur": n_rec, "n_not_recur": n_no,
            "n_undecided": n_und,
            "networks_without_cluster": [n["name"] for n in nets if n["fold"] == fold
                                         and not loaded[n["name"]]["cluster"]],
            "summary": f"{n_rec} of {len(within)} within-fold pairs recur "
                       f"({n_no} do not, {n_und} undecided)",
        }

    # -- d. curves + manifest ---------------------------------------------- #
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"x": x_grid}
    for name, L in loaded.items():
        for part in ("cluster", "random", "pool"):
            edges = L[part]
            arrays[f"{name}__{part}_curves"] = L[f"{part}_curves"].astype(np.float32)
            arrays[f"{name}__{part}_kan"] = np.asarray([e["kan"] for e in edges], dtype="U64")
            arrays[f"{name}__{part}_out"] = np.asarray([e["out"] for e in edges], dtype=np.int64)
            arrays[f"{name}__{part}_in"] = np.asarray([e["in"] for e in edges], dtype=np.int64)
            arrays[f"{name}__{part}_feature"] = np.asarray([e["feature"] for e in edges], dtype="U64")
            arrays[f"{name}__{part}_importance"] = np.asarray(
                [np.nan if e["importance"] is None else e["importance"] for e in edges],
                dtype=np.float64)
        if L["untrained_cluster_curves"]:
            arrays[f"{name}__untrained_cluster_curves"] = np.stack(
                L["untrained_cluster_curves"]).astype(np.float32)
    npz_path = args.output_dir / f"recurrence_{args.tag}_curves.npz"
    np.savez_compressed(npz_path, **arrays)

    networks_block = {
        name: {
            "fold": L["fold"], "checkpoint": str(L["checkpoint"]),
            "ablation_manifest": str(L["manifest"]),
            "ablation_checkpoint": L["ablation_checkpoint"],
            "train_seed": L["train_seed"],
            "claimable_modes": L["claimable_modes"],
            "missing_membership": L["missing_membership"],
            "excluded_modes": L["excluded_modes"],
            "n_cluster_edges": len(L["cluster"]),
            "channel_mix": R.cluster_channel_mix(L["cluster"]),
            "random_channel_mix": R.cluster_channel_mix(L["random"]),
            "n_gated_pool": len(L["pool"]), "n_x_delay_edges": L["n_x_delay_edges"],
            "cluster": L["cluster"], "random_matched": L["random"],
            "untrained": L["untrained_diag"],
        }
        for name, L in loaded.items()
    }
    manifest = {
        "analysis": "Exp 2: function-level recurrence of the necessary x_delay cluster",
        "provenance": {
            "networks": {n["name"]: {"checkpoint": str(n["checkpoint"]),
                                     "ablation_manifest": str(n["manifest"])} for n in nets},
            "stratum": STRATUM, "seed": int(args.seed), "n_windows": int(args.n_windows),
            "curve_points": int(args.curve_points), "gate_quantile": float(args.gate_quantile),
            "n_untrained": int(args.n_untrained), "untrained_seeds": list(range(int(args.n_untrained))),
            "min_null": int(args.min_null), "n_perm": int(args.n_perm),
            "exclude_mode": exclusions,
            "table": str(args.table), "npz_root": str(args.npz_root),
            "curves_npz": str(npz_path),
        },
        "rule": ("Direction A->B recurs iff S > p95(untrained null) AND S - S_random > 0; "
                 "a pair recurs iff both directions do; None = UNDECIDED, never False."),
        "networks": networks_block,
        "pairs": pairs,
        "per_fold": per_fold,
        "complete": True,
    }
    dest = args.output_dir / f"recurrence_{args.tag}.json"
    dest.write_text(json.dumps(AM.jsonable(manifest), indent=2))
    for fold, s in per_fold.items():
        print(f"fold {fold}: {s['summary']}")
    print(f"\nWrote {dest}\nWrote {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
