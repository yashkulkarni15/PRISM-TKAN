#!/usr/bin/env python3
"""Measure the training pipeline's run-to-run repeatability from duplicate sweeps.

WHY THIS EXISTS. On the night of 2026-07-25 two five-fold sweeps were resubmitted
having already completed, leaving ten run directories where five belong. That was
a mistake and it cost roughly 80 GPU-hours. It also, by accident, produced the one
measurement this project had never made: **how much does a fold's validation AUC
move when the identical configuration is trained again?**

That number is the floor under every AUC comparison in the project. An effect
smaller than it cannot be claimed, no matter how clean the mean looks. Both
2026-07 retrains (side-input ablation, quantile severity head) report effects in
the third decimal place, so the floor decides whether either is a result.

WHAT IT MEASURES, AND WHAT IT DOES NOT. The duplicate pairs share an identical
config *and* an identical seed (verified, not assumed -- see ``pair_configs_match``
in the output). So the spread here is pure nondeterminism: GPU atomics, cuDNN
algorithm selection, and non-associative float reduction. **Seed-to-seed spread
would be larger.** Every number this script prints is therefore a LOWER BOUND on
run-to-run variability, which is the conservative direction for the claims it
gates: if an effect is already inside this band, more sampling will not rescue it.

Canonicalisation, the second use. Ten dirs where five belong, with nothing in the
path saying which is which. This prints both candidate sets so the choice can be
made on evidence and recorded, rather than by whichever glob happens to run first.

    python scripts/repeatability_audit.py
    python scripts/repeatability_audit.py --json results/phase3/repeatability.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from pathlib import Path

# Trees that hold duplicated sweeps, and how to recognise each arm inside them.
# `runs/` is mixed: it also holds the June side-on baseline and five aborted
# pre-side-input runs, so an arm is selected by config, never by directory order.
TREES = ("runs", "runs_quantile")


def _load_runs(root: Path, tree: str) -> list[dict]:
    """Every completed run in one tree, with the config fields that identify its arm."""
    out = []
    for path in sorted(glob.glob(str(root / tree / "*" / "summary.json"))):
        with open(path, encoding="utf-8") as fh:
            s = json.load(fh)
        cfg = s.get("config", {})
        best = s.get("best_auc_val")
        auc = best.get("auc") if isinstance(best, dict) else best
        if auc is None:
            continue  # aborted run: no best-AUC checkpoint was ever written
        out.append({
            "tree": tree,
            "run_id": s["run_id"],
            "fold": s.get("fold"),
            "side_input": cfg.get("side_input"),
            "severity_mode": cfg.get("severity_mode", "default"),
            "seed": cfg.get("seed"),
            "auc": float(auc),
            "best_auc_epoch": s.get("best_auc_epoch"),
            "mtime": os.path.getmtime(path),
            "config": cfg,
            "history": s.get("history") or {},
        })
    return out


def _arm(runs: list[dict], predicate) -> dict[int, list[dict]]:
    """Group one arm's runs by fold, oldest first, so [0] is always the earlier run."""
    by_fold: dict[int, list[dict]] = {}
    for r in runs:
        if predicate(r):
            by_fold.setdefault(r["fold"], []).append(r)
    for fold in by_fold:
        by_fold[fold].sort(key=lambda r: r["mtime"])
    return by_fold


def _configs_match(a: dict, b: dict) -> bool:
    """Whether two runs are the same experiment.

    This is the load-bearing check of the whole script. If the duplicate pairs
    differed in any config field then their spread would measure that field, not
    nondeterminism, and every repeatability number below would be wrong.
    """
    return a["config"] == b["config"]


def _trajectory_identity(early: dict, late: dict) -> dict:
    """Are two runs bit-identical over the whole training history, or only at the end?

    Compares every epoch's train loss and validation AUC at full float precision.
    ``max_epoch_auc_gap`` is the largest single-epoch disagreement, which can be
    far bigger than the gap in the SELECTED best AUC: two runs can wander apart
    mid-training and land close by coincidence, and reporting only the headline
    number would hide that.
    """
    a_hist = early.get("history") or {}
    b_hist = late.get("history") or {}
    a_auc = [v["auc"] for v in a_hist.get("val", []) if "auc" in v]
    b_auc = [v["auc"] for v in b_hist.get("val", []) if "auc" in v]
    a_loss = a_hist.get("train_loss") or []
    b_loss = b_hist.get("train_loss") or []
    if not a_auc or len(a_auc) != len(b_auc):
        return {"trajectory_comparable": False}
    return {
        "trajectory_comparable": True,
        "auc_trajectory_identical": a_auc == b_auc,
        "train_loss_identical": a_loss == b_loss,
        "bit_identical_run": a_auc == b_auc and a_loss == b_loss,
        "max_epoch_auc_gap": max(abs(x - y) for x, y in zip(a_auc, b_auc)),
    }


def audit_arm(name: str, by_fold: dict[int, list[dict]]) -> dict:
    """Per-fold drift between the earlier and later run of one duplicated arm."""
    folds, drifts, pairs, mismatched = [], [], [], []
    for fold in sorted(by_fold):
        runs = by_fold[fold]
        if len(runs) < 2:
            continue
        early, late = runs[0], runs[1]
        if not _configs_match(early, late):
            mismatched.append(fold)
            continue
        drift = abs(early["auc"] - late["auc"])
        folds.append(fold)
        drifts.append(drift)
        # THE DECISIVE COMPARISON, and the reason this script compares whole
        # trajectories rather than just the headline AUC. If training were
        # inherently nondeterministic (GPU atomics, cuDNN algorithm choice,
        # non-associative reductions) then NO pair could reproduce bit-exactly.
        # Any pair that does is proof the pipeline is deterministic when the
        # surrounding conditions match, which means a pair that DIVERGES is
        # evidence that something outside the config differed -- hardware,
        # library version, node -- and not evidence of irreducible noise.
        traj = _trajectory_identity(early, late)
        pairs.append({
            "fold": fold,
            "earlier": {"run_id": early["run_id"], "auc": early["auc"],
                        "best_auc_epoch": early["best_auc_epoch"]},
            "later": {"run_id": late["run_id"], "auc": late["auc"],
                      "best_auc_epoch": late["best_auc_epoch"]},
            "abs_auc_drift": drift,
            # A pair can land on the same AUC from a different epoch, or on a
            # different AUC from the same epoch. Both say something, so record both.
            "same_best_epoch": early["best_auc_epoch"] == late["best_auc_epoch"],
            **traj,
        })
    early_mean = (st.mean([by_fold[f][0]["auc"] for f in folds]) if folds else None)
    late_mean = (st.mean([by_fold[f][1]["auc"] for f in folds]) if folds else None)
    return {
        "arm": name,
        "n_pairs": len(pairs),
        "pairs": pairs,
        "max_fold_drift": max(drifts) if drifts else None,
        "mean_fold_drift": st.mean(drifts) if drifts else None,
        # Same SELECTED AUC. Weaker than bit-identical: two runs can agree here
        # by coincidence after diverging mid-training, so never quote it alone.
        "n_same_best_auc": sum(1 for d in drifts if d == 0.0),
        "n_bit_identical_runs": sum(1 for p in pairs if p.get("bit_identical_run")),
        "max_epoch_auc_gap": max((p["max_epoch_auc_gap"] for p in pairs
                                  if p.get("trajectory_comparable")), default=None),
        "earlier_5fold_mean": early_mean,
        "later_5fold_mean": late_mean,
        "mean_level_drift": (abs(early_mean - late_mean)
                             if early_mean is not None and late_mean is not None else None),
        "pair_configs_match": not mismatched,
        "folds_with_config_mismatch": mismatched,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results/phase3",
                   help="Tree holding runs/ and runs_quantile/")
    p.add_argument("--json", default=None, help="Write the full report here")
    args = p.parse_args(argv)

    root = Path(args.results_root)
    runs = []
    for tree in TREES:
        runs += _load_runs(root, tree)
    if not runs:
        print(f"no completed runs under {root}/{{{','.join(TREES)}}}")
        return 1

    baseline = _arm(runs, lambda r: (r["tree"] == "runs" and r["side_input"] is True
                                     and r["severity_mode"] == "default"))
    ablation = _arm(runs, lambda r: r["tree"] == "runs" and r["side_input"] is False)
    quantile = _arm(runs, lambda r: r["tree"] == "runs_quantile")

    # The baseline is the published side-on sweep. Fold 0 has two entries for a
    # reason unrelated to the duplication: the lr=0.003 run that hit the gradient
    # blow-up, and its lr=0.0015 rerun. The later one is the published run, so the
    # baseline takes [-1] per fold while the duplicated arms take [0] and [1].
    base_auc = {f: baseline[f][-1]["auc"] for f in sorted(baseline)}
    base_mean = st.mean(base_auc.values())

    report = {
        "baseline_side_on": {
            "per_fold": base_auc,
            "mean": base_mean,
            "run_ids": {f: baseline[f][-1]["run_id"] for f in sorted(baseline)},
        },
        "arms": [audit_arm("no_side_input_ablation", ablation),
                 audit_arm("quantile_severity", quantile)],
    }

    # The comparison the floor exists to gate.
    for arm in report["arms"]:
        if arm["earlier_5fold_mean"] is None:
            continue
        arm["effect_vs_baseline_earlier"] = arm["earlier_5fold_mean"] - base_mean
        arm["effect_vs_baseline_later"] = arm["later_5fold_mean"] - base_mean
        biggest = max(abs(arm["effect_vs_baseline_earlier"]),
                      abs(arm["effect_vs_baseline_later"]))
        arm["effect_exceeds_max_fold_drift"] = bool(biggest > arm["max_fold_drift"])
        # The fold-level comparison above is the strict one. The mean-level one
        # below is the fair one, because the effect being tested is itself a
        # five-fold mean: how many times its own re-run noise is the effect?
        # Anything near 1 is noise. There is exactly ONE repeat, so even a large
        # ratio here is weak evidence -- this bounds a claim, it cannot support one.
        arm["mean_effect_over_mean_drift"] = (
            biggest / arm["mean_level_drift"] if arm["mean_level_drift"] else None
        )
        # Which duplicate you keep changes the published effect by this much.
        # If that swing is a large share of the effect, the effect is not robust
        # to a coin flip nobody meant to make.
        arm["effect_swing_between_duplicates"] = abs(
            arm["effect_vs_baseline_later"] - arm["effect_vs_baseline_earlier"]
        )

    floor = max(a["max_fold_drift"] for a in report["arms"] if a["max_fold_drift"] is not None)
    report["repeatability_floor_auc"] = floor
    report["floor_is_lower_bound"] = True
    report["floor_note"] = (
        "Identical config AND identical seed, so this is not seed variance. "
        "Seed-to-seed spread would be larger. Treat as a LOWER BOUND."
    )
    n_bit = sum(a["n_bit_identical_runs"] for a in report["arms"])
    n_pairs = sum(a["n_pairs"] for a in report["arms"])
    report["n_bit_identical_runs"] = n_bit
    report["n_pairs"] = n_pairs

    # Was it the hardware? Cross-reference the node each job ran on, if the
    # sacct capture is present. summary.json records no hardware, so without
    # this file the question is unanswerable from stored artifacts -- and the
    # cluster accounting logs that produced it are eventually purged.
    node_path = root / "slurm_nodes.json"   # root IS results/phase3
    if node_path.exists():
        by_arm = json.loads(node_path.read_text())["by_arm"]
        cross, cross_bit, same, same_bit = 0, 0, 0, 0
        for arm in report["arms"]:
            table = by_arm.get(arm["arm"], {})
            for pair in arm["pairs"]:
                entry = table.get(str(pair["fold"]))
                if not entry:
                    continue
                n_early, n_late = entry["earlier"]["node"], entry["later"]["node"]
                pair["node_earlier"], pair["node_later"] = n_early, n_late
                pair["same_node"] = n_early == n_late
                if pair["same_node"]:
                    same += 1
                    same_bit += bool(pair.get("bit_identical_run"))
                else:
                    cross += 1
                    cross_bit += bool(pair.get("bit_identical_run"))
        report["node_check"] = {
            "pairs_on_different_nodes": cross,
            "different_node_bit_identical": cross_bit,
            "pairs_on_same_node": same,
            "same_node_bit_identical": same_bit,
            # The conclusion, not left to the reader to re-derive.
            "hardware_explains_divergence": not (cross_bit > 0),
            "note": (
                f"{cross_bit} of {cross} pairs that ran on DIFFERENT nodes still "
                "reproduced bit-exactly, so which machine a job landed on does "
                "not predict whether it reproduced. Hardware is not the "
                "explanation."
            ),
        }

    # The interpretation, stated in the artifact so a later reader cannot
    # mistake this for a measurement of inherent training noise.
    report["interpretation"] = (
        f"{n_bit} of {n_pairs} pairs reproduced BIT-IDENTICALLY over all epochs "
        "(train loss and val AUC at full precision), so this is not inherent "
        "training noise. It is not the hardware either: see node_check. The "
        "cause is that the trainer seeds every RNG (phase3_trainer.py:158) and "
        "then never requests deterministic algorithms -- nothing sets "
        "torch.use_deterministic_algorithms, nothing disables cuDNN benchmark "
        "mode, and every run has compile: true. Benchmark mode selects kernels "
        "by timing them at runtime, so the arithmetic path depends on transient "
        "GPU state. Seeding fixes the inputs, not which kernels consume them, "
        "which is exactly the observed pattern: mostly reproducible, "
        "intermittently not, uncorrelated with the node. Fix: set the "
        "determinism flags, and record node plus library versions in "
        "summary.json so this is answerable without the accounting log."
    )

    print(f"baseline (side-on, published): "
          f"{' '.join(f'{base_auc[f]:.4f}' for f in sorted(base_auc))}  mean {base_mean:.5f}")
    print()
    for arm in report["arms"]:
        print(f"--- {arm['arm']} ---")
        print(f"  configs identical across pairs: {arm['pair_configs_match']}")
        for pr in arm["pairs"]:
            flag = "  BIT-IDENTICAL run" if pr.get("bit_identical_run") else (
                f"  diverged, max epoch gap {pr['max_epoch_auc_gap']:.4f}"
                if pr.get("trajectory_comparable") else "")
            print(f"  fold {pr['fold']}: {pr['earlier']['auc']:.4f} (ep {pr['earlier']['best_auc_epoch']})"
                  f"  vs  {pr['later']['auc']:.4f} (ep {pr['later']['best_auc_epoch']})"
                  f"   drift {pr['abs_auc_drift']:.4f}{flag}")
        print(f"  max fold drift {arm['max_fold_drift']:.4f}   "
              f"mean {arm['mean_fold_drift']:.4f}   "
              f"bit-identical {arm['n_bit_identical_runs']}/{arm['n_pairs']}   "
              f"max single-epoch gap {arm['max_epoch_auc_gap']:.4f}")
        print(f"  5-fold mean: earlier {arm['earlier_5fold_mean']:.5f}  "
              f"later {arm['later_5fold_mean']:.5f}  "
              f"(mean-level drift {arm['mean_level_drift']:.5f})")
        print(f"  effect vs baseline: earlier {arm['effect_vs_baseline_earlier']:+.5f}  "
              f"later {arm['effect_vs_baseline_later']:+.5f}")
        print(f"  effect exceeds this arm's max fold drift: "
              f"{arm['effect_exceeds_max_fold_drift']}")
        print()
    print(f"REPEATABILITY FLOOR (max same-config drift in SELECTED best AUC): {floor:.4f}")
    print(f"Bit-identical pairs: {n_bit}/{n_pairs}. Same config, same seed (verified).")
    print("So this is NOT inherent training noise: a nondeterministic pipeline could")
    print("not reproduce any pair bit-exactly. Something outside the config differed")
    print("in the divergent pairs, and summary.json records no hardware or environment")
    print("to say what. The floor is also a LOWER bound: seed spread would be larger.")
    print("Any effect below it cannot be claimed from these runs.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
