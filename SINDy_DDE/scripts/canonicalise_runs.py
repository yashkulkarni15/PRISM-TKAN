#!/usr/bin/env python3
"""Say which run directory is canonical for each fold of each published sweep.

U12 section 7: both five-fold arrays were submitted twice, so the results tree
holds two complete copies of the side-input ablation and two of the quantile
severity head. Nothing in the path says which one the published numbers came
from. Anything that globs the tree picks arbitrarily.

This resolves it by *matching the published number*, not by taking the newest
directory. A timestamp rule would be a guess; the published AUC is a fact. If
no run reproduces the published figure the script refuses rather than picking.

Four of the ten duplicated folds reproduced *bit-identically* (see
``results/phase3/repeatability.json``), so both copies match the published
number and the number cannot choose between them. Those are genuinely
interchangeable. The tie-break is the ``later`` run recorded in that same
repeatability audit, which took submission order from ``sacct`` End time rather
than from a local file timestamp, and it is recorded as a tie-break in the
marker rather than dressed up as a measurement.

The tree also holds runs that belong to no published sweep. Six predate the
``side_input`` config key entirely (June 2026, val AUC 0.85-0.88, the
pre-expansion generation) and the side-on baseline has a superseded fold-0 at
0.9388. Those are marked too, so "unmarked" never means "unclassified".

Writes ``results/phase3/RUNS_CANONICAL.json`` (the registry consumers should
read) and a marker file in each run directory (so a glob sees it without
loading the registry). Idempotent: re-running overwrites with the same content.

Usage:
    python3 scripts/canonicalise_runs.py              # write
    python3 scripts/canonicalise_runs.py --check      # verify, write nothing
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Published per-fold AUC, from Research Update 12 section 6. The ablation table
# gives both arms; the quantile arm is the 0.9616 mean the same section quotes,
# broken out per fold. These are the figures already sent to Prof. Abell, so
# they define which run is canonical rather than the other way round.
PUBLISHED = {
    "side_on": {
        "source": "Update 12 section 6, 'side inputs on' column",
        "tree": "runs",
        "auc": {0: 0.9597, 1: 0.9122, 2: 0.9752, 3: 0.9605, 4: 0.9695},
    },
    "side_off": {
        "source": "Update 12 section 6, 'side inputs off' column",
        "tree": "runs",
        "auc": {0: 0.9626, 1: 0.9080, 2: 0.9747, 3: 0.9546, 4: 0.9632},
    },
    "quantile": {
        "source": "Update 12 section 6, quantile severity head (mean 0.9616)",
        "tree": "runs_quantile",
        "auc": {0: 0.9629, 1: 0.9226, 2: 0.9796, 3: 0.9631, 4: 0.9799},
    },
}

TOLERANCE = 5e-5  # the published tables carry four decimals


def classify(summary: dict) -> str:
    """Which sweep does this run belong to?

    ``side_input`` absent means the run predates the key, which is the June
    generation. Present-and-False is the ablation arm. Present-and-True is the
    baseline. The quantile arm lives in its own tree.
    """
    config = summary.get("config", {}) or {}
    if "side_input" not in config:
        return "legacy"
    return "side_on" if config["side_input"] else "side_off"


def read_runs(root: pathlib.Path) -> list[dict]:
    runs = []
    for tree in ("runs", "runs_quantile"):
        tree_dir = root / "results" / "phase3" / tree
        if not tree_dir.is_dir():
            continue
        for run_dir in sorted(tree_dir.iterdir()):
            summary_path = run_dir / "summary.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text())
            arm = "quantile" if tree == "runs_quantile" else classify(summary)
            runs.append({
                "run_id": run_dir.name,
                "tree": tree,
                "path": str(run_dir.relative_to(root)),
                "arm": arm,
                "fold": summary.get("fold"),
                "auc": (summary.get("best_auc_val") or {}).get("auc"),
            })
    return runs


def later_run_ids(root: pathlib.Path) -> dict[tuple[str, int], str]:
    """(arm, fold) -> the later of the two submissions, from the audit.

    Submission order there came from ``sacct`` End time, so it survives an
    rsync that rewrites local mtimes.
    """
    audit_path = root / "results" / "phase3" / "repeatability.json"
    if not audit_path.is_file():
        return {}
    audit = json.loads(audit_path.read_text())
    arm_names = {"no_side_input_ablation": "side_off", "quantile_severity": "quantile"}
    out = {}
    for block in audit.get("arms", []):
        arm = arm_names.get(block.get("arm"))
        if arm is None:
            continue
        for pair in block.get("pairs", []):
            out[(arm, pair["fold"])] = pair["later"]["run_id"]
    return out


def build_registry(runs: list[dict], later: dict | None = None) -> dict:
    """Select the canonical run per (arm, fold). Refuse to guess."""
    registry: dict = {"arms": {}, "unpublished": []}
    problems: list[str] = []
    later = later or {}

    for arm, spec in PUBLISHED.items():
        entries: dict[str, dict] = {}
        for fold, target in spec["auc"].items():
            candidates = [r for r in runs if r["arm"] == arm and r["fold"] == fold]
            matches = [r for r in candidates
                       if r["auc"] is not None and abs(r["auc"] - target) <= TOLERANCE]
            tie_break = None
            if len(matches) > 1:
                # Several copies reproduce the published figure, so the figure
                # cannot choose. Fall back to the recorded submission order.
                preferred = later.get((arm, fold))
                chosen = [r for r in matches if r["run_id"] == preferred]
                if len(chosen) != 1:
                    problems.append(
                        f"{arm} fold {fold}: {len(matches)} runs reproduce the "
                        f"published {target:.4f} and the repeatability audit "
                        f"does not name one of them as the later submission"
                    )
                    continue
                matches = chosen
                tie_break = ("duplicates agree on the published AUC; kept the "
                             "later submission per repeatability.json")
            if len(matches) != 1:
                problems.append(
                    f"{arm} fold {fold}: {len(matches)} runs reproduce the "
                    f"published {target:.4f} (expected exactly 1) among "
                    f"{len(candidates)} candidates"
                )
                continue
            canonical = matches[0]
            entries[str(fold)] = {
                "canonical": canonical["run_id"],
                "path": canonical["path"],
                "auc": canonical["auc"],
                "matched_published_auc": target,
                "tie_break": tie_break,
                "superseded": sorted(r["run_id"] for r in candidates
                                     if r["run_id"] != canonical["run_id"]),
            }
        registry["arms"][arm] = {
            "source": spec["source"], "tree": spec["tree"], "folds": entries,
        }

    for run in runs:
        if run["arm"] == "legacy":
            registry["unpublished"].append({
                "run_id": run["run_id"], "path": run["path"], "fold": run["fold"],
                "reason": "predates the side_input config key (June 2026 generation)",
            })

    registry["problems"] = problems
    return registry


def marker_for(run_id: str, registry: dict) -> dict | None:
    for arm, block in registry["arms"].items():
        for fold, entry in block["folds"].items():
            if entry["canonical"] == run_id:
                return {"status": "CANONICAL", "arm": arm, "fold": int(fold),
                        "published_as": entry["matched_published_auc"],
                        "source": block["source"],
                        "tie_break": entry["tie_break"]}
            if run_id in entry["superseded"]:
                return {"status": "SUPERSEDED", "arm": arm, "fold": int(fold),
                        "superseded_by": entry["canonical"],
                        "reason": "duplicate submission; this copy is not the "
                                  "one the published number came from"}
    for entry in registry["unpublished"]:
        if entry["run_id"] == run_id:
            return {"status": "UNPUBLISHED", "fold": entry["fold"],
                    "reason": entry["reason"]}
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--check", action="store_true",
                        help="verify selection is unambiguous; write nothing")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    runs = read_runs(root)
    registry = build_registry(runs, later_run_ids(root))

    if registry["problems"]:
        for problem in registry["problems"]:
            print(f"REFUSING: {problem}", file=sys.stderr)
        return 1

    n_canonical = sum(len(b["folds"]) for b in registry["arms"].values())
    n_superseded = sum(len(e["superseded"])
                       for b in registry["arms"].values() for e in b["folds"].values())
    print(f"{len(runs)} run directories: {n_canonical} canonical, "
          f"{n_superseded} superseded duplicates, "
          f"{len(registry['unpublished'])} unpublished")

    if args.check:
        print("check only, nothing written")
        return 0

    registry_path = root / "results" / "phase3" / "RUNS_CANONICAL.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")

    for run in runs:
        marker = marker_for(run["run_id"], registry)
        if marker is None:
            continue
        (root / run["path"] / "CANONICAL.json").write_text(
            json.dumps(marker, indent=2) + "\n")

    print(f"registry -> {registry_path.relative_to(root)}")
    print(f"markers  -> {len(runs)} x CANONICAL.json in each run directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
