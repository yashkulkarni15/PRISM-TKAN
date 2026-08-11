#!/usr/bin/env python3
"""Bring a stored counterfactual manifest up to the corrected comparison.

Update 13's generator comparison was reworked after its first result was
mislabelled. The old metric compared the *direction* each generator moved depth
and reported 12/38 "agreement" -- but ``physics_constrained_counterfactual``
searches ``b`` in ``(b_floor, b_base]``, so it can only hold or lower depth
while DiCE varies both axes freely. The metric was scoring "chose the other
knob" as "disagrees". It was replaced by an axis breakdown plus outcome
agreement, and ``n_agrees_with_physics_cf`` was deleted with a regression test
asserting it stays gone.

The manifest on disk predates that fix. The published Update 13 figures are
already correct -- they were recomputed from the stored per-case coordinates --
but the artifact still carries the superseded field, so anything reading the
manifest instead of the update gets the retracted number.

**No network and no boundary evaluation are needed.** Both generators' per-case
stability was computed against the real ``blim`` at run time and written down,
and both sets of coordinates are stored. Agreement is a comparison of two
recorded facts, so the repair is exact rather than approximate.

The comparison itself is imported from ``counterfactuals_phase3`` rather than
reimplemented: a second copy would be free to drift from the one that produced
the published numbers, and then "what a re-run would have produced" would stop
being true.

Usage:
    python3 scripts/repair_counterfactual_manifest.py            # repair in place
    python3 scripts/repair_counterfactual_manifest.py --check    # report, write nothing
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from counterfactuals_phase3 import compare_generator_outcomes, pool_dice  # noqa: E402

DEFAULT_MANIFEST = pathlib.Path(
    "results/phase3/xai/counterfactuals/counterfactuals_manifest.json"
)
# The metric the rework deleted. Present at two levels plus once per case.
SUPERSEDED_AGGREGATE = "n_agrees_with_physics_cf"
SUPERSEDED_PER_CASE = "agrees_with_physics_cf"


def _pairs_from_records(cases: list[dict], dice_cases: list[dict]) -> list[dict]:
    """Match the two generators' records by ``file``.

    By position would usually work and would fail silently when it did not, so
    the file name -- the only stable identity either record carries -- is the
    key. A DiCE record with no physics counterpart still contributes its axis
    tally but is not comparable, exactly as in the live runner.
    """
    by_file = {c.get("file"): c for c in cases}
    pairs = []
    for rec in dice_cases:
        if not rec.get("found"):
            continue
        pair = {
            "omega_base": rec["omega_base"], "b_base": rec["b_base"],
            "dice_omega_cf": rec["omega_cf"], "dice_b_cf": rec["b_cf"],
            "dice_stable": rec["physically_stable"],
        }
        physics = by_file.get(rec.get("file"))
        if physics and physics.get("omega_cf") is not None and physics.get("b_cf") is not None:
            pair.update({
                "physics_omega_cf": physics["omega_cf"],
                "physics_b_cf": physics["b_cf"],
                "physics_stable": physics["physics_stable_cf"],
            })
        pairs.append(pair)
    return pairs


def _verify_stored_aggregates(dataset: str, dice: dict, dice_cases: list[dict]) -> None:
    """Recompute what the file already claims, and refuse on a contradiction.

    The coordinates are the evidence; the aggregates are derived. If a derived
    value disagrees with the records it came from, the file is not what it says
    it is, and overwriting it would destroy the only sign of that.
    """
    found = [r for r in dice_cases if r.get("found")]
    recomputed = sum(1 for r in found if r.get("physically_stable"))
    stored = dice.get("n_physically_stable")
    if stored is not None and int(stored) != recomputed:
        raise ValueError(
            f"dataset {dataset}: stored n_physically_stable={stored} but the "
            f"per-case records give {recomputed}. Refusing to repair a manifest "
            f"whose aggregates contradict its own records."
        )
    stored_found = dice.get("n_found")
    if stored_found is not None and int(stored_found) != len(found):
        raise ValueError(
            f"dataset {dataset}: stored n_found={stored_found} but {len(found)} "
            f"per-case records are marked found."
        )


def _repair_dice_block(dataset: str, cases: list[dict], dice: dict) -> dict:
    """Return ``dice`` with the corrected comparison in place of the old one."""
    if dice.get("error"):
        return dice
    dice_cases = dice.get("per_case", [])
    _verify_stored_aggregates(dataset, dice, dice_cases)

    pairs = _pairs_from_records(cases, dice_cases)
    comparison = compare_generator_outcomes(pairs)
    annotations = comparison.pop("per_case")

    out = {k: v for k, v in dice.items() if k != SUPERSEDED_AGGREGATE}
    out.update(comparison)

    repaired_cases, i = [], 0
    for rec in dice_cases:
        rec = {k: v for k, v in rec.items() if k != SUPERSEDED_PER_CASE}
        if rec.get("found"):
            rec.update(annotations[i])
            i += 1
        repaired_cases.append(rec)
    out["per_case"] = repaired_cases
    return out


def _repaired(manifest: dict) -> dict:
    """The manifest as the corrected code would have written it."""
    out = copy.deepcopy(manifest)
    blocks = []
    for dataset, block in out.get("per_dataset", {}).items():
        if "dice" not in block:
            continue
        block["dice"] = _repair_dice_block(dataset, block.get("cases", []), block["dice"])
        blocks.append(block["dice"])

    if blocks and "pooled" in out and "dice" in out["pooled"]:
        # Re-pool from the repaired per-dataset blocks, again through the
        # runner's own function, so the pooled row cannot disagree with the
        # rows it summarises.
        out["pooled"]["dice"] = pool_dice(blocks)
    return out


def repair_manifest(path, *, check_only: bool = False) -> bool:
    """Repair ``path`` in place. Returns whether the file needed repair.

    ``check_only`` reports without writing, mirroring
    ``scripts/canonicalise_runs.py --check``.
    """
    path = pathlib.Path(path)
    manifest = json.loads(path.read_text())
    repaired = _repaired(manifest)

    # Compare on everything except the repair stamp, so a repaired file reports
    # "no change needed" on a second pass instead of churning its timestamp.
    before = {k: v for k, v in manifest.items() if k != "repair"}
    after = {k: v for k, v in repaired.items() if k != "repair"}
    changed = before != after
    if check_only or not changed:
        return changed

    repaired["repair"] = {
        "recomputed_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "recomputed_from": "stored per-case coordinates",
        "recomputed_by": "scripts/repair_counterfactual_manifest.py",
        # Stated outright: this artifact was NOT regenerated by re-running the
        # network, so nothing downstream can mistake it for a fresh run.
        "network_reloaded": False,
        "source_generated_at": manifest.get("generated_at"),
        "note": (
            "Comparison fields recomputed with the corrected U13-B metric "
            "(axis breakdown + outcome agreement). The superseded "
            "n_agrees_with_physics_cf was removed. Stability flags and "
            "coordinates are the originals from the run at generated_at."
        ),
    }
    path.write_text(json.dumps(repaired, indent=2))
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    p.add_argument("--check", action="store_true",
                   help="Report whether a repair is needed; write nothing.")
    args = p.parse_args(argv)

    if not args.manifest.exists():
        print(f"No manifest at {args.manifest}", file=sys.stderr)
        return 2

    changed = repair_manifest(args.manifest, check_only=args.check)
    if args.check:
        print(f"{args.manifest}: {'needs repair' if changed else 'up to date'}")
        return 1 if changed else 0

    print(f"{args.manifest}: {'repaired' if changed else 'already up to date'}")
    if changed:
        dice = json.loads(args.manifest.read_text())["pooled"]["dice"]
        print(f"  outcome agreement: {dice['n_outcome_agrees']}/{dice['n_comparable']}"
              f" = {dice['outcome_agreement_rate']:.4f}")
        print(f"  both stable:       {dice['n_both_stable']}")
        print(f"  axis breakdown:    {json.dumps(dice['axis_breakdown'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
