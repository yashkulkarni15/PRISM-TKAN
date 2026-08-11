"""Offline repair of the stored counterfactual manifest.

Update 13's generator comparison was reworked after its first result was
mislabelled: the old metric scored "chose the other knob" as "disagrees", when
``physics_constrained_counterfactual`` caps ``b`` at ``b_base`` by construction
and so *cannot* raise depth while DiCE varies both axes freely. The code was
fixed and ``n_agrees_with_physics_cf`` deleted, with a regression test asserting
it stays gone -- but the manifest on disk still carries it, because it predates
the fix.

The repair recomputes the corrected block from the coordinates and stability
flags the manifest already stores, so it needs neither the network nor a blim
evaluation. Both generators' per-case stability was computed against the real
boundary at run time and written down; agreement is a comparison of those two
recorded facts.

The load-bearing test here is ``test_repair_matches_what_the_runner_computes``:
the repair and the runner must call the *same* comparison code, or the repaired
artifact silently becomes a second implementation free to drift from the one
that produced the published 0.643.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# --------------------------------------------------------------------------- #
# Fixtures: a minimal manifest in the shape the real one has
# --------------------------------------------------------------------------- #
def _case(file, omega_base, b_base, omega_cf, b_cf, physics_stable):
    """One physics-search record, as ``per_dataset[ds]["cases"][i]``."""
    return {
        "file": file,
        "omega_base": omega_base, "b_base": b_base,
        "omega_cf": omega_cf, "b_cf": b_cf,
        "flips": True, "physics_stable_cf": physics_stable,
    }


def _dice_case(file, omega_base, b_base, omega_cf, b_cf, stable):
    """One DiCE record, as ``per_dataset[ds]["dice"]["per_case"][i]``, still
    carrying the superseded field the repair must remove."""
    return {
        "file": file,
        "omega_base": omega_base, "b_base": b_base, "found": True,
        "omega_cf": omega_cf, "b_cf": b_cf,
        "physically_stable": stable,
        "agrees_with_physics_cf": False,
    }


def _manifest(cases, dice_cases) -> dict:
    n_stable = sum(1 for d in dice_cases if d["physically_stable"])
    dice_block = {
        "backend": "dice_ml", "method": "random",
        "n_cases": len(cases), "n_found": len(dice_cases),
        "n_physically_stable": n_stable,
        "found_rate": 1.0,
        "physically_stable_rate": n_stable / len(dice_cases),
        "n_agrees_with_physics_cf": 12,          # the superseded metric
        "total_CFs": 2, "seed": 0,
        "per_case": dice_cases,
    }
    return {
        "generated_at": "2026-07-29T14:14:00",
        "tag": "fold2_net",
        "checkpoint": {"path": "best_auc.pt", "sha256": "abc123"},
        "datasets": ["1"],
        "pooled": {"n_cases": len(cases), "dice": dict(dice_block)},
        "per_dataset": {"1": {"cases": cases, "dice": dice_block}},
    }


@pytest.fixture
def manifest_path(tmp_path):
    cases = [
        # physics hops speed and lands stable; DiCE cuts depth and lands stable.
        _case("a.h5", 12000.0, 2.0, 9500.0, 2.0, True),
        # physics holds at an unstable point; DiCE also unstable -> still agreement.
        _case("b.h5", 3000.0, 3.0, 3000.0, 3.0, False),
    ]
    dice_cases = [
        _dice_case("a.h5", 12000.0, 2.0, 12000.0, 0.58, True),
        _dice_case("b.h5", 3000.0, 3.0, 3000.0, 4.0, False),
    ]
    path = tmp_path / "counterfactuals_manifest.json"
    path.write_text(json.dumps(_manifest(cases, dice_cases), indent=2))
    return path


# --------------------------------------------------------------------------- #
# The superseded metric is removed, the corrected one added
# --------------------------------------------------------------------------- #
def test_repair_removes_the_superseded_direction_metric(manifest_path):
    import repair_counterfactual_manifest as R

    R.repair_manifest(manifest_path)

    out = json.loads(manifest_path.read_text())
    assert "n_agrees_with_physics_cf" not in out["per_dataset"]["1"]["dice"]
    assert "n_agrees_with_physics_cf" not in out["pooled"]["dice"]
    for rec in out["per_dataset"]["1"]["dice"]["per_case"]:
        assert "agrees_with_physics_cf" not in rec


def test_repair_writes_outcome_agreement_and_axis_breakdown(manifest_path):
    import repair_counterfactual_manifest as R

    R.repair_manifest(manifest_path)

    dice = json.loads(manifest_path.read_text())["per_dataset"]["1"]["dice"]
    # Case a: both stable. Case b: neither stable. Both count as agreement.
    assert dice["n_comparable"] == 2
    assert dice["n_both_stable"] == 1
    assert dice["n_outcome_agrees"] == 2
    assert dice["outcome_agreement_rate"] == 1.0
    # Which lever each generator pulled, kept separate from agreement.
    assert dice["axis_breakdown"]["physics"]["omega"] == 1
    assert dice["axis_breakdown"]["physics"]["none"] == 1
    assert dice["axis_breakdown"]["dice"]["b_mm"] == 2
    assert "axis_note" in dice


def test_repair_pairs_records_by_file_not_by_position(manifest_path):
    """Index alignment between the two lists is an assumption, not a guarantee."""
    import repair_counterfactual_manifest as R

    loaded = json.loads(manifest_path.read_text())
    loaded["per_dataset"]["1"]["dice"]["per_case"].reverse()
    manifest_path.write_text(json.dumps(loaded))

    R.repair_manifest(manifest_path)

    dice = json.loads(manifest_path.read_text())["per_dataset"]["1"]["dice"]
    assert dice["n_outcome_agrees"] == 2
    assert dice["axis_breakdown"]["dice"]["b_mm"] == 2


# --------------------------------------------------------------------------- #
# Provenance: a recomputed artifact must never read as a fresh run
# --------------------------------------------------------------------------- #
def test_repair_stamps_recomputed_provenance_and_keeps_generated_at(manifest_path):
    import repair_counterfactual_manifest as R

    before = json.loads(manifest_path.read_text())["generated_at"]
    R.repair_manifest(manifest_path)

    out = json.loads(manifest_path.read_text())
    assert out["generated_at"] == before, "the original run's timestamp must stand"
    assert out["repair"]["recomputed_from"] == "stored per-case coordinates"
    assert "recomputed_at" in out["repair"]
    assert "repair_counterfactual_manifest" in out["repair"]["recomputed_by"]
    assert out["repair"]["network_reloaded"] is False


def test_repair_is_idempotent(manifest_path):
    import repair_counterfactual_manifest as R

    R.repair_manifest(manifest_path)
    first = json.loads(manifest_path.read_text())
    first["repair"].pop("recomputed_at")

    R.repair_manifest(manifest_path)
    second = json.loads(manifest_path.read_text())
    second["repair"].pop("recomputed_at")

    assert first == second


# --------------------------------------------------------------------------- #
# Cross-checks: trust the stored coordinates, verify the stored aggregates
# --------------------------------------------------------------------------- #
def test_repair_refuses_when_a_stored_aggregate_contradicts_the_records(manifest_path):
    """A disagreeing aggregate means the file is not what it claims to be, and
    silently overwriting it would destroy the evidence of that."""
    import repair_counterfactual_manifest as R

    loaded = json.loads(manifest_path.read_text())
    loaded["per_dataset"]["1"]["dice"]["n_physically_stable"] = 99
    manifest_path.write_text(json.dumps(loaded))

    with pytest.raises(ValueError, match="n_physically_stable"):
        R.repair_manifest(manifest_path)


def test_check_mode_reports_without_writing(manifest_path):
    import repair_counterfactual_manifest as R

    before = manifest_path.read_text()
    needs_repair = R.repair_manifest(manifest_path, check_only=True)

    assert needs_repair is True
    assert manifest_path.read_text() == before, "--check must not write"


def test_check_mode_is_false_once_repaired(manifest_path):
    import repair_counterfactual_manifest as R

    R.repair_manifest(manifest_path)

    assert R.repair_manifest(manifest_path, check_only=True) is False


# --------------------------------------------------------------------------- #
# The anti-drift test: repair and runner must share one implementation
# --------------------------------------------------------------------------- #
def test_repair_matches_what_the_runner_computes(monkeypatch, tmp_path):
    """Run both paths over identical inputs and demand identical comparisons.

    If these ever diverge, the repaired manifest stops being "what a re-run
    would have produced" -- which is the only claim that makes an offline repair
    legitimate in the first place.
    """
    import counterfactuals_phase3 as C
    import repair_counterfactual_manifest as R
    from pri2m.xai import counterfactuals as cf_mod

    moves = [(12000.0, 0.58), (3000.0, 4.0)]
    calls = {"i": 0}

    def _fake(estimator, query_row, desired_range, **kw):
        omega, b = moves[calls["i"]]
        calls["i"] += 1
        return pd.DataFrame({"omega": [omega], "b_mm": [b]})

    monkeypatch.setattr(cf_mod, "dice_counterfactuals", _fake)

    cf_results = [
        {"file": "a.h5", "omega_base": 12000.0, "b_base": 2.0,
         "omega_cf": 9500.0, "b_cf": 2.0, "flips": True},
        {"file": "b.h5", "omega_base": 3000.0, "b_base": 3.0,
         "omega_cf": 3000.0, "b_cf": 3.0, "flips": True},
    ]
    blim = lambda w: 2.5  # noqa: E731
    from_runner = C.run_dice(object(), {"source": "model_predictions"}, cf_results,
                             blim, flip_threshold=0.5, seed=0, total_cfs=2)

    # Same inputs, expressed as a stored manifest, through the repair path.
    cases = [
        _case("a.h5", 12000.0, 2.0, 9500.0, 2.0, 2.0 < blim(9500.0)),
        _case("b.h5", 3000.0, 3.0, 3000.0, 3.0, 3.0 < blim(3000.0)),
    ]
    dice_cases = [
        _dice_case("a.h5", 12000.0, 2.0, 12000.0, 0.58, 0.58 < blim(12000.0)),
        _dice_case("b.h5", 3000.0, 3.0, 3000.0, 4.0, 4.0 < blim(3000.0)),
    ]
    path = tmp_path / "m.json"
    path.write_text(json.dumps(_manifest(cases, dice_cases)))
    R.repair_manifest(path)
    from_repair = json.loads(path.read_text())["per_dataset"]["1"]["dice"]

    for key in ("n_comparable", "n_both_stable", "n_outcome_agrees",
                "outcome_agreement_rate", "n_same_axis", "axis_breakdown"):
        assert from_repair[key] == from_runner[key], f"{key} drifted"
