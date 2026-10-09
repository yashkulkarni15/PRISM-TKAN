"""Selecting the comb-carrying edges, and building the controls that judge them.

Update 15 Sections 2 and 3 describe the delay coupling as a comb filter and
confirm the description by probe, but nothing yet DELETES those pairs. This
module is the causal arm. The tests below pin the two things that could quietly
make it lie: selecting a different set of pairs than the reading describes, and
building a control that is easier to delete than the target.
"""
from __future__ import annotations

import numpy as np
import pytest

from pri2m.xai import comb
from pri2m.xai.comb_ablation import (
    matched_control,
    select_comb_edges,
    summed_importance,
)


def _row(out, in_inst, in_delay, corr, *, gain=1.0, degenerate=False,
         imp_inst=1.0, imp_delay=1.0, kan="encoder.layers.0.kan_z"):
    return {"kan": kan, "out": out, "in_inst": in_inst, "in_delay": in_delay,
            "corr": corr, "gain": gain, "degenerate": degenerate,
            "importance_inst": imp_inst, "importance_delay": imp_delay}


# --------------------------------------------------------------------------- #
# select_comb_edges
# --------------------------------------------------------------------------- #
def test_selects_both_members_of_a_qualifying_pair():
    """The comb is a property of the PAIR, so deleting one half is not the test."""
    edges = select_comb_edges([_row(58, 0, 68, 0.99)], corr_threshold=0.9)

    assert {(e["kan"], e["out"], e["in"]) for e in edges} == {
        ("encoder.layers.0.kan_z", 58, 0),
        ("encoder.layers.0.kan_z", 58, 68),
    }


def test_excludes_pairs_below_the_correlation_threshold():
    edges = select_comb_edges(
        [_row(58, 0, 68, 0.99), _row(59, 1, 69, 0.42)], corr_threshold=0.9)

    assert all(e["out"] == 58 for e in edges)


def test_a_strongly_negative_correlation_still_qualifies():
    """comb._stats gates on ABS(corr); a sign flip is a cancelling comb, not an absent one."""
    edges = select_comb_edges([_row(58, 0, 68, -0.97)], corr_threshold=0.9)

    assert len(edges) == 2


def test_degenerate_rows_are_excluded_exactly_as_the_reading_excludes_them():
    edges = select_comb_edges([_row(58, 0, 68, 0.99, degenerate=True)],
                              corr_threshold=0.9)

    assert edges == []


def test_an_edge_shared_by_two_pairs_is_deleted_once():
    """Double-counting would inflate the arm's size against its count-matched control."""
    rows = [_row(58, 0, 68, 0.99), _row(58, 0, 69, 0.98)]

    edges = select_comb_edges(rows, corr_threshold=0.9)

    assert len(edges) == 3
    assert len({(e["kan"], e["out"], e["in"]) for e in edges}) == 3


def test_selection_matches_the_published_reading_on_a_real_manifest(tmp_path):
    """The count this deletes must equal the count Section 2 says it describes."""
    rows = [_row(o, 0, 68, 0.95) for o in range(10)]
    rows += [_row(o, 1, 69, 0.10) for o in range(10, 20)]

    n_used = comb.aggregate_comb(rows, corr_threshold=0.9)["n_used"]
    pairs = len(select_comb_edges(rows, corr_threshold=0.9)) // 2

    assert pairs == n_used


# --------------------------------------------------------------------------- #
# matched_control
# --------------------------------------------------------------------------- #
def test_control_matches_the_target_on_count():
    target = [{"kan": "k", "out": 0, "in": i, "importance": 1.0} for i in range(5)]
    pool = [{"kan": "k", "out": 1, "in": i, "importance": 1.0} for i in range(50)]

    ctrl = matched_control(pool, target, seed=0)

    assert len(ctrl) == len(target)


def test_control_is_drawn_to_match_summed_importance_not_only_count():
    """A count-matched control that carries half the importance tests size, not identity."""
    target = [{"kan": "k", "out": 0, "in": i, "importance": 5.0} for i in range(4)]
    pool = ([{"kan": "k", "out": 1, "in": i, "importance": 5.0} for i in range(20)]
            + [{"kan": "k", "out": 2, "in": i, "importance": 0.01} for i in range(20)])

    ctrl = matched_control(pool, target, seed=0, n_trials=200)

    assert summed_importance(ctrl) == pytest.approx(summed_importance(target), rel=0.15)


def test_control_never_contains_a_target_edge():
    """Deleting part of the target inside the control shrinks the very gap being measured."""
    target = [{"kan": "k", "out": 0, "in": i, "importance": 1.0} for i in range(5)]
    pool = target + [{"kan": "k", "out": 1, "in": i, "importance": 1.0} for i in range(20)]

    ctrl = matched_control(pool, target, seed=3)

    keys = {(e["kan"], e["out"], e["in"]) for e in ctrl}
    assert keys.isdisjoint({(e["kan"], e["out"], e["in"]) for e in target})


def test_control_is_deterministic_for_a_seed():
    target = [{"kan": "k", "out": 0, "in": i, "importance": 1.0} for i in range(4)]
    pool = [{"kan": "k", "out": 1, "in": i, "importance": float(i)} for i in range(40)]

    a = matched_control(pool, target, seed=7)
    b = matched_control(pool, target, seed=7)

    assert a == b


def test_too_small_a_pool_raises_rather_than_returning_a_short_arm():
    """A silently short control would be compared against a full-size target."""
    target = [{"kan": "k", "out": 0, "in": i, "importance": 1.0} for i in range(10)]
    pool = [{"kan": "k", "out": 1, "in": i, "importance": 1.0} for i in range(3)]

    with pytest.raises(ValueError, match="cannot draw"):
        matched_control(pool, target, seed=0)


def test_non_proportional_pool_excludes_every_qualifying_pair():
    """The sharp control: delay pairs that carry no comb, matched on importance."""
    rows = [_row(0, 0, 68, 0.99, imp_inst=2.0, imp_delay=2.0),
            _row(1, 1, 69, 0.20, imp_inst=2.0, imp_delay=2.0)]

    keep = select_comb_edges(rows, corr_threshold=0.9)
    drop = select_comb_edges(rows, corr_threshold=0.9, below_threshold=True)

    kept = {(e["kan"], e["out"], e["in"]) for e in keep}
    dropped = {(e["kan"], e["out"], e["in"]) for e in drop}
    assert kept and dropped and kept.isdisjoint(dropped)


# --------------------------------------------------------------------------- #
# build_controls
#
# On the real manifests fold 1 has 854 comb edges but only 682 below the gate,
# so its importance-matched non-proportional control CANNOT be built. Three
# things must be true of that case: it is detected before anything expensive
# runs, the arm is recorded as absent rather than silently shortened, and the
# fold ends UNDECIDED rather than passing on one control.
# --------------------------------------------------------------------------- #
def _edges(n, out, imp=1.0):
    return [{"kan": "k", "out": out, "in": i, "importance": imp} for i in range(n)]


def test_build_controls_returns_both_arms_when_both_pools_suffice():
    target = _edges(5, 0)
    controls, errors = __import__(
        "pri2m.xai.comb_ablation", fromlist=["build_controls"]
    ).build_controls(population=_edges(50, 1), below_gate=_edges(50, 2),
                     target=target, seed=0)

    assert set(controls) == {"random_matched", "nonproportional_matched"}
    assert errors == {}


def test_build_controls_records_the_unbuildable_arm_instead_of_shortening_it():
    from pri2m.xai.comb_ablation import build_controls

    controls, errors = build_controls(population=_edges(50, 1),
                                      below_gate=_edges(3, 2),
                                      target=_edges(10, 0), seed=0)

    assert "nonproportional_matched" not in controls
    assert "cannot draw" in errors["nonproportional_matched"]
    assert len(controls["random_matched"]) == 10


def test_build_controls_reports_both_when_neither_pool_suffices():
    from pri2m.xai.comb_ablation import build_controls

    controls, errors = build_controls(population=_edges(2, 1),
                                      below_gate=_edges(2, 2),
                                      target=_edges(10, 0), seed=0)

    assert controls == {}
    assert set(errors) == {"random_matched", "nonproportional_matched"}


def test_a_missing_control_leaves_the_verdict_undecided_not_passing():
    """The rule that killed the delay-lock: a control that did not run is not a pass."""
    from pri2m.xai.mode_ablation import ablation_verdict

    v = ablation_verdict(baseline_auc=0.90, mode_auc=0.80,
                         random_matched_auc=0.89, other_modes_auc=None, floor=0.0092)

    assert v["claimable"] is None
    assert v["undecided_reason"] == "missing_arm"


# --------------------------------------------------------------------------- #
# Regression: the control population must be the WHOLE network
#
# The first cluster run drew random_matched from the delay pairs alone. Because
# the comb set and the below-gate set exhaust that population (fold 1: 854 + 682
# = 1536), excluding the target left exactly the below-gate edges, so the two
# controls were the same pool and the "is it these edges or just this much
# capacity" question was never asked. Four folds ran it without erroring; only
# fold 1's arithmetic made it impossible and surfaced the bug.
# --------------------------------------------------------------------------- #
def test_population_from_ranked_converts_rank_edges_output():
    from pri2m.xai.comb_ablation import population_from_ranked

    pop = population_from_ranked([("kan_z", 3, 68, 1.5), ("kan_r", 1, 0, 0.25)])

    assert pop == [{"kan": "kan_z", "out": 3, "in": 68, "importance": 1.5},
                   {"kan": "kan_r", "out": 1, "in": 0, "importance": 0.25}]


def test_a_population_confined_to_the_delay_pairs_is_refused():
    """The exact bug: pool == target + below_gate makes both controls one control."""
    from pri2m.xai.comb_ablation import check_population_is_wider

    target = _edges(3, 0)
    below = _edges(3, 1)

    with pytest.raises(ValueError, match="not wider"):
        check_population_is_wider(target + below, target=target, below_gate=below)


def test_a_population_reaching_beyond_the_delay_pairs_is_accepted():
    from pri2m.xai.comb_ablation import check_population_is_wider

    target, below = _edges(3, 0), _edges(3, 1)

    check_population_is_wider(target + below + _edges(50, 9),
                              target=target, below_gate=below)


def test_the_runner_only_calls_xai_phase3_attributes_that_exist():
    """Regression: the first version called xai_phase3._resolve_preprocessing.

    No such function exists -- the real name is _find_preprocessing -- and four
    of the five cluster jobs died on that line. Nothing local caught it because
    --help never reaches it and there is no checkpoint on this machine to drive
    the model-loading path. Same family as the U14-B callback arity bug: the
    runner/library seam is exactly where an untested call hides.
    """
    import ast
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    import xai_phase3

    src = (root / "scripts" / "comb_ablation_phase3.py").read_text()
    referenced = {
        node.attr for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name) and node.value.id == "xai_phase3"
    }

    assert referenced, "test found no xai_phase3 calls; it would pass vacuously"
    missing = sorted(a for a in referenced if not hasattr(xai_phase3, a))
    assert not missing, f"runner calls xai_phase3 attributes that do not exist: {missing}"
