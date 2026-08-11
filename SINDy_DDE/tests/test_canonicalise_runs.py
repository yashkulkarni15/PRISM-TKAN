"""U12 section 7: which copy of a twice-submitted sweep is the published one?

Both five-fold arrays were submitted twice, so the results tree holds two
complete copies of the ablation and two of the quantile head. The rule is that
the *published number* selects the canonical run. These tests pin that the rule
cannot quietly guess: no match refuses, and a tie is resolved from the recorded
submission order and labelled as a tie rather than as a measurement.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import canonicalise_runs as C  # noqa: E402


def _run(run_id, arm, fold, auc, tree="runs"):
    return {"run_id": run_id, "tree": tree, "path": f"results/phase3/{tree}/{run_id}",
            "arm": arm, "fold": fold, "auc": auc}


def _full(arm, auc_by_fold, **kw):
    return [_run(f"{arm}{f}", arm, f, auc_by_fold[f], **kw) for f in range(5)]


def _complete_tree():
    """One run per fold per arm, each reproducing its published figure."""
    runs = []
    for arm, spec in C.PUBLISHED.items():
        tree = spec["tree"]
        runs += [_run(f"{arm}{f}", arm, f, spec["auc"][f], tree=tree) for f in range(5)]
    return runs


def test_a_run_missing_the_side_input_key_is_not_an_ablation_run():
    """The June generation predates the key; falsy-by-absence is not falsy-by-config."""
    assert C.classify({"config": {}}) == "legacy"
    assert C.classify({"config": {"side_input": False}}) == "side_off"
    assert C.classify({"config": {"side_input": True}}) == "side_on"


def test_the_published_number_selects_the_canonical_run():
    runs = _complete_tree()
    runs.append(_run("side_on_stale", "side_on", 0, 0.9388))   # the superseded fold 0
    registry = C.build_registry(runs)
    assert registry["problems"] == []
    fold0 = registry["arms"]["side_on"]["folds"]["0"]
    assert fold0["canonical"] == "side_on0"
    assert fold0["superseded"] == ["side_on_stale"]
    assert fold0["tie_break"] is None


def test_it_refuses_when_no_run_reproduces_the_published_number():
    runs = _complete_tree()
    runs = [r for r in runs if not (r["arm"] == "quantile" and r["fold"] == 3)]
    runs.append(_run("drifted", "quantile", 3, 0.5, tree="runs_quantile"))
    registry = C.build_registry(runs)
    assert any("quantile fold 3" in p for p in registry["problems"])
    assert "3" not in registry["arms"]["quantile"]["folds"]


def test_identical_duplicates_are_broken_by_recorded_submission_order():
    """Bit-identical pairs both match, so the number cannot choose."""
    runs = _complete_tree()
    twin = _run("side_off2_twin", "side_off", 2, C.PUBLISHED["side_off"]["auc"][2])
    runs.append(twin)
    registry = C.build_registry(runs, later={("side_off", 2): "side_off2_twin"})
    entry = registry["arms"]["side_off"]["folds"]["2"]
    assert entry["canonical"] == "side_off2_twin"
    assert entry["superseded"] == ["side_off2"]
    assert "later submission" in entry["tie_break"]


def test_a_tie_with_no_recorded_order_refuses_rather_than_picking():
    runs = _complete_tree()
    runs.append(_run("side_off2_twin", "side_off", 2, C.PUBLISHED["side_off"]["auc"][2]))
    registry = C.build_registry(runs, later={})
    assert any("does not name one of them" in p for p in registry["problems"])


def test_every_run_gets_a_marker_so_unmarked_never_means_unclassified():
    runs = _complete_tree()
    runs.append(_run("legacy0", "legacy", 0, None))
    registry = C.build_registry(runs)
    for run in runs:
        marker = C.marker_for(run["run_id"], registry)
        assert marker is not None, run["run_id"]
        assert marker["status"] in {"CANONICAL", "SUPERSEDED", "UNPUBLISHED"}


def test_the_real_tree_resolves_without_a_refusal():
    """Regression pin on the checked-in results tree."""
    root = pathlib.Path(__file__).resolve().parents[1]
    runs = C.read_runs(root)
    if not runs:
        return  # results tree not synced on this machine
    registry = C.build_registry(runs, C.later_run_ids(root))
    assert registry["problems"] == []
    assert sum(len(b["folds"]) for b in registry["arms"].values()) == 15
    for run in runs:
        assert C.marker_for(run["run_id"], registry) is not None, run["run_id"]
