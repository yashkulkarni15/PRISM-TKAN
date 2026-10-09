"""Causal mode ablation: remove the edges, measure what breaks.

U14 section 4 can say that trained networks cluster their activation shapes more
consistently than untrained ones, and cannot turn that into a verdict, because
the pass/fail flips with the number of shapes requested and that number was not
fixed in advance. An ablation does not care: zeroing a set of edges and watching
held-out accuracy is a causal measurement that does not depend on the grouping
being reproducible.

``xai/activation_modes.mode_edge_sets`` already returns the three sets this
needs -- the mode's edges, a random set matched on **count and summed
importance**, and an equal-size set drawn from the other modes -- and has been
tested since 2026-07-30. Nothing ever consumed it. This module is the missing
half: apply an edge set to a model, and put the three arms side by side.

The controls are not optional here. "Removing 300 edges cost 0.04 AUC" is
meaningless without knowing what removing 300 *arbitrary* edges of the same
total importance costs, which is exactly what the matched random arm measures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pri2m.models.tkan_model import TKANModel
from pri2m.xai.mode_ablation import (
    ablation_verdict,
    edges_are_zeroed,
    zero_edges,
)


def _model():
    torch.manual_seed(0)
    return TKANModel(input_size=4, hidden_size=8)


def _edges():
    return [{"kan": "encoder.layers.0.kan_z", "out": 1, "in": 0},
            {"kan": "encoder.layers.0.kan_r", "out": 2, "in": 3}]


# --------------------------------------------------------------------------- #
# Zeroing an edge, and putting it back
# --------------------------------------------------------------------------- #
def test_zeroing_removes_the_edge_from_both_paths():
    """An edge is base + spline. Zeroing only the spline leaves the SiLU path
    still contributing, which would understate every ablation."""
    model = _model()

    with zero_edges(model, _edges()):
        assert edges_are_zeroed(model, _edges())


def test_the_model_is_restored_exactly_on_exit():
    """Arms are measured on the same model in sequence, so an ablation that
    leaks would silently contaminate every arm after it."""
    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    with zero_edges(model, _edges()):
        pass

    after = model.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"{k} was not restored"


def test_restoration_survives_an_exception():
    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    with pytest.raises(RuntimeError):
        with zero_edges(model, _edges()):
            raise RuntimeError("boom")

    for k, v in before.items():
        assert torch.equal(v, model.state_dict()[k])


def test_zeroing_actually_changes_the_output():
    """A no-op ablation would report every mode as unnecessary."""
    model = _model()
    batch = {
        "window": torch.randn(2, 64, 4),
        "tau": torch.tensor([8, 8]),
    }
    with torch.no_grad():
        base = model(batch)["chatter_logit"].clone()
        with zero_edges(model, _edges()):
            ablated = model(batch)["chatter_logit"].clone()

    assert not torch.allclose(base, ablated), "zeroing these edges changed nothing"


def test_an_unknown_edge_is_refused_rather_than_ignored():
    """Silently skipping a bad edge would shrink the arm without saying so, and
    the arms are matched on COUNT."""
    model = _model()

    with pytest.raises((KeyError, AttributeError, IndexError)):
        with zero_edges(model, [{"kan": "encoder.layers.0.kan_nope", "out": 0, "in": 0}]):
            pass


def test_edges_are_zeroed_reports_false_before_any_ablation():
    model = _model()

    assert edges_are_zeroed(model, _edges()) is False


# --------------------------------------------------------------------------- #
# Reading the three arms
# --------------------------------------------------------------------------- #
def test_verdict_requires_the_mode_to_beat_both_controls():
    """The whole point of the controls: a mode that hurts no more than an
    arbitrary set of the same size and importance has shown nothing."""
    out = ablation_verdict(baseline_auc=0.90, mode_auc=0.70,
                           random_matched_auc=0.88, other_modes_auc=0.87,
                           floor=0.01)

    assert out["mode_drop"] == pytest.approx(0.20)
    assert out["beats_random_control"] is True
    assert out["beats_other_modes_control"] is True
    assert out["claimable"] is True


def test_a_mode_that_matches_its_random_control_is_not_claimable():
    # 0.705 rather than 0.71: the latter puts the differential exactly ON the
    # floor, where float arithmetic decides the answer instead of the rule.
    out = ablation_verdict(baseline_auc=0.90, mode_auc=0.70,
                           random_matched_auc=0.705, other_modes_auc=0.87,
                           floor=0.01)

    assert out["beats_random_control"] is False
    assert out["claimable"] is False


def test_an_effect_under_the_repeatability_floor_is_not_claimable():
    """Update 14 measured a floor of 0.0092 AUC between two runs of the same
    configuration. An ablation smaller than that is not an effect."""
    out = ablation_verdict(baseline_auc=0.9000, mode_auc=0.8950,
                           random_matched_auc=0.8999, other_modes_auc=0.8998,
                           floor=0.0092)

    assert out["exceeds_floor"] is False
    assert out["claimable"] is False


def test_the_real_clustering_pipeline_feeds_the_ablation():
    """Integration across the seam, because that seam had two bugs.

    fold_bundles returns one bundle PER STRATUM rather than one bundle, and
    assign_to_prototypes returns (labels, distances) rather than labels. Both
    were wrong in the first runner and neither would have surfaced in a unit
    test of either side. This drives the real chain end to end.
    """
    from pri2m.xai import activation_modes as AM

    torch.manual_seed(0)
    model = TKANModel(input_size=4, hidden_size=64)
    batch = {"window": torch.randn(8, 256, 4), "tau": torch.full((8,), 20)}

    bundles = AM.fold_bundles(model, batch, n=64)
    assert "encoder.L0.x_delay" in bundles, "the delayed-channel stratum must exist"

    gated = AM.gate_by_importance(bundles["encoder.L0.x_delay"], quantile=0.9)
    norm = AM.normalise_curves(gated)
    protos = AM.fit_prototypes(norm["curves"], 4, seed=0)
    labels, _distances = AM.assign_to_prototypes(norm["curves"], protos)

    sets = AM.mode_edge_sets(norm, labels, mode=int(np.bincount(labels).argmax()),
                             seed=0, population=bundles["encoder.L0.x_delay"])

    # The arms are matched on count; if they are not, the comparison is void.
    n = len(sets["mode_edges"])
    assert n > 0
    assert len(sets["random_matched"]) == n
    assert len(sets["other_modes_matched"]) == n

    # And the edge records must carry exactly what zero_edges indexes on.
    assert {"kan", "out", "in"} <= set(sets["mode_edges"][0])

    with torch.no_grad():
        base = model(batch)["chatter_logit"].clone()
        with zero_edges(model, sets["mode_edges"]):
            assert edges_are_zeroed(model, sets["mode_edges"])
            ablated = model(batch)["chatter_logit"].clone()
    assert not torch.allclose(base, ablated)


def test_verdict_is_undecided_when_an_arm_is_missing():
    """A control that failed to evaluate must never read as a control that passed."""
    out = ablation_verdict(baseline_auc=0.90, mode_auc=0.70,
                           random_matched_auc=None, other_modes_auc=0.87,
                           floor=0.01)

    assert out["claimable"] is None
    assert out["undecided_reason"] == "missing_arm"


# --------------------------------------------------------------------------- #
# Runner/library seam: the evaluator's return shape (regression, 2026-08-12)
# --------------------------------------------------------------------------- #


def test_score_reads_the_auc_out_of_the_nested_metrics_block(tmp_path, monkeypatch):
    """Job 57251412 died reporting "Baseline evaluation failed: None" after a
    clean evaluation that printed auc=0.8994.

    eval_heldout_phase3.run returns the AUC at result["metrics"]["auc"]
    (scripts/eval_heldout_phase3.py:196-213), and _score was reading the top
    level, which is always absent.
    """
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mode_ablation_phase3", root / "scripts" / "mode_ablation_phase3.py"
    )
    MA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MA)

    class _Args:
        features_csv = "f.csv"
        npz_root = "n"
        preprocessing = None
        device = "cpu"
        batch_size = 64

    monkeypatch.setattr(
        MA.EH, "run",
        lambda **kw: {"datasets": [121], "metrics": {"auc": 0.8994, "mae": 4.5}},
    )
    auc, err = MA._score("x.pt", _Args(), None, object(), "t", tmp_path)
    assert auc == 0.8994 and err is None

    monkeypatch.setattr(MA.EH, "run", lambda **kw: {"auc": 0.9})
    auc, err = MA._score("x.pt", _Args(), None, object(), "t", tmp_path)
    assert auc is None and "metrics.auc" in err


# --------------------------------------------------------------------------- #
# What the manifest records
#
# The k=2/k=3/k=4 runs each found a group of about thirty edges, and whether
# they are the SAME group cannot be checked from the stored manifests, because
# they record how many edges a group has and not which ones. That is a gap in
# the record rather than in the analysis, and this is the seam that closes it.
# --------------------------------------------------------------------------- #
def _runner():
    import importlib.util
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    for p in (str(root / "src"), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "mode_ablation_phase3", root / "scripts" / "mode_ablation_phase3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sets():
    return {
        "mode_edges": [{"kan": "kan_z", "out": 3, "in": 68, "importance": 1.5}],
        "random_matched": [{"kan": "kan_z", "out": 9, "in": 2, "importance": 1.4}],
        "other_modes_matched": [{"kan": "kan_r", "out": 4, "in": 69, "importance": 1.5}],
        "summed_importance": {"mode_edges": 1.5},
        "random_overlap_with_mode": 0,
    }


def test_mode_record_carries_edge_membership_not_only_counts():
    rec = _runner().mode_record(
        "encoder.L0.x_delay", 2, _sets(),
        {"mode_edges": (0.88, None)}, {"claimable": True})

    assert rec["edges"]["mode_edges"] == [
        {"kan": "kan_z", "out": 3, "in": 68, "importance": 1.5}]


def test_mode_record_also_records_which_edges_each_control_deleted():
    """Without these, a control cannot be reproduced or checked after the fact."""
    rec = _runner().mode_record(
        "encoder.L0.x_delay", 2, _sets(), {}, {"claimable": None})

    assert rec["edges"]["random_matched"][0]["out"] == 9
    assert rec["edges"]["other_modes_matched"][0]["in"] == 69


def test_mode_record_keeps_the_count_it_already_published():
    """The n_edges field other tooling reads must not move because edges were added."""
    rec = _runner().mode_record("s", 0, _sets(), {}, {})

    assert rec["n_edges"] == 1


# --------------------------------------------------------------------------- #
# Exp 1b: deleting two clusters at once, and every gated edge
#
# Fold 4 has no single necessary cluster. The live hypothesis is that its
# clusters cover for each other, which is only testable by removing two at once:
# if the pair hurts more than its two singles added together, the clusters were
# redundant, not unimportant. These tests pin the relabelling trick the pair arm
# rides on, the pre-committed reads, and that the runner leaves the landed Exp 1
# manifest shape alone when the new flags are absent.
# --------------------------------------------------------------------------- #
def _synthetic_bundle(n=12, seed=0):
    """A bundle shaped like activation_modes' output, small enough to reason about."""
    rng = np.random.default_rng(seed)
    records = [{"kan": "encoder.layers.0.kan_z", "out": i % 8, "in": i // 8,
                "feature": "x_delay", "importance": float(v)}
               for i, v in enumerate(rng.uniform(0.5, 2.0, size=n))]
    return {"n_edges": n, "records": records, "curves": np.zeros((n, 4)),
            "importance": np.array([r["importance"] for r in records])}


def _key(e):
    return (e["kan"], int(e["out"]), int(e["in"]))


def test_mode_edge_sets_accepts_the_relabel_sentinel():
    """The pair arm rides on mode_edge_sets(labels == -1). Pin that it works,
    rather than assume it: an off-by-one in a label check would hand back an
    empty set and a vacuous "no effect"."""
    from pri2m.xai import activation_modes as AM

    b = _synthetic_bundle()
    labels = np.array([-1, -1, 0, 0, 1, 1, -1, 2, 2, 2, 0, 1])
    sets = AM.mode_edge_sets(b, labels, mode=-1, seed=0, population=b)

    assert sorted(_key(e) for e in sets["mode_edges"]) == sorted(
        _key(b["records"][i]) for i in (0, 1, 6))


def test_combined_edge_sets_is_the_union_of_the_two_modes():
    from pri2m.xai import activation_modes as AM
    from pri2m.xai.mode_ablation import combined_edge_sets

    b = _synthetic_bundle()
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
    sets = combined_edge_sets(b, labels, [2, 0], seed=0, population=b)

    singles = set()
    for m in (0, 2):
        singles |= {_key(e) for e in
                    AM.mode_edge_sets(b, labels, mode=m, seed=0, population=b)["mode_edges"]}
    assert {_key(e) for e in sets["mode_edges"]} == singles
    assert sets["modes"] == [0, 2]
    # Matched on count, or the comparison is void.
    assert len(sets["random_matched"]) == len(sets["mode_edges"]) == 8
    # Only mode 1 is left to draw an other-modes set from, and it has 4 edges.
    assert len(sets["other_modes_matched"]) == 4


def test_combined_edge_sets_over_every_mode_has_no_other_modes_set():
    """The all-gated arm: nothing is left over, so the other-modes control is
    empty by construction -- expected, not a failure."""
    from pri2m.xai.mode_ablation import combined_edge_sets

    b = _synthetic_bundle()
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
    sets = combined_edge_sets(b, labels, [0, 1, 2], seed=0, population=b)

    assert len(sets["mode_edges"]) == 12
    assert sets["other_modes_matched"] == []
    assert len(sets["random_matched"]) == 12


def test_combined_edge_sets_is_deterministic():
    from pri2m.xai.mode_ablation import combined_edge_sets

    b = _synthetic_bundle(n=40)
    pop = _synthetic_bundle(n=80, seed=1)
    labels = np.arange(40) % 3
    one = combined_edge_sets(b, labels, [0, 1], seed=0, population=pop)
    two = combined_edge_sets(b, labels, [0, 1], seed=0, population=pop)

    assert one["random_matched"] == two["random_matched"]
    assert one["mode_edges"] == two["mode_edges"]


def test_combined_edge_sets_refuses_a_mode_that_is_not_there():
    """Silently deleting one cluster under a "pair" label would be read as a
    pair result."""
    from pri2m.xai.mode_ablation import combined_edge_sets

    b = _synthetic_bundle()
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
    with pytest.raises(ValueError):
        combined_edge_sets(b, labels, [0, 2], seed=0, population=b)


def test_combined_edge_sets_does_not_mutate_the_labels():
    from pri2m.xai.mode_ablation import combined_edge_sets

    b = _synthetic_bundle()
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
    before = labels.copy()
    combined_edge_sets(b, labels, [0, 1], seed=0, population=b)

    assert np.array_equal(labels, before)


# -- the pre-committed reads ------------------------------------------------- #
def test_pair_verdict_super_additive_and_claimable():
    """The fold-4 hypothesis in numbers: neither single hurts, the pair does."""
    from pri2m.xai.mode_ablation import pair_verdict

    out = pair_verdict(baseline_auc=0.90, pair_auc=0.80, random_matched_auc=0.89,
                       single_drops={0: 0.005, 1: 0.004}, floor=0.0092)

    assert out["pair_drop"] == pytest.approx(0.10)
    assert out["random_matched_drop"] == pytest.approx(0.01)
    assert out["exceeds_floor"] is True
    assert out["beats_random_control"] is True
    assert out["claimable"] is True
    assert out["interaction"] == pytest.approx(0.10 - 0.009)
    assert out["interaction_class"] == "super_additive"
    assert out["undecided_reason"] is None


def test_pair_verdict_sub_additive():
    from pri2m.xai.mode_ablation import pair_verdict

    out = pair_verdict(baseline_auc=0.90, pair_auc=0.85, random_matched_auc=0.90,
                       single_drops={0: 0.04, 1: 0.04}, floor=0.0092)

    assert out["interaction"] == pytest.approx(0.05 - 0.08)
    assert out["interaction_class"] == "sub_additive"
    assert out["claimable"] is True


def test_pair_verdict_additive_inside_the_floor():
    from pri2m.xai.mode_ablation import pair_verdict

    out = pair_verdict(baseline_auc=0.90, pair_auc=0.82, random_matched_auc=0.90,
                       single_drops={0: 0.04, 1: 0.035}, floor=0.0092)

    assert out["interaction"] == pytest.approx(0.08 - 0.075)
    assert out["interaction_class"] == "additive"


def test_pair_verdict_not_claimable_when_random_hurts_as_much():
    from pri2m.xai.mode_ablation import pair_verdict

    out = pair_verdict(baseline_auc=0.90, pair_auc=0.80, random_matched_auc=0.805,
                       single_drops={0: 0.0, 1: 0.0}, floor=0.0092)

    assert out["exceeds_floor"] is True
    assert out["beats_random_control"] is False
    assert out["claimable"] is False
    # Interaction is still a measured quantity; claimability does not gate it.
    assert out["interaction_class"] == "super_additive"


def test_pair_verdict_is_undecided_when_an_arm_is_missing():
    """A random control that failed to evaluate must never read as one that passed."""
    from pri2m.xai.mode_ablation import pair_verdict

    for kw in ({"pair_auc": None, "random_matched_auc": 0.89},
               {"pair_auc": 0.80, "random_matched_auc": None}):
        out = pair_verdict(baseline_auc=0.90, single_drops={0: 0.0, 1: 0.0},
                           floor=0.0092, **kw)
        assert out["claimable"] is None
        assert out["exceeds_floor"] is None
        assert out["interaction"] is None
        assert out["interaction_class"] is None
        assert out["undecided_reason"] == "missing_arm"

    out = pair_verdict(baseline_auc=None, pair_auc=0.8, random_matched_auc=0.89,
                       single_drops={0: 0.0, 1: 0.0}, floor=0.0092)
    assert out["claimable"] is None and out["undecided_reason"] == "missing_arm"


def test_pair_verdict_missing_single_drop_leaves_claimable_but_not_interaction():
    """The pair's own arms decide claimability; only the interaction needs the
    singles. Losing a single must not void a measured pair."""
    from pri2m.xai.mode_ablation import pair_verdict

    out = pair_verdict(baseline_auc=0.90, pair_auc=0.80, random_matched_auc=0.89,
                       single_drops={0: 0.005, 1: None}, floor=0.0092)

    assert out["claimable"] is True
    assert out["interaction"] is None
    assert out["interaction_class"] is None
    assert out["undecided_reason"] == "missing_single_drop"


def test_all_gated_verdict_has_no_interaction():
    from pri2m.xai.mode_ablation import all_gated_verdict

    out = all_gated_verdict(baseline_auc=0.90, all_auc=0.70,
                            random_matched_auc=0.88, floor=0.0092)
    assert out["all_drop"] == pytest.approx(0.20)
    assert out["claimable"] is True
    assert "interaction" not in out

    out = all_gated_verdict(baseline_auc=0.90, all_auc=0.70,
                            random_matched_auc=None, floor=0.0092)
    assert out["claimable"] is None and out["undecided_reason"] == "missing_arm"


# -- the runner, with the heavy pieces stubbed -------------------------------- #
_OLD_MANIFEST_KEYS = {"analysis", "provenance", "baseline_auc", "modes", "note",
                      "complete"}
_OLD_PROVENANCE_KEYS = {"checkpoint", "k", "seed", "strata", "gate_quantile",
                        "n_windows", "floor", "features_csv"}
_LABELS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])


def _stub_runner(monkeypatch, tmp_path, auc_for=None):
    """The runner with model loading, batching, clustering and scoring stubbed.

    mode_edge_sets / combined_edge_sets stay REAL: the relabel seam is the thing
    under test. The model is a real tiny TKAN so zero_edges and the
    edges_are_zeroed guard run for real against the synthetic edge records.
    """
    import types

    MA = _runner()
    from pri2m.xai import activation_modes as AM_real

    model = _model()
    bundle = _synthetic_bundle()
    monkeypatch.setattr(MA.xai_phase3, "_load_model", lambda *a, **k: model)
    monkeypatch.setattr(MA, "window_config_from_preprocessing", lambda p: None)
    monkeypatch.setattr(MA, "build_labeled_batch", lambda *a, **k: (None, None))
    monkeypatch.setattr(MA, "batch_to_model_device", lambda b, m: b)
    monkeypatch.setattr(MA, "AM", types.SimpleNamespace(
        fold_bundles=lambda m, b, n: {"encoder.L0.x_delay": bundle},
        gate_by_importance=lambda b, quantile: b,
        normalise_curves=lambda b: b,
        fit_prototypes=lambda curves, k, seed: None,
        assign_to_prototypes=lambda curves, protos: (_LABELS.copy(), None),
        mode_edge_sets=AM_real.mode_edge_sets,
    ))
    calls = []
    auc_for = auc_for or (lambda tag: 0.90 if tag == "baseline" else 0.88)

    def _score(ckpt, args, pre, mdl, tag, out_dir):
        calls.append(tag)
        return auc_for(tag), None

    monkeypatch.setattr(MA, "_score", _score)
    ckpt = tmp_path / "run" / "checkpoints" / "best_auc.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    base = ["--checkpoint", str(ckpt), "--features-csv", "f.csv", "--npz-root", "n",
            "--k", "3", "--output-dir", str(tmp_path / "out"), "--tag", "t"]
    return MA, base, calls


def _manifest(tmp_path):
    import json
    return json.loads((tmp_path / "out" / "mode_ablation_t.json").read_text())


def test_runner_without_the_new_flags_writes_the_exp1_manifest_shape(tmp_path, monkeypatch):
    """The landed Exp 1 manifests must stay reproducible: no new keys, and no
    extra evaluations, unless a flag asks for them."""
    MA, base, calls = _stub_runner(monkeypatch, tmp_path)
    assert MA.main(base) == 0
    m = _manifest(tmp_path)

    assert set(m) == _OLD_MANIFEST_KEYS
    assert set(m["provenance"]) == _OLD_PROVENANCE_KEYS
    assert len(calls) == 1 + 3 * 3
    assert not any("_p" in c.split("x_delay")[-1] or "_all_" in c for c in calls)


def test_runner_with_pairs_and_all_gated_records_both(tmp_path, monkeypatch):
    s = "encoder.L0.x_delay"
    table = {"baseline": 0.90,
             f"{s}_m0_mode_edges": 0.895, f"{s}_m1_mode_edges": 0.896,
             f"{s}_m2_mode_edges": 0.70,
             f"{s}_p01_pair_edges": 0.80, f"{s}_p01_random_matched": 0.89}
    MA, base, calls = _stub_runner(monkeypatch, tmp_path,
                                   auc_for=lambda tag: table.get(tag, 0.88))
    assert MA.main(base + ["--pairs", "--all-gated"]) == 0
    m = _manifest(tmp_path)

    assert m["provenance"]["pairs"] is True and m["provenance"]["all_gated"] is True
    # Singles first, untouched, so they reproduce Exp 1.
    assert calls[:10] == ["baseline"] + [f"{s}_m{k}_{a}" for k in range(3) for a in
                                         ("mode_edges", "random_matched",
                                          "other_modes_matched")]
    assert [p["modes"] for p in m["pairs"]] == [[0, 1], [0, 2], [1, 2]]
    for p in m["pairs"]:
        assert set(p["arms"]) == {"pair_edges", "random_matched"}
        assert set(p["edges"]) == {"pair_edges", "random_matched"}
        assert p["n_edges"] == 8 == len(p["edges"]["pair_edges"])
        assert len(p["edges"]["random_matched"]) == 8
        assert p["stratum"] == s

    p01 = m["pairs"][0]
    assert p01["verdict"]["pair_drop"] == pytest.approx(0.10)
    # Interaction uses THIS run's single drops (0.005, 0.004).
    assert p01["verdict"]["interaction"] == pytest.approx(0.10 - 0.009)
    assert p01["verdict"]["interaction_class"] == "super_additive"
    assert p01["verdict"]["claimable"] is True

    (ag,) = m["all_gated"]
    assert ag["modes"] == [0, 1, 2]
    assert ag["n_edges"] == 12
    assert set(ag["arms"]) == {"all_edges", "random_matched"}
    assert set(ag["edges"]) == {"all_edges", "random_matched"}
    assert "interaction" not in ag["verdict"]
    assert len(calls) == 1 + 9 + 2 * 3 + 2


def test_runner_pairs_alone_does_not_run_all_gated(tmp_path, monkeypatch):
    MA, base, calls = _stub_runner(monkeypatch, tmp_path)
    assert MA.main(base + ["--pairs"]) == 0
    m = _manifest(tmp_path)

    assert m["provenance"]["pairs"] is True and m["provenance"]["all_gated"] is False
    assert len(m["pairs"]) == 3
    assert m["all_gated"] == []


def test_runner_a_failed_pair_arm_is_undecided_not_false(tmp_path, monkeypatch):
    s = "encoder.L0.x_delay"
    MA, base, calls = _stub_runner(
        monkeypatch, tmp_path,
        auc_for=lambda t: None if t == f"{s}_p12_random_matched" else
        (0.90 if t == "baseline" else 0.88))
    assert MA.main(base + ["--pairs"]) == 0
    p12 = _manifest(tmp_path)["pairs"][2]

    assert p12["modes"] == [1, 2]
    assert p12["verdict"]["claimable"] is None
    assert p12["verdict"]["undecided_reason"] == "missing_arm"


def _write_reference(tmp_path, mutate=None):
    """A reference manifest built from a plain (no-flag) run of the same stub."""
    import json
    ref = _manifest(tmp_path)
    if mutate:
        mutate(ref)
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref))
    return p


def test_reference_check_confirms_an_identical_clustering(tmp_path, monkeypatch):
    MA, base, _ = _stub_runner(monkeypatch, tmp_path)
    assert MA.main(base) == 0
    ref = _write_reference(tmp_path)

    assert MA.main(base + ["--pairs", "--reference-manifest", str(ref)]) == 0
    rc = _manifest(tmp_path)["reference_check"]

    assert rc["manifest"] == str(ref)
    assert rc["modes_identical"] is True
    assert all(v["identical"] is True for v in rc["per_mode"].values())
    assert len(rc["per_mode"]) == 3


def test_reference_check_flags_a_changed_membership(tmp_path, monkeypatch):
    MA, base, _ = _stub_runner(monkeypatch, tmp_path)
    assert MA.main(base) == 0
    ref = _write_reference(
        tmp_path, mutate=lambda r: r["modes"][1]["edges"]["mode_edges"].pop())

    assert MA.main(base + ["--reference-manifest", str(ref)]) == 0
    rc = _manifest(tmp_path)["reference_check"]

    assert rc["modes_identical"] is False
    assert rc["per_mode"]["encoder.L0.x_delay:1"]["identical"] is False
    assert rc["per_mode"]["encoder.L0.x_delay:0"]["identical"] is True


def test_reference_without_membership_is_undecided_and_does_not_abort(tmp_path, monkeypatch):
    """The first fold2_w512 k=3 manifest recorded counts, not membership. A
    reference that cannot answer must say so, not pass and not crash the run."""
    MA, base, _ = _stub_runner(monkeypatch, tmp_path)
    assert MA.main(base) == 0
    ref = _write_reference(
        tmp_path, mutate=lambda r: [x.pop("edges") for x in r["modes"]])

    assert MA.main(base + ["--reference-manifest", str(ref)]) == 0
    rc = _manifest(tmp_path)["reference_check"]
    assert rc["modes_identical"] is None

    assert MA.main(base + ["--reference-manifest", str(tmp_path / "nope.json")]) == 0
    rc = _manifest(tmp_path)["reference_check"]
    assert rc["modes_identical"] is None and rc["error"]
