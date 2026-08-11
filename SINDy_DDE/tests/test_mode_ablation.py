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
