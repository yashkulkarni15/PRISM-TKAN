"""Tests for KAN-native explainability (Task 4A).

KAN layers expose their per-edge B-spline activations, so a KAN model is
interpretable without a post-hoc attribution method: we can read each edge's
learned activation curve directly, prune low-importance edges, and write a
human-readable symbolic summary. These tests pin that behaviour on a tiny
``TKANModel`` so they stay fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pri2m.models.kan import KANLinear
from pri2m.models.tkan_model import TKANModel
from pri2m.xai import kan_native


def _tiny_model(seed: int = 0) -> TKANModel:
    torch.manual_seed(seed)
    return TKANModel(input_size=4, hidden_size=8)


# --------------------------------------------------------------------------- #
# extract_spline_curves
# --------------------------------------------------------------------------- #
def test_extract_spline_curves_non_empty_finite():
    model = _tiny_model()
    curves = kan_native.extract_spline_curves(model)

    assert isinstance(curves, dict)
    assert len(curves) > 0
    for name, entry in curves.items():
        assert isinstance(name, str)
        assert set(entry) >= {"x", "curves"}
        x = np.asarray(entry["x"])
        c = np.asarray(entry["curves"])
        assert x.ndim == 1
        assert c.ndim == 3  # [out, in, n]
        assert c.shape[-1] == x.shape[0]
        assert np.isfinite(x).all()
        assert np.isfinite(c).all()


def test_extract_spline_curves_one_entry_per_kanlinear():
    model = _tiny_model()
    n_kan = sum(isinstance(m, KANLinear) for m in model.modules())
    curves = kan_native.extract_spline_curves(model)
    assert len(curves) == n_kan
    # shapes match the owning KANLinear's (out, in)
    by_name = dict(model.named_modules())
    for name, entry in curves.items():
        mod = by_name[name]
        assert isinstance(mod, KANLinear)
        assert np.asarray(entry["curves"]).shape[:2] == (mod.out_features, mod.in_features)


# --------------------------------------------------------------------------- #
# prune_edges
# --------------------------------------------------------------------------- #
def test_prune_edges_counts_sum_to_edge_count():
    model = _tiny_model()
    report = kan_native.prune_edges(model, threshold=0.0)
    assert len(report) > 0
    by_name = dict(model.named_modules())
    for name, rec in report.items():
        mod = by_name[name]
        n_edges = mod.out_features * mod.in_features
        assert rec["kept"] + rec["pruned"] == n_edges
        edge_l1 = np.asarray(rec["edge_l1"])
        assert edge_l1.shape == (mod.out_features, mod.in_features)
        assert np.isfinite(edge_l1).all()


def test_prune_edges_threshold_zero_keeps_all():
    model = _tiny_model()
    report = kan_native.prune_edges(model, threshold=0.0)
    for rec in report.values():
        # threshold 0.0 prunes nothing: every edge L1 >= 0.
        assert rec["pruned"] == 0


def test_prune_edges_higher_threshold_prunes_more():
    model = _tiny_model()
    low = kan_native.prune_edges(model, threshold=0.0)
    high = kan_native.prune_edges(model, threshold=1e9)
    total_low = sum(r["pruned"] for r in low.values())
    total_high = sum(r["pruned"] for r in high.values())
    assert total_high > total_low
    # An astronomically high threshold prunes everything.
    for name, rec in high.items():
        mod = dict(model.named_modules())[name]
        assert rec["pruned"] == mod.out_features * mod.in_features


def test_prune_edges_non_mutating():
    model = _tiny_model()
    before = {n: m.spline_weight.detach().clone() for n, m in model.named_modules() if isinstance(m, KANLinear)}
    kan_native.prune_edges(model, threshold=1e9)
    for n, m in model.named_modules():
        if isinstance(m, KANLinear):
            assert torch.equal(m.spline_weight, before[n]), f"{n} spline_weight mutated"


# --------------------------------------------------------------------------- #
# symbolic_summary
# --------------------------------------------------------------------------- #
def test_symbolic_summary_returns_string_for_head():
    from pri2m.models.heads import SeverityHead

    torch.manual_seed(0)
    head = SeverityHead(8)
    text = kan_native.symbolic_summary(head)
    assert isinstance(text, str)
    assert len(text) > 0


def test_symbolic_summary_works_on_full_model():
    model = _tiny_model()
    text = kan_native.symbolic_summary(model)
    assert isinstance(text, str)
    # Mentions at least one KANLinear module name from the model.
    assert "severity" in text or "reg_head" in text or "encoder" in text


# --------------------------------------------------------------------------- #
# plot_edge_activations
# --------------------------------------------------------------------------- #
def test_plot_edge_activations_writes_png(tmp_path):
    model = _tiny_model()
    curves = kan_native.extract_spline_curves(model)
    out_path = tmp_path / "edges.png"
    returned = kan_native.plot_edge_activations(curves, out_path)
    assert str(returned) == str(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
