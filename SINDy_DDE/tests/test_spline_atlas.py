"""Spline-edge atlas: named, data-grounded, symbolically-annotated edge panels."""
import numpy as np
import torch

from pri2m.models.tkan_model import TKANModel
from pri2m.xai import activations as act_mod
from pri2m.xai import semantics, spline_atlas, symbolic


def _model_and_batch(B=6, L=32):
    torch.manual_seed(0)
    model = TKANModel(
        input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3,
        side_input=True, side_features=("omega", "tpf", "tau"), side_hidden=4,
    ).eval()
    g = torch.Generator().manual_seed(1)
    batch = {
        "window": torch.randn(B, L, 4, generator=g),
        "tau": torch.full((B,), 3, dtype=torch.long),
        "omega": torch.full((B,), 5000.0),
        "tpf": torch.full((B,), 200.0),
        "b_over_blim": torch.rand(B, generator=g),
        "label": torch.tensor([0.0, 1.0] * (B // 2)),
    }
    stats = act_mod.collect_activations(model, batch)
    sem = semantics.edge_index_map(model)
    return model, batch, stats, sem


def test_eval_fit_linear_roundtrip():
    xs = np.linspace(-1, 1, 50)
    fit = symbolic.fit_curve(xs, 2.0 * xs + 0.5)
    ys = symbolic.eval_fit(fit, xs)
    assert np.allclose(ys, 2.0 * xs + 0.5, atol=1e-6)


def test_eval_fit_raises_on_unknown_name():
    import pytest

    with pytest.raises(ValueError):
        symbolic.eval_fit({"name": None, "params": []}, np.linspace(-1, 1, 10))
    with pytest.raises(ValueError):
        symbolic.eval_fit({"name": "not_a_family", "params": []}, np.linspace(-1, 1, 10))


def test_top_edge_entries_named_and_flagged():
    model, batch, stats, sem = _model_and_batch()
    entries = spline_atlas.top_edge_entries(stats, sem, k=12)
    assert len(entries) == 12
    assert entries == sorted(entries, key=lambda e: -e["importance"])
    for e in entries:
        named = {d["index"]: d for d in sem[e["kan"]]}
        assert e["feature"] == named[e["in"]]["name"]
        assert e["is_delay"] == (named[e["in"]]["block"] == "x_delay")
        assert e["is_side"] == (named[e["in"]]["block"] == "side")


def test_plot_spline_atlas_writes_and_annotates(tmp_path):
    model, batch, stats, sem = _model_and_batch()
    rows = spline_atlas.plot_spline_atlas(model, stats, sem, tmp_path / "atlas.png", k=6)
    assert (tmp_path / "atlas.png").is_file()
    assert len(rows) == 6
    for r in rows:
        assert {"kan", "out", "in", "feature", "block", "importance", "formula", "r2", "accepted"} <= set(r)


def test_feature_bars_collapse_hidden(tmp_path):
    model, batch, stats, sem = _model_and_batch()
    agg = spline_atlas.feature_importance_bars(stats, sem)
    gate = agg["encoder.layers.0.kan_z"]
    assert "h_prev (sum)" in gate
    assert any(k.startswith("delay:") for k in gate)
    assert not any(k.startswith("h0") for k in gate)          # individual h's collapsed
    p = spline_atlas.plot_feature_importance_bars(agg, tmp_path / "bars.png")
    assert (tmp_path / "bars.png").is_file()


def test_feature_named_heatmap_writes(tmp_path):
    model, batch, stats, sem = _model_and_batch()
    p = spline_atlas.plot_feature_named_heatmap(stats, sem, tmp_path / "heat.png")
    assert (tmp_path / "heat.png").is_file()
