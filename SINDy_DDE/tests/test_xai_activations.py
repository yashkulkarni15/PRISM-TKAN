"""xai.activations.collect_activations: data-aware per-edge statistics.

Runs real windows through the model with forward hooks, capturing each
``KANLinear``'s actual inputs and turning them into a per-edge **data-aware
importance** (std over data of the edge's full output contribution) and the input
**range the data actually visits**. This replaces the data-blind mean-|weight|
view (which pruned 0/14k edges on the real model).
"""

from __future__ import annotations

import numpy as np
import torch

from pri2m.models.tkan_model import TKANModel
from pri2m.xai.activations import collect_activations


def _batch(B=6, L=24, C=4, tau=5):
    torch.manual_seed(0)
    return {
        "window": torch.randn(B, L, C),
        "tau": torch.full((B,), tau, dtype=torch.long),
    }


def test_collect_shapes_and_finite():
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    model.eval()
    stats = collect_activations(model, _batch())

    assert "encoder.layers.0.kan_z" in stats
    z = stats["encoder.layers.0.kan_z"]
    # kan_z is KANLinear(in=4+8+4=16 -> out=8)
    assert z["importance"].shape == (8, 16)
    assert z["visited_lo"].shape == (16,)
    assert z["visited_hi"].shape == (16,)
    assert z["inputs"].shape[1] == 16
    assert np.isfinite(z["importance"]).all()
    assert np.isfinite(z["inputs"]).all()


def test_zeroed_edge_has_least_importance():
    """An edge with zero base+spline weights contributes nothing -> importance ~0,
    strictly below a deliberately amplified edge."""
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    kan = model.encoder.layers[0].kan_z
    with torch.no_grad():
        # Dead edge (o=0, i=0): no base, no spline -> phi == 0 for all x.
        kan.base_weight[0, 0] = 0.0
        kan.spline_weight[0, 0, :] = 0.0
        # Loud edge (o=1, i=1): large base path -> phi varies with the input.
        kan.base_weight[1, 1] = 5.0
        kan.spline_weight[1, 1, :] = 0.0
    model.eval()
    stats = collect_activations(model, _batch())
    imp = stats["encoder.layers.0.kan_z"]["importance"]

    assert imp[0, 0] == 0.0  # dead edge contributes nothing
    assert imp[1, 1] > imp[0, 0]  # loud edge dominates the dead one
    assert imp[0, 0] <= imp.max()


def test_visited_range_is_within_tanh_band_and_ordered():
    """Encoder gate inputs are tanh-squashed, so the visited range sits in
    [-1, 1] and lo <= hi for every edge input."""
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    model.eval()
    stats = collect_activations(model, _batch())
    z = stats["encoder.layers.0.kan_z"]
    assert (z["visited_lo"] <= z["visited_hi"]).all()
    assert (z["visited_lo"] >= -1.0001).all()
    assert (z["visited_hi"] <= 1.0001).all()


def test_max_points_caps_stored_inputs():
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    model.eval()
    stats = collect_activations(model, _batch(B=8, L=40), max_points=50)
    # B*L = 320 raw input rows; capped to <= 50 for the stored sample.
    assert stats["encoder.layers.0.kan_z"]["inputs"].shape[0] <= 50
