"""xai.symbolic: read each learned B-spline edge as a human-readable closed form.

A small candidate library (const/linear/quadratic/cubic/saturating/oscillatory/
abs/relu) is fit to 1-D samples of an edge's full activation ``phi_{o,i}`` and the
best is selected by BIC. The end-to-end table ranks edges by data-aware importance
and attaches the semantic name/block of each edge.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pri2m.models.tkan_model import TKANModel
from pri2m.xai.activations import collect_activations
from pri2m.xai.symbolic import (
    fit_curve,
    fit_edge,
    plot_top_edges,
    symbolic_table,
    write_symbolic_csv,
)


def _batch(B=6, L=24, C=4, tau=5):
    torch.manual_seed(0)
    return {
        "window": torch.randn(B, L, C),
        "tau": torch.full((B,), tau, dtype=torch.long),
    }


# ---------------------------------------------------------------------------- #
# fit_curve: ground-truth recovery
# ---------------------------------------------------------------------------- #
def test_fit_curve_linear():
    xs = np.linspace(-1, 1, 100)
    res = fit_curve(xs, 2 * xs + 1)
    assert res["name"] == "linear"
    assert res["r2"] > 0.999
    assert np.allclose(res["params"], [2.0, 1.0], atol=1e-2)


def test_fit_curve_saturating_not_linear():
    xs = np.linspace(-1, 1, 100)
    res = fit_curve(xs, np.tanh(3 * xs))
    assert res["name"] == "saturating"
    assert res["r2"] > 0.99


def test_fit_curve_quadratic():
    xs = np.linspace(-1, 1, 100)
    res = fit_curve(xs, xs**2)
    assert res["name"] == "quadratic"
    assert res["r2"] > 0.999


def test_fit_curve_returns_formula_string():
    xs = np.linspace(-1, 1, 100)
    res = fit_curve(xs, 2 * xs + 1)
    assert isinstance(res["formula"], str) and res["formula"]
    assert math.isfinite(res["bic"])


# ---------------------------------------------------------------------------- #
# fit_curve: R2 + simplicity acceptance gate (pykan-style faithfulness gate)
# ---------------------------------------------------------------------------- #
def test_fit_curve_default_path_unchanged_no_gate_keys():
    """Without r2_threshold the return is the historic BIC dict (no gate keys)."""
    xs = np.linspace(-1, 1, 120)
    res = fit_curve(xs, 2 * xs + 1)
    assert res["name"] == "linear"
    assert "accepted" not in res and "best_family" not in res


def test_fit_curve_gate_accepts_well_fit_and_prefers_simpler_family():
    xs = np.linspace(-1, 1, 120)
    res = fit_curve(xs, 2 * xs + 1, r2_threshold=0.9)
    # A clean line clears the gate and is reported as linear, not a marginally
    # better-fitting cubic.
    assert res["accepted"] is True
    assert res["name"] == "linear"
    assert res["r2"] > 0.999
    assert isinstance(res["formula"], str) and res["formula"]


def test_fit_curve_gate_rejects_unfittable_edge():
    """Pure noise has no faithful closed form -> flagged, not snapped to a formula."""
    xs = np.linspace(-1, 1, 120)
    rng = np.random.default_rng(0)
    res = fit_curve(xs, rng.standard_normal(120), r2_threshold=0.9)
    assert res["accepted"] is False
    assert res["name"] is None and res["formula"] is None
    # The closest family is still reported for diagnostics.
    assert isinstance(res["best_family"], str) and res["best_family"]
    assert res["r2"] < 0.9


def test_fit_curve_gate_still_recovers_quadratic():
    xs = np.linspace(-1, 1, 120)
    res = fit_curve(xs, xs**2, r2_threshold=0.9)
    assert res["accepted"] is True and res["name"] == "quadratic"


# ---------------------------------------------------------------------------- #
# fit_edge: evaluate phi over a sweep, then symbolic-fit
# ---------------------------------------------------------------------------- #
def test_fit_edge_keys_and_finite():
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    kan = model.encoder.layers[0].kan_z
    res = fit_edge(kan, o=0, i=0, lo=-0.8, hi=0.8, n=120)
    for key in ("name", "formula", "r2", "bic", "params", "out", "in", "lo", "hi"):
        assert key in res
    assert res["out"] == 0 and res["in"] == 0
    assert res["lo"] == -0.8 and res["hi"] == 0.8
    assert math.isfinite(res["r2"]) and res["r2"] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------- #
# symbolic_table: end-to-end ranking + semantics
# ---------------------------------------------------------------------------- #
def test_symbolic_table_end_to_end():
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    model.eval()
    activations = collect_activations(model, _batch())
    rows = symbolic_table(model, activations, top_k=5)

    assert len(rows) == 5
    for row in rows:
        for key in (
            "kan", "out", "in", "block", "name",
            "importance", "formula", "r2", "bic",
        ):
            assert key in row
        assert isinstance(row["name"], str) and row["name"]
        assert isinstance(row["block"], str) and row["block"]
        assert isinstance(row["formula"], str) and row["formula"]
        assert math.isfinite(row["r2"]) and row["r2"] <= 1.0 + 1e-6
        assert math.isfinite(row["importance"])

    # Ranked by importance descending.
    imps = [r["importance"] for r in rows]
    assert imps == sorted(imps, reverse=True)


def test_symbolic_table_ranks_loud_edge_first():
    """An amplified edge should outrank the rest and surface in the table."""
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    kan = model.encoder.layers[0].kan_z
    with torch.no_grad():
        kan.base_weight[3, 2] = 20.0  # very loud edge
        kan.spline_weight[3, 2, :] = 0.0
    model.eval()
    activations = collect_activations(model, _batch())
    rows = symbolic_table(model, activations, top_k=3)
    top = rows[0]
    assert top["kan"] == "encoder.layers.0.kan_z"
    assert top["out"] == 3 and top["in"] == 2


def test_symbolic_table_gate_flags_unfaithful_edges():
    """With an unreachable threshold every edge is flagged, never mis-labelled."""
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    model.eval()
    activations = collect_activations(model, _batch())
    rows = symbolic_table(model, activations, top_k=4, r2_threshold=1.01)
    assert len(rows) == 4
    for row in rows:
        assert row["accepted"] is False
        assert row["formula"] == "no faithful closed form"
        # formula stays a non-empty string (never None) so downstream never breaks.
        assert isinstance(row["formula"], str) and row["formula"]


def test_symbolic_table_gate_keeps_well_fit_edges():
    """Under the default gate, well-fitting edges still carry a real formula."""
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    kan = model.encoder.layers[0].kan_z
    with torch.no_grad():
        kan.base_weight[3, 2] = 20.0  # a loud, smooth (well-fittable) edge
        kan.spline_weight[3, 2, :] = 0.0
    model.eval()
    activations = collect_activations(model, _batch())
    rows = symbolic_table(model, activations, top_k=3)  # default r2_threshold=0.9
    top = rows[0]
    assert top["out"] == 3 and top["in"] == 2
    assert top["accepted"] is True
    assert top["formula"] != "no faithful closed form"
    assert all("accepted" in r for r in rows)


# ---------------------------------------------------------------------------- #
# CSV round-trip
# ---------------------------------------------------------------------------- #
def test_write_symbolic_csv_roundtrips(tmp_path):
    import pandas as pd

    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    model.eval()
    activations = collect_activations(model, _batch())
    rows = symbolic_table(model, activations, top_k=4)

    path = tmp_path / "symbolic.csv"
    out = write_symbolic_csv(rows, str(path))
    assert out == str(path)
    df = pd.read_csv(path)
    assert len(df) == len(rows)


# ---------------------------------------------------------------------------- #
# Plot
# ---------------------------------------------------------------------------- #
def test_plot_top_edges_writes_file(tmp_path):
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    model.eval()
    activations = collect_activations(model, _batch())
    rows = symbolic_table(model, activations, top_k=8)

    path = tmp_path / "top_edges.png"
    out = plot_top_edges(model, activations, rows, str(path), k=8)
    assert out == str(path)
    import os

    assert os.path.exists(path) and os.path.getsize(path) > 0
