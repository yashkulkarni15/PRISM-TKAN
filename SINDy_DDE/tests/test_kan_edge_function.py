"""KANLinear.edge_function: the FULL per-edge curve (base SiLU path + spline).

The white-box reading of a KAN edge must include the SiLU base path, not just the
spline residual that ``spline_curves`` returns. ``edge_function`` adds
``base_weight[o,i] * SiLU(x)`` onto the spline curve so each edge's plotted /
symbolified activation is the actual learned function the forward pass uses.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pri2m.models.kan import KANLinear


def _silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def test_edge_function_equals_base_plus_spline():
    """edge_function == base_weight*SiLU(x) + spline_curves, exactly."""
    torch.manual_seed(0)
    kan = KANLinear(2, 3, grid_size=4, spline_order=3)
    with torch.no_grad():
        kan.base_weight.copy_(torch.tensor([[0.5, -1.0], [2.0, 0.0], [-0.3, 0.7]]))
        kan.spline_weight.normal_(0.0, 0.4)  # make the spline part clearly nonzero

    ef = kan.edge_function(n=50)
    sc = kan.spline_curves(n=50)

    x = ef["x"]
    assert x.shape == (50,)
    assert ef["curves"].shape == (3, 2, 50)
    assert np.isfinite(ef["curves"]).all()
    # Same sweep grid as spline_curves.
    np.testing.assert_allclose(x, sc["x"])

    bw = kan.base_weight.detach().cpu().numpy()  # [out, in]
    base = bw[:, :, None] * _silu(x)[None, None, :]  # [out, in, n]
    expected = sc["curves"] + base
    np.testing.assert_allclose(ef["curves"], expected, rtol=1e-5, atol=1e-6)


def test_edge_function_differs_from_spline_when_base_nonzero():
    """With a nonzero base path, the full edge function != the spline residual."""
    torch.manual_seed(1)
    kan = KANLinear(2, 2, grid_size=5, spline_order=3)
    with torch.no_grad():
        kan.base_weight.fill_(1.0)  # definitely nonzero base path
    ef = kan.edge_function(n=40)
    sc = kan.spline_curves(n=40)
    assert not np.allclose(ef["curves"], sc["curves"])


def test_edge_function_matches_forward_single_input():
    """A single-edge layer's edge_function reproduces the layer's forward output.

    For in_features=1, out_features=1 the forward map y(x) is exactly the edge
    function phi(x); evaluating the layer on the sweep grid must equal the curve.
    """
    torch.manual_seed(2)
    kan = KANLinear(1, 1, grid_size=5, spline_order=3)
    with torch.no_grad():
        kan.spline_weight.normal_(0.0, 0.3)
    ef = kan.edge_function(n=30)
    x = torch.tensor(ef["x"], dtype=torch.float32).reshape(-1, 1)  # [30, 1]
    with torch.no_grad():
        y = kan(x).reshape(-1).cpu().numpy()  # forward on the grid
    np.testing.assert_allclose(ef["curves"][0, 0], y, rtol=1e-5, atol=1e-6)
