"""Self-built Kolmogorov-Arnold Network linear layer (efficient-kan style).

``KANLinear`` replaces a dense ``nn.Linear`` with learnable B-spline activations
on every input->output edge plus a SiLU "base" path:

    output = base_activation(x) @ base_weight^T  +  Bspline(x) @ spline_weight

This is the *efficient-kan* formulation, implemented from scratch (no external
``pykan`` / ``efficient-kan`` runtime dependency) so it is fully testable and the
per-edge spline activations can be sampled for interpretability (XAI).

Used by Phase 3 TKAN cells/encoder and the severity/trajectory heads.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """A KAN linear layer: per-edge B-spline activations + a SiLU base path.

    Parameters
    ----------
    in_features, out_features:
        Edge counts; the layer maps ``[..., in_features] -> [..., out_features]``.
    grid_size:
        Number of intervals in the (uniform) spline knot grid.
    spline_order:
        Degree of the B-spline basis (3 == cubic). The number of basis
        coefficients per edge is ``grid_size + spline_order``.
    grid_range:
        ``(lo, hi)`` span the interior knots cover.
    base_activation:
        Activation used on the residual base path (default :class:`~torch.nn.SiLU`).
    spline_scale:
        Multiplier applied to the spline contribution (and to its init noise),
        letting the base path dominate early in training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        base_activation: type[nn.Module] = nn.SiLU,
        spline_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))
        self.spline_scale = float(spline_scale)
        self.base_activation = base_activation()

        # Number of B-spline basis functions of degree ``spline_order`` over a
        # grid with ``grid_size`` intervals.
        self.n_coeff = self.grid_size + self.spline_order

        # ---- Knot vector ---------------------------------------------------- #
        # ``grid_size`` interior intervals over grid_range, uniformly extended by
        # ``spline_order`` knots on each side -> grid_size + 2*spline_order + 1 knots.
        lo, hi = self.grid_range
        h = (hi - lo) / self.grid_size
        knots = (
            torch.arange(-self.spline_order, self.grid_size + self.spline_order + 1, dtype=torch.float32)
            * h
            + lo
        )
        # Buffer (not a parameter): moves with .to(device) / state_dict but isn't trained.
        self.register_buffer("grid", knots)

        # ---- Parameters ----------------------------------------------------- #
        # Base path: standard linear weight, Kaiming-uniform like nn.Linear.
        self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        # Spline path: one coefficient vector per edge.
        self.spline_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, self.n_coeff)
        )
        self.reset_parameters()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def reset_parameters(self) -> None:
        """Stable init: Kaiming base weight, small-noise spline coefficients."""
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        # Small spline noise so training starts near the base (SiLU) path and is
        # stable; scaled by spline_scale and by 1/sqrt(in) like other KAN inits.
        with torch.no_grad():
            std = self.spline_scale * 0.1 / np.sqrt(self.in_features)
            self.spline_weight.normal_(0.0, std)

    # ------------------------------------------------------------------ #
    # B-spline basis (Cox-de Boor recursion)
    # ------------------------------------------------------------------ #
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the B-spline basis at ``x``.

        Parameters
        ----------
        x : Tensor ``[..., C]`` -- any trailing channel count ``C`` (the knot
            grid is shared across channels, so this also serves the 1-D sweep
            used by :meth:`spline_curves`).

        Returns
        -------
        Tensor ``[..., C, n_coeff]`` -- the value of each of the
        ``grid_size + spline_order`` basis functions at every input.
        """
        grid = self.grid  # [n_knots]
        x = x.unsqueeze(-1)  # [..., in, 1]

        # Degree 0: indicator of the half-open knot span [t_i, t_{i+1}).
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)  # [..., in, n_knots-1]

        # Cox-de Boor recursion: raise degree from 0 to spline_order.
        for k in range(1, self.spline_order + 1):
            left_den = grid[k:-1] - grid[: -(k + 1)]
            right_den = grid[k + 1 :] - grid[1:(-k)]
            left = (x - grid[: -(k + 1)]) / left_den * bases[..., :-1]
            right = (grid[k + 1 :] - x) / right_den * bases[..., 1:]
            bases = left + right

        return bases.contiguous()  # [..., in, n_coeff]

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[..., in_features] -> [..., out_features]``."""
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dim {self.in_features}, got {x.shape[-1]}"
            )
        lead = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)  # [M, in]

        # Base path: base_activation(x) @ base_weight^T  -> [M, out]
        base_out = F.linear(self.base_activation(x_flat), self.base_weight)

        # Spline path: sum over (in, coeff) of basis * spline_weight -> [M, out]
        bases = self.b_splines(x_flat)  # [M, in, n_coeff]
        # einsum over edges and coefficients.
        spline_out = torch.einsum("mic,oic->mo", bases, self.spline_weight)
        spline_out = spline_out * self.spline_scale

        out = base_out + spline_out
        return out.reshape(*lead, self.out_features)

    # ------------------------------------------------------------------ #
    # Interpretability: per-edge spline curves
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def spline_curves(self, n: int = 200) -> dict:
        """Sample each edge's learned spline activation across ``grid_range``.

        Returns
        -------
        dict with
            ``"x"``      : ``np.ndarray[n]`` sweep points over ``grid_range``.
            ``"curves"`` : ``np.ndarray[out_features, in_features, n]`` -- the
                           value of edge (o, i)'s spline activation at each ``x``,
                           i.e. ``spline_scale * sum_c basis_c(x) * W[o,i,c]``.
        """
        lo, hi = self.grid_range
        device = self.grid.device
        xs = torch.linspace(lo, hi, n, device=device, dtype=self.grid.dtype)  # [n]

        # Evaluate the 1-D basis for a single channel sweep: [n, 1, n_coeff].
        bases = self.b_splines(xs.unsqueeze(-1))  # [n, 1, n_coeff]
        bases = bases[:, 0, :]  # [n, n_coeff]

        # curves[o, i, x] = scale * sum_c basis[x, c] * W[o, i, c]
        curves = torch.einsum("xc,oic->oix", bases, self.spline_weight)
        curves = curves * self.spline_scale  # [out, in, n]

        return {
            "x": xs.cpu().numpy(),
            "curves": curves.cpu().numpy(),
        }

    @torch.no_grad()
    def edge_function(self, n: int = 200) -> dict:
        """Sample each edge's FULL learned activation: base (SiLU) path + spline.

        Unlike :meth:`spline_curves` (which returns only the spline residual),
        this includes the residual SiLU base path the forward pass actually uses,
        so each edge ``(o, i)``'s curve is the true learned function

            ``phi_{o,i}(x) = base_weight[o,i] * base_activation(x)
                              + spline_scale * sum_c basis_c(x) * W[o,i,c]``.

        This is the correct object for white-box reading and symbolic fitting:
        for an ``in_features == out_features == 1`` layer it equals ``forward``.

        Returns
        -------
        dict with
            ``"x"``      : ``np.ndarray[n]`` sweep points over ``grid_range``.
            ``"curves"`` : ``np.ndarray[out_features, in_features, n]`` -- the
                           full edge activation ``phi_{o,i}(x)`` at each ``x``.
        """
        sampled = self.spline_curves(n)
        xs = torch.linspace(
            self.grid_range[0], self.grid_range[1], n,
            device=self.grid.device, dtype=self.grid.dtype,
        )
        base_x = self.base_activation(xs)  # [n], e.g. SiLU(x)
        # base contribution of edge (o, i) as a function of its own input:
        # base_weight[o, i] * base_activation(x). [out, in, n]
        base = self.base_weight.unsqueeze(-1) * base_x.view(1, 1, -1)
        curves = torch.as_tensor(sampled["curves"], device=self.grid.device) + base
        return {"x": sampled["x"], "curves": curves.cpu().numpy()}

    # ------------------------------------------------------------------ #
    # Regularization (efficient-kan style)
    # ------------------------------------------------------------------ #
    def regularization_loss(self, l1: float = 1.0, entropy: float = 0.0) -> torch.Tensor:
        """efficient-kan style regularization on the spline weights.

        L1 term: mean absolute spline weight per edge, summed over edges.
        Optional entropy term: penalizes a peaked distribution of per-edge
        importances (encourages a sparse, interpretable activation set).

        Returns a non-negative scalar tensor (differentiable w.r.t. spline_weight).
        """
        # Per-edge importance = mean |coeff| over the coefficient axis. [out, in]
        edge_l1 = self.spline_weight.abs().mean(dim=-1)
        l1_term = edge_l1.sum()

        reg = l1 * l1_term
        if entropy != 0.0:
            total = edge_l1.sum() + 1e-12
            p = edge_l1 / total
            entropy_term = -torch.sum(p * torch.log(p + 1e-12))
            reg = reg + entropy * entropy_term
        return reg

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}"
        )
