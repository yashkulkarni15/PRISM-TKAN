"""Tests for ``pri2m.xai.substitution`` -- putting fitted closed forms back into
a trained KAN and measuring what the network then does.

The white-box pipeline fits a closed form to each of the 25 most important edges
and reports R^2 per edge. That is a statement about curve fitting: it does not
say the network computes those equations, because per-edge residuals may cancel
or amplify through the recurrent trunk. These tests cover the machinery that
makes the difference measurable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from pri2m.models.kan import KANLinear
from pri2m.xai import symbolic
from pri2m.xai.substitution import (
    edge_contribution,
    eval_fit_torch,
    permute_fits,
    substitute_context,
    substitutions_are_active,
)

# --------------------------------------------------------------------------- #
# Task 1 -- torch evaluation of every fitted family
# --------------------------------------------------------------------------- #

_FAMILY_CASES = [
    ("const", lambda x: np.full_like(x, 2.5)),
    ("linear", lambda x: 1.7 * x - 0.4),
    ("quadratic", lambda x: -1.3 * x**2 - 7.4 * x + 0.38),
    ("cubic", lambda x: 0.5 * x**3 - x + 0.2),
    ("saturating", lambda x: 2.0 * np.tanh(1.5 * x + 0.3) + 0.1),
    ("oscillatory", lambda x: 3.0 * np.sin(3.4 * x + 0.86) + 2.9),
    ("abs", lambda x: 1.2 * np.abs(x) - 0.3),
    ("relu", lambda x: 0.9 * np.maximum(x, 0.0) + 0.15),
]


@pytest.mark.parametrize("name,fn", _FAMILY_CASES)
def test_eval_fit_torch_agrees_with_numpy_on_every_family(name, fn):
    """The torch path must not drift from the numpy path it mirrors.

    A transposed parameter would still produce a smooth curve and a plausible
    AUC, so agreement is asserted family by family rather than spot-checked.
    """
    xs = np.linspace(-2.0, 2.0, 257)
    fit = symbolic.fit_curve(xs, fn(xs), r2_threshold=None)
    expected = symbolic.eval_fit(fit, xs)
    got = eval_fit_torch(fit, torch.from_numpy(xs).double())
    assert got.dtype == torch.float64
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-10, atol=1e-10)


def test_eval_fit_torch_rejects_an_unaccepted_fit():
    with pytest.raises(ValueError, match="no accepted family"):
        eval_fit_torch({"name": None, "params": []}, torch.zeros(4))


def test_eval_fit_torch_rejects_an_unknown_family():
    with pytest.raises(ValueError, match="unknown family"):
        eval_fit_torch({"name": "not_a_family", "params": [1.0]}, torch.zeros(4))


# --------------------------------------------------------------------------- #
# Task 2 -- exact per-edge contribution
# --------------------------------------------------------------------------- #


def _kan(seed: int = 0, in_features: int = 5, out_features: int = 3) -> KANLinear:
    torch.manual_seed(seed)
    k = KANLinear(in_features, out_features).double()
    with torch.no_grad():
        k.base_weight.normal_()
        k.spline_weight.normal_()
    return k


def test_edge_contributions_sum_to_the_forward_pass():
    """Summing every edge must reconstruct ``forward`` to float64 round-off.

    NOT bit-identity, and the difference is not a defect. ``forward`` reduces
    over all edges at once through a fused ``F.linear`` and an ``einsum``, while
    this sums edge by edge; the two summation orders round differently. The
    observed gap is a few ULP (~9e-16 at output magnitude ~3).

    The tolerance is 1e-12: roughly a thousand times the observed round-off, and
    still ten orders of magnitude below any error that would signal a real
    indexing or scaling mistake -- a wrong ``spline_scale`` or a transposed
    index moves this by order 1, not by 1e-15. Do not "tighten" it to 0.0.
    """
    k = _kan()
    x = torch.linspace(-1.0, 1.0, 40).double().reshape(8, 5).contiguous()
    expected = k(x)
    got = torch.zeros_like(expected)
    for o in range(k.out_features):
        for i in range(k.in_features):
            got[..., o] = got[..., o] + edge_contribution(k, o, i, x)
    assert torch.max(torch.abs(got - expected)).item() < 1e-12


def test_edge_contribution_detects_a_wrong_scale():
    """The reconstruction tolerance is tight enough to catch a real mistake.

    Guards the test above: if 1e-12 were too loose to matter, dropping
    ``spline_scale`` would still pass. It must not.
    """
    k = _kan()
    x = torch.linspace(-1.0, 1.0, 40).double().reshape(8, 5).contiguous()
    expected = k(x)
    got = torch.zeros_like(expected)
    for o in range(k.out_features):
        for i in range(k.in_features):
            # deliberately wrong: base path only, spline path dropped
            got[..., o] = got[..., o] + k.base_weight[o, i] * k.base_activation(x[:, i])
    assert torch.max(torch.abs(got - expected)).item() > 1e-12


def test_edge_contribution_returns_the_leading_shape():
    k = _kan()
    x = torch.randn(4, 7, 5).double()
    assert edge_contribution(k, 1, 2, x).shape == (4, 7)


def test_edge_contribution_uses_the_named_input_column_only():
    k = _kan()
    x = torch.randn(6, 5).double()
    before = edge_contribution(k, 0, 2, x).clone()
    x2 = x.clone()
    x2[:, 3] += 5.0  # a different input column
    assert torch.equal(edge_contribution(k, 0, 2, x2), before)


# --------------------------------------------------------------------------- #
# Task 3 -- the substitution context manager
# --------------------------------------------------------------------------- #


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.kan = _kan(seed=3)

    def forward(self, x):
        return self.kan(x)


def _identity_subs(model, o: int = 0, i: int = 1, lo: float = -1.0, hi: float = 1.0):
    """A substitution whose closed form is fitted to the edge's own curve."""
    xs = np.linspace(lo, hi, 400)
    grid = torch.from_numpy(np.tile(xs[:, None], (1, model.kan.in_features))).double()
    ys = edge_contribution(model.kan, o, i, grid).detach().numpy()
    fit = symbolic.fit_curve(xs, ys, r2_threshold=None)
    return {"kan": [{"out": o, "in": i, "fit": fit, "lo": lo, "hi": hi}]}, fit


def test_empty_substitution_leaves_the_forward_pass_bit_identical():
    m = _Tiny()
    x = torch.randn(16, 5).double()
    before = m(x)
    with substitute_context(m, {}):
        during = m(x)
    assert torch.max(torch.abs(during - before)).item() == 0.0


def test_substitution_moves_the_output_by_exactly_the_fit_residual():
    """The correction must be ``phi_sym - phi_orig`` and nothing else.

    Stated as an identity rather than as a magnitude, so it does not depend on
    the fixture edge happening to fit well. A randomly initialised spline is not
    necessarily close to any of the eight families -- this one fits at about
    R^2 = 0.94 -- and a test that assumed otherwise would be measuring the
    fixture rather than the code.
    """
    m = _Tiny()
    o, i = 0, 1
    subs, fit = _identity_subs(m, o=o, i=i)
    x = (torch.rand(64, 5).double() * 2.0) - 1.0

    before = m(x)
    with substitute_context(m, subs):
        during = m(x)
    delta = during - before

    expected = eval_fit_torch(fit, x[:, i]) - edge_contribution(m.kan, o, i, x)
    assert torch.max(torch.abs(delta[:, o] - expected)).item() < 1e-12

    others = [c for c in range(m.kan.out_features) if c != o]
    assert torch.max(torch.abs(delta[:, others])).item() == 0.0


def test_hooks_are_removed_after_the_context_exits():
    m = _Tiny()
    subs, _ = _identity_subs(m)
    x = torch.randn(16, 5).double()
    before = m(x)
    with substitute_context(m, subs):
        pass
    assert torch.max(torch.abs(m(x) - before)).item() == 0.0
    assert len(m.kan._forward_hooks) == 0


def test_hooks_are_removed_when_the_body_raises():
    m = _Tiny()
    subs, _ = _identity_subs(m)
    x = torch.randn(16, 5).double()
    before = m(x)
    with pytest.raises(RuntimeError, match="boom"):
        with substitute_context(m, subs):
            raise RuntimeError("boom")
    assert torch.max(torch.abs(m(x) - before)).item() == 0.0
    assert len(m.kan._forward_hooks) == 0


def test_unknown_module_name_raises_before_any_hook_is_installed():
    m = _Tiny()
    bad = {
        "nope": [
            {
                "out": 0,
                "in": 0,
                "fit": {"name": "const", "params": [0.0]},
                "lo": -1.0,
                "hi": 1.0,
            }
        ]
    }
    with pytest.raises(KeyError, match="no module named"):
        with substitute_context(m, bad):
            pass
    assert len(m.kan._forward_hooks) == 0


# --------------------------------------------------------------------------- #
# Task 4 -- the guard and the derangement control
# --------------------------------------------------------------------------- #


def test_guard_is_true_when_the_substitution_changes_the_output():
    m = _Tiny()
    subs = {
        "kan": [
            {
                "out": 0,
                "in": 1,
                "fit": {"name": "const", "params": [99.0]},
                "lo": -1.0,
                "hi": 1.0,
            }
        ]
    }
    assert substitutions_are_active(m, subs, torch.randn(16, 5).double()) is True


def test_guard_is_false_when_the_substitution_is_a_no_op():
    m = _Tiny()
    assert substitutions_are_active(m, {}, torch.randn(16, 5).double()) is False


def _const_entries(n: int = 6) -> list[dict]:
    return [
        {
            "out": 0,
            "in": j,
            "fit": {"name": "const", "params": [float(j)]},
            "lo": -1.0,
            "hi": 1.0,
        }
        for j in range(n)
    ]


def test_permute_fits_is_a_derangement_for_many_seeds():
    """No equation may land back on its own edge.

    Update 12's tau-shuffle handed 2 of 24 windows their own value back; that
    biased toward the control arm, which was the safe direction, but by luck
    rather than by design.
    """
    entries = _const_entries()
    for seed in range(50):
        out = permute_fits(entries, seed=seed)
        assert len(out) == len(entries)
        for original, moved in zip(entries, out):
            assert moved["out"] == original["out"] and moved["in"] == original["in"]
            assert moved["fit"]["params"] != original["fit"]["params"]


def test_permute_fits_preserves_the_multiset_of_equations():
    """The control is matched on family and complexity by construction."""
    got = sorted(e["fit"]["params"][0] for e in permute_fits(_const_entries(), seed=1))
    assert got == [float(j) for j in range(6)]


def test_permute_fits_refuses_a_single_entry():
    with pytest.raises(ValueError, match="derangement"):
        permute_fits(_const_entries(1), seed=0)


# --------------------------------------------------------------------------- #
# Task 5 -- extrapolation clamp
# --------------------------------------------------------------------------- #


def _steep_line_subs() -> dict:
    return {
        "kan": [
            {
                "out": 0,
                "in": 1,
                "fit": {"name": "linear", "params": [100.0, 0.0]},
                "lo": -1.0,
                "hi": 1.0,
            }
        ]
    }


def test_clamping_evaluates_the_closed_form_at_the_boundary():
    """Fits are only defined over the range their input visited.

    A spline saturates outside its grid; a fitted polynomial diverges. Held-out
    windows do leave that range, so the two arms must differ there.
    """
    m = _Tiny()
    x = torch.zeros(4, 5).double()
    x[:, 1] = 5.0  # well outside [lo, hi]

    with substitute_context(m, _steep_line_subs(), clamp=True):
        clamped = m(x)
    with substitute_context(m, _steep_line_subs(), clamp=False):
        unclamped = m(x)

    # clamped evaluates at hi=1.0 -> 100.0; unclamped at 5.0 -> 500.0
    assert torch.max(torch.abs(unclamped - clamped)).item() > 300.0


def test_clamping_changes_nothing_when_every_input_is_in_range():
    m = _Tiny()
    x = torch.zeros(4, 5).double()
    x[:, 1] = torch.linspace(-0.9, 0.9, 4).double()
    with substitute_context(m, _steep_line_subs(), clamp=True):
        a = m(x)
    with substitute_context(m, _steep_line_subs(), clamp=False):
        b = m(x)
    assert torch.max(torch.abs(a - b)).item() == 0.0


# --------------------------------------------------------------------------- #
# The guard against a REAL Phase 3 head's output shape (regression, 2026-08-12)
# --------------------------------------------------------------------------- #

# An earlier guard read out["logits"], a key no Phase 3 head produces. The unit
# fixture returned a bare tensor and never took the dict branch, so the bug
# survived until it met a real model on the cluster. These fixtures use the real
# key names from models/heads.py:140-147.


class _DictHead(nn.Module):
    """Returns the point-mode dict shape: b_over_blim + chatter_logit."""

    def __init__(self):
        super().__init__()
        self.kan = _kan(seed=5)

    def forward(self, x):
        out = self.kan(x)
        return {"b_over_blim": out[..., 0], "chatter_logit": out[..., 1]}


class _QuantileHead(_DictHead):
    """Quantile mode adds a tensor key AND a non-tensor one."""

    def forward(self, x):
        out = self.kan(x)
        return {
            "b_over_blim": out[..., 0],
            "b_over_blim_quantiles": out,
            "quantile_levels": (0.1, 0.5, 0.9),  # not a tensor
            "chatter_logit": out[..., 1],
        }


@pytest.mark.parametrize("cls", [_DictHead, _QuantileHead])
def test_guard_handles_a_dict_returning_head(cls):
    m = cls()
    subs = {
        "kan": [
            {"out": 0, "in": 1, "fit": {"name": "const", "params": [99.0]},
             "lo": -1.0, "hi": 1.0}
        ]
    }
    x = torch.randn(16, 5).double()
    assert substitutions_are_active(m, subs, x) is True
    assert substitutions_are_active(m, {}, x) is False


def test_guard_sees_a_change_in_any_output_not_just_the_first():
    """Substituting an edge feeding only chatter_logit must still register."""
    m = _DictHead()
    subs = {
        "kan": [
            {"out": 1, "in": 0, "fit": {"name": "const", "params": [50.0]},
             "lo": -1.0, "hi": 1.0}
        ]
    }
    assert substitutions_are_active(m, subs, torch.randn(16, 5).double()) is True


def test_guard_refuses_an_output_with_no_comparable_tensor():
    class _NoTensors(nn.Module):
        def forward(self, x):
            return {"quantile_levels": (0.1, 0.5)}

    with pytest.raises(ValueError, match="no comparable float tensor"):
        substitutions_are_active(_NoTensors(), {}, torch.randn(4, 5).double())
