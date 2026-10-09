"""Substitute fitted closed forms into a trained KAN at forward time.

The white-box pipeline fits a closed form to each of the 25 most important edges
(``xai/symbolic.py``), and every fold returns 25/25 accepted with R^2 between
0.9257 and 0.9999. Until this module those equations were only ever drawn on a
figure: ``symbolic.eval_fit``'s single non-test call site is
``xai/spline_atlas.py:106``.

A high per-edge R^2 is a statement about curve fitting. It does not say the
network computes those equations, because a relative residual of roughly
``sqrt(1 - R^2)`` per edge -- about 1.4% at R^2 = 0.9998, about 8.6% at
R^2 = 0.9257 -- may cancel or may amplify through the recurrent trunk over
hundreds of timesteps. This module puts the fits back into the model so the
network can be re-scored and the question answered.

Mechanism. ``KANLinear.forward`` (``models/kan.py:137-156``) computes

    out[m, o] = sum_i ( base_weight[o,i] * base_activation(x[m,i])
                        + spline_scale * sum_c b_splines(x)[m,i,c]
                                         * spline_weight[o,i,c] )

so each edge contributes additively and separably in its own input. A
substitution is therefore applied as a correction, ``phi_sym - phi_orig``, added
to output column ``o`` by a forward hook -- no change to ``KANLinear`` and no
weights overwritten.

The rejected alternative is worth recording. Projecting each closed form back
onto the spline basis by least squares and overwriting ``spline_weight`` is
circular: the fit was derived from that basis, so the projection is close to the
identity, the measured change is near zero, and the test reports a pass it
cannot fail. A design that cannot fail never reaches a control at all.

See ``docs/superpowers/specs/2026-08-12-spline-substitution-design.md``.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

__all__ = [
    "edge_contribution",
    "eval_fit_torch",
    "permute_fits",
    "substitute_context",
    "substitutions_are_active",
]


# --------------------------------------------------------------------------- #
# Closed forms on tensors
# --------------------------------------------------------------------------- #
def eval_fit_torch(fit: dict, xs: torch.Tensor) -> torch.Tensor:
    """Evaluate a :func:`pri2m.xai.symbolic.fit_curve` result on a tensor.

    A torch mirror of :func:`pri2m.xai.symbolic.eval_fit`, dispatching over the
    same families with the same parameter order (``symbolic.py:326-350``). It
    exists because the substitution hook runs inside the recurrent loop on
    device tensors, where a NumPy round-trip per timestep would be slow and
    would force a host copy.

    ``tests/test_substitution.py`` asserts the two agree to 1e-10 on every
    family, so the paths cannot drift apart -- a transposed parameter would
    otherwise still produce a smooth curve and a plausible AUC.

    Raises
    ------
    ValueError
        If ``fit["name"]`` is ``None`` (a fit the R^2 gate rejected) or is not a
        recognised family.
    """
    name = fit.get("name")
    if name is None:
        raise ValueError("eval_fit_torch: fit has no accepted family (name is None)")
    p = [float(v) for v in fit["params"]]

    if name == "const":
        return torch.full_like(xs, p[0])
    if name == "linear":
        return p[0] * xs + p[1]
    if name == "quadratic":  # highest power first: a*x^2 + b*x + c
        return p[0] * xs**2 + p[1] * xs + p[2]
    if name == "cubic":  # highest power first: a*x^3 + b*x^2 + c*x + d
        return p[0] * xs**3 + p[1] * xs**2 + p[2] * xs + p[3]
    if name == "saturating":
        return p[0] * torch.tanh(p[1] * xs + p[2]) + p[3]
    if name == "oscillatory":
        return p[0] * torch.sin(p[1] * xs + p[2]) + p[3]
    if name == "abs":
        return p[0] * torch.abs(xs) + p[1]
    if name == "relu":
        return p[0] * torch.clamp(xs, min=0.0) + p[1]
    raise ValueError(f"eval_fit_torch: unknown family {name!r}")


# --------------------------------------------------------------------------- #
# Exact per-edge contribution
# --------------------------------------------------------------------------- #
def edge_contribution(kan, o: int, i: int, x: torch.Tensor) -> torch.Tensor:
    """The exact contribution of edge ``(o, i)`` to ``kan(x)[..., o]``.

    Mirrors :meth:`pri2m.models.kan.KANLinear.forward` term by term, so that
    summing this over every ``(o, i)`` reconstructs ``forward`` to float64
    round-off (asserted in the tests at 1e-12; observed gap ~9e-16).

    It is NOT bit-identical, and that is expected rather than a defect:
    ``forward`` reduces over all edges at once through a fused ``F.linear`` and
    an ``einsum``, while a per-edge sum accumulates sequentially, so the two
    round differently. An error large enough to matter -- a transposed index or
    a dropped ``spline_scale`` -- moves this by order 1, not by 1e-15.

    Parameters
    ----------
    kan:
        A :class:`pri2m.models.kan.KANLinear`.
    o, i:
        Output and input index of the edge.
    x:
        ``[..., in_features]``. Only column ``i`` affects the result.

    Returns
    -------
    torch.Tensor
        Shape ``x.shape[:-1]``.
    """
    lead = x.shape[:-1]
    x_flat = x.reshape(-1, kan.in_features)

    base = kan.base_weight[o, i] * kan.base_activation(x_flat[:, i])

    bases = kan.b_splines(x_flat)  # [M, in, n_coeff]
    spline = (bases[:, i, :] * kan.spline_weight[o, i, :]).sum(-1) * kan.spline_scale

    return (base + spline).reshape(*lead)


# --------------------------------------------------------------------------- #
# Substitution
# --------------------------------------------------------------------------- #
def _make_hook(entries: list[dict], clamp: bool):
    """Build a forward hook adding ``phi_sym - phi_orig`` for each entry."""

    def hook(module, inputs, output):
        x = inputs[0]
        lead = x.shape[:-1]
        x_flat = x.reshape(-1, module.in_features)
        out = output.reshape(-1, module.out_features).clone()
        for e in entries:
            o, i = int(e["out"]), int(e["in"])
            xi = x_flat[:, i]
            xs = torch.clamp(xi, float(e["lo"]), float(e["hi"])) if clamp else xi
            phi_sym = eval_fit_torch(e["fit"], xs)
            phi_orig = edge_contribution(module, o, i, x_flat)
            out[:, o] = out[:, o] + (phi_sym - phi_orig)
        return out.reshape(*lead, module.out_features)

    return hook


@contextlib.contextmanager
def substitute_context(model, subs: dict, clamp: bool = False):
    """Replace named edges' learned activations with their fitted closed forms.

    Parameters
    ----------
    model:
        Any ``nn.Module`` whose ``KANLinear`` sub-layers are reachable through
        ``model.named_modules()``.
    subs:
        ``{kan_name: [{"out", "in", "fit", "lo", "hi"}, ...]}``. ``fit`` is a
        :func:`pri2m.xai.symbolic.fit_curve` result; ``lo``/``hi`` are the range
        the edge's input visited when the fit was made.
    clamp:
        Evaluate the closed form at ``clip(x, lo, hi)``. Fits are only defined
        over the range their input visited: a spline saturates outside its grid
        while a fitted polynomial diverges, and held-out windows do leave that
        range. Both arms are run and the gap between them is a reported result,
        not something to smooth away.

    Yields
    ------
    ``model`` with the substitutions installed as forward hooks. Every hook is
    removed on exit, including when the body raises.

    Raises
    ------
    KeyError
        If a name in ``subs`` is not in ``model.named_modules()``. Raised before
        any hook is installed, so a bad request cannot leave the model modified.
    """
    modules = dict(model.named_modules())
    missing = [name for name in subs if name not in modules]
    if missing:
        raise KeyError(f"no module named {missing[0]!r} in model.named_modules()")

    handles = []
    try:
        for name, entries in subs.items():
            if not entries:
                continue
            handles.append(
                modules[name].register_forward_hook(_make_hook(list(entries), clamp))
            )
        yield model
    finally:
        for h in handles:
            h.remove()


def substitutions_are_active(
    model, subs: dict, x: torch.Tensor, tol: float = 0.0
) -> bool:
    """True when applying ``subs`` measurably changes ``model(x)``.

    Call this before every arm. A substitution that silently failed to apply
    would report perfect fidelity to the unsubstituted network -- the most
    damaging false result available here, because it is indistinguishable from a
    pass. Mirrors :func:`pri2m.xai.mode_ablation.edges_are_zeroed`, which exists
    for the same reason on the ablation side.
    """
    with torch.no_grad():
        before = model(x)
        with substitute_context(model, subs):
            after = model(x)
    return _max_abs_change(before, after) > tol


def _float_tensors(out) -> dict:
    """Every floating-point tensor in a model output, keyed for pairing.

    Phase 3 heads return a dict whose keys depend on configuration:
    ``b_over_blim``/``chatter_logit`` in point mode, plus
    ``b_over_blim_quantiles`` and a non-tensor ``quantile_levels`` in quantile
    mode, plus ``trajectory`` when that head is gated on.

    Comparing every float tensor rather than naming one key is deliberate. An
    earlier version read ``out["logits"]``, a key no Phase 3 head has ever
    produced, and it went unnoticed because the unit fixture returned a bare
    tensor and never took the dict branch -- so the guard crashed the first time
    it met a real model. Non-tensor entries are skipped rather than compared.
    """
    if isinstance(out, dict):
        return {k: v for k, v in out.items()
                if isinstance(v, torch.Tensor) and v.is_floating_point()}
    return {"": out}


def _max_abs_change(before, after) -> float:
    """Largest absolute difference across every float tensor the model returns."""
    a, b = _float_tensors(before), _float_tensors(after)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError(
            f"model output carries no comparable float tensor "
            f"(before: {sorted(a)}, after: {sorted(b)})"
        )
    return max(float(torch.max(torch.abs(b[k] - a[k])).item()) for k in shared)


def permute_fits(entries: list[dict], seed: int = 0) -> list[dict]:
    """Move each edge's equation onto a *different* edge -- the matched control.

    A derangement, not a shuffle: no equation may land back on its own edge.
    Update 12's tau-shuffle handed 2 of 24 windows their own value back at
    ``shuffle_seed: 0``. That biased toward the control arm, which was the safe
    direction, but by luck rather than by design, and PROGRESS item A3 already
    records a derangement as the fix.

    The control is matched to the correct arm on family and complexity by
    construction, because it permutes the same equations over the same edges --
    only the pairing is broken.

    Raises
    ------
    ValueError
        If fewer than two entries are supplied, since no derangement exists.
    """
    n = len(entries)
    if n < 2:
        raise ValueError(f"cannot build a derangement of {n} entries; need at least 2")
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(n)
        if not np.any(order == np.arange(n)):
            break
    return [{**entries[k], "fit": entries[int(order[k])]["fit"]} for k in range(n)]
