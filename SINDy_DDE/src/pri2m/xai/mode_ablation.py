"""Remove a set of learned edges and measure what it costs.

U14 section 4 established that trained networks are more self-consistent in the
shapes of their activation functions than randomly initialised ones, at every
tested setting, and could not convert that into a verdict: the pass/fail flips
with the number of prototypes requested, and that number was not pre-committed.

An ablation sidesteps the problem entirely. Zeroing the edges assigned to a mode
and measuring held-out accuracy is a **causal** measurement that does not depend
on the grouping being reproducible, so it survives the one defect section 4
could not argue its way out of.

:func:`~pri2m.xai.activation_modes.mode_edge_sets` already produces the three
sets required -- the mode's own edges, a random set matched on **count and
summed importance**, and an equal-size set drawn from the other modes -- and has
been tested since 2026-07-30 without ever being called. This module supplies the
missing half: apply an edge set to a live model, and judge the three arms
together.

Both controls are load-bearing rather than decorative. "Removing 300 edges cost
0.04 AUC" says nothing on its own: the matched-random arm answers what removing
300 *arbitrary* edges of the same total importance costs, and the other-modes
arm answers whether this mode is special among modes. A mode that fails either
comparison has demonstrated nothing, and :func:`ablation_verdict` refuses to
call it.
"""

from __future__ import annotations

import contextlib

import torch

__all__ = ["resolve_kan", "zero_edges", "edges_are_zeroed", "ablation_verdict"]


def resolve_kan(model, kan_name: str):
    """Resolve ``encoder.layers.0.kan_z`` to the module itself.

    Raises rather than returning ``None`` on an unknown path: the arms are
    matched on edge COUNT, so silently skipping an edge would shrink one arm
    against its controls without anything saying so.
    """
    obj = model
    for part in str(kan_name).split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _edge_slices(model, edges):
    """``(kan, out, in)`` triples, validated up front."""
    resolved = []
    for e in edges:
        kan = resolve_kan(model, e["kan"])
        o, i = int(e["out"]), int(e["in"])
        # Touch the tensors now so a bad index fails before anything is mutated.
        _ = kan.base_weight[o, i]
        if hasattr(kan, "spline_weight"):
            _ = kan.spline_weight[o, i]
        resolved.append((kan, o, i))
    return resolved


@contextlib.contextmanager
def zero_edges(model, edges):
    """Temporarily remove ``edges`` from ``model``, restoring exactly on exit.

    An edge is ``base_weight[o,i] * base_activation(x)`` plus the spline term,
    so **both** paths are zeroed. Zeroing only the spline would leave the SiLU
    path contributing and understate every ablation.

    Restoration is exact and happens even when the body raises, because the arms
    are measured on one model in sequence and a leak would contaminate every arm
    after it.
    """
    resolved = _edge_slices(model, edges)
    saved = []
    try:
        with torch.no_grad():
            for kan, o, i in resolved:
                base = kan.base_weight[o, i].clone()
                spline = (kan.spline_weight[o, i].clone()
                          if hasattr(kan, "spline_weight") else None)
                saved.append((kan, o, i, base, spline))
                kan.base_weight[o, i] = 0.0
                if spline is not None:
                    kan.spline_weight[o, i] = 0.0
        yield model
    finally:
        with torch.no_grad():
            for kan, o, i, base, spline in saved:
                kan.base_weight[o, i] = base
                if spline is not None:
                    kan.spline_weight[o, i] = spline


def edges_are_zeroed(model, edges, tol: float = 0.0) -> bool:
    """Whether every edge in ``edges`` is currently removed.

    Used as a guard inside the runner: an ablation that silently failed to
    apply would report the mode as unnecessary, which is the most damaging
    possible false result here.
    """
    for kan, o, i in _edge_slices(model, edges):
        if abs(float(kan.base_weight[o, i].detach())) > tol:
            return False
        if hasattr(kan, "spline_weight") and \
                float(kan.spline_weight[o, i].detach().abs().max()) > tol:
            return False
    return True


def ablation_verdict(baseline_auc, mode_auc, random_matched_auc,
                     other_modes_auc, floor: float) -> dict:
    """Judge one mode against its two controls and the repeatability floor.

    ``floor`` is the smallest AUC difference this pipeline can distinguish from
    a re-run of itself (Update 14 measured 0.0092 as a lower bound). An effect
    below it is not an effect, however clean the ablation.

    Returns ``claimable=None`` -- never ``False`` -- when an arm is missing: a
    control that failed to evaluate must not read as a control that passed.
    """
    out = {
        "baseline_auc": baseline_auc,
        "mode_auc": mode_auc,
        "random_matched_auc": random_matched_auc,
        "other_modes_auc": other_modes_auc,
        "floor": float(floor),
        "mode_drop": None,
        "random_matched_drop": None,
        "other_modes_drop": None,
        "exceeds_floor": None,
        "beats_random_control": None,
        "beats_other_modes_control": None,
        "claimable": None,
        "undecided_reason": None,
    }
    arms = (baseline_auc, mode_auc, random_matched_auc, other_modes_auc)
    if any(a is None for a in arms):
        out["undecided_reason"] = "missing_arm"
        return out

    base = float(baseline_auc)
    out["mode_drop"] = base - float(mode_auc)
    out["random_matched_drop"] = base - float(random_matched_auc)
    out["other_modes_drop"] = base - float(other_modes_auc)
    # The mode has to hurt MORE than each control by more than the floor, not
    # merely hurt more: a difference the pipeline cannot resolve is not one.
    out["exceeds_floor"] = bool(out["mode_drop"] > float(floor))
    out["beats_random_control"] = bool(
        out["mode_drop"] - out["random_matched_drop"] > float(floor))
    out["beats_other_modes_control"] = bool(
        out["mode_drop"] - out["other_modes_drop"] > float(floor))
    out["claimable"] = bool(out["exceeds_floor"]
                            and out["beats_random_control"]
                            and out["beats_other_modes_control"])
    return out
