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

import numpy as np
import torch

__all__ = ["resolve_kan", "zero_edges", "edges_are_zeroed", "ablation_verdict",
           "combined_edge_sets", "pair_verdict", "all_gated_verdict"]

# Label the merged modes are relabelled to. assign_to_prototypes only ever emits
# labels >= 0, so -1 cannot collide with a real mode; combined_edge_sets still
# refuses labels that already contain it rather than trusting that.
_MERGED = -1


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


# --------------------------------------------------------------------------- #
# Exp 1b: deleting several modes at once
#
# One network (fold 4) has no single necessary cluster at k=3. "No cluster is
# necessary" and "the clusters cover for each other" predict the same singles
# and different pairs, so the only test that separates them deletes two at
# once. The interaction term below is that test; it was fixed before any pair
# was run (docs/superpowers/plans/2026-10-05-paper-closeout.md, "Exp 1b").
# --------------------------------------------------------------------------- #
def combined_edge_sets(norm_bundle: dict, labels, modes, *, seed: int,
                       population: dict | None = None) -> dict:
    """Edge sets for deleting several modes at once, with a matched random arm.

    Relabels every edge in ``modes`` to one merged label and hands that to
    :func:`~pri2m.xai.activation_modes.mode_edge_sets`, so the pair's random
    control is drawn by exactly the machinery the single-mode arms used --
    matched on count AND summed importance, from ``population`` -- rather than
    by a second implementation that could drift from it.

    ``other_modes_matched`` is whatever is left over: one cluster for a pair at
    k=3, and empty when ``modes`` covers every label (the all-gated arm). The
    runner does not score it; it is returned only because mode_edge_sets
    returns it.

    Raises on a mode that has no edges: deleting one cluster under a "pair"
    label would be read as a pair result, which is worse than no result.
    """
    from pri2m.xai import activation_modes as AM

    labels = np.asarray(labels)
    wanted = sorted({int(m) for m in modes})
    if not wanted:
        raise ValueError("combined_edge_sets needs at least one mode")
    present = {int(v) for v in labels.tolist()}
    absent = [m for m in wanted if m not in present]
    if absent:
        raise ValueError(f"modes {absent} have no edges (present: {sorted(present)})")
    if _MERGED in present:
        raise ValueError(f"labels already use the merge sentinel {_MERGED}")
    # np.where builds a new array, so the caller's labels -- which the
    # single-mode arms are still indexed on -- are never touched.
    merged = np.where(np.isin(labels, wanted), _MERGED, labels)
    sets = AM.mode_edge_sets(norm_bundle, merged, mode=_MERGED, seed=int(seed),
                             population=population)
    sets["modes"] = wanted
    return sets


def _matched_deletion(baseline_auc, deleted_auc, random_matched_auc,
                      floor: float) -> dict | None:
    """Drop, random drop and claimability for one deletion vs its random arm.

    The rule is ablation_verdict's minus the other-modes control: at k=3 a pair
    leaves a single cluster, and the all-gated arm leaves none, so there is no
    other-modes set to compare against. Returns ``None`` when any arm is
    missing, so the caller can mark it UNDECIDED rather than False.
    """
    if any(a is None for a in (baseline_auc, deleted_auc, random_matched_auc)):
        return None
    base = float(baseline_auc)
    drop = base - float(deleted_auc)
    rdrop = base - float(random_matched_auc)
    exceeds = bool(drop > float(floor))
    beats = bool(drop - rdrop > float(floor))
    return {"drop": drop, "random_matched_drop": rdrop, "exceeds_floor": exceeds,
            "beats_random_control": beats, "claimable": bool(exceeds and beats)}


def pair_verdict(baseline_auc, pair_auc, random_matched_auc,
                 single_drops: dict | None, floor: float) -> dict:
    """Judge a two-mode deletion, and whether the modes cover for each other.

    Pre-committed reads (Exp 1b):

    * ``claimable`` = drop > floor AND drop - random_drop > floor.
    * ``interaction`` = pair drop - (sum of the single-mode drops).
      ``super_additive`` above +floor (the clusters are redundant: each hides
      the other's loss, the fold-4 hypothesis), ``sub_additive`` below -floor
      (they overlap in what they do), else ``additive``.

    Interaction is reported whether or not the pair is claimable -- it is a
    measured quantity, and a super-additive pair that fails its random control
    is itself informative. The fold-4 hypothesis needs BOTH.

    A missing pair/random/baseline arm leaves everything ``None`` with
    ``undecided_reason="missing_arm"``: never False. A missing single drop only
    blocks the interaction (``"missing_single_drop"``); the pair's own arms
    still decide claimability.
    """
    singles = dict(single_drops or {})
    out = {
        "baseline_auc": baseline_auc,
        "pair_auc": pair_auc,
        "random_matched_auc": random_matched_auc,
        "single_drops": {str(k): v for k, v in singles.items()},
        "floor": float(floor),
        "pair_drop": None,
        "random_matched_drop": None,
        "exceeds_floor": None,
        "beats_random_control": None,
        "claimable": None,
        "sum_single_drops": None,
        "interaction": None,
        "interaction_class": None,
        "undecided_reason": None,
    }
    core = _matched_deletion(baseline_auc, pair_auc, random_matched_auc, floor)
    if core is None:
        out["undecided_reason"] = "missing_arm"
        return out
    out["pair_drop"] = core.pop("drop")
    out.update(core)

    if not singles or any(v is None for v in singles.values()):
        out["undecided_reason"] = "missing_single_drop"
        return out
    total = float(sum(float(v) for v in singles.values()))
    inter = out["pair_drop"] - total
    out["sum_single_drops"] = total
    out["interaction"] = inter
    out["interaction_class"] = ("super_additive" if inter > float(floor) else
                                "sub_additive" if inter < -float(floor) else
                                "additive")
    return out


def all_gated_verdict(baseline_auc, all_auc, random_matched_auc,
                      floor: float) -> dict:
    """Every gated edge deleted, against its matched random arm. Reference only.

    Same claimability rule as :func:`pair_verdict` and no interaction term: with
    every mode removed there is no "sum of parts" the plan committed to reading.
    """
    out = {
        "baseline_auc": baseline_auc,
        "all_auc": all_auc,
        "random_matched_auc": random_matched_auc,
        "floor": float(floor),
        "all_drop": None,
        "random_matched_drop": None,
        "exceeds_floor": None,
        "beats_random_control": None,
        "claimable": None,
        "undecided_reason": None,
    }
    core = _matched_deletion(baseline_auc, all_auc, random_matched_auc, floor)
    if core is None:
        out["undecided_reason"] = "missing_arm"
        return out
    out["all_drop"] = core.pop("drop")
    out.update(core)
    return out
