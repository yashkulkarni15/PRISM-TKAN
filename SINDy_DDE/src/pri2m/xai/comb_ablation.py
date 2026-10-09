"""Delete the comb-carrying delay pairs and measure what it costs.

Update 15 Section 2 reads the delay coupling as a delay-and-add comb filter, and
Section 3 confirms that reading by driving the real learned curves with a tone
sweep. Both are descriptions. Section 4 says so explicitly: the comb adds a
mechanism, not a line of evidence, because nothing removes those pairs and shows
the network misses them.

This module is that removal. It selects exactly the pairs the Section 2 reading
describes, and builds the two controls without which "deleting them hurt" means
nothing:

``random_matched``
    the same number of edges carrying the same total importance, drawn from the
    wider edge population. Tests whether the damage is about *these* edges or
    merely about deleting that much.
``nonproportional_matched``
    the sharp one. The same number of edges carrying the same total importance,
    drawn only from delay pairs that fall BELOW the proportionality gate, i.e.
    pairs that are structurally identical, sit on the same delayed inputs, and
    simply do not carry the comb reading. If the comb is the thing that matters
    rather than delay coupling in general, this is the arm that shows it.

Nothing here loads a network. Selection is a function of the stored U13-C rows,
so it can be tested without a checkpoint, and the runner applies the result via
``mode_ablation.zero_edges``.
"""
from __future__ import annotations

import numpy as np

from pri2m.xai.comb import DEFAULT_CORR_THRESHOLD, usable

__all__ = ["select_comb_edges", "matched_control", "summed_importance",
           "build_controls"]


def _key(edge: dict) -> tuple:
    return (edge["kan"], int(edge["out"]), int(edge["in"]))


def summed_importance(edges) -> float:
    """Total importance carried by ``edges``."""
    return float(sum(float(e.get("importance", 0.0)) for e in edges))


def select_comb_edges(rows, corr_threshold: float = DEFAULT_CORR_THRESHOLD, *,
                      below_threshold: bool = False) -> list[dict]:
    """Both edges of every pair the comb reading describes.

    ``rows`` are the U13-C regenerative rows, each carrying ``corr``/``gain``,
    the pair's two input indices and their importances.

    The gate is ``abs(corr) >= corr_threshold`` on usable rows, matching
    :func:`comb._stats` exactly: a strongly negative correlation is a comb that
    cancels rather than one that is absent, so it is selected too.

    Both members are returned because the comb is a property of the pair. Deleting
    only the delayed half would leave the instantaneous path intact and measure
    something else.

    Parameters
    ----------
    below_threshold:
        Invert the gate, returning the usable pairs that do NOT clear it. This is
        the population the ``nonproportional_matched`` control draws from.

    Returns
    -------
    list of dict
        ``{"kan", "out", "in", "importance", "role", "corr"}``, deduplicated on
        ``(kan, out, in)`` -- an edge shared by two pairs is deleted once, since
        counting it twice would inflate this arm against its matched controls.
    """
    thr = float(corr_threshold)
    edges: list[dict] = []
    seen: set[tuple] = set()
    for row in rows:
        if not usable(row):
            continue
        clears = abs(float(row["corr"])) >= thr
        if clears == bool(below_threshold):
            continue
        for role, idx_key, imp_key in (("instantaneous", "in_inst", "importance_inst"),
                                       ("delayed", "in_delay", "importance_delay")):
            edge = {
                "kan": row["kan"], "out": int(row["out"]), "in": int(row[idx_key]),
                "importance": float(row.get(imp_key, 0.0) or 0.0),
                "role": role, "corr": float(row["corr"]),
            }
            k = _key(edge)
            if k in seen:
                continue
            seen.add(k)
            edges.append(edge)
    return edges


def matched_control(pool, target, *, seed: int = 0, n_trials: int = 200) -> list[dict]:
    """A control matched to ``target`` on count and summed importance.

    Best of ``n_trials`` deterministic draws, mirroring
    :func:`activation_modes.mode_edge_sets`'s matching so the two ablations are
    judged the same way.

    Target edges are removed from ``pool`` before drawing. Unlike the mode
    ablation's random arm, overlap here is not merely reported: the target is a
    named, structurally defined set, so an overlapping control would delete part
    of the very thing under test and shrink the gap being measured.

    Raises
    ------
    ValueError
        If the pool cannot supply a full-size control. Returning a short arm
        would put a smaller deletion up against a full-size one and read as
        evidence that the target matters.
    """
    target_keys = {_key(e) for e in target}
    avail = [e for e in pool if _key(e) not in target_keys]
    n = len(target)
    if len(avail) < n:
        raise ValueError(
            f"cannot draw a control of {n} edges; only {len(avail)} available "
            "once the target is excluded"
        )
    goal = summed_importance(target)
    imp = np.array([float(e.get("importance", 0.0)) for e in avail])
    rng = np.random.default_rng(int(seed))

    best, best_err = None, None
    for _ in range(int(n_trials)):
        pick = rng.choice(len(avail), size=n, replace=False)
        err = abs(float(imp[pick].sum()) - goal)
        if best_err is None or err < best_err:
            best, best_err = pick, err
    return [avail[int(i)] for i in sorted(best)]


def build_controls(*, population, below_gate, target, seed: int = 0,
                   n_trials: int = 200) -> tuple[dict, dict]:
    """Both control arms, with any unbuildable arm reported rather than shrunk.

    On the real manifests this is not hypothetical: fold 1 has 854 comb edges
    against only 682 below the gate, so its importance-matched non-proportional
    control cannot be drawn at all. The options there are to shorten the arm
    (which puts a smaller deletion against a full-size one and reads as evidence
    the target matters), to quietly drop it (which lets one surviving control
    decide a verdict that was specified to need two), or to say so. This says so.

    A caller that gets an arm back in ``errors`` must pass ``None`` for that arm
    to :func:`mode_ablation.ablation_verdict`, which returns ``claimable=None``.
    An absent control is UNDECIDED, never a pass -- the rule that killed the
    delay-lock in Update 12.

    Returns
    -------
    (controls, errors)
        ``controls`` maps arm name to its edge list; ``errors`` maps the name of
        each arm that could not be built to why. Every arm appears in exactly one.
    """
    pools = {"random_matched": population, "nonproportional_matched": below_gate}
    controls: dict = {}
    errors: dict = {}
    for name, pool in pools.items():
        try:
            controls[name] = matched_control(pool, target, seed=seed,
                                             n_trials=n_trials)
        except ValueError as exc:
            errors[name] = str(exc)
    return controls, errors


def population_from_ranked(ranked) -> list[dict]:
    """:func:`faithfulness.rank_edges` tuples as control-ready edge dicts.

    ``rank_edges`` returns ``(kan_name, out, in, importance)`` over EVERY edge in
    the network. That is the population the random control must be drawn from,
    which is the whole point of this function existing.
    """
    return [{"kan": kan, "out": int(o), "in": int(i), "importance": float(imp)}
            for kan, o, i, imp in ranked]


def check_population_is_wider(population, *, target, below_gate) -> None:
    """Refuse a control population confined to the delay pairs themselves.

    The first cluster run drew the random control from the delay-pair edges
    alone. The comb set and the below-gate set exhaust that population, so
    excluding the target left exactly the below-gate edges: the "random" arm and
    the "non-proportional" arm became the same control, and the question the
    random arm exists to ask -- is this about these edges or about deleting this
    much capacity -- was never asked. Four folds ran it without complaint;
    only fold 1's arithmetic made the draw impossible and exposed it.

    Raises
    ------
    ValueError
        If the population adds nothing beyond ``target`` and ``below_gate``.
    """
    confined = {_key(e) for e in target} | {_key(e) for e in below_gate}
    beyond = {_key(e) for e in population} - confined
    if not beyond:
        raise ValueError(
            "control population is not wider than the delay pairs themselves "
            f"({len(confined)} edges); the random arm would duplicate the "
            "non-proportional arm. Pass the whole network's edges."
        )
