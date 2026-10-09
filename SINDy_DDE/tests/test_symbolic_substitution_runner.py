"""Tests for ``scripts/symbolic_substitution_phase3.py``.

Covers the runner/library seam, not just pure helpers. U14-B died on the cluster
in k=2 with a ``TypeError`` because a smoke test never exercised the callback
path, so the seam is what gets tested here.

One thing this file pins deliberately: ``symbolic.symbolic_table`` computes each
edge's fit internally but returns only the display ``formula``, dropping
``params``/``lo``/``hi``. The substitution needs those, so the runner replicates
the ranking rather than modifying validated code -- and
``test_ranking_matches_the_symbolic_table_tie_break`` is what guarantees the two
select the SAME edges.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "symbolic_substitution_phase3", _ROOT / "scripts" / "symbolic_substitution_phase3.py"
)
SS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SS)


def _rows(n: int = 25) -> list[dict]:
    return [
        {
            "kan": "kan",
            "out": 0,
            "in": j,
            "importance": 10.0 - j,
            "accepted": True,
            "r2": 0.99,
            "formula": "1.0*x",
            "fit": {"name": "linear", "params": [1.0, 0.0]},
            "lo": -1.0,
            "hi": 1.0,
        }
        for j in range(n)
    ]


# --------------------------------------------------------------------------- #
# Ranking agreement with the validated symbolic table
# --------------------------------------------------------------------------- #


def test_ranking_matches_the_symbolic_table_tie_break():
    """Importance descending, then (kan, out, in) ascending -- symbolic.py:449.

    The importance matrix below has a deliberate three-way tie at 5.0 spanning
    two modules, so a stable-sort-only implementation (which would keep
    insertion order) is distinguishable from the explicit tie-break.
    """
    activations = {
        "kan_b": {"importance": np.array([[5.0, 9.0], [1.0, 5.0]])},
        "kan_a": {"importance": np.array([[5.0, 2.0]])},
    }
    got = SS.rank_like_symbolic_table(activations, k=5)
    assert got == [
        ("kan_b", 0, 1),  # 9.0
        ("kan_a", 0, 0),  # 5.0, kan_a sorts before kan_b
        ("kan_b", 0, 0),  # 5.0
        ("kan_b", 1, 1),  # 5.0, same kan as above but higher (out, in)
        ("kan_a", 0, 1),  # 2.0
    ]


def test_ranking_respects_k():
    activations = {"kan": {"importance": np.array([[3.0, 2.0, 1.0]])}}
    assert SS.rank_like_symbolic_table(activations, k=2) == [("kan", 0, 0), ("kan", 0, 1)]


# --------------------------------------------------------------------------- #
# Substitution assembly
# --------------------------------------------------------------------------- #


def test_build_substitutions_takes_the_top_k_by_importance():
    subs = SS.build_substitutions(_rows(), k=5)
    assert sum(len(v) for v in subs.values()) == 5
    assert sorted(e["in"] for e in subs["kan"]) == [0, 1, 2, 3, 4]


def test_build_substitutions_skips_unaccepted_rows():
    """A rejected fit has no closed form to substitute; it must not be faked."""
    rows = _rows()
    rows[0]["accepted"] = False
    subs = SS.build_substitutions(rows, k=3)
    kept = sorted(e["in"] for e in subs["kan"])
    assert 0 not in kept and len(kept) == 3


def test_build_substitutions_reports_a_short_arm_rather_than_silently_shrinking():
    rows = _rows(3)
    for r in rows:
        r["accepted"] = False
    subs = SS.build_substitutions(rows, k=3)
    assert sum(len(v) for v in subs.values()) == 0
    assert SS.count_substituted(subs) == 0


def test_build_substitutions_carries_the_fit_and_the_visited_range():
    """Missing lo/hi would silently disable clamping."""
    entry = SS.build_substitutions(_rows(), k=1)["kan"][0]
    assert entry["fit"] == {"name": "linear", "params": [1.0, 0.0]}
    assert entry["lo"] == -1.0 and entry["hi"] == 1.0


# --------------------------------------------------------------------------- #
# Pre-committed verdict
# --------------------------------------------------------------------------- #


def test_verdict_requires_both_criteria():
    # fold 2's own baseline is 0.8994, tolerance 0.013
    assert SS.verdict(0.8994, 0.8950, 0.8500, 0.013)["passed"] is True
    assert SS.verdict(0.8994, 0.8000, 0.7000, 0.013)["passed"] is False  # absolute
    assert SS.verdict(0.8994, 0.8950, 0.8900, 0.013)["passed"] is False  # control


def test_verdict_reports_which_criterion_failed():
    v = SS.verdict(0.8994, 0.8950, 0.8900, 0.013)
    assert v["absolute_ok"] is True and v["control_ok"] is False


def test_verdict_returns_none_when_an_arm_is_missing():
    """A crashed arm must never read as a negative result."""
    assert SS.verdict(0.8994, None, 0.85, 0.013)["passed"] is None
    assert SS.verdict(None, 0.89, 0.85, 0.013)["passed"] is None


def test_verdict_control_gap_is_strict():
    """A gap exactly equal to the tolerance does not clear it.

    Uses binary-exact values (1.0, 0.5) rather than realistic AUCs on purpose:
    0.9000 - 0.8870 is 0.013000000000000012 in float64, which really is greater
    than 0.013, so realistic numbers cannot express exact equality here.
    """
    assert SS.verdict(1.0, 1.0, 0.5, 0.5)["control_ok"] is False
    assert SS.verdict(1.0, 1.0, 0.25, 0.5)["control_ok"] is True


def test_verdict_uses_the_folds_own_baseline_not_the_five_fold_mean():
    """Scoring against the mean instead of the fold changes the answer.

    A substituted AUC of 0.8940 sits 0.0054 from fold 2's own baseline (0.8994)
    and 0.0140 from the five-fold mean (0.9080). Against the fold it passes;
    against the mean it fails. Mixing the two is what Update 14 records as the
    reason 0.0092 misled.
    """
    assert SS.verdict(0.8994, 0.8940, 0.8500, 0.013)["absolute_ok"] is True
    assert SS.verdict(0.9080, 0.8940, 0.8500, 0.013)["absolute_ok"] is False


# --------------------------------------------------------------------------- #
# Runner/library seam: the evaluator's return shape
# --------------------------------------------------------------------------- #

# eval_heldout_phase3.run nests its metrics (scripts/eval_heldout_phase3.py:196-213).
# Reading the top level returns None for a run that succeeded -- job 57251412,
# 2026-08-12, died exactly that way after printing auc=0.8994.
_REAL_SHAPE = {
    "checkpoint": "x.pt",
    "datasets": [121, 122, 123, 124, 125],
    "metrics": {"auc": 0.8994, "mae": 4.4793, "rmse": 9.9311, "acc": 0.8036},
}


class _Args:
    features_csv = "f.csv"
    npz_root = "n"
    preprocessing = None
    device = "cpu"
    batch_size = 64


def test_score_reads_the_auc_out_of_the_nested_metrics_block(tmp_path, monkeypatch):
    monkeypatch.setattr(SS.EH, "run", lambda **kw: _REAL_SHAPE)
    auc, err = SS._score("x.pt", _Args(), None, object(), "t", tmp_path)
    assert auc == 0.8994 and err is None


def test_score_reports_a_missing_metrics_block(tmp_path, monkeypatch):
    monkeypatch.setattr(SS.EH, "run", lambda **kw: {"auc": 0.9})
    auc, err = SS._score("x.pt", _Args(), None, object(), "t", tmp_path)
    assert auc is None and "metrics.auc" in err


# --------------------------------------------------------------------------- #
# Restricting the substitution to one family of edges
#
# The five-fold substitution result covers the top 25 edges by importance, of
# which only three touch a delayed channel and fourteen sit in the classifier
# head. So it shows the equations reproduce the NETWORK; it does not show they
# reproduce the DELAY MECHANISM that Sections 2 and 3 describe. Restricting the
# selection to the delayed edges is what joins those two claims into one, and
# these tests pin the restriction so it cannot silently select something else.
# --------------------------------------------------------------------------- #
def _acts_with_two_families():
    """Two edges, one delayed and one instantaneous, delayed LESS important."""
    imp = np.zeros((1, 2))
    imp[0, 0] = 9.0      # instantaneous, most important
    imp[0, 1] = 1.0      # delayed, would never make a global top-1
    return {"kan": {"importance": imp,
                    "visited_lo": np.array([-1.0, -1.0]),
                    "visited_hi": np.array([1.0, 1.0])}}


def test_unrestricted_ranking_is_unchanged_when_no_filter_is_given():
    """The published five-fold numbers must not move because this option exists."""
    acts = _acts_with_two_families()

    assert SS.rank_like_symbolic_table(acts, 2) == [("kan", 0, 0), ("kan", 0, 1)]


def test_restricting_selects_the_named_family_over_global_importance():
    acts = _acts_with_two_families()

    picked = SS.rank_like_symbolic_table(acts, 1, allowed={("kan", 1)})

    assert picked == [("kan", 0, 1)]


def test_restricting_still_ranks_by_importance_within_the_family():
    imp = np.array([[1.0, 5.0, 3.0]])
    acts = {"kan": {"importance": imp,
                    "visited_lo": np.zeros(3), "visited_hi": np.ones(3)}}

    picked = SS.rank_like_symbolic_table(acts, 2, allowed={("kan", 1), ("kan", 2)})

    assert picked == [("kan", 0, 1), ("kan", 0, 2)]


def test_an_empty_restriction_raises_rather_than_silently_scoring_the_baseline():
    """Substituting nothing scores exactly baseline, which would read as a pass."""
    acts = _acts_with_two_families()

    with pytest.raises(ValueError, match="no edges match"):
        SS.rank_like_symbolic_table(acts, 2, allowed=set())


def test_restriction_asking_for_more_edges_than_exist_returns_what_there_is():
    acts = _acts_with_two_families()

    picked = SS.rank_like_symbolic_table(acts, 25, allowed={("kan", 1)})

    assert picked == [("kan", 0, 1)]


def test_edges_matching_picks_the_named_family_out_of_the_semantics_map():
    semantics = {"kan_z": [{"name": "xtpos"}, {"name": "delay:xtpos"}],
                 "cls_head": [{"name": "h4"}]}

    assert SS.edges_matching(semantics, "delay:") == {("kan_z", 1)}


def test_edges_matching_is_a_prefix_test_not_a_substring_test():
    """A channel merely NAMED like the prefix must not be swept in."""
    semantics = {"kan_z": [{"name": "predelay:xtpos"}, {"name": "delay:ytpos"}]}

    assert SS.edges_matching(semantics, "delay:") == {("kan_z", 1)}


def test_edges_matching_returns_empty_for_an_absent_family():
    semantics = {"kan_z": [{"name": "xtpos"}]}

    assert SS.edges_matching(semantics, "delay:") == set()
