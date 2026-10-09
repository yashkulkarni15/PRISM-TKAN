"""Tests for ``scripts/heldout_pruning_phase3.py``.

The pruning curve already in every fold manifest was computed on the
in-distribution sample, starts at 2% of edges kept, and has six points. It is
under-resolved at BOTH ends: there is no point below 0.02, and its steepest
measured segment is 0.50 -> 1.00 in three of five folds (fold 0 +0.094, fold 2
+0.123, fold 4 +0.076), so one segment spanning half the edges hides where that
contribution sits.

These tests cover the edge selection and its control. Without a size-matched
random control the curve measures "removing edges hurts", not "the importance
ranking is correct".
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "heldout_pruning_phase3", _ROOT / "scripts" / "heldout_pruning_phase3.py"
)
HP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(HP)


# rank_edges output shape: (kan_name, o, i, importance), sorted desc.
_RANKED = [("kan_a", 0, i, 10.0 - i) for i in range(10)]


def test_keeping_a_fraction_zeroes_the_complement():
    zeroed = HP.select_edges(_RANKED, 0.3)
    assert sum(len(v) for v in zeroed.values()) == 7  # 10 total, keep 3


def test_keeping_everything_zeroes_nothing():
    assert sum(len(v) for v in HP.select_edges(_RANKED, 1.0).values()) == 0


def test_the_kept_edges_are_the_most_important_ones():
    zeroed = HP.select_edges(_RANKED, 0.3)["kan_a"]
    for kept in ((0, 0), (0, 1), (0, 2)):
        assert kept not in zeroed
    assert (0, 9) in zeroed


def test_random_control_is_exactly_size_matched():
    """Not approximately. A control of a different size measures size."""
    ranked_zeroed = HP.select_edges(_RANKED, 0.3)
    n = sum(len(v) for v in ranked_zeroed.values())
    ctrl = HP.random_edges(_RANKED, n, seed=0)
    assert sum(len(v) for v in ctrl.values()) == n


def test_random_control_is_reproducible_and_seed_sensitive():
    a = HP.random_edges(_RANKED, 5, seed=0)
    assert a == HP.random_edges(_RANKED, 5, seed=0)
    assert a != HP.random_edges(_RANKED, 5, seed=1)


def test_a_fraction_that_rounds_to_zero_kept_edges_is_refused():
    """Silently keeping nothing would report a floor AUC as a measurement."""
    with pytest.raises(ValueError, match="keeps 0 edges"):
        HP.select_edges(_RANKED, 0.0001)


def test_random_control_refuses_to_draw_more_edges_than_exist():
    with pytest.raises(ValueError, match="only 10"):
        HP.random_edges(_RANKED, 11, seed=0)


def test_default_fractions_cover_both_under_resolved_ends():
    """Guards the reason this runner exists at all.

    The published curve's gaps are below 0.02 and across 0.50-1.00. If someone
    trims the defaults back to the published six, this fails.
    """
    fr = HP.DEFAULT_FRACTIONS
    assert min(fr) < 0.02, "no resolution below the published curve's first point"
    assert any(0.5 < f < 1.0 for f in fr), "0.50-1.00 still spanned by one segment"
    assert fr == tuple(sorted(fr)), "fractions must be ascending for a readable curve"


# --------------------------------------------------------------------------- #
# Runner/library seam: the evaluator's return shape
# --------------------------------------------------------------------------- #

# The real shape of eval_heldout_phase3.run's return (scripts/eval_heldout_phase3.py:196-213):
# the AUC lives at result["metrics"]["auc"], NOT at the top level. Reading the
# top level yields None for a run that in fact succeeded, which is how job
# 57251412 died on 2026-08-12 after a clean eval printing auc=0.8994.
_REAL_SHAPE = {
    "checkpoint": "x.pt",
    "datasets": [121, 122, 123, 124, 125],
    "n_eval_files": 957,
    "n_eval_windows": 30789,
    "metrics": {"auc": 0.8994, "mae": 4.4793, "rmse": 9.9311, "acc": 0.8036,
                "n_files": 957, "n_nonfinite_files": 0},
}


class _Args:
    features_csv = "f.csv"
    npz_root = "n"
    preprocessing = None
    device = "cpu"
    batch_size = 64


def test_score_reads_the_auc_out_of_the_nested_metrics_block(tmp_path, monkeypatch):
    monkeypatch.setattr(HP.EH, "run", lambda **kw: _REAL_SHAPE)
    auc, err = HP._score("x.pt", _Args(), None, object(), "t", tmp_path)
    assert auc == 0.8994 and err is None


def test_score_reports_a_missing_metrics_block_instead_of_returning_none_silently(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(HP.EH, "run", lambda **kw: {"auc": 0.9, "datasets": [1]})
    auc, err = HP._score("x.pt", _Args(), None, object(), "t", tmp_path)
    assert auc is None
    assert "metrics.auc" in err
