"""Smoke test for the shared Phase 3 synthetic fixtures."""

from __future__ import annotations

import numpy as np

from pri2m.phase3.splits import parent_group
from tests._phase3_synth import (
    CHANNELS,
    make_synth_folds,
    make_synth_npz_tree,
    make_synth_table,
)


def test_npz_tree_and_table_schema(tmp_path):
    root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2, 3), n_files=4, seed=1)

    npzs = sorted(root.rglob("*.npz"))
    assert len(npzs) == 9 * 4  # 3 clean + 3 (1%) + 3 (5%) datasets, 4 files each

    bundle = np.load(npzs[0])
    for ch in CHANNELS:
        assert ch in bundle.files
        assert bundle[ch].shape == (4096,)
    assert {"fs", "label", "omega", "tpf", "fn", "dataset", "source_file"} <= set(bundle.files)

    table = make_synth_table(root)
    assert len(table) == 9 * 4
    needed = {
        "dataset", "file", "label", "b_over_blim", "blim", "noise_score", "mode_label",
        "parent_group", "sample_weight", "exclude_supervised", "physics_usable",
        "omega", "tpf", "fn",
    }
    assert needed <= set(table.columns)


def test_audit_pathologies_excluded(tmp_path):
    root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2, 3), n_files=4, seed=2)
    table = make_synth_table(root)

    # The two injected pathologies exist and are excluded with zero weight.
    assert (table["blim"] <= 0).any(), "expected a negative-blim row"
    assert (table["mode_label"] == "metrics_unavailable").any()
    excluded = table[table["exclude_supervised"]]
    assert len(excluded) >= 2
    assert (excluded["sample_weight"] == 0).all()
    # Positive weights renormalize to mean 1.
    pos = table.loc[table["sample_weight"] > 0, "sample_weight"]
    assert abs(float(pos.mean()) - 1.0) < 1e-9


def test_folds_are_leakage_free_and_group_mirrors(tmp_path):
    root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2, 3), n_files=4, seed=3)
    table = make_synth_table(root)
    folds = make_synth_folds(table, n_splits=3, seed=0)

    assert folds["leakage_verified"] is True
    # Mirrors collapse onto the clean parent, so they cannot straddle a split.
    assert parent_group(1) == parent_group(61) == parent_group(91) == 1
    assert 1 in {int(k) for k in folds["group_to_fold"]}
