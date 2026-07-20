"""build_labeled_batch lives in pri2m.xai.sampling; whitebox re-exports it."""
import numpy as np

from pri2m.data.phase3_sequence import WindowConfig
from tests._phase3_synth import make_synth_npz_tree, make_synth_table, CHANNELS as SYNTH_CHANNELS


def _tree_and_table(tmp_path):
    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2), n_files=4, length=256, seed=0)
    table = make_synth_table(npz_root)
    csv = tmp_path / "table.csv"
    table.to_csv(csv, index=False)
    return npz_root, csv


def test_build_labeled_batch_moved_and_seeded(tmp_path):
    from pri2m.xai import sampling

    npz_root, csv = _tree_and_table(tmp_path)
    cfg = WindowConfig(length=128, stride=64, channels=sampling.CHANNELS)
    b1, l1 = sampling.build_labeled_batch(csv, npz_root, None, 6, cfg, seed=7)
    b2, l2 = sampling.build_labeled_batch(csv, npz_root, None, 6, cfg, seed=7)
    assert b1["window"].shape[0] == len(l1) and set(l1) == {0, 1}
    assert np.array_equal(l1, l2)
    assert (b1["window"] == b2["window"]).all()
    assert sampling.CHANNELS == SYNTH_CHANNELS


def test_whitebox_reexports_sampling():
    import whitebox_phase3 as W
    from pri2m.xai import sampling

    assert W.build_labeled_batch is sampling.build_labeled_batch
    assert W._npz_exists is sampling.npz_exists
    assert W.CHANNELS == sampling.CHANNELS
