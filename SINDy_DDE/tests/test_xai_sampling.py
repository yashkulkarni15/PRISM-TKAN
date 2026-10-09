"""build_labeled_batch lives in pri2m.xai.sampling; whitebox re-exports it."""
import numpy as np
import torch

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


# --------------------------------------------------------------------------- #
# Window geometry recovery (regression, 2026-08-12)
# --------------------------------------------------------------------------- #

# build_labeled_batch's cfg is REQUIRED. Passing None reaches
# Phase3SequenceDataset and raises "'NoneType' object has no attribute
# 'channels'" -- how job 57251934 died, after the held-out baseline had already
# scored cleanly at 0.8994.

_REAL_PREPROCESSING = {
    "channels": ["xtpos", "ytpos", "Forcex", "Forcey"],
    "window": {"length": 1024, "stride": 512},
    "pool": 4,
    "fold": 2,
}


def test_window_config_comes_from_the_checkpoints_own_preprocessing():
    from pri2m.xai import sampling

    cfg = sampling.window_config_from_preprocessing(_REAL_PREPROCESSING)
    assert cfg.length == 1024
    assert cfg.stride == 512
    assert cfg.channels == ("xtpos", "ytpos", "Forcex", "Forcey")
    # WindowConfig.channels is a tuple; a list would compare unequal downstream.
    assert isinstance(cfg.channels, tuple)


def test_window_config_falls_back_to_defaults_when_a_block_is_absent():
    """A manifest predating a block means 'default', not 'crash'."""
    from pri2m.xai import sampling

    d = WindowConfig()
    for pre in (None, {}, {"channels": None}, {"window": {}}):
        cfg = sampling.window_config_from_preprocessing(pre)
        assert (cfg.length, cfg.stride, cfg.channels) == (d.length, d.stride, d.channels)


def test_recovered_window_config_actually_drives_build_labeled_batch(tmp_path):
    """The real regression: a recovered cfg must work where None crashed."""
    from pri2m.xai import sampling

    npz_root, csv = _tree_and_table(tmp_path)
    pre = {"channels": list(sampling.CHANNELS), "window": {"length": 128, "stride": 64}}
    cfg = sampling.window_config_from_preprocessing(pre)
    batch, labels = sampling.build_labeled_batch(csv, npz_root, None, 6, cfg, seed=7)
    assert batch["window"].shape[0] > 0
    assert len(labels) == batch["window"].shape[0]


def test_passing_none_as_cfg_still_raises(tmp_path):
    """Pins the failure mode, so the helper is not quietly bypassed again."""
    import pytest

    from pri2m.xai import sampling

    npz_root, csv = _tree_and_table(tmp_path)
    with pytest.raises(AttributeError, match="channels"):
        sampling.build_labeled_batch(csv, npz_root, None, 6, None, seed=7)


# --------------------------------------------------------------------------- #
# Batch/model device agreement (regression, 2026-08-12)
# --------------------------------------------------------------------------- #

# build_labeled_batch always returns CPU tensors. eval_heldout_phase3.run calls
# model.to(dev) as a side effect (eval_heldout_phase3.py:170), so a runner that
# scores a held-out set BEFORE collecting activations ends up with a CUDA model
# and a CPU batch. Job 57252876 died that way 13 minutes in, on
# "Expected all tensors to be on the same device".


class _OneParam(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)


class _NoParams(torch.nn.Module):
    pass


def test_batch_tensors_land_on_the_models_device():
    from pri2m.xai import sampling

    model = _OneParam()
    want = next(model.parameters()).device
    batch = {"window": torch.randn(3, 4), "tau": torch.tensor([1, 2, 3])}
    moved = sampling.batch_to_model_device(batch, model)
    for k, v in moved.items():
        assert v.device == want, f"{k} on {v.device}, model on {want}"


def test_non_tensor_entries_pass_through_untouched():
    from pri2m.xai import sampling

    batch = {"window": torch.randn(2, 2), "fold": 2, "levels": (0.1, 0.9), "name": "x"}
    moved = sampling.batch_to_model_device(batch, _OneParam())
    assert moved["fold"] == 2
    assert moved["levels"] == (0.1, 0.9)
    assert moved["name"] == "x"


def test_a_model_with_no_parameters_is_not_an_error():
    """Nothing could have moved it, so the batch is already right."""
    from pri2m.xai import sampling

    batch = {"window": torch.randn(2, 2)}
    assert sampling.batch_to_model_device(batch, _NoParams()) == batch


def test_every_key_survives_the_move():
    """A dropped key would surface far away, as a missing side feature."""
    from pri2m.xai import sampling

    batch = {k: torch.randn(2) for k in ("window", "tau", "omega", "tpf", "fn", "fs")}
    moved = sampling.batch_to_model_device(batch, _OneParam())
    assert sorted(moved) == sorted(batch)
