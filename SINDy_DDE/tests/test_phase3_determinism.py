"""Determinism flags and the environment record in ``summary.json``.

Update 14 measured a repeatability floor of 0.0092 AUC from ten duplicate run
pairs that shared an identical config and seed: four pairs reproduced
bit-identically and six did not. The trainer seeds every RNG and then never
requests deterministic algorithms, while every run config carries
``compile: true``, so which kernels consume the seeded inputs is free to vary.

Two things are tested here:

* the opt-in flags, which must leave the **default** path byte-identical --
  every already-validated run was produced without them, and a global torch
  flag flipped by import would silently move numbers nobody asked to move; and
* the environment block, which must be written **unconditionally** and read
  back from torch rather than from the config, because the cluster accounting
  log that would otherwise answer "which node, which CUDA" is eventually
  purged. A record of what was *requested* cannot diagnose a run; only a record
  of what was *in effect* can.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tests._phase3_synth import (
    make_synth_folds,
    make_synth_npz_tree,
    make_synth_table,
)

from pri2m.data.phase3_sequence import WindowConfig
from pri2m.training.losses import LossWeights
from pri2m.training.phase3_trainer import (
    TrainConfig,
    _seed_everything,
    check_deterministic_env,
    environment_block,
    train_one_fold,
)

LEGAL_CUBLAS = ":4096:8"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _restore_global_torch_flags():
    """Save/restore the process-wide flags these tests deliberately mutate.

    ``torch.use_deterministic_algorithms`` is global state. Leaking it out of a
    test would change how every later test in the session computes, which is
    exactly the class of invisible coupling this module exists to remove.
    """
    was_det = torch.are_deterministic_algorithms_enabled()
    was_cudnn_det = torch.backends.cudnn.deterministic
    was_bench = torch.backends.cudnn.benchmark
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_det)
        torch.backends.cudnn.deterministic = was_cudnn_det
        torch.backends.cudnn.benchmark = was_bench


@pytest.fixture(scope="module")
def synth_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth_npz_determinism")
    npz_root = make_synth_npz_tree(root / "npz", length=256, seed=0)
    table = make_synth_table(npz_root)
    folds = make_synth_folds(table, n_splits=3, seed=0)
    return npz_root, table, folds


def _tiny_cfg(**overrides) -> TrainConfig:
    base = dict(
        model="mlp",
        epochs=2,
        lr=1e-2,
        batch_size=8,
        seed=0,
        window=WindowConfig(length=64, stride=64),
        hidden_size=8,
        num_layers=1,
        grid_size=5,
        spline_order=3,
        loss=LossWeights(),
        fold=0,
        smoke=True,
    )
    base.update(overrides)
    return TrainConfig(**base)


# --------------------------------------------------------------------------- #
# The flag defaults off, so nothing already validated moves
# --------------------------------------------------------------------------- #
def test_train_config_deterministic_defaults_false():
    assert TrainConfig().deterministic is False


def test_seed_everything_does_not_touch_global_flags_by_default():
    """The default path must not call the determinism APIs at all.

    Every returned run -- the 5-fold sweep, both retrains, the held-out eval --
    was produced with these flags untouched. Setting them "harmlessly" on the
    default path would silently change the arithmetic of every future run
    against that history.
    """
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

    _seed_everything(0)

    assert torch.are_deterministic_algorithms_enabled() is False
    assert torch.backends.cudnn.benchmark is True


def test_seed_everything_requests_deterministic_algorithms_when_asked(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", LEGAL_CUBLAS)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

    _seed_everything(0, deterministic=True)

    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


# --------------------------------------------------------------------------- #
# The CUBLAS guard: fail in seconds, not eight hours in
# --------------------------------------------------------------------------- #
def test_deterministic_run_without_cublas_workspace_config_refuses_to_start(monkeypatch):
    """cuBLAS reads this variable when it creates its handle, so the trainer
    cannot set it for itself once training has begun. Unset, a deterministic run
    raises from inside a matmul partway through instead -- the failure this
    guard converts into an immediate, named one."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        check_deterministic_env()


def test_deterministic_run_rejects_an_illegal_cublas_value(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":2:2")

    with pytest.raises(ValueError, match=":4096:8"):
        check_deterministic_env()


@pytest.mark.parametrize("value", [":4096:8", ":16:8"])
def test_deterministic_run_accepts_both_legal_cublas_values(monkeypatch, value):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)

    check_deterministic_env()  # must not raise


# --------------------------------------------------------------------------- #
# The environment block
# --------------------------------------------------------------------------- #
def test_environment_block_reports_versions_and_host():
    env = environment_block()

    for key in (
        "hostname", "python_version", "torch_version", "numpy_version",
        "cuda_version", "cudnn_version", "gpu_name", "gpu_count",
        "slurm_job_id", "slurm_nodelist", "cublas_workspace_config",
        "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark",
    ):
        assert key in env, f"environment block must record {key}"

    assert env["torch_version"] == torch.__version__


def test_environment_block_reads_flags_back_from_torch_not_from_config():
    """The record must say what was in effect, not what was requested.

    A config echo cannot diagnose a run: if a flag fails to take, or something
    else in the process sets one, the config still reads exactly as submitted.
    That is how the duplicate-pair divergence became unanswerable from stored
    artifacts in the first place.
    """
    torch.use_deterministic_algorithms(True)

    env = environment_block()

    assert env["deterministic_algorithms"] is True


def test_environment_block_records_slurm_identity_when_present(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "55408255")
    monkeypatch.setenv("SLURM_NODELIST", "gl1512")

    env = environment_block()

    assert env["slurm_job_id"] == "55408255"
    assert env["slurm_nodelist"] == "gl1512"


def test_summary_json_carries_the_environment_block(synth_inputs, tmp_path):
    """Written unconditionally, on the default non-deterministic path.

    Gating it on ``deterministic`` would leave exactly the runs that need
    diagnosing -- the ordinary ones -- with no record, and would make an absent
    block ambiguous between "not deterministic" and "predates this code".
    """
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", epochs=2)

    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs",
        tracking_uri=f"file:{tmp_path}/mlruns",
    )

    loaded = json.loads((Path(summary["run_dir"]) / "summary.json").read_text())
    assert "environment" in loaded, "every run must record its environment"
    assert loaded["environment"]["torch_version"] == torch.__version__
    assert loaded["environment"]["deterministic_algorithms"] is False


# --------------------------------------------------------------------------- #
# CLI merge (scripts/train_phase3.py) -- same CLI > YAML > default precedence
# as --side-input and --severity-mode
# --------------------------------------------------------------------------- #
def _load_train_script_module():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "train_phase3.py"
    spec = importlib.util.spec_from_file_location("_train_phase3_determinism", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_args(**overrides):
    import argparse

    base = dict(
        model=None, fold=None, device=None, num_workers=None, compile=None,
        smoke=False, lr_schedule=None, warmup_epochs=None, grad_clip=None,
        optimizer=None, weight_decay=None, side_input=None, side_hidden=None,
        severity_mode=None, deterministic=None, epochs=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_train_config_deterministic_defaults_off():
    mod = _load_train_script_module()

    cfg = mod._build_train_config({}, _cli_args())

    assert cfg.deterministic is False


def test_build_train_config_reads_deterministic_from_yaml():
    mod = _load_train_script_module()

    cfg = mod._build_train_config({"deterministic": True}, _cli_args())

    assert cfg.deterministic is True


def test_build_train_config_cli_overrides_yaml_deterministic():
    mod = _load_train_script_module()

    cfg = mod._build_train_config({"deterministic": False}, _cli_args(deterministic=True))

    assert cfg.deterministic is True


# -- --epochs, so the determinism repeat can be short without being a smoke run #
def test_build_train_config_epochs_defaults_to_yaml():
    mod = _load_train_script_module()

    cfg = mod._build_train_config({"epochs": 20}, _cli_args())

    assert cfg.epochs == 20


def test_build_train_config_cli_overrides_yaml_epochs():
    """The determinism repeat needs a *short* run, not a *different* run.

    ``--smoke`` clamps the window length and stride as well as the epoch count
    (``TrainConfig.effective_window``), so a smoke run exercises different
    geometry and therefore potentially different kernels -- which is precisely
    the thing under test. Overriding epochs alone keeps every other dimension
    of the validated config intact.
    """
    mod = _load_train_script_module()

    cfg = mod._build_train_config({"epochs": 20}, _cli_args(epochs=5))

    assert cfg.epochs == 5
    # Nothing else may move: same window geometry as the validated config.
    assert cfg.effective_window().length == cfg.window.length
