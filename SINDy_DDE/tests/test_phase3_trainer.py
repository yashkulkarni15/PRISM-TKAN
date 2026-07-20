"""Unit + integration tests for the Phase 3 training harness (Task 3).

Everything runs on the shared synthetic NPZ tree + table + folds and a *tmp*
``file:`` MLflow tracking URI, so no test touches the production tracking store.
Sizes are deliberately tiny (window length 64, hidden_size 8, <=2 epochs) so the
TKAN path stays fast.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from tests._phase3_synth import (
    make_synth_folds,
    make_synth_npz_tree,
    make_synth_table,
)

from pri2m.data.phase3_sequence import WindowConfig, build_fold_loaders
from pri2m.training.losses import LossWeights
from pri2m.training.phase3_trainer import (
    Phase3Trainer,
    TrainConfig,
    _best_auc_epoch,
    build_model,
    build_optimizer,
    maybe_compile_cells,
    resolve_device,
    schedule_factor,
    side_feature_stats,
    strip_compile_prefix,
    train_one_fold,
)
from pri2m.models.heads import Phase3Model
from pri2m.models.tkan_model import DEFAULT_SIDE_FEATURES, TKANModel


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synth_inputs(tmp_path_factory):
    """A small synthetic tree + table + folds shared across the module."""
    root = tmp_path_factory.mktemp("synth_npz")
    npz_root = make_synth_npz_tree(root / "npz", length=256, seed=0)
    table = make_synth_table(npz_root)
    folds = make_synth_folds(table, n_splits=3, seed=0)
    return npz_root, table, folds


def _tiny_cfg(**overrides) -> TrainConfig:
    """A fast config: 64-length windows, hidden 8, 2 epochs, smoke on."""
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
# build_model
# --------------------------------------------------------------------------- #
def test_build_model_mlp_is_phase3model():
    model = build_model(_tiny_cfg(model="mlp"))
    assert isinstance(model, Phase3Model)
    assert not getattr(model, "with_trajectory", False)


def test_build_model_tkan_is_phase3model():
    model = build_model(_tiny_cfg(model="tkan"))
    assert isinstance(model, Phase3Model)


def test_build_model_sigkan_maps_to_tkan():
    model = build_model(_tiny_cfg(model="sigkan"))
    assert isinstance(model, Phase3Model)


def test_build_model_unknown_raises():
    with pytest.raises((ValueError, KeyError)):
        build_model(_tiny_cfg(model="not-a-model"))


def test_build_model_passes_pool_to_tkan():
    model = build_model(_tiny_cfg(model="tkan", pool=4))
    assert getattr(model, "pool", 1) == 4


# --------------------------------------------------------------------------- #
# torch.compile speedup (metric-safe; cuda-only) + checkpoint-key hygiene
# --------------------------------------------------------------------------- #
def test_train_config_compile_defaults_false():
    assert TrainConfig().compile is False


def test_maybe_compile_is_noop_on_cpu_even_when_enabled():
    # torch.compile on the recurrent cell is a CUDA-only optimization; on CPU it
    # must be a no-op (CPU inductor is unusably slow here), so the cells stay
    # plain TKANCell and nothing in the module tree is renamed.
    model = build_model(_tiny_cfg(model="tkan", compile=True))
    out = maybe_compile_cells(model, _tiny_cfg(model="tkan", compile=True), torch.device("cpu"))
    assert out is model
    assert all(
        type(cell).__name__ == "TKANCell" for cell in model.encoder.layers
    )


def test_strip_compile_prefix_removes_orig_mod():
    sd = {
        "encoder.layers.0._orig_mod.kan_z.base_weight": torch.zeros(1),
        "severity.reg_head.base_weight": torch.zeros(1),
    }
    cleaned = strip_compile_prefix(sd)
    assert set(cleaned) == {
        "encoder.layers.0.kan_z.base_weight",
        "severity.reg_head.base_weight",
    }


def test_compiled_checkpoint_loads_into_plain_model():
    # A checkpoint saved from a compiled model must load into a fresh, UNCOMPILED
    # TKANModel (the XAI / inference path). Simulate compile's key mangling by
    # inserting the _orig_mod. prefix torch.compile adds, then assert the stripped
    # state_dict round-trips with no missing/unexpected keys.
    model = build_model(_tiny_cfg(model="tkan"))
    clean = model.state_dict()
    mangled = {
        k.replace("encoder.layers.0.", "encoder.layers.0._orig_mod."): v
        for k, v in clean.items()
    }
    assert any("_orig_mod." in k for k in mangled)  # the simulation actually mangled keys

    fresh = build_model(_tiny_cfg(model="tkan"))
    missing, unexpected = fresh.load_state_dict(strip_compile_prefix(mangled), strict=True)
    assert not missing and not unexpected


# --------------------------------------------------------------------------- #
# AUC-best checkpoint (keep-both): best.pt = MAE-best, best_auc.pt = AUC-best.
# The headline metric vs the MLP is AUC, but MAE plateaus while AUC still climbs,
# so the MAE-best epoch is not the best classifier -- keep both so neither is lost.
# --------------------------------------------------------------------------- #
def test_best_auc_epoch_selects_max_finite_auc():
    hist = [{"auc": 0.6, "mae": 1.0}, {"auc": 0.81, "mae": 0.9}, {"auc": 0.7, "mae": 0.8}]
    idx, metrics = _best_auc_epoch(hist, fallback={})
    assert idx == 1
    assert metrics["auc"] == 0.81


def test_best_auc_epoch_nan_safe_falls_back():
    hist = [{"auc": float("nan")}, {"auc": float("nan")}]
    idx, metrics = _best_auc_epoch(hist, fallback={"sentinel": 1})
    assert idx == -1
    assert metrics == {"sentinel": 1}


def test_train_one_fold_writes_and_retains_best_auc(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="tkan", epochs=3, smoke=True)
    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs",
        tracking_uri=f"file:{tmp_path}/mlruns",
    )
    ckpt_dir = Path(summary["checkpoints_dir"])
    names = {p.name for p in ckpt_dir.glob("*.pt")}
    assert "best_auc.pt" in names  # the AUC-best checkpoint is written
    assert "best.pt" in names      # the MAE-best is still kept too
    # retention keeps it (name contains "best") rather than pruning it
    assert any(k.endswith("best_auc.pt") for k in summary["retention"]["keep"])


def test_train_one_fold_summary_reports_best_auc(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="tkan", epochs=3, smoke=True)
    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs",
        tracking_uri=f"file:{tmp_path}/mlruns",
    )
    assert "best_auc_epoch" in summary
    assert "best_auc_val" in summary


def test_build_model_pool_defaults_to_one():
    model = build_model(_tiny_cfg(model="tkan"))
    assert getattr(model, "pool", None) == 1


# --------------------------------------------------------------------------- #
# Representative test 1 - smoke run returns a summary; train loss decreases
# --------------------------------------------------------------------------- #
def test_train_one_fold_mlp_smoke_returns_summary(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", epochs=3, smoke=True)
    summary = train_one_fold(
        cfg,
        table,
        folds,
        npz_root,
        output_dir=tmp_path / "runs",
        tracking_uri=f"file:{tmp_path}/mlruns",
    )
    assert "val" in summary and "mae" in summary["val"]
    assert "gate" in summary
    assert "run_id" in summary
    assert "history_tail" in summary

    history = summary["history"]
    train_loss = history["train_loss"]
    assert len(train_loss) >= 2
    # Smoke clamps epochs to <= 2 even though we asked for 3.
    assert len(train_loss) <= 2
    assert train_loss[-1] < train_loss[0]


# --------------------------------------------------------------------------- #
# Representative test 2 - determinism
# --------------------------------------------------------------------------- #
def test_train_one_fold_deterministic_val_mae(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", epochs=2, seed=7)
    s1 = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "run_a", tracking_uri=f"file:{tmp_path}/mlruns"
    )
    s2 = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "run_b", tracking_uri=f"file:{tmp_path}/mlruns"
    )
    assert s1["val"]["mae"] == pytest.approx(s2["val"]["mae"], abs=1e-6)


# --------------------------------------------------------------------------- #
# Representative test 3 - MLflow run + final.pt checkpoint
# --------------------------------------------------------------------------- #
def test_train_one_fold_logs_mlflow_and_checkpoint(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", epochs=2)
    out_dir = tmp_path / "runs"
    uri = f"file:{tmp_path}/mlruns"
    summary = train_one_fold(cfg, table, folds, npz_root, out_dir, tracking_uri=uri)

    # MLflow store materialized under the tmp URI with at least one run.
    mlruns = tmp_path / "mlruns"
    assert mlruns.exists()

    import mlflow

    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(summary["run_id"])
    assert run.data.params  # some params were logged
    assert "model" in run.data.params
    assert run.data.metrics  # some metrics were logged
    assert any("mae" in k for k in run.data.metrics)

    # A final.pt checkpoint exists somewhere under the output dir.
    finals = list(Path(out_dir).rglob("final.pt"))
    assert finals, "expected a final.pt checkpoint under the output dir"


# --------------------------------------------------------------------------- #
# Representative test 4 - retention pruned periodic checkpoints
# --------------------------------------------------------------------------- #
def test_train_one_fold_retention_prunes_periodic(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", epochs=2)
    out_dir = tmp_path / "runs"
    summary = train_one_fold(
        cfg, table, folds, npz_root, out_dir, tracking_uri=f"file:{tmp_path}/mlruns"
    )

    ckpt_dir = Path(summary["checkpoints_dir"])
    names = {p.name for p in ckpt_dir.glob("*.pt")}
    assert "best.pt" in names
    assert "final.pt" in names
    # Periodic per-epoch checkpoints are gone from disk.
    assert not any(n.startswith("epoch_") for n in names)

    # And the summary records the same outcome.
    kept = {Path(p).name for p in summary["retention"]["keep"]}
    deleted = {Path(p).name for p in summary["retention"]["delete"]}
    assert {"best.pt", "final.pt"} <= kept
    assert all(n.startswith("epoch_") for n in deleted) and deleted


# --------------------------------------------------------------------------- #
# Representative test 5 - tkan smoke completes
# --------------------------------------------------------------------------- #
def test_train_one_fold_tkan_smoke_completes(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="tkan", epochs=2, hidden_size=8, smoke=True)
    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs", tracking_uri=f"file:{tmp_path}/mlruns"
    )
    assert "val" in summary and "mae" in summary["val"]
    assert "gate" in summary
    # Severity-only tkan: gate not enabled.
    assert summary["gate"].get("trajectory_enabled") is False


# --------------------------------------------------------------------------- #
# Representative test 6 - no leakage: train/val files disjoint by parent group
# --------------------------------------------------------------------------- #
def test_trainer_loaders_no_group_leakage(synth_inputs):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", fold=0)
    train_loader, val_loader, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, cfg.window, cfg.batch_size, seed=cfg.seed
    )
    train_groups = set(train_loader.dataset.table["parent_group"].astype(int))
    val_groups = set(val_loader.dataset.table["parent_group"].astype(int))
    assert train_groups.isdisjoint(val_groups)
    train_files = set(train_loader.dataset.table["file"])
    val_files = set(val_loader.dataset.table["file"])
    assert train_files.isdisjoint(val_files)


# --------------------------------------------------------------------------- #
# Representative test 7 - run_gate on a trajectory tkan returns gate keys
# --------------------------------------------------------------------------- #
def test_run_gate_with_trajectory_returns_gate_keys(synth_inputs):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="tkan", with_trajectory=True, traj_steps=8, hidden_size=8)
    model = build_model(cfg)
    _, val_loader, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, cfg.window, cfg.batch_size, seed=cfg.seed
    )
    trainer = Phase3Trainer(cfg)
    report = trainer.run_gate(model, val_loader)
    assert "passed" in report
    assert "min_r2" in report
    assert "onset_error_pct" in report


def test_run_gate_without_trajectory_disabled(synth_inputs):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="tkan", with_trajectory=False)
    model = build_model(cfg)
    _, val_loader, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, cfg.window, cfg.batch_size, seed=cfg.seed
    )
    trainer = Phase3Trainer(cfg)
    report = trainer.run_gate(model, val_loader)
    assert report == {"trajectory_enabled": False}


# --------------------------------------------------------------------------- #
# evaluate + fit unit behavior
# --------------------------------------------------------------------------- #
def test_evaluate_returns_metric_dict(synth_inputs):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp")
    model = build_model(cfg)
    _, val_loader, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, cfg.window, cfg.batch_size, seed=cfg.seed
    )
    trainer = Phase3Trainer(cfg)
    metrics = trainer.evaluate(model, val_loader)
    for key in ("mae", "rmse", "auc", "acc"):
        assert key in metrics
    # evaluate must leave the model in train mode for the next epoch.
    assert model.training


def test_fit_writes_checkpoints_each_epoch(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", epochs=2, smoke=False)
    model = build_model(cfg)
    train_loader, val_loader, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, cfg.window, cfg.batch_size, seed=cfg.seed
    )
    trainer = Phase3Trainer(cfg)
    run_dir = tmp_path / "run"
    history = trainer.fit(model, train_loader, val_loader, output_dir=run_dir)
    assert len(history["train_loss"]) == 2
    assert len(history["val"]) == 2
    ckpts = {p.name for p in (run_dir / "checkpoints").glob("*.pt")}
    assert "epoch_0.pt" in ckpts
    assert "epoch_1.pt" in ckpts
    assert "best.pt" in ckpts
    assert "final.pt" in ckpts


# --------------------------------------------------------------------------- #
# summary.json on disk
# --------------------------------------------------------------------------- #
def test_resolve_device_auto_and_explicit():
    import torch

    auto = resolve_device("auto")
    assert auto.type in ("cuda", "cpu")
    # auto picks cuda iff available.
    assert auto.type == ("cuda" if torch.cuda.is_available() else "cpu")
    # explicit specs pass through verbatim.
    assert resolve_device("cpu").type == "cpu"


def test_summary_reports_best_epoch_and_best_val(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", epochs=2, smoke=False)
    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs", tracking_uri=f"file:{tmp_path}/mlruns"
    )
    assert "best_val" in summary and "best_epoch" in summary
    assert set(("mae", "rmse", "auc", "acc")) <= set(summary["best_val"])
    # best_epoch indexes into the recorded val history (or -1 if all-NaN MAE).
    val_hist = summary["history"]["val"]
    assert -1 <= summary["best_epoch"] < len(val_hist)
    if summary["best_epoch"] >= 0:
        # best_val's MAE is the minimum finite MAE across epochs.
        maes = [
            m["mae"] for m in val_hist
            if isinstance(m["mae"], (int, float)) and m["mae"] == m["mae"]
        ]
        assert summary["best_val"]["mae"] == pytest.approx(min(maes), abs=1e-9)


def test_summary_json_written(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", epochs=2)
    out_dir = tmp_path / "runs"
    summary = train_one_fold(
        cfg, table, folds, npz_root, out_dir, tracking_uri=f"file:{tmp_path}/mlruns"
    )
    summary_path = Path(summary["run_dir"]) / "summary.json"
    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text())
    assert "val" in loaded and "gate" in loaded and "run_id" in loaded


# --------------------------------------------------------------------------- #
# preprocessing.json - the recoverable per-fold z-score for new/raw data
# --------------------------------------------------------------------------- #
def _exclude_reasons(table) -> dict:
    """Expected ``{reason: count}`` over supervised-excluded rows of ``table``."""
    excluded = table[table["exclude_supervised"].astype(bool)]
    return {str(k): int(v) for k, v in excluded["exclude_reason"].value_counts().items()}


def test_train_one_fold_writes_preprocessing_json(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    # Attach the production ``exclude_reason`` column the real table carries so
    # the excluded_reasons grouping is exercised end-to-end.
    table = table.copy()
    table["exclude_reason"] = [
        "metrics_unavailable" if ml == "metrics_unavailable"
        else ("invalid_blim_low_speed" if ex else "")
        for ex, ml in zip(table["exclude_supervised"], table["mode_label"])
    ]
    cfg = _tiny_cfg(model="mlp", epochs=2, fold=1, pool=1)
    out_dir = tmp_path / "runs"
    summary = train_one_fold(
        cfg, table, folds, npz_root, out_dir, tracking_uri=f"file:{tmp_path}/mlruns"
    )

    pre_path = Path(summary["run_dir"]) / "preprocessing.json"
    assert pre_path.exists(), "train_one_fold must write preprocessing.json"
    pre = json.loads(pre_path.read_text())

    win = cfg.effective_window()

    # --- schema keys present --------------------------------------------- #
    assert set(pre) == {
        "norm", "channels", "window", "pool", "fold",
        "counts", "leakage_verified", "seed",
    }
    assert set(pre["norm"]) == {"mean", "std"}
    assert pre["channels"] == list(win.channels)
    assert pre["window"] == {"length": win.length, "stride": win.stride}
    assert pre["pool"] == cfg.pool
    assert pre["fold"] == cfg.fold
    assert pre["leakage_verified"] is True
    assert pre["seed"] == cfg.seed

    # --- norm equals the fold's NormStats fit on TRAIN windows only -------- #
    _, _, norm = build_fold_loaders(
        table, folds, cfg.fold, npz_root, win, cfg.batch_size, seed=cfg.seed
    )
    assert len(pre["norm"]["mean"]) == len(win.channels)
    assert pre["norm"]["mean"] == pytest.approx(norm.to_dict()["mean"], abs=1e-9)
    assert pre["norm"]["std"] == pytest.approx(norm.to_dict()["std"], abs=1e-9)

    # --- counts match the loaders + the input table ----------------------- #
    train_loader, val_loader, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, win, cfg.batch_size, seed=cfg.seed
    )
    c = pre["counts"]
    assert c["n_train_windows"] == len(train_loader.dataset)
    assert c["n_val_windows"] == len(val_loader.dataset)
    assert c["n_train_files"] == len(train_loader.dataset._files)
    assert c["n_val_files"] == len(val_loader.dataset._files)
    n_excluded = int(table["exclude_supervised"].astype(bool).sum())
    assert c["n_excluded"] == n_excluded
    assert c["excluded_reasons"] == _exclude_reasons(table)
    # sanity: the fixture really injects excluded rows with named reasons.
    assert n_excluded > 0
    assert c["excluded_reasons"]


def test_summary_embeds_preprocessing_block(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", epochs=2, fold=1)
    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs",
        tracking_uri=f"file:{tmp_path}/mlruns",
    )
    assert "preprocessing" in summary
    pre = summary["preprocessing"]
    win = cfg.effective_window()
    assert set(pre["norm"]) == {"mean", "std"}
    assert pre["channels"] == list(win.channels)
    assert pre["window"] == {"length": win.length, "stride": win.stride}
    assert pre["pool"] == cfg.pool
    assert pre["fold"] == cfg.fold

    # The embedded norm matches the fold's TRAIN-only NormStats.
    _, _, norm = build_fold_loaders(
        table, folds, cfg.fold, npz_root, win, cfg.batch_size, seed=cfg.seed
    )
    assert pre["norm"]["mean"] == pytest.approx(norm.to_dict()["mean"], abs=1e-9)
    assert pre["norm"]["std"] == pytest.approx(norm.to_dict()["std"], abs=1e-9)


def test_preprocessing_logged_as_mlflow_artifact(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", epochs=2, fold=1)
    uri = f"file:{tmp_path}/mlruns"
    summary = train_one_fold(cfg, table, folds, npz_root, tmp_path / "runs", tracking_uri=uri)

    import mlflow

    mlflow.set_tracking_uri(uri)
    artifacts = {a.path for a in mlflow.artifacts.list_artifacts(run_id=summary["run_id"])}
    assert "preprocessing.json" in artifacts


# --------------------------------------------------------------------------- #
# Optimizer stability bundle (fold-0 instability fix): schedule_factor /
# build_optimizer / grad clipping / scheduler wiring -- all flag-gated with
# backward-compatible defaults.
# --------------------------------------------------------------------------- #
def test_train_config_stability_defaults_backward_compat():
    # Defaults must reproduce the pre-bundle behavior exactly: plain Adam,
    # constant lr, no warmup, no clipping, no weight decay.
    cfg = TrainConfig()
    assert cfg.lr_schedule == "constant"
    assert cfg.warmup_epochs == 0
    assert cfg.grad_clip == 0.0
    assert cfg.optimizer == "adam"
    assert cfg.weight_decay == 0.0


# -- schedule_factor --------------------------------------------------------- #
def test_schedule_factor_warmup_ramp():
    # Linear ramp over warmup: epoch 0 of warmup 2 -> 1/2, epoch 1 -> 2/2.
    assert schedule_factor(0, 20, "cosine", 2) == pytest.approx(0.5)
    assert schedule_factor(1, 20, "cosine", 2) == pytest.approx(1.0)


def test_schedule_factor_cosine_starts_at_one_after_warmup():
    # First post-warmup epoch is cos(0) == full factor (no lr cliff).
    assert schedule_factor(2, 20, "cosine", 2) == pytest.approx(1.0)


def test_schedule_factor_cosine_monotone_decreasing_after_warmup():
    factors = [schedule_factor(e, 20, "cosine", 2) for e in range(2, 20)]
    assert all(a >= b for a, b in zip(factors, factors[1:]))
    assert factors[-1] < factors[0]


def test_schedule_factor_cosine_near_zero_at_end():
    # epoch 19 of 20 (warmup 2): progress 17/18 -> factor ~0.008.
    assert schedule_factor(19, 20, "cosine", 2) < 0.01


def test_schedule_factor_constant_is_one_after_warmup():
    assert schedule_factor(0, 20, "constant", 0) == 1.0
    assert schedule_factor(5, 20, "constant", 2) == 1.0
    assert schedule_factor(19, 20, "constant", 2) == 1.0


def test_schedule_factor_unknown_schedule_raises():
    with pytest.raises(ValueError, match="constant"):
        schedule_factor(0, 20, "step", 0)


# -- build_optimizer ---------------------------------------------------------- #
def test_build_optimizer_adam_lr_and_weight_decay():
    cfg = _tiny_cfg(lr=3e-3, optimizer="adam", weight_decay=0.0)
    opt = build_optimizer(build_model(cfg).parameters(), cfg)
    assert type(opt) is torch.optim.Adam
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-3)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.0)


def test_build_optimizer_adamw_lr_and_weight_decay():
    cfg = _tiny_cfg(lr=2e-3, optimizer="adamw", weight_decay=1e-4)
    opt = build_optimizer(build_model(cfg).parameters(), cfg)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-3)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(1e-4)


def test_build_optimizer_unknown_raises():
    cfg = _tiny_cfg(optimizer="sgd")
    with pytest.raises(ValueError, match="adam"):
        build_optimizer(build_model(cfg).parameters(), cfg)


# -- gradient clipping in fit ------------------------------------------------- #
def _fit_with_clip_recorder(synth_inputs, monkeypatch, grad_clip: float) -> list:
    """Run a 1-epoch smoke fit with clip_grad_norm_ instrumented; return max_norms."""
    npz_root, table, folds = synth_inputs
    calls: list[float] = []
    orig = torch.nn.utils.clip_grad_norm_

    def recording(parameters, max_norm, *args, **kwargs):
        calls.append(float(max_norm))
        return orig(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording)

    cfg = _tiny_cfg(model="mlp", epochs=1, grad_clip=grad_clip)
    model = build_model(cfg)
    train_loader, val_loader, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, cfg.window, cfg.batch_size, seed=cfg.seed
    )
    Phase3Trainer(cfg).fit(model, train_loader, val_loader)
    return calls, len(train_loader)


def test_fit_applies_grad_clip_when_enabled(synth_inputs, monkeypatch):
    calls, n_batches = _fit_with_clip_recorder(synth_inputs, monkeypatch, grad_clip=1.0)
    # Clipped once per optimizer step (every batch) at the configured max-norm.
    assert len(calls) == n_batches
    assert all(c == 1.0 for c in calls)


def test_fit_skips_grad_clip_by_default(synth_inputs, monkeypatch):
    calls, _ = _fit_with_clip_recorder(synth_inputs, monkeypatch, grad_clip=0.0)
    assert calls == []


# -- lr scheduler stepped by fit ---------------------------------------------- #
def test_fit_steps_lr_scheduler(synth_inputs, monkeypatch):
    npz_root, table, folds = synth_inputs
    records: list[tuple[int, float]] = []
    orig_cls = torch.optim.lr_scheduler.LambdaLR

    class RecordingLambdaLR(orig_cls):
        def step(self, *args, **kwargs):
            out = super().step(*args, **kwargs)
            records.append((int(self.last_epoch), float(self.optimizer.param_groups[0]["lr"])))
            return out

    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", RecordingLambdaLR)

    cfg = _tiny_cfg(model="mlp", epochs=2, lr=1e-2, lr_schedule="cosine", warmup_epochs=0)
    model = build_model(cfg)
    train_loader, val_loader, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, cfg.window, cfg.batch_size, seed=cfg.seed
    )
    Phase3Trainer(cfg).fit(model, train_loader, val_loader)

    # fit() steps the scheduler once at the END of each epoch (last_epoch >= 1;
    # last_epoch == 0 is LambdaLR's own construction-time step).
    fit_steps = {e: lr for e, lr in records if e >= 1}
    assert set(fit_steps) == {1, 2}
    # Cosine over 2 epochs (no warmup): epoch 1 trains at factor(1) == 0.5,
    # and after the final step the factor reaches factor(2) == 0.0.
    assert fit_steps[1] == pytest.approx(5e-3)
    assert fit_steps[2] == pytest.approx(0.0, abs=1e-12)


def test_train_one_fold_logs_lr_metric_and_stability_params(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(
        model="mlp", epochs=2, lr=1e-2,
        lr_schedule="cosine", warmup_epochs=0,
        grad_clip=1.0, optimizer="adamw", weight_decay=1e-4,
    )
    uri = f"file:{tmp_path}/mlruns"
    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs", tracking_uri=uri
    )

    import mlflow

    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(summary["run_id"])
    # New hyperparams are logged as MLflow params (stringified by mlflow).
    assert run.data.params["lr_schedule"] == "cosine"
    assert run.data.params["warmup_epochs"] == "0"
    assert float(run.data.params["grad_clip"]) == pytest.approx(1.0)
    assert run.data.params["optimizer"] == "adamw"
    assert float(run.data.params["weight_decay"]) == pytest.approx(1e-4)

    # The lr actually used each epoch is logged as the "lr" metric (read before
    # scheduler.step()): cosine over 2 epochs -> [lr, 0.5*lr].
    client = mlflow.tracking.MlflowClient()
    hist = sorted(client.get_metric_history(summary["run_id"], "lr"), key=lambda m: m.step)
    assert [m.step for m in hist] == [0, 1]
    assert hist[0].value == pytest.approx(1e-2)
    assert hist[1].value == pytest.approx(5e-3)


# -- CLI merge of the new fields (scripts/train_phase3.py) -------------------- #
def _load_train_script_module():
    """Import scripts/train_phase3.py as a module (its sys.path shim makes
    ``pri2m`` importable; the E2E test only exercises it via subprocess)."""
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "train_phase3.py"
    spec = importlib.util.spec_from_file_location("_train_phase3_for_tests", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_args(**overrides):
    """Namespace mirroring main()'s parser with everything left at 'unset'."""
    import argparse

    base = dict(
        model=None, fold=None, device=None, num_workers=None, compile=None,
        smoke=False, lr_schedule=None, warmup_epochs=None, grad_clip=None,
        optimizer=None, weight_decay=None, side_input=None, side_hidden=None,
        severity_mode=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_train_config_stability_fields_from_yaml():
    mod = _load_train_script_module()
    yaml_dict = {
        "lr_schedule": "cosine",
        "warmup_epochs": 2,
        "grad_clip": 1.0,
        "optimizer": "adamw",
        "weight_decay": 1.0e-4,
    }
    cfg = mod._build_train_config(yaml_dict, _cli_args())
    assert cfg.lr_schedule == "cosine"
    assert cfg.warmup_epochs == 2
    assert cfg.grad_clip == pytest.approx(1.0)
    assert cfg.optimizer == "adamw"
    assert cfg.weight_decay == pytest.approx(1e-4)


def test_build_train_config_cli_overrides_yaml_stability_fields():
    mod = _load_train_script_module()
    yaml_dict = {
        "lr_schedule": "cosine",
        "warmup_epochs": 2,
        "grad_clip": 1.0,
        "optimizer": "adamw",
        "weight_decay": 1.0e-4,
    }
    args = _cli_args(
        lr_schedule="constant", warmup_epochs=0, grad_clip=0.5,
        optimizer="adam", weight_decay=0.0,
    )
    cfg = mod._build_train_config(yaml_dict, args)
    assert cfg.lr_schedule == "constant"
    assert cfg.warmup_epochs == 0
    assert cfg.grad_clip == pytest.approx(0.5)
    assert cfg.optimizer == "adam"
    assert cfg.weight_decay == 0.0


def test_build_train_config_stability_fields_default_when_unset():
    mod = _load_train_script_module()
    cfg = mod._build_train_config({}, _cli_args())
    assert cfg.lr_schedule == "constant"
    assert cfg.warmup_epochs == 0
    assert cfg.grad_clip == 0.0
    assert cfg.optimizer == "adam"
    assert cfg.weight_decay == 0.0


# --------------------------------------------------------------------------- #
# Gao-style process-parameter side input (omega/tpf/tau): config -> model ->
# train-only stats -> preprocessing.json. All flag-gated; defaults keep the
# pre-side-input behavior byte-identical.
# --------------------------------------------------------------------------- #
def test_train_config_side_defaults():
    cfg = TrainConfig()
    assert cfg.side_input is False
    assert cfg.side_hidden == 16


def test_build_model_tkan_side_input_enabled():
    model = build_model(_tiny_cfg(model="tkan", side_input=True))
    assert model.side_input is True
    assert "side_mean" in model.state_dict()
    assert "side_std" in model.state_dict()


def test_build_model_tkan_side_input_off_keeps_plain_state_dict():
    model = build_model(_tiny_cfg(model="tkan"))
    assert "side_mean" not in model.state_dict()


def test_build_model_mlp_side_input_raises():
    with pytest.raises(ValueError, match="tkan"):
        build_model(_tiny_cfg(model="mlp", side_input=True))


# -- side_feature_stats (window-weighted, TRAIN-only) -------------------------- #
def _fake_side_dataset(*recs):
    """A stand-in dataset exposing only the ``_files`` records the helper reads."""
    return types.SimpleNamespace(_files=list(recs))


def test_side_feature_stats_window_weighted_exact():
    # File A: 3 windows, file B: 1 window -> weights 3 and 1 (the per-window
    # distribution the model actually sees).
    ds = _fake_side_dataset(
        {"omega": 2000.0, "tpf": 200.0, "tau": 50, "windows": np.zeros((3, 8, 2))},
        {"omega": 4000.0, "tpf": 200.0, "tau": 10, "windows": np.zeros((1, 8, 2))},
    )
    mean, std = side_feature_stats(ds, ("omega", "tpf", "tau"))
    assert mean.dtype == np.float64 and std.dtype == np.float64
    # omega: (3*2000 + 1*4000)/4 = 2500; pop var = (3*500^2 + 1500^2)/4 = 750000.
    assert mean[0] == pytest.approx(2500.0)
    assert std[0] == pytest.approx(np.sqrt(750000.0))
    # tpf is constant -> mean 200, population std 0 floored at 1e-8.
    assert mean[1] == pytest.approx(200.0)
    assert std[1] == pytest.approx(1e-8)
    # tau: (3*50 + 10)/4 = 40; pop var = (3*10^2 + 30^2)/4 = 300.
    assert mean[2] == pytest.approx(40.0)
    assert std[2] == pytest.approx(np.sqrt(300.0))


def test_side_feature_stats_constant_feature_std_floored():
    ds = _fake_side_dataset(
        {"omega": 1000.0, "windows": np.zeros((2, 4, 1))},
        {"omega": 1000.0, "windows": np.zeros((5, 4, 1))},
    )
    mean, std = side_feature_stats(ds, ("omega",))
    assert mean[0] == pytest.approx(1000.0)
    assert std[0] == pytest.approx(1e-8)


def test_side_feature_stats_empty_dataset_raises():
    with pytest.raises(ValueError, match="no files"):
        side_feature_stats(_fake_side_dataset(), DEFAULT_SIDE_FEATURES)


# -- train_one_fold wiring ------------------------------------------------------ #
def test_train_one_fold_side_input_smoke(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="tkan", epochs=2, hidden_size=8, side_input=True, smoke=True)
    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs",
        tracking_uri=f"file:{tmp_path}/mlruns",
    )

    # preprocessing carries the side block with the train-fold stats.
    side = summary["preprocessing"].get("side")
    assert side is not None, "side_input=True must add a 'side' preprocessing block"
    assert side["features"] == ["omega", "tpf", "tau"]
    assert len(side["mean"]) == 3 and len(side["std"]) == 3
    assert all(np.isfinite(v) for v in side["mean"] + side["std"])

    # best.pt loads cleanly into a fresh side-input TKANModel...
    ckpt = torch.load(Path(summary["checkpoints_dir"]) / "best.pt", weights_only=True)
    fresh = TKANModel(input_size=4, hidden_size=8, side_input=True)
    missing, unexpected = fresh.load_state_dict(ckpt, strict=True)
    assert not missing and not unexpected
    # ...and the buffered std is not the all-ones init: stats were applied
    # before fit (the synth omega varies across files, so std[omega] > 1e-8).
    assert not torch.allclose(fresh.side_std, torch.ones_like(fresh.side_std))


def test_train_one_fold_side_key_absent_when_off(synth_inputs, tmp_path):
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="tkan", epochs=2, hidden_size=8, smoke=True)
    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs",
        tracking_uri=f"file:{tmp_path}/mlruns",
    )
    # Downstream XAI detects side checkpoints by the key's PRESENCE, so the
    # default run must not carry it at all (not even as null).
    assert "side" not in summary["preprocessing"]
    pre = json.loads((Path(summary["run_dir"]) / "preprocessing.json").read_text())
    assert "side" not in pre


def test_run_gate_with_trajectory_and_side_input(synth_inputs):
    # The gate feeds the model single-sample batches; with side_input on, those
    # batches must still carry omega/tpf so forward() does not raise.
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(
        model="tkan", with_trajectory=True, traj_steps=8, hidden_size=8, side_input=True
    )
    model = build_model(cfg)
    _, val_loader, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, cfg.window, cfg.batch_size, seed=cfg.seed
    )
    report = Phase3Trainer(cfg).run_gate(model, val_loader)
    assert report.get("trajectory_enabled") is True
    assert "passed" in report


# -- CLI merge of the side fields (scripts/train_phase3.py) -------------------- #
def test_build_train_config_side_fields_from_yaml():
    mod = _load_train_script_module()
    cfg = mod._build_train_config({"side_input": True, "side_hidden": 32}, _cli_args())
    assert cfg.side_input is True
    assert cfg.side_hidden == 32


def test_build_train_config_cli_overrides_yaml_side_fields():
    mod = _load_train_script_module()
    yaml_dict = {"side_input": True, "side_hidden": 32}
    cfg = mod._build_train_config(yaml_dict, _cli_args(side_input=False, side_hidden=8))
    assert cfg.side_input is False
    assert cfg.side_hidden == 8

    cfg_on = mod._build_train_config({"side_input": False}, _cli_args(side_input=True))
    assert cfg_on.side_input is True


def test_build_train_config_side_fields_default_when_unset():
    mod = _load_train_script_module()
    cfg = mod._build_train_config({}, _cli_args())
    assert cfg.side_input is False
    assert cfg.side_hidden == 16


def test_build_train_config_yaml_side_input_dropped_for_mlp():
    # The shared YAML enables side_input for the tkan run, but the MLP baseline
    # is the side-free control: a YAML-only side_input must be dropped for mlp
    # (an explicit --side-input still reaches build_model, which raises).
    mod = _load_train_script_module()
    yaml_dict = {"model": "tkan", "side_input": True}
    cfg = mod._build_train_config(yaml_dict, _cli_args(model="mlp"))
    assert cfg.side_input is False
    # And the explicit CLI flag is honored (so build_model can enforce the guard).
    cfg_explicit = mod._build_train_config(yaml_dict, _cli_args(model="mlp", side_input=True))
    assert cfg_explicit.side_input is True


# --------------------------------------------------------------------------- #
# Quantile severity head (pinball loss) reachable from TrainConfig / the CLI.
# The head + loss already exist (tests/test_severity_quantile.py); these tests
# pin the config -> build_model -> preprocessing.json -> checkpoint-reload wiring.
# Defaults ("huber") must keep the validated point-regression path untouched.
# --------------------------------------------------------------------------- #
def _severity_batch(B=4, L=64, C=4, seed=0):
    """A tiny batch matching ``_tiny_cfg``'s window geometry (64 x 4 channels)."""
    g = torch.Generator().manual_seed(seed)
    window = torch.randn(B, L, C, generator=g)
    target = window.abs().mean(dim=(1, 2))
    return {
        "window": window,
        "tau": torch.full((B,), 4, dtype=torch.long),
        "b_over_blim": target,
        "label": (target > target.median()).float(),
        "sample_weight": torch.ones(B),
    }


def test_train_config_severity_defaults():
    cfg = TrainConfig()
    assert cfg.severity_mode == "huber"
    assert tuple(cfg.severity_quantiles) == (0.1, 0.5, 0.9)


def test_build_model_tkan_quantile_mode_emits_quantile_columns():
    cfg = _tiny_cfg(
        model="tkan", hidden_size=8,
        severity_mode="quantile", severity_quantiles=(0.1, 0.5, 0.9),
    )
    model = build_model(cfg)
    assert model.severity_mode == "quantile"
    out = model(_severity_batch())
    assert out["b_over_blim_quantiles"].shape == (4, 3)
    assert tuple(out["quantile_levels"]) == (0.1, 0.5, 0.9)


def test_build_model_mlp_quantile_raises():
    # Mirrors the side_input guard: the MLP baseline has a point head only, so
    # asking it for quantiles must fail loudly instead of silently regressing.
    with pytest.raises(ValueError, match="tkan"):
        build_model(_tiny_cfg(model="mlp", severity_mode="quantile"))


def test_build_model_unknown_severity_mode_raises():
    with pytest.raises(ValueError, match="severity_mode"):
        build_model(_tiny_cfg(model="tkan", severity_mode="banana"))
    # ...and the MLP path rejects it too (it never reaches SeverityHead).
    with pytest.raises(ValueError, match="severity_mode"):
        build_model(_tiny_cfg(model="mlp", severity_mode="banana"))


def _spy_loss_parts(monkeypatch):
    """Record the loss-part keys of every batch the REAL training path scores.

    Wraps (does not replace) ``phase3_trainer.total_loss`` so the model, the
    loaders and the optimization are entirely real; only the parts dict is
    observed. Returns the list the spy appends to.
    """
    from pri2m.training import phase3_trainer as mod

    seen: list[set] = []
    real_total_loss = mod.total_loss

    def spy(out, batch, lw):
        loss, parts = real_total_loss(out, batch, lw)
        seen.append(set(parts))
        return loss, parts

    monkeypatch.setattr(mod, "total_loss", spy)
    return seen


def test_build_model_default_severity_is_point_estimate():
    # The validated default: a bare TrainConfig yields the point head, with no
    # quantile keys anywhere in the output dict.
    model = build_model(_tiny_cfg(model="tkan", hidden_size=8))
    assert model.severity_mode == "huber"
    out = model(_severity_batch())
    assert set(out) == {"b_over_blim", "chatter_logit"}
    assert out["b_over_blim"].shape == (4,)


def test_train_one_fold_default_uses_mse_loss_part(synth_inputs, tmp_path, monkeypatch):
    # Regression guard: the default (huber) path must keep scoring every batch
    # with the weighted-MSE severity term, exactly as the validated runs did.
    npz_root, table, folds = synth_inputs
    seen = _spy_loss_parts(monkeypatch)
    cfg = _tiny_cfg(model="tkan", epochs=2, hidden_size=8, smoke=True)
    train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs",
        tracking_uri=f"file:{tmp_path}/mlruns",
    )
    assert seen, "no batch reached total_loss"
    assert all("sev_mse" in parts for parts in seen)
    assert not any("sev_pinball" in parts for parts in seen)


def test_train_one_fold_quantile_uses_pinball_loss_part(synth_inputs, tmp_path, monkeypatch):
    # End-to-end through the real training path: a quantile TrainConfig must
    # actually train against the pinball term (not silently fall back to MSE).
    npz_root, table, folds = synth_inputs
    seen = _spy_loss_parts(monkeypatch)
    cfg = _tiny_cfg(
        model="tkan", epochs=2, hidden_size=8, smoke=True,
        severity_mode="quantile", severity_quantiles=(0.1, 0.5, 0.9),
    )
    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs",
        tracking_uri=f"file:{tmp_path}/mlruns",
    )
    assert seen, "no batch reached total_loss"
    assert all("sev_pinball" in parts for parts in seen)
    assert not any("sev_mse" in parts for parts in seen)
    # The run still produces the usual per-file severity metrics (evaluate reads
    # the median column as the point estimate).
    assert np.isfinite(summary["val"]["mae"])


def test_train_one_fold_quantile_records_severity_in_preprocessing(synth_inputs, tmp_path):
    # A quantile checkpoint has a WIDER severity head, so the mode + levels must
    # be recorded or the checkpoint cannot be rebuilt downstream (XAI/eval).
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(
        model="tkan", epochs=2, hidden_size=8, smoke=True,
        severity_mode="quantile", severity_quantiles=(0.1, 0.5, 0.9),
    )
    uri = f"file:{tmp_path}/mlruns"
    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs", tracking_uri=uri
    )

    sev = summary["preprocessing"].get("severity")
    assert sev is not None, "quantile runs must record a 'severity' block"
    assert sev["mode"] == "quantile"
    assert sev["quantiles"] == pytest.approx([0.1, 0.5, 0.9])
    # ...and it is on disk, not just in the returned dict.
    pre = json.loads((Path(summary["run_dir"]) / "preprocessing.json").read_text())
    assert pre["severity"] == sev

    import mlflow

    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(summary["run_id"])
    assert run.data.params["severity_mode"] == "quantile"


def test_train_one_fold_default_severity_key_absent(synth_inputs, tmp_path):
    # The default (huber) preprocessing.json schema must be unchanged: the key
    # is ABSENT, exactly as the side block is when side_input is off.
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="tkan", epochs=2, hidden_size=8, smoke=True)
    uri = f"file:{tmp_path}/mlruns"
    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs", tracking_uri=uri
    )
    assert "severity" not in summary["preprocessing"]
    pre = json.loads((Path(summary["run_dir"]) / "preprocessing.json").read_text())
    assert "severity" not in pre

    import mlflow

    mlflow.set_tracking_uri(uri)
    assert mlflow.get_run(summary["run_id"]).data.params["severity_mode"] == "huber"


# -- CLI merge of the severity fields (scripts/train_phase3.py) ---------------- #
def test_build_train_config_severity_mode_from_yaml():
    mod = _load_train_script_module()
    cfg = mod._build_train_config(
        {"severity_mode": "quantile", "severity_quantiles": [0.05, 0.5, 0.95]},
        _cli_args(),
    )
    assert cfg.severity_mode == "quantile"
    assert tuple(cfg.severity_quantiles) == (0.05, 0.5, 0.95)


def test_build_train_config_cli_overrides_yaml_severity_mode():
    mod = _load_train_script_module()
    yaml_dict = {"severity_mode": "huber"}
    assert mod._build_train_config(
        yaml_dict, _cli_args(severity_mode="quantile")
    ).severity_mode == "quantile"
    # ...and the other direction (CLI wins over a quantile YAML too).
    assert mod._build_train_config(
        {"severity_mode": "quantile"}, _cli_args(severity_mode="huber")
    ).severity_mode == "huber"


def test_build_train_config_severity_defaults_when_unset():
    mod = _load_train_script_module()
    cfg = mod._build_train_config({}, _cli_args())
    assert cfg.severity_mode == "huber"
    assert tuple(cfg.severity_quantiles) == (0.1, 0.5, 0.9)


def test_shipped_config_keeps_validated_huber_severity():
    # configs/phase3_tkan.yaml documents every knob. severity_mode must be
    # PRESENT (so the quantile run is a one-flag override) and set to the
    # validated "huber" so in-flight sweeps are unaffected.
    mod = _load_train_script_module()
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "phase3_tkan.yaml"
    from pri2m.config import load_config

    cfg_dict = load_config(str(cfg_path))
    assert cfg_dict["severity_mode"] == "huber"
    assert mod._build_train_config(cfg_dict, _cli_args()).severity_mode == "huber"
    # ...and a single --severity-mode override flips it without touching YAML.
    assert mod._build_train_config(
        cfg_dict, _cli_args(severity_mode="quantile")
    ).severity_mode == "quantile"


def test_train_phase3_cli_exposes_severity_mode_flag(capsys):
    # The flag must be reachable from the real parser, not just the merge helper.
    mod = _load_train_script_module()
    with pytest.raises(SystemExit):
        mod.main(["--help"])
    assert "--severity-mode" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Non-finite skip-and-count guards in fit (GL job 51641089 post-mortem).
#
# A real A40 fold-0 run died at the first evaluate() with sklearn's "Input
# contains NaN": one gradient blow-up produced a NaN grad norm, and
# clip_grad_norm_ (error_if_nonfinite=False) silently scaled EVERY grad by
# NaN -> optimizer.step() NaN-poisoned all weights -> hours of silent garbage.
# fit must skip-and-count such batches so the weights stay finite, surface the
# count in history["nonfinite_batches"] / MLflow, and fail loud if an entire
# epoch was skipped.
# --------------------------------------------------------------------------- #
def _fit_setup(synth_inputs, **cfg_overrides):
    """Fresh (cfg, model, train_loader, val_loader) for a tiny fit run."""
    npz_root, table, folds = synth_inputs
    overrides = {"model": "mlp", "epochs": 1, **cfg_overrides}
    cfg = _tiny_cfg(**overrides)
    model = build_model(cfg)
    train_loader, val_loader, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, cfg.window, cfg.batch_size, seed=cfg.seed
    )
    return cfg, model, train_loader, val_loader


def test_fit_skips_single_nonfinite_loss_batch(synth_inputs, monkeypatch, capsys):
    # One inf loss must be skipped (no backward/step), counted, and warned about;
    # the rest of the epoch trains normally and the weights stay finite.
    import pri2m.training.phase3_trainer as p3t

    orig = p3t.total_loss
    calls = {"n": 0}

    def inf_once(out, batch, lw):
        calls["n"] += 1
        if calls["n"] == 1:
            return torch.tensor(float("inf")), {}
        return orig(out, batch, lw)

    monkeypatch.setattr(p3t, "total_loss", inf_once)

    cfg, model, train_loader, val_loader = _fit_setup(synth_inputs)
    history = Phase3Trainer(cfg).fit(model, train_loader, val_loader)

    assert history["nonfinite_batches"] == [1]
    assert np.isfinite(history["train_loss"][0])
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # One loud stderr warning naming the epoch and the skip count.
    err = capsys.readouterr().err
    assert "non-finite" in err
    assert "epoch 0" in err


def test_fit_skips_step_when_clip_norm_nonfinite(synth_inputs, monkeypatch):
    # The fatal path from GL job 51641089: clip_grad_norm_ returns NaN (it has
    # already scaled every grad by NaN at that point). fit must discard the
    # poisoned grads and skip the step instead of letting Adam apply them.
    orig = torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def nan_once(parameters, max_norm, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return torch.tensor(float("nan"))
        return orig(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", nan_once)

    cfg, model, train_loader, val_loader = _fit_setup(synth_inputs, grad_clip=1.0)
    history = Phase3Trainer(cfg).fit(model, train_loader, val_loader)

    assert history["nonfinite_batches"] == [1]
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # Clipping still ran on every batch; only the poisoned one was skipped.
    assert calls["n"] == len(train_loader)


def test_fit_healthy_run_reports_zero_nonfinite_batches(synth_inputs):
    # The key is ALWAYS present (uniform summaries) and all-zero when healthy.
    cfg, model, train_loader, val_loader = _fit_setup(synth_inputs, epochs=2, smoke=False)
    history = Phase3Trainer(cfg).fit(model, train_loader, val_loader)
    assert history["nonfinite_batches"] == [0, 0]
    assert all(np.isfinite(v) for v in history["train_loss"])


def test_fit_healthy_train_loss_unchanged_by_guards(synth_inputs):
    # No behavior change for healthy steps: fit's train loss must equal a
    # hand-rolled pre-guard loop (same seed, fresh identically-initialized
    # model, fresh identically-seeded loader).
    from pri2m.training.losses import total_loss as real_total_loss
    from pri2m.training.phase3_trainer import _seed_everything

    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", epochs=1)

    torch.manual_seed(123)
    model_fit = build_model(cfg)
    train_a, val_a, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, cfg.window, cfg.batch_size, seed=cfg.seed
    )
    history = Phase3Trainer(cfg).fit(model_fit, train_a, val_a)

    torch.manual_seed(123)
    model_ref = build_model(cfg)
    train_b, _, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, cfg.window, cfg.batch_size, seed=cfg.seed
    )
    _seed_everything(cfg.seed)
    opt = torch.optim.Adam(model_ref.parameters(), lr=cfg.lr)
    model_ref.train()
    losses = []
    for batch in train_b:
        opt.zero_grad()
        out = model_ref(batch)
        loss, _ = real_total_loss(out, batch, cfg.loss)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert history["train_loss"][0] == pytest.approx(float(np.mean(losses)), abs=1e-9)


def test_fit_raises_when_every_batch_nonfinite(synth_inputs, monkeypatch):
    # Fail-loud floor: an epoch with ZERO stepped batches means the model never
    # trained -- raise instead of checkpointing garbage and burning wall time.
    import pri2m.training.phase3_trainer as p3t

    monkeypatch.setattr(
        p3t, "total_loss", lambda out, batch, lw: (torch.tensor(float("nan")), {})
    )
    cfg, model, train_loader, val_loader = _fit_setup(synth_inputs)
    with pytest.raises(RuntimeError, match="never trained"):
        Phase3Trainer(cfg).fit(model, train_loader, val_loader)


def test_train_one_fold_logs_nonfinite_batches_metric(synth_inputs, tmp_path):
    # The per-epoch skip count reaches MLflow (step == epoch) so a poisoned run
    # is visible in the tracking UI, not just in stderr.
    npz_root, table, folds = synth_inputs
    cfg = _tiny_cfg(model="mlp", epochs=2, smoke=False)
    uri = f"file:{tmp_path}/mlruns"
    summary = train_one_fold(
        cfg, table, folds, npz_root, tmp_path / "runs", tracking_uri=uri
    )

    import mlflow

    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()
    hist = sorted(
        client.get_metric_history(summary["run_id"], "nonfinite_batches"),
        key=lambda m: m.step,
    )
    assert [m.step for m in hist] == [0, 1]
    assert [m.value for m in hist] == [0.0, 0.0]
