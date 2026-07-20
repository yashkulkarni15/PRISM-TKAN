#!/usr/bin/env python3
"""Train a single Phase 3 fold from the command line.

Loads a YAML config (e.g. ``configs/phase3_tkan.yaml``), applies CLI overrides,
builds a :class:`~pri2m.training.phase3_trainer.TrainConfig`, and runs
:func:`~pri2m.training.phase3_trainer.train_one_fold` on the supplied table /
folds / NPZ tree. The fold summary is printed as JSON on stdout.

Example::

    python scripts/train_phase3.py \
        --config configs/phase3_tkan.yaml --model mlp --fold 0 \
        --npz-root results/phase3/npz \
        --table results/phase3/phase3_dataset.csv \
        --folds-json results/phase3/phase3_folds.json \
        --output-dir results/phase3/runs --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
# src/ enables ``import pri2m...``; repo root keeps parity with other scripts.
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402  (after sys.path setup)

from pri2m.config import load_config  # noqa: E402
from pri2m.data.phase3_sequence import WindowConfig  # noqa: E402
from pri2m.training.losses import LossWeights  # noqa: E402
from pri2m.training.phase3_trainer import TrainConfig, train_one_fold  # noqa: E402


def _build_train_config(cfg_dict: dict, args: argparse.Namespace) -> TrainConfig:
    """Merge the YAML config with CLI overrides into a ``TrainConfig``."""
    window_cfg = cfg_dict.get("window", {}) or {}
    loss_cfg = cfg_dict.get("loss", {}) or {}

    window = WindowConfig(
        length=int(window_cfg.get("length", WindowConfig.length)),
        stride=int(window_cfg.get("stride", WindowConfig.stride)),
    )
    loss = LossWeights(
        w_sev=float(loss_cfg.get("w_sev", 1.0)),
        w_reg=float(loss_cfg.get("w_reg", 1.0)),
        w_cls=float(loss_cfg.get("w_cls", 1.0)),
        w_traj=float(loss_cfg.get("w_traj", 0.0)),
    )

    model = args.model or cfg_dict.get("model", "tkan")
    fold = args.fold if args.fold is not None else int(cfg_dict.get("fold", 0))
    device = args.device or str(cfg_dict.get("device", "auto"))
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(cfg_dict.get("num_workers", 0))
    )
    compile_cells = (
        args.compile
        if args.compile is not None
        else bool(cfg_dict.get("compile", False))
    )
    lr_schedule = (
        args.lr_schedule
        if args.lr_schedule is not None
        else str(cfg_dict.get("lr_schedule", "constant"))
    )
    warmup_epochs = (
        args.warmup_epochs
        if args.warmup_epochs is not None
        else int(cfg_dict.get("warmup_epochs", 0))
    )
    grad_clip = (
        args.grad_clip
        if args.grad_clip is not None
        else float(cfg_dict.get("grad_clip", 0.0))
    )
    optimizer = (
        args.optimizer
        if args.optimizer is not None
        else str(cfg_dict.get("optimizer", "adam"))
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(cfg_dict.get("weight_decay", 0.0))
    )
    side_input = (
        args.side_input
        if args.side_input is not None
        else bool(cfg_dict.get("side_input", False))
    )
    # The MLP baseline is the side-free control: a YAML-enabled side_input is
    # dropped for mlp so the shared config still runs the baseline as-is. An
    # explicit --side-input flows through and build_model raises the hard error.
    if side_input and model.lower() == "mlp" and args.side_input is None:
        side_input = False
    side_hidden = (
        args.side_hidden
        if args.side_hidden is not None
        else int(cfg_dict.get("side_hidden", 16))
    )
    severity_mode = (
        args.severity_mode
        if args.severity_mode is not None
        else str(cfg_dict.get("severity_mode", "huber"))
    )
    severity_quantiles = tuple(
        float(q) for q in cfg_dict.get("severity_quantiles", (0.1, 0.5, 0.9))
    )

    return TrainConfig(
        model=model,
        epochs=int(cfg_dict.get("epochs", 20)),
        lr=float(cfg_dict.get("lr", 1e-3)),
        batch_size=int(cfg_dict.get("batch_size", 32)),
        seed=int(cfg_dict.get("seed", 0)),
        window=window,
        hidden_size=int(cfg_dict.get("hidden_size", 64)),
        num_layers=int(cfg_dict.get("num_layers", 1)),
        grid_size=int(cfg_dict.get("grid_size", 5)),
        spline_order=int(cfg_dict.get("spline_order", 3)),
        loss=loss,
        fold=fold,
        with_trajectory=bool(cfg_dict.get("with_trajectory", False)),
        traj_steps=int(cfg_dict.get("traj_steps", 16)),
        pool=int(cfg_dict.get("pool", 1)),
        smoke=bool(args.smoke),
        device=device,
        num_workers=num_workers,
        compile=compile_cells,
        lr_schedule=lr_schedule,
        warmup_epochs=warmup_epochs,
        grad_clip=grad_clip,
        optimizer=optimizer,
        weight_decay=weight_decay,
        side_input=side_input,
        side_hidden=side_hidden,
        severity_mode=severity_mode,
        severity_quantiles=severity_quantiles,
    )


def _resolve_tracking_uri(cfg_dict: dict, args: argparse.Namespace) -> str | None:
    """CLI URI > this config's mlflow block > project config phase3 URI."""
    if args.mlflow_uri:
        return args.mlflow_uri
    mlflow_block = cfg_dict.get("mlflow", {}) or {}
    if "tracking_uri" in mlflow_block:
        return str(mlflow_block["tracking_uri"])
    try:
        return str(load_config()["phase3"]["mlflow"]["tracking_uri"])
    except (FileNotFoundError, KeyError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase3_tkan.yaml", help="YAML config path.")
    parser.add_argument("--model", default=None, help="Override model: tkan|mlp|sigkan.")
    parser.add_argument("--fold", type=int, default=None, help="Fold index to train.")
    parser.add_argument("--npz-root", required=True, help="Root of the dataset_<id>/*.npz tree.")
    parser.add_argument("--table", required=True, help="Phase 3 per-file table CSV.")
    parser.add_argument("--folds-json", required=True, help="Phase 3 folds JSON.")
    parser.add_argument("--output-dir", required=True, help="Where run artifacts are written.")
    parser.add_argument("--mlflow-uri", default=None, help="Override MLflow tracking URI.")
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: auto|cuda|cpu (default auto -> CUDA if available).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes (default 0 == single-process).",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="torch.compile the TKAN cells (CUDA-only, ~3.4x, metric-safe). "
        "Default: use the config's `compile` (else off). Use --no-compile to force off.",
    )
    parser.add_argument(
        "--lr-schedule",
        default=None,
        help="LR schedule: constant|cosine (default: config's `lr_schedule`, else constant).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Linear LR warmup epochs before the schedule (default: config's "
        "`warmup_epochs`, else 0 == off).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Gradient-norm clip max-norm; <= 0 disables (default: config's "
        "`grad_clip`, else 0 == off).",
    )
    parser.add_argument(
        "--optimizer",
        default=None,
        help="Optimizer: adam|adamw (default: config's `optimizer`, else adam).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Optimizer weight decay (default: config's `weight_decay`, else 0 == off).",
    )
    parser.add_argument(
        "--side-input",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Gao-style process-param side branch (omega/tpf/tau; tkan/sigkan only). "
        "Default: use the config's `side_input` (else off; dropped for mlp). "
        "Use --no-side-input to force off.",
    )
    parser.add_argument(
        "--side-hidden",
        type=int,
        default=None,
        help="Side-branch MLP width (default: config's `side_hidden`, else 16).",
    )
    parser.add_argument(
        "--severity-mode",
        default=None,
        help="Severity head: huber|quantile (tkan/sigkan only). quantile emits "
        "one column per level in the config's `severity_quantiles` and trains "
        "the regression term with a pinball loss. Default: config's "
        "`severity_mode`, else huber.",
    )
    parser.add_argument("--smoke", action="store_true", help="Clamp epochs for a fast CI run.")
    args = parser.parse_args(argv)

    cfg_dict = load_config(args.config)
    train_cfg = _build_train_config(cfg_dict, args)
    tracking_uri = _resolve_tracking_uri(cfg_dict, args)

    table = pd.read_csv(args.table)
    with open(args.folds_json, "r", encoding="utf-8") as fh:
        folds_json = json.load(fh)

    summary = train_one_fold(
        train_cfg,
        table,
        folds_json,
        args.npz_root,
        args.output_dir,
        tracking_uri=tracking_uri,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
