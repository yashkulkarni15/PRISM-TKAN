#!/usr/bin/env python3
"""Probe one Phase 3 fold's training step to locate the bottleneck.

Runs the REAL pipeline (``build_fold_loaders`` + ``build_model`` + the real
train step) on a single fold for a *capped* number of batches and reports where
wall-clock goes:

    host/data fetch  (waiting on the DataLoader)
    host->device     (moving the batch to the GPU)
    GPU fwd+bwd+step  (synced compute)

It extrapolates to a full-epoch estimate and, with ``--compile cell``, also
measures a ``torch.compile``'d recurrent cell so we can see the *metric-safe*
GPU speedup before changing the model.

This answers the only question that should decide the next optimization:
**is the epoch GPU-compute-bound (compile / pool / batch help) or
DataLoader-bound (more workers / window caching help)?**

Run on a GPU node (e.g. an spgpu A40 interactive session) -- NOT a login node::

    salloc --account=jaabell0 --partition=spgpu --gres=gpu:1 \
        --cpus-per-task=8 --mem=48G --time=00:30:00
    cd /nfs/turbo/coe-jaabell/PRI2M
    module load python3.11-anaconda && source /home/$USER/envs/prism/bin/activate
    export MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=$PWD/src
    python scripts/_probe_tkan_epoch.py --fold 0 \
        --npz-root "$PWD/phase2_signals" \
        --table results/phase3/phase3_dataset.csv \
        --folds-json results/phase3/phase3_folds.json \
        --device cuda --num-workers 4 --max-batches 40 --compile cell

Nothing is written to disk; no MLflow run is opened.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pri2m.config import load_config  # noqa: E402
from pri2m.data.phase3_sequence import WindowConfig, build_fold_loaders  # noqa: E402
from pri2m.training.losses import LossWeights, total_loss  # noqa: E402
from pri2m.training.phase3_trainer import (  # noqa: E402
    TrainConfig,
    _batch_to_device,
    build_model,
    resolve_device,
)


def _train_config(cfg_dict: dict, args: argparse.Namespace) -> TrainConfig:
    """Mirror scripts/train_phase3.py config building (epochs irrelevant here)."""
    w = cfg_dict.get("window", {}) or {}
    loss_cfg = cfg_dict.get("loss", {}) or {}
    return TrainConfig(
        model=args.model or cfg_dict.get("model", "tkan"),
        batch_size=int(cfg_dict.get("batch_size", 32)),
        lr=float(cfg_dict.get("lr", 1e-3)),
        seed=int(cfg_dict.get("seed", 0)),
        window=WindowConfig(
            length=int(w.get("length", WindowConfig.length)),
            stride=int(w.get("stride", WindowConfig.stride)),
        ),
        hidden_size=int(cfg_dict.get("hidden_size", 64)),
        num_layers=int(cfg_dict.get("num_layers", 1)),
        grid_size=int(cfg_dict.get("grid_size", 5)),
        spline_order=int(cfg_dict.get("spline_order", 3)),
        loss=LossWeights(
            w_sev=float(loss_cfg.get("w_sev", 1.0)),
            w_reg=float(loss_cfg.get("w_reg", 1.0)),
            w_cls=float(loss_cfg.get("w_cls", 1.0)),
            w_traj=float(loss_cfg.get("w_traj", 0.0)),
        ),
        fold=args.fold,
        pool=int(cfg_dict.get("pool", 1)),
        device=args.device,
        num_workers=args.num_workers,
    )


def _compile_cells(model) -> int:
    """torch.compile each recurrent cell in place. Returns how many were wrapped.

    We compile the *cell* (the launch-heavy KANLinear gate algebra called once
    per timestep), not the whole 256-step Python loop -- the cell has stable
    shapes, so it compiles once and fuses the elementwise/spline storm into a few
    kernels, which is exactly the launch-bound cost. Compiling the unrolled loop
    instead risks a giant graph / slow compile.
    """
    layers = getattr(getattr(model, "encoder", None), "layers", None)
    if layers is None:
        return 0
    n = 0
    for i in range(len(layers)):
        layers[i] = torch.compile(layers[i])
        n += 1
    return n


def _one_step(model, batch, optimizer, loss_weights):
    optimizer.zero_grad(set_to_none=True)
    out = model(batch)
    loss, _ = total_loss(out, batch, loss_weights)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase3_tkan.yaml")
    p.add_argument("--model", default="tkan")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--npz-root", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--folds-json", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=40, help="Timed batches after warmup.")
    p.add_argument("--warmup", type=int, default=5, help="Untimed batches (lazy init / compile / autotune).")
    p.add_argument("--compile", choices=["none", "cell"], default="none")
    args = p.parse_args(argv)

    cfg_dict = load_config(args.config)
    cfg = _train_config(cfg_dict, args)
    device = resolve_device(cfg.device)
    is_cuda = device.type == "cuda"

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    print("=== TKAN epoch probe ===")
    print(f"device={device}  model={cfg.model}  fold={cfg.fold}")
    print(f"batch_size={cfg.batch_size}  pool={cfg.pool}  "
          f"window={cfg.effective_window().length}/{cfg.effective_window().stride}  "
          f"num_workers={cfg.num_workers}  compile={args.compile}")
    if is_cuda:
        print(f"gpu={torch.cuda.get_device_name(0)}")

    table = pd.read_csv(args.table)
    with open(args.folds_json, "r", encoding="utf-8") as fh:
        folds_json = json.load(fh)

    t_build = time.perf_counter()
    train_loader, _val, _norm = build_fold_loaders(
        table, folds_json, cfg.fold, args.npz_root,
        cfg.effective_window(), cfg.batch_size,
        seed=cfg.seed, num_workers=cfg.num_workers, pin_memory=is_cuda,
    )
    build_s = time.perf_counter() - t_build
    n_windows = len(train_loader.dataset)
    batches_per_epoch = math.ceil(n_windows / cfg.batch_size)
    print(f"loader build: {build_s:.1f}s   train windows={n_windows}   "
          f"batches/epoch={batches_per_epoch}")

    model = build_model(cfg).to(device)
    n_compiled = _compile_cells(model) if args.compile == "cell" else 0
    if args.compile == "cell":
        print(f"torch.compile: wrapped {n_compiled} cell(s)")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()

    it = iter(train_loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(train_loader)
            return next(it)

    # Warmup (also triggers the first torch.compile, which can take a while).
    t_warm = time.perf_counter()
    for _ in range(max(args.warmup, 1)):
        b = _batch_to_device(next_batch(), device)
        _one_step(model, b, optimizer, cfg.loss)
        sync()
    print(f"warmup ({args.warmup} batches incl. any compile): {time.perf_counter()-t_warm:.1f}s")

    # Timed loop: split fetch / host->device / GPU compute.
    t_fetch = t_h2d = t_gpu = 0.0
    n = args.max_batches
    for _ in range(n):
        t0 = time.perf_counter()
        raw = next_batch()
        t1 = time.perf_counter()
        b = _batch_to_device(raw, device)
        sync()
        t2 = time.perf_counter()
        _one_step(model, b, optimizer, cfg.loss)
        sync()
        t3 = time.perf_counter()
        t_fetch += t1 - t0
        t_h2d += t2 - t1
        t_gpu += t3 - t2

    total = t_fetch + t_h2d + t_gpu
    per_batch = total / n
    est_epoch_s = per_batch * batches_per_epoch
    samples_per_s = (n * cfg.batch_size) / total

    def pct(x):
        return 100.0 * x / total if total else 0.0

    print("\n--- per-batch split (mean over "
          f"{n} timed batches) ---")
    print(f"  data fetch : {t_fetch/n*1000:8.1f} ms/batch  ({pct(t_fetch):4.1f}%)")
    print(f"  host->dev  : {t_h2d/n*1000:8.1f} ms/batch  ({pct(t_h2d):4.1f}%)")
    print(f"  GPU compute: {t_gpu/n*1000:8.1f} ms/batch  ({pct(t_gpu):4.1f}%)")
    print(f"  TOTAL      : {per_batch*1000:8.1f} ms/batch")
    print(f"\nthroughput  : {samples_per_s:.1f} samples/s")
    print(f"est. epoch  : {est_epoch_s/60:.1f} min  "
          f"({batches_per_epoch} batches x {per_batch*1000:.0f} ms)")
    bound = "DATA-bound" if t_fetch > t_gpu else "GPU-bound"
    print(f"verdict     : {bound}  "
          f"(GPU {pct(t_gpu):.0f}% vs data {pct(t_fetch):.0f}%)")

    print("\n" + json.dumps({
        "device": str(device),
        "compile": args.compile,
        "num_workers": cfg.num_workers,
        "batch_size": cfg.batch_size,
        "pool": cfg.pool,
        "n_windows": n_windows,
        "batches_per_epoch": batches_per_epoch,
        "ms_fetch": round(t_fetch / n * 1000, 2),
        "ms_h2d": round(t_h2d / n * 1000, 2),
        "ms_gpu": round(t_gpu / n * 1000, 2),
        "ms_total": round(per_batch * 1000, 2),
        "samples_per_s": round(samples_per_s, 1),
        "est_epoch_min": round(est_epoch_s / 60, 1),
        "verdict": bound,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
