"""Throughput benchmark: TKAN recurrence vs MLP, and scaling with seq length.

Diagnostic only (not committed). Measures forward+backward wall time per batch
at the real Phase 3 config to size the GL timeout problem.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import torch

from pri2m.models.tkan_model import TKANModel
from pri2m.models.baseline import MLPBaseline

torch.manual_seed(0)
torch.set_num_threads(8)  # match GL cpus-per-task=8
# Run on a GPU node with: BENCH_DEVICE=cuda python scripts/_bench_tkan.py
DEVICE = torch.device(os.environ.get("BENCH_DEVICE", "cpu"))


def _sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def make_batch(B: int, L: int, C: int = 4) -> dict:
    return {"window": torch.randn(B, L, C, device=DEVICE),
            "tau": torch.full((B,), 10, dtype=torch.long, device=DEVICE)}


def step_loss(out: dict) -> torch.Tensor:
    return out["b_over_blim"].pow(2).mean() + out["chatter_logit"].pow(2).mean()


def time_model(model, batch, iters: int = 3, warmup: int = 1) -> float:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(warmup):
        opt.zero_grad()
        step_loss(model(batch)).backward()
        opt.step()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad()
        step_loss(model(batch)).backward()
        opt.step()
    _sync()
    return (time.perf_counter() - t0) / iters


def tkan() -> TKANModel:
    return TKANModel(input_size=4, hidden_size=64, num_layers=1,
                     grid_size=5, spline_order=3, with_trajectory=False).to(DEVICE)


print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  device={DEVICE}")
if DEVICE.type == "cuda":
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)  # property query, no kernel launch
    supported = torch.cuda.get_arch_list()
    print(f"GPU: {name}  compute capability sm_{cc[0]}{cc[1]}  | torch archs: {supported}")
    if f"sm_{cc[0]}{cc[1]}" not in supported:
        print(f"!! This torch build has NO kernels for sm_{cc[0]}{cc[1]} -- "
              f"get an A40 (sm_86) node, or reinstall torch with sm_{cc[0]}{cc[1]}.")
        raise SystemExit(2)
print("=" * 64)

# 1) MLP at the real shape, for a same-machine reference.
B = 32
mlp = MLPBaseline(window_length=1024, n_channels=4, hidden_size=64).to(DEVICE)
mlp_s = time_model(mlp, make_batch(B, 1024))
print(f"MLP   B={B} L=1024 : {mlp_s*1000:8.1f} ms/batch")

# 2) TKAN scaling with sequence length (the pooling lever; cost ~ linear in L).
for L in (128, 256, 512, 1024):
    s = time_model(tkan(), make_batch(B, L), iters=3, warmup=1)
    print(f"TKAN  B={B} L={L:<4d}: {s*1000:8.1f} ms/batch   ({s/mlp_s:6.1f}x MLP)")

# 3) TKAN per-sample cost vs batch size at real L. On GPU, ms/batch should stay
#    ~flat as B grows (recurrence has the same step count) while ms/sample drops
#    sharply -> that throughput gain is the cheapest, no-modeling-change lever.
print("-" * 64)
for bb in (32, 64, 128, 256, 512):
    try:
        n = 2 if bb <= 128 else 1
        s = time_model(tkan(), make_batch(bb, 1024), iters=n, warmup=1)
        print(f"TKAN  B={bb:<4d} L=1024: {s*1000:8.1f} ms/batch   ({s/bb*1000:6.2f} ms/sample)")
    except RuntimeError as e:  # e.g. CUDA OOM at large B
        print(f"TKAN  B={bb:<4d} L=1024: SKIPPED ({str(e).splitlines()[0][:60]})")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
