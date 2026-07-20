"""Graded fuzziness (degradation) suite for the Phase 3 TKAN — eval-only.

Input-side fuzzers blur what the model SEES (noise, smoothing, tau jitter);
model-side fuzzers blur the MODEL itself (spline coarsening, weight noise).
At every fuzz level the probes re-measure not just AUC but the delay-ablation
drop — the physics-reliance signature — so the resulting curves answer
"up to what fuzz level does the model still use the regenerative delay".

All fuzzers are non-mutating: input fuzzers return a new batch dict; model
fuzzers deep-copy. AUC is per-window over the sampled batch (the whitebox
convention), via physics.delay_ablation which measures on+off in one call.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from pri2m.models.kan import KANLinear
from pri2m.xai import physics
from pri2m.xai.kan_native import _iter_kanlinear

AUC_DEGRADED = 0.75   # halfway to chance from ~1.0
DROP_FLOOR = 0.05     # delay-drop below this = physics signature gone
AUC_CHANCE = 0.55     # ≈ chance


# --------------------------------------------------------------------- #
# Input-side fuzzers (batch -> new batch; never mutate the input)
# --------------------------------------------------------------------- #
def fuzz_input_noise(batch: dict, sigma: float, seed: int = 0) -> dict:
    """Additive Gaussian noise (z-scored SD units) on the signal channels."""
    new = dict(batch)
    if sigma <= 0:
        new["window"] = batch["window"].clone()
        return new
    g = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(batch["window"].shape, generator=g) * float(sigma)
    new["window"] = batch["window"] + noise.to(batch["window"].dtype)
    return new


def fuzz_input_smooth(batch: dict, width: int) -> dict:
    """Reflect-padded centered moving average of ``width`` RAW samples."""
    new = dict(batch)
    width = int(width)
    if width <= 1:
        new["window"] = batch["window"].clone()
        return new
    x = batch["window"].permute(0, 2, 1)                      # [B, C, L]
    left = width // 2
    right = width - 1 - left
    padded = F.pad(x, (left, right), mode="reflect")
    smoothed = F.avg_pool1d(padded, kernel_size=width, stride=1)
    new["window"] = smoothed.permute(0, 2, 1).contiguous()
    return new


def fuzz_tau_jitter(batch: dict, frac: float, seed: int = 0) -> tuple[dict, dict]:
    """Perturb each window's tau by ±frac (random sign per window, seeded)."""
    tau = np.asarray(batch["tau"].detach().cpu().numpy(), dtype=float)
    if frac <= 0:
        return dict(batch), {"realized_abs_dtau_mean": 0.0, "frac_pooled_tau_changed": 0.0}
    g = np.random.default_rng(int(seed))
    signs = g.choice([-1.0, 1.0], size=tau.shape)
    new_tau = np.maximum(1, np.round(tau * (1.0 + signs * float(frac)))).astype(np.int64)
    new = dict(batch)
    new["tau"] = torch.as_tensor(new_tau, dtype=torch.long)
    info = {
        "realized_abs_dtau_mean": float(np.mean(np.abs(new_tau - tau))),
        # honesty: after the model's own pooling, small jitters can round away
        "frac_pooled_tau_changed": float(
            np.mean(np.maximum(1, np.round(new_tau / 4)) != np.maximum(1, np.round(tau / 4)))
        ),
    }
    return new, info


# --------------------------------------------------------------------- #
# Model-side fuzzers (model -> deep-copied fuzzed model)
# --------------------------------------------------------------------- #
def _refit_kanlinear_inplace(kan: KANLinear, grid_size: int, n_points: int = 64) -> float:
    """Least-squares refit of kan's spline onto a ``grid_size`` grid, in place.

    Samples the ORIGINAL learned spline component right-open on grid_range
    (the degree-0 basis is zero at exactly the right endpoint), solves one
    batched lstsq against the coarse basis, swaps coefficients + knot buffer.
    Returns the refit R^2 (1.0 = the coarse grid can represent the original).
    """
    order = kan.spline_order
    lo, hi = kan.grid_range
    xs = torch.linspace(lo, hi, n_points + 1, dtype=torch.float32)[:-1]     # right-open
    basis_orig = kan.b_splines(xs.unsqueeze(-1))[:, 0, :]                    # [n, n_coeff_orig]
    target = torch.einsum("xc,oic->oix", basis_orig, kan.spline_weight)      # [out, in, n]
    t = target.reshape(-1, n_points).T.contiguous()                          # [n, out*in]

    tmp = KANLinear(1, 1, grid_size=grid_size, spline_order=order, grid_range=(lo, hi))
    B = tmp.b_splines(xs.unsqueeze(-1))[:, 0, :]                             # [n, k+order]
    coeff = torch.linalg.lstsq(B, t).solution                                # [k+order, out*in]
    fitted = B @ coeff
    ss_res = ((t - fitted) ** 2).sum()
    ss_tot = ((t - t.mean(dim=0)) ** 2).sum().clamp_min(1e-12)
    r2 = float(1.0 - ss_res / ss_tot)

    kan.grid_size = int(grid_size)
    kan.n_coeff = int(grid_size) + order
    h = (hi - lo) / grid_size
    knots = torch.arange(-order, grid_size + order + 1, dtype=torch.float32) * h + lo
    kan._buffers["grid"] = knots.to(kan.grid.device)
    kan.spline_weight = torch.nn.Parameter(
        coeff.T.reshape(kan.out_features, kan.in_features, kan.n_coeff).contiguous()
    )
    return r2


def fuzz_spline_coarsen(model, grid_size: int) -> tuple[torch.nn.Module, dict]:
    fuzzed = copy.deepcopy(model)
    r2s = {}
    for name, kan in _iter_kanlinear(fuzzed):
        r2s[name or "kan"] = _refit_kanlinear_inplace(kan, int(grid_size))
    return fuzzed, {"refit_r2": r2s}


def fuzz_weight_noise(model, sigma_rel: float, seed: int = 0):
    fuzzed = copy.deepcopy(model)
    if sigma_rel <= 0:
        return fuzzed, {}
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for _, kan in _iter_kanlinear(fuzzed):
            for p in (kan.base_weight, kan.spline_weight):
                eps = torch.randn(p.shape, generator=g) * float(sigma_rel)
                p.mul_(1.0 + eps)
    return fuzzed, {}


# --------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------- #
def measure_probes(model, batch: dict, labels) -> dict:
    """AUC / severity MAE / delay-reliance, in one delay_ablation call."""
    abl = physics.delay_ablation(model, batch, labels)
    return {
        "auc": float(abl["auc_on"]),
        "auc_delay_off": float(abl["auc_off"]),
        "delay_drop": float(abl["auc_drop"]),
        "mae": float(abl["mae_on"]) if "mae_on" in abl else None,
        "mae_delay_off": float(abl["mae_off"]) if "mae_off" in abl else None,
    }


def xdelay_share(model, batch: dict):
    """x_delay data-aware weight share (None if the model has no delay block)."""
    from pri2m.xai import activations as act_mod

    stats = act_mod.collect_activations(model, batch)
    share = physics.delayed_weight_share(model, stats)
    overall = share.get("overall", {})
    return float(overall["x_delay"]) if "x_delay" in overall else None


# --------------------------------------------------------------------- #
# Registry + sweep loop
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fuzzer:
    name: str
    kind: str          # "input" | "model"
    levels: tuple
    stochastic: bool
    apply: Callable    # (model, batch, level, seed) -> (model, batch, info)


def _apply_input_noise(model, batch, level, seed):
    return model, fuzz_input_noise(batch, level, seed), {}


def _apply_input_smooth(model, batch, level, seed):
    return model, fuzz_input_smooth(batch, level), {}


def _apply_tau_jitter(model, batch, level, seed):
    new, info = fuzz_tau_jitter(batch, level, seed)
    return model, new, info


def _apply_spline_coarsen(model, batch, level, seed):
    fuzzed, info = fuzz_spline_coarsen(model, level)
    return fuzzed, batch, info


def _apply_weight_noise(model, batch, level, seed):
    fuzzed, info = fuzz_weight_noise(model, level, seed)
    return fuzzed, batch, info


FUZZERS: dict[str, Fuzzer] = {
    "input_noise": Fuzzer("input_noise", "input", (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0), True, _apply_input_noise),
    "input_smooth": Fuzzer("input_smooth", "input", (1, 2, 4, 8, 16, 32), False, _apply_input_smooth),
    "tau_jitter": Fuzzer("tau_jitter", "input", (0.0, 0.05, 0.10, 0.25, 0.50), True, _apply_tau_jitter),
    "spline_coarsen": Fuzzer("spline_coarsen", "model", (5, 4, 3, 2, 1), False, _apply_spline_coarsen),
    "weight_noise": Fuzzer("weight_noise", "model", (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5), True, _apply_weight_noise),
}

_PROBE_KEYS = ("auc", "auc_delay_off", "delay_drop", "mae", "mae_delay_off")


def breakdown_points(levels, mean_auc, mean_drop) -> dict:
    """First level crossing each degradation threshold (None if never)."""
    def first(seq, pred):
        for lvl, v in zip(levels, seq):
            if v is not None and pred(v):
                return lvl
        return None

    return {
        "auc_below_075": first(mean_auc, lambda a: a < AUC_DEGRADED),
        "drop_below_005": first(mean_drop, lambda d: d < DROP_FLOOR),
        "auc_at_chance": first(mean_auc, lambda a: a <= AUC_CHANCE),
    }


def run_fuzzer(model, batch, labels, fuzzer: Fuzzer, seeds=(0, 1, 2), with_share=None) -> dict:
    """Sweep one fuzzer over its levels; stochastic fuzzers get one probe per seed.

    with_share: None -> record xdelay_share only for model-side fuzzers.
    """
    if with_share is None:
        with_share = fuzzer.kind == "model"
    seeds = tuple(seeds) if fuzzer.stochastic else (0,)
    out = []
    for level in fuzzer.levels:
        per_seed, info = [], {}
        for seed in seeds:
            fm, fb, info = fuzzer.apply(model, batch, level, seed)
            probes = measure_probes(fm, fb, labels)
            if with_share:
                probes["xdelay_share"] = xdelay_share(fm, fb)
            probes["seed"] = int(seed)
            per_seed.append(probes)
        keys = [k for k in per_seed[0] if k != "seed"]
        mean = {k: (float(np.mean([p[k] for p in per_seed])) if per_seed[0][k] is not None else None) for k in keys}
        sd = {k: (float(np.std([p[k] for p in per_seed])) if per_seed[0][k] is not None else None) for k in keys}
        out.append({"level": level, "seeds": per_seed, "mean": mean, "sd": sd, "info": info})
    return {
        "name": fuzzer.name,
        "kind": fuzzer.kind,
        "stochastic": fuzzer.stochastic,
        "levels": list(fuzzer.levels),
        "results": out,
        "breakdown": breakdown_points(
            fuzzer.levels,
            [r["mean"]["auc"] for r in out],
            [r["mean"]["delay_drop"] for r in out],
        ),
    }


# --------------------------------------------------------------------- #
# Plots (values only ever come from run_fuzzer results)
# --------------------------------------------------------------------- #
def _panel(ax, result: dict, baseline: dict) -> None:
    levels = result["levels"]
    xs = np.arange(len(levels))
    auc = [r["mean"]["auc"] for r in result["results"]]
    sd = [r["sd"]["auc"] for r in result["results"]]
    drop = [r["mean"]["delay_drop"] for r in result["results"]]
    ax.plot(xs, auc, "o-", color="tab:blue", label="AUC")
    ax.fill_between(xs, np.array(auc) - np.array(sd), np.array(auc) + np.array(sd),
                    color="tab:blue", alpha=0.2)
    ax.axhline(baseline["auc"], color="tab:blue", ls=":", lw=1, label="baseline AUC")
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(xs, [str(level) for level in levels])
    ax.set_xlabel(f"{result['name']} level")
    ax.set_ylabel("per-window AUC")
    twin = ax.twinx()
    twin.plot(xs, drop, "s--", color="tab:red", label="delay-ablation drop")
    twin.axhline(DROP_FLOOR, color="tab:red", ls=":", lw=1)
    twin.set_ylabel("delay-ablation AUC drop", color="tab:red")
    twin.set_ylim(bottom=min(0.0, min(drop)))
    ax.set_title(result["name"])


def plot_degradation_panel(result: dict, baseline: dict, path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(5.5, 4))
    _panel(ax, result, baseline)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_fuzz_dashboard(all_results: list[dict], baseline: dict, path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    n = max(1, len(all_results))
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    for ax, res in zip(flat[:n], all_results):
        _panel(ax, res, baseline)
    fig.suptitle("Physics understanding vs fuzz level (per-window AUC + delay-ablation drop)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
