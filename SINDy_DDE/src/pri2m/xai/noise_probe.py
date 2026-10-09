"""Noise as the explanation for the MiniROCKET-vs-TKAN held-out gap (T1/T2).

Every held-out dataset (121-125) carries 5 % noise, while each fold's own
validation set mixes clean, 1 % and 5 % copies of the same cuts. MiniROCKET
scores the same on both (0.959 val / 0.962 held-out); TKAN drops 0.955 ->
0.908, and its held-out accuracy (0.787) falls far below what its AUC implies.
The hypothesis this module serves is therefore narrow: **TKAN is the less
noise-robust model**, and the held-out gap is a noise gap rather than a physics
gap.

Two pieces, shared by both model runners so the two models see BIT-IDENTICAL
noisy inputs:

calibration
    Clean dataset ``D`` and its mirrors ``D+60`` (1 %) / ``D+90`` (5 %) are the
    same simulated cuts, so ``mirror - clean`` IS the added noise. Its size per
    channel (relative to the clean channel's std), whiteness (lag-1
    autocorrelation) and independence from the signal are measured directly
    rather than assumed. If the difference is not additive independent noise
    (e.g. the noisy mirror was re-simulated), the injection cannot reproduce the
    tier and the summary refuses rather than guessing.
injection
    Gaussian noise, white, per channel, with std ``level * rel_sigma_per_unit[c]
    * std(file channel c)`` -- relative to each file's OWN signal, in raw units,
    before the train-fold z-score, so both models normalise it identically. The
    generator is seeded from ``(seed, key, level)`` through a stable hash, so a
    given file at a given level gets the same noise in every process and for
    every model.

Pure functions, no file IO.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

__all__ = [
    "noise_tier",
    "mirror_file",
    "mirror_noise_stats",
    "summarise_calibration",
    "inject_noise",
]

# A mirror difference this correlated with the clean signal, or this
# self-correlated, is not the additive white noise the injection draws.
_MAX_ABS_CORR_WITH_CLEAN = 0.1
_MAX_ABS_LAG1 = 0.2
# The 5 % tier's per-unit sigma must agree with the 1 % tier's to this ratio
# for "level" to be a linear dial.
_LINEAR_TOL = 0.15


def noise_tier(dataset: int) -> int | None:
    """Noise tier in percent from the 1-120 id layout; ``None`` outside it.

    1-60 clean, 61-90 = 1 % mirrors of 1-30, 91-120 = 5 % mirrors of 1-30
    (``pri2m.phase3.splits``). Held-out 121+ must take the tier from metadata.
    """
    d = int(dataset)
    if 1 <= d <= 60:
        return 0
    if 61 <= d <= 90:
        return 1
    if 91 <= d <= 120:
        return 5
    return None


_FILE_RE = re.compile(r"^time(\d+)_(\d+)(\.\w+)$")


def mirror_file(file: str, dataset: int, mirror_dataset: int) -> str:
    """``time{D}_{k}.h5`` -> ``time{M}_{k}.h5``; refuses a name not from ``D``."""
    m = _FILE_RE.match(str(file))
    if not m or int(m.group(1)) != int(dataset):
        raise ValueError(f"{file!r} is not a file of dataset {dataset}")
    return f"time{int(mirror_dataset)}_{m.group(2)}{m.group(3)}"


def _lag1(x: np.ndarray) -> float:
    x = x - x.mean()
    den = float((x * x).sum())
    return float((x[1:] * x[:-1]).sum() / den) if den > 0 else float("nan")


def mirror_noise_stats(clean: np.ndarray, mirror: np.ndarray) -> dict:
    """Per-channel statistics of ``mirror - clean`` for one matched file pair.

    Both are ``[T, C]``; truncated to the common length. ``rel_sigma`` is the
    difference std over the CLEAN channel std.
    """
    clean = np.asarray(clean, dtype=np.float64)
    mirror = np.asarray(mirror, dtype=np.float64)
    if clean.ndim != 2 or mirror.ndim != 2 or clean.shape[1] != mirror.shape[1]:
        raise ValueError(f"need two [T, C] arrays, got {clean.shape} and {mirror.shape}")
    T = min(clean.shape[0], mirror.shape[0])
    c, m = clean[:T], mirror[:T]
    diff = m - c
    cstd = c.std(axis=0)
    rel = np.divide(diff.std(axis=0), cstd, out=np.full(c.shape[1], np.nan), where=cstd > 0)
    lag1 = np.array([_lag1(diff[:, j]) for j in range(c.shape[1])])
    corr = np.array([
        float(np.corrcoef(diff[:, j], c[:, j])[0, 1]) if diff[:, j].std() > 0 and cstd[j] > 0
        else float("nan") for j in range(c.shape[1])
    ])
    additive = bool(np.all(np.isfinite(corr)) and np.all(np.abs(corr) < _MAX_ABS_CORR_WITH_CLEAN)
                    and np.all(np.isfinite(lag1)) and np.all(np.abs(lag1) < _MAX_ABS_LAG1))
    return {
        "n_samples": int(T),
        "rel_sigma": rel.tolist(),
        "lag1_autocorr": lag1.tolist(),
        "corr_with_clean": corr.tolist(),
        "additive_ok": additive,
    }


def summarise_calibration(rows: list[dict]) -> dict:
    """Median per-unit relative sigma over many pairs, with two guards.

    ``rows`` carry ``tier`` (1 or 5) plus a :func:`mirror_noise_stats` result.
    ``rel_sigma_per_unit[c]`` = median over pairs of ``rel_sigma[c] / tier``,
    taken from the 5 % tier when present (the larger, better-resolved signal).
    Refuses (``None`` + reason) when most pairs are not additive. ``linear_ok``
    says whether the 1 % and 5 % tiers agree per unit; if not, "level" is not a
    linear dial and only the calibrated tiers themselves should be injected.
    """
    out = {"rel_sigma_per_unit": None, "per_tier": {}, "linear_ok": None,
           "n_pairs": {}, "frac_additive": None, "undecided_reason": None}
    if not rows:
        out["undecided_reason"] = "no_pairs"
        return out
    ok = [r for r in rows if r.get("additive_ok")]
    out["frac_additive"] = len(ok) / len(rows)
    for tier in sorted({int(r["tier"]) for r in rows}):
        out["n_pairs"][str(tier)] = sum(int(r["tier"]) == tier for r in rows)
    if out["frac_additive"] < 0.9:
        out["undecided_reason"] = "mirror_difference_not_additive"
        return out
    per_tier = {}
    for tier in sorted({int(r["tier"]) for r in ok}):
        arr = np.array([r["rel_sigma"] for r in ok if int(r["tier"]) == tier], dtype=np.float64)
        per_tier[tier] = np.nanmedian(arr, axis=0) / float(tier)
        out["per_tier"][str(tier)] = per_tier[tier].tolist()
    ref = per_tier.get(5, per_tier.get(1))
    out["rel_sigma_per_unit"] = ref.tolist()
    if 1 in per_tier and 5 in per_tier:
        ratio = per_tier[5] / per_tier[1]
        out["linear_ok"] = bool(np.all(np.abs(ratio - 1.0) < _LINEAR_TOL))
    return out


def _rng_for(key: str, level: float, seed: int) -> np.random.Generator:
    h = hashlib.sha256(f"{int(seed)}|{key}|{float(level):.6g}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def inject_noise(signal: np.ndarray, level: float, rel_sigma_per_unit, key: str,
                 seed: int = 0) -> np.ndarray:
    """Add white Gaussian noise at ``level`` (tier units, e.g. 5.0 = the 5 % tier).

    ``signal`` is one file's raw ``[T, C]`` signal; per-channel noise std is
    ``level * rel_sigma_per_unit[c] * std(signal[:, c])``. Returns a new array;
    ``level == 0`` returns an unchanged copy.
    """
    s = np.asarray(signal, dtype=np.float64)
    u = np.asarray(rel_sigma_per_unit, dtype=np.float64).reshape(-1)
    if s.ndim != 2:
        raise ValueError(f"signal must be [T, C], got {s.shape}")
    if u.shape[0] != s.shape[1]:
        raise ValueError(f"rel_sigma_per_unit has {u.shape[0]} entries for {s.shape[1]} channels")
    if not np.isfinite(level) or level < 0:
        raise ValueError(f"level must be finite and >= 0, got {level}")
    if level == 0:
        return s.copy()
    sigma = float(level) * u * s.std(axis=0)
    rng = _rng_for(key, level, seed)
    return s + rng.standard_normal(s.shape) * sigma
