"""Exp 3 comparator: MiniROCKET + ridge on exactly the input TKAN sees.

Why this baseline, and why built this way
-----------------------------------------
The paper's question for Exp 3 is whether a strong, physics-free time-series
classifier needs the kinematic delay to detect chatter, and whether -- once it
is handed the delayed signal -- it leans on it the way TKAN does. The answer is
only interpretable if the comparator's input is TKAN's input, so everything up
to the classifier is reproduced, not approximated:

* same 1024/512 windows and the same train-fold z-score (fit on ALL training
  windows of the fold, as ``build_fold_loaders`` does);
* same temporal average-pool (:func:`pri2m.models.tkan_model.avg_pool_time`,
  called directly, so it is the same arithmetic, not a re-implementation);
* same pooled delay (:func:`pri2m.models.tkan_model.scale_tau`) and the same
  delayed slice as :func:`pri2m.models.tkan.delayed_input` (zeros before the
  delay has elapsed) -- tested element-for-element against that function;
* same misstated-delay arithmetic as the TKAN variants
  (:func:`pri2m.xai.delay_variants.encoder_tau`), so "x0.5" is the same
  integer delay for both models.

Arm A feeds the 4 physical channels; arm B appends the 4 delayed channels.

The classifier is MiniROCKET (aeon) -> per-feature standardisation -> ridge.
Everything except the transform itself lives here as aeon-free numpy code, so
it is testable without aeon and the only aeon dependency is one import in
:meth:`MiniRocketBaseline._make_transformer`.

Ridge is solved in closed form through ONE eigendecomposition of the
standardised Gram matrix ``Z^T Z`` (p ~ 10 000 features), which makes every
alpha in the grid a single matrix-vector product. It is accumulated in row
chunks, so the [n, p] feature matrix is never copied in float64 or
standardised in place -- the difference between ~9 GB and ~35 GB on a fold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from pri2m.data.phase3_sequence import NormStats, aggregate_file_predictions
from pri2m.models.tkan_model import avg_pool_time
from pri2m.xai.delay_variants import delay_drop, encoder_tau, exceeds_floor

__all__ = [
    "DEFAULT_ALPHAS",
    "pool_windows",
    "build_inputs",
    "select_train_windows",
    "StreamingNorm",
    "inner_group_split",
    "RidgePath",
    "fit_ridge_path",
    "file_level_metrics",
    "choose_alpha",
    "summarise_delay_tests",
    "MiniRocketBaseline",
]

#: Ridge grid. The ROCKET papers use logspace(-3, 3, 10) with ~1e3 training
#: cases; here n ~ 2e5, and the unpenalised term ||y - Zw||^2 grows with n, so
#: the same alpha is ~200x weaker relative to the data. The grid is extended to
#: 1e6 so a strongly-regularised optimum is reachable; half-decade steps.
DEFAULT_ALPHAS: tuple[float, ...] = tuple(float(a) for a in np.logspace(-3, 6, 19))


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def pool_windows(windows, pool: int) -> np.ndarray:
    """``[N, L, C]`` -> ``[N, L//pool, C]`` float32 via TKAN's own pooling op.

    Routed through :func:`avg_pool_time` on a float32 tensor rather than a
    numpy reshape-mean so the summation order -- and therefore every bit of the
    pooled value -- is the one TKAN's forward computes.
    """
    w = torch.as_tensor(np.ascontiguousarray(windows), dtype=torch.float32)
    if w.ndim != 3:
        raise ValueError(f"windows must be [N, L, C], got {tuple(w.shape)}")
    return avg_pool_time(w, int(pool)).numpy()


def build_inputs(
    windows,
    tau_raw,
    pool: int = 4,
    arm: str = "A",
    tau_scale: float = 1.0,
    zero_delay: bool = False,
) -> np.ndarray:
    """Build MiniROCKET inputs ``[N, C or 2C, L//pool]`` float32, channel-first.

    ``windows`` are ALREADY z-scored ``[N, L, C]`` (train-fold stats);
    ``tau_raw`` is the per-window delay in RAW samples (``round(fs / tpf)``).

    Arm ``"A"``: the pooled physical channels only. The delay arguments are
    refused rather than ignored, so a caller cannot believe it ran a delay test
    on an arm that has no delayed channel.

    Arm ``"B"``: pooled channels + delayed channels, where delayed[:, :, t] =
    pooled[:, :, t - tau_p] and 0 for ``t < tau_p``, with
    ``tau_p = encoder_tau(tau_raw, pool, Lp, "scale", tau_scale)`` -- i.e.
    ``scale_tau(round(tau_raw * tau_scale), pool)``. The scale is applied to the
    raw integer and THEN pool-scaled, the same order the TKAN variants use (see
    :func:`pri2m.xai.delay_variants.rescale_raw_tau` for why).
    ``zero_delay=True`` zeroes the delayed channels, which is what TKAN's
    encoder sees for ``tau >= L``.
    """
    if arm not in ("A", "B"):
        raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")
    if arm == "A" and (zero_delay or tau_scale != 1.0):
        raise ValueError("arm A has no delayed channels; tau_scale/zero_delay must be defaults")
    pooled = pool_windows(windows, pool)  # [N, Lp, C]
    N, Lp, C = pooled.shape
    tau_raw = np.asarray(tau_raw).reshape(-1)
    if tau_raw.shape[0] != N:
        raise ValueError(f"tau_raw has {tau_raw.shape[0]} entries for {N} windows")
    base = np.ascontiguousarray(pooled.transpose(0, 2, 1))  # [N, C, Lp]
    if arm == "A":
        return base

    delayed = np.zeros_like(base)
    if not zero_delay:
        tau_p = encoder_tau(torch.as_tensor(tau_raw.astype(np.int64)), pool, Lp,
                            "scale", tau_scale).numpy()
        t = np.arange(Lp)
        src = t[None, :] - tau_p[:, None]  # [N, Lp]
        valid = src >= 0
        src_c = np.clip(src, 0, Lp - 1)
        gathered = np.take_along_axis(base, src_c[:, None, :].repeat(C, axis=1), axis=2)
        delayed = np.where(valid[:, None, :], gathered, np.float32(0)).astype(np.float32)
    return np.concatenate([base, delayed], axis=1)


def select_train_windows(n_windows_in_file: int, max_per_file: int = 8) -> np.ndarray:
    """Evenly spaced window indices (first and last included), deterministic.

    Capping windows per file keeps the ridge's Gram pass and the feature matrix
    (n x 10k float32) bounded, and "evenly spaced" rather than random keeps the
    selection reproducible and spread over the whole cut instead of clustered.
    Files with <= ``max_per_file`` windows contribute all of them.
    """
    n = int(n_windows_in_file)
    k = int(max_per_file)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if n <= k:
        return np.arange(n, dtype=np.int64)
    if k == 1:
        return np.zeros(1, dtype=np.int64)
    # Spacing (n-1)/(k-1) >= 1 when n > k, so the rounded indices are distinct.
    return np.round(np.linspace(0, n - 1, k)).astype(np.int64)


class StreamingNorm:
    """Per-channel mean/std over many window arrays without holding them all.

    ``NormStats.fit_streaming`` materialises its iterable into a list (TKAN can
    afford that because its dataset caches every window anyway). The comparator
    only keeps <= 8 windows per training file, so it merges per-file moments
    with Chan et al.'s parallel update in float64 instead. Agrees with
    ``NormStats.fit`` on the concatenation to ~1e-12 relative (tested), and
    applies the same 1e-8 std floor.
    """

    def __init__(self, n_channels: int) -> None:
        self.n = 0
        self.mean = np.zeros(int(n_channels), dtype=np.float64)
        self.m2 = np.zeros(int(n_channels), dtype=np.float64)

    def update(self, windows) -> None:
        a = np.asarray(windows)
        if a.size == 0:
            return
        flat = a.reshape(-1, a.shape[-1]).astype(np.float64)
        nb = flat.shape[0]
        mb = flat.mean(axis=0)
        m2b = ((flat - mb) ** 2).sum(axis=0)
        n = self.n + nb
        delta = mb - self.mean
        self.mean = self.mean + delta * (nb / n)
        self.m2 = self.m2 + m2b + delta**2 * (self.n * nb / n)
        self.n = n

    def to_normstats(self) -> NormStats:
        if self.n == 0:
            raise ValueError("no windows to fit normalization")
        std = np.maximum(np.sqrt(self.m2 / self.n), NormStats._STD_FLOOR)
        return NormStats(mean=self.mean.copy(), std=std)


# --------------------------------------------------------------------------- #
# Inner split
# --------------------------------------------------------------------------- #
def inner_group_split(parent_groups, frac: float = 0.2, seed: int = 0) -> np.ndarray:
    """Boolean mask (True = inner validation) holding out whole parent groups.

    Alpha is chosen on this split, so it has to obey the same rule as the outer
    folds: a clean cut and its noisy mirrors (and parameter twins, which share
    a ``parent_group``) are one event and may not straddle it, or the inner
    score rewards memorising a cut and picks too little regularisation.
    ``ceil(frac * n_groups)`` groups are held out, clamped so both sides keep at
    least one group. Deterministic in ``seed``.
    """
    g = np.asarray(parent_groups)
    uniq = np.unique(g)
    if uniq.size < 2:
        raise ValueError(f"need >= 2 parent groups for an inner split, got {uniq.size}")
    n_val = min(max(1, int(math.ceil(float(frac) * uniq.size))), uniq.size - 1)
    rng = np.random.default_rng(int(seed))
    val_groups = rng.permutation(uniq)[:n_val]
    return np.isin(g, val_groups)


# --------------------------------------------------------------------------- #
# Ridge
# --------------------------------------------------------------------------- #
@dataclass
class RidgePath:
    """Ridge solutions for a grid of alphas on standardised features.

    ``mean``/``std`` are the training rows' feature statistics (constant
    features get std 1, so they standardise to 0 and drop out instead of
    dividing by zero). ``weights[i]`` solves alpha ``alphas[i]``;
    ``intercept`` is the training mean of ``y`` (exact, since Z is centred).
    """

    alphas: np.ndarray
    weights: np.ndarray  # [n_alphas, p] float64
    intercept: float
    mean: np.ndarray
    std: np.ndarray
    # Per-alpha intercept correction. Zero for an unweighted fit (Z is centred
    # on the training rows, so the intercept is exactly mean(y)); a weighted fit
    # centres on the WEIGHTED mean, which leaves a per-alpha offset.
    offsets: np.ndarray | None = None

    def _offset(self, i: int) -> float:
        return 0.0 if self.offsets is None else float(self.offsets[int(i)])

    def decision(self, X, alpha_index: int = 0, chunk_rows: int = 4096) -> np.ndarray:
        """``((X - mean) / std) @ w + b`` in float64, chunked over rows."""
        X = np.asarray(X)
        w = self.weights[int(alpha_index)]
        out = np.empty(X.shape[0], dtype=np.float64)
        for s in range(0, X.shape[0], int(chunk_rows)):
            Z = (X[s : s + chunk_rows].astype(np.float64) - self.mean) / self.std
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                # Spurious Accelerate FP flags (see fit_ridge_path); a real
                # non-finite feature still yields a non-finite score, which the
                # file-level metric excludes and counts.
                out[s : s + chunk_rows] = Z @ w + self.intercept + self._offset(alpha_index)
        return out


    def decision_all(self, X, chunk_rows: int = 4096) -> np.ndarray:
        """Decision for every alpha at once, ``[n, n_alphas]`` float64."""
        X = np.asarray(X)
        out = np.empty((X.shape[0], self.weights.shape[0]), dtype=np.float64)
        for s in range(0, X.shape[0], int(chunk_rows)):
            Z = (X[s : s + chunk_rows].astype(np.float64) - self.mean) / self.std
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                out[s : s + chunk_rows] = Z @ self.weights.T + self.intercept
        if self.offsets is not None:
            out += self.offsets[None, :]
        return out


def _row_chunks(rows: np.ndarray, chunk_rows: int):
    for s in range(0, rows.shape[0], int(chunk_rows)):
        yield rows[s : s + chunk_rows]


def fit_ridge_path(X, y, alphas: Sequence[float], rows=None, chunk_rows: int = 4096,
                   sample_weight=None) -> RidgePath:
    """Closed-form ridge for every alpha from one eigendecomposition.

    ``X`` is ``[n, p]`` (float32 is fine; it is upcast one chunk at a time),
    ``y`` in {-1, +1}. ``rows`` restricts the fit to a subset WITHOUT copying
    ``X`` (the inner alpha split uses this on an ~8 GB matrix).

    With ``Z`` the standardised rows and ``y_c = y - mean(y)``,
    ``Z^T Z = V diag(lam) V^T`` gives ``w(alpha) = V ((V^T Z^T y_c) / (lam + alpha))``.
    Identical to sklearn ``Ridge(alpha, fit_intercept=True)`` on ``Z`` (tested).

    ``sample_weight`` (one per row of ``X``, >= 0) gives weighted least squares,
    matching sklearn's ``sample_weight``. TKAN's loss is weighted by the
    per-file ``sample_weight`` (noisier files count less), so the comparator
    is fit under the same weighting rather than an easier, unweighted one.
    The standardisation itself stays unweighted; only the fit is weighted.
    ``None`` reproduces the unweighted path exactly.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("y must be in {-1, +1}")
    rows = np.arange(X.shape[0]) if rows is None else np.asarray(rows)
    if rows.dtype == bool:
        rows = np.flatnonzero(rows)
    n = rows.shape[0]
    if n == 0:
        raise ValueError("no rows to fit")
    p = X.shape[1]
    sw = None
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if sw.shape[0] != X.shape[0]:
            raise ValueError(f"sample_weight has {sw.shape[0]} entries for {X.shape[0]} rows")
        if not np.all(np.isfinite(sw)) or np.any(sw < 0) or float(sw[rows].sum()) <= 0:
            raise ValueError("sample_weight must be finite, >= 0, and not all zero")

    # Pass 1: feature mean / population std (Chan merge per chunk, float64).
    cnt, mean, m2 = 0, np.zeros(p), np.zeros(p)
    for idx in _row_chunks(rows, chunk_rows):
        c = X[idx].astype(np.float64)
        nb = c.shape[0]
        mb = c.mean(axis=0)
        m2b = ((c - mb) ** 2).sum(axis=0)
        tot = cnt + nb
        d = mb - mean
        mean = mean + d * (nb / tot)
        m2 = m2 + m2b + d**2 * (cnt * nb / tot)
        cnt = tot
    std = np.sqrt(m2 / cnt)
    std[~(std > 0)] = 1.0

    ys = y[rows]
    if sw is not None:
        return _weighted_path(X, y, sw, rows, alphas, mean, std, chunk_rows)
    ybar = float(ys.mean())
    # Pass 2: Gram and Z^T y_c.
    G = np.zeros((p, p))
    b = np.zeros(p)
    # errstate: numpy 2.x on macOS Accelerate raises spurious divide/overflow
    # FP flags from finite matmuls (reproduced here on finite inputs). The
    # flags are silenced and replaced by an explicit finiteness check below,
    # so a genuinely non-finite Gram still fails loudly instead of quietly.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for idx in _row_chunks(rows, chunk_rows):
            Z = (X[idx].astype(np.float64) - mean) / std
            G += Z.T @ Z
            b += Z.T @ (y[idx] - ybar)
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(b))):
        raise ValueError("non-finite Gram matrix: features contain NaN/inf")
    lam, V = np.linalg.eigh(G)
    lam = np.clip(lam, 0.0, None)  # round-off can leave tiny negatives
    alphas = np.asarray([float(a) for a in alphas], dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        vb = V.T @ b
        W = np.stack([V @ (vb / (lam + a)) for a in alphas], axis=0)
    return RidgePath(alphas=alphas, weights=W, intercept=ybar, mean=mean, std=std)


def _weighted_path(X, y, sw, rows, alphas, mean, std, chunk_rows) -> RidgePath:
    """Weighted ridge with intercept, on the unweighted standardisation.

    Accumulates ``sum w``, ``sum w z``, ``sum w y``, ``sum w z z^T`` and
    ``sum w z y`` in one pass, then centres on the weighted means:
    ``G = Szz - W m m^T``, ``b = Szy - W m ybar_w``, intercept
    ``ybar_w - m . beta(alpha)``.
    """
    p = X.shape[1]
    W, sz, sy = 0.0, np.zeros(p), 0.0
    Szz, Szy = np.zeros((p, p)), np.zeros(p)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for idx in _row_chunks(rows, chunk_rows):
            Z = (X[idx].astype(np.float64) - mean) / std
            wi, yi = sw[idx], y[idx]
            Zw = Z * wi[:, None]
            W += float(wi.sum())
            sz += Zw.sum(axis=0)
            sy += float((wi * yi).sum())
            Szz += Zw.T @ Z
            Szy += Zw.T @ yi
    m, ybar = sz / W, sy / W
    G = Szz - W * np.outer(m, m)
    b = Szy - W * m * ybar
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(b))):
        raise ValueError("non-finite Gram matrix: features contain NaN/inf")
    lam, V = np.linalg.eigh(G)
    lam = np.clip(lam, 0.0, None)
    alphas = np.asarray([float(a) for a in alphas], dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        vb = V.T @ b
        Wt = np.stack([V @ (vb / (lam + a)) for a in alphas], axis=0)
    offsets = -(Wt @ m)
    return RidgePath(alphas=alphas, weights=Wt, intercept=float(ybar), mean=mean,
                     std=std, offsets=offsets)


# --------------------------------------------------------------------------- #
# File-level metric
# --------------------------------------------------------------------------- #
def file_level_metrics(scores, labels, file_ids) -> dict:
    """TKAN's file-level aggregation applied to a ridge decision score.

    Per file: mean window score (``aggregate_file_predictions``) and mean label
    (``>= 0.5`` -> positive, the rule ``per_file_severity_metrics`` uses). AUC
    is on the mean score directly: a monotone map (TKAN's sigmoid) cannot change
    a rank-based AUC, so no squashing is needed. ``acc`` thresholds the mean
    score at 0, the ridge's own boundary for +-1 targets (the analogue of
    TKAN's sigmoid >= 0.5).

    Non-finite file means are excluded and counted; a single-class or empty set
    gives ``auc = None`` (and ``acc = None`` when no file survives) -- never a
    number standing in for "unmeasurable".
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    file_ids = list(file_ids)
    if not (len(scores) == len(labels) == len(file_ids)):
        raise ValueError("scores, labels, file_ids must have equal length")
    if not file_ids:
        return {"auc": None, "acc": None, "n_files": 0, "n_nonfinite_files": 0}
    mean_score = aggregate_file_predictions(scores, file_ids)
    mean_label = aggregate_file_predictions(labels, file_ids)
    keys = list(mean_score)
    s = np.array([mean_score[k] for k in keys])
    lab = (np.array([mean_label[k] for k in keys]) >= 0.5).astype(int)
    fin = np.isfinite(s)
    s_k, l_k = s[fin], lab[fin]
    auc = None
    if l_k.size and l_k.min() != l_k.max():
        auc = float(roc_auc_score(l_k, s_k))
    acc = float(np.mean((s_k >= 0).astype(int) == l_k)) if l_k.size else None
    return {"auc": auc, "acc": acc, "n_files": len(keys),
            "n_nonfinite_files": int(np.count_nonzero(~fin))}


def choose_alpha(X, y, file_ids, parent_groups, alphas: Sequence[float],
                 frac: float = 0.2, seed: int = 0, chunk_rows: int = 4096,
                 sample_weight=None) -> dict:
    """Pick alpha by file-level AUC on an inner parent-group split.

    Fits the whole path on the inner-train groups (their own standardisation),
    scores each alpha on the inner-validation files with
    :func:`file_level_metrics` -- the same metric the outer evaluation reports
    -- and keeps the best. Ties (AUC is rank-based, so exact ties are common on
    small sets) go to the LARGER alpha: of equally good models, the more
    regularised one is the less over-fit bet.

    Refuses (ValueError) if no alpha is scoreable, which happens when the inner
    validation groups hold a single class: choosing an alpha from no evidence
    and reporting it as "selected" would be the silent-default failure this
    project keeps having to undo.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    groups = np.asarray(parent_groups)
    files = np.asarray(file_ids)
    val = inner_group_split(groups, frac=frac, seed=seed)
    tr_rows, va_rows = np.flatnonzero(~val), np.flatnonzero(val)
    # Weighted FIT, unweighted SCORE: TKAN's val AUC is unweighted too.
    path = fit_ridge_path(X, y, alphas, rows=tr_rows, chunk_rows=chunk_rows,
                          sample_weight=sample_weight)
    Xv = np.asarray(X)
    # Standardise each inner-val chunk once and score every alpha from it.
    S = np.concatenate([path.decision_all(Xv[idx], chunk_rows)
                        for idx in _row_chunks(va_rows, chunk_rows)])
    inner = [file_level_metrics(S[:, i], (y[va_rows] > 0).astype(int),
                                files[va_rows].tolist())["auc"]
             for i in range(len(path.alphas))]
    scored = [(a, i) for i, a in enumerate(inner) if a is not None]
    if not scored:
        raise ValueError(
            "inner validation split is single class (or empty): no alpha can be "
            "scored; refusing to pick one by default"
        )
    best = max(a for a, _ in scored)
    order = np.argsort(path.alphas)
    best_i = int(max((i for a, i in scored if a == best),
                     key=lambda j: path.alphas[j]))
    return {
        "alpha": float(path.alphas[best_i]),
        "alpha_index": best_i,
        "alphas": [float(a) for a in path.alphas],
        "inner_auc": inner,
        "at_grid_edge": bool(best_i in (int(order[0]), int(order[-1]))),
        "n_inner_train_groups": int(np.unique(groups[~val]).size),
        "n_inner_val_groups": int(np.unique(groups[val]).size),
        "n_inner_train_windows": int(tr_rows.size),
        "n_inner_val_windows": int(va_rows.size),
    }


def summarise_delay_tests(metrics: dict, baseline: str = "B", floor: float = 0.0092) -> dict:
    """Per-variant ``{auc, drop, exceeds_floor}`` relative to ``metrics[baseline]``.

    ``drop = baseline_auc - variant_auc``; None when either is unmeasurable.
    """
    base = (metrics.get(baseline) or {}).get("auc")
    out = {}
    for name, m in metrics.items():
        if name == baseline or not name.startswith(baseline + "_"):
            continue
        auc = (m or {}).get("auc")
        d = delay_drop(base, auc)
        out[name] = {"auc": auc, "drop": d, "exceeds_floor": exceeds_floor(d, floor)}
    return out


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class MiniRocketBaseline:
    """MiniROCKET transform + standardised ridge, alpha chosen on parent groups.

    ``transformer_factory`` (no-arg callable returning an object with
    ``fit(X)`` / ``transform(X)``) exists so the plumbing can be tested without
    aeon; left as None it builds aeon's ``MiniRocket(n_kernels, n_jobs,
    random_state=seed)``. The transform is fit on the training inputs it will
    be applied to (its biases are quantiles of training convolutions) and
    applied in ``chunk_size`` slices so the float32 feature matrix is the only
    full-size allocation.
    """

    def __init__(self, n_kernels: int = 10_000, seed: int = 0, n_jobs: int = 1,
                 alphas: Sequence[float] = DEFAULT_ALPHAS, inner_frac: float = 0.2,
                 chunk_size: int = 4096,
                 transformer_factory: Callable[[], object] | None = None) -> None:
        self.n_kernels = int(n_kernels)
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.alphas = tuple(float(a) for a in alphas)
        self.inner_frac = float(inner_frac)
        self.chunk_size = int(chunk_size)
        self.transformer_factory = transformer_factory
        self.transformer_ = None
        self.ridge_: RidgePath | None = None
        self.alpha_: float | None = None
        self.selection_: dict | None = None
        self.n_features_: int | None = None

    def _make_transformer(self):
        if self.transformer_factory is not None:
            return self.transformer_factory()
        from aeon.transformations.collection.convolution_based import MiniRocket

        return MiniRocket(n_kernels=self.n_kernels, n_jobs=self.n_jobs,
                          random_state=self.seed)

    def transform(self, X) -> np.ndarray:
        """Features ``[n, p]`` float32, computed ``chunk_size`` cases at a time."""
        X = np.asarray(X, dtype=np.float32)
        out = None
        for s in range(0, X.shape[0], self.chunk_size):
            f = np.asarray(self.transformer_.transform(X[s : s + self.chunk_size]),
                           dtype=np.float32)
            if out is None:
                out = np.empty((X.shape[0], f.shape[1]), dtype=np.float32)
            out[s : s + f.shape[0]] = f
        if out is None:
            return np.zeros((0, self.n_features_ or 0), dtype=np.float32)
        return out

    def fit(self, X, y, groups, file_ids, sample_weight=None) -> "MiniRocketBaseline":
        """Fit transform, choose alpha on an inner group split, refit on all rows.

        ``y`` may be {0,1} or {-1,+1}; it is mapped to +-1 for the ridge.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        ypm = np.where(y > 0, 1.0, -1.0)
        self.transformer_ = self._make_transformer().fit(X)
        F = self.transform(X)
        self.n_features_ = int(F.shape[1])
        self.selection_ = choose_alpha(F, ypm, file_ids, groups, self.alphas,
                                       frac=self.inner_frac, seed=self.seed,
                                       chunk_rows=self.chunk_size,
                                       sample_weight=sample_weight)
        self.alpha_ = self.selection_["alpha"]
        self.ridge_ = fit_ridge_path(F, ypm, [self.alpha_], chunk_rows=self.chunk_size,
                                     sample_weight=sample_weight)
        return self

    def decision_from_features(self, F) -> np.ndarray:
        return self.ridge_.decision(F, 0, chunk_rows=self.chunk_size)

    def decision_function(self, X) -> np.ndarray:
        """Ridge decision per input case (float64)."""
        return self.decision_from_features(self.transform(X))
