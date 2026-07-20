"""Hand-rolled delay-embedded sparse regression for regenerative-delay recovery.

This is the SINDy-DDE cross-check: recover the regenerative (tooth-pass) delay
``tau`` AND a delayed-differential regenerative term directly from a raw cutting
signal ``x(t)`` with NO neural network -- an independent white-box estimate to
compare against the physical tooth-pass period ``1/tpf`` and the TKAN model's
delay-ablation result.

``pysindy`` has no native delay term, so we build the delay columns by hand:

1. :func:`finite_difference` estimates the target derivative (velocity for a
   first-order system, acceleration ``xddot`` for the second-order milling
   model) by central differences.
2. :func:`build_delay_library` assembles, for a *trial* delay ``tau`` (in
   samples), a candidate feature matrix with the delayed term ``x(t-tau)``, the
   regenerative term ``x(t) - x(t-tau)``, velocity (for a 2nd-order target), and
   low-order polynomials of ``x`` (and, optionally, of the regenerative term).
3. :func:`stlsq` fits the target with **sequentially-thresholded least squares**
   (ridge-regularized, column-standardized so the threshold is scale-free),
   driving small coefficients to exactly zero.
4. :func:`tau_scan` sweeps ``tau`` over a grid bracketing the physical tooth-pass
   value, fits STLSQ at each ``tau`` on a *common* interior support (so the
   residuals are comparable across the grid), and selects ``tau*`` as the
   minimizer of the sparse-regression residual. It reports the residual-vs-tau
   curve and whether the regenerative term survived with a nonzero coefficient.

Reference: delay-aware SINDy, arXiv:2407.19640. No ``pysindy`` dependency.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "finite_difference",
    "delay_embed",
    "build_delay_library",
    "stlsq",
    "tau_scan",
    "physical_delay_samples",
]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def finite_difference(x: np.ndarray, dt: float, order: int = 1) -> np.ndarray:
    """Central finite-difference derivative of ``x`` sampled at spacing ``dt``.

    ``order=1`` returns velocity ``(x[t+1]-x[t-1]) / (2 dt)``; ``order=2`` returns
    acceleration ``(x[t+1]-2 x[t]+x[t-1]) / dt**2``. The result has the same
    length as ``x``; the first/last sample of each applied difference is filled by
    replication so callers can slice a valid interior. Both formulas are exact on
    the interior for polynomials up to the matching degree.
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    out = np.zeros_like(x)
    if order == 1:
        out[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        out[0], out[-1] = out[1], out[-2]
    else:
        out[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (dt * dt)
        out[0], out[-1] = out[1], out[-2]
    return out


def delay_embed(x: np.ndarray, tau: int) -> np.ndarray:
    """Return the delayed signal ``x(t - tau)`` (zero-padded for ``t < tau``).

    ``tau`` is a non-negative integer number of samples; ``tau=0`` is the
    identity. ``result[t] == x[t - tau]`` for ``t >= tau`` and ``0`` before.
    """
    x = np.asarray(x, dtype=np.float64)
    tau = int(tau)
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    if tau == 0:
        return x.copy()
    out = np.zeros_like(x)
    out[tau:] = x[:-tau]
    return out


def physical_delay_samples(fs: float, tpf: float) -> int:
    """Kinematic tooth-pass delay in samples: ``max(1, round(fs / tpf))``.

    Mirrors :func:`pri2m.data.phase3_sequence.tau_samples` so the cross-check's
    grid can bracket the same physical value the sequence model uses.
    """
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs}")
    if tpf <= 0:
        raise ValueError(f"tpf must be > 0, got {tpf}")
    return max(1, int(round(fs / tpf)))


# --------------------------------------------------------------------------- #
# Candidate library
# --------------------------------------------------------------------------- #
def build_delay_library(
    x: np.ndarray,
    tau: int,
    dt: float,
    *,
    deriv_order: int = 2,
    poly_order: int = 3,
    regen_poly_order: int = 1,
    include_delayed: bool = True,
    include_velocity: bool | None = None,
    extra_columns: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the delay-embedded candidate feature matrix at a trial delay ``tau``.

    Columns (in order):

    * ``"1"``                          -- constant / bias,
    * ``"x"``                          -- instantaneous position,
    * ``"xdot"``                       -- velocity (only when regressing a 2nd-order
                                          / acceleration target; see
                                          ``include_velocity``),
    * ``"x_delayed"``                  -- the delayed term ``x(t-tau)`` (if
                                          ``include_delayed``),
    * ``"regen"``, ``"regen^2"`` ...   -- the regenerative term ``x(t)-x(t-tau)``
                                          and its powers up to ``regen_poly_order``,
    * ``"x^2"`` ... ``"x^{poly_order}"`` -- low-order polynomials of ``x``,
    * any ``extra_columns`` (e.g. a decoy or a second channel).

    ``x``, ``x_delayed`` and ``regen`` are linearly dependent (``regen = x -
    x_delayed``); that redundancy is intentional (the physical regenerative term
    is one of the named candidates) and is handled by the ridge term in
    :func:`stlsq`. The residual that drives :func:`tau_scan` depends only on the
    column *span*, so the redundancy does not affect delay recovery.

    Returns ``(Theta, names)`` with ``Theta`` of shape ``[len(x), n_terms]``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if include_velocity is None:
        include_velocity = deriv_order >= 2

    n = x.size
    xd = delay_embed(x, tau)
    regen = x - xd

    cols: list[np.ndarray] = [np.ones(n), x]
    names: list[str] = ["1", "x"]
    if include_velocity:
        cols.append(finite_difference(x, dt, order=1))
        names.append("xdot")
    if include_delayed:
        cols.append(xd)
        names.append("x_delayed")
    cols.append(regen)
    names.append("regen")
    for p in range(2, int(regen_poly_order) + 1):
        cols.append(regen**p)
        names.append(f"regen^{p}")
    for p in range(2, int(poly_order) + 1):
        cols.append(x**p)
        names.append(f"x^{p}")
    if extra_columns:
        for name, col in extra_columns.items():
            col = np.asarray(col, dtype=np.float64)
            if col.shape != x.shape:
                raise ValueError(
                    f"extra column {name!r} has shape {col.shape}, expected {x.shape}"
                )
            cols.append(col)
            names.append(str(name))

    return np.column_stack(cols), names


# --------------------------------------------------------------------------- #
# Sequentially-thresholded least squares (STLSQ)
# --------------------------------------------------------------------------- #
def stlsq(
    Theta: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float = 0.05,
    alpha: float = 1e-8,
    max_iter: int = 30,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sequentially-thresholded (ridge) least squares -- the SINDy sparsifier.

    Columns are standardized to unit L2 norm (when ``normalize``) so the single
    ``threshold`` is scale-free across features of wildly different magnitude
    (velocity vs. ``x^3``). Each iteration solves the ridge-regularized normal
    equations over the currently-kept columns, then permanently drops every
    coefficient whose *standardized* magnitude falls below ``threshold``, until
    the support stops changing.

    ``alpha`` regularizes the (possibly rank-deficient, because ``regen = x -
    x_delayed``) design so the solve is stable. Returns ``(coef, keep, residual)``
    where ``coef`` is in raw column units, ``keep`` is the boolean support mask,
    and ``residual`` is the sum of squared residuals on the given rows.
    """
    Theta = np.asarray(Theta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if Theta.ndim != 2 or Theta.shape[0] != y.size:
        raise ValueError(f"Theta {Theta.shape} incompatible with y {y.shape}")

    n_terms = Theta.shape[1]
    if normalize:
        scale = np.linalg.norm(Theta, axis=0)
        scale[scale < 1e-12] = 1.0
    else:
        scale = np.ones(n_terms)
    Th = Theta / scale

    keep = np.ones(n_terms, dtype=bool)
    w_std = np.zeros(n_terms)
    # The delay library is deliberately redundant (regen = x - x_delayed), so at
    # small trial delays x and x_delayed are near-identical and the ridge solve is
    # extremely ill-conditioned. The residual is still correct (the huge redundant
    # coefficients cancel), but the BLAS matmul on some platforms (e.g. macOS
    # Accelerate) sets spurious FPE flags. Suppress those here; we assert the
    # returned residual is finite below so a genuine blow-up still surfaces.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for _ in range(max_iter):
            A = Th[:, keep]
            # Ridge normal equations. With alpha > 0, G = A^T A + alpha I is SPD,
            # so a direct solve is stable; fall back to least-norm lstsq only if
            # the solve is singular (e.g. alpha == 0).
            G = A.T @ A + alpha * np.eye(A.shape[1])
            rhs = A.T @ y
            try:
                wk = np.linalg.solve(G, rhs)
            except np.linalg.LinAlgError:
                wk = np.linalg.lstsq(G, rhs, rcond=None)[0]
            w_full = np.zeros(n_terms)
            w_full[keep] = wk
            new_keep = keep & (np.abs(w_full) >= threshold)
            if new_keep.sum() == 0:
                # Everything thresholded away: keep the last non-empty fit.
                break
            if np.array_equal(new_keep, keep):
                w_std = w_full
                break
            keep = new_keep
            w_std = w_full

        coef = w_std / scale
        resid = y - Theta @ coef
        ss = float(resid @ resid)
    if not np.isfinite(ss):
        raise FloatingPointError("STLSQ residual is non-finite (design blew up)")
    return coef, keep, ss


# --------------------------------------------------------------------------- #
# Delay scan
# --------------------------------------------------------------------------- #
def tau_scan(
    x: np.ndarray,
    dt: float,
    tau_grid: Sequence[int] | np.ndarray,
    *,
    deriv_order: int = 2,
    threshold: float = 0.05,
    alpha: float = 1e-8,
    poly_order: int = 3,
    regen_poly_order: int = 1,
    include_delayed: bool = True,
    extra_columns: Mapping[str, np.ndarray] | None = None,
    valid_margin: int = 2,
) -> dict:
    """Scan the trial delay over ``tau_grid`` and pick ``tau*`` by min residual.

    For every ``tau`` we build :func:`build_delay_library`, regress the target
    derivative (``deriv_order``-th finite difference of ``x``) with :func:`stlsq`
    on a **common interior support** -- rows ``[max(tau_grid) + valid_margin :
    N - valid_margin]`` -- so residuals at different ``tau`` are measured on the
    same samples and are directly comparable. Each residual is normalized by the
    target's sum of squares (fraction of unexplained variance).

    Returns a dict with the full ``residuals`` curve, ``tau_star`` (the argmin),
    the selected ``coef``/``keep``/``names`` at ``tau_star``, and the
    ``regen_selected``/``regen_coef`` verdict on the regenerative term.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    grid = np.asarray(list(tau_grid), dtype=int)
    if grid.size == 0:
        raise ValueError("tau_grid is empty")
    if grid.min() < 0:
        raise ValueError("tau_grid must be non-negative")

    n = x.size
    tmax = int(grid.max())
    lo = tmax + max(valid_margin, deriv_order)
    hi = n - max(valid_margin, deriv_order)
    if hi - lo < 32:
        # Need enough rows for a meaningful regression on the common support.
        raise ValueError(
            f"signal too short for grid: {hi - lo} usable rows "
            f"(len={n}, max tau={tmax})"
        )
    valid = np.zeros(n, dtype=bool)
    valid[lo:hi] = True

    target = finite_difference(x, dt, order=deriv_order)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        denom = float(target[valid] @ target[valid]) + 1e-30

    residuals = np.empty(grid.size)
    per_tau: list[dict] = []
    for j, tau in enumerate(grid):
        Theta, names = build_delay_library(
            x,
            int(tau),
            dt,
            deriv_order=deriv_order,
            poly_order=poly_order,
            regen_poly_order=regen_poly_order,
            include_delayed=include_delayed,
            extra_columns=extra_columns,
        )
        coef, keep, ss = stlsq(
            Theta[valid],
            target[valid],
            threshold=threshold,
            alpha=alpha,
        )
        residuals[j] = ss / denom
        per_tau.append({"coef": coef, "keep": keep, "names": names})

    best = int(np.argmin(residuals))
    tau_star = int(grid[best])
    b = per_tau[best]
    names = b["names"]
    regen_idx = names.index("regen") if "regen" in names else -1
    regen_selected = bool(regen_idx >= 0 and b["keep"][regen_idx])
    regen_coef = float(b["coef"][regen_idx]) if regen_idx >= 0 else 0.0

    return {
        "tau_grid": grid,
        "residuals": residuals,
        "tau_star": tau_star,
        "residual_at_star": float(residuals[best]),
        "names": names,
        "coef": b["coef"],
        "keep": b["keep"],
        "regen_selected": regen_selected,
        "regen_coef": regen_coef,
        "deriv_order": int(deriv_order),
        "threshold": float(threshold),
        "n_valid_rows": int(valid.sum()),
    }
