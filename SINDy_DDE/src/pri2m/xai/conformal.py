"""Split-conformal calibration for the Phase 3 chatter head.

The 121-125 held-out eval exposed a soft spot: a raw ``0.5`` threshold on the
chatter probability gives accuracy that swings run-to-run, because the sigmoid
outputs are *uncalibrated* -- a probability of 0.8 does not mean "right 80% of
the time". Split-conformal prediction repairs this with a **distribution-free,
finite-sample guarantee**: split a held-out set into a calibration half and a
test half; the conformal prediction set built from the calibration scores
contains the true label with probability at least ``1 - alpha`` on the test
half, *no matter how mis-calibrated the underlying model is* (the only
assumption is exchangeability of the two halves).

The construction here is the standard split-conformal classifier over the two
classes ``{stable=0, chatter=1}``:

* the nonconformity score of a labeled example is ``1 - p_true`` (one minus the
  model's probability of the *true* class);
* the conformal threshold ``q`` is the ``ceil((n+1)(1-alpha))/n`` empirical
  quantile of the calibration scores;
* a class ``k`` is in a window's prediction set iff its score ``1 - p_k <= q``.

From the set we read off:

* **calibrated confidence** -- the conformal p-value of the predicted class,
  a monotone-but-*calibrated* transform of the raw probability;
* **an abstention rule** -- withhold a decision on windows whose set is
  *ambiguous* (contains BOTH classes), i.e. the model cannot rule either out at
  the requested coverage; acting only on the confident singletons stabilizes the
  accuracy the raw threshold could not.

A :func:`ensemble_probs` helper averages the five cross-fold checkpoints' chatter
probabilities into a single, lower-variance score to calibrate on.

Pure ``numpy`` -- no torch, no sklearn needed. Heavy/optional plotting deps are
imported lazily inside :func:`plot_coverage_abstention`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Cross-fold ensemble score
# --------------------------------------------------------------------------- #
def ensemble_probs(probs, weights=None) -> np.ndarray:
    """Average the per-checkpoint chatter probabilities into one score.

    Parameters
    ----------
    probs:
        Either a sequence of ``n_folds`` arrays each shape ``[N]`` (one per
        checkpoint), or a 2-D array ``[n_folds, N]``.
    weights:
        Optional per-fold weights ``[n_folds]`` (e.g. each fold's val-AUC); the
        result is the weighted mean down the fold axis. ``None`` -> uniform.

    Returns
    -------
    np.ndarray
        The ``[N]`` ensemble chatter probability (a plain or weighted mean over
        the folds). Averaging the folds cuts the per-checkpoint variance that
        drove the run-to-run threshold swings.
    """
    stack = np.asarray(probs, dtype=float)
    if stack.ndim == 1:
        return stack.copy()
    if stack.ndim != 2:
        raise ValueError(f"probs must be [n_folds, N] or a list of [N]; got {stack.ndim}D")
    if weights is None:
        return stack.mean(axis=0)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != stack.shape[0]:
        raise ValueError(f"weights length {w.shape[0]} != n_folds {stack.shape[0]}")
    w = w / w.sum()
    return (stack * w[:, None]).sum(axis=0)


# --------------------------------------------------------------------------- #
# Nonconformity + conformal threshold
# --------------------------------------------------------------------------- #
def nonconformity(probs, labels) -> np.ndarray:
    """Split-conformal nonconformity scores ``s = 1 - p_true``.

    ``probs`` is the chatter probability ``p = P(y=1)`` per window; ``labels`` is
    the binary truth. The score of the *true* class is ``1 - p`` when ``y == 1``
    and ``1 - (1 - p) = p`` when ``y == 0`` -- small when the model is confidently
    right about the truth, large when it is confidently wrong.
    """
    probs = np.asarray(probs, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(f"probs length {probs.shape[0]} != labels length {labels.shape[0]}")
    p_true = np.where(labels == 1, probs, 1.0 - probs)
    return 1.0 - p_true


def conformal_quantile(cal_scores, alpha: float) -> float:
    """The split-conformal threshold ``q`` at miscoverage ``alpha``.

    ``q`` is the ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile of the
    calibration nonconformity scores (the finite-sample correction that yields
    marginal coverage ``>= 1 - alpha``). When that level exceeds 1 (tiny ``alpha``
    relative to ``n``) there is no finite threshold and ``+inf`` is returned, so
    every set contains both classes.

    Parameters
    ----------
    cal_scores:
        Calibration nonconformity scores ``[n]`` (see :func:`nonconformity`).
    alpha:
        Target miscoverage in ``(0, 1)`` -- coverage is ``1 - alpha``.
    """
    s = np.asarray(cal_scores, dtype=float).reshape(-1)
    n = s.shape[0]
    if n == 0:
        raise ValueError("need at least one calibration score")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(s, level, method="higher"))


def prediction_sets(probs, q: float) -> np.ndarray:
    """Conformal prediction sets for each window, as a boolean ``[N, 2]``.

    Column 0 is class ``stable`` (label 0), column 1 is class ``chatter``
    (label 1). Class ``k`` is included iff its nonconformity score ``1 - p_k <=
    q``. Membership is *monotone* in ``q``: a larger threshold only ever ADDS
    labels, so higher coverage yields larger (never smaller) sets.

    Parameters
    ----------
    probs:
        Chatter probability ``p = P(y=1)`` per window ``[N]``.
    q:
        The conformal threshold from :func:`conformal_quantile`.
    """
    p1 = np.asarray(probs, dtype=float).reshape(-1)
    p0 = 1.0 - p1
    inc0 = (1.0 - p0) <= q  # stable in set iff p0 >= 1 - q
    inc1 = (1.0 - p1) <= q  # chatter in set iff p1 >= 1 - q
    return np.stack([inc0, inc1], axis=1)


# --------------------------------------------------------------------------- #
# The calibrated conformal predictor
# --------------------------------------------------------------------------- #
@dataclass
class SplitConformal:
    """A calibrated split-conformal predictor for the chatter head.

    Build one with :meth:`calibrate` on a held-out calibration split, then apply
    it to fresh test scores for prediction sets, an abstention mask, calibrated
    confidences, and a coverage/abstention/selective-accuracy report.

    Attributes
    ----------
    alpha:
        Target miscoverage (coverage ``1 - alpha``).
    q:
        The conformal threshold (see :func:`conformal_quantile`).
    cal_scores:
        The sorted calibration nonconformity scores, kept for the per-window
        conformal p-value (:meth:`calibrated_confidence`).
    """

    alpha: float
    q: float
    cal_scores: np.ndarray

    @classmethod
    def calibrate(cls, cal_probs, cal_labels, alpha: float) -> "SplitConformal":
        """Fit ``q`` on a calibration split of ``(chatter prob, label)`` pairs."""
        s = nonconformity(cal_probs, cal_labels)
        q = conformal_quantile(s, alpha)
        return cls(alpha=float(alpha), q=q, cal_scores=np.sort(s))

    # -- sets / decisions ---------------------------------------------------- #
    def prediction_sets(self, probs) -> np.ndarray:
        """Boolean ``[N, 2]`` conformal sets (see module-level :func:`prediction_sets`)."""
        return prediction_sets(probs, self.q)

    def set_sizes(self, probs) -> np.ndarray:
        """Number of classes in each window's conformal set ``[N]`` (0, 1, or 2)."""
        return self.prediction_sets(probs).sum(axis=1)

    def abstain_mask(self, probs) -> np.ndarray:
        """``True`` where the set is AMBIGUOUS (both classes) -> withhold a decision.

        At the coverage levels of interest (``1 - alpha >= ~0.8``) the threshold
        ``q >= 0.5`` so empty sets do not arise and every non-abstained window is
        a confident singleton. The ambiguous-set rate is monotone in the coverage
        demand, which is the behavior the abstention rule promises.
        """
        return self.set_sizes(probs) >= 2

    def point_predictions(self, probs) -> np.ndarray:
        """Argmax-probability label per window ``[N]`` (used on committed windows)."""
        return (np.asarray(probs, dtype=float).reshape(-1) >= 0.5).astype(int)

    def calibrated_confidence(self, probs) -> np.ndarray:
        """Per-window calibrated confidence = conformal p-value of the predicted class.

        For the predicted class ``yhat = argmax(p)`` with nonconformity score
        ``s = 1 - p_yhat``, the p-value is ``(1 + #{cal_scores >= s}) / (n + 1)``
        -- the calibrated analogue of the raw probability. It is monotone in the
        model score but expressed on the calibration distribution's scale, so it
        is comparable across folds/runs in a way the raw sigmoid is not.
        """
        p1 = np.asarray(probs, dtype=float).reshape(-1)
        p_pred = np.maximum(p1, 1.0 - p1)  # prob of the predicted (argmax) class
        s_pred = 1.0 - p_pred
        n = self.cal_scores.shape[0]
        # #{cal >= s} via the sorted calibration scores.
        idx = np.searchsorted(self.cal_scores, s_pred, side="left")
        ge = n - idx
        return (1.0 + ge) / (n + 1.0)

    # -- evaluation ---------------------------------------------------------- #
    def evaluate(self, probs, labels) -> dict:
        """Coverage / abstention / selective-accuracy on a fresh test split.

        Returns
        -------
        dict
            ``{"alpha", "target_coverage", "q", "coverage", "abstention_rate",
            "empty_rate", "commit_rate", "selective_accuracy", "raw_accuracy",
            "mean_set_size", "n"}``. ``coverage`` is the fraction of windows whose
            TRUE label lies in the conformal set (the guaranteed ``>= 1 - alpha``
            quantity); ``abstention_rate`` is the fraction of ambiguous sets;
            ``selective_accuracy`` is the point-prediction accuracy on the
            committed (singleton) windows; ``raw_accuracy`` is the plain
            0.5-threshold accuracy over ALL windows for comparison.
        """
        probs = np.asarray(probs, dtype=float).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        sets = self.prediction_sets(probs)
        sizes = sets.sum(axis=1)

        covered = sets[np.arange(len(labels)), labels]
        preds = self.point_predictions(probs)
        committed = sizes == 1
        n_commit = int(committed.sum())
        sel_acc = float((preds[committed] == labels[committed]).mean()) if n_commit else float("nan")

        return {
            "alpha": float(self.alpha),
            "target_coverage": float(1.0 - self.alpha),
            "q": float(self.q),
            "coverage": float(covered.mean()),
            "abstention_rate": float((sizes >= 2).mean()),
            "empty_rate": float((sizes == 0).mean()),
            "commit_rate": float(committed.mean()),
            "selective_accuracy": sel_acc,
            "raw_accuracy": float((preds == labels).mean()),
            "mean_set_size": float(sizes.mean()),
            "n": int(len(labels)),
        }


# --------------------------------------------------------------------------- #
# Coverage / abstention sweep over an alpha grid
# --------------------------------------------------------------------------- #
def coverage_abstention_sweep(cal_probs, cal_labels, test_probs, test_labels, alphas) -> dict:
    """Calibrate at each ``alpha`` and report coverage + abstention on the test split.

    Parameters
    ----------
    cal_probs, cal_labels:
        Calibration-split chatter probabilities and labels.
    test_probs, test_labels:
        Fresh test-split probabilities and labels (disjoint from calibration).
    alphas:
        Iterable of target miscoverages.

    Returns
    -------
    dict
        ``{"alpha": [...], "target_coverage": [...], "coverage": [...],
        "abstention_rate": [...], "selective_accuracy": [...], "commit_rate":
        [...], "q": [...], "raw_accuracy": float, "n_cal": int, "n_test": int}``
        with the per-alpha lists aligned elementwise. ``raw_accuracy`` is the
        alpha-independent 0.5-threshold test accuracy (the baseline the selective
        accuracy improves on).
    """
    alpha_list = [float(a) for a in alphas]
    rows = []
    for a in alpha_list:
        conf = SplitConformal.calibrate(cal_probs, cal_labels, alpha=a)
        rows.append(conf.evaluate(test_probs, test_labels))
    preds = (np.asarray(test_probs, dtype=float).reshape(-1) >= 0.5).astype(int)
    raw_acc = float((preds == np.asarray(test_labels).reshape(-1).astype(int)).mean())
    return {
        "alpha": alpha_list,
        "target_coverage": [r["target_coverage"] for r in rows],
        "coverage": [r["coverage"] for r in rows],
        "abstention_rate": [r["abstention_rate"] for r in rows],
        "selective_accuracy": [r["selective_accuracy"] for r in rows],
        "commit_rate": [r["commit_rate"] for r in rows],
        "q": [r["q"] for r in rows],
        "raw_accuracy": raw_acc,
        "n_cal": int(np.asarray(cal_probs).reshape(-1).shape[0]),
        "n_test": int(np.asarray(test_probs).reshape(-1).shape[0]),
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot_coverage_abstention(sweep: dict, path: str) -> str:
    """Plot a :func:`coverage_abstention_sweep` result to ``path``.

    Two panels: (left) empirical coverage vs the target ``1 - alpha`` with the
    ideal ``y = x`` reference -- the conformal guarantee is that the markers sit
    on/above the diagonal; (right) abstention rate and selective accuracy vs the
    target coverage, showing the accuracy-for-abstention trade the rule buys.

    Returns
    -------
    str
        ``path`` (the figure is written there).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = np.asarray(sweep["target_coverage"], dtype=float)
    cov = np.asarray(sweep["coverage"], dtype=float)
    abst = np.asarray(sweep["abstention_rate"], dtype=float)
    sel = np.asarray(sweep["selective_accuracy"], dtype=float)

    order = np.argsort(target)
    target, cov, abst, sel = target[order], cov[order], abst[order], sel[order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    lo = float(min(target.min(), cov.min())) - 0.02
    ax.plot([lo, 1.0], [lo, 1.0], color="0.6", linestyle="--", label="ideal (y=x)")
    ax.plot(target, cov, marker="o", color="tab:blue", label="empirical coverage")
    ax.set_xlabel("target coverage (1 - alpha)")
    ax.set_ylabel("empirical coverage on test split")
    ax.set_title("Conformal coverage guarantee")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(target, abst, marker="s", color="tab:red", label="abstention rate")
    ax.plot(target, sel, marker="^", color="tab:green", label="selective accuracy")
    raw = sweep.get("raw_accuracy")
    if raw is not None:
        ax.axhline(float(raw), color="0.5", linestyle=":", label=f"raw 0.5-thresh acc ({raw:.3f})")
    ax.set_xlabel("target coverage (1 - alpha)")
    ax.set_ylabel("rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Abstention vs selective accuracy")
    ax.legend(fontsize=8)

    fig.tight_layout()
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
