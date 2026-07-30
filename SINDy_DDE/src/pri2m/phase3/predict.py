"""Per-file chatter inference for a frozen Phase 3 TKAN checkpoint.

This module owns the *one* scoring loop shared by both entry points:

* ``scripts/predict_windows_phase3.py`` -- researcher-facing, takes an explicit
  ``--checkpoint`` (used by the XAI/counterfactual chain);
* ``scripts/predict_phase3.py`` -- team-facing, resolves the checkpoint from the
  published zoo (:mod:`pri2m.phase3.zoo`).

Keeping the loop here means the two cannot drift: a fix to the windowing or the
per-file averaging lands in both. Model *construction* deliberately stays in
``scripts/xai_phase3._load_model`` (its side-input / quantile-severity
reconstruction is exercised by ``tests/test_xai_phase3_norm.py``); callers load
the model there and hand it in.

The z-score and window geometry always come from the checkpoint's own
``preprocessing.json`` and are never refit on the scored data -- refitting would
quietly change what the model sees.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pri2m.data.phase3_sequence import (
    NormStats,
    Phase3SequenceDataset,
    WindowConfig,
    collate_windows,
)
from pri2m.phase3.splits import parent_group

__all__ = [
    "PREDICTION_COLUMNS",
    "annotate_train_overlap",
    "predict_files",
    "window_config_from_preprocessing",
]

#: Column order of the predictions CSV. ``counterfactuals_phase3.py`` merges on
#: ``file`` and picks up ``prob``, so this order is load-bearing downstream.
PREDICTION_COLUMNS = ["file", "prob", "n_windows", "dataset", "label"]


def window_config_from_preprocessing(preprocessing: dict) -> WindowConfig:
    """Rebuild the training :class:`WindowConfig` from a ``preprocessing.json``.

    ``channels`` is the saved channel order (defaults to the WindowConfig default
    so an older manifest without the key still works); ``length`` / ``stride``
    come from the ``window`` block. Matching the training geometry is required so
    the scored windows are sliced the same way the model was trained on.
    """
    window = preprocessing.get("window", {}) or {}
    channels = preprocessing.get("channels")
    kwargs: dict = {}
    if "length" in window:
        kwargs["length"] = int(window["length"])
    if "stride" in window:
        kwargs["stride"] = int(window["stride"])
    if channels:
        kwargs["channels"] = tuple(str(c) for c in channels)
    return WindowConfig(**kwargs)


def predict_files(
    model,
    eval_table: pd.DataFrame,
    npz_root: str,
    preprocessing: dict,
    *,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    """Score every file in ``eval_table``, averaging window probabilities.

    Returns a DataFrame with :data:`PREDICTION_COLUMNS`. ``prob`` is the mean of
    ``sigmoid(chatter_logit)`` over that file's windows, so a file with more
    windows does not outweigh a shorter one.

    ``eval_table`` must already be filtered to rows whose NPZ bundle exists
    (``scripts.eval_heldout_phase3.build_eval_table`` does this). Raises
    ``ValueError`` on an empty table rather than writing an empty CSV that reads
    as "nothing was chattering".

    Side effect: sets ``model.pool`` from ``preprocessing['pool']``. The pooling
    factor is part of the trained geometry -- scoring an unpooled model that was
    trained with ``pool: 4`` silently feeds a 4x longer recurrence -- so it is
    applied here rather than left to each caller to remember.
    """
    if eval_table.empty:
        raise ValueError("eval_table is empty -- nothing to score")
    if "norm" not in preprocessing:
        raise ValueError(
            "preprocessing has no 'norm' block; cannot recover the training "
            "z-score (refitting it here would change what the model sees)"
        )

    model.pool = int(preprocessing.get("pool", 1))
    model.to(device)
    model.eval()

    norm = NormStats.from_dict(preprocessing["norm"])
    cfg = window_config_from_preprocessing(preprocessing)
    dataset = Phase3SequenceDataset(eval_table, npz_root, cfg=cfg, norm=norm)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_windows
    )

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    meta: dict[str, tuple[int, int]] = {}
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            probs = torch.sigmoid(model(batch)["chatter_logit"]).detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()
            # batch["dataset"] entries are already python ints (Phase3SequenceDataset
            # casts via int(meta["dataset"])); int() below is a cheap normalization
            # in case a future loader change makes them strings.
            for f, d, lab, p in zip(batch["file"], batch["dataset"], labels, probs):
                sums[f] = sums.get(f, 0.0) + float(p)
                counts[f] = counts.get(f, 0) + 1
                meta[f] = (int(d), int(lab))

    rows = [
        {
            "file": f,
            "prob": sums[f] / counts[f],
            "n_windows": counts[f],
            "dataset": meta[f][0],
            "label": meta[f][1],
        }
        for f in sums
    ]
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def annotate_train_overlap(
    predictions: pd.DataFrame, folds: dict, fold: int
) -> tuple[pd.DataFrame, int]:
    """Flag rows the given fold was *trained* on; return ``(df, n_overlap)``.

    ``folds`` is a ``phase3_folds.json`` payload. Its ``group_to_fold`` maps a
    parent group to the fold that holds it out for **validation**, so fold ``k``
    trained on every group whose value is not ``k`` -- minus ``dev_groups``,
    which are held out of every fold entirely.

    The added ``in_train_split`` column exists so nobody publishes a number
    computed partly on data the checkpoint memorized. Rows whose group is absent
    from the map (e.g. datasets 121-155, which post-date the split) are ``False``:
    unseen by construction.
    """
    group_to_fold = {int(g): int(k) for g, k in (folds.get("group_to_fold") or {}).items()}
    dev_groups = {int(g) for g in (folds.get("dev_groups") or ())}

    out = predictions.copy()
    groups = parent_group(out["dataset"].astype(int))
    out["in_train_split"] = [
        (int(g) in group_to_fold)
        and (int(g) not in dev_groups)
        and (group_to_fold[int(g)] != int(fold))
        for g in groups
    ]
    return out, int(out["in_train_split"].sum())
