"""Seeded, class-balanced window sampling shared by the eval-only XAI runners."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pri2m.data.phase3_sequence import (
    NormStats,
    Phase3SequenceDataset,
    WindowConfig,
    collate_windows,
)


def batch_to_model_device(batch: dict, model) -> dict:
    """Move a batch's tensors onto whatever device ``model`` currently sits on.

    ``build_labeled_batch`` always returns CPU tensors, but a runner that scores
    a held-out set first has already had its model moved: ``eval_heldout_phase3
    .run`` calls ``model.to(dev)`` (``eval_heldout_phase3.py:170``) as a side
    effect. The model is then on CUDA while the batch is not, and the next
    ``model(batch)`` dies with "Expected all tensors to be on the same device".
    That is how job 57252876 died on 2026-08-12, 13 minutes in.

    The device is read from the model's own parameters rather than from a
    ``--device`` flag, so this stays correct no matter what moved the model or
    when. Non-tensor entries are passed through untouched.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:  # a model with no parameters cannot have been moved
        return batch
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def window_config_from_preprocessing(preprocessing: dict | None) -> WindowConfig:
    """Recover the window geometry a checkpoint was built with.

    ``build_labeled_batch``'s ``cfg`` is REQUIRED -- passing ``None`` reaches
    ``Phase3SequenceDataset`` and dies with ``'NoneType' object has no attribute
    'channels'``. That is how job 57251934 died on 2026-08-12, after the held-out
    baseline had already scored cleanly.

    Reading the geometry from the checkpoint's own ``preprocessing.json`` is
    preferable to taking it from CLI defaults: a mismatch would silently analyse
    windows of a different length or channel order than the network was trained
    on, and nothing downstream would flag it. Falls back to
    :class:`WindowConfig`'s defaults when a block is absent, which is what a
    manifest predating that block means.
    """
    pre = preprocessing or {}
    window = pre.get("window") or {}
    channels = pre.get("channels")
    defaults = WindowConfig()
    return WindowConfig(
        length=int(window.get("length", defaults.length)),
        stride=int(window.get("stride", defaults.stride)),
        channels=tuple(channels) if channels else defaults.channels,
    )

CHANNELS = ("xtpos", "ytpos", "Forcex", "Forcey")


# --------------------------------------------------------------------------- #
# Data: a balanced, normalized, labeled batch of real windows
# --------------------------------------------------------------------------- #
def npz_exists(npz_root: Path, row) -> bool:
    stem = Path(str(row["file"])).stem
    return (npz_root / f"dataset_{int(row['dataset'])}" / f"{stem}.npz").is_file()


def build_labeled_batch(table_csv, npz_root, preprocessing, n_windows, cfg, supervised_only=True, seed=0):
    """Build a class-balanced, normalized batch of real windows + their labels.

    Returns ``(batch, labels)`` where ``batch`` is the ``collate_windows`` dict
    (``window``/``tau``/``label``/``b_over_blim``/``omega``/``tpf``/``fn``/``fs``)
    and ``labels`` is the int chatter label array.

    Selection is **seeded random** at both the file and window level (NOT the first
    files in table order): a deterministic head() draw tends to grab a single
    operating point / easy files, which ceilings the sample AUC and makes
    permutation importances uninterpretable. ``seed`` makes the draw reproducible.
    The result is balanced (``~n_windows//2`` windows per class).
    """
    npz_root = Path(npz_root)
    rng = np.random.default_rng(seed)
    df = pd.read_csv(table_csv)
    df = df[df.apply(lambda r: npz_exists(npz_root, r), axis=1)].copy()
    if supervised_only and "exclude_supervised" in df.columns:
        df = df[~df["exclude_supervised"].astype(bool)]
    if df.empty:
        raise RuntimeError("no usable rows: none of the table's files exist under npz-root")

    norm = NormStats.from_dict(preprocessing["norm"]) if preprocessing and "norm" in preprocessing else None

    # Randomly sample up to n_windows files per class (covers diverse operating
    # points instead of the first few rows), then build the dataset over them.
    def _sample_rows(frame):
        if len(frame) <= n_windows:
            return frame
        pick = rng.choice(len(frame), size=n_windows, replace=False)
        return frame.iloc[pick]

    pos = _sample_rows(df[df["label"] == 1])
    neg = _sample_rows(df[df["label"] == 0])
    selection = pd.concat([neg, pos]).reset_index(drop=True)

    ds = Phase3SequenceDataset(selection, npz_root, cfg, norm=norm, supervised_only=False)
    if len(ds) == 0:
        raise RuntimeError("no windows fit the requested window length on the sampled files")

    # Randomly pick balanced windows per class (reproducible via the same rng).
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for i in range(len(ds)):
        by_label[int(ds[i]["label"])].append(i)
    take = max(1, n_windows // 2)

    def _pick(idxs):
        idxs = list(idxs)
        if len(idxs) <= take:
            return idxs
        sel = rng.choice(len(idxs), size=take, replace=False)
        return [idxs[int(j)] for j in sel]

    chosen = _pick(by_label[0]) + _pick(by_label[1])
    if not chosen:
        chosen = list(range(min(len(ds), n_windows)))

    items = [ds[i] for i in chosen]
    batch = collate_windows(items)
    labels = batch["label"].detach().cpu().numpy().astype(int)
    return batch, labels
