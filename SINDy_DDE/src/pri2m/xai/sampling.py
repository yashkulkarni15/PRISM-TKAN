"""Seeded, class-balanced window sampling shared by the eval-only XAI runners."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pri2m.data.phase3_sequence import NormStats, Phase3SequenceDataset, collate_windows

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
