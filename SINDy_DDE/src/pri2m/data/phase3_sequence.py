"""Phase 3 sequence data layer (Task 1A).

Turns the per-file NPZ signal bundles + the Phase 3 table into windowed,
optionally normalized PyTorch samples for the TKAN / MLP severity heads.

Design rules baked in here:

* Windowing is a pure function (:func:`iter_windows`); the channel order in
  :class:`WindowConfig` defines the column order of every ``[T, C]`` signal.
* Per-channel normalization (:class:`NormStats`) is fit on **training-fold
  windows only**; the validation loader reuses the train stats, so validation
  per-channel means are generally not ~0.
* The kinematic delay ``tau = round(fs / tpf)`` is precomputed per file.
* Folds are parent-group keyed; :func:`build_fold_loaders` calls
  :func:`pri2m.phase3.splits.assert_no_mirror_leakage` before returning so a
  clean dataset and its noisy mirrors can never straddle the split.
* No torch RNG is touched at import time; the train sampler uses a seeded
  ``torch.Generator`` so two builds with the same seed match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from pri2m.phase3.splits import assert_no_mirror_leakage, parent_group

__all__ = [
    "WindowConfig",
    "NormStats",
    "tau_samples",
    "iter_windows",
    "load_npz_signals",
    "Phase3SequenceDataset",
    "collate_windows",
    "build_fold_loaders",
    "aggregate_file_predictions",
]


# --------------------------------------------------------------------------- #
# Config + normalization
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window geometry and channel order.

    ``channels`` defines the column order of every ``[T, C]`` signal produced by
    :func:`load_npz_signals` and consumed by :func:`iter_windows`.
    """

    length: int = 1024
    stride: int = 512
    channels: tuple[str, ...] = ("xtpos", "ytpos", "Forcex", "Forcey")


@dataclass
class NormStats:
    """Per-channel mean/std for window normalization.

    Fit on training-fold windows only; ``apply`` broadcasts over the last
    (channel) axis. ``std`` is floored at ``1e-8`` to avoid divide-by-zero on
    constant channels.
    """

    mean: np.ndarray  # shape [C]
    std: np.ndarray  # shape [C]

    _STD_FLOOR = 1e-8

    @classmethod
    def fit(cls, windows: np.ndarray) -> "NormStats":
        """Fit per-channel statistics over windows of shape ``[N, L, C]``.

        Mean and std are taken over the sample and time axes ``(0, 1)``, giving
        one value per channel. ``std`` is floored at ``1e-8``.
        """
        windows = np.asarray(windows, dtype=np.float64)
        if windows.ndim != 3:
            raise ValueError(f"windows must be [N, L, C], got shape {windows.shape}")
        mean = windows.mean(axis=(0, 1))
        std = windows.std(axis=(0, 1))
        std = np.maximum(std, cls._STD_FLOOR)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    @classmethod
    def fit_streaming(cls, window_arrays: Iterable[np.ndarray]) -> "NormStats":
        """Fit per-channel stats over many ``[N_i, L, C]`` arrays incrementally.

        Numerically identical to :meth:`fit` on the fully concatenated array
        (population mean/std over axes ``(0, 1)``), but holds only one array in
        float64 at a time instead of a single concatenated copy of every window
        — the difference between fitting in a few GB and OOM-ing on a full fold.
        Two passes (mean, then variance) keep it exact; references are reused, so
        no window data is copied.
        """
        arrays = [np.asarray(w) for w in window_arrays if np.asarray(w).size]
        if not arrays:
            raise ValueError("no windows to fit normalization")
        C = arrays[0].shape[-1]
        total = np.zeros(C, dtype=np.float64)
        count = 0
        for w in arrays:
            flat = w.reshape(-1, C).astype(np.float64)
            total += flat.sum(axis=0)
            count += flat.shape[0]
        mean = total / count
        sq = np.zeros(C, dtype=np.float64)
        for w in arrays:
            flat = w.reshape(-1, C).astype(np.float64)
            sq += ((flat - mean) ** 2).sum(axis=0)
        std = np.maximum(np.sqrt(sq / count), cls._STD_FLOOR)
        return cls(mean=mean, std=std)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Return ``(x - mean) / std`` broadcasting over the last axis.

        The result preserves a floating input's dtype: a float32 window
        normalizes in float32 (no float64 intermediate), which is what the data
        loader's hot path wants since the model consumes float32. Non-float
        inputs are promoted to float64 as before.
        """
        x = np.asarray(x)
        out_dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
        mean = self.mean.astype(out_dtype, copy=False)
        std = self.std.astype(out_dtype, copy=False)
        return (x.astype(out_dtype, copy=False) - mean) / std

    def to_dict(self) -> dict:
        """Serialize to ``{"mean": [float,...], "std": [float,...]}``.

        Values are plain Python floats (not ``np.float64``), so the result is
        JSON-serializable for the per-fold ``preprocessing.json`` a checkpoint
        needs to recover its z-score on new data.
        """
        return {
            "mean": [float(v) for v in np.asarray(self.mean).reshape(-1)],
            "std": [float(v) for v in np.asarray(self.std).reshape(-1)],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        """Reconstruct a :class:`NormStats` from :meth:`to_dict` output.

        Round-trips exactly: ``mean``/``std`` are rebuilt as float64 arrays
        matching the constructor's dtype.
        """
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
        )


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def tau_samples(fs: float, tpf: float) -> int:
    """Kinematic delay in samples: ``max(1, round(fs / tpf))``.

    Raises ``ValueError`` if ``fs`` or ``tpf`` is non-positive.
    """
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs}")
    if tpf <= 0:
        raise ValueError(f"tpf must be > 0, got {tpf}")
    return max(1, int(round(fs / tpf)))


def iter_windows(signal: np.ndarray, cfg: WindowConfig) -> np.ndarray:
    """Slice a ``[T, C]`` signal into ``[N, L, C]`` windows.

    Window starts are ``range(0, T - L + 1, stride)``. If ``T < L`` no window
    fits and the result has shape ``[0, L, C]``.
    """
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"signal must be [T, C], got shape {signal.shape}")
    T, C = signal.shape
    L, S = cfg.length, cfg.stride
    starts = range(0, T - L + 1, S) if T >= L else range(0)
    windows = [signal[s : s + L] for s in starts]
    if not windows:
        return np.empty((0, L, C), dtype=signal.dtype)
    return np.stack(windows, axis=0)


def load_npz_signals(npz_path, channels) -> tuple[np.ndarray, dict]:
    """Load a per-file NPZ bundle into a ``[T, C]`` array + a python-scalar meta dict.

    Columns of the returned array follow ``channels`` order. NPZ scalar arrays
    are 0-d; they are converted to python scalars (``source_file`` to ``str``).
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    cols = [np.asarray(data[ch], dtype=np.float64) for ch in channels]
    signal = np.stack(cols, axis=1)  # [T, C]

    meta = {
        "fs": float(np.asarray(data["fs"]).item()),
        "label": float(np.asarray(data["label"]).item()),
        "omega": float(np.asarray(data["omega"]).item()),
        "tpf": float(np.asarray(data["tpf"]).item()),
        "fn": float(np.asarray(data["fn"]).item()),
        "dataset": int(np.asarray(data["dataset"]).item()),
        "source_file": str(np.asarray(data["source_file"]).item()),
    }
    return signal, meta


def _npz_path_for_row(npz_root: Path, row: pd.Series) -> Path:
    """Locate the NPZ bundle for a table row."""
    stem = Path(str(row["file"])).stem
    return npz_root / f"dataset_{int(row['dataset'])}" / f"{stem}.npz"


def _row_parent_group(row: pd.Series) -> int:
    """Prefer the table's ``parent_group`` column; fall back to computing it."""
    if "parent_group" in row and pd.notna(row["parent_group"]):
        return int(row["parent_group"])
    return int(parent_group(int(row["dataset"])))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class Phase3SequenceDataset(Dataset):
    """A flat index of all windows across the table's files.

    Each ``__getitem__`` returns one window plus its file-level targets and
    metadata. Rows with ``exclude_supervised == True`` are skipped iff
    ``supervised_only`` is set.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        npz_root,
        cfg: WindowConfig,
        norm: "NormStats | None" = None,
        supervised_only: bool = False,
    ) -> None:
        self.npz_root = Path(npz_root)
        self.cfg = cfg
        self.norm = norm
        self.supervised_only = supervised_only

        rows = table
        if supervised_only and "exclude_supervised" in rows.columns:
            rows = rows[~rows["exclude_supervised"].astype(bool)]
        # Keep a clean copy of the retained rows (positional reset) for callers
        # that inspect ``dataset.table`` (e.g. leakage assertions).
        self.table = rows.reset_index(drop=True)

        # Build a flat (file_record, window_index) index. Each file is loaded
        # once here; windows are sliced lazily-ish per item from a cached array.
        self._index: list[tuple[int, int]] = []
        self._files: list[dict] = []
        for _, row in self.table.iterrows():
            npz_path = _npz_path_for_row(self.npz_root, row)
            signal, meta = load_npz_signals(npz_path, cfg.channels)
            windows = iter_windows(signal, cfg)
            n = windows.shape[0]
            if n == 0:
                continue
            file_record = {
                # float32 storage: the model consumes float32 (see __getitem__),
                # so float64 here only doubled resident memory.
                "windows": windows.astype(np.float32),
                "b_over_blim": _safe_float(row.get("b_over_blim", np.nan)),
                "label": float(meta["label"]),
                "sample_weight": float(row.get("sample_weight", 1.0)),
                "tau": tau_samples(meta["fs"], meta["tpf"]),
                "dataset": int(meta["dataset"]),
                "file": str(row["file"]),
                "parent_group": _row_parent_group(row),
                "mode_label": str(row.get("mode_label", "")),
                "omega": float(meta["omega"]),
                "tpf": float(meta["tpf"]),
                "fn": float(meta["fn"]),
                "fs": float(meta["fs"]),
            }
            fi = len(self._files)
            self._files.append(file_record)
            self._index.extend((fi, wi) for wi in range(n))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        fi, wi = self._index[i]
        rec = self._files[fi]
        window = rec["windows"][wi]  # [L, C]
        if self.norm is not None:
            window = self.norm.apply(window)
        window_t = torch.as_tensor(np.ascontiguousarray(window), dtype=torch.float32)
        return {
            "window": window_t,
            "b_over_blim": rec["b_over_blim"],
            "label": rec["label"],
            "sample_weight": rec["sample_weight"],
            "tau": rec["tau"],
            "dataset": rec["dataset"],
            "file": rec["file"],
            "parent_group": rec["parent_group"],
            "mode_label": rec["mode_label"],
            "omega": rec["omega"],
            "tpf": rec["tpf"],
            "fn": rec["fn"],
            "fs": rec["fs"],
        }


def _safe_float(value) -> float:
    """Coerce to float, mapping NaN/None to 0.0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(v) else v


# --------------------------------------------------------------------------- #
# Collate
# --------------------------------------------------------------------------- #
def collate_windows(batch: list[dict]) -> dict:
    """Stack a list of dataset items into a batch.

    ``window`` -> ``[B, L, C]`` float32; ``b_over_blim``/``label``/
    ``sample_weight`` -> float tensors ``[B]``; ``tau`` -> long tensor ``[B]``;
    ``dataset``/``file``/``parent_group``/``mode_label`` -> python lists.
    """
    window = torch.stack([b["window"] for b in batch], dim=0).to(torch.float32)
    b_over_blim = torch.tensor([b["b_over_blim"] for b in batch], dtype=torch.float32)
    label = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    sample_weight = torch.tensor([b["sample_weight"] for b in batch], dtype=torch.float32)
    tau = torch.tensor([b["tau"] for b in batch], dtype=torch.long)
    return {
        "window": window,
        "b_over_blim": b_over_blim,
        "label": label,
        "sample_weight": sample_weight,
        "tau": tau,
        "dataset": [b["dataset"] for b in batch],
        "file": [b["file"] for b in batch],
        "parent_group": [b["parent_group"] for b in batch],
        "mode_label": [b["mode_label"] for b in batch],
        "omega": torch.tensor([b["omega"] for b in batch], dtype=torch.float32),
        "tpf": torch.tensor([b["tpf"] for b in batch], dtype=torch.float32),
        "fn": torch.tensor([b["fn"] for b in batch], dtype=torch.float32),
        "fs": torch.tensor([b["fs"] for b in batch], dtype=torch.float32),
    }


# --------------------------------------------------------------------------- #
# Fold loaders
# --------------------------------------------------------------------------- #
def build_fold_loaders(
    table: pd.DataFrame,
    folds_json: dict,
    fold_k: int,
    npz_root,
    cfg: WindowConfig,
    batch_size: int,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, NormStats]:
    """Build train/val loaders for one parent-group fold.

    Validation rows are those whose ``parent_group`` maps to ``fold_k`` via
    ``folds_json["group_to_fold"]`` (keys coerced to int); training rows are the
    rest. :class:`NormStats` is fit on TRAIN windows only and shared with both
    datasets. :func:`assert_no_mirror_leakage` is called before returning.

    Returns ``(train_loader, val_loader, norm_stats)``. The train loader shuffles
    with a seeded ``torch.Generator``; the val loader does not shuffle.

    ``num_workers`` parallelizes per-window normalization/collation across CPU
    processes (the in-memory window cache is shared copy-on-write on fork, so
    this does not multiply resident memory for read-only window arrays).
    ``pin_memory`` speeds host->GPU transfer and should be enabled only when
    training on CUDA. Both default off to preserve the single-process behavior.
    """
    group_to_fold = {int(k): int(v) for k, v in folds_json["group_to_fold"].items()}
    pg = table["parent_group"].astype(int) if "parent_group" in table.columns else parent_group(table["dataset"])
    fold_of_row = pg.astype(int).map(group_to_fold)

    val_mask = fold_of_row == fold_k
    val_table = table[val_mask]
    train_table = table[~val_mask]

    # Leakage guard before anything else touches the data.
    assert_no_mirror_leakage(train_table, val_table)

    # Build the train dataset ONCE, then fit normalization on TRAIN windows only
    # straight from its already-loaded cache. This avoids a throwaway second load
    # of every train file and a full concatenated copy of all windows — both of
    # which previously OOM-killed full folds on a 32 G node.
    train_ds = Phase3SequenceDataset(train_table, npz_root, cfg, norm=None)
    norm = NormStats.fit_streaming(rec["windows"] for rec in train_ds._files)
    train_ds.norm = norm
    val_ds = Phase3SequenceDataset(val_table, npz_root, cfg, norm=norm)

    generator = torch.Generator()
    generator.manual_seed(seed)
    # persistent_workers avoids re-forking the in-memory dataset every epoch when
    # workers are used; it is invalid (and pointless) with num_workers == 0.
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_windows,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_windows,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, norm


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate_file_predictions(window_pred: np.ndarray, file_ids: Sequence) -> dict:
    """Mean of window-level predictions grouped by file id."""
    window_pred = np.asarray(window_pred, dtype=np.float64).reshape(-1)
    file_ids = list(file_ids)
    if len(window_pred) != len(file_ids):
        raise ValueError(
            f"window_pred ({len(window_pred)}) and file_ids ({len(file_ids)}) length mismatch"
        )
    sums: dict = {}
    counts: dict = {}
    for fid, p in zip(file_ids, window_pred):
        sums[fid] = sums.get(fid, 0.0) + float(p)
        counts[fid] = counts.get(fid, 0) + 1
    return {fid: sums[fid] / counts[fid] for fid in sums}
