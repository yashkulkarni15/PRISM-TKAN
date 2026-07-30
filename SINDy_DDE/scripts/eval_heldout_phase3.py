#!/usr/bin/env python3
"""Score a trained Phase 3 checkpoint on a held-out dataset (NO retraining).

We have 5-fold TKAN checkpoints trained on datasets 1-120. This script runs ONE
such checkpoint over a *different* set of datasets (e.g. 121-125, an
out-of-distribution probe) and reports per-file AUC / severity metrics. It is
pure inference:

* no training, no fold logic, no leakage assertion (the held-out set is disjoint
  from training by construction);
* the model geometry + (optional) side branch are reconstructed from the
  checkpoint's tensor shapes and its companion ``preprocessing.json`` via the
  proven :func:`scripts.xai_phase3._load_model` path;
* the z-score applied to every window comes from the checkpoint's
  ``preprocessing.json`` ``norm`` block -- it is NEVER refit on the held-out
  windows (refitting would defeat the point of measuring transfer).

Example::

    python scripts/eval_heldout_phase3.py \
        --checkpoint results/phase3/runs/<run_id>/checkpoints/best_auc.pt \
        --features-csv data/metadata/features_datasets_121_125.csv \
        --npz-root $PWD/phase2_signals \
        --out results/phase3/heldout/fold0_121_125.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
# src/ enables ``import pri2m...``; repo root keeps parity with other scripts and
# lets us reuse ``scripts.xai_phase3`` model-reconstruction helpers.
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402  (after sys.path setup)
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pri2m.data.phase3_sequence import (  # noqa: E402
    NormStats,
    Phase3SequenceDataset,
    WindowConfig,
    collate_windows,
)
from pri2m.phase3.predict import window_config_from_preprocessing  # noqa: E402
from pri2m.phase3.splits import add_parent_group  # noqa: E402
from pri2m.training.phase3_trainer import (  # noqa: E402
    Phase3Trainer,
    TrainConfig,
    resolve_device,
)

# Reuse the EXACT model-reconstruction path the XAI tool uses, so a side-input
# checkpoint is rebuilt identically (side feature names from preprocessing.json,
# dims inferred from the state_dict, baked-in side-norm buffers restored on load).
from scripts.xai_phase3 import _find_preprocessing, _load_model  # noqa: E402

__all__ = ["build_eval_table", "run", "main"]


# Moved to pri2m.phase3.predict so the team-facing scripts/predict_phase3.py can
# reuse it without importing this script. Kept under the old private name: this
# module's callers (and tests) import it from here.
_window_config_from_preprocessing = window_config_from_preprocessing


def build_eval_table(
    features_csv: str, npz_root: str, exclude_omega_below: float | None = None
) -> tuple[pd.DataFrame, int]:
    """Build the held-out eval table from the per-file features CSV.

    Keeps only rows whose ``dataset_<id>/<stem>.npz`` bundle actually exists under
    ``npz_root`` (so a metadata row without an extracted signal is silently
    skipped rather than crashing the loader). Optionally drops rows with
    ``omega < exclude_omega_below`` to mirror the training low-speed exclusion.
    A ``parent_group`` column is added (harmless; the dataset/loader may read it),
    but NO leakage assertion is run -- this is pure inference on a disjoint set.

    Returns ``(eval_table, n_excluded)`` where ``n_excluded`` counts the rows
    dropped by the omega cutoff (NOT the missing-NPZ skips).
    """
    df = pd.read_csv(features_csv)
    if "dataset" not in df.columns or "file" not in df.columns:
        raise ValueError(
            f"{features_csv} must have 'dataset' and 'file' columns; got "
            f"{sorted(df.columns)[:12]}..."
        )

    n_excluded = 0
    if exclude_omega_below is not None:
        if "omega" not in df.columns:
            raise ValueError(
                "--exclude-omega-below was given but the features CSV has no "
                "'omega' column to filter on"
            )
        keep = df["omega"] >= float(exclude_omega_below)
        n_excluded = int((~keep).sum())
        df = df[keep]

    # Drop rows whose NPZ bundle is missing so Phase3SequenceDataset never tries
    # to load a non-existent file. ``_npz_path_for_row`` uses ``Path(file).stem``.
    root = Path(npz_root)
    present = []
    for _, row in df.iterrows():
        stem = Path(str(row["file"])).stem
        npz_path = root / f"dataset_{int(row['dataset'])}" / f"{stem}.npz"
        present.append(npz_path.is_file())
    df = df[pd.Series(present, index=df.index)]

    df = df.reset_index(drop=True)
    # add_parent_group needs an int dataset column; harmless for inference.
    df["dataset"] = df["dataset"].astype(int)
    df = add_parent_group(df)
    return df, n_excluded


def run(
    checkpoint: str,
    features_csv: str,
    npz_root: str,
    out: str,
    preprocessing: str | None = None,
    device: str = "cpu",
    batch_size: int = 64,
    exclude_omega_below: float | None = None,
) -> dict:
    """Score ``checkpoint`` on the held-out features CSV; write + return the result.

    The norm applied to every window is :class:`NormStats` rebuilt from the
    checkpoint's ``preprocessing.json`` ``norm`` block -- it is the checkpoint's
    own train-fold z-score, never refit on these held-out windows. The model
    (side-input aware) is reconstructed via the XAI ``_load_model`` path and run
    in ``eval()`` on ``device``. Returns the result dict that is also written to
    ``out``.
    """
    # 1. preprocessing.json: norm + window geometry + (optional) side block.
    pre_path = preprocessing or _find_preprocessing(checkpoint)
    if pre_path is None:
        raise FileNotFoundError(
            "no preprocessing.json supplied via --preprocessing and none found "
            f"next to the checkpoint {checkpoint!r}; it is required to recover the "
            "checkpoint's z-score (and side-feature names for side models)"
        )
    pre = json.loads(Path(pre_path).read_text(encoding="utf-8"))
    if "norm" not in pre:
        raise ValueError(f"{pre_path} has no 'norm' block; cannot recover z-score")

    # 2. Reconstruct the model (side-input aware) and freeze it. ``_load_model``
    # reads --input-size/--hidden-size/--grid-size/--spline-order off a Namespace;
    # pass None/defaults so the dims are inferred from the state_dict shapes.
    dev = resolve_device(device)
    load_args = argparse.Namespace(
        input_size=None, hidden_size=None, grid_size=5, spline_order=3
    )
    model = _load_model(checkpoint, load_args, pre)
    model.eval()
    model.to(dev)

    # 3. Held-out eval table (NPZ-present rows, optional omega cutoff, parent_group).
    eval_table, n_excluded = build_eval_table(
        features_csv, npz_root, exclude_omega_below=exclude_omega_below
    )

    # 4. The z-score MUST come from the checkpoint's preprocessing, not be refit.
    norm = NormStats.from_dict(pre["norm"])
    cfg = _window_config_from_preprocessing(pre)
    dataset = Phase3SequenceDataset(eval_table, npz_root, cfg=cfg, norm=norm)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_windows,
    )

    # 5. Per-file metrics via the real trainer eval path (handles side models:
    # the side features ride along in the collated batch from the NPZ meta).
    trainer = Phase3Trainer(TrainConfig(device=device))
    metrics = trainer.evaluate(model, loader)

    result = {
        "checkpoint": str(checkpoint),
        "preprocessing": str(pre_path),
        "features_csv": str(features_csv),
        "npz_root": str(npz_root),
        "device": str(dev),
        "side_input": bool(getattr(model, "side_input", False)),
        "exclude_omega_below": exclude_omega_below,
        "n_excluded": int(n_excluded),
        "n_eval_files": int(len(dataset._files)),
        "n_eval_windows": int(len(dataset)),
        "datasets": sorted(int(d) for d in eval_table["dataset"].unique()),
        "metrics": {
            "auc": _as_float(metrics.get("auc")),
            "mae": _as_float(metrics.get("mae")),
            "rmse": _as_float(metrics.get("rmse")),
            "acc": _as_float(metrics.get("acc")),
            "n_files": int(metrics.get("n_files", 0)),
            "n_nonfinite_files": int(metrics.get("n_nonfinite_files", 0)),
        },
    }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    m = result["metrics"]
    print(
        f"[eval_heldout] datasets={result['datasets']} "
        f"n_files={result['n_eval_files']} n_windows={result['n_eval_windows']} "
        f"n_excluded={n_excluded} | auc={m['auc']:.4f} mae={m['mae']:.4f} "
        f"rmse={m['rmse']:.4f} acc={m['acc']:.4f} -> {out_path}"
    )
    return result


def _as_float(value) -> float:
    """Coerce a metric (possibly None / numpy scalar / NaN) to a python float."""
    if value is None:
        return float("nan")
    return float(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a trained checkpoint (e.g. best_auc.pt)."
    )
    parser.add_argument(
        "--features-csv",
        required=True,
        help="Held-out per-file metadata CSV (e.g. features_datasets_121_125.csv).",
    )
    parser.add_argument(
        "--npz-root",
        required=True,
        help="Root containing dataset_<id>/*.npz signal bundles for the held-out set.",
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument(
        "--preprocessing",
        default=None,
        help="Path to the checkpoint's preprocessing.json (norm + window + side). "
        "If omitted, auto-discovered next to the checkpoint.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Compute device: cpu|cuda|cuda:0 (default cpu)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Inference batch size (default 64)."
    )
    parser.add_argument(
        "--exclude-omega-below",
        type=float,
        default=None,
        help="Drop rows with omega < X to mirror the training low-speed exclusion "
        "(default: keep all rows).",
    )
    args = parser.parse_args(argv)

    run(
        checkpoint=args.checkpoint,
        features_csv=args.features_csv,
        npz_root=args.npz_root,
        out=args.out,
        preprocessing=args.preprocessing,
        device=args.device,
        batch_size=args.batch_size,
        exclude_omega_below=args.exclude_omega_below,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
