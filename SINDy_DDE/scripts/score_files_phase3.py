#!/usr/bin/env python3
"""Per-file TKAN scores for the MiniROCKET-vs-TKAN noise comparison (T1/T2).

One row per (file, level): the file-mean ``chatter_logit`` of a frozen TKAN
checkpoint. The CSV schema is shared with the MiniROCKET runner, so the analysis
script can join the two models file by file.

Passes
------
``--split val``      every file of the checkpoint's own validation fold (fold =
                     ``preprocessing.json["fold"]``, split by the exact rule of
                     ``build_fold_loaders``), ALL windows. Reproduces the
                     training-time val AUC.
``--split heldout``  every file of the held-out features CSV whose NPZ exists
                     (``eval_heldout_phase3.build_eval_table``), ALL windows.
                     Reproduces ``eval_heldout_phase3.run``'s AUC exactly.
``--noise-levels``   (val only; written as split ``val_noise``) for each level,
                     :func:`pri2m.xai.noise_probe.inject_noise` is applied to the
                     file's FULL raw signal before windowing, keyed on the
                     table's file name, so MiniROCKET sees bit-identical noisy
                     inputs. ``--noise-files clean`` keeps only tier-0 files;
                     ``--noise-max-windows-per-file`` keeps that many evenly
                     spaced windows (``select_train_windows``). Level 0 is the
                     un-noised baseline on the same windows.

Inference path matched to ``Phase3SequenceDataset`` + ``Phase3Trainer.evaluate``
step for step: float64 signal -> windows -> float32 -> checkpoint z-score
(never refit) -> batch dict with the ``collate_windows`` keys/dtypes (side keys
omega/tpf/fs/tau from the NPZ meta) -> ``model(batch)`` -> per-file sequential
mean of the float logits. Files are scored one at a time, so memory is bounded
by one file.

The CSV and its sidecar JSON (provenance + file-level AUC per split/level, with
``"complete": true``) are written once, at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from pri2m.data.phase3_sequence import (  # noqa: E402
    NormStats,
    _npz_path_for_row,
    _row_parent_group,
    _safe_float,
    iter_windows,
    load_npz_signals,
    tau_samples,
)
from pri2m.models.minirocket_baseline import select_train_windows  # noqa: E402
from pri2m.phase3.predict import window_config_from_preprocessing  # noqa: E402
from pri2m.phase3.splits import assert_no_mirror_leakage, parent_group  # noqa: E402
from pri2m.training.phase3_trainer import _batch_to_device, resolve_device  # noqa: E402
from pri2m.xai.noise_probe import inject_noise, noise_tier  # noqa: E402

CSV_COLUMNS = ["model", "fold", "split", "level", "file", "dataset", "parent_group",
               "noise_tier", "label", "omega", "b_over_blim", "n_windows", "score"]


# --------------------------------------------------------------------------- #
# Split (copied from scripts/minirocket_phase3.py; that file is edited elsewhere)
# --------------------------------------------------------------------------- #
def split_fold(table: pd.DataFrame, folds_json: dict, fold_k: int):
    """``(train_table, val_table)`` by the rule ``build_fold_loaders`` uses."""
    group_to_fold = {int(k): int(v) for k, v in folds_json["group_to_fold"].items()}
    pg = (table["parent_group"].astype(int) if "parent_group" in table.columns
          else parent_group(table["dataset"]))
    fold_of_row = pg.astype(int).map(group_to_fold)
    val_mask = fold_of_row == int(fold_k)
    val_table, train_table = table[val_mask], table[~val_mask]
    assert_no_mirror_leakage(train_table, val_table)
    return train_table, val_table


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _file_logit_mean(model, windows: np.ndarray, meta: dict, row: pd.Series,
                     batch_size: int, device) -> float:
    """Mean chatter_logit over ``windows`` ([n, L, C] float32, normalised).

    The batch carries every key ``collate_windows`` produces, with its dtypes,
    so side-input models read omega/tpf/fs/tau exactly as in evaluation.
    """
    n = windows.shape[0]
    bob = _safe_float(row.get("b_over_blim", np.nan))
    label = float(meta["label"])
    sw = float(row.get("sample_weight", 1.0))
    tau = tau_samples(meta["fs"], meta["tpf"])
    ds = int(meta["dataset"])
    fname = str(row["file"])
    pg = _row_parent_group(row)
    ml = str(row.get("mode_label", ""))
    total = 0.0
    with torch.no_grad():
        for s in range(0, n, batch_size):
            w = windows[s : s + batch_size]
            B = w.shape[0]

            def f32(v):
                return torch.tensor([v] * B, dtype=torch.float32)

            batch = {
                "window": torch.as_tensor(np.ascontiguousarray(w), dtype=torch.float32),
                "b_over_blim": f32(bob), "label": f32(label), "sample_weight": f32(sw),
                "tau": torch.tensor([tau] * B, dtype=torch.long),
                "dataset": [ds] * B, "file": [fname] * B, "parent_group": [pg] * B,
                "mode_label": [ml] * B,
                "omega": f32(float(meta["omega"])), "tpf": f32(float(meta["tpf"])),
                "fn": f32(float(meta["fn"])), "fs": f32(float(meta["fs"])),
            }
            out = model(_batch_to_device(batch, device))
            # Sequential python-float sum, as metrics._file_means does.
            for v in out["chatter_logit"].reshape(-1).tolist():
                total += float(v)
    return total / n


def _blank(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if math.isfinite(f) else np.nan


def _tier_for(row: pd.Series, split: str):
    if split in ("val", "val_noise"):
        t = noise_tier(int(row["dataset"]))
        return np.nan if t is None else t
    if "noise_std" in row and pd.notna(row["noise_std"]):
        v = float(row["noise_std"])
        return int(v) if v.is_integer() else v
    return np.nan


def auc_block(df: pd.DataFrame) -> dict:
    out: dict = {}
    for (split, level), g in df.groupby(["split", "level"], sort=True):
        p = _sigmoid(g["score"].to_numpy())
        y = g["label"].to_numpy().astype(int)
        keep = np.isfinite(p)
        y, p = y[keep], p[keep]
        auc = (float(roc_auc_score(y, p)) if y.size and y.min() != y.max() else None)
        acc = float(np.mean((p >= 0.5).astype(int) == y)) if y.size else None
        out.setdefault(str(split), {})[str(float(level))] = {
            "auc": auc, "acc": acc, "n_files": int(len(g)),
            "n_pos": int(y.sum()), "n_nonfinite": int((~keep).sum()),
        }
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _parse_levels(spec: str) -> list[float]:
    levels = [float(x) for x in spec.split(",") if x.strip() != ""]
    if not levels or any((not math.isfinite(v)) or v < 0 for v in levels):
        raise SystemExit(f"--noise-levels must be finite non-negative floats, got {spec!r}")
    return sorted(set(levels))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--preprocessing", default=None,
                    help="Default: preprocessing.json found next to/above the checkpoint.")
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--split", choices=("val", "heldout"), required=True)
    ap.add_argument("--table", default=None, help="Phase 3 table (val split).")
    ap.add_argument("--folds-json", default=None, help="Folds JSON (val split).")
    ap.add_argument("--features-csv", default=None, help="Held-out features CSV.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--model-name", default="tkan")
    ap.add_argument("--fold-label", type=int, default=None,
                    help="Fold written to the CSV (default: preprocessing.json fold).")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--noise-levels", default=None,
                    help="Comma list in tier units, e.g. 0,1,2.5,5,7.5,10 (val only).")
    ap.add_argument("--noise-calibration", default=None)
    ap.add_argument("--noise-files", choices=("clean", "all"), default="clean")
    ap.add_argument("--noise-max-windows-per-file", type=int, default=8)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args(argv)
    t_start = time.time()

    from eval_heldout_phase3 import build_eval_table
    from xai_phase3 import _find_preprocessing, _load_model

    # ---- noise config (validated before any expensive work) ----
    noise = None
    if args.noise_levels is not None:
        if args.split != "val":
            raise SystemExit("--noise-levels is defined for --split val only")
        if not args.noise_calibration:
            raise SystemExit("--noise-levels needs --noise-calibration")
        cal_path = Path(args.noise_calibration)
        if not cal_path.is_file():
            raise SystemExit(f"noise calibration not found: {cal_path}")
        cal = json.loads(cal_path.read_text())
        summ = cal.get("summary") or {}
        rel = summ.get("rel_sigma_per_unit")
        if rel is None:
            raise SystemExit(
                f"calibration {cal_path} has no rel_sigma_per_unit "
                f"(undecided_reason={summ.get('undecided_reason')!r}); refusing to "
                "inject noise that does not match the real tiers")
        noise = {"levels": _parse_levels(args.noise_levels),
                 "rel_sigma_per_unit": [float(x) for x in rel],
                 "calibration": str(cal_path), "linear_ok": summ.get("linear_ok"),
                 "seed": int(args.noise_seed), "files": args.noise_files,
                 "max_windows_per_file": int(args.noise_max_windows_per_file),
                 "key": "table file name (e.g. time1_5.h5)"}

    # ---- checkpoint + preprocessing ----
    pre_path = args.preprocessing or _find_preprocessing(args.checkpoint)
    if pre_path is None:
        raise SystemExit(f"no preprocessing.json for {args.checkpoint}")
    pre = json.loads(Path(pre_path).read_text())
    if "norm" not in pre:
        raise SystemExit(f"{pre_path} has no 'norm' block")
    norm = NormStats.from_dict(pre["norm"])
    cfg = window_config_from_preprocessing(pre)
    if noise is not None and len(noise["rel_sigma_per_unit"]) != len(cfg.channels):
        raise SystemExit(f"calibration has {len(noise['rel_sigma_per_unit'])} channels, "
                         f"model uses {len(cfg.channels)}")
    dev = resolve_device(args.device)
    model = _load_model(args.checkpoint, argparse.Namespace(
        input_size=None, hidden_size=None, grid_size=5, spline_order=3), pre)
    model.eval()
    model.to(dev)
    fold_label = args.fold_label if args.fold_label is not None else pre.get("fold")

    # ---- the file table ----
    if args.split == "val":
        if not (args.table and args.folds_json):
            raise SystemExit("--split val needs --table and --folds-json")
        if "fold" not in pre:
            raise SystemExit(f"{pre_path} has no 'fold'; cannot pick the val split")
        table = pd.read_csv(args.table)
        folds = json.loads(Path(args.folds_json).read_text())
        _, files = split_fold(table, folds, int(pre["fold"]))
    else:
        if not args.features_csv:
            raise SystemExit("--split heldout needs --features-csv")
        files, _ = build_eval_table(args.features_csv, args.npz_root)
    files = files.reset_index(drop=True)
    split_name = args.split
    levels = [0.0]
    if noise is not None:
        split_name = "val_noise"
        levels = noise["levels"]
        if args.noise_files == "clean":
            files = files[[noise_tier(int(d)) == 0 for d in files["dataset"]]]
            files = files.reset_index(drop=True)
    print(f"[score_files] {args.model_name} fold={fold_label} split={split_name} "
          f"files={len(files)} levels={levels} ckpt={args.checkpoint}", flush=True)

    # ---- score ----
    root = Path(args.npz_root)
    rows: list[dict] = []
    n_skipped_short = 0
    n_windows_total = 0
    for i, (_, row) in enumerate(files.iterrows()):
        signal, meta = load_npz_signals(_npz_path_for_row(root, row), cfg.channels)
        for level in levels:
            sig = signal
            if noise is not None:
                sig = inject_noise(signal, level, noise["rel_sigma_per_unit"],
                                   key=str(row["file"]), seed=noise["seed"])
            w = iter_windows(sig, cfg)
            if w.shape[0] == 0:
                n_skipped_short += 1
                continue  # Phase3SequenceDataset skips files with no full window
            w = w.astype(np.float32)
            if noise is not None:
                w = w[select_train_windows(w.shape[0], noise["max_windows_per_file"])]
            w = norm.apply(w)
            score = _file_logit_mean(model, w, meta, row, args.batch_size, dev)
            n_windows_total += int(w.shape[0])
            rows.append({
                "model": args.model_name, "fold": fold_label, "split": split_name,
                "level": float(level), "file": str(row["file"]),
                "dataset": int(row["dataset"]), "parent_group": _row_parent_group(row),
                "noise_tier": _tier_for(row, split_name),
                "label": int(float(meta["label"]) >= 0.5),
                "omega": float(meta["omega"]),
                "b_over_blim": _blank(row.get("b_over_blim", np.nan)),
                "n_windows": int(w.shape[0]), "score": float(score),
            })
        if args.progress_every and (i + 1) % args.progress_every == 0:
            print(f"[score_files] {i + 1}/{len(files)} files, {n_windows_total} windows, "
                  f"{time.time() - t_start:.0f}s", flush=True)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    aucs = auc_block(df) if len(df) else {}
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    sidecar = {
        "analysis": "TKAN per-file scores for the MiniROCKET-vs-TKAN noise comparison",
        "model": args.model_name, "fold": fold_label, "split": split_name,
        "checkpoint": str(args.checkpoint), "preprocessing": str(pre_path),
        "npz_root": str(args.npz_root), "table": args.table,
        "folds_json": args.folds_json, "features_csv": args.features_csv,
        "side_input": bool(getattr(model, "side_input", False)),
        "window": {"length": cfg.length, "stride": cfg.stride},
        "channels": list(cfg.channels), "batch_size": int(args.batch_size),
        "device": str(dev), "levels": levels, "noise": noise,
        "n_files": int(len(files)), "n_rows": int(len(df)),
        "n_windows": n_windows_total, "n_skipped_no_window": n_skipped_short,
        "score": "mean chatter_logit over the file's windows",
        "auc": aucs,
        "wall_seconds": round(time.time() - t_start, 1),
        "complete": True,
    }
    out_csv.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"[score_files] wrote {out_csv} ({len(df)} rows); auc={json.dumps(aucs)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
