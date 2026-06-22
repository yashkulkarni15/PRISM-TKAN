#!/usr/bin/env python3
"""Compute Phase 2.5 per-file noise and chatter metrics from extracted NPZ signal bundles.

Reads the existing Phase 2 NPZ bundles produced by ``extract_phase2_signals.py``
and the metadata CSV(s) used to label them, then runs the full Phase 2.5 metric
vector (``pri2m.noise.metrics.compute_file_metrics``) for every selected file.

The script supports clean and noisy datasets together. It automatically routes
each ``--datasets`` token to the clean or noisy summary CSV based on the
dataset ID (clean: 1 to 60, noisy: 61 to 120). One row per file is written to
``noise_metrics.csv``. A ``run_metadata.json`` captures the config and counts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pri2m.noise.metrics import (  # noqa: E402
    NoiseMetricsConfig,
    SIGNALS,
    compute_file_metrics,
)


SIGNAL_ALIASES = {
    "xtpos": ("xtpos", "x_tpos", "xpos", "x"),
    "ytpos": ("ytpos", "y_tpos", "ypos", "y"),
    "Forcex": ("Forcex", "forcex", "force_x", "Fx", "fx"),
    "Forcey": ("Forcey", "forcey", "force_y", "Fy", "fy"),
}

CLEAN_DATASET_RANGE = (1, 60)
NOISY_DATASET_RANGE = (61, 120)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        default=str(REPO_ROOT / "data" / "metadata" / "features_all_datasets.csv"),
        help="Clean feature/metadata CSV (datasets 1 to 60).",
    )
    parser.add_argument(
        "--noisy-summary-csv",
        default=str(REPO_ROOT / "data" / "metadata" / "features_noisy_datasets.csv"),
        help="Noisy feature/metadata CSV (datasets 61 to 120).",
    )
    parser.add_argument(
        "--signals-dir",
        required=True,
        help="Directory of extracted NPZ bundles. Each file lives at "
             "<signals-dir>/dataset_<id>/<source_stem>.npz by default.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write noise_metrics.csv and run_metadata.json.",
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help="Dataset selector such as '1,61,91' or '1-60,61-120'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on total rows processed (smoke testing).",
    )
    parser.add_argument(
        "--max-files-per-dataset",
        type=int,
        default=None,
        help="Optional per-dataset row cap, applied before --limit.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="Fallback sample rate (Hz) used when an NPZ omits fs/time.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()

    requested = parse_dataset_selector(args.datasets)
    clean_ids = {d for d in requested if CLEAN_DATASET_RANGE[0] <= d <= CLEAN_DATASET_RANGE[1]}
    noisy_ids = {d for d in requested if NOISY_DATASET_RANGE[0] <= d <= NOISY_DATASET_RANGE[1]}
    other_ids = requested - clean_ids - noisy_ids
    if other_ids:
        print(f"warning: dataset IDs outside known ranges: {sorted(other_ids)}", file=sys.stderr)

    frames: list[pd.DataFrame] = []
    if clean_ids:
        clean_df = pd.read_csv(args.summary_csv)
        sel = clean_df[clean_df["dataset"].astype(int).isin(clean_ids)].copy()
        sel["__source"] = "clean"
        frames.append(sel)
    if noisy_ids:
        noisy_df = pd.read_csv(args.noisy_summary_csv)
        sel = noisy_df[noisy_df["dataset"].astype(int).isin(noisy_ids)].copy()
        sel["__source"] = "noisy"
        frames.append(sel)

    if not frames:
        print("no datasets selected after filtering", file=sys.stderr)
        return 2

    rows_df = pd.concat(frames, ignore_index=True).sort_values(["dataset", "file"]).reset_index(drop=True)
    if args.max_files_per_dataset is not None:
        rows_df = (
            rows_df.groupby("dataset", group_keys=False)
            .head(args.max_files_per_dataset)
            .reset_index(drop=True)
        )
    if args.limit is not None:
        rows_df = rows_df.head(args.limit).reset_index(drop=True)

    signals_dir = Path(args.signals_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = NoiseMetricsConfig()
    metric_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for record in rows_df.to_dict(orient="records"):
        npz_path = resolve_npz_path(record, signals_dir)
        if npz_path is None:
            skipped.append({"dataset": record["dataset"], "file": record["file"], "reason": "npz_not_found"})
            continue
        try:
            bundle, fs = load_npz_signals(npz_path, args.fs)
        except Exception as exc:
            skipped.append({"dataset": record["dataset"], "file": record["file"], "reason": f"load_error:{exc}"})
            continue

        tpf = optional_float(record.get("tpf"))
        fn = optional_float(record.get("fn"))

        try:
            metrics = compute_file_metrics(bundle, tpf=tpf, fn=fn, fs=fs, config=config)
        except Exception as exc:
            skipped.append({"dataset": record["dataset"], "file": record["file"], "reason": f"metric_error:{exc}"})
            continue

        row_out: dict[str, Any] = {
            "dataset": int(record["dataset"]),
            "file": record["file"],
            "source": record["__source"],
            "label": optional_int(record.get("label")),
            "omega": optional_float(record.get("omega")),
            "b_mm": optional_float(record.get("b_mm")),
            "blim": optional_float(record.get("blim")),
            "b_over_blim": optional_float(record.get("b_over_blim")),
            "tpf": tpf,
            "fn": fn,
            "teeth": optional_int(record.get("teeth")),
            "cut_dir": optional_int(record.get("cut_dir")),
            "noise_std": optional_float(record.get("noise_std")),
            "fs_used_hz": float(fs),
            "n_samples": int(_first_signal_length(bundle)),
            "npz_path": str(npz_path),
        }
        row_out.update(metrics)
        metric_rows.append(row_out)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "noise_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    skipped_path = output_dir / "skipped_files.csv"
    pd.DataFrame(skipped).to_csv(skipped_path, index=False)

    metadata = {
        "config": {
            "nperseg": config.nperseg,
            "overlap_fraction": config.overlap_fraction,
            "analysis_tpf_multiplier": config.analysis_tpf_multiplier,
            "tpf_harmonic_count": config.tpf_harmonic_count,
            "tpf_harmonic_halfwidth_frac": config.tpf_harmonic_halfwidth_frac,
            "fn_band_halfwidth_frac": config.fn_band_halfwidth_frac,
            "sideband_count": config.sideband_count,
            "sideband_halfwidth_frac": config.sideband_halfwidth_frac,
            "noise_floor_low_multiplier": config.noise_floor_low_multiplier,
            "wavelet": config.wavelet,
            "wavelet_level": config.wavelet_level,
            "coherence_band_tpf_multiplier": config.coherence_band_tpf_multiplier,
        },
        "selectors": {
            "requested_datasets": sorted(requested),
            "clean_datasets_seen": sorted(clean_ids),
            "noisy_datasets_seen": sorted(noisy_ids),
            "outside_known_ranges": sorted(other_ids),
        },
        "counts": {
            "rows_selected": int(len(rows_df)),
            "metric_rows": int(len(metric_rows)),
            "skipped_rows": int(len(skipped)),
        },
        "paths": {
            "summary_csv": str(args.summary_csv),
            "noisy_summary_csv": str(args.noisy_summary_csv),
            "signals_dir": str(args.signals_dir),
            "output_dir": str(args.output_dir),
            "noise_metrics_csv": str(metrics_path),
            "skipped_files_csv": str(skipped_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(json.dumps(metadata, indent=2, default=str))
    return 0


def resolve_npz_path(row: Mapping[str, Any], signals_dir: Path) -> Path | None:
    file_value = str(row.get("file") or "").strip()
    dataset = optional_int(row.get("dataset"))
    if not file_value or dataset is None:
        return None
    stem = Path(file_value).stem
    candidates = [
        signals_dir / f"dataset_{dataset}" / f"{stem}.npz",
        signals_dir / f"dataset{dataset}" / f"{stem}.npz",
        signals_dir / f"{stem}.npz",
        signals_dir / str(dataset) / f"{stem}.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_npz_signals(path: Path, fallback_fs: float | None) -> tuple[dict[str, np.ndarray], float]:
    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)
        signals: dict[str, np.ndarray] = {}
        for canonical, aliases in SIGNAL_ALIASES.items():
            for alias in aliases:
                if alias in keys:
                    signals[canonical] = np.asarray(npz[alias], dtype=float).reshape(-1)
                    break
        fs = _infer_sample_rate(dict(npz), fallback_fs)
    return signals, fs


def _infer_sample_rate(npz: Mapping[str, Any], fallback_fs: float | None) -> float:
    for key in ("fs", "sample_rate", "sampling_rate", "sr"):
        if key in npz:
            value = float(np.asarray(npz[key]).reshape(-1)[0])
            if math.isfinite(value) and value > 0:
                return value
    for key in ("t", "time", "Time"):
        if key in npz:
            t = np.asarray(npz[key], dtype=float).reshape(-1)
            t = t[np.isfinite(t)]
            if t.size >= 2:
                diffs = np.diff(t)
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if diffs.size:
                    return 1.0 / float(np.median(diffs))
    if fallback_fs is not None and math.isfinite(fallback_fs) and fallback_fs > 0:
        return float(fallback_fs)
    raise ValueError("sample rate missing; provide --fs or include fs/time in the .npz")


def _first_signal_length(bundle: Mapping[str, np.ndarray]) -> int:
    for s in SIGNALS:
        if s in bundle and bundle[s].size > 0:
            return int(bundle[s].size)
    return 0


def parse_dataset_selector(selector: str) -> set[int]:
    result: set[int] = set()
    for part in selector.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            result.update(range(min(start, end), max(start, end) + 1))
        else:
            result.add(int(token))
    return result


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def optional_int(value: Any) -> int | None:
    f = optional_float(value)
    if f is None:
        return None
    return int(f)


if __name__ == "__main__":
    raise SystemExit(main())
