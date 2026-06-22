#!/usr/bin/env python3
"""Extract Phase 2 PSD signals from raw Digital Machining H5 files to NPZ.

This script is designed for Great Lakes. It reads the lightweight feature CSV
for metadata/file names, opens the corresponding raw `time*.h5` files, and
writes compact NPZ bundles consumed by `scripts/run_psd_js.py`.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


_DATA_VERSION = os.environ.get("DATA_VERSION", "10")
_DEFAULT_DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    f"/nfs/turbo/coe-jaabell/datasets/tonylschmitz/digital-machining-database/versions/{_DATA_VERSION}",
)


CANONICAL_SIGNALS = ("xtpos", "ytpos", "Forcex", "Forcey")
SIGNAL_ALIASES = {
    "xtpos": ("xtpos", "xt_x", "x", "x_tpos"),
    "ytpos": ("ytpos", "xt_y", "y", "y_tpos"),
    "Forcex": ("Forcex", "forcex", "force_x", "Fx", "fx", "F_x"),
    "Forcey": ("Forcey", "forcey", "force_y", "Fy", "fy", "F_y"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        default=str(REPO_ROOT / "data" / "metadata" / "features_all_datasets.csv"),
        help="Feature/metadata CSV containing dataset and file columns.",
    )
    parser.add_argument(
        "--data-root",
        default=_DEFAULT_DATA_ROOT,
        help="Raw Digital Machining Database root containing 'Dataset X h5' folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/phase2_signals",
        help="Directory for extracted NPZ files and manifest.",
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help="Dataset selector such as '1', '1,4,29', or '1-60'.",
    )
    parser.add_argument("--signals", default=",".join(CANONICAL_SIGNALS))
    parser.add_argument("--max-files-per-dataset", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sindy-source-dir",
        default=str(REPO_ROOT / "sindy_pipeline"),
        help="Directory containing legacy SINDy helper files when available.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any selected file cannot be extracted.",
    )
    parser.add_argument(
        "--allow-h5-fallback",
        action="store_true",
        default=False,
        help=(
            "Allow generic recursive HDF5 traversal (load_h5_fallback) when the "
            "canonical reader (read_file.py / load_time_h5) is unavailable. "
            "WARNING: the fallback may silently change signal-selection and metadata "
            "semantics. Omitting this flag causes the script to fail closed instead."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    datasets = parse_dataset_selector(args.datasets)
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    invalid = sorted(set(signals) - set(CANONICAL_SIGNALS))
    if invalid:
        raise ValueError(f"Unsupported signals: {invalid}")

    rows = summary[summary["dataset"].astype(int).isin(datasets)].copy()
    if rows.empty:
        print(f"No rows selected for datasets {sorted(datasets)}")
        return 1

    if args.max_files_per_dataset:
        rows = (
            rows.sort_values(["dataset", "file"])
            .groupby("dataset", group_keys=False)
            .head(args.max_files_per_dataset)
            .copy()
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"manifest_{safe_selector_label(args.datasets)}.csv"
    load_time_h5 = load_sindy_reader(Path(args.sindy_source_dir))

    loader = resolve_loader(load_time_h5, allow_fallback=args.allow_h5_fallback)

    manifest_rows: list[dict[str, Any]] = []
    failures = 0
    for _, row in rows.sort_values(["dataset", "file"]).iterrows():
        dataset = int(row["dataset"])
        file_name = str(row["file"])
        source_path = Path(args.data_root) / f"Dataset {dataset} h5" / file_name
        rel_out = Path(f"dataset_{dataset}") / f"{Path(file_name).stem}.npz"
        out_path = output_dir / rel_out

        base_manifest = {
            "dataset": dataset,
            "file": file_name,
            "source_path": str(source_path),
            "npz_path": str(out_path),
            "label": int(row["label"]) if "label" in row and not pd.isna(row["label"]) else "",
            "omega": float(row["omega"]) if "omega" in row and not pd.isna(row["omega"]) else "",
            "tpf": float(row["tpf"]) if "tpf" in row and not pd.isna(row["tpf"]) else "",
            "fn": float(row["fn"]) if "fn" in row and not pd.isna(row["fn"]) else "",
            "status": "pending",
            "message": "",
        }

        if out_path.exists() and not args.overwrite:
            base_manifest["status"] = "exists"
            manifest_rows.append(base_manifest)
            continue

        try:
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            raw = loader(str(source_path))
            bundle = normalize_signal_bundle(raw, signals)
            if "fs" not in bundle:
                fs = infer_fs(raw)
                if fs is not None:
                    bundle["fs"] = np.asarray(float(fs))
            if "t" in raw:
                bundle["t"] = np.asarray(raw["t"], dtype=float)
            bundle["dataset"] = np.asarray(dataset)
            bundle["source_file"] = np.asarray(file_name)
            bundle["label"] = np.asarray(base_manifest["label"])
            bundle["omega"] = np.asarray(base_manifest["omega"])
            bundle["tpf"] = np.asarray(base_manifest["tpf"])
            bundle["fn"] = np.asarray(base_manifest["fn"])

            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, **bundle)
            base_manifest["status"] = "ok"
            base_manifest["message"] = ",".join(sorted(bundle.keys()))
        except Exception as exc:
            failures += 1
            base_manifest["status"] = "failed"
            base_manifest["message"] = str(exc)
        manifest_rows.append(base_manifest)

    write_manifest(manifest_path, manifest_rows)
    summary_payload = {
        "selected_rows": int(len(rows)),
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "failures": failures,
        "statuses": dict(pd.Series([r["status"] for r in manifest_rows]).value_counts()),
    }
    print(json.dumps(summary_payload, indent=2, default=str))
    return 2 if failures and args.strict else 0


def load_sindy_reader(sindy_source_dir: Path):
    reader = sindy_source_dir / "read_file.py"
    if not reader.exists():
        return None
    spec = importlib.util.spec_from_file_location("pri2m_sindy_read_file", reader)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "load_time_h5", None)


def resolve_loader(load_time_h5, allow_fallback: bool):
    """Return the callable to use for loading an H5 file.

    Parameters
    ----------
    load_time_h5:
        The canonical reader returned by ``load_sindy_reader``, or ``None`` when
        the reader module (``read_file.py``) is not available.
    allow_fallback:
        When ``True`` and *load_time_h5* is ``None``, fall back to the generic
        recursive HDF5 traversal (``load_h5_fallback``) and emit a warning.
        When ``False`` and *load_time_h5* is ``None``, raise ``RuntimeError``
        (fail-closed) so that silent semantic changes are never silently accepted.

    Returns
    -------
    callable
        A callable that accepts a single ``str`` path argument and returns a
        ``dict[str, Any]`` of arrays.

    Raises
    ------
    RuntimeError
        When *load_time_h5* is ``None`` and *allow_fallback* is ``False``.
    """
    if load_time_h5 is not None:
        return load_time_h5

    if not allow_fallback:
        raise RuntimeError(
            "The canonical HDF5 reader (read_file.py / load_time_h5) is not available "
            "and --allow-h5-fallback was not set. "
            "Production extraction requires the canonical reader to ensure consistent "
            "signal selection and metadata semantics. "
            "Either provide a valid --sindy-source-dir containing read_file.py, "
            "or pass --allow-h5-fallback to explicitly opt-in to the generic fallback "
            "(which may change signal-selection / metadata semantics)."
        )

    # allow_fallback is True — wrap load_h5_fallback to accept a str path
    import sys as _sys
    print(
        "WARNING: canonical reader (read_file.py / load_time_h5) is unavailable. "
        "Using generic recursive HDF5 fallback (load_h5_fallback). "
        "Signal-selection and metadata semantics may differ from production.",
        file=_sys.stderr,
    )
    return lambda path_str: load_h5_fallback(Path(path_str))


def load_h5_fallback(path: Path) -> dict[str, Any]:
    import h5py

    out: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        def visitor(name: str, obj: Any) -> None:
            if hasattr(obj, "shape"):
                key = Path(name).name
                try:
                    value = np.asarray(obj)
                except Exception:
                    return
                out.setdefault(key, value)

        handle.visititems(visitor)
    return out


def normalize_signal_bundle(raw: Mapping[str, Any], signals: list[str]) -> dict[str, np.ndarray]:
    bundle: dict[str, np.ndarray] = {}
    keys = set(raw.keys())
    for canonical in signals:
        for alias in SIGNAL_ALIASES[canonical]:
            if alias in keys:
                arr = np.asarray(raw[alias], dtype=float).reshape(-1)
                if arr.size == 0:
                    continue
                bundle[canonical] = arr
                break
        if canonical not in bundle:
            raise KeyError(f"missing signal {canonical}; available keys include {sorted(keys)[:20]}")
    lengths = {name: len(arr) for name, arr in bundle.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"signal length mismatch: {lengths}")
    return bundle


def infer_fs(raw: Mapping[str, Any]) -> float | None:
    for key in ("fs", "sample_rate", "sampling_rate"):
        if key in raw:
            value = float(np.asarray(raw[key]).reshape(-1)[0])
            if math.isfinite(value) and value > 0:
                return value
    if "dt" in raw:
        dt = float(np.asarray(raw["dt"]).reshape(-1)[0])
        if math.isfinite(dt) and dt > 0:
            return 1.0 / dt
    if "t" in raw:
        t = np.asarray(raw["t"], dtype=float).reshape(-1)
        diffs = np.diff(t[np.isfinite(t)])
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return 1.0 / float(np.median(diffs))
    return None


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


def safe_selector_label(selector: str) -> str:
    label = selector.strip().replace(",", "_").replace("-", "to")
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in label)
    return cleaned or "selected"


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
