#!/usr/bin/env python3
"""Run PRI2M Phase 2 PSD/Jensen-Shannon summaries.

The CLI accepts a summary CSV on its own. If pre-extracted signal NPZ files are
available, it also computes Welch PSD summaries and optional divergence against
a reference NPZ.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pri2m.spectral.psd_js import (  # noqa: E402
    DEFAULT_ANALYSIS_TPF_MULTIPLIER,
    DEFAULT_MIN_NPERSEG,
    DEFAULT_NPERSEG,
    SIGNAL_NAMES,
    WelchConfig,
    classify_summary_feature_row,
    compare_signal_maps,
    scipy_available,
    summarize_signal_map,
)


SIGNAL_ALIASES = {
    "xtpos": ("xtpos", "x_tpos", "xpos", "x"),
    "ytpos": ("ytpos", "y_tpos", "ypos", "y"),
    "Forcex": ("Forcex", "forcex", "force_x", "Fx", "fx"),
    "Forcey": ("Forcey", "forcey", "force_y", "Fy", "fy"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PRI2M Welch PSD/JSD metadata and signal summaries."
    )
    parser.add_argument("--summary-csv", required=True, help="Input summary CSV.")
    parser.add_argument(
        "--signals-dir",
        default=None,
        help="Optional directory containing pre-extracted .npz signal files.",
    )
    parser.add_argument(
        "--reference-npz",
        default=None,
        help="Optional reference .npz for per-row Jensen-Shannon divergence.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="Fallback sample rate in Hz when .npz has no fs/time vector.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows processed for signal PSD computation.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional dataset IDs/ranges to process for raw PSD, e.g. 1,4,44-48.",
    )
    parser.add_argument(
        "--inspect-datasets",
        default="44-48",
        help="Dataset IDs/ranges to summarize for inspection, e.g. 44-48,52.",
    )
    parser.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG)
    parser.add_argument("--min-nperseg", type=int, default=DEFAULT_MIN_NPERSEG)
    parser.add_argument("--analysis-multiplier", type=float, default=DEFAULT_ANALYSIS_TPF_MULTIPLIER)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read metadata and discover optional signals without writing artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        print(f"summary CSV not found: {summary_path}", file=sys.stderr)
        return 2

    source_rows = read_summary_csv(summary_path)
    inspect_ids = parse_dataset_selector(args.inspect_datasets)
    dataset_filter = parse_dataset_selector(args.datasets) if args.datasets else None
    rows = (
        [
            row
            for row in source_rows
            if optional_int(row.get("dataset")) in dataset_filter
        ]
        if dataset_filter is not None
        else source_rows
    )
    config = WelchConfig(
        nperseg=args.nperseg,
        min_nperseg=args.min_nperseg,
        analysis_tpf_multiplier=args.analysis_multiplier,
    )
    warnings: list[str] = []

    metadata = summarize_metadata(rows, inspect_ids, summary_path)
    metadata["source_rows_before_dataset_filter"] = len(source_rows)
    metadata["dataset_filter_ids"] = sorted(dataset_filter) if dataset_filter is not None else None
    metadata["scipy_available"] = scipy_available()
    metadata["welch_defaults"] = {
        "window": config.window,
        "nperseg": config.nperseg,
        "overlap_fraction": config.overlap_fraction,
        "detrend": config.detrend,
        "analysis_band": f"0 to {config.analysis_tpf_multiplier}*tpf",
        "min_nperseg": config.min_nperseg,
    }

    dev_set, dev_set_rows = summarize_dev_set(rows, inspect_ids)
    psd_rows: list[dict[str, Any]] = []
    divergence_rows: list[dict[str, Any]] = []

    signals_dir = Path(args.signals_dir) if args.signals_dir else None
    if signals_dir is None:
        warnings.append("no --signals-dir supplied; raw PSD/JSD signal summaries skipped")
    elif not signals_dir.exists():
        warnings.append(f"signals directory not found: {signals_dir}; raw PSD/JSD skipped")
    else:
        reference_bundle = None
        reference_fs = None
        if args.reference_npz:
            try:
                reference_bundle, reference_fs = load_npz_signals(Path(args.reference_npz), args.fs)
            except Exception as exc:
                warnings.append(f"reference NPZ could not be loaded: {exc}")

        for row_index, row in enumerate(rows):
            if args.limit is not None and row_index >= args.limit:
                break
            npz_path = resolve_npz_path(row, signals_dir)
            if npz_path is None:
                continue
            try:
                signal_bundle, fs = load_npz_signals(npz_path, args.fs)
                tpf = optional_float(row.get("tpf"))
                fn = optional_float(row.get("fn"))
                dataset = optional_int(row.get("dataset"))
                label = optional_int(row.get("label"))
                for psd_row in summarize_signal_map(
                    signal_bundle,
                    fs=fs,
                    tpf=tpf,
                    fn=fn,
                    config=config,
                    dataset=dataset,
                    label=label,
                ):
                    psd_row.update(row_identity(row, npz_path))
                    psd_rows.append(psd_row)
                if reference_bundle is not None:
                    compare_fs = reference_fs if reference_fs is not None else fs
                    if not math.isclose(compare_fs, fs, rel_tol=1.0e-9, abs_tol=1.0e-9):
                        warnings.append(
                            f"sample-rate mismatch for {npz_path.name}: "
                            f"row fs={fs}, reference fs={compare_fs}; using row fs"
                        )
                    for div_row in compare_signal_maps(
                        reference_bundle,
                        signal_bundle,
                        fs=fs,
                        tpf=tpf,
                        config=config,
                    ):
                        div_row.update(row_identity(row, npz_path))
                        divergence_rows.append(div_row)
            except Exception as exc:
                warnings.append(f"{npz_path}: {exc}")

    metadata["raw_signal_rows"] = len(psd_rows)
    metadata["divergence_rows"] = len(divergence_rows)
    metadata["warnings"] = warnings
    result = {
        "metadata": metadata,
        "dev_set": dev_set,
        "psd_summary_preview": psd_rows[:10],
        "divergence_summary_preview": divergence_rows[:10],
    }

    if args.output_dir and not args.dry_run:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "run_metadata.json", result)
        write_csv(output_dir / "psd_summary.csv", psd_rows)
        write_csv(output_dir / "divergence_summary.csv", divergence_rows)
        write_csv(output_dir / "dev_set_feature_sanity.csv", dev_set_rows)
        result["metadata"]["output_dir"] = str(output_dir)

    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    return 0


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_metadata(
    rows: list[Mapping[str, Any]],
    inspect_ids: set[int],
    summary_path: Path,
) -> dict[str, Any]:
    datasets = [optional_int(row.get("dataset")) for row in rows]
    datasets = [dataset for dataset in datasets if dataset is not None]
    labels = [optional_int(row.get("label")) for row in rows]
    labels = [label for label in labels if label is not None]
    tpf_values = [optional_float(row.get("tpf")) for row in rows]
    tpf_values = [value for value in tpf_values if value is not None]
    inspect_rows = [
        row for row in rows if optional_int(row.get("dataset")) in inspect_ids
    ]
    return {
        "summary_csv": str(summary_path),
        "rows": len(rows),
        "column_count": len(rows[0].keys()) if rows else 0,
        "columns_preview": list(rows[0].keys())[:20] if rows else [],
        "dataset_count": len(set(datasets)),
        "dataset_min": min(datasets) if datasets else None,
        "dataset_max": max(datasets) if datasets else None,
        "label_counts": counts(labels),
        "tpf_hz": numeric_stats(tpf_values),
        "inspect_dataset_ids": sorted(inspect_ids),
        "inspect_rows": len(inspect_rows),
        "inspect_label_counts": counts(
            [
                optional_int(row.get("label"))
                for row in inspect_rows
                if optional_int(row.get("label")) is not None
            ]
        ),
    }


def summarize_dev_set(
    rows: list[Mapping[str, Any]],
    inspect_ids: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inspect_rows = [
        row for row in rows if optional_int(row.get("dataset")) in inspect_ids
    ]
    per_dataset: dict[str, dict[str, Any]] = {}
    for row in inspect_rows:
        dataset = optional_int(row.get("dataset"))
        if dataset is None:
            continue
        entry = per_dataset.setdefault(
            str(dataset),
            {"rows": 0, "label_counts": {}, "files_preview": []},
        )
        entry["rows"] += 1
        label = optional_int(row.get("label"))
        label_key = str(label) if label is not None else "missing"
        entry["label_counts"][label_key] = entry["label_counts"].get(label_key, 0) + 1
        if len(entry["files_preview"]) < 8 and row.get("file"):
            entry["files_preview"].append(row["file"])

    feature_sanity_rows: list[dict[str, Any]] = []
    for row in inspect_rows:
        identity = row_identity(row, None)
        for signal_name in SIGNAL_NAMES:
            sanity = classify_summary_feature_row(row, signal_name)
            sanity.update(identity)
            feature_sanity_rows.append(sanity)

    sanity_counts = counts(
        [
            row["sanity_class"]
            for row in feature_sanity_rows
            if row.get("sanity_class") is not None
        ]
    )
    tag_counts: dict[str, int] = {}
    for row in feature_sanity_rows:
        for tag in row.get("inspection_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return {
        "inspect_rows": len(inspect_rows),
        "per_dataset": per_dataset,
        "feature_sanity_counts": sanity_counts,
        "inspection_tag_counts": tag_counts,
        "feature_sanity_preview": feature_sanity_rows[:8],
    }, feature_sanity_rows


def resolve_npz_path(row: Mapping[str, Any], signals_dir: Path) -> Path | None:
    explicit = row.get("signal_npz") or row.get("npz") or row.get("signal_path")
    candidates: list[Path] = []
    if explicit:
        explicit_path = Path(str(explicit))
        candidates.append(explicit_path if explicit_path.is_absolute() else signals_dir / explicit_path)

    file_value = str(row.get("file") or "").strip()
    dataset = optional_int(row.get("dataset"))
    if file_value:
        stem = Path(file_value).stem
        names = [f"{stem}.npz"]
        if dataset is not None:
            names.extend(
                [
                    f"dataset{dataset}_{stem}.npz",
                    f"dataset_{dataset}_{stem}.npz",
                    f"{dataset}_{stem}.npz",
                ]
            )
        for name in names:
            candidates.append(signals_dir / name)
            if dataset is not None:
                candidates.append(signals_dir / str(dataset) / name)
                candidates.append(signals_dir / f"dataset{dataset}" / name)
                candidates.append(signals_dir / f"dataset_{dataset}" / name)

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
                    signals[canonical] = np.asarray(npz[alias], dtype=float)
                    break
        if not signals:
            raise ValueError(f"no recognized signal keys found in {path}")
        fs = infer_sample_rate(npz, fallback_fs)
    return signals, fs


def infer_sample_rate(npz: Mapping[str, Any], fallback_fs: float | None) -> float:
    for key in ("fs", "sample_rate", "sampling_rate", "sr"):
        if key in npz:
            value = float(np.asarray(npz[key]).reshape(-1)[0])
            if math.isfinite(value) and value > 0:
                return value
    for key in ("time", "t", "Time"):
        if key in npz:
            time = np.asarray(npz[key], dtype=float).reshape(-1)
            time = time[np.isfinite(time)]
            if time.size >= 2:
                diffs = np.diff(time)
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if diffs.size:
                    return 1.0 / float(np.median(diffs))
    if fallback_fs is not None and math.isfinite(fallback_fs) and fallback_fs > 0:
        return float(fallback_fs)
    raise ValueError("sample rate missing; provide --fs or include fs/time in the .npz")


def row_identity(row: Mapping[str, Any], npz_path: Path | None) -> dict[str, Any]:
    return {
        "dataset": optional_int(row.get("dataset")),
        "file": row.get("file"),
        "label": optional_int(row.get("label")),
        "omega": optional_float(row.get("omega")),
        "b_mm": optional_float(row.get("b_mm")),
        "tpf": optional_float(row.get("tpf")),
        "fn": optional_float(row.get("fn")),
        "npz_path": str(npz_path) if npz_path is not None else None,
    }


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


def counts(values: Iterable[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(finite),
        "min": finite[0],
        "median": finite[len(finite) // 2],
        "max": finite[-1],
    }


def optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def optional_int(value: Any) -> int | None:
    parsed = optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in fields})


def csv_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, default=json_default)
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
