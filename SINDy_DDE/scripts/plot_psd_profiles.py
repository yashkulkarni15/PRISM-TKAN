#!/usr/bin/env python3
"""Plot per-dataset stable-vs-unstable PSD profiles for Phase 2.2."""

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
    DEFAULT_MIN_NPERSEG,
    DEFAULT_NPERSEG,
    WelchConfig,
    normalize_distribution,
    welch_psd,
)


SIGNAL_ALIASES = {
    "xtpos": ("xtpos", "x_tpos", "xpos", "x"),
    "ytpos": ("ytpos", "y_tpos", "ypos", "y"),
    "Forcex": ("Forcex", "forcex", "force_x", "Fx", "fx"),
    "Forcey": ("Forcey", "forcey", "force_y", "Fy", "fy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument(
        "--signals-dir",
        default="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/phase2_signals",
    )
    parser.add_argument(
        "--output-dir",
        default="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2/psd_profiles",
    )
    parser.add_argument("--datasets", default="1-60")
    parser.add_argument("--signals", default="xtpos,Forcex")
    parser.add_argument("--ratio-grid-size", type=int, default=512)
    parser.add_argument("--max-ratio", type=float, default=5.0)
    parser.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG)
    parser.add_argument("--min-nperseg", type=int, default=DEFAULT_MIN_NPERSEG)
    parser.add_argument("--fs", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(Path(args.summary_csv))
    dataset_filter = parse_dataset_selector(args.datasets)
    rows = [row for row in rows if optional_int(row.get("dataset")) in dataset_filter]
    signals = [signal.strip() for signal in args.signals.split(",") if signal.strip()]
    invalid = sorted(set(signals) - set(SIGNAL_ALIASES))
    if invalid:
        raise ValueError(f"unsupported signals: {invalid}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratio_grid = np.linspace(0.0, float(args.max_ratio), max(16, args.ratio_grid_size))
    config = WelchConfig(nperseg=args.nperseg, min_nperseg=args.min_nperseg)
    signals_dir = Path(args.signals_dir)

    records: dict[tuple[int, str], dict[int, list[np.ndarray]]] = {}
    fn_ratios: dict[tuple[int, str], list[float]] = {}
    failures: list[dict[str, Any]] = []

    for row in rows:
        dataset = optional_int(row.get("dataset"))
        label = optional_int(row.get("label"))
        tpf = optional_float(row.get("tpf"))
        fn = optional_float(row.get("fn"))
        if dataset is None or label not in (0, 1) or tpf is None or tpf <= 0:
            continue
        npz_path = resolve_npz_path(row, signals_dir)
        if npz_path is None:
            failures.append(failure_row(row, "npz_not_found"))
            continue
        try:
            signal_bundle, fs = load_npz_signals(npz_path, args.fs)
        except Exception as exc:
            failures.append(failure_row(row, f"load_failed: {exc}"))
            continue
        for signal_name in signals:
            if signal_name not in signal_bundle:
                failures.append(failure_row(row, f"missing_signal:{signal_name}"))
                continue
            try:
                dist = normalized_ratio_psd(
                    signal_bundle[signal_name],
                    fs=fs,
                    tpf=tpf,
                    ratio_grid=ratio_grid,
                    signal_name=signal_name,
                    config=config,
                )
            except Exception as exc:
                failures.append(failure_row(row, f"psd_failed:{signal_name}:{exc}"))
                continue
            key = (dataset, signal_name)
            records.setdefault(key, {0: [], 1: []})[label].append(dist)
            if fn is not None and fn > 0:
                fn_ratios.setdefault(key, []).append(fn / tpf)

    summary_rows = []
    plot_status = write_dataset_plots(output_dir, records, fn_ratios, ratio_grid, summary_rows)
    write_csv(output_dir / "profile_summary.csv", summary_rows)
    write_csv(output_dir / "profile_failures.csv", failures)
    metadata = {
        "summary_csv": args.summary_csv,
        "signals_dir": str(signals_dir),
        "output_dir": str(output_dir),
        "dataset_filter_ids": sorted(dataset_filter),
        "signals": signals,
        "source_rows_after_dataset_filter": len(rows),
        "profile_groups": len(records),
        "plots_written": len([row for row in summary_rows if row.get("plot_path")]),
        "failure_rows": len(failures),
        "plot_status": plot_status,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def normalized_ratio_psd(
    values: np.ndarray,
    fs: float,
    tpf: float,
    ratio_grid: np.ndarray,
    signal_name: str,
    config: WelchConfig,
) -> np.ndarray:
    estimate = welch_psd(values, fs=fs, signal_name=signal_name, config=config)
    ratios = np.asarray(estimate.frequencies, dtype=float) / float(tpf)
    psd = np.asarray(estimate.psd, dtype=float)
    mask = np.isfinite(ratios) & np.isfinite(psd) & (ratios >= 0.0) & (ratios <= ratio_grid[-1])
    if not np.any(mask):
        raise ValueError("empty normalized frequency band")
    interp = np.interp(ratio_grid, ratios[mask], psd[mask], left=0.0, right=0.0)
    return normalize_distribution(np.maximum(interp, 0.0))


def write_dataset_plots(
    output_dir: Path,
    records: Mapping[tuple[int, str], Mapping[int, list[np.ndarray]]],
    fn_ratios: Mapping[tuple[int, str], list[float]],
    ratio_grid: np.ndarray,
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"ok": False, "message": f"matplotlib unavailable: {exc}"}

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for dataset, signal_name in sorted(records):
        label_map = records[(dataset, signal_name)]
        stable = np.asarray(label_map.get(0, []), dtype=float)
        unstable = np.asarray(label_map.get(1, []), dtype=float)
        if stable.size == 0 and unstable.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        stable_peak = math.nan
        unstable_peak = math.nan
        if stable.shape[0] > 0:
            stable_mean = normalize_distribution(np.mean(stable, axis=0))
            stable_peak = float(ratio_grid[int(np.argmax(stable_mean))])
            ax.plot(ratio_grid, stable_mean, label=f"stable n={stable.shape[0]}", linewidth=1.7)
        if unstable.shape[0] > 0:
            unstable_mean = normalize_distribution(np.mean(unstable, axis=0))
            unstable_peak = float(ratio_grid[int(np.argmax(unstable_mean))])
            ax.plot(ratio_grid, unstable_mean, label=f"unstable n={unstable.shape[0]}", linewidth=1.7)

        median_fn_ratio = median_or_nan(fn_ratios.get((dataset, signal_name), []))
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, label="TPF")
        if math.isfinite(median_fn_ratio) and 0.0 <= median_fn_ratio <= ratio_grid[-1]:
            ax.axvline(median_fn_ratio, color="tab:red", linestyle=":", linewidth=1.2, label="fn/tpf")
        ax.set_title(f"Dataset {dataset} {signal_name} PSD profile")
        ax.set_xlabel("frequency / tooth-pass frequency")
        ax.set_ylabel("normalized PSD mass")
        ax.set_xlim(0.0, ratio_grid[-1])
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        path = plot_dir / f"dataset_{dataset}_{signal_name}_psd_profile.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        summary_rows.append(
            {
                "dataset": dataset,
                "signal": signal_name,
                "n_stable": int(stable.shape[0]) if stable.size else 0,
                "n_unstable": int(unstable.shape[0]) if unstable.size else 0,
                "stable_peak_ratio": stable_peak,
                "unstable_peak_ratio": unstable_peak,
                "median_fn_to_tpf": median_fn_ratio,
                "plot_path": str(path),
            }
        )
    return {"ok": True, "plots_written": len(summary_rows)}


def resolve_npz_path(row: Mapping[str, Any], signals_dir: Path) -> Path | None:
    file_value = str(row.get("file") or "").strip()
    dataset = optional_int(row.get("dataset"))
    if not file_value or dataset is None:
        return None
    stem = Path(file_value).stem
    candidates = [
        signals_dir / f"dataset_{dataset}" / f"{stem}.npz",
        signals_dir / f"dataset{dataset}" / f"{stem}.npz",
        signals_dir / str(dataset) / f"{stem}.npz",
        signals_dir / f"{stem}.npz",
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_dataset_selector(selector: str | None) -> set[int]:
    result: set[int] = set()
    if not selector:
        return result
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


def median_or_nan(values: Iterable[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    return float(finite[len(finite) // 2])


def failure_row(row: Mapping[str, Any], message: str) -> dict[str, Any]:
    return {
        "dataset": optional_int(row.get("dataset")),
        "file": row.get("file"),
        "label": optional_int(row.get("label")),
        "message": message,
    }


if __name__ == "__main__":
    raise SystemExit(main())
