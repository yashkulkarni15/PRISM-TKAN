#!/usr/bin/env python3
"""Aggregate stable-vs-unstable PSD Jensen-Shannon curves for Phase 2."""

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
    SIGNAL_NAMES,
    WelchConfig,
    jensen_shannon_divergence,
    kl_divergence,
    normalize_distribution,
    welch_psd,
)


SIGNAL_ALIASES = {
    "xtpos": ("xtpos", "x_tpos", "xpos", "x"),
    "ytpos": ("ytpos", "y_tpos", "ypos", "y"),
    "Forcex": ("Forcex", "forcex", "force_x", "Fx", "fx"),
    "Forcey": ("Forcey", "forcey", "force_y", "Fy", "fy"),
}
DEFAULT_RATIO_BINS = "0,0.5,0.75,1,1.25,1.5,2,3,5,inf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument(
        "--signals-dir",
        default="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/phase2_signals",
    )
    parser.add_argument(
        "--output-dir",
        default="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/results/phase2/js_curves",
    )
    parser.add_argument("--datasets", default=None, help="Dataset IDs/ranges to include.")
    parser.add_argument("--signals", default=",".join(SIGNAL_NAMES))
    parser.add_argument("--bin-column", default="b_over_blim")
    parser.add_argument("--ratio-bins", default=DEFAULT_RATIO_BINS)
    parser.add_argument("--ratio-grid-size", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--min-per-label", type=int, default=3)
    parser.add_argument(
        "--reference-mode",
        choices=("low_b_ratio", "label_stable"),
        default="low_b_ratio",
        help=(
            "Stable-reference rule. low_b_ratio uses the lowest b/blim samples "
            "within each signal x teeth x cut_dir facet and avoids label leakage."
        ),
    )
    parser.add_argument(
        "--reference-quantile",
        type=float,
        default=0.10,
        help="Quantile of the bin column used for low_b_ratio reference selection.",
    )
    parser.add_argument(
        "--reference-max-b-ratio",
        type=float,
        default=0.80,
        help=(
            "Hard physical ceiling for low_b_ratio reference samples. Facets with "
            "fewer than --min-per-label samples at or below this b/blim are marked "
            "insufficient_stable_reference."
        ),
    )
    parser.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG)
    parser.add_argument("--min-nperseg", type=int, default=DEFAULT_MIN_NPERSEG)
    parser.add_argument("--fs", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    rows = read_rows(summary_path)
    dataset_filter = parse_dataset_selector(args.datasets) if args.datasets else None
    if dataset_filter is not None:
        rows = [row for row in rows if optional_int(row.get("dataset")) in dataset_filter]

    signals = [signal.strip() for signal in args.signals.split(",") if signal.strip()]
    invalid_signals = sorted(set(signals) - set(SIGNAL_NAMES))
    if invalid_signals:
        raise ValueError(f"Unsupported signals: {invalid_signals}")

    bin_edges = parse_bin_edges(args.ratio_bins)
    ratio_grid = np.linspace(0.0, 5.0, max(16, int(args.ratio_grid_size)))
    config = WelchConfig(nperseg=args.nperseg, min_nperseg=args.min_nperseg)
    signals_dir = Path(args.signals_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str, str, int], dict[int, list[np.ndarray]]] = {}
    group_meta: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    facet_samples: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    processed_files = 0
    processed_signal_rows = 0

    for row in rows:
        label = optional_int(row.get("label"))
        if label not in (0, 1):
            continue
        b_ratio = optional_float(row.get(args.bin_column))
        if b_ratio is None or b_ratio <= 0:
            continue
        bin_index = find_bin(b_ratio, bin_edges)
        if bin_index is None:
            continue
        tpf = optional_float(row.get("tpf"))
        if tpf is None or tpf <= 0:
            failures.append(failure_row(row, "missing_or_invalid_tpf"))
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

        processed_files += 1
        teeth = string_value(row.get("teeth"), "unknown")
        cut_dir = string_value(row.get("cut_dir"), "unknown")
        bin_label = format_bin_label(bin_edges[bin_index], bin_edges[bin_index + 1])
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
            key = (signal_name, teeth, cut_dir, bin_index)
            facet_key = (signal_name, teeth, cut_dir)
            groups.setdefault(key, {0: [], 1: []})[label].append(dist)
            facet_samples.setdefault(facet_key, []).append(
                {
                    "b_ratio": b_ratio,
                    "bin_index": bin_index,
                    "label": label,
                    "dist": dist,
                }
            )
            group_meta.setdefault(
                key,
                {
                    "signal": signal_name,
                    "teeth": teeth,
                    "cut_dir": cut_dir,
                    "bin_left": bin_edges[bin_index],
                    "bin_right": bin_edges[bin_index + 1],
                    "bin_label": bin_label,
                },
            )
            processed_signal_rows += 1

    rng = np.random.default_rng(args.seed)
    divergence_rows = build_divergence_rows(
        groups,
        group_meta,
        bootstrap=args.bootstrap,
        min_per_label=args.min_per_label,
        rng=rng,
    )
    stable_reference_rows = build_stable_reference_rows(
        groups,
        group_meta,
        facet_samples,
        bootstrap=args.bootstrap,
        min_count=args.min_per_label,
        reference_mode=args.reference_mode,
        reference_quantile=args.reference_quantile,
        reference_max_b_ratio=args.reference_max_b_ratio,
        rng=rng,
    )
    write_csv(output_dir / "js_divergence_by_b_over_blim.csv", divergence_rows)
    write_csv(
        output_dir / "js_divergence_vs_stable_reference_by_b_over_blim.csv",
        stable_reference_rows,
    )
    write_csv(output_dir / "aggregation_failures.csv", failures)
    plot_status = write_plots(output_dir, stable_reference_rows, suffix="stable_reference")
    within_bin_plot_status = write_plots(output_dir, divergence_rows, suffix="within_bin")
    metadata = {
        "summary_csv": str(summary_path),
        "signals_dir": str(signals_dir),
        "output_dir": str(output_dir),
        "source_rows_after_dataset_filter": len(rows),
        "dataset_filter_ids": sorted(dataset_filter) if dataset_filter is not None else None,
        "signals": signals,
        "bin_column": args.bin_column,
        "ratio_bins": [format_float(edge) for edge in bin_edges],
        "ratio_grid_size": int(ratio_grid.size),
        "bootstrap": args.bootstrap,
        "min_per_label": args.min_per_label,
        "reference_mode": args.reference_mode,
        "reference_quantile": args.reference_quantile,
        "reference_max_b_ratio": args.reference_max_b_ratio,
        "method_warnings": method_warnings(args.reference_mode),
        "reference_overlap_note": (
            "The lowest b/blim bin may overlap the low_b_ratio reference set; "
            "treat that bin as an anchor, not an independent validation point."
        ),
        "processed_files": processed_files,
        "processed_signal_rows": processed_signal_rows,
        "groups": len(groups),
        "divergence_rows": len(divergence_rows),
        "ok_divergence_rows": sum(1 for row in divergence_rows if row.get("status") == "ok"),
        "stable_reference_rows": len(stable_reference_rows),
        "ok_stable_reference_rows": sum(
            1 for row in stable_reference_rows if row.get("status") == "ok"
        ),
        "failure_rows": len(failures),
        "plot_status": plot_status,
        "within_bin_plot_status": within_bin_plot_status,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=json_default)
    )
    print(json.dumps(metadata, indent=2, sort_keys=True, default=json_default))
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


def build_divergence_rows(
    groups: Mapping[tuple[str, str, str, int], Mapping[int, list[np.ndarray]]],
    group_meta: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
    bootstrap: int,
    min_per_label: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[0], item[1], item[2], item[3])):
        label_map = groups[key]
        stable = np.asarray(label_map.get(0, []), dtype=float)
        unstable = np.asarray(label_map.get(1, []), dtype=float)
        row = dict(group_meta[key])
        row.update(
            {
                "n_stable": int(stable.shape[0]),
                "n_unstable": int(unstable.shape[0]),
                "bootstrap_samples": int(bootstrap),
            }
        )
        if stable.shape[0] < min_per_label or unstable.shape[0] < min_per_label:
            row.update(
                {
                    "status": "insufficient_label_counts",
                    "js_divergence": math.nan,
                    "kl_unstable_to_stable": math.nan,
                    "kl_stable_to_unstable": math.nan,
                    "js_ci_low": math.nan,
                    "js_ci_high": math.nan,
                }
            )
            rows.append(row)
            continue

        stable_mean = normalize_distribution(np.mean(stable, axis=0))
        unstable_mean = normalize_distribution(np.mean(unstable, axis=0))
        js_value = jensen_shannon_divergence(unstable_mean, stable_mean)
        row.update(
            {
                "status": "ok",
                "js_divergence": js_value,
                "kl_unstable_to_stable": kl_divergence(unstable_mean, stable_mean),
                "kl_stable_to_unstable": kl_divergence(stable_mean, unstable_mean),
            }
        )
        if bootstrap > 0:
            samples = []
            stable_n = stable.shape[0]
            unstable_n = unstable.shape[0]
            for _ in range(bootstrap):
                stable_sample = stable[rng.integers(0, stable_n, stable_n)]
                unstable_sample = unstable[rng.integers(0, unstable_n, unstable_n)]
                samples.append(
                    jensen_shannon_divergence(
                        normalize_distribution(np.mean(unstable_sample, axis=0)),
                        normalize_distribution(np.mean(stable_sample, axis=0)),
                    )
                )
            row["js_ci_low"] = float(np.percentile(samples, 2.5))
            row["js_ci_high"] = float(np.percentile(samples, 97.5))
        else:
            row["js_ci_low"] = math.nan
            row["js_ci_high"] = math.nan
        rows.append(row)
    return rows


def build_stable_reference_rows(
    groups: Mapping[tuple[str, str, str, int], Mapping[int, list[np.ndarray]]],
    group_meta: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
    facet_samples: Mapping[tuple[str, str, str], list[Mapping[str, Any]]],
    bootstrap: int,
    min_count: int,
    reference_mode: str,
    reference_quantile: float,
    reference_max_b_ratio: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    stable_reference, reference_meta = build_reference_map(
        groups,
        facet_samples,
        min_count=min_count,
        reference_mode=reference_mode,
        reference_quantile=reference_quantile,
        reference_max_b_ratio=reference_max_b_ratio,
    )

    rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[0], item[1], item[2], item[3])):
        signal_name, teeth, cut_dir, _bin_index = key
        facet_key = (signal_name, teeth, cut_dir)
        label_map = groups[key]
        stable_ref = np.asarray(stable_reference.get(facet_key, []), dtype=float)
        ref_meta = reference_meta.get(facet_key, {})
        stable_bin = np.asarray(label_map.get(0, []), dtype=float)
        unstable_bin = np.asarray(label_map.get(1, []), dtype=float)
        bin_values_list = label_map.get(0, []) + label_map.get(1, [])
        bin_values = np.asarray(bin_values_list, dtype=float)

        row = dict(group_meta[key])
        row.update(
            {
                "curve_type": "bin_vs_stable_reference",
                "reference_mode": reference_mode,
                "reference_quantile": reference_quantile,
                "reference_max_b_ratio": reference_max_b_ratio,
                "reference_candidate_count": ref_meta.get("candidate_count", 0),
                "reference_eligible_count": ref_meta.get("eligible_count", 0),
                "n_stable_reference": int(stable_ref.shape[0]),
                "n_bin_total": int(bin_values.shape[0]),
                "n_bin_stable": int(stable_bin.shape[0]),
                "n_bin_unstable": int(unstable_bin.shape[0]),
                "bootstrap_samples": int(bootstrap),
            }
        )
        if stable_ref.shape[0] < min_count:
            row.update(
                {
                    "status": "insufficient_stable_reference",
                    "js_divergence": math.nan,
                    "kl_bin_to_stable_reference": math.nan,
                    "kl_stable_reference_to_bin": math.nan,
                    "js_ci_low": math.nan,
                    "js_ci_high": math.nan,
                }
            )
            rows.append(row)
            continue

        if bin_values.shape[0] < min_count:
            row.update(
                {
                    "status": "insufficient_bin_count",
                    "js_divergence": math.nan,
                    "kl_bin_to_stable_reference": math.nan,
                    "kl_stable_reference_to_bin": math.nan,
                    "js_ci_low": math.nan,
                    "js_ci_high": math.nan,
                }
            )
            rows.append(row)
            continue

        stable_mean = normalize_distribution(np.mean(stable_ref, axis=0))
        bin_mean = normalize_distribution(np.mean(bin_values, axis=0))
        row.update(
            {
                "status": "ok",
                "js_divergence": jensen_shannon_divergence(bin_mean, stable_mean),
                "kl_bin_to_stable_reference": kl_divergence(bin_mean, stable_mean),
                "kl_stable_reference_to_bin": kl_divergence(stable_mean, bin_mean),
            }
        )
        if bootstrap > 0:
            samples = []
            stable_n = stable_ref.shape[0]
            bin_n = bin_values.shape[0]
            for _ in range(bootstrap):
                stable_sample = stable_ref[rng.integers(0, stable_n, stable_n)]
                bin_sample = bin_values[rng.integers(0, bin_n, bin_n)]
                samples.append(
                    jensen_shannon_divergence(
                        normalize_distribution(np.mean(bin_sample, axis=0)),
                        normalize_distribution(np.mean(stable_sample, axis=0)),
                    )
                )
            row["js_ci_low"] = float(np.percentile(samples, 2.5))
            row["js_ci_high"] = float(np.percentile(samples, 97.5))
        else:
            row["js_ci_low"] = math.nan
            row["js_ci_high"] = math.nan
        rows.append(row)
    return rows


def build_reference_map(
    groups: Mapping[tuple[str, str, str, int], Mapping[int, list[np.ndarray]]],
    facet_samples: Mapping[tuple[str, str, str], list[Mapping[str, Any]]],
    min_count: int,
    reference_mode: str,
    reference_quantile: float,
    reference_max_b_ratio: float,
) -> tuple[dict[tuple[str, str, str], list[np.ndarray]], dict[tuple[str, str, str], dict[str, Any]]]:
    reference_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    if reference_mode == "label_stable":
        stable_reference: dict[tuple[str, str, str], list[np.ndarray]] = {}
        for key, label_map in groups.items():
            signal_name, teeth, cut_dir, _bin_index = key
            facet_key = (signal_name, teeth, cut_dir)
            stable_reference.setdefault(facet_key, []).extend(label_map.get(0, []))
        for facet_key, values in stable_reference.items():
            reference_meta[facet_key] = {
                "candidate_count": len(values),
                "eligible_count": len(values),
                "selected_count": len(values),
                "label_leakage_warning": "label_stable reference uses label=0 and is for methods comparison only",
            }
        return stable_reference, reference_meta

    if not 0.0 < reference_quantile <= 1.0:
        raise ValueError("--reference-quantile must be in (0, 1]")
    if reference_max_b_ratio <= 0.0:
        raise ValueError("--reference-max-b-ratio must be positive")

    stable_reference = {}
    for facet_key, samples in facet_samples.items():
        ordered = sorted(
            samples,
            key=lambda sample: float(sample["b_ratio"]),
        )
        eligible = [
            sample
            for sample in ordered
            if float(sample["b_ratio"]) <= reference_max_b_ratio
        ]
        reference_meta[facet_key] = {
            "candidate_count": len(ordered),
            "eligible_count": len(eligible),
            "reference_max_b_ratio": reference_max_b_ratio,
            "reference_quantile": reference_quantile,
        }
        if len(eligible) < min_count:
            stable_reference[facet_key] = []
            reference_meta[facet_key]["selected_count"] = 0
            continue
        count = max(min_count, int(math.ceil(len(ordered) * reference_quantile)))
        count = min(count, len(eligible))
        stable_reference[facet_key] = [
            np.asarray(sample["dist"], dtype=float)
            for sample in eligible[:count]
        ]
        reference_meta[facet_key]["selected_count"] = len(stable_reference[facet_key])
    return stable_reference, reference_meta


def write_plots(
    output_dir: Path,
    rows: list[Mapping[str, Any]],
    suffix: str = "divergence",
) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return {"ok": False, "message": "no ok divergence rows to plot"}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"ok": False, "message": f"matplotlib unavailable: {exc}"}

    plot_paths: list[str] = []
    signals = sorted({str(row["signal"]) for row in ok_rows})
    for signal_name in signals:
        signal_rows = [row for row in ok_rows if row["signal"] == signal_name]
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        facets = sorted({(str(row["teeth"]), str(row["cut_dir"])) for row in signal_rows})
        for teeth, cut_dir in facets:
            facet_rows = [
                row
                for row in signal_rows
                if str(row["teeth"]) == teeth and str(row["cut_dir"]) == cut_dir
            ]
            facet_rows.sort(key=lambda row: finite_midpoint(row["bin_left"], row["bin_right"]))
            x = [finite_midpoint(row["bin_left"], row["bin_right"]) for row in facet_rows]
            y = [float(row["js_divergence"]) for row in facet_rows]
            y_low = [float(row["js_ci_low"]) for row in facet_rows]
            y_high = [float(row["js_ci_high"]) for row in facet_rows]
            label = f"teeth={teeth}, cut_dir={cut_dir}"
            ax.plot(x, y, marker="o", linewidth=1.5, label=label)
            if all(math.isfinite(value) for value in y_low + y_high):
                ax.fill_between(x, y_low, y_high, alpha=0.15)
        ax.set_title(f"{signal_name} JS divergence ({suffix})")
        ax.set_xlabel("b / blim bin midpoint")
        ax.set_ylabel("Jensen-Shannon divergence")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        path = output_dir / f"js_divergence_{suffix}_{signal_name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(path))
    return {"ok": True, "paths": plot_paths}


def method_warnings(reference_mode: str) -> list[str]:
    if reference_mode == "label_stable":
        return [
            "REFERENCE_MODE=label_stable uses label=0 to define the reference. "
            "Use only for methods comparison; do not feed this artifact into final reporting or Phase 3 loss."
        ]
    return []


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


def parse_bin_edges(text: str) -> list[float]:
    edges = []
    for token in text.split(","):
        value = token.strip().lower()
        if value in {"inf", "+inf", "infinity"}:
            edges.append(math.inf)
        else:
            edges.append(float(value))
    if len(edges) < 2:
        raise ValueError("at least two bin edges are required")
    if any(edges[index] >= edges[index + 1] for index in range(len(edges) - 1)):
        raise ValueError(f"bin edges must be strictly increasing: {edges}")
    return edges


def find_bin(value: float, edges: list[float]) -> int | None:
    for index in range(len(edges) - 1):
        left = edges[index]
        right = edges[index + 1]
        if value >= left and (value < right or (math.isinf(right) and value <= right)):
            return index
    return None


def format_bin_label(left: float, right: float) -> str:
    return f"[{format_float(left)}, {format_float(right)})"


def finite_midpoint(left: Any, right: Any) -> float:
    left_value = float(left)
    right_value = float(right)
    if math.isinf(right_value):
        return left_value * 1.2 if left_value > 0 else left_value + 1.0
    return 0.5 * (left_value + right_value)


def failure_row(row: Mapping[str, Any], message: str) -> dict[str, Any]:
    return {
        "dataset": optional_int(row.get("dataset")),
        "file": row.get("file"),
        "label": optional_int(row.get("label")),
        "message": message,
    }


def string_value(value: Any, default: str) -> str:
    if value is None or value == "":
        return default
    parsed = optional_float(value)
    if parsed is not None and float(parsed).is_integer():
        return str(int(parsed))
    return str(value)


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


def format_float(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(parsed):
        return "inf" if parsed > 0 else "-inf"
    return f"{parsed:g}"


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
