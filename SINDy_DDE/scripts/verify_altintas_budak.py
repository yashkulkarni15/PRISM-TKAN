#!/usr/bin/env python3
"""Verifier CLI for the PRI2M Altintas-Budak stability-lobe slice."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pri2m.physics.altintas_budak import (  # noqa: E402
    CuttingParameters,
    ModalParameters,
    candidate_stability_lobes,
    nearest_point_for_speed,
)


DEFAULT_DATASET = REPO_ROOT / "data" / "metadata" / "features_all_datasets.csv"
REFERENCE_NOTES = REPO_ROOT / "docs" / "altintas_budak_reference_notes.md"

BLIM_REQUIRED_COLUMNS = {
    "omega",
    "blim",
    "fn",
    "ktc",
    "teeth",
    "cut_dir",
    "ktx",
    "zetatx",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    strict_failure: bool = False


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _as_float(row: dict[str, str], name: str, default: float | None = None) -> float:
    value = row.get(name, "")
    if value == "" or value is None:
        if default is None:
            raise ValueError(f"missing numeric field {name!r}")
        return default
    return float(value)


def _as_int(row: dict[str, str], name: str, default: int | None = None) -> int:
    value = row.get(name, "")
    if value == "" or value is None:
        if default is None:
            raise ValueError(f"missing integer field {name!r}")
        return default
    return int(float(value))


def _numeric_values(rows: Iterable[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(column, "")
        if value == "":
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if math.isfinite(parsed):
            values.append(parsed)
    return values


def _value_counts(rows: Iterable[dict[str, str]], column: str, limit: int = 12) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(column, "")
        counts[value] = counts.get(value, 0) + 1
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    shown = ", ".join(f"{key}:{value}" for key, value in items[:limit])
    if len(items) > limit:
        shown += f", ... (+{len(items) - limit} more)"
    return shown or "n/a"


def _range_text(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{min(values):.6g} .. {max(values):.6g}"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _deterministic_sample(rows: list[dict[str, str]], max_rows: int) -> list[dict[str, str]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    if max_rows == 1:
        return [rows[0]]
    step = (len(rows) - 1) / (max_rows - 1)
    return [rows[round(index * step)] for index in range(max_rows)]


def run_textbook_smoke(strict: bool, grid_points: int, max_lobe_order: int) -> CheckResult:
    print("\n=== Textbook Smoke Check ===")
    print(f"Reference notes: {REFERENCE_NOTES}")
    print("Reference constants: PLACEHOLDER - exact Altintas/Budak textbook case is not local.")

    modal_x = ModalParameters(natural_frequency_hz=500.0, stiffness_n_per_m=5.0e6, damping_ratio=0.03)
    modal_y = ModalParameters(natural_frequency_hz=500.0, stiffness_n_per_m=5.0e6, damping_ratio=0.03)
    cutting = CuttingParameters(
        teeth=4,
        tangential_force_coefficient_n_per_m2=7.0e8,
        radial_to_tangential_ratio=0.0,
        radial_immersion=0.5,
        cut_dir="down",
    )

    effective_grid_points = max(grid_points, 2)
    frequencies = [
        425.0 + (175.0 * index / (effective_grid_points - 1))
        for index in range(effective_grid_points)
    ]
    points = candidate_stability_lobes(
        modal_x,
        cutting,
        modal_y=modal_y,
        chatter_frequencies_hz=frequencies,
        lobe_orders=range(max_lobe_order + 1),
    )

    positive_depths = [point.axial_depth_mm for point in points if point.axial_depth_m > 0.0]
    positive_speeds = [point.spindle_speed_rpm for point in points if point.spindle_speed_rpm > 0.0]
    operational = bool(points and positive_depths and positive_speeds)
    if operational:
        print(f"Generated candidate points: {len(points)}")
        print(f"Spindle speed range rpm: {_range_text(positive_speeds)}")
        print(f"blim range mm: {_range_text(positive_depths)}")
    else:
        print("Generated candidate points: 0")

    if strict:
        print("STATUS: FAIL - strict textbook check requested but exact reference constants are placeholders.")
        return CheckResult("textbook-smoke", "FAIL", strict_failure=True)

    status = "PASS" if operational else "WARN"
    detail = "operational smoke passed; numeric textbook benchmark remains pending" if operational else "no lobe points generated"
    print(f"STATUS: {status} - {detail}")
    return CheckResult("textbook-smoke", status)


def run_metadata_summary(dataset_path: Path, strict: bool) -> CheckResult:
    print("\n=== Dev Dataset Metadata Summary ===")
    print(f"CSV: {dataset_path}")
    if not dataset_path.exists():
        status = "FAIL" if strict else "WARN"
        print(f"STATUS: {status} - metadata CSV not found")
        return CheckResult("metadata-summary", status, strict_failure=strict)

    columns, rows = _read_rows(dataset_path)
    missing = sorted(BLIM_REQUIRED_COLUMNS.difference(columns))
    print(f"Rows: {len(rows)}")
    print(f"Columns: {len(columns)}")
    print(f"Required blim scaffold columns missing: {missing or 'none'}")
    print(f"Datasets: {_range_text(_numeric_values(rows, 'dataset'))}")
    print(f"Label counts: {_value_counts(rows, 'label')}")
    print(f"cut_dir counts: {_value_counts(rows, 'cut_dir')}")
    print(f"teeth counts: {_value_counts(rows, 'teeth')}")
    print(f"noise_std counts: {_value_counts(rows, 'noise_std')}")

    for column in ("omega", "b_mm", "blim", "b_over_blim", "fn", "ktc", "ktx", "zetatx", "diameter_mm"):
        if column in columns:
            print(f"{column} range: {_range_text(_numeric_values(rows, column))}")

    print("Raw H5/pickle artifacts opened: none")
    if missing and strict:
        print("STATUS: FAIL - strict metadata check requires all scaffold columns")
        return CheckResult("metadata-summary", "FAIL", strict_failure=True)

    print("STATUS: PASS - metadata CSV is readable without raw artifacts")
    return CheckResult("metadata-summary", "PASS")


def _row_to_parameters(row: dict[str, str]) -> tuple[ModalParameters, ModalParameters, CuttingParameters]:
    fn_x = _as_float(row, "fn")
    fn_y = _as_float(row, "fnty", fn_x)
    stiffness = _as_float(row, "ktx")
    damping = _as_float(row, "zetatx", _as_float(row, "damping", 0.03))
    radial_ratio = _as_float(row, "kr", _as_float(row, "ro", 0.0))
    radial_immersion = _as_float(row, "radial_immersion", _as_float(row, "ae_over_d", 1.0))

    modal_x = ModalParameters(fn_x, stiffness, damping, label="x")
    modal_y = ModalParameters(fn_y, stiffness, damping, label="y")
    cutting = CuttingParameters(
        teeth=_as_int(row, "teeth"),
        tangential_force_coefficient_n_per_m2=_as_float(row, "ktc"),
        radial_to_tangential_ratio=radial_ratio,
        radial_immersion=radial_immersion,
        cut_dir=row.get("cut_dir", "slot"),
    )
    return modal_x, modal_y, cutting


def _parameter_key(row: dict[str, str]) -> tuple[str, ...]:
    key_columns = (
        "fn",
        "fnty",
        "ktx",
        "zetatx",
        "damping",
        "ktc",
        "teeth",
        "cut_dir",
        "ro",
        "kr",
        "radial_immersion",
        "ae_over_d",
    )
    return tuple(row.get(column, "") for column in key_columns)


def run_dataset_blim(
    dataset_path: Path,
    strict: bool,
    max_rows: int,
    grid_points: int,
    max_lobe_order: int,
    rel_tol: float,
) -> CheckResult:
    print("\n=== Dataset blim Validation Scaffold ===")
    print(f"CSV: {dataset_path}")
    if not dataset_path.exists():
        status = "FAIL" if strict else "WARN"
        print(f"STATUS: {status} - dataset CSV not found")
        return CheckResult("dataset-blim", status, strict_failure=strict)

    columns, rows = _read_rows(dataset_path)
    missing = sorted(BLIM_REQUIRED_COLUMNS.difference(columns))
    if missing:
        status = "FAIL" if strict else "WARN"
        print(f"Missing required columns: {missing}")
        print(f"STATUS: {status} - scaffold cannot compute comparable blim values")
        return CheckResult("dataset-blim", status, strict_failure=strict)

    sample = _deterministic_sample(rows, max_rows)
    print(f"Rows in CSV: {len(rows)}")
    print(f"Rows sampled deterministically: {len(sample)}")
    print("Assumptions: two-axis SDOF pair, dataset ktx reused for y stiffness, full slot when ae/D is absent.")

    point_cache: dict[tuple[str, ...], list] = {}
    blim_errors: list[float] = []
    speed_errors: list[float] = []
    skipped = 0
    examples: list[str] = []

    for row in sample:
        try:
            target_rpm = _as_float(row, "omega")
            reference_blim_mm = _as_float(row, "blim")
            if target_rpm <= 0.0 or reference_blim_mm <= 0.0:
                skipped += 1
                continue

            key = _parameter_key(row)
            if key not in point_cache:
                modal_x, modal_y, cutting = _row_to_parameters(row)
                lower = max(1e-9, min(modal_x.natural_frequency_hz, modal_y.natural_frequency_hz) * 0.65)
                upper = max(modal_x.natural_frequency_hz, modal_y.natural_frequency_hz) * 1.35
                effective_grid_points = max(grid_points, 2)
                frequencies = [
                    lower + ((upper - lower) * index / (effective_grid_points - 1))
                    for index in range(effective_grid_points)
                ]
                point_cache[key] = candidate_stability_lobes(
                    modal_x,
                    cutting,
                    modal_y=modal_y,
                    chatter_frequencies_hz=frequencies,
                    lobe_orders=range(max_lobe_order + 1),
                )

            nearest = nearest_point_for_speed(point_cache[key], target_rpm)
            if nearest is None:
                skipped += 1
                continue

            speed_rel_error = abs(nearest.spindle_speed_rpm - target_rpm) / target_rpm
            blim_rel_error = abs(nearest.axial_depth_mm - reference_blim_mm) / reference_blim_mm
            speed_errors.append(speed_rel_error)
            blim_errors.append(blim_rel_error)
            if len(examples) < 4:
                examples.append(
                    "row file={file} rpm={rpm:.3f} nearest={nearest:.3f} "
                    "ref_blim_mm={ref:.6g} calc_blim_mm={calc:.6g}".format(
                        file=row.get("file", "?"),
                        rpm=target_rpm,
                        nearest=nearest.spindle_speed_rpm,
                        ref=reference_blim_mm,
                        calc=nearest.axial_depth_mm,
                    )
                )
        except (ValueError, ZeroDivisionError, FloatingPointError):
            skipped += 1

    print(f"Usable comparisons: {len(blim_errors)}")
    print(f"Skipped rows: {skipped}")
    for example in examples:
        print(f"Example: {example}")

    if not blim_errors:
        status = "FAIL" if strict else "WARN"
        print(f"STATUS: {status} - no usable comparisons generated")
        return CheckResult("dataset-blim", status, strict_failure=strict)

    median_blim = statistics.median(blim_errors)
    p90_blim = _percentile(blim_errors, 0.90)
    median_speed = statistics.median(speed_errors)
    p90_speed = _percentile(speed_errors, 0.90)
    print(f"blim relative error median/p90/max: {median_blim:.4g} / {p90_blim:.4g} / {max(blim_errors):.4g}")
    print(f"speed relative miss median/p90/max: {median_speed:.4g} / {p90_speed:.4g} / {max(speed_errors):.4g}")

    strict_ok = median_blim <= rel_tol and median_speed <= rel_tol
    if strict and not strict_ok:
        print(f"STATUS: FAIL - strict scaffold tolerance {rel_tol:.4g} was not met")
        return CheckResult("dataset-blim", "FAIL", strict_failure=True)

    if strict_ok:
        print("STATUS: PASS - scaffold ran and median errors are within tolerance")
        return CheckResult("dataset-blim", "PASS")

    print("STATUS: WARN - scaffold ran, but constants/engagement assumptions need calibration")
    return CheckResult("dataset-blim", "WARN")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "textbook-smoke", "dataset-blim", "dev-metadata", "metadata-summary"),
        default="all",
        help="Verifier mode to run.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Lightweight feature CSV path.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when the requested check fails.")
    parser.add_argument("--max-rows", type=int, default=160, help="Rows to sample for dataset-blim mode.")
    parser.add_argument("--grid-points", type=int, default=301, help="Chatter-frequency grid points.")
    parser.add_argument("--max-lobe-order", type=int, default=16, help="Maximum lobe order to search.")
    parser.add_argument("--rel-tol", type=float, default=0.15, help="Strict median relative-error tolerance.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dataset_path = args.dataset if args.dataset.is_absolute() else REPO_ROOT / args.dataset

    requested = [args.mode]
    if args.mode == "all":
        requested = ["textbook-smoke", "dataset-blim", "dev-metadata"]
    elif args.mode == "metadata-summary":
        requested = ["dev-metadata"]

    results: list[CheckResult] = []
    if "textbook-smoke" in requested:
        results.append(run_textbook_smoke(args.strict, args.grid_points, args.max_lobe_order))
    if "dataset-blim" in requested:
        results.append(
            run_dataset_blim(
                dataset_path,
                args.strict,
                args.max_rows,
                args.grid_points,
                args.max_lobe_order,
                args.rel_tol,
            )
        )
    if "dev-metadata" in requested:
        results.append(run_metadata_summary(dataset_path, args.strict))

    strict_failures = [result for result in results if result.strict_failure]
    statuses = ", ".join(f"{result.name}:{result.status}" for result in results)
    if strict_failures:
        print(f"\nOPERATIONAL ACCEPTANCE: FAIL - {statuses}")
        return 2

    print(f"\nOPERATIONAL ACCEPTANCE: PASS - {statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
