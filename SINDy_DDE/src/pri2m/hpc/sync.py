"""Plan-safe Great Lakes sync helpers.

The local desktop often cannot mount `/nfs/turbo`. These utilities make the
desired artifact contract explicit and support dry-run verification.
"""

from __future__ import annotations

import fnmatch
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SyncItem:
    pattern: str
    required: bool = False


DEFAULT_SOURCE_FILES = [
    SyncItem("read_file.py", required=True),
    SyncItem("RMSsliding.py", required=True),
    SyncItem("build_state.py", required=True),
    SyncItem("sindy_dde.py", required=True),
]

DEFAULT_LIGHTWEIGHT_OUTPUTS = [
    SyncItem("preprocessed_log.csv", required=True),
    SyncItem("rejection_log.csv", required=True),
    SyncItem("sindy_fit_failures.csv", required=False),
    SyncItem("sindy_fits/dataset_*.csv", required=True),
    SyncItem("sindy_analysis/*.csv", required=False),
    SyncItem("sindy_analysis/*.png", required=False),
    SyncItem("sindy_analysis/*.pdf", required=False),
    SyncItem("sindy_analysis/*.svg", required=False),
]


def expand_patterns(source_root: Path, items: Iterable[SyncItem]) -> tuple[list[Path], list[str]]:
    """Expand relative glob patterns under a Great Lakes source root."""
    found: list[Path] = []
    missing_required: list[str] = []
    for item in items:
        matches = sorted(source_root.glob(item.pattern))
        if not matches and item.required:
            missing_required.append(item.pattern)
        found.extend([p for p in matches if p.is_file()])
    return found, missing_required


def copy_artifacts(source_root: str | Path, dest_root: str | Path, dry_run: bool = True) -> dict:
    """Copy only source/helper and summary artifacts, preserving relative paths."""
    src = Path(source_root)
    dst = Path(dest_root)
    items = [*DEFAULT_SOURCE_FILES, *DEFAULT_LIGHTWEIGHT_OUTPUTS]
    matches, missing = expand_patterns(src, items)
    copied: list[str] = []

    for path in matches:
        rel = path.relative_to(src)
        target = dst / rel
        copied.append(str(rel))
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    return {
        "source_root": str(src),
        "dest_root": str(dst),
        "dry_run": dry_run,
        "available": src.exists(),
        "matched_files": copied,
        "missing_required_patterns": missing,
    }


def verify_local_sindy_summary(
    sindy_dir: str | Path,
    expected_rows: int = 32168,
    expected_datasets: int = 120,
    expected_fit_failures: int = 0,
) -> dict:
    """Verify lightweight SINDy summary files after sync."""
    root = Path(sindy_dir)
    preprocessed = root / "preprocessed_log.csv"
    result = {
        "sindy_dir": str(root),
        "preprocessed_log_exists": preprocessed.exists(),
        "preprocessed_rows": None,
        "dataset_count": None,
        "fit_failures": None,
        "ok": False,
        "messages": [],
    }
    messages: list[str] = result["messages"]

    if preprocessed.exists():
        df = pd.read_csv(preprocessed)
        result["preprocessed_rows"] = int(len(df))
        result["dataset_count"] = int(df["dataset"].nunique()) if "dataset" in df else None
        if result["preprocessed_rows"] != expected_rows:
            messages.append(f"expected {expected_rows} preprocessed rows")
        if result["dataset_count"] != expected_datasets:
            messages.append(f"expected {expected_datasets} datasets")
    else:
        messages.append("missing preprocessed_log.csv")

    failure_log = root / "sindy_fit_failures.csv"
    if failure_log.exists():
        failures = pd.read_csv(failure_log)
        result["fit_failures"] = int(len(failures))
    else:
        log_failures = _count_fit_failures_from_logs(root)
        result["fit_failures"] = log_failures

    if result["fit_failures"] not in {None, expected_fit_failures}:
        messages.append(f"expected {expected_fit_failures} SINDy fit failures")

    result["ok"] = not messages
    return result


def _count_fit_failures_from_logs(root: Path) -> int | None:
    logs = sorted(root.glob("fit_*.log"))
    if not logs:
        return None
    failures = 0
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("Failed:"):
                try:
                    failures += int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    return None
    return failures


def filter_relative_paths(paths: Iterable[str], pattern: str) -> list[str]:
    """Small helper used by report scripts for glob-like relative filtering."""
    return [p for p in paths if fnmatch.fnmatch(p, pattern)]
