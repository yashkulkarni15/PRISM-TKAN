"""Checkpoint retention utilities for MLflow artifact directories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


CHECKPOINT_EXTENSIONS = {".ckpt", ".pt", ".pth"}


@dataclass(frozen=True)
class RetentionPlan:
    run_dir: str
    keep: list[str]
    delete: list[str]


def plan_checkpoint_retention(run_dir: str | Path) -> RetentionPlan:
    """Keep best and final checkpoints; mark periodic intermediates for deletion."""
    root = Path(run_dir)
    checkpoints = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix in CHECKPOINT_EXTENSIONS
    )
    keep: set[Path] = set()

    for path in checkpoints:
        lower = path.name.lower()
        if "best" in lower or "final" in lower:
            keep.add(path)

    if checkpoints and not any("final" in p.name.lower() for p in keep):
        keep.add(max(checkpoints, key=_epoch_sort_key))
    if checkpoints and not any("best" in p.name.lower() for p in keep):
        keep.add(min(checkpoints, key=_loss_sort_key))

    delete = [p for p in checkpoints if p not in keep]
    return RetentionPlan(
        run_dir=str(root),
        keep=[str(p) for p in sorted(keep)],
        delete=[str(p) for p in sorted(delete)],
    )


def apply_checkpoint_retention(run_dir: str | Path, dry_run: bool = True) -> RetentionPlan:
    """Apply the retention plan. Dry-run by default for quota-safe review."""
    plan = plan_checkpoint_retention(run_dir)
    if not dry_run:
        for path in plan.delete:
            Path(path).unlink(missing_ok=True)
    return plan


def _epoch_sort_key(path: Path) -> tuple[int, float]:
    match = re.search(r"epoch[=_-]?(\d+)", path.name.lower())
    epoch = int(match.group(1)) if match else -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return epoch, mtime


def _loss_sort_key(path: Path) -> float:
    lower = path.name.lower()
    match = re.search(r"(?:val[_-]?loss|loss)[=_-]?([0-9.]+)", lower)
    if not match:
        return float("inf")
    try:
        return float(match.group(1).rstrip("."))
    except ValueError:
        return float("inf")
