#!/usr/bin/env python3
"""Apply the PRI2M MLflow checkpoint retention rule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pri2m.training.mlflow_retention import apply_checkpoint_retention


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="MLflow run artifact directory or run directory.")
    parser.add_argument("--execute", action="store_true", help="Delete intermediate checkpoints.")
    args = parser.parse_args()

    plan = apply_checkpoint_retention(args.run_dir, dry_run=not args.execute)
    print(json.dumps({"run_dir": plan.run_dir, "keep": plan.keep, "delete": plan.delete}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
