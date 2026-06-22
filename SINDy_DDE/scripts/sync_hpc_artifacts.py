#!/usr/bin/env python3
"""Sync source and lightweight SINDy artifacts from Great Lakes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pri2m.hpc.sync import copy_artifacts, verify_local_sindy_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline",
        help="Great Lakes legacy SINDy pipeline source directory.",
    )
    parser.add_argument(
        "--dest-root",
        default=str(ROOT / "sindy_pipeline"),
        help="Local legacy SINDy pipeline destination directory.",
    )
    parser.add_argument("--execute", action="store_true", help="Copy files instead of dry run.")
    parser.add_argument("--verify", action="store_true", help="Verify local lightweight summaries.")
    args = parser.parse_args()

    result = copy_artifacts(args.source_root, args.dest_root, dry_run=not args.execute)
    print(json.dumps(result, indent=2))

    if result["missing_required_patterns"] and args.execute:
        return 2

    if args.verify:
        verification = verify_local_sindy_summary(args.dest_root)
        print(json.dumps(verification, indent=2))
        return 0 if verification["ok"] else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
