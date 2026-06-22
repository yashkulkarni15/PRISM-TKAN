#!/usr/bin/env python3
"""Print a compact Phase 1-3 status report from local artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pri2m.reports.phase_status import build_status_report


def main() -> int:
    print(build_status_report(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
