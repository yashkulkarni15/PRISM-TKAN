#!/usr/bin/env python3
"""RapidMiner wrapper: Phase 1 Altintas-Budak physics verifier.

Calls verify_altintas_budak.py and writes a one-row physics_check.csv to
--output-dir. RapidMiner reads that CSV to branch: PASS proceeds to Phase 2;
FAIL stops at a physics-anchor-unresolved report.

Usage (RapidMiner Execute Program):
    /opt/anaconda3/bin/python \
        /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/wrappers/run_phase1_altintas_budak.py \
        --output-dir /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/outputs/phase1
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        default=str(REPO_ROOT / "data" / "metadata" / "features_all_datasets.csv"),
        help="Feature CSV for dataset-blim validation.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "rapidminer" / "outputs" / "phase1"),
        help="Directory for physics_check.csv.",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "textbook-smoke", "dataset-blim"),
        default="all",
        help="Verifier mode passed to verify_altintas_budak.py.",
    )
    parser.add_argument("--max-rows", type=int, default=160)
    parser.add_argument("--rel-tol", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verifier = REPO_ROOT / "scripts" / "verify_altintas_budak.py"
    cmd = [
        sys.executable,
        str(verifier),
        "--mode", args.mode,
        "--dataset", args.summary_csv,
        "--max-rows", str(args.max_rows),
        "--rel-tol", str(args.rel_tol),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    status = "UNKNOWN"
    physics_pass = False
    checks_summary = ""
    for line in stdout.splitlines():
        if "OPERATIONAL ACCEPTANCE:" in line:
            acceptance_part = line.split("OPERATIONAL ACCEPTANCE:")[-1].strip()
            if acceptance_part.upper().startswith("PASS"):
                status = "PASS"
                physics_pass = True
            else:
                status = "FAIL"
                physics_pass = False
            parts = acceptance_part.split("-", 1)
            if len(parts) > 1:
                checks_summary = parts[1].strip()
            break

    out_csv = out_dir / "physics_check.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["run_ts", "mode", "status", "physics_pass", "checks_summary", "returncode"],
        )
        writer.writeheader()
        writer.writerow({
            "run_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": args.mode,
            "status": status,
            "physics_pass": "TRUE" if physics_pass else "FALSE",
            "checks_summary": checks_summary,
            "returncode": result.returncode,
        })

    # Print full verifier output so RapidMiner logs it.
    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    print(f"\n[phase1-wrapper] physics_check.csv -> {out_csv}")
    print(f"[phase1-wrapper] physics_pass={physics_pass}  returncode={result.returncode}")
    return 0 if physics_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
