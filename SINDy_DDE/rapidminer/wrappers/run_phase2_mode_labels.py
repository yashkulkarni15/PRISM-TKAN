#!/usr/bin/env python3
"""RapidMiner wrapper: Phase 2 mode label assignment.

Calls assign_signal_modes.py and writes mode_labels.csv to --output-dir.
Operator 6 in the six-operator chain. Reads noise_metrics.csv from operator 5.

Usage (RapidMiner Execute Program):
    /opt/anaconda3/bin/python \
        /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/wrappers/run_phase2_mode_labels.py \
        --noise-metrics-csv /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/outputs/phase2/noise_metrics.csv \
        --output-dir /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/outputs/phase2

Outputs (in --output-dir):
    mode_labels.csv           one row per file with mode_label and confidence
    mode_labels_summary.json  threshold settings and mode distribution counts
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--noise-metrics-csv",
        default=str(REPO_ROOT / "rapidminer" / "outputs" / "phase2" / "noise_metrics.csv"),
        help="noise_metrics.csv produced by run_phase2_noise_metrics.py.",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(REPO_ROOT / "data" / "metadata" / "features_all_datasets.csv"),
        help="Used to backfill b_over_blim when missing from noise_metrics.csv.",
    )
    parser.add_argument(
        "--noisy-summary-csv",
        default=str(REPO_ROOT / "data" / "metadata" / "features_noisy_datasets.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "rapidminer" / "outputs" / "phase2"),
    )
    parser.add_argument(
        "--calibration-json",
        default=None,
        help="Optional promoted_metrics.json. Falls back to <output-dir>/promoted_metrics.json.",
    )
    parser.add_argument("--omega-low", type=float, default=None)
    parser.add_argument("--omega-high", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script = REPO_ROOT / "scripts" / "assign_signal_modes.py"
    cmd = [
        sys.executable, str(script),
        "--noise-metrics-csv", args.noise_metrics_csv,
        "--summary-csv", args.summary_csv,
        "--noisy-summary-csv", args.noisy_summary_csv,
        "--output-dir", str(out_dir),
    ]
    if args.calibration_json:
        cmd += ["--calibration-json", args.calibration_json]
    if args.omega_low is not None:
        cmd += ["--omega-low", str(args.omega_low)]
    if args.omega_high is not None:
        cmd += ["--omega-high", str(args.omega_high)]

    print(f"[phase2-mode-labels] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)

    out_csv = out_dir / "mode_labels.csv"
    if out_csv.exists():
        import pandas as pd
        df = pd.read_csv(out_csv)
        if "mode_label" in df.columns:
            dist = df["mode_label"].value_counts().to_dict()
            print(f"[phase2-mode-labels] done -> {out_csv}  ({len(df)} rows)")
            print(f"[phase2-mode-labels] mode distribution: {dist}")
        else:
            print(f"[phase2-mode-labels] done -> {out_csv}  ({len(df)} rows)")
    else:
        print(f"[phase2-mode-labels] WARNING: mode_labels.csv not found at {out_csv}",
              file=sys.stderr)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
