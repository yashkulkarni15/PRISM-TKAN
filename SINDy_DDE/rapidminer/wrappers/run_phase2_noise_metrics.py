#!/usr/bin/env python3
"""RapidMiner wrapper: Phase 2 noise metric computation.

Calls compute_noise_metrics.py and writes noise_metrics.csv to --output-dir.
Operator 5 in the six-operator chain. Requires NPZ bundles from the cluster
(produced by extract_phase2_signals.py on Great Lakes).

Usage (RapidMiner Execute Program):
    /opt/anaconda3/bin/python \
        /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/wrappers/run_phase2_noise_metrics.py \
        --datasets 1-5 \
        --signals-dir /path/to/npz/bundles \
        --output-dir /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/outputs/phase2

Outputs (in --output-dir):
    noise_metrics.csv    one row per file, 44 metric columns
    run_metadata.json    config and row counts
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
        "--datasets",
        required=True,
        help="Dataset selector such as '1', '1,4,29', or '1-5'.",
    )
    parser.add_argument(
        "--signals-dir",
        required=True,
        help="Directory of NPZ bundles from extract_phase2_signals.py (cluster-produced).",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(REPO_ROOT / "data" / "metadata" / "features_all_datasets.csv"),
    )
    parser.add_argument(
        "--noisy-summary-csv",
        default=str(REPO_ROOT / "data" / "metadata" / "features_noisy_datasets.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "rapidminer" / "outputs" / "phase2"),
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on total rows processed (smoke testing).")
    parser.add_argument("--max-files-per-dataset", type=int, default=None)
    parser.add_argument("--fs", type=float, default=None,
                        help="Fallback sample rate (Hz) when NPZ omits fs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script = REPO_ROOT / "scripts" / "compute_noise_metrics.py"
    cmd = [
        sys.executable, str(script),
        "--datasets", args.datasets,
        "--signals-dir", args.signals_dir,
        "--summary-csv", args.summary_csv,
        "--noisy-summary-csv", args.noisy_summary_csv,
        "--output-dir", str(out_dir),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.max_files_per_dataset is not None:
        cmd += ["--max-files-per-dataset", str(args.max_files_per_dataset)]
    if args.fs is not None:
        cmd += ["--fs", str(args.fs)]

    print(f"[phase2-noise-metrics] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)

    out_csv = out_dir / "noise_metrics.csv"
    if out_csv.exists():
        import pandas as pd
        df = pd.read_csv(out_csv)
        print(f"[phase2-noise-metrics] done -> {out_csv}  ({len(df)} rows, {df.shape[1]} cols)")
    else:
        print(f"[phase2-noise-metrics] WARNING: noise_metrics.csv not found at {out_csv}",
              file=sys.stderr)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
