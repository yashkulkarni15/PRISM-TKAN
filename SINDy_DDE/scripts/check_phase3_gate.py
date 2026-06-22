#!/usr/bin/env python3
"""Evaluate the Phase 3 trajectory reproduction gate from numpy arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pri2m.models.stage_gates import TrajectoryGateConfig, evaluate_trajectory_gate


def _load_array(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        data = np.load(p)
        if len(data.files) != 1:
            raise ValueError(f"{p} must contain exactly one array or use .npy")
        return data[data.files[0]]
    raise ValueError(f"unsupported array file: {p}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-true", required=True)
    parser.add_argument("--stable-pred", required=True)
    parser.add_argument("--unstable-true", required=True)
    parser.add_argument("--unstable-pred", required=True)
    parser.add_argument("--true-onset", type=float, required=True)
    parser.add_argument("--predicted-onset", type=float, required=True)
    parser.add_argument("--previous-min-r2", type=float)
    parser.add_argument("--min-r2", type=float, default=0.85)
    parser.add_argument("--max-onset-error-pct", type=float, default=5.0)
    parser.add_argument("--max-regression-r2-drop", type=float, default=0.03)
    args = parser.parse_args()

    previous = None
    if args.previous_min_r2 is not None:
        previous = {"min_r2": args.previous_min_r2}

    result = evaluate_trajectory_gate(
        stable_true=_load_array(args.stable_true),
        stable_pred=_load_array(args.stable_pred),
        unstable_true=_load_array(args.unstable_true),
        unstable_pred=_load_array(args.unstable_pred),
        true_onset=args.true_onset,
        predicted_onset=args.predicted_onset,
        previous_metrics=previous,
        config=TrajectoryGateConfig(
            min_r2=args.min_r2,
            max_onset_error_pct=args.max_onset_error_pct,
            max_regression_r2_drop=args.max_regression_r2_drop,
        ),
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
