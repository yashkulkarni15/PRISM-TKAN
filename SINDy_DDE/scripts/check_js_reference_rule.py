#!/usr/bin/env python3
"""Small synthetic checks for the Phase 2 JS stable-reference rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AGG_PATH = ROOT / "scripts" / "aggregate_js_divergence.py"


def load_aggregate_module():
    spec = importlib.util.spec_from_file_location("aggregate_js_divergence", AGG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {AGG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    agg = load_aggregate_module()
    rng = np.random.default_rng(7)
    base = np.asarray([0.7, 0.2, 0.1], dtype=float)
    shifted = np.asarray([0.1, 0.2, 0.7], dtype=float)

    groups = {
        ("xtpos", "4", "2", 3): {0: [], 1: [shifted, shifted, shifted]},
    }
    group_meta = {
        ("xtpos", "4", "2", 3): {
            "signal": "xtpos",
            "teeth": "4",
            "cut_dir": "2",
            "bin_left": 1.0,
            "bin_right": 1.25,
            "bin_label": "[1, 1.25)",
        }
    }
    facet_samples = {
        ("xtpos", "4", "2"): [
            {"b_ratio": 1.1, "dist": shifted},
            {"b_ratio": 1.2, "dist": shifted},
            {"b_ratio": 1.3, "dist": shifted},
        ]
    }
    rows = agg.build_stable_reference_rows(
        groups,
        group_meta,
        facet_samples,
        bootstrap=0,
        min_count=3,
        reference_mode="low_b_ratio",
        reference_quantile=0.10,
        reference_max_b_ratio=0.80,
        rng=rng,
    )
    assert rows[0]["status"] == "insufficient_stable_reference", rows[0]

    groups[("xtpos", "4", "2", 0)] = {0: [base, base, base], 1: []}
    group_meta[("xtpos", "4", "2", 0)] = {
        "signal": "xtpos",
        "teeth": "4",
        "cut_dir": "2",
        "bin_left": 0.0,
        "bin_right": 0.5,
        "bin_label": "[0, 0.5)",
    }
    facet_samples[("xtpos", "4", "2")] = [
        {"b_ratio": 0.35, "dist": base},
        {"b_ratio": 0.45, "dist": base},
        {"b_ratio": 0.55, "dist": base},
        {"b_ratio": 1.1, "dist": shifted},
        {"b_ratio": 1.2, "dist": shifted},
        {"b_ratio": 1.3, "dist": shifted},
    ]
    rows = agg.build_stable_reference_rows(
        groups,
        group_meta,
        facet_samples,
        bootstrap=0,
        min_count=3,
        reference_mode="low_b_ratio",
        reference_quantile=0.10,
        reference_max_b_ratio=0.80,
        rng=rng,
    )
    by_bin = {row["bin_label"]: row for row in rows}
    assert by_bin["[1, 1.25)"]["status"] == "ok", by_bin["[1, 1.25)"]
    assert by_bin["[1, 1.25)"]["n_stable_reference"] == 3, by_bin["[1, 1.25)"]
    assert by_bin["[1, 1.25)"]["js_divergence"] > 0.0, by_bin["[1, 1.25)"]

    print("reference rule checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
