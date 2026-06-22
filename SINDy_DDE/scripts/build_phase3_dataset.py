#!/usr/bin/env python3
"""Build the leakage-safe, hygiene-cleaned Phase 3 training dataset.

Composes the three audit fixes (splits / labels / weights) into one table and
writes:
    phase3_dataset.csv   one row per file + parent_group, sample_weight, flags
    phase3_folds.json    group->fold map (mirror-safe), dev groups, fold summary
    run_metadata.json    exclusion + weight summary

Usage:
    python scripts/build_phase3_dataset.py \
        --features-clean data/metadata/features_all_datasets.csv \
        --features-noisy data/metadata/features_noisy_datasets.csv \
        --mode-labels    rapidminer/outputs/phase2/mode_labels.csv \
        --output-dir     results/phase3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pri2m.phase3.data_prep import build_phase3_table, fold_assignments, summarize


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-clean", default=str(repo / "data/metadata/features_all_datasets.csv"))
    p.add_argument("--features-noisy", default=str(repo / "data/metadata/features_noisy_datasets.csv"))
    p.add_argument("--mode-labels", default=str(repo / "rapidminer/outputs/phase2/mode_labels.csv"))
    p.add_argument("--output-dir", default=str(repo / "results/phase3"))
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--upper-q", type=float, default=0.99)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_phase3_table(
        args.features_clean, args.features_noisy, args.mode_labels, upper_q=args.upper_q
    )
    folds = fold_assignments(table, n_splits=args.n_splits, seed=args.seed)
    meta = summarize(table)
    meta["inputs"] = {
        "features_clean": args.features_clean,
        "features_noisy": args.features_noisy,
        "mode_labels": args.mode_labels,
        "upper_q": args.upper_q,
    }

    table.to_csv(out_dir / "phase3_dataset.csv", index=False)
    (out_dir / "phase3_folds.json").write_text(json.dumps(folds, indent=2))
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))
    print(f"\n[build_phase3_dataset] wrote dataset/folds/metadata to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
