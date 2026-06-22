#!/usr/bin/env python3
"""Validate Phase 3 development dataset selections against feature metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pri2m.data.metadata import (
    load_feature_metadata,
    phase3_dev_expectations,
    summarize_datasets,
    verify_dataset_expectations,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-features",
        default=str(ROOT / "data" / "metadata" / "features_all_datasets.csv"),
    )
    parser.add_argument(
        "--noisy-features",
        default=str(ROOT / "data" / "metadata" / "features_noisy_datasets.csv"),
    )
    args = parser.parse_args()

    clean_ids = [1, 4, 29, 35, 39]
    noisy_ids = [91]
    clean = load_feature_metadata(args.clean_features, datasets=clean_ids)
    noisy = load_feature_metadata(args.noisy_features, datasets=noisy_ids)
    summary = summarize_datasets(clean)
    noisy_summary = summarize_datasets(noisy)
    combined = __import__("pandas").concat([summary, noisy_summary], ignore_index=True)
    failures = verify_dataset_expectations(combined, phase3_dev_expectations())

    print("Phase 3 dev dataset summary:")
    print(combined.to_string(index=False))
    if failures:
        print("\nExpectation failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nAll Phase 3 dev dataset expectations passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
