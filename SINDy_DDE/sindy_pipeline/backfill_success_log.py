"""
backfill_success_log.py

One-off helper. Reads every state_{dataset}_{file_y}.pkl already in
preprocessed/ and writes a preprocessed_log.csv summarizing them.

Use this when pickles were generated before the success-log feature
was added, so you do not have to re-run the full preprocessing pipeline.

Usage:
    python backfill_success_log.py
"""

import glob
import os
import pickle
import re

import numpy as np
import pandas as pd

PREPROCESSED_DIR = "/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline/preprocessed"
SUCCESS_LOG_PATH = "/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline/preprocessed_log.csv"


def row_from_pickle(pkl_path):
    """Read one pickle and return a dict matching the success log schema."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    X = np.asarray(d.get("X"))
    b_over_blim = d.get("b_over_blim", float("nan"))
    label = int(b_over_blim >= 1.0) if np.isfinite(b_over_blim) else -1

    return {
        "dataset": d.get("dataset"),
        "file_y": d.get("file_y"),
        "label": label,
        "omega": d.get("omega"),
        "b": d.get("b"),
        "blim": d.get("blim"),
        "b_over_blim": b_over_blim,
        "teeth": d.get("teeth"),
        "tau_sec": d.get("tau_sec"),
        "delay_samples": d.get("delay_samples"),
        "dt": d.get("dt"),
        "raw_length": d.get("raw_length"),
        "trimmed_length": d.get("trimmed_length"),
        "trim_method": d.get("trim_method"),
        "trim_start_index": d.get("trim_start_index"),
        "X_rows": int(X.shape[0]) if X.ndim >= 1 else None,
        "X_cols": int(X.shape[1]) if X.ndim >= 2 else None,
        "x_delay_corr": d.get("x_delay_corr"),
        "include_y": d.get("include_y"),
        "pickle_path": pkl_path,
    }


def main():
    pattern = os.path.join(PREPROCESSED_DIR, "state_*_*.pkl")
    paths = sorted(glob.glob(pattern))
    print(f"Found {len(paths)} pickles in {PREPROCESSED_DIR}")

    rows = []
    failed = []
    for i, p in enumerate(paths, 1):
        try:
            rows.append(row_from_pickle(p))
        except Exception as e:
            failed.append((p, type(e).__name__, str(e)))
        if i % 1000 == 0:
            print(f"  processed {i}/{len(paths)}")

    df = pd.DataFrame(rows)

    # Sort for readability
    if "dataset" in df.columns and "file_y" in df.columns:
        df = df.sort_values(["dataset", "file_y"]).reset_index(drop=True)

    df.to_csv(SUCCESS_LOG_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {SUCCESS_LOG_PATH}")

    if failed:
        print(f"\n{len(failed)} pickles failed to read:")
        for p, etype, msg in failed[:10]:
            print(f"  {p}: {etype}: {msg}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()
