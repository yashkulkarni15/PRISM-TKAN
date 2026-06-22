#!/usr/bin/env python3
"""Phase 2.5 pre-flight check on the clean / 1% noise / 5% noise mirror ladder.

Uses the precomputed spectral features already in
``data/metadata/features_all_datasets.csv`` and
``data/metadata/features_noisy_datasets.csv``
to confirm that the basic Phase 2.5 ladder methodology is testable on this
dataset before building any new metric code.

For each candidate metric, the script reports:

- paired mean of ``(noisy_5pct - clean)`` per (omega, b_mm) match,
- standard deviation of that paired difference,
- Cohen-style effect size mean(delta)/sd(delta),
- count of file pairs that are strictly monotone clean < 1% < 5%,
- one-sided Mann-Whitney U p-value comparing 5%% noisy > clean distributions.

The intent is to gate the metric list: any candidate that fails the monotone
ladder, or fails a strict significance threshold, is not promoted to the
Phase 2.5 noise-score vector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_CLEAN_CSV = "data/metadata/features_all_datasets.csv"
DEFAULT_NOISY_CSV = "data/metadata/features_noisy_datasets.csv"
DEFAULT_CLEAN_ID = 1
DEFAULT_NOISE1_ID = 61
DEFAULT_NOISE5_ID = 91

DEFAULT_METRICS = (
    "xtpos_spectral_entropy",
    "ytpos_spectral_entropy",
    "Forcex_spectral_entropy",
    "Forcey_spectral_entropy",
    "xtpos_harmonic_energy_ratio",
    "Forcex_harmonic_energy_ratio",
    "xtpos_sideband_energy",
    "Forcex_sideband_energy",
    "xtpos_kurt",
    "Forcex_kurt",
    "xtpos_energy_ratio",
    "Forcex_energy_ratio",
    "xtpos_fn_proximity",
    "Forcex_fn_proximity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-csv", default=DEFAULT_CLEAN_CSV)
    parser.add_argument("--noisy-csv", default=DEFAULT_NOISY_CSV)
    parser.add_argument("--clean-id", type=int, default=DEFAULT_CLEAN_ID)
    parser.add_argument("--noise1-id", type=int, default=DEFAULT_NOISE1_ID)
    parser.add_argument("--noise5-id", type=int, default=DEFAULT_NOISE5_ID)
    parser.add_argument(
        "--output-csv",
        default="results/phase2_5/csv_ladder_checks/csv_ladder_check_1_61_91.csv",
        help="Output CSV with the per-metric ladder statistics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    clean = pd.read_csv(args.clean_csv)
    noisy = pd.read_csv(args.noisy_csv)

    d1 = clean[clean["dataset"] == args.clean_id].set_index(["omega", "b_mm"]).sort_index()
    d61 = noisy[noisy["dataset"] == args.noise1_id].set_index(["omega", "b_mm"]).sort_index()
    d91 = noisy[noisy["dataset"] == args.noise5_id].set_index(["omega", "b_mm"]).sort_index()

    common = d1.index.intersection(d61.index).intersection(d91.index)
    n = len(common)
    if n == 0:
        raise SystemExit("No matched (omega, b_mm) pairs across the three datasets.")

    rows: list[dict[str, object]] = []
    for col in DEFAULT_METRICS:
        if col not in d1.columns:
            continue
        v1 = d1.loc[common, col].to_numpy()
        v61 = d61.loc[common, col].to_numpy()
        v91 = d91.loc[common, col].to_numpy()

        delta_91 = v91 - v1
        sd = float(delta_91.std(ddof=1))
        mean_delta = float(delta_91.mean())
        effect = mean_delta / sd if sd > 0 else float("nan")

        n_up = int(np.sum((v1 < v61) & (v61 < v91)))
        n_dn = int(np.sum((v1 > v61) & (v61 > v91)))
        n_monotone = n_up

        try:
            _, p_mwu = stats.mannwhitneyu(v91, v1, alternative="greater")
        except ValueError:
            p_mwu = float("nan")

        rows.append(
            {
                "metric": col,
                "n_paired": n,
                "mean_delta_5pct_minus_clean": mean_delta,
                "sd_paired_delta": sd,
                "paired_effect_size": effect,
                "n_strict_monotone_up_ladder": n_up,
                "n_strict_monotone_down_ladder": n_dn,
                "mwu_p_value_5pct_vs_clean": p_mwu,
                "promoted": (
                    mean_delta > 0.0
                    and n_monotone >= int(0.95 * n)
                    and np.isfinite(p_mwu)
                    and p_mwu < 1e-3
                ),
            }
        )

    out = pd.DataFrame(rows).sort_values("paired_effect_size", key=lambda s: s.abs(), ascending=False)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print(f"Pre-flight ladder check on datasets {args.clean_id} / {args.noise1_id} / {args.noise5_id}")
    print(f"Paired samples: {n}")
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
