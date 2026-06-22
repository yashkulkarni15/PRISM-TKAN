"""Generate the leakage-safe parent-group map for datasets 1..155.

Two datasets must never straddle a train/val split when they share operating
points. That happens whenever their documented modal/cutting parameters are
identical: noise is added to the *signal*, not the params, so every noise mirror
shares its clean parent's params; and several clean datasets are exact repeats
(e.g. clean 1==23, 16==17, and every new clean 126..155 repeats one of 1..30).

We fingerprint each dataset on all params across all modes (excluding noise std
and bookkeeping columns), union datasets with equal fingerprints, and label each
connected component by its smallest dataset id. This reproduces the old
arithmetic map (D, D+60, D+90 -> D) on every dataset that has no parameter twin,
and only ever MERGES groups the arithmetic map kept apart -- never splits one.

Run from the repo root:  python scripts/build_parent_groups.py
Writes:  data/metadata/parent_groups.json   {dataset_id: group_id} for 1..155
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
XLSX = REPO / "data/metadata/Dataset descriptions.xlsx"
OUT = REPO / "data/metadata/parent_groups.json"
EXCLUDE = {"Dataset", "Mode number", "Noise std (%)"}


def _fingerprint(group: pd.DataFrame, params: list[str]) -> tuple:
    g = group.sort_values("Mode number")
    vals = np.round(g[params].to_numpy(dtype=float), 6)
    return tuple(np.nan_to_num(vals, nan=-9.99e99).ravel().tolist())


def build_map(xlsx: Path = XLSX) -> dict[int, int]:
    """Return ``{dataset_id: group_id}`` for every dataset documented in ``xlsx``."""
    df = pd.read_excel(xlsx)
    df["Dataset"] = pd.to_numeric(df["Dataset"], errors="coerce").ffill()
    df = df[df["Dataset"].notna()].copy()
    df["Dataset"] = df["Dataset"].astype(int)
    params = [c for c in df.columns if c not in EXCLUDE]
    for c in params + ["Mode number"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Mode number"])

    fp = {d: _fingerprint(g, params) for d, g in df.groupby("Dataset")}
    by_fp: dict[tuple, list[int]] = defaultdict(list)
    for d, f in fp.items():
        by_fp[f].append(d)

    group_map: dict[int, int] = {}
    for members in by_fp.values():
        gid = min(members)
        for d in members:
            group_map[d] = gid
    return dict(sorted(group_map.items()))


if __name__ == "__main__":
    gm = build_map()
    OUT.write_text(json.dumps({str(k): v for k, v in gm.items()}, indent=2))
    n_groups = len(set(gm.values()))
    print(f"[build_parent_groups] wrote {len(gm)} datasets -> {OUT}")
    print(f"  distinct groups: {n_groups}")
    # quick sanity echo of the merges + new ids
    for d in (17, 23, 77, 121, 122, 123, 124, 125, 126, 139, 150, 155):
        print(f"    parent_group({d}) = {gm.get(d)}")
