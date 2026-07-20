#!/usr/bin/env python3
"""Overlay AUC and delay-reliance degradation curves across fuzz manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if len(args.manifests) != len(args.labels):
        parser.error("--labels must contain exactly one label per manifest")

    manifests = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.manifests]
    common = set(manifests[0].get("fuzzers", {}))
    for manifest in manifests[1:]:
        common &= set(manifest.get("fuzzers", {}))
    names = [name for name in manifests[0].get("fuzzers", {}) if name in common]
    if not names:
        raise ValueError("the manifests have no fuzzer result in common")

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    })
    ncols = min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    flat = axes.ravel()
    colors = plt.get_cmap("tab10").colors

    for ax in flat[len(names):]:
        ax.axis("off")

    for ax, name in zip(flat, names):
        twin = ax.twinx()
        first_levels = manifests[0]["fuzzers"][name]["levels"]
        xs = np.arange(len(first_levels))
        for idx, (manifest, label) in enumerate(zip(manifests, args.labels)):
            result = manifest["fuzzers"][name]
            if len(result["results"]) != len(first_levels):
                raise ValueError(f"{name}: manifest level counts differ")
            auc = [row["mean"]["auc"] for row in result["results"]]
            delay_drop = [row["mean"]["delay_drop"] for row in result["results"]]
            color = colors[idx % len(colors)]
            ax.plot(xs, auc, "o-", color=color, lw=1.5, ms=3, label=label)
            twin.plot(xs, delay_drop, "s--", color=color, lw=1.1, ms=2.5, alpha=0.85)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(xs, [str(level) for level in first_levels])
        ax.set_xlabel(f"{name} level")
        ax.set_ylabel("per-window AUC")
        twin.set_ylabel("delay-ablation AUC drop")
        ax.set_title(name)
        ax.legend(loc="best", frameon=True)

    fig.suptitle("Phase 3 physics understanding under graded fuzziness", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
