#!/usr/bin/env python3
"""Reassemble a sharded TSR run into a white-box manifest.

The sharded white-box workflow splits the expensive Temporal Saliency Rescaling
stage across a SLURM array -- each task runs
``whitebox_phase3.py --tsr-only --tsr-shard K --tsr-nshards N`` and drops a shard
file under ``<output-dir>/tsr_shards/`` -- while every *other* analysis comes from
one ``whitebox_phase3.py --skip-tsr`` run that writes ``whitebox_manifest.json``
(with ``results["tsr"] = null``).

This script reads those shard files, reassembles the per-window entries in their
canonical (un-sharded) order, rebuilds the ``results["tsr"]`` block via
``whitebox_phase3.tsr_aggregate`` -- reproducing exactly what a single un-sharded
run would have written -- patches it into the manifest, refreshes the render-ready
``report`` block, and rewrites the manifest in place.

Example::

    python scripts/merge_tsr_shards.py --output-dir results/phase3/xai/fold2_w512
    python scripts/render_whitebox_report.py \
        --manifest results/phase3/xai/fold2_w512/whitebox_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import whitebox_phase3 as W  # noqa: E402


def merge_shards(manifest_path: Path, shards_dir: Path) -> dict:
    """Return the manifest with a reassembled ``results['tsr']`` and fresh report.

    Raises if the shard set is incomplete or internally inconsistent, so a partial
    array (a task still running or failed) can never be merged into a manifest that
    silently under-reports the TSR sample size.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_files = sorted(shards_dir.glob("shard_*_of_*.json"))
    if not shard_files:
        raise FileNotFoundError(f"no TSR shard files under {shards_dir}")

    by_shard: dict[int, dict] = {}
    ref: dict | None = None
    for path in shard_files:
        pl = json.loads(path.read_text(encoding="utf-8"))
        key = {k: pl[k] for k in ("nshards", "steps", "tsr_windows", "n_selected")}
        if ref is None:
            ref = key
        elif key != ref:
            raise ValueError(
                f"shard {path.name} disagrees on run parameters: {key} != {ref}"
            )
        k = int(pl["shard"])
        if k in by_shard:
            raise ValueError(f"duplicate shard index {k} ({path.name})")
        by_shard[k] = pl

    nshards = int(ref["nshards"])
    missing = sorted(set(range(nshards)) - set(by_shard))
    if missing:
        raise ValueError(
            f"incomplete TSR shard set: have {sorted(by_shard)}, missing {missing} "
            f"of {nshards} -- wait for the array to finish (or resubmit failed tasks)."
        )

    indexed: list[list] = []
    for k in range(nshards):
        indexed.extend(by_shard[k]["per_window_indexed"])
    indexed.sort(key=lambda pe: pe[0])  # restore the canonical global window order
    per_window = [entry for _pos, entry in indexed]

    n_selected = ref.get("n_selected")
    if n_selected is not None and len(per_window) != int(n_selected):
        raise ValueError(
            f"reassembled {len(per_window)} TSR window(s) but the shards report "
            f"{n_selected} selected -- shard files are inconsistent."
        )

    manifest.setdefault("results", {})["tsr"] = W.tsr_aggregate(
        per_window, int(ref["steps"])
    )

    # Rebuild the render-ready report from the patched results, preserving the
    # index-map table the runner attaches to the report outside _build_report.
    idx_table = manifest.get("report", {}).get("index_map_table")
    report = W._build_report(
        manifest["provenance"], manifest["results"], manifest["artifacts"]
    )
    if idx_table is not None:
        report["index_map_table"] = idx_table
    manifest["report"] = report
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True,
        help="Fold xai dir holding whitebox_manifest.json + tsr_shards/.",
    )
    parser.add_argument("--manifest", default=None, help="Override manifest path.")
    parser.add_argument("--shards-dir", default=None, help="Override tsr_shards dir.")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else out / "whitebox_manifest.json"
    shards_dir = Path(args.shards_dir) if args.shards_dir else out / "tsr_shards"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest not found: {manifest_path} -- run the --skip-tsr fixed stage first."
        )

    manifest = merge_shards(manifest_path, shards_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    tsr = manifest["results"]["tsr"]
    print(f"Merged {tsr['n_windows']} TSR window(s) -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
