#!/usr/bin/env python3
"""Measure the noise that turns clean dataset D into its 1 % / 5 % mirrors.

Clean datasets 1-30 have mirrors D+60 (1 %) and D+90 (5 %) that are the same
simulated cuts with noise added, so ``mirror - clean`` for the matched file
``time{D}_{k}`` <-> ``time{M}_{k}`` IS the added noise. Per pair this records
:func:`pri2m.xai.noise_probe.mirror_noise_stats` (relative std per channel,
whiteness, correlation with the clean signal), then
:func:`~pri2m.xai.noise_probe.summarise_calibration` turns the pairs into the
``rel_sigma_per_unit`` that ``inject_noise`` needs -- or refuses, with a
reason, if the difference is not additive white noise.

Pairing is checked, not assumed: the paired NPZ metas must agree on omega and
label; disagreements are counted (``n_meta_disagree``) and flagged per pair.
Missing mirror files are skipped and counted (``n_missing``).

The JSON is written once, at the end, with ``"complete": true``.

Example::

    python scripts/calibrate_mirror_noise.py --npz-root phase2_signals \\
        --output results/phase3/xai/noise_compare/mirror_noise_calibration.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from pri2m.data.phase3_sequence import WindowConfig, load_npz_signals  # noqa: E402
from pri2m.xai.noise_probe import (  # noqa: E402
    mirror_file,
    mirror_noise_stats,
    summarise_calibration,
)

DEFAULT_OUTPUT = "results/phase3/xai/noise_compare/mirror_noise_calibration.json"
CHANNELS = tuple(WindowConfig().channels)
_MIRRORS = ((60, 1), (90, 5))  # (dataset offset, tier %)
_OMEGA_RTOL = 1e-9


def parse_datasets(spec: str) -> list[int]:
    """``"1-30"`` / ``"1,2,5-7"`` -> sorted unique ints."""
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _clean_files(npz_root: Path, dataset: int) -> list[Path]:
    """``time{D}_{k}.npz`` files of a clean dataset, sorted by ``k``."""
    rx = re.compile(rf"^time{int(dataset)}_(\d+)\.npz$")
    found = []
    for p in (npz_root / f"dataset_{int(dataset)}").glob(f"time{int(dataset)}_*.npz"):
        m = rx.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def run(npz_root: str, datasets, max_files_per_dataset: int | None, output: str) -> dict:
    t0 = time.time()
    root = Path(npz_root)
    datasets = sorted(int(d) for d in datasets)
    bad = [d for d in datasets if not 1 <= d <= 30]
    if bad:
        raise ValueError(f"only clean datasets 1-30 have mirrors; got {bad}")

    pairs: list[dict] = []
    n_missing = 0
    n_meta_disagree = 0
    n_clean_files = 0
    for d in datasets:
        files = _clean_files(root, d)
        if max_files_per_dataset is not None:
            files = files[: int(max_files_per_dataset)]
        for cpath in files:
            n_clean_files += 1
            clean, cmeta = load_npz_signals(cpath, CHANNELS)
            for off, tier in _MIRRORS:
                m_id = d + off
                mname = mirror_file(cpath.name, d, m_id)
                mpath = root / f"dataset_{m_id}" / mname
                if not mpath.is_file():
                    n_missing += 1
                    continue
                mirror, mmeta = load_npz_signals(mpath, CHANNELS)
                omega_ok = bool(np.isclose(cmeta["omega"], mmeta["omega"],
                                           rtol=_OMEGA_RTOL, atol=0.0))
                label_ok = bool(float(cmeta["label"]) == float(mmeta["label"]))
                agree = omega_ok and label_ok
                if not agree:
                    n_meta_disagree += 1
                stats = mirror_noise_stats(clean, mirror)
                pairs.append({
                    "dataset": int(d),
                    "mirror": int(m_id),
                    "file": cpath.stem,
                    "mirror_file": mpath.stem,
                    "tier": int(tier),
                    "n_clean_samples": int(clean.shape[0]),
                    "n_mirror_samples": int(mirror.shape[0]),
                    "omega_clean": float(cmeta["omega"]),
                    "omega_mirror": float(mmeta["omega"]),
                    "label_clean": float(cmeta["label"]),
                    "label_mirror": float(mmeta["label"]),
                    "meta_agree": agree,
                    "stats": stats,
                })
        print(f"[calibrate] dataset {d}: {len(files)} clean files; "
              f"{len(pairs)} pairs so far, {n_missing} missing", flush=True)

    summary = summarise_calibration(
        [{"tier": p["tier"], **p["stats"]} for p in pairs])
    # Spread of the per-pair relative sigma (per tier, per channel). The summary
    # keeps only the median; a wide spread means the real noise is NOT a fixed
    # fraction of each file's own channel std, which is what inject_noise draws.
    spread: dict = {}
    for tier in sorted({p["tier"] for p in pairs}):
        arr = np.array([p["stats"]["rel_sigma"] for p in pairs if p["tier"] == tier],
                       dtype=np.float64)
        spread[str(tier)] = {
            q: np.nanpercentile(arr, float(q[1:]), axis=0).tolist()
            for q in ("p10", "p50", "p90")}
    result = {
        "analysis": "mirror - clean noise calibration (D vs D+60 / D+90)",
        "npz_root": str(npz_root),
        "datasets": datasets,
        "max_files_per_dataset": max_files_per_dataset,
        "channels": list(CHANNELS),
        "n_clean_files": n_clean_files,
        "n_pairs": len(pairs),
        "n_missing": n_missing,
        "n_meta_disagree": n_meta_disagree,
        "summary": summary,
        "rel_sigma_spread": spread,
        "pairs": pairs,
        "wall_seconds": round(time.time() - t0, 1),
        "complete": True,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[calibrate] {len(pairs)} pairs, missing={n_missing}, "
          f"meta_disagree={n_meta_disagree}; rel_sigma_per_unit="
          f"{summary['rel_sigma_per_unit']} linear_ok={summary['linear_ok']} "
          f"frac_additive={summary['frac_additive']} "
          f"undecided={summary['undecided_reason']} -> {out}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--datasets", default="1-30")
    ap.add_argument("--max-files-per-dataset", type=int, default=None)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    args = ap.parse_args(argv)
    run(args.npz_root, parse_datasets(args.datasets), args.max_files_per_dataset,
        args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
