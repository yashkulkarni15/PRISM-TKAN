#!/usr/bin/env python3
"""J2: one fixed sample of windows per fold, shared by every network (8 TKAN, 8 MiniROCKET).

Writes ``equations/windows_fold<F>.npz`` -- ``val_*`` (fold-validation) and ``ho_*``
(held-out 121-125) windows with their side scalars, ids and the 16 formula terms,
plus ``range_lo``/``range_hi`` (per-channel 1st/99th percentile of the pooled
windows of ``--n-train-range`` TRAIN-fold windows) -- and a sidecar
``windows_fold<F>.json`` (sha256, provenance, ``"complete": true``) written last.

Windows are z-scored with the ``--preprocessing`` norm (the fold's seed-0 run;
norms are identical across seeds), so every network sees identical inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eval_heldout_phase3 import build_eval_table  # noqa: E402
from pri2m.models.minirocket_baseline import pool_windows  # noqa: E402
from pri2m.xai.equation_terms import CHANNELS, TERM_NAMES, window_terms  # noqa: E402
from pri2m.xai.equations import channel_ranges  # noqa: E402
from pri2m.xai.sampling import build_labeled_batch, window_config_from_preprocessing  # noqa: E402
from score_files_phase3 import split_fold  # noqa: E402

POOL = 4  # TKAN's avg-pool factor; window_terms and the ranges both use it


def _draw(table: pd.DataFrame, npz_root, pre, cfg, n: int, seed: int):
    """Class-balanced seeded batch from ``table`` (via a temp CSV, as the sampler wants)."""
    with tempfile.TemporaryDirectory() as d:
        csv = Path(d) / "table.csv"
        table.to_csv(csv, index=False)
        batch, _ = build_labeled_batch(csv, npz_root, pre, n, cfg, supervised_only=True, seed=seed)
    return batch


def _arrays(prefix: str, batch: dict) -> dict:
    w = batch["window"].numpy()
    f = {k: batch[k].numpy() for k in ("tau", "tpf", "fs", "omega", "label")}
    out = {f"{prefix}_window": w, **{f"{prefix}_{k}": v for k, v in f.items()},
           f"{prefix}_terms": window_terms(w, f["tau"], f["tpf"], f["fs"], POOL)}
    for k in ("file", "dataset", "parent_group"):  # unicode, so np.load needs no pickle
        out[f"{prefix}_{k}"] = np.array([str(v) for v in batch[k]])
    return out


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preprocessing", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--folds-json", required=True)
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-val", type=int, default=2048)
    ap.add_argument("--n-heldout", type=int, default=1024)
    ap.add_argument("--n-train-range", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args(argv)

    pre = json.loads(Path(a.preprocessing).read_text())
    cfg = window_config_from_preprocessing(pre)
    train_df, val_df = split_fold(pd.read_csv(a.table),
                                  json.loads(Path(a.folds_json).read_text()), a.fold)
    held_df, _ = build_eval_table(a.features_csv, a.npz_root)

    val = _draw(val_df, a.npz_root, pre, cfg, a.n_val, a.seed)
    ho = _draw(held_df, a.npz_root, pre, cfg, a.n_heldout, a.seed)
    tr = _draw(train_df, a.npz_root, pre, cfg, a.n_train_range, a.seed)
    rng = channel_ranges(pool_windows(tr["window"].numpy(), POOL))

    arrays = {**_arrays("val", val), **_arrays("ho", ho), "term_names": np.array(TERM_NAMES),
              "range_lo": np.array([rng[c][0] for c in CHANNELS]),
              "range_hi": np.array([rng[c][1] for c in CHANNELS])}
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / f"windows_fold{a.fold}.npz"
    np.savez_compressed(npz, **arrays)
    (out / f"windows_fold{a.fold}.json").write_text(json.dumps({
        "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "preprocessing": str(a.preprocessing), "fold": a.fold, "seed": a.seed,
        "n_val": len(arrays["val_label"]), "n_heldout": len(arrays["ho_label"]),
        "n_train_range": len(tr["label"]), "git_sha": _git_sha(), "complete": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
