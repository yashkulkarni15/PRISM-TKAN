#!/usr/bin/env python3
"""Dump per-FILE chatter probabilities from a trained checkpoint (eval-only).

For every file in ``--table`` this rebuilds the exact windowing + z-score used
at training time (via :func:`scripts.eval_heldout_phase3.build_eval_table` /
``_window_config_from_preprocessing`` and the checkpoint's own
``preprocessing.json``), runs the frozen ``TKANModel`` (reconstructed via
:func:`scripts.xai_phase3._load_model`, the same load path ``eval_heldout_phase3.py``
uses) over every window, and averages ``sigmoid(chatter_logit)`` per file. No
training, no leakage logic -- pure inference, mirroring ``eval_heldout_phase3``'s
load path.

Output CSV (columns: ``file,prob,n_windows,dataset,label``) is the
``--predictions`` input for ``counterfactuals_phase3.py``, replacing its
table-label surrogate with the network's own decisions: that script inner-merges
on the table's ``file`` column and auto-picks the first present of
``prob``/``chatter_prob``/``severity``/``p_chatter``.

Example::

    python scripts/predict_windows_phase3.py \
        --checkpoint results/phase3/runs/<run_id>/checkpoints/best_auc.pt \
        --table results/phase3/phase3_dataset.csv \
        --npz-root results/phase3/npz \
        --out results/phase3/xai/counterfactuals/model_predictions_fold2.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
# src/ enables ``import pri2m...``; repo root lets us reuse ``scripts.xai_phase3``
# / ``scripts.eval_heldout_phase3`` -- exact mirror of eval_heldout_phase3.py's
# own sys.path setup.
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402  (after sys.path setup)
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pri2m.data.phase3_sequence import (  # noqa: E402
    NormStats,
    Phase3SequenceDataset,
    collate_windows,
)
from pri2m.training.phase3_trainer import resolve_device  # noqa: E402
from scripts.eval_heldout_phase3 import (  # noqa: E402
    _window_config_from_preprocessing,
    build_eval_table,
)
from scripts.xai_phase3 import _find_preprocessing, _load_model  # noqa: E402

__all__ = ["main"]


def _parse_dataset_filter(spec: str) -> set[int] | None:
    """``'all'`` -> ``None`` (no filter); else a set of int dataset ids."""
    if spec is None or str(spec).strip().lower() == "all":
        return None
    return {int(x) for x in str(spec).split(",") if x.strip()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Trained TKANModel state_dict (.pt).")
    ap.add_argument("--table", required=True, help="Per-file table CSV (needs dataset, file columns).")
    ap.add_argument("--npz-root", required=True, help="Root of the dataset_<id>/*.npz signal bundles.")
    ap.add_argument("--out", required=True, help="Output CSV path (file,prob,n_windows,dataset,label).")
    ap.add_argument(
        "--preprocessing", default=None,
        help="Path to the checkpoint's preprocessing.json; auto-discovered next to "
        "the checkpoint if omitted.",
    )
    ap.add_argument("--dataset", default="all", help="'all', an int, or comma list of dataset ids to score.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu", help="Compute device: cpu|cuda|cuda:0|auto (default cpu).")
    ap.add_argument("--input-size", type=int, default=None, help="Override inferred input channel count.")
    ap.add_argument("--hidden-size", type=int, default=None, help="Override inferred hidden width.")
    ap.add_argument("--grid-size", type=int, default=5, help="KAN grid size (must match training).")
    ap.add_argument("--spline-order", type=int, default=3, help="KAN spline order (must match training).")
    args = ap.parse_args(argv)

    pre_path = args.preprocessing or _find_preprocessing(args.checkpoint)
    if not pre_path:
        raise SystemExit("no preprocessing.json found; pass --preprocessing")
    pre = json.loads(Path(pre_path).read_text(encoding="utf-8"))
    if "norm" not in pre:
        raise SystemExit(f"{pre_path} has no 'norm' block; cannot recover z-score")

    dev = resolve_device(args.device)
    model = _load_model(args.checkpoint, args, pre)
    model.pool = int(pre.get("pool", 1))
    model.to(dev)
    model.eval()

    eval_table, _ = build_eval_table(args.table, args.npz_root)
    wanted = _parse_dataset_filter(args.dataset)
    if wanted is not None:
        eval_table = eval_table[eval_table["dataset"].astype(int).isin(wanted)].reset_index(drop=True)
    if eval_table.empty:
        raise SystemExit(f"no rows left to score after --dataset={args.dataset!r} filter")

    # The z-score MUST come from the checkpoint's own preprocessing.json, never
    # refit here (same discipline as eval_heldout_phase3.run).
    norm = NormStats.from_dict(pre["norm"])
    cfg = _window_config_from_preprocessing(pre)
    dataset = Phase3SequenceDataset(eval_table, args.npz_root, cfg=cfg, norm=norm)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_windows)

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    meta: dict[str, tuple[int, int]] = {}
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            probs = torch.sigmoid(model(batch)["chatter_logit"]).detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()
            # batch["dataset"] entries are already python ints (Phase3SequenceDataset
            # casts via int(meta["dataset"])); int() below is a cheap normalization
            # in case a future loader change makes them strings.
            for f, d, lab, p in zip(batch["file"], batch["dataset"], labels, probs):
                sums[f] = sums.get(f, 0.0) + float(p)
                counts[f] = counts.get(f, 0) + 1
                meta[f] = (int(d), int(lab))

    rows = [
        {
            "file": f,
            "prob": sums[f] / counts[f],
            "n_windows": counts[f],
            "dataset": meta[f][0],
            "label": meta[f][1],
        }
        for f in sums
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["file", "prob", "n_windows", "dataset", "label"]).to_csv(out, index=False)
    print(f"[predict_windows] scored {len(rows)} files ({len(dataset)} windows) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
