#!/usr/bin/env python3
"""J1 eval-path check: where does a checkpoint's validation AUC go missing?

For each checkpoint, the same fold's validation set is scored three ways and
written side by side to ``<output-dir>/evalcheck_<TAG>.json``:

* ``trainer_load_trainer_eval_auc``  model rebuilt the trainer's way
  (``build_model`` + ``strip_compile_prefix``), validation loader from
  ``build_fold_loaders`` exactly as in ``train_one_fold``, ``Phase3Trainer.evaluate``.
* ``xai_load_trainer_eval_auc``      same loader + evaluate, model loaded by
  ``xai_phase3._load_model`` (what the XAI scripts use).
* ``score_files_auc``                ``score_files_phase3.main --split val``.

``max_abs_logit_diff`` compares the two loaded models on the first validation
batch; ``recorded_val_auc`` is ``summary.json`` ``best_auc_val.auc``;
``state_dict_prefixes`` lists the distinct first key segments of the checkpoint.
The script always runs on CPU without torch.compile, so any residual gap
against the trainer's CUDA/compile-time number cannot be attributed by it.
Nothing here modifies the trainer or any scorer. ``"complete": true`` is last.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pri2m.data.phase3_sequence import WindowConfig, build_fold_loaders  # noqa: E402
from pri2m.training.losses import LossWeights  # noqa: E402
from pri2m.training.phase3_trainer import (  # noqa: E402
    Phase3Trainer,
    TrainConfig,
    _batch_to_device,
    build_model,
    resolve_device,
    strip_compile_prefix,
)


def _train_config(d: dict) -> TrainConfig:
    """Rebuild the run's ``TrainConfig`` from ``summary.json["config"]``."""
    known = {f.name for f in fields(TrainConfig)}  # legacy configs lack newer keys
    d = {k: v for k, v in d.items() if k in known}
    d["window"] = WindowConfig(**{**d["window"], "channels": tuple(d["window"]["channels"])})
    d["loss"] = LossWeights(**d["loss"])
    if "severity_quantiles" in d:
        d["severity_quantiles"] = tuple(d["severity_quantiles"])
    return TrainConfig(**d)


def check_one(ckpt: str, a, table, folds, out_dir: Path, i: int) -> dict:
    from score_files_phase3 import main as score_main
    from xai_phase3 import _find_preprocessing, _load_model

    run_dir = Path(ckpt).resolve().parent.parent
    summary = json.loads((run_dir / "summary.json").read_text())
    pre = json.loads(Path(_find_preprocessing(ckpt)).read_text())
    cfg = _train_config(summary["config"])
    npz_root, device = a.npz_root, a.device
    dev = resolve_device(device)

    # train_one_fold: loaders first, then build_model.
    _, val_loader, _ = build_fold_loaders(
        table, folds, cfg.fold, npz_root, cfg.effective_window(), cfg.batch_size,
        seed=cfg.seed, num_workers=cfg.num_workers, pin_memory=False)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    m_trainer = build_model(cfg)
    m_trainer.load_state_dict(strip_compile_prefix(state))
    m_xai = _load_model(ckpt, argparse.Namespace(
        input_size=None, hidden_size=None, grid_size=cfg.grid_size,
        spline_order=cfg.spline_order), pre)
    m_trainer.to(dev).eval()
    m_xai.to(dev).eval()

    name = Path(ckpt).name
    trainer = Phase3Trainer(cfg)
    with torch.no_grad():
        batch = _batch_to_device(next(iter(val_loader)), dev)
        diff = (m_trainer(batch)["chatter_logit"] - m_xai(batch)["chatter_logit"]).abs().max().item()

    csv = out_dir / f"evalcheck_scores_{a.tag}_{i}.csv"
    score_main(["--checkpoint", ckpt, "--npz-root", npz_root, "--split", "val",
                "--table", a.table, "--folds-json", a.folds_json,
                "--device", device, "--output-csv", str(csv)])
    score_auc = json.loads(csv.with_suffix(".json").read_text())["auc"]["val"]["0.0"]["auc"]
    return {
        "checkpoint": ckpt,
        "trainer_load_trainer_eval_auc": trainer.evaluate(m_trainer, val_loader)["auc"],
        "xai_load_trainer_eval_auc": trainer.evaluate(m_xai, val_loader)["auc"],
        "score_files_auc": score_auc,
        "max_abs_logit_diff": diff,
        "state_dict_prefixes": sorted({k.split(".")[0] for k in state}),
        "checkpoint_name": name,
        "recorded_auc_matches_checkpoint": name == "best_auc.pt",
        "recorded_val_auc": summary["best_auc_val"]["auc"] if name == "best_auc.pt" else None,
        "recorded_note": None if name == "best_auc.pt" else (
            "summary best_auc_val is the best_auc.pt epoch; this checkpoint is a "
            "different epoch, so no recorded AUC corresponds to it"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--folds-json", required=True)
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(a.table)
    folds = json.loads(Path(a.folds_json).read_text())
    results = [check_one(c, a, table, folds, out_dir, i)
               for i, c in enumerate(a.checkpoint)]
    doc = {"tag": a.tag, "checkpoints": results, "complete": True}
    (out_dir / f"evalcheck_{a.tag}.json").write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
