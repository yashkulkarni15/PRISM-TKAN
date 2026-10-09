#!/usr/bin/env python3
"""Exp 3, matched TKAN delay test: the MiniROCKET delay variants applied to TKAN.

Scores one trained TKAN checkpoint on held-out 121-125 four times:

``baseline``     the unmodified model (the number every other held-out result quotes);
``zero_delay``   encoder told tau > L, so every delayed slice is zero;
``tau_x0.5``     encoder told round(0.5 * tau) (raw samples, then pooled);
``tau_x2``       encoder told round(2 * tau).

Only the ENCODER's delay changes; the side branch keeps the raw tau (see
:mod:`pri2m.xai.delay_variants`). The tau arithmetic is shared with the
MiniROCKET arm-B variants, so "TKAN drops X, MiniROCKET drops Y" compares the
same intervention on the same evaluation set. Each score goes through
``eval_heldout_phase3.run`` -- the checkpoint's own z-score, all windows, the
trainer's file-level AUC -- so nothing here is a second evaluator.

A drop counts only above the 0.0092 floor. The manifest is written once, at the
end, with ``"complete": true``.

Example::

    python scripts/tkan_delay_heldout_phase3.py \\
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \\
        --features-csv data/metadata/features_datasets_121_125.csv \\
        --npz-root phase2_signals --tag fold2_w512
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_heldout_phase3 as EH  # noqa: E402
import xai_phase3  # noqa: E402
from pri2m.xai.delay_variants import DelayVariant, delay_drop, exceeds_floor  # noqa: E402

DEFAULT_FLOOR = 0.0092

#: name -> DelayVariant kwargs. Names match the MiniROCKET manifest's B_* keys
#: (minus the "B_" prefix) so the two tables line up row for row.
VARIANTS = {
    "zero_delay": {"mode": "zero"},
    "tau_x0.5": {"mode": "scale", "scale": 0.5},
    "tau_x2": {"mode": "scale", "scale": 2.0},
}


def _score(model, args, pre_path, name, out_dir):
    """``(auc|None, n_files|None, error|None)`` for one model via EH.run."""
    try:
        res = EH.run(
            checkpoint=str(args.checkpoint), features_csv=args.features_csv,
            npz_root=args.npz_root, out=str(out_dir / f"_eval_{args.tag}_{name}.json"),
            preprocessing=str(pre_path), device=args.device,
            batch_size=args.batch_size, model=model,
        )
    except Exception as exc:  # noqa: BLE001 - a dead arm must never read as "no drop"
        return None, None, f"{type(exc).__name__}: {exc}"
    m = (res or {}).get("metrics") or {}
    auc = m.get("auc")
    if auc is None or not math.isfinite(float(auc)):
        return None, m.get("n_files"), "evaluator returned no finite metrics.auc"
    return float(auc), int(m.get("n_files", 0)), None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--preprocessing", default=None,
                    help="preprocessing.json (default: discovered next to the checkpoint).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    ap.add_argument("--output-dir", default="results/phase3/xai/tkan_delay_heldout")
    ap.add_argument("--tag", default="fold")
    args = ap.parse_args(argv)

    t_start = time.time()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}", file=sys.stderr)
        return 2
    pre_path = args.preprocessing or xai_phase3._find_preprocessing(str(ckpt))
    if pre_path is None:
        print(f"No preprocessing.json for {ckpt}", file=sys.stderr)
        return 2
    pre = json.loads(Path(pre_path).read_text())
    load_args = argparse.Namespace(input_size=None, hidden_size=None,
                                   grid_size=5, spline_order=3)
    model = xai_phase3._load_model(str(ckpt), load_args, pre)
    model.eval()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    baseline, n_files, err = _score(model, args, pre_path, "baseline", out_dir)
    walls = {"baseline": round(time.time() - t0, 1)}
    if baseline is None:
        print(f"Baseline evaluation failed: {err}", file=sys.stderr)
        return 1
    print(f"[tkan_delay] baseline held-out AUC = {baseline:.4f} ({n_files} files)",
          flush=True)

    variants = {}
    for name, kw in VARIANTS.items():
        wrapper = DelayVariant(model, **kw)
        t0 = time.time()
        auc, nf, verr = _score(wrapper, args, pre_path, name, out_dir)
        walls[name] = round(time.time() - t0, 1)
        d = delay_drop(baseline, auc)
        variants[name] = {**wrapper.describe(), "auc": auc, "n_files": nf,
                          "drop": d, "exceeds_floor": exceeds_floor(d, args.floor),
                          "error": verr}
        print(f"[tkan_delay] {name}: auc={auc} drop={d} "
              f"exceeds_floor={variants[name]['exceeds_floor']}", flush=True)

    manifest = {
        "analysis": "Exp 3: matched TKAN delay test on held-out 121-125 (encoder tau only)",
        "provenance": {
            "checkpoint": str(ckpt), "preprocessing": str(pre_path),
            "features_csv": args.features_csv, "npz_root": args.npz_root,
            "device": args.device, "batch_size": int(args.batch_size),
            "pool": int(getattr(model, "pool", 1)),
            "side_input": bool(getattr(model, "side_input", False)),
            "n_eval_files": n_files,
            "wall_seconds": {**walls, "total": round(time.time() - t_start, 1)},
        },
        "baseline_auc": baseline,
        "variants": variants,
        "floor": float(args.floor),
        "note": ("drop = baseline_auc - variant_auc; counts only when exceeds_floor. "
                 "Side inputs keep the raw tau in every variant."),
        "complete": True,
    }
    dest = out_dir / f"tkan_delay_heldout_{args.tag}.json"
    dest.write_text(json.dumps(manifest, indent=2))
    print(f"[tkan_delay] wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
