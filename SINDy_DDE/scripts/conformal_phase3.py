#!/usr/bin/env python3
"""Split-conformal calibration runner for trained Phase 3 TKAN checkpoint(s).

Eval-only (no retraining). Loads one or more trained ``TKANModel`` checkpoints,
scores a balanced batch of real windows to get chatter probabilities (averaging
the checkpoints into a cross-fold ensemble score when more than one is given),
splits those windows into a calibration half and a test half, and runs
split-conformal calibration (:mod:`pri2m.xai.conformal`) to produce:

* a coverage/abstention sweep over an ``alpha`` grid -- the empirical coverage on
  the fresh test split lands at the target ``1 - alpha`` (the conformal
  guarantee), and the abstention rate rises as ``alpha`` shrinks;
* a headline calibrated predictor at ``--alpha`` with its selective accuracy on
  the committed (singleton) windows -- the fix for the raw 0.5-threshold accuracy
  swing seen in the 121-125 held-out eval;
* a ``conformal_manifest.json`` (provenance + headline + full sweep + artifact
  paths) and a coverage/abstention figure written both under the run's output dir
  and into ``results/figures/research_update_assets/conformal_<tag>.png``.

Example (single fold)::

    python scripts/conformal_phase3.py \
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \
        --npz-root $PWD/phase2_signals --table results/phase3/phase3_dataset.csv \
        --output-dir results/phase3/xai/conformal --tag fold2 --n-windows 512

Cross-fold ensemble: pass ``--checkpoint`` once per fold (all five). The
per-checkpoint probabilities are averaged on a shared normalization (the primary
checkpoint's ``preprocessing.json``) before calibration.
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

import numpy as np  # noqa: E402

import xai_phase3  # noqa: E402  (battle-tested checkpoint loaders)
from pri2m.data.phase3_sequence import WindowConfig  # noqa: E402
from pri2m.xai import case_studies, conformal  # noqa: E402
from whitebox_phase3 import build_labeled_batch  # noqa: E402

CHANNELS = ("xtpos", "ytpos", "Forcex", "Forcey")


def _now_date() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _git_sha(root) -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _chatter_probs(model, batch) -> np.ndarray:
    """Chatter probability ``sigmoid(chatter_logit)`` per window ``[N]``."""
    return np.asarray(case_studies.predict(model, batch)["chatter_prob"], dtype=float).reshape(-1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="Trained checkpoint (repeat once per fold for a cross-fold ensemble).",
    )
    parser.add_argument("--npz-root", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--preprocessing", default=None,
                        help="Shared preprocessing.json (default: auto-discover next to the FIRST checkpoint).")
    parser.add_argument("--output-dir", default="results/phase3/xai/conformal")
    parser.add_argument("--research-assets-dir", default="results/figures/research_update_assets")
    parser.add_argument("--n-windows", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for the random window sample.")
    parser.add_argument("--split-seed", type=int, default=0, help="Seed for the calibration/test split.")
    parser.add_argument("--cal-frac", type=float, default=0.5, help="Fraction of windows used for calibration.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Headline target miscoverage (coverage 1-alpha).")
    parser.add_argument("--alphas", default="0.2,0.1,0.05,0.02,0.01",
                        help="Comma-separated alpha grid for the coverage/abstention sweep.")
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=512)
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--tag", default=None, help="Human label (default: output dir name).")
    # xai_phase3._load_model reads these off the args namespace.
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--spline-order", type=int, default=3)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or out_dir.name
    alphas = [float(a) for a in str(args.alphas).split(",") if a.strip()]

    # ---- Preprocessing + models (shared norm from the primary checkpoint) --- #
    pre_path = args.preprocessing or xai_phase3._find_preprocessing(args.checkpoint[0])
    preprocessing = json.loads(Path(pre_path).read_text(encoding="utf-8")) if pre_path else None

    models = []
    for ckpt in args.checkpoint:
        m = xai_phase3._load_model(ckpt, args, preprocessing)
        m.pool = int(args.pool)  # forward-time op; not in the state_dict
        m.eval()
        models.append(m)

    # ---- One shared labeled batch of real windows -------------------------- #
    cfg = WindowConfig(length=args.window_length, stride=args.window_stride, channels=CHANNELS)
    batch, labels = build_labeled_batch(
        args.table, args.npz_root, preprocessing, args.n_windows, cfg, seed=args.sample_seed
    )
    labels = np.asarray(labels).reshape(-1).astype(int)
    n_windows = int(labels.shape[0])

    # ---- Chatter probs -> (cross-fold ensemble) score ---------------------- #
    fold_probs = np.stack([_chatter_probs(m, batch) for m in models], axis=0)  # [n_folds, N]
    probs = conformal.ensemble_probs(fold_probs) if len(models) > 1 else fold_probs[0]

    # ---- Calibration / test split ------------------------------------------ #
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(n_windows)
    n_cal = max(1, int(round(args.cal_frac * n_windows)))
    n_cal = min(n_cal, n_windows - 1)  # leave at least one test window
    cal_idx, test_idx = perm[:n_cal], perm[n_cal:]
    p_cal, y_cal = probs[cal_idx], labels[cal_idx]
    p_test, y_test = probs[test_idx], labels[test_idx]

    # ---- Headline predictor + coverage/abstention sweep -------------------- #
    headline_conf = conformal.SplitConformal.calibrate(p_cal, y_cal, alpha=args.alpha)
    headline = headline_conf.evaluate(p_test, y_test)
    sweep = conformal.coverage_abstention_sweep(p_cal, y_cal, p_test, y_test, alphas)

    # ---- Figure (run dir + research-update assets) ------------------------- #
    fig_run = out_dir / f"conformal_{tag}.png"
    conformal.plot_coverage_abstention(sweep, str(fig_run))
    assets_dir = Path(args.research_assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig_asset = assets_dir / f"conformal_{tag}.png"
    conformal.plot_coverage_abstention(sweep, str(fig_asset))

    # ---- Manifest ---------------------------------------------------------- #
    provenance = {
        "checkpoints": [str(c) for c in args.checkpoint],
        "n_folds": len(models),
        "ensemble": len(models) > 1,
        "preprocessing": str(pre_path) if pre_path else None,
        "tag": tag,
        "rendered_date": _now_date(),
        "git_sha": _git_sha(_ROOT),
        "n_windows": n_windows,
        "n_cal": int(n_cal),
        "n_test": int(n_windows - n_cal),
        "n_chatter": int((labels == 1).sum()),
        "n_stable": int((labels == 0).sum()),
        "alpha": float(args.alpha),
        "alphas": alphas,
        "cal_frac": float(args.cal_frac),
        "sample_seed": int(args.sample_seed),
        "split_seed": int(args.split_seed),
        "pool": int(args.pool),
    }
    manifest = {
        "provenance": provenance,
        "headline": headline,
        "sweep": sweep,
        "artifacts": {
            "coverage_abstention_fig": str(fig_run),
            "coverage_abstention_fig_asset": str(fig_asset),
        },
    }
    manifest_path = out_dir / "conformal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")

    print(f"Conformal manifest -> {manifest_path}")
    print(
        f"[conformal] tag={tag} n_windows={n_windows} (cal={n_cal}, test={n_windows - n_cal}) "
        f"alpha={args.alpha} | coverage={headline['coverage']:.3f} "
        f"(target {headline['target_coverage']:.3f}) abstain={headline['abstention_rate']:.3f} "
        f"sel_acc={headline['selective_accuracy']:.3f} raw_acc={headline['raw_accuracy']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
