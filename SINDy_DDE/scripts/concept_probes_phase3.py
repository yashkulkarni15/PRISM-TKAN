#!/usr/bin/env python3
"""Concept-probe (TCAV-style) runner for a trained Phase 3 TKAN checkpoint.

Eval-only (no retraining): loads a trained ``TKANModel``, samples a class-balanced
set of real windows, extracts the 64-d encoder hidden state ``h``, and fits a
linear probe on ``h`` for each named physics concept:

* ``regeneration_phase_sin`` / ``regeneration_phase_cos``
                              -- sin/cos of the (circular) phase lag between
                                 ``x(t)`` and ``x(t - tau)`` (the wrapped angle
                                 is degenerate for a linear probe).
* ``chatter_tpf_separation``  -- log1p |dominant window freq - tooth-pass freq| (Hz).
* ``proximity_to_blim``       -- the severity label ``b_over_blim = b/blim``.

For each concept it reports the held-out probe score (R^2 / accuracy), the unit
concept direction in ``h``-space (the probe weights), and a TCAV directional
sensitivity (dot of ``d chatter_logit / d h`` with the concept direction). It
writes ``concept_probes_manifest.json`` and a bar figure (probe score per
concept) to the output dir, plus a copy of the figure under
``results/figures/research_update_assets/``.

Example::

    python scripts/concept_probes_phase3.py \
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \
        --npz-root $PWD/phase2_signals --table results/phase3/phase3_dataset.csv \
        --output-dir results/phase3/xai/concept_probes --tag fold2 --n-windows 256
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import xai_phase3  # noqa: E402  (reuse the checkpoint loaders)
from pri2m.data.phase3_sequence import WindowConfig  # noqa: E402
from pri2m.xai import concept_probes as cp  # noqa: E402
from whitebox_phase3 import build_labeled_batch  # noqa: E402  (reuse the batch builder)

CHANNELS = ("xtpos", "ytpos", "Forcex", "Forcey")


def _git_sha(root) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _probe_summary(rec: dict) -> dict:
    """Flatten one concept's probe + TCAV record into JSON-safe numbers."""
    probe, tcav = rec["probe"], rec["tcav"]
    return {
        "kind": rec["kind"],
        "transform": rec.get("transform"),
        "probe_score": float(probe["score"]),
        "probe_train_score": float(probe["train_score"]),
        "chance_score": float(probe["chance_score"]),
        "n_used": int(probe["n_used"]),
        "n_train": int(probe["n_train"]),
        "n_test": int(probe["n_test"]),
        "tcav_mean_sensitivity": float(tcav["mean_sensitivity"]),
        "tcav_sensitivity_std": float(tcav["sensitivity_std"]),
        "tcav_fraction": float(tcav["tcav_fraction"]),
        "direction": [float(x) for x in np.asarray(probe["direction"]).reshape(-1)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz-root", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--preprocessing", default=None)
    parser.add_argument("--output-dir", default="results/phase3/xai/concept_probes")
    parser.add_argument("--assets-dir", default="results/figures/research_update_assets")
    parser.add_argument("--n-windows", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=512)
    parser.add_argument("--pool", type=int, default=4, help="Temporal pool factor (match training).")
    # xai_phase3._load_model reads these off the args namespace.
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--val-auc", type=float, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model + preprocessing (reuse xai_phase3 loaders) -------------- #
    pre_path = args.preprocessing or xai_phase3._find_preprocessing(args.checkpoint)
    preprocessing = json.loads(Path(pre_path).read_text(encoding="utf-8")) if pre_path else None
    model = xai_phase3._load_model(args.checkpoint, args, preprocessing)
    model.pool = int(args.pool)  # forward-time op; not in the state_dict
    model.eval()

    cfg = WindowConfig(length=args.window_length, stride=args.window_stride, channels=CHANNELS)
    batch, labels = build_labeled_batch(
        args.table, args.npz_root, preprocessing, args.n_windows, cfg, seed=args.sample_seed
    )
    n_windows = int(batch["window"].shape[0])

    # ---- Fit the concept probes -------------------------------------------- #
    results = cp.run_concept_probes(
        model, batch, test_size=args.test_size, seed=args.probe_seed, alpha=args.ridge_alpha
    )

    tag = args.tag or out_dir.name
    fig_name = f"concept_probes_{tag}.png"
    fig_path = out_dir / fig_name
    cp.plot_probe_scores(results, str(fig_path))
    # A second copy for the research-update asset bundle.
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    asset_fig = assets_dir / fig_name
    cp.plot_probe_scores(results, str(asset_fig))

    run_id = args.run_id or Path(args.checkpoint).resolve().parent.parent.name
    provenance = {
        "checkpoint": str(args.checkpoint),
        "preprocessing": str(pre_path) if pre_path else None,
        "run_id": run_id,
        "tag": tag,
        "val_auc": args.val_auc,
        "rendered_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "git_sha": _git_sha(_ROOT),
        "n_windows": n_windows,
        "n_pos": int(np.sum(labels == 1)),
        "n_neg": int(np.sum(labels == 0)),
        "hidden_size": int(model.hidden_size),
        "n_channels": int(model.input_size),
        "num_layers": int(model.encoder.num_layers),
        "pool": int(args.pool),
        "side_input": bool(getattr(model, "side_input", False)),
        "test_size": float(args.test_size),
        "ridge_alpha": float(args.ridge_alpha),
        "probe_seed": int(args.probe_seed),
        "sample_seed": int(args.sample_seed),
    }

    concept_block = {name: _probe_summary(rec) for name, rec in results["concepts"].items()}
    manifest = {
        "provenance": provenance,
        "concepts": concept_block,
        "concept_definitions": {
            "regeneration_phase_sin": (
                "sin of the UNWRAPPED regenerative phase 2*pi*k*tau/L between x(t) "
                "and x(t-tau) at the dominant channel-0 FFT bin k (DDE phase "
                "eps=omega_c*tau). Circular-safe: the wrapped angle is degenerate "
                "for a linear probe (held-out R^2 collapsed to -228)."
            ),
            "regeneration_phase_cos": (
                "cos of the UNWRAPPED regenerative phase 2*pi*k*tau/L (companion to "
                "regeneration_phase_sin; together they encode the circular phase)."
            ),
            "chatter_tpf_separation": (
                "log1p(|f_dom - tpf|) in Hz (transform=log1p): distance of the "
                "dominant channel-0 window frequency from the metadata tooth-pass "
                "frequency, on a log1p scale (heavy-tailed in Hz)."
            ),
            "proximity_to_blim": "Severity label b_over_blim = b_mm / blim from metadata.",
        },
        "artifacts": {
            "figure": str(fig_path),
            "figure_asset": str(asset_fig),
        },
    }
    manifest_path = out_dir / "concept_probes_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")

    print(f"Concept-probe manifest -> {manifest_path}")
    for name, s in concept_block.items():
        print(
            f"  {name:24s} score={s['probe_score']:+.3f} "
            f"TCAV mean={s['tcav_mean_sensitivity']:+.3g} frac={s['tcav_fraction']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
