#!/usr/bin/env python3
"""Model-implied stability lobe diagram (SLD) runner for a Phase 3 TKAN checkpoint.

Eval-only (no retraining): loads a trained ``TKANModel``, selects one *modal
group* of files (a dataset or a set of param-twin datasets that share all modal
params), bins their real z-scored windows by (spindle speed ω, axial depth b),
reads ``mean sigmoid(chatter_logit)`` per cell → a P(chatter) field, extracts the
model-implied stability boundary per ω-column, overlays the dataset's own
analytic ``blim(ω)`` curve, and scores a scalar global-faithfulness metric
(mean |b_model − blim| + IoU + label agreement).

Writes, under ``--output-dir`` (mirrors the white-box convention):
* ``sld_<tag>.png``      — the deliverable figure (also copied to ``--assets-dir``)
* ``sld_field.csv``      — per cell: omega, b_mm, blim, label, p_chatter, n_windows
* ``sld_boundary.csv``   — per ω: omega, blim, b_model, status
* ``sld_manifest.json``  — provenance + results + artifacts + errors + report
                           (same top-level schema as whitebox_manifest.json)

Example::

    python scripts/stability_lobe_phase3.py \
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \
        --npz-root $PWD/phase2_signals --table results/phase3/phase3_dataset.csv \
        --output-dir results/phase3/xai/fold2/sld --parent-group 29 --fold 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import xai_phase3  # noqa: E402  (reuse the battle-tested checkpoint loaders)
from pri2m.data.phase3_sequence import NormStats, WindowConfig  # noqa: E402
from pri2m.xai import stability_lobe as SL  # noqa: E402

CHANNELS = ("xtpos", "ytpos", "Forcex", "Forcey")


# --------------------------------------------------------------------------- #
# Row selection
# --------------------------------------------------------------------------- #
def _npz_exists(npz_root: Path, row) -> bool:
    stem = Path(str(row["file"])).stem
    return (npz_root / f"dataset_{int(row['dataset'])}" / f"{stem}.npz").is_file()


def select_group_rows(table_csv, npz_root, dataset_ids=None, parent_group=None):
    """The modal group's usable rows: ``blim > 0`` + an existing NPZ.

    Selects by ``--parent-group`` (all param-twins share it) or an explicit
    ``--dataset-ids`` list. Rows with ``blim <= 0`` (the low-speed no-process-
    damping artifact) are dropped so the overlay/metric are defined only where
    ``blim`` is physical.
    """
    npz_root = Path(npz_root)
    df = pd.read_csv(table_csv)
    if parent_group is not None:
        df = df[df["parent_group"].astype(int) == int(parent_group)]
    elif dataset_ids:
        df = df[df["dataset"].astype(int).isin([int(d) for d in dataset_ids])]
    else:
        raise ValueError("select_group_rows needs --parent-group or --dataset-ids")
    df = df[np.isfinite(df["blim"]) & (df["blim"] > 0)]
    df = df[df.apply(lambda r: _npz_exists(npz_root, r), axis=1)].copy()
    if df.empty:
        raise RuntimeError("no usable rows: group empty after blim>0 + npz-exists filters")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Report helpers (mirror whitebox_phase3)
# --------------------------------------------------------------------------- #
def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


def _md_table(headers, rows) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([head, sep, body]) if rows else head + "\n" + sep


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


def _now_date() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _build_report(provenance, sld, artifacts, caveats) -> dict:
    """Flatten provenance + metrics + artifact paths into render-ready strings."""
    rep: dict = {}
    for k, v in provenance.items():
        if v is None:
            continue
        # The report block must be all render-ready scalars; join list fields.
        rep[k] = ", ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v
    rep["sld_mae_b"] = _fmt(sld["mae_b"])
    rep["sld_mape_b"] = _fmt(100 * sld["mape_b"]) if np.isfinite(sld["mape_b"]) else "N/A"
    rep["sld_iou"] = _fmt(sld["iou_stable"])
    rep["sld_label_agreement"] = _fmt(sld["label_agreement"])
    rep["sld_per_cell_auc"] = _fmt(sld["per_cell_auc"])
    rep["sld_n_columns_used"] = int(sld["n_columns_used"])
    rep["sld_n_columns_censored"] = int(sld["n_columns_censored"])
    rep["sld_boundary_table"] = _md_table(
        ["omega (rpm)", "blim (mm)", "b_model (mm)", "status"],
        [[_fmt(b["omega"]), _fmt(b["blim"]),
          ("N/A" if not np.isfinite(b["b_model"]) else _fmt(b["b_model"])), b["status"]]
         for b in sld["boundary"]],
    )
    rep["caveats_list"] = "\n".join(caveats)
    if "sld_figure" in artifacts:
        rep["fig_sld"] = Path(artifacts["sld_figure"]).name
    return rep


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz-root", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--preprocessing", default=None)
    parser.add_argument("--output-dir", default="results/phase3/xai/sld")
    parser.add_argument("--assets-dir", default="results/figures/research_update_assets",
                        help="Where the SLD figure is mirrored for the research update.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--parent-group", type=int, default=None,
                       help="parent_group of the modal group (all param-twins share it).")
    group.add_argument("--dataset-ids", default=None,
                       help="Comma-separated dataset ids forming the modal group.")
    parser.add_argument("--fold", type=int, default=None, help="Fold whose checkpoint this is.")
    parser.add_argument("--val-auc", type=float, default=None)
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=512)
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--max-batch", type=int, default=256)
    # xai_phase3._load_model reads these off the args namespace.
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--spline-order", type=int, default=3)
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
    norm = NormStats.from_dict(preprocessing["norm"]) if preprocessing and "norm" in preprocessing else None

    dataset_ids = [int(d) for d in str(args.dataset_ids).split(",")] if args.dataset_ids else None
    rows = select_group_rows(args.table, args.npz_root, dataset_ids, args.parent_group)
    cfg = WindowConfig(length=args.window_length, stride=args.window_stride, channels=CHANNELS)

    results: dict = {}
    artifacts: dict = {}
    errors: dict = {}

    def _safe(name, fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - record + continue; missing -> N/A in report
            errors[name] = traceback.format_exc()
            print(f"WARNING: analysis {name!r} failed:\n{errors[name]}", file=sys.stderr)
            return None

    field = _safe("field", lambda: SL.build_pchatter_field(
        model, rows, args.npz_root, norm, cfg, max_batch=args.max_batch))
    field = field or []
    boundary = _safe("boundary", lambda: SL.extract_model_boundary(field)) or []
    analytic = _safe("analytic", lambda: SL.analytic_boundary(rows)) or []
    metrics = _safe("faithfulness", lambda: SL.boundary_faithfulness(field, boundary)) or {
        "mae_b": float("nan"), "mape_b": float("nan"), "iou_stable": float("nan"),
        "label_agreement": float("nan"), "per_cell_auc": float("nan"),
        "n_columns_used": 0, "n_columns_censored": 0,
    }

    n_omega = len(boundary)
    caveats = SL.sld_caveats(metrics, n_omega)

    # ---- Provenance -------------------------------------------------------- #
    tag = args.tag or out_dir.name
    run_id = args.run_id or Path(args.checkpoint).resolve().parent.parent.name
    n_windows = int(sum(int(c["n_windows"]) for c in field))
    omegas = [float(c["omega"]) for c in field]
    provenance = {
        "checkpoint": str(args.checkpoint),
        "preprocessing": str(pre_path) if pre_path else None,
        "run_id": run_id,
        "tag": tag,
        "val_auc": args.val_auc,
        "fold": args.fold,
        "dataset_ids": sorted(int(d) for d in rows["dataset"].unique()),
        "parent_group": int(args.parent_group) if args.parent_group is not None else None,
        "n_cells": len(field),
        "n_omega": n_omega,
        "n_files": int(len(rows)),
        "n_windows": n_windows,
        "omega_min": min(omegas) if omegas else None,
        "omega_max": max(omegas) if omegas else None,
        "side_input": bool(getattr(model, "side_input", False)),
        "pool": int(args.pool),
        "git_sha": _git_sha(_ROOT),
        "rendered_date": _now_date(),
    }

    # ---- Figure + CSV artifacts -------------------------------------------- #
    fig_name = f"sld_{tag}.png"
    title = (f"Model-implied SLD - fold {args.fold}, parent_group "
             f"{args.parent_group if args.parent_group is not None else dataset_ids}")

    def _figure():
        path = SL.plot_sld(field, boundary, analytic, metrics, out_dir / fig_name, title=title)
        assets_dir = Path(args.assets_dir)
        assets_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, assets_dir / fig_name)
        artifacts["sld_figure"] = path
        artifacts["sld_figure_asset"] = str(assets_dir / fig_name)
        return path
    _safe("figure", _figure)

    def _field_csv():
        p = out_dir / "sld_field.csv"
        pd.DataFrame(field, columns=["omega", "b_mm", "blim", "label", "p_chatter", "n_windows"]
                     ).to_csv(p, index=False)
        artifacts["sld_field_csv"] = str(p)
        return p
    _safe("field_csv", _field_csv)

    def _boundary_csv():
        p = out_dir / "sld_boundary.csv"
        pd.DataFrame(
            [{"omega": b["omega"], "blim": b["blim"], "b_model": b["b_model"],
              "status": b["status"]} for b in boundary]
        ).to_csv(p, index=False)
        artifacts["sld_boundary_csv"] = str(p)
        return p
    _safe("boundary_csv", _boundary_csv)

    # ---- Results + manifest ------------------------------------------------ #
    field_summary = {
        "n_cells": len(field),
        "n_stable_cells": int(sum(1 for c in field if np.isfinite(c["p_chatter"]) and c["p_chatter"] < 0.5)),
        "n_unstable_cells": int(sum(1 for c in field if np.isfinite(c["p_chatter"]) and c["p_chatter"] >= 0.5)),
        "mean_windows_per_cell": float(np.mean([c["n_windows"] for c in field])) if field else float("nan"),
    }
    results["sld"] = {**metrics, "boundary": boundary, "field_summary": field_summary}

    report = _build_report(provenance, results["sld"], artifacts, caveats)
    manifest = {
        "provenance": provenance,
        "results": results,
        "artifacts": artifacts,
        "errors": errors,
        "report": report,
    }
    manifest_path = out_dir / "sld_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    print(f"SLD manifest -> {manifest_path}")
    print(f"  MAE_b={_fmt(metrics['mae_b'])} mm  IoU={_fmt(metrics['iou_stable'])}  "
          f"label_agr={_fmt(metrics['label_agreement'])}  "
          f"cols {metrics['n_columns_used']} used / {metrics['n_columns_censored']} censored")
    if errors:
        print(f"WARNING: {len(errors)} analysis/analyses failed: {sorted(errors)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
