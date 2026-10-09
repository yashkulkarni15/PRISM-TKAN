"""Re-measure the pruning curve on held-out data, against a random control.

Item B of ``docs/superpowers/specs/2026-08-12-spline-substitution-design.md``.

The pruning curve already in every fold manifest was computed on the
in-distribution sample, starts at 2% of edges kept, and has six points. It is
under-resolved at BOTH ends, which was checked rather than assumed:

  * no point below 0.02, so where inside [0, 0.02] the network collapses is
    unmeasured; and
  * its steepest measured segment is 0.50 -> 1.00 in three of five folds
    (fold 0 +0.094, fold 2 +0.123, fold 4 +0.076) -- the other two are steepest
    at 0.02 -> 0.05 -- so one segment spanning half the edges hides where that
    contribution actually sits.

Every fraction is also run against a size-matched random edge set. Without that
control the curve measures "removing edges hurts", not "the importance ranking
is correct".

Reported: the smallest kept fraction whose held-out AUC stays within the
tolerance of THIS FOLD'S OWN baseline. Never the five-fold mean 0.908 -- mixing
a per-fold measurement with a five-fold statistic is what made Update 14's
0.0092 misleading.

Usage::

    python3 scripts/heldout_pruning_phase3.py \\
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \\
        --features-csv data/metadata/features_datasets_121_125.csv \\
        --npz-root phase2_signals
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_heldout_phase3 as EH  # noqa: E402
import xai_phase3  # noqa: E402
from pri2m.xai.activations import collect_activations  # noqa: E402
from pri2m.xai.faithfulness import prune_context, rank_edges  # noqa: E402
from pri2m.xai.sampling import (  # noqa: E402
    batch_to_model_device,
    build_labeled_batch,
    window_config_from_preprocessing,
)

# Held-out fold spread (docs/phase3_tkan/heldout_eval_121_155.md). Used as the
# tolerance, never as a substitute for a fold's own baseline.
DEFAULT_TOL = 0.013

# Resolution at both under-resolved ends of the published curve. The published
# six were (0.02, 0.05, 0.1, 0.25, 0.5, 1.0).
DEFAULT_FRACTIONS = (
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0,
)

DEFAULT_OUT = pathlib.Path("results/phase3/xai/heldout_pruning")


def select_edges(ranked, fraction: float) -> dict:
    """Edges to ZERO in order to keep the top ``fraction`` by importance.

    ``ranked`` is :func:`pri2m.xai.faithfulness.rank_edges` output --
    ``(kan_name, o, i, importance)`` sorted by importance descending.

    Returns ``{kan_name: [(o, i), ...]}``, the shape
    :func:`pri2m.xai.faithfulness.prune_context` expects.

    Raises
    ------
    ValueError
        If ``fraction`` rounds to zero kept edges. Silently keeping nothing
        would report a floor AUC as though it were a measurement.
    """
    n_total = len(ranked)
    n_keep = int(round(fraction * n_total))
    if n_keep <= 0:
        raise ValueError(
            f"fraction {fraction} keeps 0 edges of {n_total}; raise it"
        )
    zeroed: dict = {}
    for name, o, i, _imp in ranked[n_keep:]:
        zeroed.setdefault(name, []).append((int(o), int(i)))
    return zeroed


def random_edges(ranked, n: int, seed: int) -> dict:
    """A size-matched random edge set -- the control for :func:`select_edges`.

    Exactly ``n`` edges, drawn without replacement. A control of a different
    size measures size rather than the ranking.
    """
    if n > len(ranked):
        raise ValueError(f"cannot draw {n} edges; only {len(ranked)} exist")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ranked), size=n, replace=False)
    zeroed: dict = {}
    for k in sorted(int(v) for v in idx):
        name, o, i, _imp = ranked[k]
        zeroed.setdefault(name, []).append((int(o), int(i)))
    return zeroed


def smallest_faithful_fraction(rows: list[dict], baseline: float, tol: float):
    """Smallest kept fraction whose ranked-prune AUC stays within ``tol``.

    ``None`` when no tested fraction qualifies -- which is a result, not an
    error, and must not be rendered as the largest fraction tested.
    """
    ok = [r["fraction"] for r in rows
          if r["ranked_auc"] is not None and abs(r["ranked_auc"] - baseline) <= tol]
    return min(ok) if ok else None


def _score(checkpoint, args, preprocessing, model, tag, out_dir):
    """Held-out AUC for ``model``, via the existing held-out evaluator."""
    try:
        res = EH.run(
            checkpoint=str(checkpoint),
            features_csv=args.features_csv,
            npz_root=args.npz_root,
            out=str(out_dir / f"_eval_{tag}.json"),
            preprocessing=args.preprocessing,
            device=args.device,
            batch_size=args.batch_size,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 - an arm that dies must not be a pass
        return None, f"{type(exc).__name__}: {exc}"
    # eval_heldout_phase3.run nests its metrics: the AUC is at
    # result["metrics"]["auc"], NOT at the top level. Reading the top level
    # returns None for a run that in fact succeeded.
    auc = ((res or {}).get("metrics") or {}).get("auc")
    if auc is None:
        return None, f"evaluator returned no metrics.auc (keys: {sorted((res or {}))})"
    return float(auc), None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features-csv", required=True,
                    help="Held-out features CSV (the 121-125 path).")
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--table", default="results/phase3/phase3_dataset.csv",
                    help="Training table, used only to build the activation bundle.")
    ap.add_argument("--preprocessing", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=256,
                    help="Windows used to measure edge importance.")
    ap.add_argument("--fractions", default=None,
                    help="Comma-separated kept fractions. Defaults to "
                         f"{','.join(str(f) for f in DEFAULT_FRACTIONS)}, which "
                         "adds resolution at both ends the published curve "
                         "under-resolves.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help="Held-out fold spread; the tolerance around THIS "
                         "fold's own baseline.")
    ap.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--tag", default="fold2")
    args = ap.parse_args(argv)

    fractions = (DEFAULT_FRACTIONS if args.fractions is None else
                 tuple(float(f) for f in args.fractions.split(",") if f.strip()))

    ckpt = pathlib.Path(args.checkpoint)
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}", file=sys.stderr)
        return 2
    pre_path = (pathlib.Path(args.preprocessing) if args.preprocessing
                else ckpt.parent.parent / "preprocessing.json")
    preprocessing = json.loads(pre_path.read_text()) if pre_path.exists() else None

    load_args = argparse.Namespace(input_size=None, hidden_size=None,
                                   grid_size=5, spline_order=3)
    model = xai_phase3._load_model(str(ckpt), load_args, preprocessing)
    model.eval()

    out_dir = pathlib.Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "heldout_pruning.json"

    # Provenance BEFORE the run, so a crash still leaves a readable record of
    # what was attempted (the _git_sha lesson).
    manifest = {
        "provenance": {
            "complete": False,
            "checkpoint": str(ckpt),
            "features_csv": args.features_csv,
            "fractions": list(fractions),
            "n_windows": int(args.n_windows),
            "seed": int(args.seed),
            "tol": float(args.tol),
            "tag": args.tag,
        },
        "rows": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    baseline, err = _score(ckpt, args, preprocessing, model, "baseline", out_dir)
    if baseline is None:
        print(f"Baseline evaluation failed: {err}", file=sys.stderr)
        return 1
    manifest["baseline_auc"] = baseline
    print(f"baseline held-out AUC = {baseline:.4f}")

    # cfg is REQUIRED: passing None reaches Phase3SequenceDataset and dies on
    # cfg.channels. Take the geometry from the checkpoint's own preprocessing so
    # the windows analysed match the ones the network was trained on.
    cfg = window_config_from_preprocessing(preprocessing)
    batch, _labels = build_labeled_batch(
        args.table, args.npz_root, preprocessing, args.n_windows,
        cfg, seed=args.seed,
    )
    # The baseline eval above already ran, and eval_heldout_phase3.run moves the
    # model to its device as a side effect. build_labeled_batch always returns
    # CPU tensors, so without this the next model(batch) dies on a device
    # mismatch. Read the device from the model, not from --device.
    batch = batch_to_model_device(batch, model)
    activations = collect_activations(model, batch)
    ranked = rank_edges(activations)
    manifest["provenance"]["n_edges_total"] = len(ranked)
    print(f"ranked {len(ranked)} edges")

    for fraction in fractions:
        row = {"fraction": float(fraction)}
        try:
            ranked_zeroed = select_edges(ranked, fraction)
        except ValueError as exc:
            row.update({"ranked_auc": None, "random_auc": None, "error": str(exc)})
            manifest["rows"].append(row)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"  keep {fraction:<6} SKIPPED: {exc}")
            continue

        n_zeroed = sum(len(v) for v in ranked_zeroed.values())
        row["n_zeroed"] = n_zeroed

        if n_zeroed == 0:
            # Keeping everything: the ranked arm IS the baseline and there is no
            # control to draw, so say so rather than invent a random set of 0.
            row.update({"ranked_auc": baseline, "random_auc": None,
                        "note": "keeps every edge; identical to baseline"})
            manifest["rows"].append(row)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"  keep {fraction:<6} ranked={baseline:.4f}  (baseline)")
            continue

        with prune_context(model, ranked_zeroed):
            row["ranked_auc"], row["ranked_error"] = _score(
                ckpt, args, preprocessing, model, f"ranked_{fraction}", out_dir)

        ctrl = random_edges(ranked, n_zeroed, seed=args.seed)
        with prune_context(model, ctrl):
            row["random_auc"], row["random_error"] = _score(
                ckpt, args, preprocessing, model, f"random_{fraction}", out_dir)

        manifest["rows"].append(row)
        # Partial written after every fraction, so a wall kill leaves a
        # readable curve rather than nothing.
        manifest_path.write_text(json.dumps(manifest, indent=2))

        r, c = row.get("ranked_auc"), row.get("random_auc")
        print(f"  keep {fraction:<6} n_zeroed={n_zeroed:6d}  "
              f"ranked={'None' if r is None else format(r, '.4f')}  "
              f"random={'None' if c is None else format(c, '.4f')}")

    manifest["smallest_faithful_fraction"] = smallest_faithful_fraction(
        manifest["rows"], baseline, float(args.tol))
    manifest["provenance"]["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Assert the artifact rather than echo its path: a job exiting 0 is not
    # evidence it did the work.
    written = json.loads(manifest_path.read_text())
    assert written["provenance"]["complete"] is True, "manifest not marked complete"
    assert written["rows"], "manifest has no rows"
    print(f"\nWrote {manifest_path} ({len(written['rows'])} rows, "
          f"smallest faithful fraction = {written['smallest_faithful_fraction']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
