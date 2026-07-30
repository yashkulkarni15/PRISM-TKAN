#!/usr/bin/env python3
"""Score PRI2M files for chatter with the published Phase 3 TKAN model.

This is the team-facing entry point. It resolves a frozen checkpoint from the
model zoo (:mod:`pri2m.phase3.zoo`), verifies its sha256, runs inference, and
writes a per-file CSV plus a provenance sidecar. Nothing needs to be passed for
a normal run except which datasets to score and where to put the output::

    python scripts/predict_phase3.py --dataset 12,13 --out my_preds.csv

Output CSV: ``file, prob, n_windows, dataset, label, in_train_split``.
``prob`` is the mean chatter probability over the file's windows; ``label`` is
the table's stored ground truth (for comparison, not an input to the model);
``in_train_split`` is True when this checkpoint was *trained* on that file --
those rows are memorized and must not be averaged into a reported number.

For anything beyond a few dozen files, submit ``slurm/phase3_predict.sbatch``
rather than running on a login node; this script refuses large login-node runs.

Memory scales with ``--dataset`` scope: ``Phase3SequenceDataset`` loads every
selected file's full signal into RAM up front. A single dataset fits in ~32G;
``--dataset all`` needs far more, so shard by dataset and concatenate the CSVs.

See ``docs/phase3_tkan/USING_THE_MODEL.md``. Severity is deliberately not
emitted -- that head is unreliable (see the model card).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
# src/ enables ``import pri2m...``; repo root lets us reuse the sibling scripts'
# proven table-building and model-reconstruction helpers.
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402  (after sys.path setup)

from pri2m.phase3 import zoo as zoo_mod  # noqa: E402
from pri2m.phase3.predict import annotate_train_overlap, predict_files  # noqa: E402
from pri2m.training.phase3_trainer import resolve_device  # noqa: E402
from scripts.eval_heldout_phase3 import build_eval_table  # noqa: E402
from scripts.xai_phase3 import _load_model  # noqa: E402

__all__ = ["main"]

#: Great Lakes project root. Defaults are absolute so the command works from any
#: directory on GL; ``$PRI2M_PROJECT_ROOT`` redirects the whole set at once (e.g.
#: to a local checkout).
PROJECT_ROOT_ENV_VAR = "PRI2M_PROJECT_ROOT"
DEFAULT_PROJECT_ROOT = "/nfs/turbo/coe-jaabell/PRI2M"


def _project_root() -> Path:
    return Path(os.environ.get(PROJECT_ROOT_ENV_VAR, DEFAULT_PROJECT_ROOT))


#: Above this many files, a login-node run is antisocial (and slow). Small runs
#: stay allowed so people can sanity-check without queueing a job.
LOGIN_NODE_FILE_LIMIT = 50


def _parse_dataset_filter(spec: str) -> set[int] | None:
    """``'all'`` -> ``None`` (no filter); else a set of int dataset ids."""
    if spec is None or str(spec).strip().lower() == "all":
        return None
    return {int(x) for x in str(spec).split(",") if x.strip()}


def on_login_node(hostname: str | None = None) -> bool:
    """True on a Great Lakes login node outside a Slurm allocation.

    Inside a job ``SLURM_JOB_ID`` is set even though the hostname may still look
    like a login node for interactive ``salloc`` sessions, so the env var wins.
    """
    if os.environ.get("SLURM_JOB_ID"):
        return False
    host = hostname if hostname is not None else socket.gethostname()
    return host.startswith("gl-login")


def _heldout_note(metrics: dict | None) -> str:
    """`` (held-out 122_125: AUC 0.920, acc 0.841)`` when those numbers exist.

    Printed on every run so the person reading the output knows what this model
    is actually worth before they look at a single probability.
    """
    metrics = metrics or {}
    auc, acc = metrics.get("auc"), metrics.get("acc")
    if auc is None and acc is None:
        return ""
    parts = []
    if auc is not None:
        parts.append(f"AUC {float(auc):.3f}")
    if acc is not None:
        parts.append(f"acc {float(acc):.3f}")
    return f" (held-out {metrics.get('eval_set', '?')}: {', '.join(parts)})"


def explain_empty_selection(table_path: str, npz_root: str, spec: str, wanted: set[int] | None) -> str:
    """Say WHICH of the two reasons emptied the selection.

    A dataset can vanish either because ``--table`` never listed it (the default
    table covers 1-120; datasets 121+ have their own features CSV) or because its
    NPZ signals are not extracted. Those need different fixes, and conflating
    them sends people to check the wrong thing.
    """
    try:
        listed = set(pd.read_csv(table_path, usecols=["dataset"])["dataset"].astype(int))
    except Exception:  # noqa: BLE001 - fall back to the generic message
        return f"no files to score: --dataset={spec!r} matched nothing in {table_path}"

    if wanted is not None:
        missing = sorted(wanted - listed)
        if missing:
            lo, hi = (min(listed), max(listed)) if listed else ("?", "?")
            return (
                f"no files to score: dataset(s) {missing} are not in {table_path}, "
                f"which covers datasets {lo}-{hi}.\n"
                f"Datasets outside that range (e.g. the 121-155 held-out sets) have "
                f"their own features CSV -- pass it with --table."
            )
    return (
        f"no files to score: --dataset={spec!r} is in the table but no matching "
        f"NPZ bundles exist under {npz_root} (are those signals extracted?)"
    )


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    """The fields ``xai_phase3._load_model`` reads off an args namespace.

    Geometry is inferred from the checkpoint's tensor shapes; only the KAN spline
    hyper-parameters have to be supplied, and they must match training (the
    published configs all use grid 5 / order 3).
    """
    return argparse.Namespace(
        input_size=None,
        hidden_size=None,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--dataset", default="all", help="'all', an int, or a comma list of dataset ids.")
    ap.add_argument("--fold", type=int, default=None,
                    help="Published fold to use (default: the zoo's blessed fold -- use this unless "
                         "you are deliberately checking fold variance).")
    ap.add_argument("--zoo", default=None,
                    help=f"Model zoo root (default: ${zoo_mod.ZOO_ENV_VAR}, else {zoo_mod.DEFAULT_ZOO_PATH}).")
    root = _project_root()
    ap.add_argument("--table", default=str(root / "results/phase3/phase3_dataset.csv"),
                    help="Per-file table CSV (needs dataset, file columns).")
    ap.add_argument("--npz-root", default=str(root / "phase2_signals"),
                    help="Root of the dataset_<id>/*.npz signal bundles.")
    ap.add_argument("--folds-json", default=str(root / "results/phase3/phase3_folds.json"),
                    help="Fold assignments, for the train-overlap check.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto", help="cpu|cuda|cuda:0|auto (default auto).")
    ap.add_argument("--grid-size", type=int, default=5, help="KAN grid size (must match training).")
    ap.add_argument("--spline-order", type=int, default=3, help="KAN spline order (must match training).")
    ap.add_argument("--allow-train-overlap", action="store_true",
                    help="Silence the warning when scoring files this checkpoint trained on.")
    ap.add_argument("--allow-login-node", action="store_true",
                    help="Run a large job on a login node anyway. Please do not.")
    ap.add_argument("--skip-integrity", action="store_true",
                    help="Skip the sha256 check. Only for a zoo you are actively rebuilding.")
    args = ap.parse_args(argv)

    # 1. Resolve + verify the published model before touching any data. Zoo
    # problems are the most likely thing to go wrong for someone running this for
    # the first time, so they surface as a plain message, not a traceback.
    try:
        zoo_root = zoo_mod.resolve_zoo(args.zoo)
        manifest = zoo_mod.load_manifest(zoo_root)
        entry = zoo_mod.resolve_entry(zoo_root, args.fold, manifest=manifest)
        if not args.skip_integrity:
            zoo_mod.verify_entry(entry)
    except zoo_mod.ZooError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"[predict] {entry.label} from {zoo_root}{_heldout_note(entry.metrics)}")

    pre = json.loads(entry.preprocessing.read_text(encoding="utf-8"))

    # 2. Build the eval table (keeps only rows whose NPZ bundle exists).
    eval_table, _ = build_eval_table(args.table, args.npz_root)
    wanted = _parse_dataset_filter(args.dataset)
    if wanted is not None:
        eval_table = eval_table[eval_table["dataset"].astype(int).isin(wanted)].reset_index(drop=True)
    if eval_table.empty:
        raise SystemExit(explain_empty_selection(args.table, args.npz_root, args.dataset, wanted))

    if (
        len(eval_table) > LOGIN_NODE_FILE_LIMIT
        and on_login_node()
        and not args.allow_login_node
    ):
        raise SystemExit(
            f"refusing to score {len(eval_table)} files on a login node.\n"
            f"Submit the job instead:\n"
            f"  sbatch --export=ALL,DATASET={args.dataset},OUT={args.out} "
            f"slurm/phase3_predict.sbatch\n"
            f"(or pass --allow-login-node if you really mean it)"
        )

    # 3. Score.
    device = resolve_device(args.device)
    model = _load_model(str(entry.checkpoint), _model_args(args), pre)
    predictions = predict_files(
        model, eval_table, args.npz_root, pre,
        batch_size=args.batch_size, device=device,
    )

    # 4. Flag rows this checkpoint was trained on.
    n_overlap = 0
    folds_path = Path(args.folds_json)
    if folds_path.is_file():
        folds = json.loads(folds_path.read_text(encoding="utf-8"))
        predictions, n_overlap = annotate_train_overlap(predictions, folds, entry.fold)
        if n_overlap and not args.allow_train_overlap:
            print(
                f"WARNING: {n_overlap}/{len(predictions)} scored files were in "
                f"fold {entry.fold}'s TRAINING split -- the model has seen them and "
                f"their probabilities are optimistic. Filter on in_train_split == "
                f"False before reporting any aggregate.",
                file=sys.stderr,
            )
    else:
        print(
            f"WARNING: {folds_path} not found -- cannot check whether these files "
            f"were in the training split. The in_train_split column is omitted.",
            file=sys.stderr,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)

    # 5. Provenance sidecar: what model, what code, what data, how many rows.
    provenance = {
        "zoo": str(zoo_root),
        "zoo_version": manifest.get("version"),
        "fold": entry.fold,
        "run_id": entry.run_id,
        "checkpoint_sha256": entry.sha256,
        "integrity_verified": not args.skip_integrity,
        "code_commit": entry.code_commit,
        "heldout_metrics": entry.metrics,
        "table": str(args.table),
        "npz_root": str(args.npz_root),
        "dataset_filter": args.dataset,
        "device": str(device),
        "n_files": int(len(predictions)),
        "n_in_train_split": int(n_overlap),
    }
    sidecar = out.with_suffix(out.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(f"[predict] scored {len(predictions)} files -> {out}")
    print(f"[predict] provenance -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
