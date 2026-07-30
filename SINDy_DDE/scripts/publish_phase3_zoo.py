#!/usr/bin/env python3
"""Publish trained Phase 3 TKAN checkpoints into a frozen, versioned model zoo.

Copies each fold's ``best_auc.pt`` + ``preprocessing.json`` out of its MLflow run
directory into ``<out>/fold<k>/``, writes a ``manifest.json`` (run id, sha256,
held-out metrics, train/val groups, code commit) and a ``MODEL_CARD.md``, then
makes the whole tree read-only.

Why copy instead of symlink: ``scripts/cleanup_mlflow_checkpoints.py --execute``
deletes intermediate checkpoints inside ``results/phase3/runs/``. A symlinked zoo
would break the moment retention runs. The weights are a few MB -- copying is the
cheap half of the trade.

Run this ON GREAT LAKES (the run directories and turbo destination are both
there). It refuses to overwrite an existing version directory: publish ``v2``
rather than mutating ``v1``, so a number someone cited last month still
reproduces.

Example::

    python scripts/publish_phase3_zoo.py \
        --out /nfs/turbo/coe-jaabell/PRI2M/models/phase3_tkan/v1 \
        --default-fold 4
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pri2m.phase3.zoo import (  # noqa: E402
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    sha256_file,
)

__all__ = ["main"]

_CHECKPOINT_NAME = "best_auc.pt"
_PREPROCESSING_NAME = "preprocessing.json"

# Read-only for everyone, executable(=traversable) dirs for the group. The zoo is
# meant to be immutable once published; permissions say so out loud.
_FILE_MODE = 0o444
_DIR_MODE = 0o555


def _git_commit(repo_root: Path) -> str | None:
    """Short HEAD sha of the code being published, or None outside a git tree."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None


def discover_runs(runs_root: Path, skip_folds: set[int] | None = None) -> dict[int, str]:
    """Map ``fold -> run_id`` by reading each run's ``summary.json``.

    Only runs that actually have a ``checkpoints/best_auc.pt`` count. A fold with
    two complete runs is ambiguous and raises -- picking one silently is how the
    wrong model gets published.

    ``skip_folds`` are folds the caller already pinned with ``--run``; they are
    excluded from both discovery and the ambiguity check, so pinning is a real
    escape hatch from an ambiguous fold rather than a flag that never gets
    reached. (Fold 0 on turbo has two complete runs: the unstable lr-0.003
    attempt and the validated lr-0.0015 one.)
    """
    skip_folds = skip_folds or set()
    found: dict[int, list[str]] = {}
    for summary_path in sorted(runs_root.glob("*/summary.json")):
        run_dir = summary_path.parent
        if not (run_dir / "checkpoints" / _CHECKPOINT_NAME).is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if summary.get("fold") is None or int(summary["fold"]) in skip_folds:
            continue
        found.setdefault(int(summary["fold"]), []).append(run_dir.name)

    ambiguous = {k: v for k, v in found.items() if len(v) > 1}
    if ambiguous:
        detail = "; ".join(f"fold {k}: {', '.join(v)}" for k, v in sorted(ambiguous.items()))
        raise SystemExit(
            f"more than one complete run per fold ({detail}).\n"
            "Disambiguate with --run FOLD:RUN_ID (repeatable)."
        )
    return {k: v[0] for k, v in found.items()}


def find_heldout(heldout_dir: Path, run_id: str, tag: str) -> dict | None:
    """Held-out metrics for ``run_id``, matched by filename prefix.

    Eval outputs are named either ``<run_id>_<tag>.json`` or ``<run_id[:8]>_<tag>.json``
    depending on which invocation wrote them; both are accepted. Returns the
    ``metrics`` block, or None when that fold was never scored held-out.
    """
    if not heldout_dir.is_dir():
        return None
    for candidate in (f"{run_id}_{tag}.json", f"{run_id[:8]}_{tag}.json"):
        path = heldout_dir / candidate
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = dict(payload.get("metrics") or {})
            metrics["eval_set"] = tag
            return metrics
    return None


def fold_groups(folds: dict, fold: int) -> tuple[list[int], list[int]]:
    """``(train_groups, val_groups)`` for ``fold`` from a phase3_folds payload.

    ``group_to_fold[g] == k`` means group ``g`` is fold ``k``'s validation set,
    so fold ``k`` trained on everything else except the always-held-out
    ``dev_groups``.
    """
    group_to_fold = {int(g): int(k) for g, k in (folds.get("group_to_fold") or {}).items()}
    dev_groups = {int(g) for g in (folds.get("dev_groups") or ())}
    val = sorted(g for g, k in group_to_fold.items() if k == int(fold))
    train = sorted(
        g for g, k in group_to_fold.items() if k != int(fold) and g not in dev_groups
    )
    return train, val


def _model_card(manifest: dict) -> str:
    """Render MODEL_CARD.md -- the in-zoo copy of what to cite and what not to."""
    default = str(manifest["default_fold"])
    rows = []
    for key in sorted(manifest["folds"], key=int):
        block = manifest["folds"][key]
        held = block.get("heldout") or {}
        mark = " **(default)**" if key == default else ""
        rows.append(
            "| {fold}{mark} | `{run}` | {auc} | {acc} | {mae} |".format(
                fold=key,
                mark=mark,
                run=block["run_id"][:10],
                auc=_fmt(held.get("auc")),
                acc=_fmt(held.get("acc")),
                mae=_fmt(held.get("mae")),
            )
        )
    eval_tag = manifest.get("heldout_eval_set", "held-out")
    return f"""# PRI2M Phase 3 TKAN -- {manifest["version"]}

Published {manifest["created"]} from code commit `{manifest.get("code_commit") or "unknown"}`.

Per-file chatter classifier. Input: a 1024-sample window over
`xtpos, ytpos, Forcex, Forcey` plus the process parameters `omega, tpf, tau`.
Output: `prob`, the mean of `sigmoid(chatter_logit)` across a file's windows.

## Folds

Metrics are on the strict held-out set (`{eval_tag}`) -- datasets never seen in
any fold -- not cross-validation.

| fold | run | AUC | acc | MAE (severity) |
|---|---|---|---|---|
{chr(10).join(rows)}

**Cite fold {default}.** It is the default because it pairs top-tier held-out AUC
with the best accuracy at the 0.5 threshold; other folds have comparable AUC but
worse-calibrated thresholds. The other folds are published for variance checks,
not for reporting.

## What this model is not

- **Do not cite the 0.955 cross-validation AUC.** That number is inflated by
  twin-dataset leakage (126-155 replicate the operating points of 1-30). The
  honest number is the held-out AUC above.
- **The severity head is unreliable.** RMSE runs roughly 2x MAE because a few
  files dominate the error. `scripts/predict_phase3.py` deliberately does not
  emit severity; treat this model as a classifier.
- **Trained on datasets 1-120 only.** Transfer to a different machine, tool, or
  material is unmeasured.
- Scoring files the checkpoint trained on inflates results. The CLI flags those
  rows via `in_train_split`; do not average over them.

## Provenance

`manifest.json` records each fold's run id, sha256, and the parent groups it
trained and validated on. `scripts/predict_phase3.py` verifies the sha256 before
loading and refuses to run on a mismatch.
"""


def _fmt(value) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def _freeze(root: Path) -> None:
    """chmod the published tree read-only (files 444, dirs 555)."""
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, _DIR_MODE if path.is_dir() else _FILE_MODE)
    os.chmod(root, _DIR_MODE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Version directory to create, e.g. .../phase3_tkan/v1")
    ap.add_argument("--runs-root", default="results/phase3/runs", help="MLflow run directories.")
    ap.add_argument("--heldout-dir", default="results/phase3/heldout", help="Held-out eval JSONs.")
    ap.add_argument("--folds-json", default="results/phase3/phase3_folds.json")
    ap.add_argument("--heldout-tag", default="122_125", help="Held-out eval set to record (default: the strict one).")
    ap.add_argument("--default-fold", type=int, default=4, help="The fold the team should cite.")
    ap.add_argument("--run", action="append", default=[], metavar="FOLD:RUN_ID",
                    help="Pin a fold to a run id, overriding auto-discovery. Repeatable.")
    ap.add_argument("--created", default=None, help="Publication date (default: today, UTC).")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing version directory.")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(
            f"{out} already exists. Published versions are immutable -- publish a "
            "new version directory (v2, v3, ...) or pass --force if this one was "
            "never shared."
        )

    # Parse the pins FIRST so a pinned fold is exempt from the ambiguity check --
    # otherwise discovery would raise before the override could be applied.
    pinned: dict[int, str] = {}
    for spec in args.run:
        fold_str, _, run_id = spec.partition(":")
        if not run_id:
            raise SystemExit(f"--run expects FOLD:RUN_ID, got {spec!r}")
        pinned[int(fold_str)] = run_id

    runs = discover_runs(Path(args.runs_root), skip_folds=set(pinned))
    runs.update(pinned)
    if not runs:
        raise SystemExit(f"no complete runs (with {_CHECKPOINT_NAME}) under {args.runs_root}")
    if args.default_fold not in runs:
        raise SystemExit(
            f"--default-fold {args.default_fold} has no run; found folds {sorted(runs)}"
        )

    folds_payload = json.loads(Path(args.folds_json).read_text(encoding="utf-8"))
    created = args.created or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    if out.exists() and args.force:
        _unfreeze_and_remove(out)
    out.mkdir(parents=True)

    manifest_folds: dict[str, dict] = {}
    for fold in sorted(runs):
        run_id = runs[fold]
        run_dir = Path(args.runs_root) / run_id
        src_ckpt = run_dir / "checkpoints" / _CHECKPOINT_NAME
        src_pre = run_dir / _PREPROCESSING_NAME
        for src in (src_ckpt, src_pre):
            if not src.is_file():
                raise SystemExit(f"fold {fold}: missing {src}")

        fold_dir = out / f"fold{fold}"
        fold_dir.mkdir()
        shutil.copy2(src_ckpt, fold_dir / _CHECKPOINT_NAME)
        shutil.copy2(src_pre, fold_dir / _PREPROCESSING_NAME)

        # Hash the COPY, not the source: this verifies the copy landed intact and
        # is the value predict_phase3.py will re-check at load time.
        digest = sha256_file(fold_dir / _CHECKPOINT_NAME)
        if digest != sha256_file(src_ckpt):
            raise SystemExit(f"fold {fold}: copy of {src_ckpt} does not match the source")

        train_groups, val_groups = fold_groups(folds_payload, fold)
        manifest_folds[str(fold)] = {
            "run_id": run_id,
            "sha256": digest,
            "heldout": find_heldout(Path(args.heldout_dir), run_id, args.heldout_tag),
            "train_groups": train_groups,
            "val_groups": val_groups,
        }
        print(f"[zoo] fold {fold}: {run_id[:10]} -> {fold_dir}")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "version": out.name,
        "created": created,
        "code_commit": _git_commit(_ROOT),
        "default_fold": int(args.default_fold),
        "heldout_eval_set": args.heldout_tag,
        "folds": manifest_folds,
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "MODEL_CARD.md").write_text(_model_card(manifest), encoding="utf-8")

    _freeze(out)
    print(f"[zoo] published {len(manifest_folds)} folds (default: {args.default_fold}) -> {out}")
    print(f"[zoo] frozen read-only; teammates run: python scripts/predict_phase3.py --help")
    return 0


def _unfreeze_and_remove(root: Path) -> None:
    """Make a previously frozen zoo writable again so --force can replace it."""
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | (stat.S_IXUSR if path.is_dir() else 0))
    os.chmod(root, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
    shutil.rmtree(root)


if __name__ == "__main__":
    raise SystemExit(main())
