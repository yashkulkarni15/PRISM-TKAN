"""Frozen, versioned model zoo for published Phase 3 TKAN checkpoints.

Training writes checkpoints into MLflow run directories
(``results/phase3/runs/<run_id>/checkpoints/``). Those are **not** a stable
address for anyone but the person who launched the run:
``scripts/cleanup_mlflow_checkpoints.py --execute`` computes a keep/delete plan
over exactly those directories, so a routine retention pass can remove a
checkpoint a teammate has hard-coded.

The zoo is the answer: a versioned directory of *copies*, made read-only, with a
manifest recording provenance (run id, sha256, held-out metrics, the training
groups, the code commit). Published by ``scripts/publish_phase3_zoo.py``,
consumed by ``scripts/predict_phase3.py``.

Layout::

    <zoo>/                       # e.g. .../models/phase3_tkan/v1
      manifest.json
      MODEL_CARD.md
      fold4/best_auc.pt
      fold4/preprocessing.json
      fold0/...  fold1/...  fold2/...  fold3/...

Resolution order for the zoo root: an explicit path (CLI ``--zoo``), then
``$PRI2M_MODEL_ZOO``, then :data:`DEFAULT_ZOO_PATH`. The environment variable is
what lets a teammate point at a scratch copy without editing code.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_ZOO_PATH",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "ZOO_ENV_VAR",
    "ZooEntry",
    "ZooError",
    "ZooIntegrityError",
    "load_manifest",
    "resolve_entry",
    "resolve_zoo",
    "sha256_file",
    "verify_entry",
]

#: Published location on Great Lakes turbo. Read-only to the group.
DEFAULT_ZOO_PATH = "/nfs/turbo/coe-jaabell/PRI2M/models/phase3_tkan/v1"

#: Overrides :data:`DEFAULT_ZOO_PATH` when set (e.g. a local test copy).
ZOO_ENV_VAR = "PRI2M_MODEL_ZOO"

MANIFEST_NAME = "manifest.json"

#: Bumped only on a breaking manifest change; readers check it so an old
#: checkout fails loudly instead of misreading a newer layout.
MANIFEST_SCHEMA = 1

_CHECKPOINT_NAME = "best_auc.pt"
_PREPROCESSING_NAME = "preprocessing.json"


class ZooError(RuntimeError):
    """The zoo is missing, malformed, or does not contain the requested fold."""


class ZooIntegrityError(ZooError):
    """A published checkpoint's sha256 does not match its manifest entry."""


@dataclass(frozen=True)
class ZooEntry:
    """One published fold: where its files are and how it scored.

    ``metrics`` is the held-out block copied from the manifest (typically the
    122-125 strict held-out probe), not a training-time number.
    """

    fold: int
    run_id: str
    checkpoint: Path
    preprocessing: Path
    sha256: str
    metrics: dict
    train_groups: tuple[int, ...]
    val_groups: tuple[int, ...]
    code_commit: str | None = None

    @property
    def label(self) -> str:
        """Short human-facing id, e.g. ``fold4/0937883918``."""
        return f"fold{self.fold}/{self.run_id[:10]}"


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file, hex-encoded.

    Chunked so publishing a zoo never loads a whole checkpoint into memory.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_zoo(explicit: str | Path | None = None) -> Path:
    """Resolve the zoo root: explicit path > ``$PRI2M_MODEL_ZOO`` > default.

    Raises :class:`ZooError` with the search order spelled out when the resolved
    directory has no manifest -- the common failure is a teammate running off
    Great Lakes, and the message needs to say so rather than trip a FileNotFound
    deeper in the load path.
    """
    if explicit:
        root = Path(explicit)
        source = "--zoo"
    elif os.environ.get(ZOO_ENV_VAR):
        root = Path(os.environ[ZOO_ENV_VAR])
        source = f"${ZOO_ENV_VAR}"
    else:
        root = Path(DEFAULT_ZOO_PATH)
        source = "the built-in default"

    if not (root / MANIFEST_NAME).is_file():
        raise ZooError(
            f"no {MANIFEST_NAME} under {root} (from {source}).\n"
            f"The published zoo lives at {DEFAULT_ZOO_PATH} on Great Lakes "
            f"turbo; if you are running elsewhere, copy it and set "
            f"${ZOO_ENV_VAR} to the copy."
        )
    return root


def load_manifest(zoo: str | Path) -> dict:
    """Read and schema-check a zoo manifest."""
    path = Path(zoo) / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # corrupted publish / partial copy
        raise ZooError(f"{path} is not valid JSON: {exc}") from exc

    schema = int(manifest.get("schema", 0))
    if schema != MANIFEST_SCHEMA:
        raise ZooError(
            f"{path} has schema {schema}, this code reads {MANIFEST_SCHEMA}. "
            "Update the checkout (or republish the zoo) -- refusing to guess "
            "at the layout."
        )
    if not manifest.get("folds"):
        raise ZooError(f"{path} lists no folds")
    return manifest


def resolve_entry(
    zoo: str | Path, fold: int | None = None, manifest: dict | None = None
) -> ZooEntry:
    """Return the :class:`ZooEntry` for ``fold`` (default: the blessed fold).

    ``fold=None`` selects ``manifest["default_fold"]`` -- the one fold the team
    is meant to cite, so that two people who both "just ran the model" get the
    same number.
    """
    root = Path(zoo)
    manifest = manifest if manifest is not None else load_manifest(root)
    folds = manifest["folds"]

    key = str(manifest["default_fold"] if fold is None else fold)
    if key not in folds:
        available = ", ".join(sorted(folds, key=int))
        raise ZooError(f"fold {key} is not published in {root} (have: {available})")

    block = folds[key]
    fold_dir = root / f"fold{key}"
    return ZooEntry(
        fold=int(key),
        run_id=str(block["run_id"]),
        checkpoint=fold_dir / _CHECKPOINT_NAME,
        preprocessing=fold_dir / _PREPROCESSING_NAME,
        sha256=str(block["sha256"]),
        metrics=dict(block.get("heldout") or {}),
        train_groups=tuple(int(g) for g in block.get("train_groups") or ()),
        val_groups=tuple(int(g) for g in block.get("val_groups") or ()),
        code_commit=block.get("code_commit") or manifest.get("code_commit"),
    )


def verify_entry(entry: ZooEntry) -> None:
    """Hard-fail unless the on-disk checkpoint matches its manifest sha256.

    Cheap insurance against a truncated rsync or an edited zoo: a silently wrong
    weights file would otherwise surface as plausible-looking probabilities.
    """
    if not entry.checkpoint.is_file():
        raise ZooError(f"published checkpoint missing: {entry.checkpoint}")
    if not entry.preprocessing.is_file():
        raise ZooError(
            f"published preprocessing.json missing: {entry.preprocessing} "
            "(without it the z-score and side features cannot be recovered)"
        )
    actual = sha256_file(entry.checkpoint)
    if actual != entry.sha256:
        raise ZooIntegrityError(
            f"sha256 mismatch for {entry.checkpoint}\n"
            f"  manifest: {entry.sha256}\n"
            f"  on disk:  {actual}\n"
            "The published weights have changed since the zoo was written. "
            "Do not use this result; tell whoever owns the zoo."
        )
