"""pri2m.phase3.zoo: manifest resolution, fold selection, integrity."""
from __future__ import annotations

import json

import pytest

from pri2m.phase3.zoo import (
    MANIFEST_SCHEMA,
    ZOO_ENV_VAR,
    ZooError,
    ZooIntegrityError,
    load_manifest,
    resolve_entry,
    resolve_zoo,
    sha256_file,
    verify_entry,
)


def make_zoo(root, *, folds=(0, 4), default_fold=4, schema=MANIFEST_SCHEMA):
    """A miniature published zoo: two folds of fake weights plus a manifest."""
    root.mkdir(parents=True, exist_ok=True)
    blocks = {}
    for fold in folds:
        fold_dir = root / f"fold{fold}"
        fold_dir.mkdir()
        ckpt = fold_dir / "best_auc.pt"
        ckpt.write_bytes(f"weights-for-fold-{fold}".encode())
        (fold_dir / "preprocessing.json").write_text(json.dumps({"norm": {"mean": [0.0], "std": [1.0]}}))
        blocks[str(fold)] = {
            "run_id": f"run{fold}" + "0" * 28,
            "sha256": sha256_file(ckpt),
            "heldout": {"auc": 0.92, "acc": 0.84, "eval_set": "122_125"},
            "train_groups": [2, 3],
            "val_groups": [6, 15],
        }
    (root / "manifest.json").write_text(json.dumps({
        "schema": schema,
        "version": root.name,
        "created": "2026-07-21",
        "code_commit": "abc123",
        "default_fold": default_fold,
        "heldout_eval_set": "122_125",
        "folds": blocks,
    }))
    return root


def test_resolve_zoo_prefers_explicit_over_env(tmp_path, monkeypatch):
    explicit = make_zoo(tmp_path / "explicit")
    make_zoo(tmp_path / "from_env")
    monkeypatch.setenv(ZOO_ENV_VAR, str(tmp_path / "from_env"))
    assert resolve_zoo(explicit) == explicit


def test_resolve_zoo_falls_back_to_env(tmp_path, monkeypatch):
    zoo = make_zoo(tmp_path / "from_env")
    monkeypatch.setenv(ZOO_ENV_VAR, str(zoo))
    assert resolve_zoo(None) == zoo


def test_resolve_zoo_error_names_the_search_path(tmp_path, monkeypatch):
    monkeypatch.delenv(ZOO_ENV_VAR, raising=False)
    with pytest.raises(ZooError, match="no manifest.json"):
        resolve_zoo(tmp_path / "nope")


def test_manifest_schema_mismatch_is_loud(tmp_path):
    zoo = make_zoo(tmp_path / "v1", schema=MANIFEST_SCHEMA + 1)
    with pytest.raises(ZooError, match="schema"):
        load_manifest(zoo)


def test_resolve_entry_defaults_to_blessed_fold(tmp_path):
    zoo = make_zoo(tmp_path / "v1", folds=(0, 4), default_fold=4)
    entry = resolve_entry(zoo)
    assert entry.fold == 4
    assert entry.checkpoint == zoo / "fold4" / "best_auc.pt"
    assert entry.metrics["auc"] == pytest.approx(0.92)
    assert entry.code_commit == "abc123"


def test_resolve_entry_honours_explicit_fold(tmp_path):
    zoo = make_zoo(tmp_path / "v1", folds=(0, 4), default_fold=4)
    assert resolve_entry(zoo, fold=0).fold == 0


def test_resolve_entry_rejects_unpublished_fold(tmp_path):
    zoo = make_zoo(tmp_path / "v1", folds=(0, 4))
    with pytest.raises(ZooError, match="fold 3 is not published"):
        resolve_entry(zoo, fold=3)


def test_verify_entry_accepts_intact_checkpoint(tmp_path):
    zoo = make_zoo(tmp_path / "v1")
    verify_entry(resolve_entry(zoo))  # does not raise


def test_verify_entry_rejects_tampered_weights(tmp_path):
    zoo = make_zoo(tmp_path / "v1")
    entry = resolve_entry(zoo)
    entry.checkpoint.write_bytes(b"different weights")
    with pytest.raises(ZooIntegrityError, match="sha256 mismatch"):
        verify_entry(entry)


def test_verify_entry_requires_preprocessing(tmp_path):
    zoo = make_zoo(tmp_path / "v1")
    entry = resolve_entry(zoo)
    entry.preprocessing.unlink()
    with pytest.raises(ZooError, match="preprocessing.json missing"):
        verify_entry(entry)
