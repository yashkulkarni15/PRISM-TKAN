"""predict_phase3 + publish_phase3_zoo: the team-facing inference path."""
from __future__ import annotations

import json
import os
import pathlib
import sys

import pandas as pd
import pytest
import torch

from pri2m.models.tkan_model import TKANModel
from pri2m.phase3.predict import annotate_train_overlap
from pri2m.phase3.zoo import ZOO_ENV_VAR, load_manifest, sha256_file
from tests._phase3_synth import make_synth_npz_tree, make_synth_table

# Sibling scripts live under scripts/ (not a package on sys.path by default);
# conftest.py only puts src/ and the repo root on sys.path. Mirrors the pattern
# used by tests/test_predict_windows_phase3.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import predict_phase3 as PP  # noqa: E402
import publish_phase3_zoo as PUB  # noqa: E402

PREPROCESSING = {
    "norm": {"mean": [0.0] * 4, "std": [1.0] * 4},
    "window": {"length": 128, "stride": 64},
    "pool": 1,
}


def _make_run(runs_root, run_id, fold):
    """A minimal MLflow-shaped run directory the publisher can read."""
    torch.manual_seed(fold)
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3)
    run_dir = runs_root / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    torch.save(model.state_dict(), run_dir / "checkpoints" / "best_auc.pt")
    (run_dir / "preprocessing.json").write_text(json.dumps(PREPROCESSING))
    (run_dir / "summary.json").write_text(json.dumps({"run_id": run_id, "fold": fold, "model": "tkan"}))
    return run_dir


def _make_folds_json(path, *, group_to_fold, dev_groups=()):
    path.write_text(json.dumps({
        "n_splits": 2,
        "seed": 0,
        "dev_groups": list(dev_groups),
        "group_to_fold": {str(g): k for g, k in group_to_fold.items()},
        "leakage_verified": True,
    }))
    return path


@pytest.fixture
def published(tmp_path):
    """A published zoo (folds 0 and 1) plus the synthetic data to score."""
    runs_root = tmp_path / "runs"
    _make_run(runs_root, "a" * 32, fold=0)
    _make_run(runs_root, "b" * 32, fold=1)

    heldout = tmp_path / "heldout"
    heldout.mkdir()
    (heldout / f"{'b' * 32}_122_125.json").write_text(json.dumps(
        {"checkpoint": "x", "metrics": {"auc": 0.92, "acc": 0.84, "mae": 3.8}}
    ))

    # Group 1 is fold 0's validation set, group 2 is fold 1's -> fold 1 TRAINED
    # on group 1 and vice versa.
    folds_json = _make_folds_json(tmp_path / "folds.json", group_to_fold={1: 0, 2: 1})

    zoo = tmp_path / "zoo" / "v1"
    rc = PUB.main([
        "--out", str(zoo), "--runs-root", str(runs_root), "--heldout-dir", str(heldout),
        "--folds-json", str(folds_json), "--default-fold", "1", "--created", "2026-07-21",
    ])
    assert rc == 0

    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2), n_files=3, length=256, seed=0)
    table_csv = tmp_path / "table.csv"
    make_synth_table(npz_root).to_csv(table_csv, index=False)
    return {"zoo": zoo, "npz_root": npz_root, "table": table_csv, "folds_json": folds_json, "tmp": tmp_path}


# --------------------------------------------------------------------------- #
# publish_phase3_zoo
# --------------------------------------------------------------------------- #

def test_publish_writes_manifest_with_verifiable_hashes(published):
    manifest = load_manifest(published["zoo"])
    assert manifest["default_fold"] == 1
    assert sorted(manifest["folds"]) == ["0", "1"]
    for fold, block in manifest["folds"].items():
        ckpt = published["zoo"] / f"fold{fold}" / "best_auc.pt"
        assert sha256_file(ckpt) == block["sha256"]
    # Held-out metrics are attached to the fold that was actually scored.
    assert manifest["folds"]["1"]["heldout"]["auc"] == pytest.approx(0.92)
    assert manifest["folds"]["0"]["heldout"] is None


def test_publish_records_the_groups_each_fold_trained_on(published):
    manifest = load_manifest(published["zoo"])
    # group_to_fold {1: 0, 2: 1}: fold 1 validates on group 2, trains on group 1.
    assert manifest["folds"]["1"]["val_groups"] == [2]
    assert manifest["folds"]["1"]["train_groups"] == [1]


def test_publish_freezes_the_tree_read_only(published):
    ckpt = published["zoo"] / "fold1" / "best_auc.pt"
    assert not os.access(ckpt, os.W_OK)


def test_publish_refuses_to_overwrite_a_published_version(published, tmp_path):
    with pytest.raises(SystemExit, match="immutable"):
        PUB.main([
            "--out", str(published["zoo"]), "--runs-root", str(tmp_path / "runs"),
            "--folds-json", str(published["folds_json"]), "--default-fold", "1",
        ])


def test_publish_rejects_ambiguous_folds(tmp_path):
    runs_root = tmp_path / "runs"
    _make_run(runs_root, "a" * 32, fold=0)
    _make_run(runs_root, "c" * 32, fold=0)
    folds_json = _make_folds_json(tmp_path / "folds.json", group_to_fold={1: 0})
    with pytest.raises(SystemExit, match="more than one complete run"):
        PUB.main([
            "--out", str(tmp_path / "zoo"), "--runs-root", str(runs_root),
            "--folds-json", str(folds_json), "--default-fold", "0",
        ])


def test_pinning_a_run_resolves_an_ambiguous_fold(tmp_path):
    """The GL case: fold 0 has two complete runs, --run picks the right one."""
    runs_root = tmp_path / "runs"
    _make_run(runs_root, "a" * 32, fold=0)
    _make_run(runs_root, "c" * 32, fold=0)
    folds_json = _make_folds_json(tmp_path / "folds.json", group_to_fold={1: 0})
    zoo = tmp_path / "zoo"

    rc = PUB.main([
        "--out", str(zoo), "--runs-root", str(runs_root),
        "--folds-json", str(folds_json), "--default-fold", "0",
        "--run", f"0:{'c' * 32}", "--created", "2026-07-21",
    ])
    assert rc == 0
    assert load_manifest(zoo)["folds"]["0"]["run_id"] == "c" * 32


# --------------------------------------------------------------------------- #
# predict_phase3
# --------------------------------------------------------------------------- #

def _run_cli(published, out, *extra):
    return PP.main([
        "--out", str(out), "--zoo", str(published["zoo"]),
        "--table", str(published["table"]), "--npz-root", str(published["npz_root"]),
        "--folds-json", str(published["folds_json"]), "--batch-size", "8", *extra,
    ])


def test_cli_scores_files_with_the_default_fold(published, tmp_path):
    out = tmp_path / "preds.csv"
    assert _run_cli(published, out) == 0

    preds = pd.read_csv(out)
    assert list(preds.columns) == ["file", "prob", "n_windows", "dataset", "label", "in_train_split"]
    assert preds["prob"].between(0, 1).all()
    assert len(preds) == preds["file"].nunique()


def test_cli_writes_provenance_naming_the_exact_weights(published, tmp_path):
    out = tmp_path / "preds.csv"
    _run_cli(published, out)

    prov = json.loads((tmp_path / "preds.csv.provenance.json").read_text())
    manifest = load_manifest(published["zoo"])
    assert prov["fold"] == 1
    assert prov["checkpoint_sha256"] == manifest["folds"]["1"]["sha256"]
    assert prov["integrity_verified"] is True
    assert prov["n_files"] == len(pd.read_csv(out))


def test_cli_flags_files_the_checkpoint_trained_on(published, tmp_path, capsys):
    out = tmp_path / "preds.csv"
    _run_cli(published, out)

    preds = pd.read_csv(out)
    # Fold 1 trained on group 1 (datasets 1 and its noise mirrors) and validated
    # on group 2, so exactly the group-1 rows must be flagged.
    flagged = set(preds.loc[preds["in_train_split"], "dataset"])
    assert flagged and 1 in flagged and 2 not in flagged
    assert "TRAINING split" in capsys.readouterr().err


def test_cli_fold_override_changes_the_overlap_flags(published, tmp_path):
    out = tmp_path / "preds_fold0.csv"
    _run_cli(published, out, "--fold", "0")

    preds = pd.read_csv(out)
    # Fold 0 is the mirror image of fold 1: it trained on group 2, not group 1.
    flagged = set(preds.loc[preds["in_train_split"], "dataset"])
    assert 2 in flagged and 1 not in flagged
    assert json.loads((tmp_path / "preds_fold0.csv.provenance.json").read_text())["fold"] == 0


def test_cli_rejects_a_tampered_checkpoint(published, tmp_path):
    ckpt = published["zoo"] / "fold1" / "best_auc.pt"
    ckpt.chmod(0o644)
    ckpt.write_bytes(b"not the published weights")
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        _run_cli(published, tmp_path / "preds.csv")


def test_cli_says_when_a_dataset_is_absent_from_the_table(published, tmp_path):
    """The real GL failure: --dataset 123 against a table covering 1-120.

    The message must point at the table, not the NPZ tree -- blaming signals
    sends people to check the wrong thing entirely.
    """
    with pytest.raises(SystemExit, match=r"not in .*table.*covers datasets") as exc:
        _run_cli(published, tmp_path / "preds.csv", "--dataset", "999")
    assert "999" in str(exc.value)
    assert "NPZ" not in str(exc.value)


def test_cli_uses_the_zoo_env_var_when_no_flag_is_given(published, tmp_path, monkeypatch):
    monkeypatch.setenv(ZOO_ENV_VAR, str(published["zoo"]))
    out = tmp_path / "preds.csv"
    rc = PP.main([
        "--out", str(out), "--table", str(published["table"]),
        "--npz-root", str(published["npz_root"]),
        "--folds-json", str(published["folds_json"]), "--batch-size", "8",
    ])
    assert rc == 0 and out.is_file()


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #

def test_login_node_detection(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert PP.on_login_node("gl-login1.arc-ts.umich.edu") is True
    assert PP.on_login_node("gl1502") is False
    # Inside an allocation the hostname no longer matters.
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert PP.on_login_node("gl-login1.arc-ts.umich.edu") is False


def test_login_node_guard_blocks_large_runs(published, tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(PP, "on_login_node", lambda *a, **k: True)
    monkeypatch.setattr(PP, "LOGIN_NODE_FILE_LIMIT", 1)
    with pytest.raises(SystemExit, match="refusing to score"):
        _run_cli(published, tmp_path / "preds.csv")


def test_login_node_guard_can_be_overridden(published, tmp_path, monkeypatch):
    monkeypatch.setattr(PP, "on_login_node", lambda *a, **k: True)
    monkeypatch.setattr(PP, "LOGIN_NODE_FILE_LIMIT", 1)
    assert _run_cli(published, tmp_path / "preds.csv", "--allow-login-node") == 0


# --------------------------------------------------------------------------- #
# annotate_train_overlap: the inversion that makes the flag correct
# --------------------------------------------------------------------------- #

def test_dev_groups_are_never_counted_as_trained_on():
    preds = pd.DataFrame({"file": ["f1", "f2"], "prob": [0.5, 0.5],
                          "n_windows": [1, 1], "dataset": [1, 2], "label": [0, 1]})
    # Group 1 is a dev group: held out of EVERY fold, so it is never training data
    # even though group_to_fold marks it as fold 0's validation set.
    folds = {"group_to_fold": {"1": 0, "2": 1}, "dev_groups": [1]}
    annotated, n = annotate_train_overlap(preds, folds, fold=1)
    assert n == 0
    assert not annotated["in_train_split"].any()


def test_unknown_groups_are_treated_as_unseen():
    preds = pd.DataFrame({"file": ["f1"], "prob": [0.5],
                          "n_windows": [1], "dataset": [130], "label": [1]})
    # Datasets 121-155 post-date the split and are absent from group_to_fold.
    folds = {"group_to_fold": {"1": 0, "2": 1}, "dev_groups": []}
    annotated, n = annotate_train_overlap(preds, folds, fold=0)
    assert n == 0 and not annotated["in_train_split"].any()
