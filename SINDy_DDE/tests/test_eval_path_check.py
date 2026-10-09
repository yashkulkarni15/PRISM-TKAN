"""J1 eval-path check: one checkpoint, two loaders, three scorers, one answer.

On a freshly trained synthetic fold the trainer's validation AUC, the same
checkpoint loaded the XAI way and evaluated by the trainer, and the standalone
per-file scorer must all agree (1e-9) and the two loaded models must produce the
same logits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_path_check_phase3 import main
from tests._phase3_synth import make_synth_folds, make_synth_table
from tests.test_eval_heldout import _smoke_train_checkpoint


@pytest.mark.parametrize("side_input", [False, True])
def test_three_eval_paths_agree(tmp_path, side_input):
    ckpt, _ = _smoke_train_checkpoint(tmp_path, side_input=side_input)
    # Same table/folds the checkpoint was trained on (same builders + seed).
    train_root = tmp_path / "train_npz"
    table = make_synth_table(train_root)
    table_csv = tmp_path / "table.csv"
    table.to_csv(table_csv, index=False)
    folds_json = tmp_path / "folds.json"
    folds_json.write_text(json.dumps(make_synth_folds(table, n_splits=3, seed=0)))

    out = tmp_path / "equations"
    assert main(["--checkpoint", str(ckpt), "--table", str(table_csv),
                 "--folds-json", str(folds_json), "--npz-root", str(train_root),
                 "--device", "cpu", "--output-dir", str(out), "--tag", "t"]) == 0

    doc = json.loads((out / "evalcheck_t.json").read_text())
    assert doc["complete"] is True
    assert (out / "evalcheck_scores_t_0.csv").is_file()   # tag in the name: concurrent jobs can't collide
    (r,) = doc["checkpoints"]
    assert r["checkpoint"] == str(ckpt)
    assert r["trainer_load_trainer_eval_auc"] == pytest.approx(r["xai_load_trainer_eval_auc"], abs=1e-9)
    assert r["trainer_load_trainer_eval_auc"] == pytest.approx(r["score_files_auc"], abs=1e-9)
    assert r["trainer_load_trainer_eval_auc"] == pytest.approx(r["recorded_val_auc"], abs=1e-9)
    assert r["max_abs_logit_diff"] < 1e-5
    assert r["state_dict_prefixes"] == sorted(r["state_dict_prefixes"]) and r["state_dict_prefixes"]


def test_train_config_accepts_legacy_seed0_summary():
    # Literal `config` of results/phase3/runs/61c67bdd.../summary.json (the 0.960
    # run): predates severity_mode / severity_quantiles / deterministic.
    from scripts.eval_path_check_phase3 import _train_config
    legacy = {"model": "tkan", "epochs": 20, "lr": 0.0015, "batch_size": 256, "seed": 0,
              "window": {"length": 1024, "stride": 512,
                         "channels": ["xtpos", "ytpos", "Forcex", "Forcey"]},
              "hidden_size": 64, "num_layers": 1, "grid_size": 5, "spline_order": 3,
              "loss": {"w_sev": 1.0, "w_reg": 1.0, "w_cls": 1.0, "w_traj": 0.0},
              "fold": 0, "with_trajectory": False, "traj_steps": 16, "pool": 4,
              "smoke": False, "device": "cuda", "num_workers": 4, "compile": True,
              "lr_schedule": "cosine", "warmup_epochs": 2, "grad_clip": 1.0,
              "optimizer": "adamw", "weight_decay": 0.0001, "side_input": True,
              "side_hidden": 16, "some_future_key": 1}
    cfg = _train_config(legacy)
    assert cfg.pool == 4 and cfg.side_input is True


def test_recorded_auc_only_for_best_auc_checkpoint(tmp_path):
    ckpt, _ = _smoke_train_checkpoint(tmp_path, side_input=False)
    other = ckpt.with_name("final.pt")
    other.write_bytes(ckpt.read_bytes())
    train_root = tmp_path / "train_npz"
    table = make_synth_table(train_root)
    table.to_csv(tmp_path / "table.csv", index=False)
    (tmp_path / "folds.json").write_text(json.dumps(make_synth_folds(table, n_splits=3, seed=0)))
    out = tmp_path / "eq"
    assert main(["--checkpoint", str(ckpt), "--checkpoint", str(other),
                 "--table", str(tmp_path / "table.csv"), "--folds-json", str(tmp_path / "folds.json"),
                 "--npz-root", str(train_root), "--output-dir", str(out), "--tag", "t"]) == 0
    a, b = json.loads((out / "evalcheck_t.json").read_text())["checkpoints"]
    assert (a["checkpoint_name"], a["recorded_auc_matches_checkpoint"]) == ("best_auc.pt", True)
    assert a["recorded_val_auc"] is not None
    assert (b["checkpoint_name"], b["recorded_auc_matches_checkpoint"]) == ("final.pt", False)
    assert b["recorded_val_auc"] is None and b["recorded_note"]
