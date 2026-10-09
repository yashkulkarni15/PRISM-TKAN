"""Tests for the TKAN per-file scorer ``scripts/score_files_phase3.py``.

* Real-data scoring (no ``--noise-levels``) must reproduce
  ``eval_heldout_phase3.run``'s file-level AUC on the same input, for the
  held-out path and for the in-fold val path (val rows handed to ``run`` as a
  features CSV).
* The CSV schema is exact (shared with the MiniROCKET runner).
* Injected-noise rows are deterministic across two runs, level 0 equals the
  real-data scores on the same windows, and level > 0 actually moves them.
* A calibration without ``rel_sigma_per_unit`` is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from scripts.eval_heldout_phase3 import run as eval_run
from scripts.score_files_phase3 import CSV_COLUMNS, main
from tests._phase3_synth import make_synth_folds, make_synth_npz_tree, make_synth_table
from tests.test_eval_heldout import _heldout_inputs, _smoke_train_checkpoint


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("score_files")
    ckpt, pre = _smoke_train_checkpoint(tmp, side_input=True)
    # The same table/folds the checkpoint was trained on (same builders + seed).
    train_root = tmp / "train_npz"
    table = make_synth_table(train_root)
    folds = make_synth_folds(table, n_splits=3, seed=0)
    table_csv = tmp / "table.csv"
    table.to_csv(table_csv, index=False)
    folds_json = tmp / "folds.json"
    folds_json.write_text(json.dumps(folds))
    held_root, features_csv = _heldout_inputs(tmp)
    feats = pd.read_csv(features_csv)
    feats["noise_std"] = 5.0  # the real features CSV's "Noise std (%)" column
    feats.to_csv(features_csv, index=False)
    cal = tmp / "cal.json"
    cal.write_text(json.dumps({"summary": {"rel_sigma_per_unit": [0.05] * 4},
                               "complete": True}))
    return {"tmp": tmp, "ckpt": ckpt, "pre": pre, "table": table, "table_csv": table_csv,
            "folds": folds, "folds_json": folds_json, "train_root": train_root,
            "held_root": held_root, "features_csv": features_csv, "cal": cal}


def _read(csv: Path) -> pd.DataFrame:
    return pd.read_csv(csv, keep_default_na=False, na_values=[""])


def test_heldout_level0_reproduces_eval_heldout_auc(trained, tmp_path):
    t = trained
    ref = eval_run(checkpoint=str(t["ckpt"]), features_csv=str(t["features_csv"]),
                   npz_root=str(t["held_root"]), out=str(tmp_path / "ref.json"),
                   device="cpu", batch_size=16)
    out = tmp_path / "scores_tkan_fold0_heldout.csv"
    rc = main(["--checkpoint", str(t["ckpt"]), "--npz-root", str(t["held_root"]),
               "--split", "heldout", "--features-csv", str(t["features_csv"]),
               "--batch-size", "7", "--model-name", "tkan", "--fold-label", "0",
               "--output-csv", str(out)])
    assert rc == 0
    df = _read(out)
    assert list(df.columns) == CSV_COLUMNS
    assert (df["split"] == "heldout").all() and (df["level"] == 0.0).all()
    assert len(df) == ref["n_eval_files"]
    assert int(df["n_windows"].sum()) == ref["n_eval_windows"]
    auc = roc_auc_score(df["label"], 1 / (1 + np.exp(-df["score"])))
    assert auc == ref["metrics"]["auc"]
    side = json.loads(out.with_suffix(".json").read_text())
    assert side["complete"] is True
    assert side["auc"]["heldout"]["0.0"]["auc"] == ref["metrics"]["auc"]
    assert side["auc"]["heldout"]["0.0"]["acc"] == pytest.approx(ref["metrics"]["acc"])
    # The features CSV's noise_std becomes noise_tier.
    assert (df["noise_tier"] == 5).all()


def test_val_level0_reproduces_eval_on_val_rows(trained, tmp_path):
    t = trained
    pre = json.loads(Path(t["pre"]).read_text())
    fold = int(pre["fold"])
    g2f = {int(k): int(v) for k, v in t["folds"]["group_to_fold"].items()}
    val = t["table"][t["table"]["parent_group"].astype(int).map(g2f) == fold]
    val_csv = tmp_path / "val_rows.csv"
    val.to_csv(val_csv, index=False)
    ref = eval_run(checkpoint=str(t["ckpt"]), features_csv=str(val_csv),
                   npz_root=str(t["train_root"]), out=str(tmp_path / "ref.json"),
                   device="cpu", batch_size=16)

    out = tmp_path / "scores_tkan_fold0_val.csv"
    assert main(["--checkpoint", str(t["ckpt"]), "--npz-root", str(t["train_root"]),
                 "--split", "val", "--table", str(t["table_csv"]),
                 "--folds-json", str(t["folds_json"]), "--output-csv", str(out)]) == 0
    df = _read(out)
    assert list(df.columns) == CSV_COLUMNS
    assert len(df) == ref["n_eval_files"] == len(val)
    assert set(df["dataset"]) == set(val["dataset"])
    assert (df["fold"] == fold).all()
    auc = roc_auc_score(df["label"], 1 / (1 + np.exp(-df["score"])))
    assert auc == ref["metrics"]["auc"]
    # noise_tier from the 1-120 id layout.
    tiers = dict(zip(df["dataset"], df["noise_tier"]))
    for d, tier in tiers.items():
        assert tier == (0 if d <= 60 else 1 if d <= 90 else 5)
    # b_over_blim present where the table has it, blank (NaN) where it is NaN.
    assert df["b_over_blim"].isna().sum() == int(val["b_over_blim"].isna().sum())


def _noise_args(t, out, levels="0,1,5"):
    return ["--checkpoint", str(t["ckpt"]), "--npz-root", str(t["train_root"]),
            "--split", "val", "--table", str(t["table_csv"]),
            "--folds-json", str(t["folds_json"]), "--noise-levels", levels,
            "--noise-calibration", str(t["cal"]), "--noise-files", "clean",
            "--noise-max-windows-per-file", "2", "--noise-seed", "0",
            "--output-csv", str(out)]


def test_noise_rows_deterministic_and_level0_matches_real(trained, tmp_path):
    t = trained
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    assert main(_noise_args(t, a)) == 0
    assert main(_noise_args(t, b)) == 0
    da, db = _read(a), _read(b)
    assert list(da.columns) == CSV_COLUMNS
    pd.testing.assert_frame_equal(da, db)
    assert (da["split"] == "val_noise").all()
    assert sorted(da["level"].unique().tolist()) == [0.0, 1.0, 5.0]
    # Only clean (tier-0) files; one row per (file, level).
    assert (da["noise_tier"] == 0).all()
    assert not da.duplicated(["file", "level"]).any()
    assert (da["n_windows"] <= 2).all()
    # Noise actually moves the scores.
    s0 = da[da["level"] == 0].set_index("file")["score"]
    s5 = da[da["level"] == 5].set_index("file")["score"]
    assert not np.allclose(s0.values, s5.loc[s0.index].values)

    side = json.loads(a.with_suffix(".json").read_text())
    assert side["complete"] is True
    assert side["noise"]["rel_sigma_per_unit"] == [0.05] * 4
    assert set(side["auc"]["val_noise"]) == {"0.0", "1.0", "5.0"}


def test_level0_noise_equals_real_scoring_on_same_windows(trained, tmp_path):
    """With max windows >= every file's count, level 0 == the real-data score."""
    t = trained
    noisy = tmp_path / "n.csv"
    args = _noise_args(t, noisy, levels="0")
    args[args.index("--noise-max-windows-per-file") + 1] = "1000"
    assert main(args) == 0
    real = tmp_path / "r.csv"
    assert main(["--checkpoint", str(t["ckpt"]), "--npz-root", str(t["train_root"]),
                 "--split", "val", "--table", str(t["table_csv"]),
                 "--folds-json", str(t["folds_json"]), "--output-csv", str(real)]) == 0
    dn = _read(noisy).set_index("file")
    dr = _read(real).set_index("file").loc[dn.index]
    np.testing.assert_array_equal(dn["score"].values, dr["score"].values)
    np.testing.assert_array_equal(dn["n_windows"].values, dr["n_windows"].values)


def test_refuses_undecided_calibration(trained, tmp_path):
    t = trained
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"summary": {"rel_sigma_per_unit": None,
                                           "undecided_reason": "mirror_difference_not_additive"}}))
    args = _noise_args(t, tmp_path / "x.csv")
    args[args.index("--noise-calibration") + 1] = str(bad)
    with pytest.raises(SystemExit, match="mirror_difference_not_additive"):
        main(args)
    assert not (tmp_path / "x.csv").exists()


def test_side_off_heldout_reproduces_eval_heldout_auc(tmp_path):
    ckpt, _ = _smoke_train_checkpoint(tmp_path, side_input=False)
    held_root, features_csv = _heldout_inputs(tmp_path)  # no noise_std column
    ref = eval_run(checkpoint=str(ckpt), features_csv=str(features_csv),
                   npz_root=str(held_root), out=str(tmp_path / "ref.json"),
                   device="cpu", batch_size=16)
    out = tmp_path / "scores_tkan_side_off_fold0_heldout.csv"
    assert main(["--checkpoint", str(ckpt), "--npz-root", str(held_root),
                 "--split", "heldout", "--features-csv", str(features_csv),
                 "--model-name", "tkan_side_off", "--output-csv", str(out)]) == 0
    df = _read(out)
    assert list(df.columns) == CSV_COLUMNS
    assert (df["model"] == "tkan_side_off").all()
    assert df["noise_tier"].isna().all()  # blank without a noise_std column
    auc = roc_auc_score(df["label"], 1 / (1 + np.exp(-df["score"])))
    assert auc == ref["metrics"]["auc"]
    assert json.loads(out.with_suffix(".json").read_text())["side_input"] is False
