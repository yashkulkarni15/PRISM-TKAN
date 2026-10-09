"""Tests for the equations-across-seeds GL runners (J2 windows, ...).

Shared fixture helpers live at the top; append new runner tests below.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pri2m.data.phase3_sequence import NormStats
from pri2m.xai.equation_terms import TERM_NAMES
from scripts.equation_windows_phase3 import main as windows_main
from tests._phase3_synth import make_synth_folds, make_synth_npz_tree, make_synth_table

WINDOW_LEN = 64  # synth files are 256 samples long -> 4 windows per file


def make_equation_inputs(tmp: Path, fold: int = 0) -> dict:
    """Synthetic tree + table/folds/features CSV/preprocessing for the J2 runners.

    Datasets 1-3 (+ mirrors) are the labelled table; dataset 7 sits in the SAME
    npz root (as on GL) but only in the held-out features CSV.
    """
    npz_root = make_synth_npz_tree(tmp / "npz", clean_ids=(1, 2, 3, 7), n_files=4,
                                   length=256, seed=0)
    full = make_synth_table(npz_root)
    table = full[full["dataset"] != 7]
    held = full[full["dataset"] == 7]
    folds = make_synth_folds(table, n_splits=3, seed=0)
    paths = {k: tmp / f"{k}.csv" for k in ("table", "features")}
    table.to_csv(paths["table"], index=False)
    held.to_csv(paths["features"], index=False)
    folds_json = tmp / "folds.json"
    folds_json.write_text(json.dumps(folds))
    norm = NormStats(mean=np.zeros(4), std=np.ones(4)).to_dict()
    pre = tmp / "preprocessing.json"
    pre.write_text(json.dumps({
        "fold": fold, "norm": norm, "channels": ["xtpos", "ytpos", "Forcex", "Forcey"],
        "window": {"length": WINDOW_LEN, "stride": WINDOW_LEN},
    }))
    return {"npz_root": npz_root, "table": paths["table"], "features": paths["features"],
            "folds_json": folds_json, "preprocessing": pre, "fold": fold}


def test_windows_runner_end_to_end(tmp_path):
    t = make_equation_inputs(tmp_path)
    out = tmp_path / "equations"
    rc = windows_main([
        "--preprocessing", str(t["preprocessing"]), "--table", str(t["table"]),
        "--folds-json", str(t["folds_json"]), "--features-csv", str(t["features"]),
        "--npz-root", str(t["npz_root"]), "--fold", str(t["fold"]),
        "--n-val", "16", "--n-heldout", "8", "--n-train-range", "16",
        "--seed", "0", "--output-dir", str(out),
    ])
    assert rc == 0
    npz_path = out / f"windows_fold{t['fold']}.npz"
    z = np.load(npz_path)  # no allow_pickle: strings must be unicode, not object
    for p, n in (("val", 16), ("ho", 8)):
        assert z[f"{p}_window"].shape == (n, WINDOW_LEN, 4)
        assert z[f"{p}_window"].dtype == np.float32
        assert z[f"{p}_terms"].shape == (n, 16)
        for k in ("tau", "tpf", "fs", "omega", "label", "file", "dataset", "parent_group"):
            assert z[f"{p}_{k}"].shape == (n,), (p, k)
        assert np.isfinite(z[f"{p}_terms"]).all()
    assert list(z["term_names"]) == TERM_NAMES
    assert z["range_lo"].shape == z["range_hi"].shape == (4,)
    assert (z["range_lo"] < z["range_hi"]).all()
    # held-out windows come from dataset 7 only; val windows never do
    assert set(z["ho_dataset"].tolist()) == {"7"}
    assert "7" not in set(z["val_dataset"].tolist())

    side = json.loads((out / f"windows_fold{t['fold']}.json").read_text())
    assert side["complete"] is True
    assert side["sha256"] == hashlib.sha256(npz_path.read_bytes()).hexdigest()
    assert side["preprocessing"] == str(t["preprocessing"])


# --------------------------------------------------------------------------- #
# Task 6 (J4): MiniROCKET --window-file scoring
# --------------------------------------------------------------------------- #
import sys  # noqa: E402

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import minirocket_phase3 as MR  # noqa: E402


class _WideFake:
    """aeon-free transform, 32 features on 8 channels (> 20, so top-20 is a real cut)."""

    def fit(self, X):
        return self

    def transform(self, X):
        return np.concatenate([X.mean(-1), X.std(-1), X.max(-1), X.min(-1)],
                              axis=1).astype(np.float32)


def _mr_window_npz(path, n_val=12, n_ho=7, seed=3):
    rng = np.random.default_rng(seed)
    d = {"term_names": np.asarray([f"t{i}" for i in range(16)]),
         "range_lo": np.zeros(4), "range_hi": np.ones(4)}
    for p, n in (("val", n_val), ("ho", n_ho)):
        d[f"{p}_window"] = rng.standard_normal((n, 1024, 4)).astype(np.float32)
        d[f"{p}_tau"] = rng.integers(20, 200, n)
        for k in ("tpf", "fs", "omega", "label", "parent_group", "dataset"):
            d[f"{p}_{k}"] = rng.random(n)
        d[f"{p}_file"] = np.asarray([f"{p}_f{i}.npz" for i in range(n)])
        d[f"{p}_terms"] = rng.standard_normal((n, 16))
    np.savez(path, **d)


def _mr_setup(tmp_path, norm_shift=0.0, **side_over):
    """Synthetic fold + window npz/sidecar whose preprocessing norm is the fold's own
    train norm (+ ``norm_shift``). Returns (argv builder, window npz path)."""
    root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2, 3, 4, 5), n_files=4,
                               length=4096, seed=0)
    table = make_synth_table(root)
    table.to_csv(tmp_path / "table.csv", index=False)
    folds = make_synth_folds(table, 5, 0)
    (tmp_path / "folds.json").write_text(json.dumps(folds))
    train_t, _ = MR.split_fold(table, folds, 0)
    norm, _ = MR.collect_train(train_t, root, MR.WindowConfig(), 8)
    nd = norm.to_dict()
    nd["mean"] = [m + norm_shift for m in nd["mean"]]
    (tmp_path / "pre.json").write_text(json.dumps({"norm": nd}))
    wf = tmp_path / "windows_fold0.npz"
    _mr_window_npz(wf)
    side = {"sha256": hashlib.sha256(wf.read_bytes()).hexdigest(), "fold": 0,
            "preprocessing": str(tmp_path / "pre.json"), "complete": True, **side_over}
    side = {k: v for k, v in side.items() if v is not None}
    wf.with_suffix(".json").write_text(json.dumps(side))

    def argv(*extra, arm="B"):
        return ["--fold", "0", "--arm", arm, "--table", str(tmp_path / "table.csv"),
                "--folds-json", str(tmp_path / "folds.json"), "--npz-root", str(root),
                "--n-kernels", "168", "--output-dir", str(tmp_path / "out"), "--tag", "eq_t",
                "--alphas", "1,100", "--window-file", str(wf),
                "--equations-dir", str(tmp_path / "equations"), *extra]
    return argv, wf


def test_minirocket_window_scores(tmp_path):
    argv, wf = _mr_setup(tmp_path)
    eq = tmp_path / "equations"
    with pytest.raises(SystemExit) as e:  # arm A has no delayed channel
        MR.main(argv(arm="A"), transformer_factory=_WideFake)
    assert e.value.code == 2

    assert MR.main(argv(), transformer_factory=_WideFake) == 0
    z = np.load(eq / "minirocket_fold0_s0.npz")
    assert z["val_score"].shape == (12,) and z["ho_score"].shape == (7,)
    assert z["ho_score_top20"].shape == (7,)
    assert z["weights"].shape == (32,)
    assert list(z["val_file"]) == [f"val_f{i}.npz" for i in range(12)]
    assert len(z["ho_file"]) == 7
    assert np.isfinite(z["val_score"]).all()
    assert not np.allclose(z["ho_score_top20"], z["ho_score"])  # p=32 > 20
    side = json.loads((eq / "minirocket_fold0_s0.json").read_text())
    assert side["complete"] is True and side["arm"] == "B"
    assert side["fold"] == 0 and side["seed"] == 0
    assert side["window_sha256"] == hashlib.sha256(wf.read_bytes()).hexdigest()
    assert side["npz_sha256"] == hashlib.sha256((eq / "minirocket_fold0_s0.npz").read_bytes()).hexdigest()
    assert "manifest_heldout_auc" in side
    assert side["norm_max_abs_diff"] < 1e-6


def test_minirocket_window_norm_mismatch_refuses(tmp_path):
    argv, _ = _mr_setup(tmp_path, norm_shift=0.5)
    with pytest.raises(SystemExit) as e:
        MR.main(argv(), transformer_factory=_WideFake)
    assert "norm differs" in str(e.value.code)
    assert not (tmp_path / "equations").exists()


def test_minirocket_window_sidecar_incomplete_exits_2(tmp_path):
    argv, _ = _mr_setup(tmp_path, complete=None)
    with pytest.raises(SystemExit) as e:
        MR.main(argv(), transformer_factory=_WideFake)
    assert e.value.code == 2


def test_minirocket_window_file_needs_eq_tag(tmp_path):
    argv, _ = _mr_setup(tmp_path)
    a = argv()
    i = a.index("--tag")
    for bad in (a[:i] + a[i + 2:], a[:i + 1] + ["fold0"] + a[i + 2:]):  # no tag; landed-style tag
        with pytest.raises(SystemExit) as e:
            MR.main(bad, transformer_factory=_WideFake)
        assert e.value.code == 2
    assert not (tmp_path / "equations").exists()


def test_minirocket_window_file_refuses_existing_manifest(tmp_path):
    argv, _ = _mr_setup(tmp_path)
    (tmp_path / "out").mkdir()
    landed = tmp_path / "out" / "minirocket_eq_t.json"
    landed.write_text("landed")
    with pytest.raises(SystemExit) as e:
        MR.main(argv(), transformer_factory=_WideFake)
    assert e.value.code == 2
    assert landed.read_text() == "landed"
