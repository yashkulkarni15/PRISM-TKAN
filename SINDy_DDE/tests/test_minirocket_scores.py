"""Per-file score export + noise-injection scoring for the MiniROCKET runner.

These outputs feed the TKAN-vs-MiniROCKET noise comparison
(``pri2m.xai.model_compare``), which only means anything if

* both models write the SAME per-file schema (column order included),
* the injected-noise rows are deterministic (same file, level, seed -> same
  noise in every process), and
* level 0 of the injection path is exactly the real-data path on the same
  windows -- otherwise the dose-response curve has an offset at its origin
  that is an artefact of the code, not of noise.

All runs use the aeon-free fake transform the runner already accepts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "scripts"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import minirocket_phase3 as MR  # noqa: E402
from pri2m.xai.model_compare import SCORE_COLUMNS  # noqa: E402
from tests._phase3_synth import (  # noqa: E402
    make_synth_folds,
    make_synth_npz_tree,
    make_synth_table,
)
from tests.test_minirocket_baseline import _FakeTransform  # noqa: E402

CAL = {"summary": {"rel_sigma_per_unit": [0.02, 0.02, 0.03, 0.03]}}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mrscores")
    root = make_synth_npz_tree(tmp / "npz", clean_ids=(1, 2, 3, 4, 5), n_files=4,
                               length=4096, seed=0)
    table = make_synth_table(root)
    folds = make_synth_folds(table, n_splits=5, seed=0)
    table_csv = tmp / "table.csv"
    table.to_csv(table_csv, index=False)
    folds_json = tmp / "folds.json"
    folds_json.write_text(json.dumps(folds))
    held = make_synth_npz_tree(tmp / "held", clean_ids=(7,), with_mirrors=False,
                               n_files=6, length=4096, seed=9)
    held_csv = tmp / "held.csv"
    ht = make_synth_table(held)
    ht["noise_std"] = 5.0
    ht.to_csv(held_csv, index=False)
    cal = tmp / "cal.json"
    cal.write_text(json.dumps(CAL))
    return {"tmp": tmp, "root": root, "table": table_csv, "folds": folds_json,
            "held": held, "held_csv": held_csv, "cal": cal}


def _argv(s, out, extra=()):
    return ["--fold", "0", "--table", str(s["table"]), "--folds-json", str(s["folds"]),
            "--npz-root", str(s["root"]), "--features-csv", str(s["held_csv"]),
            "--heldout-npz-root", str(s["held"]), "--n-kernels", "168",
            "--output-dir", str(out), "--tag", "t", "--eval-chunk-files", "3",
            "--alphas", "1,100", *extra]


def _score_flags(s, scores_dir, model_path=None, levels="0,5"):
    flags = ["--save-scores", "--scores-dir", str(scores_dir),
             "--noise-levels", levels, "--noise-calibration", str(s["cal"]),
             "--noise-files", "clean", "--noise-max-windows-per-file", "3",
             "--noise-seed", "0"]
    if model_path is not None:
        flags += ["--save-model", str(model_path)]
    return flags


@pytest.fixture(scope="module")
def scored_run(synth):
    out = synth["tmp"] / "out_scores"
    sd = synth["tmp"] / "scores"
    mp = synth["tmp"] / "models" / "mr_fold0.joblib"
    assert MR.main(_argv(synth, out, _score_flags(synth, sd, mp)),
                   transformer_factory=_FakeTransform) == 0
    return {"out": out, "scores": sd, "model": mp}


@pytest.fixture(scope="module")
def plain_run(synth):
    out = synth["tmp"] / "out_plain"
    assert MR.main(_argv(synth, out), transformer_factory=_FakeTransform) == 0
    return out


def _read(sd, model, split):
    return pd.read_csv(sd / f"scores_{model}_fold0_{split}.csv")


def test_score_csvs_have_exact_schema_and_sidecars(scored_run):
    sd = scored_run["scores"]
    for model in ("minirocket_A", "minirocket_B"):
        for split in ("val", "heldout", "val_noise"):
            csv = sd / f"scores_{model}_fold0_{split}.csv"
            df = pd.read_csv(csv)
            assert list(df.columns) == list(SCORE_COLUMNS)
            assert set(df["split"]) == {split}
            assert set(df["model"]) == {model}
            assert set(df["fold"]) == {0}
            side = json.loads(csv.with_suffix(".json").read_text())
            assert side["complete"] is True
            assert side["model"] == model and side["split"] == split
            assert "provenance" in side and "auc" in side
    # No delay variants are exported.
    assert not list(sd.glob("scores_minirocket_B_zero*"))


def test_val_and_heldout_rows_match_manifest(scored_run):
    sd = scored_run["scores"]
    man = json.loads((scored_run["out"] / "minirocket_t.json").read_text())
    for arm in ("A", "B"):
        for split in ("val", "heldout"):
            df = _read(sd, f"minirocket_{arm}", split)
            assert (df["level"] == 0.0).all()
            assert len(df) == man["metrics"][split][arm]["n_files"]
            side = json.loads((sd / f"scores_minirocket_{arm}_fold0_{split}.json").read_text())
            assert side["auc"]["0"] == pytest.approx(man["metrics"][split][arm]["auc"])
            assert side["acc"]["0"] == pytest.approx(man["metrics"][split][arm]["acc"])
    val = _read(sd, "minirocket_A", "val")
    assert set(val["noise_tier"].dropna().astype(int)) <= {0, 1, 5}
    assert (val["noise_tier"] == val["dataset"].map(
        lambda d: 0 if d <= 60 else (1 if d <= 90 else 5))).all()
    held = _read(sd, "minirocket_A", "heldout")
    assert (held["noise_tier"] == 5.0).all()
    assert held["b_over_blim"].notna().any() and held["omega"].notna().all()


def test_val_noise_rows_cover_clean_val_files_at_each_level(scored_run):
    sd = scored_run["scores"]
    val = _read(sd, "minirocket_A", "val")
    vn = _read(sd, "minirocket_A", "val_noise")
    clean = set(val.loc[val["noise_tier"] == 0, "file"])
    assert clean and set(vn["file"]) == clean
    assert sorted(vn["level"].unique()) == [0.0, 5.0]
    assert len(vn) == 2 * len(clean)
    assert (vn["noise_tier"] == 0).all()
    assert (vn["n_windows"] <= 3).all()
    # Noise actually changes the scores.
    piv = vn.pivot(index="file", columns="level", values="score")
    assert not np.allclose(piv[0.0], piv[5.0])


def test_level_zero_equals_real_scores_on_same_windows(scored_run, synth):
    import joblib

    from pri2m.data.phase3_sequence import (
        WindowConfig, iter_windows, load_npz_signals, tau_samples,
    )
    from pri2m.models.minirocket_baseline import build_inputs, select_train_windows

    bundle = joblib.load(scored_run["model"])
    norm, models = bundle["norm"], bundle["models"]
    cfg = WindowConfig(length=1024, stride=512)
    sd = scored_run["scores"]
    for arm in ("A", "B"):
        vn = _read(sd, f"minirocket_{arm}", "val_noise")
        vn0 = vn[vn["level"] == 0.0]
        for _, r in vn0.iterrows():
            stem = Path(r["file"]).stem
            sig, meta = load_npz_signals(
                Path(synth["root"]) / f"dataset_{int(r['dataset'])}" / f"{stem}.npz",
                cfg.channels)
            w = iter_windows(sig, cfg).astype(np.float32)
            idx = select_train_windows(w.shape[0], 3)
            x = build_inputs(norm.apply(w[idx]),
                             np.full(len(idx), tau_samples(meta["fs"], meta["tpf"])),
                             pool=4, arm=arm)
            ref = float(np.mean(models[arm].decision_function(x)))
            assert r["score"] == pytest.approx(ref, rel=1e-9, abs=1e-12)
            assert int(r["n_windows"]) == len(idx)


def test_noise_rows_are_deterministic(scored_run, synth):
    out = synth["tmp"] / "out_scores2"
    sd2 = synth["tmp"] / "scores2"
    assert MR.main(_argv(synth, out, _score_flags(synth, sd2)),
                   transformer_factory=_FakeTransform) == 0
    for arm in ("A", "B"):
        a = _read(scored_run["scores"], f"minirocket_{arm}", "val_noise")
        b = _read(sd2, f"minirocket_{arm}", "val_noise")
        pd.testing.assert_frame_equal(a, b)


def test_flags_off_manifest_unchanged(scored_run, plain_run):
    a = json.loads((plain_run / "minirocket_t.json").read_text())
    b = json.loads((scored_run["out"] / "minirocket_t.json").read_text())
    for m in (a, b):
        m["provenance"].pop("wall_seconds")
    b.pop("scores_outputs")
    assert a == b
    assert "scores_outputs" not in json.loads((plain_run / "minirocket_t.json").read_text())


def test_flags_off_writes_no_score_files(plain_run):
    assert sorted(p.name for p in plain_run.iterdir()) == ["minirocket_t.json"]


def test_saved_model_reproduces_val_scores(scored_run):
    import joblib

    bundle = joblib.load(scored_run["model"])
    assert set(bundle["models"]) == {"A", "B"}
    assert bundle["fold"] == 0 and "provenance" in bundle


def test_refuses_calibration_without_rel_sigma(synth, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"summary": {"rel_sigma_per_unit": None,
                                           "undecided_reason": "x"}}))
    flags = ["--noise-levels", "0,5", "--noise-calibration", str(bad),
             "--scores-dir", str(tmp_path / "s")]
    with pytest.raises(SystemExit):
        MR.main(_argv(synth, tmp_path / "o", flags), transformer_factory=_FakeTransform)
    assert not (tmp_path / "o").exists()


def test_noise_levels_require_calibration(synth, tmp_path):
    with pytest.raises(SystemExit):
        MR.main(_argv(synth, tmp_path / "o", ["--noise-levels", "0,5"]),
                transformer_factory=_FakeTransform)


def test_aeon_scores_and_model_dump_roundtrip(synth, tmp_path):
    """Real aeon transform: the dump must reload and reproduce the val scores."""
    pytest.importorskip("aeon")
    import joblib

    from pri2m.data.phase3_sequence import (
        WindowConfig, iter_windows, load_npz_signals, tau_samples,
    )
    from pri2m.models.minirocket_baseline import build_inputs

    sd, mp = tmp_path / "s", tmp_path / "m.joblib"
    assert MR.main(_argv(synth, tmp_path / "o", _score_flags(synth, sd, mp, levels="0")),
                   transformer_factory=None) == 0
    bundle = joblib.load(mp)
    cfg = WindowConfig()
    val = _read(sd, "minirocket_B", "val").head(3)
    for _, r in val.iterrows():
        sig, meta = load_npz_signals(Path(synth["root"]) / f"dataset_{int(r['dataset'])}"
                                     / f"{Path(r['file']).stem}.npz", cfg.channels)
        w = iter_windows(sig, cfg).astype(np.float32)
        x = build_inputs(bundle["norm"].apply(w),
                         np.full(len(w), tau_samples(meta["fs"], meta["tpf"])), arm="B")
        ref = float(np.mean(bundle["models"]["B"].decision_function(x)))
        assert r["score"] == pytest.approx(ref, rel=1e-6, abs=1e-9)
