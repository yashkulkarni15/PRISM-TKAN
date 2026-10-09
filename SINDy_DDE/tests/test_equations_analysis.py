"""equations_phase3: the local analysis over a synthetic ``equations/`` directory."""
import hashlib
import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import equations_phase3 as eqs  # noqa: E402
from pri2m.xai.equation_terms import TERM_NAMES  # noqa: E402

FIGS = ["eq_curve_gallery", "eq_unexplained_ecdf", "eq_family_grid",
        "eq_coef_heatmap", "eq_fidelity", "eq_diff_vs_sum"]
NETS = {"f0s0": 3, "f0s1": 3, "f2s0": 3, "f2s1": 0}  # name -> n cluster edges (f2s1: no cluster)
FEATS = ["x_delay:xtpos", "x_delay:Forcex", "x_delay:Forcex"]


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _split(rng, n_files, per, tag):
    """Per-window arrays for ``n_files`` files; label constant per file; terms carry the label."""
    files = np.repeat([f"{tag}_{i}" for i in range(n_files)], per)
    flab = np.arange(n_files) % 2
    label = np.repeat(flab, per).astype(float)
    T = rng.normal(size=(len(files), 16))
    T[:, 1] += 2.0 * label
    T[:, 4] -= 1.0 * label
    group = np.repeat([f"g{i // 2}" for i in range(n_files)], per)
    return files, label, T, group


def _score(rng, T):
    return 1.5 * T[:, 1] - 0.8 * T[:, 4] + 0.1 * rng.normal(size=len(T))


@pytest.fixture
def tree(tmp_path):
    rng = np.random.default_rng(0)
    eq = tmp_path / "equations"
    eq.mkdir()
    mrdir = tmp_path / "minirocket"
    mrdir.mkdir()
    xs1 = np.linspace(-1, 1, 256)
    shapes = [lambda x: 2 * x + 0.1, lambda x: np.tanh(3 * x), lambda x: np.sin(4 * x)]
    for fold in (0, 2):
        vf, vl, vT, vg = _split(rng, 24, 5, f"v{fold}")
        hf, hl, hT, hg = _split(rng, 12, 5, f"h{fold}")
        wnpz = eq / f"windows_fold{fold}.npz"
        np.savez_compressed(
            wnpz, val_file=vf, val_label=vl, val_terms=vT, val_parent_group=vg,
            ho_file=hf, ho_label=hl, ho_terms=hT, ho_parent_group=hg,
            term_names=np.array(TERM_NAMES), range_lo=np.full(4, -1.0), range_hi=np.full(4, 1.0))
        (eq / f"windows_fold{fold}.json").write_text(json.dumps(
            {"sha256": _sha(wnpz), "fold": fold, "complete": True}))
        for seed in (0, 1):
            name = f"f{fold}s{seed}"
            n_e = NETS[name]
            ys = np.stack([shapes[i](xs1) * (1 + 0.1 * seed) + 0.01 * rng.normal(size=256)
                           for i in range(n_e)]) if n_e else np.zeros((0, 256))
            np.savez_compressed(
                eq / f"tkan_{name}.npz",
                edges=json.dumps([{"kan": "encoder.0", "out": 0, "in": i, "feature": FEATS[i]}
                                  for i in range(n_e)]),
                xs=np.tile(xs1, (n_e, 1)), ys=ys,
                untrained_ys=rng.normal(size=(20, n_e, 256)),
                val_score=_score(rng, vT), ho_score=_score(rng, hT), val_file=vf, ho_file=hf)
            (eq / f"tkan_{name}.json").write_text(json.dumps({
                "name": name, "fold": fold, "train_seed": seed, "n_edges": n_e,
                "checkpoint": f"/gl/run_{name}/checkpoints/best_auc.pt", "checkpoint_sha256": "ab",
                "windows_sha256": _sha(wnpz), "complete": True}))
            w = np.zeros(200)
            w[:6] = [5, -4, 3, 2, -1, 1]
            w[6:] = 0.01 * rng.normal(size=194)
            ho = _score(rng, hT)
            np.savez(eq / f"minirocket_fold{fold}_s{seed}.npz", weights=w,
                     val_score=_score(rng, vT), ho_score=ho,
                     ho_score_top20=ho + 0.05 * rng.normal(size=len(ho)),
                     val_file=vf, ho_file=hf)
            man = None
            if fold == 0 and seed == 0:
                man = 0.9123456789   # landed manifest and the GL re-run's own manifest agree
                for stem in ("minirocket_fold0", "minirocket_eq_fold0_s0"):
                    (mrdir / f"{stem}.json").write_text(json.dumps(
                        {"metrics": {"heldout": {"B": {"auc": man}}}}))
            (eq / f"minirocket_fold{fold}_s{seed}.json").write_text(json.dumps({
                "fold": fold, "seed": seed, "arm": "B", "window_sha256": _sha(wnpz),
                "manifest_heldout_auc": man, "norm_max_abs_diff": 0.0, "top_k": 20,
                "complete": True}))
    # f0s0's checkpoint passes the eval-path check, so it is not flagged; f2s0 stays flagged.
    (eq / "evalcheck_f0.json").write_text(json.dumps({"tag": "f0", "complete": True, "checkpoints": [{
        "checkpoint": "/gl/run_f0s0/checkpoints/best_auc.pt", "checkpoint_name": "best_auc.pt",
        "recorded_auc_matches_checkpoint": True, "recorded_val_auc": 0.96,
        "trainer_load_trainer_eval_auc": 0.96, "xai_load_trainer_eval_auc": 0.955,
        "score_files_auc": 0.958}]}))
    # recurrence manifest: the checkpoints the networks were analysed from
    (tmp_path / "recurrence.json").write_text(json.dumps({"networks": {
        n: {"checkpoint": f"/gl/run_{n}/checkpoints/best_auc.pt"} for n in NETS}}))
    run = tmp_path / "results" / "runs" / "run_f0s1"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"best_auc_val": {"auc": 0.91}}))
    return tmp_path


def _run(tree, *extra, allow_missing=True):
    out = tree / "out"
    argv = ["--equations-dir", str(tree / "equations"), "--figures-dir", str(out / "figs"),
            "--summary-md", str(out / "summary.md"),
            "--minirocket-manifests", str(tree / "minirocket"),
            "--recurrence-manifest", str(tree / "recurrence.json"),
            "--runs-root", str(tree / "results"), "--n-boot", "6", *extra]
    if allow_missing:       # the fixture has 4 of the 8 TKAN networks and 2 of 4 seeds per fold
        argv.append("--allow-missing")
    return eqs.main(argv), out


def _manifest(tree):
    return json.loads((tree / "equations" / "equations_manifest.json").read_text())


def test_manifest_blocks_figures_and_summary(tree):
    rc, out = _run(tree)
    assert rc == 0
    m = json.loads((tree / "equations" / "equations_manifest.json").read_text())
    for block in ("provenance", "checks", "curves", "formulas_native", "distilled", "contrast"):
        assert block in m
    assert m["complete"] is True
    for f in FIGS:
        p = out / "figs" / f"{f}.png"
        assert p.exists() and p.stat().st_size > 0, f
    frac = m["curves"]["frac_trained"]
    assert frac is not None
    assert f"{frac:.2f}" in (out / "summary.md").read_text()


def test_seed0_flag_cleared_only_by_evalcheck(tree):
    _run(tree)
    m = json.loads((tree / "equations" / "equations_manifest.json").read_text())
    flags = {n: v["flagged"] for n, v in m["checks"]["tkan"].items()}
    assert flags["f0s0"] is False and flags["f2s0"] is True
    assert flags["f0s1"] is False
    assert "f2s1" in m["curves"]["no_cluster"]
    assert m["checks"]["minirocket_seed0"]["0"]["status"] == "ok"
    assert m["distilled"]["stability"]["tkan"]["0"]["null_seed"] == 1


def test_mismatched_val_file_exits_2_naming_both_files(tree, capsys):
    p = tree / "equations" / "tkan_f0s1.npz"
    z = dict(np.load(p))
    z["val_file"] = z["val_file"][::-1].copy()
    np.savez_compressed(p, **z)
    rc, _ = _run(tree)
    err = capsys.readouterr().err
    assert rc == 2
    assert str(p) in err and str(tree / "equations" / "windows_fold0.npz") in err


def test_missing_dumps_exit_2_unless_allowed(tree, capsys):
    rc, _ = _run(tree, allow_missing=False)
    assert rc == 2 and "tkan_f0s2" in capsys.readouterr().err
    rc, _ = _run(tree)
    assert rc == 0
    assert "tkan_f0s2" in _manifest(tree)["checks"]["missing"]


def test_npz_without_sidecar_exits_2_naming_the_file(tree, capsys):
    (tree / "equations" / "tkan_f0s1.json").unlink()
    rc, _ = _run(tree)
    assert rc == 2 and str(tree / "equations" / "tkan_f0s1.npz") in capsys.readouterr().err


def test_minirocket_seed0_rerun_must_equal_landed_manifest(tree, capsys):
    _run(tree)
    s0 = _manifest(tree)["checks"]["minirocket_seed0"]
    assert s0["0"]["status"] == "ok" and s0["2"]["status"] == "not_checked"
    (tree / "minirocket" / "minirocket_eq_fold0_s0.json").write_text(json.dumps(
        {"metrics": {"heldout": {"B": {"auc": 0.9123456789 + 1e-6}}}}))
    rc, _ = _run(tree)
    assert rc == 2 and "minirocket_eq_fold0_s0.json" in capsys.readouterr().err


def test_missing_rerun_manifest_is_not_checked_never_ok(tree):
    (tree / "minirocket" / "minirocket_eq_fold0_s0.json").unlink()
    _run(tree)
    assert _manifest(tree)["checks"]["minirocket_seed0"]["0"]["status"] == "not_checked"


def test_checkpoint_must_match_recurrence_manifest(tree, capsys):
    _run(tree)
    assert _manifest(tree)["checks"]["tkan"]["f0s1"]["recurrence_checkpoint"] == "ok"
    (tree / "recurrence.json").write_text(json.dumps(
        {"networks": {"f0s1": {"checkpoint": "/gl/elsewhere/checkpoints/best_auc.pt"}}}))
    rc, _ = _run(tree)
    assert rc == 2 and "recurrence" in capsys.readouterr().err


def test_recorded_val_auc_read_from_run_summary(tree):
    _run(tree)
    t = _manifest(tree)["checks"]["tkan"]
    assert t["f0s1"]["recorded_val_auc"] == 0.91 and t["f0s1"]["recorded_val_auc_source"] == "summary.json"
    assert t["f2s0"]["recorded_val_auc"] is None


def test_cross_model_uses_every_seed_pair_within_fold(tree):
    _run(tree)
    cm = _manifest(tree)["distilled"]["cross_model"]
    assert len(cm) == 8                                  # 2 folds x 2 TKAN seeds x 2 MiniROCKET seeds
    assert {(c["tkan_seed"], c["minirocket_seed"]) for c in cm} == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_untestable_pair_does_not_null_the_medians():
    rng = np.random.default_rng(1)

    def net(feats):
        n = len(feats)
        return {"fold": 0, "feats": feats, "xs": np.tile(np.linspace(-1, 1, 256), (n, 1)),
                "ys": np.cumsum(rng.normal(size=(n, 256)), 1), "null": rng.normal(size=(20, n, 256)),
                "fits": [{"name": None}] * n, "fams": [None] * n}

    shared = net(["x_delay:xtpos"])
    nets = {"f0s0": shared, "f0s1": {**shared, "null": rng.normal(size=(20, 1, 256))},
            "f0s2": net(["x_delay:Forcex"])}              # no channel in common with the others
    block, _, _ = eqs.curves_block(nets)
    assert block["frac_trained"] == 1.0                   # the f0s0 <-> f0s1 pairs only
    assert block["within_fold"]["n_pairs"] == 2 and block["within_fold"]["n_untestable"] == 4


def test_cleared_needs_complete_file_and_non_null_aucs():
    e = {"recorded_auc_matches_checkpoint": True, "recorded_val_auc": 0.96,
         "trainer_load_trainer_eval_auc": 0.96, "xai_load_trainer_eval_auc": 0.96, "score_files_auc": 0.96}
    assert eqs._cleared({"complete": True}, e) is True
    assert eqs._cleared({"complete": False}, e) is False
    assert eqs._cleared({"complete": True}, {**e, "score_files_auc": None}) is False
    assert eqs._cleared({"complete": True}, {**e, "recorded_val_auc": None}) is False


def test_file_auc_rejects_mixed_labels_within_a_file():
    with pytest.raises(ValueError, match="one label per file"):
        eqs.file_auc(np.arange(4.0), np.array(["a", "a", "b", "b"]), np.array([0, 1, 0, 1]))


def test_dump_npz_sha256_is_verified_when_present(tree, capsys):
    side = tree / "equations" / "tkan_f0s1.json"
    meta = json.loads(side.read_text())
    meta["npz_sha256"] = _sha(tree / "equations" / "tkan_f0s1.npz")
    side.write_text(json.dumps(meta))
    rc, _ = _run(tree)
    assert rc == 0
    meta["npz_sha256"] = "0" * 64
    side.write_text(json.dumps(meta))
    rc, _ = _run(tree)
    assert rc == 2 and "tkan_f0s1.npz" in capsys.readouterr().err
