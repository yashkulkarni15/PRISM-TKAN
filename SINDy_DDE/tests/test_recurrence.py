"""Exp 2: does the causally-necessary delay cluster recur across seeds, by FUNCTION?

Edge IDs ``(kan, out, in)`` cannot be compared across independently trained
networks: ``out`` indexes a hidden unit, and hidden units are permutation
symmetric. These tests pin the two ways round that -- curve-shape matching on the
comparable ``in`` channel, and Hungarian alignment of hidden units before any ID
overlap -- plus the null logic that turns them into a verdict, the submitter for
the seed retrain, and the full runner against a real (smoke-trained) checkpoint.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pri2m.xai import recurrence as R

REPO_ROOT = Path(__file__).resolve().parents[1]


def _edge(kan, out, i, feature="delay:xtpos", importance=1.0):
    return {"kan": kan, "out": out, "in": i, "feature": feature, "importance": importance}


# --------------------------------------------------------------------------- #
# channel mix / lookup
# --------------------------------------------------------------------------- #

def test_cluster_channel_mix_counts_features():
    edges = [_edge("k", 0, 68, "delay:xtpos"), _edge("k", 1, 68, "delay:xtpos"),
             _edge("k", 2, 69, "delay:ytpos")]
    assert R.cluster_channel_mix(edges) == {"delay:xtpos": 2, "delay:ytpos": 1}
    assert R.cluster_channel_mix([]) == {}


def test_curve_lookup_maps_edge_ids_to_rows():
    bundle = {"records": [{"kan": "a", "out": 3, "in": 68}, {"kan": "b", "out": 0, "in": 69}],
              "curves": np.zeros((2, 4))}
    lut = R.curve_lookup(bundle)
    assert lut == {("a", 3, 68): 0, ("b", 0, 69): 1}


def test_curve_lookup_rejects_duplicate_ids():
    bundle = {"records": [{"kan": "a", "out": 3, "in": 68}, {"kan": "a", "out": 3, "in": 68}],
              "curves": np.zeros((2, 4))}
    with pytest.raises(ValueError):
        R.curve_lookup(bundle)


# --------------------------------------------------------------------------- #
# centred_unit
# --------------------------------------------------------------------------- #

def test_centred_unit_centres_and_normalises():
    c = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 5.0, 0.0, 5.0]])
    u, n_deg = R.centred_unit(c)
    assert n_deg == 0
    np.testing.assert_allclose(u.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0)


def test_centred_unit_flat_curve_is_nan_not_zero():
    """A flat curve has no shape. Scoring it 0 would pull S toward 'no match'
    with a number nobody measured; it must be NaN and counted."""
    c = np.array([[2.0, 2.0, 2.0], [1.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
    u, n_deg = R.centred_unit(c)
    assert n_deg == 2
    assert np.isnan(u[0]).all() and np.isnan(u[2]).all()
    assert np.isfinite(u[1]).all()


# --------------------------------------------------------------------------- #
# best_match_cosine / recurrence_statistic
# --------------------------------------------------------------------------- #

def _curves(rng, n, p=16):
    return rng.standard_normal((n, p))


def test_identical_curves_give_cosine_one():
    rng = np.random.default_rng(0)
    a = _curves(rng, 5)
    feats = ["x", "x", "y", "y", "y"]
    out = R.best_match_cosine(a, feats, a.copy(), feats)
    np.testing.assert_allclose(out["best_cosine"], 1.0)
    np.testing.assert_allclose(out["best_abs_cosine"], 1.0)
    assert out["n_unmatched"] == 0


def test_sign_flipped_curves_are_minus_one_signed_one_abs():
    """The comb's sign is learned, so a sign flip is a DIFFERENT function."""
    rng = np.random.default_rng(1)
    a = _curves(rng, 1)
    out = R.best_match_cosine(a, ["x"], -a, ["x"])
    np.testing.assert_allclose(out["best_cosine"], [-1.0])
    np.testing.assert_allclose(out["best_abs_cosine"], [1.0])


def test_same_input_restricts_candidates_to_the_same_channel():
    rng = np.random.default_rng(2)
    a = _curves(rng, 1)
    b = np.vstack([a, _curves(rng, 1)])
    # The perfect match sits on a different channel: same_input must ignore it.
    restricted = R.best_match_cosine(a, ["x"], b, ["y", "x"], same_input=True)
    free = R.best_match_cosine(a, ["x"], b, ["y", "x"], same_input=False)
    assert restricted["best_cosine"][0] < 0.999
    np.testing.assert_allclose(free["best_cosine"], [1.0])


def test_no_candidate_on_channel_is_nan_and_counted():
    rng = np.random.default_rng(3)
    a = _curves(rng, 2)
    b = _curves(rng, 2)
    out = R.best_match_cosine(a, ["x", "z"], b, ["x", "x"])
    assert np.isfinite(out["best_cosine"][0])
    assert np.isnan(out["best_cosine"][1])
    assert out["n_unmatched"] == 1


def test_degenerate_b_curve_is_never_a_candidate():
    rng = np.random.default_rng(4)
    a = _curves(rng, 1)
    b = np.vstack([np.ones((1, a.shape[1])), _curves(rng, 1)])
    out = R.best_match_cosine(a, ["x"], b, ["x", "x"])
    assert out["n_degenerate_b"] == 1
    assert np.isfinite(out["best_cosine"][0])


def test_recurrence_statistic_is_nanmean_with_count():
    rng = np.random.default_rng(5)
    a = _curves(rng, 3)
    st = R.recurrence_statistic(a, ["x", "x", "q"], a.copy(), ["x", "x", "x"])
    assert st["n_used"] == 2
    assert st["n_unmatched"] == 1
    assert st["S"] == pytest.approx(1.0)
    assert st["S_abs"] == pytest.approx(1.0)


def test_recurrence_statistic_nothing_measurable_is_none():
    rng = np.random.default_rng(6)
    st = R.recurrence_statistic(_curves(rng, 2), ["x", "x"], _curves(rng, 1), ["y"])
    assert st["S"] is None and st["n_used"] == 0
    empty = R.recurrence_statistic(np.zeros((0, 4)), [], _curves(rng, 1), ["y"])
    assert empty["S"] is None


# --------------------------------------------------------------------------- #
# recurrence_verdict
# --------------------------------------------------------------------------- #

def test_verdict_recurs_when_above_null_and_random():
    null = list(np.linspace(0.1, 0.5, 20))
    v = R.recurrence_verdict(0.9, 0.3, null)
    assert v["recurs"] is True
    assert v["null_p95"] == pytest.approx(np.percentile(null, 95))
    assert v["undecided_reason"] is None


def test_verdict_fails_below_null_p95():
    null = list(np.linspace(0.1, 0.95, 20))
    assert R.recurrence_verdict(0.9, 0.3, null)["recurs"] is False


def test_verdict_fails_when_random_matches_as_well():
    null = list(np.linspace(0.1, 0.5, 20))
    assert R.recurrence_verdict(0.9, 0.9, null)["recurs"] is False


@pytest.mark.parametrize("S,S_random,null,reason", [
    (None, 0.3, [0.1] * 20, "S"),
    (0.9, None, [0.1] * 20, "S_random"),
    (0.9, 0.3, [0.1] * 19, "null"),
    (0.9, 0.3, [0.1] * 19 + [None], "null"),
    (0.9, 0.3, None, "null"),
])
def test_verdict_undecided_never_false(S, S_random, null, reason):
    v = R.recurrence_verdict(S, S_random, null)
    assert v["recurs"] is None
    assert reason in v["undecided_reason"]


# --------------------------------------------------------------------------- #
# Hungarian alignment
# --------------------------------------------------------------------------- #

def test_alignment_recovers_a_permutation():
    rng = np.random.default_rng(7)
    H = 12
    Ha = rng.standard_normal((500, H))
    p = rng.permutation(H)
    Hb = np.empty_like(Ha)
    Hb[:, p] = Ha            # unit j of A is unit p[j] of B
    Hb[:, p[0]] *= -1.0      # an anti-correlated unit still matches on |corr|
    al = R.align_hidden_units(Ha, Hb)
    np.testing.assert_array_equal(al["perm"], p)
    np.testing.assert_allclose(al["matched_abs_corr"], 1.0, atol=1e-12)
    assert al["constant_a"] == [] and al["constant_b"] == []


def test_alignment_flags_constant_units_as_zero_corr():
    rng = np.random.default_rng(8)
    Ha = rng.standard_normal((200, 3))
    Hb = Ha.copy()
    Hb[:, 1] = 0.5
    al = R.align_hidden_units(Ha, Hb)
    assert al["constant_b"] == [1]
    assert np.isfinite(al["matched_abs_corr"]).all()
    assert al["matched_abs_corr"][1] == 0.0


def test_alignment_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        R.align_hidden_units(np.zeros((10, 3)), np.zeros((10, 4)))


def test_aligned_overlap_full_under_true_permutation():
    p = np.array([2, 0, 3, 1])
    ea = [_edge("kz", 0, 68), _edge("kz", 1, 69), _edge("kr", 3, 68)]
    eb = [_edge(e["kan"], int(p[e["out"]]), e["in"]) for e in ea]
    assert R.aligned_id_overlap(ea, eb, p) == 3
    # Raw ID overlap is what the invalid comparison counted; here it is 0.
    assert R.aligned_id_overlap(ea, eb, np.arange(4)) == 0


def test_overlap_null_expected_and_p():
    H = 8
    p = np.arange(H)[::-1].copy()
    ea = [_edge("kz", o, 68) for o in range(4)]
    eb = [_edge("kz", int(p[o]), 68) for o in range(4)]
    obs = R.aligned_id_overlap(ea, eb, p)
    nul = R.overlap_null(ea, eb, H, n_perm=2000, seed=0, observed=obs)
    # 4 A edges, each hits B's 4-of-8 units on the same (kan,in) with prob 1/2.
    assert nul["expected"] == pytest.approx(2.0, abs=0.15)
    assert 0 < nul["p"] < 0.05
    again = R.overlap_null(ea, eb, H, n_perm=2000, seed=0, observed=obs)
    assert again["p"] == nul["p"] and again["expected"] == nul["expected"]
    assert R.overlap_null(ea, eb, H, n_perm=10, seed=0)["p"] is None


def test_edge_curves_from_model_matches_bundle():
    """Direct read-out must equal what extract_stratified_curves stored."""
    import torch
    from pri2m.models.kan import KANLinear

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k = KANLinear(5, 3)

    torch.manual_seed(0)
    m = M()
    edges = [_edge("k", 2, 4), _edge("k", 0, 1)]
    got = R.edge_curves_from_model(m, edges, n=32)
    full = m.k.edge_function(32, component="spline")["curves"]
    np.testing.assert_allclose(got[0], full[2, 4], rtol=1e-6)
    np.testing.assert_allclose(got[1], full[0, 1], rtol=1e-6)


# --------------------------------------------------------------------------- #
# submit_seed_retrain.sh / phase3_recurrence.sbatch
# --------------------------------------------------------------------------- #

def test_shell_scripts_parse():
    for s in ("slurm/submit_seed_retrain.sh", "slurm/phase3_recurrence.sbatch"):
        res = subprocess.run(["bash", "-n", s], cwd=REPO_ROOT, capture_output=True, text=True)
        assert res.returncode == 0, f"{s}: {res.stderr}"


def test_submit_seed_retrain_dry_run_lists_six_jobs():
    env = {**os.environ, "DRY_RUN": "1"}
    res = subprocess.run(["bash", "slurm/submit_seed_retrain.sh"], cwd=REPO_ROOT,
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    lines = [ln for ln in res.stdout.splitlines() if "sbatch" in ln]
    assert len(lines) == 6
    for fold in (0, 2):
        for seed in (1, 2, 3):
            hit = [ln for ln in lines if f"--array={fold} " in ln and f"--seed\\ {seed}" in ln]
            assert len(hit) == 1, (fold, seed, lines)
    for ln in lines:
        assert "DETERMINISTIC=1" in ln and "--time=16:00:00" in ln
        assert "results/phase3/runs_seeds" in ln and "--export=ALL" in ln
        assert ln.rstrip().endswith("slurm/phase3_train_gpu.sbatch")


def test_submit_seed_retrain_passes_env_per_call_not_from_shell(tmp_path):
    """Stale exported EXTRA_ARGS in the caller's shell must not reach any job."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    fake = bindir / "sbatch"
    fake.write_text(
        "#!/bin/bash\n"
        f'echo "$DETERMINISTIC|$EXTRA_ARGS|$OUTPUT_DIR|$*" >> "{log}"\n'
        'echo "Submitted batch job 1"\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}",
           "PROJECT_ROOT": str(tmp_path), "EXTRA_ARGS": "--bogus",
           "DETERMINISTIC": "0"}
    res = subprocess.run(["bash", str(REPO_ROOT / "slurm/submit_seed_retrain.sh")],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 6
    seeds = sorted(c.split("|")[1] for c in calls)
    assert seeds == ["--seed 1"] * 2 + ["--seed 2"] * 2 + ["--seed 3"] * 2
    assert all(c.startswith("1|") for c in calls)
    assert all(c.split("|")[2] == f"{tmp_path}/results/phase3/runs_seeds" for c in calls)


def test_recurrence_sbatch_asserts_complete():
    src = (REPO_ROOT / "slurm/phase3_recurrence.sbatch").read_text()
    assert "#SBATCH --partition=standard" in src
    assert "read -r -a" in src
    assert '"complete"' in src or "complete" in src
    assert "61c67bddbdbd49ba9727ff92f4814c86" in src
    assert "a654129d2cdc460093e21373f31ee5a5" in src


# --------------------------------------------------------------------------- #
# Runner end to end, on a real smoke-trained TKAN
# --------------------------------------------------------------------------- #

from tests._phase3_synth import make_synth_folds, make_synth_npz_tree, make_synth_table  # noqa: E402

X_DELAY = "encoder.L0.x_delay"


def _permute_hidden_state_dict(sd, p, C, H):
    """B = A with its layer-0 hidden units relabelled: unit j of A -> unit p[j] of B.

    Every tensor that indexes a layer-0 hidden unit moves together: the out rows
    of kan_{z,r,h}, their h_prev input columns [C, C+H), and the severity head's
    context input columns [0, H). The delayed-channel input columns do not move,
    which is exactly why ``in`` is comparable across networks and ``out`` is not.
    """
    import torch

    p_t = torch.as_tensor(p, dtype=torch.long)
    new = {k: v.clone() for k, v in sd.items()}
    for g in ("kan_z", "kan_r", "kan_h"):
        for w in ("base_weight", "spline_weight"):
            k = f"encoder.layers.0.{g}.{w}"
            t = sd[k]
            rows = torch.empty_like(t)
            rows[p_t] = t
            cols = rows.clone()
            cols[:, C + p_t] = rows[:, C:C + H]
            new[k] = cols
    for k, t in sd.items():
        if k.startswith("severity.") and k.rsplit(".", 1)[-1] in ("base_weight", "spline_weight", "weight"):
            out = t.clone()
            out[:, p_t] = t[:, :H]
            new[k] = out
    return new


@pytest.fixture(scope="module")
def trained_pair(tmp_path_factory):
    import torch

    tmp = tmp_path_factory.mktemp("recurrence")
    npz_root = make_synth_npz_tree(tmp / "npz", clean_ids=(1, 2, 3), n_files=4, length=256, seed=0)
    table = make_synth_table(npz_root)
    folds = make_synth_folds(table, n_splits=3, seed=0)
    table_csv = tmp / "table.csv"
    table.to_csv(table_csv, index=False)
    folds_json = tmp / "folds.json"
    folds_json.write_text(json.dumps(folds))
    out_dir = tmp / "runs"
    res = subprocess.run(
        [sys.executable, "scripts/train_phase3.py", "--model", "tkan", "--fold", "0",
         "--npz-root", str(npz_root), "--table", str(table_csv),
         "--folds-json", str(folds_json), "--output-dir", str(out_dir),
         "--mlflow-uri", f"file:{tmp / 'mlruns'}", "--smoke", "--seed", "4"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr[-4000:]
    ckpts = sorted(out_dir.rglob("*.pt"))
    ckpt_a = next((c for c in ckpts if c.name == "best_auc.pt"), ckpts[0])
    run_a = ckpt_a.parent.parent
    summary = json.loads((run_a / "summary.json").read_text())
    assert summary["config"]["seed"] == 4  # --seed is recorded in summary.json

    sd = torch.load(ckpt_a, map_location="cpu", weights_only=True)
    H = int(sd["encoder.layers.0.kan_z.base_weight"].shape[0])
    # Layer-0 gate input is [x_t (C), h_prev (H), x_delay (C)].
    C = (int(sd["encoder.layers.0.kan_z.base_weight"].shape[1]) - H) // 2
    p = np.random.default_rng(11).permutation(H)
    run_b = tmp / "runs_b" / "permuted"
    (run_b / "checkpoints").mkdir(parents=True)
    shutil.copy(run_a / "preprocessing.json", run_b / "preprocessing.json")
    ckpt_b = run_b / "checkpoints" / "best_auc.pt"
    torch.save(_permute_hidden_state_dict(sd, p, C, H), ckpt_b)
    return {"tmp": tmp, "npz_root": npz_root, "table_csv": table_csv,
            "ckpt_a": ckpt_a, "ckpt_b": ckpt_b, "p": p, "H": H, "C": C}


def _load(ckpt):
    import argparse
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import xai_phase3
    pre = json.loads((Path(ckpt).parent.parent / "preprocessing.json").read_text())
    m = xai_phase3._load_model(str(ckpt), argparse.Namespace(
        input_size=None, hidden_size=None, grid_size=5, spline_order=3), pre)
    m.eval()
    return m, pre


def _batch(pre, tp):
    from pri2m.xai.sampling import build_labeled_batch, window_config_from_preprocessing
    batch, _ = build_labeled_batch(str(tp["table_csv"]), str(tp["npz_root"]), pre, 8,
                                   window_config_from_preprocessing(pre), seed=0)
    return batch


def test_permuted_network_is_functionally_identical(trained_pair):
    """Guards the test construction itself: if B is not A relabelled, every
    assertion in the runner test below is about the wrong thing."""
    import torch
    ma, pre = _load(trained_pair["ckpt_a"])
    mb, _ = _load(trained_pair["ckpt_b"])
    batch = _batch(pre, trained_pair)
    with torch.no_grad():
        oa, ob = ma(batch), mb(batch)
    torch.testing.assert_close(oa["chatter_logit"], ob["chatter_logit"], rtol=1e-4, atol=1e-5)


def _write_manifest(path, cluster, random_edges, claimable=True):
    path.write_text(json.dumps({"modes": [
        {"stratum": X_DELAY, "mode": 0, "n_edges": len(cluster),
         "verdict": {"claimable": claimable},
         "edges": {"mode_edges": cluster, "random_matched": random_edges,
                   "other_modes_matched": []}},
        {"stratum": X_DELAY, "mode": 1, "n_edges": 1,
         "verdict": {"claimable": False},
         "edges": {"mode_edges": random_edges[:1], "random_matched": [],
                   "other_modes_matched": []}},
    ], "complete": True}))


def test_runner_permuted_network_recurs(trained_pair):
    from pri2m.xai import activation_modes as AM

    tp = trained_pair
    ma, pre = _load(tp["ckpt_a"])
    bundle = AM.fold_bundles(ma, _batch(pre, tp), n=32, strata=[X_DELAY])[X_DELAY]
    order = np.argsort(-bundle["importance"], kind="stable")
    spec = lambda i: {k: bundle["records"][int(i)][k] for k in ("kan", "out", "in", "feature", "importance")}  # noqa: E731
    cluster_a = [spec(i) for i in order[:6]]
    random_a = [spec(i) for i in order[-6:]]
    p = tp["p"]
    relabel = lambda es: [{**e, "out": int(p[e["out"]])} for e in es]  # noqa: E731

    man = tp["tmp"] / "manifests"
    man.mkdir(exist_ok=True)
    _write_manifest(man / "a.json", cluster_a, random_a)
    _write_manifest(man / "b.json", relabel(cluster_a), relabel(random_a))
    _write_manifest(man / "c.json", cluster_a, random_a, claimable=False)

    out = tp["tmp"] / "recur_out"
    res = subprocess.run(
        [sys.executable, "scripts/recurrence_phase3.py",
         "--network", f"f0s0={tp['ckpt_a']}={man / 'a.json'}",
         "--network", f"f0s1={tp['ckpt_b']}={man / 'b.json'}",
         "--network", f"f1s0={tp['ckpt_a']}={man / 'c.json'}",
         "--table", str(tp["table_csv"]), "--npz-root", str(tp["npz_root"]),
         "--n-windows", "8", "--curve-points", "32", "--n-untrained", "20",
         "--n-perm", "500", "--output-dir", str(out), "--tag", "smoke"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert res.returncode == 0, f"{res.stderr[-4000:]}\n{res.stdout[-2000:]}"
    m = json.loads((out / "recurrence_smoke.json").read_text())
    assert m["complete"] is True
    assert m["provenance"]["n_untrained"] == 20

    nets = m["networks"]
    assert nets["f0s0"]["n_cluster_edges"] == 6
    assert nets["f0s0"]["channel_mix"] == R.cluster_channel_mix(cluster_a)
    assert nets["f1s0"]["n_cluster_edges"] == 0

    pairs = {(q["a"], q["b"]): q for q in m["pairs"]}
    q = pairs[("f0s0", "f0s1")]
    assert q["cross_fold"] is False
    for d in ("a_to_b", "b_to_a"):
        assert q[d]["S"] == pytest.approx(1.0, abs=1e-5)
        assert q[d]["verdict"]["recurs"] is True, q[d]
        assert len(q[d]["S_untrained"]) == 20
    assert q["recurs"] is True
    assert q["alignment"]["aligned_overlap"] == 6
    assert q["alignment"]["n_a"] == 6
    assert q["alignment"]["null"]["p"] < 0.05

    for other in (("f0s0", "f1s0"), ("f0s1", "f1s0")):
        c = pairs[other]
        assert c["cross_fold"] is True
        assert c["recurs"] is None
        assert c["undecided_reason"] == "no_necessary_cluster:f1s0"

    s0 = m["per_fold"]["0"]
    assert s0["n_pairs"] == 1 and s0["n_recur"] == 1
    assert "1 of 1" in s0["summary"]

    npz = np.load(out / "recurrence_smoke_curves.npz", allow_pickle=False)
    assert npz["f0s0__cluster_curves"].shape == (6, 32)
    np.testing.assert_allclose(npz["f0s0__cluster_curves"],
                               bundle["curves"][order[:6]], rtol=1e-5)


# --------------------------------------------------------------------------- #
# --exclude-mode: a judgement exclusion is applied AND recorded
# --------------------------------------------------------------------------- #
def _manifest_two_claimable():
    def mode(m, claim):
        e = [{"kan": "encoder.layers.0.kan_z", "out": m, "in": 68,
              "feature": "delay:xtpos", "importance": 1.0}]
        return {"stratum": "encoder.L0.x_delay", "mode": m,
                "verdict": {"claimable": claim},
                "edges": {"mode_edges": e, "random_matched": e}}
    return {"modes": [mode(0, True), mode(1, False), mode(2, True)]}


def test_necessary_cluster_exclusion_is_reported_not_silent():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import recurrence_phase3 as RP
    full = RP.necessary_cluster(_manifest_two_claimable())
    assert full["claimable_modes"] == [0, 2] and full["excluded_modes"] == []
    cut = RP.necessary_cluster(_manifest_two_claimable(), exclude=[0])
    assert cut["claimable_modes"] == [2]
    assert cut["excluded_modes"] == [0]
    assert [e["out"] for e in cut["cluster"]] == [2]
