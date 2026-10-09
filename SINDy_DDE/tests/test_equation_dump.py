"""J3: the TKAN dump runner (scripts/equation_dump_phase3.py).

One smoke-trained TKAN + a hand-written mode-ablation manifest (two claimable
modes) + a tiny window file in the Task 4 format; the runner is run as a
subprocess, as on GL.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._phase3_synth import make_synth_folds, make_synth_npz_tree, make_synth_table

REPO_ROOT = Path(__file__).resolve().parents[1]
X_DELAY = "encoder.L0.x_delay"
CH = ("xtpos", "ytpos", "Forcex", "Forcey")
NV, NH, L = 12, 6, 64


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import argparse
    import torch  # noqa: F401

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import xai_phase3
    from pri2m.xai import activation_modes as AM
    from pri2m.xai.sampling import build_labeled_batch, window_config_from_preprocessing

    tmp = tmp_path_factory.mktemp("eqdump")
    npz_root = make_synth_npz_tree(tmp / "npz", clean_ids=(1, 2, 3), n_files=4, length=256, seed=0)
    table = make_synth_table(npz_root)
    folds = make_synth_folds(table, n_splits=3, seed=0)
    table_csv, folds_json = tmp / "table.csv", tmp / "folds.json"
    table.to_csv(table_csv, index=False)
    folds_json.write_text(json.dumps(folds))
    out_dir = tmp / "runs"
    res = subprocess.run(
        [sys.executable, "scripts/train_phase3.py", "--model", "tkan", "--fold", "0",
         "--npz-root", str(npz_root), "--table", str(table_csv),
         "--folds-json", str(folds_json), "--output-dir", str(out_dir),
         "--mlflow-uri", f"file:{tmp / 'mlruns'}", "--smoke", "--seed", "4"],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-4000:]
    ckpts = sorted(out_dir.rglob("*.pt"))
    ckpt = next((c for c in ckpts if c.name == "best_auc.pt"), ckpts[0])

    pre = json.loads((ckpt.parent.parent / "preprocessing.json").read_text())
    model = xai_phase3._load_model(str(ckpt), argparse.Namespace(
        input_size=None, hidden_size=None, grid_size=5, spline_order=3), pre)
    model.eval()
    batch, _ = build_labeled_batch(str(table_csv), str(npz_root), pre, 8,
                                   window_config_from_preprocessing(pre), seed=0)
    bundle = AM.fold_bundles(model, batch, n=32, strata=[X_DELAY])[X_DELAY]
    order = np.argsort(-bundle["importance"], kind="stable")
    spec = lambda i: {k: bundle["records"][int(i)][k]  # noqa: E731
                      for k in ("kan", "out", "in", "feature", "importance")}
    modes = [{"stratum": X_DELAY, "mode": m, "verdict": {"claimable": True},
              "edges": {"mode_edges": [spec(i) for i in order[3 * m:3 * m + 3]],
                        "random_matched": [], "other_modes_matched": []}}
             for m in (0, 1)]
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({"modes": modes, "complete": True}))

    rng = np.random.default_rng(0)
    def part(n):
        return {"window": rng.normal(size=(n, L, 4)).astype(np.float32),
                "tau": np.full(n, 8), "tpf": np.full(n, 200.0, np.float32),
                "fs": np.full(n, 1e4, np.float32), "omega": np.full(n, 3000.0, np.float32),
                "file": np.array([f"f{i}" for i in range(n)])}
    arrays = {f"{p}_{k}": v for p, n in (("val", NV), ("ho", NH)) for k, v in part(n).items()}
    arrays["range_lo"], arrays["range_hi"] = np.full(4, -2.0), np.full(4, 2.0)
    win = tmp / "windows_fold0.npz"
    np.savez_compressed(win, **arrays)
    side = tmp / "windows_fold0.json"
    side.write_text(json.dumps({"sha256": _sha(win), "fold": 0, "complete": True}))
    return {"tmp": tmp, "ckpt": ckpt, "manifest": manifest, "win": win, "side": side}


def _run(env, out, name="f0s4", extra=(), win=None):
    return subprocess.run(
        [sys.executable, "scripts/equation_dump_phase3.py",
         "--network", f"{name}={env['ckpt']}={env['manifest']}",
         "--windows", str(win or env["win"]), "--n-untrained", "3", "--batch-size", "5",
         "--output-dir", str(out), *extra],
        cwd=str(REPO_ROOT), capture_output=True, text=True)


def test_tkan_dump_end_to_end(env):
    out = env["tmp"] / "out"
    res = _run(env, out)
    assert res.returncode == 0, res.stderr[-3000:]
    z = np.load(out / "tkan_f0s4.npz", allow_pickle=False)
    edges = json.loads(str(z["edges"]))
    E = len(edges)
    assert E == 6
    assert z["xs"].shape == (E, 256) and z["ys"].shape == (E, 256)
    assert z["untrained_ys"].shape == (3, E, 256)
    assert not np.allclose(z["untrained_ys"], z["ys"][None])
    # edge i is swept over its own channel's range
    assert np.allclose(z["xs"][:, 0], -2.0) and np.allclose(z["xs"][:, -1], 2.0)
    assert z["val_score"].shape == (NV,) and z["ho_score"].shape == (NH,)
    assert np.isfinite(z["val_score"]).all() and np.isfinite(z["ho_score"]).all()
    assert list(z["val_file"]) == [f"f{i}" for i in range(NV)]
    s = json.loads((out / "tkan_f0s4.json").read_text())
    assert s["complete"] is True and s["n_edges"] == E and s["fold"] == 0
    assert s["checkpoint_sha256"] == _sha(env["ckpt"])
    assert s["windows_sha256"] == _sha(env["win"])
    assert s["npz_sha256"] == _sha(out / "tkan_f0s4.npz")
    assert s["exclusions"] == []


def test_tkan_dump_empty_cluster(env):
    out = env["tmp"] / "out_empty"
    res = _run(env, out, extra=("--exclude-mode", "f0s4:0", "--exclude-mode", "f0s4:1",
                                "--exclude-mode", "other:0"))
    assert res.returncode == 0, res.stderr[-3000:]
    z = np.load(out / "tkan_f0s4.npz", allow_pickle=False)
    assert z["ys"].shape == (0, 256) and z["untrained_ys"].shape == (3, 0, 256)
    assert json.loads(str(z["edges"])) == []
    assert np.isfinite(z["val_score"]).all() and z["ho_score"].shape == (NH,)
    s = json.loads((out / "tkan_f0s4.json").read_text())
    assert s["n_edges"] == 0 and s["complete"] is True
    assert s["exclusions"] == [0, 1]


@pytest.mark.parametrize("fault", ["incomplete", "bad_sha", "wrong_fold"])
def test_window_sidecar_failure_exits_2(env, fault):
    side = json.loads(env["side"].read_text())
    name = "f0s4"
    if fault == "incomplete":
        side["complete"] = False
        msg = "not complete"
    elif fault == "bad_sha":
        side["sha256"] = "0" * 64
        msg = "sha256 mismatch"
    else:
        name = "f1s4"
        msg = "fold 0"
    d = env["tmp"] / f"bad_{fault}"
    d.mkdir(exist_ok=True)
    win = d / "windows_fold0.npz"
    win.write_bytes(env["win"].read_bytes())
    (d / "windows_fold0.json").write_text(json.dumps(side))
    res = _run(env, d / "out", name=name, win=win)
    assert res.returncode == 2, res.stderr[-2000:]
    assert msg in res.stderr, res.stderr[-2000:]
    assert not (d / "out" / f"tkan_{name}.npz").exists()
