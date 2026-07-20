"""predict_windows_phase3: per-file network probabilities for counterfactuals."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

from pri2m.models.tkan_model import TKANModel
from tests._phase3_synth import make_synth_npz_tree, make_synth_table

# Sibling scripts live under scripts/ (not a package on sys.path by default);
# conftest.py only puts src/ and the repo root on sys.path. Mirrors the pattern
# used by tests/test_whitebox_phase3.py and tests/test_render_whitebox_report.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import counterfactuals_phase3 as CF  # noqa: E402
import predict_windows_phase3 as P  # noqa: E402


def _checkpoint(tmp_path):
    torch.manual_seed(0)
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3)
    ckpt_dir = tmp_path / "runs" / "synthrun" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "best_auc.pt"
    torch.save(model.state_dict(), ckpt)
    (ckpt_dir.parent / "preprocessing.json").write_text(json.dumps({
        "norm": {"mean": [0.0] * 4, "std": [1.0] * 4},
        "window": {"length": 128, "stride": 64},
        "pool": 1,
    }))
    return ckpt


def test_cli_writes_per_file_probs(tmp_path):
    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2), n_files=3, length=256, seed=0)
    table = make_synth_table(npz_root)
    csv = tmp_path / "table.csv"
    table.to_csv(csv, index=False)
    ckpt = _checkpoint(tmp_path)
    out = tmp_path / "preds.csv"
    rc = P.main([
        "--checkpoint", str(ckpt), "--table", str(csv), "--npz-root", str(npz_root),
        "--out", str(out), "--batch-size", "8",
    ])
    assert rc == 0 and out.is_file()
    preds = pd.read_csv(out)
    assert list(preds.columns) == ["file", "prob", "n_windows", "dataset", "label"]
    assert preds["prob"].between(0, 1).all()
    assert (preds["n_windows"] >= 1).all()
    assert preds["file"].isin(table["file"]).all()          # merge key intact
    assert len(preds) == preds["file"].nunique()

    rc2 = P.main([
        "--checkpoint", str(ckpt), "--table", str(csv), "--npz-root", str(npz_root),
        "--out", str(tmp_path / "p2.csv"), "--batch-size", "8",
    ])
    assert rc2 == 0
    assert np.allclose(pd.read_csv(tmp_path / "p2.csv")["prob"], preds["prob"])  # deterministic


def test_predictions_merge_into_counterfactual_surrogate(tmp_path):
    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1,), with_mirrors=False, n_files=4, length=256)
    table = make_synth_table(npz_root)
    csv = tmp_path / "table.csv"
    table.to_csv(csv, index=False)
    out = tmp_path / "preds.csv"
    P.main(["--checkpoint", str(_checkpoint(tmp_path)), "--table", str(csv),
            "--npz-root", str(npz_root), "--out", str(out)])
    predict, info = CF.build_surrogate(table, predictions=str(out), seed=0)
    assert info["source"] == "model_predictions" and info["target_col"] == "prob"
    p = predict(np.array([[float(table["omega"].iloc[0]), float(table["b_mm"].iloc[0])]]))
    assert 0.0 <= float(p[0]) <= 1.0
