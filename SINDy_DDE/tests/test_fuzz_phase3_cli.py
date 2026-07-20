"""End-to-end fuzz runner smoke on a synthetic checkpoint."""

import json
import pathlib
import sys

import torch

from pri2m.models.tkan_model import TKANModel
from tests._phase3_synth import make_synth_npz_tree, make_synth_table

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))


def _checkpoint(tmp_path):
    torch.manual_seed(0)
    model = TKANModel(
        input_size=4,
        hidden_size=8,
        num_layers=1,
        grid_size=5,
        spline_order=3,
        side_input=True,
        side_features=("omega", "tpf", "tau"),
        side_hidden=4,
    )
    d = tmp_path / "runs" / "synthrun" / "checkpoints"
    d.mkdir(parents=True)
    torch.save(model.state_dict(), d / "best_auc.pt")
    (d.parent / "preprocessing.json").write_text(json.dumps({
        "norm": {"mean": [0.0] * 4, "std": [1.0] * 4},
        "side": {"features": ["omega", "tpf", "tau"]},
    }))
    return d / "best_auc.pt"


def test_fuzz_cli_end_to_end(tmp_path):
    import fuzz_phase3 as FZ

    npz_root = make_synth_npz_tree(
        tmp_path / "npz", clean_ids=(1, 2), n_files=4, length=256, seed=0
    )
    make_synth_table(npz_root).to_csv(tmp_path / "table.csv", index=False)
    out = tmp_path / "fuzz"
    rc = FZ.main([
        "--checkpoint", str(_checkpoint(tmp_path)),
        "--npz-root", str(npz_root),
        "--table", str(tmp_path / "table.csv"),
        "--output-dir", str(out),
        "--n-windows", "6",
        "--window-length", "128",
        "--window-stride", "64",
        "--pool", "1",
        "--quick",
        "--tag", "synth",
    ])
    assert rc == 0
    manifest = json.loads((out / "fuzz_manifest.json").read_text())
    assert set(manifest) >= {
        "provenance", "baseline", "fuzzers", "self_check", "artifacts", "errors"
    }
    assert manifest["errors"] == {}
    assert manifest["self_check"]["passed"] is True
    assert manifest["provenance"]["quick"] is True
    for name in ("input_noise", "input_smooth", "tau_jitter", "spline_coarsen", "weight_noise"):
        fz = manifest["fuzzers"][name]
        assert fz["results"][0]["level"] == fz["levels"][0]
        assert "breakdown" in fz
        assert (out / f"fuzz_{name}.png").is_file()
    assert (out / "fuzz_dashboard.png").is_file()
    for key in ("auc", "delay_drop", "mae"):
        assert manifest["baseline"][key] is not None
