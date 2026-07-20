"""End-to-end smoke for the white-box runner (scripts/whitebox_phase3.py).

Builds a tiny synthetic NPZ tree + table, saves an (untrained) TKAN checkpoint and
its preprocessing.json, then runs the whole white-box pipeline in-process and
asserts every analysis ran (no recorded errors), the manifest is well-formed, and
all referenced artifacts exist with finite scalar results. No GPU, no real data.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

from pri2m.models.tkan_model import TKANModel
from tests._phase3_synth import make_synth_npz_tree, make_synth_table

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import whitebox_phase3 as W  # noqa: E402


def _make_checkpoint(tmp_path: pathlib.Path):
    """A small side-input TKAN checkpoint + preprocessing.json (the real shape)."""
    torch.manual_seed(0)
    model = TKANModel(
        input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3,
        side_input=True, side_features=("omega", "tpf", "tau"), side_hidden=4,
    )
    run_dir = tmp_path / "runs" / "synthrun" / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best_auc.pt"
    torch.save(model.state_dict(), ckpt)

    preprocessing = {
        "norm": {"mean": [0.0, 0.0, 0.0, 0.0], "std": [0.01, 0.01, 80.0, 80.0]},
        "side": {"features": ["omega", "tpf", "tau"]},
    }
    pre_path = tmp_path / "runs" / "synthrun" / "preprocessing.json"
    pre_path.write_text(json.dumps(preprocessing), encoding="utf-8")
    return ckpt, pre_path


def test_whitebox_runner_end_to_end(tmp_path):
    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2, 3), n_files=4, length=256, seed=0)
    table = make_synth_table(npz_root)
    table_csv = tmp_path / "table.csv"
    table.to_csv(table_csv, index=False)

    ckpt, pre_path = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "xai_out"

    rc = W.main([
        "--checkpoint", str(ckpt),
        "--npz-root", str(npz_root),
        "--table", str(table_csv),
        "--preprocessing", str(pre_path),
        "--output-dir", str(out_dir),
        "--n-windows", "6",
        "--top-k", "5",
        "--window-length", "128",
        "--window-stride", "64",
        "--pool", "1",
        "--ig-steps", "8",
        "--tsr-windows", "2",
        "--tsr-steps", "2",
    ])
    assert rc == 0

    manifest_path = out_dir / "whitebox_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Structure.
    for key in ("provenance", "results", "artifacts", "report", "errors"):
        assert key in manifest, f"manifest missing {key!r}"

    # EVERY analysis must have run cleanly on the synthetic checkpoint.
    assert manifest["errors"] == {}, f"analyses failed: {manifest['errors']}"

    prov = manifest["provenance"]
    assert prov["hidden_size"] == 8
    assert prov["side_input"] is True
    assert prov["n_windows"] >= 2

    # Key scalar results are finite real numbers (not invented, not NaN).
    res = manifest["results"]
    assert np.isfinite(res["delay_ablation"]["auc_drop"])
    assert np.isfinite(res["delay_ablation"]["auc_on"])
    # Per-window-relative sweep schema: factors (x own tau) + a no-delay baseline.
    sw = res["tau_sweep"]
    assert len(sw["auc"]) == len(sw["factors"])
    assert len(sw["own_tau"]) == prov["n_windows"]
    assert np.isfinite(sw["baseline_auc"])
    i1 = sw["factors"].index(1.0)
    # x1 imposes each window's OWN physical delay, per-window.
    assert sw["imposed_tau"][i1] == sw["own_tau"]
    share = res["weight_share"]["overall"]
    assert abs(sum(share.values()) - 1.0) < 1e-6
    assert np.isfinite(res["faithfulness"]["randomization_cosine"])
    assert len(res["symbolic_top"]) == 5
    assert res["cases"]["chatter"]["why"]["predicted"] in {"chatter", "stable"}

    # Spline-edge atlas: named, ranked edges plus all three visual views.
    atlas = res["spline_atlas"]
    assert isinstance(atlas, list) and len(atlas) >= 1
    assert {"kan", "feature", "block", "importance", "formula", "r2", "accepted"} <= set(atlas[0])
    for key in ("spline_atlas", "feature_bars", "edge_importance_named"):
        assert pathlib.Path(manifest["artifacts"][key]).is_file()

    # Temporal Saliency Rescaling: per-window shares are normalized and traceable.
    tsr = res["tsr"]
    assert tsr["n_windows"] >= 1
    for row in tsr["per_window"]:
        assert set(row) >= {"label", "channel_share_tsr", "channel_share_ig", "tr_peak_lag", "tau_raw"}
        assert abs(sum(row["channel_share_tsr"].values()) - 1.0) < 1e-6
    assert manifest["errors"] == {}

    # Referenced artifacts exist on disk.
    for name, rel in manifest["artifacts"].items():
        p = pathlib.Path(rel)
        assert p.exists() and p.stat().st_size > 0, f"missing artifact {name}: {rel}"

    # The report block is renderable: every value is a str/number/bool.
    for k, v in manifest["report"].items():
        assert isinstance(v, (str, int, float, bool)), f"report[{k}] not renderable: {type(v)}"


def test_build_labeled_batch_seeded_random(tmp_path):
    """Sampling is seeded-reproducible, class-balanced, and random (not head())."""
    from pri2m.data.phase3_sequence import WindowConfig

    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2, 3), n_files=4, length=256, seed=0)
    table = make_synth_table(npz_root)
    table_csv = tmp_path / "table.csv"
    table.to_csv(table_csv, index=False)
    cfg = WindowConfig(length=128, stride=64, channels=W.CHANNELS)

    b1, l1 = W.build_labeled_batch(table_csv, npz_root, None, 6, cfg, seed=7)
    b2, l2 = W.build_labeled_batch(table_csv, npz_root, None, 6, cfg, seed=7)
    # Same seed -> identical selection.
    assert (l1 == l2).all()
    assert torch.equal(b1["window"], b2["window"])
    # Both classes present and balanced.
    assert set(l1.tolist()) == {0, 1}
    # A different seed generally changes the draw (not asserted strictly to avoid
    # flakiness on tiny synth sets, but the windows must still be valid).
    b3, l3 = W.build_labeled_batch(table_csv, npz_root, None, 6, cfg, seed=123)
    assert set(l3.tolist()) == {0, 1}


def test_whitebox_runner_renders_report(tmp_path):
    """The manifest the runner writes feeds the renderer with no leftover placeholders."""
    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2, 3), n_files=4, length=256, seed=0)
    table = make_synth_table(npz_root)
    table_csv = tmp_path / "table.csv"
    table.to_csv(table_csv, index=False)
    ckpt, pre_path = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "xai_out"

    rc = W.main([
        "--checkpoint", str(ckpt), "--npz-root", str(npz_root), "--table", str(table_csv),
        "--preprocessing", str(pre_path), "--output-dir", str(out_dir),
        "--n-windows", "6", "--top-k", "5", "--window-length", "128",
        "--window-stride", "64", "--pool", "1", "--ig-steps", "8",
        "--tsr-windows", "2", "--tsr-steps", "2",
    ])
    assert rc == 0

    import render_whitebox_report as R  # noqa: E402

    manifest = json.loads((out_dir / "whitebox_manifest.json").read_text(encoding="utf-8"))
    template = (
        pathlib.Path(__file__).resolve().parents[1] / "docs" / "phase3_tkan" / "whitebox_analysis.md"
    ).read_text(encoding="utf-8")
    rendered = R.render_report(template, manifest["report"])
    assert "{{" not in rendered and "}}" not in rendered
    # Provenance actually filled (not left as N/A).
    assert "best_auc.pt" in rendered
