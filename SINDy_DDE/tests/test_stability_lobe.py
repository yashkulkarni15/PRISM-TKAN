"""Tests for the model-implied stability lobe diagram (SLD) XAI method.

Two layers, mirroring the white-box house style:

* **Pure-core, planted (ω, b, P) fields** — boundary extraction recovers a known
  per-column crossing, censored columns are flagged (not invented), and the
  faithfulness metric is exactly 0 when the model boundary equals ``blim`` and
  degrades predictably when shifted / inverted. No model, no data.
* **Runner end-to-end** — a tiny untrained ``TKANModel`` + a synthetic NPZ tree
  with a controlled (ω, b) grid → ``stability_lobe_phase3.main`` writes the
  figure, the two CSVs, and a ``sld_manifest.json`` in the white-box schema
  (``provenance / results / artifacts / errors / report``) with a renderable
  ``report`` block (``fig_sld`` + the metric keys).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

from pri2m.xai import stability_lobe as SL

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import stability_lobe_phase3 as R  # noqa: E402


# --------------------------------------------------------------------------- #
# Planted-field helpers
# --------------------------------------------------------------------------- #
def _cell(omega, b_mm, blim, p, label=None, n=10):
    """One (ω, b) field cell; label defaults to the analytic call int(b>=blim)."""
    if label is None:
        label = int(b_mm >= blim)
    return {"omega": float(omega), "b_mm": float(b_mm), "blim": float(blim),
            "label": int(label), "p_chatter": float(p), "n_windows": int(n)}


def _matched_field(omegas, blim=2.0, shift=0.0, invert=False):
    """A field whose per-column 0.5 crossing sits at ``blim + shift``.

    Each column gets a stable cell below and an unstable cell above the target
    crossing; ``invert`` flips the P values (model call opposite to analytic).
    """
    field = []
    for w in omegas:
        lo_b, hi_b = blim + shift - 1.0, blim + shift + 1.0
        p_lo, p_hi = (0.8, 0.2) if invert else (0.2, 0.8)
        field.append(_cell(w, lo_b, blim, p_lo))
        field.append(_cell(w, hi_b, blim, p_hi))
    return field


# --------------------------------------------------------------------------- #
# 1. Boundary extraction
# --------------------------------------------------------------------------- #
def test_extract_boundary_recovers_planted_crossing():
    # Two columns, planted crossings at b=3 (ω=5000) and b=5 (ω=6000).
    field = [
        _cell(5000, 2.0, 3.0, 0.2), _cell(5000, 4.0, 3.0, 0.8),
        _cell(6000, 4.0, 5.0, 0.2), _cell(6000, 6.0, 5.0, 0.8),
    ]
    boundary = SL.extract_model_boundary(field)
    by_omega = {b["omega"]: b for b in boundary}
    assert by_omega[5000.0]["status"] == "ok"
    assert abs(by_omega[5000.0]["b_model"] - 3.0) < 1e-9
    assert abs(by_omega[6000.0]["b_model"] - 5.0) < 1e-9


def test_extract_boundary_flags_censored_column():
    # An all-stable column (every P below 0.5) must NOT invent a crossing.
    field = [_cell(7000, 1.0, 2.0, 0.1), _cell(7000, 2.0, 2.0, 0.2),
             _cell(7000, 3.0, 2.0, 0.3)]
    boundary = SL.extract_model_boundary(field)
    assert len(boundary) == 1
    assert boundary[0]["status"] == "no_crossing_low"
    assert np.isnan(boundary[0]["b_model"])


def test_extract_boundary_flags_all_chatter_column():
    field = [_cell(7000, 1.0, 2.0, 0.9), _cell(7000, 3.0, 2.0, 0.95)]
    boundary = SL.extract_model_boundary(field)
    assert boundary[0]["status"] == "no_crossing_high"
    assert np.isnan(boundary[0]["b_model"])


# --------------------------------------------------------------------------- #
# 2. Faithfulness metric
# --------------------------------------------------------------------------- #
def test_faithfulness_zero_when_boundary_matches_blim():
    field = _matched_field([5000, 6000, 7000], blim=2.0, shift=0.0)
    boundary = SL.extract_model_boundary(field)
    m = SL.boundary_faithfulness(field, boundary)
    assert m["mae_b"] < 1e-9
    assert m["mape_b"] < 1e-9
    assert abs(m["iou_stable"] - 1.0) < 1e-9
    assert abs(m["label_agreement"] - 1.0) < 1e-9
    assert m["n_columns_used"] == 3
    assert m["n_columns_censored"] == 0


def test_faithfulness_tracks_known_shift():
    delta = 0.5
    field = _matched_field([5000, 6000], blim=2.0, shift=delta)
    boundary = SL.extract_model_boundary(field)
    m = SL.boundary_faithfulness(field, boundary)
    assert abs(m["mae_b"] - delta) < 1e-9


def test_faithfulness_inverted_field_disagrees():
    field = _matched_field([5000, 6000, 7000], blim=2.0, invert=True)
    boundary = SL.extract_model_boundary(field)
    m = SL.boundary_faithfulness(field, boundary)
    assert m["label_agreement"] < 1e-9
    assert m["iou_stable"] < 1e-9


def test_faithfulness_counts_censored_columns():
    good = _matched_field([5000], blim=2.0)
    censored = [_cell(6000, 1.0, 2.0, 0.1), _cell(6000, 3.0, 2.0, 0.2)]  # all stable
    field = good + censored
    boundary = SL.extract_model_boundary(field)
    m = SL.boundary_faithfulness(field, boundary)
    assert m["n_columns_used"] == 1
    assert m["n_columns_censored"] == 1


# --------------------------------------------------------------------------- #
# 3. Analytic overlay
# --------------------------------------------------------------------------- #
def test_analytic_boundary_unique_sorted_drops_nonpositive():
    rows = pd.DataFrame({
        "omega": [6000, 6000, 5000, 5000, 1000],
        "blim":  [5.0,  5.0,  3.0,  3.0,  -1.5],   # low-speed negative blim dropped
    })
    ab = SL.analytic_boundary(rows)
    assert [a["omega"] for a in ab] == [5000.0, 6000.0]
    assert [a["blim"] for a in ab] == [3.0, 5.0]


# --------------------------------------------------------------------------- #
# 4. Caveats
# --------------------------------------------------------------------------- #
def test_caveats_fire_on_ceiling_and_censoring():
    ceil = SL.sld_caveats({"per_cell_auc": 0.999, "n_columns_censored": 0}, n_omega=10)
    assert any("ceiling" in c.lower() for c in ceil)
    cens = SL.sld_caveats({"per_cell_auc": 0.7, "n_columns_censored": 5}, n_omega=10)
    assert any("censor" in c.lower() for c in cens)
    clean = SL.sld_caveats({"per_cell_auc": 0.7, "n_columns_censored": 1}, n_omega=10)
    assert clean == []


# --------------------------------------------------------------------------- #
# Runner end-to-end
# --------------------------------------------------------------------------- #
CHANNELS = ("xtpos", "ytpos", "Forcex", "Forcey")


def _write_field_tree(root: pathlib.Path):
    """A controlled NPZ tree: 2 datasets (param-twins) × 2 ω × 3 b, blim=2.

    Same (ω, b) grid in both datasets so cells pool; each ω-column brackets blim
    so a boundary can (in principle) be traced. Returns (npz_root, table_df).
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    fs, tpf, fn, length = 10_000.0, 200.0, 900.0, 256
    rows = []
    for ds in (29, 89):  # a param-twin pair (share the grid), parent_group 29
        ds_dir = root / f"dataset_{ds}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        for wi, omega in enumerate((5000.0, 6000.0)):
            for bi, b_mm in enumerate((1.0, 2.0, 3.0)):
                blim = 2.0
                label = int(b_mm >= blim)
                stem = f"d{ds}_w{wi}_b{bi}"
                t = np.arange(length) / fs
                base = np.sin(2 * np.pi * tpf * t)
                if label:
                    base = base + 1.5 * np.sin(2 * np.pi * fn * t + rng.uniform(0, 6.28))
                base = base + 0.05 * rng.standard_normal(length)
                np.savez_compressed(
                    ds_dir / f"{stem}.npz",
                    fs=np.asarray(fs), label=np.asarray(label),
                    omega=np.asarray(omega), tpf=np.asarray(tpf), fn=np.asarray(fn),
                    dataset=np.asarray(ds), source_file=np.asarray(f"{stem}.h5"),
                    xtpos=(0.001 * base).astype(float),
                    ytpos=(0.001 * np.roll(base, 5)).astype(float),
                    Forcex=(50.0 * base).astype(float),
                    Forcey=(50.0 * np.roll(base, 3)).astype(float),
                )
                rows.append({
                    "dataset": ds, "file": f"{stem}.h5", "omega": omega,
                    "b_mm": b_mm, "blim": blim, "b_over_blim": b_mm / blim,
                    "label": label, "tpf": tpf, "fn": fn, "parent_group": 29,
                    "sample_weight": 1.0, "exclude_supervised": False,
                    "physics_usable": True, "mode_label": "synthetic",
                })
    return root, pd.DataFrame(rows)


def _make_checkpoint(tmp_path: pathlib.Path):
    from pri2m.models.tkan_model import TKANModel

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


def test_build_field_returns_one_cell_per_grid_point(tmp_path):
    from pri2m.data.phase3_sequence import NormStats, WindowConfig
    from pri2m.models.tkan_model import TKANModel

    npz_root, table = _write_field_tree(tmp_path / "npz")
    torch.manual_seed(0)
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1,
                      side_input=True, side_features=("omega", "tpf", "tau"), side_hidden=4)
    model.eval()
    norm = NormStats(mean=np.zeros(4), std=np.array([0.01, 0.01, 80.0, 80.0]))
    cfg = WindowConfig(length=128, stride=64, channels=CHANNELS)
    field = SL.build_pchatter_field(model, table, npz_root, norm, cfg)
    # 2 ω × 3 b = 6 unique cells (the two datasets pool into the same cells).
    assert len(field) == 6
    for c in field:
        assert c["n_windows"] > 0
        assert 0.0 <= c["p_chatter"] <= 1.0
        assert c["blim"] == 2.0


def test_runner_end_to_end(tmp_path):
    npz_root, table = _write_field_tree(tmp_path / "npz")
    table_csv = tmp_path / "table.csv"
    table.to_csv(table_csv, index=False)
    ckpt, pre_path = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "sld_out"
    assets_dir = tmp_path / "assets"

    rc = R.main([
        "--checkpoint", str(ckpt), "--npz-root", str(npz_root),
        "--table", str(table_csv), "--preprocessing", str(pre_path),
        "--output-dir", str(out_dir), "--assets-dir", str(assets_dir),
        "--parent-group", "29", "--fold", "2", "--tag", "fold2ds29grp",
        "--window-length", "128", "--window-stride", "64", "--pool", "1",
    ])
    assert rc == 0

    manifest_path = out_dir / "sld_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("provenance", "results", "artifacts", "errors", "report"):
        assert key in manifest, f"manifest missing {key!r}"
    assert manifest["errors"] == {}, f"analyses failed: {manifest['errors']}"

    prov = manifest["provenance"]
    assert prov["parent_group"] == 29
    assert prov["fold"] == 2
    assert prov["n_cells"] == 6
    assert prov["n_omega"] == 2
    assert prov["n_files"] == 12

    sld = manifest["results"]["sld"]
    for k in ("mae_b", "mape_b", "iou_stable", "label_agreement", "per_cell_auc",
              "n_columns_used", "n_columns_censored", "boundary", "field_summary"):
        assert k in sld, f"results.sld missing {k!r}"

    rep = manifest["report"]
    for k in ("sld_mae_b", "sld_mape_b", "sld_iou", "sld_label_agreement",
              "sld_per_cell_auc", "sld_boundary_table", "fig_sld"):
        assert k in rep, f"report missing {k!r}"
    for k, v in rep.items():
        assert isinstance(v, (str, int, float, bool)), f"report[{k}] not renderable: {type(v)}"

    # Artifacts on disk.
    for name, rel in manifest["artifacts"].items():
        p = pathlib.Path(rel)
        assert p.exists() and p.stat().st_size > 0, f"missing artifact {name}: {rel}"
    assert (out_dir / "sld_field.csv").exists()
    assert (out_dir / "sld_boundary.csv").exists()
    assert (out_dir / f"sld_{prov['tag']}.png").exists()
    # Figure mirrored into the research-update assets dir.
    assert (assets_dir / f"sld_{prov['tag']}.png").exists()
