"""Sharded TSR reproduces a single un-sharded white-box run, bit-for-bit.

The expensive TSR stage is split across a SLURM array in production
(``whitebox_phase3.py --tsr-only --tsr-shard K --tsr-nshards N``) and merged by
``scripts/merge_tsr_shards.py``. This asserts that path yields the identical
``results['tsr']`` block a single inline run would write -- the correctness
guarantee the whole shard workflow rests on -- plus that the merged manifest's
report block regains its TSR table, and that partial/inconsistent shard sets are
rejected rather than silently under-reported.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest
import torch

from pri2m.models.tkan_model import TKANModel
from tests._phase3_synth import make_synth_npz_tree, make_synth_table

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import merge_tsr_shards as M  # noqa: E402
import whitebox_phase3 as W  # noqa: E402

TSR_WINDOWS = 5   # uneven across NSHARDS=3 -> shards of size 2/2/1, exercising reorder
NSHARDS = 3


def _make_checkpoint(tmp_path: pathlib.Path):
    torch.manual_seed(0)
    model = TKANModel(
        input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3,
        side_input=True, side_features=("omega", "tpf", "tau"), side_hidden=4,
    )
    run_dir = tmp_path / "runs" / "synthrun" / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best_auc.pt"
    torch.save(model.state_dict(), ckpt)
    pre = {
        "norm": {"mean": [0.0, 0.0, 0.0, 0.0], "std": [0.01, 0.01, 80.0, 80.0]},
        "side": {"features": ["omega", "tpf", "tau"]},
    }
    pre_path = tmp_path / "runs" / "synthrun" / "preprocessing.json"
    pre_path.write_text(json.dumps(pre), encoding="utf-8")
    return ckpt, pre_path


def _args(ckpt, pre_path, npz_root, table_csv, out_dir, extra):
    return [
        "--checkpoint", str(ckpt), "--npz-root", str(npz_root), "--table", str(table_csv),
        "--preprocessing", str(pre_path), "--output-dir", str(out_dir),
        "--n-windows", "10", "--top-k", "5", "--window-length", "128",
        "--window-stride", "64", "--pool", "1", "--ig-steps", "8",
        "--tsr-windows", str(TSR_WINDOWS), "--tsr-steps", "2",
    ] + extra


def _approx_equal(a, b, path="tsr"):
    """Recursive equality with a float tolerance (both sides are JSON round-tripped)."""
    assert type(a) is type(b) or {type(a), type(b)} <= {int, float}, f"{path}: {type(a)} vs {type(b)}"
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: keys {a.keys()} != {b.keys()}"
        for k in a:
            _approx_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: len {len(a)} != {len(b)}"
        for j, (x, y) in enumerate(zip(a, b)):
            _approx_equal(x, y, f"{path}[{j}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(float(a) - float(b)) <= 1e-9, f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a} != {b}"


@pytest.fixture
def synth(tmp_path):
    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2, 3), n_files=4, length=256, seed=0)
    table = make_synth_table(npz_root)
    table_csv = tmp_path / "table.csv"
    table.to_csv(table_csv, index=False)
    ckpt, pre_path = _make_checkpoint(tmp_path)
    return npz_root, table_csv, ckpt, pre_path


def test_sharded_tsr_matches_full_run(tmp_path, synth):
    npz_root, table_csv, ckpt, pre_path = synth

    # (1) Reference: one full, un-sharded run.
    full_out = tmp_path / "full"
    assert W.main(_args(ckpt, pre_path, npz_root, table_csv, full_out, [])) == 0
    full_tsr = json.loads((full_out / "whitebox_manifest.json").read_text())["results"]["tsr"]
    assert full_tsr["n_windows"] == TSR_WINDOWS

    # (2) Sharded: fixed stages once (--skip-tsr) writes the base manifest (tsr=null)...
    sh_out = tmp_path / "sharded"
    assert W.main(_args(ckpt, pre_path, npz_root, table_csv, sh_out, ["--skip-tsr"])) == 0
    base = json.loads((sh_out / "whitebox_manifest.json").read_text())
    assert base["results"]["tsr"] is None
    assert "tsr_table" not in base["report"]

    # ...then each array task computes its shard of TSR windows.
    total = 0
    for k in range(NSHARDS):
        assert W.main(_args(ckpt, pre_path, npz_root, table_csv, sh_out,
                            ["--tsr-only", "--tsr-shard", str(k), "--tsr-nshards", str(NSHARDS)])) == 0
        pl = json.loads((sh_out / "tsr_shards" / f"shard_{k:04d}_of_{NSHARDS:04d}.json").read_text())
        assert pl["n_selected"] == TSR_WINDOWS
        total += len(pl["per_window_indexed"])
    assert total == TSR_WINDOWS   # every window covered exactly once

    # (3) Merge -> results['tsr'] identical to the full run; report regains TSR.
    assert M.main(["--output-dir", str(sh_out)]) == 0
    merged = json.loads((sh_out / "whitebox_manifest.json").read_text())
    _approx_equal(merged["results"]["tsr"], full_tsr)
    assert merged["report"]["tsr_table"]          # non-empty
    assert merged["report"]["tsr_windows"] == TSR_WINDOWS
    # The fixed-stage results were untouched by the merge.
    assert merged["results"]["symbolic_top"] == base["results"]["symbolic_top"]


def test_merge_rejects_incomplete_shard_set(tmp_path, synth):
    npz_root, table_csv, ckpt, pre_path = synth
    out = tmp_path / "partial"
    assert W.main(_args(ckpt, pre_path, npz_root, table_csv, out, ["--skip-tsr"])) == 0
    # Only shard 0 of 3 is present -> merge must refuse (never under-report the sample).
    assert W.main(_args(ckpt, pre_path, npz_root, table_csv, out,
                        ["--tsr-only", "--tsr-shard", "0", "--tsr-nshards", str(NSHARDS)])) == 0
    with pytest.raises(ValueError, match="incomplete TSR shard set"):
        M.merge_shards(out / "whitebox_manifest.json", out / "tsr_shards")
