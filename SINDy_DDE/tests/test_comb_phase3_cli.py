"""The comb runner reads stored manifests and never invents its inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comb_phase3 as C  # noqa: E402


def _regen(rows=None):
    rows = rows or [
        {"corr": 0.99, "gain": 1.0, "weight": 3.0, "kan": "encoder.layers.0.kan_z",
         "channel": "xtpos", "degenerate": 0, "out": 1, "in_inst": 0, "in_delay": 68},
        {"corr": 0.20, "gain": 1.0, "weight": 1.0, "kan": "encoder.layers.0.kan_r",
         "channel": "ytpos", "degenerate": 0, "out": 2, "in_inst": 1, "in_delay": 69},
    ]
    return {
        "provenance": {"checkpoint": "/some/best_auc.pt", "n_windows": 128},
        "rows": rows,
        "control_rows": [dict(rows[1])],
        "untrained_null_summary": {"mean": -0.02, "mean_abs_corr": 0.79},
        "top_pairs": [dict(rows[0])],
    }


@pytest.fixture
def tree(tmp_path):
    regen_root = tmp_path / "regenerative"
    (regen_root / "fold0").mkdir(parents=True)
    (regen_root / "fold0" / "regenerative.json").write_text(json.dumps(_regen()))

    fuzz_root = tmp_path / "fuzz"
    (fuzz_root / "fold0").mkdir(parents=True)
    (fuzz_root / "fold0" / "fuzz_manifest.json").write_text(json.dumps({
        "contrasts": {"band_notch": {"energy_match": {
            "tooth_centers_hz": [500.0, 1000.0, 1500.0],
            "fn_hz_mean": 1050.0, "fn_hz_sd": 400.0,
        }}}
    }))
    return regen_root, fuzz_root, tmp_path


def test_frequencies_come_from_the_fuzz_manifest(tree):
    regen_root, fuzz_root, _ = tree

    freqs = C.read_frequencies(fuzz_root, "fold0")

    assert freqs["tooth_pass_hz"] == 500.0
    assert freqs["fn_hz_mean"] == 1050.0


def test_a_fold_without_frequencies_returns_none_not_a_guess(tree):
    _, fuzz_root, _ = tree

    assert C.read_frequencies(fuzz_root, "fold9") is None


def test_regenerative_fold_name_maps_onto_the_fuzz_directory(tree):
    """regenerative uses fold2_w512 while fuzz uses fold2; the runner has to
    bridge that or every fold would silently lose its frequency axis."""
    _, fuzz_root, tmp = tree
    (fuzz_root / "fold2").mkdir()
    (fuzz_root / "fold2" / "fuzz_manifest.json").write_text(json.dumps({
        "contrasts": {"band_notch": {"energy_match": {
            "tooth_centers_hz": [507.0], "fn_hz_mean": 1072.0, "fn_hz_sd": 492.0}}}
    }))

    assert C.read_frequencies(fuzz_root, "fold2_w512")["tooth_pass_hz"] == 507.0


def test_chatter_summary_refuses_without_frequencies():
    out = C.chatter_band_summary(1.0, None)

    assert out["available"] is False
    assert "reason" in out


def test_chatter_summary_reports_both_the_point_and_the_spread():
    """A single ratio near a harmonic shows almost no attenuation while the
    spread shows plenty; reporting only one of them would mislead either way."""
    freqs = {"tooth_pass_hz": 500.0, "fn_hz_mean": 1000.0, "fn_hz_sd": 400.0}

    out = C.chatter_band_summary(1.0, freqs)

    assert out["attenuation_at_mean_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert out["expected_attenuation_over_spread"] < 0.95
    assert out["attenuation_of_forced_content"] == 1.0


def test_end_to_end_writes_a_manifest_that_states_no_network_was_loaded(tree, tmp_path):
    regen_root, fuzz_root, _ = tree
    out_dir = tmp_path / "out"

    rc = C.main([
        "--regen-root", str(regen_root), "--fuzz-root", str(fuzz_root),
        "--output-dir", str(out_dir), "--no-figures",
    ])

    assert rc == 0
    m = json.loads((out_dir / "comb_manifest.json").read_text())
    assert m["network_reloaded"] is False
    assert m["per_fold"][0]["trained"]["n_used"] == 1
    assert m["per_fold"][0]["cross_channel_control"] is not None


def test_end_to_end_refuses_an_empty_tree(tmp_path):
    rc = C.main(["--regen-root", str(tmp_path / "nothing"), "--no-figures"])

    assert rc == 2
