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

    # The band-notch contrast must actually COMPUTE here, not silently take the
    # "fn unavailable" fallback -- errors == {} alone would not catch that,
    # since the fallback is a returned dict rather than a raised exception.
    bn = manifest["contrasts"]["band_notch"]
    assert "error" not in bn, f"band_notch degraded to a fallback: {bn}"
    assert set(bn) >= {"chatter", "tooth", "energy_match"}
    assert "band_notch" not in manifest["fuzzers"]      # sibling, never nested
    em = bn["energy_match"]
    assert set(em) >= {"chatter_frac", "tooth_frac", "max_rel_err", "matched"}
    # Whether the energies matched depends on spectral resolution, but the
    # verdict must always be recorded so an unfair contrast cannot read as real.
    assert isinstance(em["matched"], bool)


# ---- Task 8: the existing --fuzzers flag must reach the new stressors ------ #
# No implementation: fuzz_phase3 already ships --fuzzers backed by
# _parse_fuzzers, which validates against FUZZERS at call time. Adding a
# parallel --stressors flag would be a second way to do the same thing. These
# tests only prove the existing flag picks up the newly registered entries.

def test_fuzzers_flag_accepts_the_new_physics_stressors():
    import fuzz_phase3 as FZ

    chosen = FZ._parse_fuzzers("phase_randomize,magnitude_randomize,lag_tau_shuffle")
    assert chosen == ["phase_randomize", "magnitude_randomize", "lag_tau_shuffle"]


def test_fuzzers_all_includes_the_new_physics_stressors():
    import fuzz_phase3 as FZ
    from pri2m.xai.degradation import FUZZERS

    chosen = FZ._parse_fuzzers("all")
    assert set(chosen) == set(FUZZERS)
    assert {"phase_randomize", "magnitude_randomize", "lag_tau_shuffle"} <= set(chosen)


def test_band_notch_contrast_is_a_sibling_of_fuzzers_not_nested_in_it():
    """The contrast is a paired comparison, not a graded sweep.

    Nesting it under "fuzzers" would put a non-sweep object where every consumer
    expects levels/results/breakdown, so the manifest key must stay a sibling.
    """
    import fuzz_phase3 as FZ

    src = pathlib.Path(FZ.__file__).read_text(encoding="utf-8")
    assert 'manifest["contrasts"]' in src
    assert '"band_notch"' in src
    # never written into the fuzzers dict
    assert 'results["band_notch"]' not in src


# ---- the caller must hand over PER-WINDOW frequencies, not batch means ----- #
# The 2026-07-29 five-fold run collapsed fs/fn/tpf to batch means before calling
# the contrast, which aims every window's notch at a frequency where that window
# has no chatter energy. These tests exercise the runner's own guard + hand-off,
# so they fail on the mean-collapsing version with a real assertion rather than
# only inside degradation.py.

def _contrast_model():
    torch.manual_seed(0)
    return TKANModel(
        input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3
    ).eval()


def _contrast_batch(fn_hz, fs_hz=10_240.0, length=1024, seed=0, noise=0.05):
    """Batch whose window ``b`` carries a dominant tone at its own ``fn``."""
    import numpy as np

    fn_arr = np.atleast_1d(np.asarray(fn_hz, dtype=float))
    n_b = fn_arr.size
    g = torch.Generator().manual_seed(seed)
    n = np.arange(length, dtype=float)
    tones = np.stack([np.sin(2 * np.pi * float(f) / float(fs_hz) * n) for f in fn_arr])
    win = torch.as_tensor(tones, dtype=torch.float32).unsqueeze(-1).repeat(1, 1, 4)
    win = win + noise * torch.randn(win.shape, generator=g)
    labels = np.array([b % 2 for b in range(n_b)])
    batch = {
        "window": win,
        "tau": torch.full((n_b,), 5, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.float32),
        "b_over_blim": torch.rand(n_b, generator=g),
        "omega": torch.full((n_b,), 5000.0),
        "tpf": torch.full((n_b,), 200.0),
        "fn": torch.as_tensor(fn_arr, dtype=torch.float32),
        "fs": torch.full((n_b,), float(fs_hz)),
        "file": [f"w{b}" for b in range(n_b)],
    }
    return batch, labels


def test_band_notch_caller_hands_over_per_window_frequencies(monkeypatch):
    import numpy as np

    import fuzz_phase3 as FZ
    from pri2m.xai import degradation

    seen = {}

    def _spy(model, batch, labels, **kwargs):
        seen.update(kwargs)
        return {"energy_match": {}}

    monkeypatch.setattr(degradation, "run_band_notch_contrast", _spy)
    batch, labels = _contrast_batch([800.0, 2000.0, 1500.0, 900.0])
    FZ._band_notch_contrast(_contrast_model(), batch, labels)
    for key in ("fs", "fn", "tpf"):
        assert np.size(seen[key]) == 4, f"{key} was collapsed to a batch statistic"
    np.testing.assert_allclose(seen["fn"], [800.0, 2000.0, 1500.0, 900.0])


def test_band_notch_caller_notches_each_windows_own_chatter_frequency():
    """End-to-end through the runner: no window's band may miss its own tone."""
    import fuzz_phase3 as FZ

    batch, labels = _contrast_batch([800.0, 2000.0])
    out = FZ._band_notch_contrast(_contrast_model(), batch, labels)
    assert "error" not in out
    assert out["energy_match"]["chatter_frac_min"] > 0.9


def test_band_notch_caller_drops_nonfinite_windows_instead_of_the_whole_batch():
    """One NaN fn used to poison the batch mean and abort the entire contrast."""
    import fuzz_phase3 as FZ

    batch, labels = _contrast_batch([800.0, 2000.0, 1500.0])
    batch["fn"][2] = float("nan")
    out = FZ._band_notch_contrast(_contrast_model(), batch, labels)
    assert "error" not in out, "a single non-finite fn must not kill the contrast"
    assert out["energy_match"]["n_windows"] == 2
    assert out["n_windows_dropped_nonfinite"] == 1


def test_band_notch_caller_still_reports_fn_unavailable_when_nothing_is_usable():
    """The existing guard survives: missing, empty or all-non-finite fn -> error."""
    import fuzz_phase3 as FZ

    model = _contrast_model()
    batch, labels = _contrast_batch([800.0, 2000.0])

    no_fn = dict(batch)
    no_fn.pop("fn")
    assert FZ._band_notch_contrast(model, no_fn, labels) == {"error": "fn unavailable"}

    empty = dict(batch, fn=torch.zeros(0))
    assert FZ._band_notch_contrast(model, empty, labels) == {"error": "fn unavailable"}

    nan_fn = dict(batch, fn=torch.full((2,), float("nan")))
    assert FZ._band_notch_contrast(model, nan_fn, labels) == {"error": "fn unavailable"}
