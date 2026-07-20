"""XAI must z-score sampled windows the same way training did.

Regression test for the Phase 3 XAI off-distribution bug: ``scripts/xai_phase3.py``
fed the model RAW windows, while the model trains on globally z-scored windows
(``Phase3SequenceDataset.__getitem__`` -> ``NormStats.apply``). Raw force channels
(O(tens-hundreds)) saturate the model's ``tanh`` grid squash to +/-1, so every
attribution was computed wildly off-distribution. The fix loads the per-run
``preprocessing.json`` and applies the saved mean/std before inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_XAI_PATH = _REPO_ROOT / "scripts" / "xai_phase3.py"


def _load_xai_module():
    """Import ``scripts/xai_phase3.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("xai_phase3", _XAI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_preprocessing(mean, std, channels) -> dict:
    """Hand-written manifest matching the companion agent's schema."""
    return {
        "norm": {"mean": list(mean), "std": list(std)},
        "channels": list(channels),
        "window": {"length": 256, "stride": 256},
        "pool": 0,
        "fold": 0,
        "counts": {},
        "leakage_verified": True,
        "seed": 0,
    }


def test_apply_norm_zscores_window_to_unit_mean_std():
    """``_apply_norm`` must turn a raw window into the saved z-scored one."""
    xai = _load_xai_module()

    rng = np.random.default_rng(0)
    L, C = 256, 4
    # Per-channel mean/std deliberately spread the channels wide apart -- the
    # force channels are O(hundreds), exactly the case that saturates tanh.
    true_mean = np.array([0.0, -5.0, 120.0, -300.0])
    true_std = np.array([1.0, 2.0, 40.0, 80.0])
    window = (rng.standard_normal((L, C)) * true_std + true_mean).astype(np.float32)

    pre = _make_preprocessing(true_mean, true_std, ["xtpos", "ytpos", "Forcex", "Forcey"])
    normed = xai._apply_norm(window, pre)

    # The window handed to the model must be the z-scored version, not raw.
    assert normed.shape == window.shape
    assert not np.allclose(normed, window)
    # Per-channel ~0 mean / ~1 std after normalization.
    np.testing.assert_allclose(normed.mean(axis=0), np.zeros(C), atol=0.15)
    np.testing.assert_allclose(normed.std(axis=0), np.ones(C), atol=0.15)


def test_apply_norm_matches_normstats_apply():
    """``_apply_norm`` must be numerically identical to ``NormStats.apply``."""
    xai = _load_xai_module()
    from pri2m.data.phase3_sequence import NormStats

    mean = [1.0, -2.0, 50.0, -50.0]
    std = [0.5, 3.0, 25.0, 10.0]
    window = np.arange(256 * 4, dtype=np.float32).reshape(256, 4)

    pre = _make_preprocessing(mean, std, ["a", "b", "c", "d"])
    normed = xai._apply_norm(window, pre)

    expected = NormStats(
        mean=np.asarray(mean, dtype=np.float64),
        std=np.asarray(std, dtype=np.float64),
    ).apply(window)
    np.testing.assert_allclose(normed, expected)


def test_find_preprocessing_walks_up_from_checkpoint(tmp_path):
    """Auto-discovery finds ``preprocessing.json`` in the run dir above the checkpoint."""
    xai = _load_xai_module()

    run_dir = tmp_path / "runs" / "run_42"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "best.pt"
    ckpt.write_bytes(b"")

    pre_path = run_dir / "preprocessing.json"
    pre = _make_preprocessing([0.0] * 4, [1.0] * 4, ["a", "b", "c", "d"])
    pre_path.write_text(json.dumps(pre), encoding="utf-8")

    found = xai._find_preprocessing(str(ckpt))
    assert found is not None
    assert Path(found).resolve() == pre_path.resolve()


def test_find_preprocessing_returns_none_when_absent(tmp_path):
    """Auto-discovery returns ``None`` when no manifest exists anywhere above."""
    xai = _load_xai_module()

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "best.pt"
    ckpt.write_bytes(b"")

    assert xai._find_preprocessing(str(ckpt)) is None


# --------------------------------------------------------------------------- #
# _load_model: side-input checkpoints
# --------------------------------------------------------------------------- #
def _model_args() -> argparse.Namespace:
    """The argparse fields ``_load_model`` consumes, all at their CLI defaults."""
    return argparse.Namespace(input_size=None, hidden_size=None, grid_size=5, spline_order=3)


def _save_side_checkpoint(tmp_path, *, side_hidden: int = 6):
    """Build a small side-input TKANModel with non-trivial norm stats, save its state_dict."""
    torch = pytest.importorskip("torch")
    from pri2m.models.tkan_model import TKANModel

    torch.manual_seed(0)
    model = TKANModel(input_size=4, hidden_size=8, side_input=True, side_hidden=side_hidden)
    model.set_side_norm([3000.0, 200.0, 50.0], [500.0, 40.0, 10.0])
    ckpt = tmp_path / "best.pt"
    torch.save(model.state_dict(), ckpt)
    return model, ckpt


def test_load_model_side_checkpoint_round_trips_with_preprocessing_side_block(tmp_path):
    """A side checkpoint + a ``side`` preprocessing block reconstructs the exact model."""
    torch = pytest.importorskip("torch")
    xai = _load_xai_module()

    model, ckpt = _save_side_checkpoint(tmp_path, side_hidden=6)
    pre = _make_preprocessing([0.0] * 4, [1.0] * 4, ["a", "b", "c", "d"])
    pre["side"] = {
        "features": ["omega", "tpf", "tau"],
        "mean": [3000.0, 200.0, 50.0],
        "std": [500.0, 40.0, 10.0],
    }

    loaded = xai._load_model(str(ckpt), _model_args(), pre)
    assert loaded.side_input is True
    assert loaded.side_hidden == 6
    assert loaded.side_features == ("omega", "tpf", "tau")
    assert loaded.hidden_size == 8
    assert loaded.input_size == 4
    torch.testing.assert_close(loaded.side_mean, model.side_mean)
    torch.testing.assert_close(loaded.side_std, model.side_std)


def test_load_model_side_checkpoint_falls_back_to_default_features(tmp_path, capsys):
    """No ``side`` block in preprocessing.json -> DEFAULT_SIDE_FEATURES + loud warning."""
    torch = pytest.importorskip("torch")
    from pri2m.models.tkan_model import DEFAULT_SIDE_FEATURES

    xai = _load_xai_module()
    model, ckpt = _save_side_checkpoint(tmp_path)
    pre = _make_preprocessing([0.0] * 4, [1.0] * 4, ["a", "b", "c", "d"])  # no "side"

    loaded = xai._load_model(str(ckpt), _model_args(), pre)
    assert loaded.side_input is True
    assert loaded.side_features == tuple(DEFAULT_SIDE_FEATURES)
    torch.testing.assert_close(loaded.side_mean, model.side_mean)
    assert "WARNING" in capsys.readouterr().err


def test_load_model_side_checkpoint_rejects_feature_count_mismatch(tmp_path):
    """A side block whose feature list disagrees with the checkpoint's S fails clearly."""
    pytest.importorskip("torch")
    xai = _load_xai_module()

    _, ckpt = _save_side_checkpoint(tmp_path)
    pre = _make_preprocessing([0.0] * 4, [1.0] * 4, ["a", "b", "c", "d"])
    pre["side"] = {"features": ["omega", "tpf"], "mean": [3000.0, 200.0], "std": [500.0, 40.0]}

    with pytest.raises(ValueError, match="side"):
        xai._load_model(str(ckpt), _model_args(), pre)


def test_load_model_non_side_checkpoint_takes_old_path(tmp_path):
    """A plain checkpoint still loads exactly as before (no side modules/buffers)."""
    torch = pytest.importorskip("torch")
    from pri2m.models.tkan_model import TKANModel

    xai = _load_xai_module()
    torch.manual_seed(0)
    model = TKANModel(input_size=4, hidden_size=8)
    ckpt = tmp_path / "best.pt"
    torch.save(model.state_dict(), ckpt)

    pre = _make_preprocessing([0.0] * 4, [1.0] * 4, ["a", "b", "c", "d"])
    loaded = xai._load_model(str(ckpt), _model_args(), pre)
    assert loaded.side_input is False
    assert loaded.hidden_size == 8
    assert loaded.input_size == 4
    assert not hasattr(loaded, "side_mean")


# --------------------------------------------------------------------------- #
# _load_model: quantile-severity checkpoints
#
# A quantile head is WIDER than the point head (Q columns instead of 1), so a
# quantile checkpoint cannot be reloaded by the default construction path. The
# count comes from the state_dict; the LEVELS themselves are unrecoverable from
# it and must come from preprocessing.json's "severity" block.
# --------------------------------------------------------------------------- #
def _save_quantile_checkpoint(tmp_path, *, quantiles=(0.1, 0.5, 0.9)):
    """Build a small quantile-severity TKANModel and save its state_dict."""
    torch = pytest.importorskip("torch")
    from pri2m.models.tkan_model import TKANModel

    torch.manual_seed(0)
    model = TKANModel(
        input_size=4, hidden_size=8,
        severity_mode="quantile", severity_quantiles=quantiles,
    )
    ckpt = tmp_path / "best.pt"
    torch.save(model.state_dict(), ckpt)
    return model, ckpt


def test_load_model_quantile_checkpoint_round_trips_with_severity_block(tmp_path):
    """A quantile checkpoint + a ``severity`` block reconstructs the exact model."""
    torch = pytest.importorskip("torch")
    xai = _load_xai_module()

    model, ckpt = _save_quantile_checkpoint(tmp_path, quantiles=(0.05, 0.5, 0.95))
    pre = _make_preprocessing([0.0] * 4, [1.0] * 4, ["a", "b", "c", "d"])
    pre["severity"] = {"mode": "quantile", "quantiles": [0.05, 0.5, 0.95]}

    loaded = xai._load_model(str(ckpt), _model_args(), pre)
    assert loaded.severity_mode == "quantile"
    assert loaded.severity.quantiles == (0.05, 0.5, 0.95)
    assert loaded.hidden_size == 8
    assert loaded.input_size == 4
    # Same weights AND the same 3-column severity output as the saved model.
    batch = {
        "window": torch.zeros(2, 32, 4),
        "tau": torch.full((2,), 4, dtype=torch.long),
    }
    with torch.no_grad():
        got, want = loaded(batch), model(batch)
    assert got["b_over_blim_quantiles"].shape == (2, 3)
    torch.testing.assert_close(got["b_over_blim_quantiles"], want["b_over_blim_quantiles"])


def test_load_model_quantile_checkpoint_rejects_level_count_mismatch(tmp_path):
    """A severity block whose level count disagrees with the checkpoint fails clearly."""
    pytest.importorskip("torch")
    xai = _load_xai_module()

    _, ckpt = _save_quantile_checkpoint(tmp_path)  # 3 columns
    pre = _make_preprocessing([0.0] * 4, [1.0] * 4, ["a", "b", "c", "d"])
    pre["severity"] = {"mode": "quantile", "quantiles": [0.5, 0.9]}

    with pytest.raises(ValueError, match="quantile"):
        xai._load_model(str(ckpt), _model_args(), pre)


def test_load_model_quantile_checkpoint_falls_back_to_default_levels(tmp_path, capsys):
    """No ``severity`` block -> the default levels + a loud warning (cf. side features)."""
    pytest.importorskip("torch")
    xai = _load_xai_module()

    _, ckpt = _save_quantile_checkpoint(tmp_path)  # 3 columns == the default
    pre = _make_preprocessing([0.0] * 4, [1.0] * 4, ["a", "b", "c", "d"])  # no "severity"

    loaded = xai._load_model(str(ckpt), _model_args(), pre)
    assert loaded.severity_mode == "quantile"
    assert loaded.severity.quantiles == (0.1, 0.5, 0.9)
    assert "WARNING" in capsys.readouterr().err


def test_load_model_point_checkpoint_keeps_default_severity_mode(tmp_path):
    """A plain (point) checkpoint still builds the huber head, unchanged."""
    pytest.importorskip("torch")
    import torch

    from pri2m.models.tkan_model import TKANModel

    xai = _load_xai_module()
    torch.manual_seed(0)
    model = TKANModel(input_size=4, hidden_size=8)
    ckpt = tmp_path / "best.pt"
    torch.save(model.state_dict(), ckpt)

    pre = _make_preprocessing([0.0] * 4, [1.0] * 4, ["a", "b", "c", "d"])
    loaded = xai._load_model(str(ckpt), _model_args(), pre)
    assert loaded.severity_mode == "huber"
    assert set(loaded({"window": torch.zeros(2, 32, 4),
                       "tau": torch.full((2,), 4, dtype=torch.long)})) == {
        "b_over_blim", "chatter_logit"
    }
