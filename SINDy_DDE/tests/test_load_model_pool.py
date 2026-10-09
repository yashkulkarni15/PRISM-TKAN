"""``_load_model`` must restore the training-time temporal pool factor.

Regression test for the silent pool=1 bug: ``pool`` is a forward-time op, not a
state_dict entry, so ``xai_phase3._load_model`` built every checkpoint at the
``TKANModel`` default ``pool=1`` and ``load_state_dict`` raised nothing. The
canonical runs train with ``pool: 4``. Every caller that did not set
``model.pool`` itself (eval_heldout, score_files, mode/comb ablation, pruning,
symbolic substitution, recurrence, tkan_delay_heldout) scored a model fed a 4x
longer recurrence with an unscaled tau -- fold 0 val AUC 0.865 instead of the
training-time 0.960 on the identical 6152 files / 145 417 windows.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from pri2m.models.tkan_model import TKANModel

_REPO_ROOT = Path(__file__).resolve().parent.parent
_XAI_PATH = _REPO_ROOT / "scripts" / "xai_phase3.py"


def _load_xai_module():
    spec = importlib.util.spec_from_file_location("xai_phase3", _XAI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args():
    return argparse.Namespace(input_size=None, hidden_size=None, grid_size=5, spline_order=3)


def _checkpoint(tmp_path) -> tuple[Path, TKANModel]:
    torch.manual_seed(0)
    trained = TKANModel(input_size=4, hidden_size=8, pool=4)
    ckpt = tmp_path / "best_auc.pt"
    torch.save(trained.state_dict(), ckpt)
    return ckpt, trained


def test_pool_comes_from_preprocessing(tmp_path):
    xai = _load_xai_module()
    ckpt, _ = _checkpoint(tmp_path)
    model = xai._load_model(str(ckpt), _args(), {"pool": 4})
    assert model.pool == 4


def test_loaded_model_matches_the_trained_forward(tmp_path):
    xai = _load_xai_module()
    ckpt, trained = _checkpoint(tmp_path)
    trained.eval()
    model = xai._load_model(str(ckpt), _args(), {"pool": 4})
    torch.manual_seed(1)
    batch = {"window": torch.randn(3, 256, 4), "tau": torch.tensor([20, 33, 47])}
    with torch.no_grad():
        want = trained(batch)["chatter_logit"]
        got = model(batch)["chatter_logit"]
    assert torch.equal(want, got)


def test_no_preprocessing_keeps_the_old_default(tmp_path):
    xai = _load_xai_module()
    ckpt, _ = _checkpoint(tmp_path)
    assert xai._load_model(str(ckpt), _args(), None).pool == 1
