"""End-to-end smoke for the comb probe, against a real (tiny) checkpoint.

This exists because of a specific failure: ``xai_phase3._load_model`` takes an
``argparse.Namespace`` and reads four attributes off it, so a namespace missing
one raises ``AttributeError`` *inside the loader*. On a cluster that is a job
which queues, starts, and dies -- GL job 56991584 did exactly that, in seconds,
after waiting in a queue. Nothing in the unit tests touched that seam.

The test therefore builds a genuine ``TKANModel``, saves its ``state_dict``, and
drives the runner end to end on CPU. It is slower than a mocked test and that is
the point: the contract being checked is precisely the one a mock would have
asserted into existence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import comb_probe_phase3 as P  # noqa: E402
from pri2m.models.tkan_model import TKANModel  # noqa: E402


@pytest.fixture
def checkpoint_tree(tmp_path):
    """A real 4-channel TKAN checkpoint plus the manifest naming its edges."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    # hidden_size 64 is the production width, and it has to be: the delayed
    # channels live at in=68..71 (xai/semantics.py), which simply do not exist on
    # a narrower model. A smaller fixture would pass while exercising different
    # indices than every stored manifest refers to.
    model = TKANModel(input_size=4, hidden_size=64)
    ckpt = run / "checkpoints" / "best_auc.pt"
    torch.save(model.state_dict(), ckpt)
    (run / "preprocessing.json").write_text(json.dumps({"norm": {}}))

    # in=0 is xtpos, in=68 its tau-delayed copy (see xai/semantics.py).
    pair = {"corr": 0.99, "gain": 1.0, "weight": 1.0,
            "kan": "encoder.layers.0.kan_z", "channel": "xtpos",
            "out": 1, "in_inst": 0, "in_delay": 68, "degenerate": 0}
    regen = tmp_path / "regenerative.json"
    regen.write_text(json.dumps({
        "provenance": {"checkpoint": str(ckpt)},
        "rows": [pair], "control_rows": [], "top_pairs": [pair],
    }))
    return ckpt, regen, tmp_path


def test_load_model_args_carries_every_attribute_the_loader_reads():
    """The regression guard for GL job 56991584.

    Asserted against the loader's real source rather than a remembered list, so
    it fails here if ``_load_model`` ever grows another attribute.
    """
    import argparse
    import re

    ns = P._load_model_args(argparse.Namespace())

    src = (_ROOT / "scripts" / "xai_phase3.py").read_text()
    body = src[src.index("def _load_model("):src.index("def _sample_window(")]
    required = set(re.findall(r"args\.([a-z_]+)", body))

    missing = required - set(vars(ns))
    assert not missing, f"_load_model reads {missing}, which the probe never sets"


def test_probe_runs_end_to_end_on_a_real_checkpoint(checkpoint_tree, tmp_path):
    ckpt, regen, _ = checkpoint_tree

    rc = P.main([
        "--checkpoint", str(ckpt), "--regen-manifest", str(regen),
        "--n-freqs", "41", "--n-pairs", "1", "--tau", "20",
        "--output-dir", str(tmp_path / "out"), "--tag", "smoke", "--no-figures",
    ])

    assert rc == 0
    out = json.loads((tmp_path / "out" / "comb_probe_smoke.json").read_text())
    assert out["complete"] is True
    assert "error" not in out["trained"][0], out["trained"][0].get("error")
    assert len(out["trained"][0]["measured"]) == 41


def test_probe_carries_a_genuine_untrained_control(checkpoint_tree, tmp_path):
    """A control that did not actually reinitialise is not a control."""
    ckpt, regen, _ = checkpoint_tree

    P.main([
        "--checkpoint", str(ckpt), "--regen-manifest", str(regen),
        "--n-freqs", "21", "--n-pairs", "1", "--tau", "20",
        "--output-dir", str(tmp_path / "out"), "--tag", "smoke", "--no-figures",
    ])

    out = json.loads((tmp_path / "out" / "comb_probe_smoke.json").read_text())
    assert out["provenance"]["untrained_mean_weight_diff"] > 0
    assert "error" not in out["untrained"][0]


def test_probe_refuses_a_missing_checkpoint(checkpoint_tree, tmp_path):
    _, regen, _ = checkpoint_tree

    rc = P.main(["--checkpoint", str(tmp_path / "nope.pt"),
                 "--regen-manifest", str(regen), "--no-figures"])

    assert rc == 2
