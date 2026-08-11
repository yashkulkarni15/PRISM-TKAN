"""The severity point mode is named after the loss it actually uses.

``_VALID_SEVERITY_MODES`` called the point-regression mode ``"huber"`` while the
point path calls :func:`weighted_mse` and reports the part as ``"sev_mse"``.
There is no Huber or smooth-L1 loss anywhere in ``pri2m.training.losses``. The
name was harmless at runtime and would have put a false loss in the methods
section of the report, which is why it is being fixed before the writing starts
rather than after.

Canonical name is ``"point"`` rather than ``"mse"``: it is the sibling of
``"quantile"``, and both then describe *what the head emits* (a point estimate
vs a set of quantile columns) rather than mixing one loss name with one output
name. The loss is stated separately, where it can be stated accurately.

``"huber"`` stays accepted as a deprecated alias. Every config, sweep and
``summary.json`` on disk uses it, and rejecting it would strand the validated
runs for a cosmetic gain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pri2m.models.heads import (
    DEFAULT_SEVERITY_MODE,
    _VALID_SEVERITY_MODES,
    SeverityHead,
    normalise_severity_mode,
)
from pri2m.models.tkan_model import TKANModel
from pri2m.training.losses import LossWeights, severity_loss, weighted_mse
from pri2m.training.phase3_trainer import TrainConfig


# --------------------------------------------------------------------------- #
# The canonical name
# --------------------------------------------------------------------------- #
def test_point_is_the_canonical_mode_name():
    assert DEFAULT_SEVERITY_MODE == "point"
    assert _VALID_SEVERITY_MODES == ("point", "quantile")


def test_the_false_loss_name_is_no_longer_a_valid_mode():
    """Regression guard: it must not creep back as a canonical spelling."""
    assert "huber" not in _VALID_SEVERITY_MODES


@pytest.mark.parametrize("given,expected", [
    ("point", "point"),
    ("quantile", "quantile"),
    ("huber", "point"),      # deprecated alias
])
def test_normalise_maps_every_accepted_spelling(given, expected):
    assert normalise_severity_mode(given) == expected


def test_normalise_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="severity_mode"):
        normalise_severity_mode("smooth_l1")


# --------------------------------------------------------------------------- #
# The name now matches the arithmetic
# --------------------------------------------------------------------------- #
def test_point_mode_regression_term_really_is_weighted_mse():
    """The point of the rename, asserted on behaviour rather than on a string.

    If the point path ever grows an actual Huber loss, this fails and the name
    has to be revisited -- which is the situation the old name pretended was
    already true.
    """
    torch.manual_seed(0)
    out = {"b_over_blim": torch.randn(8), "chatter_logit": torch.randn(8)}
    batch = {
        "b_over_blim": torch.randn(8),
        "label": torch.randint(0, 2, (8,)).float(),
        "sample_weight": torch.rand(8) + 0.1,
    }

    _, parts = severity_loss(out, batch, LossWeights())

    assert "sev_mse" in parts, "the point head must report an MSE part"
    expected = weighted_mse(out["b_over_blim"], batch["b_over_blim"],
                            batch["sample_weight"])
    assert parts["sev_mse"] == pytest.approx(float(expected))


# --------------------------------------------------------------------------- #
# The alias keeps every artifact on disk loadable
# --------------------------------------------------------------------------- #
def test_head_defaults_to_point():
    assert SeverityHead(16).severity_mode == "point"


def test_head_accepts_the_deprecated_alias_and_normalises_it():
    head = SeverityHead(16, severity_mode="huber")

    assert head.severity_mode == "point", "the alias must normalise, not persist"


def test_alias_and_canonical_build_an_identical_head():
    torch.manual_seed(0)
    aliased = SeverityHead(16, severity_mode="huber")
    torch.manual_seed(0)
    canonical = SeverityHead(16, severity_mode="point")

    x = torch.randn(4, 16)
    a, c = aliased(x), canonical(x)

    assert a.keys() == c.keys()
    for k in a:
        assert torch.equal(a[k], c[k]), f"{k} differs between alias and canonical"


def test_tkan_model_defaults_to_point():
    assert TKANModel(input_size=4, hidden_size=8).severity_mode == "point"


def test_tkan_model_normalises_the_alias():
    model = TKANModel(input_size=4, hidden_size=8, severity_mode="huber")

    assert model.severity_mode == "point", (
        "the trainer and the XAI loader read this attribute back; it must be "
        "canonical even though SeverityHead normalises its own copy"
    )


def test_train_config_defaults_to_point():
    assert TrainConfig().severity_mode == "point"


def test_train_config_normalises_the_alias():
    """Not every caller goes through the CLI.

    ``_build_train_config`` normalises, but a ``TrainConfig`` built directly --
    from a stored config dict, a notebook, or a test -- would otherwise keep the
    raw string and be rejected by ``build_model``'s membership check, which is
    exactly the "strand the validated runs" outcome the alias exists to prevent.
    """
    assert TrainConfig(severity_mode="huber").severity_mode == "point"


def test_a_directly_built_legacy_config_still_builds_a_model():
    from pri2m.training.phase3_trainer import build_model

    model = build_model(TrainConfig(model="tkan", severity_mode="huber", hidden_size=8))

    assert model.severity_mode == "point"


def test_an_unknown_mode_is_still_rejected_by_build_model():
    """Normalising must not turn the validation into a silent pass-through."""
    from pri2m.training.phase3_trainer import build_model

    with pytest.raises(ValueError, match="severity_mode"):
        build_model(TrainConfig(model="tkan", severity_mode="bogus", hidden_size=8))


# --------------------------------------------------------------------------- #
# Config surface
# --------------------------------------------------------------------------- #
def _load_train_script_module():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "train_phase3.py"
    spec = importlib.util.spec_from_file_location("_train_phase3_naming", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_args(**overrides):
    import argparse

    base = dict(
        model=None, fold=None, device=None, num_workers=None, compile=None,
        smoke=False, lr_schedule=None, warmup_epochs=None, grad_clip=None,
        optimizer=None, weight_decay=None, side_input=None, side_hidden=None,
        severity_mode=None, deterministic=None, epochs=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_a_yaml_written_before_the_rename_still_builds():
    """Every config and summary.json on disk says "huber"."""
    mod = _load_train_script_module()

    cfg = mod._build_train_config({"severity_mode": "huber"}, _cli_args())

    assert cfg.severity_mode == "point"


def test_the_alias_is_accepted_on_the_command_line_too():
    mod = _load_train_script_module()

    cfg = mod._build_train_config({}, _cli_args(severity_mode="huber"))

    assert cfg.severity_mode == "point"


def test_shipped_config_uses_the_canonical_name():
    from pri2m.config import load_config

    root = Path(__file__).resolve().parents[1]
    cfg_dict = load_config(str(root / "configs" / "phase3_tkan.yaml"))

    assert cfg_dict["severity_mode"] == "point"


# --------------------------------------------------------------------------- #
# The trap this rename sets, closed
# --------------------------------------------------------------------------- #
def test_default_mode_still_omits_the_severity_block(tmp_path):
    """The preprocessing gate compared against the literal "huber".

    Renaming the default without moving that comparison would make every
    ordinary run start emitting a ``severity`` block, silently changing the
    schema of the default path -- and the XAI loader treats an absent block as
    "the default point head", so the two must stay in agreement.
    """
    from pri2m.training import phase3_trainer as T

    src = Path(T.__file__).read_text()
    assert '!= "huber"' not in src, (
        "the preprocessing gate still compares against the old literal; it must "
        "compare against DEFAULT_SEVERITY_MODE"
    )
