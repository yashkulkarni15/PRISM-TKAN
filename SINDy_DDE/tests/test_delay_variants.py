"""Tests for the matched TKAN delay test (``pri2m.xai.delay_variants``).

The wrapper must be a pure re-plumbing of :meth:`TKANModel.forward`: with the
true delay it has to give the wrapped model's output bit-for-bit, otherwise any
"drop" it later reports is partly the wrapper's own drift. The variants may
touch the delay the ENCODER sees and nothing else -- the side branch is fed
the raw batch ``tau`` and has to keep receiving it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pri2m.models.tkan_model import TKANModel, avg_pool_time, scale_tau
from pri2m.xai.delay_variants import (
    DelayVariant,
    encoder_tau,
    rescale_raw_tau,
)


def _model(side_input: bool, pool: int = 4) -> TKANModel:
    torch.manual_seed(0)
    m = TKANModel(input_size=4, hidden_size=6, pool=pool, side_input=side_input,
                  side_hidden=4)
    if side_input:
        m.set_side_norm([3000.0, 200.0, 50.0], [1000.0, 50.0, 20.0])
    m.eval()
    return m


def _batch(B: int = 5, L: int = 64) -> dict:
    g = torch.Generator().manual_seed(1)
    return {
        "window": torch.randn(B, L, 4, generator=g),
        "tau": torch.tensor([3, 8, 13, 22, 40], dtype=torch.long)[:B],
        "omega": torch.tensor([2000.0, 2500.0, 3000.0, 3500.0, 4000.0])[:B],
        "tpf": torch.tensor([150.0, 180.0, 200.0, 220.0, 260.0])[:B],
    }


# --------------------------------------------------------------------------- #
# tau arithmetic
# --------------------------------------------------------------------------- #
def test_rescale_raw_tau_rounds_half_even_and_clamps_to_one():
    tau = torch.tensor([1, 3, 5, 10, 41])
    out = rescale_raw_tau(tau, 0.5)
    # 0.5 -> 0 -> clamped to 1; 1.5 -> 2 (half-even); 2.5 -> 2; 5; 20.5 -> 20
    assert out.tolist() == [1, 2, 2, 5, 20]
    assert out.dtype == torch.long
    assert rescale_raw_tau(tau, 2.0).tolist() == [2, 6, 10, 20, 82]
    assert torch.equal(rescale_raw_tau(tau, 1.0), tau)


def test_encoder_tau_true_and_scale_one_equal_scale_tau():
    tau = torch.tensor([3, 8, 13, 22, 40])
    for pool in (1, 4):
        ref = scale_tau(tau, pool)
        assert torch.equal(encoder_tau(tau, pool, 16, "scale", 1.0), ref)


def test_encoder_tau_scale_applies_on_raw_then_pools():
    tau = torch.tensor([20, 41])
    # raw*2 = 40, 82 -> /4 -> 10, 20.5 -> 20 (half-even)
    assert encoder_tau(tau, 4, 64, "scale", 2.0).tolist() == [10, 20]


def test_encoder_tau_zero_exceeds_every_step():
    tau = torch.tensor([3, 8])
    out = encoder_tau(tau, 4, 16, "zero")
    assert out.tolist() == [17, 17]


def test_encoder_tau_rejects_unknown_mode():
    with pytest.raises(ValueError):
        encoder_tau(torch.tensor([3]), 4, 16, "shuffle")


# --------------------------------------------------------------------------- #
# DelayVariant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side_input", [False, True])
@pytest.mark.parametrize("pool", [1, 4])
def test_scale_one_is_bit_identical_to_wrapped_model(side_input, pool):
    m = _model(side_input, pool)
    b = _batch()
    with torch.no_grad():
        ref = m(b)
        out = DelayVariant(m, mode="scale", scale=1.0)(b)
    assert set(out) == set(ref)
    for k in ref:
        assert torch.equal(out[k], ref[k]), k


@pytest.mark.parametrize("side_input", [False, True])
def test_zero_equals_encoder_called_with_tau_at_least_L(side_input):
    m = _model(side_input)
    b = _batch()
    with torch.no_grad():
        w = avg_pool_time(b["window"], m.pool)
        L = w.shape[1]
        ctx, _ = m.encoder(w, torch.full((w.shape[0],), L, dtype=torch.long))
        sev = ctx
        if side_input:
            sev = torch.cat([ctx, m._side_context(b)], dim=-1)
        ref = m.severity(sev)
        out = DelayVariant(m, mode="zero")(b)
        base = m(b)
    for k in ref:
        assert torch.equal(out[k], ref[k]), k
    # And it is a real intervention, not a no-op.
    assert not torch.equal(out["chatter_logit"], base["chatter_logit"])


@pytest.mark.parametrize("mode,scale", [("zero", 1.0), ("scale", 0.5), ("scale", 2.0)])
def test_side_branch_receives_raw_tau(mode, scale):
    m = _model(side_input=True)
    b = _batch()
    seen = []
    orig = m._side_context

    def spy(batch):
        seen.append(batch["tau"].clone())
        return orig(batch)

    m._side_context = spy
    with torch.no_grad():
        DelayVariant(m, mode=mode, scale=scale)(b)
    assert len(seen) == 1
    assert torch.equal(seen[0], b["tau"])


def test_scale_variant_feeds_encoder_the_misstated_delay():
    m = _model(side_input=False)
    b = _batch()
    seen = []
    orig = m.encoder.forward

    def spy(seq, tau):
        seen.append(tau.clone())
        return orig(seq, tau)

    m.encoder.forward = spy
    with torch.no_grad():
        DelayVariant(m, mode="scale", scale=2.0)(b)
    expected = scale_tau(rescale_raw_tau(b["tau"], 2.0), 4)
    assert torch.equal(seen[0], expected)


def test_wrapper_exposes_attributes_the_evaluators_read():
    m = _model(side_input=True)
    v = DelayVariant(m, mode="zero")
    assert v.side_input is True
    assert v.pool == 4
    # Phase3Trainer.evaluate reads the device off the parameters.
    assert next(v.parameters()) is next(m.parameters())
    assert v.describe() == {"mode": "zero", "scale": None}
    assert DelayVariant(m, mode="scale", scale=0.5).describe() == {
        "mode": "scale", "scale": 0.5}


def test_wrapper_rejects_bad_arguments():
    m = _model(side_input=False)
    with pytest.raises(ValueError):
        DelayVariant(m, mode="bogus")
    with pytest.raises(ValueError):
        DelayVariant(m, mode="scale", scale=0.0)


def test_scalar_int_tau_is_accepted():
    m = _model(side_input=False)
    b = _batch()
    b["tau"] = 13
    with torch.no_grad():
        ref = m(b)
        out = DelayVariant(m, mode="scale", scale=1.0)(b)
    assert torch.equal(out["chatter_logit"], ref["chatter_logit"])


def test_delay_drop_and_floor_helpers():
    from pri2m.xai.delay_variants import delay_drop, exceeds_floor

    assert delay_drop(0.9, 0.8) == pytest.approx(0.1)
    assert delay_drop(None, 0.8) is None
    assert delay_drop(0.9, None) is None
    assert delay_drop(0.9, float("nan")) is None
    assert exceeds_floor(0.02, 0.0092) is True
    assert exceeds_floor(0.005, 0.0092) is False
    assert exceeds_floor(-0.02, 0.0092) is False
    assert exceeds_floor(None, 0.0092) is None
    assert np.isclose(delay_drop(0.95, 0.5), 0.45)


# --------------------------------------------------------------------------- #
# Runner: scripts/tkan_delay_heldout_phase3.py
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side_input", [False, True])
def test_tkan_delay_heldout_runner_end_to_end(tmp_path, side_input):
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for p in (str(root), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tkan_delay_heldout_phase3 as TD
    from tests.test_eval_heldout import _heldout_inputs, _smoke_train_checkpoint

    ckpt, pre = _smoke_train_checkpoint(tmp_path, side_input=side_input)
    held_root, csv = _heldout_inputs(tmp_path)
    out = tmp_path / "out"
    rc = TD.main(["--checkpoint", str(ckpt), "--features-csv", str(csv),
                  "--npz-root", str(held_root), "--output-dir", str(out),
                  "--tag", "t", "--batch-size", "16"])
    assert rc == 0
    man = json.loads((out / "tkan_delay_heldout_t.json").read_text())
    assert man["complete"] is True and man["floor"] == 0.0092
    assert man["provenance"]["side_input"] is side_input
    assert set(man["variants"]) == {"zero_delay", "tau_x0.5", "tau_x2"}
    assert man["variants"]["zero_delay"]["scale"] is None
    assert man["variants"]["tau_x2"]["scale"] == 2.0
    for v in man["variants"].values():
        assert v["error"] is None
        if v["auc"] is None or man["baseline_auc"] is None:
            assert v["drop"] is None and v["exceeds_floor"] is None
        else:
            assert v["drop"] == pytest.approx(man["baseline_auc"] - v["auc"])
            assert isinstance(v["exceeds_floor"], bool)
