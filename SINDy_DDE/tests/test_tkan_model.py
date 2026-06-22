"""Tests for the assembled TKANModel (encoder trunk + heads).

Also a full data->model integration check: build synthetic NPZ + table + folds,
run the real fold loaders, and push a batch through TKANModel.
"""

from __future__ import annotations

import pytest
import torch

from pri2m.data.phase3_sequence import WindowConfig, build_fold_loaders
from pri2m.models.heads import Phase3Model
from pri2m.models.tkan_model import (
    DEFAULT_SIDE_FEATURES,
    FORBIDDEN_SIDE_FEATURES,
    TKANModel,
    avg_pool_time,
    scale_tau,
)
from pri2m.training.losses import LossWeights, severity_loss
from tests._phase3_synth import make_synth_folds, make_synth_npz_tree, make_synth_table


def _toy_batch(B=6, L=16, C=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    window = torch.randn(B, L, C, generator=g)
    # Make severity a learnable function of the window (mean absolute amplitude).
    target = window.abs().mean(dim=(1, 2))
    return {
        "window": window,
        "tau": torch.full((B,), 4, dtype=torch.long),
        "b_over_blim": target,
        "label": (target > target.median()).float(),
        "sample_weight": torch.ones(B),
    }


def _side_batch(B=6, L=16, C=4, seed=0):
    """Toy batch extended with the process-parameter side features."""
    batch = _toy_batch(B=B, L=L, C=C, seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    # O(1) values so the untrained side branch stays out of tanh saturation.
    batch["omega"] = torch.rand(B, generator=g) * 2.0 - 1.0
    batch["tpf"] = torch.rand(B, generator=g) * 2.0 - 1.0
    return batch


def test_forward_shapes_and_protocol():
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1)
    assert isinstance(model, Phase3Model)
    batch = _toy_batch()
    out = model(batch)
    assert set(out) == {"b_over_blim", "chatter_logit"}
    assert out["b_over_blim"].shape == (6,)
    assert out["chatter_logit"].shape == (6,)
    assert torch.isfinite(out["b_over_blim"]).all()


def test_backward_grads_finite():
    model = TKANModel(input_size=4, hidden_size=8)
    batch = _toy_batch()
    loss, _ = severity_loss(model(batch), batch, LossWeights())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads)


def test_trajectory_head_optional():
    model = TKANModel(input_size=4, hidden_size=8, with_trajectory=True,
                      traj_channels=(0, 2), traj_steps=5)
    out = model(_toy_batch())
    assert "trajectory" in out
    assert out["trajectory"].shape == (6, 5, 2)


def test_deterministic_under_seed():
    torch.manual_seed(123)
    m1 = TKANModel(input_size=4, hidden_size=8)
    torch.manual_seed(123)
    m2 = TKANModel(input_size=4, hidden_size=8)
    batch = _toy_batch(seed=1)
    with torch.no_grad():
        o1 = m1(batch)["b_over_blim"]
        o2 = m2(batch)["b_over_blim"]
    assert torch.allclose(o1, o2)


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1)
    batch = _toy_batch(B=6, L=16)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    first = None
    for _ in range(120):
        opt.zero_grad()
        loss, _ = severity_loss(model(batch), batch, LossWeights())
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
    assert float(loss.detach()) < 0.5 * first


def test_avg_pool_time_block_means():
    # [1, 4, 1] values 0,2,4,6; pool=2 -> block means [1, 5].
    w = torch.tensor([[[0.0], [2.0], [4.0], [6.0]]])
    out = avg_pool_time(w, 2)
    assert out.shape == (1, 2, 1)
    assert torch.allclose(out, torch.tensor([[[1.0], [5.0]]]))


def test_avg_pool_time_identity_when_pool_one():
    w = torch.randn(2, 8, 3)
    assert avg_pool_time(w, 1) is w


def test_avg_pool_time_trims_remainder():
    w = torch.randn(2, 10, 3)
    out = avg_pool_time(w, 4)  # 10 // 4 = 2 (first 8 steps used)
    assert out.shape == (2, 2, 3)


def test_scale_tau_rescales_and_clamps():
    tau = torch.tensor([8, 2, 4], dtype=torch.long)
    scaled = scale_tau(tau, 4)  # round(8/4)=2, round(2/4)=0->clamp 1, round(4/4)=1
    assert torch.equal(scaled, torch.tensor([2, 1, 1]))
    assert scale_tau(8, 4) == 2
    assert scale_tau(2, 4) == 1  # clamp to >= 1
    t = torch.tensor([5, 9])
    assert scale_tau(t, 1) is t  # identity at pool=1


def test_pool_feeds_downsampled_length_to_encoder():
    model = TKANModel(input_size=4, hidden_size=8, pool=4)
    seen = {}
    orig = model.encoder.forward

    def spy(seq, tau):
        seen["L"] = seq.shape[1]
        seen["tau"] = tau
        return orig(seq, tau)

    model.encoder.forward = spy
    model(_toy_batch(B=6, L=16, C=4))  # _toy_batch uses tau=4
    assert seen["L"] == 4             # 16 // 4
    assert int(seen["tau"][0]) == 1   # round(4 / 4)


def test_pool_default_runs_full_length():
    model = TKANModel(input_size=4, hidden_size=8)  # pool defaults to 1
    out = model(_toy_batch(B=6, L=16, C=4))
    assert out["b_over_blim"].shape == (6,)


def test_side_feature_constants():
    assert DEFAULT_SIDE_FEATURES == ("omega", "tpf", "tau")
    for key in ("b", "b_mm", "blim", "b_over_blim", "label", "sample_weight"):
        assert key in FORBIDDEN_SIDE_FEATURES


def test_default_model_has_no_side_modules():
    # Backward compat: side_input off must leave the state_dict untouched.
    model = TKANModel(input_size=4, hidden_size=8)
    names = [n for n, _ in model.named_parameters()]
    names += [n for n, _ in model.named_buffers()]
    assert not any("side" in n for n in names)


def test_side_input_forward_shapes_finite():
    model = TKANModel(input_size=4, hidden_size=8, side_input=True)
    out = model(_side_batch())
    assert out["b_over_blim"].shape == (6,)
    assert out["chatter_logit"].shape == (6,)
    assert torch.isfinite(out["b_over_blim"]).all()
    assert torch.isfinite(out["chatter_logit"]).all()


def test_side_input_forbidden_feature_raises():
    with pytest.raises(ValueError, match="blim"):
        TKANModel(input_size=4, hidden_size=8, side_input=True,
                  side_features=("omega", "blim"))


def test_side_input_missing_batch_key_raises():
    model = TKANModel(input_size=4, hidden_size=8, side_input=True)
    batch = _side_batch()
    del batch["omega"]
    with pytest.raises(ValueError, match="omega"):
        model(batch)


def test_side_input_ignores_targets_but_uses_features():
    # Leakage guard, functionally: targets must not reach the forward pass,
    # while a real side feature must (the branch is live, not dead weight).
    torch.manual_seed(7)
    model = TKANModel(input_size=4, hidden_size=8, side_input=True)
    model.eval()
    batch = _side_batch(seed=2)
    with torch.no_grad():
        base = model(batch)
        garbage = dict(batch)
        garbage["b_over_blim"] = torch.full_like(batch["b_over_blim"], 1e6)
        garbage["label"] = torch.full_like(batch["label"], -3.0)
        same = model(garbage)
        shifted = dict(batch)
        shifted["omega"] = batch["omega"] + 0.5
        diff = model(shifted)
    assert torch.equal(base["b_over_blim"], same["b_over_blim"])
    assert torch.equal(base["chatter_logit"], same["chatter_logit"])
    assert not torch.equal(base["b_over_blim"], diff["b_over_blim"])
    assert not torch.equal(base["chatter_logit"], diff["chatter_logit"])


def test_set_side_norm_applied_and_state_dict_roundtrip():
    torch.manual_seed(11)
    model = TKANModel(input_size=4, hidden_size=8, side_input=True)
    model.eval()
    batch = _side_batch(seed=3)
    with torch.no_grad():
        base = model(batch)["b_over_blim"]
    model.set_side_norm([1.0, 2.0, 3.0], [4.0, 5.0, 0.0])
    assert float(model.side_std[2]) == pytest.approx(1e-8)  # std floored
    with torch.no_grad():
        after = model(batch)["b_over_blim"]
    assert not torch.equal(base, after)  # normalization actually applied

    fresh = TKANModel(input_size=4, hidden_size=8, side_input=True)
    fresh.load_state_dict(model.state_dict())
    assert torch.equal(fresh.side_mean, model.side_mean)
    assert torch.equal(fresh.side_std, model.side_std)


def test_set_side_norm_length_mismatch_raises():
    model = TKANModel(input_size=4, hidden_size=8, side_input=True)
    with pytest.raises(ValueError):
        model.set_side_norm([0.0, 0.0], [1.0, 1.0])


def test_side_branch_grads_finite():
    model = TKANModel(input_size=4, hidden_size=8, side_input=True)
    batch = _side_batch()
    loss, _ = severity_loss(model(batch), batch, LossWeights())
    loss.backward()
    side_params = [(n, p) for n, p in model.named_parameters() if "side" in n]
    assert side_params, "no side-branch parameters found"
    for name, p in side_params:
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_data_layer_to_model_integration(tmp_path):
    root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2, 3), n_files=4,
                               length=256, seed=0)
    table = make_synth_table(root)
    folds = make_synth_folds(table, n_splits=3, seed=0)
    cfg = WindowConfig(length=64, stride=64)
    train_loader, val_loader, _ = build_fold_loaders(
        table, folds, fold_k=0, npz_root=root, cfg=cfg, batch_size=8, seed=0
    )
    model = TKANModel(input_size=len(cfg.channels), hidden_size=8)
    batch = next(iter(train_loader))
    out = model(batch)
    B = batch["window"].shape[0]
    assert out["b_over_blim"].shape == (B,)
    assert torch.isfinite(out["chatter_logit"]).all()
