"""Unit tests for the Phase 3 TKAN cell / encoder and optional sigKAN path.

Run: ``python -m pytest tests/test_tkan.py -v`` from the repo root.

Conventions: tiny tensors (B<=8, L<=32, C=4, H<=16), fixed seeds, finite-grad
checks. ``conftest.py`` puts ``src/`` on ``sys.path`` so ``import pri2m...`` works.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pri2m.models.tkan import (
    SignatureWeighting,
    TKANCell,
    TKANEncoder,
    delayed_input,
    path_signature,
)


# --------------------------------------------------------------------------- #
# delayed_input
# --------------------------------------------------------------------------- #
def test_delayed_input_scalar_tau_zeros_before_then_exact_slice():
    """Scalar tau=2: t<2 -> zeros; t>=2 -> exactly seq[:, t-2, :]."""
    torch.manual_seed(0)
    B, L, C = 3, 6, 4
    seq = torch.randn(B, L, C)
    tau = 2

    # Rows where (t - tau) < 0 must be all zeros.
    assert torch.equal(delayed_input(seq, 0, tau), torch.zeros(B, C))
    assert torch.equal(delayed_input(seq, 1, tau), torch.zeros(B, C))

    # t >= tau -> the exact delayed slice.
    for t in range(2, L):
        out = delayed_input(seq, t, tau)
        assert out.shape == (B, C)
        assert torch.equal(out, seq[:, t - tau, :])


def test_delayed_input_per_sample_tau_longtensor():
    """Per-sample tau LongTensor([1, 3]): each row delayed by its own tau."""
    torch.manual_seed(1)
    B, L, C = 2, 6, 4
    seq = torch.randn(B, L, C)
    tau = torch.tensor([1, 3], dtype=torch.long)

    # t=0: sample0 wants index -1 -> zeros; sample1 wants -3 -> zeros.
    out0 = delayed_input(seq, 0, tau)
    assert torch.equal(out0, torch.zeros(B, C))

    # t=2: sample0 -> index 1 (valid); sample1 -> index -1 -> zeros.
    out2 = delayed_input(seq, 2, tau)
    assert torch.equal(out2[0], seq[0, 1, :])
    assert torch.equal(out2[1], torch.zeros(C))

    # t=4: sample0 -> index 3; sample1 -> index 1; both valid.
    out4 = delayed_input(seq, 4, tau)
    assert torch.equal(out4[0], seq[0, 3, :])
    assert torch.equal(out4[1], seq[1, 1, :])


def test_delayed_input_per_sample_tau_dtype_and_shape():
    """Output dtype/shape preserved with a per-sample tau."""
    seq = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    tau = torch.tensor([0, 2], dtype=torch.long)
    out = delayed_input(seq, 3, tau)
    assert out.shape == (2, 4)
    assert out.dtype == seq.dtype
    # tau=0 means no delay -> current step.
    assert torch.equal(out[0], seq[0, 3, :])
    assert torch.equal(out[1], seq[1, 1, :])


# --------------------------------------------------------------------------- #
# TKANCell
# --------------------------------------------------------------------------- #
def test_tkancell_forward_shape_with_delay():
    torch.manual_seed(0)
    B, in_f, H = 4, 4, 8
    cell = TKANCell(in_f, H, use_delay=True)
    x_t = torch.randn(B, in_f)
    h_prev = torch.zeros(B, H)
    x_delay = torch.randn(B, in_f)
    h = cell(x_t, h_prev, x_delay)
    assert h.shape == (B, H)
    assert torch.isfinite(h).all()


def test_tkancell_forward_without_delay():
    """use_delay=False cell works with x_delay omitted."""
    torch.manual_seed(0)
    B, in_f, H = 4, 4, 8
    cell = TKANCell(in_f, H, use_delay=False)
    x_t = torch.randn(B, in_f)
    h_prev = torch.zeros(B, H)
    h = cell(x_t, h_prev)
    assert h.shape == (B, H)
    assert torch.isfinite(h).all()


def test_tkancell_delay_true_but_xdelay_none_falls_back():
    """A use_delay=True cell must tolerate x_delay=None (zeros substitute)."""
    torch.manual_seed(0)
    B, in_f, H = 2, 4, 6
    cell = TKANCell(in_f, H, use_delay=True)
    x_t = torch.randn(B, in_f)
    h_prev = torch.zeros(B, H)
    h = cell(x_t, h_prev, None)
    assert h.shape == (B, H)
    assert torch.isfinite(h).all()


def test_tkancell_backward_grads_finite_nonzero():
    torch.manual_seed(0)
    B, in_f, H = 3, 4, 8
    cell = TKANCell(in_f, H, use_delay=True)
    x_t = torch.randn(B, in_f)
    h_prev = torch.randn(B, H)
    x_delay = torch.randn(B, in_f)
    out = cell(x_t, h_prev, x_delay).sum()
    out.backward()
    grads = [p.grad for p in cell.parameters() if p.grad is not None]
    assert len(grads) > 0
    total = sum(g.abs().sum().item() for g in grads)
    assert np.isfinite(total)
    assert total > 0.0


# --------------------------------------------------------------------------- #
# TKANEncoder
# --------------------------------------------------------------------------- #
def test_tkanencoder_forward_shapes_scalar_tau():
    torch.manual_seed(0)
    B, L, C, H = 2, 16, 4, 8
    enc = TKANEncoder(C, H, num_layers=1)
    seq = torch.randn(B, L, C)
    context, all_hidden = enc(seq, tau=4)
    assert context.shape == (B, H)
    assert all_hidden.shape == (B, L, H)
    assert torch.isfinite(context).all()
    assert torch.isfinite(all_hidden).all()
    # context is the top layer's final hidden state.
    assert torch.allclose(context, all_hidden[:, -1, :])


def test_tkanencoder_backward_finite_nonzero_grads():
    torch.manual_seed(0)
    B, L, C, H = 2, 16, 4, 8
    enc = TKANEncoder(C, H, num_layers=1)
    seq = torch.randn(B, L, C)
    context, _ = enc(seq, tau=4)
    loss = context.sum()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert len(grads) > 0
    total = sum(g.abs().sum().item() for g in grads)
    assert np.isfinite(total)
    assert total > 0.0


def test_tkanencoder_determinism_same_seed():
    B, L, C, H = 2, 16, 4, 8
    seq_seed = 7

    def run():
        torch.manual_seed(123)
        enc = TKANEncoder(C, H, num_layers=1)
        g = torch.Generator().manual_seed(seq_seed)
        seq = torch.randn(B, L, C, generator=g)
        ctx, _ = enc(seq, tau=4)
        return ctx

    c1 = run()
    c2 = run()
    assert torch.allclose(c1, c2)


def test_tkanencoder_num_layers_2_shapes():
    torch.manual_seed(0)
    B, L, C, H = 2, 16, 4, 8
    enc = TKANEncoder(C, H, num_layers=2)
    seq = torch.randn(B, L, C)
    context, all_hidden = enc(seq, tau=4)
    assert context.shape == (B, H)
    assert all_hidden.shape == (B, L, H)
    assert torch.isfinite(context).all()
    assert torch.isfinite(all_hidden).all()


def test_tkanencoder_per_sample_tau():
    """Encoder accepts a per-sample LongTensor tau."""
    torch.manual_seed(0)
    B, L, C, H = 2, 16, 4, 8
    enc = TKANEncoder(C, H, num_layers=1)
    seq = torch.randn(B, L, C)
    tau = torch.tensor([2, 5], dtype=torch.long)
    context, all_hidden = enc(seq, tau)
    assert context.shape == (B, H)
    assert all_hidden.shape == (B, L, H)
    assert torch.isfinite(context).all()


def test_tkanencoder_delay_wiring_via_first_layer_cell_hook():
    """With use_delay=True, the layer-0 cell must receive x_delay equal to
    ``delayed_input(seq, t, tau)`` at every step (zeros for t<tau).

    We capture the x_delay argument passed into the layer-0 cell with a forward
    pre-hook and compare against the reference ``delayed_input``.
    """
    torch.manual_seed(0)
    B, L, C, H = 2, 8, 4, 6
    tau = 3
    enc = TKANEncoder(C, H, num_layers=1, use_delay=True)
    seq = torch.randn(B, L, C)

    captured: list[torch.Tensor | None] = []

    def pre_hook(_module, args, kwargs):
        # forward(x_t, h_prev, x_delay=None): x_delay may be positional or kw.
        if len(args) >= 3:
            captured.append(args[2])
        else:
            captured.append(kwargs.get("x_delay"))

    handle = enc.layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        enc(seq, tau=tau)
    finally:
        handle.remove()

    assert len(captured) == L
    for t in range(L):
        xd = captured[t]
        assert xd is not None, f"layer-0 cell got x_delay=None at t={t}"
        expected = delayed_input(seq, t, tau)
        assert torch.equal(xd, expected), f"x_delay mismatch at t={t}"


# --------------------------------------------------------------------------- #
# Optional sigKAN (skips cleanly if iisignature absent)
# --------------------------------------------------------------------------- #
def test_path_signature_dimension():
    iisignature = pytest.importorskip("iisignature")
    L, C, depth = 12, 3, 2
    window = np.random.default_rng(0).standard_normal((L, C)).astype(np.float64)
    sig = path_signature(window, depth)
    assert isinstance(sig, np.ndarray)
    assert sig.ndim == 1
    # Documented length: iisignature.siglength(C, depth).
    assert sig.shape[0] == iisignature.siglength(C, depth)
    assert np.isfinite(sig).all()


def test_path_signature_raises_without_iisignature_when_absent():
    """When iisignature is NOT installed, path_signature raises ImportError."""
    try:
        import iisignature  # noqa: F401

        pytest.skip("iisignature installed; cannot test the absent branch")
    except ImportError:
        pass
    window = np.zeros((5, 3), dtype=np.float64)
    with pytest.raises(ImportError):
        path_signature(window, 2)


def test_signature_weighting_gates_context():
    iisignature = pytest.importorskip("iisignature")
    torch.manual_seed(0)
    B, H, C, depth = 4, 8, 3, 2
    sig_dim = iisignature.siglength(C, depth)
    sw = SignatureWeighting(sig_dim=sig_dim, hidden_size=H)
    context = torch.randn(B, H)
    sig_feats = torch.randn(B, sig_dim)
    out = sw(context, sig_feats)
    assert out.shape == (B, H)
    assert torch.isfinite(out).all()
