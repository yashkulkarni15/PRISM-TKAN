"""Faithfulness checks for the TKAN white-box XAI (do the explanations reflect
the model?).

Four rock-solid invariants are exercised here -- no noisy monotonicity claims:

1. :func:`prune_context` is a *reversible* edit: it zeroes the requested
   ``base_weight``/``spline_weight`` edges on enter and restores the model
   byte-for-byte on exit.
2. :func:`pruning_curve` keeping ALL edges reproduces the unpruned AUC, while
   keeping the smallest fraction yields a genuinely different AUC; every AUC is
   finite and in ``[0, 1]``.
3. :func:`randomization_sanity` -- attributions on a trained model and on a
   re-initialized copy genuinely differ (cosine well below 1).
4. :func:`deletion_curve` -- with a stub whose output is the sum of a few
   window cells and an attribution that points exactly at those cells, deleting
   them drives the prediction toward zero.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pri2m.models.kan import KANLinear
from pri2m.models.tkan_model import TKANModel
from pri2m.xai import faithfulness


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _model(seed: int = 0) -> TKANModel:
    torch.manual_seed(seed)
    return TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)


def _batch(B: int = 8, L: int = 24, C: int = 4, tau: int = 5, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {
        "window": torch.randn(B, L, C),
        "tau": torch.full((B,), tau, dtype=torch.long),
    }


def _labels(B: int = 8, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, 2, (B,))


# --------------------------------------------------------------------------- #
# 1. prune_context restores EXACTLY
# --------------------------------------------------------------------------- #
def test_prune_context_zeroes_then_restores_exactly():
    model = _model()
    # Snapshot every parameter (cloned so later in-place edits cannot alias it).
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    edges = {
        "encoder.layers.0.kan_z": [(0, 0), (2, 3)],
        "severity.cls_head": [(0, 1)],
    }
    modules = dict(model.named_modules())

    with faithfulness.prune_context(model, edges):
        # Inside the context the requested edges must be zero on BOTH paths.
        for name, cells in edges.items():
            mod = modules[name]
            for o, i in cells:
                assert torch.equal(mod.base_weight[o, i], torch.zeros(()))
                assert torch.equal(
                    mod.spline_weight[o, i, :],
                    torch.zeros_like(mod.spline_weight[o, i, :]),
                )

    # After the context every parameter is restored byte-for-byte.
    restored = model.state_dict()
    assert set(restored) == set(snapshot)
    for k in snapshot:
        assert torch.equal(restored[k], snapshot[k]), f"param {k} not restored"


def test_prune_context_restores_even_on_exception():
    model = _model()
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    edges = {"encoder.layers.0.kan_z": [(1, 1)]}

    with pytest.raises(RuntimeError, match="boom"):
        with faithfulness.prune_context(model, edges):
            raise RuntimeError("boom")

    for k, v in model.state_dict().items():
        assert torch.equal(v, snapshot[k]), f"param {k} not restored after exception"


# --------------------------------------------------------------------------- #
# 2. rank_edges
# --------------------------------------------------------------------------- #
def test_rank_edges_sorted_desc_and_complete():
    from pri2m.xai.activations import collect_activations

    model = _model()
    acts = collect_activations(model, _batch())
    ranked = faithfulness.rank_edges(acts)

    # One entry per (kan, o, i) edge across all KANLinears.
    total_edges = sum(v["importance"].size for v in acts.values())
    assert len(ranked) == total_edges

    importances = [r[3] for r in ranked]
    assert importances == sorted(importances, reverse=True)
    # Each entry is (kan_name, o, i, importance) with valid indices.
    for name, o, i, imp in ranked:
        assert name in acts
        out_f, in_f = acts[name]["importance"].shape
        assert 0 <= o < out_f and 0 <= i < in_f
        assert np.isfinite(imp)


# --------------------------------------------------------------------------- #
# 3. pruning_curve
# --------------------------------------------------------------------------- #
def test_pruning_curve_keep_all_matches_baseline_and_smallest_differs():
    from sklearn.metrics import roc_auc_score

    from pri2m.xai.activations import collect_activations

    model = _model()
    batch = _batch()
    labels = _labels()
    acts = collect_activations(model, batch)

    # Baseline AUC computed independently of the curve machinery.
    model.eval()
    with torch.no_grad():
        logit = model(batch)["chatter_logit"]
    proba = torch.sigmoid(logit).numpy()
    baseline_auc = roc_auc_score(labels.numpy(), proba)

    fractions = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
    result = faithfulness.pruning_curve(model, batch, labels, acts, fractions=fractions)

    assert list(result["fraction_kept"]) == list(fractions)
    assert len(result["auc"]) == len(fractions)

    aucs = result["auc"]
    assert all(np.isfinite(a) for a in aucs)
    assert all(0.0 <= a <= 1.0 for a in aucs)

    # Keeping ALL edges (fraction 1.0) must reproduce the unpruned AUC exactly.
    assert abs(aucs[-1] - baseline_auc) < 1e-6
    # Pruning to the smallest fraction must change the AUC (not all identical).
    assert not np.allclose(aucs, aucs[0])


def test_pruning_curve_does_not_mutate_model():
    from pri2m.xai.activations import collect_activations

    model = _model()
    batch = _batch()
    labels = _labels()
    acts = collect_activations(model, batch)
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    faithfulness.pruning_curve(model, batch, labels, acts)

    for k, v in model.state_dict().items():
        assert torch.equal(v, snapshot[k]), f"pruning_curve mutated param {k}"


# --------------------------------------------------------------------------- #
# 4. deletion_curve (directional, with a stub model)
# --------------------------------------------------------------------------- #
class _SumStub(torch.nn.Module):
    """Output = sum of window cells (t in {3,4,5}, channel 0)."""

    def forward(self, batch: dict) -> dict:
        s = batch["window"][:, [3, 4, 5], 0].sum(dim=1)  # [B]
        return {"chatter_logit": s, "b_over_blim": s}


def test_deletion_curve_drives_prediction_to_zero():
    L, C = 12, 4
    window = torch.ones(L, C)  # positive everywhere
    attr = np.zeros((L, C), dtype=np.float32)
    attr[[3, 4, 5], 0] = 1.0  # exactly the cells the stub sums

    model = _SumStub()
    result = faithfulness.deletion_curve(
        model, window, attr, tau=1, n_steps=6, target="chatter_logit"
    )

    preds = result["pred"]
    fracs = result["fraction_deleted"]
    assert len(preds) == len(fracs)
    assert all(np.isfinite(p) for p in preds)
    assert (np.diff(fracs) >= 0).all()  # cumulative deletion
    # Deleting the highest-|attr| cells (the summed ones) drives output toward 0.
    assert abs(preds[-1]) < abs(preds[0])
    assert abs(preds[-1]) < 1e-6  # those three cells fully removed -> exactly 0


# --------------------------------------------------------------------------- #
# 5. randomization_sanity
# --------------------------------------------------------------------------- #
def test_randomization_sanity_trained_vs_reinit_differ():
    pytest.importorskip("captum")

    torch.manual_seed(7)
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    # Give the model NON-trivial spline weights so it is not near the init.
    torch.manual_seed(7)
    for mod in model.modules():
        if isinstance(mod, KANLinear):
            mod.spline_weight.data.normal_(0.0, 0.5)

    window = torch.randn(24, 4)
    out = faithfulness.randomization_sanity(
        model, window, tau=2, steps=24, target="b_over_blim", seed=0
    )
    cos = out["cosine"]
    assert isinstance(cos, float)
    assert -1.0 <= cos <= 1.0
    # Trained vs. re-initialized attributions must genuinely differ.
    assert cos < 0.99


def test_randomization_sanity_does_not_mutate_model():
    pytest.importorskip("captum")

    model = _model(seed=3)
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    window = torch.randn(24, 4)
    faithfulness.randomization_sanity(model, window, tau=1, steps=16, seed=0)
    for k, v in model.state_dict().items():
        assert torch.equal(v, snapshot[k]), f"randomization_sanity mutated param {k}"


# --------------------------------------------------------------------------- #
# 6. plot helpers (Agg; return the path)
# --------------------------------------------------------------------------- #
def test_plot_pruning_curve_writes_file(tmp_path):
    result = {"fraction_kept": [0.1, 0.5, 1.0], "auc": [0.6, 0.8, 0.9]}
    out = tmp_path / "pruning.png"
    returned = faithfulness.plot_pruning_curve(result, str(out))
    assert returned == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_deletion_curve_writes_file(tmp_path):
    result = {"fraction_deleted": [0.0, 0.5, 1.0], "pred": [3.0, 1.5, 0.0]}
    out = tmp_path / "deletion.png"
    returned = faithfulness.plot_deletion_curve(result, str(out))
    assert returned == str(out)
    assert out.exists() and out.stat().st_size > 0
