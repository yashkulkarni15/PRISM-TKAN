"""U14-B: do recurring activation *modes* exist, beyond random initialisation?

These tests pin the **science guards**, not just the plumbing. The module they
cover exists because two Phase 3 statistics already collapsed under exactly this
kind of control (the TSR delay-lock scored higher on an untrained net; the
band-notch contrast reported a solver failure as a measurement). So the tests
assert, in order of importance:

1. ``edge_function(component="full")`` is *byte-identical* to the pre-change
   behaviour -- nothing already validated may move.
2. The base path is a one-parameter family, which is *why* clustering runs on
   the spline component only.
3. The importance gate runs on raw, data-aware importance and cannot be applied
   after shape normalisation.
4. Matched-N actually matches N when the two arms pass wildly different counts.
5. A structureless (randomly initialised) arm does **not** clear the verdict
   margin, and a deliberately structured one does -- a control that always says
   "artifact" would be as useless as no control at all.
6. A missing measurement returns UNDECIDED, which is never a pass.
7. Hidden-fed edges are refused as index-comparable, because hidden-unit indices
   are permutation-arbitrary across independently trained folds.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from pri2m.models.kan import KANLinear
from pri2m.models.tkan_model import TKANModel
from pri2m.xai import activation_modes as AM
from pri2m.xai.activations import collect_activations
from pri2m.xai.kan_native import _iter_kanlinear
from pri2m.xai.semantics import edge_index_map

# --------------------------------------------------------------------------- #
# Fixtures: small synthetic models. No checkpoints exist on local disk, so every
# test builds its own network -- see tests/_phase3_synth.py for the data-side
# equivalent (not needed here: these tests feed the model tensors directly).
# --------------------------------------------------------------------------- #

_IN, _HID = 4, 8


def _batch(B: int = 4, L: int = 24, C: int = _IN, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(L, dtype=torch.float32)
    base = torch.sin(2 * np.pi * 0.05 * t)
    window = base.view(1, L, 1).repeat(B, 1, C) + 0.2 * torch.randn(B, L, C, generator=g)
    return {
        "window": window,
        "tau": torch.tensor([5, 7, 9, 11][:B], dtype=torch.long),
        "omega": torch.full((B,), 5000.0),
        "tpf": torch.full((B,), 200.0),
    }


def _model(seed: int, *, side_input: bool = False) -> TKANModel:
    torch.manual_seed(seed)
    m = TKANModel(
        input_size=_IN, hidden_size=_HID, num_layers=1,
        grid_size=5, spline_order=3, side_input=side_input,
    )
    return m.eval()


def _structured_model(seed: int, *, k_true: int = 3, noise: float = 0.02) -> TKANModel:
    """A network that genuinely has ``k_true`` recurring spline shapes.

    Every edge's spline coefficient vector is one of ``k_true`` prototypes,
    chosen by the edge's *input index* (so the same physical channel carries the
    same shape in every "fold"), scaled by a random positive gain and jittered.
    Positive gain only: scale is removed by shape normalisation, but a sign flip
    is a genuinely different function and must not be smuggled in here.

    This is the positive control. Without it, a verdict function that always
    returns "artifact" would pass every other test in this file.
    """
    m = _model(seed)
    rng = np.random.default_rng(12345)  # prototypes SHARED across folds/seeds
    protos = rng.normal(size=(k_true, 8))
    jitter = np.random.default_rng(seed)
    with torch.no_grad():
        for _, kan in _iter_kanlinear(m):
            if kan.n_coeff != protos.shape[1]:
                continue
            w = np.empty((kan.out_features, kan.in_features, kan.n_coeff))
            for i in range(kan.in_features):
                p = protos[i % k_true]
                gain = 0.5 + jitter.random((kan.out_features, 1))
                w[:, i, :] = gain * p + noise * jitter.normal(size=(kan.out_features, 8))
            kan.spline_weight.copy_(torch.tensor(w, dtype=kan.spline_weight.dtype))
    return m.eval()


def _bundles(model, batch, **kw):
    stats = collect_activations(model, batch, max_points=256)
    return AM.extract_stratified_curves(model, stats, edge_index_map(model), **kw)


# ========================================================================== #
# 1. edge_function component selector
# ========================================================================== #

def _legacy_edge_function(kan: KANLinear, n: int) -> dict:
    """Verbatim copy of the pre-``component=`` implementation.

    Kept here so "byte-identical" is asserted against the actual old arithmetic
    and not merely against the new default argument.
    """
    with torch.no_grad():
        sampled = kan.spline_curves(n)
        xs = torch.linspace(
            kan.grid_range[0], kan.grid_range[1], n,
            device=kan.grid.device, dtype=kan.grid.dtype,
        )
        base_x = kan.base_activation(xs)
        base = kan.base_weight.unsqueeze(-1) * base_x.view(1, 1, -1)
        curves = torch.as_tensor(sampled["curves"], device=kan.grid.device) + base
        return {"x": sampled["x"], "curves": curves.cpu().numpy()}


def test_component_full_is_byte_identical_to_legacy():
    """The default path must not move by a single bit."""
    torch.manual_seed(0)
    kan = KANLinear(3, 5, grid_size=5, spline_order=3)
    with torch.no_grad():
        kan.spline_weight.normal_(0.0, 0.4)

    legacy = _legacy_edge_function(kan, 64)
    default = kan.edge_function(64)
    explicit = kan.edge_function(64, component="full")

    assert np.array_equal(default["curves"], legacy["curves"])
    assert np.array_equal(default["x"], legacy["x"])
    assert np.array_equal(explicit["curves"], legacy["curves"])


def test_component_spline_and_base_decompose_the_full_curve():
    torch.manual_seed(1)
    kan = KANLinear(3, 4, grid_size=5, spline_order=3)
    with torch.no_grad():
        kan.spline_weight.normal_(0.0, 0.3)

    full = kan.edge_function(48, component="full")["curves"]
    spline = kan.edge_function(48, component="spline")["curves"]
    base = kan.edge_function(48, component="base")["curves"]

    # spline component == spline_curves exactly (same code path, no base added).
    np.testing.assert_array_equal(spline, kan.spline_curves(48)["curves"])
    np.testing.assert_allclose(full, spline + base, rtol=1e-6, atol=1e-7)
    assert not np.allclose(full, spline)  # base path is nonzero here


def test_component_rejects_unknown_selector():
    kan = KANLinear(2, 2)
    with pytest.raises(ValueError, match="component"):
        kan.edge_function(16, component="silu")


def test_base_component_is_a_one_parameter_family():
    """WHY we cluster on the spline only.

    Every base curve is ``base_weight[o,i] * SiLU(x)``: after shape
    normalisation they are all the SAME unit vector up to sign, so they would
    collapse into one huge meaningless cluster and swamp any real structure.
    """
    torch.manual_seed(2)
    kan = KANLinear(4, 4, grid_size=5, spline_order=3)
    base = kan.edge_function(64, component="base")["curves"].reshape(-1, 64)
    unit = base / np.linalg.norm(base, axis=1, keepdims=True)
    cos = np.abs(unit @ unit[0])
    np.testing.assert_allclose(cos, 1.0, atol=1e-6)


# ========================================================================== #
# 2. Stratification -- never pool a hidden-fed edge with a physical one
# ========================================================================== #

def test_strata_separate_physical_channels_from_hidden_state():
    model = _model(0)
    bundles = _bundles(model, _batch(), strata=None)  # None == all strata

    assert "encoder.L0.x_t" in bundles
    assert "encoder.L0.x_delay" in bundles
    assert "encoder.L0.h_prev" in bundles

    # layout for input_size=4, hidden=8: x_t 0-3, h_prev 4-11, x_delay 12-15
    assert {r["in"] for r in bundles["encoder.L0.x_t"]["records"]} == {0, 1, 2, 3}
    assert {r["in"] for r in bundles["encoder.L0.x_delay"]["records"]} == {12, 13, 14, 15}
    assert min(r["in"] for r in bundles["encoder.L0.h_prev"]["records"]) == 4

    assert AM.is_fold_stable("encoder.L0.x_t")
    assert AM.is_fold_stable("encoder.L0.x_delay")
    assert not AM.is_fold_stable("encoder.L0.h_prev")
    assert not AM.is_fold_stable("severity.hidden")
    # A stratum only ever holds one stratum's edges.
    for name, b in bundles.items():
        assert {r["stratum"] for r in b["records"]} == {name}


def test_layer_1_x_t_is_not_fold_stable():
    """The subtle one: ``block == "x_t"`` is physical only on layer 0.

    On layer 1 the same block name holds layer 0's hidden state, whose unit
    indices are permutation-arbitrary across folds. Keying the stratum on the
    block name alone is the obvious implementation and it is wrong.
    """
    torch.manual_seed(0)
    model = TKANModel(input_size=_IN, hidden_size=_HID, num_layers=2,
                      grid_size=5, spline_order=3).eval()
    bundles = _bundles(model, _batch(), strata=None)
    assert "encoder.L1.x_t" in bundles
    assert not AM.is_fold_stable("encoder.L1.x_t")
    assert all(r["identity"] is None for r in bundles["encoder.L1.x_t"]["records"])
    assert all(r["feature"].startswith("L0_h")
               for r in bundles["encoder.L1.x_t"]["records"])
    # ...while layer 0 of the same model still is.
    assert AM.is_fold_stable("encoder.L0.x_t")
    assert all(r["identity"] is not None for r in bundles["encoder.L0.x_t"]["records"])


def test_side_block_is_fold_stable_and_keeps_out_index():
    """The severity side block is the one stratum where ``out`` is stable too."""
    model = _model(0, side_input=True)
    bundles = _bundles(model, _batch(), strata=None)
    assert "severity.side" in bundles
    ids = [r["identity"] for r in bundles["severity.side"]["records"]]
    assert all("out=" in i for i in ids)
    # Encoder identities deliberately DROP out: hidden out-units are arbitrary.
    enc = [r["identity"] for r in bundles["encoder.L0.x_t"]["records"]]
    assert all("out=" not in i for i in enc)
    assert "encoder.layers.0.kan_z|xtpos" in enc


def test_hidden_fed_edges_refuse_index_comparison():
    """Hidden-unit indices are permutation-arbitrary across folds.

    They may be compared as a shape *population* but never by identity, so the
    identity path must raise rather than quietly return a meaningless number.
    """
    model = _model(0)
    bundles = _bundles(model, _batch(), strata=None)
    hidden = bundles["encoder.L0.h_prev"]
    assert all(r["identity"] is None for r in hidden["records"])

    with pytest.raises(ValueError, match="permutation-arbitrary|fold-stable"):
        AM.require_fold_stable(hidden)

    # The auto path must degrade to None, not to a fabricated agreement.
    gated = {f: AM.gate_by_importance(_bundles(_model(f), _batch(seed=f),
                                               strata=("encoder.L0.h_prev",))
                                      ["encoder.L0.h_prev"], quantile=0.5)
             for f in range(3)}
    res = AM.cross_fold_agreement(gated, k=3, seed=0)
    assert res["identity_agreement_mean"] is None
    assert res["prototype_fit_mean"] is not None  # population view still works


def test_cross_fold_agreement_refuses_mixed_strata():
    model = _model(0)
    b = _bundles(model, _batch(), strata=None)
    mixed = {
        0: AM.gate_by_importance(b["encoder.L0.x_t"], quantile=0.0),
        1: AM.gate_by_importance(b["encoder.L0.x_delay"], quantile=0.0),
    }
    with pytest.raises(ValueError, match="stratum"):
        AM.cross_fold_agreement(mixed, k=2, seed=0)


# ========================================================================== #
# 3. Gate BEFORE normalisation
# ========================================================================== #

def test_gate_cannot_be_applied_after_normalisation():
    b = _bundles(_model(0), _batch())["encoder.L0.x_t"]
    normalised = AM.normalise_curves(AM.gate_by_importance(b, quantile=0.0))
    with pytest.raises(ValueError, match="normalis"):
        AM.gate_by_importance(normalised, quantile=0.5)


def test_normalisation_requires_a_gate_first():
    b = _bundles(_model(0), _batch())["encoder.L0.x_t"]
    assert b["gate"] is None
    with pytest.raises(ValueError, match="gate"):
        AM.normalise_curves(b)


def test_gate_uses_raw_data_aware_importance_not_shape():
    """A dead edge and a dominant edge are identical once shape-normalised.

    Zero out one edge entirely and amplify another; the gate must drop the dead
    one on raw importance. If normalisation ran first the dead edge would be a
    degenerate (zero-norm) curve and either NaN out or, worse, be rescaled into
    a perfectly ordinary-looking shape.
    """
    model = _model(0)
    kan = model.encoder.layers[0].kan_z
    with torch.no_grad():
        kan.base_weight[0, 0] = 0.0
        kan.spline_weight[0, 0, :] = 0.0      # dead edge
        kan.base_weight[1, 1] = 5.0
        kan.spline_weight[1, 1, :] = 2.0      # dominant edge
    b = _bundles(model, _batch())["encoder.L0.x_t"]

    dead = [r for r in b["records"] if r["kan"].endswith("kan_z") and r["out"] == 0 and r["in"] == 0]
    assert len(dead) == 1 and dead[0]["importance"] == 0.0

    gated = AM.gate_by_importance(b, quantile=0.5)
    kept = {(r["kan"], r["out"], r["in"]) for r in gated["records"]}
    assert ("encoder.layers.0.kan_z", 0, 0) not in kept
    assert ("encoder.layers.0.kan_z", 1, 1) in kept
    assert gated["gate"]["n_before"] == b["n_edges"]
    assert gated["gate"]["pass_fraction"] <= 1.0


def test_degenerate_curves_are_counted_not_averaged_away():
    """Base-only edges: high importance, but no spline shape at all.

    Zeroing the splines while keeping the base path leaves every edge with a
    healthy data-aware importance (the SiLU path moves) and a completely flat
    spline curve. Those must be dropped and COUNTED, not rescaled by a naive
    epsilon into ordinary-looking unit vectors.
    """
    model = _model(0)
    with torch.no_grad():
        for _, kan in _iter_kanlinear(model):
            kan.spline_weight.zero_()
    b = _bundles(model, _batch())["encoder.L0.x_t"]
    gated = AM.gate_by_importance(b, quantile=0.0)   # importance is base-driven
    assert gated["n_edges"] > 0
    assert np.allclose([r["base_fraction"] for r in gated["records"]], 1.0)
    norm = AM.normalise_curves(gated)
    assert norm["n_degenerate_dropped"] == gated["n_edges"]
    assert norm["n_edges"] == 0


def test_stratum_dead_in_the_data_ends_undecided_not_scored():
    """The hazard found while wiring this: a stratum the data never moves.

    With constant side features every ``severity.side`` edge has importance
    exactly 0 -- yet its spline curves are perfectly ordinary and would cluster
    into perfectly ordinary modes. A quantile gate alone passes them (its
    threshold is 0 too). The hard zero-importance floor must empty the stratum
    so the verdict comes back UNDECIDED rather than scored.
    """
    folds = {f: _model(600 + f, side_input=True) for f in range(2)}
    batches = {f: _batch(seed=f) for f in range(2)}   # omega/tpf constant
    res = AM.run_activation_mode_analysis(
        folds, batches, n_seeds=3, k=3, n_points=32,
        strata=("severity.side",), gate_quantile=0.5, min_seeds=1,
    )
    st = res["strata"]["severity.side"]
    assert st["trained"]["prototype_fit_mean"] is None
    assert st["verdict"]["artifact_suspected"] is None
    assert st["gate"]["trained_pass_fraction"]["0"] == 0.0
    assert "UNDECIDED" in res["headline"]


# ========================================================================== #
# 4. Matched N
# ========================================================================== #

def test_matched_n_actually_matches_n():
    """The plan's exact failure mode: 90% of one arm, 5% of the other."""
    loud = _bundles(_model(0), _batch())["encoder.L0.x_t"]
    quiet = _bundles(_model(1), _batch())["encoder.L0.x_t"]
    # Force the asymmetry with an ABSOLUTE floor, which is where it really bites
    # (a quantile gate passes the same fraction by construction and would hide
    # the problem this constraint exists to catch).
    lo_thresh = float(np.quantile(loud["importance"], 0.10))    # passes ~90%
    hi_thresh = float(np.quantile(quiet["importance"], 0.95))   # passes ~5%

    a = AM.gate_by_importance(loud, min_importance=lo_thresh)
    b = AM.gate_by_importance(quiet, min_importance=hi_thresh)
    assert a["n_edges"] > 5 * b["n_edges"]

    ma, mb = AM.match_gate_sizes([a, b])
    assert ma["n_edges"] == mb["n_edges"] == min(a["n_edges"], b["n_edges"])
    assert ma["gate"]["matched_n"] == ma["n_edges"]
    # matching keeps the most important survivors, not an arbitrary prefix
    assert ma["importance"].min() >= a["importance"].min()
    assert set(np.round(ma["importance"], 12)).issubset(set(np.round(a["importance"], 12)))


def test_match_gate_sizes_requires_gated_bundles():
    b = _bundles(_model(0), _batch())["encoder.L0.x_t"]
    with pytest.raises(ValueError, match="gate"):
        AM.match_gate_sizes([b, b])


# ========================================================================== #
# 5. The untrained null and the pre-committed margin
# ========================================================================== #

def test_reinitialised_copy_guards():
    model = _model(0)
    fresh, n_reset = AM.reinitialised_copy(model, seed=3)
    assert n_reset > 0
    assert AM.assert_reinitialised(model, fresh) > 1e-8
    # identical models must be refused -- that is the silent-failure mode
    with pytest.raises(RuntimeError, match="did not change"):
        AM.assert_reinitialised(model, model)
    # a module with nothing to reset must be refused too
    with pytest.raises(RuntimeError, match="reset_parameters"):
        AM.reinitialised_copy(torch.nn.Sequential(torch.nn.SiLU()), seed=0)


def test_verdict_is_undecided_when_a_measurement_is_missing():
    """UNDECIDED is None and is NEVER a pass."""
    for v in (
        AM.mode_verdict(None, [0.1] * 25),
        AM.mode_verdict(0.9, []),
        AM.mode_verdict(0.9, [0.1] * 4, min_seeds=20),   # far too few seeds
        AM.mode_verdict(float("nan"), [0.1] * 25),
    ):
        assert v["artifact_suspected"] is None
        assert v["artifact_suspected"] is not False      # not a pass
        assert v["modes_learned"] is None


def test_verdict_threshold_rule_mirrors_the_tsr_precedent():
    high_null = AM.mode_verdict(0.90, list(np.linspace(0.85, 0.95, 25)), threshold=0.8)
    assert high_null["artifact_suspected"] is True
    assert high_null["modes_learned"] is False

    low_null = AM.mode_verdict(0.95, list(np.linspace(0.40, 0.50, 25)), threshold=0.8)
    assert low_null["artifact_suspected"] is False
    assert low_null["modes_learned"] is True
    assert low_null["margin"] > 0

    # A trained score at or below zero carries nothing to attribute.
    assert AM.mode_verdict(0.0, [0.0] * 25)["artifact_suspected"] is True


def test_untrained_arm_does_not_clear_the_margin():
    """Random weights vs random weights: the verdict MUST be 'artifact'.

    This is the test that would have caught the TSR and band-notch failures.
    """
    folds = {f: _model(100 + f) for f in range(3)}
    batches = {f: _batch(seed=f) for f in range(3)}
    res = AM.run_activation_mode_analysis(
        folds, batches, n_seeds=20, k=3, n_points=32, min_seeds=20,
        strata=("encoder.L0.x_t",), gate_quantile=0.5, seed=0,
    )
    st = res["strata"]["encoder.L0.x_t"]
    assert st["verdict"]["artifact_suspected"] is True
    assert st["verdict"]["modes_learned"] is False
    assert st["untrained"]["n_seeds"] == 20
    assert "UNTRAINED" in res["headline"].upper() or "ARTIFACT" in res["headline"].upper()


def test_untrained_agreement_is_already_high_in_absolute_terms():
    """NEGATIVE SIGNAL ABOUT THE METRIC -- pinned so it cannot be forgotten.

    A randomly initialised network scores ``prototype_fit`` around 0.6, not
    around 0. With only ``n_coeff = 8`` coefficients over one shared basis, no
    curve can be far from all of k prototypes, so the statistic has a high
    floor. Any write-up that quotes a trained agreement of "0.7" as evidence of
    learned structure is quoting noise. Only the margin over this floor counts,
    which is why :func:`mode_verdict` is a ratio against the untrained null and
    never a threshold on the trained number.
    """
    gated = {}
    for f in range(3):
        b = _bundles(_model(500 + f), _batch(seed=f), n=32,
                     strata=("encoder.L0.x_t",))["encoder.L0.x_t"]
        gated[f] = AM.gate_by_importance(b, quantile=0.5)
    res = AM.cross_fold_agreement(gated, k=3, seed=0)
    assert res["prototype_fit_mean"] > 0.5          # the floor is high
    assert res["identity_agreement_mean"] < 0.7     # but identity is near chance
    assert any("ABSOLUTE AGREEMENT SCORES" in c for c in AM.CAVEATS)


def test_structured_model_clears_the_margin():
    """Positive control: a metric that always says 'artifact' is worthless."""
    folds = {f: _structured_model(200 + f) for f in range(3)}
    batches = {f: _batch(seed=f) for f in range(3)}
    res = AM.run_activation_mode_analysis(
        folds, batches, n_seeds=20, k=3, n_points=32, min_seeds=20,
        strata=("encoder.L0.x_t",), gate_quantile=0.5, seed=0,
    )
    st = res["strata"]["encoder.L0.x_t"]
    assert st["verdict"]["artifact_suspected"] is False
    assert st["trained"]["prototype_fit_mean"] > st["untrained"]["p95"]
    # the shape recurs on the SAME physical channel across folds
    assert st["trained"]["identity_agreement_mean"] > 0.8


# ========================================================================== #
# 6. Reporting contracts
# ========================================================================== #

def test_spectrum_is_reported_as_a_diagnostic_never_as_a_finding():
    b = AM.gate_by_importance(_bundles(_model(0), _batch())["encoder.L0.x_t"], quantile=0.0)
    diag = AM.spectrum_diagnostic(AM.normalise_curves(b))
    assert diag["is_finding"] is False
    assert "n_coeff" in diag and diag["n_coeff"] == 8
    assert "not a finding" in diag["caveat"].lower()
    # PCA over B*c curves cannot exceed n_coeff components: that is architecture.
    evr = np.asarray(diag["explained_variance_ratio"])
    assert int((evr > 1e-8).sum()) <= diag["n_coeff"]
    assert "variance_explained" not in diag  # no result-shaped key name


def test_base_fraction_is_reported_per_edge_and_per_stratum():
    b = _bundles(_model(0), _batch())["encoder.L0.x_t"]
    bf = np.asarray([r["base_fraction"] for r in b["records"]])
    assert np.isfinite(bf).all() and (bf >= 0).all() and (bf <= 1).all()

    folds = {f: _model(300 + f) for f in range(2)}
    res = AM.run_activation_mode_analysis(
        folds, {f: _batch(seed=f) for f in range(2)}, n_seeds=6, k=3,
        n_points=32, min_seeds=20, strata=("encoder.L0.x_t",),
    )
    st = res["strata"]["encoder.L0.x_t"]
    assert 0.0 <= st["trained"]["base_fraction_mean"] <= 1.0
    # 6 seeds < min_seeds -> UNDECIDED, and that must not read as a pass
    assert st["verdict"]["artifact_suspected"] is None


def test_result_is_json_serialisable_with_no_silent_nans():
    folds = {f: _model(400 + f) for f in range(2)}
    res = AM.run_activation_mode_analysis(
        folds, {f: _batch(seed=f) for f in range(2)}, n_seeds=4, k=3,
        n_points=32, strata=("encoder.L0.x_t",),
    )
    text = json.dumps(res, allow_nan=False)  # NaN must be None, never a number
    assert json.loads(text)["strata"]["encoder.L0.x_t"]["verdict"]["threshold"] == 0.8


# ========================================================================== #
# 7. Mode ablation edge sets (step 8 -- the runner needs these)
# ========================================================================== #

def test_mode_edge_sets_match_count_and_summed_importance():
    b = AM.gate_by_importance(_bundles(_model(0), _batch())["encoder.L0.x_t"],
                              quantile=0.0)
    norm = AM.normalise_curves(b)
    protos = AM.fit_prototypes(norm["curves"], k=3, seed=0)
    labels, _ = AM.assign_to_prototypes(norm["curves"], protos)
    sets = AM.mode_edge_sets(norm, labels, mode=int(np.bincount(labels).argmax()), seed=0)

    n = len(sets["mode_edges"])
    assert n > 0
    assert len(sets["random_matched"]) == n
    assert len(sets["other_modes_matched"]) == n
    tgt = sets["summed_importance"]["mode_edges"]
    assert abs(sets["summed_importance"]["random_matched"] - tgt) / max(tgt, 1e-9) < 0.5
    # Edge specs are exactly what a zeroing routine needs.
    assert set(sets["mode_edges"][0]) >= {"kan", "out", "in"}

    def _key(edges):
        return {(e["kan"], e["out"], e["in"]) for e in edges}

    # "Other modes" must be disjoint from the ablated mode, or the control is
    # not a control.
    assert not (_key(sets["mode_edges"]) & _key(sets["other_modes_matched"]))
