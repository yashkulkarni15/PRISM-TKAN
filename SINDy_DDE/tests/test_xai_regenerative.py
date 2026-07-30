"""Tests for :mod:`pri2m.xai.regenerative` -- the learned-regeneration probe.

The floor tests build synthetic ``KANLinear`` edge pairs whose relationship is
known by construction (an exact negation, an exact copy, an unrelated pair) and
assert the metric separates them.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from pri2m.models.kan import KANLinear
from pri2m.models.tkan_model import TKANModel
from pri2m.xai import regenerative as regen


# --------------------------------------------------------------------------- #
# Helpers: synthetic KANLinear layers with a planted edge relationship
# --------------------------------------------------------------------------- #
def _blank_kan(in_features: int = 2, out_features: int = 1) -> KANLinear:
    """A KANLinear with every edge zeroed (base + spline), ready for planting."""
    kan = KANLinear(in_features, out_features, grid_size=5, spline_order=3)
    with torch.no_grad():
        kan.base_weight.zero_()
        kan.spline_weight.zero_()
    return kan


def _plant(kan, out, idx, coeffs, base=0.0):
    """Give edge ``(out, idx)`` the spline coefficients ``coeffs`` + base weight."""
    with torch.no_grad():
        kan.spline_weight[out, idx] = torch.as_tensor(coeffs, dtype=torch.float32)
        kan.base_weight[out, idx] = float(base)


def _pair_kan(coeffs_a, coeffs_b, base_a=0.0, base_b=0.0) -> KANLinear:
    """A 2-in/1-out KANLinear with edge 0 == curve A and edge 1 == curve B."""
    kan = _blank_kan(2, 1)
    _plant(kan, 0, 0, coeffs_a, base_a)
    _plant(kan, 0, 1, coeffs_b, base_b)
    return kan


def _rng_coeffs(seed, n=8, scale=1.0):
    return np.random.default_rng(seed).normal(0.0, scale, size=n)


# --------------------------------------------------------------------------- #
# Floor test 1: an exact negation IS detected
# --------------------------------------------------------------------------- #
def test_exact_negation_scores_one():
    """phi_delay == -phi_inst  =>  score == +1, corr == -1, gain == 1."""
    c = _rng_coeffs(0)
    kan = _pair_kan(c, -c, base_a=0.7, base_b=-0.7)
    row = regen.edge_pair_metrics(kan, 0, 0, 1)
    assert row["score"] == pytest.approx(1.0, abs=1e-5)
    assert row["corr"] == pytest.approx(-1.0, abs=1e-5)
    assert row["gain"] == pytest.approx(1.0, abs=1e-5)
    assert not row["degenerate"]


def test_negation_detected_with_spline_only_edges():
    """The detection does not depend on the SiLU base path being present."""
    c = _rng_coeffs(3)
    kan = _pair_kan(c, -c)
    row = regen.edge_pair_metrics(kan, 0, 0, 1)
    assert row["score"] == pytest.approx(1.0, abs=1e-5)
    assert row["spline_share_inst"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Floor test 2: an unrelated pair is NOT detected
# --------------------------------------------------------------------------- #
def test_unrelated_pair_is_not_flagged():
    """A monotone ramp vs an alternating oscillation is near-orthogonal."""
    ramp = np.linspace(-1.0, 1.0, 8)
    osc = np.array([1.0, -1.0] * 4)
    kan = _pair_kan(ramp, osc)
    row = regen.edge_pair_metrics(kan, 0, 0, 1)
    assert abs(row["score"]) < 0.3
    assert row["score"] < regen.NEGATION_THRESHOLD


def test_random_unrelated_pairs_rarely_reach_the_threshold():
    """Over 64 independent random spline pairs, few clear the negation gate."""
    scores = []
    for seed in range(64):
        kan = _pair_kan(_rng_coeffs(1000 + seed), _rng_coeffs(5000 + seed))
        scores.append(regen.edge_pair_metrics(kan, 0, 0, 1)["score"])
    scores = np.asarray(scores)
    assert np.mean(np.abs(scores)) < 0.5
    assert np.mean(scores >= regen.NEGATION_THRESHOLD) <= 0.1


def test_identical_pair_scores_minus_one():
    """A copy (a SUM composition, not a difference) is the opposite extreme."""
    c = _rng_coeffs(7)
    kan = _pair_kan(c, c, base_a=0.4, base_b=0.4)
    row = regen.edge_pair_metrics(kan, 0, 0, 1)
    assert row["score"] == pytest.approx(-1.0, abs=1e-5)
    assert row["corr"] == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# The metric's two factors: shape agreement and gain balance
# --------------------------------------------------------------------------- #
def test_gain_mismatch_is_penalized():
    """phi_delay == -0.1 * phi_inst: perfect shape, badly unbalanced gain."""
    c = _rng_coeffs(11)
    kan = _pair_kan(c, -0.1 * c)
    row = regen.edge_pair_metrics(kan, 0, 0, 1)
    assert row["corr"] == pytest.approx(-1.0, abs=1e-5)
    assert row["gain"] == pytest.approx(0.1, rel=1e-3)
    # score = -corr * 2g/(1+g^2) = 2*0.1/1.01
    assert row["score"] == pytest.approx(2 * 0.1 / 1.01, rel=1e-3)
    assert row["score"] < regen.NEGATION_THRESHOLD


def test_score_equals_corr_times_balance():
    """The identity score == -corr * balance holds on an arbitrary pair."""
    kan = _pair_kan(_rng_coeffs(21), 0.6 * _rng_coeffs(22))
    row = regen.edge_pair_metrics(kan, 0, 0, 1)
    assert row["score"] == pytest.approx(-row["corr"] * row["balance"], abs=1e-6)


def test_constant_offsets_do_not_change_the_score():
    """A DC offset on either edge is inert (absorbed downstream) -> centered."""
    c = _rng_coeffs(13)
    plain = regen.edge_pair_metrics(_pair_kan(c, -c), 0, 0, 1)
    shifted = regen.edge_pair_metrics(_pair_kan(c + 5.0, -c + 3.0), 0, 0, 1)
    assert shifted["score"] == pytest.approx(plain["score"], abs=1e-5)


# --------------------------------------------------------------------------- #
# Degenerate / artifact guards
# --------------------------------------------------------------------------- #
def test_dead_edges_are_flagged_degenerate():
    kan = _blank_kan(2, 1)
    row = regen.edge_pair_metrics(kan, 0, 0, 1)
    assert row["degenerate"] is True
    assert np.isnan(row["score"])


def test_one_dead_edge_scores_zero():
    kan = _pair_kan(_rng_coeffs(31), np.zeros(8))
    row = regen.edge_pair_metrics(kan, 0, 0, 1)
    assert row["score"] == pytest.approx(0.0, abs=1e-6)


def test_pure_base_path_pair_is_flagged_base_dominated():
    """base-only edges with opposite weights score 1.0 by CONSTRUCTION.

    Both edges are then ``w * SiLU(x)``, so the correlation is +-1 for free and
    carries no evidence about learned structure. The row must say so.
    """
    kan = _pair_kan(np.zeros(8), np.zeros(8), base_a=1.0, base_b=-1.0)
    row = regen.edge_pair_metrics(kan, 0, 0, 1)
    assert row["score"] == pytest.approx(1.0, abs=1e-5)
    assert row["spline_share_inst"] == pytest.approx(0.0, abs=1e-6)
    assert row["base_dominated"] is True


def test_spline_dominated_pair_is_not_flagged_base_dominated():
    c = _rng_coeffs(41, scale=2.0)
    kan = _pair_kan(c, -c, base_a=0.01, base_b=-0.01)
    row = regen.edge_pair_metrics(kan, 0, 0, 1)
    assert row["base_dominated"] is False


# --------------------------------------------------------------------------- #
# The evaluation range is the data-visited range
# --------------------------------------------------------------------------- #
def test_visited_range_is_respected():
    """A pair that negates only on the right half scores high only there."""
    xs = np.linspace(-1, 1, 200)
    kan = _blank_kan(2, 1)
    # Edge A: a plain ramp via the base path is not enough (SiLU is monotone on
    # both halves), so plant splines: A rises across the whole range; B is the
    # negation of A on [0, 1] and a COPY of A on [-1, 0].
    coeff_a = np.array([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    coeff_b = np.array([-1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0])
    _plant(kan, 0, 0, coeff_a)
    _plant(kan, 0, 1, coeff_b)
    right = regen.edge_pair_metrics(kan, 0, 0, 1, lo=0.35, hi=1.0)
    left = regen.edge_pair_metrics(kan, 0, 0, 1, lo=-1.0, hi=-0.35)
    assert right["lo"] == 0.35 and right["hi"] == 1.0
    assert right["score"] > 0.9
    assert left["score"] < -0.9
    assert len(xs) == 200  # sanity: the module's default sweep length


def test_visited_range_falls_back_to_grid_range_when_degenerate():
    c = _rng_coeffs(51)
    kan = _pair_kan(c, -c)
    row = regen.edge_pair_metrics(kan, 0, 0, 1, lo=np.nan, hi=np.nan)
    assert (row["lo"], row["hi"]) == kan.grid_range


# --------------------------------------------------------------------------- #
# Semantics: the delay/instantaneous index pairing is DERIVED, not hardcoded
# --------------------------------------------------------------------------- #
def test_delay_channel_pairs_matches_semantics_on_the_blessed_shape():
    """input_size=4, hidden_size=64 -> delay:xtpos == 68, delay:ytpos == 69."""
    model = TKANModel(input_size=4, hidden_size=64, num_layers=1)
    pairs = regen.delay_pairs(model)
    kan = "encoder.layers.0.kan_r"
    assert kan in pairs
    by_channel = {p["channel"]: p for p in pairs[kan]}
    assert by_channel["xtpos"]["inst_index"] == 0
    assert by_channel["xtpos"]["delay_index"] == 68
    assert by_channel["ytpos"]["inst_index"] == 1
    assert by_channel["ytpos"]["delay_index"] == 69
    assert set(by_channel) == {"xtpos", "ytpos", "Forcex", "Forcey"}
    # every delay-aware gate of the bottom layer is covered, and nothing else.
    assert set(pairs) == {
        "encoder.layers.0.kan_z",
        "encoder.layers.0.kan_r",
        "encoder.layers.0.kan_h",
    }


def test_delay_pairs_empty_without_delay():
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, use_delay=False)
    assert regen.delay_pairs(model) == {}


# --------------------------------------------------------------------------- #
# Whole-model scan
# --------------------------------------------------------------------------- #
def _small_model():
    """A 2-channel/3-unit delay-aware model. Seeded: torch's default RNG is NOT.

    Gate inputs are ``[x_t(2) | h_prev(3) | x_delay(2)]``, so channel 0 pairs
    index 0 with index 5 and channel 1 pairs index 1 with index 6.
    """
    torch.manual_seed(0)
    return TKANModel(input_size=2, hidden_size=3, num_layers=1)


def test_scan_model_finds_a_planted_negation():
    model = _small_model()
    kan = model.encoder.layers[0].kan_r
    # channel 0 -> instantaneous index 0, delayed index 2 + 3 = 5.
    c = _rng_coeffs(61, n=kan.n_coeff, scale=2.0)
    with torch.no_grad():
        kan.spline_weight[0, 0] = torch.as_tensor(c, dtype=torch.float32)
        kan.spline_weight[0, 5] = torch.as_tensor(-c, dtype=torch.float32)
        kan.base_weight[0, 0] = 0.5
        kan.base_weight[0, 5] = -0.5
    res = regen.scan_model(model)
    rows = {(r["kan"], r["out"], r["channel"]): r for r in res["rows"]}
    planted = rows[("encoder.layers.0.kan_r", 0, "ch0")]
    assert planted["score"] == pytest.approx(1.0, abs=1e-5)
    assert planted["in_inst"] == 0 and planted["in_delay"] == 5
    # 3 gates x 3 out units x 2 channels
    assert len(res["rows"]) == 18
    assert res["summary"]["n"] == 18


def test_scan_model_control_rows_are_cross_channel_only():
    res = regen.scan_model(_small_model())
    assert res["control_rows"], "control pairs must be produced"
    for r in res["control_rows"]:
        assert r["channel"] != r["control_channel"]
        assert r["in_inst"] != r["in_delay"]
    assert "control_summary" in res


def test_scan_model_respects_visited_ranges_from_stats():
    model = _small_model()
    kan_name = "encoder.layers.0.kan_r"
    kan = model.encoder.layers[0].kan_r
    n_in = kan.in_features
    stats = {
        kan_name: {
            "visited_lo": np.full(n_in, -0.5),
            "visited_hi": np.full(n_in, 0.5),
            "importance": np.ones((kan.out_features, n_in)),
        }
    }
    res = regen.scan_model(model, stats)
    rows = [r for r in res["rows"] if r["kan"] == kan_name]
    assert rows and all(r["lo"] == -0.5 and r["hi"] == 0.5 for r in rows)
    assert all(r["weight"] == 1.0 for r in rows)
    # gates without stats fall back to the full grid range
    other = [r for r in res["rows"] if r["kan"] != kan_name]
    assert all((r["lo"], r["hi"]) == kan.grid_range for r in other)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def test_summarize_rows_counts_and_weights():
    rows = [
        {"score": 1.0, "weight": 3.0, "degenerate": False, "base_dominated": False},
        {"score": -1.0, "weight": 1.0, "degenerate": False, "base_dominated": True},
        {"score": 0.0, "weight": 0.0, "degenerate": False, "base_dominated": False},
        {"score": np.nan, "weight": 0.0, "degenerate": True, "base_dominated": False},
    ]
    s = regen.summarize_rows(rows)
    assert s["n"] == 4
    assert s["n_scored"] == 3
    assert s["mean"] == pytest.approx(0.0)
    assert s["weighted_mean"] == pytest.approx((1.0 * 3.0 + -1.0 * 1.0) / 4.0)
    assert s["frac_negating"] == pytest.approx(1 / 3)
    assert s["frac_summing"] == pytest.approx(1 / 3)
    assert s["frac_base_dominated"] == pytest.approx(1 / 3)


def test_summarize_rows_handles_empty():
    s = regen.summarize_rows([])
    assert s["n"] == 0
    assert np.isnan(s["mean"])


def test_summary_separates_planted_negation_from_control():
    """Plant a negation on every matched pair: matched ~ +1, cross-channel ~ 0."""
    model = _small_model()
    for gate in ("kan_z", "kan_r", "kan_h"):
        kan = getattr(model.encoder.layers[0], gate)
        with torch.no_grad():
            for o in range(kan.out_features):
                for ch, (i, j) in enumerate([(0, 5), (1, 6)]):
                    c = _rng_coeffs(700 + 31 * o + 7 * ch, n=kan.n_coeff, scale=2.0)
                    kan.spline_weight[o, i] = torch.as_tensor(c, dtype=torch.float32)
                    kan.spline_weight[o, j] = torch.as_tensor(-c, dtype=torch.float32)
                    kan.base_weight[o, j] = -kan.base_weight[o, i]
    res = regen.scan_model(model)
    assert res["summary"]["mean"] > 0.99
    assert res["summary"]["frac_negating"] == pytest.approx(1.0)
    # the cross-channel control sees two unrelated planted curves -> no signal
    assert res["control_summary"]["mean"] < 0.5
    assert res["summary"]["mean"] > res["control_summary"]["mean"] + 0.5


# --------------------------------------------------------------------------- #
# Runner smoke test (no checkpoint / no data needed)
# --------------------------------------------------------------------------- #
def test_runner_plot_smoke(tmp_path):
    import regenerative_phase3 as runner

    model = _small_model()
    res = regen.scan_model(model)
    path = runner._plot(res, model, tmp_path / "regenerative.png", top_k=3, n=64)
    assert pathlib.Path(path).is_file()


# --------------------------------------------------------------------------- #
# The untrained-twin null
# --------------------------------------------------------------------------- #
def test_random_edge_null_destroys_a_planted_negation():
    model = _small_model()
    kan = model.encoder.layers[0].kan_r
    c = _rng_coeffs(81, n=kan.n_coeff, scale=2.0)
    with torch.no_grad():
        kan.spline_weight[0, 0] = torch.as_tensor(c, dtype=torch.float32)
        kan.spline_weight[0, 5] = torch.as_tensor(-c, dtype=torch.float32)
        kan.base_weight[0, 5] = -kan.base_weight[0, 0]
    null = regen.random_edge_null(model, seed=0)
    trained_row = regen.edge_pair_metrics(kan, 0, 0, 5)
    null_row = regen.edge_pair_metrics(null.encoder.layers[0].kan_r, 0, 0, 5)
    assert trained_row["score"] == pytest.approx(1.0, abs=1e-5)
    assert null_row["score"] < regen.NEGATION_THRESHOLD
    # the null is a copy: the original weights are untouched
    assert torch.allclose(kan.spline_weight[0, 5], torch.as_tensor(-c, dtype=torch.float32))


# --------------------------------------------------------------------------- #
# Runner end-to-end on a synthetic checkpoint + synthetic NPZ tree
# --------------------------------------------------------------------------- #
def test_runner_end_to_end(tmp_path):
    import json

    from tests._phase3_synth import make_synth_npz_tree, make_synth_table

    import regenerative_phase3 as runner

    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2), n_files=3,
                                   length=256, seed=0)
    table_csv = tmp_path / "table.csv"
    make_synth_table(npz_root).to_csv(table_csv, index=False)

    torch.manual_seed(0)
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=True,
                      side_features=("omega", "tpf", "tau"), side_hidden=4)
    ckpt = tmp_path / "runs" / "synthrun" / "checkpoints" / "best_auc.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    pre_path = tmp_path / "runs" / "synthrun" / "preprocessing.json"
    pre_path.write_text(json.dumps({
        "norm": {"mean": [0.0] * 4, "std": [0.01, 0.01, 80.0, 80.0]},
        "side": {"features": ["omega", "tpf", "tau"]},
    }), encoding="utf-8")

    out_dir = tmp_path / "out"
    rc = runner.main([
        "--checkpoint", str(ckpt), "--npz-root", str(npz_root),
        "--table", str(table_csv), "--preprocessing", str(pre_path),
        "--output-dir", str(out_dir), "--n-windows", "6",
        "--window-length", "128", "--window-stride", "64", "--pool", "1",
        "--sweep-points", "64", "--top-k", "3",
    ])
    assert rc == 0

    payload = json.loads((out_dir / "regenerative.json").read_text(encoding="utf-8"))
    # 3 gates x 8 out units x 4 channels
    assert len(payload["rows"]) == 96
    assert payload["summary"]["n_scored"] > 0
    assert np.isfinite(payload["summary"]["mean"])
    assert payload["control_summary"]["n_scored"] == 3 * 8 * 4 * 3
    assert np.isfinite(payload["untrained_null_summary"]["mean"])
    assert set(payload["by_channel"]) == {"xtpos", "ytpos", "Forcex", "Forcey"}
    # rows are scored over the DATA-VISITED range, not the whole grid
    assert all(-1.0 <= r["lo"] < r["hi"] <= 1.0 for r in payload["rows"])
    assert (out_dir / "regenerative.png").is_file()


def test_random_edge_null_leaves_the_global_rng_untouched():
    model = _small_model()
    before = torch.get_rng_state()
    regen.random_edge_null(model, seed=123)
    assert torch.equal(torch.get_rng_state(), before)


# --------------------------------------------------------------------------- #
# Checkpoint-free fallback: scoring a white-box run's symbolic_edges.csv
# --------------------------------------------------------------------------- #
_CSV_HEADER = "kan,out,in,block,name,importance,formula,r2,bic,accepted\n"
_KAN = "encoder.layers.0.kan_r"


def _symbolic_csv(tmp_path, body, name="symbolic_edges.csv"):
    path = tmp_path / name
    path.write_text(_CSV_HEADER + body, encoding="utf-8")
    return path


def test_evaluate_formula_handles_every_emitted_family():
    import regenerative_phase3 as runner

    xs = np.linspace(-1, 1, 51)
    cases = {
        "2*x + 1": 2 * xs + 1,
        "-3.5*x^2 + 0.5*x + 2": -3.5 * xs**2 + 0.5 * xs + 2,
        "1.5*tanh(2*x + 0.25) + -1": 1.5 * np.tanh(2 * xs + 0.25) - 1,
        "2*sin(3*x + 1) + 0.5": 2 * np.sin(3 * xs + 1) + 0.5,
        "0.75*|x| + 0.1": 0.75 * np.abs(xs) + 0.1,
        "1.25*max(x, 0) + -0.5": 1.25 * np.maximum(xs, 0) - 0.5,
        "0.42": np.full_like(xs, 0.42),
    }
    for formula, expected in cases.items():
        got = runner.evaluate_formula(formula, xs)
        assert got.shape == xs.shape
        assert np.allclose(got, expected), formula


def test_evaluate_formula_refuses_anything_outside_the_whitelist():
    import regenerative_phase3 as runner

    for bad in ("__import__('os').system('echo hi')", "OPEN(1)", ""):
        with pytest.raises(ValueError):
            runner.evaluate_formula(bad, np.linspace(-1, 1, 5))


def test_score_symbolic_csv_detects_a_negation_pair(tmp_path):
    import regenerative_phase3 as runner

    path = _symbolic_csv(tmp_path,
        f"{_KAN},7,1,x_t,ytpos,1.0,2*tanh(3*x + 0.5) + 1,0.99,-100,True\n"
        f"{_KAN},7,69,x_delay,delay:ytpos,0.8,-2*tanh(3*x + 0.5) + -4,0.99,-100,True\n")
    rows = runner.score_symbolic_csv(path)
    assert len(rows) == 1
    assert rows[0]["score"] == pytest.approx(1.0, abs=1e-6)
    assert rows[0]["channel"] == "ytpos"
    assert rows[0]["in_inst"] == 1 and rows[0]["in_delay"] == 69
    assert rows[0]["weight"] == pytest.approx(0.8)
    assert rows[0]["surrogate"] is True
    # the surrogate cannot know the base/spline split, and must not pretend to
    assert rows[0]["base_dominated"] is None
    assert np.isnan(regen.summarize_rows(rows)["frac_base_dominated"])


def test_score_symbolic_csv_detects_a_copy_pair(tmp_path):
    import regenerative_phase3 as runner

    path = _symbolic_csv(tmp_path,
        f"{_KAN},7,1,x_t,ytpos,1.0,2*tanh(3*x + 0.5) + 1,0.99,-100,True\n"
        f"{_KAN},7,69,x_delay,delay:ytpos,0.8,2*tanh(3*x + 0.5) + -4,0.99,-100,True\n")
    rows = runner.score_symbolic_csv(path)
    assert rows[0]["score"] == pytest.approx(-1.0, abs=1e-6)


def test_score_symbolic_csv_ignores_unmatched_edges(tmp_path):
    """A delay edge whose partner is a DIFFERENT unit / channel / gate is skipped."""
    import regenerative_phase3 as runner

    path = _symbolic_csv(tmp_path,
        f"{_KAN},7,69,x_delay,delay:ytpos,0.8,2*x,0.99,-100,True\n"
        f"{_KAN},8,1,x_t,ytpos,1.0,2*x,0.99,-100,True\n"            # other unit
        f"{_KAN},7,0,x_t,xtpos,1.0,2*x,0.99,-100,True\n"            # other channel
        f"encoder.layers.0.kan_z,7,1,x_t,ytpos,1.0,2*x,0.99,-100,True\n"  # other gate
        "severity.cls_head,0,46,hidden,h46,2.0,2*x,0.99,-100,True\n")      # not a gate
    assert runner.score_symbolic_csv(path) == []
