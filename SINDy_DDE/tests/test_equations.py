import numpy as np
import pytest
import torch

from pri2m.models.kan import KANLinear
from pri2m.xai.equations import (
    CHANNELS, best_match, channel_ranges, family_agreement, range_curves,
    recur_verdict, unexplained,
)
from pri2m.xai.symbolic import _eval_edge


@pytest.fixture
def tiny_kan_model():
    torch.manual_seed(0)
    m = torch.nn.Module()
    m.encoder = torch.nn.Module()
    m.encoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    m.encoder.layers[0].kan_z = KANLinear(5, 3)
    return m


def test_affine_copy_is_zero_unexplained():
    a = np.sin(np.linspace(-2, 2, 256))[None]
    b = -3 * a + 2
    assert unexplained(a, b)[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_unexplained_shape_and_orthogonal():
    x = np.linspace(-1, 1, 101)
    u = unexplained(np.stack([x, x]), np.stack([x**2, x, x**3]))
    assert u.shape == (2, 3)
    assert u[0, 0] == pytest.approx(1.0, abs=1e-12)   # odd vs even: nothing explained
    assert u[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_constant_curve_is_nan_and_unmatched():          # Review Focus 3
    u = unexplained(np.ones((1, 50)), np.random.default_rng(0).normal(size=(2, 50)))
    assert np.isnan(u).all()
    u_min, j = best_match(u, ["delay:ytpos"], ["delay:ytpos"] * 2)
    assert np.isnan(u_min[0]) and j[0] == -1


def test_best_match_respects_channel():
    # identical curve exists only on another channel -> not chosen
    u = np.array([[0.0, 0.4, 0.2]])
    u_min, j = best_match(u, ["delay:ytpos"], ["delay:xtpos", "delay:ytpos", "delay:ytpos"])
    assert j[0] == 2 and u_min[0] == pytest.approx(0.2)
    u_min, j = best_match(u, ["delay:Forcex"], ["delay:xtpos", "delay:ytpos", "delay:ytpos"])
    assert np.isnan(u_min[0]) and j[0] == -1


def test_best_match_skips_nan_candidate():
    u = np.array([[np.nan, 0.3]])
    u_min, j = best_match(u, ["f"], ["f", "f"])
    assert j[0] == 1 and u_min[0] == pytest.approx(0.3)


def test_channel_ranges_percentiles():
    pooled = np.zeros((1000, 3, 4))
    pooled[..., 1] = np.linspace(-5, 5, 3000).reshape(1000, 3)
    r = channel_ranges(pooled)
    assert list(r) == list(CHANNELS)
    lo, hi = r["ytpos"]
    assert lo == pytest.approx(np.percentile(pooled[..., 1], 1)) and hi == pytest.approx(np.percentile(pooled[..., 1], 99))


def test_range_curves_equal_eval_edge_and_extend_past_grid(tiny_kan_model):
    model = tiny_kan_model
    edges = [
        {"kan": "encoder.layers.0.kan_z", "out": 2, "in": 4, "feature": "delay:ytpos"},
        {"kan": "encoder.layers.0.kan_z", "out": 0, "in": 1, "feature": "delay:xtpos"},
    ]
    ranges = {"ytpos": (-3.0, 3.0), "xtpos": (-0.5, 0.5), "Forcex": (0.0, 1.0), "Forcey": (0.0, 1.0)}
    xs, ys = range_curves(model, edges, ranges, n=64)
    assert xs.shape == ys.shape == (2, 64)
    assert xs[0, 0] == -3.0 and xs[0, -1] == 3.0 and xs[1, 0] == -0.5
    kan = model.get_submodule(edges[0]["kan"])
    np.testing.assert_allclose(ys[0], _eval_edge(kan, 2, 4, xs[0]), rtol=1e-6)
    assert xs[0, 0] < kan.grid[0].item()      # evaluated beyond the knot grid, not clipped


def test_recur_verdict_rule():
    v = recur_verdict(np.array([0.01, 0.02, 0.5]), [np.array([0.3, 0.4, 0.5])] * 20)
    assert v["frac_trained"] == pytest.approx(2 / 3) and v["recurs"] is True
    assert v["null_p95"] == 0.0 and v["n_used"] == 3


def test_recur_verdict_not_recurring_and_nan_excluded():
    v = recur_verdict(np.array([0.01, np.nan, 0.5, 0.5]), [np.array([0.01, 0.5])] * 20)
    assert v["n_used"] == 3 and v["frac_trained"] == pytest.approx(1 / 3)
    assert v["null_p95"] == pytest.approx(0.5) and v["recurs"] is False


def test_family_agreement_against_chance():
    fam_a = ["sin", "sin", "poly", "poly"]
    fam_b = ["sin", "poly", "poly", "sin"]
    r = family_agreement(fam_a, fam_b, np.array([0, 1, -1, 3]))
    assert r["n"] == 3 and r["same"] == 1 and r["rate"] == pytest.approx(1 / 3)
    assert r["n_no_fit"] == 0
    # matched pairs: (sin,sin) (sin,poly) (poly,sin) -> sin 4, poly 2 of 6
    assert r["chance"] == pytest.approx((4 / 6) ** 2 + (2 / 6) ** 2)


def test_family_agreement_excludes_no_fit_pairs():
    fam_a = ["sin", None, "poly", "none", "sin"]
    fam_b = ["sin", None, "", "poly"]
    r = family_agreement(fam_a, fam_b, np.array([0, 1, 3, 2, 2]))
    # pairs: sin-sin ok, None-None drop, poly-poly ok, "none"-"" drop, sin-"" drop
    assert (r["n"], r["same"], r["n_no_fit"]) == (2, 2, 3)
    assert r["chance"] == pytest.approx(0.5)      # valid pairs pool sin x2, poly x2
    assert r["rate"] == 1.0


def test_family_agreement_one_none_excluded():
    r = family_agreement(["sin", "poly"], ["poly", None], np.array([0, 1]))
    assert (r["n"], r["same"], r["n_no_fit"]) == (1, 0, 1)
    assert r["chance"] == pytest.approx(0.5)


def test_family_agreement_all_dropped_is_nan():
    r = family_agreement([None], [None], np.array([0]))
    assert r["n"] == 0 and r["n_no_fit"] == 1 and np.isnan(r["rate"]) and np.isnan(r["chance"])


def test_best_match_empty_b():
    u_min, j = best_match(np.empty((3, 0)), ["f"] * 3, [])
    assert u_min.shape == j.shape == (3,) and np.isnan(u_min).all() and (j == -1).all()
