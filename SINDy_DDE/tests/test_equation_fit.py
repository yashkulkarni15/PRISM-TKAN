import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from pri2m.xai import equation_fit
from pri2m.xai.equation_fit import (
    bootstrap_selections, fidelity, fit_formula, jaccard, ridge_contrast, same_equation,
)

TERM_NAMES = [f"t{i}" for i in range(16)]


def _planted(seed=0):
    rng = np.random.default_rng(seed)
    T = rng.normal(size=(600, 16))
    g = np.repeat(np.arange(60), 10)
    y = 2.0 * T[:, 0] - 1.5 * T[:, 5] + 0.05 * rng.normal(size=600)
    return T, y, g, rng


def test_recovers_planted_two_term_formula():
    T, y, g, _ = _planted()
    f = fit_formula(T, y, g, TERM_NAMES)
    assert set(f["selected"]) == {TERM_NAMES[0], TERM_NAMES[5]}
    assert f["coef"][TERM_NAMES[0]] == pytest.approx(2.0, rel=0.1)
    assert f["coef"][TERM_NAMES[5]] == pytest.approx(-1.5, rel=0.1)
    assert f["intercept"] == pytest.approx(0.0, abs=0.05)
    assert f["model"].predict(T).shape == (600,)


def test_grouped_cv_never_splits_a_group(monkeypatch):
    T, y, g, _ = _planted()
    seen = []
    real_split = GroupKFold.split

    def spy(self, X, y=None, groups=None):
        for tr, te in real_split(self, X, y, groups):
            seen.append((tr, te))
            yield tr, te

    monkeypatch.setattr(GroupKFold, "split", spy)
    fit_formula(T, y, g, TERM_NAMES)
    assert seen
    for tr, te in seen:
        assert not set(g[tr]) & set(g[te])
    # every group is a test group in exactly one fold of the first alpha's CV
    first = seen[:5]
    tested = np.concatenate([g[te] for _, te in first])
    assert sorted(set(tested)) == list(range(60)) and len(set(tested)) == 60
    for gid in range(60):
        assert sum(gid in set(g[te]) for _, te in first) == 1


def test_fidelity_threshold():
    T, y, g, rng = _planted()
    f = fit_formula(T, y, g, TERM_NAMES)
    assert fidelity(f, T, y)["faithful"] is True
    assert fidelity(f, T, y)["r2"] > 0.99
    assert fidelity(f, T, rng.normal(size=600))["faithful"] is False


def test_jaccard():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_bootstrap_recovers_planted_pair_in_most_draws():
    T, y, g, _ = _planted()
    sets = bootstrap_selections(T, y, g, TERM_NAMES, n_boot=30, seed=1)
    assert len(sets) == 30
    assert all({TERM_NAMES[0], TERM_NAMES[5]} <= s for s in sets)
    # measured 23-24/30 exact with original ids; the fresh-id leak gave 13/30
    assert sum(s == {TERM_NAMES[0], TERM_NAMES[5]} for s in sets) >= 18


def test_bootstrap_draws_whole_groups_with_original_ids(monkeypatch):
    rng = np.random.default_rng(3)
    sizes = 4 + np.arange(40) % 5  # unequal group sizes 4..8
    g = np.repeat(np.arange(40), sizes)
    T = rng.normal(size=(len(g), 16))
    y = 2.0 * T[:, 0] - 1.5 * T[:, 5] + 0.05 * rng.normal(size=len(g))
    passed = []
    real_split = GroupKFold.split

    def spy(self, X, y=None, groups=None):
        passed.append(np.asarray(groups))
        return real_split(self, X, y, groups)

    monkeypatch.setattr(GroupKFold, "split", spy)
    bootstrap_selections(T, y, g, TERM_NAMES, n_boot=3, seed=2)
    assert passed
    for gr in passed:
        ids, counts = np.unique(gr, return_counts=True)
        assert set(ids) <= set(range(40))  # original ids, not relabelled
        assert all(c % sizes[i] == 0 for i, c in zip(ids, counts))  # complete groups only
    assert any(c > sizes[i] for gr in passed for i, c in zip(*np.unique(gr, return_counts=True)))  # some drawn twice


def test_back_transform_to_original_units():
    rng = np.random.default_rng(4)
    scale = 10.0 * (np.arange(16) + 1); offset = 100.0 * np.arange(16)
    T = rng.normal(size=(600, 16)) * scale + offset
    g = np.repeat(np.arange(60), 10)
    y = 0.3 * T[:, 0] - 0.02 * T[:, 5] + 7.0 + 0.01 * rng.normal(size=600)
    f = fit_formula(T, y, g, TERM_NAMES)
    assert f["coef"][TERM_NAMES[0]] == pytest.approx(0.3, rel=0.05)
    assert f["coef"][TERM_NAMES[5]] == pytest.approx(-0.02, rel=0.05)
    assert f["intercept"] == pytest.approx(7.0, abs=1.0)
    manual = T[:, 0] * f["coef"][TERM_NAMES[0]] + T[:, 5] * f["coef"][TERM_NAMES[5]] + f["intercept"]
    assert np.allclose(manual, f["model"].predict(T), atol=1e-6)


def test_same_equation_needs_two_fits():
    with pytest.raises(ValueError, match="at least 2"):
        same_equation([{"selected": ["a"], "coef": {"a": 1}}], [{"a"}] * 10)


def test_same_equation_rule():
    fits = [{"selected": ["a", "b"], "coef": {"a": 1, "b": -1}}] * 4
    assert same_equation(fits, [{"a", "b"}] * 200)["same"] is True
    flipped = fits[:3] + [{"selected": ["a", "b"], "coef": {"a": -1, "b": -1}}]
    out = same_equation(flipped, [{"a", "b"}] * 200)
    assert out["sign_consistent"] is False and out["same"] is False


def test_same_equation_fails_when_seeds_disagree_more_than_bootstrap_noise():
    fits = [{"selected": [s], "coef": {s: 1}} for s in "abcd"]
    out = same_equation(fits, [{"a", "b"}] * 200)
    assert out["median_cross_jaccard"] == 0.0 and out["same"] is False


def test_ridge_contrast():
    w = np.zeros(1000); w[:9] = 1.0; w[9:] = 0.001
    r = ridge_contrast(w)
    assert r["k90"] == 9
    assert set(r["top_idx"][:9]) == set(range(9)) and len(r["top_idx"]) == 20
