"""U13-A/U13-B: Monte Carlo survival + the real DiCE run, wired into the runner.

Both library pieces existed and neither had ever been called by the runner, so
the shipped 81.6% agreement number came from `physics_constrained_counterfactual`
alone and plan criterion 7 ("survival rates above 0.85", "DiCE counterfactuals")
was answerable by nothing on disk. These tests pin the wiring, not the maths --
the maths is covered by test_cf_monte_carlo.py and test_xai_counterfactuals.py.

The existing agreement number must not move: every assertion here is additive.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))


def _table(tmp_path, n_omega=8):
    """A minimal two-dataset table with a real stability lobe to notch against."""
    rows = []
    rng = np.random.default_rng(0)
    for ds in (1, 2):
        for i in range(n_omega):
            omega = 2000.0 + 500.0 * i
            blim = 1.0 + 0.5 * np.sin(omega / 1500.0) ** 2          # a lobe
            for b in (0.5 * blim, 1.6 * blim):                       # stable / chatter
                rows.append({
                    "dataset": ds,
                    "file": f"ds{ds}_w{omega:.0f}_b{b:.2f}",
                    "omega": omega,
                    "b_mm": float(b),
                    "blim": float(blim),
                    "label": int(b > blim),
                    "physics_usable": True,
                    # Not named "prob": --predictions merges a prob column IN, and a
                    # same-named table column would collide into prob_x/prob_y.
                    "p_true": float(np.clip(b / (2 * blim) + 0.05 * rng.standard_normal(), 0, 1)),
                })
    df = pd.DataFrame(rows)
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    return path, df


def _predictions(tmp_path, df):
    """A per-file probability file, so the surrogate is a REGRESSOR (DiCE needs one)."""
    path = tmp_path / "preds.csv"
    df[["file", "p_true"]].rename(columns={"p_true": "prob"}).to_csv(path, index=False)
    return path


# ---- U13-A: Monte Carlo survival ------------------------------------------ #

def test_runner_reports_monte_carlo_survival(tmp_path):
    """The missing half of plan criterion 7 must land in the manifest."""
    import counterfactuals_phase3 as C

    table, _ = _table(tmp_path)
    out = tmp_path / "cf"
    rc = C.main(["--table", str(table), "--dataset", "1",
                 "--output-dir", str(out), "--fig-dir", str(tmp_path / "figs"),
                 "--tag", "synth", "--mc-draws", "64"])
    assert rc == 0

    m = json.loads((out / "counterfactuals_manifest.json").read_text())
    survival = m["pooled"]["survival"]
    assert survival["n_draws"] == 64
    # The plan's criterion is answered by ONE field.
    assert "n_above_0_85" in survival and "frac_above_0_85" in survival
    assert 0.0 <= survival["mean_survival"] <= 1.0


def test_survival_is_additive_and_never_moves_the_agreement_number(tmp_path):
    """The 81.6% came from validate_against_blim; wiring survival must not touch it."""
    import counterfactuals_phase3 as C

    table, _ = _table(tmp_path)
    common = ["--table", str(table), "--dataset", "1",
              "--fig-dir", str(tmp_path / "figs"), "--tag", "synth"]

    off = tmp_path / "off"
    C.main(common + ["--output-dir", str(off), "--no-monte-carlo"])
    on = tmp_path / "on"
    C.main(common + ["--output-dir", str(on), "--mc-draws", "32"])

    m_off = json.loads((off / "counterfactuals_manifest.json").read_text())
    m_on = json.loads((on / "counterfactuals_manifest.json").read_text())

    assert m_off["pooled"]["agreement_rate"] == m_on["pooled"]["agreement_rate"]
    assert m_off["pooled"]["n_hits"] == m_on["pooled"]["n_hits"]
    assert m_off["pooled"].get("survival") is None      # opt-out really opts out


def test_survival_is_reproducible_for_a_fixed_seed(tmp_path):
    """A risk rate that moves between identical runs is not a risk rate."""
    import counterfactuals_phase3 as C

    table, _ = _table(tmp_path)
    means = []
    for name in ("a", "b"):
        out = tmp_path / name
        C.main(["--table", str(table), "--dataset", "1", "--output-dir", str(out),
                "--fig-dir", str(tmp_path / "figs"), "--tag", "synth",
                "--mc-draws", "64", "--mc-seed", "7"])
        m = json.loads((out / "counterfactuals_manifest.json").read_text())
        means.append(m["pooled"]["survival"]["mean_survival"])
    assert means[0] == means[1]


# ---- U13-B: the real DiCE run --------------------------------------------- #

def test_runner_runs_dice_on_the_same_points(tmp_path):
    """Two independent CF generators on identical inputs -- the validation a reviewer wants."""
    pytest.importorskip("dice_ml")
    import counterfactuals_phase3 as C

    table, df = _table(tmp_path)
    out = tmp_path / "cf"
    rc = C.main(["--table", str(table), "--dataset", "1",
                 "--output-dir", str(out), "--fig-dir", str(tmp_path / "figs"),
                 "--tag", "synth", "--predictions", str(_predictions(tmp_path, df)),
                 "--dice", "--max-cases", "3", "--no-monte-carlo"])
    assert rc == 0

    m = json.loads((out / "counterfactuals_manifest.json").read_text())
    dice = m["pooled"]["dice"]
    assert dice["n_cases"] == 3
    assert "n_found" in dice and "n_physically_stable" in dice
    assert dice["backend"] == "dice_ml"


def test_dice_refuses_the_classifier_surrogate_rather_than_mislabelling_it(tmp_path):
    """DiCE is wired as a REGRESSOR; without --predictions the surrogate is a classifier.

    Running it anyway would silently produce counterfactuals against the wrong
    model type -- a false number is worse than a missing one.
    """
    pytest.importorskip("dice_ml")
    import counterfactuals_phase3 as C

    table, _ = _table(tmp_path)
    out = tmp_path / "cf"
    rc = C.main(["--table", str(table), "--dataset", "1",
                 "--output-dir", str(out), "--fig-dir", str(tmp_path / "figs"),
                 "--tag", "synth", "--dice", "--no-monte-carlo"])
    assert rc == 0
    m = json.loads((out / "counterfactuals_manifest.json").read_text())
    assert m["pooled"]["dice"]["error"]
    assert "predictions" in m["pooled"]["dice"]["error"]


def test_dice_is_off_by_default(tmp_path):
    """It is the slow path; the default run must stay what it always was."""
    import counterfactuals_phase3 as C

    table, _ = _table(tmp_path)
    out = tmp_path / "cf"
    C.main(["--table", str(table), "--dataset", "1", "--output-dir", str(out),
            "--fig-dir", str(tmp_path / "figs"), "--tag", "synth", "--no-monte-carlo"])
    m = json.loads((out / "counterfactuals_manifest.json").read_text())
    assert m["pooled"].get("dice") is None


def test_a_missing_dice_ml_does_not_take_down_the_survival_arm(tmp_path, monkeypatch):
    """dice_ml is imported inside dice_counterfactuals, OUTSIDE its try block.

    If it is not installed on the cluster, an unguarded ImportError would abort
    the whole run and lose the Monte Carlo survival results too -- which are the
    half of criterion 7 that does not depend on DiCE at all.
    """
    import counterfactuals_phase3 as C
    from pri2m.xai import counterfactuals as cf_mod

    def _no_dice(*a, **k):
        raise ImportError("No module named 'dice_ml'")

    monkeypatch.setattr(cf_mod, "dice_counterfactuals", _no_dice)

    table, df = _table(tmp_path)
    out = tmp_path / "cf"
    rc = C.main(["--table", str(table), "--dataset", "1",
                 "--output-dir", str(out), "--fig-dir", str(tmp_path / "figs"),
                 "--tag", "synth", "--predictions", str(_predictions(tmp_path, df)),
                 "--dice", "--max-cases", "2", "--mc-draws", "32"])
    assert rc == 0

    m = json.loads((out / "counterfactuals_manifest.json").read_text())
    assert "dice_ml" in m["pooled"]["dice"]["error"]
    assert m["pooled"]["survival"]["n_draws"] == 32     # survival survived


# ---- U13-B: comparing the two generators fairly --------------------------- #
# The first cut compared the DIRECTION each generator moved depth. That is not a
# comparison: physics_constrained_counterfactual's b_range caps b at b_base, so
# it can only ever hold or lower depth, while DiCE varies both axes freely. It
# scored "chose a different lever" as "disagrees" -- on the real 42 cases the
# physics search left depth untouched in 35 and hopped omega, DiCE lowered depth
# in 35 and often held omega, and the old metric read that as 12/38 agreement.

def _cf_result(file, omega_base, b_base, omega_cf, b_cf, flips=True):
    return {"file": file, "omega_base": omega_base, "b_base": b_base,
            "omega_cf": omega_cf, "b_cf": b_cf, "flips": flips}


def _stub_dice(monkeypatch, moves):
    """Make dice_counterfactuals return a scripted (omega, b) per call."""
    from pri2m.xai import counterfactuals as cf_mod

    calls = {"i": 0}

    def _fake(estimator, query_row, desired_range, **kw):
        omega, b = moves[calls["i"]]
        calls["i"] += 1
        if omega is None:
            return pd.DataFrame(columns=["omega", "b_mm"])
        return pd.DataFrame({"omega": [omega], "b_mm": [b]})

    monkeypatch.setattr(cf_mod, "dice_counterfactuals", _fake)


def test_dice_reports_which_lever_each_generator_pulled(monkeypatch):
    """Speed-hop vs depth-cut is the finding; it must be visible, not collapsed."""
    import counterfactuals_phase3 as C

    cf_results = [
        # physics hops omega at unchanged depth; DiCE cuts depth at same speed.
        _cf_result("a", 12000.0, 2.0, 9500.0, 2.0),
        # both cut depth.
        _cf_result("b", 3000.0, 3.0, 3000.0, 1.0),
    ]
    _stub_dice(monkeypatch, [(12000.0, 0.58), (3000.0, 0.72)])

    out = C.run_dice(object(), {"source": "model_predictions"}, cf_results,
                     lambda w: 2.5, flip_threshold=0.5, seed=0, total_cfs=2)

    assert out["axis_breakdown"]["dice"]["b_mm"] == 2
    assert out["axis_breakdown"]["physics"]["omega"] == 1
    assert out["axis_breakdown"]["physics"]["b_mm"] == 1
    assert out["per_case"][0]["dice_axis"] == "b_mm"
    assert out["per_case"][0]["physics_axis"] == "omega"


def test_outcome_agreement_is_measured_on_where_the_moves_land(monkeypatch):
    """The real question: do both generators end up in the stable region?"""
    import counterfactuals_phase3 as C

    cf_results = [
        _cf_result("a", 12000.0, 2.0, 9500.0, 1.0),   # physics stable (1.0 < 2.5)
        _cf_result("b", 3000.0, 3.0, 3000.0, 3.0),    # physics NOT stable (3.0 > 2.5)
    ]
    _stub_dice(monkeypatch, [(12000.0, 0.58), (3000.0, 4.0)])  # stable, then not

    out = C.run_dice(object(), {"source": "model_predictions"}, cf_results,
                     lambda w: 2.5, flip_threshold=0.5, seed=0, total_cfs=2)

    # Case a: both stable. Case b: neither stable. Both are AGREEMENTS.
    assert out["n_both_stable"] == 1
    assert out["n_outcome_agrees"] == 2
    assert out["outcome_agreement_rate"] == 1.0
    assert out["per_case"][1]["physically_stable"] is False
    assert out["per_case"][1]["physics_stable"] is False


def test_the_misleading_direction_metric_is_gone(monkeypatch):
    """Regression guard: it read lever choice as disagreement."""
    import counterfactuals_phase3 as C

    _stub_dice(monkeypatch, [(1000.0, 0.5)])
    out = C.run_dice(object(), {"source": "model_predictions"},
                     [_cf_result("a", 2000.0, 2.0, 1500.0, 2.0)],
                     lambda w: 2.5, flip_threshold=0.5, seed=0, total_cfs=2)

    assert "n_agrees_with_physics_cf" not in out
    assert "agrees_with_physics_cf" not in out["per_case"][0]


def test_build_surrogate_exposes_the_fitted_estimator(tmp_path):
    """DiCE needs the estimator itself, not the predict closure around it."""
    import counterfactuals_phase3 as C

    _, df = _table(tmp_path)
    predict, info, est = C.build_surrogate(df[df.dataset == 1], None, 0)
    assert hasattr(est, "predict")
    assert callable(predict)
    assert info["n_train"] > 0
