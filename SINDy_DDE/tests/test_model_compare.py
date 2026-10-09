"""Tests for the TKAN-vs-MiniROCKET noise comparison (``pri2m.xai.model_compare``).

Every reading is tested on synthetic per-file CSVs whose truth is constructed:

* TKAN's ranking degrades with the noise tier while MiniROCKET's does not ->
  H1 SUPPORTED;
* both models identical -> H1 not supported;
* a model's files missing -> that section UNDECIDED with a reason, no number;
* TKAN's scores shifted up under noise with intact ranking -> the accuracy gap
  at threshold 0 shows up as a threshold shift;
* injected level 5 behaving like the real 5 % tier -> injection valid; not -> invalid.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pri2m.xai import model_compare as MC

FLOOR = 0.0092


# --------------------------------------------------------------------------- #
# Synthetic builders
# --------------------------------------------------------------------------- #
def _write(rows, out, model, fold, split):
    df = pd.DataFrame(rows)
    df["model"], df["fold"], df["split"] = model, fold, split
    MC.write_score_file(df, out, {"synthetic": True})


def make_world(out, *, tkan_noise_sens=0.0, mr_noise_sens=0.0, tkan_shift=0.0,
               folds=range(5), groups_per_fold=12, files_per_ds=6, seed=0,
               models=("tkan", "minirocket_A"), side_off_delta=None,
               inj_matches_real=True, levels=(0.0, 1.0, 5.0, 10.0)):
    """Per-file CSVs for every (model, fold, split).

    Each parent group is a clean dataset D (1-30) with mirrors D+60 / D+90 and
    ``files_per_ds`` files each; file k of D and of D+90 are the same cut.
    Score = signal(label) + noise with sd ``1 + sens * tier``; ``tkan_shift``
    adds ``shift * tier`` to TKAN's scores (moves both classes together).
    The real 5 % tier and the injected level 5 use the same rule when
    ``inj_matches_real``; otherwise injected noise is 4x stronger.
    """
    rng = np.random.default_rng(seed)
    sens = {"tkan": tkan_noise_sens, "minirocket_A": mr_noise_sens,
            "minirocket_B": mr_noise_sens, "tkan_side_off": tkan_noise_sens}
    ds_next = 1
    for fold in folds:
        cuts = []
        for g in range(groups_per_fold):
            D = ds_next
            ds_next = ds_next % 30 + 1
            for k in range(files_per_ds):
                lab = int(rng.random() < 0.5)
                cuts.append((D, k, lab, rng.uniform(1000, 9000), rng.uniform(0.3, 1.8),
                             rng.standard_normal(4)))
        held_cuts = [(121 + (i % 5), i, int(rng.random() < 0.5), rng.uniform(1000, 9000),
                      rng.uniform(0.3, 1.8), rng.standard_normal(2)) for i in range(200)]
        for model in models:
            s = sens[model]
            shift = tkan_shift if model.startswith("tkan") else 0.0

            def sc(lab, tier, z, extra=0.0):
                return (2 * lab - 1) * 1.0 + z * (1.0 + s * tier) + shift * tier + extra

            if model != "tkan_side_off":
                val, noise = [], []
                for D, k, lab, om, bob, z in cuts:
                    for tier, dd in ((0, D), (1, D + 60), (5, D + 90)):
                        val.append(dict(level=0.0, file=f"time{dd}_{k}.h5", dataset=dd,
                                        parent_group=D, noise_tier=tier, label=lab, omega=om,
                                        b_over_blim=bob, n_windows=20,
                                        score=sc(lab, tier, z[{0: 0, 1: 1, 5: 2}[tier]])))
                    for lev in levels:
                        # Level 5 reuses the real 5 % tier's draw: same noise law.
                        zz = z[2] if lev == 5.0 else z[0]
                        eff = lev if (inj_matches_real or lev != 5.0) else 4 * lev
                        noise.append(dict(level=lev, file=f"time{D}_{k}.h5", dataset=D,
                                          parent_group=D, noise_tier=0, label=lab, omega=om,
                                          b_over_blim=bob, n_windows=8,
                                          score=sc(lab, eff, zz)))
                _write(val, out, model, fold, "val")
                _write(noise, out, model, fold, "val_noise")
            extra = 0.0
            if model == "tkan_side_off" and side_off_delta is not None:
                extra = side_off_delta
            held = [dict(level=0.0, file=f"time{d}_{i}.h5", dataset=d, parent_group=d,
                         noise_tier=5.0, label=lab, omega=om, b_over_blim=bob, n_windows=20,
                         score=sc(lab, 5, z[0]) + (extra * (2 * lab - 1) * -1))
                    for d, i, lab, om, bob, z in held_cuts]
            _write(held, out, model, fold, "heldout")
    return out


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_weighted_auc_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(1)
    s = np.round(rng.standard_normal(300), 1)  # ties
    y = (rng.random(300) < 0.4).astype(int)
    w = rng.integers(0, 4, 300).astype(float)
    assert MC.weighted_auc(s, y, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))
    assert MC.weighted_auc(s, y) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(MC.weighted_auc(s, np.zeros(300)))


def test_best_threshold_accuracy():
    s = np.array([-3.0, -2.0, -1.0, 0.5, 1.0, 2.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    acc, thr = MC.best_threshold_accuracy(s + 5.0, y)  # all >= 0: acc@0 = 0.5
    assert acc == 1.0 and -1.0 + 5 < thr <= 0.5 + 5


def test_write_score_file_rejects_missing_columns(tmp_path):
    with pytest.raises(ValueError):
        MC.write_score_file(pd.DataFrame({"model": ["x"]}), tmp_path, {})


def test_loader_ignores_incomplete_or_bad_schema(tmp_path):
    make_world(tmp_path, folds=[0], groups_per_fold=3)
    # Incomplete sidecar -> ignored.
    side = tmp_path / "scores_tkan_fold0_val.json"
    d = json.loads(side.read_text())
    d["complete"] = False
    side.write_text(json.dumps(d))
    # Wrong column order -> ignored.
    p = tmp_path / "scores_minirocket_A_fold0_heldout.csv"
    df = pd.read_csv(p)
    df[df.columns[::-1]].to_csv(p, index=False)
    data = MC.load_scores(tmp_path)
    assert ("tkan", 0, "val") not in data.frames
    assert ("minirocket_A", 0, "heldout") not in data.frames
    assert ("minirocket_A", 0, "val") in data.frames
    reasons = " ".join(data.problems)
    assert "complete" in reasons and "columns" in reasons


# --------------------------------------------------------------------------- #
# A: T1 by noise tier
# --------------------------------------------------------------------------- #
def test_h1_supported_when_tkan_degrades_and_mr_does_not(tmp_path):
    make_world(tmp_path, tkan_noise_sens=0.6, mr_noise_sens=0.0)
    rep = MC.build_report(tmp_path, floor=FLOOR, n_boot=300, seed=0)
    a = rep["A_noise_tier"]
    assert a["h1"]["reading"] == "SUPPORTED"
    assert a["h1"]["diff_of_drops_mean"] > FLOOR
    assert a["h1"]["ci95"][0] > 0
    assert a["h1"]["n_complete_groups_per_fold"]["0"] == 12
    t = a["per_model"]["tkan"]["summary"]
    assert t["0"]["auc_mean"] > t["5"]["auc_mean"] + 0.05
    assert "pre_committed" in a["h1"]
    # Paired per-tier differences exist with CIs.
    assert set(a["paired_tkan_minus_mr"]) == {"0", "1", "5"}
    assert a["paired_tkan_minus_mr"]["5"]["ci95"][1] < 0


def test_h1_not_supported_when_models_identical(tmp_path):
    make_world(tmp_path, tkan_noise_sens=0.3, mr_noise_sens=0.3)
    rep = MC.build_report(tmp_path, floor=FLOOR, n_boot=300, seed=0)
    assert rep["A_noise_tier"]["h1"]["reading"] in ("NOT_SUPPORTED", "INCONCLUSIVE")
    assert rep["A_noise_tier"]["h1"]["diff_of_drops_mean"] == pytest.approx(0.0, abs=1e-12)


def test_h1_bootstrap_is_seeded(tmp_path):
    make_world(tmp_path, tkan_noise_sens=0.4, folds=[0, 1])
    r1 = MC.build_report(tmp_path, floor=FLOOR, n_boot=100, seed=3)
    r2 = MC.build_report(tmp_path, floor=FLOOR, n_boot=100, seed=3)
    assert r1["A_noise_tier"]["h1"]["ci95"] == r2["A_noise_tier"]["h1"]["ci95"]


def test_missing_mr_makes_tier_section_undecided(tmp_path):
    make_world(tmp_path, models=("tkan",))
    rep = MC.build_report(tmp_path, floor=FLOOR, n_boot=50, seed=0)
    a = rep["A_noise_tier"]
    assert a["h1"]["reading"] == "UNDECIDED"
    assert a["h1"]["diff_of_drops_mean"] is None
    assert "minirocket_A" in a["h1"]["reason"]
    # Per-model numbers for the model that IS there are still reported.
    assert a["per_model"]["tkan"]["summary"]["0"]["auc_mean"] is not None


def test_empty_dir_is_all_undecided(tmp_path):
    rep = MC.build_report(tmp_path, floor=FLOOR, n_boot=10, seed=0)
    for key in ("A_noise_tier", "B_score_shift", "C_heldout", "D_dose_response",
                "E_side_inputs"):
        assert rep[key]["reading"] == "UNDECIDED", key
        assert rep[key]["reason"]


# --------------------------------------------------------------------------- #
# B: score shift
# --------------------------------------------------------------------------- #
def test_threshold_shift_detected_when_scores_move_together(tmp_path):
    make_world(tmp_path, tkan_shift=0.4)  # +2.0 at the 5 % tier, ranking intact
    rep = MC.build_report(tmp_path, floor=FLOOR, n_boot=10, seed=0)
    b = rep["B_score_shift"]["per_model"]
    assert b["tkan"]["reading"] == "THRESHOLD_SHIFT"
    assert b["minirocket_A"]["reading"] == "NO_MATERIAL_SHIFT"
    tk = b["tkan"]["summary"]
    assert tk["heldout"]["median_label0_mean"] > tk["val_tier0"]["median_label0_mean"] + 1.5
    assert tk["heldout"]["shift_label0_sep_units"] > 0.5


# --------------------------------------------------------------------------- #
# C: held-out
# --------------------------------------------------------------------------- #
def test_heldout_breakdowns_and_gap(tmp_path):
    make_world(tmp_path, tkan_noise_sens=0.6)
    rep = MC.build_report(tmp_path, floor=FLOOR, n_boot=10, seed=0)
    c = rep["C_heldout"]
    tk = c["per_model"]["tkan"]
    assert len(tk["per_fold"]) == 5
    assert set(tk["by_dataset"]) == {"121", "122-125"}
    assert set(tk["by_b_over_blim"]) == {"<0.8", "0.8-1.2", ">1.2"}
    assert set(tk["by_omega_tertile"]) == {"T1", "T2", "T3"}
    g = c["val5_predicts_heldout"]["tkan"]
    # Held-out is built with the 5 % tier rule, so the 5 % gap is the small one.
    assert abs(g["gap_vs_val_tier5_mean"]) < abs(g["gap_vs_val_all_mean"])
    assert g["reading"] in ("PREDICTS", "DOES_NOT_PREDICT")


# --------------------------------------------------------------------------- #
# D: dose-response
# --------------------------------------------------------------------------- #
def test_dose_response_slopes_and_injection_valid(tmp_path):
    make_world(tmp_path, tkan_noise_sens=0.6, mr_noise_sens=0.0)
    rep = MC.build_report(tmp_path, floor=FLOOR, n_boot=10, seed=0)
    d = rep["D_dose_response"]
    assert d["per_model"]["tkan"]["slope_mean"] < -0.005
    assert abs(d["per_model"]["minirocket_A"]["slope_mean"]) < 0.005
    inj = d["injection_validity"]
    assert inj["reading"] == "VALID"
    for m in ("tkan", "minirocket_A"):
        assert inj["per_model"][m]["n_matched_files_per_fold"]["0"] == 12 * 6
        assert abs(inj["per_model"][m]["diff_mean"]) <= 0.01


def test_injection_invalid_when_injected_noise_too_strong(tmp_path):
    make_world(tmp_path, tkan_noise_sens=0.6, inj_matches_real=False)
    rep = MC.build_report(tmp_path, floor=FLOOR, n_boot=10, seed=0)
    d = rep["D_dose_response"]
    assert d["injection_validity"]["reading"] == "INVALID"
    assert d["t2_status"] == "descriptive_only"


def test_injection_undecided_without_level5(tmp_path):
    make_world(tmp_path, levels=(0.0, 1.0, 10.0))
    rep = MC.build_report(tmp_path, floor=FLOOR, n_boot=10, seed=0)
    inj = rep["D_dose_response"]["injection_validity"]
    assert inj["reading"] == "UNDECIDED" and "level 5" in inj["reason"]


# --------------------------------------------------------------------------- #
# E: side inputs
# --------------------------------------------------------------------------- #
def test_side_inputs_effect(tmp_path):
    make_world(tmp_path, models=("tkan", "minirocket_A", "tkan_side_off"),
               side_off_delta=0.8)
    rep = MC.build_report(tmp_path, floor=FLOOR, n_boot=10, seed=0)
    e = rep["E_side_inputs"]
    assert e["reading"] == "SIDE_INPUTS_MATTER"
    assert e["diff_mean"] > FLOOR and e["n_folds"] == 5


def test_side_inputs_undecided_without_side_off(tmp_path):
    make_world(tmp_path)
    rep = MC.build_report(tmp_path, floor=FLOOR, n_boot=10, seed=0)
    assert rep["E_side_inputs"]["reading"] == "UNDECIDED"
    assert rep["E_side_inputs"]["diff_mean"] is None


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def test_report_is_strict_json_and_markdown_and_figures(tmp_path):
    sd = make_world(tmp_path / "s", tkan_noise_sens=0.5)
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"summary": {"rel_sigma_per_unit": [0.01] * 4,
                                           "linear_ok": True}}))
    out, figs = tmp_path / "rep", tmp_path / "figs"
    rep = MC.run(sd, calibration=cal, floor=FLOOR, n_boot=50, seed=0,
                 output_dir=out, figures_dir=figs)
    txt = (out / "noise_compare_report.json").read_text()
    json.loads(txt, parse_constant=lambda c: pytest.fail(f"non-strict JSON {c}"))
    assert rep["calibration"]["rel_sigma_per_unit"] == [0.01] * 4
    md = (out / "noise_compare_summary.md").read_text()
    assert "H1" in md and "SUPPORTED" in md
    for f in rep["figures"]:
        assert (figs / f).is_file()
    assert len(rep["figures"]) == 3


def test_cli_runs_on_synthetic_scores(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    sd = make_world(tmp_path / "s", tkan_noise_sens=0.5, folds=[0, 1])
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts" / "compare_noise_phase3.py"),
                        "--scores-dir", str(sd), "--calibration", str(tmp_path / "none.json"),
                        "--n-boot", "20", "--output-dir", str(tmp_path / "rep"),
                        "--figures-dir", str(tmp_path / "figs")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "A_noise_tier" in r.stdout
    rep = json.loads((tmp_path / "rep" / "noise_compare_report.json").read_text())
    assert rep["calibration"]["status"] == "missing"
