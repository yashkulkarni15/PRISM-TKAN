"""Integration test for the composed Phase 3 data-prep pipeline.

Synthesizes a tiny clean dataset with its 1% and 5% mirrors plus the two
audit pathologies (a negative-blim unstable row and a metrics_unavailable row),
writes the three input CSVs to a tmp dir, builds the table, and asserts the
leakage-safe + hygiene invariants hold end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pri2m.phase3.data_prep import build_phase3_table, fold_assignments, summarize
from pri2m.phase3.splits import parent_group


def _features_frame(datasets, n_per=6):
    rows = []
    for ds in datasets:
        for i in range(n_per):
            omega = 2000.0 + 500.0 * i
            b_mm = 0.5 + 0.5 * i
            blim = 1.0
            bob = b_mm / blim
            rows.append(
                {
                    "dataset": ds,
                    "file": f"f{i}.npz",
                    "omega": omega,
                    "b_mm": b_mm,
                    "blim": blim,
                    "b_over_blim": bob,
                    "label": int(bob >= 1.0),
                    "Forcex_kurt": float(i),
                    "xtpos_kurt": float(i),
                    "ytpos_kurt": float(i),
                    "Forcey_kurt": float(i),
                    "xtvel_kurt": float(i),
                    "ytvel_kurt": float(i),
                }
            )
    return pd.DataFrame(rows)


def _write_inputs(tmp_path):
    # Clean dataset 5 plus its mirrors 65 (1%) and 95 (5%): all parent_group 5.
    clean = _features_frame([5])
    noisy = _features_frame([65, 95])

    # Inject a negative-blim unstable row into clean dataset 5 (low-speed edge).
    clean.loc[len(clean)] = {
        "dataset": 5, "file": "lowspeed.npz", "omega": 1000.0, "b_mm": 3.0,
        "blim": -2.0, "b_over_blim": np.nan, "label": 1,
        "Forcex_kurt": np.nan, "xtpos_kurt": 1.0, "ytpos_kurt": 1.0,
        "Forcey_kurt": 1.0, "xtvel_kurt": 1.0, "ytvel_kurt": 1.0,
    }

    clean_csv = tmp_path / "features_clean.csv"
    noisy_csv = tmp_path / "features_noisy.csv"
    clean.to_csv(clean_csv, index=False)
    noisy.to_csv(noisy_csv, index=False)

    # Mode labels: give every (dataset,file) a noise_score; mark one row unavailable;
    # make the noisy 95 rows heavy on noise_score.
    feats_all = pd.concat([clean, noisy], ignore_index=True)
    ml_rows = []
    for _, r in feats_all.iterrows():
        ds, f = int(r["dataset"]), r["file"]
        if ds == 95:
            ns, mode = 50.0, "broadband_noise_dominated"
        elif ds == 65:
            ns, mode = 5.0, "mixed_tpf_chatter"
        else:
            ns, mode = 0.2, "tpf_dominated_clean"
        # one metrics_unavailable row in the clean set
        if ds == 5 and f == "f0.npz":
            ns, mode = np.nan, "metrics_unavailable"
        ml_rows.append({"dataset": ds, "file": f, "noise_score": ns, "mode_label": mode})
    ml = pd.DataFrame(ml_rows)
    ml_csv = tmp_path / "mode_labels.csv"
    ml.to_csv(ml_csv, index=False)
    return str(clean_csv), str(noisy_csv), str(ml_csv)


def test_build_table_invariants(tmp_path):
    clean_csv, noisy_csv, ml_csv = _write_inputs(tmp_path)
    table = build_phase3_table(clean_csv, noisy_csv, ml_csv)

    # parent_group present and mirrors collapse to clean parent 5.
    assert set(table["parent_group"].unique()) == {5}

    # Negative-blim row: excluded, physics-unusable, zero weight.
    neg = table[table["file"] == "lowspeed.npz"].iloc[0]
    assert neg["exclude_supervised"] and not neg["physics_usable"]
    assert neg["exclude_reason"] == "invalid_blim_low_speed"
    assert neg["sample_weight"] == 0.0

    # metrics_unavailable row: excluded, zero weight.
    mu = table[(table["dataset"] == 5) & (table["file"] == "f0.npz")].iloc[0]
    assert mu["exclude_supervised"] and mu["sample_weight"] == 0.0

    # Kurtosis NaN imputed and flagged.
    assert table["Forcex_kurt"].isna().sum() == 0
    assert bool(neg["kurt_imputed"]) is True

    # Noisy (95) rows down-weighted vs clean (5) usable rows.
    w_clean = table[(table["dataset"] == 5) & (table["sample_weight"] > 0)]["sample_weight"].mean()
    w_noisy = table[table["dataset"] == 95]["sample_weight"].mean()
    assert w_noisy < w_clean

    # Positive weights renormalized to mean 1.
    pos = table["sample_weight"][table["sample_weight"] > 0]
    assert pos.mean() == pytest.approx(1.0)


def test_folds_have_no_mirror_leakage(tmp_path):
    clean_csv, noisy_csv, ml_csv = _write_inputs(tmp_path)
    # Add more clean parents so we can build several folds.
    extra_clean = _features_frame([1, 2, 3, 4])
    extra_noisy = _features_frame([61, 62, 63, 64, 91, 92, 93, 94])
    pd.concat([pd.read_csv(clean_csv), extra_clean], ignore_index=True).to_csv(clean_csv, index=False)
    pd.concat([pd.read_csv(noisy_csv), extra_noisy], ignore_index=True).to_csv(noisy_csv, index=False)
    ml = pd.read_csv(ml_csv)
    extra = pd.concat([extra_clean, extra_noisy], ignore_index=True)[["dataset", "file"]]
    extra["noise_score"] = 1.0
    extra["mode_label"] = "mixed_tpf_chatter"
    pd.concat([ml, extra], ignore_index=True).to_csv(ml_csv, index=False)

    table = build_phase3_table(clean_csv, noisy_csv, ml_csv)
    spec = fold_assignments(table, n_splits=3, seed=0)
    # leakage_verified is set only if every fold passed assert_no_mirror_leakage.
    assert spec["leakage_verified"] is True
    # Each parent group maps to exactly one fold.
    pg = parent_group(table["dataset"])
    assert set(spec["group_to_fold"]) == set(int(g) for g in pg.unique())


def test_summary_counts(tmp_path):
    clean_csv, noisy_csv, ml_csv = _write_inputs(tmp_path)
    table = build_phase3_table(clean_csv, noisy_csv, ml_csv)
    s = summarize(table)
    assert s["n_negative_blim"] == 1
    assert s["n_metrics_unavailable"] == 1
    assert s["n_weight_zero"] >= 2  # negative-blim + metrics_unavailable
    assert s["weight_positive_mean"] == pytest.approx(1.0)
