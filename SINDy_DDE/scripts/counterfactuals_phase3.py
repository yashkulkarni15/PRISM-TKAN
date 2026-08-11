#!/usr/bin/env python3
"""Physics-constrained counterfactuals for the Phase 3 chatter model (eval-only).

For every *chatter* operating point in a machine configuration this runner asks
the actionable question -- "what change in spindle speed ``omega`` or depth of
cut ``b`` flips this window to stable?" -- by searching ONLY those two axes
(:func:`pri2m.xai.counterfactuals.physics_constrained_counterfactual`). It then
*validates* each recommended move against the analytic stability boundary
``blim(omega)`` read straight from ``phase3_dataset.csv``
(:func:`pri2m.xai.counterfactuals.validate_against_blim`): does the counterfactual
land in the physically stable region ``b < blim(omega)``? The headline metric is
the **agreement rate** -- the fraction of model-flipping cases whose CF is
genuinely below the analytic limit.

Standalone: ``blim`` is interpolated directly from the per-row ``omega``/``blim``
columns of the table (no stability-lobe module is imported).

The model's ``(omega, b) -> P(chatter)`` decision surface is represented by a
surrogate. Two sources are supported:

* ``--predictions CSV`` (columns ``file`` + one of ``prob``/``severity``/
  ``chatter_prob``): the audited model's per-window outputs, fit with a
  regressor -> this audits the *model* against physics.
* default (no predictions): a classifier fit on the table's ``label`` -> this is
  the physics-consistency sanity check (a boundary-faithful classifier should
  agree with ``blim`` almost everywhere).

The ``--checkpoint`` is recorded for provenance (path + sha256); it is not
required for the search because the model cannot be queried at synthetic
``(omega, b)`` without regenerating signals, which is out of scope here.

Example::

    python scripts/counterfactuals_phase3.py \
        --table results/phase3/phase3_dataset.csv \
        --dataset 1 \
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \
        --tag fold2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pri2m.xai import counterfactuals as cf  # noqa: E402

CONTROLLABLE = ("omega", "b_mm")


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def build_blim_fn(table: pd.DataFrame):
    """Interpolator ``omega -> blim`` from the table's (omega, blim) columns.

    ``blim`` is a pure function of ``omega`` within a machine configuration (the
    stability lobe diagram), so we deduplicate on ``omega`` and linearly
    interpolate; ``np.interp`` clamps outside the observed speed range.
    """
    pts = (
        table[["omega", "blim"]]
        .dropna()
        .drop_duplicates("omega")
        .sort_values("omega")
    )
    xs = pts["omega"].to_numpy(dtype=float)
    ys = pts["blim"].to_numpy(dtype=float)

    def blim_fn(omega):
        return np.interp(np.asarray(omega, dtype=float), xs, ys)

    return blim_fn, xs, ys


def build_surrogate(table: pd.DataFrame, predictions: Path | None, seed: int):
    """Return ``(predict, info, estimator)`` for the ``(omega, b) -> P(chatter)`` surface.

    With ``predictions`` we fit a regressor to the model's per-window chatter
    probability; otherwise a classifier to the table ``label``.

    The fitted estimator is returned alongside the closure because DiCE takes
    the sklearn object itself (``dice_ml.Model(model=..., backend="sklearn")``),
    not a callable -- and it must be the SAME fit the physics-constrained search
    used, or the two generators would not be answering the same question.
    """
    X = table[["omega", "b_mm"]].to_numpy(dtype=float)

    if predictions is not None:
        preds = pd.read_csv(predictions)
        col = next(
            (c for c in ("prob", "chatter_prob", "severity", "p_chatter") if c in preds.columns),
            None,
        )
        if col is None:
            raise ValueError(
                f"--predictions {predictions} has no prob/chatter_prob/severity column; "
                f"found {list(preds.columns)}"
            )
        merged = table.merge(preds[["file", col]], on="file", how="inner")
        if merged.empty:
            raise ValueError("no rows matched between --predictions and --table on 'file'")
        from sklearn.ensemble import RandomForestRegressor

        est = RandomForestRegressor(n_estimators=300, random_state=seed)
        est.fit(merged[["omega", "b_mm"]].to_numpy(dtype=float), merged[col].to_numpy(dtype=float))
        predict = lambda pts: np.clip(est.predict(np.atleast_2d(pts)), 0.0, 1.0)  # noqa: E731
        info = {"source": "model_predictions", "predictions": str(predictions), "target_col": col,
                "n_train": int(len(merged)), "estimator": "RandomForestRegressor"}
        return predict, info, est

    from sklearn.ensemble import RandomForestClassifier

    y = table["label"].to_numpy(dtype=int)
    est = RandomForestClassifier(n_estimators=300, random_state=seed)
    est.fit(X, y)
    # index of the positive (chatter) class
    pos = int(np.where(est.classes_ == 1)[0][0]) if 1 in est.classes_ else 1
    predict = lambda pts: est.predict_proba(np.atleast_2d(pts))[:, pos]  # noqa: E731
    info = {"source": "table_label", "n_train": int(len(table)), "estimator": "RandomForestClassifier"}
    return predict, info, est


def run_dataset(
    table: pd.DataFrame,
    dataset_id: int,
    predict,
    blim_fn,
    *,
    b_floor: float,
    n_b: int,
    flip_threshold: float,
    max_cases: int | None,
):
    """Generate + validate physics-constrained CFs for one machine config."""
    d = table[(table.dataset == dataset_id) & (table.get("physics_usable", True))].copy()
    chatter = d[d.label == 1]
    if max_cases is not None:
        chatter = chatter.head(max_cases)

    omega_grid = np.sort(d.omega.unique()).astype(float)  # realistic discrete speeds

    cf_results = []
    for _, row in chatter.iterrows():
        b0 = float(row.b_mm)
        res = cf.physics_constrained_counterfactual(
            predict,
            {"omega": float(row.omega), "b_mm": b0},
            omega_grid=omega_grid,                 # hop between lobes...
            b_range=(b_floor, b0),                 # ...or lower the depth
            n_b=n_b,
            flip_threshold=flip_threshold,
        )
        res["file"] = str(row.file)
        res["dataset"] = int(dataset_id)
        res["label"] = int(row.label)
        cf_results.append(res)

    report = cf.validate_against_blim(cf_results, blim_fn)
    return cf_results, report, omega_grid


def pool_survival(blocks: list[dict]) -> dict | None:
    """Combine per-dataset survival reports into one pooled record.

    Rates are recomputed from the concatenated per-case survivals rather than
    averaged across datasets, so a dataset with three cases cannot weigh as much
    as one with forty. Unscored cases (no counterfactual found) stay excluded
    exactly as the library excludes them.
    """
    if not blocks:
        return None
    scored = [s for blk in blocks for s in
              (r["survival"] for r in blk["per_case"]) if s is not None]
    n_above = sum(int(blk["n_above_0_85"]) for blk in blocks)
    head = blocks[0]
    return {
        "n_cases": sum(int(b["n_cases"]) for b in blocks),
        "n_flips": sum(int(b["n_flips"]) for b in blocks),
        "n_scored": len(scored),
        "n_above_0_85": n_above,
        "frac_above_0_85": (n_above / len(scored)) if scored else None,
        "mean_survival": float(np.mean(scored)) if scored else None,
        "median_survival": float(np.median(scored)) if scored else None,
        "min_survival": float(np.min(scored)) if scored else None,
        "n_draws": head["n_draws"],
        "seed": head["seed"],
        "uncertainty": head["uncertainty"],
        "threshold": cf.SURVIVAL_THRESHOLD,
    }


def pool_dice(blocks: list[dict]) -> dict | None:
    """Combine per-dataset DiCE reports. An errored dataset propagates its error."""
    if not blocks:
        return None
    errored = [b for b in blocks if b.get("error")]
    if errored:
        return {"error": errored[0]["error"],
                "n_cases": sum(int(b.get("n_cases", 0)) for b in blocks),
                "n_found": 0}
    n_cases = sum(int(b["n_cases"]) for b in blocks)
    n_found = sum(int(b["n_found"]) for b in blocks)
    n_stable = sum(int(b["n_physically_stable"]) for b in blocks)
    n_comparable = sum(int(b["n_comparable"]) for b in blocks)
    n_agrees = sum(int(b["n_outcome_agrees"]) for b in blocks)
    axes = {gen: {k: sum(int(b["axis_breakdown"][gen][k]) for b in blocks)
                  for k in ("omega", "b_mm", "both", "none")}
            for gen in ("dice", "physics")}
    head = blocks[0]
    return {
        "backend": head["backend"],
        "method": head["method"],
        "n_cases": n_cases,
        "n_found": n_found,
        "n_physically_stable": n_stable,
        "found_rate": (n_found / n_cases) if n_cases else None,
        "physically_stable_rate": (n_stable / n_found) if n_found else None,
        "n_comparable": n_comparable,
        "n_both_stable": sum(int(b["n_both_stable"]) for b in blocks),
        "n_outcome_agrees": n_agrees,
        "outcome_agreement_rate": (n_agrees / n_comparable) if n_comparable else None,
        "axis_breakdown": axes,
        "n_same_axis": sum(int(b["n_same_axis"]) for b in blocks),
        "axis_note": head["axis_note"],
        "total_CFs": head["total_CFs"],
        "seed": head["seed"],
    }


AXIS_NOTE = (
    "axis_breakdown/n_same_axis describe WHICH knob each generator turned. "
    "physics_constrained_counterfactual searches b in (b_floor, b_base], so "
    "it can only hold or lower depth; DiCE varies both axes freely. Use "
    "outcome_agreement_rate to compare the methods."
)


def axis_of(omega_base, b_base, omega_cf, b_cf) -> str:
    """Which lever did this generator pull?"""
    moved_w = not np.isclose(omega_cf, omega_base, rtol=1e-6, atol=1e-9)
    moved_b = not np.isclose(b_cf, b_base, rtol=1e-6, atol=1e-9)
    if moved_w and moved_b:
        return "both"
    if moved_w:
        return "omega"
    if moved_b:
        return "b_mm"
    return "none"


def compare_generator_outcomes(pairs: list[dict]) -> dict:
    """Compare the two counterfactual generators on WHERE THEY LAND.

    Deliberately pure and free of both the network and ``blim_fn``: every input
    it needs is a coordinate or a stability flag that the caller has already
    resolved. That is what lets a stored manifest be repaired offline with the
    identical code the live runner uses -- see
    ``scripts/repair_counterfactual_manifest.py``.

    Each pair carries the DiCE move (``dice_omega_cf``/``dice_b_cf``/
    ``dice_stable``) and, when the physics search also produced one, its move
    under ``physics_*``. A pair without a physics move still counts toward the
    DiCE axis tally but is not comparable, mirroring the original loop.
    """
    empty = {"omega": 0, "b_mm": 0, "both": 0, "none": 0}
    axes = {"dice": dict(empty), "physics": dict(empty)}
    per_case, n_same_axis, n_both_stable, n_agrees, n_comparable = [], 0, 0, 0, 0

    for pair in pairs:
        omega_base, b_base = pair["omega_base"], pair["b_base"]
        d_axis = axis_of(omega_base, b_base, pair["dice_omega_cf"], pair["dice_b_cf"])
        axes["dice"][d_axis] += 1
        rec = {"dice_axis": d_axis}

        if pair.get("physics_omega_cf") is not None and pair.get("physics_b_cf") is not None:
            p_axis = axis_of(omega_base, b_base,
                             pair["physics_omega_cf"], pair["physics_b_cf"])
            axes["physics"][p_axis] += 1
            stable, p_stable = bool(pair["dice_stable"]), bool(pair["physics_stable"])
            same_axis, agrees = d_axis == p_axis, stable == p_stable
            rec.update({"physics_axis": p_axis, "physics_stable": p_stable,
                        "same_axis": same_axis, "outcome_agrees": agrees})
            n_comparable += 1
            n_same_axis += int(same_axis)
            n_both_stable += int(stable and p_stable)
            n_agrees += int(agrees)
        per_case.append(rec)

    return {
        # Agreement on the OUTCOME (both land stable, or neither does).
        "n_comparable": n_comparable,
        "n_both_stable": n_both_stable,
        "n_outcome_agrees": n_agrees,
        "outcome_agreement_rate": (n_agrees / n_comparable) if n_comparable else None,
        # Which lever each generator pulled. NOT an agreement measure: the
        # physics search cannot raise depth by construction, so a differing axis
        # is a difference in strategy, not a contradiction.
        "axis_breakdown": axes,
        "n_same_axis": n_same_axis,
        "axis_note": AXIS_NOTE,
        "per_case": per_case,
    }


def run_dice(estimator, surro_info, cf_results, blim_fn, *,
             flip_threshold: float, seed: int, total_cfs: int) -> dict:
    """U13-B: DiCE on the SAME operating points, validated the same way.

    Two independent counterfactual generators agreeing on identical inputs is a
    check the plan never asked for and a reviewer will. The comparison is only
    meaningful against the model-prediction surrogate: DiCE is wired here as a
    regressor, so a classifier surrogate is refused rather than mislabelled.
    """
    if surro_info.get("source") != "model_predictions":
        return {
            "error": "DiCE needs the regressor surrogate; re-run with --predictions "
                     f"(surrogate source was {surro_info.get('source')!r})",
            "n_cases": len(cf_results), "n_found": 0,
        }

    per_case, n_found, n_stable = [], 0, 0
    pairs, pair_rows = [], []
    for res in cf_results:
        # Column ORDER matters: the estimator was fitted on ["omega", "b_mm"]
        # positionally, and DiCE hands it a frame built from these keys.
        query = {"omega": [float(res["omega_base"])], "b_mm": [float(res["b_base"])]}
        try:
            cf_df = cf.dice_counterfactuals(
                estimator, query, [0.0, float(flip_threshold)],
                total_CFs=int(total_cfs), seed=int(seed),
            )
        except ImportError as exc:
            # dice_ml is imported OUTSIDE dice_counterfactuals' own try block, so
            # an absent package would otherwise abort the run and take the
            # survival results -- which owe nothing to DiCE -- down with it.
            return {"error": f"DiCE unavailable: {exc}",
                    "n_cases": len(cf_results), "n_found": 0}
        rec = {"file": res.get("file"), "omega_base": res["omega_base"],
               "b_base": res["b_base"], "found": bool(len(cf_df))}
        if len(cf_df):
            n_found += 1
            omega_d = float(cf_df["omega"].iloc[0])
            b_d = float(cf_df["b_mm"].iloc[0])
            stable = bool(b_d < float(blim_fn(omega_d)))
            rec.update({"omega_cf": omega_d, "b_cf": b_d,
                        "physically_stable": stable})
            n_stable += int(stable)

            # Compare the two generators on WHERE THEY LAND, not on which lever
            # they pulled: the physics search's b_range caps b at b_base, so it
            # structurally cannot raise depth while DiCE varies both axes. Both
            # ending stable -- or both failing to -- is the real agreement. The
            # comparison itself lives in compare_generator_outcomes so the
            # offline manifest repair runs the identical code.
            pair = {"omega_base": res["omega_base"], "b_base": res["b_base"],
                    "dice_omega_cf": omega_d, "dice_b_cf": b_d,
                    "dice_stable": stable}
            if res.get("omega_cf") is not None and res.get("b_cf") is not None:
                p_omega, p_b = float(res["omega_cf"]), float(res["b_cf"])
                pair.update({"physics_omega_cf": p_omega, "physics_b_cf": p_b,
                             "physics_stable": bool(p_b < float(blim_fn(p_omega)))})
            pairs.append(pair)
            pair_rows.append(rec)
        per_case.append(rec)

    comparison = compare_generator_outcomes(pairs)
    for rec, annotation in zip(pair_rows, comparison["per_case"]):
        rec.update(annotation)

    return {
        "backend": "dice_ml",
        "method": "random",
        "n_cases": len(cf_results),
        "n_found": n_found,
        "n_physically_stable": n_stable,
        "found_rate": (n_found / len(cf_results)) if cf_results else None,
        "physically_stable_rate": (n_stable / n_found) if n_found else None,
        **{k: v for k, v in comparison.items() if k != "per_case"},
        "total_CFs": int(total_cfs),
        "seed": int(seed),
        "per_case": per_case,
    }


def make_figure(
    table: pd.DataFrame,
    dataset_ids: list[int],
    per_dataset: dict,
    out_path: Path,
    tag: str,
):
    """CF moves on the ``(omega, b)`` plane vs the analytic ``blim`` curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(dataset_ids)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.0), squeeze=False)
    for ax, ds in zip(axes[0], dataset_ids):
        pack = per_dataset[ds]
        xs, ys = pack["blim_omega"], pack["blim_values"]
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], "-", color="#333333",
                lw=2.0, label="blim(omega) (stable below)", zorder=1)
        ax.fill_between(np.asarray(xs)[order], 0, np.asarray(ys)[order],
                        color="#4c9f70", alpha=0.10, zorder=0)

        for res, pc in zip(pack["cf_results"], pack["report"]["per_case"]):
            ob, bb = res["omega_base"], res["b_base"]
            ax.scatter([ob], [bb], c="#c0392b", s=34, zorder=3, edgecolors="k", linewidths=0.3)
            if res.get("omega_cf") is None:
                continue
            oc, bc = res["omega_cf"], res["b_cf"]
            col = "#1f77b4" if pc["hit"] else "#e67e22"  # blue=hit, orange=miss
            ax.annotate(
                "", xy=(oc, bc), xytext=(ob, bb),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.3, alpha=0.9), zorder=2,
            )
            ax.scatter([oc], [bc], c=col, s=26, marker="s", zorder=3,
                       edgecolors="k", linewidths=0.3)

        ar = pack["report"]["agreement_rate"]
        ax.set_title(f"dataset {ds}: agreement={ar:.0%}" if ar is not None else f"dataset {ds}")
        ax.set_xlabel("spindle speed  omega (rpm)")
        ax.set_ylabel("depth of cut  b (mm)")
        ax.grid(alpha=0.25)

    handles = [
        Line2D([0], [0], color="#333333", lw=2, label="blim(omega)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0392b",
               markeredgecolor="k", markersize=8, label="chatter base"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="k", markersize=8, label="CF hit (stable)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e67e22",
               markeredgecolor="k", markersize=8, label="CF miss (unstable)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"Physics-constrained counterfactuals vs stability boundary [{tag}]", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", default="results/phase3/phase3_dataset.csv",
                   help="Phase 3 per-file table CSV (source of omega, b, blim, label).")
    p.add_argument("--dataset", default="1",
                   help="Machine-config id, or comma list (e.g. '1,2'). 'all' = every dataset.")
    p.add_argument("--checkpoint", default=None,
                   help="Trained checkpoint (recorded for provenance; not required for search).")
    p.add_argument("--predictions", default=None,
                   help="Optional CSV of model per-window probs (columns: file, prob/severity).")
    p.add_argument("--output-dir", default="results/phase3/xai/counterfactuals")
    p.add_argument("--fig-dir", default="results/figures/research_update_assets")
    p.add_argument("--tag", default="fold2")
    p.add_argument("--b-floor", type=float, default=0.1, help="Smallest actionable depth (mm).")
    p.add_argument("--n-b", type=int, default=80, help="Depth-axis grid resolution.")
    p.add_argument("--flip-threshold", type=float, default=0.5, help="P(chatter) below this = stable.")
    p.add_argument("--max-cases", type=int, default=None, help="Cap chatter cases per dataset.")
    p.add_argument("--seed", type=int, default=0)
    # U13-A: Monte Carlo survival. On by default -- a recommendation that only
    # holds at the nominal point is not one a machinist can act on.
    p.add_argument("--mc-draws", type=int, default=512,
                   help="Monte Carlo draws per counterfactual (0 = nominal only).")
    p.add_argument("--mc-seed", type=int, default=0)
    p.add_argument("--mc-margin", type=float, default=0.0,
                   help="A draw is stable only when blim - b > margin (mm).")
    p.add_argument("--no-monte-carlo", action="store_true",
                   help="Skip survival rates entirely.")
    # U13-B: the real DiCE path. Off by default -- it is the slow arm.
    p.add_argument("--dice", action="store_true",
                   help="Also generate DiCE counterfactuals on the same points. "
                        "Needs --predictions (DiCE is wired as a regressor).")
    p.add_argument("--dice-total-cfs", type=int, default=2)
    args = p.parse_args(argv)

    table = pd.read_csv(args.table)
    if args.dataset.strip().lower() == "all":
        dataset_ids = [int(x) for x in sorted(table.dataset.unique())]
    else:
        dataset_ids = [int(x) for x in args.dataset.split(",") if x.strip()]

    predictions = Path(args.predictions) if args.predictions else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    ckpt_prov = None
    if ckpt_path is not None:
        if ckpt_path.is_file():
            ckpt_prov = {"path": str(ckpt_path), "sha256": _sha256(ckpt_path)}
        else:
            print(f"WARNING: checkpoint not found ({ckpt_path}); recording path only.",
                  file=sys.stderr)
            ckpt_prov = {"path": str(ckpt_path), "sha256": None, "found": False}

    per_dataset: dict = {}
    all_cf, all_per_case = [], []
    for ds in dataset_ids:
        d_tab = table[table.dataset == ds]
        blim_fn, blim_omega, blim_values = build_blim_fn(d_tab)
        # Fit the surrogate on this config's rows so its decision surface is
        # comparable to that config's blim curve.
        predict, surro_info, estimator = build_surrogate(d_tab, predictions, args.seed)
        cf_results, report, omega_grid = run_dataset(
            table, ds, predict, blim_fn,
            b_floor=args.b_floor, n_b=args.n_b,
            flip_threshold=args.flip_threshold, max_cases=args.max_cases,
        )
        per_dataset[ds] = {
            "cf_results": cf_results,
            "report": report,
            "blim_omega": blim_omega.tolist(),
            "blim_values": blim_values.tolist(),
            "omega_grid": omega_grid.tolist(),
            "surrogate": surro_info,
        }
        if not args.no_monte_carlo:
            per_dataset[ds]["survival"] = cf.monte_carlo_survival(
                cf_results, blim_fn, n_draws=args.mc_draws,
                seed=args.mc_seed, margin=args.mc_margin,
            )
        if args.dice:
            per_dataset[ds]["dice"] = run_dice(
                estimator, surro_info, cf_results, blim_fn,
                flip_threshold=args.flip_threshold, seed=args.seed,
                total_cfs=args.dice_total_cfs,
            )
        all_cf.extend(cf_results)
        all_per_case.extend(report["per_case"])

    # Pooled aggregate across every dataset run.
    flips = [pc for pc in all_per_case if pc["flips"]]
    n_flips = len(flips)
    n_hits = sum(1 for pc in flips if pc["hit"])
    toward = [pc["moves_toward_stable"] for pc in flips]
    pooled = {
        "n_cases": len(all_per_case),
        "n_flips": n_flips,
        "n_hits": n_hits,
        "agreement_rate": (n_hits / n_flips) if n_flips else None,
        "moves_toward_stable_rate": (float(np.mean(toward)) if toward else None),
    }
    # Additive only: everything above is the number Update 9 shipped.
    survival_pooled = pool_survival(
        [per_dataset[ds]["survival"] for ds in dataset_ids if "survival" in per_dataset[ds]]
    )
    dice_pooled = pool_dice(
        [per_dataset[ds]["dice"] for ds in dataset_ids if "dice" in per_dataset[ds]]
    )
    if survival_pooled is not None:
        pooled["survival"] = survival_pooled
    if dice_pooled is not None:
        pooled["dice"] = dice_pooled

    # Figure(s).
    fig_name = f"counterfactuals_{args.tag}.png"
    fig_out = out_dir / fig_name
    make_figure(table, dataset_ids, per_dataset, fig_out, args.tag)
    asset_out = Path(args.fig_dir) / fig_name
    asset_out.parent.mkdir(parents=True, exist_ok=True)
    make_figure(table, dataset_ids, per_dataset, asset_out, args.tag)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "table": str(args.table),
        "checkpoint": ckpt_prov,
        "datasets": dataset_ids,
        "config": {
            "controllable_axes": list(CONTROLLABLE),
            "b_floor": args.b_floor,
            "n_b": args.n_b,
            "flip_threshold": args.flip_threshold,
            "max_cases": args.max_cases,
            "seed": args.seed,
        },
        "agreement_metric": (
            "agreement_rate = (# model-flipping CFs that satisfy b_cf < blim(omega_cf)) "
            "/ (# model-flipping CFs); a 'hit' = model flips window to stable AND the CF "
            "lies in the physically stable region."
        ),
        "pooled": pooled,
        "per_dataset": {
            str(ds): {
                "surrogate": per_dataset[ds]["surrogate"],
                **({"survival": per_dataset[ds]["survival"]}
                   if "survival" in per_dataset[ds] else {}),
                **({"dice": per_dataset[ds]["dice"]}
                   if "dice" in per_dataset[ds] else {}),
                "agreement_rate": per_dataset[ds]["report"]["agreement_rate"],
                "n_cases": per_dataset[ds]["report"]["n_cases"],
                "n_flips": per_dataset[ds]["report"]["n_flips"],
                "n_hits": per_dataset[ds]["report"]["n_hits"],
                "moves_toward_stable_rate": per_dataset[ds]["report"]["moves_toward_stable_rate"],
                "mean_distance_to_boundary_cf": per_dataset[ds]["report"]["mean_distance_to_boundary_cf"],
                "cases": [
                    {
                        "file": r["file"],
                        "omega_base": r["omega_base"], "b_base": r["b_base"],
                        "base_prob": r["base_prob"], "base_chatter": r["base_chatter"],
                        "omega_cf": r["omega_cf"], "b_cf": r["b_cf"],
                        "delta_omega": r["delta_omega"], "delta_b": r["delta_b"],
                        "cf_prob": r["cf_prob"], "flips": r["flips"],
                        "blim_at_cf": pc["blim_at_cf"],
                        "distance_to_boundary_cf": pc["distance_to_boundary_cf"],
                        "physics_stable_cf": pc["physics_stable_cf"],
                        "moves_toward_stable": pc["moves_toward_stable"],
                        "hit": pc["hit"],
                    }
                    for r, pc in zip(
                        per_dataset[ds]["cf_results"], per_dataset[ds]["report"]["per_case"]
                    )
                ],
            }
            for ds in dataset_ids
        },
        "artifacts": {
            "figure": str(fig_out),
            "figure_asset": str(asset_out),
            "manifest": str(out_dir / "counterfactuals_manifest.json"),
        },
    }
    manifest_path = out_dir / "counterfactuals_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "tag": args.tag,
        "datasets": dataset_ids,
        "pooled": pooled,
        "figure": str(fig_out),
        "manifest": str(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
