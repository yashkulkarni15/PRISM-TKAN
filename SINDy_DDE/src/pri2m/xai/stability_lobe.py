"""Model-implied stability lobe diagram (SLD) — a *global* XAI probe.

Regenerative milling chatter has a classic representation in the machining-native
**(spindle speed ω, axial depth of cut b)** plane: the *stability lobe diagram*.
Below the analytic limit ``blim(ω)`` a cut is stable; above it, chatter. This
module reads a trained TKAN's chatter decision back out into that plane and
compares its implied boundary to the dataset's own precomputed ``blim`` curve.

Approach (A) of the design spec (``docs/superpowers/specs/
2026-07-03-stability-lobe-diagram-design.md``): the model never sees ``b`` and
sees ω only through the real vibration signal + side scalars, so we do **not**
synthesize inputs. Every real file already *is* a labeled (ω, b) operating point;
we bin real, z-scored windows by their (ω, b) cell, read
``mean sigmoid(chatter_logit)`` per cell → a P(chatter) field, then:

* :func:`extract_model_boundary` — per ω-column, the lowest ``b`` where P crosses
  0.5 (linear interpolation between bracketing cells). Columns that never cross
  are flagged censored (``no_crossing_low`` / ``no_crossing_high``), never
  invented.
* :func:`analytic_boundary` — the dataset's unique ``blim(ω)`` (``blim > 0``).
* :func:`boundary_faithfulness` — a scalar *shape* metric: mean |b_model − blim|
  (mm), plus relative error, stable-region IoU, analytic-label agreement, and the
  per-cell sample AUC (flagged when it ceilings).

The b-sampling is ragged and boundary-adaptive, so extraction is **per-column**
(uses only sampled cells) rather than a regular-grid ``contour(P==0.5)`` that
would invent values in un-sampled regions.

Heavy/optional deps (``matplotlib``, ``sklearn``) are imported lazily inside the
functions so importing this module never hard-fails.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

__all__ = [
    "build_pchatter_field",
    "extract_model_boundary",
    "analytic_boundary",
    "boundary_faithfulness",
    "plot_sld",
    "sld_caveats",
]

_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic sigmoid, applied elementwise."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _chatter_proba(model, batch: dict) -> np.ndarray:
    """Run ``model`` and return ``sigmoid(chatter_logit)`` as ``np[B]``."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logit = model(batch)["chatter_logit"]
    finally:
        if was_training:
            model.train()
    return _sigmoid(logit.detach().reshape(-1).cpu().numpy())


# --------------------------------------------------------------------------- #
# 1. P(chatter) field over real (ω, b) cells
# --------------------------------------------------------------------------- #
def build_pchatter_field(model, rows, npz_root, norm, cfg, max_batch: int = 256) -> list[dict]:
    """Bin real windows by their (ω, b) cell and read mean P(chatter) per cell.

    ``rows`` is the modal group's per-file table (already filtered to
    ``blim > 0`` + existing NPZ by the caller). Windows are loaded and z-scored
    by :class:`~pri2m.data.phase3_sequence.Phase3SequenceDataset` exactly as in
    training, then grouped by their file's ``(omega, b_mm)`` from the table (so
    param-twin datasets that share a grid *pool* into the same cells). The model
    runs on each cell's real windows with their true ``tau``/``omega``/``tpf``/
    ``fn``/``fs``; nothing is synthesized.

    Returns a list of cell dicts sorted by ``(omega, b_mm)``, each with keys
    ``omega, b_mm, blim, label, p_chatter, n_windows``.
    """
    from pri2m.data.phase3_sequence import Phase3SequenceDataset, collate_windows

    rows = rows.reset_index(drop=True)
    ds = Phase3SequenceDataset(rows, npz_root, cfg, norm=norm, supervised_only=False)

    # (dataset, file) -> (omega, b_mm, blim, label) from the authoritative table.
    meta: dict[tuple[int, str], tuple[float, float, float, int]] = {}
    for _, r in rows.iterrows():
        key = (int(r["dataset"]), str(r["file"]))
        meta[key] = (float(r["omega"]), float(r["b_mm"]), float(r["blim"]), int(r["label"]))

    # Materialize items once, group window indices by their (ω, b) cell.
    items = [ds[i] for i in range(len(ds))]
    cells: dict[tuple[float, float], dict] = {}
    for i, it in enumerate(items):
        key = (int(it["dataset"]), str(it["file"]))
        info = meta.get(key)
        if info is None:
            continue
        omega, b_mm, blim, label = info
        ck = (round(omega, 6), round(b_mm, 9))
        rec = cells.setdefault(
            ck, {"omega": omega, "b_mm": b_mm, "blim": blim, "label": label, "idx": []}
        )
        rec["idx"].append(i)

    field: list[dict] = []
    for rec in cells.values():
        idxs = rec["idx"]
        probs: list[np.ndarray] = []
        for start in range(0, len(idxs), max_batch):
            batch = collate_windows([items[j] for j in idxs[start : start + max_batch]])
            probs.append(_chatter_proba(model, batch))
        p = np.concatenate(probs) if probs else np.array([])
        field.append({
            "omega": float(rec["omega"]),
            "b_mm": float(rec["b_mm"]),
            "blim": float(rec["blim"]),
            "label": int(rec["label"]),
            "p_chatter": float(p.mean()) if p.size else float("nan"),
            "n_windows": int(p.size),
        })
    field.sort(key=lambda c: (c["omega"], c["b_mm"]))
    return field


# --------------------------------------------------------------------------- #
# 2. Model-implied boundary (per ω-column 0.5 crossing)
# --------------------------------------------------------------------------- #
def _column_blim(cells: list[dict]) -> float:
    """The column's analytic ``blim`` (unique ``blim > 0`` among its cells)."""
    vals = [float(c["blim"]) for c in cells if np.isfinite(c["blim"]) and c["blim"] > 0]
    return float(vals[0]) if vals else float("nan")


def _lowest_crossing(bs, ps, threshold):
    """Lowest ``b`` where P rises through ``threshold`` (linear interp).

    Returns ``(b_model, status, crossings)``. ``status`` is ``"ok"`` when a
    stable→unstable crossing exists; otherwise ``no_crossing_low`` (P never
    reaches ``threshold``) or ``no_crossing_high`` (P never drops below it), with
    ``b_model = nan`` — a censored column is never given an invented crossing.
    """
    bs = [float(b) for b in bs]
    ps = [float(p) for p in ps]
    crossings: list[float] = []
    for i in range(len(ps) - 1):
        if ps[i] < threshold <= ps[i + 1]:
            span = ps[i + 1] - ps[i]
            frac = (threshold - ps[i]) / span if span else 0.0
            crossings.append(bs[i] + frac * (bs[i + 1] - bs[i]))
    if crossings:
        return float(min(crossings)), "ok", crossings
    if all(p < threshold for p in ps):
        return float("nan"), "no_crossing_low", []
    if all(p >= threshold for p in ps):
        return float("nan"), "no_crossing_high", []
    # Mixed but no rising crossing (only falling): classify by the top of the range.
    status = "no_crossing_low" if ps[-1] < threshold else "no_crossing_high"
    return float("nan"), status, []


def extract_model_boundary(field: list[dict], threshold: float = _THRESHOLD) -> list[dict]:
    """Per ω-column, the lowest ``b`` where the P field crosses ``threshold``.

    Groups the field by ω, sorts each column by ``b_mm``, and returns one entry
    per column ``{omega, blim, b_model, status, crossings}`` sorted by ω. See
    :func:`_lowest_crossing` for the censoring statuses.
    """
    by_omega: dict[float, list[dict]] = defaultdict(list)
    for c in field:
        by_omega[round(float(c["omega"]), 6)].append(c)

    boundary: list[dict] = []
    for omega in sorted(by_omega):
        cells = sorted(by_omega[omega], key=lambda c: float(c["b_mm"]))
        blim = _column_blim(cells)
        bs = [c["b_mm"] for c in cells]
        ps = [c["p_chatter"] for c in cells]
        b_model, status, crossings = _lowest_crossing(bs, ps, threshold)
        boundary.append({
            "omega": float(omega),
            "blim": float(blim),
            "b_model": float(b_model),
            "status": status,
            "crossings": [float(x) for x in crossings],
        })
    return boundary


# --------------------------------------------------------------------------- #
# 3. Analytic overlay
# --------------------------------------------------------------------------- #
def analytic_boundary(rows) -> list[dict]:
    """Unique ``(omega, blim)`` for the group, ``blim > 0``, sorted by ω.

    ``blim`` is a pure function of ω within a modal group (verified constant per
    ``(dataset, omega)``), so this is the analytic stability lobe curve. Invariant
    to how many b-rows share each ω.
    """
    import pandas as pd

    df = pd.DataFrame(rows)[["omega", "blim"]].copy()
    df = df[np.isfinite(df["blim"]) & (df["blim"] > 0)]
    uniq = df.drop_duplicates(subset=["omega"]).sort_values("omega")
    return [{"omega": float(o), "blim": float(b)} for o, b in zip(uniq["omega"], uniq["blim"])]


# --------------------------------------------------------------------------- #
# 4. Global-faithfulness metric
# --------------------------------------------------------------------------- #
def _auc(labels, proba) -> float:
    """ROC-AUC of ``proba`` vs binary ``labels``; NaN if a single class."""
    labels = np.asarray(labels)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, np.asarray(proba)))


def boundary_faithfulness(field, boundary, threshold: float = _THRESHOLD) -> dict:
    """Scalar shape-faithfulness of the model boundary vs the analytic ``blim``.

    * ``mae_b`` (mm) / ``mape_b`` — mean absolute (and relative) boundary error
      over ω-columns with a finite model crossing and ``blim > 0``.
    * ``n_columns_used`` / ``n_columns_censored`` — so a high censor count can't
      hide behind a small MAE.
    * ``iou_stable`` — cell-wise IoU of the stable regions
      (``{p<0.5}`` vs ``{b<blim}``), robust to the ragged grid.
    * ``label_agreement`` — fraction of cells where the model's stable/unstable
      call matches the analytic label (interpolation-free accuracy).
    * ``per_cell_auc`` — AUC of ``p_chatter`` vs the true ``label`` per cell
      (ceiling flag lives in :func:`sld_caveats`).
    """
    cols = [b for b in boundary if np.isfinite(b["blim"]) and b["blim"] > 0]
    errs, rels, n_used, n_censored = [], [], 0, 0
    for b in cols:
        if np.isfinite(b["b_model"]):
            e = abs(float(b["b_model"]) - float(b["blim"]))
            errs.append(e)
            rels.append(e / float(b["blim"]))
            n_used += 1
        else:
            n_censored += 1

    cells = [c for c in field if np.isfinite(c["blim"]) and c["blim"] > 0
             and np.isfinite(c["p_chatter"])]
    model_stable = np.array([c["p_chatter"] < threshold for c in cells], dtype=bool)
    analytic_stable = np.array([c["b_mm"] < c["blim"] for c in cells], dtype=bool)
    inter = int(np.sum(model_stable & analytic_stable))
    union = int(np.sum(model_stable | analytic_stable))
    iou = float(inter / union) if union else float("nan")
    agree = float(np.mean(model_stable == analytic_stable)) if cells else float("nan")

    per_cell_auc = _auc([c["label"] for c in cells], [c["p_chatter"] for c in cells])

    return {
        "mae_b": float(np.mean(errs)) if errs else float("nan"),
        "mape_b": float(np.mean(rels)) if rels else float("nan"),
        "iou_stable": iou,
        "label_agreement": agree,
        "per_cell_auc": per_cell_auc,
        "n_columns_used": int(n_used),
        "n_columns_censored": int(n_censored),
    }


# --------------------------------------------------------------------------- #
# 5. Caveats
# --------------------------------------------------------------------------- #
def sld_caveats(metrics: dict, n_omega: int) -> list[str]:
    """Honesty caveats implied by the metric: per-cell ceiling + heavy censoring."""
    out: list[str] = []
    auc = metrics.get("per_cell_auc")
    if auc is not None and np.isfinite(auc) and auc > 0.99:
        out.append(
            f"- **Per-cell sample AUC is at the ceiling ({auc:.4g})**: easy cells "
            "saturate P->{0,1} and the crossing is set by a few near-boundary "
            "cells. The SLD *shape* metric (MAE_b) is independent of this ceiling, "
            "but the absolute per-cell AUC is NOT the model's performance."
        )
    censored = metrics.get("n_columns_censored", 0)
    if n_omega and censored / n_omega > 0.3:
        out.append(
            f"- **{censored}/{n_omega} ω-columns are censored** (their sampled "
            "b-range never brackets P=0.5), so MAE_b covers only the crossed "
            "columns. Pool more param-twin datasets to thicken the grid; a "
            "crossing is never extrapolated."
        )
    return out


# --------------------------------------------------------------------------- #
# 6. Figure
# --------------------------------------------------------------------------- #
def plot_sld(field, boundary, analytic, metrics, path, title: str | None = None) -> str:
    """Render the SLD: P(chatter) field + model boundary + analytic ``blim``.

    Scatter of sampled (ω, b) cells colored by ``p_chatter``, the model-implied
    boundary ``b_model(ω)`` (finite columns), and the analytic ``blim(ω)`` curve,
    annotated with the faithfulness metrics. Returns the path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(8, 5))
    if field:
        w = [c["omega"] for c in field]
        b = [c["b_mm"] for c in field]
        p = [c["p_chatter"] for c in field]
        sc = ax.scatter(w, b, c=p, cmap="coolwarm", vmin=0.0, vmax=1.0,
                        s=40, edgecolors="k", linewidths=0.3, zorder=2)
        fig.colorbar(sc, ax=ax, label="P(chatter)")

    mw = [b["omega"] for b in boundary if np.isfinite(b["b_model"])]
    mb = [b["b_model"] for b in boundary if np.isfinite(b["b_model"])]
    if mw:
        ax.plot(mw, mb, "-o", color="black", lw=1.8, ms=4,
                label="model boundary (P=0.5)", zorder=4)

    if analytic:
        aw = [a["omega"] for a in analytic]
        ab = [a["blim"] for a in analytic]
        ax.plot(aw, ab, "--", color="tab:green", lw=2.0,
                label="analytic blim(ω)", zorder=3)

    ax.set_xlabel("spindle speed ω (rpm)")
    ax.set_ylabel("axial depth of cut b (mm)")
    ax.set_title(title or "Model-implied stability lobe diagram")
    ax.legend(loc="upper right", fontsize=8)

    mae = metrics.get("mae_b")
    mape = metrics.get("mape_b")
    iou = metrics.get("iou_stable")
    agree = metrics.get("label_agreement")
    txt = (
        f"MAE_b = {mae:.3g} mm\n"
        f"MAPE_b = {100 * mape:.1f}%\n"
        f"IoU_stable = {iou:.3g}\n"
        f"label agr. = {agree:.3g}\n"
        f"cols used/censored = {metrics.get('n_columns_used')}/"
        f"{metrics.get('n_columns_censored')}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
