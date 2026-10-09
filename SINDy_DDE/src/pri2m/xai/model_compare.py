"""TKAN vs MiniROCKET under noise: the shared per-file score schema + analyses.

Both model runners write one CSV per (model, fold, split) in the schema
:data:`SCORE_COLUMNS` (column order is part of the contract), with a JSON
sidecar of the same stem that carries provenance, AUC/accuracy per level and
``"complete": true``. The sidecar is written LAST, so a CSV without a complete
sidecar is a partial write and is ignored by the loader.

Score conventions (threshold 0 for both, so accuracy is comparable):

* TKAN: file-mean chatter logit (sigmoid(score) >= 0.5  <=>  score >= 0).
* MiniROCKET: file-mean ridge decision on +-1 targets (score >= 0).

AUC is rank-based, so it is computed on the score directly.

Pure pandas / numpy / sklearn; matplotlib is imported only inside the plotting
functions.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCORE_COLUMNS: tuple[str, ...] = (
    "model", "fold", "split", "level", "file", "dataset", "parent_group",
    "noise_tier", "label", "omega", "b_over_blim", "n_windows", "score",
)
SPLITS: tuple[str, ...] = ("val", "heldout", "val_noise")


def score_file_stem(model: str, fold: int, split: str) -> str:
    return f"scores_{model}_fold{int(fold)}_{split}"


def _level_key(level) -> str:
    return f"{float(level):g}"


def auc_acc(scores, labels) -> dict:
    """File-level AUC (None if single-class/empty) and accuracy at threshold 0.

    Non-finite scores are excluded and counted, never imputed.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    fin = np.isfinite(s) & np.isfinite(y)
    s, yk = s[fin], (y[fin] >= 0.5).astype(int)
    auc = None
    if yk.size and yk.min() != yk.max():
        auc = float(roc_auc_score(yk, s))
    acc = float(np.mean((s >= 0).astype(int) == yk)) if yk.size else None
    return {"auc": auc, "acc": acc, "n_files": int(yk.size),
            "n_nonfinite": int(np.count_nonzero(~fin))}


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    return o


def write_score_file(rows: pd.DataFrame, out_dir, provenance: dict) -> Path:
    """Write one (model, fold, split) CSV + its sidecar; returns the CSV path.

    ``rows`` must hold exactly one model/fold/split. Missing schema columns are
    an error (a silent blank column would look like real data downstream).
    Both files are written via a temp name + rename, sidecar last.
    """
    missing = [c for c in SCORE_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"score rows missing columns {missing}")
    df = rows.loc[:, list(SCORE_COLUMNS)].copy()
    for col in ("model", "fold", "split"):
        if df[col].nunique() != 1:
            raise ValueError(f"score rows must hold one {col}, got {sorted(df[col].unique())}")
    model, fold, split = str(df["model"].iloc[0]), int(df["fold"].iloc[0]), str(df["split"].iloc[0])
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = score_file_stem(model, fold, split)
    csv, side = out_dir / f"{stem}.csv", out_dir / f"{stem}.json"
    per_level = {}
    for lev, g in df.groupby("level", sort=True):
        per_level[_level_key(lev)] = auc_acc(g["score"], g["label"])
    sidecar = {
        "model": model, "fold": fold, "split": split,
        "columns": list(SCORE_COLUMNS),
        "provenance": provenance,
        "auc": {k: v["auc"] for k, v in per_level.items()},
        "acc": {k: v["acc"] for k, v in per_level.items()},
        "n_files": {k: v["n_files"] for k, v in per_level.items()},
        "n_rows": int(len(df)),
        "complete": True,
    }
    tmp = csv.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv)
    tmp = side.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_json_safe(sidecar), indent=2))
    os.replace(tmp, side)
    return csv


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
import re  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

_STEM_RE = re.compile(r"^scores_(.+?)_fold(\d+)_(val_noise|val|heldout)$")
MODELS_PAIRED = ("tkan", "minirocket_A")
TIERS = (0, 1, 5)
UNDECIDED = "UNDECIDED"


@dataclass
class ScoreData:
    """Every complete, schema-exact score CSV under a directory."""

    frames: dict = field(default_factory=dict)  # (model, fold, split) -> DataFrame
    problems: list = field(default_factory=list)

    def by_fold(self, model: str, split: str) -> dict:
        return {f: df for (m, f, s), df in sorted(self.frames.items())
                if m == model and s == split}

    def models(self, split: str) -> list:
        return sorted({m for (m, _, s) in self.frames if s == split})


def load_scores(scores_dir) -> ScoreData:
    """Load ``scores_<model>_fold<F>_<split>.csv`` with a complete sidecar.

    Anything else (no sidecar, ``complete`` not true, columns not exactly
    :data:`SCORE_COLUMNS`, name/content mismatch) is skipped and recorded in
    ``problems`` -- never partially used.
    """
    data = ScoreData()
    d = Path(scores_dir)
    if not d.is_dir():
        data.problems.append(f"scores dir {d} does not exist")
        return data
    for csv in sorted(d.glob("scores_*.csv")):
        m = _STEM_RE.match(csv.stem)
        if not m:
            data.problems.append(f"{csv.name}: unrecognised name")
            continue
        model, fold, split = m.group(1), int(m.group(2)), m.group(3)
        side = csv.with_suffix(".json")
        if not side.is_file():
            data.problems.append(f"{csv.name}: no sidecar (partial write?)")
            continue
        try:
            meta = json.loads(side.read_text())
        except json.JSONDecodeError as exc:
            data.problems.append(f"{csv.name}: unreadable sidecar ({exc})")
            continue
        if meta.get("complete") is not True:
            data.problems.append(f"{csv.name}: sidecar not complete")
            continue
        df = pd.read_csv(csv)
        if list(df.columns) != list(SCORE_COLUMNS):
            data.problems.append(f"{csv.name}: columns {list(df.columns)} != schema columns")
            continue
        if not (df.empty or ((df["model"].astype(str) == model).all()
                             and (df["fold"].astype(int) == fold).all()
                             and (df["split"].astype(str) == split).all())):
            data.problems.append(f"{csv.name}: model/fold/split columns disagree with name")
            continue
        data.frames[(model, fold, split)] = df
    return data


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
class _AucPrep:
    """Rank blocks of one score vector, so a weighted AUC is two bincounts."""

    def __init__(self, scores, labels):
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        y = (np.asarray(labels, dtype=np.float64).reshape(-1) >= 0.5).astype(np.float64)
        self.keep = np.isfinite(s)
        _, self.inv = np.unique(s[self.keep], return_inverse=True)
        self.nb = int(self.inv.max()) + 1 if self.inv.size else 0
        self.y = y[self.keep]

    def auc(self, weights=None) -> float:
        if self.nb == 0:
            return float("nan")
        w = (np.ones(self.y.size) if weights is None
             else np.asarray(weights, dtype=np.float64)[self.keep])
        P = np.bincount(self.inv, w * self.y, minlength=self.nb)
        N = np.bincount(self.inv, w * (1.0 - self.y), minlength=self.nb)
        tp, tn = P.sum(), N.sum()
        if tp <= 0 or tn <= 0:
            return float("nan")
        below = np.cumsum(N) - N
        return float((P * (below + 0.5 * N)).sum() / (tp * tn))


def weighted_auc(scores, labels, weights=None) -> float:
    """ROC AUC with optional sample weights (ties count half); NaN if one class."""
    return _AucPrep(scores, labels).auc(weights)


def best_threshold_accuracy(scores, labels) -> tuple[float, float]:
    """``(best accuracy, threshold)`` over all cut points (predict 1 iff s >= t).

    In-sample, so slightly optimistic; used only as a descriptive contrast with
    the fixed threshold 0.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = (np.asarray(labels, dtype=np.float64) >= 0.5).astype(np.float64)
    fin = np.isfinite(s)
    s, y = s[fin], y[fin]
    if s.size == 0:
        return float("nan"), float("nan")
    u, inv = np.unique(s, return_inverse=True)
    P = np.bincount(inv, y, minlength=u.size)
    N = np.bincount(inv, 1.0 - y, minlength=u.size)
    neg_before = np.concatenate([[0.0], np.cumsum(N)])  # blocks < i predicted 0
    pos_from = P.sum() - np.concatenate([[0.0], np.cumsum(P)])
    acc = (neg_before + pos_from) / s.size  # i = 0..len(u)
    i = int(np.argmax(acc))
    if i == 0:
        thr = float(u[0] - 1.0)
    elif i == u.size:
        thr = float(u[-1] + 1.0)
    else:
        thr = float(0.5 * (u[i - 1] + u[i]))
    return float(acc[i]), thr


def _none(x):
    return None if x is None or not np.isfinite(x) else float(x)


def _auc(df) -> float | None:
    return _none(weighted_auc(df["score"], df["label"])) if len(df) else None


def _acc(df) -> float | None:
    s = df["score"].to_numpy(dtype=float)
    fin = np.isfinite(s)
    if not fin.any():
        return None
    return float(np.mean((s[fin] >= 0) == (df["label"].to_numpy()[fin] >= 0.5)))


def _mean_sd(vals) -> tuple[float | None, float | None]:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None, None
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else None)


def _undecided(reason: str, **extra) -> dict:
    return {"reading": UNDECIDED, "reason": reason, **extra}


def _tier_mask(df, tier) -> pd.Series:
    return pd.to_numeric(df["noise_tier"], errors="coerce") == float(tier)


def _bootstrap(fold_items, stat, n_boot: int, rng) -> dict:
    """Mean-over-folds statistic with a parent-group bootstrap CI.

    ``fold_items`` = list of ``(group_codes per row, n_groups, payload)``. Each
    replicate resamples each fold's parent groups with replacement (all rows of
    a group move together, as one cut family) and recomputes ``stat(payload,
    row_weights)``; the replicate value is the mean over folds.
    """
    point = [stat(p, None) for _, _, p in fold_items]
    pm = float(np.nanmean(point)) if np.isfinite(point).any() else float("nan")
    reps = np.full(int(n_boot), np.nan)
    for b in range(int(n_boot)):
        vals = []
        for codes, G, p in fold_items:
            cnt = np.bincount(rng.integers(0, G, G), minlength=G).astype(np.float64)
            vals.append(stat(p, cnt[codes]))
        vals = np.asarray(vals, dtype=float)
        if np.isfinite(vals).any():
            reps[b] = np.nanmean(vals)
    ok = reps[np.isfinite(reps)]
    ci = ([float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))]
          if ok.size else [None, None])
    return {"mean": _none(pm), "per_fold": [_none(x) for x in point], "ci95": ci,
            "n_boot_finite": int(ok.size)}


# --------------------------------------------------------------------------- #
# A. T1: performance by real noise tier (val split)
# --------------------------------------------------------------------------- #
H1_RULE = ("H1 'TKAN degrades with noise more than MiniROCKET' is SUPPORTED iff the "
           "5-fold mean of [(AUC_clean - AUC_5%)_TKAN - (AUC_clean - AUC_5%)_MR-A], "
           "on parent groups with all three tiers in val, exceeds the floor AND its "
           "95% parent-group bootstrap CI excludes 0 (lower bound > 0). "
           "NOT_SUPPORTED iff the CI upper bound is below the floor (a TKAN-specific "
           "drop larger than the floor is ruled out). Otherwise INCONCLUSIVE.")


def section_a(data: ScoreData, floor: float, n_boot: int, seed: int) -> dict:
    out = {"pre_committed": H1_RULE, "per_model": {}}
    for model in data.models("val"):
        per_fold = {}
        for fold, df in data.by_fold(model, "val").items():
            per_fold[str(fold)] = {str(t): {"auc": _auc(df[_tier_mask(df, t)]),
                                            "acc": _acc(df[_tier_mask(df, t)]),
                                            "n_files": int(_tier_mask(df, t).sum())}
                                   for t in TIERS}
        summary = {}
        for t in TIERS:
            am, asd = _mean_sd([v[str(t)]["auc"] for v in per_fold.values()])
            cm, csd = _mean_sd([v[str(t)]["acc"] for v in per_fold.values()])
            summary[str(t)] = {"auc_mean": am, "auc_sd": asd, "acc_mean": cm,
                               "acc_sd": csd, "n_folds": len(per_fold)}
        out["per_model"][model] = {"per_fold": per_fold, "summary": summary}

    tk, mr = data.by_fold("tkan", "val"), data.by_fold("minirocket_A", "val")
    folds = sorted(set(tk) & set(mr))
    if not folds:
        missing = [m for m, d in (("tkan", tk), ("minirocket_A", mr)) if not d]
        reason = (f"no val scores for {', '.join(missing)}" if missing
                  else "tkan and minirocket_A share no fold")
        out["paired_tkan_minus_mr"] = _undecided(reason)
        out["h1"] = _undecided(reason, diff_of_drops_mean=None, ci95=[None, None],
                               pre_committed=H1_RULE)
        out.update(reading=UNDECIDED, reason=reason)
        return out

    rng = np.random.default_rng(int(seed))
    merged, n_unmatched = {}, {}
    for f in folds:
        a = tk[f][["file", "parent_group", "noise_tier", "label", "score"]]
        b = mr[f][["file", "score"]]
        m = a.merge(b, on="file", suffixes=("_tk", "_mr"), how="inner")
        n_unmatched[str(f)] = int(len(a) + len(b) - 2 * len(m))
        merged[f] = m

    def items_for(rows_by_fold, masks):
        items = []
        for f, m in rows_by_fold.items():
            codes, uniq = pd.factorize(m["parent_group"])
            payload = {k: (_AucPrep(m.loc[mk, "score_tk"], m.loc[mk, "label"]),
                           _AucPrep(m.loc[mk, "score_mr"], m.loc[mk, "label"]),
                           mk.to_numpy())
                       for k, mk in masks(m).items()}
            items.append((codes, len(uniq), payload))
        return items

    def _w(w, mk):
        return None if w is None else w[mk]

    paired = {}
    for t in TIERS:
        sub = {f: m[_tier_mask(m, t)].reset_index(drop=True) for f, m in merged.items()}
        sub = {f: m for f, m in sub.items() if len(m)}
        if not sub:
            paired[str(t)] = _undecided(f"no tier-{t} files in val")
            continue
        items = items_for(sub, lambda m: {"all": pd.Series(True, index=m.index)})

        def stat(p, w):
            ptk, pmr, mk = p["all"]
            return ptk.auc(_w(w, mk)) - pmr.auc(_w(w, mk))

        r = _bootstrap(items, stat, n_boot, rng)
        paired[str(t)] = {"diff_mean": r["mean"], "ci95": r["ci95"],
                          "per_fold": dict(zip(map(str, sub), r["per_fold"])),
                          "n_files_per_fold": {str(f): int(len(m)) for f, m in sub.items()}}
    out["paired_tkan_minus_mr"] = paired
    out["n_unmatched_files_per_fold"] = n_unmatched

    complete, n_groups = {}, {}
    for f, m in merged.items():
        tiers_by_g = m.groupby("parent_group")["noise_tier"].agg(
            lambda s: set(pd.to_numeric(s, errors="coerce").dropna().astype(int)))
        keep = [g for g, ts in tiers_by_g.items() if set(TIERS) <= ts]
        n_groups[str(f)] = len(keep)
        if keep:
            complete[f] = m[m["parent_group"].isin(keep)].reset_index(drop=True)
    if not complete:
        out["h1"] = _undecided("no parent group has all three tiers in val",
                               diff_of_drops_mean=None, ci95=[None, None],
                               pre_committed=H1_RULE, n_complete_groups_per_fold=n_groups)
        out.update(reading=UNDECIDED, reason=out["h1"]["reason"])
        return out
    items = items_for(complete, lambda m: {t: _tier_mask(m, t) for t in (0, 5)})

    def h1_stat(p, w):
        tk0, mr0, m0 = p[0]
        tk5, mr5, m5 = p[5]
        return ((tk0.auc(_w(w, m0)) - tk5.auc(_w(w, m5)))
                - (mr0.auc(_w(w, m0)) - mr5.auc(_w(w, m5))))

    r = _bootstrap(items, h1_stat, n_boot, rng)
    drops = {}
    for name, col in (("tkan", "score_tk"), ("minirocket_A", "score_mr")):
        per = []
        for m in complete.values():
            c = m[_tier_mask(m, 0)].rename(columns={col: "score"})
            n = m[_tier_mask(m, 5)].rename(columns={col: "score"})
            a0, a5 = _auc(c), _auc(n)
            per.append(None if a0 is None or a5 is None else a0 - a5)
        drops[name] = _mean_sd(per)[0]
    mean, ci = r["mean"], r["ci95"]
    if mean is None or ci[0] is None:
        reading = "INCONCLUSIVE"
    elif mean > floor and ci[0] > 0:
        reading = "SUPPORTED"
    elif ci[1] < floor:
        reading = "NOT_SUPPORTED"
    else:
        reading = "INCONCLUSIVE"
    out["h1"] = {"reading": reading, "diff_of_drops_mean": mean, "ci95": ci,
                 "per_fold": dict(zip(map(str, complete), r["per_fold"])),
                 "drop_clean_minus_5_mean": drops, "floor": float(floor),
                 "n_boot": int(n_boot), "n_folds": len(complete),
                 "n_complete_groups_per_fold": n_groups,
                 "n_files_per_fold": {str(f): int(len(m)) for f, m in complete.items()},
                 "pre_committed": H1_RULE}
    out.update(reading=reading, reason=None)
    return out


# --------------------------------------------------------------------------- #
# B. Score shift / calibration
# --------------------------------------------------------------------------- #
SHIFT_RULE = ("Per model: THRESHOLD_SHIFT iff on held-out the 5-fold mean of "
              "(best in-sample accuracy - accuracy at threshold 0) >= 0.05, i.e. at "
              "least 5 accuracy points are lost to where the threshold sits rather "
              "than to ranking. Shifts are also given in 'separation units': the "
              "change in a class's median score relative to the clean tier, over "
              "the clean tier's (median chatter - median stable).")
SHIFT_ACC_GAP = 0.05


def _shift_stats(df) -> dict:
    s, y = df["score"].to_numpy(float), df["label"].to_numpy() >= 0.5
    fin = np.isfinite(s)
    s, y = s[fin], y[fin]
    best, thr = best_threshold_accuracy(s, y)
    return {
        "median_label0": _none(np.median(s[~y])) if (~y).any() else None,
        "median_label1": _none(np.median(s[y])) if y.any() else None,
        "acc_at_0": _none(np.mean((s >= 0) == y)) if s.size else None,
        "tpr_at_0": _none(np.mean(s[y] >= 0)) if y.any() else None,
        "tnr_at_0": _none(np.mean(s[~y] < 0)) if (~y).any() else None,
        "best_acc": _none(best), "best_threshold": _none(thr), "n_files": int(s.size),
    }


def section_b(data: ScoreData) -> dict:
    out = {"pre_committed": SHIFT_RULE, "per_model": {}}
    models = sorted(set(data.models("val")) | set(data.models("heldout")))
    for model in models:
        sets = {}
        for f, df in data.by_fold(model, "val").items():
            for t in TIERS:
                sets.setdefault(f"val_tier{t}", {})[str(f)] = _shift_stats(df[_tier_mask(df, t)])
        for f, df in data.by_fold(model, "heldout").items():
            sets.setdefault("heldout", {})[str(f)] = _shift_stats(df)
        clean = sets.get("val_tier0", {})
        summary = {}
        for name, per in sets.items():
            row = {}
            for k in ("median_label0", "median_label1", "acc_at_0", "tpr_at_0", "tnr_at_0",
                      "best_acc"):
                row[f"{k}_mean"] = _mean_sd([v[k] for v in per.values()])[0]
            gaps, sh0, sh1 = [], [], []
            for f, v in per.items():
                if v["best_acc"] is not None and v["acc_at_0"] is not None:
                    gaps.append(v["best_acc"] - v["acc_at_0"])
                c = clean.get(f)
                if c and None not in (c["median_label0"], c["median_label1"],
                                      v["median_label0"], v["median_label1"]):
                    sep = c["median_label1"] - c["median_label0"]
                    if sep > 0:
                        sh0.append((v["median_label0"] - c["median_label0"]) / sep)
                        sh1.append((v["median_label1"] - c["median_label1"]) / sep)
            row["acc_gap_best_minus_0_mean"] = _mean_sd(gaps)[0]
            row["shift_label0_sep_units"] = _mean_sd(sh0)[0]
            row["shift_label1_sep_units"] = _mean_sd(sh1)[0]
            row["n_folds"] = len(per)
            summary[name] = row
        if "heldout" in summary and summary["heldout"]["acc_gap_best_minus_0_mean"] is not None:
            g = summary["heldout"]["acc_gap_best_minus_0_mean"]
            reading = "THRESHOLD_SHIFT" if g >= SHIFT_ACC_GAP else "NO_MATERIAL_SHIFT"
            res = {"reading": reading, "reason": None}
        else:
            res = _undecided("no held-out scores")
        out["per_model"][model] = {**res, "per_fold": sets, "summary": summary}
    tk = out["per_model"].get("tkan")
    if tk is None:
        out.update(_undecided("no tkan scores"))
    else:
        out.update(reading=tk["reading"], reason=tk["reason"])
    return out


# --------------------------------------------------------------------------- #
# C. Held-out
# --------------------------------------------------------------------------- #
GAP_RULE = ("'val 5%-tier AUC predicts held-out AUC' (PREDICTS) iff the 5-fold mean "
            "|held-out AUC - val tier-5 AUC| <= 0.02 AND it is smaller than the gap to "
            "the all-tier val AUC. Otherwise DOES_NOT_PREDICT.")
GAP_TOL = 0.02


def _subset_summary(per_fold: dict) -> dict:
    am, asd = _mean_sd([v["auc"] for v in per_fold.values()])
    cm, csd = _mean_sd([v["acc"] for v in per_fold.values()])
    nm = _mean_sd([v["n_files"] for v in per_fold.values()])[0]
    return {"auc_mean": am, "auc_sd": asd, "acc_mean": cm, "acc_sd": csd,
            "n_files_mean": nm, "per_fold": per_fold}


def _m(df):
    return {"auc": _auc(df), "acc": _acc(df), "n_files": int(len(df))}


def section_c(data: ScoreData, floor: float) -> dict:
    out = {"pre_committed": GAP_RULE, "per_model": {}, "val5_predicts_heldout": {}}
    models = data.models("heldout")
    if not models:
        out.update(_undecided("no held-out scores"))
        return out
    allh = pd.concat([df for (m, _, s), df in data.frames.items() if s == "heldout"])
    om = allh.drop_duplicates("file")["omega"].to_numpy(float)
    om = om[np.isfinite(om)]
    edges = (np.quantile(om, [1 / 3, 2 / 3]).tolist() if om.size >= 3 else None)
    out["omega_tertile_edges"] = edges
    for model in models:
        fr = data.by_fold(model, "heldout")
        per_fold = {str(f): _m(df) for f, df in fr.items()}
        res = {"per_fold": per_fold, "summary": _subset_summary(per_fold)}
        res["by_dataset"] = {
            "121": _subset_summary({str(f): _m(df[df["dataset"] == 121]) for f, df in fr.items()}),
            "122-125": _subset_summary({str(f): _m(df[df["dataset"].between(122, 125)])
                                        for f, df in fr.items()}),
        }
        bands = {"<0.8": lambda b: b < 0.8, "0.8-1.2": lambda b: (b >= 0.8) & (b <= 1.2),
                 ">1.2": lambda b: b > 1.2}
        res["by_b_over_blim"] = {
            k: _subset_summary({str(f): _m(df[fn(pd.to_numeric(df["b_over_blim"],
                                                               errors="coerce"))])
                                for f, df in fr.items()}) for k, fn in bands.items()}
        if edges is not None:
            e1, e2 = edges
            tert = {"T1": lambda o: o <= e1, "T2": lambda o: (o > e1) & (o <= e2),
                    "T3": lambda o: o > e2}
            res["by_omega_tertile"] = {
                k: _subset_summary({str(f): _m(df[fn(df["omega"].astype(float))])
                                    for f, df in fr.items()}) for k, fn in tert.items()}
        out["per_model"][model] = res

        val = data.by_fold(model, "val")
        folds = sorted(set(val) & set(fr))
        if not folds:
            out["val5_predicts_heldout"][model] = _undecided("no val scores on the same folds")
            continue
        g5, ga, rows = [], [], {}
        for f in folds:
            h, v5, va = _auc(fr[f]), _auc(val[f][_tier_mask(val[f], 5)]), _auc(val[f])
            rows[str(f)] = {"heldout_auc": h, "val_tier5_auc": v5, "val_all_auc": va}
            if None not in (h, v5):
                g5.append(h - v5)
            if None not in (h, va):
                ga.append(h - va)
        m5, s5 = _mean_sd(g5)
        ma, sa = _mean_sd(ga)
        if m5 is None or ma is None:
            out["val5_predicts_heldout"][model] = _undecided("AUC unmeasurable", per_fold=rows)
            continue
        reading = ("PREDICTS" if abs(m5) <= GAP_TOL and abs(m5) < abs(ma)
                   else "DOES_NOT_PREDICT")
        out["val5_predicts_heldout"][model] = {
            "reading": reading, "gap_vs_val_tier5_mean": m5, "gap_vs_val_tier5_sd": s5,
            "gap_vs_val_all_mean": ma, "gap_vs_val_all_sd": sa, "per_fold": rows}
    tk, mr = data.by_fold("tkan", "heldout"), data.by_fold("minirocket_A", "heldout")
    common = sorted(set(tk) & set(mr))
    if common:
        diffs = {str(f): (lambda a, b: None if None in (a, b) else a - b)(
            _auc(tk[f]), _auc(mr[f])) for f in common}
        dm, dsd = _mean_sd(list(diffs.values()))
        out["heldout_tkan_minus_mr"] = {"diff_mean": dm, "diff_sd": dsd, "per_fold": diffs,
                                        "exceeds_floor": None if dm is None else abs(dm) > floor}
    tkr = out["val5_predicts_heldout"].get("tkan")
    if tkr is None:
        out.update(_undecided("no tkan held-out scores"))
    else:
        out.update(reading=tkr["reading"], reason=tkr.get("reason"))
    return out


# --------------------------------------------------------------------------- #
# D. T2 dose-response + injection validity
# --------------------------------------------------------------------------- #
INJ_RULE = ("Injection VALID iff, for BOTH tkan and minirocket_A, the 5-fold mean "
            "|AUC(clean files + injected level 5) - AUC(real 5% mirrors of the same "
            "cuts)| <= 0.01. Otherwise INVALID, and the T2 dose-response is reported "
            "descriptively only.")
INJ_TOL = 0.01


def _mirror_pairs(vn5: pd.DataFrame, val_t5_files: set) -> list[tuple[str, str]]:
    from pri2m.xai.noise_probe import mirror_file

    pairs = []
    for f, d in zip(vn5["file"], vn5["dataset"]):
        d = int(d)
        if not 1 <= d <= 30:
            continue
        try:
            mf = mirror_file(f, d, d + 90)
        except ValueError:
            continue
        if mf in val_t5_files:
            pairs.append((f, mf))
    return pairs


def section_d(data: ScoreData) -> dict:
    out = {"pre_committed": INJ_RULE, "per_model": {}}
    models = data.models("val_noise")
    if not models:
        out.update(_undecided("no val_noise scores"), t2_status="undecided",
                   injection_validity=_undecided("no val_noise scores"))
        return out
    for model in models:
        fr = data.by_fold(model, "val_noise")
        per_fold, slopes = {}, []
        for f, df in fr.items():
            lv = {}
            for lev, g in df.groupby("level", sort=True):
                lv[f"{float(lev):g}"] = {"auc": _auc(g), "acc": _acc(g), "n_files": int(len(g))}
            per_fold[str(f)] = lv
            pts = [(float(k), v["auc"]) for k, v in lv.items() if v["auc"] is not None]
            if len(pts) >= 2:
                x, y = np.array(pts).T
                slopes.append(float(np.polyfit(x, y, 1)[0]))
        levels = sorted({float(k) for v in per_fold.values() for k in v})
        summary = {}
        for lev in levels:
            k = f"{lev:g}"
            am, asd = _mean_sd([v.get(k, {}).get("auc") for v in per_fold.values()])
            cm, csd = _mean_sd([v.get(k, {}).get("acc") for v in per_fold.values()])
            summary[k] = {"auc_mean": am, "auc_sd": asd, "acc_mean": cm, "acc_sd": csd}
        sm, ssd = _mean_sd(slopes)
        out["per_model"][model] = {"per_fold": per_fold, "summary": summary,
                                   "slope_mean": sm, "slope_sd": ssd,
                                   "slope_units": "AUC per unit level"}
    sl = {m: out["per_model"].get(m, {}).get("slope_mean") for m in MODELS_PAIRED}
    out["slope_tkan_minus_mr"] = (None if None in sl.values()
                                  else sl["tkan"] - sl["minirocket_A"])

    inj = {"pre_committed": INJ_RULE, "per_model": {}}
    reasons = []
    for model in MODELS_PAIRED:
        vn, val = data.by_fold(model, "val_noise"), data.by_fold(model, "val")
        folds = sorted(set(vn) & set(val))
        if not folds:
            reasons.append(f"{model}: needs both val_noise and val scores")
            continue
        if not any((vn[f]["level"] == 5.0).any() for f in folds):
            reasons.append(f"{model}: no level 5 in val_noise")
            continue
        diffs, sub, nmatch, rows = [], [], {}, {}
        for f in folds:
            v = val[f]
            t5 = v[_tier_mask(v, 5)].set_index("file")
            vn5 = vn[f][vn[f]["level"] == 5.0]
            pairs = _mirror_pairs(vn5, set(t5.index))
            nmatch[str(f)] = len(pairs)
            if not pairs:
                continue
            clean_files = [p[0] for p in pairs]
            inj_auc = _auc(vn5[vn5["file"].isin(clean_files)])
            real_auc = _auc(t5.loc[[p[1] for p in pairs]].reset_index())
            vn0 = vn[f][(vn[f]["level"] == 0.0) & vn[f]["file"].isin(clean_files)]
            v0 = v[v["file"].isin(clean_files)]
            a0, b0 = _auc(vn0), _auc(v0)
            rows[str(f)] = {"auc_injected5": inj_auc, "auc_real5": real_auc,
                            "auc_injected0_subsampled": a0, "auc_real_clean_allwindows": b0}
            if None not in (inj_auc, real_auc):
                diffs.append(inj_auc - real_auc)
            if None not in (a0, b0):
                sub.append(a0 - b0)
        dm, dsd = _mean_sd(diffs)
        if dm is None:
            reasons.append(f"{model}: no matched clean/5% mirror pairs with measurable AUC")
            continue
        inj["per_model"][model] = {
            "diff_mean": dm, "diff_sd": dsd, "per_fold": rows,
            "n_matched_files_per_fold": nmatch,
            # Diagnostic: the window-subsampling effect alone (level 0 vs all windows).
            "level0_subsample_minus_allwindows_mean": _mean_sd(sub)[0],
            "within_tol": abs(dm) <= INJ_TOL}
    if reasons:
        inj.update(_undecided("; ".join(reasons)))
    else:
        ok = all(inj["per_model"][m]["within_tol"] for m in MODELS_PAIRED)
        inj.update(reading="VALID" if ok else "INVALID", reason=None)
    out["injection_validity"] = inj
    out["t2_status"] = {"VALID": "valid", "INVALID": "descriptive_only"}.get(
        inj["reading"], "descriptive_only")
    out.update(reading=inj["reading"], reason=inj.get("reason"))
    return out


# --------------------------------------------------------------------------- #
# E. Side inputs
# --------------------------------------------------------------------------- #
SIDE_RULE = ("SIDE_INPUTS_MATTER iff |5-fold mean of (AUC_tkan - AUC_tkan_side_off)| on "
             "held-out exceeds the floor AND every fold's difference has the same sign. "
             "Otherwise NO_EFFECT_ABOVE_FLOOR.")


def section_e(data: ScoreData, floor: float) -> dict:
    out = {"pre_committed": SIDE_RULE}
    tk, so = data.by_fold("tkan", "heldout"), data.by_fold("tkan_side_off", "heldout")
    folds = sorted(set(tk) & set(so))
    if not folds:
        missing = [m for m, d in (("tkan", tk), ("tkan_side_off", so)) if not d]
        out.update(_undecided(f"no held-out scores for {', '.join(missing) or 'common folds'}"),
                   diff_mean=None)
        return out
    per = {}
    for f in folds:
        a, b = _auc(tk[f]), _auc(so[f])
        per[str(f)] = {"auc_tkan": a, "auc_side_off": b,
                       "diff": None if None in (a, b) else a - b}
    d = [v["diff"] for v in per.values() if v["diff"] is not None]
    dm, dsd = _mean_sd(d)
    if dm is None:
        out.update(_undecided("AUC unmeasurable"), diff_mean=None, per_fold=per)
        return out
    same_sign = all(x > 0 for x in d) or all(x < 0 for x in d)
    reading = ("SIDE_INPUTS_MATTER" if abs(dm) > floor and same_sign
               else "NO_EFFECT_ABOVE_FLOOR")
    out.update(reading=reading, reason=None, diff_mean=dm, diff_sd=dsd,
               n_folds=len(d), n_folds_exceeding_floor=int(sum(abs(x) > floor for x in d)),
               floor=float(floor), per_fold=per)
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _load_calibration(path) -> dict:
    if path is None:
        return {"path": None, "status": "not given"}
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "status": "missing"}
    s = (json.loads(p.read_text()) or {}).get("summary") or {}
    return {"path": str(p), "status": "ok" if s.get("rel_sigma_per_unit") else "undecided",
            "rel_sigma_per_unit": s.get("rel_sigma_per_unit"),
            "linear_ok": s.get("linear_ok"), "undecided_reason": s.get("undecided_reason")}


def build_report(scores_dir, floor: float = 0.0092, n_boot: int = 2000, seed: int = 0,
                 calibration=None) -> dict:
    data = load_scores(scores_dir)
    inventory = {}
    for (m, f, s) in sorted(data.frames):
        inventory.setdefault(m, {}).setdefault(s, []).append(f)
    return _json_safe({
        "analysis": "TKAN vs MiniROCKET: is the held-out gap a noise-robustness gap?",
        "scores_dir": str(scores_dir), "floor": float(floor), "n_boot": int(n_boot),
        "seed": int(seed), "inventory": inventory, "problems": data.problems,
        "calibration": _load_calibration(calibration),
        "conventions": ("AUC from file-mean scores; accuracy at threshold 0 (TKAN logit, "
                        "MiniROCKET ridge decision). Bootstrap CIs resample parent groups "
                        "within fold; 'mean' = mean over folds."),
        "A_noise_tier": section_a(data, floor, n_boot, seed),
        "B_score_shift": section_b(data),
        "C_heldout": section_c(data, floor),
        "D_dose_response": section_d(data),
        "E_side_inputs": section_e(data, floor),
    })


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_markdown(rep: dict) -> str:
    L = ["# TKAN vs MiniROCKET under noise", "",
         f"Scores: `{rep['scores_dir']}`; floor {rep['floor']}; {rep['n_boot']} bootstrap "
         "replicates (parent groups resampled within fold).", ""]
    inv = rep["inventory"]
    L.append("Inputs: " + ("; ".join(f"{m}: " + ", ".join(f"{s} folds {v}" for s, v in d.items())
                                     for m, d in inv.items()) or "none") + ".")
    if rep["problems"]:
        L.append(f"Skipped files: {len(rep['problems'])} (see JSON `problems`).")
    cal = rep["calibration"]
    L += ["", f"Noise calibration: {cal.get('status')}"
          + (f", rel_sigma_per_unit {cal.get('rel_sigma_per_unit')}, linear_ok "
             f"{cal.get('linear_ok')}" if cal.get("rel_sigma_per_unit") else ""), ""]

    a = rep["A_noise_tier"]
    L.append("## A. Accuracy and AUC by real noise tier (in-fold val)")
    for m, d in a["per_model"].items():
        s = d["summary"]
        L.append(f"- {m}: AUC clean {_f(s['0']['auc_mean'])}, 1% {_f(s['1']['auc_mean'])}, "
                 f"5% {_f(s['5']['auc_mean'])}; accuracy clean {_f(s['0']['acc_mean'])}, "
                 f"5% {_f(s['5']['acc_mean'])}.")
    h = a["h1"]
    L.append(f"- **H1 (TKAN degrades more with noise): {h['reading']}**"
             + (f". Difference of drops {_f(h['diff_of_drops_mean'], 4)}, 95% CI "
                f"[{_f(h['ci95'][0], 4)}, {_f(h['ci95'][1], 4)}]"
                if h.get("diff_of_drops_mean") is not None else f" ({h.get('reason')})") + ".")
    L.append(f"  Rule: {h['pre_committed']}")

    b = rep["B_score_shift"]
    L += ["", "## B. Score shift under noise"]
    if not b["per_model"]:
        L.append(f"- UNDECIDED: {b['reason']}")
    for m, d in b["per_model"].items():
        hs = d["summary"].get("heldout", {})
        L.append(f"- {m}: {d['reading']}; held-out accuracy at 0 {_f(hs.get('acc_at_0_mean'))} "
                 f"vs best threshold {_f(hs.get('best_acc_mean'))}; stable-class median moved "
                 f"{_f(hs.get('shift_label0_sep_units'), 2)} and chatter-class "
                 f"{_f(hs.get('shift_label1_sep_units'), 2)} separation units vs clean val.")

    c = rep["C_heldout"]
    L += ["", "## C. Held-out (121-125)"]
    if not c.get("per_model"):
        L.append(f"- UNDECIDED: {c['reason']}")
    for m, d in c.get("per_model", {}).items():
        g = c["val5_predicts_heldout"].get(m, {})
        L.append(f"- {m}: AUC {_f(d['summary']['auc_mean'])} ± {_f(d['summary']['auc_sd'])}, "
                 f"accuracy {_f(d['summary']['acc_mean'])}; held-out minus val-5% AUC "
                 f"{_f(g.get('gap_vs_val_tier5_mean'))}, minus val-all "
                 f"{_f(g.get('gap_vs_val_all_mean'))} -> {g.get('reading')}.")

    dd = rep["D_dose_response"]
    L += ["", "## D. Injected-noise dose-response (clean val files)"]
    for m, d in dd.get("per_model", {}).items():
        lv = ", ".join(f"{k}: {_f(v['auc_mean'])}" for k, v in d["summary"].items())
        L.append(f"- {m}: AUC by level {{{lv}}}; slope {_f(d['slope_mean'], 4)} per level.")
    inj = dd.get("injection_validity", {})
    L.append(f"- Injection validity: **{inj.get('reading')}**"
             + (f" ({inj.get('reason')})" if inj.get("reason") else "")
             + f"; T2 status: {dd.get('t2_status')}.")
    for m, d in inj.get("per_model", {}).items():
        L.append(f"  - {m}: injected-5 minus real-5% AUC {_f(d['diff_mean'], 4)}; "
                 f"window-subsampling effect {_f(d['level0_subsample_minus_allwindows_mean'], 4)}.")

    e = rep["E_side_inputs"]
    L += ["", "## E. Side inputs (TKAN with vs without)",
          f"- {e['reading']}" + (f": held-out AUC difference {_f(e['diff_mean'], 4)} "
                                 f"({e['n_folds_exceeding_floor']}/{e['n_folds']} folds above floor)"
                                 if e.get("diff_mean") is not None else f" ({e.get('reason')})")]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
#: Fixed categorical slots (validated reference palette), by model, never by rank.
MODEL_COLORS = {"tkan": "#eb6834", "minirocket_A": "#2a78d6", "minirocket_B": "#1baf7a",
                "tkan_side_off": "#eda100"}
MODEL_LABELS = {"tkan": "TKAN", "minirocket_A": "MiniROCKET (4 ch)",
                "minirocket_B": "MiniROCKET (4 + delayed)", "tkan_side_off": "TKAN, no side inputs"}
LABEL_COLORS = {0: "#8a8985", 1: "#4a3aa7"}
_INK, _MUTED = "#0b0b0b", "#52514e"


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(_MUTED)
    ax.tick_params(colors=_MUTED)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_auc_by_tier(rep: dict, path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pm = rep["A_noise_tier"]["per_model"]
    models = [m for m in MODEL_COLORS if m in pm]
    if not models:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = {"0": 0, "1": 1, "5": 2}
    off = np.linspace(-0.12, 0.12, len(models)) if len(models) > 1 else [0.0]
    for o, m in zip(off, models):
        col = MODEL_COLORS[m]
        means = []
        for t, x in xs.items():
            vals = [v[t]["auc"] for v in pm[m]["per_fold"].values() if v[t]["auc"] is not None]
            ax.scatter([x + o] * len(vals), vals, s=22, color=col, alpha=0.45, linewidths=0)
            means.append(pm[m]["summary"][t]["auc_mean"])
        ok = [(x + o, y) for x, y in zip(xs.values(), means) if y is not None]
        if ok:
            ax.plot(*zip(*ok), color=col, linewidth=2, marker="o", markersize=8,
                    label=MODEL_LABELS.get(m, m))
    held = rep["C_heldout"].get("per_model", {})
    for o, m in zip(off, models):
        if m in held:
            vals = [v["auc"] for v in held[m]["per_fold"].values() if v["auc"] is not None]
            ax.scatter([3 + o] * len(vals), vals, s=22, color=MODEL_COLORS[m], alpha=0.45,
                       linewidths=0)
            if held[m]["summary"]["auc_mean"] is not None:
                ax.scatter([3 + o], [held[m]["summary"]["auc_mean"]], s=64, marker="D",
                           color=MODEL_COLORS[m], edgecolors="white", linewidths=1.5)
    ax.axvline(2.5, color="#d0cfca", linewidth=1)
    ax.set_xticks([0, 1, 2, 3], ["Clean", "1% noise", "5% noise", "Held-out\n(5% noise)"])
    ax.set_ylabel("AUC (file level)", color=_INK)
    ax.set_xlabel("Noise level of the validation files", color=_INK)
    ax.set_title("Ranking quality (AUC) by noise tier: fold points and 5-fold mean",
                 color=_INK, fontsize=11)
    _style(ax)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def plot_dose_response(rep: dict, path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pm = rep["D_dose_response"].get("per_model", {})
    models = [m for m in MODEL_COLORS if m in pm]
    if not models:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    real = rep["A_noise_tier"]["per_model"]
    for m in models:
        col = MODEL_COLORS[m]
        lv = sorted(pm[m]["summary"], key=float)
        x = np.array([float(k) for k in lv])
        y = np.array([np.nan if pm[m]["summary"][k]["auc_mean"] is None
                      else pm[m]["summary"][k]["auc_mean"] for k in lv])
        sd = np.array([0.0 if pm[m]["summary"][k]["auc_sd"] is None
                       else pm[m]["summary"][k]["auc_sd"] for k in lv])
        ax.fill_between(x, y - sd, y + sd, color=col, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=col, linewidth=2, marker="o", markersize=7,
                label=f"{MODEL_LABELS.get(m, m)}, injected")
        if m in real:
            pts = [(float(t), real[m]["summary"][t]["auc_mean"]) for t in ("0", "1", "5")
                   if real[m]["summary"][t]["auc_mean"] is not None]
            if pts:
                ax.scatter(*zip(*pts), s=80, marker="D", facecolors="white", edgecolors=col,
                           linewidths=2, zorder=3, label=f"{MODEL_LABELS.get(m, m)}, real tier")
    ax.set_xlabel("Noise injected into clean files (level 5 = the real 5% tier)", color=_INK)
    ax.set_ylabel("AUC (file level)", color=_INK)
    ax.set_title("AUC as noise is added: mean ± sd over folds", color=_INK, fontsize=11)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def plot_score_distributions(data: ScoreData, path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in MODELS_PAIRED if data.by_fold(m, "val") or data.by_fold(m, "heldout")]
    if not models:
        return False
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), squeeze=False)
    groups = [("Clean", "val", 0), ("1%", "val", 1), ("5%", "val", 5), ("Held-out", "heldout", None)]
    for ax, m in zip(axes[0], models):
        pos, ticks = 0, []
        for name, split, tier in groups:
            fr = data.by_fold(m, split)
            if not fr:
                continue
            df = pd.concat(fr.values())
            if tier is not None:
                df = df[_tier_mask(df, tier)]
            for lab in (0, 1):
                s = df.loc[df["label"] == lab, "score"].to_numpy(float)
                s = s[np.isfinite(s)]
                if s.size:
                    bp = ax.boxplot([s], positions=[pos + (lab - 0.5) * 0.36], widths=0.3,
                                    patch_artist=True, showfliers=False)
                    for patch in bp["boxes"]:
                        patch.set(facecolor=LABEL_COLORS[lab], alpha=0.75, edgecolor="white")
                    for k in ("whiskers", "caps"):
                        for ln in bp[k]:
                            ln.set(color=LABEL_COLORS[lab])
                    for ln in bp["medians"]:
                        ln.set(color="white", linewidth=2)
            ticks.append((pos, name))
            pos += 1
        ax.axhline(0, color=_MUTED, linewidth=1, linestyle="--")
        ax.set_xticks([p for p, _ in ticks], [n for _, n in ticks])
        ax.set_title(MODEL_LABELS.get(m, m), color=_INK, fontsize=11)
        ax.set_xlabel("Noise tier", color=_INK)
        ax.set_ylabel("File score (decision threshold at 0)", color=_INK)
        _style(ax)
    from matplotlib.patches import Patch

    axes[0][0].legend(handles=[Patch(color=LABEL_COLORS[0], label="Stable (label 0)"),
                               Patch(color=LABEL_COLORS[1], label="Chatter (label 1)")],
                      frameon=False, fontsize=8, loc="best")
    fig.suptitle("Score distributions by noise tier and true label", color=_INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


FIGURES = ("noise_compare_auc_by_tier.png", "noise_compare_dose_response.png",
           "noise_compare_score_distributions.png")


def run(scores_dir, calibration=None, floor: float = 0.0092, n_boot: int = 2000,
        seed: int = 0, output_dir=None, figures_dir=None) -> dict:
    """Build the report, write JSON + markdown, and draw the three figures."""
    rep = build_report(scores_dir, floor=floor, n_boot=n_boot, seed=seed,
                       calibration=calibration)
    figs = []
    if figures_dir is not None:
        fd = Path(figures_dir)
        fd.mkdir(parents=True, exist_ok=True)
        data = load_scores(scores_dir)
        for name, fn in zip(FIGURES, (lambda p: plot_auc_by_tier(rep, p),
                                      lambda p: plot_dose_response(rep, p),
                                      lambda p: plot_score_distributions(data, p))):
            if fn(fd / name):
                figs.append(name)
    rep["figures"] = figs
    rep["figures_dir"] = None if figures_dir is None else str(figures_dir)
    if output_dir is not None:
        od = Path(output_dir)
        od.mkdir(parents=True, exist_ok=True)
        (od / "noise_compare_report.json").write_text(
            json.dumps(_json_safe(rep), indent=2, allow_nan=False))
        (od / "noise_compare_summary.md").write_text(render_markdown(rep))
    return rep
