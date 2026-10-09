#!/usr/bin/env python3
"""Equations across seeds, local half: read every Great Lakes dump, write one manifest,
six figures and a one-page team summary (spec 2026-10-07-equations-across-seeds, 5-10).

Answers: (1) do retrained TKAN networks learn the same delay-edge curves and formulas,
(2) does a sparse formula distilled from each model's scores stay the same across seeds and
models, (3) how concentrated is MiniROCKET's ridge weight. Exit 2 on any inconsistent dump.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from itertools import permutations
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from pri2m.xai import symbolic  # noqa: E402
from pri2m.xai.equation_fit import (  # noqa: E402
    bootstrap_selections, fidelity, fit_formula, jaccard, ridge_contrast, same_equation)
from pri2m.xai.equations import (  # noqa: E402
    CHANNELS, best_match, family_agreement, recur_verdict, unexplained)

GROUP_A = {"f0s1", "f0s2", "f2s0"}                  # polarity groups found 2026-10-07 (spec 5.2)
GROUP_B = {"f0s3", "f2s1", "f2s2", "f2s3"}
AUC_FLOOR = 0.0092                                  # repeatability floor (spec 8.5)
EVAL_KEYS = ("trainer_load_trainer_eval_auc", "xai_load_trainer_eval_auc", "score_files_auc")
U_THR = 0.05
EXP_FOLDS, EXP_SEEDS = (0, 2), range(4)
# dataviz reference palette, slots 1-3 (validated all-pairs) plus ink and surface tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
MODEL_COLOR = {"tkan": BLUE, "minirocket": ORANGE, "label": AQUA}
MODEL_LABEL = {"tkan": "TKAN", "minirocket": "MiniROCKET", "label": "labels (reference)"}


# --------------------------------------------------------------------------- loading

def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _clean(o):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in (sorted(o) if isinstance(o, set) else o)]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    return o


def _dump(npz: Path) -> dict:
    """``{"path", "z", "meta"}`` for an npz plus its same-stem JSON sidecar."""
    side = npz.with_suffix(".json")
    return {"path": npz, "z": np.load(npz, allow_pickle=False), "meta": json.loads(side.read_text())}


def load_dir(eq: Path):
    """``(windows, tkan, minirocket, evalchecks, problems)``; an npz without sidecar is a problem."""
    problems = []

    def grab(pattern):
        for p in sorted(eq.glob(pattern)):
            if p.with_suffix(".json").is_file():
                d = _dump(p)
                if d["meta"].get("npz_sha256", _sha(p)) != _sha(p):
                    problems.append(f"{p}: sidecar npz_sha256 does not match the file")
                yield d
            else:
                problems.append(f"{p}: sidecar {p.with_suffix('.json').name} missing")

    wins = {int(re.search(r"fold(\d+)", d["path"].name).group(1)): d for d in grab("windows_fold*.npz")}
    tkan = {d["path"].stem[5:]: d for d in grab("tkan_*.npz")}
    mr = {(d["meta"]["fold"], d["meta"]["seed"]): d for d in grab("minirocket_fold*_s*.npz")}
    evs = []
    for p in sorted(eq.glob("evalcheck_*.json")):
        j = json.loads(p.read_text())
        if "checkpoints" in j:                          # skips evalcheck_scores_<i>.json
            evs.append(j)
    return wins, tkan, mr, evs, problems


def expected_missing(wins, tkan, mr, evs) -> list[str]:
    """Spec 8 dump set: 8 TKAN networks, 8 MiniROCKET runs, 2 window files, evalcheck(s)."""
    out = [f"windows_fold{f}" for f in EXP_FOLDS if f not in wins]
    out += [f"tkan_f{f}s{s}" for f in EXP_FOLDS for s in EXP_SEEDS if f"f{f}s{s}" not in tkan]
    out += [f"minirocket_fold{f}_s{s}" for f in EXP_FOLDS for s in EXP_SEEDS if (f, s) not in mr]
    return out + ([] if evs else ["evalcheck_*.json"])


def _seed(name: str) -> int:
    return int(re.match(r"f\d+s(\d+)$", name).group(1))


def file_auc(score, files, label) -> float:
    """AUC of per-file mean score against the file's label (labels are constant per file)."""
    u, inv = np.unique(files, return_inverse=True)
    mean = np.bincount(inv, weights=score) / np.bincount(inv)
    lw = np.asarray(label) > 0.5
    if (np.bincount(inv, weights=lw) % np.bincount(inv)).any():
        raise ValueError("file-level AUC needs one label per file; a file has mixed window labels")
    lab = np.zeros(len(u), bool)
    lab[inv] = lw
    return float(roc_auc_score(lab, mean)) if 0 < lab.sum() < len(lab) else float("nan")


def _window_auc(score, label) -> float:
    lab = np.asarray(label) > 0.5
    return float(roc_auc_score(lab, score)) if 0 < lab.sum() < len(lab) else float("nan")


# --------------------------------------------------------------------------- checks (spec 8)

def _evalcheck_for(ckpt: str, evs: list[dict]):
    key = Path(ckpt).parts[-3:]
    for ev in evs:
        for e in ev["checkpoints"]:
            if Path(e["checkpoint"]).parts[-3:] == key:
                return ev, e
    return None, None


def _cleared(ev: dict | None, e: dict | None) -> bool:
    """Needs a complete evalcheck file, a matching best_auc.pt record and three non-null AUCs."""
    if not ev or ev.get("complete") is not True or not e or not e.get("recorded_auc_matches_checkpoint"):
        return False
    vals = [e.get("recorded_val_auc"), *(e.get(k) for k in EVAL_KEYS)]
    return None not in vals and all(abs(v - vals[0]) <= AUC_FLOOR for v in vals[1:])


def _run_auc(ckpt: str, runs_root: Path) -> float | None:
    """``best_auc_val.auc`` from the run's summary.json (``<run>/checkpoints/best_auc.pt``), if landed."""
    c = Path(ckpt)
    if c.name != "best_auc.pt" or len(c.parts) < 3:
        return None
    for s in sorted(runs_root.glob(f"runs*/{c.parts[-3]}/summary.json")):
        return json.loads(s.read_text()).get("best_auc_val", {}).get("auc")
    return None


def _auc_of(manifest: Path):
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text()).get("metrics", {}).get("heldout", {}).get("B", {}).get("auc")


def run_checks(wins, tkan, mr, evs, mr_manifests: Path, recurrence: Path, runs_root: Path):
    """``(checks, problems)``; any problem makes ``main`` exit 2."""
    problems, checks = [], {"tkan": {}, "minirocket": {}, "minirocket_seed0": {}}
    rec = json.loads(recurrence.read_text()).get("networks", {}) if recurrence.is_file() else None
    win_sha = {}
    for fold, w in wins.items():
        win_sha[fold] = _sha(w["path"])
        if w["meta"].get("complete") is not True:
            problems.append(f"{w['path']}: sidecar not complete")
        if w["meta"].get("sha256") != win_sha[fold]:
            problems.append(f"{w['path']}: sidecar sha256 does not match the file")

    def same_ids(d, fold):
        w = wins.get(fold)
        if w is None:
            problems.append(f"{d['path']}: no window file for fold {fold}")
            return
        if d["meta"].get("complete") is not True:
            problems.append(f"{d['path']}: sidecar not complete")
        if d["meta"].get("windows_sha256", d["meta"].get("window_sha256")) != win_sha[fold]:
            problems.append(f"{d['path']}: window sha256 differs from window file {w['path']}")
        for k in ("val_file", "ho_file"):
            if not np.array_equal(d["z"][k], w["z"][k]):
                problems.append(f"{d['path']}: {k} differs from window file {w['path']}")

    for name, d in tkan.items():
        fold = d["meta"]["fold"]
        same_ids(d, fold)
        ckpt = d["meta"]["checkpoint"]
        if rec is None or name not in rec:
            ck_status = "not_checked"
        elif rec[name].get("checkpoint") == ckpt:
            ck_status = "ok"
        else:
            ck_status = "mismatch"
            problems.append(f"{d['path']}: checkpoint {ckpt} != recurrence manifest "
                            f"{recurrence} {rec[name].get('checkpoint')}")
        if problems or fold not in wins:
            continue
        w, z = wins[fold]["z"], d["z"]
        ev, e = _evalcheck_for(ckpt, evs)
        run_auc = _run_auc(ckpt, runs_root)
        checks["tkan"][name] = {
            "fold": fold, "seed": _seed(name), "n_edges": int(d["meta"]["n_edges"]),
            "checkpoint": ckpt, "checkpoint_sha256": d["meta"].get("checkpoint_sha256"),
            "recurrence_checkpoint": ck_status,
            "val_auc_window": _window_auc(z["val_score"], w["val_label"]),
            "val_auc_file": file_auc(z["val_score"], w["val_file"], w["val_label"]),
            "ho_auc_file": file_auc(z["ho_score"], w["ho_file"], w["ho_label"]),
            "recorded_val_auc": run_auc if run_auc is not None else (e or {}).get("recorded_val_auc"),
            "recorded_val_auc_source": ("summary.json" if run_auc is not None else
                                        "evalcheck" if e and e.get("recorded_val_auc") is not None else None),
            "evalcheck_tag": ev["tag"] if ev else None, "evalcheck_cleared": _cleared(ev, e),
            "flagged": _seed(name) == 0 and not _cleared(ev, e)}
    for (fold, seed), d in mr.items():
        same_ids(d, fold)
        if problems:
            continue
        w, z = wins[fold]["z"], d["z"]
        entry = {"shared_window_ho_auc_file": file_auc(z["ho_score"], w["ho_file"], w["ho_label"]),
                 "norm_max_abs_diff": d["meta"].get("norm_max_abs_diff")}
        checks["minirocket"][f"f{fold}s{seed}"] = entry
        if seed == 0:       # the GL re-run's own full held-out AUC must equal the landed manifest's
            new_p = mr_manifests / f"minirocket_eq_fold{fold}_s0.json"
            old_p = mr_manifests / f"minirocket_fold{fold}.json"
            new, old = _auc_of(new_p), _auc_of(old_p)
            if new is None or old is None:
                status = "not_checked"
            elif abs(new - old) <= 1e-9:
                status = "ok"
            else:
                status = "mismatch"
                problems.append(f"{new_p}: held-out AUC {new!r} != {old_p} {old!r} (pipeline is "
                                f"deterministic; seed 0 must reproduce)")
            checks["minirocket_seed0"][str(fold)] = {
                "status": status, "rerun_auc": new, "landed_auc": old, **entry}
    checks["evalchecks"] = [{"tag": ev["tag"], "complete": ev.get("complete")} for ev in evs]
    checks["sidecars_complete"] = all(d["meta"].get("complete") is True
                                      for d in [*wins.values(), *tkan.values(), *mr.values()])
    return checks, problems


# --------------------------------------------------------------------------- curves (spec 5)

def _polarity(a: str, b: str) -> str | None:
    ga = "A" if a in GROUP_A else "B" if a in GROUP_B else None
    gb = "A" if b in GROUP_A else "B" if b in GROUP_B else None
    if ga is None or gb is None:
        return None
    return f"within_{ga}" if ga == gb else "between_AB"


def load_nets(tkan: dict) -> dict:
    nets = {}
    for name, d in tkan.items():
        z, edges = d["z"], json.loads(str(d["z"]["edges"]))
        ys = z["ys"]
        fits = [symbolic.fit_curve(z["xs"][i], ys[i], r2_threshold=0.9) for i in range(len(edges))]
        nets[name] = {"fold": d["meta"]["fold"], "feats": [e["feature"] for e in edges],
                      "xs": z["xs"], "ys": ys, "null": z["untrained_ys"], "fits": fits,
                      "fams": [f["name"] for f in fits]}
    return nets


def _summ(pairs: list[dict]) -> dict:
    """Summary over the pairs; pairs with no same-channel match (``n_used == 0``) are counted in
    ``n_untestable`` and kept out of every median and out of the recur denominator."""
    live = [p for p in pairs if p["n_used"] > 0]
    out = {"n_pairs": len(live), "n_untestable": len(pairs) - len(live)}
    if not live:
        return out
    u = np.concatenate([p["u_min"] for p in live])
    u = u[np.isfinite(u)]
    return {**out,
            "median_frac_trained": float(np.nanmedian([p["frac_trained"] for p in live])),
            "median_null_p95": float(np.nanmedian([p["null_p95"] for p in live])),
            "n_recur": int(sum(p["recurs"] for p in live)),
            "median_u": float(np.median(u)) if u.size else None,
            "frac_u_below_thr": float((u < U_THR).mean()) if u.size else None}


def _family_pool(pairs: list[dict], nets: dict) -> tuple[dict, list]:
    """Family agreement over matched edges pooled across ``pairs``; also the (famA, famB) list."""
    fa, fb = [], []
    for p in pairs:
        A, B = nets[p["a"]], nets[p["b"]]
        for i, k in enumerate(p["j"]):
            if k >= 0:
                fa.append(A["fams"][i])
                fb.append(B["fams"][k])
    return family_agreement(fa, fb, np.arange(len(fa))), list(zip(fa, fb))


def _formula_u(pairs: list[dict], nets: dict) -> list[float]:
    """u between the two fitted formulas of matched edges whose families differ, on the shared range."""
    out = []
    for p in pairs:
        A, B = nets[p["a"]], nets[p["b"]]
        for i, k in enumerate(p["j"]):
            fa, fb = (A["fams"][i], B["fams"][k]) if k >= 0 else (None, None)
            if fa is None or fb is None or fa == fb:
                continue
            lo, hi = max(A["xs"][i][0], B["xs"][k][0]), min(A["xs"][i][-1], B["xs"][k][-1])
            if hi <= lo:
                continue
            g = np.linspace(lo, hi, 256)
            y = [symbolic.eval_fit(A["fits"][i], g), symbolic.eval_fit(B["fits"][k], g)]
            out.append(float(unexplained(y[0][None], y[1][None])[0, 0]))
    return [u for u in out if np.isfinite(u)]


def curves_block(nets: dict):
    """``(manifest curves block, formulas_native block, internals for the figures)``."""
    live = [n for n, v in nets.items() if len(v["feats"])]
    pairs = []
    for a, b in permutations(live, 2):
        A, B = nets[a], nets[b]
        u_min, j = best_match(unexplained(A["ys"], B["ys"]), A["feats"], B["feats"])
        nulls = [best_match(unexplained(A["ys"], B["null"][k]), A["feats"], B["feats"])[0]
                 for k in range(len(B["null"]))]
        pairs.append({"a": a, "b": b, "same_fold": A["fold"] == B["fold"], "group": _polarity(a, b),
                      **recur_verdict(u_min, nulls), "u_min": u_min, "j": j,
                      "u_null": np.concatenate(nulls) if nulls else np.zeros(0)})
    same = [p for p in pairs if p["same_fold"]]
    across = [p for p in pairs if not p["same_fold"]]
    groups = {g: _summ([p for p in pairs if p["group"] == g])
              for g in ("within_A", "within_B", "between_AB")}
    block = {"frac_trained": _summ(same).get("median_frac_trained"),
             "no_cluster": sorted(n for n in nets if n not in live),
             "within_fold": _summ(same), "across_folds": _summ(across), "by_polarity": groups,
             "pairs": [{k: p[k] for k in ("a", "b", "same_fold", "group", "frac_trained",
                                          "null_p95", "recurs", "n_used")} for p in pairs]}
    once = [p for p in pairs if p["a"] < p["b"]]       # unordered pairs: no double counting
    fam_all, grid = _family_pool(once, nets)
    fu = _formula_u(once, nets)
    native = {"threshold_r2": 0.9,
              "families": {n: v["fams"] for n, v in nets.items()},
              "unit": "matched edges over unordered network pairs",
              "agreement": {"all": fam_all,
                            "within_fold": _family_pool([p for p in once if p["same_fold"]], nets)[0],
                            "across_folds": _family_pool([p for p in once if not p["same_fold"]], nets)[0]},
              "different_family_formula_u": {
                  "n": len(fu), "median": float(np.median(fu)) if fu else None,
                  "frac_below_thr": float(np.mean(np.array(fu) < U_THR)) if fu else None}}
    return block, native, {"pairs": pairs, "grid": grid}


# --------------------------------------------------------------------------- distilled (spec 6)

def distilled_block(wins, tkan, mr, n_boot: int):
    fits, scores = {}, {}                              # key (model, fold, seed) -> ...
    for fold, w in wins.items():
        z = w["z"]
        T, Th, g, names = z["val_terms"], z["ho_terms"], z["val_parent_group"], list(z["term_names"])
        targets = {("label", fold, 0): (z["val_label"].astype(float), z["ho_label"].astype(float))}
        for name, d in tkan.items():
            if d["meta"]["fold"] == fold:
                targets[("tkan", fold, _seed(name))] = (d["z"]["val_score"], d["z"]["ho_score"])
        for (f, s), d in mr.items():
            if f == fold:
                targets[("minirocket", fold, s)] = (d["z"]["val_score"], d["z"]["ho_score"])
        for key, (yv, yh) in targets.items():
            fit = fit_formula(T, yv, g, names)
            fits[key] = {**fit, **fidelity(fit, Th, yh)}
            scores[key] = yv
    nets_out = {f"{m}:f{f}s{s}": {k: v for k, v in r.items() if k != "model"}
                for (m, f, s), r in fits.items()}

    stability = {}
    for model in ("tkan", "minirocket"):
        stability[model] = {}
        for fold, w in wins.items():
            seeds = sorted(s for (m, f, s) in fits if m == model and f == fold)
            if len(seeds) < 2:
                stability[model][str(fold)] = {"seeds": seeds, "same": None}
                continue
            nonzero = [s for s in seeds if s > 0]
            null_seed = 1 if 1 in seeds else (nonzero[0] if nonzero else seeds[0])
            z = w["z"]
            boot = bootstrap_selections(z["val_terms"], scores[(model, fold, null_seed)],
                                        z["val_parent_group"], list(z["term_names"]), n_boot=n_boot)
            stability[model][str(fold)] = {
                "seeds": seeds, "null_seed": null_seed, "n_boot": n_boot,
                **same_equation([fits[(model, fold, s)] for s in seeds], boot)}

    cross = []
    for (m, f, s), a in fits.items():
        for (m2, f2, s2), b in fits.items():
            if m == "tkan" and m2 == "minirocket" and f == f2:
                shared = set(a["selected"]) & set(b["selected"])
                cross.append({"fold": f, "tkan_seed": s, "minirocket_seed": s2,
                              "jaccard": jaccard(set(a["selected"]), set(b["selected"])),
                              "n_shared": len(shared),
                              "n_same_sign": int(sum(np.sign(a["coef"][t]) == np.sign(b["coef"][t])
                                                     for t in shared))})

    dvs = {}
    for model in ("tkan", "minirocket"):
        per = {c: {"S": 0, "D": 0, "none": 0} for c in CHANNELS}
        rows = []
        for (m, f, s), r in sorted(fits.items()):
            if m != model:
                continue
            for c in CHANNELS:
                d, sm = r["coef"].get(f"D_{c}", 0.0), r["coef"].get(f"S_{c}", 0.0)
                per[c]["none" if d == sm == 0 else "S" if abs(sm) > abs(d) else "D"] += 1
                rows.append({"net": f"f{f}s{s}", "channel": c, "D": d, "S": sm})
        dvs[model] = {"by_channel": per, "coefficients": rows}
    return {"networks": nets_out, "stability": stability, "cross_model": cross,
            "diff_vs_sum": dvs}


# --------------------------------------------------------------------------- contrast (spec 7)

def contrast_block(wins, mr):
    out = {}
    for (fold, seed), d in sorted(mr.items()):
        z, w = d["z"], wins[fold]["z"]
        rc = ridge_contrast(z["weights"], k=int(d["meta"].get("top_k", 20)))
        full = file_auc(z["ho_score"], w["ho_file"], w["ho_label"])
        top = file_auc(z["ho_score_top20"], w["ho_file"], w["ho_label"])
        out[f"f{fold}s{seed}"] = {"n_features": int(len(z["weights"])), "k90": rc["k90"],
                                  "auc_full": full, "auc_top20": top,
                                  "retention": top / full if full else None}
    k90 = [v["k90"] for v in out.values()]
    return {"per_seed": out, "median_k90": float(np.median(k90)) if k90 else None,
            "median_retention": float(np.median([v["retention"] for v in out.values()
                                                 if v["retention"] is not None])) if out else None}


# --------------------------------------------------------------------------- figures (spec 9)

def _style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": INK2, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 9, "legend.frameon": False})


def _empty(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=INK2, transform=ax.transAxes)
    ax.set_axis_off()


def _fig_gallery(nets, I, flagged, path):
    """Up to 6 anchor edges; every network's best-matching curve, affine-aligned onto the anchor."""
    pairs = I["pairs"]
    ref = next((n for n, v in sorted(nets.items()) if len(v["feats"])), None)
    fig = plt.figure(figsize=(15, 13))
    if ref is None:
        _empty(fig.add_subplot(111), "no network has cluster edges")
    else:
        to = {p["b"]: p for p in pairs if p["a"] == ref}
        n_edges = len(nets[ref]["feats"])
        count = [sum(p["u_min"][i] < U_THR for p in to.values()) for i in range(n_edges)]
        anchors = sorted(range(n_edges), key=lambda i: (-count[i], i))[:6]
        for pos, i in enumerate(anchors):
            ax = fig.add_subplot(2, 3, pos + 1)
            ref_y, xs = nets[ref]["ys"][i], nets[ref]["xs"][i]
            for name in sorted(nets):
                k = i if name == ref else (to[name]["j"][i] if name in to else -1)
                if k < 0:
                    continue
                y = nets[name]["ys"][k]
                a, d = np.linalg.lstsq(np.c_[y, np.ones_like(y)], ref_y, rcond=None)[0]
                grp = "A" if name in GROUP_A else "B" if name in GROUP_B else "-"
                col = {"A": BLUE, "B": ORANGE, "-": INK2}[grp]
                fam = nets[name]["fits"][k].get("formula") or "no accepted formula"
                ax.plot(xs, a * y + d, color=col, lw=1.4, ls=["-", "--", ":", "-."][_seed(name) % 4],
                        label=f"{name}{'*' if name in flagged else ''} [{grp}]: {fam}")
            ax.set_title(f"{nets[ref]['feats'][i]}  (anchor {ref} edge {i})", fontsize=9)
            ax.set_xlabel("input (z-scored, anchor range)")
            ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    fig.suptitle("Matched edge curves after sign/scale/offset alignment "
                 "(colour = polarity group A blue, B orange; * = seed-0 flagged)", fontsize=10)
    fig.subplots_adjust(hspace=0.95, wspace=0.25, bottom=0.14, top=0.93)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_ecdf(I, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    tr = np.concatenate([p["u_min"] for p in I["pairs"]]) if I["pairs"] else np.zeros(0)
    un = np.concatenate([p["u_null"] for p in I["pairs"]]) if I["pairs"] else np.zeros(0)
    for v, col, lab in ((tr, BLUE, "trained vs trained"), (un, INK2, "trained vs untrained")):
        v = np.sort(v[np.isfinite(v)])
        if v.size:
            ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=col, lw=1.6, label=lab)
    ax.axvline(U_THR, color=ORANGE, ls="--", lw=1, label=f"u = {U_THR}")
    ax.set_xlabel("unexplained variance u of best-matching edge (0 = same curve up to affine map)")
    ax.set_ylabel("cumulative fraction of edges")
    ax.set_title("Do matched curves recur beyond chance?", fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_family_grid(I, native, path):
    fams = sorted({f for p in I["grid"] for f in p if f})
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if not fams:
        _empty(ax, "no matched pair with accepted formulas")
    else:
        M = np.zeros((len(fams), len(fams)))
        for a, b in I["grid"]:
            if a and b:
                M[fams.index(a), fams.index(b)] += 1
        ax.imshow(M, cmap=LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#0d366b"]))
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, int(v), ha="center", va="center", fontsize=8,
                    color="white" if v > M.max() / 2 else INK)
        ax.set_xticks(range(len(fams)), fams, rotation=45, ha="right")
        ax.set_yticks(range(len(fams)), fams)
        ax.grid(False)
        ag = native["agreement"]["all"]
        ax.set_title(f"Formula family of matched edges: same {ag['rate']:.2f} vs chance "
                     f"{ag['chance']:.2f} (n={ag['n']})", fontsize=9)
        ax.set_xlabel("family of matched edge (network B)")
        ax.set_ylabel("family of edge (network A)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_coef_heatmap(dist, term_names, flagged, path):
    rows = sorted(dist["networks"].items(),
                  key=lambda kv: (["label", "tkan", "minirocket"].index(kv[0].split(":")[0]), kv[0]))
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(rows) + 2.5))
    M = np.zeros((len(rows), len(term_names)))
    for r, (_, v) in enumerate(rows):
        c = np.array([v["coef"].get(t, 0.0) for t in term_names])
        M[r] = c / np.abs(c).max() if np.abs(c).max() > 0 else c
    ax.imshow(M, cmap=LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", "#e34948"]),
              vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(term_names)), term_names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)), [k + ("*" if k.split(":")[1] in flagged else "")
                                     for k, _ in rows], fontsize=7)
    ax.grid(False)
    ax.set_title("Distilled formula coefficients, each row scaled by its largest |coef| "
                 "(blue = negative, red = positive; grey = not selected; * = flagged)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_fidelity(dist, flagged, path):
    rows = sorted(dist["networks"].items(),
                  key=lambda kv: (["label", "tkan", "minirocket"].index(kv[0].split(":")[0]), kv[0]))
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rows) + 2), 4.5))
    for i, (k, v) in enumerate(rows):
        m = k.split(":")[0]
        ax.bar(i, v["r2"], color=MODEL_COLOR[m], width=0.7)
        ax.text(i, v["r2"], f"{v['r2']:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.axhline(0.5, color=INK, ls="--", lw=1)
    ax.text(len(rows) - 0.5, 0.51, "faithful: R2 >= 0.5", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(range(len(rows)), [k.replace(":", "\n") + ("*" if k.split(":")[1] in flagged else "")
                                     for k, _ in rows], fontsize=6.5)
    ax.set_ylabel("held-out R2 of the distilled formula")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MODEL_COLOR.values()]
    ax.legend(handles, MODEL_LABEL.values(), loc="upper right", bbox_to_anchor=(1, 0.9))
    ax.set_title("How well does a compact formula describe each model? (* = seed-0 flagged)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_diff_vs_sum(dist, path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, c in zip(axes, CHANNELS):
        for model, mk in (("tkan", "o"), ("minirocket", "s")):
            pts = [r for r in dist["diff_vs_sum"][model]["coefficients"] if r["channel"] == c]
            ax.scatter([r["D"] for r in pts], [r["S"] for r in pts], marker=mk, s=28,
                       color=MODEL_COLOR[model], edgecolor="#fcfcfb", linewidth=0.8,
                       label=MODEL_LABEL[model])
        lim = ax.get_xlim() + ax.get_ylim()
        r = max(abs(v) for v in lim) or 1.0
        ax.plot([-r, r], [-r, r], color=GRID, lw=1)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_title(c, fontsize=9)
        ax.set_xlabel(f"D_{c} coefficient (difference)")
        ax.set_ylabel(f"S_{c} coefficient (sum)")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Delay-difference vs delay-sum coefficients (above the diagonal: sum favoured)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- summary

def _f(v, d=2):
    return "n/a" if v is None else f"{v:.{d}f}"


def render_summary(m: dict) -> str:
    """One-page team summary; every number is read from the manifest ``m``."""
    c, n, d, k = m["curves"], m["formulas_native"], m["distilled"], m["contrast"]
    wf, af, pol = c["within_fold"], c["across_folds"], c["by_polarity"]
    ag, fu = n["agreement"]["all"], n["different_family_formula_u"]
    flagged = sorted(x for x, v in m["checks"]["tkan"].items() if v["flagged"])
    L = ["# Equations across seeds", "",
         "Do retrained TKAN networks learn the same equations, and what does MiniROCKET give?", "",
         "## 1. TKAN's own delay-edge curves",
         f"- Median fraction of an edge's best match within u < {U_THR} (same-fold network pairs): "
         f"**{_f(c['frac_trained'])}** (curves.frac_trained); untrained re-init 95th percentile "
         f"{_f(wf.get('median_null_p95'))}.",
         f"- The curves recur in {wf.get('n_recur', 0)} of {wf.get('n_pairs', 0)} testable same-fold pairs "
         f"and {af.get('n_recur', 0)} of {af.get('n_pairs', 0)} cross-fold pairs "
         f"(cross-fold median fraction {_f(af.get('median_frac_trained'))}); directed pairs with "
         f"no same-channel match are excluded ({wf['n_untestable']} same-fold, "
         f"{af['n_untestable']} cross-fold).",
         "- By polarity group (median fraction): " + ", ".join(
             f"{g} {_f(v.get('median_frac_trained'))} ({v['n_pairs']} pairs)" for g, v in pol.items()) + ".",
         "- Matching ignores sign, scale and offset by design.",
         f"- Networks with no cluster edges: {', '.join(c['no_cluster']) or 'none'}.", "",
         "## 2. Do the curves get the same formula?",
         f"- Same formula family in {ag['same']} of {ag['n']} matched edges (unordered network pairs): rate {_f(ag['rate'])} vs "
         f"chance {_f(ag['chance'])} ({ag['n_no_fit']} pairs without an accepted formula).",
         f"- Where families differ, the two fitted formulas still match as functions in "
         f"{_f(fu['frac_below_thr'])} of {fu['n']} edge matches (median u {_f(fu['median'], 3)}).", "",
         "## 3. Shared distilled formula (Lasso on 16 signal and delay terms)"]
    for model in ("tkan", "minirocket"):
        nets = {x: v for x, v in d["networks"].items() if x.startswith(model)}
        nf = sum(v["faithful"] for v in nets.values())
        L.append(f"- {MODEL_LABEL[model]}: {nf} of {len(nets)} networks reach held-out R2 >= 0.5 "
                 f"(median R2 {_f(float(np.median([v['r2'] for v in nets.values()])) if nets else None)}).")
        for fold, s in d["stability"][model].items():
            verdict = {True: "same equation", False: "different equations", None: "not testable"}[s["same"]]
            L.append(f"  - fold {fold}: {verdict} across seeds {s['seeds']}"
                     + (f" (median cross-seed Jaccard {_f(s['median_cross_jaccard'])} vs bootstrap 5th "
                        f"pct {_f(s['boot_p05'])}, signs consistent {s['sign_consistent']}, null seed "
                        f"{s['null_seed']})" if s["same"] is not None else ""))
    lab = [v for x, v in d["networks"].items() if x.startswith("label")]
    L.append(f"- Label reference: median held-out R2 {_f(float(np.median([v['r2'] for v in lab])) if lab else None)}.")
    if d["cross_model"]:
        L.append(f"- TKAN vs MiniROCKET selected terms: median Jaccard "
                 f"{_f(float(np.median([x['jaccard'] for x in d['cross_model']])))} over "
                 f"{len(d['cross_model'])} TKAN x MiniROCKET seed pairs within fold.")
    for model, v in d["diff_vs_sum"].items():
        tot = {key: sum(ch[key] for ch in v["by_channel"].values()) for key in ("S", "D", "none")}
        L.append(f"- {MODEL_LABEL[model]} delay term picked: sum {tot['S']}, difference {tot['D']}, "
                 f"neither {tot['none']} (network-channel cells).")
    L += ["", "## 4. MiniROCKET contrast",
          f"- 90% of the ridge weight mass sits in a median of {_f(k['median_k90'], 0)} kernel features; "
          f"keeping only the top 20 retains {_f(k['median_retention'])} of the file-level AUC.", "",
          "## 5. Checks and caveats",
          f"- Flagged seed-0 networks (no evaluation-path clearance): {', '.join(flagged) or 'none'}.",
          f"- AUC differences below the {AUC_FLOOR} repeatability floor are not claimable.",
          f"- Sidecars complete: {m['checks']['sidecars_complete']}; window ids and hashes match "
          f"for every dump.",
          "- MiniROCKET seed-0 reproduction: " + ", ".join(
              f"fold {f} {v['status']}" for f, v in m["checks"]["minirocket_seed0"].items()) + ".",
          f"- Dumps missing from the expected set: {', '.join(m['checks']['missing']) or 'none'}."]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- main

def _git_sha():
    try:
        return subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--equations-dir", required=True, type=Path)
    ap.add_argument("--figures-dir", default="results/figures/research_update_assets", type=Path)
    ap.add_argument("--summary-md", default="docs/phase3_tkan/equations_across_seeds.md", type=Path)
    ap.add_argument("--minirocket-manifests", default="results/phase3/minirocket", type=Path)
    ap.add_argument("--recurrence-manifest", type=Path,
                    default=Path("results/phase3/xai/recurrence/recurrence_fold0_fold2.json"))
    ap.add_argument("--runs-root", type=Path, default=Path("results/phase3"),
                    help="holds runs*/<run_id>/summary.json (recorded validation AUCs)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="proceed when part of the expected dump set has not landed (recorded)")
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args(argv)

    wins, tkan, mr, evs, problems = load_dir(a.equations_dir)
    missing = expected_missing(wins, tkan, mr, evs)
    if missing and not a.allow_missing:
        problems.append("expected dumps missing (pass --allow-missing to proceed): " + ", ".join(missing))
    if not wins or not tkan:
        problems.append(f"{a.equations_dir}: need window files and TKAN dumps")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    checks, problems = run_checks(wins, tkan, mr, evs, a.minirocket_manifests,
                                  a.recurrence_manifest, a.runs_root)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    checks["missing"], checks["allow_missing"] = missing, a.allow_missing

    nets = load_nets(tkan)
    curves, native, internals = curves_block(nets)
    dist = distilled_block(wins, tkan, mr, a.n_boot)
    manifest = {
        "provenance": {
            "git_sha": _git_sha(),
            "windows": {str(f): {"path": str(w["path"]), "sha256": w["meta"]["sha256"]}
                        for f, w in wins.items()},
            "tkan": {n: {"checkpoint": d["meta"].get("checkpoint"),
                         "checkpoint_sha256": d["meta"].get("checkpoint_sha256")}
                     for n, d in tkan.items()},
            "minirocket": [f"f{f}s{s}" for f, s in sorted(mr)]},
        "checks": checks, "curves": curves, "formulas_native": native, "distilled": dist,
        "contrast": contrast_block(wins, mr)}
    flagged = {n for n, v in checks["tkan"].items() if v["flagged"]}

    _style()
    a.figures_dir.mkdir(parents=True, exist_ok=True)
    f = a.figures_dir
    _fig_gallery(nets, internals, flagged, f / "eq_curve_gallery.png")
    _fig_ecdf(internals, f / "eq_unexplained_ecdf.png")
    _fig_family_grid(internals, native, f / "eq_family_grid.png")
    term_names = list(next(iter(wins.values()))["z"]["term_names"])
    _fig_coef_heatmap(dist, term_names, flagged, f / "eq_coef_heatmap.png")
    _fig_fidelity(dist, flagged, f / "eq_fidelity.png")
    _fig_diff_vs_sum(dist, f / "eq_diff_vs_sum.png")

    manifest = _clean({**manifest, "complete": True})
    (a.equations_dir / "equations_manifest.json").write_text(json.dumps(manifest, indent=2))
    a.summary_md.parent.mkdir(parents=True, exist_ok=True)
    a.summary_md.write_text(render_summary(manifest))
    print(f"wrote manifest, 6 figures in {f}, summary {a.summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
