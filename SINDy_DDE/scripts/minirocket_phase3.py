#!/usr/bin/env python3
"""Exp 3: MiniROCKET comparator on one fold, both input arms, with delay tests.

What it answers (pre-committed in docs/superpowers/plans/2026-10-05-paper-closeout.md):

* arm A (4 channels) vs arm B (4 + 4 delayed): does a plain, physics-free
  classifier need the delay to detect chatter?
* on arm B, at TEST time only: delayed channels zeroed, delay misstated x0.5
  and x2. Does it rely on the delayed signal once handed it, and is the
  reliance specific to the physical delay?

A drop counts only above the 0.0092 repeatability floor.

Data path, matched to TKAN step for step
----------------------------------------
* Split: exactly ``build_fold_loaders`` -- val rows are those whose
  ``parent_group`` maps to ``--fold`` in ``group_to_fold``; the mirror-leakage
  guard runs before any file is read. (Replicated here rather than calling
  ``build_fold_loaders``, which would cache every window of the fold in RAM.)
* Normalisation: per-channel z-score fit on ALL windows of ALL training files
  (excluded rows included, as TKAN's train dataset includes them), streamed.
* Training set: supervised files only (``exclude_supervised`` false and
  ``sample_weight > 0`` -- the rows TKAN's loss gives weight 0 are dropped),
  up to ``--max-train-per-file`` evenly spaced windows each.
* Evaluation: ALL val files, ALL windows (TKAN's val AUC is over all val
  files), and ALL windows of held-out 121-125 via ``build_eval_table``; the
  score is the file-mean ridge decision. Streamed in file chunks.

Optional outputs for the TKAN-vs-MiniROCKET noise comparison (all off by
default; with them off the manifest is byte-for-byte what it was):

* ``--save-scores``: per-file CSVs ``scores_minirocket_{A,B}_fold<F>_{val,heldout}.csv``
  (+ JSON sidecars) in ``--scores-dir``, in the schema of
  :data:`pri2m.xai.model_compare.SCORE_COLUMNS`.
* ``--noise-levels`` + ``--noise-calibration``: ``..._val_noise.csv`` -- clean val
  files with :func:`pri2m.xai.noise_probe.inject_noise` applied to the full raw
  signal before windowing, <= ``--noise-max-windows-per-file`` evenly spaced
  windows each (see :func:`score_noise`).
* ``--save-model``: joblib dump of the fitted arms + norm.

The manifest is written once, at the end, with ``"complete": true``; nothing
partial is ever written, so an existing file is a finished run.

Example::

    python scripts/minirocket_phase3.py --fold 0 \\
        --table results/phase3/phase3_dataset.csv \\
        --folds-json results/phase3/phase3_folds.json \\
        --npz-root phase2_signals \\
        --features-csv data/metadata/features_datasets_121_125.csv --n-jobs 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pri2m.data.phase3_sequence import (  # noqa: E402
    WindowConfig,
    _npz_path_for_row,
    _row_parent_group,
    aggregate_file_predictions,
    iter_windows,
    load_npz_signals,
    tau_samples,
)
from pri2m.models.minirocket_baseline import (  # noqa: E402
    DEFAULT_ALPHAS,
    MiniRocketBaseline,
    StreamingNorm,
    build_inputs,
    file_level_metrics,
    select_train_windows,
    summarise_delay_tests,
)
from pri2m.phase3.splits import assert_no_mirror_leakage, parent_group  # noqa: E402
from pri2m.xai.delay_variants import delay_drop  # noqa: E402
from pri2m.xai.model_compare import SCORE_COLUMNS, write_score_file  # noqa: E402
from pri2m.xai.noise_probe import inject_noise, noise_tier  # noqa: E402

DEFAULT_FLOOR = 0.0092

#: name -> (arm, build_inputs kwargs). Variants are TEST-TIME only: every B_*
#: entry is scored by the model trained on unmodified arm-B inputs.
VARIANTS: dict[str, tuple[str, dict]] = {
    "A": ("A", {}),
    "B": ("B", {}),
    "B_zero_delay": ("B", {"zero_delay": True}),
    "B_tau_x0.5": ("B", {"tau_scale": 0.5}),
    "B_tau_x2": ("B", {"tau_scale": 2.0}),
}


# --------------------------------------------------------------------------- #
# Split + file iteration
# --------------------------------------------------------------------------- #
def split_fold(table: pd.DataFrame, folds_json: dict, fold_k: int):
    """``(train_table, val_table)`` by the rule ``build_fold_loaders`` uses."""
    group_to_fold = {int(k): int(v) for k, v in folds_json["group_to_fold"].items()}
    pg = (table["parent_group"].astype(int) if "parent_group" in table.columns
          else parent_group(table["dataset"]))
    fold_of_row = pg.astype(int).map(group_to_fold)
    val_mask = fold_of_row == int(fold_k)
    val_table, train_table = table[val_mask], table[~val_mask]
    assert_no_mirror_leakage(train_table, val_table)
    return train_table, val_table


def is_supervised(row: pd.Series) -> bool:
    """The rows TKAN's loss actually trains on (non-excluded, weight > 0)."""
    if "exclude_supervised" in row and bool(row["exclude_supervised"]):
        return False
    if "sample_weight" in row and pd.notna(row["sample_weight"]):
        return float(row["sample_weight"]) > 0
    return True


def iter_file_windows(table: pd.DataFrame, npz_root, cfg: WindowConfig):
    """Yield ``(row, windows float32 [n, L, C], meta)`` for files with >= 1 window.

    float32 like ``Phase3SequenceDataset`` (which stores float32 and normalises
    in float32), and files with no full window are skipped, as it skips them.
    """
    root = Path(npz_root)
    for _, row in table.iterrows():
        signal, meta = load_npz_signals(_npz_path_for_row(root, row), cfg.channels)
        w = iter_windows(signal, cfg)
        if w.shape[0] == 0:
            continue
        yield row, w.astype(np.float32), meta


def collect_train(train_table, npz_root, cfg, max_per_file):
    """One pass over training files: norm stats + the selected raw windows."""
    sn = StreamingNorm(len(cfg.channels))
    raw, tau, lab, files, groups, wts = [], [], [], [], [], []
    n_files_seen = 0
    for row, w, meta in iter_file_windows(train_table, npz_root, cfg):
        n_files_seen += 1
        sn.update(w)
        if not is_supervised(row):
            continue
        idx = select_train_windows(w.shape[0], max_per_file)
        raw.append(w[idx])
        k = len(idx)
        tau.extend([tau_samples(meta["fs"], meta["tpf"])] * k)
        lab.extend([float(meta["label"])] * k)
        files.extend([str(row["file"])] * k)
        groups.extend([_row_parent_group(row)] * k)
        # The same per-file weight TKAN's loss uses (1.0 if the table lacks it).
        sw = row["sample_weight"] if "sample_weight" in row else 1.0
        wts.extend([float(sw) if pd.notna(sw) else 1.0] * k)
    if not raw:
        raise ValueError("no supervised training windows in this fold")
    return sn.to_normstats(), {
        "raw": np.concatenate(raw), "tau": np.asarray(tau), "label": np.asarray(lab),
        "file": np.asarray(files), "group": np.asarray(groups),
        "weight": np.asarray(wts, dtype=np.float64),
        "n_norm_files": n_files_seen,
    }


def _inputs_chunked(raw, tau, norm, pool, arm, chunk=4096, **kw):
    """Normalise + build inputs in chunks (avoids a full normalised copy)."""
    parts = [build_inputs(norm.apply(raw[s : s + chunk]), tau[s : s + chunk], pool=pool,
                          arm=arm, **kw) for s in range(0, raw.shape[0], chunk)]
    return np.concatenate(parts)


def _file_record(row: pd.Series, meta: dict, n_windows: int,
                 tier_from_metadata: bool = False) -> dict:
    """The schema columns of one file that do not depend on the model.

    ``noise_tier`` comes from the 1-120 id layout, except for the held-out set
    (``tier_from_metadata``), where it is the features CSV's ``noise_std``
    (percent) and blank when that column is absent.
    """
    ds = int(row["dataset"])
    if tier_from_metadata:
        tier = (float(row["noise_std"]) if "noise_std" in row and pd.notna(row["noise_std"])
                else None)
    else:
        tier = noise_tier(ds)
    omega = meta["omega"]  # from the NPZ, as the TKAN scorer reports it
    bob = row["b_over_blim"] if "b_over_blim" in row else np.nan
    return {
        "file": str(row["file"]), "dataset": ds,
        "parent_group": int(_row_parent_group(row)),
        "noise_tier": np.nan if tier is None else tier,
        "label": int(float(meta["label"]) >= 0.5),
        "omega": float(omega),
        "b_over_blim": float(bob) if pd.notna(bob) else np.nan,
        "n_windows": int(n_windows),
    }


def score_split(table, npz_root, cfg, norm, models, pool, chunk_files,
                file_records: list | None = None, tier_from_metadata: bool = False):
    """Window scores for every variant over ALL windows of ``table``'s files.

    If ``file_records`` is a list, one dict per scored file is appended to it:
    the schema metadata plus ``score_<arm>`` (file-mean decision) for each
    fitted arm's unmodified variant. The returned metrics are unaffected.
    """
    names = [v for v in VARIANTS if VARIANTS[v][0] in models]
    scores = {v: [] for v in names}
    labels, files = [], []
    buf = []
    spans = []  # (row, meta, start, n)

    def flush():
        if not buf:
            return
        w = norm.apply(np.concatenate([b[0] for b in buf]))
        t = np.concatenate([b[1] for b in buf])
        for v in names:
            arm, kw = VARIANTS[v]
            scores[v].append(models[arm].decision_function(
                build_inputs(w, t, pool=pool, arm=arm, **kw)))
        buf.clear()

    n_files = 0
    for row, w, meta in iter_file_windows(table, npz_root, cfg):
        n = w.shape[0]
        if file_records is not None:
            spans.append((row, meta, len(files), n))
        buf.append((w, np.full(n, tau_samples(meta["fs"], meta["tpf"]))))
        labels.extend([float(meta["label"])] * n)
        files.extend([str(row["file"])] * n)
        n_files += 1
        if len(buf) >= chunk_files:
            flush()
    flush()
    metrics, all_scores = {}, {}
    for v in names:
        s = np.concatenate(scores[v]) if scores[v] else np.zeros(0)
        all_scores[v] = s
        metrics[v] = file_level_metrics(s, np.asarray(labels), files)
    if file_records is not None:
        arms = [a for a in ("A", "B") if a in models]
        for row, meta, start, n in spans:
            rec = _file_record(row, meta, n, tier_from_metadata)
            for arm in arms:
                # Same arithmetic as file_level_metrics' per-file mean.
                rec[f"score_{arm}"] = aggregate_file_predictions(
                    all_scores[arm][start : start + n], [0] * n)[0]
            file_records.append(rec)
    return metrics, n_files, len(files)


def score_noise(val_table, npz_root, cfg, norm, models, pool, levels, rel_sigma,
                max_windows, seed, chunk_files):
    """File-mean decisions on clean val files with noise injected per level.

    For every clean (tier 0) val file: the FULL raw ``[T, C]`` signal gets
    ``inject_noise(signal, level, rel_sigma, key=<file name>, seed)`` BEFORE
    windowing (level 0 = unchanged), then the same windowing, float32 cast,
    ``select_train_windows(n, max_windows)`` subset, train-fold z-score and
    ``build_inputs`` as everywhere else. The TKAN scorer does the same, so the
    two models see bit-identical noisy windows. Returns a list of records
    (schema metadata + ``level`` + ``score_<arm>``).
    """
    arms = [a for a in ("A", "B") if a in models]
    root = Path(npz_root)
    records, buf = [], []

    def flush():
        if not buf:
            return
        w = norm.apply(np.concatenate([b[1] for b in buf]))
        t = np.concatenate([b[2] for b in buf])
        dec = {arm: models[arm].decision_function(build_inputs(w, t, pool=pool, arm=arm))
               for arm in arms}
        start = 0
        for rec, ww, _ in buf:
            n = ww.shape[0]
            for arm in arms:
                rec[f"score_{arm}"] = aggregate_file_predictions(
                    dec[arm][start : start + n], [0] * n)[0]
            start += n
            records.append(rec)
        buf.clear()

    n_files = 0
    for _, row in val_table.iterrows():
        if noise_tier(int(row["dataset"])) != 0:
            continue
        signal, meta = load_npz_signals(_npz_path_for_row(root, row), cfg.channels)
        tau = tau_samples(meta["fs"], meta["tpf"])
        n_files += 1
        for level in levels:
            noisy = inject_noise(signal, level, rel_sigma, key=str(row["file"]), seed=seed)
            w = iter_windows(noisy, cfg)
            if w.shape[0] == 0:
                break
            idx = select_train_windows(w.shape[0], max_windows)
            ww = w[idx].astype(np.float32)
            rec = _file_record(row, meta, len(idx))
            rec["level"] = float(level)
            buf.append((rec, ww, np.full(len(idx), tau)))
        if len(buf) >= chunk_files:
            flush()
    flush()
    return records, n_files


def _records_to_frame(records, model, fold, split, arm, level=None) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["model"], df["fold"], df["split"] = model, int(fold), split
    if level is not None:
        df["level"] = float(level)
    df["score"] = df[f"score_{arm}"]
    return df.loc[:, list(SCORE_COLUMNS)]


def load_noise_calibration(path) -> list[float]:
    """``summary.rel_sigma_per_unit`` from the calibration JSON, or ValueError."""
    cal = json.loads(Path(path).read_text())
    summary = cal.get("summary") if isinstance(cal, dict) else None
    rel = (summary or {}).get("rel_sigma_per_unit")
    if rel is None:
        reason = (summary or {}).get("undecided_reason")
        raise ValueError(f"{path}: summary.rel_sigma_per_unit is None "
                         f"(undecided_reason={reason!r}); refusing to inject noise")
    rel = [float(x) for x in rel]
    if not all(np.isfinite(rel)) or any(x < 0 for x in rel):
        raise ValueError(f"{path}: rel_sigma_per_unit must be finite and >= 0, got {rel}")
    return rel


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _aeon_version():
    try:
        import aeon

        return str(aeon.__version__)
    except Exception:  # noqa: BLE001 - absence is recorded, not fatal here
        return None


def _write_score_outputs(args, cfg, norm, models, arm_prov, val_table, file_recs,
                         levels, rel_sigma, transformer_factory, walls) -> dict:
    """Per-file score CSVs (+ sidecars), noise rows, and the model dump."""
    arms = [a for a in ("A", "B") if a in models]
    base_prov = {
        "runner": "scripts/minirocket_phase3.py", "fold": int(args.fold),
        "seed": int(args.seed), "n_kernels": int(args.n_kernels),
        "aeon_version": None if transformer_factory is not None else _aeon_version(),
        "pool": int(args.pool), "window": {"length": cfg.length, "stride": cfg.stride},
        "channels": list(cfg.channels), "table": str(args.table),
        "folds_json": str(args.folds_json), "npz_root": str(args.npz_root),
        "features_csv": args.features_csv,
        "heldout_npz_root": args.heldout_npz_root or args.npz_root,
        "sample_weighted": not args.no_sample_weight,
        "max_train_per_file": int(args.max_train_per_file),
        "norm": norm.to_dict(),
        "score": "file-mean ridge decision (+-1 targets); accuracy threshold 0",
    }
    out = {"files": [], "model": None}
    sd = Path(args.scores_dir)
    for split in ("val", "heldout"):
        recs = file_recs.get(split)
        if not recs:
            if recs is not None:
                print(f"[minirocket] no {split} files to save scores for", flush=True)
            continue
        for arm in arms:
            prov = {**base_prov, "arm": arm, "chosen_alpha": arm_prov[arm]["chosen_alpha"],
                    "windows": "all windows per file"}
            p = write_score_file(_records_to_frame(recs, f"minirocket_{arm}", args.fold,
                                                   split, arm, level=0.0), sd, prov)
            out["files"].append(str(p))
    if levels is not None:
        t0 = time.time()
        recs, n_noise_files = score_noise(
            val_table, args.npz_root, cfg, norm, models, args.pool, levels, rel_sigma,
            args.noise_max_windows_per_file, args.noise_seed, args.eval_chunk_files)
        walls["val_noise_eval_seconds"] = round(time.time() - t0, 1)
        print(f"[minirocket] val_noise: {n_noise_files} clean val files x "
              f"{len(levels)} levels ({walls['val_noise_eval_seconds']}s)", flush=True)
        for arm in arms:
            prov = {**base_prov, "arm": arm, "chosen_alpha": arm_prov[arm]["chosen_alpha"],
                    "noise": {
                        "levels": levels, "calibration": str(args.noise_calibration),
                        "rel_sigma_per_unit": rel_sigma, "files": args.noise_files,
                        "max_windows_per_file": int(args.noise_max_windows_per_file),
                        "window_selection": "select_train_windows (evenly spaced)",
                        "seed": int(args.noise_seed),
                        "inject": ("noise_probe.inject_noise on the full raw [T, C] "
                                   "signal before windowing; key = file name"),
                    }}
            if not recs:
                print("[minirocket] no clean val files: val_noise not written", flush=True)
                break
            p = write_score_file(_records_to_frame(recs, f"minirocket_{arm}", args.fold,
                                                   "val_noise", arm), sd, prov)
            out["files"].append(str(p))
    if args.save_model:
        import joblib

        mp = Path(args.save_model)
        mp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": models, "norm": norm, "fold": int(args.fold),
                     "arms": arms, "provenance": {**base_prov, "arms": arm_prov}},
                    mp)
        out["model"] = str(mp)
        print(f"[minirocket] saved models to {mp}", flush=True)
    return out


def check_window_sidecar(args) -> str | None:
    """Problem with ``--window-file``'s sidecar/npz (None if fine); cheap, run before the fit."""
    wf = Path(args.window_file)
    side_p = wf.with_suffix(".json")
    if not side_p.is_file():
        return f"window sidecar not found: {side_p}"
    side = json.loads(side_p.read_text())
    if side.get("complete") is not True:
        return f"{side_p}: not complete (complete={side.get('complete')!r})"
    if side.get("fold") != args.fold:
        return f"{side_p}: fold {side.get('fold')!r} != --fold {args.fold}"
    if side.get("sha256") != hashlib.sha256(wf.read_bytes()).hexdigest():
        return f"{side_p}: sha256 does not match {wf}"
    with np.load(wf) as z:
        need = {f"{p}_{k}" for p in ("val", "ho") for k in ("window", "tau", "file")}
        if need - set(z.files):
            return f"{wf}: missing arrays {sorted(need - set(z.files))}"
    if not Path(side.get("preprocessing", "")).is_file():
        return f"{side_p}: preprocessing {side.get('preprocessing')!r} not found"
    return None


def window_norm_diff(args, norm) -> float:
    """Max |mean/std| gap between the script's norm and the windows' preprocessing norm."""
    side = json.loads(Path(args.window_file).with_suffix(".json").read_text())
    pn = json.loads(Path(side["preprocessing"]).read_text())["norm"]
    return float(max(np.abs(norm.mean - np.asarray(pn["mean"])).max(),
                     np.abs(norm.std - np.asarray(pn["std"])).max()))


def dump_window_scores(args, clf, norm_max_abs_diff: float, top_k: int = 20) -> Path:
    """Score a shared window file with the fitted arm-B model (equations-across-seeds).

    The npz windows are already z-scored with this fold's train stats (the same
    ``norm.apply`` output ``score_split`` hands to ``build_inputs``), so they go
    straight in. Writes ``minirocket_fold<F>_s<SEED>.npz`` + ``.json`` sidecar:
    full decision, decision with all but the ``top_k`` largest |w| zeroed, weights.
    """
    wf = Path(args.window_file)
    z = np.load(wf)
    r = clf.ridge_
    w = r.weights[0]
    w20 = np.zeros_like(w)
    top = np.argsort(-np.abs(w))[:top_k]
    w20[top] = w[top]
    out = {"weights": w}
    for p in ("val", "ho"):
        F = clf.transform(build_inputs(z[f"{p}_window"], z[f"{p}_tau"], pool=args.pool, arm="B"))
        out[f"{p}_score"] = r.decision(F, 0)
        out[f"{p}_file"] = z[f"{p}_file"]
        if p == "ho":
            out["ho_score_top20"] = (((F.astype(np.float64) - r.mean) / r.std) @ w20
                                     + r.intercept + r._offset(0))
    eq = Path(args.equations_dir)
    eq.mkdir(parents=True, exist_ok=True)
    stem = eq / f"minirocket_fold{args.fold}_s{args.seed}"
    np.savez(stem.with_suffix(".npz"), **out)
    prior = None
    man = Path(args.output_dir) / f"minirocket_fold{args.fold}.json"
    if args.seed == 0 and man.is_file():
        prior = json.loads(man.read_text()).get("metrics", {}).get("heldout", {}).get(
            "B", {}).get("auc")
    npz_sha = hashlib.sha256(stem.with_suffix(".npz").read_bytes()).hexdigest()
    stem.with_suffix(".json").write_text(json.dumps({
        "fold": int(args.fold), "seed": int(args.seed), "arm": "B",
        "window_sha256": hashlib.sha256(wf.read_bytes()).hexdigest(),
        "manifest_heldout_auc": prior, "norm_max_abs_diff": norm_max_abs_diff,
        "top_k": top_k, "npz_sha256": npz_sha, "complete": True}, indent=2))
    print(f"[minirocket] wrote window scores {stem}.npz", flush=True)
    return stem.with_suffix(".npz")


def main(argv: list[str] | None = None, transformer_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--arm", choices=("A", "B", "both"), default="both")
    ap.add_argument("--table", default="results/phase3/phase3_dataset.csv")
    ap.add_argument("--folds-json", default="results/phase3/phase3_folds.json")
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--features-csv", default=None,
                    help="Held-out 121-125 features CSV (optional).")
    ap.add_argument("--heldout-npz-root", default=None,
                    help="NPZ root for the held-out set (default: --npz-root).")
    ap.add_argument("--max-train-per-file", type=int, default=8)
    ap.add_argument("--n-kernels", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sample-weight", action="store_true",
                    help="Fit the ridge unweighted. Default weights each window by "
                         "its file's sample_weight, as TKAN's loss does.")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--alphas", default=None,
                    help="Comma-separated ridge alphas (default: logspace(-3, 6, 19)).")
    ap.add_argument("--inner-frac", type=float, default=0.2)
    ap.add_argument("--pool", type=int, default=4, help="Must match TKAN (configs: 4).")
    ap.add_argument("--window-length", type=int, default=1024)
    ap.add_argument("--window-stride", type=int, default=512)
    ap.add_argument("--chunk-size", type=int, default=4096,
                    help="Cases per transform/ridge chunk.")
    ap.add_argument("--eval-chunk-files", type=int, default=128)
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    ap.add_argument("--output-dir", default="results/phase3/minirocket")
    ap.add_argument("--tag", default=None)
    sc = ap.add_argument_group(
        "per-file scores + noise injection (TKAN-vs-MiniROCKET noise comparison)")
    sc.add_argument("--save-scores", action="store_true",
                    help="Write per-file score CSVs (+ sidecars) for val and held-out, "
                         "arms A and B (unmodified variant only).")
    sc.add_argument("--scores-dir", default="results/phase3/xai/noise_compare")
    sc.add_argument("--noise-levels", default=None,
                    help="Comma list of injected noise levels in tier units "
                         "(e.g. 0,1,2.5,5,7.5,10); scored on clean val files.")
    sc.add_argument("--noise-calibration", default=None,
                    help="mirror_noise_calibration.json (summary.rel_sigma_per_unit).")
    sc.add_argument("--noise-files", choices=("clean",), default="clean")
    sc.add_argument("--noise-max-windows-per-file", type=int, default=8)
    sc.add_argument("--noise-seed", type=int, default=0)
    sc.add_argument("--save-model", default=None,
                    help="joblib path for the fitted arms + norm (no refit for later XAI).")
    ew = ap.add_argument_group("shared-window scoring (equations across seeds)")
    ew.add_argument("--window-file", default=None,
                    help="windows_fold<F>.npz; after the fit, score its windows (arm B).")
    ew.add_argument("--equations-dir", default=None,
                    help="Where the window-score dump goes (required with --window-file).")
    args = ap.parse_args(argv)
    if args.window_file:
        if args.arm == "A":
            ap.error("--window-file needs arm B (use --arm B or both)")
        if not args.equations_dir:
            ap.error("--window-file requires --equations-dir")
        if not Path(args.window_file).is_file():
            ap.error(f"--window-file not found: {args.window_file}")
        if (problem := check_window_sidecar(args)):
            ap.error(problem)
        # The landed minirocket_fold<F>.json manifests must never be overwritten.
        if not (args.tag or "").startswith("eq_"):
            ap.error("--window-file requires --tag starting with 'eq_'")
        if (Path(args.output_dir) / f"minirocket_{args.tag}.json").exists():
            ap.error(f"{Path(args.output_dir) / f'minirocket_{args.tag}.json'} already exists; choose a new --tag")

    # Validate the noise inputs BEFORE the (hours-long) fit.
    levels, rel_sigma = None, None
    if args.noise_levels is not None:
        try:
            levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]
        except ValueError:
            ap.error(f"--noise-levels must be a comma list of numbers: {args.noise_levels!r}")
        if not levels or any((not np.isfinite(x)) or x < 0 for x in levels):
            ap.error(f"--noise-levels must be finite and >= 0: {args.noise_levels!r}")
        if len(set(levels)) != len(levels):
            ap.error(f"--noise-levels has duplicates: {args.noise_levels!r}")
        if not args.noise_calibration:
            ap.error("--noise-levels requires --noise-calibration")
        if not Path(args.noise_calibration).is_file():
            ap.error(f"--noise-calibration not found: {args.noise_calibration}")
        try:
            rel_sigma = load_noise_calibration(args.noise_calibration)
        except ValueError as exc:
            ap.error(str(exc))
        if len(rel_sigma) != len(WindowConfig().channels):
            ap.error(f"rel_sigma_per_unit has {len(rel_sigma)} entries for "
                     f"{len(WindowConfig().channels)} channels")
    elif args.noise_calibration:
        ap.error("--noise-calibration given without --noise-levels")
    if args.noise_max_windows_per_file < 1:
        ap.error("--noise-max-windows-per-file must be >= 1")
    want_files = args.save_scores

    t_start = time.time()
    arms = ["A", "B"] if args.arm == "both" else [args.arm]
    alphas = (tuple(float(a) for a in args.alphas.split(",")) if args.alphas
              else DEFAULT_ALPHAS)
    cfg = WindowConfig(length=args.window_length, stride=args.window_stride)
    tag = args.tag or f"fold{args.fold}"
    out_dir = Path(args.output_dir)
    dest = out_dir / f"minirocket_{tag}.json"

    table = pd.read_csv(args.table)
    folds_json = json.loads(Path(args.folds_json).read_text())
    train_table, val_table = split_fold(table, folds_json, args.fold)
    print(f"[minirocket] fold {args.fold}: {len(train_table)} train rows, "
          f"{len(val_table)} val rows; arms {arms}", flush=True)

    t0 = time.time()
    norm, tr = collect_train(train_table, args.npz_root, cfg, args.max_train_per_file)
    t_load = time.time() - t0
    norm_diff = None
    if args.window_file:
        norm_diff = window_norm_diff(args, norm)
        if norm_diff > 1e-6:
            raise SystemExit(f"[minirocket] window-file norm differs from this fit's train-fold "
                             f"norm by {norm_diff:.3g} (> 1e-6); windows were z-scored with a "
                             f"different norm, refusing to score")
    n_train_files = int(np.unique(tr["file"]).size)
    print(f"[minirocket] norm fit on {tr['n_norm_files']} files; "
          f"{tr['raw'].shape[0]} train windows from {n_train_files} supervised files "
          f"({t_load:.0f}s)", flush=True)

    models, arm_prov = {}, {}
    for arm in arms:
        t0 = time.time()
        X = _inputs_chunked(tr["raw"], tr["tau"], norm, args.pool, arm, args.chunk_size)
        clf = MiniRocketBaseline(n_kernels=args.n_kernels, seed=args.seed,
                                 n_jobs=args.n_jobs, alphas=alphas,
                                 inner_frac=args.inner_frac, chunk_size=args.chunk_size,
                                 transformer_factory=transformer_factory)
        clf.fit(X, tr["label"], tr["group"], tr["file"],
                sample_weight=None if args.no_sample_weight else tr["weight"])
        del X
        models[arm] = clf
        sel = clf.selection_
        arm_prov[arm] = {
            "n_channels": len(cfg.channels) * (2 if arm == "B" else 1),
            "n_features": clf.n_features_,
            "chosen_alpha": clf.alpha_,
            "alpha_at_grid_edge": sel["at_grid_edge"],
            "inner_auc_by_alpha": dict(zip([repr(a) for a in sel["alphas"]],
                                           sel["inner_auc"])),
            "n_inner_train_groups": sel["n_inner_train_groups"],
            "n_inner_val_groups": sel["n_inner_val_groups"],
            "n_train_windows": int(tr["raw"].shape[0]),
            "n_train_files": n_train_files,
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(f"[minirocket] arm {arm}: alpha={clf.alpha_:g} "
              f"(edge={sel['at_grid_edge']}) fit {arm_prov[arm]['fit_seconds']}s",
              flush=True)
    del tr
    if args.window_file:
        dump_window_scores(args, models["B"], norm_diff)

    metrics, counts, walls = {}, {}, {}
    t0 = time.time()
    file_recs = {"val": [] if want_files else None,
                 "heldout": [] if want_files else None}
    metrics["val"], counts["n_val_files"], counts["n_val_windows"] = score_split(
        val_table, args.npz_root, cfg, norm, models, args.pool, args.eval_chunk_files,
        file_records=file_recs["val"])
    walls["val_eval_seconds"] = round(time.time() - t0, 1)
    print(f"[minirocket] val: " + ", ".join(
        f"{k}={v['auc']}" for k, v in metrics["val"].items()), flush=True)

    if args.features_csv:
        import eval_heldout_phase3 as EH

        held_root = args.heldout_npz_root or args.npz_root
        held_table, _ = EH.build_eval_table(args.features_csv, held_root)
        t0 = time.time()
        metrics["heldout"], counts["n_heldout_files"], counts["n_heldout_windows"] = (
            score_split(held_table, held_root, cfg, norm, models, args.pool,
                        args.eval_chunk_files, file_records=file_recs["heldout"],
                        tier_from_metadata=True))
        walls["heldout_eval_seconds"] = round(time.time() - t0, 1)
        print(f"[minirocket] heldout: " + ", ".join(
            f"{k}={v['auc']}" for k, v in metrics["heldout"].items()), flush=True)
    else:
        counts["n_heldout_files"] = None
        counts["n_heldout_windows"] = None

    scores_outputs = None
    if want_files or levels is not None or args.save_model:
        scores_outputs = _write_score_outputs(
            args, cfg, norm, models, arm_prov, val_table, file_recs, levels, rel_sigma,
            transformer_factory, walls)

    delay_tests, b_minus_a = {}, {}
    for split, ms in metrics.items():
        delay_tests[split] = (summarise_delay_tests(ms, baseline="B", floor=args.floor)
                              if "B" in ms else None)
        b_minus_a[split] = (delay_drop((ms.get("B") or {}).get("auc"),
                                       (ms.get("A") or {}).get("auc"))
                            if "A" in ms and "B" in ms else None)

    manifest = {
        "analysis": "Exp 3: MiniROCKET comparator (arm A = 4 ch, arm B = 4 + 4 delayed)",
        "provenance": {
            "sample_weighted": not args.no_sample_weight,
            "fold": int(args.fold), "arm": args.arm, "seed": int(args.seed),
            "n_kernels": int(args.n_kernels), "aeon_version": (
                None if transformer_factory is not None else _aeon_version()),
            "n_jobs": int(args.n_jobs), "alpha_grid": [float(a) for a in alphas],
            "inner_frac": float(args.inner_frac), "pool": int(args.pool),
            "window": {"length": cfg.length, "stride": cfg.stride},
            "channels": list(cfg.channels),
            "max_train_per_file": int(args.max_train_per_file),
            "table": str(args.table), "folds_json": str(args.folds_json),
            "npz_root": str(args.npz_root), "features_csv": args.features_csv,
            "norm": norm.to_dict(),
            "arms": arm_prov,
            "n_val_files": counts["n_val_files"],
            "n_val_windows": counts["n_val_windows"],
            "n_heldout_files": counts["n_heldout_files"],
            "n_heldout_windows": counts["n_heldout_windows"],
            "wall_seconds": {"train_load": round(t_load, 1), **walls,
                             "total": round(time.time() - t_start, 1)},
        },
        "metrics": metrics,
        "delay_tests": delay_tests,
        # B - A: positive means handing the model the delayed channel helped.
        "arm_b_minus_a": b_minus_a,
        "floor": float(args.floor),
        "note": ("Delay tests are test-time only on the arm-B model. A drop counts "
                 "only when exceeds_floor is true; drop = B_auc - variant_auc. "
                 "auc None = unmeasurable (single class), never 0."),
        "complete": True,
    }
    if scores_outputs is not None:
        manifest["scores_outputs"] = scores_outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2))
    print(f"[minirocket] wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
