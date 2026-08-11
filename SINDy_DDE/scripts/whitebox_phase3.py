#!/usr/bin/env python3
"""White-box analysis runner for a trained Phase 3 TKAN checkpoint.

Loads a trained ``TKANModel`` (eval-only, no retraining), samples a balanced set
of real windows from the Phase 2 NPZ tree, and runs the full white-box analysis
suite, writing every artifact plus a single ``whitebox_manifest.json`` that
carries provenance, the raw results, the artifact paths, any per-analysis errors,
and a flat ``report`` block of render-ready values for
``scripts/render_whitebox_report.py``.

The six analyses (each isolated so one failure does not sink the run):
* feature importance  -- channel / lag / side / permutation (which inputs matter)
* physics             -- delay ablation, tau-sweep, delayed-weight share, spectrum
* faithfulness        -- pruning curve, deletion curve, randomization sanity
* symbolic            -- the top edges as closed-form B-spline activations
* case studies        -- one confident chatter + one confident stable decision
* KAN-native          -- per-layer edge importance heatmap

Example::

    python scripts/whitebox_phase3.py \
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \
        --npz-root $PWD/phase2_signals --table results/phase3/phase3_dataset.csv \
        --output-dir results/phase3/xai/fold2 --n-windows 256
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

# Reuse the battle-tested checkpoint loaders from the existing XAI script.
import xai_phase3  # noqa: E402
from pri2m.data.phase3_sequence import WindowConfig  # noqa: E402
from pri2m.xai import (  # noqa: E402
    activations as act_mod,
    attribution,
    case_studies,
    faithfulness,
    feature_importance,
    physics,
    semantics,
    spline_atlas,
    symbolic,
)
from pri2m.xai.attribution import integrated_gradients  # noqa: E402
from pri2m.xai.sampling import (  # noqa: E402,F401 - compatibility re-export
    CHANNELS,
    build_labeled_batch,
    npz_exists as _npz_exists,
)


def _extra_scalars_list(model, batch):
    """Per-window side-feature dicts for IG (omega/tpf; tau flows via the tau arg)."""
    if not getattr(model, "side_input", False):
        return None
    n = batch["window"].shape[0]
    feats = [f for f in model.side_features if f != "tau"]
    return [{f: float(batch[f][i]) for f in feats} for i in range(n)]


# --------------------------------------------------------------------------- #
# Small plotting helper unique to the runner
# --------------------------------------------------------------------------- #
def _plot_importance_heatmaps(stats: dict, path: Path) -> str:
    """Per-KANLinear data-aware edge-importance heatmaps (out x in)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(stats.keys())
    n = max(1, len(names))
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    for ax, name in zip(flat[:n], names):
        imp = np.asarray(stats[name]["importance"])
        im = ax.imshow(imp, aspect="auto", cmap="viridis")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("input edge")
        ax.set_ylabel("output unit")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Data-aware edge importance per KANLinear", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# Report formatting helpers
# --------------------------------------------------------------------------- #
def _md_table(headers, rows) -> str:
    """A small markdown table from header names + row lists (strings)."""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([head, sep, body]) if rows else head + "\n" + sep


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


def _index_map_table(model) -> str | None:
    """Build the §3 semantic index-map table from the ACTUAL model structure.

    Collapses consecutive same-block input edges of the bottom encoder gate into
    index ranges, so the table is correct for any hidden width (not hardcoded).
    """
    from pri2m.xai.semantics import edge_index_map

    emap = edge_index_map(model, channel_names=CHANNELS)
    name = next((k for k in emap if k.startswith("encoder.layers.0.kan_z")), None)
    if not name:
        return None
    entries = emap[name]
    rows = []
    start = 0
    for i in range(1, len(entries) + 1):
        if i == len(entries) or entries[i]["block"] != entries[start]["block"]:
            lo, hi = entries[start]["index"], entries[i - 1]["index"]
            rng = f"`{lo}`" if lo == hi else f"`{lo}..{hi}`"
            names = [entries[j]["name"] for j in range(start, i)]
            sem = ", ".join(names) if len(names) <= 4 else f"{names[0]} .. {names[-1]}"
            rows.append([rng, entries[start]["block"], sem])
            start = i
    return _md_table(["index range", "block", "semantic"], rows)


def _git_sha(root) -> str | None:
    """Best-effort short git sha of the repo (None if unavailable)."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _now_date() -> str:
    """Current UTC date (YYYY-MM-DD) for the report header."""
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Temporal Saliency Rescaling -- factored out of main() so the (by far most
# expensive) TSR stage can run either inline or sharded across a SLURM array.
# scripts/merge_tsr_shards.py reassembles per-window entries via tsr_aggregate()
# into the identical results["tsr"] block a single un-sharded run would write.
# Every function below depends only on its explicit args (no closure over main's
# locals) so a shard worker and the full runner compute bit-identical entries.
# --------------------------------------------------------------------------- #
def tsr_select_indices(labels, tsr_windows: int) -> list[int]:
    """Deterministic, class-balanced TSR window indices (interleave chatter/stable).

    Depends only on ``labels`` -- itself seeded via ``build_labeled_batch`` -- so
    every shard task and the merge step agree on the identical global ordering.
    """
    labels = np.asarray(labels)
    chatter = np.flatnonzero(labels == 1).tolist()
    stable = np.flatnonzero(labels == 0).tolist()
    chosen: list[int] = []
    cap = max(0, int(tsr_windows))
    for pos in range(max(len(chatter), len(stable))):
        for group in (chatter, stable):
            if pos < len(group) and len(chosen) < cap:
                chosen.append(int(group[pos]))
    return chosen


def _tsr_channel_shares(values) -> dict:
    totals = np.abs(np.asarray(values, dtype=float)).sum(axis=0)
    denom = float(totals.sum())
    if denom <= np.finfo(float).eps:
        totals = np.ones(len(CHANNELS), dtype=float)
        denom = float(totals.sum())
    return {ch: float(totals[c] / denom) for c, ch in enumerate(CHANNELS)}


def tr_delay_statistics(time_relevance, tau: int) -> dict:
    """Peak-lag and delay-lock statistics for one window's time-relevance curve.

    ``tr_peak_lag`` is retained verbatim for back-compat, but it is a *global
    argmax of the relevance autocorrelation over lags* ``[2, N/2]``. When that
    autocorrelation decays monotonically -- no periodicity resolved at all --
    ``argmax`` returns index 0 and this emits **2**, the search floor. That reads
    like a measured lag and is not one: on the 2026-07-23/24 five-fold run, 550
    of 640 windows (85.9%) came back as exactly 2. ``tr_peak_at_floor`` states
    which happened, so the sentinel can never be averaged into a mean again.

    ``tr_lock_z`` is the argmax-free replacement, and the statistic the peak lag
    was only ever a proxy for: how far the autocorrelation **at lag tau** sits
    above the spread of autocorrelations at every other searched lag. It has no
    boundary to be pinned against, it tests the physical delay directly instead
    of asking where the maximum happens to fall, and it is ``None`` -- never a
    guess -- whenever tau is untestable (outside the searched range, or a
    degenerate/constant relevance curve). The nulls include the short lags, so a
    genuine lag-tau bump must out-rank the near-lag correlation of a smooth
    curve to register. ``z > 2`` is the decision rule for "locked".
    """
    tr = np.asarray(time_relevance, dtype=float).ravel()
    centered = tr - tr.mean()
    corr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    upper = min(len(centered) // 2, len(corr) - 1)

    if upper >= 2:
        searched = corr[2 : upper + 1]
        peak_lag = int(np.argmax(searched) + 2)
        at_floor = peak_lag == 2
    else:
        searched = np.empty(0, dtype=float)
        peak_lag = int(max(1, upper))
        at_floor = False

    lock_z = None
    tau_lag = int(tau)
    if searched.size >= 2 and 2 <= tau_lag <= upper:
        spread = float(searched.std())
        if spread > np.finfo(float).eps:
            lock_z = float((corr[tau_lag] - float(searched.mean())) / spread)

    return {
        "tr_peak_lag": peak_lag,
        "tr_peak_at_floor": bool(at_floor),
        "tr_lock_z": lock_z,
        "tr_n_lags": int(searched.size),
    }


def tsr_one_window(model, batch, esl, labels, i: int, steps: int,
                   tau_override: int | None = None,
                   feature_masking: bool = True) -> dict:
    """TSR for a single window index -> one ``per_window`` entry (order-free).

    ``tau_override`` substitutes a different delay for this window without
    touching the batch. It exists for the U12-B control (:func:`run_tsr_control`),
    which needs to ask whether the delay-lock follows whatever tau is injected --
    the signature of a construction artifact -- or only the true one.

    ``feature_masking=False`` is ~3x cheaper and leaves ``tr_lock_z`` /
    ``tr_peak_lag`` **bit-identical**: the flag only selects how step 2 scores the
    *feature* dimension, while both statistics are derived from step 1's
    ``time_relevance``, which is computed the same either way. It does change the
    ``channel_share_tsr`` values, so it is a real trade for the sweep and a free
    speedup for any lag-only question.
    """
    tau_raw = int(batch["tau"][i]) if tau_override is None else int(tau_override)
    res = attribution.temporal_saliency_rescaling(
        model,
        batch["window"][i],
        target="chatter_logit",
        tau=tau_raw,
        steps=int(steps),
        extra_scalars=(esl[i] if esl else None),
        feature_masking=bool(feature_masking),
    )
    return {
        "index": int(i),
        "label": int(labels[i]),
        "channel_share_tsr": _tsr_channel_shares(res["tsr"]),
        "channel_share_ig": _tsr_channel_shares(res["ig"]),
        **tr_delay_statistics(res["time_relevance"], tau_raw),
        "tau_raw": tau_raw,
    }


def tsr_aggregate(per_window: list[dict], steps: int) -> dict:
    """Reduce ``per_window`` entries into the ``results['tsr']`` block.

    Every reduction is order-invariant, so a sharded run whose windows are
    reassembled in canonical order yields the same block as an inline run.
    """
    def _mean_share(key):
        return {
            ch: float(np.mean([row[key][ch] for row in per_window]))
            for ch in CHANNELS
        }

    # A window "resolved" iff its peak was not pinned to the search floor. Rows
    # written before 2026-07-25 carry no such key; treat them as unresolved only
    # when they sit at the floor, so old manifests reduce sensibly rather than
    # silently claiming every window resolved.
    def _at_floor(row):
        if "tr_peak_at_floor" in row:
            return bool(row["tr_peak_at_floor"])
        return int(row.get("tr_peak_lag", 2)) == 2

    resolved = [r for r in per_window if not _at_floor(r)]
    locks = [r["tr_lock_z"] for r in per_window if r.get("tr_lock_z") is not None]

    return {
        "per_window": per_window,
        "n_windows": len(per_window),
        "steps": int(steps),
        "mean_channel_share_tsr": _mean_share("channel_share_tsr"),
        "mean_channel_share_ig": _mean_share("channel_share_ig"),
        # Back-compat: the mean that pools the floor sentinel with real peaks.
        # Kept so returned manifests stay readable -- NEVER quote it alone.
        "tr_peak_lag_mean": float(np.mean([r["tr_peak_lag"] for r in per_window])),
        "tau_raw_mean": float(np.mean([r["tau_raw"] for r in per_window])),
        # What the above must always be read beside.
        "n_resolved": len(resolved),
        "resolved_fraction": (
            float(len(resolved) / len(per_window)) if per_window else 0.0
        ),
        "tr_peak_lag_mean_resolved": (
            float(np.mean([r["tr_peak_lag"] for r in resolved])) if resolved else None
        ),
        # The argmax-free delay test.
        "n_lock_tested": len(locks),
        "lock_z_mean": float(np.mean(locks)) if locks else None,
        "lock_z_frac_above_2": (
            float(sum(1 for z in locks if z > 2.0) / len(locks)) if locks else None
        ),
    }


def tsr_control_verdict(trained_lock, untrained_lock, threshold: float = 0.8,
                        *, shuffled_lock=None, n_lock_tested=None,
                        min_windows: int = 8,
                        lock_z_threshold: float = 2.0) -> dict:
    """Did training produce the delay-lock, or did the input construction?

    The delayed channel is the signal shifted by tau, so a lag-tau bump in the
    relevance autocorrelation can arise with no learning at all. The test is an
    untrained network on the same construction: if random weights lock to tau
    nearly as hard as the trained ones, the lock is not evidence about the model.

    ``artifact_suspected`` is ``True`` when the untrained lock reaches
    ``threshold`` of the trained lock (default 80%), and **None when either side
    is unmeasurable** -- a missing measurement must never read as "cleared".

    The rule as pre-committed has three parts beyond that ratio, all of which
    guard the *clear*, because a clear is what would promote TSR to a fourth
    line of delay evidence (U12 section 4, where the gap between the rule as
    written and the rule as implemented is recorded):

    ``lock_z_threshold``
        the trained side must itself be a lock. A trained value indistinguishable
        from noise, or pointing the wrong way, cannot clear -- there is no effect
        to attribute to training. Returns UNDECIDED.
    ``min_windows``
        a definitive verdict needs evidence behind it. Below ``min_windows``
        testable windows the result is UNDECIDED in **both** directions, since a
        single window should not settle this either way.
    ``shuffled_lock``
        the trained net at permuted taus. A lock no stronger at the true tau
        than at someone else's is reading the smoothness of the attribution
        curve rather than the delay, so it cannot clear. Note this separates
        "this window's tau" from "any lag"; it does **not** separate learned
        from built-in, which only the untrained arm does.

    UNDECIDED carries ``undecided_reason`` so a caller never has to infer why.
    """
    out = {
        "trained_lock": trained_lock,
        "untrained_lock": untrained_lock,
        "shuffled_lock": shuffled_lock,
        "lock_gain_over_untrained": None,
        "threshold": float(threshold),
        "min_windows": int(min_windows),
        "lock_z_threshold": float(lock_z_threshold),
        "n_lock_tested": n_lock_tested,
        "tau_specific": None,
        "artifact_suspected": None,
        "undecided_reason": None,
    }
    if trained_lock is None or untrained_lock is None:
        out["undecided_reason"] = "unmeasurable_lock"
        return out

    trained_lock, untrained_lock = float(trained_lock), float(untrained_lock)
    out["trained_lock"], out["untrained_lock"] = trained_lock, untrained_lock
    out["lock_gain_over_untrained"] = trained_lock - untrained_lock
    if shuffled_lock is not None:
        out["shuffled_lock"] = float(shuffled_lock)
        # The same ratio the untrained arm is held to, rather than a bare ">".
        # A shuffled lock a hair below the trained one is not tau-specificity,
        # and reusing the ratio avoids inventing a second constant.
        out["tau_specific"] = bool(
            float(shuffled_lock) < float(threshold) * abs(trained_lock)
        )

    if n_lock_tested is None:
        # Applies to the clear only (see below): how much evidence stands behind
        # the numbers is itself a measurement, and a missing one must not read
        # as a pass -- the rule the lock arms above already follow.
        out["undecided_reason"] = "window_count_unknown"
    elif int(n_lock_tested) < int(min_windows):
        out["undecided_reason"] = "insufficient_windows"
        return out

    # Compare against |trained|: a trained lock at or below zero carries no
    # signal to attribute, so any untrained lock at all is disqualifying.
    ref = abs(trained_lock)
    suspected = untrained_lock >= float(threshold) * ref if ref > 0 else True

    # The gates below can only block a *clear*. An artifact call is the
    # conservative outcome and stands on the ratio alone.
    if not suspected:
        if out["undecided_reason"] == "window_count_unknown":
            return out
        if trained_lock < float(lock_z_threshold):
            out["undecided_reason"] = "trained_lock_not_significant"
            return out
        if out["tau_specific"] is False:
            out["undecided_reason"] = "not_tau_specific"
            return out

    # Reached only on an artifact call or a clear that passed every gate, so any
    # provisional reason recorded above does not apply to the outcome.
    out["undecided_reason"] = None
    out["artifact_suspected"] = bool(suspected)
    return out


def run_tsr_control(trained_model, untrained_model, batch, esl, labels,
                    indices, steps: int, seed: int = 0,
                    threshold: float = 0.8,
                    feature_masking: bool = True,
                    progress=None, on_condition=None) -> dict:
    """U12-B: the construction-artifact control for the TSR delay-lock.

    Three conditions on the same windows and the same input construction:

    ``trained_true_tau``
        the real checkpoint at each window's own tau -- the measurement.
    ``untrained_true_tau``
        a randomly-initialised net of the same architecture. **This is the
        decisive one.** If it locks too, the lock is a property of how the
        delayed channel is built, not of anything learned.
    ``trained_shuffled_tau``
        the real checkpoint with taus permuted across windows, which asks
        whether the lock simply follows whichever delay is injected.

    Deliberately cheap: pass a short ``indices`` list. Settling this needs a few
    dozen windows on one fold, not the 128-window five-fold sweep -- the sweep
    measured the effect, this only has to attribute it.

    ``progress(condition, done, total)`` fires after every scored window and
    ``on_condition(name, result_so_far)`` after every finished condition, so a
    caller can log and checkpoint. Both default to None (silent, unchanged).
    The decisive pair is scored FIRST, so a run killed inside the third
    condition still hands back a decidable verdict -- job 55098053 lost twelve
    hours to a wall with nothing written, and that must not repeat.
    """
    idx = [int(i) for i in indices]
    rng = np.random.default_rng(int(seed))
    true_taus = [int(batch["tau"][i]) for i in idx]

    # Permute taus across windows. A shuffle that returns the identity would
    # make the condition vacuous, so keep drawing (bounded) until it moves.
    shuffled = list(true_taus)
    if len(set(true_taus)) > 1:
        for _ in range(64):
            shuffled = list(rng.permutation(true_taus))
            if shuffled != true_taus:
                break

    # ORDER IS LOAD-BEARING: the decisive pair first, the supporting condition
    # last, so a wall-clock kill in the third still leaves a verdict on disk.
    conditions = {
        "trained_true_tau": (trained_model, [None] * len(idx)),
        "untrained_true_tau": (untrained_model, [None] * len(idx)),
        "trained_shuffled_tau": (trained_model, shuffled),
    }

    def _assemble(done: dict) -> dict:
        """Snapshot of the result as it stands. Verdict only once both sides in."""
        pair = ("trained_true_tau", "untrained_true_tau")
        verdict = None
        if all(k in done for k in pair):
            # The shuffled arm lands last and may be absent on a partial
            # snapshot; the rule treats a missing one as "not consulted"
            # rather than as a pass, so a checkpoint verdict can only be
            # more conservative than the final one.
            shuffled_done = done.get("trained_shuffled_tau")
            verdict = tsr_control_verdict(
                done["trained_true_tau"]["lock_z_mean"],
                done["untrained_true_tau"]["lock_z_mean"],
                threshold=threshold,
                shuffled_lock=(shuffled_done or {}).get("lock_z_mean"),
                n_lock_tested=done["trained_true_tau"].get("n_lock_tested"),
            )
        return {"conditions": dict(done), "verdict": verdict}

    out = {}
    for name, (model, overrides) in conditions.items():
        per_window = []
        for done, (i, ov) in enumerate(zip(idx, overrides), start=1):
            per_window.append(
                tsr_one_window(model, batch, esl, labels, i, steps, tau_override=ov,
                               feature_masking=feature_masking)
            )
            if progress is not None:
                progress(name, done, len(idx))
        out[name] = tsr_aggregate(per_window, steps)
        if on_condition is not None:
            on_condition(name, _assemble(out))

    return _assemble(out)


def _run_tsr_shard(args, model, batch, esl, labels, out_dir: Path) -> int:
    """SLURM-array worker: compute TSR for this shard's slice of the global window
    set and write a shard file for scripts/merge_tsr_shards.py. No other analysis
    runs and no manifest is written -- the fixed stages come from a separate
    ``--skip-tsr`` run over the same seeded batch.
    """
    nshards = max(1, int(args.tsr_nshards))
    shard = int(args.tsr_shard)
    if not 0 <= shard < nshards:
        raise ValueError(f"--tsr-shard {shard} out of range for --tsr-nshards {nshards}")
    chosen = tsr_select_indices(labels, args.tsr_windows)
    if not chosen:
        raise ValueError("TSR requires at least one sampled window")
    # Strided assignment: shard k owns global positions p with p % nshards == k.
    # Each entry carries its global position p so the merge restores exact order.
    mine = [(pos, i) for pos, i in enumerate(chosen) if pos % nshards == shard]
    per_window_indexed = [
        [pos, tsr_one_window(model, batch, esl, labels, i, args.tsr_steps)]
        for pos, i in mine
    ]
    shard_dir = out_dir / "tsr_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard_{shard:04d}_of_{nshards:04d}.json"
    payload = {
        "shard": shard,
        "nshards": nshards,
        "tsr_windows": int(args.tsr_windows),
        "n_selected": len(chosen),
        "steps": int(args.tsr_steps),
        "per_window_indexed": per_window_indexed,
    }
    shard_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"TSR shard {shard}/{nshards}: {len(per_window_indexed)} window(s) -> {shard_path}")
    return 0


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz-root", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--preprocessing", default=None)
    parser.add_argument("--output-dir", default="results/phase3/xai")
    parser.add_argument("--n-windows", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for the random window sample.")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--window-length", type=int, default=1024, help="Raw window length (match training).")
    parser.add_argument("--window-stride", type=int, default=512)
    parser.add_argument("--pool", type=int, default=4, help="Temporal pool factor (match training config).")
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--tsr-windows", type=int, default=4)
    parser.add_argument("--tsr-steps", type=int, default=8)
    # Sharded-TSR workflow (the TSR stage dominates wall time at high --tsr-windows;
    # split it across a SLURM array and merge with scripts/merge_tsr_shards.py).
    parser.add_argument("--skip-tsr", action="store_true",
                        help="Run every analysis EXCEPT TSR (results['tsr']=None). "
                             "The fixed-stage run of the sharded workflow.")
    parser.add_argument("--tsr-only", action="store_true",
                        help="Compute ONLY this shard of TSR windows and write a shard "
                             "file (no other analysis, no manifest).")
    parser.add_argument("--tsr-shard", type=int, default=0, help="This shard's 0-based index.")
    parser.add_argument("--tsr-nshards", type=int, default=1, help="Total number of TSR shards.")
    parser.add_argument("--prune-threshold", type=float, default=0.01)
    # xai_phase3._load_model reads these names off the args namespace.
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--spline-order", type=int, default=3)
    # Optional provenance the runner cannot infer (filled into the report header).
    parser.add_argument("--val-auc", type=float, default=None, help="Known validation AUC for this fold.")
    parser.add_argument("--run-id", default=None, help="Run id/hash (default: inferred from checkpoint path).")
    parser.add_argument("--tag", default=None, help="Human label, e.g. 'fold2' (default: output dir name).")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model + preprocessing (reuse xai_phase3 loaders) -------------- #
    pre_path = args.preprocessing or xai_phase3._find_preprocessing(args.checkpoint)
    preprocessing = json.loads(Path(pre_path).read_text(encoding="utf-8")) if pre_path else None
    model = xai_phase3._load_model(args.checkpoint, args, preprocessing)
    model.pool = int(args.pool)  # forward-time op; not in the state_dict
    model.eval()

    cfg = WindowConfig(length=args.window_length, stride=args.window_stride, channels=CHANNELS)
    batch, labels = build_labeled_batch(
        args.table, args.npz_root, preprocessing, args.n_windows, cfg, seed=args.sample_seed
    )
    n_windows = int(batch["window"].shape[0])
    windows = batch["window"].detach().cpu().numpy()  # [N, L, C]
    taus = batch["tau"].detach().cpu().numpy().tolist()
    esl = _extra_scalars_list(model, batch)
    tau0 = int(np.median(np.asarray(taus)))
    fs0 = float(batch["fs"][0]) if "fs" in batch else 10_000.0
    tpf0 = float(batch["tpf"][0]) if "tpf" in batch else fs0 / max(1, tau0)

    # SLURM-array worker: compute one TSR shard and exit before any other stage.
    if args.tsr_only:
        return _run_tsr_shard(args, model, batch, esl, labels, out_dir)

    results: dict = {}
    artifacts: dict = {}
    errors: dict = {}

    def _safe(name, fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - record + continue; missing -> N/A in report
            errors[name] = traceback.format_exc()
            print(f"WARNING: analysis {name!r} failed:\n{errors[name]}", file=sys.stderr)
            return None

    # ---- Shared foundation: data-aware activations -------------------------- #
    stats = _safe("activations", lambda: act_mod.collect_activations(model, batch))
    sem_map = semantics.edge_index_map(model, channel_names=CHANNELS)

    # ---- Feature importance ------------------------------------------------- #
    def _feature():
        ca = feature_importance.channel_attribution(
            model, windows, taus, target="chatter_logit", steps=args.ig_steps,
            extra_scalars_list=esl,
        )
        perm = feature_importance.permutation_importance(
            model, batch, labels,
            side_features=tuple(f for f in getattr(model, "side_features", ()) if f != "tau"),
        )
        artifacts["channel_importance"] = feature_importance.plot_channel_importance(
            ca, str(out_dir / "channel_importance.png")
        )
        artifacts["lag_profile"] = feature_importance.plot_lag_profile(
            ca["per_lag"], str(out_dir / "lag_profile.png")
        )
        return {
            "per_channel": [float(v) for v in ca["per_channel"]],
            "per_lag": [float(v) for v in ca["per_lag"]],
            "baseline_auc": float(perm["baseline_auc"]),
            "channel_drop": {int(k): float(v) for k, v in perm["channel_drop"].items()},
            "side_drop": {str(k): float(v) for k, v in perm["side_drop"].items()},
        }
    results["feature_importance"] = _safe("feature_importance", _feature)

    # ---- Physics ------------------------------------------------------------ #
    results["delay_ablation"] = _safe(
        "delay_ablation", lambda: physics.delay_ablation(model, batch, labels)
    )
    def _sweep():
        # Per-window-relative sweep: impose x0.5 / x1 / x2 of EACH window's own
        # tau = round(fs / tpf), not one global delay (which blurs the peak when
        # the sample spans different operating points). fs/tpf flow per-window
        # from the batch; the no-delay baseline is measured separately.
        res = physics.tau_sweep_relative(model, batch, labels, factors=(0.5, 1.0, 2.0))
        artifacts["tau_sweep"] = physics.plot_tau_sweep(res, str(out_dir / "tau_sweep.png"))
        return res
    results["tau_sweep"] = _safe("tau_sweep", _sweep)
    results["weight_share"] = _safe(
        "weight_share", lambda: physics.delayed_weight_share(model, stats) if stats else None
    )
    def _spectrum():
        res = physics.attribution_spectrum(
            model, windows[0], taus[0], fs0, tpf0, target="chatter_logit",
            steps=args.ig_steps, extra_scalars=(esl[0] if esl else None),
        )
        artifacts["attr_spectrum"] = physics.plot_attr_spectrum(res, str(out_dir / "attr_spectrum.png"))
        return {
            "tpf": float(res["tpf"]),
            "tpf_band_energy": float(res["tpf_band_energy"]),
            "total_energy": float(res["total_energy"]),
        }
    results["attribution_spectrum"] = _safe("attribution_spectrum", _spectrum)

    # ---- Temporal Saliency Rescaling --------------------------------------- #
    # The single most expensive stage; --skip-tsr runs the fixed-stage half of the
    # sharded workflow, with TSR reassembled later by scripts/merge_tsr_shards.py.
    def _tsr():
        chosen = tsr_select_indices(labels, args.tsr_windows)
        if not chosen:
            raise ValueError("TSR requires at least one sampled window")
        per_window = [
            tsr_one_window(model, batch, esl, labels, i, args.tsr_steps) for i in chosen
        ]
        return tsr_aggregate(per_window, args.tsr_steps)

    results["tsr"] = None if args.skip_tsr else _safe("tsr", _tsr)

    # ---- Faithfulness ------------------------------------------------------- #
    def _faith():
        prune = faithfulness.pruning_curve(model, batch, labels, stats)
        artifacts["pruning_curve"] = faithfulness.plot_pruning_curve(prune, str(out_dir / "pruning_curve.png"))
        attr0 = integrated_gradients(
            model, windows[0], target="chatter_logit", tau=taus[0], steps=args.ig_steps,
            extra_scalars=(esl[0] if esl else None),
        )
        deletion = faithfulness.deletion_curve(
            model, windows[0], attr0, taus[0], target="chatter_logit",
            extra_scalars=(esl[0] if esl else None),
        )
        artifacts["deletion_curve"] = faithfulness.plot_deletion_curve(deletion, str(out_dir / "deletion_curve.png"))
        rand = faithfulness.randomization_sanity(
            model, windows[0], taus[0], steps=args.ig_steps, target="b_over_blim",
            extra_scalars=(esl[0] if esl else None),
        )
        return {
            "prune_fraction_kept": [float(x) for x in prune["fraction_kept"]],
            "prune_auc": [float(x) for x in prune["auc"]],
            "deletion_fraction": [float(x) for x in deletion["fraction_deleted"]],
            "deletion_pred": [float(x) for x in deletion["pred"]],
            "randomization_cosine": float(rand["cosine"]),
        }
    results["faithfulness"] = _safe("faithfulness", _faith) if stats else None

    # ---- Symbolic ----------------------------------------------------------- #
    def _symbolic():
        rows = symbolic.symbolic_table(model, stats, top_k=args.top_k)
        artifacts["symbolic_csv"] = symbolic.write_symbolic_csv(rows, str(out_dir / "symbolic_edges.csv"))
        artifacts["top_edges"] = symbolic.plot_top_edges(model, stats, rows, str(out_dir / "top_edges.png"))
        return rows
    results["symbolic_top"] = _safe("symbolic", _symbolic) if stats else None

    # ---- Named spline-edge atlas ------------------------------------------- #
    def _atlas():
        rows = spline_atlas.plot_spline_atlas(
            model,
            stats,
            sem_map,
            out_dir / "spline_atlas.png",
            k=int(args.top_k) if args.top_k else 16,
        )
        artifacts["spline_atlas"] = str(out_dir / "spline_atlas.png")
        agg = spline_atlas.feature_importance_bars(stats, sem_map)
        artifacts["feature_bars"] = spline_atlas.plot_feature_importance_bars(
            agg, out_dir / "feature_bars.png"
        )
        artifacts["edge_importance_named"] = spline_atlas.plot_feature_named_heatmap(
            stats, sem_map, out_dir / "edge_importance_named.png"
        )
        return rows

    results["spline_atlas"] = _safe("spline_atlas", _atlas) if stats else None

    # ---- Case studies ------------------------------------------------------- #
    def _cases():
        idx = case_studies.select_cases(model, batch, labels)
        out = {}
        for which in ("chatter", "stable"):
            i = idx[which]
            case = case_studies.explain_case(
                model, windows[i], taus[i], int(labels[i]),
                extra_scalars=(esl[i] if esl else None), top_k=8,
            )
            artifacts[f"case_{which}"] = case_studies.plot_case(
                case, windows[i], str(out_dir / f"case_{which}.png")
            )
            out[which] = {
                "index": int(i),
                "chatter_prob": float(case["chatter_prob"]),
                "b_over_blim": float(case["b_over_blim"]),
                "per_channel_ig": [float(v) for v in case["per_channel_ig"]],
                "why": {k: (v if isinstance(v, (str, bool)) else float(v)) for k, v in case["why"].items()},
            }
        return out
    results["cases"] = _safe("cases", _cases)

    # ---- KAN-native importance heatmap ------------------------------------- #
    if stats:
        _safe("edge_heatmap", lambda: artifacts.__setitem__(
            "edge_importance_heatmap", _plot_importance_heatmaps(stats, out_dir / "edge_importance.png")
        ))

    # ---- Provenance + report block ----------------------------------------- #
    n_edges = int(sum(int(np.asarray(s["importance"]).size) for s in stats.values())) if stats else 0
    run_id = args.run_id or Path(args.checkpoint).resolve().parent.parent.name
    tag = args.tag or out_dir.name
    provenance = {
        "checkpoint": str(args.checkpoint),
        "preprocessing": str(pre_path) if pre_path else None,
        "run_id": run_id,
        "tag": tag,
        "val_auc": args.val_auc,
        "rendered_date": _now_date(),
        "git_sha": _git_sha(_ROOT),
        "n_windows": n_windows,
        "n_channels": int(model.input_size),
        "hidden_size": int(model.hidden_size),
        "num_layers": int(model.encoder.num_layers),
        "grid_size": int(args.grid_size),
        "spline_order": int(args.spline_order),
        "side_input": bool(getattr(model, "side_input", False)),
        "side_features": ", ".join(getattr(model, "side_features", ())) or "none",
        "pool": int(args.pool),
        "tau_samples": tau0,
        "fs": fs0,
        "tpf": tpf0,
        "n_edges_total": n_edges,
    }

    report = _build_report(provenance, results, artifacts)
    idx_table = _safe("index_map_table", lambda: _index_map_table(model))
    if idx_table:
        report["index_map_table"] = idx_table

    manifest = {
        "provenance": provenance,
        "results": results,
        "artifacts": artifacts,
        "errors": errors,
        "report": report,
    }
    manifest_path = out_dir / "whitebox_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    print(f"White-box manifest -> {manifest_path}")
    if errors:
        print(f"WARNING: {len(errors)} analysis/analyses failed: {sorted(errors)}", file=sys.stderr)
    return 0


def _build_report(provenance: dict, results: dict, artifacts: dict) -> dict:
    """Flatten provenance + results + artifact paths into render-ready strings.

    Every value is a str / number / bool the renderer can drop straight into the
    template. Anything genuinely absent is simply omitted (the renderer writes
    ``N/A (not run)``), so a partial run never invents a number.
    """
    rep: dict = {}
    rep.update({k: v for k, v in provenance.items() if v is not None})

    fi = results.get("feature_importance")
    if fi:
        ch = fi["per_channel"]
        top_i = int(np.argmax(ch)) if ch else 0
        rep["top_channel"] = CHANNELS[top_i] if top_i < len(CHANNELS) else f"ch{top_i}"
        rep["chan_importance_table"] = _md_table(
            ["channel", "mean |IG|", "AUC drop (permute)"],
            [[CHANNELS[c] if c < len(CHANNELS) else f"ch{c}", _fmt(ch[c]),
              _fmt(fi["channel_drop"].get(c, float("nan")))] for c in range(len(ch))],
        )
        if fi["side_drop"]:
            rep["side_importance_table"] = _md_table(
                ["side feature", "AUC drop (permute)"],
                [[k, _fmt(v)] for k, v in fi["side_drop"].items()],
            )
        lag = fi["per_lag"]
        if lag:
            rep["lag_peak_samples"] = int(np.argmax(lag))

    da = results.get("delay_ablation")
    if da:
        rep["delay_auc_on"] = _fmt(da["auc_on"])
        rep["delay_auc_off"] = _fmt(da["auc_off"])
        rep["delay_auc_drop"] = _fmt(da["auc_drop"])

    sw = results.get("tau_sweep")
    if sw and "factors" in sw:
        rows = [[f"{float(f):g}x", _fmt(a)] for f, a in zip(sw["factors"], sw["auc"])]
        if sw.get("baseline_auc") is not None:
            rows.append(["no-delay", _fmt(sw["baseline_auc"])])
        rep["tau_sweep_table"] = _md_table(["delay (x own tau)", "chatter AUC"], rows)
        # Winning multiple of each window's OWN tau (relative), not an absolute
        # sample count; x1 is the physical tooth-pass delay.
        rep["tau_argmax"] = f"{float(sw['factors'][int(np.argmax(sw['auc']))]):g}x"
        rep["tau_physical"] = provenance.get("tau_samples")

    ws = results.get("weight_share")
    if ws and "overall" in ws:
        o = ws["overall"]
        rep["weight_share_xdelay"] = _fmt(o.get("x_delay"))
        rep["weight_share_xt"] = _fmt(o.get("x_t"))
        rep["weight_share_hprev"] = _fmt(o.get("h_prev"))

    sp = results.get("attribution_spectrum")
    if sp and sp.get("total_energy"):
        frac = sp["tpf_band_energy"] / sp["total_energy"] if sp["total_energy"] else float("nan")
        rep["tpf_band_energy_frac"] = _fmt(frac)

    sym = results.get("symbolic_top")
    if sym:
        rep["n_symbolic"] = len(sym)
        n_lin = sum(1 for r in sym if r.get("formula", "").find("x^") < 0 and "tanh" not in r.get("formula", "")
                    and "sin" not in r.get("formula", "") and "|x|" not in r.get("formula", "")
                    and "max(x,0)" not in r.get("formula", ""))
        rep["frac_linear"] = f"{n_lin}/{len(sym)}"
        rep["frac_nonlinear"] = f"{len(sym) - n_lin}/{len(sym)}"
        rep["symbolic_table"] = _md_table(
            ["edge", "block", "name", "importance", "formula", "R2"],
            [[f"{r['kan'].split('.')[-1]}({r['out']},{r['in']})", r["block"], r["name"],
              _fmt(r["importance"]), r["formula"], _fmt(r["r2"])] for r in sym[:12]],
        )

    atlas = results.get("spline_atlas")
    if atlas:
        rep["atlas_table"] = _md_table(
            ["gate", "feature", "unit", "importance", "formula", "R2"],
            [[r["kan"].split(".")[-1], r["feature"], f"u{r['out']}",
              _fmt(r["importance"]), r.get("formula") or "not accepted", _fmt(r.get("r2"))]
             for r in atlas[:8]],
        )
        rep["n_atlas_delay_edges"] = sum(bool(r.get("is_delay")) for r in atlas)
        rep["n_atlas_side_edges"] = sum(bool(r.get("is_side")) for r in atlas)

    tsr = results.get("tsr")
    if tsr:
        rep["tsr_table"] = _md_table(
            ["window", "label", "top TSR channel", "top IG channel",
             "TR peak lag", "resolved", "tau"],
            [[r.get("index", i), r["label"],
              max(r["channel_share_tsr"], key=r["channel_share_tsr"].get),
              max(r["channel_share_ig"], key=r["channel_share_ig"].get),
              r["tr_peak_lag"],
              "no" if r.get("tr_peak_at_floor", r["tr_peak_lag"] == 2) else "yes",
              r["tau_raw"]]
             for i, r in enumerate(tsr["per_window"])],
        )
        # The mean SIGNED change is identically zero and always was: both share
        # vectors sum to 1, so the per-channel deltas cancel by construction.
        # Report the mean ABSOLUTE shift, which is what "how far did the
        # rescaling move the attribution" actually asks.
        rep["tsr_delay_share_gain"] = _fmt(np.mean([
            abs(tsr["mean_channel_share_tsr"][ch] - tsr["mean_channel_share_ig"][ch])
            for ch in CHANNELS
        ]))
        rep["tsr_windows"] = int(tsr["n_windows"])
        rep["tsr_steps"] = int(tsr["steps"])

        # A peak-lag number must never appear without the resolved fraction
        # beside it -- the argmax returns its search floor when no peak exists.
        n_res = tsr.get("n_resolved")
        rep["tsr_resolved"] = (
            f"{n_res} of {tsr['n_windows']}" if n_res is not None else "not recorded"
        )
        rep["tsr_resolved_pct"] = (
            _fmt(100.0 * tsr["resolved_fraction"])
            if tsr.get("resolved_fraction") is not None else "n/a"
        )
        mean_res = tsr.get("tr_peak_lag_mean_resolved")
        rep["tsr_peak_lag_resolved"] = (
            _fmt(mean_res) if mean_res is not None else "no window resolved a peak"
        )
        lock_mean = tsr.get("lock_z_mean")
        rep["tsr_lock_z_mean"] = _fmt(lock_mean) if lock_mean is not None else "n/a"
        rep["tsr_lock_tested"] = int(tsr.get("n_lock_tested") or 0)
        lock_frac = tsr.get("lock_z_frac_above_2")
        rep["tsr_lock_pct"] = _fmt(100.0 * lock_frac) if lock_frac is not None else "n/a"

    fa = results.get("faithfulness")
    if fa:
        rep["randomization_cosine"] = _fmt(fa["randomization_cosine"])
        fk, au = fa["prune_fraction_kept"], fa["prune_auc"]
        rep["prune_table"] = _md_table(
            ["fraction kept", "chatter AUC"], [[_fmt(f), _fmt(a)] for f, a in zip(fk, au)]
        )
        if au:
            rep["prune_auc_full"] = _fmt(au[-1])
            rep["prune_auc_5pct"] = _fmt(au[0])
        dp = fa["deletion_pred"]
        if dp:
            rep["deletion_drop"] = _fmt(dp[0] - dp[-1])

    cs = results.get("cases")
    if cs:
        for which in ("chatter", "stable"):
            if which in cs:
                c = cs[which]
                rep[f"case_{which}"] = (
                    f"index {c['index']}: predicted **{c['why']['predicted']}** "
                    f"(chatter_prob {_fmt(c['chatter_prob'])}, b/blim {_fmt(c['b_over_blim'])}); "
                    f"top channel {c['why'].get('top_channel')}, "
                    f"leans on delay: {c['why'].get('leans_on_delay')}"
                )

    # Auto-generated caveats — surface honesty issues the numbers imply.
    caveats: list[str] = []
    base_auc = (fi or {}).get("baseline_auc")
    if base_auc is not None and base_auc > 0.99:
        caveats.append(
            f"- **Sample AUC is at the ceiling ({_fmt(base_auc)})** on these "
            f"{provenance.get('n_windows')} windows, so the *absolute* AUCs here are NOT the "
            f"model's performance (val/held-out are the honest numbers). Only the *deltas* "
            f"(delay-ablation drop, τ-sweep shape, pruning curve) are meaningful; the "
            f"side-feature permutation drop (≈0) is **inconclusive** at this ceiling. "
            f"Re-run with a larger/more diverse random sample (`--n-windows`, `--sample-seed`) "
            f"for un-ceilinged channel/side importances."
        )
    if da and da.get("auc_drop", 0) > 0.1:
        caveats.append(
            f"- The delay-ablation is the strongest physics result: zeroing the τ-delayed "
            f"input drops chatter AUC by {_fmt(da['auc_drop'])}"
            + (f" and raises severity MAE from {_fmt(da.get('mae_on'))} to {_fmt(da.get('mae_off'))}"
               if da.get("mae_off") is not None else "")
            + " — the regenerative term is load-bearing."
        )
    if sp and sp.get("total_energy"):
        frac = sp["tpf_band_energy"] / sp["total_energy"] if sp["total_energy"] else 0.0
        if frac < 0.05:
            caveats.append(
                f"- The attribution spectrum puts only {_fmt(frac)} of its energy in the "
                f"tooth-pass bin — a single-window, single-bin measure that likely misses the "
                f"chatter tone (near the natural frequency) and its sidebands; treat the "
                f"spectral fingerprint as **inconclusive**, not negative."
            )
    rep["caveats_list"] = "\n".join(caveats)

    # Artifact figure paths (basename so the report sits next to them).
    for key, dst in (
        ("fig_channel_importance", "channel_importance"),
        ("fig_lag_profile", "lag_profile"),
        ("fig_tau_sweep", "tau_sweep"),
        ("fig_attr_spectrum", "attr_spectrum"),
        ("fig_top_edges", "top_edges"),
        ("fig_pruning_curve", "pruning_curve"),
        ("fig_deletion_curve", "deletion_curve"),
        ("fig_case_chatter", "case_chatter"),
        ("fig_case_stable", "case_stable"),
        ("fig_edge_activations", "edge_importance_heatmap"),
        ("fig_spline_atlas", "spline_atlas"),
        ("fig_feature_bars", "feature_bars"),
        ("fig_edge_importance_named", "edge_importance_named"),
    ):
        if dst in artifacts:
            rep[key] = Path(artifacts[dst]).name

    return rep


if __name__ == "__main__":
    raise SystemExit(main())
