"""Score the network with its fitted equations substituted in.

Item A of ``docs/superpowers/specs/2026-08-12-spline-substitution-design.md``.

The white-box pipeline fits a closed form to each of the 25 most important edges
and every fold returns 25/25 accepted at R^2 0.9257-0.9999. That is a statement
about curve fitting. This runner replaces those edges' learned activations with
their closed forms at forward time and re-scores the network on held-out data,
which is what turns "each edge is well fit by an equation" into "the network is
these equations", or shows it is not.

PRE-COMMITTED CRITERIA, per fold, fixed before any number was seen:

  1. |AUC_substituted(k=25) - AUC_baseline(fold)| <= 0.013   (absolute fidelity)
  2. AUC_substituted(k=25) - AUC_permuted(k=25)   >  0.013   (matched control)

Both must hold. Criterion 1 alone cannot distinguish "the equations are right"
from "the network is robust to perturbing 25 edges". Criterion 2 alone does not
show the equations reproduce the network.

EVERY comparison is within a fold, against that fold's own unsubstituted
held-out AUC -- never the five-fold mean 0.908, whose 0.013 spread is used only
as the tolerance. Fold 2 sits at 0.8994, which is 0.0086 below the mean and
consumes two thirds of the tolerance on its own.

Arms: baseline; k in {1,5,10,15,20,25} x {correct, permuted} unclamped; k=25
clamped both arms. Fifteen held-out evaluations.

The guard is ``substitutions_are_active``, asserted before every arm. There is
no separate identity arm: one built from fits would differ from baseline by the
fit error and abort good runs while reporting a machinery failure.

Usage::

    python3 scripts/symbolic_substitution_phase3.py \\
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \\
        --features-csv data/metadata/features_datasets_121_125.csv \\
        --npz-root phase2_signals
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_heldout_phase3 as EH  # noqa: E402
import xai_phase3  # noqa: E402
from pri2m.xai import symbolic  # noqa: E402
from pri2m.xai.activations import collect_activations  # noqa: E402
from pri2m.xai.sampling import (  # noqa: E402
    batch_to_model_device,
    build_labeled_batch,
    window_config_from_preprocessing,
)
from pri2m.xai.semantics import edge_index_map  # noqa: E402
from pri2m.xai.substitution import (  # noqa: E402
    permute_fits,
    substitute_context,
    substitutions_are_active,
)

DEFAULT_TOL = 0.013
DEFAULT_KS = (1, 5, 10, 15, 20, 25)
DEFAULT_OUT = pathlib.Path("results/phase3/xai/symbolic_substitution")


def rank_like_symbolic_table(activations: dict, k: int,
                             allowed: set | None = None) -> list[tuple]:
    """The top ``k`` edges in exactly :func:`symbolic.symbolic_table`'s order.

    Importance descending, then ``(kan_name, out, in)`` ascending as a
    deterministic tie-break -- mirroring ``symbolic.py:449``.

    This exists because ``symbolic_table`` computes each edge's fit internally
    but returns only the display ``formula``, dropping ``params``/``lo``/``hi``
    that the substitution needs. Replicating the ranking here is preferable to
    changing code that produced published numbers; the tie-break is pinned by
    ``tests/test_symbolic_substitution_runner.py`` so the two cannot select
    different edges.

    Parameters
    ----------
    allowed:
        Optional ``{(kan_name, in_index)}`` restriction. ``None`` leaves the
        published global ranking untouched, which is why the five-fold result
        cannot move because this option exists. Passing a set restricts the
        candidates BEFORE ranking, so the top-k is the top-k *of that family*
        rather than whatever survives a filter applied afterwards.

    Raises
    ------
    ValueError
        If a restriction matches nothing. An empty selection substitutes nothing,
        which scores exactly baseline and would read as the strongest possible
        pass.
    """
    ranked: list[tuple[float, str, int, int]] = []
    for kan_name, stat in activations.items():
        imp = np.asarray(stat["importance"])
        out_n, in_n = imp.shape
        for o in range(out_n):
            for i in range(in_n):
                if allowed is not None and (kan_name, i) not in allowed:
                    continue
                ranked.append((float(imp[o, i]), kan_name, o, i))
    if allowed is not None and not ranked:
        raise ValueError(
            "no edges match the restriction; substituting nothing would score "
            "the baseline and read as a pass"
        )
    ranked.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    return [(kan, o, i) for _imp, kan, o, i in ranked[: int(k)]]


def edges_matching(semantics: dict, prefix: str) -> set:
    """``{(kan_name, in_index)}`` whose semantic name starts with ``prefix``.

    ``semantics`` is :func:`edge_index_map` output. A prefix test rather than a
    substring one, so a channel merely named like the family is not swept in.
    """
    out: set = set()
    for kan_name, entries in semantics.items():
        for i, entry in enumerate(entries):
            if str(entry.get("name", "")).startswith(prefix):
                out.add((kan_name, i))
    return out


def top_fits(model, activations: dict, k: int, *, r2_threshold: float = 0.9,
             simplicity_tol: float = 1e-3, allowed: set | None = None) -> list[dict]:
    """Fit closed forms to the top ``k`` edges, keeping ``params``/``lo``/``hi``.

    Same selection, same fit call and same R^2 gate as
    :func:`symbolic.symbolic_table`, but the returned rows carry the fit itself
    so it can be substituted rather than only printed.

    ``allowed`` restricts the candidates to one family of edges; ``None`` leaves
    the published global ranking untouched.
    """
    semantics = edge_index_map(model)
    rows: list[dict] = []
    for kan_name, o, i in rank_like_symbolic_table(activations, k, allowed=allowed):
        stat = activations[kan_name]
        lo = float(stat["visited_lo"][i])
        hi = float(stat["visited_hi"][i])
        fit = symbolic.fit_edge(stat["module"], o, i, lo, hi,
                                r2_threshold=r2_threshold,
                                simplicity_tol=simplicity_tol)
        entries = semantics.get(kan_name, [])
        block, name = "input", f"in{i}"
        if i < len(entries):
            block = entries[i].get("block", block)
            name = entries[i].get("name", name)
        accepted = bool(fit.get("accepted", True))
        rows.append({
            "kan": kan_name, "out": int(o), "in": int(i),
            "block": block, "name": name,
            "importance": float(np.asarray(stat["importance"])[o, i]),
            "formula": fit["formula"] if accepted else "no faithful closed form",
            "r2": float(fit["r2"]), "bic": float(fit["bic"]),
            "accepted": accepted, "fit": fit, "lo": lo, "hi": hi,
        })
    return rows


def build_substitutions(rows: list[dict], k: int) -> dict:
    """Top-``k`` accepted rows as a ``substitute_context`` subs dict.

    Rows whose closed form the R^2 gate rejected are skipped: there is no
    equation to substitute and inventing one would be the whole defect this
    experiment exists to avoid. Callers should report the resulting count via
    :func:`count_substituted` so a short arm is never silently short.
    """
    subs: dict = {}
    n = 0
    for r in rows:
        if n >= k:
            break
        if not r.get("accepted"):
            continue
        subs.setdefault(r["kan"], []).append({
            "out": int(r["out"]), "in": int(r["in"]),
            "fit": r["fit"], "lo": float(r["lo"]), "hi": float(r["hi"]),
        })
        n += 1
    return subs


def count_substituted(subs: dict) -> int:
    """How many edges a subs dict actually substitutes."""
    return sum(len(v) for v in subs.values())


def permute_subs(subs: dict, seed: int = 0) -> dict:
    """Derange the equations across every substituted edge -- the control.

    Permutes across the FLATTENED edge list rather than within each module, so
    a module holding a single edge still has its equation moved.
    """
    flat = [{**e, "_kan": kan} for kan, entries in subs.items() for e in entries]
    moved = permute_fits(flat, seed=seed)
    out: dict = {}
    for e in moved:
        kan = e.pop("_kan")
        out.setdefault(kan, []).append(e)
    return out


def verdict(baseline, substituted, permuted, tol: float) -> dict:
    """Both pre-committed criteria, per fold.

    Returns ``passed=None`` rather than ``False`` when an arm is missing, so a
    crashed arm can never read as a negative result.
    """
    if baseline is None or substituted is None or permuted is None:
        return {"passed": None, "absolute_ok": None, "control_ok": None,
                "reason": "an arm is missing"}
    absolute_ok = abs(substituted - baseline) <= tol
    control_ok = (substituted - permuted) > tol
    return {
        "passed": bool(absolute_ok and control_ok),
        "absolute_ok": bool(absolute_ok),
        "control_ok": bool(control_ok),
        "abs_gap": float(substituted - baseline),
        "control_gap": float(substituted - permuted),
        "tol": float(tol),
    }


def _score(checkpoint, args, preprocessing, model, tag, out_dir):
    """Held-out AUC for ``model``, via the existing held-out evaluator."""
    try:
        res = EH.run(
            checkpoint=str(checkpoint),
            features_csv=args.features_csv,
            npz_root=args.npz_root,
            out=str(out_dir / f"_eval_{tag}.json"),
            preprocessing=args.preprocessing,
            device=args.device,
            batch_size=args.batch_size,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 - an arm that dies must not be a pass
        return None, f"{type(exc).__name__}: {exc}"
    # eval_heldout_phase3.run nests its metrics: the AUC is at
    # result["metrics"]["auc"], NOT at the top level. Reading the top level
    # returns None for a run that in fact succeeded.
    auc = ((res or {}).get("metrics") or {}).get("auc")
    if auc is None:
        return None, f"evaluator returned no metrics.auc (keys: {sorted((res or {}))})"
    return float(auc), None


def _guarded_score(model, subs, probe, checkpoint, args, preprocessing, tag, out_dir,
                   *, clamp: bool):
    """Score one arm, refusing to report a number the substitution did not produce.

    ``clamp`` is passed explicitly rather than inferred from ``tag``. Deriving a
    mode flag from a string (or from a file existing) is how this project has
    produced clean-looking runs that silently did the wrong thing; the flag that
    controls the arithmetic must be the flag the caller set.
    """
    if not substitutions_are_active(model, subs, probe):
        return None, "substitution did not apply"
    with substitute_context(model, subs, clamp=clamp):
        return _score(checkpoint, args, preprocessing, model, tag, out_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features-csv", required=True,
                    help="Held-out features CSV (the 121-125 path).")
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--table", default="results/phase3/phase3_dataset.csv")
    ap.add_argument("--preprocessing", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=256,
                    help="Windows used to measure edge importance and fit edges.")
    ap.add_argument("--k-sweep", default=",".join(str(k) for k in DEFAULT_KS),
                    help="Comma-separated k values. A single k is not a result.")
    ap.add_argument("--r2-threshold", type=float, default=0.9,
                    help="Faithfulness gate forwarded to fit_edge, matching the "
                         "published symbolic table's default.")
    ap.add_argument("--restrict-to", default=None,
                    help="Only substitute edges whose semantic name starts with "
                         "this prefix, e.g. 'delay:' for the delayed channels. "
                         "Omitted, the published global top-k ranking is used.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help="Held-out fold spread; tolerance around THIS fold's "
                         "own baseline, never the five-fold mean.")
    ap.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--tag", default="fold2")
    args = ap.parse_args(argv)

    ks = [int(v) for v in args.k_sweep.split(",") if v.strip()]

    ckpt = pathlib.Path(args.checkpoint)
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}", file=sys.stderr)
        return 2
    pre_path = (pathlib.Path(args.preprocessing) if args.preprocessing
                else ckpt.parent.parent / "preprocessing.json")
    preprocessing = json.loads(pre_path.read_text()) if pre_path.exists() else None

    load_args = argparse.Namespace(input_size=None, hidden_size=None,
                                   grid_size=5, spline_order=3)
    model = xai_phase3._load_model(str(ckpt), load_args, preprocessing)
    model.eval()

    out_dir = pathlib.Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "symbolic_substitution.json"

    manifest = {
        "provenance": {
            "complete": False,
            "checkpoint": str(ckpt),
            "features_csv": args.features_csv,
            "k_sweep": ks,
            "n_windows": int(args.n_windows),
            "r2_threshold": float(args.r2_threshold),
            "seed": int(args.seed),
            "tol": float(args.tol),
            "tag": args.tag,
            # Recorded so a restricted run can never be mistaken for the
            # published unrestricted one when the two manifests sit side by side.
            "restrict_to": args.restrict_to,
        },
        "rows": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    baseline, err = _score(ckpt, args, preprocessing, model, "baseline", out_dir)
    if baseline is None:
        print(f"Baseline evaluation failed: {err}", file=sys.stderr)
        return 1
    manifest["baseline_auc"] = baseline
    print(f"baseline held-out AUC = {baseline:.4f}  (this fold's own, not the mean)")

    # cfg is REQUIRED: passing None reaches Phase3SequenceDataset and dies on
    # cfg.channels. Take the geometry from the checkpoint's own preprocessing so
    # the windows analysed match the ones the network was trained on.
    cfg = window_config_from_preprocessing(preprocessing)
    batch, _labels = build_labeled_batch(
        args.table, args.npz_root, preprocessing, args.n_windows,
        cfg, seed=args.seed,
    )
    # The baseline eval above already ran, and eval_heldout_phase3.run moves the
    # model to its device as a side effect. build_labeled_batch always returns
    # CPU tensors, so without this the next model(batch) dies on a device
    # mismatch. Read the device from the model, not from --device.
    batch = batch_to_model_device(batch, model)
    activations = collect_activations(model, batch)
    allowed = None
    if args.restrict_to:
        allowed = edges_matching(edge_index_map(model), args.restrict_to)
        # Say how many edges the family has BEFORE fitting. A restriction that
        # matches fewer edges than the k-sweep asks for silently caps every
        # larger arm at the same size, which would look like a plateau.
        print(f"restricted to {args.restrict_to!r}: {len(allowed)} candidate "
              f"input edges across {len({k for k, _ in allowed})} blocks")
        manifest["provenance"]["n_restricted_candidates"] = len(allowed)
    rows = top_fits(model, activations, max(ks), r2_threshold=args.r2_threshold,
                    allowed=allowed)
    n_rejected = sum(1 for r in rows if not r["accepted"])
    if allowed is not None and len(rows) < max(ks):
        print(f"NOTE: the family holds {len(rows)} fittable edges, fewer than the "
              f"largest k requested ({max(ks)}); arms above it are the same set.")
    manifest["edges"] = [{k: v for k, v in r.items() if k != "fit"} for r in rows]
    manifest["provenance"]["n_rejected_fits"] = n_rejected
    print(f"fitted {len(rows)} edges, {n_rejected} rejected by the R^2 gate")

    probe = batch

    def run_arm(k: int, clamped: bool):
        suffix = "_clamped" if clamped else ""
        subs = build_substitutions(rows, k)
        n_sub = count_substituted(subs)
        if n_sub < 2:
            return {"k": k, "clamped": clamped, "n_substituted": n_sub,
                    "correct_auc": None, "permuted_auc": None,
                    "error": "fewer than 2 accepted fits; no derangement exists"}
        correct, e1 = _guarded_score(model, subs, probe, ckpt, args, preprocessing,
                                     f"k{k}_correct{suffix}", out_dir, clamp=clamped)
        permuted, e2 = _guarded_score(model, permute_subs(subs, seed=args.seed),
                                      probe, ckpt, args, preprocessing,
                                      f"k{k}_permuted{suffix}", out_dir, clamp=clamped)
        return {"k": k, "clamped": clamped, "n_substituted": n_sub,
                "correct_auc": correct, "correct_error": e1,
                "permuted_auc": permuted, "permuted_error": e2,
                "verdict": verdict(baseline, correct, permuted, float(args.tol))}

    for k in ks:
        row = run_arm(k, clamped=False)
        manifest["rows"].append(row)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        c, p = row.get("correct_auc"), row.get("permuted_auc")
        print(f"  k={k:<3} n={row['n_substituted']:<3} "
              f"correct={'None' if c is None else format(c, '.4f')}  "
              f"permuted={'None' if p is None else format(p, '.4f')}")

    # Extrapolation: clamped only at the largest k, since the question concerns
    # the full substitution.
    row = run_arm(max(ks), clamped=True)
    manifest["rows"].append(row)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    c, p = row.get("correct_auc"), row.get("permuted_auc")
    print(f"  k={max(ks)} CLAMPED  "
          f"correct={'None' if c is None else format(c, '.4f')}  "
          f"permuted={'None' if p is None else format(p, '.4f')}")

    manifest["provenance"]["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2))

    written = json.loads(manifest_path.read_text())
    assert written["provenance"]["complete"] is True, "manifest not marked complete"
    assert written["rows"], "manifest has no rows"
    print(f"\nWrote {manifest_path} ({len(written['rows'])} arms)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
