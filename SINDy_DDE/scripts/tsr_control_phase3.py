#!/usr/bin/env python3
"""U12-B: is the TSR delay-lock learned, or an artifact of how the input is built?

The 2026-07-23/24 sharded run found that when TSR resolves a peak at all, the peak
lands on that window's own tau far beyond chance (34% exact match, p = 0.00005).
That is only evidence about the *model* if it is not a by-product of the input
construction: the delayed channel is literally the signal shifted by tau, so a
lag-tau bump in the relevance autocorrelation could appear with no learning at all.

This runner settles it with three conditions on the same windows:

  trained_true_tau      the real checkpoint at each window's own tau
  untrained_true_tau    the SAME architecture with re-initialised weights  <- decisive
  trained_shuffled_tau  the real checkpoint with taus permuted across windows

If the untrained network locks to tau nearly as hard as the trained one, the lock
is a property of the construction and the attribution view of the delay is dead --
say so and let the three interventions carry the claim alone. If it does not, TSR
becomes a fourth independent line of delay evidence.

Deliberately cheap. This does not need the 128-window five-fold sweep that cost
two days and 27 timed-out shards: the sweep measured the effect, this only has to
attribute it. A few dozen windows on one fold is enough.

    python scripts/tsr_control_phase3.py \
        --checkpoint results/phase3/runs/<run>/checkpoints/best_auc.pt \
        --npz-root phase2_signals --table results/phase3/phase3_dataset.csv \
        --output-dir results/phase3/xai/tsr_control/fold2 \
        --n-windows 64 --tsr-windows 24 --tsr-steps 8
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import whitebox_phase3 as W  # noqa: E402
import xai_phase3  # noqa: E402
from pri2m.data.phase3_sequence import WindowConfig  # noqa: E402
from pri2m.xai.sampling import CHANNELS, build_labeled_batch  # noqa: E402


def reinitialised_copy(model, seed: int = 0):
    """Same architecture, same shapes, freshly random weights.

    Deep-copying the loaded model and re-initialising is deliberate: it cannot
    drift from the trained model's architecture the way re-constructing from
    guessed constructor args could, and the control is only valid if the two
    networks differ in nothing but their weights.
    """
    torch.manual_seed(int(seed))
    fresh = copy.deepcopy(model)
    n_reset = 0
    for module in fresh.modules():
        if module is fresh:
            continue
        if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
            module.reset_parameters()
            n_reset += 1
    if n_reset == 0:
        raise RuntimeError(
            "no submodule exposed reset_parameters(); refusing to run a control "
            "whose 'untrained' network may still carry trained weights"
        )
    fresh.eval()
    return fresh, n_reset


def _assert_actually_untrained(trained, untrained) -> float:
    """Guard the one way this control can silently become meaningless."""
    diffs, total = [], 0
    tp = dict(trained.named_parameters())
    for name, p in untrained.named_parameters():
        if name in tp and tp[name].shape == p.shape:
            total += 1
            diffs.append(float((tp[name] - p).abs().mean()))
    if not diffs:
        raise RuntimeError("could not compare parameters between the two models")
    mean_diff = float(np.mean(diffs))
    if mean_diff <= 1e-8:
        raise RuntimeError(
            f"re-initialisation did not change the weights (mean |diff| {mean_diff:.2e}); "
            "the 'untrained' condition would be identical to the trained one"
        )
    return mean_diff


def _provenance(*, checkpoint, run_id, tag, n_sampled, n_scored, tsr_steps,
                sample_seed, shuffle_seed, init_seed, n_reset,
                mean_weight_diff) -> dict:
    """Everything needed to trace the run back to its inputs.

    Extracted and tested because assembling it is where job 55405355 died: it
    called `W._git_sha()` with no argument while every other runner passes a
    root. Nothing here may raise -- provenance is bookkeeping, and it must never
    be able to destroy a completed analysis.
    """
    return {
        "checkpoint": str(checkpoint),
        "run_id": run_id,
        "tag": tag,
        "n_windows_sampled": int(n_sampled),
        "n_windows_scored": int(n_scored),
        "tsr_steps": int(tsr_steps),
        "sample_seed": int(sample_seed),
        "shuffle_seed": int(shuffle_seed),
        "init_seed": int(init_seed),
        "n_submodules_reinitialised": int(n_reset),
        "mean_weight_diff": float(mean_weight_diff),
        "rendered_date": W._now_date(),
        "git_sha": W._git_sha(W._ROOT),
    }


def _manifest(result: dict, provenance: dict, *, complete: bool) -> dict:
    """Wrap a control result for writing.

    ``complete`` is what stops a run killed at the wall from being read as a
    finished one -- the 2026-07-26 timeout wrote nothing at all, and a partial
    file that looked final would be worse than nothing.
    """
    return {
        "provenance": {**provenance, "complete": bool(complete)},
        "control": result,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--npz-root", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--preprocessing", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--tsr-windows", type=int, default=24,
                   help="Windows actually scored. A few dozen settles this.")
    p.add_argument("--tsr-steps", type=int, default=8)
    p.add_argument("--window-length", type=int, default=1024)
    p.add_argument("--window-stride", type=int, default=512)
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.8,
                   help="Untrained lock >= threshold x trained lock => artifact.")
    p.add_argument("--feature-masking", action="store_true",
                   help="Paper-faithful TSR step 2. ~3x SLOWER and changes nothing "
                        "this control reads: tr_lock_z comes from step 1's "
                        "time_relevance, which is identical either way (pinned by "
                        "tests/test_tsr_control.py). Off by default for that reason.")
    # xai_phase3._load_model reads these off the namespace.
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--run-id", default=None)
    p.add_argument("--tag", default=None)
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_path = args.preprocessing or xai_phase3._find_preprocessing(args.checkpoint)
    preprocessing = json.loads(Path(pre_path).read_text(encoding="utf-8")) if pre_path else None
    trained = xai_phase3._load_model(args.checkpoint, args, preprocessing)
    trained.pool = int(args.pool)
    trained.eval()

    untrained, n_reset = reinitialised_copy(trained, seed=args.init_seed)
    untrained.pool = int(args.pool)
    mean_weight_diff = _assert_actually_untrained(trained, untrained)

    cfg = WindowConfig(length=args.window_length, stride=args.window_stride,
                       channels=CHANNELS)
    batch, labels = build_labeled_batch(
        args.table, args.npz_root, preprocessing, args.n_windows, cfg,
        seed=args.sample_seed,
    )
    esl = W._extra_scalars_list(trained, batch)
    indices = W.tsr_select_indices(labels, args.tsr_windows)

    print(f"=== TSR construction-artifact control (U12-B) ===")
    print(f"windows scored: {len(indices)}   tsr_steps: {args.tsr_steps}")
    print(f"re-initialised {n_reset} submodules; mean |w_trained - w_untrained| = "
          f"{mean_weight_diff:.4e}")

    print(f"feature_masking: {bool(args.feature_masking)} "
          f"({'paper-faithful, ~3x slower' if args.feature_masking else 'cheap; lag stats identical'})")

    provenance = _provenance(
        checkpoint=args.checkpoint, run_id=args.run_id,
        tag=args.tag or out_dir.name,
        n_sampled=batch["window"].shape[0], n_scored=len(indices),
        tsr_steps=args.tsr_steps, sample_seed=args.sample_seed,
        shuffle_seed=args.shuffle_seed, init_seed=args.init_seed,
        n_reset=n_reset, mean_weight_diff=mean_weight_diff,
    )
    path = out_dir / "tsr_control_manifest.json"

    def _write(res: dict, *, complete: bool) -> None:
        path.write_text(
            json.dumps(_manifest(res, provenance, complete=complete),
                       indent=2, default=float),
            encoding="utf-8",
        )

    n_cond = 3
    total_windows = n_cond * len(indices)

    def _progress(cond: str, done: int, total: int) -> None:
        seen = _progress.offset + done
        print(f"  [{seen:>4}/{total_windows}] {cond} window {done}/{total}",
              flush=True)
    _progress.offset = 0

    def _checkpoint(cond: str, res: dict) -> None:
        _progress.offset += len(indices)
        _write(res, complete=False)
        v = res["verdict"]
        note = "" if v is None else f"  (verdict now decidable: {v['artifact_suspected']})"
        print(f"  -> {cond} done; partial manifest written{note}", flush=True)

    print(f"scoring {total_windows} window-evaluations "
          f"({n_cond} conditions x {len(indices)} windows)", flush=True)

    result = W.run_tsr_control(
        trained, untrained, batch, esl, labels,
        indices=indices, steps=args.tsr_steps,
        seed=args.shuffle_seed, threshold=args.threshold,
        feature_masking=bool(args.feature_masking),
        progress=_progress, on_condition=_checkpoint,
    )

    _write(result, complete=True)

    v = result["verdict"]
    print(f"\nmanifest -> {path}")
    for name, cond in result["conditions"].items():
        print(f"  {name:22s} resolved {cond['n_resolved']}/{cond['n_windows']}  "
              f"lock_z_mean {cond['lock_z_mean']}")
    print(f"\ntrained lock   {v['trained_lock']}")
    print(f"untrained lock {v['untrained_lock']}   (threshold {v['threshold']})")
    if v["artifact_suspected"] is None:
        print("VERDICT: UNDECIDED -- a lock was not measurable on one side. "
              "This is NOT a pass; do not claim TSR either way.")
    elif v["artifact_suspected"]:
        print("VERDICT: ARTIFACT SUSPECTED -- the untrained net locks comparably. "
              "Retire the attribution view; the three interventions carry the delay.")
    else:
        print("VERDICT: LOCK SURVIVES THE CONTROL -- attributable to training, "
              "so TSR is a fourth independent line of delay evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
