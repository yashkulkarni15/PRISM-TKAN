#!/usr/bin/env python3
"""Run the eval-only Phase 3 graded-degradation (fuzziness) suite.

The runner measures predictive quality and regenerative-delay reliance while
progressively blurring either the signal batch or the trained TKAN itself.  A
single manifest is the source of truth for all reported values and figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import xai_phase3  # noqa: E402
from pri2m.data.phase3_sequence import WindowConfig  # noqa: E402
from pri2m.xai import degradation, sampling  # noqa: E402
from whitebox_phase3 import _git_sha, _now_date  # noqa: E402


def _parse_fuzzers(value: str):
    if value.strip().lower() == "all":
        names = list(degradation.FUZZERS)
    else:
        names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in degradation.FUZZERS]
    if unknown:
        raise ValueError(f"unknown fuzzers: {unknown}; choose from {list(degradation.FUZZERS)}")
    if not names:
        raise ValueError("at least one fuzzer is required")
    return names


def _self_check(fuzzers: dict, baseline: dict) -> dict:
    per_fuzzer = {}
    max_abs_dev = 0.0
    passed = True
    for name, result in fuzzers.items():
        tolerance = 5e-3 if name == "spline_coarsen" else 1e-6
        level0 = result["results"][0]["mean"]
        deviations = {}
        for key in ("auc", "delay_drop", "mae"):
            expected = baseline.get(key)
            observed = level0.get(key)
            if expected is None or observed is None:
                deviations[key] = None
                continue
            deviation = abs(float(observed) - float(expected))
            deviations[key] = deviation
            max_abs_dev = max(max_abs_dev, deviation)
            passed = passed and deviation <= tolerance
        per_fuzzer[name] = {
            "passed": all(v is None or v <= tolerance for v in deviations.values()),
            "tolerance": tolerance,
            "abs_dev": deviations,
        }
    return {"passed": bool(passed), "max_abs_dev": float(max_abs_dev), "per_fuzzer": per_fuzzer}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz-root", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--preprocessing", default=None)
    parser.add_argument("--output-dir", default="results/phase3/xai/fuzz")
    parser.add_argument("--n-windows", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=512)
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--fuzzers", default="all", help="'all' or a comma-separated registry subset")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds for stochastic fuzzers")
    parser.add_argument("--quick", action="store_true", help="Use two levels and one seed for smoke tests")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--val-auc", type=float, default=None)
    args = parser.parse_args(argv)

    if args.seeds < 1:
        parser.error("--seeds must be at least 1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_path = args.preprocessing or xai_phase3._find_preprocessing(args.checkpoint)
    if not pre_path:
        raise SystemExit("no preprocessing.json found; pass --preprocessing")
    preprocessing = json.loads(Path(pre_path).read_text(encoding="utf-8"))
    model = xai_phase3._load_model(args.checkpoint, args, preprocessing)
    model.pool = int(args.pool)
    model.eval()

    cfg = WindowConfig(
        length=int(args.window_length),
        stride=int(args.window_stride),
        channels=sampling.CHANNELS,
    )
    batch, labels = sampling.build_labeled_batch(
        args.table,
        args.npz_root,
        preprocessing,
        int(args.n_windows),
        cfg,
        seed=int(args.sample_seed),
    )

    baseline = degradation.measure_probes(model, batch, labels)
    baseline["xdelay_share"] = degradation.xdelay_share(model, batch)

    selected = {}
    for name in _parse_fuzzers(args.fuzzers):
        fuzzer = degradation.FUZZERS[name]
        selected[name] = replace(fuzzer, levels=tuple(fuzzer.levels[:2])) if args.quick else fuzzer
    seeds = (0,) if args.quick else tuple(range(int(args.seeds)))

    results: dict = {}
    artifacts: dict = {}
    errors: dict = {}

    def _safe(name, fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - preserve partial scientific run with traceback
            errors[name] = traceback.format_exc()
            print(f"WARNING: fuzz analysis {name!r} failed:\n{errors[name]}", file=sys.stderr)
            return None

    for name, fuzzer in selected.items():
        def _run(current=fuzzer):
            result = degradation.run_fuzzer(model, batch, labels, current, seeds=seeds)
            artifacts[current.name] = degradation.plot_degradation_panel(
                result, baseline, out_dir / f"fuzz_{current.name}.png"
            )
            return result

        result = _safe(name, _run)
        if result is not None:
            results[name] = result

    completed = list(results.values())
    if completed:
        dashboard = _safe(
            "dashboard",
            lambda: degradation.plot_fuzz_dashboard(
                completed, baseline, out_dir / "fuzz_dashboard.png"
            ),
        )
        if dashboard:
            artifacts["dashboard"] = dashboard

    self_check = _self_check(results, baseline)
    if not self_check["passed"]:
        print(
            f"WARNING: level-0 self-check failed (max abs dev {self_check['max_abs_dev']:.6g})",
            file=sys.stderr,
        )

    n_windows = int(batch["window"].shape[0])
    run_id = args.run_id or Path(args.checkpoint).resolve().parent.parent.name
    tag = args.tag or out_dir.name
    provenance = {
        "checkpoint": str(args.checkpoint),
        "preprocessing": str(pre_path),
        "run_id": run_id,
        "tag": tag,
        "val_auc": args.val_auc,
        "git_sha": _git_sha(REPO_ROOT),
        "rendered_date": _now_date(),
        "n_windows": n_windows,
        "sample_seed": int(args.sample_seed),
        "pool": int(args.pool),
        "seeds": list(seeds),
        "quick": bool(args.quick),
        "fuzzer_levels": {name: list(fuzzer.levels) for name, fuzzer in selected.items()},
    }
    manifest = {
        "provenance": provenance,
        "baseline": baseline,
        "fuzzers": results,
        "self_check": self_check,
        "artifacts": artifacts,
        "errors": errors,
    }
    manifest_path = out_dir / "fuzz_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")

    print(f"Fuzz manifest -> {manifest_path}")
    print(f"Baseline per-window AUC: {baseline['auc']:.4f}")
    for name, result in results.items():
        print(f"{name}: {result['breakdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
