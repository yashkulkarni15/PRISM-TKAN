#!/usr/bin/env python3
"""Why does MiniROCKET beat TKAN on held-out data? The noise comparison, run locally.

Reads the per-file score CSVs both models write (``scores_<model>_fold<F>_<split>.csv``
+ complete JSON sidecars, schema :data:`pri2m.xai.model_compare.SCORE_COLUMNS`)
and answers, each with a reading pre-committed in the report JSON:

A. Does TKAN lose more AUC from clean to the real 5 % tier than MiniROCKET (H1)?
B. Is TKAN's held-out accuracy collapse a threshold shift (scores move together)?
C. Held-out AUC by fold, dataset, b/blim band and omega tertile; does the val
   5 %-tier AUC predict the held-out AUC?
D. Injected-noise dose-response, gated by an injection-validity check against the
   real 5 % mirrors of the same cuts.
E. TKAN with vs without side inputs on held-out.

A section whose inputs are missing is UNDECIDED with a reason -- never a number.

Example::

    python scripts/compare_noise_phase3.py \\
        --scores-dir results/phase3/xai/noise_compare \\
        --calibration results/phase3/xai/noise_compare/mirror_noise_calibration.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from pri2m.xai.model_compare import run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", default="results/phase3/xai/noise_compare")
    ap.add_argument("--calibration",
                    default="results/phase3/xai/noise_compare/mirror_noise_calibration.json")
    ap.add_argument("--floor", type=float, default=0.0092)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results/phase3/xai/noise_compare/report")
    ap.add_argument("--figures-dir", default="results/figures/research_update_assets")
    args = ap.parse_args(argv)

    rep = run(args.scores_dir, calibration=args.calibration, floor=args.floor,
              n_boot=args.n_boot, seed=args.seed, output_dir=args.output_dir,
              figures_dir=args.figures_dir)
    for key in ("A_noise_tier", "B_score_shift", "C_heldout", "D_dose_response",
                "E_side_inputs"):
        sec = rep[key]
        why = f" ({sec['reason']})" if sec.get("reason") else ""
        print(f"[compare_noise] {key}: {sec['reading']}{why}")
    if rep["problems"]:
        print(f"[compare_noise] skipped {len(rep['problems'])} file(s): "
              + "; ".join(rep["problems"][:5]))
    print(f"[compare_noise] wrote {args.output_dir}/noise_compare_report.json, "
          f"summary .md, figures {rep['figures']} -> {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
