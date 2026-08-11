#!/usr/bin/env python3
"""Read the learned delay coupling as a tooth-pass comb, across all five folds.

Consumes the stored ``regenerative.json`` manifests (U13-C) and the ``fuzz``
manifests (for the physical tooth-pass and chatter frequencies). **Needs no
checkpoint, no GPU and no re-run** -- every input is already on disk, which is
why this exists as a separate reader rather than as another pass over the
network.

What it produces, per fold and pooled:

* the comb tap ``b = corr*gain`` per delay/instantaneous edge pair, and the
  passband/stopband/contrast that follow from it;
* the fraction of pairs for which the comb reading is *legitimate at all*
  (``|corr| >= threshold``), reported as a headline rather than a caveat;
* the split by GRU gate and by input channel -- which gate the delay enters is a
  mechanistic claim, not a summary statistic;
* the same statistics on the **cross-channel control rows** already in the
  manifest, so every number has its control from the same file;
* what the comb does to the chatter band, integrated over the real spread of
  ``fn/tpf`` rather than evaluated at one convenient ratio.

Usage::

    python3 scripts/comb_phase3.py
    python3 scripts/comb_phase3.py --regen-root results/phase3/xai/regenerative
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pri2m.xai.comb import (  # noqa: E402
    DEFAULT_CORR_THRESHOLD,
    aggregate_comb,
    comb_coefficient,
    comb_metrics,
    comb_response,
    expected_chatter_attenuation,
)

DEFAULT_REGEN_ROOT = pathlib.Path("results/phase3/xai/regenerative")
DEFAULT_FUZZ_ROOT = pathlib.Path("results/phase3/xai/fuzz")
DEFAULT_OUT = pathlib.Path("results/phase3/xai/comb")
DEFAULT_FIG = pathlib.Path("results/figures/research_update_assets")
# Draws used to integrate the chatter attenuation over the fn/tpf spread.
_N_RATIO_DRAWS = 20000


def fold_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    """Fold directories holding a ``regenerative.json``, in fold order."""
    return sorted(
        (d for d in root.iterdir() if d.is_dir() and (d / "regenerative.json").exists()),
        key=lambda d: d.name,
    )


def read_frequencies(fuzz_root: pathlib.Path, fold_name: str) -> dict | None:
    """Tooth-pass and chatter frequencies for a fold, from its fuzz manifest.

    The band-notch contrast recorded these while it was matching notch energies;
    they are the only per-fold physical frequencies on local disk, and they are
    what turns a normalised comb axis into Hz. Returns ``None`` -- never a guess
    -- when the fold has no fuzz manifest, so the caller can say so.
    """
    # fuzz dirs are named fold0..fold4 while regenerative uses fold2_w512.
    stem = fold_name.split("_")[0]
    path = fuzz_root / stem / "fuzz_manifest.json"
    if not path.exists():
        return None
    try:
        match = json.loads(path.read_text())["contrasts"]["band_notch"]["energy_match"]
    except (KeyError, json.JSONDecodeError):
        return None
    centers = match.get("tooth_centers_hz") or []
    if not centers or not match.get("fn_hz_mean"):
        return None
    return {
        "tooth_pass_hz": float(centers[0]),
        "tooth_centers_hz": [float(c) for c in centers],
        "fn_hz_mean": float(match["fn_hz_mean"]),
        "fn_hz_sd": float(match.get("fn_hz_sd") or 0.0),
        "source": str(path),
    }


def chatter_band_summary(b: float, freqs: dict | None, seed: int = 0) -> dict:
    """What the comb does to the chatter band, honestly.

    Reports the single-ratio number *and* the expectation over the measured
    spread, because they disagree: fold 2's mean ratio sits near the second
    tooth-pass harmonic and is barely attenuated, while the spread puts chatter
    anywhere on the comb. Quoting only the first would overstate a notch;
    quoting only the second would hide that a given cut may see none.
    """
    if freqs is None or not np.isfinite(b):
        return {"available": False, "reason": "no per-fold frequencies on disk"}
    tpf = freqs["tooth_pass_hz"]
    ratio_mean = freqs["fn_hz_mean"] / tpf
    ratio_sd = freqs["fn_hz_sd"] / tpf
    rng = np.random.default_rng(seed)
    draws = rng.normal(ratio_mean, ratio_sd, size=_N_RATIO_DRAWS) if ratio_sd > 0 \
        else np.full(_N_RATIO_DRAWS, ratio_mean)
    draws = draws[draws > 0]
    peak = comb_metrics(b)["passband"]
    return {
        "available": True,
        "fn_over_tpf_mean": float(ratio_mean),
        "fn_over_tpf_sd": float(ratio_sd),
        # Distance from the nearest tooth-pass harmonic, in units of tpf: 0.0 is
        # on a passband (no attenuation), 0.5 is in a notch.
        "mean_ratio_distance_to_harmonic": float(
            abs(ratio_mean - round(ratio_mean))
        ),
        "attenuation_at_mean_ratio": float(
            comb_response(b, np.array([ratio_mean]))[0] / peak
        ),
        "expected_attenuation_over_spread": expected_chatter_attenuation(b, draws),
        # The forced content sits on a passband in every cut by construction;
        # this is the reference the chatter numbers are against.
        "attenuation_of_forced_content": 1.0,
        "n_draws": int(draws.size),
        "seed": int(seed),
    }


def analyse_fold(path: pathlib.Path, freqs: dict | None,
                 corr_threshold: float) -> dict:
    """Comb statistics for one fold, with its cross-channel control."""
    manifest = json.loads((path / "regenerative.json").read_text())
    rows = manifest.get("rows", [])
    control = manifest.get("control_rows", [])

    trained = aggregate_comb(rows, corr_threshold=corr_threshold)
    ctrl = aggregate_comb(control, corr_threshold=corr_threshold)

    # The top pair by importance is the one worth quoting as the exemplar.
    top = (manifest.get("top_pairs") or [None])[0]
    top_comb = None
    if top is not None:
        b_top = comb_coefficient(top["corr"], top["gain"])
        top_comb = {**comb_metrics(b_top), "corr": top["corr"], "gain": top["gain"],
                    "kan": top.get("kan"), "channel": top.get("channel"),
                    "out": top.get("out")}

    b_median = trained.get("b_median")
    return {
        "fold": path.name,
        "checkpoint": (manifest.get("provenance") or {}).get("checkpoint"),
        "n_windows": (manifest.get("provenance") or {}).get("n_windows"),
        "trained": trained,
        "cross_channel_control": ctrl,
        # Summary-only in the stored manifest: no per-row null exists, so the
        # comparison that can be made is on proportionality, not on contrast.
        "untrained_null_summary": manifest.get("untrained_null_summary"),
        "top_pair": top_comb,
        "frequencies": freqs,
        "chatter_band": chatter_band_summary(b_median, freqs)
        if b_median is not None else {"available": False, "reason": "no qualifying pairs"},
    }


def make_figures(per_fold: list[dict], fig_dir: pathlib.Path) -> dict:
    """Two figures: the comb itself, and the contrast distribution vs control."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    # -- 1. the learned comb, on a physical axis, with the chatter band marked --
    ref = next((f for f in per_fold if f["frequencies"] and f["top_pair"]), None)
    if ref is not None:
        b = ref["top_pair"]["b"]
        tpf = ref["frequencies"]["tooth_pass_hz"]
        f_hz = np.linspace(0, 4.2 * tpf, 2000)
        mag = comb_response(b, f_hz / tpf)

        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(f_hz, mag, lw=1.8, color="#1f4e79",
                label=f"learned pair, b = {b:.3f}")
        for k, c in enumerate(np.arange(tpf, 4.2 * tpf, tpf), start=1):
            ax.axvline(c, color="#2e7d32", ls="--", lw=1.0, alpha=0.8,
                       label="tooth-pass harmonics" if k == 1 else None)
        fn, sd = ref["frequencies"]["fn_hz_mean"], ref["frequencies"]["fn_hz_sd"]
        ax.axvspan(max(fn - sd, 0), fn + sd, color="#c62828", alpha=0.13,
                   label=f"chatter band, fn = {fn:.0f} +/- {sd:.0f} Hz")
        ax.axvline(fn, color="#c62828", lw=1.4)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("|H(f)| of the delay-and-add pair")
        ax.set_title(
            f"The learned delay pair is a comb tuned to the tooth-pass frequency "
            f"({ref['fold']})"
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.margins(x=0)
        fig.tight_layout()
        p = fig_dir / "comb_response.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written["comb_response"] = str(p)

    # -- 2. contrast per fold, trained vs cross-channel control ----------------
    folds = [f["fold"] for f in per_fold]
    tr = [(f["trained"] or {}).get("contrast_db_median") for f in per_fold]
    ct = [(f["cross_channel_control"] or {}).get("contrast_db_median") for f in per_fold]
    prop = [(f["trained"] or {}).get("frac_proportional") for f in per_fold]
    if any(v is not None for v in tr):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
        x = np.arange(len(folds))
        a1.bar(x - 0.2, [v if v is not None else 0 for v in tr], 0.4,
               label="delay pairs", color="#1f4e79")
        a1.bar(x + 0.2, [v if v is not None else 0 for v in ct], 0.4,
               label="cross-channel control", color="#9e9e9e")
        a1.set_xticks(x); a1.set_xticklabels(folds, rotation=20, ha="right")
        a1.set_ylabel("median comb contrast (dB)")
        a1.set_title("Comb contrast, against its control")
        a1.legend(fontsize=8)

        a2.bar(x, [v if v is not None else 0 for v in prop], 0.5, color="#2e7d32")
        a2.set_xticks(x); a2.set_xticklabels(folds, rotation=20, ha="right")
        a2.set_ylabel(f"fraction with |corr| >= {DEFAULT_CORR_THRESHOLD}")
        a2.set_ylim(0, 1)
        a2.set_title("How much of the network the reading covers")
        fig.tight_layout()
        p = fig_dir / "comb_contrast_by_fold.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written["comb_contrast_by_fold"] = str(p)

    return written


def make_probe_figure(probe_paths, fig_dir: pathlib.Path) -> dict:
    """Predicted vs measured vs untrained, from the returned probe manifests.

    Rendered here rather than on the cluster so the figure can be regenerated
    from the fetched JSON without another job: the manifests carry the full
    swept curves, and only the plot was left behind on GL.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifests = []
    for p in sorted(probe_paths):
        try:
            manifests.append((pathlib.Path(p).stem.replace("comb_probe_", ""),
                              json.loads(pathlib.Path(p).read_text())))
        except (OSError, json.JSONDecodeError):
            continue
    if not manifests:
        return {}

    fig, axes = plt.subplots(1, len(manifests), figsize=(3.4 * len(manifests), 3.4),
                             sharey=True, squeeze=False)
    for ax, (fold, d) in zip(axes[0], manifests):
        ratios = np.asarray(d["ratios"], float)
        tr = next((r for r in d["trained"] if "measured_normalised" in r), None)
        un = next((r for r in d["untrained"] if "measured_normalised" in r), None)
        cmp0 = (d.get("predicted_vs_measured") or [None])[0]
        if tr is None or cmp0 is None:
            continue
        pred = comb_response(cmp0["b"], ratios)
        pred = pred / pred.max()
        ax.plot(ratios, tr["measured_normalised"], lw=1.9, color="#1f4e79",
                label="measured (trained)")
        ax.plot(ratios, pred, lw=1.2, ls="--", color="#c62828", label="predicted")
        if un is not None:
            ax.plot(ratios, un["measured_normalised"], lw=1.1, color="#9e9e9e",
                    label="untrained control")
        for k in range(1, int(ratios.max()) + 1):
            ax.axvline(k, color="#2e7d32", ls=":", lw=0.9)
        ax.set_title(fold, fontsize=9)
        ax.set_xlabel("f / tooth-pass")
        ax.margins(x=0)
    axes[0][0].set_ylabel("normalised response")
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Swept response of the top delay pair: measured, predicted, and untrained",
                 fontsize=11)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "comb_probe_all_folds.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return {"comb_probe_all_folds": str(out)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regen-root", type=pathlib.Path, default=DEFAULT_REGEN_ROOT)
    ap.add_argument("--fuzz-root", type=pathlib.Path, default=DEFAULT_FUZZ_ROOT)
    ap.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--fig-dir", type=pathlib.Path, default=DEFAULT_FIG)
    ap.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD,
                    help="Minimum |corr| for the comb reading to describe a pair.")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    if not args.regen_root.exists():
        print(f"No regenerative manifests at {args.regen_root}", file=sys.stderr)
        return 2
    folds = fold_dirs(args.regen_root)
    if not folds:
        print(f"No fold directories with regenerative.json under {args.regen_root}",
              file=sys.stderr)
        return 2

    per_fold = []
    for d in folds:
        freqs = read_frequencies(args.fuzz_root, d.name)
        per_fold.append(analyse_fold(d, freqs, args.corr_threshold))
        t = per_fold[-1]["trained"]
        print(f"{d.name}: {t['n_used']}/{t['n_usable']} pairs qualify "
              f"(|corr|>={args.corr_threshold}), "
              f"median contrast {t['contrast_db_median']:.1f} dB, "
              f"peaks-at-tooth-pass {t['frac_peaks_at_tooth_pass']:.3f} "
              f"(importance-weighted {t['weighted_frac_peaks_at_tooth_pass']:.3f})"
              if t["n_used"] else f"{d.name}: no qualifying pairs")

    pooled_rows: list[dict] = []
    for d in folds:
        pooled_rows.extend(json.loads((d / "regenerative.json").read_text())
                           .get("rows", []))
    pooled = aggregate_comb(pooled_rows, corr_threshold=args.corr_threshold)

    figures = {}
    if not args.no_figures:
        figures = make_figures(per_fold, args.fig_dir)
        # Probe manifests are optional: the comb reading stands without them,
        # and the confirmation figure simply appears once they are fetched.
        figures.update(make_probe_figure(
            sorted(args.output_dir.glob("comb_probe_*.json")), args.fig_dir))

    manifest = {
        "analysis": "tooth-pass comb reading of the learned delay coupling",
        "inputs": {"regenerative_root": str(args.regen_root),
                   "fuzz_root": str(args.fuzz_root)},
        "corr_threshold": float(args.corr_threshold),
        "network_reloaded": False,
        "note": (
            "Derived entirely from stored U13-C manifests: no checkpoint was "
            "loaded and no window was re-scored. b = corr*gain is the projection "
            "of the delayed edge curve onto its instantaneous partner; the comb "
            "reading describes only pairs above corr_threshold, and the "
            "qualifying fraction is reported per fold."
        ),
        "per_fold": per_fold,
        "pooled": pooled,
        "figures": figures,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "comb_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out}")
    if pooled["n_used"]:
        print(f"POOLED: {pooled['n_used']}/{pooled['n_usable']} pairs qualify; "
              f"peaks-at-tooth-pass {pooled['frac_peaks_at_tooth_pass']:.3f}, "
              f"importance-weighted {pooled['weighted_frac_peaks_at_tooth_pass']:.3f}")
    for name, p in figures.items():
        print(f"  figure {name}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
