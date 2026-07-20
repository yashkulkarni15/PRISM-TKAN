#!/usr/bin/env python3
"""Run Phase 3 XAI artifacts for a trained TKAN checkpoint.

Loads a trained ``TKANModel`` checkpoint (a ``state_dict`` saved by
``scripts/train_phase3.py``), then produces the KAN-native + attribution
explainability artifacts on one sample window:

* ``edge_activations.png`` -- every ``KANLinear`` edge's learned B-spline
  activation curve (:func:`pri2m.xai.kan_native.plot_edge_activations`).
* ``symbolic_summary.txt`` -- the dominant input edges per output
  (:func:`pri2m.xai.kan_native.symbolic_summary`).
* ``prune_report.json`` -- kept/pruned edge counts per layer at ``--prune-threshold``.
* ``integrated_gradients.npy`` + ``.png`` -- IG attribution ``[L, C]`` of the
  ``b_over_blim`` head on the sampled window.

The model geometry (``input_size``, ``hidden_size``, ``num_layers``) is inferred
from the checkpoint's tensor shapes so no config is required, with CLI overrides.

Example::

    python scripts/xai_phase3.py \
        --checkpoint results/phase3/runs/<run_id>/checkpoints/best.pt \
        --npz-root results/phase3/npz \
        --table results/phase3/phase3_dataset.csv \
        --output-dir results/phase3/xai
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
# src/ enables ``import pri2m...``; repo root keeps parity with other scripts.
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402  (after sys.path setup)
import torch  # noqa: E402

from pri2m.data.phase3_sequence import (  # noqa: E402
    NormStats,
    WindowConfig,
    iter_windows,
    load_npz_signals,
    tau_samples,
)
from pri2m.models.tkan_model import DEFAULT_SIDE_FEATURES, TKANModel  # noqa: E402
from pri2m.xai import attribution, kan_native  # noqa: E402

# Fallback levels for a quantile checkpoint whose run manifest lost its
# ``severity`` block -- must track ``SeverityHead``'s own default.
_DEFAULT_SEVERITY_QUANTILES = (0.1, 0.5, 0.9)


def _infer_dims(state_dict: dict) -> dict:
    """Infer ``input_size``/``hidden_size``/``num_layers`` from a TKAN state_dict.

    * ``severity.reg_head.base_weight`` is ``[1, hidden_size]`` -- except for
      side-input checkpoints, where the severity head consumes the encoder
      context concatenated with the side branch, so its width is
      ``hidden_size + side_hidden`` and the trunk width is recovered by
      subtracting ``side_hidden`` (= ``side_mlp.0.weight.shape[0]``).
    * ``encoder.layers.0.kan_z.base_weight`` is ``[hidden_size, 2*input+hidden]``
      -- the first cell's gate consumes ``(x_t, x_delay, h_prev)`` -- so
      ``input_size = (cols - hidden_size) / 2``.
    * ``num_layers`` = count of distinct ``encoder.layers.<k>`` indices.
    """
    hidden_size = int(state_dict["severity.reg_head.base_weight"].shape[1])
    if "side_mean" in state_dict:
        hidden_size -= int(state_dict["side_mlp.0.weight"].shape[0])
    z_cols = int(state_dict["encoder.layers.0.kan_z.base_weight"].shape[1])
    input_size = (z_cols - hidden_size) // 2
    layer_ids = {
        int(k.split("encoder.layers.")[1].split(".")[0])
        for k in state_dict
        if k.startswith("encoder.layers.")
    }
    num_layers = (max(layer_ids) + 1) if layer_ids else 1
    return {"input_size": input_size, "hidden_size": hidden_size, "num_layers": num_layers}


def _find_preprocessing(checkpoint: str) -> str | None:
    """Auto-discover ``preprocessing.json`` by walking up from the checkpoint.

    The checkpoint lives at ``<run_dir>/checkpoints/best.pt`` and the companion
    manifest at ``<run_dir>/preprocessing.json``, so we search the checkpoint's
    directory and each of its parents. Returns the path as a string, or ``None``
    if no manifest is found anywhere above the checkpoint.
    """
    ckpt_path = Path(checkpoint).resolve()
    for parent in (ckpt_path.parent, *ckpt_path.parents):
        candidate = parent / "preprocessing.json"
        if candidate.is_file():
            return str(candidate)
    return None


def _apply_norm(window: np.ndarray, preprocessing: dict) -> np.ndarray:
    """Z-score a sampled window ``[L, C]`` with the run's saved per-channel stats.

    ``preprocessing`` is the parsed ``preprocessing.json``; its ``norm.mean`` /
    ``norm.std`` arrays are turned into a :class:`NormStats` and applied exactly
    as the training data loader does, so XAI runs on in-distribution inputs.
    """
    norm = preprocessing["norm"]
    stats = NormStats(
        mean=np.asarray(norm["mean"], dtype=np.float64),
        std=np.asarray(norm["std"], dtype=np.float64),
    )
    return stats.apply(window)


def _load_model(
    checkpoint: str, args: argparse.Namespace, preprocessing: dict | None = None
) -> TKANModel:
    """Build a ``TKANModel`` matching the checkpoint and load its weights.

    Side-input checkpoints are detected via the ``side_mean`` buffer in the
    state_dict; ``side_hidden`` / the side-feature count ``S`` come from
    ``side_mlp.0.weight`` (shape ``[side_hidden, S]``) and the feature NAMES
    from the run's ``preprocessing.json`` ``side.features`` block, falling back
    to :data:`DEFAULT_SIDE_FEATURES` (with a stderr warning) when absent.
    Non-side checkpoints take exactly the old construction path.

    Quantile-severity checkpoints are detected the same way: the severity head's
    output width ``Q = severity.reg_head.base_weight.shape[0]`` is > 1, so the
    default point head would fail ``load_state_dict`` with a size mismatch. The
    LEVELS are not recoverable from the state_dict (only their count), so they
    come from the run's ``preprocessing.json`` ``severity.quantiles`` block,
    falling back to :data:`_DEFAULT_SEVERITY_QUANTILES` with a stderr warning.
    """
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dims = _infer_dims(state_dict)
    input_size = args.input_size or dims["input_size"]
    hidden_size = args.hidden_size or dims["hidden_size"]

    side_kwargs: dict = {}
    if "side_mean" in state_dict:
        side_hidden = int(state_dict["side_mlp.0.weight"].shape[0])
        n_side = int(state_dict["side_mlp.0.weight"].shape[1])
        side_block = (preprocessing or {}).get("side") or {}
        if side_block.get("features"):
            features = tuple(str(f) for f in side_block["features"])
        else:
            features = tuple(DEFAULT_SIDE_FEATURES)
            print(
                "WARNING: side-input checkpoint but no 'side' block in "
                "preprocessing.json -- assuming DEFAULT_SIDE_FEATURES="
                f"{features}. If the run used different side features, the "
                "attribution inputs will be WRONG.",
                file=sys.stderr,
            )
        if len(features) != n_side:
            raise ValueError(
                f"side-feature count mismatch: checkpoint expects {n_side} side "
                f"features (side_mlp.0.weight is [{side_hidden}, {n_side}]) but "
                f"preprocessing.json names {len(features)}: {features}"
            )
        side_kwargs = {
            "side_input": True,
            "side_features": features,
            "side_hidden": side_hidden,
        }

    severity_kwargs: dict = {}
    n_quantiles = int(state_dict["severity.reg_head.base_weight"].shape[0])
    if n_quantiles > 1:
        severity_block = (preprocessing or {}).get("severity") or {}
        if severity_block.get("quantiles"):
            quantiles = tuple(float(q) for q in severity_block["quantiles"])
        else:
            quantiles = tuple(_DEFAULT_SEVERITY_QUANTILES)
            print(
                "WARNING: quantile-severity checkpoint but no 'severity' block "
                "in preprocessing.json -- assuming quantiles="
                f"{quantiles}. If the run used different levels, the predicted "
                "intervals will be MISLABELLED.",
                file=sys.stderr,
            )
        if len(quantiles) != n_quantiles:
            raise ValueError(
                f"quantile-level count mismatch: checkpoint expects "
                f"{n_quantiles} levels (severity.reg_head.base_weight is "
                f"[{n_quantiles}, ...]) but preprocessing.json names "
                f"{len(quantiles)}: {quantiles}"
            )
        severity_kwargs = {
            "severity_mode": "quantile",
            "severity_quantiles": quantiles,
        }

    model = TKANModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=dims["num_layers"],
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        **side_kwargs,
        **severity_kwargs,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _sample_window(args: argparse.Namespace, n_channels: int):
    """Return one window ``[L, C]``, its tau (samples), and its scalar side meta.

    The third element is a ``{"omega": ..., "tpf": ..., "fs": ...}`` dict of the
    sample's process parameters (real NPZ meta when available) so side-input
    models can be fed the matching scalar features. Falls back to a synthetic
    chatter-like window if no NPZ tree is supplied, so the script still produces
    artifacts for a smoke run.
    """
    cfg = WindowConfig(length=args.window_length, stride=args.window_length)
    if args.npz_root:
        files = sorted(Path(args.npz_root).rglob("*.npz"))
        if files:
            signal, meta = load_npz_signals(files[0], cfg.channels)
            windows = iter_windows(signal, cfg)
            if len(windows) > 0:
                fs = float(meta.get("fs", 10_000.0))
                tpf = float(meta.get("tpf", 200.0))
                tau = tau_samples(fs, tpf)
                side_meta = {
                    "omega": float(meta.get("omega", 3000.0)),
                    "tpf": tpf,
                    "fs": fs,
                }
                return windows[0].astype(np.float32), tau, side_meta

    # Synthetic fallback: a TPF + chatter tone over the requested channels.
    fs, tpf, fn, omega = 10_000.0, 200.0, 900.0, 3000.0
    print(
        "WARNING: no NPZ window available (no --npz-root or no usable file); "
        f"using a SYNTHETIC fallback window (fs={fs}, tpf={tpf}, fn={fn}) -- "
        f"the side features (omega={omega}, tpf={tpf}) are synthetic too.",
        file=sys.stderr,
    )
    t = np.arange(cfg.length) / fs
    base = np.sin(2 * np.pi * tpf * t) + 1.0 * np.sin(2 * np.pi * fn * t)
    cols = [base * (1.0 if c % 2 == 0 else 0.5) for c in range(n_channels)]
    window = np.stack(cols, axis=-1).astype(np.float32)
    return window, tau_samples(fs, tpf), {"omega": omega, "tpf": tpf, "fs": fs}


def _plot_attribution(attr: np.ndarray, out_path: Path) -> None:
    """Heatmap of the IG attribution ``[L, C]``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(attr.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_xlabel("time step")
    ax.set_ylabel("channel")
    ax.set_title("Integrated Gradients attribution (b_over_blim)")
    fig.colorbar(im, ax=ax, label="attribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a trained TKANModel state_dict (.pt).")
    parser.add_argument("--npz-root", default=None, help="Root of the dataset_<id>/*.npz tree (sample source).")
    parser.add_argument("--table", default=None, help="Phase 3 per-file table CSV (optional context).")
    parser.add_argument("--output-dir", default="results/phase3/xai", help="Where XAI artifacts are written.")
    parser.add_argument("--window-length", type=int, default=256, help="Sample window length for IG.")
    parser.add_argument("--ig-steps", type=int, default=32, help="Integrated-Gradients Riemann steps.")
    parser.add_argument("--prune-threshold", type=float, default=0.01, help="Edge-L1 prune threshold for the report.")
    parser.add_argument("--input-size", type=int, default=None, help="Override inferred input channel count.")
    parser.add_argument("--hidden-size", type=int, default=None, help="Override inferred hidden width.")
    parser.add_argument("--grid-size", type=int, default=5, help="KAN grid size (must match training).")
    parser.add_argument("--spline-order", type=int, default=3, help="KAN spline order (must match training).")
    parser.add_argument(
        "--preprocessing",
        default=None,
        help=(
            "Path to the run's preprocessing.json (per-run norm stats). If omitted, "
            "it is auto-discovered next to the checkpoint's run dir. Required for "
            "in-distribution XAI -- the model trains on z-scored windows."
        ),
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover/parse preprocessing.json BEFORE building the model: side-input
    # checkpoints take their side-feature names from its "side" block.
    pre_path = args.preprocessing or _find_preprocessing(args.checkpoint)
    preprocessing = (
        json.loads(Path(pre_path).read_text(encoding="utf-8")) if pre_path else None
    )

    model = _load_model(args.checkpoint, args, preprocessing)
    n_channels = model.input_size

    # ---- KAN-native artifacts ------------------------------------------- #
    curves = kan_native.extract_spline_curves(model)
    png_path = kan_native.plot_edge_activations(curves, out_dir / "edge_activations.png")

    summary_text = kan_native.symbolic_summary(model)
    (out_dir / "symbolic_summary.txt").write_text(summary_text, encoding="utf-8")

    prune = kan_native.prune_edges(model, threshold=args.prune_threshold)
    prune_json = {
        name: {"kept": rec["kept"], "pruned": rec["pruned"]} for name, rec in prune.items()
    }
    (out_dir / "prune_report.json").write_text(json.dumps(prune_json, indent=2), encoding="utf-8")

    # ---- Attribution on a sample window --------------------------------- #
    window, tau, side_meta = _sample_window(args, n_channels)

    # The model trains on globally z-scored windows (NormStats.apply in the data
    # loader). Feeding it raw windows leaves the force channels O(tens-hundreds),
    # which saturate the KANLinear tanh grid squash to +/-1 and make every
    # attribution off-distribution. Apply the run's saved norm before inference.
    normalized = False
    if preprocessing is not None:
        window = _apply_norm(window, preprocessing).astype(np.float32)
        normalized = True
    else:
        print(
            "\n"
            "================================ WARNING ================================\n"
            "No preprocessing.json found (and none supplied via --preprocessing).\n"
            "XAI is running on RAW, UN-NORMALIZED windows -- the model trains on\n"
            "z-scored windows, so these inputs are OFF-DISTRIBUTION (force channels\n"
            "saturate the tanh grid squash). The resulting attributions are\n"
            "UNRELIABLE and must not be trusted. Pass --preprocessing PATH or place\n"
            "preprocessing.json in the checkpoint's run directory.\n"
            "========================================================================\n",
            file=sys.stderr,
        )

    # Side-input models also need their scalar process parameters in the batch.
    # "tau" is satisfied by the tau argument (the IG wrapper puts it in the
    # batch); everything else comes from the sample's NPZ meta (or the synthetic
    # fallback defaults).
    extra_scalars = None
    if getattr(model, "side_input", False):
        extra_scalars = {}
        for feat in model.side_features:
            if feat == "tau":
                continue  # flows through the tau argument
            if feat not in side_meta:
                raise ValueError(
                    f"side feature {feat!r} required by the checkpoint has no "
                    f"value in the sampled window's meta {sorted(side_meta)}"
                )
            extra_scalars[feat] = float(side_meta[feat])

    attr = attribution.integrated_gradients(
        model,
        window,
        target="b_over_blim",
        tau=tau,
        steps=args.ig_steps,
        extra_scalars=extra_scalars,
    )
    np.save(out_dir / "integrated_gradients.npy", attr)
    _plot_attribution(attr, out_dir / "integrated_gradients.png")

    manifest = {
        "checkpoint": str(args.checkpoint),
        "n_channels": int(n_channels),
        "window_length": int(window.shape[0]),
        "tau": int(tau),
        "normalized": bool(normalized),
        "preprocessing": str(pre_path) if pre_path else None,
        "side_input": bool(getattr(model, "side_input", False)),
        "artifacts": {
            "edge_activations": str(png_path),
            "symbolic_summary": str(out_dir / "symbolic_summary.txt"),
            "prune_report": str(out_dir / "prune_report.json"),
            "integrated_gradients_npy": str(out_dir / "integrated_gradients.npy"),
            "integrated_gradients_png": str(out_dir / "integrated_gradients.png"),
        },
        "attribution_abs_sum": float(np.abs(attr).sum()),
    }
    if extra_scalars is not None:
        # The side-feature values actually fed to the model (tau via its own arg).
        manifest["side_features"] = {**extra_scalars, "tau": float(tau)}
    manifest_path = out_dir / "xai_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
