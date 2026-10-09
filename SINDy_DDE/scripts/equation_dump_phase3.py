#!/usr/bin/env python3
"""J3: dump one TKAN's necessary-cluster delay-edge curves, a re-initialised null, and logits.

Writes ``equations/tkan_<NAME>.npz`` -- ``edges`` (JSON of the cluster edge dicts),
``xs``/``ys`` [E,256] (each edge's full function over its channel's real data range),
``untrained_ys`` [n,E,256] (the same edges in ``--n-untrained`` random re-initialisations),
``val_score``/``ho_score`` (raw ``chatter_logit`` on the shared J2 windows) and the
windows' ``val_file``/``ho_file`` -- then a sidecar ``tkan_<NAME>.json`` (``complete``
last). An empty cluster (every claimable mode excluded) still writes scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pri2m.xai import activation_modes as AM  # noqa: E402
from pri2m.xai.equations import CHANNELS, range_curves  # noqa: E402
from pri2m.xai.sampling import batch_to_model_device  # noqa: E402
from recurrence_phase3 import _load_network, necessary_cluster, parse_network  # noqa: E402

N_POINTS = 256


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_windows(npz: Path, fold: int) -> str | None:
    """Reason the window file is unusable for ``fold``, else None (checked before any model load)."""
    side = npz.with_suffix(".json")
    if not npz.exists() or not side.exists():
        return f"window file or sidecar missing: {npz}, {side.name}"
    meta = json.loads(side.read_text())
    if meta.get("complete") is not True:
        return f"window sidecar {side.name} is not complete"
    if meta.get("sha256") != _sha(npz):
        return f"window sidecar {side.name}: sha256 mismatch"
    if meta.get("fold") != fold:
        return f"window sidecar {side.name} is fold {meta.get('fold')}, network is fold {fold}"
    return None


def _score(model, w: np.lib.npyio.NpzFile, prefix: str, bs: int) -> np.ndarray:
    keys = {"window": f"{prefix}_window", "tau": f"{prefix}_tau",
            "omega": f"{prefix}_omega", "tpf": f"{prefix}_tpf"}
    t = {k: torch.as_tensor(w[v]) for k, v in keys.items()}
    t["tau"] = t["tau"].long()
    n, out = len(t["window"]), []
    with torch.no_grad():
        for i in range(0, n, bs):
            b = batch_to_model_device({k: v[i:i + bs] for k, v in t.items()}, model)
            out.append(model(b)["chatter_logit"].detach().cpu().reshape(-1).numpy())
    return np.concatenate(out).astype(np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--network", required=True, help="NAME=CKPT=MANIFEST; NAME starts with f<fold>")
    ap.add_argument("--windows", required=True, type=Path)
    ap.add_argument("--exclude-mode", action="append", default=[], metavar="NAME:MODE")
    ap.add_argument("--n-untrained", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--output-dir", required=True, type=Path)
    a = ap.parse_args(argv)

    net = parse_network(a.network)
    name = net["name"]
    excluded = []
    for spec in a.exclude_mode:
        nm, _, md = spec.rpartition(":")
        if not nm or not md.lstrip("-").isdigit():
            print(f"bad --exclude-mode {spec!r}; expected NAME:MODE", file=sys.stderr)
            return 2
        if nm == name:
            excluded.append(int(md))

    bad = _check_windows(a.windows, net["fold"])
    if bad:
        print(bad, file=sys.stderr)
        return 2

    n = _load_network(net, a)
    model = n["model"]
    cluster = necessary_cluster(n["ablation"], excluded)["cluster"]
    w = np.load(a.windows, allow_pickle=False)
    ranges = {c: (float(w["range_lo"][i]), float(w["range_hi"][i])) for i, c in enumerate(CHANNELS)}

    def curves(m):
        if not cluster:
            return np.zeros((0, N_POINTS)), np.zeros((0, N_POINTS))
        return range_curves(m, cluster, ranges, N_POINTS)

    xs, ys = curves(model)
    null = []
    for s in range(a.n_untrained):
        fresh, _ = AM.reinitialised_copy(model, seed=s)
        AM.assert_reinitialised(model, fresh)
        null.append(curves(fresh)[1])
        del fresh
    untrained = np.stack(null) if null else np.zeros((0, len(cluster), N_POINTS))

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"tkan_{name}.npz", edges=json.dumps(cluster), xs=xs, ys=ys, untrained_ys=untrained,
        val_score=_score(model, w, "val", a.batch_size), ho_score=_score(model, w, "ho", a.batch_size),
        val_file=w["val_file"], ho_file=w["ho_file"])
    npz_sha = _sha(out / f"tkan_{name}.npz")    # after the npz is closed, before the sidecar
    (out / f"tkan_{name}.json").write_text(json.dumps({
        "name": name, "fold": net["fold"], "train_seed": n["train_seed"],
        "checkpoint": str(net["checkpoint"]), "checkpoint_sha256": _sha(net["checkpoint"]),
        "windows": str(a.windows), "windows_sha256": _sha(a.windows),
        "exclusions": excluded, "n_untrained": a.n_untrained, "n_edges": len(cluster),
        "npz_sha256": npz_sha, "complete": True}, indent=2))
    print(f"{name}: {len(cluster)} edges, {a.n_untrained} null re-inits, scores written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
