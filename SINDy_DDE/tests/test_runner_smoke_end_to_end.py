"""End-to-end smoke for the three eval-only analysis runners.

WHY THIS EXISTS. Between 2026-08-11 and 2026-08-12, three cluster submissions
died in a row on three different bugs, all in the same shared runner template
and none catchable by the unit tests:

  1. a features path pointing at a dataset that has never existed
     (caught by the sbatch input check, 0s);
  2. ``_score`` reading ``auc`` at the top level when
     ``eval_heldout_phase3.run`` nests it under ``metrics`` -- died at baseline;
  3. ``cfg=None`` reaching ``Phase3SequenceDataset`` -- died after baseline,
     on ``'NoneType' object has no attribute 'channels'``.

Every one lived at the seam between a runner's ``main()`` and a real checkpoint,
which nothing local exercised: importing a runner and calling ``--help`` proves
only that its imports resolve. Each bug therefore cost a cluster round-trip and
was found one at a time, in submission order.

This drives each runner's FULL ``main()`` against a real checkpoint trained from
synthetic data, and asserts a complete manifest -- the same thing the cluster
does, minus the cluster. It is slow (it trains a model) and that is the point.

Kept deliberately cheap where it does not weaken the check: k=2 rather than 4,
two pruning fractions rather than twelve, a two-value k-sweep. The bug class this
targets fires on the first arm, not the twentieth.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests._phase3_synth import make_synth_folds, make_synth_npz_tree, make_synth_table

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_inputs(tmp_path: Path):
    npz_root = make_synth_npz_tree(
        tmp_path / "npz", clean_ids=(1, 2, 3), n_files=4, length=256, seed=0
    )
    table = make_synth_table(npz_root)
    folds = make_synth_folds(table, n_splits=3, seed=0)

    table_csv = tmp_path / "table.csv"
    table.to_csv(table_csv, index=False)
    folds_json = tmp_path / "folds.json"
    folds_json.write_text(json.dumps(folds))
    return npz_root, table_csv, folds_json


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train one smoke TKAN and hand back a real checkpoint + its inputs.

    Module-scoped: the training run is the expensive part and all three runners
    can share it.
    """
    tmp_path = tmp_path_factory.mktemp("runner_smoke")
    npz_root, table_csv, folds_json = _build_inputs(tmp_path)
    out_dir = tmp_path / "out"

    res = subprocess.run(
        [
            sys.executable, "scripts/train_phase3.py",
            "--model", "tkan",
            "--fold", "0",
            "--npz-root", str(npz_root),
            "--table", str(table_csv),
            "--folds-json", str(folds_json),
            "--output-dir", str(out_dir),
            "--mlflow-uri", f"file:{tmp_path / 'mlruns'}",
            "--smoke",
        ],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert res.returncode == 0, f"training failed\n{res.stderr[-4000:]}"

    ckpts = sorted(out_dir.rglob("*.pt"))
    assert ckpts, f"training wrote no checkpoint under {out_dir}"
    # Prefer best_auc.pt, which is what every cluster wrapper resolves to.
    ckpt = next((c for c in ckpts if c.name == "best_auc.pt"), ckpts[0])

    pre = ckpt.parent.parent / "preprocessing.json"
    assert pre.is_file(), (
        f"no preprocessing.json at {pre}; the runners resolve it as "
        "ckpt.parent.parent/preprocessing.json, so this layout is load-bearing"
    )
    return {
        "ckpt": ckpt, "preprocessing": pre, "npz_root": npz_root,
        "table_csv": table_csv, "tmp_path": tmp_path,
    }


def _run(script: str, args: list[str], out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, f"scripts/{script}", *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )


def _assert_complete(manifest: Path, res: subprocess.CompletedProcess, script: str):
    """Exit 0 is not evidence. Assert the manifest and its completion flag."""
    assert res.returncode == 0, (
        f"{script} exited {res.returncode}\n"
        f"--- stderr ---\n{res.stderr[-4000:]}\n--- stdout ---\n{res.stdout[-2000:]}"
    )
    assert manifest.is_file(), f"{script} exited 0 but wrote no manifest at {manifest}"
    m = json.loads(manifest.read_text())

    # Two legitimate shapes. The two 2026-08-12 runners write partials after
    # every arm and mark provenance.complete last, so presence alone proves
    # nothing. mode_ablation_phase3.py instead writes its manifest once,
    # atomically, at the end with a TOP-LEVEL "complete" -- for which absence is
    # the incompleteness signal. Accept either rather than force a shape.
    complete = m.get("provenance", {}).get("complete", m.get("complete"))
    assert complete is True, (
        f"{script} left a partial manifest; a wall kill and a finished run must "
        f"not look alike (found complete={complete!r})"
    )
    return m


def test_heldout_pruning_runner_completes(trained):
    out = trained["tmp_path"] / "prune"
    res = _run(
        "heldout_pruning_phase3.py",
        [
            "--checkpoint", str(trained["ckpt"]),
            "--features-csv", str(trained["table_csv"]),
            "--npz-root", str(trained["npz_root"]),
            "--table", str(trained["table_csv"]),
            "--preprocessing", str(trained["preprocessing"]),
            "--device", "cpu",
            "--n-windows", "8",
            "--fractions", "0.5,1.0",
            "--output-dir", str(out),
            "--tag", "smoke",
        ],
        out,
    )
    m = _assert_complete(out / "smoke" / "heldout_pruning.json", res, "heldout_pruning")
    assert m["baseline_auc"] is not None
    assert len(m["rows"]) == 2
    # The ranked arm at a fraction < 1.0 must have actually scored something.
    assert any(r.get("ranked_auc") is not None for r in m["rows"])


def test_symbolic_substitution_runner_completes(trained):
    out = trained["tmp_path"] / "symsub"
    res = _run(
        "symbolic_substitution_phase3.py",
        [
            "--checkpoint", str(trained["ckpt"]),
            "--features-csv", str(trained["table_csv"]),
            "--npz-root", str(trained["npz_root"]),
            "--table", str(trained["table_csv"]),
            "--preprocessing", str(trained["preprocessing"]),
            "--device", "cpu",
            "--n-windows", "8",
            "--k-sweep", "2,4",
            "--output-dir", str(out),
            "--tag", "smoke",
        ],
        out,
    )
    m = _assert_complete(
        out / "smoke" / "symbolic_substitution.json", res, "symbolic_substitution"
    )
    assert m["baseline_auc"] is not None
    # two k values unclamped + one clamped arm at max(k)
    assert len(m["rows"]) == 3
    assert m["rows"][-1]["clamped"] is True

    # The substitution must have APPLIED. "substitution did not apply" is the
    # guard firing, and it invalidates the arm regardless of the number beside it.
    for row in m["rows"]:
        for arm in ("correct_error", "permuted_error"):
            assert row.get(arm) != "substitution did not apply", (
                f"guard fired at k={row['k']} clamped={row['clamped']}: the hook "
                "did not change the output"
            )


def test_mode_ablation_runner_completes(trained):
    out = trained["tmp_path"] / "modeabl"
    res = _run(
        "mode_ablation_phase3.py",
        [
            "--checkpoint", str(trained["ckpt"]),
            "--features-csv", str(trained["table_csv"]),
            "--npz-root", str(trained["npz_root"]),
            "--table", str(trained["table_csv"]),
            "--preprocessing", str(trained["preprocessing"]),
            "--device", "cpu",
            "--n-windows", "8",
            "--k", "2",
            "--output-dir", str(out),
            "--tag", "smoke",
        ],
        out,
    )
    manifest = next(out.rglob("mode_ablation*.json"), None)
    assert manifest is not None, (
        f"mode_ablation exited {res.returncode} but wrote no manifest under {out}\n"
        f"--- stderr ---\n{res.stderr[-4000:]}"
    )
    _assert_complete(manifest, res, "mode_ablation")


# --------------------------------------------------------------------------- #
# Wiring the smoke cannot execute
# --------------------------------------------------------------------------- #

# This suite runs --device cpu, where model and batch are trivially co-located,
# so it CANNOT reproduce the CUDA/CPU mismatch that killed job 57252876. The
# behaviour of batch_to_model_device is unit-tested in test_xai_sampling.py;
# what remains unverifiable by execution here is that each runner actually CALLS
# it, between building the batch and the first model(batch). A source-level
# assertion is a weak test, but this failure mode has now cost five submissions
# and it only ever appears on hardware this machine does not have.

_RUNNERS = [
    "mode_ablation_phase3.py",
    "heldout_pruning_phase3.py",
    "symbolic_substitution_phase3.py",
]


@pytest.mark.parametrize("script", _RUNNERS)
def test_runner_moves_the_batch_onto_the_models_device(script):
    src = (REPO_ROOT / "scripts" / script).read_text()
    assert "batch_to_model_device" in src, (
        f"{script} never moves its batch to the model's device. "
        "eval_heldout_phase3.run moves the model as a side effect, so the batch "
        "built afterwards is on CPU while the model is not."
    )
    build = src.index("build_labeled_batch(")
    move = src.index("batch_to_model_device(", build)
    # The first model(batch) consumer after the batch is built must come later.
    consumers = [src.find(c, build) for c in ("collect_activations(", "fold_bundles(")]
    first = min(i for i in consumers if i != -1)
    assert build < move < first, (
        f"{script} calls batch_to_model_device out of order: it must sit between "
        "building the batch and the first model(batch) call"
    )
