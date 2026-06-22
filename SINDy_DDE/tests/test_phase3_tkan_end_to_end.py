"""Keystone end-to-end test for the Phase 3 TKAN training pipeline.

Synthesizes a tiny NPZ tree + per-file table + leakage-safe folds under a pytest
tmp directory, then runs ``scripts/train_phase3.py --smoke`` as a subprocess for
both the MLP baseline and the TKAN model. Asserts:

1. each model's run exits 0 and writes a ``summary.json`` parsing to a dict with
   ``"val"`` and ``"gate"`` keys, plus a ``final.pt`` checkpoint under the run dir;
2. determinism: two MLP smoke runs with the same seed/args produce the same final
   validation ``mae`` (within 1e-6);
3. the Great Lakes wrapper ``slurm/phase3_train.sbatch`` is syntax-clean
   (``bash -n``).

The test self-contains every artifact and does not touch Great Lakes or any data
under the repo; it only depends on the synthetic builders in ``tests._phase3_synth``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests._phase3_synth import make_synth_folds, make_synth_npz_tree, make_synth_table

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_phase3.py"
SBATCH = REPO_ROOT / "slurm" / "phase3_train.sbatch"


def _build_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write the synth NPZ tree + table CSV + folds JSON; return their paths."""
    npz_root = make_synth_npz_tree(tmp_path / "npz", clean_ids=(1, 2, 3), n_files=4, length=256, seed=0)
    table = make_synth_table(npz_root)
    folds = make_synth_folds(table, n_splits=3, seed=0)

    table_csv = tmp_path / "table.csv"
    table.to_csv(table_csv, index=False)
    folds_json = tmp_path / "folds.json"
    folds_json.write_text(json.dumps(folds))
    return npz_root, table_csv, folds_json


def _run_train(
    *, npz_root: Path, table_csv: Path, folds_json: Path, output_dir: Path, mlflow_dir: Path,
    model: str, fold: int = 0,
) -> subprocess.CompletedProcess:
    """Invoke scripts/train_phase3.py --smoke as a subprocess from the repo root."""
    cmd = [
        sys.executable, "scripts/train_phase3.py",
        "--model", model,
        "--fold", str(fold),
        "--npz-root", str(npz_root),
        "--table", str(table_csv),
        "--folds-json", str(folds_json),
        "--output-dir", str(output_dir),
        "--mlflow-uri", f"file:{mlflow_dir}",
        "--smoke",
    ]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


def _load_summary(output_dir: Path) -> dict:
    """Find the single run's summary.json under the output dir and parse it."""
    summaries = list(output_dir.rglob("summary.json"))
    assert len(summaries) == 1, f"expected exactly one summary.json under {output_dir}, got {summaries}"
    return json.loads(summaries[0].read_text())


def test_phase3_train_end_to_end_both_models(tmp_path: Path) -> None:
    npz_root, table_csv, folds_json = _build_inputs(tmp_path)

    for model in ("mlp", "tkan"):
        out_dir = tmp_path / f"out_{model}"
        mlflow_dir = tmp_path / f"mlruns_{model}"
        res = _run_train(
            npz_root=npz_root, table_csv=table_csv, folds_json=folds_json,
            output_dir=out_dir, mlflow_dir=mlflow_dir, model=model,
        )
        assert res.returncode == 0, (
            f"train_phase3.py failed for model={model!r}\n"
            f"--- stderr ---\n{res.stderr}\n--- stdout ---\n{res.stdout}"
        )

        # A summary.json must exist and parse to a dict with val + gate keys.
        summary = _load_summary(out_dir)
        assert isinstance(summary, dict), f"summary for {model} is not a dict: {type(summary)}"
        assert "val" in summary, f"summary for {model} missing 'val' key: {sorted(summary)}"
        assert "gate" in summary, f"summary for {model} missing 'gate' key: {sorted(summary)}"

        # A final.pt checkpoint must exist under the run dir.
        final_ckpts = list(out_dir.rglob("final.pt"))
        assert final_ckpts, f"no final.pt checkpoint under {out_dir} for model={model!r}"


def test_phase3_train_is_deterministic(tmp_path: Path) -> None:
    """Two MLP smoke runs with the same seed/args yield the same final val MAE."""
    npz_root, table_csv, folds_json = _build_inputs(tmp_path)

    maes: list[float] = []
    for run_idx in range(2):
        out_dir = tmp_path / f"det_out_{run_idx}"
        mlflow_dir = tmp_path / f"det_mlruns_{run_idx}"
        res = _run_train(
            npz_root=npz_root, table_csv=table_csv, folds_json=folds_json,
            output_dir=out_dir, mlflow_dir=mlflow_dir, model="mlp",
        )
        assert res.returncode == 0, (
            f"deterministic mlp run {run_idx} failed\n"
            f"--- stderr ---\n{res.stderr}\n--- stdout ---\n{res.stdout}"
        )
        summary = _load_summary(out_dir)
        mae = summary["val"].get("mae")
        assert mae is not None, f"run {run_idx} summary['val'] has no 'mae': {summary['val']}"
        maes.append(float(mae))

    assert abs(maes[0] - maes[1]) < 1e-6, (
        f"non-deterministic final val mae across identical smoke runs: {maes[0]} vs {maes[1]}"
    )


def test_phase3_sbatch_is_syntax_clean() -> None:
    """The Great Lakes wrapper must pass `bash -n`."""
    res = subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True, text=True)
    assert res.returncode == 0, f"bash -n failed for {SBATCH}:\n{res.stderr}"
