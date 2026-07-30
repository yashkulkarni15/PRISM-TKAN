"""Runner-level contracts for `scripts/activation_modes_phase3.py`.

WHY THIS FILE EXISTS. The runner shipped with a progress callback that took one
argument while `activation_modes._tick` calls `progress(stage, done, total)`.
Nothing caught it, because every existing test exercises the library directly and
never binds the runner's own callback. The mismatch surfaced on Great Lakes, after
five checkpoints had been loaded, as a `TypeError` three lines into the first k.

The lesson generalises past this one bug: a runner is code, and the seam between a
runner and the library it drives needs a test like any other. These tests bind the
REAL functions from the runner module, so a signature drift on either side fails
here in milliseconds instead of on the cluster.
"""
from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path

import pytest
import torch

from pri2m.models.tkan_model import TKANModel
from pri2m.xai import activation_modes as am

# Fixtures are rebuilt here rather than imported from test_activation_modes:
# one test module importing another couples their collection order and is exactly
# the kind of seam this file exists to avoid.
_IN, _HID = 4, 6


def _batch(B: int = 4, L: int = 24, C: int = _IN, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(L, dtype=torch.float32)
    base = torch.sin(2 * math.pi * 0.05 * t)
    window = base.view(1, L, 1).repeat(B, 1, C) + 0.2 * torch.randn(B, L, C, generator=g)
    return {
        "window": window,
        "tau": torch.tensor([5, 7, 9, 11][:B], dtype=torch.long),
        "omega": torch.full((B,), 5000.0),
        "tpf": torch.full((B,), 200.0),
    }


def _model(seed: int):
    torch.manual_seed(seed)
    return TKANModel(input_size=_IN, hidden_size=_HID, num_layers=1,
                     grid_size=5, spline_order=3, side_input=False).eval()


def _load_runner():
    """Import the runner by path; `scripts/` is not a package."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "activation_modes_phase3.py"
    spec = importlib.util.spec_from_file_location("activation_modes_phase3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trained(i):
    """A synthetic model with weights moved off their init.

    `reinitialised_copy` refuses to run when re-init reproduces the trained
    weights exactly, which is what happens on a freshly constructed fixture under
    the same seed. Perturbing stands in for training and keeps the guard happy.
    """
    m = _model(seed=i)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return m


def test_runner_progress_callback_matches_the_library_contract():
    """The exact bug that reached the cluster: arity mismatch on the callback."""
    runner = _load_runner()
    sig = inspect.signature(runner._progress)
    positional = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert len(positional) == 3, (
        "activation_modes._tick calls progress(stage, done, total); the runner's "
        f"callback accepts {len(positional)} positional argument(s)"
    )
    # Callable with the real argument types, not just the right count.
    runner._progress("trained", 1, 5)


def test_analysis_actually_drives_the_runners_callback():
    """Bind the real callback to the real analysis.

    An arity assertion alone would pass if the library changed how it calls back.
    Running the analysis with the runner's own function is what pins the seam.
    """
    runner = _load_runner()
    models = {f"fold{i}": _trained(i) for i in range(2)}
    batches = {f"fold{i}": _batch() for i in range(2)}

    seen = []

    def spy(stage, done, total):
        seen.append((stage, done, total))
        runner._progress(stage, done, total)   # must not raise

    am.run_activation_mode_analysis(
        models, batches, n_seeds=3, k=2, n_points=16,
        component="spline", gate_quantile=0.9,
        threshold=0.8, min_seeds=20, seed=0, progress=spy,
    )
    assert seen, "the analysis never called back; this test would not catch a drift"
    assert {s for s, _, _ in seen} <= {"trained", "untrained"}
    assert all(isinstance(d, int) and isinstance(t, int) for _, d, t in seen)


def test_runner_provenance_is_json_safe_and_built_before_the_run():
    """Provenance must serialise on its own, before any analysis has happened.

    `tsr_control_phase3.py` burned a 12-hour wall because provenance was assembled
    at the END and crashed there, so the run could never have written a manifest
    even had it finished.
    """
    import json

    runner = _load_runner()
    prov = runner._provenance(
        checkpoints=["a/best_auc.pt", "b/best_auc.pt"], tag="t", n_windows=8,
        n_seeds=4, k_sweep=[2, 3], gate_quantile=0.9, threshold=0.8, min_seeds=20,
        component="spline", n_points=16, sample_seed=0,
    )
    json.dumps(prov)                      # raises if a field is not serialisable
    assert prov["k_sweep"] == [2, 3]
    assert prov["n_untrained_seeds"] == 4


@pytest.mark.parametrize("argv,reason", [
    (["--checkpoints", "only/one.pt", "--npz-root", "n", "--table", "t",
      "--output-dir", "o"], "one fold cannot give cross-fold agreement"),
    (["--checkpoints", "a.pt,b.pt", "--npz-root", "n", "--table", "t",
      "--output-dir", "o", "--k-sweep", ""], "an empty k sweep would run nothing"),
    (["--checkpoints", "a.pt,b.pt", "--npz-root", "n", "--table", "t",
      "--output-dir", "o", "--preprocessing", "only-one.json"],
     "preprocessing count must match checkpoint count"),
])
def test_runner_rejects_unusable_arguments_before_loading_checkpoints(argv, reason):
    """Argument errors must fire before the expensive checkpoint loads.

    The cluster failure wasted five model loads before raising. Everything
    cheaply knowable should be knowable cheaply.
    """
    runner = _load_runner()
    with pytest.raises(SystemExit):
        runner.main(argv)
