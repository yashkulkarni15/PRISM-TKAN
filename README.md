# PRISM TKAN: physics-informed chatter detection in milling

A temporal Kolmogorov–Arnold network (TKAN) that detects regenerative chatter from milling
signals. It is given the tooth-pass delay τ as an explicit input, and the analyses test whether
it actually uses that delay. This is the code behind our NAMRC paper and the PRI2M research
updates (University of Michigan, Prof. Abell's group).

All active code lives in [`SINDy_DDE/`](SINDy_DDE/README.md). The top-level `Dataset 1/`,
`Dataset 2/`, `xgboost_xai_*.py` and `mdp*.py` are earlier team baselines kept for history.

## Main results

All results below are on the held-out datasets 121–125 over five folds. The run-to-run
repeatability floor is 0.0092 AUC.

- **Detection:** TKAN reaches a held-out AUC of 0.960 ± 0.007. MiniROCKET given identical
  inputs reaches 0.962 ± 0.006, and 0.967 ± 0.005 with the delayed channels added, so the two
  models are tied.
- **Delay reliance:** both models rely on the delayed signal, and on its phase in particular.
  - Halving τ drops TKAN by 0.118 and MiniROCKET by 0.071.
  - Doubling τ, which puts the delay one full tooth period later, costs only 0.015 and 0.004.
  - That is the signature tooth-pass periodicity predicts.
- **Necessary connections:** every fold has a cluster of delay-channel edges whose deletion
  beats matched random deletions. Deleting all 77 delay edges costs 0.07–0.12 AUC.
- **Across seeds:** networks retrained from different seeds learn the same delay-edge curves
  (up to sign and scale), but not the same symbolic formulas.

## Layout (`SINDy_DDE/`)

- `src/pri2m/` is the importable package:
  - `models/`: TKAN, the KAN layers and the MiniROCKET baseline.
  - `phase3/`: data preparation, labels, splits, sample weights and the model zoo.
  - `xai/`: deletion, recurrence, delay variants, equations and the other interpretability
    tools.
  - `physics/`: Altintas–Budak stability lobes.
  - `noise/` and `spectral/`: the Phase 2 signal metrics.
- `scripts/` holds the CLI entry points:
  - `train_phase3.py`: training.
  - `eval_heldout_phase3.py` and `predict_phase3.py`: evaluation and prediction.
  - `*_phase3.py`: one script per experiment.
  - `plot_namrc_figures.py`: the paper figures.
- `slurm/` holds the Great Lakes batch wrappers, one per experiment. Each wrapper documents its
  own submit command.
- `configs/` holds `phase3_tkan.yaml`, the training configuration.
- `tests/` holds the pytest suite.
- `docs/phase3_tkan/PROGRESS.md` is the full experiment log, newest frontier first.

The raw signals, NPZ bundles, checkpoints and `results/` are not in the repo. They live on
Great Lakes under `/nfs/turbo/coe-jaabell/PRI2M`.

## Setup

```bash
cd SINDy_DDE
python -m venv .venv && source .venv/bin/activate
pip install -e ".[phase3,test]"
PYTHONPATH=src:scripts python -m pytest -q
```

MiniROCKET needs `aeon`. Install it in a separate environment with `pip install -e ".[minirocket]"`,
because aeon's numba pins an older numpy than the TKAN runs used.

## Running on Great Lakes

1. Sync the code with `./sync_phase3_tkan_to_hpc.sh`.
2. Submit a wrapper from `/nfs/turbo/coe-jaabell/PRI2M`, for example
   `sbatch --array=0-4 slurm/phase3_tkan_delay_heldout.sbatch`.
3. Fetch the results with `./fetch_phase3_results_from_hpc.sh`.

## Known pitfall

`model.pool` is a forward-time attribute and isn't saved in the checkpoint, so it must be
restored from `preprocessing.json`. Before 2026-10-06, `scripts/xai_phase3.py::_load_model`
didn't restore it, so every checkpoint loaded through it ran at pool = 1, a recurrence four
times longer than training used. Treat any TKAN number produced through that loader before the
fix as invalid.
