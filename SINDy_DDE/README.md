# PRI2M SINDy-DDE / Phase 3 TKAN Workspace

This is the active PRI2M research workspace. Run commands from this directory:

```bash
cd SINDy_DDE
```

On Great Lakes, the matching project root is:

```bash
cd /nfs/turbo/coe-jaabell/PRI2M
```

(The `.../PRI2M/SINDy_DDE` subdirectory on Great Lakes is a stale copy; do not use it.)

## Layout

- `src/pri2m/` - importable project code for physics checks, spectral metrics, noise/mode labeling, reports, and training scaffolds.
- `scripts/` - command-line entrypoints for Phase 1 checks, Phase 2 PSD/JS analysis, Phase 2.5 noise calibration, mode assignment, plotting, and artifact sync.
- `slurm/` - Great Lakes batch scripts and run-order notes. Defaults write generated data to `data/` and analysis outputs to `results/`.
- `configs/` - project configuration, including local and Great Lakes roots.
- `tests/` - local regression tests for physics utilities, noise metrics, calibration, modes, Pareto scoring, and end-to-end Phase 2.5 flow.
- `data/metadata/` - lightweight PRI2M feature metadata CSVs and dataset reference documents.
- `data/phase2_signals/` - intended location for extracted NPZ bundles. This is generated on Great Lakes and is ignored by git.
- `results/` - copied or generated analysis outputs, plots, reports, and MLflow artifacts. This is local/output material and is ignored by git.
- `sindy_pipeline/` - legacy SINDy-DDE preprocessing, fitting, analysis scripts, logs, and lightweight provenance CSVs.
- `archive/` - stale, unrelated, or bulky local material kept out of the active project surface.

## Current Phase

Phase 3 (TKAN) is active. The model is a temporal KAN that takes the tooth-pass delay τ as an input. Five-fold training is done, and so are the NAMRC paper experiments:
- group and pairwise deletion of the delay edges,
- seed-retrain recurrence,
- the MiniROCKET comparison and delay-damage test,
- equations across seeds.

The results and the paper-figure script (`scripts/plot_namrc_figures.py`) are summarised in the top-level README. The full experiment log is `docs/phase3_tkan/PROGRESS.md`.

Phases 1–2.5 (physics checks, PSD/JS spectral analysis, noise calibration and mode labelling) are complete. Their code remains under `src/pri2m/{physics,spectral,noise}` and the matching scripts.
