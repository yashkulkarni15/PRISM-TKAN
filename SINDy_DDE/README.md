# PRI2M SINDy-DDE / Phase 2.5 Workspace

This is the active PRI2M research workspace. Run commands from this directory:

```bash
cd /Users/yash/Downloads/PRI2M/SINDy_DDE
```

On Great Lakes, the matching project root is:

```bash
cd /nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE
```

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

Phase 2.5 is the active bridge between spectral PSD/JS work and the future T-KAN-PINN. It computes multi-family noise metrics, calibrates clean/noisy mirror ladders, assigns signal modes, and prepares the filter/Pareto workflow.

The current production next step is still the pooled mirror calibration over the full clean/noisy ladder on Great Lakes after the running extraction and metrics jobs finish.
