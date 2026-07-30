#!/usr/bin/env bash
# Phase 2: noise metric computation.
# Edit DATASETS and SIGNALS_DIR before running.
# SIGNALS_DIR must point to the NPZ bundles produced by extract_phase2_signals.py on Great Lakes.
set -euo pipefail

DATASETS="1-5"
SIGNALS_DIR="/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/data/phase2_signals"

cd /Users/yash/Downloads/PRI2M/SINDy_DDE

/opt/anaconda3/bin/python \
  /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/wrappers/run_phase2_noise_metrics.py \
  --datasets "$DATASETS" \
  --signals-dir "$SIGNALS_DIR" \
  --output-dir /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/outputs/phase2
