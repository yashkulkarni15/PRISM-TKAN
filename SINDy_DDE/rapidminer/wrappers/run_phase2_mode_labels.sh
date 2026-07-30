#!/usr/bin/env bash
# Phase 2: mode label assignment.
# Reads noise_metrics.csv produced by run_phase2_noise_metrics.sh.
set -euo pipefail

cd /Users/yash/Downloads/PRI2M/SINDy_DDE

/opt/anaconda3/bin/python \
  /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/wrappers/run_phase2_mode_labels.py \
  --noise-metrics-csv /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/outputs/phase2/noise_metrics.csv \
  --output-dir /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/outputs/phase2
