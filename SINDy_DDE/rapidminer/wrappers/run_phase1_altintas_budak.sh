#!/usr/bin/env bash
# Phase 1: Altintas-Budak physics verifier.
# Run this from the RapidMiner Execute Program operator or directly from the CLI.
set -euo pipefail

cd /Users/yash/Downloads/PRI2M/SINDy_DDE

/opt/anaconda3/bin/python \
  /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/wrappers/run_phase1_altintas_budak.py \
  --output-dir /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/outputs/phase1 \
  --mode all
