#!/usr/bin/env bash
set -euo pipefail

cd /Users/yash/Downloads/PRI2M/SINDy_DDE

/opt/anaconda3/bin/python \
  /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/wrappers/run_xgboost_v3_smoke.py \
  --output-dir /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/outputs/xgboost_v3_smoke
