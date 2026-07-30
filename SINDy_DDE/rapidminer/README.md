# RapidMiner: PRI2M Phase 1 and Phase 2

This folder contains all wrappers, outputs, and RapidMiner process files for the
PRI2M pipeline. Each phase is one `Execute Program` operator that calls a shell
script, which calls a Python wrapper, which calls the underlying pipeline script.
RapidMiner does orchestration and display. Python does all the work.

---

## Process files

| File | Description |
|---|---|
| `xgboost.rmp` | Smoke test: XGBoost v3 on clean datasets 1-5 |
| `phase1_phase2.rmp` | Phase 1 + Phase 2 six-operator chain (this document) |

---

## Six-operator chain

```
Op1: Altintas-Budak verifier   -> physics_check.csv
Op2: Signal extraction          -> NPZ bundles + manifest  (HPC only, not in RapidMiner)
Op3: PSD summary                -> psd_summary.csv          (HPC only, not in RapidMiner)
Op4: JS aggregation             -> js_divergence_*.csv      (HPC only, not in RapidMiner)
Op5: Noise metric computation   -> noise_metrics.csv        (needs NPZ from Op2)
Op6: Mode label assignment      -> mode_labels.csv          (reads noise_metrics.csv)
```

Operators 2-4 run on Great Lakes and feed their CSVs into Op5 and Op6.
For a laptop demo, point Op5 at a cluster-synced NPZ directory or supply a
pre-computed `noise_metrics.csv` directly to Op6.

---

## Wrappers

| Wrapper | Wraps | Output |
|---|---|---|
| `run_phase1_altintas_budak.py` | `scripts/verify_altintas_budak.py` | `outputs/phase1/physics_check.csv` |
| `run_phase2_noise_metrics.py`  | `scripts/compute_noise_metrics.py` | `outputs/phase2/noise_metrics.csv` |
| `run_phase2_mode_labels.py`    | `scripts/assign_signal_modes.py`   | `outputs/phase2/mode_labels.csv` |

Each wrapper has a matching `.sh` driver with the correct paths hardcoded.

---

## CLI test (do this before opening RapidMiner)

Run each wrapper from the terminal first to confirm the output CSV is written.
RapidMiner must see the CSV before you connect the Read CSV operator.

**Op1 - runs today, no cluster dependency:**
```bash
bash /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/wrappers/run_phase1_altintas_budak.sh
# Expected: rapidminer/outputs/phase1/physics_check.csv  (1 row, physics_pass=TRUE)
```

**Op5 - requires cluster NPZ bundles in SIGNALS_DIR:**
```bash
# Edit SIGNALS_DIR in run_phase2_noise_metrics.sh first, then:
bash /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/wrappers/run_phase2_noise_metrics.sh
# Expected: rapidminer/outputs/phase2/noise_metrics.csv
```

**Op6 - requires noise_metrics.csv from Op5:**
```bash
bash /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/wrappers/run_phase2_mode_labels.sh
# Expected: rapidminer/outputs/phase2/mode_labels.csv
```

---

## RapidMiner process setup (phase1_phase2.rmp)

The `.rmp` file is pre-built and ready to open. If you need to rebuild it manually:

1. File > New Process
2. Add operators left to right: `Execute Program`, `Read CSV`, `Execute Program`, `Read CSV`, `Execute Program`, `Read CSV`
3. Configure each Execute Program with the matching `.sh` path (see table below)
4. Configure each Read CSV with the matching output CSV path
5. Connect: Execute Program out -> Read CSV in, daisy-chained
6. Connect final Read CSV out -> result port

| Operator | Command | Read CSV path |
|---|---|---|
| Execute Program (Op1) | `.../run_phase1_altintas_budak.sh` | `.../outputs/phase1/physics_check.csv` |
| Execute Program (Op5) | `.../run_phase2_noise_metrics.sh` | `.../outputs/phase2/noise_metrics.csv` |
| Execute Program (Op6) | `.../run_phase2_mode_labels.sh` | `.../outputs/phase2/mode_labels.csv` |

Set `log_stdout = true` and `log_stderr = true` on every Execute Program operator.
Set `working_directory = /Users/yash/Downloads/PRI2M/SINDy_DDE` on every Execute Program operator.

---

## XGBoost smoke test (original)

```bash
bash /Users/yash/Downloads/PRI2M/SINDy_DDE/rapidminer/wrappers/run_xgboost_v3_smoke.sh
```

---

## Python environment

All wrappers use `/opt/anaconda3/bin/python`. That interpreter must have:
```
pandas numpy scipy xgboost scikit-learn pywavelets
```
plus the repo installed as `pip install -e . --break-system-packages` from the
`SINDy_DDE` root.
