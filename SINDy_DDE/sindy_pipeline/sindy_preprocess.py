"""
sindy_preprocess.py

Per file pipeline:
  1.  Load h5 + stability boundary
  2.  Reject if raw length < 3000
  3.  Reject if NaN/Inf in xt_x, xt_vel_x, xt_acc_x
  4.  Trim transient (RMS for stable, RMS-or-percentage for unstable)
  5.  Reject if trimmed length < 3000
  6.  Reject if max|xt_acc_x| > 1e6
  7.  Build state matrix (X, y, scales, delay info)
  8.  Reject if post-delay length < 2000
  9.  Reject if |corr(x, x_delay)| > 0.98
  10. Save pickle to OUTPUT_ROOT/state_{X}_{Y}.pkl
  11. Log rejections to rejection_log.csv

Usage on Great Lakes:
    python sindy_preprocess.py --dataset_start 1 --dataset_end 60
    python sindy_preprocess.py --dataset_start 1 --dataset_end 120 --include_y

Notebook usage:
    Set TEST_MODE = True at top, configure TEST_DATASET / TEST_FILE_Y, run main script.
"""

import argparse
import os
import pickle
import re
import time

import numpy as np
import pandas as pd
import h5py

from read_file import load_time_h5
from RMSsliding import trim_transient_sliding_rms
from build_state import build_state_matrix, TEETH_MAP


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DATA_VERSION = os.environ.get("DATA_VERSION", "10")
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    f"/nfs/turbo/coe-jaabell/datasets/tonylschmitz/digital-machining-database/versions/{DATA_VERSION}",
)
OUTPUT_ROOT = "/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline/preprocessed"
LOG_PATH = "/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline/rejection_log.csv"
SUCCESS_LOG_PATH = "/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline/preprocessed_log.csv"

MIN_RAW_LENGTH = 3000
MIN_TRIMMED_LENGTH = 3000
MIN_DELAY_LENGTH = 2000
MAX_ACC_AMPLITUDE = 1e6
COLLINEARITY_THRESHOLD = 0.98
STABLE_THRESHOLD = 0.8

# RMS trim params as fractions of signal duration (fs-agnostic)
SUSTAIN_FRAC = 0.05
WINDOW_FRAC = 0.02
TOL_RATIO = 0.01
CHANGE_TOL_RATIO = 0.003

# Percentage-trim fallback: discard the first 15 percent of the signal
PCT_TRIM_FRAC = 0.15

# Notebook test flags. Ignored when running with argparse.
TEST_MODE = False
TEST_DATASET = 1
TEST_FILE_Y = 1


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def list_time_files(dataset):
    """Return sorted list of file_y indices present for a given dataset."""
    folder = os.path.join(DATA_ROOT, f"Dataset {dataset} h5")
    if not os.path.isdir(folder):
        return []
    pattern = re.compile(rf"^time{dataset}_(\d+)\.h5$")
    ys = []
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            ys.append(int(m.group(1)))
    return sorted(ys)


def percentage_trim(signal_length, frac=PCT_TRIM_FRAC):
    """Return a fixed-percentage start index for the transient trim."""
    return int(round(signal_length * frac))


def rms_trim_index(signal, fs):
    """
    Wrap trim_transient_sliding_rms to return a start index regardless of
    whether the underlying function returns an int index or a trimmed array.

    Assumed signature:
        trim_transient_sliding_rms(signal, fs,
                                   sustain_sec, window_sec,
                                   tol_ratio, change_tol_ratio)
    Adjust this wrapper if the real signature differs.
    """
    n = len(signal)
    duration = n / fs
    sustain_sec = duration * SUSTAIN_FRAC
    window_sec = duration * WINDOW_FRAC

    result = trim_transient_sliding_rms(
        signal, fs,
        sustain_sec=sustain_sec,
        window_sec=window_sec,
        tol_ratio=TOL_RATIO,
        change_tol_ratio=CHANGE_TOL_RATIO,
    )

    if isinstance(result, (int, np.integer)):
        return int(result)

    arr = np.asarray(result)
    return n - len(arr)


def reject(log_rows, dataset, file_y, reason, raw_length=None, b_over_blim=None):
    log_rows.append({
        "dataset": dataset,
        "file_y": file_y,
        "reason": reason,
        "raw_length": raw_length,
        "b_over_blim": b_over_blim,
    })


def write_log(log_rows, path=LOG_PATH):
    """Append rejection rows to the cumulative log CSV."""
    if not log_rows:
        return
    df = pd.DataFrame(log_rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


def write_success_log(success_rows, path=SUCCESS_LOG_PATH):
    """Append success rows to the cumulative success log CSV."""
    if not success_rows:
        return
    df = pd.DataFrame(success_rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


def load_sb_direct(sb_path):
    """
    Load a stability boundary h5 file directly via h5py.

    The file contains one dataset of shape (N, 2): col 0 = omega, col 1 = blim.
    Returns (omega_arr, blim_arr) as 1-D numpy arrays.
    """
    with h5py.File(sb_path, "r") as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"No datasets in {sb_path}")
        arr = np.asarray(f[keys[0]])
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) array in {sb_path}, got shape {arr.shape}")
    return arr[:, 0], arr[:, 1]


def trim_sim_dict(sim, trim_start):
    """
    Return a copy of the sim dict with every 1-D array sliced from
    trim_start onwards. Scalars and non-array values pass through unchanged.
    """
    out = {}
    for k, v in sim.items():
        if isinstance(v, np.ndarray) and v.ndim == 1:
            out[k] = v[trim_start:]
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------
# Per-file pipeline
# ----------------------------------------------------------------------

def process_file(dataset, file_y, include_y=False, log_rows=None):
    """
    Run the 11-step pipeline on one file.

    Returns
    -------
    ("saved", output_path) on success
    ("rejected", reason)   on rejection
    """
    if log_rows is None:
        log_rows = []

    # ---- Step 1: Load ----
    folder = os.path.join(DATA_ROOT, f"Dataset {dataset} h5")
    time_path = os.path.join(folder, f"time{dataset}_{file_y}.h5")
    sb_path = os.path.join(folder, f"stability_boundary{dataset}.h5")

    if not os.path.exists(time_path):
        reject(log_rows, dataset, file_y, "time_file_missing")
        return ("rejected", "time_file_missing")
    if not os.path.exists(sb_path):
        reject(log_rows, dataset, file_y, "sb_file_missing")
        return ("rejected", "sb_file_missing")

    try:
        data = load_time_h5(time_path)
    except Exception as e:
        reject(log_rows, dataset, file_y, f"load_failed:{type(e).__name__}")
        return ("rejected", "load_failed")

    if not isinstance(data, dict):
        reject(log_rows, dataset, file_y, "bad_shape")
        return ("rejected", "bad_shape")

    required_keys = ["omega", "b", "t", "xt_x", "xt_vel_x", "xt_acc_x"]
    if include_y:
        required_keys += ["xt_y", "xt_vel_y", "xt_acc_y"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        reject(log_rows, dataset, file_y, f"missing_keys:{','.join(missing)}")
        return ("rejected", "missing_keys")

    omega = float(data["omega"])
    b = float(data["b"])
    t = np.asarray(data["t"])
    xt_x = np.asarray(data["xt_x"])
    xt_vel_x = np.asarray(data["xt_vel_x"])
    xt_acc_x = np.asarray(data["xt_acc_x"])

    if include_y:
        xt_y = np.asarray(data["xt_y"])
        xt_vel_y = np.asarray(data["xt_vel_y"])
        xt_acc_y = np.asarray(data["xt_acc_y"])

    try:
        sb_omega, sb_blim = load_sb_direct(sb_path)
    except Exception as e:
        reject(log_rows, dataset, file_y, f"sb_load_failed:{type(e).__name__}")
        return ("rejected", "sb_load_failed")

    blim = float(np.interp(omega, np.asarray(sb_omega), np.asarray(sb_blim)))
    b_over_blim = b / blim if blim > 0 else np.nan
    raw_length = len(t)

    # ---- Step 2: Guard 1 -- raw length ----
    if raw_length < MIN_RAW_LENGTH:
        reject(log_rows, dataset, file_y, "raw_too_short", raw_length, b_over_blim)
        return ("rejected", "raw_too_short")

    # ---- Step 3: Guard 2 -- NaN/Inf ----
    finite_x = np.isfinite(xt_x).all()
    finite_v = np.isfinite(xt_vel_x).all()
    finite_a = np.isfinite(xt_acc_x).all()
    if not (finite_x and finite_v and finite_a):
        reject(log_rows, dataset, file_y, "non_finite", raw_length, b_over_blim)
        return ("rejected", "non_finite")
    if include_y:
        if not (np.isfinite(xt_y).all()
                and np.isfinite(xt_vel_y).all()
                and np.isfinite(xt_acc_y).all()):
            reject(log_rows, dataset, file_y, "non_finite_y", raw_length, b_over_blim)
            return ("rejected", "non_finite_y")

    # ---- Step 4: Trim transient ----
    if "dt" in data and data["dt"] is not None:
        dt = float(data["dt"])
    else:
        dt = float(np.median(np.diff(t)))

    if "fs" in data and data["fs"] is not None:
        fs = float(data["fs"])
    else:
        fs = 1.0 / dt if dt > 0 else 0.0

    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(fs) or fs <= 0:
        reject(log_rows, dataset, file_y, "bad_dt", raw_length, b_over_blim)
        return ("rejected", "bad_dt")

    if not np.isfinite(b_over_blim):
        reject(log_rows, dataset, file_y, "blim_invalid", raw_length, b_over_blim)
        return ("rejected", "blim_invalid")

    is_stable = b_over_blim < STABLE_THRESHOLD

    if is_stable:
        # Stable: try RMS, fall back to percentage if too short
        try:
            start_rms = rms_trim_index(xt_x, fs)
        except Exception:
            start_rms = raw_length

        if (raw_length - start_rms) >= MIN_TRIMMED_LENGTH:
            trim_start = start_rms
            trim_method = "rms"
        else:
            trim_start = percentage_trim(raw_length)
            trim_method = "percentage_fallback"
    else:
        # Unstable: try both, keep whichever leaves more samples
        try:
            start_rms = rms_trim_index(xt_x, fs)
        except Exception:
            start_rms = raw_length
        start_pct = percentage_trim(raw_length)

        if (raw_length - start_rms) >= (raw_length - start_pct):
            trim_start = start_rms
            trim_method = "rms"
        else:
            trim_start = start_pct
            trim_method = "percentage"

    t_trim = t[trim_start:]
    xt_x_trim = xt_x[trim_start:]
    xt_vel_x_trim = xt_vel_x[trim_start:]
    xt_acc_x_trim = xt_acc_x[trim_start:]

    if include_y:
        xt_y_trim = xt_y[trim_start:]
        xt_vel_y_trim = xt_vel_y[trim_start:]
        xt_acc_y_trim = xt_acc_y[trim_start:]

    # ---- Step 5: Guard 3 -- post-trim length ----
    if len(t_trim) < MIN_TRIMMED_LENGTH:
        reject(log_rows, dataset, file_y,
               f"trimmed_too_short:{trim_method}", raw_length, b_over_blim)
        return ("rejected", "trimmed_too_short")

    # ---- Step 6: Guard 4 -- amplitude check ----
    if np.max(np.abs(xt_acc_x_trim)) > MAX_ACC_AMPLITUDE:
        reject(log_rows, dataset, file_y, "acc_runaway", raw_length, b_over_blim)
        return ("rejected", "acc_runaway")

    # ---- Step 6b: Guard -- flat signal check ----
    # build_state_matrix normalizes by std and raises ValueError if any
    # column has zero std. Catch that here with a clearer reason so the
    # log shows "flat_x" instead of an opaque state_build_failed.
    if np.std(xt_x_trim) < 1e-12:
        reject(log_rows, dataset, file_y, "flat_x", raw_length, b_over_blim)
        return ("rejected", "flat_x")
    if np.std(xt_vel_x_trim) < 1e-12:
        reject(log_rows, dataset, file_y, "flat_dxdt", raw_length, b_over_blim)
        return ("rejected", "flat_dxdt")
    if np.std(xt_acc_x_trim) < 1e-12:
        reject(log_rows, dataset, file_y, "flat_d2xdt2", raw_length, b_over_blim)
        return ("rejected", "flat_d2xdt2")
    if include_y:
        if np.std(xt_y_trim) < 1e-12:
            reject(log_rows, dataset, file_y, "flat_y", raw_length, b_over_blim)
            return ("rejected", "flat_y")

    # ---- Step 7: Build state matrix ----
    teeth = TEETH_MAP.get(dataset, 4)
    trimmed_sim = trim_sim_dict(data, trim_start)

    try:
        state = build_state_matrix(trimmed_sim, dataset, include_y=include_y)
    except Exception as e:
        reject(log_rows, dataset, file_y,
               f"state_build_failed:{type(e).__name__}", raw_length, b_over_blim)
        return ("rejected", "state_build_failed")

    # Normalize return to dict form so downstream code is uniform.
    if isinstance(state, dict) and "X" in state and "y" in state:
        pass
    elif isinstance(state, tuple) and len(state) >= 2:
        X_arr = np.asarray(state[0])
        y_arr = np.asarray(state[1])
        extras = state[2] if len(state) > 2 and isinstance(state[2], dict) else {}
        state = {"X": X_arr, "y": y_arr, **extras}
    else:
        reject(log_rows, dataset, file_y, "state_bad_format", raw_length, b_over_blim)
        return ("rejected", "state_bad_format")

    X = np.asarray(state["X"])
    y = np.asarray(state["y"])

    # ---- Step 8: Guard 5 -- post-delay length ----
    if len(X) < MIN_DELAY_LENGTH:
        reject(log_rows, dataset, file_y, "delay_too_short", raw_length, b_over_blim)
        return ("rejected", "delay_too_short")

    # ---- Step 9: Guard 6 -- collinearity ----
    # State columns per build_state.py: 0 = x(t), 1 = dx/dt, 2 = chip = x(t) - x(t-tau)
    # Recover x_delay = x(t) - chip and check correlation with x(t).
    x_col = X[:, 0]
    x_delay = x_col - X[:, 2]
    if np.std(x_col) == 0 or np.std(x_delay) == 0:
        reject(log_rows, dataset, file_y, "zero_variance", raw_length, b_over_blim)
        return ("rejected", "zero_variance")
    corr = abs(np.corrcoef(x_col, x_delay)[0, 1])
    if not np.isfinite(corr) or corr > COLLINEARITY_THRESHOLD:
        reject(log_rows, dataset, file_y, f"collinear:{corr:.4f}",
               raw_length, b_over_blim)
        return ("rejected", "collinear")

    # ---- Step 10: Save pickle ----
    out_dict = {
        "X": X,
        "y": y,
        "t": np.asarray(state.get("t", t_trim[-len(X):])),
        "scales": state.get("scales", {}),
        "delay_samples": state.get("delay_samples"),
        "tau_sec": state.get("tau_sec"),
        "teeth": teeth,
        "dt": state.get("dt", dt),
        "omega": omega,
        "b": b,
        "blim": blim,
        "b_over_blim": b_over_blim,
        "dataset": dataset,
        "file_y": file_y,
        "trim_method": trim_method,
        "trim_start_index": int(trim_start),
        "raw_length": raw_length,
        "trimmed_length": len(t_trim),
        "include_y": include_y,
        "x_delay_corr": float(corr),
    }

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUTPUT_ROOT, f"state_{dataset}_{file_y}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Build a flat metadata row for the success log.
    label = int(b_over_blim >= 1.0) if np.isfinite(b_over_blim) else -1
    success_row = {
        "dataset": dataset,
        "file_y": file_y,
        "label": label,
        "omega": omega,
        "b": b,
        "blim": blim,
        "b_over_blim": b_over_blim,
        "teeth": teeth,
        "tau_sec": out_dict["tau_sec"],
        "delay_samples": out_dict["delay_samples"],
        "dt": out_dict["dt"],
        "raw_length": raw_length,
        "trimmed_length": len(t_trim),
        "trim_method": trim_method,
        "trim_start_index": int(trim_start),
        "X_rows": int(X.shape[0]),
        "X_cols": int(X.shape[1]),
        "x_delay_corr": float(corr),
        "include_y": include_y,
        "pickle_path": out_path,
    }

    return ("saved", out_path, success_row)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main(dataset_start, dataset_end, include_y=False, verbose=True):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    saved = 0
    rejected_by_reason = {}
    started = time.time()

    for ds in range(dataset_start, dataset_end + 1):
        ys = list_time_files(ds)
        if not ys:
            if verbose:
                print(f"[ds={ds}] no time files found, skipping")
            continue

        log_rows = []
        success_rows = []
        for fy in ys:
            result = process_file(ds, fy, include_y=include_y, log_rows=log_rows)
            status = result[0]
            info = result[1]
            success_meta = result[2] if len(result) > 2 else None

            if status == "saved":
                saved += 1
                if success_meta is not None:
                    success_rows.append(success_meta)
                if verbose:
                    print(f"[ds={ds} fy={fy}] saved -> {info}")
            else:
                rejected_by_reason[info] = rejected_by_reason.get(info, 0) + 1
                if verbose:
                    print(f"[ds={ds} fy={fy}] REJECTED: {info}")

        # Flush per dataset so long runs survive crashes
        write_log(log_rows)
        write_success_log(success_rows)

    elapsed = time.time() - started
    print("=" * 60)
    print(f"Datasets processed: {dataset_start}..{dataset_end}")
    print(f"Files saved:        {saved}")
    print(f"Files rejected:     {sum(rejected_by_reason.values())}")
    for reason, count in sorted(rejected_by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:30s} {count}")
    print(f"Elapsed:            {elapsed:.1f} s")
    print(f"Output dir:         {OUTPUT_ROOT}")
    print(f"Rejection log:      {LOG_PATH}")
    print(f"Success log:        {SUCCESS_LOG_PATH}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if TEST_MODE:
        log_rows = []
        result = process_file(
            TEST_DATASET, TEST_FILE_Y, include_y=False, log_rows=log_rows
        )
        status = result[0]
        info = result[1]
        success_meta = result[2] if len(result) > 2 else None
        print(f"TEST_MODE result: status={status}, info={info}")
        write_log(log_rows)
        if success_meta is not None:
            write_success_log([success_meta])
    else:
        parser = argparse.ArgumentParser(
            description="Preprocess milling data for SINDy-DDE."
        )
        parser.add_argument("--dataset_start", type=int, default=1)
        parser.add_argument("--dataset_end", type=int, default=120)
        parser.add_argument("--include_y", action="store_true",
                            help="Also include y-direction signals.")
        parser.add_argument("--quiet", action="store_true",
                            help="Suppress per-file status output.")
        args = parser.parse_args()
        main(
            args.dataset_start,
            args.dataset_end,
            include_y=args.include_y,
            verbose=not args.quiet,
        )
