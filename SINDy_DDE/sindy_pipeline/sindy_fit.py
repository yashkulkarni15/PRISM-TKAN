"""
sindy_fit.py

Fit SINDy-DDE per file using the preprocessed pickles in preprocessed/.

Per file pipeline:
  1. Load pickle.
  2. Inject feature_names (deterministic from include_y).
  3. Run silent lambda sweep over 30 log-spaced thresholds in [0.01, 1.0].
  4. Detect plateaus (contiguous runs of equal surviving-term counts).
  5. Pick the threshold from the middle of the longest plateau in a
     physically reasonable term-count range. Falls back to longest
     plateau overall if no in-range plateau exists.
  6. Fit at the chosen threshold via fit_sindy_dde.
  7. Append a flat row to a per-dataset CSV with metadata, threshold,
     plateau diagnostics, scales, and one column per library term
     coefficient (zero where the term was pruned).

Outputs:
  sindy_fits/dataset_{X}.csv          one row per successful fit
  sindy_fit_failures.csv              one row per file that errored

Usage on Great Lakes:
  python sindy_fit.py --dataset_start 1 --dataset_end 20
  python sindy_fit.py --dataset_start 21 --dataset_end 40
  ...

Notebook test:
  Set TEST_MODE = True, configure TEST_DATASET / TEST_FILE_Y, run.
"""

import argparse
import glob
import os
import pickle
import re
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import pysindy as ps

from sindy_dde import fit_sindy_dde


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

PREPROCESSED_DIR = "/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline/preprocessed"
FITS_DIR         = "/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline/sindy_fits"
FAILURE_LOG      = "/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline/sindy_fit_failures.csv"


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DEGREE = 2

# Lambda sweep parameters (matches lambda_sweep defaults in build_state.py)
N_LAMBDAS  = 30
LAMBDA_MIN = 1e-2
LAMBDA_MAX = 1.0

# Plateau selection
MIN_TERMS_ACCEPT = 2     # below: under-fit
MAX_TERMS_ACCEPT = 8     # above: over-fit
MIN_PLATEAU_LEN  = 2     # plateaus shorter than this ignored

# Notebook test
TEST_MODE    = False
TEST_DATASET = 1
TEST_FILE_Y  = 1


# ----------------------------------------------------------------------
# Feature names
# ----------------------------------------------------------------------

FEATURE_NAMES_X_ONLY = ["x", "dxdt", "chip_x"]
FEATURE_NAMES_X_Y    = ["x", "dxdt", "chip_x", "y", "dydt", "chip_y"]


def get_feature_names(include_y):
    return FEATURE_NAMES_X_Y if include_y else FEATURE_NAMES_X_ONLY


# ----------------------------------------------------------------------
# Silent lambda sweep
# ----------------------------------------------------------------------

def lambda_sweep_quiet(X, y, dt, feature_names, degree=DEGREE, lambdas=None):
    """
    Run the SINDy sparsity sweep without printing.
    Returns (lambdas, n_terms_per_lambda).
    """
    if lambdas is None:
        lambdas = np.logspace(np.log10(LAMBDA_MIN),
                              np.log10(LAMBDA_MAX),
                              N_LAMBDAS)
    n_terms = np.zeros(len(lambdas), dtype=int)

    for i, lam in enumerate(lambdas):
        optimizer = ps.STLSQ(threshold=lam)
        lib       = ps.PolynomialLibrary(degree=degree, include_bias=False)
        model     = ps.SINDy(optimizer=optimizer, feature_library=lib)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, t=dt, x_dot=y.reshape(-1, 1),
                      feature_names=feature_names)
        coefs = model.coefficients().flatten()
        n_terms[i] = int(np.sum(np.abs(coefs) > 0))

    return lambdas, n_terms


# ----------------------------------------------------------------------
# Plateau detection and threshold selection
# ----------------------------------------------------------------------

def find_plateaus(n_terms):
    """Return list of (start_idx, end_idx, n_terms_in_run)."""
    plateaus = []
    i = 0
    while i < len(n_terms):
        j = i
        while j < len(n_terms) and n_terms[j] == n_terms[i]:
            j += 1
        plateaus.append((i, j - 1, int(n_terms[i])))
        i = j
    return plateaus


def select_threshold(lambdas, n_terms,
                     min_terms=MIN_TERMS_ACCEPT,
                     max_terms=MAX_TERMS_ACCEPT,
                     min_len=MIN_PLATEAU_LEN):
    """
    Pick the threshold from the longest plateau in the [min_terms, max_terms]
    range. Falls back to the longest plateau overall if no in-range plateau
    of length >= min_len exists.

    Returns (threshold, info_dict).
    threshold is None if no usable plateau exists.
    """
    plateaus = find_plateaus(n_terms)

    in_range = [
        p for p in plateaus
        if min_terms <= p[2] <= max_terms
        and (p[1] - p[0] + 1) >= min_len
    ]

    info = {
        "n_plateaus": len(plateaus),
        "plateaus":   plateaus,
        "fallback":   False,
    }

    if in_range:
        # Longest plateau, ties broken toward fewer terms (parsimony)
        in_range.sort(key=lambda p: (-(p[1] - p[0] + 1), p[2]))
        best = in_range[0]
    else:
        # Fallback: longest plateau overall, even if outside [min,max]
        plateaus.sort(key=lambda p: -(p[1] - p[0] + 1))
        best = plateaus[0] if plateaus else None
        info["fallback"] = True

    if best is None:
        info["chosen_terms"] = None
        info["chosen_plateau_len"] = 0
        return None, info

    mid_idx = (best[0] + best[1]) // 2
    info["chosen_terms"]       = best[2]
    info["chosen_plateau_len"] = best[1] - best[0] + 1
    info["chosen_lambda_idx"]  = mid_idx
    return float(lambdas[mid_idx]), info


# ----------------------------------------------------------------------
# Per-file fit
# ----------------------------------------------------------------------

def fit_one_file(pkl_path):
    """
    Load a pickle, run the sweep, pick the threshold, fit, and
    return a flat row dict ready for CSV.
    Raises on any error so the caller can log it.
    """
    with open(pkl_path, "rb") as f:
        state = pickle.load(f)

    state["feature_names"] = get_feature_names(state.get("include_y", False))

    lambdas, n_terms = lambda_sweep_quiet(
        state["X"], state["y"], state["dt"], state["feature_names"],
        degree=DEGREE,
    )

    threshold, info = select_threshold(lambdas, n_terms)

    if threshold is None:
        raise ValueError("no usable plateau in lambda sweep")

    # Fit at the chosen threshold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_sindy_dde(state, threshold=threshold, degree=DEGREE)

    term_names = result["term_names"]
    coefs      = result["coefs"]
    n_active   = int(np.sum(np.abs(coefs) > 0))

    row = {
        "dataset":            int(state["dataset"]),
        "file_y":             int(state["file_y"]),
        "omega":              float(state["omega"]),
        "b":                  float(state["b"]),
        "blim":               float(state["blim"]),
        "b_over_blim":        float(state["b_over_blim"]),
        "label":              int(state["b_over_blim"] >= 1.0),
        "teeth":              int(state["teeth"]),
        "tau_sec":            float(state["tau_sec"]),
        "delay_samples":      int(state["delay_samples"]),
        "dt":                 float(state["dt"]),
        "include_y":          bool(state["include_y"]),
        "X_rows":             int(state["X"].shape[0]),
        "X_cols":             int(state["X"].shape[1]),
        # Sweep diagnostics
        "threshold":          float(threshold),
        "n_active_terms":     n_active,
        "n_plateaus":         info["n_plateaus"],
        "chosen_plateau_len": info["chosen_plateau_len"],
        "fallback_threshold": info["fallback"],
        # Scales for unit recovery
        "scale_x":            float(state["scales"]["x"]),
        "scale_dxdt":         float(state["scales"]["dxdt"]),
        "scale_chip_x":       float(state["scales"]["chip_x"]),
        "scale_d2xdt2":       float(state["scales"]["d2xdt2"]),
        "pickle_path":        pkl_path,
    }

    if state["include_y"]:
        row["scale_y"]      = float(state["scales"]["y"])
        row["scale_dydt"]   = float(state["scales"]["dydt"])
        row["scale_chip_y"] = float(state["scales"]["chip_y"])

    # One column per library term, using cleaned name
    for name, coef in zip(term_names, coefs):
        clean = name.replace(" ", "_")
        row[f"coef_{clean}"] = float(coef)

    return row


# ----------------------------------------------------------------------
# Driver helpers
# ----------------------------------------------------------------------

def list_pickles_for_dataset(ds):
    pattern = os.path.join(PREPROCESSED_DIR, f"state_{ds}_*.pkl")
    return sorted(glob.glob(pattern))


def write_dataset_csv(rows, ds):
    if not rows:
        return None
    os.makedirs(FITS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    if "file_y" in df.columns:
        df = df.sort_values("file_y").reset_index(drop=True)
    out_path = os.path.join(FITS_DIR, f"dataset_{ds}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def append_failure_log(failure_rows, path=FAILURE_LOG):
    if not failure_rows:
        return
    df = pd.DataFrame(failure_rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def main(dataset_start, dataset_end, verbose=True):
    os.makedirs(FITS_DIR, exist_ok=True)

    total_fit  = 0
    total_fail = 0
    started    = time.time()

    for ds in range(dataset_start, dataset_end + 1):
        pkls = list_pickles_for_dataset(ds)
        if not pkls:
            if verbose:
                print(f"[ds={ds}] no pickles, skipping")
            continue

        ds_started = time.time()
        rows       = []
        failures   = []

        for pkl in pkls:
            fy_match = re.search(r"state_\d+_(\d+)\.pkl", pkl)
            fy = int(fy_match.group(1)) if fy_match else -1

            try:
                row = fit_one_file(pkl)
                rows.append(row)
                if verbose:
                    print(f"[ds={ds} fy={fy}] thr={row['threshold']:.4f} "
                          f"n_terms={row['n_active_terms']}")
            except Exception as e:
                failures.append({
                    "dataset":       ds,
                    "file_y":        fy,
                    "pickle_path":   pkl,
                    "error_type":    type(e).__name__,
                    "error_message": str(e),
                })
                if verbose:
                    print(f"[ds={ds} fy={fy}] FAILED: {type(e).__name__}: {e}")

        out_path = write_dataset_csv(rows, ds)
        append_failure_log(failures)

        ds_elapsed = time.time() - ds_started
        total_fit  += len(rows)
        total_fail += len(failures)

        print(f"[ds={ds}] {len(rows)} fit, {len(failures)} failed, "
              f"{ds_elapsed:.1f}s -> {out_path}")

    elapsed = time.time() - started
    print("=" * 60)
    print(f"Datasets:  {dataset_start}..{dataset_end}")
    print(f"Fit:       {total_fit}")
    print(f"Failed:    {total_fail}")
    print(f"Elapsed:   {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output:    {FITS_DIR}")
    print(f"Failures:  {FAILURE_LOG}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if TEST_MODE:
        pkl = os.path.join(
            PREPROCESSED_DIR,
            f"state_{TEST_DATASET}_{TEST_FILE_Y}.pkl",
        )
        try:
            row = fit_one_file(pkl)
            print("TEST_MODE result: SUCCESS")
            for k, v in row.items():
                print(f"  {k}: {v}")
        except Exception:
            traceback.print_exc()
    else:
        parser = argparse.ArgumentParser(
            description="Fit SINDy-DDE per file using preprocessed pickles."
        )
        parser.add_argument("--dataset_start", type=int, default=1)
        parser.add_argument("--dataset_end",   type=int, default=120)
        parser.add_argument("--quiet", action="store_true")
        args = parser.parse_args()
        main(args.dataset_start, args.dataset_end, verbose=not args.quiet)
