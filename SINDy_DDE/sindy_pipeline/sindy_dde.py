import os
import numpy as np
import pandas as pd
import pysindy as ps


def fit_sindy_dde(state, threshold=0.1, degree=2):
    """
    Fit a SINDy-DDE model to the state matrix and return the recovered
    equation as a dict of {term: coefficient}.

    Parameters
    ----------
    state : dict
        Output of build_state_matrix.
    threshold : float
        STLSQ sparsity threshold. Use lambda_sweep to choose this.
    degree : int
        Polynomial library degree. Default 2.

    Returns
    -------
    result : dict
        Contains term names, coefficients, and metadata.
    """
    X  = state["X"]
    y  = state["y"]
    dt = state["dt"]

    # Feature names come from build_state_matrix (x-only or x+y)
    feature_names = state["feature_names"]
    lib       = ps.PolynomialLibrary(degree=degree, include_bias=False)
    optimizer = ps.STLSQ(threshold=threshold)
    model     = ps.SINDy(
        feature_library=lib,
        optimizer=optimizer,
    )

    model.fit(X, t=dt, x_dot=y.reshape(-1, 1),
              feature_names=feature_names)

    term_names = model.get_feature_names()
    coefs      = model.coefficients().flatten()

    result = {
        "term_names": term_names,
        "coefs":      coefs,
        "model":      model,
    }

    return result


def print_equation(result):
    """
    Print the recovered SINDy equation, showing only surviving terms.

    Parameters
    ----------
    result : dict
        Output of fit_sindy_dde.
    """
    print("\n--- Recovered equation: d²x/dt² = ---")
    surviving = [(name, coef) for name, coef in
                 zip(result["term_names"], result["coefs"]) if abs(coef) > 0]

    if not surviving:
        print("  (no terms survived — threshold may be too high)")
    else:
        for name, coef in surviving:
            print(f"  {coef:+.6f}  *  {name}")

    print(f"\n  Total terms: {len(surviving)} / {len(result['term_names'])}")


def save_to_csv(result, sim, state, dataset_number, file_number, stab, threshold, output_csv):
    """
    Append one row of results to the CSV file. All library terms are
    columns — zero where a term did not survive. Creates the file if
    it does not exist yet.

    Parameters
    ----------
    result : dict
        Output of fit_sindy_dde.
    sim : dict
        Loaded simulation dict (trimmed).
    state : dict
        Output of build_state_matrix.
    dataset_number : int
    file_number : int
    stab : dict
        Output of load_stability_boundary.
    output_csv : str
        Full path to the CSV file.
    """
    term_names = result["term_names"]
    coefs      = result["coefs"]

    # Build row as a flat dict
    row = {
        "dataset":       dataset_number,
        "file_y":        file_number,
        "omega":         sim["omega"],
        "b":             sim["b"],
        "b_over_blim":   stab["b_over_blim"],
        "blim":          stab["blim"],
        "tau_sec":       state["tau_sec"],
        "delay_samples": state["delay_samples"],
        "teeth":         state["teeth"],
        "signal_length": len(sim["t"]),
        "include_y":     state["include_y"],
        "scale_x":       state["scales"]["x"],
        "scale_dxdt":    state["scales"]["dxdt"],
        "scale_chip_x":  state["scales"]["chip_x"],
        "scale_d2xdt2":  state["scales"]["d2xdt2"],
        "dt":            state["dt"],
        "fs":            sim["fs"],
        "threshold":     threshold,
    }

    # Add y-direction scales if present
    if state["include_y"]:
        row["scale_y"]      = state["scales"]["y"]
        row["scale_dydt"]   = state["scales"]["dydt"]
        row["scale_chip_y"] = state["scales"]["chip_y"]

    # Add one column per library term (zero if pruned)
    for name, coef in zip(term_names, coefs):
        row[f"coef_{name}"] = coef

    row_df = pd.DataFrame([row])

    if os.path.exists(output_csv):
        # Load existing CSV and align columns — fill missing with 0
        existing = pd.read_csv(output_csv)
        combined = pd.concat([existing, row_df], ignore_index=True).fillna(0)
        combined.to_csv(output_csv, index=False)
    else:
        row_df.to_csv(output_csv, index=False)

    print(f"\nResults saved to: {output_csv}")