"""
sindy_analysis.py

Post-fit analysis on the per-dataset SINDy CSVs produced by sindy_fit.py.

Four analyses, each produces a CSV and a PNG:

  1. Aggregate: term inclusion by stability label across all datasets.
     Replicates the dataset-1 figure but on the full ~32K file sample.

  2. Noise tier: compare clean (datasets 1-30) vs 1% noise (61-90)
     vs 5% noise (91-120). Datasets 31-60 are clean but have no noisy
     mirrors, so they are excluded from the tier comparison.

  3. b/blim gradient: bin files by b_over_blim and plot chip-term
     inclusion vs bin. Tests whether chatter physics emerges sharply
     at the analytical boundary or as a gradient.

  4. Per-dataset replication: per-dataset chip_x inclusion delta
     (unstable minus stable). Tests whether the dataset-1 chip
     discrimination story repeats across all 120 datasets.

Outputs land in /nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline/sindy_analysis/

Run after both production sindy_fit jobs finish. If you run while
jobs are still in progress, results reflect whatever per-dataset
CSVs exist at that moment.
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

FITS_DIR     = "/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline/sindy_fits"
ANALYSIS_DIR = "/nfs/turbo/coe-jaabell/PRI2M/SINDy_DDE/sindy_pipeline/sindy_analysis"

# Color palette (matches the dataset-1 preview figure)
COLOR_STABLE   = "#2E86AB"
COLOR_UNSTABLE = "#E63946"

# Chip terms are the chatter signature, isolated for focused plots
CHIP_TERMS = ["chip_x", "chip_x^2", "x_chip_x", "dxdt_chip_x"]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_all_fits():
    """Concatenate every dataset_*.csv in FITS_DIR into one dataframe."""
    paths = sorted(glob.glob(os.path.join(FITS_DIR, "dataset_*.csv")))
    print(f"Found {len(paths)} per-dataset CSVs")
    if not paths:
        raise FileNotFoundError(f"No dataset_*.csv files in {FITS_DIR}")

    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    # Coef columns may be NaN if some files had different libraries.
    # Treat NaN as zero (term not present) before any inclusion math.
    coef_cols = [c for c in df.columns if c.startswith("coef_")]
    df[coef_cols] = df[coef_cols].fillna(0.0)

    print(f"Total rows:          {len(df)}")
    print(f"Datasets present:    {sorted(df['dataset'].unique())}")
    print(f"Stable (label=0):    {(df['label'] == 0).sum()}")
    print(f"Unstable (label=1):  {(df['label'] == 1).sum()}")
    return df


def get_coef_cols(df):
    return [c for c in df.columns if c.startswith("coef_")]


# ----------------------------------------------------------------------
# Analysis 1: Aggregate inclusion by label
# ----------------------------------------------------------------------

def analysis_aggregate(df, out_dir):
    coef_cols = get_coef_cols(df)
    incl = df.groupby("label")[coef_cols].apply(
        lambda g: (g != 0).mean()
    ).T
    # Map column ints to readable names
    rename = {0: "Stable", 1: "Unstable"}
    incl.rename(columns=rename, inplace=True)
    incl["delta"] = incl["Unstable"] - incl["Stable"]
    incl["ratio"] = (
        incl["Unstable"] / incl["Stable"].replace(0, np.nan)
    )

    incl.to_csv(os.path.join(out_dir, "aggregate_inclusion_by_label.csv"))

    incl_sorted = incl.sort_values(
        ["delta", "ratio"], ascending=False, na_position="first"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(incl_sorted))
    ax.barh(y - 0.2, incl_sorted["Stable"], height=0.4,
            label="Stable", color=COLOR_STABLE)
    ax.barh(y + 0.2, incl_sorted["Unstable"], height=0.4,
            label="Unstable", color=COLOR_UNSTABLE)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("coef_", "") for c in incl_sorted.index])
    ax.axvline(0.5, ls="--", color="gray", alpha=0.5)
    n_total = len(df)
    n_stab  = (df["label"] == 0).sum()
    n_unst  = (df["label"] == 1).sum()
    ax.set_title(
        f"SINDy term inclusion by stability label\n"
        f"All datasets ({n_total} files: {n_stab} stable, {n_unst} unstable)"
    )
    ax.set_xlabel("Term inclusion rate")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aggregate_inclusion_by_label.png"),
                dpi=150)
    plt.close()

    print("\n=== Aggregate inclusion by label ===")
    print(incl[["Stable", "Unstable", "delta", "ratio"]].round(3).to_string())


# ----------------------------------------------------------------------
# Analysis 2: Noise tier comparison
# ----------------------------------------------------------------------

def noise_tier(ds):
    if 1 <= ds <= 30:
        return "0% (clean)"
    if 61 <= ds <= 90:
        return "1% noise"
    if 91 <= ds <= 120:
        return "5% noise"
    return None


def analysis_noise_tier(df, out_dir):
    df = df.copy()
    df["noise_tier"] = df["dataset"].apply(noise_tier)
    df_tier = df.dropna(subset=["noise_tier"]).copy()

    coef_cols = get_coef_cols(df_tier)

    rows = []
    for tier, group in df_tier.groupby("noise_tier"):
        for label, lab_group in group.groupby("label"):
            incl = (lab_group[coef_cols] != 0).mean()
            for term, rate in incl.items():
                rows.append({
                    "noise_tier": tier,
                    "label": "Stable" if label == 0 else "Unstable",
                    "term": term.replace("coef_", ""),
                    "inclusion": rate,
                    "n_files": len(lab_group),
                })

    incl_df = pd.DataFrame(rows)
    incl_df.to_csv(
        os.path.join(out_dir, "noise_tier_comparison.csv"), index=False
    )

    chip_incl = incl_df[incl_df["term"].isin(CHIP_TERMS)]
    tier_order = ["0% (clean)", "1% noise", "5% noise"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, term in zip(axes.flat, CHIP_TERMS):
        sub = chip_incl[chip_incl["term"] == term]
        x = np.arange(len(tier_order))
        stab_vals = []
        unst_vals = []
        for t in tier_order:
            s = sub[(sub["noise_tier"] == t) & (sub["label"] == "Stable")]
            u = sub[(sub["noise_tier"] == t) & (sub["label"] == "Unstable")]
            stab_vals.append(s["inclusion"].values[0] if len(s) else 0.0)
            unst_vals.append(u["inclusion"].values[0] if len(u) else 0.0)
        ax.bar(x - 0.2, stab_vals, width=0.4,
               label="Stable", color=COLOR_STABLE)
        ax.bar(x + 0.2, unst_vals, width=0.4,
               label="Unstable", color=COLOR_UNSTABLE)
        ax.set_xticks(x)
        ax.set_xticklabels(tier_order)
        ax.set_title(term)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    axes[0, 0].set_ylabel("Inclusion rate")
    axes[1, 0].set_ylabel("Inclusion rate")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(
        "SINDy chip-term inclusion across noise tiers\n"
        "(matched mirrors only: clean 1-30, 1% noise 61-90, 5% noise 91-120)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "noise_tier_comparison.png"), dpi=150)
    plt.close()

    print("\n=== Noise tier comparison (chip terms) ===")
    pivot = chip_incl.pivot_table(
        index="term", columns=["noise_tier", "label"], values="inclusion"
    ).round(3)
    print(pivot.to_string())


# ----------------------------------------------------------------------
# Analysis 3: b/blim gradient
# ----------------------------------------------------------------------

def analysis_bblim_gradient(df, out_dir):
    bins       = [0, 0.5, 0.8, 0.95, 1.05, 1.5, np.inf]
    bin_labels = ["<0.5", "0.5-0.8", "0.8-0.95", "0.95-1.05", "1.05-1.5", ">1.5"]

    df = df.copy()
    df["bblim_bin"] = pd.cut(
        df["b_over_blim"], bins=bins, labels=bin_labels, right=False
    )

    coef_cols = get_coef_cols(df)

    rows = []
    for bin_lab, group in df.groupby("bblim_bin"):
        if len(group) == 0:
            continue
        incl = (group[coef_cols] != 0).mean()
        for term, rate in incl.items():
            rows.append({
                "bblim_bin": str(bin_lab),
                "term": term.replace("coef_", ""),
                "inclusion": rate,
                "n_files": len(group),
            })

    bin_df = pd.DataFrame(rows)
    bin_df.to_csv(
        os.path.join(out_dir, "bblim_gradient.csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for term in CHIP_TERMS:
        sub = bin_df[bin_df["term"] == term].copy()
        sub["order"] = sub["bblim_bin"].apply(
            lambda b: bin_labels.index(b) if b in bin_labels else -1
        )
        sub = sub.sort_values("order")
        ax.plot(
            sub["bblim_bin"], sub["inclusion"], "o-",
            label=term, lw=2, markersize=8,
        )

    # Mark the analytical stability boundary (b/blim = 1.0)
    ax.axvline(x=3, color="gray", ls="--", alpha=0.5)
    ax.text(3.05, 0.05, "b/blim = 1.0", color="gray", fontsize=9)

    counts = bin_df.groupby("bblim_bin")["n_files"].first()
    counts = counts.reindex(bin_labels)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(
        [f"{lab}\n(n={int(counts.get(lab, 0))})" for lab in bin_labels]
    )

    ax.set_xlabel("b / blim bin")
    ax.set_ylabel("Term inclusion rate")
    ax.set_title("Chip-term inclusion across the stability boundary")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bblim_gradient.png"), dpi=150)
    plt.close()

    print("\n=== b/blim gradient (chip-term inclusion) ===")
    chip_pivot = bin_df[bin_df["term"].isin(CHIP_TERMS)].pivot_table(
        index="bblim_bin", columns="term", values="inclusion"
    )
    chip_pivot = chip_pivot.reindex(bin_labels)
    print(chip_pivot.round(3).to_string())


# ----------------------------------------------------------------------
# Analysis 4: Per-dataset replication
# ----------------------------------------------------------------------

def analysis_per_dataset(df, out_dir):
    rows = []
    for ds, group in df.groupby("dataset"):
        stab = group[group["label"] == 0]
        unst = group[group["label"] == 1]
        if len(stab) == 0 or len(unst) == 0:
            continue
        chip_stab = (stab["coef_chip_x"] != 0).mean()
        chip_unst = (unst["coef_chip_x"] != 0).mean()
        rows.append({
            "dataset":         ds,
            "n_stable":        len(stab),
            "n_unstable":      len(unst),
            "chip_stable":     chip_stab,
            "chip_unstable":   chip_unst,
            "discrimination":  chip_unst - chip_stab,
            "noise_tier":      noise_tier(ds) or "31-60 (clean, no mirror)",
        })

    pd_df = pd.DataFrame(rows)
    pd_df.to_csv(
        os.path.join(out_dir, "per_dataset_chip_inclusion.csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    x = pd_df["dataset"]
    ax.scatter(x, pd_df["chip_stable"],   color=COLOR_STABLE,   s=22,
               alpha=0.75, label="Stable")
    ax.scatter(x, pd_df["chip_unstable"], color=COLOR_UNSTABLE, s=22,
               alpha=0.75, label="Unstable")

    for _, row in pd_df.iterrows():
        ax.plot(
            [row["dataset"], row["dataset"]],
            [row["chip_stable"], row["chip_unstable"]],
            color="gray", alpha=0.25, lw=0.6,
        )

    for boundary in (30.5, 60.5, 90.5):
        ax.axvline(x=boundary, color="gray", ls=":", alpha=0.5)

    y_top = 1.05
    ax.text(15.5,  y_top, "Clean 1-30",      ha="center", fontsize=9)
    ax.text(45.5,  y_top, "Clean 31-60",     ha="center", fontsize=9)
    ax.text(75.5,  y_top, "1% noise 61-90",  ha="center", fontsize=9)
    ax.text(105.5, y_top, "5% noise 91-120", ha="center", fontsize=9)

    ax.set_xlabel("Dataset number")
    ax.set_ylabel("chip_x inclusion rate")
    ax.set_title(
        "Per-dataset chip_x inclusion: replication across all datasets"
    )
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "per_dataset_chip_inclusion.png"), dpi=150
    )
    plt.close()

    print("\n=== Per-dataset replication ===")
    print(f"Datasets with both labels:                    {len(pd_df)}")
    print(f"Mean discrimination (unstable minus stable): "
          f"{pd_df['discrimination'].mean():.3f}")
    print(f"Median discrimination:                        "
          f"{pd_df['discrimination'].median():.3f}")
    pos = (pd_df["discrimination"] > 0).sum()
    strong = (pd_df["discrimination"] > 0.3).sum()
    print(f"Datasets with positive discrimination:        {pos} / {len(pd_df)}")
    print(f"Datasets with strong discrimination (>0.3):   {strong} / {len(pd_df)}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    df = load_all_fits()
    df.to_csv(os.path.join(ANALYSIS_DIR, "all_fits_combined.csv"), index=False)

    analysis_aggregate(df, ANALYSIS_DIR)
    analysis_noise_tier(df, ANALYSIS_DIR)
    analysis_bblim_gradient(df, ANALYSIS_DIR)
    analysis_per_dataset(df, ANALYSIS_DIR)

    print("\n" + "=" * 60)
    print(f"All analyses written to: {ANALYSIS_DIR}")
    print("Files produced:")
    for f in sorted(os.listdir(ANALYSIS_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
