"""Human-readable status summaries for Phase 1-3."""

from __future__ import annotations

from pathlib import Path

from pri2m.data.metadata import load_feature_metadata, summarize_datasets
from pri2m.hpc.sync import verify_local_sindy_summary


def build_status_report(root: str | Path = ".") -> str:
    """Build a compact status report from local lightweight artifacts."""
    base = Path(root)
    lines = ["# PRI2M Phase 1-3 Status", ""]

    sindy = verify_local_sindy_summary(base / "sindy_pipeline")
    lines.append("## SINDy Lightweight Summary")
    lines.append(f"- OK: {sindy['ok']}")
    lines.append(f"- Preprocessed rows: {sindy['preprocessed_rows']}")
    lines.append(f"- Dataset count: {sindy['dataset_count']}")
    lines.append(f"- Fit failures: {sindy['fit_failures']}")
    if sindy["messages"]:
        lines.append(f"- Messages: {'; '.join(sindy['messages'])}")

    feature_csv = base / "data" / "metadata" / "features_all_datasets.csv"
    if feature_csv.exists():
        df = load_feature_metadata(feature_csv, datasets=[1, 4, 29, 35, 39])
        summary = summarize_datasets(df)
        lines.extend(["", "## Phase 3 Clean Dev Metadata", summary.to_string(index=False)])

    noisy_csv = base / "data" / "metadata" / "features_noisy_datasets.csv"
    if noisy_csv.exists():
        df_noisy = load_feature_metadata(noisy_csv, datasets=[91])
        summary_noisy = summarize_datasets(df_noisy)
        lines.extend(["", "## Phase 3 Noisy Dev Metadata", summary_noisy.to_string(index=False)])

    return "\n".join(lines) + "\n"
