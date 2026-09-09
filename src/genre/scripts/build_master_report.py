"""Joins the AP/AUC/calibration summary and bootstrap CIs into one long table with BY-FDR q's."""

import argparse
from pathlib import Path

import pandas as pd

from library.master_report import finalise_long_metrics, melt_to_long, pivot_metrics_wide
from library.rows_output import write_dataframe_parquet
from library.tie_diagnostics import tie_fractions_by_model

_SUMMARY_METRICS = [
    "n_same_genre",
    "n_different_genre",
    "prevalence",
    "average_precision",
    "same_genre_effect_size",
    "different_genre_effect_size",
    "gap",
    "separation_auc",
    "separation_p",
]
_BOOTSTRAP_METRICS = [
    "point_ap",
    "ap_ci_low",
    "ap_ci_high",
    "ap_ci_low_pct",
    "ap_ci_high_pct",
    "point_gap",
    "gap_ci_low",
    "gap_ci_high",
    "gap_ci_low_pct",
    "gap_ci_high_pct",
    "point_auc",
    "auc_ci_low",
    "auc_ci_high",
    "auc_ci_low_pct",
    "auc_ci_high_pct",
    "n_valid_resamples",
    "n_valid_jackknife",
]


def build_long_metrics(
    summary_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    pair_detail_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combines the AP/AUC/calibration summary and bootstrap CIs into one tidy long table."""
    parts = [
        melt_to_long(summary_df, _SUMMARY_METRICS, "genre_discrimination"),
        melt_to_long(bootstrap_df, _BOOTSTRAP_METRICS, "bootstrap_ci"),
    ]
    if pair_detail_df is not None:
        #: Reports where an average precision is decided by one tie block rather than by ordering.
        parts.append(tie_fractions_by_model(pair_detail_df))
    return finalise_long_metrics(parts)


def main(argv: list[str] | None = None) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-csv", type=Path, required=True)
    parser.add_argument("--detail-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.summary_csv)
    bootstrap_df = pd.read_csv(args.bootstrap_csv)
    pair_detail_df = pd.read_parquet(
        args.detail_dir / "genre_pair_detail.parquet", columns=["model", "raw_similarity"]
    )

    long_df = build_long_metrics(summary_df, bootstrap_df, pair_detail_df)
    write_dataframe_parquet(args.output_dir / "genre_metrics_long.parquet", long_df)

    wide_df = pivot_metrics_wide(long_df, ["model", "model_base", "text_variant"])
    write_dataframe_parquet(args.output_dir / "genre_metrics_wide.parquet", wide_df)

    print(f"genre_metrics_long: {len(long_df)} rows")
    print(f"genre_metrics_wide: {len(wide_df)} rows")


if __name__ == "__main__":
    main()
