"""Joins every metric from compare_models/compare_true_similarity/export_detail into one parquet."""

import argparse
from pathlib import Path

import pandas as pd

from library.master_report import finalise_long_metrics, melt_to_long, pivot_metrics_wide
from library.rows_output import write_dataframe_parquet
from library.tie_diagnostics import tie_fractions_by_model

_RETRIEVAL_OVERALL_METRICS = [
    "n_pairs",
    "separation_auc",
    "separation_p",
    "discrimination_p",
    "discrimination_rank_biserial",
    "type_gap_z",
    "type_gap_p",
    "mrr_forward",
    "mrr_backward",
    "recall_at_1_forward",
    "recall_at_5_forward",
    "recall_at_10_forward",
    "recall_at_1_backward",
    "recall_at_5_backward",
    "recall_at_10_backward",
]
_RETRIEVAL_TYPE_METRICS = [
    "n_pairs",
    "separation_auc",
    "separation_p",
    "discrimination_p",
    "discrimination_rank_biserial",
    "mrr_forward",
    "mrr_backward",
    "recall_at_1_forward",
    "recall_at_5_forward",
    "recall_at_1_backward",
    "recall_at_5_backward",
]
_CALIBRATION_OVERALL_METRICS = [
    "n_pairs",
    "mean_true_similarity",
    "median_true_similarity",
    "std_true_similarity",
    "background_mean",
    "background_std",
    "background_n_vectors",
    "calibrated_effect_size",
]
_CALIBRATION_TYPE_METRICS = ["n_pairs", "mean_true_similarity", "calibrated_effect_size"]

_TYPES = ("Antithetic", "Emblematic", "Staircase", "Synonymous", "Synthetic")


def _melt_wide_by_type(scored_df: pd.DataFrame, metrics: list[str], source: str) -> pd.DataFrame:
    """One long row per (model, metric) for every parallelism type present in the frame."""
    frames = []
    for ptype in _TYPES:
        rename = {f"{m}_{ptype}": m for m in metrics}
        present = [c for c in rename if c in scored_df.columns]
        if not present:
            continue
        sub = scored_df[["model", *present]].rename(columns=rename)
        present_metrics = [rename[c] for c in present]
        frames.append(melt_to_long(sub, present_metrics, source, scope=ptype, scope_kind="type"))
    return pd.concat(frames, ignore_index=True)


def build_long_metrics(
    retrieval_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    scope_baseline_df: pd.DataFrame,
    pair_detail_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combines all three metric sources into one tidy (model, scope, metric, value) table."""
    parts = [
        melt_to_long(retrieval_df, _RETRIEVAL_OVERALL_METRICS, "retrieval_separation"),
        _melt_wide_by_type(retrieval_df, _RETRIEVAL_TYPE_METRICS, "retrieval_separation"),
        melt_to_long(calibration_df, _CALIBRATION_OVERALL_METRICS, "calibrated_similarity"),
        _melt_wide_by_type(calibration_df, _CALIBRATION_TYPE_METRICS, "calibrated_similarity"),
    ]
    baseline_metrics = [
        "n_true",
        "n_baseline",
        "prevalence",
        "average_precision",
        "true_effect_size",
        "baseline_effect_size",
        "gap",
        "auc_vs_baseline",
        "p_vs_baseline",
    ]
    baseline_long = scope_baseline_df.melt(
        id_vars=["model", "scope", "scope_kind"],
        value_vars=baseline_metrics,
        var_name="metric",
        value_name="value",
    )
    baseline_long["source"] = "vs_baseline"
    parts.append(baseline_long)

    if pair_detail_df is not None:
        #: Reports where a separation AUC is decided by one tie block rather than by ordering.
        parts.append(tie_fractions_by_model(pair_detail_df))

    return finalise_long_metrics(parts)


def main(argv: list[str] | None = None) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-csv", type=Path, required=True)
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--detail-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_df = pd.read_csv(args.retrieval_csv)
    calibration_df = pd.read_csv(args.calibration_csv)
    scope_baseline_df = pd.read_parquet(args.detail_dir / "type_vs_baseline.parquet")
    pair_detail_df = pd.read_parquet(
        args.detail_dir / "pair_detail.parquet", columns=["model", "raw_similarity"]
    )

    long_df = build_long_metrics(retrieval_df, calibration_df, scope_baseline_df, pair_detail_df)
    write_dataframe_parquet(args.output_dir / "model_metrics_long.parquet", long_df)

    overall_wide = _pivot_scope(long_df, "overall", ["model", "model_base", "text_variant"])
    write_dataframe_parquet(args.output_dir / "model_metrics_overall.parquet", overall_wide)

    by_type_wide = _pivot_scope(long_df, "type", ["model", "model_base", "text_variant", "scope"])
    write_dataframe_parquet(args.output_dir / "model_metrics_by_type.parquet", by_type_wide)

    print(f"model_metrics_long: {len(long_df)} rows")
    print(f"model_metrics_overall: {len(overall_wide)} rows")
    print(f"model_metrics_by_type: {len(by_type_wide)} rows")


def _pivot_scope(long_df: pd.DataFrame, scope_kind: str, index_cols: list[str]) -> pd.DataFrame:
    """Pivots one scope_kind's rows, the overall and per-type tables being separate."""
    return pivot_metrics_wide(long_df[long_df["scope_kind"] == scope_kind], index_cols)


if __name__ == "__main__":
    main()
