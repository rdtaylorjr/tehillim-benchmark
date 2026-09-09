"""Selects the results UI's required columns from one representation domain's results."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from library.embeddings import split_model_name
from library.rows_output import write_json

_PARALLELISM_OVERALL_COLUMNS = [
    "model",
    "model_base",
    "text_variant",
    "separation_auc",
    "separation_p_q",
    "auc_vs_baseline",
    "p_vs_baseline_q",
    "average_precision",
    "calibrated_effect_size",
    "mrr_forward",
    "n_true",
]
_PARALLELISM_BY_TYPE_COLUMNS = [
    "model",
    "model_base",
    "text_variant",
    "scope",
    "separation_auc",
    "separation_p_q",
    "auc_vs_baseline",
    "p_vs_baseline_q",
    "average_precision",
    "calibrated_effect_size",
    "mrr_forward",
    "n_true",
]
_GENRE_OVERALL_COLUMNS = [
    "model",
    "model_base",
    "text_variant",
    "separation_auc",
    "auc_ci_low",
    "auc_ci_high",
    "average_precision",
    "ap_ci_low",
    "ap_ci_high",
    "prevalence",
    "n_same_genre",
    "n_different_genre",
]
_GENRE_BY_GENRE_COLUMNS = [
    "model",
    "model_base",
    "text_variant",
    "genre",
    "separation_auc",
    "auc_ci_low",
    "auc_ci_high",
    "average_precision",
    "ap_ci_low",
    "ap_ci_high",
    "prevalence",
    "n_same_genre",
    "n_different_genre",
]


_PSALM_LEVEL_MODEL = r"_psalm(?:_shuffle\d+)?$"
_SHUFFLE_CONTROL_MODEL = r"_shuffle\d+"


def _drop_psalm_level_models(models_df: pd.DataFrame) -> pd.DataFrame:
    """Excludes _psalm[_shuffleNN]-suffixed models: degenerate for a half-verse-pair task."""
    return models_df[~models_df["model"].str.contains(_PSALM_LEVEL_MODEL, regex=True)]


def _drop_shuffle_control_models(models_df: pd.DataFrame) -> pd.DataFrame:
    """Excludes _shuffleNN models: a null-order control checked against one model, not rankable."""
    return models_df[~models_df["model"].str.contains(_SHUFFLE_CONTROL_MODEL, regex=True)]


def _drop_shuffle_control_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same exclusion as _drop_shuffle_control_models, for a plain row-dict list."""
    return [row for row in rows if not re.search(_SHUFFLE_CONTROL_MODEL, row.get("model", ""))]


def _add_model_base_and_text_variant(models_df: pd.DataFrame) -> pd.DataFrame:
    """Derives model_base/text_variant from `model`, for tables that don't already carry them."""
    models_df = models_df.copy()
    split = [split_model_name(model) for model in models_df["model"]]
    models_df["model_base"] = [base for base, _ in split]
    models_df["text_variant"] = [variant for _, variant in split]
    return models_df


def build_domain_data(
    parallelism_overall_df: pd.DataFrame,
    parallelism_by_type_df: pd.DataFrame,
    genre_overall_df: pd.DataFrame,
    genre_by_genre_df: pd.DataFrame,
    trajectory_rows: list[dict[str, Any]],
    trajectory_by_genre_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One domain's UI payload: the 6 tables the UI's tabs render."""
    parallelism_overall_df = _drop_shuffle_control_models(
        _drop_psalm_level_models(parallelism_overall_df)
    )
    parallelism_by_type_df = _drop_shuffle_control_models(
        _drop_psalm_level_models(parallelism_by_type_df)
    )
    genre_overall_df = _drop_shuffle_control_models(genre_overall_df)
    genre_by_genre_df = _drop_shuffle_control_models(
        _add_model_base_and_text_variant(genre_by_genre_df)
    )
    trajectory_rows = _drop_shuffle_control_rows(trajectory_rows)
    trajectory_by_genre_rows = _drop_shuffle_control_rows(trajectory_by_genre_rows or [])
    return {
        "parallelism_overall": parallelism_overall_df[_PARALLELISM_OVERALL_COLUMNS].to_dict(
            "records"
        ),
        "parallelism_by_type": parallelism_by_type_df[_PARALLELISM_BY_TYPE_COLUMNS].to_dict(
            "records"
        ),
        "genre_overall": genre_overall_df[_GENRE_OVERALL_COLUMNS].to_dict("records"),
        "genre_by_genre": genre_by_genre_df[_GENRE_BY_GENRE_COLUMNS].to_dict("records"),
        "trajectory": trajectory_rows,
        "trajectory_by_genre": trajectory_by_genre_rows,
    }


def split_payloads(
    domain: str, data: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Splits the per-genre trajectory rows out by metric, since one view reads one metric."""
    rows = data.get("trajectory_by_genre")
    if rows is None:
        return {domain: data}, {}

    core = {domain: {key: value for key, value in data.items() if key != "trajectory_by_genre"}}
    by_metric: dict[str, list[Any]] = {}
    for row in rows:
        by_metric.setdefault(row["metric"], []).append(row)
    slices = {
        metric: {domain: {"trajectory_by_genre": metric_rows}}
        for metric, metric_rows in by_metric.items()
    }
    return core, slices


def main(argv: list[str] | None = None) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="representation domain name, e.g. semantic, lexical")
    parser.add_argument("--parallelism-dir", type=Path, required=True)
    parser.add_argument("--genre-dir", type=Path, required=True)
    parser.add_argument("--trajectory-ui-rows", type=Path, required=True)
    parser.add_argument("--trajectory-by-genre-rows", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    parallelism_overall_df = pd.read_parquet(
        args.parallelism_dir / "stage=master" / "model_metrics_overall.parquet"
    )
    parallelism_by_type_df = pd.read_parquet(
        args.parallelism_dir / "stage=master" / "model_metrics_by_type.parquet"
    )
    genre_overall_df = pd.read_parquet(
        args.genre_dir / "stage=master" / "genre_metrics_wide.parquet"
    )
    genre_by_genre_df = pd.read_csv(args.genre_dir / "stage=raw" / "by_genre.csv")
    trajectory_rows = json.loads(args.trajectory_ui_rows.read_text())
    trajectory_by_genre_rows = (
        json.loads(args.trajectory_by_genre_rows.read_text())
        if args.trajectory_by_genre_rows
        else None
    )

    data = build_domain_data(
        parallelism_overall_df,
        parallelism_by_type_df,
        genre_overall_df,
        genre_by_genre_df,
        trajectory_rows,
        trajectory_by_genre_rows,
    )
    core, slices = split_payloads(args.domain, data)
    write_json(args.output, core)
    print(f"wrote domain={args.domain} to {args.output}")

    for metric, payload in sorted(slices.items()):
        path = args.output.with_name(f"{args.output.stem}_trajectory_{metric}.json")
        write_json(path, payload)
        print(f"wrote domain={args.domain} metric={metric} to {path}")


if __name__ == "__main__":
    main()
