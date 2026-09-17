"""Selects the results UI's required columns from one representation domain's results."""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from core.datasets import split_model_name

from library.rows_output import write_json
from library.stages import genre_partition, genre_registers

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
    "taxonomy",
    "unit",
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
    "taxonomy",
    "unit",
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


_GENRE_BASELINE_COLUMNS = [
    "taxonomy",
    "unit",
    "predictor",
    "average_precision",
    "separation_auc",
    "prevalence",
    "n_same_genre",
    "n_different_genre",
]
#: A table is sliced into one file per value of these columns, loaded when that view opens.
SLICES: dict[str, tuple[str, tuple[str, ...]]] = {
    "trajectory_by_genre": ("trajectory", ("metric",)),
    "genre_by_genre": ("genre", ("taxonomy", "unit")),
}
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


@dataclass(frozen=True, slots=True)
class GenreTables:
    """One taxonomy read at one register: its master, by-genre, and length-baseline tables."""

    taxonomy: str
    unit: str | None
    overall: pd.DataFrame
    by_genre: pd.DataFrame
    baseline: pd.DataFrame

    def tagged(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The frame with the register it belongs to on every row."""
        return frame.assign(taxonomy=self.taxonomy, unit=self.unit)


def read_genre_tables(data_root: Path, domain: str) -> list[GenreTables]:
    """Every register's tables for one domain, read from the taxonomy and unit partitions."""
    tables = []
    for taxonomy, unit in genre_registers():
        partition = genre_partition(data_root, taxonomy, unit, domain)
        tables.append(
            GenreTables(
                taxonomy,
                unit,
                pd.read_parquet(partition / "stage=master" / "genre_metrics_wide.parquet"),
                pd.read_csv(partition / "stage=raw" / "by_genre.csv"),
                pd.read_csv(partition / "stage=raw" / "baseline.csv"),
            )
        )
    return tables


def _register_catalog(genre: list[GenreTables]) -> list[dict[str, Any]]:
    """Each register with the labels it assigns, so the interface offers what the data holds."""
    return [
        {
            "taxonomy": tables.taxonomy,
            "unit": tables.unit,
            "genres": sorted(tables.by_genre["genre"].unique()),
        }
        for tables in genre
    ]


def build_domain_data(
    parallelism_overall_df: pd.DataFrame,
    parallelism_by_type_df: pd.DataFrame,
    genre: list[GenreTables],
    trajectory_rows: list[dict[str, Any]],
    trajectory_by_genre_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One domain's UI payload: the tables the UI's tabs render, every register in one table."""
    parallelism_overall_df = _drop_shuffle_control_models(
        _drop_psalm_level_models(parallelism_overall_df)
    )
    parallelism_by_type_df = _drop_shuffle_control_models(
        _drop_psalm_level_models(parallelism_by_type_df)
    )
    genre_overall_df = _drop_shuffle_control_models(
        pd.concat([tables.tagged(tables.overall) for tables in genre], ignore_index=True)
    )
    genre_by_genre_df = _drop_shuffle_control_models(
        _add_model_base_and_text_variant(
            pd.concat([tables.tagged(tables.by_genre) for tables in genre], ignore_index=True)
        )
    )
    genre_baseline_df = pd.concat(
        [tables.tagged(tables.baseline) for tables in genre], ignore_index=True
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
        "genre_registers": _register_catalog(genre),
        "genre_overall": genre_overall_df[_GENRE_OVERALL_COLUMNS].to_dict("records"),
        "genre_by_genre": genre_by_genre_df[_GENRE_BY_GENRE_COLUMNS].to_dict("records"),
        "genre_baseline": genre_baseline_df[_GENRE_BASELINE_COLUMNS].to_dict("records"),
        "trajectory": trajectory_rows,
        "trajectory_by_genre": trajectory_by_genre_rows,
    }


def slice_name(table: str, row: dict[str, Any]) -> str:
    """The file suffix one sliced row belongs under: the table's prefix and its key values."""
    prefix, keys = SLICES[table]
    values = [str(row[key]) for key in keys if row[key] is not None]
    return "_".join([prefix, *values])


def split_payloads(
    domain: str, data: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Splits every sliced table out of the core payload, one file per key, loaded on demand."""
    core = {domain: {key: value for key, value in data.items() if key not in SLICES}}
    slices: dict[str, dict[str, Any]] = {}
    for table in SLICES:
        for row in data.get(table) or []:
            name = slice_name(table, row)
            slices.setdefault(name, {domain: {table: []}})[domain][table].append(row)
    return core, slices


def main(argv: list[str] | None = None) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="representation domain name, e.g. semantic, lexical")
    parser.add_argument("--parallelism-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="tehillim-data checkout")
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
    genre = read_genre_tables(args.data_root, args.domain)
    trajectory_rows = json.loads(args.trajectory_ui_rows.read_text())
    trajectory_by_genre_rows = (
        json.loads(args.trajectory_by_genre_rows.read_text())
        if args.trajectory_by_genre_rows
        else None
    )

    data = build_domain_data(
        parallelism_overall_df,
        parallelism_by_type_df,
        genre,
        trajectory_rows,
        trajectory_by_genre_rows,
    )
    core, slices = split_payloads(args.domain, data)
    write_json(args.output, core)
    print(f"wrote domain={args.domain} to {args.output}")

    for name, payload in sorted(slices.items()):
        path = args.output.with_name(f"{args.output.stem}_{name}.json")
        write_json(path, payload)
        print(f"wrote domain={args.domain} slice={name} to {path}")


if __name__ == "__main__":
    main()
