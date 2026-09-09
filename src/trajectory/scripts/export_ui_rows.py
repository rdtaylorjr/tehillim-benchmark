"""Converts validate_against_genre.py's validation.csv into the UI's JSON row shape."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from library.embeddings import split_model_name
from library.rows_output import write_json

_FIELDS = (
    "n_pairs_total",
    "n_pairs_valid",
    "raw_gap",
    "raw_p",
    "raw_effect_size",
    "raw_q",
    "length_controlled_gap",
    "length_controlled_p",
    "length_controlled_effect_size",
    "length_controlled_q",
    "length_and_content_controlled_gap",
    "length_and_content_controlled_p",
    "length_and_content_controlled_effect_size",
    "length_and_content_controlled_q",
)


def trajectory_ui_rows(validation_df: pd.DataFrame) -> list[dict[str, Any]]:
    """One UI row per validation.csv row, with model split into model_base/text_variant."""
    split = validation_df["model"].map(split_model_name)
    df = validation_df.assign(
        model_base=[base for base, _ in split], text_variant=[variant for _, variant in split]
    )
    rows: list[dict[str, Any]] = df[
        ["model", "model_base", "text_variant", "metric", *_FIELDS]
    ].to_dict("records")
    return rows


_BY_GENRE_FIELDS = (
    "source",
    "genre",
    "gap",
    "p_perm",
    "p_maxT",
    "perm_q",
    "perm_q_by",
    "maxT_q",
    "maxT_q_by",
)


def trajectory_by_genre_ui_rows(breakdown_df: pd.DataFrame) -> list[dict[str, Any]]:
    """One UI row per validate_against_genre_by_genre.csv row, model split as above."""
    split = breakdown_df["model"].map(split_model_name)
    df = breakdown_df.assign(
        model_base=[base for base, _ in split], text_variant=[variant for _, variant in split]
    )
    rows: list[dict[str, Any]] = df[
        ["model", "model_base", "text_variant", "metric", *_BY_GENRE_FIELDS]
    ].to_dict("records")
    return rows


def main(argv: list[str] | None = None) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_csv", type=Path)
    parser.add_argument("--breakdown-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--breakdown-output", type=Path, default=None)
    args = parser.parse_args(argv)

    validation_df = pd.read_csv(args.validation_csv)
    rows = trajectory_ui_rows(validation_df)
    write_json(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")

    if args.breakdown_csv and args.breakdown_output:
        breakdown_df = pd.read_csv(args.breakdown_csv)
        breakdown_rows = trajectory_by_genre_ui_rows(breakdown_df)
        write_json(args.breakdown_output, breakdown_rows)
        print(f"wrote {len(breakdown_rows)} by-genre rows to {args.breakdown_output}")


if __name__ == "__main__":
    main()
