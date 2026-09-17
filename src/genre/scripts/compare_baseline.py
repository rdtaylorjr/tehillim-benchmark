"""Scores passage length alone on the genre pairs, the reference a model's AP is read against."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from genre.length_baseline import length_baseline_rows
from genre.pairs import build_genre_pairs
from genre.passages import load_passages
from library.bhsa import DEFAULT_CHECKOUT, load_bhsa_api
from library.cli import add_taxonomy_arguments
from library.rows_output import write_rows_csv


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
) -> None:
    """Parses the arguments this module documents, scores the predictors, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_taxonomy_arguments(parser)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA checkout spec")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    passages = load_passages(args.taxonomy, args.unit, args.labels_csv, api)
    rows = length_baseline_rows(build_genre_pairs(passages), passages)

    for row in rows:
        print(
            f"{row['predictor']:20s} AP={row['average_precision']:.4f} "
            f"(chance={row['prevalence']:.3f}) auc={row['separation_auc']:.4f}"
        )
    write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
