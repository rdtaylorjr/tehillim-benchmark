"""Runs evaluate_genre_discrimination across every embedding file in a directory."""

import argparse
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, cast

from core.datasets import dataset_identifier
from core.parallel import map_in_order

from genre.evaluate import evaluate_genre_discrimination
from genre.pairs import GenrePair, build_genre_pairs
from genre.passages import load_half_verse_weights, load_passages
from library.bhsa import load_bhsa_api
from library.centroid import Weights
from library.cli import (
    add_embeddings_dir_argument,
    add_scoring_arguments,
    add_taxonomy_arguments,
    resume_from_cache,
)
from library.psalm_vectors import load_psalm_vectors
from library.rows_output import write_rows_csv
from library.scoring import skipping_unscorable


def score_model(
    path: Path, half_verses_by_psalm: Mapping[str, Weights], pairs: list[GenrePair]
) -> dict[str, str | int | float]:
    """One model file's genre-discrimination row, scored independently of every other model."""
    psalm_vectors = load_psalm_vectors(path, half_verses_by_psalm)
    report = evaluate_genre_discrimination(pairs, psalm_vectors)
    return {
        "model": dataset_identifier(path),
        "n_same_genre": report.n_same_genre,
        "n_different_genre": report.n_different_genre,
        "prevalence": report.prevalence,
        "average_precision": report.average_precision,
        "separation_auc": report.separation_auc,
        "separation_p": report.separation_p,
    }


def compare_genre_models(
    pairs: list[GenrePair],
    model_paths: list[Path],
    half_verses_by_psalm: Mapping[str, Weights],
    max_workers: int | None = None,
) -> list[dict[str, str | int | float]]:
    """Scores every model file across workers, rows sorted by Average Precision descending."""
    score = partial(score_model, half_verses_by_psalm=half_verses_by_psalm, pairs=pairs)
    scored = map_in_order(skipping_unscorable(score), model_paths, max_workers, label="models")
    rows = [row for row in scored if row is not None]
    rows.sort(key=lambda r: cast("float", r["average_precision"]), reverse=True)
    return rows


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_taxonomy_arguments(parser)
    add_embeddings_dir_argument(parser)
    add_scoring_arguments(parser)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    passages = load_passages(args.taxonomy, args.unit, args.labels_csv, api)
    pairs = build_genre_pairs(passages)
    half_verses_by_psalm = load_half_verse_weights(passages, api)

    cached_rows, model_paths = resume_from_cache(args.embeddings_dir, args.output)
    new_rows = compare_genre_models(
        pairs, model_paths, half_verses_by_psalm, max_workers=args.workers
    )
    rows = sorted(
        cached_rows + new_rows, key=lambda r: cast("float", r["average_precision"]), reverse=True
    )

    for row in rows:
        print(
            f"{row['model']:55s} AP={row['average_precision']:.4f} "
            f"(chance={row['prevalence']:.3f}) auc={row['separation_auc']:.4f}"
        )

    if args.output:
        write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
