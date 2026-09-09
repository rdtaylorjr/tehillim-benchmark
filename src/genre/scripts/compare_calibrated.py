"""Adds calibrated same/different-genre effect size on top of the raw AP/AUC report."""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from genre.calibrated import compare_genre_calibrated, genre_calibrated_row
from genre.genre_labels import load_genre_by_psalm
from genre.pairs import GenrePair, build_genre_pairs
from library.bhsa import list_psalms_half_verses_by_psalm, load_bhsa_api
from library.calibration import background_similarity_stats
from library.cli import (
    add_embeddings_dir_argument,
    add_genre_csv_argument,
    add_scoring_arguments,
    resume_from_cache,
)
from library.embeddings import dataset_identifier
from library.psalm_vectors import load_psalm_vectors
from library.rows_output import write_rows_csv
from library.scoring import skipping_unscorable
from library.worker_pool import map_in_order


def score_model(
    path: Path,
    half_verses_by_psalm: dict[int, list[int]],
    pairs: list[GenrePair],
) -> dict[str, str | int | float]:
    """One model file's calibrated row, raising when its psalm vectors cannot be calibrated."""
    model = dataset_identifier(path)
    psalm_vectors = load_psalm_vectors(path, half_verses_by_psalm)
    # Genre pairs cover every psalm, so the background is the full psalm-centroid population.
    background = background_similarity_stats(np.stack(list(psalm_vectors.values())))
    result = compare_genre_calibrated(pairs, psalm_vectors, background)
    return genre_calibrated_row(model, result)


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_genre_csv_argument(parser)
    add_embeddings_dir_argument(parser)
    add_scoring_arguments(parser)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    pairs = build_genre_pairs(genre_by_psalm)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)

    rows, model_paths = resume_from_cache(args.embeddings_dir, args.output)
    score = partial(score_model, half_verses_by_psalm=half_verses_by_psalm, pairs=pairs)
    rows.extend(
        row
        for row in map_in_order(skipping_unscorable(score), model_paths, args.workers)
        if row is not None
    )
    rows.sort(key=lambda r: r["average_precision"], reverse=True)

    for row in rows:
        print(
            f"{row['model']:55s} AP={row['average_precision']:.3f} "
            f"(chance={row['prevalence']:.3f}) auc={row['separation_auc']:.3f} gap={row['gap']:.3f}"
        )

    if args.output:
        write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
