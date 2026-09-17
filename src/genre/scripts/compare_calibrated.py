"""Adds calibrated same/different-genre effect size on top of the raw AP/AUC report."""

import argparse
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from core.datasets import dataset_identifier
from core.parallel import map_in_order

from genre.calibrated import compare_genre_calibrated, genre_calibrated_row
from genre.pairs import GenrePair, build_genre_pairs
from genre.passages import load_half_verse_weights, load_passages
from library.bhsa import load_bhsa_api
from library.calibration import background_similarity_stats
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
    path: Path,
    half_verses_by_psalm: Mapping[str, Weights],
    pairs: list[GenrePair],
) -> dict[str, str | int | float]:
    """One model file's calibrated row, raising when its item vectors cannot be calibrated."""
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
    add_taxonomy_arguments(parser)
    add_embeddings_dir_argument(parser)
    add_scoring_arguments(parser)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    passages = load_passages(args.taxonomy, args.unit, args.labels_csv, api)
    pairs = build_genre_pairs(passages)
    half_verses_by_psalm = load_half_verse_weights(passages, api)

    rows, model_paths = resume_from_cache(args.embeddings_dir, args.output)
    score = partial(score_model, half_verses_by_psalm=half_verses_by_psalm, pairs=pairs)
    rows.extend(
        row
        for row in map_in_order(
            skipping_unscorable(score), model_paths, args.workers, label="models"
        )
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
