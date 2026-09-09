"""Order-shuffle null: does half-verse order carry more genre signal than a shuffled null."""

import argparse
import csv
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from families.shuffle import Draws, draw, load_draws

from genre.evaluate import evaluate_genre_discrimination
from genre.genre_labels import load_genre_by_psalm
from genre.pairs import GenrePair, build_genre_pairs, filter_pairs_by_genre
from library.bhsa import list_psalms_half_verses_by_psalm, load_bhsa_api
from library.cli import (
    add_genre_csv_argument,
    add_scoring_arguments,
    add_shuffle_family_arguments,
)
from library.order_shuffle import order_shuffle_result
from library.psalm_vectors import draw_psalm_vectors, load_psalm_vectors
from library.worker_pool import map_in_order


def score_genre_ap(
    path: Path,
    half_verses_by_psalm: dict[int, list[int]],
    pairs: list[GenrePair],
    genres: list[str],
) -> dict[str, float]:
    """Per-genre Average Precision (no permutation testing) for one embeddings file."""
    return genre_ap(load_psalm_vectors(path, half_verses_by_psalm), pairs, genres)


def genre_ap(
    psalm_vectors: dict[int, np.ndarray], pairs: list[GenrePair], genres: list[str]
) -> dict[str, float]:
    """Per-genre Average Precision over psalm centroids already in memory, file or freshly built."""
    return {
        genre: evaluate_genre_discrimination(
            filter_pairs_by_genre(pairs, genre), psalm_vectors
        ).average_precision
        for genre in genres
    }


def score_built_seed(
    draws: Draws,
    half_verses_by_psalm: dict[int, list[int]],
    pairs: list[GenrePair],
    genres: list[str],
    seed: int,
) -> dict[str, float]:
    """Builds one seed's draw, scores it, and releases the vectors before the next seed."""
    vectors = draw_psalm_vectors(draw(draws, seed), draws.sparse_width, half_verses_by_psalm)
    return genre_ap(vectors, pairs, genres)


def built_null_scores(
    draws: Draws,
    n_shuffles: int,
    half_verses_by_psalm: dict[int, list[int]],
    pairs: list[GenrePair],
    genres: list[str],
    workers: int | None,
) -> list[dict[str, float]]:
    """One per-genre score set per seed, each draw built in memory and never written to a file."""
    seeds = list(range(1, n_shuffles + 1))
    score = partial(score_built_seed, draws, half_verses_by_psalm, pairs, genres)
    return map_in_order(score, seeds, workers)


def shuffled_scores_by_genre(
    scores_by_seed: list[dict[str, float]], genres: list[str]
) -> dict[str, np.ndarray]:
    """Transposes per-seed genre scores into one null array per genre, keeping seed order."""
    return {
        genre: np.array([scores[genre] for scores in scores_by_seed], dtype=float)
        for genre in genres
    }


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
    draws_factory: Callable[[str, Path], Draws] = load_draws,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_genre_csv_argument(parser)
    parser.add_argument("real_embeddings", type=Path)
    add_shuffle_family_arguments(parser)
    add_scoring_arguments(parser, with_shuffles=True)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    pairs = build_genre_pairs(genre_by_psalm)
    genres = sorted(set(genre_by_psalm.values()))

    real_ap = score_genre_ap(args.real_embeddings, half_verses_by_psalm, pairs, genres)
    per_seed = built_null_scores(
        draws_factory(args.family, args.config_root),
        args.n_shuffles,
        half_verses_by_psalm,
        pairs,
        genres,
        args.workers,
    )
    shuffled_ap = shuffled_scores_by_genre(per_seed, genres)

    rows = []
    for genre in genres:
        result = order_shuffle_result(
            real_score=real_ap[genre],
            shuffled_scores=shuffled_ap[genre],
            n_hypotheses=len(genres),
        )
        rows.append(
            {
                "genre": genre,
                "ap_real": real_ap[genre],
                "ap_shuffled_mean": float(np.mean(shuffled_ap[genre])),
                "n_shuffles": len(shuffled_ap[genre]),
                "delta_order": result.delta_order,
                "p_value": result.p_value,
            }
        )
        print(
            f"{genre:15s} ap_real={real_ap[genre]:.4f} "
            f"delta_order={result.delta_order:+.4f} p={result.p_value:.4f}"
        )

    if args.output:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
