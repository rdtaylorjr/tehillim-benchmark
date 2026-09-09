"""Order-shuffle null: does half-verse order carry more genre signal than a shuffled null."""

import argparse
import csv
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from genre.evaluate import evaluate_genre_discrimination
from genre.genre_labels import load_genre_by_psalm
from genre.pairs import GenrePair, build_genre_pairs, filter_pairs_by_genre
from library.bhsa import list_psalms_half_verses_by_psalm, load_bhsa_api
from library.cli import add_genre_csv_argument, add_scoring_arguments
from library.order_shuffle import order_shuffle_result
from library.psalm_vectors import load_psalm_vectors
from library.shuffle_draws import select_shuffle_draws
from library.worker_pool import map_in_order


def score_genre_ap(
    path: Path,
    half_verses_by_psalm: dict[int, list[int]],
    pairs: list[GenrePair],
    genres: list[str],
) -> dict[str, float]:
    """Per-genre Average Precision (no permutation testing) for one embeddings file."""
    vectors = load_psalm_vectors(path, half_verses_by_psalm)
    return {
        genre: evaluate_genre_discrimination(
            filter_pairs_by_genre(pairs, genre), vectors
        ).average_precision
        for genre in genres
    }


def shuffled_scores_by_genre(
    scores_by_file: list[dict[str, float]], genres: list[str]
) -> dict[str, np.ndarray]:
    """Transposes per-file genre scores into one null array per genre, keeping file order."""
    return {
        genre: np.array([scores[genre] for scores in scores_by_file], dtype=float)
        for genre in genres
    }


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_genre_csv_argument(parser)
    parser.add_argument("real_embeddings", type=Path)
    parser.add_argument("shuffled_embeddings_dir", type=Path)
    add_scoring_arguments(parser, with_shuffles=True)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    pairs = build_genre_pairs(genre_by_psalm)
    genres = sorted(set(genre_by_psalm.values()))

    real_ap = score_genre_ap(args.real_embeddings, half_verses_by_psalm, pairs, genres)
    shuffled_paths = select_shuffle_draws(args.shuffled_embeddings_dir, args.n_shuffles)
    score = partial(
        score_genre_ap, half_verses_by_psalm=half_verses_by_psalm, pairs=pairs, genres=genres
    )
    shuffled_ap = shuffled_scores_by_genre(
        map_in_order(score, shuffled_paths, args.workers), genres
    )

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
