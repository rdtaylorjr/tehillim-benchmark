"""Psalm vertex-resampling BCa bootstrap 95% CIs for AP (primary), gap, and AUC, every model."""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from genre.bootstrap import (
    block_bootstrap_genre_ap_gap_and_auc,
    build_similarity_and_genre_matrices,
)
from genre.genre_labels import load_genre_by_psalm
from library.ap_gap_auc_bootstrap import ci_row
from library.bhsa import list_psalms_half_verses_by_psalm, load_bhsa_api
from library.calibration import background_stats_from_matrix
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
    genre_by_psalm: dict[int, str],
    n_resamples: int,
    seed: int,
) -> dict[str, str | int | float]:
    """One model file's CI row, raising when its psalm population cannot support a CI."""
    model = dataset_identifier(path)
    psalm_vectors = load_psalm_vectors(path, half_verses_by_psalm)
    psalm_ids = sorted(psalm_vectors)
    similarity_matrix, genre_match_matrix = build_similarity_and_genre_matrices(
        psalm_ids, psalm_vectors, genre_by_psalm
    )
    background = background_stats_from_matrix(similarity_matrix)
    result = block_bootstrap_genre_ap_gap_and_auc(
        psalm_ids,
        similarity_matrix,
        genre_match_matrix,
        background,
        n_resamples=n_resamples,
        rng=np.random.default_rng(seed),
    )
    return ci_row(model, result)


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_genre_csv_argument(parser)
    add_embeddings_dir_argument(parser)
    add_scoring_arguments(parser, with_seed=True, with_resamples=True)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)

    rows, model_paths = resume_from_cache(args.embeddings_dir, args.output)
    score = partial(
        score_model,
        half_verses_by_psalm=half_verses_by_psalm,
        genre_by_psalm=genre_by_psalm,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    rows.extend(
        row
        for row in map_in_order(skipping_unscorable(score), model_paths, args.workers)
        if row is not None
    )

    for row in rows:
        print(
            f"{row['model']:55s} "
            f"AP={row['point_ap']:.3f} [{row['ap_ci_low']:.3f}, {row['ap_ci_high']:.3f}] "
            f"(chance={row['prevalence']:.3f}) "
            f"auc={row['point_auc']:.3f} gap={row['point_gap']:.3f}"
        )

    if args.output:
        write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
