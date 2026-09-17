"""Psalm vertex-resampling BCa bootstrap 95% CIs for AP (primary), gap, and AUC, every model."""

import argparse
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from core.datasets import dataset_identifier
from core.parallel import map_in_order

from genre.bootstrap import (
    block_bootstrap_genre_ap_gap_and_auc,
    build_similarity_and_genre_matrices,
)
from genre.passages import (
    Passage,
    admissible_mask,
    cluster_codes,
    load_half_verse_weights,
    load_passages,
)
from library.ap_gap_auc_bootstrap import ci_row
from library.bhsa import load_bhsa_api
from library.calibration import background_stats_from_matrix
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
    passages: list[Passage],
    weights: Mapping[str, Weights],
    n_resamples: int,
    seed: int,
) -> dict[str, str | int | float]:
    """One model file's CI row, raising when its passage population cannot support a CI."""
    model = dataset_identifier(path)
    psalm_vectors = load_psalm_vectors(path, weights)
    scored = [passage for passage in passages if passage.id in psalm_vectors]
    psalm_ids = [passage.id for passage in scored]
    similarity_matrix, genre_match_matrix = build_similarity_and_genre_matrices(
        psalm_ids, psalm_vectors, {passage.id: passage.gattung for passage in scored}
    )
    background = background_stats_from_matrix(similarity_matrix)
    result = block_bootstrap_genre_ap_gap_and_auc(
        psalm_ids,
        similarity_matrix,
        genre_match_matrix,
        background,
        n_resamples=n_resamples,
        rng=np.random.default_rng(seed),
        population_mask=admissible_mask(scored),
        clusters=cluster_codes(scored),
    )
    return ci_row(model, result)


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_taxonomy_arguments(parser)
    add_embeddings_dir_argument(parser)
    add_scoring_arguments(parser, with_seed=True, with_resamples=True)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    passages = load_passages(args.taxonomy, args.unit, args.labels_csv, api)

    rows, model_paths = resume_from_cache(args.embeddings_dir, args.output)
    score = partial(
        score_model,
        passages=passages,
        weights=load_half_verse_weights(passages, api),
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    rows.extend(
        row
        for row in map_in_order(
            skipping_unscorable(score), model_paths, args.workers, label="models"
        )
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
