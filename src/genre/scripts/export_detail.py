"""Exports row-per-pair genre-pair detail plus a per-model AP/AUC/calibration summary."""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genre.calibrated import compare_genre_calibrated, genre_calibrated_row
from genre.evaluate import pair_similarities
from genre.genre_labels import load_genre_by_psalm
from genre.pairs import GenrePair, build_genre_pairs
from library.bhsa import list_psalms_half_verses_by_psalm, load_bhsa_api
from library.calibration import BackgroundStats, background_similarity_stats, calibrated_z_score
from library.cli import (
    add_embeddings_dir_argument,
    add_genre_csv_argument,
    add_scoring_arguments,
    report_reuse,
)
from library.embeddings import dataset_identifier
from library.incremental_cache import load_cached_parquet_set
from library.model_files import uncached_model_paths
from library.psalm_vectors import load_psalm_vectors
from library.rows_output import write_dataframe_parquet
from library.scoring import skipping_unscorable
from library.worker_pool import map_in_order

_OUTPUT_FILES = ("genre_pair_detail.parquet", "genre_summary.parquet")


def build_pair_detail_rows(
    model: str,
    pairs: list[GenrePair],
    psalm_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """One row per usable pair: raw similarity, calibrated z, and a same_genre flag only."""
    usable, similarities = pair_similarities(pairs, psalm_vectors)
    rows = []
    for pair, sim in zip(usable, similarities, strict=True):
        rows.append(
            {
                "model": model,
                "psalm_a": pair.psalm_a,
                "psalm_b": pair.psalm_b,
                "same_genre": pair.same_genre,
                "raw_similarity": float(sim),
                "calibrated_z": calibrated_z_score(float(sim), background),
            }
        )
    return rows


def build_summary_rows(
    model: str,
    pairs: list[GenrePair],
    psalm_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """Single-row-per-model AP/AUC/calibration summary, wrapping compare_genre_calibrated."""
    return [genre_calibrated_row(model, compare_genre_calibrated(pairs, psalm_vectors, background))]


def score_model(
    path: Path,
    half_verses_by_psalm: dict[int, list[int]],
    pairs: list[GenrePair],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One model file's (pair rows, summary rows), scored independently of every other model."""
    model = dataset_identifier(path)
    psalm_vectors = load_psalm_vectors(path, half_verses_by_psalm)
    background = background_similarity_stats(np.stack(list(psalm_vectors.values())))
    return (
        build_pair_detail_rows(model, pairs, psalm_vectors, background),
        build_summary_rows(model, pairs, psalm_vectors, background),
    )


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_genre_csv_argument(parser)
    add_embeddings_dir_argument(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_scoring_arguments(parser)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = api_factory(args.checkout)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    pairs = build_genre_pairs(genre_by_psalm)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)

    (cached_pair_rows, cached_summary_rows), cached_models = load_cached_parquet_set(
        args.output_dir, _OUTPUT_FILES
    )
    report_reuse(cached_models, args.output_dir)

    model_paths = uncached_model_paths(args.embeddings_dir, cached_models)
    pair_rows: list[dict[str, Any]] = list(cached_pair_rows)
    summary_rows: list[dict[str, Any]] = list(cached_summary_rows)
    score = partial(score_model, half_verses_by_psalm=half_verses_by_psalm, pairs=pairs)
    for scored in map_in_order(skipping_unscorable(score), model_paths, args.workers):
        if scored is None:
            continue
        model_pair_rows, model_summary_rows = scored
        pair_rows.extend(model_pair_rows)
        summary_rows.extend(model_summary_rows)

    write_dataframe_parquet(args.output_dir / "genre_pair_detail.parquet", pd.DataFrame(pair_rows))
    write_dataframe_parquet(args.output_dir / "genre_summary.parquet", pd.DataFrame(summary_rows))
    print(f"wrote {len(pair_rows)} pair rows, {len(summary_rows)} summary rows")


if __name__ == "__main__":
    main()
