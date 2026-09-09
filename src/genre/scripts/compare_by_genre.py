"""One-vs-rest genre discrimination per model: AP/AUC, jackknife CIs, and permutation p-values."""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from genre.across_genres import (
    GenreRunConfig,
    compare_model_across_genres,
)
from genre.genre_labels import load_genre_by_psalm
from genre.pairs import build_genre_pairs
from library.bhsa import list_psalms_half_verses_by_psalm, load_bhsa_api
from library.cli import (
    add_embeddings_dir_argument,
    add_genre_csv_argument,
    add_scoring_arguments,
    report_reuse,
)
from library.embeddings import dataset_identifier
from library.incremental_cache import load_cached_rows
from library.model_files import uncached_model_paths
from library.multiple_comparisons import add_source_q_columns
from library.psalm_vectors import load_psalm_vectors
from library.rows_output import write_rows_csv
from library.scoring import skipping_unscorable
from library.worker_pool import map_in_order

_SOURCES = ("naive", "perm", "maxT")


_RAW_COLUMNS = (
    "model",
    "genre",
    "n_same_genre",
    "n_different_genre",
    "prevalence",
    "average_precision",
    "ap_ci_low",
    "ap_ci_high",
    "separation_auc",
    "auc_ci_low",
    "auc_ci_high",
    "separation_p_naive",
    "separation_p_perm",
    "separation_p_maxT",
    "n_permutations",
)


def load_cached_genre_rows(cache_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Reads a prior output CSV's raw rows (dropping stale q-values) and its covered model set."""
    return load_cached_rows(cache_path, columns=_RAW_COLUMNS)


def add_fdr_columns(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Adds BH/BY q-values to every source, corrected within each scope's family."""
    return add_source_q_columns(
        rows,
        sources=_SOURCES,
        scope_column="genre",
        p_value_template="separation_p_{source}",
    )


def score_model(
    path: Path,
    half_verses_by_psalm: dict[int, list[int]],
    config: GenreRunConfig,
) -> list[dict[str, str | int | float]]:
    """One row per genre for one model file, scored independently of every other model."""
    model = dataset_identifier(path)
    psalm_vectors = load_psalm_vectors(path, half_verses_by_psalm)
    psalm_ids = sorted(set(config.genre_by_psalm) & set(psalm_vectors))
    return compare_model_across_genres(model, psalm_ids, psalm_vectors, config)


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_genre_csv_argument(parser)
    add_embeddings_dir_argument(parser)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="prior output CSV to reuse; defaults to --output, so reruns cache automatically",
    )
    add_scoring_arguments(parser, with_seed=True, with_group_permutations=True, with_resamples=True)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    pairs = build_genre_pairs(genre_by_psalm)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)
    genres = tuple(sorted(set(genre_by_psalm.values())))

    cache_path = args.cache or args.output
    rows, cached_models = load_cached_genre_rows(cache_path) if cache_path else ([], set())
    report_reuse(cached_models, cache_path)

    model_paths = uncached_model_paths(args.embeddings_dir, cached_models)
    config = GenreRunConfig(
        genre_by_psalm=genre_by_psalm,
        genres=genres,
        pairs=pairs,
        n_permutations=args.n_permutations,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    score = partial(score_model, half_verses_by_psalm=half_verses_by_psalm, config=config)
    for model_rows in map_in_order(skipping_unscorable(score), model_paths, args.workers):
        if model_rows is None:
            continue
        rows.extend(model_rows)

    result_df = add_fdr_columns(rows)

    ordered = result_df.sort_values(["genre", "average_precision"], ascending=[True, False])
    for _, row in ordered.iterrows():
        print(
            f"{row['genre']:15s} {row['model']:55s} "
            f"AP={row['average_precision']:.4f} [{row['ap_ci_low']:.4f},{row['ap_ci_high']:.4f}] "
            f"(chance={row['prevalence']:.3f}) "
            f"p_naive={row['separation_p_naive']:.4f} p_perm={row['separation_p_perm']:.4f} "
            f"p_maxT={row['separation_p_maxT']:.4f}"
        )

    if args.output:
        write_rows_csv(args.output, result_df.to_dict("records"))


if __name__ == "__main__":
    main()
