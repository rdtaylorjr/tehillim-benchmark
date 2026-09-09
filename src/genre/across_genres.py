"""Scores one model against each genre one-vs-rest, with its CI and permutation inference."""

import sys
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from genre.bootstrap import block_bootstrap_genre_ap_gap_and_auc, psalm_similarity_matrix
from genre.evaluate import evaluate_genre_discrimination_from_matrix
from genre.pairs import GenrePair, filter_pairs_by_genre
from genre.permutation import joint_psalm_label_permutation_test, one_vs_rest_masks
from library.ap_gap_auc_bootstrap import ApGapAucCI
from library.calibration import BackgroundStats, background_stats_from_matrix
from library.errors import InsufficientDataError
from library.retrieval_metrics import sparse_cosine_similarity_matrix


@dataclass(frozen=True, slots=True)
class GenreRunConfig:
    """The labels and resampling budget every model in one run is scored against."""

    genre_by_psalm: dict[int, str]
    genres: tuple[str, ...]
    pairs: list[GenrePair]
    n_permutations: int
    n_resamples: int
    seed: int


def _bootstrap_ci_or_none(
    psalm_ids: list[int],
    similarity_matrix: np.ndarray,
    same_mask: np.ndarray,
    population_mask: np.ndarray,
    background: BackgroundStats,
    n_resamples: int,
    seed: int,
    genre: str,
) -> ApGapAucCI | None:
    """None for a genre whose one-vs-rest population is too small to define a CI, not a crash."""
    try:
        return block_bootstrap_genre_ap_gap_and_auc(
            psalm_ids,
            similarity_matrix,
            same_mask,
            background,
            n_resamples=n_resamples,
            rng=np.random.default_rng(seed),
            population_mask=population_mask,
        )
    except InsufficientDataError as error:
        print(f"no bootstrap CI for genre {genre!r}: {error}", file=sys.stderr)
        return None


def _compare_from_similarity_matrix(
    model: str,
    psalm_ids: list[int],
    similarity_matrix: np.ndarray,
    config: GenreRunConfig,
) -> list[dict[str, str | int | float]]:
    """Shared per-genre report step for both the dense and sparse psalm-vector entry points."""
    genres = config.genres
    psalm_index = {p: i for i, p in enumerate(psalm_ids)}
    code_of = {genre: index for index, genre in enumerate(genres)}
    genre_codes = np.array([code_of[config.genre_by_psalm[p]] for p in psalm_ids])
    background = background_stats_from_matrix(similarity_matrix)

    perm_result = joint_psalm_label_permutation_test(
        similarity_matrix,
        genre_codes,
        genres,
        n_permutations=config.n_permutations,
        rng=np.random.default_rng(config.seed),
    )

    rows: list[dict[str, str | int | float]] = []
    for index, genre in enumerate(genres):
        restricted = filter_pairs_by_genre(config.pairs, genre)
        report = evaluate_genre_discrimination_from_matrix(
            restricted, similarity_matrix, psalm_index
        )

        same_mask, population_mask = one_vs_rest_masks(genre_codes, index)
        ci = _bootstrap_ci_or_none(
            psalm_ids,
            similarity_matrix,
            same_mask,
            population_mask,
            background,
            config.n_resamples,
            config.seed,
            genre,
        )

        rows.append(
            {
                "model": model,
                "genre": genre,
                "n_same_genre": report.n_same_genre,
                "n_different_genre": report.n_different_genre,
                "prevalence": report.prevalence,
                "average_precision": report.average_precision,
                "ap_ci_low": ci.ap_ci_low if ci is not None else float("nan"),
                "ap_ci_high": ci.ap_ci_high if ci is not None else float("nan"),
                "separation_auc": report.separation_auc,
                "auc_ci_low": ci.auc_ci_low if ci is not None else float("nan"),
                "auc_ci_high": ci.auc_ci_high if ci is not None else float("nan"),
                "separation_p_naive": report.separation_p,
                "separation_p_perm": perm_result.p_perm[index],
                "separation_p_maxT": perm_result.p_maxt[index],
                "n_permutations": config.n_permutations,
            }
        )
    return rows


def compare_model_across_genres(
    model: str,
    psalm_ids: list[int],
    psalm_vectors: dict[int, np.ndarray],
    config: GenreRunConfig,
) -> list[dict[str, str | int | float]]:
    """One row per genre: AP (point, unchanged), AUC, jackknife CIs, and three p-value sources."""
    similarity_matrix = psalm_similarity_matrix(psalm_ids, psalm_vectors)
    return _compare_from_similarity_matrix(model, psalm_ids, similarity_matrix, config)


def compare_model_across_genres_sparse(
    model: str,
    psalm_ids: list[int],
    psalm_vectors: sp.csr_matrix,
    config: GenreRunConfig,
) -> list[dict[str, str | int | float]]:
    """Same rows as compare_model_across_genres, comparing sparse psalm vectors, never densified."""
    similarity_matrix = sparse_cosine_similarity_matrix(psalm_vectors, psalm_vectors)
    return _compare_from_similarity_matrix(model, psalm_ids, similarity_matrix, config)
