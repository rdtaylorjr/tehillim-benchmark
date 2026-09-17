"""Scores same-genre vs different-genre psalm-pair similarity: AP (primary) and AUC (secondary)."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from genre.bootstrap import psalm_similarity_matrix
from genre.pairs import GenrePair
from library.fast_metrics import fast_average_precision
from library.retrieval_metrics import sparse_cosine_similarity_matrix


@dataclass(frozen=True, slots=True)
class GenreEvaluationReport:
    """One model's genre discrimination under the MTEB Pair Classification protocol."""

    n_same_genre: int
    n_different_genre: int
    prevalence: float
    average_precision: float
    separation_auc: float
    separation_p: float


def report_from_similarities(
    usable: list[GenrePair], similarities: np.ndarray
) -> GenreEvaluationReport:
    """Shared AP/AUC report-building step for both the vector- and matrix-based entry points."""
    labels = np.array([p.same_genre for p in usable], dtype=int)
    same_sims = similarities[labels == 1]
    different_sims = similarities[labels == 0]

    #: Spacing representations are all-zero for most word-level psalms, leaving no pair to rank.
    if len(same_sims) == 0 or len(different_sims) == 0:
        return GenreEvaluationReport(
            n_same_genre=len(same_sims),
            n_different_genre=len(different_sims),
            prevalence=len(same_sims) / len(usable) if usable else float("nan"),
            average_precision=float("nan"),
            separation_auc=float("nan"),
            separation_p=float("nan"),
        )

    ap = average_precision_score(labels, similarities)
    statistic, p_value = mannwhitneyu(same_sims, different_sims, alternative="greater")
    auc = statistic / (len(same_sims) * len(different_sims))

    return GenreEvaluationReport(
        n_same_genre=len(same_sims),
        n_different_genre=len(different_sims),
        prevalence=len(same_sims) / len(usable),
        average_precision=float(ap),
        separation_auc=float(auc),
        separation_p=float(p_value),
    )


def pair_similarities(
    pairs: list[GenrePair], psalm_vectors: dict[str, np.ndarray]
) -> tuple[list[GenrePair], np.ndarray]:
    """Similarity of every usable pair, read off one psalm-by-psalm matrix, not stacked rows."""
    usable = [p for p in pairs if p.item_a in psalm_vectors and p.item_b in psalm_vectors]
    if not usable:
        return usable, np.empty(0)
    psalm_ids = sorted(psalm_vectors)
    index = {psalm: position for position, psalm in enumerate(psalm_ids)}
    matrix = psalm_similarity_matrix(psalm_ids, psalm_vectors)
    rows = np.fromiter((index[p.item_a] for p in usable), dtype=np.intp, count=len(usable))
    cols = np.fromiter((index[p.item_b] for p in usable), dtype=np.intp, count=len(usable))
    return usable, matrix[rows, cols]


def evaluate_genre_discrimination(
    pairs: list[GenrePair], psalm_vectors: dict[str, np.ndarray]
) -> GenreEvaluationReport:
    """MTEB Pair Classification protocol: AP ranks same-genre pairs above different-genre pairs."""
    usable, similarities = pair_similarities(pairs, psalm_vectors)
    return report_from_similarities(usable, similarities)


def evaluate_genre_discrimination_sparse(
    pairs: list[GenrePair], psalm_ids: list[str], psalm_vectors: sp.csr_matrix
) -> GenreEvaluationReport:
    """Same report as evaluate_genre_discrimination, comparing sparse psalm vectors once."""
    psalm_index = {p: i for i, p in enumerate(psalm_ids)}
    similarity_matrix = sparse_cosine_similarity_matrix(psalm_vectors, psalm_vectors)
    return evaluate_genre_discrimination_from_matrix(pairs, similarity_matrix, psalm_index)


def evaluate_genre_discrimination_from_matrix(
    pairs: list[GenrePair], similarity_matrix: np.ndarray, psalm_index: dict[str, int]
) -> GenreEvaluationReport:
    """Same report as evaluate_genre_discrimination, indexing an already-computed matrix instead."""
    usable = [p for p in pairs if p.item_a in psalm_index and p.item_b in psalm_index]
    similarities = np.array(
        [similarity_matrix[psalm_index[p.item_a], psalm_index[p.item_b]] for p in usable]
    )
    return report_from_similarities(usable, similarities)


@dataclass(frozen=True, slots=True)
class GenrePairIndex:
    """The pair list as arrays: endpoint ids, sameness, and the two genres, indexed once."""

    item_a: np.ndarray
    item_b: np.ndarray
    same_genre: np.ndarray
    genre_a: np.ndarray
    genre_b: np.ndarray


def index_genre_pairs(pairs: list[GenrePair]) -> GenrePairIndex:
    """Turns the pair list into arrays, so every later restriction is a mask, not a loop."""
    return GenrePairIndex(
        item_a=np.array([p.item_a for p in pairs], dtype=object),
        item_b=np.array([p.item_b for p in pairs], dtype=object),
        same_genre=np.array([p.same_genre for p in pairs], dtype=bool),
        genre_a=np.array([p.genre_a for p in pairs], dtype=object),
        genre_b=np.array([p.genre_b for p in pairs], dtype=object),
    )


def average_precision_by_genre(
    index: GenrePairIndex,
    similarity_matrix: np.ndarray,
    psalm_ids: list[str],
    genres: list[str],
) -> dict[str, float]:
    """Per-genre one-vs-rest AP over the usable pairs, the per-genre report's statistic."""
    position = {psalm: i for i, psalm in enumerate(psalm_ids)}
    row = np.array([position.get(item, -1) for item in index.item_a], dtype=np.intp)
    col = np.array([position.get(item, -1) for item in index.item_b], dtype=np.intp)
    usable = (row >= 0) & (col >= 0)
    similarities = similarity_matrix[row[usable], col[usable]]
    same = index.same_genre[usable]
    genre_a, genre_b = index.genre_a[usable], index.genre_b[usable]
    scores: dict[str, float] = {}
    for genre in genres:
        population = (genre_a == genre) | (genre_b == genre)
        positive = similarities[population & same]
        negative = similarities[population & ~same]
        if len(positive) == 0 or len(negative) == 0:
            scores[genre] = float("nan")
        else:
            scores[genre] = fast_average_precision(positive, negative)
    return scores
