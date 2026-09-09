import math

import numpy as np
import pytest
import scipy.sparse as sp

from genre.evaluate import (
    evaluate_genre_discrimination,
    evaluate_genre_discrimination_from_matrix,
    evaluate_genre_discrimination_sparse,
)
from genre.pairs import GenrePair, build_genre_pairs
from library.retrieval_metrics import cosine_similarity_matrix


def test_reports_the_correct_same_and_different_genre_counts() -> None:
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(2, 3, "Lament", "Praise", same_genre=False),
    ]
    psalm_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.9, 0.1]),
        3: np.array([0.0, 1.0]),
    }

    report = evaluate_genre_discrimination(pairs, psalm_vectors)

    assert report.n_same_genre == 1
    assert report.n_different_genre == 2
    assert report.prevalence == pytest.approx(1 / 3)


def _matrix_and_index(psalm_vectors: dict[int, np.ndarray]) -> tuple[np.ndarray, dict[int, int]]:
    psalm_ids = sorted(psalm_vectors)
    vectors = np.stack([psalm_vectors[p] for p in psalm_ids])
    return cosine_similarity_matrix(vectors, vectors), {p: i for i, p in enumerate(psalm_ids)}


class TestEvaluateGenreDiscriminationFromMatrix:
    def test_matches_the_vector_based_report_exactly(self) -> None:
        pairs = [
            GenrePair(1, 2, "Lament", "Lament", same_genre=True),
            GenrePair(3, 4, "Praise", "Praise", same_genre=True),
            GenrePair(1, 3, "Lament", "Praise", same_genre=False),
            GenrePair(1, 4, "Lament", "Praise", same_genre=False),
            GenrePair(2, 3, "Lament", "Praise", same_genre=False),
            GenrePair(2, 4, "Lament", "Praise", same_genre=False),
        ]
        psalm_vectors = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.8, 0.3]),
            3: np.array([0.0, 1.0]),
            4: np.array([0.1, 0.9]),
        }

        from_vectors = evaluate_genre_discrimination(pairs, psalm_vectors)
        similarity_matrix, psalm_index = _matrix_and_index(psalm_vectors)
        from_matrix = evaluate_genre_discrimination_from_matrix(
            pairs, similarity_matrix, psalm_index
        )

        assert from_matrix == from_vectors

    def test_skips_a_pair_whose_psalm_is_not_in_the_index(self) -> None:
        pairs = [
            GenrePair(1, 2, "Lament", "Lament", same_genre=True),
            GenrePair(1, 3, "Lament", "Praise", same_genre=False),
            GenrePair(1, 4, "Lament", "Praise", same_genre=False),
            GenrePair(2, 3, "Lament", "Praise", same_genre=False),
            GenrePair(2, 4, "Lament", "Praise", same_genre=False),
            GenrePair(3, 4, "Praise", "Praise", same_genre=True),
        ]
        psalm_vectors = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.9, 0.1]),
            3: np.array([0.0, 1.0]),
            # psalm 4 has no vector: every pair touching it must be excluded.
        }
        similarity_matrix, psalm_index = _matrix_and_index(psalm_vectors)

        report = evaluate_genre_discrimination_from_matrix(pairs, similarity_matrix, psalm_index)

        assert report.n_same_genre == 1
        assert report.n_different_genre == 2


def test_perfect_separation_gives_ap_and_auc_of_one() -> None:
    """Same-genre psalms point the same direction, different-genre psalms are orthogonal."""
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(3, 4, "Praise", "Praise", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(1, 4, "Lament", "Praise", same_genre=False),
        GenrePair(2, 3, "Lament", "Praise", same_genre=False),
        GenrePair(2, 4, "Lament", "Praise", same_genre=False),
    ]
    psalm_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([0.0, 1.0]),
        4: np.array([0.0, 1.0]),
    }

    report = evaluate_genre_discrimination(pairs, psalm_vectors)

    assert report.average_precision == pytest.approx(1.0)
    assert report.separation_auc == pytest.approx(1.0)


def test_random_labels_give_chance_level_auc() -> None:
    """Same-genre and different-genre pairs drawn from identical similarity distributions."""
    rng = np.random.default_rng(0)
    psalm_vectors = {i: rng.normal(size=8) for i in range(1, 41)}
    genres = ["Lament", "Praise"] * 20
    pairs = []
    for i in range(1, 41):
        for j in range(i + 1, 41):
            genre_a, genre_b = genres[i - 1], genres[j - 1]
            pairs.append(GenrePair(i, j, genre_a, genre_b, same_genre=genre_a == genre_b))

    report = evaluate_genre_discrimination(pairs, psalm_vectors)

    assert 0.35 < report.separation_auc < 0.65


def test_evaluate_genre_discrimination_sparse_matches_the_dense_function_exactly() -> None:
    """Proves the sparse psalm-similarity path reports identically to the dense one."""
    rng = np.random.default_rng(5)
    dim = 300
    psalm_ids = [1, 2, 3, 4]
    dense_vectors = {}
    for p in psalm_ids:
        row = np.zeros(dim)
        n_nonzero = rng.integers(1, 6)
        idx = rng.choice(dim, size=n_nonzero, replace=False)
        row[idx] = rng.uniform(0.1, 5.0, size=n_nonzero)
        dense_vectors[p] = row
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(3, 4, "Praise", "Praise", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(1, 4, "Lament", "Praise", same_genre=False),
        GenrePair(2, 3, "Lament", "Praise", same_genre=False),
        GenrePair(2, 4, "Lament", "Praise", same_genre=False),
    ]
    sparse_matrix = sp.csr_matrix(np.stack([dense_vectors[p] for p in psalm_ids]))

    dense_report = evaluate_genre_discrimination(pairs, dense_vectors)
    sparse_report = evaluate_genre_discrimination_sparse(pairs, psalm_ids, sparse_matrix)

    assert sparse_report == dense_report


def test_skips_a_pair_whose_psalm_has_no_vector() -> None:
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(1, 4, "Lament", "Praise", same_genre=False),
        GenrePair(2, 3, "Lament", "Praise", same_genre=False),
        GenrePair(2, 4, "Lament", "Praise", same_genre=False),
        GenrePair(3, 4, "Praise", "Praise", same_genre=True),
    ]
    psalm_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.9, 0.1]),
        3: np.array([0.0, 1.0]),
        # psalm 4 has no vector: every pair touching it must be excluded.
    }

    report = evaluate_genre_discrimination(pairs, psalm_vectors)

    assert report.n_same_genre == 1
    assert report.n_different_genre == 2


def test_evaluate_genre_discrimination_sparse_agrees_with_dense_at_realistic_density() -> None:
    """The exact-equality case has at most 5 nonzeros a row; dense rows agree only to float32."""
    rng = np.random.default_rng(11)
    dim = 512
    psalm_ids = list(range(1, 13))
    dense_vectors = {p: rng.standard_normal(dim) for p in psalm_ids}
    pairs = [
        GenrePair(
            a, b, "Lament", "Lament" if (a + b) % 2 else "Praise", same_genre=(a + b) % 2 == 1
        )
        for a in psalm_ids
        for b in psalm_ids
        if a < b
    ]
    sparse_matrix = sp.csr_matrix(np.stack([dense_vectors[p] for p in psalm_ids]))

    dense_report = evaluate_genre_discrimination(pairs, dense_vectors)
    sparse_report = evaluate_genre_discrimination_sparse(pairs, psalm_ids, sparse_matrix)

    assert sparse_report.n_same_genre == dense_report.n_same_genre
    assert sparse_report.n_different_genre == dense_report.n_different_genre
    assert sparse_report.average_precision == pytest.approx(
        dense_report.average_precision, abs=1e-6
    )
    assert sparse_report.separation_auc == pytest.approx(dense_report.separation_auc, abs=1e-6)


def test_reports_nan_when_a_representation_leaves_no_usable_pairs() -> None:
    """Spacing features are all-zero for most word-level psalms, so a genre can lose every pair."""
    report = evaluate_genre_discrimination([], {})

    assert report.n_same_genre == 0
    assert report.n_different_genre == 0
    assert math.isnan(report.average_precision)
    assert math.isnan(report.separation_auc)
    assert math.isnan(report.separation_p)
    assert math.isnan(report.prevalence)


def test_reports_nan_when_one_side_of_the_comparison_is_empty() -> None:
    """A genre whose psalms all survive but whose counterpart set is empty has no AUC to report."""
    pairs = [GenrePair(psalm_a=1, psalm_b=2, genre_a="Hymn", genre_b="Hymn", same_genre=True)]
    vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}

    report = evaluate_genre_discrimination(pairs, vectors)

    assert report.n_same_genre == 1
    assert report.n_different_genre == 0
    assert math.isnan(report.separation_auc)


def test_dense_and_matrix_entry_points_return_the_same_report_exactly() -> None:
    """One metric must have one implementation, or the two routes drift at tie boundaries."""
    from genre.bootstrap import psalm_similarity_matrix

    rng = np.random.default_rng(5)
    psalm_ids = list(range(1, 41))
    vectors = {p: rng.standard_normal(64).astype(np.float32) for p in psalm_ids}
    genres = {p: "A" if p % 3 else "B" for p in psalm_ids}
    pairs = build_genre_pairs(genres)

    dense = evaluate_genre_discrimination(pairs, vectors)
    ordered = sorted(vectors)
    from_matrix = evaluate_genre_discrimination_from_matrix(
        pairs, psalm_similarity_matrix(ordered, vectors), {p: i for i, p in enumerate(ordered)}
    )

    assert dense == from_matrix


def test_dense_entry_point_does_not_stack_one_row_per_pair() -> None:
    """150 psalms make 11175 pairs, so stacking pair rows duplicates each vector 149 times."""
    import tracemalloc

    rng = np.random.default_rng(6)
    psalm_ids = list(range(1, 151))
    dim = 2048
    vectors = {p: rng.standard_normal(dim).astype(np.float32) for p in psalm_ids}
    genres = {p: "A" if p % 3 else "B" for p in psalm_ids}
    pairs = build_genre_pairs(genres)
    one_row_per_pair = len(pairs) * dim * 4

    tracemalloc.start()
    evaluate_genre_discrimination(pairs, vectors)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < one_row_per_pair // 4, (
        f"peak {peak} implies a per-pair stack of {one_row_per_pair}"
    )


def test_pair_similarities_reads_every_pair_off_one_psalm_matrix() -> None:
    """The one place pair similarity is derived, so no caller stacks a row per pair again."""
    from genre.evaluate import pair_similarities

    rng = np.random.default_rng(9)
    vectors = {p: rng.standard_normal(32).astype(np.float32) for p in range(1, 21)}
    genres = {p: "A" if p % 2 else "B" for p in vectors}
    pairs = build_genre_pairs(genres)

    usable, similarities = pair_similarities(pairs, vectors)

    assert len(usable) == len(pairs)
    assert similarities.shape == (len(pairs),)
    assert similarities.dtype == np.float64


def test_pair_similarities_drops_pairs_whose_psalms_have_no_vector() -> None:
    from genre.evaluate import pair_similarities

    rng = np.random.default_rng(10)
    vectors = {p: rng.standard_normal(16).astype(np.float32) for p in (1, 2, 3)}
    pairs = build_genre_pairs({1: "A", 2: "A", 3: "B", 4: "B"})

    usable, similarities = pair_similarities(pairs, vectors)

    assert all(p.psalm_a in vectors and p.psalm_b in vectors for p in usable)
    assert len(similarities) == len(usable) == 3


def test_pair_similarities_returns_empty_when_no_pair_is_usable() -> None:
    from genre.evaluate import pair_similarities

    usable, similarities = pair_similarities(build_genre_pairs({1: "A", 2: "B"}), {})

    assert usable == []
    assert similarities.shape == (0,)
