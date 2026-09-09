import numpy as np
import pytest

from library.errors import InsufficientDataError
from parallelism.separation import similarity_separation


def test_similarity_separation_is_perfect_when_true_pairs_dominate() -> None:
    sim = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    result = similarity_separation(sim)

    assert result.auc == 1.0
    assert result.n_positive == 3
    assert result.n_negative == 6


def test_similarity_separation_is_half_when_indistinguishable() -> None:
    sim = np.full((4, 4), 0.5)

    result = similarity_separation(sim)

    assert result.auc == 0.5


def test_similarity_separation_is_low_when_true_pairs_are_less_similar() -> None:
    sim = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    result = similarity_separation(sim)

    assert result.auc == 0.0


def test_similarity_separation_negatives_match_a_naive_per_row_loop_at_scale() -> None:
    rng = np.random.default_rng(31)
    n = 30
    sim = rng.normal(size=(n, n))
    row_mask = rng.integers(0, 2, size=n).astype(bool)

    off_diagonal = ~np.eye(n, dtype=bool)
    rows = np.flatnonzero(row_mask)
    naive_negative = np.concatenate([sim[i, off_diagonal[i]] for i in rows])

    result = similarity_separation(sim, row_mask=row_mask)

    assert result.n_negative == len(naive_negative)


def test_similarity_separation_restricts_positives_with_a_row_mask() -> None:
    """Only masked rows count as positives, while their off-diagonal entries stay negatives."""
    sim = np.array(
        [
            [0.9, 0.1, 0.1],
            [0.1, 0.2, 0.1],
            [0.1, 0.1, 0.9],
        ]
    )

    result = similarity_separation(sim, row_mask=np.array([True, False, True]))

    assert result.n_positive == 2
    assert result.n_negative == 4
    assert result.auc == 1.0


def test_similarity_separation_rejects_a_single_pair() -> None:
    """One pair leaves no off-diagonal negative, so the AUC is undefined, not NaN."""
    with pytest.raises(InsufficientDataError, match="at least 2"):
        similarity_separation(np.array([[1.0]]))
