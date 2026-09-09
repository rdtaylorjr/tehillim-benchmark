import numpy as np
import pytest

from genre.bootstrap import (
    _leave_one_out_splits,
    _resample_split,
    _upper_triangle_same_and_different,
    block_bootstrap_genre_ap_gap_and_auc,
    build_similarity_and_genre_matrices,
)
from library.ap_gap_auc_bootstrap import jackknife_statistics
from library.calibration import BackgroundStats


def _matrices() -> tuple[np.ndarray, np.ndarray]:
    psalm_ids = [1, 2, 3, 4, 5, 6]
    psalm_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.9, 0.3]),
        3: np.array([0.7, 0.5]),
        4: np.array([0.4, 0.8]),  # overlaps into genre B's region, so AP isn't perfect
        5: np.array([0.2, 0.9]),
        6: np.array([0.0, 1.0]),
    }
    genre_by_psalm = {1: "A", 2: "A", 3: "A", 4: "A", 5: "B", 6: "B"}
    return build_similarity_and_genre_matrices(psalm_ids, psalm_vectors, genre_by_psalm)


def test_build_similarity_and_genre_matrices_has_unit_diagonal_and_matching_shape() -> None:
    similarity_matrix, genre_match_matrix = _matrices()

    assert similarity_matrix.shape == (6, 6)
    assert genre_match_matrix.shape == (6, 6)
    assert np.allclose(np.diag(similarity_matrix), 1.0)
    assert np.all(np.diag(genre_match_matrix))


def test_upper_triangle_splits_same_and_different_genre_similarities() -> None:
    similarity_matrix, genre_match_matrix = _matrices()

    same_sims, different_sims = _upper_triangle_same_and_different(
        similarity_matrix, genre_match_matrix
    )

    # 6 psalms -> C(6,2)=15 pairs; genre A has C(4,2)=6 same pairs, genre B has C(2,2)=1
    assert len(same_sims) == 7
    assert len(different_sims) == 8


def test_block_bootstrap_ci_contains_the_point_estimate() -> None:
    similarity_matrix, genre_match_matrix = _matrices()
    background = BackgroundStats(mean=0.3, std=0.3, n_vectors=6)
    rng = np.random.default_rng(0)

    result = block_bootstrap_genre_ap_gap_and_auc(
        [1, 2, 3, 4, 5, 6],
        similarity_matrix,
        genre_match_matrix,
        background,
        n_resamples=200,
        rng=rng,
    )

    assert result.ap_ci_low <= result.point_ap <= result.ap_ci_high
    assert result.gap_ci_low <= result.point_gap <= result.gap_ci_high
    assert result.auc_ci_low <= result.point_auc <= result.auc_ci_high
    assert result.n_valid_resamples > 0
    assert result.prevalence == pytest.approx(7 / 15)


def test_block_bootstrap_rejects_a_population_with_no_same_genre_pairs() -> None:
    """2 psalms of different genres give 0 same-genre pairs, leaving AP and AUC undefined."""
    similarity_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    genre_match_matrix = np.array([[True, False], [False, True]])
    background = BackgroundStats(mean=0.3, std=0.3, n_vectors=2)

    with pytest.raises(ValueError, match="at least 2 values on each side"):
        block_bootstrap_genre_ap_gap_and_auc(
            [1, 2],
            similarity_matrix,
            genre_match_matrix,
            background,
            n_resamples=20,
            rng=np.random.default_rng(0),
        )


def test_block_bootstrap_returns_nan_ci_when_no_resample_is_drawn() -> None:
    """The point estimate still stands when the scheme yields nothing to build a CI from."""
    similarity_matrix, genre_match_matrix = _matrices()
    background = BackgroundStats(mean=0.3, std=0.3, n_vectors=6)

    result = block_bootstrap_genre_ap_gap_and_auc(
        [1, 2, 3, 4, 5, 6],
        similarity_matrix,
        genre_match_matrix,
        background,
        n_resamples=0,
        rng=np.random.default_rng(0),
    )

    assert result.n_valid_resamples == 0
    assert not np.isnan(result.point_ap)
    assert np.isnan(result.gap_ci_low)
    assert np.isnan(result.gap_ci_high)


def test_block_bootstrap_raises_a_clear_error_with_only_one_psalm() -> None:
    """A 1x1 similarity matrix used to crash via division by zero computing prevalence."""
    similarity_matrix = np.array([[1.0]])
    genre_match_matrix = np.array([[True]])
    background = BackgroundStats(mean=0.3, std=0.3, n_vectors=1)

    with pytest.raises(ValueError, match="no genre pairs"):
        block_bootstrap_genre_ap_gap_and_auc(
            [1],
            similarity_matrix,
            genre_match_matrix,
            background,
            n_resamples=20,
            rng=np.random.default_rng(0),
        )


def _one_vs_rest_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """6 psalms: 0,1,2 in genre A; 3=B, 4=C, 5=D. One-vs-rest population excludes B/C/D pairs."""
    similarity_matrix = np.array(
        [
            [1.00, 0.90, 0.80, 0.30, 0.20, 0.10],
            [0.90, 1.00, 0.85, 0.35, 0.25, 0.15],
            [0.80, 0.85, 1.00, 0.40, 0.30, 0.20],
            [0.30, 0.35, 0.40, 1.00, 0.50, 0.60],
            [0.20, 0.25, 0.30, 0.50, 1.00, 0.70],
            [0.10, 0.15, 0.20, 0.60, 0.70, 1.00],
        ]
    )
    is_a = np.array([True, True, True, False, False, False])
    same_mask = is_a[:, None] & is_a[None, :]
    population_mask = is_a[:, None] | is_a[None, :]
    return similarity_matrix, same_mask, population_mask


def test_upper_triangle_population_mask_excludes_pairs_touching_neither_side() -> None:
    similarity_matrix, same_mask, population_mask = _one_vs_rest_fixture()

    same_sims, different_sims = _upper_triangle_same_and_different(
        similarity_matrix, same_mask, population_mask=population_mask
    )

    # same: (0,1),(0,2),(1,2) = 3. different: A-vs-{B,C,D} = 3*3 = 9. Excludes B-C,B-D,C-D.
    assert len(same_sims) == 3
    assert len(different_sims) == 9
    assert 0.50 not in different_sims
    assert 0.60 not in different_sims
    assert 0.70 not in different_sims


def test_upper_triangle_without_population_mask_keeps_the_old_behavior() -> None:
    similarity_matrix, same_mask, _ = _one_vs_rest_fixture()

    same_sims, different_sims = _upper_triangle_same_and_different(similarity_matrix, same_mask)

    assert len(same_sims) == 3
    assert len(different_sims) == 12  # includes the 3 B/C/D-only pairs


def test_jackknife_with_population_mask_differs_from_unmasked() -> None:
    similarity_matrix, same_mask, population_mask = _one_vs_rest_fixture()
    background = BackgroundStats(mean=0.3, std=0.2, n_vectors=6)

    masked_gaps = jackknife_statistics(
        _leave_one_out_splits(6, similarity_matrix, same_mask, population_mask), background
    )[1]
    unmasked_gaps = jackknife_statistics(
        _leave_one_out_splits(6, similarity_matrix, same_mask, None), background
    )[1]

    assert not np.allclose(masked_gaps, unmasked_gaps, equal_nan=True)
    assert not np.isnan(masked_gaps).all()


def test_block_bootstrap_with_population_mask_differs_from_unmasked() -> None:
    similarity_matrix, same_mask, population_mask = _one_vs_rest_fixture()
    background = BackgroundStats(mean=0.3, std=0.2, n_vectors=6)
    psalm_ids = [1, 2, 3, 4, 5, 6]

    masked = block_bootstrap_genre_ap_gap_and_auc(
        psalm_ids,
        similarity_matrix,
        same_mask,
        background,
        n_resamples=200,
        rng=np.random.default_rng(0),
        population_mask=population_mask,
    )
    unmasked = block_bootstrap_genre_ap_gap_and_auc(
        psalm_ids,
        similarity_matrix,
        same_mask,
        background,
        n_resamples=200,
        rng=np.random.default_rng(0),
    )

    assert masked.point_gap != unmasked.point_gap
    assert masked.prevalence == pytest.approx(3 / 12)
    assert unmasked.prevalence == pytest.approx(3 / 15)
    assert masked.ap_ci_low <= masked.point_ap <= masked.ap_ci_high


def test_jackknife_returns_nan_when_removing_a_psalm_leaves_too_few_same_genre_pairs() -> None:
    """A 2/2 genre split always drops to 1 same-genre pair on leave-one-out, hence NaN."""
    similarity_matrix = np.array(
        [
            [1.0, 0.95, 0.1, 0.05],
            [0.95, 1.0, 0.05, 0.1],
            [0.1, 0.05, 1.0, 0.9],
            [0.05, 0.1, 0.9, 1.0],
        ]
    )
    genre_match_matrix = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ]
    )
    background = BackgroundStats(mean=0.3, std=0.2, n_vectors=4)

    aps, gaps, aucs = jackknife_statistics(
        _leave_one_out_splits(4, similarity_matrix, genre_match_matrix, None), background
    )

    assert np.isnan(aps).all()
    assert np.isnan(gaps).all()
    assert np.isnan(aucs).all()


def test_resample_split_excludes_pairs_of_a_psalm_with_its_own_duplicate() -> None:
    """A psalm paired with its own copy would inject self-similarity 1.0 as a positive."""
    similarity_matrix = np.array([[1.0, 0.4], [0.4, 1.0]])
    genre_match_matrix = np.array([[True, True], [True, True]])

    same_sims, different_sims = _resample_split(
        np.array([0, 0, 1]), similarity_matrix, genre_match_matrix, None
    )

    assert sorted(same_sims.tolist()) == [0.4, 0.4]
    assert len(different_sims) == 0


def test_resample_split_keeps_repeated_pairs_between_two_distinct_psalms() -> None:
    """Only a psalm paired with itself is dropped; a genuine pair drawn twice still counts twice."""
    similarity_matrix = np.array([[1.0, 0.4], [0.4, 1.0]])
    genre_match_matrix = np.array([[True, False], [False, True]])

    same_sims, different_sims = _resample_split(
        np.array([0, 1, 0, 1]), similarity_matrix, genre_match_matrix, None
    )

    assert len(same_sims) == 0
    assert different_sims.tolist() == [0.4, 0.4, 0.4, 0.4]


def _tied_similarity_fixture() -> tuple[np.ndarray, np.ndarray]:
    """8 psalms, 4 per genre, every off-diagonal similarity tied, so there is no real signal."""
    n = 8
    similarity_matrix = np.full((n, n), 0.4)
    np.fill_diagonal(similarity_matrix, 1.0)
    genres = np.array(["A"] * 4 + ["B"] * 4)
    return similarity_matrix, genres[:, None] == genres[None, :]


def test_bootstrap_of_a_zero_signal_matrix_never_ranks_same_genre_above_chance() -> None:
    """Duplicate-psalm self-pairs used to inject similarity-1.0 positives and invent an AUC gap."""
    similarity_matrix, genre_match_matrix = _tied_similarity_fixture()
    background = BackgroundStats(mean=0.3, std=0.2, n_vectors=8)

    result = block_bootstrap_genre_ap_gap_and_auc(
        list(range(8)),
        similarity_matrix,
        genre_match_matrix,
        background,
        n_resamples=300,
        rng=np.random.default_rng(0),
    )

    assert result.point_auc == pytest.approx(0.5)
    assert result.auc_ci_low_pct == pytest.approx(0.5)
    assert result.auc_ci_high_pct == pytest.approx(0.5)
    assert result.auc_ci_low <= result.point_auc <= result.auc_ci_high


def test_bootstrap_of_a_zero_signal_matrix_keeps_the_calibrated_gap_at_zero() -> None:
    """Same-genre and different-genre means are tied here, so every resample gap is exactly 0."""
    similarity_matrix, genre_match_matrix = _tied_similarity_fixture()
    background = BackgroundStats(mean=0.3, std=0.2, n_vectors=8)

    result = block_bootstrap_genre_ap_gap_and_auc(
        list(range(8)),
        similarity_matrix,
        genre_match_matrix,
        background,
        n_resamples=300,
        rng=np.random.default_rng(0),
    )

    assert result.point_gap == pytest.approx(0.0)
    assert result.gap_ci_low_pct == pytest.approx(0.0)
    assert result.gap_ci_high_pct == pytest.approx(0.0)
