from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from genre.permutation import (
    _batched_separation,
    _one_vs_rest_auc,
    joint_psalm_label_permutation_test,
    one_vs_rest_masks,
)


def test_raises_a_clear_error_instead_of_crashing_when_only_one_psalm_survives_filtering() -> None:
    """A 1x1 similarity matrix used to crash in _batched_separation via empty triu_indices."""
    similarity_matrix = np.array([[1.0]])
    genre_codes = np.array([0])

    with pytest.raises(ValueError, match="at least 2 psalms"):
        joint_psalm_label_permutation_test(
            similarity_matrix, genre_codes, ("A",), rng=np.random.default_rng(0)
        )


def test_one_vs_rest_masks_matches_touches_genre_semantics() -> None:
    # psalms 0,1 = genre 0 (A); 2 = genre 1 (B); 3 = genre 2 (C)
    genre_codes = np.array([0, 0, 1, 2])

    same_mask, population_mask = one_vs_rest_masks(genre_codes, genre_index=0)

    # same: (0,1). population: pairs touching A. Excludes (2,3), which touches neither.
    assert same_mask.tolist() == [
        [True, True, False, False],
        [True, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
    assert population_mask[2, 3] is np.False_ or not population_mask[2, 3]
    assert population_mask[0, 2]
    assert population_mask[0, 3]
    assert population_mask[1, 2]


def _strong_signal_fixture() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    # 12 psalms; genre A (0-3) tightly clustered, B (4-7) and C (8-11) each their own cluster.
    n = 12
    rng = np.random.default_rng(42)
    base = np.zeros((n, n))
    codes = np.array([0] * 4 + [1] * 4 + [2] * 4)
    for i in range(n):
        for j in range(n):
            if i == j:
                base[i, j] = 1.0
            elif codes[i] == codes[j]:
                base[i, j] = 0.9 + rng.uniform(-0.02, 0.02)
            else:
                base[i, j] = 0.1 + rng.uniform(-0.02, 0.02)
    base = (base + base.T) / 2
    np.fill_diagonal(base, 1.0)
    return base, codes, ("A", "B", "C")


def test_batched_separation_matches_a_naive_per_permutation_loop_no_ties() -> None:
    rng = np.random.default_rng(51)
    n = 16
    rows, cols = np.triu_indices(n, k=1)
    sims = rng.normal(size=len(rows))
    is_target_batch = rng.integers(0, 2, size=(40, n)).astype(bool)

    naive = np.empty(40)
    for b in range(40):
        is_target = is_target_batch[b]
        same_b = is_target[rows] & is_target[cols]
        population_b = is_target[rows] | is_target[cols]
        auc_b = _one_vs_rest_auc(sims, same_b, population_b)
        naive[b] = auc_b - 0.5 if not np.isnan(auc_b) else np.nan

    batched = _batched_separation(sims, rows, cols, is_target_batch)

    assert np.allclose(naive, batched, equal_nan=True)


def test_batched_separation_matches_a_naive_loop_with_exact_ties() -> None:
    rng = np.random.default_rng(52)
    n = 14
    rows, cols = np.triu_indices(n, k=1)
    # Force many exact ties: only 4 distinct similarity values across all pairs.
    sims = rng.integers(0, 4, size=len(rows)).astype(float)
    is_target_batch = rng.integers(0, 2, size=(30, n)).astype(bool)

    naive = np.empty(30)
    for b in range(30):
        is_target = is_target_batch[b]
        same_b = is_target[rows] & is_target[cols]
        population_b = is_target[rows] | is_target[cols]
        auc_b = _one_vs_rest_auc(sims, same_b, population_b)
        naive[b] = auc_b - 0.5 if not np.isnan(auc_b) else np.nan

    batched = _batched_separation(sims, rows, cols, is_target_batch)

    assert np.allclose(naive, batched, equal_nan=True)


def test_permutation_p_is_small_for_a_genre_with_strong_real_separation() -> None:
    similarity_matrix, codes, genres = _strong_signal_fixture()

    result = joint_psalm_label_permutation_test(
        similarity_matrix, codes, genres, n_permutations=500, rng=np.random.default_rng(0)
    )

    assert all(p < 0.05 for p in result.p_perm)
    assert all(p < 0.05 for p in result.p_maxt)


def test_permutation_p_is_large_when_labels_are_unrelated_to_similarity() -> None:
    n = 12
    rng = np.random.default_rng(1)
    codes = np.array([0] * 4 + [1] * 4 + [2] * 4)
    rng.shuffle(codes)  # unrelated to the similarity structure below
    similarity_matrix = np.full((n, n), 0.5)
    np.fill_diagonal(similarity_matrix, 1.0)
    similarity_matrix += rng.normal(scale=0.01, size=(n, n))
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    np.fill_diagonal(similarity_matrix, 1.0)
    genres = ("A", "B", "C")

    result = joint_psalm_label_permutation_test(
        similarity_matrix, codes, genres, n_permutations=500, rng=np.random.default_rng(2)
    )

    # One genre can draw a ~5% false positive by chance, so the median should stay non-significant.
    assert float(np.median(result.p_perm)) > 0.05


def test_permutation_p_is_large_for_a_genre_separated_in_the_wrong_direction() -> None:
    """Same-genre psalms LESS similar than cross-genre ones: a one-sided test must not flag this."""
    n = 12
    rng = np.random.default_rng(9)
    codes = np.array([0] * 4 + [1] * 4 + [2] * 4)
    base = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                base[i, j] = 1.0
            elif codes[i] == codes[j] and codes[i] == 0:
                # Genre A specifically pushed BELOW chance similarity, the wrong direction.
                base[i, j] = 0.1 + rng.uniform(-0.02, 0.02)
            elif codes[i] == codes[j]:
                base[i, j] = 0.9 + rng.uniform(-0.02, 0.02)
            else:
                base[i, j] = 0.5 + rng.uniform(-0.02, 0.02)
    base = (base + base.T) / 2
    np.fill_diagonal(base, 1.0)
    genres = ("A", "B", "C")

    result = joint_psalm_label_permutation_test(
        base, codes, genres, n_permutations=500, rng=np.random.default_rng(10)
    )

    assert result.observed[0] < 0.5
    assert result.p_perm[0] > 0.5


def test_maxt_p_values_never_smaller_than_per_genre_p_perm() -> None:
    similarity_matrix, codes, genres = _strong_signal_fixture()

    result = joint_psalm_label_permutation_test(
        similarity_matrix, codes, genres, n_permutations=300, rng=np.random.default_rng(3)
    )

    for p_perm, p_maxt in zip(result.p_perm, result.p_maxt, strict=True):
        assert p_maxt >= p_perm


def test_p_values_are_never_exactly_zero() -> None:
    similarity_matrix, codes, genres = _strong_signal_fixture()

    result = joint_psalm_label_permutation_test(
        similarity_matrix, codes, genres, n_permutations=50, rng=np.random.default_rng(4)
    )

    assert all(p > 0.0 for p in result.p_perm)
    assert all(p > 0.0 for p in result.p_maxt)


def test_same_seed_reproduces_identical_results() -> None:
    similarity_matrix, codes, genres = _strong_signal_fixture()

    result_a = joint_psalm_label_permutation_test(
        similarity_matrix, codes, genres, n_permutations=200, rng=np.random.default_rng(7)
    )
    result_b = joint_psalm_label_permutation_test(
        similarity_matrix, codes, genres, n_permutations=200, rng=np.random.default_rng(7)
    )

    assert result_a.p_perm == result_b.p_perm
    assert result_a.p_maxt == result_b.p_maxt


def test_observed_auc_matches_scipy_mannwhitneyu_on_the_real_labels() -> None:
    similarity_matrix, codes, genres = _strong_signal_fixture()
    n = similarity_matrix.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    is_a = codes == 0
    same = is_a[rows] & is_a[cols]
    population = is_a[rows] | is_a[cols]
    sims = similarity_matrix[rows, cols]
    same_sims = sims[population][same[population]]
    diff_sims = sims[population][~same[population]]
    statistic, _ = mannwhitneyu(same_sims, diff_sims, alternative="greater")
    expected_auc = statistic / (len(same_sims) * len(diff_sims))

    result = joint_psalm_label_permutation_test(
        similarity_matrix, codes, genres, n_permutations=10, rng=np.random.default_rng(5)
    )

    assert result.observed[0] == expected_auc


def test_batched_separation_chunks_draws_without_changing_a_single_bit() -> None:
    """Draws are independent, so chunking them must be exact, not merely close."""
    from genre.permutation import _batched_separation
    from library.blocking import rows_per_block

    rng = np.random.default_rng(3)
    n_psalms = 40
    rows_pairs = n_psalms * (n_psalms - 1) // 2
    n_draws = rows_per_block(rows_pairs) * 3 + 7
    rows, cols = np.triu_indices(n_psalms, k=1)
    sims = rng.random(len(rows))
    is_target_batch = rng.random((n_draws, n_psalms)) < 0.4

    chunked = _batched_separation(sims, rows, cols, is_target_batch)
    one_shot = np.concatenate(
        [_batched_separation(sims, rows, cols, is_target_batch[i : i + 1]) for i in range(n_draws)]
    )

    assert np.array_equal(chunked, one_shot)


def test_batched_separation_peak_memory_does_not_grow_with_the_draw_count() -> None:
    """All draws at once allocated two float64 arrays of draws by pairs."""
    import tracemalloc

    from genre.permutation import _batched_separation

    rng = np.random.default_rng(4)
    n_psalms = 60
    rows, cols = np.triu_indices(n_psalms, k=1)
    sims = rng.random(len(rows))

    def peak_for(n_draws: int) -> int:
        batch = rng.random((n_draws, n_psalms)) < 0.4
        tracemalloc.start()
        _batched_separation(sims, rows, cols, batch)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    small, large = peak_for(200), peak_for(2000)

    assert large < small * 3, f"peak grew from {small} to {large} with ten times the draws"


def test_one_vs_rest_auc_is_nan_when_a_genre_has_no_pair_on_one_side() -> None:
    """A genre with no same-genre or no different-genre pair cannot be ranked, so it reports NaN."""
    from genre.permutation import _one_vs_rest_auc

    sims = np.array([0.1, 0.2, 0.3])
    all_same = np.array([True, True, True])
    population = np.array([True, True, True])

    assert np.isnan(_one_vs_rest_auc(sims, all_same, population))
