from __future__ import annotations

import numpy as np

from genre.permutation import one_vs_rest_masks
from trajectory.genre_breakdown import (
    _batched_one_vs_rest_gap,
    _dense_pair_matrices,
    _one_vs_rest_gap,
    joint_genre_breakdown_permutation_test,
)


def _batched_one_vs_rest_gap_naive(
    distances: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray, is_target_batch: np.ndarray
) -> np.ndarray:
    """Reference implementation of the batched gap: explicit (permutations x pairs) masking."""
    same_batch = is_target_batch[:, idx_a] & is_target_batch[:, idx_b]
    population_batch = is_target_batch[:, idx_a] | is_target_batch[:, idx_b]
    d = distances.astype(np.float64, copy=False)

    same_sum = same_batch.astype(np.float64) @ d
    same_count = same_batch.sum(axis=1).astype(np.float64)
    population_sum = population_batch.astype(np.float64) @ d
    population_count = population_batch.sum(axis=1).astype(np.float64)
    # same is a subset of population, so the different side is population minus same.
    diff_sum = population_sum - same_sum
    diff_count = population_count - same_count

    with np.errstate(invalid="ignore", divide="ignore"):
        gap = (diff_sum / diff_count) - (same_sum / same_count)
    invalid = (same_count == 0) | (diff_count == 0)
    return np.where(invalid, np.nan, gap)


def _hand_built_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    # 4 psalms: 0,1 genre A (code 0); 2,3 genre B (code 1). All C(4,2)=6 pairs.
    idx_a = np.array([0, 0, 0, 1, 1, 2])
    idx_b = np.array([1, 2, 3, 2, 3, 3])
    distances = np.array([1.0, 5.0, 6.0, 7.0, 8.0, 2.0])
    genre_codes = np.array([0, 0, 1, 1])
    genres = ("A", "B")
    return idx_a, idx_b, distances, genre_codes, genres


def test_one_vs_rest_gap_matches_hand_computed_values() -> None:
    idx_a, idx_b, distances, genre_codes, _genres = _hand_built_fixture()

    for g, expected in enumerate([5.5, 4.5]):
        same_mask, population_mask = one_vs_rest_masks(genre_codes, g)
        gap = _one_vs_rest_gap(distances, same_mask[idx_a, idx_b], population_mask[idx_a, idx_b])
        assert gap == expected


def test_one_vs_rest_gap_is_nan_when_a_side_is_empty() -> None:
    same = np.array([False, False])
    population = np.array([True, True])

    assert np.isnan(_one_vs_rest_gap(np.array([1.0, 2.0]), same, population))


def _dense_batch_inputs(
    n: int, idx_a: np.ndarray, idx_b: np.ndarray, distances: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    distance_matrix, mask_matrix = _dense_pair_matrices(n, idx_a, idx_b, distances)
    return (
        distance_matrix,
        mask_matrix,
        distance_matrix.sum(axis=0),
        mask_matrix.sum(axis=0),
        float(distances.sum()),
        float(len(distances)),
    )


def test_batched_one_vs_rest_gap_matches_a_naive_per_permutation_loop() -> None:
    """idx_a and idx_b must be distinct unordered pairs, which every real caller guarantees."""
    rng = np.random.default_rng(11)
    n = 10
    idx_a, idx_b = np.triu_indices(n, k=1)
    distances = rng.normal(size=len(idx_a))
    is_target_batch = rng.integers(0, 2, size=(25, n)).astype(bool)

    naive = np.empty(25)
    for b in range(25):
        is_target = is_target_batch[b]
        same_b = is_target[idx_a] & is_target[idx_b]
        population_b = is_target[idx_a] | is_target[idx_b]
        naive[b] = _one_vs_rest_gap(distances, same_b, population_b)

    dm, mm, dcs, mcs, tsum, tcount = _dense_batch_inputs(n, idx_a, idx_b, distances)
    batched = _batched_one_vs_rest_gap(is_target_batch, dm, mm, dcs, mcs, tsum, tcount)

    assert np.allclose(naive, batched, equal_nan=True)


def test_batched_one_vs_rest_gap_matches_the_naive_masked_reference_implementation() -> None:
    """Proves the dense quadratic-form reformation is exact, not an approximation."""
    rng = np.random.default_rng(21)
    n = 14
    idx_a, idx_b = np.triu_indices(n, k=1)
    distances = rng.normal(size=len(idx_a))
    is_target_batch = rng.integers(0, 2, size=(50, n)).astype(bool)

    reference = _batched_one_vs_rest_gap_naive(distances, idx_a, idx_b, is_target_batch)
    dm, mm, dcs, mcs, tsum, tcount = _dense_batch_inputs(n, idx_a, idx_b, distances)
    fast = _batched_one_vs_rest_gap(is_target_batch, dm, mm, dcs, mcs, tsum, tcount)

    assert np.allclose(reference, fast, equal_nan=True)


def test_batched_one_vs_rest_gap_matches_the_naive_reference_with_missing_pairs() -> None:
    """Some pairs excluded (NaN-filtered upstream), not the full C(n,2) set: still must match."""
    rng = np.random.default_rng(22)
    n = 12
    full_a, full_b = np.triu_indices(n, k=1)
    keep = rng.random(len(full_a)) < 0.7
    idx_a, idx_b = full_a[keep], full_b[keep]
    distances = rng.normal(size=len(idx_a))
    is_target_batch = rng.integers(0, 2, size=(40, n)).astype(bool)

    reference = _batched_one_vs_rest_gap_naive(distances, idx_a, idx_b, is_target_batch)
    dm, mm, dcs, mcs, tsum, tcount = _dense_batch_inputs(n, idx_a, idx_b, distances)
    fast = _batched_one_vs_rest_gap(is_target_batch, dm, mm, dcs, mcs, tsum, tcount)

    assert np.allclose(reference, fast, equal_nan=True)


def _clustered_distance_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]
]:
    rng = np.random.default_rng(3)
    n = 12
    codes = np.array([0] * 4 + [1] * 4 + [2] * 4)
    idx_a, idx_b = np.triu_indices(n, k=1)
    distances = np.empty(len(idx_a))
    for k, (i, j) in enumerate(zip(idx_a, idx_b, strict=True)):
        if codes[i] == codes[j]:
            distances[k] = 0.1 + rng.uniform(-0.02, 0.02)
        else:
            distances[k] = 0.9 + rng.uniform(-0.02, 0.02)
    return idx_a, idx_b, distances, codes, ("A", "B", "C")


def test_joint_permutation_p_is_small_for_strong_genre_clustering() -> None:
    idx_a, idx_b, distances, codes, genres = _clustered_distance_fixture()

    result = joint_genre_breakdown_permutation_test(
        distances, idx_a, idx_b, codes, genres, n_permutations=500, rng=np.random.default_rng(0)
    )

    assert all(p < 0.05 for p in result.p_perm)
    assert all(p < 0.05 for p in result.p_maxt)


def test_permutation_p_is_large_when_labels_are_unrelated_to_distance() -> None:
    rng = np.random.default_rng(1)
    n = 12
    codes = np.array([0] * 4 + [1] * 4 + [2] * 4)
    rng.shuffle(codes)
    idx_a, idx_b = np.triu_indices(n, k=1)
    distances = 0.5 + rng.normal(scale=0.01, size=len(idx_a))
    genres = ("A", "B", "C")

    result = joint_genre_breakdown_permutation_test(
        distances, idx_a, idx_b, codes, genres, n_permutations=500, rng=np.random.default_rng(2)
    )

    assert float(np.median(result.p_perm)) > 0.05


def test_p_values_never_zero_and_maxt_at_least_p_perm() -> None:
    idx_a, idx_b, distances, codes, genres = _clustered_distance_fixture()

    result = joint_genre_breakdown_permutation_test(
        distances, idx_a, idx_b, codes, genres, n_permutations=200, rng=np.random.default_rng(4)
    )

    assert all(p > 0.0 for p in result.p_perm)
    assert all(p > 0.0 for p in result.p_maxt)
    for p_perm, p_maxt in zip(result.p_perm, result.p_maxt, strict=True):
        assert p_maxt >= p_perm


def test_same_seed_reproduces_identical_results() -> None:
    idx_a, idx_b, distances, codes, genres = _clustered_distance_fixture()

    result_a = joint_genre_breakdown_permutation_test(
        distances, idx_a, idx_b, codes, genres, n_permutations=100, rng=np.random.default_rng(6)
    )
    result_b = joint_genre_breakdown_permutation_test(
        distances, idx_a, idx_b, codes, genres, n_permutations=100, rng=np.random.default_rng(6)
    )

    assert result_a.p_perm == result_b.p_perm
    assert result_a.p_maxt == result_b.p_maxt


def test_observed_gap_matches_hand_built_fixture_through_the_full_function() -> None:
    idx_a, idx_b, distances, genre_codes, genres = _hand_built_fixture()

    result = joint_genre_breakdown_permutation_test(
        distances,
        idx_a,
        idx_b,
        genre_codes,
        genres,
        n_permutations=10,
        rng=np.random.default_rng(5),
    )

    assert result.observed == (5.5, 4.5)
