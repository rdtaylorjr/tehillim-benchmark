import numpy as np
import pytest

from trajectory.distance import (
    content_distance,
    dtw_curve_distance,
    structural_distance,
    structural_distance_dtw,
)
from trajectory.self_similarity import self_similarity_matrix


def test_content_distance_is_zero_for_identical_centroids() -> None:
    centroid = np.array([1.0, 2.0, 3.0])

    assert content_distance(centroid, centroid) == pytest.approx(0.0)


def test_content_distance_is_one_for_orthogonal_centroids() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])

    assert content_distance(a, b) == pytest.approx(1.0)


def test_content_distance_is_two_for_opposite_centroids() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])

    assert content_distance(a, b) == pytest.approx(2.0)


def test_structural_distance_is_zero_for_identical_profiles() -> None:
    profile = np.array([[1.0, 0.5], [0.5, 1.0]])

    assert structural_distance(profile, profile) == pytest.approx(0.0)


def test_structural_distance_of_a_constant_offset_equals_the_offset() -> None:
    a = np.full((4, 4), 0.5)
    b = np.full((4, 4), 0.8)

    assert structural_distance(a, b) == pytest.approx(0.3)


def test_structural_distance_is_symmetric() -> None:
    a = np.array([[1.0, 0.2], [0.2, 1.0]])
    b = np.array([[1.0, 0.9], [0.9, 1.0]])

    assert structural_distance(a, b) == pytest.approx(structural_distance(b, a))


def test_dtw_curve_distance_is_zero_when_one_curve_is_a_stretched_repeat_of_the_other() -> None:
    """B repeats one element of A: a perfect zero-cost DTW alignment exists."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 2.0, 3.0])

    assert dtw_curve_distance(a, b) == pytest.approx(0.0)


def test_dtw_curve_distance_of_a_constant_offset_equals_the_offset() -> None:
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])

    assert dtw_curve_distance(a, b) == pytest.approx(1.0)


def test_dtw_curve_distance_is_symmetric() -> None:
    a = np.array([0.1, 0.5, 0.2])
    b = np.array([0.3, 0.1, 0.6, 0.4])

    assert dtw_curve_distance(a, b) == pytest.approx(dtw_curve_distance(b, a))


def test_structural_distance_dtw_is_zero_for_a_stretched_repeat_of_the_same_pattern() -> None:
    """B holds A's middle half-verses an extra step: DTW should align them with no residual."""
    a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])

    distance = structural_distance_dtw(a, b, self_similarity_matrix(a), self_similarity_matrix(b))

    assert distance == pytest.approx(0.0)


def test_structural_distance_dtw_is_zero_for_identical_equal_length_sequences() -> None:
    a = np.array([[1.0, 0.0], [0.3, 0.9], [0.0, 1.0]])
    self_sim = self_similarity_matrix(a)

    assert structural_distance_dtw(a, a, self_sim, self_sim) == pytest.approx(0.0)


def test_structural_distance_dtw_is_nonzero_for_genuinely_different_shapes() -> None:
    """One psalm holds steady (A,A,A), the other alternates (A,B,A): different self-similarity."""
    steady = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    alternating = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    distance = structural_distance_dtw(
        steady, alternating, self_similarity_matrix(steady), self_similarity_matrix(alternating)
    )

    assert distance > 0.1
