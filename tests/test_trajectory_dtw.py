import itertools

import numpy as np
import pytest

from trajectory.dtw import dtw_accumulated_cost, dtw_distance, dtw_warping_path


def test_dtw_accumulated_cost_matches_hand_computed_repeated_element_example() -> None:
    """A=[1,2,3], B=[1,2,2,3]: B repeats "2", so a perfect zero-cost alignment exists."""
    cost = np.abs(np.array([1.0, 2.0, 3.0])[:, None] - np.array([1.0, 2.0, 2.0, 3.0])[None, :])

    accumulated = dtw_accumulated_cost(cost)

    expected = np.array(
        [
            [0.0, 1.0, 2.0, 4.0],
            [1.0, 0.0, 0.0, 1.0],
            [3.0, 1.0, 1.0, 0.0],
        ]
    )
    assert accumulated == pytest.approx(expected)


def test_dtw_accumulated_cost_matches_hand_computed_constant_offset_example() -> None:
    """A=[0,0], B=[1,1]: every cell costs 1, no free alignment exists."""
    cost = np.abs(np.array([0.0, 0.0])[:, None] - np.array([1.0, 1.0])[None, :])

    accumulated = dtw_accumulated_cost(cost)

    expected = np.array([[1.0, 2.0], [2.0, 2.0]])
    assert accumulated == pytest.approx(expected)


def test_dtw_warping_path_follows_the_repeated_element_alignment() -> None:
    cost = np.abs(np.array([1.0, 2.0, 3.0])[:, None] - np.array([1.0, 2.0, 2.0, 3.0])[None, :])
    accumulated = dtw_accumulated_cost(cost)

    path = dtw_warping_path(accumulated)

    assert path == [(0, 0), (1, 1), (1, 2), (2, 3)]


def test_dtw_warping_path_starts_at_origin_and_ends_at_the_far_corner() -> None:
    cost = np.abs(np.array([0.0, 0.0])[:, None] - np.array([1.0, 1.0])[None, :])
    accumulated = dtw_accumulated_cost(cost)

    path = dtw_warping_path(accumulated)

    assert path[0] == (0, 0)
    assert path[-1] == (1, 1)


def test_dtw_warping_path_is_monotone_and_continuous() -> None:
    """Every step advances i, j, or both by exactly one: no skips, no going backward."""
    rng = np.random.default_rng(0)
    cost = rng.random((7, 11))
    accumulated = dtw_accumulated_cost(cost)

    path = dtw_warping_path(accumulated)

    for (i0, j0), (i1, j1) in itertools.pairwise(path):
        assert i1 - i0 in (0, 1)
        assert j1 - j0 in (0, 1)
        assert (i1, j1) != (i0, j0)


def test_dtw_distance_of_a_perfect_alignment_is_zero() -> None:
    cost = np.abs(np.array([1.0, 2.0, 3.0])[:, None] - np.array([1.0, 2.0, 2.0, 3.0])[None, :])

    distance, path = dtw_distance(cost)

    assert distance == pytest.approx(0.0)
    assert path == [(0, 0), (1, 1), (1, 2), (2, 3)]


def test_dtw_distance_normalizes_by_path_length() -> None:
    """A=[0,0], B=[1,1]: total accumulated cost 2 over a 2-step path, normalized distance 1.0."""
    cost = np.abs(np.array([0.0, 0.0])[:, None] - np.array([1.0, 1.0])[None, :])

    distance, path = dtw_distance(cost)

    assert len(path) == 2
    assert distance == pytest.approx(1.0)


def test_dtw_distance_of_identical_equal_length_sequences_is_zero() -> None:
    cost = np.abs(np.array([1.0, 5.0, 2.0])[:, None] - np.array([1.0, 5.0, 2.0])[None, :])

    distance, path = dtw_distance(cost)

    assert distance == pytest.approx(0.0)
    assert path == [(0, 0), (1, 1), (2, 2)]


def test_dtw_accumulated_cost_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one"):
        dtw_accumulated_cost(np.zeros((0, 3)))
