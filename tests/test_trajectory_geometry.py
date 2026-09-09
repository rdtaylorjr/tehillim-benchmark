import numpy as np
import pytest

from trajectory.geometry import adjacent_similarity, step_magnitude, turning_angle


def test_adjacent_similarity_has_length_n_minus_one() -> None:
    sequence = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    result = adjacent_similarity(sequence)

    assert len(result) == 2


def test_adjacent_similarity_matches_cosine_of_each_consecutive_pair() -> None:
    sequence = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    result = adjacent_similarity(sequence)

    assert result == pytest.approx([0.0, 0.0])


def test_step_magnitude_has_length_n_minus_one() -> None:
    sequence = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])

    result = step_magnitude(sequence)

    assert len(result) == 2
    assert result == pytest.approx([5.0, 0.0])


def test_turning_angle_has_length_n_minus_two() -> None:
    sequence = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    result = turning_angle(sequence)

    assert len(result) == 1


def test_turning_angle_is_zero_for_a_straight_line() -> None:
    sequence = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    result = turning_angle(sequence)

    assert result == pytest.approx([0.0])


def test_turning_angle_is_a_right_angle_for_an_l_shaped_path() -> None:
    sequence = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    result = turning_angle(sequence)

    assert result == pytest.approx([np.pi / 2, np.pi / 2])


def test_turning_angle_is_pi_for_a_direct_reversal() -> None:
    sequence = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

    result = turning_angle(sequence)

    assert result == pytest.approx([np.pi])


def test_turning_angle_is_nan_where_a_step_has_zero_length() -> None:
    """Two consecutive identical half-verses (e.g. a repeated refrain) give a zero displacement."""
    sequence = np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]])

    result = turning_angle(sequence)

    assert np.isnan(result[0])
    assert result[1] == pytest.approx(np.pi / 2)
