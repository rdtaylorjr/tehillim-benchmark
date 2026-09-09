import numpy as np
import pytest

from library.permutation_test import maxt_p_values, permuted_label_batches


def test_permuted_labels_keep_every_label_exactly_once_per_draw() -> None:
    """A permutation test relabels, so each draw must be a rearrangement and not a resample."""
    codes = np.array([0, 0, 1, 1, 2])

    batches = permuted_label_batches(codes, 50, np.random.default_rng(0))

    assert batches.shape == (50, len(codes))
    for draw in batches:
        assert sorted(draw.tolist()) == sorted(codes.tolist())


def test_permuted_labels_are_reproducible_from_the_generator() -> None:
    codes = np.array([0, 1, 2, 3])

    first = permuted_label_batches(codes, 10, np.random.default_rng(7))
    again = permuted_label_batches(codes, 10, np.random.default_rng(7))

    assert np.array_equal(first, again)


def test_p_values_follow_the_count_plus_one_over_n_plus_one_convention() -> None:
    """The floor 1/(n+1) is what bounds the smallest reachable p, so the convention must hold."""
    observed = np.array([10.0])
    null = np.zeros((99, 1))

    result = maxt_p_values(observed, null)

    assert result.p_per_group[0] == pytest.approx(1 / 100)


def test_an_observed_statistic_no_better_than_its_null_gets_p_of_one() -> None:
    observed = np.array([0.0])
    null = np.zeros((9, 1))

    assert maxt_p_values(observed, null).p_per_group[0] == pytest.approx(1.0)


def test_maxt_p_is_never_smaller_than_the_per_group_p() -> None:
    """MaxT controls family-wise error, so it can only be more conservative."""
    rng = np.random.default_rng(1)
    observed = np.array([1.0, 2.0, 3.0])
    null = rng.standard_normal((500, 3))

    result = maxt_p_values(observed, null)

    assert np.all(result.p_maxt >= result.p_per_group - 1e-12)


def test_nan_draws_are_excluded_from_the_denominator_not_counted_as_failures() -> None:
    """A group whose draw could not be scored must not silently inflate the p-value."""
    observed = np.array([1.0])
    null = np.array([[np.nan], [0.0], [0.0], [2.0]])

    result = maxt_p_values(observed, null)

    assert result.p_per_group[0] == pytest.approx(2 / 4)


def test_a_group_with_no_valid_draw_reports_nan_rather_than_a_fabricated_p() -> None:
    observed = np.array([1.0])
    null = np.full((5, 1), np.nan)

    assert np.isnan(maxt_p_values(observed, null).p_per_group[0])
