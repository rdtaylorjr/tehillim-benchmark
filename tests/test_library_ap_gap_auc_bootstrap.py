import numpy as np
import pytest

from library.ap_gap_auc_bootstrap import (
    MIN_PER_SIDE,
    ApGapAucCI,
    bootstrap_ap_gap_and_auc,
    calibrated_gap,
    ci_row,
    has_enough_per_side,
    point_ap_gap_and_auc,
    resample_ap_gap_and_auc,
)
from library.calibration import BackgroundStats

_BACKGROUND = BackgroundStats(mean=0.3, std=0.2, n_vectors=10)


def _separable_split() -> tuple[np.ndarray, np.ndarray]:
    positive = np.array([0.9, 0.8, 0.7, 0.6])
    negative = np.array([0.5, 0.4, 0.3, 0.2])
    return positive, negative


def test_calibrated_gap_is_the_difference_of_two_standardized_group_means() -> None:
    positive = np.array([0.5, 0.7])
    negative = np.array([0.1, 0.3])

    gap = calibrated_gap(positive, negative, _BACKGROUND)

    assert gap == pytest.approx((0.6 - 0.3) / 0.2 - (0.2 - 0.3) / 0.2)


def test_point_and_resample_paths_agree_on_ap_and_auc() -> None:
    """The fast path exists only as a speed-up, so it must match sklearn/scipy exactly."""
    positive, negative = _separable_split()

    point = point_ap_gap_and_auc(positive, negative, _BACKGROUND)
    resample = resample_ap_gap_and_auc(positive, negative, _BACKGROUND)

    assert point == pytest.approx(resample)


def test_point_estimate_of_a_perfectly_separated_split_is_one() -> None:
    positive, negative = _separable_split()

    ap, _gap, auc = point_ap_gap_and_auc(positive, negative, _BACKGROUND)

    assert ap == pytest.approx(1.0)
    assert auc == pytest.approx(1.0)


def test_has_enough_per_side_requires_both_sides_to_reach_the_minimum() -> None:
    enough = np.ones(MIN_PER_SIDE)
    too_few = np.ones(MIN_PER_SIDE - 1)

    assert has_enough_per_side((enough, enough))
    assert not has_enough_per_side((too_few, enough))
    assert not has_enough_per_side((enough, too_few))


def test_bootstrap_returns_a_ci_bracketing_the_point_estimate() -> None:
    positive, negative = _separable_split()
    rng = np.random.default_rng(0)
    resamples = [
        (rng.choice(positive, size=len(positive)), rng.choice(negative, size=len(negative)))
        for _ in range(200)
    ]
    jackknife = [(np.delete(positive, i), negative) for i in range(len(positive))]

    result = bootstrap_ap_gap_and_auc((positive, negative), resamples, jackknife, _BACKGROUND)

    assert isinstance(result, ApGapAucCI)
    assert result.ap_ci_low <= result.point_ap <= result.ap_ci_high
    assert result.auc_ci_low <= result.point_auc <= result.auc_ci_high
    assert result.prevalence == pytest.approx(0.5)
    assert result.n_valid_resamples == 200
    assert result.n_valid_jackknife == len(positive)


def test_bootstrap_skips_resamples_that_are_too_small_and_counts_the_survivors() -> None:
    positive, negative = _separable_split()
    resamples = [(positive, negative), (positive[:1], negative), (positive, negative)]

    result = bootstrap_ap_gap_and_auc((positive, negative), resamples, [], _BACKGROUND)

    assert result.n_valid_resamples == 2


def test_bootstrap_reports_nan_intervals_when_no_resample_is_usable() -> None:
    positive, negative = _separable_split()

    result = bootstrap_ap_gap_and_auc((positive, negative), [], [], _BACKGROUND)

    assert result.n_valid_resamples == 0
    assert result.point_ap == pytest.approx(1.0)
    assert np.isnan(result.ap_ci_low)
    assert np.isnan(result.ap_ci_high)
    assert np.isnan(result.ap_ci_low_pct)
    assert np.isnan(result.gap_ci_low)
    assert np.isnan(result.auc_ci_high)


def test_bootstrap_records_a_none_jackknife_entry_as_nan() -> None:
    positive, negative = _separable_split()
    jackknife = [(positive, negative), None, (positive, negative)]

    result = bootstrap_ap_gap_and_auc(
        (positive, negative), [(positive, negative)] * 50, jackknife, _BACKGROUND
    )

    assert result.n_valid_jackknife == 2


@pytest.mark.parametrize("empty_side", [0, 1])
def test_bootstrap_rejects_an_observed_split_with_too_few_values_on_one_side(
    empty_side: int,
) -> None:
    """A one-sided split makes AP and AUC undefined, which must fail loudly, not return NaN."""
    split = [np.array([0.9, 0.8]), np.array([0.2, 0.1])]
    split[empty_side] = np.array([0.5])

    with pytest.raises(ValueError, match="at least 2 values on each side"):
        bootstrap_ap_gap_and_auc((split[0], split[1]), [], [], _BACKGROUND)


def test_bootstrap_records_a_jackknife_split_that_is_too_small_as_nan() -> None:
    """Leaving out a cluster can drop a side below the minimum, which is NaN, not a crash."""
    positive, negative = _separable_split()
    jackknife = [(positive, negative), (positive[:1], negative)]

    result = bootstrap_ap_gap_and_auc(
        (positive, negative), [(positive, negative)] * 50, jackknife, _BACKGROUND
    )

    assert result.n_valid_jackknife == 1


def _ci(**overrides: float) -> ApGapAucCI:
    fields = dict.fromkeys(ApGapAucCI.__dataclass_fields__, 0.0)
    return ApGapAucCI(**{**fields, **overrides})  # type: ignore[arg-type]


def test_ci_row_starts_with_the_model_and_carries_every_field() -> None:
    row = ci_row("m", _ci(point_ap=0.8))

    assert next(iter(row)) == "model"
    assert row["point_ap"] == 0.8
    assert set(row) == {"model", *ApGapAucCI.__dataclass_fields__}


def test_ci_row_places_scope_directly_after_the_model_when_given() -> None:
    """Column order is the CSV header, so it must stay fixed across runs."""
    row = ci_row("m", _ci(), scope="Synonymous")

    assert list(row)[:3] == ["model", "scope", "prevalence"]


def test_ci_row_omits_scope_entirely_when_not_given() -> None:
    assert "scope" not in ci_row("m", _ci())


# The published CSV header, pinned so reordering the dataclass cannot silently move a column.
_EXPECTED_CI_COLUMNS = [
    "model",
    "prevalence",
    "point_ap",
    "ap_ci_low",
    "ap_ci_high",
    "ap_ci_low_pct",
    "ap_ci_high_pct",
    "point_gap",
    "gap_ci_low",
    "gap_ci_high",
    "gap_ci_low_pct",
    "gap_ci_high_pct",
    "point_auc",
    "auc_ci_low",
    "auc_ci_high",
    "auc_ci_low_pct",
    "auc_ci_high_pct",
    "n_valid_resamples",
    "n_valid_jackknife",
]


def test_ci_row_column_order_is_pinned_to_the_published_header() -> None:
    """The order is a published CSV header, so a dataclass field reorder must fail here first."""
    assert list(ci_row("m", _ci())) == _EXPECTED_CI_COLUMNS


def test_ci_row_with_a_scope_inserts_it_without_disturbing_the_rest() -> None:
    expected = [_EXPECTED_CI_COLUMNS[0], "scope", *_EXPECTED_CI_COLUMNS[1:]]

    assert list(ci_row("m", _ci(), scope="overall")) == expected
