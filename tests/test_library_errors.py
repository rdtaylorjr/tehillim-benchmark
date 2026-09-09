import pytest

from library.errors import (
    BenchmarkDataError,
    DegenerateVectorError,
    InsufficientDataError,
)


@pytest.mark.parametrize("error", [InsufficientDataError, DegenerateVectorError])
def test_every_degenerate_data_error_is_catchable_as_one_type(error: type[Exception]) -> None:
    """Scripts skip a degenerate model by catching one base, never a blanket ValueError."""
    assert issubclass(error, BenchmarkDataError)


def test_the_base_stays_a_value_error_for_callers_that_predate_it() -> None:
    assert issubclass(BenchmarkDataError, ValueError)
