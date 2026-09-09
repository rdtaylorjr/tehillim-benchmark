"""Errors for data a statistic cannot be computed on, raised instead of returning a silent NaN."""


class BenchmarkDataError(ValueError):
    """A model's data is degenerate for the requested statistic, so the model is skipped."""


class InsufficientDataError(BenchmarkDataError):
    """A sample is too small for the requested statistic to be defined."""


class DegenerateVectorError(BenchmarkDataError):
    """A vector or distribution has no direction or no spread, leaving the statistic undefined."""
