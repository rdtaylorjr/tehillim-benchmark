"""One rule for a batch scorer meeting a model whose data cannot support the statistic."""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, overload

from library.embeddings import dataset_identifier
from library.errors import BenchmarkDataError


class _SkippingUnscorable[ItemT, ResultT]:
    """Callable rather than a closure, because a process pool carries the scorer by pickle."""

    def __init__(self, score: Callable[[ItemT], ResultT], label: Callable[[ItemT], str]) -> None:
        self._score = score
        self._label = label

    def __call__(self, item: ItemT) -> ResultT | None:
        try:
            return self._score(item)
        except BenchmarkDataError as error:
            print(f"skipping {self._label(item)}: {error}", file=sys.stderr)
            return None


def _identify(path: Path) -> str:
    """Names a model by its embeddings file, which is what most batches map over."""
    #: Reporting a skip must not raise over the error it reports, so an unnamed file uses its path.
    try:
        return dataset_identifier(path)
    except BenchmarkDataError:
        return str(path)


@overload
def skipping_unscorable[ResultT](
    score: Callable[[Path], ResultT],
) -> Callable[[Path], ResultT | None]: ...


@overload
def skipping_unscorable[ItemT, ResultT](
    score: Callable[[ItemT], ResultT], label: Callable[[ItemT], str]
) -> Callable[[ItemT], ResultT | None]: ...


def skipping_unscorable[ResultT](
    score: Callable[[Any], ResultT], label: Callable[[Any], str] | None = None
) -> Callable[[Any], ResultT | None]:
    """Wraps a per-model scorer so one degenerate model is reported and skipped, never fatal."""
    return _SkippingUnscorable(score, label if label is not None else _identify)
