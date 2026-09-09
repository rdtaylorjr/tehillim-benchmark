"""Collects scored rows columnwise, so peak memory tracks the data rather than Python dicts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["FrameAccumulator"]


class FrameAccumulator:
    """Accumulates result rows as frames, concatenating once when the whole run is scored."""

    def __init__(self, cached: pd.DataFrame | None = None) -> None:
        """Starts from a prior run's frame, or empty when there is none."""
        self.parts: list[pd.DataFrame] = []
        self._n_rows = 0
        if cached is not None and not cached.empty:
            self.parts.append(cached)
            self._n_rows += len(cached)

    def extend(self, rows: Sequence[dict[str, Any]]) -> None:
        """Adds one model's rows, converting them to columns before the next model is scored."""
        if not rows:
            return
        self.parts.append(pd.DataFrame(list(rows)))
        self._n_rows += len(rows)

    def frame(self) -> pd.DataFrame:
        """Every accumulated row as one frame."""
        if not self.parts:
            return pd.DataFrame()
        if len(self.parts) == 1:
            return self.parts[0]
        return pd.concat(self.parts, ignore_index=True)

    def __len__(self) -> int:
        """How many rows have been accumulated, without building the frame."""
        return self._n_rows
