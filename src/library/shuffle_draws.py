"""Selects the draws a shuffle-null control scores, refusing a set that cannot support the null."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def select_shuffle_draws(shuffled_embeddings_dir: Path, n_shuffles: int) -> list[Path]:
    """The first n_shuffles draws under the directory, in seed order."""
    #: glob does not descend a symlinked directory, so a staged tree can look empty and score NaN.
    draws = sorted(shuffled_embeddings_dir.glob("**/*.parquet"))[:n_shuffles]
    if not draws:
        raise ValueError(
            f"no shuffle draws under {shuffled_embeddings_dir}: "
            "a symlinked draw directory is not descended, so link the parquet files themselves"
        )
    if len(draws) < n_shuffles:
        raise ValueError(
            f"asked for {n_shuffles} shuffle draws, found {len(draws)} "
            f"under {shuffled_embeddings_dir}"
        )
    return draws
