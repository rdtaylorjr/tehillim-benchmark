"""Widening a large array in pieces, so peak memory tracks its width and not its length."""

from collections.abc import Iterator

#: 8 MB per widened block, wide enough to keep numpy efficient and narrow enough to bound a pool.
BLOCK_BYTES = 8 * 1024 * 1024


def rows_per_block(width: int, itemsize: int = 8) -> int:
    """How many rows of this width fit in one block once widened to itemsize bytes per value."""
    return max(1, BLOCK_BYTES // max(1, width * itemsize))


def row_blocks(n_rows: int, width: int, itemsize: int = 8) -> Iterator[slice]:
    """Row slices covering n_rows in order, each small enough to widen inside the budget."""
    step = rows_per_block(width, itemsize)
    for start in range(0, n_rows, step):
        yield slice(start, min(start + step, n_rows))
