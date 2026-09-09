import numpy as np

from library.blocking import BLOCK_BYTES, row_blocks, rows_per_block


def test_rows_per_block_shrinks_as_the_row_widens() -> None:
    """Peak memory has to track the array's width, which a fixed row count cannot do."""
    assert rows_per_block(256) > rows_per_block(3072)
    assert rows_per_block(3072) * 3072 * 8 <= BLOCK_BYTES


def test_rows_per_block_never_returns_zero_for_a_row_wider_than_the_budget() -> None:
    assert rows_per_block(BLOCK_BYTES) == 1


def test_rows_per_block_honours_a_narrower_item_size() -> None:
    """A boolean source widens to float64, so callers size the block by the widened item."""
    assert rows_per_block(100, itemsize=1) > rows_per_block(100, itemsize=8)


def test_row_blocks_cover_every_row_exactly_once_in_order() -> None:
    covered = [i for block in row_blocks(1000, 64) for i in range(block.start, block.stop)]

    assert covered == list(range(1000))


def test_row_blocks_yield_nothing_for_an_empty_array() -> None:
    assert list(row_blocks(0, 64)) == []


def test_row_blocks_keep_each_block_inside_the_budget() -> None:
    width = 3072
    for block in row_blocks(10_000, width):
        assert (block.stop - block.start) * width * 8 <= BLOCK_BYTES


def test_row_blocks_reconstruct_an_array_unchanged() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((5000, 128)).astype(np.float32)

    rebuilt = np.concatenate([data[block] for block in row_blocks(len(data), data.shape[1])])

    assert np.array_equal(rebuilt, data)
