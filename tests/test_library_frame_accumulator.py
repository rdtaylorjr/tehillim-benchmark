from __future__ import annotations

import pandas as pd
import pytest

from library.frame_accumulator import FrameAccumulator


class TestFrameAccumulator:
    def test_an_empty_accumulator_yields_an_empty_frame(self) -> None:
        assert FrameAccumulator().frame().empty

    def test_collects_rows_into_one_frame(self) -> None:
        accumulator = FrameAccumulator()
        accumulator.extend([{"model": "a", "x": 1}, {"model": "a", "x": 2}])
        accumulator.extend([{"model": "b", "x": 3}])

        frame = accumulator.frame()

        assert list(frame["model"]) == ["a", "a", "b"]
        assert list(frame["x"]) == [1, 2, 3]

    def test_seeds_from_a_cached_frame(self) -> None:
        cached = pd.DataFrame({"model": ["old"], "x": [0]})
        accumulator = FrameAccumulator(cached)
        accumulator.extend([{"model": "new", "x": 1}])

        assert list(accumulator.frame()["model"]) == ["old", "new"]

    def test_an_empty_batch_adds_nothing(self) -> None:
        accumulator = FrameAccumulator()
        accumulator.extend([{"model": "a", "x": 1}])
        accumulator.extend([])

        assert len(accumulator.frame()) == 1

    def test_counts_rows_without_building_the_frame(self) -> None:
        accumulator = FrameAccumulator()
        accumulator.extend([{"x": 1}, {"x": 2}])
        accumulator.extend([{"x": 3}])

        assert len(accumulator) == 3

    def test_holds_no_python_row_dicts_after_a_batch(self) -> None:
        """Rows convert to a frame on arrival, so peak memory tracks columns and not dicts."""
        accumulator = FrameAccumulator()
        accumulator.extend([{"model": "a", "x": 1}])

        assert all(isinstance(part, pd.DataFrame) for part in accumulator.parts)

    def test_floats_round_trip_exactly(self) -> None:
        value = 4.1377175583932606e-42
        accumulator = FrameAccumulator()
        accumulator.extend([{"p": value}])

        assert accumulator.frame()["p"].iloc[0] == value

    def test_a_cached_frame_alone_is_returned_unchanged(self) -> None:
        cached = pd.DataFrame({"model": ["old"], "x": [7]})

        frame = FrameAccumulator(cached).frame()

        assert list(frame["x"]) == [7]

    def test_column_order_follows_the_first_batch(self) -> None:
        accumulator = FrameAccumulator()
        accumulator.extend([{"b": 1, "a": 2}])

        assert list(accumulator.frame().columns) == ["b", "a"]

    def test_the_index_is_reset_so_concatenated_parts_do_not_repeat_labels(self) -> None:
        accumulator = FrameAccumulator()
        accumulator.extend([{"x": 1}])
        accumulator.extend([{"x": 2}])

        assert list(accumulator.frame().index) == [0, 1]


@pytest.mark.parametrize("batches", [1, 5, 20])
def test_row_count_matches_the_rows_added(batches: int) -> None:
    accumulator = FrameAccumulator()
    for batch in range(batches):
        accumulator.extend([{"x": batch}, {"x": batch}])

    assert len(accumulator.frame()) == batches * 2
