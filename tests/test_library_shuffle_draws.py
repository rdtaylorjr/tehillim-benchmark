"""The draw selection every shuffle-null control shares, and what it refuses."""

from __future__ import annotations

from pathlib import Path

import pytest

from library.shuffle_draws import select_shuffle_draws


def _draws(root: Path, count: int) -> Path:
    for seed in range(1, count + 1):
        directory = root / f"construction=c_shuffle{seed:04d}"
        directory.mkdir(parents=True)
        (directory / "part-0.parquet").write_bytes(b"")
    return root


class TestSelectShuffleDraws:
    def test_returns_the_requested_number_of_draws_in_seed_order(self, tmp_path: Path) -> None:
        draws = select_shuffle_draws(_draws(tmp_path, 5), 3)

        assert [p.parent.name for p in draws] == [
            "construction=c_shuffle0001",
            "construction=c_shuffle0002",
            "construction=c_shuffle0003",
        ]

    def test_refuses_a_directory_holding_no_draws(self, tmp_path: Path) -> None:
        """Scoring zero draws produced a NaN row and exited 0, which looked like a finished run."""
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ValueError, match="no shuffle draws"):
            select_shuffle_draws(empty, 1000)

    def test_refuses_fewer_draws_than_asked_for(self, tmp_path: Path) -> None:
        """A short draw set weakens the null's resolution, so it is refused not truncated."""
        with pytest.raises(ValueError, match="asked for 1000 shuffle draws, found 4"):
            select_shuffle_draws(_draws(tmp_path, 4), 1000)

    def test_a_symlinked_draw_directory_is_reported_rather_than_silently_skipped(
        self, tmp_path: Path
    ) -> None:
        """Path.glob does not descend a symlinked directory, which once emptied a whole run."""
        real = _draws(tmp_path / "real", 3)
        staged = tmp_path / "staged"
        staged.mkdir()
        for directory in sorted(real.iterdir()):
            (staged / directory.name).symlink_to(directory)

        with pytest.raises(ValueError, match="no shuffle draws"):
            select_shuffle_draws(staged, 3)

    def test_accepts_symlinked_draw_files_inside_real_directories(self, tmp_path: Path) -> None:
        real = _draws(tmp_path / "real", 3)
        staged = tmp_path / "staged"
        for directory in sorted(real.iterdir()):
            target = staged / directory.name
            target.mkdir(parents=True)
            (target / "part-0.parquet").symlink_to(directory / "part-0.parquet")

        assert len(select_shuffle_draws(staged, 3)) == 3
