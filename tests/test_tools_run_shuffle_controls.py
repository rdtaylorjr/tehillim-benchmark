"""The sweep entry point: what it selects, what it skips, and what it reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from library.shuffle_control_sweep import Control, Job
from tools.run_shuffle_controls import main, parse_arguments


class TestParseArguments:
    def test_the_roots_and_the_run_size_are_read_from_the_command_line(self) -> None:
        args = parse_arguments(
            [
                "--data-root",
                "/data",
                "--embeddings-root",
                "/emb",
                "--config-root",
                "/emb/config",
                "--genre-csv",
                "/g.csv",
                "--n-shuffles",
                "50",
                "--workers",
                "3",
            ]
        )

        assert args.data_root == Path("/data")
        assert args.n_shuffles == 50
        assert args.workers == 3

    def test_a_family_can_be_named_to_score_one_instead_of_every(self) -> None:
        args = parse_arguments(
            [
                "--data-root",
                "/data",
                "--embeddings-root",
                "/emb",
                "--config-root",
                "/emb/config",
                "--genre-csv",
                "/g.csv",
                "--family",
                "morphological/sp/1_2gram",
            ]
        )

        assert args.family == ["morphological/sp/1_2gram"]


class TestMain:
    def _argv(self, tmp_path: Path, *extra: str) -> list[str]:
        return [
            "--data-root",
            str(tmp_path / "data"),
            "--embeddings-root",
            str(tmp_path / "emb"),
            "--config-root",
            str(tmp_path / "emb" / "config"),
            "--genre-csv",
            str(tmp_path / "g.csv"),
            "--n-shuffles",
            "5",
            *extra,
        ]

    def test_it_runs_only_the_families_named(self, tmp_path: Path) -> None:
        ran: list[Job] = []

        main(
            self._argv(tmp_path, "--family", "morphological/sp/1_2gram"),
            controls={"parallelism": Control(lambda argv: None)},
            runner=ran.append,
        )

        assert [job.key for job in ran] == ["morphological/sp/1_2gram"]

    def test_it_writes_nothing_and_runs_nothing_in_a_dry_run(self, tmp_path: Path) -> None:
        """A dry run is how an operator sees the size of a sweep before paying for it."""
        ran: list[Job] = []

        main(
            self._argv(tmp_path, "--dry-run"),
            controls={"parallelism": Control(lambda argv: None)},
            runner=ran.append,
        )

        assert ran == []
        assert not (tmp_path / "data").exists()

    def test_a_failing_family_is_reported_and_the_exit_is_non_zero(self, tmp_path: Path) -> None:
        def _runner(job: Job) -> None:
            raise RuntimeError("no support table")

        with pytest.raises(SystemExit) as exit_info:
            main(
                self._argv(tmp_path, "--family", "morphological/sp/1_2gram"),
                controls={"parallelism": Control(lambda argv: None)},
                runner=_runner,
            )

        assert exit_info.value.code == 1

    def test_an_unknown_family_is_refused_before_any_work_starts(self, tmp_path: Path) -> None:
        ran: list[Job] = []

        with pytest.raises(SystemExit):
            main(
                self._argv(tmp_path, "--family", "not/a/family"),
                controls={"parallelism": Control(lambda argv: None)},
                runner=ran.append,
            )

        assert ran == []
