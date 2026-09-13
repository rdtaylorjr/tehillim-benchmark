"""The sweep resolves every family's control output and runs only what is missing."""

from __future__ import annotations

from pathlib import Path

import pytest

from library.shuffle_control_sweep import (
    CONTROLS,
    Control,
    Job,
    control_filename,
    pending_jobs,
    run_job,
    shared_api_factory,
    sweep,
)


class TestControlFilename:
    """The name a control writes under, which must reproduce what the tree already holds."""

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("lexical/homograph/icf_position4", "homograph_icf_position4"),
            ("morphological/sp/1_2gram", "sp_1_2gram"),
            ("syntactic/phrase/typ/1_2gram", "typ_1_2gram"),
            ("syntactic/clause/tab/transition_psalm", "clause_tab_transition_psalm"),
        ],
    )
    def test_the_domain_and_the_phrase_level_are_dropped(self, key: str, expected: str) -> None:
        assert control_filename(key) == expected

    def test_every_registered_family_maps_to_a_distinct_name(self) -> None:
        """Two families sharing a name would have one silently overwrite the other."""
        from families.shuffle import FAMILIES

        names = [control_filename(key) for key in FAMILIES]

        assert len(set(names)) == len(names)


class TestPendingJobs:
    def test_a_family_with_no_output_is_pending(self, tmp_path: Path) -> None:
        jobs = pending_jobs(
            {"genre": CONTROLS["genre"]},
            ["morphological/sp/1_2gram"],
            data_root=tmp_path,
            embeddings_root=tmp_path,
            parameters={"genre_csv": tmp_path / "genre.csv"},
        )

        assert [job.key for job in jobs] == ["morphological/sp/1_2gram"]

    def test_a_family_whose_output_exists_is_skipped(self, tmp_path: Path) -> None:
        """A sweep is resumable, so an interrupted run does not rescore what it finished."""
        jobs = pending_jobs(
            {"genre": CONTROLS["genre"]},
            ["morphological/sp/1_2gram"],
            data_root=tmp_path,
            embeddings_root=tmp_path,
            parameters={"genre_csv": tmp_path / "genre.csv"},
        )
        jobs[0].output.parent.mkdir(parents=True, exist_ok=True)
        jobs[0].output.write_text("already scored\n")

        remaining = pending_jobs(
            {"genre": CONTROLS["genre"]},
            ["morphological/sp/1_2gram"],
            data_root=tmp_path,
            embeddings_root=tmp_path,
            parameters={"genre_csv": tmp_path / "genre.csv"},
        )

        assert remaining == []

    def test_the_output_lands_under_the_partition_the_committed_tree_uses(
        self, tmp_path: Path
    ) -> None:
        jobs = pending_jobs(
            {"parallelism": CONTROLS["parallelism"]},
            ["syntactic/clause/tab/transition_psalm"],
            data_root=tmp_path,
            embeddings_root=tmp_path,
            parameters={"genre_csv": tmp_path / "genre.csv"},
        )

        assert jobs[0].output == tmp_path / (
            "analysis=benchmark/benchmark=parallelism/domain=syntactic/"
            "stage=shuffle_control/clause_tab_transition_psalm.csv"
        )

    def test_every_benchmark_gets_a_job_for_every_family(self, tmp_path: Path) -> None:
        keys = ["morphological/sp/1_2gram", "syntactic/phrase/typ/1_2gram"]

        jobs = pending_jobs(
            CONTROLS,
            keys,
            data_root=tmp_path,
            embeddings_root=tmp_path,
            parameters={"genre_csv": tmp_path / "genre.csv"},
        )

        assert len(jobs) == len(CONTROLS) * len(keys)


class TestSweep:
    def test_each_job_is_run_once_and_its_parent_directory_created(self, tmp_path: Path) -> None:
        ran: list[Job] = []

        jobs = pending_jobs(
            {"genre": CONTROLS["genre"]},
            ["morphological/sp/1_2gram"],
            data_root=tmp_path,
            embeddings_root=tmp_path,
            parameters={"genre_csv": tmp_path / "genre.csv"},
        )
        sweep(jobs, runner=ran.append)

        assert [job.key for job in ran] == ["morphological/sp/1_2gram"]
        assert jobs[0].output.parent.is_dir()

    def test_a_failing_job_does_not_stop_the_rest(self, tmp_path: Path) -> None:
        """A sweep of many families must report a failure rather than abandon the remainder."""
        seen: list[str] = []

        def _runner(job: Job) -> None:
            seen.append(job.key)
            if job.key.startswith("morphological"):
                raise RuntimeError("no support table")

        jobs = pending_jobs(
            CONTROLS,
            ["morphological/sp/1_2gram", "syntactic/phrase/typ/1_2gram"],
            data_root=tmp_path,
            embeddings_root=tmp_path,
            parameters={"genre_csv": tmp_path / "genre.csv"},
        )
        failures = sweep(jobs, runner=_runner)

        assert len(seen) == len(jobs)
        assert {key for key, _ in failures} == {"morphological/sp/1_2gram"}

    def test_an_interrupt_ends_the_sweep_instead_of_being_recorded(self, tmp_path: Path) -> None:
        """Ctrl-C during one family must stop the run, since the operator asked for that."""
        seen: list[str] = []

        def _runner(job: Job) -> None:
            seen.append(job.key)
            raise KeyboardInterrupt

        jobs = pending_jobs(
            CONTROLS,
            ["morphological/sp/1_2gram", "syntactic/phrase/typ/1_2gram"],
            data_root=tmp_path,
            embeddings_root=tmp_path,
            parameters={"genre_csv": tmp_path / "genre.csv"},
        )
        with pytest.raises(KeyboardInterrupt):
            sweep(jobs, runner=_runner)

        assert len(seen) == 1

    def test_a_control_that_exits_is_recorded_as_a_failure(self, tmp_path: Path) -> None:
        """A control's argparse exit is a bad job, which the sweep reports and moves past."""

        def _runner(job: Job) -> None:
            raise SystemExit(2)

        jobs = pending_jobs(
            {"genre": CONTROLS["genre"]},
            ["morphological/sp/1_2gram"],
            data_root=tmp_path,
            embeddings_root=tmp_path,
            parameters={"genre_csv": tmp_path / "genre.csv"},
        )
        failures = sweep(jobs, runner=_runner)

        assert [key for key, _ in failures] == ["morphological/sp/1_2gram"]


class TestJobArgv:
    def test_a_parallelism_job_names_its_family_and_its_real_embeddings(
        self, tmp_path: Path
    ) -> None:
        jobs = pending_jobs(
            {"parallelism": CONTROLS["parallelism"]},
            ["syntactic/phrase/typ/1_2gram"],
            data_root=tmp_path,
            embeddings_root=tmp_path,
            parameters={"genre_csv": tmp_path / "genre.csv"},
        )

        argv = jobs[0].argv(n_shuffles=1000, workers=5, config_root=tmp_path / "config")

        assert argv[0] == str(jobs[0].real_embeddings)
        assert "--family" in argv
        assert argv[argv.index("--family") + 1] == "syntactic/phrase/typ/1_2gram"
        assert argv[argv.index("--n-shuffles") + 1] == "1000"
        assert argv[argv.index("--workers") + 1] == "5"

    def test_a_genre_job_passes_the_genre_csv_before_the_embeddings(self, tmp_path: Path) -> None:
        """The genre control takes its third-party CSV as the first positional."""
        csv = tmp_path / "psalms-browser.csv"
        jobs = pending_jobs(
            {"genre": CONTROLS["genre"]},
            ["syntactic/phrase/typ/1_2gram"],
            data_root=tmp_path,
            embeddings_root=tmp_path,
            parameters={"genre_csv": csv},
        )

        argv = jobs[0].argv(n_shuffles=10, workers=1, config_root=tmp_path / "config")

        assert argv[:2] == [str(csv), str(jobs[0].real_embeddings)]


class TestRunJob:
    def test_a_job_is_handed_to_the_benchmark_its_key_names(self, tmp_path: Path) -> None:
        """A job routed to the wrong control would score a family under the wrong benchmark."""
        seen: list[list[str]] = []
        control = Control(seen.append)
        jobs = pending_jobs(
            {"parallelism": control},
            ["syntactic/phrase/typ/1_2gram"],
            data_root=tmp_path,
            embeddings_root=tmp_path,
        )

        run_job(
            jobs[0],
            controls={"parallelism": control},
            n_shuffles=7,
            workers=1,
            config_root=tmp_path / "config",
        )

        assert seen[0][seen[0].index("--n-shuffles") + 1] == "7"


class TestSharedApi:
    """Every declared control reuses one loaded BHSA, so a sweep does not reload it per family."""

    def test_a_second_job_on_the_same_benchmark_reuses_the_first_load(self) -> None:
        loads: list[str] = []

        def _factory(checkout: str) -> object:
            loads.append(checkout)
            return object()

        shared = shared_api_factory(_factory)
        first = shared("v1.8.1")
        second = shared("v1.8.1")

        assert first is second
        assert loads == ["v1.8.1"]

    def test_only_the_most_recent_checkout_is_held(self) -> None:
        """Holding every loaded corpus swaps a long sweep to disk, costing more than a reload."""
        loads: list[str] = []

        def _factory(checkout: str) -> object:
            loads.append(checkout)
            return object()

        shared = shared_api_factory(_factory)
        shared("v1.8.1")
        shared("v1.9.0")
        shared("v1.8.1")

        assert loads == ["v1.8.1", "v1.9.0", "v1.8.1"]

    def test_a_different_checkout_loads_separately(self) -> None:
        loads: list[str] = []

        def _factory(checkout: str) -> object:
            loads.append(checkout)
            return object()

        shared = shared_api_factory(_factory)
        shared("v1.8.1")
        shared("v1.9.0")

        assert loads == ["v1.8.1", "v1.9.0"]

    def test_every_declared_control_binds_a_shared_loader(self) -> None:
        """An unbound control would reload BHSA for each of the families the sweep runs."""
        for control in CONTROLS.values():
            assert control.main.keywords["api_factory"] is not None
