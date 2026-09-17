"""The sweep runs the planned control cells whose results are missing, and only those."""

from __future__ import annotations

from pathlib import Path

import pytest

from library.shuffle_control_sweep import (
    CONTROLS,
    Job,
    pending_jobs,
    run_job,
    shared_api_factory,
    sweep,
)
from library.stages import SCOPE, Roots, control_filename, genre_registers


@pytest.fixture
def roots(tmp_path: Path) -> Roots:
    """Roots over a tree holding one morphological and one syntactic dataset."""
    embeddings = tmp_path / "emb"
    for domain, unit in (
        ("morphological", "feature=sp/construction=1_2gram"),
        ("syntactic", "level=phrase/feature=typ/construction=1_2gram"),
    ):
        part = embeddings / SCOPE.directory / f"domain={domain}" / unit / "part-0.parquet"
        part.parent.mkdir(parents=True)
        part.write_bytes(b"x")
    return Roots(
        data_root=tmp_path / "data",
        embeddings_root=embeddings,
        config_root=tmp_path / "config",
        genre_csv=tmp_path / "genre.csv",
        gunkel_csv=tmp_path / "gunkel.csv",
        ui_root=tmp_path / "ui",
        workers=2,
    )


class TestControlFilename:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("morphological/sp/1_2gram", "sp_1_2gram"),
            ("syntactic/phrase/typ/1_2gram", "typ_1_2gram"),
            ("syntactic/clause/tab/transition_psalm", "clause_tab_transition_psalm"),
        ],
    )
    def test_the_domain_and_the_phrase_level_are_dropped(self, key: str, expected: str) -> None:
        assert control_filename(key) == expected

    def test_every_registered_family_maps_to_a_distinct_name(self) -> None:
        from families.shuffle import FAMILIES

        names = [control_filename(key) for key in FAMILIES]

        assert len(set(names)) == len(names)


class TestPendingJobs:
    def test_a_family_with_no_output_is_pending_under_every_benchmark_and_register(
        self, roots: Roots
    ) -> None:
        jobs = pending_jobs(roots, ["morphological/sp/1_2gram"])

        assert {job.key for job in jobs} == {"morphological/sp/1_2gram"}
        assert {job.benchmark for job in jobs} == {"parallelism", "genre"}
        taxonomies = {taxonomy for taxonomy, _ in genre_registers()}
        assert len(jobs) == 1 + len(taxonomies)
        assert sum(len(job.outputs) for job in jobs) == 1 + len(genre_registers())

    def test_a_family_whose_output_exists_is_skipped(self, roots: Roots) -> None:
        """A sweep is resumable, so an interrupted run does not rescore what it finished."""
        jobs = pending_jobs(roots, ["morphological/sp/1_2gram"])
        for output in (output for job in jobs for output in job.outputs):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("already scored\n")

        assert pending_jobs(roots, ["morphological/sp/1_2gram"]) == []

    def test_the_output_is_the_planned_cells_output(self, roots: Roots) -> None:
        jobs = pending_jobs(roots, ["syntactic/phrase/typ/1_2gram"])
        parallelism = next(job for job in jobs if job.benchmark == "parallelism")
        gunkel = next(job for job in jobs if ".gunkel.shuffle." in job.cell.name)

        assert parallelism.outputs == (
            roots.data_root
            / (
                "analysis=benchmark/benchmark=parallelism/domain=syntactic/"
                "stage=shuffle_control/typ_1_2gram.csv"
            ),
        )
        assert gunkel.outputs[0] == roots.data_root / (
            "analysis=benchmark/benchmark=genre/taxonomy=gunkel/unit=song/domain=syntactic/"
            "stage=shuffle_control/typ_1_2gram.csv"
        )
        assert [p.parent.parent.parent.name for p in gunkel.outputs] == [
            "unit=song",
            "unit=song_component",
            "unit=song_component_motif",
        ]

    def test_only_the_named_families_are_planned(self, roots: Roots) -> None:
        jobs = pending_jobs(roots, ["syntactic/phrase/typ/1_2gram"])

        assert {job.key for job in jobs} == {"syntactic/phrase/typ/1_2gram"}


class TestSweep:
    def test_each_job_is_run_once_and_its_parent_directory_created(self, roots: Roots) -> None:
        ran: list[Job] = []

        jobs = pending_jobs(roots, ["morphological/sp/1_2gram"])
        sweep(jobs, runner=ran.append)

        assert [job.cell.name for job in ran] == [job.cell.name for job in jobs]
        assert all(output.parent.is_dir() for output in jobs[0].outputs)

    def test_a_failing_job_does_not_stop_the_rest(self, roots: Roots) -> None:
        """A sweep of many families must report a failure rather than abandon the remainder."""
        seen: list[str] = []

        def _runner(job: Job) -> None:
            seen.append(job.key)
            if job.key.startswith("morphological"):
                raise RuntimeError("no support table")

        jobs = pending_jobs(roots, ["morphological/sp/1_2gram", "syntactic/phrase/typ/1_2gram"])
        failures = sweep(jobs, runner=_runner)

        assert len(seen) == len(jobs)
        assert {key for key, _ in failures} == {"morphological/sp/1_2gram"}

    def test_an_interrupt_ends_the_sweep_instead_of_being_recorded(self, roots: Roots) -> None:
        """Ctrl-C during one family must stop the run, since the operator asked for that."""
        seen: list[str] = []

        def _runner(job: Job) -> None:
            seen.append(job.key)
            raise KeyboardInterrupt

        jobs = pending_jobs(roots, ["morphological/sp/1_2gram", "syntactic/phrase/typ/1_2gram"])
        with pytest.raises(KeyboardInterrupt):
            sweep(jobs, runner=_runner)

        assert len(seen) == 1

    def test_a_control_that_exits_is_recorded_as_a_failure(self, roots: Roots) -> None:
        """A control's argparse exit is a bad job, which the sweep reports and moves past."""

        def _runner(job: Job) -> None:
            raise SystemExit(2)

        jobs = pending_jobs(roots, ["morphological/sp/1_2gram"])
        failures = sweep(jobs, runner=_runner)

        assert {key for key, _ in failures} == {"morphological/sp/1_2gram"}


class TestJobArgv:
    def test_a_job_carries_the_planned_arguments_plus_the_sweeps_run_size(
        self, roots: Roots
    ) -> None:
        jobs = pending_jobs(roots, ["syntactic/phrase/typ/1_2gram"])
        genre = next(job for job in jobs if ".logos." in job.cell.name)

        argv = genre.argv(n_shuffles=1000)

        assert argv[:3] == [str(roots.genre_csv), "--taxonomy", "logos"]
        assert argv[argv.index("--family") + 1] == "syntactic/phrase/typ/1_2gram"
        assert argv[argv.index("--n-shuffles") + 1] == "1000"
        assert argv[argv.index("--workers") + 1] == "2"


class TestRunJob:
    def test_a_job_is_handed_to_the_control_its_cell_names(self, roots: Roots) -> None:
        """A job routed to the wrong control would score a family under the wrong benchmark."""
        seen: dict[str, list[str]] = {}
        controls = {
            module: (lambda argv, module=module: seen.setdefault(module, argv))
            for module in CONTROLS
        }
        jobs = pending_jobs(roots, ["syntactic/phrase/typ/1_2gram"])
        parallelism = next(job for job in jobs if job.benchmark == "parallelism")

        run_job(parallelism, controls=controls, n_shuffles=7)

        argv = seen["parallelism.scripts.shuffle_order_control"]
        assert argv[argv.index("--n-shuffles") + 1] == "7"
        assert "genre.scripts.shuffle_order_control" not in seen


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
            assert control.keywords["api_factory"] is not None  # type: ignore[attr-defined]
