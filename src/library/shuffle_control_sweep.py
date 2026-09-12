"""Runs every registered family's order-shuffle control for every benchmark that lacks one."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from families.shuffle import dataset_source

import genre.scripts.shuffle_order_control as genre_control
import parallelism.scripts.shuffle_order_control as parallelism_control
from library.bhsa import load_bhsa_api
from parallelism.tf_features import load_api

__all__ = [
    "CONTROLS",
    "Control",
    "Job",
    "control_filename",
    "pending_jobs",
    "run_job",
    "shared_api_factory",
    "sweep",
]

#: A control's domain-level partition, which the committed tree already names.
STAGE = "shuffle_control"

#: Dropped from a control's filename, because the committed tree omits it.
IMPLICIT_LEVEL = "phrase"


def shared_api_factory(factory: Callable[[str], Any]) -> Callable[[str], Any]:
    """Wraps a BHSA loader so a sweep loads each checkout once instead of once per family."""
    #: One entry, because a loaded corpus is gigabytes and a sweep uses one checkout throughout.
    return lru_cache(maxsize=1)(factory)


@dataclass(frozen=True, slots=True)
class Control:
    """One benchmark's order-shuffle control: how it is invoked and what it needs first."""

    main: Any
    #: Run parameters this benchmark takes as leading positionals, resolved by name at sweep time.
    leading: tuple[str, ...] = ()


CONTROLS: Mapping[str, Control] = {
    "genre": Control(
        partial(genre_control.main, api_factory=shared_api_factory(load_bhsa_api)),
        leading=("genre_csv",),
    ),
    "parallelism": Control(
        partial(
            parallelism_control.main,
            api_factory=shared_api_factory(load_api),
        )
    ),
}


def control_filename(key: str) -> str:
    """The stem a family's control writes under, dropping the domain and the implicit level."""
    return "_".join(part for part in key.split("/")[1:] if part != IMPLICIT_LEVEL)


@dataclass(frozen=True, slots=True)
class Job:
    """One family scored under one benchmark, with every path it needs already resolved."""

    benchmark: str
    key: str
    real_embeddings: Path
    output: Path
    leading: tuple[str, ...] = field(default=())

    def argv(self, *, n_shuffles: int, workers: int, config_root: Path) -> list[str]:
        """The command line the benchmark's own control script parses."""
        return [
            *self.leading,
            str(self.real_embeddings),
            "--family",
            self.key,
            "--config-root",
            str(config_root),
            "--n-shuffles",
            str(n_shuffles),
            "--workers",
            str(workers),
            "--output",
            str(self.output),
        ]


def _output_path(benchmark: str, key: str, data_root: Path) -> Path:
    """Where a control result lands, under the partition the committed tree uses."""
    return (
        data_root
        / "analysis=benchmark"
        / f"benchmark={benchmark}"
        / f"domain={key.split('/', maxsplit=1)[0]}"
        / f"stage={STAGE}"
        / f"{control_filename(key)}.csv"
    )


def pending_jobs(
    controls: Mapping[str, Control],
    keys: Iterable[str],
    *,
    data_root: Path,
    embeddings_root: Path,
    parameters: Mapping[str, Path] | None = None,
) -> list[Job]:
    """Every benchmark and family pairing whose result is not in the tree yet."""
    supplied = parameters or {}
    jobs = []
    for benchmark, control in controls.items():
        leading = tuple(str(supplied[name]) for name in control.leading)
        for key in keys:
            output = _output_path(benchmark, key, data_root)
            if output.exists():
                continue
            jobs.append(
                Job(
                    benchmark=benchmark,
                    key=key,
                    real_embeddings=dataset_source(key, embeddings_root),
                    output=output,
                    leading=leading,
                )
            )
    return jobs


def run_job(
    job: Job,
    *,
    controls: Mapping[str, Control],
    n_shuffles: int,
    workers: int,
    config_root: Path,
) -> None:
    """Invokes the control the job's benchmark declares, with the arguments that job resolved."""
    controls[job.benchmark].main(
        job.argv(n_shuffles=n_shuffles, workers=workers, config_root=config_root)
    )


def sweep(jobs: Sequence[Job], *, runner: Callable[[Job], None]) -> list[tuple[str, BaseException]]:
    """Runs every job, returning the ones that raised so one bad family cannot end the sweep."""
    failures: list[tuple[str, BaseException]] = []
    #: An interrupt is the operator ending the run, so only what a control raises is collected.
    for job in jobs:
        job.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            runner(job)
        except (Exception, SystemExit) as error:  # noqa: BLE001
            failures.append((job.key, error))
    return failures
