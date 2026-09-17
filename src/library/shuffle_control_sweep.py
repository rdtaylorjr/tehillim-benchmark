"""Runs every planned order-shuffle control whose output is missing, sharing one loaded BHSA."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from core.driver import Cell

import genre.scripts.shuffle_order_control as genre_control
import parallelism.scripts.shuffle_order_control as parallelism_control
from library.bhsa import load_bhsa_api
from library.stages import SHUFFLE_MODULE_SUFFIX, Roots, plan_cells
from parallelism.tf_features import load_api

__all__ = [
    "CONTROLS",
    "Job",
    "pending_jobs",
    "run_job",
    "shared_api_factory",
    "sweep",
]


def shared_api_factory(factory: Callable[[str], Any]) -> Callable[[str], Any]:
    """Wraps a BHSA loader so a sweep loads each checkout once instead of once per family."""
    #: One entry, because a loaded corpus is gigabytes and a sweep uses one checkout throughout.
    return lru_cache(maxsize=1)(factory)


#: Each control script's main by module, bound to a loader the whole sweep shares.
CONTROLS: Mapping[str, Callable[[list[str]], None]] = {
    "genre.scripts.shuffle_order_control": partial(
        genre_control.main, api_factory=shared_api_factory(load_bhsa_api)
    ),
    "parallelism.scripts.shuffle_order_control": partial(
        parallelism_control.main, api_factory=shared_api_factory(load_api)
    ),
}


@dataclass(frozen=True, slots=True)
class Job:
    """One planned control cell, with the family it scores read off its arguments."""

    cell: Cell

    @property
    def benchmark(self) -> str:
        """The benchmark segment of the cell name."""
        return self.cell.name.split(".")[0]

    @property
    def key(self) -> str:
        """The family the control scores."""
        args = self.cell.command_args
        return args[args.index("--family") + 1]

    @property
    def outputs(self) -> tuple[Path, ...]:
        """Where the control's results land, one per register it scores."""
        return self.cell.outputs

    def argv(self, *, n_shuffles: int) -> list[str]:
        """The command line the control script parses, at the sweep's run size."""
        return [*self.cell.command_args, "--n-shuffles", str(n_shuffles)]


def pending_jobs(roots: Roots, keys: Iterable[str]) -> list[Job]:
    """Every planned control for the named families whose result is not in the tree yet."""
    wanted = set(keys)
    return [
        job
        for job in (Job(cell) for cell in plan_cells(roots))
        if job.cell.module.endswith(SHUFFLE_MODULE_SUFFIX)
        and job.key in wanted
        and not all(output.exists() for output in job.outputs)
    ]


def run_job(
    job: Job, *, controls: Mapping[str, Callable[[list[str]], None]], n_shuffles: int
) -> None:
    """Invokes the control the job's cell declares, with the arguments the plan resolved."""
    controls[job.cell.module](job.argv(n_shuffles=n_shuffles))


def sweep(jobs: Sequence[Job], *, runner: Callable[[Job], None]) -> list[tuple[str, BaseException]]:
    """Runs every job, returning the ones that raised so one bad family cannot end the sweep."""
    failures: list[tuple[str, BaseException]] = []
    #: An interrupt is the operator ending the run, so only what a control raises is collected.
    for job in jobs:
        for output in job.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
        try:
            runner(job)
        except (Exception, SystemExit) as error:  # noqa: BLE001
            failures.append((job.key, error))
    return failures
