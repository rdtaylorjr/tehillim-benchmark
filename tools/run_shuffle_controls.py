"""Scores every planned order-shuffle control that lacks a result, sharing one loaded corpus."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path

from core.parallel import default_max_workers
from families.shuffle import FAMILIES

from library.driver import ROOT_KEYS
from library.order_shuffle import DEFAULT_N_SHUFFLES
from library.shuffle_control_sweep import CONTROLS, Job, pending_jobs, run_job, sweep
from library.stages import Roots


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the roots a sweep needs and the run size it scores each family at."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ROOT_KEYS:
        parser.add_argument(f"--{key.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--family", action="append", metavar="KEY")
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    parser.add_argument("--workers", type=int, default=default_max_workers())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    controls: Mapping[str, Callable[[list[str]], None]] = CONTROLS,
    runner: Callable[[Job], None] | None = None,
) -> None:
    """Runs every pending control, reporting what it skipped and what failed."""
    args = parse_arguments(argv)
    keys = sorted(args.family or FAMILIES)
    unknown = [key for key in keys if key not in FAMILIES]
    if unknown:
        raise SystemExit(f"not order-sensitive constructions: {unknown}")
    roots = Roots(**{key: getattr(args, key) for key in ROOT_KEYS}, workers=args.workers)
    jobs = pending_jobs(roots, keys)
    print(f"{len(jobs)} pending")
    if args.dry_run:
        for job in jobs:
            print(f"  {job.cell.name}")
        return
    execute = runner or partial(run_job, controls=controls, n_shuffles=args.n_shuffles)
    failures = sweep(jobs, runner=_announced(execute, len(jobs)))
    for key, error in failures:
        print(f"FAILED {key}: {error!r}", file=sys.stderr)
    if failures:
        raise SystemExit(1)


def _announced(execute: Callable[[Job], None], total: int) -> Callable[[Job], None]:
    """Prints each job as it starts, so a long sweep shows progress rather than silence."""
    state = {"done": 0}

    def _run(job: Job) -> None:
        state["done"] += 1
        print(f"[{state['done']}/{total}] {job.cell.name}", flush=True)
        execute(job)

    return _run


if __name__ == "__main__":
    main()
