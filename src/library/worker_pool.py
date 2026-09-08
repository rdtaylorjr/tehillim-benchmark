"""Runs an independent per-item job across worker processes, preserving submission order."""

# Named for the pool: "parallelism" here is the Hebrew poetic kind, benchmarked in src/parallelism.

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

#: Every backend numpy might link, since each reads only its own variable.
BLAS_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    #: numpy links Accelerate on this platform, which ignores the OpenMP variable entirely.
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pin_worker_blas_threads() -> None:
    """Gives each worker a single BLAS thread, leaving an operator's explicit setting alone."""
    for variable in BLAS_THREAD_VARIABLES:
        os.environ.setdefault(variable, "1")


def default_max_workers() -> int:
    """One worker per core, which only pays once each worker's BLAS is pinned to one thread."""
    return os.cpu_count() or 1


def chunksize_for(n_items: int, max_workers: int) -> int:
    """Items per task: enough chunks to balance load, few enough to stop repickling the payload."""
    return max(1, n_items // (max_workers * 4))


def map_in_order[ItemT, ResultT](
    fn: Callable[[ItemT], ResultT],
    items: Sequence[ItemT],
    max_workers: int | None = None,
) -> list[ResultT]:
    """Applies fn to every item, returning results in submission order so reruns stay comparable."""
    workers = default_max_workers() if max_workers is None else max_workers
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    #: Set before the pool spawns, because a worker reads these only as it imports numpy.
    pin_worker_blas_threads()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items, chunksize=chunksize_for(len(items), workers)))
