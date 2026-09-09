import os

from library.worker_pool import (
    chunksize_for,
    default_max_workers,
    map_in_order,
)


def _double(value: int) -> int:
    return value * 2


def _pid(value: int) -> tuple[int, int]:
    return value, os.getpid()


def test_map_in_order_returns_one_result_per_item() -> None:
    assert map_in_order(_double, [1, 2, 3], max_workers=2) == [2, 4, 6]


def test_map_in_order_preserves_submission_order_not_completion_order() -> None:
    """Row order must never depend on worker timing, or reruns stop being byte-comparable."""
    items = list(range(24))

    assert map_in_order(_double, items, max_workers=4) == [i * 2 for i in items]


def test_map_in_order_matches_a_plain_sequential_map() -> None:
    items = list(range(10))

    assert map_in_order(_double, items, max_workers=4) == [_double(i) for i in items]


def test_map_in_order_runs_in_this_process_when_workers_is_one() -> None:
    """A single worker stays in-process, so tests and debugging avoid the spawn cost."""
    results = map_in_order(_pid, [1, 2], max_workers=1)

    assert [value for value, _ in results] == [1, 2]
    assert {pid for _, pid in results} == {os.getpid()}


def test_map_in_order_uses_worker_processes_when_workers_exceeds_one() -> None:
    results = map_in_order(_pid, list(range(8)), max_workers=2)

    assert os.getpid() not in {pid for _, pid in results}


def test_map_in_order_handles_an_empty_item_list() -> None:
    assert map_in_order(_double, [], max_workers=4) == []


def test_default_max_workers_is_a_positive_worker_count() -> None:
    assert default_max_workers() >= 1


def test_map_in_order_chunks_work_so_a_shared_payload_is_not_repickled_per_item() -> None:
    """Per-model jobs close over the same corpus, so chunking keeps IPC off the critical path."""
    assert chunksize_for(n_items=1000, max_workers=4) > 1


def test_chunksize_is_one_when_there_is_at_most_one_item_per_worker() -> None:
    assert chunksize_for(n_items=3, max_workers=4) == 1


def test_chunksize_is_never_zero() -> None:
    assert chunksize_for(n_items=0, max_workers=4) == 1


def test_default_worker_count_does_not_oversubscribe_the_machine() -> None:
    """Scoring is compute-bound, so one worker per core is the most the machine can run."""
    #: Safe only because the pool pins each worker's BLAS to a single thread before spawning.
    assert 1 <= default_max_workers() <= (os.cpu_count() or 1)
