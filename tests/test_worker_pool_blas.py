"""Workers each link a multithreaded BLAS, so a pool must pin it or the cores thrash."""

from __future__ import annotations

import os

import pytest

from library.worker_pool import (
    BLAS_THREAD_VARIABLES,
    default_max_workers,
    map_in_order,
    pin_worker_blas_threads,
)


def _double(value: int) -> int:
    return value * 2


class TestPinWorkerBlasThreads:
    @pytest.mark.parametrize("variable", BLAS_THREAD_VARIABLES)
    def test_it_pins_every_backend_numpy_might_link(
        self, variable: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Accelerate reads VECLIB, OpenBLAS reads its own: missing one leaves the thrash in."""
        monkeypatch.delenv(variable, raising=False)

        pin_worker_blas_threads()

        assert os.environ[variable] == "1"

    def test_it_covers_apple_accelerate(self) -> None:
        """Numpy on this platform links Accelerate, which ignores OMP_NUM_THREADS."""
        assert "VECLIB_MAXIMUM_THREADS" in BLAS_THREAD_VARIABLES

    def test_an_explicit_setting_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An operator who asked for 4 threads per worker meant it."""
        monkeypatch.setenv("OMP_NUM_THREADS", "4")

        pin_worker_blas_threads()

        assert os.environ["OMP_NUM_THREADS"] == "4"


class TestDefaultMaxWorkers:
    def test_it_uses_the_whole_machine(self) -> None:
        """Capping at 3 was a workaround for the oversubscription pinning now removes."""
        assert default_max_workers() == (os.cpu_count() or 1)

    def test_it_is_never_zero(self) -> None:
        assert default_max_workers() >= 1


class TestMapInOrder:
    def test_results_stay_in_submission_order(self) -> None:
        assert map_in_order(_double, [1, 2, 3, 4], 2) == [2, 4, 6, 8]

    def test_the_pool_pins_blas_before_it_spawns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Children inherit the parent's environment at spawn, so it must be set by then."""
        monkeypatch.delenv("VECLIB_MAXIMUM_THREADS", raising=False)

        map_in_order(_double, [1, 2, 3, 4], 2)

        assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"

    def test_a_single_worker_still_runs_serially(self) -> None:
        assert map_in_order(_double, [1, 2, 3], 1) == [2, 4, 6]

    def test_it_defaults_to_the_whole_machine(self) -> None:
        assert map_in_order(_double, [1, 2]) == [2, 4]
