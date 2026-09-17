"""Checks that every benchmark table covers every dataset of its domain, or names the gap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from core.datasets import dataset_identifier, discover_domains
from core.skips import skipped_in_log

from library.stages import (
    BENCHMARK_ROOT,
    SCOPE,
    Roots,
    domain_datasets,
    genre_dir,
    genre_registers,
    register_name,
)


class ParityError(RuntimeError):
    """A benchmark table lacks a dataset with no recorded skip, or carries a stale one."""


def coverage_tables(roots: Roots, domain: str) -> list[tuple[str, Path, str]]:
    """Each chain of one domain: its cell prefix, its coverage table, and its raw stage."""
    benchmark = roots.data_root / BENCHMARK_ROOT
    return [
        (
            f"parallelism.{domain}",
            benchmark
            / f"benchmark=parallelism/domain={domain}"
            / "stage=master/model_metrics_overall.parquet",
            "retrieval",
        ),
        *(
            (
                f"genre.{domain}.{register_name(taxonomy, unit)}",
                genre_dir(roots, taxonomy, unit, domain)
                / "stage=master/genre_metrics_wide.parquet",
                "summary",
            )
            for taxonomy, unit in genre_registers()
        ),
        (
            f"trajectory.{domain}",
            benchmark
            / f"benchmark=trajectory/domain={domain}"
            / "stage=raw/trajectory_distances.parquet",
            "distances",
        ),
    ]


def master_models(path: Path) -> set[str]:
    """The models one coverage table holds, empty when the table is absent."""
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=["model"])["model"].unique())


def check_parity(roots: Roots, log_root: Path) -> dict[str, dict[str, object]]:
    """Compares every benchmark table with its domain's datasets, raising on an unexplained gap."""
    report: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for domain in discover_domains(roots.embeddings_root, SCOPE):
        datasets = {dataset_identifier(p) for p in domain_datasets(roots, domain)}
        for key, table, raw_stage in coverage_tables(roots, domain):
            scored = master_models(table)
            skipped = skipped_in_log(log_root / f"{key}.{raw_stage}.log")
            missing = datasets - scored
            unexplained = sorted(missing - skipped)
            stale = sorted(scored - datasets)
            report[key] = {
                "datasets": len(datasets),
                "scored": len(scored & datasets),
                "skipped": sorted(missing & skipped),
                "unexplained": unexplained,
                "stale": stale,
            }
            if unexplained or stale:
                failures.append(f"{key}: unexplained {unexplained} stale {stale}")
    if failures:
        raise ParityError("\n".join(failures))
    return report
