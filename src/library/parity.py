"""Checks that every benchmark table covers every dataset of its domain, or names the gap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from core.datasets import dataset_identifier, discover_domains
from core.skips import skipped_in_log

from library.stages import BENCHMARK_ROOT, Roots, domain_datasets

#: Each benchmark's coverage table, and the raw stage whose log explains any skipped dataset.
MASTER_TABLES: dict[str, tuple[str, str]] = {
    "parallelism": ("stage=master/model_metrics_overall.parquet", "retrieval"),
    "genre": ("stage=master/genre_metrics_wide.parquet", "summary"),
    "trajectory": ("stage=raw/trajectory_distances.parquet", "distances"),
}


class ParityError(RuntimeError):
    """A benchmark table lacks a dataset with no recorded skip, or carries a stale one."""


def master_models(roots: Roots, benchmark: str, domain: str) -> set[str]:
    """The models one benchmark's coverage table holds for one domain, empty when absent."""
    relative, _ = MASTER_TABLES[benchmark]
    path = roots.data_root / BENCHMARK_ROOT / f"benchmark={benchmark}/domain={domain}" / relative
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=["model"])["model"].unique())


def check_parity(roots: Roots, log_root: Path) -> dict[str, dict[str, object]]:
    """Compares every benchmark table with its domain's datasets, raising on an unexplained gap."""
    report: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for domain in discover_domains(roots.embeddings_root):
        datasets = {dataset_identifier(p) for p in domain_datasets(roots, domain)}
        for benchmark, (_, raw_stage) in MASTER_TABLES.items():
            scored = master_models(roots, benchmark, domain)
            skipped = skipped_in_log(log_root / f"{benchmark}.{domain}.{raw_stage}.log")
            missing = datasets - scored
            unexplained = sorted(missing - skipped)
            stale = sorted(scored - datasets)
            key = f"{benchmark}.{domain}"
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
