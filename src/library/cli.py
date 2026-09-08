"""The arguments and cache handling every batch scoring script shares."""

import argparse
import sys
from pathlib import Path
from typing import Any

from library.bhsa import DEFAULT_CHECKOUT
from library.incremental_cache import load_cached_rows
from library.model_files import uncached_model_paths
from library.order_shuffle import DEFAULT_N_SHUFFLES
from library.protocol import (
    DEFAULT_N_GROUP_PERMUTATIONS,
    DEFAULT_N_PERMUTATIONS,
    DEFAULT_N_RESAMPLES,
)
from library.worker_pool import default_max_workers


def add_genre_csv_argument(parser: argparse.ArgumentParser) -> None:
    """The third-party genre CSV every genre script takes as its first positional."""
    parser.add_argument(
        "genre_csv",
        type=Path,
        help="third-party genre CSV, e.g. psalms-browser.csv (not in this repo)",
    )


def add_embeddings_dir_argument(parser: argparse.ArgumentParser) -> None:
    """The embeddings directory every batch script scores, declared once for all of them."""
    parser.add_argument("embeddings_dir", type=Path)


def add_scoring_arguments(
    parser: argparse.ArgumentParser,
    *,
    with_seed: bool = False,
    with_resamples: bool = False,
    with_permutations: bool = False,
    with_group_permutations: bool = False,
    with_shuffles: bool = False,
) -> None:
    """Adds the options every batch script takes, so a changed default lands in one place."""
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA checkout spec")
    parser.add_argument("--workers", type=int, default=default_max_workers())
    parser.add_argument("--output", type=Path, default=None)
    if with_seed:
        parser.add_argument("--seed", type=int, default=0)
    if with_resamples:
        parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    if with_permutations:
        parser.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    if with_group_permutations:
        parser.add_argument("--n-permutations", type=int, default=DEFAULT_N_GROUP_PERMUTATIONS)
    if with_shuffles:
        parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)


def report_reuse(models: set[str], source: Path) -> None:
    """Says what a run is reusing, worded once so every script reports it identically."""
    if models:
        print(f"reusing {len(models)} cached models from {source}", file=sys.stderr)


def load_cache(path: Path | None) -> tuple[list[dict[str, Any]], set[str]]:
    """Rows a prior run already scored and their model names, reported the same way everywhere."""
    if path is None or not path.exists():
        return [], set()
    rows, models = load_cached_rows(path)
    report_reuse(models, path)
    return rows, models


def resume_from_cache(
    embeddings_dir: Path, output: Path | None
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Rows a prior run already produced, and the embedding files still left to score."""
    cached_rows, cached_models = load_cache(output)
    return cached_rows, uncached_model_paths(embeddings_dir, cached_models)
