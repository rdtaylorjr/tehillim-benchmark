"""Selects which embedding files a batch run still needs to score."""

from pathlib import Path

from core.datasets import dataset_identifier, dataset_paths


def uncached_model_paths(embeddings_dir: Path, cached_models: set[str]) -> list[Path]:
    """Every dataset file, in canonical order, whose model a prior run has not already scored."""
    paths = dataset_paths(embeddings_dir)
    return [path for path in paths if dataset_identifier(path) not in cached_models]
