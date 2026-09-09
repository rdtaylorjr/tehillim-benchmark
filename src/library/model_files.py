"""Selects which embedding files a batch run still needs to score."""

import re
from pathlib import Path

from library.embeddings import dataset_identifier

#: A trailing draw number marks one sample of the order-shuffle null, not a model to score.
_SHUFFLE_DRAW = re.compile(r"shuffle\d+$")


def is_shuffle_draw(path: Path) -> bool:
    """True when a file is one draw of the order-shuffle null rather than a model to score."""
    return any(_SHUFFLE_DRAW.search(part.partition("=")[2] or part) for part in path.parts)


def names_a_model(path: Path) -> bool:
    """A parquet outside a Hive partition tree is some other output, so a batch passes it over."""
    parent = path.parent.name
    return "=" in parent and not parent.startswith("domain=")


def uncached_model_paths(embeddings_dir: Path, cached_models: set[str]) -> list[Path]:
    """Every embeddings file, in canonical order, whose model a prior run has not already scored."""
    return [
        path
        for path in sorted(embeddings_dir.glob("**/*.parquet"))
        if path.is_file()
        and not is_shuffle_draw(path)
        and names_a_model(path)
        and dataset_identifier(path) not in cached_models
    ]
