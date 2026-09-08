"""Loader checks against the real committed embeddings, which synthetic fixtures cannot cover."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from library.embeddings import is_sparse_embeddings, load_embeddings, load_sparse_embeddings
from library.psalm_vectors import load_psalm_vectors

#: Overridable so the suite is not tied to one checkout layout.
DATA_ROOT = Path(
    os.environ.get(
        "TEHILLIM_EMBEDDINGS_DATA",
        Path.home() / "Developer" / "research" / "tehillim-embeddings" / "data",
    )
)
DENSE = DATA_ROOT / "domain=morphological/feature=sp/construction=1_2_3gram/part-0.parquet"
SPARSE = (
    DATA_ROOT / "domain=syntactic/level=phrase/feature=typ/construction=1_2_3gram/part-0.parquet"
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATA_ROOT.exists(), reason="no local tehillim-embeddings checkout"),
]


def _psalm_groups(node_ids: list[int], per_group: int = 35) -> dict[int, list[int]]:
    """Deterministic node grouping, standing in for the real psalm partition."""
    groups: dict[int, list[int]] = {}
    for index, node in enumerate(node_ids):
        groups.setdefault(index // per_group, []).append(node)
    return groups


def test_the_dense_family_is_detected_as_dense() -> None:
    assert is_sparse_embeddings(DENSE) is False


def test_the_sparse_family_is_detected_as_sparse() -> None:
    assert is_sparse_embeddings(SPARSE) is True


def test_a_real_dense_family_loads_as_float32_rows() -> None:
    vectors = load_embeddings(DENSE)

    assert len(vectors) > 5000
    assert next(iter(vectors.values())).dtype == np.dtype("<f4")


def test_a_real_sparse_family_loads_through_the_same_entry_point() -> None:
    """A sparse schema raised here before the loader learned to dispatch on it."""
    vectors = load_embeddings(SPARSE)

    assert len(vectors) > 5000
    assert next(iter(vectors.values())).dtype == np.dtype("<f4")


def test_densified_sparse_rows_match_the_sparse_matrix_exactly() -> None:
    """The densifying branch must reproduce the stored values bit for bit."""
    node_ids, matrix = load_sparse_embeddings(SPARSE)
    dense_rows = load_embeddings(SPARSE)

    for index in (0, len(node_ids) // 2, len(node_ids) - 1):
        expected = np.asarray(matrix[index].todense()).ravel().astype("<f4")
        assert np.array_equal(dense_rows[node_ids[index]], expected)


def test_psalm_centroids_agree_between_the_two_storage_forms() -> None:
    """Sparse pooling and dense pooling differ only by float32 summation order."""
    node_ids, _ = load_sparse_embeddings(SPARSE)
    groups = _psalm_groups(node_ids)

    sparse_pooled = load_psalm_vectors(SPARSE, groups)
    dense_rows = load_embeddings(SPARSE)
    for psalm, nodes in groups.items():
        expected = np.mean([dense_rows[n] for n in nodes], axis=0)
        assert np.allclose(sparse_pooled[psalm], expected, rtol=0, atol=1e-6)
