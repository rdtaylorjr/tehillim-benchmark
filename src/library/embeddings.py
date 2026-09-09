"""Reads BHSA-node-keyed embedding vectors from tehillim-embeddings' Parquet files."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

from library.errors import BenchmarkDataError

TEXT_VARIANTS = ("consonantal", "vocalized", "cantillation")


def dataset_identifier(path: Path) -> str:
    """Joins every Hive partition value between the file and its `domain=` root, deepest first."""
    parts = []
    node = path.parent
    while "=" in node.name and not node.name.startswith("domain="):
        parts.append(node.name.split("=", 1)[1])
        node = node.parent
    #: An unpartitioned file names no model, and a blank name would reach the output as a real row.
    if not parts:
        raise BenchmarkDataError(f"{path} carries no Hive partition, so it names no model")
    return "_".join(reversed(parts))


def split_model_name(model: str, text_variants: tuple[str, ...] = TEXT_VARIANTS) -> tuple[str, str]:
    """Splits into (base_model, text_variant): a trailing suffix, else a variant token anywhere."""
    for variant in text_variants:
        suffix = f"_{variant}"
        if model.endswith(suffix):
            return model[: -len(suffix)].removeprefix("semantic_"), variant
    tokens = model.split("_")
    for variant in text_variants:
        if variant in tokens:
            base = "_".join(t for t in tokens if t != variant)
            return base, variant
    return model.removeprefix("semantic_"), "unknown"


def is_sparse_embeddings(path: Path) -> bool:
    """Sparse files carry indices/values in place of a dense vector column."""
    return "vector" not in pq.read_schema(path).names


def load_embeddings(path: Path) -> dict[int, np.ndarray]:
    """Reads a Parquet embeddings file into a {node: vector} map, excluding zero-norm vectors."""
    if is_sparse_embeddings(path):
        #: `load_psalm_vectors` pools while still sparse, so prefer it for centroids.
        node_ids, matrix = load_sparse_embeddings(path)
        dense = matrix.toarray().astype("<f4", copy=False)
        return {node: dense[i] for i, node in enumerate(node_ids)}

    table = pq.read_table(path, columns=["node_id", "vector"])
    node_ids = table["node_id"].to_numpy(zero_copy_only=False)
    vector_column = table["vector"].combine_chunks()
    dim = vector_column.type.list_size
    matrix = vector_column.values.to_numpy(zero_copy_only=False).astype("<f4", copy=False)
    matrix = matrix.reshape(len(node_ids), dim)
    nonzero = np.any(matrix, axis=1)
    return {int(node_ids[i]): matrix[i] for i in np.flatnonzero(nonzero)}


def load_sparse_embeddings(path: Path) -> tuple[list[int], sp.csr_matrix]:
    """Reads a sparse Parquet embeddings file: node ids in row order, and one CSR matrix."""
    table = pq.read_table(path, columns=["node_id", "indices", "values"])
    dim = int(table.schema.metadata[b"dim"])
    node_ids = table["node_id"].to_pylist()
    indices_col = table["indices"].to_pylist()
    values_col = table["values"].to_pylist()

    row_lengths = [len(indices) for indices in indices_col]
    indptr = np.concatenate([[0], np.cumsum(row_lengths)])
    flat_indices = (
        np.concatenate([np.asarray(idx, dtype=np.int32) for idx in indices_col])
        if any(row_lengths)
        else np.zeros(0, dtype=np.int32)
    )
    flat_values = (
        np.concatenate([np.asarray(val, dtype="<f4") for val in values_col])
        if any(row_lengths)
        else np.zeros(0, dtype="<f4")
    )
    matrix = sp.csr_matrix((flat_values, flat_indices, indptr), shape=(len(node_ids), dim))

    nonzero_rows = np.flatnonzero(np.diff(matrix.indptr) > 0)
    return [node_ids[i] for i in nonzero_rows], matrix[nonzero_rows]
