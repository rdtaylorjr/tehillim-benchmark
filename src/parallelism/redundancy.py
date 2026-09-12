"""Conditional redundancy: what one representation adds to another, and what it only repeats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from library.embeddings import dataset_identifier, load_embeddings
from library.retrieval_metrics import cosine_similarity_matrix
from parallelism.evaluate import build_side_vectors
from parallelism.pairs import filter_pairs_with_vectors
from parallelism.separation import similarity_separation

if TYPE_CHECKING:
    from pathlib import Path

    from parallelism.pairs import RetrievalPair

__all__ = ["RedundancyResult", "conditional_redundancy", "joined_vectors"]


@dataclass(frozen=True, slots=True)
class RedundancyResult:
    """One pair of representations, each scored alone and together on the half-verses they share."""

    subject: str
    reference: str
    n_pairs: int
    auc_subject: float
    auc_reference: float
    auc_joined: float
    delta_over_reference: float
    delta_over_subject: float


def joined_vectors(
    subject: dict[int, np.ndarray], reference: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """Each half-verse's two vectors concatenated, both L2-normalised so neither width dominates."""
    shared = subject.keys() & reference.keys()
    joined: dict[int, np.ndarray] = {}
    for node in shared:
        blocks = []
        for source in (subject[node], reference[node]):
            widened = source.astype(np.float64)
            norm = np.linalg.norm(widened)
            blocks.append(widened / norm if norm else widened)
        joined[node] = np.concatenate(blocks)
    return joined


def _separation_auc(node_vectors: dict[int, np.ndarray], pairs: list[RetrievalPair]) -> float:
    """Separation AUC of one representation over the pairs it covers."""
    usable = filter_pairs_with_vectors(pairs, node_vectors)
    similarities = cosine_similarity_matrix(
        build_side_vectors(usable, "source", node_vectors),
        build_side_vectors(usable, "target", node_vectors),
    )
    return similarity_separation(similarities).auc


def conditional_redundancy(
    subject_path: Path, reference_path: Path, pairs: list[RetrievalPair]
) -> RedundancyResult:
    """Scores subject, reference and their join on the half-verses both representations cover."""
    subject = load_embeddings(subject_path)
    reference = load_embeddings(reference_path)
    joined = joined_vectors(subject, reference)
    #: Every AUC is taken on the shared population, so a delta is not a population difference.
    shared_pairs = filter_pairs_with_vectors(pairs, joined)
    auc_subject = _separation_auc({n: subject[n] for n in joined}, shared_pairs)
    auc_reference = _separation_auc({n: reference[n] for n in joined}, shared_pairs)
    auc_joined = _separation_auc(joined, shared_pairs)
    return RedundancyResult(
        subject=dataset_identifier(subject_path),
        reference=dataset_identifier(reference_path),
        n_pairs=len(shared_pairs),
        auc_subject=auc_subject,
        auc_reference=auc_reference,
        auc_joined=auc_joined,
        delta_over_reference=auc_joined - auc_reference,
        delta_over_subject=auc_joined - auc_subject,
    )
