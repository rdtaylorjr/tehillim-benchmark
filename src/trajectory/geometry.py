"""Discrete-curve geometry of a psalm's half-verses sequence: step direction, size, and turning."""

import numpy as np

from library.retrieval_metrics import paired_cosine_similarity


def adjacent_similarity(sequence: np.ndarray) -> np.ndarray:
    """Cosine similarity between each half-verses and the next, length n-1."""
    return paired_cosine_similarity(sequence[:-1], sequence[1:])


def step_magnitude(sequence: np.ndarray) -> np.ndarray:
    """Euclidean displacement between each half-verses and the next, length n-1."""
    return np.asarray(np.linalg.norm(sequence[1:] - sequence[:-1], axis=1))


def turning_angle(sequence: np.ndarray) -> np.ndarray:
    """Angle between consecutive displacement pairs, length n-2, NaN where a step is zero-length."""
    displacements = sequence[1:] - sequence[:-1]
    a, b = displacements[:-1], displacements[1:]
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    valid = (a_norm > 0) & (b_norm > 0)
    cosines = np.full(len(a), np.nan)
    cosines[valid] = np.sum(a[valid] * b[valid], axis=1) / (a_norm[valid] * b_norm[valid])
    return np.asarray(np.arccos(np.clip(cosines, -1.0, 1.0)))
