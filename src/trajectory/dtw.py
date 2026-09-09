"""Dynamic time warping (Sakoe & Chiba 1978) for aligning two variable-length sequences."""

import numba
import numpy as np

from library.errors import InsufficientDataError


@numba.njit(cache=True)
def _dtw_accumulated_cost_jit(cost_matrix: np.ndarray) -> np.ndarray:
    """JIT-compiled Sakoe-Chiba symmetric-form DP recursion, the hot path over 325 half-verses."""
    n, m = cost_matrix.shape
    accumulated = np.zeros((n, m))
    accumulated[0, 0] = cost_matrix[0, 0]
    for i in range(1, n):
        accumulated[i, 0] = accumulated[i - 1, 0] + cost_matrix[i, 0]
    for j in range(1, m):
        accumulated[0, j] = accumulated[0, j - 1] + cost_matrix[0, j]
    for i in range(1, n):
        for j in range(1, m):
            up, left, diag = accumulated[i - 1, j], accumulated[i, j - 1], accumulated[i - 1, j - 1]
            accumulated[i, j] = cost_matrix[i, j] + min(up, left, diag)
    return accumulated


def dtw_accumulated_cost(cost_matrix: np.ndarray) -> np.ndarray:
    """Sakoe-Chiba symmetric-form DP recursion over a precomputed n x m pairwise cost matrix."""
    n, m = cost_matrix.shape
    if n == 0 or m == 0:
        raise InsufficientDataError(
            "dtw_accumulated_cost needs at least one element in each sequence"
        )
    return _dtw_accumulated_cost_jit(np.ascontiguousarray(cost_matrix, dtype=np.float64))


def dtw_warping_path(accumulated_cost: np.ndarray) -> list[tuple[int, int]]:
    """Backtracks the accumulated-cost matrix to the optimal warping path, in forward order."""
    i, j = accumulated_cost.shape[0] - 1, accumulated_cost.shape[1] - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                accumulated_cost[i - 1, j],
                accumulated_cost[i, j - 1],
                accumulated_cost[i - 1, j - 1],
            ]
            step = int(np.argmin(candidates))
            if step == 0:
                i -= 1
            elif step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    return path[::-1]


def dtw_distance(cost_matrix: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """DTW alignment distance (accumulated cost normalized by warping-path length) and the path."""
    accumulated = dtw_accumulated_cost(cost_matrix)
    path = dtw_warping_path(accumulated)
    total = accumulated[-1, -1]
    return float(total / len(path)), path
