import numpy as np
import pytest

from trajectory.sequence import normalize_sequence, psalm_half_verse_sequences


def test_psalm_half_verse_sequences_preserves_node_order() -> None:
    half_verses_by_psalm = {1: [100, 101, 102]}
    node_vectors = {
        100: np.array([1.0, 0.0]),
        101: np.array([0.0, 1.0]),
        102: np.array([1.0, 1.0]),
    }

    result = psalm_half_verse_sequences(half_verses_by_psalm, node_vectors)

    assert np.allclose(result[1], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_psalm_half_verse_sequences_keeps_one_sequence_per_psalm() -> None:
    half_verses_by_psalm = {1: [100], 2: [200, 201]}
    node_vectors = {
        100: np.array([1.0, 0.0]),
        200: np.array([0.0, 1.0]),
        201: np.array([0.0, 3.0]),
    }

    result = psalm_half_verse_sequences(half_verses_by_psalm, node_vectors)

    assert set(result) == {1, 2}
    assert result[1].shape == (1, 2)
    assert result[2].shape == (2, 2)


def test_psalm_half_verse_sequences_skips_a_psalm_with_a_partially_missing_node() -> None:
    half_verses_by_psalm = {1: [100, 101]}
    node_vectors = {100: np.array([1.0, 0.0])}

    result = psalm_half_verse_sequences(half_verses_by_psalm, node_vectors)

    assert result == {}


def test_normalize_sequence_gives_every_row_unit_norm() -> None:
    sequence = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])

    result = normalize_sequence(sequence)

    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0)


def test_normalize_sequence_preserves_direction() -> None:
    sequence = np.array([[3.0, 4.0]])

    result = normalize_sequence(sequence)

    assert np.allclose(result, [[0.6, 0.8]])


def test_normalize_sequence_rejects_a_zero_length_half_verse_vector() -> None:
    """A zero row would normalize to inf and silently poison every downstream curve."""
    sequence = np.array([[1.0, 0.0], [0.0, 0.0]])

    with pytest.raises(ValueError, match="zero vector"):
        normalize_sequence(sequence)


def test_psalm_half_verse_sequences_sparse_matches_the_dense_function_exactly() -> None:
    """Densifying per psalm must give bit-identical sequences to densifying the whole file."""
    import scipy.sparse as sp

    from trajectory.sequence import psalm_half_verse_sequences_sparse

    rng = np.random.default_rng(1)
    node_ids = [10, 11, 12, 13, 14]
    dense = {n: rng.standard_normal(6).astype("<f4") for n in node_ids}
    matrix = sp.csr_matrix(np.stack([dense[n] for n in node_ids]))
    by_psalm = {1: [10, 11], 2: [12, 13, 14]}

    sparse_seq = psalm_half_verse_sequences_sparse(by_psalm, node_ids, matrix)
    dense_seq = psalm_half_verse_sequences(by_psalm, dense)

    assert sorted(sparse_seq) == sorted(dense_seq)
    assert all(np.array_equal(sparse_seq[p], dense_seq[p]) for p in dense_seq)


def test_psalm_half_verse_sequences_sparse_skips_a_psalm_with_a_missing_node() -> None:
    """A psalm whose half-verse is absent from the file is dropped, matching the dense function."""
    import scipy.sparse as sp

    from trajectory.sequence import psalm_half_verse_sequences_sparse

    node_ids = [10, 11]
    matrix = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))

    result = psalm_half_verse_sequences_sparse({1: [10, 11], 2: [10, 99]}, node_ids, matrix)

    assert sorted(result) == [1]
