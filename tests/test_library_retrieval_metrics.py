import numpy as np
import pytest
import scipy.sparse as sp

from library.errors import DegenerateVectorError, InsufficientDataError
from library.retrieval_metrics import (
    _combine_by_stratum,
    _combine_by_stratum_batch,
    _per_anchor_gap,
    _per_anchor_gap_batch,
    cosine_similarity_matrix,
    mean_reciprocal_rank,
    paired_cosine_similarity,
    paired_discrimination_test,
    recall_at_k,
    sparse_cosine_similarity_matrix,
    sparse_paired_cosine_similarity,
    stratified_mean_gap_test,
)


def test_paired_cosine_similarity_is_one_for_identical_vectors() -> None:
    source = np.array([[1.0, 0.0], [0.0, 2.0]])
    target = np.array([[3.0, 0.0], [0.0, 5.0]])

    result = paired_cosine_similarity(source, target)

    assert np.allclose(result, [1.0, 1.0])


def test_paired_cosine_similarity_is_zero_for_orthogonal_vectors() -> None:
    source = np.array([[1.0, 0.0]])
    target = np.array([[0.0, 1.0]])

    result = paired_cosine_similarity(source, target)

    assert np.allclose(result, [0.0])


def test_paired_cosine_similarity_does_not_compare_across_rows() -> None:
    """Row i's similarity must only ever involve source[i] and target[i], nothing else."""
    source = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.array([[1.0, 0.0], [1.0, 0.0]])

    result = paired_cosine_similarity(source, target)

    assert np.allclose(result, [1.0, 0.0])


def test_cosine_similarity_matrix_of_identical_unit_vectors_is_one() -> None:
    a = np.array([[1.0, 0.0]])
    b = np.array([[1.0, 0.0]])
    result = cosine_similarity_matrix(a, b)
    assert result[0, 0] == pytest.approx(1.0)


def test_cosine_similarity_matrix_of_orthogonal_vectors_is_zero() -> None:
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    result = cosine_similarity_matrix(a, b)
    assert result[0, 0] == pytest.approx(0.0)


def test_cosine_similarity_matrix_is_scale_invariant() -> None:
    a = np.array([[2.0, 0.0]])
    b = np.array([[100.0, 0.0]])
    result = cosine_similarity_matrix(a, b)
    assert result[0, 0] == pytest.approx(1.0)


def test_cosine_similarity_matrix_raises_on_a_zero_vector() -> None:
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="zero"):
        cosine_similarity_matrix(a, b)


def test_sparse_cosine_similarity_matrix_of_identical_unit_vectors_is_one() -> None:
    a = sp.csr_matrix(np.array([[1.0, 0.0]]))
    b = sp.csr_matrix(np.array([[1.0, 0.0]]))
    result = sparse_cosine_similarity_matrix(a, b)
    assert result[0, 0] == pytest.approx(1.0)


def test_sparse_cosine_similarity_matrix_raises_on_a_zero_vector() -> None:
    a = sp.csr_matrix(np.array([[0.0, 0.0]]))
    b = sp.csr_matrix(np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="zero"):
        sparse_cosine_similarity_matrix(a, b)


def test_sparse_cosine_similarity_matrix_matches_the_dense_function_to_float64_last_bits() -> None:
    """Proves row-normalize-then-matmul over sparse inputs gives the identical dense result."""
    rng = np.random.default_rng(0)
    dim = 500
    a_dense = np.zeros((6, dim))
    b_dense = np.zeros((4, dim))
    for row in a_dense:
        n_nonzero = rng.integers(1, 6)
        idx = rng.choice(dim, size=n_nonzero, replace=False)
        row[idx] = rng.uniform(0.1, 5.0, size=n_nonzero)
    for row in b_dense:
        n_nonzero = rng.integers(1, 6)
        idx = rng.choice(dim, size=n_nonzero, replace=False)
        row[idx] = rng.uniform(0.1, 5.0, size=n_nonzero)

    expected = cosine_similarity_matrix(a_dense, b_dense)
    actual = sparse_cosine_similarity_matrix(sp.csr_matrix(a_dense), sp.csr_matrix(b_dense))

    #: Both paths accumulate in float64, measured to agree within 3.4e-16 over 20 draws.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


def test_mean_reciprocal_rank_of_all_first_place_is_one() -> None:
    assert mean_reciprocal_rank([1, 1, 1]) == 1.0


def test_mean_reciprocal_rank_averages_reciprocals() -> None:
    assert mean_reciprocal_rank([1, 2, 4]) == pytest.approx((1 + 0.5 + 0.25) / 3)


def test_recall_at_k_counts_ranks_within_k() -> None:
    assert recall_at_k([1, 2, 3, 10], k=3) == 0.75


def test_paired_discrimination_test_detects_strong_positive_signal() -> None:
    rng = np.random.default_rng(0)
    true_sims = rng.normal(0.8, 0.02, size=50)
    null_sims = rng.normal(0.2, 0.02, size=50)

    result = paired_discrimination_test(true_sims, null_sims)

    assert result.p_value < 0.001
    assert result.rank_biserial == pytest.approx(1.0)


def test_paired_discrimination_test_finds_no_signal_when_ties_cancel() -> None:
    true_sims = np.array([0.5, 0.6, 0.4, 0.5])
    null_sims = np.array([0.5, 0.4, 0.6, 0.5])

    result = paired_discrimination_test(true_sims, null_sims)

    assert result.rank_biserial == pytest.approx(0.0)


def test_per_anchor_gap_excludes_the_real_diagonal_even_under_permutation() -> None:
    """A permuted "true" column must not let the anchor's own diagonal leak into its null mean."""
    matrix = np.array([[10.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    true_positions = np.array([1, 1, 2])

    gaps = _per_anchor_gap(matrix, true_positions)

    assert gaps == pytest.approx([1.0 - 2.0, 4.0 - 4.0, 8.0 - 6.5])


def test_per_anchor_gap_falls_back_gracefully_when_n_is_too_small_to_exclude_both() -> None:
    """n=2 leaves no third column."""
    matrix = np.array([[10.0, 1.0], [2.0, 20.0]])

    gaps = _per_anchor_gap(matrix, np.array([1, 0]))

    assert not np.any(np.isnan(gaps))
    assert gaps == pytest.approx([1.0 - 10.0, 2.0 - 20.0])


def test_per_anchor_gap_matches_original_formula_when_true_position_is_the_diagonal() -> None:
    matrix = np.array([[10.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

    gaps = _per_anchor_gap(matrix, np.array([0, 1, 2]))

    assert gaps == pytest.approx([10.0 - 1.5, 4.0 - 4.0, 8.0 - 6.5])


def test_per_anchor_gap_batch_matches_a_naive_loop_over_permutations_bit_for_bit() -> None:
    rng = np.random.default_rng(11)
    n = 12
    matrix = rng.normal(0.4, 0.1, size=(n, n))
    true_positions_batch = rng.integers(0, n, size=(50, n))

    naive = np.stack([_per_anchor_gap(matrix, row) for row in true_positions_batch])
    batched = _per_anchor_gap_batch(matrix, true_positions_batch)

    assert np.array_equal(naive, batched)


def test_combine_by_stratum_batch_matches_a_naive_loop_bit_for_bit() -> None:
    rng = np.random.default_rng(12)
    n = 12
    gaps_batch = rng.normal(0.0, 0.2, size=(50, n))
    stratum = np.repeat(["low", "mid", "high"], 4)

    for weighted in (True, False):
        naive = np.array(
            [_combine_by_stratum(row, stratum, weighted=weighted) for row in gaps_batch]
        )
        batched = _combine_by_stratum_batch(gaps_batch, stratum, weighted=weighted)
        assert np.array_equal(naive, batched)


def test_stratified_mean_gap_test_detects_signal_beyond_confound() -> None:
    """Every anchor's diagonal (true) entry gets a +0.3 bonus over the shared off-diagonal level."""
    rng = np.random.default_rng(1)
    n = 60
    strata = np.repeat(["low", "mid", "high"], 20)
    matrix = rng.normal(0.4, 0.05, size=(n, n))
    np.fill_diagonal(matrix, rng.normal(0.4, 0.05, size=n) + 0.3)

    result = stratified_mean_gap_test(matrix, strata, n_permutations=2000, rng=rng)

    assert result.observed_gap == pytest.approx(0.3, abs=0.03)
    assert result.p_value < 0.01
    assert result.z_score > 5


def test_stratified_mean_gap_test_finds_no_signal_when_diagonal_matches_baseline() -> None:
    rng = np.random.default_rng(2)
    n = 40
    strata = np.array(["only"] * n)
    matrix = rng.normal(0.5, 0.1, size=(n, n))

    result = stratified_mean_gap_test(matrix, strata, n_permutations=2000, rng=rng)

    assert result.p_value > 0.05
    assert abs(result.z_score) < 3


def test_stratified_mean_gap_test_z_score_stays_graduated_past_the_p_value_floor() -> None:
    """A much stronger effect should read as a much larger z-score, even once p hits the floor."""
    rng = np.random.default_rng(9)
    n = 30
    strata = np.array(["only"] * n)

    weak_matrix = rng.normal(0.4, 0.05, size=(n, n))
    np.fill_diagonal(weak_matrix, rng.normal(0.4, 0.05, size=n) + 0.15)
    strong_matrix = rng.normal(0.4, 0.05, size=(n, n))
    np.fill_diagonal(strong_matrix, rng.normal(0.4, 0.05, size=n) + 0.6)

    weak = stratified_mean_gap_test(weak_matrix, strata, n_permutations=500, rng=rng)
    strong = stratified_mean_gap_test(strong_matrix, strata, n_permutations=500, rng=rng)

    assert weak.p_value == pytest.approx(strong.p_value)
    assert strong.z_score > weak.z_score * 1.5


def test_stratified_mean_gap_test_is_not_fooled_by_hub_anchors() -> None:
    """Some anchors are generically similar to everything (embedding-space anisotropy/"hubness")."""
    rng = np.random.default_rng(6)
    n = 30
    anchor_level = rng.uniform(0.0, 1.0, size=n)
    matrix = rng.normal(anchor_level[:, None], 0.02, size=(n, n))
    strata = np.array(["only"] * n)

    result = stratified_mean_gap_test(matrix, strata, n_permutations=2000, rng=rng)

    assert result.p_value > 0.05


def test_stratified_mean_gap_test_weighted_and_unweighted_differ_with_unequal_strata() -> None:
    n_small, n_large = 5, 45
    n = n_small + n_large
    strata = np.array(["small"] * n_small + ["large"] * n_large)
    matrix = np.full((n, n), 0.2)
    diag = np.full(n, 0.2)
    diag[:n_small] += 0.5
    np.fill_diagonal(matrix, diag)

    unweighted = stratified_mean_gap_test(
        matrix, strata, n_permutations=50, weighted=False, rng=np.random.default_rng(0)
    )
    weighted = stratified_mean_gap_test(
        matrix, strata, n_permutations=50, weighted=True, rng=np.random.default_rng(0)
    )

    assert unweighted.observed_gap == pytest.approx(0.25)
    assert weighted.observed_gap == pytest.approx(0.05)


def test_paired_cosine_similarity_rejects_a_zero_vector() -> None:
    """A zero vector has no direction, so a silent NaN would flow on into AP and AUC unnoticed."""
    source = np.array([[1.0, 0.0], [0.0, 0.0]])
    target = np.array([[1.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="zero vector"):
        paired_cosine_similarity(source, target)


def test_paired_cosine_similarity_rejects_a_zero_vector_on_the_target_side() -> None:
    source = np.array([[1.0, 0.0]])
    target = np.array([[0.0, 0.0]])

    with pytest.raises(ValueError, match="zero vector"):
        paired_cosine_similarity(source, target)


def test_stratified_mean_gap_test_rejects_a_single_anchor() -> None:
    """One anchor leaves no other column to average as the null, so the gap is undefined."""
    with pytest.raises(InsufficientDataError, match="at least 2"):
        stratified_mean_gap_test(
            np.array([[1.0]]), np.array(["a"]), n_permutations=10, rng=np.random.default_rng(0)
        )


class TestSparsePairedCosineSimilarity:
    def test_matches_the_dense_function_on_sparse_rows(self) -> None:
        import scipy.sparse as sp

        from library.retrieval_metrics import sparse_paired_cosine_similarity

        dense_a = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        dense_b = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

        result = sparse_paired_cosine_similarity(sp.csr_matrix(dense_a), sp.csr_matrix(dense_b))

        assert np.allclose(result, paired_cosine_similarity(dense_a, dense_b), rtol=0, atol=1e-9)

    def test_compares_only_row_i_against_row_i(self) -> None:
        import scipy.sparse as sp

        from library.retrieval_metrics import sparse_paired_cosine_similarity

        a = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
        b = sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0]]))

        result = sparse_paired_cosine_similarity(a, b)

        assert result.shape == (2,)
        assert np.allclose(result, [1.0, 0.0], atol=1e-9)

    def test_rejects_a_zero_row_rather_than_dividing_by_zero(self) -> None:
        import scipy.sparse as sp

        from library.retrieval_metrics import sparse_paired_cosine_similarity

        a = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
        b = sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0]]))

        with pytest.raises(DegenerateVectorError):
            sparse_paired_cosine_similarity(a, b)


def test_cosine_similarity_matrix_accumulates_in_float64_from_float32_input() -> None:
    """Float32 accumulation reorders exact ties, which AP and AUC then rank differently."""
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((64, 128)).astype(np.float32)

    result = cosine_similarity_matrix(vectors, vectors)

    exact = vectors.astype(np.float64)
    exact = exact / np.linalg.norm(exact, axis=1, keepdims=True)
    float32_result = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)) @ (
        vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ).T

    assert result.dtype == np.float64
    #: Matches the float64 reference far closer than float32 could, which is the point of the cast.
    assert np.allclose(result, exact @ exact.T, rtol=0, atol=1e-14)
    assert np.abs(result - float32_result).max() > 1e-9


def test_paired_cosine_similarity_accumulates_in_float64_from_float32_input() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((32, 96)).astype(np.float32)
    b = rng.standard_normal((32, 96)).astype(np.float32)

    result = paired_cosine_similarity(a, b)

    ea, eb = a.astype(np.float64), b.astype(np.float64)
    ea = ea / np.linalg.norm(ea, axis=1, keepdims=True)
    eb = eb / np.linalg.norm(eb, axis=1, keepdims=True)
    assert result.dtype == np.float64
    assert np.array_equal(result, np.sum(ea * eb, axis=1))


def test_sparse_cosine_similarity_matrix_accumulates_in_float64() -> None:
    rng = np.random.default_rng(2)
    dense = (rng.standard_normal((24, 64)) * (rng.random((24, 64)) < 0.3)).astype(np.float32)
    dense[dense.sum(axis=1) == 0, 0] = 1.0
    sparse = sp.csr_matrix(dense)

    result = sparse_cosine_similarity_matrix(sparse, sparse)

    exact = dense.astype(np.float64)
    exact = exact / np.linalg.norm(exact, axis=1, keepdims=True)
    assert result.dtype == np.float64
    assert np.allclose(result, exact @ exact.T, rtol=0, atol=1e-12)


def test_paired_cosine_similarity_blocks_without_changing_a_single_bit() -> None:
    """Rows are independent, so blocking must be exact, not merely close."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((5000, 64)).astype(np.float32)
    b = rng.standard_normal((5000, 64)).astype(np.float32)

    ea, eb = a.astype(np.float64), b.astype(np.float64)
    unblocked = np.sum(
        (ea / np.linalg.norm(ea, axis=1, keepdims=True))
        * (eb / np.linalg.norm(eb, axis=1, keepdims=True)),
        axis=1,
    )

    assert np.array_equal(paired_cosine_similarity(a, b), unblocked)


def test_paired_cosine_similarity_peak_memory_does_not_grow_with_the_input() -> None:
    """Casting the whole stack to float64 is what exhausted memory, so peak must track the block."""
    import tracemalloc

    def peak_for(rows: int) -> int:
        rng = np.random.default_rng(8)
        a = rng.standard_normal((rows, 256)).astype(np.float32)
        b = rng.standard_normal((rows, 256)).astype(np.float32)
        tracemalloc.start()
        paired_cosine_similarity(a, b)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    small, large = peak_for(5_000), peak_for(40_000)

    #: Eight times the rows must not cost eight times the memory: the block caps it.
    assert large < small * 2, f"peak grew from {small} to {large} with the input"


def test_discrimination_reports_no_effect_when_every_difference_is_zero() -> None:
    """Identical true and null similarities leave nothing to rank, so nothing is significant."""
    identical = np.array([0.5, 0.6, 0.7])

    result = paired_discrimination_test(identical, identical)

    assert result.p_value == 1.0
    assert result.statistic == 0.0
    assert result.rank_biserial == 0.0


class TestEveryCosineRoutineAccumulatesAtTheSamePrecision:
    """Embeddings are float32, so every routine must widen before accumulating, sparse included."""

    @staticmethod
    def _reference_paired(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """cos(a_i, b_i) from the definition, in float128, independent of the code under test."""
        wide_a = a.astype(np.longdouble)
        wide_b = b.astype(np.longdouble)
        norms = np.sqrt((wide_a * wide_a).sum(axis=1)) * np.sqrt((wide_b * wide_b).sum(axis=1))
        return ((wide_a * wide_b).sum(axis=1) / norms).astype(np.float64)

    @staticmethod
    def _vectors() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        a = rng.normal(size=(40, 64)).astype(np.float32)
        b = rng.normal(size=(40, 64)).astype(np.float32)
        return a, b

    def test_the_dense_paired_routine_matches_a_float128_reference(self) -> None:
        a, b = self._vectors()

        produced = paired_cosine_similarity(a, b)

        assert np.max(np.abs(produced - self._reference_paired(a, b))) < 1e-15

    def test_the_sparse_paired_routine_matches_a_float128_reference(self) -> None:
        """It read float32 rows without widening: a float32 answer beside a float64 one."""
        a, b = self._vectors()

        produced = sparse_paired_cosine_similarity(sp.csr_matrix(a), sp.csr_matrix(b))

        assert np.max(np.abs(produced - self._reference_paired(a, b))) < 1e-15

    def test_the_two_paired_routines_agree_with_each_other(self) -> None:
        """One statistic, so storing a model sparsely must not change the number it reports."""
        a, b = self._vectors()

        dense = paired_cosine_similarity(a, b)
        sparse = sparse_paired_cosine_similarity(sp.csr_matrix(a), sp.csr_matrix(b))

        assert np.max(np.abs(dense - sparse)) < 1e-15
