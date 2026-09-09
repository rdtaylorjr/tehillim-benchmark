import inspect

from library import protocol


def test_the_shared_counts_are_positive_and_named() -> None:
    assert protocol.DEFAULT_N_RESAMPLES > 0
    assert protocol.DEFAULT_N_PERMUTATIONS > 0
    assert protocol.DEFAULT_N_GROUP_PERMUTATIONS > 0
    assert 0 < protocol.DEFAULT_ALPHA < 1


def test_a_joint_group_test_uses_fewer_draws_than_a_single_statistic_test() -> None:
    """A maxT test rebuilds the null once per group, so it carries a smaller default."""
    assert protocol.DEFAULT_N_GROUP_PERMUTATIONS < protocol.DEFAULT_N_PERMUTATIONS


def test_no_entry_point_hard_codes_a_count_beside_the_shared_one() -> None:
    """Two copies of one z-score already drifted on ddof, so the counts are named in one place."""
    from genre.bootstrap import block_bootstrap_genre_ap_gap_and_auc
    from genre.permutation import joint_psalm_label_permutation_test
    from library.retrieval_metrics import stratified_mean_gap_test
    from trajectory.genre_breakdown import joint_genre_breakdown_permutation_test

    resamples = inspect.signature(block_bootstrap_genre_ap_gap_and_auc).parameters["n_resamples"]
    assert resamples.default == protocol.DEFAULT_N_RESAMPLES

    for fn in (joint_psalm_label_permutation_test, joint_genre_breakdown_permutation_test):
        assert (
            inspect.signature(fn).parameters["n_permutations"].default
            == protocol.DEFAULT_N_GROUP_PERMUTATIONS
        )

    assert (
        inspect.signature(stratified_mean_gap_test).parameters["n_permutations"].default
        == protocol.DEFAULT_N_PERMUTATIONS
    )
