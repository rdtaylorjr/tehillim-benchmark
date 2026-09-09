from parallelism.pairs import build_retrieval_pairs, filter_pairs_by_type, filter_pairs_with_vectors
from parallelism.tf_features import ReconstructedGroup


def _group(
    signature: str,
    member_ids: tuple[int, ...],
    member_indicators: tuple[str, ...],
    member_nodes: tuple[tuple[int, ...], ...],
    member_ambiguous: tuple[bool, ...] | None = None,
    group_range: str = "g",
    parallelism_type: str = "Synonymous",
) -> ReconstructedGroup:
    return ReconstructedGroup(
        group_range=group_range,
        parallelism_type=parallelism_type,
        signature=signature,
        member_ids=member_ids,
        member_indicators=member_indicators,
        member_nodes=member_nodes,
        member_ambiguous=member_ambiguous or tuple(False for _ in member_ids),
    )


def test_build_retrieval_pairs_pairs_a_simple_couplet() -> None:
    groups = [_group("AB", (0, 1), ("A", "B"), ((100,), (101,)))]

    pairs = build_retrieval_pairs(groups)

    assert len(pairs) == 1
    assert pairs[0].source_nodes == (100,)
    assert pairs[0].target_nodes == (101,)
    assert pairs[0].source_indicator == "A"
    assert pairs[0].target_indicator == "B"


def test_build_retrieval_pairs_does_not_cross_repeated_couplet_boundaries() -> None:
    """AB-AB-AB is three independent couplets; B1 must never pair with A2."""
    groups = [
        _group(
            "AB-AB-AB",
            (0, 1, 2, 3, 4, 5),
            ("A", "B", "A", "B", "A", "B"),
            ((1,), (2,), (3,), (4,), (5,), (6,)),
        )
    ]

    pairs = build_retrieval_pairs(groups)

    assert len(pairs) == 3
    assert [(p.source_nodes, p.target_nodes) for p in pairs] == [
        ((1,), (2,)),
        ((3,), (4,)),
        ((5,), (6,)),
    ]


def test_build_retrieval_pairs_uses_overlapping_adjacent_pairs_within_one_strophe() -> None:
    """A dashless three-slot segment is one connected unit: A-B and B-C both count."""
    groups = [_group("ABC", (0, 1, 2), ("A", "B", "C"), ((1,), (2,), (3,)))]

    pairs = build_retrieval_pairs(groups)

    assert [(p.source_nodes, p.target_nodes) for p in pairs] == [((1,), (2,)), ((2,), (3,))]


def test_build_retrieval_pairs_drops_a_lone_single_member_segment() -> None:
    """A-AB: the lone 'A' segment has no partner within itself and contributes no pair."""
    groups = [_group("A-AB", (0, 1, 2), ("A", "A", "B"), ((1,), (2,), (3,)))]

    pairs = build_retrieval_pairs(groups)

    assert len(pairs) == 1
    assert (pairs[0].source_nodes, pairs[0].target_nodes) == ((2,), (3,))


def test_build_retrieval_pairs_drops_a_pair_with_a_slot_missing_from_the_reconstructed_group() -> (
    None
):
    """A slot never resolved during alignment must drop its pair rather than half-build it."""
    groups = [_group("AB", (0,), ("A",), ((100,),))]

    pairs = build_retrieval_pairs(groups)

    assert pairs == []


def test_build_retrieval_pairs_drops_a_pair_flagged_ambiguous() -> None:
    groups = [
        _group(
            "AB",
            (0, 1),
            ("A", "B"),
            ((100,), (101,)),
            member_ambiguous=(False, True),
        )
    ]

    pairs = build_retrieval_pairs(groups)

    assert pairs == []


def test_build_retrieval_pairs_drops_a_pair_whose_members_share_one_node() -> None:
    """A resumptive member repeating within the same half-verse isn't a meaningful pair."""
    groups = [_group("AB", (0, 1), ("A", "B"), ((100,), (100,)))]

    pairs = build_retrieval_pairs(groups)

    assert pairs == []


def test_build_retrieval_pairs_matches_by_letter_identity_in_a_chiasm() -> None:
    """Ps 140:4, signature AB-BA: A='sharpen tongue', B='like a serpent', B='venom of an asp'."""
    groups = [
        _group(
            "AB-BA",
            (0, 1, 2, 3),
            ("A", "B", "B", "A"),
            ((10,), (20,), (30,), (40,)),
        )
    ]

    pairs = build_retrieval_pairs(groups)

    assert {(p.source_nodes, p.target_nodes) for p in pairs} == {((10,), (40,)), ((20,), (30,))}


def test_build_retrieval_pairs_matches_by_letter_identity_in_a_full_reversal_chiasm() -> None:
    """ABC-CBA: full reversal, three matched pairs A-A', B-B', C-C'."""
    groups = [
        _group(
            "ABC-CBA",
            (0, 1, 2, 3, 4, 5),
            ("A", "B", "C", "C", "B", "A"),
            ((1,), (2,), (3,), (4,), (5,), (6,)),
        )
    ]

    pairs = build_retrieval_pairs(groups)

    assert {(p.source_nodes, p.target_nodes) for p in pairs} == {
        ((1,), (6,)),
        ((2,), (5,)),
        ((3,), (4,)),
    }


def test_build_retrieval_pairs_matches_by_letter_identity_in_a_rotated_chiasm() -> None:
    """ABC-BCA: rotation, not reversal; letters still matched by identity, not position."""
    groups = [
        _group(
            "ABC-BCA",
            (0, 1, 2, 3, 4, 5),
            ("A", "B", "C", "B", "C", "A"),
            ((1,), (2,), (3,), (4,), (5,), (6,)),
        )
    ]

    pairs = build_retrieval_pairs(groups)

    assert {(p.source_nodes, p.target_nodes) for p in pairs} == {
        ((1,), (6,)),  # A - A
        ((2,), (4,)),  # B - B
        ((3,), (5,)),  # C - C
    }


def test_build_retrieval_pairs_does_not_treat_identical_repeats_as_a_chiasm() -> None:
    """AB-AB (identical order) stays two independent couplets, not a chiastic link."""
    groups = [
        _group(
            "AB-AB",
            (0, 1, 2, 3),
            ("A", "B", "A", "B"),
            ((1,), (2,), (3,), (4,)),
        )
    ]

    pairs = build_retrieval_pairs(groups)

    assert {(p.source_nodes, p.target_nodes) for p in pairs} == {((1,), (2,)), ((3,), (4,))}


def test_build_retrieval_pairs_assigns_stable_unique_pair_ids() -> None:
    groups = [
        _group("AB", (0, 1), ("A", "B"), ((1,), (2,)), group_range="g1"),
        _group("AB", (0, 1), ("A", "B"), ((3,), (4,)), group_range="g1"),
    ]

    pairs = build_retrieval_pairs(groups)

    assert len({p.pair_id for p in pairs}) == 2


def test_filter_pairs_by_type_keeps_only_matching_types() -> None:
    groups = [
        _group(
            "AB", (0, 1), ("A", "B"), ((1,), (2,)), group_range="g1", parallelism_type="Synonymous"
        ),
        _group(
            "AB", (0, 1), ("A", "B"), ((3,), (4,)), group_range="g2", parallelism_type="Antithetic"
        ),
        _group(
            "AB", (0, 1), ("A", "B"), ((5,), (6,)), group_range="g3", parallelism_type="Staircase"
        ),
    ]
    pairs = build_retrieval_pairs(groups)

    kept = filter_pairs_by_type(pairs, frozenset({"Synonymous", "Staircase"}))

    assert {p.group_range for p in kept} == {"g1", "g3"}


def test_filter_pairs_by_type_returns_empty_for_no_match() -> None:
    groups = [_group("AB", (0, 1), ("A", "B"), ((1,), (2,)), parallelism_type="Synthetic")]
    pairs = build_retrieval_pairs(groups)

    assert filter_pairs_by_type(pairs, frozenset({"Antithetic"})) == []


def test_filter_pairs_with_vectors_keeps_a_pair_whose_nodes_are_all_present() -> None:
    groups = [
        _group("AB", (0, 1), ("A", "B"), ((1,), (2,)), group_range="g1"),
    ]
    pairs = build_retrieval_pairs(groups)
    node_vectors = {1: object(), 2: object()}

    kept = filter_pairs_with_vectors(pairs, node_vectors)

    assert kept == pairs


def test_filter_pairs_with_vectors_drops_a_pair_missing_a_source_node() -> None:
    groups = [
        _group("AB", (0, 1), ("A", "B"), ((1,), (2,)), group_range="g1"),
    ]
    pairs = build_retrieval_pairs(groups)
    node_vectors = {2: object()}  # node 1 excluded, e.g. a zero-norm vector filtered upstream

    assert filter_pairs_with_vectors(pairs, node_vectors) == []


def test_filter_pairs_with_vectors_drops_a_pair_missing_a_target_node() -> None:
    groups = [
        _group("AB", (0, 1), ("A", "B"), ((1,), (2,)), group_range="g1"),
    ]
    pairs = build_retrieval_pairs(groups)
    node_vectors = {1: object()}  # node 2 excluded

    assert filter_pairs_with_vectors(pairs, node_vectors) == []


def test_filter_pairs_with_vectors_keeps_only_the_pairs_with_full_coverage() -> None:
    groups = [
        _group("AB", (0, 1), ("A", "B"), ((1,), (2,)), group_range="g1"),
        _group("AB", (0, 1), ("A", "B"), ((3,), (4,)), group_range="g2"),
    ]
    pairs = build_retrieval_pairs(groups)
    node_vectors = {1: object(), 2: object(), 3: object()}  # node 4 excluded, drops g2's pair

    kept = filter_pairs_with_vectors(pairs, node_vectors)

    assert {p.group_range for p in kept} == {"g1"}
