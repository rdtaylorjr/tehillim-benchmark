from parallelism.pairs import RetrievalPair, build_retrieval_pairs, slot_pairs_for_signature
from parallelism.tf_features import ReconstructedGroup


def test_shared_letters_pair_across_lines() -> None:
    """`AB-AB`: A pairs with the second A, B with the second B, never A with B."""
    assert slot_pairs_for_signature("AB-AB") == [((0,), (2,)), ((1,), (3,))]


def test_chiasm_pairs_by_letter_not_position() -> None:
    """`AB-BA`: the reordered letters still pair by identity."""
    assert slot_pairs_for_signature("AB-BA") == [((0,), (3,)), ((1,), (2,))]


def test_partial_overlap_pairs_only_the_shared_letter() -> None:
    """`AB-BC`: B pairs with B, A and C have no stated partner."""
    assert slot_pairs_for_signature("AB-BC") == [((1,), (2,))]


def test_repeated_letter_inside_one_line_pairs_within_it() -> None:
    """`A-BB`: the two B members correspond to each other."""
    assert slot_pairs_for_signature("A-BB") == [((1,), (2,))]


def test_three_synonymous_lines_pair_every_two() -> None:
    """`AAA`: three members sharing a letter give three pairs."""
    assert slot_pairs_for_signature("AAA") == [((0,), (1,)), ((0,), (2,)), ((1,), (2,))]


def test_two_lines_without_shared_letters_pair_as_whole_lines() -> None:
    """`A-B`: the plain bicolon pairs line one with line two."""
    assert slot_pairs_for_signature("A-B") == [((0,), (1,))]


def test_multi_member_lines_without_shared_letters_pair_as_unions() -> None:
    """`AB-CD`: each line is one unit, so the pair is the union of each side's members."""
    assert slot_pairs_for_signature("AB-CD") == [((0, 1), (2, 3))]


def test_three_lines_without_shared_letters_pair_adjacent_lines() -> None:
    """`A-B-C`: adjacent lines pair, non-adjacent lines do not."""
    assert slot_pairs_for_signature("A-B-C") == [((0,), (1,)), ((1,), (2,))]


def test_single_line_with_distinct_letters_pairs_adjacent_members() -> None:
    """`ABC` without a dash: members are lines, so adjacent members pair."""
    assert slot_pairs_for_signature("ABC") == [((0,), (1,)), ((1,), (2,))]


def test_single_member_pairs_nothing() -> None:
    """`A` has nothing to pair with."""
    assert slot_pairs_for_signature("A") == []


def _group(
    signature: str, nodes: tuple[tuple[int, ...], ...], ambiguous: tuple[bool, ...] | None = None
) -> ReconstructedGroup:
    """A group whose members are lettered by the signature and sit on the given nodes."""
    letters = signature.replace("-", "")
    return ReconstructedGroup(
        group_range="r",
        parallelism_type="Synonymous",
        signature=signature,
        member_ids=tuple(range(len(letters))),
        member_indicators=tuple(letters),
        member_nodes=nodes,
        member_ambiguous=ambiguous or tuple(False for _ in letters),
    )


def test_pairs_resolving_to_one_node_set_collapse_to_one_pair() -> None:
    """`AB-AB` on two half-verses yields one pair, since both letters resolve to the same nodes."""
    pairs = build_retrieval_pairs([_group("AB-AB", ((10,), (10,), (11,), (11,)))])
    assert [(p.source_nodes, p.target_nodes) for p in pairs] == [((10,), (11,))]


def test_whole_line_pair_unions_member_nodes_in_order() -> None:
    """`AB-CD` unions each side's nodes without repeating a node."""
    pairs = build_retrieval_pairs([_group("AB-CD", ((10,), (10, 11), (12,), (13,)))])
    assert [(p.source_nodes, p.target_nodes) for p in pairs] == [((10, 11), (12, 13))]


def test_pair_on_one_node_set_is_dropped() -> None:
    """A member pair whose sides resolve to the same nodes cannot be scored."""
    assert build_retrieval_pairs([_group("A-B", ((10,), (10,)))]) == []


def test_ambiguous_member_drops_its_pairs_only() -> None:
    """An ambiguous member removes the pairs it is in and leaves the rest."""
    pairs = build_retrieval_pairs(
        [_group("AAA", ((10,), (11,), (12,)), ambiguous=(False, False, True))]
    )
    assert [(p.source_nodes, p.target_nodes) for p in pairs] == [((10,), (11,))]


def test_missing_slot_drops_its_pairs_only() -> None:
    """A slot absent from the reconstruction produces no pair and no error."""
    group = ReconstructedGroup(
        group_range="r",
        parallelism_type="Synthetic",
        signature="A-B-C",
        member_ids=(0, 2),
        member_indicators=("A", "C"),
        member_nodes=((10,), (12,)),
        member_ambiguous=(False, False),
    )
    assert build_retrieval_pairs([group]) == []


def test_pair_records_indicators_and_identity() -> None:
    """The pair carries the joined member letters of each side and a stable id."""
    (pair,) = build_retrieval_pairs([_group("AB-CD", ((10,), (11,), (12,), (13,)))])
    assert isinstance(pair, RetrievalPair)
    assert (pair.source_indicator, pair.target_indicator) == ("AB", "CD")
    assert pair.pair_id == "r:0:0+1-2+3"
