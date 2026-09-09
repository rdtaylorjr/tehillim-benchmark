from parallelism.baseline import build_unmarked_half_verse_pairs


def test_build_unmarked_half_verse_pairs_pairs_adjacent_unmarked_nodes() -> None:
    half_verses_by_psalm = {1: [10, 11, 12, 13]}

    pairs = build_unmarked_half_verse_pairs(half_verses_by_psalm, marked_nodes=set())

    assert pairs == [(10, 11), (11, 12), (12, 13)]


def test_build_unmarked_half_verse_pairs_excludes_a_pair_touching_a_marked_node() -> None:
    """A pair is dropped if EITHER side was ever a source or target of a true parallel pair."""
    half_verses_by_psalm = {1: [10, 11, 12, 13]}

    pairs = build_unmarked_half_verse_pairs(half_verses_by_psalm, marked_nodes={11})

    assert pairs == [(12, 13)]  # (10,11) and (11,12) both touch node 11


def test_build_unmarked_half_verse_pairs_never_crosses_a_psalm_boundary() -> None:
    half_verses_by_psalm = {1: [10, 11], 2: [20, 21]}

    pairs = build_unmarked_half_verse_pairs(half_verses_by_psalm, marked_nodes=set())

    assert (11, 20) not in pairs
    assert pairs == [(10, 11), (20, 21)]


def test_build_unmarked_half_verse_pairs_handles_a_single_half_verse_psalm() -> None:
    half_verses_by_psalm = {1: [10]}

    pairs = build_unmarked_half_verse_pairs(half_verses_by_psalm, marked_nodes=set())

    assert pairs == []
