def test_a_chiasm_letter_repeated_unevenly_pairs_only_what_it_can() -> None:
    """Letters pair in order across a chiastic link, so a surplus occurrence finds no partner."""
    from parallelism.pairs import _positions_for_signature

    balanced = _positions_for_signature("ab-ba")
    lopsided = _positions_for_signature("aab-baa")

    assert len(balanced) >= 2
    assert len(lopsided) >= 2


def test_only_segments_with_the_same_letters_are_ever_linked() -> None:
    """Position pairing relies on this, so it is asserted rather than guarded against at runtime."""
    import collections
    import itertools

    from parallelism.pairs import _chiasm_links

    for first in itertools.product("ab", repeat=3):
        for second in itertools.product("ab", repeat=3):
            segments = ["".join(first), "".join(second)]
            for j, i in _chiasm_links(segments).items():
                assert collections.Counter(segments[i]) == collections.Counter(segments[j])
