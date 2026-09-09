from genre.pairs import GenrePair, build_genre_pairs, filter_pairs_by_genre


def test_builds_every_unique_unordered_pair() -> None:
    genre_by_psalm = {1: "Lament", 2: "Lament", 3: "Praise"}

    pairs = build_genre_pairs(genre_by_psalm)

    psalm_pairs = {(p.psalm_a, p.psalm_b) for p in pairs}
    assert psalm_pairs == {(1, 2), (1, 3), (2, 3)}


def test_marks_same_genre_pairs_true_and_different_genre_pairs_false() -> None:
    genre_by_psalm = {1: "Lament", 2: "Lament", 3: "Praise"}

    pairs = build_genre_pairs(genre_by_psalm)

    by_pair = {(p.psalm_a, p.psalm_b): p for p in pairs}
    assert by_pair[(1, 2)] == GenrePair(1, 2, "Lament", "Lament", same_genre=True)
    assert by_pair[(1, 3)] == GenrePair(1, 3, "Lament", "Praise", same_genre=False)
    assert by_pair[(2, 3)] == GenrePair(2, 3, "Lament", "Praise", same_genre=False)


def test_orders_each_pair_by_ascending_psalm_number() -> None:
    genre_by_psalm = {5: "Wisdom", 2: "Wisdom"}

    pairs = build_genre_pairs(genre_by_psalm)

    assert pairs == [GenrePair(2, 5, "Wisdom", "Wisdom", same_genre=True)]


def test_produces_no_pairs_for_a_single_psalm() -> None:
    assert build_genre_pairs({1: "Lament"}) == []


def test_produces_the_correct_total_pair_count_for_n_psalms() -> None:
    genre_by_psalm = dict.fromkeys(range(10), "Lament")

    pairs = build_genre_pairs(genre_by_psalm)

    assert len(pairs) == 10 * 9 // 2


def test_filter_pairs_by_genre_keeps_pairs_touching_the_genre_on_either_side() -> None:
    genre_by_psalm = {1: "Lament", 2: "Lament", 3: "Praise", 4: "Hymn"}
    pairs = build_genre_pairs(genre_by_psalm)

    filtered = filter_pairs_by_genre(pairs, "Lament")

    psalm_pairs = {(p.psalm_a, p.psalm_b) for p in filtered}
    assert psalm_pairs == {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)}


def test_filter_pairs_by_genre_excludes_pairs_between_two_other_genres() -> None:
    genre_by_psalm = {1: "Lament", 2: "Praise", 3: "Hymn"}
    pairs = build_genre_pairs(genre_by_psalm)

    filtered = filter_pairs_by_genre(pairs, "Lament")

    psalm_pairs = {(p.psalm_a, p.psalm_b) for p in filtered}
    assert (2, 3) not in psalm_pairs


def test_filter_pairs_by_genre_preserves_the_same_genre_flag() -> None:
    genre_by_psalm = {1: "Lament", 2: "Lament", 3: "Praise"}
    pairs = build_genre_pairs(genre_by_psalm)

    filtered = filter_pairs_by_genre(pairs, "Lament")

    same_flags = {(p.psalm_a, p.psalm_b): p.same_genre for p in filtered}
    assert same_flags[(1, 2)] is True
    assert same_flags[(1, 3)] is False
