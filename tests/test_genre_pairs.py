from genre.pairs import GenrePair, build_genre_pairs, filter_pairs_by_genre
from genre.passages import Passage

LAMENT_1 = Passage("1", 1, "Lament", (100, 101), "1")
LAMENT_2 = Passage("2", 2, "Lament", (200,), "2")
PRAISE_3 = Passage("3", 3, "Praise", (300,), "3")
HYMN_4 = Passage("4", 4, "Hymn", (400,), "4")


def test_builds_every_unordered_pair_of_disjoint_passages() -> None:
    pairs = build_genre_pairs([LAMENT_1, LAMENT_2, PRAISE_3])
    assert {(p.item_a, p.item_b) for p in pairs} == {("1", "2"), ("1", "3"), ("2", "3")}


def test_marks_same_genre_pairs_true_and_different_genre_pairs_false() -> None:
    by_pair = {(p.item_a, p.item_b): p for p in build_genre_pairs([LAMENT_1, LAMENT_2, PRAISE_3])}
    assert by_pair[("1", "2")] == GenrePair("1", "2", 1, 2, "Lament", "Lament", same_genre=True)
    assert by_pair[("1", "3")] == GenrePair("1", "3", 1, 3, "Lament", "Praise", same_genre=False)


def test_a_passage_is_never_paired_with_text_it_shares() -> None:
    """A motif inside its host song, or two labels on one whole psalm, compare text with itself."""
    host = Passage(
        "9:1-21:Mischgedicht", 9, "Mischgedicht", (900, 901, 902, 903), "9:1-21:Mischgedicht"
    )
    motif = Passage("9:2-2:Königspsalm", 9, "Königspsalm", (901,), "9:2-2:Königspsalm")
    other = Passage("9:4-4:Fluch", 9, "Fluch", (903,), "9:4-4:Fluch")
    pairs = build_genre_pairs([host, motif, other, LAMENT_1])
    keys = {(p.item_a, p.item_b) for p in pairs}
    assert ("9:1-21:Mischgedicht", "9:2-2:Königspsalm") not in keys
    assert ("9:1-21:Mischgedicht", "9:4-4:Fluch") not in keys
    assert ("9:2-2:Königspsalm", "9:4-4:Fluch") in keys
    assert ("9:1-21:Mischgedicht", "1") in keys


def test_two_disjoint_passages_of_one_psalm_are_compared() -> None:
    """Gunkel's own analysis puts several Gattungen in one psalm, so their pieces are compared."""
    danklied = Passage("9:1-5:Danklied", 9, "Danklied", (900, 901), "9:1-5:Danklied")
    hymnus = Passage("9:6-13:Hymnus", 9, "Hymnus", (902, 903), "9:6-13:Hymnus")
    pairs = build_genre_pairs([danklied, hymnus])
    assert pairs == [
        GenrePair("9:1-5:Danklied", "9:6-13:Hymnus", 9, 9, "Danklied", "Hymnus", same_genre=False)
    ]


def test_orders_each_pair_as_the_passages_were_given() -> None:
    pairs = build_genre_pairs([LAMENT_2, LAMENT_1])
    assert pairs == [GenrePair("2", "1", 2, 1, "Lament", "Lament", same_genre=True)]


def test_produces_no_pairs_for_a_single_passage() -> None:
    assert build_genre_pairs([LAMENT_1]) == []


def test_produces_the_correct_total_pair_count_for_n_disjoint_passages() -> None:
    passages = [Passage(str(i), i, "Lament", (i * 10,), f"Ps {i}") for i in range(10)]
    assert len(build_genre_pairs(passages)) == 10 * 9 // 2


def test_filter_pairs_by_genre_keeps_pairs_touching_the_genre_on_either_side() -> None:
    pairs = build_genre_pairs([LAMENT_1, LAMENT_2, PRAISE_3, HYMN_4])
    filtered = filter_pairs_by_genre(pairs, "Lament")
    assert {(p.item_a, p.item_b) for p in filtered} == {
        ("1", "2"),
        ("1", "3"),
        ("1", "4"),
        ("2", "3"),
        ("2", "4"),
    }


def test_filter_pairs_by_genre_excludes_pairs_between_two_other_genres() -> None:
    pairs = build_genre_pairs([LAMENT_1, PRAISE_3, HYMN_4])
    assert ("3", "4") not in {(p.item_a, p.item_b) for p in filter_pairs_by_genre(pairs, "Lament")}


def test_filter_pairs_by_genre_preserves_the_same_genre_flag() -> None:
    pairs = build_genre_pairs([LAMENT_1, LAMENT_2, PRAISE_3])
    same = {(p.item_a, p.item_b): p.same_genre for p in filter_pairs_by_genre(pairs, "Lament")}
    assert same[("1", "2")] is True
    assert same[("1", "3")] is False
