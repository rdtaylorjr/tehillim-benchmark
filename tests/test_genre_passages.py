"""Passages: the items a genre benchmark compares, at the word extent the source assigned them."""

from pathlib import Path

import numpy as np
import pytest

from genre.passages import (
    Bound,
    Passage,
    admissible_mask,
    cluster_codes,
    gunkel_passages,
    half_verse_weights,
    load_passages,
    logos_passages,
    passage_label,
    word_run,
)

#: Psalm 1 has two verses of two half-verses each, psalm 2 two verses of one; three words a half.
WORDS_BY_HALF_VERSE = {
    100: [1001, 1002, 1003],
    101: [1011, 1012, 1013],
    102: [1021, 1022, 1023],
    103: [1031, 1032, 1033],
    200: [2001, 2002, 2003],
    201: [2011, 2012, 2013],
}
WORDS_BY_VERSE = {
    (1, 1): [1001, 1002, 1003, 1011, 1012, 1013],
    (1, 2): [1021, 1022, 1023, 1031, 1032, 1033],
    (2, 1): [2001, 2002, 2003],
    (2, 2): [2011, 2012, 2013],
}

GUNKEL_COLUMNS = (
    "1933,1926,psalm,first_verse,first_word,first_halbzeile,last_verse,last_word,last_halbzeile,"
    "gattung,gattung_en,unit,unit_en,Gunkel,Begrich"
)
GUNKEL_HEADER = f"{GUNKEL_COLUMNS},citations"
GUNKEL_ROWS = [
    "1933,1926,1,1,1,,2,6,,Hymnus,Hymn,Lied,Song,Gunkel,,p. 1",
    "1933,,1,2,4,b,2,6,,Fluch,Curse,Motiv,Motif,,Begrich,p. 2",
    "1933,1926,2,1,1,,1,3,,Klagelied des Einzelnen,Individual Lament,Stück,Component,Gunkel,,p. 3",
    "1933,1926,2,2,1,,2,3,,Hymnus,Hymn,Lied,Song,Gunkel,,p. 4",
]


def _gunkel_csv(tmp_path: Path) -> Path:
    path = tmp_path / "gunkel.csv"
    path.write_text("\n".join([GUNKEL_HEADER, *GUNKEL_ROWS]) + "\n")
    return path


def _logos_csv(tmp_path: Path) -> Path:
    path = tmp_path / "psalms-browser.csv"
    path.write_text(
        "Psalm,Attribution,Genre,Structure,Tag,SupplementalDataId\n"
        '"Ps 1","David","Lament","","",1\n'
        '"Ps 2","David","Royal","","",2\n'
    )
    return path


class TestLogosPassages:
    def test_one_whole_psalm_passage_per_labelled_psalm(self) -> None:
        passages = logos_passages({1: "Lament", 2: "Royal"}, WORDS_BY_VERSE)
        assert passages == [
            Passage("1", 1, "Lament", tuple(WORDS_BY_VERSE[1, 1] + WORDS_BY_VERSE[1, 2]), "Ps 1"),
            Passage("2", 2, "Royal", tuple(WORDS_BY_VERSE[2, 1] + WORDS_BY_VERSE[2, 2]), "Ps 2"),
        ]

    def test_a_psalm_the_corpus_lacks_is_refused_rather_than_silently_empty(self) -> None:
        with pytest.raises(KeyError, match="3"):
            logos_passages({3: "Lament"}, WORDS_BY_VERSE)


class TestWordRun:
    def test_runs_from_the_first_bound_to_the_last_across_verses(self) -> None:
        run = word_run(1, Bound(1, 5, ""), Bound(2, 2, ""), WORDS_BY_VERSE)
        assert run == (1012, 1013, 1021, 1022)

    def test_a_bound_past_the_end_of_its_verse_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Ps 2:1 has 3 words"):
            word_run(2, Bound(1, 1, ""), Bound(1, 4, ""), WORDS_BY_VERSE)


LENGTHS = {1: 6, 2: 6}


class TestPassageLabel:
    def test_a_whole_psalm_is_named_by_its_number(self) -> None:
        assert passage_label(1, Bound(1, 1, ""), Bound(2, 6, ""), LENGTHS) == "Ps 1"

    def test_whole_verses_are_named_by_their_run(self) -> None:
        assert passage_label(1, Bound(1, 1, ""), Bound(1, 6, ""), LENGTHS) == "Ps 1:1"
        assert passage_label(9, Bound(1, 1, ""), Bound(2, 6, ""), {1: 6, 2: 6, 3: 4}) == "Ps 9:1-2"

    def test_a_partial_verse_carries_the_halbzeile_letter_or_the_word_position(self) -> None:
        assert passage_label(1, Bound(1, 4, "b"), Bound(2, 6, ""), LENGTHS) == "Ps 1:1b-2"
        assert passage_label(1, Bound(1, 1, ""), Bound(2, 3, ""), LENGTHS) == "Ps 1:1-2.3"
        assert passage_label(1, Bound(2, 2, ""), Bound(2, 5, "b"), LENGTHS) == "Ps 1:2.2-2b"


class TestGunkelPassages:
    def test_a_register_keeps_only_the_units_it_names(self, tmp_path: Path) -> None:
        songs = gunkel_passages(_gunkel_csv(tmp_path), ("Song",), WORDS_BY_VERSE)
        assert [p.id for p in songs] == ["1:1.1-2.6:Hymnus", "2:2.1-2.3:Hymnus"]
        wider = gunkel_passages(
            _gunkel_csv(tmp_path), ("Song", "Component", "Motif"), WORDS_BY_VERSE
        )
        assert len(wider) == 4

    def test_a_passage_holds_the_words_of_its_extent_and_a_label(self, tmp_path: Path) -> None:
        passages = gunkel_passages(_gunkel_csv(tmp_path), ("Song", "Motif"), WORDS_BY_VERSE)
        by_id = {p.id: p for p in passages}
        assert by_id["1:1.1-2.6:Hymnus"].words == tuple(WORDS_BY_VERSE[1, 1] + WORDS_BY_VERSE[1, 2])
        assert by_id["1:1.1-2.6:Hymnus"].label == "Ps 1"
        assert by_id["1:2.4-2.6:Fluch"].words == (1031, 1032, 1033)
        assert by_id["1:2.4-2.6:Fluch"].label == "Ps 1:2b"

    def test_a_verse_the_corpus_lacks_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "gunkel.csv"
        path.write_text(GUNKEL_HEADER + "\n1933,,1,1,1,,9,3,,Hymnus,Hymn,Lied,Song,Gunkel,,p. 1\n")
        with pytest.raises(KeyError, match=r"\(1, 3\)"):
            gunkel_passages(path, ("Song",), WORDS_BY_VERSE)


class TestLoadPassages:
    def test_dispatches_on_the_taxonomy_and_reads_the_corpus_once(self, tmp_path: Path) -> None:
        class _Api:
            pass

        def words_by_verse(api: object) -> dict[tuple[int, int], list[int]]:
            return WORDS_BY_VERSE

        logos = load_passages(
            "logos", None, _logos_csv(tmp_path), _Api(), words_by_verse=words_by_verse
        )
        gunkel = load_passages(
            "gunkel", "song_component", _gunkel_csv(tmp_path), _Api(), words_by_verse=words_by_verse
        )
        assert [p.gattung for p in logos] == ["Lament", "Royal"]
        assert [p.id for p in gunkel] == [
            "1:1.1-2.6:Hymnus",
            "2:1.1-1.3:Klagelied des Einzelnen",
            "2:2.1-2.3:Hymnus",
        ]


PASSAGES = [
    Passage(
        "1:1.1-2.6:Hymnus", 1, "Hymnus", tuple(WORDS_BY_VERSE[1, 1] + WORDS_BY_VERSE[1, 2]), "Ps 1"
    ),
    Passage("1:2.4-2.6:Fluch", 1, "Fluch", (1031, 1032, 1033), "Ps 1:2b"),
    Passage(
        "2:1.1-1.3:Klagelied des Einzelnen",
        2,
        "Klagelied des Einzelnen",
        (2001, 2002, 2003),
        "Ps 2:1",
    ),
    Passage("2:2.1-2.3:Hymnus", 2, "Hymnus", (2011, 2012, 2013), "Ps 2:2"),
]


class TestHalfVerseWeights:
    def test_a_whole_passage_weights_every_half_verse_at_one(self) -> None:
        weights = half_verse_weights(PASSAGES[:1], WORDS_BY_HALF_VERSE)
        assert weights == {"1:1.1-2.6:Hymnus": {100: 1.0, 101: 1.0, 102: 1.0, 103: 1.0}}

    def test_a_partial_half_verse_carries_the_share_of_its_words_inside_the_passage(self) -> None:
        partial = Passage(
            "1:1.2-2.1:x", 1, "x", (1002, 1003, 1011, 1012, 1013, 1021), "Ps 1:1.2-2.1"
        )
        weights = half_verse_weights([partial], WORDS_BY_HALF_VERSE)
        assert weights["1:1.2-2.1:x"] == pytest.approx({100: 2 / 3, 101: 1.0, 102: 1 / 3})


class TestPassageStructure:
    def test_admissible_pairs_share_no_word_and_exclude_the_diagonal(self) -> None:
        """A motif is not compared with its host, but two disjoint passages of one psalm are."""
        mask = admissible_mask(PASSAGES)
        expected = np.array(
            [
                [False, False, True, True],
                [False, False, True, True],
                [True, True, False, True],
                [True, True, True, False],
            ]
        )
        np.testing.assert_array_equal(mask, expected)

    def test_two_passages_splitting_one_half_verse_at_a_word_are_admissible(self) -> None:
        left = Passage("a", 1, "x", (1001, 1002), "Ps 1:1.1-1.2")
        right = Passage("b", 1, "y", (1003, 1011), "Ps 1:1.3-1.4")
        assert admissible_mask([left, right])[0, 1]

    def test_cluster_codes_group_passages_by_psalm_in_psalm_order(self) -> None:
        np.testing.assert_array_equal(cluster_codes(PASSAGES), np.array([0, 0, 1, 1]))
