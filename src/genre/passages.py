"""The passages a genre benchmark compares, each at the word extent its source assigned it."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from genre.genre_labels import load_genre_by_psalm
from genre.taxonomies import units_of
from library.bhsa import list_psalms_words_by_half_verse, list_psalms_words_by_verse


@dataclass(frozen=True, slots=True)
class Passage:
    """One assignment: the word nodes of its extent, the one Gattung it bears, and its name."""

    id: str
    psalm: int
    gattung: str
    words: tuple[int, ...]
    label: str


@dataclass(frozen=True, slots=True)
class Bound:
    """One end of a word run: the verse, the position of the word, and the Halbzeile letter."""

    verse: int
    word: int
    halbzeile: str


def _bound_label(bound: Bound, verse_length: int, *, opening: bool) -> str:
    """The verse alone at a verse boundary, else the Halbzeile letter or the word position."""
    at_edge = bound.word == 1 if opening else bound.word == verse_length
    if at_edge:
        return str(bound.verse)
    return f"{bound.verse}{bound.halbzeile}" if bound.halbzeile else f"{bound.verse}.{bound.word}"


def passage_label(psalm: int, first: Bound, last: Bound, verse_lengths: Mapping[int, int]) -> str:
    """Ps 9 for a whole psalm, else Ps 9:1-5 with any partial bound marked by letter or word."""
    verses = sorted(verse_lengths)
    whole = (
        first.verse == verses[0]
        and first.word == 1
        and last.verse == verses[-1]
        and last.word == verse_lengths[last.verse]
    )
    if whole:
        return f"Ps {psalm}"
    start = _bound_label(first, verse_lengths[first.verse], opening=True)
    end = _bound_label(last, verse_lengths[last.verse], opening=False)
    #: A run inside one verse reaching its end is named by where it starts, as Gunkel does.
    if first.verse == last.verse and end == str(last.verse):
        return f"Ps {psalm}:{start}"
    return f"Ps {psalm}:{start}-{end}"


def word_run(
    psalm: int, first: Bound, last: Bound, words_by_verse: Mapping[tuple[int, int], list[int]]
) -> tuple[int, ...]:
    """The word nodes from the first bound to the last, inclusive, in text order."""
    run: list[int] = []
    for verse in range(first.verse, last.verse + 1):
        words = words_by_verse[(psalm, verse)]
        start = first.word - 1 if verse == first.verse else 0
        stop = last.word if verse == last.verse else len(words)
        if start >= len(words) or stop > len(words):
            raise ValueError(f"Ps {psalm}:{verse} has {len(words)} words, bound outside it")
        run.extend(words[start:stop])
    return tuple(run)


def _verse_lengths(
    words_by_verse: Mapping[tuple[int, int], list[int]],
) -> dict[int, dict[int, int]]:
    """How many words each verse holds, by psalm, in verse order."""
    lengths: dict[int, dict[int, int]] = {}
    for (psalm, verse), words in sorted(words_by_verse.items()):
        lengths.setdefault(psalm, {})[verse] = len(words)
    return lengths


def logos_passages(
    genre_by_psalm: Mapping[int, str], words_by_verse: Mapping[tuple[int, int], list[int]]
) -> list[Passage]:
    """Logos classifies whole psalms, so each labelled psalm is one passage over all its words."""
    lengths = _verse_lengths(words_by_verse)
    return [
        Passage(
            str(psalm),
            psalm,
            genre,
            tuple(node for verse in lengths[psalm] for node in words_by_verse[(psalm, verse)]),
            f"Ps {psalm}",
        )
        for psalm, genre in sorted(genre_by_psalm.items())
    ]


def gunkel_passages(
    path: Path,
    units: tuple[str, ...],
    words_by_verse: Mapping[tuple[int, int], list[int]],
) -> list[Passage]:
    """Every `gunkel.csv` assignment at the given units, at the word extent the table records."""
    lengths = _verse_lengths(words_by_verse)
    passages: list[Passage] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["unit_en"] not in units:
                continue
            psalm = int(row["psalm"])
            first = Bound(int(row["first_verse"]), int(row["first_word"]), row["first_halbzeile"])
            last = Bound(int(row["last_verse"]), int(row["last_word"]), row["last_halbzeile"])
            passages.append(
                Passage(
                    f"{psalm}:{first.verse}.{first.word}-{last.verse}.{last.word}:{row['gattung']}",
                    psalm,
                    row["gattung"],
                    word_run(psalm, first, last, words_by_verse),
                    passage_label(psalm, first, last, lengths[psalm]),
                )
            )
    return passages


def load_passages(
    taxonomy: str,
    unit: str | None,
    path: Path,
    api: Any,
    *,
    words_by_verse: Callable[[Any], dict[tuple[int, int], list[int]]] = list_psalms_words_by_verse,
) -> list[Passage]:
    """The passages of one taxonomy, at one register where it has them, against the corpus."""
    units = units_of(taxonomy, unit)
    if not units:
        return logos_passages(load_genre_by_psalm(path), words_by_verse(api))
    return gunkel_passages(path, units, words_by_verse(api))


def half_verse_weights(
    passages: list[Passage], words_by_half_verse: Mapping[int, list[int]]
) -> dict[str, dict[int, float]]:
    """Each passage's half-verses by id, weighted by the share of their words inside the passage."""
    half_verse_of = {
        word: half_verse for half_verse, words in words_by_half_verse.items() for word in words
    }
    weights: dict[str, dict[int, float]] = {}
    for passage in passages:
        covered: dict[int, int] = {}
        for word in passage.words:
            half_verse = half_verse_of[word]
            covered[half_verse] = covered.get(half_verse, 0) + 1
        weights[passage.id] = {
            half_verse: count / len(words_by_half_verse[half_verse])
            for half_verse, count in covered.items()
        }
    return weights


def load_half_verse_weights(
    passages: list[Passage],
    api: Any,
    *,
    words_by_half_verse: Callable[[Any], dict[int, list[int]]] = list_psalms_words_by_half_verse,
) -> dict[str, dict[int, float]]:
    """half_verse_weights against the corpus, the shape every vector loader pools over."""
    return half_verse_weights(passages, words_by_half_verse(api))


def admissible_mask(passages: list[Passage]) -> np.ndarray:
    """Which pairs may be compared: those sharing no word, never a passage with itself."""
    words = sorted({word for passage in passages for word in passage.words})
    column_of = {word: column for column, word in enumerate(words)}
    rows = np.fromiter(
        (i for i, passage in enumerate(passages) for _ in passage.words), dtype=np.intp
    )
    columns = np.fromiter(
        (column_of[word] for passage in passages for word in passage.words), dtype=np.intp
    )
    incidence = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, columns)), shape=(len(passages), len(words))
    )
    shared: np.ndarray = (incidence @ incidence.T).toarray()
    return np.asarray(shared == 0)


def cluster_codes(passages: list[Passage]) -> np.ndarray:
    """Each passage's psalm as a code, the unit the bootstrap resamples."""
    psalms = sorted({passage.psalm for passage in passages})
    code_of = {psalm: code for code, psalm in enumerate(psalms)}
    return np.array([code_of[passage.psalm] for passage in passages], dtype=np.intp)
