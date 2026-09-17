"""Builds every admissible unordered passage pair, labeled by whether the two share a genre."""

from dataclasses import dataclass

import numpy as np

from genre.passages import Passage, admissible_mask


@dataclass(frozen=True, slots=True)
class GenrePair:
    """Two passages compared for genre discrimination, and whether they share a genre."""

    item_a: str
    item_b: str
    psalm_a: int
    psalm_b: int
    genre_a: str
    genre_b: str
    same_genre: bool


def build_genre_pairs(passages: list[Passage]) -> list[GenrePair]:
    """One GenrePair per unordered pair of passages sharing no half-verse, in the given order."""
    rows, columns = np.nonzero(np.triu(admissible_mask(passages), k=1))
    return [
        GenrePair(
            a.id, b.id, a.psalm, b.psalm, a.gattung, b.gattung, same_genre=a.gattung == b.gattung
        )
        for a, b in ((passages[i], passages[j]) for i, j in zip(rows, columns, strict=True))
    ]


def filter_pairs_by_genre(pairs: list[GenrePair], genre: str) -> list[GenrePair]:
    """Genre pairs touching `genre` on at least one side, a one-vs-rest restriction."""
    return [pair for pair in pairs if genre in (pair.genre_a, pair.genre_b)]
