"""Builds every unordered psalm pair, labeled by whether the two psalms share a genre."""

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True, slots=True)
class GenrePair:
    """Two psalms compared for genre discrimination, and whether they share a genre."""

    psalm_a: int
    psalm_b: int
    genre_a: str
    genre_b: str
    same_genre: bool


def build_genre_pairs(genre_by_psalm: dict[int, str]) -> list[GenrePair]:
    """One GenrePair per unordered psalm pair, psalm_a < psalm_b, same_genre from genre_by_psalm."""
    psalms = sorted(genre_by_psalm)
    pairs = []
    for a, b in combinations(psalms, 2):
        genre_a, genre_b = genre_by_psalm[a], genre_by_psalm[b]
        pairs.append(GenrePair(a, b, genre_a, genre_b, same_genre=genre_a == genre_b))
    return pairs


def filter_pairs_by_genre(pairs: list[GenrePair], genre: str) -> list[GenrePair]:
    """Genre pairs touching `genre` on at least one side, a one-vs-rest restriction."""
    return [pair for pair in pairs if genre in (pair.genre_a, pair.genre_b)]
