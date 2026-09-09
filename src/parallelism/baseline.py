"""Builds a negative-control set of adjacent half-verse pairs never annotated as parallel."""

import itertools


def build_unmarked_half_verse_pairs(
    half_verses_by_psalm: dict[int, list[int]], marked_nodes: set[int]
) -> list[tuple[int, int]]:
    """Adjacent half-verse pairs, within one psalm, where neither node is in marked_nodes."""
    pairs = []
    for nodes in half_verses_by_psalm.values():
        for a, b in itertools.pairwise(nodes):
            if a not in marked_nodes and b not in marked_nodes:
                pairs.append((a, b))
    return pairs
