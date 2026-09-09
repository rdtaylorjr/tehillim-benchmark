"""Decomposes parallelism groups into retrieval pairs; a chiasm pairs by letter identity."""

import itertools
from collections.abc import Container
from dataclasses import dataclass

from parallelism.tf_features import ReconstructedGroup


@dataclass(frozen=True, slots=True)
class RetrievalPair:
    """One annotated parallelism: its two half-verse spans and how they were labelled."""

    pair_id: str
    group_range: str
    parallelism_type: str
    signature: str
    source_nodes: tuple[int, ...]
    target_nodes: tuple[int, ...]
    source_indicator: str
    target_indicator: str


def _chiasm_links(segments: list[str]) -> dict[int, int]:
    """Maps a later segment's index to the earliest segment it mirrors (same letters, reordered)."""
    links: dict[int, int] = {}
    for j, seg_j in enumerate(segments):
        for i in range(j):
            seg_i = segments[i]
            already_linked = i in links or i in links.values()
            if not already_linked and sorted(seg_i) == sorted(seg_j) and seg_i != seg_j:
                links[j] = i
                break
    return links


def _positions_for_signature(signature: str) -> list[tuple[int, int]]:
    """Position-index pairs (0-based, dashes stripped) representing the intended relationships."""
    segments = signature.split("-")
    offsets = []
    offset = 0
    for segment in segments:
        offsets.append(offset)
        offset += len(segment)

    links = _chiasm_links(segments)
    resolved = set(links.keys()) | set(links.values())

    position_pairs: list[tuple[int, int]] = []
    for j, i in sorted(links.items()):
        seg_i, seg_j = segments[i], segments[j]
        next_index_for_letter: dict[str, int] = {}
        seg_j_positions: dict[str, list[int]] = {}
        for pos, letter in enumerate(seg_j):
            seg_j_positions.setdefault(letter, []).append(pos)
        for pos_i, letter in enumerate(seg_i):
            #: The linker matches letters as multisets, so each occurrence has exactly one partner.
            occurrence = next_index_for_letter.get(letter, 0)
            next_index_for_letter[letter] = occurrence + 1
            partner = seg_j_positions[letter][occurrence]
            position_pairs.append((offsets[i] + pos_i, offsets[j] + partner))

    for idx, segment in enumerate(segments):
        if idx in resolved:
            continue
        positions = list(range(offsets[idx], offsets[idx] + len(segment)))
        for a, b in itertools.pairwise(positions):
            position_pairs.append((a, b))

    return sorted(position_pairs)


def filter_pairs_by_type(pairs: list[RetrievalPair], types: frozenset[str]) -> list[RetrievalPair]:
    """Retrieval pairs whose parallelism_type is in the given set."""
    return [pair for pair in pairs if pair.parallelism_type in types]


def filter_pairs_with_vectors(
    pairs: list[RetrievalPair], node_vectors: Container[int]
) -> list[RetrievalPair]:
    """Retrieval pairs whose source and target nodes are all present in node_vectors."""
    return [
        pair
        for pair in pairs
        if all(n in node_vectors for n in pair.source_nodes + pair.target_nodes)
    ]


def build_retrieval_pairs(groups: list[ReconstructedGroup]) -> list[RetrievalPair]:
    """Decomposes each group into retrieval pairs per its signature's structure."""
    pairs = []
    for group_index, group in enumerate(groups):
        member_details = zip(
            group.member_indicators, group.member_nodes, group.member_ambiguous, strict=True
        )
        by_slot = dict(zip(group.member_ids, member_details, strict=True))
        for i, j in _positions_for_signature(group.signature):
            if i not in by_slot or j not in by_slot:
                continue
            indicator_i, nodes_i, ambiguous_i = by_slot[i]
            indicator_j, nodes_j, ambiguous_j = by_slot[j]
            if ambiguous_i or ambiguous_j or nodes_i == nodes_j:
                continue
            pairs.append(
                RetrievalPair(
                    pair_id=f"{group.group_range}:{group_index}:{i}-{j}",
                    group_range=group.group_range,
                    parallelism_type=group.parallelism_type,
                    signature=group.signature,
                    source_nodes=nodes_i,
                    target_nodes=nodes_j,
                    source_indicator=indicator_i,
                    target_indicator=indicator_j,
                )
            )
    return pairs
