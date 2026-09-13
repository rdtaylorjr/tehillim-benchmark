"""Decomposes parallelism groups into retrieval pairs by shared letter, else by whole line."""

import itertools
from collections import defaultdict
from collections.abc import Container
from dataclasses import dataclass

from parallelism.tf_features import ReconstructedGroup

Slots = tuple[int, ...]


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


def _segment_slots(signature: str) -> list[Slots]:
    """Member positions of each dash-separated line, numbered left to right."""
    slots: list[Slots] = []
    offset = 0
    for segment in signature.split("-"):
        slots.append(tuple(range(offset, offset + len(segment))))
        offset += len(segment)
    return slots


def slot_pairs_for_signature(signature: str) -> list[tuple[Slots, Slots]]:
    """Corresponding member sets: same-letter members, else adjacent lines as wholes."""
    letters = signature.replace("-", "")
    by_letter: defaultdict[str, list[int]] = defaultdict(list)
    for position, letter in enumerate(letters):
        by_letter[letter].append(position)
    shared = [positions for positions in by_letter.values() if len(positions) > 1]
    if shared:
        pairs = [
            ((a,), (b,)) for positions in shared for a, b in itertools.combinations(positions, 2)
        ]
        return sorted(pairs)
    lines = _segment_slots(signature)
    if len(lines) == 1:
        return [((a,), (b,)) for a, b in itertools.pairwise(lines[0])]
    return list(itertools.pairwise(lines))


def _resolve(slots: Slots, nodes_by_slot: dict[int, tuple[int, ...]]) -> tuple[int, ...]:
    """Union of the slots' nodes in first-seen order, or empty when any slot is missing."""
    if any(slot not in nodes_by_slot for slot in slots):
        return ()
    return tuple(dict.fromkeys(itertools.chain.from_iterable(nodes_by_slot[s] for s in slots)))


def build_retrieval_pairs(groups: list[ReconstructedGroup]) -> list[RetrievalPair]:
    """Decomposes each group into pairs, keeping one pair per distinct node-set pairing."""
    pairs = []
    for group_index, group in enumerate(groups):
        nodes_by_slot = dict(zip(group.member_ids, group.member_nodes, strict=True))
        indicator_by_slot = dict(zip(group.member_ids, group.member_indicators, strict=True))
        flags = zip(group.member_ids, group.member_ambiguous, strict=True)
        ambiguous = {slot for slot, flag in flags if flag}
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        for source_slots, target_slots in slot_pairs_for_signature(group.signature):
            if ambiguous.intersection(source_slots + target_slots):
                continue
            source_nodes = _resolve(source_slots, nodes_by_slot)
            target_nodes = _resolve(target_slots, nodes_by_slot)
            if not source_nodes or not target_nodes or source_nodes == target_nodes:
                continue
            if (source_nodes, target_nodes) in seen:
                continue
            seen.add((source_nodes, target_nodes))
            source_id = "+".join(map(str, source_slots))
            target_id = "+".join(map(str, target_slots))
            pairs.append(
                RetrievalPair(
                    pair_id=f"{group.group_range}:{group_index}:{source_id}-{target_id}",
                    group_range=group.group_range,
                    parallelism_type=group.parallelism_type,
                    signature=group.signature,
                    source_nodes=source_nodes,
                    target_nodes=target_nodes,
                    source_indicator="".join(indicator_by_slot[s] for s in source_slots),
                    target_indicator="".join(indicator_by_slot[s] for s in target_slots),
                )
            )
    return pairs


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
