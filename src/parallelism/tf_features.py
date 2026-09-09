"""Loads the parallel_* Text-Fabric features and reconstructs groups; no alignment import."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from library.bhsa import DEFAULT_CHECKOUT, load_bhsa_api, psalms_book_node

_FEATURES = (
    "parallel_group_id parallel_member_id parallel_type parallel_group parallel_member "
    "parallel_signature parallel_ambiguous"
)
# rdtaylorjr/tehillim-logos has its own release history and cannot share DEFAULT_CHECKOUT.
_TEHILLIM_LOGOS_CHECKOUT = "v1.0"


def load_api(checkout: str = DEFAULT_CHECKOUT) -> Any:
    """Loads BHSA plus tehillim-logos's parallel_* features via Text-Fabric use()."""
    mod = f"rdtaylorjr/tehillim-logos/tf:{_TEHILLIM_LOGOS_CHECKOUT}"
    api = load_bhsa_api(checkout=checkout, mod=mod)
    api.TF.load(_FEATURES, add=True, silent="deep")
    missing = [f for f in _FEATURES.split() if not hasattr(api.F, f)]
    if missing:
        raise RuntimeError(f"required features not loaded: {missing}")
    return api


def read_node_feature_values(api: Any) -> dict[int, dict[str, str]]:
    """Reads every Psalms half-verse node's parallel_* values, skipping nodes with none."""
    F, L = api.F, api.L  # noqa: N806
    book_node = psalms_book_node(api)

    values: dict[int, dict[str, str]] = {}
    for chapter_node in L.d(book_node, otype="chapter"):
        for half_verse_node in L.d(chapter_node, otype="half_verse"):
            group_id = F.parallel_group_id.v(half_verse_node)
            if group_id is None:
                continue
            values[half_verse_node] = {
                "parallel_group_id": group_id,
                "parallel_member_id": F.parallel_member_id.v(half_verse_node),
                "parallel_type": F.parallel_type.v(half_verse_node),
                "parallel_group": F.parallel_group.v(half_verse_node),
                "parallel_member": F.parallel_member.v(half_verse_node),
                "parallel_signature": F.parallel_signature.v(half_verse_node),
                "parallel_ambiguous": F.parallel_ambiguous.v(half_verse_node),
            }
    return values


@dataclass(frozen=True, slots=True)
class ReconstructedGroup:
    """One parallelism group rebuilt from BHSA's per-node annotation features."""

    group_range: str
    parallelism_type: str
    signature: str
    member_ids: tuple[int, ...]
    member_indicators: tuple[str, ...]
    member_nodes: tuple[tuple[int, ...], ...]
    member_ambiguous: tuple[bool, ...]


def reconstruct_groups(node_values: dict[int, dict[str, str]]) -> list[ReconstructedGroup]:
    """Rebuilds each group's member sequence, collapsing same-member_id nodes into one span."""
    group_order: list[str] = []
    group_meta: dict[str, tuple[str, str, str]] = {}
    member_nodes: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    member_info: dict[str, dict[str, tuple[str, bool]]] = defaultdict(dict)
    first_seen: dict[str, dict[str, int]] = defaultdict(dict)

    for node in sorted(node_values):
        raw = node_values[node]
        columns = zip(
            raw["parallel_group_id"].split("; "),
            raw["parallel_member_id"].split("; "),
            raw["parallel_type"].split("; "),
            raw["parallel_group"].split("; "),
            raw["parallel_member"].split("; "),
            raw["parallel_signature"].split("; "),
            raw["parallel_ambiguous"].split("; "),
            strict=True,
        )
        for group_id, member_id, ptype, group_range, member, signature, ambiguous in columns:
            if group_id not in group_meta:
                group_order.append(group_id)
                group_meta[group_id] = (group_range, ptype, signature)
            member_nodes[group_id][member_id].append(node)
            if member_id not in member_info[group_id]:
                member_info[group_id][member_id] = (member, ambiguous == "1")
                first_seen[group_id][member_id] = node

    groups = []
    for group_id in group_order:
        group_range, ptype, signature = group_meta[group_id]
        member_ids = sorted(member_info[group_id], key=lambda mid: first_seen[group_id][mid])
        groups.append(
            ReconstructedGroup(
                group_range=group_range,
                parallelism_type=ptype,
                signature=signature,
                member_ids=tuple(int(mid) for mid in member_ids),
                member_indicators=tuple(member_info[group_id][mid][0] for mid in member_ids),
                member_nodes=tuple(tuple(member_nodes[group_id][mid]) for mid in member_ids),
                member_ambiguous=tuple(member_info[group_id][mid][1] for mid in member_ids),
            )
        )
    return groups
