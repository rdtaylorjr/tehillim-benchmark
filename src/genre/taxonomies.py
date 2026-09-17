"""The genre taxonomies the benchmark scores against, each with its unit registers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """One classification of the Psalms, and the registers of units it is read at, if any."""

    name: str
    #: Each register names the source units whose assignments count as items, in hive order.
    registers: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def units(self) -> tuple[str, ...]:
        """The register names, the `unit=` values of the hive, none for a whole-psalm taxonomy."""
        return tuple(self.registers)


#: Gunkel's Lied, Stück, and Motiv under tehillim-gunkel's English unit names.
GUNKEL_SONG = "Song"
GUNKEL_COMPONENT = "Component"
GUNKEL_MOTIF = "Motif"

TAXONOMIES: dict[str, Taxonomy] = {
    "logos": Taxonomy("logos"),
    "gunkel": Taxonomy(
        "gunkel",
        {
            "song": (GUNKEL_SONG,),
            "song_component": (GUNKEL_SONG, GUNKEL_COMPONENT),
            "song_component_motif": (GUNKEL_SONG, GUNKEL_COMPONENT, GUNKEL_MOTIF),
        },
    ),
}


def register(taxonomy: str, unit: str | None) -> tuple[Taxonomy, str | None]:
    """The taxonomy and register a pair of hive values names, refusing one the tree lacks."""
    found = TAXONOMIES.get(taxonomy)
    if found is None:
        raise ValueError(f"unknown taxonomy {taxonomy!r}, expected one of {sorted(TAXONOMIES)}")
    if not found.registers:
        if unit is not None:
            raise ValueError(f"taxonomy {taxonomy!r} classifies whole psalms and has no unit")
        return found, None
    if unit not in found.registers:
        raise ValueError(f"taxonomy {taxonomy!r} needs a unit among {found.units}, got {unit!r}")
    return found, unit


def units_of(taxonomy: str, unit: str | None) -> tuple[str, ...]:
    """The source units a register counts as items, none for a whole-psalm taxonomy."""
    found, name = register(taxonomy, unit)
    return () if name is None else found.registers[name]
