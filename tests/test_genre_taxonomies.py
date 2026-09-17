"""The taxonomies a genre benchmark reads and the unit registers each offers."""

import pytest

from genre.taxonomies import TAXONOMIES, register, units_of


class TestTaxonomies:
    def test_logos_classifies_whole_psalms_and_has_no_unit_register(self) -> None:
        assert TAXONOMIES["logos"].units == ()
        assert register("logos", None) == (TAXONOMIES["logos"], None)
        assert units_of("logos", None) == ()

    def test_gunkel_offers_the_three_nested_registers_in_widening_order(self) -> None:
        assert TAXONOMIES["gunkel"].units == ("song", "song_component", "song_component_motif")

    def test_each_gunkel_register_holds_the_source_units_it_names(self) -> None:
        assert units_of("gunkel", "song") == ("Song",)
        assert units_of("gunkel", "song_component") == ("Song", "Component")
        assert units_of("gunkel", "song_component_motif") == ("Song", "Component", "Motif")

    def test_register_refuses_a_unit_for_a_whole_psalm_taxonomy(self) -> None:
        with pytest.raises(ValueError, match="no unit"):
            register("logos", "song")

    def test_register_refuses_a_missing_or_unknown_unit_for_gunkel(self) -> None:
        with pytest.raises(ValueError, match="needs a unit"):
            register("gunkel", None)
        with pytest.raises(ValueError, match="needs a unit"):
            register("gunkel", "psalm")

    def test_register_refuses_an_unknown_taxonomy(self) -> None:
        with pytest.raises(ValueError, match="bellinger"):
            register("bellinger", None)
