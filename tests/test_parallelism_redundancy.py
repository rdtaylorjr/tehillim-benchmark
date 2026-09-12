from __future__ import annotations

import numpy as np
import pytest

from parallelism.redundancy import joined_vectors


class TestJoinedVectors:
    def test_concatenates_both_representations_for_a_shared_half_verse(self):
        subject = {1: np.array([3.0, 4.0])}
        reference = {1: np.array([0.0, 5.0])}

        joined = joined_vectors(subject, reference)

        assert joined[1].shape == (4,)

    def test_normalises_each_side_so_neither_width_dominates(self):
        """A 4096-wide side would otherwise swamp a 3-wide one in the cosine."""
        subject = {1: np.array([3.0, 4.0])}
        reference = {1: np.array([0.0, 100.0])}

        joined = joined_vectors(subject, reference)

        assert np.linalg.norm(joined[1][:2]) == pytest.approx(1.0)
        assert np.linalg.norm(joined[1][2:]) == pytest.approx(1.0)

    def test_keeps_only_half_verses_both_representations_cover(self):
        subject = {1: np.array([1.0]), 2: np.array([1.0])}
        reference = {2: np.array([1.0]), 3: np.array([1.0])}

        assert set(joined_vectors(subject, reference)) == {2}

    def test_a_zero_vector_side_is_carried_rather_than_dividing_by_zero(self):
        subject = {1: np.zeros(2)}
        reference = {1: np.array([0.0, 1.0])}

        joined = joined_vectors(subject, reference)

        assert joined[1][:2].tolist() == [0.0, 0.0]
        assert np.linalg.norm(joined[1][2:]) == pytest.approx(1.0)

    def test_no_shared_half_verse_yields_nothing(self):
        assert joined_vectors({1: np.array([1.0])}, {2: np.array([1.0])}) == {}
