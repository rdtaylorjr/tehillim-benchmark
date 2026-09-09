from parallelism.tf_features import ReconstructedGroup, reconstruct_groups


def test_reconstruct_groups_rebuilds_a_simple_couplet() -> None:
    node_values = {
        100: {
            "parallel_group_id": "0",
            "parallel_member_id": "0",
            "parallel_type": "Synonymous",
            "parallel_group": "g1",
            "parallel_member": "A",
            "parallel_signature": "AB",
            "parallel_ambiguous": "0",
        },
        101: {
            "parallel_group_id": "0",
            "parallel_member_id": "1",
            "parallel_type": "Synonymous",
            "parallel_group": "g1",
            "parallel_member": "B",
            "parallel_signature": "AB",
            "parallel_ambiguous": "0",
        },
    }

    groups = reconstruct_groups(node_values)

    assert groups == [
        ReconstructedGroup(
            group_range="g1",
            parallelism_type="Synonymous",
            signature="AB",
            member_ids=(0, 1),
            member_indicators=("A", "B"),
            member_nodes=((100,), (101,)),
            member_ambiguous=(False, False),
        )
    ]


def test_reconstruct_groups_disambiguates_two_instances_sharing_one_group_range() -> None:
    """Two couplets sharing a group_range must stay separate groups via distinct group_id."""
    node_values = {
        200: {
            "parallel_group_id": "0",
            "parallel_member_id": "0",
            "parallel_type": "Synthetic",
            "parallel_group": "g1",
            "parallel_member": "A",
            "parallel_signature": "A-B",
            "parallel_ambiguous": "0",
        },
        201: {
            "parallel_group_id": "0",
            "parallel_member_id": "1",
            "parallel_type": "Synthetic",
            "parallel_group": "g1",
            "parallel_member": "B",
            "parallel_signature": "A-B",
            "parallel_ambiguous": "0",
        },
        202: {
            "parallel_group_id": "1",
            "parallel_member_id": "0",
            "parallel_type": "Synthetic",
            "parallel_group": "g1",
            "parallel_member": "A",
            "parallel_signature": "A-B",
            "parallel_ambiguous": "0",
        },
        203: {
            "parallel_group_id": "1",
            "parallel_member_id": "1",
            "parallel_type": "Synthetic",
            "parallel_group": "g1",
            "parallel_member": "B",
            "parallel_signature": "A-B",
            "parallel_ambiguous": "0",
        },
    }

    groups = reconstruct_groups(node_values)

    assert len(groups) == 2
    assert groups[0].group_range == groups[1].group_range == "g1"
    assert groups[0].member_nodes == ((200,), (201,))
    assert groups[1].member_nodes == ((202,), (203,))


def test_reconstruct_groups_collapses_a_member_spanning_two_nodes() -> None:
    """A member spanning two half-verses reconstructs as one occurrence with two nodes."""
    node_values = {
        1: {
            "parallel_group_id": "0",
            "parallel_member_id": "0",
            "parallel_type": "Synthetic",
            "parallel_group": "g",
            "parallel_member": "A",
            "parallel_signature": "A",
            "parallel_ambiguous": "0",
        },
        2: {
            "parallel_group_id": "0",
            "parallel_member_id": "0",
            "parallel_type": "Synthetic",
            "parallel_group": "g",
            "parallel_member": "A",
            "parallel_signature": "A",
            "parallel_ambiguous": "0",
        },
    }

    groups = reconstruct_groups(node_values)

    assert len(groups) == 1
    assert groups[0].member_indicators == ("A",)
    assert groups[0].member_nodes == ((1, 2),)


def test_reconstruct_groups_reads_position_aligned_multi_occurrence_nodes_in_order() -> None:
    """A node touched by two distinct groups: entries at that node must be split by position."""
    node_values = {
        300: {
            "parallel_group_id": "0; 1",
            "parallel_member_id": "0; 0",
            "parallel_type": "Synonymous; Synthetic",
            "parallel_group": "ga; gb",
            "parallel_member": "A; A",
            "parallel_signature": "AB; A-B",
            "parallel_ambiguous": "0; 0",
        },
    }

    groups = reconstruct_groups(node_values)

    assert len(groups) == 2
    assert groups[0].group_range == "ga"
    assert groups[0].parallelism_type == "Synonymous"
    assert groups[1].group_range == "gb"
    assert groups[1].parallelism_type == "Synthetic"


def test_reconstruct_groups_marks_an_ambiguous_occurrence() -> None:
    node_values = {
        100: {
            "parallel_group_id": "0",
            "parallel_member_id": "0",
            "parallel_type": "Synonymous",
            "parallel_group": "g1",
            "parallel_member": "A",
            "parallel_signature": "AB",
            "parallel_ambiguous": "1",
        },
    }

    groups = reconstruct_groups(node_values)

    assert groups[0].member_ambiguous == (True,)
