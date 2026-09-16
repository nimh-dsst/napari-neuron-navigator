"""Tests for named cluster-assignment state."""

from types import SimpleNamespace

import numpy as np

from napari_neuron_navigator.cluster_assignments import ClusterAssignmentStore


def test_store_keeps_sparse_runs_and_switches_active_assignment() -> None:
    store = ClusterAssignmentStore()
    soma = store.add(
        name="Soma Location 1",
        assignments={"n1": 1, "n2": 1, "n3": 2},
        input_file_ids=["n1", "n2", "n3"],
    )
    voxel = store.add(
        name="Voxel Correlation 1",
        assignments={"n1": 2, "n2": 1},
        input_file_ids=["n1", "n2"],
        parent_assignment_id=soma.assignment_id,
        parent_cluster_ids=[1],
    )

    assert store.active is voxel
    assert store.active_map(["n1", "n2", "n3"]) == {
        "n1": 2,
        "n2": 1,
        "n3": None,
    }
    store.set_active(soma.assignment_id)
    assert store.active_map(["n1", "n2", "n3"]) == {
        "n1": 1,
        "n2": 1,
        "n3": 2,
    }


def test_store_add_result_generates_names_and_durable_provenance() -> None:
    store = ClusterAssignmentStore()
    result = SimpleNamespace(
        neuron_ids=["n1", "n2"],
        labels=np.array([1, 2], dtype=np.int32),
        unassigned_neuron_ids=["n3"],
    )

    first = store.add_result(
        result,
        method_name="Soma Location",
        input_file_ids=["n1", "n2", "n3"],
        run_metadata={"algorithm": "hierarchical"},
        input_scope="selected",
        coordinate_space="CCFv3 Coordinates",
    )
    assert first.name == "Soma Location 1"
    store.rename(first.assignment_id, "Spatial groups")
    second = store.add_result(result, method_name="Soma Location")

    assert first.name == "Spatial groups"
    assert first.column_name == "cluster_soma_location_1"
    assert first.assignments == {"n1": 1, "n2": 2}
    assert first.unassigned_neuron_ids == ("n3",)
    assert first.input_scope == "selected"
    assert first.runtime_result is result
    assert first.run_metadata["clustering_method"] == "Soma Location"
    assert first.run_metadata["method_run_index"] == 1
    assert second.name == "Soma Location 2"


def test_store_round_trip_omits_runtime_result_and_preserves_stable_column() -> None:
    store = ClusterAssignmentStore()
    assignment = store.add(
        name="Soma Location 1",
        assignments={"n1": 3},
        input_file_ids=["n1", "n2"],
        unassigned_neuron_ids=["n2"],
        label_colors={3: [0.1, 0.2, 0.3, 1.0]},
        run_metadata={
            "method": "soma",
            "region_filter": {
                "version": 1,
                "overlap_policy": "exclude_wins",
                "include_rules": [
                    {"region_id": 184, "dilation_fraction": 0.2}
                ],
                "exclude_rules": [
                    {
                        "region_id": 500,
                        "node_types": [2, 99],
                        "minimum_node_count": 3,
                    }
                ],
            },
        },
        runtime_result=object(),
    )
    old_column = assignment.column_name
    store.rename(assignment.assignment_id, "Spatial groups")

    restored = ClusterAssignmentStore.from_state(store.to_state())

    assert restored.active is not None
    assert restored.active.name == "Spatial groups"
    assert restored.active.column_name == old_column
    assert restored.active.label_colors == {3: [0.1, 0.2, 0.3, 1.0]}
    assert restored.active.run_metadata["region_filter"] == (
        assignment.run_metadata["region_filter"]
    )
    assert restored.active.runtime_result is None


def test_delete_active_uses_most_recent_remaining_assignment() -> None:
    store = ClusterAssignmentStore()
    first = store.add(
        name="First",
        assignments={"n1": 1},
        input_file_ids=["n1"],
    )
    second = store.add(
        name="Second",
        assignments={"n1": 2},
        input_file_ids=["n1"],
    )

    store.delete(second.assignment_id)
    assert store.active is first
    store.delete(first.assignment_id)
    assert store.active is None


def test_import_legacy_cluster_values_creates_one_active_set() -> None:
    store = ClusterAssignmentStore()

    imported = store.import_legacy({"n1": 4, "n2": None, "bad": "x"})

    assert imported is not None
    assert imported.name == "Imported Cluster Assignment"
    assert imported.assignments == {"n1": 4}
    assert imported.label_colors[4][3] == 1.0
    assert store.active is imported


def test_fallback_label_colors_are_deterministic_across_input_order() -> None:
    first_store = ClusterAssignmentStore()
    first = first_store.add(
        name="First",
        assignments={"n1": 2, "n2": 1},
        input_file_ids=["n1", "n2"],
    )
    second_store = ClusterAssignmentStore()
    second = second_store.add(
        name="Second",
        assignments={"n2": 1, "n1": 2},
        input_file_ids=["n2", "n1"],
    )

    assert first.label_colors == second.label_colors
