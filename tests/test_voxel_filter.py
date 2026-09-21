"""Tests for independent voxel-correlation node filters."""

from __future__ import annotations

import duckdb
import pandas as pd

from napari_neuron_navigator.analysis.correlation import (
    count_correlation_input_nodes,
)
from napari_neuron_navigator.analysis.voxel_filter import (
    VoxelNodeFilter,
    prepare_voxel_node_filter_from_parquet,
    query_dendrite_label_coverage,
)


def _write_filter_parquet(path) -> None:
    pd.DataFrame(
        {
            # The repeated display id is deliberate: file_id must keep these
            # morphologies separate.
            "file_id": ["a", "a", "a", "b", "b", "c", "c", "d", "d", "d"],
            "neuron_id": ["shared"] * 7 + ["other"] * 3,
            "node_id": [1, 2, 3, 1, 2, 1, 2, 1, 2, 3],
            "type": [1, 3, 2, 1, 2, 3, 2, 1, 2, 2],
            "x": [0.0, 50.0, 100.0, 0.0, 100.0, 50.0, 100.0, 0.0, 25.0, 25.1],
            "y": [0.0] * 10,
            "z": [0.0] * 10,
        }
    ).to_parquet(path, index=False)


def test_dendrite_coverage_is_whole_neuron_and_keyed_by_file_id(tmp_path) -> None:
    path = tmp_path / "filters.parquet"
    _write_filter_parquet(path)

    coverage = query_dendrite_label_coverage(path)

    assert coverage.input_file_ids == ("a", "b", "c", "d")
    assert coverage.labeled_file_ids == ("a", "c")
    assert coverage.labeled_neuron_count == 2
    assert coverage.excluded_neuron_count == 2


def test_dendrite_filter_with_no_matching_neurons_returns_no_rows(tmp_path) -> None:
    path = tmp_path / "filters.parquet"
    _write_filter_parquet(path)
    settings = VoxelNodeFilter(
        require_dendrite_labels=True,
        dendrite_node_types=(4,),
    )

    conn = duckdb.connect()
    try:
        count = count_correlation_input_nodes(
            conn,
            str(path),
            voxel_id_map=None,
            resolution=1.0,
            voxel_node_filter=settings,
        )
    finally:
        conn.close()

    assert count == 0


def test_type_include_dendrite_cohort_and_soma_distance_compose(tmp_path) -> None:
    path = tmp_path / "filters.parquet"
    _write_filter_parquet(path)
    settings = VoxelNodeFilter(
        node_type_mode="include",
        node_types=(2,),
        require_dendrite_labels=True,
        exclude_within_soma_um=25.0,
        dendrite_node_types=(3,),
    )
    prepared = prepare_voxel_node_filter_from_parquet(path, settings)

    conn = duckdb.connect()
    try:
        count = count_correlation_input_nodes(
            conn,
            str(path),
            voxel_id_map=None,
            resolution=1.0,
            voxel_node_filter=settings,
            prepared_voxel_filter=prepared,
        )
    finally:
        conn.close()

    # a survives all three filters. b lacks a dendrite label; c lacks a soma;
    # d has a soma but lacks a dendrite label.
    assert count == 1
    assert prepared.dendrite_labeled_file_ids == ("a", "c")
    assert prepared.soma_file_ids == ("a", "b", "d")
    assert prepared.missing_soma_neuron_count == 1


def test_exclude_mode_keeps_only_nodes_not_explicitly_excluded(tmp_path) -> None:
    path = tmp_path / "filters.parquet"
    _write_filter_parquet(path)
    settings = VoxelNodeFilter(
        node_type_mode="exclude",
        node_types=(1, 3, 4),
    )

    conn = duckdb.connect()
    try:
        count = count_correlation_input_nodes(
            conn,
            str(path),
            voxel_id_map=None,
            resolution=1.0,
            voxel_node_filter=settings,
        )
    finally:
        conn.close()

    assert count == 5


def test_soma_distance_excludes_the_boundary_and_missing_somas(tmp_path) -> None:
    path = tmp_path / "filters.parquet"
    _write_filter_parquet(path)
    settings = VoxelNodeFilter(exclude_within_soma_um=25.0)

    conn = duckdb.connect()
    try:
        count = count_correlation_input_nodes(
            conn,
            str(path),
            voxel_id_map=None,
            resolution=1.0,
            file_ids=["d", "c"],
            voxel_node_filter=settings,
        )
    finally:
        conn.close()

    # d's soma and node at exactly 25 um are excluded; 25.1 um remains. c has
    # no soma and is excluded from this mode.
    assert count == 1


def test_filter_metadata_records_reproducible_semantics(tmp_path) -> None:
    path = tmp_path / "filters.parquet"
    _write_filter_parquet(path)
    settings = VoxelNodeFilter(
        node_type_mode="include",
        node_types=(2,),
        require_dendrite_labels=True,
        exclude_within_soma_um=25.0,
        dendrite_node_types=(3,),
    )
    metadata = prepare_voxel_node_filter_from_parquet(path, settings).metadata()

    assert metadata["version"] == 1
    assert metadata["node_types"] == [2]
    assert metadata["node_type_labels"] == ["Axon-typed (type 2)"]
    assert metadata["soma_distance_boundary"] == "exclude_less_than_or_equal"
    assert metadata["dendrite_labeled_neuron_count"] == 2
    assert metadata["missing_soma_neuron_count"] == 1
