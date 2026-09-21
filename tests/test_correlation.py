"""Tests for CCF voxel-correlation input preparation."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from napari_neuron_navigator.analysis.clustering import (
    ClusterExclusionRule,
    ClusterRegionFilter,
)
from napari_neuron_navigator.analysis.correlation import (
    compute_pearson_correlation_matrix,
    correlation_long_to_matrix,
    count_correlation_input_nodes,
)
from napari_neuron_navigator.analysis.region_filter import PreparedClusterRegionFilter


def test_count_correlation_input_nodes_supports_unfiltered_file_scope(tmp_path) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n2", "n2", "n3"],
            "x": [10.0, 30.0, 10.0, 55.0, np.nan],
            "y": [10.0, 30.0, 10.0, 55.0, 10.0],
            "z": [10.0, 30.0, 10.0, 55.0, 10.0],
        }
    ).to_parquet(path, index=False)
    conn = duckdb.connect()
    try:
        count = count_correlation_input_nodes(
            conn,
            str(path),
            None,
            25.0,
            file_ids=["n1", "n3"],
        )
    finally:
        conn.close()

    assert count == 2


def test_count_correlation_input_nodes_applies_region_lookup(tmp_path) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["inside", "outside"],
            "x": [10.0, 55.0],
            "y": [10.0, 55.0],
            "z": [10.0, 55.0],
        }
    ).to_parquet(path, index=False)
    voxel_id_map = np.full((4, 4, 4), -1, dtype=np.int32)
    voxel_id_map[0, 0, 0] = 0
    conn = duckdb.connect()
    try:
        count = count_correlation_input_nodes(
            conn,
            str(path),
            voxel_id_map,
            25.0,
        )
    finally:
        conn.close()

    assert count == 1


def test_count_correlation_input_nodes_applies_exclusion_and_keeps_out_of_bounds(
    tmp_path,
) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["kept", "excluded", "outside"],
            "x": [10.0, 30.0, 200.0],
            "y": [10.0, 30.0, 200.0],
            "z": [10.0, 30.0, 200.0],
        }
    ).to_parquet(path, index=False)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    rule = ClusterExclusionRule(
        region_id=10,
        acronym="EXC",
        node_types=(1,),
    )
    prepared = PreparedClusterRegionFilter(
        region_filter=ClusterRegionFilter(exclude_rules=(rule,)),
        resolution_um=(25.0, 25.0, 25.0),
        atlas_shape=mask.shape,
        include_mask=None,
        exclude_masks=(mask,),
        exclude_mask=mask,
    )
    conn = duckdb.connect()
    try:
        count = count_correlation_input_nodes(
            conn,
            str(path),
            None,
            25.0,
            prepared_region_filter=prepared,
        )
    finally:
        conn.close()

    assert count == 2


def test_empty_correlation_table_returns_empty_matrix_for_actionable_worker_error() -> (
    None
):
    frame, matrix = correlation_long_to_matrix(
        pd.DataFrame(columns=["swc_id_1", "swc_id_2", "r"])
    )

    assert frame.empty
    assert matrix.shape == (0, 0)


def test_ccf_voxel_filter_omits_neuron_with_no_surviving_nodes(tmp_path) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n2", "n2", "n2", "n3", "n3", "n3"],
            "x": [10.0, 10.0, 30.0, 30.0, 60.0, 30.0, 60.0, 60.0],
            "y": [10.0, 10.0, 30.0, 30.0, 60.0, 30.0, 60.0, 60.0],
            "z": [10.0, 10.0, 30.0, 30.0, 60.0, 30.0, 60.0, 60.0],
        }
    ).to_parquet(path, index=False)
    excluded = np.zeros((4, 4, 4), dtype=bool)
    excluded[0, 0, 0] = True
    rule = ClusterExclusionRule(region_id=10, acronym="EXC")
    prepared = PreparedClusterRegionFilter(
        region_filter=ClusterRegionFilter(exclude_rules=(rule,)),
        resolution_um=(25.0, 25.0, 25.0),
        atlas_shape=excluded.shape,
        include_mask=None,
        exclude_masks=(excluded,),
        exclude_mask=excluded,
    )
    conn = duckdb.connect()
    try:
        correlations = compute_pearson_correlation_matrix(
            conn,
            str(path),
            None,
            25.0,
            prepared_region_filter=prepared,
        )
    finally:
        conn.close()

    ids = set(correlations["swc_id_1"]) | set(correlations["swc_id_2"])
    assert ids == {"n2", "n3"}
