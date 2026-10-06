"""Tests for clustering include/exclude rules and prepared atlas masks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import duckdb
import numpy as np
import pandas as pd
import pytest

from napari_neuron_navigator.analysis.clustering import (
    ClusterExclusionRule,
    ClusterRegionFilter,
    ClusterRegionRule,
    ClusterRunMetadata,
)
from napari_neuron_navigator.analysis.region_filter import (
    PreparedClusterRegionFilter,
    cleanup_filtered_source_view,
    excluded_soma_file_ids,
    filter_voxel_frame,
    points_in_mask,
    prepare_cluster_region_filter,
    register_filtered_source_view,
)


@pytest.fixture
def mos_projection_challenge(tmp_path):
    """Known projection counts outside a synthetic bilateral MOs parent mask."""
    annotation = np.zeros((7, 5, 9), dtype=np.int32)
    mos_voxels = ((1, 1, 1), (1, 1, 7))
    target_voxels = ((2, 1, 7), (2, 2, 7), (5, 1, 1), (5, 2, 1))
    for region_id, voxel in zip((11, 12), mos_voxels):
        annotation[voxel] = region_id
    for region_id, voxel in enumerate(target_voxels, start=20):
        annotation[voxel] = region_id
    masks = {"MOs": np.isin(annotation, [11, 12]), "root": annotation > 0}
    atlas = SimpleNamespace(
        annotation=annotation,
        resolution=(25.0, 25.0, 25.0),
        atlas_name="synthetic_projection_challenge",
        get_structure_mask=lambda acronym: masks[acronym],
    )
    vectors = {
        "cortex-a": [3, 1, 0, 0],
        "cortex-b": [6, 2, 0, 0],
        "brainstem-a": [0, 0, 1, 3],
        "brainstem-b": [0, 0, 2, 6],
        "mos-only": [0, 0, 0, 0],
    }
    rows = []
    for file_id, counts in vectors.items():
        node_id = 7
        # Same display ID and repeated noncontiguous node IDs in every file.
        # Shared local arbor dominates the raw vectors and must be excluded.
        for voxel, count in zip((*mos_voxels, *target_voxels), (30, 20, *counts)):
            for _ in range(count):
                rows.append(
                    {
                        "file_id": file_id,
                        "neuron_id": "shared",
                        "subject": file_id,
                        "node_id": node_id,
                        "type": 2,
                        "x": (voxel[0] + 0.75) * 25.0,
                        "y": (voxel[1] + 0.75) * 25.0,
                        "z": (voxel[2] + 0.75) * 25.0,
                        "region_id": int(annotation[voxel]),
                    }
                )
                node_id += 11
    frame = pd.DataFrame(rows)
    path = tmp_path / "mos_challenge.parquet"
    frame.to_parquet(path, index=False)
    # Independent oracle: manually retain only the known projection regions.
    oracle_path = tmp_path / "known_projection_nodes.parquet"
    frame.loc[frame["region_id"].isin(range(20, 24))].to_parquet(
        oracle_path, index=False
    )
    return atlas, path, oracle_path, vectors


@pytest.mark.parametrize("linkage", ["ward", "average"])
@pytest.mark.parametrize(
    "include_root", [False, True], ids=["exclusion-only", "include-root"]
)
def test_ccf_mos_exclusion_preserves_distinct_projection_populations(
    mos_projection_challenge, linkage, include_root
):
    """At 0% dilation, preflight and the real worker use only projection nodes."""
    from napari_neuron_navigator.analysis.correlation import (
        compute_pearson_correlation_matrix,
        correlation_long_to_matrix,
    )
    from tests.test_workers import _import_workers_module

    atlas, path, oracle_path, vectors = mos_projection_challenge
    region_filter = ClusterRegionFilter(
        include_rules=(ClusterRegionRule(region_id=1, acronym="root"),)
        if include_root
        else (),
        exclude_rules=(
            ClusterExclusionRule(
                region_id=10,
                acronym="MOs",
                represented_region_ids=(11, 12),
                # These settings are Soma Location only and must not weaken voxel exclusion.
                node_types=(99,),
                minimum_node_count=1000,
                dilation_fraction=0.0,
            ),
        ),
    )
    workers = _import_workers_module()
    preflight = workers.ClusteringPreflightWorker(
        parquet_path=str(path),
        atlas=atlas,
        coordinate_space="ccf",
        clustering_method="voxel",
        file_ids=list(vectors),
        region_filter=region_filter,
    )
    counted, errors = [], []
    preflight.finished.connect(counted.append)
    preflight.error.connect(errors.append)
    preflight.run()
    assert errors == []
    assert len(counted) == 1
    assert counted[0].node_count == 24
    worker = workers.CorrelationWorker(
        parquet_path=str(path),
        atlas=atlas,
        region_selection=None,
        region_filter=region_filter,
        prepared_region_filter=counted[0].prepared_region_filter,
        file_ids=list(vectors),
        linkage_method=linkage,
        n_clusters=2,
    )
    finished = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.run()
    assert errors == []
    assert len(finished) == 1
    result = finished[0]

    with duckdb.connect() as conn:
        oracle = compute_pearson_correlation_matrix(
            conn,
            str(oracle_path),
            None,
            resolution=25.0,
        )
    oracle_frame, oracle_matrix = correlation_long_to_matrix(oracle)
    assert result.neuron_ids == oracle_frame.columns.tolist()
    np.testing.assert_allclose(result.correlation_matrix, oracle_matrix)
    assert result.unassigned_neuron_ids == ["mos-only"]
    assert result.metadata.region_filter == region_filter
    labels = dict(zip(result.neuron_ids, result.labels))
    assert labels["cortex-a"] == labels["cortex-b"]
    assert labels["brainstem-a"] == labels["brainstem-b"]
    assert labels["cortex-a"] != labels["brainstem-a"]
    # Independent dense Pearson oracle includes zero cross-products for
    # disjoint populations; their r is -2/3, not perfect anticorrelation.
    np.testing.assert_allclose(
        result.correlation_matrix,
        np.corrcoef([vectors[file_id] for file_id in result.neuron_ids]),
        atol=1e-7,
    )


class _Atlas:
    def __init__(self) -> None:
        self.annotation = np.zeros((4, 4, 4), dtype=np.int32)
        self.resolution = (10.0, 10.0, 10.0)
        self.atlas_name = "test"
        self._masks = {
            "INC": np.zeros((4, 4, 4), dtype=np.uint8),
            "EXC": np.zeros((4, 4, 4), dtype=np.uint8),
        }
        self._masks["INC"][1:3, 1:3, 1:3] = 1
        self._masks["EXC"][2, 2, 2] = 1

    def get_structure_mask(self, acronym: str) -> np.ndarray:
        return self._masks[acronym]


def _include_rule() -> ClusterRegionRule:
    return ClusterRegionRule(
        region_id=10,
        acronym="INC",
        represented_region_ids=(11,),
        represented_region_acronyms=("INC1",),
    )


def _exclude_rule(
    *,
    region_id: int = 20,
    acronym: str = "EXC",
    node_types: tuple[int, ...] | None = (1,),
    minimum: int = 1,
) -> ClusterExclusionRule:
    return ClusterExclusionRule(
        region_id=region_id,
        acronym=acronym,
        represented_region_ids=(region_id,),
        represented_region_acronyms=(acronym,),
        node_types=node_types,
        minimum_node_count=minimum,
    )


def test_prepare_region_filter_unions_rules_and_exclusion_wins() -> None:
    atlas = _Atlas()
    region_filter = ClusterRegionFilter(
        include_rules=(_include_rule(),),
        exclude_rules=(_exclude_rule(),),
    )

    prepared = prepare_cluster_region_filter(atlas, region_filter)

    assert prepared is not None
    assert prepared.include_mask is not None
    assert prepared.exclude_mask is not None
    assert prepared.include_mask[2, 2, 2]
    assert prepared.exclude_mask[2, 2, 2]


def test_prepare_region_filter_dilates_each_rule_and_caches_duplicates(
    monkeypatch,
) -> None:
    atlas = _Atlas()
    get_mask = MagicMock(
        side_effect=lambda _atlas, acronym: atlas.get_structure_mask(acronym)
    )
    dilate = MagicMock(side_effect=lambda mask, **_kwargs: np.asarray(mask, dtype=bool))
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.region_filter.get_region_mask",
        get_mask,
    )
    monkeypatch.setattr(
        "napari_neuron_navigator.analysis.region_filter.dilate_mask_to_volume_increase",
        dilate,
    )
    duplicated = ClusterRegionRule(
        region_id=10,
        acronym="INC",
        dilation_fraction=0.1,
    )
    region_filter = ClusterRegionFilter(
        include_rules=(duplicated,),
        exclude_rules=(
            ClusterExclusionRule(
                region_id=10,
                acronym="INC",
                dilation_fraction=0.1,
            ),
            ClusterExclusionRule(
                region_id=20,
                acronym="EXC",
                dilation_fraction=0.3,
            ),
        ),
    )

    prepared = prepare_cluster_region_filter(atlas, region_filter)

    assert prepared is not None
    assert get_mask.call_count == 2
    assert [call.kwargs["increase_fraction"] for call in dilate.call_args_list] == [
        0.1,
        0.3,
    ]


def test_filtered_source_keeps_out_of_atlas_rows_for_exclusion_only(tmp_path) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["inside", "excluded", "outside"],
            "node_id": [7, 7, 7],
            "x": [10.0, 20.0, 90.0],
            "y": [10.0, 20.0, 90.0],
            "z": [10.0, 20.0, 90.0],
        }
    ).to_parquet(path)
    excluded = np.zeros((4, 4, 4), dtype=bool)
    excluded[2, 2, 2] = True
    prepared = PreparedClusterRegionFilter(
        region_filter=ClusterRegionFilter(exclude_rules=(_exclude_rule(),)),
        resolution_um=(10.0, 10.0, 10.0),
        atlas_shape=(4, 4, 4),
        include_mask=None,
        exclude_masks=(excluded,),
        exclude_mask=excluded,
    )

    conn = duckdb.connect()
    source, relations = register_filtered_source_view(
        conn,
        f"read_parquet('{path}')",
        prepared,
    )
    try:
        ids = [
            row[0] for row in conn.execute(f"SELECT file_id FROM {source}").fetchall()
        ]
    finally:
        cleanup_filtered_source_view(
            conn,
            "cluster_region_filtered_source",
            relations,
        )
        conn.close()

    assert set(ids) == {"inside", "outside"}


def test_filtered_source_exclusion_wins_inside_include_mask(tmp_path) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["included", "overlap", "outside"],
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 20.0, 30.0],
            "z": [10.0, 20.0, 30.0],
        }
    ).to_parquet(path)
    include = np.zeros((4, 4, 4), dtype=bool)
    include[1, 1, 1] = True
    include[2, 2, 2] = True
    exclude = np.zeros_like(include)
    exclude[2, 2, 2] = True
    prepared = PreparedClusterRegionFilter(
        region_filter=ClusterRegionFilter(
            include_rules=(_include_rule(),),
            exclude_rules=(_exclude_rule(),),
        ),
        resolution_um=(10.0, 10.0, 10.0),
        atlas_shape=include.shape,
        include_mask=include,
        exclude_masks=(exclude,),
        exclude_mask=exclude,
    )

    conn = duckdb.connect()
    source, relations = register_filtered_source_view(
        conn,
        f"read_parquet('{path}')",
        prepared,
    )
    try:
        ids = [
            row[0] for row in conn.execute(f"SELECT file_id FROM {source}").fetchall()
        ]
    finally:
        cleanup_filtered_source_view(
            conn,
            "cluster_region_filtered_source",
            relations,
        )
        conn.close()

    assert ids == ["included"]


def test_soma_exclusion_thresholds_are_per_rule_and_file_id(tmp_path) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            # neuron_id and node_id deliberately collide across files.
            "file_id": ["a", "a", "b", "b"],
            "neuron_id": [1, 1, 1, 1],
            "node_id": [7, 8, 7, 8],
            "type": [2, 2, 2, 0],
            "x": [10.0, 10.0, 10.0, 10.0],
            "y": [10.0, 10.0, 10.0, 10.0],
            "z": [10.0, 10.0, 10.0, 10.0],
        }
    ).to_parquet(path)
    axon_mask = np.zeros((4, 4, 4), dtype=bool)
    axon_mask[1, 1, 1] = True
    undefined_mask = np.zeros((4, 4, 4), dtype=bool)
    # Both rules overlap. File "b" stays below the axon-typed threshold but
    # independently reaches the undefined-node threshold in the same voxel.
    undefined_mask[1, 1, 1] = True
    rules = (
        _exclude_rule(
            region_id=20,
            acronym="AXR",
            node_types=(2,),
            minimum=2,
        ),
        _exclude_rule(
            region_id=30,
            acronym="UNR",
            node_types=(0,),
            minimum=1,
        ),
    )
    prepared = PreparedClusterRegionFilter(
        region_filter=ClusterRegionFilter(exclude_rules=rules),
        resolution_um=(10.0, 10.0, 10.0),
        atlas_shape=(4, 4, 4),
        include_mask=None,
        exclude_masks=(axon_mask, undefined_mask),
        exclude_mask=axon_mask | undefined_mask,
    )

    excluded = excluded_soma_file_ids(path, prepared)

    assert excluded == {"a", "b"}


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_soma_exclusion_covers_the_entire_boundary_voxel(tmp_path, axis) -> None:
    """The bounding query must not discard the upper half of a masked voxel."""
    resolution = np.array([10.0, 20.0, 40.0])
    mask = np.zeros((3, 4, 5), dtype=bool)
    mask[1, 2, 4] = True
    lower = np.array([1, 2, 4]) * resolution
    upper = lower + resolution
    coords = np.tile(lower + 0.25 * resolution, (5, 1))
    coords[1, axis] = lower[axis] + 0.75 * resolution[axis]
    coords[2, axis] = np.nextafter(upper[axis], -np.inf)
    coords[3, axis] = upper[axis]
    coords[4, axis] = np.nextafter(lower[axis], -np.inf)
    frame = pd.DataFrame(coords, columns=["x", "y", "z"])
    frame["file_id"] = [
        "lower",
        "upper",
        "upper_edge",
        "outside_upper",
        "outside_lower",
    ]
    frame["type"] = 1
    path = tmp_path / "boundary.parquet"
    frame.to_parquet(path, index=False)
    prepared = PreparedClusterRegionFilter(
        region_filter=ClusterRegionFilter(exclude_rules=(_exclude_rule(),)),
        resolution_um=tuple(resolution),
        atlas_shape=mask.shape,
        include_mask=None,
        exclude_masks=(mask,),
        exclude_mask=mask,
    )

    assert points_in_mask(coords, mask, resolution).tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert set(filter_voxel_frame(frame, prepared)["file_id"]) == {
        "outside_upper",
        "outside_lower",
    }
    assert excluded_soma_file_ids(path, prepared) == {"lower", "upper", "upper_edge"}

    with duckdb.connect() as conn:
        source, relations = register_filtered_source_view(
            conn, f"read_parquet('{path}')", prepared
        )
        try:
            retained_ids = {
                row[0]
                for row in conn.execute(f"SELECT file_id FROM {source}").fetchall()
            }
        finally:
            cleanup_filtered_source_view(conn, source, relations)
    assert retained_ids == {"outside_upper", "outside_lower"}


def test_region_filter_metadata_keeps_legacy_includes_and_canonical_rules() -> None:
    region_filter = ClusterRegionFilter(
        include_rules=(
            _include_rule(),
            ClusterRegionRule(
                region_id=12,
                acronym="INC2",
                represented_region_ids=(13,),
                represented_region_acronyms=("INC2a",),
                dilation_fraction=0.2,
            ),
        ),
        exclude_rules=(_exclude_rule(node_types=(0, 2, 99), minimum=3),),
    )

    metadata = ClusterRunMetadata.from_region_filter(
        region_filter=region_filter,
        analysis_method="voxel_correlation",
        clustering_algorithm="hierarchical",
        distance_metric="one_minus_pearson_r",
        clustering_linkage="average",
        dendrogram_linkage="average",
        requested_cluster_count=2,
        actual_cluster_count=2,
        dbscan_eps=None,
        dbscan_min_samples=None,
        atlas_name="test",
        atlas_resolution_um=(10.0, 10.0, 10.0),
        source_parquet_path="nodes.parquet",
        dendrogram_leaf_order=[0, 1],
    )
    payload = metadata.to_dict()

    assert payload["selected_region_ids"] == [10, 12]
    assert payload["represented_region_ids"] == [11, 13]
    assert payload["dilation_fraction"] is None
    assert payload["region_filter"]["overlap_policy"] == "exclude_wins"
    exclusion = payload["region_filter"]["exclude_rules"][0]
    assert exclusion["node_types"] == [0, 2, 99]
    assert exclusion["node_type_labels"] == [
        "Undefined",
        "Axon-typed (type 2)",
        "Type 99",
    ]
    assert exclusion["minimum_node_count"] == 3
