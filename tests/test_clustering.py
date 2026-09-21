"""Tests for clustering helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from napari_neuron_navigator.analysis import clustering
from napari_neuron_navigator.analysis.clustering import (
    ClusterExclusionRule,
    ClusterRegionFilter,
    ClusterRegionRule,
)
from napari_neuron_navigator.analysis.region_filter import PreparedClusterRegionFilter


def test_query_ccf_soma_coordinates_counts_retained_source_rows(tmp_path) -> None:
    path = tmp_path / "somas.parquet"
    pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n1", "n2", "n2"],
            "type": [1, 1, 3, 1, 3],
            "x": [10.0, 20.0, 30.0, 80.0, 80.0],
            "y": [10.0, 20.0, 30.0, 80.0, 80.0],
            "z": [10.0, 20.0, 30.0, 80.0, 80.0],
        }
    ).to_parquet(path, index=False)

    ids, coords, node_count = clustering.query_ccf_soma_coordinates(
        str(path),
        resolution=25.0,
        file_ids=["n1"],
    )

    assert ids == ["n1"]
    np.testing.assert_allclose(coords, [[15.0, 15.0, 15.0]])
    assert node_count == 2


def test_query_ccf_soma_coordinates_applies_optional_region_map(tmp_path) -> None:
    path = tmp_path / "somas.parquet"
    pd.DataFrame(
        {
            "file_id": ["inside", "outside"],
            "type": [1, 1],
            "x": [10.0, 80.0],
            "y": [10.0, 80.0],
            "z": [10.0, 80.0],
        }
    ).to_parquet(path, index=False)
    voxel_id_map = np.full((4, 4, 4), -1, dtype=np.int32)
    voxel_id_map[0, 0, 0] = 0

    ids, _coords, node_count = clustering.query_ccf_soma_coordinates(
        str(path),
        resolution=25.0,
        voxel_id_map=voxel_id_map,
    )

    assert ids == ["inside"]
    assert node_count == 1


def test_query_ccf_soma_coordinates_applies_include_and_per_rule_exclusion(
    tmp_path,
) -> None:
    path = tmp_path / "somas.parquet"
    pd.DataFrame(
        {
            "file_id": ["a", "a", "a", "b", "b", "c"],
            "neuron_id": [1, 1, 1, 1, 1, 2],
            "node_id": [1, 7, 8, 1, 7, 1],
            "type": [1, 99, 99, 1, 99, 1],
            "x": [10.0, 10.0, 10.0, 10.0, 10.0, 80.0],
            "y": [10.0, 10.0, 10.0, 10.0, 10.0, 80.0],
            "z": [10.0, 10.0, 10.0, 10.0, 10.0, 80.0],
        }
    ).to_parquet(path, index=False)
    include = np.zeros((4, 4, 4), dtype=bool)
    include[0, 0, 0] = True
    exclude = include.copy()
    region_filter = ClusterRegionFilter(
        include_rules=(ClusterRegionRule(region_id=10, acronym="INC"),),
        exclude_rules=(
            ClusterExclusionRule(
                region_id=20,
                acronym="EXC",
                node_types=(99,),
                minimum_node_count=2,
            ),
        ),
    )
    prepared = PreparedClusterRegionFilter(
        region_filter=region_filter,
        resolution_um=(25.0, 25.0, 25.0),
        atlas_shape=include.shape,
        include_mask=include,
        exclude_masks=(exclude,),
        exclude_mask=exclude,
    )

    ids, coords, node_count = clustering.query_ccf_soma_coordinates(
        str(path),
        resolution=25.0,
        prepared_region_filter=prepared,
    )

    assert ids == ["b"]
    np.testing.assert_allclose(coords, [[10.0, 10.0, 10.0]])
    assert node_count == 1


def test_cluster_somas_hierarchical_uses_condensed_distances_for_linkage(
    monkeypatch,
) -> None:
    """Ward soma clustering should link directly from condensed distances."""
    calls: dict[str, object] = {}

    def fake_pdist(coords, metric):
        calls["pdist_shape"] = tuple(coords.shape)
        calls["pdist_metric"] = metric
        return np.array([10.0, 20.0, 30.0], dtype=np.float64)

    def fake_linkage(values, method):
        calls["linkage_ndim"] = int(np.asarray(values).ndim)
        calls["linkage_dtype"] = np.asarray(values).dtype
        calls["linkage_method"] = method
        return np.array(
            [[0.0, 1.0, 10.0, 2.0], [2.0, 3.0, 20.0, 3.0]], dtype=np.float64
        )

    def fake_squareform(values, checks=False):
        calls["squareform_ndim"] = int(np.asarray(values).ndim)
        calls["squareform_checks"] = checks
        return np.array(
            [
                [0.0, 10.0, 20.0],
                [10.0, 0.0, 30.0],
                [20.0, 30.0, 0.0],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(clustering, "pdist", fake_pdist)
    monkeypatch.setattr(clustering, "linkage", fake_linkage)
    monkeypatch.setattr(clustering, "squareform", fake_squareform)
    monkeypatch.setattr(
        clustering,
        "fcluster",
        lambda _linkage_matrix, t, criterion: np.array([1, 1, 2], dtype=np.int32),
    )
    monkeypatch.setattr(
        clustering,
        "leaves_list",
        lambda _linkage_matrix: np.array([2, 1, 0], dtype=np.intp),
    )

    result = clustering.cluster_somas_hierarchical(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        ["n1", "n2", "n3"],
        method="ward",
        n_clusters=2,
    )

    assert calls["pdist_shape"] == (3, 3)
    assert calls["pdist_metric"] == "euclidean"
    assert calls["linkage_ndim"] == 1
    assert calls["linkage_method"] == "ward"
    assert calls["squareform_ndim"] == 1
    assert calls["squareform_checks"] is False
    assert result.distance_matrix.shape == (3, 3)
    assert result.reorder_indices.tolist() == [2, 1, 0]
    assert result.labels.tolist() == [1, 1, 2]
