"""End-to-end tests for parquet-driven flatmap-space clustering.

These exercise the real DuckDB path (no rendered heatmap and no mocking of the
analysis functions), covering both voxel correlation and soma-distance
clustering in flat map + depth space.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from napari_neuron_navigator.analysis.clustering import (
    ClusterExclusionRule,
    ClusterRegionFilter,
    ClusterRegionRule,
)
from napari_neuron_navigator.analysis.flatmap_correlation import (
    compute_flatmap_voxel_correlation_from_parquet,
    count_flatmap_voxel_correlation_nodes,
    query_flatmap_soma_coordinates,
    query_flatmap_soma_coordinates_and_count,
)
from napari_neuron_navigator.analysis.region_filter import PreparedClusterRegionFilter
from napari_neuron_navigator.flatmap_parquet import read_flatmap_parquet_transform_info

_V3_COLUMNS = (
    "file_id",
    "node_id",
    "parent_id",
    "type",
    "x",
    "y",
    "z",
    "x_flat_shaped",
    "y_flat_shaped",
    "flatmap_shaped_valid",
    "flatmap_shaped_projection_valid",
    "flatmap_shaped_invalid_code",
    "flatmap_shaped_lookup_mode",
    "x_flat_square",
    "y_flat_square",
    "flatmap_square_valid",
    "flatmap_square_projection_valid",
    "flatmap_square_invalid_code",
    "flatmap_square_lookup_mode",
    "depth_um",
    "depth_valid",
    "depth_invalid_code",
    "depth_lookup_mode",
)


def _v3_augmented_frame(
    *,
    n_neurons: int = 4,
    nodes_per_neuron: int = 25,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a version-3-style neuron frame with the full augmented columns."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {name: [] for name in _V3_COLUMNS}
    node_id = 0
    for neuron in range(n_neurons):
        # Cluster each neuron's nodes around a distinct flatmap centroid so
        # correlation and soma distance are both well defined.
        cx = 10.0 + neuron * 20.0
        cy = 10.0 + neuron * 15.0
        cd = 100.0 + neuron * 120.0
        for local in range(nodes_per_neuron):
            xf = float(cx + rng.uniform(-5.0, 5.0))
            yf = float(cy + rng.uniform(-5.0, 5.0))
            depth = float(cd + rng.uniform(-20.0, 20.0))
            node_type = 1 if local == 0 else 3
            rows["file_id"].append(f"neuron_{neuron}")
            rows["node_id"].append(node_id)
            rows["parent_id"].append(-1 if local == 0 else node_id - 1)
            rows["type"].append(node_type)
            rows["x"].append(xf * 10.0)
            rows["y"].append(yf * 10.0)
            rows["z"].append(depth)
            rows["x_flat_shaped"].append(xf)
            rows["y_flat_shaped"].append(yf)
            rows["flatmap_shaped_valid"].append(True)
            rows["flatmap_shaped_projection_valid"].append(True)
            rows["flatmap_shaped_invalid_code"].append(0)
            rows["flatmap_shaped_lookup_mode"].append("direct")
            rows["x_flat_square"].append(xf + 1.0)
            rows["y_flat_square"].append(yf + 1.0)
            rows["flatmap_square_valid"].append(True)
            rows["flatmap_square_projection_valid"].append(True)
            rows["flatmap_square_invalid_code"].append(0)
            rows["flatmap_square_lookup_mode"].append("direct")
            rows["depth_um"].append(depth)
            rows["depth_valid"].append(True)
            rows["depth_invalid_code"].append(0)
            rows["depth_lookup_mode"].append("direct")
            node_id += 1
    frame = pd.DataFrame(rows)
    frame["node_id"] = frame["node_id"].astype(np.int32)
    frame["parent_id"] = frame["parent_id"].astype(np.int32)
    frame["type"] = frame["type"].astype(np.int32)
    for name in (
        "x",
        "y",
        "z",
        "x_flat_shaped",
        "y_flat_shaped",
        "x_flat_square",
        "y_flat_square",
        "depth_um",
    ):
        frame[name] = frame[name].astype(np.float32)
    return frame


@pytest.fixture()
def flatmap_parquet(tmp_path):
    frame = _v3_augmented_frame()
    path = tmp_path / "neurons_v3.parquet"
    frame.to_parquet(path, index=False)
    return frame, str(path)


def _fake_atlas():
    return types.SimpleNamespace(
        atlas_name="test_atlas",
        resolution=(25.0, 25.0, 25.0),
    )


def test_transform_info_detects_v3_styles(flatmap_parquet) -> None:
    _frame, path = flatmap_parquet
    info = read_flatmap_parquet_transform_info(path)
    assert set(info.available_styles) == {"both_shaped", "both_square"}
    assert info.has_v3_depth is True


def test_compute_flatmap_voxel_correlation_from_parquet(flatmap_parquet) -> None:
    _frame, path = flatmap_parquet
    result, count_data, provenance = compute_flatmap_voxel_correlation_from_parquet(
        path,
        style="both_shaped",
        y_bins=32,
        depth_bin_um=50.0,
        n_clusters=2,
    )
    assert len(result.neuron_ids) == 4
    assert len(result.labels) == 4
    assert count_data.count_matrix.shape[0] == 4
    assert count_data.count_matrix.shape[1] == len(count_data.voxel_ids)
    assert count_data.rendered_node_count > 0
    assert provenance.style == "both_shaped"
    assert provenance.y_bins == 32
    # x is derived from the style's own bounds, so it is not 32.
    assert provenance.x_bins == 41
    assert provenance.volume_shape[-2:] == (32, 41)
    assert provenance.volume_shape[1] == 32
    assert result.metadata is None  # metadata attached by the worker, not here


def _exclude_first_neuron_filter() -> PreparedClusterRegionFilter:
    mask = np.zeros((100, 100, 100), dtype=bool)
    mask[:8, :8, :8] = True
    rule = ClusterExclusionRule(
        region_id=10,
        acronym="TEST",
        represented_region_ids=(10,),
        represented_region_acronyms=("TEST",),
        node_types=(3,),
        minimum_node_count=1,
    )
    return PreparedClusterRegionFilter(
        region_filter=ClusterRegionFilter(exclude_rules=(rule,)),
        resolution_um=(25.0, 25.0, 25.0),
        atlas_shape=mask.shape,
        include_mask=None,
        exclude_masks=(mask,),
        exclude_mask=mask,
    )


def test_flatmap_voxel_filter_uses_ccf_coordinates_before_binning(
    flatmap_parquet,
) -> None:
    frame, path = flatmap_parquet
    prepared = _exclude_first_neuron_filter()

    count = count_flatmap_voxel_correlation_nodes(
        path,
        style="both_shaped",
        y_bins=32,
        depth_bin_um=50.0,
        prepared_region_filter=prepared,
    )
    result, count_data, _provenance = compute_flatmap_voxel_correlation_from_parquet(
        path,
        style="both_shaped",
        y_bins=32,
        depth_bin_um=50.0,
        n_clusters=2,
        prepared_region_filter=prepared,
    )

    expected = int((frame["file_id"] != "neuron_0").sum())
    assert count == expected
    assert count_data.rendered_node_count == expected
    assert result.neuron_ids == ["neuron_1", "neuron_2", "neuron_3"]
    assert result.unassigned_neuron_ids == ["neuron_0"]


def test_flatmap_soma_filter_rejects_neuron_by_morphology_rule(
    flatmap_parquet,
) -> None:
    _frame, path = flatmap_parquet
    prepared = _exclude_first_neuron_filter()

    ids, coords, count = query_flatmap_soma_coordinates_and_count(
        path,
        style="both_shaped",
        prepared_region_filter=prepared,
    )

    assert ids == ["neuron_1", "neuron_2", "neuron_3"]
    assert coords.shape == (3, 3)
    assert count == 3


def test_flatmap_soma_exclusion_counts_nodes_without_flatmap_projection(
    tmp_path,
) -> None:
    frame = _v3_augmented_frame()
    morphology = (frame["file_id"] == "neuron_0") & (frame["type"] == 3)
    frame.loc[morphology, "flatmap_shaped_valid"] = False
    frame.loc[morphology, "flatmap_shaped_projection_valid"] = False
    path = tmp_path / "invalid_morphology_projection.parquet"
    frame.to_parquet(path, index=False)

    mask = np.zeros((100, 100, 100), dtype=bool)
    mask[:8, :8, :8] = True
    rule = ClusterExclusionRule(
        region_id=10,
        acronym="TEST",
        node_types=(3,),
        minimum_node_count=int(morphology.sum()),
    )
    prepared = PreparedClusterRegionFilter(
        region_filter=ClusterRegionFilter(exclude_rules=(rule,)),
        resolution_um=(25.0, 25.0, 25.0),
        atlas_shape=mask.shape,
        include_mask=None,
        exclude_masks=(mask,),
        exclude_mask=mask,
    )

    ids, _coords, count = query_flatmap_soma_coordinates_and_count(
        str(path),
        style="both_shaped",
        prepared_region_filter=prepared,
    )

    assert ids == ["neuron_1", "neuron_2", "neuron_3"]
    assert count == 3


def test_flatmap_soma_include_uses_all_soma_ccf_rows(tmp_path) -> None:
    frame = _v3_augmented_frame()
    extra_soma = frame[
        (frame["file_id"] == "neuron_0") & (frame["type"] == 1)
    ].copy()
    extra_soma.loc[:, "node_id"] = int(frame["node_id"].max()) + 1
    extra_soma.loc[:, ["x", "y", "z"]] = 1_000.0
    extra_soma.loc[:, "flatmap_shaped_valid"] = False
    extra_soma.loc[:, "flatmap_shaped_projection_valid"] = False
    frame = pd.concat([frame, extra_soma], ignore_index=True)
    path = tmp_path / "multiple_soma_rows.parquet"
    frame.to_parquet(path, index=False)

    include = np.zeros((100, 100, 100), dtype=bool)
    include[:8, :8, :8] = True
    rule = ClusterRegionRule(region_id=10, acronym="TEST")
    prepared = PreparedClusterRegionFilter(
        region_filter=ClusterRegionFilter(include_rules=(rule,)),
        resolution_um=(25.0, 25.0, 25.0),
        atlas_shape=include.shape,
        include_mask=include,
        exclude_masks=(),
        exclude_mask=None,
    )

    ids, coords, count = query_flatmap_soma_coordinates_and_count(
        str(path),
        style="both_shaped",
        file_ids=["neuron_0"],
        prepared_region_filter=prepared,
    )

    assert ids == []
    assert coords.shape == (0, 3)
    assert count == 0


def _layered_frame() -> pd.DataFrame:
    """Build neuron pairs that share a flat map footprint across depths.

    ``neuron_0``/``neuron_1`` occupy identical flat map positions but sit in
    different depth bins, and ``neuron_2``/``neuron_3`` do the same at a
    different footprint.  Uncollapsed, each pair occupies disjoint voxels; once
    depth is collapsed, each pair's count vectors become identical.
    """
    footprints = {
        "neuron_0": [((0.1, 0.1), 10), ((0.9, 0.9), 5)],
        "neuron_1": [((0.1, 0.1), 10), ((0.9, 0.9), 5)],
        "neuron_2": [((0.1, 0.9), 10), ((0.9, 0.1), 5)],
        "neuron_3": [((0.1, 0.9), 10), ((0.9, 0.1), 5)],
    }
    depths = {
        "neuron_0": 100.0,
        "neuron_1": 900.0,
        "neuron_2": 100.0,
        "neuron_3": 900.0,
    }
    rows: dict[str, list] = {name: [] for name in _V3_COLUMNS}
    node_id = 0
    for file_id, places in footprints.items():
        depth = depths[file_id]
        local = 0
        for (xf, yf), repeats in places:
            for _ in range(repeats):
                rows["file_id"].append(file_id)
                rows["node_id"].append(node_id)
                rows["parent_id"].append(-1 if local == 0 else node_id - 1)
                rows["type"].append(1 if local == 0 else 3)
                rows["x"].append(xf * 1000.0)
                rows["y"].append(yf * 1000.0)
                rows["z"].append(depth)
                rows["x_flat_shaped"].append(xf)
                rows["y_flat_shaped"].append(yf)
                rows["flatmap_shaped_valid"].append(True)
                rows["flatmap_shaped_projection_valid"].append(True)
                rows["flatmap_shaped_invalid_code"].append(0)
                rows["flatmap_shaped_lookup_mode"].append("direct")
                rows["x_flat_square"].append(xf + 1.0)
                rows["y_flat_square"].append(yf)
                rows["flatmap_square_valid"].append(True)
                rows["flatmap_square_projection_valid"].append(True)
                rows["flatmap_square_invalid_code"].append(0)
                rows["flatmap_square_lookup_mode"].append("direct")
                rows["depth_um"].append(depth)
                rows["depth_valid"].append(True)
                rows["depth_invalid_code"].append(0)
                rows["depth_lookup_mode"].append("direct")
                node_id += 1
                local += 1
    frame = pd.DataFrame(rows)
    frame["node_id"] = frame["node_id"].astype(np.int32)
    frame["parent_id"] = frame["parent_id"].astype(np.int32)
    frame["type"] = frame["type"].astype(np.int32)
    for name in (
        "x",
        "y",
        "z",
        "x_flat_shaped",
        "y_flat_shaped",
        "x_flat_square",
        "y_flat_square",
        "depth_um",
    ):
        frame[name] = frame[name].astype(np.float32)
    return frame


@pytest.fixture()
def layered_parquet(tmp_path):
    frame = _layered_frame()
    path = tmp_path / "layered_v3.parquet"
    frame.to_parquet(path, index=False)
    return frame, str(path)


def test_collapse_depth_keeps_node_count_and_shrinks_the_grid(
    layered_parquet,
) -> None:
    """Collapsing removes the depth axis without changing which nodes count."""
    _frame, path = layered_parquet
    kwargs = dict(style="both_shaped", y_bins=8, depth_bin_um=50.0, n_clusters=2)

    _r3, counts_3d, prov_3d = compute_flatmap_voxel_correlation_from_parquet(
        path, **kwargs
    )
    _r2, counts_2d, prov_2d = compute_flatmap_voxel_correlation_from_parquet(
        path, collapse_depth=True, **kwargs
    )

    assert prov_3d.collapse_depth is False
    assert prov_2d.collapse_depth is True
    assert len(prov_3d.volume_shape) == 3
    assert prov_2d.volume_shape == (8, 8)

    # Same nodes counted either way -- only their grouping changes.
    assert counts_2d.rendered_node_count == counts_3d.rendered_node_count
    # Merging depth planes can only reduce the number of distinct voxels.
    assert len(counts_2d.voxel_ids) < len(counts_3d.voxel_ids)


def test_collapse_depth_merges_layers_at_one_flatmap_position(
    layered_parquet,
) -> None:
    """Neurons sharing a footprint across layers must correlate once collapsed.

    Uncollapsed they occupy disjoint voxels, so the correlation cannot see that
    they cover the same part of the cortical sheet.
    """
    from napari_neuron_navigator.analysis.flatmap_correlation import (
        pearson_correlation_from_counts,
    )

    _frame, path = layered_parquet
    kwargs = dict(style="both_shaped", y_bins=8, depth_bin_um=50.0, n_clusters=2)

    _r3, counts_3d, _p3 = compute_flatmap_voxel_correlation_from_parquet(path, **kwargs)
    result_2d, counts_2d, _p2 = compute_flatmap_voxel_correlation_from_parquet(
        path, collapse_depth=True, **kwargs
    )

    def _r(count_data, matrix, left, right):
        i = count_data.neuron_ids.index(left)
        j = count_data.neuron_ids.index(right)
        return float(matrix[i, j])

    corr_3d = pearson_correlation_from_counts(counts_3d.count_matrix)
    corr_2d = pearson_correlation_from_counts(counts_2d.count_matrix)

    # Same footprint, different layers: invisible in 3-D, identical in 2-D.
    assert _r(counts_3d, corr_3d, "neuron_0", "neuron_1") < 0.5
    assert _r(counts_2d, corr_2d, "neuron_0", "neuron_1") == pytest.approx(1.0)
    assert _r(counts_2d, corr_2d, "neuron_2", "neuron_3") == pytest.approx(1.0)
    # Different footprints stay distinguishable.
    assert _r(counts_2d, corr_2d, "neuron_0", "neuron_2") < 0.5

    labels = dict(zip(result_2d.neuron_ids, result_2d.labels.tolist()))
    assert labels["neuron_0"] == labels["neuron_1"]
    assert labels["neuron_2"] == labels["neuron_3"]
    assert labels["neuron_0"] != labels["neuron_2"]


def test_collapse_depth_preflight_count_matches_the_run(layered_parquet) -> None:
    """The preflight count must not change when the depth axis is collapsed."""
    _frame, path = layered_parquet
    kwargs = dict(style="both_shaped", y_bins=8, depth_bin_um=50.0)

    flat = count_flatmap_voxel_correlation_nodes(path, collapse_depth=True, **kwargs)
    deep = count_flatmap_voxel_correlation_nodes(path, collapse_depth=False, **kwargs)
    assert flat == deep

    _result, counts, _prov = compute_flatmap_voxel_correlation_from_parquet(
        path, collapse_depth=True, n_clusters=2, **kwargs
    )
    assert counts.rendered_node_count == flat


def test_count_matrix_rejects_3d_shape_without_depth_bins() -> None:
    """A 3-D volume needs a depth_bin column; the rank decides, not the column."""
    from napari_neuron_navigator.analysis.flatmap_correlation import (
        build_flatmap_count_matrix_from_bin_counts,
    )

    collapsed = pd.DataFrame(
        {
            "file_id": ["a", "b"],
            "y_bin": [0, 1],
            "x_bin": [0, 1],
            "node_count": [3, 4],
        }
    )
    with pytest.raises(ValueError, match="missing required column"):
        build_flatmap_count_matrix_from_bin_counts(
            collapsed,
            (4, 4, 4),
            input_file_ids=("a", "b"),
        )

    # The same counts are valid against a 2-D volume.
    matrix = build_flatmap_count_matrix_from_bin_counts(
        collapsed,
        (4, 4),
        input_file_ids=("a", "b"),
    )
    assert matrix.count_matrix.shape == (2, 2)
    assert matrix.rendered_node_count == 7


def test_count_matrix_rejects_unsupported_volume_rank() -> None:
    from napari_neuron_navigator.analysis.flatmap_correlation import (
        build_flatmap_count_matrix_from_bin_counts,
    )

    with pytest.raises(ValueError, match="must be 2D or 3D"):
        build_flatmap_count_matrix_from_bin_counts(
            pd.DataFrame(),
            (4,),
            input_file_ids=(),
        )


def test_compute_flatmap_correlation_respects_file_id_subset(
    flatmap_parquet,
) -> None:
    _frame, path = flatmap_parquet
    result, _count, _prov = compute_flatmap_voxel_correlation_from_parquet(
        path,
        style="both_square",
        y_bins=16,
        depth_bin_um=75.0,
        n_clusters=2,
        file_ids=["neuron_0", "neuron_1", "neuron_2"],
    )
    assert set(result.neuron_ids) == {"neuron_0", "neuron_1", "neuron_2"}


def test_query_flatmap_soma_coordinates(flatmap_parquet) -> None:
    _frame, path = flatmap_parquet
    ids, coords = query_flatmap_soma_coordinates(path, style="both_shaped")
    assert ids == ["neuron_0", "neuron_1", "neuron_2", "neuron_3"]
    assert coords.shape == (4, 3)
    # Soma centroids should be ordered and well separated along each axis.
    assert np.all(np.diff(coords[:, 0]) > 0)
    assert np.all(np.diff(coords[:, 2]) > 0)


def test_flatmap_preflight_counts_exact_contributing_nodes(flatmap_parquet) -> None:
    frame, path = flatmap_parquet

    assert (
        count_flatmap_voxel_correlation_nodes(
            path,
            style="both_shaped",
            y_bins=32,
            depth_bin_um=50.0,
            file_ids=["neuron_0", "neuron_1"],
        )
        == 2 * 25
    )
    ids, coords, soma_node_count = query_flatmap_soma_coordinates_and_count(
        path,
        style="both_shaped",
        file_ids=["neuron_0", "neuron_1"],
    )
    assert ids == ["neuron_0", "neuron_1"]
    assert coords.shape == (2, 3)
    assert soma_node_count == int(
        frame[
            frame["file_id"].isin(["neuron_0", "neuron_1"]) & (frame["type"] == 1)
        ].shape[0]
    )


def test_flatmap_parquet_correlation_worker(flatmap_parquet) -> None:
    from napari_neuron_navigator import workers

    _frame, path = flatmap_parquet
    finished: list = []
    errors: list = []
    worker = workers.FlatmapParquetCorrelationWorker(
        parquet_path=path,
        atlas=_fake_atlas(),
        style="both_shaped",
        y_bins=32,
        depth_bin_um=50.0,
        n_clusters=2,
    )
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.run()

    assert errors == []
    assert len(finished) == 1
    result = finished[0]
    assert len(result.neuron_ids) == 4
    assert result.metadata is not None
    assert result.metadata.analysis_method == "flatmap_voxel_correlation"


def test_flatmap_correlation_worker_records_collapse_provenance(
    layered_parquet,
) -> None:
    """A collapsed run must be identifiable from its recorded metadata."""
    from napari_neuron_navigator import workers

    _frame, path = layered_parquet
    finished: list = []
    errors: list = []
    worker = workers.FlatmapParquetCorrelationWorker(
        parquet_path=path,
        atlas=_fake_atlas(),
        style="both_shaped",
        y_bins=8,
        depth_bin_um=50.0,
        n_clusters=2,
        collapse_depth=True,
    )
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.run()

    assert errors == []
    assert len(finished) == 1
    result = finished[0]
    assert result.metadata is not None
    assert result.metadata.analysis_method == "flatmap_voxel_correlation"
    assert result.metadata.distance_metric == "one_minus_pearson_r_flatmap_xy"
    assert result.metadata.extra_metadata["flatmap_collapse_depth"] is True
    assert result.metadata.extra_metadata["flatmap_volume_shape"] == [8, 8]


def test_flatmap_soma_cluster_worker(flatmap_parquet) -> None:
    from napari_neuron_navigator import workers

    _frame, path = flatmap_parquet
    finished: list = []
    errors: list = []
    worker = workers.FlatmapSomaClusterWorker(
        parquet_path=path,
        atlas=_fake_atlas(),
        style="both_shaped",
        algorithm="hierarchical",
        linkage_method="ward",
        n_clusters=2,
    )
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.run()

    assert errors == []
    assert len(finished) == 1
    result = finished[0]
    assert len(result.neuron_ids) == 4
    assert result.metadata is not None
    assert result.metadata.analysis_method == "flatmap_soma_location"
