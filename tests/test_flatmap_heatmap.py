from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from napari_neuron_navigator.flatmap_heatmap import (
    ALLEN_LAYER_OUTPUT_PROJECTION,
    ALLEN_LAYER_OUTPUT_STACK,
    FLATMAP_HEATMAP_COLOR_CLUSTER,
    FLATMAP_HEATMAP_COLOR_INDIVIDUAL,
    FLATMAP_HEATMAP_COLOR_SINGLE,
    MAX_FLATMAP_HEATMAP_VOXELS,
    _bin_flat_values,
    build_allen_layer_cluster_volumes,
    build_allen_layer_file_id_volumes,
    build_allen_layer_heatmap_volume_result,
    build_allen_layer_render_from_projected_nodes,
    build_allen_layer_stack_from_projected_nodes,
    build_flatmap_cluster_volumes,
    build_flatmap_file_id_volumes,
    build_flatmap_heatmap_volume_result,
    build_flatmap_render_data,
    build_flatmap_render_data_from_projected_nodes,
    build_flatmap_segment_vectors,
    compute_depth_range,
    compute_flatmap_bounds_from_parquet,
    compute_flatmap_lookup_stats,
    compute_flatmap_xy_bounds,
    depth_plane_labels,
    flatmap_pixel_coordinates,
    rendered_plane_points,
)
from napari_neuron_navigator.isocortex_layers import AllenIsocortexLayerMap


def _lookup_volumes() -> tuple[np.ndarray, np.ndarray]:
    flatmap = np.zeros((2, 2, 2, 2), dtype=float)
    flatmap[..., 0] = np.asarray([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, -1.0]]])
    flatmap[..., 1] = np.asarray([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, -1.0]]])
    depth = np.asarray([[[0.0, 25.0], [50.0, 75.0]], [[100.0, 125.0], [150.0, -1.0]]])
    return flatmap, depth


def _projected_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "b.swc", "c.swc", "d.swc"],
            "x_flat": [0.0, 0.1, 6.0, 2.0, -1.0],
            "y_flat": [0.0, 0.1, 6.0, 2.0, -1.0],
            "depth_um": [0.0, 10.0, -1.0, 50.0, 10.0],
            "flatmap_valid": [True, True, True, True, False],
            "depth_valid": [True, True, False, True, True],
        }
    )


def _legacy_flatmap_xy_bounds(
    flatmap: np.ndarray,
    *,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
) -> tuple[tuple[float, float], tuple[float, float]]:
    flat_xy = np.asarray(flatmap, dtype=float).reshape(-1, 2)
    valid = np.all(np.isfinite(flat_xy), axis=1)
    if invalid_negative_one_sentinel:
        valid &= ~((flat_xy[:, 0] == -1.0) & (flat_xy[:, 1] == -1.0))
    if invalid_zero_sentinel:
        valid &= ~((flat_xy[:, 0] == 0.0) & (flat_xy[:, 1] == 0.0))
    valid_xy = flat_xy[valid]
    return (
        (float(np.min(valid_xy[:, 0])), float(np.max(valid_xy[:, 0]))),
        (float(np.min(valid_xy[:, 1])), float(np.max(valid_xy[:, 1]))),
    )


def _legacy_depth_range(depth: np.ndarray) -> tuple[float, float]:
    values = np.asarray(depth, dtype=float)
    valid_values = values[np.isfinite(values) & (values >= 0.0)]
    lower = float(np.min(valid_values))
    upper = float(np.max(valid_values))
    return lower, upper if upper > lower else lower + 1.0


def test_compute_bounds_ignore_negative_one_sentinel() -> None:
    flatmap, depth = _lookup_volumes()

    assert compute_flatmap_xy_bounds(flatmap) == ((0.0, 6.0), (0.0, 6.0))
    assert compute_depth_range(depth) == (0.0, 150.0)


def test_chunked_lookup_stats_match_full_array_reference() -> None:
    flatmap, depth = _lookup_volumes()
    flatmap[0, 0, 1] = (np.nan, 99.0)
    flatmap[0, 1, 0] = (np.inf, 4.0)
    flatmap[1, 0, 1] = (0.0, 0.0)
    depth[0, 0, 0] = np.nan
    depth[0, 0, 1] = np.inf

    stats = compute_flatmap_lookup_stats(
        flatmap,
        depth,
        invalid_zero_sentinel=True,
        chunk_voxels=2,
    )

    expected_x_bounds, expected_y_bounds = _legacy_flatmap_xy_bounds(
        flatmap,
        invalid_zero_sentinel=True,
    )
    assert stats.x_bounds == expected_x_bounds
    assert stats.y_bounds == expected_y_bounds
    assert stats.depth_range_um == _legacy_depth_range(depth)
    assert stats.flatmap_valid_voxels == 3
    assert stats.depth_valid_voxels == 5


def test_chunked_lookup_stats_reject_all_invalid_flatmap() -> None:
    flatmap = np.full((2, 2, 2, 2), -1.0)
    depth = np.ones((2, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="valid x/y"):
        compute_flatmap_lookup_stats(flatmap, depth, chunk_voxels=2)


def test_chunked_lookup_stats_reject_all_invalid_depth() -> None:
    flatmap = np.ones((2, 2, 2, 2), dtype=float)
    depth = np.full((2, 2, 2), -1.0)

    with pytest.raises(ValueError, match="valid non-negative depths"):
        compute_flatmap_lookup_stats(flatmap, depth, chunk_voxels=2)


def test_flatmap_heatmap_includes_depth_minus_one_in_sentinel_plane() -> None:
    flatmap, depth = _lookup_volumes()

    result = build_flatmap_render_data(
        _projected_nodes(),
        flatmap,
        depth,
        y_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=True,
    )

    assert result.volume.shape == (8, 4, 4)
    assert result.summary.rendered_nodes == 4
    assert result.summary.depth_minus_one_nodes == 1
    assert result.summary.nonzero_voxels == 3
    assert result.projected_nodes["render_valid"].tolist() == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert result.projected_nodes["depth_bin"].tolist() == [1, 1, 0, 3, -1]
    assert result.projected_nodes["depth_bin_label"].tolist()[2] == "depth -1"
    assert result.volume[1, 0, 0] == 2.0
    assert result.volume[0, 3, 3] == 1.0
    assert result.volume[3, 1, 1] == 1.0
    np.testing.assert_array_equal(
        result.points,
        np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 3.0], [3.0, 1.0, 1.0]]
        ),
    )
    assert result.point_file_ids == ["a.swc", "a.swc", "b.swc", "c.swc"]


def test_flatmap_heatmap_excludes_depth_minus_one_when_requested() -> None:
    flatmap, depth = _lookup_volumes()

    result = build_flatmap_render_data(
        _projected_nodes(),
        flatmap,
        depth,
        y_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=False,
    )

    assert result.volume.shape == (7, 4, 4)
    assert result.summary.rendered_nodes == 3
    assert result.summary.excluded_depth_minus_one_nodes == 1
    assert result.projected_nodes["render_valid"].tolist() == [
        True,
        True,
        False,
        True,
        False,
    ]
    assert result.projected_nodes["depth_bin"].tolist() == [0, 0, -1, 2, -1]


def test_flatmap_heatmap_reuses_precomputed_lookup_stats() -> None:
    flatmap, depth = _lookup_volumes()
    stats = compute_flatmap_lookup_stats(flatmap, depth, chunk_voxels=2)

    expected = build_flatmap_render_data(
        _projected_nodes(),
        flatmap,
        depth,
        y_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=True,
    )
    actual = build_flatmap_render_data(
        _projected_nodes(),
        flatmap,
        depth,
        y_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=True,
        lookup_stats=stats,
    )

    np.testing.assert_array_equal(actual.volume, expected.volume)
    np.testing.assert_array_equal(actual.points, expected.points)
    assert actual.point_file_ids == expected.point_file_ids
    assert actual.summary.to_dict() == expected.summary.to_dict()
    pd.testing.assert_frame_equal(actual.projected_nodes, expected.projected_nodes)


def test_flatmap_heatmap_from_projected_nodes_uses_projected_bounds() -> None:
    result = build_flatmap_render_data_from_projected_nodes(
        _projected_nodes(),
        y_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=True,
    )

    assert result.volume.shape == (4, 4, 4)
    assert result.summary.x_flat_min == 0.0
    assert result.summary.x_flat_max == 6.0
    assert result.summary.depth_min_um == 0.0
    assert result.summary.depth_max_um == 50.0
    assert result.summary.rendered_nodes == 4
    assert result.summary.depth_minus_one_nodes == 1
    assert result.projected_nodes["render_valid"].tolist() == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert result.projected_nodes["depth_bin"].tolist() == [1, 1, 0, 3, -1]
    assert result.volume[1, 0, 0] == 2.0
    assert result.volume[0, 3, 3] == 1.0
    assert result.volume[3, 1, 1] == 1.0


def test_flatmap_heatmap_from_projected_nodes_infers_validity_flags() -> None:
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "b.swc", "c.swc"],
            "x_flat": [0.0, np.nan, 2.0],
            "y_flat": [0.0, 1.0, 2.0],
            "depth_um": [0.0, 10.0, -1.0],
        }
    )

    result = build_flatmap_render_data_from_projected_nodes(
        projected,
        y_bins=2,
        depth_bin_um=25.0,
        include_depth_minus_one=True,
    )

    assert result.summary.rendered_nodes == 2
    assert result.summary.flatmap_valid_nodes == 2
    assert result.summary.depth_valid_nodes == 1
    assert result.projected_nodes["flatmap_valid"].tolist() == [True, False, True]
    assert result.projected_nodes["depth_valid"].tolist() == [True, True, False]
    assert result.projected_nodes["render_valid"].tolist() == [True, False, True]


def test_precomputed_subset_uses_canonical_bounds_for_cache_alignment() -> None:
    full = pd.DataFrame(
        {
            "file_id": ["left.swc", "right.swc"],
            "x_flat": [10.0, 90.0],
            "y_flat": [10.0, 90.0],
            "depth_um": [10.0, 90.0],
            "flatmap_valid": [True, True],
            "depth_valid": [True, True],
        }
    )
    canonical = {
        "x_bounds": (0.0, 100.0),
        "y_bounds": (0.0, 100.0),
        "depth_range_um": (0.0, 100.0),
    }
    complete = build_flatmap_render_data_from_projected_nodes(
        full,
        y_bins=10,
        depth_bin_um=10.0,
        include_depth_minus_one=False,
        **canonical,
    )
    subset = build_flatmap_render_data_from_projected_nodes(
        full.iloc[[1]],
        y_bins=10,
        depth_bin_um=10.0,
        include_depth_minus_one=False,
        **canonical,
    )

    assert subset.volume.shape == complete.volume.shape == (11, 10, 10)
    row = subset.projected_nodes.iloc[0]
    assert (int(row.depth_bin), int(row.y_flat_bin), int(row.x_flat_bin)) == (
        9,
        9,
        9,
    )


def _binned_projected_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "b.swc", "c.swc", "skip.swc"],
            "render_valid": [True, True, True, True, False],
            "depth_bin": [0, 0, 1, 1, 0],
            "y_flat_bin": [1, 1, 2, 0, 0],
            "x_flat_bin": [2, 2, 0, 1, 0],
        }
    )


def test_build_flatmap_file_id_volumes_splits_rendered_node_counts() -> None:
    groups = build_flatmap_file_id_volumes(
        _binned_projected_nodes(),
        (2, 3, 3),
    )

    assert [group.label for group in groups] == ["a.swc", "b.swc", "c.swc"]
    assert [group.source_file_ids for group in groups] == [
        ("a.swc",),
        ("b.swc",),
        ("c.swc",),
    ]
    assert [group.rendered_nodes for group in groups] == [2, 1, 1]
    assert groups[0].volume[0, 1, 2] == 2.0
    assert groups[1].volume[1, 2, 0] == 1.0
    assert groups[2].volume[1, 0, 1] == 1.0
    assert sum(float(group.volume.sum()) for group in groups) == 4.0


def test_build_flatmap_cluster_volumes_groups_by_cluster_with_unclustered_last() -> (
    None
):
    groups = build_flatmap_cluster_volumes(
        _binned_projected_nodes(),
        (2, 3, 3),
        {"a.swc": 2, "b.swc": 1},
    )

    assert [group.group_key for group in groups] == [1, 2, None]
    assert [group.label for group in groups] == [
        "Cluster 1",
        "Cluster 2",
        "Unclustered",
    ]
    assert [group.source_file_ids for group in groups] == [
        ("b.swc",),
        ("a.swc",),
        ("c.swc",),
    ]
    assert groups[0].volume[1, 2, 0] == 1.0
    assert groups[1].volume[0, 1, 2] == 2.0
    assert groups[2].volume[1, 0, 1] == 1.0


def _flat_binned_projected_nodes() -> pd.DataFrame:
    """Rendered nodes as a depth-collapsed render writes them: no depth bin."""
    return pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "b.swc", "c.swc", "skip.swc"],
            "render_valid": [True, True, True, True, False],
            "y_flat_bin": [1, 1, 2, 0, 0],
            "x_flat_bin": [2, 2, 0, 1, 0],
        }
    )


@pytest.mark.parametrize("include", [False, True])
def test_collapsed_render_equals_depth_render_summed_over_planes(include) -> None:
    kwargs = dict(
        y_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=include,
        x_bounds=(0.0, 6.0),
        y_bounds=(0.0, 6.0),
        depth_range_um=(0.0, 150.0),
    )
    depth = build_flatmap_render_data_from_projected_nodes(
        _projected_nodes(),
        **kwargs,
    )
    flat = build_flatmap_render_data_from_projected_nodes(
        _projected_nodes(),
        collapse_depth=True,
        **kwargs,
    )

    assert depth.volume.ndim == 3
    assert flat.volume.shape == (4, 4)
    np.testing.assert_array_equal(flat.volume, depth.volume.sum(axis=0))
    # Collapsing removes the depth axis, never a node: the checkbox alone
    # decides which nodes render.
    assert flat.summary.rendered_nodes == depth.summary.rendered_nodes
    assert flat.summary.depth_valid_nodes == depth.summary.depth_valid_nodes
    assert flat.summary.depth_minus_one_nodes == depth.summary.depth_minus_one_nodes


def test_collapsed_render_reports_no_depth_binning() -> None:
    flat = build_flatmap_render_data_from_projected_nodes(
        _projected_nodes(),
        y_bins=4,
        depth_bin_um=25.0,
        collapse_depth=True,
        x_bounds=(0.0, 6.0),
        y_bounds=(0.0, 6.0),
        depth_range_um=(0.0, 150.0),
    )

    # Zeros mean "no depth axis", so no plane labels get invented for it.
    assert flat.summary.depth_bins == 0
    assert flat.summary.depth_bin_um == 0.0
    assert depth_plane_labels(flat.summary.to_dict()) == ()
    assert "depth_bin" not in flat.projected_nodes.columns
    assert "depth_bin_label" not in flat.projected_nodes.columns
    assert flat.points.shape[1] == 2


def test_collapsed_render_excludes_depth_minus_one_when_requested() -> None:
    included = build_flatmap_render_data_from_projected_nodes(
        _projected_nodes(),
        y_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=True,
        collapse_depth=True,
        x_bounds=(0.0, 6.0),
        y_bounds=(0.0, 6.0),
        depth_range_um=(0.0, 150.0),
    )
    excluded = build_flatmap_render_data_from_projected_nodes(
        _projected_nodes(),
        y_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=False,
        collapse_depth=True,
        x_bounds=(0.0, 6.0),
        y_bounds=(0.0, 6.0),
        depth_range_um=(0.0, 150.0),
    )

    # b.swc carries the only depth == -1 node in the fixture.
    assert included.summary.rendered_nodes == excluded.summary.rendered_nodes + 1
    assert float(included.volume.sum()) == float(excluded.volume.sum()) + 1.0


def test_collapsed_lookup_render_matches_summed_depth_volume() -> None:
    flatmap, depth_volume = _lookup_volumes()
    kwargs = dict(y_bins=4, depth_bin_um=25.0)
    depth = build_flatmap_render_data(
        _projected_nodes(), flatmap, depth_volume, **kwargs
    )
    flat = build_flatmap_render_data(
        _projected_nodes(), flatmap, depth_volume, collapse_depth=True, **kwargs
    )

    np.testing.assert_array_equal(flat.volume, depth.volume.sum(axis=0))


def test_flatmap_pixel_coordinates_center_on_image_pixels() -> None:
    bounds, bins = (0.0, 10.0), 16
    rng = np.random.default_rng(3)
    values = rng.uniform(*bounds, size=500)

    pixels = flatmap_pixel_coordinates(values, bounds, bins)
    bin_indices = _bin_flat_values(values, bounds, bins)

    # napari centers pixel k on coordinate k, so rounding a pixel coordinate
    # must return the bin that holds the value.  A missing 0.5 offset shifts
    # every overlay half a pixel off the heatmap it is drawn over.
    np.testing.assert_array_equal(np.floor(pixels + 0.5).astype(np.int64), bin_indices)

    bin_centers = (np.arange(bins) + 0.5) * (bounds[1] - bounds[0]) / bins
    np.testing.assert_allclose(
        flatmap_pixel_coordinates(bin_centers, bounds, bins),
        np.arange(bins, dtype=float),
    )


def test_flatmap_pixel_coordinates_clip_to_the_image_edges() -> None:
    pixels = flatmap_pixel_coordinates(
        np.asarray([-100.0, 100.0]),
        (0.0, 10.0),
        16,
    )

    np.testing.assert_allclose(pixels, [-0.5, 15.5])


def _segment_endpoints() -> tuple[np.ndarray, list[str]]:
    # (M, 2, 2): [parent, child] x (x_flat, y_flat), matching
    # build_projected_segments' output layout.
    data = np.asarray(
        [
            [[0.0, 0.0], [10.0, 20.0]],
            [[10.0, 20.0], [20.0, 40.0]],
        ],
        dtype=float,
    )
    return data, ["a.swc", "a.swc"]


def test_build_flatmap_segment_vectors_uses_row_col_start_direction() -> None:
    endpoints, file_ids = _segment_endpoints()
    bounds, bins = (0.0, 40.0), 8

    vectors = build_flatmap_segment_vectors(
        endpoints,
        file_ids,
        x_bounds=bounds,
        y_bounds=bounds,
        y_bins=bins,
        x_bins=bins,
    )

    assert vectors.data.shape == (2, 2, 2)
    assert vectors.data.dtype == np.float32
    assert vectors.total_segments == 2
    assert vectors.file_ids == ("a.swc", "a.swc")

    expected_start = np.column_stack(
        (
            flatmap_pixel_coordinates(endpoints[:, 0, 1], bounds, bins),
            flatmap_pixel_coordinates(endpoints[:, 0, 0], bounds, bins),
        )
    )
    expected_end = np.column_stack(
        (
            flatmap_pixel_coordinates(endpoints[:, 1, 1], bounds, bins),
            flatmap_pixel_coordinates(endpoints[:, 1, 0], bounds, bins),
        )
    )
    # Row leads column, and the second entry is a direction, not an endpoint.
    np.testing.assert_allclose(vectors.data[:, 0], expected_start, atol=1e-5)
    np.testing.assert_allclose(
        vectors.data[:, 0] + vectors.data[:, 1],
        expected_end,
        atol=1e-5,
    )


def test_build_flatmap_segment_vectors_refuses_above_the_limit() -> None:
    endpoints, file_ids = _segment_endpoints()

    with pytest.raises(ValueError, match="2D Vector mode would draw 2"):
        build_flatmap_segment_vectors(
            endpoints,
            file_ids,
            x_bounds=(0.0, 40.0),
            y_bounds=(0.0, 40.0),
            y_bins=8,
            x_bins=8,
            max_segments=1,
        )


def test_build_flatmap_segment_vectors_handles_an_empty_selection() -> None:
    vectors = build_flatmap_segment_vectors(
        np.empty((0, 2, 2), dtype=float),
        [],
        x_bounds=(0.0, 40.0),
        y_bounds=(0.0, 40.0),
        y_bins=8,
        x_bins=8,
    )

    assert vectors.data.shape == (0, 2, 2)
    assert vectors.file_ids == ()
    assert vectors.total_segments == 0


def test_build_flatmap_file_id_volumes_accepts_a_two_dimensional_shape() -> None:
    groups = build_flatmap_file_id_volumes(
        _flat_binned_projected_nodes(),
        (3, 3),
    )

    assert [group.label for group in groups] == ["a.swc", "b.swc", "c.swc"]
    assert [group.volume.shape for group in groups] == [(3, 3)] * 3
    assert [group.rendered_nodes for group in groups] == [2, 1, 1]
    assert groups[0].volume[1, 2] == 2.0
    assert groups[1].volume[2, 0] == 1.0
    assert groups[2].volume[0, 1] == 1.0


def test_build_flatmap_cluster_volumes_accepts_a_two_dimensional_shape() -> None:
    groups = build_flatmap_cluster_volumes(
        _flat_binned_projected_nodes(),
        (3, 3),
        {"a.swc": 2, "b.swc": 1},
    )

    assert [group.label for group in groups] == [
        "Cluster 1",
        "Cluster 2",
        "Unclustered",
    ]
    assert groups[0].volume[2, 0] == 1.0
    assert groups[1].volume[1, 2] == 2.0
    assert groups[2].volume[0, 1] == 1.0


def test_rendered_plane_points_reads_the_named_plane_column() -> None:
    points, file_ids = rendered_plane_points(
        _binned_projected_nodes(),
        plane_column="depth_bin",
    )

    np.testing.assert_array_equal(
        points,
        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1.0, 2.0, 0.0], [1.0, 0.0, 1.0]],
    )
    assert file_ids == ["a.swc", "a.swc", "b.swc", "c.swc"]


def test_rendered_plane_points_returns_two_columns_without_a_plane() -> None:
    points, file_ids = rendered_plane_points(
        _flat_binned_projected_nodes(),
        plane_column=None,
    )

    np.testing.assert_array_equal(
        points,
        [[1.0, 2.0], [1.0, 2.0], [2.0, 0.0], [0.0, 1.0]],
    )
    assert file_ids == ["a.swc", "a.swc", "b.swc", "c.swc"]


def test_rendered_plane_points_match_a_depth_render_points_array() -> None:
    render = build_flatmap_render_data_from_projected_nodes(
        _projected_nodes(),
        y_bins=4,
        depth_bin_um=25.0,
        x_bounds=(0.0, 6.0),
        y_bounds=(0.0, 6.0),
        depth_range_um=(0.0, 150.0),
    )

    points, file_ids = rendered_plane_points(
        render.projected_nodes,
        plane_column="depth_bin",
    )

    np.testing.assert_array_equal(points, render.points)
    assert file_ids == render.point_file_ids


def _allen_layer_map() -> AllenIsocortexLayerMap:
    region_ids = (101, 102, 103, 104, 105, 106)
    return AllenIsocortexLayerMap(
        atlas_name="allen_mouse_25um",
        isocortex_region_id=315,
        region_to_layer_index={
            region_id: index for index, region_id in enumerate(region_ids)
        },
        region_ids_by_layer=tuple((region_id,) for region_id in region_ids),
    )


def _allen_layer_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": [
                "a.swc",
                "a.swc",
                "b.swc",
                "b.swc",
                "c.swc",
                "d.swc",
                "e.swc",
                "f.swc",
            ],
            "x_flat": [0.0, 0.0, 9.9, 5.0, 2.0, 3.0, 4.0, np.nan],
            "y_flat": [0.0, 0.0, 9.9, 5.0, 2.0, 3.0, 4.0, 1.0],
            "flatmap_valid": [True, True, True, True, True, True, True, True],
            "region_id": [101, 101, 102, 104, 105, 106, 999, 103],
        }
    )


def test_build_allen_layer_stack_counts_and_excludes_nodes() -> None:
    result = build_allen_layer_stack_from_projected_nodes(
        _allen_layer_nodes(),
        _allen_layer_map(),
        y_bins=10,
        x_bounds=(0.0, 10.0),
        y_bounds=(0.0, 10.0),
    )

    assert result.volume.shape == (6, 10, 10)
    assert tuple(result.volume.sum(axis=(1, 2)).astype(int)) == (
        2,
        1,
        0,
        1,
        1,
        1,
    )
    assert result.volume[0, 0, 0] == 2.0
    assert result.volume[1, 9, 9] == 1.0
    assert result.summary.rendered_nodes == 6
    assert result.summary.excluded_nodes == 2
    assert result.summary.invalid_flatmap_nodes == 1
    assert result.summary.excluded_non_layer_nodes == 1
    assert result.summary.layer_node_counts == (2, 1, 0, 1, 1, 1)
    assert result.summary.to_dict()["excluded_nodes"] == 2
    assert result.projected_nodes["allen_layer_label"].tolist()[:4] == [
        "L1",
        "L1",
        "L2/3",
        "L5",
    ]
    assert result.projected_nodes["render_valid"].sum() == 6


def test_selected_allen_layers_build_compact_stack_and_exact_projection() -> None:
    stack = build_allen_layer_render_from_projected_nodes(
        _allen_layer_nodes(),
        _allen_layer_map(),
        y_bins=10,
        x_bounds=(0.0, 10.0),
        y_bounds=(0.0, 10.0),
        selected_layer_indices=[4, 1, 4],
        output_mode=ALLEN_LAYER_OUTPUT_STACK,
    )
    projection = build_allen_layer_render_from_projected_nodes(
        _allen_layer_nodes(),
        _allen_layer_map(),
        y_bins=10,
        x_bounds=(0.0, 10.0),
        y_bounds=(0.0, 10.0),
        selected_layer_indices=[1, 4],
        output_mode=ALLEN_LAYER_OUTPUT_PROJECTION,
    )

    assert stack.volume.shape == (2, 10, 10)
    assert stack.summary.selected_layer_indices == (1, 4)
    assert stack.summary.layer_labels == ("L2/3", "L6a")
    assert stack.summary.layer_node_counts == (1, 1)
    assert stack.summary.excluded_unselected_layer_nodes == 4
    assert stack.summary.excluded_non_layer_nodes == 1
    assert stack.summary.invalid_flatmap_nodes == 1
    assert stack.summary.excluded_nodes == 6
    assert stack.summary.output_shape == stack.volume.shape
    assert stack.summary.to_dict()["selected_layer_labels"] == ["L2/3", "L6a"]
    assert stack.summary.to_dict()["output_mode"] == "stack"
    assert projection.volume.shape == (10, 10)
    np.testing.assert_array_equal(projection.volume, stack.volume.sum(axis=0))

    table = stack.projected_nodes
    assert table.loc[table["allen_layer_index"] == 1, "allen_layer_selected"].all()
    assert table.loc[table["allen_layer_index"] == 4, "allen_layer_selected"].all()
    assert set(
        table.loc[table["render_valid"], "allen_layer_render_index"].tolist()
    ) == {0, 1}
    assert not table.loc[table["allen_layer_index"] == 0, "render_valid"].any()


def test_selected_allen_layer_render_rejects_an_empty_selection() -> None:
    with pytest.raises(ValueError, match="Select at least one Allen layer"):
        build_allen_layer_render_from_projected_nodes(
            _allen_layer_nodes(),
            _allen_layer_map(),
            y_bins=10,
            x_bounds=(0.0, 10.0),
            y_bounds=(0.0, 10.0),
            selected_layer_indices=[],
        )


def test_projected_allen_individual_groups_stay_separated_by_file_id() -> None:
    nodes = _allen_layer_nodes().iloc[:3].copy()
    nodes["neuron_id"] = "duplicate-display-id"
    result = build_allen_layer_render_from_projected_nodes(
        nodes,
        _allen_layer_map(),
        y_bins=10,
        x_bounds=(0.0, 10.0),
        y_bounds=(0.0, 10.0),
        selected_layer_indices=[0, 1],
        output_mode=ALLEN_LAYER_OUTPUT_PROJECTION,
    )

    groups = build_allen_layer_file_id_volumes(
        result.projected_nodes,
        result.volume.shape,
    )

    assert [group.group_key for group in groups] == ["a.swc", "b.swc"]
    assert [group.rendered_nodes for group in groups] == [2, 1]


def test_allen_layer_grouped_volumes_preserve_counts() -> None:
    result = build_allen_layer_stack_from_projected_nodes(
        _allen_layer_nodes(),
        _allen_layer_map(),
        y_bins=10,
        x_bounds=(0.0, 10.0),
        y_bounds=(0.0, 10.0),
    )

    individual = build_allen_layer_file_id_volumes(
        result.projected_nodes,
        result.volume.shape,
    )
    clusters = build_allen_layer_cluster_volumes(
        result.projected_nodes,
        result.volume.shape,
        {"a.swc": 1, "b.swc": 2},
    )

    assert sum(float(group.volume.sum()) for group in individual) == 6.0
    assert [group.label for group in clusters] == [
        "Cluster 1",
        "Cluster 2",
        "Unclustered",
    ]
    assert sum(float(group.volume.sum()) for group in clusters) == 6.0


def test_allen_layer_stack_clips_bins_to_canonical_bounds() -> None:
    nodes = pd.DataFrame(
        {
            "file_id": ["low.swc", "high.swc"],
            "x_flat": [-100.0, 100.0],
            "y_flat": [-100.0, 100.0],
            "flatmap_valid": [True, True],
            "region_id": [101, 101],
        }
    )

    result = build_allen_layer_stack_from_projected_nodes(
        nodes,
        _allen_layer_map(),
        y_bins=4,
        x_bounds=(0.0, 10.0),
        y_bounds=(0.0, 10.0),
    )

    assert result.volume[0, 0, 0] == 1.0
    assert result.volume[0, 3, 3] == 1.0


# ---------------------------------------------------------------------------
# DuckDB-backed precomputed heatmap rendering.
# ---------------------------------------------------------------------------


def _v3_flatmap_frame(seed: int = 0, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 120.0, n).astype(np.float32)
    y = rng.uniform(-5.0, 90.0, n).astype(np.float32)
    depth = rng.uniform(-30.0, 900.0, n).astype(np.float32)
    depth[rng.integers(0, n, 150)] = -1.0
    flatmap_valid = rng.random(n) > 0.1
    depth_valid = (depth >= 0.0) & (rng.random(n) > 0.05)
    file_index = rng.integers(0, 6, n)
    return pd.DataFrame(
        {
            "file_id": [f"neuron_{i}" for i in file_index],
            "node_id": np.arange(n, dtype=np.int32),
            "parent_id": np.full(n, -1, dtype=np.int32),
            "type": np.full(n, 3, dtype=np.int32),
            "x_flat_shaped": x,
            "y_flat_shaped": y,
            "x_flat_square": x + 1.0,
            "y_flat_square": y + 1.0,
            "depth_um": depth,
            "flatmap_shaped_valid": flatmap_valid,
            "flatmap_square_valid": flatmap_valid,
            "depth_valid": depth_valid,
        }
    )


def _pandas_reference_volume(
    frame: pd.DataFrame,
    *,
    suffix: str,
    y_bins: int,
    depth_bin_um: float,
    include: bool,
    x_bounds,
    y_bounds,
    depth_range_um,
    collapse_depth: bool = False,
):
    projected = pd.DataFrame(
        {
            "file_id": frame["file_id"].to_numpy(),
            "x_flat": pd.to_numeric(frame[f"x_flat_{suffix}"]),
            "y_flat": pd.to_numeric(frame[f"y_flat_{suffix}"]),
            "depth_um": pd.to_numeric(frame["depth_um"]),
            "flatmap_valid": frame[f"flatmap_{suffix}_valid"].astype(bool),
            "depth_valid": frame["depth_valid"].astype(bool),
        }
    )
    return build_flatmap_render_data_from_projected_nodes(
        projected,
        y_bins=y_bins,
        depth_bin_um=depth_bin_um,
        include_depth_minus_one=include,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        depth_range_um=depth_range_um,
        collapse_depth=collapse_depth,
    )


@pytest.fixture()
def _flatmap_parquet(tmp_path):
    frame = _v3_flatmap_frame()
    path = tmp_path / "flatmap_v3.parquet"
    frame.to_parquet(path, index=False)
    return frame, str(path)


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize(
    "style_key,suffix", [("both_shaped", "shaped"), ("both_square", "square")]
)
def test_duckdb_single_volume_matches_pandas(
    _flatmap_parquet, include, style_key, suffix
):
    _frame, path = _flatmap_parquet
    x_bounds, y_bounds, depth_range = (0.0, 118.0), (0.0, 88.0), (0.0, 890.0)
    reference = _pandas_reference_volume(
        _frame,
        suffix=suffix,
        y_bins=32,
        depth_bin_um=50.0,
        include=include,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        depth_range_um=depth_range,
    )
    conn = duckdb.connect()
    try:
        result = build_flatmap_heatmap_volume_result(
            conn,
            path,
            style_key=style_key,
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            depth_range_um=depth_range,
            y_bins=32,
            depth_bin_um=50.0,
            include_depth_minus_one=include,
        )
    finally:
        conn.close()

    assert result.volume.shape == reference.volume.shape
    np.testing.assert_array_equal(result.volume, reference.volume)
    assert result.render_summary.rendered_nodes == reference.summary.rendered_nodes
    assert result.render_summary.nonzero_voxels == reference.summary.nonzero_voxels
    assert (
        result.render_summary.flatmap_valid_nodes
        == reference.summary.flatmap_valid_nodes
    )
    assert (
        result.render_summary.depth_valid_nodes == reference.summary.depth_valid_nodes
    )


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize(
    "style_key,suffix", [("both_shaped", "shaped"), ("both_square", "square")]
)
def test_duckdb_collapsed_volume_matches_pandas(
    _flatmap_parquet, include, style_key, suffix
):
    _frame, path = _flatmap_parquet
    x_bounds, y_bounds, depth_range = (0.0, 118.0), (0.0, 88.0), (0.0, 890.0)
    reference = _pandas_reference_volume(
        _frame,
        suffix=suffix,
        y_bins=32,
        depth_bin_um=50.0,
        include=include,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        depth_range_um=depth_range,
        collapse_depth=True,
    )
    conn = duckdb.connect()
    try:
        result = build_flatmap_heatmap_volume_result(
            conn,
            path,
            style_key=style_key,
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            depth_range_um=depth_range,
            y_bins=32,
            depth_bin_um=50.0,
            include_depth_minus_one=include,
            collapse_depth=True,
        )
    finally:
        conn.close()

    # Not square: the x count is derived from the bounds' aspect ratio, so the
    # DuckDB grid must match the pandas grid on both axes, not just on y.
    expected_shape = (
        reference.summary.y_bins,
        reference.summary.x_bins,
    )
    assert expected_shape[0] != expected_shape[1]
    assert result.volume.shape == expected_shape
    assert result.volume_shape == expected_shape
    np.testing.assert_array_equal(result.volume, reference.volume)
    assert result.render_summary.rendered_nodes == reference.summary.rendered_nodes
    assert result.render_summary.depth_bins == 0
    assert result.render_summary.depth_bin_um == 0.0


@pytest.mark.parametrize("include", [False, True])
def test_duckdb_collapsed_volume_equals_summed_depth_volume(_flatmap_parquet, include):
    _frame, path = _flatmap_parquet
    x_bounds, y_bounds, depth_range = (0.0, 118.0), (0.0, 88.0), (0.0, 890.0)
    shared = dict(
        style_key="both_shaped",
        color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        depth_range_um=depth_range,
        y_bins=24,
        depth_bin_um=75.0,
        include_depth_minus_one=include,
    )
    conn = duckdb.connect()
    try:
        depth = build_flatmap_heatmap_volume_result(conn, path, **shared)
        flat = build_flatmap_heatmap_volume_result(
            conn, path, collapse_depth=True, **shared
        )
    finally:
        conn.close()

    np.testing.assert_array_equal(flat.volume, depth.volume.sum(axis=0))
    assert flat.render_summary.rendered_nodes == depth.render_summary.rendered_nodes


def test_duckdb_collapsed_grouped_volumes_sum_to_single(_flatmap_parquet):
    _frame, path = _flatmap_parquet
    x_bounds, y_bounds, depth_range = (0.0, 118.0), (0.0, 88.0), (0.0, 890.0)
    shared = dict(
        style_key="both_shaped",
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        depth_range_um=depth_range,
        y_bins=24,
        depth_bin_um=75.0,
        collapse_depth=True,
    )
    conn = duckdb.connect()
    try:
        single = build_flatmap_heatmap_volume_result(
            conn, path, color_mode=FLATMAP_HEATMAP_COLOR_SINGLE, **shared
        )
        individual = build_flatmap_heatmap_volume_result(
            conn, path, color_mode=FLATMAP_HEATMAP_COLOR_INDIVIDUAL, **shared
        )
    finally:
        conn.close()

    assert individual.volume is None
    assert individual.grouped_volumes
    # Every group volume must use the same derived rectangular grid as the
    # single-colour volume, or summing the groups would not reproduce it.
    collapsed_shape = (
        single.render_summary.y_bins,
        single.render_summary.x_bins,
    )
    assert collapsed_shape[0] != collapsed_shape[1]
    assert all(
        group.volume.shape == collapsed_shape for group in individual.grouped_volumes
    )
    combined = sum(group.volume for group in individual.grouped_volumes)
    np.testing.assert_array_equal(combined, single.volume)


@pytest.mark.parametrize("include", [False, True])
def test_duckdb_grouped_volumes_sum_to_single(_flatmap_parquet, include):
    _frame, path = _flatmap_parquet
    x_bounds, y_bounds, depth_range = (0.0, 118.0), (0.0, 88.0), (0.0, 890.0)
    reference = _pandas_reference_volume(
        _frame,
        suffix="shaped",
        y_bins=24,
        depth_bin_um=75.0,
        include=include,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        depth_range_um=depth_range,
    )
    conn = duckdb.connect()
    try:
        individual = build_flatmap_heatmap_volume_result(
            conn,
            path,
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_INDIVIDUAL,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            depth_range_um=depth_range,
            y_bins=24,
            depth_bin_um=75.0,
            include_depth_minus_one=include,
        )
        cluster_map = {f"neuron_{i}": i % 3 for i in range(6)}
        cluster = build_flatmap_heatmap_volume_result(
            conn,
            path,
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_CLUSTER,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            depth_range_um=depth_range,
            y_bins=24,
            depth_bin_um=75.0,
            include_depth_minus_one=include,
            cluster_map=cluster_map,
        )
    finally:
        conn.close()

    assert individual.volume is None
    combined_individual = np.zeros(individual.volume_shape, dtype=np.float32)
    for group in individual.grouped_volumes:
        combined_individual += group.volume
    np.testing.assert_array_equal(combined_individual, reference.volume)

    combined_cluster = np.zeros(cluster.volume_shape, dtype=np.float32)
    for group in cluster.grouped_volumes:
        combined_cluster += group.volume
    np.testing.assert_array_equal(combined_cluster, reference.volume)
    assert [group.label for group in cluster.grouped_volumes][:3] == [
        "Cluster 0",
        "Cluster 1",
        "Cluster 2",
    ]


def test_duckdb_file_id_subset_and_empty_selection(_flatmap_parquet):
    _frame, path = _flatmap_parquet
    bounds = dict(
        x_bounds=(0.0, 118.0), y_bounds=(0.0, 88.0), depth_range_um=(0.0, 890.0)
    )
    conn = duckdb.connect()
    try:
        subset = build_flatmap_heatmap_volume_result(
            conn,
            path,
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            y_bins=16,
            depth_bin_um=100.0,
            include_depth_minus_one=False,
            file_ids=["neuron_0", "neuron_1"],
            **bounds,
        )
        empty = build_flatmap_heatmap_volume_result(
            conn,
            path,
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            y_bins=16,
            depth_bin_um=100.0,
            include_depth_minus_one=False,
            file_ids=[],
            **bounds,
        )
    finally:
        conn.close()

    assert subset.render_summary.traces_represented <= 2
    assert subset.render_summary.rendered_nodes > 0
    assert empty.render_summary.rendered_nodes == 0
    assert float(empty.volume.sum()) == 0.0


def test_duckdb_bounds_fallback_matches_projected_nodes_bounds(_flatmap_parquet):
    _frame, path = _flatmap_parquet
    conn = duckdb.connect()
    try:
        bounds = compute_flatmap_bounds_from_parquet(
            conn, path, style_key="both_shaped", file_ids=None
        )
    finally:
        conn.close()
    assert set(bounds) == {"x_bounds", "y_bounds", "depth_range_um"}
    assert bounds["x_bounds"][0] < bounds["x_bounds"][1]
    assert bounds["depth_range_um"][0] >= 0.0


def test_duckdb_voxel_guard_rejects_oversized_grid(_flatmap_parquet):
    _frame, path = _flatmap_parquet
    conn = duckdb.connect()
    try:
        with pytest.raises(ValueError, match="too large"):
            build_flatmap_heatmap_volume_result(
                conn,
                path,
                style_key="both_shaped",
                color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
                x_bounds=(0.0, 118.0),
                y_bounds=(0.0, 88.0),
                depth_range_um=(0.0, 890.0),
                y_bins=int(MAX_FLATMAP_HEATMAP_VOXELS**0.5) + 1000,
                depth_bin_um=1.0,
                include_depth_minus_one=False,
            )
    finally:
        conn.close()


def test_duckdb_allen_layer_stack_matches_pandas(tmp_path) -> None:
    frame = _v3_flatmap_frame(n=600)
    layer_regions = np.asarray([101, 102, 103, 104, 105, 106, 999])
    frame["region_id"] = np.resize(layer_regions, len(frame))
    path = tmp_path / "allen_layers.parquet"
    frame.to_parquet(path, index=False)
    layer_map = _allen_layer_map()
    projected = pd.DataFrame(
        {
            "file_id": frame["file_id"],
            "x_flat": frame["x_flat_shaped"],
            "y_flat": frame["y_flat_shaped"],
            "flatmap_valid": frame["flatmap_shaped_valid"],
            "region_id": frame["region_id"],
        }
    )
    reference = build_allen_layer_stack_from_projected_nodes(
        projected,
        layer_map,
        y_bins=20,
        x_bounds=(0.0, 118.0),
        y_bounds=(0.0, 88.0),
    )

    conn = duckdb.connect()
    try:
        single = build_allen_layer_heatmap_volume_result(
            conn,
            str(path),
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            layer_map=layer_map,
            x_bounds=(0.0, 118.0),
            y_bounds=(0.0, 88.0),
            y_bins=20,
        )
        individual = build_allen_layer_heatmap_volume_result(
            conn,
            str(path),
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_INDIVIDUAL,
            layer_map=layer_map,
            x_bounds=(0.0, 118.0),
            y_bounds=(0.0, 88.0),
            y_bins=20,
        )
        cluster = build_allen_layer_heatmap_volume_result(
            conn,
            str(path),
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_CLUSTER,
            layer_map=layer_map,
            x_bounds=(0.0, 118.0),
            y_bounds=(0.0, 88.0),
            y_bins=20,
            cluster_map={f"neuron_{index}": index % 2 for index in range(6)},
        )
    finally:
        conn.close()

    np.testing.assert_array_equal(single.volume, reference.volume)
    assert single.summary.layer_node_counts == reference.summary.layer_node_counts
    assert single.summary.excluded_non_layer_nodes == (
        reference.summary.excluded_non_layer_nodes
    )
    for grouped in (individual, cluster):
        combined = np.zeros(grouped.volume_shape, dtype=np.float32)
        for group in grouped.grouped_volumes:
            combined += group.volume
        np.testing.assert_array_equal(combined, reference.volume)


@pytest.mark.parametrize(
    "output_mode",
    [ALLEN_LAYER_OUTPUT_STACK, ALLEN_LAYER_OUTPUT_PROJECTION],
)
def test_duckdb_selected_allen_layers_match_pandas_for_all_color_modes(
    tmp_path,
    output_mode,
) -> None:
    frame = _v3_flatmap_frame(n=600)
    frame["region_id"] = np.resize(
        np.asarray([101, 102, 103, 104, 105, 106, 999]),
        len(frame),
    )
    path = tmp_path / f"selected_allen_{output_mode}.parquet"
    frame.to_parquet(path, index=False)
    layer_map = _allen_layer_map()
    projected = pd.DataFrame(
        {
            "file_id": frame["file_id"],
            "x_flat": frame["x_flat_shaped"],
            "y_flat": frame["y_flat_shaped"],
            "flatmap_valid": frame["flatmap_shaped_valid"],
            "region_id": frame["region_id"],
        }
    )
    selected_indices = (1, 4)
    reference = build_allen_layer_render_from_projected_nodes(
        projected,
        layer_map,
        y_bins=20,
        x_bounds=(0.0, 118.0),
        y_bounds=(0.0, 88.0),
        selected_layer_indices=selected_indices,
        output_mode=output_mode,
    )

    conn = duckdb.connect()
    try:
        results = [
            build_allen_layer_heatmap_volume_result(
                conn,
                str(path),
                style_key="both_shaped",
                color_mode=color_mode,
                layer_map=layer_map,
                x_bounds=(0.0, 118.0),
                y_bounds=(0.0, 88.0),
                y_bins=20,
                selected_layer_indices=selected_indices,
                output_mode=output_mode,
                cluster_map={
                    f"neuron_{index}": index % 2 for index in range(6)
                },
            )
            for color_mode in (
                FLATMAP_HEATMAP_COLOR_SINGLE,
                FLATMAP_HEATMAP_COLOR_INDIVIDUAL,
                FLATMAP_HEATMAP_COLOR_CLUSTER,
            )
        ]
    finally:
        conn.close()

    single, *grouped_results = results
    np.testing.assert_array_equal(single.volume, reference.volume)
    assert single.summary.selected_layer_indices == selected_indices
    assert single.summary.layer_labels == ("L2/3", "L6a")
    assert single.summary.excluded_unselected_layer_nodes == (
        reference.summary.excluded_unselected_layer_nodes
    )
    for grouped in grouped_results:
        combined = np.zeros(reference.volume.shape, dtype=np.float32)
        for group in grouped.grouped_volumes:
            combined += group.volume
        np.testing.assert_array_equal(combined, reference.volume)


def test_duckdb_allen_layer_stack_requires_region_id(
    _flatmap_parquet,
) -> None:
    _frame, path = _flatmap_parquet
    conn = duckdb.connect()
    try:
        with pytest.raises(ValueError, match="region_id"):
            build_allen_layer_heatmap_volume_result(
                conn,
                path,
                style_key="both_shaped",
                color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
                layer_map=_allen_layer_map(),
                x_bounds=(0.0, 118.0),
                y_bounds=(0.0, 88.0),
                y_bins=20,
            )
    finally:
        conn.close()


def test_duckdb_allen_layer_stack_selected_files_and_empty_selection(
    tmp_path,
) -> None:
    frame = _v3_flatmap_frame(n=300)
    frame["region_id"] = np.resize(
        np.asarray([101, 102, 103, 104, 105, 106, 999]),
        len(frame),
    )
    path = tmp_path / "allen_layer_selection.parquet"
    frame.to_parquet(path, index=False)
    selected_ids = ["neuron_0", "neuron_1"]
    reference_nodes = pd.DataFrame(
        {
            "file_id": frame["file_id"],
            "x_flat": frame["x_flat_shaped"],
            "y_flat": frame["y_flat_shaped"],
            "flatmap_valid": frame["flatmap_shaped_valid"],
            "region_id": frame["region_id"],
        }
    )
    reference_nodes = reference_nodes[
        reference_nodes["file_id"].isin(selected_ids)
    ].reset_index(drop=True)
    reference = build_allen_layer_stack_from_projected_nodes(
        reference_nodes,
        _allen_layer_map(),
        y_bins=12,
        x_bounds=(0.0, 118.0),
        y_bounds=(0.0, 88.0),
    )

    conn = duckdb.connect()
    try:
        selected = build_allen_layer_heatmap_volume_result(
            conn,
            str(path),
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            layer_map=_allen_layer_map(),
            x_bounds=(0.0, 118.0),
            y_bounds=(0.0, 88.0),
            y_bins=12,
            file_ids=selected_ids,
        )
        empty = build_allen_layer_heatmap_volume_result(
            conn,
            str(path),
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            layer_map=_allen_layer_map(),
            x_bounds=(0.0, 118.0),
            y_bounds=(0.0, 88.0),
            y_bins=12,
            file_ids=[],
        )
    finally:
        conn.close()

    np.testing.assert_array_equal(selected.volume, reference.volume)
    assert selected.summary.traces_represented <= 2
    # An empty selection still reports the derived rectangular grid, so a caller
    # sizing a layer from it cannot disagree with a populated render.
    assert empty.volume.shape == (6, selected.summary.y_bins, selected.summary.x_bins)
    assert empty.volume.shape == (6, 12, 16)
    assert float(empty.volume.sum()) == 0.0
    assert empty.summary.rendered_nodes == 0


def test_depth_plane_labels_name_each_micron_interval() -> None:
    labels = depth_plane_labels(
        {
            "depth_bins": 3,
            "depth_bin_um": 25.0,
            "depth_min_um": 0.0,
            "includes_depth_minus_one_plane": False,
        }
    )

    assert labels == ("0-25 um", "25-50 um", "50-75 um")


def test_depth_plane_labels_name_the_sentinel_plane_first() -> None:
    labels = depth_plane_labels(
        {
            "depth_bins": 3,
            "depth_bin_um": 25.0,
            "depth_min_um": 0.0,
            "includes_depth_minus_one_plane": True,
        }
    )

    assert labels == ("depth -1", "0-25 um", "25-50 um")


def test_depth_plane_labels_offset_by_the_lower_depth_bound() -> None:
    labels = depth_plane_labels(
        {
            "depth_bins": 2,
            "depth_bin_um": 50.0,
            "depth_min_um": 100.0,
            "includes_depth_minus_one_plane": False,
        }
    )

    assert labels == ("100-150 um", "150-200 um")


@pytest.mark.parametrize(
    "summary",
    [
        {},
        {"depth_bins": 0, "depth_bin_um": 25.0},
        {"depth_bins": 4, "depth_bin_um": 0.0},
        {"depth_bins": "many", "depth_bin_um": 25.0},
    ],
)
def test_depth_plane_labels_return_empty_without_depth_binning(summary) -> None:
    assert depth_plane_labels(summary) == ()
