"""Tests that a flat map grid with unequal per-axis bin counts stays coherent.

Every one of these is blind while ``x_bins == y_bins``: a transposed index still
lands in bounds, and an overlay built with a single shared count still lines up.
They use deliberately unequal counts *and* unequal spans so an axis swap or a
one-count shortcut cannot pass.

See ``tests/test_flatmap_bin_counts.py`` for the derivation policy itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_neuron_navigator.flatmap_heatmap import (
    ALLEN_LAYER_OUTPUT_PROJECTION,
    _bin_flat_values,
    build_allen_layer_render_from_projected_nodes,
    build_allen_layer_stack_from_projected_nodes,
    build_flatmap_render_data_from_projected_nodes,
    build_flatmap_segment_vectors,
    flatmap_pixel_coordinates,
    resolve_flatmap_bin_counts,
)
from napari_neuron_navigator.flatmap_labels import build_flatmap_region_label_volume
from napari_neuron_navigator.flatmap_projection import build_projected_segments
from napari_neuron_navigator.isocortex_layers import AllenIsocortexLayerMap

# Unequal on both axes and in both spans: 7 x bins over 14 units, 3 y bins over
# 3 units.  A builder that reused one count, or swapped the axes, cannot agree
# with an independent recomputation under these numbers.
X_BOUNDS = (0.0, 14.0)
Y_BOUNDS = (0.0, 3.0)
DEPTH_RANGE = (0.0, 100.0)
Y_BINS = 3
X_BINS = 7


def _asymmetric_nodes() -> pd.DataFrame:
    """Four connected nodes at positions that differ under any axis swap."""
    return pd.DataFrame(
        {
            "file_id": ["a.swc"] * 4,
            "node_id": [1, 2, 3, 4],
            "parent_id": [-1, 1, 2, 3],
            # Deliberately off bin centers so the half-pixel convention matters.
            "x_flat": [0.7, 5.3, 9.1, 13.4],
            "y_flat": [0.2, 1.6, 2.9, 1.1],
            "depth_um": [10.0, 30.0, 55.0, 80.0],
            "flatmap_valid": [True] * 4,
            "depth_valid": [True] * 4,
        }
    )


def _render(**overrides):
    return build_flatmap_render_data_from_projected_nodes(
        _asymmetric_nodes(),
        y_bins=Y_BINS,
        x_bins=X_BINS,
        depth_bin_um=25.0,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        depth_range_um=DEPTH_RANGE,
        **overrides,
    )


def test_volume_axes_are_depth_y_x_not_transposed() -> None:
    """The lit voxel must be at ``(depth, y_bin, x_bin)``, in that order.

    With equal counts a ``(x, y)`` transpose is invisible: the index is still in
    range and the render still looks plausible.  A 3-tall by 7-wide grid makes it
    an ``IndexError`` or a wrong voxel.
    """
    render = _render()
    assert render.volume.shape[1:] == (Y_BINS, X_BINS)

    table = render.projected_nodes
    for _, row in table.iterrows():
        depth_bin = int(row["depth_bin"])
        y_bin = int(row["y_flat_bin"])
        x_bin = int(row["x_flat_bin"])
        assert 0 <= y_bin < Y_BINS
        assert 0 <= x_bin < X_BINS
        assert render.volume[depth_bin, y_bin, x_bin] > 0.0

    # Independent recomputation of the whole volume from the raw coordinates.
    expected = np.zeros(render.volume.shape, dtype=np.float32)
    x_bins = _bin_flat_values(table["x_flat"].to_numpy(float), X_BOUNDS, X_BINS)
    y_bins = _bin_flat_values(table["y_flat"].to_numpy(float), Y_BOUNDS, Y_BINS)
    np.add.at(
        expected,
        (table["depth_bin"].to_numpy(int), y_bins, x_bins),
        1.0,
    )
    np.testing.assert_array_equal(render.volume, expected)


def test_a_single_node_lights_the_bin_named_by_its_own_coordinates() -> None:
    """One node on a 3x7 grid, at a position no swap can reproduce."""
    nodes = pd.DataFrame(
        {
            "file_id": ["a.swc"],
            "node_id": [1],
            "parent_id": [-1],
            # x is in bin 3 of 7; y is in bin 0 of 3.  Swapping would ask for
            # y bin 3, which does not exist.
            "x_flat": [7.0],
            "y_flat": [0.5],
            "depth_um": [10.0],
            "flatmap_valid": [True],
            "depth_valid": [True],
        }
    )
    render = build_flatmap_render_data_from_projected_nodes(
        nodes,
        y_bins=Y_BINS,
        x_bins=X_BINS,
        depth_bin_um=25.0,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        depth_range_um=DEPTH_RANGE,
        collapse_depth=True,
    )
    assert render.volume.shape == (Y_BINS, X_BINS)
    lit = np.argwhere(render.volume > 0.0)
    assert lit.tolist() == [[0, 3]]


def test_vectors_and_heatmap_share_one_grid_per_node() -> None:
    """The overlay must land on the heatmap bins, per node, on both axes.

    This is the check the pre-existing alignment test could not make: it used
    square bounds with equal counts, so a vector builder that scaled both axes
    by a single count stayed green while compressing every vector 2x along x.
    """
    render = _render()
    summary = render.summary
    assert (summary.y_bins, summary.x_bins) == (Y_BINS, X_BINS)

    segments = build_projected_segments(
        render.projected_nodes,
        validity_column="render_valid",
    )
    vectors = build_flatmap_segment_vectors(
        segments.data,
        segments.file_ids,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        y_bins=summary.y_bins,
        x_bins=summary.x_bins,
    )

    starts = vectors.data[:, 0]
    ends = starts + vectors.data[:, 1]
    # flatmap_pixel_coordinates returns ``scaled * bins - 0.5`` so that pixel k
    # is drawn centered on bin k; adding the half pixel back recovers the bin
    # index _bin_flat_values would floor to.
    for endpoint_pixels, raw in (
        (starts, segments.data[:, 0]),
        (ends, segments.data[:, 1]),
    ):
        expected_x = _bin_flat_values(raw[:, 0], X_BOUNDS, X_BINS)
        expected_y = _bin_flat_values(raw[:, 1], Y_BOUNDS, Y_BINS)
        rows = np.floor(endpoint_pixels[:, 0] + 0.5).astype(int)
        columns = np.floor(endpoint_pixels[:, 1] + 0.5).astype(int)
        # Row leads column, and each axis uses its own count.
        np.testing.assert_array_equal(rows, expected_y)
        np.testing.assert_array_equal(columns, expected_x)
        assert rows.max() < Y_BINS
        assert columns.max() < X_BINS


def test_reusing_one_count_for_both_axes_visibly_misregisters() -> None:
    """Pin the failure the split prevents, so the guard cannot rot.

    Passing ``y_bins`` for both axes is exactly the old behavior.  It must move
    the columns, which is what a whole-dataset x compression looks like.
    """
    render = _render()
    segments = build_projected_segments(
        render.projected_nodes,
        validity_column="render_valid",
    )
    correct = build_flatmap_segment_vectors(
        segments.data,
        segments.file_ids,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        y_bins=Y_BINS,
        x_bins=X_BINS,
    )
    one_count = build_flatmap_segment_vectors(
        segments.data,
        segments.file_ids,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        y_bins=Y_BINS,
        x_bins=Y_BINS,
    )
    # Rows are unaffected; only the x axis is compressed.
    np.testing.assert_allclose(correct.data[:, 0, 0], one_count.data[:, 0, 0])
    assert not np.allclose(correct.data[:, 0, 1], one_count.data[:, 0, 1])


def test_pixel_coordinates_stay_inside_each_axis_extent() -> None:
    """Clipping must use each axis's own count, not a shared one."""
    far_outside = np.asarray([-100.0, 100.0])
    columns = flatmap_pixel_coordinates(far_outside, X_BOUNDS, X_BINS)
    rows = flatmap_pixel_coordinates(far_outside, Y_BOUNDS, Y_BINS)
    np.testing.assert_allclose(columns, [-0.5, X_BINS - 0.5])
    np.testing.assert_allclose(rows, [-0.5, Y_BINS - 0.5])


def test_explicit_x_bins_is_used_verbatim_not_re_derived() -> None:
    """A stored count must survive untouched, even against the policy.

    A cache-backed render has to reproduce its profile's recorded grid exactly;
    silently "correcting" it to the derived count would make the render mismatch
    the cached mask.
    """
    derived = resolve_flatmap_bin_counts(
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        y_bins=Y_BINS,
    )
    # The policy wants 14 here, so 5 is a value it would never choose.
    assert derived.x_bins == 14
    render = build_flatmap_render_data_from_projected_nodes(
        _asymmetric_nodes(),
        y_bins=Y_BINS,
        x_bins=5,
        depth_bin_um=25.0,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        depth_range_um=DEPTH_RANGE,
    )
    assert render.summary.x_bins == 5
    assert render.volume.shape[1:] == (Y_BINS, 5)

    allen_nodes = _asymmetric_nodes()
    allen_nodes["region_id"] = [10, 10, 11, 11]
    selected_stack = build_allen_layer_render_from_projected_nodes(
        allen_nodes,
        _layer_map(),
        y_bins=Y_BINS,
        x_bins=5,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        selected_layer_indices=[0, 1],
    )
    selected_projection = build_allen_layer_render_from_projected_nodes(
        allen_nodes,
        _layer_map(),
        y_bins=Y_BINS,
        x_bins=5,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        selected_layer_indices=[0, 1],
        output_mode=ALLEN_LAYER_OUTPUT_PROJECTION,
    )
    assert selected_stack.volume.shape == (2, Y_BINS, 5)
    assert selected_projection.volume.shape == (Y_BINS, 5)
    np.testing.assert_array_equal(
        selected_projection.volume,
        selected_stack.volume.sum(axis=0),
    )


def test_omitted_x_bins_derives_the_square_bin_count() -> None:
    """The default path applies the policy to the resolved bounds."""
    render = build_flatmap_render_data_from_projected_nodes(
        _asymmetric_nodes(),
        y_bins=Y_BINS,
        depth_bin_um=25.0,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        depth_range_um=DEPTH_RANGE,
    )
    assert (render.summary.y_bins, render.summary.x_bins) == (3, 14)
    assert render.volume.shape[1:] == (3, 14)
    x_width = (X_BOUNDS[1] - X_BOUNDS[0]) / render.summary.x_bins
    y_width = (Y_BOUNDS[1] - Y_BOUNDS[0]) / render.summary.y_bins
    assert x_width == pytest.approx(y_width)


def _layer_map() -> AllenIsocortexLayerMap:
    return AllenIsocortexLayerMap(
        atlas_name="test_mouse",
        isocortex_region_id=1,
        region_to_layer_index={10: 0, 11: 1},
        region_ids_by_layer=((10,), (11,), (), (), (), ()),
    )


def test_every_grid_builder_agrees_on_the_trailing_axes() -> None:
    """One bounds set and one ``y_bins`` must give one ``(y, x)`` everywhere.

    A subsystem that derived x differently -- or not at all -- would put a region
    mask, a label volume, or a correlation grid on a different grid than the
    heatmap it is combined with.
    """
    expected = resolve_flatmap_bin_counts(
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        y_bins=Y_BINS,
    )
    trailing = (expected.y_bins, expected.x_bins)

    depth_render = build_flatmap_render_data_from_projected_nodes(
        _asymmetric_nodes(),
        y_bins=Y_BINS,
        depth_bin_um=25.0,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        depth_range_um=DEPTH_RANGE,
    )
    flat_render = build_flatmap_render_data_from_projected_nodes(
        _asymmetric_nodes(),
        y_bins=Y_BINS,
        depth_bin_um=25.0,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        depth_range_um=DEPTH_RANGE,
        collapse_depth=True,
    )
    allen_nodes = _asymmetric_nodes()
    allen_nodes["region_id"] = [10, 10, 11, 11]
    allen_stack = build_allen_layer_stack_from_projected_nodes(
        allen_nodes,
        _layer_map(),
        y_bins=Y_BINS,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
    )

    assert depth_render.volume.shape[1:] == trailing
    assert flat_render.volume.shape == trailing
    assert allen_stack.volume.shape[1:] == trailing
    for summary in (
        depth_render.summary,
        flat_render.summary,
        allen_stack.summary,
    ):
        assert (summary.y_bins, summary.x_bins) == trailing


def test_label_volume_shares_the_render_grid() -> None:
    """A region mask must be built on the same derived grid as the heatmap.

    The worker raises on a mask/volume shape mismatch, so a label volume that
    derived x differently would break region-filtered clustering outright.
    """
    # A lookup grid whose valid flat map coordinates span x twice as far as y.
    flatmap = np.full((1, 2, 2, 2), -1.0, dtype=np.float32)
    flatmap[0, 0, 0] = (0.0, 0.0)
    flatmap[0, 0, 1] = (4.0, 1.0)
    flatmap[0, 1, 0] = (2.0, 0.5)
    flatmap[0, 1, 1] = (1.0, 0.25)
    depth = np.full((1, 2, 2), 10.0, dtype=np.float32)
    annotation = np.full((1, 2, 2), 315, dtype=np.int32)

    labels = build_flatmap_region_label_volume(
        annotation,
        flatmap,
        depth,
        selected_region_ids=[315],
        y_bins=4,
        depth_bin_um=25.0,
        mirror_depth_fallback=False,
    )
    summary = labels.summary
    # x spans 0..4 and y spans 0..1, a ratio of 4, so 4 y bins derive 16 x bins.
    assert (summary.y_bins, summary.x_bins) == (4, 16)
    assert labels.labels.shape == (summary.depth_bins, 4, 16)

    explicit = build_flatmap_region_label_volume(
        annotation,
        flatmap,
        depth,
        selected_region_ids=[315],
        y_bins=4,
        x_bins=9,
        depth_bin_um=25.0,
        mirror_depth_fallback=False,
    )
    assert explicit.summary.x_bins == 9
    assert explicit.labels.shape[1:] == (4, 9)


@pytest.mark.parametrize(
    "bad",
    [{"y_bins": 0}, {"y_bins": -3}, {"y_bins": 4, "x_bins": 0}],
)
def test_an_unusable_resolution_is_rejected_before_the_lookup_scan(bad) -> None:
    """Fail on the argument, not after scanning a whole lookup volume.

    Deriving x needs bounds, which for the NRRD path means scanning the lookup
    volume first; a positivity check has to run before that so a typo does not
    cost a full scan.  A volume shape that would itself be rejected proves the
    check runs first.
    """
    from napari_neuron_navigator.flatmap_heatmap import build_flatmap_render_data

    with pytest.raises(ValueError, match="must be positive"):
        build_flatmap_render_data(
            _asymmetric_nodes(),
            # Not a valid flatmap lookup volume: if this were reached first the
            # error would name the volume shape instead of the bin count.
            np.zeros((1, 1, 1), dtype=float),
            np.zeros((1, 1, 1), dtype=float),
            depth_bin_um=25.0,
            **bad,
        )
