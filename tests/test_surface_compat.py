"""Real napari model regressions for surface/image axis registration."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest
from napari.components import Dims, ViewerModel
from napari.layers import Surface
from napari.layers.surface._slice import _SurfaceSliceRequest

from napari_neuron_navigator import surface_compat as compat
from napari_neuron_navigator.isocortex_layers import CustomRegionSelectionGroup

ORDERS = tuple(permutations(range(3)))
VERTICES = np.asarray([[1, 2, 3], [4, 2, 3], [1, 6, 3], [1, 2, 8]], dtype=np.float32)
FACES = np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3]], dtype=np.int32)
VALUES = np.asarray([1, 3, 5, 7], dtype=np.float32)
COLORS = np.asarray(
    [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 0.5]],
    dtype=np.float32,
)


@pytest.fixture(autouse=True)
def clear_probe_cache():
    probe = compat._needs_surface_axis_order_fix
    probe.cache_clear()
    yield
    probe.cache_clear()


@pytest.fixture
def viewer():
    model = ViewerModel(ndisplay=3)
    try:
        with model._layer_slicer.force_sync():
            yield model
    finally:
        model._layer_slicer.shutdown()


def _add_mesh(viewer, **kwargs):
    return compat.add_surface_with_axis_order(
        viewer, (VERTICES.copy(), FACES.copy(), VALUES.copy()), **kwargs
    )


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("transpose_first", [True, False])
def test_surface_matches_image_labels_and_points(viewer, order, transpose_first):
    volume = np.arange(11 * 13 * 17).reshape(11, 13, 17)
    image = viewer.add_image(volume)
    labels = viewer.add_labels((volume % 3).astype(np.uint8))
    points = viewer.add_points(VERTICES)
    if transpose_first:
        viewer.dims.order = order
    layer = _add_mesh(viewer, vertex_colors=COLORS, metadata={"region_id": 1})
    if not transpose_first:
        viewer.dims.order = order

    assert viewer.dims.order == order
    np.testing.assert_array_equal(image._data_view, volume.transpose(order))
    np.testing.assert_array_equal(labels._data_view, labels.data.transpose(order))
    np.testing.assert_array_equal(points._view_data, VERTICES[:, order])
    np.testing.assert_array_equal(layer._view_vertices, points._view_data)
    np.testing.assert_array_equal(layer._view_faces, FACES)
    np.testing.assert_array_equal(layer._view_vertex_colors, COLORS)
    np.testing.assert_array_equal(layer.vertices, VERTICES)
    np.testing.assert_array_equal(layer.vertex_values, VALUES)
    assert layer.metadata == {"region_id": 1}
    assert type(layer) is Surface
    data, _, layer_type = layer.as_layer_data_tuple()
    assert layer_type == "surface"
    np.testing.assert_array_equal(data[0], VERTICES)


@pytest.mark.parametrize("order", ORDERS)
def test_anisotropic_scale_and_translation_follow_axis_order(viewer, order):
    layer = _add_mesh(viewer, scale=(0.04, 0.02, 0.01), translate=(10, 20, 30))
    extent = layer.extent.world.copy()
    viewer.dims.order = order
    actual = layer._data_to_world.set_slice(order)(layer._view_vertices)
    expected = (VERTICES * layer.scale + layer.translate)[:, order]
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(layer.extent.world, extent)


def test_transpose_roll_hide_recreate_and_refresh_do_not_accumulate(viewer):
    viewer.add_image(np.zeros((11, 13, 17), dtype=np.uint8))
    viewer.dims.transpose()
    layer = _add_mesh(viewer, visible=False)
    viewer.dims.roll()
    layer.visible = True
    for _ in range(3):
        viewer.dims.transpose()
        viewer.dims.roll()
        compat._install_surface_axis_order_fix(layer)
        layer.refresh()
        np.testing.assert_array_equal(
            layer._view_vertices, VERTICES[:, viewer.dims.displayed]
        )
    viewer.layers.remove(layer)
    recreated = _add_mesh(viewer)
    np.testing.assert_array_equal(
        recreated._view_vertices, VERTICES[:, viewer.dims.displayed]
    )
    # New data use the same response handler, without caching old vertices.
    recreated.data = (VERTICES + 1, FACES, VALUES)
    np.testing.assert_array_equal(
        recreated._view_vertices, (VERTICES + 1)[:, viewer.dims.displayed]
    )


@pytest.mark.parametrize("order", ORDERS)
def test_2d_slicing_stays_native_and_return_to_3d_is_aligned(viewer, order):
    layer = _add_mesh(viewer)
    viewer.dims.ndisplay = 2
    viewer.dims.order = order
    viewer.dims.set_point(order[0], float(VERTICES[0, order[0]]))
    native = Surface((VERTICES, FACES, VALUES))
    native._slice_dims(viewer.dims)
    np.testing.assert_array_equal(layer._view_vertices, native._view_vertices)
    np.testing.assert_array_equal(layer._view_faces, native._view_faces)
    np.testing.assert_array_equal(layer._view_vertex_values, native._view_vertex_values)
    viewer.dims.ndisplay = 3
    np.testing.assert_array_equal(layer._view_vertices, VERTICES[:, order])


def test_async_response_uses_its_own_order_and_preserves_other_fields(viewer):
    layer = _add_mesh(viewer, vertex_colors=COLORS)
    state = layer._slicing_state
    dims = Dims(ndim=3, ndisplay=3, order=(2, 0, 1))
    request = state._make_slice_request(dims)
    with ThreadPoolExecutor(max_workers=1) as executor:
        response = executor.submit(request).result(timeout=10)
    # The currently applied state has a different order from the request.
    assert state._slice_input.order == (0, 1, 2)
    original_vertices = response.vertices.copy()
    state._update_slice_response(response)
    np.testing.assert_array_equal(layer._view_vertices, VERTICES[:, (2, 0, 1)])
    np.testing.assert_array_equal(response.vertices, original_vertices)
    assert layer._view_faces is response.faces
    assert layer._view_vertex_values is response.values
    assert layer._view_vertex_colors is response.vertex_colors
    assert layer._slice_input is response.slice_input


def test_three_dimensional_mesh_in_four_dimensional_viewer(viewer):
    viewer.add_image(np.zeros((2, 11, 13, 17), dtype=np.uint8))
    viewer.dims.order = (0, 3, 1, 2)
    layer = _add_mesh(viewer)
    np.testing.assert_array_equal(layer._view_vertices, VERTICES[:, (2, 0, 1)])


def test_extra_vertex_value_dimension_uses_native_slicing(viewer):
    values = np.stack([VALUES, VALUES + 10])
    layer = compat.add_surface_with_axis_order(viewer, (VERTICES, FACES, values))
    viewer.dims.order = (0, 3, 1, 2)
    viewer.dims.set_point(0, 1)
    native = Surface((VERTICES, FACES, values))
    native._slice_dims(viewer.dims)
    np.testing.assert_array_equal(layer._view_vertices, native._view_vertices)
    np.testing.assert_array_equal(layer._view_vertex_values, native._view_vertex_values)


def test_empty_geometry_remains_empty_after_transpose(viewer):
    layer = compat.add_surface_with_axis_order(
        viewer,
        (np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)),
        contrast_limits=(0, 1),
    )
    viewer.dims.transpose()
    assert layer._view_vertices.shape == (0, 3)
    assert layer._view_faces.shape == (0, 3)


def _override_request_vertices(monkeypatch, transform):
    original = _SurfaceSliceRequest.__call__

    def request(self):
        response = original(self)
        if self.slice_input.ndisplay == 3 and not self.slice_input.not_displayed:
            response = replace(response, vertices=transform(self))
        return response

    monkeypatch.setattr(_SurfaceSliceRequest, "__call__", request)


def test_probe_detects_known_defect(monkeypatch):
    _override_request_vertices(monkeypatch, lambda request: request.data[0])
    assert compat._needs_surface_axis_order_fix() is True


def test_upstream_fix_is_detected_and_not_applied_twice(viewer, monkeypatch):
    _override_request_vertices(
        monkeypatch,
        lambda request: request.data[0][:, request.slice_input.displayed],
    )
    assert compat._needs_surface_axis_order_fix() is False
    layer = _add_mesh(viewer)
    for order in ORDERS:
        viewer.dims.order = order
        np.testing.assert_array_equal(layer._view_vertices, VERTICES[:, order])
    assert not getattr(layer._slicing_state, compat._INSTALLED, False)


def test_unexpected_behavior_fails_before_adding_layer(viewer, monkeypatch):
    _override_request_vertices(monkeypatch, lambda request: request.data[0] * 2)
    with pytest.raises(RuntimeError, match="Unexpected napari Surface"):
        _add_mesh(viewer)
    assert not viewer.layers


def test_failed_installation_removes_uncorrected_layer(viewer, monkeypatch):
    def fail(_layer):
        raise RuntimeError("incompatible response API")

    monkeypatch.setattr(compat, "_needs_surface_axis_order_fix", lambda: True)
    monkeypatch.setattr(compat, "_install_surface_axis_order_fix", fail)
    with pytest.raises(RuntimeError, match="incompatible response API"):
        _add_mesh(viewer)
    assert not viewer.layers


@pytest.mark.parametrize("factory", ["region", "group", "outline"])
@pytest.mark.parametrize("order", ORDERS)
def test_atlas_surface_constructors_preserve_registration(viewer, factory, order):
    from tests.test_reference_layers_logging import (
        _FakeMesh,
        _FakeReferenceAtlas,
        _import_reference_layers_module,
    )

    module = _import_reference_layers_module(real_surfaces=True)
    atlas = _FakeReferenceAtlas()
    atlas.resolution = (25, 50, 100)
    atlas.reference = np.zeros((11, 13, 17), dtype=np.uint8)
    atlas._meshes["AAA1"] = _FakeMesh(VERTICES * atlas.resolution, FACES)
    atlas._meshes["root"] = atlas._meshes["AAA1"]
    module.add_allen_template(viewer, atlas)
    viewer.dims.order = order
    if factory == "region":
        layer = module.add_region_mesh(viewer, atlas, "AAA1")
    elif factory == "group":
        group = CustomRegionSelectionGroup(
            label="L1", region_ids=(101,), acronyms=("AAA1",)
        )
        layer, missing = module.add_region_mesh_group(viewer, atlas, group)
        assert not missing
    else:
        layer = module.add_brain_outline(viewer, atlas)
    actual = layer._data_to_world.set_slice(order)(layer._view_vertices)
    np.testing.assert_allclose(actual, VERTICES[:, order])
    assert viewer.dims.order == order
    count = len(viewer.layers)
    assert module.add_region_mesh(viewer, atlas, "MISSING") is None
    assert len(viewer.layers) == count
