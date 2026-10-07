"""Tests for reference-layer startup timing logs."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys
import types

import numpy as np

from napari_neuron_navigator.region_appearance import (
    RegionAppearanceOverride,
    RegionAppearanceStore,
)

from napari_neuron_navigator.isocortex_layers import CustomRegionSelectionGroup


def _import_reference_layers_module(*, real_surfaces=False):
    """Import ``reference_layers.py`` without importing the widgets package."""
    module_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "napari_neuron_navigator"
        / "widgets"
        / "reference_layers.py"
    )
    spec = importlib.util.spec_from_file_location(
        "napari_neuron_navigator.widgets.reference_layers_test_module",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not real_surfaces:
        # These tests inspect constructor arguments using fake layers. Real
        # napari slicing is covered separately in test_surface_compat.py.
        module.add_surface_with_axis_order = (
            lambda viewer, data, **kwargs: viewer.add_surface(data, **kwargs)
        )
    return module


class _FakeAtlas:
    def __init__(self) -> None:
        self.reference = np.zeros((2, 3, 4), dtype=np.float32)


class _FakeViewer:
    def __init__(self) -> None:
        self.calls = []

    def add_image(self, data, **kwargs):
        self.calls.append((data, kwargs))
        return types.SimpleNamespace(data=data, kwargs=kwargs)


class _FakeReferenceViewer:
    def __init__(self) -> None:
        self.surface_calls = []
        self.labels_calls = []

    def add_surface(self, data, **kwargs):
        layer = types.SimpleNamespace(data=data, **kwargs)
        self.surface_calls.append((data, kwargs))
        return layer

    def add_labels(self, data, **kwargs):
        layer = types.SimpleNamespace(data=data, **kwargs)
        self.labels_calls.append((data, kwargs))
        return layer


class _FakeMesh:
    def __init__(self, points: np.ndarray, faces: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=np.float32)
        self.cells = [types.SimpleNamespace(data=np.asarray(faces, dtype=np.int32))]


class _FakeReferenceAtlas:
    resolution = (25.0, 25.0, 25.0)

    def __init__(self) -> None:
        region_a = {
            "id": 101,
            "acronym": "AAA1",
            "rgb_triplet": [255, 0, 0],
        }
        region_b = {
            "id": 102,
            "acronym": "BBB1",
            "rgb_triplet": [0, 255, 0],
        }
        unrelated = {
            "id": 999,
            "acronym": "OTHER",
            "rgb_triplet": [0, 0, 255],
        }
        self.structures = {
            101: region_a,
            102: region_b,
            999: unrelated,
            "AAA1": region_a,
            "BBB1": region_b,
            "OTHER": unrelated,
        }
        self.annotation = np.asarray([[[101, 102, 999, 0]]], dtype=np.int32)
        self._meshes = {
            "AAA1": _FakeMesh(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[0, 1, 2]],
            ),
            "BBB1": _FakeMesh(
                [[2, 0, 0], [3, 0, 0], [2, 1, 0]],
                [[0, 1, 2]],
            ),
        }

    def mesh_from_structure(self, acronym: str):
        if acronym not in self._meshes:
            raise FileNotFoundError(acronym)
        return self._meshes[acronym]


def test_add_allen_template_logs_reference_and_add_image_timings(caplog):
    """Template timing should isolate atlas access from napari layer creation."""
    module = _import_reference_layers_module()
    viewer = _FakeViewer()
    atlas = _FakeAtlas()

    with caplog.at_level(logging.DEBUG, logger=module.logger.name):
        layer = module.add_allen_template(viewer, atlas, name="Template")

    assert layer.data is atlas.reference
    assert viewer.calls[0][1]["name"] == "Template"

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "startup_timing event=add_allen_template_phase status=ok" in message
        and "phase=atlas.reference" in message
        and "shape=2x3x4" in message
        and "dtype=float32" in message
        for message in messages
    )
    assert any(
        "startup_timing event=add_allen_template_phase status=ok" in message
        and "phase=viewer.add_image" in message
        and "shape=2x3x4" in message
        and "dtype=float32" in message
        for message in messages
    )


def test_add_region_mesh_group_offsets_faces_and_preserves_colors() -> None:
    module = _import_reference_layers_module()
    viewer = _FakeReferenceViewer()
    atlas = _FakeReferenceAtlas()
    group = CustomRegionSelectionGroup(
        label="L1",
        region_ids=(101, 102),
        acronyms=("AAA1", "BBB1"),
    )

    layer, missing = module.add_region_mesh_group(viewer, atlas, group)

    assert missing == ()
    assert layer is not None
    assert len(viewer.surface_calls) == 1
    vertices, faces = viewer.surface_calls[0][0]
    np.testing.assert_array_equal(
        faces,
        np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
    )
    assert vertices.shape == (6, 3)
    kwargs = viewer.surface_calls[0][1]
    np.testing.assert_allclose(
        kwargs["vertex_colors"][:3],
        np.asarray([[1, 0, 0, 1]] * 3),
    )
    np.testing.assert_allclose(
        kwargs["vertex_colors"][3:],
        np.asarray([[0, 1, 0, 1]] * 3),
    )
    assert kwargs["name"] == "Region: Custom L1"
    assert kwargs["scale"] == [0.04, 0.04, 0.04]
    assert kwargs["metadata"]["selected_region_ids"] == [101, 102]


def test_add_region_mesh_group_skips_unavailable_terminal_meshes() -> None:
    module = _import_reference_layers_module()
    viewer = _FakeReferenceViewer()
    atlas = _FakeReferenceAtlas()
    group = CustomRegionSelectionGroup(
        label="L1",
        region_ids=(101, 103),
        acronyms=("AAA1", "MISSING1"),
    )

    layer, missing = module.add_region_mesh_group(viewer, atlas, group)

    assert layer is not None
    assert missing == ("MISSING1",)
    assert layer.metadata["selected_region_ids"] == [101]
    assert layer.metadata["missing_region_acronyms"] == ["MISSING1"]


def test_add_region_id_segmentation_keeps_exact_ids_and_atlas_colors(
    monkeypatch,
) -> None:
    module = _import_reference_layers_module()
    viewer = _FakeReferenceViewer()
    atlas = _FakeReferenceAtlas()

    class _DirectLabelColormap:
        def __init__(self, *, color_dict) -> None:
            self.color_dict = color_dict

    fake_napari_utils = types.ModuleType("napari.utils")
    fake_napari_utils.DirectLabelColormap = _DirectLabelColormap
    monkeypatch.setitem(sys.modules, "napari.utils", fake_napari_utils)

    layer = module.add_region_id_segmentation(
        viewer,
        atlas,
        [102, 101, 102],
    )

    assert layer is not None
    np.testing.assert_array_equal(
        layer.data,
        np.asarray([[[101, 102, 0, 0]]], dtype=np.int32),
    )
    assert layer.metadata == {
        "region_layer_kind": "segmentation",
        "region_selection_source": "custom",
        "selected_region_ids": [101, 102],
        "represented_region_ids": [101, 102],
        "exact_region_ids": True,
    }
    np.testing.assert_allclose(layer.colormap.color_dict[101], [1, 0, 0, 1])
    np.testing.assert_allclose(layer.colormap.color_dict[102], [0, 1, 0, 1])


def test_region_segmentation_appearance_updates_colormap_without_rebuilding_data(
    monkeypatch,
) -> None:
    module = _import_reference_layers_module()
    viewer = _FakeReferenceViewer()
    atlas = _FakeReferenceAtlas()

    class _DirectLabelColormap:
        def __init__(self, *, color_dict) -> None:
            self.color_dict = color_dict

    fake_napari_utils = types.ModuleType("napari.utils")
    fake_napari_utils.DirectLabelColormap = _DirectLabelColormap
    monkeypatch.setitem(sys.modules, "napari.utils", fake_napari_utils)
    initial_store = RegionAppearanceStore(
        overrides={
            101: RegionAppearanceOverride(
                color_mode="custom",
                color_rgb=(0.25, 0.5, 0.75),
            ),
            102: RegionAppearanceOverride(fill_visible=False),
        }
    )
    layer = module.add_region_id_segmentation(
        viewer,
        atlas,
        [101, 102],
        appearance_store=initial_store,
    )
    original_data = layer.data

    np.testing.assert_allclose(layer.colormap.color_dict[101], [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(layer.colormap.color_dict[102], [0.0, 1.0, 0.0, 0.0])

    updated_store = RegionAppearanceStore(
        overrides={101: RegionAppearanceOverride(fill_opacity=0.4)}
    )
    assert module.apply_region_appearance_to_layer(
        layer,
        atlas,
        updated_store,
    )
    assert layer.data is original_data
    np.testing.assert_allclose(layer.colormap.color_dict[101], [1.0, 0.0, 0.0, 0.4])
    np.testing.assert_allclose(layer.colormap.color_dict[102], [0.0, 1.0, 0.0, 1.0])


def test_custom_mesh_group_uses_per_region_fill_visibility() -> None:
    module = _import_reference_layers_module()
    viewer = _FakeReferenceViewer()
    atlas = _FakeReferenceAtlas()
    group = CustomRegionSelectionGroup(
        label="L1",
        region_ids=(101, 102),
        acronyms=("AAA1", "BBB1"),
    )
    store = RegionAppearanceStore(
        overrides={
            101: RegionAppearanceOverride(
                color_mode="custom",
                color_rgb=(0.25, 0.5, 0.75),
            ),
            102: RegionAppearanceOverride(fill_visible=False),
        }
    )

    layer, missing = module.add_region_mesh_group(
        viewer,
        atlas,
        group,
        appearance_store=store,
    )

    assert missing == ()
    np.testing.assert_allclose(
        layer.vertex_colors[:3],
        np.asarray([[0.25, 0.5, 0.75, 1.0]] * 3),
    )
    np.testing.assert_allclose(
        layer.vertex_colors[3:],
        np.asarray([[0.0, 1.0, 0.0, 0.0]] * 3),
    )


def test_region_mesh_restyle_preserves_manual_layer_visibility() -> None:
    module = _import_reference_layers_module()
    viewer = _FakeReferenceViewer()
    atlas = _FakeReferenceAtlas()
    layer = module.add_region_mesh(viewer, atlas, "AAA1")
    assert layer is not None

    layer.visible = False
    module.apply_region_appearance_to_layer(
        layer,
        atlas,
        RegionAppearanceStore(
            overrides={101: RegionAppearanceOverride(fill_visible=True)}
        ),
    )

    assert layer.visible is False
