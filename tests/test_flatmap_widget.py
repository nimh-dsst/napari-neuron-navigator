from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_neuron_navigator.flatmap_heatmap import flatmap_pixel_coordinates
from napari_neuron_navigator.flatmap_labels import (
    FlatmapRegionLabelsResult,
    FlatmapRegionLabelsSummary,
)
from napari_neuron_navigator.region_appearance import (
    RegionAppearanceOverride,
    RegionAppearanceStore,
)


class _FakeWidget:
    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _FakeFileDialog:
    @staticmethod
    def getOpenFileName(*_args, **_kwargs):
        return "", ""

    @staticmethod
    def getSaveFileName(*_args, **_kwargs):
        return "", ""


class _DummySignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs) -> None:
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


def _load_flatmap_widget_module(monkeypatch):
    fake_qtwidgets = types.ModuleType("qtpy.QtWidgets")
    for name in (
        "QAbstractItemView",
        "QCheckBox",
        "QComboBox",
        "QDoubleSpinBox",
        "QGroupBox",
        "QHBoxLayout",
        "QLabel",
        "QListWidget",
        "QProgressBar",
        "QPushButton",
        "QScrollArea",
        "QSpinBox",
        "QVBoxLayout",
        "QWidget",
    ):
        setattr(fake_qtwidgets, name, _FakeWidget)
    fake_qtwidgets.QFileDialog = _FakeFileDialog
    fake_qtcore = types.ModuleType("qtpy.QtCore")
    fake_qtcore.QThread = _FakeWidget
    fake_qtpy = types.ModuleType("qtpy")
    fake_qtpy.QtWidgets = fake_qtwidgets
    fake_qtpy.QtCore = fake_qtcore
    fake_notifications = types.ModuleType("napari.utils.notifications")
    fake_notifications.show_info = lambda *_args, **_kwargs: None
    fake_notifications.show_warning = lambda *_args, **_kwargs: None
    fake_utils = types.ModuleType("napari.utils")
    fake_utils.notifications = fake_notifications
    fake_colormaps = types.ModuleType("napari.utils.colormaps")
    fake_colormaps.Colormap = lambda *args, **kwargs: types.SimpleNamespace(
        colors=kwargs.get("colors", args[0] if args else None),
        name=kwargs.get("name"),
    )
    fake_napari = types.ModuleType("napari")
    fake_napari.utils = fake_utils

    monkeypatch.setitem(sys.modules, "qtpy", fake_qtpy)
    monkeypatch.setitem(sys.modules, "qtpy.QtWidgets", fake_qtwidgets)
    monkeypatch.setitem(sys.modules, "qtpy.QtCore", fake_qtcore)
    monkeypatch.setitem(sys.modules, "napari", fake_napari)
    monkeypatch.setitem(sys.modules, "napari.utils", fake_utils)
    monkeypatch.setitem(
        sys.modules,
        "napari.utils.notifications",
        fake_notifications,
    )
    monkeypatch.setitem(sys.modules, "napari.utils.colormaps", fake_colormaps)

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "napari_neuron_navigator"
        / "widgets"
        / "flatmap.py"
    )
    module_name = "napari_neuron_navigator.widgets.flatmap_test_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyLabel:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


class _DummyButton:
    def __init__(self) -> None:
        self.enabled = True

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


class _DummyListItem:
    def __init__(self, text: str) -> None:
        self._text = str(text)
        self._selected = False

    def text(self) -> str:
        return self._text

    def setSelected(self, selected: bool) -> None:
        self._selected = bool(selected)


class _DummyListWidget:
    def __init__(self) -> None:
        self.items: list[_DummyListItem] = []

    def addItem(self, text: str) -> None:
        self.items.append(_DummyListItem(text))

    def clear(self) -> None:
        self.items.clear()

    def count(self) -> int:
        return len(self.items)

    def item(self, index: int) -> _DummyListItem:
        return self.items[index]

    def selectedItems(self) -> list[_DummyListItem]:
        return [item for item in self.items if item._selected]


class _DummyValueControl:
    def __init__(self, value=0, *, checked: bool = False) -> None:
        self.value = value
        self.checked = bool(checked)
        self.enabled = True

    def setValue(self, value) -> None:
        self.value = value

    def setChecked(self, checked: bool) -> None:
        self.checked = bool(checked)

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def isChecked(self) -> bool:
        return self.checked


class _DummyProgressBar:
    def __init__(self) -> None:
        self.visible = False
        self.range = (0, 1)
        self.value = 0

    def setVisible(self, visible: bool) -> None:
        self.visible = bool(visible)

    def setRange(self, minimum: int, maximum: int) -> None:
        self.range = (int(minimum), int(maximum))

    def setValue(self, value: int) -> None:
        self.value = int(value)


class _DummyCombo:
    def __init__(self, text: str = "allen_mouse_10um") -> None:
        self.text = text
        self.enabled = True

    def currentText(self) -> str:
        return self.text

    def setCurrentText(self, text: str) -> None:
        self.text = str(text)

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


class _DummyDataCombo:
    def __init__(self, data) -> None:
        self.data = data
        self.enabled = True

    def currentData(self):
        return self.data

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


class _DummyLayer:
    def __init__(self, data, **kwargs) -> None:
        # Kept verbatim so a test can assert what was and was not passed at
        # construction time (napari's Vectors color mode depends on it).
        self.init_kwargs = dict(kwargs)
        self.data = np.asarray(data)
        self.name = kwargs["name"]
        self.metadata = kwargs.get("metadata", {})
        self.gamma = kwargs.get("gamma", 1.0)
        self.edge_color = np.asarray(kwargs.get("edge_color", []))
        self.edge_width = kwargs.get("edge_width")
        self.face_color = np.asarray(kwargs.get("face_color", []))
        self.size = kwargs.get("size")
        self.contrast_limits = kwargs.get("contrast_limits")
        self.colormap = kwargs.get("colormap")
        self.blending = kwargs.get("blending")
        self.rendering = kwargs.get("rendering")
        self.opacity = kwargs.get("opacity")
        self.axis_labels = kwargs.get("axis_labels")
        self.ndim = self.data.ndim if self.data.ndim else 0
        self.contrast_limits_range = kwargs.get(
            "contrast_limits_range",
            self.contrast_limits,
        )
        self._keep_auto_contrast = False
        self._slice_input = types.SimpleNamespace(ndisplay=2)
        self.visible = kwargs.get("visible", True)
        self.refresh_count = 0
        self.thumbnail_updates = 0
        self.slice_updates: list[object] = []
        self.slice_dims_calls = []
        self.raise_status_error = False
        self.status_error_message = (
            "too many indices for array: array is 2-dimensional, but 3 were indexed"
        )

    def refresh(self) -> None:
        self.refresh_count += 1

    def _update_thumbnail(self) -> None:
        self.thumbnail_updates += 1

    def reset_contrast_limits(self, mode=None) -> None:
        self.contrast_limits = (-1.0, float(mode or -1.0))

    def reset_contrast_limits_range(self, mode=None) -> None:
        self.contrast_limits_range = (-2.0, float(mode or -2.0))

    def _update_slice_response(self, response) -> object:
        self.slice_updates.append(response)
        self.contrast_limits = (4.0, 5.0)
        self.contrast_limits_range = (4.0, 5.0)
        return response

    def _slice_dims(self, dims, force: bool = False) -> None:
        self.slice_dims_calls.append((dims, force))

    def _get_source_info(self) -> dict[str, str]:
        return {
            "layer_name": self.name,
            "layer_base": self.name,
            "source_type": "",
            "plugin": "",
        }

    def get_status(
        self,
        position=None,
        *,
        view_direction=None,
        dims_displayed=None,
        world=False,
        value=None,
    ) -> dict[str, str]:
        if self.raise_status_error:
            raise IndexError(self.status_error_message)
        return {
            "layer_name": self.name,
            "layer_base": self.name,
            "source_type": "",
            "plugin": "",
            "coordinates": "",
            "coords": "",
            "value": "",
        }


class _DummyEmitter:
    """Minimal stand-in for a napari event emitter."""

    def __init__(self) -> None:
        self.callbacks: list[object] = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)

    def disconnect(self, callback) -> None:
        self.callbacks.remove(callback)

    def emit(self) -> None:
        for callback in list(self.callbacks):
            callback()


class _DummyDims:
    """Stand-in for ``viewer.dims`` with the fields the widget reads."""

    def __init__(self) -> None:
        self.ndisplay = 3
        self.ndim = 3
        self.axis_labels = ("0", "1", "2")
        self.current_step = (0, 0, 0)
        self.events = types.SimpleNamespace(current_step=_DummyEmitter())

    def set_current_step(self, index: int) -> None:
        """Move the plane slider and notify listeners, as napari would."""
        self.current_step = (int(index),) + tuple(self.current_step[1:])
        self.events.current_step.emit()


class _DummyViewer:
    def __init__(self) -> None:
        self.layers: list[_DummyLayer] = []
        self.dims = _DummyDims()
        camera = types.SimpleNamespace(center=None, zoom=None)
        axes = types.SimpleNamespace(visible=False, labels=True)
        text_overlay = types.SimpleNamespace(
            visible=False,
            text="",
            position="top_left",
            font_size=10,
        )
        self.scene = types.SimpleNamespace(
            camera=camera,
            overlays=types.SimpleNamespace(axes=axes),
        )
        self.canvas = types.SimpleNamespace(
            overlays=types.SimpleNamespace(text=text_overlay),
        )
        # Compatibility aliases keep older assertions readable while the code
        # under test exclusively exercises napari 0.9's public models.
        self.camera = camera
        self.axes = axes
        self.text_overlay = text_overlay

    def _append_layer(self, layer: _DummyLayer) -> _DummyLayer:
        self.layers.append(layer)
        if layer.axis_labels:
            self.dims.axis_labels = tuple(layer.axis_labels)
        return layer

    def add_shapes(self, data, **kwargs) -> _DummyLayer:
        layer = _DummyLayer(data, **kwargs)
        return self._append_layer(layer)

    def add_image(self, data, **kwargs) -> _DummyLayer:
        layer = _DummyLayer(data, **kwargs)
        return self._append_layer(layer)

    def add_points(self, data, **kwargs) -> _DummyLayer:
        layer = _DummyLayer(data, **kwargs)
        return self._append_layer(layer)

    def add_labels(self, data, **kwargs) -> _DummyLayer:
        layer = _DummyLayer(data, **kwargs)
        return self._append_layer(layer)

    def add_surface(self, data, **kwargs) -> _DummyLayer:
        vertices, faces, values = data
        layer = _DummyLayer(vertices, **kwargs)
        layer.faces = np.asarray(faces)
        layer.values = np.asarray(values)
        return self._append_layer(layer)

    def add_vectors(self, data, **kwargs) -> _DummyLayer:
        layer = _DummyLayer(data, **kwargs)
        return self._append_layer(layer)


def _widget(module):
    widget = module.FlatmapProjectionWidget.__new__(module.FlatmapProjectionWidget)
    widget._viewer = _DummyViewer()
    widget._projection_layer = None
    widget._region_labels_layer = None
    widget._flatmap_path = Path("flatmap_both_shaped.nrrd")
    widget._depth_path = Path("depth.nrrd")
    widget._status_label = _DummyLabel()
    widget._region_labels_status_label = _DummyLabel()
    widget._project_btn = _DummyButton()
    widget._projection_progress_bar = _DummyProgressBar()
    widget._region_label_atlas_combo = _DummyCombo("allen_mouse_10um")
    widget._region_labels_btn = _DummyButton()
    widget._clear_region_labels_btn = _DummyButton()
    widget._region_label_atlas_cache = {}
    widget._region_label_atlas_load_thread = None
    widget._region_label_atlas_load_worker = None
    widget._pending_region_label_request = False
    widget._color_map_provider = lambda: {
        "a.swc": [1.0, 0.0, 0.0, 1.0],
        "b.swc": [0.0, 1.0, 0.0, 0.5],
    }
    widget._cluster_map_provider = lambda: {}
    widget._atlas_provider = lambda: None
    widget._selected_region_acronyms_provider = lambda: ["VISp"]
    widget._zero_sentinel_cb = types.SimpleNamespace(isChecked=lambda: False)
    widget._negative_one_sentinel_cb = types.SimpleNamespace(isChecked=lambda: True)
    widget._flatmap_correlation_source_changed_callback = None
    return widget


def test_flatmap_widget_exposes_no_plugin_close_callback(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget_class = module.FlatmapProjectionWidget

    assert "close_display_viewer_callback" not in inspect.signature(
        widget_class.__init__
    ).parameters
    assert not hasattr(widget_class, "set_flatmap_window_open")
    assert not hasattr(widget_class, "_close_flatmap_window_requested")


def test_stale_precomputed_completion_cannot_insert_a_layer(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._precomputed_heatmap_display_generation = 4
    widget._display_generation_provider = lambda: 5
    widget._apply_precomputed_heatmap_result = lambda _result: pytest.fail(
        "stale result was applied"
    )

    widget._on_precomputed_heatmap_finished(object())

    assert "window changed or closed" in widget._status_label.text


class _DummySection:
    """Stand-in for ``CollapsibleSection`` recording its expanded state."""

    def __init__(self, expanded: bool = True) -> None:
        self.expanded = bool(expanded)

    def is_expanded(self) -> bool:
        return self.expanded

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)


class _DummyDatabase:
    """Parquet-backed database double answering only schema questions."""

    def __init__(self, columns) -> None:
        self.columns = set(columns)
        self.described = 0

    def has_column(self, name: str) -> bool:
        self.described += 1
        return str(name) in self.columns


_LEGACY_FLATMAP_COLUMNS = ("file_id", "x_flat", "y_flat", "depth_um")
_V3_FLATMAP_COLUMNS = (
    "file_id",
    "x_flat_shaped",
    "y_flat_shaped",
    "x_flat_square",
    "y_flat_square",
    "depth_um",
)


def _lookup_files_widget(module, columns=None):
    widget = _widget(module)
    widget._lookup_files_section = _DummySection()
    widget._database_provider = lambda: (
        _DummyDatabase(columns) if columns is not None else None
    )
    return widget


def test_lookup_files_section_collapses_for_version_3_flatmap_parquet(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _lookup_files_widget(module, _V3_FLATMAP_COLUMNS)

    widget.invalidate_loaded_parquet_projection()

    assert widget._lookup_files_section.is_expanded() is False


def test_lookup_files_section_collapses_for_legacy_flatmap_parquet(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _lookup_files_widget(module, _LEGACY_FLATMAP_COLUMNS)

    widget.invalidate_loaded_parquet_projection()

    assert widget._lookup_files_section.is_expanded() is False


def test_lookup_files_section_stays_open_without_flatmap_columns(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _lookup_files_widget(module, ("file_id", "x", "y", "z", "region_id"))
    widget._lookup_files_section.set_expanded(False)

    widget.invalidate_loaded_parquet_projection()

    assert widget._lookup_files_section.is_expanded() is True


def test_lookup_files_section_stays_open_for_partial_flatmap_columns(
    monkeypatch,
) -> None:
    """Half a column family cannot drive a projection, so the files still matter."""
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _lookup_files_widget(
        module,
        ("file_id", "x_flat_shaped", "y_flat_shaped", "depth_um"),
    )

    widget.invalidate_loaded_parquet_projection()

    assert widget._lookup_files_section.is_expanded() is True


def test_lookup_files_section_stays_open_without_a_loaded_parquet(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _lookup_files_widget(module)
    widget._lookup_files_section.set_expanded(False)

    widget.invalidate_loaded_parquet_projection()

    assert widget._lookup_files_section.is_expanded() is True


def test_lookup_files_section_reads_schema_not_rows(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)

    def refuse_rows(*_args, **_kwargs):
        pytest.fail("sizing the UI must not query neuron rows")

    database = _DummyDatabase(_V3_FLATMAP_COLUMNS)
    database.get_neurons_for_rendering = refuse_rows
    widget = _widget(module)
    widget._lookup_files_section = _DummySection()
    widget._database_provider = lambda: database

    widget.invalidate_loaded_parquet_projection()

    assert database.described > 0
    assert widget._lookup_files_section.is_expanded() is False


def test_lookup_files_section_survives_an_unreadable_schema(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._lookup_files_section = _DummySection(expanded=False)

    def raising_has_column(_name):
        raise RuntimeError("schema unavailable")

    widget._database_provider = lambda: types.SimpleNamespace(
        has_column=raising_has_column
    )

    widget.invalidate_loaded_parquet_projection()

    assert widget._lookup_files_section.is_expanded() is True


def test_lookup_stats_cache_reuses_matching_file_and_sentinel_settings(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    flatmap_path = tmp_path / "flatmap.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    flatmap_path.write_text("flatmap")
    depth_path.write_text("depth")
    volume_set = types.SimpleNamespace(
        flatmap=np.zeros((2, 2, 2, 2), dtype=np.float32),
        depth=np.ones((2, 2, 2), dtype=np.float32),
        flatmap_path=flatmap_path,
        depth_path=depth_path,
    )
    calls = []

    def fake_compute(flatmap, depth, **kwargs):
        calls.append(kwargs)
        return module.FlatmapLookupStats(
            x_bounds=(0.0, 1.0),
            y_bounds=(0.0, 1.0),
            depth_range_um=(0.0, 1.0),
            flatmap_valid_voxels=1,
            depth_valid_voxels=1,
            flatmap_shape=tuple(flatmap.shape),
            depth_shape=tuple(depth.shape),
            flatmap_dtype=str(flatmap.dtype),
            depth_dtype=str(depth.dtype),
            invalid_zero_sentinel=kwargs["invalid_zero_sentinel"],
            invalid_negative_one_sentinel=kwargs["invalid_negative_one_sentinel"],
        )

    monkeypatch.setattr(module, "compute_flatmap_lookup_stats", fake_compute)

    first = widget._lookup_stats_for_volume_set(
        volume_set,
        invalid_zero_sentinel=False,
        invalid_negative_one_sentinel=True,
    )
    second = widget._lookup_stats_for_volume_set(
        volume_set,
        invalid_zero_sentinel=False,
        invalid_negative_one_sentinel=True,
    )
    third = widget._lookup_stats_for_volume_set(
        volume_set,
        invalid_zero_sentinel=True,
        invalid_negative_one_sentinel=True,
    )

    assert first is second
    assert third is not first
    assert len(calls) == 2


def test_project_cache_profile_restore_waits_for_atlas_then_selects_saved_profile(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    class _ProfileCombo:
        def __init__(self) -> None:
            self.items: list[tuple[str, object]] = []
            self.index = -1

        def blockSignals(self, _blocked: bool) -> None:
            return None

        def clear(self) -> None:
            self.items.clear()
            self.index = -1

        def addItem(self, label: str, value: object) -> None:
            self.items.append((str(label), value))
            if self.index < 0:
                self.index = 0

        def count(self) -> int:
            return len(self.items)

        def itemData(self, index: int):
            return self.items[index][1]

        def setCurrentIndex(self, index: int) -> None:
            self.index = int(index)

        def currentData(self):
            return self.itemData(self.index) if self.index >= 0 else None

    class _ValueControl:
        def __init__(self) -> None:
            self.value = None
            self.checked = False
            self.enabled = True

        def setValue(self, value) -> None:
            self.value = value

        def setChecked(self, checked: bool) -> None:
            self.checked = bool(checked)

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = bool(enabled)

    class _Profile:
        def __init__(self, profile_id: str, y_bins: int) -> None:
            self.profile_id = profile_id
            self.atlas = {"name": "allen_mouse_25um"}
            self._grid = {
                "y_bins": y_bins,
                "x_bins": y_bins,
                "depth_bin_um": 25.0,
            }
            self.compatibility_calls: list[dict[str, object]] = []

        def compatibility_mismatches(self, **requirements):
            self.compatibility_calls.append(requirements)
            return ()

        def style(self, style: str):
            assert style == "both_shaped"
            return types.SimpleNamespace(grid_spec=self._grid)

    first = _Profile("first-profile", 128)
    saved = _Profile("saved-profile", 256)
    cache = types.SimpleNamespace(
        profiles={first.profile_id: first, saved.profile_id: saved}
    )
    import napari_neuron_navigator.flatmap_region_cache as cache_module

    monkeypatch.setattr(cache_module, "open_region_cache", lambda _path: cache)
    monkeypatch.setattr(
        module,
        "read_flatmap_parquet_transform_info",
        lambda _path: types.SimpleNamespace(
            format_version=3,
            lookup_set_id="lookup-set",
            metadata={"shared_depth_definition": {"mirror_coord_axis": 2}},
        ),
    )

    widget._region_cache_dir = None
    widget._region_cache = None
    widget._active_cache_profile = None
    widget._pending_cache_profile_id = None
    widget._cache_dir_label = _DummyLabel()
    widget._cache_status_label = _DummyLabel()
    widget._cache_profile_combo = _ProfileCombo()
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._current_source_parquet_path = lambda: Path("neurons.parquet")
    widget._invalidate_flatmap_grid_layers = lambda: None
    widget._y_bins_spin = _ValueControl()
    widget._depth_bin_spin = _ValueControl()
    widget._exclude_depth_minus_one_cb = _ValueControl()
    widget._negative_one_sentinel_cb = _ValueControl()
    widget._zero_sentinel_cb = _ValueControl()
    widget._region_surfaces_btn = _DummyButton()
    widget._region_outlines_btn = _DummyButton()
    widget._clear_region_geometry_btn = _DummyButton()
    atlas_holder = {"atlas": None}
    widget._atlas_provider = lambda: atlas_holder["atlas"]
    widget._request_cache_directory_open = lambda path, profile_id=None: (
        widget.set_cache_directory(
            path,
            profile_id=profile_id,
        )
    )

    widget.restore_cache_reference(
        {"path": "/relocated/cache", "profile_id": saved.profile_id}
    )

    assert widget._pending_cache_profile_id == saved.profile_id
    assert widget._active_cache_profile is None
    assert widget._cache_profile_combo.count() == 0

    atlas_holder["atlas"] = types.SimpleNamespace(
        atlas_name="allen_mouse_10um",
        local_version=(1, 2),
        structures={
            1: {
                "id": 1,
                "name": "root",
                "acronym": "root",
                "structure_id_path": [1],
                "rgb_triplet": [0, 0, 0],
            }
        },
    )
    widget.refresh_cache_profiles()

    assert widget._active_cache_profile is saved
    assert widget._cache_profile_combo.currentData() is saved
    assert widget._pending_cache_profile_id == saved.profile_id
    assert widget._y_bins_spin.value == 256
    assert widget._depth_bin_spin.value == 25.0
    assert saved.compatibility_calls[-1]["atlas_version"] == "1.2"


def test_set_cache_directory_commits_before_closing_superseded_cache(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    events: list[str] = []

    class _Cache:
        def __init__(self, name: str) -> None:
            self.name = name
            self.closed = False

        def close(self) -> None:
            self.closed = True
            events.append(f"close:{self.name}")

    old_cache = _Cache("old")
    new_cache = _Cache("new")
    new_profile = types.SimpleNamespace(profile_id="new-profile")
    import napari_neuron_navigator.flatmap_region_cache as cache_module

    monkeypatch.setattr(cache_module, "open_region_cache", lambda _path: new_cache)
    widget._region_cache_dir = Path("old-cache")
    widget._region_cache = old_cache
    widget._active_cache_profile = object()
    widget._pending_cache_profile_id = None
    widget._cache_dir_label = _DummyLabel()
    widget._compatible_cache_profile_entries = lambda _cache: (
        ("new profile", new_profile),
    )
    widget._populate_cache_profile_combo = lambda _entries, _profile: None
    widget._activate_cache_profile = lambda profile, **_kwargs: (
        events.append(f"activate:{profile.profile_id}"),
        setattr(widget, "_active_cache_profile", profile),
    )
    widget._set_cache_grid_locked = lambda _locked: None

    widget.set_cache_directory(Path("new-cache"))

    assert events == ["activate:new-profile", "close:old"]
    assert old_cache.closed is True
    assert widget._region_cache is new_cache
    assert widget._active_cache_profile is new_profile

    widget._active_cache_profile = object()
    widget._deactivate_cache_profile = lambda: events.append("deactivate")
    widget.set_cache_directory(None)

    assert events[-2:] == ["deactivate", "close:new"]
    assert new_cache.closed is True
    assert widget._region_cache is None


def test_cache_build_finished_closes_worker_returned_profile(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    closed: list[bool] = []
    profile = types.SimpleNamespace(
        profile_id="built-profile",
        close=lambda: closed.append(True),
    )
    widget._region_cache_dir = Path("cache")
    requests: list[tuple[Path, str | None]] = []
    widget._request_cache_directory_open = lambda path, profile_id=None: (
        requests.append((Path(path), profile_id))
    )
    widget._cache_status_label = _DummyLabel()

    widget._on_cache_build_finished(profile)

    assert closed == [True]
    assert requests == [(Path("cache"), "built-profile")]


class _CacheProfile:
    def __init__(self, profile_id: str, *, output_shape=(1, 4, 4)) -> None:
        self.profile_id = profile_id
        self.atlas = {"name": "allen_mouse_25um"}
        self.grid = {
            "output_shape": list(output_shape),
            "y_bins": 4,
            "x_bins": 4,
            "depth_bins": 1,
            "depth_bin_um": 25.0,
            "x_bounds": [0.0, 1.0],
            "y_bounds": [0.0, 1.0],
            "depth_bounds_um": [0.0, 25.0],
            "includes_depth_minus_one_plane": False,
        }

    def style(self, style: str):
        assert style == "both_shaped"
        return types.SimpleNamespace(grid_spec=self.grid)


def _configure_cache_activation_widget(widget, module) -> _DummyLayer:
    layer = _DummyLayer(
        np.ones((1, 4, 4), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        metadata={"flatmap_render_mode": "heatmap"},
    )
    widget._viewer.layers.append(layer)
    widget._projection_layer = layer
    widget._region_surfaces_layers = []
    widget._region_outlines_layers = []
    widget._active_cache_profile = None
    widget._pending_cache_profile_id = None
    widget._region_cache_dir = Path("cache")
    widget._cache_status_label = _DummyLabel()
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._y_bins_spin = _DummyValueControl(4)
    widget._depth_bin_spin = _DummyValueControl(25.0)
    widget._exclude_depth_minus_one_cb = _DummyValueControl(checked=False)
    widget._negative_one_sentinel_cb = _DummyValueControl(checked=True)
    widget._zero_sentinel_cb = _DummyValueControl(checked=False)
    widget._region_surfaces_btn = _DummyButton()
    widget._region_outlines_btn = _DummyButton()
    widget._clear_region_geometry_btn = _DummyButton()
    widget._last_projected_nodes = pd.DataFrame({"file_id": ["a.swc"]})
    widget._last_summary = _simple_projection_summary(module)
    widget._last_render_summary = _simple_render_summary(
        module,
        includes_depth_minus_one_plane=False,
    )
    widget._last_render_mode = module._RENDER_HEATMAP
    widget._last_flatmap_style = "both_shaped"
    widget._last_coordinate_mode = "parquet_columns"
    widget._last_volume_shape = (1, 4, 4)
    widget._last_projection_source = module._PROJECTION_SOURCE_PRECOMPUTED
    widget._last_cache_dir = None
    widget._last_cache_profile_id = None
    return layer


def test_choose_cache_directory_only_schedules_background_open(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _DummyLayer(
        np.ones((1, 4, 4), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        metadata={"flatmap_render_mode": "heatmap"},
    )
    widget._viewer.layers.append(layer)
    requests = []
    widget._request_cache_directory_open = lambda path, profile_id=None: (
        requests.append((path, profile_id))
    )
    monkeypatch.setattr(
        module.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: "/cache/path",
        raising=False,
    )

    widget._choose_cache_directory()

    assert requests == [("/cache/path", None)]
    assert widget._viewer.layers == [layer]
    assert layer.visible is True


def test_matching_cache_profile_preserves_live_heatmap(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _configure_cache_activation_widget(widget, module)
    profile = _CacheProfile("matching-profile")
    queued = []
    widget._queue_gui_callback = queued.append

    widget._activate_cache_profile(profile, force_transition=True)

    assert widget._viewer.layers == [layer]
    assert layer.visible is True
    assert queued == []
    assert widget._last_cache_dir == "cache"
    assert widget._last_cache_profile_id == "matching-profile"
    assert layer.metadata["cache_path"] == "cache"
    assert layer.metadata["cache_profile_id"] == "matching-profile"
    assert "matching heatmap kept" in widget._cache_status_label.text


def test_matching_cache_profile_preserves_live_allen_layer_stack(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _configure_cache_activation_widget(widget, module)
    layer.name = module._ALLEN_LAYER_HEATMAP_LAYER_NAME
    layer.data = np.ones((6, 4, 4), dtype=np.float32)
    layer.metadata = {"flatmap_render_mode": module._RENDER_ALLEN_LAYERS}
    widget._last_render_summary = _simple_allen_layer_summary(module)
    widget._last_render_mode = module._RENDER_ALLEN_LAYERS
    widget._last_volume_shape = (6, 4, 4)
    profile = _CacheProfile("matching-planar-profile")
    queued = []
    widget._queue_gui_callback = queued.append

    widget._activate_cache_profile(profile, force_transition=True)

    assert widget._viewer.layers == [layer]
    assert layer.visible is True
    assert queued == []
    assert widget._last_cache_profile_id == "matching-planar-profile"
    assert layer.metadata["cache_profile_id"] == "matching-planar-profile"
    assert "matching heatmap kept" in widget._cache_status_label.text


def test_matching_cache_profile_only_retires_other_profile_annotations(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    heatmap = _configure_cache_activation_widget(widget, module)
    profile = _CacheProfile("matching-profile")
    matching_labels = _DummyLayer(
        np.ones((1, 4, 4), dtype=np.int32),
        name="Flatmap Region Labels",
        metadata={
            "cache_path": "old-cache",
            "cache_profile_id": "matching-profile",
            "flatmap_style": "both_shaped",
        },
    )
    stale_surface = _DummyLayer(
        np.ones((1, 3), dtype=np.float32),
        name="Flatmap Region Surfaces",
        metadata={
            "cache_path": "old-cache",
            "cache_profile_id": "other-profile",
            "flatmap_style": "both_shaped",
        },
    )
    widget._viewer.layers.extend([matching_labels, stale_surface])
    widget._region_labels_layer = matching_labels
    widget._region_surfaces_layers = [stale_surface]
    queued = []
    widget._queue_gui_callback = queued.append

    widget._activate_cache_profile(profile, force_transition=True)

    assert heatmap.visible is True
    assert matching_labels.visible is True
    assert matching_labels.metadata["cache_path"] == "cache"
    assert widget._region_labels_layer is matching_labels
    assert stale_surface.visible is False
    assert widget._region_surfaces_layers == []
    assert len(queued) == 1

    queued[0]()

    assert widget._viewer.layers == [heatmap, matching_labels]


def test_mismatched_cache_profile_hides_then_defers_heatmap_removal(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _configure_cache_activation_widget(widget, module)
    profile = _CacheProfile("different-grid", output_shape=(2, 4, 4))
    queued = []
    widget._queue_gui_callback = queued.append

    widget._activate_cache_profile(profile, force_transition=True)

    assert widget._viewer.layers == [layer]
    assert layer.visible is False
    assert len(queued) == 1
    assert widget._last_render_summary is None
    assert "click Project to Flatmap again" in widget._cache_status_label.text

    queued[0]()

    assert widget._viewer.layers == []


def test_failed_candidate_cache_preserves_active_cache_and_heatmap(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _DummyLayer(
        np.ones((1, 4, 4), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        metadata={"flatmap_render_mode": "heatmap"},
    )
    widget._viewer.layers.append(layer)
    closed = []
    old_cache = object()
    old_profile = types.SimpleNamespace(profile_id="old-profile")
    candidate = types.SimpleNamespace(close=lambda: closed.append(True))
    widget._region_cache = old_cache
    widget._region_cache_dir = Path("old-cache")
    widget._active_cache_profile = old_profile
    widget._cache_status_label = _DummyLabel()
    widget._cache_open_request_serial = 3
    widget._cache_open_shutting_down = False
    warnings = []
    monkeypatch.setattr(module, "show_warning", warnings.append)

    def incompatible(_cache):
        raise RuntimeError("no compatible profile")

    widget._compatible_cache_profile_entries = incompatible

    widget._on_cache_open_finished(candidate, 3, Path("bad-cache"), None)

    assert closed == [True]
    assert widget._region_cache is old_cache
    assert widget._active_cache_profile is old_profile
    assert widget._viewer.layers == [layer]
    assert layer.visible is True
    assert warnings and "no compatible profile" in warnings[0]


def test_stale_background_cache_result_is_closed(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    closed = []
    candidate = types.SimpleNamespace(close=lambda: closed.append(True))
    widget._cache_open_request_serial = 9
    widget._cache_open_shutting_down = False

    widget._on_cache_open_finished(candidate, 8, Path("stale-cache"), None)

    assert closed == [True]


def test_refresh_cache_profiles_activates_selected_profile_once(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    class _Combo:
        def __init__(self) -> None:
            self.items = []
            self.index = -1

        def blockSignals(self, _blocked) -> None:
            return None

        def clear(self) -> None:
            self.items = []
            self.index = -1

        def addItem(self, label, value) -> None:
            self.items.append((label, value))
            if self.index < 0:
                self.index = 0

        def setCurrentIndex(self, index) -> None:
            self.index = int(index)

        def currentData(self):
            return self.items[self.index][1] if self.index >= 0 else None

    first = _CacheProfile("first")
    selected = _CacheProfile("selected")
    widget._region_cache = object()
    widget._cache_profile_combo = _Combo()
    widget._pending_cache_profile_id = "selected"
    widget._active_cache_profile = None
    widget._compatible_cache_profile_entries = lambda _cache: (
        ("first", first),
        ("selected", selected),
    )
    activated = []
    widget._activate_cache_profile = activated.append

    widget._refresh_cache_profiles()

    assert activated == [selected]
    assert widget._cache_profile_combo.currentData() is selected


def test_file_ids_for_source_uses_selected_then_all_fallback(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._table_file_ids_provider = lambda: ["a.swc", "b.swc", "a.swc"]
    widget._selected_file_ids_provider = lambda: ["b.swc", "b.swc"]

    assert widget._file_ids_for_source("selected") == ["b.swc"]
    assert widget._file_ids_for_source("all") == ["a.swc", "b.swc"]

    widget._selected_file_ids_provider = lambda: []
    assert widget._file_ids_for_source("selected") == ["a.swc", "b.swc"]


def test_region_label_atlas_defaults_to_10um(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    assert widget._current_region_label_atlas_name() == "allen_mouse_10um"


def _label_result(region_id: int = 7) -> FlatmapRegionLabelsResult:
    summary = FlatmapRegionLabelsSummary(
        input_voxels=1,
        selected_region_count=1,
        selected_source_voxels=1,
        valid_source_voxels=1,
        labeled_voxels=1,
        collision_voxels=0,
        y_bins=1,
        x_bins=1,
        depth_bins=1,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=25.0,
    )
    return FlatmapRegionLabelsResult(
        labels=np.asarray([[[region_id]]], dtype=np.int32),
        summary=summary,
        selected_region_ids=[region_id],
        represented_region_ids=[region_id],
    )


def _configure_region_label_creation_widget(widget) -> None:
    widget._selected_region_ids_provider = lambda: [7]
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._zero_sentinel_cb = types.SimpleNamespace(isChecked=lambda: False)
    widget._negative_one_sentinel_cb = types.SimpleNamespace(isChecked=lambda: True)
    widget._y_bins_spin = types.SimpleNamespace(value=lambda: 1)
    widget._depth_bin_spin = types.SimpleNamespace(value=lambda: 25)
    widget._focus_projection_view = lambda *_args, **_kwargs: None
    widget._lookup_stats_for_volume_set = lambda *_args, **_kwargs: object()


def test_create_region_labels_uses_flatmap_selected_atlas_not_main_loaded_atlas(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    _configure_region_label_creation_widget(widget)
    widget._selected_region_source_provider = lambda: "custom_regions"
    widget._selected_region_scope_provider = lambda: "current_table"
    widget._selected_region_acronyms_provider = lambda: ["C7"]
    atlas10 = types.SimpleNamespace(
        atlas_name="allen_mouse_10um",
        annotation=np.asarray([[[10]]], dtype=np.int32),
        structures={7: {"rgb_triplet": [255, 0, 0]}},
    )
    atlas25 = types.SimpleNamespace(
        atlas_name="allen_mouse_25um",
        annotation=np.asarray([[[25]]], dtype=np.int32),
        structures={7: {"rgb_triplet": [0, 255, 0]}},
    )
    widget._region_label_atlas_cache = {"allen_mouse_10um": atlas10}
    widget._atlas_provider = lambda: atlas25
    captured = {}

    monkeypatch.setattr(
        module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            flatmap=np.zeros((1, 1, 1, 2), dtype=np.float32),
            depth=np.zeros((1, 1, 1), dtype=np.float32),
            flatmap_path=Path("flatmap.nrrd"),
            depth_path=Path("depth.nrrd"),
        ),
    )

    def fake_build(annotation, *_args, **_kwargs):
        captured["annotation"] = np.asarray(annotation).copy()
        captured["mirror_depth_fallback"] = _kwargs["mirror_depth_fallback"]
        captured["mirror_coord_axis"] = _kwargs["mirror_coord_axis"]
        return _label_result(7)

    monkeypatch.setattr(module, "build_flatmap_region_label_volume", fake_build)

    widget._create_region_labels_from_current_state()

    np.testing.assert_array_equal(captured["annotation"], atlas10.annotation)
    assert captured["mirror_depth_fallback"] is True
    assert captured["mirror_coord_axis"] == 2
    metadata = widget._viewer.layers[-1].metadata
    assert metadata["atlas_name"] == "allen_mouse_10um"
    assert metadata["selected_region_ids"] == [7]
    assert metadata["selected_region_acronyms"] == ["C7"]
    assert metadata["region_selection_source"] == "custom_regions"
    assert metadata["region_selection_scope"] == "current_table"


def _install_fake_region_label_atlas_worker(monkeypatch, module):
    workers = []
    threads = []

    class _FakeThread:
        def __init__(self) -> None:
            self.started = _DummySignal()
            self.finished = _DummySignal()
            self.running = False
            self.started_called = False
            threads.append(self)

        def start(self) -> None:
            self.running = True
            self.started_called = True
            self.started.emit()

        def quit(self, *_args) -> None:
            self.running = False
            self.finished.emit()

        def isRunning(self) -> bool:
            return self.running

        def deleteLater(self) -> None:
            return None

    class _FakeAtlasLoadWorker:
        def __init__(self, atlas_name: str) -> None:
            self.atlas_name = atlas_name
            self.status = _DummySignal()
            self.progress = _DummySignal()
            self.finished = _DummySignal()
            self.error = _DummySignal()
            self.thread = None
            workers.append(self)

        def moveToThread(self, thread) -> None:
            self.thread = thread

        def run(self) -> None:
            return None

        def deleteLater(self) -> None:
            return None

    fake_qtcore = sys.modules["qtpy.QtCore"]
    fake_qtcore.QThread = _FakeThread
    fake_workers = types.ModuleType("napari_neuron_navigator.workers")
    fake_workers.AtlasLoadWorker = _FakeAtlasLoadWorker
    monkeypatch.setitem(sys.modules, "napari_neuron_navigator.workers", fake_workers)
    return workers, threads


def test_unloaded_region_label_atlas_loads_then_creates_labels(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    workers, threads = _install_fake_region_label_atlas_worker(monkeypatch, module)
    widget = _widget(module)
    _configure_region_label_creation_widget(widget)
    calls = []

    def fake_create_for_atlas(atlas, selected_region_ids):
        calls.append((atlas, selected_region_ids))
        return _label_result(7)

    widget._create_region_labels_for_atlas = fake_create_for_atlas

    widget._create_region_labels()

    assert len(workers) == 1
    assert workers[0].atlas_name == "allen_mouse_10um"
    assert workers[0].thread is threads[0]
    assert threads[0].started_called is True
    assert widget._region_label_atlas_combo.enabled is False
    assert widget._region_labels_btn.enabled is False
    assert "Loading region-label atlas allen_mouse_10um" in widget._status_label.text
    atlas10 = types.SimpleNamespace(
        atlas_name="allen_mouse_10um",
        annotation=np.zeros((1, 1, 1), dtype=np.int32),
        structures={},
    )

    workers[0].finished.emit(atlas10)

    assert widget._region_label_atlas_cache["allen_mouse_10um"] is atlas10
    assert calls == [(atlas10, [7])]
    assert widget._region_label_atlas_combo.enabled is True
    assert widget._region_labels_btn.enabled is True
    assert widget._region_label_atlas_load_thread is None
    assert widget._region_label_atlas_load_worker is None


def test_region_label_atlas_load_error_restores_controls(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    workers, _threads = _install_fake_region_label_atlas_worker(monkeypatch, module)
    widget = _widget(module)
    _configure_region_label_creation_widget(widget)
    calls = []
    widget._create_region_labels_for_atlas = lambda *_args: calls.append("created")

    widget._create_region_labels()
    workers[0].error.emit("download failed")

    assert calls == []
    assert widget._pending_region_label_request is False
    assert widget._region_label_atlas_combo.enabled is True
    assert widget._region_labels_btn.enabled is True
    assert "download failed" in widget._status_label.text


def _configure_augmentation_widget(widget, source_mode: str, source_path: Path) -> None:
    widget._database_provider = lambda: types.SimpleNamespace(parquet_path=source_path)
    widget._table_file_ids_provider = lambda: ["a.swc", "b.swc", "a.swc"]
    widget._selected_file_ids_provider = lambda: ["b.swc", "b.swc"]
    widget._source_combo = types.SimpleNamespace(currentData=lambda: source_mode)
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._coordinate_mode_combo = types.SimpleNamespace(currentData=lambda: "microns")
    widget._zero_sentinel_cb = types.SimpleNamespace(isChecked=lambda: False)
    widget._negative_one_sentinel_cb = types.SimpleNamespace(isChecked=lambda: True)


def test_augment_current_parquet_to_path_passes_selected_file_ids(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    source = tmp_path / "source.parquet"
    output = tmp_path / "augmented.parquet"
    _configure_augmentation_widget(widget, module._SOURCE_SELECTED, source)
    captured = {}

    def fake_augment(source_path, output_path, flatmap_path, depth_path, **kwargs):
        captured.update(
            {
                "source_path": source_path,
                "output_path": output_path,
                "flatmap_path": flatmap_path,
                "depth_path": depth_path,
                "kwargs": kwargs,
            }
        )
        return types.SimpleNamespace(
            output_parquet=Path(output_path),
            rows=2,
            direct_rows=1,
            mirrored_rows=1,
            unmapped_rows=0,
        )

    monkeypatch.setattr(module, "augment_neuron_parquet_with_flatmap", fake_augment)

    summary = widget._augment_current_parquet_to_path(output)

    assert summary.rows == 2
    assert captured["source_path"] == source
    assert captured["output_path"] == output
    assert captured["kwargs"]["file_ids"] == ["b.swc"]
    assert "2 rows from 1 file ID" in widget._status_label.text


def test_augment_current_parquet_to_path_passes_all_table_file_ids(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    source = tmp_path / "source.parquet"
    output = tmp_path / "augmented.parquet"
    _configure_augmentation_widget(widget, module._SOURCE_ALL, source)
    captured = {}

    def fake_augment(_source_path, _output_path, _flatmap_path, _depth_path, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            output_parquet=output,
            rows=3,
            direct_rows=2,
            mirrored_rows=0,
            unmapped_rows=1,
        )

    monkeypatch.setattr(module, "augment_neuron_parquet_with_flatmap", fake_augment)

    widget._augment_current_parquet_to_path(output)

    assert captured["file_ids"] == ["a.swc", "b.swc"]
    assert "3 rows from 2 file ID" in widget._status_label.text


def _augmented_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "b.swc"],
            "neuron_id": ["a", "a", "b"],
            "subject": ["s", "s", "s"],
            "node_id": [1, 2, 1],
            "parent_id": [-1, 1, -1],
            "type": [1, 3, 1],
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 20.0, 30.0],
            "z": [10.0, 20.0, 30.0],
            "region_id": [1, 1, 2],
            "region_acronym": ["R1", "R1", "R2"],
            "x_flat": [0.0, 1.0, np.nan],
            "y_flat": [0.0, 1.0, 2.0],
            "depth_um": [0.0, 25.0, 10.0],
            "flatmap_valid": [True, True, False],
            "depth_valid": [True, True, True],
            "flatmap_projection_valid": [True, True, False],
            "flatmap_invalid_code": [0, 0, 3],
            "flatmap_lookup_mode": ["direct", "direct", "unmapped"],
        }
    )


def _v3_augmented_nodes() -> pd.DataFrame:
    nodes = _augmented_nodes().drop(
        columns=[
            "x_flat",
            "y_flat",
            "flatmap_valid",
            "flatmap_projection_valid",
            "flatmap_invalid_code",
            "flatmap_lookup_mode",
        ]
    )
    nodes["x_flat_shaped"] = [1.0, 2.0, np.nan]
    nodes["y_flat_shaped"] = [3.0, 4.0, np.nan]
    nodes["flatmap_shaped_valid"] = [True, True, False]
    nodes["flatmap_shaped_projection_valid"] = [True, True, False]
    nodes["flatmap_shaped_invalid_code"] = [0, 0, 3]
    nodes["flatmap_shaped_lookup_mode"] = ["direct", "direct", "unmapped"]
    nodes["x_flat_square"] = [11.0, 12.0, np.nan]
    nodes["y_flat_square"] = [13.0, 14.0, np.nan]
    nodes["flatmap_square_valid"] = [True, True, False]
    nodes["flatmap_square_projection_valid"] = [True, True, False]
    nodes["flatmap_square_invalid_code"] = [0, 0, 3]
    nodes["flatmap_square_lookup_mode"] = ["direct", "direct", "unmapped"]
    nodes["depth_invalid_code"] = [0, 0, 0]
    nodes["depth_lookup_mode"] = ["direct", "mirrored_depth", "mirrored_depth"]
    return nodes


def _configure_projection_widget(widget, module, nodes: pd.DataFrame) -> None:
    widget._database_provider = lambda: types.SimpleNamespace(
        get_neurons_for_rendering=lambda file_ids: nodes[
            nodes["file_id"].isin(file_ids)
        ].reset_index(drop=True),
    )
    widget._table_file_ids_provider = lambda: ["a.swc", "b.swc"]
    widget._selected_file_ids_provider = lambda: []
    widget._source_combo = types.SimpleNamespace(currentData=lambda: module._SOURCE_ALL)
    widget._y_bins_spin = types.SimpleNamespace(value=lambda: 4)
    widget._depth_bin_spin = types.SimpleNamespace(value=lambda: 25)
    widget._exclude_depth_minus_one_cb = types.SimpleNamespace(isChecked=lambda: False)
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._coordinate_mode_combo = types.SimpleNamespace(currentData=lambda: "microns")


def test_project_without_nrrds_uses_augmented_parquet_columns(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._flatmap_path = None
    widget._depth_path = None
    nodes = _augmented_nodes()
    nodes.loc[1, "flatmap_lookup_mode"] = "mirrored_depth"
    _configure_projection_widget(widget, module, nodes)
    captured = {}

    def fake_apply(result, render_result, **kwargs):
        captured["result"] = result
        captured["render_result"] = render_result
        captured["kwargs"] = kwargs

    widget._apply_projection_result = fake_apply

    widget._project()

    assert captured["kwargs"]["flatmap_style"] == "precomputed_parquet"
    assert captured["kwargs"]["coordinate_mode"] == "parquet_columns"
    projected = captured["result"].projected_nodes
    assert projected["valid"].tolist() == [True, True, False]
    assert projected["flatmap_lookup_mode"].tolist() == [
        "direct",
        "mirrored_depth",
        "unmapped",
    ]
    assert projected["invalid_reason"].tolist() == ["", "", "invalid_flatmap"]
    assert captured["render_result"].summary.rendered_nodes == 2
    assert "Parquet flatmap/depth columns" in widget._status_label.text


def test_project_with_nrrds_overrides_augmented_parquet_columns(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    _configure_projection_widget(widget, module, _augmented_nodes())
    calls = {"lookup": 0, "parquet": 0}

    def fake_lookup(self, nodes, **_kwargs):
        calls["lookup"] += 1
        result = types.SimpleNamespace(projected_nodes=nodes, summary=object())
        render = types.SimpleNamespace(
            projected_nodes=nodes,
            summary=types.SimpleNamespace(rendered_nodes=1, total_nodes=1),
        )
        return result, render, None

    def fake_parquet(self, _nodes, **_kwargs):
        calls["parquet"] += 1
        raise AssertionError("Parquet branch should not run when both NRRDs are set")

    widget._project_from_lookup_files = types.MethodType(fake_lookup, widget)
    widget._project_from_parquet_columns = types.MethodType(fake_parquet, widget)
    widget._apply_projection_result = lambda *_args, **_kwargs: None

    widget._project()

    assert calls == {"lookup": 1, "parquet": 0}
    assert "lookup NRRDs" in widget._status_label.text


def _soma_render_result(
    *,
    plane_column: str | None = "depth_bin",
    plane_values: list[float] | None = None,
    file_ids: list[str] | None = None,
    layer_labels: tuple[str, ...] | None = None,
):
    """Build a fake render result in the shape the soma layer reads.

    ``_create_or_update_soma_layer`` takes coordinates from the bin columns the
    render wrote into ``projected_nodes``, so a fake has to carry them rather
    than a ready-made ``points`` array.
    """
    resolved_file_ids = file_ids if file_ids is not None else ["a.swc", "b.swc"]
    rows = len(resolved_file_ids)
    frame = pd.DataFrame(
        {
            "file_id": resolved_file_ids,
            "render_valid": [True] * rows,
            "y_flat_bin": [1.0, 3.0][:rows],
            "x_flat_bin": [2.0, 4.0][:rows],
        }
    )
    if plane_column is not None:
        frame[plane_column] = (
            plane_values if plane_values is not None else [0.0, 1.0][:rows]
        )
    summary_fields = {
        "rendered_nodes": rows,
        "total_nodes": rows,
        "to_dict": lambda: {"rendered_nodes": rows},
    }
    if layer_labels is not None:
        summary_fields["layer_labels"] = layer_labels
    return types.SimpleNamespace(
        projected_nodes=frame,
        summary=types.SimpleNamespace(**summary_fields),
    )


def test_add_soma_projects_only_soma_nodes_to_dedicated_layer(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    _configure_projection_widget(widget, module, _augmented_nodes())
    captured = {}

    def fake_lookup(self, nodes, **_kwargs):
        captured["nodes"] = nodes
        render = _soma_render_result()
        result = types.SimpleNamespace(
            projected_nodes=nodes,
            summary=types.SimpleNamespace(to_dict=lambda: {"total_nodes": 2}),
        )
        return result, render, None

    widget._project_from_lookup_files = types.MethodType(fake_lookup, widget)

    widget._add_soma()

    # Only soma nodes (type == 1) are handed to the projector.
    assert captured["nodes"]["type"].tolist() == [1, 1]
    assert captured["nodes"]["node_id"].tolist() == [1, 1]

    # A dedicated soma layer is added and tracked, separate from the main
    # projection layer.
    soma_layers = [
        layer
        for layer in widget._viewer.layers
        if layer.name == module._SOMA_POINTS_LAYER_NAME
    ]
    assert len(soma_layers) == 1
    assert widget._soma_layer is soma_layers[0]
    assert soma_layers[0].metadata["flatmap_soma_only"] is True
    assert soma_layers[0].metadata["napari_neuron_navigator_space"] == "flatmap"
    np.testing.assert_allclose(
        soma_layers[0].face_color,
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.5]],
    )
    assert "soma node" in widget._status_label.text


def test_add_soma_uses_duckdb_soma_query_not_full_node_scan(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    all_nodes = _augmented_nodes()
    soma_nodes = all_nodes[all_nodes["type"] == 1].reset_index(drop=True)
    calls = {"soma": 0, "full": 0}

    def _get_soma(file_ids):
        calls["soma"] += 1
        return soma_nodes[soma_nodes["file_id"].isin(file_ids)].reset_index(drop=True)

    def _get_full(_file_ids):
        calls["full"] += 1
        raise AssertionError(
            "Full node scan must not run when a soma query is available"
        )

    widget._database_provider = lambda: types.SimpleNamespace(
        get_soma_nodes_for_rendering=_get_soma,
        get_neurons_for_rendering=_get_full,
    )
    widget._table_file_ids_provider = lambda: ["a.swc", "b.swc"]
    widget._selected_file_ids_provider = lambda: []
    widget._source_combo = types.SimpleNamespace(currentData=lambda: module._SOURCE_ALL)
    widget._y_bins_spin = types.SimpleNamespace(value=lambda: 4)
    widget._depth_bin_spin = types.SimpleNamespace(value=lambda: 25)
    widget._exclude_depth_minus_one_cb = types.SimpleNamespace(isChecked=lambda: False)
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._coordinate_mode_combo = types.SimpleNamespace(currentData=lambda: "microns")
    captured = {}

    def fake_lookup(self, nodes, **_kwargs):
        captured["nodes"] = nodes
        render = _soma_render_result()
        result = types.SimpleNamespace(
            projected_nodes=nodes,
            summary=types.SimpleNamespace(to_dict=lambda: {}),
        )
        return result, render, None

    widget._project_from_lookup_files = types.MethodType(fake_lookup, widget)

    widget._add_soma()

    assert calls == {"soma": 1, "full": 0}
    assert captured["nodes"]["type"].tolist() == [1, 1]


def test_add_soma_reuses_existing_layer_on_reprojection(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    _configure_projection_widget(widget, module, _augmented_nodes())

    def fake_lookup(self, nodes, **_kwargs):
        render = _soma_render_result()
        result = types.SimpleNamespace(
            projected_nodes=nodes,
            summary=types.SimpleNamespace(to_dict=lambda: {}),
        )
        return result, render, None

    widget._project_from_lookup_files = types.MethodType(fake_lookup, widget)

    widget._add_soma()
    first = widget._soma_layer
    widget._add_soma()

    assert widget._soma_layer is first
    soma_layers = [
        layer
        for layer in widget._viewer.layers
        if layer.name == module._SOMA_POINTS_LAYER_NAME
    ]
    assert len(soma_layers) == 1


def _add_soma_with_render(module, widget, render, *, nodes=None):
    """Run Add Soma with a stubbed projection returning ``render``."""
    _configure_projection_widget(widget, module, nodes or _augmented_nodes())

    def fake_lookup(self, soma_nodes, **_kwargs):
        return (
            types.SimpleNamespace(
                projected_nodes=soma_nodes,
                summary=types.SimpleNamespace(to_dict=lambda: {}),
            ),
            render,
            None,
        )

    widget._project_from_lookup_files = types.MethodType(fake_lookup, widget)
    widget._add_soma()
    return widget._soma_layer


def test_add_soma_uses_allen_layer_planes_in_allen_mode(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_ALLEN_LAYERS
    )

    layer = _add_soma_with_render(
        module,
        widget,
        _soma_render_result(
            plane_column="allen_layer_index",
            plane_values=[0.0, 3.0],
            layer_labels=("L1", "L2/3", "L4", "L5", "L6a", "L6b"),
        ),
    )

    # Axis 0 is the Allen layer index, not a depth bin: a depth bin of 30-80
    # would place the somas outside the six-plane stack entirely.
    np.testing.assert_array_equal(layer.data[:, 0], [0.0, 3.0])
    assert layer.data.shape == (2, 3)
    assert widget._viewer.dims.ndisplay == 2
    assert layer.axis_labels == ("Allen layer", "Flatmap Y", "Flatmap X")
    assert layer.metadata["flatmap_plane_mode"] == "allen_layers"
    assert layer.metadata["flatmap_soma_space_render_mode"] == (
        module._RENDER_ALLEN_LAYERS
    )
    assert layer.metadata["allen_layer_labels"][1] == "L2/3"


def test_add_soma_preserves_the_allen_plane_caption(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    _render_allen_layer_stack(module, widget)
    assert widget._viewer.text_overlay.text == "Allen layer: L1  (plane 1 of 6)"

    _add_soma_with_render(
        module,
        widget,
        _soma_render_result(
            plane_column="allen_layer_index",
            plane_values=[0.0, 1.0],
            layer_labels=("L1", "L2/3", "L4", "L5", "L6a", "L6b"),
        ),
    )

    # A soma layer without flatmap axis captions used to read as a foreign
    # layer and wipe all of this.
    assert widget._viewer.dims.axis_labels == (
        "Allen layer",
        "Flatmap Y",
        "Flatmap X",
    )
    assert widget._viewer.axes.visible is True
    assert widget._viewer.text_overlay.visible is True
    assert widget._viewer.text_overlay.text == "Allen layer: L1  (plane 1 of 6)"
    assert widget._viewer.dims.ndisplay == 2


@pytest.mark.parametrize(
    "render_mode_name", ["_RENDER_FLAT_HEATMAP", "_RENDER_FLAT_VECTOR"]
)
def test_add_soma_uses_two_dimensional_points_in_flat_modes(
    monkeypatch, render_mode_name
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    render_mode = getattr(module, render_mode_name)
    widget._render_mode_combo = types.SimpleNamespace(currentData=lambda: render_mode)

    layer = _add_soma_with_render(
        module,
        widget,
        _soma_render_result(plane_column=None),
    )

    assert layer.data.shape == (2, 2)
    np.testing.assert_array_equal(layer.data, [[1.0, 2.0], [3.0, 4.0]])
    assert widget._viewer.dims.ndisplay == 2
    assert layer.axis_labels == ("Flatmap Y", "Flatmap X")
    assert layer.metadata["flatmap_plane_mode"] == "flat"


def test_add_soma_keeps_depth_space_in_depth_modes(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_HEATMAP
    )

    layer = _add_soma_with_render(
        module,
        widget,
        _soma_render_result(plane_column="depth_bin", plane_values=[0.0, 7.0]),
    )

    np.testing.assert_array_equal(layer.data[:, 0], [0.0, 7.0])
    assert layer.data.shape == (2, 3)
    assert widget._viewer.dims.ndisplay == 3
    assert layer.axis_labels == ("Depth bin", "Flatmap Y", "Flatmap X")
    assert layer.metadata["flatmap_plane_mode"] == "depth"


def test_add_soma_in_allen_mode_without_region_id_reports_the_fix(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_ALLEN_LAYERS
    )
    nodes = _augmented_nodes().drop(columns=["region_id"])

    def fail_lookup(self, _nodes, **_kwargs):
        raise AssertionError("Allen soma projection must not fall back to depth space")

    _configure_projection_widget(widget, module, nodes)
    widget._project_from_lookup_files = types.MethodType(fail_lookup, widget)

    widget._add_soma()

    assert "region_id" in widget._status_label.text
    assert widget._viewer.layers == []
    assert getattr(widget, "_soma_layer", None) is None


def test_render_mode_change_removes_a_stale_soma_layer(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_HEATMAP
    )
    layer = _add_soma_with_render(
        module,
        widget,
        _soma_render_result(plane_column="depth_bin"),
    )
    assert layer in widget._viewer.layers
    widget._region_surfaces_layers = []
    widget._region_outlines_layers = []
    widget._cache_grid_locked = False
    widget._heatmap_color_mode_combo = _DummyButton()
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_HEATMAP
    )

    widget._on_render_mode_changed()

    # Depth-bin soma coordinates are meaningless in the new 2D space.
    assert layer not in widget._viewer.layers
    assert widget._soma_layer is None


def test_add_soma_without_soma_nodes_reports_and_adds_no_layer(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    nodes = _augmented_nodes()
    nodes["type"] = [3, 3, 3]
    _configure_projection_widget(widget, module, nodes)

    def fail_lookup(self, _nodes, **_kwargs):
        raise AssertionError("Projection should not run without soma nodes")

    widget._project_from_lookup_files = types.MethodType(fail_lookup, widget)

    widget._add_soma()

    assert "No soma nodes" in widget._status_label.text
    assert widget._viewer.layers == []
    assert getattr(widget, "_soma_layer", None) is None


def test_explicit_precomputed_v3_square_never_loads_selected_nrrds(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    nodes = _v3_augmented_nodes()
    # Unrelated custom columns retained by whole-Parquet preprocessing must not
    # override the namespaced v3 projection validity/reason columns.
    nodes["valid"] = [False, False, True]
    nodes["invalid_reason"] = ["out_of_bounds"] * 3
    _configure_projection_widget(widget, module, nodes)
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_square")
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    # Points render mode drives the per-node precomputed pipeline this test
    # exercises; heatmap mode now uses the DuckDB volume fast path instead.
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_POINTS
    )
    widget._validate_precomputed_parquet_contract = lambda _nodes: None
    widget._flatmap_path = Path("selected-but-unused.nrrd")
    widget._depth_path = Path("selected-but-unused-depth.nrrd")
    widget._canonical_render_bounds = lambda: {
        "x_bounds": (0.0, 20.0),
        "y_bounds": (0.0, 20.0),
        "depth_range_um": (0.0, 50.0),
    }
    monkeypatch.setattr(
        module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("precomputed projection must not load NRRDs")
        ),
    )
    captured = {}
    widget._apply_projection_result = lambda result, render, **kwargs: captured.update(
        result=result,
        render=render,
        kwargs=kwargs,
    )

    widget._project()

    projected = captured["result"].projected_nodes
    assert projected["x_flat"].tolist()[:2] == [11.0, 12.0]
    assert projected["y_flat"].tolist()[:2] == [13.0, 14.0]
    assert projected["flatmap_lookup_mode"].tolist() == [
        "direct",
        "mirrored_depth",
        "unmapped",
    ]
    assert projected["valid"].tolist() == [True, True, False]
    assert projected["invalid_reason"].tolist() == ["", "", "invalid_flatmap"]
    assert captured["kwargs"]["flatmap_style"] == "both_square"


def test_explicit_precomputed_rejects_partial_v3_columns(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    nodes = _v3_augmented_nodes().drop(columns=["depth_invalid_code"])
    _configure_projection_widget(widget, module, nodes)
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    # Points render mode keeps the per-node validation path under test.
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_POINTS
    )
    applied: list[object] = []
    widget._apply_projection_result = lambda *args, **kwargs: applied.append(
        (args, kwargs)
    )

    widget._project()

    assert applied == []
    assert "missing required flatmap column(s): ['depth_invalid_code']" in (
        widget._status_label.text
    )
    assert "Regenerate it with Prepare Whole Parquet" in widget._status_label.text


@pytest.mark.parametrize(
    ("transform_info", "expected_message"),
    [
        (
            types.SimpleNamespace(
                format_version=0,
                lookup_set_id=None,
                metadata=None,
            ),
            "complete version-3 metadata with a lookup-set ID",
        ),
        (
            types.SimpleNamespace(
                format_version=3,
                lookup_set_id="lookup-set-id",
                metadata={"version": 3, "canonical_bounds": {}},
            ),
            "no valid canonical bounds for both_shaped",
        ),
    ],
)
def test_explicit_precomputed_rejects_incomplete_canonical_metadata(
    monkeypatch,
    transform_info,
    expected_message: str,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    _configure_projection_widget(widget, module, _v3_augmented_nodes())
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._current_source_parquet_path = lambda: Path("neurons.parquet")
    # Points render mode keeps the per-node validation path under test.
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_POINTS
    )
    monkeypatch.setattr(
        module,
        "read_flatmap_parquet_transform_info",
        lambda _path: transform_info,
    )
    applied: list[object] = []
    widget._apply_projection_result = lambda *args, **kwargs: applied.append(
        (args, kwargs)
    )

    widget._project()

    assert applied == []
    assert expected_message in widget._status_label.text


def test_precomputed_cache_grid_values_are_used_without_ui_rounding(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    style_cache = types.SimpleNamespace(
        grid_spec={"y_bins": 3, "x_bins": 5, "depth_bin_um": 12.3456}
    )
    widget._active_cache_profile = types.SimpleNamespace(
        style=lambda _style: style_cache
    )
    # The UI controls cannot represent this small/fractional profile exactly.
    widget._y_bins_spin = types.SimpleNamespace(value=lambda: 16)
    widget._depth_bin_spin = types.SimpleNamespace(value=lambda: 12.346)

    # Both counts come from the profile verbatim, never re-derived from bounds.
    assert widget._current_y_bins() == 3
    assert widget._current_x_bins() == 5
    assert widget._current_depth_bin_um() == 12.3456


def test_project_from_lookup_files_uses_depth_mirror_fallback(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    nodes = pd.DataFrame(
        {
            "file_id": ["a.swc"],
            "neuron_id": ["a"],
            "subject": ["s"],
            "node_id": [1],
            "parent_id": [-1],
            "type": [3],
            "x": [1.0],
            "y": [0.0],
            "z": [0.0],
            "region_id": [1],
            "region_acronym": ["R1"],
        }
    )
    _configure_projection_widget(widget, module, nodes)
    widget._coordinate_mode_combo = types.SimpleNamespace(
        currentData=lambda: module.COORDINATE_MODE_VOXELS
    )

    grid = np.indices((4, 4, 4), dtype=float)
    flatmap = np.stack((grid[0] + 0.25, grid[1] + 0.5), axis=-1).astype(np.float32)
    depth = (grid[2] + 100.0).astype(np.float32)
    depth[1, 0, 0] = -1.0
    volume_set = types.SimpleNamespace(
        flatmap=flatmap,
        depth=depth,
        flatmap_path=Path("flatmap.nrrd"),
        depth_path=Path("depth.nrrd"),
        space_directions=None,
        space_origin=None,
    )
    monkeypatch.setattr(module, "load_flatmap_volume_set", lambda *_args: volume_set)

    result, render_result, _lookup_stats = widget._project_from_lookup_files(nodes)

    projected = result.projected_nodes
    assert projected["flatmap_lookup_mode"].tolist() == ["mirrored_depth"]
    assert projected["voxel_k"].tolist() == [0]
    assert projected["x_flat"].tolist() == pytest.approx([1.25])
    assert projected["depth_um"].tolist() == pytest.approx([103.0])
    assert result.summary.mirrored_depth_lookup_nodes == 1
    assert result.summary.mirrored_lookup_nodes == 0
    assert render_result.summary.rendered_nodes == 1
    summary_text = widget._format_render_summary(
        result.summary,
        render_result.summary,
    )
    assert "Lookup direct/mirrored-depth/mirrored/unmapped" in summary_text
    assert summary_text.endswith("0/1/0/0")


def test_project_updates_progress_bar_and_restores_button(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._flatmap_path = None
    widget._depth_path = None
    _configure_projection_widget(widget, module, _augmented_nodes())
    progress_events = []
    original_set_progress = widget._set_projection_progress

    def record_progress(message: str, current: int, total: int) -> None:
        progress_events.append((message, current, total))
        original_set_progress(message, current, total)

    widget._set_projection_progress = record_progress
    widget._apply_projection_result = lambda *_args, **_kwargs: None

    widget._project()

    assert progress_events[0] == ("Querying neuron rows...", 0, 4)
    assert ("Reading precomputed flatmap columns...", 1, 4) in progress_events
    assert ("Building flatmap render data...", 2, 4) in progress_events
    assert progress_events[-1] == ("Done", 4, 4)
    assert widget._project_btn.enabled is True
    assert widget._projection_progress_bar.visible is False
    assert widget._projection_progress_bar.range == (0, 1)
    assert widget._projection_progress_bar.value == 0


def test_latest_flatmap_correlation_source_requires_heatmap_render(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._last_projected_nodes = pd.DataFrame(
        {
            "file_id": ["a.swc", "b.swc"],
            "render_valid": [True, True],
            "depth_bin": [0, 1],
            "y_flat_bin": [0, 1],
            "x_flat_bin": [0, 1],
        }
    )
    widget._last_render_summary = module.FlatmapRenderSummary(
        total_nodes=2,
        flatmap_valid_nodes=2,
        depth_valid_nodes=2,
        depth_minus_one_nodes=0,
        rendered_nodes=2,
        excluded_depth_minus_one_nodes=0,
        nonzero_voxels=2,
        traces_represented=2,
        y_bins=4,
        x_bins=4,
        depth_bins=2,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=25.0,
        includes_depth_minus_one_plane=False,
    )
    widget._last_volume_shape = (2, 4, 4)
    widget._last_input_file_ids = ("a.swc", "b.swc")
    widget._last_flatmap_style = "both"
    widget._last_coordinate_mode = "microns"
    widget._last_flatmap_path = str(widget._flatmap_path)
    widget._last_depth_path = str(widget._depth_path)
    widget._last_lookup_stats = module.FlatmapLookupStats(
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
        depth_range_um=(0.0, 25.0),
        flatmap_valid_voxels=2,
        depth_valid_voxels=2,
        flatmap_shape=(1, 1, 1, 2),
        depth_shape=(1, 1, 1),
        flatmap_dtype="float32",
        depth_dtype="float32",
        invalid_zero_sentinel=False,
        invalid_negative_one_sentinel=True,
    )

    widget._last_render_mode = module._RENDER_POINTS
    assert widget.latest_flatmap_correlation_source() is None

    # Flatmap voxel correlation is defined on the depth grid, so a
    # depth-collapsed render is not a source for it.
    for flat_mode in (module._RENDER_FLAT_HEATMAP, module._RENDER_FLAT_VECTOR):
        widget._last_render_mode = flat_mode
        assert widget.latest_flatmap_correlation_source() is None

    widget._last_render_mode = module._RENDER_HEATMAP
    assert widget.latest_flatmap_correlation_source() is None

    layer = _DummyLayer(
        np.zeros(widget._last_volume_shape, dtype=np.float32),
        name=module._HEATMAP_LAYER_NAME,
        metadata={"flatmap_render_mode": module._RENDER_HEATMAP},
    )
    widget._projection_layer = layer
    widget._viewer.layers.append(layer)
    source = widget.latest_flatmap_correlation_source()

    assert source is not None
    assert source.volume_shape == (2, 4, 4)
    assert source.input_file_ids == ("a.swc", "b.swc")
    assert source.y_bins == 4
    assert source.x_bins == 4
    assert source.depth_bin_um == 25.0
    assert source.mirror_depth_fallback is True
    assert source.mirror_coord_axis == 2


def test_project_without_nrrds_requires_augmented_parquet_columns(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._flatmap_path = None
    widget._depth_path = None
    nodes = _augmented_nodes().drop(columns=["x_flat", "y_flat", "depth_um"])
    _configure_projection_widget(widget, module, nodes)

    widget._project()

    assert (
        "augmented Parquet with x_flat, y_flat, and depth_um"
        in widget._status_label.text
    )


def test_create_heatmap_layer_uses_metadata_and_3d_focus(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    projected = pd.DataFrame({"file_id": ["a.swc", "b.swc"]})
    volume = np.zeros((2, 4, 4), dtype=np.float32)
    volume[0, 1, 2] = 1.0
    volume[1, 3, 0] = 2.0
    render_summary = module.FlatmapRenderSummary(
        total_nodes=3,
        flatmap_valid_nodes=3,
        depth_valid_nodes=2,
        depth_minus_one_nodes=1,
        rendered_nodes=3,
        excluded_depth_minus_one_nodes=0,
        nonzero_voxels=2,
        traces_represented=2,
        y_bins=4,
        x_bins=4,
        depth_bins=2,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=25.0,
        includes_depth_minus_one_plane=True,
    )
    render_result = module.FlatmapRenderResult(
        projected_nodes=projected,
        volume=volume,
        points=np.asarray([[0.0, 1.0, 2.0], [1.0, 3.0, 0.0]]),
        point_file_ids=["a.swc", "b.swc"],
        summary=render_summary,
    )
    summary = module.ProjectionSummary(3, 2, 1, 0, 1, 0, 0, 2, 1)
    old_layer = _DummyLayer([], name="Isocortex Flatmap Traces")
    widget._viewer.layers.append(old_layer)

    layer = widget._create_or_update_render_layer(
        render_result,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert layer.name == "Isocortex Flatmap Heatmap"
    assert layer.metadata["projection_kind"] == "isocortex_flatmap"
    assert layer.metadata["napari_neuron_navigator_space"] == "flatmap"
    assert layer.metadata["flatmap_render_mode"] == "heatmap"
    assert layer.metadata["flatmap_heatmap_color_mode"] == "single"
    assert layer.metadata["render_summary"]["rendered_nodes"] == 3
    assert layer.metadata["flatmap_heatmap_contrast_limits"] == (0.0, 2.0)
    assert layer.colormap == "hot"
    assert layer.contrast_limits == (0.0, 2.0)
    assert layer._napari_neuron_navigator_flatmap_projected_nodes is projected
    assert layer._napari_neuron_navigator_flatmap_summary is summary
    assert layer._napari_neuron_navigator_flatmap_render_summary is render_summary
    assert old_layer not in widget._viewer.layers
    assert widget._viewer.dims.ndisplay == 3
    assert layer.slice_dims_calls[-1] == (widget._viewer.dims, True)
    assert widget._viewer.camera.center == (0.5, 2.0, 1.0)
    assert widget._viewer.camera.zoom == 300.0


def _grouped_volume(module, *, peak: float = 200.0):
    """One neuron's volume: a dense soma bin plus a one-node-per-bin projection."""
    volume = np.zeros((2, 8, 8), dtype=np.float32)
    volume[0, 4, 4] = peak
    volume[0, 4, 5:8] = 1.0
    return module.FlatmapGroupedVolume(
        group_key="a.swc",
        label="a.swc",
        source_file_ids=("a.swc",),
        volume=volume,
        rendered_nodes=int(peak) + 3,
        nonzero_voxels=4,
    )


def test_individual_heatmap_opens_at_a_fraction_of_its_own_maximum(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    layer = widget._add_grouped_heatmap_layer(
        _grouped_volume(module),
        {},
        np.asarray([1.0, 0.0, 0.0, 1.0]),
        heatmap_color_mode=module._HEATMAP_COLOR_INDIVIDUAL,
    )

    fraction = module._INDIVIDUAL_HEATMAP_CONTRAST_FRACTION
    assert layer.contrast_limits == (0.0, 200.0 * fraction)
    assert layer.metadata["flatmap_heatmap_contrast_limits"] == (0.0, 200.0 * fraction)
    # The slider still spans the real data, so the dense core stays reachable.
    assert layer.contrast_limits_range == (0.0, 200.0)
    assert layer.metadata["flatmap_heatmap_contrast_limits_range"] == (0.0, 200.0)


@pytest.mark.parametrize(
    "color_mode_name", ["_HEATMAP_COLOR_CLUSTER", "_HEATMAP_COLOR_SINGLE"]
)
def test_non_individual_grouped_heatmaps_keep_the_full_range(
    monkeypatch, color_mode_name
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    layer = widget._add_grouped_heatmap_layer(
        _grouped_volume(module),
        {},
        np.asarray([1.0, 0.0, 0.0, 1.0]),
        heatmap_color_mode=getattr(module, color_mode_name),
    )

    # A cluster group aggregates many neurons, so it is not dominated by one
    # neuron's soma the way a per-neuron layer is.
    assert layer.contrast_limits == (0.0, 200.0)
    assert layer.contrast_limits_range == (0.0, 200.0)


def test_single_color_heatmap_keeps_the_full_range(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    volume = _grouped_volume(module).volume

    layer = widget._create_or_update_heatmap_layer_from_volume(None, volume, {})

    assert layer.contrast_limits == (0.0, 200.0)
    assert layer.metadata["flatmap_heatmap_contrast_limits"] == (0.0, 200.0)


def test_individual_heatmap_contrast_survives_a_napari_reset(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = widget._add_grouped_heatmap_layer(
        _grouped_volume(module),
        {},
        np.asarray([1.0, 0.0, 0.0, 1.0]),
        heatmap_color_mode=module._HEATMAP_COLOR_INDIVIDUAL,
    )
    layer._slice_input = types.SimpleNamespace(ndisplay=3)

    layer.reset_contrast_limits()
    layer.reset_contrast_limits_range()

    fraction = module._INDIVIDUAL_HEATMAP_CONTRAST_FRACTION
    assert layer.contrast_limits == (0.0, 200.0 * fraction)
    # The range must not collapse onto the opening window.
    assert layer.contrast_limits_range == (0.0, 200.0)


def test_individual_heatmap_keeps_the_full_range_for_a_flat_volume(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    empty = module.FlatmapGroupedVolume(
        group_key="a.swc",
        label="a.swc",
        source_file_ids=("a.swc",),
        volume=np.zeros((2, 4, 4), dtype=np.float32),
        rendered_nodes=0,
        nonzero_voxels=0,
    )

    layer = widget._add_grouped_heatmap_layer(
        empty,
        {},
        np.asarray([1.0, 0.0, 0.0, 1.0]),
        heatmap_color_mode=module._HEATMAP_COLOR_INDIVIDUAL,
    )

    # An all-zero volume has no maximum to scale, so the fallback range stands
    # and the limits stay ascending rather than collapsing to (0, 0).
    assert layer.contrast_limits == (0.0, 1.0)
    assert layer.contrast_limits_range == (0.0, 1.0)


def test_flatmap_render_layers_use_display_viewer_provider(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    main_viewer = widget._viewer
    display_viewer = _DummyViewer()
    widget._display_viewer_provider = lambda create=True: (
        display_viewer if create else display_viewer
    )
    projected = pd.DataFrame({"file_id": ["a.swc"]})
    volume = np.zeros((1, 4, 4), dtype=np.float32)
    volume[0, 1, 2] = 1.0
    render_result = module.FlatmapRenderResult(
        projected_nodes=projected,
        volume=volume,
        points=np.asarray([[0.0, 1.0, 2.0]]),
        point_file_ids=["a.swc"],
        summary=_simple_render_summary(module),
    )

    layer = widget._create_or_update_render_layer(
        render_result,
        _simple_projection_summary(module),
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert main_viewer.layers == []
    assert display_viewer.layers == [layer]
    assert display_viewer.dims.ndisplay == 3
    assert layer.slice_dims_calls[-1] == (display_viewer.dims, True)
    assert display_viewer.camera.center == (0.0, 1.0, 2.0)


@pytest.mark.parametrize(
    ("render_mode", "heatmap_color_mode"),
    [
        ("heatmap", "single"),
        ("heatmap", "individual"),
        ("points", "single"),
    ],
)
def test_first_render_reports_ready_after_layer_focus(
    monkeypatch,
    render_mode: str,
    heatmap_color_mode: str,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    display_viewer = _DummyViewer()
    widget._display_viewer_provider = lambda create=True: display_viewer
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: heatmap_color_mode
    )
    events = []
    original_focus = widget._focus_projection_view

    def record_focus(layer, data, **kwargs) -> None:
        original_focus(layer, data, **kwargs)
        events.append(("focused", layer))

    def record_ready(viewer, layer) -> None:
        assert viewer is display_viewer
        assert layer in display_viewer.layers
        assert events[-1] == ("focused", layer)
        events.append(("ready", layer))

    widget._focus_projection_view = record_focus
    widget._display_viewer_ready_callback = record_ready

    layer = widget._create_or_update_render_layer(
        _binned_render_result(module),
        _simple_projection_summary(module, total_nodes=4),
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode=render_mode,
    )

    assert layer is not None
    assert events == [("focused", layer), ("ready", layer)]


def test_display_failure_callback_receives_current_viewer(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    display_viewer = _DummyViewer()
    widget._display_viewer_provider = lambda create=True: display_viewer
    calls = []
    widget._display_viewer_failed_callback = lambda viewer, reason: calls.append(
        (viewer, reason)
    )

    widget._notify_display_viewer_failed("projection_failed")

    assert calls == [(display_viewer, "projection_failed")]


def test_release_display_viewer_clears_only_matching_viewer_layer_handles(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    viewer = _DummyViewer()
    other_viewer = _DummyViewer()
    projection_layer = object()
    labels_layer = object()
    surfaces = [object()]
    outlines = [object()]
    projected_nodes = pd.DataFrame({"file_id": ["a.swc"]})
    cache_profile = object()
    widget._last_display_viewer = viewer
    widget._projection_layer = projection_layer
    widget._region_labels_layer = labels_layer
    widget._region_surfaces_layers = surfaces
    widget._region_outlines_layers = outlines
    widget._last_projected_nodes = projected_nodes
    widget._active_cache_profile = cache_profile

    assert widget._release_display_viewer(other_viewer) is False
    assert widget._last_display_viewer is viewer
    assert widget._projection_layer is projection_layer
    assert widget._region_labels_layer is labels_layer
    assert widget._region_surfaces_layers is surfaces
    assert widget._region_outlines_layers is outlines

    assert widget._release_display_viewer(viewer) is True
    assert widget._last_display_viewer is None
    assert widget._projection_layer is None
    assert widget._region_labels_layer is None
    assert widget._region_surfaces_layers == []
    assert widget._region_outlines_layers == []
    assert widget._last_projected_nodes is projected_nodes
    assert widget._active_cache_profile is cache_profile


def test_begin_new_display_viewer_preserves_old_layers_and_resets_active_handles(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    first = _DummyViewer()
    second = _DummyViewer()
    first_layer = first.add_image(
        np.ones((2, 2)),
        name="first comparison",
        metadata={"flatmap_render_mode": module._RENDER_FLAT_HEATMAP},
    )
    state = {"viewer": first}

    def activate_second():
        state["viewer"] = second
        return second

    widget._display_viewer_provider = lambda create=True: state["viewer"]
    widget._new_display_viewer_callback = activate_second
    widget._last_display_viewer = first
    widget._projection_layer = first_layer
    widget._soma_layer = object()
    widget._region_labels_layer = object()
    widget._region_surfaces_layers = [object()]
    widget._region_outlines_layers = [object()]

    assert widget._begin_new_display_viewer() is second

    assert first.layers == [first_layer]
    assert second.layers == []
    assert widget._last_display_viewer is second
    assert widget._projection_layer is None
    assert widget._soma_layer is None
    assert widget._region_labels_layer is None
    assert widget._region_surfaces_layers == []
    assert widget._region_outlines_layers == []


def test_project_in_new_window_activates_new_viewer_before_fast_worker(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    first = _DummyViewer()
    second = _DummyViewer()
    state = {"viewer": first}
    widget._display_viewer_provider = lambda create=True: state["viewer"]

    def activate_second():
        state["viewer"] = second
        return second

    widget._new_display_viewer_callback = activate_second
    widget._last_display_viewer = first
    widget._projection_source_combo = _DummyDataCombo(
        module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._render_mode_combo = _DummyDataCombo(module._RENDER_HEATMAP)
    started_in = []
    widget._start_precomputed_heatmap_worker = lambda: started_in.append(
        widget._current_display_viewer()
    )

    widget._project(new_window=True)

    assert started_in == [second]
    assert widget._last_display_viewer is second


def test_flatmap_heatmap_selector_lists_only_gamma_adjustable_heatmaps(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    heatmap_3d = _DummyLayer(
        np.ones((2, 2, 2)),
        name="3D heatmap",
        metadata={"flatmap_render_mode": module._RENDER_HEATMAP},
    )
    heatmap_2d = _DummyLayer(
        np.ones((2, 2)),
        name="2D heatmap",
        metadata={"flatmap_render_mode": module._RENDER_FLAT_HEATMAP},
    )
    allen_heatmap = _DummyLayer(
        np.ones((6, 2, 2)),
        name="Allen heatmap",
        metadata={"flatmap_render_mode": module._RENDER_ALLEN_LAYERS},
    )
    points = _DummyLayer(
        np.ones((2, 3)),
        name="Flatmap points",
        metadata={"flatmap_render_mode": module._RENDER_POINTS},
    )
    region_labels = _DummyLayer(
        np.ones((2, 2)),
        name="Region labels",
        metadata={"projection_kind": "flatmap_region_labels"},
    )
    widget._viewer.layers.extend(
        [heatmap_3d, heatmap_2d, allen_heatmap, points, region_labels]
    )

    layers = widget._flatmap_heatmap_layers()

    assert layers == [heatmap_3d, heatmap_2d, allen_heatmap]


def test_flatmap_heatmap_gamma_actions_apply_to_multi_selection(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    selected_a = _DummyLayer(
        np.ones((2, 2)),
        name="Neuron A",
        metadata={"flatmap_render_mode": module._RENDER_FLAT_HEATMAP},
    )
    selected_b = _DummyLayer(
        np.ones((2, 2)),
        name="Neuron B",
        metadata={"flatmap_render_mode": module._RENDER_FLAT_HEATMAP},
        gamma=0.8,
    )
    unselected = _DummyLayer(
        np.ones((2, 2)),
        name="Neuron C",
        metadata={"flatmap_render_mode": module._RENDER_FLAT_HEATMAP},
        gamma=0.6,
    )
    widget._viewer.layers.extend([selected_a, selected_b, unselected])
    widget._flatmap_heatmap_layer_list = _DummyListWidget()
    widget._flatmap_heatmap_gamma_status_label = _DummyLabel()
    widget._flatmap_enhance_fine_projections_btn = _DummyButton()
    widget._flatmap_reset_gamma_btn = _DummyButton()

    widget._refresh_flatmap_heatmap_layer_list()
    widget._flatmap_heatmap_layer_list.item(0).setSelected(True)
    widget._flatmap_heatmap_layer_list.item(1).setSelected(True)
    widget._update_flatmap_heatmap_gamma_controls()

    assert widget._flatmap_enhance_fine_projections_btn.enabled is True
    assert widget._flatmap_reset_gamma_btn.enabled is True

    widget._enhance_selected_flatmap_heatmap_projections()

    assert selected_a.gamma == pytest.approx(0.2)
    assert selected_b.gamma == pytest.approx(0.2)
    assert unselected.gamma == pytest.approx(0.6)
    assert widget._flatmap_heatmap_gamma_status_label.text == (
        "Enhanced fine projections on 2 flatmap heatmap layer(s) (gamma 0.20)."
    )

    widget._reset_selected_flatmap_heatmap_gamma()

    assert selected_a.gamma == pytest.approx(1.0)
    assert selected_b.gamma == pytest.approx(1.0)
    assert unselected.gamma == pytest.approx(0.6)
    assert widget._flatmap_heatmap_gamma_status_label.text == (
        "Reset gamma on 2 flatmap heatmap layer(s) (gamma 1.00)."
    )


def test_flatmap_heatmap_gamma_controls_disable_without_selection(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    heatmap = _DummyLayer(
        np.ones((2, 2)),
        name="Heatmap",
        metadata={"flatmap_render_mode": module._RENDER_FLAT_HEATMAP},
    )
    widget._viewer.layers.append(heatmap)
    widget._flatmap_heatmap_layer_list = _DummyListWidget()
    widget._flatmap_heatmap_gamma_status_label = _DummyLabel()
    widget._flatmap_enhance_fine_projections_btn = _DummyButton()
    widget._flatmap_reset_gamma_btn = _DummyButton()

    widget._refresh_flatmap_heatmap_layer_list()

    assert widget._flatmap_enhance_fine_projections_btn.enabled is False
    assert widget._flatmap_reset_gamma_btn.enabled is False
    assert widget._flatmap_heatmap_gamma_status_label.text == (
        "1 flatmap heatmap layer(s) available."
    )

    widget._enhance_selected_flatmap_heatmap_projections()

    assert heatmap.gamma == pytest.approx(1.0)
    assert widget._flatmap_heatmap_gamma_status_label.text == (
        "Select at least one flatmap heatmap layer."
    )


def _simple_projection_summary(module, total_nodes: int = 1):
    return module.ProjectionSummary(total_nodes, total_nodes, 0, 0, 0, 0, 0, 1, 0)


def _simple_render_summary(
    module,
    total_nodes: int = 1,
    *,
    includes_depth_minus_one_plane: bool = True,
):
    return module.FlatmapRenderSummary(
        total_nodes=total_nodes,
        flatmap_valid_nodes=total_nodes,
        depth_valid_nodes=total_nodes,
        depth_minus_one_nodes=0,
        rendered_nodes=total_nodes,
        excluded_depth_minus_one_nodes=0,
        nonzero_voxels=total_nodes,
        traces_represented=1,
        y_bins=4,
        x_bins=4,
        depth_bins=1,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=25.0,
        includes_depth_minus_one_plane=includes_depth_minus_one_plane,
    )


def _simple_allen_layer_summary(module, total_nodes: int = 2):
    return module.AllenLayerStackSummary(
        total_nodes=total_nodes,
        flatmap_valid_nodes=total_nodes,
        layer_classified_nodes=total_nodes,
        rendered_nodes=total_nodes,
        excluded_non_layer_nodes=0,
        nonzero_voxels=2,
        traces_represented=1,
        y_bins=4,
        x_bins=4,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        layer_labels=("L1", "L2/3", "L4", "L5", "L6a", "L6b"),
        layer_node_counts=(1, 1, 0, 0, 0, 0),
        atlas_name="allen_mouse_25um",
        atlas_version="1.2.3",
    )


def test_allen_layer_stack_uses_one_2d_categorical_image(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_ALLEN_LAYERS
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._HEATMAP_COLOR_SINGLE
    )
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc"],
            "render_valid": [True, True],
            "allen_layer_index": [0, 1],
            "allen_layer_label": ["L1", "L2/3"],
            "y_flat_bin": [1, 2],
            "x_flat_bin": [2, 3],
        }
    )
    volume = np.zeros((6, 4, 4), dtype=np.float32)
    volume[0, 1, 2] = 1.0
    volume[1, 2, 3] = 1.0
    stack = module.AllenLayerStackResult(
        projected_nodes=projected,
        volume=volume,
        summary=_simple_allen_layer_summary(module),
    )

    layer = widget._create_or_update_allen_layer_stack(
        stack,
        _simple_projection_summary(module, total_nodes=2),
        flatmap_style="both_shaped",
        coordinate_mode="parquet_columns",
    )

    assert layer.name == module._ALLEN_LAYER_HEATMAP_LAYER_NAME
    assert widget._viewer.layers == [layer]
    assert widget._viewer.dims.ndisplay == 2
    assert layer.axis_labels == ("Allen layer", "Flatmap Y", "Flatmap X")
    assert layer.metadata["flatmap_plane_mode"] == "allen_layers"
    assert layer.metadata["allen_layer_labels"] == [
        "L1",
        "L2/3",
        "L4",
        "L5",
        "L6a",
        "L6b",
    ]
    assert layer.metadata["allen_atlas_name"] == "allen_mouse_25um"
    assert layer.metadata["allen_atlas_identity"] == {
        "name": "allen_mouse_25um",
        "version": "1.2.3",
    }
    assert layer.metadata["flatmap_projection_source"] == "legacy_auto"


def test_allen_layer_selection_helpers_preserve_canonical_order(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = _DummyDataCombo(module._RENDER_ALLEN_LAYERS)
    widget._allen_layer_checkboxes = [
        _DummyValueControl(checked=value)
        for value in (False, True, False, False, True, False)
    ]
    widget._allen_layer_output_combo = _DummyDataCombo(
        module.ALLEN_LAYER_OUTPUT_PROJECTION
    )

    assert widget._current_allen_layer_indices() == (1, 4)
    assert widget._is_allen_layer_projection() is True
    assert widget._current_plane_mode() == "allen_layer_projection"
    assert widget._axis_labels_for_render_mode(module._RENDER_ALLEN_LAYERS) == (
        "Flatmap Y",
        "Flatmap X",
    )


def test_allen_layer_options_default_to_all_layers_and_stack(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    assert widget._current_allen_layer_indices() == (0, 1, 2, 3, 4, 5)
    assert widget._current_allen_layer_output_mode() == "stack"


def test_allen_layer_options_are_visible_only_in_allen_mode(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    options_group = _DummyProgressBar()
    widget._allen_layer_options_group = options_group
    widget._render_mode_combo = _DummyDataCombo(module._RENDER_ALLEN_LAYERS)

    widget._update_render_mode_controls()
    assert options_group.visible is True

    widget._render_mode_combo.data = module._RENDER_HEATMAP
    widget._update_render_mode_controls()
    assert options_group.visible is False


def test_allen_layer_option_change_invalidates_existing_render(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = _DummyDataCombo(module._RENDER_ALLEN_LAYERS)
    widget._allen_layer_checkboxes = [
        _DummyValueControl(checked=index == 3) for index in range(6)
    ]
    invalidated = []
    widget._invalidate_flatmap_grid_layers = lambda: invalidated.append(True)
    widget._update_render_mode_controls = lambda: None
    widget._update_cached_region_controls = lambda: None

    widget._on_allen_layer_options_changed()

    assert invalidated == [True]
    assert widget._status_label.text == (
        "Allen layers L5 will render as a compact 2D stack."
    )


def test_empty_allen_layer_selection_disables_projection_and_labels(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = _DummyDataCombo(module._RENDER_ALLEN_LAYERS)
    widget._projection_source_combo = _DummyDataCombo(
        module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._allen_layer_checkboxes = [
        _DummyValueControl(checked=False)
        for _label in module.ALLEN_ISOCORTEX_LAYER_LABELS
    ]
    widget._active_cache_profile = object()
    widget._add_soma_btn = _DummyButton()

    widget._set_projection_controls_enabled(True)

    assert widget._project_btn.enabled is False
    assert widget._add_soma_btn.enabled is False
    assert widget._cached_region_control_states()["_region_labels_btn"] is False


def test_allen_layer_projection_uses_rank_two_layer_and_metadata(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = _DummyDataCombo(module._RENDER_ALLEN_LAYERS)
    widget._heatmap_color_mode_combo = _DummyDataCombo(
        module._HEATMAP_COLOR_SINGLE
    )
    summary = module.AllenLayerStackSummary(
        total_nodes=3,
        flatmap_valid_nodes=3,
        layer_classified_nodes=3,
        rendered_nodes=2,
        excluded_non_layer_nodes=0,
        excluded_unselected_layer_nodes=1,
        nonzero_voxels=1,
        traces_represented=2,
        y_bins=4,
        x_bins=4,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        layer_labels=("L2/3", "L6a"),
        layer_node_counts=(1, 1),
        selected_layer_indices=(1, 4),
        output_mode=module.ALLEN_LAYER_OUTPUT_PROJECTION,
        output_shape=(4, 4),
        atlas_name="allen_mouse_25um",
    )
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "b.swc"],
            "render_valid": [True, True],
            "allen_layer_index": [1, 4],
            "allen_layer_render_index": [0, 1],
            "y_flat_bin": [1, 1],
            "x_flat_bin": [2, 2],
        }
    )
    result = module.AllenLayerStackResult(
        projected_nodes=projected,
        volume=np.asarray(
            [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.float32,
        ),
        summary=summary,
    )

    layer = widget._create_or_update_allen_layer_stack(
        result,
        _simple_projection_summary(module, total_nodes=3),
        flatmap_style="both_shaped",
        coordinate_mode="parquet_columns",
    )

    assert layer.name == module._ALLEN_LAYER_PROJECTION_LAYER_NAME
    assert layer.data.shape == (4, 4)
    assert layer.axis_labels == ("Flatmap Y", "Flatmap X")
    assert layer.metadata["flatmap_plane_mode"] == "allen_layer_projection"
    assert layer.metadata["allen_layer_output_mode"] == "projection"
    assert layer.metadata["allen_layer_indices"] == [1, 4]
    assert layer.metadata["allen_layer_labels"] == ["L2/3", "L6a"]
    assert widget._viewer.canvas.overlays.text.visible is False


def test_selected_allen_stack_caption_uses_compact_plane_number(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = _DummyDataCombo(module._RENDER_ALLEN_LAYERS)
    summary = module.AllenLayerStackSummary(
        total_nodes=2,
        flatmap_valid_nodes=2,
        layer_classified_nodes=2,
        rendered_nodes=2,
        excluded_non_layer_nodes=0,
        nonzero_voxels=2,
        traces_represented=1,
        y_bins=4,
        x_bins=4,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        layer_labels=("L2/3", "L5"),
        layer_node_counts=(1, 1),
        selected_layer_indices=(1, 3),
        output_mode=module.ALLEN_LAYER_OUTPUT_STACK,
        output_shape=(2, 4, 4),
        atlas_name="allen_mouse_25um",
    )
    result = module.AllenLayerStackResult(
        projected_nodes=pd.DataFrame(
            {
                "file_id": ["a.swc", "a.swc"],
                "render_valid": [True, True],
                "allen_layer_render_index": [0, 1],
                "y_flat_bin": [1, 2],
                "x_flat_bin": [1, 2],
            }
        ),
        volume=np.ones((2, 4, 4), dtype=np.float32),
        summary=summary,
    )

    widget._create_or_update_allen_layer_stack(
        result,
        _simple_projection_summary(module, total_nodes=2),
        flatmap_style="both_shaped",
        coordinate_mode="parquet_columns",
    )
    widget._viewer.dims.set_current_step(1)

    assert widget._viewer.text_overlay.text == "Allen layer: L5  (plane 2 of 2)"


def test_allen_layer_projection_places_selected_somas_in_xy_only(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = _DummyDataCombo(module._RENDER_ALLEN_LAYERS)
    widget._allen_layer_output_combo = _DummyDataCombo(
        module.ALLEN_LAYER_OUTPUT_PROJECTION
    )
    widget._soma_layer = None
    summary = module.AllenLayerStackSummary(
        total_nodes=2,
        flatmap_valid_nodes=2,
        layer_classified_nodes=2,
        rendered_nodes=1,
        excluded_non_layer_nodes=0,
        excluded_unselected_layer_nodes=1,
        nonzero_voxels=1,
        traces_represented=1,
        y_bins=4,
        x_bins=4,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        layer_labels=("L5",),
        layer_node_counts=(1,),
        selected_layer_indices=(3,),
        output_mode=module.ALLEN_LAYER_OUTPUT_PROJECTION,
        output_shape=(4, 4),
        atlas_name="allen_mouse_25um",
    )
    render = module.AllenLayerStackResult(
        projected_nodes=pd.DataFrame(
            {
                "file_id": ["a.swc", "b.swc"],
                "render_valid": [True, False],
                "y_flat_bin": [2, -1],
                "x_flat_bin": [3, -1],
                "allen_layer_index": [3, 4],
                "allen_layer_render_index": [0, -1],
            }
        ),
        volume=np.zeros((4, 4), dtype=np.float32),
        summary=summary,
    )

    layer = widget._create_or_update_soma_layer(
        render,
        _simple_projection_summary(module, total_nodes=2),
    )

    np.testing.assert_array_equal(layer.data, np.asarray([[2.0, 3.0]]))
    assert layer.axis_labels == ("Flatmap Y", "Flatmap X")
    assert layer.metadata["flatmap_plane_mode"] == "allen_layer_projection"
    assert layer.metadata["allen_layer_indices"] == [3]


def _render_allen_layer_stack(module, widget, *, color_mode=None):
    """Render a six-plane Allen stack through the widget's normal entry point."""
    color_mode = color_mode or module._HEATMAP_COLOR_SINGLE
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_ALLEN_LAYERS
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: color_mode
    )
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "b.swc"],
            "render_valid": [True, True],
            "allen_layer_index": [0, 1],
            "allen_layer_label": ["L1", "L2/3"],
            "y_flat_bin": [1, 2],
            "x_flat_bin": [2, 3],
        }
    )
    volume = np.zeros((6, 4, 4), dtype=np.float32)
    volume[0, 1, 2] = 1.0
    volume[1, 2, 3] = 1.0
    stack = module.AllenLayerStackResult(
        projected_nodes=projected,
        volume=volume,
        summary=_simple_allen_layer_summary(module),
    )
    return widget._create_or_update_allen_layer_stack(
        stack,
        _simple_projection_summary(module, total_nodes=2),
        flatmap_style="both_shaped",
        coordinate_mode="parquet_columns",
    )


def _flat_render_summary(
    module,
    total_nodes: int = 2,
    *,
    y_bins: int = 4,
    x_bins: int | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
):
    """A depth-collapsed summary: real depth counts, no depth axis."""
    return module.FlatmapRenderSummary(
        total_nodes=total_nodes,
        flatmap_valid_nodes=total_nodes,
        depth_valid_nodes=total_nodes,
        depth_minus_one_nodes=0,
        rendered_nodes=total_nodes,
        excluded_depth_minus_one_nodes=0,
        nonzero_voxels=total_nodes,
        traces_represented=1,
        y_bins=y_bins,
        x_bins=y_bins if x_bins is None else x_bins,
        depth_bins=0,
        depth_bin_um=0.0,
        x_flat_min=bounds[0],
        x_flat_max=bounds[1],
        y_flat_min=bounds[0],
        y_flat_max=bounds[1],
        depth_min_um=0.0,
        depth_max_um=25.0,
        includes_depth_minus_one_plane=True,
    )


def _render_flat_heatmap(module, widget, *, color_mode=None):
    """Render a depth-collapsed 2D heatmap through the normal entry point."""
    color_mode = color_mode or module._HEATMAP_COLOR_SINGLE
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_HEATMAP
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: color_mode
    )
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "b.swc"],
            "render_valid": [True, True],
            "y_flat_bin": [1, 2],
            "x_flat_bin": [2, 3],
        }
    )
    volume = np.zeros((4, 4), dtype=np.float32)
    volume[1, 2] = 1.0
    volume[2, 3] = 1.0
    render_result = module.FlatmapRenderResult(
        projected_nodes=projected,
        volume=volume,
        points=np.asarray([[1.0, 2.0], [2.0, 3.0]]),
        point_file_ids=["a.swc", "b.swc"],
        summary=_flat_render_summary(module),
    )
    return widget._create_or_update_render_layer(
        render_result,
        _simple_projection_summary(module, total_nodes=2),
        flatmap_style="both_shaped",
        coordinate_mode="parquet_columns",
        render_mode=module._RENDER_FLAT_HEATMAP,
    )


def _flat_vector_render_result(module, *, rendered_nodes: int = 3):
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "b.swc"],
            "node_id": [1, 2, 1],
            "parent_id": [-1, 1, -1],
            "x_flat": [0.0, 2.0, 4.0],
            "y_flat": [0.0, 4.0, 2.0],
            "render_valid": [True, True, True],
        }
    )
    return module.FlatmapRenderResult(
        projected_nodes=projected,
        volume=np.zeros((4, 4), dtype=np.float32),
        points=np.asarray([[0.0, 0.0], [2.0, 1.0], [1.0, 2.0]]),
        point_file_ids=["a.swc", "a.swc", "b.swc"],
        summary=_flat_render_summary(module, rendered_nodes, bounds=(0.0, 4.0)),
    )


def _render_flat_vector(module, widget, *, rendered_nodes: int = 3):
    """Render 2D vectors through the normal entry point."""
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_VECTOR
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._HEATMAP_COLOR_SINGLE
    )
    return widget._create_or_update_render_layer(
        _flat_vector_render_result(module, rendered_nodes=rendered_nodes),
        _simple_projection_summary(module, total_nodes=3),
        flatmap_style="both_shaped",
        coordinate_mode="parquet_columns",
        render_mode=module._RENDER_FLAT_VECTOR,
    )


def test_flat_heatmap_render_uses_one_two_dimensional_image(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    layer = _render_flat_heatmap(module, widget)

    assert layer.name == module._FLAT_HEATMAP_LAYER_NAME
    assert layer.data.ndim == 2
    assert layer.axis_labels == ("Flatmap Y", "Flatmap X")
    assert layer.metadata["flatmap_render_mode"] == module._RENDER_FLAT_HEATMAP
    assert layer.metadata["flatmap_plane_mode"] == "flat"
    assert widget._viewer.dims.ndisplay == 2
    assert widget._viewer.dims.axis_labels == ("Flatmap Y", "Flatmap X")
    assert widget._viewer.axes.visible is True
    # No plane axis means no plane caption to write.
    assert widget._viewer.text_overlay.visible is False


def test_flat_heatmap_render_retires_a_stale_plane_caption(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    _render_allen_layer_stack(module, widget)
    assert widget._viewer.text_overlay.text.startswith("Allen layer")

    _render_flat_heatmap(module, widget)

    # A stale "plane 1 of 6" caption would describe a stack that is no longer
    # on screen.
    assert widget._viewer.text_overlay.visible is False
    assert widget._viewer.text_overlay.text == ""


def test_flat_heatmap_grouped_colors_add_one_layer_per_neuron(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    _render_flat_heatmap(module, widget, color_mode=module._HEATMAP_COLOR_INDIVIDUAL)

    grouped = [
        layer
        for layer in widget._viewer.layers
        if layer.name.startswith(module._GROUPED_FLAT_HEATMAP_PREFIX)
    ]
    assert [layer.name for layer in grouped] == [
        f"{module._GROUPED_FLAT_HEATMAP_PREFIX}a.swc",
        f"{module._GROUPED_FLAT_HEATMAP_PREFIX}b.swc",
    ]
    assert all(layer.data.shape == (4, 4) for layer in grouped)
    assert all(layer.axis_labels == ("Flatmap Y", "Flatmap X") for layer in grouped)
    assert widget.__class__._is_flatmap_render_layer_name(grouped[0].name)


def test_flat_vector_render_sets_edge_color_after_creation(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    layer = _render_flat_vector(module, widget)

    assert layer.name == module._FLAT_VECTOR_LAYER_NAME
    # napari only enters DIRECT per-vector color mode when edge_color is
    # assigned after construction.
    assert "edge_color" not in layer.init_kwargs
    assert layer.init_kwargs["vector_style"] == "line"
    assert layer.axis_labels == ("Flatmap Y", "Flatmap X")
    # One color per drawn segment; the fixture's only edge belongs to a.swc.
    np.testing.assert_allclose(layer.edge_color, [[1.0, 0.0, 0.0, 1.0]])


def test_flat_vector_render_uses_row_col_start_direction_data(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    layer = _render_flat_vector(module, widget)

    # One edge per valid parent-child pair; b.swc is a lone root.
    assert layer.data.shape == (1, 2, 2)
    expected_start = np.column_stack(
        (
            flatmap_pixel_coordinates(np.asarray([0.0]), (0.0, 4.0), 4),
            flatmap_pixel_coordinates(np.asarray([0.0]), (0.0, 4.0), 4),
        )
    )
    expected_end = np.column_stack(
        (
            flatmap_pixel_coordinates(np.asarray([4.0]), (0.0, 4.0), 4),
            flatmap_pixel_coordinates(np.asarray([2.0]), (0.0, 4.0), 4),
        )
    )
    np.testing.assert_allclose(layer.data[:, 0], expected_start, atol=1e-5)
    np.testing.assert_allclose(
        layer.data[:, 0] + layer.data[:, 1],
        expected_end,
        atol=1e-5,
    )
    assert layer.metadata["flatmap_vector_segments"] == 1
    assert widget._viewer.dims.ndisplay == 2


def test_flat_vector_render_updates_the_existing_layer(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    first = _render_flat_vector(module, widget)
    second = _render_flat_vector(module, widget)

    assert second is first
    vector_layers = [
        layer
        for layer in widget._viewer.layers
        if layer.name == module._FLAT_VECTOR_LAYER_NAME
    ]
    assert len(vector_layers) == 1
    assert first.refresh_count >= 1


def test_flat_vector_render_refuses_too_many_segments(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    monkeypatch.setattr(module, "MAX_FLATMAP_VECTOR_SEGMENTS", 2)

    with pytest.raises(RuntimeError, match="above the 2 limit"):
        _render_flat_vector(module, widget, rendered_nodes=3)

    assert widget._viewer.layers == []


def test_flat_vector_render_reports_a_selection_without_edges(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_VECTOR
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._HEATMAP_COLOR_SINGLE
    )
    render_result = _flat_vector_render_result(module)
    # Every node is its own root, so no parent-child pair shares an edge.
    render_result.projected_nodes.loc[:, "parent_id"] = -1

    with pytest.raises(RuntimeError, match="parent-child edges"):
        widget._create_or_update_render_layer(
            render_result,
            _simple_projection_summary(module, total_nodes=3),
            flatmap_style="both_shaped",
            coordinate_mode="parquet_columns",
            render_mode=module._RENDER_FLAT_VECTOR,
        )

    assert widget._viewer.layers == []


def test_failed_vector_rerender_keeps_the_existing_flatmap_layer(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    existing = _DummyLayer(
        np.ones((2, 4, 4), dtype=np.float32),
        name=module._HEATMAP_LAYER_NAME,
        metadata={
            "napari_neuron_navigator_space": "flatmap",
            "flatmap_render_mode": module._RENDER_HEATMAP,
        },
    )
    widget._viewer.layers.append(existing)
    widget._projection_layer = existing
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_VECTOR
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._HEATMAP_COLOR_SINGLE
    )
    render_result = _flat_vector_render_result(module)
    render_result.projected_nodes.loc[:, "parent_id"] = -1

    with pytest.raises(RuntimeError, match="parent-child edges"):
        widget._create_or_update_render_layer(
            render_result,
            _simple_projection_summary(module, total_nodes=3),
            flatmap_style="both_shaped",
            coordinate_mode="parquet_columns",
            render_mode=module._RENDER_FLAT_VECTOR,
        )

    assert widget._viewer.layers == [existing]
    assert widget._projection_layer is existing


def test_flat_vector_projection_failure_reaches_the_status_label(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    _configure_projection_widget(widget, module, _augmented_nodes())
    widget._summary_label = _DummyLabel()
    widget._export_btn = _DummyButton()
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_VECTOR
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._HEATMAP_COLOR_SINGLE
    )
    monkeypatch.setattr(module, "MAX_FLATMAP_VECTOR_SEGMENTS", 1)

    def fake_lookup(self, nodes, **_kwargs):
        return (
            types.SimpleNamespace(
                projected_nodes=nodes,
                summary=_simple_projection_summary(module, total_nodes=3),
            ),
            _flat_vector_render_result(module),
            None,
        )

    widget._project_from_lookup_files = types.MethodType(fake_lookup, widget)

    widget._project()

    assert "above the 1 limit" in widget._status_label.text
    assert widget._viewer.layers == []


def test_flat_modes_disable_depth_bin_but_keep_the_depth_minus_one_checkbox(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._y_bins_spin = _DummyButton()
    widget._depth_bin_spin = _DummyButton()
    widget._exclude_depth_minus_one_cb = _DummyButton()
    widget._negative_one_sentinel_cb = _DummyButton()
    widget._zero_sentinel_cb = _DummyButton()
    widget._heatmap_color_mode_combo = _DummyButton()
    widget._cache_grid_locked = False

    for render_mode, color_combo_enabled in (
        (module._RENDER_FLAT_HEATMAP, True),
        (module._RENDER_FLAT_VECTOR, False),
    ):
        widget._render_mode_combo = types.SimpleNamespace(
            currentData=lambda mode=render_mode: mode
        )
        widget._update_render_mode_controls()

        assert widget._y_bins_spin.enabled is True
        # No depth bins to size, but the checkbox still selects nodes.
        assert widget._depth_bin_spin.enabled is False
        assert widget._exclude_depth_minus_one_cb.enabled is True
        assert widget._heatmap_color_mode_combo.enabled is color_combo_enabled


def test_flat_modes_offer_collapsed_labels_and_outlines_but_not_surfaces(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._active_cache_profile = object()
    widget._region_surfaces_btn = _DummyButton()
    widget._region_outlines_btn = _DummyButton()
    widget._clear_region_geometry_btn = _DummyButton()

    for render_mode in (module._RENDER_FLAT_HEATMAP, module._RENDER_FLAT_VECTOR):
        widget._render_mode_combo = types.SimpleNamespace(
            currentData=lambda mode=render_mode: mode
        )
        widget._update_cached_region_controls()

        # Labels and outlines collapse into one plane; a cached surface is a 3D
        # voxel shell with no 2D form.
        assert widget._region_labels_btn.enabled is True
        assert widget._region_outlines_btn.enabled is True
        assert widget._clear_region_geometry_btn.enabled is True
        assert widget._region_surfaces_btn.enabled is False
        # Only the NRRD recompute path picks its own label atlas.
        assert widget._region_label_atlas_combo.enabled is False

    # 3D Points still uses the depth grid, so its geometry stays available.
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_POINTS
    )
    widget._update_cached_region_controls()

    assert widget._region_surfaces_btn.enabled is True
    assert widget._region_outlines_btn.enabled is True


def test_recomputed_region_labels_are_refused_in_flat_modes(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_HEATMAP
    )
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_RECOMPUTE
    )

    with pytest.raises(RuntimeError, match="Recomputed region labels"):
        widget._create_region_labels_from_current_state()


def test_flat_mode_labels_route_to_the_cache_instead_of_refusing(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_HEATMAP
    )
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    sentinel = object()
    widget._create_cached_region_labels = lambda: sentinel

    assert widget._create_region_labels_from_current_state() is sentinel


def test_cached_region_control_gating_matches_between_both_enablers(
    monkeypatch,
) -> None:
    """Two call sites apply the same matrix; they must not drift apart."""
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._active_cache_profile = object()
    names = (
        "_region_label_atlas_combo",
        "_region_labels_btn",
        "_region_surfaces_btn",
        "_region_outlines_btn",
        "_clear_region_geometry_btn",
    )
    render_modes = (
        module._RENDER_HEATMAP,
        module._RENDER_POINTS,
        module._RENDER_FLAT_HEATMAP,
        module._RENDER_FLAT_VECTOR,
        module._RENDER_ALLEN_LAYERS,
    )
    sources = (
        module._PROJECTION_SOURCE_PRECOMPUTED,
        module._PROJECTION_SOURCE_RECOMPUTE,
    )

    for render_mode in render_modes:
        for source in sources:
            widget._render_mode_combo = types.SimpleNamespace(
                currentData=lambda mode=render_mode: mode
            )
            widget._projection_source_combo = types.SimpleNamespace(
                currentData=lambda value=source: value
            )
            for name in names:
                setattr(widget, name, _DummyButton())
            widget._update_cached_region_controls()
            from_update = {name: getattr(widget, name).enabled for name in names}

            for name in names:
                setattr(widget, name, _DummyButton())
            widget._set_region_label_controls_enabled(True)
            from_setter = {name: getattr(widget, name).enabled for name in names}

            assert from_update == from_setter, (render_mode, source)


def test_flat_render_summary_reports_the_collapsed_depth_axis(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)

    text = module.FlatmapProjectionWidget._format_flat_render_summary(
        _simple_projection_summary(module, total_nodes=2),
        _flat_render_summary(module),
    )

    assert "collapsed into one flatmap plane" in text
    # The depth counts stay honest because the same depth rules applied.
    assert "Depth-valid nodes: 2" in text


def test_precomputed_flat_heatmap_uses_the_duckdb_worker(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_HEATMAP
    )
    started = []
    widget._start_precomputed_heatmap_worker = lambda: started.append(True)
    widget._project_from_lookup_files = types.MethodType(
        lambda self, *_args, **_kwargs: pytest.fail(
            "2D Heatmap must take the DuckDB fast path"
        ),
        widget,
    )

    widget._project()

    assert started == [True]
    assert widget._current_plane_mode() == "flat"


def test_matching_cache_profile_preserves_a_live_flat_heatmap(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _configure_cache_activation_widget(widget, module)
    layer.name = module._FLAT_HEATMAP_LAYER_NAME
    layer.data = np.ones((4, 4), dtype=np.float32)
    layer.metadata["flatmap_render_mode"] = module._RENDER_FLAT_HEATMAP
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_HEATMAP
    )
    widget._last_render_mode = module._RENDER_FLAT_HEATMAP
    widget._last_render_summary = _flat_render_summary(module)
    widget._last_volume_shape = (4, 4)
    widget._queue_gui_callback = lambda callback: pytest.fail(
        "A matching XY grid must not retire the 2D heatmap"
    )

    widget._activate_cache_profile(
        _CacheProfile("matching-flat-profile"), force_transition=True
    )

    assert widget._viewer.layers == [layer]
    assert widget._last_cache_profile_id == "matching-flat-profile"


def test_flat_vector_render_is_not_preserved_across_cache_profiles(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    _configure_cache_activation_widget(widget, module)
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_VECTOR
    )
    widget._last_render_mode = module._RENDER_FLAT_VECTOR
    widget._last_render_summary = _flat_render_summary(module)
    widget._last_volume_shape = (4, 4)

    # A per-node vector render is not a volume on the cache grid.
    assert widget._render_matches_cache_profile(_CacheProfile("any-profile")) is False


def test_focus_projection_view_bounds_a_two_dimensional_image(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    volume = np.zeros((8, 8), dtype=np.float32)
    volume[3, 5] = 1.0
    layer = _DummyLayer(volume, name="flat")
    widget._viewer.layers.append(layer)

    widget._focus_projection_view(layer, volume, ndisplay=2)

    assert widget._viewer.dims.ndisplay == 2
    assert widget._viewer.camera.center == (3.0, 5.0)


def test_focus_projection_view_bounds_vector_starts_and_endpoints(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    # Start (2, 2) with direction (4, 6) reaches (6, 8).
    vectors = np.asarray([[[2.0, 2.0], [4.0, 6.0]]], dtype=np.float32)
    layer = _DummyLayer(vectors, name="vectors")
    widget._viewer.layers.append(layer)

    widget._focus_projection_view(layer, vectors, ndisplay=2, data_kind="vectors")

    assert widget._viewer.camera.center == (4.0, 5.0)


def test_focus_projection_view_bounds_two_dimensional_points(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    points = np.asarray([[1.0, 2.0], [5.0, 8.0]])
    layer = _DummyLayer(points, name="points")
    widget._viewer.layers.append(layer)

    widget._focus_projection_view(layer, points, ndisplay=2, data_kind="points")

    assert widget._viewer.camera.center == (3.0, 5.0)


def test_allen_layer_stack_names_the_viewer_axes_and_current_plane(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    _render_allen_layer_stack(module, widget)

    viewer = widget._viewer
    # napari draws the slider caption and the axes overlay from viewer.dims,
    # never from layer.axis_labels.
    assert viewer.dims.axis_labels == ("Allen layer", "Flatmap Y", "Flatmap X")
    assert viewer.axes.visible is True
    assert viewer.axes.labels is True
    assert viewer.text_overlay.visible is True
    assert viewer.text_overlay.text == "Allen layer: L1  (plane 1 of 6)"
    assert viewer.text_overlay.position == "top_left"
    assert viewer.text_overlay.font_size == 12


def test_allen_layer_plane_label_follows_the_dims_slider(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    _render_allen_layer_stack(module, widget)
    widget._viewer.dims.set_current_step(3)

    assert widget._viewer.text_overlay.text == "Allen layer: L5  (plane 4 of 6)"

    widget._viewer.dims.set_current_step(5)

    assert widget._viewer.text_overlay.text == "Allen layer: L6b  (plane 6 of 6)"


def test_grouped_allen_layer_stack_names_the_current_plane(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    layer = _render_allen_layer_stack(
        module,
        widget,
        color_mode=module._HEATMAP_COLOR_INDIVIDUAL,
    )

    assert layer.name.startswith(module._GROUPED_ALLEN_LAYER_PREFIX)
    assert layer.axis_labels == ("Allen layer", "Flatmap Y", "Flatmap X")
    assert widget._viewer.dims.axis_labels == (
        "Allen layer",
        "Flatmap Y",
        "Flatmap X",
    )
    assert widget._viewer.text_overlay.text == "Allen layer: L1  (plane 1 of 6)"


def test_depth_heatmap_plane_label_reports_the_depth_range(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    projected = pd.DataFrame({"file_id": ["a.swc"]})
    volume = np.zeros((3, 4, 4), dtype=np.float32)
    volume[1, 1, 2] = 1.0
    render_summary = module.FlatmapRenderSummary(
        total_nodes=1,
        flatmap_valid_nodes=1,
        depth_valid_nodes=1,
        depth_minus_one_nodes=0,
        rendered_nodes=1,
        excluded_depth_minus_one_nodes=0,
        nonzero_voxels=1,
        traces_represented=1,
        y_bins=4,
        x_bins=4,
        depth_bins=3,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=75.0,
        includes_depth_minus_one_plane=False,
    )
    render_result = module.FlatmapRenderResult(
        projected_nodes=projected,
        volume=volume,
        points=np.zeros((1, 3), dtype=float),
        point_file_ids=["a.swc"],
        summary=render_summary,
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._HEATMAP_COLOR_SINGLE
    )

    layer = widget._create_or_update_render_layer(
        render_result,
        _simple_projection_summary(module),
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode=module._RENDER_HEATMAP,
    )

    assert layer.axis_labels == ("Depth bin", "Flatmap Y", "Flatmap X")
    assert widget._viewer.dims.axis_labels == ("Depth bin", "Flatmap Y", "Flatmap X")
    assert widget._viewer.text_overlay.text == "Depth bin: 0-25 um  (plane 1 of 3)"

    widget._viewer.dims.set_current_step(2)

    assert widget._viewer.text_overlay.text == "Depth bin: 50-75 um  (plane 3 of 3)"


def test_plane_label_reports_position_when_planes_are_unnamed(monkeypatch) -> None:
    """A depth-mode region-labels layer records no bin size; do not invent one."""
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    labels = np.zeros((5, 4, 4), dtype=np.uint16)
    layer = widget._viewer.add_labels(
        labels,
        name=module._REGION_LABELS_LAYER_NAME,
        metadata={"projection_kind": "flatmap_region_labels"},
        axis_labels=widget._depth_axis_labels(),
    )

    widget._apply_display_axis_annotations(layer)

    assert widget._viewer.text_overlay.text == "Depth bin: plane 1 of 5"

    widget._viewer.dims.set_current_step(4)

    assert widget._viewer.text_overlay.text == "Depth bin: plane 5 of 5"


def test_points_render_leaves_the_viewer_overlays_untouched(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    points = widget._viewer.add_points(
        np.zeros((2, 3), dtype=float),
        name=module._POINTS_LAYER_NAME,
        metadata={},
    )

    widget._apply_display_axis_annotations(points)

    assert widget._viewer.dims.axis_labels == ("0", "1", "2")
    assert widget._viewer.axes.visible is False
    assert widget._viewer.text_overlay.text == ""


def test_removing_render_layers_restores_the_viewer_overlays(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    viewer = widget._viewer

    _render_allen_layer_stack(module, widget)
    assert viewer.text_overlay.visible is True

    widget._remove_projection_layer()

    # The widget no longer writes dims labels directly. A real napari 0.9
    # ViewerModel derives its negative defaults when the last layer is removed;
    # this minimal list double deliberately retains the most recent labels.
    assert viewer.dims.axis_labels == (
        "Allen layer",
        "Flatmap Y",
        "Flatmap X",
    )
    assert viewer.axes.visible is False
    assert viewer.text_overlay.visible is False
    assert viewer.text_overlay.text == ""
    assert viewer.dims.events.current_step.callbacks == []


def test_release_display_viewer_stops_following_the_slider(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._last_display_viewer = widget._viewer
    widget._region_surfaces_layers = []
    widget._region_outlines_layers = []
    viewer = widget._viewer

    _render_allen_layer_stack(module, widget)
    assert viewer.dims.events.current_step.callbacks != []

    assert widget._release_display_viewer(viewer) is True

    assert viewer.dims.events.current_step.callbacks == []
    assert widget._display_axis_annotation_state is None
    assert viewer.text_overlay.visible is False


def test_plane_captions_keep_following_each_side_by_side_viewer(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    first = _DummyViewer()
    second = _DummyViewer()
    active = {"viewer": first}
    widget._display_viewer_provider = lambda create=True: active["viewer"]

    first_layer = first.add_image(
        np.zeros((2, 3, 3)),
        name="first",
        axis_labels=("Depth bin", "Flatmap Y", "Flatmap X"),
    )
    widget._apply_display_axis_annotations(first_layer)

    active["viewer"] = second
    second_layer = second.add_image(
        np.zeros((3, 3, 3)),
        name="second",
        axis_labels=("Depth bin", "Flatmap Y", "Flatmap X"),
    )
    widget._apply_display_axis_annotations(second_layer)

    first.dims.set_current_step(1)
    second.dims.set_current_step(2)

    assert first.text_overlay.text == "Depth bin: plane 2 of 2"
    assert second.text_overlay.text == "Depth bin: plane 3 of 3"
    assert len(widget._display_axis_annotation_states) == 2


def test_allen_layer_mode_enables_cached_labels_but_disables_depth_and_geometry(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_ALLEN_LAYERS
    )
    widget._cache_grid_locked = False
    widget._y_bins_spin = _DummyValueControl()
    widget._depth_bin_spin = _DummyValueControl()
    widget._exclude_depth_minus_one_cb = _DummyValueControl()
    widget._negative_one_sentinel_cb = _DummyValueControl()
    widget._zero_sentinel_cb = _DummyValueControl()
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._active_cache_profile = object()
    widget._region_surfaces_btn = _DummyButton()
    widget._region_outlines_btn = _DummyButton()
    widget._clear_region_geometry_btn = _DummyButton()
    widget._region_labels_btn = _DummyButton()
    widget._region_label_atlas_combo = _DummyCombo()

    widget._update_render_mode_controls()
    widget._update_cached_region_controls()

    assert widget._y_bins_spin.enabled is True
    assert widget._depth_bin_spin.enabled is False
    assert widget._exclude_depth_minus_one_cb.enabled is False
    assert widget._negative_one_sentinel_cb.enabled is True
    assert widget._region_labels_btn.enabled is True
    assert widget._region_surfaces_btn.enabled is False
    assert widget._region_outlines_btn.enabled is False


def test_allen_layer_mode_requires_active_cache_for_region_labels(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_ALLEN_LAYERS
    )
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._active_cache_profile = None
    widget._region_surfaces_btn = _DummyButton()
    widget._region_outlines_btn = _DummyButton()
    widget._clear_region_geometry_btn = _DummyButton()
    widget._region_labels_btn = _DummyButton()
    widget._region_label_atlas_combo = _DummyCombo()

    widget._update_cached_region_controls()

    assert widget._region_labels_btn.enabled is False
    assert widget._region_surfaces_btn.enabled is False
    assert widget._region_outlines_btn.enabled is False


def test_allen_layer_map_requires_loaded_atlas(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._atlas_provider = lambda: None

    with pytest.raises(RuntimeError, match="Load an Allen mouse atlas"):
        widget._current_allen_layer_map()


def test_allen_layer_nrrd_projection_requires_region_id(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_ALLEN_LAYERS
    )

    with pytest.raises(RuntimeError, match="requires a region_id column"):
        widget._project_from_lookup_files(
            pd.DataFrame({"file_id": ["a.swc"]}),
        )


def _binned_render_result(module):
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "b.swc", "c.swc"],
            "render_valid": [True, True, True, True],
            "depth_bin": [0, 0, 1, 1],
            "y_flat_bin": [1, 1, 2, 0],
            "x_flat_bin": [2, 2, 0, 1],
        }
    )
    volume = np.zeros((2, 3, 3), dtype=np.float32)
    volume[0, 1, 2] = 2.0
    volume[1, 2, 0] = 1.0
    volume[1, 0, 1] = 1.0
    return module.FlatmapRenderResult(
        projected_nodes=projected,
        volume=volume,
        points=np.asarray(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        ),
        point_file_ids=["a.swc", "a.swc", "b.swc", "c.swc"],
        summary=_simple_render_summary(module, total_nodes=4),
    )


def test_individual_heatmap_color_mode_creates_one_layer_per_file_id(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: "individual"
    )
    summary = _simple_projection_summary(module, total_nodes=4)

    layer = widget._create_or_update_render_layer(
        _binned_render_result(module),
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert layer.name == "Isocortex Flatmap Heatmap: a.swc"
    assert [layer.name for layer in widget._viewer.layers] == [
        "Isocortex Flatmap Heatmap: a.swc",
        "Isocortex Flatmap Heatmap: b.swc",
        "Isocortex Flatmap Heatmap: c.swc",
    ]
    first, second, third = widget._viewer.layers
    assert first.data[0, 1, 2] == 2.0
    assert second.data[1, 2, 0] == 1.0
    assert third.data[1, 0, 1] == 1.0
    assert first.metadata["flatmap_heatmap_color_mode"] == "individual"
    assert first.metadata["flatmap_heatmap_group_label"] == "a.swc"
    assert first.metadata["source_file_ids"] == ["a.swc"]
    assert first.metadata["flatmap_heatmap_group_color"] == [1.0, 0.0, 0.0, 1.0]
    assert second.metadata["flatmap_heatmap_group_color"] == [0.0, 1.0, 0.0, 0.5]


def test_cluster_heatmap_color_mode_creates_cluster_and_unclustered_layers(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: "cluster"
    )
    widget._cluster_map_provider = lambda: {"a.swc": 2, "b.swc": 1}
    summary = _simple_projection_summary(module, total_nodes=4)

    widget._create_or_update_render_layer(
        _binned_render_result(module),
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert [layer.name for layer in widget._viewer.layers] == [
        "Isocortex Flatmap Heatmap: Cluster 1",
        "Isocortex Flatmap Heatmap: Cluster 2",
        "Isocortex Flatmap Heatmap: Unclustered",
    ]
    cluster_1, cluster_2, unclustered = widget._viewer.layers
    assert cluster_1.data[1, 2, 0] == 1.0
    assert cluster_2.data[0, 1, 2] == 2.0
    assert unclustered.data[1, 0, 1] == 1.0
    assert cluster_1.metadata["flatmap_heatmap_group_key"] == 1
    assert cluster_1.metadata["source_file_ids"] == ["b.swc"]
    assert cluster_1.metadata["flatmap_heatmap_group_color"] == [
        0.0,
        1.0,
        0.0,
        0.5,
    ]
    assert unclustered.metadata["flatmap_heatmap_group_key"] is None
    assert unclustered.metadata["flatmap_heatmap_group_color"] == [
        0.5,
        0.5,
        0.5,
        1.0,
    ]


def test_switching_heatmap_color_modes_removes_stale_group_layers(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    summary = _simple_projection_summary(module, total_nodes=4)
    render_result = _binned_render_result(module)
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: "individual"
    )
    widget._create_or_update_render_layer(
        render_result,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )
    assert len(widget._viewer.layers) == 3

    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: "single"
    )
    single = widget._create_or_update_render_layer(
        render_result,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert widget._viewer.layers == [single]
    assert single.name == "Isocortex Flatmap Heatmap"
    assert single.metadata["flatmap_heatmap_color_mode"] == "single"


def test_deleted_heatmap_layer_is_recreated_from_stale_cache(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    summary = _simple_projection_summary(module)
    first_render = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["a.swc"]}),
        volume=np.ones((1, 4, 4), dtype=np.float32),
        points=np.asarray([[0.0, 0.0, 0.0]]),
        point_file_ids=["a.swc"],
        summary=_simple_render_summary(module),
    )
    second_render = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["a.swc"]}),
        volume=np.full((1, 4, 4), 2.0, dtype=np.float32),
        points=np.asarray([[0.0, 1.0, 1.0]]),
        point_file_ids=["a.swc"],
        summary=_simple_render_summary(module),
    )

    first = widget._create_or_update_render_layer(
        first_render,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )
    widget._viewer.layers.remove(first)
    assert widget._projection_layer is first

    second = widget._create_or_update_render_layer(
        second_render,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert second is not first
    assert widget._projection_layer is second
    assert widget._viewer.layers == [second]
    np.testing.assert_array_equal(second.data, second_render.volume)


def test_heatmap_layer_is_recreated_when_display_viewer_changes(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    first_viewer = _DummyViewer()
    active_viewer = first_viewer
    widget._display_viewer_provider = lambda create=True: active_viewer
    summary = _simple_projection_summary(module)
    first_render = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["a.swc"]}),
        volume=np.ones((1, 4, 4), dtype=np.float32),
        points=np.asarray([[0.0, 0.0, 0.0]]),
        point_file_ids=["a.swc"],
        summary=_simple_render_summary(module),
    )
    second_render = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["a.swc"]}),
        volume=np.full((1, 4, 4), 2.0, dtype=np.float32),
        points=np.asarray([[0.0, 1.0, 1.0]]),
        point_file_ids=["a.swc"],
        summary=_simple_render_summary(module),
    )

    first = widget._create_or_update_render_layer(
        first_render,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )
    second_viewer = _DummyViewer()
    active_viewer = second_viewer

    second = widget._create_or_update_render_layer(
        second_render,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert second is not first
    assert widget._projection_layer is second
    assert first_viewer.layers == [first]
    assert second_viewer.layers == [second]
    np.testing.assert_array_equal(second.data, second_render.volume)


def test_create_region_labels_layer_adds_and_updates_labels(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    summary = FlatmapRegionLabelsSummary(
        input_voxels=4,
        selected_region_count=1,
        selected_source_voxels=2,
        valid_source_voxels=2,
        labeled_voxels=1,
        collision_voxels=1,
        y_bins=2,
        x_bins=2,
        depth_bins=1,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=25.0,
    )
    first = FlatmapRegionLabelsResult(
        labels=np.asarray([[[0, 7], [0, 0]]], dtype=np.int32),
        summary=summary,
        selected_region_ids=[7],
        represented_region_ids=[7],
    )
    metadata = {
        "projection_kind": "flatmap_region_labels",
        "selected_region_ids": [7],
        "summary": summary.to_dict(),
    }

    layer = widget._create_or_update_region_labels_layer(first, metadata)

    assert layer.name == "Flatmap Region Labels"
    np.testing.assert_array_equal(layer.data, first.labels)
    assert layer.metadata["projection_kind"] == "flatmap_region_labels"
    assert layer.metadata["napari_neuron_navigator_space"] == "flatmap"
    assert layer._napari_neuron_navigator_flatmap_region_labels_result is first
    assert widget._viewer.layers == [layer]

    second = FlatmapRegionLabelsResult(
        labels=np.asarray([[[8, 0], [0, 0]]], dtype=np.int32),
        summary=summary,
        selected_region_ids=[8],
        represented_region_ids=[8],
    )
    updated = widget._create_or_update_region_labels_layer(second, metadata)

    assert updated is layer
    np.testing.assert_array_equal(layer.data, second.labels)
    assert layer._napari_neuron_navigator_flatmap_region_labels_result is second
    assert layer.refresh_count == 1


def test_region_labels_layer_uses_display_viewer_provider(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    main_viewer = widget._viewer
    display_viewer = _DummyViewer()
    widget._display_viewer_provider = lambda create=True: (
        display_viewer if create else display_viewer
    )
    summary = FlatmapRegionLabelsSummary(
        input_voxels=4,
        selected_region_count=1,
        selected_source_voxels=2,
        valid_source_voxels=2,
        labeled_voxels=1,
        collision_voxels=0,
        y_bins=2,
        x_bins=2,
        depth_bins=1,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=25.0,
    )
    result = FlatmapRegionLabelsResult(
        labels=np.asarray([[[0, 7], [0, 0]]], dtype=np.int32),
        summary=summary,
        selected_region_ids=[7],
        represented_region_ids=[7],
    )

    layer = widget._create_or_update_region_labels_layer(
        result,
        {"projection_kind": "flatmap_region_labels"},
    )

    assert main_viewer.layers == []
    assert display_viewer.layers == [layer]
    assert layer.name == "Flatmap Region Labels"


def test_cached_region_labels_do_not_access_nrrd_or_atlas_annotation(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    profile = types.SimpleNamespace(profile_id="profile-1")
    widget._active_cache_profile = profile
    widget._region_cache_dir = Path("cache")
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._selected_region_ids_provider = lambda: [10, 11]
    widget._selected_geometry_region_ids_provider = lambda: [10, 11]
    widget._selected_region_acronyms_provider = lambda: ["R10", "R11"]
    widget._selected_region_source_provider = lambda: "custom_regions"
    widget._selected_region_scope_provider = lambda: "current_table"

    class _AtlasWithoutAnnotation:
        structures = {}

        @property
        def annotation(self):
            raise AssertionError("cached labels must not access atlas.annotation")

    widget._atlas_provider = _AtlasWithoutAnnotation
    result = types.SimpleNamespace(
        labels=np.array([[[10, 0], [0, 11]]], dtype=np.int32),
        profile_id="profile-1",
        selected_region_ids=(10, 11),
        represented_region_ids=(10, 11),
        summary=types.SimpleNamespace(
            labeled_bins=2,
            to_dict=lambda: {"labeled_bins": 2},
        ),
    )
    import napari_neuron_navigator.flatmap_region_cache as cache_module

    captured = {}
    monkeypatch.setattr(
        cache_module,
        "materialize_region_selection",
        lambda received_profile, region_ids, **kwargs: (
            captured.update(
                profile=received_profile,
                region_ids=region_ids,
                kwargs=kwargs,
            )
            or result
        ),
    )
    monkeypatch.setattr(
        module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached labels must not load NRRDs")
        ),
    )

    def _capture_layer(received, metadata, **_kwargs):
        captured["metadata"] = metadata
        return types.SimpleNamespace(data=received.labels)

    widget._create_or_update_region_labels_layer = _capture_layer
    widget._focus_projection_view = lambda *_args, **_kwargs: None
    widget._set_region_labels_status = lambda message: captured.update(message=message)

    actual = widget._create_cached_region_labels()

    assert actual is result
    assert captured["profile"] is profile
    assert captured["region_ids"] == [10, 11]
    assert captured["kwargs"]["direct_region_ids"] == [10, 11]
    assert captured["kwargs"]["include_surfaces"] is False
    assert captured["kwargs"]["include_outlines"] is False
    assert captured["metadata"]["region_selection_source"] == "custom_regions"
    assert captured["metadata"]["region_selection_scope"] == "current_table"
    assert captured["metadata"]["selected_region_acronyms"] == ["R10", "R11"]


def test_cached_allen_layer_labels_create_synchronized_planar_stack(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    profile = types.SimpleNamespace(profile_id="profile-1")
    widget._active_cache_profile = profile
    widget._region_cache_dir = Path("cache")
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_ALLEN_LAYERS
    )
    widget._selected_region_ids_provider = lambda: [1, 10, 11]
    widget._selected_region_source_provider = lambda: "custom_regions"
    widget._selected_region_scope_provider = lambda: "whole_parquet"

    class _AtlasWithoutAnnotation:
        atlas_name = "allen_mouse_25um"
        structures = {}

        @property
        def annotation(self):
            raise AssertionError("cached planar labels must not access annotation")

    atlas = _AtlasWithoutAnnotation()
    widget._atlas_provider = lambda: atlas
    layer_map = types.SimpleNamespace(
        atlas_name="allen_mouse_25um",
        atlas_version="1.2.3",
        layer_labels=("L1", "L2/3", "L4", "L5", "L6a", "L6b"),
    )
    widget._current_allen_layer_map = lambda: layer_map
    summary = types.SimpleNamespace(
        labeled_bins=2,
        to_dict=lambda: {
            "labeled_bins": 2,
            "output_shape": [6, 2, 2],
        },
    )
    result = types.SimpleNamespace(
        labels=np.asarray(
            [
                [[10, 0], [0, 0]],
                [[0, 0], [0, 11]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            dtype=np.int32,
        ),
        profile_id="profile-1",
        selected_region_ids=(1, 10, 11),
        layer_mapped_region_ids=(10, 11),
        represented_region_ids=(10, 11),
        layer_labels=layer_map.layer_labels,
        summary=summary,
    )
    import napari_neuron_navigator.flatmap_region_cache as cache_module

    captured = {}
    monkeypatch.setattr(
        cache_module,
        "materialize_allen_layer_region_selection",
        lambda received_profile, region_ids, **kwargs: (
            captured.update(
                profile=received_profile,
                region_ids=region_ids,
                kwargs=kwargs,
            )
            or result
        ),
    )
    monkeypatch.setattr(
        cache_module,
        "materialize_region_selection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("planar labels must use the Allen-layer materializer")
        ),
    )
    monkeypatch.setattr(
        module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached planar labels must not load NRRDs")
        ),
    )

    actual = widget._create_cached_region_labels()

    assert actual is result
    assert captured["profile"] is profile
    assert captured["region_ids"] == [1, 10, 11]
    assert captured["kwargs"]["style"] == "both_shaped"
    assert captured["kwargs"]["layer_map"] is layer_map
    layer = widget._region_labels_layer
    assert layer.name == "Flatmap Region Labels"
    np.testing.assert_array_equal(layer.data, result.labels)
    assert layer.axis_labels == widget._allen_layer_axis_labels()
    assert layer.metadata["flatmap_plane_mode"] == "allen_layers"
    assert layer.metadata["allen_layer_labels"] == list(layer_map.layer_labels)
    assert layer.metadata["allen_atlas_identity"] == {
        "name": "allen_mouse_25um",
        "version": "1.2.3",
    }
    assert layer.metadata["layer_mapped_region_ids"] == [10, 11]
    assert layer.metadata["region_selection_source"] == "custom_regions"
    assert layer.metadata["region_selection_scope"] == "whole_parquet"
    assert widget._viewer.dims.ndisplay == 2
    assert layer.slice_dims_calls[-1] == (widget._viewer.dims, True)
    assert "6 Allen layer planes" in widget._region_labels_status_label.text


def test_cached_allen_layer_projection_labels_share_selected_xy_space(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._active_cache_profile = types.SimpleNamespace(profile_id="profile-1")
    widget._region_cache_dir = Path("cache")
    widget._style_combo = _DummyDataCombo("both_shaped")
    widget._projection_source_combo = _DummyDataCombo(
        module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._render_mode_combo = _DummyDataCombo(module._RENDER_ALLEN_LAYERS)
    widget._allen_layer_output_combo = _DummyDataCombo(
        module.ALLEN_LAYER_OUTPUT_PROJECTION
    )
    widget._allen_layer_checkboxes = [
        _DummyValueControl(checked=index in {1, 4}) for index in range(6)
    ]
    widget._selected_region_ids_provider = lambda: [10, 11]
    widget._selected_region_acronyms_provider = lambda: ["R10", "R11"]
    widget._selected_region_source_provider = lambda: "custom_regions"
    widget._selected_region_scope_provider = lambda: "whole_parquet"
    widget._atlas_provider = lambda: types.SimpleNamespace(
        atlas_name="allen_mouse_25um",
        structures={},
    )
    layer_map = types.SimpleNamespace(
        atlas_name="allen_mouse_25um",
        atlas_version="1.2.3",
        layer_labels=("L1", "L2/3", "L4", "L5", "L6a", "L6b"),
    )
    widget._current_allen_layer_map = lambda: layer_map
    result = types.SimpleNamespace(
        labels=np.asarray([[10, 0], [0, 11]], dtype=np.int32),
        profile_id="profile-1",
        selected_region_ids=(10, 11),
        layer_mapped_region_ids=(10, 11),
        represented_region_ids=(10, 11),
        layer_labels=("L2/3", "L6a"),
        summary=types.SimpleNamespace(
            labeled_bins=2,
            to_dict=lambda: {
                "labeled_bins": 2,
                "output_shape": [2, 2],
            },
        ),
    )
    import napari_neuron_navigator.flatmap_region_cache as cache_module

    captured = {}
    monkeypatch.setattr(
        cache_module,
        "materialize_allen_layer_region_selection",
        lambda _profile, _region_ids, **kwargs: (
            captured.update(kwargs) or result
        ),
    )

    actual = widget._create_cached_region_labels()

    assert actual is result
    assert captured["selected_layer_indices"] == (1, 4)
    assert captured["output_mode"] == "projection"
    layer = widget._region_labels_layer
    assert layer.name == module._ALLEN_LAYER_PROJECTION_REGION_LABELS_LAYER_NAME
    assert layer.data.shape == (2, 2)
    assert layer.axis_labels == ("Flatmap Y", "Flatmap X")
    assert layer.metadata["flatmap_plane_mode"] == "allen_layer_projection"
    assert layer.metadata["allen_layer_indices"] == [1, 4]
    assert "projected from 2 selected Allen layer" in (
        widget._region_labels_status_label.text
    )


def test_cached_allen_layer_labels_reject_unmapped_and_clear_empty_results(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._active_cache_profile = types.SimpleNamespace(profile_id="profile-1")
    widget._region_cache_dir = Path("cache")
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_ALLEN_LAYERS
    )
    widget._selected_region_ids_provider = lambda: [99]
    widget._current_allen_layer_map = lambda: object()
    import napari_neuron_navigator.flatmap_region_cache as cache_module

    result = types.SimpleNamespace(
        labels=np.zeros((6, 2, 2), dtype=np.int32),
        layer_mapped_region_ids=(),
        represented_region_ids=(),
        summary=types.SimpleNamespace(labeled_bins=0),
    )
    monkeypatch.setattr(
        cache_module,
        "materialize_allen_layer_region_selection",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(RuntimeError, match="no terminal Allen Isocortex"):
        widget._create_cached_region_labels()

    stale = _DummyLayer(
        np.ones((6, 2, 2), dtype=np.int32),
        name="Flatmap Region Labels",
    )
    widget._viewer.layers.append(stale)
    widget._region_labels_layer = stale
    result.layer_mapped_region_ids = (99,)

    with pytest.raises(RuntimeError, match="no occupancy"):
        widget._create_cached_region_labels()

    assert stale not in widget._viewer.layers
    assert widget._region_labels_layer is None


def test_cached_depth_labels_clear_stale_layer_when_selection_has_no_occupancy(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._active_cache_profile = types.SimpleNamespace(profile_id="profile-1")
    widget._region_cache_dir = Path("cache")
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._selected_region_ids_provider = lambda: [101]
    widget._selected_geometry_region_ids_provider = lambda: [101]
    widget._selected_region_source_provider = lambda: "custom_regions"
    stale = _DummyLayer(
        np.ones((1, 2, 2), dtype=np.int32),
        name="Flatmap Region Labels",
    )
    widget._viewer.layers.append(stale)
    widget._region_labels_layer = stale
    result = types.SimpleNamespace(
        labels=np.zeros((1, 2, 2), dtype=np.int32),
        selected_region_ids=(101,),
        represented_region_ids=(),
        summary=types.SimpleNamespace(labeled_bins=0),
    )
    import napari_neuron_navigator.flatmap_region_cache as cache_module

    monkeypatch.setattr(
        cache_module,
        "materialize_region_selection",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(RuntimeError, match="no occupancy"):
        widget._create_cached_region_labels()

    assert stale not in widget._viewer.layers
    assert widget._region_labels_layer is None


def test_flatmap_region_selection_errors_are_source_aware(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._selected_region_ids_provider = lambda: []
    widget._selected_region_source_provider = lambda: "custom_regions"

    with pytest.raises(RuntimeError, match="Custom Region"):
        widget._selected_region_ids_for_labels()

    message = "The loaded atlas cannot provide Custom Isocortex Layers."
    widget._selected_region_error_provider = lambda: message
    with pytest.raises(RuntimeError, match=message):
        widget._selected_region_ids_for_labels()

    widget._selected_region_error_provider = lambda: None
    widget._selected_region_source_provider = lambda: "mask_layer"
    with pytest.raises(RuntimeError, match="do not support Mask Layer"):
        widget._selected_region_ids_for_labels()

    widget._active_cache_profile = types.SimpleNamespace(profile_id="profile-1")
    widget._selected_region_source_provider = lambda: "custom_regions"
    widget._selected_geometry_region_ids_provider = lambda: []
    widget._atlas_provider = lambda: types.SimpleNamespace(structures={})
    with pytest.raises(RuntimeError, match="Custom Region"):
        widget._cached_geometry_inputs()


def test_cached_region_geometry_uses_only_materialized_cache_arrays(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    profile = types.SimpleNamespace(profile_id="profile-1")
    widget._active_cache_profile = profile
    widget._region_cache_dir = Path("cache")
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._selected_geometry_region_ids_provider = lambda: [11, 10, 10]
    widget._selected_region_acronyms_provider = lambda: ["VISp", "MOp"]
    widget._selected_region_source_provider = lambda: "custom_regions"
    widget._selected_region_scope_provider = lambda: "current_table"
    widget._region_surfaces_layers = []
    widget._region_outlines_layers = []

    class _AtlasWithoutRuntimeGeometry:
        structures = {
            10: {
                "id": 10,
                "acronym": "VISp",
                "name": "Primary visual area",
                "rgb_triplet": [12, 34, 56],
            },
            11: {
                "id": 11,
                "acronym": "MOp",
                "name": "Primary motor area",
                "rgb_triplet": [78, 90, 123],
            },
        }

        @property
        def annotation(self):
            raise AssertionError("cached geometry must not access atlas.annotation")

        def mesh_from_structure(self, *_args, **_kwargs):
            raise AssertionError("cached geometry must not project atlas meshes")

    widget._atlas_provider = _AtlasWithoutRuntimeGeometry
    widget._set_region_labels_status = lambda _message: None

    fake_colormaps = types.ModuleType("napari.utils.colormaps")
    fake_colormaps.Colormap = lambda colors: np.asarray(colors)
    monkeypatch.setitem(sys.modules, "napari.utils.colormaps", fake_colormaps)

    import napari_neuron_navigator.flatmap_region_cache as cache_module

    materialized_surfaces = []
    materialized_outlines = []
    surface = types.SimpleNamespace(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        component_count=1,
    )
    outlines = types.SimpleNamespace(
        vectors=np.asarray(
            [[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]],
            dtype=np.float32,
        ).reshape(1, 2, 3)
    )
    monkeypatch.setattr(
        cache_module,
        "materialize_region_surface",
        lambda received_profile, region_id, **kwargs: (
            materialized_surfaces.append((received_profile, region_id, kwargs["style"]))
            or surface
        ),
    )
    monkeypatch.setattr(
        cache_module,
        "materialize_region_outlines",
        lambda received_profile, region_id, **kwargs: (
            materialized_outlines.append((received_profile, region_id, kwargs["style"]))
            or outlines
        ),
    )
    monkeypatch.setattr(
        module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached geometry must not load NRRDs")
        ),
    )

    widget._create_region_surfaces()
    widget._create_region_outlines()

    assert [layer.name for layer in widget._viewer.layers] == [
        "Flatmap Region Surfaces: VISp (10)",
        "Flatmap Region Surfaces: MOp (11)",
        "Flatmap Region Outlines: VISp (10)",
        "Flatmap Region Outlines: MOp (11)",
    ]
    assert materialized_surfaces == [
        (profile, 10, "both_shaped"),
        (profile, 11, "both_shaped"),
    ]
    assert materialized_outlines == [
        (profile, 10, "both_shaped"),
        (profile, 11, "both_shaped"),
    ]
    expected_visp = np.asarray([12, 34, 56, 255], dtype=float) / 255
    np.testing.assert_allclose(widget._viewer.layers[0].colormap[0], expected_visp)
    np.testing.assert_allclose(widget._viewer.layers[2].edge_color, expected_visp)
    for layer in widget._viewer.layers:
        assert layer.metadata["source"] == "precomputed_cache"
        assert layer.metadata["napari_neuron_navigator_space"] == "flatmap"
        assert layer.metadata["region_selection_source"] == "custom_regions"
        assert layer.metadata["region_selection_scope"] == "current_table"
        assert layer.metadata["selected_region_ids"] == [10, 11]
        assert layer.metadata["selected_region_acronyms"] == ["VISp", "MOp"]
    assert widget._viewer.layers[0].metadata["region_name"] == "Primary visual area"
    assert widget._viewer.layers[1].metadata["region_acronym"] == "MOp"


def test_apply_region_appearance_restyles_layers_without_materializing_cache(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    structures = {
        10: {
            "id": 10,
            "acronym": "VISp",
            "rgb_triplet": [12, 34, 56],
            "structure_id_path": [10],
        }
    }
    widget._atlas_provider = lambda: types.SimpleNamespace(structures=structures)
    store = RegionAppearanceStore(
        overrides={
            10: RegionAppearanceOverride(
                color_mode="custom",
                color_rgb=(0.25, 0.5, 0.75),
                fill_opacity=0.4,
                outline_visible=False,
            )
        }
    )
    widget._region_appearance_provider = lambda: store

    class _DirectLabelColormap:
        def __init__(self, *, color_dict) -> None:
            self.color_dict = color_dict

    fake_utils = sys.modules["napari.utils"]
    fake_utils.DirectLabelColormap = _DirectLabelColormap
    fake_colormaps = types.ModuleType("napari.utils.colormaps")
    fake_colormaps.Colormap = lambda colors: np.asarray(colors)
    monkeypatch.setitem(sys.modules, "napari.utils.colormaps", fake_colormaps)
    monkeypatch.setattr(
        module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("restyling must not load NRRDs")
        ),
    )

    label_data = np.asarray([[10, 0], [0, 10]], dtype=np.int32)
    labels = _DummyLayer(
        label_data,
        name="Flatmap Region Labels 2D",
        opacity=0.35,
        metadata={
            "region_layer_kind": "flatmap_labels",
            "represented_region_ids": [10],
        },
    )
    surface = _DummyLayer(
        np.zeros((3, 3), dtype=np.float32),
        name="Flatmap Region Surfaces: VISp (10)",
        opacity=0.45,
        metadata={"region_layer_kind": "flatmap_surface", "region_id": 10},
    )
    setattr(surface, "_napari_neuron_navigator_region_base_opacity", 0.45)
    outline = _DummyLayer(
        np.zeros((1, 2, 3), dtype=np.float32),
        name="Flatmap Region Outlines: VISp (10)",
        edge_color=[1.0, 0.0, 0.0, 1.0],
        opacity=0.9,
        metadata={"region_layer_kind": "flatmap_outline", "region_id": 10},
    )
    setattr(outline, "_napari_neuron_navigator_region_base_opacity", 0.9)
    widget._viewer.layers.extend([labels, surface, outline])

    widget.apply_region_appearance()

    assert labels.data is label_data
    np.testing.assert_allclose(
        labels.colormap.color_dict[10],
        [0.25, 0.5, 0.75, 0.4],
    )
    np.testing.assert_allclose(surface.colormap[0], [0.25, 0.5, 0.75, 1.0])
    assert surface.opacity == pytest.approx(0.18)
    np.testing.assert_allclose(outline.edge_color, [0.25, 0.5, 0.75, 1.0])
    assert outline.visible is False

    # A direct napari layer toggle remains the global gate, while removing a
    # per-region hide can re-enable a layer that was hidden only by the style.
    surface.visible = False
    store = RegionAppearanceStore()
    widget.apply_region_appearance()
    assert surface.visible is False
    assert outline.visible is True


def test_apply_region_appearance_restyles_flat_labels_by_descendant_id(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    structures = {
        1: {
            "id": 1,
            "acronym": "ROOT",
            "rgb_triplet": [128, 128, 128],
            "structure_id_path": [1],
        },
        2: {
            "id": 2,
            "acronym": "C2",
            "rgb_triplet": [12, 34, 56],
            "structure_id_path": [1, 2],
        },
        3: {
            "id": 3,
            "acronym": "C3",
            "rgb_triplet": [78, 90, 123],
            "structure_id_path": [1, 3],
        },
    }
    widget._atlas_provider = lambda: types.SimpleNamespace(structures=structures)
    widget._region_appearance_provider = lambda: RegionAppearanceStore(
        overrides={
            1: RegionAppearanceOverride(
                color_mode="custom",
                color_rgb=(1.0, 0.0, 0.0),
                fill_opacity=0.6,
            ),
            2: RegionAppearanceOverride(
                color_mode="custom",
                color_rgb=(0.0, 0.0, 1.0),
            ),
        }
    )

    class _DirectLabelColormap:
        def __init__(self, *, color_dict) -> None:
            self.color_dict = color_dict

    sys.modules["napari.utils"].DirectLabelColormap = _DirectLabelColormap
    label_data = np.asarray([[2, 3], [2, 0]], dtype=np.int32)
    labels = _DummyLayer(
        label_data,
        name="Flatmap Region Labels 2D",
        opacity=0.35,
        metadata={
            "region_layer_kind": "flatmap_labels",
            "flatmap_plane_mode": module.FLATMAP_PLANE_MODE_FLAT,
            "represented_region_ids": [1],
            "represented_source_region_ids": [2, 3],
        },
    )
    setattr(
        labels,
        "_napari_neuron_navigator_flatmap_region_labels_result",
        types.SimpleNamespace(
            represented_region_ids=(1,),
            represented_source_region_ids=(2, 3),
        ),
    )
    widget._viewer.layers.append(labels)

    widget.apply_region_appearance()

    assert labels.data is label_data
    assert 1 not in labels.colormap.color_dict
    np.testing.assert_allclose(
        labels.colormap.color_dict[2],
        [0.0, 0.0, 1.0, 0.6],
    )
    np.testing.assert_allclose(
        labels.colormap.color_dict[3],
        [1.0, 0.0, 0.0, 0.6],
    )


def test_cached_flat_region_labels_create_a_two_dimensional_layer(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    profile = types.SimpleNamespace(profile_id="profile-1")
    widget._active_cache_profile = profile
    widget._region_cache_dir = Path("cache")
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_HEATMAP
    )
    widget._selected_region_ids_provider = lambda: [10, 11]
    widget._selected_geometry_region_ids_provider = lambda: [315]
    widget._selected_region_acronyms_provider = lambda: ["Isocortex"]
    widget._selected_region_source_provider = lambda: "atlas_regions"
    widget._selected_region_scope_provider = lambda: "whole_parquet"
    structures = {
        315: {"id": 315, "acronym": "Isocortex", "rgb_triplet": [1, 2, 3]},
        10: {
            "id": 10,
            "acronym": "C10",
            "rgb_triplet": [4, 5, 6],
            "structure_id_path": [315, 10],
        },
        11: {
            "id": 11,
            "acronym": "C11",
            "rgb_triplet": [7, 8, 9],
            "structure_id_path": [315, 11],
        },
    }
    widget._atlas_provider = lambda: types.SimpleNamespace(structures=structures)

    result = types.SimpleNamespace(
        labels=np.array([[10, 0], [0, 11]], dtype=np.int32),
        profile_id="profile-1",
        selected_region_ids=(10, 11),
        direct_region_ids=(315,),
        represented_region_ids=(315,),
        represented_source_region_ids=(10, 11),
        grid_spec={
            "label_grouping": "source_region",
            "geometry_grouping": "selected_root",
        },
        summary=types.SimpleNamespace(
            labeled_bins=2,
            represented_region_count=1,
            to_dict=lambda: {"labeled_bins": 2, "output_shape": [2, 2]},
        ),
    )
    import napari_neuron_navigator.flatmap_region_cache as cache_module

    captured = {}
    monkeypatch.setattr(
        cache_module,
        "materialize_flat_region_selection",
        lambda received_profile, region_ids, **kwargs: (
            captured.update(
                profile=received_profile,
                region_ids=region_ids,
                kwargs=kwargs,
                calls=captured.get("calls", 0) + 1,
            )
            or result
        ),
    )
    monkeypatch.setattr(
        module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("collapsed labels must not load NRRDs")
        ),
    )
    widget._focus_projection_view = lambda *args, **kwargs: captured.update(
        focus_ndisplay=kwargs.get("ndisplay")
    )
    widget._set_region_labels_status = lambda message: captured.update(message=message)

    actual = widget._create_cached_region_labels()

    assert actual is result
    assert captured["calls"] == 1
    assert captured["profile"] is profile
    assert captured["kwargs"]["direct_region_ids"] == [315]
    assert captured["kwargs"]["atlas_structures"] is structures
    assert captured["kwargs"]["include_outlines"] is False
    assert captured["focus_ndisplay"] == 2

    layer = widget._region_labels_layer
    assert layer.name == "Flatmap Region Labels 2D"
    assert layer.data.shape == (2, 2)
    assert layer.axis_labels == (
        module._FLATMAP_AXIS_LABEL_Y,
        module._FLATMAP_AXIS_LABEL_X,
    )
    assert layer.metadata["flatmap_plane_mode"] == module.FLATMAP_PLANE_MODE_FLAT
    assert layer.metadata["direct_region_ids"] == [315]
    assert layer.metadata["label_grouping"] == "source_region"
    assert layer.metadata["geometry_grouping"] == "selected_root"
    assert sorted(np.unique(layer.data).tolist()) == [0, 10, 11]
    # Overlays must stay pixel-aligned with the anisotropic 2D heatmap.
    assert "scale" not in layer.init_kwargs
    assert "translate" not in layer.init_kwargs
    assert "collapsed region bin(s)" in captured["message"]


def test_cached_flat_region_outlines_use_only_materialized_cache_arrays(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    profile = types.SimpleNamespace(profile_id="profile-1")
    widget._active_cache_profile = profile
    widget._region_cache_dir = Path("cache")
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_FLAT_VECTOR
    )
    widget._selected_region_ids_provider = lambda: [10, 11]
    widget._selected_geometry_region_ids_provider = lambda: [10, 11]
    widget._selected_region_acronyms_provider = lambda: ["VISp", "MOp"]
    widget._selected_region_source_provider = lambda: "atlas_regions"
    widget._selected_region_scope_provider = lambda: "whole_parquet"
    widget._region_surfaces_layers = []
    widget._region_outlines_layers = []

    class _AtlasWithoutRuntimeGeometry:
        structures = {
            10: {
                "id": 10,
                "acronym": "VISp",
                "name": "Primary visual area",
                "rgb_triplet": [12, 34, 56],
            },
            11: {
                "id": 11,
                "acronym": "MOp",
                "name": "Primary motor area",
                "rgb_triplet": [78, 90, 123],
            },
        }

        @property
        def annotation(self):
            raise AssertionError("collapsed outlines must not access atlas.annotation")

        def mesh_from_structure(self, *_args, **_kwargs):
            raise AssertionError("collapsed outlines must not project atlas meshes")

    widget._atlas_provider = _AtlasWithoutRuntimeGeometry
    widget._set_region_labels_status = lambda _message: None

    import napari_neuron_navigator.flatmap_region_cache as cache_module

    def _outline(region_id):
        return types.SimpleNamespace(
            region_id=region_id,
            vectors=np.asarray([[[0.5, 0.5], [0.0, 1.0]]], dtype=np.float32),
            union_region_ids=(region_id,),
            represented_region_ids=(region_id,),
            planar_bin_count=3,
        )

    calls = []
    monkeypatch.setattr(
        cache_module,
        "materialize_flat_region_selection",
        lambda received_profile, region_ids, **kwargs: (
            calls.append((received_profile, list(region_ids), kwargs))
            or types.SimpleNamespace(
                profile_id="profile-1",
                grid_spec={
                    "label_grouping": "source_region",
                    "geometry_grouping": "selected_root",
                },
                outlines=(_outline(10), _outline(11)),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("collapsed outlines must not load NRRDs")
        ),
    )

    widget._create_region_outlines()

    # One call for the whole selection, not one per selected region.
    assert len(calls) == 1
    assert calls[0][0] is profile
    assert calls[0][2]["direct_region_ids"] == [10, 11]
    assert calls[0][2]["include_outlines"] is True
    assert [layer.name for layer in widget._viewer.layers] == [
        "Flatmap Region Outlines 2D: VISp (10)",
        "Flatmap Region Outlines 2D: MOp (11)",
    ]
    assert widget._viewer.dims.ndisplay == 2
    for layer in widget._viewer.layers:
        assert layer.data.shape[1:] == (2, 2)
        # napari would otherwise draw an arrowhead on every perimeter segment.
        assert layer.init_kwargs["vector_style"] == "line"
        assert layer.axis_labels == (
            module._FLATMAP_AXIS_LABEL_Y,
            module._FLATMAP_AXIS_LABEL_X,
        )
        assert layer.metadata["flatmap_plane_mode"] == module.FLATMAP_PLANE_MODE_FLAT
        assert layer.metadata["projection_kind"] == "flatmap_flat_region_outlines"
        assert layer.metadata["napari_neuron_navigator_space"] == "flatmap"
        assert layer.metadata["planar_bin_count"] == 3
        assert layer.metadata["geometry_grouping"] == "selected_root"
        assert "scale" not in layer.init_kwargs
        assert "translate" not in layer.init_kwargs
    expected_visp = np.asarray([12, 34, 56, 255], dtype=float) / 255
    np.testing.assert_allclose(widget._viewer.layers[0].edge_color, expected_visp)


def test_flat_region_overlays_are_retired_with_the_grid(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    labels_layer = _DummyLayer(
        np.zeros((2, 2), dtype=np.int32),
        name="Flatmap Region Labels 2D",
    )
    outline_layer = _DummyLayer(
        np.zeros((1, 2, 2), dtype=np.float32),
        name="Flatmap Region Outlines 2D: VISp (10)",
    )
    widget._viewer.layers.extend([labels_layer, outline_layer])
    widget._region_labels_layer = labels_layer
    widget._region_outlines_layers = [outline_layer]
    widget._region_surfaces_layers = []

    # Both names keep the depth-grid prefixes, so the prefix-based clear paths
    # already cover them.
    assert set(widget._current_cached_region_layers()) == {labels_layer, outline_layer}

    widget._invalidate_flatmap_grid_layers()

    assert labels_layer not in widget._viewer.layers
    assert outline_layer not in widget._viewer.layers
    assert widget._region_labels_layer is None
    assert widget._region_outlines_layers == []


def test_empty_custom_geometry_replaces_stale_layer_families(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._active_cache_profile = types.SimpleNamespace(profile_id="profile-1")
    widget._region_cache_dir = Path("cache")
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._selected_geometry_region_ids_provider = lambda: [101]
    widget._selected_region_source_provider = lambda: "custom_regions"
    widget._atlas_provider = lambda: types.SimpleNamespace(
        structures={
            101: {
                "id": 101,
                "acronym": "C101",
                "rgb_triplet": [1, 2, 3],
            }
        }
    )
    stale_surface = _DummyLayer(
        np.zeros((1, 3), dtype=np.float32),
        name="Flatmap Region Surfaces: OLD (1)",
    )
    stale_outline = _DummyLayer(
        np.zeros((1, 2, 3), dtype=np.float32),
        name="Flatmap Region Outlines: OLD (1)",
    )
    widget._viewer.layers.extend([stale_surface, stale_outline])
    widget._region_surfaces_layers = [stale_surface]
    widget._region_outlines_layers = [stale_outline]
    import napari_neuron_navigator.flatmap_region_cache as cache_module

    monkeypatch.setattr(
        cache_module,
        "materialize_region_surface",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cache_module,
        "materialize_region_outlines",
        lambda *_args, **_kwargs: None,
    )

    widget._create_region_surfaces()

    assert stale_surface not in widget._viewer.layers
    assert stale_outline in widget._viewer.layers
    assert widget._region_surfaces_layers == []

    widget._create_region_outlines()

    assert stale_outline not in widget._viewer.layers
    assert widget._region_outlines_layers == []

    # A collapsed 2D outline shares the prefix, so the depth path retires it too.
    stale_flat_outline = _DummyLayer(
        np.zeros((1, 2, 2), dtype=np.float32),
        name="Flatmap Region Outlines 2D: OLD (1)",
    )
    widget._viewer.layers.append(stale_flat_outline)
    widget._region_outlines_layers = [stale_flat_outline]

    widget._create_region_outlines()

    assert stale_flat_outline not in widget._viewer.layers
    assert widget._region_outlines_layers == []


def test_heatmap_workaround_swallows_thumbnail_rank_mismatch(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    class _CrashLayer(_DummyLayer):
        def _update_thumbnail(self) -> None:
            raise RuntimeError("sequence argument must have length equal to input rank")

    layer = _CrashLayer(
        np.zeros((2, 4, 4), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        metadata={"flatmap_heatmap_contrast_limits": (0.0, 5.0)},
        contrast_limits=(0.0, 5.0),
    )

    widget._install_heatmap_layer_workarounds(layer)

    layer._update_thumbnail()
    assert layer._napari_neuron_navigator_flatmap_thumbnail_warning_logged is True


def test_heatmap_workaround_keeps_stable_limits_during_3d_slice_update(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _DummyLayer(
        np.zeros((2, 4, 4), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        metadata={"flatmap_heatmap_contrast_limits": (0.0, 11.0)},
        contrast_limits=(0.0, 11.0),
    )
    layer._keep_auto_contrast = True
    layer._slice_input = types.SimpleNamespace(ndisplay=3)

    widget._install_heatmap_layer_workarounds(layer)
    response = types.SimpleNamespace(
        slice_input=types.SimpleNamespace(ndisplay=3),
        payload={"slice": 1},
    )
    result = layer._update_slice_response(response)

    assert result is response
    assert layer.slice_updates == [response]
    assert layer._keep_auto_contrast is True
    assert layer.contrast_limits == (0.0, 11.0)
    assert layer.contrast_limits_range == (0.0, 11.0)


def test_heatmap_workaround_preserves_2d_auto_contrast(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _DummyLayer(
        np.zeros((2, 4, 4), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        metadata={"flatmap_heatmap_contrast_limits": (0.0, 11.0)},
        contrast_limits=(0.0, 11.0),
    )
    layer._keep_auto_contrast = True
    layer._slice_input = types.SimpleNamespace(ndisplay=2)

    widget._install_heatmap_layer_workarounds(layer)
    response = types.SimpleNamespace(
        slice_input=types.SimpleNamespace(ndisplay=2),
        payload={"slice": 2},
    )
    result = layer._update_slice_response(response)

    assert result is response
    assert layer.slice_updates == [response]
    assert layer.contrast_limits == (4.0, 5.0)
    assert layer.contrast_limits_range == (4.0, 5.0)


def test_heatmap_status_guard_handles_stale_2d_slice(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _DummyLayer(np.zeros((2, 4, 4)), name="Isocortex Flatmap Heatmap")
    layer._slice = types.SimpleNamespace(
        image=types.SimpleNamespace(raw=np.zeros((4, 4), dtype=np.float32))
    )
    layer.raise_status_error = True

    widget._install_heatmap_status_guard(layer)

    status = layer.get_status(
        np.asarray([1.2, 2.5, 3.6]),
        view_direction=np.asarray([1.0, 0.0, 0.0]),
        dims_displayed=[0, 1, 2],
        world=True,
    )

    assert status["coords"] == " [1 2 4]"
    assert status["coordinates"] == " [1 2 4]: "
    assert status["value"] == ""


def test_heatmap_status_guard_reraises_unrelated_index_errors(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _DummyLayer(np.zeros((2, 4, 4)), name="Isocortex Flatmap Heatmap")
    layer._slice = types.SimpleNamespace(
        image=types.SimpleNamespace(raw=np.zeros((2, 4, 4), dtype=np.float32))
    )
    layer.raise_status_error = True
    layer.status_error_message = "index 99 is out of bounds for axis 0"

    widget._install_heatmap_status_guard(layer)

    with np.testing.assert_raises(IndexError):
        layer.get_status(
            np.asarray([1.2, 2.5, 3.6]),
            view_direction=np.asarray([1.0, 0.0, 0.0]),
            dims_displayed=[0, 1, 2],
            world=True,
        )


def test_create_points_layer_uses_table_colors(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    render_summary = module.FlatmapRenderSummary(
        total_nodes=2,
        flatmap_valid_nodes=2,
        depth_valid_nodes=1,
        depth_minus_one_nodes=1,
        rendered_nodes=2,
        excluded_depth_minus_one_nodes=0,
        nonzero_voxels=2,
        traces_represented=2,
        y_bins=8,
        x_bins=8,
        depth_bins=2,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=25.0,
        includes_depth_minus_one_plane=True,
    )
    render_result = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["a.swc", "b.swc"]}),
        volume=np.zeros((2, 8, 8), dtype=np.float32),
        points=np.asarray([[0.0, 1.0, 2.0], [1.0, 3.0, 4.0]]),
        point_file_ids=["a.swc", "b.swc"],
        summary=render_summary,
    )
    summary = module.ProjectionSummary(2, 1, 0, 0, 1, 0, 0, 2, 1)

    layer = widget._create_or_update_render_layer(
        render_result,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="points",
    )

    assert layer.name == "Isocortex Flatmap Points"
    np.testing.assert_allclose(
        layer.face_color,
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.5]],
    )
    np.testing.assert_allclose(layer.data, render_result.points)
    assert layer.metadata["flatmap_render_mode"] == "points"


def test_deleted_points_layer_is_recreated_from_stale_cache(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    summary = _simple_projection_summary(module)
    first_render = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["a.swc"]}),
        volume=np.zeros((1, 4, 4), dtype=np.float32),
        points=np.asarray([[0.0, 0.0, 0.0]]),
        point_file_ids=["a.swc"],
        summary=_simple_render_summary(module),
    )
    second_render = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["b.swc"]}),
        volume=np.zeros((1, 4, 4), dtype=np.float32),
        points=np.asarray([[0.0, 1.0, 1.0]]),
        point_file_ids=["b.swc"],
        summary=_simple_render_summary(module),
    )

    first = widget._create_or_update_render_layer(
        first_render,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="points",
    )
    widget._viewer.layers.remove(first)
    assert widget._projection_layer is first

    second = widget._create_or_update_render_layer(
        second_render,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="points",
    )

    assert second is not first
    assert widget._projection_layer is second
    assert widget._viewer.layers == [second]
    np.testing.assert_array_equal(second.data, second_render.points)


def test_export_current_projection_to_path_writes_csv(monkeypatch, tmp_path) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._last_projected_nodes = pd.DataFrame(
        {
            "file_id": ["a.swc"],
            "node_id": [1],
            "valid": [True],
            "x_flat": [0.25],
            "y_flat": [0.5],
            "flatmap_lookup_mode": ["mirrored_depth"],
            "flatmap_valid": [True],
            "depth_valid": [True],
            "render_valid": [True],
            "x_flat_bin": [10],
            "y_flat_bin": [20],
            "depth_bin": [1],
            "depth_bin_label": ["0-25 um"],
            "allen_layer_index": [0],
            "allen_layer_label": ["L1"],
            "allen_layer_selected": [True],
            "allen_layer_render_index": [0],
        }
    )

    output = widget._export_current_projection_to_path(tmp_path / "projection.csv")

    assert output.exists()
    exported = pd.read_csv(output)
    assert exported["file_id"].tolist() == ["a.swc"]
    assert "depth_um" in exported.columns
    assert "coordinate_mode" in exported.columns
    assert exported["flatmap_lookup_mode"].tolist() == ["mirrored_depth"]
    assert "render_valid" in exported.columns
    assert exported["x_flat_bin"].tolist() == [10]
    assert exported["allen_layer_index"].tolist() == [0]
    assert exported["allen_layer_label"].tolist() == ["L1"]
    assert bool(exported["allen_layer_selected"].iloc[0]) is True
    assert exported["allen_layer_render_index"].tolist() == [0]
    assert "Exported flatmap projection" in widget._status_label.text


def test_precomputed_heatmap_uses_duckdb_fast_path(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_HEATMAP
    )
    called: list[bool] = []
    widget._start_precomputed_heatmap_worker = lambda: called.append(True)

    def _fail_query(_file_ids):
        raise AssertionError("fast path must not load nodes into pandas")

    widget._query_nodes = _fail_query
    widget._apply_projection_result = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("fast path must not use the synchronous apply")
    )

    widget._project()

    assert called == [True]


def test_apply_precomputed_heatmap_result_single_disables_per_node(monkeypatch) -> None:
    from napari_neuron_navigator import flatmap_heatmap as fh

    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._summary_label = _DummyLabel()
    widget._export_btn = _DummyButton()
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_HEATMAP
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._HEATMAP_COLOR_SINGLE
    )
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._active_cache_profile = None
    widget._region_cache_dir = None
    widget._cache_profile_id = lambda _profile: None
    widget._precomputed_heatmap_file_ids = ["a.swc", "b.swc"]
    notified: list[bool] = []
    widget._notify_flatmap_correlation_source_changed = lambda: notified.append(True)

    volume = np.zeros((2, 4, 4), dtype=np.float32)
    volume[0, 1, 2] = 3.0
    stats = fh.FlatmapAggregateStats(
        total_nodes=10,
        total_traces=2,
        flatmap_valid_nodes=8,
        depth_valid_nodes=6,
        depth_minus_one_nodes=0,
        rendered_nodes=6,
        traces_represented=2,
    )
    result = fh.FlatmapHeatmapVolumeResult(
        color_mode=module._HEATMAP_COLOR_SINGLE,
        volume=volume,
        grouped_volumes=(),
        render_summary=_simple_render_summary(module),
        stats=stats,
        volume_shape=(2, 4, 4),
    )

    widget._apply_precomputed_heatmap_result(result)

    assert widget._last_render_mode == module._RENDER_HEATMAP
    assert widget._last_projected_nodes is None
    assert widget._last_projection_source == module._PROJECTION_SOURCE_PRECOMPUTED
    assert widget._last_coordinate_mode == "parquet_columns"
    assert widget._export_btn.enabled is False
    assert notified == [True]
    layer = widget._projection_layer
    assert layer is not None
    assert layer.name == module._HEATMAP_LAYER_NAME
    assert layer.metadata["flatmap_render_mode"] == "heatmap"
    # Per-node correlation source is unavailable for a volume-only fast render.
    assert widget.latest_flatmap_correlation_source() is None


def test_apply_precomputed_heatmap_result_grouped_creates_layer_per_group(
    monkeypatch,
) -> None:
    from napari_neuron_navigator import flatmap_heatmap as fh

    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._summary_label = _DummyLabel()
    widget._export_btn = _DummyButton()
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_HEATMAP
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._HEATMAP_COLOR_INDIVIDUAL
    )
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._active_cache_profile = None
    widget._region_cache_dir = None
    widget._cache_profile_id = lambda _profile: None
    widget._precomputed_heatmap_file_ids = ["a.swc", "b.swc"]
    widget._notify_flatmap_correlation_source_changed = lambda: None

    groups = []
    for name in ("a.swc", "b.swc"):
        volume = np.zeros((2, 4, 4), dtype=np.float32)
        volume[0, 1, 2] = 1.0
        groups.append(
            fh.FlatmapGroupedVolume(
                group_key=name,
                label=name,
                source_file_ids=(name,),
                volume=volume,
                rendered_nodes=1,
                nonzero_voxels=1,
            )
        )
    stats = fh.FlatmapAggregateStats(4, 2, 4, 4, 0, 2, 2)
    result = fh.FlatmapHeatmapVolumeResult(
        color_mode=module._HEATMAP_COLOR_INDIVIDUAL,
        volume=None,
        grouped_volumes=tuple(groups),
        render_summary=_simple_render_summary(module, total_nodes=2),
        stats=stats,
        volume_shape=(2, 4, 4),
    )

    widget._apply_precomputed_heatmap_result(result)

    grouped_layers = [
        layer
        for layer in widget._viewer.layers
        if str(getattr(layer, "name", "")).startswith(
            module._GROUPED_HEATMAP_LAYER_PREFIX
        )
    ]
    assert len(grouped_layers) == 2
    assert widget._export_btn.enabled is False
    assert widget._last_projected_nodes is None


def test_apply_precomputed_allen_layer_result_uses_2d_stack(
    monkeypatch,
) -> None:
    from napari_neuron_navigator import flatmap_heatmap as fh

    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._summary_label = _DummyLabel()
    widget._export_btn = _DummyButton()
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_ALLEN_LAYERS
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._HEATMAP_COLOR_SINGLE
    )
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._active_cache_profile = None
    widget._region_cache_dir = None
    widget._cache_profile_id = lambda _profile: None
    widget._precomputed_heatmap_file_ids = ["a.swc"]
    widget._notify_flatmap_correlation_source_changed = lambda: None
    volume = np.zeros((6, 4, 4), dtype=np.float32)
    volume[0, 1, 2] = 1.0
    volume[1, 2, 3] = 1.0
    stats = fh.AllenLayerAggregateStats(
        total_nodes=2,
        total_traces=1,
        flatmap_valid_nodes=2,
        layer_classified_nodes=2,
        rendered_nodes=2,
        traces_represented=1,
    )
    result = fh.AllenLayerHeatmapVolumeResult(
        color_mode=module._HEATMAP_COLOR_SINGLE,
        volume=volume,
        grouped_volumes=(),
        summary=_simple_allen_layer_summary(module),
        stats=stats,
        volume_shape=(6, 4, 4),
    )

    widget._apply_precomputed_allen_layer_result(result)

    assert widget._last_render_mode == module._RENDER_ALLEN_LAYERS
    assert widget._last_projected_nodes is None
    assert widget._export_btn.enabled is False
    assert widget._viewer.dims.ndisplay == 2
    assert widget._projection_layer.name == module._ALLEN_LAYER_HEATMAP_LAYER_NAME
    assert widget._projection_layer.metadata["flatmap_plane_mode"] == "allen_layers"
    assert widget.latest_flatmap_correlation_source() is None


def test_apply_precomputed_allen_layer_result_rejects_no_mapped_nodes(
    monkeypatch,
) -> None:
    from napari_neuron_navigator import flatmap_heatmap as fh

    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._summary_label = _DummyLabel()
    widget._export_btn = _DummyButton()
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._active_cache_profile = None
    widget._region_cache_dir = None
    widget._cache_profile_id = lambda _profile: None
    widget._precomputed_heatmap_file_ids = ["outside.swc"]
    widget._notify_flatmap_correlation_source_changed = lambda: None
    stats = fh.AllenLayerAggregateStats(
        total_nodes=1,
        total_traces=1,
        flatmap_valid_nodes=1,
        layer_classified_nodes=0,
        rendered_nodes=0,
        traces_represented=0,
    )
    summary = module.AllenLayerStackSummary(
        total_nodes=1,
        flatmap_valid_nodes=1,
        layer_classified_nodes=0,
        rendered_nodes=0,
        excluded_non_layer_nodes=1,
        nonzero_voxels=0,
        traces_represented=0,
        y_bins=4,
        x_bins=4,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        layer_labels=("L1", "L2/3", "L4", "L5", "L6a", "L6b"),
        layer_node_counts=(0, 0, 0, 0, 0, 0),
        atlas_name="allen_mouse_25um",
    )
    result = fh.AllenLayerHeatmapVolumeResult(
        color_mode=module._HEATMAP_COLOR_SINGLE,
        volume=np.zeros((6, 4, 4), dtype=np.float32),
        grouped_volumes=(),
        summary=summary,
        stats=stats,
        volume_shape=(6, 4, 4),
    )

    with pytest.raises(RuntimeError, match="No selected flatmap-valid nodes"):
        widget._apply_precomputed_allen_layer_result(result)

    assert widget._viewer.layers == []


class _RectangularCacheProfile:
    """A profile whose grid is 3 tall by 7 wide, as a derived grid really is."""

    def __init__(self, profile_id: str = "rectangular-profile") -> None:
        self.profile_id = profile_id
        self.atlas = {"name": "allen_mouse_25um"}
        self.grid = {
            "output_shape": [1, 3, 7],
            "y_bins": 3,
            "x_bins": 7,
            "depth_bins": 1,
            "depth_bin_um": 25.0,
            "x_bounds": [0.0, 1.0],
            "y_bounds": [0.0, 1.0],
            "depth_bounds_um": [0.0, 25.0],
            "includes_depth_minus_one_plane": False,
        }

    def style(self, style: str):
        assert style == "both_shaped"
        return types.SimpleNamespace(grid_spec=self.grid)


def _rectangular_match_widget(module, monkeypatch):
    widget = _widget(module)
    _configure_cache_activation_widget(widget, module)
    widget._last_render_summary = module.FlatmapRenderSummary(
        total_nodes=1,
        flatmap_valid_nodes=1,
        depth_valid_nodes=1,
        depth_minus_one_nodes=0,
        rendered_nodes=1,
        excluded_depth_minus_one_nodes=0,
        nonzero_voxels=1,
        traces_represented=1,
        y_bins=3,
        x_bins=7,
        depth_bins=1,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=25.0,
        includes_depth_minus_one_plane=False,
    )
    widget._last_volume_shape = (1, 3, 7)
    return widget


def test_render_matches_a_rectangular_cache_profile(monkeypatch) -> None:
    """A render on the profile's own 3x7 grid must be recognized as matching."""
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _rectangular_match_widget(module, monkeypatch)

    assert widget._render_matches_cache_profile(_RectangularCacheProfile()) is True


def test_render_with_swapped_axes_does_not_match_a_rectangular_profile(
    monkeypatch,
) -> None:
    """A 7x3 render must not pass as a 3x7 profile.

    With a square profile this transpose is invisible, so the comparison has to
    check each axis against its own stored count.
    """
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _rectangular_match_widget(module, monkeypatch)
    summary = widget._last_render_summary
    widget._last_render_summary = dataclasses.replace(summary, y_bins=7, x_bins=3)
    widget._last_volume_shape = (1, 7, 3)

    assert widget._render_matches_cache_profile(_RectangularCacheProfile()) is False


def test_render_with_a_re_derived_x_count_does_not_match(monkeypatch) -> None:
    """A render that ignored the stored x count must be discarded, not kept.

    This is why cache-backed renders pass the profile's ``x_bins`` verbatim: a
    silently re-derived count would put the heatmap on a different grid than the
    cached region mask, which the worker later rejects outright.
    """
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _rectangular_match_widget(module, monkeypatch)
    summary = widget._last_render_summary
    widget._last_render_summary = dataclasses.replace(summary, x_bins=6)
    widget._last_volume_shape = (1, 3, 6)

    assert widget._render_matches_cache_profile(_RectangularCacheProfile()) is False


def test_current_bins_read_a_rectangular_profile_verbatim(monkeypatch) -> None:
    """The render path must take both counts from the active profile."""
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._active_cache_profile = _RectangularCacheProfile()
    widget._y_bins_spin = types.SimpleNamespace(value=lambda: 256)

    assert widget._current_y_bins() == 3
    assert widget._current_x_bins() == 7

    # With no cache profile the x count is left for the builder to derive.
    widget._active_cache_profile = None
    assert widget._current_y_bins() == 256
    assert widget._current_x_bins() is None
