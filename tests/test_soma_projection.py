from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from napari_neuron_navigator.isocortex_layers import CustomRegionSelectionGroup
from napari_neuron_navigator.neuron_table_ops import (
    ClusterFilterSelection,
    distinct_colors_for_file_ids,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_PATCHED_MODULE_NAMES = [
    "napari",
    "napari.utils",
    "napari.utils.notifications",
    "qtpy",
    "qtpy.QtCore",
    "qtpy.QtWidgets",
    "napari_neuron_navigator.widgets",
    "napari_neuron_navigator.widgets.analysis_tab",
    "napari_neuron_navigator.widgets.custom_region_selector",
    "napari_neuron_navigator.widgets.mask_layer_selector",
    "napari_neuron_navigator.widgets.neuron_table",
    "napari_neuron_navigator.widgets.region_selector",
    "napari_neuron_navigator.widgets.reference_layers",
    "napari_neuron_navigator.widgets.neuron_viewer",
    "napari_neuron_navigator.widgets.slice_projection",
]
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _PATCHED_MODULE_NAMES}


class _DummyAnalysisSignal:
    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, *_args, **_kwargs) -> None:
        if _args:
            self._callbacks.append(_args[0])

    def emit(self, *args, **kwargs) -> None:
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


class _Signal:
    """Minimal descriptor stand-in for ``qtpy.QtCore.Signal``."""

    def __init__(self, *_args, **_kwargs) -> None:
        self._storage_name = ""

    def __set_name__(self, _owner, name: str) -> None:
        self._storage_name = f"__signal_{name}"

    def __get__(self, instance, _owner):
        if instance is None:
            return self
        if self._storage_name not in instance.__dict__:
            instance.__dict__[self._storage_name] = _DummyAnalysisSignal()
        return instance.__dict__[self._storage_name]


class _FakeAnalysisTabWidget:
    def __init__(self, *_args, **_kwargs) -> None:
        self.cluster_colors_updated = _DummyAnalysisSignal()

    def set_slice_projector(self, *_args, **_kwargs) -> None:
        return None

    def set_database(self, *_args, **_kwargs) -> None:
        return None

    def set_atlas(self, *_args, **_kwargs) -> None:
        return None

    def apply_cluster_colors(self) -> None:
        return None


class _FakeQTimer:
    def __init__(self, *_args, **_kwargs) -> None:
        self.timeout = _DummyAnalysisSignal()

    def setSingleShot(self, *_args, **_kwargs) -> None:
        return None

    def setInterval(self, *_args, **_kwargs) -> None:
        return None

    def start(self, *_args, **_kwargs) -> None:
        return None

    def stop(self, *_args, **_kwargs) -> None:
        return None

    @staticmethod
    def singleShot(_delay, callback) -> None:
        callback()


class _FakeQt:
    Horizontal = 1
    WindowNoState = 0
    WindowFullScreen = 4


class _FakeQEvent:
    MouseButtonPress = 2
    MouseButtonRelease = 3
    MouseButtonDblClick = 4
    Show = 5
    Close = 6
    Hide = 7
    DeferredDelete = 8
    Destroy = 9

    class Type:
        Close = 6


class _FakeWidget:
    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _FakeApplication(_FakeWidget):
    @staticmethod
    def processEvents() -> None:
        return None


fake_qtcore = types.ModuleType("qtpy.QtCore")
fake_qtcore.QEvent = _FakeQEvent
fake_qtcore.QObject = _FakeWidget
fake_qtcore.Qt = _FakeQt
fake_qtcore.QThread = _FakeWidget
fake_qtcore.QTimer = _FakeQTimer
fake_qtcore.Signal = _Signal
sys.modules["qtpy.QtCore"] = fake_qtcore

fake_qtwidgets = types.ModuleType("qtpy.QtWidgets")
for _name, _value in {
    "QAbstractItemView": _FakeWidget,
    "QApplication": _FakeApplication,
    "QCheckBox": _FakeWidget,
    "QComboBox": _FakeWidget,
    "QDoubleSpinBox": _FakeWidget,
    "QFileDialog": _FakeWidget,
    "QGroupBox": _FakeWidget,
    "QHeaderView": _FakeWidget,
    "QHBoxLayout": _FakeWidget,
    "QLabel": _FakeWidget,
    "QListWidget": _FakeWidget,
    "QMenu": _FakeWidget,
    "QProgressBar": _FakeWidget,
    "QPushButton": _FakeWidget,
    "QScrollArea": _FakeWidget,
    "QSlider": _FakeWidget,
    "QSpinBox": _FakeWidget,
    "QStackedWidget": _FakeWidget,
    "QTableWidget": _FakeWidget,
    "QTableWidgetItem": _FakeWidget,
    "QTabWidget": _FakeWidget,
    "QVBoxLayout": _FakeWidget,
    "QWidget": _FakeWidget,
}.items():
    setattr(fake_qtwidgets, _name, _value)
sys.modules["qtpy.QtWidgets"] = fake_qtwidgets

fake_qtpy = types.ModuleType("qtpy")
fake_qtpy.QtCore = fake_qtcore
fake_qtpy.QtWidgets = fake_qtwidgets
sys.modules["qtpy"] = fake_qtpy

fake_napari_notifications = types.ModuleType("napari.utils.notifications")
fake_napari_notifications.show_info = lambda *args, **kwargs: None
fake_napari_notifications.show_warning = lambda *args, **kwargs: None
sys.modules["napari.utils.notifications"] = fake_napari_notifications

fake_napari_utils = types.ModuleType("napari.utils")
fake_napari_utils.notifications = fake_napari_notifications
sys.modules["napari.utils"] = fake_napari_utils

fake_napari = types.ModuleType("napari")
fake_napari.utils = fake_napari_utils
sys.modules["napari"] = fake_napari

fake_analysis_module = types.ModuleType("napari_neuron_navigator.widgets.analysis_tab")
fake_analysis_module.AnalysisTabWidget = _FakeAnalysisTabWidget
sys.modules["napari_neuron_navigator.widgets.analysis_tab"] = fake_analysis_module

fake_custom_region_selector_module = types.ModuleType(
    "napari_neuron_navigator.widgets.custom_region_selector"
)
fake_custom_region_selector_module.CustomRegionSelectorWidget = _FakeWidget
sys.modules["napari_neuron_navigator.widgets.custom_region_selector"] = (
    fake_custom_region_selector_module
)

fake_mask_selector_module = types.ModuleType(
    "napari_neuron_navigator.widgets.mask_layer_selector"
)
fake_mask_selector_module.MaskLayerSelectorWidget = _FakeWidget
sys.modules["napari_neuron_navigator.widgets.mask_layer_selector"] = fake_mask_selector_module

fake_neuron_table_module = types.ModuleType("napari_neuron_navigator.widgets.neuron_table")
fake_neuron_table_module.NeuronTableWidget = _FakeWidget
sys.modules["napari_neuron_navigator.widgets.neuron_table"] = fake_neuron_table_module

fake_region_selector_module = types.ModuleType(
    "napari_neuron_navigator.widgets.region_selector"
)
fake_region_selector_module.RegionSelectorWidget = _FakeWidget
sys.modules["napari_neuron_navigator.widgets.region_selector"] = fake_region_selector_module

fake_reference_layers_module = types.ModuleType(
    "napari_neuron_navigator.widgets.reference_layers"
)
for _name in (
    "add_allen_template",
    "add_brain_outline",
    "add_region_id_segmentation",
    "add_region_mesh",
    "add_region_mesh_group",
    "add_region_segmentation",
    "remove_region_layers",
    "remove_region_segmentation",
):
    setattr(fake_reference_layers_module, _name, lambda *args, **kwargs: None)
sys.modules["napari_neuron_navigator.widgets.reference_layers"] = fake_reference_layers_module

sys.modules.pop("napari_neuron_navigator.widgets.neuron_viewer", None)
widgets_package = types.ModuleType("napari_neuron_navigator.widgets")
widgets_package.__path__ = [
    str(Path(__file__).resolve().parents[1] / "src" / "napari_neuron_navigator" / "widgets")
]
sys.modules["napari_neuron_navigator.widgets"] = widgets_package

NeuronViewerWidget = importlib.import_module(
    "napari_neuron_navigator.widgets.neuron_viewer"
).NeuronViewerWidget
SomaSliceProjector = importlib.import_module(
    "napari_neuron_navigator.widgets.slice_projection"
).SomaSliceProjector

for _name, _module in _ORIGINAL_MODULES.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module


class _DummySignal:
    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def disconnect(self, callback) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def emit(self, *args, **kwargs) -> None:
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


class _DummyDims:
    def __init__(
        self,
        ndisplay: int = 2,
        not_displayed: tuple[int, ...] = (0,),
        point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.ndisplay = ndisplay
        self.not_displayed = not_displayed
        self.point = point
        self.order = (0, 1, 2)
        self.events = types.SimpleNamespace(
            current_step=_DummySignal(),
            ndisplay=_DummySignal(),
            order=_DummySignal(),
        )


class _DummyBlocker:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _DummyLayerEvents:
    def __init__(self) -> None:
        self.highlight = _DummySignal()
        self.name = _DummySignal()

    def blocker_all(self) -> _DummyBlocker:
        return _DummyBlocker()


class _DummyPointsLayer:
    def __init__(self, data: np.ndarray, **kwargs) -> None:
        self.data = np.asarray(data, dtype=float)
        face_color = kwargs.get("face_color")
        self.face_color = (
            np.asarray(face_color, dtype=float) if face_color is not None else None
        )
        self.name = kwargs["name"]
        self.scale = kwargs.get("scale")
        self.metadata = kwargs.get("metadata", {})
        self.opacity = kwargs.get("opacity", 1.0)
        self.visible = kwargs.get("visible", True)
        self.size = kwargs.get("size")
        self.text = kwargs.get("text")
        self.mode = "pan_zoom"
        self.border_color = kwargs.get("border_color")
        self.border_width = kwargs.get("border_width")
        self.events = _DummyLayerEvents()
        self.selected_data: set[int] = set()
        self.refresh_count = 0

    def refresh(self) -> None:
        self.refresh_count += 1


class _DummyShapesLayer:
    def __init__(self, data: np.ndarray, **kwargs) -> None:
        self.data = np.asarray(data, dtype=float)
        self.name = kwargs["name"]
        self.scale = kwargs.get("scale")
        self.metadata = kwargs.get("metadata", {})
        self.opacity = kwargs.get("opacity", 1.0)
        self.visible = True
        self.edge_color = kwargs.get("edge_color")
        self.edge_width = kwargs.get("edge_width")


class _DummyImageLayer:
    def __init__(self, data: np.ndarray, **kwargs) -> None:
        self.data = np.asarray(data, dtype=float)
        self.name = kwargs["name"]
        self.metadata = kwargs.get("metadata", {})
        self.gamma = kwargs.get("gamma", 1.0)
        self.opacity = kwargs.get("opacity", 1.0)
        self.colormap = kwargs.get("colormap")
        self.blending = kwargs.get("blending")
        self.rendering = kwargs.get("rendering")
        self.contrast_limits = kwargs.get("contrast_limits")
        self.visible = kwargs.get("visible", True)
        self.events = _DummyLayerEvents()


class _DummyViewer:
    def __init__(
        self,
        ndisplay: int = 2,
        not_displayed: tuple[int, ...] = (0,),
        point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.layers: list[_DummyPointsLayer] = []
        self.dims = _DummyDims(
            ndisplay=ndisplay,
            not_displayed=not_displayed,
            point=point,
        )
        self.status = ""

    def add_points(self, data: np.ndarray, **kwargs) -> _DummyPointsLayer:
        layer = _DummyPointsLayer(data, **kwargs)
        self.layers.append(layer)
        return layer

    def add_shapes(self, data: np.ndarray, **kwargs) -> _DummyShapesLayer:
        layer = _DummyShapesLayer(data, **kwargs)
        self.layers.append(layer)
        return layer

    def add_image(self, data: np.ndarray, **kwargs) -> _DummyImageLayer:
        layer = _DummyImageLayer(data, **kwargs)
        self.layers.append(layer)
        return layer


class _DummyCheckBox:
    def __init__(self, checked: bool) -> None:
        self._checked = checked
        self.enabled = True
        self.signals_blocked = False

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        self._checked = checked

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def blockSignals(self, blocked: bool) -> bool:
        previous = self.signals_blocked
        self.signals_blocked = bool(blocked)
        return previous


class _DummyComboBox:
    def __init__(self, text: str, data=None) -> None:
        self._text = text
        self._data = text if data is None else data
        self.enabled = True

    def currentText(self) -> str:
        return self._text

    def currentData(self):
        return self._data

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


class _DummyMutableComboBox:
    def __init__(self) -> None:
        self.items: list[tuple[str, object]] = []
        self.current_index = -1
        self.enabled = True
        self.signals_blocked = False

    def addItem(self, text: str, data=None) -> None:
        self.items.append((text, data))
        if self.current_index < 0:
            self.current_index = 0

    def clear(self) -> None:
        self.items.clear()
        self.current_index = -1

    def count(self) -> int:
        return len(self.items)

    def setCurrentIndex(self, index: int) -> None:
        self.current_index = int(index)

    def currentData(self):
        if self.current_index < 0 or self.current_index >= len(self.items):
            return None
        return self.items[self.current_index][1]

    def currentText(self) -> str:
        if self.current_index < 0 or self.current_index >= len(self.items):
            return ""
        return self.items[self.current_index][0]

    def blockSignals(self, blocked: bool) -> bool:
        previous = self.signals_blocked
        self.signals_blocked = bool(blocked)
        return previous

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


class _DummyRegionSelector:
    def __init__(
        self,
        *,
        direct_acronyms: list[str] | None = None,
        query_acronyms: list[str] | None = None,
        direct_ids: list[int] | None = None,
        query_ids: list[int] | None = None,
        structure_map: dict[int, dict] | None = None,
        include_children: bool = True,
    ) -> None:
        self._direct_acronyms = list(direct_acronyms or [])
        self._query_acronyms = list(
            self._direct_acronyms if query_acronyms is None else query_acronyms
        )
        self._direct_ids = list(direct_ids or [])
        self._query_ids = list(self._direct_ids if query_ids is None else query_ids)
        self._structure_map = dict(structure_map or {})
        for struct_id, acronym in zip(self._direct_ids, self._direct_acronyms):
            self._structure_map.setdefault(int(struct_id), {"acronym": acronym})
        self._include_children = bool(include_children)

    def get_selected_acronyms(self, include_children: bool = True) -> list[str]:
        if include_children:
            return list(self._query_acronyms)
        return list(self._direct_acronyms)

    def get_selected_ids(self, include_children: bool = True) -> list[int]:
        if include_children:
            return list(self._query_ids)
        return list(self._direct_ids)

    def get_query_acronyms(self) -> list[str]:
        return list(self._query_acronyms)

    def include_children_enabled(self) -> bool:
        return self._include_children


class _DummyCustomRegionSelector:
    def __init__(
        self,
        region_ids: list[int] | None = None,
        *,
        region_groups: tuple[CustomRegionSelectionGroup, ...] | None = None,
        has_hierarchy: bool = True,
        unavailable_message: str = "Custom regions unavailable.",
    ) -> None:
        self._region_ids = list(region_ids or [])
        self._region_groups = tuple(region_groups or ())
        self._has_hierarchy = bool(has_hierarchy)
        self._unavailable_message = unavailable_message
        self.hierarchies: list[object] = []
        self.clear_messages: list[str] = []

    def get_selected_region_ids(self) -> list[int]:
        return list(self._region_ids)

    def get_selected_region_groups(
        self,
    ) -> tuple[CustomRegionSelectionGroup, ...]:
        return self._region_groups

    def has_hierarchy(self) -> bool:
        return self._has_hierarchy

    def unavailable_message(self) -> str:
        return self._unavailable_message

    def set_hierarchy(self, hierarchy: object) -> None:
        self.hierarchies.append(hierarchy)
        self._has_hierarchy = True

    def clear_with_message(self, message: str) -> None:
        self.clear_messages.append(message)
        self._has_hierarchy = False
        self._unavailable_message = message


class _DummyLabel:
    def __init__(self) -> None:
        self.visible = None
        self.text = ""

    def setVisible(self, value: bool) -> None:
        self.visible = value

    def setText(self, value: str) -> None:
        self.text = value


class _DummyButton:
    def __init__(self, text: str = "") -> None:
        self.enabled = True
        self.visible = True
        self.text = text

    def setEnabled(self, value: bool) -> None:
        self.enabled = value

    def setVisible(self, value: bool) -> None:
        self.visible = value

    def setText(self, value: str) -> None:
        self.text = value


class _DummyMenuAction:
    def __init__(self, text: str) -> None:
        self.text = text
        self.triggered = _DummySignal()


class _DummyMenu:
    def __init__(self, parent=None) -> None:
        self.parent = parent
        self.actions: list[_DummyMenuAction] = []

    def addAction(self, text: str) -> _DummyMenuAction:
        action = _DummyMenuAction(text)
        self.actions.append(action)
        return action


class _DummyMenuButton(_DummyButton):
    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.menu = None

    def setMenu(self, menu) -> None:
        self.menu = menu


class _DummyClusterFilterCombo:
    def __init__(self, selection: ClusterFilterSelection | None = None) -> None:
        self.selection = selection or ClusterFilterSelection()
        self.calls: list[dict[str, object]] = []

    def cluster_filter_selection(self) -> ClusterFilterSelection:
        return self.selection

    def set_cluster_options(
        self,
        cluster_ids: list[int],
        *,
        include_unclustered: bool,
        selection: ClusterFilterSelection,
    ) -> None:
        self.calls.append(
            {
                "cluster_ids": list(cluster_ids),
                "include_unclustered": include_unclustered,
                "selection": selection,
            }
        )
        available_ids = {int(cluster_id) for cluster_id in cluster_ids}
        if selection.is_all:
            self.selection = ClusterFilterSelection()
        else:
            self.selection = ClusterFilterSelection(
                selection.cluster_ids & available_ids,
                selection.include_unclustered and include_unclustered,
            )


class _DummyStack:
    def __init__(self) -> None:
        self.index = None

    def setCurrentIndex(self, index: int) -> None:
        self.index = index


class _DummyProgressBar:
    def __init__(self) -> None:
        self.visible = False
        self.range = (0, 0)
        self.value = 0

    def setRange(self, minimum: int, maximum: int) -> None:
        self.range = (minimum, maximum)

    def setValue(self, value: int) -> None:
        self.value = value

    def setVisible(self, value: bool) -> None:
        self.visible = value


class _DummyValueControl:
    def __init__(self, value) -> None:
        self._value = value

    def value(self):
        return self._value


class _DummyStatusLabel:
    def __init__(self) -> None:
        self.text = ""
        self.repaint_count = 0

    def setText(self, value: str) -> None:
        self.text = value

    def repaint(self) -> None:
        self.repaint_count += 1


def _make_soma_projector(
    viewer: _DummyViewer,
    *,
    tolerance: float = 5.0,
    point_size: int = 7,
) -> SomaSliceProjector:
    projector = SomaSliceProjector.__new__(SomaSliceProjector)
    projector._viewer = viewer
    projector._tolerance = tolerance
    projector._point_size = point_size
    projector._highlight_callback = None
    projector._source_data = {}
    projector._projection_layer = None
    projector._scale = None
    projector._enabled = True
    projector._connected = False
    projector._all_coords = None
    projector._all_colors = None
    projector._all_file_ids = None
    projector._axis_index = {}
    projector._last_result_key = None
    projector._last_result = None
    projector._update_timer = MagicMock()
    return projector


def test_widget_startup_schedules_cached_template_autoload_without_atlas_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup should only schedule the cached-background template path."""
    single_shot_calls = []

    class _TimerRecorder:
        @staticmethod
        def singleShot(*args, **kwargs) -> None:
            single_shot_calls.append((args, kwargs))

    monkeypatch.setitem(
        NeuronViewerWidget.__init__.__globals__,
        "QTimer",
        _TimerRecorder,
    )
    atlas_factory = MagicMock()
    monkeypatch.setitem(
        NeuronViewerWidget.__init__.__globals__,
        "BrainGlobeAtlas",
        atlas_factory,
    )
    monkeypatch.setattr(NeuronViewerWidget, "_setup_ui", lambda self: None)
    monkeypatch.setattr(
        NeuronViewerWidget,
        "_connect_layer_events",
        lambda self: None,
    )
    monkeypatch.setattr(
        NeuronViewerWidget,
        "_refresh_heatmap_layer_list",
        lambda self: None,
    )
    monkeypatch.setattr(
        NeuronViewerWidget,
        "_refresh_histogram_layer_list",
        lambda self: None,
    )
    monkeypatch.setattr(
        NeuronViewerWidget,
        "_refresh_mask_layer_options",
        lambda self: None,
    )

    widget = NeuronViewerWidget(_DummyViewer())

    assert len(single_shot_calls) == 1
    assert single_shot_calls[0][0][0] == 0
    assert single_shot_calls[0][0][1].__self__ is widget
    assert single_shot_calls[0][0][1].__name__ == "_start_cached_template_autoload"
    atlas_factory.assert_not_called()


class _SceneLayer:
    def __init__(
        self,
        name: str,
        *,
        flatmap: bool = False,
        visible: bool = True,
    ) -> None:
        self.name = name
        self.visible = visible
        self.metadata = (
            {"napari_neuron_navigator_space": "flatmap"} if flatmap else {"source": "main"}
        )


def _scene_layer(name: str, *, flatmap: bool = False, visible: bool = True):
    return _SceneLayer(name, flatmap=flatmap, visible=visible)


class _FlatmapQtWindow:
    def __init__(self) -> None:
        self.hidden = False
        self.fullscreen = False
        self.show_normal_calls = 0
        self.updates_enabled = True
        self.installed_filters = []
        self.removed_filters = []

    def installEventFilter(self, event_filter) -> None:
        self.installed_filters.append(event_filter)

    def removeEventFilter(self, event_filter) -> None:
        self.removed_filters.append(event_filter)

    def setUpdatesEnabled(self, enabled: bool) -> None:
        self.updates_enabled = bool(enabled)

    def hide(self) -> None:
        self.hidden = True

    def isFullScreen(self) -> bool:
        return self.fullscreen

    def showNormal(self) -> None:
        self.show_normal_calls += 1
        self.fullscreen = False


class _FlatmapWindow:
    def __init__(self) -> None:
        self.open = True
        self._qt_window = _FlatmapQtWindow()

    def geometry(self):
        if not self.open:
            raise RuntimeError("wrapped C/C++ object has been deleted")
        return (0, 0, 800, 600)


class _FlatmapViewer:
    def __init__(self, *, title: str, ndisplay: int, show: bool) -> None:
        self.title = title
        self.ndisplay = ndisplay
        self.initial_show = show
        self.layers = []
        self.window = _FlatmapWindow()
        self.show_calls = 0
        self.close_calls = 0

    def show(self) -> None:
        if not self.window.open:
            raise RuntimeError("viewer window is closed")
        self.show_calls += 1
        self.window._qt_window.hidden = False

    def close(self) -> None:
        self.close_calls += 1
        self.layers.clear()
        self.window.open = False


class _FlatmapTab:
    def __init__(self) -> None:
        self.released = []

    def _release_display_viewer(self, viewer) -> None:
        self.released.append(viewer)


def _flatmap_window_widget(main_viewer=None):
    widget = NeuronViewerWidget.__new__(NeuronViewerWidget)
    widget.viewer = main_viewer or types.SimpleNamespace(layers=[])
    widget._flatmap_viewers = []
    widget._flatmap_viewer = None
    widget._flatmap_viewer_pending_show = False
    widget._flatmap_viewer_generation = 0
    widget._flatmap_viewer_serial = 0
    widget._flatmap_hidden_viewer_ids = set()
    widget._flatmap_viewer_watch_timer = None
    widget._flatmap_close_guards = {}
    widget._flatmap_tab = _FlatmapTab()
    return widget


def test_flatmap_display_provider_creates_hidden_secondary_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_layer = _scene_layer("main")
    main_viewer = types.SimpleNamespace(layers=[main_layer])
    widget = _flatmap_window_widget(main_viewer)
    created = []

    def create_viewer(**kwargs):
        viewer = _FlatmapViewer(**kwargs)
        created.append(viewer)
        return viewer

    monkeypatch.setitem(
        sys.modules, "napari", types.SimpleNamespace(Viewer=create_viewer)
    )

    display = widget._get_or_create_flatmap_viewer(create=True)

    assert display is created[0]
    assert display is not main_viewer
    assert display.title == "Neuron Navigator Flatmap"
    assert display.ndisplay == 3
    assert display.initial_show is False
    assert display.show_calls == 0
    assert main_viewer.layers == [main_layer]
    assert widget._get_or_create_flatmap_viewer(create=False) is display
    assert widget._get_or_create_flatmap_viewer(create=True) is display
    assert len(created) == 1
    for obsolete_method in (
        "_cleanup_flatmap_viewer",
        "_connect_flatmap_viewer_destroyed",
        "_install_flatmap_fullscreen_close_guard",
        "_show_flatmap_viewer_window",
        "_stop_flatmap_status_thread",
    ):
        assert not hasattr(NeuronViewerWidget, obsolete_method)


def test_flatmap_display_provider_keeps_two_windows_for_side_by_side_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    widget = _flatmap_window_widget()
    created = []

    def create_viewer(**kwargs):
        viewer = _FlatmapViewer(**kwargs)
        created.append(viewer)
        return viewer

    monkeypatch.setitem(
        sys.modules, "napari", types.SimpleNamespace(Viewer=create_viewer)
    )

    first = widget._get_or_create_flatmap_viewer(create=True)
    first_layer = _scene_layer("first", flatmap=True)
    first.layers.append(first_layer)
    widget._on_flatmap_display_viewer_ready(first, first_layer)

    second = widget._create_new_flatmap_viewer()
    second_layer = _scene_layer("second", flatmap=True)
    second.layers.append(second_layer)
    widget._on_flatmap_display_viewer_ready(second, second_layer)

    assert created == [first, second]
    assert first.title == "Neuron Navigator Flatmap"
    assert second.title == "Neuron Navigator Flatmap 2"
    assert first.layers == [first_layer]
    assert second.layers == [second_layer]
    assert first.show_calls == 1
    assert second.show_calls == 1
    assert widget._flatmap_viewers == [first, second]
    assert widget._get_or_create_flatmap_viewer(create=False) is second


def test_closing_older_flatmap_window_does_not_change_active_window() -> None:
    widget = _flatmap_window_widget()
    first = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    second = _FlatmapViewer(
        title="Neuron Navigator Flatmap 2",
        ndisplay=3,
        show=False,
    )
    widget._flatmap_viewers = [first, second]
    widget._flatmap_viewer = second
    generation = widget._flatmap_viewer_generation
    first.window.open = False

    assert widget._retire_closed_flatmap_viewers() is True

    assert widget._flatmap_viewers == [second]
    assert widget._flatmap_viewer is second
    assert widget._flatmap_viewer_generation == generation
    assert widget._flatmap_tab.released == [first]


def test_flatmap_viewer_shows_only_after_marked_layer_is_ready() -> None:
    widget = _flatmap_window_widget()
    viewer = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    widget._flatmap_viewer = viewer
    widget._flatmap_viewer_pending_show = True
    layer = _scene_layer("flatmap", flatmap=True)
    viewer.layers.append(layer)

    widget._on_flatmap_display_viewer_ready(viewer, layer)

    assert viewer.show_calls == 1
    assert widget._flatmap_viewer_pending_show is False


def test_unmarked_first_layer_closes_hidden_flatmap_viewer() -> None:
    widget = _flatmap_window_widget()
    viewer = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    widget._flatmap_viewer = viewer
    widget._flatmap_viewer_pending_show = True
    layer = _scene_layer("not flatmap")
    viewer.layers.append(layer)

    widget._on_flatmap_display_viewer_ready(viewer, layer)

    assert viewer.show_calls == 0
    assert viewer.close_calls == 1
    assert widget._flatmap_viewer is None
    assert widget._flatmap_viewer_generation == 1


def test_failed_first_render_closes_window_but_failed_rerender_keeps_it() -> None:
    widget = _flatmap_window_widget()
    viewer = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    widget._flatmap_viewer = viewer
    widget._flatmap_viewer_pending_show = True

    widget._on_flatmap_display_viewer_failed(viewer, "first_render_failed")

    assert viewer.close_calls == 1
    assert widget._flatmap_viewer is None

    replacement = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    widget._flatmap_viewer = replacement
    widget._flatmap_viewer_pending_show = False

    widget._on_flatmap_display_viewer_failed(replacement, "rerender_failed")

    assert replacement.close_calls == 0
    assert widget._flatmap_viewer is replacement


def test_failed_new_comparison_window_preserves_previous_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    widget = _flatmap_window_widget()
    first = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    first_layer = _scene_layer("first", flatmap=True)
    first.layers.append(first_layer)
    widget._flatmap_viewers = [first]
    widget._flatmap_viewer = first
    created = []

    def create_viewer(**kwargs):
        viewer = _FlatmapViewer(**kwargs)
        created.append(viewer)
        return viewer

    monkeypatch.setitem(
        sys.modules, "napari", types.SimpleNamespace(Viewer=create_viewer)
    )

    failed = widget._create_new_flatmap_viewer()
    widget._on_flatmap_display_viewer_failed(failed, "projection_failed")

    assert failed.close_calls == 1
    assert first.layers == [first_layer]
    assert widget._flatmap_viewers == [first]
    assert widget._flatmap_viewer is first


def test_close_flatmap_viewer_uses_public_close_off_macos_and_invalidates_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        NeuronViewerWidget._close_flatmap_viewer.__globals__, "_IS_MACOS", False
    )
    widget = _flatmap_window_widget()
    viewer = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    widget._flatmap_viewer = viewer

    assert widget._close_flatmap_viewer() is True

    assert viewer.close_calls == 1
    assert widget._flatmap_viewer is None
    assert widget._flatmap_viewer_generation == 1
    assert widget._flatmap_tab.released == [viewer]


def test_macos_close_hides_clears_and_retains_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method_globals = NeuronViewerWidget._close_flatmap_viewer.__globals__
    monkeypatch.setitem(method_globals, "_IS_MACOS", True)
    widget = _flatmap_window_widget()
    viewer = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    widget._flatmap_viewer = viewer
    widget._flatmap_viewer_pending_show = False
    widget._install_flatmap_macos_close_guard(viewer)
    qt_window = viewer.window._qt_window
    guard = qt_window.installed_filters[0]
    viewer.layers.append(_scene_layer("flatmap", flatmap=True))

    assert widget._close_flatmap_viewer() is True

    assert qt_window.hidden is True
    assert qt_window.updates_enabled is True
    assert viewer.close_calls == 0
    assert viewer.layers == []
    assert widget._flatmap_viewer is viewer
    assert widget._flatmap_viewer_pending_show is True
    assert widget._flatmap_viewer_generation == 1
    assert widget._flatmap_tab.released == [viewer]
    assert qt_window.removed_filters == []
    assert widget._flatmap_close_guards == {id(viewer): (qt_window, guard)}


def test_macos_window_close_event_uses_the_same_hide_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CloseEvent:
        def __init__(self) -> None:
            self.ignored = False

        def type(self):
            return _FakeQEvent.Type.Close

        def ignore(self) -> None:
            self.ignored = True

    method_globals = NeuronViewerWidget._close_flatmap_viewer.__globals__
    monkeypatch.setitem(method_globals, "_IS_MACOS", True)
    widget = _flatmap_window_widget()
    viewer = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    widget._flatmap_viewer = viewer
    widget._flatmap_viewer_pending_show = False
    widget._install_flatmap_macos_close_guard(viewer)
    qt_window = viewer.window._qt_window
    event = _CloseEvent()

    consumed = qt_window.installed_filters[0].eventFilter(qt_window, event)

    assert consumed is True
    assert event.ignored is True
    assert qt_window.hidden is True
    assert viewer.close_calls == 0
    assert widget._flatmap_viewer is viewer
    assert widget._flatmap_viewer_pending_show is True


def test_macos_close_event_can_hide_an_older_comparison_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CloseEvent:
        def __init__(self) -> None:
            self.ignored = False

        def type(self):
            return _FakeQEvent.Type.Close

        def ignore(self) -> None:
            self.ignored = True

    method_globals = NeuronViewerWidget._close_flatmap_viewer.__globals__
    monkeypatch.setitem(method_globals, "_IS_MACOS", True)
    widget = _flatmap_window_widget()
    first = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    second = _FlatmapViewer(
        title="Neuron Navigator Flatmap 2",
        ndisplay=3,
        show=False,
    )
    first_layer = _scene_layer("first", flatmap=True)
    second_layer = _scene_layer("second", flatmap=True)
    first.layers.append(first_layer)
    second.layers.append(second_layer)
    widget._flatmap_viewers = [first, second]
    widget._flatmap_viewer = second
    widget._install_flatmap_macos_close_guard(first)
    event = _CloseEvent()

    consumed = first.window._qt_window.installed_filters[0].eventFilter(
        first.window._qt_window,
        event,
    )

    assert consumed is True
    assert event.ignored is True
    assert first.window._qt_window.hidden is True
    assert first.layers == []
    assert second.layers == [second_layer]
    assert widget._flatmap_viewer is second
    assert widget._flatmap_viewer_pending_show is False


def test_macos_fullscreen_close_exits_fullscreen_before_hiding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method_globals = NeuronViewerWidget._close_flatmap_viewer.__globals__
    monkeypatch.setitem(method_globals, "_IS_MACOS", True)
    widget = _flatmap_window_widget()
    viewer = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    widget._flatmap_viewer = viewer
    widget._flatmap_viewer_pending_show = False
    widget._install_flatmap_macos_close_guard(viewer)
    qt_window = viewer.window._qt_window
    qt_window.fullscreen = True

    assert widget._close_flatmap_viewer() is True

    assert qt_window.show_normal_calls == 1
    assert qt_window.fullscreen is False
    assert qt_window.hidden is True
    assert viewer.close_calls == 0
    assert widget._flatmap_viewer is viewer


def test_macos_widget_teardown_never_closes_a_visible_flatmap_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        NeuronViewerWidget._close_flatmap_viewer.__globals__, "_IS_MACOS", True
    )
    widget = _flatmap_window_widget()
    viewer = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    widget._flatmap_viewer = viewer
    widget._flatmap_viewer_pending_show = False
    widget._install_flatmap_macos_close_guard(viewer)
    qt_window = viewer.window._qt_window
    guard = qt_window.installed_filters[0]
    viewer.layers.append(_scene_layer("flatmap", flatmap=True))

    widget._on_neuron_viewer_destroyed()

    assert qt_window.hidden is True
    assert viewer.close_calls == 0
    assert viewer.layers == []
    assert widget._flatmap_viewer is None
    assert qt_window.removed_filters == [guard]
    assert widget._flatmap_close_guards == {}


def test_closed_window_is_retired_without_private_qt_state() -> None:
    widget = _flatmap_window_widget()
    viewer = _FlatmapViewer(
        title="Neuron Navigator Flatmap",
        ndisplay=3,
        show=False,
    )
    widget._flatmap_viewer = viewer
    viewer.window.open = False

    assert widget._flatmap_viewer_generation_value() == 1
    assert widget._flatmap_viewer is None
    assert viewer.close_calls == 0
    assert widget._get_or_create_flatmap_viewer(create=False) is None
    assert widget._flatmap_tab.released == [viewer]


def test_macos_flatmap_viewer_supports_three_hide_and_reuse_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        NeuronViewerWidget._close_flatmap_viewer.__globals__, "_IS_MACOS", True
    )
    widget = _flatmap_window_widget()
    created = []

    def create_viewer(**kwargs):
        viewer = _FlatmapViewer(**kwargs)
        created.append(viewer)
        return viewer

    monkeypatch.setitem(
        sys.modules, "napari", types.SimpleNamespace(Viewer=create_viewer)
    )

    for cycle in range(3):
        viewer = widget._get_or_create_flatmap_viewer(create=True)
        layer = _scene_layer(f"flatmap {cycle}", flatmap=True)
        viewer.layers.append(layer)
        widget._on_flatmap_display_viewer_ready(viewer, layer)
        assert widget._close_flatmap_viewer() is True

    assert len(created) == 1
    assert created[0].show_calls == 3
    assert created[0].close_calls == 0
    assert created[0].layers == []
    assert widget._flatmap_viewer is created[0]
    assert widget._flatmap_viewer_pending_show is True
    assert widget._flatmap_viewer_generation == 3


def test_reference_template_checkbox_defaults_to_lazy_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reference template should be opt-in, not loaded at startup."""

    class _Layout:
        def __init__(self, *_args, **_kwargs) -> None:
            self.items = []

        def addWidget(self, widget) -> None:
            self.items.append(widget)

        def addLayout(self, layout) -> None:
            self.items.append(layout)

        def addStretch(self) -> None:
            self.items.append("stretch")

    class _SignalStub:
        def __init__(self) -> None:
            self.callbacks = []

        def connect(self, callback) -> None:
            self.callbacks.append(callback)

    class _CheckBoxStub:
        def __init__(self, text: str = "") -> None:
            self.text = text
            self.checked = None
            self.tooltip = ""
            self.stateChanged = _SignalStub()

        def setChecked(self, checked: bool) -> None:
            self.checked = bool(checked)

        def isChecked(self) -> bool:
            return bool(self.checked)

        def setToolTip(self, text: str) -> None:
            self.tooltip = text

    class _SliderStub:
        def __init__(self, *_args, **_kwargs) -> None:
            self.valueChanged = _SignalStub()

        def setRange(self, *_args) -> None:
            return None

        def setValue(self, *_args) -> None:
            return None

    class _LabelStub:
        def __init__(self, text: str = "") -> None:
            self.text = text
            self.word_wrap = False

        def setWordWrap(self, enabled: bool) -> None:
            self.word_wrap = bool(enabled)

    globals_dict = NeuronViewerWidget._setup_reference_tab.__globals__
    monkeypatch.setitem(globals_dict, "QVBoxLayout", _Layout)
    monkeypatch.setitem(globals_dict, "QHBoxLayout", _Layout)
    monkeypatch.setitem(globals_dict, "QGroupBox", _FakeWidget)
    monkeypatch.setitem(globals_dict, "QLabel", _LabelStub)
    monkeypatch.setitem(globals_dict, "QCheckBox", _CheckBoxStub)
    monkeypatch.setitem(globals_dict, "QSlider", _SliderStub)

    widget = NeuronViewerWidget.__new__(NeuronViewerWidget)
    widget._toggle_template = lambda _state: None
    widget._update_template_opacity = lambda _value: None
    widget._toggle_outline = lambda _state: None
    widget._toggle_region_meshes = lambda _state: None
    widget._toggle_region_segmentation = lambda _state: None
    widget._update_seg_opacity = lambda _value: None

    NeuronViewerWidget._setup_reference_tab(widget, _FakeWidget())

    assert widget._show_template_cb.isChecked() is False
    assert "on demand" in widget._show_template_cb.tooltip
    assert "top-level parent regions" in widget._region_mesh_scope_label.text
    assert "higher computational cost" in widget._region_mesh_scope_label.text
    assert widget._region_mesh_scope_label.word_wrap is True


def test_region_appearance_is_a_collapsed_bottom_section() -> None:
    source = inspect.getsource(NeuronViewerWidget._setup_regions_tab)

    assert 'CollapsibleSection(\n            "Region Appearance"' in source
    assert "expanded=False" in source
    status_position = source.index("layout.addWidget(self._regions_status_label)")
    appearance_position = source.index(
        "layout.addWidget(self._region_appearance_section)"
    )
    stretch_position = source.rindex("layout.addStretch()")
    assert status_position < appearance_position < stretch_position


def test_data_tab_offers_all_supported_atlas_resolutions() -> None:
    """The Data-tab atlas selector should include the 100 um atlas."""
    atlas_options = NeuronViewerWidget._setup_data_tab.__globals__["_ATLAS_OPTIONS"]

    assert atlas_options == (
        "allen_mouse_10um",
        "allen_mouse_25um",
        "allen_mouse_50um",
        "allen_mouse_100um",
    )


def test_selected_neurons_starts_with_adaptive_height_toggle() -> None:
    source = inspect.getsource(NeuronViewerWidget._setup_data_tab)

    toggle_position = source.index('QCheckBox("Adaptive table height")')
    table_position = source.index("neurons_layout.addWidget(self._neuron_table)")

    assert toggle_position < table_position
    assert "self._neuron_table.set_adaptive_height_enabled" in source


def test_load_atlas_starts_background_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 100 um atlas load should start a background worker and busy UI."""
    workers = []
    threads = []

    class _FakeThread:
        def __init__(self) -> None:
            self.started = _DummyAnalysisSignal()
            self.finished = _DummyAnalysisSignal()
            self.started_called = False
            self.running = False
            threads.append(self)

        def start(self) -> None:
            self.started_called = True
            self.running = True

        def quit(self) -> None:
            self.running = False

        def isRunning(self) -> bool:
            return self.running

        def deleteLater(self) -> None:
            return None

    class _FakeAtlasLoadWorker:
        def __init__(self, atlas_name: str) -> None:
            self.atlas_name = atlas_name
            self.status = _DummyAnalysisSignal()
            self.progress = _DummyAnalysisSignal()
            self.finished = _DummyAnalysisSignal()
            self.error = _DummyAnalysisSignal()
            self.thread = None
            workers.append(self)

        def moveToThread(self, thread) -> None:
            self.thread = thread

        def run(self) -> None:
            return None

        def deleteLater(self) -> None:
            return None

    fake_workers = types.ModuleType("napari_neuron_navigator.workers")
    fake_workers.AtlasLoadWorker = _FakeAtlasLoadWorker
    monkeypatch.setitem(sys.modules, "napari_neuron_navigator.workers", fake_workers)
    monkeypatch.setitem(
        NeuronViewerWidget._load_atlas.__globals__,
        "QThread",
        _FakeThread,
    )

    widget = NeuronViewerWidget.__new__(NeuronViewerWidget)
    widget._atlas_combo = _DummyComboBox("allen_mouse_100um")
    widget._load_atlas_btn = _DummyButton("Load Atlas")
    widget._atlas_status_label = _DummyStatusLabel()
    widget._atlas_progress = _DummyProgressBar()
    widget._cached_atlas_thread = None
    widget._atlas_load_thread = None
    widget._atlas_load_worker = None
    widget._pending_reference_action = None

    NeuronViewerWidget._load_atlas(widget)

    assert len(workers) == 1
    assert workers[0].atlas_name == "allen_mouse_100um"
    assert workers[0].thread is threads[0]
    assert threads[0].started_called is True
    assert widget._atlas_load_worker is workers[0]
    assert widget._atlas_load_thread is threads[0]
    assert widget._atlas_combo.enabled is False
    assert widget._load_atlas_btn.enabled is False
    assert widget._atlas_progress.visible is True
    assert widget._atlas_progress.range == (0, 0)
    assert (
        widget._atlas_status_label.text
        == "Atlas: Preparing to load allen_mouse_100um..."
    )


def test_atlas_load_progress_updates_progress_bar() -> None:
    """Atlas worker progress should update the Data-tab progress bar."""
    widget = NeuronViewerWidget.__new__(NeuronViewerWidget)
    widget._atlas_progress = _DummyProgressBar()

    NeuronViewerWidget._on_atlas_load_progress(widget, 0, 100, 42)

    assert widget._atlas_progress.visible is True
    assert widget._atlas_progress.range == (0, 100)
    assert widget._atlas_progress.value == 42


def test_atlas_load_finished_applies_atlas_and_prompts_reference_tab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual atlas loads should apply the atlas and show the Reference prompt."""
    atlas = types.SimpleNamespace(
        atlas_name="allen_mouse_25um",
        structures={1: {"acronym": "R1"}},
    )
    prompts = []
    widget = NeuronViewerWidget.__new__(NeuronViewerWidget)
    widget._atlas_combo = _DummyComboBox("allen_mouse_25um")
    widget._load_atlas_btn = _DummyButton("Load Atlas")
    widget._atlas_status_label = _DummyStatusLabel()
    widget._atlas_progress = _DummyProgressBar()
    widget._atlas_progress.setVisible(True)
    widget._pending_reference_action = None
    widget._apply_loaded_atlas = MagicMock()

    monkeypatch.setitem(
        NeuronViewerWidget._on_atlas_load_finished.__globals__,
        "show_info",
        prompts.append,
    )

    NeuronViewerWidget._on_atlas_load_finished(widget, atlas)

    widget._apply_loaded_atlas.assert_called_once_with(atlas, "allen_mouse_25um")
    assert widget._atlas_combo.enabled is True
    assert widget._load_atlas_btn.enabled is True
    assert widget._atlas_progress.visible is False
    assert "Reference tab" in widget._atlas_status_label.text
    assert prompts == [
        "Atlas loaded. Go to the Reference tab to show the template, outline, "
        "or selected region meshes."
    ]


def test_atlas_load_finished_completes_pending_reference_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference-tab initiated loads should complete the requested action."""
    atlas = types.SimpleNamespace(
        atlas_name="allen_mouse_25um",
        structures={1: {"acronym": "R1"}},
    )
    prompts = []
    actions = []
    widget = NeuronViewerWidget.__new__(NeuronViewerWidget)
    widget._atlas_combo = _DummyComboBox("allen_mouse_25um")
    widget._load_atlas_btn = _DummyButton("Load Atlas")
    widget._atlas_status_label = _DummyStatusLabel()
    widget._atlas_progress = _DummyProgressBar()
    widget._pending_reference_action = "template"
    widget._show_template_cb = _DummyCheckBox(True)
    widget._apply_loaded_atlas = MagicMock()
    widget._toggle_template = lambda state: actions.append(("template", state))

    monkeypatch.setitem(
        NeuronViewerWidget._on_atlas_load_finished.__globals__,
        "show_info",
        prompts.append,
    )

    NeuronViewerWidget._on_atlas_load_finished(widget, atlas)

    widget._apply_loaded_atlas.assert_called_once_with(atlas, "allen_mouse_25um")
    assert widget._pending_reference_action is None
    assert actions == [("template", True)]
    assert prompts == []


def test_toggle_template_loads_atlas_on_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checking the template box should queue an atlas load when needed."""
    load_calls = []
    template_calls = []

    widget = NeuronViewerWidget.__new__(NeuronViewerWidget)
    widget._atlas = None
    widget._cached_atlas_thread = None
    widget._atlas_load_thread = None
    widget._pending_reference_action = None
    widget.viewer = _DummyViewer()
    widget._show_template_cb = _DummyCheckBox(True)
    widget._template_opacity_slider = _DummyValueControl(30)

    def _load_atlas(*, pending_reference_action=None) -> None:
        load_calls.append(pending_reference_action)

    widget._load_atlas = _load_atlas

    def _add_template(viewer, loaded_atlas, **kwargs) -> None:
        template_calls.append((viewer, loaded_atlas, kwargs))

    monkeypatch.setitem(
        NeuronViewerWidget._toggle_template.__globals__,
        "add_allen_template",
        _add_template,
    )

    NeuronViewerWidget._toggle_template(widget, True)

    assert load_calls == ["template"]
    assert template_calls == []


def test_toggle_template_hide_without_atlas_does_not_load() -> None:
    """Turning the template off before atlas load should not load an atlas."""
    widget = NeuronViewerWidget.__new__(NeuronViewerWidget)
    widget._atlas = None
    widget._cached_atlas_thread = None
    widget._show_template_after_cached_atlas_load = True
    widget._load_atlas = MagicMock()
    widget._show_template_cb = _DummyCheckBox(False)

    NeuronViewerWidget._toggle_template(widget, False)

    widget._load_atlas.assert_not_called()
    assert widget._show_template_after_cached_atlas_load is False


def test_toggle_template_waits_for_cached_autoload() -> None:
    """Manual template-on requests should not start a second atlas load."""
    widget = NeuronViewerWidget.__new__(NeuronViewerWidget)
    widget._atlas = None
    widget._cached_atlas_thread = types.SimpleNamespace(isRunning=lambda: True)
    widget._show_template_after_cached_atlas_load = False
    widget._load_atlas = MagicMock()
    widget._show_template_cb = _DummyCheckBox(True)

    NeuronViewerWidget._toggle_template(widget, True)

    widget._load_atlas.assert_not_called()
    assert widget._show_template_after_cached_atlas_load is True


def test_cached_template_atlas_loaded_applies_atlas_and_shows_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finished background loads should refresh atlas UI and show the template."""
    atlas = types.SimpleNamespace(
        atlas_name="fake_atlas",
        structures={1: {"acronym": "R1"}},
    )
    template_calls = []
    widget = NeuronViewerWidget.__new__(NeuronViewerWidget)
    widget._atlas = None
    widget._atlas_combo = _DummyComboBox("fake_atlas")
    widget._show_template_cb = _DummyCheckBox(True)
    widget._show_template_after_cached_atlas_load = True
    widget._template_opacity_slider = _DummyValueControl(30)
    widget.viewer = _DummyViewer()
    widget._apply_loaded_atlas = MagicMock(
        side_effect=lambda loaded, _name: setattr(widget, "_atlas", loaded)
    )

    def _add_template(viewer, loaded_atlas, **kwargs) -> None:
        template_calls.append((viewer, loaded_atlas, kwargs))

    monkeypatch.setitem(
        NeuronViewerWidget._toggle_template.__globals__,
        "add_allen_template",
        _add_template,
    )

    NeuronViewerWidget._on_cached_template_atlas_loaded(widget, atlas)

    widget._apply_loaded_atlas.assert_called_once_with(atlas, "fake_atlas")
    assert template_calls == [(widget.viewer, atlas, {"opacity": 0.3})]
    assert widget._show_template_cb.isChecked() is True
    assert widget._show_template_cb.enabled is True


def _install_fake_colormaps(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyColormap:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_napari = types.ModuleType("napari")
    fake_utils = types.ModuleType("napari.utils")
    fake_utils.__path__ = []
    fake_colormaps = types.ModuleType("napari.utils.colormaps")
    fake_colormaps.Colormap = _DummyColormap
    fake_napari.utils = fake_utils
    fake_utils.colormaps = fake_colormaps

    monkeypatch.setitem(sys.modules, "napari", fake_napari)
    monkeypatch.setitem(sys.modules, "napari.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "napari.utils.colormaps", fake_colormaps)


def _bind_region_isolation_methods(widget) -> None:
    for method_name in (
        "_current_region_isolation_create_mode",
        "_selected_region_isolation_entries",
        "_selected_region_isolation_region_ids",
        "_region_isolation_label",
        "_add_region_isolated_heatmap_layer",
        "_source_file_ids_for_layers",
        "_unique_layer_name",
        "_iter_viewer_layers",
        "_current_atlas_name",
    ):
        setattr(
            widget,
            method_name,
            types.MethodType(getattr(NeuronViewerWidget, method_name), widget),
        )
    widget._normalise_layer_file_ids = NeuronViewerWidget._normalise_layer_file_ids


def _bind_projection_helpers(widget) -> None:
    widget._soma_projection_active_in_2d = types.MethodType(
        NeuronViewerWidget._soma_projection_active_in_2d,
        widget,
    )
    widget._set_neuron_points_soma_visibility = types.MethodType(
        NeuronViewerWidget._set_neuron_points_soma_visibility,
        widget,
    )
    widget._sync_soma_projection_overlay_state = types.MethodType(
        NeuronViewerWidget._sync_soma_projection_overlay_state,
        widget,
    )


def _bind_scene_helpers(widget) -> None:
    widget._current_scene_file_ids = types.MethodType(
        NeuronViewerWidget._current_scene_file_ids,
        widget,
    )
    widget._base_display_state_for_file_id = types.MethodType(
        NeuronViewerWidget._base_display_state_for_file_id,
        widget,
    )
    widget._build_effective_color_map = types.MethodType(
        NeuronViewerWidget._build_effective_color_map,
        widget,
    )
    widget._build_soma_projection_batch = types.MethodType(
        NeuronViewerWidget._build_soma_projection_batch,
        widget,
    )
    widget._clear_neuron_layers = types.MethodType(
        NeuronViewerWidget._clear_neuron_layers,
        widget,
    )
    widget._render_selected_with_mode = types.MethodType(
        NeuronViewerWidget._render_selected_with_mode,
        widget,
    )


def _bind_table_membership_helpers(widget) -> None:
    widget._current_scene_file_ids = types.MethodType(
        NeuronViewerWidget._current_scene_file_ids,
        widget,
    )
    widget._current_table_file_ids = types.MethodType(
        NeuronViewerWidget._current_table_file_ids,
        widget,
    )
    widget._current_table_file_ids_in_scene = types.MethodType(
        NeuronViewerWidget._current_table_file_ids_in_scene,
        widget,
    )
    widget._base_display_state_for_file_id = types.MethodType(
        NeuronViewerWidget._base_display_state_for_file_id,
        widget,
    )
    widget._cache_scene_display_state = types.MethodType(
        NeuronViewerWidget._cache_scene_display_state,
        widget,
    )
    widget._discard_scene_display_state = types.MethodType(
        NeuronViewerWidget._discard_scene_display_state,
        widget,
    )
    widget._build_effective_color_map = types.MethodType(
        NeuronViewerWidget._build_effective_color_map,
        widget,
    )
    widget._sync_after_neuron_table_membership_change = types.MethodType(
        NeuronViewerWidget._sync_after_neuron_table_membership_change,
        widget,
    )


def _bind_manual_heatmap_helpers(widget) -> None:
    widget._iter_viewer_layers = types.MethodType(
        NeuronViewerWidget._iter_viewer_layers,
        widget,
    )
    widget._manual_heatmap_layers = types.MethodType(
        NeuronViewerWidget._manual_heatmap_layers,
        widget,
    )
    widget._normalise_layer_file_ids = NeuronViewerWidget._normalise_layer_file_ids
    widget._manual_heatmap_combo_options = types.MethodType(
        NeuronViewerWidget._manual_heatmap_combo_options,
        widget,
    )
    widget._current_selected_neuron_heatmap_layers_by_file_id = types.MethodType(
        NeuronViewerWidget._current_selected_neuron_heatmap_layers_by_file_id,
        widget,
    )
    widget._sync_neuron_table_heatmap_membership = types.MethodType(
        NeuronViewerWidget._sync_neuron_table_heatmap_membership,
        widget,
    )
    widget._manual_heatmap_combo_data = types.MethodType(
        NeuronViewerWidget._manual_heatmap_combo_data,
        widget,
    )
    widget._manual_heatmap_combo_key = types.MethodType(
        NeuronViewerWidget._manual_heatmap_combo_key,
        widget,
    )
    widget._selected_manual_heatmap_file_ids = types.MethodType(
        NeuronViewerWidget._selected_manual_heatmap_file_ids,
        widget,
    )
    widget._selected_cluster_filter = types.MethodType(
        NeuronViewerWidget._selected_cluster_filter,
        widget,
    )
    widget._apply_neuron_table_filters = types.MethodType(
        NeuronViewerWidget._apply_neuron_table_filters,
        widget,
    )
    widget._refresh_manual_heatmap_combo = types.MethodType(
        NeuronViewerWidget._refresh_manual_heatmap_combo,
        widget,
    )
    widget._on_manual_heatmap_selection_changed = types.MethodType(
        NeuronViewerWidget._on_manual_heatmap_selection_changed,
        widget,
    )


def _bind_layer_name_event_helpers(widget) -> None:
    widget._sync_layer_name_event_connections = types.MethodType(
        NeuronViewerWidget._sync_layer_name_event_connections,
        widget,
    )
    widget._disconnect_stale_layer_name_event_connections = types.MethodType(
        NeuronViewerWidget._disconnect_stale_layer_name_event_connections,
        widget,
    )
    widget._disconnect_layer_name_event_connection = types.MethodType(
        NeuronViewerWidget._disconnect_layer_name_event_connection,
        widget,
    )
    widget._on_viewer_layer_name_changed = types.MethodType(
        NeuronViewerWidget._on_viewer_layer_name_changed,
        widget,
    )
    widget._on_viewer_layers_changed = types.MethodType(
        NeuronViewerWidget._on_viewer_layers_changed,
        widget,
    )


def _bind_cluster_filter_helpers(widget) -> None:
    widget._normalise_layer_file_ids = NeuronViewerWidget._normalise_layer_file_ids
    widget._selected_cluster_filter = types.MethodType(
        NeuronViewerWidget._selected_cluster_filter,
        widget,
    )
    widget._manual_heatmap_combo_data = types.MethodType(
        NeuronViewerWidget._manual_heatmap_combo_data,
        widget,
    )
    widget._selected_manual_heatmap_file_ids = types.MethodType(
        NeuronViewerWidget._selected_manual_heatmap_file_ids,
        widget,
    )
    widget._apply_neuron_table_filters = types.MethodType(
        NeuronViewerWidget._apply_neuron_table_filters,
        widget,
    )
    widget._on_cluster_filter_changed = types.MethodType(
        NeuronViewerWidget._on_cluster_filter_changed,
        widget,
    )


def _bind_region_query_scope_helpers(widget) -> None:
    widget._normalise_layer_file_ids = NeuronViewerWidget._normalise_layer_file_ids
    widget._source_file_ids_for_layers = types.MethodType(
        NeuronViewerWidget._source_file_ids_for_layers,
        widget,
    )
    widget._mask_source_exclusion_enabled = types.MethodType(
        NeuronViewerWidget._mask_source_exclusion_enabled,
        widget,
    )
    widget._selected_region_query_scope = types.MethodType(
        NeuronViewerWidget._selected_region_query_scope,
        widget,
    )
    widget._current_table_file_ids = types.MethodType(
        NeuronViewerWidget._current_table_file_ids,
        widget,
    )
    widget._resolve_region_query_file_scope = types.MethodType(
        NeuronViewerWidget._resolve_region_query_file_scope,
        widget,
    )
    widget._query_scope_status_suffix = NeuronViewerWidget._query_scope_status_suffix
    widget._effective_query_node_types = NeuronViewerWidget._effective_query_node_types
    widget._region_selector_for_scope = types.MethodType(
        NeuronViewerWidget._region_selector_for_scope,
        widget,
    )
    widget._active_region_selector = types.MethodType(
        NeuronViewerWidget._active_region_selector,
        widget,
    )
    widget._custom_region_selector_for_scope = types.MethodType(
        NeuronViewerWidget._custom_region_selector_for_scope,
        widget,
    )
    widget._active_custom_region_selector = types.MethodType(
        NeuronViewerWidget._active_custom_region_selector,
        widget,
    )
    widget._active_custom_region_groups = types.MethodType(
        NeuronViewerWidget._active_custom_region_groups,
        widget,
    )
    widget._active_flatmap_region_ids = types.MethodType(
        NeuronViewerWidget._active_flatmap_region_ids,
        widget,
    )
    widget._active_flatmap_parent_region_ids = types.MethodType(
        NeuronViewerWidget._active_flatmap_parent_region_ids,
        widget,
    )
    widget._active_flatmap_region_acronyms = types.MethodType(
        NeuronViewerWidget._active_flatmap_region_acronyms,
        widget,
    )
    widget._active_flatmap_region_source = types.MethodType(
        NeuronViewerWidget._active_flatmap_region_source,
        widget,
    )
    widget._active_flatmap_region_scope = types.MethodType(
        NeuronViewerWidget._active_flatmap_region_scope,
        widget,
    )
    widget._active_flatmap_region_error = types.MethodType(
        NeuronViewerWidget._active_flatmap_region_error,
        widget,
    )
    widget._active_region_preview_acronyms = types.MethodType(
        NeuronViewerWidget._active_region_preview_acronyms,
        widget,
    )
    widget._sync_region_query_scope_selector = types.MethodType(
        NeuronViewerWidget._sync_region_query_scope_selector,
        widget,
    )
    widget._sync_active_region_reference_layers = types.MethodType(
        NeuronViewerWidget._sync_active_region_reference_layers,
        widget,
    )
    widget._sync_active_region_meshes = types.MethodType(
        NeuronViewerWidget._sync_active_region_meshes,
        widget,
    )
    widget._sync_active_region_segmentation = types.MethodType(
        NeuronViewerWidget._sync_active_region_segmentation,
        widget,
    )


@pytest.mark.parametrize("slice_axis", [0, 1, 2])
def test_soma_slice_projector_flattens_points_onto_active_slice(
    slice_axis: int,
) -> None:
    viewer = _DummyViewer()
    projector = _make_soma_projector(viewer, tolerance=1.5)
    coords_a = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
    coords_b = np.array([[40.0, 50.0, 60.0]])

    projector.add_soma_data_batch(
        {
            "neuron-a": (coords_a, (1.0, 0.0, 0.0, 1.0)),
            "neuron-b": (coords_b, (0.0, 1.0, 0.0, 1.0)),
        }
    )

    slice_position = float(coords_a[0, slice_axis])
    points, colors, file_ids = projector._compute_slice_projection(
        slice_position,
        slice_axis,
    )

    assert points is not None
    assert colors is not None
    assert file_ids == ["neuron-a", "neuron-a"]
    np.testing.assert_allclose(points[:, slice_axis], slice_position)
    kept_axes = [axis for axis in range(3) if axis != slice_axis]
    np.testing.assert_allclose(points[:, kept_axes], coords_a[:, kept_axes])


def test_soma_slice_projector_returns_none_when_slice_has_no_hits() -> None:
    viewer = _DummyViewer()
    projector = _make_soma_projector(viewer, tolerance=0.5)
    projector.add_soma_data("neuron-a", np.array([[10.0, 20.0, 30.0]]))

    points, colors, file_ids = projector._compute_slice_projection(99.0, 0)

    assert points is None
    assert colors is None
    assert file_ids is None


def test_soma_slice_projector_updates_colors_without_rebuilding_geometry() -> None:
    viewer = _DummyViewer()
    projector = _make_soma_projector(viewer)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    projector.add_soma_data("neuron-a", coords, (0.0, 1.0, 1.0, 1.0))

    projector.update_neuron_colors({"neuron-a": [1.0, 0.0, 0.0, 1.0]})

    assert projector._source_data["neuron-a"][1] == (1.0, 0.0, 0.0, 1.0)
    np.testing.assert_array_almost_equal(
        projector._all_colors,
        np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]),
    )


def test_soma_slice_projector_updates_layer_and_clear_removes_it() -> None:
    viewer = _DummyViewer(
        ndisplay=2,
        not_displayed=(0,),
        point=(10.0, 0.0, 0.0),
    )
    projector = _make_soma_projector(viewer, tolerance=1.0, point_size=9)
    projector.add_soma_data("neuron-a", np.array([[10.0, 5.0, 5.0]]))

    projector._do_update_projection()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "Soma Slice Projection"
    assert layer.metadata["file_ids"] == ["neuron-a"]
    assert layer.size == 9
    assert layer.border_color == "#39ff14"
    assert layer.border_width == 0.15
    assert layer.text == {"visible": False}
    assert layer.visible is True

    projector.clear()

    assert viewer.layers == []
    assert projector._projection_layer is None


def test_soma_slice_projector_recreates_hidden_layer_in_3d() -> None:
    viewer = _DummyViewer(
        ndisplay=3,
        not_displayed=(),
        point=(0.0, 0.0, 0.0),
    )
    projector = _make_soma_projector(viewer, tolerance=1.0, point_size=9)
    projector.add_soma_data("neuron-a", np.array([[10.0, 5.0, 5.0]]))

    projector._do_update_projection()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "Soma Slice Projection"
    assert layer.metadata["file_ids"] == []
    assert layer.visible is False
    assert layer.data.shape == (0, 3)
    np.testing.assert_allclose(layer.face_color, np.array([1.0, 0.0, 0.0, 1.0]))


def test_soma_slice_projector_keeps_empty_layer_when_2d_slice_has_no_hits() -> None:
    viewer = _DummyViewer(
        ndisplay=2,
        not_displayed=(0,),
        point=(0.0, 0.0, 0.0),
    )
    projector = _make_soma_projector(viewer, tolerance=1.0, point_size=9)
    projector.add_soma_data("neuron-a", np.array([[10.0, 5.0, 5.0]]))

    projector._do_update_projection()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "Soma Slice Projection"
    assert layer.metadata["file_ids"] == []
    assert layer.visible is True
    assert layer.data.shape == (0, 3)
    np.testing.assert_allclose(layer.face_color, np.array([1.0, 0.0, 0.0, 1.0]))


def test_build_soma_projection_batch_uses_rendered_soma_points_when_available() -> None:
    widget = types.SimpleNamespace(_db=MagicMock())
    points_df = pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n1"],
            "type": [1, 2, 1],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "z": [7.0, 8.0, 9.0],
        }
    )

    batch = NeuronViewerWidget._build_soma_projection_batch(
        widget,
        file_ids=["n1"],
        neuron_colors=[[0.1, 0.2, 0.3, 0.4]],
        points_df=points_df,
    )

    widget._db.get_soma_points.assert_not_called()
    coords, color = batch["n1"]
    np.testing.assert_allclose(coords, np.array([[1.0, 4.0, 7.0], [3.0, 6.0, 9.0]]))
    assert color == (0.1, 0.2, 0.3, 0.4)


def test_build_soma_projection_batch_queries_db_for_lines_only_rendering() -> None:
    soma_df = pd.DataFrame(
        {
            "file_id": ["n1", "n2"],
            "neuron_id": ["N1", "N2"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
        }
    )
    widget = types.SimpleNamespace(_db=MagicMock())
    widget._db.get_soma_points.return_value = soma_df

    batch = NeuronViewerWidget._build_soma_projection_batch(
        widget,
        file_ids=["n1", "n2"],
        neuron_colors=[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
        points_df=None,
    )

    widget._db.get_soma_points.assert_called_once_with(["n1", "n2"])
    assert set(batch) == {"n1", "n2"}
    np.testing.assert_allclose(batch["n1"][0], np.array([[1.0, 3.0, 5.0]]))
    assert batch["n2"][1] == (0.0, 1.0, 0.0, 1.0)


def test_build_soma_projection_batch_queries_only_missing_soma_only_ids() -> None:
    points_df = pd.DataFrame(
        {
            "file_id": ["n1", "n1"],
            "type": [1, 2],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
        }
    )
    soma_df = pd.DataFrame(
        {
            "file_id": ["n2"],
            "neuron_id": ["N2"],
            "x": [7.0],
            "y": [8.0],
            "z": [9.0],
        }
    )
    widget = types.SimpleNamespace(_db=MagicMock())
    widget._db.get_soma_points.return_value = soma_df

    batch = NeuronViewerWidget._build_soma_projection_batch(
        widget,
        file_ids=["n1", "n2"],
        neuron_colors=[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
        points_df=points_df,
    )

    widget._db.get_soma_points.assert_called_once_with(["n2"])
    np.testing.assert_allclose(batch["n1"][0], np.array([[1.0, 3.0, 5.0]]))
    np.testing.assert_allclose(batch["n2"][0], np.array([[7.0, 8.0, 9.0]]))
    assert batch["n2"][1] == (0.0, 1.0, 0.0, 1.0)


def test_render_selected_soma_only_rebuilds_scene_with_soma_mode() -> None:
    widget = types.SimpleNamespace(
        _scene_render_modes={"n1": "full"},
        _neuron_table=MagicMock(),
        _render_status_label=_DummyLabel(),
        _render_scene=MagicMock(),
    )
    widget._neuron_table.get_selected_file_ids.return_value = ["n1", "n2"]
    _bind_scene_helpers(widget)

    NeuronViewerWidget._render_selected_soma_only(widget)

    assert widget._scene_render_modes == {"n1": "soma", "n2": "soma"}
    widget._render_scene.assert_called_once_with()


def test_render_selected_neurons_switches_soma_only_back_to_full() -> None:
    widget = types.SimpleNamespace(
        _scene_render_modes={"n1": "soma"},
        _neuron_table=MagicMock(),
        _render_status_label=_DummyLabel(),
        _render_scene=MagicMock(),
    )
    widget._neuron_table.get_selected_file_ids.return_value = ["n1"]
    _bind_scene_helpers(widget)

    NeuronViewerWidget._render_selected_neurons(widget)

    assert widget._scene_render_modes == {"n1": "full"}
    widget._render_scene.assert_called_once_with()


def test_remove_selected_neurons_preserves_remaining_render_modes() -> None:
    widget = types.SimpleNamespace(
        _scene_render_modes={"n1": "full", "n2": "soma"},
        _neuron_table=MagicMock(),
        _render_status_label=_DummyLabel(),
        _render_scene=MagicMock(),
        _capture_depth_state=MagicMock(return_value=None),
        _restore_depth_state=MagicMock(),
        _clear_neuron_layers=MagicMock(),
    )
    widget._neuron_table.get_selected_file_ids.return_value = ["n1"]
    widget._current_scene_file_ids = types.MethodType(
        NeuronViewerWidget._current_scene_file_ids,
        widget,
    )

    NeuronViewerWidget._remove_selected_neurons(widget)

    assert widget._scene_render_modes == {"n2": "soma"}
    widget._render_scene.assert_called_once_with()
    widget._clear_neuron_layers.assert_not_called()


def test_clear_all_neuron_layers_ignores_qt_checked_argument_and_resets_scene_state() -> (
    None
):
    viewer = _DummyViewer(ndisplay=3)
    soma_layer = _DummyPointsLayer(
        np.array([[0.0, 0.0, 0.0]]),
        name="Soma Labels",
        metadata={"file_ids": ["n1"]},
    )
    viewer.layers.append(soma_layer)
    widget = types.SimpleNamespace(
        viewer=viewer,
        _current_neuron_layers=[soma_layer],
        _scene_render_modes={"n1": "soma"},
        _scene_display_state={"n1": {"color": [1.0, 0.0, 0.0, 1.0], "visible": True}},
        _slice_projector=MagicMock(),
        _soma_slice_projector=MagicMock(),
        _neuron_table=types.SimpleNamespace(set_added_file_ids=MagicMock()),
    )
    widget._current_scene_file_ids = types.MethodType(
        NeuronViewerWidget._current_scene_file_ids,
        widget,
    )
    widget._clear_neuron_layers = types.MethodType(
        NeuronViewerWidget._clear_neuron_layers,
        widget,
    )

    NeuronViewerWidget._clear_all_neuron_layers(widget, False)

    assert viewer.layers == []
    assert widget._current_neuron_layers == []
    assert widget._scene_render_modes == {}
    assert widget._scene_display_state == {}
    widget._slice_projector.clear.assert_called_once_with()
    widget._soma_slice_projector.clear.assert_called_once_with()
    widget._neuron_table.set_added_file_ids.assert_called_once_with(set())


def test_render_scene_queries_full_trace_data_only_for_full_mode_neurons() -> None:
    viewer = _DummyViewer(ndisplay=3)
    widget = types.SimpleNamespace(
        _scene_render_modes={"n1": "full", "n2": "soma"},
        _db=MagicMock(),
        viewer=viewer,
        _render_btn=_DummyButton(),
        _render_soma_only_btn=_DummyButton(),
        _remove_selected_btn=_DummyButton(),
        _render_progress=_DummyProgressBar(),
        _render_status_label=_DummyLabel(),
        _render_mode_combo=_DummyComboBox("Both"),
        _opacity_slider=_DummyValueControl(100),
        _point_size_spin=_DummyValueControl(5),
        _line_width_spin=_DummyValueControl(4),
        _color_by_type_cb=_DummyCheckBox(False),
        _show_slice_projection_cb=_DummyCheckBox(False),
        _neuron_table=types.SimpleNamespace(
            get_color=lambda fid: [1.0, 0.0, 0.0, 1.0],
            set_added_file_ids=MagicMock(),
        ),
        _atlas=None,
        _slice_projector=MagicMock(),
        _soma_slice_projector=MagicMock(),
        _analysis_tab=MagicMock(),
        _current_neuron_layers=[],
        _capture_depth_state=MagicMock(return_value=None),
        _restore_depth_state=MagicMock(),
        _maybe_auto_center_slice=MagicMock(return_value=False),
        _use_auto_centering=lambda: False,
        _apply_layer_visibility=MagicMock(),
        _last_soma_selection=set(),
        _on_soma_selected=MagicMock(),
    )
    _bind_scene_helpers(widget)

    widget._db.get_neuron_lines_batch.return_value = {
        "n1": (
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
            np.array([[0, 1]], dtype=np.int32),
        )
    }
    widget._db.get_neurons_for_rendering.return_value = pd.DataFrame(
        {
            "file_id": ["n1", "n1"],
            "type": [1, 2],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
        }
    )
    widget._db.get_soma_locations.return_value = pd.DataFrame(
        {
            "file_id": ["n1", "n2"],
            "neuron_id": ["N1", "N2"],
            "x": [1.0, 7.0],
            "y": [3.0, 8.0],
            "z": [5.0, 9.0],
        }
    )
    widget._db.get_soma_points.return_value = pd.DataFrame(
        {
            "file_id": ["n2"],
            "neuron_id": ["N2"],
            "x": [7.0],
            "y": [8.0],
            "z": [9.0],
        }
    )

    NeuronViewerWidget._render_scene(widget)

    widget._db.get_neuron_lines_batch.assert_called_once_with(["n1"])
    widget._db.get_neurons_for_rendering.assert_called_once_with(["n1"])
    widget._db.get_soma_locations.assert_called_once_with(["n1", "n2"])
    widget._db.get_soma_points.assert_called_once_with(["n2"])
    assert widget._scene_render_modes == {"n1": "full", "n2": "soma"}
    widget._neuron_table.set_added_file_ids.assert_called_with({"n1", "n2"})
    widget._slice_projector.add_neuron_data_batch.assert_called_once()
    widget._soma_slice_projector.add_soma_data_batch.assert_called_once()
    assert {layer.name for layer in widget._current_neuron_layers} == {
        "Neuron Lines",
        "Neuron Points",
        "Soma Labels",
    }
    soma_layer = next(
        layer for layer in widget._current_neuron_layers if layer.name == "Soma Labels"
    )
    assert soma_layer.text == {
        "string": ["N1", "N2"],
        "size": 10,
        "color": "white",
        "visible": False,
    }


def test_apply_layer_visibility_hides_duplicate_soma_markers_in_2d_points_mode() -> (
    None
):
    viewer = _DummyViewer(ndisplay=2)
    neuron_points = _DummyPointsLayer(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        name="Neuron Points",
        face_color=np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        ),
        metadata={
            "file_ids_per_point": ["n1", "n1", "n2"],
            "point_types": [1, 2, 1],
            "base_face_colors": np.array(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
    )
    soma_labels = _DummyPointsLayer(
        np.array([[0.0, 0.0, 0.0]]),
        name="Soma Labels",
        face_color=np.array([[1.0, 0.0, 0.0, 1.0]]),
        metadata={"file_ids": ["n1"]},
    )
    neuron_lines = types.SimpleNamespace(name="Neuron Lines", visible=True)
    widget = types.SimpleNamespace(
        viewer=viewer,
        _show_slice_projection_cb=_DummyCheckBox(True),
        _render_mode_combo=_DummyComboBox("Points"),
        _current_neuron_layers=[neuron_lines, neuron_points, soma_labels],
    )
    _bind_projection_helpers(widget)

    NeuronViewerWidget._apply_layer_visibility(widget, False)

    assert neuron_lines.visible is False
    assert neuron_points.visible is True
    assert soma_labels.visible is False
    assert neuron_points.face_color[0, 3] == 0.0
    assert neuron_points.face_color[1, 3] == 1.0
    assert neuron_points.face_color[2, 3] == 0.0

    widget._show_slice_projection_cb = _DummyCheckBox(False)
    NeuronViewerWidget._apply_layer_visibility(widget, False)

    assert soma_labels.visible is True
    np.testing.assert_allclose(
        neuron_points.face_color,
        neuron_points.metadata["base_face_colors"],
    )


def test_toggle_slice_projection_updates_both_projectors_and_reapplies_2d_visibility() -> (
    None
):
    viewer = _DummyViewer(ndisplay=2)
    line_projector = types.SimpleNamespace(enabled=False)
    soma_projector = types.SimpleNamespace(enabled=False)
    widget = types.SimpleNamespace(
        viewer=viewer,
        _slice_projector=line_projector,
        _soma_slice_projector=soma_projector,
        _slice_warning_label=_DummyLabel(),
        _current_neuron_layers=[object()],
        _apply_layer_visibility=MagicMock(),
    )

    NeuronViewerWidget._toggle_slice_projection(widget, 1)

    assert line_projector.enabled is True
    assert soma_projector.enabled is True
    assert widget._slice_warning_label.visible is True
    widget._apply_layer_visibility.assert_called_once_with(False)


def test_update_layer_colors_updates_both_projection_systems() -> None:
    neuron_points = _DummyPointsLayer(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        name="Neuron Points",
        face_color=np.array([[0.2, 0.2, 0.2, 1.0], [0.3, 0.3, 0.3, 1.0]]),
        metadata={
            "file_ids_per_point": ["n1", "n2"],
            "point_types": [1, 2],
        },
    )
    soma_labels = _DummyPointsLayer(
        np.array([[0.0, 0.0, 0.0]]),
        name="Soma Labels",
        face_color=np.array([[0.2, 0.2, 0.2, 1.0]]),
        metadata={"file_ids": ["n1"]},
    )
    widget = types.SimpleNamespace(
        _current_neuron_layers=[neuron_points, soma_labels],
        _slice_projector=MagicMock(),
        _soma_slice_projector=MagicMock(),
        _sync_soma_projection_overlay_state=MagicMock(),
    )

    NeuronViewerWidget._update_layer_colors(
        widget,
        {"n1": [1.0, 0.0, 0.0, 1.0], "n2": [0.0, 1.0, 0.0, 1.0]},
    )

    np.testing.assert_allclose(
        neuron_points.metadata["base_face_colors"],
        np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]),
    )
    widget._slice_projector.update_neuron_colors.assert_called_once()
    widget._soma_slice_projector.update_neuron_colors.assert_called_once()
    widget._sync_soma_projection_overlay_state.assert_called_once()


def test_on_soma_selected_uses_metadata_file_ids_for_projected_layer() -> None:
    table = MagicMock()
    widget = types.SimpleNamespace(_last_soma_selection=set(), _neuron_table=table)
    layer = types.SimpleNamespace(
        selected_data={0, 2},
        metadata={"file_ids": ["n1", "n2", "n3"]},
    )
    event = types.SimpleNamespace(source=layer)

    NeuronViewerWidget._on_soma_selected(widget, event)

    table.select_file_ids.assert_called_once_with(["n1", "n3"])


def test_on_soma_selected_deduplicates_projected_file_ids() -> None:
    table = MagicMock()
    widget = types.SimpleNamespace(_last_soma_selection=set(), _neuron_table=table)
    layer = types.SimpleNamespace(
        selected_data={0, 1, 2},
        metadata={"file_ids": ["n1", "n1", "n2"]},
    )
    event = types.SimpleNamespace(source=layer)

    NeuronViewerWidget._on_soma_selected(widget, event)

    table.select_file_ids.assert_called_once_with(["n1", "n2"])


def test_on_soma_selected_same_indices_from_different_layers_are_not_noop() -> None:
    table = MagicMock()
    widget = types.SimpleNamespace(_last_soma_selection=set(), _neuron_table=table)
    layer_a = types.SimpleNamespace(
        selected_data={0},
        metadata={"file_ids": ["n1"]},
    )
    layer_b = types.SimpleNamespace(
        selected_data={0},
        metadata={"file_ids": ["n2"]},
    )

    NeuronViewerWidget._on_soma_selected(
        widget,
        types.SimpleNamespace(source=layer_a),
    )
    NeuronViewerWidget._on_soma_selected(
        widget,
        types.SimpleNamespace(source=layer_b),
    )

    assert [call.args[0] for call in table.select_file_ids.call_args_list] == [
        ["n1"],
        ["n2"],
    ]


def test_on_soma_selected_reprocesses_same_projected_indices_after_metadata_change() -> (
    None
):
    table = MagicMock()
    widget = types.SimpleNamespace(_last_soma_selection=set(), _neuron_table=table)
    layer = types.SimpleNamespace(
        selected_data={0},
        metadata={"file_ids": ["n1"]},
    )
    event = types.SimpleNamespace(source=layer)

    NeuronViewerWidget._on_soma_selected(widget, event)
    table.select_file_ids.reset_mock()
    layer.metadata = {"file_ids": ["n2"]}

    NeuronViewerWidget._on_soma_selected(widget, event)

    table.select_file_ids.assert_called_once_with(["n2"])


def test_on_soma_selected_empty_selection_clears_after_soma_selection() -> None:
    table = MagicMock()
    widget = types.SimpleNamespace(_last_soma_selection=set(), _neuron_table=table)
    layer = types.SimpleNamespace(
        selected_data={0},
        metadata={"file_ids": ["n1"]},
    )
    event = types.SimpleNamespace(source=layer)

    NeuronViewerWidget._on_soma_selected(widget, event)
    table.select_file_ids.reset_mock()
    layer.selected_data = set()

    NeuronViewerWidget._on_soma_selected(widget, event)

    table.select_file_ids.assert_called_once_with([])


@pytest.mark.parametrize(
    ("handler_name", "expected_message", "expected_soma_only"),
    [
        (
            "_query_atlas_neurons_any_node",
            "Searching for neurons with any node in selected atlas regions. Please wait...",
            False,
        ),
        (
            "_query_atlas_neurons_soma",
            "Searching for neurons with soma in selected atlas regions. Please wait...",
            True,
        ),
    ],
)
def test_atlas_query_handlers_set_wait_message_before_dispatch(
    handler_name: str,
    expected_message: str,
    expected_soma_only: bool,
) -> None:
    observed: dict[str, str] = {}

    widget = types.SimpleNamespace(
        _regions_status_label=_DummyLabel(),
    )

    def _record_status(*, soma_only: bool = False) -> None:
        observed["status"] = widget._regions_status_label.text
        observed["soma_only"] = soma_only

    widget._query_neurons_by_region = _record_status

    getattr(NeuronViewerWidget, handler_name)(widget)

    assert observed["status"] == expected_message
    assert observed["soma_only"] is expected_soma_only


@pytest.mark.parametrize(
    ("handler_name", "expected_message", "expected_soma_only"),
    [
        (
            "_query_mask_neurons_any_node",
            "Searching for neurons with any node in selected mask layers. Please wait...",
            False,
        ),
        (
            "_query_mask_neurons_soma",
            "Searching for neurons with soma in selected mask layers. Please wait...",
            True,
        ),
    ],
)
def test_mask_query_handlers_set_wait_message_before_dispatch(
    handler_name: str,
    expected_message: str,
    expected_soma_only: bool,
) -> None:
    observed: dict[str, str] = {}

    widget = types.SimpleNamespace(
        _regions_status_label=_DummyLabel(),
    )

    def _record_status(*, soma_only: bool = False) -> None:
        observed["status"] = widget._regions_status_label.text
        observed["soma_only"] = soma_only

    widget._query_neurons_by_mask = _record_status

    getattr(NeuronViewerWidget, handler_name)(widget)

    assert observed["status"] == expected_message
    assert observed["soma_only"] is expected_soma_only


@pytest.mark.parametrize(
    ("source", "expected_status", "expected_target"),
    [
        (
            "Atlas Regions",
            "Searching for neurons with basal dendrite or apical dendrite nodes "
            "in selected atlas regions. Please wait...",
            "atlas",
        ),
        (
            "Custom Regions",
            "Searching for neurons with basal dendrite or apical dendrite nodes "
            "in selected custom regions. Please wait...",
            "custom",
        ),
        (
            "Mask Layer",
            "Searching for neurons with basal dendrite or apical dendrite nodes "
            "in selected mask layers. Please wait...",
            "mask",
        ),
    ],
)
def test_active_region_query_dispatches_selected_node_types(
    source: str,
    expected_status: str,
    expected_target: str,
) -> None:
    observed: dict[str, object] = {}
    widget = types.SimpleNamespace(
        _region_query_source=source,
        _regions_status_label=_DummyLabel(),
        _region_node_type_combo=types.SimpleNamespace(
            selected_node_types=lambda: (3, 4)
        ),
    )

    def _record_atlas(*, soma_only: bool = False, node_types=None) -> None:
        observed["target"] = "atlas"
        observed["soma_only"] = soma_only
        observed["node_types"] = node_types

    def _record_mask(*, soma_only: bool = False, node_types=None) -> None:
        observed["target"] = "mask"
        observed["soma_only"] = soma_only
        observed["node_types"] = node_types

    def _record_custom(*, soma_only: bool = False, node_types=None) -> None:
        observed["target"] = "custom"
        observed["soma_only"] = soma_only
        observed["node_types"] = node_types

    widget._query_neurons_by_region = _record_atlas
    widget._query_neurons_by_custom_region = _record_custom
    widget._query_neurons_by_mask = _record_mask

    NeuronViewerWidget._query_neurons_for_active_region_source(widget)

    assert widget._regions_status_label.text == expected_status
    assert observed == {
        "target": expected_target,
        "soma_only": False,
        "node_types": (3, 4),
    }


def test_resolve_region_query_file_scope_defaults_to_whole_parquet() -> None:
    widget = types.SimpleNamespace(
        _region_query_scope="whole",
        _region_query_scope_combo=_DummyComboBox("Whole Parquet", data="whole"),
        _regions_status_label=_DummyLabel(),
    )
    _bind_region_query_scope_helpers(widget)

    assert widget._selected_region_query_scope() == "whole"
    assert widget._resolve_region_query_file_scope() == (
        True,
        None,
        "whole parquet",
        None,
    )


def test_query_neurons_by_region_uses_current_table_scope_without_inheriting_whole_selection() -> (
    None
):
    result = pd.DataFrame(
        {
            "file_id": ["n1"],
            "neuron_id": ["N1"],
            "subject": ["s1"],
        }
    )
    widget = types.SimpleNamespace(
        _db=MagicMock(),
        _whole_parquet_region_selector=_DummyRegionSelector(
            direct_acronyms=["ROOT"],
            query_acronyms=["ROOT", "R1"],
            include_children=True,
        ),
        _current_table_region_selector=_DummyRegionSelector(
            direct_acronyms=["R1"],
            query_acronyms=["R1"],
            include_children=False,
        ),
        _region_query_scope_combo=_DummyComboBox("Current Table", data="current"),
        _regions_status_label=_DummyLabel(),
        _neuron_table=types.SimpleNamespace(file_ids=lambda: ["n1", "n2"]),
        _populate_neuron_table=MagicMock(),
    )
    widget._db.get_neurons_by_region.return_value = result
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._query_neurons_by_region(widget, soma_only=True)

    widget._db.get_neurons_by_region.assert_called_once_with(
        ["R1"],
        soma_only=True,
        file_ids=["n1", "n2"],
        node_types=(1,),
    )
    widget._populate_neuron_table.assert_called_once_with(
        result,
        preserve_existing=True,
    )
    assert widget._regions_status_label.text == (
        "Found 1 neuron(s) with soma in selected atlas regions "
        "within current table (from 2 input neurons). "
        "Query: R1; descendants: off."
    )


def test_query_neurons_by_region_passes_explicit_node_type_filter() -> None:
    result = pd.DataFrame(
        {
            "file_id": ["n2"],
            "neuron_id": ["N2"],
            "subject": ["s2"],
        }
    )
    widget = types.SimpleNamespace(
        _db=MagicMock(),
        _whole_parquet_region_selector=_DummyRegionSelector(
            direct_acronyms=["R1"],
            query_acronyms=["R1"],
            include_children=True,
        ),
        _current_table_region_selector=_DummyRegionSelector(),
        _region_query_scope_combo=_DummyComboBox("Whole Parquet", data="whole"),
        _regions_status_label=_DummyLabel(),
        _neuron_table=types.SimpleNamespace(file_ids=lambda: []),
        _populate_neuron_table=MagicMock(),
    )
    widget._db.get_neurons_by_region.return_value = result
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._query_neurons_by_region(widget, node_types=(3, 4))

    widget._db.get_neurons_by_region.assert_called_once_with(
        ["R1"],
        soma_only=False,
        file_ids=None,
        node_types=(3, 4),
    )
    widget._populate_neuron_table.assert_called_once_with(
        result,
        preserve_existing=False,
    )
    assert widget._regions_status_label.text == (
        "Found 1 neuron(s) with basal dendrite or apical dendrite nodes "
        "in selected atlas regions within whole parquet. "
        "Query: R1; descendants: on."
    )


def test_query_neurons_by_mask_current_table_scope_requires_nonempty_table() -> None:
    widget = types.SimpleNamespace(
        _db=MagicMock(),
        _atlas=types.SimpleNamespace(annotation=np.zeros((2, 2, 2), dtype=np.uint8)),
        _selected_mask_query_layers=lambda: [
            types.SimpleNamespace(
                name="Mask A",
                data=np.ones((2, 2, 2), dtype=np.uint8),
            )
        ],
        _region_query_scope_combo=_DummyComboBox("Current Table", data="current"),
        _regions_status_label=_DummyLabel(),
        _neuron_table=types.SimpleNamespace(file_ids=lambda: []),
    )
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._query_neurons_by_mask(widget, soma_only=False)

    widget._db.get_neurons_by_mask.assert_not_called()
    assert widget._regions_status_label.text == (
        "Current table is empty; switch search scope to Whole Parquet or "
        "populate the table first."
    )


def test_source_file_ids_for_layers_deduplicates_metadata_sources() -> None:
    widget = types.SimpleNamespace(
        _normalise_layer_file_ids=NeuronViewerWidget._normalise_layer_file_ids,
    )
    layers = [
        types.SimpleNamespace(
            metadata={
                "source_file_ids": ["n1", "n2"],
                "file_ids": ["n2"],
            },
        ),
        types.SimpleNamespace(
            metadata={
                "query_excluded_file_ids": ["n3", "n1"],
                "source_file_ids": ["n4"],
            },
        ),
    ]

    assert NeuronViewerWidget._source_file_ids_for_layers(widget, layers) == [
        "n1",
        "n2",
        "n3",
        "n4",
    ]


def test_mask_source_exclusion_defaults_to_enabled_without_checkbox() -> None:
    widget = types.SimpleNamespace()

    assert NeuronViewerWidget._mask_source_exclusion_enabled(widget) is True


def test_query_neurons_by_mask_uses_current_layer_data_and_excludes_sources() -> None:
    result = pd.DataFrame(
        {
            "file_id": ["n3"],
            "neuron_id": ["N3"],
            "subject": ["s3"],
        }
    )
    db = MagicMock()
    db.get_neurons_by_mask.return_value = result
    mask_data = np.zeros((2, 2, 2), dtype=np.uint8)
    mask_data[1, 0, 1] = 1
    layer = types.SimpleNamespace(
        name="Mask A",
        data=mask_data,
        metadata={"query_excluded_file_ids": ["n1", "n2", "n1"]},
    )
    widget = types.SimpleNamespace(
        _db=db,
        _atlas=types.SimpleNamespace(annotation=np.zeros((2, 2, 2), dtype=np.uint8)),
        _selected_mask_query_layers=lambda: [layer],
        _region_query_scope_combo=_DummyComboBox("Whole Parquet", data="whole"),
        _regions_status_label=_DummyLabel(),
        _mask_exclude_source_neurons_cb=_DummyCheckBox(True),
        _populate_neuron_table=MagicMock(),
    )
    widget._normalise_layer_file_ids = NeuronViewerWidget._normalise_layer_file_ids
    widget._source_file_ids_for_layers = types.MethodType(
        NeuronViewerWidget._source_file_ids_for_layers,
        widget,
    )
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._query_neurons_by_mask(widget, soma_only=False)

    args, kwargs = db.get_neurons_by_mask.call_args
    np.testing.assert_array_equal(args[0], mask_data > 0)
    assert args[1] is widget._atlas
    assert kwargs == {
        "soma_only": False,
        "file_ids": None,
        "exclude_file_ids": ["n1", "n2"],
        "node_types": None,
    }
    widget._populate_neuron_table.assert_called_once_with(
        result,
        preserve_existing=False,
    )
    assert widget._regions_status_label.text == (
        "Found 1 neuron(s) with any node in 1 selected mask layer(s) "
        "within whole parquet: Mask A; excluded 2 source neurons"
    )


def test_query_neurons_by_mask_soma_uses_soma_only_and_status_text() -> None:
    result = pd.DataFrame(
        {
            "file_id": ["n2"],
            "neuron_id": ["N2"],
            "subject": ["s2"],
        }
    )
    db = MagicMock()
    db.get_neurons_by_mask.return_value = result
    mask_data = np.zeros((2, 2, 2), dtype=np.uint8)
    mask_data[1, 1, 0] = 1
    layer = types.SimpleNamespace(
        name="Mask Soma",
        data=mask_data,
        metadata={},
    )
    widget = types.SimpleNamespace(
        _db=db,
        _atlas=types.SimpleNamespace(annotation=np.zeros((2, 2, 2), dtype=np.uint8)),
        _selected_mask_query_layers=lambda: [layer],
        _region_query_scope_combo=_DummyComboBox("Whole Parquet", data="whole"),
        _regions_status_label=_DummyLabel(),
        _populate_neuron_table=MagicMock(),
    )
    widget._normalise_layer_file_ids = NeuronViewerWidget._normalise_layer_file_ids
    widget._source_file_ids_for_layers = types.MethodType(
        NeuronViewerWidget._source_file_ids_for_layers,
        widget,
    )
    widget._mask_source_exclusion_enabled = types.MethodType(
        NeuronViewerWidget._mask_source_exclusion_enabled,
        widget,
    )
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._query_neurons_by_mask(widget, soma_only=True)

    args, kwargs = db.get_neurons_by_mask.call_args
    np.testing.assert_array_equal(args[0], mask_data > 0)
    assert args[1] is widget._atlas
    assert kwargs == {
        "soma_only": True,
        "file_ids": None,
        "exclude_file_ids": None,
        "node_types": (1,),
    }
    widget._populate_neuron_table.assert_called_once_with(
        result,
        preserve_existing=False,
    )
    assert widget._regions_status_label.text == (
        "Found 1 neuron(s) with soma in 1 selected mask layer(s) "
        "within whole parquet: Mask Soma"
    )


def test_query_neurons_by_mask_includes_sources_when_checkbox_unchecked() -> None:
    result = pd.DataFrame(
        {
            "file_id": ["n1", "n3"],
            "neuron_id": ["N1", "N3"],
            "subject": ["s1", "s3"],
        }
    )
    db = MagicMock()
    db.get_neurons_by_mask.return_value = result
    layer = types.SimpleNamespace(
        name="Mask A",
        data=np.ones((2, 2, 2), dtype=np.uint8),
        metadata={"query_excluded_file_ids": ["n1", "n2"]},
    )
    widget = types.SimpleNamespace(
        _db=db,
        _atlas=types.SimpleNamespace(annotation=np.zeros((2, 2, 2), dtype=np.uint8)),
        _selected_mask_query_layers=lambda: [layer],
        _region_query_scope_combo=_DummyComboBox("Whole Parquet", data="whole"),
        _regions_status_label=_DummyLabel(),
        _mask_exclude_source_neurons_cb=_DummyCheckBox(False),
        _populate_neuron_table=MagicMock(),
    )
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._query_neurons_by_mask(widget, soma_only=False)

    _args, kwargs = db.get_neurons_by_mask.call_args
    assert kwargs["exclude_file_ids"] is None
    assert kwargs["node_types"] is None
    assert widget._regions_status_label.text == (
        "Found 2 neuron(s) with any node in 1 selected mask layer(s) "
        "within whole parquet: Mask A"
    )


def test_update_mask_query_summary_counts_unique_selected_source_neurons() -> None:
    layer_a = types.SimpleNamespace(
        name="Mask A",
        metadata={"query_excluded_file_ids": ["n1", "n2"]},
    )
    layer_b = types.SimpleNamespace(
        name="Mask B",
        metadata={"source_file_ids": ["n2", "n3"]},
    )
    label = _DummyLabel()
    widget = types.SimpleNamespace(
        _mask_query_hint_label=label,
        _mask_exclude_source_neurons_cb=_DummyCheckBox(True),
        _generated_mask_layers=lambda: [layer_a, layer_b],
        _selected_mask_query_layers=lambda: [layer_a, layer_b],
    )
    widget._normalise_layer_file_ids = NeuronViewerWidget._normalise_layer_file_ids
    widget._source_file_ids_for_layers = types.MethodType(
        NeuronViewerWidget._source_file_ids_for_layers,
        widget,
    )
    widget._mask_source_exclusion_enabled = types.MethodType(
        NeuronViewerWidget._mask_source_exclusion_enabled,
        widget,
    )

    NeuronViewerWidget._update_mask_query_summary(widget)

    assert label.text == (
        "Selected mask layer(s) were generated from 3 unique source neuron(s); "
        "source neurons will be excluded."
    )

    widget._mask_exclude_source_neurons_cb.setChecked(False)
    NeuronViewerWidget._update_mask_query_summary(widget)

    assert label.text == (
        "Selected mask layer(s) were generated from 3 unique source neuron(s); "
        "source neurons will be included."
    )


def test_update_mask_query_summary_handles_empty_selection_and_missing_sources() -> (
    None
):
    layer_a = types.SimpleNamespace(name="Mask A", metadata={})
    layer_b = types.SimpleNamespace(name="Mask B", metadata={})
    label = _DummyLabel()
    widget = types.SimpleNamespace(
        _mask_query_hint_label=label,
        _mask_exclude_source_neurons_cb=_DummyCheckBox(True),
        _generated_mask_layers=lambda: [layer_a, layer_b],
        _selected_mask_query_layers=lambda: [],
    )
    widget._normalise_layer_file_ids = NeuronViewerWidget._normalise_layer_file_ids
    widget._source_file_ids_for_layers = types.MethodType(
        NeuronViewerWidget._source_file_ids_for_layers,
        widget,
    )
    widget._mask_source_exclusion_enabled = types.MethodType(
        NeuronViewerWidget._mask_source_exclusion_enabled,
        widget,
    )

    NeuronViewerWidget._update_mask_query_summary(widget)

    assert label.text == (
        "2 generated mask layer(s) available. "
        "Select mask layers to see source-neuron count."
    )

    widget._selected_mask_query_layers = lambda: [layer_a]
    NeuronViewerWidget._update_mask_query_summary(widget)

    assert label.text == "Selected mask layer(s) do not record source neurons."

    widget._generated_mask_layers = lambda: []
    NeuronViewerWidget._update_mask_query_summary(widget)

    assert label.text == "No generated mask layers are available."


def test_on_region_query_source_changed_uses_named_pages_and_scope_stacks() -> None:
    widget = types.SimpleNamespace(
        _region_query_source="Atlas Regions",
        _region_query_stack=_DummyStack(),
        _atlas_region_scope_stack=_DummyStack(),
        _custom_region_scope_stack=_DummyStack(),
        _region_query_scope_combo=_DummyComboBox("Current Table", data="current"),
        _regions_status_label=_DummyLabel(),
        _atlas_query_any_node_btn=_DummyButton(),
        _atlas_query_soma_btn=_DummyButton(),
        _mask_query_any_node_btn=_DummyButton(),
        _mask_query_soma_btn=_DummyButton(),
        _whole_parquet_region_selector=_DummyRegionSelector(direct_acronyms=["ROOT"]),
        _current_table_region_selector=_DummyRegionSelector(direct_acronyms=["R1"]),
        _show_region_meshes_cb=_DummyCheckBox(False),
        _show_region_seg_cb=_DummyCheckBox(False),
    )
    widget._atlas_region_query_buttons = types.MethodType(
        NeuronViewerWidget._atlas_region_query_buttons,
        widget,
    )
    widget._mask_layer_query_buttons = types.MethodType(
        NeuronViewerWidget._mask_layer_query_buttons,
        widget,
    )
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._on_region_query_source_changed(widget, "Atlas Regions")

    assert widget._region_query_stack.index == 0
    assert widget._atlas_region_scope_stack.index == 1
    assert widget._custom_region_scope_stack.index == 1
    assert widget._atlas_query_any_node_btn.visible is True
    assert widget._atlas_query_soma_btn.visible is True
    assert widget._mask_query_any_node_btn.visible is False
    assert widget._mask_query_soma_btn.visible is False

    NeuronViewerWidget._on_region_query_source_changed(widget, "Custom Regions")

    assert widget._region_query_stack.index == 1
    assert widget._atlas_query_any_node_btn.visible is True
    assert widget._atlas_query_soma_btn.visible is True
    assert widget._mask_query_any_node_btn.visible is False
    assert widget._mask_query_soma_btn.visible is False

    NeuronViewerWidget._on_region_query_source_changed(widget, "Mask Layer")

    assert widget._region_query_stack.index == 2
    assert widget._atlas_query_any_node_btn.visible is False
    assert widget._atlas_query_soma_btn.visible is False
    assert widget._mask_query_any_node_btn.visible is True
    assert widget._mask_query_soma_btn.visible is True


def test_custom_region_source_sets_button_text_and_updates_custom_previews() -> None:
    group = CustomRegionSelectionGroup(
        label="L1",
        region_ids=(101,),
        acronyms=("R1",),
    )
    widget = types.SimpleNamespace(
        _region_query_source="Atlas Regions",
        _region_query_stack=_DummyStack(),
        _atlas_region_scope_stack=_DummyStack(),
        _custom_region_scope_stack=_DummyStack(),
        _region_query_scope_combo=_DummyComboBox("Whole Parquet", data="whole"),
        _regions_status_label=_DummyLabel(),
        _region_query_find_btn=_DummyButton(),
        _whole_parquet_region_selector=_DummyRegionSelector(direct_acronyms=["R1"]),
        _whole_parquet_custom_region_selector=_DummyCustomRegionSelector(
            [101],
            region_groups=(group,),
        ),
        _show_region_meshes_cb=_DummyCheckBox(True),
        _show_region_seg_cb=_DummyCheckBox(True),
        _update_region_meshes=MagicMock(),
        _update_region_segmentation=MagicMock(),
        _update_custom_region_meshes=MagicMock(),
        _update_custom_region_segmentation=MagicMock(),
    )
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._on_region_query_source_changed(widget, "Custom Regions")

    assert widget._region_query_stack.index == 1
    assert (
        widget._region_query_find_btn.text == "Find Neurons in Selected Custom Regions"
    )
    widget._update_region_meshes.assert_not_called()
    widget._update_region_segmentation.assert_not_called()
    widget._update_custom_region_meshes.assert_called_once_with((group,))
    widget._update_custom_region_segmentation.assert_called_once_with((group,))

    NeuronViewerWidget._on_region_query_source_changed(widget, "Atlas Regions")

    widget._update_region_meshes.assert_called_once_with(["R1"])
    widget._update_region_segmentation.assert_called_once_with(["R1"])


def test_on_region_query_scope_changed_switches_preview_to_active_selector() -> None:
    widget = types.SimpleNamespace(
        _region_query_source="Atlas Regions",
        _region_query_scope="whole",
        _region_query_scope_combo=_DummyComboBox("Current Table", data="current"),
        _atlas_region_scope_stack=_DummyStack(),
        _regions_status_label=_DummyLabel(),
        _whole_parquet_region_selector=_DummyRegionSelector(
            direct_acronyms=["ROOT"],
            query_acronyms=["ROOT", "R1"],
            include_children=True,
        ),
        _current_table_region_selector=_DummyRegionSelector(
            direct_acronyms=["R1"],
            query_acronyms=["R1"],
            include_children=False,
        ),
        _show_region_meshes_cb=_DummyCheckBox(True),
        _show_region_seg_cb=_DummyCheckBox(True),
        _update_region_meshes=MagicMock(),
        _update_region_segmentation=MagicMock(),
    )
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._on_region_query_scope_changed(widget, "Current Table")

    assert widget._region_query_scope == "current"
    assert widget._atlas_region_scope_stack.index == 1
    widget._update_region_meshes.assert_called_once_with(["R1"])
    widget._update_region_segmentation.assert_called_once_with(["R1"])

    widget._region_query_scope_combo = _DummyComboBox("Whole Parquet", data="whole")
    widget._update_region_meshes.reset_mock()
    widget._update_region_segmentation.reset_mock()

    NeuronViewerWidget._on_region_query_scope_changed(widget, "Whole Parquet")

    assert widget._region_query_scope == "whole"
    assert widget._atlas_region_scope_stack.index == 0
    widget._update_region_meshes.assert_called_once_with(["ROOT"])
    widget._update_region_segmentation.assert_called_once_with(["ROOT"])


def test_custom_region_scope_change_updates_scope_specific_reference_layers() -> None:
    whole_group = CustomRegionSelectionGroup(
        label="L1",
        region_ids=(101,),
        acronyms=("R1",),
    )
    current_group = CustomRegionSelectionGroup(
        label="L2/3",
        region_ids=(202,),
        acronyms=("R2",),
    )
    widget = types.SimpleNamespace(
        _region_query_source="Custom Regions",
        _region_query_scope="whole",
        _region_query_scope_combo=_DummyComboBox("Current Table", data="current"),
        _atlas_region_scope_stack=_DummyStack(),
        _custom_region_scope_stack=_DummyStack(),
        _regions_status_label=_DummyLabel(),
        _whole_parquet_custom_region_selector=_DummyCustomRegionSelector(
            [101],
            region_groups=(whole_group,),
        ),
        _current_table_custom_region_selector=_DummyCustomRegionSelector(
            [202],
            region_groups=(current_group,),
        ),
        _show_region_meshes_cb=_DummyCheckBox(True),
        _show_region_seg_cb=_DummyCheckBox(True),
        _update_region_meshes=MagicMock(),
        _update_region_segmentation=MagicMock(),
        _update_custom_region_meshes=MagicMock(),
        _update_custom_region_segmentation=MagicMock(),
    )
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._on_region_query_scope_changed(widget, "Current Table")

    assert widget._atlas_region_scope_stack.index == 1
    assert widget._custom_region_scope_stack.index == 1
    widget._update_region_meshes.assert_not_called()
    widget._update_region_segmentation.assert_not_called()
    widget._update_custom_region_meshes.assert_called_once_with((current_group,))
    widget._update_custom_region_segmentation.assert_called_once_with((current_group,))


def test_custom_selection_change_refreshes_only_when_custom_source_is_active() -> None:
    widget = types.SimpleNamespace(
        _region_query_source="Custom Regions",
        _regions_status_label=_DummyLabel(),
        _sync_active_region_reference_layers=MagicMock(),
    )

    NeuronViewerWidget._on_custom_regions_selected(widget, [101])

    assert widget._regions_status_label.text == ""
    widget._sync_active_region_reference_layers.assert_called_once_with()

    widget._region_query_source = "Atlas Regions"
    widget._sync_active_region_reference_layers.reset_mock()
    NeuronViewerWidget._on_custom_regions_selected(widget, [101])
    widget._sync_active_region_reference_layers.assert_not_called()


def test_flatmap_region_selection_follows_active_source_and_scope() -> None:
    whole_group = CustomRegionSelectionGroup(
        label="L1",
        region_ids=(102, 101),
        acronyms=("C102", "C101"),
    )
    current_group = CustomRegionSelectionGroup(
        label="L2/3",
        region_ids=(202,),
        acronyms=("C202",),
    )
    widget = types.SimpleNamespace(
        _region_query_source="Atlas Regions",
        _region_query_scope_combo=_DummyComboBox("Whole Parquet", data="whole"),
        _whole_parquet_region_selector=_DummyRegionSelector(
            direct_ids=[10],
            query_ids=[12, 10, 11],
            direct_acronyms=["PARENT"],
            query_acronyms=["R12", "PARENT", "R11"],
        ),
        _current_table_region_selector=_DummyRegionSelector(
            direct_ids=[20],
            query_ids=[21, 20],
            direct_acronyms=["CURRENT"],
            query_acronyms=["R21", "CURRENT"],
        ),
        _whole_parquet_custom_region_selector=_DummyCustomRegionSelector(
            [102, 101, 101],
            region_groups=(whole_group,),
        ),
        _current_table_custom_region_selector=_DummyCustomRegionSelector(
            [202],
            region_groups=(current_group,),
        ),
    )
    _bind_region_query_scope_helpers(widget)

    assert widget._active_flatmap_region_ids() == [10, 11, 12]
    assert widget._active_flatmap_parent_region_ids() == [10]
    assert widget._active_flatmap_region_acronyms() == [
        "R12",
        "PARENT",
        "R11",
    ]
    assert widget._active_flatmap_region_source() == "atlas_regions"
    assert widget._active_flatmap_region_scope() == "whole_parquet"
    assert widget._active_flatmap_region_error() is None

    widget._region_query_source = "Custom Regions"

    assert widget._active_flatmap_region_ids() == [101, 102]
    assert widget._active_flatmap_parent_region_ids() == [101, 102]
    assert widget._active_flatmap_region_acronyms() == ["C101", "C102"]
    assert widget._active_flatmap_region_source() == "custom_regions"
    assert widget._active_flatmap_region_error() is None

    widget._region_query_scope_combo = _DummyComboBox(
        "Current Table",
        data="current",
    )

    assert widget._active_flatmap_region_ids() == [202]
    assert widget._active_flatmap_parent_region_ids() == [202]
    assert widget._active_flatmap_region_acronyms() == ["C202"]
    assert widget._active_flatmap_region_scope() == "current_table"

    widget._region_query_source = "Mask Layer"

    assert widget._active_flatmap_region_ids() == []
    assert widget._active_flatmap_parent_region_ids() == []
    assert widget._active_flatmap_region_acronyms() == []
    assert widget._active_flatmap_region_source() == "mask_layer"
    assert "do not support Mask Layer" in widget._active_flatmap_region_error()


def test_flatmap_custom_region_error_preserves_hierarchy_message() -> None:
    message = "The loaded atlas has no compatible terminal layer hierarchy."
    widget = types.SimpleNamespace(
        _region_query_source="Custom Regions",
        _region_query_scope_combo=_DummyComboBox("Whole Parquet", data="whole"),
        _whole_parquet_custom_region_selector=_DummyCustomRegionSelector(
            has_hierarchy=False,
            unavailable_message=message,
        ),
    )
    _bind_region_query_scope_helpers(widget)

    assert widget._active_flatmap_region_ids() == []
    assert widget._active_flatmap_region_error() == message


def test_update_custom_region_meshes_creates_at_most_one_layer_per_group(
    monkeypatch,
) -> None:
    groups = (
        CustomRegionSelectionGroup(
            label="L1",
            region_ids=(101, 102),
            acronyms=("R1", "R2"),
        ),
        CustomRegionSelectionGroup(
            label="L2/3",
            region_ids=(202,),
            acronyms=("R3",),
        ),
    )
    viewer = _DummyViewer(ndisplay=2)
    atlas = object()
    add_group = MagicMock(
        side_effect=[
            (object(), ()),
            (None, ("R3",)),
        ]
    )
    remove_layers = MagicMock()
    warnings = []
    globals_dict = NeuronViewerWidget._update_custom_region_meshes.__globals__
    monkeypatch.setitem(globals_dict, "add_region_mesh_group", add_group)
    monkeypatch.setitem(globals_dict, "remove_region_layers", remove_layers)
    monkeypatch.setitem(globals_dict, "show_warning", warnings.append)
    widget = types.SimpleNamespace(
        _atlas=atlas,
        viewer=viewer,
        _show_region_meshes_cb=_DummyCheckBox(True),
        _mesh_opacity_slider=_DummyValueControl(30),
    )

    NeuronViewerWidget._update_custom_region_meshes(widget, groups)

    remove_layers.assert_called_once_with(viewer)
    assert viewer.dims.ndisplay == 3
    assert add_group.call_count == 2
    add_group.assert_any_call(viewer, atlas, groups[0], opacity=0.3)
    add_group.assert_any_call(viewer, atlas, groups[1], opacity=0.3)
    assert len(warnings) == 1
    assert "R3" in warnings[0]


def test_update_custom_region_segmentation_passes_exact_terminal_ids(
    monkeypatch,
) -> None:
    groups = (
        CustomRegionSelectionGroup(
            label="L1",
            region_ids=(102, 101),
            acronyms=("R2", "R1"),
        ),
        CustomRegionSelectionGroup(
            label="L2/3",
            region_ids=(202,),
            acronyms=("R3",),
        ),
    )
    viewer = _DummyViewer()
    atlas = object()
    add_segmentation = MagicMock()
    remove_segmentation = MagicMock()
    globals_dict = NeuronViewerWidget._update_custom_region_segmentation.__globals__
    monkeypatch.setitem(
        globals_dict,
        "add_region_id_segmentation",
        add_segmentation,
    )
    monkeypatch.setitem(
        globals_dict,
        "remove_region_segmentation",
        remove_segmentation,
    )
    widget = types.SimpleNamespace(
        _atlas=atlas,
        viewer=viewer,
        _show_region_seg_cb=_DummyCheckBox(True),
        _seg_opacity_slider=_DummyValueControl(40),
    )

    NeuronViewerWidget._update_custom_region_segmentation(widget, groups)

    remove_segmentation.assert_called_once_with(viewer)
    add_segmentation.assert_called_once_with(
        viewer,
        atlas,
        [101, 102, 202],
        opacity=0.4,
    )


def test_empty_custom_selection_clears_existing_reference_layers(
    monkeypatch,
) -> None:
    viewer = _DummyViewer()
    remove_meshes = MagicMock()
    remove_segmentation = MagicMock()
    add_meshes = MagicMock()
    add_segmentation = MagicMock()
    monkeypatch.setitem(
        NeuronViewerWidget._update_custom_region_meshes.__globals__,
        "remove_region_layers",
        remove_meshes,
    )
    monkeypatch.setitem(
        NeuronViewerWidget._update_custom_region_meshes.__globals__,
        "add_region_mesh_group",
        add_meshes,
    )
    monkeypatch.setitem(
        NeuronViewerWidget._update_custom_region_segmentation.__globals__,
        "remove_region_segmentation",
        remove_segmentation,
    )
    monkeypatch.setitem(
        NeuronViewerWidget._update_custom_region_segmentation.__globals__,
        "add_region_id_segmentation",
        add_segmentation,
    )
    widget = types.SimpleNamespace(
        _atlas=object(),
        viewer=viewer,
        _show_region_meshes_cb=_DummyCheckBox(True),
        _show_region_seg_cb=_DummyCheckBox(True),
    )

    NeuronViewerWidget._update_custom_region_meshes(widget, ())
    NeuronViewerWidget._update_custom_region_segmentation(widget, ())

    remove_meshes.assert_called_once_with(viewer)
    remove_segmentation.assert_called_once_with(viewer)
    add_meshes.assert_not_called()
    add_segmentation.assert_not_called()


@pytest.mark.parametrize(
    ("action", "sync_method"),
    [
        ("meshes", "_sync_active_region_meshes"),
        ("segmentation", "_sync_active_region_segmentation"),
    ],
)
def test_pending_reference_action_uses_source_aware_sync(
    action: str,
    sync_method: str,
) -> None:
    checkbox = _DummyCheckBox(True)
    widget = types.SimpleNamespace(
        _reference_action_checkbox=lambda _action: checkbox,
        _sync_active_region_meshes=MagicMock(),
        _sync_active_region_segmentation=MagicMock(),
    )

    NeuronViewerWidget._complete_pending_reference_action(widget, action)

    getattr(widget, sync_method).assert_called_once_with()


def test_set_custom_region_hierarchy_populates_both_scope_selectors(
    monkeypatch,
) -> None:
    hierarchy = object()
    builder = MagicMock(return_value=hierarchy)
    monkeypatch.setitem(
        NeuronViewerWidget._set_custom_region_hierarchy_for_atlas.__globals__,
        "isocortex_layer_hierarchy_from_atlas",
        builder,
    )
    whole_selector = _DummyCustomRegionSelector()
    current_selector = _DummyCustomRegionSelector()
    atlas = object()
    widget = types.SimpleNamespace(
        _whole_parquet_custom_region_selector=whole_selector,
        _current_table_custom_region_selector=current_selector,
    )

    NeuronViewerWidget._set_custom_region_hierarchy_for_atlas(widget, atlas)

    builder.assert_called_once_with(atlas)
    assert whole_selector.hierarchies == [hierarchy]
    assert current_selector.hierarchies == [hierarchy]


def test_set_custom_region_hierarchy_reports_incompatible_atlas(monkeypatch) -> None:
    def fail(_atlas):
        raise ValueError("missing terminal Isocortex layers")

    monkeypatch.setitem(
        NeuronViewerWidget._set_custom_region_hierarchy_for_atlas.__globals__,
        "isocortex_layer_hierarchy_from_atlas",
        fail,
    )
    whole_selector = _DummyCustomRegionSelector()
    current_selector = _DummyCustomRegionSelector()
    widget = types.SimpleNamespace(
        _whole_parquet_custom_region_selector=whole_selector,
        _current_table_custom_region_selector=current_selector,
    )

    NeuronViewerWidget._set_custom_region_hierarchy_for_atlas(widget, object())

    expected = (
        "Custom Isocortex Layers are unavailable for the loaded atlas: "
        "missing terminal Isocortex layers"
    )
    assert whole_selector.clear_messages == [expected]
    assert current_selector.clear_messages == [expected]


def test_query_neurons_by_custom_region_uses_exact_terminal_ids_and_filters() -> None:
    result = pd.DataFrame(
        {
            "file_id": ["n1", "n2"],
            "neuron_id": ["N1", "N2"],
        }
    )
    db = MagicMock()
    db.has_column.return_value = True
    db.get_neurons_by_region_id.return_value = result
    widget = types.SimpleNamespace(
        _db=db,
        _atlas=object(),
        _whole_parquet_custom_region_selector=_DummyCustomRegionSelector(
            [302, 101, 302]
        ),
        _current_table_custom_region_selector=_DummyCustomRegionSelector([999]),
        _region_query_scope_combo=_DummyComboBox("Whole Parquet", data="whole"),
        _regions_status_label=_DummyLabel(),
        _neuron_table=types.SimpleNamespace(file_ids=lambda: ["n1"]),
        _populate_neuron_table=MagicMock(),
    )
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._query_neurons_by_custom_region(
        widget,
        node_types=(1, 3),
    )

    db.get_neurons_by_region_id.assert_called_once_with(
        [101, 302],
        soma_only=False,
        file_ids=None,
        node_types=(1, 3),
    )
    widget._populate_neuron_table.assert_called_once_with(
        result,
        preserve_existing=False,
    )
    assert "2 selected terminal custom regions within whole parquet" in (
        widget._regions_status_label.text
    )


def test_query_neurons_by_custom_region_preserves_current_table_scope() -> None:
    result = pd.DataFrame({"file_id": ["n2"], "neuron_id": ["N2"]})
    db = MagicMock()
    db.has_column.return_value = True
    db.get_neurons_by_region_id.return_value = result
    widget = types.SimpleNamespace(
        _db=db,
        _atlas=object(),
        _whole_parquet_custom_region_selector=_DummyCustomRegionSelector([101]),
        _current_table_custom_region_selector=_DummyCustomRegionSelector([202]),
        _region_query_scope_combo=_DummyComboBox("Current Table", data="current"),
        _regions_status_label=_DummyLabel(),
        _neuron_table=types.SimpleNamespace(file_ids=lambda: ["n2", "n1"]),
        _populate_neuron_table=MagicMock(),
    )
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._query_neurons_by_custom_region(widget, soma_only=True)

    db.get_neurons_by_region_id.assert_called_once_with(
        [202],
        soma_only=True,
        file_ids=["n2", "n1"],
        node_types=(1,),
    )
    widget._populate_neuron_table.assert_called_once_with(
        result,
        preserve_existing=True,
    )
    assert "within current table (from 2 input neurons)" in (
        widget._regions_status_label.text
    )


@pytest.mark.parametrize(
    ("atlas", "selector", "has_region_id", "expected_status"),
    [
        (
            None,
            _DummyCustomRegionSelector([101]),
            True,
            "Load a compatible Allen atlas",
        ),
        (
            object(),
            _DummyCustomRegionSelector(
                has_hierarchy=False,
                unavailable_message="Custom map is incompatible.",
            ),
            True,
            "Custom map is incompatible.",
        ),
        (
            object(),
            _DummyCustomRegionSelector(),
            True,
            "Select at least one terminal Custom Region.",
        ),
        (
            object(),
            _DummyCustomRegionSelector([101]),
            False,
            "Custom Regions require a region_id column",
        ),
    ],
)
def test_query_neurons_by_custom_region_reports_actionable_errors(
    atlas,
    selector,
    has_region_id,
    expected_status,
) -> None:
    db = MagicMock()
    db.has_column.return_value = has_region_id
    widget = types.SimpleNamespace(
        _db=db,
        _atlas=atlas,
        _whole_parquet_custom_region_selector=selector,
        _current_table_custom_region_selector=_DummyCustomRegionSelector(),
        _region_query_scope_combo=_DummyComboBox("Whole Parquet", data="whole"),
        _regions_status_label=_DummyLabel(),
    )
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._query_neurons_by_custom_region(widget)

    assert expected_status in widget._regions_status_label.text
    db.get_neurons_by_region_id.assert_not_called()


def test_query_neurons_by_custom_region_reports_zero_matches() -> None:
    db = MagicMock()
    db.has_column.return_value = True
    db.get_neurons_by_region_id.return_value = pd.DataFrame()
    widget = types.SimpleNamespace(
        _db=db,
        _atlas=object(),
        _whole_parquet_custom_region_selector=_DummyCustomRegionSelector([101]),
        _current_table_custom_region_selector=_DummyCustomRegionSelector(),
        _region_query_scope_combo=_DummyComboBox("Whole Parquet", data="whole"),
        _regions_status_label=_DummyLabel(),
        _neuron_table=types.SimpleNamespace(file_ids=lambda: []),
        _populate_neuron_table=MagicMock(),
    )
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._query_neurons_by_custom_region(widget)

    assert widget._regions_status_label.text == (
        "No neurons found with any node in 1 selected terminal custom region "
        "within whole parquet."
    )


def test_query_neurons_by_region_reports_union_details() -> None:
    result = pd.DataFrame(
        {
            "file_id": ["n1", "n2"],
            "neuron_id": ["N1", "N2"],
            "subject": ["s1", "s2"],
        }
    )
    widget = types.SimpleNamespace(
        _db=MagicMock(),
        _whole_parquet_region_selector=_DummyRegionSelector(
            direct_acronyms=["R1", "R2"],
            query_acronyms=["R1", "R2"],
            include_children=True,
        ),
        _current_table_region_selector=_DummyRegionSelector(),
        _region_query_scope_combo=_DummyComboBox("Whole Parquet", data="whole"),
        _regions_status_label=_DummyLabel(),
        _neuron_table=types.SimpleNamespace(file_ids=lambda: ["n1", "n2"]),
        _populate_neuron_table=MagicMock(),
    )
    widget._db.get_neurons_by_region.return_value = result
    _bind_region_query_scope_helpers(widget)

    NeuronViewerWidget._query_neurons_by_region(widget, soma_only=False)

    widget._db.get_neurons_by_region.assert_called_once_with(
        ["R1", "R2"],
        soma_only=False,
        file_ids=None,
        node_types=None,
    )
    widget._populate_neuron_table.assert_called_once_with(
        result,
        preserve_existing=False,
    )
    assert widget._regions_status_label.text == (
        "Found 2 neuron(s) with any node in selected atlas regions "
        "within whole parquet. Query: union of R1, R2; descendants: on."
    )


def test_refresh_neuron_table_summary_formats_counts_and_clusters() -> None:
    summary = types.SimpleNamespace(
        table_count=4,
        added_count=2,
        visible_count=3,
        cluster_counts=((1, 2), (3, 1), (None, 1)),
    )
    widget = types.SimpleNamespace(
        _neuron_table=types.SimpleNamespace(summary=lambda: summary),
        _neuron_table_summary_label=_DummyLabel(),
        _cluster_assignment_store=types.SimpleNamespace(active=None),
    )

    NeuronViewerWidget._refresh_neuron_table_summary(widget)

    assert widget._neuron_table_summary_label.text == (
        "In table: 4 | Added to scene: 2 | Visible: 3\n"
        "Clusters: Cluster 1: 2, Cluster 3: 1, Unclustered: 1"
    )


def test_on_cluster_colors_updated_sorts_and_refreshes_filters() -> None:
    neuron_table = types.SimpleNamespace(
        update_cluster_assignments=MagicMock(),
        update_colors=MagicMock(),
        sort_by_cluster=MagicMock(),
    )
    widget = types.SimpleNamespace(
        _neuron_table=neuron_table,
        _refresh_cluster_assignment_controls=MagicMock(),
        _active_assignment_color_map=MagicMock(return_value={}),
        _refresh_cluster_filter_controls=MagicMock(),
        _refresh_apply_existing_clusters_button=MagicMock(),
    )
    result = types.SimpleNamespace(neuron_ids=["n1"])
    color_map = {"n1": [0.1, 0.2, 0.3, 1.0]}

    NeuronViewerWidget._on_cluster_colors_updated(
        widget,
        result,
        color_map,
    )

    widget._refresh_cluster_assignment_controls.assert_called_once_with()
    neuron_table.update_cluster_assignments.assert_called_once_with(result)
    neuron_table.update_colors.assert_called_once_with(
        color_map,
        emit_signal=False,
    )
    neuron_table.sort_by_cluster.assert_called_once_with()
    widget._refresh_cluster_filter_controls.assert_called_once_with()
    widget._refresh_apply_existing_clusters_button.assert_called_once_with()


def test_refresh_cluster_filter_controls_preserves_valid_multi_selection() -> None:
    selection = ClusterFilterSelection({1, 3}, include_unclustered=True)
    combo = _DummyClusterFilterCombo(selection)
    applied: list[ClusterFilterSelection] = []
    table = types.SimpleNamespace(
        available_cluster_ids=lambda: [1, 2, 3],
        has_unclustered_entries=lambda: True,
        apply_cluster_filter=lambda selected: applied.append(selected),
        get_visibility_map=lambda: {"n1": True},
    )
    widget = types.SimpleNamespace(
        _cluster_filter_combo=combo,
        _neuron_table=table,
        _hide_others_btn=_DummyButton(),
        _recolor_cluster_btn=_DummyButton(),
        _show_all_btn=_DummyButton(),
    )
    _bind_cluster_filter_helpers(widget)

    NeuronViewerWidget._refresh_cluster_filter_controls(widget)

    assert combo.calls == [
        {
            "cluster_ids": [1, 2, 3],
            "include_unclustered": True,
            "selection": selection,
        }
    ]
    assert combo.selection == selection
    assert applied == [selection]
    assert widget._hide_others_btn.enabled is True
    assert widget._recolor_cluster_btn.enabled is True
    assert widget._show_all_btn.enabled is True


def test_refresh_cluster_filter_controls_falls_back_to_all_when_invalid() -> None:
    combo = _DummyClusterFilterCombo(
        ClusterFilterSelection({9}, include_unclustered=True)
    )
    applied: list[ClusterFilterSelection] = []
    table = types.SimpleNamespace(
        available_cluster_ids=lambda: [1, 2],
        has_unclustered_entries=lambda: False,
        apply_cluster_filter=lambda selected: applied.append(selected),
        get_visibility_map=lambda: {"n1": True},
    )
    widget = types.SimpleNamespace(
        _cluster_filter_combo=combo,
        _neuron_table=table,
        _hide_others_btn=_DummyButton(),
        _recolor_cluster_btn=_DummyButton(),
        _show_all_btn=_DummyButton(),
    )
    _bind_cluster_filter_helpers(widget)

    NeuronViewerWidget._refresh_cluster_filter_controls(widget)

    assert combo.selection == ClusterFilterSelection()
    assert applied == [ClusterFilterSelection()]
    assert widget._hide_others_btn.enabled is False
    assert widget._recolor_cluster_btn.enabled is False
    assert widget._show_all_btn.enabled is True


def test_cluster_filter_change_preserves_manual_heatmap_filter() -> None:
    selection = ClusterFilterSelection({1})
    manual_combo = _DummyMutableComboBox()
    manual_combo.addItem("All Manual Heatmaps", None)
    manual_combo.addItem("alpha Heatmap", ("alpha Heatmap", ("n1", "n2")))
    manual_combo.setCurrentIndex(1)
    table = types.SimpleNamespace(
        apply_filters=MagicMock(),
        get_visibility_map=lambda: {"n1": True},
    )
    widget = types.SimpleNamespace(
        _cluster_filter_combo=_DummyClusterFilterCombo(selection),
        _manual_heatmap_combo=manual_combo,
        _neuron_table=table,
        _hide_others_btn=_DummyButton(),
        _recolor_cluster_btn=_DummyButton(),
        _show_all_btn=_DummyButton(),
    )
    _bind_cluster_filter_helpers(widget)

    NeuronViewerWidget._on_cluster_filter_changed(widget)

    table.apply_filters.assert_called_once_with(selection, ("n1", "n2"))
    assert widget._hide_others_btn.enabled is True
    assert widget._recolor_cluster_btn.enabled is True
    assert widget._show_all_btn.enabled is True


def test_cluster_selection_actions_apply_full_selection() -> None:
    selection = ClusterFilterSelection({1, 2}, include_unclustered=True)
    table = types.SimpleNamespace(
        hide_all_not_in_cluster=MagicMock(),
        recolor_cluster_distinct=MagicMock(),
    )
    widget = types.SimpleNamespace(
        _cluster_filter_combo=_DummyClusterFilterCombo(selection),
        _neuron_table=table,
    )
    _bind_cluster_filter_helpers(widget)

    NeuronViewerWidget._hide_not_in_selected_cluster(widget)
    NeuronViewerWidget._recolor_selected_cluster(widget)

    table.hide_all_not_in_cluster.assert_called_once_with(selection)
    table.recolor_cluster_distinct.assert_called_once_with(
        selection,
        gray_others=True,
    )


def test_refresh_apply_existing_clusters_button_hidden_without_overlap() -> None:
    button = _DummyButton()
    widget = types.SimpleNamespace(
        _analysis_tab=types.SimpleNamespace(
            has_cached_clusters_for_current_table=lambda: False
        ),
        _apply_existing_clusters_btn=button,
    )

    NeuronViewerWidget._refresh_apply_existing_clusters_button(widget)

    assert button.visible is False
    assert button.enabled is False


def test_refresh_apply_existing_clusters_button_visible_with_overlap() -> None:
    button = _DummyButton()
    widget = types.SimpleNamespace(
        _analysis_tab=types.SimpleNamespace(
            has_cached_clusters_for_current_table=lambda: True
        ),
        _apply_existing_clusters_btn=button,
    )

    NeuronViewerWidget._refresh_apply_existing_clusters_button(widget)

    assert button.visible is True
    assert button.enabled is True


def test_sync_after_neuron_table_membership_change_refreshes_apply_existing_clusters_button() -> (
    None
):
    widget = types.SimpleNamespace(
        _last_soma_selection={"n1"},
        _refresh_cluster_filter_controls=MagicMock(),
        _refresh_apply_existing_clusters_button=MagicMock(),
        _refresh_neuron_table_summary=MagicMock(),
        _neuron_table=types.SimpleNamespace(get_selected_file_ids=lambda: []),
        _current_neuron_layers=[],
        _highlighted_file_ids={"n1"},
    )

    NeuronViewerWidget._sync_after_neuron_table_membership_change(widget)

    widget._refresh_cluster_filter_controls.assert_called_once_with()
    widget._refresh_apply_existing_clusters_button.assert_called_once_with()
    widget._refresh_neuron_table_summary.assert_called_once_with()


def test_apply_existing_clusters_from_analysis_updates_render_status() -> None:
    summary = types.SimpleNamespace(
        matched_table_count=3,
        rendered_count=2,
        colored_count=2,
        gray_count=1,
    )
    widget = types.SimpleNamespace(
        _analysis_tab=types.SimpleNamespace(apply_cluster_colors=lambda: summary),
        _render_status_label=_DummyLabel(),
        _refresh_apply_existing_clusters_button=MagicMock(),
    )

    NeuronViewerWidget._apply_existing_clusters_from_analysis(widget)

    assert widget._render_status_label.text == (
        "Applied cached cluster data to 3 table neuron(s). "
        "Recolored 2/2 rendered neuron(s). 1 shown in gray."
    )


def test_apply_existing_clusters_from_analysis_no_overlap_only_refreshes_button() -> (
    None
):
    summary = types.SimpleNamespace(
        matched_table_count=0,
        rendered_count=0,
        colored_count=0,
        gray_count=0,
    )
    widget = types.SimpleNamespace(
        _analysis_tab=types.SimpleNamespace(apply_cluster_colors=lambda: summary),
        _render_status_label=_DummyLabel(),
        _refresh_apply_existing_clusters_button=MagicMock(),
    )

    NeuronViewerWidget._apply_existing_clusters_from_analysis(widget)

    assert widget._render_status_label.text == ""
    widget._refresh_apply_existing_clusters_button.assert_called_once_with()


def test_remove_unselected_from_table_keeps_selection_and_preserves_scene_state() -> (
    None
):
    table = types.SimpleNamespace(
        _entries={
            "n1": types.SimpleNamespace(
                color=[1.0, 0.0, 0.0, 1.0],
                visible=True,
            ),
            "n2": types.SimpleNamespace(
                color=[0.2, 0.3, 0.4, 1.0],
                visible=True,
            ),
            "n3": types.SimpleNamespace(
                color=[0.5, 0.6, 0.7, 1.0],
                visible=False,
            ),
        },
    )

    def _remove_file_ids(file_ids) -> None:
        for file_id in file_ids:
            table._entries.pop(file_id, None)

    table.file_ids = lambda: list(table._entries.keys())
    table.get_selected_file_ids = MagicMock(return_value=["n1"])
    table.remove_file_ids = MagicMock(side_effect=_remove_file_ids)
    table.set_added_file_ids = MagicMock()
    table.select_file_ids = MagicMock()

    widget = types.SimpleNamespace(
        _highlighted_file_ids=None,
        _current_neuron_layers=[object()],
        _neuron_table=table,
        _last_soma_selection={"n2"},
        _refresh_cluster_filter_controls=MagicMock(),
        _refresh_neuron_table_summary=MagicMock(),
        _render_status_label=_DummyLabel(),
        _regions_status_label=_DummyLabel(),
        _scene_render_modes={"n1": "full", "n2": "full"},
        _scene_display_state={},
        _update_layer_colors=MagicMock(),
    )
    _bind_table_membership_helpers(widget)

    NeuronViewerWidget._remove_unselected_from_table(widget)

    table.remove_file_ids.assert_called_once_with(["n2", "n3"])
    table.set_added_file_ids.assert_called_once_with({"n1", "n2"})
    table.select_file_ids.assert_called_once_with(["n1"])
    assert list(table._entries) == ["n1"]
    assert widget._scene_display_state == {
        "n2": {"color": [0.2, 0.3, 0.4, 1.0], "visible": True}
    }
    assert widget._highlighted_file_ids == {"n1"}
    assert widget._last_soma_selection == set()
    assert widget._render_status_label.text == (
        "Removed 2 unselected neuron(s) from the table."
    )
    assert widget._regions_status_label.text == (
        "Removed 2 unselected neuron(s) from the table."
    )


def test_remove_unselected_from_table_requires_selection() -> None:
    table = types.SimpleNamespace(
        get_selected_file_ids=MagicMock(return_value=[]),
        remove_file_ids=MagicMock(),
        set_added_file_ids=MagicMock(),
        select_file_ids=MagicMock(),
    )
    widget = types.SimpleNamespace(
        _neuron_table=table,
        _render_status_label=_DummyLabel(),
        _regions_status_label=_DummyLabel(),
    )

    NeuronViewerWidget._remove_unselected_from_table(widget)

    table.remove_file_ids.assert_not_called()
    table.set_added_file_ids.assert_not_called()
    table.select_file_ids.assert_not_called()
    assert widget._render_status_label.text == (
        "Select at least one neuron row to keep in the table."
    )


def test_remove_unselected_from_table_noops_when_all_rows_selected() -> None:
    table = types.SimpleNamespace(
        _entries={
            "n1": types.SimpleNamespace(color=[1.0, 0.0, 0.0, 1.0], visible=True),
            "n2": types.SimpleNamespace(color=[0.0, 1.0, 0.0, 1.0], visible=True),
        },
    )
    table.file_ids = lambda: list(table._entries.keys())
    table.get_selected_file_ids = MagicMock(return_value=["n1", "n2"])
    table.remove_file_ids = MagicMock()
    table.set_added_file_ids = MagicMock()
    table.select_file_ids = MagicMock()
    widget = types.SimpleNamespace(
        _neuron_table=table,
        _render_status_label=_DummyLabel(),
        _regions_status_label=_DummyLabel(),
    )
    widget._current_table_file_ids = types.MethodType(
        NeuronViewerWidget._current_table_file_ids,
        widget,
    )

    NeuronViewerWidget._remove_unselected_from_table(widget)

    table.remove_file_ids.assert_not_called()
    table.set_added_file_ids.assert_not_called()
    table.select_file_ids.assert_not_called()
    assert widget._render_status_label.text == (
        "All table neurons are selected; no unselected neurons to remove."
    )
    assert widget._regions_status_label.text == (
        "All table neurons are selected; no unselected neurons to remove."
    )


def test_clear_neuron_table_preserves_scene_render_modes() -> None:
    table = types.SimpleNamespace(
        _entries={
            "n1": types.SimpleNamespace(color=[1.0, 0.0, 0.0, 1.0], visible=True)
        },
    )
    table.clear = MagicMock(side_effect=table._entries.clear)
    table.set_added_file_ids = MagicMock()
    table.get_selected_file_ids = MagicMock(return_value=[])
    table.file_ids = lambda: list(table._entries.keys())

    widget = types.SimpleNamespace(
        _highlighted_file_ids=None,
        _current_neuron_layers=[object()],
        _neuron_table=table,
        _last_soma_selection={"n1"},
        _refresh_cluster_filter_controls=MagicMock(),
        _refresh_neuron_table_summary=MagicMock(),
        _render_status_label=_DummyLabel(),
        _regions_status_label=_DummyLabel(),
        _scene_render_modes={"n1": "full"},
        _scene_display_state={},
        _update_layer_colors=MagicMock(),
    )
    _bind_table_membership_helpers(widget)

    NeuronViewerWidget._clear_neuron_table(widget)

    widget._neuron_table.clear.assert_called_once_with()
    widget._refresh_cluster_filter_controls.assert_called_once_with()
    widget._refresh_neuron_table_summary.assert_called_once_with()
    widget._update_layer_colors.assert_called_once_with({"n1": [1.0, 0.0, 0.0, 1.0]})
    assert widget._scene_render_modes == {"n1": "full"}
    assert widget._last_soma_selection == set()
    assert widget._render_status_label.text == "Cleared neuron table."
    assert widget._regions_status_label.text == "Cleared neuron table."


def test_clear_neuron_table_clears_highlight_without_recoloring_scene_to_gray() -> None:
    entry = types.SimpleNamespace(color=[0.2, 0.3, 0.4, 1.0], visible=True)
    table = types.SimpleNamespace(_entries={"n1": entry})
    table.clear = MagicMock(side_effect=table._entries.clear)
    table.set_added_file_ids = MagicMock()
    table.get_selected_file_ids = MagicMock(return_value=[])
    table.file_ids = lambda: list(table._entries.keys())
    widget = types.SimpleNamespace(
        _highlighted_file_ids={"n1"},
        _current_neuron_layers=[object()],
        _neuron_table=table,
        _last_soma_selection=set(),
        _refresh_cluster_filter_controls=MagicMock(),
        _refresh_neuron_table_summary=MagicMock(),
        _render_status_label=_DummyLabel(),
        _regions_status_label=_DummyLabel(),
        _update_layer_colors=MagicMock(),
        _scene_render_modes={"n1": "full"},
        _scene_display_state={},
    )
    _bind_table_membership_helpers(widget)

    NeuronViewerWidget._clear_neuron_table(widget)

    widget._update_layer_colors.assert_called_once_with({"n1": [0.2, 0.3, 0.4, 1.0]})
    assert widget._highlighted_file_ids is None


def test_populate_neuron_table_preserves_rendered_color_when_subset_filter_removes_row() -> (
    None
):
    entry = types.SimpleNamespace(color=[0.2, 0.3, 0.4, 1.0], visible=True)
    table = types.SimpleNamespace(_entries={"n1": entry})

    def _retain_file_ids(file_ids) -> None:
        keep = set(file_ids)
        table._entries = {
            fid: value for fid, value in table._entries.items() if fid in keep
        }

    table.retain_file_ids = _retain_file_ids
    table.set_added_file_ids = MagicMock()
    table.get_selected_file_ids = MagicMock(return_value=[])
    table.file_ids = lambda: list(table._entries.keys())

    widget = types.SimpleNamespace(
        _neuron_table=table,
        _scene_render_modes={"n1": "full"},
        _scene_display_state={},
        _current_neuron_layers=[object()],
        _highlighted_file_ids=None,
        _last_soma_selection=set(),
        _refresh_cluster_filter_controls=MagicMock(),
        _refresh_neuron_table_summary=MagicMock(),
        _update_layer_colors=MagicMock(),
    )
    _bind_table_membership_helpers(widget)

    empty_result = pd.DataFrame(columns=["file_id", "neuron_id", "subject"])

    NeuronViewerWidget._populate_neuron_table(
        widget,
        empty_result,
        preserve_existing=True,
    )

    assert table._entries == {}
    assert widget._scene_render_modes == {"n1": "full"}
    assert widget._scene_display_state == {
        "n1": {"color": [0.2, 0.3, 0.4, 1.0], "visible": True}
    }
    widget._update_layer_colors.assert_called_once_with({"n1": [0.2, 0.3, 0.4, 1.0]})


def test_populate_neuron_table_does_not_require_subject_column() -> None:
    populated_file_ids: list[object] = []
    table = types.SimpleNamespace(_entries={})
    table.populate = MagicMock(side_effect=populated_file_ids.extend)
    table.set_added_file_ids = MagicMock()
    table.get_selected_file_ids = MagicMock(return_value=[])
    table.file_ids = lambda: list(populated_file_ids)

    widget = types.SimpleNamespace(
        _neuron_table=table,
        _scene_render_modes={},
        _scene_display_state={},
        _current_neuron_layers=[],
        _highlighted_file_ids=None,
        _last_soma_selection=set(),
        _refresh_cluster_filter_controls=MagicMock(),
        _refresh_neuron_table_summary=MagicMock(),
        _update_layer_colors=MagicMock(),
    )
    _bind_table_membership_helpers(widget)

    result = pd.DataFrame({"file_id": ["n1", "n2"]})

    NeuronViewerWidget._populate_neuron_table(widget, result)

    table.populate.assert_called_once_with(["n1", "n2"])


def test_add_heatmap_menu_exposes_single_and_individual_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        NeuronViewerWidget._configure_selected_heatmap_menu.__globals__,
        "QMenu",
        _DummyMenu,
    )
    single = MagicMock()
    individual = MagicMock()
    widget = types.SimpleNamespace(
        _add_selected_heatmap_btn=_DummyMenuButton("Add Heatmap"),
        _add_selected_neurons_heatmap=single,
        _add_selected_neurons_individual_heatmaps=individual,
    )

    NeuronViewerWidget._configure_selected_heatmap_menu(widget)

    assert widget._add_selected_heatmap_btn.text == "Add Heatmap"
    assert [action.text for action in widget._add_selected_heatmap_menu.actions] == [
        "Single Heatmap",
        "Individual Heatmaps",
    ]
    widget._add_single_heatmap_action.triggered.emit(False)
    widget._add_individual_heatmaps_action.triggered.emit(False)
    single.assert_called_once_with(False)
    individual.assert_called_once_with(False)


def _selected_heatmap_action_widget(table, atlas=None):
    widget = types.SimpleNamespace(
        _db=types.SimpleNamespace(parquet_path=Path("/tmp/neurons.parquet")),
        _atlas=atlas
        or types.SimpleNamespace(
            annotation=np.zeros((2, 2, 2), dtype=np.uint8),
            atlas_name="fake_atlas",
        ),
        _neuron_table=table,
        _render_status_label=_DummyLabel(),
        _selected_heatmap_running=lambda: False,
        _start_selected_neuron_heatmap_requests=MagicMock(),
    )
    widget._selected_neuron_heatmap_selection = types.MethodType(
        NeuronViewerWidget._selected_neuron_heatmap_selection,
        widget,
    )
    widget._individual_heatmap_estimated_bytes = (
        NeuronViewerWidget._individual_heatmap_estimated_bytes
    )
    widget._selected_neuron_colors_are_monochrome = (
        NeuronViewerWidget._selected_neuron_colors_are_monochrome
    )
    widget._confirm_large_individual_heatmap_request = MagicMock(return_value=True)
    return widget


def test_single_heatmap_action_snapshots_sorted_selection_in_one_request() -> None:
    table = types.SimpleNamespace(get_selected_file_ids=lambda: ["n2", "n1"])
    widget = _selected_heatmap_action_widget(table)

    NeuronViewerWidget._add_selected_neurons_heatmap(widget)

    requests = widget._start_selected_neuron_heatmap_requests.call_args.args[0]
    assert len(requests) == 1
    assert requests[0].file_ids == ("n1", "n2")
    assert requests[0].creation_mode == "single"
    assert requests[0].color is None


def test_individual_heatmaps_recolor_only_monochrome_selected_neurons() -> None:
    initial_colors = {
        "n1": [0.5, 0.5, 0.5, 1.0],
        "n2": [0.5, 0.5, 0.5, 0.4],
        "n3": [0.2, 0.3, 0.4, 1.0],
    }
    table = types.SimpleNamespace(
        get_selected_file_ids=lambda: ["n2", "n1"],
        get_color=lambda file_id: initial_colors[file_id],
        update_colors=MagicMock(),
    )
    widget = _selected_heatmap_action_widget(table)

    NeuronViewerWidget._add_selected_neurons_individual_heatmaps(widget)

    expected_colors = distinct_colors_for_file_ids(["n1", "n2"])
    table.update_colors.assert_called_once_with(expected_colors)
    assert "n3" not in table.update_colors.call_args.args[0]
    requests = widget._start_selected_neuron_heatmap_requests.call_args.args[0]
    assert [request.file_ids for request in requests] == [("n1",), ("n2",)]
    assert [request.creation_mode for request in requests] == [
        "individual",
        "individual",
    ]
    assert [request.color for request in requests] == [
        tuple(expected_colors["n1"]),
        tuple(expected_colors["n2"]),
    ]


@pytest.mark.parametrize(
    "selected_file_ids, colors",
    [
        (["n1"], {"n1": [0.5, 0.5, 0.5, 1.0]}),
        (
            ["n1", "n2"],
            {
                "n1": [1.0, 0.0, 0.0, 1.0],
                "n2": [0.0, 1.0, 0.0, 1.0],
            },
        ),
    ],
)
def test_individual_heatmaps_preserve_singleton_or_distinct_colors(
    selected_file_ids: list[str],
    colors: dict[str, list[float]],
) -> None:
    table = types.SimpleNamespace(
        get_selected_file_ids=lambda: list(selected_file_ids),
        get_color=lambda file_id: colors[file_id],
        update_colors=MagicMock(),
    )
    widget = _selected_heatmap_action_widget(table)

    NeuronViewerWidget._add_selected_neurons_individual_heatmaps(widget)

    table.update_colors.assert_not_called()
    requests = widget._start_selected_neuron_heatmap_requests.call_args.args[0]
    assert [request.color for request in requests] == [
        tuple(colors[file_id]) for file_id in sorted(selected_file_ids)
    ]


def test_large_individual_heatmap_cancellation_changes_nothing() -> None:
    table = types.SimpleNamespace(
        get_selected_file_ids=lambda: ["n1", "n2"],
        get_color=MagicMock(),
        update_colors=MagicMock(),
    )
    atlas = types.SimpleNamespace(
        annotation=types.SimpleNamespace(shape=(1024, 1024, 129)),
        atlas_name="fake_atlas",
    )
    widget = _selected_heatmap_action_widget(table, atlas=atlas)
    widget._confirm_large_individual_heatmap_request.return_value = False

    NeuronViewerWidget._add_selected_neurons_individual_heatmaps(widget)

    widget._confirm_large_individual_heatmap_request.assert_called_once()
    table.get_color.assert_not_called()
    table.update_colors.assert_not_called()
    widget._start_selected_neuron_heatmap_requests.assert_not_called()
    assert "cancelled" in widget._render_status_label.text


def test_individual_heatmap_memory_estimate_uses_float32_atlas_volumes() -> None:
    atlas = types.SimpleNamespace(annotation=np.zeros((2, 3, 4), dtype=np.uint8))

    estimated = NeuronViewerWidget._individual_heatmap_estimated_bytes(atlas, 5)

    assert estimated == 2 * 3 * 4 * 4 * 5
    warning = NeuronViewerWidget._individual_heatmap_memory_warning_text(
        5,
        2 * 1024**3,
    )
    assert "5 individual heatmaps" in warning
    assert "2.00 GiB" in warning


def _selected_heatmap_request(
    file_id: str,
    *,
    creation_mode: str = "individual",
):
    request_type = (
        NeuronViewerWidget._start_selected_neuron_heatmap_requests.__globals__[
            "_SelectedNeuronHeatmapRequest"
        ]
    )
    return request_type(
        file_ids=(file_id,),
        creation_mode=creation_mode,
        color=(0.1, 0.2, 0.3, 1.0) if creation_mode == "individual" else None,
    )


def test_selected_heatmap_request_queue_initializes_and_starts_first() -> None:
    requests = [_selected_heatmap_request("n1"), _selected_heatmap_request("n2")]
    widget = types.SimpleNamespace(
        _render_progress=_DummyProgressBar(),
        _update_selected_neuron_heatmap_controls=MagicMock(),
        _start_next_selected_neuron_heatmap_request=MagicMock(),
    )

    NeuronViewerWidget._start_selected_neuron_heatmap_requests(widget, requests)

    assert widget._selected_heatmap_pending_requests == requests
    assert widget._selected_heatmap_completed_requests == []
    assert widget._selected_heatmap_total == 2
    assert widget._selected_heatmap_index == 0
    assert widget._selected_heatmap_failed is False
    assert widget._render_progress.visible is True
    widget._update_selected_neuron_heatmap_controls.assert_called_once_with()
    widget._start_next_selected_neuron_heatmap_request.assert_called_once_with()


def test_selected_heatmap_button_stays_disabled_while_queue_is_pending() -> None:
    widget = types.SimpleNamespace(
        _selected_heatmap_thread=None,
        _selected_heatmap_current_request=None,
        _selected_heatmap_pending_requests=[_selected_heatmap_request("n1")],
        _add_selected_heatmap_btn=_DummyButton(),
    )
    widget._selected_heatmap_running = types.MethodType(
        NeuronViewerWidget._selected_heatmap_running,
        widget,
    )

    NeuronViewerWidget._update_selected_neuron_heatmap_controls(widget)
    assert widget._add_selected_heatmap_btn.enabled is False

    widget._selected_heatmap_pending_requests = []
    NeuronViewerWidget._update_selected_neuron_heatmap_controls(widget)
    assert widget._add_selected_heatmap_btn.enabled is True


def test_selected_heatmap_queue_starts_singleton_requests_in_order() -> None:
    first = _selected_heatmap_request("n1")
    second = _selected_heatmap_request("n2")
    widget = types.SimpleNamespace(
        _selected_heatmap_pending_requests=[first, second],
        _selected_heatmap_completed_requests=[],
        _start_selected_neuron_heatmap=MagicMock(),
    )

    NeuronViewerWidget._start_next_selected_neuron_heatmap_request(widget)

    assert widget._selected_heatmap_current_request is first
    assert widget._selected_heatmap_request_file_ids == ("n1",)
    assert widget._selected_heatmap_pending_requests == [second]
    assert widget._selected_heatmap_index == 1
    widget._start_selected_neuron_heatmap.assert_called_once_with(first)


def test_selected_heatmap_cleanup_advances_queue_before_enabling_button() -> None:
    thread = object()
    worker = object()
    pending = _selected_heatmap_request("n2")
    widget = types.SimpleNamespace(
        _selected_heatmap_thread=thread,
        _selected_heatmap_worker=worker,
        _selected_heatmap_request_file_ids=("n1",),
        _selected_heatmap_current_request=_selected_heatmap_request("n1"),
        _selected_heatmap_pending_requests=[pending],
        _selected_heatmap_completed_requests=[_selected_heatmap_request("n1")],
        _selected_heatmap_failed=False,
        _selected_heatmap_total=2,
        _selected_heatmap_index=1,
        _render_progress=_DummyProgressBar(),
        _render_status_label=_DummyLabel(),
        _start_next_selected_neuron_heatmap_request=MagicMock(),
        _update_selected_neuron_heatmap_controls=MagicMock(),
    )

    NeuronViewerWidget._cleanup_selected_heatmap_thread(widget, thread, worker)

    assert widget._selected_heatmap_thread is None
    assert widget._selected_heatmap_worker is None
    assert widget._selected_heatmap_current_request is None
    widget._start_next_selected_neuron_heatmap_request.assert_called_once_with()
    widget._update_selected_neuron_heatmap_controls.assert_not_called()


def test_selected_heatmap_error_stops_pending_queue_and_reports_partial_result() -> (
    None
):
    widget = types.SimpleNamespace(
        _selected_heatmap_pending_requests=[_selected_heatmap_request("n3")],
        _selected_heatmap_completed_requests=[_selected_heatmap_request("n1")],
        _selected_heatmap_total=3,
        _render_progress=_DummyProgressBar(),
        _render_status_label=_DummyLabel(),
    )

    NeuronViewerWidget._on_selected_neuron_heatmap_error(widget, "query failed")

    assert widget._selected_heatmap_failed is True
    assert widget._selected_heatmap_pending_requests == []
    assert widget._render_progress.visible is False
    assert widget._render_status_label.text == (
        "Error after 1/3 individual heatmaps: query failed"
    )


def test_selected_heatmap_cleanup_finishes_successful_batch() -> None:
    thread = object()
    worker = object()
    completed = [_selected_heatmap_request("n1"), _selected_heatmap_request("n2")]
    widget = types.SimpleNamespace(
        _selected_heatmap_thread=thread,
        _selected_heatmap_worker=worker,
        _selected_heatmap_request_file_ids=("n2",),
        _selected_heatmap_current_request=completed[-1],
        _selected_heatmap_pending_requests=[],
        _selected_heatmap_completed_requests=completed,
        _selected_heatmap_failed=False,
        _selected_heatmap_total=2,
        _selected_heatmap_index=2,
        _render_progress=_DummyProgressBar(),
        _render_status_label=_DummyLabel(),
        _update_selected_neuron_heatmap_controls=MagicMock(),
    )

    NeuronViewerWidget._cleanup_selected_heatmap_thread(widget, thread, worker)

    assert widget._selected_heatmap_completed_requests == []
    assert widget._selected_heatmap_total == 0
    assert widget._render_progress.visible is False
    assert (
        widget._render_status_label.text == "Added 2 individual heatmaps to the scene."
    )
    widget._update_selected_neuron_heatmap_controls.assert_called_once_with()


def test_selected_neuron_heatmap_layer_name_uses_greek_identifiers() -> None:
    viewer = _DummyViewer(ndisplay=3)
    viewer.layers.extend(
        [
            types.SimpleNamespace(name="alpha Heatmap"),
            types.SimpleNamespace(name="beta Heatmap"),
            types.SimpleNamespace(name="Cluster 1 Heatmap"),
        ]
    )
    widget = types.SimpleNamespace(viewer=viewer)
    _bind_manual_heatmap_helpers(widget)
    widget._next_manual_heatmap_identifier = types.MethodType(
        NeuronViewerWidget._next_manual_heatmap_identifier,
        widget,
    )

    layer_name = NeuronViewerWidget._selected_neuron_heatmap_layer_name(
        widget,
        ["n1"],
    )

    assert layer_name == "gamma Heatmap"


def test_selected_neuron_heatmap_layer_name_continues_after_omega() -> None:
    viewer = _DummyViewer(ndisplay=3)
    viewer.layers.extend(
        [
            types.SimpleNamespace(
                name=f"{NeuronViewerWidget._greek_heatmap_identifier(index)} Heatmap"
            )
            for index in range(24)
        ]
    )
    widget = types.SimpleNamespace(viewer=viewer)
    _bind_manual_heatmap_helpers(widget)
    widget._next_manual_heatmap_identifier = types.MethodType(
        NeuronViewerWidget._next_manual_heatmap_identifier,
        widget,
    )

    layer_name = NeuronViewerWidget._selected_neuron_heatmap_layer_name(
        widget,
        ["n1"],
    )

    assert layer_name == "alpha alpha Heatmap"


def test_add_selected_neuron_heatmap_layer_sets_single_selection_metadata() -> None:
    viewer = _DummyViewer(ndisplay=3)
    widget = types.SimpleNamespace(
        viewer=viewer,
        _db=types.SimpleNamespace(parquet_path=Path("/tmp/neurons.parquet")),
        _atlas=types.SimpleNamespace(atlas_name="fake_atlas"),
        _opacity_slider=_DummyValueControl(80),
    )
    _bind_manual_heatmap_helpers(widget)
    widget._next_manual_heatmap_identifier = types.MethodType(
        NeuronViewerWidget._next_manual_heatmap_identifier,
        widget,
    )
    widget._selected_neuron_heatmap_layer_name = types.MethodType(
        NeuronViewerWidget._selected_neuron_heatmap_layer_name,
        widget,
    )
    widget._current_atlas_name = types.MethodType(
        NeuronViewerWidget._current_atlas_name,
        widget,
    )

    layer = NeuronViewerWidget._add_selected_neuron_heatmap_layer(
        widget,
        np.array([[[0.0, 5.0]]], dtype=np.float32),
        ["n1"],
    )

    assert layer.name == "alpha Heatmap"
    assert layer.contrast_limits == (0.0, 5.0)
    assert layer.metadata["heatmap_kind"] == "selected_neurons"
    assert layer.metadata["manual_heatmap_id"] == "alpha"
    assert layer.metadata["atlas_name"] == "fake_atlas"
    assert layer.metadata["source_path"] == str(Path("/tmp/neurons.parquet"))
    assert layer.metadata["file_ids"] == ["n1"]
    assert layer.metadata["selection_count"] == 1
    assert layer.metadata["heatmap_source"] is True
    assert layer.metadata["heatmap_native_grid"] is True
    assert layer.metadata["heatmap_creation_mode"] == "single"
    assert layer.metadata["heatmap_autocontrast_policy"] == "stable_full_volume"
    assert layer.colormap == "hot"


def test_add_individual_heatmap_layer_uses_neuron_color_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_colormaps(monkeypatch)
    viewer = _DummyViewer(ndisplay=3)
    widget = types.SimpleNamespace(
        viewer=viewer,
        _db=types.SimpleNamespace(parquet_path=Path("/tmp/neurons.parquet")),
        _atlas=types.SimpleNamespace(atlas_name="fake_atlas"),
        _opacity_slider=_DummyValueControl(60),
    )
    _bind_manual_heatmap_helpers(widget)
    widget._next_manual_heatmap_identifier = types.MethodType(
        NeuronViewerWidget._next_manual_heatmap_identifier,
        widget,
    )
    widget._selected_neuron_heatmap_layer_name = types.MethodType(
        NeuronViewerWidget._selected_neuron_heatmap_layer_name,
        widget,
    )
    widget._current_atlas_name = types.MethodType(
        NeuronViewerWidget._current_atlas_name,
        widget,
    )
    color = (0.1, 0.2, 0.3, 0.4)

    layer = NeuronViewerWidget._add_selected_neuron_heatmap_layer(
        widget,
        np.array([[[0.0, 5.0]]], dtype=np.float32),
        ["n1"],
        creation_mode="individual",
        color=color,
    )

    assert layer.name == "alpha Heatmap"
    assert layer.colormap.kwargs == {
        "colors": [[0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 1.0]],
        "name": "manual_heatmap_alpha",
    }
    assert layer.opacity == 0.6
    assert layer.contrast_limits == (0.0, 1.0)
    assert layer.metadata["file_ids"] == ["n1"]
    assert layer.metadata["selection_count"] == 1
    assert layer.metadata["heatmap_creation_mode"] == "individual"
    assert layer.metadata["heatmap_contrast_limits"] == (0.0, 1.0)
    assert layer.metadata["heatmap_autocontrast_policy"] == "stable_20_percent_max"
    assert layer.metadata["color"] == list(color)


def test_current_selected_neuron_heatmap_layers_ignores_analysis_heatmaps() -> None:
    viewer = _DummyViewer(ndisplay=3)
    viewer.layers.extend(
        [
            types.SimpleNamespace(
                name="alpha Heatmap",
                metadata={
                    "heatmap_kind": "selected_neurons",
                    "file_ids": ["n2"],
                    "manual_heatmap_id": "alpha",
                },
            ),
            types.SimpleNamespace(
                name="beta Heatmap",
                metadata={
                    "heatmap_kind": "selected_neurons",
                    "file_ids": ["n1", "n2"],
                    "manual_heatmap_id": "beta",
                },
            ),
            types.SimpleNamespace(
                name="Cluster 1 Heatmap",
                metadata={
                    "heatmap_kind": "analysis",
                    "heatmap_cluster": 1,
                    "file_ids": ["n3"],
                },
            ),
            types.SimpleNamespace(
                name="Points Heatmap: imported",
                metadata={
                    "heatmap_kind": "point_import",
                    "file_ids": ["n4"],
                },
            ),
        ]
    )
    widget = types.SimpleNamespace(viewer=viewer)
    _bind_manual_heatmap_helpers(widget)

    layer_names_by_file_id = (
        NeuronViewerWidget._current_selected_neuron_heatmap_layers_by_file_id(widget)
    )

    assert layer_names_by_file_id == {
        "n1": ("beta Heatmap",),
        "n2": ("alpha Heatmap", "beta Heatmap"),
    }


def _make_layer_name_sync_widget(viewer: _DummyViewer, table) -> object:
    widget = types.SimpleNamespace(
        viewer=viewer,
        _neuron_table=table,
        _manual_heatmap_combo=None,
        _layer_name_event_connections={},
        _refresh_heatmap_layer_list=MagicMock(),
        _refresh_histogram_layer_list=MagicMock(),
        _refresh_mask_layer_options=MagicMock(),
        _update_tools_controls=MagicMock(),
        _update_histogram_controls=MagicMock(),
    )
    _bind_manual_heatmap_helpers(widget)
    _bind_layer_name_event_helpers(widget)
    return widget


def test_selected_neuron_heatmap_layer_rename_updates_table_membership() -> None:
    viewer = _DummyViewer(ndisplay=3)
    layer = viewer.add_image(
        np.ones((2, 2, 2), dtype=np.float32),
        name="alpha Heatmap",
        metadata={
            "heatmap_kind": "selected_neurons",
            "file_ids": ["n1", "n2"],
            "manual_heatmap_id": "alpha",
        },
    )
    table = types.SimpleNamespace(set_heatmap_layers_by_file_id=MagicMock())
    widget = _make_layer_name_sync_widget(viewer, table)

    NeuronViewerWidget._sync_layer_name_event_connections(widget)
    layer.name = "Renamed Heatmap"
    layer.events.name.emit(types.SimpleNamespace(source=layer))

    table.set_heatmap_layers_by_file_id.assert_called_once_with(
        {
            "n1": ("Renamed Heatmap",),
            "n2": ("Renamed Heatmap",),
        }
    )
    widget._refresh_heatmap_layer_list.assert_called_once_with()
    widget._refresh_histogram_layer_list.assert_called_once_with()
    widget._refresh_mask_layer_options.assert_called_once_with()


def test_layer_name_event_sync_does_not_duplicate_callbacks() -> None:
    viewer = _DummyViewer(ndisplay=3)
    layer = viewer.add_image(
        np.ones((2, 2, 2), dtype=np.float32),
        name="alpha Heatmap",
        metadata={
            "heatmap_kind": "selected_neurons",
            "file_ids": ["n1"],
            "manual_heatmap_id": "alpha",
        },
    )
    table = types.SimpleNamespace(set_heatmap_layers_by_file_id=MagicMock())
    widget = _make_layer_name_sync_widget(viewer, table)

    NeuronViewerWidget._sync_layer_name_event_connections(widget)
    NeuronViewerWidget._sync_layer_name_event_connections(widget)
    layer.name = "Renamed Heatmap"
    layer.events.name.emit(types.SimpleNamespace(source=layer))

    assert len(layer.events.name._callbacks) == 1
    table.set_heatmap_layers_by_file_id.assert_called_once_with(
        {"n1": ("Renamed Heatmap",)}
    )


def test_removed_layer_name_event_disconnects_from_table_membership_sync() -> None:
    viewer = _DummyViewer(ndisplay=3)
    layer = viewer.add_image(
        np.ones((2, 2, 2), dtype=np.float32),
        name="alpha Heatmap",
        metadata={
            "heatmap_kind": "selected_neurons",
            "file_ids": ["n1"],
            "manual_heatmap_id": "alpha",
        },
    )
    table = types.SimpleNamespace(set_heatmap_layers_by_file_id=MagicMock())
    widget = _make_layer_name_sync_widget(viewer, table)

    NeuronViewerWidget._sync_layer_name_event_connections(widget)
    viewer.layers.remove(layer)
    NeuronViewerWidget._sync_layer_name_event_connections(widget)
    layer.name = "Stale Heatmap"
    layer.events.name.emit(types.SimpleNamespace(source=layer))

    assert layer.events.name._callbacks == []
    table.set_heatmap_layers_by_file_id.assert_not_called()


def test_manual_heatmap_combo_preserves_selection_by_stable_id_after_rename() -> None:
    viewer = _DummyViewer(ndisplay=3)
    layer = viewer.add_image(
        np.ones((2, 2, 2), dtype=np.float32),
        name="alpha Heatmap",
        metadata={
            "heatmap_kind": "selected_neurons",
            "file_ids": ["n1"],
            "manual_heatmap_id": "alpha",
        },
    )
    combo = _DummyMutableComboBox()
    table = types.SimpleNamespace(apply_filters=MagicMock())
    widget = types.SimpleNamespace(
        viewer=viewer,
        _manual_heatmap_combo=combo,
        _cluster_filter_combo=_DummyClusterFilterCombo(),
        _neuron_table=table,
    )
    _bind_manual_heatmap_helpers(widget)

    NeuronViewerWidget._refresh_manual_heatmap_combo(widget)
    combo.setCurrentIndex(1)
    layer.name = "Renamed Heatmap"
    NeuronViewerWidget._refresh_manual_heatmap_combo(widget)

    assert combo.currentText() == "Renamed Heatmap"
    assert NeuronViewerWidget._manual_heatmap_combo_data(widget) == (
        "Renamed Heatmap",
        ("n1",),
    )
    table.apply_filters.assert_not_called()


def test_manual_heatmap_combo_lists_only_manual_heatmaps_and_preserves_selection() -> (
    None
):
    viewer = _DummyViewer(ndisplay=3)
    viewer.layers.extend(
        [
            types.SimpleNamespace(
                name="alpha Heatmap",
                metadata={
                    "heatmap_kind": "selected_neurons",
                    "file_ids": ["n1"],
                    "manual_heatmap_id": "alpha",
                },
            ),
            types.SimpleNamespace(
                name="Cluster 1 Heatmap",
                metadata={
                    "heatmap_kind": "analysis",
                    "file_ids": ["n2"],
                },
            ),
            types.SimpleNamespace(
                name="beta Heatmap",
                metadata={
                    "heatmap_kind": "selected_neurons",
                    "file_ids": ["n3", "n4"],
                    "manual_heatmap_id": "beta",
                },
            ),
            types.SimpleNamespace(
                name="Points Heatmap: imported",
                metadata={
                    "heatmap_kind": "point_import",
                    "file_ids": ["n5"],
                },
            ),
        ]
    )
    combo = _DummyMutableComboBox()
    table = types.SimpleNamespace(apply_filters=MagicMock())
    widget = types.SimpleNamespace(
        viewer=viewer,
        _manual_heatmap_combo=combo,
        _neuron_table=table,
    )
    _bind_manual_heatmap_helpers(widget)

    NeuronViewerWidget._refresh_manual_heatmap_combo(widget)
    combo.setCurrentIndex(2)
    NeuronViewerWidget._refresh_manual_heatmap_combo(widget)

    assert [text for text, _data in combo.items] == [
        "All Manual Heatmaps",
        "alpha Heatmap",
        "beta Heatmap",
    ]
    assert combo.currentText() == "beta Heatmap"
    assert combo.enabled is True
    table.apply_filters.assert_not_called()


def test_manual_heatmap_combo_filters_rows_and_all_clears_manual_filter() -> None:
    combo = _DummyMutableComboBox()
    combo.addItem("All Manual Heatmaps", None)
    combo.addItem("alpha Heatmap", ("alpha Heatmap", ("n1", "n2")))
    table = types.SimpleNamespace(apply_filters=MagicMock())
    selection = ClusterFilterSelection({1})
    widget = types.SimpleNamespace(
        _manual_heatmap_combo=combo,
        _cluster_filter_combo=_DummyClusterFilterCombo(selection),
        _neuron_table=table,
    )
    _bind_manual_heatmap_helpers(widget)

    combo.setCurrentIndex(1)
    NeuronViewerWidget._on_manual_heatmap_selection_changed(widget)
    combo.setCurrentIndex(0)
    NeuronViewerWidget._on_manual_heatmap_selection_changed(widget)

    assert [record.args for record in table.apply_filters.call_args_list] == [
        (selection, ("n1", "n2")),
        (selection, None),
    ]


def test_manual_heatmap_combo_removed_selection_clears_manual_filter() -> None:
    viewer = _DummyViewer(ndisplay=3)
    viewer.layers.extend(
        [
            types.SimpleNamespace(
                name="alpha Heatmap",
                metadata={
                    "heatmap_kind": "selected_neurons",
                    "file_ids": ["n1"],
                    "manual_heatmap_id": "alpha",
                },
            ),
            types.SimpleNamespace(
                name="beta Heatmap",
                metadata={
                    "heatmap_kind": "selected_neurons",
                    "file_ids": ["n2"],
                    "manual_heatmap_id": "beta",
                },
            ),
        ]
    )
    combo = _DummyMutableComboBox()
    selection = ClusterFilterSelection({2})
    table = types.SimpleNamespace(apply_filters=MagicMock())
    widget = types.SimpleNamespace(
        viewer=viewer,
        _manual_heatmap_combo=combo,
        _cluster_filter_combo=_DummyClusterFilterCombo(selection),
        _neuron_table=table,
    )
    _bind_manual_heatmap_helpers(widget)

    NeuronViewerWidget._refresh_manual_heatmap_combo(widget)
    combo.setCurrentIndex(1)
    viewer.layers.pop(0)
    NeuronViewerWidget._refresh_manual_heatmap_combo(widget)

    assert combo.currentText() == "All Manual Heatmaps"
    assert [text for text, _data in combo.items] == [
        "All Manual Heatmaps",
        "beta Heatmap",
    ]
    table.apply_filters.assert_called_once_with(selection, None)


def test_selected_neuron_heatmap_finished_adds_unique_multi_selection_layer() -> None:
    viewer = _DummyViewer(ndisplay=3)
    viewer.layers.extend(
        [
            types.SimpleNamespace(name="alpha Heatmap"),
            types.SimpleNamespace(name="beta Heatmap"),
        ]
    )
    widget = types.SimpleNamespace(
        viewer=viewer,
        _db=types.SimpleNamespace(parquet_path=Path("/tmp/neurons.parquet")),
        _atlas=types.SimpleNamespace(atlas_name="fake_atlas"),
        _opacity_slider=_DummyValueControl(75),
        _render_progress=_DummyProgressBar(),
        _render_status_label=_DummyLabel(),
        _selected_heatmap_request_file_ids=("n1", "n2"),
        _refresh_heatmap_layer_list=MagicMock(),
        _refresh_histogram_layer_list=MagicMock(),
        _refresh_mask_layer_options=MagicMock(),
    )
    _bind_manual_heatmap_helpers(widget)
    widget._next_manual_heatmap_identifier = types.MethodType(
        NeuronViewerWidget._next_manual_heatmap_identifier,
        widget,
    )
    widget._selected_neuron_heatmap_layer_name = types.MethodType(
        NeuronViewerWidget._selected_neuron_heatmap_layer_name,
        widget,
    )
    widget._current_atlas_name = types.MethodType(
        NeuronViewerWidget._current_atlas_name,
        widget,
    )
    widget._add_selected_neuron_heatmap_layer = types.MethodType(
        NeuronViewerWidget._add_selected_neuron_heatmap_layer,
        widget,
    )

    NeuronViewerWidget._on_selected_neuron_heatmap_finished(
        widget,
        np.ones((2, 2, 2), dtype=np.float32),
    )

    created_layer = viewer.layers[-1]
    assert created_layer.name == "gamma Heatmap"
    assert created_layer.metadata["file_ids"] == ["n1", "n2"]
    assert created_layer.metadata["selection_count"] == 2
    assert created_layer.metadata["manual_heatmap_id"] == "gamma"
    widget._refresh_heatmap_layer_list.assert_called_once_with()
    widget._refresh_histogram_layer_list.assert_called_once_with()
    widget._refresh_mask_layer_options.assert_called_once_with()
    assert "gamma Heatmap" in widget._render_status_label.text


def test_selected_neuron_heatmap_finished_updates_table_heatmap_membership() -> None:
    viewer = _DummyViewer(ndisplay=3)
    table = types.SimpleNamespace(set_heatmap_layers_by_file_id=MagicMock())
    widget = types.SimpleNamespace(
        viewer=viewer,
        _db=types.SimpleNamespace(parquet_path=Path("/tmp/neurons.parquet")),
        _atlas=types.SimpleNamespace(atlas_name="fake_atlas"),
        _opacity_slider=_DummyValueControl(75),
        _render_progress=_DummyProgressBar(),
        _render_status_label=_DummyLabel(),
        _selected_heatmap_request_file_ids=("n1", "n2"),
        _neuron_table=table,
        _refresh_heatmap_layer_list=MagicMock(),
        _refresh_histogram_layer_list=MagicMock(),
        _refresh_mask_layer_options=MagicMock(),
    )
    _bind_manual_heatmap_helpers(widget)
    widget._next_manual_heatmap_identifier = types.MethodType(
        NeuronViewerWidget._next_manual_heatmap_identifier,
        widget,
    )
    widget._selected_neuron_heatmap_layer_name = types.MethodType(
        NeuronViewerWidget._selected_neuron_heatmap_layer_name,
        widget,
    )
    widget._current_atlas_name = types.MethodType(
        NeuronViewerWidget._current_atlas_name,
        widget,
    )
    widget._sync_neuron_table_heatmap_membership = types.MethodType(
        NeuronViewerWidget._sync_neuron_table_heatmap_membership,
        widget,
    )
    widget._add_selected_neuron_heatmap_layer = types.MethodType(
        NeuronViewerWidget._add_selected_neuron_heatmap_layer,
        widget,
    )

    NeuronViewerWidget._on_selected_neuron_heatmap_finished(
        widget,
        np.ones((2, 2, 2), dtype=np.float32),
    )

    table.set_heatmap_layers_by_file_id.assert_called_once_with(
        {
            "n1": ("alpha Heatmap",),
            "n2": ("alpha Heatmap",),
        }
    )


def test_region_isolation_region_ids_follow_include_children_state() -> None:
    selector = _DummyRegionSelector(
        direct_acronyms=["R1"],
        direct_ids=[1],
        query_ids=[1, 2],
        include_children=True,
    )
    widget = types.SimpleNamespace(_tools_region_selector=selector)

    assert NeuronViewerWidget._selected_region_isolation_region_ids(widget) == [1, 2]
    assert NeuronViewerWidget._selected_region_isolation_entries(widget) == [(1, "R1")]

    selector._include_children = False
    assert NeuronViewerWidget._selected_region_isolation_region_ids(widget) == [1]


def test_heatmap_gamma_actions_apply_to_all_selected_layers() -> None:
    source_a = types.SimpleNamespace(name="Source A", gamma=1.0)
    source_b = types.SimpleNamespace(name="Source B", gamma=0.8)
    status = _DummyLabel()
    widget = types.SimpleNamespace(
        _selected_heatmap_layers=lambda: [source_a, source_b],
        _tools_status_label=status,
    )
    widget._set_selected_heatmap_gamma = types.MethodType(
        NeuronViewerWidget._set_selected_heatmap_gamma,
        widget,
    )

    NeuronViewerWidget._enhance_selected_heatmap_projections(widget)

    assert source_a.gamma == pytest.approx(0.2)
    assert source_b.gamma == pytest.approx(0.2)
    assert status.text == (
        "Enhanced fine projections on 2 heatmap layer(s) (gamma 0.20)."
    )

    NeuronViewerWidget._reset_selected_heatmap_gamma(widget)

    assert source_a.gamma == pytest.approx(1.0)
    assert source_b.gamma == pytest.approx(1.0)
    assert status.text == "Reset gamma on 2 heatmap layer(s) (gamma 1.00)."


def test_heatmap_gamma_action_requires_a_selected_layer() -> None:
    status = _DummyLabel()
    widget = types.SimpleNamespace(
        _selected_heatmap_layers=lambda: [],
        _tools_status_label=status,
    )

    NeuronViewerWidget._set_selected_heatmap_gamma(
        widget,
        0.2,
        action="Enhanced fine projections",
    )

    assert status.text == "Select at least one eligible heatmap layer."


def test_tools_selection_enables_gamma_actions() -> None:
    enhance_button = _DummyButton()
    reset_button = _DummyButton()
    selected_layers = [types.SimpleNamespace(name="Source A")]
    widget = types.SimpleNamespace(
        _atlas=object(),
        _heatmap_layer_list=object(),
        _selected_heatmap_layers=lambda: selected_layers,
        _enhance_fine_projections_btn=enhance_button,
        _reset_heatmap_gamma_btn=reset_button,
    )

    NeuronViewerWidget._update_tools_controls(widget)

    assert enhance_button.enabled is True
    assert reset_button.enabled is True

    selected_layers.clear()
    NeuronViewerWidget._update_tools_controls(widget)

    assert enhance_button.enabled is False
    assert reset_button.enabled is False


def test_create_region_isolated_heatmaps_separate_layers_sets_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_colormaps(monkeypatch)
    viewer = _DummyViewer(ndisplay=3)
    annotation = np.array([[[1, 2, 3], [2, 1, 0]]], dtype=np.int32)
    atlas = types.SimpleNamespace(annotation=annotation, atlas_name="fake_atlas")
    source_a = _DummyImageLayer(
        np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32),
        name="Source A",
        colormap="source_cmap",
        blending="additive",
        rendering="mip",
        opacity=0.6,
        metadata={
            "heatmap_source": True,
            "heatmap_native_grid": True,
            "atlas_name": "fake_atlas",
            "color": (1.0, 0.0, 0.0, 1.0),
            "heatmap_selected_region_id": 99,
            "source_file_ids": ["n1"],
        },
    )
    source_b = _DummyImageLayer(
        np.array([[[0.0, 1.0, 0.0], [3.0, 0.0, 9.0]]], dtype=np.float32),
        name="Source B",
        metadata={
            "heatmap_source": True,
            "heatmap_native_grid": True,
            "atlas_name": "fake_atlas",
            "source_file_ids": ["n2"],
        },
    )
    viewer.layers.extend([source_a, source_b])
    selector = _DummyRegionSelector(
        direct_acronyms=["R1"],
        direct_ids=[1],
        query_ids=[1, 2],
        include_children=True,
    )
    widget = types.SimpleNamespace(
        viewer=viewer,
        _atlas=atlas,
        _opacity_slider=_DummyValueControl(50),
        _tools_status_label=_DummyLabel(),
        _tools_region_selector=selector,
        _region_isolation_create_mode_combo=_DummyComboBox("Separate", "separate"),
        _selected_heatmap_layers=lambda: [source_a, source_b],
        _refresh_heatmap_layer_list=MagicMock(),
        _refresh_histogram_layer_list=MagicMock(),
        _select_heatmap_layer_names=MagicMock(),
        _select_histogram_layer_names=MagicMock(),
    )
    _bind_region_isolation_methods(widget)

    NeuronViewerWidget._create_region_isolated_heatmaps(widget)

    created_a, created_b = viewer.layers[-2:]
    expected_a = np.where(np.isin(annotation, [1, 2]), source_a.data, 0.0)
    expected_b = np.where(np.isin(annotation, [1, 2]), source_b.data, 0.0)
    np.testing.assert_array_equal(created_a.data, expected_a)
    np.testing.assert_array_equal(created_b.data, expected_b)
    assert created_a.name == "Region Isolated (R1): Source A"
    assert created_b.name == "Region Isolated (R1): Source B"
    assert created_a.colormap == "source_cmap"
    assert created_a.blending == "additive"
    assert created_a.rendering == "mip"
    assert created_a.opacity == 0.6
    assert created_a.contrast_limits == (0.0, 5.0)
    assert created_a.metadata["heatmap_kind"] == "region_isolated"
    assert created_a.metadata["source_heatmap_layers"] == ["Source A"]
    assert created_a.metadata["source_file_ids"] == ["n1"]
    assert created_a.metadata["file_ids"] == ["n1"]
    assert created_a.metadata["heatmap_selected_region_ids"] == [1]
    assert created_a.metadata["heatmap_selected_region_acronyms"] == ["R1"]
    assert created_a.metadata["heatmap_region_ids"] == [1, 2]
    assert created_a.metadata["heatmap_include_child_regions"] is True
    assert created_a.metadata["merge_mode"] == "separate"
    assert created_a.metadata["atlas_name"] == "fake_atlas"
    assert "heatmap_selected_region_id" not in created_a.metadata
    widget._select_heatmap_layer_names.assert_called_once_with(
        ["Region Isolated (R1): Source A", "Region Isolated (R1): Source B"]
    )
    assert "Created 2 isolated heatmap layer(s)" in widget._tools_status_label.text


def test_create_region_isolated_heatmaps_merged_sums_before_masking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_colormaps(monkeypatch)
    viewer = _DummyViewer(ndisplay=3)
    annotation = np.array([[[1, 2], [1, 0]]], dtype=np.int32)
    atlas = types.SimpleNamespace(annotation=annotation, atlas_name="fake_atlas")
    source_a = _DummyImageLayer(
        np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        name="Source A",
        metadata={
            "heatmap_source": True,
            "heatmap_native_grid": True,
            "atlas_name": "fake_atlas",
            "color": (1.0, 0.0, 0.0, 1.0),
            "source_file_ids": ["n1"],
        },
    )
    source_b = _DummyImageLayer(
        np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        name="Source B",
        metadata={
            "heatmap_source": True,
            "heatmap_native_grid": True,
            "atlas_name": "fake_atlas",
            "color": (0.0, 1.0, 0.0, 1.0),
            "source_file_ids": ["n2", "n1"],
        },
    )
    viewer.layers.extend([source_a, source_b])
    selector = _DummyRegionSelector(
        direct_acronyms=["R1"],
        direct_ids=[1],
        query_ids=[1, 2],
        include_children=False,
    )
    widget = types.SimpleNamespace(
        viewer=viewer,
        _atlas=atlas,
        _opacity_slider=_DummyValueControl(50),
        _tools_status_label=_DummyLabel(),
        _tools_region_selector=selector,
        _region_isolation_create_mode_combo=_DummyComboBox("Merged", "merged"),
        _selected_heatmap_layers=lambda: [source_a, source_b],
        _refresh_heatmap_layer_list=MagicMock(),
        _refresh_histogram_layer_list=MagicMock(),
        _select_heatmap_layer_names=MagicMock(),
        _select_histogram_layer_names=MagicMock(),
    )
    _bind_region_isolation_methods(widget)

    NeuronViewerWidget._create_region_isolated_heatmaps(widget)

    created = viewer.layers[-1]
    expected = np.where(annotation == 1, source_a.data + source_b.data, 0.0)
    np.testing.assert_array_equal(created.data, expected)
    assert created.name == "Region Isolated (R1): merged 2 heatmaps"
    assert created.metadata["source_heatmap_layers"] == ["Source A", "Source B"]
    assert created.metadata["source_file_ids"] == ["n1", "n2"]
    assert created.metadata["file_ids"] == ["n1", "n2"]
    assert created.metadata["heatmap_region_ids"] == [1]
    assert created.metadata["heatmap_include_child_regions"] is False
    assert created.metadata["merge_mode"] == "merged_sum"
    assert created.contrast_limits == (0.0, 10.0)


def test_load_flatmap_transform_status_warns_for_legacy_mirror_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = types.SimpleNamespace(
        present_transform_text="flatmap and depth",
        has_full_transform=True,
        uses_legacy_mirror_fallback=True,
    )
    monkeypatch.setitem(
        NeuronViewerWidget._load_flatmap_transform_status.__globals__,
        "read_flatmap_parquet_transform_info",
        lambda _path: info,
    )
    widget = NeuronViewerWidget.__new__(NeuronViewerWidget)
    widget._flatmap_transform_status_label = _DummyLabel()

    result = NeuronViewerWidget._load_flatmap_transform_status(
        widget,
        "legacy.parquet",
    )

    assert result == "flatmap and depth"
    assert "version-1 transform" in widget._flatmap_transform_status_label.text
    assert "Regenerate the augmented Parquet" in (
        widget._flatmap_transform_status_label.text
    )
