"""Tests for Analysis tab hierarchical region selection helpers."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from napari_neuron_navigator.analysis.clustering import (
    ClusterRegionSelection,
    ClusterResult,
)
from napari_neuron_navigator.cluster_assignments import ClusterAssignmentStore


class _BoundSignal:
    """Minimal signal object used by widget stubs."""

    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in list(self._callbacks):
            callback(*args)


class _Signal:
    """Descriptor stand-in for ``qtpy.QtCore.Signal``."""

    def __init__(self, *_args, **_kwargs) -> None:
        self._storage_name = ""

    def __set_name__(self, _owner, name: str) -> None:
        self._storage_name = f"__signal_{name}"

    def __get__(self, instance, _owner):
        if instance is None:
            return self
        if self._storage_name not in instance.__dict__:
            instance.__dict__[self._storage_name] = _BoundSignal()
        return instance.__dict__[self._storage_name]


class _DummyWidget:
    """Very small QWidget-like stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        self._enabled = True
        self._visible = True
        self._tooltip = ""
        self.geometry_updates = 0

    def setEnabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def isEnabled(self) -> bool:
        return self._enabled

    def setVisible(self, visible: bool) -> None:
        self._visible = bool(visible)

    def setToolTip(self, text: str) -> None:
        self._tooltip = str(text)

    def toolTip(self) -> str:
        return self._tooltip

    def updateGeometry(self) -> None:
        self.geometry_updates += 1


class _DummyLayout:
    """Simple layout stub that records added children."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.children: list[object] = []

    def addWidget(self, widget) -> None:
        self.children.append(widget)

    def addLayout(self, layout) -> None:
        self.children.append(layout)

    def addStretch(self, *_args) -> None:
        return None

    def setContentsMargins(self, *_args) -> None:
        return None


class _DummyLabel(_DummyWidget):
    """Small QLabel stand-in."""

    def __init__(self, text: str = "", *_args, **_kwargs) -> None:
        super().__init__()
        self._text = text
        self._word_wrap = False

    def setText(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setWordWrap(self, wrap: bool) -> None:
        self._word_wrap = bool(wrap)


class _DummyButton(_DummyWidget):
    """Small QPushButton stand-in."""

    def __init__(self, text: str = "", *_args, **_kwargs) -> None:
        super().__init__()
        self._text = text
        self.clicked = _BoundSignal()


class _DummyCheckBox(_DummyWidget):
    """Small QCheckBox stand-in."""

    def __init__(self, text: str = "", *_args, **_kwargs) -> None:
        super().__init__()
        self._text = text
        self._checked = False
        self.toggled = _BoundSignal()

    def setChecked(self, checked: bool) -> None:
        checked = bool(checked)
        changed = checked != self._checked
        self._checked = checked
        if changed:
            self.toggled.emit(checked)

    def isChecked(self) -> bool:
        return self._checked


class _DummyLineEdit(_DummyWidget):
    """Small QLineEdit stand-in."""

    def __init__(self, text: str = "", *_args, **_kwargs) -> None:
        super().__init__()
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = str(text)


class _DummyCombo(_DummyWidget):
    """Small QComboBox stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.currentTextChanged = _BoundSignal()
        self.currentIndexChanged = _BoundSignal()
        self._items: list[dict[str, object]] = []
        self._current_index = -1
        self._blocked = False
        self._editable = False
        self._edit_text = ""

    def addItems(self, items: list[str]) -> None:
        for item in items:
            self.addItem(item)

    def addItem(self, *args) -> None:
        if len(args) == 1:
            text = str(args[0])
            data = None
        else:
            text = str(args[0])
            data = args[1]
        self._items.append({"text": text, "data": data})
        if self._current_index < 0:
            self._current_index = 0

    def clear(self) -> None:
        self._items = []
        self._current_index = -1
        self._edit_text = ""

    def setEditable(self, editable: bool) -> None:
        self._editable = bool(editable)

    def currentText(self) -> str:
        if 0 <= self._current_index < len(self._items):
            return str(self._items[self._current_index]["text"])
        return self._edit_text

    def setCurrentText(self, text: str) -> None:
        for index, item in enumerate(self._items):
            if item["text"] == text:
                self._current_index = index
                break
        else:
            self._edit_text = text
        if not self._blocked:
            self.currentTextChanged.emit(text)

    def setEditText(self, text: str) -> None:
        self._edit_text = text

    def blockSignals(self, blocked: bool) -> None:
        self._blocked = bool(blocked)

    def currentIndex(self) -> int:
        return self._current_index

    def setCurrentIndex(self, index: int) -> None:
        self._current_index = int(index)
        if not self._blocked:
            self.currentIndexChanged.emit(index)
            self.currentTextChanged.emit(self.currentText())

    def itemData(self, index: int):
        return self._items[index]["data"]

    def currentData(self):
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index]["data"]
        return None

    def setItemData(self, index: int, data) -> None:
        self._items[index]["data"] = data

    def count(self) -> int:
        return len(self._items)


class _DummySpinBox(_DummyWidget):
    """Small spinbox stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.valueChanged = _BoundSignal()
        self._value = 0

    def setRange(self, *_args) -> None:
        return None

    def setValue(self, value) -> None:
        self._value = value

    def value(self):
        return self._value

    def setSuffix(self, *_args) -> None:
        return None

    def setDecimals(self, *_args) -> None:
        return None

    def setSingleStep(self, *_args) -> None:
        return None


class _DummyProgressBar(_DummyWidget):
    """Small progress bar stand-in."""

    def setRange(self, *_args) -> None:
        return None

    def setValue(self, *_args) -> None:
        return None


class _DummyScrollArea(_DummyWidget):
    """Small scroll-area stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.widget_resizable = False
        self.widget = None

    def setWidgetResizable(self, resizable: bool) -> None:
        self.widget_resizable = bool(resizable)

    def setWidget(self, widget) -> None:
        self.widget = widget


class _DummyStack(_DummyWidget):
    """Small QStackedWidget stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.children: list[object] = []
        self.index = 0

    def addWidget(self, widget) -> None:
        self.children.append(widget)

    def setCurrentIndex(self, index: int) -> None:
        self.index = int(index)


class _DummyFigure:
    """Minimal figure stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.cleared = 0
        self.canvas = None
        self.dpi = 100.0
        self.size_inches: tuple[float, float] | None = (6.0, 6.0)
        self.set_size_inches_calls = 0

    def clear(self) -> None:
        self.cleared += 1

    def add_subplot(self, *_args, **_kwargs):
        return types.SimpleNamespace(text=lambda *a, **k: None)

    def set_canvas(self, canvas) -> None:
        self.canvas = canvas

    def get_dpi(self) -> float:
        return self.dpi

    def get_size_inches(self):
        return self.size_inches

    def set_size_inches(
        self, width: float, height: float, forward: bool = True
    ) -> None:
        self.set_size_inches_calls += 1
        self.size_inches = (float(width), float(height), bool(forward))


class _DummyCanvas(_DummyWidget):
    """Minimal canvas stand-in."""

    def __init__(self, figure, *_args, **_kwargs) -> None:
        super().__init__()
        self.figure = figure
        self._width = 600
        self._height = 400
        self.device_pixel_ratio = 1.0
        self._physical_width: int | None = None
        self._physical_height: int | None = None

    def setMinimumHeight(self, *_args) -> None:
        return None

    def get_width_height(self, *, physical: bool = False):
        if physical:
            if self._physical_width is not None and self._physical_height is not None:
                return (self._physical_width, self._physical_height)
            return (
                int(round(self._width * float(self.device_pixel_ratio))),
                int(round(self._height * float(self.device_pixel_ratio))),
            )
        return (self._width, self._height)

    def draw(self) -> None:
        return None


class _DummyScrollArea(_DummyWidget):
    """Minimal QScrollArea stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.widget_resizable = False
        self.widget = None

    def setWidgetResizable(self, enabled: bool) -> None:
        self.widget_resizable = bool(enabled)

    def setWidget(self, widget) -> None:
        self.widget = widget


class _DummyColormap:
    """Minimal napari Colormap stand-in."""

    def __init__(self, colors, name: str) -> None:
        self.colors = colors
        self.name = name


class _DummyCollapsibleSection(_DummyWidget):
    """Simple collapsible section stub that records constructor args."""

    instances: list["_DummyCollapsibleSection"] = []

    def __init__(self, title: str, *, expanded: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.expanded = expanded
        self.expanded_changed = _BoundSignal()
        self._layout = _DummyLayout()
        self.__class__.instances.append(self)

    def content_layout(self) -> _DummyLayout:
        return self._layout

    def emit_expanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)
        self.expanded_changed.emit(bool(expanded))


class _DummyRegionSelector(_DummyWidget):
    """Simple region selector stub used for Analysis tab tests."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.selection_changed = _BoundSignal()
        self.allowed_ids: set[int] | None = None
        self.atlas = None
        self.selected: tuple[int, str] | None = None
        self.selected_ids: list[int] = []
        self._structure_map: dict[int, dict] = {}
        self.clear_calls = 0

    def get_single_selected_region(self) -> tuple[int, str] | None:
        if self.selected is not None:
            return self.selected
        if self.selected_ids:
            struct_id = int(self.selected_ids[0])
            struct = self._structure_map.get(struct_id, {})
            acronym = str(struct.get("acronym", f"R{struct_id}"))
            return (struct_id, acronym)
        return self.selected

    def get_selected_ids(self, include_children: bool = True) -> list[int]:
        _ = include_children
        if self.selected_ids:
            return list(self.selected_ids)
        if self.selected is not None:
            return [int(self.selected[0])]
        return []

    def set_allowed_structure_ids(self, structure_ids: set[int] | None) -> None:
        self.allowed_ids = None if structure_ids is None else set(structure_ids)

    def set_atlas(self, atlas) -> None:
        self.atlas = atlas
        self._structure_map = getattr(atlas, "structures", {})

    def select_region_by_id(self, region_id: int | None) -> None:
        if region_id is None:
            self.selected = None
            self.selected_ids = []
            return
        struct = self._structure_map.get(region_id, {})
        acronym = str(struct.get("acronym", f"R{region_id}"))
        self.selected = (int(region_id), acronym)
        self.selected_ids = [int(region_id)]

    def select_regions(self, acronyms: list[str]) -> None:
        selected_ids = [
            int(struct_id)
            for struct_id, struct in self._structure_map.items()
            if struct.get("acronym") in set(acronyms)
        ]
        self.selected_ids = selected_ids
        self.selected = self.get_single_selected_region()

    def clear(self) -> None:
        self.clear_calls += 1
        self.selected = None
        self.selected_ids = []
        self._structure_map = {}


class _DummyFileDialog:
    """Minimal QFileDialog stand-in."""

    @staticmethod
    def getSaveFileName(*_args, **_kwargs):
        return ("", "")


class _DummyQColor:
    """Minimal QColor stand-in."""

    @staticmethod
    def fromRgbF(*_args):
        return None


class _DummyQPixmap:
    """Minimal QPixmap stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None

    def fill(self, *_args) -> None:
        return None


class _DummyQIcon:
    """Minimal QIcon stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _DummyThread:
    """Minimal QThread stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.started = _BoundSignal()
        self.finished = _BoundSignal()

    def isRunning(self) -> bool:
        return False

    def start(self) -> None:
        return None

    def quit(self) -> None:
        return None

    def deleteLater(self) -> None:
        return None


class _DummyQTimer:
    """Minimal QTimer stand-in with immediate singleShot execution."""

    @staticmethod
    def singleShot(_interval: int, callback) -> None:
        callback()


class _DummyOrderSignal:
    """Dummy napari dims order event."""

    def connect(self, *_args, **_kwargs) -> None:
        return None


class _DummyViewer:
    """Very small viewer stand-in."""

    def __init__(self) -> None:
        self.dims = types.SimpleNamespace(
            not_displayed=(0,),
            events=types.SimpleNamespace(order=_DummyOrderSignal()),
        )
        self.layers = []
        self.last_image_kwargs: dict[str, object] | None = None

    def add_image(self, data, **kwargs):
        layer = _DummyImageLayer(data, **kwargs)
        self.layers.append(layer)
        self.last_image_kwargs = dict(kwargs)
        return layer


class _DummyImageLayer:
    """Minimal image layer stand-in for analysis heatmap tests."""

    def __init__(self, data, **kwargs) -> None:
        self.data = np.asarray(data)
        self.name = str(kwargs["name"])
        self.metadata = dict(kwargs.get("metadata", {}))
        self.contrast_limits = tuple(kwargs.get("contrast_limits", (0.0, 1.0)))
        self.contrast_limits_range = tuple(
            kwargs.get("contrast_limits_range", self.contrast_limits)
        )
        self.rendering = kwargs.get("rendering")
        self.blending = kwargs.get("blending")
        self.opacity = kwargs.get("opacity")
        self.visible = kwargs.get("visible", True)
        self.scale = list(kwargs.get("scale", []))
        self._keep_auto_contrast = False
        self._slice_input = types.SimpleNamespace(ndisplay=2)
        self.thumbnail_updates = 0
        self.slice_updates: list[object] = []

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


class _FakeAtlas:
    """Atlas-like stand-in for ancestry calculations."""

    def __init__(self, structures: dict[int, dict]) -> None:
        self.structures = structures
        self.resolution = (25.0, 25.0, 25.0)
        self.annotation = types.SimpleNamespace(shape=(4, 4, 4))


def _import_analysis_tab_module():
    """Import ``analysis_tab.py`` with stubbed UI dependencies."""
    backend_module = types.ModuleType("matplotlib.backends.backend_qtagg")
    backend_module.FigureCanvasQTAgg = _DummyCanvas

    figure_module = types.ModuleType("matplotlib.figure")
    figure_module.Figure = _DummyFigure

    napari_module = types.ModuleType("napari")
    napari_module.__path__ = []
    napari_utils_module = types.ModuleType("napari.utils")
    napari_utils_module.__path__ = []
    napari_colormaps_module = types.ModuleType("napari.utils.colormaps")
    napari_colormaps_module.Colormap = _DummyColormap
    napari_module.utils = napari_utils_module
    napari_utils_module.colormaps = napari_colormaps_module

    qtcore_module = types.ModuleType("qtpy.QtCore")
    qtcore_module.QThread = _DummyThread
    qtcore_module.QTimer = _DummyQTimer
    qtcore_module.Signal = _Signal

    qtgui_module = types.ModuleType("qtpy.QtGui")
    qtgui_module.QColor = _DummyQColor
    qtgui_module.QIcon = _DummyQIcon
    qtgui_module.QPixmap = _DummyQPixmap

    qtwidgets_module = types.ModuleType("qtpy.QtWidgets")
    for name, value in {
        "QCheckBox": _DummyCheckBox,
        "QComboBox": _DummyCombo,
        "QDoubleSpinBox": _DummySpinBox,
        "QFileDialog": _DummyFileDialog,
        "QGroupBox": _DummyWidget,
        "QHBoxLayout": _DummyLayout,
        "QLabel": _DummyLabel,
        "QLineEdit": _DummyLineEdit,
        "QProgressBar": _DummyProgressBar,
        "QPushButton": _DummyButton,
        "QScrollArea": _DummyScrollArea,
        "QSpinBox": _DummySpinBox,
        "QStackedWidget": _DummyStack,
        "QVBoxLayout": _DummyLayout,
        "QWidget": _DummyWidget,
    }.items():
        setattr(qtwidgets_module, name, value)

    repo_root = Path(__file__).resolve().parent.parent
    package_root = repo_root / "src" / "napari_neuron_navigator"
    widgets_root = package_root / "widgets"

    napari_package = types.ModuleType("napari_neuron_navigator")
    napari_package.__path__ = [str(package_root)]

    widgets_package = types.ModuleType("napari_neuron_navigator.widgets")
    widgets_package.__path__ = [str(widgets_root)]

    collapsible_module = types.ModuleType(
        "napari_neuron_navigator.widgets.collapsible_section"
    )
    collapsible_module.CollapsibleSection = _DummyCollapsibleSection

    region_selector_module = types.ModuleType(
        "napari_neuron_navigator.widgets.region_selector"
    )
    region_selector_module.RegionSelectorWidget = _DummyRegionSelector

    replacements = {
        "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
        "seaborn": types.ModuleType("seaborn"),
        "matplotlib.backends.backend_qtagg": backend_module,
        "matplotlib.figure": figure_module,
        "napari": napari_module,
        "napari.utils": napari_utils_module,
        "napari.utils.colormaps": napari_colormaps_module,
        "qtpy.QtCore": qtcore_module,
        "qtpy.QtGui": qtgui_module,
        "qtpy.QtWidgets": qtwidgets_module,
        "napari_neuron_navigator": napari_package,
        "napari_neuron_navigator.widgets": widgets_package,
        "napari_neuron_navigator.widgets.collapsible_section": collapsible_module,
        "napari_neuron_navigator.widgets.region_selector": region_selector_module,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    module_name = "napari_neuron_navigator.widgets.analysis_tab"
    previous_module = sys.modules.get(module_name)

    try:
        sys.modules.update(replacements)
        sys.modules.pop(module_name, None)

        module_path = widgets_root / "analysis_tab.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _make_selector(
    selected: tuple[int, str] | None = None,
    structure_map: dict[int, dict] | None = None,
    selected_ids: list[int] | None = None,
):
    selector = _DummyRegionSelector()
    selector.selected = selected
    selector.selected_ids = list(
        selected_ids or ([] if selected is None else [selected[0]])
    )
    selector._structure_map = structure_map or {}
    return selector


def _install_fake_napari_colormaps():
    """Install a minimal ``napari.utils.colormaps`` module for one test."""
    napari_module = types.ModuleType("napari")
    napari_module.__path__ = []
    napari_utils_module = types.ModuleType("napari.utils")
    napari_utils_module.__path__ = []
    napari_colormaps_module = types.ModuleType("napari.utils.colormaps")
    napari_colormaps_module.Colormap = _DummyColormap
    napari_module.utils = napari_utils_module
    napari_utils_module.colormaps = napari_colormaps_module

    replacements = {
        "napari": napari_module,
        "napari.utils": napari_utils_module,
        "napari.utils.colormaps": napari_colormaps_module,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    return previous


def _restore_modules(previous: dict[str, object | None]) -> None:
    """Restore modules temporarily replaced during a test."""
    for name, original in previous.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def test_analysis_region_sections_are_collapsed_by_default():
    """Analysis tab should expose collapsible top-level sections and compact region trees."""
    _DummyCollapsibleSection.instances = []
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget

    _widget = AnalysisTabWidget(_DummyViewer())

    titles = [section.title for section in _DummyCollapsibleSection.instances]
    expanded = [section.expanded for section in _DummyCollapsibleSection.instances]

    assert titles == [
        "Clustering",
        "Region Filters",
        "Node Count Heatmap",
        "Select Heatmap Region",
        "Progress",
        "Clustermap",
        "Export Results",
    ]
    assert expanded == [True, False, True, False, True, False, False]


def test_analysis_tab_wraps_content_in_scroll_area():
    """Analysis tab should use a scroll area so the dock can stay bounded."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())

    assert isinstance(widget._scroll_area, _DummyScrollArea)
    assert widget._scroll_area.widget_resizable is True
    assert widget._scroll_area.widget is widget._scroll_content


def test_analysis_tab_defaults_to_ward_linkage_and_no_dilation():
    """Analysis clustering controls should start with Ward linkage and 0% dilation."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())

    assert widget._method_combo.currentText() == "ward"
    assert widget._dilation_spin.value() == 0


def test_analysis_tab_input_scope_precedes_target_region_summary():
    """Input-neuron scope should appear before the target region summary."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())

    clustering_labels = [
        child.children[0].text()
        for child in widget._clustering_section.content_layout().children
        if isinstance(child, _DummyLayout)
        and child.children
        and isinstance(child.children[0], _DummyLabel)
    ]

    assert clustering_labels.index("Input neurons:") < clustering_labels.index(
        "Included regions:"
    )


def test_analysis_tab_exposes_bulk_cluster_heatmap_button():
    """Analysis tab should expose a dedicated bulk cluster-heatmap action."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())

    assert widget._add_all_cluster_heatmaps_btn._text == "Add All Cluster Heatmaps"
    assert not widget._add_all_cluster_heatmaps_btn.isEnabled()


def test_analysis_tab_defaults_soma_distance_filter_off():
    """Soma-distance heatmap filtering should require an explicit opt-in."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())

    assert widget._heat_soma_radius_enabled_cb._text == ("Filter by soma distance")
    assert widget._heat_soma_radius_enabled_cb.isChecked() is False
    assert not widget._heat_soma_radius_spin.isEnabled()

    widget._heat_soma_radius_spin.setValue(50.0)
    assert widget._selected_heatmap_soma_radius_um() is None

    widget._heat_soma_radius_enabled_cb.setChecked(True)
    assert widget._heat_soma_radius_spin.isEnabled()
    assert widget._selected_heatmap_soma_radius_um() == 50.0

    widget._heat_soma_radius_spin.setValue(0.0)
    assert widget._selected_heatmap_soma_radius_um() is None


def test_analysis_tab_export_section_omits_y_label_field():
    """Export Results should no longer expose a separate y-label input."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())

    assert hasattr(widget, "_export_title_edit")
    assert hasattr(widget, "_export_xlabel_edit")
    assert not hasattr(widget, "_export_ylabel_edit")


def test_analysis_allowed_structure_ids_include_dataset_regions_and_ancestors():
    """Dataset-backed Analysis trees should expose represented leaves plus ancestors."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._atlas = _FakeAtlas(
        {
            997: {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "structure_id_path": [997],
            },
            315: {
                "id": 315,
                "acronym": "ISO",
                "name": "Isocortex",
                "structure_id_path": [997, 315],
            },
            184: {
                "id": 184,
                "acronym": "FRP",
                "name": "Frontal pole",
                "structure_id_path": [997, 315, 184],
            },
            68: {
                "id": 68,
                "acronym": "FRP1",
                "name": "FRP1",
                "structure_id_path": [997, 315, 184, 68],
            },
        }
    )
    widget._dataset_region_ids = {68}

    assert widget._analysis_allowed_structure_ids() == {68, 184, 315, 997}


def test_refresh_analysis_region_selectors_waits_for_atlas_and_dataset():
    """Selectors should stay empty until both atlas and represented dataset IDs exist."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_region_selector = _make_selector()
    widget._heat_region_selector = _make_selector()
    widget._cluster_region_summary_label = _DummyLabel()
    widget._heat_region_summary_label = _DummyLabel()
    widget._atlas = None
    widget._dataset_region_ids = {68}

    widget._refresh_analysis_region_selectors()

    assert widget._cluster_region_selector.clear_calls == 1
    assert widget._heat_region_selector.clear_calls == 1
    assert widget._cluster_region_summary_label.text() == "None selected"
    assert widget._heat_region_summary_label.text() == "All regions"


def test_refresh_analysis_region_selectors_populates_both_selectors_from_dataset():
    """Both Analysis selectors should receive the same dataset-backed visible tree IDs."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_region_selector = _make_selector()
    widget._heat_region_selector = _make_selector()
    widget._cluster_region_summary_label = _DummyLabel()
    widget._heat_region_summary_label = _DummyLabel()
    widget._atlas = _FakeAtlas(
        {
            997: {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "structure_id_path": [997],
            },
            315: {
                "id": 315,
                "acronym": "ISO",
                "name": "Isocortex",
                "structure_id_path": [997, 315],
            },
            184: {
                "id": 184,
                "acronym": "FRP",
                "name": "Frontal pole",
                "structure_id_path": [997, 315, 184],
            },
            68: {
                "id": 68,
                "acronym": "FRP1",
                "name": "FRP1",
                "structure_id_path": [997, 315, 184, 68],
            },
        }
    )
    widget._dataset_region_ids = {68}

    widget._refresh_analysis_region_selectors()

    expected = {68, 184, 315, 997}
    assert widget._cluster_region_selector.allowed_ids == expected
    assert widget._heat_region_selector.allowed_ids == expected
    assert widget._cluster_region_selector.atlas is widget._atlas
    assert widget._heat_region_selector.atlas is widget._atlas


def test_region_filter_node_types_come_from_loaded_dataset() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = types.SimpleNamespace(
        get_unique_regions=lambda: {"region_id": [184]},
        get_unique_node_types=lambda: (0, 1, 2, 99),
    )
    widget._atlas = _FakeAtlas(
        {
            184: {
                "id": 184,
                "acronym": "FRP",
                "name": "Frontal pole",
                "structure_id_path": [184],
            }
        }
    )

    widget.refresh_available_regions_from_database()

    for editor in widget._region_filter_editors():
        assert editor._node_types == (0, 1, 2, 99)
    option_builder = type(
        widget._whole_parquet_region_filter_editor
    ).set_node_types.__globals__["_node_type_options"]
    assert dict(option_builder((0, 1, 2, 99))) == {
        0: "Undefined",
        1: "Soma",
        2: "Axon-typed (type 2)",
        99: "Type 99",
    }


def test_refresh_analysis_region_selectors_preserves_independent_selections():
    """Clustering and heatmap selectors should keep separate valid selections."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    cluster_selector = _make_selector(
        selected=(184, "FRP"),
        structure_map={184: {"acronym": "FRP"}},
    )
    heat_selector = _make_selector(
        selected=(500, "CP"),
        structure_map={500: {"acronym": "CP"}},
    )

    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_region_selector = cluster_selector
    widget._heat_region_selector = heat_selector
    widget._cluster_region_summary_label = _DummyLabel()
    widget._heat_region_summary_label = _DummyLabel()
    widget._atlas = _FakeAtlas(
        {
            997: {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "structure_id_path": [997],
            },
            184: {
                "id": 184,
                "acronym": "FRP",
                "name": "Frontal pole",
                "structure_id_path": [997, 184],
            },
            500: {
                "id": 500,
                "acronym": "CP",
                "name": "Caudoputamen",
                "structure_id_path": [997, 500],
            },
        }
    )
    widget._dataset_region_ids = {184, 500}

    widget._refresh_analysis_region_selectors()

    assert widget._selected_cluster_region() == (184, "FRP")
    assert widget._selected_heat_region() == (500, "CP")


def test_refresh_analysis_region_selectors_preserves_multiple_cluster_selections():
    """Clustering selector should retain multiple direct selections after refresh."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    cluster_selector = _make_selector(
        structure_map={
            184: {"acronym": "FRP"},
            500: {"acronym": "CP"},
        },
        selected_ids=[184, 500],
    )
    heat_selector = _make_selector(
        selected=(500, "CP"), structure_map={500: {"acronym": "CP"}}
    )

    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_region_selector = cluster_selector
    widget._heat_region_selector = heat_selector
    widget._cluster_region_summary_label = _DummyLabel()
    widget._heat_region_summary_label = _DummyLabel()
    widget._atlas = _FakeAtlas(
        {
            997: {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "structure_id_path": [997],
            },
            184: {
                "id": 184,
                "acronym": "FRP",
                "name": "Frontal pole",
                "structure_id_path": [997, 184],
            },
            500: {
                "id": 500,
                "acronym": "CP",
                "name": "Caudoputamen",
                "structure_id_path": [997, 500],
            },
        }
    )
    widget._dataset_region_ids = {184, 500}

    widget._refresh_analysis_region_selectors()

    assert widget._selected_cluster_regions() == [(184, "FRP"), (500, "CP")]


def test_active_cluster_selector_switches_with_scope() -> None:
    """Each clustering input scope should retain its own target selector."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    structures = {
        184: {"name": "Frontal pole", "acronym": "FRP"},
        500: {"name": "Caudoputamen", "acronym": "CP"},
    }
    widget._whole_parquet_cluster_region_selector._structure_map = structures
    widget._current_table_cluster_region_selector._structure_map = structures
    widget._selected_rows_cluster_region_selector._structure_map = structures
    widget._whole_parquet_cluster_region_selector.select_region_by_id(184)
    widget._current_table_cluster_region_selector.select_region_by_id(500)
    widget._selected_rows_cluster_region_selector.select_region_by_id(184)

    widget._cluster_region_scope_combo.setCurrentText("Whole Parquet")

    assert widget._selected_cluster_region() == (184, "FRP")
    assert widget._cluster_region_scope_stack.index == 0
    assert widget._cluster_region_summary_label.text() == "FRP (Frontal pole)"

    widget._cluster_region_scope_combo.setCurrentText("Current Table")

    assert widget._selected_cluster_region() == (500, "CP")
    assert widget._cluster_region_scope_stack.index == 1
    assert widget._cluster_region_summary_label.text() == "CP (Caudoputamen)"

    widget._cluster_region_scope_combo.setCurrentText("Selected Rows")

    assert widget._selected_cluster_region() == (184, "FRP")
    assert widget._cluster_region_scope_stack.index == 2
    assert widget._cluster_region_summary_label.text() == "FRP (Frontal pole)"


def test_region_filter_editor_copies_defaults_and_keeps_rule_values() -> None:
    """Defaults affect new rules while existing include/exclude rows stay editable."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    editor = widget._current_table_region_filter_editor
    structures = {
        184: {"name": "Frontal pole", "acronym": "FRP"},
        500: {"name": "Caudoputamen", "acronym": "CP"},
    }
    editor.include_selector._structure_map = structures
    editor.exclude_selector._structure_map = structures

    editor.include_default_dilation.setValue(20)
    editor.include_selector.selected_ids = [184]
    editor._on_selection_changed(False)
    editor.include_default_dilation.setValue(40)
    editor.include_selector.selected_ids = [184, 500]
    editor._on_selection_changed(False)
    editor._set_dilation(False, 184, 10)

    editor.exclude_default_dilation.setValue(15)
    editor.exclude_selector.selected_ids = [500]
    editor._on_selection_changed(True)
    editor._set_node_types(500, (0, 2, 99))
    editor._set_minimum(500, 3)

    soma_setting_control = _DummyWidget()
    editor._exclude_soma_widgets = [soma_setting_control]
    widget._clustering_method_combo.setCurrentText("Voxel Correlation")
    assert not editor._soma_mode
    assert not soma_setting_control.isEnabled()
    widget._clustering_method_combo.setCurrentText("Soma Location")
    assert editor._soma_mode
    assert soma_setting_control.isEnabled()
    assert editor._exclude_settings[500].node_types == (0, 2, 99)
    assert editor._exclude_settings[500].minimum_node_count == 3

    region_filter = editor.region_filter(
        lambda region_ids: [
            (int(region_id), structures[int(region_id)]["acronym"])
            for region_id in region_ids
        ]
    )

    assert region_filter is not None
    assert [rule.dilation_fraction for rule in region_filter.include_rules] == [
        0.1,
        0.4,
    ]
    exclusion = region_filter.exclude_rules[0]
    assert exclusion.dilation_fraction == 0.15
    assert exclusion.node_types == (0, 2, 99)
    assert exclusion.minimum_node_count == 3


def test_region_filter_state_is_independent_for_each_input_scope() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    structures = {
        184: {"name": "Frontal pole", "acronym": "FRP"},
        500: {"name": "Caudoputamen", "acronym": "CP"},
    }
    whole = widget._whole_parquet_region_filter_editor
    current = widget._current_table_region_filter_editor
    selected = widget._selected_rows_region_filter_editor
    for editor in (whole, current, selected):
        editor.include_selector._structure_map = structures
        editor.exclude_selector._structure_map = structures

    whole.include_selector.selected_ids = [184]
    whole._on_selection_changed(False)
    current.exclude_selector.selected_ids = [500]
    current._on_selection_changed(True)
    selected.include_selector.selected_ids = [500]
    selected._on_selection_changed(False)

    widget._cluster_region_scope_combo.setCurrentText("Whole Parquet")
    assert widget._active_region_filter_editor() is whole
    assert set(whole._include_settings) == {184}
    assert not whole._exclude_settings

    widget._cluster_region_scope_combo.setCurrentText("Current Table")
    assert widget._active_region_filter_editor() is current
    assert not current._include_settings
    assert set(current._exclude_settings) == {500}

    widget._cluster_region_scope_combo.setCurrentText("Selected Rows")
    assert widget._active_region_filter_editor() is selected
    assert set(selected._include_settings) == {500}
    assert not selected._exclude_settings


def test_represented_region_ids_for_selection_expands_parent_to_dataset_descendants():
    """Parent selection for heatmaps should expand to represented descendant region IDs."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._atlas = _FakeAtlas(
        {
            997: {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "structure_id_path": [997],
            },
            184: {
                "id": 184,
                "acronym": "FRP",
                "name": "Frontal pole",
                "structure_id_path": [997, 184],
            },
            68: {
                "id": 68,
                "acronym": "FRP1",
                "name": "FRP1",
                "structure_id_path": [997, 184, 68],
            },
            667: {
                "id": 667,
                "acronym": "FRP2/3",
                "name": "FRP2/3",
                "structure_id_path": [997, 184, 667],
            },
            500: {
                "id": 500,
                "acronym": "CP",
                "name": "Caudoputamen",
                "structure_id_path": [997, 500],
            },
        }
    )
    widget._dataset_region_ids = {68, 667, 500}

    assert widget._represented_region_ids_for_selection(184) == [68, 667]
    assert widget._represented_region_ids_for_selection(500) == [500]


def test_update_region_summary_labels_uses_selected_region_and_blank_heatmap():
    """Summary labels should show the selected clustering region and blank heatmap state."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_region_selector = _make_selector(
        selected=(184, "FRP"),
        structure_map={184: {"name": "Frontal pole", "acronym": "FRP"}},
    )
    widget._heat_region_selector = _make_selector(selected=None)
    widget._cluster_region_summary_label = _DummyLabel()
    widget._heat_region_summary_label = _DummyLabel()

    widget._update_region_summary_labels()

    assert widget._cluster_region_summary_label.text() == "FRP (Frontal pole)"
    assert widget._heat_region_summary_label.text() == "All regions"


def test_update_region_summary_labels_compacts_multiple_cluster_regions():
    """Cluster summary should compact long direct-selection lists."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_region_selector = _make_selector(
        selected_ids=[184, 500, 315],
        structure_map={
            184: {"name": "Frontal pole", "acronym": "FRP"},
            500: {"name": "Caudoputamen", "acronym": "CP"},
            315: {"name": "Isocortex", "acronym": "ISO"},
        },
    )
    widget._heat_region_selector = _make_selector(selected=None)
    widget._cluster_region_summary_label = _DummyLabel()
    widget._heat_region_summary_label = _DummyLabel()

    widget._update_region_summary_labels()

    assert (
        widget._cluster_region_summary_label.text()
        == "FRP (Frontal pole), CP (Caudoputamen) +1 more"
    )


def test_run_clustering_pipeline_current_scope_requires_nonempty_table() -> None:
    """Current-table clustering scope should stop before launching workers when empty."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget.set_current_table_file_ids_provider(lambda: [])
    widget._cluster_region_scope_combo.setCurrentText("Current Table")
    widget._selected_cluster_region_selection = lambda: ClusterRegionSelection(
        selected_region_ids=[184],
        selected_region_acronyms=["FRP"],
        represented_region_ids=[68],
        represented_region_acronyms=["FRP1"],
    )
    widget._run_soma_clustering = MagicMock()
    widget._run_correlation_clustering = MagicMock()

    widget._run_clustering_pipeline()

    assert widget._progress_label.text() == (
        "Current table is empty; switch clustering scope to Whole Parquet or populate the table first."
    )
    widget._run_soma_clustering.assert_not_called()
    widget._run_correlation_clustering.assert_not_called()


def test_current_table_clustering_allows_no_target_region() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget.set_current_table_file_ids_provider(lambda: ["n1", "n2"])
    widget._cluster_region_scope_combo.setCurrentText("Current Table")
    widget._selected_cluster_region_selection = lambda: None
    widget._dilation_spin.setValue(40)
    widget._start_clustering_preflight = MagicMock()

    widget._run_clustering_pipeline()

    request = widget._start_clustering_preflight.call_args.args[0]
    assert request.file_ids == ("n1", "n2")
    assert request.region_selection is None
    assert request.dilation_fraction == 0.0
    assert widget._cluster_region_summary_label.text() == "All regions (optional)"
    assert widget._cluster_region_target_label.text() == "Included regions (optional):"


def test_selected_rows_clustering_allows_no_target_region() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget.set_selected_table_file_ids_provider(lambda: ["n2", "n3"])
    widget._cluster_region_scope_combo.setCurrentText("Selected Rows")
    widget._selected_cluster_region_selection = lambda: None
    widget._start_clustering_preflight = MagicMock()

    widget._run_clustering_pipeline()

    request = widget._start_clustering_preflight.call_args.args[0]
    assert request.file_ids == ("n2", "n3")
    assert request.region_selection is None


def test_current_table_clustering_allows_exclusion_only_region_filter() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    atlas = _FakeAtlas(
        {
            500: {
                "id": 500,
                "acronym": "CP",
                "name": "Caudoputamen",
                "structure_id_path": [500],
            }
        }
    )
    widget._db = object()
    widget._atlas = atlas
    widget._dataset_region_ids = {500}
    widget.set_current_table_file_ids_provider(lambda: ["n1", "n2"])
    widget._cluster_region_scope_combo.setCurrentText("Current Table")
    editor = widget._current_table_region_filter_editor
    editor.exclude_selector._structure_map = atlas.structures
    editor.exclude_selector.selected_ids = [500]
    editor._on_selection_changed(True)
    widget._start_clustering_preflight = MagicMock()

    widget._run_clustering_pipeline()

    request = widget._start_clustering_preflight.call_args.args[0]
    assert request.region_filter is not None
    assert request.region_filter.include_rules == ()
    assert [rule.region_id for rule in request.region_filter.exclude_rules] == [500]
    assert request.file_ids == ("n1", "n2")


def test_whole_parquet_clustering_still_requires_target_region() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget._selected_cluster_region_selection = lambda: None
    widget._start_clustering_preflight = MagicMock()

    widget._run_clustering_pipeline()

    assert widget._progress_label.text() == "Select at least one included region."
    widget._start_clustering_preflight.assert_not_called()


def test_run_clustering_pipeline_passes_current_table_file_ids_to_clustering() -> None:
    """Current-table clustering should snapshot its table subset for preflight."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget.set_current_table_file_ids_provider(lambda: ["n1", "n2"])
    widget._cluster_region_scope_combo.setCurrentText("Current Table")
    widget._clustering_method_combo.setCurrentText("Soma Location")
    widget._dilation_spin.setValue(35)
    selection = ClusterRegionSelection(
        selected_region_ids=[184],
        selected_region_acronyms=["FRP"],
        represented_region_ids=[68],
        represented_region_acronyms=["FRP1"],
    )
    widget._selected_cluster_region_selection = lambda: selection
    widget._start_clustering_preflight = MagicMock()

    widget._run_clustering_pipeline()

    request = widget._start_clustering_preflight.call_args.args[0]
    assert request.region_selection == selection
    assert request.dilation_fraction == 0.35
    assert request.file_ids == ("n1", "n2")


def test_run_clustering_pipeline_passes_selected_row_file_ids_to_clustering() -> None:
    """Selected Rows should pass only explicit row selections into workers."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget.set_current_table_file_ids_provider(lambda: ["n1", "n2", "n3"])
    widget.set_selected_table_file_ids_provider(lambda: ["n2", "n3", "n2"])
    widget._cluster_region_scope_combo.setCurrentText("Selected Rows")
    widget._clustering_method_combo.setCurrentText("Soma Location")
    selection = ClusterRegionSelection(
        selected_region_ids=[184],
        selected_region_acronyms=["FRP"],
        represented_region_ids=[68],
        represented_region_acronyms=["FRP1"],
    )
    widget._selected_cluster_region_selection = lambda: selection
    widget._start_clustering_preflight = MagicMock()

    widget._run_clustering_pipeline()

    request = widget._start_clustering_preflight.call_args.args[0]
    assert request.region_selection == selection
    assert request.file_ids == ("n2", "n3")
    assert widget._pending_cluster_context["input_scope"] == "selected"
    assert widget._pending_cluster_context["input_file_ids"] == ["n2", "n3"]


def test_ccf_voxel_clustering_passes_selected_row_file_ids() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget.set_selected_table_file_ids_provider(lambda: ["n1", "n3"])
    widget._cluster_region_scope_combo.setCurrentText("Selected Rows")
    widget._clustering_method_combo.setCurrentText("Voxel Correlation")
    selection = ClusterRegionSelection(
        selected_region_ids=[184],
        selected_region_acronyms=["FRP"],
        represented_region_ids=[68],
        represented_region_acronyms=["FRP1"],
    )
    widget._selected_cluster_region_selection = lambda: selection
    widget._start_clustering_preflight = MagicMock()

    widget._run_clustering_pipeline()

    request = widget._start_clustering_preflight.call_args.args[0]
    assert request.region_selection == selection
    assert request.file_ids == ("n1", "n3")


def test_selected_rows_clustering_requires_at_least_two_rows() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget.set_selected_table_file_ids_provider(lambda: ["n1"])
    widget._cluster_region_scope_combo.setCurrentText("Selected Rows")
    widget._run_correlation_clustering = MagicMock()

    widget._run_clustering_pipeline()

    assert "Only one table row is selected" in widget._progress_label.text()
    widget._run_correlation_clustering.assert_not_called()


def _captured_unfiltered_request(widget):
    widget._db = object()
    widget._atlas = object()
    widget.set_current_table_file_ids_provider(lambda: ["n1", "n2"])
    widget._cluster_region_scope_combo.setCurrentText("Current Table")
    widget._selected_cluster_region_selection = lambda: None
    widget._start_clustering_preflight = MagicMock()
    widget._run_clustering_pipeline()
    return widget._start_clustering_preflight.call_args.args[0]


def test_large_clustering_warning_text_formats_node_count() -> None:
    module = _import_analysis_tab_module()

    assert module._large_clustering_warning_text(10_000_001) == (
        "This clustering run will process 10,000,001 nodes, which exceeds the "
        "10,000,000-node warning threshold. Continue?"
    )


def test_preflight_at_threshold_launches_without_warning() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    request = _captured_unfiltered_request(widget)
    widget._pending_clustering_request = request
    widget._pending_clustering_preflight = types.SimpleNamespace(
        node_count=10_000_000,
        voxel_id_map=None,
    )
    widget._confirm_large_clustering_run = MagicMock(return_value=False)
    widget._launch_clustering_request = MagicMock()

    widget._on_clustering_preflight_thread_finished()

    widget._confirm_large_clustering_run.assert_not_called()
    widget._launch_clustering_request.assert_called_once_with(request, None)


def test_large_preflight_can_cancel_without_launching() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    request = _captured_unfiltered_request(widget)
    widget._pending_clustering_request = request
    widget._pending_clustering_preflight = types.SimpleNamespace(
        node_count=10_000_001,
        voxel_id_map=None,
    )
    widget._confirm_large_clustering_run = MagicMock(return_value=False)
    widget._launch_clustering_request = MagicMock()

    widget._on_clustering_preflight_thread_finished()

    widget._confirm_large_clustering_run.assert_called_once_with(10_000_001)
    widget._launch_clustering_request.assert_not_called()
    assert widget._progress_label.text() == (
        "Clustering cancelled; 10,000,001 nodes would have been processed."
    )
    assert widget._pending_cluster_context == {}


def test_large_preflight_confirmation_launches_prepared_request() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    request = _captured_unfiltered_request(widget)
    voxel_id_map = np.zeros((2, 2, 2), dtype=np.int32)
    widget._pending_clustering_request = request
    widget._pending_clustering_preflight = types.SimpleNamespace(
        node_count=10_500_000,
        voxel_id_map=voxel_id_map,
    )
    widget._confirm_large_clustering_run = MagicMock(return_value=True)
    widget._launch_clustering_request = MagicMock()

    widget._on_clustering_preflight_thread_finished()

    widget._launch_clustering_request.assert_called_once_with(
        request,
        voxel_id_map,
    )


def test_preflight_hands_prepared_region_filter_to_clustering_worker() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    request = _captured_unfiltered_request(widget)
    prepared = object()
    widget._pending_clustering_request = request
    widget._pending_clustering_preflight = types.SimpleNamespace(
        node_count=25,
        voxel_id_map=None,
        prepared_region_filter=prepared,
    )
    widget._confirm_large_clustering_run = MagicMock(return_value=True)
    widget._launch_clustering_request = MagicMock()

    widget._on_clustering_preflight_thread_finished()

    widget._launch_clustering_request.assert_called_once_with(
        request,
        None,
        prepared,
    )


def _enable_flatmap_coords(widget, styles=("both_shaped", "both_square")):
    """Force the widget to treat the loaded Parquet as flatmap-capable."""
    widget._detect_flatmap_coordinates = lambda: (True, styles)
    widget.refresh_flatmap_coordinate_availability()


def _coord_space_items(widget):
    return [item["text"] for item in widget._coordinate_space_combo._items]


def test_flatmap_space_hidden_without_coordinates() -> None:
    """Flat map + Depth must not appear when the Parquet lacks coordinates."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())

    assert _coord_space_items(widget) == ["CCFv3 Coordinates"]


def test_flatmap_space_appears_when_coordinates_present() -> None:
    """Flat map + Depth appears and styles populate when coords are present."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())

    _enable_flatmap_coords(widget)

    assert "Flat map + Depth" in _coord_space_items(widget)
    style_keys = [item["data"] for item in widget._flatmap_style_combo._items]
    assert style_keys == ["both_shaped", "both_square"]


def test_flatmap_space_removed_when_coordinates_absent() -> None:
    """Losing flatmap coords removes the option and resets to CCFv3 space."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    assert widget._current_coordinate_space() == "Flat map + Depth"

    widget._detect_flatmap_coordinates = lambda: (False, ())
    widget.refresh_flatmap_coordinate_availability()

    assert "Flat map + Depth" not in _coord_space_items(widget)
    assert widget._current_coordinate_space() == "CCFv3 Coordinates"


def test_flatmap_voxel_keeps_region_filters_and_shows_binning() -> None:
    """Flatmap voxel mode keeps anatomical filters and disables soma settings."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._clustering_method_combo.setCurrentText("Voxel Correlation")
    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")

    assert widget._cluster_region_scope_label._visible is True
    assert widget._cluster_region_scope_combo._visible is True
    assert widget._cluster_region_section._visible is True
    assert widget._cluster_exclude_region_label._visible is True
    assert all(
        not editor._soma_mode for editor in widget._region_filter_editors()
    )
    assert widget._dilation_label._visible is False
    assert widget._dilation_spin._visible is False
    assert widget._flatmap_style_combo._visible is True
    assert widget._flatmap_y_bins_spin._visible is True
    assert widget._flatmap_depth_bin_spin._visible is True
    assert widget._flatmap_include_depth_minus_one_cb._visible is True
    assert widget._flatmap_coords_status_label._visible is True


def test_flatmap_soma_hides_binning_and_shows_algorithm() -> None:
    """Flat map + Depth soma mode hides binning and shows the algorithm row."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    widget._clustering_method_combo.setCurrentText("Soma Location")

    assert widget._algorithm_combo._visible is True
    assert widget._flatmap_style_combo._visible is True
    assert widget._flatmap_y_bins_spin._visible is False
    assert widget._flatmap_depth_bin_spin._visible is False
    assert widget._flatmap_include_depth_minus_one_cb._visible is False
    assert widget._cluster_region_section._visible is True
    assert all(editor._soma_mode for editor in widget._region_filter_editors())


def test_depth_scale_is_soma_only_but_ignore_depth_serves_both() -> None:
    """A depth weight only makes sense where a distance is measured.

    Voxel correlation compares voxel occupancy, which has no notion of distance
    between voxels, so a depth scale there would merely duplicate the Depth bin
    control.  Ignoring depth is meaningful for both methods, though: soma
    clustering drops the axis and correlation collapses the grid's depth planes.
    """
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    _enable_flatmap_coords(widget, styles=("both_shaped",))

    # CCFv3 space has no flat map depth handling at all.
    widget._coordinate_space_combo.setCurrentText("CCFv3 Coordinates")
    widget._clustering_method_combo.setCurrentText("Soma Location")
    assert widget._flatmap_depth_scale_spin._visible is False
    assert widget._flatmap_ignore_depth_cb._visible is False

    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    widget._clustering_method_combo.setCurrentText("Voxel Correlation")
    assert widget._flatmap_ignore_depth_cb._visible is True
    assert widget._flatmap_depth_scale_spin._visible is False

    widget._clustering_method_combo.setCurrentText("Soma Location")
    assert widget._flatmap_ignore_depth_cb._visible is True
    assert widget._flatmap_depth_scale_spin._visible is True


def test_collapsing_depth_disables_the_depth_bin_size() -> None:
    """A collapsed voxel grid has no depth bins left to size.

    ``Include depth -1 plane`` must stay enabled, because depth still decides
    which nodes are counted even once the axis is gone.
    """
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    widget._clustering_method_combo.setCurrentText("Voxel Correlation")

    assert widget._flatmap_depth_bin_spin.isEnabled() is True
    widget._flatmap_ignore_depth_cb.setChecked(True)
    assert widget._flatmap_depth_bin_spin.isEnabled() is False
    assert widget._flatmap_include_depth_minus_one_cb.isEnabled() is True
    assert widget._flatmap_y_bins_spin.isEnabled() is True

    widget._flatmap_ignore_depth_cb.setChecked(False)
    assert widget._flatmap_depth_bin_spin.isEnabled() is True


def test_collapse_depth_reaches_the_correlation_worker(monkeypatch) -> None:
    """The checkbox must drive the correlation worker, not just the UI."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget._parquet_path = "neurons.parquet"
    widget.set_current_table_file_ids_provider(lambda: ["n1", "n2"])
    widget._cluster_region_scope_combo.setCurrentText("Current Table")
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    widget._clustering_method_combo.setCurrentText("Voxel Correlation")
    widget._flatmap_ignore_depth_cb.setChecked(True)
    widget._start_background_worker = MagicMock()
    widget._start_clustering_preflight = MagicMock()

    created_workers = []

    class _FakeWorker:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            created_workers.append(self)

    workers_module = types.ModuleType("napari_neuron_navigator.workers")
    workers_module.FlatmapParquetCorrelationWorker = _FakeWorker
    workers_module.CorrelationWorker = object
    workers_module.FlatmapSomaClusterWorker = object
    workers_module.SomaClusterWorker = object
    monkeypatch.setitem(sys.modules, "napari_neuron_navigator.workers", workers_module)

    widget._run_clustering_pipeline()
    request = widget._start_clustering_preflight.call_args.args[0]
    assert request.flatmap_include_depth is False
    widget._launch_clustering_request(request, None)

    assert len(created_workers) == 1
    assert created_workers[0].kwargs["collapse_depth"] is True


def test_depth_scale_tooltip_states_which_direction_weights_depth() -> None:
    """The tooltip must say which way the knob moves, not just that it exists."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())

    tooltip = widget._flatmap_depth_scale_spin.toolTip()
    assert "Higher values weight depth MORE" in tooltip
    assert "Lower values weight depth LESS" in tooltip
    assert widget._flatmap_depth_scale_label.toolTip() == tooltip


def test_depth_scale_tooltip_disclaims_micron_equivalence() -> None:
    """The depth scale is a ratio of axis fractions, not of physical distances.

    The flat map projection distorts the cortical surface, so flat map x/y have
    no reliable micron conversion and the weight must not be presented as a
    physical ratio.  The tooltip has to say so rather than stay silent, since a
    reader who assumes a micron equivalence would misinterpret every value.
    """
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())

    tooltip = widget._flatmap_depth_scale_spin.toolTip()
    assert "not physical distances" in tooltip
    assert "distorts" in tooltip
    assert "no reliable conversion to microns" in tooltip


def test_ignoring_depth_disables_the_depth_scale() -> None:
    """A disabled scale makes it obvious the value is no longer in play."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    widget._clustering_method_combo.setCurrentText("Soma Location")

    assert widget._flatmap_depth_scale_spin.isEnabled() is True
    widget._flatmap_ignore_depth_cb.setChecked(True)
    assert widget._flatmap_depth_scale_spin.isEnabled() is False
    widget._flatmap_ignore_depth_cb.setChecked(False)
    assert widget._flatmap_depth_scale_spin.isEnabled() is True


def test_ignore_depth_reaches_the_clustering_request() -> None:
    """The checkbox must actually change the request, not just the UI."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget._parquet_path = "neurons.parquet"
    widget.set_current_table_file_ids_provider(lambda: ["n1", "n2"])
    widget._cluster_region_scope_combo.setCurrentText("Current Table")
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    widget._clustering_method_combo.setCurrentText("Soma Location")
    widget._flatmap_ignore_depth_cb.setChecked(True)
    widget._flatmap_depth_scale_spin.setValue(3.5)
    widget._start_clustering_preflight = MagicMock()

    widget._run_clustering_pipeline()

    request = widget._start_clustering_preflight.call_args.args[0]
    assert request.flatmap_include_depth is False
    assert request.flatmap_depth_scale == 3.5


def test_eps_units_switch_between_microns_and_hemisphere_fraction() -> None:
    """Flat map space measures eps in normalized units, not microns.

    A micron-valued eps in normalized space would exceed the whole hemisphere
    and collapse every soma into one cluster, so the control has to retarget.
    """
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._coordinate_space_combo.setCurrentText("CCFv3 Coordinates")
    widget._clustering_method_combo.setCurrentText("Soma Location")

    assert widget._eps_label.text() == "Eps (μm):"
    assert widget._eps_spin.value() == 100.0

    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    assert "hemisphere" in widget._eps_label.text()
    assert widget._eps_spin.value() == pytest.approx(0.05)


def test_eps_remembers_each_spaces_value_across_switches() -> None:
    """Switching coordinate space must not destroy the other space's setting."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._coordinate_space_combo.setCurrentText("CCFv3 Coordinates")
    widget._clustering_method_combo.setCurrentText("Soma Location")
    widget._eps_spin.setValue(250.0)

    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    widget._eps_spin.setValue(0.2)
    widget._coordinate_space_combo.setCurrentText("CCFv3 Coordinates")
    assert widget._eps_spin.value() == 250.0

    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    assert widget._eps_spin.value() == pytest.approx(0.2)


def test_ccf_voxel_correlation_unaffected_by_flatmap_availability() -> None:
    """CCFv3 space still runs atlas voxel correlation even when coords exist."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    _enable_flatmap_coords(widget)
    selection = ClusterRegionSelection(
        selected_region_ids=[184],
        selected_region_acronyms=["FRP"],
        represented_region_ids=[68],
        represented_region_acronyms=["FRP1"],
    )
    widget._selected_cluster_region_selection = lambda: selection
    widget._start_clustering_preflight = MagicMock()

    widget._run_clustering_pipeline()

    request = widget._start_clustering_preflight.call_args.args[0]
    assert request.coordinate_space == "CCFv3 Coordinates"
    assert request.region_selection == selection


def test_flatmap_voxel_dispatch_constructs_parquet_worker(monkeypatch) -> None:
    """Flat map voxel correlation launches FlatmapParquetCorrelationWorker."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget._parquet_path = "neurons.parquet"
    widget.set_current_table_file_ids_provider(lambda: ["n1", "n2"])
    widget._cluster_region_scope_combo.setCurrentText("Current Table")
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    widget._clustering_method_combo.setCurrentText("Voxel Correlation")
    widget._method_combo.setCurrentText("complete")
    widget._n_clusters_spin.setValue(7)
    widget._flatmap_y_bins_spin.setValue(128)
    widget._flatmap_depth_bin_spin.setValue(50.0)
    widget._flatmap_include_depth_minus_one_cb.setChecked(False)
    widget._start_background_worker = MagicMock()
    widget._start_clustering_preflight = MagicMock()

    created_workers = []

    class _FakeFlatmapParquetCorrelationWorker:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            created_workers.append(self)

    workers_module = types.ModuleType("napari_neuron_navigator.workers")
    workers_module.FlatmapParquetCorrelationWorker = (
        _FakeFlatmapParquetCorrelationWorker
    )
    workers_module.CorrelationWorker = object
    workers_module.FlatmapSomaClusterWorker = object
    workers_module.SomaClusterWorker = object
    monkeypatch.setitem(sys.modules, "napari_neuron_navigator.workers", workers_module)

    widget._run_clustering_pipeline()
    request = widget._start_clustering_preflight.call_args.args[0]
    widget._launch_clustering_request(request, None)

    assert len(created_workers) == 1
    assert created_workers[0].kwargs == {
        "parquet_path": "neurons.parquet",
        "atlas": widget._atlas,
        "style": "both_shaped",
        "y_bins": 128,
        "depth_bin_um": 50.0,
        "include_depth_minus_one": False,
        "linkage_method": "complete",
        "n_clusters": 7,
        "file_ids": ["n1", "n2"],
        "collapse_depth": False,
    }
    widget._start_background_worker.assert_called_once_with(
        created_workers[0],
        widget._on_correlation_finished,
    )


def test_flatmap_soma_dispatch_constructs_soma_worker(monkeypatch) -> None:
    """Flat map soma clustering launches FlatmapSomaClusterWorker."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget._parquet_path = "neurons.parquet"
    widget.set_current_table_file_ids_provider(lambda: ["n1", "n2"])
    widget._cluster_region_scope_combo.setCurrentText("Current Table")
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    widget._clustering_method_combo.setCurrentText("Soma Location")
    widget._algorithm_combo.setCurrentText("K-Means")
    widget._method_combo.setCurrentText("ward")
    widget._n_clusters_spin.setValue(4)
    widget._eps_spin.setValue(120.0)
    widget._min_samples_spin.setValue(6)
    widget._start_background_worker = MagicMock()
    widget._start_clustering_preflight = MagicMock()

    created_workers = []

    class _FakeFlatmapSomaClusterWorker:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            created_workers.append(self)

    workers_module = types.ModuleType("napari_neuron_navigator.workers")
    workers_module.FlatmapSomaClusterWorker = _FakeFlatmapSomaClusterWorker
    workers_module.CorrelationWorker = object
    workers_module.FlatmapParquetCorrelationWorker = object
    workers_module.SomaClusterWorker = object
    monkeypatch.setitem(sys.modules, "napari_neuron_navigator.workers", workers_module)

    widget._run_clustering_pipeline()
    request = widget._start_clustering_preflight.call_args.args[0]
    widget._launch_clustering_request(request, None)

    assert len(created_workers) == 1
    assert created_workers[0].kwargs == {
        "parquet_path": "neurons.parquet",
        "atlas": widget._atlas,
        "style": "both_shaped",
        "algorithm": "kmeans",
        "linkage_method": "ward",
        "n_clusters": 4,
        "eps": 120.0,
        "min_samples": 6,
        "file_ids": ["n1", "n2"],
        "depth_scale": 1.0,
        "include_depth": True,
    }
    widget._start_background_worker.assert_called_once_with(
        created_workers[0],
        widget._on_correlation_finished,
    )


def test_flatmap_selected_rows_scope_passes_file_ids_to_worker(monkeypatch) -> None:
    """Flatmap clustering should honor the same Selected Rows scope."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget._parquet_path = "neurons.parquet"
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    widget._clustering_method_combo.setCurrentText("Voxel Correlation")
    widget._cluster_region_scope_combo.setCurrentText("Selected Rows")
    widget.set_selected_table_file_ids_provider(lambda: ["n2", "n3"])
    widget._start_background_worker = MagicMock()
    widget._start_clustering_preflight = MagicMock()
    created_workers = []

    class _FakeFlatmapParquetCorrelationWorker:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            created_workers.append(self)

    workers_module = types.ModuleType("napari_neuron_navigator.workers")
    workers_module.FlatmapParquetCorrelationWorker = (
        _FakeFlatmapParquetCorrelationWorker
    )
    workers_module.CorrelationWorker = object
    workers_module.FlatmapSomaClusterWorker = object
    workers_module.SomaClusterWorker = object
    monkeypatch.setitem(sys.modules, "napari_neuron_navigator.workers", workers_module)

    widget._run_clustering_pipeline()
    request = widget._start_clustering_preflight.call_args.args[0]
    widget._launch_clustering_request(request, None)

    assert created_workers[0].kwargs["file_ids"] == ["n2", "n3"]
    assert widget._pending_cluster_context["coordinate_space"] == ("Flat map + Depth")


def test_flatmap_soma_selected_rows_scope_passes_file_ids_to_worker(
    monkeypatch,
) -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget._parquet_path = "neurons.parquet"
    _enable_flatmap_coords(widget, styles=("both_shaped",))
    widget._coordinate_space_combo.setCurrentText("Flat map + Depth")
    widget._clustering_method_combo.setCurrentText("Soma Location")
    widget._cluster_region_scope_combo.setCurrentText("Selected Rows")
    widget.set_selected_table_file_ids_provider(lambda: ["n2", "n3"])
    widget._start_background_worker = MagicMock()
    widget._start_clustering_preflight = MagicMock()
    created_workers = []

    class _FakeFlatmapSomaClusterWorker:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            created_workers.append(self)

    workers_module = types.ModuleType("napari_neuron_navigator.workers")
    workers_module.FlatmapSomaClusterWorker = _FakeFlatmapSomaClusterWorker
    workers_module.CorrelationWorker = object
    workers_module.FlatmapParquetCorrelationWorker = object
    workers_module.SomaClusterWorker = object
    monkeypatch.setitem(sys.modules, "napari_neuron_navigator.workers", workers_module)

    widget._run_clustering_pipeline()
    request = widget._start_clustering_preflight.call_args.args[0]
    widget._launch_clustering_request(request, None)

    assert created_workers[0].kwargs["file_ids"] == ["n2", "n3"]


def test_update_button_states_enables_export_controls_after_clustering():
    """Export controls should enable only when a clustering result exists and the UI is idle."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget._last_cluster_result = object()

    widget._update_button_states()

    assert widget._render_clustermap_btn.isEnabled()
    assert widget._save_cluster_workbook_btn.isEnabled()
    assert widget._save_distance_workbook_btn.isEnabled()
    assert widget._save_extended_parquet_btn.isEnabled()
    assert widget._save_dendrogram_btn.isEnabled()


def test_update_button_states_enables_bulk_heatmap_button_with_cluster_options():
    """Bulk heatmap action should enable only when concrete cluster entries exist."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget._last_cluster_result = object()
    widget._heat_cluster_combo.addItem("Cluster 1")
    widget._heat_cluster_combo.setItemData(1, 1)

    widget._update_button_states()

    assert widget._add_all_cluster_heatmaps_btn.isEnabled()


def test_update_button_states_disables_export_controls_without_result():
    """Export controls should stay disabled until clustering has completed."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget._last_cluster_result = None

    widget._update_button_states()

    assert not widget._render_clustermap_btn.isEnabled()
    assert not widget._save_cluster_workbook_btn.isEnabled()
    assert not widget._save_distance_workbook_btn.isEnabled()
    assert not widget._save_extended_parquet_btn.isEnabled()
    assert not widget._save_dendrogram_btn.isEnabled()
    assert not widget._add_all_cluster_heatmaps_btn.isEnabled()


def test_on_correlation_finished_leaves_clustermap_unrendered():
    """Clustering completion should not auto-render the dendrogram preview."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._n_clusters_spin = _DummySpinBox()
    widget._n_clusters_spin.setValue(2)
    widget._progress_bar = _DummyProgressBar()
    widget._progress_label = _DummyLabel()
    widget._viewer = _DummyViewer()
    widget._slice_projector = None
    widget._update_button_states = lambda: None
    widget._update_cluster_filter_combo = lambda: None
    placeholder_messages: list[str] = []
    widget._show_clustermap_message = lambda message: placeholder_messages.append(
        message
    )
    draw_calls: list[object] = []
    widget._draw_clustermap = lambda result: draw_calls.append(result)
    emitted: list[tuple[object, dict]] = []
    widget.cluster_colors_updated.connect(
        lambda result, color_map: emitted.append((result, color_map))
    )

    result = types.SimpleNamespace(
        neuron_ids=["n1", "n2"],
        labels=np.array([1, 2], dtype=np.int32),
    )

    widget._on_correlation_finished(result)

    assert widget._last_cluster_result is result
    assert draw_calls == []
    assert emitted == [(result, widget._cluster_color_map)]
    assert placeholder_messages == [
        "Clustering complete. Click Render Dendrogram to view."
    ]
    assert "Table updated and sorted by cluster." in widget._progress_label.text()
    assert (
        "0 neuron(s) are unclustered after region/coordinate filtering."
        in widget._progress_label.text()
    )
    assert "Auto-colored" not in widget._progress_label.text()


def test_render_clustermap_requires_button_press():
    """Render button handler should draw only when a result is available."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    result = types.SimpleNamespace()
    widget._last_cluster_result = result
    draw_calls: list[object] = []
    widget._draw_clustermap = lambda cluster_result: draw_calls.append(cluster_result)

    widget._render_clustermap_requested()

    assert draw_calls == [result]


def test_on_correlation_finished_auto_colors_rendered_layers_and_emits_updates():
    """Clustering completion should auto-color rendered neurons and update the table."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    lines_layer = MagicMock()
    lines_layer.name = "Neuron Lines"
    lines_layer.metadata = {
        "file_ids": ["n1", "n2"],
        "segments_per_neuron": [3, 5],
    }
    widget._viewer = types.SimpleNamespace(layers=[lines_layer])
    widget._slice_projector = None
    widget._n_clusters_spin = _DummySpinBox()
    widget._n_clusters_spin.setValue(2)
    widget._progress_bar = _DummyProgressBar()
    widget._progress_label = _DummyLabel()
    widget._update_button_states = lambda: None
    widget._update_cluster_filter_combo = lambda: None
    result = types.SimpleNamespace(
        neuron_ids=["n1", "n2"],
        labels=np.array([1, 2], dtype=np.int32),
    )
    emitted: list[tuple[object, dict]] = []
    widget.cluster_colors_updated.connect(
        lambda result, color_map: emitted.append((result, color_map))
    )

    widget._on_correlation_finished(result)

    color_array = lines_layer.edge_color
    assert isinstance(color_array, np.ndarray)
    assert color_array.shape == (8, 4)
    assert emitted == [(result, widget._cluster_color_map)]
    assert "Table updated and sorted by cluster." in widget._progress_label.text()
    assert (
        "Auto-colored 2/2 rendered neurons by cluster." in widget._progress_label.text()
    )


def test_completed_selected_run_saves_sparse_assignment_and_lineage() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    store = ClusterAssignmentStore()
    parent = store.add(
        name="Soma Location 1",
        assignments={"n1": 1, "n2": 1, "n3": 2},
        input_file_ids=["n1", "n2", "n3"],
    )
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_assignment_store = store
    widget._pending_cluster_context = {
        "method_name": "Voxel Correlation",
        "input_scope": "selected",
        "input_file_ids": ["n1", "n2"],
        "coordinate_space": "CCFv3 Coordinates",
        "parent_assignment_id": parent.assignment_id,
        "parent_cluster_ids": [1],
    }
    widget._cluster_label_colors = {
        1: [0.1, 0.2, 0.3, 1.0],
        2: [0.8, 0.7, 0.6, 1.0],
    }
    result = ClusterResult(
        correlation_matrix=np.eye(2, dtype=np.float32),
        distance_matrix=np.zeros((2, 2), dtype=np.float32),
        linkage_matrix=np.zeros((1, 4), dtype=np.float64),
        neuron_ids=["n1", "n2"],
        reorder_indices=np.array([0, 1], dtype=np.intp),
        labels=np.array([2, 1], dtype=np.int32),
    )

    widget._save_cluster_assignment(result)

    saved = store.active
    assert saved is not None
    assert saved.name == "Voxel Correlation 1"
    assert saved.assignments == {"n1": 2, "n2": 1}
    assert saved.label_for("n3") is None
    assert saved.parent_assignment_id == parent.assignment_id
    assert saved.parent_cluster_ids == (1,)
    assert saved.runtime_result is result


def test_restored_assignment_drives_heatmap_groups_without_runtime_matrices() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    store = ClusterAssignmentStore()
    store.add(
        name="Restored Groups",
        assignments={"n1": 1, "n2": 2, "n3": 1},
        input_file_ids=["n1", "n2", "n3"],
        label_colors={
            1: [0.1, 0.2, 0.3, 1.0],
            2: [0.8, 0.7, 0.6, 1.0],
        },
    )

    widget.set_cluster_assignment_store(store)

    assert widget._last_cluster_result is None
    assert widget._cluster_file_ids(1) == ("n1", "n3")
    assert widget._heat_cluster_combo.count() == 3
    assert "Rerun required" in widget._progress_label.text()


def test_deleting_last_assignment_clears_live_analysis_result() -> None:
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    store = ClusterAssignmentStore()
    assignment = store.add(
        name="Soma Location 1",
        assignments={"n1": 1, "n2": 2},
        input_file_ids=["n1", "n2"],
        runtime_result=object(),
    )
    widget.set_cluster_assignment_store(store)

    store.delete(assignment.assignment_id)
    widget.on_active_cluster_assignment_changed()

    assert widget._last_cluster_result is None
    assert widget._cluster_color_map is None
    assert widget._heat_cluster_combo.count() == 1
    assert not widget._save_distance_workbook_btn.isEnabled()
    assert not widget._save_distance_workbook_btn.isEnabled()


def test_has_cached_clusters_for_current_table_false_without_cache_or_provider():
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._last_cluster_result = None
    widget._cluster_color_map = None
    widget._current_table_file_ids_provider = None

    assert widget.has_cached_clusters_for_current_table() is False


def test_has_cached_clusters_for_current_table_true_with_overlap_and_mixed_id_types():
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._last_cluster_result = types.SimpleNamespace(neuron_ids=["1", "n2"])
    widget._cluster_color_map = {"1": [0.1, 0.2, 0.3, 1.0], "n2": [0.4, 0.5, 0.6, 1.0]}
    widget.set_current_table_file_ids_provider(lambda: [1, "x"])

    assert widget.has_cached_clusters_for_current_table() is True


def test_apply_cluster_colors_emits_updates_and_returns_table_match_summary():
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._viewer = _DummyViewer()
    widget._slice_projector = None
    widget._last_cluster_result = types.SimpleNamespace(
        neuron_ids=["n1", "n2"],
        labels=np.array([1, 2], dtype=np.int32),
    )
    widget._cluster_color_map = {
        "n1": [0.12, 0.47, 0.71, 1.0],
        "n2": [0.84, 0.15, 0.16, 1.0],
    }
    widget.set_current_table_file_ids_provider(lambda: ["n2", "other"])
    emitted: list[tuple[object, dict]] = []
    widget.cluster_colors_updated.connect(
        lambda result, color_map: emitted.append((result, color_map))
    )

    summary = widget.apply_cluster_colors()

    assert summary.matched_table_count == 1
    assert summary.rendered_count == 0
    assert summary.colored_count == 0
    assert summary.gray_count == 0
    assert emitted == [(widget._last_cluster_result, widget._cluster_color_map)]


def test_on_heatmap_finished_adds_stable_analysis_contrast_limits():
    """Analysis heatmaps should be added with explicit full-volume contrast limits."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._viewer = _DummyViewer()
    widget._progress_bar = _DummyProgressBar()
    widget._progress_label = _DummyLabel()
    widget._atlas = types.SimpleNamespace(atlas_name="fake_atlas")
    widget._completed_heatmap_requests = []
    widget._active_heatmap_total = 1
    widget._active_heatmap_index = 1
    widget._heatmap_batch_mode = False
    widget._cluster_label_colors = {1: [0.1, 0.2, 0.3, 1.0]}
    widget._current_heatmap_request = module._HeatmapRequest(
        selected_region_id=567,
        selected_region_acronym="CH",
        region_ids=(567, 568),
        cluster_label=1,
        file_ids=("n1", "n2"),
        node_types=(3, 4),
        soma_radius_um=100.0,
        depth_bin_factor=3,
        depth_axis=1,
        assignment_id="assignment-a",
        assignment_name="Run A",
    )

    volume = np.zeros((2, 3, 4), dtype=np.float32)
    volume[1, 2, 3] = 7.0

    previous = _install_fake_napari_colormaps()
    try:
        widget._on_heatmap_finished(volume)
    finally:
        _restore_modules(previous)

    layer = widget._heatmap_layer
    assert layer.name == (
        "Cluster 1 CH (Basal dendrite + Apical dendrite, 100 μm soma radius) Heatmap"
    )
    assert layer.contrast_limits == (0.0, 7.0)
    assert layer.contrast_limits_range == (0.0, 7.0)
    assert layer.scale == [1.0, 3.0, 1.0]
    assert layer.metadata["heatmap_kind"] == "analysis"
    assert layer.metadata["heatmap_contrast_limits"] == (0.0, 7.0)
    assert layer.metadata["heatmap_selected_region_id"] == 567
    assert layer.metadata["heatmap_selected_region_acronym"] == "CH"
    assert layer.metadata["heatmap_region_ids"] == [567, 568]
    assert layer.metadata["heatmap_node_types"] == [3, 4]
    assert layer.metadata["heatmap_node_type_labels"] == [
        "Basal dendrite",
        "Apical dendrite",
    ]
    assert layer.metadata["heatmap_soma_radius_um"] == 100.0
    assert layer.metadata["file_ids"] == ["n1", "n2"]
    assert layer.metadata["source_file_ids"] == ["n1", "n2"]
    assert layer.metadata["heatmap_autocontrast_policy"] == "stable_full_volume"
    assert layer.metadata["comparison_source_id"]
    assert layer.metadata["comparison_assignment_id"] == "assignment-a"
    assert layer.metadata["comparison_assignment_name"] == "Run A"
    assert layer.metadata["comparison_atlas_provenance"] == {
        "atlas_name": "fake_atlas",
        "resolution_um": [1.0, 1.0, 1.0],
        "atlas_version": None,
    }
    assert layer.metadata["comparison_filter_signature"] == {
        "atlas_provenance": {
            "atlas_name": "fake_atlas",
            "resolution_um": [1.0, 1.0, 1.0],
            "atlas_version": None,
        },
        "region_ids": [567, 568],
        "node_types": [3, 4],
        "soma_radius_um": 100.0,
        "depth_axis": 1,
        "depth_bin_factor": 3,
    }


def test_all_cluster_heatmap_requests_excludes_all_neurons_entry():
    """Bulk heatmap request generation should include only concrete clusters."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._heat_cluster_combo = _DummyCombo()
    widget._heat_cluster_combo.addItem("All neurons")
    widget._heat_cluster_combo.addItem("Cluster 1")
    widget._heat_cluster_combo.setItemData(1, 1)
    widget._heat_cluster_combo.addItem("Cluster 2")
    widget._heat_cluster_combo.setItemData(2, 2)
    widget._last_cluster_result = types.SimpleNamespace(
        neuron_ids=["n1", "n2", "n3"],
        labels=np.array([1, 2, 2], dtype=np.int32),
    )
    widget._depth_bin_spin = _DummySpinBox()
    widget._depth_bin_spin.setValue(4)
    widget._current_depth_axis = lambda: 2
    widget._selected_heat_region = lambda: None
    widget._progress_label = _DummyLabel()

    requests = widget._all_cluster_heatmap_requests()

    assert [request.cluster_label for request in requests] == [1, 2]
    assert [request.file_ids for request in requests] == [("n1",), ("n2", "n3")]
    assert all(request.depth_bin_factor == 4 for request in requests)
    assert all(request.depth_axis == 2 for request in requests)


def test_bulk_heatmap_queue_advances_and_summarizes_completion():
    """Queued cluster heatmaps should advance one-by-one and finish with a summary."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    request1 = module._HeatmapRequest(
        selected_region_id=10,
        selected_region_acronym="CH",
        region_ids=(10,),
        cluster_label=1,
        file_ids=("n1",),
        node_types=None,
        soma_radius_um=None,
        depth_bin_factor=2,
        depth_axis=1,
    )
    request2 = module._HeatmapRequest(
        selected_region_id=10,
        selected_region_acronym="CH",
        region_ids=(10,),
        cluster_label=2,
        file_ids=("n2",),
        node_types=None,
        soma_radius_um=None,
        depth_bin_factor=2,
        depth_axis=1,
    )

    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._viewer = _DummyViewer()
    widget._atlas = types.SimpleNamespace(atlas_name="fake_atlas")
    widget._progress_bar = _DummyProgressBar()
    widget._progress_label = _DummyLabel()
    widget._cluster_label_colors = {
        1: [0.1, 0.2, 0.3, 1.0],
        2: [0.6, 0.5, 0.4, 1.0],
    }
    widget._completed_heatmap_requests = []
    widget._pending_heatmap_requests = [request2]
    widget._current_heatmap_request = request1
    widget._active_heatmap_total = 2
    widget._active_heatmap_index = 1
    widget._heatmap_batch_mode = True
    widget._on_thread_finished = lambda: None
    widget._update_button_states = lambda: None
    started: list[str] = []
    widget._start_next_heatmap_request = lambda: started.append("next")

    volume1 = np.zeros((2, 2, 2), dtype=np.float32)
    volume1[0, 1, 1] = 3.0
    volume2 = np.zeros((2, 2, 2), dtype=np.float32)
    volume2[1, 0, 1] = 5.0

    previous = _install_fake_napari_colormaps()
    try:
        widget._on_heatmap_finished(volume1)
        widget._on_heatmap_thread_finished()
        widget._current_heatmap_request = request2
        widget._pending_heatmap_requests = []
        widget._active_heatmap_index = 2
        widget._on_heatmap_finished(volume2)
        widget._on_heatmap_thread_finished()
    finally:
        _restore_modules(previous)

    assert started == ["next"]
    assert [layer.name for layer in widget._viewer.layers] == [
        "Cluster 1 CH Heatmap",
        "Cluster 2 CH Heatmap",
    ]
    assert widget._viewer.layers[0].metadata["heatmap_cluster"] == 1
    assert widget._viewer.layers[1].metadata["heatmap_cluster"] == 2
    assert widget._progress_label.text() == "Added 2 cluster heatmaps to scene"
    assert widget._last_heatmap_requests == [request1, request2]


def test_dims_order_rebuilds_tracked_heatmap_request_set():
    """Dims-order changes should reuse the last built heatmap requests."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._db = object()
    widget._atlas = object()
    widget._worker_thread = None
    widget._current_depth_axis = lambda: 2
    widget._last_heatmap_requests = [
        module._HeatmapRequest(
            selected_region_id=10,
            selected_region_acronym="CH",
            region_ids=(10,),
            cluster_label=5,
            file_ids=("n5",),
            node_types=(2,),
            soma_radius_um=50.0,
            depth_bin_factor=3,
            depth_axis=0,
        ),
        module._HeatmapRequest(
            selected_region_id=None,
            selected_region_acronym=None,
            region_ids=None,
            cluster_label=None,
            file_ids=None,
            node_types=None,
            soma_radius_um=None,
            depth_bin_factor=1,
            depth_axis=1,
        ),
    ]
    calls: list[tuple[list[object], bool]] = []
    widget._start_heatmap_requests = lambda requests, batch_mode: calls.append(
        (requests, batch_mode)
    )

    widget._on_dims_order_changed()

    assert len(calls) == 1
    requests, batch_mode = calls[0]
    assert batch_mode is True
    assert [request.cluster_label for request in requests] == [5, None]
    assert [request.file_ids for request in requests] == [("n5",), None]
    assert [request.node_types for request in requests] == [(2,), None]
    assert [request.soma_radius_um for request in requests] == [50.0, None]
    assert [request.depth_axis for request in requests] == [2, 2]


def test_analysis_heatmap_workaround_swallows_thumbnail_rank_mismatch():
    """Known napari thumbnail rank errors should be suppressed for analysis heatmaps."""
    module = _import_analysis_tab_module()

    class _CrashLayer(_DummyImageLayer):
        def _update_thumbnail(self) -> None:
            raise RuntimeError("sequence argument must have length equal to input rank")

    layer = _CrashLayer(
        np.zeros((2, 2, 2), dtype=np.float32),
        name="Cluster 1 CH Heatmap",
        metadata={
            "heatmap_kind": "analysis",
            "heatmap_contrast_limits": (0.0, 5.0),
        },
        contrast_limits=(0.0, 5.0),
        contrast_limits_range=(0.0, 5.0),
    )

    module._install_analysis_heatmap_layer_workarounds(layer)

    layer._update_thumbnail()
    assert layer._analysis_heatmap_thumbnail_warning_logged is True


def test_analysis_heatmap_workaround_resets_to_stored_limits():
    """Reset hooks should restore stored full-volume contrast limits."""
    module = _import_analysis_tab_module()
    layer = _DummyImageLayer(
        np.zeros((2, 2, 2), dtype=np.float32),
        name="Cluster 1 CH Heatmap",
        metadata={
            "heatmap_kind": "analysis",
            "heatmap_contrast_limits": (0.0, 9.0),
        },
        contrast_limits=(1.0, 2.0),
        contrast_limits_range=(1.0, 2.0),
    )
    layer._slice_input = types.SimpleNamespace(ndisplay=3)

    module._install_analysis_heatmap_layer_workarounds(layer)
    layer.reset_contrast_limits()

    assert layer.contrast_limits == (0.0, 9.0)
    assert layer.contrast_limits_range == (0.0, 9.0)

    layer.contrast_limits_range = (3.0, 4.0)
    layer.reset_contrast_limits_range()

    assert layer.contrast_limits_range == (0.0, 9.0)


def test_analysis_heatmap_workaround_keeps_stable_limits_during_slice_updates():
    """Continuous auto-contrast should not replace stored full-volume heatmap limits."""
    module = _import_analysis_tab_module()
    layer = _DummyImageLayer(
        np.zeros((2, 2, 2), dtype=np.float32),
        name="Cluster 1 CH Heatmap",
        metadata={
            "heatmap_kind": "analysis",
            "heatmap_contrast_limits": (0.0, 11.0),
        },
        contrast_limits=(0.0, 11.0),
        contrast_limits_range=(0.0, 11.0),
    )
    layer._keep_auto_contrast = True
    layer._slice_input = types.SimpleNamespace(ndisplay=3)

    module._install_analysis_heatmap_layer_workarounds(layer)
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


def test_analysis_heatmap_workaround_preserves_2d_auto_contrast_behavior():
    """2D continuous auto-contrast should still use the original napari path."""
    module = _import_analysis_tab_module()
    layer = _DummyImageLayer(
        np.zeros((2, 2, 2), dtype=np.float32),
        name="Cluster 1 CH Heatmap",
        metadata={
            "heatmap_kind": "analysis",
            "heatmap_contrast_limits": (0.0, 11.0),
        },
        contrast_limits=(0.0, 11.0),
        contrast_limits_range=(0.0, 11.0),
    )
    layer._keep_auto_contrast = True
    layer._slice_input = types.SimpleNamespace(ndisplay=2)

    module._install_analysis_heatmap_layer_workarounds(layer)
    response = types.SimpleNamespace(
        slice_input=types.SimpleNamespace(ndisplay=2),
        payload={"slice": 2},
    )
    result = layer._update_slice_response(response)

    assert result is response
    assert layer.slice_updates == [response]
    assert layer.contrast_limits == (4.0, 5.0)
    assert layer.contrast_limits_range == (4.0, 5.0)

    layer.reset_contrast_limits()
    assert layer.contrast_limits == (-1.0, -1.0)


def test_analysis_heatmap_contrast_limits_fallback_for_zero_volume():
    """All-zero analysis heatmaps should keep the default visible contrast range."""
    module = _import_analysis_tab_module()

    limits = module._analysis_heatmap_contrast_limits(
        np.zeros((3, 3, 3), dtype=np.float32)
    )

    assert limits == (0.0, 1.0)


def test_draw_clustermap_emits_debug_logs(caplog):
    """Clustermap drawing should use backend-managed figure size once."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._clustermap_rendered = False
    widget._clustermap_section = _DummyCollapsibleSection("Clustermap", expanded=True)
    widget._figure = _DummyFigure()
    widget._figure.size_inches = (5.0, 3.0)
    widget._canvas = _DummyCanvas(widget._figure)
    widget._canvas.draw = MagicMock()
    widget._canvas.draw_idle = MagicMock()
    widget._cluster_color_map = {
        "n1": [0.1, 0.2, 0.3, 1.0],
        "n2": [0.2, 0.3, 0.4, 1.0],
    }
    populate_calls: list[tuple[object, object, object, tuple[float, float], int]] = []

    def _fake_populate(figure, result, cluster_color_map, *, figsize, dpi):
        populate_calls.append(
            (figure, result, cluster_color_map, tuple(figsize), int(dpi))
        )
        return figure

    module._populate_embedded_clustermap_figure = _fake_populate
    result = ClusterResult(
        correlation_matrix=np.eye(2, dtype=np.float32),
        distance_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        linkage_matrix=np.array([[0.0, 1.0, 1.0, 2.0]], dtype=np.float64),
        neuron_ids=["n1", "n2"],
        reorder_indices=np.array([0, 1], dtype=np.intp),
        labels=np.array([1, 2], dtype=np.int32),
    )

    with caplog.at_level(
        logging.DEBUG, logger="napari_neuron_navigator.widgets.analysis_tab"
    ):
        widget._draw_clustermap(result)

    messages = [record.getMessage() for record in caplog.records]
    assert any("_draw_clustermap start" in message for message in messages)
    assert any("populate_clustermap_figure complete" in message for message in messages)
    assert any(
        "_draw_clustermap canvas draw complete" in message for message in messages
    )
    assert populate_calls == [
        (
            widget._figure,
            result,
            widget._cluster_color_map,
            (5.0, 3.0),
            100,
        )
    ]
    widget._canvas.draw.assert_called_once_with()
    widget._canvas.draw_idle.assert_not_called()
    assert widget._figure.canvas is None
    assert widget._figure.size_inches == (5.0, 3.0)
    assert widget._figure.set_size_inches_calls == 0


def test_draw_clustermap_uses_physical_canvas_size_when_figure_size_unavailable():
    """Preview sizing should fall back to physical canvas pixels on HiDPI displays."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._clustermap_rendered = False
    widget._figure = _DummyFigure()
    widget._figure.size_inches = None
    widget._canvas = _DummyCanvas(widget._figure)
    widget._canvas._width = 600
    widget._canvas._height = 400
    widget._canvas._physical_width = 1200
    widget._canvas._physical_height = 800
    widget._canvas.draw = MagicMock()
    widget._cluster_color_map = None
    populate_calls: list[tuple[object, object, object, tuple[float, float], int]] = []

    def _fake_populate(figure, result, cluster_color_map, *, figsize, dpi):
        populate_calls.append(
            (figure, result, cluster_color_map, tuple(figsize), int(dpi))
        )
        return figure

    module._populate_embedded_clustermap_figure = _fake_populate
    result = ClusterResult(
        correlation_matrix=np.eye(2, dtype=np.float32),
        distance_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        linkage_matrix=np.array([[0.0, 1.0, 1.0, 2.0]], dtype=np.float64),
        neuron_ids=["n1", "n2"],
        reorder_indices=np.array([0, 1], dtype=np.intp),
        labels=np.array([1, 2], dtype=np.int32),
    )

    widget._draw_clustermap(result)

    assert populate_calls == [
        (
            widget._figure,
            result,
            None,
            (12.0, 8.0),
            100,
        )
    ]
    widget._canvas.draw.assert_called_once_with()
    assert widget._figure.set_size_inches_calls == 0


def test_build_clustermap_on_demand_draws_cached_result_and_logs(caplog):
    """Opt-in dendrogram rendering should draw only when the button handler runs."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._build_clustermap_btn = _DummyButton("Build Dendrogram")
    widget._clustermap_status_label = _DummyLabel()
    widget._update_button_states = MagicMock()
    widget._draw_clustermap = MagicMock()
    result = ClusterResult(
        correlation_matrix=np.eye(2, dtype=np.float32),
        distance_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        linkage_matrix=np.array([[0.0, 1.0, 1.0, 2.0]], dtype=np.float64),
        neuron_ids=["n1", "n2"],
        reorder_indices=np.array([0, 1], dtype=np.intp),
        labels=np.array([1, 2], dtype=np.int32),
    )
    widget._last_cluster_result = result

    with caplog.at_level(
        logging.DEBUG, logger="napari_neuron_navigator.widgets.analysis_tab"
    ):
        widget._build_clustermap_on_demand()

    messages = [record.getMessage() for record in caplog.records]
    assert any("_build_clustermap_on_demand start" in message for message in messages)
    assert any(
        "_build_clustermap_on_demand complete" in message for message in messages
    )
    widget._draw_clustermap.assert_called_once_with(result)
    widget._update_button_states.assert_called_once_with()
    assert widget._clustermap_status_label.text() == "Dendrogram ready for 2 neurons."


def test_on_correlation_finished_emits_debug_logs(caplog):
    """UI completion handler should defer dendrogram rendering until a button click."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._last_cluster_result = None
    widget._cluster_color_map = {"n1": [0.1, 0.2, 0.3, 1.0]}
    widget._actual_n_clusters = 2
    widget._n_clusters_spin = types.SimpleNamespace(value=lambda: 5)
    widget._progress_bar = _DummyProgressBar()
    widget._progress_label = _DummyLabel()
    widget._viewer = _DummyViewer()
    widget._slice_projector = None
    widget._clustermap_status_label = _DummyLabel()
    widget._build_clustermap_btn = _DummyButton("Build Dendrogram")
    widget._update_button_states = MagicMock()
    widget._update_cluster_filter_combo = lambda: None
    widget._build_cluster_color_map = MagicMock()
    widget._draw_clustermap = MagicMock()
    emitted: list[tuple[object, dict[str, list[float]]]] = []
    widget.cluster_colors_updated = types.SimpleNamespace(
        emit=lambda result, color_map: emitted.append((result, color_map))
    )
    result = ClusterResult(
        correlation_matrix=np.eye(2, dtype=np.float32),
        distance_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        linkage_matrix=np.array([[0.0, 1.0, 1.0, 2.0]], dtype=np.float64),
        neuron_ids=["n1", "n2"],
        reorder_indices=np.array([0, 1], dtype=np.intp),
        labels=np.array([1, 2], dtype=np.int32),
    )

    with caplog.at_level(
        logging.DEBUG, logger="napari_neuron_navigator.widgets.analysis_tab"
    ):
        widget._on_correlation_finished(result)

    messages = [record.getMessage() for record in caplog.records]
    assert any("_on_correlation_finished start" in message for message in messages)
    assert any("color map built" in message for message in messages)
    assert any(
        "clustermap render deferred until button click" in message
        for message in messages
    )
    widget._build_cluster_color_map.assert_called_once_with()
    widget._update_button_states.assert_called_once_with()
    widget._draw_clustermap.assert_not_called()
    assert emitted == [(result, {"n1": [0.1, 0.2, 0.3, 1.0]})]
    assert widget._clustermap_status_label.text() == (
        "Clustering complete. Click 'Build Dendrogram' to render the cluster map."
    )
