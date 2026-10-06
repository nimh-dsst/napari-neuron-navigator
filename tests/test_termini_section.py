"""Tests for the Data tab Termini section."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


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

    def setEnabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def isEnabled(self) -> bool:
        return self._enabled

    def setVisible(self, visible: bool) -> None:
        self._visible = bool(visible)

    def isVisible(self) -> bool:
        return self._visible


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
        self._tooltip = ""
        self._style = ""

    def setText(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setWordWrap(self, _wrap: bool) -> None:
        return None

    def setToolTip(self, text: str) -> None:
        self._tooltip = str(text)

    def toolTip(self) -> str:
        return self._tooltip

    def setStyleSheet(self, style: str) -> None:
        self._style = str(style)

    def styleSheet(self) -> str:
        return self._style


class _DummyButton(_DummyWidget):
    """Small QPushButton stand-in."""

    def __init__(self, text: str = "", *_args, **_kwargs) -> None:
        super().__init__()
        self._text = text
        self.clicked = _BoundSignal()

    def text(self) -> str:
        return self._text


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

    def addItem(self, *args) -> None:
        text = str(args[0])
        data = args[1] if len(args) > 1 else None
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

    def setDecimals(self, *_args) -> None:
        return None


class _DummyProgressBar(_DummyWidget):
    """Small progress bar stand-in."""

    def setRange(self, *_args) -> None:
        return None

    def setValue(self, *_args) -> None:
        return None


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


class _DummyPointsLayer:
    """Minimal points layer stand-in for terminus tests."""

    def __init__(self, data, **kwargs) -> None:
        self.data = np.asarray(data)
        self.name = str(kwargs["name"])
        self.metadata = dict(kwargs.get("metadata", {}))
        self.size = kwargs.get("size")
        self.face_color = kwargs.get("face_color")
        self.scale = kwargs.get("scale")


class _DummyViewer:
    """Very small viewer stand-in."""

    def __init__(self) -> None:
        self.layers: list[_DummyPointsLayer] = []
        self.last_points_kwargs: dict[str, object] | None = None

    def add_points(self, data, **kwargs):
        layer = _DummyPointsLayer(data, **kwargs)
        self.layers.append(layer)
        self.last_points_kwargs = dict(kwargs)
        return layer


class _FakeNeuronTable:
    """Neuron-table stand-in exposing only what the section consumes."""

    def __init__(self, file_ids, hidden=()) -> None:
        self._file_ids = list(file_ids)
        self._hidden = {str(value) for value in hidden}
        self._selected: list[object] = []
        self.colors: dict[object, list[float]] = {}

    def file_ids(self) -> list[object]:
        return list(self._file_ids)

    def get_selected_file_ids(self) -> list[object]:
        return list(self._selected)

    def select_file_ids(self, file_ids) -> None:
        # Mirrors the real widget: rows hidden by a filter cannot be selected.
        self._selected = [
            file_id for file_id in file_ids if str(file_id) not in self._hidden
        ]

    def get_full_color_map(self) -> dict[object, list[float]]:
        return dict(self.colors)


class _FakeCoverage:
    """Stand-in for TerminusCoverage carrying a prebuilt summary."""

    def __init__(self, summary, file_ids_without=(), truncated=False) -> None:
        self._summary = summary
        self.file_ids_without = list(file_ids_without)
        self.file_ids_without_truncated = truncated

    def summary(self) -> str:
        return self._summary


def _import_termini_section_module():
    """Import ``termini_section.py`` with stubbed UI dependencies."""
    qtcore_module = types.ModuleType("qtpy.QtCore")
    qtcore_module.QThread = _DummyThread
    qtcore_module.Signal = _Signal

    qtwidgets_module = types.ModuleType("qtpy.QtWidgets")
    for name, value in {
        "QComboBox": _DummyCombo,
        "QDoubleSpinBox": _DummySpinBox,
        "QHBoxLayout": _DummyLayout,
        "QLabel": _DummyLabel,
        "QProgressBar": _DummyProgressBar,
        "QPushButton": _DummyButton,
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

    replacements = {
        "qtpy.QtCore": qtcore_module,
        "qtpy.QtWidgets": qtwidgets_module,
        "napari_neuron_navigator": napari_package,
        "napari_neuron_navigator.widgets": widgets_package,
        "napari_neuron_navigator.widgets.collapsible_section": collapsible_module,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    module_name = "napari_neuron_navigator.widgets.termini_section"
    previous_module = sys.modules.get(module_name)
    selector_module_name = "napari_neuron_navigator.widgets.node_type_selector"
    previous[selector_module_name] = sys.modules.get(selector_module_name)

    try:
        sys.modules.update(replacements)
        # Bind the shared selector to this test's QComboBox stub, regardless of
        # whether earlier tests imported it with real Qt or different stubs.
        sys.modules.pop(selector_module_name, None)
        sys.modules.pop(module_name, None)

        module_path = widgets_root / "termini_section.py"
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


def _termini_widget(table: _FakeNeuronTable | None = None):
    """Return a Termini section wired up enough to run terminus detection."""
    module = _import_termini_section_module()
    widget = module.TerminiSectionWidget(_DummyViewer())
    widget._db = object()
    widget._parquet_path = "neurons.parquet"
    if table is not None:
        widget.set_current_table_file_ids_provider(table.file_ids)
        widget.set_selected_table_file_ids_provider(table.get_selected_file_ids)
        widget.set_table_color_map_provider(table.get_full_color_map)
        widget.set_select_table_file_ids_callback(table.select_file_ids)
    return widget


def _install_fake_terminus_worker(monkeypatch, created):
    class _FakeTerminusWorker:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            created.append(self)

    workers_module = types.ModuleType("napari_neuron_navigator.workers")
    workers_module.TerminusWorker = _FakeTerminusWorker
    monkeypatch.setitem(sys.modules, "napari_neuron_navigator.workers", workers_module)


def _frame(file_ids, node_ids=None):
    """Build a minimal terminus result frame."""
    count = len(file_ids)
    node_ids = list(node_ids) if node_ids is not None else list(range(1, count + 1))
    return pd.DataFrame(
        {
            "file_id": list(file_ids),
            "node_id": node_ids,
            "type": [2] * count,
            "x": [float(i) for i in range(count)],
            "y": [float(i) for i in range(count)],
            "z": [float(i) for i in range(count)],
        }
    )


# --- Section shape ----------------------------------------------------------


@pytest.mark.parametrize("selector_cached", [False, True])
def test_termini_import_isolates_node_type_selector(monkeypatch, selector_cached):
    """Termini tests must neither reuse nor leak another test's Qt bindings."""
    module_name = "napari_neuron_navigator.widgets.node_type_selector"
    cached_module = types.ModuleType(module_name)
    cached_module.NodeTypeSelectorComboBox = MagicMock(
        side_effect=AssertionError("Reused a selector from a different Qt context")
    )
    if selector_cached:
        monkeypatch.setitem(sys.modules, module_name, cached_module)
    else:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    module = _import_termini_section_module()
    widget = module.TerminiSectionWidget(_DummyViewer())

    assert isinstance(widget._termini_node_type_combo, _DummyCombo)
    assert widget._termini_node_type_combo.selected_node_types() == (2,)
    if selector_cached:
        assert sys.modules[module_name] is cached_module
        cached_module.NodeTypeSelectorComboBox.assert_not_called()
    else:
        assert module_name not in sys.modules


def test_section_is_titled_termini_and_starts_collapsed() -> None:
    """The Data tab section is named 'Termini' and stays out of the way."""
    _DummyCollapsibleSection.instances = []
    _termini_widget()

    section = _DummyCollapsibleSection.instances[-1]
    assert section.title == "Termini"
    assert section.expanded is False


def test_termini_section_defaults_to_axon_node_type() -> None:
    """Axon is the default node type; see the caution test on what that means."""
    widget = _termini_widget()
    assert widget._selected_terminus_node_types() == (2,)


def test_section_warns_that_axon_typed_is_not_verified_axon() -> None:
    """Some neurons type dendritic projections as axon, so the UI must say so.

    Without this the points read as 'these are axon termini', which the source
    annotations do not support. Guards against the caution being dropped.
    """
    module = _import_termini_section_module()
    widget = module.TerminiSectionWidget(_DummyViewer())

    caution = widget._termini_caution_label.text()
    assert "axon-typed" in caution
    assert "dendrit" in caution
    assert "upper bound" in caution
    # Styled so it does not read as more body copy.
    assert widget._termini_caution_label.styleSheet()
    # The full explanation, including that this is a source-data defect.
    detail = widget._termini_caution_label.toolTip()
    assert "not in the detection" in detail
    assert "has not been measured" in detail


def test_termini_scope_defaults_to_current_table() -> None:
    """Pruning the table is the point, so the table is the default scope."""
    widget = _termini_widget()
    assert widget._selected_terminus_scope() == "current"
    assert widget._termini_scope_combo.currentText() == "Current Table"


def test_find_termini_button_enabled_without_an_atlas() -> None:
    """Terminus detection is pure topology, so it must not require an atlas."""
    widget = _termini_widget()
    widget._atlas = None
    widget._update_button_states()
    assert widget._find_termini_btn.isEnabled() is True


# --- Running detection ------------------------------------------------------


def test_run_terminus_detection_whole_parquet_passes_no_file_ids(monkeypatch) -> None:
    widget = _termini_widget()
    widget._start_background_worker = MagicMock()
    created: list = []
    _install_fake_terminus_worker(monkeypatch, created)

    widget._termini_scope_combo.setCurrentText("Whole Parquet")
    widget._run_terminus_detection()

    assert len(created) == 1
    assert created[0].kwargs == {
        "parquet_path": "neurons.parquet",
        "file_ids": None,
        "node_types": [2],
    }
    widget._start_background_worker.assert_called_once()


def test_run_terminus_detection_current_table_passes_table_file_ids(
    monkeypatch,
) -> None:
    table = _FakeNeuronTable(["a.swc", "b.swc"])
    widget = _termini_widget(table)
    widget._start_background_worker = MagicMock()
    created: list = []
    _install_fake_terminus_worker(monkeypatch, created)

    widget._run_terminus_detection()

    assert created[0].kwargs["file_ids"] == ["a.swc", "b.swc"]


def test_run_terminus_detection_selected_rows_passes_file_ids(monkeypatch) -> None:
    widget = _termini_widget()
    widget._start_background_worker = MagicMock()
    widget.set_selected_table_file_ids_provider(lambda: ["b.swc", "a.swc", "b.swc"])
    created: list = []
    _install_fake_terminus_worker(monkeypatch, created)

    widget._termini_scope_combo.setCurrentText("Selected Rows")
    widget._run_terminus_detection()

    assert created[0].kwargs["file_ids"] == ["b.swc", "a.swc"]


def test_run_terminus_detection_selected_rows_without_selection_reports(
    monkeypatch,
) -> None:
    widget = _termini_widget()
    widget._start_background_worker = MagicMock()
    widget.set_selected_table_file_ids_provider(lambda: [])
    created: list = []
    _install_fake_terminus_worker(monkeypatch, created)

    widget._termini_scope_combo.setCurrentText("Selected Rows")
    widget._run_terminus_detection()

    assert created == []
    assert "No table rows are selected" in widget._termini_status_label.text()


def test_run_terminus_detection_current_table_without_rows_reports(monkeypatch) -> None:
    widget = _termini_widget(_FakeNeuronTable([]))
    widget._start_background_worker = MagicMock()
    created: list = []
    _install_fake_terminus_worker(monkeypatch, created)

    widget._run_terminus_detection()

    assert created == []
    assert "Current table is empty" in widget._termini_status_label.text()


def test_clearing_node_types_falls_back_to_all_types(monkeypatch) -> None:
    """The selector treats an empty selection as 'all', so detection proceeds."""
    widget = _termini_widget(_FakeNeuronTable(["a.swc"]))
    widget._start_background_worker = MagicMock()
    widget._termini_node_type_combo.set_selected_node_types(())
    created: list = []
    _install_fake_terminus_worker(monkeypatch, created)

    widget._run_terminus_detection()

    assert widget._selected_terminus_node_types() is None
    assert created[0].kwargs["node_types"] is None


def test_run_terminus_detection_rejects_an_empty_node_type_selection(
    monkeypatch,
) -> None:
    """Guard for callers that do supply an empty selection."""
    widget = _termini_widget(_FakeNeuronTable(["a.swc"]))
    widget._start_background_worker = MagicMock()
    widget._selected_terminus_node_types = lambda: ()
    created: list = []
    _install_fake_terminus_worker(monkeypatch, created)

    widget._run_terminus_detection()

    assert created == []
    assert "Select at least one node type" in widget._termini_status_label.text()


# --- Results ----------------------------------------------------------------


def test_on_termini_finished_adds_points_layer_with_node_metadata() -> None:
    widget = _termini_widget()
    frame = _frame(["a.swc", "b.swc"], node_ids=[3, 7])
    coverage = _FakeCoverage("2 termini (Axon) in 2 of 2 neurons")

    widget._on_termini_finished(frame, coverage)

    layers = [layer for layer in widget._viewer.layers if "Termini" in layer.name]
    assert len(layers) == 1
    layer = layers[0]
    assert layer.data.tolist() == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert layer.metadata["file_ids_per_point"] == ["a.swc", "b.swc"]
    assert layer.metadata["node_ids"] == [3, 7]
    assert widget._termini_coverage_label.text() == (
        "2 termini (Axon) in 2 of 2 neurons"
    )


def test_on_termini_finished_always_reports_skipped_neurons() -> None:
    """The skipped-neuron count must never be silent."""
    widget = _termini_widget()
    coverage = _FakeCoverage(
        "1 termini (Axon) in 1 of 2 neurons — 1 neurons skipped",
        file_ids_without=["u.swc"],
    )

    widget._on_termini_finished(_frame(["a.swc"]), coverage)

    assert "1 neurons skipped" in widget._termini_coverage_label.text()
    assert widget._skipped_terminus_file_ids == ["u.swc"]
    assert widget._copy_skipped_termini_btn.isEnabled() is True


def test_on_termini_finished_reports_coverage_even_with_no_termini() -> None:
    """An empty result still has to explain why nothing was found."""
    widget = _termini_widget()
    coverage = _FakeCoverage(
        "0 termini (Axon) in 0 of 3 neurons — 3 neurons skipped",
        file_ids_without=["u1.swc", "u2.swc", "u3.swc"],
    )

    widget._on_termini_finished(pd.DataFrame(), coverage)

    assert "3 neurons skipped" in widget._termini_coverage_label.text()
    assert widget._copy_skipped_termini_btn.isEnabled() is True
    assert not [layer for layer in widget._viewer.layers if "Termini" in layer.name]


def test_on_termini_finished_flags_truncated_skipped_list() -> None:
    """A capped skipped list must say so rather than look complete."""
    widget = _termini_widget()
    coverage = _FakeCoverage(
        "0 termini (Axon) in 0 of 300 neurons — 300 neurons skipped",
        file_ids_without=[f"u{i}.swc" for i in range(200)],
        truncated=True,
    )

    widget._on_termini_finished(pd.DataFrame(), coverage)

    assert "Only the first 200 skipped neuron IDs are listed." in (
        widget._termini_coverage_label.text()
    )


def test_on_termini_finished_replaces_a_previous_layer() -> None:
    widget = _termini_widget()
    frame = _frame(["a.swc"])
    coverage = _FakeCoverage("1 termini")

    widget._on_termini_finished(frame, coverage)
    widget._on_termini_finished(frame, coverage)

    layers = [layer for layer in widget._viewer.layers if "Termini" in layer.name]
    assert len(layers) == 1


def test_on_termini_finished_colors_points_from_the_table() -> None:
    """Termini stay color-matched to their rows after clustering recolors them."""
    table = _FakeNeuronTable(["a.swc", "b.swc"])
    table.colors = {"a.swc": [1.0, 0.0, 0.0, 1.0], "b.swc": [0.0, 1.0, 0.0, 1.0]}
    widget = _termini_widget(table)

    widget._on_termini_finished(_frame(["a.swc", "b.swc"]), _FakeCoverage("2 termini"))

    layer = widget._viewer.layers[-1]
    assert layer.face_color.tolist() == [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ]


# --- Group selection --------------------------------------------------------


def _run_and_finish(widget, table_frame, coverage=None, scope_text=None):
    """Complete a detection run so the group selection has results."""
    if scope_text is not None:
        widget._termini_scope_combo.setCurrentText(scope_text)
    proceed, file_ids = widget._resolve_terminus_file_ids()
    assert proceed
    widget._termini_analyzed_file_ids = (
        None if file_ids is None else {str(value) for value in file_ids}
    )
    widget._on_termini_finished(table_frame, coverage or _FakeCoverage("run"))


def test_select_button_is_disabled_until_a_run_completes() -> None:
    widget = _termini_widget(_FakeNeuronTable(["a.swc"]))
    assert widget._select_termini_group_btn.isEnabled() is False

    _run_and_finish(widget, _frame(["a.swc"]))

    assert widget._select_termini_group_btn.isEnabled() is True


def test_select_group_with_termini_selects_matching_rows() -> None:
    table = _FakeNeuronTable(["a.swc", "b.swc", "c.swc"])
    widget = _termini_widget(table)
    _run_and_finish(widget, _frame(["a.swc", "c.swc"]))

    widget._termini_group_combo.setCurrentText("Neurons with termini")
    widget._select_terminus_group()

    assert table.get_selected_file_ids() == ["a.swc", "c.swc"]
    assert "Selected 2 of 3 table neurons with termini." in (
        widget._termini_status_label.text()
    )


def test_select_group_lacking_termini_selects_the_complement() -> None:
    table = _FakeNeuronTable(["a.swc", "b.swc", "c.swc"])
    widget = _termini_widget(table)
    _run_and_finish(widget, _frame(["a.swc", "c.swc"]))

    widget._termini_group_combo.setCurrentText("Neurons lacking termini")
    widget._select_terminus_group()

    assert table.get_selected_file_ids() == ["b.swc"]
    assert "lacking termini" in widget._termini_status_label.text()


def test_empty_result_puts_every_analyzed_neuron_in_the_lacking_group() -> None:
    """0 termini is a completed run, not a no-op: all rows are removable."""
    table = _FakeNeuronTable(["a.swc", "b.swc"])
    widget = _termini_widget(table)
    _run_and_finish(widget, pd.DataFrame())

    widget._termini_group_combo.setCurrentText("Neurons lacking termini")
    widget._select_terminus_group()

    assert table.get_selected_file_ids() == ["a.swc", "b.swc"]


def test_rows_outside_the_analyzed_scope_are_excluded_and_reported() -> None:
    """Rows the run never covered have unknown status, so never select them."""
    table = _FakeNeuronTable(["a.swc", "b.swc"])
    widget = _termini_widget(table)
    widget.set_selected_table_file_ids_provider(lambda: ["a.swc"])
    _run_and_finish(widget, _frame(["a.swc"]), scope_text="Selected Rows")

    # Restore the table-backed provider so the read-back counts real selections.
    widget.set_selected_table_file_ids_provider(table.get_selected_file_ids)
    widget._termini_group_combo.setCurrentText("Neurons lacking termini")
    widget._select_terminus_group()

    assert table.get_selected_file_ids() == []
    assert "outside the analyzed scope" in widget._termini_status_label.text()


def test_whole_parquet_scope_never_reports_out_of_scope_rows() -> None:
    table = _FakeNeuronTable(["a.swc", "b.swc"])
    widget = _termini_widget(table)
    _run_and_finish(widget, _frame(["a.swc"]), scope_text="Whole Parquet")

    widget._termini_group_combo.setCurrentText("Neurons lacking termini")
    widget._select_terminus_group()

    assert table.get_selected_file_ids() == ["b.swc"]
    assert "outside the analyzed scope" not in widget._termini_status_label.text()


def test_select_group_leaves_the_selection_alone_when_nothing_matches() -> None:
    table = _FakeNeuronTable(["a.swc"])
    table.select_file_ids(["a.swc"])
    widget = _termini_widget(table)
    _run_and_finish(widget, _frame(["a.swc"]))

    widget._termini_group_combo.setCurrentText("Neurons lacking termini")
    widget._select_terminus_group()

    assert table.get_selected_file_ids() == ["a.swc"]
    assert "No table neurons are lacking termini" in (
        widget._termini_status_label.text()
    )


def test_select_group_reports_rows_hidden_by_a_filter() -> None:
    """A cluster filter can hide matching rows; the shortfall must be visible."""
    table = _FakeNeuronTable(["a.swc", "b.swc"], hidden=["b.swc"])
    widget = _termini_widget(table)
    _run_and_finish(widget, pd.DataFrame())

    widget._termini_group_combo.setCurrentText("Neurons lacking termini")
    widget._select_terminus_group()

    assert table.get_selected_file_ids() == ["a.swc"]
    assert "hidden by a filter" in widget._termini_status_label.text()


def test_select_group_passes_the_tables_own_file_id_objects() -> None:
    """Non-string file_ids must still match, so pass the raw table objects."""
    table = _FakeNeuronTable([101, 102])
    widget = _termini_widget(table)
    _run_and_finish(widget, _frame(["101"]))

    widget._termini_group_combo.setCurrentText("Neurons with termini")
    widget._select_terminus_group()

    assert table.get_selected_file_ids() == [101]


def test_select_group_before_a_run_reports_instead_of_selecting() -> None:
    table = _FakeNeuronTable(["a.swc"])
    widget = _termini_widget(table)

    widget._select_terminus_group()

    assert table.get_selected_file_ids() == []
    assert "Run Find Termini" in widget._termini_status_label.text()


def test_select_group_with_an_empty_table_reports() -> None:
    table = _FakeNeuronTable([])
    widget = _termini_widget(table)
    widget._termini_run_complete = True

    widget._select_terminus_group()

    assert "table is empty" in widget._termini_status_label.text()


def test_loading_a_new_parquet_clears_stale_results() -> None:
    """A new dataset must not leave the previous run's groups selectable."""
    widget = _termini_widget(_FakeNeuronTable(["a.swc"]))
    _run_and_finish(widget, _frame(["a.swc"]))

    widget.set_database(types.SimpleNamespace(parquet_path="other.parquet"))

    assert widget._termini_run_complete is False
    assert widget._termini_file_ids_with == set()
    assert widget._select_termini_group_btn.isEnabled() is False
