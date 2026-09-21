"""Reusable SWC node-type selector controls."""

from __future__ import annotations

from collections.abc import Iterable

from qtpy.QtCore import Signal

try:  # pragma: no cover - exercised by real Qt, not lightweight test doubles
    from qtpy.QtCore import QEvent, Qt
except ImportError:  # pragma: no cover - keeps import-only widget tests small
    QEvent = None

    class _QtFallback:
        Checked = 2
        Unchecked = 0
        CheckStateRole = 10
        ItemIsUserCheckable = 1
        ItemIsEnabled = 2

    Qt = _QtFallback()

from qtpy.QtWidgets import QComboBox

from ..swc import (
    STANDARD_NODE_TYPE_OPTIONS,
    NodeType,
    node_type_label,
    node_type_labels,
    normalize_node_types,
)

_ALL_NODE_TYPES = "all"
_QT_CHECKED = getattr(Qt, "Checked", 2)
_QT_UNCHECKED = getattr(Qt, "Unchecked", 0)
_QT_CHECK_STATE_ROLE = getattr(Qt, "CheckStateRole", 10)
_QT_ITEM_IS_USER_CHECKABLE = getattr(Qt, "ItemIsUserCheckable", 1)
_QT_ITEM_IS_ENABLED = getattr(Qt, "ItemIsEnabled", 2)


def node_type_options(values: Iterable[int]) -> tuple[tuple[int, str], ...]:
    """Return display options for the node types represented in a dataset."""
    options: list[tuple[int, str]] = []
    for value in sorted({int(item) for item in values}):
        label = node_type_label(value)
        if value == NodeType.AXON:
            label = "Axon-typed (type 2)"
        options.append((value, label))
    return tuple(options)


class NodeTypeSelectorComboBox(QComboBox):
    """Checkable combo box for SWC node-type filtering."""

    selection_changed = Signal(object)

    def __init__(
        self,
        parent=None,
        *,
        options: Iterable[tuple[int, str]] | None = None,
    ):
        super().__init__(parent)
        self._options = tuple(
            STANDARD_NODE_TYPE_OPTIONS if options is None else options
        )
        self._option_labels = {
            int(node_type): str(label) for node_type, label in self._options
        }
        self._updating = False
        self._fallback_data: list[object] = []
        self._fallback_checked: dict[int, object] = {}
        self._fallback_current_index = -1
        set_editable = getattr(self, "setEditable", None)
        if callable(set_editable):
            set_editable(True)
        self._set_line_text("All node types")

        view_getter = getattr(self, "view", None)
        view = view_getter() if callable(view_getter) else None
        viewport_getter = getattr(view, "viewport", None)
        viewport = viewport_getter() if callable(viewport_getter) else None
        if viewport is not None:
            viewport.installEventFilter(self)
        elif view is not None:
            pressed = getattr(view, "pressed", None)
            connect = getattr(pressed, "connect", None)
            if callable(connect):
                connect(self._on_item_pressed)

        activated = getattr(self, "activated", None)
        connect = getattr(activated, "connect", None)
        if callable(connect):
            connect(self._restore_display_text)

        self._populate()

    def set_options(self, options: Iterable[tuple[int, str]]) -> None:
        """Replace the available node types, retaining valid selections."""
        previous = self.selected_node_types()
        self._options = tuple((int(value), str(label)) for value, label in options)
        self._option_labels = {
            int(node_type): str(label) for node_type, label in self._options
        }
        self._populate()
        if previous is None:
            return
        available = {value for value, _label in self._options}
        retained = tuple(value for value in previous if value in available)
        self.set_selected_node_types(retained or None)

    def addItem(self, *args) -> None:
        """Add an item while tracking fallback data for test doubles."""
        if len(args) == 1:
            data = None
        elif len(args) >= 2:
            data = args[-1]
        else:
            data = None
        self._fallback_data.append(data)
        add_item = getattr(super(), "addItem", None)
        if callable(add_item):
            add_item(*args)
        self._fallback_current_index = max(self._fallback_current_index, 0)

    def clear(self) -> None:
        """Clear items while supporting minimal QComboBox doubles."""
        self._fallback_data.clear()
        self._fallback_checked.clear()
        self._fallback_current_index = -1
        clear = getattr(super(), "clear", None)
        if callable(clear):
            clear()

    def count(self) -> int:
        """Return item count, falling back for minimal QComboBox doubles."""
        count = getattr(super(), "count", None)
        if callable(count):
            return int(count())
        return len(self._fallback_data)

    def itemData(self, index: int, *args):
        """Return item data, including check-state fallback data."""
        item_data = getattr(super(), "itemData", None)
        if callable(item_data):
            try:
                return item_data(index, *args)
            except TypeError:
                pass
        if args and args[0] == _QT_CHECK_STATE_ROLE:
            return self._fallback_checked.get(index)
        return self._fallback_data[index]

    def setItemData(self, index: int, data, *args) -> None:
        """Set item data, including check-state fallback data."""
        set_item_data = getattr(super(), "setItemData", None)
        if callable(set_item_data):
            try:
                set_item_data(index, data, *args)
                return
            except TypeError:
                pass
        if args and args[0] == _QT_CHECK_STATE_ROLE:
            self._fallback_checked[index] = data
        elif 0 <= index < len(self._fallback_data):
            self._fallback_data[index] = data

    def setCurrentIndex(self, index: int) -> None:
        """Set current index with a fallback for minimal test doubles."""
        self._fallback_current_index = int(index)
        set_current_index = getattr(super(), "setCurrentIndex", None)
        if callable(set_current_index):
            set_current_index(index)

    def blockSignals(self, blocked: bool) -> bool:
        """Block signals if supported and return the previous state when known."""
        block_signals = getattr(super(), "blockSignals", None)
        if callable(block_signals):
            previous = block_signals(blocked)
            return bool(previous)
        return False

    def selected_node_types(self) -> tuple[int, ...] | None:
        """Return selected node types, or ``None`` for the unfiltered all mode."""
        if self._item_checked(0):
            return None

        selected: list[int] = []
        for index in range(1, self.count()):
            if self._item_checked(index):
                selected.append(int(self.itemData(index)))
        return normalize_node_types(selected)

    def set_selected_node_types(self, node_types: Iterable[int] | None) -> None:
        """Programmatically set the selected node types."""
        normalized = normalize_node_types(node_types)
        signals_blocked = self.blockSignals(True)
        self._updating = True
        try:
            if normalized is None:
                self._check_all_only()
            else:
                selected = set(normalized)
                self._set_item_checked(0, False)
                for index in range(1, self.count()):
                    self._set_item_checked(
                        index,
                        int(self.itemData(index)) in selected,
                    )
                if not self._has_specific_selection():
                    self._set_item_checked(0, True)
            self.setCurrentIndex(0)
            self._update_display_text()
        finally:
            self._updating = False
            self.blockSignals(signals_blocked)

    @staticmethod
    def selection_text(node_types: Iterable[int] | None) -> str:
        """Return compact display text for a node-type selection."""
        normalized = normalize_node_types(node_types)
        if normalized is None:
            return "All node types"

        labels = node_type_labels(normalized)
        if not labels:
            return "No node types"
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"{labels[0]} + {labels[1]}"
        return f"{len(labels)} node types"

    @staticmethod
    def query_text(node_types: Iterable[int] | None) -> str:
        """Return status-text wording for a node-type query."""
        normalized = normalize_node_types(node_types)
        if normalized is None:
            return "any node"

        labels = [label.lower() for label in node_type_labels(normalized)]
        if labels == ["soma"]:
            return "soma"
        if len(labels) == 1:
            return f"{labels[0]} nodes"
        return f"{' or '.join(labels)} nodes"

    @staticmethod
    def metadata_labels(node_types: Iterable[int] | None) -> list[str] | None:
        """Return metadata labels matching ``selected_node_types`` semantics."""
        normalized = normalize_node_types(node_types)
        if normalized is None:
            return None
        return node_type_labels(normalized)

    def _populate(self) -> None:
        self.clear()
        self._fallback_checked.clear()
        self._add_check_item("All node types", _ALL_NODE_TYPES, checked=True)
        for node_type, label in self._options:
            self._add_check_item(label, int(node_type), checked=False)
        self.setCurrentIndex(0)
        self._update_display_text()

    def _add_check_item(self, text: str, data: object, *, checked: bool) -> None:
        self.addItem(text, data)
        index = self.count() - 1
        self._set_item_checked(index, checked)
        model_getter = getattr(self, "model", None)
        model = model_getter() if callable(model_getter) else None
        item_getter = getattr(model, "item", None)
        item = item_getter(index, self.modelColumn()) if callable(item_getter) else None
        if item is not None:
            item.setFlags(
                item.flags() | _QT_ITEM_IS_USER_CHECKABLE | _QT_ITEM_IS_ENABLED
            )

    def _item_checked(self, index: int) -> bool:
        try:
            return self.itemData(index, _QT_CHECK_STATE_ROLE) == _QT_CHECKED
        except TypeError:
            return self._fallback_checked.get(index) == _QT_CHECKED

    def _set_item_checked(self, index: int, checked: bool) -> None:
        state = _QT_CHECKED if checked else _QT_UNCHECKED
        self._fallback_checked[index] = state
        try:
            self.setItemData(index, state, _QT_CHECK_STATE_ROLE)
        except TypeError:
            return

    def eventFilter(self, watched, event) -> bool:
        """Toggle popup checkboxes directly and keep the popup open."""
        view_getter = getattr(self, "view", None)
        view = view_getter() if callable(view_getter) else None
        viewport_getter = getattr(view, "viewport", None)
        viewport = viewport_getter() if callable(viewport_getter) else None
        if watched is viewport:
            row = self._event_row(event)
            event_type_getter = getattr(event, "type", None)
            event_type = event_type_getter() if callable(event_type_getter) else None
            if (
                QEvent is not None
                and row is not None
                and event_type
                in (
                    QEvent.MouseButtonPress,
                    QEvent.MouseButtonDblClick,
                )
            ):
                return True
            if (
                QEvent is not None
                and row is not None
                and event_type == QEvent.MouseButtonRelease
            ):
                self._toggle_item_at_row(row)
                return True
        parent_event_filter = getattr(super(), "eventFilter", None)
        if callable(parent_event_filter):
            return bool(parent_event_filter(watched, event))
        return False

    def _event_row(self, event) -> int | None:
        view_getter = getattr(self, "view", None)
        view = view_getter() if callable(view_getter) else None
        index_at = getattr(view, "indexAt", None)
        pos_getter = getattr(event, "pos", None)
        if not callable(index_at) or not callable(pos_getter):
            return None
        model_index = index_at(pos_getter())
        is_valid = getattr(model_index, "isValid", None)
        if callable(is_valid) and not is_valid():
            return None
        row_getter = getattr(model_index, "row", None)
        if not callable(row_getter):
            return None
        row = int(row_getter())
        if row < 0 or row >= self.count():
            return None
        return row

    def _on_item_pressed(self, model_index) -> None:
        row_getter = getattr(model_index, "row", None)
        if callable(row_getter):
            self._toggle_item_at_row(int(row_getter()))

    def _toggle_item_at_row(self, index: int) -> None:
        if self._updating or index < 0 or index >= self.count():
            return

        if index == 0:
            self._check_all_only()
        else:
            self._set_item_checked(index, not self._item_checked(index))
            self._set_item_checked(0, False)
            if not self._has_specific_selection():
                self._set_item_checked(0, True)

        self.setCurrentIndex(0)
        self._update_display_text()
        self.selection_changed.emit(self.selected_node_types())

    def _check_all_only(self) -> None:
        for index in range(self.count()):
            self._set_item_checked(index, index == 0)

    def _has_specific_selection(self) -> bool:
        return any(self._item_checked(index) for index in range(1, self.count()))

    def _restore_display_text(self, _index: int) -> None:
        if self._updating:
            return
        self.setCurrentIndex(0)
        self._update_display_text()

    def _set_line_text(self, text: str) -> None:
        line_edit_getter = getattr(self, "lineEdit", None)
        line_edit = line_edit_getter() if callable(line_edit_getter) else None
        if line_edit is not None:
            set_read_only = getattr(line_edit, "setReadOnly", None)
            if callable(set_read_only):
                set_read_only(True)
            line_edit.setText(text)
        else:
            set_edit_text = getattr(self, "setEditText", None)
            if callable(set_edit_text):
                set_edit_text(text)
        set_tool_tip = getattr(self, "setToolTip", None)
        if callable(set_tool_tip):
            set_tool_tip(text)

    def _update_display_text(self) -> None:
        selected = self.selected_node_types()
        if selected is None:
            text = "All node types"
        else:
            labels = [
                self._option_labels.get(node_type, f"Type {node_type}")
                for node_type in selected
            ]
            if not labels:
                text = "No node types"
            elif len(labels) == 1:
                text = labels[0]
            elif len(labels) == 2:
                text = f"{labels[0]} + {labels[1]}"
            else:
                text = f"{len(labels)} node types"
        self._set_line_text(text)
