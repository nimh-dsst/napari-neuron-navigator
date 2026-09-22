"""Neuron selection table widget.

Provides an interactive table for viewing and controlling neurons:
- Toggle neuron visibility on/off
- Track whether a neuron is currently added to the scene
- Switch between focused column views
- View neuron name and cluster assignments
- Edit neuron colors via color picker
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from qtpy.QtCore import QItemSelection, QItemSelectionModel, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..cluster_assignments import ClusterAssignmentStore
from ..neuron_palette import neuron_palette
from ..neuron_table_ops import (
    GRAY_RGBA,
    ClusterFilterSelection,
    NeuronTableSummary,
    added_flags,
    cluster_filter_matches,
    cluster_ids_available,
    cluster_sort_value,
    has_unclustered_entries,
    recolor_cluster_distinct,
    summarize_neuron_table,
    visibility_for_selected_cluster,
)

if TYPE_CHECKING:
    from ..analysis.clustering import ClusterResult

logger = logging.getLogger(__name__)

# Column indices
COL_VISIBLE = 0
COL_ADDED = 1
COL_HEATMAP = 2
COL_NEURON_ID = 3
COL_LABEL = 4
COL_GROUP = 5
COL_TAGS = 6
COL_NOTES = 7
COL_CLUSTER = 8
COL_COLOR = 9
_BASE_HEADERS = [
    "Vis",
    "Added",
    "Heatmap",
    "Neuron Name",
    "Label",
    "Group",
    "Tags",
    "Notes",
]
_EDITABLE_METADATA_COLUMNS = {
    COL_LABEL: "label",
    COL_GROUP: "group",
    COL_TAGS: "tags",
    COL_NOTES: "notes",
}

_DEFAULT_TABLE_VIEW = "default"
_TABLE_VIEW_DEFINITIONS = {
    "default": (
        "Default",
        frozenset({COL_NEURON_ID}),
        True,
    ),
    "notes": (
        "Notes",
        frozenset({COL_NEURON_ID, COL_NOTES}),
        True,
    ),
    "visibility": (
        "Visibility",
        frozenset({COL_VISIBLE, COL_ADDED, COL_HEATMAP, COL_NEURON_ID}),
        False,
    ),
    "label": (
        "Label",
        frozenset({COL_NEURON_ID, COL_LABEL, COL_GROUP, COL_TAGS}),
        True,
    ),
    "full": (
        "Full",
        frozenset(range(COL_CLUSTER)),
        True,
    ),
}

# Keep a useful empty-state viewport, then grow with the result set until the
# table is tall enough for routine multi-neuron work.  Larger result sets keep
# using the table's own scrollbar so the controls below it remain reachable.
_MIN_PREFERRED_VISIBLE_ROWS = 4
_MAX_PREFERRED_VISIBLE_ROWS = 20
_FIXED_VISIBLE_ROWS = 8


def _preferred_visible_row_count(
    visible_row_count: int,
    *,
    adaptive: bool = True,
) -> int:
    """Return the bounded number of rows the table should make visible."""
    if not adaptive:
        return _FIXED_VISIBLE_ROWS
    return min(
        max(int(visible_row_count), _MIN_PREFERRED_VISIBLE_ROWS),
        _MAX_PREFERRED_VISIBLE_ROWS,
    )


class _NumericSortItem(QTableWidgetItem):
    """Table item that sorts by numeric key stored in ``Qt.UserRole``."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.data(Qt.UserRole)
        right = other.data(Qt.UserRole)
        if left is not None and right is not None:
            return float(left) < float(right)
        return super().__lt__(other)


@dataclass
class NeuronEntry:
    """Per-neuron state tracked by the neuron selection table."""

    file_id: object
    color: list[float] = field(default_factory=lambda: list(GRAY_RGBA))
    cluster_id: int | None = None
    visible: bool = True
    added_to_scene: bool = False
    heatmap_layer_names: tuple[str, ...] = ()
    label: str = ""
    group: str = ""
    tags: tuple[str, ...] = ()
    notes: str = ""

    def to_state(self) -> dict[str, object]:
        """Return a JSON-safe representation of this row."""
        return {
            "file_id": self.file_id,
            "color": list(self.color),
            "cluster_id": self.cluster_id,
            "visible": self.visible,
            "added_to_scene": self.added_to_scene,
            "heatmap_layer_names": list(self.heatmap_layer_names),
            "label": self.label,
            "group": self.group,
            "tags": list(self.tags),
            "notes": self.notes,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> NeuronEntry:
        """Create an entry from a previously exported table-state row."""
        file_id = state.get("file_id")
        tags = state.get("tags", ())
        if isinstance(tags, str):
            normalized_tags = tuple(
                part.strip() for part in tags.split(",") if part.strip()
            )
        else:
            try:
                normalized_tags = tuple(
                    str(part).strip() for part in tags if str(part).strip()
                )
            except TypeError:
                normalized_tags = ()
        cluster_value = state.get("cluster_id", state.get("cluster_assignment"))
        cluster_id: int | None
        if cluster_value in (None, ""):
            cluster_id = None
        else:
            try:
                cluster_id = int(cluster_value)
            except (TypeError, ValueError):
                cluster_id = None
        color = state.get("color", GRAY_RGBA)
        if isinstance(color, (str, bytes)):
            color_values = list(GRAY_RGBA)
        else:
            try:
                color_values = [float(value) for value in color]  # type: ignore[union-attr]
            except TypeError:
                color_values = list(GRAY_RGBA)
        while len(color_values) < 4:
            color_values.append(1.0)
        heatmap_names = state.get("heatmap_layer_names", ())
        if isinstance(heatmap_names, str):
            normalized_heatmaps = (heatmap_names,) if heatmap_names else ()
        else:
            try:
                normalized_heatmaps = tuple(str(name) for name in heatmap_names)
            except TypeError:
                normalized_heatmaps = ()
        return cls(
            file_id="" if file_id is None else file_id,
            color=color_values[:4],
            cluster_id=cluster_id,
            visible=bool(state.get("visible", True)),
            added_to_scene=bool(state.get("added_to_scene", False)),
            heatmap_layer_names=normalized_heatmaps,
            label=str(state.get("label", state.get("neuron_label", "")) or ""),
            group=str(state.get("group", state.get("neuron_group", "")) or ""),
            tags=normalized_tags,
            notes=str(state.get("notes", state.get("neuron_notes", "")) or ""),
        )


@dataclass(frozen=True)
class AppendSummary:
    """Outcome of appending unique neuron identities to the table."""

    added_file_ids: tuple[object, ...] = ()
    already_present_file_ids: tuple[object, ...] = ()

    @property
    def added_count(self) -> int:
        """Return the number of newly appended neurons."""
        return len(self.added_file_ids)

    @property
    def already_present_count(self) -> int:
        """Return the number of requested neurons already in the table."""
        return len(self.already_present_file_ids)

class NeuronTableWidget(QWidget):
    """Interactive table for neuron selection, color editing, and visibility.

    Signals
    -------
    colors_changed : dict
        Emitted when neuron colors change. Payload is ``{file_id: [r,g,b,a]}``.
    visibility_changed : dict
        Emitted when neuron visibility changes. Payload is ``{file_id: bool}``.
    state_changed
        Emitted when tracked table state changes in a way that affects summary UI.
    """

    colors_changed = Signal(dict)
    visibility_changed = Signal(dict)
    selection_changed = Signal(list)
    state_changed = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        assignment_store: ClusterAssignmentStore | None = None,
    ):
        super().__init__(parent)
        self._entries: dict[object, NeuronEntry] = {}
        # ``ClusterAssignmentStore`` is falsey while empty because it defines
        # ``__len__``.  Test explicitly so the initially empty shared store is
        # retained rather than silently replaced with a private one.
        self._assignment_store = (
            assignment_store
            if assignment_store is not None
            else ClusterAssignmentStore()
        )
        self._cluster_column_by_id: dict[str, int] = {}
        self._current_table_view = _DEFAULT_TABLE_VIEW
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Table view:"))
        self._table_view_combo = QComboBox()
        for view_key, (
            label,
            _columns,
            _show_clusters,
        ) in _TABLE_VIEW_DEFINITIONS.items():
            self._table_view_combo.addItem(label, view_key)
        self._table_view_combo.currentIndexChanged.connect(self._on_table_view_changed)
        view_row.addWidget(self._table_view_combo, 1)
        layout.addLayout(view_row)

        self._table = QTableWidget(0, COL_COLOR + 1)
        self._adaptive_height_enabled = True
        self._table_maximum_height = self._table.maximumHeight()
        self._configure_cluster_columns()
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self._configure_header_resize_modes()
        self._apply_table_view()

        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.setSortingEnabled(True)

        layout.addWidget(self._table)
        self._refresh_preferred_height(visible_row_count=0)

    def _refresh_preferred_height(
        self,
        *,
        visible_row_count: int | None = None,
    ) -> None:
        """Grow the table with its visible rows, up to the viewport row cap."""
        adaptive = bool(self.__dict__.get("_adaptive_height_enabled", True))
        if visible_row_count is None:
            visible_row_count = (
                sum(
                    not self._table.isRowHidden(row)
                    for row in range(self._table.rowCount())
                )
                if adaptive
                else 0
            )

        preferred_rows = _preferred_visible_row_count(
            visible_row_count,
            adaptive=adaptive,
        )
        horizontal_header = self._table.horizontalHeader()
        header_height = max(
            horizontal_header.height(),
            horizontal_header.sizeHint().height(),
        )
        row_height = self._table.verticalHeader().defaultSectionSize()
        scrollbar_height = self._table.horizontalScrollBar().sizeHint().height()
        frame_height = 2 * self._table.frameWidth()
        preferred_height = (
            header_height
            + preferred_rows * row_height
            + scrollbar_height
            + frame_height
        )

        if adaptive:
            # Reset the maximum left behind by ``setFixedHeight`` before
            # lowering or raising the adaptive minimum.
            self._table.setMaximumHeight(
                self.__dict__.get(
                    "_table_maximum_height",
                    self._table.maximumHeight(),
                )
            )
            self._table.setMinimumHeight(preferred_height)
        else:
            self._table.setFixedHeight(preferred_height)
        self._table.updateGeometry()
        self.updateGeometry()

    def set_adaptive_height_enabled(self, enabled: bool) -> None:
        """Switch between a fixed viewport and row-aware table height."""
        self._adaptive_height_enabled = bool(enabled)
        self._refresh_preferred_height()

    def _assignment_sets(self):
        """Return saved assignment sets, tolerating legacy test widgets."""
        store = self.__dict__.get("_assignment_store")
        return () if store is None else store.sets()

    def _active_assignment_id(self) -> str | None:
        store = self.__dict__.get("_assignment_store")
        return None if store is None else store.active_assignment_id

    def _color_column(self) -> int:
        """Return the current Color column index."""
        return COL_CLUSTER + max(len(self._assignment_sets()), 1)

    def _cluster_column(self, assignment_id: str | None = None) -> int:
        """Return the requested or active cluster column index."""
        columns = self.__dict__.get("_cluster_column_by_id", {})
        resolved_id = assignment_id or self._active_assignment_id()
        return columns.get(resolved_id, COL_CLUSTER)

    def _configure_cluster_columns(self) -> None:
        """Configure headers for all saved assignment sets."""
        assignment_sets = self._assignment_sets()
        self._cluster_column_by_id = {
            assignment.assignment_id: COL_CLUSTER + index
            for index, assignment in enumerate(assignment_sets)
        }
        active_id = self._active_assignment_id()
        cluster_headers = [
            (
                f"{assignment.name} (active)"
                if assignment.assignment_id == active_id
                else assignment.name
            )
            for assignment in assignment_sets
        ]
        if not cluster_headers:
            cluster_headers = ["Cluster"]
        headers = [*_BASE_HEADERS, *cluster_headers, "Color"]
        set_column_count = getattr(self._table, "setColumnCount", None)
        set_headers = getattr(self._table, "setHorizontalHeaderLabels", None)
        if not callable(set_column_count) or not callable(set_headers):
            return
        set_column_count(len(headers))
        set_headers(headers)
        self._apply_table_view()

    def _configure_header_resize_modes(self) -> None:
        """Apply resize policies after dynamic column changes."""
        header_getter = getattr(self._table, "horizontalHeader", None)
        if not callable(header_getter):
            return
        header = header_getter()
        header.setSectionResizeMode(COL_VISIBLE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_ADDED, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_HEATMAP, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_NEURON_ID, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_LABEL, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_GROUP, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_TAGS, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_NOTES, QHeaderView.Stretch)
        for column in range(COL_CLUSTER, self._color_column()):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self._color_column(), QHeaderView.ResizeToContents)

    def _on_table_view_changed(self, _index: int) -> None:
        """Apply the focused table view selected by the user."""
        view_key = self._table_view_combo.currentData()
        if view_key not in _TABLE_VIEW_DEFINITIONS:
            return
        self._current_table_view = str(view_key)
        self._apply_table_view()

    def _apply_table_view(self) -> None:
        """Show only the columns included in the current focused view."""
        set_column_hidden = getattr(self._table, "setColumnHidden", None)
        if not callable(set_column_hidden):
            return

        view_key = self.__dict__.get(
            "_current_table_view",
            _DEFAULT_TABLE_VIEW,
        )
        definition = _TABLE_VIEW_DEFINITIONS.get(
            view_key,
            _TABLE_VIEW_DEFINITIONS[_DEFAULT_TABLE_VIEW],
        )
        _label, base_columns, show_clusters = definition
        color_column = self._color_column()
        for column in range(self._table.columnCount()):
            is_cluster_column = COL_CLUSTER <= column < color_column
            visible = (
                column in base_columns
                or column == color_column
                or (show_clusters and is_cluster_column)
            )
            set_column_hidden(column, not visible)

    def populate(self, file_ids: list[object]) -> None:
        """Fill the table with neurons from a query result.

        Parameters
        ----------
        file_ids : list[object]
            Per-neuron ``file_id`` values to show as neuron names.
        """
        sorting_enabled = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)

        self._table.setRowCount(0)
        self._entries.clear()
        # A newly loaded Parquet may replace the assignment registry before
        # the table membership is populated.  Rebuild the dynamic columns here
        # so stale assignment columns from the previous dataset cannot linger.
        self._configure_cluster_columns()
        self._configure_header_resize_modes()

        n = len(file_ids)
        colors = neuron_palette(n)

        self._table.setRowCount(n)

        for row, file_id in enumerate(file_ids):
            color = colors[row] if row < len(colors) else list(GRAY_RGBA)
            entry = NeuronEntry(file_id=file_id, color=color)
            self._entries[file_id] = entry
            self._populate_row(row, entry)

        self._table.setSortingEnabled(sorting_enabled)
        self._refresh_preferred_height(visible_row_count=n)
        self.state_changed.emit()

    def _populate_row(self, row: int, entry: NeuronEntry) -> None:
        """Populate a single table row from a NeuronEntry."""
        # Visible checkbox
        cb = QCheckBox()
        cb.setChecked(entry.visible)
        cb.stateChanged.connect(partial(self._on_visibility_toggled, entry.file_id))
        cb_widget = QWidget()
        cb_layout = QHBoxLayout(cb_widget)
        cb_layout.addWidget(cb)
        cb_layout.setAlignment(Qt.AlignCenter)
        cb_layout.setContentsMargins(0, 0, 0, 0)
        self._table.setCellWidget(row, COL_VISIBLE, cb_widget)

        # Added state
        self._set_added_cell(row, entry.added_to_scene)

        # Heatmap state
        self._set_heatmap_cell(row, entry.heatmap_layer_names)

        # Neuron name (the unique file_id key)
        id_item = QTableWidgetItem(str(entry.file_id))
        id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
        id_item.setData(Qt.UserRole, entry.file_id)
        self._table.setItem(row, COL_NEURON_ID, id_item)

        self._set_text_cell(row, COL_LABEL, entry.label, editable=True)
        self._set_text_cell(row, COL_GROUP, entry.group, editable=True)
        self._set_text_cell(
            row, COL_TAGS, self._tags_display(entry.tags), editable=True
        )
        self._set_text_cell(row, COL_NOTES, entry.notes, editable=True)

        # Saved cluster assignments. Legacy widgets without a store retain the
        # original single Cluster column.
        assignment_sets = self._assignment_sets()
        if assignment_sets:
            for assignment in assignment_sets:
                self._set_cluster_cell(
                    row,
                    assignment.label_for(entry.file_id),
                    assignment_id=assignment.assignment_id,
                )
        else:
            self._set_cluster_cell(row, entry.cluster_id)

        # Color swatch button
        btn = QPushButton()
        btn.setFixedSize(24, 24)
        self._apply_color_style(btn, entry.color)
        btn.clicked.connect(partial(self._on_color_clicked, entry.file_id))
        self._table.setCellWidget(row, self._color_column(), btn)

    def _set_added_cell(self, row: int, added: bool) -> None:
        """Set the Added column cell text and sort key."""
        item = self._table.item(row, COL_ADDED)
        if item is None:
            item = QTableWidgetItem()
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, COL_ADDED, item)

        item.setText("Yes" if added else "No")
        item.setData(Qt.UserRole, 1 if added else 0)

    def _set_heatmap_cell(
        self,
        row: int,
        layer_names: Iterable[object],
    ) -> None:
        """Set the Heatmap column cell text and sort key."""
        item = self._table.item(row, COL_HEATMAP)
        if item is None:
            item = QTableWidgetItem()
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, COL_HEATMAP, item)

        display = ", ".join(str(name) for name in layer_names)
        item.setText(display)
        item.setData(Qt.UserRole, display.casefold())

    def _set_cluster_cell(
        self,
        row: int,
        cluster_id: int | None,
        *,
        assignment_id: str | None = None,
    ) -> None:
        """Set cluster cell text and numeric sort key."""
        column = self._cluster_column(assignment_id)
        item = self._table.item(row, column)
        if item is None or not isinstance(item, _NumericSortItem):
            item = _NumericSortItem()
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, column, item)

        item.setText("" if cluster_id is None else str(cluster_id))
        item.setData(Qt.UserRole, cluster_sort_value(cluster_id))

    def _set_text_cell(
        self,
        row: int,
        column: int,
        text: str,
        *,
        editable: bool,
    ) -> None:
        """Set a plain-text cell and its case-insensitive sort key."""
        item = self._table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            flags = item.flags()
            if not editable:
                flags &= ~Qt.ItemIsEditable
            item.setFlags(flags)
            self._table.setItem(row, column, item)
        item.setText(str(text))
        item.setData(Qt.UserRole, str(text).casefold())

    @staticmethod
    def _normalise_tags(value: object) -> tuple[str, ...]:
        """Return tags as a normalized tuple."""
        if value is None:
            return ()
        if isinstance(value, (str, bytes)):
            text = value.decode("utf-8") if isinstance(value, bytes) else value
            return tuple(part.strip() for part in text.split(",") if part.strip())
        try:
            return tuple(str(part).strip() for part in value if str(part).strip())
        except TypeError:
            text = str(value).strip()
            return (text,) if text else ()

    @staticmethod
    def _tags_display(tags: Iterable[object]) -> str:
        """Return tags for display in the editable table cell."""
        return ", ".join(str(tag) for tag in tags)

    def _apply_color_style(self, btn: QPushButton, color: list[float]) -> None:
        """Set the button background to the given RGBA color."""
        r, g, b, a = [int(c * 255) for c in color[:4]]
        btn.setStyleSheet(
            f"background-color: rgba({r},{g},{b},{a}); border: 1px solid gray;"
        )

    def _on_color_clicked(self, file_id: str) -> None:
        """Open a color dialog when a color swatch is clicked."""
        entry = self._entries.get(file_id)
        if entry is None:
            return

        current = QColor.fromRgbF(*entry.color[:4])
        new_color = QColorDialog.getColor(
            current, self, "Choose Neuron Color", QColorDialog.ShowAlphaChannel
        )
        if not new_color.isValid():
            return

        rgba = [
            new_color.redF(),
            new_color.greenF(),
            new_color.blueF(),
            new_color.alphaF(),
        ]
        entry.color = rgba
        self._update_color_swatch(file_id)
        self.colors_changed.emit({file_id: rgba})

    def _on_selection_changed(self) -> None:
        """Emit selection_changed when table row selection changes."""
        self.selection_changed.emit(self.get_selected_file_ids())

    def _on_visibility_toggled(self, file_id: str, state: int) -> None:
        """Handle a visibility checkbox state change."""
        entry = self._entries.get(file_id)
        if entry is None:
            return

        entry.visible = bool(state)
        self.visibility_changed.emit(
            {fid: e.visible for fid, e in self._entries.items()}
        )
        self.state_changed.emit()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Update entry metadata after an editable text cell changes."""
        column = item.column()
        field_name = _EDITABLE_METADATA_COLUMNS.get(column)
        if field_name is None:
            return

        file_id = self._file_id_from_row(item.row())
        if file_id is None:
            return
        entry = self._entries.get(file_id)
        if entry is None:
            return

        text = item.text().strip()
        if field_name == "tags":
            entry.tags = self._normalise_tags(text)
            display = self._tags_display(entry.tags)
            if item.text() != display:
                item.setText(display)
            item.setData(Qt.UserRole, display.casefold())
        else:
            setattr(entry, field_name, text)
            item.setData(Qt.UserRole, text.casefold())
        self.state_changed.emit()

    def _file_id_from_row(self, row: int) -> object | None:
        """Resolve a row to its file_id using item metadata."""
        item = self._table.item(row, COL_NEURON_ID)
        if item is None:
            return None

        file_id = item.data(Qt.UserRole)
        if file_id is not None:
            return file_id

        text = item.text()
        return text if text else None

    def _iter_rows_with_file_ids(self) -> list[tuple[int, object]]:
        """Return all table rows that map to a neuron file_id."""
        out: list[tuple[int, object]] = []
        for row in range(self._table.rowCount()):
            file_id = self._file_id_from_row(row)
            if file_id is not None:
                out.append((row, file_id))
        return out

    def _file_id_to_row_map(self) -> dict[object, int]:
        """Build a map from file_id to current table row."""
        return {file_id: row for row, file_id in self._iter_rows_with_file_ids()}

    def _file_id_to_row(self, file_id: object) -> int | None:
        """Get the current table row for a given file_id."""
        return self._file_id_to_row_map().get(file_id)

    def _update_visibility_checkbox(self, row: int, visible: bool) -> None:
        """Set row visibility checkbox state without emitting checkbox signals."""
        cb_widget = self._table.cellWidget(row, COL_VISIBLE)
        if cb_widget is None:
            return

        cb = cb_widget.findChild(QCheckBox)
        if cb is None:
            return

        was_blocked = cb.blockSignals(True)
        cb.setChecked(visible)
        cb.blockSignals(was_blocked)

    def _update_color_swatch(self, file_id: str) -> None:
        """Update the color swatch button for a given neuron."""
        entry = self._entries.get(file_id)
        if entry is None:
            return

        row = self._file_id_to_row(file_id)
        if row is None:
            return

        self._update_color_swatch_for_row(row, entry.color)

    def _update_color_swatch_for_row(self, row: int, color: list[float]) -> None:
        """Update the color swatch button for one known table row."""
        btn = self._table.cellWidget(row, self._color_column())
        if btn is not None:
            self._apply_color_style(btn, color)

    def _entries_in_table_order(self) -> list[NeuronEntry]:
        """Return tracked entries in the table's current row order."""
        ordered: list[NeuronEntry] = []
        for _row, file_id in self._iter_rows_with_file_ids():
            entry = self._entries.get(file_id)
            if entry is not None:
                ordered.append(entry)
        return ordered

    def _replace_entries(self, entries: list[NeuronEntry]) -> None:
        """Replace the table contents with the provided entries."""
        sorting_enabled = self._table.isSortingEnabled()
        signals_blocked = self._table.blockSignals(True)
        self._table.setSortingEnabled(False)
        try:
            self._table.clearSelection()
            self._table.clearContents()
            self._configure_cluster_columns()
            self._configure_header_resize_modes()
            self._table.setRowCount(0)
            self._entries = {entry.file_id: entry for entry in entries}
            self._table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                self._populate_row(row, entry)
        finally:
            self._table.setSortingEnabled(sorting_enabled)
            self._table.blockSignals(signals_blocked)

        self._refresh_preferred_height(visible_row_count=len(entries))
        self.selection_changed.emit([])
        self.state_changed.emit()

    def refresh_cluster_assignments(self, *, preserve_selection: bool = True) -> None:
        """Rebuild dynamic assignment columns from the shared store."""
        selected = self.get_selected_file_ids() if preserve_selection else []
        entries = self._entries_in_table_order()
        active_map = self.get_cluster_map()
        for entry in entries:
            entry.cluster_id = active_map.get(entry.file_id)
        self._replace_entries(entries)
        if selected:
            self.select_file_ids(selected)

    # --- Public API ---

    def get_selected_file_ids(self) -> list[object]:
        """Return the file_ids of the currently selected rows."""
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        selected: list[object] = []
        for row in rows:
            # A Qt ``Select All`` can leave indexes selected after a cluster
            # filter hides their rows.  "Selected Rows" clustering should use
            # only the rows the user can currently see and act on.
            if self._table.isRowHidden(row):
                continue
            file_id = self._file_id_from_row(row)
            if file_id is not None:
                selected.append(file_id)
        return selected

    def get_color(self, file_id: str) -> list[float]:
        """Return the RGBA color for a neuron."""
        entry = self._entries.get(file_id)
        return list(entry.color) if entry else list(GRAY_RGBA)

    def get_full_color_map(self) -> dict[str, list[float]]:
        """Return a mapping of all file_ids to their current RGBA colors."""
        return {fid: list(e.color) for fid, e in self._entries.items()}

    def get_cluster_map(self) -> dict[object, int | None]:
        """Return a mapping of all file_ids to their current cluster assignment."""
        store = self.__dict__.get("_assignment_store")
        if store is not None and store.active is not None:
            return store.active_map(self._entries.keys())
        return {fid: e.cluster_id for fid, e in self._entries.items()}

    def get_visibility_map(self) -> dict[str, bool]:
        """Return a mapping of all file_ids to their visibility state."""
        return {fid: e.visible for fid, e in self._entries.items()}

    def file_ids(self) -> list[object]:
        """Return the file_ids currently shown in the table row order."""
        return [file_id for _row, file_id in self._iter_rows_with_file_ids()]

    def export_state(self) -> dict[str, object]:
        """Return all table rows and per-neuron UI state as JSON-safe data."""
        cluster_map = self.get_cluster_map()
        for file_id, entry in self._entries.items():
            entry.cluster_id = cluster_map.get(file_id)
        state = {
            "version": 2 if "_assignment_store" in self.__dict__ else 1,
            "entries": [entry.to_state() for entry in self._entries_in_table_order()],
            "selected_file_ids": self.get_selected_file_ids(),
        }
        if "_assignment_store" in self.__dict__:
            state["cluster_assignments"] = self.assignment_state()
        return state

    def assignment_state(self) -> dict[str, object]:
        """Return the complete named-assignment registry."""
        store = self.__dict__.get("_assignment_store")
        if store is None:
            return ClusterAssignmentStore().to_state()
        return store.to_state()

    def import_state(self, state: Mapping[str, object]) -> None:
        """Replace table contents from a previously exported state payload."""
        raw_entries = state.get("entries", [])
        if isinstance(raw_entries, Mapping):
            iterable = raw_entries.values()
        else:
            try:
                iterable = list(raw_entries)  # type: ignore[arg-type]
            except TypeError:
                iterable = []
        entries = [
            NeuronEntry.from_state(entry)
            for entry in iterable
            if isinstance(entry, Mapping)
        ]
        self._load_assignment_state(state, entries)
        self._replace_entries(entries)
        selected = state.get("selected_file_ids", [])
        if isinstance(selected, (str, bytes)):
            selected_file_ids = [selected]
        else:
            try:
                selected_file_ids = list(selected)  # type: ignore[arg-type]
            except TypeError:
                selected_file_ids = []
        self.select_file_ids(selected_file_ids)

    def _load_assignment_state(
        self,
        state: Mapping[str, object],
        entries: Iterable[NeuronEntry],
    ) -> None:
        """Load v2 assignments or migrate legacy row cluster values."""
        if "_assignment_store" not in self.__dict__:
            return
        store = self.__dict__.get("_assignment_store")
        if store is None:
            store = ClusterAssignmentStore()
            self._assignment_store = store
        raw_assignments = state.get("cluster_assignments")
        if isinstance(raw_assignments, Mapping):
            store.load_state(raw_assignments)
            return
        if len(store) > 0:
            return
        legacy = {entry.file_id: entry.cluster_id for entry in entries}
        store.import_legacy(legacy)

    def apply_state(
        self,
        state: Mapping[str, object],
        *,
        preserve_membership: bool = True,
    ) -> None:
        """Apply saved per-neuron fields to matching current table rows."""
        raw_entries = state.get("entries", [])
        if isinstance(raw_entries, Mapping):
            iterable = raw_entries.values()
        else:
            try:
                iterable = list(raw_entries)  # type: ignore[arg-type]
            except TypeError:
                iterable = []

        parsed_entries: list[NeuronEntry] = []
        by_string_file_id = {}
        for raw_entry in iterable:
            if not isinstance(raw_entry, Mapping):
                continue
            entry = NeuronEntry.from_state(raw_entry)
            parsed_entries.append(entry)
            by_string_file_id[str(entry.file_id)] = entry

        self._load_assignment_state(state, parsed_entries)

        if not by_string_file_id:
            return

        row_map = self._file_id_to_row_map()
        sorting_enabled = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        try:
            for file_id, entry in list(self._entries.items()):
                saved = by_string_file_id.get(str(file_id))
                if saved is None:
                    continue
                entry.color = list(saved.color)
                store = self.__dict__.get("_assignment_store")
                entry.cluster_id = (
                    store.active.label_for(file_id)
                    if store is not None and store.active is not None
                    else saved.cluster_id
                )
                entry.visible = saved.visible
                entry.label = saved.label
                entry.group = saved.group
                entry.tags = saved.tags
                entry.notes = saved.notes
                if not preserve_membership:
                    entry.added_to_scene = saved.added_to_scene
                    entry.heatmap_layer_names = saved.heatmap_layer_names

                row = row_map.get(file_id)
                if row is None:
                    continue
                self._update_visibility_checkbox(row, entry.visible)
                self._update_color_swatch_for_row(row, entry.color)
                for assignment in self._assignment_sets():
                    self._set_cluster_cell(
                        row,
                        assignment.label_for(file_id),
                        assignment_id=assignment.assignment_id,
                    )
                if not self._assignment_sets():
                    self._set_cluster_cell(row, entry.cluster_id)
                self._set_text_cell(row, COL_LABEL, entry.label, editable=True)
                self._set_text_cell(row, COL_GROUP, entry.group, editable=True)
                self._set_text_cell(
                    row,
                    COL_TAGS,
                    self._tags_display(entry.tags),
                    editable=True,
                )
                self._set_text_cell(row, COL_NOTES, entry.notes, editable=True)
                if not preserve_membership:
                    self._set_added_cell(row, entry.added_to_scene)
                    self._set_heatmap_cell(row, entry.heatmap_layer_names)
        finally:
            self._table.setSortingEnabled(sorting_enabled)

        self.visibility_changed.emit(self.get_visibility_map())
        self.colors_changed.emit(self.get_full_color_map())
        if self._assignment_sets():
            self.refresh_cluster_assignments(preserve_selection=True)
        self.state_changed.emit()

    def summary(self) -> NeuronTableSummary:
        """Return summary counts for the current table contents."""
        return summarize_neuron_table(
            self.get_cluster_map(),
            {fid: entry.added_to_scene for fid, entry in self._entries.items()},
            {fid: entry.visible for fid, entry in self._entries.items()},
        )

    def clear(self) -> None:
        """Clear all table rows and tracked neuron state."""
        sorting_enabled = self._table.isSortingEnabled()
        signals_blocked = self._table.blockSignals(True)
        self._table.setSortingEnabled(False)
        try:
            self._table.clearContents()
            self._table.setRowCount(0)
            self._entries.clear()
        finally:
            self._table.setSortingEnabled(sorting_enabled)
            self._table.blockSignals(signals_blocked)

        self._refresh_preferred_height(visible_row_count=0)
        self.state_changed.emit()

    def retain_file_ids(
        self,
        file_ids: list[object] | tuple[object, ...],
    ) -> None:
        """Keep only the supplied neuron IDs while preserving survivor state."""
        keep_ids = set(file_ids)
        survivors = [
            entry
            for entry in self._entries_in_table_order()
            if entry.file_id in keep_ids
        ]
        self._replace_entries(survivors)

    def append_file_ids(
        self,
        file_ids: Iterable[object],
    ) -> AppendSummary:
        """Append new ``file_id`` rows while preserving all existing state."""
        requested = list(dict.fromkeys(file_ids))
        existing_entries = self._entries_in_table_order()
        existing_ids = {entry.file_id for entry in existing_entries}
        already_present = tuple(
            file_id for file_id in requested if file_id in existing_ids
        )
        added_ids = tuple(
            file_id for file_id in requested if file_id not in existing_ids
        )
        if not added_ids:
            return AppendSummary(
                added_file_ids=(),
                already_present_file_ids=already_present,
            )

        selected_ids = self.get_selected_file_ids()
        palette = neuron_palette(len(existing_entries) + len(added_ids))
        new_entries: list[NeuronEntry] = []
        for offset, file_id in enumerate(added_ids, start=len(existing_entries)):
            color = (
                palette[offset]
                if offset < len(palette)
                else list(GRAY_RGBA)
            )
            new_entries.append(NeuronEntry(file_id=file_id, color=list(color)))

        self._replace_entries([*existing_entries, *new_entries])
        if selected_ids:
            self.select_file_ids(selected_ids)
        return AppendSummary(
            added_file_ids=added_ids,
            already_present_file_ids=already_present,
        )

    def remove_file_ids(
        self,
        file_ids: list[object] | tuple[object, ...],
    ) -> None:
        """Remove the supplied neuron IDs while preserving survivor state."""
        remove_ids = set(file_ids)
        survivors = [
            entry
            for entry in self._entries_in_table_order()
            if entry.file_id not in remove_ids
        ]
        self._replace_entries(survivors)

    def set_added_file_ids(self, file_ids_in_scene: set[object] | list[object]) -> None:
        """Set whether each neuron is currently added to the scene."""
        flags = added_flags(self._entries.keys(), file_ids_in_scene)
        row_map = self._file_id_to_row_map()

        sorting_enabled = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        try:
            for file_id, added in flags.items():
                entry = self._entries.get(file_id)
                if entry is None:
                    continue

                entry.added_to_scene = added
                row = row_map.get(file_id)
                if row is not None:
                    self._set_added_cell(row, added)
        finally:
            self._table.setSortingEnabled(sorting_enabled)
        self.state_changed.emit()

    @staticmethod
    def _normalized_heatmap_layer_names(layer_names: object) -> tuple[str, ...]:
        """Return layer names as a deduplicated display tuple."""
        if layer_names is None:
            return ()
        if isinstance(layer_names, (str, bytes)):
            raw_names = [layer_names]
        else:
            try:
                raw_names = list(layer_names)
            except TypeError:
                raw_names = [layer_names]

        names: list[str] = []
        seen: set[str] = set()
        for name in raw_names:
            display = str(name)
            if not display or display in seen:
                continue
            names.append(display)
            seen.add(display)
        return tuple(names)

    def set_heatmap_layers_by_file_id(
        self,
        heatmap_layers_by_file_id: Mapping[object, object],
    ) -> None:
        """Set Data-tab heatmap layer names for each neuron."""
        exact = {
            file_id: self._normalized_heatmap_layer_names(layer_names)
            for file_id, layer_names in heatmap_layers_by_file_id.items()
        }
        by_string = {
            str(file_id): layer_names for file_id, layer_names in exact.items()
        }
        row_map = self._file_id_to_row_map()

        sorting_enabled = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        try:
            for file_id in self._entries:
                entry = self._entries.get(file_id)
                if entry is None:
                    continue

                layer_names = exact.get(file_id, by_string.get(str(file_id), ()))
                entry.heatmap_layer_names = layer_names
                row = row_map.get(file_id)
                if row is not None:
                    self._set_heatmap_cell(row, layer_names)
        finally:
            self._table.setSortingEnabled(sorting_enabled)
        self.state_changed.emit()

    def available_cluster_ids(self) -> list[int]:
        """Return sorted unique cluster IDs in the table."""
        return cluster_ids_available(self.get_cluster_map())

    def has_unclustered_entries(self) -> bool:
        """Return whether any table row has no cluster assignment."""
        return has_unclustered_entries(self.get_cluster_map())

    def apply_cluster_filter(
        self,
        selection: ClusterFilterSelection | int | None,
    ) -> None:
        """Hide rows outside the selected cluster groups."""
        self.apply_filters(selection)

    def apply_filters(
        self,
        selection: ClusterFilterSelection | int | None,
        heatmap_file_ids: set[object] | list[object] | tuple[object, ...] | None = None,
    ) -> None:
        """Hide rows outside the combined cluster and manual heatmap filters."""
        cluster_map = self.get_cluster_map()
        cluster_matches = cluster_filter_matches(cluster_map, selection)
        if heatmap_file_ids is None:
            heatmap_matches = {file_id: True for file_id in self._entries}
        else:
            heatmap_matches = added_flags(self._entries.keys(), heatmap_file_ids)

        visible_row_count = 0
        for row, file_id in self._iter_rows_with_file_ids():
            visible = cluster_matches.get(file_id, True) and heatmap_matches.get(
                file_id,
                False,
            )
            self._table.setRowHidden(row, not visible)
            visible_row_count += int(visible)
        self._refresh_preferred_height(visible_row_count=visible_row_count)

    def hide_all_not_in_cluster(
        self,
        selection: ClusterFilterSelection | int | None,
    ) -> None:
        """Set visibility off for neurons outside the selected cluster groups."""
        cluster_map = self.get_cluster_map()
        visibility = visibility_for_selected_cluster(cluster_map, selection)

        row_map = self._file_id_to_row_map()
        changed = False
        for file_id, visible in visibility.items():
            entry = self._entries.get(file_id)
            if entry is None:
                continue
            if entry.visible != visible:
                changed = True
            entry.visible = visible
            row = row_map.get(file_id)
            if row is not None:
                self._update_visibility_checkbox(row, visible)

        if changed:
            self.visibility_changed.emit(self.get_visibility_map())
            self.state_changed.emit()

    def set_all_visible(self) -> None:
        """Set visibility on for all neurons."""
        row_map = self._file_id_to_row_map()
        changed = False
        for file_id, entry in self._entries.items():
            if not entry.visible:
                changed = True
            entry.visible = True
            row = row_map.get(file_id)
            if row is not None:
                self._update_visibility_checkbox(row, True)

        if changed:
            self.visibility_changed.emit(self.get_visibility_map())
            self.state_changed.emit()

    def recolor_cluster_distinct(
        self,
        selection: ClusterFilterSelection | int | None,
        gray_others: bool = True,
    ) -> None:
        """Recolor selected groups distinctly; optionally gray non-selected."""
        cluster_map = self.get_cluster_map()
        updates = recolor_cluster_distinct(
            cluster_map, selection, gray_others=gray_others
        )
        if not updates:
            return

        row_map = self._file_id_to_row_map()
        changed: dict[str, list[float]] = {}
        for file_id, rgba in updates.items():
            entry = self._entries.get(file_id)
            if entry is None:
                continue
            if list(entry.color) == list(rgba):
                continue
            entry.color = list(rgba)
            row = row_map.get(file_id)
            if row is not None:
                self._update_color_swatch(file_id)
            changed[file_id] = list(rgba)

        if changed:
            self.colors_changed.emit(changed)

    def sort_by_cluster(self) -> None:
        """Sort rows by ascending numeric cluster assignment."""
        self._table.sortByColumn(self._cluster_column(), Qt.AscendingOrder)

    def update_cluster_assignments(self, result: ClusterResult) -> None:
        """Update the Cluster column from a ClusterResult.

        Parameters
        ----------
        result : ClusterResult
            Clustering result containing neuron_ids and labels.
        """
        store = self.__dict__.get("_assignment_store")
        if store is not None and store.active is not None:
            if store.active.runtime_result is not result:
                store.add_result(result, method_name="Cluster Assignment")
            self.refresh_cluster_assignments()
            return

        row_map = self._file_id_to_row_map()
        row_map_by_string = {str(file_id): row for file_id, row in row_map.items()}
        entries_by_string = {
            str(file_id): entry for file_id, entry in self._entries.items()
        }

        sorting_enabled = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        try:
            for neuron_id in getattr(result, "unassigned_neuron_ids", []) or []:
                entry = self._entries.get(neuron_id)
                if entry is None:
                    entry = entries_by_string.get(str(neuron_id))
                if entry is None:
                    continue
                entry.cluster_id = None
                row = row_map.get(neuron_id, row_map_by_string.get(str(neuron_id)))
                if row is not None:
                    self._set_cluster_cell(row, entry.cluster_id)

            for neuron_id, label in zip(result.neuron_ids, result.labels):
                entry = self._entries.get(neuron_id)
                if entry is None:
                    entry = entries_by_string.get(str(neuron_id))
                if entry is None:
                    continue
                entry.cluster_id = int(label)
                row = row_map.get(neuron_id, row_map_by_string.get(str(neuron_id)))
                if row is not None:
                    self._set_cluster_cell(row, entry.cluster_id)
        finally:
            self._table.setSortingEnabled(sorting_enabled)
        self.state_changed.emit()

    def select_file_ids(self, file_ids: list[str]) -> None:
        """Programmatically select table rows matching *file_ids*.

        Temporarily blocks the ``selection_changed`` signal to avoid
        feedback loops (e.g. soma click → table select → signal → …).
        """
        row_map = self._file_id_to_row_map()
        selected_rows: list[int] = []
        seen_file_ids: set[object] = set()
        for fid in file_ids:
            if fid in seen_file_ids:
                continue
            seen_file_ids.add(fid)

            row = row_map.get(fid)
            if row is not None and not self._table.isRowHidden(row):
                selected_rows.append(row)

        selection_model = self._table.selectionModel()
        selection = QItemSelection()
        model = self._table.model()
        last_column = max(self._table.columnCount() - 1, 0)
        for row in selected_rows:
            selection.select(
                model.index(row, 0),
                model.index(row, last_column),
            )

        previous_blocked = self._table.blockSignals(True)
        try:
            if selection_model is None or not selected_rows:
                self._table.clearSelection()
            else:
                selection_model.select(
                    selection,
                    QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
                )
        finally:
            self._table.blockSignals(previous_blocked)
        # Emit once after all rows are selected
        self.selection_changed.emit(self.get_selected_file_ids())

    def update_colors(
        self,
        color_map: dict[str, list[float]],
        *,
        emit_signal: bool = True,
    ) -> None:
        """Batch-update neuron colors from a color map.

        Emits a single ``colors_changed`` signal at the end when requested.

        Parameters
        ----------
        color_map : dict[str, list[float]]
            Mapping of file_id to RGBA color.
        """
        row_map = self._file_id_to_row_map()
        sorting_enabled = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        changed = {}
        try:
            for file_id, rgba in color_map.items():
                entry = self._entries.get(file_id)
                if entry is None:
                    continue
                new_color = list(rgba)
                if list(entry.color) == new_color:
                    continue
                entry.color = new_color
                row = row_map.get(file_id)
                if row is not None:
                    self._update_color_swatch_for_row(row, entry.color)
                changed[file_id] = list(entry.color)
        finally:
            self._table.setSortingEnabled(sorting_enabled)

        if emit_signal and changed:
            self.colors_changed.emit(changed)
