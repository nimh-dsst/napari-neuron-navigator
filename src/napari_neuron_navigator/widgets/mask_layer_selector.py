"""Widget for selecting generated or imported mask layers with checkboxes."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MaskLayerSelectorWidget(QWidget):
    """Widget for selecting registered mask layers for querying."""

    selection_changed = Signal(list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._items_by_name: dict[str, QTreeWidgetItem] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        search_layout = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search mask layers...")
        self._search_input.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self._search_input)

        self._clear_search_btn = QPushButton("Clear")
        self._clear_search_btn.clicked.connect(self._clear_search)
        search_layout.addWidget(self._clear_search_btn)
        layout.addLayout(search_layout)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Mask Layer", "Source"])
        self._tree.setColumnWidth(0, 220)
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree)

        info_layout = QHBoxLayout()
        self._selection_label = QLabel("Selected: 0 mask layers")
        info_layout.addWidget(self._selection_label)

        self._clear_btn = QPushButton("Clear Selection")
        self._clear_btn.clicked.connect(self._clear_selection)
        info_layout.addWidget(self._clear_btn)
        layout.addLayout(info_layout)

    def set_mask_layers(self, masks: list[dict[str, object]]) -> None:
        """Populate the selector with mask layer metadata."""
        checked_names = set(self.get_selected_layer_names())
        self._tree.blockSignals(True)
        self._tree.clear()
        self._items_by_name.clear()

        for mask in masks:
            name = str(mask.get("name", ""))
            sources = mask.get("sources", [])
            if isinstance(sources, (list, tuple)):
                source_text = ", ".join(str(value) for value in sources)
            else:
                source_text = str(sources)

            item = QTreeWidgetItem(self._tree, [name, source_text])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if name in checked_names else Qt.Unchecked)
            item.setData(0, Qt.UserRole, name)
            self._items_by_name[name] = item

        self._tree.blockSignals(False)
        self._update_selection_label()
        self._emit_selection_changed()

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle checkbox changes."""
        if column != 0:
            return
        self._update_selection_label()
        self._emit_selection_changed()

    def _on_search_changed(self, text: str) -> None:
        """Filter rows based on search text."""
        pattern = text.lower().strip()
        for item in self._items_by_name.values():
            name = item.text(0).lower()
            sources = item.text(1).lower()
            visible = not pattern or pattern in name or pattern in sources
            item.setHidden(not visible)

    def _clear_search(self) -> None:
        """Clear the search filter."""
        self._search_input.clear()
        for item in self._items_by_name.values():
            item.setHidden(False)

    def _clear_selection(self) -> None:
        """Clear all selected mask layers."""
        self._tree.blockSignals(True)
        for item in self._items_by_name.values():
            item.setCheckState(0, Qt.Unchecked)
        self._tree.blockSignals(False)
        self._update_selection_label()
        self._emit_selection_changed()

    def _update_selection_label(self) -> None:
        """Update the selection count label."""
        self._selection_label.setText(
            f"Selected: {len(self.get_selected_layer_names())} mask layers"
        )

    def _emit_selection_changed(self) -> None:
        """Emit the selection_changed signal with selected mask layer names."""
        self.selection_changed.emit(self.get_selected_layer_names())

    def get_selected_layer_names(self) -> list[str]:
        """Return the checked mask layer names."""
        names = []
        for name, item in self._items_by_name.items():
            if item.checkState(0) == Qt.Checked:
                names.append(name)
        return sorted(names)

    def set_selected_layer_names(self, names: list[str]) -> None:
        """Check exactly the named masks after registering an imported layer."""
        selected = set(names)
        blocked = self._tree.blockSignals(True)
        try:
            for name, item in self._items_by_name.items():
                item.setCheckState(0, Qt.Checked if name in selected else Qt.Unchecked)
        finally:
            self._tree.blockSignals(blocked)
        self._update_selection_label()
        self._emit_selection_changed()
