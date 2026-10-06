"""Region selector widget with hierarchical Allen CCF structure tree."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..logging_utils import startup_timing

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas

logger = logging.getLogger(__name__)


def _get_parent_structure_id(struct: dict[str, Any]) -> int | None:
    """Return a structure's parent ID, deriving it from the hierarchy path if needed."""
    parent_id = struct.get("parent_structure_id")
    if parent_id is not None:
        return int(parent_id)

    structure_id_path = struct.get("structure_id_path") or []
    if len(structure_id_path) < 2:
        return None
    return int(structure_id_path[-2])


class RegionSelectorWidget(QWidget):
    """Widget for hierarchical brain region selection.

    This widget displays the Allen CCF brain region hierarchy as a tree
    and allows users to select regions for filtering neurons.

    Signals
    -------
    selection_changed
        Emitted when the selection changes, with list of selected region acronyms.
    """

    selection_changed = Signal(list)  # Emits list of selected acronyms

    def __init__(
        self,
        atlas: BrainGlobeAtlas | None = None,
        *,
        single_select: bool = False,
        allowed_structure_ids: set[int] | None = None,
        show_include_children: bool = True,
        force_include_children: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._atlas = atlas
        self._structure_map: dict[int, dict] = {}
        self._items_by_id: dict[int, QTreeWidgetItem] = {}
        self._single_select = bool(single_select)
        self._allowed_structure_ids = (
            {int(struct_id) for struct_id in allowed_structure_ids}
            if allowed_structure_ids is not None
            else None
        )
        self._show_include_children = bool(show_include_children)
        self._force_include_children = bool(force_include_children)
        self._selection_change_depth = 0

        self._setup_ui()
        self._apply_include_children_visibility()

        if atlas is not None:
            self.set_atlas(atlas)

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Search bar
        search_layout = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search regions...")
        self._search_input.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self._search_input)

        self._clear_search_btn = QPushButton("Clear")
        self._clear_search_btn.clicked.connect(self._clear_search)
        search_layout.addWidget(self._clear_search_btn)
        layout.addLayout(search_layout)

        # Tree widget
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Region", "Acronym"])
        header = self._tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree)

        # Selection info
        info_layout = QHBoxLayout()
        self._selection_label = QLabel("Selected: 0 regions")
        info_layout.addWidget(self._selection_label)

        self._clear_btn = QPushButton("Clear Selection")
        self._clear_btn.clicked.connect(self._clear_selection)
        info_layout.addWidget(self._clear_btn)
        layout.addLayout(info_layout)

        # Include children option
        self._include_children_cb = QCheckBox("Include child regions")
        self._include_children_cb.setChecked(True)
        self._include_children_cb.stateChanged.connect(self._emit_selection_changed)
        layout.addWidget(self._include_children_cb)

    def _apply_include_children_visibility(self) -> None:
        """Update include-children checkbox visibility and enabled state."""
        if not hasattr(self, "_include_children_cb"):
            return

        visible = self._show_include_children and not self._force_include_children
        self._include_children_cb.setVisible(visible)
        self._include_children_cb.setEnabled(not self._force_include_children)
        if self._force_include_children:
            self._include_children_cb.setChecked(True)

    def set_atlas(self, atlas: BrainGlobeAtlas) -> None:
        """Set the atlas and populate the tree.

        Parameters
        ----------
        atlas : BrainGlobeAtlas
            The atlas to use for region hierarchy.
        """
        self._atlas = atlas
        self._populate_tree()

    def clear(self) -> None:
        """Clear the current tree contents and selection state."""
        self._tree.blockSignals(True)
        self._tree.clear()
        self._tree.blockSignals(False)
        self._items_by_id.clear()
        self._structure_map.clear()
        self._update_selection_label()

    def set_single_select(self, enabled: bool) -> None:
        """Configure whether only one checked region is allowed."""
        self._single_select = bool(enabled)

    def set_allowed_structure_ids(self, structure_ids: set[int] | None) -> None:
        """Restrict the visible atlas tree to the supplied structure IDs."""
        self._allowed_structure_ids = (
            {int(struct_id) for struct_id in structure_ids}
            if structure_ids is not None
            else None
        )
        if self._atlas is not None:
            self._populate_tree()

    def set_include_children_controls(
        self,
        *,
        show_include_children: bool | None = None,
        force_include_children: bool | None = None,
    ) -> None:
        """Update include-children checkbox behavior."""
        if show_include_children is not None:
            self._show_include_children = bool(show_include_children)
        if force_include_children is not None:
            self._force_include_children = bool(force_include_children)
        self._apply_include_children_visibility()

    def _is_structure_allowed(self, struct_id: int) -> bool:
        """Return whether a structure should be shown in the tree."""
        allowed = self._allowed_structure_ids
        return allowed is None or int(struct_id) in allowed

    def _populate_tree(self) -> None:
        """Populate the tree with the atlas structure hierarchy."""
        if self._atlas is None:
            return

        atlas_name = getattr(self._atlas, "atlas_name", "unknown")
        with startup_timing(
            logger,
            "region_selector_populate_tree",
            log_start=False,
            atlas=atlas_name,
        ) as timing:
            self._tree.blockSignals(True)
            self._tree.clear()
            self._items_by_id.clear()
            self._structure_map.clear()

            total_structures = 0
            # Build structure lookup
            for struct_id, struct in self._atlas.structures.items():
                total_structures += 1
                if isinstance(struct_id, int):
                    self._structure_map[struct_id] = struct

            # Find root structures (those without a parent in our map)
            root_ids = set()
            allowed_structures = 0
            for struct_id, struct in self._structure_map.items():
                if not self._is_structure_allowed(struct_id):
                    continue
                allowed_structures += 1
                parent_id = _get_parent_structure_id(struct)
                if parent_id is None or not self._is_structure_allowed(parent_id):
                    root_ids.add(struct_id)

            # Build tree recursively from roots
            for root_id in sorted(root_ids):
                self._add_structure_to_tree(root_id, None)

            timing.set(
                total_structures=total_structures,
                allowed_structures=allowed_structures,
                root_count=len(root_ids),
                created_items=len(self._items_by_id),
            )
            self._tree.blockSignals(False)
            self._update_selection_label()

    def _add_structure_to_tree(
        self,
        struct_id: int,
        parent_item: QTreeWidgetItem | None,
    ) -> QTreeWidgetItem | None:
        """Add a structure and its children to the tree.

        Parameters
        ----------
        struct_id : int
            The structure ID to add.
        parent_item : QTreeWidgetItem or None
            The parent tree item, or None for root items.

        Returns
        -------
        QTreeWidgetItem or None
            The created tree item.
        """
        struct = self._structure_map.get(struct_id)
        if struct is None or not self._is_structure_allowed(struct_id):
            return None

        name = struct.get("name", f"Region {struct_id}")
        acronym = struct.get("acronym", "")

        if parent_item is None:
            item = QTreeWidgetItem(self._tree, [name, acronym])
        else:
            item = QTreeWidgetItem(parent_item, [name, acronym])

        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Unchecked)
        item.setData(0, Qt.UserRole, struct_id)

        self._items_by_id[struct_id] = item

        # Add children
        children_ids = self._get_child_structure_ids(struct_id)
        for child_id in sorted(children_ids):
            self._add_structure_to_tree(child_id, item)

        return item

    def _get_child_structure_ids(self, parent_id: int) -> list[int]:
        """Get direct child structure IDs for a parent.

        Parameters
        ----------
        parent_id : int
            The parent structure ID.

        Returns
        -------
        list[int]
            List of child structure IDs.
        """
        children = []
        for struct_id, struct in self._structure_map.items():
            if _get_parent_structure_id(struct) == parent_id:
                children.append(struct_id)
        return children

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle item check state changes."""
        if column != 0:
            return

        if self._selection_change_depth > 0:
            return

        if self._single_select and item.checkState(0) == Qt.Checked:
            self._selection_change_depth += 1
            self._tree.blockSignals(True)
            try:
                for other_item in self._items_by_id.values():
                    if other_item is not item:
                        other_item.setCheckState(0, Qt.Unchecked)
            finally:
                self._tree.blockSignals(False)
                self._selection_change_depth -= 1

        # Update selection count and emit signal
        self._update_selection_label()
        self._emit_selection_changed()

    def _on_search_changed(self, text: str) -> None:
        """Filter tree items based on search text."""
        text = text.lower().strip()

        def set_item_visibility(item: QTreeWidgetItem) -> bool:
            """Recursively set item visibility. Returns True if visible."""
            # Check children first
            child_visible = False
            for i in range(item.childCount()):
                if set_item_visibility(item.child(i)):
                    child_visible = True

            # Check this item's text
            name = item.text(0).lower()
            acronym = item.text(1).lower()
            matches = not text or text in name or text in acronym

            # Item is visible if it matches or has visible children
            visible = matches or child_visible
            item.setHidden(not visible)

            # Expand items that match
            if matches and text:
                item.setExpanded(True)

            return visible

        for i in range(self._tree.topLevelItemCount()):
            set_item_visibility(self._tree.topLevelItem(i))

    def _clear_search(self) -> None:
        """Clear the search input and show all items."""
        self._search_input.clear()
        self._show_all_items()

    def _show_all_items(self) -> None:
        """Show all items in the tree."""

        def show_item(item: QTreeWidgetItem) -> None:
            item.setHidden(False)
            for i in range(item.childCount()):
                show_item(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            show_item(self._tree.topLevelItem(i))

    def _clear_selection(self) -> None:
        """Clear all selections."""
        self._tree.blockSignals(True)

        def uncheck_item(item: QTreeWidgetItem) -> None:
            item.setCheckState(0, Qt.Unchecked)
            for i in range(item.childCount()):
                uncheck_item(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            uncheck_item(self._tree.topLevelItem(i))

        self._tree.blockSignals(False)
        self._update_selection_label()
        self._emit_selection_changed()

    def _update_selection_label(self) -> None:
        """Update the selection count label."""
        selected = self.get_selected_acronyms(include_children=False)
        self._selection_label.setText(f"Selected: {len(selected)} regions")

    def _emit_selection_changed(self) -> None:
        """Emit the selection_changed signal with current selection."""
        acronyms = self.get_query_acronyms()
        self.selection_changed.emit(acronyms)

    def include_children_enabled(self) -> bool:
        """Return whether descendant regions should be included in queries."""
        if getattr(self, "_force_include_children", False):
            return True
        return self._include_children_cb.isChecked()

    def set_include_children_enabled(self, enabled: bool) -> None:
        """Set whether queries include descendants and emit one update."""
        if getattr(self, "_force_include_children", False):
            enabled = True
        self._include_children_cb.setChecked(bool(enabled))

    def get_query_acronyms(self) -> list[str]:
        """Return selected acronyms using the current include-children toggle."""
        return self.get_selected_acronyms(
            include_children=self.include_children_enabled()
        )

    def get_selected_acronyms(self, include_children: bool = True) -> list[str]:
        """Get the list of selected region acronyms.

        Parameters
        ----------
        include_children : bool, default=True
            If True, include child region acronyms for selected parent regions.

        Returns
        -------
        list[str]
            List of selected region acronyms.
        """
        selected_ids = set()

        def collect_checked(item: QTreeWidgetItem) -> None:
            if item.checkState(0) == Qt.Checked:
                struct_id = item.data(0, Qt.UserRole)
                if struct_id is not None:
                    selected_ids.add(struct_id)

                    if include_children:
                        # Add all descendants
                        self._collect_descendant_ids(struct_id, selected_ids)
            else:
                # Check children even if parent is unchecked
                for i in range(item.childCount()):
                    collect_checked(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            collect_checked(self._tree.topLevelItem(i))

        # Convert IDs to acronyms
        acronyms = []
        for struct_id in selected_ids:
            struct = self._structure_map.get(struct_id)
            if struct and struct.get("acronym"):
                acronyms.append(struct["acronym"])

        return sorted(set(acronyms))

    def _collect_descendant_ids(self, struct_id: int, result: set[int]) -> None:
        """Recursively collect all descendant structure IDs.

        Parameters
        ----------
        struct_id : int
            The structure ID to get descendants for.
        result : set[int]
            Set to add descendant IDs to.
        """
        children = self._get_child_structure_ids(struct_id)
        for child_id in children:
            result.add(child_id)
            self._collect_descendant_ids(child_id, result)

    def get_selected_ids(self, include_children: bool = True) -> list[int]:
        """Get the list of selected region IDs.

        Parameters
        ----------
        include_children : bool, default=True
            If True, include child region IDs for selected parent regions.

        Returns
        -------
        list[int]
            List of selected region IDs.
        """
        selected_ids = set()

        def collect_checked(item: QTreeWidgetItem) -> None:
            if item.checkState(0) == Qt.Checked:
                struct_id = item.data(0, Qt.UserRole)
                if struct_id is not None:
                    selected_ids.add(struct_id)

                    if include_children:
                        self._collect_descendant_ids(struct_id, selected_ids)
            else:
                for i in range(item.childCount()):
                    collect_checked(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            collect_checked(self._tree.topLevelItem(i))

        return sorted(selected_ids)

    def get_single_selected_region(self) -> tuple[int, str] | None:
        """Return the single directly selected region as ``(id, acronym)``."""
        selected_ids = self.get_selected_ids(include_children=False)
        if not selected_ids:
            return None

        struct_id = int(selected_ids[0])
        struct = self._structure_map.get(struct_id)
        if struct is None:
            return None

        acronym = str(struct.get("acronym", "")).strip()
        if not acronym:
            return None
        return struct_id, acronym

    def select_region_by_id(self, struct_id: int | None) -> None:
        """Programmatically select one region by structure ID."""
        self.set_selected_ids(() if struct_id is None else (struct_id,))

    def set_selected_ids(self, struct_ids: list[int] | tuple[int, ...]) -> None:
        """Replace the directly checked selection with atlas structure IDs.

        Unknown IDs are ignored.  This is intentionally a direct-selection
        API: descendant expansion remains controlled by the separate
        include-children option.
        """
        selected_ids = {int(struct_id) for struct_id in struct_ids}
        self._tree.blockSignals(True)
        try:
            for current_id, item in self._items_by_id.items():
                state = Qt.Checked if current_id in selected_ids else Qt.Unchecked
                item.setCheckState(0, state)
        finally:
            self._tree.blockSignals(False)

        self._update_selection_label()
        self._emit_selection_changed()

    def select_regions(self, acronyms: list[str]) -> None:
        """Programmatically select regions by acronym.

        Parameters
        ----------
        acronyms : list[str]
            List of region acronyms to select.
        """
        acronym_set = set(acronyms)
        self._tree.blockSignals(True)

        def check_matching(item: QTreeWidgetItem) -> None:
            acronym = item.text(1)
            if acronym in acronym_set:
                item.setCheckState(0, Qt.Checked)
            for i in range(item.childCount()):
                check_matching(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            check_matching(self._tree.topLevelItem(i))

        self._tree.blockSignals(False)
        self._update_selection_label()
        self._emit_selection_changed()
