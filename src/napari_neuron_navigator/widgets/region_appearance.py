"""Staged editor for shared atlas-region appearance overrides."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..neuron_palette import neuron_palette
from ..region_appearance import (
    RegionAppearanceStore,
    atlas_identity,
    load_region_palette,
    prepare_region_palette_import,
    save_region_palette,
    structure_catalog,
    structure_path,
)


_ROLE_REGION_ID = Qt.UserRole


class RegionAppearanceEditor(QWidget):
    """Edit a draft palette and emit only explicitly applied changes."""

    appearance_applied = Signal(object)
    dirty_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._atlas = None
        self._identity = ("", "")
        self._applied_by_identity: dict[tuple[str, str], RegionAppearanceStore] = {}
        self._draft_by_identity: dict[tuple[str, str], RegionAppearanceStore] = {}
        self._root_region_ids: tuple[int, ...] = ()
        self._catalog: dict[int, Mapping] = {}
        self._items_by_id: dict[int, QTreeWidgetItem] = {}
        self._row_widgets: dict[int, dict[str, object]] = {}
        self._updating_widgets = False
        self._setup_ui()
        self._refresh_tree()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._hint_label = QLabel(
            "Appearance changes are staged. Apply them to update existing CCF "
            "and flatmap overlays; region query selections are unaffected. "
            "Outline controls apply to flatmap outlines (CCF contours are not "
            "created)."
        )
        self._hint_label.setWordWrap(True)
        layout.addWidget(self._hint_label)

        search_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search selected region subtree...")
        self._search_edit.textChanged.connect(self._apply_search_filter)
        search_row.addWidget(self._search_edit)
        self._clear_search_btn = QPushButton("Clear")
        self._clear_search_btn.clicked.connect(self._search_edit.clear)
        search_row.addWidget(self._clear_search_btn)
        layout.addLayout(search_row)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(
            ["Region", "Color", "Fill", "Fill %", "Outline", "Outline %", "Source"]
        )
        self._tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        header = self._tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 7):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        layout.addWidget(self._tree)

        color_row = QHBoxLayout()
        self._distinct_btn = QPushButton("Assign Distinct Colors")
        self._distinct_btn.clicked.connect(self._assign_distinct_colors)
        color_row.addWidget(self._distinct_btn)
        self._atlas_color_btn = QPushButton("Use Atlas Color")
        self._atlas_color_btn.clicked.connect(self._use_atlas_color)
        color_row.addWidget(self._atlas_color_btn)
        self._inherit_btn = QPushButton("Inherit")
        self._inherit_btn.clicked.connect(self._inherit_selected)
        color_row.addWidget(self._inherit_btn)
        layout.addLayout(color_row)

        file_row = QHBoxLayout()
        self._import_btn = QPushButton("Import Palette...")
        self._import_btn.clicked.connect(self._import_palette)
        file_row.addWidget(self._import_btn)
        self._export_btn = QPushButton("Export Applied Palette...")
        self._export_btn.clicked.connect(self._export_palette)
        file_row.addWidget(self._export_btn)
        layout.addLayout(file_row)

        apply_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self.apply_draft)
        apply_row.addWidget(self._apply_btn)
        self._revert_btn = QPushButton("Revert")
        self._revert_btn.clicked.connect(self.discard_draft)
        apply_row.addWidget(self._revert_btn)
        self._status_label = QLabel("No appearance changes staged.")
        self._status_label.setWordWrap(True)
        apply_row.addWidget(self._status_label, stretch=1)
        layout.addLayout(apply_row)

    @property
    def applied_store(self) -> RegionAppearanceStore:
        return self._applied_store().copy()

    @property
    def draft_store(self) -> RegionAppearanceStore:
        return self._draft_store().copy()

    def _applied_store(self) -> RegionAppearanceStore:
        store = self._applied_by_identity.get(self._identity)
        if store is None:
            store = RegionAppearanceStore(
                atlas_name=self._identity[0],
                atlas_version=self._identity[1],
            )
            self._applied_by_identity[self._identity] = store
        return store

    def _draft_store(self) -> RegionAppearanceStore:
        store = self._draft_by_identity.get(self._identity)
        if store is None:
            store = self._applied_store().copy()
            self._draft_by_identity[self._identity] = store
        return store

    def set_atlas(self, atlas: object | None) -> None:
        self._atlas = atlas
        self._identity = atlas_identity(atlas)
        self._catalog = structure_catalog(getattr(atlas, "structures", None))
        if self._identity not in self._applied_by_identity and self._identity[0]:
            versionless_identity = (self._identity[0], "")
            versionless = self._applied_by_identity.get(versionless_identity)
            if versionless is not None:
                current = versionless.copy()
                current.atlas_version = self._identity[1]
                self._applied_by_identity[self._identity] = current.copy()
                self._draft_by_identity[self._identity] = current
        self._applied_store()
        self._draft_store()
        self._refresh_tree()
        self._update_dirty_state()

    def set_selection(self, root_region_ids: list[int] | tuple[int, ...]) -> None:
        self._root_region_ids = tuple(
            sorted({int(value) for value in root_region_ids if int(value) > 0})
        )
        self._refresh_tree()

    def load_applied_store(self, store: RegionAppearanceStore) -> None:
        identity = (store.atlas_name, store.atlas_version)
        self._applied_by_identity[identity] = store.copy()
        self._draft_by_identity[identity] = store.copy()
        if identity == self._identity or (
            store.atlas_name == self._identity[0]
            and (not store.atlas_version or not self._identity[1])
        ):
            if identity != self._identity:
                current = store.copy()
                current.atlas_version = self._identity[1]
                self._applied_by_identity[self._identity] = current.copy()
                self._draft_by_identity[self._identity] = current
            self._refresh_tree()
            self._update_dirty_state()

    def has_unapplied_changes(self) -> bool:
        return self._draft_store() != self._applied_store()

    def apply_draft(self) -> None:
        applied = self._draft_store().copy()
        self._applied_by_identity[self._identity] = applied
        self._draft_by_identity[self._identity] = applied.copy()
        self._refresh_row_widgets()
        self._update_dirty_state()
        self._status_label.setText(
            f"Applied appearance overrides for {len(applied.region_ids)} region(s)."
        )
        self.appearance_applied.emit(applied.copy())

    def discard_draft(self) -> None:
        self._draft_by_identity[self._identity] = self._applied_store().copy()
        self._refresh_row_widgets()
        self._update_dirty_state()
        self._status_label.setText("Reverted staged appearance changes.")

    def resolve_unapplied_before_save(self) -> bool:
        """Resolve dirty state before project save; return whether to continue."""
        if not self.has_unapplied_changes():
            return True
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Question)
        message.setWindowTitle("Unapplied Region Appearance Changes")
        message.setText(
            "Region appearance edits are staged but not applied. What should be "
            "saved in the project?"
        )
        apply_button = message.addButton("Apply", QMessageBox.AcceptRole)
        discard_button = message.addButton("Discard", QMessageBox.DestructiveRole)
        message.addButton("Cancel", QMessageBox.RejectRole)
        message.setDefaultButton(apply_button)
        message.exec()
        clicked = message.clickedButton()
        if clicked is apply_button:
            self.apply_draft()
            return True
        if clicked is discard_button:
            self.discard_draft()
            return True
        return False

    def _refresh_tree(self) -> None:
        selected_ids = (
            set(self._selected_region_ids()) if hasattr(self, "_tree") else set()
        )
        self._updating_widgets = True
        self._tree.clear()
        self._items_by_id.clear()
        self._row_widgets.clear()
        if not self._catalog or not self._root_region_ids:
            self._tree.setEnabled(False)
            self._hint_label.setText(
                "Select one or more Atlas or Custom Regions to edit their appearance."
            )
            self._updating_widgets = False
            self._update_dirty_state()
            return

        self._tree.setEnabled(True)
        self._hint_label.setText(
            "Appearance changes are staged. Apply them to update existing CCF "
            "and flatmap overlays; region query selections are unaffected. "
            "Outline controls apply to flatmap outlines (CCF contours are not "
            "created)."
        )
        children_by_parent: dict[int, list[int]] = {}
        for region_id in self._catalog:
            path = structure_path(region_id, self._catalog)
            if len(path) >= 2:
                children_by_parent.setdefault(path[-2], []).append(region_id)

        selected_roots = set(self._root_region_ids)
        roots = [
            region_id
            for region_id in self._root_region_ids
            if not any(
                ancestor in selected_roots
                for ancestor in structure_path(region_id, self._catalog)[:-1]
            )
        ]
        for region_id in roots:
            self._add_region_item(region_id, None, children_by_parent)
        for region_id in selected_ids:
            item = self._items_by_id.get(region_id)
            if item is not None:
                item.setSelected(True)
        self._updating_widgets = False
        self._refresh_row_widgets()
        self._apply_search_filter(self._search_edit.text())
        self._update_dirty_state()

    def _add_region_item(
        self,
        region_id: int,
        parent: QTreeWidgetItem | None,
        children_by_parent: Mapping[int, list[int]],
    ) -> None:
        structure = self._catalog.get(region_id, {})
        name = str(structure.get("name", f"Region {region_id}"))
        acronym = str(structure.get("acronym", "") or "")
        label = (
            f"{name} ({acronym}, {region_id})" if acronym else f"{name} ({region_id})"
        )
        item = QTreeWidgetItem(self._tree if parent is None else parent, [label])
        item.setData(0, _ROLE_REGION_ID, int(region_id))
        self._items_by_id[int(region_id)] = item

        color_button = QPushButton()
        color_button.setToolTip("Choose a custom color for this region.")
        color_button.clicked.connect(
            lambda _checked=False, rid=int(region_id): self._choose_color(rid)
        )
        self._tree.setItemWidget(item, 1, color_button)

        fill_combo = self._visibility_combo()
        fill_combo.currentIndexChanged.connect(
            lambda _index, rid=int(region_id), combo=fill_combo: self._set_visibility(
                rid, "fill_visible", combo.currentIndex()
            )
        )
        self._tree.setItemWidget(item, 2, fill_combo)
        fill_opacity = self._opacity_spin()
        fill_opacity.valueChanged.connect(
            lambda value, rid=int(region_id): self._set_opacity(
                rid, "fill_opacity", value
            )
        )
        self._tree.setItemWidget(item, 3, fill_opacity)

        outline_combo = self._visibility_combo()
        outline_combo.currentIndexChanged.connect(
            lambda _index, rid=int(region_id), combo=outline_combo: (
                self._set_visibility(rid, "outline_visible", combo.currentIndex())
            )
        )
        self._tree.setItemWidget(item, 4, outline_combo)
        outline_opacity = self._opacity_spin()
        outline_opacity.valueChanged.connect(
            lambda value, rid=int(region_id): self._set_opacity(
                rid, "outline_opacity", value
            )
        )
        self._tree.setItemWidget(item, 5, outline_opacity)

        source_label = QLabel()
        self._tree.setItemWidget(item, 6, source_label)
        self._row_widgets[int(region_id)] = {
            "color": color_button,
            "fill_visible": fill_combo,
            "fill_opacity": fill_opacity,
            "outline_visible": outline_combo,
            "outline_opacity": outline_opacity,
            "source": source_label,
        }

        for child_id in sorted(
            children_by_parent.get(int(region_id), ()),
            key=lambda value: str(self._catalog.get(value, {}).get("name", value)),
        ):
            self._add_region_item(child_id, item, children_by_parent)

    @staticmethod
    def _visibility_combo() -> QComboBox:
        combo = QComboBox()
        combo.addItems(["Inherit", "Show", "Hide"])
        return combo

    @staticmethod
    def _opacity_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1.0, 100.0)
        spin.setDecimals(0)
        spin.setSingleStep(5.0)
        spin.setSpecialValueText("Inherit")
        return spin

    def _refresh_row_widgets(self) -> None:
        if not hasattr(self, "_tree"):
            return
        self._updating_widgets = True
        draft = self._draft_store()
        for region_id, widgets in self._row_widgets.items():
            override = draft.override_for(region_id)
            effective = draft.resolve(region_id, self._catalog)
            color_button = widgets["color"]
            assert isinstance(color_button, QPushButton)
            red, green, blue = [
                int(round(component * 255)) for component in effective.color_rgba[:3]
            ]
            color_button.setStyleSheet(
                f"background-color: rgb({red}, {green}, {blue}); min-width: 52px;"
            )
            color_button.setText(
                "Custom"
                if override.color_mode == "custom"
                else "Atlas"
                if override.color_mode == "atlas"
                else "Inherit"
            )
            for field_name in ("fill_visible", "outline_visible"):
                combo = widgets[field_name]
                assert isinstance(combo, QComboBox)
                value = getattr(override, field_name)
                combo.setCurrentIndex(0 if value is None else 1 if value else 2)
            for field_name in ("fill_opacity", "outline_opacity"):
                spin = widgets[field_name]
                assert isinstance(spin, QDoubleSpinBox)
                value = getattr(override, field_name)
                spin.setValue(-1.0 if value is None else float(value) * 100.0)
            source = widgets["source"]
            assert isinstance(source, QLabel)
            if override.color_mode == "custom":
                color_source = "Custom"
            elif override.color_mode == "atlas":
                color_source = "Atlas color"
            else:
                color_source = "Atlas default"
                for ancestor_id in reversed(
                    structure_path(region_id, self._catalog)[:-1]
                ):
                    ancestor = draft.override_for(ancestor_id)
                    if ancestor.color_mode is None:
                        continue
                    structure = self._catalog.get(ancestor_id, {})
                    acronym = str(structure.get("acronym", ancestor_id) or ancestor_id)
                    color_source = f"Inherited: {acronym}"
                    break
            source.setText(color_source)
        self._updating_widgets = False

    def _set_visibility(self, region_id: int, field_name: str, index: int) -> None:
        if self._updating_widgets:
            return
        value = None if index == 0 else index == 1
        self._update_override(region_id, **{field_name: value})

    def _set_opacity(self, region_id: int, field_name: str, value: float) -> None:
        if self._updating_widgets:
            return
        opacity = None if value < 0.0 else float(value) / 100.0
        self._update_override(region_id, **{field_name: opacity})

    def _choose_color(self, region_id: int) -> None:
        effective = self._draft_store().resolve(region_id, self._catalog)
        initial = QColor.fromRgbF(*effective.color_rgba)
        selected = QColorDialog.getColor(initial, self, "Choose Region Color")
        if not selected.isValid():
            return
        self._update_override(
            region_id,
            color_mode="custom",
            color_rgb=(selected.redF(), selected.greenF(), selected.blueF()),
        )

    def _update_override(self, region_id: int, **changes: object) -> None:
        draft = self._draft_store()
        override = draft.override_for(region_id).updated(**changes)
        draft.set_override(region_id, override)
        self._refresh_row_widgets()
        self._update_dirty_state()

    def _selected_region_ids(self) -> list[int]:
        ids: list[int] = []
        for item in self._tree.selectedItems():
            region_id = item.data(0, _ROLE_REGION_ID)
            if region_id is not None:
                ids.append(int(region_id))
        return sorted(set(ids))

    def _require_selected_ids(self) -> list[int]:
        ids = self._selected_region_ids()
        if not ids:
            show_warning("Select one or more rows in Region Appearance first.")
        return ids

    def _assign_distinct_colors(self) -> None:
        region_ids = self._require_selected_ids()
        if not region_ids:
            return
        draft = self._draft_store()
        for region_id, rgba in zip(region_ids, neuron_palette(len(region_ids))):
            draft.set_override(
                region_id,
                draft.override_for(region_id).updated(
                    color_mode="custom", color_rgb=tuple(rgba[:3])
                ),
            )
        self._refresh_row_widgets()
        self._update_dirty_state()

    def _use_atlas_color(self) -> None:
        region_ids = self._require_selected_ids()
        if not region_ids:
            return
        for region_id in region_ids:
            self._update_override(region_id, color_mode="atlas", color_rgb=None)

    def _inherit_selected(self) -> None:
        region_ids = self._require_selected_ids()
        if not region_ids:
            return
        draft = self._draft_store()
        for region_id in region_ids:
            draft.clear_override(region_id)
        self._refresh_row_widgets()
        self._update_dirty_state()

    def _update_dirty_state(self) -> None:
        dirty = self.has_unapplied_changes()
        if hasattr(self, "_apply_btn"):
            self._apply_btn.setEnabled(dirty)
            self._revert_btn.setEnabled(dirty)
            if dirty:
                self._status_label.setText("Unapplied appearance changes.")
        self.dirty_changed.emit(dirty)

    def _apply_search_filter(self, text: str) -> None:
        query = str(text).strip().casefold()

        def update(item: QTreeWidgetItem) -> bool:
            child_visible = False
            for index in range(item.childCount()):
                child_visible = update(item.child(index)) or child_visible
            matches = not query or query in item.text(0).casefold()
            visible = matches or child_visible
            item.setHidden(not visible)
            if query and child_visible:
                item.setExpanded(True)
            return visible

        for index in range(self._tree.topLevelItemCount()):
            update(self._tree.topLevelItem(index))

    def _import_palette(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Region Palette",
            "",
            "Region Palette (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            imported = load_region_palette(path)
        except Exception as exc:
            show_warning(f"Could not import region palette: {exc}")
            return
        current_name, current_version = self._identity
        if not current_name or not self._catalog:
            show_warning("Load an atlas before importing a region palette.")
            return
        try:
            summary = prepare_region_palette_import(
                imported,
                atlas_name=current_name,
                atlas_version=current_version,
                known_region_ids=self._catalog,
            )
        except ValueError as exc:
            show_warning(str(exc))
            return
        if summary.version_mismatch:
            answer = QMessageBox.question(
                self,
                "Atlas Version Mismatch",
                f"The palette targets atlas version {imported.atlas_version}, but "
                f"the loaded atlas is {current_version}. Import matching region IDs?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        filtered = summary.store
        missing = list(summary.unknown_region_ids)

        prompt = QMessageBox(self)
        prompt.setWindowTitle("Import Region Palette")
        prompt.setText(
            f"Found {len(filtered.region_ids)} matching region override(s). "
            "Merge them with the staged palette or replace it?"
        )
        merge_button = prompt.addButton("Merge", QMessageBox.AcceptRole)
        replace_button = prompt.addButton("Replace", QMessageBox.DestructiveRole)
        cancel_button = prompt.addButton("Cancel", QMessageBox.RejectRole)
        prompt.setDefaultButton(merge_button)
        prompt.exec()
        clicked = prompt.clickedButton()
        if clicked is cancel_button or clicked is None:
            return
        if clicked is replace_button:
            self._draft_by_identity[self._identity] = filtered
        elif clicked is merge_button:
            self._draft_store().merge(filtered)
        else:
            return
        self._refresh_row_widgets()
        self._update_dirty_state()
        suffix = f" Skipped {len(missing)} unknown region ID(s)." if missing else ""
        self._status_label.setText(
            f"Imported {len(filtered.region_ids)} override(s) into the draft.{suffix}"
        )

    def _export_palette(self) -> None:
        applied = self._applied_store()
        if not applied.region_ids:
            show_warning(
                "No applied region appearance overrides to export. "
                "Customize region colors, visibility, or opacity, then click "
                "Apply before exporting. Staged edits are not included."
            )
            return
        default_name = f"{self._identity[0] or 'atlas'}_region_palette.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Applied Region Palette",
            default_name,
            "Region Palette (*.json);;All Files (*)",
        )
        if not path:
            return
        output = Path(path)
        if output.suffix.lower() != ".json":
            output = output.with_suffix(".json")
        try:
            save_region_palette(output, applied)
        except Exception as exc:
            show_warning(f"Could not export region palette: {exc}")
            return
        show_info(f"Exported region palette: {output.name}")
