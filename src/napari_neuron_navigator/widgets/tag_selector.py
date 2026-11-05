"""Compact checkable tag picker shared by filtering and tag actions."""

from __future__ import annotations

from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtWidgets import QComboBox

from ..neuron_table_ops import label_sort_key


class TagSelector(QComboBox):
    """Select named tags, all neurons, or untagged neurons exclusively."""

    selection_changed = Signal(object)

    def __init__(self, parent=None, *, all_text="All tags", include_untagged=True):
        super().__init__(parent)
        self._all_text = all_text
        self._include_untagged = include_untagged
        self._selection = None
        self._updating = False
        self._popup_open = False
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(12)
        self.view().viewport().installEventFilter(self)
        self.view().installEventFilter(self)
        self.currentIndexChanged.connect(self._index_changed)
        self.set_tag_options({}, 0)

    def selected_tags(self) -> tuple[str, ...]:
        return self._selection if isinstance(self._selection, tuple) else ()

    def tag_filter(self):
        """Return None (all), empty string (untagged), or checked tag names."""
        return self._selection

    def set_selection(self, value, *, emit=True):
        if value is None or value == "":
            selected = value
        else:
            tags = (value,) if isinstance(value, str) else value
            selected = tuple(sorted(set(tags), key=label_sort_key)) or None
        changed = selected != self._selection
        self._selection = selected
        self._update_display()
        if emit and changed:
            self.selection_changed.emit(selected)

    def set_tag_options(self, counts, empty_count):
        counts = dict(counts)
        for tag in self.selected_tags():
            counts.setdefault(tag, 0)
        blocked = self.blockSignals(True)
        self._updating = True
        try:
            self.clear()
            self.addItem(self._all_text, None)
            if self._include_untagged:
                self.addItem(f"No tags ({empty_count:,})", "")
            for tag in sorted(counts, key=label_sort_key):
                self.addItem(f"{tag} ({counts[tag]:,})", tag)
            for index in range(self.count()):
                item = self.model().item(index)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        finally:
            self._updating = False
            self.blockSignals(blocked)
        self._update_display()

    def _update_display(self):
        self._updating = True
        blocked = self.blockSignals(True)
        try:
            checked = []
            selected_tags = set(self.selected_tags())
            for index in range(self.count()):
                data = self.itemData(index)
                active = (
                    data == self._selection
                    if data in (None, "")
                    else data in selected_tags
                )
                self.setItemData(
                    index, Qt.Checked if active else Qt.Unchecked, Qt.CheckStateRole
                )
                if active:
                    checked.append(index)
            self.setCurrentIndex(checked[0] if checked else 0)
            tags = self.selected_tags()
            text = (
                f"{len(tags)} tags selected"
                if len(tags) > 1
                else self.itemText(self.currentIndex())
            )
            self.lineEdit().setText(text)
            self.setToolTip(
                "Checked tags: " + ", ".join(tags)
                if tags
                else self._all_text
                if self._selection is None
                else "Untagged neurons"
            )
        finally:
            self.blockSignals(blocked)
            self._updating = False

    def _index_changed(self, index):
        # Retain QComboBox's single-selection API for programmatic callers.
        if not self._updating and not self._popup_open and index >= 0:
            self.set_selection(self.itemData(index))

    def _toggle(self, index):
        if not 0 <= index < self.count():
            return
        value = self.itemData(index)
        if value in (None, ""):
            self.set_selection(value)
        else:
            selected = set(self.selected_tags())
            if value in selected:
                selected.remove(value)
            else:
                selected.add(value)
            self.set_selection(selected)

    def showPopup(self):
        self._popup_open = True
        super().showPopup()

    def hidePopup(self):
        super().hidePopup()
        self._popup_open = False
        self._update_display()

    def eventFilter(self, watched, event):
        if watched is self.view().viewport():
            if event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick):
                return True
            if event.type() == QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                if index.isValid():
                    self._toggle(index.row())
                return True
        if (
            watched in (self.view(), self.view().viewport())
            and event.type() == QEvent.KeyPress
            and event.key()
            in (
                Qt.Key_Space,
                Qt.Key_Return,
                Qt.Key_Enter,
            )
        ):
            self._toggle(self.view().currentIndex().row())
            if event.key() != Qt.Key_Space:
                self.hidePopup()
            return True
        return super().eventFilter(watched, event)
