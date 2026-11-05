"""Named live-set controls for the Data table's Sets view."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class NeuronSetsWidget(QWidget):
    changed = Signal()
    heatmap_requested = Signal(object)

    def __init__(self, table):
        super().__init__(table)
        self.table = table
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Sets follow saved Label/Tag rules across the entire Data cohort. "
            "Membership updates when annotations or the cohort change. "
            "Cluster, Manual Heatmap, and Set row filters are not saved in a rule."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Set name")
        row.addWidget(self.name_edit)
        self.create_button = QPushButton("Create Set from Filters")
        self.create_button.clicked.connect(self.create_set)
        row.addWidget(self.create_button)
        layout.addLayout(row)
        self.summary_table = QTableWidget(0, 3)
        self.summary_table.setHorizontalHeaderLabels(["Set", "Neurons", "Live rule"])
        self.summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.summary_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.summary_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.summary_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        self.summary_table.setMaximumHeight(180)
        self.summary_table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.summary_table)
        actions = (
            ("Show Members", self.show_members),
            ("Load Rule into Filters", self.load_rule),
            ("Create Set Heatmap", self.create_heatmap),
            ("Update Set from Filters", self.update_set),
            ("Delete Set", self.delete_set),
        )
        for group in (actions[:3], actions[3:]):
            row = QHBoxLayout()
            for text, callback in group:
                button = QPushButton(text)
                button.clicked.connect(callback)
                row.addWidget(button)
            layout.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def selected_id(self):
        row = self.summary_table.currentRow()
        item = self.summary_table.item(row, 0) if row >= 0 else None
        return item.data(Qt.UserRole) if item else None

    def _selection_changed(self):
        item = self.table.neuron_sets.get(self.selected_id())
        if item:
            self.name_edit.setText(item.name)

    def refresh(self):
        selected = self.selected_id()
        items = self.table.neuron_sets.sets()
        blocked = self.summary_table.blockSignals(True)
        try:
            self.summary_table.setRowCount(len(items))
            for row, item in enumerate(items):
                name = QTableWidgetItem(item.name)
                name.setData(Qt.UserRole, item.set_id)
                self.summary_table.setItem(row, 0, name)
                self.summary_table.setItem(
                    row,
                    1,
                    QTableWidgetItem(str(len(item.rule.members(self.table._entries)))),
                )
                self.summary_table.setItem(
                    row, 2, QTableWidgetItem(item.rule.description())
                )
                if selected == item.set_id:
                    self.summary_table.selectRow(row)
        finally:
            self.summary_table.blockSignals(blocked)

    def _save(self, set_id=None):
        try:
            item = self.table.neuron_sets.save(
                self.name_edit.text(), self.table.annotation_rule(), set_id=set_id
            )
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.changed.emit()
        for row in range(self.summary_table.rowCount()):
            if self.summary_table.item(row, 0).data(Qt.UserRole) == item.set_id:
                self.summary_table.selectRow(row)
        self.status.setText(f"Saved live set {item.name!r}.")

    def create_set(self):
        self._save()

    def update_set(self):
        if self.selected_id():
            self._save(self.selected_id())
        else:
            self.status.setText("Select a set to update.")

    def delete_set(self):
        if self.selected_id():
            self.table.neuron_sets.remove(self.selected_id())
            self.changed.emit()

    def load_rule(self):
        item = self.table.neuron_sets.get(self.selected_id())
        if item:
            self.table.set_annotation_rule(item.rule)

    def show_members(self):
        item = self.table.neuron_sets.get(self.selected_id())
        if item:
            self.table.show_set_members(item.set_id)

    def create_heatmap(self):
        item = self.table.neuron_sets.get(self.selected_id())
        if item is None:
            self.status.setText("Select a set first.")
            return
        members = item.rule.members(self.table._entries)
        if not members:
            self.status.setText("The selected set has no neurons in Data.")
            return
        self.heatmap_requested.emit({**item.to_state(), "file_ids": list(members)})
