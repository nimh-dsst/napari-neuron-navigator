"""Expandable anatomical cohort report in the Regions tab."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import duckdb
from qtpy.QtCore import QObject, Qt, QThread, QTimer, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..analysis.region_profile import (
    RegionalProfileAtlas,
    default_region_profile_path,
    register_region_profile,
)
from ..analysis.region_report import (
    COMBINED_LATERALITY,
    REPORT_COLUMNS,
    REPORT_LATERALITIES,
    RegionReport,
    RegionReportMeasurements,
    aggregate_region_report,
)
from .collapsible_section import CollapsibleSection
from .region_filter_editor import RegionFilterEditorWidget

# A closing dock must not destroy a still-running QThread. Workers own only
# snapshots and finish after cancellation; this keeps their threads alive then.
_LIVE_REPORT_THREADS: set[QThread] = set()
logger = logging.getLogger(__name__)


class RegionalReportWorker(QObject):
    """Validate and summarize a sidecar on a worker-owned DuckDB connection."""

    ready = Signal(int, object)
    error = Signal(int, str)
    done = Signal()

    def __init__(self, generation, source_path, atlas, file_ids, split, exclusions):
        super().__init__()
        self.generation = generation
        self.source_path = source_path
        self.atlas = atlas
        self.file_ids = file_ids
        self.split = split
        self.exclusions = exclusions
        self._cancelled = Event()
        self._connection = None

    def cancel(self) -> None:
        self._cancelled.set()
        if self._connection is not None:
            try:
                self._connection.interrupt()
            except duckdb.Error:
                pass

    def run(self) -> None:
        connection = duckdb.connect()
        self._connection = connection
        try:
            if self._cancelled.is_set():
                return
            atlas = (
                self.atlas
                if isinstance(self.atlas, RegionalProfileAtlas)
                else RegionalProfileAtlas.from_atlas(self.atlas)
            )
            register_region_profile(
                connection,
                default_region_profile_path(self.source_path),
                source_path=self.source_path,
                atlas=atlas,
            )
            if self._cancelled.is_set():
                return
            result = aggregate_region_report(
                connection,
                structures=atlas.structures,
                file_ids=self.file_ids,
                split_laterality=self.split,
                excluded_region_ids=self.exclusions,
            )
            if not self._cancelled.is_set():
                self.ready.emit(self.generation, result)
        except Exception as exc:
            if not self._cancelled.is_set():
                logger.exception("Regional cohort report failed")
                self.error.emit(self.generation, str(exc))
        finally:
            self._connection = None
            connection.close()
            self.done.emit()


class _ReportItem(QTreeWidgetItem):
    """Sort counts and measurements numerically, including expanded groups."""

    def __lt__(self, other):
        column = self.treeWidget().sortColumn()
        left = self.data(column, Qt.UserRole)
        right = other.data(column, Qt.UserRole)
        if (
            isinstance(left, (int, float))
            and isinstance(right, (int, float))
            and left != right
        ):
            return left < right
        if self.text(column) != other.text(column):
            return self.text(column) < other.text(column)
        return self.text(0) < other.text(0)


class RegionReportWidget(QWidget):
    """Show direct regions expanded and unique-neuron aggregates collapsed."""

    build_profile_requested = Signal()

    def __init__(
        self,
        cohort_provider: Callable[[str], Sequence[object]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cohort_provider = cohort_provider
        self._source_path = None
        self._atlas = None
        self._worker = None
        self._thread = None
        self._generation = 0
        self._displayed_generation = 0
        self._request_key = None
        self._closing = False
        self._populating = False
        self._collapsed_groups: set[tuple[int, str]] = set()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._start_report)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.section = CollapsibleSection("Region Report", expanded=False)
        layout.addWidget(self.section)
        content = self.section.content_layout()
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Cohort:"))
        self.cohort_combo = QComboBox()
        self.cohort_combo.addItem("Data table", "table")
        self.cohort_combo.addItem("Selected rows", "selected")
        controls.addWidget(self.cohort_combo)
        self.split_laterality = QCheckBox("Split by laterality")
        controls.addWidget(self.split_laterality)
        content.addLayout(controls)
        actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Report")
        self.refresh_button.clicked.connect(self.refresh)
        actions.addWidget(self.refresh_button)
        self.build_button = QPushButton("Build Regional Profile")
        self.build_button.clicked.connect(self.build_profile_requested.emit)
        actions.addWidget(self.build_button)
        content.addLayout(actions)
        grouping = QHBoxLayout()
        grouping.addWidget(QLabel("Anatomical grouping:"))
        expand = QPushButton("Expand All")
        expand.clicked.connect(self._expand_all)
        grouping.addWidget(expand)
        collapse = QPushButton("Collapse All")
        collapse.clicked.connect(self._collapse_all)
        grouping.addWidget(collapse)
        content.addLayout(grouping)
        note = QLabel(
            "Expand to show direct-region contributions; collapse a branch to "
            "aggregate its descendants. Each neuron is counted once per group. "
            "Percentages and means use the entire selected cohort."
        )
        note.setWordWrap(True)
        content.addWidget(note)
        self.tree = QTreeWidget()
        self.tree.setMinimumHeight(340)
        # Qt's native alternate-row palette can stay white under napari's
        # dark stylesheet. Use the inherited theme background for every row.
        self.tree.setAlternatingRowColors(False)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree.setSortingEnabled(True)
        self.tree.itemExpanded.connect(self._expanded)
        self.tree.itemCollapsed.connect(self._collapsed)
        content.addWidget(self.tree)
        self.exclusion_section = CollapsibleSection("Exclude Regions", expanded=False)
        self.exclusion_editor = RegionFilterEditorWidget(
            anatomical_exclusions_only=True
        )
        self.exclusion_section.content_layout().addWidget(self.exclusion_editor)
        content.addWidget(self.exclusion_section)
        self.status = QLabel("Load a neuron Parquet and atlas, then choose a cohort.")
        self.status.setWordWrap(True)
        content.addWidget(self.status)
        self._configure_columns(False)
        self.build_button.setEnabled(False)
        self.cohort_combo.currentIndexChanged.connect(self.invalidate)
        self.split_laterality.toggled.connect(self.invalidate)
        self.exclusion_editor.rules_changed.connect(self.invalidate)
        self.section.expanded_changed.connect(self._section_toggled)

    def _configure_columns(self, split: bool) -> None:
        headers = list(REPORT_COLUMNS)
        if split:
            headers.insert(1, "Laterality")
        self.tree.setColumnCount(len(headers))
        self.tree.setHeaderLabels(headers)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Interactive)
        self.tree.setColumnWidth(0, 320)
        self.tree.sortByColumn(3 if split else 2, Qt.DescendingOrder)
        self.tree.headerItem().setToolTip(
            0,
            "Atlas region name and acronym. Sorting applies within anatomical siblings.",
        )

    def set_context(self, source_path: str | Path | None, atlas: object | None) -> None:
        """Invalidate snapshots when the source or loaded atlas changes."""
        path = None if source_path is None else str(source_path)
        if path == self._source_path and atlas is self._atlas:
            return
        self._source_path = path
        self._atlas = atlas
        self._collapsed_groups.clear()
        self.exclusion_editor.clear()
        self.build_button.setEnabled(path is not None and atlas is not None)
        self.refresh()

    def _section_toggled(self, expanded: bool) -> None:
        if expanded:
            self.refresh()
        else:
            self._timer.stop()

    def refresh(self, *_args) -> None:
        self._request_key = None
        self.invalidate()

    def invalidate(self, *_args) -> None:
        """Debounce cohort changes, ignoring color/visibility-only table edits."""
        if self._closing:
            return
        scope = self.cohort_combo.currentData()
        ids = tuple(sorted({str(value) for value in self._cohort_provider(scope)}))
        key = (
            self._source_path,
            id(self._atlas),
            ids,
            self.split_laterality.isChecked(),
            self.exclusion_editor.excluded_region_ids(),
        )
        if key == self._request_key:
            return
        self._request_key = key
        self._generation += 1
        if self._worker is not None:
            self._worker.cancel()
        self.tree.clear()
        if self._source_path is None or self._atlas is None:
            self.status.setText(
                "Load a neuron Parquet and compatible atlas for the report."
            )
            self._displayed_generation = self._generation
        elif not ids:
            self.status.setText(
                "Select neuron rows in Data to report on Selected rows."
                if scope == "selected"
                else "Add neurons to the Data table to generate a report."
            )
            self._displayed_generation = self._generation
        else:
            self.status.setText(f"Updating report for {len(ids):,} cohort neurons...")
            if self.section.is_expanded():
                self._timer.start()

    def _start_report(self) -> None:
        if self._closing or self._worker is not None or self._request_key is None:
            return
        source, _atlas_id, ids, split, exclusions = self._request_key
        if source is None or self._atlas is None or not ids:
            return
        if not default_region_profile_path(source).is_file():
            self._report_error(
                self._generation,
                "No regional profile is available. Click Build Regional Profile first.",
            )
            return
        thread = QThread()
        worker = RegionalReportWorker(
            self._generation, source, self._atlas, ids, split, exclusions
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.ready.connect(self._accept_report)
        worker.error.connect(self._report_error)
        worker.done.connect(worker.deleteLater)
        worker.done.connect(thread.quit)
        thread.finished.connect(lambda: self._thread_finished(thread))
        thread.finished.connect(thread.deleteLater)
        self._worker, self._thread = worker, thread
        _LIVE_REPORT_THREADS.add(thread)
        thread.start()

    def _thread_finished(self, thread) -> None:
        _LIVE_REPORT_THREADS.discard(thread)
        if self._thread is thread:
            self._worker = self._thread = None
        if (
            not self._closing
            and self.section.is_expanded()
            and self._displayed_generation != self._generation
        ):
            self._timer.start(0)

    def _report_error(self, generation: int, message: str) -> None:
        if not self._closing and generation == self._generation:
            self._displayed_generation = generation
            self.status.setText(f"Report unavailable: {message}")

    def _accept_report(self, generation: int, report: RegionReport) -> None:
        if self._closing or generation != self._generation:
            return
        self._displayed_generation = generation
        # Reuse the clustering picker, including an explicit unmapped bucket.
        structures = dict(self._atlas.structures)
        structures[0] = {
            "id": 0,
            "name": "Outside / unmapped",
            "acronym": "unmapped",
            "structure_id_path": [0],
        }
        picker_atlas = SimpleNamespace(
            atlas_name=getattr(self._atlas, "atlas_name", ""), structures=structures
        )
        self.exclusion_editor.set_atlas_and_allowed_ids(
            picker_atlas, set(report.available_region_ids)
        )
        self._populate_report(report)
        region_count = sum(bool(node.direct) for node in report.nodes)
        self.status.setText(
            f"Cohort: {report.cohort_count:,} neurons; "
            f"{region_count:,} direct regions with neurites after exclusions. "
            "Cable and termini include all non-soma neurites."
        )

    def _populate_report(self, report: RegionReport) -> None:
        self._populating = True
        self.tree.setSortingEnabled(False)
        self.tree.clear()
        self._configure_columns(report.split_laterality)
        sides = (
            REPORT_LATERALITIES if report.split_laterality else (COMBINED_LATERALITY,)
        )
        try:
            for side in sides:
                nodes = {
                    node.region_id: node
                    for node in report.nodes
                    if any(value.laterality == side for value in node.aggregate)
                }
                items = {}
                for region_id, node in nodes.items():
                    metrics = next(v for v in node.aggregate if v.laterality == side)
                    item = self._make_item(node.label, metrics, report.split_laterality)
                    item.group_key = (region_id, side)
                    item.setToolTip(
                        0, f"Atlas ID {region_id}. Collapse to aggregate descendants."
                    )
                    items[region_id] = item
                for region_id, node in nodes.items():
                    parent = items.get(node.parent_id)
                    if parent is None:
                        self.tree.addTopLevelItem(items[region_id])
                    else:
                        parent.addChild(items[region_id])
                for region_id, node in nodes.items():
                    item = items[region_id]
                    direct = next(
                        (v for v in node.direct if v.laterality == side), None
                    )
                    if item.childCount() and direct is not None:
                        item.addChild(
                            self._make_item(
                                f"{node.label} [direct]",
                                direct,
                                report.split_laterality,
                            )
                        )
                    item.setExpanded(item.group_key not in self._collapsed_groups)
                    self._update_group_cells(item)
        finally:
            self._populating = False
            self.tree.setSortingEnabled(True)

    def _make_item(self, label, metrics, split) -> _ReportItem:
        item = _ReportItem([label])
        item.metrics = metrics
        item.group_key = None
        item.split = split
        if split:
            item.setText(1, metrics.laterality.capitalize())
        self._update_group_cells(item)
        return item

    def _update_group_cells(self, item: _ReportItem) -> None:
        metrics: RegionReportMeasurements = item.metrics
        start = 2 if item.split else 1
        expanded_group = item.childCount() > 0 and item.isExpanded()
        for index, value in enumerate(metrics.values()):
            column = start + index
            item.setData(column, Qt.UserRole, value)
            text = f"{value:,}" if isinstance(value, int) else f"{value:,.6f}"
            if index in (1, 4):
                text = f"{value:.2f}"
            item.setText(column, "" if expanded_group else text)
            item.setTextAlignment(column, Qt.AlignRight | Qt.AlignVCenter)

    def _expanded(self, item) -> None:
        if not self._populating and item.group_key is not None:
            self._collapsed_groups.discard(item.group_key)
            self._update_group_cells(item)

    def _collapsed(self, item) -> None:
        if not self._populating and item.group_key is not None:
            self._collapsed_groups.add(item.group_key)
            self._update_group_cells(item)

    def _expand_all(self) -> None:
        self.tree.expandAll()

    def _collapse_all(self) -> None:
        self.tree.collapseAll()

    def shutdown(self) -> None:
        """Stop scheduling work and discard results after dock teardown."""
        self._closing = True
        self._timer.stop()
        if self._worker is not None:
            self._worker.cancel()
