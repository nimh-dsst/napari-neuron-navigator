"""Search tab for ranking neurons by voxel-count Pearson distance."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .collapsible_section import CollapsibleSection
from .node_type_selector import NodeTypeSelectorComboBox, node_type_options
from .region_filter_editor import RegionFilterEditorWidget

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas

    from ..analysis.search import VoxelSearchRequest, VoxelSearchResult
    from ..db import NeuronDatabase

logger = logging.getLogger(__name__)

_REFERENCE_SINGLE = "single"
_REFERENCE_AGGREGATE = "aggregate"
_LARGE_SEARCH_NODE_THRESHOLD = 10_000_000
_VOXEL_NODE_TYPE_WARNING = (
    "Caution: node-type filtering trusts the Parquet type column. Confirm that "
    "dendrites are correctly labeled before interpreting axon-typed nodes as "
    "axons. Unlabelled or mislabelled dendrites may be type 0 or type 2."
)
_SOMA_DISTANCE_WARNING = (
    "Soma distance is a geometric proxy, not a compartment label. It can remove "
    "proximal axon and retain dendrites beyond the radius. Neurons without a "
    "valid soma are excluded."
)


class _NumericItem(QTableWidgetItem):
    """A table item that sorts by its numeric user-role value."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.data(Qt.UserRole)
        right = other.data(Qt.UserRole)
        if left is not None and right is not None:
            return float(left) < float(right)
        return super().__lt__(other)


class SearchTabWidget(QWidget):
    """Rank whole-Parquet candidates against one captured reference vector."""

    add_file_ids_requested = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db: NeuronDatabase | None = None
        self._atlas: BrainGlobeAtlas | None = None
        self._parquet_path: str | None = None
        self._catalog = pd.DataFrame(columns=["file_id", "neuron_id", "subject"])
        self._available_file_ids: set[str] = set()
        self._dataset_region_ids: set[int] = set()
        self._dataset_node_types: tuple[int, ...] = ()
        self._selected_table_file_ids_provider = None
        self._captured_aggregate_file_ids: tuple[str, ...] = ()
        self._dendrite_coverage = None
        self._dendrite_filter_active = False
        self._worker_thread: QThread | None = None
        self._current_worker = None
        self._pending_request: VoxelSearchRequest | None = None
        self._pending_preflight = None
        self._last_result: VoxelSearchResult | None = None
        self._result_frame = pd.DataFrame()
        self._pending_add_unavailable = 0
        self._setup_ui()

    # --- External dependencies -------------------------------------------------

    def set_database(self, db: NeuronDatabase) -> None:
        """Set the loaded Parquet database and clear stale search state."""
        self._db = db
        self._parquet_path = str(db.parquet_path)
        self._catalog = db.get_neuron_catalog().copy()
        self._catalog["file_id"] = self._catalog["file_id"].astype(str)
        self._available_file_ids = set(self._catalog["file_id"])
        regions = db.get_unique_regions()
        self._dataset_region_ids = {
            int(value)
            for value in regions.get("region_id", ())
            if value is not None
            and not (isinstance(value, (float, np.floating)) and np.isnan(value))
            and int(value) > 0
        }
        self._dataset_node_types = tuple(db.get_unique_node_types())
        self._region_filter_editor.clear()
        self._region_filter_editor.set_node_types(self._dataset_node_types)
        self._node_type_combo.set_options(node_type_options(self._dataset_node_types))
        self._clear_dendrite_restriction()
        self._captured_aggregate_file_ids = ()
        self._clear_results()
        self._populate_reference_catalog()
        self._refresh_region_editor()
        self._update_reference_summary()
        self._update_button_states()

    def set_atlas(self, atlas: BrainGlobeAtlas) -> None:
        """Set the atlas used for anatomical include/exclude masks."""
        self._atlas = atlas
        self._refresh_region_editor()
        self._update_button_states()

    def set_selected_table_file_ids_provider(self, provider) -> None:
        """Set a callback returning selected Data-table ``file_id`` values."""
        self._selected_table_file_ids_provider = provider

    def on_file_ids_added(self, summary, *, unavailable_count: int | None = None) -> None:
        """Display the outcome of a synchronous Data-table append request."""
        unavailable = (
            self._pending_add_unavailable
            if unavailable_count is None
            else int(unavailable_count)
        )
        added = int(getattr(summary, "added_count", 0))
        present = int(getattr(summary, "already_present_count", 0))
        self._status_label.setText(
            f"Added {added:,} neuron(s) to Data; {present:,} already present; "
            f"{unavailable:,} unavailable in the loaded Parquet."
        )
        self._pending_add_unavailable = 0

    # --- UI --------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        self._reference_section = CollapsibleSection("Reference Sample", expanded=True)
        reference_layout = self._reference_section.content_layout()
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Reference mode:"))
        self._reference_mode_combo = QComboBox()
        self._reference_mode_combo.addItem("Single neuron", _REFERENCE_SINGLE)
        self._reference_mode_combo.addItem(
            "Aggregate selected rows", _REFERENCE_AGGREGATE
        )
        self._reference_mode_combo.currentIndexChanged.connect(
            self._on_reference_mode_changed
        )
        mode_row.addWidget(self._reference_mode_combo)
        reference_layout.addLayout(mode_row)

        self._single_reference_row = QWidget()
        single_layout = QHBoxLayout(self._single_reference_row)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.addWidget(QLabel("Neuron:"))
        self._reference_combo = QComboBox()
        self._reference_combo.setEditable(True)
        self._reference_combo.setInsertPolicy(QComboBox.NoInsert)
        completer = QCompleter(self._reference_combo.model(), self._reference_combo)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        match_contains = getattr(Qt, "MatchContains", None)
        if match_contains is not None:
            completer.setFilterMode(match_contains)
        self._reference_combo.setCompleter(completer)
        self._reference_combo.currentIndexChanged.connect(
            self._update_reference_summary
        )
        single_layout.addWidget(self._reference_combo, 1)
        self._use_selected_single_btn = QPushButton("Use selected Data row")
        self._use_selected_single_btn.clicked.connect(self._capture_single_selection)
        single_layout.addWidget(self._use_selected_single_btn)
        reference_layout.addWidget(self._single_reference_row)

        self._aggregate_reference_row = QWidget()
        aggregate_layout = QHBoxLayout(self._aggregate_reference_row)
        aggregate_layout.setContentsMargins(0, 0, 0, 0)
        self._use_selected_aggregate_btn = QPushButton("Use selected Data rows")
        self._use_selected_aggregate_btn.clicked.connect(
            self._capture_aggregate_selection
        )
        aggregate_layout.addWidget(self._use_selected_aggregate_btn)
        aggregate_layout.addStretch()
        reference_layout.addWidget(self._aggregate_reference_row)

        self._reference_summary_label = QLabel("Load a Parquet to choose a reference.")
        self._reference_summary_label.setWordWrap(True)
        reference_layout.addWidget(self._reference_summary_label)
        aggregate_note = QLabel(
            "Aggregate mode sums the filtered voxel-count vectors before "
            "calculating Pearson distance."
        )
        aggregate_note.setWordWrap(True)
        reference_layout.addWidget(aggregate_note)
        layout.addWidget(self._reference_section)

        self._filter_section = CollapsibleSection("Search Filters", expanded=False)
        filter_layout = self._filter_section.content_layout()
        self._region_filter_editor = RegionFilterEditorWidget(node_types=())
        self._region_filter_editor.set_soma_mode(False)
        filter_layout.addWidget(self._region_filter_editor)

        node_row = QHBoxLayout()
        node_row.addWidget(QLabel("Node-type filter:"))
        self._node_type_mode_combo = QComboBox()
        self._node_type_mode_combo.addItem("Off", "all")
        self._node_type_mode_combo.addItem("Include selected", "include")
        self._node_type_mode_combo.addItem("Exclude selected", "exclude")
        self._node_type_mode_combo.currentIndexChanged.connect(
            self._update_voxel_filter_controls
        )
        node_row.addWidget(self._node_type_mode_combo)
        self._node_type_combo = NodeTypeSelectorComboBox(options=())
        node_row.addWidget(self._node_type_combo)
        filter_layout.addLayout(node_row)
        node_warning = QLabel(_VOXEL_NODE_TYPE_WARNING)
        node_warning.setWordWrap(True)
        node_warning.setStyleSheet("color: #d9822b; font-weight: bold;")
        filter_layout.addWidget(node_warning)

        coverage_row = QHBoxLayout()
        self._dendrite_scan_btn = QPushButton(
            "Find and exclude neurons lacking dendrite labels"
        )
        self._dendrite_scan_btn.clicked.connect(self._start_dendrite_scan)
        coverage_row.addWidget(self._dendrite_scan_btn)
        self._dendrite_clear_btn = QPushButton("Clear restriction")
        self._dendrite_clear_btn.clicked.connect(self._clear_dendrite_restriction)
        coverage_row.addWidget(self._dendrite_clear_btn)
        filter_layout.addLayout(coverage_row)
        self._dendrite_status_label = QLabel("No dendrite-label cohort restriction.")
        self._dendrite_status_label.setWordWrap(True)
        filter_layout.addWidget(self._dendrite_status_label)

        soma_row = QHBoxLayout()
        self._soma_distance_enabled_cb = QCheckBox(
            "Exclude nodes within soma distance"
        )
        self._soma_distance_enabled_cb.toggled.connect(
            self._update_voxel_filter_controls
        )
        soma_row.addWidget(self._soma_distance_enabled_cb)
        self._soma_distance_spin = QDoubleSpinBox()
        self._soma_distance_spin.setRange(0.0, 100000.0)
        self._soma_distance_spin.setDecimals(1)
        self._soma_distance_spin.setSingleStep(25.0)
        self._soma_distance_spin.setValue(200.0)
        self._soma_distance_spin.setSuffix(" μm")
        soma_row.addWidget(self._soma_distance_spin)
        filter_layout.addLayout(soma_row)
        soma_warning = QLabel(_SOMA_DISTANCE_WARNING)
        soma_warning.setWordWrap(True)
        filter_layout.addWidget(soma_warning)
        layout.addWidget(self._filter_section)

        self._search_section = CollapsibleSection("Search", expanded=True)
        search_layout = self._search_section.content_layout()
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Top results:"))
        self._top_n_spin = QSpinBox()
        self._top_n_spin.setRange(1, 100000)
        self._top_n_spin.setValue(100)
        top_row.addWidget(self._top_n_spin)
        top_row.addStretch()
        search_layout.addLayout(top_row)
        self._run_btn = QPushButton("Run Search")
        self._run_btn.clicked.connect(self._run_search)
        search_layout.addWidget(self._run_btn)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        search_layout.addWidget(self._progress_bar)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        search_layout.addWidget(self._status_label)
        layout.addWidget(self._search_section)

        self._results_section = CollapsibleSection("Results", expanded=True)
        results_layout = self._results_section.content_layout()
        self._results_table = QTableWidget(0, 4)
        self._results_table.setHorizontalHeaderLabels(
            ["Rank", "Neuron", "File ID", "Pearson distance (1 - r)"]
        )
        self._results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._results_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._results_table.setSortingEnabled(True)
        self._results_table.verticalHeader().setVisible(False)
        header = self._results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._results_table.itemSelectionChanged.connect(self._update_button_states)
        results_layout.addWidget(self._results_table)

        add_row = QHBoxLayout()
        self._add_selected_btn = QPushButton("Add Selected to Data")
        self._add_selected_btn.clicked.connect(self._add_selected_results)
        add_row.addWidget(self._add_selected_btn)
        self._add_all_btn = QPushButton("Add All Results to Data")
        self._add_all_btn.clicked.connect(self._add_all_results)
        add_row.addWidget(self._add_all_btn)
        results_layout.addLayout(add_row)

        file_row = QHBoxLayout()
        self._save_csv_btn = QPushButton("Save Results CSV...")
        self._save_csv_btn.clicked.connect(self._save_results_csv)
        file_row.addWidget(self._save_csv_btn)
        self._load_csv_btn = QPushButton("Load Results CSV...")
        self._load_csv_btn.clicked.connect(self._load_results_csv)
        file_row.addWidget(self._load_csv_btn)
        results_layout.addLayout(file_row)
        layout.addWidget(self._results_section)
        layout.addStretch()

        self._on_reference_mode_changed()
        self._update_voxel_filter_controls()
        self._update_button_states()

    # --- Reference and filters -------------------------------------------------

    def _populate_reference_catalog(self) -> None:
        previous = self._reference_combo.currentData()
        blocked = self._reference_combo.blockSignals(True)
        try:
            self._reference_combo.clear()
            for row in self._catalog.itertuples(index=False):
                file_id = str(row.file_id)
                subject = str(row.subject or "").strip()
                neuron_id = str(row.neuron_id or "").strip()
                identity = " / ".join(value for value in (subject, neuron_id) if value)
                label = f"{identity} — {file_id}" if identity else file_id
                self._reference_combo.addItem(label, file_id)
            if previous is not None:
                index = self._reference_combo.findData(str(previous))
                if index >= 0:
                    self._reference_combo.setCurrentIndex(index)
        finally:
            self._reference_combo.blockSignals(blocked)

    def _selected_table_file_ids(self) -> list[str]:
        provider = self._selected_table_file_ids_provider
        if not callable(provider):
            return []
        try:
            values = provider() or []
        except Exception:
            logger.exception("Could not read the selected Data-table rows")
            return []
        return list(dict.fromkeys(str(value) for value in values))

    def _capture_single_selection(self) -> None:
        selected = self._selected_table_file_ids()
        if len(selected) != 1:
            self._status_label.setText(
                "Select exactly one visible Data-table row for a single reference."
            )
            return
        index = self._reference_combo.findData(selected[0])
        if index < 0:
            self._status_label.setText(
                f"Selected file_id {selected[0]!r} is not in the loaded Parquet."
            )
            return
        self._reference_combo.setCurrentIndex(index)
        self._update_reference_summary()

    def _capture_aggregate_selection(self) -> None:
        selected = self._selected_table_file_ids()
        if len(selected) < 2:
            self._status_label.setText(
                "Select at least two visible Data-table rows for an aggregate reference."
            )
            return
        missing = [value for value in selected if value not in self._available_file_ids]
        if missing:
            self._status_label.setText(
                "Selected reference file_id value(s) are unavailable: "
                + ", ".join(missing)
            )
            return
        self._captured_aggregate_file_ids = tuple(selected)
        self._update_reference_summary()

    def _on_reference_mode_changed(self, _index: int | None = None) -> None:
        single = self._reference_mode_combo.currentData() == _REFERENCE_SINGLE
        self._single_reference_row.setVisible(single)
        self._aggregate_reference_row.setVisible(not single)
        self._update_reference_summary()
        self._update_button_states()

    def _reference_file_ids(self) -> tuple[str, ...]:
        if self._reference_mode_combo.currentData() == _REFERENCE_AGGREGATE:
            return self._captured_aggregate_file_ids
        value = self._reference_combo.currentData()
        return () if value is None else (str(value),)

    def _catalog_display(self, file_id: str) -> str:
        matched = self._catalog[self._catalog["file_id"] == str(file_id)]
        if matched.empty:
            return str(file_id)
        row = matched.iloc[0]
        values = [
            str(row.get("subject", "") or "").strip(),
            str(row.get("neuron_id", "") or "").strip(),
        ]
        identity = " / ".join(value for value in values if value)
        return f"{identity} (file_id: {file_id})" if identity else f"file_id: {file_id}"

    def _update_reference_summary(self, _index: int | None = None) -> None:
        references = self._reference_file_ids()
        if not references:
            message = (
                "Capture at least two selected Data rows."
                if self._reference_mode_combo.currentData() == _REFERENCE_AGGREGATE
                else "Choose a neuron from the loaded Parquet."
            )
        elif len(references) == 1:
            message = "Reference: " + self._catalog_display(references[0])
        else:
            message = f"Aggregate: {len(references):,} captured neurons"
        self._reference_summary_label.setText(message)
        self._update_button_states()

    def _refresh_region_editor(self) -> None:
        if self._atlas is None:
            return
        self._region_filter_editor.set_atlas_and_allowed_ids(
            self._atlas,
            self._dataset_region_ids,
        )

    def _represented_region_entries(self, region_ids: list[int]):
        if self._atlas is None:
            return []
        represented: list[tuple[int, str]] = []
        seen: set[int] = set()
        for selected_id in region_ids:
            for candidate_id in sorted(self._dataset_region_ids):
                struct = self._atlas.structures.get(int(candidate_id))
                if struct is None:
                    continue
                try:
                    path = [
                        int(value)
                        for value in struct.get("structure_id_path", []) or []
                    ]
                except (TypeError, ValueError):
                    continue
                if int(selected_id) not in path or candidate_id in seen:
                    continue
                acronym = str(struct.get("acronym", "")).strip()
                if acronym:
                    seen.add(candidate_id)
                    represented.append((candidate_id, acronym))
        return represented

    def _available_dendrite_node_types(self) -> tuple[int, ...]:
        represented = set(self._dataset_node_types)
        return tuple(value for value in (3, 4) if value in represented)

    def _update_voxel_filter_controls(self, _value=None) -> None:
        mode = str(self._node_type_mode_combo.currentData() or "all")
        self._node_type_combo.setEnabled(mode != "all")
        self._soma_distance_spin.setEnabled(
            self._soma_distance_enabled_cb.isChecked()
        )
        self._dendrite_clear_btn.setEnabled(self._dendrite_filter_active)

    def _selected_voxel_node_filter(self):
        from ..analysis.voxel_filter import VoxelNodeFilter

        mode = str(self._node_type_mode_combo.currentData() or "all")
        selected_types: tuple[int, ...] = ()
        if mode != "all":
            selected = self._node_type_combo.selected_node_types()
            if not selected:
                raise ValueError(
                    "Select at least one specific node type or turn the "
                    "node-type filter off."
                )
            selected_types = tuple(int(value) for value in selected)
        distance = (
            float(self._soma_distance_spin.value())
            if self._soma_distance_enabled_cb.isChecked()
            else None
        )
        dendrite_types = self._available_dendrite_node_types()
        settings = VoxelNodeFilter(
            node_type_mode=mode,
            node_types=selected_types,
            require_dendrite_labels=self._dendrite_filter_active,
            exclude_within_soma_um=distance,
            dendrite_node_types=dendrite_types or (3, 4),
        )
        return None if settings.is_empty else settings

    def _clear_dendrite_restriction(
        self,
        message: str = "No dendrite-label cohort restriction.",
    ) -> None:
        self._dendrite_coverage = None
        self._dendrite_filter_active = False
        self._dendrite_status_label.setText(message)
        self._update_voxel_filter_controls()

    # --- Background work -------------------------------------------------------

    def _is_busy(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.isRunning()

    def _start_thread(self, worker, finished_slot, thread_finished_slot) -> None:
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(finished_slot)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread_finished_slot)
        thread.finished.connect(thread.deleteLater)
        self._worker_thread = thread
        self._current_worker = worker
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._update_button_states()
        thread.start()

    def _release_thread(self) -> None:
        self._worker_thread = None
        self._current_worker = None
        self._progress_bar.setVisible(False)
        self._update_button_states()

    def _on_progress(self, message: str, current: int, total: int) -> None:
        self._status_label.setText(message)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, max(1, int(total)))
        self._progress_bar.setValue(int(current))

    def _on_error(self, message: str) -> None:
        self._pending_request = None
        self._pending_preflight = None
        self._status_label.setText(f"Error: {message}")
        self._progress_bar.setVisible(False)
        logger.error("Search error: %s", message)

    def _start_dendrite_scan(self) -> None:
        if self._parquet_path is None or self._is_busy():
            return
        dendrite_types = self._available_dendrite_node_types()
        if not dendrite_types:
            self._clear_dendrite_restriction(
                "The loaded Parquet contains no type 3/4 dendrite labels; no "
                "restriction was applied."
            )
            return
        from ..workers import DendriteCoverageWorker

        self._clear_dendrite_restriction("Scanning complete neurons...")
        worker = DendriteCoverageWorker(
            parquet_path=self._parquet_path,
            file_ids=None,
            dendrite_node_types=dendrite_types,
        )
        self._start_thread(
            worker,
            self._on_dendrite_scan_finished,
            self._on_dendrite_thread_finished,
        )

    def _on_dendrite_scan_finished(self, coverage) -> None:
        self._dendrite_coverage = coverage
        self._dendrite_filter_active = coverage.labeled_neuron_count > 0
        if coverage.input_neuron_count == 0:
            message = "No neurons were available in the loaded Parquet."
        elif coverage.labeled_neuron_count == 0:
            message = (
                f"0 of {coverage.input_neuron_count:,} neurons contain type 3/4 "
                "labels; no restriction was applied."
            )
        else:
            message = (
                f"Dendrite-label restriction active: "
                f"{coverage.labeled_neuron_count:,} of "
                f"{coverage.input_neuron_count:,} neurons retained; "
                f"{coverage.excluded_neuron_count:,} excluded. Label presence "
                "does not prove label correctness."
            )
        self._dendrite_status_label.setText(message)
        self._status_label.setText(message)
        self._update_voxel_filter_controls()

    def _on_dendrite_thread_finished(self) -> None:
        self._release_thread()

    def _build_request(self):
        from ..analysis.search import VoxelSearchRequest

        references = self._reference_file_ids()
        mode = self._reference_mode_combo.currentData()
        if mode == _REFERENCE_AGGREGATE and len(references) < 2:
            raise ValueError("Capture at least two Data rows for aggregate mode.")
        if mode == _REFERENCE_SINGLE and len(references) != 1:
            raise ValueError("Choose exactly one reference neuron.")
        region_filter = self._region_filter_editor.region_filter(
            self._represented_region_entries
        )
        if region_filter is not None:
            empty_rules = [
                rule.acronym
                for rule in (*region_filter.include_rules, *region_filter.exclude_rules)
                if not rule.represented_region_ids
            ]
            if empty_rules:
                raise ValueError(
                    "Selected region(s) have no represented dataset regions: "
                    + ", ".join(empty_rules)
                )
        voxel_filter = self._selected_voxel_node_filter()
        return VoxelSearchRequest(
            reference_file_ids=references,
            candidate_file_ids=None,
            region_filter=region_filter,
            voxel_node_filter=voxel_filter,
            resolution_um=float(self._atlas.resolution[0]),
            top_n=int(self._top_n_spin.value()),
            exclude_references=True,
        )

    def _run_search(self) -> None:
        if self._db is None or self._atlas is None or self._is_busy():
            return
        try:
            request = self._build_request()
        except ValueError as error:
            self._status_label.setText(str(error))
            return
        from ..workers import SearchPreflightWorker

        self._pending_request = request
        self._pending_preflight = None
        worker = SearchPreflightWorker(
            parquet_path=self._parquet_path,
            atlas=self._atlas,
            request=request,
        )
        self._status_label.setText("Counting search nodes...")
        self._start_thread(
            worker,
            self._on_preflight_finished,
            self._on_preflight_thread_finished,
        )

    def _on_preflight_finished(self, result) -> None:
        self._pending_preflight = result

    def _on_preflight_thread_finished(self) -> None:
        request = self._pending_request
        preflight = self._pending_preflight
        self._release_thread()
        if request is None or preflight is None:
            return
        if preflight.node_count > _LARGE_SEARCH_NODE_THRESHOLD:
            prompt = QMessageBox(self)
            prompt.setIcon(QMessageBox.Warning)
            prompt.setWindowTitle("Large Similarity Search")
            prompt.setText(
                f"This search will process {preflight.node_count:,} nodes. Continue?"
            )
            continue_button = prompt.addButton("Continue", QMessageBox.AcceptRole)
            cancel_button = prompt.addButton("Cancel", QMessageBox.RejectRole)
            prompt.setDefaultButton(cancel_button)
            prompt.exec()
            if prompt.clickedButton() is not continue_button:
                self._pending_request = None
                self._pending_preflight = None
                self._status_label.setText(
                    f"Search cancelled; {preflight.node_count:,} nodes would "
                    "have been processed."
                )
                return
        self._launch_search(request, preflight)

    def _launch_search(self, request, preflight) -> None:
        from ..workers import SearchWorker

        worker = SearchWorker(
            parquet_path=self._parquet_path,
            atlas=self._atlas,
            request=request,
            prepared_region_filter=preflight.prepared_region_filter,
            prepared_voxel_filter=preflight.prepared_voxel_filter,
        )
        self._status_label.setText("Searching for similar neurons...")
        self._start_thread(
            worker,
            self._on_search_finished,
            self._on_search_thread_finished,
        )

    def _on_search_finished(self, result) -> None:
        self._last_result = result
        frame = result.hits.copy()
        frame["available"] = True
        self._set_result_frame(frame)
        omitted = len(result.omitted_candidate_file_ids)
        self._status_label.setText(
            f"Reference neurons: {len(result.reference_file_ids):,}; scanned "
            f"{result.input_candidate_count:,} candidates; "
            f"{result.usable_candidate_count:,} usable; {omitted:,} omitted; "
            f"returned {len(result.hits):,}; retained "
            f"{result.retained_node_count:,} node rows. Lower distance is more similar."
        )

    def _on_search_thread_finished(self) -> None:
        self._pending_request = None
        self._pending_preflight = None
        self._release_thread()

    # --- Results and CSV -------------------------------------------------------

    def _set_result_frame(self, frame: pd.DataFrame) -> None:
        self._result_frame = frame.reset_index(drop=True).copy()
        table = self._results_table
        sorting = table.isSortingEnabled()
        table.setSortingEnabled(False)
        table.clearContents()
        table.setRowCount(len(frame))
        unavailable_brush = QBrush(QColor("gray"))
        for row_index, row in enumerate(frame.itertuples(index=False)):
            rank = int(row.rank)
            file_id = str(row.file_id)
            neuron_id = str(row.neuron_id or "").strip()
            subject = str(row.subject or "").strip()
            display = " / ".join(value for value in (subject, neuron_id) if value)
            if not display:
                display = file_id
            available = bool(getattr(row, "available", True))

            rank_item = _NumericItem(str(rank))
            rank_item.setData(Qt.UserRole, rank)
            name_item = QTableWidgetItem(display)
            file_item = QTableWidgetItem(file_id)
            file_item.setData(Qt.UserRole, file_id)
            distance = float(row.pearson_distance)
            distance_item = _NumericItem(f"{distance:.8g}")
            distance_item.setData(Qt.UserRole, distance)
            if not available:
                for item in (rank_item, name_item, file_item, distance_item):
                    item.setForeground(unavailable_brush)
                    item.setToolTip("Unavailable in the currently loaded Parquet")
            table.setItem(row_index, 0, rank_item)
            table.setItem(row_index, 1, name_item)
            table.setItem(row_index, 2, file_item)
            table.setItem(row_index, 3, distance_item)
        table.setSortingEnabled(sorting)
        table.resizeRowsToContents()
        self._update_button_states()

    def _clear_results(self) -> None:
        self._last_result = None
        self._result_frame = pd.DataFrame()
        self._results_table.clearContents()
        self._results_table.setRowCount(0)
        self._update_button_states()

    def _selected_result_rows(self) -> list[int]:
        return sorted({index.row() for index in self._results_table.selectedIndexes()})

    def _emit_add_request(self, file_ids: list[str]) -> None:
        if self._result_frame.empty:
            return
        requested = {str(value) for value in file_ids}
        subset = self._result_frame[
            self._result_frame["file_id"].astype(str).isin(requested)
        ]
        availability = (
            subset["available"].astype(bool)
            if "available" in subset
            else pd.Series(True, index=subset.index)
        )
        file_ids = subset.loc[availability, "file_id"].astype(str).tolist()
        self._pending_add_unavailable = int((~availability).sum())
        if not file_ids:
            self._status_label.setText(
                f"No available neurons to add; {self._pending_add_unavailable:,} "
                "selected result(s) are unavailable."
            )
            return
        self.add_file_ids_requested.emit(file_ids)

    def _add_selected_results(self) -> None:
        rows = self._selected_result_rows()
        if not rows:
            self._status_label.setText("Select at least one result row to add.")
            return
        file_ids: list[str] = []
        for row in rows:
            item = self._results_table.item(row, 2)
            if item is not None:
                file_ids.append(str(item.data(Qt.UserRole) or item.text()))
        self._emit_add_request(file_ids)

    def _add_all_results(self) -> None:
        self._emit_add_request(self._result_frame["file_id"].astype(str).tolist())

    def _save_results_csv(self) -> None:
        if self._result_frame.empty:
            return
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Search Results",
            "similar_neurons.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not output_path:
            return
        from ..analysis.search import export_search_results_csv

        try:
            saved = export_search_results_csv(output_path, self._result_frame)
        except Exception as error:  # noqa: BLE001 - report file-system/CSV failures in UI
            self._status_label.setText(f"Could not save search results: {error}")
            return
        self._status_label.setText(f"Saved search results to {saved.name}.")

    def _load_results_csv(self) -> None:
        if self._db is None:
            self._status_label.setText("Load a neuron Parquet before importing results.")
            return
        input_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Search Results",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not input_path:
            return
        from ..analysis.search import load_search_results_csv

        try:
            frame = load_search_results_csv(
                input_path,
                available_file_ids=self._available_file_ids,
            )
        except Exception as error:  # noqa: BLE001 - report parser failures in UI
            self._status_label.setText(f"Could not load search results: {error}")
            return
        self._last_result = None
        self._set_result_frame(frame)
        unavailable = int((~frame["available"].astype(bool)).sum())
        self._status_label.setText(
            f"Loaded {len(frame):,} result(s) from {Path(input_path).name}; "
            f"{unavailable:,} unavailable in the loaded Parquet."
        )

    def _update_button_states(self) -> None:
        busy = self._is_busy()
        ready = self._db is not None and self._atlas is not None
        references = self._reference_file_ids()
        reference_ready = (
            len(references) >= 2
            if self._reference_mode_combo.currentData() == _REFERENCE_AGGREGATE
            else len(references) == 1
        )
        self._run_btn.setEnabled(ready and reference_ready and not busy)
        self._dendrite_scan_btn.setEnabled(
            self._db is not None
            and bool(self._available_dendrite_node_types())
            and not busy
        )
        has_results = not self._result_frame.empty
        self._add_all_btn.setEnabled(has_results and not busy)
        self._add_selected_btn.setEnabled(
            has_results and bool(self._selected_result_rows()) and not busy
        )
        self._save_csv_btn.setEnabled(has_results and not busy)
        self._load_csv_btn.setEnabled(self._db is not None and not busy)
        self._reference_section.setEnabled(not busy)
        self._filter_section.setEnabled(not busy)
        self._top_n_spin.setEnabled(not busy)
