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
    QMenu,
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

from ..analysis.search import (
    SEARCH_HEATMAP_MODE_SCORED,
    SEARCH_HEATMAP_MODE_WHOLE,
    SEARCH_RESULTS_FORMAT_VERSION,
    SEARCH_SCOPE_CURRENT,
    SEARCH_SCOPE_LABELS,
    SEARCH_SCOPE_SELECTED,
    SEARCH_SCOPE_WHOLE,
    SEARCH_SPACE_CCF,
    SEARCH_SPACE_FLATMAP,
    SEARCH_SPACE_LABELS,
    pearson_distance_color_domain,
)
from ..flatmap_heatmap import (
    DEFAULT_FLATMAP_DEPTH_BIN_UM,
    DEFAULT_FLATMAP_Y_BINS,
    FLATMAP_Y_BINS_TOOLTIP,
    MAX_FLATMAP_Y_BINS,
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
_FLATMAP_STYLE_LABELS = {
    "both_shaped": "Bilateral shaped",
    "both_square": "Bilateral square",
}
_FLATMAP_COORDS_INFO_TEXT = (
    "Flatmap search uses coordinates stored in the loaded Parquet. Region and "
    "node filters are applied before flatmap binning. X bins are derived from "
    "the selected style's aspect ratio so X/Y bins stay square."
)
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
    """Rank scoped candidates against one captured reference vector."""

    add_file_ids_requested = Signal(list)
    annotate_search_requested = Signal(object)
    apply_search_colors_requested = Signal(object)
    search_heatmaps_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db: NeuronDatabase | None = None
        self._atlas: BrainGlobeAtlas | None = None
        self._parquet_path: str | None = None
        self._catalog = pd.DataFrame(columns=["file_id", "neuron_id", "subject"])
        self._available_file_ids: set[str] = set()
        self._dataset_region_ids: set[int] = set()
        self._dataset_node_types: tuple[int, ...] = ()
        self._flatmap_available_styles: tuple[str, ...] = ()
        self._current_table_file_ids_provider = None
        self._selected_table_file_ids_provider = None
        self._captured_aggregate_file_ids: tuple[str, ...] = ()
        self._dendrite_coverage = None
        self._dendrite_filter_active = False
        self._worker_thread: QThread | None = None
        self._current_worker = None
        self._pending_request: VoxelSearchRequest | None = None
        self._pending_preflight = None
        self._last_result: VoxelSearchResult | None = None
        self._result_document = None
        self._result_frame = pd.DataFrame()
        self._pending_add_unavailable = 0
        self._heatmap_busy = False
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
        self._refresh_flatmap_coordinate_availability()
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

    def set_current_table_file_ids_provider(self, provider) -> None:
        """Set a callback returning every current Data-table ``file_id``."""
        self._current_table_file_ids_provider = provider

    def on_file_ids_added(
        self, summary, *, unavailable_count: int | None = None
    ) -> None:
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

    def on_search_annotated(
        self,
        append_summary,
        metadata_summary,
        *,
        unavailable_count: int = 0,
    ) -> None:
        """Display one parent-owned add-and-annotation transaction outcome."""
        self._status_label.setText(
            f"Added {int(getattr(append_summary, 'added_count', 0)):,}; "
            f"{int(getattr(append_summary, 'already_present_count', 0)):,} "
            f"already present; annotated "
            f"{int(getattr(metadata_summary, 'updated_count', 0)):,}; "
            f"{int(unavailable_count):,} unavailable."
        )

    def on_search_colors_applied(
        self,
        colored_count: int,
        missing_count: int,
    ) -> None:
        """Display the outcome of one Data-table Search color transfer."""
        message = (
            f"Applied Search colors to {int(colored_count):,} matching Data row(s)."
        )
        if missing_count:
            message += (
                f" {int(missing_count):,} Search neuron(s) were not in Data and "
                "were not added."
            )
        self._status_label.setText(message)

    def set_search_heatmap_busy(self, busy: bool) -> None:
        """Reflect the viewer-owned Search heatmap worker state."""
        self._heatmap_busy = bool(busy)
        self._update_button_states()

    def on_search_heatmap_progress(
        self,
        message: str,
        current: int,
        total: int,
    ) -> None:
        """Display viewer-owned Search heatmap progress in the Search tab."""
        self._status_label.setText(f"{message} ({int(current):,}/{int(total):,})")

    def on_search_heatmaps_finished(self, message: str) -> None:
        """Show the terminal status of a viewer-owned Search heatmap batch."""
        self._status_label.setText(str(message))
        self.set_search_heatmap_busy(False)

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
        self._soma_distance_enabled_cb = QCheckBox("Exclude nodes within soma distance")
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

        coordinate_row = QHBoxLayout()
        coordinate_row.addWidget(QLabel("Coordinate space:"))
        self._coordinate_space_combo = QComboBox()
        self._coordinate_space_combo.addItem(
            SEARCH_SPACE_LABELS[SEARCH_SPACE_CCF], SEARCH_SPACE_CCF
        )
        self._coordinate_space_combo.currentIndexChanged.connect(
            self._on_coordinate_space_changed
        )
        coordinate_row.addWidget(self._coordinate_space_combo)
        coordinate_row.addStretch()
        search_layout.addLayout(coordinate_row)

        self._flatmap_style_row = QWidget()
        flatmap_style_layout = QHBoxLayout(self._flatmap_style_row)
        flatmap_style_layout.setContentsMargins(0, 0, 0, 0)
        flatmap_style_layout.addWidget(QLabel("Flatmap style:"))
        self._flatmap_style_combo = QComboBox()
        flatmap_style_layout.addWidget(self._flatmap_style_combo)
        flatmap_style_layout.addStretch()
        search_layout.addWidget(self._flatmap_style_row)

        self._flatmap_y_bins_row = QWidget()
        flatmap_y_bins_layout = QHBoxLayout(self._flatmap_y_bins_row)
        flatmap_y_bins_layout.setContentsMargins(0, 0, 0, 0)
        flatmap_y_bins_label = QLabel("Y bins:")
        flatmap_y_bins_label.setToolTip(FLATMAP_Y_BINS_TOOLTIP)
        flatmap_y_bins_layout.addWidget(flatmap_y_bins_label)
        self._flatmap_y_bins_spin = QSpinBox()
        self._flatmap_y_bins_spin.setRange(2, MAX_FLATMAP_Y_BINS)
        self._flatmap_y_bins_spin.setValue(DEFAULT_FLATMAP_Y_BINS)
        self._flatmap_y_bins_spin.setToolTip(FLATMAP_Y_BINS_TOOLTIP)
        flatmap_y_bins_layout.addWidget(self._flatmap_y_bins_spin)
        flatmap_y_bins_layout.addStretch()
        search_layout.addWidget(self._flatmap_y_bins_row)

        self._flatmap_ignore_depth_cb = QCheckBox("Ignore depth (flat map X/Y only)")
        self._flatmap_ignore_depth_cb.toggled.connect(
            self._on_flatmap_ignore_depth_toggled
        )
        search_layout.addWidget(self._flatmap_ignore_depth_cb)

        self._flatmap_depth_bin_row = QWidget()
        flatmap_depth_bin_layout = QHBoxLayout(self._flatmap_depth_bin_row)
        flatmap_depth_bin_layout.setContentsMargins(0, 0, 0, 0)
        flatmap_depth_bin_layout.addWidget(QLabel("Depth bin (μm):"))
        self._flatmap_depth_bin_spin = QDoubleSpinBox()
        self._flatmap_depth_bin_spin.setRange(1.0, 10000.0)
        self._flatmap_depth_bin_spin.setDecimals(1)
        self._flatmap_depth_bin_spin.setValue(float(DEFAULT_FLATMAP_DEPTH_BIN_UM))
        self._flatmap_depth_bin_spin.setSuffix(" μm")
        flatmap_depth_bin_layout.addWidget(self._flatmap_depth_bin_spin)
        flatmap_depth_bin_layout.addStretch()
        search_layout.addWidget(self._flatmap_depth_bin_row)

        self._flatmap_include_depth_minus_one_cb = QCheckBox("Include depth -1 plane")
        self._flatmap_include_depth_minus_one_cb.setChecked(True)
        search_layout.addWidget(self._flatmap_include_depth_minus_one_cb)

        self._flatmap_coords_status_label = QLabel(_FLATMAP_COORDS_INFO_TEXT)
        self._flatmap_coords_status_label.setWordWrap(True)
        search_layout.addWidget(self._flatmap_coords_status_label)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Input neurons:"))
        self._scope_combo = QComboBox()
        for scope in (
            SEARCH_SCOPE_WHOLE,
            SEARCH_SCOPE_CURRENT,
            SEARCH_SCOPE_SELECTED,
        ):
            self._scope_combo.addItem(SEARCH_SCOPE_LABELS[scope], scope)
        self._scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        scope_row.addWidget(self._scope_combo)
        scope_row.addStretch()
        search_layout.addLayout(scope_row)
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

        cohort_row = QHBoxLayout()
        self._annotate_btn = QPushButton("Add & Annotate Search in Data")
        self._annotate_btn.clicked.connect(self._annotate_search_in_data)
        cohort_row.addWidget(self._annotate_btn)
        self._heatmap_btn = QPushButton("Add Search Heatmaps")
        heatmap_menu = QMenu(self._heatmap_btn)
        scored_menu = heatmap_menu.addMenu("Scored Voxels")
        self._heatmap_scored_selected_action = scored_menu.addAction(
            "Selected Results + Reference"
        )
        self._heatmap_scored_selected_action.triggered.connect(
            lambda _checked=False: self._add_selected_search_heatmaps(
                SEARCH_HEATMAP_MODE_SCORED
            )
        )
        self._heatmap_scored_all_action = scored_menu.addAction(
            "All Results + Reference"
        )
        self._heatmap_scored_all_action.triggered.connect(
            lambda _checked=False: self._add_all_search_heatmaps(
                SEARCH_HEATMAP_MODE_SCORED
            )
        )
        whole_menu = heatmap_menu.addMenu("Whole Neuron")
        self._heatmap_whole_selected_action = whole_menu.addAction(
            "Selected Results + Reference"
        )
        self._heatmap_whole_selected_action.triggered.connect(
            lambda _checked=False: self._add_selected_search_heatmaps(
                SEARCH_HEATMAP_MODE_WHOLE
            )
        )
        self._heatmap_whole_all_action = whole_menu.addAction("All Results + Reference")
        self._heatmap_whole_all_action.triggered.connect(
            lambda _checked=False: self._add_all_search_heatmaps(
                SEARCH_HEATMAP_MODE_WHOLE
            )
        )
        self._heatmap_btn.setMenu(heatmap_menu)
        cohort_row.addWidget(self._heatmap_btn)
        results_layout.addLayout(cohort_row)

        self._apply_colors_btn = QPushButton("Apply Search Colors to Data")
        self._apply_colors_btn.setToolTip(
            "Color matching Data rows with magenta references and the "
            "result-table Pearson-distance hot mapping. This works for both "
            "CCFv3 and flatmap Search results and does not add missing rows."
        )
        self._apply_colors_btn.clicked.connect(self._apply_search_colors_to_data)
        results_layout.addWidget(self._apply_colors_btn)

        self._heatmap_legend = QLabel(
            "Reference heatmaps are magenta. Result heatmap color uses the full "
            "result table: white/yellow = closest; dark red = farthest. Scored "
            "Voxels shows the filtered search input; Whole Neuron shows all "
            "valid, in-atlas source voxels. Use Apply Search Colors to Data to "
            "reuse this palette in Flatmap."
        )
        self._heatmap_legend.setWordWrap(True)
        results_layout.addWidget(self._heatmap_legend)

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
        self._update_coordinate_controls()
        self._update_voxel_filter_controls()
        self._update_button_states()

    # --- Coordinate space ------------------------------------------------------

    def _detect_flatmap_coordinates(self) -> tuple[bool, tuple[str, ...]]:
        """Return whether the loaded Parquet supports flatmap Search."""
        if not self._parquet_path:
            return False, ()
        try:
            from ..flatmap_parquet import read_flatmap_parquet_transform_info

            info = read_flatmap_parquet_transform_info(self._parquet_path)
            styles = tuple(info.available_styles)
            return bool(styles) and bool(info.has_v3_depth), styles
        except Exception:
            logger.debug(
                "Could not inspect Search flatmap coordinates in %s",
                self._parquet_path,
                exc_info=True,
            )
            return False, ()

    def _refresh_flatmap_coordinate_availability(self) -> None:
        """Offer flatmap space only when version-3 coordinates are available."""
        available, styles = self._detect_flatmap_coordinates()
        self._flatmap_available_styles = styles
        current = self._coordinate_space_combo.currentData()
        blocked = self._coordinate_space_combo.blockSignals(True)
        try:
            self._coordinate_space_combo.clear()
            self._coordinate_space_combo.addItem(
                SEARCH_SPACE_LABELS[SEARCH_SPACE_CCF], SEARCH_SPACE_CCF
            )
            if available:
                self._coordinate_space_combo.addItem(
                    SEARCH_SPACE_LABELS[SEARCH_SPACE_FLATMAP],
                    SEARCH_SPACE_FLATMAP,
                )
            index = self._coordinate_space_combo.findData(current)
            self._coordinate_space_combo.setCurrentIndex(max(0, index))
        finally:
            self._coordinate_space_combo.blockSignals(blocked)

        previous_style = self._flatmap_style_combo.currentData()
        blocked = self._flatmap_style_combo.blockSignals(True)
        try:
            self._flatmap_style_combo.clear()
            for style in styles:
                self._flatmap_style_combo.addItem(
                    _FLATMAP_STYLE_LABELS.get(style, style), style
                )
            index = self._flatmap_style_combo.findData(previous_style)
            if index >= 0:
                self._flatmap_style_combo.setCurrentIndex(index)
        finally:
            self._flatmap_style_combo.blockSignals(blocked)
        self._update_coordinate_controls()

    def _selected_coordinate_space(self) -> str:
        value = self._coordinate_space_combo.currentData()
        return str(value) if value in SEARCH_SPACE_LABELS else SEARCH_SPACE_CCF

    def _selected_flatmap_style(self) -> str | None:
        value = self._flatmap_style_combo.currentData()
        if value:
            return str(value)
        return (
            self._flatmap_available_styles[0]
            if self._flatmap_available_styles
            else None
        )

    def _on_coordinate_space_changed(self, _index: int | None = None) -> None:
        self._update_coordinate_controls()
        self._update_button_states()

    def _on_flatmap_ignore_depth_toggled(self, ignored: bool) -> None:
        self._flatmap_depth_bin_spin.setEnabled(not bool(ignored))

    def _update_coordinate_controls(self) -> None:
        is_flatmap = self._selected_coordinate_space() == SEARCH_SPACE_FLATMAP
        for widget in (
            self._flatmap_style_row,
            self._flatmap_y_bins_row,
            self._flatmap_ignore_depth_cb,
            self._flatmap_depth_bin_row,
            self._flatmap_include_depth_minus_one_cb,
            self._flatmap_coords_status_label,
        ):
            widget.setVisible(is_flatmap)
        self._on_flatmap_ignore_depth_toggled(self._flatmap_ignore_depth_cb.isChecked())

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

    def _current_table_file_ids(self) -> list[str]:
        provider = self._current_table_file_ids_provider
        if not callable(provider):
            return []
        try:
            values = provider() or []
        except Exception:
            logger.exception("Could not read the current Data-table rows")
            return []
        return list(dict.fromkeys(str(value) for value in values))

    def _selected_scope(self) -> str:
        scope = self._scope_combo.currentData()
        return str(scope) if scope in SEARCH_SCOPE_LABELS else SEARCH_SCOPE_WHOLE

    def _resolve_candidate_scope(
        self,
        references: tuple[str, ...],
        *,
        require_candidate: bool = True,
    ) -> tuple[tuple[str, ...] | None, str, int | None]:
        scope = self._selected_scope()
        if scope == SEARCH_SCOPE_WHOLE:
            candidates = tuple(sorted(self._available_file_ids))
            candidate_ids = None
            input_count = len(candidates)
        elif scope == SEARCH_SCOPE_CURRENT:
            candidates = tuple(self._current_table_file_ids())
            candidate_ids = candidates
            input_count = len(candidates)
            if not candidates:
                raise ValueError(
                    "Current Table is empty; switch to Whole Parquet or populate "
                    "the Data table first."
                )
        else:
            candidates = tuple(self._selected_table_file_ids())
            candidate_ids = candidates
            input_count = len(candidates)
            if not candidates:
                raise ValueError(
                    "No Data rows are selected; select at least one candidate "
                    "or change the Search input scope."
                )
        non_reference_count = len(set(candidates) - set(references))
        if require_candidate and non_reference_count < 1:
            raise ValueError(
                f"{SEARCH_SCOPE_LABELS[scope]} contains no non-reference candidate "
                "neurons."
            )
        label = SEARCH_SCOPE_LABELS[scope]
        if input_count is not None:
            label = (
                f"{label} ({input_count:,} rows; "
                f"{non_reference_count:,} non-reference candidates)"
            )
        return candidate_ids, label, input_count

    def _on_scope_changed(self, _index: int | None = None) -> None:
        self._clear_dendrite_restriction(
            "Input scope changed; rerun the dendrite-label scan if needed."
        )
        self._update_button_states()

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
        self._soma_distance_spin.setEnabled(self._soma_distance_enabled_cb.isChecked())
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

        try:
            file_ids, scope_label, _input_count = self._resolve_candidate_scope(
                self._reference_file_ids(),
                require_candidate=False,
            )
        except ValueError as error:
            self._status_label.setText(str(error))
            return
        self._clear_dendrite_restriction("Scanning complete neurons...")
        worker = DendriteCoverageWorker(
            parquet_path=self._parquet_path,
            file_ids=None if file_ids is None else list(file_ids),
            dendrite_node_types=dendrite_types,
        )
        self._status_label.setText(f"Scanning {scope_label} for dendrite labels...")
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
        candidate_ids, _scope_label, _input_count = self._resolve_candidate_scope(
            references
        )
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
        coordinate_space = self._selected_coordinate_space()
        flatmap_style = None
        if coordinate_space == SEARCH_SPACE_FLATMAP:
            flatmap_style = self._selected_flatmap_style()
            if flatmap_style is None:
                raise ValueError("No flatmap style is available in the loaded Parquet.")
        return VoxelSearchRequest(
            reference_file_ids=references,
            candidate_file_ids=candidate_ids,
            region_filter=region_filter,
            voxel_node_filter=voxel_filter,
            resolution_um=float(self._atlas.resolution[0]),
            top_n=int(self._top_n_spin.value()),
            exclude_references=True,
            candidate_scope=self._selected_scope(),
            coordinate_space=coordinate_space,
            flatmap_style=flatmap_style,
            flatmap_y_bins=int(self._flatmap_y_bins_spin.value()),
            flatmap_depth_bin_um=float(self._flatmap_depth_bin_spin.value()),
            flatmap_include_depth_minus_one=(
                self._flatmap_include_depth_minus_one_cb.isChecked()
            ),
            flatmap_collapse_depth=self._flatmap_ignore_depth_cb.isChecked(),
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
        input_count = (
            len(self._available_file_ids)
            if request.candidate_file_ids is None
            else len(request.candidate_file_ids)
        )
        non_reference_count = input_count - len(
            set(request.reference_file_ids)
            & (
                self._available_file_ids
                if request.candidate_file_ids is None
                else set(request.candidate_file_ids)
            )
        )
        scope_label = (
            f"{SEARCH_SCOPE_LABELS[request.candidate_scope]} "
            f"({input_count:,} rows; "
            f"{non_reference_count:,} non-reference candidates)"
        )
        space_label = SEARCH_SPACE_LABELS[request.coordinate_space]
        self._status_label.setText(
            f"Counting {space_label} search nodes in {scope_label}..."
        )
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
        from ..analysis.search import SearchResultsDocument

        self._last_result = result
        frame = result.hits.copy()
        frame["available"] = True
        references = result.reference_rows.copy()
        references.insert(0, "rank", 0)
        references["available"] = True
        self._result_document = SearchResultsDocument(
            references=references,
            hits=frame.copy(),
            metadata=dict(result.metadata),
            format_version=SEARCH_RESULTS_FORMAT_VERSION,
        )
        self._set_result_frame(frame)
        omitted = len(result.omitted_candidate_file_ids)
        self._status_label.setText(
            f"Reference neurons: {len(result.reference_file_ids):,}; scanned "
            f"{result.input_candidate_count:,} candidates; "
            f"{result.usable_candidate_count:,} usable; {omitted:,} omitted; "
            f"returned {len(result.hits):,}; retained "
            f"{result.retained_node_count:,} node rows in "
            f"{SEARCH_SPACE_LABELS.get(result.metadata.get('coordinate_space'), 'the selected space')} from "
            f"{result.metadata.get('candidate_scope_label', 'the selected scope')}. "
            "Lower distance is more similar."
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
        self._update_heatmap_legend()
        self._update_button_states()

    def _result_distance_domain(self) -> tuple[float, float] | None:
        """Return the observed Pearson-distance range in the result table."""
        if self._result_frame.empty or "pearson_distance" not in self._result_frame:
            return None
        try:
            return pearson_distance_color_domain(self._result_frame["pearson_distance"])
        except ValueError:
            return None

    def _update_heatmap_legend(self) -> None:
        """Describe the result-table range used for Search heatmap colors."""
        if (
            self._result_document is not None
            and self._result_document.metadata.get(
                "coordinate_space", SEARCH_SPACE_CCF
            )
            == SEARCH_SPACE_FLATMAP
        ):
            self._heatmap_legend.setText(
                "CCFv3 Search heatmaps are unavailable for flatmap-space "
                "results. Add the cohort to Data, use Apply Search Colors to "
                "Data, then render it from the Flatmap tab. References become "
                "magenta; results use the Pearson-distance hot mapping."
            )
            return
        distance_domain = self._result_distance_domain()
        if distance_domain is None:
            message = (
                "Reference heatmaps are magenta. Result heatmap color uses the "
                "full result table: white/yellow = closest; dark red = farthest. "
                "Scored Voxels shows the filtered search input; Whole Neuron "
                "shows all valid, in-atlas source voxels. Use Apply Search "
                "Colors to Data to reuse this palette in Flatmap."
            )
        else:
            distance_min, distance_max = distance_domain
            message = (
                "Reference heatmaps are magenta. Result heatmap color uses the "
                "full result-table range "
                f"{distance_min:.6g}–{distance_max:.6g}: white/yellow = closest; "
                "dark red = farthest. Scored Voxels shows the filtered search "
                "input; Whole Neuron shows all valid, in-atlas source voxels. "
                "Use Apply Search Colors to Data to reuse this palette in Flatmap."
            )
        self._heatmap_legend.setText(message)

    def _clear_results(self) -> None:
        self._last_result = None
        self._result_document = None
        self._result_frame = pd.DataFrame()
        self._results_table.clearContents()
        self._results_table.setRowCount(0)
        self._update_heatmap_legend()
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

    def _annotate_search_in_data(self) -> None:
        document = self._result_document
        if document is None or document.hits.empty:
            return
        from ..analysis.search import SearchAnnotationRequest

        references = document.references
        hits = document.hits
        unavailable: list[str] = []
        for frame in (references, hits):
            if "available" in frame:
                unavailable.extend(
                    frame.loc[~frame["available"].astype(bool), "file_id"]
                    .astype(str)
                    .tolist()
                )
        raw_tags = document.metadata.get("filter_tags", ())
        filter_tags = tuple(str(value) for value in raw_tags)
        if not filter_tags:
            filter_tags = ("Search filters: unavailable (version 1 CSV)",)
        request = SearchAnnotationRequest(
            reference_file_ids=tuple(references["file_id"].astype(str).tolist()),
            ranked_hits=tuple(
                (str(row.file_id), int(row.rank))
                for row in hits.sort_values("rank").itertuples(index=False)
            ),
            filter_tags=filter_tags,
            unavailable_file_ids=tuple(dict.fromkeys(unavailable)),
        )
        self.annotate_search_requested.emit(request)

    def _apply_search_colors_to_data(self) -> None:
        """Apply the completed result's distance palette to matching Data rows."""
        document = self._result_document
        if document is None or document.hits.empty:
            return
        distance_domain = self._result_distance_domain()
        if distance_domain is None:
            self._status_label.setText(
                "Cannot apply Search colors because the result table has no "
                "finite Pearson distances."
            )
            return

        from ..analysis.search import (
            SEARCH_REFERENCE_HEATMAP_RGBA,
            SearchTableColorRequest,
            pearson_distance_to_hot_rgba,
        )

        colors: list[tuple[str, tuple[float, float, float, float]]] = []
        seen: set[str] = set()
        reference_file_ids = tuple(
            document.references.get("file_id", pd.Series(dtype=str))
            .astype(str)
            .tolist()
        )
        for file_id in reference_file_ids:
            if file_id in seen:
                continue
            seen.add(file_id)
            colors.append((file_id, SEARCH_REFERENCE_HEATMAP_RGBA))
        for row in document.hits.sort_values("rank").itertuples(index=False):
            file_id = str(row.file_id)
            if file_id in seen:
                continue
            seen.add(file_id)
            colors.append(
                (
                    file_id,
                    pearson_distance_to_hot_rgba(
                        float(row.pearson_distance),
                        distance_domain=distance_domain,
                    ),
                )
            )
        self.apply_search_colors_requested.emit(
            SearchTableColorRequest(
                colors_by_file_id=tuple(colors),
                reference_file_ids=reference_file_ids,
                distance_color_domain=distance_domain,
            )
        )

    def _selected_result_file_ids(self) -> list[str]:
        file_ids: list[str] = []
        for row in self._selected_result_rows():
            item = self._results_table.item(row, 2)
            if item is not None:
                file_ids.append(str(item.data(Qt.UserRole) or item.text()))
        return file_ids

    def _add_selected_search_heatmaps(
        self,
        voxel_mode: str = SEARCH_HEATMAP_MODE_SCORED,
    ) -> None:
        selected = self._selected_result_file_ids()
        if not selected:
            self._status_label.setText(
                "Select at least one Search result row to create heatmaps."
            )
            return
        self._emit_search_heatmap_request(selected, voxel_mode=voxel_mode)

    def _add_all_search_heatmaps(
        self,
        voxel_mode: str = SEARCH_HEATMAP_MODE_SCORED,
    ) -> None:
        self._emit_search_heatmap_request(
            self._result_frame["file_id"].astype(str).tolist(),
            voxel_mode=voxel_mode,
        )

    def _emit_search_heatmap_request(
        self,
        file_ids: list[str],
        *,
        voxel_mode: str = SEARCH_HEATMAP_MODE_SCORED,
    ) -> None:
        document = self._result_document
        if document is None or self._atlas is None or self._heatmap_busy:
            return
        if document.metadata.get("coordinate_space", SEARCH_SPACE_CCF) != (
            SEARCH_SPACE_CCF
        ):
            self._status_label.setText(
                "Search heatmap actions currently render CCFv3 volumes and are "
                "unavailable for flatmap-space results. Use the Flatmap tab to "
                "inspect these neurons."
            )
            return
        if document.format_version < SEARCH_RESULTS_FORMAT_VERSION:
            self._status_label.setText(
                "Version 1 Search CSVs do not contain recoverable reference "
                "neurons; rerun the search first."
            )
            return
        from ..analysis.search import (
            SEARCH_REFERENCE_HEATMAP_RGBA,
            SEARCH_ROW_REFERENCE,
            SEARCH_ROW_RESULT,
            SearchHeatmapLayerRequest,
            SearchHeatmapRequest,
            cluster_region_filter_from_dict,
            pearson_distance_to_hot_rgba,
            voxel_node_filter_from_dict,
        )

        references = document.references
        if references.empty:
            self._status_label.setText(
                "This Search result has no recoverable reference neurons."
            )
            return
        requested = {str(value) for value in file_ids}
        hits = document.hits[
            document.hits["file_id"].astype(str).isin(requested)
        ].sort_values("rank")
        if hits.empty:
            self._status_label.setText("No Search result heatmaps were requested.")
            return
        requested_rows = pd.concat([references, hits], ignore_index=True)
        if "available" in requested_rows:
            unavailable = requested_rows.loc[
                ~requested_rows["available"].astype(bool), "file_id"
            ].astype(str)
            if not unavailable.empty:
                self._status_label.setText(
                    "Cannot create Search heatmaps because these file_id values "
                    "are unavailable: " + ", ".join(unavailable.tolist()[:10])
                )
                return
        metadata = dict(document.metadata)
        resolution = metadata.get("resolution_um")
        if resolution is None:
            self._status_label.setText(
                "This Search result lacks voxel-resolution context; rerun the search."
            )
            return
        current_resolution = float(self._atlas.resolution[0])
        try:
            saved_resolution = float(resolution)
        except (TypeError, ValueError):
            self._status_label.setText(
                "This Search result has invalid voxel-resolution context; "
                "rerun the search."
            )
            return
        if not np.isfinite(saved_resolution) or not np.isclose(
            saved_resolution,
            current_resolution,
        ):
            self._status_label.setText(
                "Cannot create Search heatmaps because the completed search "
                f"used {saved_resolution:g} μm voxels but the current atlas "
                f"uses {current_resolution:g} μm voxels."
            )
            return
        saved_atlas_name = metadata.get("atlas_name")
        current_atlas_name = getattr(self._atlas, "atlas_name", None)
        if (
            saved_atlas_name
            and current_atlas_name
            and str(saved_atlas_name) != str(current_atlas_name)
        ):
            self._status_label.setText(
                "Cannot create Search heatmaps because the completed search "
                f"used atlas {saved_atlas_name!s}, while the current atlas is "
                f"{current_atlas_name!s}."
            )
            return
        region_filter = None
        voxel_filter = None
        if voxel_mode == SEARCH_HEATMAP_MODE_SCORED:
            try:
                region_filter = cluster_region_filter_from_dict(
                    metadata.get("region_filter")
                )
                voxel_filter = voxel_node_filter_from_dict(
                    metadata.get("voxel_node_filter")
                )
            except (TypeError, ValueError, KeyError) as error:
                self._status_label.setText(
                    f"Could not reconstruct Search heatmap filters: {error}"
                )
                return

        distance_domain = self._result_distance_domain()
        if distance_domain is None:
            self._status_label.setText(
                "Cannot create Search heatmaps because the result table has no "
                "finite Pearson distances."
            )
            return
        metadata["heatmap_distance_color_domain"] = list(distance_domain)
        metadata["heatmap_distance_color_basis"] = "completed_result_table"
        metadata["heatmap_reference_color"] = "magenta"
        metadata["heatmap_reference_rgba"] = list(SEARCH_REFERENCE_HEATMAP_RGBA)
        metadata["heatmap_voxel_mode"] = voxel_mode
        metadata["heatmap_filters_applied"] = voxel_mode == SEARCH_HEATMAP_MODE_SCORED
        layers = [
            SearchHeatmapLayerRequest(
                file_ids=tuple(references["file_id"].astype(str).tolist()),
                role=SEARCH_ROW_REFERENCE,
                rank=0,
                pearson_distance=None,
                color=SEARCH_REFERENCE_HEATMAP_RGBA,
            )
        ]
        layers.extend(
            SearchHeatmapLayerRequest(
                file_ids=(str(row.file_id),),
                role=SEARCH_ROW_RESULT,
                rank=int(row.rank),
                pearson_distance=float(row.pearson_distance),
                color=pearson_distance_to_hot_rgba(
                    float(row.pearson_distance),
                    distance_domain=distance_domain,
                ),
            )
            for row in hits.itertuples(index=False)
        )
        request = SearchHeatmapRequest(
            layers=tuple(layers),
            region_filter=region_filter,
            voxel_node_filter=voxel_filter,
            resolution_um=saved_resolution,
            metadata=metadata,
            distance_color_domain=distance_domain,
            voxel_mode=voxel_mode,
        )
        self._heatmap_busy = True
        self._update_button_states()
        self.search_heatmaps_requested.emit(request)

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
            saved = export_search_results_csv(
                output_path,
                self._result_document or self._result_frame,
            )
        except Exception as error:  # noqa: BLE001 - report file-system/CSV failures in UI
            self._status_label.setText(f"Could not save search results: {error}")
            return
        self._status_label.setText(f"Saved search results to {saved.name}.")

    def _load_results_csv(self) -> None:
        if self._db is None:
            self._status_label.setText(
                "Load a neuron Parquet before importing results."
            )
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
            document = load_search_results_csv(
                input_path,
                available_file_ids=self._available_file_ids,
            )
        except Exception as error:  # noqa: BLE001 - report parser failures in UI
            self._status_label.setText(f"Could not load search results: {error}")
            return
        self._last_result = None
        self._result_document = document
        self._set_result_frame(document.hits)
        unavailable = sum(
            int((~frame["available"].astype(bool)).sum())
            for frame in (document.references, document.hits)
            if "available" in frame
        )
        self._status_label.setText(
            f"Loaded {len(document.references):,} reference(s) and "
            f"{len(document.hits):,} result(s) from {Path(input_path).name}; "
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
        self._annotate_btn.setEnabled(has_results and not busy)
        self._apply_colors_btn.setEnabled(has_results and not busy)
        ccf_heatmap_result = (
            self._result_document is not None
            and self._result_document.metadata.get("coordinate_space", SEARCH_SPACE_CCF)
            == SEARCH_SPACE_CCF
        )
        heatmap_ready = (
            has_results
            and not busy
            and not self._heatmap_busy
            and self._atlas is not None
            and self._result_document is not None
            and self._result_document.format_version >= SEARCH_RESULTS_FORMAT_VERSION
            and not self._result_document.references.empty
            and ccf_heatmap_result
        )
        self._heatmap_btn.setEnabled(heatmap_ready)
        self._heatmap_btn.setToolTip(
            ""
            if ccf_heatmap_result or not has_results
            else (
                "CCFv3 Search heatmaps are unavailable for flatmap-space "
                "results. Use Apply Search Colors to Data, then inspect the "
                "cohort from the Flatmap tab."
            )
        )
        for action in (
            self._heatmap_scored_all_action,
            self._heatmap_whole_all_action,
        ):
            action.setEnabled(heatmap_ready)
        selected_heatmap_ready = heatmap_ready and bool(self._selected_result_rows())
        for action in (
            self._heatmap_scored_selected_action,
            self._heatmap_whole_selected_action,
        ):
            action.setEnabled(selected_heatmap_ready)
        self._save_csv_btn.setEnabled(has_results and not busy)
        self._load_csv_btn.setEnabled(self._db is not None and not busy)
        self._reference_section.setEnabled(not busy)
        self._filter_section.setEnabled(not busy)
        self._scope_combo.setEnabled(not busy)
        self._coordinate_space_combo.setEnabled(not busy)
        self._flatmap_style_combo.setEnabled(not busy)
        self._flatmap_y_bins_spin.setEnabled(not busy)
        self._flatmap_ignore_depth_cb.setEnabled(not busy)
        self._flatmap_depth_bin_spin.setEnabled(
            not busy and not self._flatmap_ignore_depth_cb.isChecked()
        )
        self._flatmap_include_depth_minus_one_cb.setEnabled(not busy)
        self._top_n_spin.setEnabled(not busy)
