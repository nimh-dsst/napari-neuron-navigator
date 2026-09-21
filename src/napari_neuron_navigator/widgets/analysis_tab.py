"""Analysis tab widget for the clustering pipeline.

Provides UI for:
1. Region mask selection and dilation parameters
2. Correlation matrix computation with hierarchical clustering
3. Clustermap visualization (embedded matplotlib canvas)
4. Node count heatmap generation
5. Applying neuron colors by cluster assignment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QThread, Signal
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..analysis.flatmap_correlation import (
    DEFAULT_FLATMAP_DEPTH_SCALE,
    DEFAULT_FLATMAP_SOMA_DBSCAN_EPS,
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
from .region_selector import RegionSelectorWidget

if TYPE_CHECKING:
    import napari
    from brainglobe_atlasapi import BrainGlobeAtlas

    from ..analysis.clustering import (
        ClusterRegionFilter,
        ClusterRegionSelection,
        ClusterResult,
    )
    from ..analysis.voxel_filter import DendriteLabelCoverage, VoxelNodeFilter
    from ..db import NeuronDatabase

logger = logging.getLogger(__name__)

_ANALYSIS_SCOPE_WHOLE = "whole"
_ANALYSIS_SCOPE_CURRENT = "current"
_ANALYSIS_SCOPE_SELECTED = "selected"
_COORD_SPACE_CCF = "CCFv3 Coordinates"
_COORD_SPACE_FLATMAP = "Flat map + Depth"
_CLUSTER_METHOD_VOXEL = "Voxel Correlation"
_CLUSTER_METHOD_SOMA = "Soma Location"
_LARGE_CLUSTER_NODE_THRESHOLD = 10_000_000
_FLATMAP_STYLE_LABELS = {
    "both_shaped": "Bilateral shaped",
    "both_square": "Bilateral square",
}
_FLATMAP_COORDS_INFO_TEXT = (
    "Clustering uses flatmap/depth coordinates in the loaded Parquet. "
    "Region filters use each node's original CCFv3 coordinates before "
    "flatmap binning or soma clustering."
)
_DEPTH_SCALE_TOOLTIP = (
    "Weight of cortical depth relative to flat map position.\n\n"
    "Both flat map axes are divided by the same number, so flat map space is "
    "scaled but never distorted: 1.0 is one hemisphere height in any "
    "direction. One full cortical thickness spans 1.0 in depth.\n\n"
    "Higher values weight depth MORE, pulling clusters toward cortical layers "
    "(laminar grouping).\n"
    "Lower values weight depth LESS, pulling clusters toward flat map "
    "position (areal grouping).\n\n"
    "1.0 treats a full cortical thickness of depth separation as equivalent "
    "to one hemisphere height of tangential separation.\n\n"
    "These are fractions of each axis, not physical distances. The flat map "
    "projection distorts the cortical surface, so flat map X/Y have no "
    "reliable conversion to microns and this weight cannot be read as a "
    "physical ratio."
)
_FLATMAP_EPS_TOOLTIP = (
    "DBSCAN neighbourhood radius in normalized flat map units, where 1.0 is "
    "one hemisphere height. The radius is circular: both flat map axes share "
    "one divisor, so the same value means the same distance in any direction."
)
_IGNORE_DEPTH_TOOLTIP = (
    "Cluster on flat map X/Y only, dropping the depth axis entirely.\n\n"
    "Groups neurons by where they sit across the cortical sheet regardless of "
    "which layer they occupy.\n\n"
    "Soma Location: drops depth from the coordinates, leaving a 2-D distance.\n"
    "Voxel Correlation: collapses the voxel grid's depth planes, so nodes at "
    "one flat map position share a voxel whatever their depth. Depth still "
    "decides which nodes are counted, so the node count does not change and "
    "Include depth -1 plane keeps its meaning."
)
_VOXEL_NODE_TYPE_WARNING = (
    "Caution: node-type filtering trusts the Parquet type column. Confirm that "
    "dendrites are correctly labeled in every neuron before interpreting "
    "axon-typed nodes as axons. Unlabelled or mislabelled dendrites may be type "
    "0 or type 2. A coverage scan only detects whether type 3/4 labels exist; "
    "it does not prove that every dendrite is labeled correctly."
)
_SOMA_DISTANCE_WARNING = (
    "Soma distance is a geometric proxy, not a compartment label. It can remove "
    "proximal axon and retain dendrites that extend beyond the selected radius. "
    "Neurons without a valid soma are excluded."
)


@dataclass(frozen=True)
class _HeatmapRequest:
    """Immutable specification for one analysis heatmap build."""

    selected_region_id: int | None
    selected_region_acronym: str | None
    region_ids: tuple[int, ...] | None
    cluster_label: int | None
    file_ids: tuple[str, ...] | None
    node_types: tuple[int, ...] | None
    soma_radius_um: float | None
    depth_bin_factor: int
    depth_axis: int
    assignment_id: str | None = None
    assignment_name: str | None = None


@dataclass(frozen=True)
class _ClusteringRequest:
    """Immutable snapshot of every setting used by one clustering run."""

    coordinate_space: str
    clustering_method: str
    scope: str
    file_ids: tuple[str, ...] | None
    region_selection: ClusterRegionSelection | None
    region_filter: ClusterRegionFilter | None
    dilation_fraction: float
    linkage_method: str
    n_clusters: int
    algorithm: str
    eps: float
    min_samples: int
    flatmap_style: str | None
    flatmap_y_bins: int
    flatmap_depth_bin_um: float
    flatmap_include_depth_minus_one: bool
    flatmap_depth_scale: float = DEFAULT_FLATMAP_DEPTH_SCALE
    flatmap_include_depth: bool = True
    voxel_node_filter: VoxelNodeFilter | None = None


@dataclass(frozen=True)
class _ClusterColorApplicationSummary:
    """Summary of cached cluster application to the table and rendered layers."""

    matched_table_count: int
    updated_layer_count: int
    rendered_count: int
    colored_count: int
    gray_count: int


def _analysis_heatmap_contrast_limits(
    volume: np.ndarray,
) -> tuple[float, float]:
    """Return stable full-volume contrast limits for an analysis heatmap."""
    if volume.size == 0:
        return (0.0, 1.0)

    upper = float(np.nanmax(volume))
    if not np.isfinite(upper) or upper <= 0.0:
        return (0.0, 1.0)
    return (0.0, upper)


def _large_clustering_warning_text(node_count: int) -> str:
    """Return the confirmation text for an oversized clustering input."""
    return (
        f"This clustering run will process {int(node_count):,} nodes, "
        "which exceeds the 10,000,000-node warning threshold. Continue?"
    )


def _populate_embedded_clustermap_figure(
    figure: Figure,
    result: ClusterResult,
    cluster_color_map: dict[str, list[float]] | None = None,
    *,
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int | None = None,
) -> Figure:
    """Populate the persistent dendrogram preview figure used by the tab."""
    from ..analysis.export import populate_clustermap_figure

    return populate_clustermap_figure(
        figure,
        result,
        cluster_color_map,
        figsize=figsize,
        dpi=dpi,
    )


def _analysis_heatmap_stored_limits(
    layer: Any,
) -> tuple[float, float] | None:
    """Return stored contrast limits from analysis heatmap metadata."""
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return None

    raw_limits = metadata.get("heatmap_contrast_limits")
    if raw_limits is None:
        return None

    try:
        values = np.asarray(raw_limits, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if values.size < 2:
        return None

    lower = float(values[0])
    upper = float(values[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None
    return (lower, upper)


def _apply_analysis_heatmap_contrast_limits(
    layer: Any,
    limits: tuple[float, float],
) -> None:
    """Apply stored full-volume contrast limits to an analysis heatmap layer."""
    keep_auto = bool(getattr(layer, "_keep_auto_contrast", False))
    if keep_auto:
        layer._keep_auto_contrast = False
    try:
        layer.contrast_limits_range = limits
        layer.contrast_limits = limits
    finally:
        layer._keep_auto_contrast = keep_auto


def _is_thumbnail_rank_mismatch_error(error: RuntimeError) -> bool:
    """Return whether napari thumbnail generation hit the known rank bug."""
    return "sequence argument must have length equal to input rank" in str(error)


def _analysis_heatmap_ndisplay(layer: Any, response: Any | None = None) -> int | None:
    """Return the current display dimensionality for a heatmap layer."""
    slice_input = getattr(response, "slice_input", None)
    ndisplay = getattr(slice_input, "ndisplay", None)
    if isinstance(ndisplay, int):
        return ndisplay

    slice_input = getattr(layer, "_slice_input", None)
    ndisplay = getattr(slice_input, "ndisplay", None)
    if isinstance(ndisplay, int):
        return ndisplay
    return None


def _analysis_heatmap_requires_stable_limits(
    layer: Any, response: Any | None = None
) -> bool:
    """Return whether the 3D napari workaround should be active."""
    return _analysis_heatmap_ndisplay(layer, response) == 3


def _install_analysis_heatmap_layer_workarounds(layer: Any) -> None:
    """Install stable contrast behavior on one analysis heatmap layer."""
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return
    if metadata.get("heatmap_kind") != "analysis":
        return
    if getattr(layer, "_analysis_heatmap_workarounds_installed", False):
        return

    original_update_thumbnail = getattr(layer, "_update_thumbnail", None)
    if callable(original_update_thumbnail):

        def _safe_update_thumbnail(self) -> None:
            try:
                original_update_thumbnail()
            except RuntimeError as error:
                if not _is_thumbnail_rank_mismatch_error(error):
                    raise
                if not getattr(
                    self, "_analysis_heatmap_thumbnail_warning_logged", False
                ):
                    logger.warning(
                        "Suppressed napari thumbnail update failure for "
                        "analysis heatmap '%s': %s",
                        getattr(self, "name", "<unnamed>"),
                        error,
                    )
                    self._analysis_heatmap_thumbnail_warning_logged = True

        layer._update_thumbnail = MethodType(_safe_update_thumbnail, layer)

    original_reset_contrast_limits = getattr(layer, "reset_contrast_limits", None)
    if callable(original_reset_contrast_limits):

        def _stable_reset_contrast_limits(self, mode=None) -> None:
            if not _analysis_heatmap_requires_stable_limits(self):
                original_reset_contrast_limits(mode)
                return
            limits = _analysis_heatmap_stored_limits(self)
            if limits is None:
                original_reset_contrast_limits(mode)
                return
            _apply_analysis_heatmap_contrast_limits(self, limits)

        layer.reset_contrast_limits = MethodType(_stable_reset_contrast_limits, layer)

    original_reset_contrast_limits_range = getattr(
        layer, "reset_contrast_limits_range", None
    )
    if callable(original_reset_contrast_limits_range):

        def _stable_reset_contrast_limits_range(self, mode=None) -> None:
            if not _analysis_heatmap_requires_stable_limits(self):
                original_reset_contrast_limits_range(mode)
                return
            limits = _analysis_heatmap_stored_limits(self)
            if limits is None:
                original_reset_contrast_limits_range(mode)
                return
            self.contrast_limits_range = limits

        layer.reset_contrast_limits_range = MethodType(
            _stable_reset_contrast_limits_range, layer
        )

    original_update_slice_response = getattr(layer, "_update_slice_response", None)
    if callable(original_update_slice_response):

        def _stable_update_slice_response(self, response) -> Any:
            keep_auto = bool(getattr(self, "_keep_auto_contrast", False))
            if not keep_auto or not _analysis_heatmap_requires_stable_limits(
                self, response
            ):
                return original_update_slice_response(response)

            self._keep_auto_contrast = False
            try:
                result = original_update_slice_response(response)
            finally:
                self._keep_auto_contrast = True

            limits = _analysis_heatmap_stored_limits(self)
            if limits is not None:
                _apply_analysis_heatmap_contrast_limits(self, limits)
            return result

        layer._update_slice_response = MethodType(_stable_update_slice_response, layer)

    layer._analysis_heatmap_workarounds_installed = True


class AnalysisTabWidget(QWidget):
    """Widget providing the Analysis tab UI.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    parent : QWidget, optional
        Parent widget.
    """

    cluster_colors_updated = Signal(object, dict)

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None):
        super().__init__(parent)
        self._viewer = viewer
        self._db: NeuronDatabase | None = None
        self._atlas: BrainGlobeAtlas | None = None
        self._parquet_path: str | None = None
        self._worker_thread: QThread | None = None
        self._current_worker = None
        self._last_cluster_result: ClusterResult | None = None
        self._cluster_color_map: dict[str, list[float]] | None = None
        self._actual_n_clusters: int = 0
        self._heatmap_layer = None
        self._current_heatmap_request: _HeatmapRequest | None = None
        self._pending_heatmap_requests: list[_HeatmapRequest] = []
        self._completed_heatmap_requests: list[_HeatmapRequest] = []
        self._last_heatmap_requests: list[_HeatmapRequest] = []
        self._active_heatmap_total: int = 0
        self._active_heatmap_index: int = 0
        self._heatmap_batch_mode: bool = False
        self._slice_projector = None
        self._dataset_region_ids: set[int] = set()
        self._dataset_node_types: tuple[int, ...] = ()
        self._clustermap_rendered = False
        self._cluster_region_query_scope = _ANALYSIS_SCOPE_WHOLE
        self._current_table_file_ids_provider = None
        self._selected_table_file_ids_provider = None
        self._cluster_assignment_store = None
        self._pending_cluster_context: dict[str, object] = {}
        self._pending_clustering_request: _ClusteringRequest | None = None
        self._pending_clustering_preflight = None
        self._flatmap_correlation_source_provider = None
        self._flatmap_coords_available = False
        self._flatmap_available_styles: tuple[str, ...] = ()
        self._flatmap_coords_cache_path: str | None = None
        self._voxel_dendrite_coverage: DendriteLabelCoverage | None = None
        self._voxel_dendrite_filter_active = False
        self._setup_ui()

        # Rebuild heatmap when the user reorders axes in napari
        self._viewer.dims.events.order.connect(self._on_dims_order_changed)

    def set_database(self, db: NeuronDatabase) -> None:
        """Set the database connection."""
        self._db = db
        self._parquet_path = str(db.parquet_path)
        self._clear_dendrite_label_restriction()
        self.refresh_available_regions_from_database()
        self._update_button_states()

    def refresh_available_regions_from_database(self) -> None:
        """Cache dataset region IDs and refresh analysis selectors."""
        if self._db is None:
            self._dataset_region_ids = set()
            self._dataset_node_types = ()
            self._refresh_analysis_region_selectors()
            return

        regions_df = self._db.get_unique_regions()
        normalized_ids: set[int] = set()
        region_values = regions_df["region_id"] if "region_id" in regions_df else []
        for region_id in list(region_values):
            if region_id is None:
                continue
            if isinstance(region_id, (float, np.floating)) and np.isnan(region_id):
                continue

            try:
                value = int(region_id)
            except (TypeError, ValueError):
                continue
            if value > 0:
                normalized_ids.add(value)

        self._dataset_region_ids = normalized_ids
        get_types = getattr(self._db, "get_unique_node_types", None)
        if callable(get_types):
            self._dataset_node_types = tuple(get_types())
        else:
            self._dataset_node_types = ()
        node_type_combo = getattr(self, "_voxel_node_type_combo", None)
        set_options = getattr(node_type_combo, "set_options", None)
        if callable(set_options):
            set_options(node_type_options(self._dataset_node_types))
        self._refresh_analysis_region_selectors()

    def set_atlas(self, atlas: BrainGlobeAtlas) -> None:
        """Set the atlas instance."""
        self._atlas = atlas
        self._refresh_analysis_region_selectors()
        self._update_button_states()
        self._update_voxel_depth_label()

    def set_slice_projector(self, projector) -> None:
        """Set the slice projector for updating 2D projection colors."""
        self._slice_projector = projector

    def set_current_table_file_ids_provider(self, provider) -> None:
        """Set a callback returning the current neuron-table file IDs."""
        self._current_table_file_ids_provider = provider

    def set_selected_table_file_ids_provider(self, provider) -> None:
        """Set a callback returning explicitly selected table-row file IDs."""
        self._selected_table_file_ids_provider = provider

    def set_cluster_assignment_store(self, store) -> None:
        """Set the shared named cluster-assignment store."""
        self._cluster_assignment_store = store
        self.on_active_cluster_assignment_changed()

    def set_flatmap_correlation_source_provider(self, provider) -> None:
        """Retained for backward compatibility (no longer gates availability).

        Flat map + Depth clustering availability is now derived from the
        loaded Parquet's precomputed coordinate columns rather than a rendered
        heatmap source.
        """
        self._flatmap_correlation_source_provider = provider

    def _raw_current_table_file_ids(self) -> list[object]:
        """Return current neuron-table file IDs from the configured provider."""
        provider = self.__dict__.get("_current_table_file_ids_provider")
        if not callable(provider):
            return []
        try:
            file_ids = provider()
        except Exception:
            return []
        if file_ids is None:
            return []
        return list(file_ids)

    def _matched_current_table_file_ids(self) -> list[object]:
        """Return current table file IDs that exist in the cached cluster result."""
        store = self.__dict__.get("_cluster_assignment_store")
        assignment = getattr(store, "active", None)
        if assignment is not None:
            current_file_ids = self._raw_current_table_file_ids()
            return [
                file_id
                for file_id in current_file_ids
                if assignment.label_for(file_id) is not None
            ]
        result = getattr(self, "_last_cluster_result", None)
        if result is None:
            return []

        current_file_ids = self._raw_current_table_file_ids()
        if not current_file_ids:
            return []

        result_ids = list(getattr(result, "neuron_ids", []))
        if not result_ids:
            return []

        result_id_set = set(result_ids)
        result_id_strs = {str(file_id) for file_id in result_ids}
        return [
            file_id
            for file_id in current_file_ids
            if file_id in result_id_set or str(file_id) in result_id_strs
        ]

    def has_cached_clusters_for_current_table(self) -> bool:
        """Return whether cached clustering overlaps the current table."""
        if self._cluster_color_map is None or self._last_cluster_result is None:
            return False
        return bool(self._matched_current_table_file_ids())

    def _update_button_states(self) -> None:
        """Enable/disable buttons based on loaded data."""
        self.refresh_flatmap_coordinate_availability()
        ready = self._db is not None and self._atlas is not None
        busy = self._worker_thread is not None and self._worker_thread.isRunning()
        self._run_corr_btn.setEnabled(ready and not busy)
        self._run_heat_btn.setEnabled(ready and not busy)
        dendrite_scan = getattr(self, "_voxel_dendrite_scan_btn", None)
        if dendrite_scan is not None:
            dendrite_scan.setEnabled(ready and not busy)
        has_cluster_heatmap_options = (
            ready
            and not busy
            and getattr(self, "_heat_cluster_combo", None) is not None
            and self._heat_cluster_combo.count() > 1
        )
        if hasattr(self, "_add_all_cluster_heatmaps_btn"):
            self._add_all_cluster_heatmaps_btn.setEnabled(has_cluster_heatmap_options)
        analysis_ready = self._last_cluster_result is not None and not busy
        if hasattr(self, "_render_clustermap_btn"):
            self._render_clustermap_btn.setEnabled(analysis_ready)
        if hasattr(self, "_build_clustermap_btn"):
            self._build_clustermap_btn.setEnabled(analysis_ready)
        if hasattr(self, "_export_title_edit"):
            self._export_title_edit.setEnabled(analysis_ready)
            self._export_xlabel_edit.setEnabled(analysis_ready)
            self._export_dpi_combo.setEnabled(analysis_ready)
            self._save_cluster_workbook_btn.setEnabled(analysis_ready)
            self._save_distance_workbook_btn.setEnabled(analysis_ready)
            self._save_extended_parquet_btn.setEnabled(analysis_ready)
            self._save_dendrogram_btn.setEnabled(analysis_ready)

    def _on_thread_finished(self) -> None:
        """Clear worker references after the thread has stopped."""
        self._worker_thread = None
        self._current_worker = None

    def _setup_ui(self) -> None:
        outer_layout = QVBoxLayout(self)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(self._scroll_area)

        self._scroll_content = QWidget()
        self._scroll_area.setWidget(self._scroll_content)

        layout = QVBoxLayout(self._scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Clustering group ---
        self._clustering_section = CollapsibleSection(
            "Clustering",
            expanded=True,
        )
        corr_layout = self._clustering_section.content_layout()

        # Coordinate space
        coord_space_row = QHBoxLayout()
        coord_space_row.addWidget(QLabel("Coordinate space:"))
        self._coordinate_space_combo = QComboBox()
        self._coordinate_space_combo.addItem(_COORD_SPACE_CCF)
        self._coordinate_space_combo.currentTextChanged.connect(
            self._on_coordinate_space_changed
        )
        coord_space_row.addWidget(self._coordinate_space_combo)
        corr_layout.addLayout(coord_space_row)

        self._flatmap_coords_status_label = QLabel(_FLATMAP_COORDS_INFO_TEXT)
        set_word_wrap = getattr(
            self._flatmap_coords_status_label,
            "setWordWrap",
            None,
        )
        if callable(set_word_wrap):
            set_word_wrap(True)
        self._flatmap_coords_status_label.setVisible(False)
        corr_layout.addWidget(self._flatmap_coords_status_label)

        # Clustering method
        method_type_row = QHBoxLayout()
        method_type_row.addWidget(QLabel("Method:"))
        self._clustering_method_combo = QComboBox()
        self._clustering_method_combo.addItems(
            [_CLUSTER_METHOD_VOXEL, _CLUSTER_METHOD_SOMA]
        )
        self._clustering_method_combo.currentTextChanged.connect(
            self._on_clustering_method_changed
        )
        method_type_row.addWidget(self._clustering_method_combo)
        corr_layout.addLayout(method_type_row)

        # Algorithm (only for Soma Location)
        self._algorithm_row = QHBoxLayout()
        self._algorithm_label = QLabel("Algorithm:")
        self._algorithm_row.addWidget(self._algorithm_label)
        self._algorithm_combo = QComboBox()
        self._algorithm_combo.addItems(["Hierarchical", "K-Means", "DBSCAN"])
        self._algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)
        self._algorithm_row.addWidget(self._algorithm_combo)
        corr_layout.addLayout(self._algorithm_row)

        # Input neuron scope
        scope_row = QHBoxLayout()
        self._cluster_region_scope_label = QLabel("Input neurons:")
        scope_row.addWidget(self._cluster_region_scope_label)
        self._cluster_region_scope_combo = QComboBox()
        self._cluster_region_scope_combo.addItem(
            "Whole Parquet",
            _ANALYSIS_SCOPE_WHOLE,
        )
        self._cluster_region_scope_combo.addItem(
            "Current Table",
            _ANALYSIS_SCOPE_CURRENT,
        )
        self._cluster_region_scope_combo.addItem(
            "Selected Rows",
            _ANALYSIS_SCOPE_SELECTED,
        )
        self._cluster_region_scope_combo.currentTextChanged.connect(
            self._on_cluster_region_scope_changed
        )
        scope_row.addWidget(self._cluster_region_scope_combo)
        corr_layout.addLayout(scope_row)

        # Per-scope anatomical include/exclude rules.
        region_row = QHBoxLayout()
        self._cluster_region_target_label = QLabel("Included regions:")
        region_row.addWidget(self._cluster_region_target_label)
        self._cluster_region_summary_label = QLabel("None selected")
        region_row.addWidget(self._cluster_region_summary_label)
        corr_layout.addLayout(region_row)

        exclude_region_row = QHBoxLayout()
        self._cluster_exclude_region_label = QLabel("Excluded regions:")
        exclude_region_row.addWidget(self._cluster_exclude_region_label)
        self._cluster_exclude_region_summary_label = QLabel("None")
        exclude_region_row.addWidget(self._cluster_exclude_region_summary_label)
        corr_layout.addLayout(exclude_region_row)

        self._cluster_region_section = CollapsibleSection(
            "Region Filters",
            expanded=False,
        )
        cluster_region_layout = self._cluster_region_section.content_layout()

        self._cluster_region_scope_stack = QStackedWidget()

        whole_page = QWidget()
        whole_layout = QVBoxLayout(whole_page)
        whole_layout.setContentsMargins(0, 0, 0, 0)
        self._whole_parquet_region_filter_editor = RegionFilterEditorWidget(
            node_types=self._dataset_node_types,
        )
        self._whole_parquet_region_filter_editor.rules_changed.connect(
            self._on_cluster_region_rules_changed
        )
        whole_layout.addWidget(self._whole_parquet_region_filter_editor)
        self._cluster_region_scope_stack.addWidget(whole_page)

        current_page = QWidget()
        current_layout = QVBoxLayout(current_page)
        current_layout.setContentsMargins(0, 0, 0, 0)
        self._current_table_region_filter_editor = RegionFilterEditorWidget(
            node_types=self._dataset_node_types,
        )
        self._current_table_region_filter_editor.rules_changed.connect(
            self._on_cluster_region_rules_changed
        )
        current_layout.addWidget(self._current_table_region_filter_editor)
        self._cluster_region_scope_stack.addWidget(current_page)

        selected_page = QWidget()
        selected_layout = QVBoxLayout(selected_page)
        selected_layout.setContentsMargins(0, 0, 0, 0)
        self._selected_rows_region_filter_editor = RegionFilterEditorWidget(
            node_types=self._dataset_node_types,
        )
        self._selected_rows_region_filter_editor.rules_changed.connect(
            self._on_cluster_region_rules_changed
        )
        selected_layout.addWidget(self._selected_rows_region_filter_editor)
        self._cluster_region_scope_stack.addWidget(selected_page)

        # Compatibility aliases keep legacy inclusion-only helpers and callers
        # working while the visible editor owns both sides of the filter.
        self._whole_parquet_cluster_region_selector = (
            self._whole_parquet_region_filter_editor.include_selector
        )
        self._current_table_cluster_region_selector = (
            self._current_table_region_filter_editor.include_selector
        )
        self._selected_rows_cluster_region_selector = (
            self._selected_rows_region_filter_editor.include_selector
        )

        cluster_region_layout.addWidget(self._cluster_region_scope_stack)
        corr_layout.addWidget(self._cluster_region_section)

        # Hidden legacy controls support older programmatic callers. New UI
        # stores dilation independently on each Region Filters table row.
        self._dilation_label = QLabel("Dilation %:")
        self._dilation_label.setVisible(False)
        self._dilation_spin = QSpinBox()
        self._dilation_spin.setRange(0, 100)
        self._dilation_spin.setValue(0)
        self._dilation_spin.setSuffix("%")
        self._dilation_spin.setVisible(False)

        # Linkage method
        self._linkage_row = QHBoxLayout()
        self._linkage_label = QLabel("Linkage:")
        self._linkage_row.addWidget(self._linkage_label)
        self._method_combo = QComboBox()
        self._method_combo.addItems(["average", "ward", "complete", "single"])
        self._method_combo.setCurrentText("ward")
        self._linkage_row.addWidget(self._method_combo)
        corr_layout.addLayout(self._linkage_row)

        # Number of clusters
        self._clusters_row = QHBoxLayout()
        self._clusters_label = QLabel("Clusters:")
        self._clusters_row.addWidget(self._clusters_label)
        self._n_clusters_spin = QSpinBox()
        self._n_clusters_spin.setRange(2, 50)
        self._n_clusters_spin.setValue(5)
        self._clusters_row.addWidget(self._n_clusters_spin)
        corr_layout.addLayout(self._clusters_row)

        # DBSCAN eps.  Units depend on the coordinate space: microns for CCFv3,
        # normalized hemisphere fractions for flat map space.  Both remembered
        # values live here so switching spaces preserves each setting.
        self._eps_value_um = 100.0
        self._eps_value_normalized = float(DEFAULT_FLATMAP_SOMA_DBSCAN_EPS)
        self._eps_units_normalized: bool | None = None
        self._eps_row = QHBoxLayout()
        self._eps_label = QLabel("Eps (μm):")
        self._eps_row.addWidget(self._eps_label)
        self._eps_spin = QDoubleSpinBox()
        self._eps_spin.setRange(1.0, 10000.0)
        self._eps_spin.setValue(self._eps_value_um)
        self._eps_spin.setSuffix(" μm")
        self._eps_spin.setDecimals(1)
        self._eps_row.addWidget(self._eps_spin)
        corr_layout.addLayout(self._eps_row)

        # DBSCAN min_samples
        self._min_samples_row = QHBoxLayout()
        self._min_samples_label = QLabel("Min samples:")
        self._min_samples_row.addWidget(self._min_samples_label)
        self._min_samples_spin = QSpinBox()
        self._min_samples_spin.setRange(1, 100)
        self._min_samples_spin.setValue(5)
        self._min_samples_row.addWidget(self._min_samples_spin)
        corr_layout.addLayout(self._min_samples_row)

        # Flatmap binning controls (Flat map + Depth voxel correlation only)
        self._flatmap_style_row = QHBoxLayout()
        self._flatmap_style_label = QLabel("Flatmap style:")
        self._flatmap_style_row.addWidget(self._flatmap_style_label)
        self._flatmap_style_combo = QComboBox()
        self._flatmap_style_row.addWidget(self._flatmap_style_combo)
        corr_layout.addLayout(self._flatmap_style_row)

        # Flat map soma clustering: depth weighting.  Raw Parquet coordinates
        # mix normalized flat map units with microns, so soma coordinates are
        # scaled by one shared flat map divisor and depth is then weighted here.
        self._flatmap_ignore_depth_cb = QCheckBox("Ignore depth (flat map X/Y only)")
        self._flatmap_ignore_depth_cb.setChecked(False)
        self._flatmap_ignore_depth_cb.setToolTip(_IGNORE_DEPTH_TOOLTIP)
        self._flatmap_ignore_depth_cb.toggled.connect(self._on_ignore_depth_toggled)
        corr_layout.addWidget(self._flatmap_ignore_depth_cb)

        self._flatmap_depth_scale_row = QHBoxLayout()
        self._flatmap_depth_scale_label = QLabel("Depth scale:")
        self._flatmap_depth_scale_label.setToolTip(_DEPTH_SCALE_TOOLTIP)
        self._flatmap_depth_scale_row.addWidget(self._flatmap_depth_scale_label)
        self._flatmap_depth_scale_spin = QDoubleSpinBox()
        self._flatmap_depth_scale_spin.setRange(0.0, 100.0)
        self._flatmap_depth_scale_spin.setDecimals(2)
        self._flatmap_depth_scale_spin.setSingleStep(0.25)
        self._flatmap_depth_scale_spin.setValue(float(DEFAULT_FLATMAP_DEPTH_SCALE))
        self._flatmap_depth_scale_spin.setToolTip(_DEPTH_SCALE_TOOLTIP)
        self._flatmap_depth_scale_row.addWidget(self._flatmap_depth_scale_spin)
        corr_layout.addLayout(self._flatmap_depth_scale_row)

        self._flatmap_y_bins_row = QHBoxLayout()
        self._flatmap_y_bins_label = QLabel("Y bins:")
        self._flatmap_y_bins_row.addWidget(self._flatmap_y_bins_label)
        self._flatmap_y_bins_spin = QSpinBox()
        self._flatmap_y_bins_spin.setRange(2, MAX_FLATMAP_Y_BINS)
        self._flatmap_y_bins_spin.setValue(DEFAULT_FLATMAP_Y_BINS)
        self._flatmap_y_bins_spin.setToolTip(FLATMAP_Y_BINS_TOOLTIP)
        self._flatmap_y_bins_label.setToolTip(FLATMAP_Y_BINS_TOOLTIP)
        self._flatmap_y_bins_row.addWidget(self._flatmap_y_bins_spin)
        corr_layout.addLayout(self._flatmap_y_bins_row)

        self._flatmap_depth_bin_row = QHBoxLayout()
        self._flatmap_depth_bin_label = QLabel("Depth bin (μm):")
        self._flatmap_depth_bin_row.addWidget(self._flatmap_depth_bin_label)
        self._flatmap_depth_bin_spin = QDoubleSpinBox()
        self._flatmap_depth_bin_spin.setRange(1.0, 10000.0)
        self._flatmap_depth_bin_spin.setDecimals(1)
        self._flatmap_depth_bin_spin.setValue(float(DEFAULT_FLATMAP_DEPTH_BIN_UM))
        self._flatmap_depth_bin_spin.setSuffix(" μm")
        self._flatmap_depth_bin_row.addWidget(self._flatmap_depth_bin_spin)
        corr_layout.addLayout(self._flatmap_depth_bin_row)

        self._flatmap_include_depth_minus_one_cb = QCheckBox("Include depth -1 plane")
        self._flatmap_include_depth_minus_one_cb.setChecked(True)
        corr_layout.addWidget(self._flatmap_include_depth_minus_one_cb)

        # Node-row filters shared by CCF and flatmap voxel correlation.  These
        # stay separate from anatomical region rules and soma clustering.
        self._voxel_node_filter_section = CollapsibleSection(
            "Voxel Node Filters",
            expanded=False,
        )
        voxel_filter_layout = self._voxel_node_filter_section.content_layout()

        node_type_filter_row = QHBoxLayout()
        node_type_filter_row.addWidget(QLabel("Node-type filter:"))
        self._voxel_node_type_mode_combo = QComboBox()
        self._voxel_node_type_mode_combo.addItem("Off", "all")
        self._voxel_node_type_mode_combo.addItem("Include selected", "include")
        self._voxel_node_type_mode_combo.addItem("Exclude selected", "exclude")
        self._voxel_node_type_mode_combo.currentTextChanged.connect(
            self._update_voxel_node_filter_controls
        )
        node_type_filter_row.addWidget(self._voxel_node_type_mode_combo)
        self._voxel_node_type_combo = NodeTypeSelectorComboBox(
            options=node_type_options(self._dataset_node_types)
        )
        node_type_filter_row.addWidget(self._voxel_node_type_combo)
        voxel_filter_layout.addLayout(node_type_filter_row)

        self._voxel_node_type_warning_label = QLabel(_VOXEL_NODE_TYPE_WARNING)
        self._voxel_node_type_warning_label.setWordWrap(True)
        style_setter = getattr(
            self._voxel_node_type_warning_label,
            "setStyleSheet",
            None,
        )
        if callable(style_setter):
            style_setter("color: #d9822b; font-weight: bold;")
        voxel_filter_layout.addWidget(self._voxel_node_type_warning_label)

        dendrite_action_row = QHBoxLayout()
        self._voxel_dendrite_scan_btn = QPushButton(
            "Find and exclude neurons lacking dendrite labels"
        )
        self._voxel_dendrite_scan_btn.clicked.connect(
            self._scan_and_enable_dendrite_label_filter
        )
        dendrite_action_row.addWidget(self._voxel_dendrite_scan_btn)
        self._voxel_dendrite_clear_btn = QPushButton("Clear restriction")
        self._voxel_dendrite_clear_btn.clicked.connect(
            self._clear_dendrite_label_restriction
        )
        dendrite_action_row.addWidget(self._voxel_dendrite_clear_btn)
        voxel_filter_layout.addLayout(dendrite_action_row)
        self._voxel_dendrite_status_label = QLabel(
            "No dendrite-label cohort restriction."
        )
        self._voxel_dendrite_status_label.setWordWrap(True)
        voxel_filter_layout.addWidget(self._voxel_dendrite_status_label)

        soma_distance_row = QHBoxLayout()
        self._voxel_soma_distance_enabled_cb = QCheckBox(
            "Exclude nodes within soma distance"
        )
        self._voxel_soma_distance_enabled_cb.setChecked(False)
        self._voxel_soma_distance_enabled_cb.toggled.connect(
            self._update_voxel_node_filter_controls
        )
        soma_distance_row.addWidget(self._voxel_soma_distance_enabled_cb)
        self._voxel_soma_distance_spin = QDoubleSpinBox()
        self._voxel_soma_distance_spin.setRange(0.0, 100000.0)
        self._voxel_soma_distance_spin.setDecimals(1)
        self._voxel_soma_distance_spin.setSingleStep(25.0)
        self._voxel_soma_distance_spin.setValue(200.0)
        self._voxel_soma_distance_spin.setSuffix(" μm")
        soma_distance_row.addWidget(self._voxel_soma_distance_spin)
        voxel_filter_layout.addLayout(soma_distance_row)
        self._voxel_soma_distance_warning_label = QLabel(_SOMA_DISTANCE_WARNING)
        self._voxel_soma_distance_warning_label.setWordWrap(True)
        voxel_filter_layout.addWidget(self._voxel_soma_distance_warning_label)
        corr_layout.addWidget(self._voxel_node_filter_section)
        self._update_voxel_node_filter_controls()

        # Run button
        self._run_corr_btn = QPushButton("Run Clustering")
        self._run_corr_btn.setEnabled(False)
        self._run_corr_btn.clicked.connect(self._run_clustering_pipeline)
        corr_layout.addWidget(self._run_corr_btn)

        layout.addWidget(self._clustering_section)

        # Set initial visibility
        self._on_clustering_method_changed(self._clustering_method_combo.currentText())

        # --- Node Count Heatmap group ---
        self._heatmap_section = CollapsibleSection(
            "Node Count Heatmap",
            expanded=True,
        )
        heat_layout = self._heatmap_section.content_layout()

        heat_region_row = QHBoxLayout()
        heat_region_row.addWidget(QLabel("Region filter:"))
        self._heat_region_summary_label = QLabel("All regions")
        heat_region_row.addWidget(self._heat_region_summary_label)
        heat_layout.addLayout(heat_region_row)

        self._heat_region_section = CollapsibleSection(
            "Select Heatmap Region",
            expanded=False,
        )
        heat_region_selector_layout = self._heat_region_section.content_layout()
        self._heat_region_selector = RegionSelectorWidget(
            single_select=True,
            show_include_children=False,
            force_include_children=True,
        )
        self._heat_region_selector.selection_changed.connect(
            self._on_heat_region_selection_changed
        )
        heat_region_selector_layout.addWidget(self._heat_region_selector)
        heat_layout.addWidget(self._heat_region_section)

        # Cluster filter
        cluster_filter_row = QHBoxLayout()
        cluster_filter_row.addWidget(QLabel("Cluster filter:"))
        self._heat_cluster_combo = QComboBox()
        self._heat_cluster_combo.addItem("All neurons")
        self._heat_cluster_combo.setEnabled(False)
        cluster_filter_row.addWidget(self._heat_cluster_combo)
        heat_layout.addLayout(cluster_filter_row)

        node_type_row = QHBoxLayout()
        node_type_row.addWidget(QLabel("Node types:"))
        self._heat_node_type_combo = NodeTypeSelectorComboBox()
        node_type_row.addWidget(self._heat_node_type_combo)
        heat_layout.addLayout(node_type_row)

        soma_radius_row = QHBoxLayout()
        self._heat_soma_radius_enabled_cb = QCheckBox("Filter by soma distance")
        self._heat_soma_radius_enabled_cb.setChecked(False)
        self._heat_soma_radius_enabled_cb.setToolTip(
            "Restrict heatmap nodes to a radius around each neuron's soma"
        )
        soma_radius_row.addWidget(self._heat_soma_radius_enabled_cb)
        soma_radius_row.addWidget(QLabel("Soma radius:"))
        self._heat_soma_radius_spin = QDoubleSpinBox()
        self._heat_soma_radius_spin.setRange(0.0, 100000.0)
        self._heat_soma_radius_spin.setValue(0.0)
        self._heat_soma_radius_spin.setSuffix(" μm")
        self._heat_soma_radius_spin.setDecimals(1)
        self._heat_soma_radius_spin.setToolTip(
            "Maximum node distance from each neuron's soma when enabled"
        )
        self._heat_soma_radius_spin.setEnabled(False)
        self._heat_soma_radius_enabled_cb.toggled.connect(
            self._heat_soma_radius_spin.setEnabled
        )
        soma_radius_row.addWidget(self._heat_soma_radius_spin)
        heat_layout.addLayout(soma_radius_row)

        # Depth bin factor
        depth_bin_row = QHBoxLayout()
        depth_bin_row.addWidget(QLabel("Depth bin factor:"))
        self._depth_bin_spin = QSpinBox()
        self._depth_bin_spin.setRange(1, 20)
        self._depth_bin_spin.setValue(1)
        self._depth_bin_spin.setToolTip(
            "Merge N depth-planes into one voxel along the slicing axis"
        )
        self._depth_bin_spin.valueChanged.connect(self._update_voxel_depth_label)
        depth_bin_row.addWidget(self._depth_bin_spin)
        heat_layout.addLayout(depth_bin_row)

        self._voxel_depth_label = QLabel("Voxel depth: — μm")
        heat_layout.addWidget(self._voxel_depth_label)

        heat_action_row = QHBoxLayout()
        self._run_heat_btn = QPushButton("Build Heatmap Volume")
        self._run_heat_btn.setEnabled(False)
        self._run_heat_btn.clicked.connect(self._run_heatmap_pipeline)
        heat_action_row.addWidget(self._run_heat_btn)

        self._add_all_cluster_heatmaps_btn = QPushButton("Add All Cluster Heatmaps")
        self._add_all_cluster_heatmaps_btn.setEnabled(False)
        self._add_all_cluster_heatmaps_btn.clicked.connect(
            self._run_all_cluster_heatmaps
        )
        heat_action_row.addWidget(self._add_all_cluster_heatmaps_btn)
        heat_layout.addLayout(heat_action_row)

        layout.addWidget(self._heatmap_section)

        # --- Progress bar ---
        self._progress_section = CollapsibleSection(
            "Progress",
            expanded=True,
        )
        progress_layout = self._progress_section.content_layout()
        self._progress_label = QLabel("")
        progress_layout.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        progress_layout.addWidget(self._progress_bar)
        layout.addWidget(self._progress_section)

        # --- Matplotlib canvas for clustermap ---
        self._clustermap_section = CollapsibleSection(
            "Clustermap",
            expanded=False,
        )
        clustermap_layout = self._clustermap_section.content_layout()
        self._clustermap_status_label = QLabel(
            "Run clustering, then click 'Build Dendrogram' to render the cluster map."
        )
        clustermap_layout.addWidget(self._clustermap_status_label)
        self._build_clustermap_btn = QPushButton("Build Dendrogram")
        self._build_clustermap_btn.clicked.connect(self._render_clustermap_requested)
        clustermap_layout.addWidget(self._build_clustermap_btn)
        self._render_clustermap_btn = self._build_clustermap_btn
        self._figure = Figure(figsize=(6, 6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(400)
        clustermap_layout.addWidget(self._canvas)
        layout.addWidget(self._clustermap_section)

        self._export_section = CollapsibleSection(
            "Export Results",
            expanded=False,
        )
        export_layout = self._export_section.content_layout()

        export_title_row = QHBoxLayout()
        export_title_row.addWidget(QLabel("Title:"))
        self._export_title_edit = QLineEdit()
        export_title_row.addWidget(self._export_title_edit)
        export_layout.addLayout(export_title_row)

        export_xlabel_row = QHBoxLayout()
        export_xlabel_row.addWidget(QLabel("X label:"))
        self._export_xlabel_edit = QLineEdit()
        export_xlabel_row.addWidget(self._export_xlabel_edit)
        export_layout.addLayout(export_xlabel_row)

        export_dpi_row = QHBoxLayout()
        export_dpi_row.addWidget(QLabel("DPI:"))
        self._export_dpi_combo = QComboBox()
        self._export_dpi_combo.addItem("150 dpi")
        self._export_dpi_combo.setItemData(0, 150)
        self._export_dpi_combo.addItem("300 dpi")
        self._export_dpi_combo.setItemData(1, 300)
        self._export_dpi_combo.addItem("600 dpi")
        self._export_dpi_combo.setItemData(2, 600)
        self._export_dpi_combo.setCurrentIndex(1)
        export_dpi_row.addWidget(self._export_dpi_combo)
        export_layout.addLayout(export_dpi_row)

        self._save_cluster_workbook_btn = QPushButton("Save Cluster Workbook")
        self._save_cluster_workbook_btn.clicked.connect(self._save_cluster_workbook)
        export_layout.addWidget(self._save_cluster_workbook_btn)

        self._save_distance_workbook_btn = QPushButton("Save Distance Workbook")
        self._save_distance_workbook_btn.clicked.connect(self._save_distance_workbook)
        export_layout.addWidget(self._save_distance_workbook_btn)

        self._save_extended_parquet_btn = QPushButton("Save Extended Parquet")
        self._save_extended_parquet_btn.clicked.connect(self._save_extended_parquet)
        export_layout.addWidget(self._save_extended_parquet_btn)

        self._save_dendrogram_btn = QPushButton("Save Dendrogram")
        self._save_dendrogram_btn.clicked.connect(self._save_dendrogram)
        export_layout.addWidget(self._save_dendrogram_btn)

        layout.addWidget(self._export_section)

        layout.addStretch()
        self._show_clustermap_message("Run clustering, then click Render Dendrogram.")
        self._sync_cluster_region_scope_selector()
        self._refresh_analysis_region_selectors()
        self._update_button_states()

    def _format_selected_region_text(
        self,
        selector: RegionSelectorWidget | None,
        *,
        empty_text: str,
    ) -> str:
        """Return a compact summary string for a selector's current region(s)."""
        regions = self._selected_regions(selector)
        if not regions:
            return empty_text

        labels: list[str] = []
        for struct_id, acronym in regions[:2]:
            struct = selector._structure_map.get(struct_id, {})
            name = str(struct.get("name", "")).strip()
            if name and name != acronym:
                labels.append(f"{acronym} ({name})")
            else:
                labels.append(acronym)

        if len(regions) <= 2:
            return ", ".join(labels)
        return f"{', '.join(labels)} +{len(regions) - 2} more"

    def _update_region_summary_labels(self) -> None:
        """Refresh the visible region summary labels for Analysis workflows."""
        cluster_scope = self._selected_cluster_region_scope()
        cluster_empty_text = (
            "None selected"
            if cluster_scope == _ANALYSIS_SCOPE_WHOLE
            else "All regions (optional)"
        )
        editor = self._active_region_filter_editor()
        if editor is not None:
            include_text = editor.summary(excluded=False)
            if include_text == "None":
                include_text = self._format_selected_region_text(
                    self._active_cluster_region_selector(),
                    empty_text=cluster_empty_text,
                )
            exclude_text = editor.summary(excluded=True)
        else:
            include_text = self._format_selected_region_text(
                self._active_cluster_region_selector(),
                empty_text=cluster_empty_text,
            )
            exclude_text = "None"
        self._cluster_region_summary_label.setText(include_text)
        exclude_summary = getattr(
            self,
            "_cluster_exclude_region_summary_label",
            None,
        )
        if exclude_summary is not None:
            exclude_summary.setText(exclude_text)
        target_label = getattr(self, "_cluster_region_target_label", None)
        if target_label is not None:
            target_label.setText(
                "Included regions:"
                if cluster_scope == _ANALYSIS_SCOPE_WHOLE
                else "Included regions (optional):"
            )
        self._heat_region_summary_label.setText(
            self._format_selected_region_text(
                getattr(self, "_heat_region_selector", None),
                empty_text="All regions",
            )
        )

    def _analysis_allowed_structure_ids(self) -> set[int]:
        """Return dataset-backed visible structure IDs for Analysis selectors."""
        if self._atlas is None or not self._dataset_region_ids:
            return set()

        allowed_ids: set[int] = set()
        for region_id in self._dataset_region_ids:
            struct = self._atlas.structures.get(int(region_id))
            if struct is None:
                continue
            allowed_ids.add(int(region_id))
            for path_id in struct.get("structure_id_path", []) or []:
                try:
                    allowed_ids.add(int(path_id))
                except (TypeError, ValueError):
                    continue
        return allowed_ids

    def _refresh_analysis_region_selectors(self) -> None:
        """Rebuild Analysis region selectors from atlas hierarchy and dataset IDs."""
        heat_selector = getattr(self, "_heat_region_selector", None)
        editors = self._region_filter_editors()
        cluster_selectors = self._cluster_region_selectors()
        if not cluster_selectors or heat_selector is None:
            return

        previous_heat_regions = self._selected_regions(heat_selector)

        if self._atlas is None or not self._dataset_region_ids:
            if editors:
                for editor in editors:
                    editor.clear()
            else:
                for selector in cluster_selectors:
                    selector.clear()
            if heat_selector is not None:
                heat_selector.clear()
            self._update_region_summary_labels()
            return

        allowed_ids = self._analysis_allowed_structure_ids()
        if editors:
            for editor in editors:
                editor.set_node_types(self._dataset_node_types)
                editor.set_atlas_and_allowed_ids(self._atlas, allowed_ids)
        else:
            previous_regions = [
                self._selected_regions(selector) for selector in cluster_selectors
            ]
            for selector, prior_regions in zip(cluster_selectors, previous_regions):
                selector.set_allowed_structure_ids(allowed_ids)
                selector.set_atlas(self._atlas)
                retained = [
                    (region_id, acronym)
                    for region_id, acronym in prior_regions
                    if region_id in allowed_ids
                ]
                if len(retained) > 1 and hasattr(selector, "select_regions"):
                    selector.select_regions(
                        [acronym for _region_id, acronym in retained]
                    )
                else:
                    previous_id = retained[0][0] if retained else None
                    selector.select_region_by_id(previous_id)

        heat_selector.set_allowed_structure_ids(allowed_ids)
        heat_selector.set_atlas(self._atlas)
        retained_heat = [
            (region_id, acronym)
            for region_id, acronym in previous_heat_regions
            if region_id in allowed_ids
        ]
        if len(retained_heat) > 1 and hasattr(heat_selector, "select_regions"):
            heat_selector.select_regions(
                [acronym for _region_id, acronym in retained_heat]
            )
        else:
            previous_id = retained_heat[0][0] if retained_heat else None
            heat_selector.select_region_by_id(previous_id)

        self._update_region_summary_labels()

    def _selected_cluster_region_scope(self) -> str:
        """Return the active clustering scope."""
        combo = getattr(self, "_cluster_region_scope_combo", None)
        if combo is None:
            scope = getattr(
                self,
                "_cluster_region_query_scope",
                _ANALYSIS_SCOPE_WHOLE,
            )
            if scope in {
                _ANALYSIS_SCOPE_WHOLE,
                _ANALYSIS_SCOPE_CURRENT,
                _ANALYSIS_SCOPE_SELECTED,
            }:
                return scope
            return _ANALYSIS_SCOPE_WHOLE

        current_data = getattr(combo, "currentData", None)
        if callable(current_data):
            data = current_data()
            if data in {
                _ANALYSIS_SCOPE_WHOLE,
                _ANALYSIS_SCOPE_CURRENT,
                _ANALYSIS_SCOPE_SELECTED,
            }:
                return str(data)

        current_text = getattr(combo, "currentText", None)
        text = current_text() if callable(current_text) else ""
        if text == "Current Table":
            return _ANALYSIS_SCOPE_CURRENT
        if text == "Selected Rows":
            return _ANALYSIS_SCOPE_SELECTED
        return _ANALYSIS_SCOPE_WHOLE

    def _cluster_region_selectors(self) -> tuple[RegionSelectorWidget, ...]:
        """Return all clustering region selectors, de-duplicated."""
        selectors = []
        for attr_name in (
            "_whole_parquet_cluster_region_selector",
            "_current_table_cluster_region_selector",
            "_selected_rows_cluster_region_selector",
            "_cluster_region_selector",
        ):
            selector = getattr(self, attr_name, None)
            if selector is None or selector in selectors:
                continue
            selectors.append(selector)
        return tuple(selectors)

    def _region_filter_editors(self) -> tuple[RegionFilterEditorWidget, ...]:
        """Return all scope-specific filter editors, de-duplicated."""
        editors: list[RegionFilterEditorWidget] = []
        for name in (
            "_whole_parquet_region_filter_editor",
            "_current_table_region_filter_editor",
            "_selected_rows_region_filter_editor",
        ):
            editor = getattr(self, name, None)
            if editor is not None and editor not in editors:
                editors.append(editor)
        return tuple(editors)

    def _active_region_filter_editor(self) -> RegionFilterEditorWidget | None:
        """Return the include/exclude editor for the selected input scope."""
        scope = self._selected_cluster_region_scope()
        name = {
            _ANALYSIS_SCOPE_WHOLE: "_whole_parquet_region_filter_editor",
            _ANALYSIS_SCOPE_CURRENT: "_current_table_region_filter_editor",
            _ANALYSIS_SCOPE_SELECTED: "_selected_rows_region_filter_editor",
        }.get(scope, "_whole_parquet_region_filter_editor")
        return getattr(self, name, None)

    def _cluster_region_selector_for_scope(
        self,
        scope: str | None = None,
    ) -> RegionSelectorWidget | None:
        """Return the clustering region selector for one scope."""
        selected_scope = scope
        if selected_scope not in {
            _ANALYSIS_SCOPE_WHOLE,
            _ANALYSIS_SCOPE_CURRENT,
            _ANALYSIS_SCOPE_SELECTED,
        }:
            selected_scope = self._selected_cluster_region_scope()

        if selected_scope == _ANALYSIS_SCOPE_SELECTED:
            selector = getattr(
                self,
                "_selected_rows_cluster_region_selector",
                None,
            )
            if selector is not None:
                return selector
        elif selected_scope == _ANALYSIS_SCOPE_CURRENT:
            selector = getattr(self, "_current_table_cluster_region_selector", None)
            if selector is not None:
                return selector
        else:
            selector = getattr(self, "_whole_parquet_cluster_region_selector", None)
            if selector is not None:
                return selector

        return getattr(self, "_cluster_region_selector", None)

    def _active_cluster_region_selector(self) -> RegionSelectorWidget | None:
        """Return the currently visible clustering selector."""
        return self._cluster_region_selector_for_scope()

    def _sync_cluster_region_scope_selector(self) -> None:
        """Show the clustering selector matching the active scope."""
        stack = getattr(self, "_cluster_region_scope_stack", None)
        if stack is None:
            return

        scope = self._selected_cluster_region_scope()
        index = {
            _ANALYSIS_SCOPE_WHOLE: 0,
            _ANALYSIS_SCOPE_CURRENT: 1,
            _ANALYSIS_SCOPE_SELECTED: 2,
        }.get(scope, 0)
        stack.setCurrentIndex(index)

    def _current_table_file_ids(self) -> list[str]:
        """Return file IDs currently present in the main neuron table."""
        return [str(file_id) for file_id in self._raw_current_table_file_ids()]

    def _selected_table_file_ids(self) -> list[str]:
        """Return de-duplicated explicitly selected table-row file IDs."""
        provider = self.__dict__.get("_selected_table_file_ids_provider")
        if not callable(provider):
            return []
        try:
            values = provider()
        except Exception:
            return []
        if values is None:
            return []
        return list(dict.fromkeys(str(file_id) for file_id in values))

    def _resolve_cluster_query_file_scope(
        self,
    ) -> tuple[bool, list[str] | None, str, int | None]:
        """Resolve the optional current-table restriction for clustering."""
        scope = self._selected_cluster_region_scope()
        if scope == _ANALYSIS_SCOPE_SELECTED:
            file_ids = self._selected_table_file_ids()
            if len(file_ids) >= 2:
                return True, file_ids, "selected rows", len(file_ids)
            if not file_ids:
                message = (
                    "No table rows are selected; select at least two neurons "
                    "before clustering Selected Rows."
                )
            else:
                message = (
                    "Only one table row is selected; select at least two "
                    "neurons before clustering Selected Rows."
                )
            self._progress_label.setText(message)
            return False, None, "selected rows", len(file_ids)

        if scope != _ANALYSIS_SCOPE_CURRENT:
            return True, None, "whole parquet", None

        file_ids = self._current_table_file_ids()
        if file_ids:
            return True, file_ids, "current table", len(file_ids)

        self._progress_label.setText(
            "Current table is empty; switch clustering scope to Whole Parquet or populate the table first."
        )
        return False, None, "current table", 0

    def _detect_flatmap_coordinates(self) -> tuple[bool, tuple[str, ...]]:
        """Return (available, styles) for flatmap coords in the loaded Parquet.

        Availability requires version-3 bilateral flatmap columns plus depth;
        legacy single-column Parquets are not supported by the DuckDB-based
        flatmap clustering path. Results are cached per Parquet path.
        """
        parquet_path = self._parquet_path
        if not parquet_path:
            self._flatmap_coords_cache_path = None
            self._flatmap_coords_available = False
            self._flatmap_available_styles = ()
            return False, ()

        if parquet_path == self._flatmap_coords_cache_path:
            return self._flatmap_coords_available, self._flatmap_available_styles

        available = False
        styles: tuple[str, ...] = ()
        try:
            from ..flatmap_parquet import read_flatmap_parquet_transform_info

            info = read_flatmap_parquet_transform_info(parquet_path)
            styles = tuple(info.available_styles)
            available = bool(styles) and bool(info.has_v3_depth)
        except Exception:
            logger.debug(
                "Could not inspect flatmap coordinates in %s",
                parquet_path,
                exc_info=True,
            )
            available = False
            styles = ()

        self._flatmap_coords_cache_path = parquet_path
        self._flatmap_coords_available = available
        self._flatmap_available_styles = styles
        return available, styles

    def _current_coordinate_space(self) -> str:
        """Return the selected clustering coordinate space."""
        combo = getattr(self, "_coordinate_space_combo", None)
        if combo is None:
            return _COORD_SPACE_CCF
        return combo.currentText() or _COORD_SPACE_CCF

    def _selected_flatmap_style(self) -> str | None:
        """Return the selected flatmap style key, or the first available."""
        combo = getattr(self, "_flatmap_style_combo", None)
        if combo is not None:
            style = combo.currentData()
            if style:
                return str(style)
        styles = getattr(self, "_flatmap_available_styles", ())
        return styles[0] if styles else None

    def _populate_flatmap_style_combo(self, styles: tuple[str, ...]) -> None:
        """Fill the flatmap style combo with the available bilateral styles."""
        combo = getattr(self, "_flatmap_style_combo", None)
        if combo is None:
            return
        previous = combo.currentData()
        blocker = getattr(combo, "blockSignals", None)
        if callable(blocker):
            blocker(True)
        try:
            combo.clear()
            for style in styles:
                combo.addItem(_FLATMAP_STYLE_LABELS.get(style, style), style)
            if previous in styles:
                combo.setCurrentText(_FLATMAP_STYLE_LABELS.get(previous, str(previous)))
        finally:
            if callable(blocker):
                blocker(False)

    def refresh_flatmap_coordinate_availability(self) -> None:
        """Show the Flat map + Depth space only when the Parquet has coords."""
        combo = getattr(self, "_coordinate_space_combo", None)
        if combo is None:
            return

        available, styles = self._detect_flatmap_coordinates()
        options = [_COORD_SPACE_CCF]
        if available:
            options.append(_COORD_SPACE_FLATMAP)

        current_text = combo.currentText()
        if current_text not in options:
            current_text = _COORD_SPACE_CCF

        blocker = getattr(combo, "blockSignals", None)
        if callable(blocker):
            blocker(True)
        try:
            combo.clear()
            combo.addItems(options)
            combo.setCurrentText(current_text)
        finally:
            if callable(blocker):
                blocker(False)

        self._populate_flatmap_style_combo(styles)
        self._update_clustering_visibility()

    def _selected_regions(
        self,
        selector: RegionSelectorWidget | None,
    ) -> list[tuple[int, str]]:
        """Return the directly selected regions for a selector."""
        if selector is None:
            return []

        if hasattr(selector, "get_selected_ids"):
            selected_ids = selector.get_selected_ids(include_children=False)
            regions: list[tuple[int, str]] = []
            for struct_id in selected_ids:
                struct = selector._structure_map.get(int(struct_id), {})
                acronym = str(struct.get("acronym", "")).strip()
                if acronym:
                    regions.append((int(struct_id), acronym))
            if regions:
                return regions

        selected = selector.get_single_selected_region()
        return [selected] if selected is not None else []

    def _selected_region(
        self, selector: RegionSelectorWidget | None
    ) -> tuple[int, str] | None:
        """Return the first directly selected region for a selector, if any."""
        regions = self._selected_regions(selector)
        return regions[0] if regions else None

    def _selected_cluster_region(self) -> tuple[int, str] | None:
        """Return the first currently selected clustering region."""
        return self._selected_region(self._active_cluster_region_selector())

    def _selected_cluster_regions(self) -> list[tuple[int, str]]:
        """Return all currently selected clustering regions."""
        return self._selected_regions(self._active_cluster_region_selector())

    def _selected_heat_region(self) -> tuple[int, str] | None:
        """Return the currently selected heatmap region."""
        return self._selected_region(getattr(self, "_heat_region_selector", None))

    def _represented_region_ids_for_selection(self, region_id: int) -> list[int]:
        """Return represented dataset region IDs inside a selected atlas region."""
        if self._atlas is None or not self._dataset_region_ids:
            return []

        selected_region_id = int(region_id)
        represented_ids: list[int] = []
        for candidate_id in sorted(self._dataset_region_ids):
            struct = self._atlas.structures.get(int(candidate_id))
            if struct is None:
                continue
            try:
                path_ids = [
                    int(path_id)
                    for path_id in struct.get("structure_id_path", []) or []
                ]
            except (TypeError, ValueError):
                continue
            if selected_region_id in path_ids:
                represented_ids.append(int(candidate_id))
        return represented_ids

    def _represented_region_entries_for_selections(
        self,
        region_ids: list[int],
    ) -> list[tuple[int, str]]:
        """Return represented dataset region IDs/acronyms for selected regions."""
        represented: list[tuple[int, str]] = []
        seen_ids: set[int] = set()
        if self._atlas is None:
            return represented

        for region_id in region_ids:
            for represented_id in self._represented_region_ids_for_selection(region_id):
                if represented_id in seen_ids:
                    continue
                struct = self._atlas.structures.get(int(represented_id), {})
                acronym = str(struct.get("acronym", "")).strip()
                if not acronym:
                    continue
                seen_ids.add(int(represented_id))
                represented.append((int(represented_id), acronym))
        return represented

    def _selected_cluster_region_selection(
        self,
    ) -> ClusterRegionSelection | None:
        """Return the full clustering region-selection payload."""
        from ..analysis.clustering import ClusterRegionSelection

        selected_regions = self._selected_cluster_regions()
        if not selected_regions:
            return None

        represented_regions = self._represented_region_entries_for_selections(
            [region_id for region_id, _acronym in selected_regions]
        )
        return ClusterRegionSelection(
            selected_region_ids=[region_id for region_id, _acronym in selected_regions],
            selected_region_acronyms=[
                acronym for _region_id, acronym in selected_regions
            ],
            represented_region_ids=[
                region_id for region_id, _acronym in represented_regions
            ],
            represented_region_acronyms=[
                acronym for _region_id, acronym in represented_regions
            ],
        )

    def _selected_cluster_region_filter(self):
        """Return canonical include/exclude rules for the active scope."""
        editor = self._active_region_filter_editor()
        if editor is not None:
            return editor.region_filter(self._represented_region_entries_for_selections)
        return None

    def _on_cluster_region_rules_changed(self) -> None:
        """Keep summaries synchronized with per-region rule edits."""
        self._update_region_summary_labels()

    def _on_cluster_region_selection_changed(self, _acronyms: list[str]) -> None:
        """Keep the clustering region summary in sync with tree selection."""
        self._update_region_summary_labels()

    def _on_cluster_region_scope_changed(self, _text: str) -> None:
        """Switch the visible clustering selector to the active scope."""
        self._clear_dendrite_label_restriction(
            "Input scope changed; rerun the dendrite-label scan if needed."
        )
        self._cluster_region_query_scope = self._selected_cluster_region_scope()
        self._sync_cluster_region_scope_selector()
        self._update_region_summary_labels()

    def _on_heat_region_selection_changed(self, _acronyms: list[str]) -> None:
        """Keep the heatmap region summary in sync with tree selection."""
        self._update_region_summary_labels()

    def _on_coordinate_space_changed(self, _text: str) -> None:
        """Update visible controls when the coordinate space changes."""
        self._update_clustering_visibility()

    def _on_clustering_method_changed(self, _text: str) -> None:
        """Update visible controls when the clustering method changes."""
        self._update_clustering_visibility()

    def _update_clustering_visibility(self) -> None:
        """Show/hide clustering controls for the coord space + method combo."""
        is_flatmap = self._current_coordinate_space() == _COORD_SPACE_FLATMAP
        method = self._clustering_method_combo.currentText()
        is_soma = method == _CLUSTER_METHOD_SOMA

        voxel_filter_section = getattr(self, "_voxel_node_filter_section", None)
        if voxel_filter_section is not None:
            voxel_filter_section.setVisible(not is_soma)

        status_label = getattr(self, "_flatmap_coords_status_label", None)
        if status_label is not None:
            status_label.setVisible(is_flatmap)

        # Region rules use CCF coordinates before either CCF or flatmap
        # clustering, so the editor remains available in both spaces.
        for widget in (
            getattr(self, "_cluster_region_target_label", None),
            getattr(self, "_cluster_region_summary_label", None),
            getattr(self, "_cluster_exclude_region_label", None),
            getattr(self, "_cluster_exclude_region_summary_label", None),
            getattr(self, "_cluster_region_section", None),
        ):
            if widget is not None:
                widget.setVisible(True)
        for editor in self._region_filter_editors():
            editor.set_soma_mode(is_soma)

        # Flatmap style selector (both flatmap methods need the column family).
        for widget in (
            getattr(self, "_flatmap_style_label", None),
            getattr(self, "_flatmap_style_combo", None),
        ):
            if widget is not None:
                widget.setVisible(is_flatmap)

        # Flatmap binning controls only matter for voxel correlation.
        flatmap_voxel = is_flatmap and not is_soma
        for widget in (
            getattr(self, "_flatmap_y_bins_label", None),
            getattr(self, "_flatmap_y_bins_spin", None),
            getattr(self, "_flatmap_depth_bin_label", None),
            getattr(self, "_flatmap_depth_bin_spin", None),
            getattr(self, "_flatmap_include_depth_minus_one_cb", None),
        ):
            if widget is not None:
                widget.setVisible(flatmap_voxel)

        # Ignoring depth is meaningful for both flat map methods: soma
        # clustering drops the axis from its distance, voxel correlation
        # collapses the grid's depth planes.
        flatmap_soma = is_flatmap and is_soma
        ignore_depth_cb = getattr(self, "_flatmap_ignore_depth_cb", None)
        if ignore_depth_cb is not None:
            ignore_depth_cb.setVisible(is_flatmap)

        # A depth *weight*, by contrast, only applies to soma clustering.  The
        # correlation compares voxel occupancy, which has no notion of distance
        # between voxels, so scaling depth there would merely duplicate the
        # Depth bin control.
        for widget in (
            getattr(self, "_flatmap_depth_scale_label", None),
            getattr(self, "_flatmap_depth_scale_spin", None),
        ):
            if widget is not None:
                widget.setVisible(flatmap_soma)
        if is_flatmap and ignore_depth_cb is not None:
            self._on_ignore_depth_toggled(ignore_depth_cb.isChecked())
        self._apply_eps_units(flatmap_soma)

        # Algorithm row: only for soma (in either coordinate space).
        self._algorithm_label.setVisible(is_soma)
        self._algorithm_combo.setVisible(is_soma)

        if is_soma:
            self._on_algorithm_changed(self._algorithm_combo.currentText())
        else:
            # Voxel correlation: show linkage + clusters, hide DBSCAN params.
            self._linkage_label.setVisible(True)
            self._method_combo.setVisible(True)
            self._clusters_label.setVisible(True)
            self._n_clusters_spin.setVisible(True)
            self._eps_label.setVisible(False)
            self._eps_spin.setVisible(False)
            self._min_samples_label.setVisible(False)
            self._min_samples_spin.setVisible(False)

    def _update_voxel_node_filter_controls(self, *_args) -> None:
        """Synchronize enabled states for the independent voxel-node filters."""
        mode_combo = getattr(self, "_voxel_node_type_mode_combo", None)
        mode = "all"
        if mode_combo is not None:
            current_data = getattr(mode_combo, "currentData", None)
            if callable(current_data):
                mode = str(current_data() or "all")
        type_enabled = mode != "all"
        type_combo = getattr(self, "_voxel_node_type_combo", None)
        if type_combo is not None:
            type_combo.setEnabled(type_enabled)
        warning = getattr(self, "_voxel_node_type_warning_label", None)
        if warning is not None:
            warning.setVisible(
                type_enabled
                or bool(getattr(self, "_voxel_dendrite_filter_active", False))
            )

        distance_checkbox = getattr(self, "_voxel_soma_distance_enabled_cb", None)
        distance_enabled = bool(
            distance_checkbox is not None and distance_checkbox.isChecked()
        )
        distance_spin = getattr(self, "_voxel_soma_distance_spin", None)
        if distance_spin is not None:
            distance_spin.setEnabled(distance_enabled)
        distance_warning = getattr(
            self,
            "_voxel_soma_distance_warning_label",
            None,
        )
        if distance_warning is not None:
            distance_warning.setVisible(distance_enabled)

        clear_button = getattr(self, "_voxel_dendrite_clear_btn", None)
        if clear_button is not None:
            clear_button.setEnabled(
                bool(getattr(self, "_voxel_dendrite_filter_active", False))
            )

    def _on_ignore_depth_toggled(self, ignored: bool) -> None:
        """Disable the controls that stop meaning anything without depth.

        The depth scale has nothing to weight once the axis is gone, and a
        collapsed voxel grid has no depth bins to size.  ``Include depth -1
        plane`` stays enabled either way, because depth still decides which
        nodes are counted even when the axis is collapsed.
        """
        for name in (
            "_flatmap_depth_scale_spin",
            "_flatmap_depth_scale_label",
            "_flatmap_depth_bin_spin",
            "_flatmap_depth_bin_label",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not ignored)

    def _apply_eps_units(self, normalized_space: bool) -> None:
        """Retarget the DBSCAN eps control at the active coordinate space.

        Flat map soma clustering measures distance in normalized flat map units,
        where one hemisphere height is 1.0 — the micron range the CCFv3
        clustering needs would collapse every soma into a single cluster.  Each
        space keeps its own remembered value so switching back and forth does
        not destroy the user's setting.
        """
        spin = getattr(self, "_eps_spin", None)
        label = getattr(self, "_eps_label", None)
        if spin is None:
            return
        if getattr(self, "_eps_units_normalized", None) == normalized_space:
            return

        # Remember the outgoing value before the range change clamps it.
        previous = getattr(self, "_eps_units_normalized", None)
        if previous is not None:
            attr = "_eps_value_normalized" if previous else "_eps_value_um"
            setattr(self, attr, float(spin.value()))

        if normalized_space:
            restored = getattr(
                self,
                "_eps_value_normalized",
                float(DEFAULT_FLATMAP_SOMA_DBSCAN_EPS),
            )
            spin.setDecimals(3)
            spin.setRange(0.001, 10.0)
            spin.setSingleStep(0.01)
            spin.setSuffix("")
            if label is not None:
                label.setText("Eps (hemisphere fraction):")
                label.setToolTip(_FLATMAP_EPS_TOOLTIP)
            spin.setToolTip(_FLATMAP_EPS_TOOLTIP)
        else:
            restored = getattr(self, "_eps_value_um", 100.0)
            spin.setDecimals(1)
            spin.setRange(1.0, 10000.0)
            spin.setSingleStep(1.0)
            spin.setSuffix(" μm")
            if label is not None:
                label.setText("Eps (μm):")
                label.setToolTip("DBSCAN neighbourhood radius in microns.")
            spin.setToolTip("DBSCAN neighbourhood radius in microns.")
        spin.setValue(float(restored))
        self._eps_units_normalized = bool(normalized_space)

    def _on_algorithm_changed(self, text: str) -> None:
        """Show/hide UI rows based on the selected soma algorithm."""
        is_dbscan = text == "DBSCAN"
        is_hierarchical = text == "Hierarchical"

        self._linkage_label.setVisible(is_hierarchical)
        self._method_combo.setVisible(is_hierarchical)
        self._clusters_label.setVisible(not is_dbscan)
        self._n_clusters_spin.setVisible(not is_dbscan)
        self._eps_label.setVisible(is_dbscan)
        self._eps_spin.setVisible(is_dbscan)
        self._min_samples_label.setVisible(is_dbscan)
        self._min_samples_spin.setVisible(is_dbscan)

    def _available_dendrite_node_types(self) -> tuple[int, ...]:
        """Return represented standard basal/apical dendrite type codes."""
        represented = set(getattr(self, "_dataset_node_types", ()))
        return tuple(value for value in (3, 4) if value in represented)

    def _clear_dendrite_label_restriction(
        self,
        message: str = "No dendrite-label cohort restriction.",
    ) -> None:
        """Clear the reversible neuron-level dendrite-label requirement."""
        self._voxel_dendrite_coverage = None
        self._voxel_dendrite_filter_active = False
        status = getattr(self, "_voxel_dendrite_status_label", None)
        if status is not None:
            status.setText(message)
        self._update_voxel_node_filter_controls()

    def _scan_and_enable_dendrite_label_filter(self) -> None:
        """Scan the active input scope and require neurons with type 3/4 labels."""
        if self._parquet_path is None:
            return
        if self._worker_thread is not None and self._worker_thread.isRunning():
            return
        dendrite_types = self._available_dendrite_node_types()
        if not dendrite_types:
            self._clear_dendrite_label_restriction(
                "The loaded Parquet contains no basal/apical dendrite node types "
                "(type 3 or 4); no cohort restriction was applied."
            )
            return
        proceed, file_ids, _scope_label, _input_count = (
            self._resolve_cluster_query_file_scope()
        )
        if not proceed:
            return

        from ..workers import DendriteCoverageWorker

        self._clear_dendrite_label_restriction(
            "Scanning complete neurons for dendrite labels..."
        )
        self._dendrite_scan_scope = self._selected_cluster_region_scope()
        worker = DendriteCoverageWorker(
            parquet_path=self._parquet_path,
            file_ids=file_ids,
            dendrite_node_types=dendrite_types,
        )
        self._progress_label.setText("Scanning complete neurons for dendrite labels...")
        self._start_background_worker(worker, self._on_dendrite_coverage_finished)

    def _on_dendrite_coverage_finished(self, coverage) -> None:
        """Activate a completed coverage scan if its input scope is still current."""
        self._progress_bar.setVisible(False)
        if getattr(self, "_dendrite_scan_scope", None) != (
            self._selected_cluster_region_scope()
        ):
            self._clear_dendrite_label_restriction(
                "Input scope changed during the scan; no restriction was applied."
            )
            return
        self._voxel_dendrite_coverage = coverage
        self._voxel_dendrite_filter_active = coverage.labeled_neuron_count > 0
        if coverage.input_neuron_count == 0:
            message = "No neurons were available in the selected input scope."
        elif coverage.labeled_neuron_count == 0:
            message = (
                f"0 of {coverage.input_neuron_count:,} input neurons contain "
                "dendrite-type nodes; no restriction was applied."
            )
        else:
            message = (
                "Dendrite-label restriction active: "
                f"{coverage.labeled_neuron_count:,} of "
                f"{coverage.input_neuron_count:,} input neurons retained; "
                f"{coverage.excluded_neuron_count:,} lacking represented type "
                "3/4 labels excluded. This detects label presence, not label "
                "correctness."
            )
        self._voxel_dendrite_status_label.setText(message)
        self._progress_label.setText(message)
        self._update_voxel_node_filter_controls()

    def _selected_voxel_node_filter(self):
        """Return validated voxel-row filtering settings from the current UI."""
        from ..analysis.voxel_filter import VoxelNodeFilter

        if self._clustering_method_combo.currentText() != _CLUSTER_METHOD_VOXEL:
            return None
        current_data = getattr(self._voxel_node_type_mode_combo, "currentData", None)
        mode = str(current_data() or "all") if callable(current_data) else "all"
        selected_types: tuple[int, ...] = ()
        if mode != "all":
            selected = self._voxel_node_type_combo.selected_node_types()
            if selected is None or not selected:
                raise ValueError(
                    "Select at least one specific node type or turn the "
                    "node-type filter off."
                )
            selected_types = tuple(int(value) for value in selected)

        distance = None
        if self._voxel_soma_distance_enabled_cb.isChecked():
            distance = float(self._voxel_soma_distance_spin.value())

        require_dendrite_labels = bool(self._voxel_dendrite_filter_active)
        dendrite_types = self._available_dendrite_node_types()
        settings = VoxelNodeFilter(
            node_type_mode=mode,
            node_types=selected_types,
            require_dendrite_labels=require_dendrite_labels,
            exclude_within_soma_um=distance,
            dendrite_node_types=dendrite_types or (3, 4),
        )
        return None if settings.is_empty else settings

    def _run_clustering_pipeline(self) -> None:
        """Validate and start an exact asynchronous clustering preflight."""
        if self._db is None or self._atlas is None:
            return

        if self._worker_thread is not None and self._worker_thread.isRunning():
            return

        proceed, base_file_ids, _scope_label, _input_count = (
            self._resolve_cluster_query_file_scope()
        )
        if not proceed:
            return

        coordinate_space = self._current_coordinate_space()
        clustering_method = self._clustering_method_combo.currentText()
        scope = self._selected_cluster_region_scope()
        region_filter = self._selected_cluster_region_filter()
        region_selection = None
        if region_filter is not None:
            region_selection = region_filter.legacy_selection()
        elif coordinate_space == _COORD_SPACE_CCF:
            region_selection = self._selected_cluster_region_selection()

        has_includes = bool(
            region_filter.include_rules
            if region_filter is not None
            else region_selection is not None
        )
        if not has_includes and scope == _ANALYSIS_SCOPE_WHOLE:
            self._progress_label.setText("Select at least one included region.")
            return
        if region_filter is not None:
            empty_rules = [
                rule.acronym
                for rule in (
                    *region_filter.include_rules,
                    *region_filter.exclude_rules,
                )
                if not rule.represented_region_ids
            ]
            if empty_rules:
                self._progress_label.setText(
                    "Selected region(s) have no represented dataset regions: "
                    + ", ".join(empty_rules)
                )
                return
        elif (
            region_selection is not None and not region_selection.represented_region_ids
        ):
            self._progress_label.setText(
                "Selected region(s) have no represented dataset regions."
            )
            return

        flatmap_style = None
        if coordinate_space == _COORD_SPACE_FLATMAP:
            flatmap_style = self._selected_flatmap_style()
            if flatmap_style is None:
                self._progress_label.setText(
                    "No flatmap style available in the loaded Parquet."
                )
                return

        algorithm = {
            "Hierarchical": "hierarchical",
            "K-Means": "kmeans",
            "DBSCAN": "dbscan",
        }[self._algorithm_combo.currentText()]
        try:
            voxel_node_filter = self._selected_voxel_node_filter()
        except ValueError as error:
            self._progress_label.setText(str(error))
            return
        request = _ClusteringRequest(
            coordinate_space=coordinate_space,
            clustering_method=clustering_method,
            scope=scope,
            file_ids=(
                None
                if base_file_ids is None
                else tuple(str(file_id) for file_id in base_file_ids)
            ),
            region_selection=region_selection,
            region_filter=region_filter,
            dilation_fraction=(
                self._dilation_spin.value() / 100.0
                if region_selection is not None and region_filter is None
                else 0.0
            ),
            linkage_method=self._method_combo.currentText(),
            n_clusters=int(self._n_clusters_spin.value()),
            algorithm=algorithm,
            eps=float(self._eps_spin.value()),
            min_samples=int(self._min_samples_spin.value()),
            flatmap_style=flatmap_style,
            flatmap_y_bins=int(self._flatmap_y_bins_spin.value()),
            flatmap_depth_bin_um=float(self._flatmap_depth_bin_spin.value()),
            flatmap_include_depth_minus_one=(
                self._flatmap_include_depth_minus_one_cb.isChecked()
            ),
            flatmap_depth_scale=float(self._flatmap_depth_scale_spin.value()),
            flatmap_include_depth=(not self._flatmap_ignore_depth_cb.isChecked()),
            voxel_node_filter=voxel_node_filter,
        )
        self._capture_cluster_run_context(
            clustering_method,
            base_file_ids,
        )
        self._start_clustering_preflight(request)

    def _start_clustering_preflight(self, request: _ClusteringRequest) -> None:
        """Count one immutable clustering request on a background thread."""
        from ..workers import ClusteringPreflightWorker

        worker = ClusteringPreflightWorker(
            parquet_path=self._parquet_path,
            atlas=self._atlas,
            coordinate_space=(
                "flatmap" if request.coordinate_space == _COORD_SPACE_FLATMAP else "ccf"
            ),
            clustering_method=(
                "soma" if request.clustering_method == _CLUSTER_METHOD_SOMA else "voxel"
            ),
            region_selection=request.region_selection,
            region_filter=request.region_filter,
            dilation_fraction=request.dilation_fraction,
            file_ids=(None if request.file_ids is None else list(request.file_ids)),
            flatmap_style=request.flatmap_style,
            flatmap_y_bins=request.flatmap_y_bins,
            flatmap_depth_bin_um=request.flatmap_depth_bin_um,
            flatmap_include_depth_minus_one=(request.flatmap_include_depth_minus_one),
            flatmap_collapse_depth=not request.flatmap_include_depth,
            voxel_node_filter=request.voxel_node_filter,
        )
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_clustering_preflight_finished)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_clustering_preflight_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(self._on_clustering_preflight_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._pending_clustering_request = request
        self._pending_clustering_preflight = None
        self._worker_thread = thread
        self._current_worker = worker
        self._progress_label.setText("Counting clustering nodes...")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._update_button_states()
        thread.start()

    def _on_clustering_preflight_finished(self, result) -> None:
        """Store a preflight result until its worker thread has stopped."""
        self._pending_clustering_preflight = result

    def _on_clustering_preflight_error(self, message: str) -> None:
        """Stop a clustering request when exact preflight counting fails."""
        self._pending_clustering_request = None
        self._pending_clustering_preflight = None
        self._pending_cluster_context = {}
        self._on_error(message)

    def _on_clustering_preflight_thread_finished(self) -> None:
        """Prompt when needed, then launch the snapshotted clustering run."""
        self._on_thread_finished()
        request = self._pending_clustering_request
        result = self._pending_clustering_preflight
        self._pending_clustering_request = None
        self._pending_clustering_preflight = None
        if request is None or result is None:
            self._update_button_states()
            return

        node_count = int(result.node_count)
        if (
            node_count > _LARGE_CLUSTER_NODE_THRESHOLD
            and not self._confirm_large_clustering_run(node_count)
        ):
            self._pending_cluster_context = {}
            self._progress_bar.setVisible(False)
            self._progress_label.setText(
                f"Clustering cancelled; {node_count:,} nodes would have been processed."
            )
            self._update_button_states()
            return
        prepared_region_filter = getattr(result, "prepared_region_filter", None)
        prepared_voxel_filter = getattr(result, "prepared_voxel_filter", None)
        if prepared_voxel_filter is not None:
            self._launch_clustering_request(
                request,
                result.voxel_id_map,
                prepared_region_filter,
                prepared_voxel_filter,
            )
        elif prepared_region_filter is not None:
            self._launch_clustering_request(
                request,
                result.voxel_id_map,
                prepared_region_filter,
            )
        else:
            self._launch_clustering_request(request, result.voxel_id_map)

    def _confirm_large_clustering_run(self, node_count: int) -> bool:
        """Ask whether a clustering input above the warning threshold may run."""
        from qtpy.QtWidgets import QMessageBox

        message = QMessageBox(self)
        message.setIcon(QMessageBox.Warning)
        message.setWindowTitle("Large Clustering Run")
        message.setText(_large_clustering_warning_text(node_count))
        continue_button = message.addButton("Continue", QMessageBox.AcceptRole)
        cancel_button = message.addButton("Cancel", QMessageBox.RejectRole)
        message.setDefaultButton(cancel_button)
        message.exec()
        return message.clickedButton() is continue_button

    def _launch_clustering_request(
        self,
        request: _ClusteringRequest,
        voxel_id_map: np.ndarray | None,
        prepared_region_filter=None,
        prepared_voxel_filter=None,
    ) -> None:
        """Launch a previously counted immutable clustering request."""
        from ..workers import (
            CorrelationWorker,
            FlatmapParquetCorrelationWorker,
            FlatmapSomaClusterWorker,
            SomaClusterWorker,
        )

        file_ids = None if request.file_ids is None else list(request.file_ids)
        common = {
            "parquet_path": self._parquet_path,
            "atlas": self._atlas,
        }
        filter_kwargs = {}
        if request.region_filter is not None:
            filter_kwargs["region_filter"] = request.region_filter
        if prepared_region_filter is not None:
            filter_kwargs["prepared_region_filter"] = prepared_region_filter
        voxel_filter_kwargs = {}
        if request.voxel_node_filter is not None:
            voxel_filter_kwargs["voxel_node_filter"] = request.voxel_node_filter
        if prepared_voxel_filter is not None:
            voxel_filter_kwargs["prepared_voxel_filter"] = prepared_voxel_filter
        if request.coordinate_space == _COORD_SPACE_FLATMAP:
            if request.clustering_method == _CLUSTER_METHOD_SOMA:
                worker = FlatmapSomaClusterWorker(
                    **common,
                    style=request.flatmap_style,
                    algorithm=request.algorithm,
                    linkage_method=request.linkage_method,
                    n_clusters=request.n_clusters,
                    eps=request.eps,
                    min_samples=request.min_samples,
                    file_ids=file_ids,
                    depth_scale=request.flatmap_depth_scale,
                    include_depth=request.flatmap_include_depth,
                    **filter_kwargs,
                )
            else:
                worker = FlatmapParquetCorrelationWorker(
                    **common,
                    style=request.flatmap_style,
                    y_bins=request.flatmap_y_bins,
                    depth_bin_um=request.flatmap_depth_bin_um,
                    include_depth_minus_one=(request.flatmap_include_depth_minus_one),
                    linkage_method=request.linkage_method,
                    n_clusters=request.n_clusters,
                    file_ids=file_ids,
                    collapse_depth=not request.flatmap_include_depth,
                    **filter_kwargs,
                    **voxel_filter_kwargs,
                )
        elif request.clustering_method == _CLUSTER_METHOD_SOMA:
            worker = SomaClusterWorker(
                **common,
                region_selection=request.region_selection,
                dilation_fraction=request.dilation_fraction,
                algorithm=request.algorithm,
                linkage_method=request.linkage_method,
                n_clusters=request.n_clusters,
                eps=request.eps,
                min_samples=request.min_samples,
                file_ids=file_ids,
                voxel_id_map=voxel_id_map,
                **filter_kwargs,
            )
        else:
            worker = CorrelationWorker(
                **common,
                region_selection=request.region_selection,
                dilation_fraction=request.dilation_fraction,
                linkage_method=request.linkage_method,
                n_clusters=request.n_clusters,
                file_ids=file_ids,
                voxel_id_map=voxel_id_map,
                **filter_kwargs,
                **voxel_filter_kwargs,
            )
        self._start_background_worker(worker, self._on_correlation_finished)

    def _capture_cluster_run_context(
        self,
        clustering_method: str,
        file_ids: list[str] | None,
    ) -> None:
        """Capture UI-only cohort and lineage provenance for a worker run."""
        scope = self._selected_cluster_region_scope()
        parent_assignment_id = None
        parent_cluster_ids: list[int] = []
        store = self.__dict__.get("_cluster_assignment_store")
        active = getattr(store, "active", None)
        if scope == _ANALYSIS_SCOPE_SELECTED and active is not None:
            parent_assignment_id = active.assignment_id
            parent_cluster_ids = sorted(
                {
                    int(label)
                    for file_id in file_ids or []
                    if (label := active.label_for(file_id)) is not None
                }
            )
        self._pending_cluster_context = {
            "method_name": str(clustering_method),
            "input_scope": scope,
            "input_file_ids": None if file_ids is None else list(file_ids),
            "coordinate_space": self._current_coordinate_space(),
            "parent_assignment_id": parent_assignment_id,
            "parent_cluster_ids": parent_cluster_ids,
        }

    def _start_background_worker(self, worker, finished_slot) -> None:
        """Wire up and start a background worker in a QThread."""
        thread = QThread()
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(finished_slot)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(self._update_button_states)
        thread.finished.connect(thread.deleteLater)

        # Keep references to prevent garbage collection
        self._worker_thread = thread
        self._current_worker = worker

        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._update_button_states()

        thread.start()

    def _run_heatmap_pipeline(self) -> None:
        """Start the heatmap pipeline in a background thread."""
        if self._db is None or self._atlas is None:
            return

        if self._worker_thread is not None and self._worker_thread.isRunning():
            return

        request = self._selected_heatmap_request()
        if request is None:
            return

        self._start_heatmap_requests([request], batch_mode=False)

    def _run_all_cluster_heatmaps(self) -> None:
        """Queue heatmaps for every cluster shown in the dropdown."""
        if self._db is None or self._atlas is None:
            return

        if self._worker_thread is not None and self._worker_thread.isRunning():
            return

        requests = self._all_cluster_heatmap_requests()
        if not requests:
            self._progress_label.setText(
                "Run clustering before adding cluster heatmaps."
            )
            return

        self._start_heatmap_requests(requests, batch_mode=True)

    def _current_heatmap_region_filter(
        self,
    ) -> tuple[int | None, str | None, tuple[int, ...] | None] | None:
        """Return the current region filter for heatmap creation."""
        selected_region = self._selected_heat_region()
        if selected_region is None:
            return (None, None, None)

        region_id, acronym = selected_region
        represented_ids = tuple(self._represented_region_ids_for_selection(region_id))
        if not represented_ids:
            self._progress_label.setText(
                "Selected region has no represented dataset regions."
            )
            return None
        return (int(region_id), acronym, represented_ids)

    def _cluster_file_ids(self, cluster_label: int) -> tuple[str, ...]:
        """Return neuron IDs assigned to one cluster label."""
        store = self.__dict__.get("_cluster_assignment_store")
        assignment = getattr(store, "active", None)
        if assignment is not None:
            return assignment.file_ids_for_label(cluster_label)
        result = self._last_cluster_result
        if result is None:
            return ()

        mask = np.asarray(result.labels) == int(cluster_label)
        return tuple(
            str(neuron_id)
            for neuron_id, matched in zip(result.neuron_ids, mask)
            if matched
        )

    def _selected_heatmap_node_types(self) -> tuple[int, ...] | None:
        """Return the Analysis heatmap node-type filter."""
        combo = getattr(self, "_heat_node_type_combo", None)
        getter = getattr(combo, "selected_node_types", None)
        if callable(getter):
            return getter()
        return None

    def _selected_heatmap_soma_radius_um(self) -> float | None:
        """Return the Analysis heatmap soma-radius filter."""
        checkbox = getattr(self, "_heat_soma_radius_enabled_cb", None)
        is_checked = getattr(checkbox, "isChecked", None)
        if not callable(is_checked) or not is_checked():
            return None

        spin = getattr(self, "_heat_soma_radius_spin", None)
        value_getter = getattr(spin, "value", None)
        if not callable(value_getter):
            return None
        radius = float(value_getter())
        if radius <= 0.0:
            return None
        return radius

    def _build_heatmap_request(
        self,
        cluster_label: int | None,
        *,
        depth_axis: int | None = None,
        depth_bin_factor: int | None = None,
    ) -> _HeatmapRequest | None:
        """Build one heatmap request from current UI state."""
        region_filter = self._current_heatmap_region_filter()
        if region_filter is None:
            return None

        selected_region_id, selected_region_acronym, region_ids = region_filter
        request_cluster = None if cluster_label is None else int(cluster_label)
        file_ids = None
        if request_cluster is not None:
            file_ids = self._cluster_file_ids(request_cluster)
            if not file_ids:
                self._progress_label.setText(
                    f"No neurons found for cluster {request_cluster}."
                )
                return None

        resolved_depth_axis = (
            self._current_depth_axis() if depth_axis is None else int(depth_axis)
        )
        resolved_depth_bin = (
            self._depth_bin_spin.value()
            if depth_bin_factor is None
            else int(depth_bin_factor)
        )

        assignment = None
        if request_cluster is not None:
            store = self.__dict__.get("_cluster_assignment_store")
            assignment = getattr(store, "active", None)

        return _HeatmapRequest(
            selected_region_id=selected_region_id,
            selected_region_acronym=selected_region_acronym,
            region_ids=region_ids,
            cluster_label=request_cluster,
            file_ids=file_ids,
            node_types=self._selected_heatmap_node_types(),
            soma_radius_um=self._selected_heatmap_soma_radius_um(),
            depth_bin_factor=resolved_depth_bin,
            depth_axis=resolved_depth_axis,
            assignment_id=(
                str(assignment.assignment_id) if assignment is not None else None
            ),
            assignment_name=(str(assignment.name) if assignment is not None else None),
        )

    def _selected_heatmap_request(self) -> _HeatmapRequest | None:
        """Return the heatmap request for the current dropdown selection."""
        cluster_label = None
        cluster_idx = self._heat_cluster_combo.currentIndex()
        if cluster_idx > 0:
            cluster_label = self._heat_cluster_combo.itemData(cluster_idx)
        return self._build_heatmap_request(cluster_label)

    def _all_cluster_heatmap_labels(self) -> list[int]:
        """Return all concrete cluster labels shown in the heatmap dropdown."""
        labels: list[int] = []
        seen: set[int] = set()
        combo = getattr(self, "_heat_cluster_combo", None)
        if combo is not None:
            for index in range(1, combo.count()):
                label = combo.itemData(index)
                if label is None:
                    continue
                value = int(label)
                if value in seen:
                    continue
                seen.add(value)
                labels.append(value)
        if labels:
            return labels

        result = self._last_cluster_result
        if result is None:
            return []
        return [int(label) for label in sorted(np.unique(result.labels).tolist())]

    def on_active_cluster_assignment_changed(self) -> None:
        """Synchronize Analysis controls with the shared active assignment."""
        store = self.__dict__.get("_cluster_assignment_store")
        assignment = getattr(store, "active", None)
        if assignment is None:
            self._last_cluster_result = None
            self._cluster_label_colors = {}
            self._cluster_color_map = None
            self._actual_n_clusters = 0
            combo = getattr(self, "_heat_cluster_combo", None)
            if combo is not None:
                self._update_cluster_filter_combo()
            self._update_button_states()
            return
        self._last_cluster_result = assignment.runtime_result
        self._cluster_label_colors = {
            int(label): list(color) for label, color in assignment.label_colors.items()
        }
        self._cluster_color_map = {
            file_id: list(
                assignment.label_colors.get(
                    int(label),
                    [0.5, 0.5, 0.5, 1.0],
                )
            )
            for file_id, label in assignment.assignments.items()
        }
        self._actual_n_clusters = len(set(assignment.assignments.values()))
        combo = getattr(self, "_heat_cluster_combo", None)
        if combo is not None:
            self._update_cluster_filter_combo()
        self._update_button_states()
        if assignment.runtime_result is None:
            message = (
                f"{assignment.name} assignments restored. Rerun required for "
                "dendrogram and distance exports."
            )
            progress = getattr(self, "_progress_label", None)
            if progress is not None:
                progress.setText(message)
            clustermap_status = getattr(self, "_clustermap_status_label", None)
            if clustermap_status is not None:
                clustermap_status.setText(message)

    def _all_cluster_heatmap_requests(self) -> list[_HeatmapRequest]:
        """Return heatmap requests for every cluster-specific option."""
        depth_axis = self._current_depth_axis()
        depth_bin_factor = self._depth_bin_spin.value()
        requests: list[_HeatmapRequest] = []
        for cluster_label in self._all_cluster_heatmap_labels():
            request = self._build_heatmap_request(
                cluster_label,
                depth_axis=depth_axis,
                depth_bin_factor=depth_bin_factor,
            )
            if request is not None:
                requests.append(request)
        return requests

    def _start_heatmap_requests(
        self,
        requests: list[_HeatmapRequest],
        *,
        batch_mode: bool,
    ) -> None:
        """Start one or more heatmap requests in sequence."""
        if not requests:
            return

        self._pending_heatmap_requests = list(requests)
        self._completed_heatmap_requests = []
        self._current_heatmap_request = None
        self._active_heatmap_total = len(requests)
        self._active_heatmap_index = 0
        self._heatmap_batch_mode = bool(batch_mode)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._update_button_states()
        self._start_next_heatmap_request()

    def _start_next_heatmap_request(self) -> None:
        """Start the next queued heatmap request."""
        if not self._pending_heatmap_requests:
            return

        from ..workers import HeatmapWorker

        request = self._pending_heatmap_requests.pop(0)
        self._current_heatmap_request = request
        self._active_heatmap_index = len(self._completed_heatmap_requests) + 1

        worker = HeatmapWorker(
            parquet_path=self._parquet_path,
            atlas=self._atlas,
            region_ids=(
                list(request.region_ids) if request.region_ids is not None else None
            ),
            file_ids=(list(request.file_ids) if request.file_ids is not None else None),
            node_types=(
                list(request.node_types) if request.node_types is not None else None
            ),
            soma_radius_um=request.soma_radius_um,
            depth_bin_factor=request.depth_bin_factor,
            depth_axis=request.depth_axis,
        )

        thread = QThread()
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_heatmap_progress)
        worker.finished.connect(self._on_heatmap_finished)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_heatmap_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(self._on_heatmap_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._worker_thread = thread
        self._current_worker = worker

        if self._heatmap_batch_mode:
            self._progress_label.setText(
                f"Building heatmap {self._active_heatmap_index}/"
                f"{self._active_heatmap_total}: "
                f"{self._heatmap_layer_name(request)}"
            )
        else:
            self._progress_label.setText("Building heatmap volume...")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._update_button_states()

        thread.start()

    def _on_progress(self, step_name: str, current: int, total: int) -> None:
        """Handle progress updates from workers."""
        self._progress_label.setText(f"Step {current}/{total}: {step_name}")
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(current)

    def _on_heatmap_progress(
        self,
        step_name: str,
        current: int,
        total: int,
    ) -> None:
        """Handle progress updates for queued heatmap builds."""
        request = self._current_heatmap_request
        if not self._heatmap_batch_mode or request is None:
            self._on_progress(step_name, current, total)
            return

        overall_total = max(int(total) * self._active_heatmap_total, 1)
        overall_current = ((self._active_heatmap_index - 1) * int(total)) + int(current)
        self._progress_label.setText(
            f"Heatmap {self._active_heatmap_index}/"
            f"{self._active_heatmap_total}: "
            f"{self._heatmap_layer_name(request)} - {step_name}"
        )
        self._progress_bar.setRange(0, overall_total)
        self._progress_bar.setValue(overall_current)

    def _on_correlation_finished(self, result: ClusterResult) -> None:
        """Handle completed correlation pipeline."""
        finish_start = perf_counter()
        distance_matrix = getattr(result, "distance_matrix", None)
        logger.debug(
            "_on_correlation_finished start: neurons=%d distance_shape=%s distance_dtype=%s distance_nbytes=%s",
            len(result.neuron_ids),
            getattr(distance_matrix, "shape", None),
            getattr(distance_matrix, "dtype", None),
            getattr(distance_matrix, "nbytes", None),
        )
        self._last_cluster_result = result
        color_map_start = perf_counter()
        self._build_cluster_color_map()
        self._save_cluster_assignment(result)
        color_summary = self._color_neurons_by_cluster()
        logger.debug(
            "_on_correlation_finished color map built: elapsed=%.3fs actual_clusters=%d",
            perf_counter() - color_map_start,
            self._actual_n_clusters,
        )
        self._progress_bar.setVisible(False)

        requested_k = self._n_clusters_spin.value()
        actual_k = self._actual_n_clusters
        if actual_k < requested_k:
            cluster_msg = f"{actual_k} of {requested_k} requested clusters found"
        else:
            cluster_msg = f"{actual_k} clusters"
        progress_message = (
            f"Clustering complete: {len(result.neuron_ids)} neurons, "
            f"{cluster_msg}. Click Render Dendrogram to view. "
            "Table updated and sorted by cluster."
        )
        unassigned_count = len(getattr(result, "unassigned_neuron_ids", ()) or ())
        progress_message += (
            f" {unassigned_count} neuron(s) are unclustered after "
            "region/coordinate filtering."
        )
        metadata = getattr(result, "metadata", None)
        extra_metadata = getattr(metadata, "extra_metadata", {})
        voxel_filter_metadata = (
            extra_metadata.get("voxel_node_filter")
            if isinstance(extra_metadata, dict)
            else None
        )
        if isinstance(voxel_filter_metadata, dict):
            missing_soma_count = voxel_filter_metadata.get("missing_soma_neuron_count")
            if missing_soma_count:
                progress_message += (
                    f" {int(missing_soma_count):,} neuron(s) lacked a valid soma "
                    "and were excluded by soma-distance filtering."
                )
        if color_summary.colored_count > 0 and color_summary.rendered_count > 0:
            neuron_word = "neuron" if color_summary.rendered_count == 1 else "neurons"
            progress_message += (
                f" Auto-colored {color_summary.colored_count}/"
                f"{color_summary.rendered_count} rendered {neuron_word} "
                "by cluster."
            )
            if color_summary.gray_count > 0:
                progress_message += f" {color_summary.gray_count} shown in gray."
        self._progress_label.setText(progress_message)
        clustermap_message = (
            "Clustering complete. Click 'Build Dendrogram' to render the cluster map."
        )
        clustermap_status = getattr(self, "_clustermap_status_label", None)
        if clustermap_status is not None:
            clustermap_status.setText(clustermap_message)
        self._update_cluster_filter_combo()
        self._update_button_states()
        show_placeholder = "_show_clustermap_message" in self.__dict__ or (
            hasattr(self, "_figure") and hasattr(self, "_canvas")
        )
        if show_placeholder:
            self._show_clustermap_message(
                "Clustering complete. Click Render Dendrogram to view."
            )
        logger.debug(
            "_on_correlation_finished worker result ready; clustermap render deferred until button click"
        )
        if self._cluster_color_map is not None:
            try:
                self.cluster_colors_updated.emit(result, self._cluster_color_map)
            except RuntimeError:
                pass
        logger.debug(
            "_on_correlation_finished complete: total_elapsed=%.3fs",
            perf_counter() - finish_start,
        )

    def _save_cluster_assignment(self, result: ClusterResult) -> None:
        """Persist labels and provenance for one completed clustering run."""
        store = self.__dict__.get("_cluster_assignment_store")
        if store is None:
            return
        context = dict(self.__dict__.get("_pending_cluster_context", {}))
        method_name = str(
            context.get("method_name")
            or getattr(self._clustering_method_combo, "currentText", lambda: "")()
            or "Cluster Assignment"
        )
        metadata = getattr(result, "metadata", None)
        if metadata is not None and hasattr(metadata, "to_dict"):
            run_metadata = metadata.to_dict()
        else:
            run_metadata = {}
        run_metadata.update(
            {
                "input_scope": str(context.get("input_scope") or "whole"),
                "coordinate_space": str(
                    context.get("coordinate_space") or _COORD_SPACE_CCF
                ),
            }
        )
        store.add_result(
            result,
            method_name=method_name,
            input_file_ids=context.get("input_file_ids"),
            label_colors=getattr(self, "_cluster_label_colors", {}),
            run_metadata=run_metadata,
            input_scope=str(context.get("input_scope") or "whole"),
            coordinate_space=str(context.get("coordinate_space") or _COORD_SPACE_CCF),
            parent_assignment_id=context.get("parent_assignment_id"),
            parent_cluster_ids=context.get("parent_cluster_ids", ()),
        )
        self._pending_cluster_context = {}

    def _on_heatmap_finished(self, volume: np.ndarray) -> None:
        """Handle completed heatmap pipeline."""
        request = self._current_heatmap_request
        if request is None:
            return

        layer = self._add_analysis_heatmap_layer(volume, request)
        self._completed_heatmap_requests.append(request)
        self._heatmap_layer = layer

        if self._heatmap_batch_mode:
            self._progress_label.setText(
                f"Added {self._heatmap_layer_name(request)} "
                f"({self._active_heatmap_index}/"
                f"{self._active_heatmap_total})"
            )
            return

        self._progress_label.setText(
            f"{layer.name}: {(volume > 0).sum():,} non-zero voxels"
        )

    def _heatmap_layer_name(self, request: _HeatmapRequest) -> str:
        """Return the napari layer name for one heatmap request."""
        region_part = (
            f" {request.selected_region_acronym}"
            if request.selected_region_acronym
            else ""
        )
        filter_parts: list[str] = []
        if request.node_types is not None:
            filter_parts.append(
                NodeTypeSelectorComboBox.selection_text(request.node_types)
            )
        if request.soma_radius_um is not None:
            filter_parts.append(f"{request.soma_radius_um:g} μm soma radius")
        filter_part = f" ({', '.join(filter_parts)})" if filter_parts else ""
        if request.cluster_label is None:
            return f"Node Count{region_part}{filter_part} Heatmap"
        return f"Cluster {request.cluster_label}{region_part}{filter_part} Heatmap"

    def _add_analysis_heatmap_layer(
        self,
        volume: np.ndarray,
        request: _HeatmapRequest,
    ):
        """Add or replace one analysis heatmap layer."""
        from napari.utils.colormaps import Colormap

        layer_name = self._heatmap_layer_name(request)
        contrast_limits = _analysis_heatmap_contrast_limits(volume)

        if request.cluster_label is not None:
            rgba = getattr(self, "_cluster_label_colors", {}).get(
                request.cluster_label, [0.5, 0.5, 0.5, 1.0]
            )
            colormap = Colormap(
                colors=[[0, 0, 0, 0], [rgba[0], rgba[1], rgba[2], 1.0]],
                name=f"cluster_{request.cluster_label}",
            )
        else:
            colormap = "hot"

        for layer in list(self._viewer.layers):
            if layer.name == layer_name:
                self._viewer.layers.remove(layer)

        scale = [1.0, 1.0, 1.0]
        scale[request.depth_axis] = float(request.depth_bin_factor)
        source_file_ids = (
            list(request.file_ids) if request.file_ids is not None else None
        )
        atlas_version = getattr(self._atlas, "version", None)
        atlas_provenance = {
            "atlas_name": getattr(self._atlas, "atlas_name", None),
            "resolution_um": [
                float(value)
                for value in getattr(self._atlas, "resolution", (1.0, 1.0, 1.0))
            ],
            "atlas_version": (None if atlas_version is None else str(atlas_version)),
        }
        metadata = {
            "heatmap_source": True,
            "heatmap_native_grid": request.depth_bin_factor == 1,
            "atlas_name": getattr(self._atlas, "atlas_name", None),
            "heatmap_kind": "analysis",
            "heatmap_region": request.selected_region_acronym,
            "heatmap_selected_region_id": request.selected_region_id,
            "heatmap_selected_region_acronym": request.selected_region_acronym,
            "heatmap_region_ids": (
                list(request.region_ids) if request.region_ids is not None else None
            ),
            "heatmap_node_types": (
                list(request.node_types) if request.node_types is not None else None
            ),
            "heatmap_node_type_labels": (
                NodeTypeSelectorComboBox.metadata_labels(request.node_types)
            ),
            "heatmap_soma_radius_um": request.soma_radius_um,
            "heatmap_cluster": request.cluster_label,
            "file_ids": source_file_ids,
            "source_file_ids": source_file_ids,
            "depth_bin_factor": request.depth_bin_factor,
            "depth_axis": request.depth_axis,
            "heatmap_contrast_limits": contrast_limits,
            "heatmap_autocontrast_policy": "stable_full_volume",
            # Stable machine-readable provenance used by the comparison board.
            # The UUID identifies this exact layer payload; assignment identity
            # remains separate so several cluster layers can form one set.
            "comparison_source_id": uuid4().hex,
            "comparison_assignment_id": request.assignment_id,
            "comparison_assignment_name": request.assignment_name,
            "comparison_atlas_provenance": atlas_provenance,
            "comparison_filter_signature": {
                "atlas_provenance": atlas_provenance,
                "region_ids": (
                    sorted({int(value) for value in request.region_ids})
                    if request.region_ids is not None
                    else None
                ),
                "node_types": (
                    sorted({int(value) for value in request.node_types})
                    if request.node_types is not None
                    else None
                ),
                "soma_radius_um": request.soma_radius_um,
                "depth_axis": int(request.depth_axis),
                "depth_bin_factor": int(request.depth_bin_factor),
            },
        }

        layer = self._viewer.add_image(
            volume,
            name=layer_name,
            colormap=colormap,
            blending="additive",
            rendering="mip",
            opacity=0.7,
            visible=True,
            scale=scale,
            contrast_limits=contrast_limits,
            metadata=metadata,
        )
        _install_analysis_heatmap_layer_workarounds(layer)
        return layer

    def _on_heatmap_thread_finished(self) -> None:
        """Advance the heatmap queue or finalize the completed run."""
        self._on_thread_finished()

        if self._pending_heatmap_requests:
            self._start_next_heatmap_request()
            return

        if self._completed_heatmap_requests:
            self._last_heatmap_requests = list(self._completed_heatmap_requests)

        completed_count = len(self._completed_heatmap_requests)
        batch_mode = self._heatmap_batch_mode
        self._current_heatmap_request = None
        self._completed_heatmap_requests = []
        self._active_heatmap_total = 0
        self._active_heatmap_index = 0
        self._heatmap_batch_mode = False
        self._progress_bar.setVisible(False)
        if batch_mode and completed_count > 0:
            suffix = "" if completed_count == 1 else "s"
            self._progress_label.setText(
                f"Added {completed_count} cluster heatmap{suffix} to scene"
            )
        self._update_button_states()

    def _on_heatmap_error(self, message: str) -> None:
        """Handle a heatmap worker failure and stop any remaining queue."""
        if self._completed_heatmap_requests:
            self._last_heatmap_requests = list(self._completed_heatmap_requests)
        self._pending_heatmap_requests = []
        self._current_heatmap_request = None
        self._completed_heatmap_requests = []
        self._active_heatmap_total = 0
        self._active_heatmap_index = 0
        self._heatmap_batch_mode = False
        self._on_error(message)

    def _on_error(self, message: str) -> None:
        """Handle worker errors."""
        self._progress_bar.setVisible(False)
        self._progress_label.setText(f"Error: {message}")
        self._update_button_states()
        logger.error(f"Analysis pipeline error: {message}")

    def _render_clustermap_requested(self) -> None:
        """Render the latest clustermap only when explicitly requested."""
        if self._last_cluster_result is None:
            return
        if (
            getattr(self, "_build_clustermap_btn", None) is not None
            and getattr(self, "_clustermap_status_label", None) is not None
        ):
            self._build_clustermap_on_demand()
            return
        self._draw_clustermap(self._last_cluster_result)

    def _show_clustermap_message(self, message: str) -> None:
        """Reset the embedded clustermap canvas to a placeholder message."""
        if not hasattr(self, "_figure") or not hasattr(self, "_canvas"):
            return
        self._clustermap_rendered = False
        clear = getattr(self._figure, "clear", None)
        if callable(clear):
            clear()
            add_subplot = getattr(self._figure, "add_subplot", None)
            if callable(add_subplot):
                ax = self._figure.add_subplot(111)
                text_kwargs = {
                    "ha": "center",
                    "va": "center",
                }
                transform = getattr(ax, "transAxes", None)
                if transform is not None:
                    text_kwargs["transform"] = transform
                ax.text(
                    0.5,
                    0.5,
                    message,
                    **text_kwargs,
                )
                axis_off = getattr(ax, "set_axis_off", None)
                if callable(axis_off):
                    axis_off()
        self._canvas.draw()

    def _clustermap_canvas_logical_pixel_size(self) -> tuple[int, int] | None:
        """Return the current canvas size in logical Qt pixels, if available."""
        width = height = None

        width_method = getattr(self._canvas, "width", None)
        if callable(width_method):
            try:
                width = int(width_method())
            except (TypeError, ValueError):
                width = None

        height_method = getattr(self._canvas, "height", None)
        if callable(height_method):
            try:
                height = int(height_method())
            except (TypeError, ValueError):
                height = None

        if width is None:
            try:
                width = int(getattr(self._canvas, "_width", 0))
            except (TypeError, ValueError):
                width = None
        if height is None:
            try:
                height = int(getattr(self._canvas, "_height", 0))
            except (TypeError, ValueError):
                height = None

        if width is None or height is None:
            size_attr = getattr(self._canvas, "size", None)
            size_obj = size_attr() if callable(size_attr) else size_attr
            if size_obj is not None:
                if width is None:
                    width_getter = getattr(size_obj, "width", None)
                    if callable(width_getter):
                        width = int(width_getter())
                if height is None:
                    height_getter = getattr(size_obj, "height", None)
                    if callable(height_getter):
                        height = int(height_getter())

        if width is None or height is None or width <= 0 or height <= 0:
            return None
        return (width, height)

    def _clustermap_canvas_physical_pixel_size(
        self,
    ) -> tuple[int, int] | None:
        """Return the current canvas size in physical pixels, if available."""
        get_width_height = getattr(self._canvas, "get_width_height", None)
        if callable(get_width_height):
            try:
                width, height = get_width_height(physical=True)
            except TypeError:
                width = height = None
            else:
                try:
                    width = int(width)
                    height = int(height)
                except (TypeError, ValueError):
                    width = height = None
                if width > 0 and height > 0:
                    return (width, height)

        logical_size = self._clustermap_canvas_logical_pixel_size()
        if logical_size is None:
            return None

        device_pixel_ratio = getattr(self._canvas, "device_pixel_ratio", 1.0)
        if callable(device_pixel_ratio):
            try:
                ratio = float(device_pixel_ratio())
            except (TypeError, ValueError):
                ratio = 1.0
        else:
            try:
                ratio = float(device_pixel_ratio)
            except (TypeError, ValueError):
                ratio = 1.0

        if not np.isfinite(ratio) or ratio <= 0.0:
            ratio = 1.0

        width, height = logical_size
        physical_width = int(round(width * ratio))
        physical_height = int(round(height * ratio))
        if physical_width <= 0 or physical_height <= 0:
            return None
        return (physical_width, physical_height)

    def _clustermap_preview_figsize(self, dpi: float) -> tuple[float, float] | None:
        """Return the preview figure size using backend-owned geometry."""
        get_size_inches = getattr(self._figure, "get_size_inches", None)
        if callable(get_size_inches):
            try:
                size_values = np.asarray(get_size_inches(), dtype=float).reshape(-1)
            except (TypeError, ValueError):
                size_values = np.array([], dtype=float)
            if size_values.size >= 2:
                width_in = float(size_values[0])
                height_in = float(size_values[1])
                if (
                    np.isfinite(width_in)
                    and np.isfinite(height_in)
                    and width_in > 0.0
                    and height_in > 0.0
                ):
                    return (width_in, height_in)

        size = self._clustermap_canvas_physical_pixel_size()
        if size is None:
            return None

        width_px, height_px = size
        if not np.isfinite(dpi) or dpi <= 0.0:
            return None
        return (max(width_px, 1) / dpi, max(height_px, 1) / dpi)

    def _draw_clustermap(self, result: ClusterResult) -> None:
        """Draw the clustermap preview into the embedded canvas."""
        draw_start = perf_counter()
        self._figure.clear()

        try:
            dpi_getter = getattr(self._figure, "get_dpi", None)
            dpi = (
                float(dpi_getter())
                if callable(dpi_getter)
                else float(getattr(self._figure, "dpi", 100.0) or 100.0)
            )
            figsize = self._clustermap_preview_figsize(dpi)
            physical_size = self._clustermap_canvas_physical_pixel_size()
            if figsize is None:
                figsize = (6.0, 6.0)
            logger.debug(
                "_draw_clustermap start: distance_shape=%s distance_dtype=%s distance_nbytes=%s linkage_shape=%s physical_canvas_size=%s figsize=%s dpi=%.3f",
                getattr(result.distance_matrix, "shape", None),
                getattr(result.distance_matrix, "dtype", None),
                getattr(result.distance_matrix, "nbytes", None),
                getattr(result.linkage_matrix, "shape", None),
                physical_size,
                figsize,
                dpi,
            )
            figure_start = perf_counter()
            _populate_embedded_clustermap_figure(
                self._figure,
                result,
                self._cluster_color_map,
                figsize=figsize,
                dpi=int(round(dpi)),
            )
            logger.debug(
                "_draw_clustermap populate_clustermap_figure complete: elapsed=%.3fs",
                perf_counter() - figure_start,
            )

            self._clustermap_rendered = True
            canvas_start = perf_counter()
            self._canvas.draw()
            logger.debug(
                "_draw_clustermap canvas draw complete: elapsed=%.3fs",
                perf_counter() - canvas_start,
            )
            logger.debug(
                "_draw_clustermap complete: total_elapsed=%.3fs",
                perf_counter() - draw_start,
            )

        except Exception as e:
            logger.exception("Failed to draw clustermap")
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Error drawing clustermap:\n{e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            self._clustermap_rendered = True
            self._canvas.draw()

    def _build_clustermap_on_demand(self) -> None:
        """Render the cached clustering result into the clustermap canvas."""
        result = self._last_cluster_result
        if result is None:
            self._clustermap_status_label.setText(
                "Run clustering before building the dendrogram."
            )
            return

        logger.debug(
            "_build_clustermap_on_demand start: neurons=%d distance_shape=%s",
            len(result.neuron_ids),
            getattr(result.distance_matrix, "shape", None),
        )
        self._build_clustermap_btn.setEnabled(False)
        self._clustermap_status_label.setText(
            f"Rendering dendrogram for {len(result.neuron_ids)} neurons..."
        )
        draw_start = perf_counter()
        try:
            self._draw_clustermap(result)
            self._clustermap_status_label.setText(
                f"Dendrogram ready for {len(result.neuron_ids)} neurons."
            )
        finally:
            self._update_button_states()
        logger.debug(
            "_build_clustermap_on_demand complete: elapsed=%.3fs",
            perf_counter() - draw_start,
        )

    def _build_cluster_color_map(self) -> None:
        """Build and cache the neuron_id -> RGBA color mapping from cluster results.

        Called once when clustering completes. The cached map is reused
        when rendered neurons are auto-colored after clustering and when
        cached colors are reapplied manually.
        """
        if self._last_cluster_result is None:
            self._cluster_color_map = None
            return

        start = perf_counter()
        result = self._last_cluster_result
        unique_labels = np.unique(result.labels)
        n_clusters = int(len(unique_labels))

        logger.info(
            f"Building cluster color map: {len(result.neuron_ids)} neurons, "
            f"{n_clusters} unique clusters (labels: {unique_labels.tolist()})"
        )

        # Use explicit colors for small cluster counts to guarantee
        # visually distinct colors; fall back to tab10/tab20 otherwise.
        _CUSTOM_COLORS: dict[int, list[list[float]]] = {
            1: [
                [0.12, 0.47, 0.71, 1.0],  # blue (all same cluster)
            ],
            2: [
                [0.12, 0.47, 0.71, 1.0],  # blue
                [0.84, 0.15, 0.16, 1.0],  # red
            ],
            3: [
                [0.12, 0.47, 0.71, 1.0],  # blue
                [0.84, 0.15, 0.16, 1.0],  # red
                [0.17, 0.63, 0.17, 1.0],  # green
            ],
        }

        # Map each unique label to a color index (0, 1, 2, ...) so colors
        # are assigned correctly even when labels are non-contiguous.
        label_to_idx = {int(lab): i for i, lab in enumerate(unique_labels)}

        color_map: dict[str, list[float]] = {}
        if n_clusters in _CUSTOM_COLORS:
            palette = _CUSTOM_COLORS[n_clusters]
            for neuron_id, label in zip(result.neuron_ids, result.labels):
                color_map[neuron_id] = palette[label_to_idx[int(label)]]
        else:
            cmap = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")
            for neuron_id, label in zip(result.neuron_ids, result.labels):
                idx = label_to_idx[int(label)]
                color_map[neuron_id] = list(cmap(idx / n_clusters))

        self._cluster_color_map = color_map
        self._actual_n_clusters = n_clusters
        # Build reverse map: cluster_label -> RGBA color (first neuron's color)
        self._cluster_label_colors: dict[int, list[float]] = {}
        for neuron_id, label in zip(result.neuron_ids, result.labels):
            lab = int(label)
            if lab not in self._cluster_label_colors:
                self._cluster_label_colors[lab] = color_map[neuron_id]
        logger.info(
            f"Built cluster color map: {len(color_map)} neurons, {n_clusters} clusters"
        )
        logger.debug(
            "_build_cluster_color_map complete: elapsed=%.3fs",
            perf_counter() - start,
        )

    def _update_cluster_filter_combo(self) -> None:
        """Populate the heatmap cluster filter dropdown with cluster options.

        Each item shows a color swatch icon, the cluster number, and the
        neuron count for that cluster.
        """
        self._heat_cluster_combo.clear()
        self._heat_cluster_combo.addItem("All neurons")

        store = self.__dict__.get("_cluster_assignment_store")
        assignment = getattr(store, "active", None)
        result = self._last_cluster_result
        if assignment is not None:
            unique_labels = sorted(set(assignment.assignments.values()))
            counts = {
                label: sum(value == label for value in assignment.assignments.values())
                for label in unique_labels
            }
        elif result is not None:
            unique_labels = sorted(np.unique(result.labels).tolist())
            counts = {
                int(label): int(np.sum(result.labels == label))
                for label in unique_labels
            }
        else:
            self._heat_cluster_combo.setEnabled(False)
            return

        for label in unique_labels:
            rgba = self._cluster_label_colors.get(label, [0.5, 0.5, 0.5, 1.0])
            count = int(counts[int(label)])

            # Create a small color swatch icon
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor.fromRgbF(rgba[0], rgba[1], rgba[2], rgba[3]))
            icon = QIcon(pixmap)

            self._heat_cluster_combo.addItem(
                icon, f"Cluster {label}  ({count} neurons)"
            )
            # Store the label as item data for retrieval
            self._heat_cluster_combo.setItemData(
                self._heat_cluster_combo.count() - 1, label
            )

        self._heat_cluster_combo.setEnabled(True)

    def _analysis_export_basename(self) -> str:
        """Return the basename used for analysis export defaults."""
        if self._parquet_path:
            stem = Path(self._parquet_path).stem
            if stem:
                return stem
        return "analysis"

    def _export_title(self) -> str:
        """Return the current export figure title."""
        return self._export_title_edit.text().strip()

    def _export_x_label(self) -> str:
        """Return the current export x-axis label."""
        return self._export_xlabel_edit.text().strip()

    def _selected_export_dpi(self) -> int:
        """Return the selected raster DPI preset."""
        index = self._export_dpi_combo.currentIndex()
        value = self._export_dpi_combo.itemData(index) if index >= 0 else None
        return int(value) if value is not None else 300

    def _prompt_analysis_export_path(
        self,
        dialog_title: str,
        default_name: str,
        file_filter: str,
    ) -> str | None:
        """Prompt for a save path for one analysis export."""
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            dialog_title,
            default_name,
            file_filter,
        )
        return output_path or None

    def _start_analysis_export(self, export_kind: str, output_path: str) -> None:
        """Start one workbook/parquet export in a background thread."""
        from ..workers import AnalysisExportWorker

        if self._last_cluster_result is None:
            return

        worker = AnalysisExportWorker(
            export_kind,
            output_path,
            self._last_cluster_result,
            self._cluster_color_map,
            figure_title=self._export_title(),
            x_label=self._export_x_label(),
            y_label="",
        )
        self._progress_label.setText(f"Saving {Path(output_path).name}...")
        self._start_background_worker(worker, self._on_export_finished)

    def _on_export_finished(self, output_path: str) -> None:
        """Handle a completed workbook/parquet export."""
        self._progress_bar.setVisible(False)
        self._progress_label.setText(f"Saved {Path(output_path).name}")
        self._update_button_states()

    def _save_cluster_workbook(self) -> None:
        """Prompt for and save the cluster workbook export."""
        output_path = self._prompt_analysis_export_path(
            "Save Cluster Workbook",
            f"{self._analysis_export_basename()}_clusters.xlsx",
            "Excel Files (*.xlsx);;All Files (*)",
        )
        if output_path:
            self._start_analysis_export("cluster_workbook", output_path)

    def _save_distance_workbook(self) -> None:
        """Prompt for and save the distance workbook export."""
        output_path = self._prompt_analysis_export_path(
            "Save Distance Workbook",
            f"{self._analysis_export_basename()}_distances.xlsx",
            "Excel Files (*.xlsx);;All Files (*)",
        )
        if output_path:
            self._start_analysis_export("distance_workbook", output_path)

    def _save_extended_parquet(self) -> None:
        """Prompt for and save the extended parquet export."""
        output_path = self._prompt_analysis_export_path(
            "Save Extended Parquet",
            f"{self._analysis_export_basename()}_cluster_analysis.parquet",
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if output_path:
            self._start_analysis_export("extended_parquet", output_path)

    def _save_dendrogram(self) -> None:
        """Prompt for and save the current clustermap-style dendrogram."""
        from ..analysis.export import save_dendrogram_figure

        if self._last_cluster_result is None:
            return

        dpi = self._selected_export_dpi()
        output_path = self._prompt_analysis_export_path(
            "Save Dendrogram",
            f"{self._analysis_export_basename()}_clustermap_{dpi}dpi.png",
            "PNG Files (*.png);;All Files (*)",
        )
        if not output_path:
            return

        self._progress_label.setText(f"Saving {Path(output_path).name}...")
        try:
            save_dendrogram_figure(
                output_path,
                self._last_cluster_result,
                self._cluster_color_map,
                title=self._export_title(),
                x_label=self._export_x_label(),
                y_label="",
                dpi=dpi,
            )
        except Exception as error:
            self._on_error(str(error))
            return

        self._progress_label.setText(f"Saved {Path(output_path).name}")

    def _current_depth_axis(self) -> int:
        """Return the current depth (slicing) axis from napari dims.

        The depth axis is the first non-displayed dimension. Falls back
        to axis 0 if all dimensions are displayed.
        """
        try:
            not_displayed = list(self._viewer.dims.not_displayed)
            if not_displayed:
                return int(not_displayed[0])
        except Exception:
            pass
        return 0

    def _update_voxel_depth_label(self, _value: int | None = None) -> None:
        """Update the voxel depth label based on atlas resolution and bin factor."""
        if self._atlas is None:
            self._voxel_depth_label.setText("Voxel depth: — μm")
            return
        resolution = float(self._atlas.resolution[0])
        depth = resolution * self._depth_bin_spin.value()
        self._voxel_depth_label.setText(f"Voxel depth: {depth:g} μm")

    def _on_dims_order_changed(self, event=None) -> None:
        """Rebuild heatmap when the user reorders axes in napari."""
        if not self._last_heatmap_requests:
            return
        if self._worker_thread is not None and self._worker_thread.isRunning():
            return
        if self._db is None or self._atlas is None:
            return
        updated_requests = [
            replace(request, depth_axis=self._current_depth_axis())
            for request in self._last_heatmap_requests
        ]
        self._start_heatmap_requests(
            updated_requests,
            batch_mode=len(updated_requests) > 1,
        )

    def apply_cluster_colors(self) -> _ClusterColorApplicationSummary:
        """Reapply cached cluster colors and assignments to the current table."""
        if self._cluster_color_map is None or self._last_cluster_result is None:
            return _ClusterColorApplicationSummary(
                matched_table_count=0,
                updated_layer_count=0,
                rendered_count=0,
                colored_count=0,
                gray_count=0,
            )

        summary = self._color_neurons_by_cluster()
        try:
            self.cluster_colors_updated.emit(
                self._last_cluster_result,
                self._cluster_color_map,
            )
        except RuntimeError:
            pass
        return summary

    def _color_neurons_by_cluster(self) -> _ClusterColorApplicationSummary:
        """Color existing neuron layers by their cluster assignment.

        Works with the batched single-layer rendering where all neurons
        are merged into one ``Neuron Lines`` and/or ``Neuron Points``
        layer.  Layer metadata (``file_ids``, ``segments_per_neuron``,
        ``file_ids_per_point``) is used to map cluster labels back to
        individual segments/points. Returns a small summary for status
        text composed by the caller.
        """
        if self._cluster_color_map is None or self._last_cluster_result is None:
            return _ClusterColorApplicationSummary(
                matched_table_count=0,
                updated_layer_count=0,
                rendered_count=0,
                colored_count=0,
                gray_count=0,
            )

        color_map = self._cluster_color_map
        matched_table_count = len(self._matched_current_table_file_ids())
        default_color = [0.5, 0.5, 0.5, 1.0]
        updated = 0
        n_rendered = 0
        n_colored = 0

        for layer in self._viewer.layers:
            if layer.name == "Neuron Lines":
                meta = layer.metadata or {}
                file_ids = meta.get("file_ids", [])
                seg_counts = meta.get("segments_per_neuron", [])
                if file_ids and seg_counts:
                    n_rendered += len(file_ids)
                    n_colored += sum(1 for fid in file_ids if fid in color_map)
                    parts = []
                    for fid, count in zip(file_ids, seg_counts):
                        c = color_map.get(fid, default_color)
                        arr = np.empty((count, 4))
                        arr[:] = c[:4]
                        parts.append(arr)
                    layer.edge_color = np.concatenate(parts)
                    updated += 1

            elif layer.name == "Neuron Points":
                meta = layer.metadata or {}
                fids = meta.get("file_ids_per_point", [])
                if fids:
                    unique_fids = set(fids)
                    if n_rendered == 0:
                        n_rendered = len(unique_fids)
                        n_colored = sum(1 for fid in unique_fids if fid in color_map)
                    colors = np.array(
                        [color_map.get(fid, default_color)[:4] for fid in fids]
                    )
                    layer.face_color = colors
                    updated += 1

        # Also update the 2D slice projector colors
        if self._slice_projector is not None:
            self._slice_projector.update_neuron_colors(color_map)

        n_gray = max(n_rendered - n_colored, 0)
        return _ClusterColorApplicationSummary(
            matched_table_count=matched_table_count,
            updated_layer_count=updated,
            rendered_count=n_rendered,
            colored_count=n_colored,
            gray_count=n_gray,
        )

    def _flush_progress_updates(self) -> None:
        """Give Qt a chance to repaint synchronous progress text."""
        try:
            from qtpy.QtWidgets import QApplication
        except ImportError:
            return

        app = QApplication.instance()
        if app is not None:
            app.processEvents()
