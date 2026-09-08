"""Main neuron viewer widget combining all components.

This widget provides a unified interface for:
1. Loading neuron data from Parquet files
2. Selecting brain regions to filter neurons
3. Visualizing neurons as points or lines
4. Displaying Allen CCF reference layers
"""

from __future__ import annotations

import colorsys
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import QEvent, QObject, Qt, QThread, QTimer, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..analysis.histogram import _build_histogram_plot_series
from ..analysis.mask import (
    build_binary_mask_from_threshold_range,
    isolate_heatmap_volume_to_region_ids,
    merge_heatmap_volumes,
    otsu_threshold_positive,
    smooth_heatmap_volume,
)
from ..auto_center import (
    center_to_depth_world,
    compute_center_of_rendered_neurons,
    depth_axis_from_not_displayed,
)
from ..cluster_assignments import ClusterAssignmentStore
from ..comparison import ComparisonBoardState, compare_cluster_memberships
from ..db import NeuronDatabase
from ..flatmap_parquet import read_flatmap_parquet_transform_info
from ..isocortex_layers import (
    CustomRegionSelectionGroup,
    isocortex_layer_hierarchy_from_atlas,
)
from ..logging_utils import configure_debug_logging, startup_timing
from ..neuron_table_ops import (
    ClusterFilterSelection,
    distinct_colors_for_file_ids,
)
from ..point_import import (
    POINT_PARQUET_ORIGIN_NOT_RECORDED,
    PointImportError,
    build_grouped_point_heatmap_volumes,
    format_atlas_validation_summary,
    load_and_standardize_point_csv,
    load_standard_point_parquet_selection,
    summarize_standard_point_parquet_groups,
    validate_point_metadata_against_atlas,
)
from ..project_io import (
    PROJECT_SUFFIX,
    ProjectBundle,
    export_enhanced_neuron_parquet,
    is_recognized_project_bundle,
    load_project_bundle,
    read_enhanced_parquet_metadata,
    save_project_bundle,
)
from ..region_appearance import RegionAppearanceStore
from .analysis_tab import AnalysisTabWidget
from .collapsible_section import CollapsibleSection
from .custom_region_selector import CustomRegionSelectorWidget
from .mask_layer_selector import MaskLayerSelectorWidget
from .neuron_table import NeuronTableWidget
from .node_type_selector import NodeTypeSelectorComboBox
from .reference_layers import (
    add_allen_template,
    add_brain_outline,
    add_region_id_segmentation,
    add_region_mesh,
    add_region_mesh_group,
    add_region_segmentation,
    remove_region_layers,
    remove_region_segmentation,
)
from .region_selector import RegionSelectorWidget
from .slice_projection import NeuronSliceProjector, SomaSliceProjector
from .termini_section import TerminiSectionWidget

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)

_REGION_QUERY_SOURCE_ATLAS = "Atlas Regions"
_REGION_QUERY_SOURCE_CUSTOM = "Custom Regions"
_REGION_QUERY_SOURCE_MASK = "Mask Layer"

_REGION_QUERY_PAGE_ATLAS = 0
_REGION_QUERY_PAGE_CUSTOM = 1
_REGION_QUERY_PAGE_MASK = 2

_FLATMAP_LAYER_SPACE_KEY = "napari_neuron_navigator_space"
_LEGACY_FLATMAP_LAYER_SPACE_KEY = "napari_swc_viewer_space"
_FLATMAP_LAYER_SPACE_VALUE = "flatmap"
_IS_MACOS = sys.platform == "darwin"


def _flatmap_layer(layer: object) -> bool:
    """Return whether a layer belongs to the flatmap display viewer."""
    metadata = getattr(layer, "metadata", None)
    return bool(
        isinstance(metadata, Mapping)
        and any(
            metadata.get(key) == _FLATMAP_LAYER_SPACE_VALUE
            for key in (
                _FLATMAP_LAYER_SPACE_KEY,
                _LEGACY_FLATMAP_LAYER_SPACE_KEY,
            )
        )
    )


class _FlatmapWindowCloseGuard(QObject):
    """Route a macOS Qt close event through the safe flatmap hide path."""

    def __init__(self, viewer, callback, parent=None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._callback = callback

    def eventFilter(self, watched, event) -> bool:
        if event.type() != QEvent.Type.Close:
            return False
        return bool(self._callback(self._viewer, watched, event))


_SomaSelectionKey = tuple[int, frozenset[int], tuple[object, ...]]

_POINT_HEATMAP_BASE_COLORS = [
    (1.0, 0.0, 0.0, 1.0),  # red
    (0.0, 0.8, 0.0, 1.0),  # green
    (1.0, 0.9, 0.0, 1.0),  # yellow
    (0.1, 0.4, 1.0, 1.0),  # blue
    (1.0, 0.4, 0.0, 1.0),  # orange
    (0.8, 0.0, 0.8, 1.0),  # magenta
    (0.0, 0.8, 0.8, 1.0),  # cyan
    (0.6, 0.3, 0.0, 1.0),  # brown
]

_SOMA_SLICE_PROJECTION_POINT_SIZE = 100
_HISTOGRAM_BIN_COUNT = 256
_POINT_PREVIEW_LABEL_COLUMN = 0
_POINT_PREVIEW_ORIGIN_COLUMN = 1
_POINT_PREVIEW_COUNT_COLUMN = 2
_SCENE_RENDER_MODE_FULL = "full"
_SCENE_RENDER_MODE_SOMA = "soma"
_REGION_QUERY_SCOPE_WHOLE = "whole"
_REGION_QUERY_SCOPE_CURRENT = "current"
_REFERENCE_ACTION_TEMPLATE = "template"
_REFERENCE_ACTION_OUTLINE = "outline"
_REFERENCE_ACTION_MESHES = "meshes"
_REFERENCE_ACTION_SEGMENTATION = "segmentation"
_ATLAS_OPTIONS = (
    "allen_mouse_10um",
    "allen_mouse_25um",
    "allen_mouse_50um",
    "allen_mouse_100um",
)
_ATLAS_REFERENCE_PROMPT = (
    "Atlas loaded. Go to the Reference tab to show the template, outline, "
    "or selected region meshes."
)
_DEFAULT_NEURON_RGBA = (0.5, 0.5, 0.5, 1.0)
_CLUSTER_FILTER_ALL = "all"
_CLUSTER_FILTER_UNCLUSTERED = "unclustered"
_CLUSTER_FILTER_CLUSTER = "cluster"
_MANUAL_HEATMAP_ALL_LABEL = "All Manual Heatmaps"
_SELECTED_HEATMAP_MODE_SINGLE = "single"
_SELECTED_HEATMAP_MODE_INDIVIDUAL = "individual"
_INDIVIDUAL_HEATMAP_MEMORY_WARNING_BYTES = 1024**3
_INDIVIDUAL_HEATMAP_CONTRAST_FRACTION = 0.2
_FINE_PROJECTION_GAMMA = 0.2
_DEFAULT_HEATMAP_GAMMA = 1.0
_GREEK_HEATMAP_IDENTIFIERS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
)


@dataclass(frozen=True)
class _SelectedNeuronHeatmapRequest:
    """Immutable specification for one Data-tab heatmap build."""

    file_ids: tuple[str, ...]
    creation_mode: str
    color: tuple[float, float, float, float] | None = None


def _point_heatmap_color(index: int) -> tuple[float, float, float, float]:
    """Return a distinct RGBA color for a heatmap layer."""
    if index < len(_POINT_HEATMAP_BASE_COLORS):
        return _POINT_HEATMAP_BASE_COLORS[index]

    # Spread additional labels across the hue wheel to avoid reusing colors.
    hue = (index * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (red, green, blue, 1.0)


def _point_heatmap_layer_name(
    label: str,
    origin_csv: str,
    *,
    include_origin: bool,
) -> str:
    """Return the viewer layer name for an imported point heatmap."""

    if include_origin:
        return f"Points Heatmap: {label} [{origin_csv}]"
    return f"Points Heatmap: {label}"


def _layer_metadata(layer) -> dict:
    """Return layer metadata as a mutable dict-like object."""
    metadata = getattr(layer, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _mask_layer_color(source_layers: list) -> tuple[float, float, float, float] | None:
    """Derive a labels-layer color from source heatmap metadata."""
    colors = []
    for layer in source_layers:
        color = _layer_metadata(layer).get("color")
        if color is None:
            continue
        rgba = np.asarray(color, dtype=float).reshape(-1)
        if rgba.size == 3:
            rgba = np.append(rgba, 1.0)
        if rgba.size >= 4:
            colors.append(rgba[:4])

    if not colors:
        return None

    if len(colors) == 1:
        rgba = colors[0]
    else:
        rgba = np.mean(np.vstack(colors), axis=0)
        rgba[3] = 1.0
    rgba = np.clip(rgba, 0.0, 1.0)
    return tuple(float(value) for value in rgba[:4])


def _shared_blur_sigma(source_layers: list) -> float | None:
    """Return a common blur sigma if all source layers share one."""
    values: list[float] = []
    for layer in source_layers:
        value = _layer_metadata(layer).get("blur_sigma")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue

    if not values:
        return None
    first = values[0]
    if all(np.isclose(first, value) for value in values[1:]):
        return first
    return None


def _shared_metadata_value(source_layers: list, key: str):
    """Return a common metadata value if all source layers agree."""
    values = []
    for layer in source_layers:
        metadata = _layer_metadata(layer)
        if key not in metadata:
            continue
        values.append(metadata.get(key))

    if not values:
        return None
    first = values[0]
    if all(value == first for value in values[1:]):
        return first
    return None


def _source_heatmap_filter_metadata(source_layers: list) -> dict[str, object]:
    """Return compact node-type/radius provenance for derived layers."""
    filters = []
    for layer in source_layers:
        metadata = _layer_metadata(layer)
        filters.append(
            {
                "layer": getattr(layer, "name", None),
                "heatmap_node_types": metadata.get("heatmap_node_types"),
                "heatmap_node_type_labels": metadata.get("heatmap_node_type_labels"),
                "heatmap_soma_radius_um": metadata.get("heatmap_soma_radius_um"),
            }
        )

    return {
        "heatmap_node_types": _shared_metadata_value(
            source_layers,
            "heatmap_node_types",
        ),
        "heatmap_node_type_labels": _shared_metadata_value(
            source_layers,
            "heatmap_node_type_labels",
        ),
        "heatmap_soma_radius_um": _shared_metadata_value(
            source_layers,
            "heatmap_soma_radius_um",
        ),
        "source_heatmap_filters": filters,
    }


def _heatmap_contrast_limits(volume: np.ndarray) -> tuple[float, float]:
    """Return stable full-volume contrast limits for a heatmap layer."""
    if volume.size == 0:
        return (0.0, 1.0)

    upper = float(np.nanmax(volume))
    if not np.isfinite(upper) or upper <= 0.0:
        return (0.0, 1.0)
    return (0.0, upper)


class _ClusterFilterComboBox(QComboBox):
    """Compact checkable cluster filter for the Data tab."""

    selection_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._updating = False
        self._available_cluster_ids: tuple[int, ...] = ()
        self._has_unclustered_option = False
        self.setEditable(True)
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setReadOnly(True)
            line_edit.setText("All")
        view = self.view()
        viewport = view.viewport()
        if viewport is not None:
            viewport.installEventFilter(self)
        else:
            view.pressed.connect(self._on_item_pressed)
        self.activated.connect(self._restore_display_text)

    def set_cluster_options(
        self,
        cluster_ids: list[int] | tuple[int, ...],
        *,
        include_unclustered: bool,
        selection: ClusterFilterSelection | None = None,
    ) -> None:
        """Populate the dropdown while preserving a valid selection."""
        available_ids = tuple(sorted(int(cluster_id) for cluster_id in cluster_ids))
        available_set = set(available_ids)
        self._available_cluster_ids = available_ids
        self._has_unclustered_option = bool(include_unclustered)

        selected = selection or ClusterFilterSelection()
        if not selected.is_all:
            selected = ClusterFilterSelection(
                selected.cluster_ids & available_set,
                selected.include_unclustered and include_unclustered,
            )

        signals_blocked = self.blockSignals(True)
        self._updating = True
        try:
            self.clear()
            all_checked = selected.is_all
            self._add_check_item(
                "All",
                _CLUSTER_FILTER_ALL,
                checked=all_checked,
            )
            if include_unclustered:
                self._add_check_item(
                    "Unclustered",
                    _CLUSTER_FILTER_UNCLUSTERED,
                    checked=selected.include_unclustered,
                )
            for cluster_id in available_ids:
                self._add_check_item(
                    f"Cluster {cluster_id}",
                    (_CLUSTER_FILTER_CLUSTER, cluster_id),
                    checked=cluster_id in selected.cluster_ids,
                )
            self.setCurrentIndex(0)
        finally:
            self._updating = False
            self.blockSignals(signals_blocked)

        self._update_display_text()

    def cluster_filter_selection(self) -> ClusterFilterSelection:
        """Return the currently checked cluster filter."""
        cluster_ids: set[int] = set()
        include_unclustered = False
        for index in range(self.count()):
            if not self._item_checked(index):
                continue
            data = self.itemData(index)
            if data == _CLUSTER_FILTER_UNCLUSTERED:
                include_unclustered = True
            elif (
                isinstance(data, tuple)
                and len(data) == 2
                and data[0] == _CLUSTER_FILTER_CLUSTER
            ):
                cluster_ids.add(int(data[1]))
        return ClusterFilterSelection(frozenset(cluster_ids), include_unclustered)

    def _add_check_item(
        self,
        text: str,
        data: object,
        *,
        checked: bool,
    ) -> None:
        self.addItem(text, data)
        index = self.count() - 1
        self._set_item_checked(index, checked)
        item_getter = getattr(self.model(), "item", None)
        item = item_getter(index, self.modelColumn()) if callable(item_getter) else None
        if item is not None:
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)

    def _item_checked(self, index: int) -> bool:
        return self.itemData(index, Qt.CheckStateRole) == Qt.Checked

    def _set_item_checked(self, index: int, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        self.setItemData(index, state, Qt.CheckStateRole)

    def eventFilter(self, watched, event) -> bool:
        """Toggle popup checkboxes directly and keep the popup open."""
        viewport = self.view().viewport()
        if watched is viewport:
            row = self._event_row(event)
            event_type = event.type()
            if row is not None and event_type in (
                QEvent.MouseButtonPress,
                QEvent.MouseButtonDblClick,
            ):
                return True
            if row is not None and event_type == QEvent.MouseButtonRelease:
                self._toggle_item_at_row(row)
                return True
        return super().eventFilter(watched, event)

    def _event_row(self, event) -> int | None:
        pos_getter = getattr(event, "pos", None)
        if not callable(pos_getter):
            return None
        model_index = self.view().indexAt(pos_getter())
        is_valid = getattr(model_index, "isValid", None)
        if callable(is_valid) and not is_valid():
            return None
        row_getter = getattr(model_index, "row", None)
        if not callable(row_getter):
            return None
        row = int(row_getter())
        if row < 0 or row >= self.count():
            return None
        return row

    def _on_item_pressed(self, model_index) -> None:
        """Toggle the pressed row without relying on current-index changes."""
        row_getter = getattr(model_index, "row", None)
        if not callable(row_getter):
            return
        self._toggle_item_at_row(int(row_getter()))

    def _toggle_item_at_row(self, index: int) -> None:
        """Toggle one popup item and emit the updated selection."""
        if self._updating:
            return
        if index < 0 or index >= self.count():
            return

        data = self.itemData(index)
        if data == _CLUSTER_FILTER_ALL:
            self._check_all_only()
        else:
            self._set_item_checked(index, not self._item_checked(index))
            self._set_item_checked(0, False)
            if not self._has_specific_selection():
                self._set_item_checked(0, True)

        self.setCurrentIndex(0)
        self._update_display_text()
        self.selection_changed.emit(self.cluster_filter_selection())

    def _restore_display_text(self, _index: int) -> None:
        """Keep the editable combo display on the compact selection summary."""
        if self._updating:
            return
        self.setCurrentIndex(0)
        self._update_display_text()

    def _check_all_only(self) -> None:
        for index in range(self.count()):
            self._set_item_checked(index, index == 0)

    def _has_specific_selection(self) -> bool:
        return any(self._item_checked(index) for index in range(1, self.count()))

    def _update_display_text(self) -> None:
        selection = self.cluster_filter_selection()
        text = self._selection_text(selection)
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setText(text)
        self.setToolTip(text)

    @staticmethod
    def _selection_text(selection: ClusterFilterSelection) -> str:
        if selection.is_all:
            return "All"

        cluster_ids = sorted(selection.cluster_ids)
        if not cluster_ids:
            return "Unclustered"
        if len(cluster_ids) == 1:
            cluster_text = f"Cluster {cluster_ids[0]}"
        else:
            cluster_text = f"{len(cluster_ids)} clusters"
        if selection.include_unclustered:
            return f"{cluster_text} + Unclustered"
        return cluster_text


class NeuronViewerWidget(QWidget):
    """Main widget for viewing neurons with brain region filtering.

    This widget integrates:
    - Parquet file loading and database querying
    - Hierarchical region selection
    - Neuron visualization (points or lines)
    - Allen CCF reference layers

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer instance.
    """

    def __init__(self, napari_viewer: napari.Viewer):
        log_path = configure_debug_logging()
        with startup_timing(logger, "neuron_viewer_init") as timing:
            super().__init__()
            if log_path is not None:
                timing.set(log_path=log_path)
                logger.debug(
                    "Debug logging enabled for NeuronViewerWidget: %s",
                    log_path,
                )
            self.viewer = napari_viewer
            self._db: NeuronDatabase | None = None
            self._atlas: BrainGlobeAtlas | None = None
            self._current_neuron_layers: list = []
            self._current_region_layers: list = []
            self._highlighted_file_ids: set[str] | None = None
            self._last_soma_selection: _SomaSelectionKey | set = set()
            self._auto_center_applied_once = False
            self._region_query_source = _REGION_QUERY_SOURCE_ATLAS
            self._region_query_scope = _REGION_QUERY_SCOPE_WHOLE
            self._mask_bounds_source = "manual"
            self._histogram_line_sync_active = False
            self._point_parquet_path: str | None = None
            self._point_parquet_has_origin_csv = False
            self._point_preview_counts: dict[tuple[str, str], int] = {}
            self._saved_table_state: dict[str, object] = {}
            self._current_project_path: Path | None = None
            self._cluster_assignment_store = ClusterAssignmentStore()
            self._comparison_board_state = ComparisonBoardState()
            self._comparison_data_provider = None
            self._comparison_window = None
            self._scene_render_modes: dict[object, str] = {}
            self._scene_display_state: dict[object, dict[str, object]] = {}
            self._layer_name_event_connections: dict[
                int,
                tuple[object, object, object],
            ] = {}
            self._flatmap_viewers: list[object] = []
            self._flatmap_viewer = None
            self._flatmap_viewer_pending_show = False
            self._flatmap_viewer_generation = 0
            self._flatmap_viewer_serial = 0
            self._flatmap_hidden_viewer_ids: set[int] = set()
            self._flatmap_viewer_watch_timer = None
            self._flatmap_close_guards: dict[int, tuple[object, object]] = {}

            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="slice_projectors",
            ):
                self._slice_projector = NeuronSliceProjector(
                    napari_viewer,
                    tolerance=100.0,
                )
                self._soma_slice_projector = SomaSliceProjector(
                    napari_viewer,
                    tolerance=100.0,
                    point_size=_SOMA_SLICE_PROJECTION_POINT_SIZE,
                    highlight_callback=self._on_soma_selected,
                )

            # Conversion worker state
            self._convert_thread: QThread | None = None
            self._convert_worker = None
            self._point_convert_thread: QThread | None = None
            self._point_convert_worker = None
            self._point_append_thread: QThread | None = None
            self._point_append_worker = None
            self._convert_source_mode: str | None = None
            self._convert_ui_start_time: float | None = None
            self._selected_heatmap_thread: QThread | None = None
            self._selected_heatmap_worker = None
            self._selected_heatmap_request_file_ids: tuple[str, ...] = ()
            self._selected_heatmap_pending_requests: list[
                _SelectedNeuronHeatmapRequest
            ] = []
            self._selected_heatmap_completed_requests: list[
                _SelectedNeuronHeatmapRequest
            ] = []
            self._selected_heatmap_current_request: (
                _SelectedNeuronHeatmapRequest | None
            ) = None
            self._selected_heatmap_total = 0
            self._selected_heatmap_index = 0
            self._selected_heatmap_failed = False
            self._cached_atlas_thread: QThread | None = None
            self._cached_atlas_worker = None
            self._atlas_load_thread: QThread | None = None
            self._atlas_load_worker = None
            self._pending_reference_action: str | None = None
            self._show_template_after_cached_atlas_load = False

            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="setup_ui",
            ):
                self._setup_ui()
            destroyed = getattr(self, "destroyed", None)
            connect_destroyed = getattr(destroyed, "connect", None)
            if callable(connect_destroyed):
                connect_destroyed(self._on_neuron_viewer_destroyed)
            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="connect_layer_events",
            ):
                self._connect_layer_events()
            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="refresh_heatmap_layer_list",
            ):
                self._refresh_heatmap_layer_list()
            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="refresh_histogram_layer_list",
            ):
                self._refresh_histogram_layer_list()
            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="refresh_mask_layer_options",
            ):
                self._refresh_mask_layer_options()

            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="connect_ndisplay_event",
            ):
                self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)

            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="schedule_cached_template_autoload",
            ):
                QTimer.singleShot(0, self._start_cached_template_autoload)

    @staticmethod
    def _flatmap_viewer_is_open(viewer) -> bool:
        """Probe a viewer through napari's public window API."""
        if viewer is None:
            return False
        geometry = getattr(getattr(viewer, "window", None), "geometry", None)
        if not callable(geometry):
            return False
        try:
            geometry()
        except (AttributeError, RuntimeError, TypeError):
            return False
        return True

    def _start_flatmap_viewer_watch(self) -> None:
        timer = getattr(self, "_flatmap_viewer_watch_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(250)
            timer.timeout.connect(self._retire_closed_flatmap_viewers)
            self._flatmap_viewer_watch_timer = timer
        timer.start()

    def _stop_flatmap_viewer_watch(self) -> None:
        timer = getattr(self, "_flatmap_viewer_watch_timer", None)
        stop = getattr(timer, "stop", None)
        if callable(stop):
            stop()

    def _install_flatmap_macos_close_guard(self, viewer) -> None:
        """Intercept close before napari deletes a still-visible Vispy canvas."""
        if not _IS_MACOS:
            return
        qt_window = getattr(getattr(viewer, "window", None), "_qt_window", None)
        install = getattr(qt_window, "installEventFilter", None)
        if not callable(install):
            logger.warning(
                "The macOS flatmap close guard could not access napari's Qt window."
            )
            return
        guards = getattr(self, "_flatmap_close_guards", None)
        if guards is None:
            guards = {}
            self._flatmap_close_guards = guards
        key = id(viewer)
        if key in guards:
            return
        guard = _FlatmapWindowCloseGuard(
            viewer,
            self._on_flatmap_macos_close_event,
            parent=self,
        )
        guards[key] = (qt_window, guard)
        install(guard)

    def _remove_flatmap_macos_close_guard(self, viewer) -> None:
        guards = getattr(self, "_flatmap_close_guards", None)
        record = guards.pop(id(viewer), None) if guards is not None else None
        if record is None:
            return
        qt_window, guard = record
        remove = getattr(qt_window, "removeEventFilter", None)
        if callable(remove):
            try:
                remove(guard)
            except RuntimeError:
                pass

    def _hide_flatmap_macos_viewer(self, viewer) -> bool:
        """Hide and reset a macOS viewer without destroying its Vispy canvas."""
        if not _IS_MACOS:
            return False
        managed = getattr(self, "_flatmap_viewers", None)
        if managed is None:
            managed = []
            self._flatmap_viewers = managed
        active_viewer = getattr(self, "_flatmap_viewer", None)
        if active_viewer is not None and not any(
            candidate is active_viewer for candidate in managed
        ):
            managed.append(active_viewer)
        if not any(candidate is viewer for candidate in managed):
            return False
        hidden_ids = getattr(self, "_flatmap_hidden_viewer_ids", None)
        if hidden_ids is None:
            hidden_ids = set()
            self._flatmap_hidden_viewer_ids = hidden_ids
        was_visible = id(viewer) not in hidden_ids and not bool(
            viewer is active_viewer
            and getattr(self, "_flatmap_viewer_pending_show", False)
        )
        guards = getattr(self, "_flatmap_close_guards", None) or {}
        record = guards.get(id(viewer))
        qt_window = (
            record[0]
            if record is not None
            else getattr(getattr(viewer, "window", None), "_qt_window", None)
        )
        hide = getattr(qt_window, "hide", None)
        is_fullscreen = getattr(qt_window, "isFullScreen", None)
        show_normal = getattr(qt_window, "showNormal", None)
        if not callable(hide):
            logger.warning(
                "The macOS flatmap viewer could not be hidden safely; "
                "leaving it open."
            )
            return False
        try:
            if callable(is_fullscreen) and callable(show_normal) and is_fullscreen():
                # Return the hidden window to a normal state so the next
                # public Viewer.show() does not reopen it in a stale Space.
                show_normal()
            hide()
        except RuntimeError:
            return False

        hidden_ids.add(id(viewer))
        if not was_visible:
            return True

        if viewer is active_viewer:
            self._flatmap_viewer_pending_show = True
            self._flatmap_viewer_generation = (
                int(getattr(self, "_flatmap_viewer_generation", 0)) + 1
            )

        # The viewer and canvas stay alive on macOS. Disconnect plugin-owned
        # handlers and discard transient layers only after the native window
        # is hidden, then reuse this empty viewer for the next projection.
        flatmap_tab = getattr(self, "_flatmap_tab", None)
        release = getattr(flatmap_tab, "_release_display_viewer", None)
        if callable(release):
            release(viewer)
        layers = getattr(viewer, "layers", None)
        clear = getattr(layers, "clear", None)
        if callable(clear):
            try:
                clear()
            except (AttributeError, RuntimeError):
                logger.debug(
                    "Could not clear hidden flatmap layers.",
                    exc_info=True,
                )
        return True

    @staticmethod
    def _close_flatmap_viewer_now(viewer) -> None:
        close_viewer = getattr(viewer, "close", None)
        if not callable(close_viewer):
            return
        try:
            close_viewer()
        except (AttributeError, RuntimeError):
            logger.debug(
                "Flatmap viewer was already closed while being released.",
                exc_info=True,
            )

    def _on_flatmap_macos_close_event(self, viewer, qt_window, event) -> bool:
        """Consume a native close and hide the viewer without canvas teardown."""
        guards = getattr(self, "_flatmap_close_guards", None) or {}
        record = guards.get(id(viewer))
        managed = getattr(self, "_flatmap_viewers", ())
        viewer_is_managed = viewer is getattr(
            self, "_flatmap_viewer", None
        ) or any(candidate is viewer for candidate in managed)
        if not viewer_is_managed or record is None or record[0] is not qt_window:
            return False
        ignore = getattr(event, "ignore", None)
        if callable(ignore):
            ignore()
        if not self._hide_flatmap_macos_viewer(viewer):
            logger.warning(
                "Ignored an unsafe macOS flatmap close because the window "
                "could not be hidden."
            )
        return True

    def _release_flatmap_viewer(self, viewer, *, close: bool) -> bool:
        """Forget one flatmap viewer and optionally close it safely."""
        managed = getattr(self, "_flatmap_viewers", None)
        if managed is None:
            managed = []
            self._flatmap_viewers = managed
        active_viewer = getattr(self, "_flatmap_viewer", None)
        if active_viewer is not None and not any(
            candidate is active_viewer for candidate in managed
        ):
            managed.append(active_viewer)
        if not any(candidate is viewer for candidate in managed):
            return False

        flatmap_tab = getattr(self, "_flatmap_tab", None)
        release = getattr(flatmap_tab, "_release_display_viewer", None)
        if callable(release):
            release(viewer)

        self._remove_flatmap_macos_close_guard(viewer)
        managed[:] = [candidate for candidate in managed if candidate is not viewer]
        hidden_ids = getattr(self, "_flatmap_hidden_viewer_ids", None)
        if hidden_ids is not None:
            hidden_ids.discard(id(viewer))

        if active_viewer is viewer:
            visible = [
                candidate
                for candidate in managed
                if hidden_ids is None or id(candidate) not in hidden_ids
            ]
            self._flatmap_viewer = visible[-1] if visible else None
            self._flatmap_viewer_pending_show = False
            self._flatmap_viewer_generation = (
                int(getattr(self, "_flatmap_viewer_generation", 0)) + 1
            )

        if close:
            self._close_flatmap_viewer_now(viewer)
        if not managed:
            self._stop_flatmap_viewer_watch()
        return True

    def _retire_closed_flatmap_viewers(self) -> bool:
        """Drop references after users close any flatmap window."""
        managed = getattr(self, "_flatmap_viewers", None)
        if managed is None:
            managed = []
            self._flatmap_viewers = managed
        active_viewer = getattr(self, "_flatmap_viewer", None)
        if active_viewer is not None and not any(
            candidate is active_viewer for candidate in managed
        ):
            managed.append(active_viewer)

        retired = False
        for viewer in list(managed):
            if self._flatmap_viewer_is_open(viewer):
                continue
            retired = self._release_flatmap_viewer(viewer, close=False) or retired
        return retired

    def _retire_closed_flatmap_viewer(self) -> bool:
        """Compatibility wrapper for the former single-viewer lifecycle."""
        return self._retire_closed_flatmap_viewers()

    def _get_or_create_flatmap_viewer(
        self,
        *,
        create: bool = True,
        new_window: bool = False,
    ):
        """Return the active flatmap viewer, optionally creating a new one."""
        self._retire_closed_flatmap_viewers()
        viewer = getattr(self, "_flatmap_viewer", None)
        if viewer is not None and not new_window:
            return viewer
        if not create:
            return viewer

        # A failed/closed first render can leave a hidden macOS viewer ready for
        # reuse. Do not strand another empty top-level when the user retries.
        if viewer is not None and getattr(self, "_flatmap_viewer_pending_show", False):
            return viewer

        managed = getattr(self, "_flatmap_viewers", None)
        if managed is None:
            managed = []
            self._flatmap_viewers = managed
        hidden_ids = getattr(self, "_flatmap_hidden_viewer_ids", None)
        if new_window and hidden_ids:
            reusable = next(
                (
                    candidate
                    for candidate in reversed(managed)
                    if candidate is not viewer and id(candidate) in hidden_ids
                ),
                None,
            )
            if reusable is not None:
                self._flatmap_viewer = reusable
                self._flatmap_viewer_pending_show = True
                hidden_ids.discard(id(reusable))
                self._flatmap_viewer_generation = (
                    int(getattr(self, "_flatmap_viewer_generation", 0)) + 1
                )
                return reusable

        import napari

        serial = int(getattr(self, "_flatmap_viewer_serial", 0)) + 1
        self._flatmap_viewer_serial = serial
        title = "Neuron Navigator Flatmap"
        if serial > 1:
            title = f"{title} {serial}"
        viewer = napari.Viewer(
            title=title,
            ndisplay=3,
            show=False,
        )
        previous = getattr(self, "_flatmap_viewer", None)
        managed.append(viewer)
        self._flatmap_viewer = viewer
        self._flatmap_viewer_pending_show = True
        if hidden_ids is not None:
            hidden_ids.discard(id(viewer))
        if previous is not None:
            self._flatmap_viewer_generation = (
                int(getattr(self, "_flatmap_viewer_generation", 0)) + 1
            )
        self._install_flatmap_macos_close_guard(viewer)
        self._start_flatmap_viewer_watch()
        return viewer

    def _create_new_flatmap_viewer(self):
        """Create and activate another flatmap viewer for side-by-side use."""
        return self._get_or_create_flatmap_viewer(create=True, new_window=True)

    def _on_flatmap_display_viewer_ready(self, viewer, layer) -> None:
        """Show the flatmap window after its first layer is fully configured."""
        if (
            viewer is not getattr(self, "_flatmap_viewer", None)
            or not _flatmap_layer(layer)
            or not any(candidate is layer for candidate in list(viewer.layers))
        ):
            logger.warning(
                "Rejected a flatmap display layer without the flatmap space marker."
            )
            if getattr(self, "_flatmap_viewer_pending_show", False):
                self._release_flatmap_viewer(viewer, close=True)
            return

        try:
            viewer.show()
        except (AttributeError, RuntimeError):
            logger.warning("Could not show the flatmap viewer.", exc_info=True)
            if _IS_MACOS:
                self._hide_flatmap_macos_viewer(viewer)
                self._release_flatmap_viewer(viewer, close=False)
            else:
                self._release_flatmap_viewer(viewer, close=True)
            return
        self._flatmap_viewer_pending_show = False
        hidden_ids = getattr(self, "_flatmap_hidden_viewer_ids", None)
        if hidden_ids is not None:
            hidden_ids.discard(id(viewer))

    def _on_flatmap_display_viewer_failed(self, viewer, reason: str) -> None:
        """Close a hidden viewer after an unsuccessful first render."""
        if viewer is not getattr(self, "_flatmap_viewer", None) or not getattr(
            self, "_flatmap_viewer_pending_show", False
        ):
            return
        logger.debug("Closing hidden flatmap viewer after failure: %s", reason)
        self._release_flatmap_viewer(viewer, close=True)

    def _flatmap_viewer_generation_value(self) -> int:
        self._retire_closed_flatmap_viewer()
        return int(getattr(self, "_flatmap_viewer_generation", 0))

    def _close_flatmap_viewer(self, *_args) -> bool:
        """Dismiss the flatmap window without racing macOS canvas paints."""
        viewer = getattr(self, "_flatmap_viewer", None)
        if viewer is None:
            return False
        if _IS_MACOS:
            return self._hide_flatmap_macos_viewer(viewer)
        return self._release_flatmap_viewer(viewer, close=True)

    def _on_neuron_viewer_destroyed(self, *_args) -> None:
        """Release all plugin-owned flatmap viewers during widget teardown."""
        try:
            self._retire_closed_flatmap_viewers()
            viewers = list(getattr(self, "_flatmap_viewers", ()))
            released = False
            for viewer in viewers:
                if _IS_MACOS:
                    self._hide_flatmap_macos_viewer(viewer)
                    current = self._release_flatmap_viewer(viewer, close=False)
                else:
                    current = self._release_flatmap_viewer(viewer, close=True)
                released = current or released
            if not released:
                self._flatmap_viewer_generation = (
                    int(getattr(self, "_flatmap_viewer_generation", 0)) + 1
                )
        except Exception:
            logger.debug(
                "Could not release flatmap viewer during teardown", exc_info=True
            )

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)

        # Tabs for organization
        tabs = QTabWidget()
        self._tabs = tabs
        layout.addWidget(tabs)

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Data"):
            data_tab = QWidget()
            tabs.addTab(data_tab, "Data")
            self._setup_data_tab(data_tab)

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Regions"):
            regions_tab = QWidget()
            tabs.addTab(regions_tab, "Regions")
            self._setup_regions_tab(regions_tab)

        with startup_timing(
            logger,
            "neuron_viewer_setup_tab",
            tab="Visualization",
        ):
            viz_tab = QWidget()
            tabs.addTab(viz_tab, "Visualization")
            self._setup_viz_tab(viz_tab)

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Flatmap"):
            from .flatmap import FlatmapProjectionWidget

            self._flatmap_tab = FlatmapProjectionWidget(
                self.viewer,
                database_provider=lambda: self._db,
                selected_file_ids_provider=self._neuron_table.get_selected_file_ids,
                table_file_ids_provider=self._neuron_table.file_ids,
                color_map_provider=self._neuron_table.get_full_color_map,
                cluster_map_provider=self._neuron_table.get_cluster_map,
                atlas_provider=lambda: self._atlas,
                selected_region_ids_provider=self._active_flatmap_region_ids,
                selected_geometry_region_ids_provider=(
                    self._active_flatmap_parent_region_ids
                ),
                selected_region_acronyms_provider=self._active_flatmap_region_acronyms,
                selected_region_source_provider=(self._active_flatmap_region_source),
                selected_region_scope_provider=self._active_flatmap_region_scope,
                selected_region_error_provider=self._active_flatmap_region_error,
                region_appearance_provider=self._region_appearance_store,
                display_viewer_provider=self._get_or_create_flatmap_viewer,
                new_display_viewer_callback=self._create_new_flatmap_viewer,
                display_viewer_ready_callback=(self._on_flatmap_display_viewer_ready),
                display_viewer_failed_callback=(self._on_flatmap_display_viewer_failed),
                display_generation_provider=(self._flatmap_viewer_generation_value),
            )
            tabs.addTab(self._flatmap_tab, "Flatmap")

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Reference"):
            ref_tab = QWidget()
            tabs.addTab(ref_tab, "Reference")
            self._setup_reference_tab(ref_tab)

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Analysis"):
            self._analysis_tab = AnalysisTabWidget(self.viewer)
            self._analysis_tab.set_slice_projector(self._slice_projector)
            self._analysis_tab.set_cluster_assignment_store(
                self._cluster_assignment_store
            )
            self._analysis_tab.set_current_table_file_ids_provider(
                self._current_table_file_ids
            )
            self._analysis_tab.set_selected_table_file_ids_provider(
                self._neuron_table.get_selected_file_ids
            )
            self._analysis_tab.set_flatmap_correlation_source_provider(
                self._flatmap_tab.latest_flatmap_correlation_source
            )
            self._flatmap_tab.set_flatmap_correlation_source_changed_callback(
                self._analysis_tab.refresh_flatmap_coordinate_availability
            )
            self._analysis_tab.cluster_colors_updated.connect(
                self._on_cluster_colors_updated
            )
            tabs.addTab(self._analysis_tab, "Analysis")

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Compare"):
            compare_tab = QWidget()
            tabs.addTab(compare_tab, "Compare")
            self._setup_compare_tab(compare_tab)

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Tools"):
            tools_tab = QWidget()
            tabs.addTab(tools_tab, "Tools")
            self._setup_tools_tab(tools_tab)

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Histogram"):
            histogram_tab = QWidget()
            tabs.addTab(histogram_tab, "Histogram")
            self._setup_histogram_tab(histogram_tab)

    def _setup_compare_tab(self, parent: QWidget) -> None:
        """Set up the launcher and status for the comparison board."""
        layout = QVBoxLayout(parent)

        introduction = QLabel(
            "Compare saved cluster assignments in a separate interactive "
            "1×1 through 4×4 board. Cells can show flatmap somas or arbor "
            "density, and CCFv3 somas or Analysis heatmaps."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)

        board_group = QGroupBox("Comparison Board")
        board_layout = QVBoxLayout(board_group)
        open_button = QPushButton("Open Comparison Board")
        open_button.clicked.connect(self._open_comparison_board)
        board_layout.addWidget(open_button)

        clear_button = QPushButton("Clear Comparison Board")
        clear_button.clicked.connect(self._clear_comparison_board)
        board_layout.addWidget(clear_button)

        self._comparison_status_label = QLabel()
        self._comparison_status_label.setWordWrap(True)
        board_layout.addWidget(self._comparison_status_label)
        layout.addWidget(board_group)

        metrics_group = QGroupBox("Cluster Membership Comparison")
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_help = QLabel(
            "Compare two saved assignments over their shared file_id cohort. "
            "ARI, NMI, and matched agreement use neurons assigned in both runs."
        )
        metrics_help.setWordWrap(True)
        metrics_layout.addWidget(metrics_help)

        reference_row = QHBoxLayout()
        reference_row.addWidget(QLabel("Reference assignment:"))
        self._comparison_metrics_reference_combo = QComboBox()
        self._comparison_metrics_reference_combo.currentIndexChanged.connect(
            self._on_comparison_metrics_reference_changed
        )
        reference_row.addWidget(self._comparison_metrics_reference_combo, 1)
        metrics_layout.addLayout(reference_row)

        candidate_row = QHBoxLayout()
        candidate_row.addWidget(QLabel("Candidate assignment:"))
        self._comparison_metrics_candidate_combo = QComboBox()
        self._comparison_metrics_candidate_combo.currentIndexChanged.connect(
            self._update_comparison_metrics
        )
        candidate_row.addWidget(self._comparison_metrics_candidate_combo, 1)
        metrics_layout.addLayout(candidate_row)

        self._comparison_metrics_summary_label = QLabel(
            "Save at least two cluster assignments to calculate membership metrics."
        )
        self._comparison_metrics_summary_label.setWordWrap(True)
        self._comparison_metrics_summary_label.setToolTip(
            "ARI corrects pairwise grouping agreement for chance. NMI measures "
            "shared information. Matched agreement applies maximum-overlap "
            "one-to-one cluster label matching."
        )
        metrics_layout.addWidget(self._comparison_metrics_summary_label)

        self._comparison_metrics_coverage_label = QLabel("")
        self._comparison_metrics_coverage_label.setWordWrap(True)
        metrics_layout.addWidget(self._comparison_metrics_coverage_label)

        metrics_layout.addWidget(
            QLabel("Cluster overlap counts (reference rows × candidate columns):")
        )
        self._comparison_overlap_table = QTableWidget(0, 0)
        self._comparison_overlap_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._comparison_overlap_table.setMinimumHeight(140)
        self._comparison_overlap_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self._comparison_overlap_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        metrics_layout.addWidget(self._comparison_overlap_table)

        metrics_layout.addWidget(QLabel("Optimally matched clusters:"))
        self._comparison_cluster_match_table = QTableWidget(0, 7)
        self._comparison_cluster_match_table.setHorizontalHeaderLabels(
            [
                "Reference",
                "Candidate",
                "Overlap",
                "Reference size",
                "Candidate size",
                "Union",
                "Jaccard",
            ]
        )
        self._comparison_cluster_match_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self._comparison_cluster_match_table.setMinimumHeight(160)
        self._comparison_cluster_match_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        metrics_layout.addWidget(self._comparison_cluster_match_table)
        layout.addWidget(metrics_group, 1)

        self._comparison_metrics_has_defaulted = False
        self._update_comparison_status()
        self._refresh_comparison_metric_controls()

    def _refresh_comparison_metric_controls(self) -> None:
        """Refresh metric selectors while preserving stable assignment IDs."""
        reference_combo = getattr(
            self,
            "_comparison_metrics_reference_combo",
            None,
        )
        candidate_combo = getattr(
            self,
            "_comparison_metrics_candidate_combo",
            None,
        )
        if reference_combo is None or candidate_combo is None:
            return
        assignments = self._cluster_assignment_store.sets()
        assignment_ids = {assignment.assignment_id for assignment in assignments}
        reference_id = self._comparison_board_state.reference_assignment_id
        previous_candidate = candidate_combo.currentData()
        defaulted_reference = False

        if (
            reference_id is None
            and assignments
            and not self._comparison_metrics_has_defaulted
        ):
            reference_id = assignments[0].assignment_id
            self._comparison_board_state.reference_assignment_id = reference_id
            self._comparison_metrics_has_defaulted = True
            defaulted_reference = True
        elif reference_id is not None:
            self._comparison_metrics_has_defaulted = True

        blocked = reference_combo.blockSignals(True)
        reference_combo.clear()
        reference_combo.addItem("No reference / color matching", None)
        for assignment in assignments:
            reference_combo.addItem(assignment.name, assignment.assignment_id)
        if reference_id is not None and reference_id not in assignment_ids:
            reference_combo.addItem(
                f"Missing reference ({reference_id})",
                reference_id,
            )
        reference_index = reference_combo.findData(reference_id)
        reference_combo.setCurrentIndex(max(0, reference_index))
        reference_combo.blockSignals(blocked)

        blocked = candidate_combo.blockSignals(True)
        candidate_combo.clear()
        candidates = [
            assignment
            for assignment in assignments
            if assignment.assignment_id != reference_id
        ]
        if not candidates:
            candidate_combo.addItem("No different assignment available", None)
        else:
            for assignment in candidates:
                candidate_combo.addItem(assignment.name, assignment.assignment_id)
            candidate_index = candidate_combo.findData(previous_candidate)
            candidate_combo.setCurrentIndex(max(0, candidate_index))
        candidate_combo.blockSignals(blocked)
        if defaulted_reference:
            window = getattr(self, "_comparison_window", None)
            setter = getattr(window, "set_reference_assignment_id", None)
            if callable(setter):
                setter(reference_id)
        self._update_comparison_metrics()

    def _on_comparison_metrics_reference_changed(self, index: int) -> None:
        """Use the metrics reference as the board's color-matching reference."""
        combo = self._comparison_metrics_reference_combo
        reference_id = combo.itemData(index) if index >= 0 else None
        self._comparison_metrics_has_defaulted = True
        self._comparison_board_state.reference_assignment_id = (
            str(reference_id) if reference_id not in (None, "") else None
        )
        window = getattr(self, "_comparison_window", None)
        setter = getattr(window, "set_reference_assignment_id", None)
        if callable(setter):
            setter(self._comparison_board_state.reference_assignment_id)
        self._update_comparison_status()
        self._refresh_comparison_metric_controls()

    @staticmethod
    def _comparison_metric_text(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    def _clear_comparison_metric_tables(self) -> None:
        overlap = getattr(self, "_comparison_overlap_table", None)
        if overlap is not None:
            overlap.clear()
            overlap.setRowCount(0)
            overlap.setColumnCount(0)
        matches = getattr(self, "_comparison_cluster_match_table", None)
        if matches is not None:
            matches.setRowCount(0)

    def _update_comparison_metrics(self, _index: int | None = None) -> None:
        """Calculate and display one selected saved-assignment comparison."""
        reference_combo = getattr(
            self,
            "_comparison_metrics_reference_combo",
            None,
        )
        candidate_combo = getattr(
            self,
            "_comparison_metrics_candidate_combo",
            None,
        )
        if reference_combo is None or candidate_combo is None:
            return
        reference = self._cluster_assignment_store.get(reference_combo.currentData())
        candidate = self._cluster_assignment_store.get(candidate_combo.currentData())
        summary = self._comparison_metrics_summary_label
        coverage = self._comparison_metrics_coverage_label
        if reference is None or candidate is None:
            self._clear_comparison_metric_tables()
            reference_id = reference_combo.currentData()
            if reference_id not in (None, "") and reference is None:
                summary.setText(
                    f"Reference assignment {reference_id!s} is unavailable. "
                    "Choose another saved assignment; no name-based substitute "
                    "will be used."
                )
            else:
                summary.setText(
                    "Choose two different saved assignments to calculate "
                    "membership metrics."
                )
            coverage.setText("")
            return

        result = compare_cluster_memberships(reference, candidate)
        agreement = (
            "—"
            if result.matched_agreement is None
            else f"{100.0 * result.matched_agreement:.1f}% "
            f"({result.matched_file_ids:,} / "
            f"{result.cohort_counts.assigned_in_both:,})"
        )
        summary.setText(
            f"ARI: {self._comparison_metric_text(result.adjusted_rand_index)} · "
            f"NMI: {self._comparison_metric_text(result.normalized_mutual_information)} · "
            f"Matched agreement: {agreement}"
        )
        counts = result.cohort_counts
        coverage.setText(
            f"Shared cohort: {counts.shared_cohort:,} · assigned in both: "
            f"{counts.assigned_in_both:,} · reference assigned only: "
            f"{counts.reference_assigned_only:,} · candidate assigned only: "
            f"{counts.candidate_assigned_only:,} · unassigned in both: "
            f"{counts.unassigned_in_both:,} · cohort-only: "
            f"{counts.reference_cohort_only:,} reference / "
            f"{counts.candidate_cohort_only:,} candidate"
        )
        if result.status != "ok":
            coverage.setText(f"{result.status_message} {coverage.text()}")

        overlap = self._comparison_overlap_table
        overlap.clear()
        overlap.setRowCount(len(result.reference_cluster_ids))
        overlap.setColumnCount(len(result.candidate_cluster_ids))
        overlap.setVerticalHeaderLabels(
            [str(label) for label in result.reference_cluster_ids]
        )
        overlap.setHorizontalHeaderLabels(
            [str(label) for label in result.candidate_cluster_ids]
        )
        for row, values in enumerate(result.overlap_counts):
            for column, value in enumerate(values):
                overlap.setItem(row, column, QTableWidgetItem(f"{value:,}"))

        matches = self._comparison_cluster_match_table
        matches.setRowCount(len(result.cluster_matches))
        for row, match in enumerate(result.cluster_matches):
            values = (
                "Unmatched" if match.reference_label is None else match.reference_label,
                "Unmatched" if match.candidate_label is None else match.candidate_label,
                f"{match.intersection:,}",
                f"{match.reference_size:,}",
                f"{match.candidate_size:,}",
                f"{match.union:,}",
                f"{match.jaccard:.3f}",
            )
            for column, value in enumerate(values):
                matches.setItem(row, column, QTableWidgetItem(str(value)))

    def _comparison_state_payload(self) -> dict[str, object]:
        """Return the latest serializable comparison-board recipe."""
        window = getattr(self, "_comparison_window", None)
        if window is not None:
            try:
                self._comparison_board_state = window.state()
            except Exception:
                logger.debug("Could not read comparison board state", exc_info=True)
        provider = getattr(self, "_comparison_data_provider", None)
        if provider is not None:
            for index, cell in enumerate(self._comparison_board_state.cells):
                try:
                    self._comparison_board_state.cells[index] = provider.prepare_spec(
                        cell
                    )
                except Exception:
                    logger.debug(
                        "Could not finalize comparison cell grid before save",
                        exc_info=True,
                    )
        return self._comparison_board_state.to_state()

    def _on_comparison_board_state_changed(
        self,
        state: ComparisonBoardState,
    ) -> None:
        """Retain board edits independently of the window's visibility."""
        prior_reference = self._comparison_board_state.reference_assignment_id
        self._comparison_board_state = ComparisonBoardState.from_state(state.to_state())
        self._update_comparison_status()
        if prior_reference != self._comparison_board_state.reference_assignment_id:
            self._refresh_comparison_metric_controls()

    def _update_comparison_status(self) -> None:
        """Describe the current project board in the Compare tab."""
        label = getattr(self, "_comparison_status_label", None)
        if label is None:
            return
        state = self._comparison_board_state
        source_count = len(state.cells)
        if source_count:
            label.setText(
                f"Current board: {state.rows}×{state.columns} · "
                f"{source_count} configured cell(s)."
            )
        else:
            label.setText(
                f"Current board: {state.rows}×{state.columns} · no cells configured."
            )

    def _comparison_provider(self):
        """Create the project-backed comparison data provider on demand."""
        provider = getattr(self, "_comparison_data_provider", None)
        if provider is None:
            from ..comparison_data import ComparisonDataProvider

            provider = ComparisonDataProvider(
                database_provider=lambda: self._db,
                assignment_store_provider=lambda: self._cluster_assignment_store,
                viewer_layers_provider=self._iter_viewer_layers,
                atlas_provider=lambda: self._atlas,
            )
            self._comparison_data_provider = provider
        return provider

    def _open_comparison_board(self) -> None:
        """Show the project's separate lightweight comparison window."""
        if self._db is None:
            show_warning("Load a neuron Parquet before opening the comparison board.")
            return
        window = getattr(self, "_comparison_window", None)
        if window is None:
            from .comparison_board import ComparisonBoardWindow

            window = ComparisonBoardWindow(
                provider=self._comparison_provider(),
                state=ComparisonBoardState.from_state(
                    self._comparison_board_state.to_state()
                ),
                state_changed_callback=self._on_comparison_board_state_changed,
                parent=None,
            )
            self._comparison_window = window
        else:
            window.refresh_sources()
        window.show()
        window.raise_()
        window.activateWindow()

    def _clear_comparison_board(self) -> None:
        """Reset the current project's board recipe to an empty 2×2 board."""
        self._comparison_board_state = ComparisonBoardState()
        window = getattr(self, "_comparison_window", None)
        if window is not None:
            window.set_state(self._comparison_board_state)
        self._update_comparison_status()
        self._refresh_comparison_metric_controls()

    def _restore_comparison_board(self, payload: object) -> None:
        """Restore a versioned board recipe or an empty legacy-project state."""
        state = ComparisonBoardState.from_state(
            payload if isinstance(payload, dict) else None
        )
        self._comparison_board_state = state
        if state.reference_assignment_id is not None:
            self._comparison_metrics_has_defaulted = True
        provider = getattr(self, "_comparison_data_provider", None)
        if provider is not None:
            provider.clear_cache()
        window = getattr(self, "_comparison_window", None)
        if window is not None:
            window.set_state(state)
        self._update_comparison_status()
        self._refresh_comparison_metric_controls()

    def _setup_data_tab(self, parent: QWidget) -> None:
        """Set up the data loading tab."""
        parent_layout = QVBoxLayout(parent)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        parent_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        # SWC to Parquet conversion
        convert_section = CollapsibleSection(
            "Convert SWC to Parquet",
            expanded=False,
        )
        convert_layout = convert_section.content_layout()

        convert_btn_row = QHBoxLayout()
        self._convert_dir_btn = QPushButton("From Directory...")
        self._convert_dir_btn.clicked.connect(self._convert_from_directory)
        convert_btn_row.addWidget(self._convert_dir_btn)

        self._convert_files_btn = QPushButton("From Files...")
        self._convert_files_btn.clicked.connect(self._convert_from_files)
        convert_btn_row.addWidget(self._convert_files_btn)
        convert_layout.addLayout(convert_btn_row)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution (μm):"))
        self._convert_resolution_spin = QSpinBox()
        self._convert_resolution_spin.setRange(10, 100)
        self._convert_resolution_spin.setValue(25)
        res_row.addWidget(self._convert_resolution_spin)
        convert_layout.addLayout(res_row)

        hemisphere_row = QHBoxLayout()
        hemisphere_row.addWidget(QLabel("Hemisphere alignment:"))
        self._convert_hemisphere_combo = QComboBox()
        self._convert_hemisphere_combo.addItem("None", None)
        self._convert_hemisphere_combo.addItem("Left", "left")
        self._convert_hemisphere_combo.addItem("Right", "right")
        hemisphere_row.addWidget(self._convert_hemisphere_combo)
        convert_layout.addLayout(hemisphere_row)

        self._convert_add_flatmap_cb = QCheckBox("Add bilateral flatmap/depth columns")
        convert_layout.addWidget(self._convert_add_flatmap_cb)

        lookup_row = QHBoxLayout()
        self._convert_lookup_dir_label = QLabel("No lookup directory selected")
        self._convert_lookup_dir_label.setWordWrap(True)
        lookup_row.addWidget(self._convert_lookup_dir_label, stretch=1)
        self._convert_lookup_dir_btn = QPushButton("Lookup directory...")
        self._convert_lookup_dir_btn.clicked.connect(
            self._choose_conversion_lookup_directory
        )
        lookup_row.addWidget(self._convert_lookup_dir_btn)
        convert_layout.addLayout(lookup_row)

        lookup_resolution_row = QHBoxLayout()
        lookup_resolution_row.addWidget(QLabel("Lookup resolution:"))
        self._convert_lookup_resolution_spin = QSpinBox()
        self._convert_lookup_resolution_spin.setRange(0, 100)
        self._convert_lookup_resolution_spin.setSpecialValueText("From NRRD header")
        self._convert_lookup_resolution_spin.setSuffix(" μm")
        lookup_resolution_row.addWidget(self._convert_lookup_resolution_spin)
        convert_layout.addLayout(lookup_resolution_row)

        self._convert_cancel_btn = QPushButton("Cancel conversion")
        self._convert_cancel_btn.setEnabled(False)
        self._convert_cancel_btn.clicked.connect(self._cancel_conversion)
        convert_layout.addWidget(self._convert_cancel_btn)

        self._convert_progress = QProgressBar()
        self._convert_progress.setVisible(False)
        convert_layout.addWidget(self._convert_progress)

        self._convert_status_label = QLabel("")
        convert_layout.addWidget(self._convert_status_label)

        layout.addWidget(convert_section)

        # File selection
        file_section = CollapsibleSection("SWC Parquet Data")
        file_layout = file_section.content_layout()

        file_row = QHBoxLayout()
        self._file_label = QLabel("No file loaded")
        self._file_label.setWordWrap(True)
        file_row.addWidget(self._file_label)

        load_btn = QPushButton("Load...")
        load_btn.clicked.connect(self._load_parquet)
        file_row.addWidget(load_btn)
        file_layout.addLayout(file_row)

        project_io_row = QHBoxLayout()
        self._save_project_btn = QPushButton("Save Project")
        self._save_project_btn.setEnabled(False)
        self._save_project_btn.setToolTip(
            "Load or create a Neuron Navigator project before saving in place."
        )
        self._save_project_btn.clicked.connect(self._save_current_project)
        project_io_row.addWidget(self._save_project_btn)

        self._save_project_as_btn = QPushButton("Save Project As...")
        self._save_project_as_btn.clicked.connect(self._save_project_as_dialog)
        project_io_row.addWidget(self._save_project_as_btn)

        self._load_project_btn = QPushButton("Load Project...")
        self._load_project_btn.clicked.connect(self._load_project_bundle_dialog)
        project_io_row.addWidget(self._load_project_btn)

        self._export_enhanced_parquet_btn = QPushButton("Export Enhanced Parquet...")
        self._export_enhanced_parquet_btn.clicked.connect(
            self._export_enhanced_parquet_dialog
        )
        project_io_row.addWidget(self._export_enhanced_parquet_btn)
        file_layout.addLayout(project_io_row)

        self._project_io_note_label = QLabel(
            "Note: Enhanced Parquet exports the entire input Parquet. "
            "Save Project stores only the current table subset and depends on "
            "the original input Parquet for full-dataset reproduction."
        )
        self._project_io_note_label.setWordWrap(True)
        self._project_io_note_label.setStyleSheet("color: #cc7700; font-style: italic;")
        file_layout.addWidget(self._project_io_note_label)

        self._project_progress = QProgressBar()
        self._project_progress.setVisible(False)
        file_layout.addWidget(self._project_progress)

        self._project_status_label = QLabel("")
        self._project_status_label.setWordWrap(True)
        file_layout.addWidget(self._project_status_label)

        # Stats
        self._stats_label = QLabel("")
        file_layout.addWidget(self._stats_label)

        self._flatmap_transform_status_label = QLabel("")
        self._flatmap_transform_status_label.setWordWrap(True)
        file_layout.addWidget(self._flatmap_transform_status_label)

        layout.addWidget(file_section)

        # Atlas selection
        atlas_section = CollapsibleSection("Atlas")
        atlas_layout = atlas_section.content_layout()

        atlas_row = QHBoxLayout()

        atlas_row.addWidget(QLabel("Atlas:"))
        self._atlas_combo = QComboBox()
        self._atlas_combo.addItems(_ATLAS_OPTIONS)
        self._atlas_combo.setCurrentText("allen_mouse_25um")
        atlas_row.addWidget(self._atlas_combo)

        self._load_atlas_btn = QPushButton("Load Atlas")
        self._load_atlas_btn.clicked.connect(self._load_atlas)
        atlas_row.addWidget(self._load_atlas_btn)

        atlas_layout.addLayout(atlas_row)

        # Atlas status label
        self._atlas_status_label = QLabel("Atlas: Not loaded")
        self._atlas_status_label.setWordWrap(True)
        atlas_layout.addWidget(self._atlas_status_label)

        self._atlas_progress = QProgressBar()
        self._atlas_progress.setRange(0, 100)
        self._atlas_progress.setValue(0)
        self._atlas_progress.setVisible(False)
        atlas_layout.addWidget(self._atlas_progress)
        layout.addWidget(atlas_section)

        # Standardized point Parquet import
        point_section = CollapsibleSection(
            "Point Parquet Import",
            expanded=False,
        )
        point_layout = point_section.content_layout()

        point_create_row = QHBoxLayout()
        self._create_point_from_directory_btn = QPushButton("Create From Directory...")
        self._create_point_from_directory_btn.clicked.connect(
            self._convert_point_csv_from_directory
        )
        point_create_row.addWidget(self._create_point_from_directory_btn)

        self._create_point_from_files_btn = QPushButton("Create From File(s)...")
        self._create_point_from_files_btn.clicked.connect(
            self._convert_point_csv_from_files
        )
        point_create_row.addWidget(self._create_point_from_files_btn)
        point_layout.addLayout(point_create_row)

        point_row = QHBoxLayout()
        self._point_file_label = QLabel("No point parquet imported")
        self._point_file_label.setWordWrap(True)
        point_row.addWidget(self._point_file_label)

        self._open_point_parquet_btn = QPushButton("Open Point Parquet...")
        self._open_point_parquet_btn.clicked.connect(self._open_point_parquet)
        point_row.addWidget(self._open_point_parquet_btn)

        self._append_point_file_btn = QPushButton("Append Point file")
        self._append_point_file_btn.clicked.connect(self._append_point_file)
        point_row.addWidget(self._append_point_file_btn)
        point_layout.addLayout(point_row)

        self._point_preview_table = QTableWidget(0, 3)
        self._point_preview_table.setHorizontalHeaderLabels(
            ["Label", "Origin CSV", "Points"]
        )
        self._point_preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._point_preview_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._point_preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._point_preview_table.verticalHeader().setVisible(False)
        point_header = self._point_preview_table.horizontalHeader()
        point_header.setSectionResizeMode(
            _POINT_PREVIEW_LABEL_COLUMN,
            QHeaderView.Stretch,
        )
        point_header.setSectionResizeMode(
            _POINT_PREVIEW_ORIGIN_COLUMN,
            QHeaderView.Stretch,
        )
        point_header.setSectionResizeMode(
            _POINT_PREVIEW_COUNT_COLUMN,
            QHeaderView.ResizeToContents,
        )
        self._point_preview_table.itemSelectionChanged.connect(
            self._update_point_import_controls
        )
        point_layout.addWidget(self._point_preview_table)

        self._import_selected_point_heatmaps_btn = QPushButton(
            "Import Selected Heatmaps"
        )
        self._import_selected_point_heatmaps_btn.setEnabled(False)
        self._import_selected_point_heatmaps_btn.clicked.connect(
            self._import_selected_point_heatmaps
        )
        point_layout.addWidget(self._import_selected_point_heatmaps_btn)

        self._point_append_progress = QProgressBar()
        self._point_append_progress.setVisible(False)
        point_layout.addWidget(self._point_append_progress)

        self._point_import_status_label = QLabel("")
        self._point_import_status_label.setWordWrap(True)
        point_layout.addWidget(self._point_import_status_label)

        layout.addWidget(point_section)

        # Selected neurons table
        neurons_section = CollapsibleSection("Selected Neurons")
        neurons_layout = neurons_section.content_layout()

        self._neuron_table = NeuronTableWidget(
            assignment_store=self._cluster_assignment_store
        )
        self._neuron_table.colors_changed.connect(self._apply_neuron_colors)
        self._neuron_table.visibility_changed.connect(self._apply_neuron_visibility)
        self._neuron_table.selection_changed.connect(self._highlight_selected_neurons)
        state_changed = getattr(self._neuron_table, "state_changed", None)
        if state_changed is not None:
            state_changed.connect(self._refresh_neuron_table_summary)

        self._adaptive_table_height_cb = QCheckBox("Adaptive table height")
        self._adaptive_table_height_cb.setChecked(True)
        self._adaptive_table_height_cb.setToolTip(
            "Grow the table with its visible rows (up to 20). Uncheck to use "
            "a fixed eight-row viewport."
        )
        self._adaptive_table_height_cb.toggled.connect(
            self._neuron_table.set_adaptive_height_enabled
        )
        neurons_layout.addWidget(self._adaptive_table_height_cb)
        neurons_layout.addWidget(self._neuron_table)

        assignment_row = QHBoxLayout()
        assignment_row.addWidget(QLabel("Cluster assignment:"))
        self._cluster_assignment_combo = QComboBox()
        self._cluster_assignment_combo.currentIndexChanged.connect(
            self._on_active_cluster_assignment_changed
        )
        assignment_row.addWidget(self._cluster_assignment_combo, 1)
        self._rename_cluster_assignment_btn = QPushButton("Rename...")
        self._rename_cluster_assignment_btn.clicked.connect(
            self._rename_active_cluster_assignment
        )
        assignment_row.addWidget(self._rename_cluster_assignment_btn)
        self._delete_cluster_assignment_btn = QPushButton("Delete")
        self._delete_cluster_assignment_btn.clicked.connect(
            self._delete_active_cluster_assignment
        )
        assignment_row.addWidget(self._delete_cluster_assignment_btn)
        neurons_layout.addLayout(assignment_row)
        self._refresh_cluster_assignment_controls()

        manual_heatmap_row = QHBoxLayout()
        manual_heatmap_row.addWidget(QLabel("Manual Heatmap:"))
        self._manual_heatmap_combo = QComboBox()
        self._manual_heatmap_combo.currentIndexChanged.connect(
            self._on_manual_heatmap_selection_changed
        )
        manual_heatmap_row.addWidget(self._manual_heatmap_combo, 1)
        neurons_layout.addLayout(manual_heatmap_row)
        self._refresh_manual_heatmap_combo()

        cluster_filter_row = QHBoxLayout()
        cluster_filter_row.addWidget(QLabel("Cluster:"))

        self._cluster_filter_combo = _ClusterFilterComboBox()
        self._cluster_filter_combo.set_cluster_options(
            [],
            include_unclustered=False,
            selection=ClusterFilterSelection(),
        )
        self._cluster_filter_combo.selection_changed.connect(
            self._on_cluster_filter_changed
        )
        cluster_filter_row.addWidget(self._cluster_filter_combo, 1)
        neurons_layout.addLayout(cluster_filter_row)

        cluster_action_row = QHBoxLayout()

        self._hide_others_btn = QPushButton("Hide Others")
        self._hide_others_btn.setEnabled(False)
        self._hide_others_btn.clicked.connect(self._hide_not_in_selected_cluster)
        cluster_action_row.addWidget(self._hide_others_btn)

        self._show_all_btn = QPushButton("Show All")
        self._show_all_btn.setEnabled(False)
        self._show_all_btn.clicked.connect(self._show_all_neurons)
        cluster_action_row.addWidget(self._show_all_btn)

        self._recolor_cluster_btn = QPushButton("Recolor Selection")
        self._recolor_cluster_btn.setEnabled(False)
        self._recolor_cluster_btn.clicked.connect(self._recolor_selected_cluster)
        cluster_action_row.addWidget(self._recolor_cluster_btn)

        self._apply_existing_clusters_btn = QPushButton("Apply Existing Clusters")
        self._apply_existing_clusters_btn.setEnabled(False)
        self._apply_existing_clusters_btn.setVisible(False)
        self._apply_existing_clusters_btn.clicked.connect(
            self._apply_existing_clusters_from_analysis
        )
        cluster_action_row.addWidget(self._apply_existing_clusters_btn)

        neurons_layout.addLayout(cluster_action_row)

        self._selected_neurons_hint_label = QLabel(
            "Only Neurons highlighted in the above table will be added to "
            "scene. Use cmd+A or ctrl+A to select all."
        )
        self._selected_neurons_hint_label.setWordWrap(True)
        neurons_layout.addWidget(self._selected_neurons_hint_label)

        neuron_btn_row = QHBoxLayout()
        self._render_btn = QPushButton("Add Full Neurons")
        self._render_btn.clicked.connect(self._render_selected_neurons)
        neuron_btn_row.addWidget(self._render_btn)

        self._render_soma_only_btn = QPushButton("Add Soma Only")
        self._render_soma_only_btn.clicked.connect(self._render_selected_soma_only)
        neuron_btn_row.addWidget(self._render_soma_only_btn)

        self._remove_selected_btn = QPushButton("Remove Selected")
        self._remove_selected_btn.clicked.connect(self._remove_selected_neurons)
        neuron_btn_row.addWidget(self._remove_selected_btn)

        self._clear_neurons_btn = QPushButton("Clear All")
        self._clear_neurons_btn.clicked.connect(self._clear_all_neuron_layers)
        neuron_btn_row.addWidget(self._clear_neurons_btn)

        neurons_layout.addLayout(neuron_btn_row)

        heatmap_btn_row = QHBoxLayout()
        self._add_selected_heatmap_btn = QPushButton("Add Heatmap")
        self._configure_selected_heatmap_menu()
        heatmap_btn_row.addWidget(self._add_selected_heatmap_btn)

        self._remove_selected_from_table_btn = QPushButton("Remove Selected From Table")
        self._remove_selected_from_table_btn.clicked.connect(
            self._remove_selected_from_table
        )
        heatmap_btn_row.addWidget(self._remove_selected_from_table_btn)

        self._remove_unselected_from_table_btn = QPushButton(
            "Remove Unselected from Table"
        )
        self._remove_unselected_from_table_btn.clicked.connect(
            self._remove_unselected_from_table
        )
        heatmap_btn_row.addWidget(self._remove_unselected_from_table_btn)

        self._clear_table_btn = QPushButton("Clear Table")
        self._clear_table_btn.clicked.connect(self._clear_neuron_table)
        heatmap_btn_row.addWidget(self._clear_table_btn)
        heatmap_btn_row.addStretch()
        neurons_layout.addLayout(heatmap_btn_row)

        self._neuron_table_summary_label = QLabel("")
        self._neuron_table_summary_label.setWordWrap(True)
        neurons_layout.addWidget(self._neuron_table_summary_label)

        centering_row = QHBoxLayout()
        centering_row.addWidget(QLabel("Centering:"))
        self._centering_mode_combo = QComboBox()
        self._centering_mode_combo.addItem("Same", "same")
        self._centering_mode_combo.addItem("Auto", "auto")
        self._centering_mode_combo.setCurrentIndex(0)
        centering_row.addWidget(self._centering_mode_combo)
        centering_row.addStretch()
        neurons_layout.addLayout(centering_row)

        self._render_progress = QProgressBar()
        self._render_progress.setVisible(False)
        neurons_layout.addWidget(self._render_progress)

        self._render_status_label = QLabel("")
        neurons_layout.addWidget(self._render_status_label)
        self._refresh_neuron_table_summary()
        self._refresh_apply_existing_clusters_button()
        self._update_selected_neuron_heatmap_controls()

        layout.addWidget(neurons_section)

        # --- Termini ---
        # Lives next to the table because its main use is selecting the
        # neurons that produced no termini so they can be removed before
        # clustering.
        self._termini_section_widget = TerminiSectionWidget(self.viewer)
        self._termini_section_widget.set_current_table_file_ids_provider(
            self._current_table_file_ids
        )
        self._termini_section_widget.set_selected_table_file_ids_provider(
            self._neuron_table.get_selected_file_ids
        )
        self._termini_section_widget.set_table_color_map_provider(
            self._neuron_table.get_full_color_map
        )
        self._termini_section_widget.set_select_table_file_ids_callback(
            self._neuron_table.select_file_ids
        )
        layout.addWidget(self._termini_section_widget)

        layout.addStretch()

    def _configure_selected_heatmap_menu(self) -> None:
        """Attach the combined and individual actions to Add Heatmap."""
        self._add_selected_heatmap_menu = QMenu(self._add_selected_heatmap_btn)
        self._add_single_heatmap_action = self._add_selected_heatmap_menu.addAction(
            "Single Heatmap"
        )
        self._add_single_heatmap_action.triggered.connect(
            self._add_selected_neurons_heatmap
        )
        self._add_individual_heatmaps_action = (
            self._add_selected_heatmap_menu.addAction("Individual Heatmaps")
        )
        self._add_individual_heatmaps_action.triggered.connect(
            self._add_selected_neurons_individual_heatmaps
        )
        self._add_selected_heatmap_btn.setMenu(self._add_selected_heatmap_menu)

    def _setup_regions_tab(self, parent: QWidget) -> None:
        """Set up the region selection tab."""
        from .region_appearance import RegionAppearanceEditor

        parent_layout = QVBoxLayout(parent)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        parent_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Query source:"))
        self._region_query_source_combo = QComboBox()
        self._region_query_source_combo.addItems(
            [
                _REGION_QUERY_SOURCE_ATLAS,
                _REGION_QUERY_SOURCE_CUSTOM,
                _REGION_QUERY_SOURCE_MASK,
            ]
        )
        self._region_query_source_combo.currentTextChanged.connect(
            self._on_region_query_source_changed
        )
        source_row.addWidget(self._region_query_source_combo)
        layout.addLayout(source_row)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Search scope:"))
        self._region_query_scope_combo = QComboBox()
        self._region_query_scope_combo.addItem(
            "Whole Parquet", _REGION_QUERY_SCOPE_WHOLE
        )
        self._region_query_scope_combo.addItem(
            "Current Table", _REGION_QUERY_SCOPE_CURRENT
        )
        self._region_query_scope_combo.currentTextChanged.connect(
            self._on_region_query_scope_changed
        )
        scope_row.addWidget(self._region_query_scope_combo)
        layout.addLayout(scope_row)

        node_type_row = QHBoxLayout()
        node_type_row.addWidget(QLabel("Node types:"))
        self._region_node_type_combo = NodeTypeSelectorComboBox()
        node_type_row.addWidget(self._region_node_type_combo)
        layout.addLayout(node_type_row)

        self._region_query_stack = QStackedWidget()

        atlas_page = QWidget()
        atlas_layout = QVBoxLayout(atlas_page)
        atlas_layout.setContentsMargins(0, 0, 0, 0)

        self._atlas_region_scope_stack = QStackedWidget()

        whole_page = QWidget()
        whole_layout = QVBoxLayout(whole_page)
        whole_layout.setContentsMargins(0, 0, 0, 0)
        self._whole_parquet_region_selector = RegionSelectorWidget()
        self._whole_parquet_region_selector.selection_changed.connect(
            self._on_regions_selected
        )
        whole_layout.addWidget(self._whole_parquet_region_selector)
        self._atlas_region_scope_stack.addWidget(whole_page)

        current_page = QWidget()
        current_layout = QVBoxLayout(current_page)
        current_layout.setContentsMargins(0, 0, 0, 0)
        self._current_table_region_selector = RegionSelectorWidget()
        self._current_table_region_selector.selection_changed.connect(
            self._on_regions_selected
        )
        current_layout.addWidget(self._current_table_region_selector)
        self._atlas_region_scope_stack.addWidget(current_page)

        atlas_layout.addWidget(self._atlas_region_scope_stack)
        self._region_query_stack.addWidget(atlas_page)

        custom_page = QWidget()
        custom_layout = QVBoxLayout(custom_page)
        custom_layout.setContentsMargins(0, 0, 0, 0)

        self._custom_region_scope_stack = QStackedWidget()

        custom_whole_page = QWidget()
        custom_whole_layout = QVBoxLayout(custom_whole_page)
        custom_whole_layout.setContentsMargins(0, 0, 0, 0)
        self._whole_parquet_custom_region_selector = CustomRegionSelectorWidget()
        self._whole_parquet_custom_region_selector.selection_changed.connect(
            self._on_custom_regions_selected
        )
        custom_whole_layout.addWidget(self._whole_parquet_custom_region_selector)
        self._custom_region_scope_stack.addWidget(custom_whole_page)

        custom_current_page = QWidget()
        custom_current_layout = QVBoxLayout(custom_current_page)
        custom_current_layout.setContentsMargins(0, 0, 0, 0)
        self._current_table_custom_region_selector = CustomRegionSelectorWidget()
        self._current_table_custom_region_selector.selection_changed.connect(
            self._on_custom_regions_selected
        )
        custom_current_layout.addWidget(self._current_table_custom_region_selector)
        self._custom_region_scope_stack.addWidget(custom_current_page)

        custom_layout.addWidget(self._custom_region_scope_stack)
        self._region_query_stack.addWidget(custom_page)

        mask_page = QWidget()
        mask_layout = QVBoxLayout(mask_page)
        mask_layout.setContentsMargins(0, 0, 0, 0)

        self._mask_layer_selector = MaskLayerSelectorWidget()
        self._mask_layer_selector.selection_changed.connect(
            self._on_mask_layer_selection_changed
        )
        mask_layout.addWidget(self._mask_layer_selector)

        self._mask_exclude_source_neurons_cb = QCheckBox(
            "Exclude neurons used to generate selected mask layer(s)"
        )
        set_checked = getattr(
            self._mask_exclude_source_neurons_cb,
            "setChecked",
            None,
        )
        if callable(set_checked):
            set_checked(True)
        toggled = getattr(self._mask_exclude_source_neurons_cb, "toggled", None)
        connect = getattr(toggled, "connect", None)
        if callable(connect):
            connect(lambda _checked: self._update_mask_query_summary())
        mask_layout.addWidget(self._mask_exclude_source_neurons_cb)

        self._mask_query_hint_label = QLabel("")
        self._mask_query_hint_label.setWordWrap(True)
        mask_layout.addWidget(self._mask_query_hint_label)
        mask_layout.addStretch()
        self._region_query_stack.addWidget(mask_page)

        layout.addWidget(self._region_query_stack)

        query_btn_row = QHBoxLayout()
        self._region_query_find_btn = QPushButton("Find Neurons")
        self._region_query_find_btn.clicked.connect(
            self._query_neurons_for_active_region_source
        )
        self._region_query_find_btn.setEnabled(False)
        query_btn_row.addWidget(self._region_query_find_btn)
        layout.addLayout(query_btn_row)

        clear_table_row = QHBoxLayout()
        self._regions_clear_table_btn = QPushButton("Clear Table")
        self._regions_clear_table_btn.clicked.connect(self._clear_neuron_table)
        clear_table_row.addWidget(self._regions_clear_table_btn)
        clear_table_row.addStretch()
        layout.addLayout(clear_table_row)

        self._regions_status_label = QLabel("")
        self._regions_status_label.setWordWrap(True)
        layout.addWidget(self._regions_status_label)

        self._region_appearance_section = CollapsibleSection(
            "Region Appearance",
            expanded=False,
        )
        appearance_layout = self._region_appearance_section.content_layout()
        self._region_appearance_editor = RegionAppearanceEditor()
        self._region_appearance_editor.appearance_applied.connect(
            self._on_region_appearance_applied
        )
        appearance_layout.addWidget(self._region_appearance_editor)
        layout.addWidget(self._region_appearance_section)

        layout.addStretch()
        self._on_region_query_source_changed(
            self._region_query_source_combo.currentText()
        )

    def _setup_tools_tab(self, parent: QWidget) -> None:
        """Set up the blur-generation tools tab."""
        parent_layout = QVBoxLayout(parent)

        self._tools_scroll_area = QScrollArea()
        self._tools_scroll_area.setWidgetResizable(True)
        parent_layout.addWidget(self._tools_scroll_area)

        self._tools_scroll_content = QWidget()
        self._tools_scroll_area.setWidget(self._tools_scroll_content)

        layout = QVBoxLayout(self._tools_scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)

        sources_group = QGroupBox("Heatmap Sources")
        sources_layout = QVBoxLayout(sources_group)
        self._heatmap_layer_list = QListWidget()
        self._heatmap_layer_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._heatmap_layer_list.itemSelectionChanged.connect(
            self._update_tools_controls
        )
        sources_layout.addWidget(self._heatmap_layer_list)

        gamma_btn_row = QHBoxLayout()
        self._enhance_fine_projections_btn = QPushButton("Enhance Fine Projections")
        self._enhance_fine_projections_btn.setToolTip(
            "Set gamma to 0.20 on each selected heatmap to brighten "
            "low-intensity projections."
        )
        self._enhance_fine_projections_btn.clicked.connect(
            self._enhance_selected_heatmap_projections
        )
        gamma_btn_row.addWidget(self._enhance_fine_projections_btn)

        self._reset_heatmap_gamma_btn = QPushButton("Reset Gamma")
        self._reset_heatmap_gamma_btn.setToolTip(
            "Restore gamma to 1.00 on each selected heatmap."
        )
        self._reset_heatmap_gamma_btn.clicked.connect(
            self._reset_selected_heatmap_gamma
        )
        gamma_btn_row.addWidget(self._reset_heatmap_gamma_btn)
        sources_layout.addLayout(gamma_btn_row)

        self._tools_hint_label = QLabel("")
        self._tools_hint_label.setWordWrap(True)
        sources_layout.addWidget(self._tools_hint_label)
        layout.addWidget(sources_group)

        blur_group = QGroupBox("Blur Generation")
        blur_layout = QVBoxLayout(blur_group)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Create mode:"))
        self._tools_create_mode_combo = QComboBox()
        self._tools_create_mode_combo.addItem("Separate layers", "separate")
        self._tools_create_mode_combo.addItem("Merged layer", "merged")
        self._tools_create_mode_combo.currentIndexChanged.connect(
            self._update_tools_controls
        )
        mode_row.addWidget(self._tools_create_mode_combo)
        blur_layout.addLayout(mode_row)

        sigma_row = QHBoxLayout()
        self._mask_sigma_label = QLabel("Gaussian sigma (voxels):")
        sigma_row.addWidget(self._mask_sigma_label)
        self._mask_sigma_spin = QDoubleSpinBox()
        self._mask_sigma_spin.setRange(0.0, 20.0)
        self._mask_sigma_spin.setDecimals(2)
        self._mask_sigma_spin.setSingleStep(0.25)
        self._mask_sigma_spin.setValue(1.0)
        sigma_row.addWidget(self._mask_sigma_spin)
        blur_layout.addLayout(sigma_row)
        self._mask_sigma_units_label = QLabel(
            "1 voxel = atlas voxel size; load an atlas to see the micron equivalent."
        )
        self._mask_sigma_units_label.setWordWrap(True)
        blur_layout.addWidget(self._mask_sigma_units_label)

        self._create_blur_btn = QPushButton("Create Blurred Layer")
        self._create_blur_btn.clicked.connect(self._create_blurred_layers_from_heatmaps)
        blur_layout.addWidget(self._create_blur_btn)

        layout.addWidget(blur_group)

        isolation_group = QGroupBox("Region Isolation")
        isolation_layout = QVBoxLayout(isolation_group)

        isolation_mode_row = QHBoxLayout()
        isolation_mode_row.addWidget(QLabel("Create mode:"))
        self._region_isolation_create_mode_combo = QComboBox()
        self._region_isolation_create_mode_combo.addItem("Separate layers", "separate")
        self._region_isolation_create_mode_combo.addItem("Merged layer", "merged")
        self._region_isolation_create_mode_combo.currentIndexChanged.connect(
            self._update_tools_controls
        )
        isolation_mode_row.addWidget(self._region_isolation_create_mode_combo)
        isolation_layout.addLayout(isolation_mode_row)

        with startup_timing(
            logger,
            "neuron_viewer_tools_region_selector",
            log_start=False,
        ):
            self._tools_region_selector = RegionSelectorWidget()
            self._tools_region_selector.selection_changed.connect(
                lambda _acronyms: self._update_tools_controls()
            )
            isolation_layout.addWidget(self._tools_region_selector)

        self._create_region_isolated_heatmap_btn = QPushButton(
            "Create Isolated Heatmap"
        )
        self._create_region_isolated_heatmap_btn.clicked.connect(
            self._create_region_isolated_heatmaps
        )
        isolation_layout.addWidget(self._create_region_isolated_heatmap_btn)

        layout.addWidget(isolation_group)

        self._tools_status_label = QLabel("")
        self._tools_status_label.setWordWrap(True)
        layout.addWidget(self._tools_status_label)

        layout.addStretch()
        self._update_mask_sigma_units_label()
        self._update_tools_controls()

    def _setup_histogram_tab(self, parent: QWidget) -> None:
        """Set up the histogram and thresholding tab."""
        import pyqtgraph as pg

        layout = QVBoxLayout(parent)
        self._histogram_pg = pg

        self._histogram_sources_section = CollapsibleSection(
            "Histogram Sources",
            expanded=True,
        )
        sources_layout = self._histogram_sources_section.content_layout()
        self._histogram_layer_list = QListWidget()
        self._histogram_layer_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._histogram_layer_list.itemSelectionChanged.connect(
            self._on_histogram_layer_selection_changed
        )
        sources_layout.addWidget(self._histogram_layer_list)
        self._histogram_hint_label = QLabel("")
        self._histogram_hint_label.setWordWrap(True)
        sources_layout.addWidget(self._histogram_hint_label)
        layout.addWidget(self._histogram_sources_section)

        self._histogram_plot_section = CollapsibleSection(
            "Intensity Histogram",
            expanded=True,
        )
        plot_layout = self._histogram_plot_section.content_layout()

        self._histogram_include_zero_cb = QCheckBox("Include zero-valued background")
        self._histogram_include_zero_cb.setChecked(False)
        self._histogram_include_zero_cb.toggled.connect(self._update_histogram_plot)
        plot_layout.addWidget(self._histogram_include_zero_cb)

        self._histogram_plot_widget = pg.PlotWidget()
        self._histogram_plot_widget.setMinimumHeight(260)
        self._histogram_plot_item = self._histogram_plot_widget.getPlotItem()
        self._histogram_plot_item.setTitle("Intensity Histogram")
        self._histogram_plot_item.setLabel("bottom", "Intensity")
        self._histogram_plot_item.setLabel("left", "Voxel count")
        self._histogram_plot_item.showGrid(x=True, y=True, alpha=0.15)
        self._histogram_plot_item.getViewBox().setMouseEnabled(x=True, y=True)
        self._histogram_curve_items: list = []
        self._histogram_message_item = None
        self._histogram_plot_legend = None
        self._histogram_lower_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen(color="#c43c39", width=2),
        )
        self._histogram_lower_line.setBounds((-1_000_000.0, 1_000_000.0))
        self._histogram_lower_line.sigPositionChanged.connect(
            lambda _line: self._on_histogram_bound_line_moved("lower")
        )
        self._histogram_lower_line.sigPositionChangeFinished.connect(
            lambda _line: self._on_histogram_bound_line_move_finished("lower")
        )
        self._histogram_upper_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen(color="#2f6db2", width=2),
        )
        self._histogram_upper_line.setBounds((-1_000_000.0, 1_000_000.0))
        self._histogram_upper_line.sigPositionChanged.connect(
            lambda _line: self._on_histogram_bound_line_moved("upper")
        )
        self._histogram_upper_line.sigPositionChangeFinished.connect(
            lambda _line: self._on_histogram_bound_line_move_finished("upper")
        )
        plot_layout.addWidget(self._histogram_plot_widget)

        layout.addWidget(self._histogram_plot_section)

        self._histogram_mask_section = CollapsibleSection(
            "Mask Creation",
            expanded=False,
        )
        mask_layout = self._histogram_mask_section.content_layout()

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Create mode:"))
        self._histogram_create_mode_combo = QComboBox()
        self._histogram_create_mode_combo.addItem("Separate layers", "separate")
        self._histogram_create_mode_combo.addItem("Merged layer", "merged")
        self._histogram_create_mode_combo.currentIndexChanged.connect(
            self._update_histogram_controls
        )
        mask_layout.addLayout(mode_row)
        mode_row.addWidget(self._histogram_create_mode_combo)

        lower_row = QHBoxLayout()
        lower_row.addWidget(QLabel("Lower bound:"))
        self._mask_lower_threshold_spin = QDoubleSpinBox()
        self._mask_lower_threshold_spin.setRange(-1_000_000.0, 1_000_000.0)
        self._mask_lower_threshold_spin.setDecimals(4)
        self._mask_lower_threshold_spin.setSingleStep(0.01)
        self._mask_lower_threshold_spin.setValue(1.0)
        self._mask_lower_threshold_spin.valueChanged.connect(
            self._mark_mask_bounds_manual
        )
        lower_row.addWidget(self._mask_lower_threshold_spin)
        mask_layout.addLayout(lower_row)

        bounds_btn_row = QHBoxLayout()
        self._mask_use_otsu_btn = QPushButton("Use Otsu Lower")
        self._mask_use_otsu_btn.clicked.connect(self._fill_mask_lower_bound_from_otsu)
        bounds_btn_row.addWidget(self._mask_use_otsu_btn)

        self._mask_use_contrast_btn = QPushButton("Use Layer Contrast")
        self._mask_use_contrast_btn.clicked.connect(
            self._copy_selected_layer_contrast_to_bounds
        )
        bounds_btn_row.addWidget(self._mask_use_contrast_btn)
        mask_layout.addLayout(bounds_btn_row)

        upper_row = QHBoxLayout()
        self._mask_use_upper_bound_cb = QCheckBox("Use upper bound")
        self._mask_use_upper_bound_cb.toggled.connect(self._on_mask_upper_bound_toggled)
        upper_row.addWidget(self._mask_use_upper_bound_cb)
        self._mask_upper_threshold_spin = QDoubleSpinBox()
        self._mask_upper_threshold_spin.setRange(-1_000_000.0, 1_000_000.0)
        self._mask_upper_threshold_spin.setDecimals(4)
        self._mask_upper_threshold_spin.setSingleStep(0.01)
        self._mask_upper_threshold_spin.setValue(0.0)
        self._mask_upper_threshold_spin.valueChanged.connect(
            self._mark_mask_bounds_manual
        )
        upper_row.addWidget(self._mask_upper_threshold_spin)
        mask_layout.addLayout(upper_row)

        self._create_mask_btn = QPushButton("Create Mask Layer")
        self._create_mask_btn.clicked.connect(self._create_masks_from_heatmaps)
        mask_layout.addWidget(self._create_mask_btn)

        layout.addWidget(self._histogram_mask_section)

        self._histogram_status_label = QLabel("")
        self._histogram_status_label.setWordWrap(True)
        layout.addWidget(self._histogram_status_label)

        layout.addStretch()
        self._on_mask_upper_bound_toggled(False)
        self._update_histogram_controls()
        self._update_histogram_plot()

    def _setup_viz_tab(self, parent: QWidget) -> None:
        """Set up the visualization settings tab."""
        layout = QVBoxLayout(parent)

        # Render mode
        mode_group = QGroupBox("Render Mode")
        mode_layout = QVBoxLayout(mode_group)

        self._render_mode_combo = QComboBox()
        self._render_mode_combo.addItems(["Lines", "Points", "Both"])
        self._render_mode_combo.setCurrentText("Lines")
        mode_layout.addWidget(self._render_mode_combo)

        layout.addWidget(mode_group)

        # Point settings
        point_group = QGroupBox("Point Settings")
        point_layout = QVBoxLayout(point_group)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size:"))
        self._point_size_spin = QSpinBox()
        self._point_size_spin.setRange(1, 50)
        self._point_size_spin.setValue(5)
        size_row.addWidget(self._point_size_spin)
        point_layout.addLayout(size_row)

        self._color_by_type_cb = QCheckBox("Color by node type")
        self._color_by_type_cb.setChecked(True)
        point_layout.addWidget(self._color_by_type_cb)

        layout.addWidget(point_group)

        # Line settings
        line_group = QGroupBox("Line Settings")
        line_layout = QVBoxLayout(line_group)

        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("Width:"))
        self._line_width_spin = QSpinBox()
        self._line_width_spin.setRange(1, 20)
        self._line_width_spin.setValue(4)
        self._line_width_spin.valueChanged.connect(self._update_line_width)
        width_row.addWidget(self._line_width_spin)
        line_layout.addLayout(width_row)

        layout.addWidget(line_group)

        # 2D Slice Projection settings
        slice_group = QGroupBox("2D Slice Projection")
        slice_layout = QVBoxLayout(slice_group)

        self._show_slice_projection_cb = QCheckBox("Show in 2D slices")
        self._show_slice_projection_cb.setChecked(False)
        self._show_slice_projection_cb.stateChanged.connect(
            self._toggle_slice_projection
        )
        slice_layout.addWidget(self._show_slice_projection_cb)

        self._slice_warning_label = QLabel(
            "Warning: Slice navigation is slower when projection is on."
        )
        self._slice_warning_label.setStyleSheet("color: #cc7700; font-style: italic;")
        self._slice_warning_label.setWordWrap(True)
        self._slice_warning_label.setVisible(False)
        slice_layout.addWidget(self._slice_warning_label)

        thickness_row = QHBoxLayout()
        thickness_row.addWidget(QLabel("Slice thickness (μm):"))
        self._slice_thickness_spin = QSpinBox()
        self._slice_thickness_spin.setRange(10, 2500)
        self._slice_thickness_spin.setValue(100)
        self._slice_thickness_spin.valueChanged.connect(self._update_slice_thickness)
        thickness_row.addWidget(self._slice_thickness_spin)
        slice_layout.addLayout(thickness_row)

        layout.addWidget(slice_group)

        # Opacity
        opacity_group = QGroupBox("Opacity")
        opacity_layout = QHBoxLayout(opacity_group)

        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(80)
        opacity_layout.addWidget(self._opacity_slider)

        self._opacity_label = QLabel("80%")
        self._opacity_slider.valueChanged.connect(
            lambda v: self._opacity_label.setText(f"{v}%")
        )
        opacity_layout.addWidget(self._opacity_label)

        layout.addWidget(opacity_group)

        layout.addStretch()

    def _setup_reference_tab(self, parent: QWidget) -> None:
        """Set up the reference layers tab."""
        layout = QVBoxLayout(parent)

        # Template
        template_group = QGroupBox("Reference Template")
        template_layout = QVBoxLayout(template_group)

        self._show_template_cb = QCheckBox("Show template")
        self._show_template_cb.setChecked(False)
        self._show_template_cb.setToolTip(
            "Load and show the Allen reference template on demand."
        )
        self._show_template_cb.stateChanged.connect(self._toggle_template)
        template_layout.addWidget(self._show_template_cb)

        template_opacity_row = QHBoxLayout()
        template_opacity_row.addWidget(QLabel("Opacity:"))
        self._template_opacity_slider = QSlider(Qt.Horizontal)
        self._template_opacity_slider.setRange(0, 100)
        self._template_opacity_slider.setValue(30)
        self._template_opacity_slider.valueChanged.connect(
            self._update_template_opacity
        )
        template_opacity_row.addWidget(self._template_opacity_slider)
        template_layout.addLayout(template_opacity_row)

        layout.addWidget(template_group)

        # Brain outline
        outline_group = QGroupBox("Brain Outline")
        outline_layout = QVBoxLayout(outline_group)

        self._show_outline_cb = QCheckBox("Show brain outline")
        self._show_outline_cb.setChecked(False)
        self._show_outline_cb.stateChanged.connect(self._toggle_outline)
        outline_layout.addWidget(self._show_outline_cb)

        layout.addWidget(outline_group)

        # Region meshes
        mesh_group = QGroupBox("Region Meshes")
        mesh_layout = QVBoxLayout(mesh_group)

        self._show_region_meshes_cb = QCheckBox("Show selected region meshes")
        self._show_region_meshes_cb.setChecked(False)
        self._show_region_meshes_cb.stateChanged.connect(self._toggle_region_meshes)
        mesh_layout.addWidget(self._show_region_meshes_cb)

        self._region_mesh_scope_label = QLabel(
            "Only directly selected top-level parent regions are shown as meshes. "
            "Child-region meshes are omitted because loading and rendering them "
            "would incur a higher computational cost."
        )
        self._region_mesh_scope_label.setWordWrap(True)
        mesh_layout.addWidget(self._region_mesh_scope_label)

        mesh_opacity_row = QHBoxLayout()
        mesh_opacity_row.addWidget(QLabel("Opacity:"))
        self._mesh_opacity_slider = QSlider(Qt.Horizontal)
        self._mesh_opacity_slider.setRange(0, 100)
        self._mesh_opacity_slider.setValue(30)
        self._mesh_opacity_slider.valueChanged.connect(self._update_mesh_opacity)
        mesh_opacity_row.addWidget(self._mesh_opacity_slider)
        mesh_layout.addLayout(mesh_opacity_row)

        layout.addWidget(mesh_group)

        # Region Segmentation (2D)
        seg_group = QGroupBox("Region Segmentation (2D)")
        seg_layout = QVBoxLayout(seg_group)

        self._show_region_seg_cb = QCheckBox("Show selected region segmentation")
        self._show_region_seg_cb.setChecked(False)
        self._show_region_seg_cb.stateChanged.connect(self._toggle_region_segmentation)
        seg_layout.addWidget(self._show_region_seg_cb)

        seg_opacity_row = QHBoxLayout()
        seg_opacity_row.addWidget(QLabel("Opacity:"))
        self._seg_opacity_slider = QSlider(Qt.Horizontal)
        self._seg_opacity_slider.setRange(0, 100)
        self._seg_opacity_slider.setValue(30)
        self._seg_opacity_slider.valueChanged.connect(self._update_seg_opacity)
        seg_opacity_row.addWidget(self._seg_opacity_slider)
        seg_layout.addLayout(seg_opacity_row)

        layout.addWidget(seg_group)

        layout.addStretch()

    def _load_parquet(self) -> None:
        """Open file dialog and load Parquet file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Parquet File",
            "",
            "Parquet Files (*.parquet);;All Files (*)",
        )

        if not filepath:
            return

        try:
            self._load_parquet_path(filepath)
            logger.info(f"Loaded Parquet file: {filepath}")

        except Exception as e:
            logger.error(f"Failed to load Parquet file: {e}")
            self._file_label.setText(f"Error: {e}")

    def _load_parquet_path(self, filepath: str | Path) -> None:
        """Load a neuron Parquet path and capture optional enhanced metadata."""
        filepath = str(filepath)
        new_db = NeuronDatabase(filepath)
        old_db = self._db
        self._db = new_db
        if old_db is not None:
            try:
                old_db.close()
            except Exception:
                logger.debug("Failed to close previous neuron database", exc_info=True)
        self._set_current_project_path(None)
        comparison_restorer = getattr(self, "_restore_comparison_board", None)
        if callable(comparison_restorer):
            comparison_restorer(None)

        self._file_label.setText(Path(filepath).name)

        stats = self._db.get_statistics()
        stats_text = (
            f"Nodes: {stats['n_nodes']:,} | "
            f"Files: {stats['n_files']:,} | "
            f"Subjects: {stats['n_subjects']:,} | "
            f"Regions: {stats['n_regions']:,}"
        )

        enhanced_count = self._load_enhanced_table_state(filepath)
        if enhanced_count:
            stats_text += f" | Enhanced labels: {enhanced_count:,}"
        transform_info = self._load_flatmap_transform_status(filepath)
        if transform_info:
            stats_text += f" | Transform: {transform_info}"
        self._stats_label.setText(stats_text)

        self._set_region_query_buttons_enabled(True)
        self._analysis_tab.set_database(self._db)
        self._termini_section_widget.set_database(self._db)
        self._regions_status_label.setText("")
        flatmap_tab = getattr(self, "_flatmap_tab", None)
        invalidate_flatmap = getattr(
            flatmap_tab,
            "invalidate_loaded_parquet_projection",
            None,
        )
        if callable(invalidate_flatmap):
            invalidate_flatmap()
        refresh_cache = getattr(flatmap_tab, "refresh_cache_profiles", None)
        if callable(refresh_cache):
            refresh_cache()
        saved_state_applier = getattr(self, "_apply_saved_table_state_to_table", None)
        if callable(saved_state_applier):
            saved_state_applier()

    def _load_flatmap_transform_status(self, filepath: str | Path) -> str:
        """Display whether the loaded parquet has reusable flatmap/depth columns."""
        label = getattr(self, "_flatmap_transform_status_label", None)
        try:
            info = read_flatmap_parquet_transform_info(filepath)
        except Exception:
            logger.debug(
                "No readable flatmap transform metadata for %s",
                filepath,
                exc_info=True,
            )
            if label is not None:
                label.setText("")
            return ""

        transform_text = info.present_transform_text
        if not transform_text:
            if label is not None:
                label.setText("")
            return ""

        if info.has_full_transform:
            if int(getattr(info, "format_version", 0) or 0) >= 3:
                message = (
                    "Loaded version-3 Parquet contains bilateral shaped, square, "
                    "and depth transform columns "
                    f"(lookup set {getattr(info, 'lookup_set_id', None)}). "
                    "The Flatmap tab uses them by default; NRRD conversion only "
                    "runs when Recompute from NRRDs is selected explicitly."
                )
            else:
                message = (
                    f"Loaded Parquet contains {transform_text} transform columns. "
                    "The Flatmap tab can render it without loading NRRD files."
                )
        else:
            message = (
                f"Loaded Parquet contains {transform_text} transform columns. "
                "Flatmap rendering without NRRDs requires x_flat, y_flat, and "
                "depth_um columns."
            )
        if info.uses_legacy_mirror_fallback:
            warning = (
                " Warning: this version-1 transform used the legacy full-mirror "
                "fallback, so mirrored rows may be placed in the opposite flatmap "
                "hemisphere. Regenerate the augmented Parquet or select the original "
                "flatmap/depth NRRDs to recompute corrected coordinates."
            )
            message += warning
            logger.warning("%s: %s", Path(filepath).name, warning.strip())
        if label is not None:
            label.setText(message)
        return transform_text

    def _load_enhanced_table_state(self, filepath: str | Path) -> int:
        """Read enhanced parquet table metadata without making it mandatory."""
        self._saved_table_state = {}
        self._cluster_assignment_store.load_state({})
        refresh_assignments = getattr(
            self,
            "_refresh_cluster_assignment_controls",
            None,
        )
        if callable(refresh_assignments):
            refresh_assignments()
        table = getattr(self, "_neuron_table", None)
        refresh_table_assignments = getattr(
            table,
            "refresh_cluster_assignments",
            None,
        )
        if callable(refresh_table_assignments):
            refresh_table_assignments(preserve_selection=True)
        analysis_tab = getattr(self, "_analysis_tab", None)
        assignment_handler = getattr(
            analysis_tab,
            "on_active_cluster_assignment_changed",
            None,
        )
        if callable(assignment_handler):
            assignment_handler()
        try:
            payload = read_enhanced_parquet_metadata(filepath)
        except Exception:
            logger.debug(
                "No readable enhanced parquet metadata for %s",
                filepath,
                exc_info=True,
            )
            return 0

        table_state = payload.get("table_state")
        if not isinstance(table_state, dict):
            return 0
        entries = table_state.get("entries", [])
        if not entries:
            return 0
        self._saved_table_state = table_state
        assignment_state = table_state.get("cluster_assignments")
        if isinstance(assignment_state, dict):
            self._cluster_assignment_store.load_state(assignment_state)
            if callable(refresh_assignments):
                refresh_assignments()
            if callable(refresh_table_assignments):
                refresh_table_assignments(preserve_selection=True)
            if callable(assignment_handler):
                assignment_handler()
        return len(entries)

    def _current_table_state(self) -> dict[str, object]:
        """Return the current neuron table state for project/enhanced exports."""
        exporter = getattr(self._neuron_table, "export_state", None)
        if callable(exporter):
            return exporter()
        return {"version": 1, "entries": []}

    def _analysis_project_metadata(self) -> dict[str, object]:
        """Return compact analysis metadata for project-bundle JSON."""
        result = getattr(self._analysis_tab, "_last_cluster_result", None)
        if result is None:
            return {}

        run_metadata = getattr(result, "metadata", None)
        payload: dict[str, object] = {
            "run_metadata": (
                run_metadata.to_dict()
                if run_metadata is not None and hasattr(run_metadata, "to_dict")
                else None
            ),
            "neuron_ids": list(getattr(result, "neuron_ids", [])),
        }
        labels = getattr(result, "labels", None)
        if labels is not None:
            payload["cluster_labels"] = [
                int(value) for value in np.asarray(labels).reshape(-1)
            ]
        reorder = getattr(result, "reorder_indices", None)
        if reorder is not None:
            payload["dendrogram_order"] = [
                int(value) for value in np.asarray(reorder).reshape(-1)
            ]
        return payload

    def _set_current_project_path(self, path: str | Path | None) -> None:
        """Set the current project target and refresh its save control."""
        self._current_project_path = (
            Path(path).expanduser().resolve() if path is not None else None
        )
        button = getattr(self, "_save_project_btn", None)
        if button is None:
            return
        button.setEnabled(self._current_project_path is not None)
        if self._current_project_path is None:
            button.setToolTip(
                "Load or create a Neuron Navigator project before saving in place."
            )
        else:
            button.setToolTip(f"Save changes to {self._current_project_path}")

    def _set_project_save_in_progress(self, saving: bool) -> None:
        """Disable save actions during synchronous project serialization."""
        self._save_project_as_btn.setEnabled(not saving)
        self._save_project_btn.setEnabled(
            not saving and self._current_project_path is not None
        )

    def _save_project_as_dialog(self) -> None:
        """Prompt for and create a new lossless project bundle directory."""
        if self._db is None:
            message = "Load a neuron Parquet before saving a project."
            self._regions_status_label.setText(message)
            show_warning(message)
            return

        if self._current_project_path is None:
            default_name = f"{Path(self._db.parquet_path).stem}{PROJECT_SUFFIX}"
        else:
            default_name = str(
                self._current_project_path.with_name(
                    f"{self._current_project_path.stem}_copy{PROJECT_SUFFIX}"
                )
            )
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Neuron Navigator Project As",
            default_name,
            f"Neuron Navigator Project (*{PROJECT_SUFFIX});;All Files (*)",
        )
        if not output_path:
            return
        bundle_path = Path(output_path)
        if bundle_path.suffix.lower() != PROJECT_SUFFIX:
            bundle_path = bundle_path.with_suffix(PROJECT_SUFFIX)
        if bundle_path.exists() or bundle_path.is_symlink():
            message = (
                f"Project destination already exists: {bundle_path}. "
                "Choose a new destination. Save Project can only overwrite the "
                "current project."
            )
            self._project_status_label.setText(message)
            self._regions_status_label.setText(message)
            show_warning(message)
            return

        self._save_project_to_path(bundle_path, overwrite=False)

    @staticmethod
    def _project_overwrite_confirmation_text(bundle_path: Path) -> str:
        """Return the destructive-save confirmation text for a project path."""
        return (
            "Replace the current Neuron Navigator project?\n\n"
            f"{bundle_path}\n\n"
            "All existing contents in this project folder will be replaced. "
            "This cannot be undone."
        )

    def _confirm_project_overwrite(self, bundle_path: Path) -> bool:
        """Ask whether the current recognized project may be overwritten."""
        from qtpy.QtWidgets import QMessageBox

        message = QMessageBox(self)
        message.setIcon(QMessageBox.Warning)
        message.setWindowTitle("Overwrite Neuron Navigator Project?")
        message.setText(self._project_overwrite_confirmation_text(bundle_path))
        overwrite_button = message.addButton("Overwrite", QMessageBox.DestructiveRole)
        cancel_button = message.addButton("Cancel", QMessageBox.RejectRole)
        message.setDefaultButton(cancel_button)
        message.exec()
        return message.clickedButton() is overwrite_button

    def _save_current_project(self) -> None:
        """Confirm and overwrite the project associated with this session."""
        bundle_path = self._current_project_path
        if bundle_path is None:
            return
        if not is_recognized_project_bundle(bundle_path):
            message = (
                "The current project is no longer available or is not a recognized "
                f"Neuron Navigator project: {bundle_path}"
            )
            self._set_current_project_path(None)
            self._project_status_label.setText(message)
            self._regions_status_label.setText(message)
            show_warning(message)
            return
        if not self._confirm_project_overwrite(bundle_path):
            return
        self._save_project_to_path(bundle_path, overwrite=True)

    def _save_project_to_path(self, bundle_path: Path, *, overwrite: bool) -> None:
        """Save current session state to one new or replaceable project path."""
        if self._db is None:
            message = "Load a neuron Parquet before saving a project."
            self._regions_status_label.setText(message)
            show_warning(message)
            return
        appearance_editor = getattr(self, "_region_appearance_editor", None)
        if (
            appearance_editor is not None
            and not appearance_editor.resolve_unapplied_before_save()
        ):
            return

        self._set_project_save_in_progress(True)
        self._project_progress.setVisible(True)
        self._project_progress.setRange(0, 0)
        self._project_progress.setValue(0)
        self._project_status_label.setText("Saving project bundle...")
        QApplication.processEvents()

        def _on_save_progress(message: str, current: int, total: int) -> None:
            self._project_progress.setVisible(True)
            if total > 0:
                self._project_progress.setRange(0, total)
                self._project_progress.setValue(max(0, min(current, total)))
            else:
                self._project_progress.setRange(0, 0)
            self._project_status_label.setText(message)
            QApplication.processEvents()

        try:
            appearance_getter = getattr(self, "_region_appearance_store", None)
            appearance_store = (
                appearance_getter()
                if callable(appearance_getter)
                else RegionAppearanceStore()
            )
            comparison_getter = getattr(
                self,
                "_comparison_state_payload",
                None,
            )
            saved = save_project_bundle(
                bundle_path,
                source_parquet_path=self._db.parquet_path,
                table_state=self._current_table_state(),
                layers=self._iter_viewer_layers(),
                atlas_name=self._current_atlas_name(),
                analysis_metadata=self._analysis_project_metadata(),
                region_appearance=appearance_store.to_palette_dict(),
                flatmap_cache_reference=(self._flatmap_tab.active_cache_reference()),
                comparison_board=(
                    comparison_getter() if callable(comparison_getter) else None
                ),
                progress_callback=_on_save_progress,
                overwrite=overwrite,
            )
        except Exception as exc:
            logger.error("Failed to save project bundle: %s", exc)
            if overwrite and not is_recognized_project_bundle(bundle_path):
                self._set_current_project_path(None)
            self._set_project_save_in_progress(False)
            self._project_progress.setVisible(False)
            self._project_progress.setRange(0, 1)
            self._project_progress.setValue(0)
            self._project_status_label.setText(f"Failed to save project: {exc}")
            self._regions_status_label.setText(f"Failed to save project: {exc}")
            show_warning(f"Failed to save project: {exc}")
            return

        self._set_current_project_path(saved)
        self._set_project_save_in_progress(False)
        self._project_progress.setVisible(False)
        self._project_progress.setRange(0, 1)
        self._project_progress.setValue(0)
        message = f"Saved project bundle: {saved.name}"
        self._project_status_label.setText(message)
        self._regions_status_label.setText(message)
        show_info(message)

    def _load_project_bundle_dialog(self) -> None:
        """Prompt for and restore a project bundle directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Open Neuron Navigator Project",
            "",
        )
        if not directory:
            return
        try:
            bundle = load_project_bundle(directory)
            self._restore_project_bundle(bundle)
        except Exception as exc:
            logger.error("Failed to load project bundle: %s", exc)
            self._regions_status_label.setText(f"Failed to load project: {exc}")
            show_warning(f"Failed to load project: {exc}")
            return

        message = f"Loaded project bundle: {Path(directory).name}"
        self._regions_status_label.setText(message)
        show_info(message)

    def _export_enhanced_parquet_dialog(self) -> None:
        """Prompt for and export the loaded neuron parquet with table metadata."""
        if self._db is None:
            message = "Load a neuron Parquet before exporting enhanced Parquet."
            self._regions_status_label.setText(message)
            show_warning(message)
            return

        default_name = f"{Path(self._db.parquet_path).stem}_enhanced.parquet"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Enhanced Neuron Parquet",
            default_name,
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if not output_path:
            return

        try:
            saved = export_enhanced_neuron_parquet(
                self._db.parquet_path,
                output_path,
                table_state=self._current_table_state(),
                metadata={
                    "atlas_name": self._current_atlas_name(),
                    "analysis": self._analysis_project_metadata(),
                },
            )
        except Exception as exc:
            logger.error("Failed to export enhanced parquet: %s", exc)
            self._regions_status_label.setText(
                f"Failed to export enhanced Parquet: {exc}"
            )
            show_warning(f"Failed to export enhanced Parquet: {exc}")
            return

        message = f"Exported enhanced Parquet: {saved.name}"
        self._regions_status_label.setText(message)
        show_info(message)

    def _restore_project_bundle(self, bundle: ProjectBundle) -> None:
        """Restore a loaded project bundle into the current widget/viewer."""
        self._load_parquet_path(bundle.source_parquet_path)
        self._saved_table_state = dict(bundle.table_state)
        if bundle.region_appearance:
            try:
                appearance_store = RegionAppearanceStore.from_palette_dict(
                    bundle.region_appearance
                )
                appearance_editor = getattr(
                    self,
                    "_region_appearance_editor",
                    None,
                )
                if appearance_editor is not None:
                    appearance_editor.load_applied_store(appearance_store)
            except Exception as exc:
                logger.warning("Could not restore region appearance: %s", exc)
        cache_reference = getattr(bundle, "flatmap_cache_reference", None)
        if cache_reference:
            try:
                self._flatmap_tab.restore_cache_reference(cache_reference)
            except Exception as exc:
                logger.warning(
                    "Could not restore external flatmap cache reference: %s",
                    exc,
                )
                self._regions_status_label.setText(
                    f"Project loaded; flatmap cache unavailable: {exc}"
                )

        importer = getattr(self._neuron_table, "import_state", None)
        if callable(importer):
            importer(bundle.table_state)
            self._neuron_table.set_added_file_ids(self._current_scene_file_ids())
        self._refresh_cluster_assignment_controls()
        self._apply_active_cluster_assignment()

        self._restore_project_layers(bundle)
        self._restore_comparison_board(getattr(bundle, "comparison_board", None))
        if bundle.region_appearance:
            self._on_region_appearance_applied(self._region_appearance_store())
        self._sync_neuron_table_heatmap_membership()
        self._refresh_manual_heatmap_combo()
        self._refresh_heatmap_layer_list()
        self._refresh_histogram_layer_list()
        self._refresh_mask_layer_options()
        self._sync_after_neuron_table_membership_change()
        self._set_current_project_path(bundle.path)

    def _restore_project_layers(self, bundle: ProjectBundle) -> None:
        """Recreate saved app-created image and label layers from bundle arrays."""
        for project_layer in bundle.layers:
            payload = project_layer.metadata
            layer_name = str(payload.get("name") or project_layer.layer_id)
            self._remove_existing_app_layer(layer_name)

            layer_type = str(payload.get("layer_type") or "image")
            display = payload.get("display", {})
            if not isinstance(display, dict):
                display = {}
            layer_metadata = payload.get("metadata", {})
            if not isinstance(layer_metadata, dict):
                layer_metadata = {}

            kwargs = {
                "name": layer_name,
                "opacity": float(display.get("opacity", 0.7)),
                "visible": bool(display.get("visible", True)),
                "metadata": layer_metadata,
            }
            for attr_name in ("scale", "translate"):
                if attr_name in display and display[attr_name] is not None:
                    kwargs[attr_name] = display[attr_name]

            if layer_type == "labels":
                self._restore_project_labels_layer(
                    project_layer.data,
                    kwargs,
                    display,
                    layer_metadata,
                )
            else:
                self._restore_project_image_layer(
                    project_layer.data,
                    kwargs,
                    display,
                    layer_metadata,
                )

    def _remove_existing_app_layer(self, layer_name: str) -> None:
        """Remove an existing app-created layer with the same name, if present."""
        for layer in list(self._iter_viewer_layers()):
            if getattr(layer, "name", None) != layer_name:
                continue
            metadata = _layer_metadata(layer)
            if metadata.get("heatmap_source") or metadata.get("mask_query_source"):
                self.viewer.layers.remove(layer)
                return

    def _restore_project_labels_layer(
        self,
        data: np.ndarray,
        kwargs: dict[str, object],
        display: dict[str, object],
        layer_metadata: dict[str, object],
    ) -> None:
        """Restore one saved labels layer with a binary color when available."""
        restored = self._apply_project_colormap_kwargs(kwargs, display)
        if restored:
            self.viewer.add_labels(data, **kwargs)
            return

        color = layer_metadata.get("color")
        try:
            from napari.utils import DirectLabelColormap

            rgba = (
                np.asarray(color, dtype=np.float32)
                if color is not None
                else np.asarray((0.8, 0.8, 0.8, 1.0), dtype=np.float32)
            )
            kwargs["colormap"] = DirectLabelColormap(
                color_dict={
                    None: np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32),
                    0: np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32),
                    1: rgba,
                }
            )
        except Exception:
            logger.debug("Could not restore project labels colormap", exc_info=True)
        self.viewer.add_labels(data, **kwargs)

    def _restore_project_image_layer(
        self,
        data: np.ndarray,
        kwargs: dict[str, object],
        display: dict[str, object],
        layer_metadata: dict[str, object],
    ) -> None:
        """Restore one saved image layer with app metadata and display options."""
        for attr_name in ("blending", "rendering", "gamma"):
            if attr_name in display and display[attr_name] is not None:
                kwargs[attr_name] = display[attr_name]
        limits = display.get("contrast_limits")
        if limits is not None:
            kwargs["contrast_limits"] = limits

        color = layer_metadata.get("color")
        restored = self._apply_project_colormap_kwargs(kwargs, display)
        if not restored and color is not None:
            try:
                from napari.utils.colormaps import Colormap

                rgba = np.asarray(color, dtype=float).reshape(-1)
                if rgba.size >= 3:
                    alpha = float(rgba[3]) if rgba.size >= 4 else 1.0
                    kwargs["colormap"] = Colormap(
                        colors=[
                            [0.0, 0.0, 0.0, 0.0],
                            [float(rgba[0]), float(rgba[1]), float(rgba[2]), alpha],
                        ],
                        name=f"restored_{str(kwargs['name']).lower().replace(' ', '_')}",
                    )
            except Exception:
                logger.debug("Could not restore project image colormap", exc_info=True)

        self.viewer.add_image(data, **kwargs)

    @staticmethod
    def _project_label_key(payload: object) -> object:
        """Return a labels-colormap key from a saved JSON key payload."""
        if not isinstance(payload, dict):
            return payload
        key_type = payload.get("type")
        value = payload.get("value")
        if key_type == "none":
            return None
        if key_type == "int":
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        return value

    def _apply_project_colormap_kwargs(
        self,
        kwargs: dict[str, object],
        display: dict[str, object],
    ) -> bool:
        """Apply a saved project colormap to layer-add kwargs when possible."""
        payload = display.get("colormap")
        if not isinstance(payload, dict):
            return False

        payload_type = payload.get("type")
        try:
            if payload_type == "direct_label_colormap":
                from napari.utils import DirectLabelColormap

                color_dict = {}
                for item in payload.get("color_dict", []):
                    if not isinstance(item, dict):
                        continue
                    color = item.get("color")
                    if color is None:
                        continue
                    color_dict[self._project_label_key(item.get("label"))] = np.asarray(
                        color,
                        dtype=np.float32,
                    )
                if not color_dict:
                    return False
                kwargs["colormap"] = DirectLabelColormap(
                    color_dict=color_dict,
                    use_selection=bool(payload.get("use_selection", False)),
                )
                return True

            if payload_type == "colormap":
                from napari.utils.colormaps import Colormap

                colors = payload.get("colors")
                if colors is None:
                    return False
                colormap_kwargs = {
                    "colors": colors,
                    "name": str(payload.get("name") or "restored_colormap"),
                }
                if payload.get("controls") is not None:
                    colormap_kwargs["controls"] = payload.get("controls")
                if payload.get("interpolation") is not None:
                    colormap_kwargs["interpolation"] = payload.get("interpolation")
                for key in ("low_color", "high_color", "nan_color"):
                    if payload.get(key) is not None:
                        colormap_kwargs[key] = payload.get(key)
                kwargs["colormap"] = Colormap(**colormap_kwargs)
                return True

            if payload_type == "named_colormap" and payload.get("name"):
                kwargs["colormap"] = str(payload["name"])
                return True
        except Exception:
            logger.debug("Could not restore saved project colormap", exc_info=True)
        return False

    def _load_atlas(
        self,
        _checked: bool = False,
        *,
        pending_reference_action: str | None = None,
    ) -> None:
        """Load the selected BrainGlobe atlas in the background."""
        atlas_name = self._atlas_combo.currentText()

        if self._cached_atlas_autoload_running():
            self._atlas_status_label.setText(
                f"Atlas: Loading cached {atlas_name} in background..."
            )
            return

        if self._atlas_load_running():
            if pending_reference_action is not None:
                self._pending_reference_action = pending_reference_action
            self._atlas_status_label.setText(
                f"Atlas: Loading {atlas_name} in background..."
            )
            self._show_atlas_progress()
            return

        from ..workers import AtlasLoadWorker

        self._pending_reference_action = pending_reference_action
        self._atlas_status_label.setText(f"Atlas: Preparing to load {atlas_name}...")
        self._atlas_status_label.repaint()
        self._set_atlas_load_controls_enabled(False)
        self._on_atlas_load_progress(0, 0, 0)

        thread = QThread()
        worker = AtlasLoadWorker(atlas_name)
        self._atlas_load_thread = thread
        self._atlas_load_worker = worker
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.status.connect(self._on_atlas_load_status)
        worker.progress.connect(self._on_atlas_load_progress)
        worker.finished.connect(self._on_atlas_load_finished)
        worker.error.connect(self._on_atlas_load_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._cleanup_atlas_load_thread(thread, worker))
        thread.start()

    def _atlas_load_running(self) -> bool:
        """Return whether an explicit atlas load worker is active."""
        thread = getattr(self, "_atlas_load_thread", None)
        is_running = getattr(thread, "isRunning", None)
        return bool(thread is not None and callable(is_running) and is_running())

    def _show_atlas_progress(self) -> None:
        """Show the atlas progress bar when the Data tab controls exist."""
        progress = getattr(self, "_atlas_progress", None)
        if progress is not None and hasattr(progress, "setVisible"):
            progress.setVisible(True)

    def _hide_atlas_progress(self) -> None:
        """Hide and reset the atlas progress bar."""
        progress = getattr(self, "_atlas_progress", None)
        if progress is None:
            return
        if hasattr(progress, "setRange"):
            progress.setRange(0, 100)
        if hasattr(progress, "setValue"):
            progress.setValue(0)
        if hasattr(progress, "setVisible"):
            progress.setVisible(False)

    def _set_atlas_load_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable atlas controls while a load is running."""
        for attr_name in ("_atlas_combo", "_load_atlas_btn"):
            widget = getattr(self, attr_name, None)
            set_enabled = getattr(widget, "setEnabled", None)
            if callable(set_enabled):
                set_enabled(enabled)

    def _on_atlas_load_status(self, message: str) -> None:
        """Display atlas worker status text."""
        self._atlas_status_label.setText(message)

    def _on_atlas_load_progress(
        self,
        minimum: int,
        maximum: int,
        value: int,
    ) -> None:
        """Display atlas worker progress."""
        progress = getattr(self, "_atlas_progress", None)
        if progress is None:
            return
        if hasattr(progress, "setVisible"):
            progress.setVisible(True)
        if hasattr(progress, "setRange"):
            progress.setRange(int(minimum), int(maximum))
        if hasattr(progress, "setValue"):
            progress.setValue(int(value))

    def _on_atlas_load_finished(self, atlas) -> None:
        """Apply a background-loaded atlas and complete pending UI actions."""
        atlas_name = str(
            getattr(atlas, "atlas_name", None) or self._atlas_combo.currentText()
        )
        pending_action = self._pending_reference_action
        self._pending_reference_action = None
        self._set_atlas_load_controls_enabled(True)
        self._hide_atlas_progress()

        try:
            with startup_timing(
                logger,
                "atlas_load_apply",
                atlas=atlas_name,
            ) as timing:
                self._apply_loaded_atlas(atlas, atlas_name)
                timing.set(structures=len(getattr(atlas, "structures", {})))
            logger.info(f"Loaded atlas: {atlas_name}")

            if pending_action is not None:
                self._complete_pending_reference_action(pending_action)
            else:
                self._atlas_status_label.setText(
                    f"Atlas: {atlas_name} "
                    f"({len(getattr(atlas, 'structures', {}))} structures). "
                    f"{_ATLAS_REFERENCE_PROMPT}"
                )
                show_info(_ATLAS_REFERENCE_PROMPT)
        except Exception as e:
            logger.exception("Failed to apply loaded atlas")
            self._pending_reference_action = pending_action
            self._on_atlas_load_error(str(e))

    def _on_atlas_load_error(self, error_msg: str) -> None:
        """Handle explicit atlas load failure."""
        logger.error("Atlas load failed: %s", error_msg)
        pending_action = self._pending_reference_action
        self._pending_reference_action = None
        self._set_atlas_load_controls_enabled(True)
        self._hide_atlas_progress()
        if pending_action is not None:
            self._set_reference_action_checked(
                pending_action,
                False,
                emit_signal=False,
            )
        self._atlas_status_label.setText(f"Atlas: Error - {error_msg}")
        self._update_mask_sigma_units_label()
        self._update_point_import_controls()

    def _apply_loaded_atlas(self, atlas, atlas_name: str) -> None:
        """Store a loaded atlas and refresh atlas-dependent UI."""
        self._atlas = atlas

        selectors = []
        for attr_name in (
            "_whole_parquet_region_selector",
            "_current_table_region_selector",
            "_region_selector",
            "_tools_region_selector",
        ):
            selector = getattr(self, attr_name, None)
            if selector is None or any(
                existing is selector for _name, existing in selectors
            ):
                continue
            selectors.append((attr_name, selector))

        for attr_name, selector in selectors:
            set_atlas = getattr(selector, "set_atlas", None)
            if not callable(set_atlas):
                continue
            with startup_timing(
                logger,
                "load_atlas_selector",
                selector=attr_name,
                atlas=atlas_name,
            ) as selector_timing:
                set_atlas(atlas)
                selector_timing.set(items=len(getattr(selector, "_items_by_id", {})))
        self._set_custom_region_hierarchy_for_atlas(atlas)
        appearance_editor = getattr(self, "_region_appearance_editor", None)
        if appearance_editor is not None:
            appearance_editor.set_atlas(atlas)
            self._refresh_region_appearance_selection()
            apply_appearance = getattr(self, "_on_region_appearance_applied", None)
            if callable(apply_appearance):
                apply_appearance(appearance_editor.applied_store)
        self._atlas_status_label.setText(
            f"Atlas: {atlas_name} ({len(atlas.structures)} structures)"
        )
        with startup_timing(
            logger,
            "load_atlas_phase",
            phase="analysis_tab_set_atlas",
            atlas=atlas_name,
        ):
            self._analysis_tab.set_atlas(atlas)
        self._termini_section_widget.set_atlas(atlas)
        flatmap_tab = getattr(self, "_flatmap_tab", None)
        refresh_cache_profiles = getattr(
            flatmap_tab,
            "refresh_cache_profiles",
            None,
        )
        if callable(refresh_cache_profiles):
            with startup_timing(
                logger,
                "load_atlas_phase",
                phase="refresh_flatmap_cache_profiles",
                atlas=atlas_name,
            ):
                refresh_cache_profiles()
        with startup_timing(
            logger,
            "load_atlas_phase",
            phase="update_mask_sigma_units_label",
            atlas=atlas_name,
        ):
            self._update_mask_sigma_units_label()
        with startup_timing(
            logger,
            "load_atlas_phase",
            phase="refresh_heatmap_layer_list",
            atlas=atlas_name,
        ):
            self._refresh_heatmap_layer_list()
        with startup_timing(
            logger,
            "load_atlas_phase",
            phase="refresh_histogram_layer_list",
            atlas=atlas_name,
        ):
            self._refresh_histogram_layer_list()
        with startup_timing(
            logger,
            "load_atlas_phase",
            phase="refresh_mask_layer_options",
            atlas=atlas_name,
        ):
            self._refresh_mask_layer_options()
        with startup_timing(
            logger,
            "load_atlas_phase",
            phase="update_point_import_controls",
            atlas=atlas_name,
        ):
            self._update_point_import_controls()
        comparison_window = getattr(self, "_comparison_window", None)
        if comparison_window is not None and comparison_window.isVisible():
            comparison_window.refresh_sources()

    def _set_custom_region_hierarchy_for_atlas(self, atlas: object) -> None:
        """Populate every scope-specific custom selector from one atlas."""
        custom_selectors = tuple(
            selector
            for selector in (
                getattr(
                    self,
                    "_whole_parquet_custom_region_selector",
                    None,
                ),
                getattr(
                    self,
                    "_current_table_custom_region_selector",
                    None,
                ),
            )
            if selector is not None
        )
        if not custom_selectors:
            return

        try:
            hierarchy = isocortex_layer_hierarchy_from_atlas(atlas)
        except ValueError as exc:
            message = (
                f"Custom Isocortex Layers are unavailable for the loaded atlas: {exc}"
            )
            logger.warning(message)
            for selector in custom_selectors:
                selector.clear_with_message(message)
            return

        for selector in custom_selectors:
            selector.set_hierarchy(hierarchy)

    def _cached_atlas_autoload_running(self) -> bool:
        """Return whether a cached atlas auto-load worker is active."""
        thread = getattr(self, "_cached_atlas_thread", None)
        is_running = getattr(thread, "isRunning", None)
        return bool(thread is not None and callable(is_running) and is_running())

    def _set_template_checkbox_checked(
        self,
        checked: bool,
        *,
        emit_signal: bool,
    ) -> None:
        """Set the template checkbox without optionally emitting stateChanged."""
        checkbox = getattr(self, "_show_template_cb", None)
        if checkbox is None:
            return

        if emit_signal or not hasattr(checkbox, "blockSignals"):
            checkbox.setChecked(checked)
            return

        previous = checkbox.blockSignals(True)
        try:
            checkbox.setChecked(checked)
        finally:
            checkbox.blockSignals(previous)

    def _set_template_checkbox_enabled(self, enabled: bool) -> None:
        """Enable or disable the template checkbox when available."""
        checkbox = getattr(self, "_show_template_cb", None)
        if checkbox is not None and hasattr(checkbox, "setEnabled"):
            checkbox.setEnabled(enabled)

    def _reference_action_checkbox(self, action: str):
        """Return the checkbox associated with a pending reference action."""
        attr_name = {
            _REFERENCE_ACTION_TEMPLATE: "_show_template_cb",
            _REFERENCE_ACTION_OUTLINE: "_show_outline_cb",
            _REFERENCE_ACTION_MESHES: "_show_region_meshes_cb",
            _REFERENCE_ACTION_SEGMENTATION: "_show_region_seg_cb",
        }.get(action)
        if attr_name is None:
            return None
        return getattr(self, attr_name, None)

    def _set_reference_action_checked(
        self,
        action: str,
        checked: bool,
        *,
        emit_signal: bool,
    ) -> None:
        """Set a reference action checkbox without optionally emitting signals."""
        checkbox = self._reference_action_checkbox(action)
        if checkbox is None or not hasattr(checkbox, "setChecked"):
            return

        if emit_signal or not hasattr(checkbox, "blockSignals"):
            checkbox.setChecked(checked)
            return

        previous = checkbox.blockSignals(True)
        try:
            checkbox.setChecked(checked)
        finally:
            checkbox.blockSignals(previous)

    def _complete_pending_reference_action(self, action: str) -> None:
        """Run the reference action that requested an atlas load."""
        checkbox = self._reference_action_checkbox(action)
        is_checked = getattr(checkbox, "isChecked", None)
        if callable(is_checked) and not is_checked():
            return

        if action == _REFERENCE_ACTION_TEMPLATE:
            self._toggle_template(True)
        elif action == _REFERENCE_ACTION_OUTLINE:
            self._toggle_outline(True)
        elif action == _REFERENCE_ACTION_MESHES:
            self._sync_active_region_meshes()
        elif action == _REFERENCE_ACTION_SEGMENTATION:
            self._sync_active_region_segmentation()

    def _start_cached_template_autoload(self) -> None:
        """Auto-load and show the reference template only when the atlas is cached."""
        if self._atlas is not None or self._cached_atlas_autoload_running():
            return

        from ..workers import CachedAtlasLoadWorker, cached_brainglobe_atlas_dir

        atlas_name = self._atlas_combo.currentText()
        with startup_timing(
            logger,
            "cached_template_autoload",
            atlas=atlas_name,
        ) as timing:
            atlas_dir = cached_brainglobe_atlas_dir(atlas_name)
            timing.set(cached=atlas_dir is not None, atlas_dir=atlas_dir)
            if atlas_dir is None:
                self._set_template_checkbox_checked(False, emit_signal=False)
                return

            self._show_template_after_cached_atlas_load = True
            self._set_template_checkbox_checked(True, emit_signal=False)
            self._set_template_checkbox_enabled(False)
            self._atlas_status_label.setText(
                f"Atlas: Loading cached {atlas_name} in background..."
            )

            thread = QThread()
            worker = CachedAtlasLoadWorker(atlas_name, atlas_dir)
            self._cached_atlas_thread = thread
            self._cached_atlas_worker = worker
            worker.moveToThread(thread)

            thread.started.connect(worker.run)
            worker.finished.connect(self._on_cached_template_atlas_loaded)
            worker.error.connect(self._on_cached_template_atlas_error)
            worker.finished.connect(thread.quit)
            worker.error.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(
                lambda: self._cleanup_cached_atlas_thread(thread, worker)
            )
            thread.start()

    def _on_cached_template_atlas_loaded(self, atlas) -> None:
        """Apply a cached atlas loaded in the background and show the template."""
        atlas_name = str(
            getattr(atlas, "atlas_name", None) or self._atlas_combo.currentText()
        )
        try:
            with startup_timing(
                logger,
                "cached_template_autoload_apply",
                atlas=atlas_name,
            ):
                if self._atlas is None:
                    self._apply_loaded_atlas(atlas, atlas_name)
                self._set_template_checkbox_enabled(True)
                if (
                    self._show_template_after_cached_atlas_load
                    or self._show_template_cb.isChecked()
                ):
                    self._toggle_template(True)
                    self._set_template_checkbox_checked(
                        True,
                        emit_signal=False,
                    )
        except Exception as e:
            logger.error("Failed to show cached atlas template: %s", e)
            self._atlas_status_label.setText(f"Atlas: Template error - {e}")
            self._set_template_checkbox_checked(False, emit_signal=False)
            self._set_template_checkbox_enabled(True)
        finally:
            self._show_template_after_cached_atlas_load = False

    def _on_cached_template_atlas_error(self, error_msg: str) -> None:
        """Handle cached atlas auto-load failure."""
        self._show_template_after_cached_atlas_load = False
        self._set_template_checkbox_checked(False, emit_signal=False)
        self._set_template_checkbox_enabled(True)
        self._atlas_status_label.setText(
            f"Atlas: Cached auto-load failed - {error_msg}"
        )
        logger.error("Cached atlas auto-load failed: %s", error_msg)

    def _cleanup_cached_atlas_thread(
        self,
        thread: QThread,
        worker: object,
    ) -> None:
        """Release cached atlas worker objects after the thread stops."""
        if self._cached_atlas_thread is thread:
            self._cached_atlas_thread = None
        if self._cached_atlas_worker is worker:
            self._cached_atlas_worker = None

    def _cleanup_atlas_load_thread(
        self,
        thread: QThread,
        worker: object,
    ) -> None:
        """Release explicit atlas load worker objects after the thread stops."""
        if self._atlas_load_thread is thread:
            self._atlas_load_thread = None
        if self._atlas_load_worker is worker:
            self._atlas_load_worker = None

    def _open_point_parquet(self) -> None:
        """Open file dialog and preview a standardized point Parquet."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Point Parquet File",
            "",
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if not filepath:
            return

        self._load_point_parquet_file(filepath)

    def _append_point_file(self) -> None:
        """Prompt for a point CSV or point Parquet and save a combined Parquet."""

        input_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Point File to Append",
            "",
            "Point Files (*.csv *.parquet);;CSV Files (*.csv);;Parquet Files (*.parquet);;All Files (*)",
        )
        if not input_path:
            return

        input_suffix = Path(input_path).suffix.lower()
        mapping_path: str | None = None
        if input_suffix == ".csv":
            try:
                _standardized, source_description = load_and_standardize_point_csv(
                    input_path
                )
            except PointImportError:
                mapping_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Mapping JSON",
                    "",
                    "JSON Files (*.json);;All Files (*)",
                )
                if not mapping_path:
                    return
            else:
                self._point_import_status_label.setText(
                    f"Detected {source_description} in {Path(input_path).name}; "
                    "no mapping JSON needed."
                )
        elif input_suffix == ".parquet":
            self._point_import_status_label.setText(
                f"Selected point Parquet {Path(input_path).name}; "
                "it must exactly match the target schema."
            )
        else:
            self._point_import_status_label.setText(
                "Select a point CSV or point Parquet file to append."
            )
            return

        parquet_path = self._select_point_parquet_source_for_append()
        if parquet_path is None:
            return

        if (
            input_suffix == ".parquet"
            and Path(input_path).resolve() == Path(parquet_path).resolve()
        ):
            self._point_import_status_label.setText(
                "Choose a different point Parquet to append than the target file."
            )
            return

        default_output_name = (
            f"{Path(parquet_path).stem}_with_{Path(input_path).stem}.parquet"
        )
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save New Point Parquet File",
            str(Path(parquet_path).with_name(default_output_name)),
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if not output_path:
            return

        if Path(output_path).resolve() == Path(parquet_path).resolve():
            self._point_import_status_label.setText(
                "Choose a new output Parquet path instead of overwriting the source file."
            )
            return

        self._start_point_file_append(
            input_path,
            mapping_path,
            parquet_path,
            output_path,
        )

    def _select_point_parquet_source_for_append(self) -> str | None:
        """Return the loaded point Parquet when available, otherwise prompt for one."""

        if self._point_parquet_path:
            loaded_path = Path(self._point_parquet_path)
            if loaded_path.exists():
                self._point_import_status_label.setText(
                    f"Using loaded point Parquet {loaded_path.name} as the append source. "
                    "Choose where to save the combined file."
                )
                return str(loaded_path)

            self._point_import_status_label.setText(
                f"Loaded point Parquet {loaded_path.name} is no longer available. "
                "Select a source Parquet file."
            )

        parquet_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Existing Point Parquet File",
            "",
            "Parquet Files (*.parquet);;All Files (*)",
        )
        return parquet_path or None

    def _convert_point_csv_from_directory(self) -> None:
        """Create a point Parquet from all top-level CSV files in a directory."""

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory of Point CSV Files",
        )
        if not directory:
            return

        csv_paths = sorted(
            str(path) for path in Path(directory).glob("*.csv") if path.is_file()
        )
        if not csv_paths:
            self._point_import_status_label.setText("No CSV files found in directory.")
            return

        default_name = f"{Path(directory).name}_points.parquet"
        self._prompt_point_csv_output_and_convert(csv_paths, default_name)

    def _convert_point_csv_from_files(self) -> None:
        """Create a point Parquet from one or more selected CSV files."""

        csv_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Point CSV Files",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not csv_paths:
            return

        default_name = (
            f"{Path(csv_paths[0]).stem}.parquet"
            if len(csv_paths) == 1
            else "points.parquet"
        )
        self._prompt_point_csv_output_and_convert(csv_paths, default_name)

    def _prompt_point_csv_output_and_convert(
        self,
        csv_paths: list[str],
        default_name: str,
    ) -> None:
        """Ask for mapping if needed, then ask for output path and start conversion."""

        mapping_path = self._select_point_csv_mapping_if_needed(csv_paths)
        if mapping_path is False:
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Point Parquet File",
            default_name,
            "Parquet Files (*.parquet)",
        )
        if not output_path:
            return

        self._start_point_csv_conversion(
            csv_paths,
            output_path,
            mapping_path if isinstance(mapping_path, str) else None,
        )

    def _select_point_csv_mapping_if_needed(
        self,
        csv_paths: list[str],
    ) -> str | bool | None:
        """Return a mapping path if autodetect fails for any selected CSV."""

        for csv_path in csv_paths:
            try:
                _standardized, _source = load_and_standardize_point_csv(csv_path)
            except PointImportError:
                mapping_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Mapping JSON",
                    "",
                    "JSON Files (*.json);;All Files (*)",
                )
                return mapping_path if mapping_path else False

        if len(csv_paths) == 1:
            self._point_import_status_label.setText(
                f"Detected point CSV headers in {Path(csv_paths[0]).name}; "
                "no mapping JSON needed."
            )
        else:
            self._point_import_status_label.setText(
                "Detected point CSV headers automatically; no mapping JSON needed."
            )
        return None

    def _point_preview_key_from_row(self, row: int) -> tuple[str, str] | None:
        """Return the selected label/origin key for a preview table row."""

        item = self._point_preview_table.item(row, _POINT_PREVIEW_LABEL_COLUMN)
        if item is None:
            return None

        key = item.data(Qt.UserRole)
        if isinstance(key, tuple) and len(key) == 2:
            return (str(key[0]), str(key[1]))

        label = item.text().strip()
        origin_item = self._point_preview_table.item(row, _POINT_PREVIEW_ORIGIN_COLUMN)
        origin_csv = origin_item.text().strip() if origin_item is not None else ""
        if not label:
            return None
        return (label, origin_csv or POINT_PARQUET_ORIGIN_NOT_RECORDED)

    def _selected_point_preview_keys(self) -> list[tuple[str, str]]:
        """Return the selected label/origin pairs from the preview table."""

        rows = sorted(
            {index.row() for index in self._point_preview_table.selectedIndexes()}
        )
        selected: list[tuple[str, str]] = []
        for row in rows:
            key = self._point_preview_key_from_row(row)
            if key is not None:
                selected.append(key)
        return selected

    def _update_point_import_controls(self) -> None:
        """Enable or disable point import actions based on current selection."""

        operation_running = self._point_operation_running()
        ready = (
            bool(self._point_parquet_path)
            and bool(self._selected_point_preview_keys())
            and self._atlas is not None
            and not operation_running
        )
        self._import_selected_point_heatmaps_btn.setEnabled(ready)
        self._open_point_parquet_btn.setEnabled(not operation_running)
        self._append_point_file_btn.setEnabled(not operation_running)
        self._create_point_from_directory_btn.setEnabled(not operation_running)
        self._create_point_from_files_btn.setEnabled(not operation_running)

    def _point_operation_running(self) -> bool:
        """Return whether a point conversion or append worker is currently active."""

        return bool(
            (
                self._point_append_thread is not None
                and self._point_append_thread.isRunning()
            )
            or (
                self._point_convert_thread is not None
                and self._point_convert_thread.isRunning()
            )
        )

    def _populate_point_parquet_preview(self, summary_df) -> None:
        """Populate the point parquet preview table from a grouped summary."""

        self._point_preview_counts.clear()
        was_blocked = self._point_preview_table.blockSignals(True)
        try:
            self._point_preview_table.clearContents()
            self._point_preview_table.setRowCount(len(summary_df))
            for row_index, row in enumerate(summary_df.itertuples(index=False)):
                label = str(row.label)
                origin_csv = str(row.origin_csv)
                point_count = int(row.point_count)
                key = (label, origin_csv)
                self._point_preview_counts[key] = point_count

                label_item = QTableWidgetItem(label)
                label_item.setData(Qt.UserRole, key)
                origin_item = QTableWidgetItem(origin_csv)
                count_item = QTableWidgetItem(f"{point_count:,}")
                count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                count_item.setData(Qt.UserRole, point_count)

                self._point_preview_table.setItem(
                    row_index,
                    _POINT_PREVIEW_LABEL_COLUMN,
                    label_item,
                )
                self._point_preview_table.setItem(
                    row_index,
                    _POINT_PREVIEW_ORIGIN_COLUMN,
                    origin_item,
                )
                self._point_preview_table.setItem(
                    row_index,
                    _POINT_PREVIEW_COUNT_COLUMN,
                    count_item,
                )
        finally:
            self._point_preview_table.blockSignals(was_blocked)

        self._point_preview_table.clearSelection()
        self._update_point_import_controls()

    def _load_point_parquet_file(
        self,
        filepath: str,
        *,
        success_message: str | None = None,
    ) -> None:
        """Load standardized point Parquet metadata into the preview table."""

        try:
            summary_df = summarize_standard_point_parquet_groups(filepath)
        except PointImportError as e:
            logger.error(f"Failed to load point Parquet: {e}")
            self._point_import_status_label.setText(f"Error: {e}")
            return

        self._point_parquet_path = filepath
        self._point_parquet_has_origin_csv = bool(
            summary_df.attrs.get("has_origin_csv", False)
        )
        self._point_file_label.setText(Path(filepath).name)

        if summary_df.empty:
            self._point_preview_counts.clear()
            self._point_preview_table.clearContents()
            self._point_preview_table.setRowCount(0)
            self._update_point_import_controls()
            self._point_import_status_label.setText("No points found in file.")
            return

        self._populate_point_parquet_preview(summary_df)
        total_points = int(summary_df["point_count"].sum())
        if success_message is None:
            action_message = "Select rows and click Import Selected Heatmaps."
            if self._atlas is None:
                action_message = "Load an atlas to enable Import Selected Heatmaps."
            success_message = (
                f"Loaded {total_points:,} point(s) across {len(summary_df)} "
                f"label/origin row(s). {action_message}"
            )
        self._point_import_status_label.setText(success_message)
        logger.info(
            "Loaded point Parquet preview %s with %d selectable row(s) and %d points",
            filepath,
            len(summary_df),
            total_points,
        )

    def _start_point_file_append(
        self,
        input_path: str,
        mapping_path: str | None,
        parquet_path: str,
        output_path: str,
    ) -> None:
        """Launch the background point-file append worker."""

        from ..workers import AppendPointFileWorker

        self._point_append_progress.setVisible(True)
        self._point_append_progress.setRange(0, 0)
        self._point_import_status_label.setText(
            f"Saving {Path(output_path).name} from {Path(parquet_path).name} + {Path(input_path).name}..."
        )

        thread = QThread()
        worker = AppendPointFileWorker(
            input_path,
            mapping_path,
            parquet_path,
            output_path,
        )
        self._point_append_thread = thread
        self._point_append_worker = worker
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_point_append_progress)
        worker.finished.connect(self._on_point_append_finished)
        worker.error.connect(self._on_point_append_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda: self._cleanup_point_append_thread(thread, worker)
        )

        self._update_point_import_controls()
        thread.start()

    def _start_point_csv_conversion(
        self,
        csv_paths: list[str],
        output_path: str,
        mapping_path: str | None,
    ) -> None:
        """Launch the background point-CSV conversion worker."""

        from ..workers import ConvertPointCSVWorker

        self._point_append_progress.setVisible(True)
        self._point_append_progress.setRange(0, max(1, len(csv_paths)))
        self._point_append_progress.setValue(0)
        self._point_import_status_label.setText(
            f"Creating {Path(output_path).name} from {len(csv_paths)} point CSV file(s)..."
        )

        thread = QThread()
        worker = ConvertPointCSVWorker(
            csv_paths,
            output_path,
            mapping_path,
        )
        self._point_convert_thread = thread
        self._point_convert_worker = worker
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_point_convert_progress)
        worker.finished.connect(self._on_point_convert_finished)
        worker.error.connect(self._on_point_convert_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda: self._cleanup_point_convert_thread(thread, worker)
        )

        self._update_point_import_controls()
        thread.start()

    def _on_point_convert_progress(
        self, message: str, current: int, total: int
    ) -> None:
        """Handle point-CSV conversion progress updates."""

        self._point_append_progress.setRange(0, max(1, total))
        self._point_append_progress.setValue(current)
        self._point_import_status_label.setText(message)

    def _on_point_convert_finished(self, parquet_path: str, summary: object) -> None:
        """Handle point-CSV conversion completion."""

        self._point_append_progress.setVisible(False)
        self._point_append_progress.setRange(0, 1)
        self._point_append_progress.setValue(0)
        message = (
            f"Created {Path(parquet_path).name} from {summary.processed_files} CSV "
            f"file(s) with {summary.rows_written:,} point(s)."
        )
        self._load_point_parquet_file(parquet_path, success_message=message)
        self._update_point_import_controls()
        logger.info(
            "Created point parquet %s from %d CSV file(s) with %d point(s)",
            parquet_path,
            summary.processed_files,
            summary.rows_written,
        )

    def _on_point_convert_error(self, error_msg: str) -> None:
        """Handle point-CSV conversion failure."""

        self._point_append_progress.setVisible(False)
        self._point_append_progress.setRange(0, 1)
        self._point_append_progress.setValue(0)
        self._point_import_status_label.setText(f"Error: {error_msg}")
        self._update_point_import_controls()
        logger.error(f"Point CSV conversion failed: {error_msg}")

    def _on_point_append_progress(
        self, message: str, _current: int, _total: int
    ) -> None:
        """Handle point-file append progress updates."""

        self._point_import_status_label.setText(message)

    def _on_point_append_finished(self, parquet_path: str, summary: object) -> None:
        """Handle point-file append completion."""

        self._point_append_progress.setVisible(False)
        self._point_append_progress.setRange(0, 1)
        self._point_append_progress.setValue(0)
        message = (
            f"Saved {Path(parquet_path).name} with {summary.appended_rows:,} "
            f"added point(s) ({summary.total_rows:,} total)."
        )
        self._load_point_parquet_file(parquet_path, success_message=message)
        self._update_point_import_controls()
        logger.info(
            "Saved point parquet %s with %d added point(s) (%d total)",
            parquet_path,
            summary.appended_rows,
            summary.total_rows,
        )

    def _on_point_append_error(self, error_msg: str) -> None:
        """Handle point-file append failure."""

        self._point_append_progress.setVisible(False)
        self._point_append_progress.setRange(0, 1)
        self._point_append_progress.setValue(0)
        self._point_import_status_label.setText(f"Error: {error_msg}")
        self._update_point_import_controls()
        logger.error(f"Point CSV append failed: {error_msg}")

    def _cleanup_point_convert_thread(self, thread: QThread, worker: object) -> None:
        """Release point conversion worker objects after the thread stops."""

        if self._point_convert_thread is thread:
            self._point_convert_thread = None
        if self._point_convert_worker is worker:
            self._point_convert_worker = None
        self._update_point_import_controls()

    def _cleanup_point_append_thread(self, thread: QThread, worker: object) -> None:
        """Release point append worker objects after the thread stops."""

        if self._point_append_thread is thread:
            self._point_append_thread = None
        if self._point_append_worker is worker:
            self._point_append_worker = None
        self._update_point_import_controls()

    def _import_selected_point_heatmaps(self) -> None:
        """Create heatmap layers for the selected preview rows."""
        from napari.utils.colormaps import Colormap

        if self._atlas is None:
            message = "Load an atlas before importing point Parquet."
            self._point_import_status_label.setText(message)
            show_warning(message)
            return

        if not self._point_parquet_path:
            self._point_import_status_label.setText(
                "Load a point Parquet before importing heatmaps."
            )
            return

        selected_keys = self._selected_point_preview_keys()
        if not selected_keys:
            self._point_import_status_label.setText(
                "Select at least one label/origin row."
            )
            return

        try:
            points_df = load_standard_point_parquet_selection(
                self._point_parquet_path,
                selected_keys,
            )
        except PointImportError as e:
            logger.error(f"Failed to load selected point Parquet rows: {e}")
            self._point_import_status_label.setText(f"Error: {e}")
            return

        if points_df.empty:
            self._point_import_status_label.setText(
                "No points matched the selected rows."
            )
            return

        validation_summary = validate_point_metadata_against_atlas(
            points_df,
            self._atlas,
        )
        if validation_summary.has_mismatches:
            show_warning(format_atlas_validation_summary(validation_summary))

        opacity = self._opacity_slider.value() / 100.0
        group_columns = (
            ("label", "origin_csv")
            if self._point_parquet_has_origin_csv
            else ("label",)
        )
        grouped_heatmaps = build_grouped_point_heatmap_volumes(
            points_df,
            self._atlas,
            group_columns,
        )
        columns = list(points_df.columns)
        created_layers = 0

        for color_idx, (label, origin_csv) in enumerate(selected_keys):
            group_key = (
                (label, origin_csv) if self._point_parquet_has_origin_csv else (label,)
            )
            volume = grouped_heatmaps.get(group_key)
            if volume is None:
                continue

            layer_name = _point_heatmap_layer_name(
                label,
                origin_csv,
                include_origin=self._point_parquet_has_origin_csv,
            )
            for layer in list(self.viewer.layers):
                if layer.name == layer_name:
                    self.viewer.layers.remove(layer)

            nonzero_voxels = int((volume > 0).sum())
            rgba = _point_heatmap_color(color_idx)
            colormap = Colormap(
                colors=[[0.0, 0.0, 0.0, 0.0], list(rgba)],
                name=f"point_heatmap_{color_idx}",
            )
            self.viewer.add_image(
                volume,
                name=layer_name,
                blending="additive",
                rendering="mip",
                colormap=colormap,
                opacity=opacity,
                metadata={
                    "source_path": self._point_parquet_path,
                    "label": label,
                    "origin_csv": origin_csv,
                    "point_count": self._point_preview_counts.get(
                        (label, origin_csv), 0
                    ),
                    "nonzero_voxels": nonzero_voxels,
                    "columns": columns,
                    "color": rgba,
                    "heatmap_source": True,
                    "heatmap_native_grid": True,
                    "atlas_name": getattr(self._atlas, "atlas_name", None),
                    "heatmap_kind": "point_import",
                },
            )
            created_layers += 1

        message = (
            f"Imported {len(points_df):,} selected point(s) into {created_layers} "
            "heatmap layer(s)."
        )
        if validation_summary.has_mismatches:
            message += (
                f" Atlas validation found "
                f"{validation_summary.total_mismatched_rows} mismatched row(s)."
            )
        self._point_import_status_label.setText(message)
        logger.info(
            "Imported %d selected point heatmaps from %s covering %d points",
            created_layers,
            self._point_parquet_path,
            len(points_df),
        )
        self._refresh_heatmap_layer_list()
        self._refresh_histogram_layer_list()
        self._refresh_mask_layer_options()

    def _connect_layer_events(self) -> None:
        """Refresh tool and mask selectors when viewer layers change."""
        layer_events = getattr(getattr(self.viewer, "layers", None), "events", None)
        if layer_events is not None:
            for event_name in ("inserted", "removed", "reordered"):
                signal = getattr(layer_events, event_name, None)
                if signal is not None:
                    signal.connect(self._on_viewer_layers_changed)
        self._sync_layer_name_event_connections()

    def _sync_layer_name_event_connections(self) -> None:
        """Track layer-name events so layer renames refresh layer-dependent UI."""
        connections = getattr(self, "_layer_name_event_connections", None)
        if connections is None:
            connections = {}
            self._layer_name_event_connections = connections

        layers = self._iter_viewer_layers()
        active_layer_ids = {id(layer) for layer in layers}
        self._disconnect_stale_layer_name_event_connections(active_layer_ids)

        for layer in layers:
            layer_id = id(layer)
            existing = connections.get(layer_id)
            if existing is not None and existing[0] is layer:
                continue
            if existing is not None:
                self._disconnect_layer_name_event_connection(layer_id, existing)

            signal = getattr(getattr(layer, "events", None), "name", None)
            connector = getattr(signal, "connect", None)
            if not callable(connector):
                continue
            callback = self._on_viewer_layer_name_changed
            try:
                connector(callback)
            except Exception:
                logger.debug(
                    "Could not connect layer name event for %s",
                    getattr(layer, "name", "<unnamed>"),
                    exc_info=True,
                )
                continue
            connections[layer_id] = (layer, signal, callback)

    def _disconnect_stale_layer_name_event_connections(
        self,
        active_layer_ids: set[int],
    ) -> None:
        """Disconnect tracked name events for layers no longer in the viewer."""
        connections = getattr(self, "_layer_name_event_connections", None)
        if not connections:
            return
        for layer_id, connection in list(connections.items()):
            if layer_id not in active_layer_ids:
                self._disconnect_layer_name_event_connection(layer_id, connection)

    def _disconnect_layer_name_event_connection(
        self,
        layer_id: int,
        connection: tuple[object, object, object],
    ) -> None:
        """Disconnect one tracked layer-name event connection."""
        _layer, signal, callback = connection
        disconnect = getattr(signal, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect(callback)
            except Exception:
                logger.debug("Could not disconnect layer name event", exc_info=True)
        connections = getattr(self, "_layer_name_event_connections", None)
        if connections is not None:
            connections.pop(layer_id, None)

    def _on_viewer_layer_name_changed(self, event=None) -> None:
        """Refresh layer-dependent UI after a layer is renamed."""
        self._on_viewer_layers_changed(event)

    def _on_viewer_layers_changed(self, _event=None) -> None:
        """Refresh UI that depends on viewer layers."""
        sync_layer_names = getattr(self, "_sync_layer_name_event_connections", None)
        if callable(sync_layer_names):
            sync_layer_names()
        self._refresh_heatmap_layer_list()
        self._refresh_histogram_layer_list()
        self._refresh_mask_layer_options()
        self._update_tools_controls()
        self._update_histogram_controls()
        sync_heatmaps = getattr(self, "_sync_neuron_table_heatmap_membership", None)
        if callable(sync_heatmaps):
            sync_heatmaps()
        refresh_manual_heatmaps = getattr(self, "_refresh_manual_heatmap_combo", None)
        if callable(refresh_manual_heatmaps):
            refresh_manual_heatmaps()
        comparison_window = getattr(self, "_comparison_window", None)
        if comparison_window is not None and comparison_window.isVisible():
            comparison_window.refresh_sources()

    def _iter_viewer_layers(self) -> list:
        """Return current viewer layers as a list."""
        try:
            return list(self.viewer.layers)
        except Exception:
            return []

    def _manual_heatmap_layers(self) -> list:
        """Return selected-neuron heatmap layers created from the Data tab."""
        return [
            layer
            for layer in self._iter_viewer_layers()
            if _layer_metadata(layer).get("heatmap_kind") == "selected_neurons"
        ]

    @staticmethod
    def _normalise_layer_file_ids(file_ids: object) -> tuple[object, ...]:
        """Return layer file IDs as a tuple without splitting string IDs."""
        if file_ids is None:
            return ()
        if isinstance(file_ids, (str, bytes)):
            return (file_ids,)
        try:
            return tuple(file_ids)
        except TypeError:
            return (file_ids,)

    def _current_selected_neuron_heatmap_layers_by_file_id(
        self,
    ) -> dict[object, tuple[str, ...]]:
        """Return Data-tab selected-neuron heatmap layer names by file ID."""
        layer_names_by_file_id: dict[object, list[str]] = {}
        for layer in self._manual_heatmap_layers():
            layer_name = str(getattr(layer, "name", ""))
            if not layer_name:
                continue

            metadata = _layer_metadata(layer)
            for file_id in self._normalise_layer_file_ids(metadata.get("file_ids", [])):
                names = layer_names_by_file_id.setdefault(file_id, [])
                if layer_name not in names:
                    names.append(layer_name)

        return {
            file_id: tuple(layer_names)
            for file_id, layer_names in layer_names_by_file_id.items()
        }

    def _sync_neuron_table_heatmap_membership(self) -> None:
        """Sync the neuron table Heatmap column from Data-tab heatmap layers."""
        table = getattr(self, "_neuron_table", None)
        setter = getattr(table, "set_heatmap_layers_by_file_id", None)
        if not callable(setter):
            return
        setter(self._current_selected_neuron_heatmap_layers_by_file_id())

    def _manual_heatmap_combo_options(
        self,
    ) -> list[tuple[str, tuple[object, ...], str]]:
        """Return manual heatmap dropdown options as layer-name/file-ID/key tuples."""
        options: list[tuple[str, tuple[object, ...], str]] = []
        for layer in self._manual_heatmap_layers():
            layer_name = str(getattr(layer, "name", ""))
            if not layer_name:
                continue
            metadata = _layer_metadata(layer)
            file_ids = self._normalise_layer_file_ids(metadata.get("file_ids", []))
            key = str(metadata.get("manual_heatmap_id") or layer_name)
            options.append((layer_name, file_ids, key))
        return options

    def _manual_heatmap_combo_data(self) -> tuple[str, tuple[object, ...]] | None:
        """Return the currently selected manual heatmap dropdown payload."""
        combo = getattr(self, "_manual_heatmap_combo", None)
        if combo is None:
            return None
        current_data = getattr(combo, "currentData", None)
        data = current_data() if callable(current_data) else None
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], str):
            return data[0], self._normalise_layer_file_ids(data[1])
        if isinstance(data, tuple) and len(data) == 3 and isinstance(data[0], str):
            return data[0], self._normalise_layer_file_ids(data[1])
        return None

    def _manual_heatmap_combo_key(self) -> str | None:
        """Return the stable key for the selected manual heatmap, if any."""
        combo = getattr(self, "_manual_heatmap_combo", None)
        if combo is None:
            return None
        current_data = getattr(combo, "currentData", None)
        data = current_data() if callable(current_data) else None
        if (
            isinstance(data, tuple)
            and len(data) == 3
            and isinstance(data[2], str)
            and data[2]
        ):
            return data[2]
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], str):
            return data[0]
        return None

    def _refresh_manual_heatmap_combo(self) -> None:
        """Refresh the Data-tab manual heatmap selector."""
        combo = getattr(self, "_manual_heatmap_combo", None)
        if combo is None:
            return

        previous = self._manual_heatmap_combo_data()
        previous_name = previous[0] if previous is not None else None
        previous_key = self._manual_heatmap_combo_key()
        options = self._manual_heatmap_combo_options()

        signals_blocked = combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(_MANUAL_HEATMAP_ALL_LABEL, None)
            selected_index = 0
            for layer_name, file_ids, key in options:
                combo.addItem(layer_name, (layer_name, tuple(file_ids), key))
                if key == previous_key or (
                    previous_key is None and layer_name == previous_name
                ):
                    selected_index = combo.count() - 1
            combo.setCurrentIndex(selected_index)
        finally:
            combo.blockSignals(signals_blocked)

        set_enabled = getattr(combo, "setEnabled", None)
        if callable(set_enabled):
            set_enabled(bool(options))
        if previous_name is not None and selected_index == 0:
            self._on_manual_heatmap_selection_changed()

    def _on_manual_heatmap_selection_changed(self, _index: int = 0) -> None:
        """Filter table rows for the chosen manual heatmap layer."""
        self._apply_neuron_table_filters()

    def _selected_manual_heatmap_file_ids(self) -> tuple[object, ...] | None:
        """Return the currently selected manual heatmap file IDs, if any."""
        data = self._manual_heatmap_combo_data()
        return None if data is None else data[1]

    def _apply_neuron_table_filters(self) -> None:
        """Apply current Data-tab row filters to the neuron table."""
        table = getattr(self, "_neuron_table", None)
        if table is None:
            return

        selection = self._selected_cluster_filter()
        heatmap_file_ids = self._selected_manual_heatmap_file_ids()
        applier = getattr(table, "apply_filters", None)
        if callable(applier):
            applier(selection, heatmap_file_ids)
            return

        cluster_applier = getattr(table, "apply_cluster_filter", None)
        if callable(cluster_applier):
            cluster_applier(selection)

    def _current_atlas_name(self) -> str | None:
        """Return the currently loaded atlas name, if any."""
        if self._atlas is None:
            return None
        return str(getattr(self._atlas, "atlas_name", "")) or None

    def _selected_heatmap_running(self) -> bool:
        """Return whether a selected-neuron heatmap worker is active."""
        thread = getattr(self, "_selected_heatmap_thread", None)
        if thread is not None and thread.isRunning():
            return True
        if getattr(self, "_selected_heatmap_current_request", None) is not None:
            return True
        return bool(getattr(self, "_selected_heatmap_pending_requests", []))

    def _update_selected_neuron_heatmap_controls(self) -> None:
        """Enable or disable the selected-neuron heatmap action."""
        if hasattr(self, "_add_selected_heatmap_btn"):
            self._add_selected_heatmap_btn.setEnabled(
                not self._selected_heatmap_running()
            )

    def _refresh_neuron_table_summary(self) -> None:
        """Refresh the summary text shown below the neuron table actions."""
        if not hasattr(self, "_neuron_table_summary_label"):
            return
        if not hasattr(self, "_neuron_table"):
            self._neuron_table_summary_label.setText("")
            return

        summary = self._neuron_table.summary()
        counts_line = (
            f"In table: {summary.table_count:,} | "
            f"Added to scene: {summary.added_count:,} | "
            f"Visible: {summary.visible_count:,}"
        )
        if not summary.cluster_counts:
            clusters_line = "Clusters: none"
        else:
            parts = []
            for cluster_id, count in summary.cluster_counts:
                if cluster_id is None:
                    parts.append(f"Unclustered: {count:,}")
                else:
                    parts.append(f"Cluster {cluster_id}: {count:,}")
            clusters_line = "Clusters: " + ", ".join(parts)

        active_assignment = self._cluster_assignment_store.active
        if active_assignment is not None:
            clusters_line = f"{active_assignment.name} — {clusters_line}"
        self._neuron_table_summary_label.setText(f"{counts_line}\n{clusters_line}")

    def _selected_neuron_heatmap_base_name(
        self,
        file_ids: list[object] | tuple[object, ...],
    ) -> str:
        """Return the base layer name for a selected-neuron heatmap."""
        return f"{self._next_manual_heatmap_identifier()} Heatmap"

    def _unique_layer_name(self, base_name: str) -> str:
        """Return a viewer layer name that does not collide with existing names."""
        existing_names = {
            str(getattr(layer, "name", "")) for layer in self._iter_viewer_layers()
        }
        if base_name not in existing_names:
            return base_name

        suffix = 2
        while f"{base_name} ({suffix})" in existing_names:
            suffix += 1
        return f"{base_name} ({suffix})"

    @staticmethod
    def _greek_heatmap_identifier(index: int) -> str:
        """Return a deterministic Greek-word identifier for a zero-based index."""
        if index < 0:
            raise ValueError("index must be non-negative")

        alphabet = _GREEK_HEATMAP_IDENTIFIERS
        base = len(alphabet)
        length = 1
        remaining = int(index)
        block_size = base
        while remaining >= block_size:
            remaining -= block_size
            length += 1
            block_size *= base

        indices = [0] * length
        for offset in range(length - 1, -1, -1):
            indices[offset] = remaining % base
            remaining //= base
        return " ".join(alphabet[i] for i in indices)

    def _next_manual_heatmap_identifier(self) -> str:
        """Return the first Greek identifier whose layer name is unused."""
        existing_names = {
            str(getattr(layer, "name", "")) for layer in self._iter_viewer_layers()
        }
        index = 0
        while True:
            identifier = NeuronViewerWidget._greek_heatmap_identifier(index)
            if f"{identifier} Heatmap" not in existing_names:
                return identifier
            index += 1

    def _selected_neuron_heatmap_layer_name(
        self,
        file_ids: list[object] | tuple[object, ...],
    ) -> str:
        """Return the next unique layer name for a selected-neuron heatmap."""
        return f"{self._next_manual_heatmap_identifier()} Heatmap"

    def _update_mask_sigma_units_label(self) -> None:
        """Show Gaussian sigma units in voxels with atlas micron equivalence."""
        if not hasattr(self, "_mask_sigma_units_label"):
            return

        if self._atlas is None:
            self._mask_sigma_units_label.setText(
                "Sigma is measured in atlas voxels. Load an atlas to see the micron equivalent."
            )
            return

        resolution = np.asarray(self._atlas.resolution, dtype=float)
        if np.allclose(resolution, resolution[0]):
            self._mask_sigma_units_label.setText(
                f"Sigma is measured in atlas voxels. "
                f"With the current atlas, 1 voxel = {resolution[0]:g} µm."
            )
        else:
            self._mask_sigma_units_label.setText(
                "Sigma is measured in atlas voxels. "
                f"With the current atlas, 1 voxel = "
                f"{resolution[2]:g} µm (X), {resolution[1]:g} µm (Y), "
                f"{resolution[0]:g} µm (Z)."
            )

    def _heatmap_layer_eligibility(self, layer) -> tuple[bool, str]:
        """Return whether a layer is eligible as a Tools heatmap source."""
        if self._atlas is None:
            return False, "Load an atlas to select heatmaps."

        metadata = _layer_metadata(layer)
        if not metadata.get("heatmap_source"):
            return False, "Not an app heatmap layer."
        if not metadata.get("heatmap_native_grid", False):
            return False, "Heatmap is not on the native atlas voxel grid."
        if metadata.get("atlas_name") != self._current_atlas_name():
            return False, "Heatmap atlas does not match the loaded atlas."

        data = np.asarray(getattr(layer, "data", np.array([])))
        atlas_shape = tuple(np.asarray(self._atlas.annotation).shape)
        if data.ndim != 3 or tuple(data.shape) != atlas_shape:
            return False, "Heatmap shape does not match the loaded atlas."
        return True, ""

    def _generated_mask_layers(self) -> list:
        """Return generated mask layers eligible for Regions queries."""
        masks = []
        atlas_name = self._current_atlas_name()
        for layer in self._iter_viewer_layers():
            metadata = _layer_metadata(layer)
            if not metadata.get("mask_query_source"):
                continue
            if atlas_name is not None and metadata.get("atlas_name") != atlas_name:
                continue
            masks.append(layer)
        return masks

    def _eligible_heatmap_layers_with_exclusions(self) -> tuple[list, list[str]]:
        """Return eligible heatmap layers and exclusion messages."""
        eligible_layers = []
        excluded_messages: list[str] = []
        for layer in self._iter_viewer_layers():
            eligible, reason = self._heatmap_layer_eligibility(layer)
            if eligible:
                eligible_layers.append(layer)
            elif _layer_metadata(layer).get("heatmap_source"):
                excluded_messages.append(f"{layer.name}: {reason}")
        return eligible_layers, excluded_messages

    def _refresh_heatmap_layer_list(self) -> None:
        """Refresh the Tools heatmap selector list."""
        if not hasattr(self, "_heatmap_layer_list"):
            return

        previous = {item.text() for item in self._heatmap_layer_list.selectedItems()}
        self._heatmap_layer_list.clear()

        eligible_layers, excluded_messages = (
            self._eligible_heatmap_layers_with_exclusions()
        )
        eligible_names = [layer.name for layer in eligible_layers]
        for layer in eligible_layers:
            self._heatmap_layer_list.addItem(layer.name)

        for index in range(self._heatmap_layer_list.count()):
            item = self._heatmap_layer_list.item(index)
            if item.text() in previous:
                item.setSelected(True)

        if not eligible_names:
            if excluded_messages:
                self._tools_hint_label.setText(
                    "No eligible native-grid heatmaps. "
                    + " ".join(excluded_messages[:3])
                )
            else:
                self._tools_hint_label.setText(
                    "No eligible heatmap layers are available."
                )
        elif excluded_messages:
            self._tools_hint_label.setText(
                f"{len(eligible_names)} eligible heatmap layer(s). "
                + "Excluded: "
                + "; ".join(excluded_messages[:3])
            )
        else:
            self._tools_hint_label.setText(
                f"{len(eligible_names)} eligible heatmap layer(s)."
            )
        self._update_tools_controls()

    def _refresh_histogram_layer_list(self) -> None:
        """Refresh the Histogram-tab source list."""
        if not hasattr(self, "_histogram_layer_list"):
            return

        previous = {item.text() for item in self._histogram_layer_list.selectedItems()}
        self._histogram_layer_list.clear()

        eligible_layers, excluded_messages = (
            self._eligible_heatmap_layers_with_exclusions()
        )
        eligible_names = [layer.name for layer in eligible_layers]
        for layer in eligible_layers:
            self._histogram_layer_list.addItem(layer.name)

        for index in range(self._histogram_layer_list.count()):
            item = self._histogram_layer_list.item(index)
            if item.text() in previous:
                item.setSelected(True)

        if not eligible_names:
            if excluded_messages:
                self._histogram_hint_label.setText(
                    "No eligible native-grid heatmaps. "
                    + " ".join(excluded_messages[:3])
                )
            else:
                self._histogram_hint_label.setText(
                    "No eligible heatmap layers are available."
                )
        elif excluded_messages:
            self._histogram_hint_label.setText(
                f"{len(eligible_names)} eligible heatmap layer(s). "
                + "Excluded: "
                + "; ".join(excluded_messages[:3])
            )
        else:
            self._histogram_hint_label.setText(
                f"{len(eligible_names)} eligible heatmap layer(s)."
            )
        self._update_histogram_controls()
        self._update_histogram_plot()

    def _refresh_mask_layer_options(self) -> None:
        """Refresh Regions-tab mask layer options."""
        if not hasattr(self, "_mask_layer_selector"):
            return

        masks = self._generated_mask_layers()
        mask_entries = []
        for layer in masks:
            metadata = _layer_metadata(layer)
            mask_entries.append(
                {
                    "name": layer.name,
                    "sources": metadata.get("source_heatmap_layers", []),
                }
            )
        self._mask_layer_selector.set_mask_layers(mask_entries)
        self._update_mask_query_summary()

    def _current_tools_create_mode(self) -> str:
        """Return the active Tools create mode."""
        combo = getattr(self, "_tools_create_mode_combo", None)
        mode = combo.currentData() if combo is not None else None
        return "merged" if mode == "merged" else "separate"

    def _current_region_isolation_create_mode(self) -> str:
        """Return the active Tools region-isolation create mode."""
        combo = getattr(self, "_region_isolation_create_mode_combo", None)
        mode = combo.currentData() if combo is not None else None
        return "merged" if mode == "merged" else "separate"

    def _current_histogram_create_mode(self) -> str:
        """Return the active Histogram create mode."""
        combo = getattr(self, "_histogram_create_mode_combo", None)
        mode = combo.currentData() if combo is not None else None
        return "merged" if mode == "merged" else "separate"

    def _selected_region_isolation_entries(self) -> list[tuple[int, str]]:
        """Return directly selected Tools isolation regions as IDs and acronyms."""
        selector = getattr(self, "_tools_region_selector", None)
        if selector is None:
            return []

        entries: list[tuple[int, str]] = []
        if hasattr(selector, "get_selected_ids"):
            selected_ids = selector.get_selected_ids(include_children=False)
            structure_map = getattr(selector, "_structure_map", {})
            for struct_id in selected_ids:
                struct = structure_map.get(int(struct_id), {})
                acronym = str(struct.get("acronym", "")).strip()
                if acronym:
                    entries.append((int(struct_id), acronym))
            if entries:
                return entries

        if hasattr(selector, "get_single_selected_region"):
            selected = selector.get_single_selected_region()
            if selected is not None:
                struct_id, acronym = selected
                return [(int(struct_id), str(acronym))]
        return []

    def _selected_region_isolation_region_ids(self) -> list[int]:
        """Return effective atlas annotation IDs for Tools region isolation."""
        selector = getattr(self, "_tools_region_selector", None)
        if selector is None or not hasattr(selector, "get_selected_ids"):
            return []

        include_children = True
        if hasattr(selector, "include_children_enabled"):
            include_children = bool(selector.include_children_enabled())
        return [
            int(region_id)
            for region_id in selector.get_selected_ids(
                include_children=include_children
            )
        ]

    def _region_isolation_label(self, acronyms: list[str]) -> str:
        """Return a compact region label for isolated heatmap layer names."""
        if not acronyms:
            return "selected regions"
        if len(acronyms) <= 2:
            return ", ".join(acronyms)
        return f"{', '.join(acronyms[:2])} +{len(acronyms) - 2} more"

    def _selected_layer_names_from_widget(self, widget: QListWidget | None) -> set[str]:
        """Return selected layer names from a list widget."""
        if widget is None:
            return set()
        return {item.text() for item in widget.selectedItems()}

    def _selected_layers_from_widget(self, widget: QListWidget | None) -> list:
        """Return selected eligible heatmap layers for a list widget."""
        selected_names = self._selected_layer_names_from_widget(widget)
        if not selected_names:
            return []
        layers = []
        for layer in self._iter_viewer_layers():
            if (
                layer.name in selected_names
                and self._heatmap_layer_eligibility(layer)[0]
            ):
                layers.append(layer)
        return layers

    def _selected_heatmap_layers(self) -> list:
        """Return selected eligible heatmap layers from the Tools tab."""
        return self._selected_layers_from_widget(
            getattr(self, "_heatmap_layer_list", None)
        )

    def _selected_histogram_layers(self) -> list:
        """Return selected eligible heatmap layers from the Histogram tab."""
        return self._selected_layers_from_widget(
            getattr(self, "_histogram_layer_list", None)
        )

    def _layer_contrast_limits(self, layer) -> tuple[float, float] | None:
        """Return a layer's contrast limits if available."""
        contrast_limits = getattr(layer, "contrast_limits", None)
        if contrast_limits is None:
            return None
        try:
            limits_array = np.asarray(contrast_limits, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if limits_array.size < 2:
            return None
        lower, upper = sorted((float(limits_array[0]), float(limits_array[1])))
        return lower, upper

    def _histogram_plot_color(self, layer) -> tuple[float, float, float, float] | None:
        """Return a plotting color derived from layer metadata."""
        color = _layer_metadata(layer).get("color")
        if color is None:
            return None
        rgba = np.asarray(color, dtype=float).reshape(-1)
        if rgba.size == 3:
            rgba = np.append(rgba, 1.0)
        if rgba.size < 4:
            return None
        rgba = np.clip(rgba[:4], 0.0, 1.0)
        return tuple(float(value) for value in rgba)

    def _histogram_plot_pen(self, layer, series_index: int):
        """Return a pyqtgraph pen for one histogram series."""
        pg = getattr(self, "_histogram_pg", None)
        if pg is None:
            return None

        rgba = self._histogram_plot_color(layer)
        if rgba is None:
            return pg.mkPen(color=pg.intColor(series_index), width=2)

        rgba255 = tuple(int(round(value * 255.0)) for value in rgba)
        return pg.mkPen(color=rgba255, width=2)

    def _reset_histogram_plot_items(self) -> None:
        """Clear histogram curves and annotations while keeping the widget."""
        plot_item = getattr(self, "_histogram_plot_item", None)
        if plot_item is None:
            return

        legend = getattr(self, "_histogram_plot_legend", None)
        if legend is not None:
            scene = legend.scene()
            if scene is not None:
                scene.removeItem(legend)
            if getattr(plot_item, "legend", None) is legend:
                plot_item.legend = None
            self._histogram_plot_legend = None

        plot_item.clear()
        plot_item.setTitle("Intensity Histogram")
        plot_item.setLabel("bottom", "Intensity")
        plot_item.setLabel("left", "Voxel count")
        plot_item.showGrid(x=True, y=True, alpha=0.15)

        self._histogram_curve_items = []
        self._histogram_message_item = None
        self._histogram_plot_legend = plot_item.addLegend(offset=(12, 12))
        plot_item.addItem(self._histogram_lower_line, ignoreBounds=True)
        plot_item.addItem(self._histogram_upper_line, ignoreBounds=True)

    def _set_histogram_message(
        self,
        message: str,
        *,
        x_center: float = 0.5,
        y_center: float = 0.5,
    ) -> None:
        """Display a centered placeholder message in the histogram plot."""
        plot_item = getattr(self, "_histogram_plot_item", None)
        pg = getattr(self, "_histogram_pg", None)
        if plot_item is None or pg is None:
            return

        message_item = pg.TextItem(text=message, anchor=(0.5, 0.5))
        message_item.setPos(float(x_center), float(y_center))
        plot_item.addItem(message_item)
        self._histogram_message_item = message_item

    def _refresh_histogram_threshold_lines(self) -> None:
        """Update threshold guide positions without rebuilding the chart."""
        lower_line = getattr(self, "_histogram_lower_line", None)
        upper_line = getattr(self, "_histogram_upper_line", None)
        if lower_line is None or upper_line is None:
            return

        has_selection = bool(self._selected_histogram_layers())
        lower_value = (
            float(self._mask_lower_threshold_spin.value())
            if hasattr(self, "_mask_lower_threshold_spin")
            else 0.0
        )
        show_upper = bool(
            has_selection
            and hasattr(self, "_mask_use_upper_bound_cb")
            and self._mask_use_upper_bound_cb.isChecked()
        )

        self._histogram_line_sync_active = True
        try:
            lower_line.setPos(lower_value)
            lower_line.setVisible(has_selection)
            lower_line.setMovable(has_selection)

            if show_upper and hasattr(self, "_mask_upper_threshold_spin"):
                upper_line.setPos(float(self._mask_upper_threshold_spin.value()))
            upper_line.setVisible(show_upper)
            upper_line.setMovable(show_upper)
        finally:
            self._histogram_line_sync_active = False

    def _sync_mask_bounds_from_lines(
        self,
        *,
        lower: float,
        upper: float | None,
    ) -> None:
        """Update mask bound widgets from dragged histogram lines."""
        self._mask_lower_threshold_spin.blockSignals(True)
        self._mask_upper_threshold_spin.blockSignals(True)
        try:
            self._mask_lower_threshold_spin.setValue(float(lower))
            if upper is not None:
                self._mask_upper_threshold_spin.setValue(float(upper))
        finally:
            self._mask_upper_threshold_spin.blockSignals(False)
            self._mask_lower_threshold_spin.blockSignals(False)

    def _on_histogram_bound_line_moved(self, which: str) -> None:
        """Sync dragged histogram guide lines back into the bound controls."""
        if self._histogram_line_sync_active:
            return

        selected_layers = self._selected_histogram_layers()
        if not selected_layers:
            return

        lower_line = getattr(self, "_histogram_lower_line", None)
        upper_line = getattr(self, "_histogram_upper_line", None)
        if lower_line is None or upper_line is None:
            return

        lower = float(self._mask_lower_threshold_spin.value())
        use_upper = bool(self._mask_use_upper_bound_cb.isChecked())
        upper = float(self._mask_upper_threshold_spin.value()) if use_upper else None

        if which == "lower":
            lower = float(lower_line.value())
            if upper is not None and lower > upper:
                upper = lower
        else:
            if upper is None:
                return
            upper = float(upper_line.value())
            if upper < lower:
                lower = upper

        self._sync_mask_bounds_from_lines(lower=lower, upper=upper)
        self._refresh_histogram_threshold_lines()

    def _on_histogram_bound_line_move_finished(self, which: str) -> None:
        """Mark dragged histogram bounds as manual after the user releases them."""
        if self._histogram_line_sync_active:
            return
        if which == "upper" and not self._mask_use_upper_bound_cb.isChecked():
            return
        self._mask_bounds_source = "manual"

    def _update_tools_controls(self, *_args) -> None:
        """Enable or disable Tools actions based on current selection."""
        if not hasattr(self, "_heatmap_layer_list"):
            return

        selected_layers = self._selected_heatmap_layers()
        ready = self._atlas is not None and bool(selected_layers)
        if hasattr(self, "_enhance_fine_projections_btn"):
            self._enhance_fine_projections_btn.setEnabled(ready)
        if hasattr(self, "_reset_heatmap_gamma_btn"):
            self._reset_heatmap_gamma_btn.setEnabled(ready)
        if hasattr(self, "_create_blur_btn"):
            self._create_blur_btn.setEnabled(ready)
        if hasattr(self, "_create_region_isolated_heatmap_btn"):
            has_regions = bool(self._selected_region_isolation_region_ids())
            self._create_region_isolated_heatmap_btn.setEnabled(ready and has_regions)

    def _set_selected_heatmap_gamma(self, gamma: float, *, action: str) -> None:
        """Set gamma on every heatmap selected in the Tools source list."""
        selected_layers = self._selected_heatmap_layers()
        if not selected_layers:
            self._tools_status_label.setText(
                "Select at least one eligible heatmap layer."
            )
            return

        updated_names: list[str] = []
        failed_names: list[str] = []
        for layer in selected_layers:
            try:
                layer.gamma = float(gamma)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                failed_names.append(str(getattr(layer, "name", "<unnamed>")))
                logger.warning(
                    "Could not set gamma on heatmap layer '%s'.",
                    getattr(layer, "name", "<unnamed>"),
                    exc_info=True,
                )
            else:
                updated_names.append(str(getattr(layer, "name", "<unnamed>")))

        message = (
            f"{action} on {len(updated_names)} heatmap layer(s) "
            f"(gamma {float(gamma):.2f})."
        )
        if failed_names:
            message += " Could not update: " + ", ".join(failed_names[:3])
            if len(failed_names) > 3:
                message += f" and {len(failed_names) - 3} more"
            message += "."
        self._tools_status_label.setText(message)

    def _enhance_selected_heatmap_projections(self) -> None:
        """Brighten low-intensity projections in selected heatmap layers."""
        self._set_selected_heatmap_gamma(
            _FINE_PROJECTION_GAMMA,
            action="Enhanced fine projections",
        )

    def _reset_selected_heatmap_gamma(self) -> None:
        """Restore the default gamma on selected heatmap layers."""
        self._set_selected_heatmap_gamma(
            _DEFAULT_HEATMAP_GAMMA,
            action="Reset gamma",
        )

    def _update_histogram_controls(self) -> None:
        """Enable or disable Histogram actions based on current selection."""
        if not hasattr(self, "_histogram_layer_list"):
            return

        selected_layers = self._selected_histogram_layers()
        has_selection = bool(selected_layers)
        ready = self._atlas is not None and has_selection
        can_use_otsu = ready and (
            len(selected_layers) == 1
            or self._current_histogram_create_mode() == "merged"
        )

        contrast_reason = None
        if self._atlas is None:
            contrast_reason = "Load an atlas before syncing layer contrast."
        elif not selected_layers:
            contrast_reason = (
                "Select one eligible heatmap layer to sync contrast limits."
            )
        elif len(selected_layers) != 1:
            contrast_reason = (
                "Select exactly one eligible heatmap layer to sync contrast limits."
            )
        elif self._layer_contrast_limits(selected_layers[0]) is None:
            contrast_reason = "The selected layer does not expose contrast limits."

        if hasattr(self, "_create_mask_btn"):
            self._create_mask_btn.setEnabled(ready)
        if hasattr(self, "_mask_use_otsu_btn"):
            self._mask_use_otsu_btn.setEnabled(can_use_otsu)
            self._mask_use_otsu_btn.setToolTip(
                ""
                if can_use_otsu
                else "Select one heatmap layer, or switch Create mode to Merged layer."
            )
        if hasattr(self, "_mask_use_contrast_btn"):
            can_use_contrast = contrast_reason is None
            self._mask_use_contrast_btn.setEnabled(can_use_contrast)
            self._mask_use_contrast_btn.setToolTip(
                "" if can_use_contrast else contrast_reason
            )
        self._refresh_histogram_threshold_lines()

    def _mark_mask_bounds_manual(self, *_args) -> None:
        """Record that mask bounds were edited manually."""
        self._mask_bounds_source = "manual"
        self._refresh_histogram_threshold_lines()

    def _on_mask_upper_bound_toggled(self, checked: bool) -> None:
        """Enable or disable the upper threshold input."""
        if hasattr(self, "_mask_upper_threshold_spin"):
            self._mask_upper_threshold_spin.setEnabled(bool(checked))
        self._mask_bounds_source = "manual"
        self._refresh_histogram_threshold_lines()

    def _set_mask_bounds(
        self,
        lower: float,
        *,
        upper: float | None = None,
        enable_upper: bool,
        bounds_source: str,
    ) -> None:
        """Update Histogram mask bounds without marking them as manual edits."""
        self._mask_lower_threshold_spin.blockSignals(True)
        self._mask_upper_threshold_spin.blockSignals(True)
        self._mask_use_upper_bound_cb.blockSignals(True)

        self._mask_lower_threshold_spin.setValue(float(lower))
        if upper is not None:
            self._mask_upper_threshold_spin.setValue(float(upper))
        self._mask_use_upper_bound_cb.setChecked(bool(enable_upper))

        self._mask_use_upper_bound_cb.blockSignals(False)
        self._mask_upper_threshold_spin.blockSignals(False)
        self._mask_lower_threshold_spin.blockSignals(False)

        self._mask_upper_threshold_spin.setEnabled(bool(enable_upper))
        self._mask_bounds_source = bounds_source
        self._refresh_histogram_threshold_lines()

    def _on_histogram_layer_selection_changed(self) -> None:
        """Refresh histogram controls and plot after source selection changes."""
        self._update_histogram_controls()
        self._update_histogram_plot()

    def _update_histogram_plot(self, *_args) -> None:
        """Redraw the histogram plot for the selected layers."""
        plot_item = getattr(self, "_histogram_plot_item", None)
        if plot_item is None:
            return

        self._reset_histogram_plot_items()
        selected_layers = self._selected_histogram_layers()

        if not selected_layers:
            plot_item.setXRange(0.0, 1.0, padding=0.0)
            plot_item.setYRange(0.0, 1.0, padding=0.0)
            self._set_histogram_message("Select one or more eligible heatmap layers.")
            self._refresh_histogram_threshold_lines()
            return

        include_zero = bool(self._histogram_include_zero_cb.isChecked())
        _, series = _build_histogram_plot_series(
            [
                (layer.name, np.asarray(layer.data, dtype=np.float32))
                for layer in selected_layers
            ],
            bins=_HISTOGRAM_BIN_COUNT,
            include_zero=include_zero,
        )

        has_values = False
        for index, (layer, entry) in enumerate(zip(selected_layers, series)):
            x = np.asarray(entry["x"], dtype=np.float32)
            y = np.asarray(entry["y"], dtype=np.float32)
            if x.size == 0 or y.size == 0:
                continue
            has_values = True
            curve = plot_item.plot(
                x=x,
                y=y,
                pen=self._histogram_plot_pen(layer, index),
            )
            self._histogram_curve_items.append(curve)
            if self._histogram_plot_legend is not None:
                self._histogram_plot_legend.addItem(curve, str(entry["name"]))

        if not has_values:
            message = (
                "No finite values remain after excluding zero-valued voxels."
                if not include_zero
                else "No finite values are available for the selected layers."
            )
            plot_item.setXRange(0.0, 1.0, padding=0.0)
            plot_item.setYRange(0.0, 1.0, padding=0.0)
            self._set_histogram_message(message)
        else:
            plot_item.autoRange()

        self._refresh_histogram_threshold_lines()

    def _fill_mask_lower_bound_from_otsu(self) -> None:
        """Populate the lower bound from an Otsu threshold."""
        selected_layers = self._selected_histogram_layers()
        if not selected_layers:
            message = "Select at least one eligible heatmap layer."
            self._histogram_status_label.setText(message)
            return

        if (
            len(selected_layers) > 1
            and self._current_histogram_create_mode() != "merged"
        ):
            message = "Use Otsu Lower with one selected layer, or switch Create mode to Merged layer."
            self._histogram_status_label.setText(message)
            return

        if len(selected_layers) == 1:
            volume = np.asarray(selected_layers[0].data, dtype=np.float32)
        else:
            volume = merge_heatmap_volumes(
                [np.asarray(layer.data, dtype=np.float32) for layer in selected_layers]
            )

        threshold = otsu_threshold_positive(volume)
        self._set_mask_bounds(
            threshold,
            upper=float(self._mask_upper_threshold_spin.value())
            if self._mask_use_upper_bound_cb.isChecked()
            else None,
            enable_upper=self._mask_use_upper_bound_cb.isChecked(),
            bounds_source="otsu_lower",
        )
        self._histogram_status_label.setText(
            f"Lower bound set to Otsu threshold {threshold:.4f}."
        )

    def _copy_selected_layer_contrast_to_bounds(self) -> None:
        """Copy the selected layer contrast limits into the mask bounds."""
        selected_layers = self._selected_histogram_layers()
        if len(selected_layers) != 1:
            message = (
                "Select exactly one eligible heatmap layer to sync contrast limits."
            )
            self._histogram_status_label.setText(message)
            return

        limits = self._layer_contrast_limits(selected_layers[0])
        if limits is None:
            message = "The selected layer does not expose contrast limits."
            self._histogram_status_label.setText(message)
            return

        lower, upper = limits
        self._set_mask_bounds(
            lower,
            upper=upper,
            enable_upper=True,
            bounds_source="contrast_limits",
        )
        self._histogram_status_label.setText(
            f"Copied contrast limits to mask bounds: {lower:.4f} to {upper:.4f}."
        )

    def _selected_mask_threshold_bounds(self) -> tuple[float, float | None] | None:
        """Return the configured lower and optional upper bounds."""
        lower = float(self._mask_lower_threshold_spin.value())
        upper = None
        if self._mask_use_upper_bound_cb.isChecked():
            upper = float(self._mask_upper_threshold_spin.value())
            if upper < lower:
                self._histogram_status_label.setText(
                    "Upper bound must be greater than or equal to lower bound."
                )
                return None
        return lower, upper

    def _select_heatmap_layer_names(self, names: list[str]) -> None:
        """Select Tools heatmap list items by name."""
        if not hasattr(self, "_heatmap_layer_list"):
            return

        selected_names = set(names)
        self._heatmap_layer_list.blockSignals(True)
        self._heatmap_layer_list.clearSelection()
        for index in range(self._heatmap_layer_list.count()):
            item = self._heatmap_layer_list.item(index)
            item.setSelected(item.text() in selected_names)
        self._heatmap_layer_list.blockSignals(False)
        self._update_tools_controls()

    def _select_histogram_layer_names(self, names: list[str]) -> None:
        """Select Histogram heatmap list items by name."""
        if not hasattr(self, "_histogram_layer_list"):
            return

        selected_names = set(names)
        self._histogram_layer_list.blockSignals(True)
        self._histogram_layer_list.clearSelection()
        for index in range(self._histogram_layer_list.count()):
            item = self._histogram_layer_list.item(index)
            item.setSelected(item.text() in selected_names)
        self._histogram_layer_list.blockSignals(False)
        self._update_histogram_controls()
        self._update_histogram_plot()

    def _selected_mask_query_layers(self) -> list:
        """Return the currently selected generated mask layers."""
        if not hasattr(self, "_mask_layer_selector"):
            return []
        names = set(self._mask_layer_selector.get_selected_layer_names())
        if not names:
            return []
        return [layer for layer in self._generated_mask_layers() if layer.name in names]

    @staticmethod
    def _deduplicate_file_ids(file_ids: list[object]) -> list[object]:
        """Return file IDs in first-seen order using string-equivalent identity."""
        deduplicated: list[object] = []
        seen: set[str] = set()
        for file_id in file_ids:
            key = str(file_id)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(file_id)
        return deduplicated

    def _source_file_ids_for_layers(self, layers: list) -> list[object]:
        """Return source neuron IDs carried by one or more heatmap/mask layers."""
        file_ids: list[object] = []
        for layer in layers:
            metadata = _layer_metadata(layer)
            for key in (
                "query_excluded_file_ids",
                "source_file_ids",
                "file_ids",
            ):
                file_ids.extend(self._normalise_layer_file_ids(metadata.get(key, ())))
        return NeuronViewerWidget._deduplicate_file_ids(file_ids)

    def _mask_source_exclusion_enabled(self) -> bool:
        """Return whether mask queries should exclude source neurons."""
        checkbox = getattr(self, "_mask_exclude_source_neurons_cb", None)
        is_checked = getattr(checkbox, "isChecked", None)
        if callable(is_checked):
            return bool(is_checked())
        return True

    def _update_mask_query_summary(self) -> None:
        """Update the Regions-tab summary for selected mask source neurons."""
        label = getattr(self, "_mask_query_hint_label", None)
        if label is None:
            return

        masks = self._generated_mask_layers()
        if not masks:
            label.setText("No generated mask layers are available.")
            return

        selected_layers = self._selected_mask_query_layers()
        if not selected_layers:
            label.setText(
                f"{len(masks)} generated mask layer(s) available. "
                "Select mask layers to see source-neuron count."
            )
            return

        source_file_ids = self._source_file_ids_for_layers(selected_layers)
        if not source_file_ids:
            label.setText("Selected mask layer(s) do not record source neurons.")
            return

        action = "excluded" if self._mask_source_exclusion_enabled() else "included"
        label.setText(
            "Selected mask layer(s) were generated from "
            f"{len(source_file_ids)} unique source neuron(s); "
            f"source neurons will be {action}."
        )

    def _on_mask_layer_selection_changed(self, selected_names: list[str]) -> None:
        """Update Regions status text when mask selection changes."""
        count = len(selected_names)
        if count == 0:
            self._regions_status_label.setText("")
        elif count == 1:
            self._regions_status_label.setText("1 mask layer selected for querying.")
        else:
            self._regions_status_label.setText(
                f"{count} mask layers selected for querying."
            )
        self._update_mask_query_summary()

    def _create_blurred_layers_from_heatmaps(self) -> None:
        """Create blurred image layers from the selected heatmaps."""
        if self._atlas is None:
            message = "Load an atlas before creating blurred heatmap layers."
            self._tools_status_label.setText(message)
            show_warning(message)
            return

        selected_layers = self._selected_heatmap_layers()
        if not selected_layers:
            message = "Select at least one eligible heatmap layer."
            self._tools_status_label.setText(message)
            return

        sigma = float(self._mask_sigma_spin.value())
        create_mode = self._current_tools_create_mode()
        created_layers = []

        try:
            if create_mode == "merged":
                merged_volume = merge_heatmap_volumes(
                    [
                        np.asarray(layer.data, dtype=np.float32)
                        for layer in selected_layers
                    ]
                )
                created_layers.append(
                    self._add_blurred_heatmap_layer(
                        layer_name=f"Blurred: merged {len(selected_layers)} heatmaps",
                        volume=merged_volume,
                        source_layers=selected_layers,
                        sigma=sigma,
                        merge_mode="merged_sum",
                    )
                )
            else:
                for layer in selected_layers:
                    created_layers.append(
                        self._add_blurred_heatmap_layer(
                            layer_name=f"Blurred: {layer.name}",
                            volume=np.asarray(layer.data, dtype=np.float32),
                            source_layers=[layer],
                            sigma=sigma,
                            merge_mode="separate",
                        )
                    )
        except Exception as e:
            logger.error("Failed to create blurred heatmap layers: %s", e)
            self._tools_status_label.setText(
                f"Failed to create blurred heatmap layers: {e}"
            )
            return

        self._refresh_heatmap_layer_list()
        self._refresh_histogram_layer_list()
        self._select_heatmap_layer_names([layer.name for layer in created_layers])
        self._select_histogram_layer_names([layer.name for layer in created_layers])
        nonempty = sum(
            int(np.any(np.asarray(layer.data) > 0)) for layer in created_layers
        )
        self._tools_status_label.setText(
            f"Created {len(created_layers)} blurred layer(s); {nonempty} contain nonzero voxels."
        )

    def _create_region_isolated_heatmaps(self) -> None:
        """Create heatmap image layers limited to selected atlas regions."""
        if self._atlas is None:
            message = "Load an atlas before creating isolated heatmap layers."
            self._tools_status_label.setText(message)
            show_warning(message)
            return

        selected_layers = self._selected_heatmap_layers()
        if not selected_layers:
            message = "Select at least one eligible heatmap layer."
            self._tools_status_label.setText(message)
            return

        region_ids = self._selected_region_isolation_region_ids()
        if not region_ids:
            message = "Select at least one atlas region for isolation."
            self._tools_status_label.setText(message)
            return

        selected_regions = self._selected_region_isolation_entries()
        selected_region_ids = [region_id for region_id, _acronym in selected_regions]
        selected_region_acronyms = [acronym for _region_id, acronym in selected_regions]
        region_label = self._region_isolation_label(selected_region_acronyms)
        selector = getattr(self, "_tools_region_selector", None)
        include_children = (
            bool(selector.include_children_enabled())
            if selector is not None and hasattr(selector, "include_children_enabled")
            else True
        )
        create_mode = self._current_region_isolation_create_mode()
        created_layers = []

        try:
            if create_mode == "merged":
                merged_volume = merge_heatmap_volumes(
                    [
                        np.asarray(layer.data, dtype=np.float32)
                        for layer in selected_layers
                    ]
                )
                isolated = isolate_heatmap_volume_to_region_ids(
                    merged_volume,
                    self._atlas,
                    region_ids,
                )
                created_layers.append(
                    self._add_region_isolated_heatmap_layer(
                        layer_name=(
                            f"Region Isolated ({region_label}): "
                            f"merged {len(selected_layers)} heatmaps"
                        ),
                        volume=isolated,
                        source_layers=selected_layers,
                        selected_region_ids=selected_region_ids,
                        selected_region_acronyms=selected_region_acronyms,
                        region_ids=region_ids,
                        include_children=include_children,
                        merge_mode="merged_sum",
                    )
                )
            else:
                for layer in selected_layers:
                    isolated = isolate_heatmap_volume_to_region_ids(
                        np.asarray(layer.data, dtype=np.float32),
                        self._atlas,
                        region_ids,
                    )
                    created_layers.append(
                        self._add_region_isolated_heatmap_layer(
                            layer_name=f"Region Isolated ({region_label}): {layer.name}",
                            volume=isolated,
                            source_layers=[layer],
                            selected_region_ids=selected_region_ids,
                            selected_region_acronyms=selected_region_acronyms,
                            region_ids=region_ids,
                            include_children=include_children,
                            merge_mode="separate",
                        )
                    )
        except Exception as e:
            logger.error("Failed to create isolated heatmap layers: %s", e)
            self._tools_status_label.setText(
                f"Failed to create isolated heatmap layers: {e}"
            )
            return

        self._refresh_heatmap_layer_list()
        self._refresh_histogram_layer_list()
        self._select_heatmap_layer_names([layer.name for layer in created_layers])
        self._select_histogram_layer_names([layer.name for layer in created_layers])
        nonempty = sum(
            int(np.any(np.asarray(layer.data) > 0)) for layer in created_layers
        )
        self._tools_status_label.setText(
            f"Created {len(created_layers)} isolated heatmap layer(s); "
            f"{nonempty} contain nonzero voxels."
        )

    def _create_masks_from_heatmaps(self) -> None:
        """Create binary mask label layers from selected heatmap image layers."""
        if self._atlas is None:
            message = "Load an atlas before creating mask layers."
            self._histogram_status_label.setText(message)
            show_warning(message)
            return

        selected_layers = self._selected_histogram_layers()
        if not selected_layers:
            message = "Select at least one eligible heatmap layer."
            self._histogram_status_label.setText(message)
            return

        bounds = self._selected_mask_threshold_bounds()
        if bounds is None:
            return
        lower_threshold, upper_threshold = bounds

        create_mode = self._current_histogram_create_mode()
        created_layers = []

        try:
            if create_mode == "merged":
                merged_volume = merge_heatmap_volumes(
                    [
                        np.asarray(layer.data, dtype=np.float32)
                        for layer in selected_layers
                    ]
                )
                mask = build_binary_mask_from_threshold_range(
                    merged_volume,
                    lower_threshold=lower_threshold,
                    upper_threshold=upper_threshold,
                )
                layer_name = f"Mask: merged {len(selected_layers)} heatmaps"
                created_layers.append(
                    self._add_mask_layer(
                        layer_name=layer_name,
                        mask=mask,
                        source_layers=selected_layers,
                        lower_threshold=lower_threshold,
                        upper_threshold=upper_threshold,
                        bounds_source=self._mask_bounds_source,
                        merge_mode="merged_sum",
                    )
                )
            else:
                for layer in selected_layers:
                    mask = build_binary_mask_from_threshold_range(
                        np.asarray(layer.data, dtype=np.float32),
                        lower_threshold=lower_threshold,
                        upper_threshold=upper_threshold,
                    )
                    created_layers.append(
                        self._add_mask_layer(
                            layer_name=f"Mask: {layer.name}",
                            mask=mask,
                            source_layers=[layer],
                            lower_threshold=lower_threshold,
                            upper_threshold=upper_threshold,
                            bounds_source=self._mask_bounds_source,
                            merge_mode="separate",
                        )
                    )
        except Exception as e:
            logger.error("Failed to create mask layers: %s", e)
            self._histogram_status_label.setText(f"Failed to create mask layers: {e}")
            return

        nonempty = sum(
            int(np.asarray(layer.data).sum() > 0) for layer in created_layers
        )
        bounds_text = f"lower {lower_threshold:.4f}"
        if upper_threshold is not None:
            bounds_text += f", upper {upper_threshold:.4f}"
        self._histogram_status_label.setText(
            f"Created {len(created_layers)} mask layer(s); {nonempty} contain nonzero voxels ({bounds_text})."
        )
        self._refresh_mask_layer_options()

    def _add_blurred_heatmap_layer(
        self,
        layer_name: str,
        volume: np.ndarray,
        source_layers: list,
        sigma: float,
        merge_mode: str,
    ):
        """Add or replace a blurred heatmap image layer."""
        from napari.utils.colormaps import Colormap

        for layer in list(self._iter_viewer_layers()):
            if layer.name == layer_name:
                self.viewer.layers.remove(layer)

        blurred = smooth_heatmap_volume(volume, sigma=sigma)
        first_layer = source_layers[0]
        rgba = _mask_layer_color(source_layers)
        source_file_ids = self._source_file_ids_for_layers(source_layers)

        colormap = (
            getattr(first_layer, "colormap", None) if len(source_layers) == 1 else None
        )
        if colormap is None and rgba is not None:
            colormap = Colormap(
                colors=[[0.0, 0.0, 0.0, 0.0], list(rgba)],
                name=f"blurred_{layer_name.lower().replace(' ', '_')}",
            )
        elif colormap is None:
            colormap = "hot"

        metadata = dict(_layer_metadata(first_layer)) if len(source_layers) == 1 else {}
        metadata.update(
            {
                "heatmap_source": True,
                "heatmap_native_grid": True,
                "heatmap_kind": "blurred",
                "blur_sigma": float(sigma),
                "source_heatmap_layers": [layer.name for layer in source_layers],
                "file_ids": source_file_ids or None,
                "source_file_ids": source_file_ids or None,
                "atlas_name": self._current_atlas_name(),
                "color": rgba,
                "merge_mode": merge_mode,
            }
        )
        metadata.update(_source_heatmap_filter_metadata(source_layers))

        add_kwargs = {
            "name": layer_name,
            "colormap": colormap,
            "blending": getattr(first_layer, "blending", "additive"),
            "rendering": getattr(first_layer, "rendering", "mip"),
            "opacity": getattr(
                first_layer, "opacity", self._opacity_slider.value() / 100.0
            ),
            "visible": True,
            "metadata": metadata,
        }
        for attr_name in ("scale", "translate"):
            value = getattr(first_layer, attr_name, None)
            if value is not None:
                add_kwargs[attr_name] = value

        layer = self.viewer.add_image(
            blurred,
            **add_kwargs,
        )
        return layer

    def _add_region_isolated_heatmap_layer(
        self,
        layer_name: str,
        volume: np.ndarray,
        source_layers: list,
        selected_region_ids: list[int],
        selected_region_acronyms: list[str],
        region_ids: list[int],
        include_children: bool,
        merge_mode: str,
    ):
        """Add one post-hoc region-isolated heatmap image layer."""
        from napari.utils.colormaps import Colormap

        first_layer = source_layers[0]
        rgba = _mask_layer_color(source_layers)
        source_file_ids = self._source_file_ids_for_layers(source_layers)

        colormap = (
            getattr(first_layer, "colormap", None) if len(source_layers) == 1 else None
        )
        if colormap is None and rgba is not None:
            colormap = Colormap(
                colors=[[0.0, 0.0, 0.0, 0.0], list(rgba)],
                name=f"region_isolated_{layer_name.lower().replace(' ', '_')}",
            )
        elif colormap is None:
            colormap = "hot"

        contrast_limits = _heatmap_contrast_limits(volume)
        metadata = dict(_layer_metadata(first_layer)) if len(source_layers) == 1 else {}
        for stale_key in (
            "heatmap_region",
            "heatmap_selected_region_id",
            "heatmap_selected_region_acronym",
        ):
            metadata.pop(stale_key, None)
        metadata.update(
            {
                "heatmap_source": True,
                "heatmap_native_grid": True,
                "heatmap_kind": "region_isolated",
                "source_heatmap_layers": [layer.name for layer in source_layers],
                "file_ids": source_file_ids or None,
                "source_file_ids": source_file_ids or None,
                "heatmap_selected_region_ids": [
                    int(region_id) for region_id in selected_region_ids
                ],
                "heatmap_selected_region_acronyms": [
                    str(acronym) for acronym in selected_region_acronyms
                ],
                "heatmap_region_ids": [int(region_id) for region_id in region_ids],
                "heatmap_include_child_regions": bool(include_children),
                "atlas_name": self._current_atlas_name(),
                "color": rgba,
                "merge_mode": merge_mode,
                "heatmap_contrast_limits": contrast_limits,
                "heatmap_autocontrast_policy": "stable_full_volume",
            }
        )
        metadata.update(_source_heatmap_filter_metadata(source_layers))

        add_kwargs = {
            "name": self._unique_layer_name(layer_name),
            "colormap": colormap,
            "blending": getattr(first_layer, "blending", "additive"),
            "rendering": getattr(first_layer, "rendering", "mip"),
            "opacity": getattr(
                first_layer, "opacity", self._opacity_slider.value() / 100.0
            ),
            "visible": True,
            "contrast_limits": contrast_limits,
            "metadata": metadata,
        }
        for attr_name in ("scale", "translate"):
            value = getattr(first_layer, attr_name, None)
            if value is not None:
                add_kwargs[attr_name] = value

        return self.viewer.add_image(
            np.asarray(volume, dtype=np.float32),
            **add_kwargs,
        )

    def _add_mask_layer(
        self,
        layer_name: str,
        mask: np.ndarray,
        source_layers: list,
        lower_threshold: float,
        upper_threshold: float | None,
        bounds_source: str,
        merge_mode: str,
    ):
        """Add or replace a generated binary mask layer."""
        from napari.utils import DirectLabelColormap

        for layer in list(self._iter_viewer_layers()):
            if layer.name == layer_name:
                self.viewer.layers.remove(layer)

        labels = np.asarray(mask, dtype=np.uint8)
        rgba = _mask_layer_color(source_layers)
        source_file_ids = self._source_file_ids_for_layers(source_layers)
        color_dict = {
            None: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        color_dict[1] = np.array(
            rgba if rgba is not None else (0.8, 0.8, 0.8, 1.0),
            dtype=np.float32,
        )

        layer = self.viewer.add_labels(
            labels,
            name=layer_name,
            opacity=self._opacity_slider.value() / 100.0,
            visible=True,
            colormap=DirectLabelColormap(color_dict=color_dict),
            metadata={
                "mask_query_source": True,
                "source_heatmap_layers": [layer.name for layer in source_layers],
                "file_ids": source_file_ids or None,
                "source_file_ids": source_file_ids or None,
                "query_excluded_file_ids": source_file_ids or None,
                "sigma": _shared_blur_sigma(source_layers),
                "threshold_mode": "range",
                "threshold_value": float(lower_threshold),
                "lower_threshold": float(lower_threshold),
                "upper_threshold": (
                    None if upper_threshold is None else float(upper_threshold)
                ),
                "bounds_source": bounds_source,
                "merge_mode": merge_mode,
                "atlas_name": self._current_atlas_name(),
                "color": rgba,
                **_source_heatmap_filter_metadata(source_layers),
            },
        )
        return layer

    def _on_region_query_source_changed(self, text: str) -> None:
        """Switch among atlas, custom-region, and mask query modes."""
        self._region_query_source = text
        if not hasattr(self, "_region_query_stack"):
            return

        page_index = {
            _REGION_QUERY_SOURCE_ATLAS: _REGION_QUERY_PAGE_ATLAS,
            _REGION_QUERY_SOURCE_CUSTOM: _REGION_QUERY_PAGE_CUSTOM,
            _REGION_QUERY_SOURCE_MASK: _REGION_QUERY_PAGE_MASK,
        }.get(text, _REGION_QUERY_PAGE_ATLAS)
        self._region_query_stack.setCurrentIndex(page_index)
        self._sync_region_query_scope_selector()
        find_button = getattr(self, "_region_query_find_btn", None)
        if find_button is not None:
            if text == _REGION_QUERY_SOURCE_MASK:
                find_button.setText("Find Neurons in Selected Mask Layers")
            elif text == _REGION_QUERY_SOURCE_CUSTOM:
                find_button.setText("Find Neurons in Selected Custom Regions")
            else:
                find_button.setText("Find Neurons in Selected Regions")
        else:
            show_mask_buttons = text == _REGION_QUERY_SOURCE_MASK
            for button in self._atlas_region_query_buttons():
                button.setVisible(not show_mask_buttons)
            for button in self._mask_layer_query_buttons():
                button.setVisible(show_mask_buttons)
        self._regions_status_label.setText("")
        if text in {
            _REGION_QUERY_SOURCE_ATLAS,
            _REGION_QUERY_SOURCE_CUSTOM,
        }:
            self._sync_active_region_reference_layers()
        refresh_appearance = getattr(self, "_refresh_region_appearance_selection", None)
        if callable(refresh_appearance):
            refresh_appearance()

    def _on_region_query_scope_changed(self, _text: str) -> None:
        """Track the current region-query scope selection."""
        self._region_query_scope = self._selected_region_query_scope()
        self._sync_region_query_scope_selector()
        if hasattr(self, "_regions_status_label"):
            self._regions_status_label.setText("")
        if getattr(
            self,
            "_region_query_source",
            _REGION_QUERY_SOURCE_ATLAS,
        ) in {
            _REGION_QUERY_SOURCE_ATLAS,
            _REGION_QUERY_SOURCE_CUSTOM,
        }:
            self._sync_active_region_reference_layers()
        refresh_appearance = getattr(self, "_refresh_region_appearance_selection", None)
        if callable(refresh_appearance):
            refresh_appearance()

    def _selected_region_query_scope(self) -> str:
        """Return the active search scope for region and mask queries."""
        combo = getattr(self, "_region_query_scope_combo", None)
        if combo is None:
            scope = getattr(self, "_region_query_scope", _REGION_QUERY_SCOPE_WHOLE)
            if scope in {_REGION_QUERY_SCOPE_WHOLE, _REGION_QUERY_SCOPE_CURRENT}:
                return scope
            return _REGION_QUERY_SCOPE_WHOLE

        current_data = getattr(combo, "currentData", None)
        if callable(current_data):
            data = current_data()
            if data in {_REGION_QUERY_SCOPE_WHOLE, _REGION_QUERY_SCOPE_CURRENT}:
                return str(data)

        current_text = getattr(combo, "currentText", None)
        text = current_text() if callable(current_text) else ""
        if text == "Current Table":
            return _REGION_QUERY_SCOPE_CURRENT
        return _REGION_QUERY_SCOPE_WHOLE

    def _current_table_file_ids(self) -> list[object]:
        """Return the file IDs currently present in the neuron table."""
        table = getattr(self, "_neuron_table", None)
        if table is None:
            return []

        file_ids_getter = getattr(table, "file_ids", None)
        if callable(file_ids_getter):
            return list(file_ids_getter())

        entries = getattr(table, "_entries", {})
        return list(entries.keys())

    def _resolve_region_query_file_scope(
        self,
    ) -> tuple[bool, list[object] | None, str, int | None]:
        """Resolve the optional base file-ID restriction for region queries."""
        scope = self._selected_region_query_scope()
        if scope != _REGION_QUERY_SCOPE_CURRENT:
            return True, None, "whole parquet", None

        file_ids = self._current_table_file_ids()
        if file_ids:
            return True, file_ids, "current table", len(file_ids)

        self._regions_status_label.setText(
            "Current table is empty; switch search scope to Whole Parquet or populate the table first."
        )
        return False, None, "current table", 0

    @staticmethod
    def _query_scope_status_suffix(scope_label: str, input_count: int | None) -> str:
        """Format a scope-aware suffix for query status messages."""
        suffix = f" within {scope_label}"
        if input_count is not None:
            suffix += f" (from {input_count} input neurons)"
        return suffix

    def _region_selector_for_scope(
        self,
        scope: str | None = None,
    ) -> RegionSelectorWidget | None:
        """Return the atlas-region selector widget for the requested scope."""
        selected_scope = scope
        if selected_scope not in {
            _REGION_QUERY_SCOPE_WHOLE,
            _REGION_QUERY_SCOPE_CURRENT,
        }:
            selected_scope = self._selected_region_query_scope()

        if selected_scope == _REGION_QUERY_SCOPE_CURRENT:
            selector = getattr(self, "_current_table_region_selector", None)
            if selector is not None:
                return selector
        else:
            selector = getattr(self, "_whole_parquet_region_selector", None)
            if selector is not None:
                return selector

        return getattr(self, "_region_selector", None)

    def _active_region_selector(self) -> RegionSelectorWidget | None:
        """Return the atlas-region selector for the current search scope."""
        return self._region_selector_for_scope()

    def _custom_region_selector_for_scope(
        self,
        scope: str | None = None,
    ) -> CustomRegionSelectorWidget | None:
        """Return the custom-region selector for the requested search scope."""
        selected_scope = scope
        if selected_scope not in {
            _REGION_QUERY_SCOPE_WHOLE,
            _REGION_QUERY_SCOPE_CURRENT,
        }:
            selected_scope = self._selected_region_query_scope()
        if selected_scope == _REGION_QUERY_SCOPE_CURRENT:
            return getattr(
                self,
                "_current_table_custom_region_selector",
                None,
            )
        return getattr(
            self,
            "_whole_parquet_custom_region_selector",
            None,
        )

    def _active_custom_region_selector(
        self,
    ) -> CustomRegionSelectorWidget | None:
        """Return the custom selector for the current search scope."""
        return self._custom_region_selector_for_scope()

    def _active_custom_region_groups(
        self,
    ) -> tuple[CustomRegionSelectionGroup, ...]:
        """Return selected custom terminal regions grouped by layer."""
        selector = self._active_custom_region_selector()
        if selector is None:
            return ()
        get_groups = getattr(selector, "get_selected_region_groups", None)
        if not callable(get_groups):
            return ()
        return tuple(get_groups())

    def _active_region_preview_acronyms(self) -> list[str]:
        """Return directly selected atlas acronyms for preview layers."""
        selector = self._active_region_selector()
        if selector is None:
            return []

        get_selected = getattr(selector, "get_selected_acronyms", None)
        if not callable(get_selected):
            return []
        return list(get_selected(include_children=False))

    def _active_region_include_children(self) -> bool:
        """Return the active atlas selector's include-children state."""
        selector = self._active_region_selector()
        if selector is None:
            return True

        include_children = getattr(selector, "include_children_enabled", None)
        if callable(include_children):
            return bool(include_children())
        return True

    def _active_flatmap_region_ids(self) -> list[int]:
        """Return active-source atlas IDs for flatmap region-label overlays."""
        source = getattr(
            self,
            "_region_query_source",
            _REGION_QUERY_SOURCE_ATLAS,
        )
        if source == _REGION_QUERY_SOURCE_CUSTOM:
            selector = self._active_custom_region_selector()
            if selector is None:
                return []
            get_selected = getattr(selector, "get_selected_region_ids", None)
            if not callable(get_selected):
                return []
            return sorted(
                {int(region_id) for region_id in get_selected() if int(region_id) > 0}
            )
        if source != _REGION_QUERY_SOURCE_ATLAS:
            return []

        selector = self._active_region_selector()
        if selector is None:
            return []

        get_selected = getattr(selector, "get_selected_ids", None)
        if not callable(get_selected):
            return []
        return sorted(
            {
                int(region_id)
                for region_id in get_selected(include_children=True)
                if int(region_id) > 0
            }
        )

    def _active_flatmap_parent_region_ids(self) -> list[int]:
        """Return active-source IDs for cached flatmap geometry."""
        source = getattr(
            self,
            "_region_query_source",
            _REGION_QUERY_SOURCE_ATLAS,
        )
        if source == _REGION_QUERY_SOURCE_CUSTOM:
            return self._active_flatmap_region_ids()
        if source != _REGION_QUERY_SOURCE_ATLAS:
            return []

        selector = self._active_region_selector()
        if selector is None:
            return []
        get_selected = getattr(selector, "get_selected_ids", None)
        if not callable(get_selected):
            return []
        return sorted(
            {
                int(region_id)
                for region_id in get_selected(include_children=False)
                if int(region_id) > 0
            }
        )

    def _active_flatmap_region_acronyms(self) -> list[str]:
        """Return active-source acronyms for flatmap-region metadata."""
        source = getattr(
            self,
            "_region_query_source",
            _REGION_QUERY_SOURCE_ATLAS,
        )
        if source == _REGION_QUERY_SOURCE_CUSTOM:
            acronym_by_id: dict[int, str] = {}
            for group in self._active_custom_region_groups():
                for region_id, acronym in zip(
                    group.region_ids,
                    group.acronyms,
                    strict=True,
                ):
                    acronym_by_id.setdefault(int(region_id), str(acronym))
            return [
                acronym_by_id[region_id]
                for region_id in self._active_flatmap_region_ids()
                if region_id in acronym_by_id
            ]
        if source != _REGION_QUERY_SOURCE_ATLAS:
            return []

        selector = self._active_region_selector()
        if selector is None:
            return []

        get_selected = getattr(selector, "get_selected_acronyms", None)
        if not callable(get_selected):
            return []
        return [str(acronym) for acronym in get_selected(include_children=True)]

    def _active_flatmap_region_source(self) -> str:
        """Return the normalized Regions source used by flatmap actions."""
        source = getattr(
            self,
            "_region_query_source",
            _REGION_QUERY_SOURCE_ATLAS,
        )
        return {
            _REGION_QUERY_SOURCE_ATLAS: "atlas_regions",
            _REGION_QUERY_SOURCE_CUSTOM: "custom_regions",
            _REGION_QUERY_SOURCE_MASK: "mask_layer",
        }.get(source, "atlas_regions")

    def _active_flatmap_region_scope(self) -> str:
        """Return the normalized selector scope used by flatmap actions."""
        return (
            "current_table"
            if self._selected_region_query_scope() == _REGION_QUERY_SCOPE_CURRENT
            else "whole_parquet"
        )

    def _active_flatmap_region_error(self) -> str | None:
        """Return an actionable error for an unavailable selection source."""
        source = getattr(
            self,
            "_region_query_source",
            _REGION_QUERY_SOURCE_ATLAS,
        )
        if source == _REGION_QUERY_SOURCE_MASK:
            return (
                "Flatmap atlas overlays do not support Mask Layer selections. "
                "Choose Atlas Regions or Custom Regions."
            )
        if source != _REGION_QUERY_SOURCE_CUSTOM:
            return None

        selector = self._active_custom_region_selector()
        if selector is None:
            return "Custom Isocortex Layers are unavailable for the loaded atlas."
        has_hierarchy = getattr(selector, "has_hierarchy", None)
        if callable(has_hierarchy) and not has_hierarchy():
            unavailable_message = getattr(selector, "unavailable_message", None)
            if callable(unavailable_message):
                message = str(unavailable_message()).strip()
                if message:
                    return message
            return "Custom Isocortex Layers are unavailable for the loaded atlas."
        return None

    def _region_appearance_store(self) -> RegionAppearanceStore:
        """Return a copy of the currently applied atlas appearance state."""
        editor = getattr(self, "_region_appearance_editor", None)
        if editor is not None:
            return editor.applied_store
        return RegionAppearanceStore(atlas_name=self._current_atlas_name() or "")

    def _refresh_region_appearance_selection(self) -> None:
        """Show the active Atlas/Custom selection subtree in the editor."""
        editor = getattr(self, "_region_appearance_editor", None)
        if editor is None:
            return
        source = getattr(
            self,
            "_region_query_source",
            _REGION_QUERY_SOURCE_ATLAS,
        )
        root_ids = (
            self._active_flatmap_parent_region_ids()
            if source in {_REGION_QUERY_SOURCE_ATLAS, _REGION_QUERY_SOURCE_CUSTOM}
            else []
        )
        editor.set_selection(root_ids)

    def _on_region_appearance_applied(
        self,
        _store: RegionAppearanceStore,
    ) -> None:
        """Restyle existing region overlays after an explicit Apply."""
        self._restyle_reference_region_layers()
        flatmap_tab = getattr(self, "_flatmap_tab", None)
        restyle_flatmap = getattr(flatmap_tab, "apply_region_appearance", None)
        if callable(restyle_flatmap):
            restyle_flatmap()

    def _sync_region_query_scope_selector(self) -> None:
        """Show the atlas and custom selectors matching the query scope."""
        index = (
            1
            if self._selected_region_query_scope() == _REGION_QUERY_SCOPE_CURRENT
            else 0
        )
        for name in (
            "_atlas_region_scope_stack",
            "_custom_region_scope_stack",
        ):
            stack = getattr(self, name, None)
            if stack is not None:
                stack.setCurrentIndex(index)

    def _sync_active_region_reference_layers(self) -> None:
        """Refresh reference layers from the active Atlas or Custom source."""
        show_meshes = getattr(self, "_show_region_meshes_cb", None)
        if show_meshes is not None and show_meshes.isChecked():
            self._sync_active_region_meshes()

        show_segmentation = getattr(self, "_show_region_seg_cb", None)
        if show_segmentation is not None and show_segmentation.isChecked():
            self._sync_active_region_segmentation()

    def _sync_active_region_meshes(self) -> None:
        """Refresh selected-region meshes for the current query source."""
        source = getattr(
            self,
            "_region_query_source",
            _REGION_QUERY_SOURCE_ATLAS,
        )
        if source == _REGION_QUERY_SOURCE_CUSTOM:
            self._update_custom_region_meshes(self._active_custom_region_groups())
            return
        self._update_region_meshes(self._active_region_preview_acronyms())

    def _sync_active_region_segmentation(self) -> None:
        """Refresh selected-region labels for the current query source."""
        source = getattr(
            self,
            "_region_query_source",
            _REGION_QUERY_SOURCE_ATLAS,
        )
        if source == _REGION_QUERY_SOURCE_CUSTOM:
            self._update_custom_region_segmentation(self._active_custom_region_groups())
            return
        self._update_region_segmentation(self._active_region_preview_acronyms())

    @staticmethod
    def _atlas_query_details(
        acronyms: list[str],
        *,
        include_children: bool,
    ) -> str:
        """Format the exact atlas-region query that was executed."""
        if len(acronyms) == 1:
            selection = acronyms[0]
        else:
            selection = f"union of {', '.join(acronyms)}"
        descendants = "on" if include_children else "off"
        return f"Query: {selection}; descendants: {descendants}."

    def _selected_region_query_node_types(self) -> tuple[int, ...] | None:
        """Return the active Regions-tab node-type filter."""
        combo = getattr(self, "_region_node_type_combo", None)
        getter = getattr(combo, "selected_node_types", None)
        if callable(getter):
            return getter()
        return None

    @staticmethod
    def _effective_query_node_types(
        node_types: tuple[int, ...] | list[int] | None,
        soma_only: bool,
    ) -> tuple[int, ...] | None:
        """Resolve explicit node types and old soma-only calls."""
        if node_types is None and soma_only:
            return (1,)
        return None if node_types is None else tuple(int(value) for value in node_types)

    def _atlas_region_query_buttons(self) -> tuple[QPushButton, ...]:
        """Return the atlas-region query buttons."""
        find_button = getattr(self, "_region_query_find_btn", None)
        if find_button is not None:
            return (find_button,)
        buttons = []
        for name in ("_atlas_query_any_node_btn", "_atlas_query_soma_btn"):
            button = getattr(self, name, None)
            if button is not None:
                buttons.append(button)
        return tuple(buttons)

    def _mask_layer_query_buttons(self) -> tuple[QPushButton, ...]:
        """Return the mask-layer query buttons."""
        find_button = getattr(self, "_region_query_find_btn", None)
        if find_button is not None:
            return (find_button,)
        buttons = []
        for name in ("_mask_query_any_node_btn", "_mask_query_soma_btn"):
            button = getattr(self, name, None)
            if button is not None:
                buttons.append(button)
        return tuple(buttons)

    def _set_region_query_buttons_enabled(self, enabled: bool) -> None:
        """Enable or disable all Regions-tab query buttons."""
        buttons: list[QPushButton] = []
        seen: set[int] = set()
        for button in (
            *self._atlas_region_query_buttons(),
            *self._mask_layer_query_buttons(),
        ):
            key = id(button)
            if key in seen:
                continue
            seen.add(key)
            buttons.append(button)
        for button in buttons:
            button.setEnabled(enabled)

    def _query_neurons_for_active_region_source(self) -> None:
        """Query neurons from the active Regions-tab source."""
        selector = getattr(self, "_selected_region_query_node_types", None)
        if callable(selector):
            node_types = selector()
        else:
            combo = getattr(self, "_region_node_type_combo", None)
            getter = getattr(combo, "selected_node_types", None)
            node_types = getter() if callable(getter) else None
        membership = NodeTypeSelectorComboBox.query_text(node_types)
        source = getattr(
            self,
            "_region_query_source",
            _REGION_QUERY_SOURCE_ATLAS,
        )
        if source == _REGION_QUERY_SOURCE_MASK:
            self._regions_status_label.setText(
                f"Searching for neurons with {membership} in selected mask layers. Please wait..."
            )
            QApplication.processEvents()
            self._query_neurons_by_mask(node_types=node_types)
            return
        if source == _REGION_QUERY_SOURCE_CUSTOM:
            self._regions_status_label.setText(
                f"Searching for neurons with {membership} in selected custom "
                "regions. Please wait..."
            )
            QApplication.processEvents()
            self._query_neurons_by_custom_region(node_types=node_types)
            return

        self._regions_status_label.setText(
            f"Searching for neurons with {membership} in selected atlas regions. Please wait..."
        )
        QApplication.processEvents()
        self._query_neurons_by_region(node_types=node_types)

    def _query_atlas_neurons_any_node(self) -> None:
        """Query neurons with any node in the selected atlas regions."""
        self._regions_status_label.setText(
            "Searching for neurons with any node in selected atlas regions. Please wait..."
        )
        QApplication.processEvents()
        self._query_neurons_by_region(soma_only=False)

    def _query_atlas_neurons_soma(self) -> None:
        """Query neurons with soma in the selected atlas regions."""
        self._regions_status_label.setText(
            "Searching for neurons with soma in selected atlas regions. Please wait..."
        )
        QApplication.processEvents()
        self._query_neurons_by_region(soma_only=True)

    def _query_mask_neurons_any_node(self) -> None:
        """Query neurons with any node in the selected mask layers."""
        self._regions_status_label.setText(
            "Searching for neurons with any node in selected mask layers. Please wait..."
        )
        QApplication.processEvents()
        self._query_neurons_by_mask(soma_only=False)

    def _query_mask_neurons_soma(self) -> None:
        """Query neurons with soma in the selected mask layers."""
        self._regions_status_label.setText(
            "Searching for neurons with soma in selected mask layers. Please wait..."
        )
        QApplication.processEvents()
        self._query_neurons_by_mask(soma_only=True)

    def _on_regions_selected(self, acronyms: list[str]) -> None:
        """Handle region selection changes."""
        _ = acronyms
        refresh_appearance = getattr(self, "_refresh_region_appearance_selection", None)
        if callable(refresh_appearance):
            refresh_appearance()
        if (
            getattr(
                self,
                "_region_query_source",
                _REGION_QUERY_SOURCE_ATLAS,
            )
            != _REGION_QUERY_SOURCE_ATLAS
        ):
            return
        preview_acronyms = self._active_region_preview_acronyms()

        # Update region meshes if enabled
        if self._show_region_meshes_cb.isChecked():
            self._update_region_meshes(preview_acronyms)

        # Update region segmentation if enabled
        if self._show_region_seg_cb.isChecked():
            self._update_region_segmentation(preview_acronyms)

    def _on_custom_regions_selected(self, _region_ids: list[int]) -> None:
        """Refresh reference previews for an active custom selection."""
        refresh_appearance = getattr(self, "_refresh_region_appearance_selection", None)
        if callable(refresh_appearance):
            refresh_appearance()
        if hasattr(self, "_regions_status_label"):
            self._regions_status_label.setText("")
        if (
            getattr(
                self,
                "_region_query_source",
                _REGION_QUERY_SOURCE_ATLAS,
            )
            == _REGION_QUERY_SOURCE_CUSTOM
        ):
            self._sync_active_region_reference_layers()

    def _query_neurons_by_custom_region(
        self,
        soma_only: bool = False,
        node_types: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        """Query neurons by exactly the selected custom terminal region IDs."""
        if self._db is None:
            return
        if self._atlas is None:
            self._regions_status_label.setText(
                "Load a compatible Allen atlas before querying Custom Regions."
            )
            return

        selector = self._active_custom_region_selector()
        if selector is None:
            self._regions_status_label.setText(
                "Custom Isocortex Layers are unavailable."
            )
            return
        has_hierarchy = getattr(selector, "has_hierarchy", None)
        if callable(has_hierarchy) and not has_hierarchy():
            unavailable_message = getattr(selector, "unavailable_message", None)
            message = (
                str(unavailable_message())
                if callable(unavailable_message)
                else "Custom Isocortex Layers are unavailable."
            )
            self._regions_status_label.setText(message)
            return
        region_ids = sorted(
            {int(region_id) for region_id in selector.get_selected_region_ids()}
        )
        if not region_ids:
            self._regions_status_label.setText(
                "Select at least one terminal Custom Region."
            )
            return

        has_column = getattr(self._db, "has_column", None)
        if callable(has_column) and not has_column("region_id"):
            self._regions_status_label.setText(
                "Custom Regions require a region_id column in the loaded "
                "Parquet. Regenerate it with Allen region annotations."
            )
            return

        try:
            proceed, base_file_ids, scope_label, input_count = (
                self._resolve_region_query_file_scope()
            )
            if not proceed:
                return
            effective_node_types = self._effective_query_node_types(
                node_types,
                soma_only,
            )
            result = self._db.get_neurons_by_region_id(
                region_ids,
                soma_only=soma_only,
                file_ids=base_file_ids,
                node_types=effective_node_types,
            )
            self._populate_neuron_table(
                result,
                preserve_existing=base_file_ids is not None,
            )
            membership = NodeTypeSelectorComboBox.query_text(effective_node_types)
            scope_suffix = self._query_scope_status_suffix(
                scope_label,
                input_count,
            )
            region_word = "region" if len(region_ids) == 1 else "regions"
            if result.empty:
                self._regions_status_label.setText(
                    f"No neurons found with {membership} in "
                    f"{len(region_ids)} selected terminal custom {region_word}"
                    f"{scope_suffix}."
                )
            else:
                self._regions_status_label.setText(
                    f"Found {len(result)} neuron(s) with {membership} in "
                    f"{len(region_ids)} selected terminal custom {region_word}"
                    f"{scope_suffix}."
                )
            logger.info(
                "Found %d neurons with %s in %d terminal custom regions within %s",
                len(result),
                membership,
                len(region_ids),
                scope_label,
            )
        except Exception as exc:
            logger.error("Custom region query failed: %s", exc)
            self._regions_status_label.setText(f"Custom region query failed: {exc}")

    def _query_neurons_by_region(
        self,
        soma_only: bool = False,
        node_types: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        """Query neurons in selected regions."""
        if self._db is None:
            return

        selector = self._active_region_selector()
        if selector is None:
            self._regions_status_label.setText("Select at least one atlas region.")
            return

        get_query_acronyms = getattr(selector, "get_query_acronyms", None)
        acronyms = list(get_query_acronyms()) if callable(get_query_acronyms) else []
        if not acronyms:
            self._regions_status_label.setText("Select at least one atlas region.")
            return

        try:
            proceed, base_file_ids, scope_label, input_count = (
                self._resolve_region_query_file_scope()
            )
            if not proceed:
                return

            effective_node_types = self._effective_query_node_types(
                node_types,
                soma_only,
            )
            result = self._db.get_neurons_by_region(
                acronyms,
                soma_only=soma_only,
                file_ids=base_file_ids,
                node_types=effective_node_types,
            )
            self._populate_neuron_table(
                result,
                preserve_existing=base_file_ids is not None,
            )
            membership = NodeTypeSelectorComboBox.query_text(effective_node_types)
            include_children = False
            include_children_enabled = getattr(
                selector,
                "include_children_enabled",
                None,
            )
            if callable(include_children_enabled):
                include_children = bool(include_children_enabled())
            query_details = NeuronViewerWidget._atlas_query_details(
                acronyms,
                include_children=include_children,
            )
            self._regions_status_label.setText(
                "Found "
                f"{len(result)} neuron(s) with {membership} in selected atlas regions"
                f"{self._query_scope_status_suffix(scope_label, input_count)}. "
                f"{query_details}"
            )
            logger.info(
                "Found %d neurons with %s in selected atlas regions within %s",
                len(result),
                membership,
                scope_label,
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            self._regions_status_label.setText(f"Region query failed: {e}")

    def _query_neurons_by_mask(
        self,
        soma_only: bool = False,
        node_types: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        """Query neurons using a generated mask layer."""
        if self._db is None or self._atlas is None:
            return

        layers = self._selected_mask_query_layers()
        if not layers:
            self._regions_status_label.setText(
                "Select at least one generated mask layer."
            )
            return

        mask = np.logical_or.reduce([np.asarray(layer.data) > 0 for layer in layers])
        if not mask.any():
            message = "Selected mask layer selection is empty and cannot be queried."
            self._regions_status_label.setText(message)
            show_warning(message)
            return

        source_file_ids = self._source_file_ids_for_layers(layers)
        exclude_file_ids = (
            source_file_ids if self._mask_source_exclusion_enabled() else []
        )
        try:
            proceed, base_file_ids, scope_label, input_count = (
                self._resolve_region_query_file_scope()
            )
            if not proceed:
                return

            effective_node_types = self._effective_query_node_types(
                node_types,
                soma_only,
            )
            result = self._db.get_neurons_by_mask(
                mask,
                self._atlas,
                soma_only=soma_only,
                file_ids=base_file_ids,
                exclude_file_ids=exclude_file_ids or None,
                node_types=effective_node_types,
            )
            self._populate_neuron_table(
                result,
                preserve_existing=base_file_ids is not None,
            )
            membership = NodeTypeSelectorComboBox.query_text(effective_node_types)
            selected_names = ", ".join(layer.name for layer in layers[:3])
            if len(layers) > 3:
                selected_names += ", ..."
            exclusion_text = ""
            if exclude_file_ids:
                excluded_word = "neuron" if len(exclude_file_ids) == 1 else "neurons"
                exclusion_text = (
                    f"; excluded {len(exclude_file_ids)} source {excluded_word}"
                )
            self._regions_status_label.setText(
                "Found "
                f"{len(result)} neuron(s) with {membership} in "
                f"{len(layers)} selected mask layer(s)"
                f"{self._query_scope_status_suffix(scope_label, input_count)}: "
                f"{selected_names}{exclusion_text}"
            )
            logger.info(
                (
                    "Found %d neurons with %s in %d selected mask layers "
                    "within %s after excluding %d source neurons"
                ),
                len(result),
                membership,
                len(layers),
                scope_label,
                len(exclude_file_ids),
            )
        except Exception as e:
            logger.error(f"Mask query failed: {e}")
            self._regions_status_label.setText(f"Mask query failed: {e}")

    def _current_table_file_ids_in_scene(self) -> list[object]:
        """Return the subset of current table IDs that are rendered in the scene."""
        rendered = self._current_scene_file_ids()
        return [
            file_id for file_id in self._current_table_file_ids() if file_id in rendered
        ]

    def _cache_scene_display_state(
        self, file_ids: list[object] | tuple[object, ...]
    ) -> None:
        """Preserve base color and visibility for rendered neurons leaving the table."""
        scene_state = getattr(self, "_scene_display_state", None)
        if scene_state is None:
            self._scene_display_state = {}
            scene_state = self._scene_display_state

        for file_id in file_ids:
            color, visible = self._base_display_state_for_file_id(file_id)
            scene_state[file_id] = {
                "color": list(color),
                "visible": bool(visible),
            }

    def _discard_scene_display_state(
        self,
        file_ids: list[object] | tuple[object, ...],
    ) -> None:
        """Drop scene-only display state for neurons that are back in the table."""
        scene_state = getattr(self, "_scene_display_state", None)
        if not scene_state:
            return
        for file_id in file_ids:
            scene_state.pop(file_id, None)

    def _sync_after_neuron_table_membership_change(self) -> None:
        """Refresh derived UI and rendered colors after table membership changes."""
        self._last_soma_selection = set()
        self._refresh_cluster_filter_controls()
        refresh_clusters = getattr(
            self, "_refresh_apply_existing_clusters_button", None
        )
        if callable(refresh_clusters):
            refresh_clusters()
        self._refresh_neuron_table_summary()

        selected_getter = getattr(self._neuron_table, "get_selected_file_ids", None)
        selected_file_ids = selected_getter() if callable(selected_getter) else []
        current_layers = getattr(self, "_current_neuron_layers", [])
        if current_layers:
            rendered_ids = set(getattr(self, "_scene_render_modes", {}).keys())
            selected = set(selected_file_ids) & rendered_ids
            if not selected or selected == rendered_ids:
                self._highlighted_file_ids = None
            else:
                self._highlighted_file_ids = selected
            self._update_layer_colors(self._build_effective_color_map())
        else:
            self._highlighted_file_ids = None

    def _populate_neuron_table(
        self, result, *, preserve_existing: bool = False
    ) -> None:
        """Populate or subset the neuron table from a query result."""
        self._cache_scene_display_state(self._current_table_file_ids_in_scene())

        if preserve_existing:
            matched_file_ids = result["file_id"].tolist()
            self._neuron_table.retain_file_ids(matched_file_ids)
        else:
            self._neuron_table.populate(result["file_id"].tolist())

        saved_state_applier = getattr(self, "_apply_saved_table_state_to_table", None)
        if callable(saved_state_applier):
            saved_state_applier()
        self._discard_scene_display_state(self._current_table_file_ids())
        self._neuron_table.set_added_file_ids(self._current_scene_file_ids())
        sync_heatmaps = getattr(self, "_sync_neuron_table_heatmap_membership", None)
        if callable(sync_heatmaps):
            sync_heatmaps()
        refresh_manual_heatmaps = getattr(self, "_refresh_manual_heatmap_combo", None)
        if callable(refresh_manual_heatmaps):
            refresh_manual_heatmaps()
        manual_heatmap_data = getattr(self, "_manual_heatmap_combo_data", None)
        manual_heatmap_handler = getattr(
            self,
            "_on_manual_heatmap_selection_changed",
            None,
        )
        if (
            callable(manual_heatmap_data)
            and manual_heatmap_data() is not None
            and callable(manual_heatmap_handler)
        ):
            manual_heatmap_handler()
        self._sync_after_neuron_table_membership_change()

    def _apply_saved_table_state_to_table(self) -> None:
        """Apply enhanced parquet or project table state to current table rows."""
        table_state = getattr(self, "_saved_table_state", None)
        if not isinstance(table_state, dict) or not table_state.get("entries"):
            return
        applier = getattr(self._neuron_table, "apply_state", None)
        if callable(applier):
            applier(table_state, preserve_membership=True)
            self._refresh_cluster_assignment_controls()
            self._apply_active_cluster_assignment()

    def _selected_cluster_filter(self) -> ClusterFilterSelection:
        """Return selected cluster groups from the Data tab dropdown."""
        getter = getattr(self._cluster_filter_combo, "cluster_filter_selection", None)
        if callable(getter):
            return getter()

        idx = self._cluster_filter_combo.currentIndex()
        if idx < 0:
            return ClusterFilterSelection()
        data = self._cluster_filter_combo.itemData(idx)
        if data is None:
            return ClusterFilterSelection()
        try:
            return ClusterFilterSelection(frozenset({int(data)}))
        except (TypeError, ValueError):
            return ClusterFilterSelection()

    def _refresh_cluster_assignment_controls(self) -> None:
        """Refresh the named assignment selector and management buttons."""
        combo = getattr(self, "_cluster_assignment_combo", None)
        if combo is None:
            return
        store = self._cluster_assignment_store
        active_id = store.active_assignment_id
        blocked = combo.blockSignals(True)
        try:
            combo.clear()
            for assignment in store.sets():
                combo.addItem(assignment.name, assignment.assignment_id)
            if active_id is not None:
                index = combo.findData(active_id)
                combo.setCurrentIndex(index if index >= 0 else -1)
            else:
                combo.setCurrentIndex(-1)
        finally:
            combo.blockSignals(blocked)
        enabled = len(store) > 0
        self._rename_cluster_assignment_btn.setEnabled(enabled)
        self._delete_cluster_assignment_btn.setEnabled(enabled)
        self._refresh_comparison_metric_controls()

    def _active_assignment_color_map(self) -> dict[object, list[float]]:
        """Return active-set colors for all table and rendered neurons."""
        assignment = self._cluster_assignment_store.active
        if assignment is None:
            return {}
        default = [0.5, 0.5, 0.5, 1.0]
        file_ids = list(self._neuron_table.file_ids())
        seen = {str(file_id) for file_id in file_ids}
        for file_id in self._current_scene_file_ids():
            if str(file_id) not in seen:
                file_ids.append(file_id)
                seen.add(str(file_id))
        return {
            file_id: list(
                assignment.label_colors.get(
                    assignment.label_for(file_id),
                    default,
                )
            )
            for file_id in file_ids
        }

    def _apply_active_cluster_assignment(self) -> None:
        """Propagate the active assignment to all cluster-aware consumers."""
        self._neuron_table.refresh_cluster_assignments(preserve_selection=True)
        active_colors = self._active_assignment_color_map()
        if active_colors:
            self._neuron_table.update_colors(active_colors, emit_signal=False)
            effective_colors = self._build_effective_color_map()
            effective_colors.update(active_colors)
            self._update_layer_colors(effective_colors)
        self._refresh_cluster_filter_controls()
        self._refresh_neuron_table_summary()
        analysis_tab = getattr(self, "_analysis_tab", None)
        handler = getattr(analysis_tab, "on_active_cluster_assignment_changed", None)
        if callable(handler):
            handler()
        comparison_window = getattr(self, "_comparison_window", None)
        if comparison_window is not None and comparison_window.isVisible():
            comparison_window.refresh_sources()

    def _on_active_cluster_assignment_changed(self, index: int) -> None:
        """Activate the assignment selected in the Data tab."""
        combo = self._cluster_assignment_combo
        assignment_id = combo.itemData(index) if index >= 0 else None
        if assignment_id is None:
            return
        try:
            self._cluster_assignment_store.set_active(str(assignment_id))
        except KeyError:
            self._refresh_cluster_assignment_controls()
            return
        self._apply_active_cluster_assignment()

    def _rename_active_cluster_assignment(self) -> None:
        """Prompt for a new display name for the active assignment."""
        assignment = self._cluster_assignment_store.active
        if assignment is None:
            return
        from qtpy.QtWidgets import QInputDialog

        name, accepted = QInputDialog.getText(
            self,
            "Rename Cluster Assignment",
            "Assignment name:",
            text=assignment.name,
        )
        if not accepted or not str(name).strip():
            return
        self._cluster_assignment_store.rename(
            assignment.assignment_id,
            str(name),
        )
        self._refresh_cluster_assignment_controls()
        self._neuron_table.refresh_cluster_assignments(preserve_selection=True)
        self._refresh_neuron_table_summary()
        comparison_window = getattr(self, "_comparison_window", None)
        if comparison_window is not None:
            comparison_window.refresh_sources()

    def _delete_active_cluster_assignment(self) -> None:
        """Confirm and delete the active assignment without removing neurons."""
        assignment = self._cluster_assignment_store.active
        if assignment is None:
            return
        from qtpy.QtWidgets import QMessageBox

        response = QMessageBox.question(
            self,
            "Delete Cluster Assignment",
            f"Delete cluster assignment '{assignment.name}'? Neurons will not be removed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return
        self._cluster_assignment_store.delete(assignment.assignment_id)
        self._refresh_cluster_assignment_controls()
        self._apply_active_cluster_assignment()

    def _selected_cluster_from_filter(self) -> int | None:
        """Return the single selected cluster, if the filter is single-cluster."""
        selection = self._selected_cluster_filter()
        if selection.include_unclustered or len(selection.cluster_ids) != 1:
            return None
        return next(iter(selection.cluster_ids))

    def _refresh_cluster_filter_controls(self) -> None:
        """Refresh cluster filter dropdown options from table cluster assignments."""
        previous = self._selected_cluster_filter()
        cluster_ids = self._neuron_table.available_cluster_ids()
        has_unclustered = False
        unclustered_getter = getattr(
            self._neuron_table, "has_unclustered_entries", None
        )
        if callable(unclustered_getter):
            has_unclustered = bool(unclustered_getter())

        setter = getattr(self._cluster_filter_combo, "set_cluster_options", None)
        if callable(setter):
            setter(
                cluster_ids,
                include_unclustered=has_unclustered,
                selection=previous,
            )
        else:
            self._cluster_filter_combo.blockSignals(True)
            try:
                self._cluster_filter_combo.clear()
                self._cluster_filter_combo.addItem("All")
                self._cluster_filter_combo.setItemData(0, None)
                for cluster_id in cluster_ids:
                    self._cluster_filter_combo.addItem(f"Cluster {cluster_id}")
                    self._cluster_filter_combo.setItemData(
                        self._cluster_filter_combo.count() - 1,
                        int(cluster_id),
                    )
                if previous.include_unclustered or len(previous.cluster_ids) != 1:
                    selected_cluster = None
                else:
                    selected_cluster = next(iter(previous.cluster_ids), None)
                if selected_cluster is not None:
                    idx = self._cluster_filter_combo.findData(selected_cluster)
                    self._cluster_filter_combo.setCurrentIndex(idx if idx >= 0 else 0)
                else:
                    self._cluster_filter_combo.setCurrentIndex(0)
            finally:
                self._cluster_filter_combo.blockSignals(False)

        self._on_cluster_filter_changed(self._selected_cluster_filter())

    def _on_cluster_filter_changed(self, _selection: object = None) -> None:
        """Filter table rows by selected cluster groups and update action buttons."""
        selection = self._selected_cluster_filter()
        self._apply_neuron_table_filters()

        has_filter = not selection.is_all
        has_entries = bool(self._neuron_table.get_visibility_map())
        self._hide_others_btn.setEnabled(has_filter)
        self._recolor_cluster_btn.setEnabled(has_filter)
        self._show_all_btn.setEnabled(has_entries)

    def _refresh_apply_existing_clusters_button(self) -> None:
        """Show the cached-cluster reapply button only when it can do useful work."""
        button = getattr(self, "_apply_existing_clusters_btn", None)
        analysis_tab = getattr(self, "_analysis_tab", None)
        if button is None:
            return

        has_overlap = False
        checker = getattr(analysis_tab, "has_cached_clusters_for_current_table", None)
        if callable(checker):
            try:
                has_overlap = bool(checker())
            except Exception:
                has_overlap = False

        button.setVisible(has_overlap)
        button.setEnabled(has_overlap)

    def _apply_existing_clusters_from_analysis(self) -> None:
        """Reapply cached cluster state to the current table and rendered neurons."""
        analysis_tab = getattr(self, "_analysis_tab", None)
        applier = getattr(analysis_tab, "apply_cluster_colors", None)
        if not callable(applier):
            return

        summary = applier()
        matched_table_count = int(getattr(summary, "matched_table_count", 0))
        if matched_table_count <= 0:
            self._refresh_apply_existing_clusters_button()
            return

        message = (
            f"Applied cached cluster data to {matched_table_count} table neuron(s)."
        )
        rendered_count = int(getattr(summary, "rendered_count", 0))
        if rendered_count > 0:
            colored_count = int(getattr(summary, "colored_count", 0))
            message += (
                f" Recolored {colored_count}/{rendered_count} rendered neuron(s)."
            )
            gray_count = int(getattr(summary, "gray_count", 0))
            if gray_count > 0:
                message += f" {gray_count} shown in gray."

        self._render_status_label.setText(message)

    def _hide_not_in_selected_cluster(self) -> None:
        """Set visibility off for neurons outside the selected cluster groups."""
        selection = self._selected_cluster_filter()
        if selection.is_all:
            return
        self._neuron_table.hide_all_not_in_cluster(selection)

    def _show_all_neurons(self) -> None:
        """Restore visibility on for all neurons in the table."""
        self._neuron_table.set_all_visible()

    def _recolor_selected_cluster(self) -> None:
        """Recolor selected cluster groups distinctly and gray non-selected neurons."""
        selection = self._selected_cluster_filter()
        if selection.is_all:
            return
        self._neuron_table.recolor_cluster_distinct(
            selection,
            gray_others=True,
        )

    def _remove_selected_from_table(self) -> None:
        """Remove selected neurons from the table while keeping the scene intact."""
        selected_file_ids = self._neuron_table.get_selected_file_ids()
        if not selected_file_ids:
            self._render_status_label.setText(
                "Select at least one neuron row to remove from the table."
            )
            return

        rendered_selected = [
            file_id
            for file_id in selected_file_ids
            if file_id in self._current_scene_file_ids()
        ]
        self._cache_scene_display_state(rendered_selected)
        self._neuron_table.remove_file_ids(selected_file_ids)
        self._discard_scene_display_state(self._current_table_file_ids())
        self._neuron_table.set_added_file_ids(self._current_scene_file_ids())
        self._sync_after_neuron_table_membership_change()

        message = f"Removed {len(selected_file_ids)} neuron(s) from the table."
        self._render_status_label.setText(message)
        self._regions_status_label.setText(message)

    def _remove_unselected_from_table(self) -> None:
        """Keep selected neurons in the table and remove all other rows."""
        selected_file_ids = self._neuron_table.get_selected_file_ids()
        if not selected_file_ids:
            self._render_status_label.setText(
                "Select at least one neuron row to keep in the table."
            )
            return

        selected_file_id_set = set(selected_file_ids)
        current_file_ids = self._current_table_file_ids()
        unselected_file_ids = [
            file_id
            for file_id in current_file_ids
            if file_id not in selected_file_id_set
        ]
        if not unselected_file_ids:
            message = "All table neurons are selected; no unselected neurons to remove."
            self._render_status_label.setText(message)
            self._regions_status_label.setText(message)
            return

        scene_file_ids = self._current_scene_file_ids()
        rendered_unselected = [
            file_id for file_id in unselected_file_ids if file_id in scene_file_ids
        ]
        self._cache_scene_display_state(rendered_unselected)
        self._neuron_table.remove_file_ids(unselected_file_ids)
        self._discard_scene_display_state(self._current_table_file_ids())
        self._neuron_table.set_added_file_ids(scene_file_ids)
        self._neuron_table.select_file_ids(selected_file_ids)
        self._sync_after_neuron_table_membership_change()

        message = (
            f"Removed {len(unselected_file_ids)} unselected neuron(s) from the table."
        )
        self._render_status_label.setText(message)
        self._regions_status_label.setText(message)

    def _clear_neuron_table(self) -> None:
        """Clear the neuron table while leaving rendered scene layers in place."""
        self._cache_scene_display_state(self._current_table_file_ids_in_scene())
        self._neuron_table.clear()
        self._discard_scene_display_state(self._current_table_file_ids())
        self._neuron_table.set_added_file_ids(self._current_scene_file_ids())
        self._sync_after_neuron_table_membership_change()

        if hasattr(self, "_render_status_label"):
            self._render_status_label.setText("Cleared neuron table.")
        if hasattr(self, "_regions_status_label"):
            self._regions_status_label.setText("Cleared neuron table.")

    def _selected_neuron_heatmap_selection(self) -> tuple[object, ...]:
        """Validate and return a stable snapshot of selected table neuron IDs."""
        if self._selected_heatmap_running():
            return ()
        if self._db is None:
            self._render_status_label.setText(
                "Load a neuron Parquet before creating a heatmap."
            )
            return ()
        if self._atlas is None:
            self._render_status_label.setText(
                "Load an atlas before creating a neuron heatmap."
            )
            return ()

        selected_file_ids = tuple(
            sorted(self._neuron_table.get_selected_file_ids(), key=str)
        )
        if not selected_file_ids:
            self._render_status_label.setText(
                "Select at least one neuron row to create a heatmap."
            )
        return selected_file_ids

    def _add_selected_neurons_heatmap(self, _checked: bool = False) -> None:
        """Build one node-count heatmap for all currently selected neurons."""
        selected_file_ids = self._selected_neuron_heatmap_selection()
        if not selected_file_ids:
            return

        request = _SelectedNeuronHeatmapRequest(
            file_ids=tuple(str(file_id) for file_id in selected_file_ids),
            creation_mode=_SELECTED_HEATMAP_MODE_SINGLE,
        )
        self._start_selected_neuron_heatmap_requests([request])

    @staticmethod
    def _individual_heatmap_estimated_bytes(
        atlas: BrainGlobeAtlas,
        neuron_count: int,
    ) -> int:
        """Estimate retained float32 image bytes for individual heatmap layers."""
        shape = getattr(getattr(atlas, "annotation", None), "shape", ())
        if not shape:
            return 0
        voxel_count = int(np.prod(shape, dtype=np.int64))
        return voxel_count * np.dtype(np.float32).itemsize * max(0, int(neuron_count))

    @staticmethod
    def _individual_heatmap_memory_warning_text(
        neuron_count: int,
        estimated_bytes: int,
    ) -> str:
        """Return the confirmation text for a memory-heavy individual batch."""
        estimated_gib = float(estimated_bytes) / float(1024**3)
        return (
            f"Creating {int(neuron_count):,} individual heatmaps is estimated "
            f"to retain {estimated_gib:.2f} GiB of image data before rendering "
            "overhead. Continue?"
        )

    def _confirm_large_individual_heatmap_request(
        self,
        neuron_count: int,
        estimated_bytes: int,
    ) -> bool:
        """Ask whether an individual heatmap request above 1 GiB may run."""
        from qtpy.QtWidgets import QMessageBox

        message = QMessageBox(self)
        message.setIcon(QMessageBox.Warning)
        message.setWindowTitle("Large Individual Heatmap Request")
        message.setText(
            self._individual_heatmap_memory_warning_text(
                neuron_count,
                estimated_bytes,
            )
        )
        continue_button = message.addButton("Continue", QMessageBox.AcceptRole)
        cancel_button = message.addButton("Cancel", QMessageBox.RejectRole)
        message.setDefaultButton(cancel_button)
        message.exec()
        return message.clickedButton() is continue_button

    @staticmethod
    def _selected_neuron_colors_are_monochrome(
        colors: list[list[float]],
    ) -> bool:
        """Return whether at least two colors have matching RGB channels."""
        if len(colors) < 2:
            return False
        first_rgb = np.asarray(colors[0][:3], dtype=float)
        return all(
            np.allclose(first_rgb, np.asarray(color[:3], dtype=float))
            for color in colors[1:]
        )

    def _add_selected_neurons_individual_heatmaps(
        self,
        _checked: bool = False,
    ) -> None:
        """Build one color-matched node-count heatmap per selected neuron."""
        selected_file_ids = self._selected_neuron_heatmap_selection()
        if not selected_file_ids:
            return

        estimated_bytes = self._individual_heatmap_estimated_bytes(
            self._atlas,
            len(selected_file_ids),
        )
        if (
            estimated_bytes > _INDIVIDUAL_HEATMAP_MEMORY_WARNING_BYTES
            and not self._confirm_large_individual_heatmap_request(
                len(selected_file_ids),
                estimated_bytes,
            )
        ):
            self._render_status_label.setText(
                "Individual heatmap creation cancelled; no colors or layers changed."
            )
            return

        colors_by_file_id = {
            file_id: list(self._neuron_table.get_color(file_id))
            for file_id in selected_file_ids
        }
        if self._selected_neuron_colors_are_monochrome(
            list(colors_by_file_id.values())
        ):
            colors_by_file_id = distinct_colors_for_file_ids(selected_file_ids)
            self._neuron_table.update_colors(colors_by_file_id)

        requests = [
            _SelectedNeuronHeatmapRequest(
                file_ids=(str(file_id),),
                creation_mode=_SELECTED_HEATMAP_MODE_INDIVIDUAL,
                color=tuple(float(channel) for channel in colors_by_file_id[file_id]),
            )
            for file_id in selected_file_ids
        ]
        self._start_selected_neuron_heatmap_requests(requests)

    def _start_selected_neuron_heatmap_requests(
        self,
        requests: list[_SelectedNeuronHeatmapRequest],
    ) -> None:
        """Start one or more Data-tab heatmap requests in sequence."""
        if not requests:
            return

        self._selected_heatmap_pending_requests = list(requests)
        self._selected_heatmap_completed_requests = []
        self._selected_heatmap_current_request = None
        self._selected_heatmap_total = len(requests)
        self._selected_heatmap_index = 0
        self._selected_heatmap_failed = False
        self._render_progress.setVisible(True)
        self._render_progress.setRange(0, 0)
        self._update_selected_neuron_heatmap_controls()
        self._start_next_selected_neuron_heatmap_request()

    def _start_next_selected_neuron_heatmap_request(self) -> None:
        """Start the next queued selected-neuron heatmap request."""
        if not self._selected_heatmap_pending_requests:
            return

        request = self._selected_heatmap_pending_requests.pop(0)
        self._selected_heatmap_current_request = request
        self._selected_heatmap_request_file_ids = request.file_ids
        self._selected_heatmap_index = (
            len(self._selected_heatmap_completed_requests) + 1
        )
        self._start_selected_neuron_heatmap(request)

    def _start_selected_neuron_heatmap(
        self,
        request: _SelectedNeuronHeatmapRequest,
    ) -> None:
        """Start the background worker that builds a selected-neuron heatmap."""
        from ..workers import HeatmapWorker

        self._render_progress.setVisible(True)
        self._render_progress.setRange(0, 0)
        if request.creation_mode == _SELECTED_HEATMAP_MODE_INDIVIDUAL:
            self._render_status_label.setText(
                f"Building individual heatmap {self._selected_heatmap_index}/"
                f"{self._selected_heatmap_total} for {request.file_ids[0]}..."
            )
        else:
            self._render_status_label.setText(
                f"Building heatmap for {len(request.file_ids)} selected neuron(s)..."
            )

        thread = QThread()
        worker = HeatmapWorker(
            parquet_path=str(self._db.parquet_path),
            atlas=self._atlas,
            file_ids=list(request.file_ids),
        )
        self._selected_heatmap_thread = thread
        self._selected_heatmap_worker = worker
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_selected_neuron_heatmap_progress)
        worker.finished.connect(self._on_selected_neuron_heatmap_finished)
        worker.error.connect(self._on_selected_neuron_heatmap_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            lambda: self._cleanup_selected_heatmap_thread(thread, worker)
        )
        thread.finished.connect(thread.deleteLater)

        self._update_selected_neuron_heatmap_controls()
        thread.start()

    def _on_selected_neuron_heatmap_progress(
        self,
        message: str,
        current: int,
        total: int,
    ) -> None:
        """Update the Data-tab progress widgets for selected heatmap creation."""
        self._render_progress.setVisible(True)
        self._render_progress.setRange(0, max(1, total))
        self._render_progress.setValue(current)
        request = getattr(self, "_selected_heatmap_current_request", None)
        if (
            request is not None
            and request.creation_mode == _SELECTED_HEATMAP_MODE_INDIVIDUAL
        ):
            self._render_status_label.setText(
                f"Individual heatmap {self._selected_heatmap_index}/"
                f"{self._selected_heatmap_total} for {request.file_ids[0]}: {message}"
            )
        else:
            self._render_status_label.setText(message)

    def _add_selected_neuron_heatmap_layer(
        self,
        volume: np.ndarray,
        file_ids: list[str] | tuple[str, ...],
        *,
        creation_mode: str = _SELECTED_HEATMAP_MODE_SINGLE,
        color: tuple[float, float, float, float] | None = None,
    ):
        """Add one selected-neuron heatmap layer to the viewer."""
        layer_name = self._selected_neuron_heatmap_layer_name(file_ids)
        manual_heatmap_id = layer_name.removesuffix(" Heatmap")
        contrast_limits = _heatmap_contrast_limits(volume)
        if creation_mode == _SELECTED_HEATMAP_MODE_INDIVIDUAL:
            contrast_limits = (
                contrast_limits[0],
                contrast_limits[1] * _INDIVIDUAL_HEATMAP_CONTRAST_FRACTION,
            )
        colormap = "hot"
        if creation_mode == _SELECTED_HEATMAP_MODE_INDIVIDUAL and color is not None:
            from napari.utils.colormaps import Colormap

            colormap = Colormap(
                colors=[
                    [0.0, 0.0, 0.0, 0.0],
                    [float(color[0]), float(color[1]), float(color[2]), 1.0],
                ],
                name=f"manual_heatmap_{manual_heatmap_id}",
            )
        metadata = {
            "heatmap_source": True,
            "heatmap_native_grid": True,
            "heatmap_kind": "selected_neurons",
            "manual_heatmap_id": manual_heatmap_id,
            "atlas_name": self._current_atlas_name(),
            "source_path": (
                str(self._db.parquet_path) if self._db is not None else None
            ),
            "file_ids": [str(file_id) for file_id in file_ids],
            "selection_count": len(file_ids),
            "heatmap_creation_mode": creation_mode,
            "heatmap_contrast_limits": contrast_limits,
            "heatmap_autocontrast_policy": (
                "stable_20_percent_max"
                if creation_mode == _SELECTED_HEATMAP_MODE_INDIVIDUAL
                else "stable_full_volume"
            ),
        }
        if color is not None:
            metadata["color"] = [float(channel) for channel in color]
        return self.viewer.add_image(
            volume,
            name=layer_name,
            colormap=colormap,
            blending="additive",
            rendering="mip",
            opacity=self._opacity_slider.value() / 100.0,
            contrast_limits=contrast_limits,
            metadata=metadata,
        )

    def _on_selected_neuron_heatmap_finished(self, volume: np.ndarray) -> None:
        """Handle successful selected-neuron heatmap creation."""
        request = getattr(self, "_selected_heatmap_current_request", None)
        if request is None:
            file_ids = self._selected_heatmap_request_file_ids
            request = _SelectedNeuronHeatmapRequest(
                file_ids=tuple(file_ids),
                creation_mode=_SELECTED_HEATMAP_MODE_SINGLE,
            )
        else:
            file_ids = request.file_ids
        if not file_ids:
            return

        layer = self._add_selected_neuron_heatmap_layer(
            volume,
            file_ids,
            creation_mode=request.creation_mode,
            color=request.color,
        )
        completed = getattr(self, "_selected_heatmap_completed_requests", None)
        if completed is not None:
            completed.append(request)
        if getattr(self, "_selected_heatmap_total", 0) > 1:
            self._render_status_label.setText(
                f"Added {layer.name} ({self._selected_heatmap_index}/"
                f"{self._selected_heatmap_total})."
            )
        else:
            self._render_status_label.setText(
                f"Added {layer.name} with {(volume > 0).sum():,} non-zero voxels."
            )
        logger.info(
            "Created %s selected-neuron heatmap %s for %d neuron(s)",
            request.creation_mode,
            layer.name,
            len(file_ids),
        )
        refresher = getattr(
            self,
            "_refresh_selected_neuron_heatmap_dependents",
            None,
        )
        if callable(refresher):
            refresher()
        else:
            NeuronViewerWidget._refresh_selected_neuron_heatmap_dependents(self)

    def _refresh_selected_neuron_heatmap_dependents(self) -> None:
        """Refresh UI whose contents depend on Data-tab heatmap layers."""
        sync_heatmaps = getattr(self, "_sync_neuron_table_heatmap_membership", None)
        if callable(sync_heatmaps):
            sync_heatmaps()
        refresh_manual_heatmaps = getattr(self, "_refresh_manual_heatmap_combo", None)
        if callable(refresh_manual_heatmaps):
            refresh_manual_heatmaps()
        self._refresh_heatmap_layer_list()
        self._refresh_histogram_layer_list()
        self._refresh_mask_layer_options()

    def _on_selected_neuron_heatmap_error(self, error_msg: str) -> None:
        """Handle a worker failure and stop the remaining heatmap queue."""
        self._selected_heatmap_failed = True
        self._selected_heatmap_pending_requests = []
        self._render_progress.setVisible(False)
        self._render_progress.setRange(0, 1)
        self._render_progress.setValue(0)
        completed_count = len(getattr(self, "_selected_heatmap_completed_requests", []))
        total = int(getattr(self, "_selected_heatmap_total", 0))
        if total > 1:
            self._render_status_label.setText(
                f"Error after {completed_count}/{total} individual heatmaps: "
                f"{error_msg}"
            )
        else:
            self._render_status_label.setText(f"Error: {error_msg}")
        logger.error("Selected-neuron heatmap failed: %s", error_msg)

    def _cleanup_selected_heatmap_thread(
        self,
        thread: QThread,
        worker: object,
    ) -> None:
        """Release selected-neuron heatmap worker objects after completion."""
        if self._selected_heatmap_thread is thread:
            self._selected_heatmap_thread = None
        if self._selected_heatmap_worker is worker:
            self._selected_heatmap_worker = None
        self._selected_heatmap_request_file_ids = ()
        self._selected_heatmap_current_request = None

        if not getattr(self, "_selected_heatmap_failed", False) and getattr(
            self, "_selected_heatmap_pending_requests", []
        ):
            self._start_next_selected_neuron_heatmap_request()
            return

        completed_count = len(getattr(self, "_selected_heatmap_completed_requests", []))
        total = int(getattr(self, "_selected_heatmap_total", 0))
        failed = bool(getattr(self, "_selected_heatmap_failed", False))
        self._selected_heatmap_pending_requests = []
        self._selected_heatmap_completed_requests = []
        self._selected_heatmap_total = 0
        self._selected_heatmap_index = 0
        self._selected_heatmap_failed = False
        self._render_progress.setVisible(False)
        self._render_progress.setRange(0, 1)
        self._render_progress.setValue(0)
        if not failed and total > 1:
            self._render_status_label.setText(
                f"Added {completed_count} individual heatmaps to the scene."
            )
        self._update_selected_neuron_heatmap_controls()

    def _render_selected_neurons(self) -> None:
        """Render selected neurons with full trace geometry."""
        self._render_selected_with_mode(_SCENE_RENDER_MODE_FULL)

    def _render_selected_soma_only(self) -> None:
        """Render selected neurons using the shared soma-only layers."""
        self._render_selected_with_mode(_SCENE_RENDER_MODE_SOMA)

    def _render_selected_with_mode(self, render_mode: str) -> None:
        """Apply a scene render mode to the selected neurons and rebuild."""
        selected_file_ids = self._neuron_table.get_selected_file_ids()
        if not selected_file_ids:
            return

        changed = False
        for file_id in selected_file_ids:
            if self._scene_render_modes.get(file_id) == render_mode:
                continue
            self._scene_render_modes[file_id] = render_mode
            changed = True

        if not changed:
            mode_label = (
                "full traces" if render_mode == _SCENE_RENDER_MODE_FULL else "soma only"
            )
            self._render_status_label.setText(
                f"All selected neurons are already in the scene as {mode_label}."
            )
            return

        self._render_scene()

    def _remove_selected_neurons(self) -> None:
        """Remove selected neurons from the scene while leaving others in place."""
        selected_file_ids = set(self._neuron_table.get_selected_file_ids())
        if not selected_file_ids:
            return

        current_file_ids = self._current_scene_file_ids()
        if not current_file_ids:
            return

        removed_file_ids = current_file_ids & selected_file_ids
        if not removed_file_ids:
            self._render_status_label.setText(
                "No selected neurons are currently in the scene."
            )
            return

        for file_id in removed_file_ids:
            self._scene_render_modes.pop(file_id, None)
            scene_state = getattr(self, "_scene_display_state", None)
            if scene_state is not None:
                scene_state.pop(file_id, None)

        if not self._scene_render_modes:
            depth_state = self._capture_depth_state()
            self._clear_neuron_layers()
            self._restore_depth_state(depth_state)
            self._render_status_label.setText("Cleared all neurons from the scene.")
            return

        self._render_scene()

    def _render_scene(self) -> None:
        """Rebuild the neuron scene from the current per-neuron render modes."""
        if not self._scene_render_modes or self._db is None:
            return

        scene_render_modes = dict(self._scene_render_modes)
        file_ids = sorted(scene_render_modes, key=str)
        full_file_ids = [
            fid
            for fid in file_ids
            if scene_render_modes.get(fid) == _SCENE_RENDER_MODE_FULL
        ]
        n = len(file_ids)
        depth_state = self._capture_depth_state()
        use_auto_centering = self._use_auto_centering()

        # Show progress UI
        self._render_btn.setEnabled(False)
        self._render_soma_only_btn.setEnabled(False)
        self._remove_selected_btn.setEnabled(False)
        self._render_progress.setRange(0, n)
        self._render_progress.setValue(0)
        self._render_progress.setVisible(True)
        self._render_status_label.setText(f"Querying {n} neurons...")
        QApplication.processEvents()

        # Clear existing neuron layers
        self._clear_neuron_layers(reset_render_state=False)

        render_mode = self._render_mode_combo.currentText()
        opacity = self._opacity_slider.value() / 100.0

        color_map = self._build_effective_color_map()
        neuron_colors = []
        render_color_map: dict[object, list[float]] = {}
        for file_id in file_ids:
            fallback_color, _visible = self._base_display_state_for_file_id(file_id)
            color = list(color_map.get(file_id, fallback_color))
            neuron_colors.append(color)
            render_color_map[file_id] = color

        # Scale to match atlas mesh (coordinates are in microns)
        scale = None
        if self._atlas is not None:
            scale = [1.0 / res for res in self._atlas.resolution]

        line_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        points_df = None

        # --- Lines ---
        if full_file_ids and render_mode in ("Lines", "Both"):
            # Single batch query for all neurons
            line_data = self._db.get_neuron_lines_batch(full_file_ids)

            self._render_status_label.setText(
                f"Building line segments for {n} neurons..."
            )
            QApplication.processEvents()

            all_lines = []
            all_edge_colors = []
            projector_batch = {}
            rendered_file_ids = []
            segments_per_neuron = []

            for i, file_id in enumerate(full_file_ids):
                if file_id not in line_data:
                    continue
                color = render_color_map[file_id]
                coords, edges = line_data[file_id]
                if len(edges) == 0:
                    continue

                # Vectorized line segment building
                segments = np.stack([coords[edges[:, 0]], coords[edges[:, 1]]], axis=1)
                all_lines.append(segments)

                color_arr = np.empty((len(edges), 4))
                color_arr[:] = color[:4]
                all_edge_colors.append(color_arr)

                projector_batch[file_id] = (coords, edges, tuple(color))
                rendered_file_ids.append(file_id)
                segments_per_neuron.append(len(edges))

                self._render_progress.setValue(i + 1)
                if (i + 1) % 10 == 0:
                    QApplication.processEvents()

            if all_lines:
                merged_lines = np.concatenate(all_lines)
                merged_colors = np.concatenate(all_edge_colors)

                total_segs = len(merged_lines)
                self._render_status_label.setText(
                    f"Adding {total_segs:,} line segments to viewer..."
                )
                self._render_progress.setRange(0, 0)  # indeterminate
                QApplication.processEvents()

                layer = self.viewer.add_shapes(
                    merged_lines,
                    shape_type="line",
                    edge_width=self._line_width_spin.value(),
                    edge_color=merged_colors,
                    name="Neuron Lines",
                    opacity=opacity,
                    scale=scale,
                    metadata={
                        "file_ids": rendered_file_ids,
                        "segments_per_neuron": segments_per_neuron,
                    },
                )
                self._current_neuron_layers.append(layer)

            # Batch update slice projector (single rebuild)
            self._slice_projector.set_scale(scale)
            self._slice_projector.add_neuron_data_batch(projector_batch)

        # --- Points ---
        if full_file_ids and render_mode in ("Points", "Both"):
            self._render_status_label.setText("Querying point data...")
            self._render_progress.setRange(0, 0)  # indeterminate
            QApplication.processEvents()

            # Single batch query for all neurons
            points_df = self._db.get_neurons_for_rendering(full_file_ids)

            if not points_df.empty:
                self._render_status_label.setText(
                    f"Adding {len(points_df):,} points to viewer..."
                )
                QApplication.processEvents()

                coords = points_df[["x", "y", "z"]].values

                if self._color_by_type_cb.isChecked():
                    type_colors = {
                        1: [1, 0, 0, 1],  # Soma - red
                        2: [0, 0, 1, 1],  # Axon - blue
                        3: [0, 1, 0, 1],  # Basal dendrite - green
                        4: [1, 1, 0, 1],  # Apical dendrite - yellow
                    }
                    colors = np.array(
                        [
                            type_colors.get(t, [0.5, 0.5, 0.5, 1])
                            for t in points_df["type"].values
                        ]
                    )
                else:
                    # Per-point color based on which neuron each point belongs to
                    colors = np.array(
                        [
                            render_color_map.get(fid, list(_DEFAULT_NEURON_RGBA))[:4]
                            for fid in points_df["file_id"].values
                        ]
                    )

                layer = self.viewer.add_points(
                    coords,
                    size=self._point_size_spin.value(),
                    face_color=colors,
                    name="Neuron Points",
                    opacity=opacity,
                    scale=scale,
                    metadata={
                        "file_ids_per_point": points_df["file_id"].values.tolist(),
                        "point_types": points_df["type"].values.tolist(),
                        "base_face_colors": colors.copy(),
                    },
                )
                self._current_neuron_layers.append(layer)

        # --- Soma Labels ---
        soma_df = self._db.get_soma_locations(file_ids)
        if not soma_df.empty:
            soma_coords = soma_df[["x", "y", "z"]].values
            soma_fids = soma_df["file_id"].values.tolist()
            soma_colors = np.array(
                [
                    render_color_map.get(fid, list(_DEFAULT_NEURON_RGBA))[:4]
                    for fid in soma_fids
                ]
            )
            # Use neuron_id for the label text (shorter than file_id)
            labels = soma_df["neuron_id"].astype(str).values.tolist()

            soma_layer = self.viewer.add_points(
                soma_coords,
                size=50,
                face_color=soma_colors,
                border_color="white",
                border_width=0.05,
                text={
                    "string": labels,
                    "size": 10,
                    "color": "white",
                    "visible": False,
                },
                name="Soma Labels",
                opacity=0.7,
                scale=scale,
                metadata={"file_ids": soma_fids},
            )
            soma_layer.mode = "select"
            # Disable point movement — select mode allows clicking to
            # select but dragging would move points; override _move to
            # prevent that.
            soma_layer._move = lambda indices, position: None
            soma_layer.events.highlight.connect(self._on_soma_selected)
            self._current_neuron_layers.append(soma_layer)

        soma_projection_batch = self._build_soma_projection_batch(
            file_ids=file_ids,
            neuron_colors=neuron_colors,
            points_df=points_df,
        )
        self._soma_slice_projector.set_scale(scale)
        self._soma_slice_projector.point_size = _SOMA_SLICE_PROJECTION_POINT_SIZE
        self._soma_slice_projector.add_soma_data_batch(soma_projection_batch)

        # Hide neuron layers if currently in 2D mode (the ndisplay event
        # only fires on *changes*, so layers added while already in 2D
        # would otherwise stay visible).
        if self.viewer.dims.ndisplay == 2:
            self._apply_layer_visibility(False)

        # Default to showing the 2D slice projection once neurons are present.
        # This keeps "Show in 2D slices" on across subsequent additions.
        if self._current_neuron_layers:
            self._show_slice_projection_cb.setChecked(True)

        if use_auto_centering:
            centered = self._maybe_auto_center_slice(
                line_data=line_data,
                points_df=points_df,
                soma_df=soma_df,
                scale=scale,
            )
            if not centered:
                self._restore_depth_state(depth_state)
        else:
            self._restore_depth_state(depth_state)
        self._neuron_table.set_added_file_ids(self._current_scene_file_ids())

        # Hide progress UI
        self._render_progress.setVisible(False)
        self._render_status_label.setText(f"Rendered {n} neurons.")
        self._render_btn.setEnabled(True)
        self._render_soma_only_btn.setEnabled(True)
        self._remove_selected_btn.setEnabled(True)

    def _use_auto_centering(self) -> bool:
        """Return whether the Data tab centering mode is set to Auto."""
        return self._centering_mode_combo.currentData() == "auto"

    def _capture_depth_state(self) -> tuple[int, float] | None:
        """Capture current depth axis and point so Add/Remove can restore it."""
        depth_axis = depth_axis_from_not_displayed(
            getattr(self.viewer.dims, "not_displayed", None)
        )
        try:
            depth_value = float(self.viewer.dims.point[depth_axis])
        except Exception:
            return None
        return depth_axis, depth_value

    def _restore_depth_state(self, depth_state: tuple[int, float] | None) -> None:
        """Restore a previously captured depth state."""
        if depth_state is None:
            return

        depth_axis, depth_value = depth_state
        try:
            self.viewer.dims.set_point(depth_axis, depth_value)
            return
        except Exception:
            pass

        fallback_axis = depth_axis_from_not_displayed(
            getattr(self.viewer.dims, "not_displayed", None)
        )
        try:
            self.viewer.dims.set_point(fallback_axis, depth_value)
        except Exception:
            logger.debug("Failed to restore depth slice position.", exc_info=True)

    def _current_scene_file_ids(self) -> set[object]:
        """Collect neuron file IDs currently tracked in the scene state."""
        return set(self._scene_render_modes.keys())

    def _build_soma_projection_batch(
        self,
        file_ids: list[object],
        neuron_colors: list[list[float]],
        points_df,
    ) -> dict[str, tuple[np.ndarray, tuple]]:
        """Build a per-neuron soma-point batch for the 2D projection layer."""
        soma_points_df = None
        if points_df is not None and not points_df.empty and "type" in points_df:
            soma_points_df = points_df[points_df["type"] == 1]

        color_map = {
            fid: tuple(color[:4]) for fid, color in zip(file_ids, neuron_colors)
        }
        default_color = (0.5, 0.5, 0.5, 1.0)
        batch = {}

        if soma_points_df is not None and not soma_points_df.empty:
            for file_id, group in soma_points_df.groupby("file_id", sort=True):
                coords = group[["x", "y", "z"]].values.astype(np.float64)
                batch[file_id] = (coords, color_map.get(file_id, default_color))

        missing_file_ids = [fid for fid in file_ids if fid not in batch]
        if missing_file_ids and self._db is not None:
            missing_soma_points_df = self._db.get_soma_points(missing_file_ids)
            if not missing_soma_points_df.empty:
                for file_id, group in missing_soma_points_df.groupby(
                    "file_id", sort=True
                ):
                    coords = group[["x", "y", "z"]].values.astype(np.float64)
                    batch[file_id] = (coords, color_map.get(file_id, default_color))

        return batch

    def _soma_projection_active_in_2d(self) -> bool:
        """Return whether the shared 2D projection is active in 2D mode."""
        return bool(
            self.viewer.dims.ndisplay == 2
            and self._show_slice_projection_cb.isChecked()
        )

    def _set_neuron_points_soma_visibility(
        self,
        layer,
        hide_soma_points: bool,
    ) -> None:
        """Hide or restore soma-node entries within the Neuron Points layer."""
        meta = _layer_metadata(layer)
        point_types = np.asarray(meta.get("point_types", []))
        if point_types.size == 0:
            return

        base_colors = meta.get("base_face_colors")
        if base_colors is None:
            face_color = getattr(layer, "face_color", None)
            if face_color is None:
                return
            base_colors = np.asarray(face_color, dtype=float)
        else:
            base_colors = np.asarray(base_colors, dtype=float)

        if base_colors.ndim != 2 or base_colors.shape[0] != point_types.shape[0]:
            return

        meta["base_face_colors"] = base_colors.copy()
        layer.metadata = meta

        colors = base_colors.copy()
        if hide_soma_points:
            colors[point_types == 1, 3] = 0.0
        layer.face_color = colors

    def _sync_soma_projection_overlay_state(self) -> None:
        """Keep soma labels and raw soma nodes in sync with 2D projection state."""
        projection_active = self._soma_projection_active_in_2d()
        hide_soma_points = projection_active and (
            self._render_mode_combo.currentText() == "Points"
        )
        for layer in self._current_neuron_layers:
            if layer.name == "Soma Labels" and self.viewer.dims.ndisplay == 2:
                layer.visible = not projection_active
            elif layer.name == "Neuron Points":
                self._set_neuron_points_soma_visibility(layer, hide_soma_points)

    def _compute_center_of_rendered_neurons(
        self,
        line_data: dict[str, tuple[np.ndarray, np.ndarray]] | None,
        points_df,
        soma_df,
    ) -> np.ndarray | None:
        """Compute a center point with line > points > soma fallback priority."""
        return compute_center_of_rendered_neurons(
            line_data=line_data,
            points_df=points_df,
            soma_df=soma_df,
        )

    def _set_slice_depth_from_center(
        self,
        center_xyz: np.ndarray,
        scale: list[float] | None,
    ) -> bool:
        """Move the active depth slice to the center point."""
        depth_axis = depth_axis_from_not_displayed(
            getattr(self.viewer.dims, "not_displayed", None)
        )
        target_world = center_to_depth_world(center_xyz, depth_axis, scale)

        try:
            self.viewer.dims.set_point(depth_axis, target_world)
            return True
        except Exception:
            logger.debug("Failed to auto-center depth slice.", exc_info=True)
            return False

    def _maybe_auto_center_slice(
        self,
        line_data: dict[str, tuple[np.ndarray, np.ndarray]] | None,
        points_df,
        soma_df,
        scale: list[float] | None,
    ) -> bool:
        """Auto-center the depth slice once per widget session."""
        if self._auto_center_applied_once:
            return False

        center_xyz = self._compute_center_of_rendered_neurons(
            line_data=line_data,
            points_df=points_df,
            soma_df=soma_df,
        )
        if center_xyz is None:
            return False

        if self._set_slice_depth_from_center(center_xyz, scale):
            self._auto_center_applied_once = True
            return True
        return False

    def _base_display_state_for_file_id(
        self,
        file_id: object,
    ) -> tuple[list[float], bool]:
        """Return the stored base color and visibility for a neuron."""
        entries = getattr(self._neuron_table, "_entries", {})
        entry = entries.get(file_id)
        if entry is not None:
            return list(entry.color), bool(entry.visible)

        color_getter = getattr(self._neuron_table, "get_color", None)
        visibility_getter = getattr(self._neuron_table, "get_visibility_map", None)
        if callable(color_getter):
            color = list(color_getter(file_id))
            visible_map = visibility_getter() if callable(visibility_getter) else {}
            visible = bool(visible_map.get(file_id, True))
            return color, visible

        scene_state = getattr(self, "_scene_display_state", {})
        state = scene_state.get(file_id)
        if isinstance(state, dict):
            color = state.get("color", _DEFAULT_NEURON_RGBA)
            visible = bool(state.get("visible", True))
            return list(color), visible

        return list(_DEFAULT_NEURON_RGBA), True

    def _build_effective_color_map(self) -> dict[str, list[float]]:
        """Build a color map accounting for visibility and highlight state.

        - Hidden neurons get alpha=0.
        - When a highlight is active, non-highlighted neurons are dimmed to
          alpha=0.1 so the highlighted ones stand out.
        """
        highlight = getattr(self, "_highlighted_file_ids", None)
        result = {}
        entries = getattr(self._neuron_table, "_entries", {})
        file_ids = list(entries.keys())
        scene_state = getattr(self, "_scene_display_state", {})
        for fid in scene_state:
            if fid not in entries:
                file_ids.append(fid)

        for fid in file_ids:
            color, visible = self._base_display_state_for_file_id(fid)
            if not visible:
                color[3] = 0.0
            elif highlight is not None and fid not in highlight:
                color[3] = 0.1
            result[fid] = color
        return result

    def _update_layer_colors(self, color_map: dict[str, list[float]]) -> None:
        """Apply a color map to all neuron layers."""
        default_color = list(_DEFAULT_NEURON_RGBA)

        for layer in self._current_neuron_layers:
            if layer.name == "Neuron Lines":
                meta = layer.metadata or {}
                file_ids = meta.get("file_ids", [])
                seg_counts = meta.get("segments_per_neuron", [])
                if file_ids and seg_counts:
                    parts = []
                    for fid, count in zip(file_ids, seg_counts):
                        c = color_map.get(fid, default_color)
                        arr = np.empty((count, 4))
                        arr[:] = c[:4]
                        parts.append(arr)
                    layer.edge_color = np.concatenate(parts)

            elif layer.name == "Neuron Points":
                meta = layer.metadata or {}
                fids = meta.get("file_ids_per_point", [])
                if fids:
                    colors = np.array(
                        [color_map.get(fid, default_color)[:4] for fid in fids]
                    )
                    meta["base_face_colors"] = colors.copy()
                    layer.metadata = meta
                    layer.face_color = colors

            elif layer.name == "Soma Labels":
                meta = layer.metadata or {}
                fids = meta.get("file_ids", [])
                if fids:
                    colors = np.array(
                        [color_map.get(fid, default_color)[:4] for fid in fids]
                    )
                    layer.face_color = colors

        # Update slice projector
        self._slice_projector.update_neuron_colors(color_map)
        self._soma_slice_projector.update_neuron_colors(color_map)
        self._sync_soma_projection_overlay_state()

    def _apply_neuron_colors(self, changed: dict[str, list[float]]) -> None:
        """Handle color changes from the neuron table."""
        if not self._current_neuron_layers:
            return
        color_map = self._build_effective_color_map()
        self._update_layer_colors(color_map)

    def _apply_neuron_visibility(self, visibility_map: dict[str, bool]) -> None:
        """Handle visibility changes from the neuron table."""
        if not self._current_neuron_layers:
            return
        color_map = self._build_effective_color_map()
        self._update_layer_colors(color_map)

    def _highlight_selected_neurons(self, selected_file_ids: list[str]) -> None:
        """Highlight selected neurons by dimming all others.

        When some (but not all) neurons are selected in the table, the
        non-selected rendered neurons are dimmed to alpha=0.1. When nothing
        is selected or all are selected, highlighting is cleared.
        """
        if not self._current_neuron_layers:
            self._highlighted_file_ids = None
            return

        rendered_ids = set(getattr(self, "_scene_render_modes", {}).keys())
        selected = set(selected_file_ids) & rendered_ids
        if not selected or selected == rendered_ids:
            # Nothing selected or everything selected → clear highlight
            self._highlighted_file_ids = None
        else:
            self._highlighted_file_ids = selected

        color_map = self._build_effective_color_map()
        self._update_layer_colors(color_map)

    def _on_soma_selected(self, event) -> None:
        """Handle point selection on the Soma Labels layer.

        The highlight event fires on every mouse hover, not just clicks.
        We track the previous selection and only process when it actually
        changes to avoid expensive color updates on every mouse move.
        """
        layer = event.source
        current = set(layer.selected_data)
        file_ids = layer.metadata.get("file_ids", [])
        selected_fids = []
        seen_fids = set()
        for i in sorted(current):
            if i >= len(file_ids):
                continue
            fid = file_ids[i]
            if fid in seen_fids:
                continue
            seen_fids.add(fid)
            selected_fids.append(fid)

        current_key = (id(layer), frozenset(current), tuple(selected_fids))
        if current_key == self._last_soma_selection:
            return

        if not current:
            if not self._last_soma_selection:
                return
            self._last_soma_selection = current_key
            self._neuron_table.select_file_ids([])
            return

        self._last_soma_selection = current_key
        if selected_fids:
            self._neuron_table.select_file_ids(selected_fids)

    def _on_cluster_colors_updated(self, result, color_map: dict) -> None:
        """Handle cluster color updates from the analysis tab."""
        self._refresh_cluster_assignment_controls()
        self._neuron_table.update_cluster_assignments(result)
        active_colors = self._active_assignment_color_map()
        self._neuron_table.update_colors(
            active_colors or color_map,
            emit_signal=False,
        )
        self._neuron_table.sort_by_cluster()
        self._refresh_cluster_filter_controls()
        self._refresh_apply_existing_clusters_button()
        comparison_window = getattr(self, "_comparison_window", None)
        if comparison_window is not None and comparison_window.isVisible():
            comparison_window.refresh_sources()

    def _clear_all_neuron_layers(self, _checked: bool = False) -> None:
        """Clear all neuron layers from the UI button without preserving scene state."""
        self._clear_neuron_layers(reset_render_state=True)

    def _clear_neuron_layers(self, reset_render_state: bool = True) -> None:
        """Remove all current neuron layers and optionally reset scene state."""
        for layer in self._current_neuron_layers:
            try:
                self.viewer.layers.remove(layer)
            except ValueError:
                pass  # Layer already removed

        self._current_neuron_layers.clear()
        if reset_render_state:
            self._scene_render_modes.clear()
            scene_state = getattr(self, "_scene_display_state", None)
            if scene_state is not None:
                scene_state.clear()

        # Clear slice projector data
        self._slice_projector.clear()
        self._soma_slice_projector.clear()
        self._neuron_table.set_added_file_ids(self._current_scene_file_ids())

    def _toggle_template(self, state: int) -> None:
        """Toggle the template layer visibility."""
        requested = bool(state)
        with startup_timing(
            logger,
            "toggle_template",
            requested_state=requested,
            atlas_loaded=self._atlas is not None,
        ) as timing:
            if self._atlas is None:
                if not requested:
                    self._show_template_after_cached_atlas_load = False
                    timing.set(result="no_atlas_hide")
                    return
                if self._cached_atlas_autoload_running():
                    self._show_template_after_cached_atlas_load = True
                    timing.set(result="cached_atlas_loading")
                    return
                if self._atlas_load_running():
                    self._pending_reference_action = _REFERENCE_ACTION_TEMPLATE
                    timing.set(result="atlas_loading")
                    return
                with startup_timing(
                    logger,
                    "toggle_template_phase",
                    phase="load_atlas",
                ):
                    self._load_atlas(
                        pending_reference_action=_REFERENCE_ACTION_TEMPLATE
                    )
                timing.set(result="atlas_load_started")
                return

            layer_name = "Allen Template"

            if requested:
                existing = [
                    layer for layer in self.viewer.layers if layer.name == layer_name
                ]
                timing.set(existing_template_layers=len(existing))
                if not existing:
                    opacity = self._template_opacity_slider.value() / 100.0
                    with startup_timing(
                        logger,
                        "toggle_template_phase",
                        phase="add_allen_template",
                        opacity=opacity,
                    ):
                        add_allen_template(
                            self.viewer,
                            self._atlas,
                            opacity=opacity,
                        )
            else:
                removed = False
                for layer in self.viewer.layers:
                    if layer.name == layer_name:
                        self.viewer.layers.remove(layer)
                        removed = True
                        break
                timing.set(removed=removed)

    def _update_template_opacity(self, value: int) -> None:
        """Update the template layer opacity."""
        opacity = value / 100.0
        for layer in self.viewer.layers:
            if layer.name == "Allen Template":
                layer.opacity = opacity
                break

    def _toggle_outline(self, state: int) -> None:
        """Toggle the brain outline visibility."""
        if self._atlas is None:
            if not bool(state):
                return
            if self._atlas_load_running():
                self._pending_reference_action = _REFERENCE_ACTION_OUTLINE
                return
            self._load_atlas(pending_reference_action=_REFERENCE_ACTION_OUTLINE)
            return

        layer_name = "Brain Outline"

        if bool(state):
            existing = [
                layer for layer in self.viewer.layers if layer.name == layer_name
            ]
            if not existing:
                # Switch to 3D mode for mesh viewing
                if self.viewer.dims.ndisplay == 2:
                    self.viewer.dims.ndisplay = 3
                    show_info("Switched to 3D view for mesh display")
                add_brain_outline(self.viewer, self._atlas)
        else:
            for layer in self.viewer.layers:
                if layer.name == layer_name:
                    self.viewer.layers.remove(layer)
                    break

    def _toggle_region_meshes(self, state: int) -> None:
        """Toggle region mesh visibility."""
        if bool(state):
            self._sync_active_region_meshes()
        else:
            remove_region_layers(self.viewer)

    def _update_region_meshes(self, acronyms: list[str]) -> None:
        """Update displayed region meshes."""
        if self._atlas is None:
            if self._atlas_load_running():
                self._pending_reference_action = _REFERENCE_ACTION_MESHES
                return
            self._load_atlas(pending_reference_action=_REFERENCE_ACTION_MESHES)
            return

        # Remove existing region meshes
        remove_region_layers(self.viewer)

        if not self._show_region_meshes_cb.isChecked():
            return

        if not acronyms:
            return

        # Switch to 3D mode for mesh viewing
        if self.viewer.dims.ndisplay == 2:
            self.viewer.dims.ndisplay = 3
            show_info("Switched to 3D view for mesh display")

        # Add new meshes
        opacity = self._mesh_opacity_slider.value() / 100.0
        appearance_getter = getattr(self, "_region_appearance_store", None)
        appearance_store = appearance_getter() if callable(appearance_getter) else None
        appearance_kwargs = (
            {"appearance_store": appearance_store}
            if appearance_store is not None
            else {}
        )
        for acronym in acronyms:
            add_region_mesh(
                self.viewer,
                self._atlas,
                acronym,
                opacity=opacity,
                **appearance_kwargs,
            )

    def _update_custom_region_meshes(
        self,
        groups: tuple[CustomRegionSelectionGroup, ...],
    ) -> None:
        """Render selected custom terminal meshes in canonical layer groups."""
        if self._atlas is None:
            if self._atlas_load_running():
                self._pending_reference_action = _REFERENCE_ACTION_MESHES
                return
            self._load_atlas(pending_reference_action=_REFERENCE_ACTION_MESHES)
            return

        remove_region_layers(self.viewer)
        if not self._show_region_meshes_cb.isChecked() or not groups:
            return

        if self.viewer.dims.ndisplay == 2:
            self.viewer.dims.ndisplay = 3
            show_info("Switched to 3D view for mesh display")

        opacity = self._mesh_opacity_slider.value() / 100.0
        appearance_getter = getattr(self, "_region_appearance_store", None)
        appearance_store = appearance_getter() if callable(appearance_getter) else None
        appearance_kwargs = (
            {"appearance_store": appearance_store}
            if appearance_store is not None
            else {}
        )
        created_count = 0
        missing_acronyms: list[str] = []
        for group in groups:
            layer, missing = add_region_mesh_group(
                self.viewer,
                self._atlas,
                group,
                opacity=opacity,
                **appearance_kwargs,
            )
            if layer is not None:
                created_count += 1
            missing_acronyms.extend(missing)

        if missing_acronyms:
            preview = ", ".join(missing_acronyms[:8])
            suffix = "…" if len(missing_acronyms) > 8 else ""
            unavailable_suffix = (
                " No Custom Region mesh layers were created."
                if created_count == 0
                else ""
            )
            show_warning(
                f"Skipped {len(missing_acronyms)} Custom Region mesh(es) "
                f"that were unavailable: {preview}{suffix}.{unavailable_suffix}"
            )
        elif created_count == 0 and groups:
            show_warning("No meshes were available for the selected Custom Regions.")

    def _toggle_region_segmentation(self, state: int) -> None:
        """Toggle region segmentation visibility."""
        if bool(state):
            self._sync_active_region_segmentation()
        else:
            remove_region_segmentation(self.viewer)

    def _update_region_segmentation(self, acronyms: list[str]) -> None:
        """Update the region segmentation layer for selected regions."""
        if self._atlas is None:
            if self._atlas_load_running():
                self._pending_reference_action = _REFERENCE_ACTION_SEGMENTATION
                return
            self._load_atlas(pending_reference_action=_REFERENCE_ACTION_SEGMENTATION)
            return

        remove_region_segmentation(self.viewer)

        if not self._show_region_seg_cb.isChecked():
            return

        if not acronyms:
            return

        opacity = self._seg_opacity_slider.value() / 100.0
        appearance_getter = getattr(self, "_region_appearance_store", None)
        appearance_store = appearance_getter() if callable(appearance_getter) else None
        appearance_kwargs = (
            {"appearance_store": appearance_store}
            if appearance_store is not None
            else {}
        )
        add_region_segmentation(
            self.viewer,
            self._atlas,
            acronyms,
            opacity=opacity,
            **appearance_kwargs,
        )

    def _update_custom_region_segmentation(
        self,
        groups: tuple[CustomRegionSelectionGroup, ...],
    ) -> None:
        """Render exactly the selected custom terminal IDs as labels."""
        if self._atlas is None:
            if self._atlas_load_running():
                self._pending_reference_action = _REFERENCE_ACTION_SEGMENTATION
                return
            self._load_atlas(pending_reference_action=_REFERENCE_ACTION_SEGMENTATION)
            return

        remove_region_segmentation(self.viewer)
        if not self._show_region_seg_cb.isChecked():
            return

        region_ids = sorted(
            {int(region_id) for group in groups for region_id in group.region_ids}
        )
        if not region_ids:
            return

        opacity = self._seg_opacity_slider.value() / 100.0
        appearance_getter = getattr(self, "_region_appearance_store", None)
        appearance_store = appearance_getter() if callable(appearance_getter) else None
        appearance_kwargs = (
            {"appearance_store": appearance_store}
            if appearance_store is not None
            else {}
        )
        add_region_id_segmentation(
            self.viewer,
            self._atlas,
            region_ids,
            opacity=opacity,
            **appearance_kwargs,
        )

    def _restyle_reference_region_layers(self) -> None:
        """Apply committed region styles to current CCF layers in place."""
        if self._atlas is None:
            return
        from .reference_layers import apply_region_appearance_to_layer

        store = self._region_appearance_store()
        for layer in list(self.viewer.layers):
            try:
                apply_region_appearance_to_layer(layer, self._atlas, store)
            except Exception:
                logger.exception(
                    "Failed to apply region appearance to layer %s",
                    getattr(layer, "name", "<unnamed>"),
                )

    def _update_seg_opacity(self, value: int) -> None:
        """Update the region segmentation layer opacity."""
        opacity = value / 100.0
        for layer in self.viewer.layers:
            if layer.name == "Region Segmentation":
                layer.opacity = opacity
                break

    def _update_mesh_opacity(self, value: int) -> None:
        """Update the global CCF mesh opacity beneath region styling."""
        base_opacity = value / 100.0
        store = self._region_appearance_store()
        for layer in self.viewer.layers:
            metadata = getattr(layer, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            kind = str(metadata.get("region_layer_kind", "") or "")
            if kind not in {"mesh", "mesh_group"}:
                continue
            setattr(layer, "_napari_neuron_navigator_region_base_opacity", base_opacity)
            if kind == "mesh":
                region_id = int(metadata.get("region_id", 0) or 0)
                if region_id > 0 and self._atlas is not None:
                    effective = store.resolve(region_id, self._atlas.structures)
                    layer.opacity = base_opacity * effective.fill_opacity
            else:
                layer.opacity = base_opacity

    def _on_ndisplay_changed(self, event) -> None:
        """Auto-hide neuron line/point layers in 2D to keep slice scrubbing fast."""
        if not self._current_neuron_layers:
            return

        is_2d = self.viewer.dims.ndisplay == 2
        if is_2d:
            self.viewer.status = "Switching to 2D view..."
        else:
            self.viewer.status = "Rendering 3D neuron layers..."
        # Defer the heavy work so the status bar paints first
        QTimer.singleShot(0, lambda: self._apply_layer_visibility(not is_2d))

    def _apply_layer_visibility(self, visible: bool) -> None:
        """Set visibility on all neuron layers and clear the status message.

        In 2D mode, the raw points layer can stay visible for `Points` render
        mode while line layers are hidden for responsiveness.
        """
        show_points_in_2d = (
            not visible and self._render_mode_combo.currentText() == "Points"
        )
        for layer in self._current_neuron_layers:
            if not visible and layer.name == "Soma Labels":
                layer.visible = True
                continue
            if show_points_in_2d and layer.name == "Neuron Points":
                layer.visible = True
                continue
            layer.visible = visible
        self._sync_soma_projection_overlay_state()
        self.viewer.status = "Ready"

    def _toggle_slice_projection(self, state: int) -> None:
        """Toggle the 2D slice projection visibility."""
        enabled = bool(state)
        self._slice_projector.enabled = enabled
        self._soma_slice_projector.enabled = enabled
        self._slice_warning_label.setVisible(enabled)
        if self._current_neuron_layers and self.viewer.dims.ndisplay == 2:
            self._apply_layer_visibility(False)

    def _update_slice_thickness(self, value: int) -> None:
        """Update the slice projection thickness/tolerance."""
        self._slice_projector.tolerance = float(value)
        self._soma_slice_projector.tolerance = float(value)

    def _update_line_width(self, value: int) -> None:
        """Update line width for both neuron layers and projection."""
        for layer in self._current_neuron_layers:
            if hasattr(layer, "edge_width"):
                layer.edge_width = value
        self._slice_projector.edge_width = value

    # --- SWC-to-Parquet conversion ---

    def _choose_conversion_lookup_directory(self) -> None:
        """Choose bilateral shaped/square/depth lookup files for conversion."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Flatmap Lookup Directory",
        )
        if not directory:
            return
        self._convert_lookup_dir = Path(directory)
        self._convert_lookup_dir_label.setText(str(self._convert_lookup_dir))
        self._convert_add_flatmap_cb.setChecked(True)

    def _cancel_conversion(self) -> None:
        """Request cancellation of the active conversion pipeline."""
        worker = getattr(self, "_convert_worker", None)
        cancel = getattr(worker, "cancel", None)
        if callable(cancel):
            cancel()
            self._convert_status_label.setText("Cancelling conversion...")
            self._convert_cancel_btn.setEnabled(False)

    def _set_conversion_controls_enabled(self, enabled: bool) -> None:
        """Prevent overlapping SWC conversion/augmentation pipelines."""
        for name in (
            "_convert_dir_btn",
            "_convert_files_btn",
            "_convert_resolution_spin",
            "_convert_hemisphere_combo",
            "_convert_add_flatmap_cb",
            "_convert_lookup_dir_btn",
            "_convert_lookup_resolution_spin",
        ):
            control = getattr(self, name, None)
            if control is not None:
                control.setEnabled(bool(enabled))

    def _convert_from_directory(self) -> None:
        """Pick a directory of SWC files and convert to Parquet."""
        dialog_start = perf_counter()
        logger.debug("swc_conversion_ui_directory_dialog_start source_mode=directory")
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory of SWC Files"
        )
        if not directory:
            logger.debug(
                (
                    "swc_conversion_ui_directory_dialog_canceled "
                    "source_mode=directory elapsed_s=%.6f"
                ),
                perf_counter() - dialog_start,
            )
            return

        logger.debug(
            (
                "swc_conversion_ui_directory_dialog_ok "
                "source_mode=directory path=%s elapsed_s=%.6f"
            ),
            directory,
            perf_counter() - dialog_start,
        )
        self._prompt_output_and_convert(
            directory,
            recursive=True,
            source_mode="directory",
        )

    def _convert_from_files(self) -> None:
        """Pick individual SWC files and convert to Parquet."""
        dialog_start = perf_counter()
        logger.debug("swc_conversion_ui_files_dialog_start source_mode=files")
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select SWC Files",
            "",
            "SWC Files (*.swc);;All Files (*)",
        )
        if not filepaths:
            logger.debug(
                (
                    "swc_conversion_ui_files_dialog_canceled "
                    "source_mode=files elapsed_s=%.6f"
                ),
                perf_counter() - dialog_start,
            )
            return

        logger.debug(
            (
                "swc_conversion_ui_files_dialog_ok "
                "source_mode=files file_count=%d elapsed_s=%.6f files=%s"
            ),
            len(filepaths),
            perf_counter() - dialog_start,
            filepaths if len(filepaths) <= 50 else filepaths[:10] + filepaths[-10:],
        )
        self._prompt_output_and_convert(filepaths, source_mode="files")

    def _prompt_output_and_convert(
        self,
        swc_paths: str | list[str],
        *,
        recursive: bool = False,
        source_mode: str,
    ) -> None:
        """Ask for output path and start conversion."""
        output_dialog_start = perf_counter()
        logger.debug(
            ("swc_conversion_ui_output_dialog_start source_mode=%s recursive=%s"),
            source_mode,
            recursive,
        )
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Parquet File",
            "neurons.parquet",
            "Parquet Files (*.parquet)",
        )
        if not output_path:
            logger.debug(
                (
                    "swc_conversion_ui_output_dialog_canceled "
                    "source_mode=%s elapsed_s=%.6f"
                ),
                source_mode,
                perf_counter() - output_dialog_start,
            )
            return

        logger.debug(
            (
                "swc_conversion_ui_output_dialog_ok "
                "source_mode=%s output=%s elapsed_s=%.6f"
            ),
            source_mode,
            output_path,
            perf_counter() - output_dialog_start,
        )
        self._start_conversion(
            swc_paths,
            output_path,
            recursive=recursive,
            source_mode=source_mode,
        )

    def _conversion_cached_atlas_inputs(
        self,
        atlas_name: str,
        resolution: int,
    ) -> tuple[BrainGlobeAtlas | None, bool]:
        """Return loaded atlas reuse options for SWC conversion."""
        if self._atlas is None:
            logger.debug(
                (
                    "swc_conversion_ui_cached_atlas_unavailable "
                    "reason=no_loaded_atlas atlas=%s resolution=%s"
                ),
                atlas_name,
                resolution,
            )
            return None, False

        loaded_atlas_name = str(getattr(self._atlas, "atlas_name", "") or "")
        if loaded_atlas_name and loaded_atlas_name != atlas_name:
            logger.debug(
                (
                    "swc_conversion_ui_cached_atlas_unavailable "
                    "reason=atlas_mismatch requested=%s loaded=%s "
                    "resolution=%s"
                ),
                atlas_name,
                loaded_atlas_name,
                resolution,
            )
            return None, False

        use_cached_annotation = False
        atlas_resolution = None
        try:
            atlas_resolution = tuple(
                float(value)
                for value in np.asarray(self._atlas.resolution, dtype=float).tolist()
            )
            use_cached_annotation = bool(
                np.allclose(
                    np.asarray(atlas_resolution, dtype=float),
                    float(resolution),
                )
            )
        except Exception:
            logger.debug(
                "Failed to inspect loaded atlas resolution for conversion cache.",
                exc_info=True,
            )

        logger.debug(
            (
                "swc_conversion_ui_cached_atlas_available atlas=%s "
                "requested_resolution=%s atlas_resolution=%s "
                "use_cached_annotation=%s"
            ),
            loaded_atlas_name or atlas_name,
            resolution,
            atlas_resolution,
            use_cached_annotation,
        )
        return self._atlas, use_cached_annotation

    def _start_conversion(
        self,
        swc_paths: str | list[str],
        output_path: str,
        *,
        recursive: bool = False,
        source_mode: str,
    ) -> None:
        """Launch the background conversion worker."""
        from ..workers import ConvertWorker

        active_thread = getattr(self, "_convert_thread", None)
        is_running = getattr(active_thread, "isRunning", None)
        if active_thread is not None and (
            not callable(is_running) or bool(is_running())
        ):
            show_warning("An SWC conversion is already running.")
            return

        resolution = self._convert_resolution_spin.value()
        hemisphere = self._convert_hemisphere_combo.currentData()
        atlas_name = self._atlas_combo.currentText()
        add_flatmaps_control = getattr(self, "_convert_add_flatmap_cb", None)
        add_flatmaps = bool(
            add_flatmaps_control is not None and add_flatmaps_control.isChecked()
        )
        lookup_dir = getattr(self, "_convert_lookup_dir", None)
        lookup_resolution_control = getattr(
            self, "_convert_lookup_resolution_spin", None
        )
        raw_lookup_resolution = (
            int(lookup_resolution_control.value())
            if lookup_resolution_control is not None
            else 0
        )
        if add_flatmaps and lookup_dir is None:
            show_warning(
                "Choose a lookup directory before enabling bilateral "
                "flatmap/depth preprocessing."
            )
            return
        known_count = len(swc_paths) if isinstance(swc_paths, list) else None
        cached_atlas, use_cached_annotation = self._conversion_cached_atlas_inputs(
            atlas_name,
            resolution,
        )
        self._convert_source_mode = source_mode
        self._convert_ui_start_time = perf_counter()
        logger.debug(
            (
                "swc_conversion_ui_start source_mode=%s known_count=%s "
                "recursive=%s output=%s resolution=%s hemisphere=%s atlas=%s "
                "source=%s cached_atlas=%s cached_annotation=%s"
            ),
            source_mode,
            known_count if known_count is not None else "unknown",
            recursive,
            output_path,
            resolution,
            hemisphere,
            atlas_name,
            swc_paths
            if isinstance(swc_paths, str)
            else f"{known_count} explicit files",
            cached_atlas is not None,
            use_cached_annotation,
        )

        status = (
            f"Converting {known_count} SWC files..."
            if known_count is not None
            else "Searching selected directory for SWC files..."
        )
        if hemisphere is not None:
            if known_count is None:
                status = (
                    "Searching selected directory for SWC files "
                    f"(aligning to {str(hemisphere).title()})..."
                )
            else:
                status = (
                    f"Converting {known_count} SWC files "
                    f"(aligning to {str(hemisphere).title()})..."
                )

        self._convert_progress.setVisible(True)
        if known_count is None:
            self._convert_progress.setRange(0, 0)
        else:
            self._convert_progress.setRange(0, known_count)
        self._convert_progress.setValue(0)
        self._convert_status_label.setText(status)
        cancel_button = getattr(self, "_convert_cancel_btn", None)
        if cancel_button is not None:
            cancel_button.setEnabled(True)
        self._set_conversion_controls_enabled(False)

        thread = QThread()
        worker = ConvertWorker(
            swc_paths,
            output_path,
            resolution,
            hemisphere=hemisphere,
            atlas_name=atlas_name,
            recursive=recursive,
            source_mode=source_mode,
            cached_atlas=cached_atlas,
            use_cached_annotation=use_cached_annotation,
            flatmap_lookup_dir=lookup_dir if add_flatmaps else None,
            flatmap_lookup_resolution_um=(
                float(raw_lookup_resolution)
                if add_flatmaps and raw_lookup_resolution > 0
                else None
            ),
        )
        self._convert_thread = thread
        self._convert_worker = worker
        worker.moveToThread(thread)
        logger.debug(
            "swc_conversion_ui_worker_created source_mode=%s thread=%s worker=%s",
            source_mode,
            thread,
            worker,
        )

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_convert_progress)
        worker.finished.connect(self._on_convert_finished)
        worker.error.connect(self._on_convert_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._cleanup_convert_thread(thread, worker))
        logger.debug(
            "swc_conversion_ui_worker_signals_connected source_mode=%s",
            source_mode,
        )

        thread.start()
        logger.debug("swc_conversion_ui_thread_started source_mode=%s", source_mode)

    def _on_convert_progress(self, message: str, current: int, total: int) -> None:
        """Handle conversion progress updates."""
        logger.debug(
            (
                "swc_conversion_ui_progress source_mode=%s current=%d "
                "total=%d message=%s"
            ),
            self._convert_source_mode,
            current,
            total,
            message,
        )
        if total > 0:
            self._convert_progress.setRange(0, total)
            self._convert_progress.setValue(min(current, total))
        else:
            self._convert_progress.setRange(0, 0)
        self._convert_status_label.setText(message)

    def _on_convert_finished(self, output_path: str, summary: object) -> None:
        """Handle conversion completion."""
        elapsed_s = (
            perf_counter() - self._convert_ui_start_time
            if self._convert_ui_start_time is not None
            else -1.0
        )
        logger.debug(
            (
                "swc_conversion_ui_finished source_mode=%s output=%s "
                "elapsed_s=%.6f discovered=%s processed=%s failed=%s "
                "rows=%s flipped=%s"
            ),
            self._convert_source_mode,
            output_path,
            elapsed_s,
            getattr(summary, "discovered_files", "unknown"),
            getattr(summary, "processed_files", "unknown"),
            getattr(summary, "failed_files", "unknown"),
            getattr(summary, "rows_written", "unknown"),
            getattr(summary, "flipped_files", "unknown"),
        )
        self._convert_progress.setVisible(False)
        self._convert_progress.setRange(0, 1)
        self._convert_progress.setValue(0)
        cancel_button = getattr(self, "_convert_cancel_btn", None)
        if cancel_button is not None:
            cancel_button.setEnabled(False)
        summary_parts = [f"Converted {summary.processed_files} file(s)"]
        if summary.failed_files:
            summary_parts.append(f"skipped {summary.failed_files}")
        if summary.flipped_files:
            summary_parts.append(f"flipped {summary.flipped_files}")
        self._convert_status_label.setText(
            f"Done! {', '.join(summary_parts)} -> {Path(output_path).name}"
        )
        logger.info(f"SWC-to-Parquet conversion complete: {output_path}")

    def _on_convert_error(self, error_msg: str) -> None:
        """Handle conversion error."""
        elapsed_s = (
            perf_counter() - self._convert_ui_start_time
            if self._convert_ui_start_time is not None
            else -1.0
        )
        logger.debug(
            ("swc_conversion_ui_error source_mode=%s elapsed_s=%.6f error=%s"),
            self._convert_source_mode,
            elapsed_s,
            error_msg,
        )
        self._convert_progress.setVisible(False)
        self._convert_progress.setRange(0, 1)
        self._convert_progress.setValue(0)
        cancel_button = getattr(self, "_convert_cancel_btn", None)
        if cancel_button is not None:
            cancel_button.setEnabled(False)
        self._convert_status_label.setText(f"Error: {error_msg}")
        logger.error(f"SWC-to-Parquet conversion failed: {error_msg}")

    def _cleanup_convert_thread(self, thread: QThread, worker: object) -> None:
        """Release SWC conversion worker objects after the thread stops."""

        elapsed_s = (
            perf_counter() - self._convert_ui_start_time
            if self._convert_ui_start_time is not None
            else -1.0
        )
        logger.debug(
            (
                "swc_conversion_ui_cleanup source_mode=%s elapsed_s=%.6f "
                "thread_matches=%s worker_matches=%s"
            ),
            self._convert_source_mode,
            elapsed_s,
            self._convert_thread is thread,
            self._convert_worker is worker,
        )
        if self._convert_thread is thread:
            self._convert_thread = None
        if self._convert_worker is worker:
            self._convert_worker = None
        self._set_conversion_controls_enabled(True)
        self._convert_source_mode = None
        self._convert_ui_start_time = None
