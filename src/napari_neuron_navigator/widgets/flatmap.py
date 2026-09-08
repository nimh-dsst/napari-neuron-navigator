"""Qt widget for lookup-based isocortex flatmap projection."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import Callable

import numpy as np
import pandas as pd
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..analysis.flatmap_correlation import FlatmapVoxelCorrelationSource
from ..flatmap_export import export_projected_nodes_csv
from ..flatmap_heatmap import (
    ALLEN_LAYER_OUTPUT_PROJECTION,
    ALLEN_LAYER_OUTPUT_STACK,
    DEFAULT_FLATMAP_DEPTH_BIN_UM,
    DEFAULT_FLATMAP_Y_BINS,
    FLATMAP_PLANE_MODE_ALLEN_LAYER_PROJECTION,
    FLATMAP_PLANE_MODE_ALLEN_LAYERS,
    FLATMAP_PLANE_MODE_DEPTH,
    FLATMAP_PLANE_MODE_FLAT,
    FLATMAP_Y_BINS_TOOLTIP,
    MAX_FLATMAP_VECTOR_SEGMENTS,
    MAX_FLATMAP_Y_BINS,
    AllenLayerHeatmapVolumeResult,
    AllenLayerStackResult,
    AllenLayerStackSummary,
    FlatmapGroupedVolume,
    FlatmapLookupStats,
    FlatmapRenderResult,
    FlatmapRenderSummary,
    FlatmapSegmentVectors,
    build_allen_layer_cluster_volumes,
    build_allen_layer_file_id_volumes,
    build_allen_layer_render_from_projected_nodes,
    build_flatmap_cluster_volumes,
    build_flatmap_file_id_volumes,
    build_flatmap_render_data,
    build_flatmap_render_data_from_projected_nodes,
    build_flatmap_segment_vectors,
    compute_flatmap_lookup_stats,
    depth_plane_labels,
    rendered_plane_points,
)
from ..flatmap_labels import (
    FlatmapRegionLabelsResult,
    build_flatmap_region_label_volume,
)
from ..flatmap_loader import FLATMAP_STYLE_FILENAMES, load_flatmap_volume_set
from ..flatmap_parquet import (
    FLATMAP_V3_AUGMENTED_COLUMNS,
    FLATMAP_V3_STYLE_COLUMN_MAPPING,
    augment_neuron_parquet_with_flatmap,
    flatmap_invalid_code_to_reason,
    read_flatmap_parquet_transform_info,
)
from ..flatmap_projection import (
    COORDINATE_MODE_MICRONS,
    COORDINATE_MODE_VOXELS,
    DEFAULT_CCF_RESOLUTION_UM,
    FLATMAP_LOOKUP_DIRECT,
    FLATMAP_LOOKUP_MIRRORED,
    FLATMAP_LOOKUP_MIRRORED_DEPTH,
    FLATMAP_LOOKUP_UNMAPPED,
    FlatmapProjectionResult,
    ProjectionSummary,
    build_projected_segments,
    project_and_build_segments,
    summarize_projection,
)
from ..isocortex_layers import (
    ALLEN_ISOCORTEX_LAYER_LABELS,
    AllenIsocortexLayerMap,
    layer_map_from_atlas,
)
from ..region_appearance import RegionAppearanceStore, structure_catalog
from ..swc import NodeType

logger = logging.getLogger(__name__)


def _region_layer_base_visible(layer) -> bool:
    """Return the current napari layer visibility before region styling."""
    current = bool(getattr(layer, "visible", True))
    base = bool(getattr(layer, "_napari_neuron_navigator_region_base_visible", True))
    previous = getattr(layer, "_napari_neuron_navigator_region_applied_visible", None)
    if previous is not None and current != bool(previous):
        base = current
        setattr(layer, "_napari_neuron_navigator_region_base_visible", base)
    return base


def _set_region_layer_visible(layer, visible: bool) -> None:
    layer.visible = bool(visible)
    setattr(layer, "_napari_neuron_navigator_region_applied_visible", bool(visible))


_SOURCE_SELECTED = "selected"
_SOURCE_ALL = "all"
_RENDER_HEATMAP = "heatmap"
_RENDER_POINTS = "points"
_RENDER_ALLEN_LAYERS = "allen_layer_heatmap"
# The two depth-free renders: flatmap XY only, with the depth axis collapsed.
_RENDER_FLAT_HEATMAP = "flat_heatmap"
_RENDER_FLAT_VECTOR = "flat_vector"
_HEATMAP_COLOR_SINGLE = "single"
_HEATMAP_COLOR_INDIVIDUAL = "individual"
_HEATMAP_COLOR_CLUSTER = "cluster"
_FINE_PROJECTION_GAMMA = 0.2
_DEFAULT_HEATMAP_GAMMA = 1.0
# One neuron's heatmap is dominated by a few dense bins -- the soma and tight
# local arbor -- while its long-range projections leave one or two nodes per
# bin.  Against the full range those projections are nearly black, so a
# per-neuron layer opens with its upper limit at this fraction of its own
# maximum.  The slider still spans the full range, so the dense core remains
# one drag away.
_INDIVIDUAL_HEATMAP_CONTRAST_FRACTION = 0.1
_PROJECTION_SOURCE_PRECOMPUTED = "precomputed"
_PROJECTION_SOURCE_RECOMPUTE = "recompute"
_OLD_SHAPES_LAYER_NAME = "Isocortex Flatmap Traces"
_HEATMAP_LAYER_NAME = "Isocortex Flatmap Heatmap"
_GROUPED_HEATMAP_LAYER_PREFIX = f"{_HEATMAP_LAYER_NAME}: "
_ALLEN_LAYER_HEATMAP_LAYER_NAME = "Isocortex Flatmap Allen Layers"
_GROUPED_ALLEN_LAYER_PREFIX = f"{_ALLEN_LAYER_HEATMAP_LAYER_NAME}: "
_ALLEN_LAYER_PROJECTION_LAYER_NAME = "Isocortex Flatmap Allen Layer Projection"
_GROUPED_ALLEN_LAYER_PROJECTION_PREFIX = (
    f"{_ALLEN_LAYER_PROJECTION_LAYER_NAME}: "
)
_FLAT_HEATMAP_LAYER_NAME = "Isocortex Flatmap 2D Heatmap"
_GROUPED_FLAT_HEATMAP_PREFIX = f"{_FLAT_HEATMAP_LAYER_NAME}: "
_FLAT_VECTOR_LAYER_NAME = "Isocortex Flatmap 2D Vectors"
_POINTS_LAYER_NAME = "Isocortex Flatmap Points"
_SOMA_POINTS_LAYER_NAME = "Isocortex Flatmap Somas"
_REGION_LABELS_LAYER_NAME = "Flatmap Region Labels"
_REGION_SURFACES_LAYER_NAME = "Flatmap Region Surfaces"
_REGION_OUTLINES_LAYER_NAME = "Flatmap Region Outlines"
# The depth-free overlays keep the depth-grid prefixes so every prefix-based
# clear/retire path already covers them, but need their own names: assigning
# 2D data to a layer created with a depth axis is a rank mismatch.
_FLAT_REGION_LABELS_LAYER_NAME = f"{_REGION_LABELS_LAYER_NAME} 2D"
_FLAT_REGION_OUTLINES_LAYER_NAME = f"{_REGION_OUTLINES_LAYER_NAME} 2D"
_ALLEN_LAYER_PROJECTION_REGION_LABELS_LAYER_NAME = (
    f"{_REGION_LABELS_LAYER_NAME} Allen Projection"
)
# A collapsed perimeter traces one plane instead of 75 stacked ones, so it can
# afford a thinner stroke than the depth-grid outlines.
_FLAT_REGION_OUTLINE_EDGE_WIDTH = 0.6
_REGION_SURFACES_2D_TOOLTIP = (
    "Cached surfaces are 3D voxel shells. In 2D modes use Show Region Labels "
    "for a filled region."
)
_REGION_LABEL_ATLAS_DEFAULT = "allen_mouse_10um"
_REGION_LABEL_ATLAS_OPTIONS = (
    "allen_mouse_10um",
    "allen_mouse_25um",
    "allen_mouse_50um",
)
_FLATMAP_RENDER_LAYER_NAMES = {
    _OLD_SHAPES_LAYER_NAME,
    _HEATMAP_LAYER_NAME,
    _ALLEN_LAYER_HEATMAP_LAYER_NAME,
    _ALLEN_LAYER_PROJECTION_LAYER_NAME,
    _FLAT_HEATMAP_LAYER_NAME,
    _FLAT_VECTOR_LAYER_NAME,
    _POINTS_LAYER_NAME,
}
_DEFAULT_TRACE_COLOR = np.asarray([0.5, 0.5, 0.5, 1.0], dtype=float)

# Axis captions shown on the display viewer's dims sliders and axes overlay.
# The flatmap images are binned in index space, so these name the axes without
# claiming physical units or anatomical direction.
_FLATMAP_AXIS_LABEL_X = "Flatmap X"
_FLATMAP_AXIS_LABEL_Y = "Flatmap Y"
_ALLEN_LAYER_AXIS_LABEL = "Allen layer"
_DEPTH_AXIS_LABEL = "Depth bin"
_PLANE_TEXT_OVERLAY_FONT_SIZE = 12
_PLANE_TEXT_OVERLAY_POSITION = "top_left"
_FLATMAP_LAYER_SPACE_KEY = "napari_neuron_navigator_space"
_FLATMAP_LAYER_SPACE_VALUE = "flatmap"

_LOOKUP_FILES_PURPOSE_TEXT = (
    "These lookup files generate flatmap coordinates from CCF coordinates. "
    "They are needed only when the loaded Parquet has no flatmap columns, or "
    "to write them into one with Prepare Whole Parquet. A Parquet that already "
    "carries flatmap coordinates projects without them."
)


class _CacheCompatibilityUnavailable(RuntimeError):
    """Compatibility cannot be decided until required viewer state is loaded."""


class FlatmapProjectionWidget(QWidget):
    """Project loaded neuron rows into precomputed isocortex flatmap space."""

    def __init__(
        self,
        viewer,
        *,
        database_provider: Callable[[], object | None],
        selected_file_ids_provider: Callable[[], list[object]],
        table_file_ids_provider: Callable[[], list[object]],
        color_map_provider: Callable[[], dict[object, list[float]]],
        cluster_map_provider: Callable[[], dict[object, int | None]] | None = None,
        atlas_provider: Callable[[], object | None] | None = None,
        selected_region_ids_provider: Callable[[], list[int]] | None = None,
        selected_geometry_region_ids_provider: Callable[[], list[int]] | None = None,
        selected_parent_region_ids_provider: Callable[[], list[int]] | None = None,
        selected_region_acronyms_provider: Callable[[], list[str]] | None = None,
        selected_region_source_provider: Callable[[], str] | None = None,
        selected_region_scope_provider: Callable[[], str] | None = None,
        selected_region_error_provider: Callable[[], str | None] | None = None,
        region_appearance_provider: Callable[[], RegionAppearanceStore] | None = None,
        display_viewer_provider: Callable[..., object | None] | None = None,
        display_viewers_provider: Callable[[], Iterable[object]] | None = None,
        new_display_viewer_callback: Callable[[], object | None] | None = None,
        display_viewer_ready_callback: Callable[[object, object], None] | None = None,
        display_viewer_failed_callback: Callable[[object, str], None] | None = None,
        display_generation_provider: Callable[[], int] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._display_viewer_provider = display_viewer_provider
        self._display_viewers_provider = display_viewers_provider
        self._new_display_viewer_callback = new_display_viewer_callback
        self._display_viewer_ready_callback = display_viewer_ready_callback
        self._display_viewer_failed_callback = display_viewer_failed_callback
        self._display_generation_provider = display_generation_provider or (lambda: 0)
        self._last_display_viewer = None
        self._display_axis_annotation_state: dict | None = None
        self._display_axis_annotation_states: dict[int, dict] = {}
        self._flatmap_display_layer_event_viewer = None
        self._flatmap_display_layer_event_connections: list[tuple[object, object]] = []
        self._flatmap_heatmap_name_event_connections: dict[
            int, tuple[object, object, object]
        ] = {}
        self._flatmap_heatmap_viewers: dict[int, object] = {}
        self._flatmap_heatmap_selected_names: dict[int, set[str]] = {}
        self._flatmap_heatmap_list_viewer_id: int | None = None
        self._preferred_flatmap_heatmap_viewer_id: int | None = None
        self._refreshing_flatmap_heatmap_window_list = False
        self._flatmap_heatmap_refresh_pending = False
        self._flatmap_render_mode_change_pending = False
        self._database_provider = database_provider
        self._selected_file_ids_provider = selected_file_ids_provider
        self._table_file_ids_provider = table_file_ids_provider
        self._color_map_provider = color_map_provider
        self._cluster_map_provider = cluster_map_provider or (lambda: {})
        self._atlas_provider = atlas_provider or (lambda: None)
        self._selected_region_ids_provider = selected_region_ids_provider or (
            lambda: []
        )
        self._selected_geometry_region_ids_provider = (
            selected_geometry_region_ids_provider
            or selected_parent_region_ids_provider
            or self._selected_region_ids_provider
        )
        # Compatibility alias for callers that still use the parent-only name.
        self._selected_parent_region_ids_provider = (
            self._selected_geometry_region_ids_provider
        )
        self._selected_region_acronyms_provider = selected_region_acronyms_provider or (
            lambda: []
        )
        self._selected_region_source_provider = selected_region_source_provider or (
            lambda: "atlas_regions"
        )
        self._selected_region_scope_provider = selected_region_scope_provider or (
            lambda: "whole_parquet"
        )
        self._selected_region_error_provider = selected_region_error_provider or (
            lambda: None
        )
        self._region_appearance_provider = region_appearance_provider or (
            lambda: RegionAppearanceStore()
        )

        self._flatmap_path: Path | None = None
        self._depth_path: Path | None = None
        self._preprocess_lookup_dir: Path | None = None
        self._region_cache_dir: Path | None = None
        self._region_cache = None
        self._active_cache_profile = None
        self._pending_cache_profile_id: str | None = None
        self._cache_open_thread = None
        self._cache_open_worker = None
        self._cache_open_request_serial = 0
        self._cache_open_active_request_id: int | None = None
        self._cache_open_active_request: tuple[int, Path, str | None] | None = None
        self._pending_cache_open_request: tuple[int, Path, str | None] | None = None
        self._pending_validated_cache: tuple[Path, object, str | None] | None = None
        self._cache_open_shutting_down = False
        self._projection_layer = None
        self._soma_layer = None
        self._region_labels_layer = None
        self._region_surfaces_layers: list[object] = []
        self._region_outlines_layers: list[object] = []
        self._region_label_atlas_cache: dict[str, object] = {}
        self._region_label_atlas_load_thread = None
        self._region_label_atlas_load_worker = None
        self._region_label_request_display_generation: int | None = None
        self._pending_region_label_request = False
        self._precomputed_heatmap_thread = None
        self._precomputed_heatmap_worker = None
        self._precomputed_heatmap_display_generation: int | None = None
        self._last_projected_nodes: pd.DataFrame | None = None
        self._last_summary: ProjectionSummary | None = None
        self._last_render_summary: FlatmapRenderSummary | None = None
        self._last_render_mode: str | None = None
        self._last_flatmap_style: str | None = None
        self._last_coordinate_mode: str | None = None
        self._last_volume_shape: tuple[int, int, int] | None = None
        self._last_lookup_stats: FlatmapLookupStats | None = None
        self._last_input_file_ids: tuple[str, ...] = ()
        self._last_flatmap_path: str | None = None
        self._last_depth_path: str | None = None
        self._last_projection_source: str | None = None
        self._last_cache_dir: str | None = None
        self._last_cache_profile_id: str | None = None
        self._flatmap_correlation_source_changed_callback = None
        self._lookup_stats_cache_key: tuple[object, ...] | None = None
        self._lookup_stats_cache: FlatmapLookupStats | None = None
        self._allen_layer_map_cache_key: tuple[object, ...] | None = None
        self._allen_layer_map_cache: AllenIsocortexLayerMap | None = None

        self._setup_ui()
        self.destroyed.connect(self._on_cache_widget_destroyed)

    def _resolve_display_viewer(self, *, create: bool):
        """Return the viewer used for flatmap display layers."""
        provider = getattr(self, "_display_viewer_provider", None)
        if callable(provider):
            try:
                viewer = provider(create=create)
            except TypeError:
                if not create:
                    return getattr(self, "_last_display_viewer", None)
                viewer = provider()
            if viewer is not None:
                self._last_display_viewer = viewer
            return viewer
        return getattr(self, "_viewer", None)

    def _display_viewer(self):
        return self._resolve_display_viewer(create=True)

    def _current_display_viewer(self):
        return self._resolve_display_viewer(create=False)

    def _release_display_viewer(self, viewer) -> bool:
        """Forget layer handles when a flatmap display viewer closes."""
        self._clear_display_axis_annotations(viewer)
        self._disconnect_flatmap_display_layer_events(viewer)
        viewer_id = id(viewer)
        getattr(self, "_flatmap_heatmap_viewers", {}).pop(viewer_id, None)
        getattr(self, "_flatmap_heatmap_selected_names", {}).pop(viewer_id, None)
        if getattr(self, "_flatmap_heatmap_list_viewer_id", None) == viewer_id:
            self._flatmap_heatmap_list_viewer_id = None
        if getattr(self, "_preferred_flatmap_heatmap_viewer_id", None) == viewer_id:
            self._preferred_flatmap_heatmap_viewer_id = None

        released_active = getattr(self, "_last_display_viewer", None) is viewer
        if released_active:
            self._last_display_viewer = None
            self._clear_display_layer_handles()
        self._refresh_flatmap_heatmap_layer_list()
        if callable(getattr(self, "_display_viewer_provider", None)):
            self._update_flatmap_render_mode_change_pending()
        return released_active

    def _clear_display_layer_handles(self) -> None:
        """Forget layers controlled in the active flatmap viewer."""
        self._projection_layer = None
        self._soma_layer = None
        self._region_labels_layer = None
        self._region_surfaces_layers = []
        self._region_outlines_layers = []

    def _begin_new_display_viewer(self):
        """Activate an empty display viewer while preserving the current one."""
        callback = getattr(self, "_new_display_viewer_callback", None)
        if not callable(callback):
            return self._display_viewer()

        previous = self._current_display_viewer()
        viewer = callback()
        if viewer is None:
            raise RuntimeError("Could not create another flatmap window.")
        if viewer is previous:
            return viewer

        self._last_display_viewer = viewer
        self._preferred_flatmap_heatmap_viewer_id = id(viewer)
        self._clear_display_layer_handles()
        self._refresh_flatmap_heatmap_layer_list()
        return viewer

    def _notify_display_viewer_ready(self, layer) -> None:
        """Report that one display layer is configured and ready to show."""
        viewer = self._current_display_viewer()
        if viewer is None or not self._layer_is_in_viewer(layer, viewer=viewer):
            return
        callback = getattr(self, "_display_viewer_ready_callback", None)
        if callable(callback):
            try:
                callback(viewer, layer)
            except Exception:
                logger.debug(
                    "Failed to report that the flatmap display viewer is ready.",
                    exc_info=True,
                )
        self._refresh_flatmap_heatmap_layer_list()

    def _notify_display_viewer_failed(self, reason: str) -> None:
        """Report an unsuccessful first render so the main scene can recover."""
        viewer = self._current_display_viewer()
        if viewer is None:
            return
        callback = getattr(self, "_display_viewer_failed_callback", None)
        if not callable(callback):
            return
        try:
            callback(viewer, reason)
        except Exception:
            logger.debug(
                "Failed to report an unsuccessful flatmap display render.",
                exc_info=True,
            )

    def _display_generation(self) -> int:
        provider = getattr(self, "_display_generation_provider", None)
        try:
            return int(provider()) if callable(provider) else 0
        except Exception:
            logger.debug("Could not read flatmap viewer generation", exc_info=True)
            return 0

    def _display_generation_matches(self, expected: int | None) -> bool:
        return expected is None or self._display_generation() == int(expected)

    def _display_layers(self, *, create: bool = True):
        viewer = self._resolve_display_viewer(create=create)
        if viewer is None:
            return None
        return getattr(viewer, "layers", None)

    @staticmethod
    def _is_flatmap_heatmap_layer(layer) -> bool:
        """Return whether a layer is a gamma-adjustable flatmap heatmap."""
        metadata = getattr(layer, "metadata", {}) or {}
        return bool(
            hasattr(layer, "gamma")
            and metadata.get("flatmap_render_mode")
            in {_RENDER_HEATMAP, _RENDER_FLAT_HEATMAP, _RENDER_ALLEN_LAYERS}
        )

    def _available_display_viewers(self) -> list[object]:
        """Return live flatmap viewers available to appearance controls."""
        provider = getattr(self, "_display_viewers_provider", None)
        if callable(provider):
            candidates = provider() or ()
        else:
            current = self._current_display_viewer()
            candidates = (current,) if current is not None else ()

        viewers: list[object] = []
        viewer_ids: set[int] = set()
        for viewer in candidates:
            if viewer is None or id(viewer) in viewer_ids:
                continue
            viewers.append(viewer)
            viewer_ids.add(id(viewer))
        return viewers

    @staticmethod
    def _flatmap_display_viewer_label(viewer, index: int) -> str:
        """Return the visible name for one selectable flatmap viewer."""
        title = str(getattr(viewer, "title", "") or "").strip()
        return title or f"Flatmap Window {index}"

    def _refresh_flatmap_heatmap_window_list(self):
        """Refresh the appearance window selector and return its viewer."""
        combo = getattr(self, "_flatmap_heatmap_window_combo", None)
        if combo is None:
            return self._current_display_viewer()

        previous_id = combo.currentData()
        viewers = self._available_display_viewers()
        viewer_map = {id(viewer): viewer for viewer in viewers}
        self._flatmap_heatmap_viewers = viewer_map

        preferred_id = getattr(
            self,
            "_preferred_flatmap_heatmap_viewer_id",
            None,
        )
        current = self._current_display_viewer()
        current_id = id(current) if current is not None else None
        selected_id = next(
            (
                candidate_id
                for candidate_id in (preferred_id, previous_id, current_id)
                if candidate_id in viewer_map
            ),
            id(viewers[-1]) if viewers else None,
        )

        self._refreshing_flatmap_heatmap_window_list = True
        signals_were_blocked = combo.blockSignals(True)
        try:
            combo.clear()
            for index, viewer in enumerate(viewers, start=1):
                combo.addItem(
                    self._flatmap_display_viewer_label(viewer, index),
                    id(viewer),
                )
            selected_index = combo.findData(selected_id)
            if selected_index >= 0:
                combo.setCurrentIndex(selected_index)
            combo.setEnabled(bool(viewers))
        finally:
            combo.blockSignals(signals_were_blocked)
            self._refreshing_flatmap_heatmap_window_list = False

        if selected_id == preferred_id:
            self._preferred_flatmap_heatmap_viewer_id = None
        return viewer_map.get(selected_id)

    def _selected_flatmap_heatmap_viewer(self):
        """Return the flatmap viewer selected in Heatmap Appearance."""
        combo = getattr(self, "_flatmap_heatmap_window_combo", None)
        viewer_map = getattr(self, "_flatmap_heatmap_viewers", {})
        if combo is not None:
            viewer = viewer_map.get(combo.currentData())
            if viewer is not None:
                return viewer
        return self._current_display_viewer()

    def _flatmap_heatmap_layers(self, *, viewer=None) -> list[object]:
        """Return heatmap layers from the selected flatmap viewer."""
        if viewer is None:
            viewer = self._selected_flatmap_heatmap_viewer()
        layers = getattr(viewer, "layers", ()) if viewer is not None else ()
        return [layer for layer in layers if self._is_flatmap_heatmap_layer(layer)]

    def _selected_flatmap_heatmap_layers(self) -> list[object]:
        """Return flatmap heatmaps highlighted in the appearance section."""
        layer_list = getattr(self, "_flatmap_heatmap_layer_list", None)
        if layer_list is None:
            return []
        selected_names = {item.text() for item in layer_list.selectedItems()}
        if not selected_names:
            return []
        return [
            layer
            for layer in self._flatmap_heatmap_layers()
            if str(getattr(layer, "name", "")) in selected_names
        ]

    def _refresh_flatmap_heatmap_layer_list(self) -> None:
        """Refresh the Flatmap-tab list of gamma-adjustable heatmaps."""
        layer_list = getattr(self, "_flatmap_heatmap_layer_list", None)
        if layer_list is None:
            return

        previous_viewer_id = getattr(
            self,
            "_flatmap_heatmap_list_viewer_id",
            None,
        )
        previous = {item.text() for item in layer_list.selectedItems()}
        selections = getattr(self, "_flatmap_heatmap_selected_names", None)
        if selections is None:
            selections = {}
            self._flatmap_heatmap_selected_names = selections
        if previous_viewer_id is not None:
            selections[previous_viewer_id] = previous

        viewer = self._refresh_flatmap_heatmap_window_list()
        viewer_id = id(viewer) if viewer is not None else None
        layers = (
            self._flatmap_heatmap_layers(viewer=viewer) if viewer is not None else []
        )
        selected_names = selections.get(viewer_id, set())
        layer_list.clear()
        for layer in layers:
            layer_list.addItem(str(getattr(layer, "name", "<unnamed>")))
        for index in range(layer_list.count()):
            item = layer_list.item(index)
            if item.text() in selected_names:
                item.setSelected(True)
        self._flatmap_heatmap_list_viewer_id = viewer_id

        self._connect_flatmap_display_layer_events(viewer)
        status = getattr(self, "_flatmap_heatmap_gamma_status_label", None)
        if status is not None:
            if layers:
                status.setText(f"{len(layers)} flatmap heatmap layer(s) available.")
            else:
                status.setText("No rendered flatmap heatmaps are available.")
        self._update_flatmap_heatmap_gamma_controls()

    def _on_flatmap_heatmap_window_changed(self, _index: int) -> None:
        """Show heatmaps belonging to the selected flatmap window."""
        if getattr(self, "_refreshing_flatmap_heatmap_window_list", False):
            return
        self._refresh_flatmap_heatmap_layer_list()

    def _update_flatmap_heatmap_gamma_controls(self, *_args) -> None:
        """Enable flatmap gamma actions only when heatmaps are selected."""
        ready = bool(self._selected_flatmap_heatmap_layers())
        for attribute in (
            "_flatmap_enhance_fine_projections_btn",
            "_flatmap_reset_gamma_btn",
        ):
            button = getattr(self, attribute, None)
            if button is not None:
                button.setEnabled(ready)

    def _set_selected_flatmap_heatmap_gamma(
        self,
        gamma: float,
        *,
        action: str,
    ) -> None:
        """Set gamma on all heatmaps selected in the Flatmap tab."""
        selected_layers = self._selected_flatmap_heatmap_layers()
        status = getattr(self, "_flatmap_heatmap_gamma_status_label", None)
        if not selected_layers:
            if status is not None:
                status.setText("Select at least one flatmap heatmap layer.")
            return

        updated = 0
        failed_names: list[str] = []
        for layer in selected_layers:
            try:
                layer.gamma = float(gamma)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                failed_names.append(str(getattr(layer, "name", "<unnamed>")))
                logger.warning(
                    "Could not set gamma on flatmap heatmap layer '%s'.",
                    getattr(layer, "name", "<unnamed>"),
                    exc_info=True,
                )
            else:
                updated += 1

        message = (
            f"{action} on {updated} flatmap heatmap layer(s) "
            f"(gamma {float(gamma):.2f})."
        )
        if failed_names:
            message += " Could not update: " + ", ".join(failed_names[:3])
            if len(failed_names) > 3:
                message += f" and {len(failed_names) - 3} more"
            message += "."
        if status is not None:
            status.setText(message)

    def _enhance_selected_flatmap_heatmap_projections(self) -> None:
        """Brighten fine projections in selected flatmap heatmaps."""
        self._set_selected_flatmap_heatmap_gamma(
            _FINE_PROJECTION_GAMMA,
            action="Enhanced fine projections",
        )

    def _reset_selected_flatmap_heatmap_gamma(self) -> None:
        """Restore default gamma on selected flatmap heatmaps."""
        self._set_selected_flatmap_heatmap_gamma(
            _DEFAULT_HEATMAP_GAMMA,
            action="Reset gamma",
        )

    def _on_flatmap_heatmap_section_expanded(self, expanded: bool) -> None:
        """Refresh the heatmap list whenever its section is opened."""
        if expanded:
            self._refresh_flatmap_heatmap_layer_list()

    def _on_flatmap_display_layers_changed(self, _event=None) -> None:
        """Refresh gamma controls after the layer event finishes dispatching."""
        if getattr(self, "_flatmap_heatmap_refresh_pending", False):
            return
        self._flatmap_heatmap_refresh_pending = True
        self._queue_gui_callback(self._run_pending_flatmap_heatmap_refresh)

    def _run_pending_flatmap_heatmap_refresh(self) -> None:
        """Apply one coalesced heatmap refresh from the Qt event loop."""
        self._flatmap_heatmap_refresh_pending = False
        try:
            self._refresh_flatmap_heatmap_layer_list()
        except RuntimeError:
            logger.debug(
                "Skipped a flatmap heatmap refresh after its Qt widget closed.",
                exc_info=True,
            )

    def _connect_flatmap_display_layer_events(self, viewer) -> None:
        """Follow additions, removals, and renames in the display viewer."""
        if viewer is None:
            self._disconnect_flatmap_display_layer_events()
            return
        if getattr(self, "_flatmap_display_layer_event_viewer", None) is viewer:
            self._sync_flatmap_heatmap_name_event_connections(
                self._flatmap_heatmap_layers(viewer=viewer)
            )
            return

        self._disconnect_flatmap_display_layer_events()
        self._flatmap_display_layer_event_viewer = viewer
        connections: list[tuple[object, object]] = []
        events = getattr(getattr(viewer, "layers", None), "events", None)
        if events is not None:
            callback = self._on_flatmap_display_layers_changed
            for event_name in ("inserted", "removed", "reordered"):
                signal = getattr(events, event_name, None)
                connect = getattr(signal, "connect", None)
                if not callable(connect):
                    continue
                try:
                    connect(callback)
                except Exception:
                    logger.debug(
                        "Could not follow flatmap display-layer changes.",
                        exc_info=True,
                    )
                else:
                    connections.append((signal, callback))
        self._flatmap_display_layer_event_connections = connections
        self._sync_flatmap_heatmap_name_event_connections(
            self._flatmap_heatmap_layers(viewer=viewer)
        )

    def _sync_flatmap_heatmap_name_event_connections(
        self,
        layers: list[object],
    ) -> None:
        """Track heatmap renames so the selector never keeps stale names."""
        connections = getattr(
            self,
            "_flatmap_heatmap_name_event_connections",
            None,
        )
        if connections is None:
            connections = {}
            self._flatmap_heatmap_name_event_connections = connections

        active_ids = {id(layer) for layer in layers}
        for layer_id, connection in list(connections.items()):
            if layer_id not in active_ids:
                self._disconnect_flatmap_heatmap_name_event_connection(
                    layer_id,
                    connection,
                )

        for layer in layers:
            layer_id = id(layer)
            existing = connections.get(layer_id)
            if existing is not None and existing[0] is layer:
                continue
            if existing is not None:
                self._disconnect_flatmap_heatmap_name_event_connection(
                    layer_id,
                    existing,
                )
            signal = getattr(getattr(layer, "events", None), "name", None)
            connect = getattr(signal, "connect", None)
            if not callable(connect):
                continue
            callback = self._on_flatmap_display_layers_changed
            try:
                connect(callback)
            except Exception:
                logger.debug(
                    "Could not follow flatmap heatmap layer rename.",
                    exc_info=True,
                )
            else:
                connections[layer_id] = (layer, signal, callback)

    def _disconnect_flatmap_heatmap_name_event_connection(
        self,
        layer_id: int,
        connection: tuple[object, object, object],
    ) -> None:
        """Disconnect one tracked flatmap heatmap name signal."""
        _layer, signal, callback = connection
        disconnect = getattr(signal, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect(callback)
            except Exception:
                logger.debug(
                    "Could not disconnect flatmap heatmap name event.",
                    exc_info=True,
                )
        connections = getattr(
            self,
            "_flatmap_heatmap_name_event_connections",
            None,
        )
        if connections is not None:
            connections.pop(layer_id, None)

    def _disconnect_flatmap_display_layer_events(self, viewer=None) -> None:
        """Stop following a flatmap viewer that is closing or being replaced."""
        tracked_viewer = getattr(self, "_flatmap_display_layer_event_viewer", None)
        if viewer is not None and tracked_viewer is not viewer:
            return

        for signal, callback in getattr(
            self,
            "_flatmap_display_layer_event_connections",
            (),
        ):
            disconnect = getattr(signal, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect(callback)
                except Exception:
                    logger.debug(
                        "Could not disconnect flatmap display-layer event.",
                        exc_info=True,
                    )
        self._flatmap_display_layer_event_connections = []

        name_connections = getattr(
            self,
            "_flatmap_heatmap_name_event_connections",
            {},
        )
        for layer_id, connection in list(name_connections.items()):
            self._disconnect_flatmap_heatmap_name_event_connection(
                layer_id,
                connection,
            )
        self._flatmap_display_layer_event_viewer = None

    def _setup_ui(self) -> None:
        """Build the tab UI."""
        # Imported here because the sibling package's ``__init__`` pulls in every
        # widget module, which is more than this module needs at import time.
        from .collapsible_section import CollapsibleSection

        parent_layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        parent_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        self._lookup_files_section = CollapsibleSection("Flatmap Lookup Files")
        files_layout = self._lookup_files_section.content_layout()

        lookup_files_purpose_label = QLabel(_LOOKUP_FILES_PURPOSE_TEXT)
        lookup_files_purpose_label.setWordWrap(True)
        files_layout.addWidget(lookup_files_purpose_label)

        projection_source_row = QHBoxLayout()
        projection_source_row.addWidget(QLabel("Source:"))
        self._projection_source_combo = QComboBox()
        self._projection_source_combo.addItem(
            "Precomputed Parquet + Cache",
            _PROJECTION_SOURCE_PRECOMPUTED,
        )
        self._projection_source_combo.addItem(
            "Recompute from NRRDs",
            _PROJECTION_SOURCE_RECOMPUTE,
        )
        self._projection_source_combo.currentIndexChanged.connect(
            self._on_projection_source_changed
        )
        projection_source_row.addWidget(self._projection_source_combo)
        files_layout.addLayout(projection_source_row)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Style:"))
        self._style_combo = QComboBox()
        self._style_combo.addItem("Both hemispheres, shaped", "both_shaped")
        self._style_combo.addItem("Both hemispheres, square", "both_square")
        self._style_combo.addItem("Single hemisphere, shaped", "single_shaped")
        self._style_combo.addItem("Single hemisphere, square", "single_square")
        self._style_combo.currentIndexChanged.connect(self._on_flatmap_style_changed)
        style_row.addWidget(self._style_combo)
        files_layout.addLayout(style_row)

        self._expected_filename_label = QLabel("")
        self._expected_filename_label.setWordWrap(True)
        files_layout.addWidget(self._expected_filename_label)

        flatmap_row = QHBoxLayout()
        self._flatmap_path_label = QLabel("No flatmap selected")
        self._flatmap_path_label.setWordWrap(True)
        flatmap_row.addWidget(self._flatmap_path_label, stretch=1)
        flatmap_btn = QPushButton("Choose Flatmap...")
        flatmap_btn.clicked.connect(self._choose_flatmap_path)
        flatmap_row.addWidget(flatmap_btn)
        files_layout.addLayout(flatmap_row)

        depth_row = QHBoxLayout()
        self._depth_path_label = QLabel("No depth selected")
        self._depth_path_label.setWordWrap(True)
        depth_row.addWidget(self._depth_path_label, stretch=1)
        depth_btn = QPushButton("Choose Depth...")
        depth_btn.clicked.connect(self._choose_depth_path)
        depth_row.addWidget(depth_btn)
        files_layout.addLayout(depth_row)

        lookup_dir_row = QHBoxLayout()
        self._lookup_dir_label = QLabel("No preprocessing lookup directory selected")
        self._lookup_dir_label.setWordWrap(True)
        lookup_dir_row.addWidget(self._lookup_dir_label, stretch=1)
        lookup_dir_btn = QPushButton("Lookup directory...")
        lookup_dir_btn.clicked.connect(self._choose_preprocess_lookup_dir)
        lookup_dir_row.addWidget(lookup_dir_btn)
        files_layout.addLayout(lookup_dir_row)
        lookup_resolution_row = QHBoxLayout()
        lookup_resolution_row.addWidget(QLabel("Lookup resolution:"))
        self._lookup_resolution_spin = QSpinBox()
        self._lookup_resolution_spin.setRange(0, 100)
        self._lookup_resolution_spin.setSpecialValueText("From NRRD header")
        self._lookup_resolution_spin.setSuffix(" um")
        lookup_resolution_row.addWidget(self._lookup_resolution_spin)
        files_layout.addLayout(lookup_resolution_row)
        layout.addWidget(self._lookup_files_section)

        cache_group = QGroupBox("Flatmap Region Cache")
        cache_layout = QVBoxLayout(cache_group)
        cache_dir_row = QHBoxLayout()
        self._cache_dir_label = QLabel("No cache directory selected")
        self._cache_dir_label.setWordWrap(True)
        cache_dir_row.addWidget(self._cache_dir_label, stretch=1)
        self._cache_dir_btn = QPushButton("Choose Cache Directory...")
        self._cache_dir_btn.clicked.connect(self._choose_cache_directory)
        cache_dir_row.addWidget(self._cache_dir_btn)
        cache_layout.addLayout(cache_dir_row)

        cache_profile_row = QHBoxLayout()
        cache_profile_row.addWidget(QLabel("Profile:"))
        self._cache_profile_combo = QComboBox()
        self._cache_profile_combo.currentIndexChanged.connect(
            self._on_cache_profile_changed
        )
        cache_profile_row.addWidget(self._cache_profile_combo, stretch=1)
        cache_layout.addLayout(cache_profile_row)

        cache_grid_row = QHBoxLayout()
        cache_build_y_bins_label = QLabel("New profile Y bins:")
        cache_build_y_bins_label.setToolTip(FLATMAP_Y_BINS_TOOLTIP)
        cache_grid_row.addWidget(cache_build_y_bins_label)
        self._cache_build_y_bins_spin = QSpinBox()
        self._cache_build_y_bins_spin.setRange(1, MAX_FLATMAP_Y_BINS)
        self._cache_build_y_bins_spin.setSingleStep(16)
        self._cache_build_y_bins_spin.setValue(DEFAULT_FLATMAP_Y_BINS)
        self._cache_build_y_bins_spin.setToolTip(FLATMAP_Y_BINS_TOOLTIP)
        cache_grid_row.addWidget(self._cache_build_y_bins_spin)
        cache_grid_row.addWidget(QLabel("Depth bin:"))
        self._cache_build_depth_bin_spin = QDoubleSpinBox()
        self._cache_build_depth_bin_spin.setRange(0.001, 1000.0)
        self._cache_build_depth_bin_spin.setDecimals(3)
        self._cache_build_depth_bin_spin.setSingleStep(5.0)
        self._cache_build_depth_bin_spin.setSuffix(" um")
        self._cache_build_depth_bin_spin.setValue(float(DEFAULT_FLATMAP_DEPTH_BIN_UM))
        cache_grid_row.addWidget(self._cache_build_depth_bin_spin)
        cache_layout.addLayout(cache_grid_row)

        cache_build_row = QHBoxLayout()
        self._build_cache_btn = QPushButton("Build Cache Profile...")
        self._build_cache_btn.clicked.connect(self._build_cache_profile)
        cache_build_row.addWidget(self._build_cache_btn)
        self._cancel_cache_btn = QPushButton("Cancel")
        self._cancel_cache_btn.setEnabled(False)
        self._cancel_cache_btn.clicked.connect(self._cancel_cache_build)
        cache_build_row.addWidget(self._cancel_cache_btn)
        self._cache_status_label = QLabel("No cache profile active.")
        self._cache_status_label.setWordWrap(True)
        cache_build_row.addWidget(self._cache_status_label, stretch=1)
        cache_layout.addLayout(cache_build_row)
        layout.addWidget(cache_group)

        options_group = QGroupBox("Projection Options")
        options_layout = QVBoxLayout(options_group)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Input:"))
        self._source_combo = QComboBox()
        self._source_combo.addItem(
            "Selected table rows, otherwise all", _SOURCE_SELECTED
        )
        self._source_combo.addItem("All table rows", _SOURCE_ALL)
        source_row.addWidget(self._source_combo)
        options_layout.addLayout(source_row)

        coordinate_row = QHBoxLayout()
        coordinate_row.addWidget(QLabel("Coordinates:"))
        self._coordinate_mode_combo = QComboBox()
        self._coordinate_mode_combo.addItem(
            "CCF microns from NRRD header",
            COORDINATE_MODE_MICRONS,
        )
        self._coordinate_mode_combo.addItem(
            "10 um voxel indices", COORDINATE_MODE_VOXELS
        )
        coordinate_row.addWidget(self._coordinate_mode_combo)
        options_layout.addLayout(coordinate_row)

        render_row = QHBoxLayout()
        render_row.addWidget(QLabel("Render:"))
        self._render_mode_combo = QComboBox()
        self._render_mode_combo.addItem("3D Heatmap", _RENDER_HEATMAP)
        self._render_mode_combo.addItem("3D Points", _RENDER_POINTS)
        self._render_mode_combo.addItem("2D Heatmap", _RENDER_FLAT_HEATMAP)
        self._render_mode_combo.addItem("2D Vector", _RENDER_FLAT_VECTOR)
        self._render_mode_combo.addItem(
            "Allen Layer Heatmap",
            _RENDER_ALLEN_LAYERS,
        )
        self._render_mode_combo.setToolTip(
            "2D Heatmap and 2D Vector collapse the depth axis into a single "
            "flatmap image; Exclude depth -1 nodes still applies. 2D Vector "
            "draws one line per parent-child edge and is limited to "
            f"{MAX_FLATMAP_VECTOR_SEGMENTS:,} segments."
        )
        self._render_mode_combo.currentIndexChanged.connect(
            self._on_render_mode_changed
        )
        render_row.addWidget(self._render_mode_combo)
        render_row.addWidget(QLabel("Heatmap colors:"))
        self._heatmap_color_mode_combo = QComboBox()
        self._heatmap_color_mode_combo.addItem("Single color", _HEATMAP_COLOR_SINGLE)
        self._heatmap_color_mode_combo.addItem(
            "Individual neurons",
            _HEATMAP_COLOR_INDIVIDUAL,
        )
        self._heatmap_color_mode_combo.addItem("Cluster", _HEATMAP_COLOR_CLUSTER)
        render_row.addWidget(self._heatmap_color_mode_combo)
        options_layout.addLayout(render_row)

        self._allen_layer_options_group = QGroupBox("Allen Layer Options")
        allen_options_layout = QVBoxLayout(self._allen_layer_options_group)
        self._allen_layer_checkboxes: list[QCheckBox] = []
        for row_start in range(0, len(ALLEN_ISOCORTEX_LAYER_LABELS), 3):
            layer_row = QHBoxLayout()
            if row_start == 0:
                layer_row.addWidget(QLabel("Layers:"))
            else:
                layer_row.addWidget(QLabel(""))
            for layer_index, layer_label in enumerate(
                ALLEN_ISOCORTEX_LAYER_LABELS[row_start : row_start + 3],
                start=row_start,
            ):
                checkbox = QCheckBox(layer_label)
                checkbox.setChecked(True)
                checkbox.setProperty("allen_layer_index", layer_index)
                checkbox.toggled.connect(self._on_allen_layer_options_changed)
                self._allen_layer_checkboxes.append(checkbox)
                layer_row.addWidget(checkbox)
            layer_row.addStretch()
            allen_options_layout.addLayout(layer_row)

        layer_actions_row = QHBoxLayout()
        self._select_all_allen_layers_btn = QPushButton("Select All")
        self._select_all_allen_layers_btn.clicked.connect(
            lambda _checked=False: self._set_allen_layer_selection(True)
        )
        layer_actions_row.addWidget(self._select_all_allen_layers_btn)
        self._clear_allen_layers_btn = QPushButton("Clear")
        self._clear_allen_layers_btn.clicked.connect(
            lambda _checked=False: self._set_allen_layer_selection(False)
        )
        layer_actions_row.addWidget(self._clear_allen_layers_btn)
        layer_actions_row.addWidget(QLabel("View:"))
        self._allen_layer_output_combo = QComboBox()
        self._allen_layer_output_combo.addItem(
            "2D stack",
            ALLEN_LAYER_OUTPUT_STACK,
        )
        self._allen_layer_output_combo.addItem(
            "Single 2D projection",
            ALLEN_LAYER_OUTPUT_PROJECTION,
        )
        self._allen_layer_output_combo.currentIndexChanged.connect(
            self._on_allen_layer_options_changed
        )
        layer_actions_row.addWidget(self._allen_layer_output_combo)
        allen_options_layout.addLayout(layer_actions_row)
        options_layout.addWidget(self._allen_layer_options_group)

        y_bins_row = QHBoxLayout()
        y_bins_label = QLabel("Y bins:")
        y_bins_label.setToolTip(FLATMAP_Y_BINS_TOOLTIP)
        y_bins_row.addWidget(y_bins_label)
        self._y_bins_spin = QSpinBox()
        self._y_bins_spin.setRange(1, MAX_FLATMAP_Y_BINS)
        self._y_bins_spin.setSingleStep(16)
        self._y_bins_spin.setValue(DEFAULT_FLATMAP_Y_BINS)
        self._y_bins_spin.setToolTip(FLATMAP_Y_BINS_TOOLTIP)
        y_bins_row.addWidget(self._y_bins_spin)
        options_layout.addLayout(y_bins_row)

        depth_bin_row = QHBoxLayout()
        depth_bin_row.addWidget(QLabel("Depth bin:"))
        self._depth_bin_spin = QDoubleSpinBox()
        self._depth_bin_spin.setRange(0.001, 1000.0)
        self._depth_bin_spin.setDecimals(3)
        self._depth_bin_spin.setSingleStep(5.0)
        self._depth_bin_spin.setSuffix(" um")
        self._depth_bin_spin.setValue(float(DEFAULT_FLATMAP_DEPTH_BIN_UM))
        depth_bin_row.addWidget(self._depth_bin_spin)
        options_layout.addLayout(depth_bin_row)

        self._negative_one_sentinel_cb = QCheckBox("Treat flatmap (-1, -1) as invalid")
        self._negative_one_sentinel_cb.setChecked(True)
        options_layout.addWidget(self._negative_one_sentinel_cb)
        self._zero_sentinel_cb = QCheckBox("Treat flatmap (0, 0) as invalid")
        options_layout.addWidget(self._zero_sentinel_cb)
        self._exclude_depth_minus_one_cb = QCheckBox("Exclude depth -1 nodes")
        self._exclude_depth_minus_one_cb.setChecked(True)
        options_layout.addWidget(self._exclude_depth_minus_one_cb)
        layout.addWidget(options_group)

        actions_row = QHBoxLayout()
        self._project_btn = QPushButton("Project to Flatmap")
        self._project_btn.clicked.connect(self._project)
        actions_row.addWidget(self._project_btn)
        self._new_window_btn = QPushButton("Project in New Window")
        self._new_window_btn.setToolTip(
            "Keep the current flatmap window open and render this projection "
            "in another window for side-by-side comparison."
        )
        self._new_window_btn.clicked.connect(self._project_in_new_window)
        actions_row.addWidget(self._new_window_btn)
        self._add_soma_btn = QPushButton("Add Soma")
        self._add_soma_btn.setToolTip(
            "Project only soma nodes into flatmap + depth space as a "
            "separate point layer."
        )
        self._add_soma_btn.clicked.connect(self._add_soma)
        actions_row.addWidget(self._add_soma_btn)
        self._export_btn = QPushButton("Export CSV...")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        actions_row.addWidget(self._export_btn)
        self._augment_parquet_btn = QPushButton("Prepare Whole Parquet...")
        self._augment_parquet_btn.clicked.connect(self._augment_parquet)
        actions_row.addWidget(self._augment_parquet_btn)
        self._cancel_augment_btn = QPushButton("Cancel Preparation")
        self._cancel_augment_btn.setEnabled(False)
        self._cancel_augment_btn.clicked.connect(self._cancel_parquet_preparation)
        actions_row.addWidget(self._cancel_augment_btn)
        layout.addLayout(actions_row)

        self._projection_progress_bar = QProgressBar()
        self._projection_progress_bar.setRange(0, 1)
        self._projection_progress_bar.setValue(0)
        self._projection_progress_bar.setVisible(False)
        layout.addWidget(self._projection_progress_bar)

        self._flatmap_heatmap_appearance_section = CollapsibleSection(
            "Heatmap Appearance",
            expanded=False,
        )
        appearance_layout = self._flatmap_heatmap_appearance_section.content_layout()
        appearance_window_row = QHBoxLayout()
        appearance_window_row.addWidget(QLabel("Window:"))
        self._flatmap_heatmap_window_combo = QComboBox()
        self._flatmap_heatmap_window_combo.setToolTip(
            "Choose which open flatmap window supplies the heatmap layers below."
        )
        self._flatmap_heatmap_window_combo.currentIndexChanged.connect(
            self._on_flatmap_heatmap_window_changed
        )
        appearance_window_row.addWidget(self._flatmap_heatmap_window_combo, 1)
        appearance_layout.addLayout(appearance_window_row)

        appearance_hint = QLabel(
            "Select one or more rendered flatmap heatmaps to adjust together."
        )
        appearance_hint.setWordWrap(True)
        appearance_layout.addWidget(appearance_hint)

        self._flatmap_heatmap_layer_list = QListWidget()
        self._flatmap_heatmap_layer_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self._flatmap_heatmap_layer_list.itemSelectionChanged.connect(
            self._update_flatmap_heatmap_gamma_controls
        )
        appearance_layout.addWidget(self._flatmap_heatmap_layer_list)

        gamma_actions = QHBoxLayout()
        self._flatmap_enhance_fine_projections_btn = QPushButton(
            "Enhance Fine Projections"
        )
        self._flatmap_enhance_fine_projections_btn.setToolTip(
            "Set gamma to 0.20 on each selected flatmap heatmap to brighten "
            "low-intensity projections."
        )
        self._flatmap_enhance_fine_projections_btn.clicked.connect(
            self._enhance_selected_flatmap_heatmap_projections
        )
        gamma_actions.addWidget(self._flatmap_enhance_fine_projections_btn)

        self._flatmap_reset_gamma_btn = QPushButton("Reset Gamma")
        self._flatmap_reset_gamma_btn.setToolTip(
            "Restore gamma to 1.00 on each selected flatmap heatmap."
        )
        self._flatmap_reset_gamma_btn.clicked.connect(
            self._reset_selected_flatmap_heatmap_gamma
        )
        gamma_actions.addWidget(self._flatmap_reset_gamma_btn)
        appearance_layout.addLayout(gamma_actions)

        self._flatmap_heatmap_gamma_status_label = QLabel(
            "No rendered flatmap heatmaps are available."
        )
        self._flatmap_heatmap_gamma_status_label.setWordWrap(True)
        appearance_layout.addWidget(self._flatmap_heatmap_gamma_status_label)
        self._flatmap_heatmap_appearance_section.expanded_changed.connect(
            self._on_flatmap_heatmap_section_expanded
        )
        layout.addWidget(self._flatmap_heatmap_appearance_section)

        labels_group = QGroupBox("Cached Regions")
        labels_layout = QVBoxLayout(labels_group)
        atlas_row = QHBoxLayout()
        atlas_row.addWidget(QLabel("Atlas:"))
        self._region_label_atlas_combo = QComboBox()
        for atlas_name in _REGION_LABEL_ATLAS_OPTIONS:
            self._region_label_atlas_combo.addItem(atlas_name, atlas_name)
        self._region_label_atlas_combo.setCurrentText(_REGION_LABEL_ATLAS_DEFAULT)
        atlas_row.addWidget(self._region_label_atlas_combo)
        labels_layout.addLayout(atlas_row)
        labels_actions_row = QHBoxLayout()
        self._region_labels_btn = QPushButton("Show Region Labels")
        self._region_labels_btn.clicked.connect(self._create_region_labels)
        labels_actions_row.addWidget(self._region_labels_btn)
        self._clear_region_labels_btn = QPushButton("Clear Region Labels")
        self._clear_region_labels_btn.clicked.connect(self._clear_region_labels)
        labels_actions_row.addWidget(self._clear_region_labels_btn)
        labels_layout.addLayout(labels_actions_row)

        geometry_actions_row = QHBoxLayout()
        self._region_surfaces_btn = QPushButton("Show Region Surfaces")
        self._region_surfaces_btn.clicked.connect(self._create_region_surfaces)
        geometry_actions_row.addWidget(self._region_surfaces_btn)
        self._region_outlines_btn = QPushButton("Show Region Outlines")
        self._region_outlines_btn.clicked.connect(self._create_region_outlines)
        geometry_actions_row.addWidget(self._region_outlines_btn)
        self._clear_region_geometry_btn = QPushButton("Clear Geometry")
        self._clear_region_geometry_btn.clicked.connect(self._clear_region_geometry)
        geometry_actions_row.addWidget(self._clear_region_geometry_btn)
        labels_layout.addLayout(geometry_actions_row)
        self._region_labels_status_label = QLabel("No flatmap region labels created.")
        self._region_labels_status_label.setWordWrap(True)
        labels_layout.addWidget(self._region_labels_status_label)
        layout.addWidget(labels_group)

        summary_group = QGroupBox("Projection Summary")
        summary_layout = QVBoxLayout(summary_group)
        self._summary_label = QLabel("No projection run yet.")
        self._summary_label.setWordWrap(True)
        summary_layout.addWidget(self._summary_label)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        summary_layout.addWidget(self._status_label)
        layout.addWidget(summary_group)

        warning = QLabel(
            "Flatmap coordinates are for visualization and indexing only. "
            "Keep neurite length, branch geometry, and 3D distance calculations "
            "in the original CCF coordinate space."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        layout.addStretch()

        self._update_expected_filename_label()
        self._update_render_mode_controls()
        self._update_cached_region_controls()
        self._update_lookup_files_section()
        viewer = self._current_display_viewer()
        if viewer is not None:
            self._connect_flatmap_display_layer_events(viewer)
        self._refresh_flatmap_heatmap_layer_list()

    def set_flatmap_path(self, path: str | Path | None) -> None:
        """Set the flatmap path, primarily for tests and scripted use."""
        self._flatmap_path = Path(path) if path else None
        text = str(self._flatmap_path) if self._flatmap_path else "No flatmap selected"
        self._flatmap_path_label.setText(text)
        self._notify_flatmap_correlation_source_changed()

    def set_depth_path(self, path: str | Path | None) -> None:
        """Set the depth path, primarily for tests and scripted use."""
        self._depth_path = Path(path) if path else None
        text = str(self._depth_path) if self._depth_path else "No depth selected"
        self._depth_path_label.setText(text)
        self._notify_flatmap_correlation_source_changed()

    def set_flatmap_correlation_source_changed_callback(self, callback) -> None:
        """Set a callback invoked when the latest flatmap clustering source changes."""
        self._flatmap_correlation_source_changed_callback = callback

    def _notify_flatmap_correlation_source_changed(self) -> None:
        callback = getattr(
            self,
            "_flatmap_correlation_source_changed_callback",
            None,
        )
        if callable(callback):
            callback()

    def _current_projection_source(self) -> str:
        """Return the explicit precomputed/recompute selection.

        Widgets constructed directly by legacy tests do not have the selector;
        those retain the historical auto-detection behavior.
        """
        combo = getattr(self, "_projection_source_combo", None)
        current_data = getattr(combo, "currentData", None)
        if not callable(current_data):
            return "legacy_auto"
        value = current_data()
        if value == _PROJECTION_SOURCE_RECOMPUTE:
            return _PROJECTION_SOURCE_RECOMPUTE
        return _PROJECTION_SOURCE_PRECOMPUTED

    def _on_projection_source_changed(self) -> None:
        source = self._current_projection_source()
        self._invalidate_flatmap_grid_layers()
        self._update_style_choices_for_source(source)
        if source == _PROJECTION_SOURCE_RECOMPUTE:
            message = "NRRD recomputation is selected explicitly."
            self._set_cache_grid_locked(False)
        else:
            message = "Viewing will use precomputed Parquet/cache data only."
            if getattr(self, "_active_cache_profile", None) is not None:
                self._on_cache_profile_changed()
        self._update_cached_region_controls()
        status = getattr(self, "_status_label", None)
        if status is not None:
            status.setText(message)
        self._notify_flatmap_correlation_source_changed()

    def _update_style_choices_for_source(self, source: str) -> None:
        combo = getattr(self, "_style_combo", None)
        count = getattr(combo, "count", None)
        item_data = getattr(combo, "itemData", None)
        model_getter = getattr(combo, "model", None)
        if not all(callable(value) for value in (count, item_data, model_getter)):
            return
        bilateral = {"both_shaped", "both_square"}
        model = model_getter()
        for index in range(count()):
            item = model.item(index)
            if item is not None:
                item.setEnabled(
                    source == _PROJECTION_SOURCE_RECOMPUTE
                    or str(item_data(index)) in bilateral
                )
        if (
            source == _PROJECTION_SOURCE_PRECOMPUTED
            and self._current_style_key() not in bilateral
        ):
            for index in range(count()):
                if str(item_data(index)) == "both_shaped":
                    combo.setCurrentIndex(index)
                    break

    def _on_flatmap_style_changed(self) -> None:
        self._invalidate_flatmap_grid_layers()
        self._update_expected_filename_label()
        self._refresh_cache_profiles()
        self._notify_flatmap_correlation_source_changed()

    def _invalidate_flatmap_grid_layers(self) -> None:
        """Remove render state that belongs to a previous style/cache grid."""
        self._remove_projection_layer(create=False)
        self._remove_soma_layer()
        self._clear_named_region_layers(_REGION_LABELS_LAYER_NAME)
        self._clear_region_surface_layers()
        self._clear_region_outline_layers()
        self._region_labels_layer = None
        self._reset_flatmap_render_state()

    def _reset_flatmap_render_state(self) -> None:
        """Forget the latest projection without performing any layer mutation."""
        self._last_projected_nodes = None
        self._last_summary = None
        self._last_render_summary = None
        self._last_render_mode = None
        self._last_flatmap_style = None
        self._last_coordinate_mode = None
        self._last_volume_shape = None
        self._last_lookup_stats = None
        self._last_input_file_ids = ()
        self._last_flatmap_path = None
        self._last_depth_path = None
        self._last_projection_source = None
        self._last_cache_dir = None
        self._last_cache_profile_id = None
        export_button = getattr(self, "_export_btn", None)
        if export_button is not None:
            export_button.setEnabled(False)

    def _current_flatmap_render_layers(self) -> list[object]:
        layers = self._display_layers(create=False)
        if layers is None:
            return []
        return [
            layer
            for layer in list(layers)
            if self._is_flatmap_render_layer_name(getattr(layer, "name", None))
        ]

    def _current_cached_region_layers(self) -> list[object]:
        layers = self._display_layers(create=False)
        if layers is None:
            return []
        prefixes = (
            _REGION_LABELS_LAYER_NAME,
            _REGION_SURFACES_LAYER_NAME,
            _REGION_OUTLINES_LAYER_NAME,
        )
        return [
            layer
            for layer in list(layers)
            if str(getattr(layer, "name", "")).startswith(prefixes)
        ]

    def _defer_cached_region_layer_removal(
        self,
        *,
        keep_profile_id: str | None = None,
        keep_style: str | None = None,
    ) -> None:
        """Defer removal of overlays that do not match the retained cache grid."""
        targets = []
        for layer in self._current_cached_region_layers():
            metadata = getattr(layer, "metadata", None)
            layer_profile_id = (
                str(metadata.get("cache_profile_id", ""))
                if isinstance(metadata, dict)
                else ""
            )
            layer_style = (
                str(metadata.get("flatmap_style", ""))
                if isinstance(metadata, dict)
                else ""
            )
            layer_plane_mode = (
                str(metadata.get("flatmap_plane_mode", ""))
                if isinstance(metadata, dict)
                else ""
            )
            if (
                not keep_profile_id
                or layer_profile_id != keep_profile_id
                or (keep_style is not None and layer_style != keep_style)
                # A depth-grid overlay never belongs beside a collapsed render.
                or (layer_plane_mode and layer_plane_mode != self._current_plane_mode())
            ):
                targets.append(layer)
        self._hide_and_queue_layer_removal(targets)
        if any(layer is self._region_labels_layer for layer in targets):
            self._region_labels_layer = None
        self._region_surfaces_layers = [
            layer
            for layer in self._region_surfaces_layers
            if not any(layer is target for target in targets)
        ]
        self._region_outlines_layers = [
            layer
            for layer in self._region_outlines_layers
            if not any(layer is target for target in targets)
        ]

    def _defer_flatmap_grid_layer_removal(self) -> None:
        """Retire a stale grid without deleting live GPU resources inline."""
        soma_layer = self._cached_soma_layer()
        targets = (
            self._current_flatmap_render_layers() + self._current_cached_region_layers()
        )
        if soma_layer is not None:
            # Soma bin coordinates belong to the retired grid too.
            targets.append(soma_layer)
        self._hide_and_queue_layer_removal(targets)
        self._projection_layer = None
        self._soma_layer = None
        self._region_labels_layer = None
        self._region_surfaces_layers = []
        self._region_outlines_layers = []
        self._reset_flatmap_render_state()

    def _hide_and_queue_layer_removal(self, targets: list[object]) -> None:
        if not targets:
            return
        unique_targets: list[object] = []
        for target in targets:
            if not any(existing is target for existing in unique_targets):
                unique_targets.append(target)
        for layer in unique_targets:
            try:
                layer.visible = False
            except Exception:
                logger.debug(
                    "Failed to hide stale flatmap layer %s.",
                    getattr(layer, "name", "<unnamed>"),
                    exc_info=True,
                )

        layers = self._display_layers(create=False)

        def remove_hidden_layers() -> None:
            if layers is None:
                return
            for layer in unique_targets:
                try:
                    if any(existing is layer for existing in layers):
                        layers.remove(layer)
                except (RuntimeError, ValueError):
                    logger.debug(
                        "Stale flatmap layer was already removed: %s",
                        getattr(layer, "name", "<unnamed>"),
                        exc_info=True,
                    )

        self._queue_gui_callback(remove_hidden_layers)

    @staticmethod
    def _queue_gui_callback(callback: Callable[[], None]) -> None:
        try:
            from qtpy import QtCore

            timer = getattr(QtCore, "QTimer", None)
            single_shot = getattr(timer, "singleShot", None)
            if callable(single_shot):
                single_shot(0, callback)
                return
        except ImportError:
            pass
        callback()

    def _render_matches_cache_profile(self, profile) -> bool:
        """Return whether the live precomputed render uses the profile grid."""
        render_mode = getattr(self, "_last_render_mode", None)
        if (
            getattr(self, "_last_projection_source", None)
            != _PROJECTION_SOURCE_PRECOMPUTED
            or render_mode
            not in {_RENDER_HEATMAP, _RENDER_FLAT_HEATMAP, _RENDER_ALLEN_LAYERS}
            or getattr(self, "_last_flatmap_style", None) != self._current_style_key()
            or not self._latest_render_mode_is_rendered(render_mode)
        ):
            return False

        summary = getattr(self, "_last_render_summary", None)
        volume_shape = getattr(self, "_last_volume_shape", None)
        if summary is None or volume_shape is None:
            return False
        try:
            grid = profile.style(self._current_style_key()).grid_spec
            y_bins = int(grid["y_bins"])
            x_bins = int(grid["x_bins"])
            x_bounds = tuple(float(value) for value in grid["x_bounds"])
            y_bounds = tuple(float(value) for value in grid["y_bounds"])
        except (KeyError, TypeError, ValueError):
            return False

        bins_match = int(summary.y_bins) == y_bins and int(summary.x_bins) == x_bins
        xy_bounds_match = all(
            np.allclose(cached, rendered, rtol=1e-9, atol=1e-9)
            for cached, rendered in (
                (
                    x_bounds,
                    (float(summary.x_flat_min), float(summary.x_flat_max)),
                ),
                (
                    y_bounds,
                    (float(summary.y_flat_min), float(summary.y_flat_max)),
                ),
            )
        )
        if render_mode == _RENDER_ALLEN_LAYERS:
            layer_labels = tuple(str(value) for value in summary.layer_labels)
            selected_indices = tuple(
                int(value)
                for value in getattr(
                    summary,
                    "selected_layer_indices",
                    tuple(range(len(layer_labels))),
                )
            )
            output_mode = self._allen_layer_output_mode_for_summary(summary)
            expected_shape = (
                (y_bins, x_bins)
                if output_mode == ALLEN_LAYER_OUTPUT_PROJECTION
                else (len(layer_labels), y_bins, x_bins)
            )
            return bool(
                selected_indices == self._current_allen_layer_indices()
                and output_mode == self._current_allen_layer_output_mode()
                and tuple(int(value) for value in volume_shape) == expected_shape
                and bins_match
                and xy_bounds_match
            )

        if render_mode == _RENDER_FLAT_HEATMAP:
            # A collapsed render has no depth axis, so the XY grid alone decides
            # whether the live layer still matches the profile.
            return bool(
                tuple(int(value) for value in volume_shape) == (y_bins, x_bins)
                and bins_match
                and xy_bounds_match
            )

        try:
            output_shape = tuple(int(value) for value in grid["output_shape"])
            depth_bin_um = float(grid["depth_bin_um"])
            includes_minus_one = bool(grid["includes_depth_minus_one_plane"])
            depth_bounds = tuple(float(value) for value in grid["depth_bounds_um"])
        except (KeyError, TypeError, ValueError):
            return False
        depth_bounds_match = np.allclose(
            depth_bounds,
            (float(summary.depth_min_um), float(summary.depth_max_um)),
            rtol=1e-9,
            atol=1e-9,
        )
        return bool(
            tuple(int(value) for value in volume_shape) == output_shape
            and bins_match
            and np.isclose(
                float(summary.depth_bin_um),
                depth_bin_um,
                rtol=1e-9,
                atol=1e-9,
            )
            and bool(summary.includes_depth_minus_one_plane) == includes_minus_one
            and xy_bounds_match
            and depth_bounds_match
        )

    def _adopt_render_for_cache_profile(self, profile) -> None:
        """Attach new provenance to compatible layers without re-uploading them."""
        cache_dir = getattr(self, "_region_cache_dir", None)
        profile_id = self._cache_profile_id(profile)
        self._last_cache_dir = str(cache_dir) if cache_dir is not None else None
        self._last_cache_profile_id = profile_id
        cached_layers = self._current_cached_region_layers()
        for layer in self._current_flatmap_render_layers() + cached_layers:
            metadata = getattr(layer, "metadata", None)
            if isinstance(metadata, dict):
                existing_profile_id = str(metadata.get("cache_profile_id", ""))
                is_cached_layer = any(
                    layer is cached_layer for cached_layer in cached_layers
                )
                if is_cached_layer and existing_profile_id != profile_id:
                    continue
                metadata["cache_path"] = self._last_cache_dir or ""
                metadata["cache_profile_id"] = profile_id

    def _choose_preprocess_lookup_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Bilateral Flatmap Lookup Directory",
        )
        if not path:
            return
        self._preprocess_lookup_dir = Path(path)
        self._lookup_dir_label.setText(str(self._preprocess_lookup_dir))

    def set_cache_directory(
        self,
        path: str | Path | None,
        *,
        profile_id: str | None = None,
    ) -> None:
        """Synchronously open and transactionally activate a region cache.

        GUI entry points use :meth:`_request_cache_directory_open` so full
        array validation never blocks the Qt/VisPy rendering thread.  This
        synchronous method remains useful for tests and scripted callers.
        """
        next_directory = Path(path) if path else None
        if next_directory is None:
            self._invalidate_cache_open_requests()
            self._replace_pending_validated_cache(None)
            previous_cache = getattr(self, "_region_cache", None)
            self._deactivate_cache_profile()
            self._region_cache_dir = None
            self._region_cache = None
            self._pending_cache_profile_id = None
            self._close_region_cache(previous_cache)
            self._cache_dir_label.setText("No cache directory selected")
            self._refresh_cache_profiles()
            return

        self._invalidate_cache_open_requests()
        self._replace_pending_validated_cache(None)
        from ..flatmap_region_cache import open_region_cache

        started = perf_counter()
        logger.info("Opening flatmap region cache synchronously: %s", next_directory)
        next_cache = open_region_cache(next_directory)
        try:
            self._commit_open_region_cache(
                next_directory,
                next_cache,
                profile_id=profile_id,
            )
        except _CacheCompatibilityUnavailable as exc:
            self._replace_pending_validated_cache(
                (next_directory, next_cache, profile_id)
            )
            self._cache_dir_label.setText(str(next_directory))
            self._cache_status_label.setText(str(exc))
            return
        except Exception:
            if next_cache is not getattr(self, "_region_cache", None):
                self._close_region_cache(next_cache)
            raise
        logger.info(
            "Activated flatmap region cache synchronously in %.3fs: %s",
            perf_counter() - started,
            next_directory,
        )

    def _request_cache_directory_open(
        self,
        path: str | Path,
        *,
        profile_id: str | None = None,
    ) -> None:
        """Queue background validation of a cache selected by the GUI."""
        next_directory = Path(path)
        self._replace_pending_validated_cache(None)
        request_id = int(getattr(self, "_cache_open_request_serial", 0)) + 1
        self._cache_open_request_serial = request_id
        request = (
            request_id,
            next_directory,
            str(profile_id) if profile_id else None,
        )
        if self._cache_open_is_running():
            self._pending_cache_open_request = request
            self._cache_status_label.setText(
                f"Waiting to validate flatmap region cache {next_directory}..."
            )
            logger.info(
                "Queued flatmap region-cache open request %d: %s",
                request_id,
                next_directory,
            )
            return
        self._start_cache_open_request(request)

    def _start_cache_open_request(
        self,
        request: tuple[int, Path, str | None],
    ) -> None:
        from qtpy.QtCore import QThread

        from ..workers import RegionCacheOpenWorker

        request_id, cache_dir, profile_id = request
        worker = RegionCacheOpenWorker(cache_dir)
        thread = QThread()
        self._cache_open_thread = thread
        self._cache_open_worker = worker
        self._cache_open_active_request_id = request_id
        self._cache_open_active_request = request
        self._pending_cache_open_request = None
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Bound QObject methods ensure Qt queues all widget/UI work back onto
        # the GUI thread when these signals are emitted by the worker thread.
        worker.finished.connect(self._on_active_cache_open_finished)
        worker.error.connect(self._on_active_cache_open_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_active_cache_open_request)
        self._set_cache_open_controls_enabled(False)
        self._cache_status_label.setText(
            f"Validating flatmap region cache {cache_dir}..."
        )
        logger.info(
            "Started flatmap region-cache open request %d: %s",
            request_id,
            cache_dir,
        )
        thread.start()

    def _on_active_cache_open_finished(self, cache) -> None:
        request = getattr(self, "_cache_open_active_request", None)
        if request is None:
            logger.info("Closing unclaimed flatmap region-cache result.")
            self._close_region_cache(cache)
            return
        request_id, cache_dir, profile_id = request
        self._on_cache_open_finished(cache, request_id, cache_dir, profile_id)

    def _on_active_cache_open_error(self, message: str) -> None:
        request = getattr(self, "_cache_open_active_request", None)
        if request is None:
            return
        request_id, cache_dir, _profile_id = request
        self._on_cache_open_error(message, request_id, cache_dir)

    def _cleanup_active_cache_open_request(self) -> None:
        request = getattr(self, "_cache_open_active_request", None)
        thread = getattr(self, "_cache_open_thread", None)
        worker = getattr(self, "_cache_open_worker", None)
        if request is None or thread is None or worker is None:
            return
        self._cleanup_cache_open_request(request[0], thread, worker)

    def _on_cache_open_finished(
        self,
        cache,
        request_id: int,
        cache_dir: Path,
        profile_id: str | None,
    ) -> None:
        if not self._cache_open_request_is_current(request_id):
            logger.info(
                "Closing stale flatmap region-cache result %d: %s",
                request_id,
                cache_dir,
            )
            self._close_region_cache(cache)
            return
        try:
            self._commit_open_region_cache(
                cache_dir,
                cache,
                profile_id=profile_id,
            )
        except _CacheCompatibilityUnavailable as exc:
            self._replace_pending_validated_cache((cache_dir, cache, profile_id))
            self._cache_dir_label.setText(str(cache_dir))
            self._cache_status_label.setText(str(exc))
            logger.info(
                "Holding validated flatmap region cache until compatibility "
                "inputs are available: %s",
                cache_dir,
            )
        except Exception as exc:
            self._close_region_cache(cache)
            self._report_cache_open_failure(cache_dir, exc)

    def _on_cache_open_error(
        self,
        message: str,
        request_id: int,
        cache_dir: Path,
    ) -> None:
        if not self._cache_open_request_is_current(request_id):
            return
        self._report_cache_open_failure(cache_dir, message)

    def _report_cache_open_failure(
        self,
        cache_dir: Path,
        error: object,
    ) -> None:
        logger.error("Failed to activate flatmap region cache %s: %s", cache_dir, error)
        message = f"Flatmap region cache is incompatible or corrupt: {error}"
        self._cache_status_label.setText(message)
        show_warning(message)

    def _cleanup_cache_open_request(
        self,
        request_id: int,
        thread,
        worker,
    ) -> None:
        if getattr(self, "_cache_open_thread", None) is thread:
            self._cache_open_thread = None
        if getattr(self, "_cache_open_worker", None) is worker:
            self._cache_open_worker = None
        if getattr(self, "_cache_open_active_request_id", None) == request_id:
            self._cache_open_active_request_id = None
        active_request = getattr(self, "_cache_open_active_request", None)
        if active_request is not None and active_request[0] == request_id:
            self._cache_open_active_request = None

        pending = getattr(self, "_pending_cache_open_request", None)
        serial = int(getattr(self, "_cache_open_request_serial", 0))
        shutting_down = bool(getattr(self, "_cache_open_shutting_down", False))
        if pending is not None and pending[0] == serial and not shutting_down:
            self._start_cache_open_request(pending)
            return
        if not shutting_down:
            self._set_cache_open_controls_enabled(True)

    def _cache_open_is_running(self) -> bool:
        thread = getattr(self, "_cache_open_thread", None)
        is_running = getattr(thread, "isRunning", None)
        return bool(thread is not None and callable(is_running) and is_running())

    def _cache_open_request_is_current(self, request_id: int) -> bool:
        return bool(
            not getattr(self, "_cache_open_shutting_down", False)
            and int(request_id) == int(getattr(self, "_cache_open_request_serial", 0))
        )

    def _invalidate_cache_open_requests(self) -> None:
        self._cache_open_request_serial = (
            int(getattr(self, "_cache_open_request_serial", 0)) + 1
        )
        self._pending_cache_open_request = None

    def _on_cache_widget_destroyed(self, *_args) -> None:
        self._cache_open_shutting_down = True
        self._invalidate_cache_open_requests()
        self._replace_pending_validated_cache(None)

    def _replace_pending_validated_cache(
        self,
        pending: tuple[Path, object, str | None] | None,
    ) -> None:
        previous = getattr(self, "_pending_validated_cache", None)
        self._pending_validated_cache = pending
        if previous is not None and (pending is None or previous[1] is not pending[1]):
            self._close_region_cache(previous[1])

    def _set_cache_open_controls_enabled(self, enabled: bool) -> None:
        for name in ("_cache_dir_btn", "_cache_profile_combo", "_build_cache_btn"):
            widget = getattr(self, name, None)
            set_enabled = getattr(widget, "setEnabled", None)
            if callable(set_enabled):
                set_enabled(bool(enabled))

    @staticmethod
    def _close_region_cache(cache) -> None:
        """Release a superseded cache without assuming a concrete cache type."""
        if cache is None:
            return
        close = getattr(cache, "close", None)
        if callable(close):
            started = perf_counter()
            try:
                close()
            except Exception:
                logger.warning(
                    "Failed to close flatmap region-cache memory maps.",
                    exc_info=True,
                )
            else:
                logger.info(
                    "Closed flatmap region-cache memory maps in %.3fs.",
                    perf_counter() - started,
                )

    def active_cache_reference(self) -> dict[str, str] | None:
        """Return the external cache reference stored in project bundles."""
        cache_dir = getattr(self, "_region_cache_dir", None)
        profile = getattr(self, "_active_cache_profile", None)
        profile_id = self._cache_profile_id(profile)
        if cache_dir is None or not profile_id:
            return None
        return {"path": str(cache_dir), "profile_id": profile_id}

    def restore_cache_reference(self, reference: object) -> None:
        """Restore a project bundle's external cache path/profile selection."""
        if not isinstance(reference, dict):
            return
        path = reference.get("path")
        profile_id = str(reference.get("profile_id") or "")
        if not path:
            return
        self._pending_cache_profile_id = profile_id or None
        self._request_cache_directory_open(
            str(path),
            profile_id=profile_id or None,
        )

    def refresh_cache_profiles(self) -> None:
        """Re-evaluate cache compatibility after atlas or Parquet changes."""
        pending = getattr(self, "_pending_validated_cache", None)
        if pending is not None:
            cache_dir, cache, profile_id = pending
            try:
                self._commit_open_region_cache(
                    cache_dir,
                    cache,
                    profile_id=profile_id,
                )
            except _CacheCompatibilityUnavailable as exc:
                self._cache_status_label.setText(str(exc))
            except Exception as exc:
                self._replace_pending_validated_cache(None)
                self._report_cache_open_failure(cache_dir, exc)
            return
        self._refresh_cache_profiles()

    def invalidate_loaded_parquet_projection(self) -> None:
        """Clear flatmap state before associating the tab with a new Parquet."""
        self._invalidate_flatmap_grid_layers()
        self._update_lookup_files_section()

    def _deactivate_cache_profile(self) -> None:
        if getattr(self, "_active_cache_profile", None) is not None:
            self._invalidate_flatmap_grid_layers()
        self._active_cache_profile = None
        self._set_cache_grid_locked(False)
        self._update_cached_region_controls()
        self._notify_flatmap_correlation_source_changed()

    def _choose_cache_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Flatmap Region Cache Directory",
        )
        if not path:
            return
        self._request_cache_directory_open(path, profile_id=None)

    @staticmethod
    def _cache_profile_id(profile) -> str:
        return str(getattr(profile, "profile_id", "") or "")

    @staticmethod
    def _atlas_family_name(name: object) -> str:
        return re.sub(r"_(?:10|25|50)um$", "", str(name or ""))

    def _refresh_cache_profiles(self) -> None:
        """Show only profiles compatible with the v3 Parquet/style/catalog."""
        combo = getattr(self, "_cache_profile_combo", None)
        if combo is None:
            return
        cache = getattr(self, "_region_cache", None)
        if cache is None:
            self._populate_cache_profile_combo((), None)
            self._deactivate_cache_profile()
            return

        try:
            entries = self._compatible_cache_profile_entries(cache)
        except RuntimeError as exc:
            self._populate_cache_profile_combo((), None)
            self._deactivate_cache_profile()
            self._cache_status_label.setText(str(exc))
            return

        pending_profile_id = str(
            getattr(self, "_pending_cache_profile_id", "")
            or self._cache_profile_id(getattr(self, "_active_cache_profile", None))
        )
        profile = self._select_cache_profile(entries, pending_profile_id)
        self._populate_cache_profile_combo(entries, profile)
        self._activate_cache_profile(profile)

    def _compatible_cache_profile_entries(
        self,
        cache,
    ) -> tuple[tuple[str, object], ...]:
        """Return display labels and compatible profiles without mutating UI."""
        try:
            info = read_flatmap_parquet_transform_info(
                self._current_source_parquet_path()
            )
        except Exception as exc:
            raise _CacheCompatibilityUnavailable(
                f"Load a version-3 neuron Parquet to select a cache profile: {exc}"
            ) from exc
        if info.format_version < 3 or not info.lookup_set_id:
            raise _CacheCompatibilityUnavailable(
                "Legacy Parquets can render neurons, but exact cache overlays "
                "require version-3 preprocessing."
            )

        atlas = self._atlas_provider()
        if atlas is None:
            raise _CacheCompatibilityUnavailable(
                "Load a matching BrainGlobe atlas structure catalog to use the cache."
            )

        style = self._current_style_key()
        current_atlas_name = str(getattr(atlas, "atlas_name", "") or "")
        from ..flatmap_region_cache import structure_catalog_id

        try:
            current_catalog_id = structure_catalog_id(
                getattr(atlas, "structures", None)
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise _CacheCompatibilityUnavailable(
                "The loaded atlas does not expose a usable structure catalog."
            ) from exc
        current_atlas_version = self._atlas_version_text(
            getattr(atlas, "local_version", None)
            or getattr(atlas, "atlas_version", None)
            or getattr(atlas, "version", None)
        )
        shared_depth = (info.metadata or {}).get("shared_depth_definition")
        if not isinstance(shared_depth, dict):
            raise RuntimeError(
                "Version-3 Parquet metadata is missing its shared depth definition."
            )
        try:
            parquet_mirror_axis = int(shared_depth["mirror_coord_axis"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Version-3 Parquet metadata has no valid depth mirror axis."
            ) from exc

        entries: list[tuple[str, object]] = []
        mismatch_messages: list[str] = []
        for profile in cache.profiles.values():
            mismatches = list(
                profile.compatibility_mismatches(
                    lookup_set_id=info.lookup_set_id,
                    atlas_name=current_atlas_name,
                    atlas_version=current_atlas_version,
                    structure_catalog_id=current_catalog_id,
                    style=style,
                    mirror_depth_fallback=True,
                    mirror_coord_axis=parquet_mirror_axis,
                )
            )
            cached_atlas_name = str(profile.atlas.get("name", "") or "")
            if mismatches:
                mismatch_messages.append(
                    f"{profile.profile_id[:12]}: " + "; ".join(mismatches)
                )
                continue
            grid = profile.style(style).grid_spec
            label = (
                f"{profile.profile_id[:12]} — {cached_atlas_name}, "
                f"{grid.get('x_bins')}x{grid.get('y_bins')} XY / "
                f"{grid.get('depth_bin_um')} um"
            )
            entries.append((label, profile))

        if not entries:
            detail = mismatch_messages[0] if mismatch_messages else "no profiles"
            raise RuntimeError(
                f"No compatible cache profile: {detail}. "
                "Recomputation will not start automatically."
            )
        return tuple(entries)

    @classmethod
    def _select_cache_profile(
        cls,
        entries: tuple[tuple[str, object], ...],
        profile_id: str | None,
    ):
        requested = str(profile_id or "")
        if requested:
            for _label, profile in entries:
                if cls._cache_profile_id(profile) == requested:
                    return profile
        return entries[0][1] if entries else None

    def _populate_cache_profile_combo(
        self,
        entries: tuple[tuple[str, object], ...],
        profile,
    ) -> None:
        combo = getattr(self, "_cache_profile_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        try:
            combo.clear()
            target_index = -1
            target_id = self._cache_profile_id(profile)
            for index, (label, candidate) in enumerate(entries):
                combo.addItem(label, candidate)
                if self._cache_profile_id(candidate) == target_id:
                    target_index = index
            if target_index >= 0:
                combo.setCurrentIndex(target_index)
        finally:
            combo.blockSignals(False)

    def _commit_open_region_cache(
        self,
        cache_dir: Path,
        cache,
        *,
        profile_id: str | None,
    ) -> None:
        """Commit an opened candidate only after compatibility succeeds."""
        compatibility_started = perf_counter()
        try:
            entries = self._compatible_cache_profile_entries(cache)
        except Exception:
            logger.info(
                "Rejected flatmap region cache after %.3fs compatibility check: %s",
                perf_counter() - compatibility_started,
                cache_dir,
            )
            raise
        logger.info(
            "Found %d compatible flatmap cache profile(s) in %.3fs: %s",
            len(entries),
            perf_counter() - compatibility_started,
            cache_dir,
        )
        profile = self._select_cache_profile(entries, profile_id)
        if profile is None:
            raise RuntimeError("The cache contains no compatible profile.")

        previous_directory = getattr(self, "_region_cache_dir", None)
        previous_cache = getattr(self, "_region_cache", None)
        previous_profile_id = self._cache_profile_id(
            getattr(self, "_active_cache_profile", None)
        )
        next_profile_id = self._cache_profile_id(profile)
        force_transition = bool(
            previous_directory != cache_dir or previous_profile_id != next_profile_id
        )

        self._region_cache_dir = Path(cache_dir)
        self._region_cache = cache
        self._pending_cache_profile_id = next_profile_id
        self._cache_dir_label.setText(str(cache_dir))
        self._populate_cache_profile_combo(entries, profile)
        self._activate_cache_profile(profile, force_transition=force_transition)
        pending = getattr(self, "_pending_validated_cache", None)
        if pending is not None and pending[1] is cache:
            self._pending_validated_cache = None

        if previous_cache is not None and previous_cache is not cache:
            self._close_region_cache(previous_cache)
            logger.info(
                "Closed superseded flatmap region cache: %s", previous_directory
            )
        logger.info(
            "Committed flatmap region cache %s with profile %s",
            cache_dir,
            next_profile_id,
        )

    def _on_cache_profile_changed(self) -> None:
        combo = getattr(self, "_cache_profile_combo", None)
        profile = combo.currentData() if combo is not None else None
        self._activate_cache_profile(profile)

    def _activate_cache_profile(
        self,
        profile,
        *,
        force_transition: bool = False,
    ) -> None:
        previous_profile_id = self._cache_profile_id(
            getattr(self, "_active_cache_profile", None)
        )
        next_profile_id = self._cache_profile_id(profile)
        transition = bool(force_transition or previous_profile_id != next_profile_id)
        had_render = bool(self._current_flatmap_render_layers())
        preserved_render = bool(
            transition
            and had_render
            and profile is not None
            and self._render_matches_cache_profile(profile)
        )
        stale_render = bool(transition and had_render and not preserved_render)

        if transition:
            if preserved_render:
                self._defer_cached_region_layer_removal(
                    keep_profile_id=next_profile_id,
                    keep_style=self._current_style_key(),
                )
            elif stale_render:
                self._defer_flatmap_grid_layer_removal()
        self._active_cache_profile = profile
        if profile is None:
            self._set_cache_grid_locked(False)
            self._update_cached_region_controls()
            self._notify_flatmap_correlation_source_changed()
            return
        self._pending_cache_profile_id = self._cache_profile_id(profile)
        style_cache = profile.style(self._current_style_key())
        grid = style_cache.grid_spec
        self._y_bins_spin.setValue(int(grid["y_bins"]))
        self._depth_bin_spin.setValue(float(grid["depth_bin_um"]))
        self._exclude_depth_minus_one_cb.setChecked(True)
        locked = self._current_projection_source() != _PROJECTION_SOURCE_RECOMPUTE
        self._set_cache_grid_locked(locked)
        if preserved_render:
            self._adopt_render_for_cache_profile(profile)
            suffix = "matching heatmap kept; grid controls are locked"
            logger.info(
                "Preserved rendered flatmap heatmap for cache profile %s",
                profile.profile_id,
            )
        elif stale_render:
            suffix = "heatmap grid changed; click Project to Flatmap again"
            logger.info(
                "Deferred stale flatmap heatmap removal for cache profile %s",
                profile.profile_id,
            )
        else:
            suffix = "grid controls are locked" if locked else "NRRD fallback is active"
        self._cache_status_label.setText(
            f"Active cache profile {profile.profile_id}; {suffix}."
        )
        self._update_cached_region_controls()
        self._notify_flatmap_correlation_source_changed()

    def _cached_region_control_states(self) -> dict[str, bool]:
        """Return the single source of truth for cached-region button state.

        Labels and outlines can be collapsed into one flatmap plane, so both are
        offered in the depth-free renders. Cached surfaces are 3D voxel shells
        with no 2D form and stay depth-grid only. The recompute path builds
        labels from NRRDs on the depth grid, so it needs a depth render.
        """
        precomputed = (
            self._current_projection_source() == _PROJECTION_SOURCE_PRECOMPUTED
        )
        cache_available = (
            precomputed and getattr(self, "_active_cache_profile", None) is not None
        )
        depth_grid_mode = self._is_depth_grid_mode()
        flat_mode = self._is_flat_render_mode()
        recompute_depth = depth_grid_mode and not precomputed
        allen_selection_ready = self._allen_layer_selection_ready()
        render_mode_ready = not bool(
            getattr(self, "_flatmap_render_mode_change_pending", False)
        )
        return {
            "_region_labels_btn": (
                cache_available or recompute_depth
            )
            and allen_selection_ready
            and render_mode_ready,
            "_region_surfaces_btn": (
                cache_available and depth_grid_mode and render_mode_ready
            ),
            "_region_outlines_btn": (
                cache_available and (depth_grid_mode or flat_mode) and render_mode_ready
            ),
            "_clear_region_geometry_btn": cache_available
            and (depth_grid_mode or flat_mode)
            and render_mode_ready,
            "_region_label_atlas_combo": recompute_depth and render_mode_ready,
        }

    def _update_cached_region_controls(self) -> None:
        for name, enabled in self._cached_region_control_states().items():
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(enabled)
        surfaces_button = getattr(self, "_region_surfaces_btn", None)
        set_tooltip = getattr(surfaces_button, "setToolTip", None)
        if callable(set_tooltip) and self._is_flat_render_mode():
            set_tooltip(_REGION_SURFACES_2D_TOOLTIP)
        elif callable(set_tooltip):
            set_tooltip("")

    def _set_cache_grid_locked(self, locked: bool) -> None:
        self._cache_grid_locked = bool(locked)
        self._update_render_mode_controls()

    @staticmethod
    def _cache_profile_bounds(
        profile,
        style: str,
    ) -> dict[str, tuple[float, float]] | None:
        if profile is None:
            return None
        try:
            grid = profile.style(style).grid_spec
            return {
                "x_bounds": tuple(float(value) for value in grid["x_bounds"]),
                "y_bounds": tuple(float(value) for value in grid["y_bounds"]),
                "depth_range_um": tuple(
                    float(value) for value in grid["depth_bounds_um"]
                ),
            }
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _atlas_annotation_tiff_path(atlas) -> Path:
        """Resolve BrainGlobe's on-disk annotation without reading the volume."""
        direct = getattr(atlas, "annotation_path", None)
        if direct and Path(direct).is_file():
            return Path(direct)
        for attribute in ("root_dir", "atlas_dir", "brainglobe_dir"):
            root = getattr(atlas, attribute, None)
            if root:
                candidate = Path(root) / "annotation.tiff"
                if candidate.is_file():
                    return candidate
        from ..workers import cached_brainglobe_atlas_dir

        atlas_name = str(getattr(atlas, "atlas_name", "") or "")
        atlas_dir = cached_brainglobe_atlas_dir(atlas_name)
        if atlas_dir is not None:
            candidate = atlas_dir / "annotation.tiff"
            if candidate.is_file():
                return candidate
        raise RuntimeError(
            "The matching BrainGlobe annotation.tiff could not be located. "
            "Cache generation requires the exact on-disk atlas grid."
        )

    @staticmethod
    def _atlas_version_text(value: object | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, (tuple, list)):
            text = ".".join(str(part).strip() for part in value)
        else:
            text = str(value).strip()
        return text or None

    @staticmethod
    def _atlas_version(atlas, annotation_path: Path) -> str | None:
        raw = (
            getattr(atlas, "local_version", None)
            or getattr(atlas, "atlas_version", None)
            or getattr(atlas, "version", None)
        )
        normalized = FlatmapProjectionWidget._atlas_version_text(raw)
        if normalized is not None:
            return normalized
        match = re.search(r"_v([^/]+)$", annotation_path.parent.name)
        return match.group(1) if match else None

    def _build_cache_profile(self) -> None:
        """Validate inputs and launch one atomic cache-profile build."""
        lookup_dir = getattr(self, "_preprocess_lookup_dir", None)
        if lookup_dir is None:
            show_warning("Choose the bilateral lookup directory first.")
            return
        atlas = self._atlas_provider()
        if atlas is None:
            show_warning("Load the exact BrainGlobe atlas before building a cache.")
            return
        cache_dir = getattr(self, "_region_cache_dir", None)
        if cache_dir is None:
            selected = QFileDialog.getExistingDirectory(
                self,
                "Choose or Create Flatmap Region Cache Directory",
            )
            if not selected:
                return
            cache_dir = Path(selected)
            self._region_cache_dir = cache_dir
            self._cache_dir_label.setText(str(cache_dir))
        try:
            annotation_path = self._atlas_annotation_tiff_path(atlas)
        except Exception as exc:
            show_warning(str(exc))
            return

        from qtpy.QtCore import QThread

        from ..workers import RegionCacheBuildWorker

        resolution_control = getattr(self, "_lookup_resolution_spin", None)
        raw_lookup_resolution = (
            int(resolution_control.value()) if resolution_control is not None else 0
        )
        atlas_resolution = tuple(
            float(value) for value in np.asarray(atlas.resolution).reshape(-1)
        )
        worker = RegionCacheBuildWorker(
            cache_dir=cache_dir,
            lookup_dir=lookup_dir,
            annotation_path=annotation_path,
            atlas_name=str(getattr(atlas, "atlas_name", "") or ""),
            atlas_version=self._atlas_version(atlas, annotation_path),
            atlas_resolution_um=atlas_resolution,
            atlas_structures=getattr(atlas, "structures", None),
            y_bins=self._current_cache_build_y_bins(),
            depth_bin_um=self._current_cache_build_depth_bin_um(),
            lookup_resolution_um=(
                float(raw_lookup_resolution) if raw_lookup_resolution > 0 else None
            ),
        )
        thread = QThread()
        self._cache_build_thread = thread
        self._cache_build_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_cache_build_progress)
        worker.finished.connect(self._on_cache_build_finished)
        worker.error.connect(self._on_cache_build_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._cleanup_cache_build(thread, worker))
        self._build_cache_btn.setEnabled(False)
        self._cancel_cache_btn.setEnabled(True)
        self._cache_status_label.setText("Starting region-cache build...")
        thread.start()

    def _on_cache_build_progress(
        self,
        message: str,
        current: int,
        total: int,
    ) -> None:
        suffix = f" ({current}/{total})" if total > 0 else ""
        self._cache_status_label.setText(f"{message}{suffix}")

    def _cancel_cache_build(self) -> None:
        worker = getattr(self, "_cache_build_worker", None)
        cancel = getattr(worker, "cancel", None)
        if callable(cancel):
            cancel()
            self._cache_status_label.setText("Cancelling cache build...")
            self._cancel_cache_btn.setEnabled(False)

    def _on_cache_build_finished(self, profile) -> None:
        profile_id = self._cache_profile_id(profile)
        try:
            self._cache_status_label.setText(
                f"Built cache profile {profile_id}; validating it for viewing..."
            )
            self._request_cache_directory_open(
                self._region_cache_dir,
                profile_id=profile_id,
            )
            show_info(f"Built flatmap region-cache profile {profile_id}")
        finally:
            close = getattr(profile, "close", None)
            if callable(close):
                close()

    def _on_cache_build_error(self, message: str) -> None:
        self._cache_status_label.setText(f"Region-cache build failed: {message}")
        show_warning(f"Region-cache build failed: {message}")

    def _cleanup_cache_build(self, thread, worker) -> None:
        if getattr(self, "_cache_build_thread", None) is thread:
            self._cache_build_thread = None
        if getattr(self, "_cache_build_worker", None) is worker:
            self._cache_build_worker = None
        self._build_cache_btn.setEnabled(not self._cache_open_is_running())
        self._cancel_cache_btn.setEnabled(False)

    def _current_style_key(self) -> str:
        key = self._style_combo.currentData()
        return str(key or "both_shaped")

    def _current_style_filename(self) -> str:
        return FLATMAP_STYLE_FILENAMES.get(
            self._current_style_key(),
            FLATMAP_STYLE_FILENAMES["both_shaped"],
        )

    def _current_coordinate_mode(self) -> str:
        mode = self._coordinate_mode_combo.currentData()
        return str(mode or COORDINATE_MODE_MICRONS)

    def _current_render_mode(self) -> str:
        combo = getattr(self, "_render_mode_combo", None)
        current_data = getattr(combo, "currentData", None)
        mode = current_data() if callable(current_data) else None
        return str(mode or _RENDER_HEATMAP)

    def _is_allen_layer_mode(self) -> bool:
        return self._current_render_mode() == _RENDER_ALLEN_LAYERS

    def _current_allen_layer_indices(self) -> tuple[int, ...]:
        """Return checked canonical Allen layer indices in anatomical order."""
        checkboxes = getattr(self, "_allen_layer_checkboxes", None)
        if checkboxes is None:
            # Widgets assembled by older integrations and test doubles retain
            # the original all-six behavior.
            return tuple(range(len(ALLEN_ISOCORTEX_LAYER_LABELS)))
        return tuple(
            index
            for index, checkbox in enumerate(checkboxes)
            if bool(checkbox.isChecked())
        )

    def _current_allen_layer_output_mode(self) -> str:
        combo = getattr(self, "_allen_layer_output_combo", None)
        current_data = getattr(combo, "currentData", None)
        value = current_data() if callable(current_data) else None
        if value == ALLEN_LAYER_OUTPUT_PROJECTION:
            return ALLEN_LAYER_OUTPUT_PROJECTION
        return ALLEN_LAYER_OUTPUT_STACK

    def _is_allen_layer_projection(self) -> bool:
        return bool(
            self._is_allen_layer_mode()
            and self._current_allen_layer_output_mode()
            == ALLEN_LAYER_OUTPUT_PROJECTION
        )

    def _allen_layer_selection_ready(self) -> bool:
        return bool(
            not self._is_allen_layer_mode()
            or self._current_allen_layer_indices()
        )

    def _set_allen_layer_selection(self, checked: bool) -> None:
        checkboxes = getattr(self, "_allen_layer_checkboxes", ())
        for checkbox in checkboxes:
            previous = checkbox.blockSignals(True)
            try:
                checkbox.setChecked(bool(checked))
            finally:
                checkbox.blockSignals(previous)
        self._on_allen_layer_options_changed()

    def _on_allen_layer_options_changed(self, *_args) -> None:
        """Retire output whose categorical selection or dimensionality changed."""
        if self._is_allen_layer_mode():
            self._invalidate_flatmap_grid_layers()
        self._update_render_mode_controls()
        self._update_cached_region_controls()
        status = getattr(self, "_status_label", None)
        if status is not None and self._is_allen_layer_mode():
            indices = self._current_allen_layer_indices()
            if not indices:
                status.setText("Select at least one Allen layer.")
            else:
                labels = ", ".join(
                    ALLEN_ISOCORTEX_LAYER_LABELS[index] for index in indices
                )
                view = (
                    "a single 2D projection"
                    if self._is_allen_layer_projection()
                    else "a compact 2D stack"
                )
                status.setText(f"Allen layers {labels} will render as {view}.")
        self._notify_flatmap_correlation_source_changed()

    def _is_flat_render_mode(self) -> bool:
        """Return whether the render collapses depth into one flatmap plane."""
        return self._current_render_mode() in {
            _RENDER_FLAT_HEATMAP,
            _RENDER_FLAT_VECTOR,
        }

    def _is_depth_grid_mode(self) -> bool:
        """Return whether the render's plane axis is the cached depth grid.

        Cached region geometry is built per depth bin, so it only describes the
        depth-binned renders.  Both 2D modes and the Allen layer stack use a
        different plane axis (or none at all).
        """
        return self._current_render_mode() in {_RENDER_HEATMAP, _RENDER_POINTS}

    def _current_plane_mode(self) -> str:
        """Return the plane-axis mode of the current render mode."""
        if self._is_allen_layer_mode():
            return (
                FLATMAP_PLANE_MODE_ALLEN_LAYER_PROJECTION
                if self._is_allen_layer_projection()
                else FLATMAP_PLANE_MODE_ALLEN_LAYERS
            )
        if self._is_flat_render_mode():
            return FLATMAP_PLANE_MODE_FLAT
        return FLATMAP_PLANE_MODE_DEPTH

    @staticmethod
    def _render_ndisplay(render_mode: str) -> int:
        """Return the display dimensionality a render mode needs."""
        if render_mode in {
            _RENDER_FLAT_HEATMAP,
            _RENDER_FLAT_VECTOR,
            _RENDER_ALLEN_LAYERS,
        }:
            return 2
        return 3

    def _update_flatmap_render_mode_change_pending(self) -> bool:
        """Record whether the active scene uses another render coordinate space."""
        render_mode = self._current_render_mode()
        render_layers = self._current_flatmap_render_layers()
        render_mismatch = any(
            (getattr(layer, "metadata", {}) or {}).get("flatmap_render_mode")
            != render_mode
            for layer in render_layers
        )
        soma_layer = self._cached_soma_layer()
        soma_metadata = getattr(soma_layer, "metadata", {}) or {}
        soma_mismatch = bool(
            soma_layer is not None
            and soma_metadata.get("flatmap_soma_space_render_mode") != render_mode
        )
        plane_mode = self._current_plane_mode()
        region_mismatch = any(
            (getattr(layer, "metadata", {}) or {}).get("flatmap_plane_mode")
            not in {None, "", plane_mode}
            for layer in self._current_cached_region_layers()
        )
        self._flatmap_render_mode_change_pending = bool(
            render_mismatch or soma_mismatch or region_mismatch
        )
        return self._flatmap_render_mode_change_pending

    def _on_render_mode_changed(self, *_args) -> None:
        # Keep the displayed scene intact while the user chooses settings for
        # a possible comparison window. If they re-project into this viewer,
        # the stale grid is retired from the Project button callback instead
        # of from this Qt combo-box callback.
        self._update_flatmap_render_mode_change_pending()
        self._update_render_mode_controls()
        self._update_cached_region_controls()
        status = getattr(self, "_status_label", None)
        if status is not None and self._is_allen_layer_mode():
            indices = self._current_allen_layer_indices()
            if not indices:
                status.setText("Select at least one Allen layer.")
            else:
                labels = ", ".join(
                    ALLEN_ISOCORTEX_LAYER_LABELS[index] for index in indices
                )
                view = (
                    "a single 2D projection"
                    if self._is_allen_layer_projection()
                    else "a compact 2D stack"
                )
                status.setText(f"Allen layers {labels} will render as {view}.")
        elif status is not None and self._is_flat_render_mode():
            status.setText(
                "2D mode collapses the depth axis into one flatmap plane; "
                "Exclude depth -1 nodes still decides whether depth -1 nodes "
                "are rendered."
            )
        self._notify_flatmap_correlation_source_changed()

    def _update_render_mode_controls(self) -> None:
        layer_mode = self._is_allen_layer_mode()
        flat_mode = self._is_flat_render_mode()
        vector_mode = self._current_render_mode() == _RENDER_FLAT_VECTOR
        cache_locked = bool(getattr(self, "_cache_grid_locked", False))
        options_group = getattr(self, "_allen_layer_options_group", None)
        set_visible = getattr(options_group, "setVisible", None)
        if callable(set_visible):
            set_visible(layer_mode)
        control_states = {
            "_y_bins_spin": not cache_locked,
            # A collapsed render has no depth bins to size, but the depth -1
            # checkbox still selects which nodes it counts.
            "_depth_bin_spin": not cache_locked and not layer_mode and not flat_mode,
            "_exclude_depth_minus_one_cb": not cache_locked and not layer_mode,
            "_negative_one_sentinel_cb": not cache_locked,
            "_zero_sentinel_cb": not cache_locked,
            # Vectors are always colored per neuron from the table's colors.
            "_heatmap_color_mode_combo": not vector_mode,
        }
        for name, enabled in control_states.items():
            widget = getattr(self, name, None)
            set_enabled = getattr(widget, "setEnabled", None)
            if callable(set_enabled):
                set_enabled(bool(enabled))
        if not self._precomputed_heatmap_is_running():
            self._set_projection_controls_enabled(True)

    def _current_allen_layer_map(self) -> AllenIsocortexLayerMap:
        atlas = self._atlas_provider()
        structures = getattr(atlas, "structures", None) if atlas is not None else None
        key = (
            id(atlas),
            id(structures),
            str(getattr(atlas, "atlas_name", "") or ""),
        )
        if (
            key == getattr(self, "_allen_layer_map_cache_key", None)
            and getattr(self, "_allen_layer_map_cache", None) is not None
        ):
            return self._allen_layer_map_cache
        try:
            layer_map = layer_map_from_atlas(atlas)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        self._allen_layer_map_cache_key = key
        self._allen_layer_map_cache = layer_map
        return layer_map

    def _current_heatmap_color_mode(self) -> str:
        combo = getattr(self, "_heatmap_color_mode_combo", None)
        current_data = getattr(combo, "currentData", None)
        mode = current_data() if callable(current_data) else None
        if mode in {
            _HEATMAP_COLOR_SINGLE,
            _HEATMAP_COLOR_INDIVIDUAL,
            _HEATMAP_COLOR_CLUSTER,
        }:
            return str(mode)
        return _HEATMAP_COLOR_SINGLE

    def _active_cache_grid_spec(self):
        """Return the active profile's grid spec, or ``None`` for a live render."""
        profile = getattr(self, "_active_cache_profile", None)
        if (
            profile is None
            or self._current_projection_source() != _PROJECTION_SOURCE_PRECOMPUTED
        ):
            return None
        try:
            return profile.style(self._current_style_key()).grid_spec
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def _current_y_bins(self) -> int:
        grid = self._active_cache_grid_spec()
        if grid is not None:
            return int(grid["y_bins"])
        return int(self._y_bins_spin.value())

    def _current_x_bins(self) -> int | None:
        """Return the x count to render with, or ``None`` to derive it.

        A cache-backed render must reproduce the profile's *stored* count
        exactly; re-deriving it from JSON-round-tripped bounds could differ at a
        rounding tie, and the render would then be discarded as a mismatch.
        """
        grid = self._active_cache_grid_spec()
        if grid is not None:
            return int(grid["x_bins"])
        return None

    def _current_depth_bin_um(self) -> float:
        profile = getattr(self, "_active_cache_profile", None)
        if (
            profile is not None
            and self._current_projection_source() == _PROJECTION_SOURCE_PRECOMPUTED
        ):
            return float(
                profile.style(self._current_style_key()).grid_spec["depth_bin_um"]
            )
        return float(self._depth_bin_spin.value())

    def _current_cache_build_y_bins(self) -> int:
        control = getattr(self, "_cache_build_y_bins_spin", None)
        if control is None:
            return self._current_y_bins()
        return int(control.value())

    def _current_cache_build_depth_bin_um(self) -> float:
        control = getattr(self, "_cache_build_depth_bin_spin", None)
        if control is None:
            return self._current_depth_bin_um()
        return float(control.value())

    def _current_source_mode(self) -> str:
        mode = self._source_combo.currentData()
        return str(mode or _SOURCE_SELECTED)

    def _current_region_label_atlas_name(self) -> str:
        combo = getattr(self, "_region_label_atlas_combo", None)
        current_text = getattr(combo, "currentText", None)
        if callable(current_text):
            atlas_name = str(current_text() or "").strip()
            if atlas_name:
                return atlas_name
        return _REGION_LABEL_ATLAS_DEFAULT

    def _update_expected_filename_label(self) -> None:
        filename = self._current_style_filename()
        self._expected_filename_label.setText(
            f"Expected Zenodo v4.1 flatmap file for this style: {filename}"
        )

    def _choose_flatmap_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Flatmap NRRD",
            "",
            "NRRD Files (*.nrrd);;All Files (*)",
        )
        if path:
            self.set_flatmap_path(path)

    def _choose_depth_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Depth NRRD",
            "",
            "NRRD Files (*.nrrd);;All Files (*)",
        )
        if path:
            self.set_depth_path(path)

    @staticmethod
    def _deduplicate_file_ids(file_ids: list[object]) -> list[object]:
        out: list[object] = []
        seen: set[object] = set()
        for file_id in file_ids:
            if file_id in seen:
                continue
            seen.add(file_id)
            out.append(file_id)
        return out

    def _file_ids_for_source(self, source_mode: str | None = None) -> list[object]:
        mode = source_mode or self._current_source_mode()
        table_ids = self._deduplicate_file_ids(
            list(self._table_file_ids_provider() or [])
        )
        if mode == _SOURCE_ALL:
            return table_ids

        selected_ids = self._deduplicate_file_ids(
            list(self._selected_file_ids_provider() or [])
        )
        return selected_ids if selected_ids else table_ids

    def _query_nodes(self, file_ids: list[object]) -> pd.DataFrame:
        db = self._database_provider()
        if db is None:
            raise RuntimeError("Load a neuron Parquet before projecting to flatmap.")
        if not file_ids:
            raise RuntimeError("No neurons are available to project.")

        getter = getattr(db, "get_neurons_for_rendering", None)
        if not callable(getter):
            raise RuntimeError(
                "Loaded neuron database does not support rendering queries."
            )
        nodes = getter(file_ids)
        if nodes is None or nodes.empty:
            raise RuntimeError("No neuron rows matched the requested file IDs.")
        return nodes

    def _query_soma_nodes(self, file_ids: list[object]) -> pd.DataFrame:
        """Query only soma node rows (type == 1) for the given files.

        Prefers a DuckDB-side ``type = 1`` filter so only soma rows are
        materialized. Falls back to loading every node and filtering in
        pandas when the database lacks the soma-scoped query.
        """
        db = self._database_provider()
        if db is None:
            raise RuntimeError("Load a neuron Parquet before projecting to flatmap.")
        if not file_ids:
            raise RuntimeError("No neurons are available to project.")

        getter = getattr(db, "get_soma_nodes_for_rendering", None)
        if callable(getter):
            nodes = getter(file_ids)
        else:
            nodes = self._query_nodes(file_ids)
            nodes = nodes[nodes["type"] == NodeType.SOMA]
        if nodes is None or nodes.empty:
            raise RuntimeError(
                "No soma nodes (type == 1) were found for the selected neurons."
            )
        return nodes

    def _lookup_files_ready(self) -> bool:
        return self._flatmap_path is not None and self._depth_path is not None

    def _projection_request_ready(self) -> None:
        if self._flatmap_path is None:
            raise RuntimeError("Choose a flatmap NRRD file before this action.")
        if self._depth_path is None:
            raise RuntimeError("Choose depth.nrrd before this action.")

    # Either column family is enough to project without the lookup NRRDs: the
    # legacy single-style set, or the version-3 bilateral shaped/square set.
    _FLATMAP_COLUMN_FAMILIES = (
        ("x_flat", "y_flat", "depth_um"),
        (
            "x_flat_shaped",
            "y_flat_shaped",
            "x_flat_square",
            "y_flat_square",
            "depth_um",
        ),
    )

    @classmethod
    def _has_parquet_flatmap_depth_columns(cls, nodes: pd.DataFrame) -> bool:
        names = set(nodes.columns)
        return any(
            set(family).issubset(names) for family in cls._FLATMAP_COLUMN_FAMILIES
        )

    def _loaded_parquet_has_flatmap_columns(self) -> bool:
        """Report whether the loaded Parquet already carries flatmap columns.

        Reads the database schema rather than querying rows, so it is cheap
        enough to call whenever a Parquet is loaded.
        """
        db = self._database_provider()
        has_column = getattr(db, "has_column", None)
        if not callable(has_column):
            return False
        for family in self._FLATMAP_COLUMN_FAMILIES:
            try:
                if all(has_column(name) for name in family):
                    return True
            except Exception:
                logger.debug(
                    "Failed to inspect the loaded Parquet for flatmap columns.",
                    exc_info=True,
                )
                return False
        return False

    def _update_lookup_files_section(self) -> None:
        """Collapse the lookup-file controls when the Parquet does not need them."""
        section = getattr(self, "_lookup_files_section", None)
        set_expanded = getattr(section, "set_expanded", None)
        if not callable(set_expanded):
            return
        set_expanded(not self._loaded_parquet_has_flatmap_columns())

    def _resolve_projection_plan(self, projection_source: str) -> tuple[bool, int]:
        """Decide whether to use lookup NRRDs and how many progress steps."""
        if projection_source == _PROJECTION_SOURCE_RECOMPUTE:
            self._projection_request_ready()
            use_lookup_files = True
        elif projection_source == _PROJECTION_SOURCE_PRECOMPUTED:
            use_lookup_files = False
        else:
            use_lookup_files = self._lookup_files_ready()
        total_steps = 6 if use_lookup_files else 4
        return use_lookup_files, total_steps

    def _build_projection_artifacts(
        self,
        nodes: pd.DataFrame,
        use_lookup_files: bool,
        *,
        projection_source: str,
        total_steps: int,
    ) -> tuple[
        FlatmapProjectionResult,
        FlatmapRenderResult | AllenLayerStackResult,
        FlatmapLookupStats | None,
        str,
        str,
        str,
    ]:
        """Project queried nodes and build render data for the current mode."""
        if use_lookup_files:
            result, render_result, lookup_stats = self._project_from_lookup_files(
                nodes,
                progress_callback=self._set_projection_progress,
                progress_total=total_steps,
            )
            flatmap_style = self._current_style_filename()
            coordinate_mode = self._current_coordinate_mode()
            source_note = "lookup NRRDs"
        else:
            if not self._has_parquet_flatmap_depth_columns(nodes):
                if projection_source == "legacy_auto":
                    raise RuntimeError(
                        "Choose both flatmap and depth NRRD files, or load an "
                        "augmented Parquet with x_flat, y_flat, and depth_um "
                        "columns."
                    )
                raise RuntimeError(
                    "Precomputed viewing requires a version-3 Parquet with "
                    "bilateral shaped/square flatmap and depth columns. "
                    "Choose Recompute from NRRDs explicitly to use lookup files."
                )
            if projection_source == _PROJECTION_SOURCE_PRECOMPUTED:
                self._validate_precomputed_parquet_contract(nodes)
            result, render_result, lookup_stats = self._project_from_parquet_columns(
                nodes,
                progress_callback=self._set_projection_progress,
                progress_total=total_steps,
            )
            flatmap_style = (
                "precomputed_parquet"
                if projection_source == "legacy_auto"
                else self._current_style_key()
            )
            coordinate_mode = "parquet_columns"
            source_note = "Parquet flatmap/depth columns"
        return (
            result,
            render_result,
            lookup_stats,
            flatmap_style,
            coordinate_mode,
            source_note,
        )

    def _add_soma(self) -> None:
        """Project only soma nodes into a dedicated flatmap point layer.

        Mirrors the Data tab's "Add Soma Only" action but renders into the
        separate flatmap + depth viewer as an independent point layer that
        does not replace the main projection layer.  The somas are built for the
        *current* render mode so they land in the coordinate space the visible
        render uses -- depth bins, Allen layer planes, or a single flat plane.
        """
        if getattr(self, "_flatmap_render_mode_change_pending", False):
            message = (
                "Project the selected render mode before adding soma points, "
                "or change Render back to the displayed mode."
            )
            self._status_label.setText(message)
            show_warning(message)
            return

        projection_source = self._current_projection_source()
        # Soma projection always renders per-node points, so the DuckDB
        # heatmap fast path (which cannot filter by node type) is skipped.
        use_lookup_files, total_steps = self._resolve_projection_plan(projection_source)
        self._set_projection_controls_enabled(False)
        self._set_projection_progress(
            "Querying soma rows...",
            0,
            total_steps,
        )
        try:
            file_ids = self._file_ids_for_source()
            soma_nodes = self._query_soma_nodes(file_ids)
            if self._is_allen_layer_mode() and "region_id" not in soma_nodes.columns:
                # Falling back to depth bins here would put the somas in a
                # different coordinate space from the selected Allen render.
                raise RuntimeError(
                    "Add Soma in Allen Layer Heatmap mode requires a region_id "
                    "column so somas can be filtered to the selected layers. "
                    "Regenerate the Parquet with Allen region annotations, or "
                    "switch Render to a depth or 2D mode."
                )

            (
                result,
                render_result,
                _lookup_stats,
                _flatmap_style,
                _coordinate_mode,
                source_note,
            ) = self._build_projection_artifacts(
                soma_nodes,
                use_lookup_files,
                projection_source=projection_source,
                total_steps=total_steps,
            )

            self._set_projection_progress(
                "Updating soma layer...",
                total_steps - 1,
                total_steps,
            )
            layer = self._create_or_update_soma_layer(render_result, result.summary)
            if layer is None:
                raise RuntimeError("No soma nodes mapped into the flatmap render.")
            self._set_projection_progress("Done", total_steps, total_steps)
            self._status_label.setText(
                f"Rendered {render_result.summary.rendered_nodes:,} of "
                f"{render_result.summary.total_nodes:,} soma node(s) using "
                f"{source_note}."
            )
            show_info("Flatmap soma projection complete.")
        except Exception as exc:
            logger.exception("Flatmap soma projection failed")
            self._status_label.setText(f"Flatmap soma projection failed: {exc}")
            show_warning(f"Flatmap soma projection failed: {exc}")
        finally:
            self._hide_projection_progress()
            self._set_projection_controls_enabled(True)

    def _project_in_new_window(self, _checked: bool = False) -> None:
        """Render the current projection in another flatmap window."""
        self._project(new_window=True)

    def _project(self, _checked: bool = False, *, new_window: bool = False) -> None:
        """Run projection from the current UI state and render the layer."""
        if new_window:
            try:
                self._begin_new_display_viewer()
            except Exception as exc:
                logger.exception("Could not create another flatmap window")
                self._status_label.setText(f"Flatmap projection failed: {exc}")
                show_warning(f"Flatmap projection failed: {exc}")
                return
            self._flatmap_render_mode_change_pending = False
        elif getattr(self, "_flatmap_render_mode_change_pending", False):
            # The user chose to replace the active scene. Hide its incompatible
            # visuals now, then delete them after Qt has returned to its event
            # loop so the macOS OpenGL compositor cannot repaint freed buffers.
            self._defer_flatmap_grid_layer_removal()
            self._flatmap_render_mode_change_pending = False
        projection_source = self._current_projection_source()
        if projection_source == _PROJECTION_SOURCE_PRECOMPUTED and (
            self._current_render_mode()
            in {_RENDER_HEATMAP, _RENDER_FLAT_HEATMAP, _RENDER_ALLEN_LAYERS}
        ):
            # Fast path: bin the precomputed flatmap columns inside DuckDB on a
            # worker thread instead of loading every node into pandas.
            self._start_precomputed_heatmap_worker()
            return
        use_lookup_files, total_steps = self._resolve_projection_plan(projection_source)
        self._set_projection_controls_enabled(False)
        self._set_projection_progress(
            "Querying neuron rows...",
            0,
            total_steps,
        )
        try:
            file_ids = self._file_ids_for_source()
            nodes = self._query_nodes(file_ids)

            (
                result,
                render_result,
                lookup_stats,
                flatmap_style,
                coordinate_mode,
                source_note,
            ) = self._build_projection_artifacts(
                nodes,
                use_lookup_files,
                projection_source=projection_source,
                total_steps=total_steps,
            )

            self._set_projection_progress(
                "Updating flatmap layer...",
                total_steps - 1,
                total_steps,
            )
            self._apply_projection_result(
                result,
                render_result,
                flatmap_style=flatmap_style,
                coordinate_mode=coordinate_mode,
                lookup_stats=lookup_stats,
                input_file_ids=tuple(str(file_id) for file_id in file_ids),
            )
            self._set_projection_progress("Done", total_steps, total_steps)
            self._status_label.setText(
                f"Rendered {render_result.summary.rendered_nodes:,} of "
                f"{render_result.summary.total_nodes:,} projected node(s) using "
                f"{source_note}."
            )
            show_info("Flatmap projection complete.")
        except Exception as exc:
            logger.exception("Flatmap projection failed")
            self._notify_display_viewer_failed("projection_failed")
            self._status_label.setText(f"Flatmap projection failed: {exc}")
            show_warning(f"Flatmap projection failed: {exc}")
        finally:
            self._hide_projection_progress()
            self._set_projection_controls_enabled(True)

    def _precomputed_heatmap_is_running(self) -> bool:
        thread = getattr(self, "_precomputed_heatmap_thread", None)
        is_running = getattr(thread, "isRunning", None)
        return bool(callable(is_running) and is_running())

    @staticmethod
    def _precomputed_parquet_schema_frame(source_path: Path) -> pd.DataFrame:
        """Return an empty frame whose columns mirror the Parquet schema.

        The v3 contract validator only inspects column names, so reading the
        Parquet footer (instant, metadata-only) avoids loading any rows.
        """
        import pyarrow.parquet as pq

        schema = pq.read_schema(source_path)
        return pd.DataFrame(columns=list(schema.names))

    def _start_precomputed_heatmap_worker(self) -> None:
        """Bin the precomputed flatmap heatmap in DuckDB on a worker thread."""
        if self._precomputed_heatmap_is_running():
            return
        try:
            source_path = self._current_source_parquet_path()
            self._validate_precomputed_parquet_contract(
                self._precomputed_parquet_schema_frame(source_path)
            )
            layer_map = None
            if self._is_allen_layer_mode():
                schema_names = set(
                    self._precomputed_parquet_schema_frame(source_path).columns
                )
                if "region_id" not in schema_names:
                    raise RuntimeError(
                        "Allen layer rendering requires a region_id column. "
                        "Regenerate the Parquet with Allen region annotations."
                    )
                layer_map = self._current_allen_layer_map()
            file_ids = self._file_ids_for_source()
            if not file_ids:
                raise RuntimeError("No neurons are available to project.")
            bounds = self._canonical_render_bounds()
        except Exception as exc:
            logger.exception("Flatmap heatmap projection failed to start")
            self._notify_display_viewer_failed("projection_failed")
            self._status_label.setText(f"Flatmap projection failed: {exc}")
            show_warning(f"Flatmap projection failed: {exc}")
            return

        from qtpy.QtCore import QThread

        from ..workers import FlatmapHeatmapWorker

        color_mode = self._current_heatmap_color_mode()
        cluster_map = (
            dict(self._cluster_map_provider() or {})
            if color_mode == _HEATMAP_COLOR_CLUSTER
            else None
        )
        self._precomputed_heatmap_file_ids = [str(file_id) for file_id in file_ids]
        self._precomputed_heatmap_display_generation = self._display_generation()
        self._precomputed_heatmap_render_config = (
            self._current_render_mode(),
            self._current_style_key(),
            self._current_allen_layer_indices(),
            self._current_allen_layer_output_mode(),
        )
        worker = FlatmapHeatmapWorker(
            str(source_path),
            style_key=self._current_style_key(),
            color_mode=color_mode,
            x_bounds=bounds.get("x_bounds"),
            y_bounds=bounds.get("y_bounds"),
            depth_range_um=bounds.get("depth_range_um"),
            y_bins=self._current_y_bins(),
            x_bins=self._current_x_bins(),
            depth_bin_um=self._current_depth_bin_um(),
            include_depth_minus_one=(not self._exclude_depth_minus_one_cb.isChecked()),
            file_ids=list(file_ids),
            cluster_map=cluster_map,
            plane_mode=self._current_plane_mode(),
            allen_layer_map=layer_map,
            allen_layer_indices=self._current_allen_layer_indices(),
            allen_layer_output_mode=self._current_allen_layer_output_mode(),
        )
        thread = QThread()
        self._precomputed_heatmap_thread = thread
        self._precomputed_heatmap_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._set_projection_progress)
        worker.finished.connect(self._on_precomputed_heatmap_finished)
        worker.error.connect(self._on_precomputed_heatmap_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda: self._cleanup_precomputed_heatmap(thread, worker)
        )
        self._set_projection_controls_enabled(False)
        self._set_projection_progress("Querying flatmap heatmap...", 0, 3)
        thread.start()

    def _on_precomputed_heatmap_finished(self, result) -> None:
        if not self._display_generation_matches(
            getattr(self, "_precomputed_heatmap_display_generation", None)
        ):
            self._status_label.setText(
                "Flatmap projection finished after its window changed or closed; "
                "run it again to display the result."
            )
            return
        current_config = (
            self._current_render_mode(),
            self._current_style_key(),
            self._current_allen_layer_indices(),
            self._current_allen_layer_output_mode(),
        )
        if current_config != getattr(
            self,
            "_precomputed_heatmap_render_config",
            current_config,
        ):
            self._status_label.setText(
                "Flatmap projection finished after its render options changed; "
                "run it again to display the result."
            )
            return
        try:
            if isinstance(result, AllenLayerHeatmapVolumeResult):
                self._apply_precomputed_allen_layer_result(result)
                rendered_nodes = result.summary.rendered_nodes
            else:
                self._apply_precomputed_heatmap_result(result)
                rendered_nodes = result.render_summary.rendered_nodes
            self._set_projection_progress("Done", 3, 3)
            self._status_label.setText(
                f"Rendered {rendered_nodes:,} of "
                f"{result.stats.total_nodes:,} projected node(s) using "
                "precomputed Parquet columns (DuckDB)."
            )
            show_info("Flatmap projection complete.")
        except Exception as exc:
            logger.exception("Flatmap heatmap render failed")
            self._notify_display_viewer_failed("projection_failed")
            self._status_label.setText(f"Flatmap projection failed: {exc}")
            show_warning(f"Flatmap projection failed: {exc}")

    def _on_precomputed_heatmap_error(self, message: str) -> None:
        if not self._display_generation_matches(
            getattr(self, "_precomputed_heatmap_display_generation", None)
        ):
            return
        logger.error("Flatmap heatmap pipeline error: %s", message)
        self._notify_display_viewer_failed("projection_failed")
        self._status_label.setText(f"Flatmap projection failed: {message}")
        show_warning(f"Flatmap projection failed: {message}")

    def _cleanup_precomputed_heatmap(self, thread, worker) -> None:
        if getattr(self, "_precomputed_heatmap_thread", None) is thread:
            self._precomputed_heatmap_thread = None
        if getattr(self, "_precomputed_heatmap_worker", None) is worker:
            self._precomputed_heatmap_worker = None
            self._precomputed_heatmap_display_generation = None
            self._precomputed_heatmap_render_config = None
        self._hide_projection_progress()
        self._set_projection_controls_enabled(True)

    @staticmethod
    def _projection_summary_from_stats(stats) -> ProjectionSummary:
        """Build a display-only ProjectionSummary from DuckDB aggregate counts."""
        total_nodes = int(stats.total_nodes)
        rendered = int(stats.rendered_nodes)
        return ProjectionSummary(
            total_nodes=total_nodes,
            valid_nodes=rendered,
            out_of_bounds_nodes=0,
            invalid_flatmap_nodes=int(total_nodes - stats.flatmap_valid_nodes),
            invalid_depth_nodes=int(
                stats.flatmap_valid_nodes - stats.depth_valid_nodes
            ),
            missing_input_nodes=0,
            rendered_segments=0,
            total_traces=int(stats.total_traces),
            traces_with_partial_projection=0,
            direct_lookup_nodes=rendered,
            mirrored_lookup_nodes=0,
            unmapped_lookup_nodes=int(total_nodes - rendered),
            mirrored_depth_lookup_nodes=0,
        )

    @staticmethod
    def _projection_summary_from_allen_layer_stats(
        stats: object,
    ) -> ProjectionSummary:
        total_nodes = int(getattr(stats, "total_nodes", 0))
        rendered = int(getattr(stats, "rendered_nodes", 0))
        flatmap_valid = int(getattr(stats, "flatmap_valid_nodes", 0))
        return ProjectionSummary(
            total_nodes=total_nodes,
            valid_nodes=rendered,
            out_of_bounds_nodes=0,
            invalid_flatmap_nodes=max(0, total_nodes - flatmap_valid),
            invalid_depth_nodes=0,
            missing_input_nodes=0,
            rendered_segments=0,
            total_traces=int(getattr(stats, "total_traces", 0)),
            traces_with_partial_projection=0,
            direct_lookup_nodes=rendered,
            mirrored_lookup_nodes=0,
            unmapped_lookup_nodes=max(0, total_nodes - rendered),
            mirrored_depth_lookup_nodes=0,
        )

    def _apply_precomputed_heatmap_result(self, result) -> None:
        """Render a DuckDB-built flatmap heatmap and record volume-only state.

        The same worker result type serves the depth-binned and the
        depth-collapsed heatmap; the volume's rank is what differs, so the
        current render mode decides the layer name, axis captions, and display
        dimensionality.
        """
        render_summary = result.render_summary
        projection_summary = self._projection_summary_from_stats(result.stats)
        color_mode = result.color_mode
        style_key = self._current_style_key()
        render_mode = (
            _RENDER_FLAT_HEATMAP if self._is_flat_render_mode() else _RENDER_HEATMAP
        )

        # The fast path never materializes a per-node table.  Clearing it keeps
        # the per-node features (Export CSV, flatmap correlation, region-mask
        # projection) disabled until a per-node projection is run instead.
        self._last_projected_nodes = None
        self._last_summary = projection_summary
        self._last_render_summary = render_summary
        self._last_render_mode = render_mode
        self._last_flatmap_style = style_key
        self._last_coordinate_mode = "parquet_columns"
        self._last_volume_shape = tuple(int(size) for size in result.volume_shape)
        self._last_lookup_stats = None
        self._last_input_file_ids = tuple(
            getattr(self, "_precomputed_heatmap_file_ids", ()) or ()
        )
        self._last_flatmap_path = (
            str(self._flatmap_path) if self._flatmap_path else None
        )
        self._last_depth_path = str(self._depth_path) if self._depth_path else None
        self._last_projection_source = _PROJECTION_SOURCE_PRECOMPUTED
        active_profile = getattr(self, "_active_cache_profile", None)
        self._last_cache_dir = (
            str(self._region_cache_dir) if self._region_cache_dir else None
        )
        self._last_cache_profile_id = self._cache_profile_id(active_profile)

        self._summary_label.setText(
            self._format_flat_render_summary(projection_summary, render_summary)
            if render_mode == _RENDER_FLAT_HEATMAP
            else self._format_render_summary(projection_summary, render_summary)
        )

        if render_summary.rendered_nodes == 0:
            self._notify_display_viewer_failed("no_render_layer")
        else:
            metadata = self._render_metadata(
                projection_summary,
                render_summary,
                flatmap_style=self._current_style_filename(),
                coordinate_mode="parquet_columns",
                render_mode=render_mode,
                heatmap_color_mode=color_mode,
            )
            layer = self._render_precomputed_heatmap_layers(
                result,
                metadata,
                render_mode=render_mode,
            )
            if layer is None:
                self._notify_display_viewer_failed("no_render_layer")

        # Per-node features are unavailable for a volume-only fast render.
        self._export_btn.setEnabled(False)
        self._notify_flatmap_correlation_source_changed()

    def _apply_precomputed_allen_layer_result(
        self,
        result: AllenLayerHeatmapVolumeResult,
    ) -> None:
        summary = result.summary
        projection_summary = self._projection_summary_from_allen_layer_stats(
            result.stats
        )
        style_key = self._current_style_key()
        self._last_projected_nodes = None
        self._last_summary = projection_summary
        self._last_render_summary = summary
        self._last_render_mode = _RENDER_ALLEN_LAYERS
        self._last_flatmap_style = style_key
        self._last_coordinate_mode = "parquet_columns"
        self._last_volume_shape = tuple(int(size) for size in result.volume_shape)
        self._last_lookup_stats = None
        self._last_input_file_ids = tuple(
            getattr(self, "_precomputed_heatmap_file_ids", ()) or ()
        )
        self._last_flatmap_path = (
            str(self._flatmap_path) if self._flatmap_path else None
        )
        self._last_depth_path = str(self._depth_path) if self._depth_path else None
        self._last_projection_source = _PROJECTION_SOURCE_PRECOMPUTED
        active_profile = getattr(self, "_active_cache_profile", None)
        self._last_cache_dir = (
            str(self._region_cache_dir) if self._region_cache_dir else None
        )
        self._last_cache_profile_id = self._cache_profile_id(active_profile)
        self._summary_label.setText(
            self._format_allen_layer_summary(projection_summary, summary)
        )

        if summary.rendered_nodes == 0:
            self._notify_display_viewer_failed("no_render_layer")
            raise RuntimeError(
                "No selected flatmap-valid nodes belong to a terminal Allen "
                "Isocortex layer."
            )
        else:
            metadata = self._render_metadata(
                projection_summary,
                summary,
                flatmap_style=self._current_style_filename(),
                coordinate_mode="parquet_columns",
                render_mode=_RENDER_ALLEN_LAYERS,
                heatmap_color_mode=result.color_mode,
            )
            layer = self._render_precomputed_allen_layer_layers(
                result,
                metadata,
            )
            if layer is None:
                self._notify_display_viewer_failed("no_render_layer")
        self._export_btn.setEnabled(False)
        self._notify_flatmap_correlation_source_changed()

    def _render_precomputed_allen_layer_layers(
        self,
        result: AllenLayerHeatmapVolumeResult,
        metadata: dict[str, object],
    ):
        axis_labels = self._allen_layer_axis_labels_for_summary(result.summary)
        layer_name = self._allen_layer_name_for_summary(result.summary)
        if result.color_mode == _HEATMAP_COLOR_SINGLE:
            self._remove_projection_layer(except_name=layer_name)
            layer = self._cached_projection_layer_for_name(
                layer_name
            ) or self._find_layer_by_name(layer_name)
            layer = self._create_or_update_heatmap_layer_from_volume(
                layer,
                result.volume,
                metadata,
                layer_name=layer_name,
                axis_labels=axis_labels,
            )
            self._projection_layer = layer
            if layer is None:
                return None
            self._set_layer_state(
                layer,
                None,
                self._last_summary,
                result.summary,
            )
            self._focus_projection_view(
                layer,
                result.volume,
                ndisplay=2,
            )
            self._notify_display_viewer_ready(layer)
            return layer

        self._remove_projection_layer()
        first_layer = None
        focus_volume = None
        for group in result.grouped_volumes:
            color = self._color_for_heatmap_group(
                group,
                heatmap_color_mode=result.color_mode,
            )
            layer = self._add_grouped_heatmap_layer(
                group,
                metadata,
                color,
                heatmap_color_mode=result.color_mode,
                render_mode=_RENDER_ALLEN_LAYERS,
                axis_labels=axis_labels,
            )
            self._set_layer_state(
                layer,
                None,
                self._last_summary,
                result.summary,
            )
            if first_layer is None:
                first_layer = layer
                focus_volume = group.volume
        self._projection_layer = first_layer
        if first_layer is None:
            return None
        self._focus_projection_view(
            first_layer,
            focus_volume,
            ndisplay=2,
        )
        self._notify_display_viewer_ready(first_layer)
        return first_layer

    def _render_precomputed_heatmap_layers(
        self,
        result,
        metadata: dict[str, object],
        *,
        render_mode: str = _RENDER_HEATMAP,
    ):
        """Create/update napari image layers from DuckDB-built volume(s)."""
        color_mode = result.color_mode
        layer_name = self._render_layer_name(render_mode)
        axis_labels = self._axis_labels_for_render_mode(render_mode)
        ndisplay = self._render_ndisplay(render_mode)
        if color_mode == _HEATMAP_COLOR_SINGLE:
            self._remove_projection_layer(except_name=layer_name)
            layer = self._cached_projection_layer_for_name(
                layer_name
            ) or self._find_layer_by_name(layer_name)
            layer = self._create_or_update_heatmap_layer_from_volume(
                layer,
                result.volume,
                metadata,
                layer_name=layer_name,
                axis_labels=axis_labels,
            )
            self._projection_layer = layer
            if layer is None:
                return None
            self._set_layer_state(
                layer,
                None,
                self._last_summary,
                result.render_summary,
            )
            self._focus_projection_view(layer, result.volume, ndisplay=ndisplay)
            self._notify_display_viewer_ready(layer)
            return layer

        # Grouped (individual / cluster) coloring: one image layer per group.
        self._remove_projection_layer()
        first_layer = None
        focus_volume = None
        for group in result.grouped_volumes:
            color = self._color_for_heatmap_group(
                group,
                heatmap_color_mode=color_mode,
            )
            layer = self._add_grouped_heatmap_layer(
                group,
                metadata,
                color,
                heatmap_color_mode=color_mode,
                render_mode=render_mode,
                axis_labels=axis_labels,
            )
            self._set_layer_state(
                layer,
                None,
                self._last_summary,
                result.render_summary,
            )
            if first_layer is None:
                first_layer = layer
                focus_volume = group.volume
        self._projection_layer = first_layer
        if first_layer is None:
            return None
        self._focus_projection_view(first_layer, focus_volume, ndisplay=ndisplay)
        self._notify_display_viewer_ready(first_layer)
        return first_layer

    def _validate_precomputed_parquet_contract(self, nodes: pd.DataFrame) -> None:
        """Reject partial/corrupt v3 data before fixed-grid rendering."""
        names = set(nodes.columns)
        v3_markers = {
            column
            for mapping in FLATMAP_V3_STYLE_COLUMN_MAPPING.values()
            for column in mapping.values()
        }
        if not names.intersection(v3_markers):
            # Complete legacy x_flat/y_flat/depth_um files remain usable for
            # neuron-only rendering with their historical subset bounds.
            return
        missing = sorted(set(FLATMAP_V3_AUGMENTED_COLUMNS).difference(names))
        if missing:
            raise RuntimeError(
                "Version-3 Parquet is missing required flatmap column(s): "
                f"{missing}. Regenerate it with Prepare Whole Parquet."
            )
        style = self._current_style_key()
        if style not in FLATMAP_V3_STYLE_COLUMN_MAPPING:
            raise RuntimeError(
                "Version-3 precomputed viewing supports only bilateral shaped "
                "and bilateral square styles. Choose Recompute from NRRDs for "
                "a unilateral style."
            )
        info = read_flatmap_parquet_transform_info(self._current_source_parquet_path())
        if info.format_version < 3 or not info.lookup_set_id:
            raise RuntimeError(
                "Bilateral flatmap columns require complete version-3 metadata "
                "with a lookup-set ID. Regenerate the Parquet."
            )
        metadata = info.metadata
        bounds = (
            self._bounds_from_projection_metadata(metadata, style)
            if isinstance(metadata, dict)
            else None
        )
        if bounds is None:
            raise RuntimeError(
                f"Version-3 Parquet has no valid canonical bounds for {style}. "
                "Regenerate the Parquet instead of deriving bounds from this query."
            )

    def _project_from_lookup_files(
        self,
        nodes: pd.DataFrame,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
        progress_total: int = 6,
    ) -> tuple[
        FlatmapProjectionResult,
        FlatmapRenderResult | AllenLayerStackResult,
        FlatmapLookupStats,
    ]:
        use_allen_layers = self._is_allen_layer_mode()
        if use_allen_layers and "region_id" not in nodes.columns:
            raise RuntimeError(
                "Allen layer rendering requires a region_id column. "
                "Regenerate the Parquet with Allen region annotations."
            )
        self._emit_projection_progress(
            progress_callback,
            "Loading flatmap lookup files...",
            1,
            progress_total,
        )
        volume_set = load_flatmap_volume_set(self._flatmap_path, self._depth_path)
        self._emit_projection_progress(
            progress_callback,
            "Computing flatmap lookup statistics...",
            2,
            progress_total,
        )
        lookup_stats = self._lookup_stats_for_volume_set(
            volume_set,
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=(self._negative_one_sentinel_cb.isChecked()),
        )
        self._emit_projection_progress(
            progress_callback,
            "Projecting nodes into flatmap space...",
            3,
            progress_total,
        )
        result = project_and_build_segments(
            nodes,
            volume_set.flatmap,
            volume_set.depth,
            flatmap_style=self._current_style_filename(),
            coordinate_mode=self._current_coordinate_mode(),
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=(self._negative_one_sentinel_cb.isChecked()),
            resolution_um=DEFAULT_CCF_RESOLUTION_UM,
            space_directions=volume_set.space_directions,
            space_origin=volume_set.space_origin,
            mirror_fallback=True,
        )
        self._emit_projection_progress(
            progress_callback,
            "Building flatmap render data...",
            4,
            progress_total,
        )
        if use_allen_layers:
            render_result = build_allen_layer_render_from_projected_nodes(
                result.projected_nodes,
                self._current_allen_layer_map(),
                y_bins=self._current_y_bins(),
                x_bins=self._current_x_bins(),
                x_bounds=lookup_stats.x_bounds,
                y_bounds=lookup_stats.y_bounds,
                selected_layer_indices=self._current_allen_layer_indices(),
                output_mode=self._current_allen_layer_output_mode(),
            )
        else:
            render_result = build_flatmap_render_data(
                result.projected_nodes,
                volume_set.flatmap,
                volume_set.depth,
                y_bins=self._current_y_bins(),
                x_bins=self._current_x_bins(),
                depth_bin_um=self._current_depth_bin_um(),
                include_depth_minus_one=(
                    not self._exclude_depth_minus_one_cb.isChecked()
                ),
                invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
                invalid_negative_one_sentinel=(
                    self._negative_one_sentinel_cb.isChecked()
                ),
                lookup_stats=lookup_stats,
                collapse_depth=self._is_flat_render_mode(),
            )
        return result, render_result, lookup_stats

    def _project_from_parquet_columns(
        self,
        nodes: pd.DataFrame,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
        progress_total: int = 4,
    ) -> tuple[
        FlatmapProjectionResult,
        FlatmapRenderResult | AllenLayerStackResult,
        None,
    ]:
        self._emit_projection_progress(
            progress_callback,
            "Reading precomputed flatmap columns...",
            1,
            progress_total,
        )
        result = self._projection_result_from_parquet_columns(nodes)
        self._emit_projection_progress(
            progress_callback,
            "Building flatmap render data...",
            2,
            progress_total,
        )
        canonical_bounds = self._canonical_render_bounds()
        if self._is_allen_layer_mode():
            if "region_id" not in nodes.columns:
                raise RuntimeError(
                    "Allen layer rendering requires a region_id column. "
                    "Regenerate the Parquet with Allen region annotations."
                )
            x_bounds = canonical_bounds.get("x_bounds")
            y_bounds = canonical_bounds.get("y_bounds")
            if x_bounds is None or y_bounds is None:
                raise RuntimeError(
                    "Allen layer rendering requires canonical flatmap XY "
                    "bounds from a version-3 Parquet or cache profile."
                )
            render_result = build_allen_layer_render_from_projected_nodes(
                result.projected_nodes,
                self._current_allen_layer_map(),
                y_bins=self._current_y_bins(),
                x_bins=self._current_x_bins(),
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                selected_layer_indices=self._current_allen_layer_indices(),
                output_mode=self._current_allen_layer_output_mode(),
            )
        else:
            render_result = build_flatmap_render_data_from_projected_nodes(
                result.projected_nodes,
                y_bins=self._current_y_bins(),
                x_bins=self._current_x_bins(),
                depth_bin_um=self._current_depth_bin_um(),
                include_depth_minus_one=(
                    not self._exclude_depth_minus_one_cb.isChecked()
                ),
                collapse_depth=self._is_flat_render_mode(),
                **canonical_bounds,
            )
        return result, render_result, None

    def _canonical_render_bounds(self) -> dict[str, tuple[float, float]]:
        """Return canonical style/depth bounds from the cache or v3 Parquet."""
        profile = getattr(self, "_active_cache_profile", None)
        if profile is not None:
            bounds = self._cache_profile_bounds(profile, self._current_style_key())
            if bounds is not None:
                return bounds

        try:
            info = read_flatmap_parquet_transform_info(
                self._current_source_parquet_path()
            )
        except Exception:
            logger.debug("Could not inspect flatmap Parquet bounds", exc_info=True)
            return {}
        metadata = info.metadata
        if not isinstance(metadata, dict):
            return {}
        return (
            self._bounds_from_projection_metadata(
                metadata,
                self._current_style_key(),
            )
            or {}
        )

    @staticmethod
    def _bounds_from_projection_metadata(
        metadata: dict[str, object],
        style: str,
    ) -> dict[str, tuple[float, float]] | None:
        canonical = metadata.get("canonical_bounds")
        if not isinstance(canonical, dict):
            return None
        style_bounds = canonical.get(style)
        if not isinstance(style_bounds, dict):
            return None
        try:
            x_values = tuple(float(value) for value in style_bounds["x"])
            y_values = tuple(float(value) for value in style_bounds["y"])
            depth_values = tuple(float(value) for value in style_bounds["depth_um"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(len(values) == 2 for values in (x_values, y_values, depth_values)):
            return None
        return {
            "x_bounds": (x_values[0], x_values[1]),
            "y_bounds": (y_values[0], y_values[1]),
            "depth_range_um": (depth_values[0], depth_values[1]),
        }

    @staticmethod
    def _emit_projection_progress(
        progress_callback: Callable[[str, int, int], None] | None,
        message: str,
        current: int,
        total: int,
    ) -> None:
        if progress_callback is not None:
            progress_callback(message, current, total)

    def _set_projection_controls_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled and self._allen_layer_selection_ready())
        for name in ("_project_btn", "_new_window_btn"):
            button = getattr(self, name, None)
            set_enabled = getattr(button, "setEnabled", None)
            if callable(set_enabled):
                set_enabled(enabled)
        add_soma = getattr(self, "_add_soma_btn", None)
        set_add_soma_enabled = getattr(add_soma, "setEnabled", None)
        if callable(set_add_soma_enabled):
            set_add_soma_enabled(
                enabled
                and not bool(
                    getattr(self, "_flatmap_render_mode_change_pending", False)
                )
            )

    def _set_projection_progress(
        self,
        message: str,
        current: int,
        total: int,
    ) -> None:
        status_label = getattr(self, "_status_label", None)
        if status_label is not None:
            status_label.setText(str(message))

        progress_bar = getattr(self, "_projection_progress_bar", None)
        if progress_bar is not None:
            set_visible = getattr(progress_bar, "setVisible", None)
            if callable(set_visible):
                set_visible(True)
            if int(total) > 0:
                maximum = int(total)
                value = max(0, min(int(current), maximum))
                set_range = getattr(progress_bar, "setRange", None)
                if callable(set_range):
                    set_range(0, maximum)
                set_value = getattr(progress_bar, "setValue", None)
                if callable(set_value):
                    set_value(value)
            else:
                set_range = getattr(progress_bar, "setRange", None)
                if callable(set_range):
                    set_range(0, 0)
        self._flush_projection_progress_updates()

    def _hide_projection_progress(self) -> None:
        progress_bar = getattr(self, "_projection_progress_bar", None)
        if progress_bar is None:
            return
        set_range = getattr(progress_bar, "setRange", None)
        if callable(set_range):
            set_range(0, 1)
        set_value = getattr(progress_bar, "setValue", None)
        if callable(set_value):
            set_value(0)
        set_visible = getattr(progress_bar, "setVisible", None)
        if callable(set_visible):
            set_visible(False)

    @staticmethod
    def _flush_projection_progress_updates() -> None:
        try:
            from qtpy.QtWidgets import QApplication
        except ImportError:
            return

        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _projection_result_from_parquet_columns(
        self,
        nodes: pd.DataFrame,
    ) -> FlatmapProjectionResult:
        source = self._normalise_precomputed_style_columns(nodes.reset_index(drop=True))
        missing = [
            column
            for column in ("x_flat", "y_flat", "depth_um")
            if column not in source.columns
        ]
        if missing:
            raise RuntimeError(
                f"Loaded Parquet is missing reusable flatmap/depth column(s): {missing}"
            )

        x_flat = pd.to_numeric(source["x_flat"], errors="coerce").to_numpy(dtype=float)
        y_flat = pd.to_numeric(source["y_flat"], errors="coerce").to_numpy(dtype=float)
        depth_um = pd.to_numeric(source["depth_um"], errors="coerce").to_numpy(
            dtype=float
        )
        flatmap_valid = self._bool_column_or_default(
            source,
            "flatmap_valid",
            np.isfinite(x_flat) & np.isfinite(y_flat),
        )
        depth_valid = self._bool_column_or_default(
            source,
            "depth_valid",
            np.isfinite(depth_um) & (depth_um >= 0.0),
        )
        if "flatmap_projection_valid" in source.columns:
            valid = (
                source["flatmap_projection_valid"].fillna(False).astype(bool).to_numpy()
            )
        elif "valid" in source.columns:
            valid = source["valid"].fillna(False).astype(bool).to_numpy()
        else:
            valid = flatmap_valid & depth_valid

        invalid_reason = self._parquet_invalid_reasons(
            source,
            flatmap_valid=flatmap_valid,
            depth_valid=depth_valid,
            valid=valid,
        )
        if "flatmap_lookup_mode" in source.columns:
            lookup_mode = source["flatmap_lookup_mode"].fillna("").astype(str)
            lookup_mode = lookup_mode.reset_index(drop=True)
            lookup_mode = lookup_mode.where(
                lookup_mode.isin(
                    [
                        FLATMAP_LOOKUP_DIRECT,
                        FLATMAP_LOOKUP_MIRRORED_DEPTH,
                        FLATMAP_LOOKUP_MIRRORED,
                        FLATMAP_LOOKUP_UNMAPPED,
                    ]
                ),
                np.where(
                    valid,
                    FLATMAP_LOOKUP_DIRECT,
                    FLATMAP_LOOKUP_UNMAPPED,
                ),
            )
        else:
            lookup_mode = pd.Series(
                np.where(
                    valid,
                    FLATMAP_LOOKUP_DIRECT,
                    FLATMAP_LOOKUP_UNMAPPED,
                ),
                index=range(len(source)),
            )

        projected = pd.DataFrame(
            {
                "file_id": source["file_id"].reset_index(drop=True),
                "neuron_id": self._column_or_default(source, "neuron_id", ""),
                "subject": self._column_or_default(source, "subject", ""),
                "node_id": source["node_id"].reset_index(drop=True),
                "parent_id": source["parent_id"].reset_index(drop=True),
                "type": source["type"].reset_index(drop=True),
                "x_um": pd.to_numeric(source["x"], errors="coerce"),
                "y_um": pd.to_numeric(source["y"], errors="coerce"),
                "z_um": pd.to_numeric(source["z"], errors="coerce"),
                "voxel_i": self._column_or_default(source, "voxel_i", pd.NA),
                "voxel_j": self._column_or_default(source, "voxel_j", pd.NA),
                "voxel_k": self._column_or_default(source, "voxel_k", pd.NA),
                "x_flat": x_flat,
                "y_flat": y_flat,
                "depth_um": depth_um,
                "flatmap_valid": flatmap_valid,
                "depth_valid": depth_valid,
                "valid": valid,
                "invalid_reason": invalid_reason,
                "region_id": self._column_or_default(source, "region_id", pd.NA),
                "region_acronym": self._column_or_default(
                    source,
                    "region_acronym",
                    "",
                ),
                "flatmap_style": "precomputed_parquet",
                "coordinate_mode": "parquet_columns",
                "flatmap_lookup_mode": lookup_mode,
            }
        )

        segments = build_projected_segments(projected)
        summary = summarize_projection(projected, segments)
        return FlatmapProjectionResult(projected, segments, summary)

    def _normalise_precomputed_style_columns(
        self,
        source: pd.DataFrame,
    ) -> pd.DataFrame:
        """Map the selected v3 style columns onto the renderer's generic names."""
        style_key = self._current_style_key()
        if style_key == "both_shaped":
            suffix = "shaped"
        elif style_key == "both_square":
            suffix = "square"
        else:
            raise RuntimeError(
                "Version-3 precomputed coordinates support bilateral shaped "
                "and bilateral square styles only."
            )

        x_column = f"x_flat_{suffix}"
        y_column = f"y_flat_{suffix}"
        if not {x_column, y_column}.issubset(source.columns):
            if {"x_flat", "y_flat"}.issubset(source.columns):
                return source
        missing = [
            column
            for column in (x_column, y_column, "depth_um")
            if column not in source.columns
        ]
        if missing:
            raise RuntimeError(
                f"Loaded Parquet is missing version-3 precomputed column(s): {missing}"
            )

        normalised = source.copy()
        normalised["x_flat"] = normalised[x_column]
        normalised["y_flat"] = normalised[y_column]
        aliases = {
            f"flatmap_{suffix}_valid": "flatmap_valid",
            f"flatmap_{suffix}_projection_valid": "flatmap_projection_valid",
            f"flatmap_{suffix}_invalid_code": "flatmap_invalid_code",
            f"flatmap_{suffix}_lookup_mode": "flatmap_lookup_mode",
        }
        for style_column, generic_column in aliases.items():
            if style_column in normalised.columns:
                normalised[generic_column] = normalised[style_column]
        if "depth_lookup_mode" in normalised.columns:
            depth_modes = normalised["depth_lookup_mode"].fillna("").astype(str)
            combined_modes = (
                normalised.get("flatmap_lookup_mode", "")
                if "flatmap_lookup_mode" in normalised.columns
                else pd.Series([""] * len(normalised), index=normalised.index)
            )
            combined_modes = combined_modes.fillna("").astype(str).copy()
            # A recovered depth only makes a valid mirrored-depth projection
            # when the selected style's original-voxel XY lookup succeeded.
            # Keep an independently unmapped XY lookup unmapped.
            combined_modes.loc[
                (depth_modes == FLATMAP_LOOKUP_MIRRORED_DEPTH)
                & (combined_modes == FLATMAP_LOOKUP_DIRECT)
            ] = FLATMAP_LOOKUP_MIRRORED_DEPTH
            combined_modes.loc[depth_modes == FLATMAP_LOOKUP_UNMAPPED] = (
                FLATMAP_LOOKUP_UNMAPPED
            )
            normalised["flatmap_lookup_mode"] = combined_modes
        return normalised

    @staticmethod
    def _column_or_default(
        table: pd.DataFrame,
        column: str,
        default: object,
    ) -> pd.Series:
        if column in table.columns:
            return table[column].reset_index(drop=True)
        return pd.Series([default] * len(table), index=range(len(table)))

    @staticmethod
    def _bool_column_or_default(
        table: pd.DataFrame,
        column: str,
        default: np.ndarray,
    ) -> np.ndarray:
        if column in table.columns:
            return table[column].fillna(False).astype(bool).to_numpy()
        return np.asarray(default, dtype=bool)

    @staticmethod
    def _parquet_invalid_reasons(
        table: pd.DataFrame,
        *,
        flatmap_valid: np.ndarray,
        depth_valid: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        if "flatmap_invalid_code" in table.columns:
            reasons = [
                flatmap_invalid_code_to_reason(code)
                for code in table["flatmap_invalid_code"].tolist()
            ]
            return np.asarray(reasons, dtype=object)
        if "invalid_reason" in table.columns:
            return table["invalid_reason"].fillna("").astype(str).to_numpy()

        reasons = np.full(len(table), "", dtype=object)
        reasons[~flatmap_valid] = "invalid_flatmap"
        reasons[flatmap_valid & ~depth_valid] = "invalid_depth"
        reasons[valid] = ""
        return reasons

    @staticmethod
    def _path_signature(path: Path) -> tuple[str, int | None, int | None]:
        try:
            stat = path.stat()
        except OSError:
            return (str(path), None, None)
        return (str(path), int(stat.st_size), int(stat.st_mtime_ns))

    def _lookup_stats_cache_key_for(
        self,
        volume_set,
        *,
        invalid_zero_sentinel: bool,
        invalid_negative_one_sentinel: bool,
    ) -> tuple[object, ...]:
        return (
            self._path_signature(Path(volume_set.flatmap_path)),
            self._path_signature(Path(volume_set.depth_path)),
            bool(invalid_zero_sentinel),
            bool(invalid_negative_one_sentinel),
            tuple(volume_set.flatmap.shape),
            str(volume_set.flatmap.dtype),
            tuple(volume_set.depth.shape),
            str(volume_set.depth.dtype),
        )

    def _lookup_stats_for_volume_set(
        self,
        volume_set,
        *,
        invalid_zero_sentinel: bool,
        invalid_negative_one_sentinel: bool,
    ) -> FlatmapLookupStats:
        key = self._lookup_stats_cache_key_for(
            volume_set,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        )
        cached_key = getattr(self, "_lookup_stats_cache_key", None)
        cached_stats = getattr(self, "_lookup_stats_cache", None)
        if cached_key == key and cached_stats is not None:
            return cached_stats

        stats = compute_flatmap_lookup_stats(
            volume_set.flatmap,
            volume_set.depth,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        )
        self._lookup_stats_cache_key = key
        self._lookup_stats_cache = stats
        return stats

    def _apply_projection_result(
        self,
        result: FlatmapProjectionResult,
        render_result: FlatmapRenderResult | AllenLayerStackResult,
        *,
        flatmap_style: str | None = None,
        coordinate_mode: str | None = None,
        lookup_stats: FlatmapLookupStats | None = None,
        input_file_ids: tuple[str, ...] = (),
    ) -> None:
        if isinstance(render_result, AllenLayerStackResult):
            self._apply_allen_layer_projection_result(
                result,
                render_result,
                flatmap_style=flatmap_style,
                coordinate_mode=coordinate_mode,
                lookup_stats=lookup_stats,
                input_file_ids=input_file_ids,
            )
            return
        self._last_projected_nodes = render_result.projected_nodes
        self._last_summary = result.summary
        self._last_render_summary = render_result.summary
        self._last_render_mode = self._current_render_mode()
        self._last_flatmap_style = flatmap_style or self._current_style_filename()
        self._last_coordinate_mode = coordinate_mode or self._current_coordinate_mode()
        self._last_volume_shape = tuple(
            int(size) for size in render_result.volume.shape
        )
        self._last_lookup_stats = lookup_stats
        self._last_input_file_ids = tuple(input_file_ids)
        self._last_flatmap_path = (
            str(self._flatmap_path) if self._flatmap_path else None
        )
        self._last_depth_path = str(self._depth_path) if self._depth_path else None
        self._last_projection_source = self._current_projection_source()
        active_profile = getattr(self, "_active_cache_profile", None)
        if self._last_projection_source == _PROJECTION_SOURCE_PRECOMPUTED:
            self._last_cache_dir = (
                str(self._region_cache_dir) if self._region_cache_dir else None
            )
            self._last_cache_profile_id = self._cache_profile_id(active_profile)
        else:
            self._last_cache_dir = None
            self._last_cache_profile_id = None
        render_mode = self._current_render_mode()
        self._summary_label.setText(
            self._format_flat_render_summary(result.summary, render_result.summary)
            if render_mode in {_RENDER_FLAT_HEATMAP, _RENDER_FLAT_VECTOR}
            else self._format_render_summary(result.summary, render_result.summary)
        )
        layer = self._create_or_update_render_layer(
            render_result,
            result.summary,
            flatmap_style=flatmap_style or self._current_style_filename(),
            coordinate_mode=coordinate_mode or self._current_coordinate_mode(),
            render_mode=render_mode,
        )
        if layer is None:
            self._notify_display_viewer_failed("no_render_layer")
        self._export_btn.setEnabled(not render_result.projected_nodes.empty)
        self._notify_flatmap_correlation_source_changed()

    def _apply_allen_layer_projection_result(
        self,
        result: FlatmapProjectionResult,
        stack_result: AllenLayerStackResult,
        *,
        flatmap_style: str | None,
        coordinate_mode: str | None,
        lookup_stats: FlatmapLookupStats | None,
        input_file_ids: tuple[str, ...],
    ) -> None:
        self._last_projected_nodes = stack_result.projected_nodes
        self._last_summary = result.summary
        self._last_render_summary = stack_result.summary
        self._last_render_mode = _RENDER_ALLEN_LAYERS
        self._last_flatmap_style = flatmap_style or self._current_style_filename()
        self._last_coordinate_mode = coordinate_mode or self._current_coordinate_mode()
        self._last_volume_shape = tuple(int(size) for size in stack_result.volume.shape)
        self._last_lookup_stats = lookup_stats
        self._last_input_file_ids = tuple(input_file_ids)
        self._last_flatmap_path = (
            str(self._flatmap_path) if self._flatmap_path else None
        )
        self._last_depth_path = str(self._depth_path) if self._depth_path else None
        self._last_projection_source = self._current_projection_source()
        active_profile = getattr(self, "_active_cache_profile", None)
        if self._last_projection_source == _PROJECTION_SOURCE_PRECOMPUTED:
            self._last_cache_dir = (
                str(self._region_cache_dir) if self._region_cache_dir else None
            )
            self._last_cache_profile_id = self._cache_profile_id(active_profile)
        else:
            self._last_cache_dir = None
            self._last_cache_profile_id = None
        self._summary_label.setText(
            self._format_allen_layer_summary(
                result.summary,
                stack_result.summary,
            )
        )
        if stack_result.summary.rendered_nodes == 0:
            self._remove_projection_layer(create=False)
            raise RuntimeError(
                "No selected flatmap-valid nodes belong to a terminal Allen "
                "Isocortex layer."
            )
        layer = self._create_or_update_allen_layer_stack(
            stack_result,
            result.summary,
            flatmap_style=self._last_flatmap_style,
            coordinate_mode=self._last_coordinate_mode,
        )
        if layer is None:
            self._notify_display_viewer_failed("no_render_layer")
        self._export_btn.setEnabled(not stack_result.projected_nodes.empty)
        self._notify_flatmap_correlation_source_changed()

    def latest_flatmap_correlation_source(
        self,
    ) -> FlatmapVoxelCorrelationSource | None:
        """Return the latest heatmap render as a flatmap-clustering source."""
        if self._last_render_mode != _RENDER_HEATMAP:
            return None
        if not self._latest_heatmap_layer_is_rendered():
            return None
        projected_nodes = getattr(self, "_last_projected_nodes", None)
        render_summary = getattr(self, "_last_render_summary", None)
        volume_shape = getattr(self, "_last_volume_shape", None)
        if projected_nodes is None or render_summary is None or volume_shape is None:
            return None
        if projected_nodes.empty or int(render_summary.traces_represented) < 2:
            return None
        last_source = getattr(self, "_last_projection_source", None)
        if last_source == _PROJECTION_SOURCE_PRECOMPUTED:
            if self._current_style_key() != getattr(self, "_last_flatmap_style", None):
                return None
            current_profile_id = self._cache_profile_id(
                getattr(self, "_active_cache_profile", None)
            )
            if current_profile_id != str(
                getattr(self, "_last_cache_profile_id", "") or ""
            ):
                return None
        else:
            if (
                getattr(self, "_last_lookup_stats", None) is None
                or not self._last_flatmap_path
                or not self._last_depth_path
            ):
                return None
            if (
                self._flatmap_path is None
                or self._depth_path is None
                or str(self._flatmap_path) != self._last_flatmap_path
                or str(self._depth_path) != self._last_depth_path
            ):
                return None

        input_file_ids = tuple(getattr(self, "_last_input_file_ids", ()) or ())
        if not input_file_ids and "file_id" in projected_nodes.columns:
            input_file_ids = tuple(
                str(value)
                for value in self._deduplicate_file_ids(
                    projected_nodes["file_id"].tolist()
                )
            )

        return FlatmapVoxelCorrelationSource(
            projected_nodes=projected_nodes,
            volume_shape=tuple(int(size) for size in volume_shape),
            input_file_ids=input_file_ids,
            y_bins=int(render_summary.y_bins),
            x_bins=int(render_summary.x_bins),
            depth_bin_um=float(render_summary.depth_bin_um),
            include_depth_minus_one=bool(render_summary.includes_depth_minus_one_plane),
            flatmap_style=getattr(self, "_last_flatmap_style", None),
            coordinate_mode=getattr(self, "_last_coordinate_mode", None),
            flatmap_path=self._last_flatmap_path,
            depth_path=self._last_depth_path,
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=self._negative_one_sentinel_cb.isChecked(),
            mirror_depth_fallback=True,
            mirror_coord_axis=2,
            lookup_stats=getattr(self, "_last_lookup_stats", None),
            cache_dir=getattr(self, "_last_cache_dir", None),
            cache_profile_id=getattr(self, "_last_cache_profile_id", None),
            cache_style=(
                self._current_style_key()
                if hasattr(self, "_style_combo")
                else getattr(self, "_last_flatmap_style", None)
            ),
        )

    def _create_region_labels(self) -> None:
        """Create or update the selected-region flatmap labels layer."""
        try:
            result = self._create_region_labels_from_current_state()
            if result is not None:
                show_info("Flatmap region labels complete.")
        except Exception as exc:
            logger.exception("Flatmap region label creation failed")
            self._notify_display_viewer_failed("region_labels_failed")
            message = f"Flatmap region labels failed: {exc}"
            self._set_region_labels_status(message)
            show_warning(message)

    def _create_region_labels_from_current_state(self):
        """Build a flatmap region-label volume and show it as a Labels layer."""
        if self._current_projection_source() == _PROJECTION_SOURCE_PRECOMPUTED:
            return self._create_cached_region_labels()
        if self._is_flat_render_mode() or self._is_allen_layer_mode():
            # Only the cache can collapse depth. The NRRD path builds a depth
            # stack, so this guards a programmatic call from placing one next to
            # a 2D render.
            raise RuntimeError(
                "Recomputed region labels are built on the depth grid and are "
                "not available in 2D or Allen-layer render modes. Choose "
                "Precomputed Parquet + "
                "Cache to use the cached 2D region overlay, or switch Render to "
                "3D Heatmap or 3D Points."
            )
        self._projection_request_ready()
        selected_region_ids = self._selected_region_ids_for_labels()
        atlas_name = self._current_region_label_atlas_name()
        atlas = self._region_label_atlas_for_name(atlas_name)
        if atlas is None:
            self._start_region_label_atlas_load(atlas_name)
            return None
        return self._create_region_labels_for_atlas(
            atlas,
            selected_region_ids,
        )

    def _require_active_cache_profile(self):
        profile = getattr(self, "_active_cache_profile", None)
        if profile is None:
            raise RuntimeError(
                "No compatible cache profile is active. Choose a cache directory "
                "containing the loaded Parquet lookup-set ID and atlas catalog."
            )
        return profile

    def _region_selection_source(self) -> str:
        provider = getattr(self, "_selected_region_source_provider", None)
        value = provider() if callable(provider) else "atlas_regions"
        source = str(value or "").strip()
        return {
            "Atlas Regions": "atlas_regions",
            "Custom Regions": "custom_regions",
            "Mask Layer": "mask_layer",
        }.get(source, source or "atlas_regions")

    def _region_selection_scope(self) -> str:
        provider = getattr(self, "_selected_region_scope_provider", None)
        value = provider() if callable(provider) else "whole_parquet"
        scope = str(value or "").strip()
        return {
            "whole": "whole_parquet",
            "current": "current_table",
            "Whole Parquet": "whole_parquet",
            "Current Table": "current_table",
        }.get(scope, scope or "whole_parquet")

    def _region_selection_metadata(self) -> dict[str, str]:
        return {
            "region_selection_source": self._region_selection_source(),
            "region_selection_scope": self._region_selection_scope(),
        }

    def _raise_region_selection_error(self) -> None:
        provider = getattr(self, "_selected_region_error_provider", None)
        message = provider() if callable(provider) else None
        if message:
            raise RuntimeError(str(message))
        if self._region_selection_source() == "mask_layer":
            raise RuntimeError(
                "Flatmap atlas overlays do not support Mask Layer selections. "
                "Choose Atlas Regions or Custom Regions."
            )

    def _selected_geometry_region_ids(self) -> list[int]:
        self._raise_region_selection_error()
        provider = getattr(self, "_selected_geometry_region_ids_provider", None)
        if not callable(provider):
            provider = getattr(self, "_selected_parent_region_ids_provider", None)
        values = (provider() or []) if callable(provider) else []
        return sorted({int(value) for value in values if int(value) > 0})

    def _selected_parent_region_ids(self) -> list[int]:
        """Compatibility alias for the former parent-only geometry provider."""
        return self._selected_geometry_region_ids()

    def _create_cached_region_labels(self):
        from ..flatmap_region_cache import (
            materialize_allen_layer_region_selection,
            materialize_flat_region_selection,
            materialize_region_selection,
        )

        profile = self._require_active_cache_profile()
        selected_region_ids = self._selected_region_ids_for_labels()
        atlas = self._atlas_provider()
        plane_mode = self._current_plane_mode()
        layer_map = None
        axis_labels = None
        layer_name = _REGION_LABELS_LAYER_NAME
        flat_result = None
        if plane_mode in {
            FLATMAP_PLANE_MODE_ALLEN_LAYERS,
            FLATMAP_PLANE_MODE_ALLEN_LAYER_PROJECTION,
        }:
            layer_map = self._current_allen_layer_map()
            result = materialize_allen_layer_region_selection(
                profile,
                selected_region_ids,
                style=self._current_style_key(),
                layer_map=layer_map,
                selected_layer_indices=self._current_allen_layer_indices(),
                output_mode=self._current_allen_layer_output_mode(),
            )
            if not result.layer_mapped_region_ids:
                raise RuntimeError(
                    "The selected regions contain no terminal Allen "
                    "Isocortex layer regions."
                )
            if plane_mode == FLATMAP_PLANE_MODE_ALLEN_LAYER_PROJECTION:
                axis_labels = self._flat_axis_labels()
                layer_name = _ALLEN_LAYER_PROJECTION_REGION_LABELS_LAYER_NAME
            else:
                axis_labels = self._allen_layer_axis_labels()
        elif plane_mode == FLATMAP_PLANE_MODE_FLAT:
            try:
                result = materialize_flat_region_selection(
                    profile,
                    selected_region_ids,
                    style=self._current_style_key(),
                    direct_region_ids=self._selected_geometry_region_ids(),
                    atlas_structures=getattr(atlas, "structures", None),
                    include_outlines=False,
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Load a matching BrainGlobe atlas structure catalog to "
                    "combine more than one selected region into a 2D map, or "
                    "select a single region."
                ) from exc
            axis_labels = self._flat_axis_labels()
            layer_name = _FLAT_REGION_LABELS_LAYER_NAME
            flat_result = result
        else:
            result = materialize_region_selection(
                profile,
                selected_region_ids,
                style=self._current_style_key(),
                direct_region_ids=self._selected_geometry_region_ids(),
                include_surfaces=False,
                include_outlines=False,
            )
            axis_labels = self._depth_axis_labels()
        if result.summary.labeled_bins == 0:
            self._clear_named_region_layers(_REGION_LABELS_LAYER_NAME)
            self._region_labels_layer = None
            raise RuntimeError(
                "The selected regions have no occupancy in the active flatmap cache."
            )
        metadata = {
            _FLATMAP_LAYER_SPACE_KEY: _FLATMAP_LAYER_SPACE_VALUE,
            "projection_kind": "flatmap_region_labels",
            "source": "precomputed_cache",
            "cache_path": str(self._region_cache_dir),
            "cache_profile_id": result.profile_id,
            "flatmap_style": self._current_style_key(),
            "selected_region_ids": [int(value) for value in result.selected_region_ids],
            "selected_region_acronyms": [
                str(value)
                for value in (self._selected_region_acronyms_provider() or [])
            ],
            "represented_region_ids": [
                int(value) for value in result.represented_region_ids
            ],
            "summary": result.summary.to_dict(),
            **self._region_selection_metadata(),
        }
        if layer_map is not None:
            metadata.update(
                {
                    "flatmap_plane_mode": plane_mode,
                    "allen_layer_output_mode": (
                        self._current_allen_layer_output_mode()
                    ),
                    "allen_layer_indices": [
                        int(value)
                        for value in self._current_allen_layer_indices()
                    ],
                    "allen_layer_labels": list(result.layer_labels),
                    "allen_atlas_name": layer_map.atlas_name,
                    "allen_atlas_version": layer_map.atlas_version,
                    "allen_atlas_identity": {
                        "name": layer_map.atlas_name,
                        "version": layer_map.atlas_version,
                    },
                    "layer_mapped_region_ids": [
                        int(value) for value in result.layer_mapped_region_ids
                    ],
                }
            )
        elif flat_result is not None:
            metadata.update(
                {
                    "flatmap_plane_mode": FLATMAP_PLANE_MODE_FLAT,
                    "direct_region_ids": [
                        int(value) for value in flat_result.direct_region_ids
                    ],
                    "represented_source_region_ids": [
                        int(value)
                        for value in flat_result.represented_source_region_ids
                    ],
                    "label_grouping": str(flat_result.grid_spec["label_grouping"]),
                    "geometry_grouping": str(
                        flat_result.grid_spec.get(
                            "geometry_grouping",
                            flat_result.grid_spec["label_grouping"],
                        )
                    ),
                }
            )
        layer = self._create_or_update_region_labels_layer(
            result,
            metadata,
            atlas=atlas,
            axis_labels=axis_labels,
            layer_name=layer_name,
        )
        self._region_labels_layer = layer
        self._focus_projection_view(
            layer,
            result.labels,
            ndisplay=3 if plane_mode == FLATMAP_PLANE_MODE_DEPTH else 2,
        )
        self._notify_display_viewer_ready(layer)
        if layer_map is not None:
            if plane_mode == FLATMAP_PLANE_MODE_ALLEN_LAYER_PROJECTION:
                message = (
                    f"Loaded {result.summary.labeled_bins:,} cached planar "
                    f"region bin(s) projected from {len(result.layer_labels)} "
                    f"selected Allen layer(s) from profile {result.profile_id}."
                )
            else:
                message = (
                    f"Loaded {result.summary.labeled_bins:,} cached planar region "
                    f"bin(s) across {len(result.layer_labels)} Allen layer planes "
                    f"from profile {result.profile_id}."
                )
        elif flat_result is not None:
            message = (
                f"Loaded {result.summary.labeled_bins:,} collapsed region bin(s) "
                f"for {result.summary.represented_region_count} selected "
                f"region(s) from profile {result.profile_id}."
            )
        else:
            message = (
                f"Loaded {result.summary.labeled_bins:,} cached region bin(s) "
                f"from profile {result.profile_id}."
            )
        self._set_region_labels_status(message)
        return result

    def _selected_region_ids_for_labels(self) -> list[int]:
        self._raise_region_selection_error()
        selected_region_ids = sorted(
            {
                int(region_id)
                for region_id in (self._selected_region_ids_provider() or [])
                if int(region_id) > 0
            }
        )
        if not selected_region_ids:
            source = self._region_selection_source()
            if source == "custom_regions":
                selection_name = "Custom Region"
            elif source == "atlas_regions":
                selection_name = "Atlas Region"
            else:
                selection_name = "atlas region"
            raise RuntimeError(
                f"Select at least one {selection_name} before creating labels."
            )
        return selected_region_ids

    def _create_region_labels_for_atlas(
        self,
        atlas,
        selected_region_ids: list[int],
    ):
        volume_set = load_flatmap_volume_set(self._flatmap_path, self._depth_path)
        lookup_stats = self._lookup_stats_for_volume_set(
            volume_set,
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=self._negative_one_sentinel_cb.isChecked(),
        )
        result = build_flatmap_region_label_volume(
            np.asarray(atlas.annotation),
            volume_set.flatmap,
            volume_set.depth,
            selected_region_ids=selected_region_ids,
            y_bins=self._current_y_bins(),
            x_bins=self._current_x_bins(),
            depth_bin_um=self._current_depth_bin_um(),
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=self._negative_one_sentinel_cb.isChecked(),
            lookup_stats=lookup_stats,
            mirror_depth_fallback=True,
            mirror_coord_axis=2,
        )
        metadata = self._region_labels_metadata(result, atlas)
        layer = self._create_or_update_region_labels_layer(
            result,
            metadata,
            atlas=atlas,
        )
        self._region_labels_layer = layer
        self._focus_projection_view(layer, result.labels)
        self._notify_display_viewer_ready(layer)

        message = (
            "Created flatmap region labels: "
            f"{result.summary.labeled_voxels:,} labeled voxel(s) from "
            f"{len(result.selected_region_ids):,} selected region ID(s)."
        )
        self._status_label.setText(message)
        label = getattr(self, "_region_labels_status_label", None)
        if label is not None:
            label.setText(message)
        return result

    def _region_label_atlas_for_name(self, atlas_name: str):
        cached = self._region_label_atlas_cache.get(atlas_name)
        if cached is not None:
            return cached

        provider_atlas = self._atlas_provider()
        provider_name = str(getattr(provider_atlas, "atlas_name", "") or "")
        if provider_atlas is not None and provider_name == atlas_name:
            self._region_label_atlas_cache[atlas_name] = provider_atlas
            return provider_atlas
        return None

    def _start_region_label_atlas_load(self, atlas_name: str) -> None:
        self._pending_region_label_request = True
        self._region_label_request_display_generation = self._display_generation()
        if self._region_label_atlas_load_running():
            self._set_region_labels_status(
                f"Loading region-label atlas {atlas_name}..."
            )
            return

        from qtpy.QtCore import QThread

        from ..workers import AtlasLoadWorker

        self._set_region_label_controls_enabled(False)
        self._set_region_labels_status(f"Loading region-label atlas {atlas_name}...")

        thread = QThread()
        worker = AtlasLoadWorker(atlas_name)
        self._region_label_atlas_load_thread = thread
        self._region_label_atlas_load_worker = worker
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.status.connect(self._on_region_label_atlas_load_status)
        worker.finished.connect(
            lambda atlas, expected=atlas_name: (
                self._on_region_label_atlas_load_finished(
                    atlas,
                    expected,
                )
            )
        )
        worker.error.connect(self._on_region_label_atlas_load_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda: self._cleanup_region_label_atlas_load_thread(thread, worker)
        )
        thread.start()

    def _region_label_atlas_load_running(self) -> bool:
        thread = getattr(self, "_region_label_atlas_load_thread", None)
        is_running = getattr(thread, "isRunning", None)
        return bool(thread is not None and callable(is_running) and is_running())

    def _on_region_label_atlas_load_status(self, message: str) -> None:
        self._set_region_labels_status(str(message))

    def _on_region_label_atlas_load_finished(self, atlas, atlas_name: str) -> None:
        resolved_name = str(getattr(atlas, "atlas_name", "") or atlas_name)
        self._region_label_atlas_cache[atlas_name] = atlas
        self._region_label_atlas_cache[resolved_name] = atlas
        self._set_region_label_controls_enabled(True)
        self._set_region_labels_status(f"Loaded region-label atlas {resolved_name}.")

        if self._pending_region_label_request and self._display_generation_matches(
            getattr(self, "_region_label_request_display_generation", None)
        ):
            self._pending_region_label_request = False
            self._create_region_labels()
        elif self._pending_region_label_request:
            self._pending_region_label_request = False
            self._set_region_labels_status(
                "Region-label atlas loaded after the flatmap window changed or "
                "closed; "
                "choose Show Region Labels again."
            )

    def _on_region_label_atlas_load_error(self, error_msg: str) -> None:
        self._pending_region_label_request = False
        self._region_label_request_display_generation = None
        self._set_region_label_controls_enabled(True)
        message = f"Region-label atlas load failed: {error_msg}"
        logger.error(message)
        self._set_region_labels_status(message)
        show_warning(message)

    def _cleanup_region_label_atlas_load_thread(self, thread, worker) -> None:
        if getattr(self, "_region_label_atlas_load_thread", None) is thread:
            self._region_label_atlas_load_thread = None
        if getattr(self, "_region_label_atlas_load_worker", None) is worker:
            self._region_label_atlas_load_worker = None
            self._region_label_request_display_generation = None

    def _set_region_label_controls_enabled(self, enabled: bool) -> None:
        states = self._cached_region_control_states()
        for widget_name in (
            "_region_label_atlas_combo",
            "_region_labels_btn",
            "_clear_region_labels_btn",
            "_region_surfaces_btn",
            "_region_outlines_btn",
            "_clear_region_geometry_btn",
        ):
            widget = getattr(self, widget_name, None)
            set_enabled = getattr(widget, "setEnabled", None)
            if callable(set_enabled):
                # The clear button only tracks the caller's gate; every other
                # control also has to satisfy the render/source matrix.
                set_enabled(bool(enabled) and states.get(widget_name, True))

    def _set_region_labels_status(self, message: str) -> None:
        status_label = getattr(self, "_status_label", None)
        if status_label is not None:
            status_label.setText(message)
        region_status_label = getattr(self, "_region_labels_status_label", None)
        if region_status_label is not None:
            region_status_label.setText(message)

    def _region_labels_metadata(
        self,
        result: FlatmapRegionLabelsResult,
        atlas,
    ) -> dict[str, object]:
        acronyms = [
            str(acronym)
            for acronym in (self._selected_region_acronyms_provider() or [])
        ]
        return {
            _FLATMAP_LAYER_SPACE_KEY: _FLATMAP_LAYER_SPACE_VALUE,
            "projection_kind": "flatmap_region_labels",
            "flatmap_style": self._current_style_filename(),
            "flatmap_path": str(self._flatmap_path) if self._flatmap_path else "",
            "depth_path": str(self._depth_path) if self._depth_path else "",
            "atlas_name": str(getattr(atlas, "atlas_name", "")),
            "selected_region_ids": [
                int(region_id) for region_id in result.selected_region_ids
            ],
            "selected_region_acronyms": acronyms,
            "represented_region_ids": [
                int(region_id) for region_id in result.represented_region_ids
            ],
            "summary": result.summary.to_dict(),
            **self._region_selection_metadata(),
        }

    def _create_or_update_region_labels_layer(
        self,
        result,
        metadata: dict[str, object],
        *,
        atlas=None,
        axis_labels: tuple[str, ...] | None = None,
        layer_name: str = _REGION_LABELS_LAYER_NAME,
    ):
        metadata = dict(metadata)
        metadata[_FLATMAP_LAYER_SPACE_KEY] = _FLATMAP_LAYER_SPACE_VALUE
        metadata["region_layer_kind"] = "flatmap_labels"
        viewer = self._display_viewer()
        layer = self._region_labels_layer
        if not self._layer_is_in_viewer(layer, viewer=viewer):
            self._region_labels_layer = None
            layer = None
        if layer is not None and str(getattr(layer, "name", "")) != layer_name:
            # A depth-grid layer cannot take collapsed data, and vice versa.
            layer = None
        layer = layer or self._find_layer_by_name(
            layer_name,
            viewer=viewer,
        )
        colormap = self._region_label_colormap(
            atlas,
            self._region_label_ids(result=result),
        )
        kwargs: dict[str, object] = {
            "name": layer_name,
            "opacity": 0.35,
            "visible": True,
            "metadata": metadata,
        }
        if colormap is not None:
            kwargs["colormap"] = colormap
        if axis_labels is not None:
            kwargs["axis_labels"] = axis_labels

        if layer is None:
            layer = viewer.add_labels(result.labels, **kwargs)
        else:
            blocker = getattr(getattr(layer, "events", None), "blocker_all", None)
            if callable(blocker):
                with blocker():
                    self._set_region_labels_layer_data(
                        layer,
                        result,
                        metadata,
                        colormap,
                        axis_labels,
                    )
            else:
                self._set_region_labels_layer_data(
                    layer,
                    result,
                    metadata,
                    colormap,
                    axis_labels,
                )
            refresh = getattr(layer, "refresh", None)
            if callable(refresh):
                refresh()

        setattr(layer, "_napari_neuron_navigator_flatmap_region_labels_result", result)
        setattr(layer, "_napari_neuron_navigator_region_base_opacity", 0.35)
        return layer

    @staticmethod
    def _set_region_labels_layer_data(
        layer,
        result,
        metadata: dict[str, object],
        colormap,
        axis_labels: tuple[str, str, str] | None,
    ) -> None:
        layer.data = result.labels
        layer.metadata = metadata
        layer.opacity = 0.35
        layer.visible = True
        if colormap is not None:
            layer.colormap = colormap
        if axis_labels is not None:
            layer.axis_labels = axis_labels

    @staticmethod
    def _atlas_structure_for_region_id(atlas, region_id: int):
        structures = getattr(atlas, "structures", None)
        if structures is None:
            return None
        try:
            return structures[int(region_id)]
        except (KeyError, TypeError):
            return None

    def _region_appearance_store(self) -> RegionAppearanceStore:
        provider = getattr(self, "_region_appearance_provider", None)
        store = provider() if callable(provider) else None
        return (
            store
            if isinstance(store, RegionAppearanceStore)
            else RegionAppearanceStore()
        )

    @staticmethod
    def _region_label_ids(
        *,
        result=None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[int]:
        """Return the atlas IDs stored as values in a region Labels layer."""
        if result is not None:
            source_ids = getattr(result, "represented_source_region_ids", None)
            if source_ids is not None:
                return [int(value) for value in source_ids]
            represented_ids = getattr(result, "represented_region_ids", ())
            return [int(value) for value in represented_ids]
        if metadata is not None:
            source_ids = metadata.get("represented_source_region_ids")
            if source_ids is not None:
                return [int(value) for value in source_ids]  # type: ignore[union-attr]
            represented_ids = metadata.get("represented_region_ids", ())
            return [int(value) for value in represented_ids]  # type: ignore[union-attr]
        return []

    def _region_label_colormap(self, atlas, region_ids: list[int]):
        if atlas is None:
            return None
        try:
            from napari.utils import DirectLabelColormap
        except Exception:
            return None

        color_dict: dict[int | None, np.ndarray] = {
            None: np.array([0, 0, 0, 0], dtype=np.float32),
            0: np.array([0, 0, 0, 0], dtype=np.float32),
        }
        appearance_store = self._region_appearance_store()
        catalog = structure_catalog(getattr(atlas, "structures", None))
        for region_id in region_ids:
            effective = appearance_store.resolve(
                int(region_id),
                catalog,
            )
            color_dict[int(region_id)] = effective.fill_rgba
        return DirectLabelColormap(color_dict=color_dict)

    def _clear_region_labels(self) -> None:
        """Remove the flatmap region labels layer if present."""
        layers = self._display_layers(create=False)
        layer = self._region_labels_layer
        if not self._layer_is_in_viewer(layer):
            layer = self._find_layer_by_name(
                _REGION_LABELS_LAYER_NAME,
                create=False,
            )
        if layer is not None and layers is not None:
            try:
                layers.remove(layer)
            except ValueError:
                pass
        self._region_labels_layer = None
        message = "Cleared flatmap region labels."
        self._status_label.setText(message)
        label = getattr(self, "_region_labels_status_label", None)
        if label is not None:
            label.setText("No flatmap region labels created.")

    @classmethod
    def _atlas_region_rgba(cls, atlas, region_id: int) -> np.ndarray:
        structure = cls._atlas_structure_for_region_id(atlas, region_id)
        rgb = (
            structure.get("rgb_triplet", [128, 128, 128])
            if structure is not None
            else [128, 128, 128]
        )
        return np.asarray(
            [float(rgb[0]) / 255, float(rgb[1]) / 255, float(rgb[2]) / 255, 1],
            dtype=np.float32,
        )

    def _effective_region_appearance(self, atlas, region_id: int):
        return self._region_appearance_store().resolve(
            int(region_id),
            getattr(atlas, "structures", None),
        )

    @classmethod
    def _atlas_region_identity(
        cls,
        atlas,
        region_id: int,
    ) -> tuple[str, str]:
        structure = cls._atlas_structure_for_region_id(atlas, region_id)
        if structure is None:
            return "", ""
        return (
            str(structure.get("acronym", "") or ""),
            str(structure.get("name", "") or ""),
        )

    def _cached_geometry_layer_name(
        self,
        prefix: str,
        atlas,
        region_id: int,
        *,
        selection_count: int,
    ) -> str:
        if selection_count == 1 and self._region_selection_source() != "custom_regions":
            return prefix
        acronym, region_name = self._atlas_region_identity(atlas, region_id)
        identity = acronym or region_name
        if not identity:
            return f"{prefix}: {region_id}"
        return f"{prefix}: {identity} ({region_id})"

    def _cached_geometry_metadata(
        self,
        *,
        projection_kind: str,
        profile,
        atlas,
        region_id: int,
        selected_region_ids: list[int],
    ) -> dict[str, object]:
        acronym, region_name = self._atlas_region_identity(atlas, region_id)
        return {
            _FLATMAP_LAYER_SPACE_KEY: _FLATMAP_LAYER_SPACE_VALUE,
            "projection_kind": projection_kind,
            "source": "precomputed_cache",
            "cache_path": str(self._region_cache_dir),
            "cache_profile_id": profile.profile_id,
            "flatmap_style": self._current_style_key(),
            "region_id": int(region_id),
            "region_acronym": acronym,
            "region_name": region_name,
            "selected_region_ids": [int(value) for value in selected_region_ids],
            "selected_region_acronyms": [
                str(value)
                for value in (self._selected_region_acronyms_provider() or [])
            ],
            **self._region_selection_metadata(),
        }

    def _cached_geometry_inputs(self):
        profile = self._require_active_cache_profile()
        geometry_ids = self._selected_geometry_region_ids()
        if not geometry_ids:
            selection_name = (
                "Custom Region"
                if self._region_selection_source() == "custom_regions"
                else "Atlas Region"
            )
            raise RuntimeError(
                f"Select at least one {selection_name} before showing cached geometry."
            )
        atlas = self._atlas_provider()
        if atlas is None:
            raise RuntimeError(
                "Load a matching BrainGlobe atlas structure catalog for region colors."
            )
        return profile, geometry_ids, atlas

    def _create_region_surfaces(self) -> None:
        """Show cached exposed-face shells for active region selections."""
        try:
            from napari.utils.colormaps import Colormap

            from ..flatmap_region_cache import materialize_region_surface

            profile, geometry_ids, atlas = self._cached_geometry_inputs()
            prepared = []
            for region_id in geometry_ids:
                surface = materialize_region_surface(
                    profile,
                    region_id,
                    style=self._current_style_key(),
                )
                if surface is None or not len(surface.faces):
                    continue
                effective = self._effective_region_appearance(atlas, region_id)
                rgba = np.asarray(effective.color_rgba, dtype=np.float32)
                name = self._cached_geometry_layer_name(
                    _REGION_SURFACES_LAYER_NAME,
                    atlas,
                    region_id,
                    selection_count=len(geometry_ids),
                )
                metadata = self._cached_geometry_metadata(
                    projection_kind="flatmap_region_surface",
                    profile=profile,
                    atlas=atlas,
                    region_id=region_id,
                    selected_region_ids=geometry_ids,
                )
                metadata["component_count"] = int(surface.component_count)
                metadata["region_layer_kind"] = "flatmap_surface"
                prepared.append(
                    (
                        (
                            np.array(surface.vertices, dtype=np.float32, copy=True),
                            np.array(surface.faces, dtype=np.int32, copy=True),
                            np.ones(len(surface.vertices), dtype=np.float32),
                        ),
                        name,
                        rgba,
                        effective,
                        metadata,
                    )
                )

            if not prepared:
                self._clear_region_surface_layers()
                self._notify_display_viewer_failed("region_surfaces_empty")
                self._set_region_labels_status(
                    "Loaded 0 cached region surface layer(s)."
                )
                show_warning(
                    "The selected cache profile has no surface for this selection."
                )
                return

            self._clear_region_surface_layers()
            viewer = self._display_viewer()
            created = []
            for data, name, rgba, effective, metadata in prepared:
                layer = viewer.add_surface(
                    data,
                    name=name,
                    colormap=Colormap(np.vstack([rgba, rgba])),
                    contrast_limits=(0.0, 1.0),
                    opacity=0.45 * effective.fill_opacity,
                    visible=effective.fill_visible,
                    metadata=metadata,
                )
                setattr(layer, "_napari_neuron_navigator_region_base_opacity", 0.45)
                setattr(layer, "_napari_neuron_navigator_region_base_visible", True)
                setattr(
                    layer,
                    "_napari_neuron_navigator_region_applied_visible",
                    bool(layer.visible),
                )
                created.append(layer)
            self._region_surfaces_layers = created
            self._notify_display_viewer_ready(created[0])
            message = f"Loaded {len(created)} cached region surface layer(s)."
            self._set_region_labels_status(message)
        except Exception as exc:
            logger.exception("Cached flatmap region surfaces failed")
            self._notify_display_viewer_failed("region_surfaces_failed")
            show_warning(f"Cached flatmap region surfaces failed: {exc}")

    def _create_region_outlines(self) -> None:
        """Show cached XY perimeter vectors for the active region selections."""
        if self._current_plane_mode() == FLATMAP_PLANE_MODE_FLAT:
            self._create_flat_region_outlines()
            return
        self._create_depth_region_outlines()

    def _create_flat_region_outlines(self) -> None:
        """Show depth-collapsed 2D perimeter vectors for active selections."""
        try:
            from ..flatmap_region_cache import materialize_flat_region_selection

            profile, geometry_ids, atlas = self._cached_geometry_inputs()
            # One call for the whole selection: it resolves the region hierarchy
            # once and returns one collapsed perimeter per selected region.
            try:
                result = materialize_flat_region_selection(
                    profile,
                    self._selected_region_ids_for_labels(),
                    style=self._current_style_key(),
                    direct_region_ids=geometry_ids,
                    atlas_structures=getattr(atlas, "structures", None),
                    include_outlines=True,
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Load a matching BrainGlobe atlas structure catalog to "
                    "combine more than one selected region into a 2D outline, "
                    "or select a single region."
                ) from exc
            prepared = []
            for outlines in result.outlines:
                if not len(outlines.vectors):
                    continue
                region_id = int(outlines.region_id)
                metadata = self._cached_geometry_metadata(
                    projection_kind="flatmap_flat_region_outlines",
                    profile=profile,
                    atlas=atlas,
                    region_id=region_id,
                    selected_region_ids=geometry_ids,
                )
                metadata.update(
                    {
                        "flatmap_plane_mode": FLATMAP_PLANE_MODE_FLAT,
                        "union_region_ids": [
                            int(value) for value in outlines.union_region_ids
                        ],
                        "represented_source_region_ids": [
                            int(value) for value in outlines.represented_region_ids
                        ],
                        "planar_bin_count": int(outlines.planar_bin_count),
                        "label_grouping": str(result.grid_spec["label_grouping"]),
                        "geometry_grouping": str(
                            result.grid_spec.get(
                                "geometry_grouping",
                                result.grid_spec["label_grouping"],
                            )
                        ),
                    }
                )
                effective = self._effective_region_appearance(atlas, region_id)
                metadata["region_layer_kind"] = "flatmap_outline"
                prepared.append(
                    (
                        np.array(outlines.vectors, dtype=np.float32, copy=True),
                        self._cached_geometry_layer_name(
                            _FLAT_REGION_OUTLINES_LAYER_NAME,
                            atlas,
                            region_id,
                            selection_count=len(geometry_ids),
                        ),
                        np.asarray(effective.color_rgba, dtype=np.float32),
                        effective,
                        metadata,
                    )
                )

            if not prepared:
                self._clear_region_outline_layers()
                self._notify_display_viewer_failed("region_outlines_empty")
                self._set_region_labels_status(
                    "Loaded 0 collapsed region outline layer(s) "
                    f"from profile {result.profile_id}."
                )
                show_warning(
                    "The selected cache profile has no outlines for this selection."
                )
                return

            self._clear_region_outline_layers()
            viewer = self._display_viewer()
            created = []
            for data, name, edge_color, effective, metadata in prepared:
                layer = viewer.add_vectors(
                    data,
                    name=name,
                    edge_color=edge_color,
                    edge_width=_FLAT_REGION_OUTLINE_EDGE_WIDTH,
                    opacity=0.9 * effective.outline_opacity,
                    visible=effective.outline_visible,
                    # napari draws an arrowhead per segment by default, which
                    # reads as a field of arrows rather than a boundary.
                    vector_style="line",
                    blending="translucent",
                    axis_labels=self._flat_axis_labels(),
                    metadata=metadata,
                )
                setattr(layer, "_napari_neuron_navigator_region_base_opacity", 0.9)
                setattr(layer, "_napari_neuron_navigator_region_base_visible", True)
                setattr(
                    layer,
                    "_napari_neuron_navigator_region_applied_visible",
                    bool(layer.visible),
                )
                created.append(layer)
            self._region_outlines_layers = created
            # Deliberately not _focus_projection_view: re-centring would yank
            # the user off the heatmap they are inspecting.
            self._ensure_flat_overlay_ndisplay()
            self._apply_display_axis_annotations(created[0])
            self._notify_display_viewer_ready(created[0])
            message = (
                f"Loaded {len(created)} collapsed region outline layer(s) "
                f"from profile {result.profile_id}."
            )
            self._set_region_labels_status(message)
        except Exception as exc:
            logger.exception("Collapsed flatmap region outlines failed")
            self._notify_display_viewer_failed("region_outlines_failed")
            show_warning(f"Collapsed flatmap region outlines failed: {exc}")

    def _ensure_flat_overlay_ndisplay(self) -> None:
        """Put the display viewer in 2D without disturbing the camera."""
        viewer = self._display_viewer()
        dims = getattr(viewer, "dims", None)
        if dims is not None and getattr(dims, "ndisplay", None) != 2:
            try:
                dims.ndisplay = 2
            except Exception:
                logger.debug("Could not set ndisplay for flat overlay", exc_info=True)

    def _create_depth_region_outlines(self) -> None:
        """Show cached per-depth XY perimeter vectors for active selections."""
        try:
            from ..flatmap_region_cache import materialize_region_outlines

            profile, geometry_ids, atlas = self._cached_geometry_inputs()
            prepared = []
            for region_id in geometry_ids:
                outlines = materialize_region_outlines(
                    profile,
                    region_id,
                    style=self._current_style_key(),
                )
                if outlines is None or not len(outlines.vectors):
                    continue
                effective = self._effective_region_appearance(atlas, region_id)
                rgba = np.asarray(effective.color_rgba, dtype=np.float32)
                name = self._cached_geometry_layer_name(
                    _REGION_OUTLINES_LAYER_NAME,
                    atlas,
                    region_id,
                    selection_count=len(geometry_ids),
                )
                metadata = self._cached_geometry_metadata(
                    projection_kind="flatmap_region_outlines",
                    profile=profile,
                    atlas=atlas,
                    region_id=region_id,
                    selected_region_ids=geometry_ids,
                )
                metadata["region_layer_kind"] = "flatmap_outline"
                prepared.append(
                    (
                        np.array(outlines.vectors, dtype=np.float32, copy=True),
                        name,
                        rgba,
                        effective,
                        metadata,
                    )
                )

            if not prepared:
                self._clear_region_outline_layers()
                self._notify_display_viewer_failed("region_outlines_empty")
                self._set_region_labels_status(
                    "Loaded 0 cached region outline layer(s)."
                )
                show_warning(
                    "The selected cache profile has no outlines for this selection."
                )
                return

            self._clear_region_outline_layers()
            viewer = self._display_viewer()
            created = []
            for data, name, rgba, effective, metadata in prepared:
                layer = viewer.add_vectors(
                    data,
                    name=name,
                    edge_color=rgba,
                    edge_width=1.5,
                    opacity=0.9 * effective.outline_opacity,
                    visible=effective.outline_visible,
                    # Perimeter segments are boundaries, not directed vectors;
                    # napari's default style would put an arrowhead on each one.
                    vector_style="line",
                    metadata=metadata,
                )
                setattr(layer, "_napari_neuron_navigator_region_base_opacity", 0.9)
                setattr(layer, "_napari_neuron_navigator_region_base_visible", True)
                setattr(
                    layer,
                    "_napari_neuron_navigator_region_applied_visible",
                    bool(layer.visible),
                )
                created.append(layer)
            self._region_outlines_layers = created
            self._notify_display_viewer_ready(created[0])
            message = f"Loaded {len(created)} cached region outline layer(s)."
            self._set_region_labels_status(message)
        except Exception as exc:
            logger.exception("Cached flatmap region outlines failed")
            self._notify_display_viewer_failed("region_outlines_failed")
            show_warning(f"Cached flatmap region outlines failed: {exc}")

    def _clear_named_region_layers(self, prefix: str) -> None:
        layers = self._display_layers(create=False)
        if layers is None:
            return
        for layer in list(layers):
            if str(getattr(layer, "name", "")).startswith(prefix):
                try:
                    layers.remove(layer)
                except ValueError:
                    pass

    def _clear_region_surface_layers(self) -> None:
        self._clear_named_region_layers(_REGION_SURFACES_LAYER_NAME)
        self._region_surfaces_layers = []

    def _clear_region_outline_layers(self) -> None:
        self._clear_named_region_layers(_REGION_OUTLINES_LAYER_NAME)
        self._region_outlines_layers = []

    def _clear_region_geometry(self) -> None:
        self._clear_region_surface_layers()
        self._clear_region_outline_layers()
        self._set_region_labels_status("Cleared cached flatmap region geometry.")

    def apply_region_appearance(self) -> None:
        """Restyle current flatmap region layers without materializing data."""
        layers = self._display_layers(create=False)
        atlas = self._atlas_provider()
        if layers is None or atlas is None:
            return
        appearance_store = self._region_appearance_store()
        try:
            from napari.utils.colormaps import Colormap
        except Exception:
            Colormap = None  # type: ignore[assignment,misc]

        for layer in list(layers):
            metadata = getattr(layer, "metadata", None)
            if not isinstance(metadata, Mapping):
                continue
            kind = str(metadata.get("region_layer_kind", "") or "")
            if kind == "flatmap_labels":
                result = getattr(
                    layer,
                    "_napari_neuron_navigator_flatmap_region_labels_result",
                    None,
                )
                colormap = self._region_label_colormap(
                    atlas,
                    self._region_label_ids(result=result, metadata=metadata),
                )
                if colormap is not None:
                    layer.colormap = colormap
            elif kind == "flatmap_surface":
                region_id = int(metadata.get("region_id", 0) or 0)
                if region_id <= 0 or Colormap is None:
                    continue
                effective = appearance_store.resolve(
                    region_id,
                    getattr(atlas, "structures", None),
                )
                rgba = np.asarray(effective.color_rgba, dtype=np.float32)
                layer.colormap = Colormap(np.vstack([rgba, rgba]))
                base_opacity = float(
                    getattr(layer, "_napari_neuron_navigator_region_base_opacity", 0.45)
                )
                layer.opacity = base_opacity * effective.fill_opacity
                _set_region_layer_visible(
                    layer,
                    _region_layer_base_visible(layer) and effective.fill_visible,
                )
            elif kind == "flatmap_outline":
                region_id = int(metadata.get("region_id", 0) or 0)
                if region_id <= 0:
                    continue
                effective = appearance_store.resolve(
                    region_id,
                    getattr(atlas, "structures", None),
                )
                layer.edge_color = np.asarray(
                    effective.color_rgba,
                    dtype=np.float32,
                )
                base_opacity = float(
                    getattr(layer, "_napari_neuron_navigator_region_base_opacity", 0.9)
                )
                layer.opacity = base_opacity * effective.outline_opacity
                _set_region_layer_visible(
                    layer,
                    _region_layer_base_visible(layer) and effective.outline_visible,
                )
            else:
                continue
            refresh = getattr(layer, "refresh", None)
            if callable(refresh):
                refresh()

    def _color_for_file_id(
        self,
        file_id: object,
        color_map: dict[object, list[float]],
    ) -> np.ndarray:
        raw = color_map.get(file_id)
        if raw is None:
            raw = color_map.get(str(file_id))
        if raw is None:
            return _DEFAULT_TRACE_COLOR.copy()

        color = np.asarray(raw, dtype=float).reshape(-1)
        if color.size < 4:
            color = np.pad(color, (0, 4 - color.size), constant_values=1.0)
        return np.clip(color[:4], 0.0, 1.0)

    def _colors_for_file_ids(self, file_ids: list[object]) -> np.ndarray:
        color_map = self._color_map_provider() or {}
        if len(file_ids) == 0:
            return np.empty((0, 4), dtype=float)
        return np.vstack(
            [self._color_for_file_id(file_id, color_map) for file_id in file_ids]
        )

    @staticmethod
    def _heatmap_contrast_limits(volume: np.ndarray) -> tuple[float, float]:
        upper = float(np.nanmax(volume)) if volume.size else 0.0
        if not np.isfinite(upper) or upper <= 0.0:
            return (0.0, 1.0)
        return (0.0, upper)

    @staticmethod
    def _render_layer_name(render_mode: str) -> str:
        if render_mode == _RENDER_POINTS:
            return _POINTS_LAYER_NAME
        if render_mode == _RENDER_ALLEN_LAYERS:
            return _ALLEN_LAYER_HEATMAP_LAYER_NAME
        if render_mode == _RENDER_FLAT_HEATMAP:
            return _FLAT_HEATMAP_LAYER_NAME
        if render_mode == _RENDER_FLAT_VECTOR:
            return _FLAT_VECTOR_LAYER_NAME
        return _HEATMAP_LAYER_NAME

    @staticmethod
    def _is_flatmap_render_layer_name(name: object) -> bool:
        return name in _FLATMAP_RENDER_LAYER_NAMES or (
            isinstance(name, str)
            and name.startswith(
                (
                    _GROUPED_HEATMAP_LAYER_PREFIX,
                    _GROUPED_ALLEN_LAYER_PREFIX,
                    _GROUPED_ALLEN_LAYER_PROJECTION_PREFIX,
                    _GROUPED_FLAT_HEATMAP_PREFIX,
                )
            )
        )

    def _find_layer_by_name(self, name: str, *, viewer=None, create: bool = True):
        if viewer is None:
            layers = self._display_layers(create=create)
        else:
            layers = getattr(viewer, "layers", None)
        if layers is None:
            return None
        for layer in layers:
            if getattr(layer, "name", None) == name:
                return layer
        return None

    def _layer_is_in_viewer(self, layer, *, viewer=None) -> bool:
        layers = (
            getattr(viewer, "layers", None)
            if viewer is not None
            else self._display_layers(create=False)
        )
        if layer is None or layers is None:
            return False
        return any(existing is layer for existing in layers)

    def _latest_render_mode_is_rendered(self, render_mode: str) -> bool:
        """Return whether a flatmap layer for ``render_mode`` is still rendered."""
        layer = getattr(self, "_projection_layer", None)
        if self._layer_is_in_viewer(layer):
            metadata = getattr(layer, "metadata", {}) or {}
            if metadata.get("flatmap_render_mode") == render_mode:
                return True

        layers = self._display_layers(create=False) or ()
        for candidate in layers:
            name = getattr(candidate, "name", None)
            metadata = getattr(candidate, "metadata", {}) or {}
            if (
                self._is_flatmap_render_layer_name(name)
                and metadata.get("flatmap_render_mode") == render_mode
            ):
                return True
        return False

    def _latest_heatmap_layer_is_rendered(self) -> bool:
        """Return whether the latest depth heatmap layer is still rendered."""
        return self._latest_render_mode_is_rendered(_RENDER_HEATMAP)

    def _cached_projection_layer_for_name(self, name: str):
        layer = getattr(self, "_projection_layer", None)
        if layer is None:
            return None
        if getattr(layer, "name", None) != name or not self._layer_is_in_viewer(layer):
            self._projection_layer = None
            return None
        return layer

    def _remove_projection_layer(
        self,
        *,
        except_name: str | None = None,
        create: bool = True,
    ) -> None:
        layers = self._display_layers(create=create)
        if layers is None:
            self._projection_layer = None
            return
        for layer in list(layers):
            name = getattr(layer, "name", None)
            if not self._is_flatmap_render_layer_name(name) or name == except_name:
                continue
            try:
                layers.remove(layer)
            except ValueError:
                pass
            if layer is self._projection_layer:
                self._projection_layer = None
        if self._projection_layer is not None and (
            getattr(self._projection_layer, "name", None) != except_name
            or not self._layer_is_in_viewer(self._projection_layer)
        ):
            self._projection_layer = None
        if not any(
            self._is_flatmap_render_layer_name(getattr(layer, "name", None))
            for layer in list(layers)
        ):
            # Nothing plane-stacked is left on screen; a stale plane name would
            # describe a layer the user can no longer see.
            self._clear_display_axis_annotations()

    def _render_metadata(
        self,
        projection_summary: ProjectionSummary,
        render_summary: FlatmapRenderSummary | AllenLayerStackSummary,
        *,
        flatmap_style: str,
        coordinate_mode: str,
        render_mode: str,
        heatmap_color_mode: str | None = None,
    ) -> dict[str, object]:
        metadata = {
            _FLATMAP_LAYER_SPACE_KEY: _FLATMAP_LAYER_SPACE_VALUE,
            "projection_kind": "isocortex_flatmap",
            "flatmap_render_mode": render_mode,
            "flatmap_projection_source": str(
                getattr(self, "_last_projection_source", None)
                or self._current_projection_source()
            ),
            "flatmap_style": flatmap_style,
            "coordinate_mode": coordinate_mode,
            "projection_summary": projection_summary.to_dict(),
            "render_summary": render_summary.to_dict(),
            "flatmap_path": str(self._flatmap_path) if self._flatmap_path else "",
            "depth_path": str(self._depth_path) if self._depth_path else "",
        }
        if render_mode == _RENDER_HEATMAP:
            metadata["flatmap_heatmap_color_mode"] = (
                heatmap_color_mode or _HEATMAP_COLOR_SINGLE
            )
            metadata["flatmap_plane_mode"] = FLATMAP_PLANE_MODE_DEPTH
        elif render_mode == _RENDER_FLAT_HEATMAP:
            metadata["flatmap_heatmap_color_mode"] = (
                heatmap_color_mode or _HEATMAP_COLOR_SINGLE
            )
            metadata["flatmap_plane_mode"] = FLATMAP_PLANE_MODE_FLAT
        elif render_mode == _RENDER_FLAT_VECTOR:
            metadata["flatmap_plane_mode"] = FLATMAP_PLANE_MODE_FLAT
        elif render_mode == _RENDER_ALLEN_LAYERS:
            metadata["flatmap_heatmap_color_mode"] = (
                heatmap_color_mode or _HEATMAP_COLOR_SINGLE
            )
            output_mode = str(
                getattr(render_summary, "output_mode", ALLEN_LAYER_OUTPUT_STACK)
            )
            metadata["flatmap_plane_mode"] = (
                FLATMAP_PLANE_MODE_ALLEN_LAYER_PROJECTION
                if output_mode == ALLEN_LAYER_OUTPUT_PROJECTION
                else FLATMAP_PLANE_MODE_ALLEN_LAYERS
            )
            metadata["allen_layer_output_mode"] = output_mode
            metadata["allen_layer_indices"] = [
                int(value)
                for value in getattr(
                    render_summary,
                    "selected_layer_indices",
                    tuple(range(len(render_summary.layer_labels))),
                )
            ]
            metadata["allen_layer_labels"] = list(render_summary.layer_labels)
            metadata["allen_layer_node_counts"] = [
                int(value) for value in render_summary.layer_node_counts
            ]
            metadata["allen_atlas_name"] = render_summary.atlas_name
            metadata["allen_atlas_version"] = render_summary.atlas_version
            metadata["allen_atlas_identity"] = {
                "name": render_summary.atlas_name,
                "version": render_summary.atlas_version,
            }
        return metadata

    def _set_layer_state(
        self,
        layer,
        projected_nodes: pd.DataFrame | None,
        summary: ProjectionSummary,
        render_summary: FlatmapRenderSummary | AllenLayerStackSummary,
    ) -> None:
        setattr(layer, "_napari_neuron_navigator_flatmap_projected_nodes", projected_nodes)
        setattr(layer, "_napari_neuron_navigator_flatmap_summary", summary)
        setattr(layer, "_napari_neuron_navigator_flatmap_render_summary", render_summary)

    @staticmethod
    def _format_render_summary(
        projection_summary: ProjectionSummary,
        render_summary: FlatmapRenderSummary,
    ) -> str:
        depth_minus_one_action = (
            "rendered" if render_summary.includes_depth_minus_one_plane else "excluded"
        )
        return (
            f"Input nodes: {projection_summary.total_nodes:,}\n"
            f"Flatmap-valid nodes: {render_summary.flatmap_valid_nodes:,}\n"
            f"Depth-valid nodes: {render_summary.depth_valid_nodes:,}\n"
            f"Depth -1 nodes {depth_minus_one_action}: "
            f"{render_summary.depth_minus_one_nodes:,}\n"
            f"Rendered nodes: {render_summary.rendered_nodes:,}\n"
            f"Nonzero heatmap voxels: {render_summary.nonzero_voxels:,}\n"
            f"Represented traces: {render_summary.traces_represented:,} "
            f"of {projection_summary.total_traces:,}\n"
            f"Invalid flatmap/depth: "
            f"{projection_summary.invalid_flatmap_nodes:,}/"
            f"{projection_summary.invalid_depth_nodes:,}\n"
            f"Lookup direct/mirrored-depth/mirrored/unmapped: "
            f"{projection_summary.direct_lookup_nodes:,}/"
            f"{projection_summary.mirrored_depth_lookup_nodes:,}/"
            f"{projection_summary.mirrored_lookup_nodes:,}/"
            f"{projection_summary.unmapped_lookup_nodes:,}"
        )

    @classmethod
    def _format_flat_render_summary(
        cls,
        projection_summary: ProjectionSummary,
        render_summary: FlatmapRenderSummary,
        *,
        rendered_segments: int | None = None,
        total_segments: int | None = None,
    ) -> str:
        """Return the depth render summary annotated as depth-collapsed.

        The depth counts stay meaningful because a collapsed render applies the
        same depth rules; only the depth *axis* is gone.
        """
        text = cls._format_render_summary(projection_summary, render_summary)
        text += "\nDepth: collapsed into one flatmap plane"
        if rendered_segments is not None:
            text += f"\nRendered segments: {int(rendered_segments):,}"
            if total_segments is not None and int(total_segments) != int(
                rendered_segments
            ):
                text += f" of {int(total_segments):,}"
        return text

    @staticmethod
    def _format_allen_layer_summary(
        projection_summary: ProjectionSummary,
        render_summary: AllenLayerStackSummary,
    ) -> str:
        per_layer = ", ".join(
            f"{label}: {count:,}"
            for label, count in zip(
                render_summary.layer_labels,
                render_summary.layer_node_counts,
                strict=True,
            )
        )
        return (
            f"Input nodes: {projection_summary.total_nodes:,}\n"
            f"Flatmap-valid nodes: {render_summary.flatmap_valid_nodes:,}\n"
            f"Allen-layer nodes: {render_summary.layer_classified_nodes:,}\n"
            f"Rendered nodes: {render_summary.rendered_nodes:,}\n"
            f"Excluded nodes: {render_summary.excluded_nodes:,} "
            f"({render_summary.invalid_flatmap_nodes:,} invalid flatmap; "
            f"{render_summary.excluded_non_layer_nodes:,} flatmap-valid "
            "non-layer; "
            f"{getattr(render_summary, 'excluded_unselected_layer_nodes', 0):,} "
            "in unselected layers)\n"
            f"Nonzero heatmap voxels: {render_summary.nonzero_voxels:,}\n"
            f"Represented traces: {render_summary.traces_represented:,} "
            f"of {projection_summary.total_traces:,}\n"
            "View: "
            f"{getattr(render_summary, 'output_mode', ALLEN_LAYER_OUTPUT_STACK)}\n"
            f"Layer counts — {per_layer}"
        )

    @staticmethod
    def _allen_layer_axis_labels() -> tuple[str, str, str]:
        """Return dims captions for a categorical Allen-layer plane stack."""
        return (
            _ALLEN_LAYER_AXIS_LABEL,
            _FLATMAP_AXIS_LABEL_Y,
            _FLATMAP_AXIS_LABEL_X,
        )

    @staticmethod
    def _depth_axis_labels() -> tuple[str, str, str]:
        """Return dims captions for a depth-binned flatmap volume."""
        return (
            _DEPTH_AXIS_LABEL,
            _FLATMAP_AXIS_LABEL_Y,
            _FLATMAP_AXIS_LABEL_X,
        )

    @staticmethod
    def _flat_axis_labels() -> tuple[str, str]:
        """Return dims captions for a depth-collapsed flatmap render.

        Only two axes: a flat render has no plane axis to name.
        """
        return (_FLATMAP_AXIS_LABEL_Y, _FLATMAP_AXIS_LABEL_X)

    @staticmethod
    def _allen_layer_output_mode_for_summary(render_summary) -> str:
        if (
            getattr(render_summary, "output_mode", ALLEN_LAYER_OUTPUT_STACK)
            == ALLEN_LAYER_OUTPUT_PROJECTION
        ):
            return ALLEN_LAYER_OUTPUT_PROJECTION
        return ALLEN_LAYER_OUTPUT_STACK

    @classmethod
    def _allen_layer_axis_labels_for_summary(cls, render_summary) -> tuple[str, ...]:
        if (
            cls._allen_layer_output_mode_for_summary(render_summary)
            == ALLEN_LAYER_OUTPUT_PROJECTION
        ):
            return cls._flat_axis_labels()
        return cls._allen_layer_axis_labels()

    @classmethod
    def _allen_layer_name_for_summary(cls, render_summary) -> str:
        if (
            cls._allen_layer_output_mode_for_summary(render_summary)
            == ALLEN_LAYER_OUTPUT_PROJECTION
        ):
            return _ALLEN_LAYER_PROJECTION_LAYER_NAME
        return _ALLEN_LAYER_HEATMAP_LAYER_NAME

    def _axis_labels_for_render_mode(self, render_mode: str) -> tuple[str, ...]:
        """Return the dims captions matching a render mode's coordinate space."""
        if render_mode == _RENDER_ALLEN_LAYERS:
            return (
                self._flat_axis_labels()
                if self._is_allen_layer_projection()
                else self._allen_layer_axis_labels()
            )
        if render_mode in {_RENDER_FLAT_HEATMAP, _RENDER_FLAT_VECTOR}:
            return self._flat_axis_labels()
        return self._depth_axis_labels()

    def _create_or_update_allen_layer_stack(
        self,
        stack_result: AllenLayerStackResult,
        projection_summary: ProjectionSummary,
        *,
        flatmap_style: str,
        coordinate_mode: str,
    ):
        if stack_result.summary.rendered_nodes == 0:
            return None

        color_mode = self._current_heatmap_color_mode()
        metadata = self._render_metadata(
            projection_summary,
            stack_result.summary,
            flatmap_style=flatmap_style,
            coordinate_mode=coordinate_mode,
            render_mode=_RENDER_ALLEN_LAYERS,
            heatmap_color_mode=color_mode,
        )
        axis_labels = self._allen_layer_axis_labels_for_summary(stack_result.summary)
        layer_name = self._allen_layer_name_for_summary(stack_result.summary)
        if color_mode == _HEATMAP_COLOR_SINGLE:
            self._remove_projection_layer(except_name=layer_name)
            layer = self._cached_projection_layer_for_name(
                layer_name
            ) or self._find_layer_by_name(layer_name)
            layer = self._create_or_update_heatmap_layer_from_volume(
                layer,
                stack_result.volume,
                metadata,
                layer_name=layer_name,
                axis_labels=axis_labels,
            )
        else:
            self._remove_projection_layer()
            groups = self._grouped_allen_layer_volumes(
                stack_result,
                heatmap_color_mode=color_mode,
            )
            layers = []
            for group in groups:
                color = self._color_for_heatmap_group(
                    group,
                    heatmap_color_mode=color_mode,
                )
                grouped_layer = self._add_grouped_heatmap_layer(
                    group,
                    metadata,
                    color,
                    heatmap_color_mode=color_mode,
                    render_mode=_RENDER_ALLEN_LAYERS,
                    axis_labels=axis_labels,
                )
                self._set_layer_state(
                    grouped_layer,
                    stack_result.projected_nodes,
                    projection_summary,
                    stack_result.summary,
                )
                layers.append(grouped_layer)
            layer = layers[0] if layers else None

        self._projection_layer = layer
        if layer is None:
            return None
        self._set_layer_state(
            layer,
            stack_result.projected_nodes,
            projection_summary,
            stack_result.summary,
        )
        self._focus_projection_view(
            layer,
            stack_result.volume,
            ndisplay=2,
        )
        self._notify_display_viewer_ready(layer)
        return layer

    def _grouped_allen_layer_volumes(
        self,
        stack_result: AllenLayerStackResult,
        *,
        heatmap_color_mode: str,
    ) -> list[FlatmapGroupedVolume]:
        volume_shape = tuple(int(size) for size in stack_result.volume.shape)
        if heatmap_color_mode == _HEATMAP_COLOR_INDIVIDUAL:
            return build_allen_layer_file_id_volumes(
                stack_result.projected_nodes,
                volume_shape,
            )
        if heatmap_color_mode == _HEATMAP_COLOR_CLUSTER:
            return build_allen_layer_cluster_volumes(
                stack_result.projected_nodes,
                volume_shape,
                self._cluster_map_provider() or {},
            )
        return []

    def _create_or_update_render_layer(
        self,
        render_result: FlatmapRenderResult,
        projection_summary: ProjectionSummary,
        *,
        flatmap_style: str,
        coordinate_mode: str,
        render_mode: str,
    ):
        """Create or update the napari depth-aware flatmap render layer."""
        if render_result.summary.rendered_nodes == 0:
            return None

        # Validate and build vectors before changing the visible scene. Both
        # the segment limit and a selection with no usable edges are expected
        # user-facing failures, and a failed re-render must keep the previous
        # valid flatmap intact.
        vectors = (
            self._flat_vector_data(render_result)
            if render_mode == _RENDER_FLAT_VECTOR
            else None
        )

        grouped_capable = render_mode in {_RENDER_HEATMAP, _RENDER_FLAT_HEATMAP}
        heatmap_color_mode = (
            self._current_heatmap_color_mode()
            if grouped_capable
            else _HEATMAP_COLOR_SINGLE
        )
        layer_name = self._render_layer_name(render_mode)
        if grouped_capable and heatmap_color_mode != _HEATMAP_COLOR_SINGLE:
            self._remove_projection_layer()
        else:
            self._remove_projection_layer(except_name=layer_name)
        metadata = self._render_metadata(
            projection_summary,
            render_result.summary,
            flatmap_style=flatmap_style,
            coordinate_mode=coordinate_mode,
            render_mode=render_mode,
            heatmap_color_mode=heatmap_color_mode,
        )
        layer = self._cached_projection_layer_for_name(
            layer_name
        ) or self._find_layer_by_name(layer_name)

        axis_labels = self._axis_labels_for_render_mode(render_mode)
        data: np.ndarray = render_result.volume
        data_kind = "image"

        if render_mode == _RENDER_FLAT_VECTOR:
            assert vectors is not None
            metadata = dict(metadata)
            metadata["flatmap_vector_segments"] = int(len(vectors.data))
            metadata["flatmap_vector_total_segments"] = int(vectors.total_segments)
            layer = self._create_or_update_flat_vector_layer(
                layer,
                vectors,
                metadata,
                axis_labels=axis_labels,
            )
            data = vectors.data
            data_kind = "vectors"
        elif render_mode == _RENDER_POINTS:
            layer = self._create_or_update_points_layer(layer, render_result, metadata)
            data = render_result.points
            data_kind = "points"
        elif heatmap_color_mode == _HEATMAP_COLOR_SINGLE:
            layer = self._create_or_update_heatmap_layer_from_volume(
                layer,
                render_result.volume,
                metadata,
                layer_name=layer_name,
                axis_labels=axis_labels,
            )
        else:
            layers = self._create_grouped_heatmap_layers(
                render_result,
                projection_summary,
                metadata,
                heatmap_color_mode=heatmap_color_mode,
                render_mode=render_mode,
                axis_labels=axis_labels,
            )
            layer = layers[0] if layers else None

        self._projection_layer = layer
        if layer is None:
            return None
        self._set_layer_state(
            layer,
            render_result.projected_nodes,
            projection_summary,
            render_result.summary,
        )
        self._focus_projection_view(
            layer,
            data,
            ndisplay=self._render_ndisplay(render_mode),
            data_kind=data_kind,
        )
        self._notify_display_viewer_ready(layer)
        return layer

    def _flat_vector_data(
        self,
        render_result: FlatmapRenderResult,
    ) -> FlatmapSegmentVectors:
        """Build napari vector data for a depth-collapsed flatmap render.

        Edges are selected with the render's own ``render_valid`` column so the
        drawn traces cover exactly the nodes the render included, and the grid
        comes from the render summary so vectors land on the same pixels a 2D
        heatmap of the same projection would.
        """
        summary = render_result.summary
        rendered_nodes = int(summary.rendered_nodes)
        if rendered_nodes > MAX_FLATMAP_VECTOR_SEGMENTS:
            # Each node contributes at most one parent edge, so this refuses
            # before the segment merge instead of after building millions of rows.
            raise RuntimeError(
                f"2D Vector mode would draw up to {rendered_nodes:,} flatmap "
                f"segments, above the {MAX_FLATMAP_VECTOR_SEGMENTS:,} limit "
                "napari can render interactively. Select fewer neurons, or use "
                "2D Heatmap to see the whole set."
            )
        segments = build_projected_segments(
            render_result.projected_nodes,
            validity_column="render_valid",
        )
        if len(segments.data) == 0:
            # An empty Vectors layer draws nothing and napari would warn about
            # its empty color array, so say why instead of rendering a blank.
            raise RuntimeError(
                "2D Vector mode draws parent-child edges, and no rendered node "
                "pair shares an edge. Select neurons with more than one "
                "flatmap-valid node, or use 2D Heatmap."
            )
        return build_flatmap_segment_vectors(
            segments.data,
            segments.file_ids,
            x_bounds=(summary.x_flat_min, summary.x_flat_max),
            y_bounds=(summary.y_flat_min, summary.y_flat_max),
            # Both counts come from the summary of the render these vectors are
            # drawn over, so the overlay cannot disagree with the heatmap grid.
            y_bins=summary.y_bins,
            x_bins=summary.x_bins,
        )

    def _create_or_update_flat_vector_layer(
        self,
        layer,
        vectors: FlatmapSegmentVectors,
        metadata: dict[str, object],
        *,
        axis_labels: tuple[str, ...],
    ):
        """Create or update the 2D flatmap Vectors layer."""
        colors = self._colors_for_file_ids(list(vectors.file_ids))
        if layer is None:
            # edge_color is assigned after creation so napari enters DIRECT
            # color mode and honors one color per vector.
            layer = self._display_viewer().add_vectors(
                vectors.data,
                name=_FLAT_VECTOR_LAYER_NAME,
                edge_width=0.5,
                opacity=0.9,
                vector_style="line",
                blending="translucent",
                metadata=metadata,
                axis_labels=axis_labels,
            )
            layer.edge_color = colors
            return layer

        # Data and edge_color must land together: setting them separately lets
        # vispy rebuild the mesh against the old color array and mismatch face
        # counts.
        blocker = getattr(getattr(layer, "events", None), "blocker_all", None)
        if callable(blocker):
            with blocker():
                layer.data = vectors.data
                layer.edge_color = colors
                layer.metadata = metadata
                if hasattr(layer, "axis_labels"):
                    layer.axis_labels = axis_labels
                layer.visible = True
        else:
            layer.data = vectors.data
            layer.edge_color = colors
            layer.metadata = metadata
            if hasattr(layer, "axis_labels"):
                layer.axis_labels = axis_labels
            layer.visible = True
        refresh = getattr(layer, "refresh", None)
        if callable(refresh):
            refresh()
        return layer

    def _create_or_update_heatmap_layer_from_volume(
        self,
        layer,
        volume: np.ndarray,
        metadata: dict[str, object],
        *,
        layer_name: str = _HEATMAP_LAYER_NAME,
        axis_labels: tuple[str, ...] | None = None,
    ):
        # Every heatmap volume is a plane stack over flatmap XY; callers that do
        # not name their plane axis are rendering depth bins.
        axis_labels = axis_labels or self._depth_axis_labels()
        contrast_limits = self._heatmap_contrast_limits(volume)
        metadata = dict(metadata)
        metadata["flatmap_heatmap_contrast_limits"] = contrast_limits
        if layer is None:
            kwargs = dict(
                name=layer_name,
                colormap="hot",
                blending="additive",
                rendering="mip",
                opacity=0.8,
                contrast_limits=contrast_limits,
                metadata=metadata,
            )
            if axis_labels is not None:
                kwargs["axis_labels"] = axis_labels
            layer = self._display_viewer().add_image(volume, **kwargs)
            self._install_heatmap_layer_workarounds(layer)
            self._store_heatmap_contrast_limits(layer, contrast_limits)
            return layer

        self._install_heatmap_layer_workarounds(layer)
        blocker = getattr(getattr(layer, "events", None), "blocker_all", None)
        if callable(blocker):
            with blocker():
                layer.data = volume
                layer.metadata = metadata
                if axis_labels is not None and hasattr(layer, "axis_labels"):
                    layer.axis_labels = axis_labels
                self._apply_heatmap_contrast_limits(layer, contrast_limits)
                layer.visible = True
        else:
            layer.data = volume
            layer.metadata = metadata
            if axis_labels is not None and hasattr(layer, "axis_labels"):
                layer.axis_labels = axis_labels
            self._apply_heatmap_contrast_limits(layer, contrast_limits)
            layer.visible = True
        refresh = getattr(layer, "refresh", None)
        if callable(refresh):
            refresh()
        self._store_heatmap_contrast_limits(layer, contrast_limits)
        return layer

    def _create_grouped_heatmap_layers(
        self,
        render_result: FlatmapRenderResult,
        projection_summary: ProjectionSummary,
        metadata: dict[str, object],
        *,
        heatmap_color_mode: str,
        render_mode: str = _RENDER_HEATMAP,
        axis_labels: tuple[str, ...] | None = None,
    ) -> list[object]:
        groups = self._grouped_heatmap_volumes(
            render_result,
            heatmap_color_mode=heatmap_color_mode,
        )
        layers = []
        for group in groups:
            color = self._color_for_heatmap_group(
                group,
                heatmap_color_mode=heatmap_color_mode,
            )
            layer = self._add_grouped_heatmap_layer(
                group,
                metadata,
                color,
                heatmap_color_mode=heatmap_color_mode,
                render_mode=render_mode,
                axis_labels=axis_labels,
            )
            self._set_layer_state(
                layer,
                render_result.projected_nodes,
                projection_summary,
                render_result.summary,
            )
            layers.append(layer)
        return layers

    def _grouped_heatmap_volumes(
        self,
        render_result: FlatmapRenderResult,
        *,
        heatmap_color_mode: str,
    ) -> list[FlatmapGroupedVolume]:
        if heatmap_color_mode == _HEATMAP_COLOR_INDIVIDUAL:
            return build_flatmap_file_id_volumes(
                render_result.projected_nodes,
                tuple(render_result.volume.shape),
            )
        if heatmap_color_mode == _HEATMAP_COLOR_CLUSTER:
            return build_flatmap_cluster_volumes(
                render_result.projected_nodes,
                tuple(render_result.volume.shape),
                self._cluster_map_provider() or {},
            )
        return []

    @staticmethod
    def _grouped_heatmap_layer_name(
        group: FlatmapGroupedVolume,
        *,
        heatmap_color_mode: str,
        render_mode: str = _RENDER_HEATMAP,
        allen_layer_output_mode: str = ALLEN_LAYER_OUTPUT_STACK,
    ) -> str:
        if render_mode == _RENDER_ALLEN_LAYERS:
            prefix = (
                _GROUPED_ALLEN_LAYER_PROJECTION_PREFIX
                if allen_layer_output_mode == ALLEN_LAYER_OUTPUT_PROJECTION
                else _GROUPED_ALLEN_LAYER_PREFIX
            )
        elif render_mode == _RENDER_FLAT_HEATMAP:
            prefix = _GROUPED_FLAT_HEATMAP_PREFIX
        else:
            prefix = _GROUPED_HEATMAP_LAYER_PREFIX
        if heatmap_color_mode == _HEATMAP_COLOR_CLUSTER:
            return f"{prefix}{group.label}"
        return f"{prefix}{group.label}"

    def _color_for_heatmap_group(
        self,
        group: FlatmapGroupedVolume,
        *,
        heatmap_color_mode: str,
    ) -> np.ndarray:
        if heatmap_color_mode == _HEATMAP_COLOR_CLUSTER and group.group_key is None:
            return _DEFAULT_TRACE_COLOR.copy()
        color_map = self._color_map_provider() or {}
        for file_id in group.source_file_ids:
            if file_id in color_map or str(file_id) in color_map:
                return self._color_for_file_id(file_id, color_map)
        return _DEFAULT_TRACE_COLOR.copy()

    @staticmethod
    def _solid_tint_colormap(color: np.ndarray, name: str):
        rgba = np.asarray(color, dtype=float).reshape(-1)
        if rgba.size < 4:
            rgba = np.pad(rgba, (0, 4 - rgba.size), constant_values=1.0)
        rgba = np.clip(rgba[:4], 0.0, 1.0)
        try:
            from napari.utils.colormaps import Colormap
        except Exception:
            return "hot"
        return Colormap(
            colors=[
                [0.0, 0.0, 0.0, 0.0],
                [float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])],
            ],
            name=name,
        )

    def _add_grouped_heatmap_layer(
        self,
        group: FlatmapGroupedVolume,
        metadata: dict[str, object],
        color: np.ndarray,
        *,
        heatmap_color_mode: str,
        render_mode: str = _RENDER_HEATMAP,
        axis_labels: tuple[str, ...] | None = None,
    ):
        axis_labels = axis_labels or self._depth_axis_labels()
        volume = group.volume
        contrast_limits, contrast_limits_range = self._grouped_heatmap_contrast_limits(
            volume,
            heatmap_color_mode=heatmap_color_mode,
        )
        group_metadata = dict(metadata)
        color_values = [float(value) for value in np.asarray(color)[:4]]
        group_metadata.update(
            {
                "flatmap_heatmap_color_mode": heatmap_color_mode,
                "flatmap_heatmap_group_key": group.group_key,
                "flatmap_heatmap_group_label": group.label,
                "flatmap_heatmap_group_color": color_values,
                "flatmap_heatmap_group_rendered_nodes": group.rendered_nodes,
                "flatmap_heatmap_group_nonzero_voxels": group.nonzero_voxels,
                "source_file_ids": list(group.source_file_ids),
                "file_ids": list(group.source_file_ids),
                "color": color_values,
                "flatmap_heatmap_contrast_limits": contrast_limits,
                "flatmap_heatmap_contrast_limits_range": contrast_limits_range,
            }
        )
        layer_name = self._grouped_heatmap_layer_name(
            group,
            heatmap_color_mode=heatmap_color_mode,
            render_mode=render_mode,
            allen_layer_output_mode=str(
                metadata.get(
                    "allen_layer_output_mode",
                    ALLEN_LAYER_OUTPUT_STACK,
                )
            ),
        )
        kwargs = dict(
            name=layer_name,
            colormap=self._solid_tint_colormap(color, layer_name),
            blending="additive",
            rendering="mip",
            opacity=0.8,
            contrast_limits=contrast_limits,
            metadata=group_metadata,
        )
        if axis_labels is not None:
            kwargs["axis_labels"] = axis_labels
        layer = self._display_viewer().add_image(volume, **kwargs)
        self._install_heatmap_layer_workarounds(layer)
        self._store_heatmap_contrast_limits(
            layer,
            contrast_limits,
            limits_range=contrast_limits_range,
        )
        if contrast_limits_range != contrast_limits:
            # napari narrows the slider range to whatever contrast_limits it was
            # given, so widen it back to the real data span. Without this the
            # user could not raise the limit past the opening fraction.
            self._apply_heatmap_contrast_limits(
                layer,
                contrast_limits,
                limits_range=contrast_limits_range,
            )
        return layer

    @classmethod
    def _grouped_heatmap_contrast_limits(
        cls,
        volume: np.ndarray,
        *,
        heatmap_color_mode: str,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``(display limits, slider range)`` for a grouped heatmap layer.

        Per-neuron layers open at a fraction of their own maximum so long-range
        projections are visible, while the slider keeps the full data range so
        the dense core can still be inspected.  Cluster groups aggregate many
        neurons and keep the full range as their opening window.
        """
        full_range = cls._heatmap_contrast_limits(volume)
        if heatmap_color_mode != _HEATMAP_COLOR_INDIVIDUAL:
            return full_range, full_range
        peak = float(np.nanmax(volume)) if volume.size else 0.0
        if not np.isfinite(peak) or peak <= 0.0:
            # No counts to take a fraction of, so the neutral unit fallback
            # from _heatmap_contrast_limits stands unscaled.
            return full_range, full_range
        lower, upper = full_range
        scaled = upper * _INDIVIDUAL_HEATMAP_CONTRAST_FRACTION
        if not np.isfinite(scaled) or scaled <= lower:
            return full_range, full_range
        return (lower, scaled), full_range

    def _install_heatmap_layer_workarounds(self, layer) -> None:
        self._install_heatmap_status_guard(layer)
        self._install_heatmap_thumbnail_workarounds(layer)

    def _install_heatmap_status_guard(self, layer) -> None:
        """Avoid napari status errors while a heatmap slice catches up to 3D."""
        if getattr(layer, "_napari_neuron_navigator_flatmap_status_guard_installed", False):
            return

        original_get_status = getattr(layer, "get_status", None)
        if not callable(original_get_status):
            return

        def guarded_get_status(
            position=None,
            *,
            view_direction=None,
            dims_displayed=None,
            world=False,
            value=None,
        ):
            try:
                return original_get_status(
                    position,
                    view_direction=view_direction,
                    dims_displayed=dims_displayed,
                    world=world,
                    value=value,
                )
            except IndexError as exc:
                if not self._is_stale_3d_status_slice(layer, dims_displayed, exc):
                    raise
                return self._status_without_sampled_value(layer, position)

        setattr(layer, "_napari_neuron_navigator_flatmap_original_get_status", original_get_status)
        setattr(layer, "get_status", guarded_get_status)
        setattr(layer, "_napari_neuron_navigator_flatmap_status_guard_installed", True)

    def _install_heatmap_thumbnail_workarounds(self, layer) -> None:
        """Keep generated heatmap thumbnails stable across 2D/3D axis changes."""
        if getattr(layer, "_napari_neuron_navigator_flatmap_thumbnail_workarounds_installed", False):
            return

        widget = self
        original_update_thumbnail = getattr(layer, "_update_thumbnail", None)
        if callable(original_update_thumbnail):

            def safe_update_thumbnail(bound_layer) -> None:
                try:
                    original_update_thumbnail()
                except RuntimeError as error:
                    if not widget._is_thumbnail_rank_mismatch_error(error):
                        raise
                    if not getattr(
                        bound_layer,
                        "_napari_neuron_navigator_flatmap_thumbnail_warning_logged",
                        False,
                    ):
                        logger.warning(
                            "Suppressed napari thumbnail update failure for "
                            "flatmap heatmap '%s': %s",
                            getattr(bound_layer, "name", "<unnamed>"),
                            error,
                        )
                        bound_layer._napari_neuron_navigator_flatmap_thumbnail_warning_logged = True

            layer._update_thumbnail = MethodType(safe_update_thumbnail, layer)

        original_reset_contrast_limits = getattr(layer, "reset_contrast_limits", None)
        if callable(original_reset_contrast_limits):

            def stable_reset_contrast_limits(bound_layer, mode=None) -> None:
                if not widget._heatmap_requires_stable_limits(bound_layer):
                    original_reset_contrast_limits(mode)
                    return
                limits = widget._heatmap_stored_contrast_limits(bound_layer)
                if limits is None:
                    original_reset_contrast_limits(mode)
                    return
                widget._apply_heatmap_contrast_limits(
                    bound_layer,
                    limits,
                    widget._heatmap_stored_contrast_limits_range(bound_layer),
                )

            layer.reset_contrast_limits = MethodType(
                stable_reset_contrast_limits, layer
            )

        original_reset_contrast_limits_range = getattr(
            layer,
            "reset_contrast_limits_range",
            None,
        )
        if callable(original_reset_contrast_limits_range):

            def stable_reset_contrast_limits_range(bound_layer, mode=None) -> None:
                if not widget._heatmap_requires_stable_limits(bound_layer):
                    original_reset_contrast_limits_range(mode)
                    return
                limits = widget._heatmap_stored_contrast_limits_range(bound_layer)
                if limits is None:
                    original_reset_contrast_limits_range(mode)
                    return
                bound_layer.contrast_limits_range = limits

            layer.reset_contrast_limits_range = MethodType(
                stable_reset_contrast_limits_range,
                layer,
            )

        original_update_slice_response = getattr(layer, "_update_slice_response", None)
        if callable(original_update_slice_response):

            def stable_update_slice_response(bound_layer, response):
                keep_auto = bool(getattr(bound_layer, "_keep_auto_contrast", False))
                if not keep_auto or not widget._heatmap_requires_stable_limits(
                    bound_layer,
                    response,
                ):
                    return original_update_slice_response(response)

                bound_layer._keep_auto_contrast = False
                try:
                    result = original_update_slice_response(response)
                finally:
                    bound_layer._keep_auto_contrast = True

                limits = widget._heatmap_stored_contrast_limits(bound_layer)
                if limits is not None:
                    widget._apply_heatmap_contrast_limits(
                        bound_layer,
                        limits,
                        widget._heatmap_stored_contrast_limits_range(bound_layer),
                    )
                return result

            layer._update_slice_response = MethodType(
                stable_update_slice_response,
                layer,
            )

        setattr(layer, "_napari_neuron_navigator_flatmap_thumbnail_workarounds_installed", True)

    @staticmethod
    def _is_thumbnail_rank_mismatch_error(error: RuntimeError) -> bool:
        return "sequence argument must have length equal to input rank" in str(error)

    @staticmethod
    def _heatmap_ndisplay(layer, response=None) -> int | None:
        slice_input = getattr(response, "slice_input", None)
        ndisplay = getattr(slice_input, "ndisplay", None)
        if isinstance(ndisplay, (int, np.integer)):
            return int(ndisplay)

        slice_input = getattr(layer, "_slice_input", None)
        ndisplay = getattr(slice_input, "ndisplay", None)
        if isinstance(ndisplay, (int, np.integer)):
            return int(ndisplay)
        return None

    def _heatmap_requires_stable_limits(self, layer, response=None) -> bool:
        return self._heatmap_ndisplay(layer, response) == 3

    @staticmethod
    def _coerce_contrast_limits(raw_limits) -> tuple[float, float] | None:
        """Return a finite, ascending ``(lower, upper)`` pair, or ``None``."""
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

    @classmethod
    def _heatmap_stored_contrast_limits(cls, layer) -> tuple[float, float] | None:
        raw_limits = getattr(
            layer,
            "_napari_neuron_navigator_flatmap_heatmap_contrast_limits",
            None,
        )
        if raw_limits is None:
            metadata = getattr(layer, "metadata", None)
            if isinstance(metadata, dict):
                raw_limits = metadata.get("flatmap_heatmap_contrast_limits")
        return cls._coerce_contrast_limits(raw_limits)

    @staticmethod
    def _store_heatmap_contrast_limits(
        layer,
        limits: tuple[float, float],
        *,
        limits_range: tuple[float, float] | None = None,
    ) -> None:
        setattr(layer, "_napari_neuron_navigator_flatmap_heatmap_contrast_limits", limits)
        resolved_range = limits_range or limits
        setattr(
            layer,
            "_napari_neuron_navigator_flatmap_heatmap_contrast_limits_range",
            resolved_range,
        )
        metadata = getattr(layer, "metadata", None)
        if isinstance(metadata, dict):
            metadata["flatmap_heatmap_contrast_limits"] = limits
            metadata["flatmap_heatmap_contrast_limits_range"] = resolved_range

    @classmethod
    def _heatmap_stored_contrast_limits_range(
        cls,
        layer,
    ) -> tuple[float, float] | None:
        """Return a layer's stored slider range, or its display limits.

        Layers stored before the range was tracked separately -- and every layer
        whose opening window already spans the data -- fall back to the display
        limits, which is what the range used to be pinned to.
        """
        stored = cls._coerce_contrast_limits(
            getattr(
                layer,
                "_napari_neuron_navigator_flatmap_heatmap_contrast_limits_range",
                None,
            )
        )
        if stored is not None:
            return stored
        metadata = getattr(layer, "metadata", None)
        if isinstance(metadata, dict):
            stored = cls._coerce_contrast_limits(
                metadata.get("flatmap_heatmap_contrast_limits_range")
            )
            if stored is not None:
                return stored
        return cls._heatmap_stored_contrast_limits(layer)

    @staticmethod
    def _apply_heatmap_contrast_limits(
        layer,
        limits: tuple[float, float],
        limits_range: tuple[float, float] | None = None,
    ) -> None:
        keep_auto = bool(getattr(layer, "_keep_auto_contrast", False))
        if keep_auto:
            layer._keep_auto_contrast = False
        try:
            # The range is set first so a narrower display window is not clamped
            # to a stale, smaller range.
            layer.contrast_limits_range = limits_range or limits
            layer.contrast_limits = limits
        finally:
            if hasattr(layer, "_keep_auto_contrast"):
                layer._keep_auto_contrast = keep_auto

    @staticmethod
    def _is_stale_3d_status_slice(layer, dims_displayed, exc: IndexError) -> bool:
        if dims_displayed is None or len(dims_displayed) != 3:
            return False

        raw = getattr(
            getattr(getattr(layer, "_slice", None), "image", None),
            "raw",
            None,
        )
        if raw is not None and np.asarray(raw).ndim < len(dims_displayed):
            return True

        return "too many indices for array" in str(exc)

    @staticmethod
    def _status_without_sampled_value(layer, position) -> dict[str, str]:
        source_info = getattr(layer, "_get_source_info", None)
        if callable(source_info):
            status = source_info().copy()
        else:
            name = str(getattr(layer, "name", ""))
            status = {
                "layer_name": name,
                "layer_base": name,
                "source_type": "",
                "plugin": "",
            }

        coords_str = ""
        if position is not None:
            ndim = int(getattr(layer, "ndim", 0) or 0)
            coords = np.asarray(position)
            if ndim > 0:
                coords = coords[-ndim:]
            rounded = np.round(coords).astype(int)
            coords_str = f" [{' '.join(map(str, rounded))}]"

        status["coordinates"] = ": ".join((coords_str, ""))
        status["coords"] = coords_str
        status["value"] = ""
        return status

    def _create_or_update_points_layer(
        self,
        layer,
        render_result: FlatmapRenderResult,
        metadata: dict[str, object],
    ):
        points = render_result.points
        colors = self._colors_for_file_ids(render_result.point_file_ids)
        if layer is None:
            return self._display_viewer().add_points(
                points,
                name=_POINTS_LAYER_NAME,
                size=2.0,
                face_color=colors,
                border_width=0.0,
                blending="translucent",
                metadata=metadata,
            )

        blocker = getattr(getattr(layer, "events", None), "blocker_all", None)
        if callable(blocker):
            with blocker():
                layer.data = points
                layer.face_color = colors
                layer.metadata = metadata
                layer.visible = True
        else:
            layer.data = points
            layer.face_color = colors
            layer.metadata = metadata
            layer.visible = True
        refresh = getattr(layer, "refresh", None)
        if callable(refresh):
            refresh()
        return layer

    def _cached_soma_layer(self):
        """Return the tracked soma layer if it still lives in the viewer."""
        layer = getattr(self, "_soma_layer", None)
        if layer is None:
            return None
        if not self._layer_is_in_viewer(layer):
            self._soma_layer = None
            return None
        return layer

    def _soma_plane_column(self) -> str | None:
        """Return the axis-0 bin column of the current render's point space.

        ``None`` means the render has no plane axis, so soma points are 2-D.
        """
        if self._is_allen_layer_mode():
            if self._is_allen_layer_projection():
                return None
            return "allen_layer_render_index"
        if self._is_flat_render_mode():
            return None
        return "depth_bin"

    def _remove_soma_layer(self) -> None:
        """Drop a soma layer whose bin space belongs to a previous render.

        The soma layer is deliberately absent from
        ``_FLATMAP_RENDER_LAYER_NAMES`` so an ordinary re-projection leaves it
        alone.  Only a change of render mode, style, or cache grid invalidates
        its coordinates, and those all route through here.
        """
        layer = self._cached_soma_layer() or self._find_layer_by_name(
            _SOMA_POINTS_LAYER_NAME,
            create=False,
        )
        self._soma_layer = None
        if layer is None:
            return
        layers = self._display_layers(create=False)
        if layers is None:
            return
        try:
            layers.remove(layer)
        except (ValueError, KeyError, RuntimeError):
            logger.debug(
                "Flatmap soma layer was already removed.",
                exc_info=True,
            )

    def _create_or_update_soma_layer(
        self,
        render_result: FlatmapRenderResult | AllenLayerStackResult,
        projection_summary: ProjectionSummary,
    ):
        """Create or update the dedicated flatmap soma point layer.

        Coordinates come from the bin columns the render itself wrote, so the
        somas share the visible render's plane axis instead of always using
        depth bins.
        """
        render_mode = self._current_render_mode()
        plane_column = self._soma_plane_column()
        if (
            plane_column == "allen_layer_render_index"
            and plane_column not in render_result.projected_nodes.columns
        ):
            plane_column = "allen_layer_index"
        points, point_file_ids = rendered_plane_points(
            render_result.projected_nodes,
            plane_column=plane_column,
        )
        if len(points) == 0:
            return None

        colors = self._colors_for_file_ids(point_file_ids)
        axis_labels = self._axis_labels_for_render_mode(render_mode)
        metadata = {
            _FLATMAP_LAYER_SPACE_KEY: _FLATMAP_LAYER_SPACE_VALUE,
            "projection_kind": "isocortex_flatmap",
            "flatmap_render_mode": _RENDER_POINTS,
            "flatmap_soma_only": True,
            "flatmap_soma_space_render_mode": render_mode,
            "flatmap_plane_mode": self._current_plane_mode(),
            "projection_summary": projection_summary.to_dict(),
            "render_summary": render_result.summary.to_dict(),
            "flatmap_path": str(self._flatmap_path) if self._flatmap_path else "",
            "depth_path": str(self._depth_path) if self._depth_path else "",
        }
        layer_labels = getattr(render_result.summary, "layer_labels", None)
        if layer_labels is not None:
            # Name the planes from this render rather than the module default.
            metadata["allen_layer_labels"] = [str(label) for label in layer_labels]
            metadata["allen_layer_indices"] = [
                int(value)
                for value in getattr(
                    render_result.summary,
                    "selected_layer_indices",
                    tuple(range(len(layer_labels))),
                )
            ]
            metadata["allen_layer_output_mode"] = str(
                getattr(
                    render_result.summary,
                    "output_mode",
                    ALLEN_LAYER_OUTPUT_STACK,
                )
            )

        layer = self._cached_soma_layer() or self._find_layer_by_name(
            _SOMA_POINTS_LAYER_NAME
        )
        if layer is None:
            layer = self._display_viewer().add_points(
                points,
                name=_SOMA_POINTS_LAYER_NAME,
                size=1.0,
                face_color=colors,
                border_width=0.0,
                blending="translucent",
                metadata=metadata,
                # Without axis captions the soma layer looks like a foreign
                # layer to _apply_display_axis_annotations, which would then
                # clear the render's plane caption and axis names.
                axis_labels=axis_labels,
            )
        else:
            blocker = getattr(getattr(layer, "events", None), "blocker_all", None)
            if callable(blocker):
                with blocker():
                    layer.data = points
                    layer.face_color = colors
                    layer.metadata = metadata
                    if hasattr(layer, "axis_labels"):
                        layer.axis_labels = axis_labels
                    layer.visible = True
            else:
                layer.data = points
                layer.face_color = colors
                layer.metadata = metadata
                if hasattr(layer, "axis_labels"):
                    layer.axis_labels = axis_labels
                layer.visible = True
            refresh = getattr(layer, "refresh", None)
            if callable(refresh):
                refresh()

        self._soma_layer = layer
        self._focus_projection_view(
            layer,
            points,
            ndisplay=self._render_ndisplay(render_mode),
            data_kind="points",
        )
        self._notify_display_viewer_ready(layer)
        return layer

    @staticmethod
    def _render_bounds_for_focus(
        data: np.ndarray,
        data_kind: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(lower, upper)`` coordinate bounds of rendered data.

        ``data_kind`` names the layout: ``"image"`` for a node-count array of any
        rank, ``"points"`` for ``(N, ndim)`` coordinates, and ``"vectors"`` for
        napari's ``(M, 2, ndim)`` ``[start, direction]`` pairs.  Returns ``None``
        when nothing finite is present to bound.
        """
        array = np.asarray(data, dtype=float)
        if data_kind == "image":
            coords = np.argwhere(array > 0)
            if len(coords) == 0:
                return (
                    np.zeros(array.ndim, dtype=float),
                    np.asarray(array.shape, dtype=float) - 1.0,
                )
            return (
                np.min(coords, axis=0).astype(float),
                np.max(coords, axis=0).astype(float),
            )

        if data_kind == "vectors":
            if array.ndim != 3 or array.shape[0] == 0:
                return None
            starts = array[:, 0, :]
            # Vectors store a direction, so the far end has to be reconstructed.
            coords = np.vstack((starts, starts + array[:, 1, :]))
        else:
            coords = array.reshape(len(array), -1) if array.ndim > 1 else array
            if coords.ndim != 2 or coords.shape[0] == 0:
                return None

        finite_mask = np.all(np.isfinite(coords), axis=1)
        if not finite_mask.any():
            return None
        finite = coords[finite_mask]
        return np.min(finite, axis=0), np.max(finite, axis=0)

    def _focus_projection_view(
        self,
        layer,
        data: np.ndarray,
        *,
        ndisplay: int = 3,
        data_kind: str = "image",
    ) -> None:
        """Set display dimensionality and center on the flatmap render bounds."""
        try:
            layer.visible = True
        except Exception:
            pass

        viewer = self._display_viewer()
        dims = getattr(viewer, "dims", None)
        if dims is not None and getattr(dims, "ndisplay", None) != ndisplay:
            try:
                dims.ndisplay = int(ndisplay)
            except Exception:
                logger.debug(
                    "Failed to switch flatmap display dimensionality.",
                    exc_info=True,
                )
        self._reslice_layer_for_current_dims(layer)
        # Applied before the camera work below, which has early returns.
        self._apply_display_axis_annotations(layer)

        layers = getattr(viewer, "layers", None)
        selection = getattr(layers, "selection", None)
        if selection is not None:
            try:
                selection.active = layer
            except Exception:
                logger.debug("Failed to activate flatmap layer.", exc_info=True)

        bounds = self._render_bounds_for_focus(data, data_kind)
        if bounds is None:
            return
        lower, upper = bounds
        center = tuple(((lower + upper) / 2.0).tolist())
        span = float(np.max(upper - lower))

        camera = getattr(getattr(viewer, "scene", None), "camera", None)
        if camera is None:
            reset_view = getattr(viewer, "reset_view", None)
            if callable(reset_view):
                reset_view()
            return

        try:
            camera.center = center
        except Exception:
            logger.debug("Failed to center camera on flatmap layer.", exc_info=True)

        if span > 0.0 and np.isfinite(span):
            try:
                camera.zoom = float(np.clip(600.0 / span, 0.01, 10_000.0))
            except Exception:
                logger.debug("Failed to zoom camera to flatmap layer.", exc_info=True)

    def _plane_labels_for_layer(self, layer) -> tuple[str, ...] | None:
        """Return one label per plane of a flatmap layer, or ``None`` if unknown.

        ``None`` is a real answer: the depth-mode region-labels layer records no
        depth bin size, so naming its planes in microns would be invented.
        """
        metadata = getattr(layer, "metadata", None) or {}
        try:
            plane_mode = metadata.get("flatmap_plane_mode")
        except AttributeError:
            return None
        if plane_mode == FLATMAP_PLANE_MODE_ALLEN_LAYERS:
            labels = metadata.get("allen_layer_labels") or ALLEN_ISOCORTEX_LAYER_LABELS
            return tuple(str(label) for label in labels)
        if plane_mode == FLATMAP_PLANE_MODE_FLAT:
            # A collapsed render has a single unnamed plane; there is no depth
            # bin or atlas layer to caption.
            return None
        if plane_mode == FLATMAP_PLANE_MODE_DEPTH:
            render_summary = metadata.get("render_summary")
            if isinstance(render_summary, Mapping):
                return depth_plane_labels(render_summary) or None
        return None

    @staticmethod
    def _flatmap_axis_labels_for_layer(layer) -> tuple[str, ...] | None:
        """Return a layer's axis captions only when this widget set them.

        Points renders and any foreign layer keep napari's generic ``axis -1``
        style names, which must not be copied onto the viewer.
        """
        axis_labels = getattr(layer, "axis_labels", None)
        if not axis_labels:
            return None
        try:
            labels = tuple(str(label) for label in axis_labels)
        except TypeError:
            return None
        if labels[-2:] != (_FLATMAP_AXIS_LABEL_Y, _FLATMAP_AXIS_LABEL_X):
            return None
        return labels

    def _apply_display_axis_annotations(self, layer) -> None:
        """Name the display viewer's axes and show which plane is on screen.

        napari 0.9 derives ``viewer.dims.axis_labels`` from layer axis labels,
        so this method only manages the public axes and text overlay models.
        """
        viewer = self._current_display_viewer()
        if viewer is None or layer is None:
            return
        dims = getattr(viewer, "dims", None)
        if dims is None:
            return

        axis_labels = self._flatmap_axis_labels_for_layer(layer)
        if axis_labels is None:
            self._clear_display_axis_annotations()
            return

        state = self._capture_display_axis_annotation_state(viewer)
        if state is None:
            return

        axes = getattr(
            getattr(getattr(viewer, "scene", None), "overlays", None),
            "axes",
            None,
        )
        if axes is not None:
            try:
                axes.visible = True
                axes.labels = True
            except Exception:
                logger.debug("Failed to show the flatmap axes overlay.", exc_info=True)

        state["plane_labels"] = self._plane_labels_for_layer(layer)
        state["plane_caption"] = axis_labels[0]
        state["plane_count"] = self._plane_count_for_layer(
            layer,
            state["plane_labels"],
            axis_labels,
        )
        self._connect_display_dims_events(viewer)
        self._on_display_dims_step_changed()

    @staticmethod
    def _plane_count_for_layer(
        layer,
        plane_labels: tuple[str, ...] | None,
        axis_labels: tuple[str, ...],
    ) -> int:
        """Return how many planes a layer stacks, or ``0`` for none.

        ``axis_labels`` decides whether a plane axis exists at all: a
        depth-collapsed render names two axes and has no planes.  The data-shape
        fallback then applies only to layers whose array rank matches the axis
        count -- an image.  A Points array is ``(N, ndim)`` and a Vectors array
        is ``(M, 2, ndim)``, so their leading size counts records, not planes.
        """
        if plane_labels:
            return len(plane_labels)
        if len(axis_labels) < 3:
            return 0
        data = getattr(layer, "data", None)
        shape = getattr(data, "shape", None)
        if shape is not None and len(shape) == len(axis_labels):
            return int(shape[0])
        return 0

    def _capture_display_axis_annotation_state(self, viewer) -> dict | None:
        """Remember a viewer's pre-existing overlay state so it can be restored."""
        states = getattr(self, "_display_axis_annotation_states", None)
        if states is None:
            states = {}
            existing = getattr(self, "_display_axis_annotation_state", None)
            if existing is not None and existing.get("viewer") is not None:
                states[id(existing["viewer"])] = existing
            self._display_axis_annotation_states = states
        state = states.get(id(viewer))
        if state is not None and state.get("viewer") is viewer:
            self._display_axis_annotation_state = state
            return state

        axes = getattr(
            getattr(getattr(viewer, "scene", None), "overlays", None),
            "axes",
            None,
        )
        text_overlay = getattr(
            getattr(getattr(viewer, "canvas", None), "overlays", None),
            "text",
            None,
        )
        state = {
            "viewer": viewer,
            "connected": False,
            "plane_labels": None,
            "plane_caption": _DEPTH_AXIS_LABEL,
            "plane_count": 0,
            "previous_axes_visible": getattr(axes, "visible", None),
            "previous_axes_labels": getattr(axes, "labels", None),
            "previous_text_visible": getattr(text_overlay, "visible", None),
            "previous_text": getattr(text_overlay, "text", None),
        }
        states[id(viewer)] = state
        self._display_axis_annotation_state = state
        return state

    def _connect_display_dims_events(self, viewer) -> None:
        states = getattr(self, "_display_axis_annotation_states", {})
        state = states.get(id(viewer))
        if state is None:
            state = getattr(self, "_display_axis_annotation_state", None)
            if state is not None and state.get("viewer") is not viewer:
                state = None
        if state is None or state.get("connected"):
            return
        emitter = getattr(
            getattr(getattr(viewer, "dims", None), "events", None),
            "current_step",
            None,
        )
        connect = getattr(emitter, "connect", None)
        if not callable(connect):
            return
        callback = lambda event=None, target=viewer: (
            self._on_display_dims_step_changed(event, viewer=target)
        )
        try:
            connect(callback)
        except Exception:
            logger.debug(
                "Failed to follow the flatmap display slider.",
                exc_info=True,
            )
            return
        state["connected"] = True
        state["callback"] = callback

    def _on_display_dims_step_changed(self, event=None, *, viewer=None) -> None:
        """Write the on-canvas name of the plane currently under the slider."""
        if viewer is None:
            state = getattr(self, "_display_axis_annotation_state", None)
        else:
            state = getattr(self, "_display_axis_annotation_states", {}).get(
                id(viewer)
            )
        if state is None:
            return
        viewer = state.get("viewer")
        text_overlay = getattr(
            getattr(getattr(viewer, "canvas", None), "overlays", None),
            "text",
            None,
        )
        if text_overlay is None:
            return

        plane_count = int(state.get("plane_count") or 0)
        if plane_count <= 0:
            # A depth-collapsed render has no planes to name.  Leaving the
            # overlay alone would keep a previous render's caption (for example
            # "Allen layer: L2/3  (plane 2 of 6)") on a canvas that no longer
            # shows a plane stack, so retire it instead.
            try:
                text_overlay.text = ""
                text_overlay.visible = False
            except Exception:
                logger.debug(
                    "Failed to hide the flatmap plane label.",
                    exc_info=True,
                )
            return
        current_step = getattr(getattr(viewer, "dims", None), "current_step", None)
        try:
            index = int(current_step[0])
        except (IndexError, TypeError, ValueError):
            index = 0
        index = max(0, min(index, plane_count - 1))

        caption = str(state.get("plane_caption") or _DEPTH_AXIS_LABEL)
        plane_labels = state.get("plane_labels")
        position = f"plane {index + 1} of {plane_count}"
        if plane_labels:
            text = f"{caption}: {plane_labels[index]}  ({position})"
        else:
            text = f"{caption}: {position}"

        try:
            text_overlay.text = text
            text_overlay.position = _PLANE_TEXT_OVERLAY_POSITION
            text_overlay.font_size = _PLANE_TEXT_OVERLAY_FONT_SIZE
            text_overlay.visible = True
        except Exception:
            logger.debug("Failed to update the flatmap plane label.", exc_info=True)

    def _clear_display_axis_annotations(self, viewer=None) -> None:
        """Disconnect the slider follower and restore the viewer's own overlays."""
        states = getattr(self, "_display_axis_annotation_states", None)
        if viewer is None:
            viewer = self._current_display_viewer()
        state = (
            states.get(id(viewer))
            if states is not None and viewer is not None
            else None
        )
        if state is None:
            state = getattr(self, "_display_axis_annotation_state", None)
            if (
                viewer is not None
                and state is not None
                and state.get("viewer") is not viewer
            ):
                state = None
        if state is None:
            return
        if states is not None:
            states.pop(id(state.get("viewer")), None)
        if getattr(self, "_display_axis_annotation_state", None) is state:
            self._display_axis_annotation_state = None

        target = state.get("viewer")
        if target is None:
            return
        dims = getattr(target, "dims", None)
        if state.get("connected"):
            emitter = getattr(getattr(dims, "events", None), "current_step", None)
            disconnect = getattr(emitter, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect(
                        state.get("callback", self._on_display_dims_step_changed)
                    )
                except Exception:
                    logger.debug(
                        "Failed to stop following the flatmap display slider.",
                        exc_info=True,
                    )

        axes = getattr(
            getattr(getattr(target, "scene", None), "overlays", None),
            "axes",
            None,
        )
        text_overlay = getattr(
            getattr(getattr(target, "canvas", None), "overlays", None),
            "text",
            None,
        )
        for owner, attribute, key in (
            (axes, "visible", "previous_axes_visible"),
            (axes, "labels", "previous_axes_labels"),
            (
                text_overlay,
                "visible",
                "previous_text_visible",
            ),
            (text_overlay, "text", "previous_text"),
        ):
            previous = state.get(key)
            if owner is None or previous is None:
                continue
            try:
                setattr(owner, attribute, previous)
            except Exception:
                logger.debug(
                    "Failed to restore flatmap display overlay state.",
                    exc_info=True,
                )

    def _reslice_layer_for_current_dims(self, layer) -> None:
        viewer = self._current_display_viewer()
        dims = getattr(viewer, "dims", None)
        if dims is None:
            return
        slice_dims = getattr(layer, "_slice_dims", None)
        if not callable(slice_dims):
            return
        try:
            slice_dims(dims, force=True)
        except Exception:
            logger.debug("Failed to refresh flatmap layer slice.", exc_info=True)

    def _export_csv(self) -> None:
        if self._last_projected_nodes is None or self._last_projected_nodes.empty:
            show_warning("Run a flatmap projection before exporting CSV.")
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Flatmap Projection CSV",
            "flatmap_projection.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if output_path:
            self._export_current_projection_to_path(output_path)

    def _export_current_projection_to_path(self, output_path: str | Path) -> Path:
        """Export the current projected node table to a specific CSV path."""
        if self._last_projected_nodes is None or self._last_projected_nodes.empty:
            raise RuntimeError("Run a flatmap projection before exporting CSV.")
        saved = export_projected_nodes_csv(self._last_projected_nodes, output_path)
        self._status_label.setText(f"Exported flatmap projection to {saved}.")
        show_info(f"Exported flatmap projection to {saved}")
        return saved

    def _current_source_parquet_path(self) -> Path:
        db = self._database_provider()
        if db is None:
            raise RuntimeError("Load a neuron Parquet before saving augmented Parquet.")
        parquet_path = getattr(db, "parquet_path", None)
        if parquet_path is None:
            raise RuntimeError("Loaded neuron database does not expose a Parquet path.")
        return Path(parquet_path)

    def _augment_parquet(self) -> None:
        try:
            source_path = self._current_source_parquet_path()
            lookup_dir = getattr(self, "_preprocess_lookup_dir", None)
            if lookup_dir is None:
                raise RuntimeError(
                    "Choose a lookup directory containing bilateral shaped, "
                    "bilateral square, and depth NRRDs first."
                )
        except Exception as exc:
            show_warning(str(exc))
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Prepare Whole Flatmap Parquet",
            str(source_path.with_name(f"{source_path.stem}_flatmap.parquet")),
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if not output_path:
            return

        output = Path(output_path)
        if output.resolve() == source_path.resolve():
            from qtpy.QtWidgets import QMessageBox

            answer = QMessageBox.question(
                self,
                "Replace Source Parquet?",
                "This will atomically replace the loaded source Parquet after "
                "all rows are prepared. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self._start_parquet_preparation(source_path, output, lookup_dir)

    def _start_parquet_preparation(
        self,
        source_path: Path,
        output_path: Path,
        lookup_dir: Path,
    ) -> None:
        """Run whole-file bilateral preprocessing in a cancellable QThread."""
        from qtpy.QtCore import QThread

        from ..workers import FlatmapParquetPreparationWorker

        resolution_control = getattr(self, "_lookup_resolution_spin", None)
        raw_resolution = (
            int(resolution_control.value()) if resolution_control is not None else 0
        )
        lookup_resolution_um = float(raw_resolution) if raw_resolution > 0 else None
        thread = QThread()
        worker = FlatmapParquetPreparationWorker(
            source_path,
            output_path,
            lookup_dir,
            lookup_resolution_um=lookup_resolution_um,
        )
        self._augment_thread = thread
        self._augment_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._set_projection_progress)
        worker.finished.connect(self._on_parquet_preparation_finished)
        worker.error.connect(self._on_parquet_preparation_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda: self._cleanup_parquet_preparation(thread, worker)
        )
        self._augment_parquet_btn.setEnabled(False)
        self._cancel_augment_btn.setEnabled(True)
        self._set_projection_progress("Preparing whole Parquet...", 0, 0)
        thread.start()

    def _cancel_parquet_preparation(self) -> None:
        worker = getattr(self, "_augment_worker", None)
        cancel = getattr(worker, "cancel", None)
        if callable(cancel):
            cancel()
            self._status_label.setText("Cancelling Parquet preparation...")
            self._cancel_augment_btn.setEnabled(False)

    def _on_parquet_preparation_finished(self, summary) -> None:
        self._hide_projection_progress()
        message = (
            f"Prepared {getattr(summary, 'rows', 0):,} row(s) in "
            f"{getattr(summary, 'output_parquet', '')}."
        )
        self._status_label.setText(message)
        show_info(message)

    def _on_parquet_preparation_error(self, message: str) -> None:
        self._hide_projection_progress()
        self._status_label.setText(f"Flatmap Parquet preparation failed: {message}")
        show_warning(f"Flatmap Parquet preparation failed: {message}")

    def _cleanup_parquet_preparation(self, thread, worker) -> None:
        if getattr(self, "_augment_thread", None) is thread:
            self._augment_thread = None
        if getattr(self, "_augment_worker", None) is worker:
            self._augment_worker = None
        self._augment_parquet_btn.setEnabled(True)
        self._cancel_augment_btn.setEnabled(False)

    def _augment_current_parquet_to_path(self, output_path: str | Path):
        """Save a Parquet file augmented with NRRD-derived flatmap columns."""
        self._projection_request_ready()
        source_path = self._current_source_parquet_path()
        file_ids = self._file_ids_for_source()
        if not file_ids:
            raise RuntimeError("No neurons are available to save.")
        summary = augment_neuron_parquet_with_flatmap(
            source_path,
            output_path,
            self._flatmap_path,
            self._depth_path,
            file_ids=file_ids,
            flatmap_style=self._current_style_filename(),
            coordinate_mode=self._current_coordinate_mode(),
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=self._negative_one_sentinel_cb.isChecked(),
        )
        self._status_label.setText(
            "Saved augmented Parquet to "
            f"{summary.output_parquet} "
            f"({summary.rows:,} rows from {len(file_ids):,} file ID(s); "
            f"{summary.direct_rows:,} direct, "
            f"{getattr(summary, 'mirrored_depth_rows', 0):,} mirrored-depth, "
            f"{summary.mirrored_rows:,} mirrored, "
            f"{summary.unmapped_rows:,} unmapped)."
        )
        show_info(f"Saved augmented Parquet to {summary.output_parquet}")
        return summary
