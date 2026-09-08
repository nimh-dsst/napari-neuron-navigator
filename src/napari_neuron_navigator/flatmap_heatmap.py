"""Depth-aware flatmap render helpers."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from .isocortex_layers import AllenIsocortexLayerMap

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

#: Bin count along the flat map ``y`` axis, which spans ONE hemisphere.  The
#: ``x`` count is derived from the aspect ratio -- see
#: :func:`resolve_flatmap_bin_counts` -- so bins stay square.
DEFAULT_FLATMAP_Y_BINS = 256

#: Retained so a single control cannot exceed the region cache's portable
#: uint32 key format once ``x`` is derived: the widest style roughly doubles the
#: ``x`` count, so 2048 keeps the derived count within the old 4096 ceiling.
MAX_FLATMAP_Y_BINS = 2048

#: Shared wording for the ``Y bins`` control, so the flat map tab and the analysis
#: tab explain the derived ``x`` count identically.
FLATMAP_Y_BINS_TOOLTIP = (
    "Number of voxel bins across the flat map Y extent.\n\n"
    "The X count is derived from the flat map aspect ratio so that bins are "
    "square. X spans BOTH hemispheres while Y spans one, so equal counts "
    "would make every bin about twice as wide as it is tall.\n\n"
    "At 256 Y bins the bilateral square style resolves to 512 X bins and the "
    "bilateral shaped style to 491. The ratio is measured from the data's own "
    "bounds, so it is not exactly 2 for every style.\n\n"
    "Flat map X/Y are dimensionless: the projection distorts the cortical "
    "surface, so a bin width has no reliable conversion to microns."
)

DEFAULT_FLATMAP_DEPTH_BIN_UM = 25.0
MAX_FLATMAP_HEATMAP_VOXELS = 100_000_000
DEFAULT_LOOKUP_STATS_CHUNK_VOXELS = 10_000_000

# Upper bound on segments drawn by a 2D flatmap Vectors layer.  napari builds a
# mesh per vector, so a whole-file render (tens of millions of parent-child
# edges) stalls vispy.  Rendering only the first N would draw some neurons
# completely and omit others with no indication which, so callers refuse
# instead of truncating.
MAX_FLATMAP_VECTOR_SEGMENTS = 250_000

# Heatmap color modes. These values match the widget-side ``_HEATMAP_COLOR_*``
# constants so a mode selected in the UI can be passed straight through.
FLATMAP_HEATMAP_COLOR_SINGLE = "single"
FLATMAP_HEATMAP_COLOR_INDIVIDUAL = "individual"
FLATMAP_HEATMAP_COLOR_CLUSTER = "cluster"
FLATMAP_PLANE_MODE_DEPTH = "depth"
FLATMAP_PLANE_MODE_ALLEN_LAYERS = "allen_layers"
# Selected Allen layers collapsed into one categorical, depth-free XY plane.
# This stays distinct from ``FLATMAP_PLANE_MODE_FLAT`` because it is filtered by
# atlas layer membership rather than by the numeric-depth inclusion controls.
FLATMAP_PLANE_MODE_ALLEN_LAYER_PROJECTION = "allen_layer_projection"
# A flat render has no plane axis at all: flatmap XY only, depth collapsed.
FLATMAP_PLANE_MODE_FLAT = "flat"

ALLEN_LAYER_OUTPUT_STACK = "stack"
ALLEN_LAYER_OUTPUT_PROJECTION = "projection"

# Bilateral precomputed styles map onto the ``*_shaped``/``*_square`` column
# families that carry ready-to-render flatmap coordinates.
_FLATMAP_STYLE_SUFFIX = {
    "both_shaped": "shaped",
    "both_square": "square",
}


@dataclass(frozen=True)
class FlatmapLookupStats:
    """Reusable full-lookup statistics for flatmap heatmap rendering."""

    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    depth_range_um: tuple[float, float]
    flatmap_valid_voxels: int
    depth_valid_voxels: int
    flatmap_shape: tuple[int, int, int, int]
    depth_shape: tuple[int, int, int]
    flatmap_dtype: str
    depth_dtype: str
    invalid_zero_sentinel: bool
    invalid_negative_one_sentinel: bool

    def to_dict(self) -> dict[str, int | float | bool | str | list[int] | list[float]]:
        """Return a JSON-safe dictionary."""
        return {
            "x_bounds": [float(self.x_bounds[0]), float(self.x_bounds[1])],
            "y_bounds": [float(self.y_bounds[0]), float(self.y_bounds[1])],
            "depth_range_um": [
                float(self.depth_range_um[0]),
                float(self.depth_range_um[1]),
            ],
            "flatmap_valid_voxels": int(self.flatmap_valid_voxels),
            "depth_valid_voxels": int(self.depth_valid_voxels),
            "flatmap_shape": [int(size) for size in self.flatmap_shape],
            "depth_shape": [int(size) for size in self.depth_shape],
            "flatmap_dtype": self.flatmap_dtype,
            "depth_dtype": self.depth_dtype,
            "invalid_zero_sentinel": bool(self.invalid_zero_sentinel),
            "invalid_negative_one_sentinel": bool(self.invalid_negative_one_sentinel),
        }


@dataclass(frozen=True, kw_only=True)
class FlatmapRenderSummary:
    """Counts and ranges for a depth-aware flatmap render.

    Keyword-only so that adding or splitting a field cannot silently shift
    positional arguments at a construction site -- the per-axis bin-count split
    is exactly that kind of change.
    """

    total_nodes: int
    flatmap_valid_nodes: int
    depth_valid_nodes: int
    depth_minus_one_nodes: int
    rendered_nodes: int
    excluded_depth_minus_one_nodes: int
    nonzero_voxels: int
    traces_represented: int
    y_bins: int
    x_bins: int
    depth_bins: int
    depth_bin_um: float
    x_flat_min: float
    x_flat_max: float
    y_flat_min: float
    y_flat_max: float
    depth_min_um: float
    depth_max_um: float
    includes_depth_minus_one_plane: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a JSON-safe dictionary."""
        return {
            "total_nodes": int(self.total_nodes),
            "flatmap_valid_nodes": int(self.flatmap_valid_nodes),
            "depth_valid_nodes": int(self.depth_valid_nodes),
            "depth_minus_one_nodes": int(self.depth_minus_one_nodes),
            "rendered_nodes": int(self.rendered_nodes),
            "excluded_depth_minus_one_nodes": int(self.excluded_depth_minus_one_nodes),
            "nonzero_voxels": int(self.nonzero_voxels),
            "traces_represented": int(self.traces_represented),
            "y_bins": int(self.y_bins),
            "x_bins": int(self.x_bins),
            "depth_bins": int(self.depth_bins),
            "depth_bin_um": float(self.depth_bin_um),
            "x_flat_min": float(self.x_flat_min),
            "x_flat_max": float(self.x_flat_max),
            "y_flat_min": float(self.y_flat_min),
            "y_flat_max": float(self.y_flat_max),
            "depth_min_um": float(self.depth_min_um),
            "depth_max_um": float(self.depth_max_um),
            "includes_depth_minus_one_plane": bool(self.includes_depth_minus_one_plane),
        }


@dataclass(frozen=True)
class FlatmapRenderResult:
    """Binned flatmap render data and the projected table used to create it."""

    projected_nodes: pd.DataFrame
    volume: np.ndarray
    points: np.ndarray
    point_file_ids: list[object]
    summary: FlatmapRenderSummary


@dataclass(frozen=True)
class FlatmapGroupedVolume:
    """One scalar flatmap heatmap volume for a rendered neuron group."""

    group_key: object
    label: str
    source_file_ids: tuple[object, ...]
    volume: np.ndarray
    rendered_nodes: int
    nonzero_voxels: int


@dataclass(frozen=True)
class FlatmapSegmentVectors:
    """Parent-child flatmap segments in napari ``Vectors`` layout.

    ``data`` is ``(M, 2, 2)``: ``data[:, 0]`` holds each segment's start and
    ``data[:, 1]`` its direction, both in ``(row, col)`` image-pixel order --
    the ``[start, direction]`` form a napari ``Vectors`` layer expects.
    """

    data: np.ndarray
    file_ids: tuple[object, ...]
    total_segments: int


@dataclass(frozen=True, kw_only=True)
class AllenLayerRenderSummary:
    """Counts and grid provenance for one categorical Allen-layer render.

    Keyword-only for the same reason as :class:`FlatmapRenderSummary`.
    """

    total_nodes: int
    flatmap_valid_nodes: int
    layer_classified_nodes: int
    rendered_nodes: int
    excluded_non_layer_nodes: int
    nonzero_voxels: int
    traces_represented: int
    y_bins: int
    x_bins: int
    x_flat_min: float
    x_flat_max: float
    y_flat_min: float
    y_flat_max: float
    layer_labels: tuple[str, ...]
    layer_node_counts: tuple[int, ...]
    atlas_name: str
    atlas_version: str = ""
    selected_layer_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    output_mode: str = ALLEN_LAYER_OUTPUT_STACK
    excluded_unselected_layer_nodes: int = 0
    output_shape: tuple[int, ...] = ()

    @property
    def plane_count(self) -> int:
        """Return the number of categorical image planes."""
        if self.output_mode == ALLEN_LAYER_OUTPUT_PROJECTION:
            return 0
        return len(self.layer_labels)

    @property
    def excluded_nodes(self) -> int:
        """Return all input nodes omitted from the rendered output."""
        return max(0, int(self.total_nodes) - int(self.rendered_nodes))

    @property
    def invalid_flatmap_nodes(self) -> int:
        """Return input nodes omitted because flatmap XY is invalid."""
        return max(0, int(self.total_nodes) - int(self.flatmap_valid_nodes))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dictionary."""
        return {
            "plane_mode": (
                FLATMAP_PLANE_MODE_ALLEN_LAYER_PROJECTION
                if self.output_mode == ALLEN_LAYER_OUTPUT_PROJECTION
                else FLATMAP_PLANE_MODE_ALLEN_LAYERS
            ),
            "output_mode": self.output_mode,
            "allen_layer_output_mode": self.output_mode,
            "total_nodes": int(self.total_nodes),
            "flatmap_valid_nodes": int(self.flatmap_valid_nodes),
            "layer_classified_nodes": int(self.layer_classified_nodes),
            "rendered_nodes": int(self.rendered_nodes),
            "excluded_nodes": self.excluded_nodes,
            "invalid_flatmap_nodes": self.invalid_flatmap_nodes,
            "excluded_non_layer_nodes": int(self.excluded_non_layer_nodes),
            "excluded_unselected_layer_nodes": int(
                self.excluded_unselected_layer_nodes
            ),
            "nonzero_voxels": int(self.nonzero_voxels),
            "traces_represented": int(self.traces_represented),
            "y_bins": int(self.y_bins),
            "x_bins": int(self.x_bins),
            "x_flat_min": float(self.x_flat_min),
            "x_flat_max": float(self.x_flat_max),
            "y_flat_min": float(self.y_flat_min),
            "y_flat_max": float(self.y_flat_max),
            "layer_labels": list(self.layer_labels),
            "selected_layer_labels": list(self.layer_labels),
            "selected_layer_indices": [
                int(value) for value in self.selected_layer_indices
            ],
            "layer_node_counts": [int(value) for value in self.layer_node_counts],
            "output_shape": [int(value) for value in self.output_shape],
            "atlas_name": self.atlas_name,
            "atlas_version": self.atlas_version,
        }


@dataclass(frozen=True)
class AllenLayerRenderResult:
    """Materialized node table and node-count volume for Allen layers."""

    projected_nodes: pd.DataFrame
    volume: np.ndarray
    summary: AllenLayerRenderSummary


# Compatibility names for code that imports the original all-six stack types.
AllenLayerStackSummary = AllenLayerRenderSummary
AllenLayerStackResult = AllenLayerRenderResult


@dataclass(frozen=True)
class AllenLayerAggregateStats:
    """Tallies gathered by the DuckDB categorical-layer path."""

    total_nodes: int
    total_traces: int
    flatmap_valid_nodes: int
    layer_classified_nodes: int
    rendered_nodes: int
    traces_represented: int
    excluded_unselected_layer_nodes: int = 0


@dataclass(frozen=True)
class AllenLayerHeatmapVolumeResult:
    """Categorical Allen-layer heatmap volume(s) built in DuckDB."""

    color_mode: str
    volume: np.ndarray | None
    grouped_volumes: tuple[FlatmapGroupedVolume, ...]
    summary: AllenLayerRenderSummary
    stats: AllenLayerAggregateStats
    volume_shape: tuple[int, ...]


def _validate_resolution(
    y_bins: int,
    x_bins: int,
    depth_bin_um: float,
) -> tuple[int, int, float]:
    """Coerce and check a per-axis grid resolution.

    The single coercion funnel for every flat map grid builder, so the two bin
    counts are validated in one place rather than at each entry point.
    """
    y_bins = int(y_bins)
    x_bins = int(x_bins)
    depth_bin_um = float(depth_bin_um)
    if y_bins <= 0:
        raise ValueError("y_bins must be positive.")
    if x_bins <= 0:
        raise ValueError("x_bins must be positive.")
    if depth_bin_um <= 0.0:
        raise ValueError("depth_bin_um must be positive.")
    return y_bins, x_bins, depth_bin_um


def _nondegenerate_bounds(lower: float, upper: float) -> tuple[float, float]:
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Flatmap bounds must be finite.")
    if upper > lower:
        return float(lower), float(upper)
    pad = max(abs(float(lower)) * 0.01, 0.5)
    return float(lower - pad), float(upper + pad)


class FlatmapBinCounts(NamedTuple):
    """Per-axis flat map bin counts.

    A named tuple rather than a bare pair because this module mixes ``(row,
    col)`` and ``(x, y)`` ordering throughout, so a positional pair of bin
    counts is a silent swap waiting to happen.
    """

    y_bins: int
    x_bins: int


def resolve_flatmap_bin_counts(
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    y_bins: int,
) -> FlatmapBinCounts:
    """Return per-axis bin counts that make flat map bins square.

    The bilateral flat map lays the two hemispheres side by side along ``x``,
    so ``x`` spans roughly twice the extent of ``y``.  Giving both axes the same
    bin count therefore makes every bin about twice as wide as it is tall, which
    discards ``x`` detail at twice the rate of ``y`` and renders the map
    horizontally squashed.  Scaling the ``x`` count by the aspect ratio fixes
    both at once.

    ``y_bins`` is the control: it counts bins across the ``y`` extent, which is
    one hemisphere tall.  Do **not** think of it as "bins per hemisphere" and do
    **not** derive ``x`` as ``2 * y_bins`` -- that holds only for
    ``both_square``.  ``both_shaped`` has an aspect ratio of 1.9162, so doubling
    would leave 4.2% anisotropy.

    Rounding is pinned to half-up rather than :func:`round`, whose banker's
    rounding is non-monotone at ties (``round(490.5)`` is 490 but
    ``round(491.5)`` is 492).  This count is recorded in the region-cache
    identity digest, so its tie behavior must be stable and explicit.
    """
    count = int(y_bins)
    if count <= 0:
        raise ValueError(f"y_bins must be positive; got {y_bins!r}.")

    x_lower, x_upper = _nondegenerate_bounds(x_bounds[0], x_bounds[1])
    y_lower, y_upper = _nondegenerate_bounds(y_bounds[0], y_bounds[1])
    aspect = (x_upper - x_lower) / (y_upper - y_lower)
    if not np.isfinite(aspect) or aspect <= 0.0:
        raise ValueError(
            f"Flat map bounds give a non-positive aspect ratio; got {aspect!r}."
        )
    return FlatmapBinCounts(
        y_bins=count,
        x_bins=max(1, math.floor(count * aspect + 0.5)),
    )


def _check_bin_count_inputs(
    y_bins: int,
    x_bins: int | None,
    depth_bin_um: float,
) -> None:
    """Reject an unusable resolution before any expensive scan runs.

    Deriving ``x`` needs bounds, which for the NRRD paths means scanning the
    whole lookup volume first.  This positivity check costs nothing and keeps a
    bad argument from surfacing only after that scan.  It deliberately does not
    derive anything -- ``_resolve_axis_bin_counts`` remains the one place that
    does.
    """
    _validate_resolution(y_bins, y_bins if x_bins is None else x_bins, depth_bin_um)


def _resolve_axis_bin_counts(
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    y_bins: int,
    x_bins: int | None,
    depth_bin_um: float = 1.0,
) -> tuple[int, int, float]:
    """Resolve per-axis bin counts, deriving ``x`` only when it is not supplied.

    The single choke point every flat map grid builder goes through, so the
    square-bin policy cannot be applied in one subsystem and skipped in another.

    A supplied ``x_bins`` is used **verbatim**.  A cache-backed render has to
    reproduce its profile's stored count exactly -- the widget discards a render
    whose grid disagrees, and the worker raises on a mask/volume shape mismatch
    -- and re-deriving from bounds that have been through a JSON float round trip
    is not guaranteed to return the same integer at a rounding tie.  Never
    re-derive what was stored.
    """
    if x_bins is None:
        derived = resolve_flatmap_bin_counts(
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            y_bins=y_bins,
        )
        y_bins, x_bins = derived.y_bins, derived.x_bins
    return _validate_resolution(y_bins, x_bins, depth_bin_um)


def _flatmap_valid_mask(
    flat_xy: np.ndarray,
    *,
    invalid_zero_sentinel: bool,
    invalid_negative_one_sentinel: bool,
) -> np.ndarray:
    finite = np.all(np.isfinite(flat_xy), axis=1)
    valid = finite.copy()
    if invalid_negative_one_sentinel:
        valid &= ~((flat_xy[:, 0] == -1.0) & (flat_xy[:, 1] == -1.0))
    if invalid_zero_sentinel:
        valid &= ~((flat_xy[:, 0] == 0.0) & (flat_xy[:, 1] == 0.0))
    return valid


def _spatial_chunk_slices(
    shape: tuple[int, int, int],
    *,
    chunk_voxels: int = DEFAULT_LOOKUP_STATS_CHUNK_VOXELS,
) -> list[slice]:
    chunk_voxels = max(1, int(chunk_voxels))
    plane_voxels = max(1, int(shape[1]) * int(shape[2]))
    chunk_size = max(1, chunk_voxels // plane_voxels)
    return [
        slice(start, min(start + chunk_size, shape[0]))
        for start in range(0, shape[0], chunk_size)
    ]


def _validate_flatmap_volume(flatmap_volume: np.ndarray) -> np.ndarray:
    flatmap = np.asarray(flatmap_volume)
    if flatmap.ndim != 4 or flatmap.shape[-1] != 2:
        raise ValueError(
            f"flatmap_volume must have shape (nx, ny, nz, 2); got {flatmap.shape}."
        )
    return flatmap


def _validate_depth_volume(depth_volume: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_volume)
    if depth.ndim != 3:
        raise ValueError(f"depth_volume must be 3D; got {depth.shape}.")
    return depth


def _compute_flatmap_xy_bounds_and_count(
    flatmap_volume: np.ndarray,
    *,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
    chunk_voxels: int = DEFAULT_LOOKUP_STATS_CHUNK_VOXELS,
    cancel_callback: Callable[[], bool] | None = None,
) -> tuple[tuple[float, float], tuple[float, float], int]:
    flatmap = _validate_flatmap_volume(flatmap_volume)
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    valid_count = 0

    for chunk_slice in _spatial_chunk_slices(
        flatmap.shape[:3],
        chunk_voxels=chunk_voxels,
    ):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("Flatmap lookup statistics cancelled.")
        chunk = flatmap[chunk_slice]
        x_values = chunk[..., 0]
        y_values = chunk[..., 1]
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        if invalid_negative_one_sentinel:
            valid &= ~((x_values == -1.0) & (y_values == -1.0))
        if invalid_zero_sentinel:
            valid &= ~((x_values == 0.0) & (y_values == 0.0))
        if not valid.any():
            continue

        valid_count += int(valid.sum())
        x_valid = x_values[valid]
        y_valid = y_values[valid]
        x_min = min(x_min, float(np.min(x_valid)))
        x_max = max(x_max, float(np.max(x_valid)))
        y_min = min(y_min, float(np.min(y_valid)))
        y_max = max(y_max, float(np.max(y_valid)))

    if valid_count <= 0:
        raise ValueError("Flatmap volume does not contain valid x/y lookup values.")
    return (
        _nondegenerate_bounds(x_min, x_max),
        _nondegenerate_bounds(y_min, y_max),
        valid_count,
    )


def compute_flatmap_xy_bounds(
    flatmap_volume: np.ndarray,
    *,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
    chunk_voxels: int = DEFAULT_LOOKUP_STATS_CHUNK_VOXELS,
    cancel_callback: Callable[[], bool] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return finite x/y flatmap coordinate bounds from a lookup volume."""
    x_bounds, y_bounds, _valid_count = _compute_flatmap_xy_bounds_and_count(
        flatmap_volume,
        invalid_zero_sentinel=invalid_zero_sentinel,
        invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        chunk_voxels=chunk_voxels,
        cancel_callback=cancel_callback,
    )
    return x_bounds, y_bounds


def _compute_depth_range_and_count(
    depth_volume: np.ndarray,
    *,
    chunk_voxels: int = DEFAULT_LOOKUP_STATS_CHUNK_VOXELS,
    cancel_callback: Callable[[], bool] | None = None,
) -> tuple[tuple[float, float], int]:
    depth = _validate_depth_volume(depth_volume)
    lower = np.inf
    upper = -np.inf
    valid_count = 0

    for chunk_slice in _spatial_chunk_slices(
        depth.shape,
        chunk_voxels=chunk_voxels,
    ):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("Flatmap lookup statistics cancelled.")
        chunk = depth[chunk_slice]
        valid = np.isfinite(chunk) & (chunk >= 0.0)
        if not valid.any():
            continue
        valid_count += int(valid.sum())
        values = chunk[valid]
        lower = min(lower, float(np.min(values)))
        upper = max(upper, float(np.max(values)))

    if valid_count <= 0:
        raise ValueError("Depth volume does not contain valid non-negative depths.")
    if upper <= lower:
        upper = lower + 1.0
    return (float(lower), float(upper)), valid_count


def compute_depth_range(depth_volume: np.ndarray) -> tuple[float, float]:
    """Return the finite non-negative depth range from a depth lookup volume."""
    depth_range, _valid_count = _compute_depth_range_and_count(depth_volume)
    return depth_range


def compute_flatmap_lookup_stats(
    flatmap_volume: np.ndarray,
    depth_volume: np.ndarray,
    *,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
    chunk_voxels: int = DEFAULT_LOOKUP_STATS_CHUNK_VOXELS,
    cancel_callback: Callable[[], bool] | None = None,
) -> FlatmapLookupStats:
    """Return reusable lookup bounds and depth range using bounded chunks."""
    flatmap = _validate_flatmap_volume(flatmap_volume)
    depth = _validate_depth_volume(depth_volume)
    if depth.shape != flatmap.shape[:3]:
        raise ValueError(
            "depth_volume shape must match the flatmap lookup grid; "
            f"got depth {depth.shape} and flatmap grid {flatmap.shape[:3]}."
        )

    x_bounds, y_bounds, flatmap_valid_voxels = _compute_flatmap_xy_bounds_and_count(
        flatmap,
        invalid_zero_sentinel=invalid_zero_sentinel,
        invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        chunk_voxels=chunk_voxels,
        cancel_callback=cancel_callback,
    )
    depth_range, depth_valid_voxels = _compute_depth_range_and_count(
        depth,
        chunk_voxels=chunk_voxels,
        cancel_callback=cancel_callback,
    )
    return FlatmapLookupStats(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        depth_range_um=depth_range,
        flatmap_valid_voxels=flatmap_valid_voxels,
        depth_valid_voxels=depth_valid_voxels,
        flatmap_shape=tuple(int(size) for size in flatmap.shape),
        depth_shape=tuple(int(size) for size in depth.shape),
        flatmap_dtype=str(flatmap.dtype),
        depth_dtype=str(depth.dtype),
        invalid_zero_sentinel=bool(invalid_zero_sentinel),
        invalid_negative_one_sentinel=bool(invalid_negative_one_sentinel),
    )


def _bin_flat_values(
    values: np.ndarray,
    bounds: tuple[float, float],
    bins: int,
) -> np.ndarray:
    lower, upper = _nondegenerate_bounds(bounds[0], bounds[1])
    scaled = (np.asarray(values, dtype=float) - lower) / (upper - lower)
    out = np.floor(scaled * bins).astype(np.int64)
    return np.clip(out, 0, bins - 1)


def flatmap_pixel_coordinates(
    values: np.ndarray,
    bounds: tuple[float, float],
    bins: int,
) -> np.ndarray:
    """Return continuous image-pixel coordinates for raw flatmap values.

    :func:`_bin_flat_values` floors a scaled value to an integer bin index.
    napari draws image pixel ``k`` centered on coordinate ``k``, spanning
    ``[k - 0.5, k + 0.5]``, so a value at the center of bin ``k`` scales to
    ``k + 0.5`` and its pixel-aligned coordinate is ``k``.  Hence the 0.5
    subtraction: without it every overlay built from these coordinates sits half
    a pixel off the heatmap it is drawn over.

    Results are clipped to ``[-0.5, bins - 0.5]``, the image's outer edges, so a
    value outside ``bounds`` cannot drag the camera away from the render.
    """
    lower, upper = _nondegenerate_bounds(bounds[0], bounds[1])
    scaled = (np.asarray(values, dtype=float) - lower) / (upper - lower)
    out = scaled * int(bins) - 0.5
    return np.clip(out, -0.5, int(bins) - 0.5)


def build_flatmap_segment_vectors(
    segment_endpoints: np.ndarray,
    file_ids: Sequence[object],
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    y_bins: int,
    x_bins: int,
    max_segments: int = MAX_FLATMAP_VECTOR_SEGMENTS,
) -> FlatmapSegmentVectors:
    """Convert projected parent-child segments into napari vector data.

    ``segment_endpoints`` is the ``(M, 2, 2)`` array
    :func:`napari_neuron_navigator.flatmap_projection.build_projected_segments`
    returns: ``[:, 0]`` is the parent endpoint, ``[:, 1]`` the child endpoint,
    and the last axis is ``(x_flat, y_flat)`` in raw flatmap units.  Output
    coordinates are image pixels in ``(row, col)`` order so the vectors land on
    the same grid as a flatmap heatmap built with the same bounds and bin counts.

    The two counts must be the *same pair* the heatmap was built with.  Scaling
    both axes by a single count compresses every vector along ``x`` relative to
    the heatmap it is drawn over -- a whole-dataset misregistration, not a
    half-pixel offset -- because ``x`` spans two hemispheres and ``y`` one.

    Raises ``ValueError`` above ``max_segments`` rather than truncating, because
    a truncated render silently omits whole neurons.
    """
    endpoints = np.asarray(segment_endpoints, dtype=float)
    if endpoints.ndim != 3 or endpoints.shape[1:] != (2, 2):
        raise ValueError(
            f"segment_endpoints must have shape (M, 2, 2); got {endpoints.shape}."
        )
    total_segments = int(endpoints.shape[0])
    resolved_file_ids = tuple(file_ids)
    if len(resolved_file_ids) != total_segments:
        raise ValueError(
            "file_ids must name one neuron per segment; got "
            f"{len(resolved_file_ids)} for {total_segments} segment(s)."
        )
    if total_segments > int(max_segments):
        raise ValueError(
            f"2D Vector mode would draw {total_segments:,} flatmap segments, "
            f"above the {int(max_segments):,} limit napari can render "
            "interactively. Select fewer neurons, or use 2D Heatmap to see the "
            "whole set."
        )

    if total_segments == 0:
        return FlatmapSegmentVectors(
            data=np.empty((0, 2, 2), dtype=np.float32),
            file_ids=(),
            total_segments=0,
        )

    columns = flatmap_pixel_coordinates(endpoints[:, :, 0], x_bounds, x_bins)
    rows = flatmap_pixel_coordinates(endpoints[:, :, 1], y_bounds, y_bins)
    # napari image axes are (row, col), so y leads x.
    start = np.column_stack((rows[:, 0], columns[:, 0]))
    end = np.column_stack((rows[:, 1], columns[:, 1]))
    data = np.stack([start, end - start], axis=1).astype(np.float32)
    return FlatmapSegmentVectors(
        data=data,
        file_ids=resolved_file_ids,
        total_segments=total_segments,
    )


def _depth_bin_count(
    depth_range_um: tuple[float, float],
    depth_bin_um: float,
) -> int:
    lower, upper = depth_range_um
    return max(1, int(np.floor((upper - lower) / depth_bin_um)) + 1)


def _projected_nodes_with_validity_flags(projected_nodes: pd.DataFrame) -> pd.DataFrame:
    """Return projected nodes with inferred validity flags when absent."""
    required = ("x_flat", "y_flat", "depth_um")
    missing = [column for column in required if column not in projected_nodes.columns]
    if missing:
        raise ValueError(f"Projected nodes are missing required column(s): {missing}")

    table = projected_nodes.copy()
    x_values = pd.to_numeric(table["x_flat"], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(table["y_flat"], errors="coerce").to_numpy(dtype=float)
    depth_values = pd.to_numeric(table["depth_um"], errors="coerce").to_numpy(
        dtype=float
    )
    if "flatmap_valid" not in table.columns:
        table.loc[:, "flatmap_valid"] = np.isfinite(x_values) & np.isfinite(y_values)
    if "depth_valid" not in table.columns:
        table.loc[:, "depth_valid"] = np.isfinite(depth_values) & (depth_values >= 0.0)
    return table


def _projected_nodes_bounds(
    projected_nodes: pd.DataFrame,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    table = _projected_nodes_with_validity_flags(projected_nodes)
    flatmap_valid = table["flatmap_valid"].fillna(False).astype(bool).to_numpy()
    depth_valid = table["depth_valid"].fillna(False).astype(bool).to_numpy()
    x_values = pd.to_numeric(table["x_flat"], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(table["y_flat"], errors="coerce").to_numpy(dtype=float)
    depth_values = pd.to_numeric(table["depth_um"], errors="coerce").to_numpy(
        dtype=float
    )

    flatmap_mask = flatmap_valid & np.isfinite(x_values) & np.isfinite(y_values)
    if not flatmap_mask.any():
        raise ValueError("Projected nodes do not contain valid flatmap coordinates.")
    depth_mask = (
        flatmap_valid & depth_valid & np.isfinite(depth_values) & (depth_values >= 0.0)
    )
    if not depth_mask.any():
        raise ValueError("Projected nodes do not contain valid non-negative depths.")

    return (
        _nondegenerate_bounds(
            float(np.min(x_values[flatmap_mask])),
            float(np.max(x_values[flatmap_mask])),
        ),
        _nondegenerate_bounds(
            float(np.min(y_values[flatmap_mask])),
            float(np.max(y_values[flatmap_mask])),
        ),
        _nondegenerate_bounds(
            float(np.min(depth_values[depth_mask])),
            float(np.max(depth_values[depth_mask])),
        ),
    )


def _validate_lookup_stats(
    lookup_stats: FlatmapLookupStats,
    flatmap: np.ndarray,
    depth: np.ndarray,
    *,
    invalid_zero_sentinel: bool,
    invalid_negative_one_sentinel: bool,
) -> None:
    if tuple(lookup_stats.flatmap_shape) != tuple(flatmap.shape):
        raise ValueError(
            "lookup_stats flatmap shape does not match flatmap_volume; "
            f"got stats {lookup_stats.flatmap_shape} and volume {flatmap.shape}."
        )
    if tuple(lookup_stats.depth_shape) != tuple(depth.shape):
        raise ValueError(
            "lookup_stats depth shape does not match depth_volume; "
            f"got stats {lookup_stats.depth_shape} and volume {depth.shape}."
        )
    if lookup_stats.invalid_zero_sentinel != bool(invalid_zero_sentinel):
        raise ValueError(
            "lookup_stats were computed with different zero-sentinel settings."
        )
    if lookup_stats.invalid_negative_one_sentinel != bool(
        invalid_negative_one_sentinel
    ):
        raise ValueError(
            "lookup_stats were computed with different negative-one sentinel settings."
        )


def _depth_labels(
    depth_bins: np.ndarray,
    *,
    depth_min_um: float,
    depth_bin_um: float,
    sentinel_offset: int,
) -> list[str]:
    labels: list[str] = []
    for bin_index in depth_bins:
        if bin_index < 0:
            labels.append("")
            continue
        if sentinel_offset and bin_index == 0:
            labels.append("depth -1")
            continue
        valid_bin = int(bin_index) - sentinel_offset
        lower = depth_min_um + valid_bin * depth_bin_um
        upper = lower + depth_bin_um
        labels.append(f"{lower:g}-{upper:g} um")
    return labels


def depth_plane_labels(render_summary: Mapping[str, object]) -> tuple[str, ...]:
    """Return one human-readable label per depth plane of a flatmap render.

    ``render_summary`` is a :meth:`FlatmapRenderSummary.to_dict` payload, which
    is what render layers carry in ``layer.metadata["render_summary"]``. Returns
    an empty tuple when the payload does not describe a depth-binned render.
    """
    try:
        depth_bins = int(render_summary.get("depth_bins", 0))
        depth_bin_um = float(render_summary.get("depth_bin_um", 0.0))
        depth_min_um = float(render_summary.get("depth_min_um", 0.0))
    except (AttributeError, TypeError, ValueError):
        return ()
    if depth_bins <= 0 or depth_bin_um <= 0.0:
        return ()
    sentinel_offset = (
        1 if bool(render_summary.get("includes_depth_minus_one_plane", False)) else 0
    )
    return tuple(
        _depth_labels(
            np.arange(depth_bins, dtype=np.int64),
            depth_min_um=depth_min_um,
            depth_bin_um=depth_bin_um,
            sentinel_offset=sentinel_offset,
        )
    )


def build_flatmap_render_data(
    projected_nodes: pd.DataFrame,
    flatmap_volume: np.ndarray,
    depth_volume: np.ndarray,
    *,
    y_bins: int = DEFAULT_FLATMAP_Y_BINS,
    x_bins: int | None = None,
    depth_bin_um: float = DEFAULT_FLATMAP_DEPTH_BIN_UM,
    include_depth_minus_one: bool = True,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
    lookup_stats: FlatmapLookupStats | None = None,
    lookup_stats_chunk_voxels: int = DEFAULT_LOOKUP_STATS_CHUNK_VOXELS,
    collapse_depth: bool = False,
) -> FlatmapRenderResult:
    """Build a depth-aware flatmap node-count volume and point coordinates.

    ``y_bins`` counts bins across the flat map ``y`` extent; ``x_bins`` defaults
    to the count that keeps bins square (see :func:`resolve_flatmap_bin_counts`).
    Pass ``x_bins`` only to reproduce a grid that was recorded elsewhere.

    ``collapse_depth`` drops the depth *axis* while keeping depth's say over
    which nodes render; see :func:`_build_flatmap_render_data_for_bounds`.
    """
    _check_bin_count_inputs(y_bins, x_bins, depth_bin_um)
    flatmap = _validate_flatmap_volume(flatmap_volume)
    depth = _validate_depth_volume(depth_volume)
    if depth.shape != flatmap.shape[:3]:
        raise ValueError(
            "depth_volume shape must match the flatmap lookup grid; "
            f"got depth {depth.shape} and flatmap grid {flatmap.shape[:3]}."
        )
    if lookup_stats is None:
        lookup_stats = compute_flatmap_lookup_stats(
            flatmap,
            depth,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
            chunk_voxels=lookup_stats_chunk_voxels,
        )
    else:
        _validate_lookup_stats(
            lookup_stats,
            flatmap,
            depth,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        )

    # x_bins is derived inside _build_flatmap_render_data_for_bounds, the choke
    # point where the bounds the aspect ratio needs are already resolved.
    return _build_flatmap_render_data_for_bounds(
        projected_nodes,
        x_bounds=lookup_stats.x_bounds,
        y_bounds=lookup_stats.y_bounds,
        depth_range=lookup_stats.depth_range_um,
        y_bins=y_bins,
        x_bins=x_bins,
        depth_bin_um=depth_bin_um,
        include_depth_minus_one=include_depth_minus_one,
        collapse_depth=collapse_depth,
    )


def build_flatmap_render_data_from_projected_nodes(
    projected_nodes: pd.DataFrame,
    *,
    y_bins: int = DEFAULT_FLATMAP_Y_BINS,
    x_bins: int | None = None,
    depth_bin_um: float = DEFAULT_FLATMAP_DEPTH_BIN_UM,
    include_depth_minus_one: bool = True,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    depth_range_um: tuple[float, float] | None = None,
    collapse_depth: bool = False,
) -> FlatmapRenderResult:
    """Build a depth-aware render using coordinates already stored in a table.

    ``x_bounds``, ``y_bounds``, and ``depth_range_um`` should be the canonical
    bounds recorded in a version-3 Parquet or selected region-cache profile.
    Supplying them keeps queries that contain only part of the flatmap aligned
    to the same output grid.  The subset-derived fallback is retained only for
    legacy version-1/2 Parquets, which did not record canonical bounds.
    """
    projected = _projected_nodes_with_validity_flags(projected_nodes)
    supplied = (x_bounds, y_bounds, depth_range_um)
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        raise ValueError(
            "x_bounds, y_bounds, and depth_range_um must be provided together."
        )
    if all(value is not None for value in supplied):
        resolved_x_bounds = _nondegenerate_bounds(*x_bounds)  # type: ignore[arg-type]
        resolved_y_bounds = _nondegenerate_bounds(*y_bounds)  # type: ignore[arg-type]
        resolved_depth_range = _nondegenerate_bounds(
            *depth_range_um  # type: ignore[arg-type]
        )
    else:
        (
            resolved_x_bounds,
            resolved_y_bounds,
            resolved_depth_range,
        ) = _projected_nodes_bounds(projected)
    return _build_flatmap_render_data_for_bounds(
        projected,
        x_bounds=resolved_x_bounds,
        y_bounds=resolved_y_bounds,
        depth_range=resolved_depth_range,
        y_bins=y_bins,
        x_bins=x_bins,
        depth_bin_um=depth_bin_um,
        include_depth_minus_one=include_depth_minus_one,
        collapse_depth=collapse_depth,
    )


def _resolve_allen_layer_render_options(
    layer_map: AllenIsocortexLayerMap,
    selected_layer_indices: Sequence[int] | None,
    output_mode: str,
) -> tuple[tuple[int, ...], tuple[str, ...], str]:
    """Validate and canonicalize one Allen-layer render selection."""
    layer_labels = tuple(str(label) for label in layer_map.layer_labels)
    if not layer_labels:
        raise ValueError("Allen layer labels cannot be empty.")

    if selected_layer_indices is None:
        selected = tuple(range(len(layer_labels)))
    else:
        try:
            selected = tuple(sorted({int(value) for value in selected_layer_indices}))
        except (TypeError, ValueError) as exc:
            raise ValueError("Allen layer indices must be integers.") from exc
    if not selected:
        raise ValueError("Select at least one Allen layer.")
    invalid = [index for index in selected if index < 0 or index >= len(layer_labels)]
    if invalid:
        raise ValueError(
            "Allen layer indices must be between 0 and "
            f"{len(layer_labels) - 1}; got {invalid}."
        )

    mode = str(output_mode)
    if mode not in {ALLEN_LAYER_OUTPUT_STACK, ALLEN_LAYER_OUTPUT_PROJECTION}:
        raise ValueError(
            "Allen layer output_mode must be 'stack' or 'projection'; "
            f"got {output_mode!r}."
        )
    return selected, tuple(layer_labels[index] for index in selected), mode


def build_allen_layer_render_from_projected_nodes(
    projected_nodes: pd.DataFrame,
    layer_map: AllenIsocortexLayerMap,
    *,
    y_bins: int = DEFAULT_FLATMAP_Y_BINS,
    x_bins: int | None = None,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    selected_layer_indices: Sequence[int] | None = None,
    output_mode: str = ALLEN_LAYER_OUTPUT_STACK,
) -> AllenLayerRenderResult:
    """Bin selected atlas-defined Isocortex layers into a stack or projection."""
    required = ("x_flat", "y_flat", "region_id")
    missing = [column for column in required if column not in projected_nodes.columns]
    if missing:
        raise ValueError(
            f"Projected nodes are missing Allen-layer column(s): {missing}."
        )

    selected_indices, selected_labels, output_mode = (
        _resolve_allen_layer_render_options(
            layer_map,
            selected_layer_indices,
            output_mode,
        )
    )

    # Bounds first: the derived x count needs them, and the size guard needs the
    # derived count.
    resolved_x_bounds = _nondegenerate_bounds(*x_bounds)
    resolved_y_bounds = _nondegenerate_bounds(*y_bounds)
    y_bins, x_bins, _unused_depth_bin = _resolve_axis_bin_counts(
        x_bounds=resolved_x_bounds,
        y_bounds=resolved_y_bounds,
        y_bins=y_bins,
        x_bins=x_bins,
    )

    canonical_plane_count = len(layer_map.layer_labels)
    output_plane_count = (
        len(selected_indices) if output_mode == ALLEN_LAYER_OUTPUT_STACK else 1
    )
    voxel_count = int(output_plane_count * y_bins * x_bins)
    if voxel_count > MAX_FLATMAP_HEATMAP_VOXELS:
        raise ValueError(
            "Allen layer heatmap is too large: "
            f"{output_plane_count}x{y_bins}x{x_bins} voxels. Use fewer Y bins."
        )

    table = projected_nodes.copy()
    x_values = pd.to_numeric(table["x_flat"], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(table["y_flat"], errors="coerce").to_numpy(dtype=float)
    finite_xy = np.isfinite(x_values) & np.isfinite(y_values)
    if "flatmap_valid" in table.columns:
        flatmap_valid = (
            table["flatmap_valid"].fillna(False).astype(bool).to_numpy() & finite_xy
        )
    else:
        flatmap_valid = finite_xy

    numeric_region_ids = pd.to_numeric(table["region_id"], errors="coerce")
    mapped_layers = numeric_region_ids.map(layer_map.region_to_layer_index)
    layer_classified = mapped_layers.notna().to_numpy()
    layer_indices = np.full(len(table), -1, dtype=np.int64)
    if layer_classified.any():
        layer_indices[layer_classified] = (
            mapped_layers[layer_classified].astype(np.int64).to_numpy()
        )
    layer_selected = layer_classified & np.isin(layer_indices, selected_indices)
    render_valid = flatmap_valid & layer_selected
    render_plane_indices = np.full(len(table), -1, dtype=np.int64)
    for render_index, canonical_index in enumerate(selected_indices):
        render_plane_indices[layer_indices == canonical_index] = render_index

    x_bin_indices = np.full(len(table), -1, dtype=np.int64)
    y_bin_indices = np.full(len(table), -1, dtype=np.int64)
    if render_valid.any():
        x_bin_indices[render_valid] = _bin_flat_values(
            x_values[render_valid],
            resolved_x_bounds,
            x_bins,
        )
        y_bin_indices[render_valid] = _bin_flat_values(
            y_values[render_valid],
            resolved_y_bounds,
            y_bins,
        )

    volume_shape = (
        (len(selected_indices), y_bins, x_bins)
        if output_mode == ALLEN_LAYER_OUTPUT_STACK
        else (y_bins, x_bins)
    )
    volume = np.zeros(volume_shape, dtype=np.float32)
    render_indices = np.flatnonzero(render_valid)
    if render_indices.size:
        if output_mode == ALLEN_LAYER_OUTPUT_STACK:
            np.add.at(
                volume,
                (
                    render_plane_indices[render_indices],
                    y_bin_indices[render_indices],
                    x_bin_indices[render_indices],
                ),
                1.0,
            )
        else:
            np.add.at(
                volume,
                (
                    y_bin_indices[render_indices],
                    x_bin_indices[render_indices],
                ),
                1.0,
            )

    layer_labels = np.full(len(table), "", dtype=object)
    for index, label in enumerate(layer_map.layer_labels):
        layer_labels[layer_indices == index] = label
    table.loc[:, "render_valid"] = render_valid
    table.loc[:, "x_flat_bin"] = x_bin_indices
    table.loc[:, "y_flat_bin"] = y_bin_indices
    table.loc[:, "allen_layer_index"] = layer_indices
    table.loc[:, "allen_layer_label"] = layer_labels
    table.loc[:, "allen_layer_selected"] = layer_selected
    table.loc[:, "allen_layer_render_index"] = render_plane_indices

    if render_indices.size and "file_id" in table.columns:
        traces_represented = int(table.iloc[render_indices]["file_id"].nunique())
    else:
        traces_represented = 0
    canonical_counts = np.bincount(
        layer_indices[render_valid],
        minlength=canonical_plane_count,
    )
    layer_node_counts = tuple(
        int(canonical_counts[index]) for index in selected_indices
    )
    summary = AllenLayerRenderSummary(
        total_nodes=int(len(table)),
        flatmap_valid_nodes=int(flatmap_valid.sum()),
        layer_classified_nodes=int(layer_classified.sum()),
        rendered_nodes=int(render_valid.sum()),
        excluded_non_layer_nodes=int((flatmap_valid & ~layer_classified).sum()),
        excluded_unselected_layer_nodes=int(
            (flatmap_valid & layer_classified & ~layer_selected).sum()
        ),
        nonzero_voxels=int(np.count_nonzero(volume)),
        traces_represented=traces_represented,
        y_bins=y_bins,
        x_bins=x_bins,
        x_flat_min=resolved_x_bounds[0],
        x_flat_max=resolved_x_bounds[1],
        y_flat_min=resolved_y_bounds[0],
        y_flat_max=resolved_y_bounds[1],
        layer_labels=selected_labels,
        layer_node_counts=layer_node_counts,
        atlas_name=layer_map.atlas_name,
        atlas_version=layer_map.atlas_version,
        selected_layer_indices=selected_indices,
        output_mode=output_mode,
        output_shape=tuple(int(value) for value in volume.shape),
    )
    return AllenLayerRenderResult(
        projected_nodes=table,
        volume=volume,
        summary=summary,
    )


def build_allen_layer_stack_from_projected_nodes(
    projected_nodes: pd.DataFrame,
    layer_map: AllenIsocortexLayerMap,
    *,
    y_bins: int = DEFAULT_FLATMAP_Y_BINS,
    x_bins: int | None = None,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> AllenLayerStackResult:
    """Compatibility wrapper that builds the original all-six layer stack."""
    return build_allen_layer_render_from_projected_nodes(
        projected_nodes,
        layer_map,
        y_bins=y_bins,
        x_bins=x_bins,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    )


def _rendered_binned_nodes(
    projected_nodes: pd.DataFrame,
    volume_shape: tuple[int, ...],
) -> pd.DataFrame:
    """Filter a projected table to the rows a flatmap volume actually rendered.

    A 2-length ``volume_shape`` selects a depth-collapsed render, which carries
    no ``depth_bin`` column at all.  The rank of the volume decides that -- never
    the presence of a column -- so a stray ``depth_bin`` cannot change behavior.
    """
    if len(volume_shape) not in (2, 3):
        raise ValueError(f"volume_shape must be 2D or 3D; got {volume_shape}.")
    has_depth_axis = len(volume_shape) == 3

    bin_columns = (
        ("depth_bin", "y_flat_bin", "x_flat_bin")
        if has_depth_axis
        else (
            "y_flat_bin",
            "x_flat_bin",
        )
    )
    required = ("render_valid", *bin_columns, "file_id")
    missing = [column for column in required if column not in projected_nodes.columns]
    if missing:
        raise ValueError(f"Projected nodes are missing render column(s): {missing}")

    table = projected_nodes.copy()
    render_valid = table["render_valid"].fillna(False).astype(bool).to_numpy()
    in_bounds = np.ones(len(table), dtype=bool)
    for column, size in zip(bin_columns, volume_shape, strict=True):
        bins = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        in_bounds &= np.isfinite(bins) & (bins >= 0) & (bins < int(size))

    filtered = table.loc[render_valid & in_bounds].copy()
    for column in bin_columns:
        filtered.loc[:, column] = pd.to_numeric(
            filtered[column], errors="coerce"
        ).astype(np.int64)
    return filtered


def rendered_plane_points(
    projected_nodes: pd.DataFrame,
    *,
    plane_column: str | None,
) -> tuple[np.ndarray, list[object]]:
    """Return ``(points, file_ids)`` for the rows a render marked ``render_valid``.

    ``plane_column`` names the axis-0 bin column of the render's coordinate
    space -- ``"depth_bin"`` for a depth stack, ``"allen_layer_index"`` for an
    Allen layer stack -- and ``None`` yields 2-D ``(y_bin, x_bin)`` points for a
    depth-collapsed render.  Every render family writes ``render_valid``,
    ``y_flat_bin`` and ``x_flat_bin``, so this reads coordinates the render
    already assigned instead of each result type carrying its own point array.
    """
    bin_columns = ("y_flat_bin", "x_flat_bin")
    if plane_column is not None:
        bin_columns = (plane_column, *bin_columns)
    required = ("render_valid", *bin_columns)
    missing = [column for column in required if column not in projected_nodes.columns]
    if missing:
        raise ValueError(f"Projected nodes are missing render column(s): {missing}")

    render_valid = projected_nodes["render_valid"].fillna(False).astype(bool).to_numpy()
    columns = [
        pd.to_numeric(projected_nodes[column], errors="coerce").to_numpy(dtype=float)
        for column in bin_columns
    ]
    finite = np.ones(len(projected_nodes), dtype=bool)
    for values in columns:
        finite &= np.isfinite(values) & (values >= 0)
    keep = np.flatnonzero(render_valid & finite)
    if keep.size == 0:
        return np.empty((0, len(bin_columns)), dtype=np.float64), []

    points = np.column_stack([values[keep] for values in columns]).astype(np.float64)
    file_ids = (
        projected_nodes.iloc[keep]["file_id"].tolist()
        if "file_id" in projected_nodes.columns
        else []
    )
    return points, file_ids


def _unique_in_order(values) -> tuple[object, ...]:
    unique: list[object] = []
    seen: set[object] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return tuple(unique)


def _volume_for_node_group(
    group: pd.DataFrame,
    volume_shape: tuple[int, ...],
) -> np.ndarray:
    volume = np.zeros(volume_shape, dtype=np.float32)
    if group.empty:
        return volume
    bin_columns = (
        ("depth_bin", "y_flat_bin", "x_flat_bin")
        if len(volume_shape) == 3
        else ("y_flat_bin", "x_flat_bin")
    )
    np.add.at(
        volume,
        tuple(group[column].to_numpy(dtype=np.int64) for column in bin_columns),
        1.0,
    )
    return volume


def build_flatmap_file_id_volumes(
    projected_nodes: pd.DataFrame,
    volume_shape: tuple[int, ...],
) -> list[FlatmapGroupedVolume]:
    """Split rendered flatmap nodes into one scalar heatmap volume per file ID."""
    rendered = _rendered_binned_nodes(projected_nodes, volume_shape)
    groups: list[FlatmapGroupedVolume] = []
    for file_id in _unique_in_order(rendered["file_id"].tolist()):
        group = rendered[rendered["file_id"] == file_id]
        volume = _volume_for_node_group(group, volume_shape)
        groups.append(
            FlatmapGroupedVolume(
                group_key=file_id,
                label=str(file_id),
                source_file_ids=(file_id,),
                volume=volume,
                rendered_nodes=int(len(group)),
                nonzero_voxels=int(np.count_nonzero(volume)),
            )
        )
    return groups


def _cluster_for_file_id(
    file_id: object,
    cluster_map: dict[object, int | None],
) -> int | None:
    value = cluster_map.get(file_id)
    if value is None and str(file_id) in cluster_map:
        value = cluster_map.get(str(file_id))
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_flatmap_cluster_volumes(
    projected_nodes: pd.DataFrame,
    volume_shape: tuple[int, ...],
    cluster_map: dict[object, int | None],
) -> list[FlatmapGroupedVolume]:
    """Split rendered flatmap nodes into scalar heatmap volumes by cluster ID."""
    rendered = _rendered_binned_nodes(projected_nodes, volume_shape)
    if rendered.empty:
        return []

    table = rendered.copy()
    table.loc[:, "_flatmap_cluster_id"] = [
        _cluster_for_file_id(file_id, cluster_map)
        for file_id in table["file_id"].tolist()
    ]

    cluster_ids = sorted(
        {
            int(cluster_id)
            for cluster_id in table["_flatmap_cluster_id"].tolist()
            if pd.notna(cluster_id)
        }
    )
    group_keys: list[int | None] = list(cluster_ids)
    if table["_flatmap_cluster_id"].isna().any():
        group_keys.append(None)

    groups: list[FlatmapGroupedVolume] = []
    for group_key in group_keys:
        if group_key is None:
            group = table[table["_flatmap_cluster_id"].isna()]
            label = "Unclustered"
        else:
            group = table[table["_flatmap_cluster_id"] == group_key]
            label = f"Cluster {group_key}"
        volume = _volume_for_node_group(group, volume_shape)
        source_file_ids = _unique_in_order(group["file_id"].tolist())
        groups.append(
            FlatmapGroupedVolume(
                group_key=group_key,
                label=label,
                source_file_ids=source_file_ids,
                volume=volume,
                rendered_nodes=int(len(group)),
                nonzero_voxels=int(np.count_nonzero(volume)),
            )
        )
    return groups


def _rendered_allen_layer_nodes(
    projected_nodes: pd.DataFrame,
    volume_shape: tuple[int, ...],
) -> pd.DataFrame:
    required = (
        "render_valid",
        "y_flat_bin",
        "x_flat_bin",
        "file_id",
    )
    missing = [column for column in required if column not in projected_nodes.columns]
    if missing:
        raise ValueError(
            f"Projected nodes are missing Allen-layer render column(s): {missing}"
        )
    if len(volume_shape) not in (2, 3):
        raise ValueError(f"volume_shape must be 2D or 3D; got {volume_shape}.")
    has_plane_axis = len(volume_shape) == 3
    if has_plane_axis:
        plane_size, y_size, x_size = (int(size) for size in volume_shape)
        plane_column = (
            "allen_layer_render_index"
            if "allen_layer_render_index" in projected_nodes.columns
            else "allen_layer_index"
        )
        if plane_column not in projected_nodes.columns:
            raise ValueError(
                "Projected nodes are missing an Allen-layer render index column."
            )
    else:
        y_size, x_size = (int(size) for size in volume_shape)
        plane_column = None
    table = projected_nodes.copy()
    render_valid = table["render_valid"].fillna(False).astype(bool).to_numpy()
    y_bins = pd.to_numeric(table["y_flat_bin"], errors="coerce").to_numpy(dtype=float)
    x_bins = pd.to_numeric(table["x_flat_bin"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y_bins) & np.isfinite(x_bins)
    in_bounds = (
        finite
        & (y_bins >= 0)
        & (y_bins < y_size)
        & (x_bins >= 0)
        & (x_bins < x_size)
    )
    if plane_column is not None:
        plane_bins = pd.to_numeric(
            table[plane_column], errors="coerce"
        ).to_numpy(dtype=float)
        in_bounds &= (
            np.isfinite(plane_bins)
            & (plane_bins >= 0)
            & (plane_bins < plane_size)
        )
    filtered = table.loc[render_valid & in_bounds].copy()
    integer_columns = ["y_flat_bin", "x_flat_bin"]
    if plane_column is not None:
        integer_columns.append(plane_column)
    for column in integer_columns:
        filtered.loc[:, column] = pd.to_numeric(
            filtered[column], errors="coerce"
        ).astype(np.int64)
    return filtered


def _allen_layer_volume_for_node_group(
    group: pd.DataFrame,
    volume_shape: tuple[int, ...],
) -> np.ndarray:
    volume = np.zeros(volume_shape, dtype=np.float32)
    if group.empty:
        return volume
    if len(volume_shape) == 3:
        plane_column = (
            "allen_layer_render_index"
            if "allen_layer_render_index" in group.columns
            else "allen_layer_index"
        )
        coordinates = (
            group[plane_column].to_numpy(dtype=np.int64),
            group["y_flat_bin"].to_numpy(dtype=np.int64),
            group["x_flat_bin"].to_numpy(dtype=np.int64),
        )
    elif len(volume_shape) == 2:
        coordinates = (
            group["y_flat_bin"].to_numpy(dtype=np.int64),
            group["x_flat_bin"].to_numpy(dtype=np.int64),
        )
    else:
        raise ValueError(f"volume_shape must be 2D or 3D; got {volume_shape}.")
    np.add.at(volume, coordinates, 1.0)
    return volume


def build_allen_layer_file_id_volumes(
    projected_nodes: pd.DataFrame,
    volume_shape: tuple[int, ...],
) -> list[FlatmapGroupedVolume]:
    """Split an Allen-layer stack into one scalar volume per neuron."""
    rendered = _rendered_allen_layer_nodes(projected_nodes, volume_shape)
    groups: list[FlatmapGroupedVolume] = []
    for file_id in _unique_in_order(rendered["file_id"].tolist()):
        group = rendered[rendered["file_id"] == file_id]
        volume = _allen_layer_volume_for_node_group(group, volume_shape)
        groups.append(
            FlatmapGroupedVolume(
                group_key=file_id,
                label=str(file_id),
                source_file_ids=(file_id,),
                volume=volume,
                rendered_nodes=int(len(group)),
                nonzero_voxels=int(np.count_nonzero(volume)),
            )
        )
    return groups


def build_allen_layer_cluster_volumes(
    projected_nodes: pd.DataFrame,
    volume_shape: tuple[int, ...],
    cluster_map: dict[object, int | None],
) -> list[FlatmapGroupedVolume]:
    """Split an Allen-layer stack into scalar volumes by cluster ID."""
    rendered = _rendered_allen_layer_nodes(projected_nodes, volume_shape)
    if rendered.empty:
        return []
    table = rendered.copy()
    table.loc[:, "_flatmap_cluster_id"] = [
        _cluster_for_file_id(file_id, cluster_map)
        for file_id in table["file_id"].tolist()
    ]
    cluster_ids = sorted(
        {
            int(cluster_id)
            for cluster_id in table["_flatmap_cluster_id"].tolist()
            if pd.notna(cluster_id)
        }
    )
    group_keys: list[int | None] = list(cluster_ids)
    if table["_flatmap_cluster_id"].isna().any():
        group_keys.append(None)

    groups: list[FlatmapGroupedVolume] = []
    for group_key in group_keys:
        if group_key is None:
            group = table[table["_flatmap_cluster_id"].isna()]
            label = "Unclustered"
        else:
            group = table[table["_flatmap_cluster_id"] == group_key]
            label = f"Cluster {group_key}"
        volume = _allen_layer_volume_for_node_group(group, volume_shape)
        groups.append(
            FlatmapGroupedVolume(
                group_key=group_key,
                label=label,
                source_file_ids=_unique_in_order(group["file_id"].tolist()),
                volume=volume,
                rendered_nodes=int(len(group)),
                nonzero_voxels=int(np.count_nonzero(volume)),
            )
        )
    return groups


def _build_flatmap_render_data_for_bounds(
    projected_nodes: pd.DataFrame,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    depth_range: tuple[float, float],
    y_bins: int,
    x_bins: int | None = None,
    depth_bin_um: float,
    include_depth_minus_one: bool,
    collapse_depth: bool = False,
) -> FlatmapRenderResult:
    """Bin projected nodes into a flatmap node-count volume.

    The grid-construction choke point for the pandas render path: ``x_bins`` is
    derived here, where the bounds the aspect ratio needs are already resolved.

    ``collapse_depth`` removes the depth *axis* only: ``render_valid`` is
    computed exactly as for a depth-binned render, so ``include_depth_minus_one``
    still decides whether depth ``-1`` nodes contribute, and the result is the
    depth volume summed over its plane axis.  The volume becomes
    ``(y_bins, x_bins)``, points become ``(N, 2)``, and no ``depth_bin``
    column is written.
    """
    y_bins, x_bins, depth_bin_um = _resolve_axis_bin_counts(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        y_bins=y_bins,
        x_bins=x_bins,
        depth_bin_um=depth_bin_um,
    )
    valid_depth_bins = _depth_bin_count(depth_range, depth_bin_um)
    sentinel_offset = 1 if include_depth_minus_one else 0
    total_depth_bins = valid_depth_bins + sentinel_offset
    volume_shape: tuple[int, ...] = (
        (y_bins, x_bins) if collapse_depth else (total_depth_bins, y_bins, x_bins)
    )
    voxel_count = int(np.prod(volume_shape))
    if voxel_count > MAX_FLATMAP_HEATMAP_VOXELS:
        raise ValueError(
            "Flatmap heatmap is too large: "
            f"{'x'.join(str(size) for size in volume_shape)} voxels. "
            "Use fewer Y bins or a larger depth bin."
        )

    table = projected_nodes.copy()
    for column, default in (
        ("flatmap_valid", False),
        ("depth_valid", False),
    ):
        if column not in table.columns:
            table[column] = default

    flatmap_valid = table["flatmap_valid"].fillna(False).astype(bool).to_numpy()
    depth_valid = table["depth_valid"].fillna(False).astype(bool).to_numpy()
    depth_values = pd.to_numeric(table["depth_um"], errors="coerce").to_numpy(
        dtype=float
    )
    depth_minus_one = np.isfinite(depth_values) & (depth_values == -1.0)
    render_valid = flatmap_valid & (
        depth_valid | (include_depth_minus_one & depth_minus_one)
    )

    x_bin_indices = np.full(len(table), -1, dtype=np.int64)
    y_bin_indices = np.full(len(table), -1, dtype=np.int64)
    depth_bin_indices = np.full(len(table), -1, dtype=np.int64)

    if render_valid.any():
        x_values = pd.to_numeric(table["x_flat"], errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(table["y_flat"], errors="coerce").to_numpy(dtype=float)
        x_bin_indices[render_valid] = _bin_flat_values(
            x_values[render_valid],
            x_bounds,
            x_bins,
        )
        y_bin_indices[render_valid] = _bin_flat_values(
            y_values[render_valid],
            y_bounds,
            y_bins,
        )

        sentinel_nodes = render_valid & depth_minus_one
        valid_depth_nodes = render_valid & depth_valid
        if sentinel_nodes.any():
            depth_bin_indices[sentinel_nodes] = 0
        if valid_depth_nodes.any():
            raw_depth_bins = np.floor(
                (depth_values[valid_depth_nodes] - depth_range[0]) / depth_bin_um
            ).astype(np.int64)
            raw_depth_bins = np.clip(raw_depth_bins, 0, valid_depth_bins - 1)
            depth_bin_indices[valid_depth_nodes] = raw_depth_bins + sentinel_offset

    volume = np.zeros(volume_shape, dtype=np.float32)
    render_indices = np.flatnonzero(render_valid)
    if collapse_depth:
        scatter_columns: tuple[np.ndarray, ...] = (
            y_bin_indices[render_indices],
            x_bin_indices[render_indices],
        )
    else:
        scatter_columns = (
            depth_bin_indices[render_indices],
            y_bin_indices[render_indices],
            x_bin_indices[render_indices],
        )
    if render_indices.size:
        np.add.at(volume, scatter_columns, 1.0)

    points = (
        np.column_stack(scatter_columns).astype(np.float64)
        if render_indices.size
        else np.empty((0, len(volume_shape)), dtype=np.float64)
    )
    point_file_ids = (
        table.iloc[render_indices]["file_id"].tolist()
        if render_indices.size and "file_id" in table.columns
        else []
    )

    table.loc[:, "render_valid"] = render_valid
    table.loc[:, "x_flat_bin"] = x_bin_indices
    table.loc[:, "y_flat_bin"] = y_bin_indices
    if not collapse_depth:
        table.loc[:, "depth_bin"] = depth_bin_indices
        table.loc[:, "depth_bin_label"] = _depth_labels(
            depth_bin_indices,
            depth_min_um=depth_range[0],
            depth_bin_um=depth_bin_um,
            sentinel_offset=sentinel_offset,
        )

    if render_indices.size and "file_id" in table.columns:
        traces_represented = int(table.iloc[render_indices]["file_id"].nunique())
    else:
        traces_represented = 0
    excluded_depth_minus_one = int(
        (flatmap_valid & depth_minus_one & ~render_valid).sum()
    )

    summary = FlatmapRenderSummary(
        total_nodes=int(len(table)),
        flatmap_valid_nodes=int(flatmap_valid.sum()),
        depth_valid_nodes=int((flatmap_valid & depth_valid).sum()),
        depth_minus_one_nodes=int((flatmap_valid & depth_minus_one).sum()),
        rendered_nodes=int(render_valid.sum()),
        excluded_depth_minus_one_nodes=excluded_depth_minus_one,
        nonzero_voxels=int(np.count_nonzero(volume)),
        traces_represented=traces_represented,
        y_bins=y_bins,
        x_bins=x_bins,
        # A collapsed render has no depth planes and no depth bin size.  Zeros
        # say exactly that: ``depth_plane_labels`` returns an empty tuple for
        # them rather than inventing plane names for an axis that is not there.
        depth_bins=0 if collapse_depth else total_depth_bins,
        depth_bin_um=0.0 if collapse_depth else depth_bin_um,
        x_flat_min=x_bounds[0],
        x_flat_max=x_bounds[1],
        y_flat_min=y_bounds[0],
        y_flat_max=y_bounds[1],
        depth_min_um=depth_range[0],
        depth_max_um=depth_range[1],
        includes_depth_minus_one_plane=bool(include_depth_minus_one),
    )
    return FlatmapRenderResult(
        projected_nodes=table,
        volume=volume,
        points=points,
        point_file_ids=point_file_ids,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# DuckDB-backed precomputed heatmap rendering.
#
# The pandas render path above materializes every queried node in memory before
# binning it with ``np.add.at``.  For whole-brain Parquets (hundreds of millions
# of rows) that is prohibitively slow.  The functions below mirror the CCFv3
# ``build_node_counts_volume`` approach instead: read only the handful of
# precomputed flatmap columns via projection pushdown and let DuckDB bin the
# nodes with a ``GROUP BY``, returning just the sparse non-zero voxel counts.
# The bin math is kept identical to ``_build_flatmap_render_data_for_bounds`` so
# the fast path produces the same volume as the pandas path.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlatmapAggregateStats:
    """Node/trace tallies gathered in a single streaming DuckDB pass."""

    total_nodes: int
    total_traces: int
    flatmap_valid_nodes: int
    depth_valid_nodes: int
    depth_minus_one_nodes: int
    rendered_nodes: int
    traces_represented: int


@dataclass(frozen=True)
class FlatmapHeatmapVolumeResult:
    """Binned flatmap heatmap volume(s) built directly from Parquet columns."""

    color_mode: str
    volume: np.ndarray | None
    grouped_volumes: tuple[FlatmapGroupedVolume, ...]
    render_summary: FlatmapRenderSummary
    stats: FlatmapAggregateStats
    volume_shape: tuple[int, ...]


def _sql_number(value: float) -> str:
    """Render a Python float as a lossless SQL numeric literal."""
    return repr(float(value))


def _sql_identifier(name: str) -> str:
    """Quote a column name for safe inline use in SQL."""
    escaped = str(name).replace('"', '""')
    return f'"{escaped}"'


def _duckdb_source_path(parquet_path: str) -> str:
    return str(parquet_path).replace("\\", "/")


def _duckdb_column_names(conn: duckdb.DuckDBPyConnection, source_sql: str) -> set[str]:
    relation = conn.execute(f"SELECT * FROM {source_sql} LIMIT 0")
    return {str(description[0]) for description in relation.description}


def _style_suffix(style_key: str) -> str:
    suffix = _FLATMAP_STYLE_SUFFIX.get(str(style_key))
    if suffix is None:
        raise ValueError(
            "DuckDB flatmap heatmap rendering supports the bilateral shaped and "
            f"bilateral square styles only; got {style_key!r}."
        )
    return suffix


def _flatmap_sql_expressions(
    column_names: set[str],
    *,
    suffix: str,
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
    depth_lower: float,
    depth_bin_um: float,
    y_bins: int,
    x_bins: int,
    valid_depth_bins: int,
    sentinel_offset: int,
    include_depth_minus_one: bool,
) -> dict[str, str]:
    """Build the SQL boolean/bin expressions matching the pandas render path."""
    x_column = f"x_flat_{suffix}"
    y_column = f"y_flat_{suffix}"
    for required in (x_column, y_column, "depth_um", "file_id"):
        if required not in column_names:
            raise ValueError(
                f"Parquet is missing required column {required!r} for flatmap "
                "heatmap rendering."
            )

    x_ref = _sql_identifier(x_column)
    y_ref = _sql_identifier(y_column)
    depth_ref = _sql_identifier("depth_um")

    flatmap_valid_column = f"flatmap_{suffix}_valid"
    if flatmap_valid_column in column_names:
        flatmap_valid = f"COALESCE({_sql_identifier(flatmap_valid_column)}, FALSE)"
    else:
        flatmap_valid = (
            f"({x_ref} IS NOT NULL AND isfinite({x_ref}) "
            f"AND {y_ref} IS NOT NULL AND isfinite({y_ref}))"
        )

    if "depth_valid" in column_names:
        depth_valid = f"COALESCE({_sql_identifier('depth_valid')}, FALSE)"
    else:
        depth_valid = (
            f"({depth_ref} IS NOT NULL AND isfinite({depth_ref}) "
            f"AND {depth_ref} >= 0.0)"
        )

    depth_minus_one = (
        f"({depth_ref} IS NOT NULL AND isfinite({depth_ref}) AND {depth_ref} = -1.0)"
    )

    if include_depth_minus_one:
        render_where = f"({flatmap_valid}) AND (({depth_valid}) OR ({depth_minus_one}))"
    else:
        render_where = f"({flatmap_valid}) AND ({depth_valid})"

    # Non-finite coordinates never survive ``render_where`` for a correctly built
    # Parquet, but guard the CAST anyway: DuckDB raises on CAST(FLOOR(inf)) while
    # numpy silently clamps.  Falling back to bin 0 keeps corrupt rows from
    # aborting the whole render.
    x_span = x_upper - x_lower
    y_span = y_upper - y_lower
    x_bin = (
        f"CASE WHEN isfinite({x_ref}) THEN "
        f"LEAST(GREATEST(CAST(FLOOR(({x_ref} - {_sql_number(x_lower)}) "
        f"/ {_sql_number(x_span)} * {int(x_bins)}) AS BIGINT), 0), {int(x_bins) - 1}) "
        f"ELSE 0 END"
    )
    y_bin = (
        f"CASE WHEN isfinite({y_ref}) THEN "
        f"LEAST(GREATEST(CAST(FLOOR(({y_ref} - {_sql_number(y_lower)}) "
        f"/ {_sql_number(y_span)} * {int(y_bins)}) AS BIGINT), 0), {int(y_bins) - 1}) "
        f"ELSE 0 END"
    )
    depth_bin = (
        f"CASE WHEN ({depth_valid}) THEN "
        f"(CASE WHEN isfinite({depth_ref}) THEN "
        f"LEAST(GREATEST(CAST(FLOOR(({depth_ref} - {_sql_number(depth_lower)}) "
        f"/ {_sql_number(depth_bin_um)}) AS BIGINT), 0), {int(valid_depth_bins) - 1}) "
        f"ELSE 0 END) + {int(sentinel_offset)} "
        f"ELSE 0 END"
    )

    return {
        "flatmap_valid": flatmap_valid,
        "depth_valid": depth_valid,
        "depth_minus_one": depth_minus_one,
        "render_where": render_where,
        "x_bin": x_bin,
        "y_bin": y_bin,
        "depth_bin": depth_bin,
    }


def _file_id_filter(
    file_ids: list[object] | None,
) -> tuple[str, list[object]] | None:
    """Return ``(sql, params)`` for a ``file_id IN (...)`` clause.

    Returns ``("", [])`` when no filter is requested (``file_ids is None``) and
    ``None`` when the caller passed an explicit empty selection (render nothing).
    """
    if file_ids is None:
        return "", []
    if len(file_ids) == 0:
        return None
    placeholders = ", ".join(["?"] * len(file_ids))
    return f"file_id IN ({placeholders})", [str(file_id) for file_id in file_ids]


def _combine_where(render_where: str, file_filter_sql: str) -> str:
    if file_filter_sql:
        return f"({render_where}) AND {file_filter_sql}"
    return render_where


def _query_flatmap_stats(
    conn: duckdb.DuckDBPyConnection,
    source_sql: str,
    expressions: dict[str, str],
    file_filter_sql: str,
    params: list[object],
) -> FlatmapAggregateStats:
    flatmap_valid = expressions["flatmap_valid"]
    depth_valid = expressions["depth_valid"]
    depth_minus_one = expressions["depth_minus_one"]
    render_where = expressions["render_where"]
    where_sql = f"WHERE {file_filter_sql}" if file_filter_sql else ""
    query = f"""
        SELECT
            COUNT(*) AS total_nodes,
            COUNT(DISTINCT file_id) AS total_traces,
            COUNT(*) FILTER (WHERE {flatmap_valid}) AS flatmap_valid_nodes,
            COUNT(*) FILTER (WHERE ({flatmap_valid}) AND ({depth_valid}))
                AS depth_valid_nodes,
            COUNT(*) FILTER (WHERE ({flatmap_valid}) AND ({depth_minus_one}))
                AS depth_minus_one_nodes,
            COUNT(*) FILTER (WHERE {render_where}) AS rendered_nodes,
            COUNT(DISTINCT file_id) FILTER (WHERE {render_where})
                AS traces_represented
        FROM {source_sql}
        {where_sql}
    """
    row = (
        conn.execute(query, params).fetchone()
        if params
        else conn.execute(query).fetchone()
    )
    if row is None:
        row = (0, 0, 0, 0, 0, 0, 0)
    return FlatmapAggregateStats(
        total_nodes=int(row[0] or 0),
        total_traces=int(row[1] or 0),
        flatmap_valid_nodes=int(row[2] or 0),
        depth_valid_nodes=int(row[3] or 0),
        depth_minus_one_nodes=int(row[4] or 0),
        rendered_nodes=int(row[5] or 0),
        traces_represented=int(row[6] or 0),
    )


def _scatter_bin_counts(
    volume: np.ndarray,
    frame: pd.DataFrame,
) -> None:
    """Accumulate ``node_count`` into ``volume`` at its bin coordinates.

    A 3D ``volume`` is indexed by ``(depth_bin, y_bin, x_bin)``; a 2D one -- a
    depth-collapsed render -- by ``(y_bin, x_bin)``, and the query that built
    ``frame`` selected no depth column at all.
    """
    if frame.empty:
        return
    if volume.ndim == 3:
        bin_columns = ("depth_bin", "y_bin", "x_bin")
    elif volume.ndim == 2:
        bin_columns = ("y_bin", "x_bin")
    else:
        raise ValueError(f"volume must be 2D or 3D; got {volume.ndim}D.")
    np.add.at(
        volume,
        tuple(frame[column].to_numpy(dtype=np.intp) for column in bin_columns),
        frame["node_count"].to_numpy(dtype=np.float32),
    )


def _query_flatmap_bin_counts(
    conn: duckdb.DuckDBPyConnection,
    source_sql: str,
    expressions: dict[str, str],
    where_sql: str,
    params: list[object],
    *,
    include_file_id: bool,
    include_depth_bin: bool = True,
) -> pd.DataFrame:
    """Group rendered nodes into sparse bin counts.

    ``include_depth_bin=False`` collapses the depth axis: the depth bin is
    dropped from both the projection and the grouping keys, so DuckDB returns
    one row per flatmap XY bin.  ``where_sql`` is unchanged either way, so depth
    still decides which nodes are counted.
    """
    file_select = "file_id, " if include_file_id else ""
    file_group = ", file_id" if include_file_id else ""
    depth_select = (
        f"{expressions['depth_bin']} AS depth_bin,\n            "
        if include_depth_bin
        else ""
    )
    depth_group = "depth_bin, " if include_depth_bin else ""
    query = f"""
        SELECT
            {file_select}{depth_select}{expressions["y_bin"]} AS y_bin,
            {expressions["x_bin"]} AS x_bin,
            COUNT(*) AS node_count
        FROM {source_sql}
        WHERE {where_sql}
        GROUP BY {depth_group}y_bin, x_bin{file_group}
    """
    return (
        conn.execute(query, params).fetchdf()
        if params
        else conn.execute(query).fetchdf()
    )


def _build_grouped_flatmap_volumes(
    counts: pd.DataFrame,
    volume_shape: tuple[int, ...],
    *,
    color_mode: str,
    cluster_map: dict[object, int | None] | None,
) -> tuple[list[FlatmapGroupedVolume], np.ndarray]:
    """Scatter per-file bin counts into one volume per group + a combined volume."""
    combined = np.zeros(volume_shape, dtype=np.float32)
    if counts.empty:
        return [], combined
    _scatter_bin_counts(combined, counts)

    file_ids_in_order = _unique_in_order(counts["file_id"].tolist())

    if color_mode == FLATMAP_HEATMAP_COLOR_CLUSTER:
        resolved_map = cluster_map or {}
        cluster_for_file = {
            file_id: _cluster_for_file_id(file_id, resolved_map)
            for file_id in file_ids_in_order
        }
        cluster_ids = sorted(
            {
                cluster_id
                for cluster_id in cluster_for_file.values()
                if cluster_id is not None
            }
        )
        group_keys: list[int | None] = list(cluster_ids)
        if any(cluster_id is None for cluster_id in cluster_for_file.values()):
            group_keys.append(None)

        groups: list[FlatmapGroupedVolume] = []
        for group_key in group_keys:
            source_file_ids = tuple(
                file_id
                for file_id in file_ids_in_order
                if cluster_for_file[file_id] == group_key
            )
            label = "Unclustered" if group_key is None else f"Cluster {group_key}"
            group_counts = counts[counts["file_id"].isin(source_file_ids)]
            volume = np.zeros(volume_shape, dtype=np.float32)
            _scatter_bin_counts(volume, group_counts)
            groups.append(
                FlatmapGroupedVolume(
                    group_key=group_key,
                    label=label,
                    source_file_ids=source_file_ids,
                    volume=volume,
                    rendered_nodes=int(group_counts["node_count"].sum()),
                    nonzero_voxels=int(np.count_nonzero(volume)),
                )
            )
        return groups, combined

    # Individual (per-file) coloring.
    groups = []
    for file_id in file_ids_in_order:
        group_counts = counts[counts["file_id"] == file_id]
        volume = np.zeros(volume_shape, dtype=np.float32)
        _scatter_bin_counts(volume, group_counts)
        groups.append(
            FlatmapGroupedVolume(
                group_key=file_id,
                label=str(file_id),
                source_file_ids=(file_id,),
                volume=volume,
                rendered_nodes=int(group_counts["node_count"].sum()),
                nonzero_voxels=int(np.count_nonzero(volume)),
            )
        )
    return groups, combined


def _empty_flatmap_volume_result(
    color_mode: str,
    volume_shape: tuple[int, ...],
    render_summary: FlatmapRenderSummary,
    stats: FlatmapAggregateStats,
) -> FlatmapHeatmapVolumeResult:
    volume = (
        np.zeros(volume_shape, dtype=np.float32)
        if color_mode == FLATMAP_HEATMAP_COLOR_SINGLE
        else None
    )
    return FlatmapHeatmapVolumeResult(
        color_mode=color_mode,
        volume=volume,
        grouped_volumes=(),
        render_summary=render_summary,
        stats=stats,
        volume_shape=volume_shape,
    )


def build_flatmap_heatmap_volume_result(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    *,
    style_key: str,
    color_mode: str,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    depth_range_um: tuple[float, float],
    y_bins: int = DEFAULT_FLATMAP_Y_BINS,
    x_bins: int | None = None,
    depth_bin_um: float = DEFAULT_FLATMAP_DEPTH_BIN_UM,
    include_depth_minus_one: bool = True,
    file_ids: list[object] | None = None,
    cluster_map: dict[object, int | None] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    progress_total: int = 3,
    collapse_depth: bool = False,
) -> FlatmapHeatmapVolumeResult:
    """Build a flatmap heatmap volume from precomputed Parquet columns via DuckDB.

    The bin math mirrors :func:`_build_flatmap_render_data_for_bounds` so this
    fast path yields the same volume the pandas path would, but only the needed
    coordinate/validity columns are read and the binning happens inside DuckDB.

    Parameters
    ----------
    conn
        Open DuckDB connection.
    parquet_path
        Path to a version-3 augmented Parquet with ``*_shaped``/``*_square``
        flatmap coordinate columns and ``depth_um``.
    style_key
        ``"both_shaped"`` or ``"both_square"`` (selects the column family).
    color_mode
        One of :data:`FLATMAP_HEATMAP_COLOR_SINGLE`,
        :data:`FLATMAP_HEATMAP_COLOR_INDIVIDUAL`, or
        :data:`FLATMAP_HEATMAP_COLOR_CLUSTER`.
    x_bounds, y_bounds, depth_range_um
        Canonical render bounds (typically from the region cache profile or the
        v3 Parquet metadata).  They are made non-degenerate exactly as the
        pandas path does.
    file_ids
        Restrict rendering to these ``file_id`` values.  ``None`` renders every
        row; an empty list renders nothing.
    cluster_map
        ``file_id -> cluster id`` mapping used only when ``color_mode`` is
        ``cluster``.
    collapse_depth
        Drop the depth axis, yielding a ``(y_bins, x_bins)`` volume equal to
        the depth volume summed over its planes.  Which nodes are counted is
        unchanged -- ``depth_range_um``, ``depth_bin_um``, and
        ``include_depth_minus_one`` still govern the ``WHERE`` clause -- so only
        the grouping keys differ.
    """
    suffix = _style_suffix(style_key)
    x_lower, x_upper = _nondegenerate_bounds(x_bounds[0], x_bounds[1])
    y_lower, y_upper = _nondegenerate_bounds(y_bounds[0], y_bounds[1])
    depth_lower, depth_upper = _nondegenerate_bounds(
        depth_range_um[0], depth_range_um[1]
    )
    y_bins, x_bins, depth_bin_um = _resolve_axis_bin_counts(
        x_bounds=(x_lower, x_upper),
        y_bounds=(y_lower, y_upper),
        y_bins=y_bins,
        x_bins=x_bins,
        depth_bin_um=depth_bin_um,
    )

    valid_depth_bins = _depth_bin_count((depth_lower, depth_upper), depth_bin_um)
    sentinel_offset = 1 if include_depth_minus_one else 0
    total_depth_bins = valid_depth_bins + sentinel_offset
    volume_shape: tuple[int, ...] = (
        (y_bins, x_bins) if collapse_depth else (total_depth_bins, y_bins, x_bins)
    )

    voxel_count = int(np.prod(volume_shape))
    if voxel_count > MAX_FLATMAP_HEATMAP_VOXELS:
        raise ValueError(
            "Flatmap heatmap is too large: "
            f"{'x'.join(str(size) for size in volume_shape)} voxels. "
            "Use fewer Y bins or a larger depth bin."
        )

    source_sql = f"read_parquet('{_duckdb_source_path(parquet_path)}')"
    column_names = _duckdb_column_names(conn, source_sql)
    expressions = _flatmap_sql_expressions(
        column_names,
        suffix=suffix,
        x_lower=x_lower,
        x_upper=x_upper,
        y_lower=y_lower,
        y_upper=y_upper,
        depth_lower=depth_lower,
        depth_bin_um=depth_bin_um,
        y_bins=y_bins,
        x_bins=x_bins,
        valid_depth_bins=valid_depth_bins,
        sentinel_offset=sentinel_offset,
        include_depth_minus_one=include_depth_minus_one,
    )

    file_filter = _file_id_filter(file_ids)

    def _summary(
        stats: FlatmapAggregateStats, nonzero_voxels: int
    ) -> FlatmapRenderSummary:
        excluded_depth_minus_one = (
            0 if include_depth_minus_one else stats.depth_minus_one_nodes
        )
        return FlatmapRenderSummary(
            total_nodes=stats.total_nodes,
            flatmap_valid_nodes=stats.flatmap_valid_nodes,
            depth_valid_nodes=stats.depth_valid_nodes,
            depth_minus_one_nodes=stats.depth_minus_one_nodes,
            rendered_nodes=stats.rendered_nodes,
            excluded_depth_minus_one_nodes=excluded_depth_minus_one,
            nonzero_voxels=int(nonzero_voxels),
            traces_represented=stats.traces_represented,
            y_bins=y_bins,
            x_bins=x_bins,
            # Zeros mark "no depth axis", matching the pandas collapsed render.
            depth_bins=0 if collapse_depth else total_depth_bins,
            depth_bin_um=0.0 if collapse_depth else depth_bin_um,
            x_flat_min=x_lower,
            x_flat_max=x_upper,
            y_flat_min=y_lower,
            y_flat_max=y_upper,
            depth_min_um=depth_lower,
            depth_max_um=depth_upper,
            includes_depth_minus_one_plane=bool(include_depth_minus_one),
        )

    empty_stats = FlatmapAggregateStats(0, 0, 0, 0, 0, 0, 0)
    if file_filter is None:
        # Explicit empty selection: render nothing without touching the Parquet.
        return _empty_flatmap_volume_result(
            color_mode, volume_shape, _summary(empty_stats, 0), empty_stats
        )

    file_filter_sql, file_params = file_filter

    _emit_flatmap_progress(
        progress_callback, "Counting flatmap nodes...", 1, progress_total
    )
    stats = _query_flatmap_stats(
        conn, source_sql, expressions, file_filter_sql, file_params
    )

    _emit_flatmap_progress(
        progress_callback, "Binning flatmap heatmap in DuckDB...", 2, progress_total
    )
    where_sql = _combine_where(expressions["render_where"], file_filter_sql)

    if color_mode == FLATMAP_HEATMAP_COLOR_SINGLE:
        counts = _query_flatmap_bin_counts(
            conn,
            source_sql,
            expressions,
            where_sql,
            file_params,
            include_file_id=False,
            include_depth_bin=not collapse_depth,
        )
        volume = np.zeros(volume_shape, dtype=np.float32)
        _scatter_bin_counts(volume, counts)
        render_summary = _summary(stats, int(np.count_nonzero(volume)))
        logger.info(
            "Flatmap heatmap volume: shape %s, rendered nodes %d, non-zero voxels %d",
            volume.shape,
            render_summary.rendered_nodes,
            render_summary.nonzero_voxels,
        )
        return FlatmapHeatmapVolumeResult(
            color_mode=color_mode,
            volume=volume,
            grouped_volumes=(),
            render_summary=render_summary,
            stats=stats,
            volume_shape=volume_shape,
        )

    counts = _query_flatmap_bin_counts(
        conn,
        source_sql,
        expressions,
        where_sql,
        file_params,
        include_file_id=True,
        include_depth_bin=not collapse_depth,
    )
    groups, combined = _build_grouped_flatmap_volumes(
        counts,
        volume_shape,
        color_mode=color_mode,
        cluster_map=cluster_map,
    )
    render_summary = _summary(stats, int(np.count_nonzero(combined)))
    logger.info(
        "Flatmap heatmap groups: %d group(s), rendered nodes %d, non-zero voxels %d",
        len(groups),
        render_summary.rendered_nodes,
        render_summary.nonzero_voxels,
    )
    return FlatmapHeatmapVolumeResult(
        color_mode=color_mode,
        volume=None,
        grouped_volumes=tuple(groups),
        render_summary=render_summary,
        stats=stats,
        volume_shape=volume_shape,
    )


def _allen_layer_sql_expressions(
    column_names: set[str],
    *,
    suffix: str,
    layer_map: AllenIsocortexLayerMap,
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
    y_bins: int,
    x_bins: int,
    selected_layer_indices: tuple[int, ...],
) -> dict[str, str]:
    x_column = f"x_flat_{suffix}"
    y_column = f"y_flat_{suffix}"
    for required in (x_column, y_column, "region_id", "file_id"):
        if required not in column_names:
            raise ValueError(
                f"Parquet is missing required column {required!r} for an "
                "Allen layer heatmap."
            )
    x_ref = _sql_identifier(x_column)
    y_ref = _sql_identifier(y_column)
    region_ref = _sql_identifier("region_id")
    flatmap_valid_column = f"flatmap_{suffix}_valid"
    if flatmap_valid_column in column_names:
        flatmap_valid = (
            f"COALESCE({_sql_identifier(flatmap_valid_column)}, FALSE) "
            f"AND isfinite({x_ref}) AND isfinite({y_ref})"
        )
    else:
        flatmap_valid = (
            f"({x_ref} IS NOT NULL AND isfinite({x_ref}) "
            f"AND {y_ref} IS NOT NULL AND isfinite({y_ref}))"
        )

    case_parts = []
    for index, region_ids in enumerate(layer_map.region_ids_by_layer):
        id_list = ", ".join(str(int(value)) for value in region_ids)
        case_parts.append(f"WHEN {region_ref} IN ({id_list}) THEN {int(index)}")
    layer_index = "CASE " + " ".join(case_parts) + " ELSE NULL END"
    selected_values = ", ".join(str(index) for index in selected_layer_indices)
    layer_selected = f"({layer_index}) IN ({selected_values})"
    render_index_parts = " ".join(
        f"WHEN {canonical_index} THEN {render_index}"
        for render_index, canonical_index in enumerate(selected_layer_indices)
    )
    render_plane_index = (
        f"CASE ({layer_index}) {render_index_parts} ELSE NULL END"
    )

    x_span = x_upper - x_lower
    y_span = y_upper - y_lower
    x_bin = (
        f"LEAST(GREATEST(CAST(FLOOR(({x_ref} - {_sql_number(x_lower)}) "
        f"/ {_sql_number(x_span)} * {int(x_bins)}) AS BIGINT), 0), "
        f"{int(x_bins) - 1})"
    )
    y_bin = (
        f"LEAST(GREATEST(CAST(FLOOR(({y_ref} - {_sql_number(y_lower)}) "
        f"/ {_sql_number(y_span)} * {int(y_bins)}) AS BIGINT), 0), "
        f"{int(y_bins) - 1})"
    )
    return {
        "flatmap_valid": flatmap_valid,
        "layer_index": layer_index,
        "layer_classified": f"({layer_index}) IS NOT NULL",
        "layer_selected": layer_selected,
        "render_plane_index": render_plane_index,
        "render_where": f"({flatmap_valid}) AND ({layer_selected})",
        "x_bin": x_bin,
        "y_bin": y_bin,
    }


def _query_allen_layer_stats(
    conn: duckdb.DuckDBPyConnection,
    source_sql: str,
    expressions: dict[str, str],
    file_filter_sql: str,
    params: list[object],
) -> AllenLayerAggregateStats:
    where_sql = f"WHERE {file_filter_sql}" if file_filter_sql else ""
    flatmap_valid = expressions["flatmap_valid"]
    layer_classified = expressions["layer_classified"]
    layer_selected = expressions["layer_selected"]
    render_where = expressions["render_where"]
    query = f"""
        SELECT
            COUNT(*) AS total_nodes,
            COUNT(DISTINCT file_id) AS total_traces,
            COUNT(*) FILTER (WHERE {flatmap_valid}) AS flatmap_valid_nodes,
            COUNT(*) FILTER (WHERE {layer_classified})
                AS layer_classified_nodes,
            COUNT(*) FILTER (WHERE {render_where}) AS rendered_nodes,
            COUNT(DISTINCT file_id) FILTER (WHERE {render_where})
                AS traces_represented,
            COUNT(*) FILTER (
                WHERE ({flatmap_valid}) AND ({layer_classified})
                    AND NOT ({layer_selected})
            ) AS excluded_unselected_layer_nodes
        FROM {source_sql}
        {where_sql}
    """
    row = (
        conn.execute(query, params).fetchone()
        if params
        else conn.execute(query).fetchone()
    )
    if row is None:
        row = (0, 0, 0, 0, 0, 0, 0)
    return AllenLayerAggregateStats(
        total_nodes=int(row[0] or 0),
        total_traces=int(row[1] or 0),
        flatmap_valid_nodes=int(row[2] or 0),
        layer_classified_nodes=int(row[3] or 0),
        rendered_nodes=int(row[4] or 0),
        traces_represented=int(row[5] or 0),
        excluded_unselected_layer_nodes=int(row[6] or 0),
    )


def _query_allen_layer_bin_counts(
    conn: duckdb.DuckDBPyConnection,
    source_sql: str,
    expressions: dict[str, str],
    where_sql: str,
    params: list[object],
    *,
    include_file_id: bool,
    include_plane_bin: bool,
) -> pd.DataFrame:
    file_select = "file_id, " if include_file_id else ""
    file_group = ", file_id" if include_file_id else ""
    plane_select = (
        f'{expressions["render_plane_index"]} AS depth_bin,\n            '
        if include_plane_bin
        else ""
    )
    plane_group = "depth_bin, " if include_plane_bin else ""
    query = f"""
        SELECT
            {file_select}{expressions["layer_index"]} AS allen_layer_index,
            {plane_select}
            {expressions["y_bin"]} AS y_bin,
            {expressions["x_bin"]} AS x_bin,
            COUNT(*) AS node_count
        FROM {source_sql}
        WHERE {where_sql}
        GROUP BY allen_layer_index, {plane_group}y_bin, x_bin{file_group}
    """
    return (
        conn.execute(query, params).fetchdf()
        if params
        else conn.execute(query).fetchdf()
    )


def build_allen_layer_heatmap_volume_result(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    *,
    style_key: str,
    color_mode: str,
    layer_map: AllenIsocortexLayerMap,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    y_bins: int = DEFAULT_FLATMAP_Y_BINS,
    x_bins: int | None = None,
    file_ids: list[object] | None = None,
    cluster_map: dict[object, int | None] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    progress_total: int = 3,
    selected_layer_indices: Sequence[int] | None = None,
    output_mode: str = ALLEN_LAYER_OUTPUT_STACK,
) -> AllenLayerHeatmapVolumeResult:
    """Build a selected Allen-layer stack or XY projection in DuckDB."""
    selected_indices, selected_labels, output_mode = (
        _resolve_allen_layer_render_options(
            layer_map,
            selected_layer_indices,
            output_mode,
        )
    )
    suffix = _style_suffix(style_key)
    x_lower, x_upper = _nondegenerate_bounds(*x_bounds)
    y_lower, y_upper = _nondegenerate_bounds(*y_bounds)
    y_bins, x_bins, _unused_depth_bin = _resolve_axis_bin_counts(
        x_bounds=(x_lower, x_upper),
        y_bounds=(y_lower, y_upper),
        y_bins=y_bins,
        x_bins=x_bins,
    )
    canonical_plane_count = len(layer_map.layer_labels)
    volume_shape = (
        (len(selected_indices), y_bins, x_bins)
        if output_mode == ALLEN_LAYER_OUTPUT_STACK
        else (y_bins, x_bins)
    )
    if int(np.prod(volume_shape)) > MAX_FLATMAP_HEATMAP_VOXELS:
        raise ValueError(
            "Allen layer heatmap is too large: "
            f"{'x'.join(str(value) for value in volume_shape)} voxels. "
            "Use fewer Y bins."
        )

    source_sql = f"read_parquet('{_duckdb_source_path(parquet_path)}')"
    column_names = _duckdb_column_names(conn, source_sql)
    expressions = _allen_layer_sql_expressions(
        column_names,
        suffix=suffix,
        layer_map=layer_map,
        x_lower=x_lower,
        x_upper=x_upper,
        y_lower=y_lower,
        y_upper=y_upper,
        y_bins=y_bins,
        x_bins=x_bins,
        selected_layer_indices=selected_indices,
    )
    file_filter = _file_id_filter(file_ids)

    def _summary(
        stats: AllenLayerAggregateStats,
        *,
        nonzero_voxels: int,
        layer_node_counts: tuple[int, ...],
    ) -> AllenLayerRenderSummary:
        return AllenLayerRenderSummary(
            total_nodes=stats.total_nodes,
            flatmap_valid_nodes=stats.flatmap_valid_nodes,
            layer_classified_nodes=stats.layer_classified_nodes,
            rendered_nodes=stats.rendered_nodes,
            excluded_non_layer_nodes=max(
                0,
                stats.flatmap_valid_nodes
                - stats.rendered_nodes
                - stats.excluded_unselected_layer_nodes,
            ),
            excluded_unselected_layer_nodes=(
                stats.excluded_unselected_layer_nodes
            ),
            nonzero_voxels=int(nonzero_voxels),
            traces_represented=stats.traces_represented,
            y_bins=y_bins,
            x_bins=x_bins,
            x_flat_min=x_lower,
            x_flat_max=x_upper,
            y_flat_min=y_lower,
            y_flat_max=y_upper,
            layer_labels=selected_labels,
            layer_node_counts=layer_node_counts,
            atlas_name=layer_map.atlas_name,
            atlas_version=layer_map.atlas_version,
            selected_layer_indices=selected_indices,
            output_mode=output_mode,
            output_shape=tuple(int(value) for value in volume_shape),
        )

    empty_stats = AllenLayerAggregateStats(0, 0, 0, 0, 0, 0)
    empty_counts = tuple(0 for _label in selected_labels)
    if file_filter is None:
        volume = (
            np.zeros(volume_shape, dtype=np.float32)
            if color_mode == FLATMAP_HEATMAP_COLOR_SINGLE
            else None
        )
        return AllenLayerHeatmapVolumeResult(
            color_mode=color_mode,
            volume=volume,
            grouped_volumes=(),
            summary=_summary(
                empty_stats,
                nonzero_voxels=0,
                layer_node_counts=empty_counts,
            ),
            stats=empty_stats,
            volume_shape=volume_shape,
        )

    file_filter_sql, params = file_filter
    _emit_flatmap_progress(
        progress_callback,
        "Counting Allen layer nodes...",
        1,
        progress_total,
    )
    stats = _query_allen_layer_stats(
        conn,
        source_sql,
        expressions,
        file_filter_sql,
        params,
    )
    _emit_flatmap_progress(
        progress_callback,
        "Binning Allen layer heatmap in DuckDB...",
        2,
        progress_total,
    )
    where_sql = _combine_where(
        expressions["render_where"],
        file_filter_sql,
    )
    include_file_id = color_mode != FLATMAP_HEATMAP_COLOR_SINGLE
    counts = _query_allen_layer_bin_counts(
        conn,
        source_sql,
        expressions,
        where_sql,
        params,
        include_file_id=include_file_id,
        include_plane_bin=(output_mode == ALLEN_LAYER_OUTPUT_STACK),
    )
    layer_node_counts_array = np.zeros(canonical_plane_count, dtype=np.int64)
    if not counts.empty:
        grouped_counts = counts.groupby("allen_layer_index")["node_count"].sum()
        for layer_index, node_count in grouped_counts.items():
            layer_node_counts_array[int(layer_index)] = int(node_count)
    layer_node_counts = tuple(
        int(layer_node_counts_array[index]) for index in selected_indices
    )

    if color_mode == FLATMAP_HEATMAP_COLOR_SINGLE:
        volume = np.zeros(volume_shape, dtype=np.float32)
        _scatter_bin_counts(volume, counts)
        summary = _summary(
            stats,
            nonzero_voxels=int(np.count_nonzero(volume)),
            layer_node_counts=layer_node_counts,
        )
        return AllenLayerHeatmapVolumeResult(
            color_mode=color_mode,
            volume=volume,
            grouped_volumes=(),
            summary=summary,
            stats=stats,
            volume_shape=volume_shape,
        )

    groups, combined = _build_grouped_flatmap_volumes(
        counts,
        volume_shape,
        color_mode=color_mode,
        cluster_map=cluster_map,
    )
    summary = _summary(
        stats,
        nonzero_voxels=int(np.count_nonzero(combined)),
        layer_node_counts=layer_node_counts,
    )
    return AllenLayerHeatmapVolumeResult(
        color_mode=color_mode,
        volume=None,
        grouped_volumes=tuple(groups),
        summary=summary,
        stats=stats,
        volume_shape=volume_shape,
    )


def compute_flatmap_bounds_from_parquet(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    *,
    style_key: str,
    file_ids: list[object] | None = None,
) -> dict[str, tuple[float, float]]:
    """Derive fallback flatmap/depth bounds directly from Parquet columns.

    Mirrors :func:`_projected_nodes_bounds`: x/y bounds span flatmap-valid nodes
    and the depth range spans flatmap-and-depth-valid, non-negative depths.  Used
    only when a v3 Parquet does not carry canonical bounds metadata.
    """
    suffix = _style_suffix(style_key)
    source_sql = f"read_parquet('{_duckdb_source_path(parquet_path)}')"
    column_names = _duckdb_column_names(conn, source_sql)
    x_column = f"x_flat_{suffix}"
    y_column = f"y_flat_{suffix}"
    for required in (x_column, y_column, "depth_um", "file_id"):
        if required not in column_names:
            raise ValueError(
                f"Parquet is missing required column {required!r} for flatmap "
                "bounds computation."
            )
    x_ref = _sql_identifier(x_column)
    y_ref = _sql_identifier(y_column)
    depth_ref = _sql_identifier("depth_um")

    if f"flatmap_{suffix}_valid" in column_names:
        flatmap_valid = f"COALESCE({_sql_identifier(f'flatmap_{suffix}_valid')}, FALSE)"
    else:
        flatmap_valid = (
            f"({x_ref} IS NOT NULL AND isfinite({x_ref}) "
            f"AND {y_ref} IS NOT NULL AND isfinite({y_ref}))"
        )
    if "depth_valid" in column_names:
        depth_valid = f"COALESCE({_sql_identifier('depth_valid')}, FALSE)"
    else:
        depth_valid = (
            f"({depth_ref} IS NOT NULL AND isfinite({depth_ref}) "
            f"AND {depth_ref} >= 0.0)"
        )

    flatmap_mask = f"({flatmap_valid}) AND isfinite({x_ref}) AND isfinite({y_ref})"
    depth_mask = (
        f"({flatmap_valid}) AND ({depth_valid}) AND isfinite({depth_ref}) "
        f"AND {depth_ref} >= 0.0"
    )

    file_filter = _file_id_filter(file_ids)
    if file_filter is None:
        raise ValueError("No file IDs selected for flatmap bounds computation.")
    file_filter_sql, params = file_filter
    where_sql = f"WHERE {file_filter_sql}" if file_filter_sql else ""

    query = f"""
        SELECT
            MIN({x_ref}) FILTER (WHERE {flatmap_mask}) AS x_min,
            MAX({x_ref}) FILTER (WHERE {flatmap_mask}) AS x_max,
            MIN({y_ref}) FILTER (WHERE {flatmap_mask}) AS y_min,
            MAX({y_ref}) FILTER (WHERE {flatmap_mask}) AS y_max,
            MIN({depth_ref}) FILTER (WHERE {depth_mask}) AS depth_min,
            MAX({depth_ref}) FILTER (WHERE {depth_mask}) AS depth_max
        FROM {source_sql}
        {where_sql}
    """
    row = (
        conn.execute(query, params).fetchone()
        if params
        else conn.execute(query).fetchone()
    )
    if row is None or row[0] is None or row[2] is None:
        raise ValueError(
            "Parquet does not contain valid flatmap coordinates for the "
            f"{style_key} style."
        )
    if row[4] is None or row[5] is None:
        raise ValueError(
            "Parquet does not contain valid non-negative depths for the "
            f"{style_key} style."
        )
    return {
        "x_bounds": _nondegenerate_bounds(float(row[0]), float(row[1])),
        "y_bounds": _nondegenerate_bounds(float(row[2]), float(row[3])),
        "depth_range_um": _nondegenerate_bounds(float(row[4]), float(row[5])),
    }


def _emit_flatmap_progress(
    progress_callback: Callable[[str, int, int], None] | None,
    message: str,
    current: int,
    total: int,
) -> None:
    if progress_callback is not None:
        progress_callback(message, current, total)
