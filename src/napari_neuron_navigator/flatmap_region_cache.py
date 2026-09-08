"""Persistent, memory-mappable BrainGlobe regions in flatmap space.

The cache in this module is deliberately independent of napari and of a live
BrainGlobe atlas.  Cache construction projects annotation voxels once; opening
and materialising selections only reads ``.npy`` arrays referenced by the root
manifest.
"""

from __future__ import annotations

from contextlib import ExitStack, closing, contextmanager
import hashlib
import heapq
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .flatmap_heatmap import (
    ALLEN_LAYER_OUTPUT_STACK,
    DEFAULT_FLATMAP_DEPTH_BIN_UM,
    DEFAULT_FLATMAP_Y_BINS,
    DEFAULT_LOOKUP_STATS_CHUNK_VOXELS,
    _bin_flat_values,
    _depth_bin_count,
    _flatmap_valid_mask,
    _resolve_allen_layer_render_options,
    _spatial_chunk_slices,
    compute_flatmap_lookup_stats,
    resolve_flatmap_bin_counts,
)
from .flatmap_loader import (
    FlatmapLookupLoadCancelledError,
    load_flatmap_volume_set,
)
from .isocortex_layers import AllenIsocortexLayerMap

REGION_CACHE_MANIFEST_FILENAME = "flatmap-region-cache.json"
REGION_CACHE_FORMAT = "napari_neuron_navigator.flatmap_region_cache"
LEGACY_REGION_CACHE_FORMAT = "napari_swc_viewer.flatmap_region_cache"
REGION_CACHE_FORMATS = frozenset({REGION_CACHE_FORMAT, LEGACY_REGION_CACHE_FORMAT})
#: Version 2 replaced the single ``xy_bins`` grid key with per-axis ``x_bins`` /
#: ``y_bins``, because the flat map ``x`` axis spans both hemispheres and ``y``
#: only one, so equal counts made every bin twice as wide as it was tall.  A
#: version-1 cache cannot be reinterpreted -- its arrays were binned on the old
#: grid -- so it must be rebuilt.
REGION_CACHE_FORMAT_VERSION = 2

_OCCUPANCY_ALGORITHM = "source-voxel-counts-v1"
_COLLISION_ALGORITHM = "majority-count-then-smaller-region-id-v1"
_SURFACE_ALGORITHM = "voxel-exposed-face-shell-v1"
_OUTLINE_ALGORITHM = "depth-slice-four-neighbour-perimeter-v1"
_DEPTH_ALGORITHM = "original-then-mirror-invalid-depth-v1"

_STYLE_ALIASES = {
    "shaped": "shaped",
    "both_shaped": "shaped",
    "bilateral_shaped": "shaped",
    "square": "square",
    "both_square": "square",
    "bilateral_square": "square",
}

_ATLAS_RESOLUTION_SUFFIX = re.compile(r"_(?:10|25|50|100)um(?=$|_)", re.IGNORECASE)

logger = logging.getLogger(__name__)


class RegionCacheError(RuntimeError):
    """Base error for flatmap region-cache operations."""


class RegionCacheValidationError(RegionCacheError):
    """Raised when a cache manifest or one of its arrays is invalid."""


class RegionCacheCancelled(RegionCacheError):
    """Raised when cache construction is cancelled."""


def _close_memmap(array: np.ndarray, *, flush: bool = False) -> None:
    """Close one NumPy memmap without relying on garbage collection."""
    if isinstance(array, np.memmap):
        mapping = getattr(array, "_mmap", None)
        try:
            if flush:
                array.flush()
        finally:
            if mapping is not None and not mapping.closed:
                mapping.close()


@contextmanager
def _open_npy_memmap(path: Path):
    """Open one temporary NPY mapping and always release its Windows handle."""
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        yield array
    finally:
        _close_memmap(array)


@contextmanager
def _create_npy_memmap(
    path: Path,
    *,
    dtype: np.dtype | str,
    shape: tuple[int, ...],
):
    """Create a writable NPY mapping and close it on success or failure."""
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )
    try:
        yield array
    except BaseException:
        try:
            _close_memmap(array)
        except Exception:
            logger.warning(
                "Failed to close temporary cache array %s", path, exc_info=True
            )
        raise
    else:
        _close_memmap(array, flush=True)


@dataclass(frozen=True)
class CachedArraySpec:
    """Manifest description of one memory-mappable array."""

    path: str
    dtype: str
    shape: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: object, *, name: str) -> "CachedArraySpec":
        if not isinstance(value, Mapping):
            raise RegionCacheValidationError(
                f"Cache array specification for {name!r} must be an object."
            )
        path = value.get("path")
        dtype = value.get("dtype")
        shape = value.get("shape")
        if not isinstance(path, str) or not path:
            raise RegionCacheValidationError(
                f"Cache array {name!r} has an invalid path."
            )
        if not isinstance(dtype, str) or not dtype:
            raise RegionCacheValidationError(
                f"Cache array {name!r} has an invalid dtype."
            )
        if not isinstance(shape, list) or not all(
            isinstance(size, int) and size >= 0 for size in shape
        ):
            raise RegionCacheValidationError(
                f"Cache array {name!r} has an invalid shape."
            )
        return cls(path=path, dtype=dtype, shape=tuple(shape))


@dataclass(frozen=True)
class CachedRegionSurface:
    """One directly selectable region's cached closed surface."""

    region_id: int
    vertices: np.ndarray
    faces: np.ndarray
    footprint_index: int
    component_count: int


@dataclass(frozen=True)
class CachedRegionOutlines:
    """One directly selectable region's cached per-depth outline vectors."""

    region_id: int
    vectors: np.ndarray
    footprint_index: int


@dataclass(frozen=True)
class CachedRegionSelectionSummary:
    """Counts for a selection materialised from sparse cache occupancy."""

    selected_region_count: int
    represented_region_count: int
    labeled_bins: int
    collision_bins: int
    source_voxel_count: int
    output_shape: tuple[int, int, int]

    @property
    def labeled_voxels(self) -> int:
        """Compatibility alias for on-demand region-label summaries."""
        return self.labeled_bins

    @property
    def collision_voxels(self) -> int:
        """Compatibility alias for on-demand region-label summaries."""
        return self.collision_bins

    @property
    def valid_source_voxels(self) -> int:
        """Compatibility alias for selected source-voxel counts."""
        return self.source_voxel_count

    def to_dict(self) -> dict[str, int | list[int]]:
        return {
            "selected_region_count": int(self.selected_region_count),
            "represented_region_count": int(self.represented_region_count),
            "labeled_bins": int(self.labeled_bins),
            "collision_bins": int(self.collision_bins),
            "source_voxel_count": int(self.source_voxel_count),
            "output_shape": [int(size) for size in self.output_shape],
        }


@dataclass(frozen=True)
class CachedRegionSelection:
    """Labels and optional geometry materialised for a region selection."""

    labels: np.ndarray
    selected_region_ids: tuple[int, ...]
    represented_region_ids: tuple[int, ...]
    surfaces: tuple[CachedRegionSurface, ...]
    outlines: tuple[CachedRegionOutlines, ...]
    summary: CachedRegionSelectionSummary
    grid_spec: Mapping[str, Any]
    style: str
    profile_id: str


@dataclass(frozen=True)
class CachedAllenLayerRegionSelectionSummary:
    """Counts for a cached Allen-layer stack or projection."""

    selected_region_count: int
    layer_mapped_region_count: int
    represented_region_count: int
    labeled_bins: int
    collision_bins: int
    source_voxel_count: int
    output_shape: tuple[int, ...]
    layer_labels: tuple[str, ...]
    selected_layer_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    output_mode: str = ALLEN_LAYER_OUTPUT_STACK
    excluded_unselected_layer_region_count: int = 0

    @property
    def excluded_non_layer_region_count(self) -> int:
        """Return selected IDs that do not map to an Allen layer plane."""
        return max(
            0,
            self.selected_region_count
            - self.layer_mapped_region_count
            - self.excluded_unselected_layer_region_count,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_region_count": int(self.selected_region_count),
            "layer_mapped_region_count": int(self.layer_mapped_region_count),
            "represented_region_count": int(self.represented_region_count),
            "excluded_non_layer_region_count": int(
                self.excluded_non_layer_region_count
            ),
            "excluded_unselected_layer_region_count": int(
                self.excluded_unselected_layer_region_count
            ),
            "labeled_bins": int(self.labeled_bins),
            "collision_bins": int(self.collision_bins),
            "source_voxel_count": int(self.source_voxel_count),
            "output_shape": [int(size) for size in self.output_shape],
            "layer_labels": list(self.layer_labels),
            "selected_layer_labels": list(self.layer_labels),
            "selected_layer_indices": [
                int(value) for value in self.selected_layer_indices
            ],
            "output_mode": self.output_mode,
            "allen_layer_output_mode": self.output_mode,
        }


@dataclass(frozen=True)
class CachedAllenLayerRegionSelection:
    """Cache-backed atlas labels collapsed into categorical Allen planes."""

    labels: np.ndarray
    selected_region_ids: tuple[int, ...]
    layer_mapped_region_ids: tuple[int, ...]
    represented_region_ids: tuple[int, ...]
    layer_labels: tuple[str, ...]
    summary: CachedAllenLayerRegionSelectionSummary
    grid_spec: Mapping[str, Any]
    style: str
    profile_id: str


@dataclass(frozen=True)
class CachedFlatRegionOutlines:
    """One selected region's depth-collapsed 2D perimeter, derived on read.

    ``vectors`` is a napari Vectors array of shape ``(N, 2, 2)`` holding
    ``[start(y, x), direction(dy, dx)]`` in flatmap bin-index units.  The cache
    stores only per-depth-slice perimeters, so this geometry is built at read
    time from the union of ``union_region_ids`` occupancy.
    """

    region_id: int
    vectors: np.ndarray
    union_region_ids: tuple[int, ...]
    represented_region_ids: tuple[int, ...]
    planar_bin_count: int
    source_voxel_count: int


@dataclass(frozen=True)
class CachedFlatRegionSelectionSummary:
    """Counts for a region selection collapsed into one flatmap plane."""

    selected_region_count: int
    direct_region_count: int
    represented_region_count: int
    represented_source_region_count: int
    labeled_bins: int
    collision_bins: int
    source_voxel_count: int
    output_shape: tuple[int, int]

    @property
    def labeled_voxels(self) -> int:
        """Compatibility alias for on-demand region-label summaries."""
        return self.labeled_bins

    @property
    def collision_voxels(self) -> int:
        """Compatibility alias for on-demand region-label summaries."""
        return self.collision_bins

    @property
    def valid_source_voxels(self) -> int:
        """Compatibility alias for selected source-voxel counts."""
        return self.source_voxel_count

    def to_dict(self) -> dict[str, int | list[int]]:
        return {
            "selected_region_count": int(self.selected_region_count),
            "direct_region_count": int(self.direct_region_count),
            "represented_region_count": int(self.represented_region_count),
            "represented_source_region_count": int(
                self.represented_source_region_count
            ),
            "labeled_bins": int(self.labeled_bins),
            "collision_bins": int(self.collision_bins),
            "source_voxel_count": int(self.source_voxel_count),
            "output_shape": [int(size) for size in self.output_shape],
        }


@dataclass(frozen=True)
class CachedFlatRegionSelection:
    """Cache-backed atlas labels collapsed into one depth-free flatmap plane.

    ``represented_region_ids`` are the directly selected regions whose groups
    contributed occupancy.  ``represented_source_region_ids`` are both the
    cached occupancy regions that contributed counts and the label values
    written into ``labels``.  Retaining those source IDs lets a 2D Labels
    colormap apply descendant appearance overrides after depth is collapsed.
    """

    labels: np.ndarray
    selected_region_ids: tuple[int, ...]
    direct_region_ids: tuple[int, ...]
    represented_region_ids: tuple[int, ...]
    represented_source_region_ids: tuple[int, ...]
    outlines: tuple[CachedFlatRegionOutlines, ...]
    summary: CachedFlatRegionSelectionSummary
    grid_spec: Mapping[str, Any]
    style: str
    profile_id: str


def normalise_atlas_family(atlas_name: str) -> str:
    """Return an atlas identity that is stable across voxel resolutions."""
    return _ATLAS_RESOLUTION_SUFFIX.sub("", str(atlas_name).strip())


def structure_catalog_id(
    atlas_structures: Mapping[object, Mapping[str, Any]],
) -> str:
    """Return a stable ID for the selection/color-relevant structure catalog."""
    records_by_id: dict[int, dict[str, Any]] = {}
    for key, structure in atlas_structures.items():
        try:
            region_id = int(structure.get("id", key))
        except (TypeError, ValueError):
            continue
        raw_path = structure.get("structure_id_path") or structure.get(
            "structure_id_path_ids"
        )
        if isinstance(raw_path, str):
            path_ids: object = [
                int(part) for part in raw_path.strip("/").split("/") if part
            ]
        elif isinstance(raw_path, Iterable):
            path_ids = [int(part) for part in raw_path]
        else:
            path_ids = [region_id]
        color = structure.get("rgb_triplet", structure.get("color_hex_triplet"))
        records_by_id[region_id] = {
            "id": region_id,
            "acronym": str(structure.get("acronym", "")),
            "name": str(structure.get("name", "")),
            "structure_id_path": path_ids,
            "color": _json_safe(color),
        }
    records = [records_by_id[key] for key in sorted(records_by_id)]
    return f"bgsc1-{_canonical_digest({'structures': records})}"


def _normalise_style(style: str) -> str:
    normalised = _STYLE_ALIASES.get(str(style).strip().lower())
    if normalised is None:
        raise ValueError(
            f"Unknown flatmap cache style {style!r}; expected shaped or square."
        )
    return normalised


def _normalise_region_ids(region_ids: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in region_ids if int(value) > 0}))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cancel_requested(cancel_callback: Callable[[], bool] | object | None) -> bool:
    if cancel_callback is None:
        return False
    is_set = getattr(cancel_callback, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancel_callback())


def _check_cancel(cancel_callback: Callable[[], bool] | object | None) -> None:
    if _cancel_requested(cancel_callback):
        raise RegionCacheCancelled("Flatmap region-cache construction was cancelled.")


def _progress(
    callback: Callable[[str, int, int], None] | None,
    message: str,
    current: int,
    total: int,
) -> None:
    if callback is not None:
        callback(str(message), int(current), int(total))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise RegionCacheValidationError(
            f"Could not read region-cache manifest {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RegionCacheValidationError("Region-cache manifest must be an object.")
    return value


def _empty_manifest() -> dict[str, Any]:
    return {
        "format": REGION_CACHE_FORMAT,
        "format_version": REGION_CACHE_FORMAT_VERSION,
        "profiles": {},
    }


def _validate_root_manifest(value: Mapping[str, Any]) -> None:
    if value.get("format") not in REGION_CACHE_FORMATS:
        raise RegionCacheValidationError(
            "Not a napari-neuron-navigator flatmap region cache: unexpected format."
        )
    found_version = value.get("format_version")
    if found_version != REGION_CACHE_FORMAT_VERSION:
        raise RegionCacheValidationError(
            "This flatmap region cache was built by an older version of the "
            f"plugin (format version {found_version!r}, this version needs "
            f"{REGION_CACHE_FORMAT_VERSION}). Its voxel grid used one bin count "
            "for both flat map axes, which made every bin about twice as wide "
            "as it was tall. Rebuild the cache to use square bins."
        )
    profiles = value.get("profiles")
    if not isinstance(profiles, Mapping):
        raise RegionCacheValidationError("Region-cache profiles must be an object.")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_json_safe(value), stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _array_spec_from_shape(
    path: Path,
    *,
    dtype: np.dtype | str,
    shape: Sequence[int],
    base: Path,
) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "dtype": np.dtype(dtype).name,
        "shape": [int(size) for size in shape],
    }


def _array_spec(path: Path, array: np.ndarray, *, base: Path) -> dict[str, Any]:
    return _array_spec_from_shape(
        path,
        dtype=array.dtype,
        shape=array.shape,
        base=base,
    )


def _save_array(path: Path, values: np.ndarray, *, dtype: np.dtype | str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    np.save(path, array, allow_pickle=False)
    return array


def _remove_tree(
    path: Path,
    *,
    purpose: str,
    suppress_errors: bool = False,
) -> bool:
    """Remove a cache tree without letting secondary cleanup hide a failure."""
    if not path.is_dir():
        return True
    try:
        shutil.rmtree(path)
    except OSError:
        if not suppress_errors:
            raise
        logger.warning(
            "Could not remove %s at %s; it may be deleted after napari closes.",
            purpose,
            path,
            exc_info=True,
        )
        return False
    return True


def _safe_array_path(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    base_resolved = base.resolve()
    if path != base_resolved and base_resolved not in path.parents:
        raise RegionCacheValidationError(
            f"Cache array path escapes its profile directory: {relative!r}."
        )
    return path


class FlatmapRegionStyleCache:
    """Memory-mapped arrays and grid metadata for one flatmap style."""

    def __init__(
        self, profile: "FlatmapRegionCacheProfile", style: str, data: Mapping[str, Any]
    ):
        self.profile = profile
        self.style = _normalise_style(style)
        self._data = dict(data)
        grid = data.get("grid")
        arrays = data.get("arrays")
        if not isinstance(grid, Mapping):
            raise RegionCacheValidationError(
                f"Profile {profile.profile_id} style {style} has no valid grid."
            )
        if not isinstance(arrays, Mapping):
            raise RegionCacheValidationError(
                f"Profile {profile.profile_id} style {style} has no array map."
            )
        self.grid_spec: Mapping[str, Any] = dict(grid)
        self._specs = {
            str(name): CachedArraySpec.from_dict(spec, name=str(name))
            for name, spec in arrays.items()
        }
        self._arrays: dict[str, np.ndarray] = {}
        self._closed = False

    @property
    def directory(self) -> Path:
        """Directory containing this style's arrays."""
        directory = _safe_array_path(self.profile.directory, self.style)
        if not directory.is_dir():
            raise RegionCacheValidationError(
                f"Cache style directory is missing: {directory}"
            )
        return directory

    @property
    def output_shape(self) -> tuple[int, int, int]:
        shape = self.grid_spec.get("output_shape")
        if not isinstance(shape, list) or len(shape) != 3:
            raise RegionCacheValidationError(
                f"Style {self.style} has an invalid output shape."
            )
        return tuple(int(size) for size in shape)

    @property
    def lookup_grid_spec(self) -> object | None:
        """Canonical :class:`FlatmapGridSpec` when stored by the builder."""
        payload = self.grid_spec.get("lookup_grid_spec")
        if not isinstance(payload, Mapping):
            return None
        try:
            from .flatmap_profiles import FlatmapGridSpec

            return FlatmapGridSpec.from_dict(payload)
        except (ImportError, KeyError, TypeError, ValueError):
            return dict(payload)

    def array(self, name: str) -> np.ndarray:
        if self._closed:
            raise RegionCacheError(
                f"Flatmap region-cache style {self.style!r} is closed."
            )
        if name in self._arrays:
            return self._arrays[name]
        try:
            spec = self._specs[name]
        except KeyError as exc:
            raise RegionCacheValidationError(
                f"Style {self.style} is missing required array {name!r}."
            ) from exc
        path = _safe_array_path(self.directory, spec.path)
        if not path.is_file():
            raise RegionCacheValidationError(f"Cache array is missing: {path}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RegionCacheValidationError(
                f"Could not open cache array {path}: {exc}"
            ) from exc
        try:
            try:
                expected_dtype = np.dtype(spec.dtype)
            except (TypeError, ValueError) as exc:
                raise RegionCacheValidationError(
                    f"Cache array {name!r} declares invalid dtype {spec.dtype!r}."
                ) from exc
            if np.dtype(array.dtype).name != expected_dtype.name:
                raise RegionCacheValidationError(
                    f"Cache array {name!r} has dtype {array.dtype}, "
                    f"expected {spec.dtype}."
                )
            if tuple(array.shape) != spec.shape:
                raise RegionCacheValidationError(
                    f"Cache array {name!r} has shape {array.shape}, "
                    f"expected {spec.shape}."
                )
        except BaseException:
            try:
                _close_memmap(array)
            except Exception:
                logger.warning(
                    "Failed to close rejected cache array %s.", path, exc_info=True
                )
            raise
        self._arrays[name] = array
        return array

    def close(self) -> None:
        """Release all memory-mapped arrays owned by this style cache."""
        if self._closed:
            return
        _close_style_mmaps(self)
        self._closed = True

    def _validate(self) -> None:
        required = {
            "occupancy_region_ids",
            "occupancy_region_offsets",
            "occupancy_linear_bins",
            "occupancy_source_voxel_counts",
            "geometry_region_ids",
            "geometry_region_footprint_indices",
            "geometry_component_counts",
            "surface_vertex_offsets",
            "surface_face_offsets",
            "surface_vertices",
            "surface_faces",
            "outline_offsets",
            "outline_vectors",
        }
        missing = sorted(required.difference(self._specs))
        if missing:
            raise RegionCacheValidationError(
                f"Style {self.style} is missing required array(s): {missing}."
            )
        contract_dtypes = {
            "occupancy_region_ids": np.dtype(np.int32),
            "occupancy_region_offsets": np.dtype(np.int64),
            "occupancy_linear_bins": np.dtype(np.int64),
            "occupancy_source_voxel_counts": np.dtype(np.int64),
            "geometry_region_ids": np.dtype(np.int32),
            "geometry_region_footprint_indices": np.dtype(np.int32),
            "geometry_component_counts": np.dtype(np.int32),
            "surface_vertex_offsets": np.dtype(np.int64),
            "surface_face_offsets": np.dtype(np.int64),
            "surface_vertices": np.dtype(np.float32),
            "surface_faces": np.dtype(np.int32),
            "outline_offsets": np.dtype(np.int64),
            "outline_vectors": np.dtype(np.float32),
        }
        for name, expected_dtype in contract_dtypes.items():
            try:
                declared_dtype = np.dtype(self._specs[name].dtype)
            except (TypeError, ValueError) as exc:
                raise RegionCacheValidationError(
                    f"Cache array {name!r} declares an invalid dtype."
                ) from exc
            if declared_dtype != expected_dtype:
                raise RegionCacheValidationError(
                    f"Cache array {name!r} must use {expected_dtype.name}; "
                    f"got {declared_dtype.name}."
                )
        shape = self.output_shape
        if any(size <= 0 for size in shape):
            raise RegionCacheValidationError(
                f"Style {self.style} output shape must be positive; got {shape}."
            )
        if self.grid_spec.get("coordinate_order") != ["depth", "y", "x"]:
            raise RegionCacheValidationError(
                f"Style {self.style} has an unsupported coordinate order."
            )
        try:
            y_bins = int(self.grid_spec["y_bins"])
            x_bins = int(self.grid_spec["x_bins"])
            depth_bins = int(self.grid_spec["depth_bins"])
            depth_bin_um = float(self.grid_spec["depth_bin_um"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RegionCacheValidationError(
                f"Style {self.style} has invalid fixed-grid dimensions."
            ) from exc
        # Internal consistency only -- never a check that x_bins conforms to the
        # current square-bin policy.  The stored count came from bounds that have
        # since been through a JSON float round trip, so a policy check could
        # make a perfectly good cache unopenable at a rounding tie.
        if (
            y_bins <= 0
            or x_bins <= 0
            or depth_bins <= 0
            or not np.isfinite(depth_bin_um)
            or depth_bin_um <= 0
            or shape != (depth_bins, y_bins, x_bins)
        ):
            raise RegionCacheValidationError(
                f"Style {self.style} fixed-grid dimensions do not match its output shape."
            )
        for bounds_name in ("x_bounds", "y_bounds", "depth_bounds_um"):
            raw_bounds = self.grid_spec.get(bounds_name)
            try:
                bounds = tuple(float(value) for value in raw_bounds)
            except (TypeError, ValueError) as exc:
                raise RegionCacheValidationError(
                    f"Style {self.style} has invalid {bounds_name}."
                ) from exc
            if (
                len(bounds) != 2
                or not all(np.isfinite(value) for value in bounds)
                or bounds[1] <= bounds[0]
            ):
                raise RegionCacheValidationError(
                    f"Style {self.style} has invalid {bounds_name}."
                )
        if bool(self.grid_spec.get("includes_depth_minus_one_plane", True)):
            raise RegionCacheValidationError(
                f"Style {self.style} unexpectedly includes a depth -1 sentinel plane."
            )

        region_ids = self.array("occupancy_region_ids")
        offsets = self.array("occupancy_region_offsets")
        bins = self.array("occupancy_linear_bins")
        counts = self.array("occupancy_source_voxel_counts")
        if region_ids.ndim != 1 or offsets.shape != (len(region_ids) + 1,):
            raise RegionCacheValidationError("Invalid occupancy region offsets.")
        if bins.ndim != 1 or counts.shape != bins.shape:
            raise RegionCacheValidationError("Invalid occupancy bins/counts shapes.")
        if len(region_ids) and (
            np.any(region_ids <= 0) or np.any(region_ids[1:] <= region_ids[:-1])
        ):
            raise RegionCacheValidationError(
                "Occupancy region IDs must be positive and strictly sorted."
            )
        if offsets[0] != 0 or offsets[-1] != len(bins) or np.any(np.diff(offsets) < 0):
            raise RegionCacheValidationError("Occupancy offsets are not monotonic.")
        if len(bins) and (
            np.any(bins < 0)
            or np.any(bins >= int(np.prod(shape)))
            or np.any(counts <= 0)
        ):
            raise RegionCacheValidationError("Occupancy bins or counts are invalid.")
        for index in range(len(region_ids)):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            if stop - start > 1 and np.any(
                bins[start + 1 : stop] <= bins[start : stop - 1]
            ):
                raise RegionCacheValidationError(
                    "Occupancy bins must be strictly sorted within each region."
                )

        geometry_ids = self.array("geometry_region_ids")
        mapping = self.array("geometry_region_footprint_indices")
        components = self.array("geometry_component_counts")
        vertex_offsets = self.array("surface_vertex_offsets")
        face_offsets = self.array("surface_face_offsets")
        vertices = self.array("surface_vertices")
        faces = self.array("surface_faces")
        outline_offsets = self.array("outline_offsets")
        vectors = self.array("outline_vectors")
        if geometry_ids.ndim != 1 or mapping.shape != geometry_ids.shape:
            raise RegionCacheValidationError("Invalid geometry region mapping.")
        if len(geometry_ids) and (
            np.any(geometry_ids <= 0) or np.any(geometry_ids[1:] <= geometry_ids[:-1])
        ):
            raise RegionCacheValidationError(
                "Geometry region IDs must be positive and strictly sorted."
            )
        footprint_count = int(len(vertex_offsets) - 1)
        if footprint_count < 0 or len(face_offsets) != footprint_count + 1:
            raise RegionCacheValidationError("Invalid surface offsets.")
        if len(outline_offsets) != footprint_count + 1 or components.shape != (
            footprint_count,
        ):
            raise RegionCacheValidationError("Invalid outline/component offsets.")
        if len(mapping) and (np.any(mapping < 0) or np.any(mapping >= footprint_count)):
            raise RegionCacheValidationError(
                "Geometry footprint mapping is out of range."
            )
        if (
            vertex_offsets[0] != 0
            or vertex_offsets[-1] != len(vertices)
            or np.any(np.diff(vertex_offsets) < 0)
            or face_offsets[0] != 0
            or face_offsets[-1] != len(faces)
            or np.any(np.diff(face_offsets) < 0)
            or outline_offsets[0] != 0
            or outline_offsets[-1] != len(vectors)
            or np.any(np.diff(outline_offsets) < 0)
        ):
            raise RegionCacheValidationError("Geometry offsets are not monotonic.")
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not np.all(np.isfinite(vertices))
        ):
            raise RegionCacheValidationError(
                "Surface vertices must be finite (N, 3) values."
            )
        if faces.ndim != 2 or faces.shape[1:] != (3,):
            raise RegionCacheValidationError("Surface faces must have shape (N, 3).")
        if (
            vectors.ndim != 3
            or vectors.shape[1:] != (2, 3)
            or not np.all(np.isfinite(vectors))
        ):
            raise RegionCacheValidationError(
                "Outline vectors must be finite napari (N, 2, 3) values."
            )
        if len(components) and np.any(components < 0):
            raise RegionCacheValidationError("Geometry component counts are invalid.")
        for index in range(footprint_count):
            vertex_count = int(vertex_offsets[index + 1] - vertex_offsets[index])
            start, stop = int(face_offsets[index]), int(face_offsets[index + 1])
            local_faces = faces[start:stop]
            if len(local_faces) and (
                np.any(local_faces < 0) or np.any(local_faces >= vertex_count)
            ):
                raise RegionCacheValidationError(
                    f"Surface footprint {index} contains an invalid local face index."
                )

        count_metadata = self._data.get("counts")
        if not isinstance(count_metadata, Mapping):
            raise RegionCacheValidationError(
                f"Style {self.style} has no validation counts."
            )
        derived_counts = {
            "occupied_region_count": len(region_ids),
            "region_bin_pair_count": len(bins),
            "source_voxel_count": int(np.asarray(counts, dtype=np.int64).sum()),
            "direct_region_count": len(geometry_ids),
            "unique_footprint_count": footprint_count,
            "component_count": int(np.asarray(components, dtype=np.int64).sum()),
            "surface_vertex_count": len(vertices),
            "surface_face_count": len(faces),
            "outline_segment_count": len(vectors),
        }
        for name, actual in derived_counts.items():
            try:
                recorded = int(count_metadata[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise RegionCacheValidationError(
                    f"Style {self.style} has invalid validation count {name!r}."
                ) from exc
            if recorded != int(actual):
                raise RegionCacheValidationError(
                    f"Style {self.style} validation count {name!r} differs "
                    f"(manifest={recorded}, arrays={int(actual)})."
                )

    def occupancy_slice(self, region_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        region_ids = self.array("occupancy_region_ids")
        index = int(np.searchsorted(region_ids, int(region_id)))
        if index >= len(region_ids) or int(region_ids[index]) != int(region_id):
            return None
        offsets = self.array("occupancy_region_offsets")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        return (
            self.array("occupancy_linear_bins")[start:stop],
            self.array("occupancy_source_voxel_counts")[start:stop],
        )

    def geometry_footprint_index(self, region_id: int) -> int | None:
        region_ids = self.array("geometry_region_ids")
        index = int(np.searchsorted(region_ids, int(region_id)))
        if index >= len(region_ids) or int(region_ids[index]) != int(region_id):
            return None
        return int(self.array("geometry_region_footprint_indices")[index])

    def surface(self, region_id: int) -> CachedRegionSurface | None:
        footprint = self.geometry_footprint_index(region_id)
        if footprint is None:
            return None
        vertex_offsets = self.array("surface_vertex_offsets")
        face_offsets = self.array("surface_face_offsets")
        vertex_start, vertex_stop = (
            int(vertex_offsets[footprint]),
            int(vertex_offsets[footprint + 1]),
        )
        face_start, face_stop = (
            int(face_offsets[footprint]),
            int(face_offsets[footprint + 1]),
        )
        return CachedRegionSurface(
            region_id=int(region_id),
            vertices=self.array("surface_vertices")[vertex_start:vertex_stop],
            faces=self.array("surface_faces")[face_start:face_stop],
            footprint_index=footprint,
            component_count=int(self.array("geometry_component_counts")[footprint]),
        )

    def outlines(self, region_id: int) -> CachedRegionOutlines | None:
        footprint = self.geometry_footprint_index(region_id)
        if footprint is None:
            return None
        offsets = self.array("outline_offsets")
        start, stop = int(offsets[footprint]), int(offsets[footprint + 1])
        return CachedRegionOutlines(
            region_id=int(region_id),
            vectors=self.array("outline_vectors")[start:stop],
            footprint_index=footprint,
        )


class FlatmapRegionCacheProfile:
    """One atlas/lookup/fixed-grid pair in a region cache."""

    def __init__(self, root: Path, data: Mapping[str, Any]):
        self.root = root
        self._data = dict(data)
        profile_id = data.get("profile_id")
        directory = data.get("directory")
        styles = data.get("styles")
        if not isinstance(profile_id, str) or not profile_id:
            raise RegionCacheValidationError("Cache profile has no valid profile ID.")
        if not isinstance(directory, str) or not directory:
            raise RegionCacheValidationError(f"Profile {profile_id} has no directory.")
        if not isinstance(styles, Mapping):
            raise RegionCacheValidationError(f"Profile {profile_id} has no styles.")
        self.profile_id = profile_id
        self.directory = _safe_array_path(root, directory)
        if not self.directory.is_dir():
            raise RegionCacheValidationError(
                f"Cache profile directory is missing: {self.directory}"
            )
        self.lookup_set_id = str(data.get("lookup_set_id", ""))
        atlas = data.get("atlas")
        if not isinstance(atlas, Mapping):
            raise RegionCacheValidationError(
                f"Profile {profile_id} has no atlas metadata."
            )
        self.atlas = dict(atlas)
        validity = data.get("validity")
        if not isinstance(validity, Mapping):
            raise RegionCacheValidationError(
                f"Profile {profile_id} has no validity metadata."
            )
        self.validity = dict(validity)
        self._styles = {
            _normalise_style(name): FlatmapRegionStyleCache(self, name, style_data)
            for name, style_data in styles.items()
            if isinstance(style_data, Mapping)
        }
        if set(self._styles) != {"shaped", "square"}:
            raise RegionCacheValidationError(
                f"Profile {profile_id} must contain shaped and square styles."
            )
        self._closed = False

    @property
    def styles(self) -> Mapping[str, FlatmapRegionStyleCache]:
        return dict(self._styles)

    def style(self, style: str) -> FlatmapRegionStyleCache:
        return self._styles[_normalise_style(style)]

    def close(self) -> None:
        """Release all memory-mapped arrays owned by this cache profile."""
        if self._closed:
            return
        for style_cache in self._styles.values():
            style_cache.close()
        self._closed = True

    def compatibility_mismatches(
        self,
        *,
        lookup_set_id: str | None = None,
        atlas_name: str | None = None,
        atlas_version: str | None = None,
        atlas_resolution_um: float | Sequence[float] | None = None,
        annotation_shape: Sequence[int] | None = None,
        structure_catalog_id: str | None = None,
        style: str | None = None,
        mirror_depth_fallback: bool | None = None,
        mirror_coord_axis: int | None = None,
    ) -> tuple[str, ...]:
        """Return precise reasons this profile is not compatible."""
        mismatches: list[str] = []
        if lookup_set_id is not None and self.lookup_set_id != str(lookup_set_id):
            mismatches.append(
                f"lookup_set_id differs (cache={self.lookup_set_id!r}, requested={str(lookup_set_id)!r})"
            )
        actual_name = str(self.atlas.get("name", ""))
        actual_family = str(
            self.atlas.get("family", normalise_atlas_family(actual_name))
        )
        requested_family = (
            normalise_atlas_family(atlas_name) if atlas_name is not None else None
        )
        if requested_family is not None and actual_family != requested_family:
            mismatches.append(
                "atlas family differs "
                f"(cache={actual_family!r} from {actual_name!r}, "
                f"requested={requested_family!r} from {str(atlas_name)!r})"
            )
        actual_version = _normalise_atlas_version(self.atlas.get("version"))
        requested_version = _normalise_atlas_version(atlas_version)
        if requested_version is not None and actual_version != requested_version:
            mismatches.append(
                "atlas version differs "
                f"(cache={actual_version!r}, requested={requested_version!r})"
            )
        if annotation_shape is not None:
            expected = tuple(int(size) for size in annotation_shape)
            actual = tuple(int(size) for size in self.atlas.get("annotation_shape", []))
            if actual != expected:
                mismatches.append(
                    f"annotation shape differs (cache={actual}, requested={expected})"
                )
        if atlas_resolution_um is not None:
            expected_resolution = _normalise_resolution(atlas_resolution_um)
            actual_resolution = tuple(
                float(value) for value in self.atlas.get("resolution_um", [])
            )
            if actual_resolution != expected_resolution:
                mismatches.append(
                    "atlas resolution differs "
                    f"(cache={actual_resolution}, requested={expected_resolution})"
                )
        actual_catalog_id = self.atlas.get("structure_catalog_id")
        if structure_catalog_id is not None and str(actual_catalog_id or "") != str(
            structure_catalog_id
        ):
            mismatches.append(
                "structure catalog differs "
                f"(cache={actual_catalog_id!r}, requested={str(structure_catalog_id)!r})"
            )
        if mirror_depth_fallback is not None:
            actual_mirror_fallback = bool(
                self.validity.get("mirror_depth_fallback", False)
            )
            if actual_mirror_fallback != bool(mirror_depth_fallback):
                mismatches.append(
                    "depth mirror fallback differs "
                    f"(cache={actual_mirror_fallback}, "
                    f"requested={bool(mirror_depth_fallback)})"
                )
        if mirror_coord_axis is not None:
            try:
                actual_mirror_axis = int(self.validity["mirror_coord_axis"])
            except (KeyError, TypeError, ValueError):
                actual_mirror_axis = -1
            if actual_mirror_axis != int(mirror_coord_axis):
                mismatches.append(
                    "depth mirror axis differs "
                    f"(cache={actual_mirror_axis}, requested={int(mirror_coord_axis)})"
                )
        if style is not None:
            try:
                normalised = _normalise_style(style)
            except ValueError as exc:
                mismatches.append(str(exc))
            else:
                if normalised not in self._styles:
                    mismatches.append(f"cache has no {normalised} style")
        return tuple(mismatches)

    def _validate(self) -> None:
        if not self.lookup_set_id:
            raise RegionCacheValidationError(
                f"Profile {self.profile_id} has no lookup-set ID."
            )
        for style_cache in self._styles.values():
            style_cache._validate()
        try:
            canonical_profile = {
                "lookup_set_id": self.lookup_set_id,
                "atlas": self.atlas,
                # ``y_bins`` is the only bin count invariant across styles: one
                # profile builds both, and shaped/square resolve to DIFFERENT
                # x counts (491 vs 512 at 256).  The per-style x/y counts are
                # digested anyway through ``style_grids``.
                "y_bins": int(self._data["y_bins"]),
                "depth_bin_um": float(self._data["depth_bin_um"]),
                "style_grids": {
                    "shaped": dict(self._styles["shaped"].grid_spec),
                    "square": dict(self._styles["square"].grid_spec),
                },
                "algorithms": self._data["algorithms"],
                "validity": self.validity,
            }
            expected_profile_id = _canonical_digest(canonical_profile)
        except (KeyError, TypeError, ValueError) as exc:
            raise RegionCacheValidationError(
                f"Profile {self.profile_id} has invalid identity metadata."
            ) from exc
        if expected_profile_id != self.profile_id:
            raise RegionCacheValidationError(
                f"Profile identity metadata does not match {self.profile_id}."
            )


class FlatmapRegionCache:
    """An opened root manifest containing one or more grid profiles."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]):
        self.root = root
        self.manifest = dict(manifest)
        profiles = manifest.get("profiles", {})
        self._profiles = {
            str(profile_id): FlatmapRegionCacheProfile(root, data)
            for profile_id, data in profiles.items()
            if isinstance(data, Mapping)
        }
        if len(self._profiles) != len(profiles):
            raise RegionCacheValidationError("One or more cache profiles are invalid.")
        for manifest_id, profile in self._profiles.items():
            if manifest_id != profile.profile_id:
                raise RegionCacheValidationError(
                    "Cache profile manifest key does not match its profile ID: "
                    f"{manifest_id!r} != {profile.profile_id!r}."
                )
        self._closed = False

    @property
    def profiles(self) -> Mapping[str, FlatmapRegionCacheProfile]:
        return dict(self._profiles)

    def profile(self, profile_id: str | None = None) -> FlatmapRegionCacheProfile:
        if profile_id is None:
            if len(self._profiles) != 1:
                raise RegionCacheValidationError(
                    "A profile ID is required when a cache contains multiple profiles."
                )
            return next(iter(self._profiles.values()))
        try:
            return self._profiles[str(profile_id)]
        except KeyError as exc:
            raise KeyError(f"Region-cache profile not found: {profile_id}") from exc

    def compatible_profiles(
        self, **requirements: Any
    ) -> tuple[FlatmapRegionCacheProfile, ...]:
        return tuple(
            profile
            for profile in self._profiles.values()
            if not profile.compatibility_mismatches(**requirements)
        )

    def close(self) -> None:
        """Release all memory-mapped arrays owned by this opened cache."""
        if self._closed:
            return
        for profile in self._profiles.values():
            profile.close()
        self._closed = True


def _close_style_mmaps(style: FlatmapRegionStyleCache) -> None:
    """Close arrays opened only for internal validation or failed publication."""
    for name, array in tuple(style._arrays.items()):
        try:
            _close_memmap(array)
        except Exception:
            logger.warning(
                "Failed to close cache array %s for style %s.",
                name,
                style.style,
                exc_info=True,
            )
    style._arrays.clear()


def _close_profile_mmaps(profile: FlatmapRegionCacheProfile) -> None:
    profile.close()


def _close_cache_mmaps(cache: FlatmapRegionCache) -> None:
    cache.close()


def open_region_cache(cache_dir: str | Path) -> FlatmapRegionCache:
    """Open and fully validate a flatmap region cache using memory maps."""
    root = Path(cache_dir)
    manifest_path = root / REGION_CACHE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Flatmap region-cache manifest not found: {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    _validate_root_manifest(manifest)
    cache = FlatmapRegionCache(root, manifest)
    try:
        for profile in cache.profiles.values():
            profile._validate()
    except BaseException:
        _close_cache_mmaps(cache)
        raise
    return cache


def _normalise_resolution(value: float | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError("Atlas resolution must be numeric.")
    if np.isscalar(value):
        values = (float(value),) * 3
    else:
        values = tuple(float(item) for item in value)
        if len(values) != 3:
            raise ValueError("Atlas resolution must be one value or three axis values.")
    if not all(np.isfinite(item) and item > 0 for item in values):
        raise ValueError("Atlas resolution values must be finite and positive.")
    return values


def _normalise_atlas_version(value: object | None) -> str | None:
    """Return one stable text form for BrainGlobe string/tuple versions."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        text = ".".join(str(part).strip() for part in value)
    else:
        text = str(value).strip()
    return text or None


def _lookup_value(lookup_set: object | None, *names: str) -> Any:
    if lookup_set is None:
        return None
    if isinstance(lookup_set, Mapping):
        for name in names:
            if name in lookup_set:
                return lookup_set[name]
        return None
    for name in names:
        if hasattr(lookup_set, name):
            return getattr(lookup_set, name)
    return None


def _object_value(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name)


def _lookup_grid_validity_value(value: object, name: str, default: bool) -> bool:
    if isinstance(value, Mapping):
        if name in value:
            return bool(value[name])
        validity = value.get("validity")
        if isinstance(validity, Mapping):
            return bool(validity.get(name, default))
        return bool(default)
    return bool(getattr(value, name))


def _array_digest(array: np.ndarray, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    contiguous = np.asarray(array)
    digest = hashlib.sha256()
    byte_view = contiguous.reshape(-1).view(np.uint8)
    for start in range(0, len(byte_view), chunk_bytes):
        digest.update(memoryview(byte_view[start : start + chunk_bytes]))
    return digest.hexdigest()


def _load_annotation(
    annotation: np.ndarray | str | Path | None, annotation_path: str | Path | None
) -> np.ndarray:
    source: np.ndarray | str | Path | None = annotation
    if source is None:
        source = annotation_path
    if source is None:
        raise ValueError("annotation or annotation_path is required.")
    if not isinstance(source, (str, Path)):
        result = np.asarray(source)
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Atlas annotation not found: {path}")
        if path.suffix.lower() == ".npy":
            result = np.load(path, mmap_mode="r", allow_pickle=False)
        else:
            try:
                import tifffile
            except ImportError as exc:  # pragma: no cover - transitive dependency
                raise RuntimeError(
                    "tifffile is required to memory-map BrainGlobe annotation.tiff."
                ) from exc
            try:
                result = tifffile.memmap(path)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"BrainGlobe annotation must be memory-mappable: {path}"
                ) from exc
    if result.ndim != 3:
        raise ValueError(f"Atlas annotation must be 3D; got {result.shape}.")
    if result.dtype.kind not in "iu":
        raise ValueError(
            f"Atlas annotation must contain integer region IDs; got {result.dtype}."
        )
    return result


def _resolve_lookup_arrays(
    lookup_set: object | None,
    *,
    shaped_flatmap: np.ndarray | None,
    square_flatmap: np.ndarray | None,
    depth: np.ndarray | None,
    shaped_flatmap_path: str | Path | None,
    square_flatmap_path: str | Path | None,
    depth_path: str | Path | None,
    npy_cache_dir: Path,
    cancel_callback: Callable[[], bool] | object | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shaped_flatmap = (
        shaped_flatmap
        if shaped_flatmap is not None
        else _lookup_value(lookup_set, "shaped_flatmap", "flatmap_shaped", "shaped")
    )
    square_flatmap = (
        square_flatmap
        if square_flatmap is not None
        else _lookup_value(lookup_set, "square_flatmap", "flatmap_square", "square")
    )
    depth = (
        depth
        if depth is not None
        else _lookup_value(lookup_set, "depth", "depth_volume")
    )
    shaped_flatmap_path = shaped_flatmap_path or _lookup_value(
        lookup_set, "shaped_path", "shaped_flatmap_path", "flatmap_shaped_path"
    )
    square_flatmap_path = square_flatmap_path or _lookup_value(
        lookup_set, "square_path", "square_flatmap_path", "flatmap_square_path"
    )
    depth_path = depth_path or _lookup_value(lookup_set, "depth_path")

    if shaped_flatmap is None:
        if shaped_flatmap_path is None or depth_path is None:
            raise ValueError(
                "A shaped flatmap array/path and depth array/path are required."
            )
        volume_set = load_flatmap_volume_set(
            shaped_flatmap_path,
            depth_path,
            npy_cache_dir=npy_cache_dir,
            mmap_npy=True,
            cancel_callback=lambda: _cancel_requested(cancel_callback),
        )
        shaped_flatmap = volume_set.flatmap
        if depth is None:
            depth = volume_set.depth
    if square_flatmap is None:
        if square_flatmap_path is None or depth_path is None:
            raise ValueError(
                "A square flatmap array/path and depth array/path are required."
            )
        volume_set = load_flatmap_volume_set(
            square_flatmap_path,
            depth_path,
            npy_cache_dir=npy_cache_dir,
            mmap_npy=True,
            cancel_callback=lambda: _cancel_requested(cancel_callback),
        )
        square_flatmap = volume_set.flatmap
        if depth is None:
            depth = volume_set.depth
    if depth is None:
        raise ValueError("A shared depth array/path is required.")

    shaped = np.asarray(shaped_flatmap)
    square = np.asarray(square_flatmap)
    depth_array = np.asarray(depth)
    for name, value in (("shaped", shaped), ("square", square)):
        if value.ndim != 4 or value.shape[-1] != 2:
            raise ValueError(
                f"{name} flatmap must have shape (X, Y, Z, 2); got {value.shape}."
            )
    if depth_array.ndim != 3:
        raise ValueError(f"Depth lookup must be 3D; got {depth_array.shape}.")
    if shaped.shape[:3] != square.shape[:3] or shaped.shape[:3] != depth_array.shape:
        raise ValueError(
            "Shaped flatmap, square flatmap, and depth must share the same spatial shape; "
            f"got {shaped.shape[:3]}, {square.shape[:3]}, and {depth_array.shape}."
        )
    return shaped, square, depth_array


def _derive_region_descendants(
    region_descendants: Mapping[int, Iterable[int]] | None,
    atlas_structures: Mapping[object, Mapping[str, Any]] | None,
    occupied_ids: Iterable[int],
) -> dict[int, tuple[int, ...]]:
    if region_descendants is not None:
        result = {
            int(parent): _normalise_region_ids((*descendants, int(parent)))
            for parent, descendants in region_descendants.items()
            if int(parent) > 0
        }
        return dict(sorted(result.items()))
    if atlas_structures:
        descendants: dict[int, set[int]] = {}
        for key, structure in atlas_structures.items():
            try:
                region_id = int(structure.get("id", key))
            except (TypeError, ValueError):
                continue
            if region_id <= 0:
                continue
            raw_path = structure.get("structure_id_path") or structure.get(
                "structure_id_path_ids"
            )
            if isinstance(raw_path, str):
                path_ids = [
                    int(part) for part in raw_path.strip("/").split("/") if part
                ]
            elif isinstance(raw_path, Iterable):
                path_ids = [int(part) for part in raw_path]
            else:
                path_ids = [region_id]
            for ancestor in (*path_ids, region_id):
                if ancestor > 0:
                    descendants.setdefault(ancestor, set()).add(region_id)
        return {
            parent: tuple(sorted(children | {parent}))
            for parent, children in sorted(descendants.items())
        }
    return {int(region_id): (int(region_id),) for region_id in sorted(occupied_ids)}


def _fill_mirrored_depth(
    depth_values: np.ndarray,
    depth: np.ndarray,
    missing: np.ndarray,
    *,
    chunk_slice: slice,
    mirror_coord_axis: int,
) -> int:
    local = np.argwhere(missing)
    if not len(local):
        return 0
    coordinates = local.copy()
    coordinates[:, 0] += int(chunk_slice.start or 0)
    coordinates[:, mirror_coord_axis] = (
        int(depth.shape[mirror_coord_axis]) - 1 - coordinates[:, mirror_coord_axis]
    )
    values = np.asarray(depth[tuple(coordinates.T)], dtype=float)
    valid = np.isfinite(values) & (values >= 0.0)
    if np.any(valid):
        rescued = local[valid]
        depth_values[tuple(rescued.T)] = values[valid]
    return int(np.count_nonzero(valid))


def _pair_run_paths(run_dir: Path, index: int) -> tuple[Path, Path]:
    return run_dir / f"keys-{index:06d}.npy", run_dir / f"counts-{index:06d}.npy"


def _write_occupancy_runs(
    *,
    annotation: np.ndarray,
    flatmap: np.ndarray,
    depth: np.ndarray,
    grid: Mapping[str, Any],
    run_dir: Path,
    chunk_voxels: int,
    invalid_zero_sentinel: bool,
    invalid_negative_one_sentinel: bool,
    mirror_depth_fallback: bool,
    mirror_coord_axis: int,
    cancel_callback: Callable[[], bool] | object | None,
    progress_callback: Callable[[str, int, int], None] | None,
    style: str,
) -> tuple[list[tuple[Path, Path]], dict[str, int]]:
    run_dir.mkdir(parents=True, exist_ok=False)
    slices = _spatial_chunk_slices(annotation.shape, chunk_voxels=chunk_voxels)
    run_paths: list[tuple[Path, Path]] = []
    valid_source_voxels = 0
    mirrored_depth_source_voxels = 0
    positive_annotation_voxels = 0
    y_bins = int(grid["y_bins"])
    x_bins = int(grid["x_bins"])
    depth_bins = int(grid["depth_bins"])
    x_bounds = tuple(float(value) for value in grid["x_bounds"])
    y_bounds = tuple(float(value) for value in grid["y_bounds"])
    depth_bounds = tuple(float(value) for value in grid["depth_bounds_um"])
    depth_bin_um = float(grid["depth_bin_um"])
    max_linear_bin = depth_bins * y_bins * x_bins
    if max_linear_bin > np.iinfo(np.uint32).max:
        raise ValueError(
            "Region-cache grid is too fine for its portable key format: "
            f"{depth_bins}x{y_bins}x{x_bins} is {max_linear_bin:,} bins, above "
            f"the {np.iinfo(np.uint32).max:,} limit. Reduce Y bins or use a "
            "larger depth bin."
        )

    for chunk_index, chunk_slice in enumerate(slices):
        _check_cancel(cancel_callback)
        annotation_chunk = np.asarray(annotation[chunk_slice])
        positive = annotation_chunk > 0
        positive_annotation_voxels += int(np.count_nonzero(positive))
        if not np.any(positive):
            _progress(
                progress_callback,
                f"Projecting {style} regions",
                chunk_index + 1,
                len(slices),
            )
            continue
        flat_xy = np.asarray(flatmap[chunk_slice], dtype=float)
        depth_values = np.array(depth[chunk_slice], dtype=float, copy=True)
        flat_valid = _flatmap_valid_mask(
            flat_xy.reshape(-1, 2),
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        ).reshape(annotation_chunk.shape)
        depth_valid = np.isfinite(depth_values) & (depth_values >= 0.0)
        if mirror_depth_fallback:
            mirrored_depth_source_voxels += _fill_mirrored_depth(
                depth_values,
                depth,
                positive & flat_valid & ~depth_valid,
                chunk_slice=chunk_slice,
                mirror_coord_axis=mirror_coord_axis,
            )
            depth_valid = np.isfinite(depth_values) & (depth_values >= 0.0)
        valid = positive & flat_valid & depth_valid
        if np.any(valid):
            region_ids = annotation_chunk[valid].astype(np.int64, copy=False)
            if np.any(region_ids > np.iinfo(np.int32).max):
                raise ValueError("Atlas region IDs must fit in positive int32 values.")
            x_bin_indices = _bin_flat_values(flat_xy[..., 0][valid], x_bounds, x_bins)
            y_bin_indices = _bin_flat_values(flat_xy[..., 1][valid], y_bounds, y_bins)
            depth_bin_indices = np.floor(
                (depth_values[valid] - depth_bounds[0]) / depth_bin_um
            ).astype(np.int64)
            depth_bin_indices = np.clip(depth_bin_indices, 0, depth_bins - 1)
            linear = (
                depth_bin_indices * y_bins * x_bins
                + y_bin_indices * x_bins
                + x_bin_indices
            )
            packed = (region_ids.astype(np.uint64) << np.uint64(32)) | linear.astype(
                np.uint64
            )
            keys, counts = np.unique(packed, return_counts=True)
            paths = _pair_run_paths(run_dir, len(run_paths))
            np.save(paths[0], keys, allow_pickle=False)
            np.save(paths[1], counts.astype(np.int64, copy=False), allow_pickle=False)
            run_paths.append(paths)
            valid_source_voxels += int(np.count_nonzero(valid))
        _progress(
            progress_callback,
            f"Projecting {style} regions",
            chunk_index + 1,
            len(slices),
        )
    return run_paths, {
        "positive_annotation_voxels": positive_annotation_voxels,
        "valid_source_voxels": valid_source_voxels,
        "mirrored_depth_source_voxels": mirrored_depth_source_voxels,
    }


def _merged_run_records(run_paths: Sequence[tuple[Path, Path]]):
    with ExitStack() as stack:
        arrays = [
            (
                stack.enter_context(_open_npy_memmap(keys_path)),
                stack.enter_context(_open_npy_memmap(counts_path)),
            )
            for keys_path, counts_path in run_paths
        ]
        heap: list[tuple[int, int, int]] = []
        for run_index, (keys, _counts) in enumerate(arrays):
            if len(keys):
                heapq.heappush(heap, (int(keys[0]), run_index, 0))
        while heap:
            key = heap[0][0]
            total = 0
            while heap and heap[0][0] == key:
                _same_key, run_index, position = heapq.heappop(heap)
                keys, counts = arrays[run_index]
                total += int(counts[position])
                position += 1
                if position < len(keys):
                    heapq.heappush(heap, (int(keys[position]), run_index, position))
            yield key, total


def _merge_occupancy_runs(
    run_paths: Sequence[tuple[Path, Path]],
    *,
    output_dir: Path,
    cancel_callback: Callable[[], bool] | object | None,
) -> tuple[dict[str, dict[str, Any]], np.ndarray, np.ndarray, Path, int]:
    record_count = 0
    with closing(_merged_run_records(run_paths)) as records:
        for record_count, _record in enumerate(records, start=1):
            if record_count % 100_000 == 0:
                _check_cancel(cancel_callback)

    bins_path = output_dir / "occupancy-linear-bins.npy"
    counts_path = output_dir / "occupancy-source-voxel-counts.npy"
    region_ids_list: list[int] = []
    offsets: list[int] = [0]
    previous_region: int | None = None
    source_voxel_count = 0
    with (
        _create_npy_memmap(
            bins_path,
            dtype=np.int64,
            shape=(record_count,),
        ) as bins,
        _create_npy_memmap(
            counts_path,
            dtype=np.int64,
            shape=(record_count,),
        ) as counts,
        closing(_merged_run_records(run_paths)) as records,
    ):
        for index, (key, count) in enumerate(records):
            if index % 100_000 == 0:
                _check_cancel(cancel_callback)
            region_id = key >> 32
            linear_bin = key & 0xFFFFFFFF
            if previous_region is None:
                region_ids_list.append(region_id)
                previous_region = region_id
            elif region_id != previous_region:
                offsets.append(index)
                region_ids_list.append(region_id)
                previous_region = region_id
            bins[index] = linear_bin
            counts[index] = count
            source_voxel_count += int(count)
    if previous_region is not None:
        offsets.append(record_count)
    region_ids_path = output_dir / "occupancy-region-ids.npy"
    offsets_path = output_dir / "occupancy-region-offsets.npy"
    region_ids = _save_array(region_ids_path, region_ids_list, dtype=np.int32)
    region_offsets = _save_array(offsets_path, offsets, dtype=np.int64)
    arrays = {
        "occupancy_region_ids": _array_spec(
            region_ids_path, region_ids, base=output_dir
        ),
        "occupancy_region_offsets": _array_spec(
            offsets_path, region_offsets, base=output_dir
        ),
        "occupancy_linear_bins": _array_spec_from_shape(
            bins_path,
            dtype=np.int64,
            shape=(record_count,),
            base=output_dir,
        ),
        "occupancy_source_voxel_counts": _array_spec_from_shape(
            counts_path,
            dtype=np.int64,
            shape=(record_count,),
            base=output_dir,
        ),
    }
    return arrays, region_ids, region_offsets, bins_path, source_voxel_count


def _neighbour_presence(
    bins: np.ndarray,
    neighbours: np.ndarray,
    in_bounds: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(bins), dtype=bool)
    if not np.any(in_bounds):
        return result
    candidate = neighbours[in_bounds]
    positions = np.searchsorted(bins, candidate)
    present = positions < len(bins)
    if np.any(present):
        matched = np.zeros_like(present)
        matched[present] = bins[positions[present]] == candidate[present]
        present = matched
    result[in_bounds] = present
    return result


_SURFACE_DIRECTIONS = (
    (
        -1,
        0,
        0,
        ((-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)),
    ),
    (1, 0, 0, ((0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5))),
    (
        0,
        -1,
        0,
        ((-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)),
    ),
    (0, 1, 0, ((-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5))),
    (
        0,
        0,
        -1,
        ((-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5)),
    ),
    (0, 0, 1, ((-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5))),
)


def _decoded_bins(
    bins: np.ndarray, shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_bins, y_bins, x_bins = shape
    del depth_bins
    plane = y_bins * x_bins
    z = bins // plane
    remainder = bins % plane
    y = remainder // x_bins
    x = remainder % x_bins
    return z, y, x


def _direction_neighbours(
    bins: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    shape: tuple[int, int, int],
    direction: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    dz, dy, dx = direction
    depth_bins, y_bins, x_bins = shape
    in_bounds = (
        (z + dz >= 0)
        & (z + dz < depth_bins)
        & (y + dy >= 0)
        & (y + dy < y_bins)
        & (x + dx >= 0)
        & (x + dx < x_bins)
    )
    delta = dz * y_bins * x_bins + dy * x_bins + dx
    return bins + delta, in_bounds


def _surface_for_bins(
    bins: np.ndarray, shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    bins = np.asarray(bins, dtype=np.int64)
    if not len(bins):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    z, y, x = _decoded_bins(bins, shape)
    centers = np.column_stack((z, y, x)).astype(np.float32, copy=False)
    exposed: list[tuple[np.ndarray, np.ndarray]] = []
    exposed_count = 0
    for dz, dy, dx, corner_values in _SURFACE_DIRECTIONS:
        neighbours, in_bounds = _direction_neighbours(
            bins, z, y, x, shape, (dz, dy, dx)
        )
        present = _neighbour_presence(bins, neighbours, in_bounds)
        indices = np.flatnonzero(~present)
        corners = np.asarray(corner_values, dtype=np.float32)
        exposed.append((indices, corners))
        exposed_count += len(indices)
    vertices = np.empty((exposed_count * 4, 3), dtype=np.float32)
    faces = np.empty((exposed_count * 2, 3), dtype=np.int32)
    face_cursor = 0
    for indices, corners in exposed:
        count = len(indices)
        if not count:
            continue
        vertex_start = face_cursor * 4
        vertices[vertex_start : vertex_start + count * 4] = (
            centers[indices, None, :] + corners[None, :, :]
        ).reshape(-1, 3)
        base = vertex_start + np.arange(count, dtype=np.int32) * 4
        faces[face_cursor * 2 : (face_cursor + count) * 2 : 2] = np.column_stack(
            (base, base + 1, base + 2)
        )
        faces[face_cursor * 2 + 1 : (face_cursor + count) * 2 : 2] = np.column_stack(
            (base, base + 2, base + 3)
        )
        face_cursor += count
    # Weld coincident face corners so the indexed triangles form a genuinely
    # closed shell rather than a visually closed collection of disjoint quads.
    welded_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    welded_faces = inverse[faces].astype(np.int32, copy=False)
    return welded_vertices.astype(np.float32, copy=False), welded_faces


def _outlines_for_bins(bins: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    bins = np.asarray(bins, dtype=np.int64)
    if not len(bins):
        return np.empty((0, 2, 3), dtype=np.float32)
    z, y, x = _decoded_bins(bins, shape)
    centers = np.column_stack((z, y, x)).astype(np.float32, copy=False)
    definitions = (
        ((0, -1, 0), np.array((0.0, -0.5, -0.5)), np.array((0.0, 0.0, 1.0))),
        ((0, 1, 0), np.array((0.0, 0.5, 0.5)), np.array((0.0, 0.0, -1.0))),
        ((0, 0, -1), np.array((0.0, 0.5, -0.5)), np.array((0.0, -1.0, 0.0))),
        ((0, 0, 1), np.array((0.0, -0.5, 0.5)), np.array((0.0, 1.0, 0.0))),
    )
    chunks: list[np.ndarray] = []
    for direction, start_offset, delta in definitions:
        neighbours, in_bounds = _direction_neighbours(bins, z, y, x, shape, direction)
        present = _neighbour_presence(bins, neighbours, in_bounds)
        indices = np.flatnonzero(~present)
        if not len(indices):
            continue
        vectors = np.empty((len(indices), 2, 3), dtype=np.float32)
        vectors[:, 0, :] = centers[indices] + start_offset.astype(np.float32)
        vectors[:, 1, :] = delta.astype(np.float32)
        chunks.append(vectors)
    if not chunks:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _component_count(
    bins: np.ndarray,
    shape: tuple[int, int, int],
    *,
    cancel_callback: Callable[[], bool] | object | None = None,
) -> int:
    remaining = {int(value) for value in bins}
    if not remaining:
        return 0
    depth_bins, y_bins, x_bins = shape
    plane = y_bins * x_bins
    components = 0
    visited = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            value = stack.pop()
            visited += 1
            if visited % 100_000 == 0:
                _check_cancel(cancel_callback)
            z = value // plane
            remainder = value % plane
            y = remainder // x_bins
            x = remainder % x_bins
            neighbours: list[int] = []
            if z > 0:
                neighbours.append(value - plane)
            if z + 1 < depth_bins:
                neighbours.append(value + plane)
            if y > 0:
                neighbours.append(value - x_bins)
            if y + 1 < y_bins:
                neighbours.append(value + x_bins)
            if x > 0:
                neighbours.append(value - 1)
            if x + 1 < x_bins:
                neighbours.append(value + 1)
            for neighbour in neighbours:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
    return components


def _region_bin_map(
    region_ids: np.ndarray, offsets: np.ndarray
) -> dict[int, tuple[int, int]]:
    return {
        int(region_id): (int(offsets[index]), int(offsets[index + 1]))
        for index, region_id in enumerate(region_ids)
    }


def _concatenate_npy_runs(
    paths: Sequence[Path],
    output_path: Path,
    *,
    dtype: np.dtype | str,
    tail_shape: tuple[int, ...],
    base: Path,
    cancel_callback: Callable[[], bool] | object | None = None,
) -> tuple[dict[str, Any], int]:
    lengths: list[int] = []
    for path in paths:
        with _open_npy_memmap(path) as source:
            lengths.append(int(source.shape[0]))
    total = sum(lengths)
    shape = (total, *tail_shape)
    with _create_npy_memmap(output_path, dtype=dtype, shape=shape) as output:
        cursor = 0
        for path, length in zip(paths, lengths, strict=True):
            with _open_npy_memmap(path) as source:
                for source_start in range(0, length, 1_000_000):
                    _check_cancel(cancel_callback)
                    source_stop = min(source_start + 1_000_000, length)
                    chunk_length = source_stop - source_start
                    output[cursor : cursor + chunk_length] = source[
                        source_start:source_stop
                    ]
                    cursor += chunk_length
    return (
        _array_spec_from_shape(
            output_path,
            dtype=dtype,
            shape=shape,
            base=base,
        ),
        total,
    )


def _build_geometry_runs(
    *,
    output_dir: Path,
    run_dir: Path,
    region_ids: np.ndarray,
    region_offsets: np.ndarray,
    linear_bins: np.ndarray,
    descendants: Mapping[int, Sequence[int]],
    output_shape: tuple[int, int, int],
    cancel_callback: Callable[[], bool] | object | None,
    progress_callback: Callable[[str, int, int], None] | None,
    style: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    bin_slices = _region_bin_map(region_ids, region_offsets)
    geometry_region_ids = np.asarray(sorted(descendants), dtype=np.int32)
    footprint_mapping = np.empty(len(geometry_region_ids), dtype=np.int32)
    hash_to_footprints: dict[str, list[int]] = {}
    vertex_paths: list[Path] = []
    face_paths: list[Path] = []
    outline_paths: list[Path] = []
    footprint_bin_paths: list[Path] = []
    component_counts: list[int] = []
    vertex_offsets = [0]
    face_offsets = [0]
    outline_offsets = [0]
    empty_regions = 0

    for region_index, parent_id in enumerate(geometry_region_ids.tolist()):
        _check_cancel(cancel_callback)
        chunks = []
        for descendant_id in descendants[int(parent_id)]:
            value = bin_slices.get(int(descendant_id))
            if value is not None:
                start, stop = value
                chunks.append(np.asarray(linear_bins[start:stop], dtype=np.int64))
        if chunks:
            footprint_bins = np.unique(np.concatenate(chunks))
        else:
            footprint_bins = np.empty(0, dtype=np.int64)
            empty_regions += 1
        digest = hashlib.sha256(
            footprint_bins.astype("<i8", copy=False).tobytes()
        ).hexdigest()
        footprint_index: int | None = None
        for candidate in hash_to_footprints.get(digest, []):
            with _open_npy_memmap(footprint_bin_paths[candidate]) as cached_bins:
                matches = np.array_equal(cached_bins, footprint_bins)
            if matches:
                footprint_index = candidate
                break
        if footprint_index is None:
            footprint_index = len(vertex_paths)
            hash_to_footprints.setdefault(digest, []).append(footprint_index)
            bin_path = run_dir / f"footprint-{footprint_index:06d}-bins.npy"
            vertex_path = run_dir / f"footprint-{footprint_index:06d}-vertices.npy"
            face_path = run_dir / f"footprint-{footprint_index:06d}-faces.npy"
            outline_path = run_dir / f"footprint-{footprint_index:06d}-outlines.npy"
            np.save(bin_path, footprint_bins, allow_pickle=False)
            vertices, faces = _surface_for_bins(footprint_bins, output_shape)
            outlines = _outlines_for_bins(footprint_bins, output_shape)
            np.save(vertex_path, vertices, allow_pickle=False)
            np.save(face_path, faces, allow_pickle=False)
            np.save(outline_path, outlines, allow_pickle=False)
            footprint_bin_paths.append(bin_path)
            vertex_paths.append(vertex_path)
            face_paths.append(face_path)
            outline_paths.append(outline_path)
            component_counts.append(
                _component_count(
                    footprint_bins,
                    output_shape,
                    cancel_callback=cancel_callback,
                )
            )
            vertex_offsets.append(vertex_offsets[-1] + len(vertices))
            face_offsets.append(face_offsets[-1] + len(faces))
            outline_offsets.append(outline_offsets[-1] + len(outlines))
        footprint_mapping[region_index] = footprint_index
        _progress(
            progress_callback,
            f"Building {style} region geometry",
            region_index + 1,
            len(geometry_region_ids),
        )

    geometry_region_ids_path = output_dir / "geometry-region-ids.npy"
    mapping_path = output_dir / "geometry-region-footprint-indices.npy"
    components_path = output_dir / "geometry-component-counts.npy"
    vertex_offsets_path = output_dir / "surface-vertex-offsets.npy"
    face_offsets_path = output_dir / "surface-face-offsets.npy"
    outline_offsets_path = output_dir / "outline-offsets.npy"
    vertices_path = output_dir / "surface-vertices.npy"
    faces_path = output_dir / "surface-faces.npy"
    outlines_path = output_dir / "outline-vectors.npy"
    saved_region_ids = _save_array(
        geometry_region_ids_path, geometry_region_ids, dtype=np.int32
    )
    saved_mapping = _save_array(mapping_path, footprint_mapping, dtype=np.int32)
    saved_components = _save_array(components_path, component_counts, dtype=np.int32)
    saved_vertex_offsets = _save_array(
        vertex_offsets_path, vertex_offsets, dtype=np.int64
    )
    saved_face_offsets = _save_array(face_offsets_path, face_offsets, dtype=np.int64)
    saved_outline_offsets = _save_array(
        outline_offsets_path, outline_offsets, dtype=np.int64
    )
    vertices_spec, vertex_count = _concatenate_npy_runs(
        vertex_paths,
        vertices_path,
        dtype=np.float32,
        tail_shape=(3,),
        base=output_dir,
        cancel_callback=cancel_callback,
    )
    faces_spec, face_count = _concatenate_npy_runs(
        face_paths,
        faces_path,
        dtype=np.int32,
        tail_shape=(3,),
        base=output_dir,
        cancel_callback=cancel_callback,
    )
    outlines_spec, outline_count = _concatenate_npy_runs(
        outline_paths,
        outlines_path,
        dtype=np.float32,
        tail_shape=(2, 3),
        base=output_dir,
        cancel_callback=cancel_callback,
    )
    arrays = {
        "geometry_region_ids": _array_spec(
            geometry_region_ids_path, saved_region_ids, base=output_dir
        ),
        "geometry_region_footprint_indices": _array_spec(
            mapping_path, saved_mapping, base=output_dir
        ),
        "geometry_component_counts": _array_spec(
            components_path, saved_components, base=output_dir
        ),
        "surface_vertex_offsets": _array_spec(
            vertex_offsets_path, saved_vertex_offsets, base=output_dir
        ),
        "surface_face_offsets": _array_spec(
            face_offsets_path, saved_face_offsets, base=output_dir
        ),
        "surface_vertices": vertices_spec,
        "surface_faces": faces_spec,
        "outline_offsets": _array_spec(
            outline_offsets_path, saved_outline_offsets, base=output_dir
        ),
        "outline_vectors": outlines_spec,
    }
    counts = {
        "direct_region_count": int(len(geometry_region_ids)),
        "empty_direct_region_count": int(empty_regions),
        "unique_footprint_count": int(len(vertex_paths)),
        "component_count": int(sum(component_counts)),
        "surface_vertex_count": int(vertex_count),
        "surface_face_count": int(face_count),
        "outline_segment_count": int(outline_count),
    }
    return arrays, counts


def _build_geometry(
    *,
    output_dir: Path,
    region_ids: np.ndarray,
    region_offsets: np.ndarray,
    linear_bins: np.ndarray,
    descendants: Mapping[int, Sequence[int]],
    output_shape: tuple[int, int, int],
    cancel_callback: Callable[[], bool] | object | None,
    progress_callback: Callable[[str, int, int], None] | None,
    style: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Build geometry and release all temporary mappings before cleanup."""
    run_dir = output_dir / ".geometry-runs"
    run_dir.mkdir()
    try:
        result = _build_geometry_runs(
            output_dir=output_dir,
            run_dir=run_dir,
            region_ids=region_ids,
            region_offsets=region_offsets,
            linear_bins=linear_bins,
            descendants=descendants,
            output_shape=output_shape,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
            style=style,
        )
    except BaseException:
        _remove_tree(
            run_dir,
            purpose=f"temporary {style} geometry runs",
            suppress_errors=True,
        )
        raise
    _remove_tree(run_dir, purpose=f"temporary {style} geometry runs")
    return result


def _style_grid(
    flatmap: np.ndarray,
    depth: np.ndarray,
    *,
    y_bins: int,
    depth_bin_um: float,
    bounds: Mapping[str, Sequence[float]] | None,
    invalid_zero_sentinel: bool,
    invalid_negative_one_sentinel: bool,
    chunk_voxels: int,
    lookup_grid_spec: object | None = None,
    cancel_callback: Callable[[], bool] | object | None = None,
) -> dict[str, Any]:
    stats = None
    if lookup_grid_spec is None:
        stats = compute_flatmap_lookup_stats(
            flatmap,
            depth,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
            chunk_voxels=chunk_voxels,
            cancel_callback=(
                None
                if cancel_callback is None
                else lambda: _check_cancel(cancel_callback) or False
            ),
        )
        default_x_bounds = stats.x_bounds
        default_y_bounds = stats.y_bounds
        default_depth_bounds = stats.depth_range_um
    else:
        default_x_bounds = _object_value(lookup_grid_spec, "x_bounds")
        default_y_bounds = _object_value(lookup_grid_spec, "y_bounds")
        default_depth_bounds = _object_value(lookup_grid_spec, "depth_bounds_um")
    x_bounds = tuple(
        float(value) for value in (bounds or {}).get("x_bounds", default_x_bounds)
    )
    y_bounds = tuple(
        float(value) for value in (bounds or {}).get("y_bounds", default_y_bounds)
    )
    depth_bounds = tuple(
        float(value)
        for value in (bounds or {}).get("depth_bounds_um", default_depth_bounds)
    )
    for name, values in (
        ("x_bounds", x_bounds),
        ("y_bounds", y_bounds),
        ("depth_bounds_um", depth_bounds),
    ):
        if (
            len(values) != 2
            or not np.all(np.isfinite(values))
            or values[1] <= values[0]
        ):
            raise ValueError(
                f"{name} must contain finite increasing lower/upper values."
            )
    depth_bins = _depth_bin_count(depth_bounds, depth_bin_um)
    # Derive here: this is where each style's own bounds are resolved, and the
    # two styles legitimately resolve to different x counts (491 vs 512 at 256).
    counts = resolve_flatmap_bin_counts(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        y_bins=y_bins,
    )
    grid: dict[str, Any] = {
        "coordinate_order": ["depth", "y", "x"],
        "y_bins": int(counts.y_bins),
        "x_bins": int(counts.x_bins),
        "depth_bins": int(depth_bins),
        "depth_bin_um": float(depth_bin_um),
        "x_bounds": list(x_bounds),
        "y_bounds": list(y_bounds),
        "depth_bounds_um": list(depth_bounds),
        "output_shape": [
            int(depth_bins),
            int(counts.y_bins),
            int(counts.x_bins),
        ],
        "includes_depth_minus_one_plane": False,
    }
    if stats is not None:
        grid["flatmap_valid_voxels"] = int(stats.flatmap_valid_voxels)
        grid["depth_valid_voxels"] = int(stats.depth_valid_voxels)
    if lookup_grid_spec is not None:
        to_dict = getattr(lookup_grid_spec, "to_dict", None)
        payload = to_dict() if callable(to_dict) else lookup_grid_spec
        if isinstance(payload, Mapping):
            grid["lookup_grid_spec"] = _json_safe(payload)
    return grid


def _build_style(
    *,
    profile_dir: Path,
    style: str,
    annotation: np.ndarray,
    flatmap: np.ndarray,
    depth: np.ndarray,
    grid: Mapping[str, Any],
    descendants: Mapping[int, Sequence[int]] | None,
    atlas_structures: Mapping[object, Mapping[str, Any]] | None,
    chunk_voxels: int,
    invalid_zero_sentinel: bool,
    invalid_negative_one_sentinel: bool,
    mirror_depth_fallback: bool,
    mirror_coord_axis: int,
    cancel_callback: Callable[[], bool] | object | None,
    progress_callback: Callable[[str, int, int], None] | None,
) -> dict[str, Any]:
    output_dir = profile_dir / style
    output_dir.mkdir()
    run_dir = output_dir / ".occupancy-runs"
    try:
        run_paths, scan_counts = _write_occupancy_runs(
            annotation=annotation,
            flatmap=flatmap,
            depth=depth,
            grid=grid,
            run_dir=run_dir,
            chunk_voxels=chunk_voxels,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
            mirror_depth_fallback=mirror_depth_fallback,
            mirror_coord_axis=mirror_coord_axis,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
            style=style,
        )
        (
            occupancy_arrays,
            region_ids,
            region_offsets,
            bins_path,
            source_voxel_count,
        ) = _merge_occupancy_runs(
            run_paths,
            output_dir=output_dir,
            cancel_callback=cancel_callback,
        )
    except BaseException:
        _remove_tree(
            run_dir,
            purpose=f"temporary {style} occupancy runs",
            suppress_errors=True,
        )
        raise
    _remove_tree(run_dir, purpose=f"temporary {style} occupancy runs")
    resolved_descendants = (
        dict(descendants)
        if descendants is not None
        else _derive_region_descendants(None, atlas_structures, region_ids.tolist())
    )
    with _open_npy_memmap(bins_path) as linear_bins:
        geometry_arrays, geometry_counts = _build_geometry(
            output_dir=output_dir,
            region_ids=region_ids,
            region_offsets=region_offsets,
            linear_bins=linear_bins,
            descendants=resolved_descendants,
            output_shape=tuple(int(size) for size in grid["output_shape"]),
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
            style=style,
        )
        occupied_bins = int(len(linear_bins))
    arrays = {**occupancy_arrays, **geometry_arrays}
    return {
        "grid": dict(grid),
        "arrays": arrays,
        "counts": {
            **scan_counts,
            **geometry_counts,
            "occupied_region_count": int(len(region_ids)),
            "region_bin_pair_count": occupied_bins,
            "source_voxel_count": int(source_voxel_count),
        },
    }


def build_region_cache_profile(
    cache_dir: str | Path,
    lookup_set: object | None = None,
    *,
    annotation: np.ndarray | str | Path | None = None,
    annotation_path: str | Path | None = None,
    shaped_flatmap: np.ndarray | None = None,
    square_flatmap: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    shaped_flatmap_path: str | Path | None = None,
    square_flatmap_path: str | Path | None = None,
    depth_path: str | Path | None = None,
    lookup_set_id: str | None = None,
    atlas_name: str,
    atlas_version: str | None = None,
    atlas_resolution_um: float | Sequence[float],
    atlas_structures: Mapping[object, Mapping[str, Any]] | None = None,
    atlas_structure_catalog_id: str | None = None,
    region_descendants: Mapping[int, Iterable[int]] | None = None,
    y_bins: int = DEFAULT_FLATMAP_Y_BINS,
    depth_bin_um: float = DEFAULT_FLATMAP_DEPTH_BIN_UM,
    bounds_by_style: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
    mirror_depth_fallback: bool = True,
    mirror_coord_axis: int = 2,
    chunk_voxels: int = DEFAULT_LOOKUP_STATS_CHUNK_VOXELS,
    cancel_callback: Callable[[], bool] | object | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    replace: bool = False,
) -> FlatmapRegionCacheProfile:
    """Build and atomically publish a shaped/square region-cache profile.

    ``annotation_path`` should normally name BrainGlobe's ``annotation.tiff``;
    it is memory-mapped.  Arrays are also accepted to make the computational
    API easy to test and reuse.  Existing profiles remain referenced until all
    new files have been published and the root manifest is atomically replaced.
    """
    root = Path(cache_dir)
    if int(y_bins) <= 0:
        raise ValueError("y_bins must be positive.")
    if not np.isfinite(depth_bin_um) or float(depth_bin_um) <= 0:
        raise ValueError("depth_bin_um must be finite and positive.")
    if int(chunk_voxels) <= 0:
        raise ValueError("chunk_voxels must be positive.")
    if mirror_coord_axis not in (0, 1, 2):
        raise ValueError("mirror_coord_axis must be 0, 1, or 2.")
    if not str(atlas_name).strip():
        raise ValueError("atlas_name is required.")
    atlas_resolution = _normalise_resolution(atlas_resolution_um)
    annotation_array = _load_annotation(annotation, annotation_path)
    root.mkdir(parents=True, exist_ok=True)
    profiles_root = root / "profiles"
    profiles_root.mkdir(exist_ok=True)
    lookup_cache_dir = root / "lookup-arrays"
    lookup_cache_dir.mkdir(exist_ok=True)
    try:
        shaped, square, depth_array = _resolve_lookup_arrays(
            lookup_set,
            shaped_flatmap=shaped_flatmap,
            square_flatmap=square_flatmap,
            depth=depth,
            shaped_flatmap_path=shaped_flatmap_path,
            square_flatmap_path=square_flatmap_path,
            depth_path=depth_path,
            npy_cache_dir=lookup_cache_dir,
            cancel_callback=cancel_callback,
        )
    except FlatmapLookupLoadCancelledError as exc:
        raise RegionCacheCancelled(
            "Flatmap region-cache construction was cancelled."
        ) from exc
    if tuple(annotation_array.shape) != tuple(depth_array.shape):
        raise ValueError(
            "Atlas annotation must exactly match the lookup spatial grid; "
            f"got annotation {annotation_array.shape} and lookups {depth_array.shape}."
        )
    lookup_resolution = _lookup_value(lookup_set, "lookup_resolution_um")
    if lookup_resolution is not None:
        normalised_lookup_resolution = _normalise_resolution(lookup_resolution)
        if not np.allclose(
            normalised_lookup_resolution,
            atlas_resolution,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Cache generation requires an exact atlas/lookup resolution match; "
                f"got atlas {atlas_resolution} um and lookup "
                f"{normalised_lookup_resolution} um."
            )
    lookup_spatial_shape = _lookup_value(lookup_set, "spatial_shape")
    if lookup_spatial_shape is not None and tuple(
        int(size) for size in lookup_spatial_shape
    ) != tuple(annotation_array.shape):
        raise ValueError(
            "Cache generation requires an exact atlas/lookup annotation shape match; "
            f"got atlas {annotation_array.shape} and lookup "
            f"{tuple(int(size) for size in lookup_spatial_shape)}."
        )
    _check_cancel(cancel_callback)

    inferred_lookup_id = _lookup_value(lookup_set, "lookup_set_id", "id")
    if lookup_set_id is None and inferred_lookup_id is not None:
        lookup_set_id = str(inferred_lookup_id)
    if lookup_set_id is None:
        lookup_set_id = _canonical_digest(
            {
                "shape": list(depth_array.shape),
                "shaped_sha256": _array_digest(shaped),
                "square_sha256": _array_digest(square),
                "depth_sha256": _array_digest(depth_array),
            }
        )

    style_bounds = {
        _normalise_style(name): value for name, value in (bounds_by_style or {}).items()
    }
    shaped_lookup_grid = _lookup_value(lookup_set, "shaped_grid")
    square_lookup_grid = _lookup_value(lookup_set, "square_grid")
    for lookup_grid in (shaped_lookup_grid, square_lookup_grid):
        if lookup_grid is None:
            continue
        lookup_zero_sentinel = _lookup_grid_validity_value(
            lookup_grid, "invalid_zero_sentinel", False
        )
        lookup_negative_one_sentinel = _lookup_grid_validity_value(
            lookup_grid, "invalid_negative_one_sentinel", True
        )
        if lookup_zero_sentinel != bool(invalid_zero_sentinel) or (
            lookup_negative_one_sentinel != bool(invalid_negative_one_sentinel)
        ):
            raise ValueError(
                "Region-cache validity policy must match the lookup set; "
                f"cache requested zero={bool(invalid_zero_sentinel)}, "
                f"negative_one={bool(invalid_negative_one_sentinel)} but lookup "
                f"uses zero={lookup_zero_sentinel}, "
                f"negative_one={lookup_negative_one_sentinel}."
            )
    if shaped_lookup_grid is not None and "shaped" not in style_bounds:
        style_bounds["shaped"] = {
            "x_bounds": _object_value(shaped_lookup_grid, "x_bounds"),
            "y_bounds": _object_value(shaped_lookup_grid, "y_bounds"),
            "depth_bounds_um": _object_value(shaped_lookup_grid, "depth_bounds_um"),
        }
    if square_lookup_grid is not None and "square" not in style_bounds:
        style_bounds["square"] = {
            "x_bounds": _object_value(square_lookup_grid, "x_bounds"),
            "y_bounds": _object_value(square_lookup_grid, "y_bounds"),
            "depth_bounds_um": _object_value(square_lookup_grid, "depth_bounds_um"),
        }
    shaped_grid = _style_grid(
        shaped,
        depth_array,
        y_bins=int(y_bins),
        depth_bin_um=float(depth_bin_um),
        bounds=style_bounds.get("shaped"),
        invalid_zero_sentinel=invalid_zero_sentinel,
        invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        chunk_voxels=int(chunk_voxels),
        lookup_grid_spec=shaped_lookup_grid,
        cancel_callback=cancel_callback,
    )
    square_grid = _style_grid(
        square,
        depth_array,
        y_bins=int(y_bins),
        depth_bin_um=float(depth_bin_um),
        bounds=style_bounds.get("square"),
        invalid_zero_sentinel=invalid_zero_sentinel,
        invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        chunk_voxels=int(chunk_voxels),
        lookup_grid_spec=square_lookup_grid,
        cancel_callback=cancel_callback,
    )
    if shaped_grid["depth_bounds_um"] != square_grid["depth_bounds_um"]:
        raise ValueError(
            "Shaped and square cache grids must use the same depth bounds."
        )

    resolved_catalog_id = atlas_structure_catalog_id
    if resolved_catalog_id is None and atlas_structures is not None:
        resolved_catalog_id = structure_catalog_id(atlas_structures)
    validity = {
        "invalid_zero_sentinel": bool(invalid_zero_sentinel),
        "invalid_negative_one_sentinel": bool(invalid_negative_one_sentinel),
        "mirror_depth_fallback": bool(mirror_depth_fallback),
        "mirror_coord_axis": int(mirror_coord_axis),
        "excludes_depth_minus_one_plane": True,
    }
    canonical_profile = {
        "lookup_set_id": str(lookup_set_id),
        "atlas": {
            "name": str(atlas_name),
            "family": normalise_atlas_family(atlas_name),
            "version": _normalise_atlas_version(atlas_version),
            "resolution_um": list(atlas_resolution),
            "annotation_shape": [int(size) for size in annotation_array.shape],
            "structure_catalog_id": resolved_catalog_id,
        },
        "y_bins": int(y_bins),
        "depth_bin_um": float(depth_bin_um),
        "style_grids": {"shaped": shaped_grid, "square": square_grid},
        "algorithms": {
            "occupancy": _OCCUPANCY_ALGORITHM,
            "collision": _COLLISION_ALGORITHM,
            "surface": _SURFACE_ALGORITHM,
            "outline": _OUTLINE_ALGORITHM,
            "depth": _DEPTH_ALGORITHM,
        },
        "validity": validity,
    }
    profile_id = _canonical_digest(canonical_profile)
    manifest_path = root / REGION_CACHE_MANIFEST_FILENAME
    if manifest_path.exists():
        existing_manifest = _read_json(manifest_path)
        _validate_root_manifest(existing_manifest)
    else:
        existing_manifest = _empty_manifest()
    if profile_id in existing_manifest["profiles"] and not replace:
        raise FileExistsError(
            f"Flatmap region-cache profile already exists: {profile_id}"
        )

    temporary_dir = profiles_root / f".{profile_id[:16]}.{uuid.uuid4().hex}.tmp"
    published_dir: Path | None = None
    try:
        temporary_dir.mkdir()
        direct_descendants = (
            _derive_region_descendants(region_descendants, atlas_structures, [])
            if region_descendants is not None or atlas_structures
            else None
        )
        shaped_data = _build_style(
            profile_dir=temporary_dir,
            style="shaped",
            annotation=annotation_array,
            flatmap=shaped,
            depth=depth_array,
            grid=shaped_grid,
            descendants=direct_descendants,
            atlas_structures=atlas_structures,
            chunk_voxels=int(chunk_voxels),
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
            mirror_depth_fallback=mirror_depth_fallback,
            mirror_coord_axis=mirror_coord_axis,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
        )
        _check_cancel(cancel_callback)
        square_data = _build_style(
            profile_dir=temporary_dir,
            style="square",
            annotation=annotation_array,
            flatmap=square,
            depth=depth_array,
            grid=square_grid,
            descendants=direct_descendants,
            atlas_structures=atlas_structures,
            chunk_voxels=int(chunk_voxels),
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
            mirror_depth_fallback=mirror_depth_fallback,
            mirror_coord_axis=mirror_coord_axis,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
        )
        _check_cancel(cancel_callback)
        published_dir = profiles_root / f"{profile_id[:16]}-{uuid.uuid4().hex[:12]}"
        os.replace(temporary_dir, published_dir)
        _check_cancel(cancel_callback)
        relative_directory = published_dir.relative_to(root).as_posix()
        profile_data = {
            "profile_id": profile_id,
            "directory": relative_directory,
            "lookup_set_id": str(lookup_set_id),
            "atlas": canonical_profile["atlas"],
            "y_bins": int(y_bins),
            "depth_bin_um": float(depth_bin_um),
            "algorithms": canonical_profile["algorithms"],
            "validity": canonical_profile["validity"],
            "styles": {"shaped": shaped_data, "square": square_data},
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        # Validate every published mmap before it can become reachable from the
        # root manifest. A generated/truncated array must not poison otherwise
        # usable profiles.
        validation_profile = FlatmapRegionCacheProfile(root, profile_data)
        try:
            validation_profile._validate()
        finally:
            _close_profile_mmaps(validation_profile)

        # Re-read immediately before publication so profiles added by another
        # builder during the expensive projection are retained.
        if manifest_path.exists():
            final_manifest = _read_json(manifest_path)
            _validate_root_manifest(final_manifest)
        else:
            final_manifest = _empty_manifest()
        if profile_id in final_manifest["profiles"] and not replace:
            raise FileExistsError(
                f"Flatmap region-cache profile already exists: {profile_id}"
            )
        previous_manifest = final_manifest
        updated_manifest = dict(final_manifest)
        updated_manifest["profiles"] = dict(final_manifest["profiles"])
        old_profile = updated_manifest["profiles"].get(profile_id)
        updated_manifest["profiles"][profile_id] = profile_data
        _atomic_write_json(manifest_path, updated_manifest)
        try:
            opened_profile = open_region_cache(root).profile(profile_id)
        except BaseException:
            _atomic_write_json(manifest_path, previous_manifest)
            raise
        # A replaced physical directory is intentionally left in place until
        # after the new manifest is durable.  Remove only the now-unreferenced
        # profile directory and only when it is inside this cache root.
        if replace and isinstance(old_profile, Mapping):
            old_directory = old_profile.get("directory")
            if isinstance(old_directory, str) and old_directory != relative_directory:
                old_path = _safe_array_path(root, old_directory)
                if old_path.is_dir():
                    _remove_tree(
                        old_path,
                        purpose="unreferenced replaced cache profile",
                        suppress_errors=True,
                    )
        return opened_profile
    except BaseException:
        _remove_tree(
            temporary_dir,
            purpose="failed temporary cache profile",
            suppress_errors=True,
        )
        if published_dir is not None and published_dir.is_dir():
            # If publication succeeded, the manifest now references this
            # directory and it must remain usable.  Otherwise it is an orphan.
            try:
                current = (
                    _read_json(manifest_path)
                    if manifest_path.exists()
                    else _empty_manifest()
                )
                referenced = any(
                    isinstance(item, Mapping)
                    and item.get("directory")
                    == published_dir.relative_to(root).as_posix()
                    for item in current.get("profiles", {}).values()
                )
            except RegionCacheError:
                referenced = False
            if not referenced:
                _remove_tree(
                    published_dir,
                    purpose="unreferenced failed cache profile",
                    suppress_errors=True,
                )
        raise


def _profile_from_value(
    cache_or_profile: FlatmapRegionCache | FlatmapRegionCacheProfile | str | Path,
    profile_id: str | None,
) -> FlatmapRegionCacheProfile:
    if isinstance(cache_or_profile, FlatmapRegionCacheProfile):
        if profile_id is not None and profile_id != cache_or_profile.profile_id:
            raise ValueError("profile_id does not match the provided cache profile.")
        return cache_or_profile
    if isinstance(cache_or_profile, FlatmapRegionCache):
        return cache_or_profile.profile(profile_id)
    return open_region_cache(cache_or_profile).profile(profile_id)


def _collapsed_occupancy(
    style_cache: FlatmapRegionStyleCache,
    region_id: int,
    plane_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return one region's sorted planar bins and depth-summed source counts.

    The cached linear bins index a ``(depth, y, x)`` grid, so the depth plane
    drops out with one modulo.  Counts for the same ``(y, x)`` column are summed
    rather than compared, which is what makes a collapsed plane a footprint
    instead of a contest between a region's own depth bins.
    """
    occupancy = style_cache.occupancy_slice(region_id)
    if occupancy is None:
        return None
    depth_linear_bins, source_counts = occupancy
    if not len(depth_linear_bins):
        return None
    planar_bins = np.asarray(depth_linear_bins, dtype=np.int64) % int(plane_size)
    counts = np.asarray(source_counts, dtype=np.int64)
    unique_planar_bins, inverse = np.unique(planar_bins, return_inverse=True)
    collapsed_counts = np.zeros(len(unique_planar_bins), dtype=np.int64)
    np.add.at(collapsed_counts, inverse, counts)
    return unique_planar_bins, collapsed_counts


def _resolve_flat_region_groups(
    style_cache: FlatmapRegionStyleCache,
    *,
    member_region_ids: Sequence[int],
    direct_region_ids: Sequence[int],
    region_descendants: Mapping[int, Iterable[int]] | None = None,
    atlas_structures: Mapping[object, Mapping[str, Any]] | None = None,
) -> tuple[dict[int, tuple[int, ...]], str]:
    """Map each selected geometry root to the occupancy regions it aggregates.

    Labels retain source region IDs so descendant appearance overrides remain
    addressable after depth is collapsed.  These groups are still required for
    direct-selection summaries and for union outlines, which follow the selected
    parent footprint rather than the winning source label in each planar bin.
    """
    occupied = {
        int(value) for value in style_cache.array("occupancy_region_ids").tolist()
    }
    members = tuple(
        int(region_id) for region_id in member_region_ids if int(region_id) in occupied
    )
    roots = tuple(int(region_id) for region_id in direct_region_ids)
    if not roots:
        return {region_id: (region_id,) for region_id in members}, "source_region"
    if len(roots) == 1:
        # Every include-child-expanded member belongs to the sole selected root
        # by construction, so no hierarchy is needed for the common case.
        return {roots[0]: members}, "selected_root"
    root_set = set(roots)
    if all(member in root_set for member in members):
        # Terminal selections (and the identity default) already name their own
        # label values, so there is nothing left for a hierarchy to decide.
        return {member: (member,) for member in members}, "selected_root"
    if region_descendants is None and not atlas_structures:
        raise ValueError(
            "Collapsing more than one selected region into flatmap space needs "
            "region_descendants or atlas_structures to decide which cached "
            "region belongs to which selection."
        )
    descendants = _derive_region_descendants(
        region_descendants,
        atlas_structures,
        members,
    )
    # Deepest root first so a nested selection claims its own members before an
    # ancestor can absorb them.
    ordered_roots = sorted(
        roots,
        key=lambda root: (len(descendants.get(root, (root,))), root),
    )
    groups: dict[int, list[int]] = {root: [] for root in roots}
    for member in members:
        for root in ordered_roots:
            if member in descendants.get(root, (root,)):
                groups[root].append(member)
                break
    return {
        root: tuple(members_for_root)
        for root, members_for_root in groups.items()
        if members_for_root
    }, "selected_root"


def materialize_region_selection(
    cache_or_profile: FlatmapRegionCache | FlatmapRegionCacheProfile | str | Path,
    region_ids: Iterable[int],
    *,
    style: str,
    profile_id: str | None = None,
    direct_region_ids: Iterable[int] | None = None,
    include_surfaces: bool = True,
    include_outlines: bool = True,
) -> CachedRegionSelection:
    """Materialise majority labels and parent-union geometry from cached arrays.

    ``region_ids`` should be the annotation IDs used for labels. Atlas-region
    selections normally provide their Include-child-expanded IDs, while custom
    selections provide exact terminal IDs. ``direct_region_ids`` contains the
    geometry roots selected in the UI; each root may be an atlas parent with a
    precomputed descendant union or an exact terminal region.
    """
    profile = _profile_from_value(cache_or_profile, profile_id)
    style_cache = profile.style(style)
    selected = _normalise_region_ids(region_ids)
    direct = (
        selected
        if direct_region_ids is None
        else _normalise_region_ids(direct_region_ids)
    )
    output_shape = style_cache.output_shape
    labels = np.zeros(output_shape, dtype=np.int32)
    pair_bins: list[np.ndarray] = []
    pair_counts: list[np.ndarray] = []
    pair_ids: list[np.ndarray] = []
    source_voxel_count = 0
    represented: list[int] = []
    for region_id in selected:
        occupancy = style_cache.occupancy_slice(region_id)
        if occupancy is None:
            continue
        bins, counts = occupancy
        if not len(bins):
            continue
        pair_bins.append(np.asarray(bins, dtype=np.int64))
        pair_counts.append(np.asarray(counts, dtype=np.int64))
        pair_ids.append(np.full(len(bins), region_id, dtype=np.int32))
        source_voxel_count += int(np.asarray(counts, dtype=np.int64).sum())
        represented.append(region_id)
    collision_bins = 0
    if pair_bins:
        bins = np.concatenate(pair_bins)
        counts = np.concatenate(pair_counts)
        ids = np.concatenate(pair_ids)
        order = np.lexsort((ids, -counts, bins))
        sorted_bins = bins[order]
        sorted_ids = ids[order]
        _unique, first, competing = np.unique(
            sorted_bins, return_index=True, return_counts=True
        )
        labels.reshape(-1)[sorted_bins[first]] = sorted_ids[first]
        collision_bins = int(np.count_nonzero(competing > 1))

    surfaces: list[CachedRegionSurface] = []
    outlines: list[CachedRegionOutlines] = []
    for region_id in direct:
        if include_surfaces:
            surface = style_cache.surface(region_id)
            if surface is not None:
                surfaces.append(surface)
        if include_outlines:
            outline = style_cache.outlines(region_id)
            if outline is not None:
                outlines.append(outline)
    summary = CachedRegionSelectionSummary(
        selected_region_count=len(selected),
        represented_region_count=len(represented),
        labeled_bins=int(np.count_nonzero(labels)),
        collision_bins=collision_bins,
        source_voxel_count=source_voxel_count,
        output_shape=output_shape,
    )
    return CachedRegionSelection(
        labels=labels,
        selected_region_ids=selected,
        represented_region_ids=tuple(represented),
        surfaces=tuple(surfaces),
        outlines=tuple(outlines),
        summary=summary,
        grid_spec=style_cache.grid_spec,
        style=style_cache.style,
        profile_id=profile.profile_id,
    )


def materialize_allen_layer_region_selection(
    cache_or_profile: FlatmapRegionCache | FlatmapRegionCacheProfile | str | Path,
    region_ids: Iterable[int],
    *,
    style: str,
    layer_map: AllenIsocortexLayerMap,
    profile_id: str | None = None,
    selected_layer_indices: Sequence[int] | None = None,
    output_mode: str = ALLEN_LAYER_OUTPUT_STACK,
) -> CachedAllenLayerRegionSelection:
    """Collapse cached occupancy into a selected Allen stack or projection.

    Source-voxel counts are first summed across depth for each region and XY
    bin. Competing regions in one Allen plane use the cache's normal majority
    rule, with the smaller region ID winning equal-count ties.
    """
    profile = _profile_from_value(cache_or_profile, profile_id)
    style_cache = profile.style(style)
    selected = _normalise_region_ids(region_ids)
    selected_indices, layer_labels, output_mode = (
        _resolve_allen_layer_render_options(
            layer_map,
            selected_layer_indices,
            output_mode,
        )
    )
    render_index_by_layer = {
        canonical_index: render_index
        for render_index, canonical_index in enumerate(selected_indices)
    }

    depth_shape = style_cache.output_shape
    y_bins, x_bins = int(depth_shape[1]), int(depth_shape[2])
    plane_size = y_bins * x_bins
    output_shape = (
        (len(selected_indices), y_bins, x_bins)
        if output_mode == ALLEN_LAYER_OUTPUT_STACK
        else (y_bins, x_bins)
    )
    labels = np.zeros(output_shape, dtype=np.int32)

    mapped: list[int] = []
    represented: list[int] = []
    pair_bins: list[np.ndarray] = []
    pair_counts: list[np.ndarray] = []
    pair_ids: list[np.ndarray] = []
    source_voxel_count = 0
    excluded_unselected_layer_region_count = 0
    for region_id in selected:
        raw_layer_index = layer_map.region_to_layer_index.get(region_id)
        if raw_layer_index is None:
            continue
        layer_index = int(raw_layer_index)
        if layer_index < 0 or layer_index >= len(layer_map.layer_labels):
            raise ValueError(
                f"Region ID {region_id} maps to invalid Allen layer index "
                f"{layer_index}; expected 0 through {len(layer_map.layer_labels) - 1}."
            )
        if layer_index not in render_index_by_layer:
            excluded_unselected_layer_region_count += 1
            continue
        mapped.append(region_id)
        collapsed = _collapsed_occupancy(style_cache, region_id, plane_size)
        if collapsed is None:
            continue
        unique_planar_bins, collapsed_counts = collapsed
        if output_mode == ALLEN_LAYER_OUTPUT_STACK:
            output_bins = (
                render_index_by_layer[layer_index] * plane_size + unique_planar_bins
            )
        else:
            output_bins = unique_planar_bins

        pair_bins.append(output_bins)
        pair_counts.append(collapsed_counts)
        pair_ids.append(np.full(len(unique_planar_bins), region_id, dtype=np.int32))
        source_voxel_count += int(collapsed_counts.sum())
        represented.append(region_id)

    collision_bins = 0
    if pair_bins:
        bins = np.concatenate(pair_bins)
        counts = np.concatenate(pair_counts)
        ids = np.concatenate(pair_ids)
        order = np.lexsort((ids, -counts, bins))
        sorted_bins = bins[order]
        sorted_ids = ids[order]
        _unique, first, competing = np.unique(
            sorted_bins,
            return_index=True,
            return_counts=True,
        )
        labels.reshape(-1)[sorted_bins[first]] = sorted_ids[first]
        collision_bins = int(np.count_nonzero(competing > 1))

    summary = CachedAllenLayerRegionSelectionSummary(
        selected_region_count=len(selected),
        layer_mapped_region_count=len(mapped),
        represented_region_count=len(represented),
        labeled_bins=int(np.count_nonzero(labels)),
        collision_bins=collision_bins,
        source_voxel_count=source_voxel_count,
        output_shape=output_shape,
        layer_labels=layer_labels,
        selected_layer_indices=selected_indices,
        output_mode=output_mode,
        excluded_unselected_layer_region_count=(
            excluded_unselected_layer_region_count
        ),
    )
    grid_spec = {
        "coordinate_order": (
            ["allen_layer", "y", "x"]
            if output_mode == ALLEN_LAYER_OUTPUT_STACK
            else ["y", "x"]
        ),
        "plane_mode": (
            "allen_layers"
            if output_mode == ALLEN_LAYER_OUTPUT_STACK
            else "allen_layer_projection"
        ),
        "allen_layer_output_mode": output_mode,
        "layer_labels": list(layer_labels),
        "selected_layer_indices": [int(value) for value in selected_indices],
        "y_bins": int(style_cache.grid_spec["y_bins"]),
        "x_bins": int(style_cache.grid_spec["x_bins"]),
        "x_bounds": list(style_cache.grid_spec["x_bounds"]),
        "y_bounds": list(style_cache.grid_spec["y_bounds"]),
        "output_shape": [int(size) for size in output_shape],
    }
    return CachedAllenLayerRegionSelection(
        labels=labels,
        selected_region_ids=selected,
        layer_mapped_region_ids=tuple(mapped),
        represented_region_ids=tuple(represented),
        layer_labels=layer_labels,
        summary=summary,
        grid_spec=grid_spec,
        style=style_cache.style,
        profile_id=profile.profile_id,
    )


def materialize_flat_region_selection(
    cache_or_profile: FlatmapRegionCache | FlatmapRegionCacheProfile | str | Path,
    region_ids: Iterable[int],
    *,
    style: str,
    profile_id: str | None = None,
    direct_region_ids: Iterable[int] | None = None,
    region_descendants: Mapping[int, Iterable[int]] | None = None,
    atlas_structures: Mapping[object, Mapping[str, Any]] | None = None,
    include_outlines: bool = True,
) -> CachedFlatRegionSelection:
    """Collapse cached depth occupancy into one depth-free flatmap plane.

    ``region_ids`` are the cached occupancy regions to read, normally the
    include-child-expanded selection.  ``direct_region_ids`` are the regions the
    user actually selected and define the union footprints used for outlines.
    The 2D label values remain the source occupancy region IDs so the displayed
    colormap can resolve parent inheritance and descendant appearance overrides.
    When multiple source regions occupy the same flatmap column, their counts
    are summed through depth and the cache's majority rule selects the visible
    source ID; the smaller ID wins an equal-count tie.

    Outlines are derived here rather than read from the cache: the stored
    outlines are per-depth-slice perimeters, whose depth-collapsed projection is
    a strict subset of the true collapsed footprint.
    """
    profile = _profile_from_value(cache_or_profile, profile_id)
    style_cache = profile.style(style)
    selected = _normalise_region_ids(region_ids)
    direct = (
        selected
        if direct_region_ids is None
        else _normalise_region_ids(direct_region_ids)
    )
    depth_shape = style_cache.output_shape
    y_bins, x_bins = int(depth_shape[1]), int(depth_shape[2])
    plane_size = y_bins * x_bins
    output_shape = (y_bins, x_bins)
    labels = np.zeros(output_shape, dtype=np.int32)

    groups, grouping = _resolve_flat_region_groups(
        style_cache,
        member_region_ids=selected,
        direct_region_ids=direct,
        region_descendants=region_descendants,
        atlas_structures=atlas_structures,
    )

    represented: list[int] = []
    represented_sources: set[int] = set()
    source_voxel_count = 0
    group_bins: dict[int, np.ndarray] = {}
    group_source_counts: dict[int, int] = {}
    group_members: dict[int, tuple[int, ...]] = {}
    pair_bins: list[np.ndarray] = []
    pair_counts: list[np.ndarray] = []
    pair_ids: list[np.ndarray] = []
    for label_id in direct:
        members = groups.get(label_id, ())
        group_members[label_id] = members
        member_bins: list[np.ndarray] = []
        member_counts: list[np.ndarray] = []
        contributing: list[int] = []
        for member in members:
            collapsed = _collapsed_occupancy(style_cache, member, plane_size)
            if collapsed is None:
                continue
            member_bins.append(collapsed[0])
            member_counts.append(collapsed[1])
            contributing.append(member)
            pair_bins.append(collapsed[0])
            pair_counts.append(collapsed[1])
            pair_ids.append(np.full(len(collapsed[0]), member, dtype=np.int32))
        if not member_bins:
            continue
        # Build the selected root's union footprint for its outline; source IDs
        # still compete independently in the Labels layer above.
        merged_bins, inverse = np.unique(
            np.concatenate(member_bins), return_inverse=True
        )
        merged_counts = np.zeros(len(merged_bins), dtype=np.int64)
        np.add.at(merged_counts, inverse, np.concatenate(member_counts))
        group_bins[label_id] = merged_bins
        group_source_counts[label_id] = int(merged_counts.sum())
        group_members[label_id] = tuple(contributing)
        represented.append(label_id)
        represented_sources.update(contributing)
        source_voxel_count += int(merged_counts.sum())

    collision_bins = 0
    if pair_bins:
        bins = np.concatenate(pair_bins)
        counts = np.concatenate(pair_counts)
        ids = np.concatenate(pair_ids)
        order = np.lexsort((ids, -counts, bins))
        sorted_bins = bins[order]
        sorted_ids = ids[order]
        _unique, first, competing = np.unique(
            sorted_bins,
            return_index=True,
            return_counts=True,
        )
        labels.reshape(-1)[sorted_bins[first]] = sorted_ids[first]
        collision_bins = int(np.count_nonzero(competing > 1))

    outlines: list[CachedFlatRegionOutlines] = []
    if include_outlines:
        for label_id in direct:
            merged_bins = group_bins.get(label_id)
            if merged_bins is None or not len(merged_bins):
                continue
            # The stored tracer already emits only in-plane edges, so a single
            # depth plane reduces it to a 2D perimeter.
            vectors = _outlines_for_bins(merged_bins, (1, y_bins, x_bins))[:, :, 1:]
            outlines.append(
                CachedFlatRegionOutlines(
                    region_id=int(label_id),
                    vectors=np.ascontiguousarray(vectors, dtype=np.float32),
                    union_region_ids=groups.get(label_id, ()),
                    represented_region_ids=group_members.get(label_id, ()),
                    planar_bin_count=int(len(merged_bins)),
                    source_voxel_count=int(group_source_counts.get(label_id, 0)),
                )
            )

    summary = CachedFlatRegionSelectionSummary(
        selected_region_count=len(selected),
        direct_region_count=len(direct),
        represented_region_count=len(represented),
        represented_source_region_count=len(represented_sources),
        labeled_bins=int(np.count_nonzero(labels)),
        collision_bins=collision_bins,
        source_voxel_count=source_voxel_count,
        output_shape=output_shape,
    )
    grid_spec = {
        "coordinate_order": ["y", "x"],
        "plane_mode": "flat",
        "y_bins": int(style_cache.grid_spec["y_bins"]),
        "x_bins": int(style_cache.grid_spec["x_bins"]),
        "x_bounds": list(style_cache.grid_spec["x_bounds"]),
        "y_bounds": list(style_cache.grid_spec["y_bounds"]),
        "output_shape": [int(size) for size in output_shape],
        "label_grouping": "source_region",
        "geometry_grouping": grouping,
    }
    return CachedFlatRegionSelection(
        labels=labels,
        selected_region_ids=selected,
        direct_region_ids=direct,
        represented_region_ids=tuple(represented),
        represented_source_region_ids=tuple(sorted(represented_sources)),
        outlines=tuple(outlines),
        summary=summary,
        grid_spec=grid_spec,
        style=style_cache.style,
        profile_id=profile.profile_id,
    )


def materialize_flat_region_outlines(
    cache_or_profile: FlatmapRegionCache | FlatmapRegionCacheProfile | str | Path,
    region_id: int,
    *,
    style: str,
    profile_id: str | None = None,
    region_ids: Iterable[int] | None = None,
    region_descendants: Mapping[int, Iterable[int]] | None = None,
    atlas_structures: Mapping[object, Mapping[str, Any]] | None = None,
) -> CachedFlatRegionOutlines | None:
    """Return one region's depth-collapsed 2D perimeter, or ``None`` when empty.

    ``region_ids`` are the occupancy regions to union; they default to the
    region itself plus any descendants derivable from the supplied hierarchy.
    Callers showing several regions should use
    :func:`materialize_flat_region_selection` instead, which resolves the
    hierarchy once for the whole selection.
    """
    if region_ids is None:
        descendants = _derive_region_descendants(
            region_descendants,
            atlas_structures,
            (int(region_id),),
        )
        region_ids = descendants.get(int(region_id), (int(region_id),))
    result = materialize_flat_region_selection(
        cache_or_profile,
        region_ids,
        style=style,
        profile_id=profile_id,
        direct_region_ids=(int(region_id),),
        region_descendants=region_descendants,
        atlas_structures=atlas_structures,
        include_outlines=True,
    )
    return result.outlines[0] if result.outlines else None


def materialize_region_surface(
    cache_or_profile: FlatmapRegionCache | FlatmapRegionCacheProfile | str | Path,
    region_id: int,
    *,
    style: str,
    profile_id: str | None = None,
) -> CachedRegionSurface | None:
    """Return one parent's cached descendant-union surface."""
    return (
        _profile_from_value(cache_or_profile, profile_id)
        .style(style)
        .surface(region_id)
    )


def materialize_region_outlines(
    cache_or_profile: FlatmapRegionCache | FlatmapRegionCacheProfile | str | Path,
    region_id: int,
    *,
    style: str,
    profile_id: str | None = None,
) -> CachedRegionOutlines | None:
    """Return one parent's cached descendant-union slice outlines."""
    return (
        _profile_from_value(cache_or_profile, profile_id)
        .style(style)
        .outlines(region_id)
    )
